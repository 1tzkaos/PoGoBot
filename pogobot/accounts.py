"""The PGSharp account list, read from the Android view tree.

A second perception channel, deliberately narrow. Everything else the bot knows comes
from pixels; this reads real Android views, which is possible only because the PGSharp
overlay is drawn as views rather than into the Unity canvas. It cannot read the game:
uiautomator sees Pokemon GO itself as one opaque `View "Game view"`.

Why it is worth a second channel at all: the account list states which account is logged
in, as text, with an asterisk. That is ground truth. Every alternative - OCR of the map's
bottom-left name, or a classifier - would be an inference about something the system can
simply be told.

Safety note that drives the whole module: each row's delete button sits ~24px from its
login button. Every coordinate here is some node's OWN bounds. Nothing is a constant, an
offset, or a guess, because the failure mode is an irreversibly deleted account.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

from .effects import Tap

log = logging.getLogger("pogobot")

#: Resource-id suffixes. Matched on suffix because the package prefix (`me.underw.hp`)
#: is PGSharp's and may vary between builds.
ID_LOGIN = "hl_account_item_logicon"
ID_DELETE = "hl_account_item_delete"
ID_TAB_ACCOUNTS = "hl_cdhist_cat_accounts"
ID_CLOSE = "hl_page_close"
ID_COOLDOWN_TEXT = "hl_cd_text"

_BOUNDS = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


@dataclass(frozen=True)
class AccountRow:
    """A single account entry in the PGSharp account list."""

    name: str
    active: bool
    level: Optional[int]
    login_norm: tuple[float, float]
    delete_norm: Optional[tuple[float, float]]
    row_norm: tuple[float, float, float, float]


@dataclass(frozen=True)
class AccountView:
    """The current state of the PGSharp account-list panel."""

    rows: tuple[AccountRow, ...] = ()
    launcher_norm: Optional[tuple[float, float]] = None
    accounts_tab_norm: Optional[tuple[float, float]] = None
    close_norm: Optional[tuple[float, float]] = None
    available: bool = False
    panel_open: bool = False

    @property
    def active(self) -> Optional[AccountRow]:
        return next((r for r in self.rows if r.active), None)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.rows)

    def by_name(self, name: str) -> Optional[AccountRow]:
        return next((r for r in self.rows if r.name == name), None)


def _rect(node) -> Optional[tuple[int, int, int, int]]:
    m = _BOUNDS.match(node.get("bounds", ""))
    return tuple(int(g) for g in m.groups()) if m else None


def _centre_norm(node, w: int, h: int) -> Optional[tuple[float, float]]:
    r = _rect(node)
    if r is None or w <= 0 or h <= 0:
        return None
    return ((r[0] + r[2]) / 2.0 / w, (r[1] + r[3]) / 2.0 / h)


def _ends_with(node, suffix: str) -> bool:
    return node.get("resource-id", "").endswith(suffix)


def parse_dump(xml: bytes, screen_wh: tuple[int, int]) -> AccountView:
    """Pure: uiautomator XML -> AccountView. Never raises."""
    w, h = screen_wh
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return AccountView(available=False)

    parents = {child: parent for parent in root.iter() for child in parent}

    launcher = accounts_tab = close = None
    for n in root.iter("node"):
        if _ends_with(n, ID_TAB_ACCOUNTS):
            accounts_tab = _centre_norm(n, w, h)
        elif _ends_with(n, ID_CLOSE):
            close = _centre_norm(n, w, h)
        elif _ends_with(n, ID_COOLDOWN_TEXT):
            # The launcher is the clickable ancestor of the cooldown readout. Locating it
            # this way rather than by coordinate is the difference between "works" and
            # "works on this phone at this resolution".
            cur = parents.get(n)
            while cur is not None:
                if cur.get("clickable") == "true":
                    launcher = _centre_norm(cur, w, h)
                    break
                cur = parents.get(cur)

    rows: list[AccountRow] = []
    for row in root.iter("node"):
        kids = list(row)
        ids = [k.get("resource-id", "") for k in kids]
        login_i = next((i for i, r in enumerate(ids) if r.endswith(ID_LOGIN)), None)
        if login_i is None:
            continue
        texts = [(k.get("text", ""), _rect(k)) for k in kids if k.get("text")]
        texts = [(t, r) for t, r in texts if r is not None]
        if not texts:
            continue
        # The name is the widest text node; the level digits sit in their own narrow one.
        name_text = max(texts, key=lambda tr: tr[1][2] - tr[1][0])[0]
        digits = [t for t, _ in texts if t.isdigit()]
        login_norm = _centre_norm(kids[login_i], w, h)
        if login_norm is None:
            continue
        delete_i = next((i for i, r in enumerate(ids) if r.endswith(ID_DELETE)), None)
        row_rect = _rect(row)
        rows.append(AccountRow(
            name=name_text.lstrip("*").strip(),
            active=name_text.startswith("*"),
            level=int(digits[-1]) if digits else None,
            login_norm=login_norm,
            delete_norm=_centre_norm(kids[delete_i], w, h) if delete_i is not None else None,
            row_norm=((row_rect[0] / w, row_rect[1] / h, row_rect[2] / w, row_rect[3] / h)
                      if row_rect else (0.0, 0.0, 0.0, 0.0)),
        ))

    return AccountView(
        rows=tuple(rows),
        launcher_norm=launcher,
        accounts_tab_norm=accounts_tab,
        close_norm=close,
        available=True,
        panel_open=close is not None,
    )


DUMP_PATH = "/sdcard/pogobot_ui.xml"


class UiTreeReader:
    """Runs `uiautomator dump` and parses the result. The only adb caller in this module.

    Blocking, roughly a second per call, so it is used during an account switch and never
    per frame. Any failure - adb gone, the dump timing out because the UI never went idle,
    a torn file - yields `available=False`, which the state machine treats as "could not
    look", not as "there are no accounts". The timeout is applied per subprocess call, not
    total for read(); worst case blocks for a multiple of it.
    """

    def __init__(self, screen_wh: tuple[int, int], serial: Optional[str] = None,
                 timeout: float = 20.0):
        self.screen_wh = screen_wh
        self.serial = serial
        self.timeout = timeout

    def _adb(self, *args: str) -> list[str]:
        return ["adb"] + (["-s", self.serial] if self.serial else []) + list(args)

    def _run(self) -> bytes:
        # uiautomator can report success while writing nothing; delete the file first
        # so a failed dump is indistinguishable from an empty file, not a stale read.
        subprocess.run(self._adb("shell", "rm", "-f", DUMP_PATH),
                       capture_output=True, timeout=self.timeout)
        subprocess.run(self._adb("shell", "uiautomator", "dump", DUMP_PATH),
                       capture_output=True, timeout=self.timeout)
        return subprocess.run(self._adb("shell", "cat", DUMP_PATH),
                              capture_output=True, timeout=self.timeout).stdout

    def read(self) -> AccountView:
        try:
            payload = self._run()
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("uiautomator dump failed: %s", exc)
            return AccountView(available=False)
        if not payload or b"<hierarchy" not in payload:
            return AccountView(available=False)
        return parse_dump(payload, self.screen_wh)


class FakeTreeReader:
    """Test double. Yields queued views, then repeats the last one forever."""

    def __init__(self, views):
        self._views = list(views) or [AccountView(available=False)]
        self.reads = 0

    def read(self) -> AccountView:
        self.reads += 1
        return self._views.pop(0) if len(self._views) > 1 else self._views[0]


#: Budget name for identify_account's taps, distinct from every FSM budget ("switch",
#: "tap", ...) so a startup identification can never share - or be starved by - a live
#: run's own rate-limit state for those budgets.
IDENTIFY_BUDGET = "identify"


def identify_account(tree_reader: "UiTreeReader", actuator,
                     settle: float = 1.0) -> Optional[AccountView]:
    """Best-effort, one-shot: open the PGSharp account panel, read it, close it again.
    Returns the panel AS READ - who is active AND which accounts exist - or None when it
    could not be read at all.

    The roster matters as much as the active name, which is why the whole view comes back
    rather than one string: the panel is shut for the rest of the run, so this is the only
    enumeration of the accounts there are. `Runner` decides which account to switch to
    from that cache, because a live read at any other moment reports `rows=()` however
    healthy PGSharp is - measured on the device as `available=True, panel_open=False,
    rows=0`, which is what made account switching never fire at all.

    `parse_dump` only ever sees account rows while the panel is open (see
    `AccountRow`/`AccountView` above) - a bare `tree_reader.read()` with the panel closed
    reports `rows=()` and `active=None` even when PGSharp and the account list are both
    completely healthy. This is the one-shot equivalent of what `Switching` already does
    one tap at a time (`pogobot/fsm.py`), run once at startup so the very first session
    can be attributed to a real account instead of the unattributed bucket.

    PGSharp also remembers the last-viewed tab across openings, so the panel can open
    already showing something other than the account list - Cooldown History, measured
    live - with zero rows to find anyone active in. `Switching.step`'s "tab" phase
    already handles exactly this by tapping `accounts_tab_norm`; mirrored here rather
    than inventing a second shape for the same problem.

    Every coordinate comes from a location the tree itself just reported - `launcher_norm`
    from the first read, `accounts_tab_norm` from the second if it opened on the wrong
    tab, `close_norm` from whichever read is current when it is time to close - never a
    constant, an offset, or a row's `delete_norm`, which sits close enough to `login_norm`
    that a guessed tap is how an account gets irreversibly deleted (see the module
    docstring).

    A first read that is unavailable, or that does not locate the launcher, is left
    strictly alone: nothing is tapped, and the function returns None. A control that was
    not located - the accounts tab, or the close button - is simply not tapped; a missing
    node still means do nothing, exactly as for the launcher. Whichever read is current by
    the end, the panel is left as this function found it - closed - by tapping its
    `close_norm` if one was located; an unavailable read never produces a guessed close.
    """
    view = tree_reader.read()
    if not view.available or view.launcher_norm is None:
        log.warning("could not locate the PGSharp overlay; per-account tracking is "
                    "unavailable unless --account is given")
        return None
    actuator.apply(Tap(*view.launcher_norm, "identify: open the PGSharp overlay",
                       budget=IDENTIFY_BUDGET))
    if settle:
        time.sleep(settle)
    opened = tree_reader.read()

    if opened.available and not opened.rows and opened.accounts_tab_norm is not None:
        actuator.apply(Tap(*opened.accounts_tab_norm, "identify: select the Accounts tab",
                           budget=IDENTIFY_BUDGET))
        if settle:
            time.sleep(settle)
        opened = tree_reader.read()

    if not opened.available:
        log.warning("PGSharp overlay did not respond after opening; per-account "
                    "tracking is unavailable unless --account is given")
        return None
    if opened.active is not None:
        log.info("logged in as %s (L%s), %d account(s) available",
                 opened.active.name, opened.active.level, len(opened.rows))
    elif not opened.panel_open:
        # Never report an opening that did not happen. A dry run suppresses the tap, so
        # the panel is exactly as it was found, and "opened but nobody is active" would
        # describe a screen nobody ever saw.
        log.warning("the PGSharp overlay did not open%s; per-account tracking is "
                    "unavailable unless --account is given",
                    " (taps are suppressed in a dry run)"
                    if getattr(actuator, "dry_run", False) else "")
    else:
        log.warning("PGSharp overlay opened but no account is marked active; "
                    "per-account tracking is unavailable unless --account is given")
    if opened.close_norm is not None:
        actuator.apply(Tap(*opened.close_norm, "identify: close the PGSharp overlay",
                           budget=IDENTIFY_BUDGET))
    return opened
