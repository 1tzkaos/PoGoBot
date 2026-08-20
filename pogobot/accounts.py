"""The PGSharp account list, read from the Android view tree.

A second perception channel, deliberately narrow. Everything else the bot knows comes
from pixels; this reads real Android views, which is possible only because the PGSharp
overlay is drawn as views rather than into the Unity canvas. It cannot read the game:
uiautomator sees Pokemon GO itself as one opaque `View "Game view"`.

Why it is worth a second channel at all: the account list states which account is logged
in, as text, with an asterisk. That is ground truth. Every alternative - OCR of the map's
bottom-left name, or a classifier - would be an inference about something the system can
simply be told.

Safety note that drives the whole module: each row's delete button starts ~24px from the
edge of its login button - 157px centre to centre, measured off
tests/fixtures/uiautomator/accounts_open.xml. Every coordinate here is some node's OWN
bounds. Nothing is a constant, an offset, or a guess, because the failure mode is an
irreversibly deleted account.

The same channel also reads PGSharp's floating "star" shortcut widget, the menu it opens,
and the AutoWalk dialog that menu can lead to (see `fsm.Switching._autowalk_open` and
neighbours) - not a second channel, the same dump, parsed further. The star moves (it is
described as draggable, and was measured at two different positions hours apart - see
tests/fixtures/uiautomator/star_moved.xml), so it is located the same way the cooldown
launcher already is: through a stable descendant id (`hl_floating_icon`, confirmed present
in tests/fixtures/uiautomator/accounts_open.xml) walked up to its nearest clickable
ancestor, never by class+clickable alone or by a remembered coordinate.

Because BOTH of those widgets float and are draggable, they can also end up drawn on top
of one another - measured immediately after an app restart, and reproduced in
tests/fixtures/uiautomator/overlay_collapsed.xml. That is why this module reports each
one's clickable RECT as well as its centre: two centres cannot express an overlap, and the
overlap is what turns a tap meant for the star into a tap on the accounts launcher. See
`AccountView.overlay_collapsed` for the measurement and `fsm.Switching._separate_star` for
what acts on it.
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
#: The icon ImageView inside the star widget - see the module docstring. Located the
#: same way ID_COOLDOWN_TEXT is: walk up from this id to the nearest clickable ancestor.
ID_STAR_ICON = "hl_floating_icon"
#: Shortcut-menu entries the star opens ('Map', 'AutoWalk', 'Feeds', ...). Each is its
#: own directly-clickable text node - the same shape ID_TAB_ACCOUNTS already is.
ID_SHORTCUT_ITEM = "hl_shortcut_menu_item_txt"
#: The AutoWalk dialog. `alertTitle`/`button1`/`button2`/`button3` are Android's own
#: framework AlertDialog ids, not PGSharp's - matched by suffix like everything else here
#: so the exact package prefix (`android:id/...`) is never assumed. `hl_aw_input` and the
#: toggle ids are never looked up at all: this module has no coordinate for any of them,
#: which is what makes them untappable rather than merely un-tapped (see
#: `fsm.Switching._autowalk_dialog`).
ID_AW_TITLE = "alertTitle"
ID_AW_TITLE_TEXT = "Auto-Generated GPX"
ID_AW_OK = "button1"            # OK - the default (50 POIs)
ID_AW_CANCEL = "button2"        # CANCEL - never tapped by this module
ID_AW_CONTINUE_LAST = "button3"  # CONTINUE LAST - present only sometimes; preferred when it is

_BOUNDS = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


@dataclass(frozen=True)
class AccountRow:
    """A single account entry in the PGSharp account list."""

    name: str
    active: bool
    level: Optional[int]
    login_norm: tuple[float, float]
    delete_norm: Optional[tuple[float, float]]


@dataclass(frozen=True)
class AccountView:
    """The current state of the PGSharp overlay: the account-list panel, and (see the
    module docstring) the star shortcut widget, the menu it opens, and the AutoWalk
    dialog that menu can lead to. All parsed from the same dump by `parse_dump`."""

    rows: tuple[AccountRow, ...] = ()
    launcher_norm: Optional[tuple[float, float]] = None
    #: The accounts/cooldown launcher's own clickable bounds, normalized - the SAME node
    #: `launcher_norm` is the centre of. Present for exactly one reason: an overlap cannot
    #: be expressed by two centres (see `overlay_collapsed`), and neither can the landing
    #: zone a separating drag has to aim at (see `star_clear_y_norm`).
    launcher_rect_norm: Optional[tuple[float, float, float, float]] = None
    accounts_tab_norm: Optional[tuple[float, float]] = None
    close_norm: Optional[tuple[float, float]] = None
    available: bool = False
    panel_open: bool = False
    #: The star widget's own current position - see ID_STAR_ICON. None means "not found
    #: in this dump", never a stale or assumed location.
    star_norm: Optional[tuple[float, float]] = None
    #: The star's own clickable bounds, normalized - the SAME node `star_norm` is the
    #: centre of, and the counterpart of `launcher_rect_norm` above.
    star_rect_norm: Optional[tuple[float, float, float, float]] = None
    #: The shortcut menu's "AutoWalk" entry, present only once the star has been tapped
    #: and the menu has actually rendered.
    autowalk_menu_norm: Optional[tuple[float, float]] = None
    #: The AutoWalk entry's own ICON BOX: x from the screen's own left edge to the
    #: label's own left edge, y over the label's own vertical bounds - both taken from
    #: the SAME node `autowalk_menu_norm` was located from, in the same pass, never a
    #: hardcoded rectangle. This is bounds only; see `perception.autowalk_active_signal`
    #: for what reads the colour inside it and `fsm.Switching._autowalk_menu` for the
    #: decision that colour drives ("if the icon reads blue, this account is already
    #: autowalking - do not tap it again"). None whenever `autowalk_menu_norm` is None -
    #: the two can only ever appear or disappear together.
    autowalk_icon_rect_norm: Optional[tuple[float, float, float, float]] = None
    #: True only when a node identifies THIS dialog by its title text - never inferred
    #: from button1/button2/button3 alone, which are generic Android AlertDialog ids that
    #: could in principle belong to a different dialog.
    autowalk_dialog_open: bool = False
    #: CONTINUE LAST (button3) - present only sometimes; preferred over OK when it is.
    autowalk_continue_last_norm: Optional[tuple[float, float]] = None
    #: OK (button1) - the default (50 POIs).
    autowalk_ok_norm: Optional[tuple[float, float]] = None

    @property
    def overlay_collapsed(self) -> bool:
        """True when PGSharp's two floating widgets are drawn on top of each other.

        Measured on the device immediately after `effects.RestartApp` relaunched the game:
        the star's clickable rect was (0,152)-(108,260) and the launcher's (0,152)-(272,245)
        - a 108x93 overlap covering nearly the whole star, with the star's own CENTRE
        (54,206) inside the launcher's rect. That centre is exactly what `star_norm`
        reports and what `fsm.Switching._autowalk_open` taps, so in this layout a tap meant
        for the star is delivered to the accounts launcher instead - which opens the
        PGSharp accounts panel, the screen `fsm.Recovering._panel_close` and the restart
        ladder above it exist to get back out of. The restart is itself what produces the
        layout, so a restart shipped without this would have the bot cause the wedge it
        recovers from.

        Rect overlap, deliberately NOT the narrower "the star's centre is inside the
        launcher's rect". That narrower condition is what makes the tap land wrong, and it
        is a strict subset of this one; the wider test also covers the partial stack the
        user reports having to drag apart by hand, and errs toward separating widgets that
        were already usable rather than toward tapping a control that opens the panel.

        False whenever either rect is missing. An absent widget is "could not look", never
        "they are separated" - the same distinction `available` draws for the dump as a
        whole.
        """
        star, launcher = self.star_rect_norm, self.launcher_rect_norm
        if star is None or launcher is None:
            return False
        return (star[0] < launcher[2] and launcher[0] < star[2]
                and star[1] < launcher[3] and launcher[1] < star[3])

    @property
    def star_clear_y_norm(self) -> Optional[float]:
        """Where the star's CENTRE has to end up for it to clear the accounts launcher.

        Derived from the two rects THIS dump reported and nothing else: the launcher's own
        bottom edge, plus half a star so the star's top edge is level with it, plus one
        more whole star height of margin.

        The margin is measured in the star's own height because that is the only length
        the tree offers for this widget, and because it is exactly the amount by which the
        drag is allowed to fall short. `input swipe` on this overlay does not land where
        it is aimed: measured, asking for y=626 landed the centre at 837, asking for 339
        landed at 443, and asking for 356 moved it the other way entirely, to 125. A
        landing zone a full widget clear of the launcher survives a miss of that size in
        the near direction; a miss in the other direction is what `fsm.Switching
        ._separate_star` re-reads the tree for, rather than trusting any one drag.

        BELOW the launcher, not above it: after a restart the launcher's top edge is the
        top of the usable screen (measured y=152 of 2340, just under the status bar), so
        there is no room above it, and the separation that was actually verified on the
        device left the star at (0,389)-(108,497) - below.

        None when either rect is missing, or when the result would push the star off the
        bottom of the screen. A drag has two endpoints and both have to be somewhere the
        screen actually has; refusing is what stops this inventing one.

        Clearing the launcher is necessary and not sufficient, and the rest of the
        judgement is deliberately not here: whether the answer also sits outside the reach
        ellipse SCANNING taps into is checked by `fsm.Switching._separate_star`, because
        that ellipse lives in fsm.py and fsm.py is what imports THIS module. A widget
        geometry module that guessed at the ellipse would be a second copy of it.
        """
        star, launcher = self.star_rect_norm, self.launcher_rect_norm
        if star is None or launcher is None:
            return None
        height = star[3] - star[1]
        y = launcher[3] + height / 2.0 + height
        return y if y + height / 2.0 <= 1.0 else None

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


def _rect_norm(node, w: int, h: int) -> Optional[tuple[float, float, float, float]]:
    """A node's own bounds, normalized. Used only where the shape of the bounds matters,
    not just their centre - see `autowalk_icon_rect_norm`'s derivation below."""
    r = _rect(node)
    if r is None or w <= 0 or h <= 0:
        return None
    x0, y0, x1, y1 = r
    return (x0 / w, y0 / h, x1 / w, y1 / h)


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

    def _clickable_ancestor(node):
        """Walk up from `node` to its nearest clickable ancestor, or None. Locating a
        launcher this way - through a stable descendant id rather than a coordinate, and
        through the tree's own parent links rather than an assumption about how many
        levels separate the two - is the difference between "works" and "works on this
        phone at this resolution"."""
        cur = parents.get(node)
        while cur is not None:
            if cur.get("clickable") == "true":
                return cur
            cur = parents.get(cur)
        return None

    launcher = accounts_tab = close = star = None
    launcher_rect = star_rect = None
    autowalk_menu = None
    autowalk_icon_rect = None
    autowalk_dialog_open = False
    autowalk_continue_last = autowalk_ok = None
    for n in root.iter("node"):
        if _ends_with(n, ID_TAB_ACCOUNTS):
            accounts_tab = _centre_norm(n, w, h)
        elif _ends_with(n, ID_CLOSE):
            close = _centre_norm(n, w, h)
        elif _ends_with(n, ID_COOLDOWN_TEXT):
            anc = _clickable_ancestor(n)
            if anc is not None:
                launcher = _centre_norm(anc, w, h)
                # The rect as well as the centre, for both floating widgets: two centres
                # cannot say whether the widgets overlap, and both float (see the module
                # docstring and AccountView.overlay_collapsed).
                launcher_rect = _rect_norm(anc, w, h)
        elif _ends_with(n, ID_STAR_ICON):
            # Same idiom as the cooldown launcher just above, deliberately not folded
            # into one branch: the two anchor ids identify two DIFFERENT widgets, and
            # collapsing them would make a future third widget one `elif` away from
            # being confused with either.
            anc = _clickable_ancestor(n)
            if anc is not None:
                star = _centre_norm(anc, w, h)
                star_rect = _rect_norm(anc, w, h)
        elif _ends_with(n, ID_SHORTCUT_ITEM):
            if n.get("text") == "AutoWalk":
                autowalk_menu = _centre_norm(n, w, h)
                item_rect = _rect_norm(n, w, h)
                if item_rect is not None:
                    # The icon box: x from the screen's own left edge to the label's OWN
                    # left edge, y over the label's own vertical bounds - see
                    # AccountView.autowalk_icon_rect_norm and perception.autowalk_active_signal.
                    autowalk_icon_rect = (0.0, item_rect[1], item_rect[0], item_rect[3])
        elif _ends_with(n, ID_AW_TITLE):
            # The one place this module reads TEXT to decide something rather than just
            # to display it: button1/2/3 are generic Android AlertDialog ids that some
            # other dialog could also use, so "this is the AutoWalk dialog" is only ever
            # true when its own title says so.
            autowalk_dialog_open = n.get("text") == ID_AW_TITLE_TEXT
        elif _ends_with(n, ID_AW_CONTINUE_LAST):
            autowalk_continue_last = _centre_norm(n, w, h)
        elif _ends_with(n, ID_AW_OK):
            autowalk_ok = _centre_norm(n, w, h)

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
        rows.append(AccountRow(
            name=name_text.lstrip("*").strip(),
            active=name_text.startswith("*"),
            level=int(digits[-1]) if digits else None,
            login_norm=login_norm,
            delete_norm=_centre_norm(kids[delete_i], w, h) if delete_i is not None else None,
        ))

    return AccountView(
        rows=tuple(rows),
        launcher_norm=launcher,
        launcher_rect_norm=launcher_rect,
        accounts_tab_norm=accounts_tab,
        close_norm=close,
        available=True,
        # Account rows are only ever in the tree while the panel is open, so either
        # signal alone proves it. Hanging this on `hl_page_close` by itself made one
        # PGSharp resource-id load-bearing: if it ever moves, `Switching.step` reads an
        # open panel as closed and taps the launcher, which TOGGLES the overlay shut and
        # then open again until the switch times out.
        panel_open=close is not None or bool(rows),
        star_norm=star,
        star_rect_norm=star_rect,
        autowalk_menu_norm=autowalk_menu,
        autowalk_icon_rect_norm=autowalk_icon_rect,
        autowalk_dialog_open=autowalk_dialog_open,
        autowalk_continue_last_norm=autowalk_continue_last,
        autowalk_ok_norm=autowalk_ok,
    )


DUMP_PATH = "/sdcard/pogobot_ui.xml"


class UiTreeReader:
    """Runs `uiautomator dump` and parses the result. The only adb caller in this module.

    Blocking, roughly a second per call, so it is used during an account switch or a
    recovery attempt and never per frame. Any failure - adb gone, the dump timing out
    because the UI never went idle, a torn file - yields `available=False`, which the
    state machine treats as "could not look", not as "there are no accounts".

    `timeout` is the budget for the WHOLE of `read()`, shared across its three adb calls.
    It used to be applied per call, which made it no bound at all: the caller runs this on
    the run loop's own thread (`Runner._refresh_accounts`), and three calls at the old 20s
    default was a 60s stall in which no frame is perceived, no key is read and no signal
    is serviced. Five seconds is five times the measured cost of a real read, and it is
    deliberately shorter than the ~10s `uiautomator dump` spends waiting for a window that
    never goes idle: against a rendering Unity surface there is no PGSharp panel to find,
    so giving up first is the answer we wanted anyway.
    """

    def __init__(self, screen_wh: tuple[int, int], serial: Optional[str] = None,
                 timeout: float = 5.0):
        self.screen_wh = screen_wh
        self.serial = serial
        self.timeout = timeout

    def _adb(self, *args: str) -> list[str]:
        return ["adb"] + (["-s", self.serial] if self.serial else []) + list(args)

    def _run(self) -> bytes:
        deadline = time.perf_counter() + self.timeout

        def left() -> float:
            # Floored, never zero: subprocess.run treats a non-positive timeout as already
            # expired and raises before the process is even started, which would report a
            # merely slow first call as "adb went away" rather than as a dump that ran out
            # of time. Both end in available=False, but only one of them is true.
            return max(0.05, deadline - time.perf_counter())

        # uiautomator can report success while writing nothing; delete the file first
        # so a failed dump is indistinguishable from an empty file, not a stale read.
        subprocess.run(self._adb("shell", "rm", "-f", DUMP_PATH),
                       capture_output=True, timeout=left())
        subprocess.run(self._adb("shell", "uiautomator", "dump", DUMP_PATH),
                       capture_output=True, timeout=left())
        return subprocess.run(self._adb("shell", "cat", DUMP_PATH),
                              capture_output=True, timeout=left()).stdout

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
    can be attributed to a real account instead of the unattributed bucket, and so a
    switch has a roster to choose a target from. Whether it runs at all is
    `cli.prepare_accounts`'s decision, not this function's: these are three taps into the
    panel that holds the delete buttons, which is a fair price for a run that is going to
    drive that panel anyway and no price for a run that will never switch.

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
