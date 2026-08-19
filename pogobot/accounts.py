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

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

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
    name: str
    active: bool
    level: Optional[int]
    login_norm: tuple[float, float]
    delete_norm: Optional[tuple[float, float]]
    row_norm: tuple[float, float, float, float]


@dataclass(frozen=True)
class AccountView:
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
