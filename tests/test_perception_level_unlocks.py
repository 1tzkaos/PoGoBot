"""The "NEW LEVEL UNLOCKS" screen (a reward card on a blue backdrop, closed by a round X
low-centre) wedged the bot on the real device.

Unlike the level-up screen, every signal here was already correct: the classifier reads
Menu @ 0.94, `x_button_signal` reads True, so `in_overlay` routed to POPUP exactly as
designed. POPUP then had nothing to press, because `find_close_button` returned None -
and POPUP only ever taps a button it actually located. So POPUP idled to its 4s timeout,
handed off to RECOVERING, whose BACK does not dismiss this screen, and RECOVERING found
no button either. The bot cycled there until a human intervened.

The locator missed because this button is painted in a cooler, paler palette than the
usual mint X: measured on the device capture, its ring runs hue 96-107 - past MINT_HI's
ceiling of 95 - and its glyph is nearly white (S~20, under MINT's S>=40 floor). So the
mint mask keeps only the small inner core of the glyph rather than the whole ring. That
core is perfectly round (aspect 1.00) and perfectly centred (cx 0.500), but only 3.2% of
frame width, so the old `cw < w * 0.05` floor discarded the one true candidate.

See perception.CLOSE_MIN_W for the floor's measurement against the labelled corpus.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest

from pogobot import perception as P
from pogobot.config import DEFAULT as C

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "screens"

#: The pre-fix floor. Used to prove this test exercises the fix rather than passing
#: vacuously.
_OLD_CLOSE_MIN_W = 0.05


def _load(name: str) -> np.ndarray:
    im = cv2.imread(str(FIXTURES / f"{name}.png"))
    assert im is not None, f"missing fixture tests/fixtures/screens/{name}.png"
    return im


def test_find_close_button_locates_the_x_on_new_level_unlocks():
    """The X sits at ~(0.50, 0.885) onscreen - dead centre, just above the nav bar."""
    got = P.find_close_button(_load("level_unlocks"), C)
    assert got is not None, "close button not located: POPUP would have nothing to tap"
    x, y = got
    assert x == pytest.approx(0.500, abs=0.01)
    assert y == pytest.approx(0.885, abs=0.01)


def test_old_width_floor_missed_the_new_level_unlocks_x(monkeypatch):
    """Red-green check: restoring only the 5% floor reproduces the MISS this change
    exists to fix."""
    monkeypatch.setattr(P, "CLOSE_MIN_W", _OLD_CLOSE_MIN_W)
    assert P.find_close_button(_load("level_unlocks"), C) is None


def test_level_unlocks_routes_to_popup_with_a_button_to_press():
    """The whole point: `in_overlay` must be true AND a button must be located, or POPUP
    idles to its timeout exactly as it did on the device."""
    im = _load("level_unlocks")
    mb = P.map_ball_signal(im, C)
    xb = P.x_button_signal(im, C, mb.value)
    assert xb.value, "x_button must read True or the bot never routes to POPUP"
    assert not mb.value, "this screen is not the map"
    assert P.find_close_button(im, C) is not None
