"""A Gym screen wedged the bot with its close X located the whole time.

Unlike the "NEW LEVEL UNLOCKS" screen (see test_perception_level_unlocks.py), the locator
was never the problem here: `find_close_button` returned (0.500, 0.890) on every frame.
What failed was the gate in front of it. `x_button_signal` measured mint=0.0321 against a
0.035 floor, so `x_button` read False, `Observation.in_overlay` stayed False, and
`desired_state` never routed the screen to POPUP - the located button was simply never
consulted, because nothing believed there was an overlay to close.

The fixture is committed at 590x1280 because that is what the bot actually sees: scrcpy
scales the long side to `Config.max_size`, so a 1080x2340 phone streams at 590x1280, and
the same frame measures 0.0296 at native resolution against 0.0321 here.

The gym's real-world name identified where the user plays and has been blacked out. Every
signal this fixture exists for lives at y >= 0.74, well clear of the redaction.

See config.Thresholds.x_button_mint for the corpus sweep behind the chosen floor.
"""
from __future__ import annotations

import pathlib
from dataclasses import replace

import cv2
import numpy as np
import pytest

from pogobot import perception as P
from pogobot.config import DEFAULT as C

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "screens"

#: The pre-fix floor, to prove these tests exercise the change rather than pass vacuously.
_OLD_X_BUTTON_MINT = 0.035


def _load(name: str) -> np.ndarray:
    im = cv2.imread(str(FIXTURES / f"{name}.png"))
    assert im is not None, f"missing fixture tests/fixtures/screens/{name}.png"
    return im


def test_the_gym_close_button_was_always_locatable():
    """The half that was never broken, pinned so a future locator change cannot quietly
    take it away and leave the gate looking like the culprit."""
    got = P.find_close_button(_load("gym_close"), C)
    assert got is not None
    x, y = got
    assert x == pytest.approx(0.500, abs=0.01)
    assert y == pytest.approx(0.890, abs=0.01)


def test_the_gym_screen_now_reads_as_a_closable_overlay():
    """`in_overlay` is what routes to POPUP, so this is the assertion that matters."""
    im = _load("gym_close")
    mb = P.map_ball_signal(im, C)
    xb = P.x_button_signal(im, C, mb.value)
    assert xb.value, f"x_button still False at mint={xb.detail['mint']:.4f}"
    assert xb.detail["mint"] == pytest.approx(0.0321, abs=0.002)


def test_the_old_floor_reproduces_the_wedge():
    """Red-green: restoring only the 0.035 floor puts the screen back where it was - a
    located button nothing was allowed to press."""
    im = _load("gym_close")
    old = replace(C, thresholds=replace(C.thresholds, x_button_mint=_OLD_X_BUTTON_MINT))
    mb = P.map_ball_signal(im, old)
    assert not P.x_button_signal(im, old, mb.value).value
    assert P.find_close_button(im, old) is not None, "the locator was never the problem"
