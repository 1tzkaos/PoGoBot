"""Pokemon GO's own SPONSORED interstitial nearly cost the bot every run it started.

The screen is a full-bleed advertisement with a teal "LEARN MORE" pill where a Rocket
fight's affirmative sits, and an X below it. It classifies as `Rocket@0.62` - over the 0.60
`screen_min_conf` - so `desired_state` routes to ROCKET and `Rocket.step` presses the pill,
believing it to be the affirmative.

Pressing it opens the advertiser's site. Captured live from logcat:

    ActivityTaskManager: START u0 {act=VIEW dat=https://www.mlb.com:443/...
                                   cmp=com.android.chrome/...}

after which the bot is not looking at the game at all: three consecutive runs ended in a
browser, two of them halting on "no usable frames" because a near-static web page barely
encodes any.

The screens separate cleanly on classifier confidence - every labelled Rocket frame in the
corpus reads 1.00, the ad reads 0.62 - which is what `rocket_pill_min_conf` spends.
"""
from __future__ import annotations

import glob
import pathlib
import time

import cv2
import pytest

from pogobot import fsm
from pogobot import perception as P
from pogobot.config import DEFAULT as C
from pogobot.effects import BotState, Tap

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "screens"


@pytest.fixture(scope="module")
def perceive():
    from ultralytics import YOLO
    from pogobot.config import Config, BASE_DIR
    from pogobot.frames import Frame
    from pogobot.perception import Perceptor
    cfg = Config()
    big = BASE_DIR / "models" / "v3" / "det_s" / "weights" / "best.pt"
    if big.exists():
        cfg = cfg.scaled(det_model=big)
    per = Perceptor(cfg, det_model=YOLO(str(cfg.det_model)), cls_model=YOLO(str(cfg.cls_model)),
                    device="mps", square_cls_input=True)

    def go(im):
        for k in range(4):
            o = per.observe(Frame(seq=k, ts=time.perf_counter(), bgr=im))
        return o
    return go


def _ctx(now=100.0):
    return fsm.Context(cfg=C, state=BotState.ROCKET, state_since=now - 0.5, now=now)


def test_the_ad_carries_a_pill_exactly_where_the_affirmative_sits(perceive):
    """The trap itself, pinned so nobody can call this screen harmless."""
    o = perceive(cv2.imread(str(FIXTURES / "sponsored_ad.png")))
    assert o.action_pill_xy is not None, "no pill - this fixture no longer tests anything"
    assert o.screen.label == "Rocket"
    assert o.screen.conf < C.rocket_pill_min_conf
    assert o.screen.conf > C.screen_min_conf, (
        "the ad must still clear the ordinary bar, or the gate is not what saves us")


def test_the_bot_never_presses_learn_more(perceive):
    """The assertion that matters: pressing it leaves the game entirely."""
    o = perceive(cv2.imread(str(FIXTURES / "sponsored_ad.png")))
    taps = [e for e in fsm.step(o, _ctx()) if isinstance(e, Tap)]
    for t in taps:
        assert abs(t.y - o.action_pill_xy[1]) > 0.03, (
            f"tapped {(t.x, t.y)}, which is the LEARN MORE pill at {o.action_pill_xy}")


def test_a_real_rocket_fight_still_presses_its_affirmative(perceive):
    """The bar must cost nothing: every labelled Rocket frame classifies at 1.00."""
    pressed = 0
    for f in sorted(glob.glob("datasets/state_v3/*/GruntBattleButton/*")):
        im = cv2.imread(f)
        if im is None:
            continue
        o = perceive(im)
        if o.action_pill_xy is None:
            continue
        assert o.screen.conf >= C.rocket_pill_min_conf, f"{f} would now be refused"
        taps = [e for e in fsm.step(o, _ctx()) if isinstance(e, Tap)]
        if taps and abs(taps[0].y - o.action_pill_xy[1]) < 0.01:
            pressed += 1
    assert pressed >= 3, f"only {pressed} real fights still press their affirmative"


def test_the_gate_is_the_thing_that_refuses_it(perceive):
    """Red-green in one test: drop the bar to the ordinary one and the ad is pressed."""
    from dataclasses import replace
    o = perceive(cv2.imread(str(FIXTURES / "sponsored_ad.png")))
    loose = replace(C, rocket_pill_min_conf=C.screen_min_conf)
    c = fsm.Context(cfg=loose, state=BotState.ROCKET, state_since=99.5, now=100.0)
    taps = [e for e in fsm.step(o, c) if isinstance(e, Tap)]
    assert taps and abs(taps[0].y - o.action_pill_xy[1]) < 0.01, (
        "with the bar lowered the ad should be pressed again - if not, something else "
        "is refusing it and this fixture is not testing the gate")
