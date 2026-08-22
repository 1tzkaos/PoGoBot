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


# --------------------------------------------- the operator's own tell: it offers to SAVE

def test_the_ad_offers_to_save_itself_and_a_real_fight_never_does():
    """The identification, in the operator's words: an advertisement has a control in the
    bottom right to save the promotion, beside the X. Measured 0 of 13 across every
    Rocket-class frame in the corpus."""
    from pogobot.perception import promo_save_button
    ad = promo_save_button(cv2.imread(str(FIXTURES / "sponsored_ad.png")), C)
    assert ad is not None
    assert ad[0] > 0.8, "the save control sits in the bottom RIGHT"

    fires = [f for d in ("GruntBattleButton", "GruntDialogue", "ChooseParty",
                         "ExitTrainerBattle")
             for f in glob.glob(f"datasets/state_v3/*/{d}/*")
             if cv2.imread(f) is not None
             and promo_save_button(cv2.imread(f), C) is not None]
    assert fires == [], f"a real Rocket screen appeared to offer a save: {fires}"


def test_the_ad_is_not_treated_as_a_rocket_fight(perceive):
    """Following it into ROCKET costs the 150s that state holds for, during which nothing
    else may claim the screen - which is how three runs ended in a browser."""
    o = perceive(cv2.imread(str(FIXTURES / "sponsored_ad.png")))
    assert o.promo_save_xy is not None
    assert not fsm.rocket_screen(o, C)
    c = fsm.Context(cfg=C, state=BotState.SCANNING, state_since=99.5, now=100.0)
    assert fsm.desired_state(o, c) is not BotState.ROCKET


def test_a_real_fight_is_still_a_rocket_screen(perceive):
    """The veto must cost nothing."""
    kept = 0
    for f in sorted(glob.glob("datasets/state_v3/*/GruntBattleButton/*")):
        im = cv2.imread(f)
        if im is None:
            continue
        if fsm.rocket_screen(perceive(im), C):
            kept += 1
    assert kept >= 4, f"only {kept} real fights still read as Rocket screens"


def test_the_save_control_is_never_tapped(perceive):
    """Saving an advertisement is not the bot's business; the control exists here only to
    identify the screen."""
    o = perceive(cv2.imread(str(FIXTURES / "sponsored_ad.png")))
    for state in (BotState.SCANNING, BotState.ROCKET, BotState.POPUP, BotState.RECOVERING):
        c = fsm.Context(cfg=C, state=state, state_since=99.5, now=100.0)
        for e in fsm.step(o, c):
            if isinstance(e, Tap):
                assert abs(e.x - o.promo_save_xy[0]) > 0.05, f"{state} tapped the save button"
