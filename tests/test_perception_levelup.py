"""The level-up screen ("LEVEL n", teal background, CLAIM REWARDS pill) wedged the bot
on the real device: `claim_pill_signal` reads True on it, but the old unbounded-
saturation `GREEN_PILL_HI` band let the screen's own teal background - same hue as the
pill, S~253 vs the pill's S~153 - merge into one contour that failed every
width/height/aspect check in `find_action_pill`. With `action_pill_xy` staying None,
`fsm.interrupts()`'s claim branch had nothing to tap, so `in_overlay` routed to POPUP,
POPUP's "close button" was actually the SHARE row, the tap missed, POPUP timed out into
RECOVERING, and RECOVERING's BACK press opened the game's own exit-confirmation dialog.

See perception.GREEN_PILL_HI for the saturation-cap fix and its full measurement.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest

from pogobot import perception as P
from pogobot.config import DEFAULT as C
from pogobot.frames import Frame
from pogobot.perception import Perceptor

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "screens"
CORPUS = pathlib.Path(__file__).resolve().parent.parent / "datasets" / "state_v3"

LEVELUP_FIXTURES = ("levelup_12", "levelup_14")

#: The pre-fix band: identical except the saturation ceiling was left wide open at 255.
#: Used to prove these tests actually exercise the fix rather than passing vacuously.
_OLD_GREEN_PILL_HI = np.array([95, 255, 255])


def _load(name: str) -> np.ndarray:
    im = cv2.imread(str(FIXTURES / f"{name}.png"))
    assert im is not None, f"missing fixture tests/fixtures/screens/{name}.png"
    return im


@pytest.mark.parametrize("name", LEVELUP_FIXTURES)
def test_find_action_pill_locates_claim_rewards_on_levelup(name):
    """Both real captures locate CLAIM REWARDS at ~(0.499, 0.801) - exactly where it
    sits onscreen, per the measurement recorded on GREEN_PILL_HI."""
    got = P.find_action_pill(_load(name), C)
    assert got is not None, f"{name}: pill not located"
    x, y = got
    assert x == pytest.approx(0.499, abs=0.01)
    assert y == pytest.approx(0.801, abs=0.01)


@pytest.mark.parametrize("name", LEVELUP_FIXTURES)
def test_old_unbounded_saturation_band_missed_the_levelup_pill(name, monkeypatch):
    """Red-green check: reverting only the saturation cap reproduces the MISS this
    change exists to fix, so the fix (not something else) is what makes the test above
    pass."""
    monkeypatch.setattr(P, "GREEN_PILL_HI", _OLD_GREEN_PILL_HI)
    assert P.find_action_pill(_load(name), C) is None


def test_levelup_observation_has_a_tappable_action_pill():
    """End-to-end consequence: an Observation built from a level-up capture the way
    Perceptor.observe builds every other one gives fsm.interrupts()'s claim branch
    (fsm.py, `if obs.claim_pill.value ... obs.action_pill_xy is not None`) something to
    tap, instead of routing into the POPUP/RECOVERING/BACK chain that opened the game's
    exit-confirmation dialog."""
    perceptor = Perceptor(C)
    frame = Frame(seq=1, ts=0.0, bgr=_load("levelup_14"))
    o = perceptor.observe(frame, run_detector=False)
    assert o.claim_pill.value, "fixture no longer reads as a claim screen"
    assert o.action_pill_xy is not None


# ------------------------------------------------------------- labelled-corpus safety


def _pill_hit_rate(cls: str) -> tuple[int, int]:
    hits = total = 0
    for split in ("train", "valid", "test"):
        d = CORPUS / split / cls
        if not d.exists():
            continue
        for p in sorted(d.glob("*")):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            im = cv2.imread(str(p))
            if im is None:
                continue
            total += 1
            if P.find_action_pill(im, C) is not None:
                hits += 1
    return hits, total


@pytest.mark.skipif(not CORPUS.exists(), reason="datasets/state_v3 not present (gitignored)")
@pytest.mark.parametrize("cls", ["GruntBattleButton", "ChooseParty"])
def test_pills_already_found_are_still_found(cls):
    """The saturation cap must not cost the finder the rocket-screen pills it already
    located: GruntBattleButton and ChooseParty both measured 5/5 before this change."""
    hits, total = _pill_hit_rate(cls)
    if total == 0:
        pytest.skip(f"no {cls} samples in the corpus")
    assert hits == total, f"{cls}: {hits}/{total} pills found - regression from the saturation cap"


@pytest.mark.skipif(not CORPUS.exists(), reason="datasets/state_v3 not present (gitignored)")
@pytest.mark.parametrize("cls", ["Overworld", "PokemonEncounter", "GoBattleLeagueMain", "Pokedex", "Shop"])
def test_no_new_false_positives_on_non_pill_screens(cls, monkeypatch):
    """The saturation cap must not hallucinate pills the unbounded band never saw:
    whatever the old band found on these classes, the new band finds exactly the same -
    zero new detections, per the corpus-wide sweep."""
    after_hits, total = _pill_hit_rate(cls)
    if total == 0:
        pytest.skip(f"no {cls} samples in the corpus")
    monkeypatch.setattr(P, "GREEN_PILL_HI", _OLD_GREEN_PILL_HI)
    before_hits, _ = _pill_hit_rate(cls)
    assert after_hits == before_hits, (
        f"{cls}: saturation cap changed the pill hit rate ({before_hits} -> {after_hits})"
    )
