"""The post-login "Stay Aware of Your Surroundings" splash has a green OK pill that
`find_action_pill` missed, stranding the bot on that screen. Not the colour mask, and not
the shape gate - the candidate contour passes both (0.51w x 0.063h) - it failed the
white-text gate: the finder measured white over the pill's full inner width (`wide`),
and "OK" is two characters on a wide pill, where "CLAIM REWARDS" fills it.

See `perception.PILL_WHITE_MIDDLE` for the full corpus sweep behind the fix: accept a
candidate when `wide >= PILL_WHITE_WIDE` OR `middle >= PILL_WHITE_MIDDLE`, where `middle`
is the same white measure taken only over the pill's centred quarter (x 0.38-0.62), the
window a short centred label still lands in.
"""
from __future__ import annotations

import pathlib

import cv2
import pytest

from pogobot import perception as P
from pogobot.config import DEFAULT as C

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "screens"
CORPUS = pathlib.Path(__file__).resolve().parent.parent / "datasets" / "state_v3"


def _load(name: str):
    im = cv2.imread(str(FIXTURES / f"{name}.png"))
    assert im is not None, f"missing fixture tests/fixtures/screens/{name}.png"
    return im


def test_find_action_pill_locates_ok_on_the_stay_aware_splash():
    """The real capture, already downscaled to 590x1280 - the resolution the bot
    actually processes (see tests/factories.py's `obs()` default `stream_wh`)."""
    got = P.find_action_pill(_load("postlogin_ok"), C)
    assert got is not None, "OK pill not located"


def test_the_old_wide_only_rule_missed_it(monkeypatch):
    """Red-green check: pushing PILL_WHITE_MIDDLE out of reach reproduces the wide-only
    rule this change replaces - proving the MIDDLE gate, not something else, is what
    makes the test above pass. (Verified independently at the fixture's own resolution:
    the pre-fix code returns None for it.)"""
    monkeypatch.setattr(P, "PILL_WHITE_MIDDLE", 1.0)   # unreachable -> wide alone decides
    assert P.find_action_pill(_load("postlogin_ok"), C) is None


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
    """The middle-window OR-gate must not cost the finder the rocket-screen pills it
    already located: both classes measured 5/5 before this change."""
    hits, total = _pill_hit_rate(cls)
    if total == 0:
        pytest.skip(f"no {cls} samples in the corpus")
    assert hits == total, (
        f"{cls}: {hits}/{total} pills found - regression from the middle-window gate")


@pytest.mark.skipif(not CORPUS.exists(), reason="datasets/state_v3 not present (gitignored)")
@pytest.mark.parametrize("cls", ["Overworld", "PokemonEncounter"])
def test_no_new_false_positives(cls, monkeypatch):
    """Whatever the wide-only rule found on these classes, the OR-gate finds exactly the
    same - zero new detections, per the corpus-wide sweep behind PILL_WHITE_MIDDLE."""
    after_hits, total = _pill_hit_rate(cls)
    if total == 0:
        pytest.skip(f"no {cls} samples in the corpus")
    monkeypatch.setattr(P, "PILL_WHITE_MIDDLE", 1.0)
    before_hits, _ = _pill_hit_rate(cls)
    assert after_hits == before_hits, (
        f"{cls}: the middle-window gate changed the pill hit rate "
        f"({before_hits} -> {after_hits})")
