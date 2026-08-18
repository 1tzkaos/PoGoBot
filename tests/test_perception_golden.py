"""Golden tests over the labelled corpus.

These lock in the calibration measured on 235 labelled frames. They are the only way to
verify the optical layer without a phone, and they exist because v1's thresholds were
absolute pixel counts that silently stopped working when capture resolution changed.
"""
import collections
import pathlib

import cv2
import pytest

from pogobot import perception as P
from pogobot.config import DEFAULT as C

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "datasets" / "state_cls5"
pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="labelled corpus not present")


@pytest.fixture(scope="module")
def rates():
    out = collections.defaultdict(collections.Counter)
    for p in CORPUS.rglob("*.png"):
        im = cv2.imread(str(p))
        if im is None:
            continue
        r = out[p.parent.name]
        r["n"] += 1
        mb = P.map_ball_signal(im, C)
        if mb.value:
            r["map_ball"] += 1
        if P.find_close_button(im, C) and not mb.value:
            r["x_button"] += 1
        if P.find_action_pill(im, C):
            r["pill"] += 1
    return out


def frac(rates, cls, key):
    r = rates[cls]
    return r[key] / max(1, r["n"])


def test_map_signal_is_high_precision(rates):
    """It is the veto that prevents the v1 menu loop, so false positives matter most."""
    for cls in ("Menu", "Poi", "Rocket"):
        assert frac(rates, cls, "map_ball") <= 0.05, f"map_ball fired on {cls}"
    assert frac(rates, "PokemonEncounter", "map_ball") <= 0.05


def test_map_signal_recall_has_not_regressed(rates):
    assert frac(rates, "Overworld", "map_ball") >= 0.72


def test_close_button_found_on_screens_that_have_one(rates):
    assert frac(rates, "Menu", "x_button") >= 0.85


def test_close_button_not_hallucinated_on_encounters(rates):
    assert frac(rates, "PokemonEncounter", "x_button") <= 0.05


def test_action_pill_finds_the_rocket_buttons(rates):
    assert frac(rates, "Rocket", "pill") >= 0.70


def test_action_pill_does_not_fire_on_ordinary_menus(rates):
    assert frac(rates, "Menu", "pill") <= 0.10


def test_thresholds_are_fractions_not_pixel_counts():
    """A regression guard: every threshold must be scale free (<= 1.0 or a grey level)."""
    t = C.thresholds
    for name in ("map_ball_red", "map_bino_orange", "x_button_mint", "x_button_teal",
                 "encounter_ball", "encounter_flee_white", "out_of_range_pink",
                 "claim_teal", "claim_white_text"):
        assert 0.0 < getattr(t, name) <= 1.0, f"{name} looks like a pixel count"


def test_signals_are_resolution_invariant():
    """The v1 failure: at --max-size 720 orange measured 479 px against a > 500 bar and
    overworld detection silently stopped working."""
    src = next(iter((CORPUS / "train" / "Overworld").glob("*.png")), None)
    if src is None:
        pytest.skip("no overworld sample")
    im = cv2.imread(str(src))
    base = P.map_ball_signal(im, C)
    for max_size in (1920, 1280, 960, 720, 540):
        s = max_size / max(im.shape[:2])
        small = cv2.resize(im, (max(1, int(im.shape[1] * s)), int(im.shape[0] * s)),
                           interpolation=cv2.INTER_AREA) if s < 1 else im
        got = P.map_ball_signal(small, C)
        assert got.value == base.value, f"map_ball flipped at --max-size {max_size}"


def test_screen_stabilizer_rejects_a_single_bad_frame():
    """One misclassified frame must not move the state machine."""
    from pogobot.observation import ScreenGuess
    from pogobot.perception import ScreenStabilizer
    s = ScreenStabilizer(window=5, needed=3)
    for _ in range(4):
        s.push(ScreenGuess("Overworld", 0.99))
    out = s.push(ScreenGuess("PokemonEncounter", 0.97))
    assert out.label == "Overworld", "a lone outlier must not flip the stable label"


def test_screen_stabilizer_accepts_a_sustained_change():
    from pogobot.observation import ScreenGuess
    from pogobot.perception import ScreenStabilizer
    s = ScreenStabilizer(window=5, needed=3)
    for _ in range(5):
        s.push(ScreenGuess("Overworld", 0.99))
    for _ in range(3):
        out = s.push(ScreenGuess("PokemonEncounter", 0.97))
    assert out.label == "PokemonEncounter", "a real transition must be believed"


def test_stabilizer_passes_through_when_no_classifier():
    from pogobot.observation import ScreenGuess
    from pogobot.perception import ScreenStabilizer
    s = ScreenStabilizer()
    g = ScreenGuess("unknown", 0.0, available=False)
    assert s.push(g) is g
