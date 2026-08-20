"""Pokemon GO's own "Do you want to exit Pokemon GO?" confirm dialog - found by live
testing, twice. It classifies as Rocket (its own flat teal background plus a dialog
card reads as a Rocket dialogue screen), which used to route the machine into ROCKET,
where the fixed dialogue-advance tap (0.50, 0.62) lands on empty space below the
dialog's CANCEL label and the state sits there for the whole 150s ROCKET timeout before
handing back to RECOVERING, which hands straight back to ROCKET - measured live: 430s
consumed for two ROCKET<->RECOVERING round trips and nothing else, twice blocking live
verification of this branch.

Two layers, tested separately, the same split every other optical signal in this suite
uses (tests/test_goplus.py):

  * `perception.exit_dialog_signal` is a pure function of the ROI alone. The synthetic
    frames below are built at the exact measured HSV band for the teal surround (H
    70-100, S>=60, V>=90 - the one number the task brief actually reports) and
    round-tripped through real cv2 HSV<->BGR conversion, not injected as a mock.
  * `fsm.rocket_screen`'s veto and `fsm.interrupts`'s BACK response are where "on the
    map" is irrelevant and the actual response is decided. Their tests drive `fsm.step`
    the same way every other interrupt/veto test in this suite does.

Only TWO real positive samples exist for this signal (see config.Thresholds); that is
said here as plainly as it is said in the code, not implied to be a well-sampled
threshold. What makes that acceptable is the response: BACK carries no coordinate at
all, unlike this dialog's own OK button which sits close enough to ROCKET's fixed tap
that a coordinate-based response would risk quitting the game outright - so a false
positive here costs one extra BACK press, nothing more.
"""
import cv2
import numpy as np
import pytest

from pogobot import fsm
from pogobot import perception as P
from pogobot.config import DEFAULT
from pogobot.effects import Back, BotState, Note, Tap, Transition
from tests.factories import obs

C = DEFAULT

# ------------------------------------------------------------------ synthetic frames

def _roi_pixel_shape(rect, size=(1080, 2340)):
    """The exact pixel rectangle `perception.crop` will cut for `rect` at this frame
    size - computed the same way `crop` does, so a patch built to this shape lands back
    in the signal's hands unchanged. Mirrors tests/test_goplus.py's helper of the same
    purpose."""
    w, h = size
    x0, y0, x1, y1 = rect
    px0, py0 = int(w * x0), int(h * y0)
    px1, py1 = int(w * x1), int(h * y1)
    return (py1 - py0, px1 - px0), (px0, py0)


def _flat(shape, hsv):
    ph, pw = shape
    flat = np.empty((ph, pw, 3), dtype=np.uint8)
    flat[:] = hsv
    return flat


def _frame(*, top=None, bottom=None, card=None, size=(1080, 2340)):
    """A full frame with each named ROI filled with a uniform HSV colour and everything
    else black - the same "embed a patch, convert once" journey test_goplus.py's `_frame`
    uses. Any ROI left None stays black (reads as neither teal nor card)."""
    w, h = size
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    for rect, colour in (
        (C.rois.exit_dialog_surround_top, top),
        (C.rois.exit_dialog_surround_bottom, bottom),
        (C.rois.exit_dialog_card, card),
    ):
        if colour is None:
            continue
        shape, (px0, py0) = _roi_pixel_shape(rect, size)
        ph, pw = shape
        hsv[py0:py0 + ph, px0:px0 + pw] = _flat(shape, colour)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


#: Deep inside the measured teal-surround band (H 70-100, S>=60, V>=90).
_TEAL = (85, 130, 150)
#: Bright and low-saturation - "white dialog card" independent of hue.
_CARD = (0, 10, 240)
#: Well outside the teal band (pure red) - stands in for "whatever else is on screen".
_NOT_TEAL = (0, 200, 200)


def _positive_frame():
    return _frame(top=_TEAL, bottom=_TEAL, card=_CARD)


# ------------------------------------------------------------------ perception: the signal alone

def test_the_positive_construction_fires():
    sig = P.exit_dialog_signal(_positive_frame(), C)
    assert sig.value is True
    assert sig.detail["teal"] >= C.thresholds.exit_dialog_teal
    assert sig.detail["card_white"] >= C.thresholds.exit_dialog_card


def test_a_flat_black_frame_never_fires():
    """Stands in for Overworld/menus with nothing resembling this dialog anywhere."""
    sig = P.exit_dialog_signal(_frame(), C)
    assert sig.value is False


def test_teal_surround_without_a_card_does_not_fire():
    """A flat teal/green screen with no bright dialog card - both bars are required."""
    sig = P.exit_dialog_signal(_frame(top=_TEAL, bottom=_TEAL), C)
    assert sig.detail["teal"] >= C.thresholds.exit_dialog_teal
    assert sig.value is False


def test_a_bright_card_without_teal_surround_does_not_fire():
    """Stands in for the measured PokemonEncounter false-positive risk: max card 1.00,
    but max teal only 0.44 - well under the 0.55 bar. AND, not OR, is what separates it."""
    sig = P.exit_dialog_signal(_frame(top=_NOT_TEAL, bottom=_NOT_TEAL, card=_CARD), C)
    assert sig.detail["card_white"] >= C.thresholds.exit_dialog_card
    assert sig.detail["teal"] < C.thresholds.exit_dialog_teal
    assert sig.value is False


def test_one_clean_surround_band_is_not_enough():
    """Both the top AND bottom bands must read teal - a card that spills into (or is
    mis-placed relative to) either one must be able to sink that band's own score. Proven
    here by giving the two bands genuinely different colours: the MINIMUM, not the mean,
    decides, so one dirty band cannot be smoothed away by a clean partner."""
    sig = P.exit_dialog_signal(_frame(top=_TEAL, bottom=_NOT_TEAL, card=_CARD), C)
    assert sig.value is False
    assert sig.detail["teal"] < C.thresholds.exit_dialog_teal


def test_missing_rois_read_as_the_lowest_possible_score_not_a_crash():
    """A degenerate zero-area frame (crop.size == 0 for every ROI) must return a Signal
    that reads as absent, mirroring every other optical test in perception.py - never a
    raised exception, and never a false positive."""
    sig = P.exit_dialog_signal(np.zeros((0, 0, 3), dtype=np.uint8), C)
    assert sig.value is False


def test_thresholds_match_the_rule_given_in_the_task_brief():
    """`teal >= 0.55 AND card_white >= 0.35`, taken from the brief verbatim - not
    re-derived, and not implied to be a well-sampled threshold: only two real positive
    samples exist (see config.Thresholds' own comment)."""
    assert C.thresholds.exit_dialog_teal == pytest.approx(0.55)
    assert C.thresholds.exit_dialog_card == pytest.approx(0.35)


def test_the_hue_band_matches_the_measurement():
    """H 70-100, S>=60, V>=90 - the one number in this whole feature that the task brief
    actually reports as measured, rather than chosen to fit a qualitative description."""
    lo, hi = P.EXIT_TEAL_LO, P.EXIT_TEAL_HI
    assert (lo[0], hi[0]) == (70, 100)
    assert lo[1] == 60 and lo[2] == 90


# ------------------------------------------------------------------ fsm: the rocket_screen veto

def test_exit_dialog_vetoes_the_rocket_screen_classification():
    """The actual defect: this dialog classifies as Rocket. Without the veto,
    `desired_state` would route straight into BotState.ROCKET, whose fixed
    dialogue-advance tap (0.50, 0.62) lands on empty space below this dialog's CANCEL
    label and stalls for the whole 150s ROCKET timeout."""
    o = obs(on_map=False, screen="Rocket", conf=0.99, exit_dialog=True)
    assert fsm.rocket_screen(o, C) is False


def test_a_real_rocket_screen_is_unaffected():
    o = obs(on_map=False, screen="Rocket", conf=0.99, exit_dialog=False)
    assert fsm.rocket_screen(o, C) is True


def test_desired_state_never_routes_to_rocket_while_the_exit_dialog_is_up():
    c = fsm.Context(cfg=C, state=BotState.RECOVERING, now=10.0, state_since=0.0)
    o = obs(on_map=False, screen="Rocket", conf=0.99, exit_dialog=True)
    assert fsm.desired_state(o, c) is not BotState.ROCKET


# ------------------------------------------------------------------ fsm: the interrupt

def ictx(state=BotState.RECOVERING, **kw):
    c = fsm.Context(cfg=C, state=state, now=10.0, state_since=0.0)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_the_interrupt_fires_back_with_no_coordinate():
    c = ictx()
    effects = fsm.interrupts(obs(exit_dialog=True), c)
    backs = [e for e in effects if isinstance(e, Back)]
    assert len(backs) == 1
    # Back carries no x/y at all - structurally, not just by convention - which is the
    # entire reason it is preferred over any coordinate-based response here.
    assert not hasattr(backs[0], "x") and not hasattr(backs[0], "y")
    assert any(isinstance(e, Note) for e in effects)


def test_the_interrupt_is_what_fsm_step_returns_immediately():
    """`interrupts()` runs before `desired_state`/any handler's own `step` - the exit
    dialog must be handled at the top of `fsm.step`, not by whichever state happens to be
    current, which is what "protects the whole machine" means in the task brief."""
    for state in (BotState.SCANNING, BotState.ROCKET, BotState.RECOVERING, BotState.POPUP):
        c = ictx(state=state)
        effects = fsm.step(obs(on_map=False, screen="Rocket", conf=0.99, exit_dialog=True), c)
        assert any(isinstance(e, Back) for e in effects)
        assert not any(isinstance(e, Tap) for e in effects)


def test_the_interrupt_fires_even_while_switching():
    """Deliberately not gated on state: a false positive here costs one harmless extra
    BACK press (see the module docstring), and SWITCHING's own `_settle` already presses
    BACK unprompted for exactly the same "whatever is on screen must go" reason."""
    c = ictx(state=BotState.SWITCHING, switch_phase="settle", switch_target="TrainerTwo")
    effects = fsm.step(obs(on_map=False, exit_dialog=True), c)
    assert any(isinstance(e, Back) for e in effects)


def test_the_interrupt_takes_priority_over_keyboard_and_claim():
    """All three could in principle co-occur on one frame; exit_dialog is checked first
    because it is the safety-critical one - see perception.exit_dialog_signal's
    docstring for why a coordinate-based response here is the actual danger."""
    from pogobot.observation import Tristate
    c = ictx()
    o = obs(exit_dialog=True, keyboard=Tristate.TRUE, claim=True, pill_xy=(0.5, 0.9))
    effects = fsm.interrupts(o, c)
    assert any(isinstance(e, Back) for e in effects)
    assert not any(isinstance(e, Tap) for e in effects)


def test_the_interrupt_is_paced_not_spammed_every_tick():
    c = ictx()
    first = fsm.interrupts(obs(exit_dialog=True), c)
    assert any(isinstance(e, Back) for e in first)
    c.last_action["back"] = c.now          # Runner.apply's bookkeeping, mirrored here
    c.now += 0.1                            # well inside exit_dialog_back (1.0s default)
    again = fsm.interrupts(obs(exit_dialog=True), c)
    assert again == []


def test_the_interrupt_fires_again_once_its_own_pace_clears():
    c = ictx()
    c.last_action["back"] = c.now
    c.now += C.timings.exit_dialog_back + 0.1
    effects = fsm.interrupts(obs(exit_dialog=True), c)
    assert any(isinstance(e, Back) for e in effects)


def test_no_interrupt_fires_without_the_signal():
    c = ictx()
    assert fsm.interrupts(obs(), c) == []


# ------------------------------------------------------------------ integration: measured live

def test_one_back_returns_straight_to_the_map_as_measured_live():
    """The exact sequence reported as verified by hand: RECOVERING sees the dialog,
    presses BACK, and the very next frame - the dialog gone, the map confirmed - moves on
    normally with no ROCKET detour and no second BACK needed."""
    c = ictx(state=BotState.RECOVERING)
    stuck = obs(on_map=False, screen="Rocket", conf=0.9, exit_dialog=True)
    effects = fsm.step(stuck, c)
    assert any(isinstance(e, Back) for e in effects)
    assert not any(isinstance(e, Transition) for e in effects)

    c.last_action["back"] = c.now
    c.now += 1.0
    recovered = obs(on_map=True)
    effects = fsm.step(recovered, c)
    tr = [e for e in effects if isinstance(e, Transition)]
    assert tr and tr[0].to is BotState.SCANNING
