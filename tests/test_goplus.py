"""Virtual Go Plus: re-enabling the toggle PGSharp turns off on every account switch.

Two layers, tested separately:

  * `perception.goplus_signal` is a pure function of the ROI alone - it has no notion of
    "on the map" and cannot have one (see its docstring). The synthetic frames below are
    built at the exact measured HSV values from the task brief, round-tripped through
    real cv2 HSV<->BGR conversion rather than injected as a mock, so what is under test
    is the real classifier reading real pixels.
  * `fsm.Switching._goplus` is where "on the map" is actually enforced, and where the
    tap-or-not decision is made. Its tests drive `fsm.step` the same way every other
    Switching-phase test in this suite does (see tests/test_switching.py,
    tests/test_switch_zoom.py).
"""
import json

import cv2
import numpy as np
import pytest

from pogobot import fsm
from pogobot.config import DEFAULT, GoPlusToggle
from pogobot.effects import BotState, IntentOutcome, SetFlag, Tap, Transition
from pogobot.observation import Tristate
from pogobot import perception as P
from tests.factories import obs
from tests.test_switch_runner import _fail_a_switch, _quota_switcher, make_runner
from tests.test_switching import budget, ctx as switching_ctx, panel

C = DEFAULT

# ------------------------------------------------------------------ synthetic frames

#: Saturated pure green, deep inside the H 40-90 / S>=80 / V>=80 band this signal looks
#: for - used as the "green centre" ingredient of an ON patch.
_GREEN_HSV = (60, 255, 255)


def _roi_pixel_shape(cfg=C, size=(1080, 2340)):
    """The exact pixel rectangle `perception.crop` will cut for `cfg.rois.goplus_toggle`
    at this frame size - computed the same way `crop` does, so a patch built to this
    shape lands back in the classifier's hands unchanged."""
    w, h = size
    x0, y0, x1, y1 = cfg.rois.goplus_toggle
    px0, py0 = int(w * x0), int(h * y0)
    px1, py1 = int(w * x1), int(h * y1)
    return (py1 - py0, px1 - px0), (px0, py0)


def _mixed_patch(mean_v, mean_s, green_frac, shape, ring_h=170):
    """An HSV patch whose OVERALL mean V/S and green fraction hit the given targets
    exactly (up to uint8 rounding): `green_frac` of the pixels are the saturated green
    above, the rest are a uniform "ring" colour solved algebraically so the mix averages
    to `mean_v`/`mean_s`. `ring_h` sits outside the green band regardless of its S/V, so
    it never itself counts as green.
    """
    ph, pw = shape
    n = ph * pw
    n_green = int(round(green_frac * n))
    frac = n_green / n if n else 0.0
    gh, gs, gv = _GREEN_HSV
    if frac < 1.0:
        ring_v = (mean_v - frac * gv) / (1 - frac)
        ring_s = (mean_s - frac * gs) / (1 - frac)
    else:
        ring_v, ring_s = mean_v, mean_s
    ring_v = min(255, max(0, ring_v))
    ring_s = min(255, max(0, ring_s))
    flat = np.empty((n, 3), dtype=np.uint8)
    flat[:] = (ring_h, int(round(ring_s)), int(round(ring_v)))
    flat[:n_green] = (gh, gs, gv)
    return flat.reshape(ph, pw, 3)


def _frame(patch_hsv, cfg=C, size=(1080, 2340)):
    """Embed an HSV patch at the toggle's ROI inside an otherwise-black full frame, and
    hand back real BGR pixels - the same journey a captured frame makes."""
    w, h = size
    (ph, pw), (px0, py0) = _roi_pixel_shape(cfg, size)
    assert patch_hsv.shape[:2] == (ph, pw)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[py0:py0 + ph, px0:px0 + pw] = patch_hsv
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _off_frame():
    shape, _ = _roi_pixel_shape()
    return _frame(_mixed_patch(160.6, 121.8, 0.0, shape))


def _on_frame():
    shape, _ = _roi_pixel_shape()
    return _frame(_mixed_patch(253.3, 166.2, 0.258, shape))


def _uniform_frame(h, s, v):
    """A flat, single-colour ROI - stands in for an unmeasured third appearance (no
    Virtual Go Plus at all, a re-skinned icon, whatever)."""
    shape, _ = _roi_pixel_shape()
    flat = np.empty(shape + (3,), dtype=np.uint8)
    flat[:] = (h, s, v)
    return _frame(flat)


# ------------------------------------------------------------------ perception: the classifier alone

def test_off_frame_classifies_as_off():
    assert P.goplus_signal(_off_frame(), C) is Tristate.FALSE


def test_on_frame_classifies_as_on():
    assert P.goplus_signal(_on_frame(), C) is Tristate.TRUE


def test_absent_reads_unknown_not_forced_into_either_state():
    """Black (nothing there) and white are both plausible 'something else is drawn
    here' appearances, and neither was measured - both must read UNKNOWN, not be pulled
    into OFF just because they are not bright and green enough to be ON."""
    assert P.goplus_signal(_uniform_frame(0, 0, 0), C) is Tristate.UNKNOWN
    assert P.goplus_signal(_uniform_frame(0, 0, 255), C) is Tristate.UNKNOWN


def test_a_value_between_the_two_measured_points_reads_unknown():
    """V=200 sits above OFF's measured 160.6 and below ON's measured 253.3 - the dead
    zone the margins deliberately leave, rather than snapping to whichever is closer."""
    assert P.goplus_signal(_uniform_frame(170, 100, 200), C) is Tristate.UNKNOWN


def test_the_false_on_trap_reads_on_from_the_roi_alone():
    """Measured on a PokeStop reward screen: the same ROI, on a green background, reads
    100% green - and by every number in this test, that genuinely does look like ON.
    `goplus_signal` has no way to know it is not on the map (see its docstring); that is
    the FSM's job, tested below (`test_the_trap_frame_is_not_acted_on_off_the_map`)."""
    trap = _uniform_frame(60, 200, 255)         # solid green, well past every ON bound
    assert P.goplus_signal(trap, C) is Tristate.TRUE


def test_thresholds_sit_between_the_two_measured_points():
    """Regression guard for the measured numbers themselves and the margins around
    them - if either drifts across the other's actual value, the classifier stops
    separating what was measured."""
    t = C.thresholds
    assert t.goplus_off_v_min < 160.6 < t.goplus_off_v_max < t.goplus_on_v_min < 253.3
    assert t.goplus_off_s_min < 121.8 < t.goplus_off_s_max < t.goplus_on_s_min < 166.2
    assert 0.0 <= t.goplus_off_green_max < t.goplus_on_green_min < 0.258


def test_goplus_toggle_constants_match_the_measurement():
    g = GoPlusToggle()
    assert g.tap_x == pytest.approx(0.915) and g.tap_y == pytest.approx(0.212)
    x0, y0, x1, y1 = C.rois.goplus_toggle
    assert x0 == pytest.approx(0.898) and y0 == pytest.approx(0.200)
    assert x1 == pytest.approx(0.932) and y1 == pytest.approx(0.225)


# ------------------------------------------------------------------ fsm: the "goplus" phase

def gctx(**kw):
    return switching_ctx(phase="goplus", **kw)


def test_off_and_on_map_presses_the_toggle():
    c = gctx()
    effects = fsm.step(obs(on_map=True, goplus=Tristate.FALSE), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1
    t = taps[0]
    assert t.x == pytest.approx(C.goplus.tap_x) and t.y == pytest.approx(C.goplus.tap_y)
    assert t.budget == "goplus"
    assert not any(isinstance(e, Transition) for e in effects)


def test_on_does_not_press_and_confirms_the_switch():
    c = gctx(target="TrainerTwo")
    effects = fsm.step(obs(on_map=True, goplus=Tristate.TRUE), c)
    assert not [e for e in effects if isinstance(e, Tap)]
    tr = [e for e in effects if isinstance(e, Transition)]
    assert len(tr) == 1
    assert tr[0].to is BotState.SCANNING and tr[0].outcome is IntentOutcome.CONFIRMED
    assert tr[0].reason.endswith("TrainerTwo")


def test_absent_does_not_press_and_still_confirms_the_switch():
    """No Virtual Go Plus at all must not block anything - the whole feature is a no-op
    for an account that never had the toggle."""
    c = gctx(target="TrainerTwo")
    effects = fsm.step(obs(on_map=True, goplus=Tristate.UNKNOWN), c)
    assert not [e for e in effects if isinstance(e, Tap)]
    tr = [e for e in effects if isinstance(e, Transition)]
    assert tr and tr[0].outcome is IntentOutcome.CONFIRMED


def test_does_nothing_until_the_map_is_confirmed_back():
    """Same reasoning as the zoom phase: acting before the map is actually back would be
    driving this off whatever was on screen a moment ago, not the toggle."""
    c = gctx()
    assert fsm.step(obs(on_map=False, screen="Menu", conf=0.99, goplus=Tristate.FALSE), c) == []


def test_the_trap_frame_is_not_acted_on_off_the_map():
    """The other half of the false-ON trap test: even an Observation whose `goplus`
    field genuinely reads TRUE (built from the real green-background trap frame via
    `goplus_signal`, not asserted by hand) must not be confirmed on, tapped, or trusted
    in any way while `on_map` is False."""
    trap_on = P.goplus_signal(_uniform_frame(60, 200, 255), C)
    assert trap_on is Tristate.TRUE, "the trap must genuinely read ON from the ROI alone"
    c = gctx()
    effects = fsm.step(obs(on_map=False, screen="Poi", conf=0.99, goplus=trap_on), c)
    assert effects == []


def test_a_second_press_waits_for_the_first_to_take():
    """Pressed once; re-checked only after `press_wait`, not on the very next tick -
    the game shows 'connecting...' for up to 4.5s measured."""
    c = gctx()
    first = fsm.step(obs(on_map=True, goplus=Tristate.FALSE), c)
    assert [e for e in first if isinstance(e, Tap)]
    c.switch_goplus_attempts = 1
    c.last_action["goplus"] = c.now          # Runner.apply's bookkeeping, mirrored here
    c.now += 1.0                              # well inside press_wait (6.0s default)
    again = fsm.step(obs(on_map=True, goplus=Tristate.FALSE), c)
    assert not [e for e in again if isinstance(e, Tap)]
    assert again == []


def test_attempts_are_bounded_and_never_block_confirmation():
    """Still OFF after `max_attempts` presses: the switch confirms anyway rather than
    sitting there forever - it must never block a switch from confirming."""
    c = gctx(target="TrainerTwo")
    c.switch_goplus_attempts = C.goplus.max_attempts
    effects = fsm.step(obs(on_map=True, goplus=Tristate.FALSE), c)
    assert not [e for e in effects if isinstance(e, Tap)]
    tr = [e for e in effects if isinstance(e, Transition)]
    assert tr and tr[0].outcome is IntentOutcome.CONFIRMED


def test_zoom_hands_off_to_goplus_once_every_repeat_has_fired():
    """The phase this work slots into, per the brief: after the zoom step, before the
    switch is confirmed."""
    c = switching_ctx(phase="zoom")
    c.switch_zoom_reps = c.cfg.zoom.repeats
    effects = fsm.step(obs(on_map=True, goplus=Tristate.FALSE), c)
    assert any(isinstance(e, SetFlag) and e.name == "switch_phase" and e.value == "goplus"
               for e in effects)
    # The hand-off runs _goplus in the SAME tick - still OFF, so a press is queued
    # immediately rather than losing a tick to the phase change.
    assert any(isinstance(e, Tap) and e.budget == "goplus" for e in effects)
    assert not any(isinstance(e, Transition) for e in effects)


def test_zoom_hand_off_confirms_immediately_when_goplus_is_absent():
    """The common case in this whole test suite: `obs()` defaults `goplus` to UNKNOWN,
    so every pre-existing zoom test that never mentions Virtual Go Plus at all still
    confirms on the same tick zoom's repeats complete - see test_switch_zoom.py."""
    c = switching_ctx(phase="zoom")
    c.switch_zoom_reps = c.cfg.zoom.repeats
    effects = fsm.step(obs(on_map=True), c)
    tr = [e for e in effects if isinstance(e, Transition)]
    assert len(tr) == 1
    assert tr[0].to is BotState.SCANNING and tr[0].outcome is IntentOutcome.CONFIRMED


# ------------------------------------------------------------------ never on a failed/timed-out switch

def test_timeout_from_the_goplus_phase_never_presses_the_toggle():
    c = gctx(cfg=budget(17.0))
    c.now = c.state_since + 18.0
    effects = fsm.step(obs(on_map=True, goplus=Tristate.FALSE), c)
    assert not any(isinstance(e, Tap) and getattr(e, "budget", None) == "goplus"
                   for e in effects)
    tr = [e for e in effects if isinstance(e, Transition)][0]
    assert tr.to is BotState.RECOVERING and tr.outcome is IntentOutcome.EXPIRED


def test_a_mismatch_in_verify_never_reaches_the_goplus_phase():
    """`_goplus` is reachable only through `_zoom`'s completion, itself reachable only
    through a `_verify` match - so a switch that never confirms can never even attempt
    the toggle, whatever `obs.goplus` says."""
    c = switching_ctx(phase="verify", accounts=panel(active="TrainerOne"))
    effects = fsm.step(obs(on_map=True, goplus=Tristate.FALSE), c)
    assert not any(isinstance(e, Tap) and getattr(e, "budget", None) == "goplus"
                   for e in effects)
    assert not any(isinstance(e, SetFlag) and e.value == "goplus" for e in effects)


def test_a_failed_switch_never_presses_the_goplus_toggle():
    """Driven the way tests/test_switch_zoom.py drives the equivalent zoom-gesture
    guard: a real failed switch, through the real Runner and FSM, produces no `goplus`
    tap anywhere in its applied effects."""
    r = _quota_switcher()
    assert _fail_a_switch(r, r.ctx.now + 1.0, tap_login=True)
    assert not any(getattr(e, "budget", None) == "goplus" for e in r.actuator.applied)


# The delete-guard for this phase lives in tests/test_switching.py's
# test_no_tap_ever_lands_on_a_delete_button, extended to cover "goplus" alongside every
# other phase that can emit a tap - not duplicated here.


# ------------------------------------------------------------------ the trace

def test_the_trace_records_what_the_classifier_read(tmp_path):
    """Every threshold here was measured on two frames, and the band between them (V
    195-220) reads UNKNOWN by construction - deliberately, since "we do not know" must
    never become "so tap it". Whether real frames actually land in that band is a
    question only a live run can answer, and it can only answer it if the trace says
    which of the three readings each frame produced. All three have to appear: UNKNOWN
    is also what "there is no Virtual Go Plus here" looks like, so its absence would be
    as informative as its presence."""
    path = tmp_path / "trace.jsonl"
    r = make_runner(trace_path=path)
    for reading in (Tristate.TRUE, Tristate.FALSE, Tristate.UNKNOWN):
        r._write_trace(obs(goplus=reading), [])
    r.close()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["goplus"] for row in rows] == ["TRUE", "FALSE", "UNKNOWN"]
