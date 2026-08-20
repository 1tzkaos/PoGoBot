"""The post-switch zoom-out: the one-finger tap+drag gesture `Switching` fires once a
switch is confirmed and the map is back (see `fsm.Switching._zoom` and the `DoubleTapDrag`
branch of `actions.Actuator.render`).

Multi-touch is unavailable on the device (sendevent blocked by SELinux, `input
motionevent` single-pointer, two concurrent `input swipe`s doing nothing), but a tap
immediately followed by a press-and-drag from the same point IS a single pointer, and
Android reads it as its one-finger zoom. Measured dragging UP from screen centre on a
1080x2340 device: a map-region diff of 43.1 after the first application and 17.6 after
the second, against an 11.1 no-input baseline - so two repeats, the second near whatever
ceiling the game's zoom-out has.
"""
import pytest

from pogobot import fsm
from pogobot.accounts import FakeTreeReader
from pogobot.actions import Actuator
from pogobot.config import Config
from pogobot.effects import BotState, DoubleTapDrag, IntentOutcome, SetFlag, Tap, Transition
from tests.factories import obs
from tests.test_switch_runner import (
    ROSTER,
    _fail_a_switch,
    _quota_switcher,
    make_runner,
)
from tests.test_switching import budget, ctx, panel


# ------------------------------------------------------------------ config constants

def test_the_zoom_constants_match_the_measurement():
    z = Config().zoom
    assert z.center_x == pytest.approx(0.5) and z.center_y == pytest.approx(0.5)
    assert z.drag_frac == pytest.approx(370.0 / 2340.0)   # 370px UP on a 2340px-tall screen
    assert z.duration_ms == 400
    assert z.repeats == 2                                  # only this many were measured


# ------------------------------------------------------------------ fsm: when it fires

def test_zoom_does_nothing_until_the_map_is_confirmed_back():
    """`_verify`'s close tap has only just been queued when "zoom" is first entered -
    acting on this same tick would drive the gesture against a view that still shows the
    account panel, not the map underneath it."""
    c = ctx(phase="zoom")
    # obs() alone defaults to screen="Overworld"@0.99, which reads as on_map regardless of
    # this flag (see tests/factories.py) - "off the map" needs a different screen guess.
    assert fsm.step(obs(on_map=False, screen="Menu", conf=0.99), c) == []


def test_zoom_fires_the_first_gesture_once_the_map_is_back():
    c = ctx(phase="zoom")
    effects = fsm.step(obs(on_map=True), c)
    drags = [e for e in effects if isinstance(e, DoubleTapDrag)]
    assert len(drags) == 1
    z = c.cfg.zoom
    d = drags[0]
    assert d.x1 == pytest.approx(z.center_x) and d.y1 == pytest.approx(z.center_y)
    assert d.x2 == pytest.approx(z.center_x)               # straight up, not sideways
    assert d.y2 == pytest.approx(z.center_y - z.drag_frac)  # UP zooms OUT (measured)
    assert d.duration_ms == z.duration_ms
    # No self-reported SetFlag for the rep count: `Runner.apply` owns it, and only when
    # this same gesture is actually accepted by the actuator (see test_switch_zoom_reps_*
    # below) - a pure handler cannot know that in advance.
    assert not any(isinstance(e, SetFlag) and e.name == "switch_zoom_reps" for e in effects)
    assert not any(isinstance(e, Transition) for e in effects)


def test_zoom_repeats_exactly_the_configured_number_of_times():
    """Two applications were measured; this proves the handler stops there rather than
    at whatever number `repeats` happens to hold. `switch_zoom_reps` is advanced here the
    way `Runner.apply` now does it - only for a gesture that actually fired - not via a
    SetFlag the handler no longer emits."""
    c = ctx(phase="zoom")
    reps = c.cfg.zoom.repeats
    fired = 0
    for _ in range(reps + 3):          # extra ticks to prove it does NOT overshoot either
        effects = fsm.step(obs(on_map=True), c)
        for e in effects:
            if isinstance(e, DoubleTapDrag):
                fired += 1
                c.switch_zoom_reps += 1
    assert fired == reps


def test_zoom_confirms_only_after_every_repeat_has_fired():
    c = ctx(phase="zoom")
    reps = c.cfg.zoom.repeats
    transitions = []
    for _ in range(reps + 1):
        effects = fsm.step(obs(on_map=True), c)
        for e in effects:
            if isinstance(e, DoubleTapDrag):
                c.switch_zoom_reps += 1
            elif isinstance(e, Transition):
                transitions.append(e)
    assert len(transitions) == 1
    tr = transitions[0]
    assert tr.to is BotState.SCANNING and tr.outcome is IntentOutcome.CONFIRMED
    assert tr.reason.endswith(c.switch_target)


def test_zoom_gesture_coordinates_never_land_on_a_delete_button():
    """The gesture is a screen-centre drag nowhere near an account row, but it is still a
    coordinate this handler emits - this is the assertion that actually matters."""
    c = ctx(phase="zoom")
    deletes = {r.delete_norm for r in c.accounts.rows if r.delete_norm}
    effects = fsm.step(obs(on_map=True), c)
    drags = [e for e in effects if isinstance(e, DoubleTapDrag)]
    assert drags, "an empty gesture list proves nothing here"
    for d in drags:
        assert (d.x1, d.y1) not in deletes
        assert (d.x2, d.y2) not in deletes


# ------------------------------------------------------------------ fsm: when it must NOT fire

def test_zoom_phase_is_never_entered_from_a_mismatch():
    c = ctx(phase="verify", accounts=panel(active="TrainerOne"))
    effects = fsm.step(obs(on_map=True), c)
    assert not any(isinstance(e, SetFlag) and e.value == "zoom" for e in effects)
    assert not any(isinstance(e, DoubleTapDrag) for e in effects)


def test_timeout_from_the_zoom_phase_never_fires_the_gesture():
    """A switch that expires mid-zoom - or in any other phase - ends in RECOVERING with
    no gesture ever issued, exactly like every other exit through on_timeout."""
    c = ctx(phase="zoom", cfg=budget(17.0))
    c.now = c.state_since + 18.0
    effects = fsm.step(obs(on_map=True), c)
    assert not any(isinstance(e, DoubleTapDrag) for e in effects)
    tr = [e for e in effects if isinstance(e, Transition)][0]
    assert tr.to is BotState.RECOVERING and tr.outcome is IntentOutcome.EXPIRED


# ------------------------------------------------------------------ actuator

def _act(dry_run=True, **kw):
    return Actuator(screen_wh=(1080, 2340), dry_run=dry_run, **kw)


def test_the_gesture_renders_as_one_adb_invocation():
    """Both `input tap` and `input swipe` must reach the device in the SAME `adb shell`
    call, or the second touch misses the double-tap window `input` needs to read them as
    one continuous gesture rather than two independent touches."""
    a = _act()
    cmd = a.render(DoubleTapDrag(0.5, 0.5, 0.5, 0.342, "zoom out", duration_ms=400))
    assert cmd.argv[:2] == ("adb", "shell")
    assert len(cmd.argv) == 3                # ONE shell argument, not two invocations
    shell_arg = cmd.argv[2]
    assert "input tap" in shell_arg and "input swipe" in shell_arg
    assert shell_arg.index("input tap") < shell_arg.index("input swipe")
    assert ";" in shell_arg


def test_the_gesture_coordinates_convert_like_every_other_effect():
    a = _act()
    cmd = a.render(DoubleTapDrag(0.5, 0.5, 0.5, 0.5 - 370.0 / 2340.0, "zoom",
                                 duration_ms=400))
    x1, y1 = a.to_device(0.5, 0.5)
    x2, y2 = a.to_device(0.5, 0.5 - 370.0 / 2340.0)
    assert f"{x1} {y1}" in cmd.argv[2]
    assert f"{x2} {y2}" in cmd.argv[2]


def test_dry_run_suppresses_the_gesture_like_every_other_actuation():
    a = _act(dry_run=True)
    accepted = a.apply(DoubleTapDrag(0.5, 0.5, 0.5, 0.34, "zoom"), now=0.0)
    assert accepted is True                  # dry-run still advances FSM pacing
    stats = a.stats()
    assert stats["suppressed_dry_run"] == 1
    assert stats["sent"] == 0


def test_the_gesture_has_its_own_rate_limit_budget():
    """Proof "zoom" is tracked independently: exhausting the "tap" budget must not block
    a DoubleTapDrag issued moments later."""
    a = _act(dry_run=True, intervals={"tap": 5.0, "zoom": 0.25})
    assert a.apply(Tap(0.5, 0.5, "unrelated tap", budget="tap"), now=0.0) is True
    assert a.apply(DoubleTapDrag(0.5, 0.5, 0.5, 0.34, "zoom", budget="zoom"),
                   now=0.5) is True
    assert a.stats()["by_budget"] == {"tap": 1, "zoom": 1}


# ------------------------------------------------------------------ end to end (real Runner)

def _drive(r, o):
    r._refresh_accounts(r.ctx.now)
    r.apply(fsm.step(o, r.ctx), o)


def test_a_confirmed_switch_fires_the_zoom_gesture_then_rolls_the_session_over(tmp_path):
    """Driven through the real Runner and real FSM, not constructed by hand: the target is
    already logged in (the "already active" shortcut in `Switching.step`), so no login
    grace period needs simulating - open -> settle -> verify -> zoom x2 -> confirmed."""
    r = make_runner(stats_path=tmp_path / "sessions.jsonl",
                    tree_reader=FakeTreeReader([panel(active="TrainerTwo")]),
                    roster=ROSTER)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.now = 1_000.0
    r._begin_switch("TrainerTwo")
    r._accounts_read_at = 0.0
    o = obs(on_map=True)
    for _ in range(20):
        r.ctx.now += 3.0            # clears both the tree-refresh throttle and ui_settle
        _drive(r, o)
        if r.ctx.state is BotState.SCANNING:
            break
    assert r.ctx.state is BotState.SCANNING
    assert r.stats.account == "TrainerTwo"          # _on_switch_confirmed actually ran
    drags = [e for e in r.actuator.applied if isinstance(e, DoubleTapDrag)]
    assert len(drags) == r.cfg.zoom.repeats


class _FlakyAct:
    """Stands in for `Actuator` refusing a live command - rate-limit (`_allowed()`) or
    queue backpressure (`queue.Full`) - which returns False without raising and without
    sending anything to the device. The test controls acceptance per call, one bool per
    `apply()`."""

    def __init__(self, accepts):
        self._accepts = iter(accepts)
        self.applied = []

    def apply(self, effect, now=None):
        ok = next(self._accepts)
        if ok:
            self.applied.append(effect)
        return ok

    def healthy(self):
        return True

    def stats(self):
        return {"sent": len(self.applied)}

    def close(self):
        pass


def test_switch_zoom_reps_does_not_advance_on_a_rejected_gesture(tmp_path):
    """A rejected `DoubleTapDrag` (rate-limit / backpressure) must not move
    `switch_zoom_reps` - the count has to reflect what was actually sent, or `_zoom` can
    confirm the switch having applied fewer than `repeats` real zoom-outs with nothing
    anywhere recording that anything was skipped."""
    r = make_runner(stats_path=tmp_path / "sessions.jsonl", roster=ROSTER)
    r.actuator = _FlakyAct([False, True])
    r.ctx = ctx(phase="zoom")
    o = obs(on_map=True)

    r.apply(fsm.step(o, r.ctx), o)
    assert r.ctx.switch_zoom_reps == 0           # rejected: nothing was actually sent
    assert r.actuator.applied == []

    r.apply(fsm.step(o, r.ctx), o)
    assert r.ctx.switch_zoom_reps == 1            # accepted: now it advances
    assert len(r.actuator.applied) == 1


def test_a_failed_switch_never_fires_the_zoom_gesture():
    r = _quota_switcher()
    assert _fail_a_switch(r, r.ctx.now + 1.0, tap_login=True)
    assert not any(isinstance(e, DoubleTapDrag) for e in r.actuator.applied)
