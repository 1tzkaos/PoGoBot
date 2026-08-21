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
    """The original numbers were justified by a "map-region diff" - i.e. that the screen
    CHANGED - which a pan and a zoom IN also do. Re-measured against side-by-side captures:
    direction UP is right, but 370px moved the scale too little to tell from the map's own
    animation, while 1050px visibly shrank the trainer. See config.ZoomOut."""
    z = Config().zoom
    assert z.center_x == pytest.approx(0.5)
    assert z.drag_frac == pytest.approx(1050.0 / 2340.0)  # 1050px UP on a 2340px screen
    assert z.duration_ms == 700
    assert z.repeats == 2                                  # only this many were measured
    # The drag must still land on screen from the configured start.
    assert z.center_y - z.drag_frac > 0.0


def test_the_zoom_anchor_avoids_everything_the_detector_can_see():
    """The gesture's first touch decides what the game thinks it is. On a PokeStop it
    opens the stop and the drag belongs to that screen instead - measured by hand as
    `screen=Poi@0.83` after a run of centre-anchored drags."""
    from pogobot import fsm
    from tests.factories import det
    spots = ((0.50, 0.55), (0.35, 0.60), (0.65, 0.50), (0.50, 0.68))
    o = obs(detections=tuple(det(n, 0.8, x, y) for n, (x, y) in
                             zip(("pokestop", "pokemon", "gym", "pokestop"), spots)))
    ax, ay = fsm.zoom_anchor(o, Config())
    nearest = min(((ax - x) ** 2 + (ay - y) ** 2) ** 0.5 for x, y in spots)
    assert nearest > 0.12, f"anchor ({ax:.2f},{ay:.2f}) sits {nearest:.3f} from a detection"


def test_the_zoom_anchor_stays_on_playable_map():
    """Off the HUD columns, off PGSharp's own overlay down the left, and far enough down
    that the drag still ends on screen."""
    from pogobot import fsm
    from tests.factories import det
    for o in (obs(), obs(detections=tuple(det("pokemon", 0.8, 0.2 + 0.1 * i, 0.5)
                                          for i in range(7)))):
        ax, ay = fsm.zoom_anchor(o, Config())
        assert 0.15 <= ax <= 0.85
        assert ay - Config().zoom.drag_frac > 0.0
        assert ay <= 0.75


def test_the_zoom_gesture_starts_where_the_anchor_says():
    """The emitted DoubleTapDrag must use the chosen anchor, not the config centre."""
    from pogobot import fsm
    from tests.factories import det
    o = obs(on_map=True, detections=(det("pokestop", 0.9, 0.5, 0.60),))
    c = ctx(phase="zoom")
    drags = [e for e in fsm.step(o, c) if isinstance(e, DoubleTapDrag)]
    assert drags, "no zoom gesture emitted"
    ax, ay = fsm.zoom_anchor(o, Config())
    assert (drags[0].x1, drags[0].y1) == (ax, ay)
    assert drags[0].x2 == ax and drags[0].y2 == pytest.approx(ay - Config().zoom.drag_frac)


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
    # The start point is now CHOSEN to avoid whatever the detector can see, so it is the
    # anchor rather than the config centre - see fsm.zoom_anchor and config.ZoomOut.
    ax, ay = fsm.zoom_anchor(obs(on_map=True), c.cfg)
    assert d.x1 == pytest.approx(ax) and d.y1 == pytest.approx(ay)
    assert d.x2 == pytest.approx(ax)                       # straight up, not sideways
    assert d.y2 == pytest.approx(ay - z.drag_frac)          # UP zooms OUT (measured)
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


def test_zoom_hands_off_to_autowalk_only_after_every_repeat_has_fired():
    """`_zoom` no longer confirms the switch itself - it hands off to "goplus", which
    (obs() defaults `goplus` to absent) cascades straight on to "autowalk_open" in the
    same tick (see test_goplus.py for that chain, tests/test_autowalk.py for what
    "autowalk_open" itself does). What this proves still holds: the hand-off fires
    exactly once, and only once every zoom repeat has actually fired - not before, and
    not a second time."""
    c = ctx(phase="zoom")
    reps = c.cfg.zoom.repeats
    reached = []
    for _ in range(reps + 1):
        effects = fsm.step(obs(on_map=True), c)
        for e in effects:
            if isinstance(e, DoubleTapDrag):
                c.switch_zoom_reps += 1
            elif isinstance(e, SetFlag) and e.name == "switch_phase" and e.value == "autowalk_open":
                reached.append(e)
        assert not any(isinstance(e, Transition) for e in effects)
    assert len(reached) == 1


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


def test_zoom_closes_a_screen_its_own_tap_opened():
    """`zoom_anchor` avoids what the detector reports, but not what it misses - measured
    live, an anchored drag opened a gym the detector never named. During SWITCHING
    `desired_state` returns None, so POPUP cannot come to the rescue and this phase would
    wait out the whole 240s switch timeout behind that screen."""
    c = ctx(phase="zoom")
    off = obs(screen="Menu", conf=0.95, x_button=True, close_xy=(0.5, 0.885))
    taps = [e for e in fsm.step(off, c) if isinstance(e, Tap)]
    assert taps, "nothing was pressed to clear the screen the zoom opened"
    assert (taps[0].x, taps[0].y) == (0.5, 0.885)
    assert taps[0].budget == "close"
    assert not [e for e in fsm.step(off, c) if isinstance(e, DoubleTapDrag)], \
        "the gesture must not fire while the map is not visible"


def test_zoom_still_just_waits_when_no_close_button_is_located():
    """Nothing is invented when the tree/optics name no button: the phase waits exactly as
    it did before, and the switch's own timeout owns the outcome."""
    c = ctx(phase="zoom")
    off = obs(screen="Menu", conf=0.95, x_button=True, close_xy=None)
    assert fsm.step(off, c) == []
