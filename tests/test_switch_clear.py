"""The BACK storm: `Switching._settle` used to press BACK every `Timings.switch_clear`
(2.5s) for as long as the map stayed hidden, with nothing bounding it.

Measured live: the post-login news modal needs one or two presses, but a post-login
LOADING screen legitimately runs for minutes, and nothing told the two apart. The
actuator tally from that run - `by_budget: {'back': 100, ...}` - was about 90 BACKs
into the loading screen over four minutes, and the outcome was worse than a stall:
PGSharp's own account panel afterward showed NEITHER account with an asterisk, and the
account had to be recovered by hand.

See config.Timings.switch_clear_max for the bound and its full measurement, and
fsm.Switching._settle for the fix. `tests/test_switching.py` covers the phase-level
decision (single fsm.step calls) directly; this file covers the CADENCE, which a
single-step test cannot see at all - the same class of gap test_autowalk.py's own
module docstring describes, and the reason this drives the real Runner at a realistic
sub-second tick rather than jumping straight to the state timeout the way
`test_switch_runner._fail_a_switch` does.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pogobot import fsm
from pogobot.accounts import FakeTreeReader
from pogobot.config import Config
from pogobot.effects import Back, BotState
from tests.factories import obs
from tests.test_switch_runner import ROSTER, make_runner
from tests.test_switch_zoom import _FlakyAct
from tests.test_switching import ctx, panel


def _drive(r, o):
    r._refresh_accounts(r.ctx.now)
    r.apply(fsm.step(o, r.ctx), o)


#: The screen the switch never gets past: no map, no close button, no claim pill - the
#: shape a LOADING screen (or a modal we never learned to close) actually has.
LOADING = obs(on_map=False, screen="Menu", conf=0.99)


def _run_a_stuck_switch(dt: float, cfg: Config) -> list:
    """Begin a switch to TrainerTwo and tick the real Runner, at a fixed `dt`, through a
    map that never returns - the exact shape a silently-refused login or a stuck LOADING
    screen leaves behind. Returns every effect the actuator actually accepted."""
    r = make_runner(cfg, tree_reader=FakeTreeReader([panel(active="TrainerOne")]),
                    roster=ROSTER)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.now = 1_000.0
    r.ctx.last_map_ts = r.ctx.now
    r._begin_switch("TrainerTwo")
    r._accounts_read_at = 0.0
    assert r.ctx.state is BotState.SWITCHING

    t0 = r.ctx.now
    budget = cfg.timings.switch_timeout + 5.0
    while r.ctx.state is BotState.SWITCHING and r.ctx.now - t0 < budget:
        r.ctx.now = round(r.ctx.now + dt, 6)
        _drive(r, LOADING)
    assert r.ctx.state is BotState.RECOVERING, "the switch must end at its own timeout"
    return r.actuator.applied


def test_a_switch_whose_map_never_returns_issues_at_most_the_bounded_backs():
    """The fix, driven at 0.1s - the cadence the live loop actually ticks at (see
    test_autowalk.py's TICK_INTERVALS docstring for why a coarser tick cannot be trusted
    to reveal this class of bug: a test that only jumps straight to the state timeout,
    the way `test_switch_runner._fail_a_switch` does, applies `fsm.step` exactly once
    and would show a single Back either way, fixed or not)."""
    applied = _run_a_stuck_switch(0.1, Config())
    backs = [e for e in applied if isinstance(e, Back)]
    bound = Config().timings.switch_clear_max
    assert 0 < len(backs) <= bound, (
        f"{len(backs)} BACK presses issued; the storm this test exists to catch presses "
        f"far more than the {bound}-press bound")


def test_an_unbounded_clear_budget_reproduces_the_original_back_storm():
    """Red-green check: an effectively unlimited switch_clear_max reproduces the
    measured live shape - dozens of presses into a screen that never clears - proving
    the drive loop above actually exercises the real cadence rather than passing
    vacuously regardless of the bound."""
    unbounded = Config(timings=replace(Config().timings, switch_clear_max=1_000_000))
    applied = _run_a_stuck_switch(0.1, unbounded)
    backs = [e for e in applied if isinstance(e, Back)]
    # ~240s / 2.5s (switch_clear) ~= 96 presses measured live as "100 BACKs" total
    # (a couple attributed to other budgets) - loose bound, robust to the exact tick
    # boundary, but nowhere near the 5-press default bound above.
    assert len(backs) >= 80, (
        f"only {len(backs)} BACK presses with an effectively unbounded budget - the "
        f"drive loop is not exercising the real per-tick cadence")


def test_a_rejected_back_does_not_consume_the_clear_budget(tmp_path):
    """A press the actuator rejects (rate-limit, queue backpressure) must not advance
    `switch_clear_presses` - the same pattern `test_switch_zoom.py`'s
    `test_switch_zoom_reps_does_not_advance_on_a_rejected_gesture` proves for the zoom
    gesture, and for the same reason: `_settle` is pure and cannot know in advance
    whether its Back will actually reach the device."""
    r = make_runner(stats_path=tmp_path / "sessions.jsonl", roster=ROSTER)
    r.actuator = _FlakyAct([False, True])
    r.ctx = ctx(phase="settle", accounts=panel(active="TrainerTwo"))
    o = obs(on_map=False, screen="Menu", conf=0.99)

    r.apply(fsm.step(o, r.ctx), o)
    assert r.ctx.switch_clear_presses == 0        # rejected: nothing was actually sent
    assert r.actuator.applied == []

    r.apply(fsm.step(o, r.ctx), o)
    assert r.ctx.switch_clear_presses == 1         # accepted: now it advances
    assert len(r.actuator.applied) == 1


def test_a_located_close_button_is_still_tapped_and_never_bounded(tmp_path):
    """The other half of the fix: a located close button is targeted, not blind, and
    must keep firing every tick regardless of how many BLIND BACK presses have already
    been spent - driven through the real Runner, not just one fsm.step call."""
    cfg = Config(timings=replace(Config().timings, switch_clear_max=2))
    r = make_runner(cfg, tree_reader=FakeTreeReader([panel(active="TrainerOne")]),
                    roster=ROSTER)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.now = 1_000.0
    r.ctx.last_map_ts = r.ctx.now
    r._begin_switch("TrainerTwo")
    r._accounts_read_at = 0.0

    close_screen = obs(on_map=False, screen="Menu", conf=0.99, close_xy=(0.5, 0.9))
    t0 = r.ctx.now
    while r.ctx.now - t0 < 30.0:
        r.ctx.now = round(r.ctx.now + 0.1, 6)
        _drive(r, close_screen)

    from pogobot.effects import Tap
    close_taps = [e for e in r.actuator.applied
                 if isinstance(e, Tap) and e.budget == "close"]
    assert len(close_taps) > cfg.timings.switch_clear_max, (
        "the located close button must keep being tapped past the BACK bound, not stall")


@pytest.mark.parametrize("dt", [0.1, 1.0])
def test_the_bound_holds_and_the_switch_still_confirms_once_the_map_returns(dt):
    """Once the bound is spent, `_settle` must still confirm normally the instant the
    map actually comes back - the bound only removes the blind hammering, it must never
    itself cost a switch its confirmation."""
    cfg = Config(timings=replace(Config().timings, switch_clear_max=2))
    r = make_runner(cfg, tree_reader=FakeTreeReader([panel(active="TrainerTwo")]),
                    roster=ROSTER)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.now = 1_000.0
    r.ctx.last_map_ts = r.ctx.now
    r._begin_switch("TrainerTwo")
    r._accounts_read_at = 0.0

    # Stay off the map long enough to exhaust the bound (2 presses at 2.5s apart), then
    # let the map come back.
    t0 = r.ctx.now
    while r.ctx.now - t0 < cfg.timings.switch_clear * (cfg.timings.switch_clear_max + 2):
        r.ctx.now = round(r.ctx.now + dt, 6)
        _drive(r, LOADING)
    assert r.ctx.switch_clear_presses >= cfg.timings.switch_clear_max
    assert r.ctx.state is BotState.SWITCHING, "must not have timed out yet"

    on_map = obs(on_map=True)
    t1 = r.ctx.now
    while r.ctx.state is BotState.SWITCHING and r.ctx.now - t1 < 60.0:
        r.ctx.now = round(r.ctx.now + dt, 6)
        _drive(r, on_map)
    assert r.ctx.state is BotState.SCANNING
    assert r.stats.account == "TrainerTwo"
