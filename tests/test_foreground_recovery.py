"""A tap on a sponsored stop can throw the bot out of the game entirely.

Pokemon GO carries sponsored stops and gyms. Measured live, two seconds after the bot
entered ROCKET and tapped the fixed dialogue coordinate:

    ActivityTaskManager: START u0 {act=VIEW dat=https://www.mlb.com:443/...
                                   cmp=com.android.chrome/...}

The game keeps running; it is simply no longer what the screen shows. Every rung of the
recovery ladder is then aimed at the wrong window - BACK navigates the BROWSER - and no map
can appear while another app owns the display. That run spent 603 frames in ROCKET on a
cookie banner and died on the frame guard, because a near-static web page barely encodes
any frames either.
"""
from __future__ import annotations

import pytest

from pogobot import fsm
from pogobot.config import DEFAULT
from pogobot.effects import BotState, Back, ForegroundApp, RestartApp, Tap
from pogobot.observation import Tristate
from tests.factories import obs


def ctx(**kw):
    """RECOVERING entered a moment ago - `state_since` must be near `now`, or the handler
    is already past its 6s timeout and `on_timeout` answers before any rung runs."""
    c = fsm.Context(cfg=DEFAULT, state=BotState.RECOVERING, state_since=99.5, now=100.0)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


#: A genuinely off-map frame. `obs(on_map=False)` is not one: the factory defaults
#: screen="Overworld" at 0.99, which satisfies `Observation.on_map` on its own.
OFF = obs(screen="Menu", conf=0.97)


@pytest.mark.parametrize("state", [BotState.RECOVERING, BotState.ROCKET,
                                   BotState.SCANNING, BotState.ENCOUNTER,
                                   BotState.POPUP, BotState.SWITCHING])
def test_the_game_is_raised_from_any_state(state):
    """An interrupt, not a rung: measured twice, the bot sat in ROCKET on a browser for the
    whole 150s ROCKET timeout, so anything scoped to RECOVERING never ran."""
    c = ctx(app_foreground=Tristate.FALSE)
    c.state = state
    out = fsm.step(OFF, c)
    assert [e for e in out if isinstance(e, ForegroundApp)], f"not raised from {state}"
    assert not [e for e in out if isinstance(e, (Back, Tap))], \
        f"something was pressed from {state} while another app owned the screen"


def test_the_game_is_raised_before_anything_else_is_pressed():
    """Ahead of BACK, which would navigate the browser rather than the game."""
    out = fsm.step(OFF, ctx(app_foreground=Tristate.FALSE))
    assert [e for e in out if isinstance(e, ForegroundApp)], "the game was not raised"
    assert not [e for e in out if isinstance(e, (Back, Tap))], \
        "something was pressed while another app owned the screen"


def test_it_names_the_configured_app():
    out = fsm.step(OFF, ctx(app_foreground=Tristate.FALSE))
    fg = [e for e in out if isinstance(e, ForegroundApp)][0]
    assert fg.package == DEFAULT.app_package
    assert fg.activity == DEFAULT.app_activity


def test_unknown_is_not_the_same_as_no():
    """Until the runner has actually looked, this rung must say nothing and the ladder must
    behave exactly as it did before - otherwise every recovery on a healthy run would open
    with an `am start`."""
    out = fsm.step(OFF, ctx(app_foreground=Tristate.UNKNOWN))
    assert not [e for e in out if isinstance(e, ForegroundApp)]
    assert [e for e in out if isinstance(e, Back)], "the ordinary ladder should have run"


def test_a_game_already_in_front_is_left_alone():
    out = fsm.step(OFF, ctx(app_foreground=Tristate.TRUE))
    assert not [e for e in out if isinstance(e, ForegroundApp)]


def test_it_is_bounded_and_escalates_to_a_restart():
    """A game that will not come forward must not be asked for ever: `am start` being
    accepted and ignored is what a relaunch fixes and a repeat does not."""
    spent = ctx(app_foreground=Tristate.FALSE,
                foregrounds=DEFAULT.max_foreground_attempts,
                last_map_ts=0.0)
    out = fsm.step(OFF, spent)
    assert not [e for e in out if isinstance(e, ForegroundApp)]
    # ...and the ladder below is reachable again.
    timed_out = fsm.Context(cfg=DEFAULT, state=BotState.RECOVERING, state_since=0.0,
                            now=DEFAULT.timings.stuck_watchdog + 50.0)
    timed_out.app_foreground = Tristate.FALSE
    timed_out.foregrounds = DEFAULT.max_foreground_attempts
    timed_out.last_map_ts = 0.0
    assert [e for e in fsm.Recovering().on_timeout(OFF, timed_out)
            if isinstance(e, RestartApp)], "a restart should still be reachable"


def test_the_handler_writes_nothing_to_the_context():
    import copy
    c = ctx(app_foreground=Tristate.FALSE)
    before = copy.deepcopy(vars(c))
    fsm.step(OFF, c)
    assert vars(c) == before


def test_the_runner_asks_whenever_the_map_is_missing():
    """Scoping this to RECOVERING was wrong and cost a live run: the bot sat in ROCKET on a
    browser for the whole 150s ROCKET timeout and never reached RECOVERING, so the check
    never ran. The gate is "no map", which is true of every wedge worth a `dumpsys` and
    false on the overwhelming majority of frames."""
    import inspect
    from pogobot import runner as R
    src = inspect.getsource(R.Runner._refresh_foreground)
    assert "if on_map:" in src, "the gate should be the map, not a state"
    assert "BotState.RECOVERING" not in src
    assert "FOREGROUND_CHECK" in src


def test_the_run_loop_actually_asks():
    """The rung is useless unless something sets `app_foreground`, and only the Runner can.
    Asserted as the CALL rather than the name: the name also appears in the method's own
    definition, so a bare substring check passes with the call deleted (verified by
    mutation)."""
    import inspect
    from pogobot import runner as R
    src = inspect.getsource(R.Runner.run)
    assert "self._refresh_foreground(real, obs.on_map)" in src


def test_an_accepted_raise_is_counted_by_the_runner():
    """The bound lives on ctx and only the Runner may write it - and only when the actuator
    ACCEPTED the command, the same rule the zoom and restart counters follow."""
    from tests.test_switch_runner import make_runner
    r = make_runner()
    r.ctx.state = BotState.RECOVERING
    before = r.ctx.foregrounds
    r.apply([ForegroundApp(DEFAULT.app_package, DEFAULT.app_activity, "recover: raise")],
            OFF)
    assert r.ctx.foregrounds == before + 1
    assert r.ctx.app_foreground is Tristate.UNKNOWN, \
        "the old answer must not survive the command that invalidates it"
