"""One cause, three symptoms - and the silence that hid it.

The user ran the bot for four hours and reported that it "didn't zoom out, nor did it
continue autowalk nor did it turn on the Pokemon Go Plus", assuming an account switch had
happened. It had not. Measured in logs/trace.jsonl for that run: 115,210 frames and ZERO
SWITCHING ones. All three of those steps live inside the Switching handler, so no switch
means none of the three ever ran.

Why no switch was ever attempted: logs/sessions.jsonl's last record reads `ended 08-20
18:59:47 account=None uptime=4h09m28s`. `Runner.choose_next_account` refuses to act
without a known origin inside a known roster, and both come from ONE `identify_account`
call at startup. That call failed, switching was disabled for the whole run, and nothing
said so afterwards. Run by hand on the same phone minutes later the identical read worked
("logged in as NickStanki (L62), 2 account(s) available", 14.0s) - so the failure is
situational, which is exactly what a bounded retry and a loud give-up are for.

Two changes, tested here together because they are two halves of one report:

  * `fsm.Preflight` - a state that runs the switch's OWN zoom -> goplus -> autowalk phase
    methods once at startup, so a run that never switches still gets all three. It reuses
    those methods by subclassing `Switching`; what is asserted below is that it really is
    the same code (the same taps, the same "already autowalking" skip, the same
    do-nothing-on-an-absent-Go-Plus rule) and that it can never stop the bot playing.
  * `cli.prepare_accounts` - retries the identification and, when it still cannot name an
    account, says plainly that account switching is disabled for the run.

The end-to-end tests drive the REAL Runner and are parametrized over the tick interval for
the reason tests/test_autowalk.py states: `Runner.apply` drops `ctx.accounts` after every
actuation taken while PREFLIGHT (the star TOGGLES the shortcut menu), and
`_refresh_accounts` only puts it back every `runner.ACCOUNTS_REFRESH`. A test that ticks
slower than that throttle never sees the resulting `None` view and can pass while a phase
is silently skipping its own step.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import replace

import numpy as np
import pytest

from pogobot import fsm
from pogobot import runner as runner_mod
from pogobot.accounts import AccountView, FakeTreeReader
from pogobot.cli import (IDENTIFY_ATTEMPTS, IDENTIFY_RETRY_WAIT, build_parser,
                         config_from_args, prepare_accounts)
from pogobot.config import DEFAULT, Config
from pogobot.effects import (
    BotState,
    DoubleTapDrag,
    IntentOutcome,
    Note,
    SetFlag,
    Tap,
    Transition,
)
from pogobot.frames import Frame
from pogobot.observation import Tristate
from tests.factories import det, obs
from tests.test_autowalk import LADDER, STAR, _autowalk_reasons, _full_view
from tests.test_switch_runner import ROSTER, make_runner
from tests.test_switching import panel

MAP = obs(on_map=True)
OFF_MAP = obs(screen="Menu", conf=0.99)


# ------------------------------------------------------------------ the handler exists

def test_every_state_still_has_a_handler_with_a_budget_and_a_timeout():
    """The import-time contract in fsm.py, restated as the reason a new state is safe to
    add: a state without a handler, a numeric `timeout_s` or an `on_timeout` is a startup
    error rather than a livelock nobody notices."""
    assert set(fsm.HANDLERS) == set(BotState)
    h = fsm.HANDLERS[BotState.PREFLIGHT]
    assert isinstance(h.timeout_s, (int, float))
    assert h.on_timeout.__func__ is not fsm.Handler.on_timeout
    assert h.timeout(fsm.Context(cfg=Config())) == Config().timings.preflight_timeout


#: The only phase methods `Preflight` is allowed to wrap rather than inherit outright, and
#: what each wrapper is for. Both still delegate to `super()` - asserted below - so the
#: phase logic itself is never a second implementation; what they add is the one thing that
#: differs between the two runs (no view tree means AutoWalk can never be located; a
#: startup give-up has to be audible).
PREFLIGHT_WRAPPED = {"_autowalk_open", "_autowalk_deadline"}


def test_the_preflight_reuses_the_switch_phase_methods_rather_than_copying_them():
    """The whole design in one assertion: every phase it runs comes from `Switching`, so
    the measured zoom gesture, the three-answer Go Plus reading and the "already
    autowalking" rule cannot drift into a second implementation. A wrapper is allowed only
    where it is declared above, and only if it still calls the inherited method."""
    import inspect
    assert issubclass(fsm.Preflight, fsm.Switching)
    for method in [f"_{phase}" for phase in fsm.PREFLIGHT_PHASES] + ["_autowalk_deadline"]:
        own, inherited = getattr(fsm.Preflight, method), getattr(fsm.Switching, method)
        if method in PREFLIGHT_WRAPPED:
            assert own is not inherited, f"{method} is listed as wrapped but is not"
            assert f"super().{method}(" in inspect.getsource(own), \
                f"{method} re-implements the phase instead of delegating to Switching"
        else:
            assert own is inherited, f"{method} was overridden without being declared"


def test_the_preflight_is_not_switching():
    """Its own state, not a mode of SWITCHING. Counting SWITCHING frames in the trace is
    precisely how "no switch ever happened" was established for the four-hour run; a
    preflight that logged itself as a switch would have destroyed that diagnostic."""
    assert fsm.Preflight.state is BotState.PREFLIGHT
    assert BotState.PREFLIGHT is not BotState.SWITCHING


def test_the_budget_is_bounded_and_smaller_than_the_stuck_watchdog():
    """`Timings.preflight_timeout` has to cover the phases it reuses - two zoom gestures,
    two Go Plus press-and-rechecks, and an AutoWalk ladder that bounds itself - while
    staying under `stuck_watchdog`, which is what lets it reuse a switch's phases without
    also needing the watchdog credit `Context.switch_exit_ts` grants a switch."""
    t = Config().timings
    floor = (Config().zoom.repeats * t.ui_settle
             + Config().goplus.max_attempts * Config().goplus.press_wait
             + Config().autowalk.budget_s + Config().autowalk.close_grace_s)
    assert t.preflight_timeout > floor
    assert t.preflight_timeout < t.stuck_watchdog


# ------------------------------------------------------------------ driving it

def pctx(**kw):
    """A Context in PREFLIGHT: no target, no login, a phase from the preflight's own
    chain. The absence of `switch_target` is the whole point - it is what tells the shared
    phase methods which of the two runs they are in."""
    c = fsm.Context(cfg=kw.pop("cfg", Config()), state=BotState.PREFLIGHT,
                    now=100.0, state_since=100.0)
    c.switch_target = None
    c.switch_phase = kw.pop("phase", fsm.PREFLIGHT_PHASES[0])
    c.accounts = kw.pop("accounts", _full_view())
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def taps(effects):
    return [e for e in effects if isinstance(e, Tap)]


def kinds(effects, t):
    return [e for e in effects if isinstance(e, t)]


def test_the_zoom_phase_fires_the_same_measured_gesture():
    out = fsm.step(MAP, pctx(phase="zoom"))
    drags = kinds(out, DoubleTapDrag)
    z = Config().zoom
    assert len(drags) == 1
    # The start point is CHOSEN to avoid whatever the detector can see - a tap that lands
    # on a stop opens the stop and the drag belongs to that screen instead, so no zoom
    # happens (measured by hand as screen=Poi@0.83). See fsm.zoom_anchor.
    ax, ay = fsm.zoom_anchor(MAP, Config())
    assert (drags[0].x1, drags[0].y1) == (ax, ay)
    assert drags[0].y2 == pytest.approx(ay - z.drag_frac)
    assert drags[0].duration_ms == z.duration_ms and drags[0].budget == "zoom"


def test_the_zoom_phase_waits_for_the_map_like_the_switch_does():
    """It is a blind gesture at a fixed coordinate; off the map it would be dragging
    whatever screen happens to be up."""
    assert fsm.step(OFF_MAP, pctx(phase="zoom")) == []


def test_goplus_is_pressed_only_when_it_positively_reads_off():
    c = pctx(phase="goplus")
    pressed = taps(fsm.step(obs(on_map=True, goplus=Tristate.FALSE), c))
    assert len(pressed) == 1 and pressed[0].budget == "goplus"


@pytest.mark.parametrize("reading", [Tristate.TRUE, Tristate.UNKNOWN])
def test_an_on_or_absent_go_plus_is_never_tapped_and_never_blocks(reading):
    """The preserved rule, restated for the preflight: UNKNOWN is also what an account
    with no Virtual Go Plus at all looks like, and "we do not know" must never become "so
    tap it". Either way the chain moves on."""
    out = fsm.step(obs(on_map=True, goplus=reading), pctx(phase="goplus"))
    assert not [t for t in taps(out) if t.budget == "goplus"]
    assert any(isinstance(e, SetFlag) and e.name == "switch_phase"
               and e.value == "autowalk_open" for e in out)


def test_an_already_running_autowalk_is_not_started_again():
    """The other preserved rule (the blue glyph the user reported): a positively-read
    TRUE skips the AutoWalk tap and closes the menu instead."""
    c = pctx(phase="autowalk_menu", switch_autowalk_active=Tristate.TRUE,
             switch_autowalk_since=99.0)
    out = fsm.step(MAP, c)
    reasons = [t.reason for t in taps(out)]
    assert "autowalk: select AutoWalk from the shortcut menu" not in reasons
    assert "autowalk: close the shortcut menu" in reasons


def test_a_phase_the_preflight_was_never_given_never_drives_the_account_panel():
    """`Switching.step`'s fall-through is the login-driving "open" behaviour, and a
    preflight has no target to log into - `by_name(None)` would tap the accounts tab of a
    panel nobody asked to open. Refused outright, and it still lands in SCANNING."""
    c = pctx(phase="open", accounts=panel(active="TrainerOne"))
    out = fsm.step(MAP, c)
    assert not taps(out)
    tr = kinds(out, Transition)
    assert tr and tr[0].to is BotState.SCANNING


def test_the_preflight_owns_the_screen_while_it_runs():
    """Same standing as SWITCHING in `desired_state`: PGSharp's shortcut menu and its
    AutoWalk AlertDialog sit over the map exactly while the ladder needs to act, and a
    full-screen dialog can read as an overlay or as not-on-map."""
    c = pctx(phase="autowalk_menu", switch_autowalk_since=99.0)
    assert fsm.desired_state(obs(screen="PokemonEncounter", conf=0.99), c) is None
    assert fsm.desired_state(obs(on_map=True), c) is None


@pytest.mark.parametrize("phase", fsm.PREFLIGHT_PHASES)
def test_the_handler_writes_nothing_to_the_context(phase):
    """The FSM is pure: (Observation, Context) -> list[Effect]. Only the runner mutates
    the context, which is what keeps a dry run and a live run on the same trajectory.
    Both branches, each against its own context - `on_timeout` needs `state_since` far
    enough in the past to be reached at all."""
    c = pctx(phase=phase, switch_autowalk_since=99.0)
    before = copy.deepcopy(c.__dict__)
    fsm.step(obs(on_map=True, goplus=Tristate.FALSE), c)
    assert c.__dict__ == before

    t = pctx(phase=phase, switch_autowalk_since=99.0, state_since=0.0, now=10_000.0)
    before_timeout = copy.deepcopy(t.__dict__)
    out = fsm.step(OFF_MAP, t)
    assert kinds(out, Note), "meant to exercise on_timeout"
    assert t.__dict__ == before_timeout


# ------------------------------------------------------------------ nothing says "None"

def _strings(effects):
    return ([e.text for e in effects if isinstance(e, Note)]
            + [e.reason for e in effects if hasattr(e, "reason")])


@pytest.mark.parametrize("phase, o, extra", [
    ("zoom", MAP, {}),
    ("goplus", obs(on_map=True, goplus=Tristate.FALSE), {}),
    ("autowalk_menu", MAP, {"switch_autowalk_active": Tristate.TRUE}),
    ("autowalk_close", MAP, {}),
])
def test_no_operator_facing_string_names_a_missing_account(phase, o, extra):
    """"logged into None" is not a cosmetic defect. The failure this whole change exists
    for was invisible precisely because nothing said which account anything belonged to -
    a line that INVENTS one is worse than the silence it replaces."""
    out = fsm.step(o, pctx(phase=phase, switch_autowalk_since=99.0, **extra))
    assert out, "nothing was emitted, so no string was actually checked"
    for s in _strings(out):
        assert "None" not in s, s


@pytest.mark.parametrize("phase", ["autowalk_open", "autowalk_menu", "autowalk_close"])
def test_the_autowalk_give_up_paths_name_no_account_either(phase):
    """The wall-clock deadline (`config.AutoWalk.budget_s`) is the likeliest exit of all
    on a preflight - it fires whenever the star, the menu or a dialog button never
    appears - so it is the path most likely to put a bad line in front of an operator."""
    c = pctx(phase=phase, accounts=AccountView(available=True), switch_autowalk_since=1.0)
    c.now = c.state_since + Config().autowalk.budget_s + Config().autowalk.close_grace_s + 1.0
    out = fsm.step(MAP, c)
    assert kinds(out, Transition), "the deadline did not fire"
    for s in _strings(out):
        assert "None" not in s, s


def test_a_switch_still_says_which_account_it_logged_into():
    """The other half of the same rule: naming the account is exactly right when there IS
    one, and the shared wording must not have flattened that away."""
    from tests.test_switching import ctx as switching_ctx
    c = switching_ctx(phase="autowalk_close", target="TrainerTwo", accounts=_full_view())
    out = fsm.step(MAP, c)
    assert any("logged into TrainerTwo" == e.reason
               for e in kinds(out, Transition)), _strings(out)


def test_the_star_separation_give_up_names_no_account_during_a_preflight():
    from tests.test_star_separation import COLLAPSED
    c = pctx(phase="autowalk_open", accounts=COLLAPSED, switch_autowalk_since=99.0,
             star_drags=Config().star_separation.max_drags)
    out = fsm.step(MAP, c)
    assert kinds(out, Note) and kinds(out, Transition)
    for s in _strings(out):
        assert "None" not in s, s


# ------------------------------------------------------------------ end to end

#: "not given", so a test can ask for a runner with NO tree reader at all - the shape a run
#: with no switch trigger armed really has - without it being confused with the default.
_DEFAULT_READER = object()


def _runner(cfg=None, tree_reader=_DEFAULT_READER, **kw):
    r = make_runner(cfg or Config(), roster=ROSTER,
                    tree_reader=FakeTreeReader([_full_view("TrainerOne")])
                    if tree_reader is _DEFAULT_READER else tree_reader, **kw)
    r.stats.account = "TrainerOne"
    r.ctx.now = 1_000.0
    r.ctx.last_map_ts = r.ctx.now
    r._accounts_read_at = 0.0
    return r


def _tick(r, o):
    """Exactly the order the run loop uses (see Runner.run): refresh, maybe preflight,
    maybe switch, step, apply."""
    r._refresh_accounts(r.ctx.now)
    r._maybe_preflight(o)
    r._maybe_switch(o)
    r.apply(fsm.step(o, r.ctx), o)


def _drive(r, o, dt=0.1, seconds=200.0):
    """Tick until the preflight has both started and handed the screen back, or `seconds`
    of simulated clock have gone by. Returns how long that took, or None."""
    t0 = r.ctx.now
    while r.ctx.now - t0 < seconds:
        r.ctx.now = round(r.ctx.now + dt, 6)
        _tick(r, o)
        if r._preflight_done and r.ctx.state is BotState.SCANNING:
            return r.ctx.now - t0
    return None


#: The runner ticks at frame rate, so 0.1s is the cadence the live loop actually has and
#: the case that has to hold. 3.0s is kept for the reason tests/test_autowalk.py keeps it:
#: it is the only cadence at which a tick outlasts the 2.5s tree-refresh throttle, so the
#: view is never `None` when a phase looks - which is how a missing tap once passed.
TICK_INTERVALS = [0.1, 0.5, 1.0, 3.0]


@pytest.mark.parametrize("dt", TICK_INTERVALS)
def test_the_startup_preflight_runs_all_three_steps_in_order_then_plays(dt):
    """The user's request, end to end through the real Runner and the real FSM: zoom out,
    then Virtual Go Plus, then AutoWalk, then play."""
    r = _runner()
    elapsed = _drive(r, obs(on_map=True, goplus=Tristate.FALSE), dt)
    assert elapsed is not None, f"never reached SCANNING (state={r.ctx.state})"
    assert r.ctx.state is BotState.SCANNING

    applied = r.actuator.applied
    steps = [("zoom" if isinstance(e, DoubleTapDrag) else
              "goplus" if getattr(e, "budget", "") == "goplus" else
              "autowalk" if "autowalk" in getattr(e, "reason", "") else "other")
             for e in applied]
    # Collapsed to the ORDER of the three steps: how many actuations each spends is its
    # own bounded business (`ZoomOut.repeats`, `GoPlusToggle.max_attempts` - the toggle
    # here never reads back ON, so it legitimately re-presses), while the sequence is what
    # a preflight has to get right. "other" would be a tap belonging to no step at all.
    collapsed = [s for i, s in enumerate(steps) if i == 0 or steps[i - 1] != s]
    assert collapsed == ["zoom", "goplus", "autowalk"], steps
    assert len(kinds(applied, DoubleTapDrag)) == Config().zoom.repeats
    assert 1 <= steps.count("goplus") <= Config().goplus.max_attempts
    assert _autowalk_reasons(r) == LADDER
    star_taps = [(e.x, e.y) for e in applied
                 if isinstance(e, Tap) and (e.x, e.y) == pytest.approx(STAR)]
    assert len(star_taps) == 2, "the shortcut menu was opened but not closed again"
    assert elapsed < Config().timings.preflight_timeout


@pytest.mark.parametrize("dt", TICK_INTERVALS)
def test_a_preflight_skips_an_autowalk_that_is_already_running(dt):
    """The user's own report, preserved end to end: a blue AutoWalk glyph means that
    account is ALREADY autowalking and must not be tapped again.

    Driven through the real Runner because that is where the rule actually lives - the
    handler only reads `ctx.switch_autowalk_active`, and it is `Runner._refresh_accounts`
    that fills it in, from the icon bounds the dump just supplied and the colour of the
    frame this tick already ran on. Written for SWITCHING, that refresh would have left a
    preflight reading UNKNOWN forever and tapping AutoWalk on an account already walking.
    The icon box and the frame are the ONE captured moment PGSharp ever showed this
    (tests/fixtures/{uiautomator,screens}/autowalk_menu_active.*); the rest of the view is
    tests/test_autowalk.py's synthetic composite, for the reason its own docstring gives.
    """
    from tests.test_autowalk import _active_bgr, view as fixture_view
    icon_rect = fixture_view("autowalk_menu_active.xml").autowalk_icon_rect_norm
    assert icon_rect is not None, "the fixture no longer parses"
    combined = replace(_full_view("TrainerOne"), autowalk_icon_rect_norm=icon_rect)
    r = _runner(tree_reader=FakeTreeReader([combined]))
    r._last_frame = Frame(seq=1, ts=0.0, bgr=_active_bgr())

    assert _drive(r, obs(on_map=True, goplus=Tristate.UNKNOWN), dt) is not None
    assert r.ctx.state is BotState.SCANNING
    assert _autowalk_reasons(r) == ["autowalk: open the PGSharp shortcut menu",
                                    "autowalk: close the shortcut menu"]


def test_the_preflight_starts_from_boot_without_giving_scanning_a_tick_first():
    """It is called before `fsm.step`, so on the tick BOOT first sees the map the state is
    still BOOT and the preflight takes the screen there. Otherwise the bot gets one
    SCANNING tick in which to tap a target and start playing zoomed in - the exact thing
    the preflight exists to prevent."""
    r = _runner()
    assert r.ctx.state is BotState.BOOT
    _tick(r, obs(on_map=True, detections=(det(conf=0.9),)))
    assert r.ctx.state is BotState.PREFLIGHT
    assert not [e for e in r.actuator.applied if getattr(e, "budget", "") == "tap"]


class _MapSrc:
    """A capture source that hands back real frames, so `run()` reaches perception. Local,
    like every other runner-shaped fake in this suite (tests/test_panel_recovery.py)."""

    def __init__(self, stop_after: int = 2):
        self.reads = 0
        self.stop_after = stop_after
        self.runner = None

    def read(self):
        self.reads += 1
        if self.runner is not None and self.reads >= self.stop_after:
            self.runner._stop = True
        return Frame(seq=self.reads, ts=time.perf_counter(),
                     bgr=np.zeros((1280, 590, 3), dtype=np.uint8))

    def healthy(self):
        return True

    def release(self):
        pass


class _SeesTheMap:
    def observe(self, frame, keyboard=None):
        return obs(on_map=True, seq=frame.seq, ts=frame.ts, goplus=Tristate.FALSE)


def test_the_real_run_loop_actually_starts_the_preflight():
    """Bound to `Runner.run()` itself, not to a test that mirrors its order.

    Every other end-to-end test here calls `_maybe_preflight` the way the loop does, which
    proves the state machine but not the wiring: deleting the call from `run()` outright
    left the whole suite green. This is the same gap tests/test_cli_wiring.py exists for,
    and the same shape of fix - drive the real thing and assert the first gesture reached
    the actuator."""
    r = _runner(cfg=Config(infer_fps=1_000.0))
    r.source = _MapSrc()
    r.source.runner = r
    r.perceptor = _SeesTheMap()

    assert r.run() == 0
    assert r._ticks >= 1, "the loop never completed a tick, so nothing was exercised"
    assert r._preflight_done is True, "run() never started the startup preflight"
    assert r.ctx.state is BotState.PREFLIGHT
    assert kinds(r.actuator.applied, DoubleTapDrag), "no zoom-out reached the actuator"


def test_a_slow_start_that_reaches_the_map_through_recovering_still_preflights():
    """BOOT's own budget is 30s and a cold-started game is measured in tens of seconds, so
    a startup that overruns arrives at the map from RECOVERING and never passes through
    BOOT again. Hanging the preflight off BOOT alone would lose it on exactly the startups
    that need it most."""
    r = _runner()
    r.enter_state(BotState.SCANNING, IntentOutcome.CARRIED, "recovered late")
    _tick(r, obs(on_map=True))
    assert r.ctx.state is BotState.PREFLIGHT


def test_it_is_skipped_when_the_knob_is_off():
    """Off means untouched, not merely quicker: no gesture, no toggle, no star - and no
    tree read either, which is the part that costs the run loop ~3s of blindness a time."""
    r = _runner(cfg=Config(preflight=False))
    _drive(r, obs(on_map=True, goplus=Tristate.FALSE), dt=0.5, seconds=60.0)
    assert r.ctx.state is not BotState.PREFLIGHT
    assert r._preflight_done is False
    assert not kinds(r.actuator.applied, DoubleTapDrag)
    assert not [e for e in r.actuator.applied if getattr(e, "budget", "") == "goplus"]
    assert not _autowalk_reasons(r)
    assert r.tree_reader.reads == 0, "a disabled preflight still paid for a tree read"


def test_the_knob_defaults_to_on():
    assert Config().preflight is True


@pytest.mark.parametrize("dt", [0.1, 1.0])
def test_it_runs_once_per_run_not_once_per_visit_to_scanning(dt):
    """SCANNING is returned to constantly - after every encounter, popup and recovery - so
    a preflight keyed on the state rather than on a latch would re-take the screen for the
    rest of the run instead of playing."""
    r = _runner()
    assert _drive(r, obs(on_map=True, goplus=Tristate.FALSE), dt) is not None
    entries = 0
    for _ in range(400):
        r.ctx.now = round(r.ctx.now + dt, 6)
        _tick(r, obs(on_map=True))
        if r.ctx.state is BotState.PREFLIGHT:
            entries += 1
    assert entries == 0, "the preflight took the screen back after it had finished"
    assert r._preflight_done is True


def test_the_latch_holds_even_when_the_preflight_could_not_finish():
    """A preflight that gave up must not be retried on the next SCANNING tick: PREFLIGHT
    hands control back to SCANNING either way, so retrying on the state would be a
    livelock rather than a second chance."""
    r = _runner(tree_reader=None)   # no overlay reader: AutoWalk cannot even be located
    assert _drive(r, obs(on_map=True, goplus=Tristate.UNKNOWN), dt=0.5) is not None
    for _ in range(50):
        r.ctx.now = round(r.ctx.now + 0.5, 6)
        _tick(r, obs(on_map=True))
        assert r.ctx.state is not BotState.PREFLIGHT


# ------------------------------------------------------------------ it never stops the bot playing

def test_without_a_tree_reader_it_still_zooms_and_still_ends_up_playing(caplog):
    """AutoWalk is located in the uiautomator tree and there is nothing to locate it with,
    but the zoom-out and the Go Plus toggle are a fixed gesture and an optical reading. Two
    of the three still happen, the ladder gives up on its own wall clock, and the bot
    plays - with a line saying why, rather than a silent 30s of nothing."""
    r = _runner(tree_reader=None)
    with caplog.at_level(logging.WARNING, logger="pogobot"):
        elapsed = _drive(r, obs(on_map=True, goplus=Tristate.FALSE), dt=0.5)
    assert elapsed is not None and r.ctx.state is BotState.SCANNING
    assert len(kinds(r.actuator.applied, DoubleTapDrag)) == Config().zoom.repeats
    assert [e for e in r.actuator.applied if getattr(e, "budget", "") == "goplus"]
    assert not _autowalk_reasons(r)
    assert any("AutoWalk" in m and "view tree" in m for m in caplog.messages), caplog.messages


def test_a_panel_that_never_answers_still_ends_in_scanning():
    """The tree read succeeds and reports nothing usable - no star, no menu, no dialog -
    which is what a PGSharp update or an unexpected screen looks like. The AutoWalk ladder
    is bounded by wall clock precisely so zero locatable nodes still ends the state."""
    r = _runner(tree_reader=FakeTreeReader([AccountView(available=True)]))
    elapsed = _drive(r, obs(on_map=True, goplus=Tristate.UNKNOWN), dt=0.5)
    assert elapsed is not None and r.ctx.state is BotState.SCANNING
    assert not _autowalk_reasons(r), "nothing was locatable, so nothing may be tapped"
    assert elapsed < Config().timings.preflight_timeout


def test_a_preflight_that_never_gets_the_map_back_times_out_into_scanning(caplog):
    """The state timeout is the backstop under every other bound: `_zoom` waits for the
    map and does nothing without it, so a screen that never returns is a preflight that
    never advances a phase. It must end in SCANNING - the bot is perfectly able to play -
    and it must say which of the three steps did not happen."""
    r = _runner()
    with caplog.at_level(logging.WARNING, logger="pogobot"):
        for _ in range(400):
            r.ctx.now = round(r.ctx.now + 0.5, 6)
            _tick(r, OFF_MAP if r.ctx.state is BotState.PREFLIGHT else MAP)
            if r._preflight_done and r.ctx.state is not BotState.PREFLIGHT:
                break
    assert r._preflight_done and r.ctx.state is BotState.SCANNING
    warned = [m for m in caplog.messages if "startup preflight ran out" in m]
    assert len(warned) == 1, caplog.messages
    assert "zoom" in warned[0] and "Virtual Go Plus" in warned[0] and "AutoWalk" in warned[0]


def test_the_timeout_warning_names_only_what_is_actually_outstanding():
    """Stuck at the last rung, everything but the menu closure has already happened -
    reporting the zoom-out as skipped there would send an operator looking for a bug that
    is not in front of them."""
    c = pctx(phase="autowalk_close", state_since=0.0, now=10_000.0)
    notes = kinds(fsm.step(MAP, c), Note)
    assert notes and "zoom" not in notes[0].text
    assert "menu" in notes[0].text


def test_the_preflight_never_hands_off_to_recovering():
    """Nothing about an incomplete startup check is evidence that the bot is stuck, and
    RECOVERING is where a run goes to spend BACK presses and, eventually, app restarts."""
    for phase in fsm.PREFLIGHT_PHASES:
        c = pctx(phase=phase, state_since=0.0, now=10_000.0)
        out = kinds(fsm.step(OFF_MAP, c), Transition)
        assert out, f"{phase} emitted no transition, so nothing was actually checked"
        assert all(tr.to is BotState.SCANNING for tr in out), out


# ------------------------------------------------------------------ runner plumbing

def test_the_preflight_gets_a_tree_read_at_the_switch_cadence():
    """It has to be able to look at all - `_refresh_accounts` used to read only while
    SWITCHING or RECOVERING - and it needs the FAST cadence: `config.AutoWalk.budget_s` is
    30s for four locate-and-tap steps, and at RECOVER_ACCOUNTS_REFRESH (10s) the ladder
    would spend that budget waiting for its second usable view."""
    reader = FakeTreeReader([_full_view("TrainerOne")])
    r = make_runner(tree_reader=reader)
    r.ctx.state = BotState.PREFLIGHT
    real = r._real
    r._refresh_accounts(real)
    assert reader.reads == 1 and r.ctx.accounts is not None
    r._refresh_accounts(real + runner_mod.ACCOUNTS_REFRESH - 0.1)
    assert reader.reads == 1, "the throttle is not applied"
    r._refresh_accounts(real + runner_mod.ACCOUNTS_REFRESH + 0.5)
    assert reader.reads == 2


def test_a_preflight_read_leaves_the_switch_bookkeeping_alone():
    """`_last_seen_active` is the record of who the tree named during a switch ATTEMPT and
    is spent by `_on_switch_failed`. A preflight is not an attempt, so it must not write
    evidence a later failed switch would then read as its own."""
    r = make_runner(tree_reader=FakeTreeReader([panel(active="TrainerOne")]))
    r.ctx.state = BotState.PREFLIGHT
    r._refresh_accounts(r._real)
    assert r.ctx.accounts is not None
    assert r._last_seen_active is None


def test_the_view_is_dropped_after_every_preflight_actuation():
    """The star TOGGLES PGSharp's shortcut menu, so a second decision taken from one view
    closes the menu the first one opened. `Runner.apply` drops the view after every
    actuation while SWITCHING for that reason; PREFLIGHT drives the same widget."""
    r = make_runner()
    r.ctx.state = BotState.PREFLIGHT
    r.ctx.accounts = _full_view("TrainerOne")
    r.apply([Tap(*STAR, "autowalk: open the PGSharp shortcut menu", budget="switch")], MAP)
    assert r.ctx.accounts is None


def test_the_zoom_counter_is_advanced_by_the_runner_here_too():
    """`switch_zoom_reps` is what lets `_zoom` know it has sent `repeats` REAL gestures.
    The handler is pure and cannot count its own actuations."""
    r = make_runner()
    r.ctx.state = BotState.PREFLIGHT
    z = Config().zoom
    r.apply([DoubleTapDrag(z.center_x, z.center_y, z.center_x, 0.3, "preflight: zoom out",
                           duration_ms=z.duration_ms, budget="zoom")], MAP)
    assert r.ctx.switch_zoom_reps == 1


def test_beginning_a_preflight_clears_what_a_stale_run_would_poison():
    r = make_runner()
    r.ctx.accounts = panel(active="TrainerOne")
    r.ctx.switch_target = "TrainerTwo"
    r.ctx.switch_autowalk_since = 5.0
    r.ctx.switch_zoom_reps = 2
    r._begin_preflight()
    assert r.ctx.state is BotState.PREFLIGHT
    assert r.ctx.accounts is None
    assert r.ctx.switch_target is None, "a preflight must not name an account"
    assert r.ctx.switch_phase == fsm.PREFLIGHT_PHASES[0] == "zoom"
    assert r.ctx.switch_autowalk_since == 0.0
    assert r.ctx.switch_zoom_reps == 0


def test_a_pending_tap_does_not_outlive_the_preflight_that_took_the_screen():
    """An Intent is a causal claim - "the screen changed BECAUSE of this tap" - and the
    ledger writes a training sample on the strength of it. A preflight then drives the
    screen for up to its whole budget, so anything it produces is not that tap's answer.
    Same reasoning, and the same call, as `_begin_switch`."""
    r = make_runner()
    r.ctx.intent = fsm.Intent(ts=r.ctx.now, target_name="pokemon", confidence=0.9,
                              tap_norm=(0.5, 0.6), xywhn=(0.5, 0.6, 0.1, 0.1),
                              expected=BotState.ENCOUNTER, frame_seq=1)
    r._begin_preflight()
    assert r.ctx.intent is None


def test_a_due_switch_does_not_take_the_screen_before_the_preflight(tmp_path):
    """An account that starts the run already capped makes the quota trigger due on the
    very first map frame. The preflight goes first, and the switch simply stays due."""
    from pogobot.quota import SpinQuota
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerOne")
    r = _runner(cfg=Config(switch_on_quota=True), quota=q)
    r.ctx.spins_exhausted = True
    _tick(r, obs(on_map=True))
    assert r.ctx.state is BotState.PREFLIGHT


def test_a_switch_can_still_start_once_the_preflight_is_over(tmp_path):
    """The other half: the preflight delays a switch by its own bounded budget and no
    more - it must not leave switching wedged off."""
    from pogobot.quota import SpinQuota
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerOne")
    r = _runner(cfg=Config(switch_on_quota=True), quota=q)
    r.ctx.spins_exhausted = True
    assert _drive(r, obs(on_map=True, goplus=Tristate.UNKNOWN), dt=0.5) is not None
    r.ctx.now = round(r.ctx.now + 0.5, 6)
    _tick(r, obs(on_map=True))
    assert r.ctx.state is BotState.SWITCHING


# ------------------------------------------------------------------ the CLI knob

def test_the_flag_turns_it_off_and_is_on_without_it():
    assert config_from_args(build_parser().parse_args([])).preflight is True
    assert config_from_args(build_parser().parse_args(["--no-preflight"])).preflight is False


# ------------------------------------------------------------------ startup identification

class _Act:
    """Records what would have gone to the phone. Local, like every other runner-shaped
    fake in this suite (tests/test_accounts.py, tests/test_cli_wiring.py)."""

    def __init__(self):
        self.applied = []
        self.dry_run = False

    def apply(self, effect, now=None):
        self.applied.append(effect)
        return True


SWITCHING_ON = DEFAULT.scaled(switch_on_quota=True)


def _identify(views, requested=None, attempts=IDENTIFY_ATTEMPTS, waits=None,
              monkeypatch=None):
    """Run the real `prepare_accounts` against a queued tree, recording the waits it asks
    for instead of sleeping them (see cli.IDENTIFY_RETRY_WAIT for why the real value is
    ten seconds and why a suite must not spend them)."""
    reader = FakeTreeReader(views)
    act = _Act()
    if monkeypatch is not None and waits is not None:
        monkeypatch.setattr("pogobot.cli.time.sleep", waits.append)
    out = prepare_accounts(SWITCHING_ON, requested=requested, pause_file=None,
                           make_reader=lambda: reader, actuator=act, settle=0,
                           attempts=attempts,
                           retry_wait=IDENTIFY_RETRY_WAIT if waits is not None else 0)
    return out, reader, act


def _closed():
    from tests.test_cli_wiring import _closed as closed
    return closed()


def test_a_read_that_fails_the_first_time_is_tried_again(caplog):
    """The measured failure: one read at startup failed, switching was disabled for four
    hours, and by hand on the same phone minutes later the identical read worked. The
    whole fix is that the first answer is no longer the last one."""
    views = [AccountView(available=False), AccountView(available=False),
             _closed(), panel(active="TrainerTwo")]
    with caplog.at_level(logging.INFO, logger="pogobot"):
        (reader, account, roster), tree, act = _identify(views)
    assert account == "TrainerTwo"
    assert roster == ("TrainerOne", "TrainerTwo")
    assert any("attempt 3 of 3" in m for m in caplog.messages), caplog.messages
    assert [e for e in act.applied if isinstance(e, Tap)], "the retry never opened the panel"


def test_the_attempts_are_bounded(monkeypatch):
    """Three, not forever: past that the cause is not timing, and each further look only
    delays a run that will have to be restarted anyway."""
    calls = []

    def never(reader, actuator, settle=1.0):
        calls.append(settle)
        return None

    monkeypatch.setattr("pogobot.accounts.identify_account", never)
    _identify([AccountView(available=False)])
    assert len(calls) == IDENTIFY_ATTEMPTS == 3


def test_the_retries_are_spaced_and_the_last_attempt_does_not_wait(monkeypatch):
    """The gap is deliberate - the hypothesis is "not ready YET" - and there is nothing to
    wait for after the final attempt."""
    waits = []
    _identify([AccountView(available=False)], waits=waits, monkeypatch=monkeypatch)
    assert waits == [IDENTIFY_RETRY_WAIT] * (IDENTIFY_ATTEMPTS - 1)
    assert IDENTIFY_RETRY_WAIT > 0


def test_a_read_that_works_first_time_costs_no_retry_and_no_wait(monkeypatch):
    waits = []
    (reader, account, roster), tree, act = _identify(
        [_closed(), panel(active="TrainerTwo")], waits=waits, monkeypatch=monkeypatch)
    assert account == "TrainerTwo" and waits == []
    assert tree.reads == 2, "the successful path re-read the panel it had already read"


def test_giving_up_says_plainly_that_switching_is_disabled(caplog):
    """Not a quiet info line. `Runner.choose_next_account` returns None without an origin,
    so NOTHING is attempted and nothing further is ever logged - the whole four hours of
    the reported run. The wording matches the paused-at-startup warning beside it because
    the consequence is the same."""
    with caplog.at_level(logging.INFO, logger="pogobot"):
        (reader, account, roster), tree, act = _identify([AccountView(available=False)])
    loud = [r for r in caplog.records
            if r.levelno >= logging.WARNING and "SWITCHING IS DISABLED" in r.getMessage()]
    assert len(loud) == 1, [r.getMessage() for r in caplog.records]
    assert "attempt" in loud[0].getMessage()


def test_giving_up_invents_neither_an_account_nor_a_roster():
    """A guessed name books every spin to an account that may be capped; an invented
    roster would have a switch tap a login row PGSharp's panel does not contain and burn
    the whole `switch_timeout` finding out."""
    (reader, account, roster), tree, act = _identify([AccountView(available=False)])
    assert account is None and roster == ()
    assert act.applied == [], "an unreadable overlay must never be tapped at a guess"


def test_the_account_flag_survives_a_failed_identification(caplog):
    """--account is still used to attribute the run; what it cannot do is supply a roster,
    which is why the warning still fires."""
    with caplog.at_level(logging.WARNING, logger="pogobot"):
        (reader, account, roster), tree, act = _identify([AccountView(available=False)],
                                                         requested="TrainerOne")
    assert account == "TrainerOne" and roster == ()
    assert any("SWITCHING IS DISABLED" in m for m in caplog.messages)


def test_a_panel_that_opens_on_nobody_is_retried_too():
    """PGSharp showing NEITHER account with an asterisk is a state the field has actually
    produced (see config.Timings.switch_clear_max). A roster without an origin still means
    no switch can start, so it is a failure like any other."""
    nobody = replace(panel(active=None), rows=())
    views = [_closed(), nobody, _closed(), panel(active="TrainerTwo")]
    (reader, account, roster), tree, act = _identify(views)
    assert account == "TrainerTwo"


# ------------------------------------------------------------------ it is never the reason the bot stops

def test_every_timeout_warning_reads_as_a_sentence():
    """The clause per phase is a whole clause, not a noun phrase slotted into a shared
    tail. Slotted, the last rung read "but nothing, but PGSharp's shortcut menu may still
    be over the map did not happen this run" - unreadable, and taken at face value the
    opposite of the truth, on the one warning that matters most: a menu left over the
    reach ellipse silently kills AutoWalk for the rest of the run."""
    for phase in fsm.PREFLIGHT_PHASES:
        c = pctx(phase=phase, state_since=0.0, now=10_000.0)
        text = kinds(fsm.step(OFF_MAP, c), Note)[0].text
        assert phase in text, text
        assert "nothing" not in text, text
        assert text.count(" but ") <= 1, text
        head, _, clause = text.partition(" - playing anyway")
        assert clause.startswith(":") and len(clause) > 10, text


def test_the_last_rung_says_autowalk_is_running_rather_than_that_nothing_happened():
    """"is running" rather than "started": by this phase a route is up, but the ladder may
    have found it ALREADY up rather than started it - PGSharp answers the menu entry with
    its Stop/Pause dialog when one is running (see tests/test_autowalk_running.py), and
    "AutoWalk started" would be a plain untruth about that run. What the operator needs
    from this line either way is that a route is up and only the menu is unaccounted for."""
    c = pctx(phase="autowalk_close", state_since=0.0, now=10_000.0)
    text = kinds(fsm.step(OFF_MAP, c), Note)[0].text
    assert "AutoWalk is running" in text and "shortcut menu" in text, text


@pytest.mark.parametrize("phase", ["zoom", "goplus"])
def test_a_closable_overlay_hands_the_screen_back_during_the_blind_phases(phase):
    """`desired_state` gives this state the screen and its POPUP branch fires only from
    SCANNING/TARGETING, so while a preflight holds a closable overlay NOTHING closes it -
    and `_zoom`/`_goplus` cannot advance either, since both refuse to fire without the
    map. Driven against the real Runner that was 90.5s in PREFLIGHT with one actuation,
    against 8 close attempts over the same 90s with the preflight off."""
    c = pctx(phase=phase)
    out = fsm.step(obs(x_button=True, screen="Menu", conf=0.99), c)
    tr = kinds(out, Transition)
    assert tr and tr[0].to is BotState.SCANNING, out
    assert kinds(out, Note) and kinds(out, Note)[0].level == "warn"
    assert not taps(out) and not kinds(out, DoubleTapDrag)


@pytest.mark.parametrize("phase", ["autowalk_open", "autowalk_menu", "autowalk_dialog",
                                   "autowalk_close"])
def test_the_autowalk_rungs_do_not_yield_to_an_overlay(phase):
    """The other half, and the reason the yield is scoped to two phases: these act on
    coordinates read from the live tree, under PGSharp's own menu and AlertDialog - which
    is exactly the kind of screen that reads as an overlay while the ladder still has to
    act on it. Yielding here abandons a half-driven menu over the map."""
    c = pctx(phase=phase, switch_autowalk_since=99.0)
    out = fsm.step(obs(x_button=True, screen="Menu", conf=0.99), c)
    assert not any(isinstance(e, Transition) and e.reason.endswith("closable overlay")
                   for e in out), out


def test_one_misread_frame_cannot_end_the_preflight():
    """`in_overlay` demands a smoothed X button AND no map AND no encounter. A frame that
    is merely not-on-map - a loading screen mid-zoom - leaves the state where it is, and
    the 90s budget stays the backstop under it."""
    assert fsm.step(OFF_MAP, pctx(phase="zoom")) == []


def test_without_a_tree_reader_the_autowalk_ladder_is_not_even_entered():
    """`cli.prepare_accounts` hands over a reader only when a switch trigger is armed,
    which is not the default - so this is what an ordinary run does. Every autowalk rung
    locates its widget in the view tree, so with no reader the inherited ladder can only
    wait out `AutoWalk.budget_s`: measured as two zoom drags and then 30.4s of empty ticks
    during which `desired_state` hands PREFLIGHT the screen and an encounter, a Rocket
    screen or a popup is ignored."""
    r = _runner(tree_reader=None)
    assert r.ctx.tree_available is False
    elapsed = _drive(r, obs(on_map=True, goplus=Tristate.FALSE), dt=0.5)
    assert elapsed is not None and r.ctx.state is BotState.SCANNING
    assert elapsed < Config().autowalk.budget_s, \
        "the preflight waited out a ladder that could never locate anything"
    assert len(kinds(r.actuator.applied, DoubleTapDrag)) == Config().zoom.repeats
    assert [e for e in r.actuator.applied if getattr(e, "budget", "") == "goplus"]


def test_a_preflight_that_cannot_start_autowalk_says_so_out_loud(caplog):
    """"AutoWalk is not running" is one of the three symptoms actually reported, and the
    likely ways to reach it - a star that cannot be located, a menu that never renders -
    all leave through the ladder's deadline, which is otherwise one INFO transition line.
    The same change made a failed identification a WARNING; this is the same silence."""
    r = _runner(tree_reader=FakeTreeReader([AccountView(available=True)]))
    with caplog.at_level(logging.INFO, logger="pogobot"):
        assert _drive(r, obs(on_map=True, goplus=Tristate.UNKNOWN), dt=0.5) is not None
    loud = [rec for rec in caplog.records if rec.levelno >= logging.WARNING
            and "could not start AutoWalk" in rec.getMessage()]
    assert len(loud) == 1, [rec.getMessage() for rec in caplog.records]


def test_a_switch_that_gives_up_on_autowalk_is_not_made_noisy_too():
    """`Switching` is deliberately left alone: a switch that got this far has already
    logged the login it confirmed, and the wrapper is on `Preflight` only."""
    from tests.test_switching import ctx as switching_ctx
    c = switching_ctx(phase="autowalk_open", target="TrainerTwo",
                      accounts=AccountView(available=True))
    c.switch_autowalk_since = 1.0
    c.now = c.state_since + Config().autowalk.budget_s + 1.0
    out = fsm.step(MAP, c)
    assert kinds(out, Transition), "the deadline did not fire"
    assert not [n for n in kinds(out, Note) if n.level == "warn"], out


# ------------------------------------------------------------------ which run owns the phases

@pytest.mark.parametrize("phase, o, extra", [
    ("zoom", MAP, {}),
    ("goplus", obs(on_map=True, goplus=Tristate.FALSE), {}),
])
def test_a_preflight_labels_its_own_gestures_preflight(phase, o, extra):
    """`_label` is the one string that separates the two runs in the trace, and the trace
    is the artifact the four-hour failure was diagnosed from - by counting switch
    evidence in it. A preflight whose gestures read "switch: ..." corrupts exactly that."""
    out = fsm.step(o, pctx(phase=phase, **extra))
    reasons = [e.reason for e in out if hasattr(e, "reason")]
    assert reasons, "nothing was emitted, so no label was actually checked"
    for r in reasons:
        assert r.startswith("preflight: "), r


def test_a_switch_labels_its_gestures_switch_byte_for_byte():
    """The mirror, and the claim the change makes about itself: the switch path's strings
    are unchanged. Written out in full rather than asserted by prefix, because "unchanged"
    is the whole claim."""
    from tests.test_switching import ctx as switching_ctx
    c = switching_ctx(phase="zoom", target="TrainerTwo", accounts=_full_view())
    drags = kinds(fsm.step(MAP, c), DoubleTapDrag)
    assert [d.reason for d in drags] == \
        [f"switch: zoom out after confirming TrainerTwo (1/{Config().zoom.repeats})"]

    g = switching_ctx(phase="goplus", target="TrainerTwo", accounts=_full_view())
    pressed = taps(fsm.step(obs(on_map=True, goplus=Tristate.FALSE), g))
    assert [t.reason for t in pressed] == ["switch: re-enable Virtual Go Plus"]


# ------------------------------------------------------------------ the run loop's own order

def test_the_real_run_loop_preflights_before_it_switches(tmp_path):
    """Bound to `Runner.run()`, not to a helper that mirrors its order. Swapping the two
    calls in the loop, or moving the preflight below `fsm.step`, left the whole suite
    green: the other ordering tests reach the loop through `_tick`, which hardcodes the
    order it claims to test. This is the slow-start path - already in SCANNING, with a
    trigger due - which is the only path where the order can actually differ."""
    from pogobot.quota import SpinQuota
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerOne")
    r = _runner(cfg=Config(switch_on_quota=True, infer_fps=1_000.0), quota=q)
    r.enter_state(BotState.SCANNING, IntentOutcome.CARRIED, "already playing")
    r.ctx.spins_exhausted = True
    r.source = _MapSrc()
    r.source.runner = r
    r.perceptor = _SeesTheMap()

    assert r.run() == 0
    assert r._ticks >= 1, "the loop never completed a tick, so nothing was exercised"
    assert r._preflight_done is True, "the switch took the screen first"
    assert r.ctx.state is BotState.PREFLIGHT, r.ctx.state


def test_the_preflight_waits_for_a_confirmed_map_before_taking_the_screen():
    """`_zoom` and `_goplus` fire a blind gesture at a fixed coordinate. Started against a
    loading screen the state advances nothing, burns its whole budget and - because the
    latch is already set - never runs again; worse, BOOT's own "no map within 30s" escape
    to RECOVERING can no longer fire, because PREFLIGHT is not BOOT."""
    r = _runner()
    r._maybe_preflight(OFF_MAP)
    assert r.ctx.state is BotState.BOOT
    assert r._preflight_done is False, "the one chance at a preflight was spent on no map"


def test_a_boot_that_never_sees_the_map_still_escalates_to_recovering():
    """The consequence of the guard above, driven through the loop: BOOT's 30s budget has
    to stay reachable."""
    r = _runner()
    r.ctx.state_since = r.ctx.now          # BOOT started now, not at process start
    for _ in range(400):
        r.ctx.now = round(r.ctx.now + 0.5, 6)
        _tick(r, OFF_MAP)
        if r.ctx.state is BotState.RECOVERING:
            break
    assert r.ctx.state is BotState.RECOVERING, r.ctx.state
    assert r.ctx.now - 1_000.0 < fsm.Boot.timeout_s + 5.0


# ------------------------------------------------------------------ once per game, not once per process

def test_an_accepted_app_restart_re_arms_the_preflight():
    """A cold relaunch resets the camera, Virtual Go Plus and the AutoWalk route exactly
    as a login does - `_separate_star`'s own measurement is taken "immediately after
    effects.RestartApp relaunched the game". A run-lifetime latch hands the reported
    symptom straight back: the run that motivated this change logged 215 recoveries."""
    from pogobot.effects import RestartApp
    r = _runner()
    assert _drive(r, obs(on_map=True, goplus=Tristate.UNKNOWN), dt=0.5) is not None
    assert r._preflight_done is True
    r.apply([RestartApp(package="com.x", activity=".Y", reason="wedged")], MAP)
    assert r._preflight_done is False, "the restart left the run zoomed in for good"
    r.ctx.now = round(r.ctx.now + 0.5, 6)
    _tick(r, obs(on_map=True))
    assert r.ctx.state is BotState.PREFLIGHT


def test_a_refused_restart_does_not_re_arm_it():
    """Same reasoning as every other counter in `apply`: a pure handler cannot know its
    RestartApp reached the device, so only an ACCEPTED one may mean the game restarted."""
    from pogobot.effects import RestartApp

    class _Refuses(_Act):
        def apply(self, effect, now=None):
            return False

    r = _runner()
    assert _drive(r, obs(on_map=True, goplus=Tristate.UNKNOWN), dt=0.5) is not None
    r.actuator = _Refuses()
    r.apply([RestartApp(package="com.x", activity=".Y", reason="wedged")], MAP)
    assert r._preflight_done is True


def test_a_dry_run_does_not_spend_the_retry_budget_on_a_tap_it_suppressed():
    """`Actuator.apply` suppresses the launcher tap under --dry-run, so the panel is never
    opened and no further look can open it. Retrying spends `IDENTIFY_RETRY_WAIT` of real
    `time.sleep` between multi-second blocking dumps to reach a give-up line that then
    blames the phone. The retry is for a SITUATIONAL failure; this one is structural."""
    waits = []
    act = _Act()
    act.dry_run = True
    reader = FakeTreeReader([_closed()])
    prepare_accounts(SWITCHING_ON, requested=None, pause_file=None,
                     make_reader=lambda: reader, actuator=act, settle=0,
                     attempts=IDENTIFY_ATTEMPTS, retry_wait=0)
    assert waits == []
    assert reader.reads == 2, f"a suppressed tap was looked for {reader.reads} times"


def test_the_give_up_line_blames_the_dry_run_rather_than_the_phone(caplog):
    """An operator told to "restart the bot with both on screen" goes to look at a phone
    that is behaving perfectly. The cause has to match the run that actually happened."""
    act = _Act()
    act.dry_run = True
    reader = FakeTreeReader([_closed()])
    with caplog.at_level(logging.WARNING, logger="pogobot"):
        prepare_accounts(SWITCHING_ON, requested=None, pause_file=None,
                         make_reader=lambda: reader, actuator=act, settle=0, retry_wait=0)
    loud = [m for m in caplog.messages if "SWITCHING IS DISABLED" in m]
    assert len(loud) == 1, caplog.messages
    assert "--dry-run" in loud[0] and "restart the bot" not in loud[0], loud[0]
    assert "in 1 attempt(s)" in loud[0], "it claimed looks it never took"


def test_a_live_run_still_names_the_phone_as_the_likely_cause(caplog):
    with caplog.at_level(logging.WARNING, logger="pogobot"):
        _identify([AccountView(available=False)])
    loud = [m for m in caplog.messages if "SWITCHING IS DISABLED" in m]
    assert loud and "restart the bot" in loud[0] and "--dry-run" not in loud[0]


def test_zero_attempts_still_takes_one_look():
    """The floor is defensive: a caller that asked for none would read nothing at all and
    then report an unreadable panel it never looked at."""
    (reader, account, roster), tree, act = _identify(
        [_closed(), panel(active="TrainerTwo")], attempts=0)
    assert account == "TrainerTwo"
    assert tree.reads == 2


def test_a_panel_a_previous_attempt_left_open_is_read_where_it_stands():
    """The launcher TOGGLES the panel. `identify_account` leaves it up on two paths - an
    unavailable second read, and no `close_norm` located - and both are inside the retry's
    own failure criterion, so attempt 2 used to spend itself SHUTTING the panel attempt 1
    had opened. `Switching.step` and `_verify` both guard their launcher taps this way."""
    from pogobot.accounts import identify_account
    act = _Act()
    already_open = replace(panel(active="TrainerTwo"), close_norm=None)
    assert already_open.panel_open
    reader = FakeTreeReader([already_open])
    out = identify_account(reader, act, settle=0)
    assert out is not None and out.active is not None and out.active.name == "TrainerTwo"
    assert act.applied == [], "the launcher tap closed the panel it came to read"


def test_a_replay_never_preflights():
    """A replay exists to reproduce a recorded session. A preflight is a state the
    recording never had: it takes the screen on the first on-map frame and, while
    `desired_state` gives PREFLIGHT the screen, ignores the very encounters and popups the
    operator opened the recording to look at. There is nothing for it to do either - the
    actuator is a NullActuator and there is no device to zoom, toggle or route."""
    a = build_parser().parse_args(["--replay", "logs/trace.jsonl"])
    assert config_from_args(a).preflight is False
    assert config_from_args(build_parser().parse_args([])).preflight is True
