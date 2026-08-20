"""The PGSharp accounts panel wedged the bot, and nothing on screen could see it.

Measured in logs/trace.jsonl: 599 of one run's 1520 frames (39%) in a RECOVERING x47 ->
SCANNING x1 cycle repeating to the end of the run, every frame reading
screen=Menu@0.941, map=False, close=False, one BACK per cycle (~13 in total) and no
recovery. The screen was PGSharp's own accounts panel - a native Android overlay drawn
over the game, listing accounts with an "OK" control top-left. Read live at the bot's own
stream resolution, real perception reports screen=Menu@0.95, on_map=False,
in_overlay=False, x_button=False, find_close_button=None, detections=[] - so every rung
RECOVERING had was blind to it: BACK does not dismiss this panel, and the optical locator
has nothing to find.

The view tree does see it. `UiTreeReader.read()` on the stuck phone returned
available=True, panel_open=True, close_norm=(0.0634, 0.1021) - and tapping that control
closed the panel. The bot could not consult it because `Runner._refresh_accounts` read the
tree only while SWITCHING, and this run contained no SWITCHING at all.

Two things are covered here: `fsm.Recovering._panel_close`, the rung that taps that
located control, and `fsm.Recovering.on_timeout`'s escalation to `effects.RestartApp` -
force-stop plus relaunch - for the case where no button, located or otherwise, is the
answer. `tests/test_fsm_livelocks.py` still owns the 120s watchdog gate itself; this file
owns what now happens once that gate fires.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pytest

from pogobot import fsm
from pogobot import runner as runner_mod
from pogobot.accounts import AccountView, FakeTreeReader
from pogobot.actions import ADB_TIMEOUT, Actuator
from pogobot.config import Config
from pogobot.effects import (
    Back,
    BotState,
    Halt,
    IntentOutcome,
    RestartApp,
    Tap,
    Transition,
    is_actuation,
)
from pogobot.frames import Frame
from tests.factories import obs
from tests.test_switch_runner import ROSTER, closed_panel, make_runner
from tests.test_switching import panel

#: The frame signature the whole livelock ran on, as the trace recorded it: a confident
#: Menu, no map, and - the part that matters - no close button for the optical rung to
#: aim at.
PANEL_FRAME = obs(on_map=False, screen="Menu", conf=0.941, close_xy=None)

#: The control the live tree reported for the stuck panel, and the one that closed it.
CLOSE_NORM = (0.0634, 0.1021)


def _open_panel(**kw) -> AccountView:
    """The tree's account panel as it reads while it is covering the game."""
    return replace(panel(), close_norm=CLOSE_NORM, **kw)


def recovering(accounts=None, cfg=None, now=1_000.0, **kw) -> fsm.Context:
    c = fsm.Context(cfg=cfg or Config(), state=BotState.RECOVERING,
                    state_since=now, now=now)
    c.last_map_ts = now
    c.accounts = accounts
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def kinds(effects, t):
    return [e for e in effects if isinstance(e, t)]


# ----------------------------------------------------- the located panel close rung

def test_recovering_taps_the_close_control_the_tree_located():
    out = fsm.step(PANEL_FRAME, recovering(_open_panel()))
    taps = kinds(out, Tap)
    assert taps, "the tree named a close control and RECOVERING must use it"
    assert (taps[0].x, taps[0].y) == CLOSE_NORM
    assert taps[0].budget == "close"


def test_the_located_close_outranks_the_blind_back():
    """Ordering, and the whole reason for it. BACK is measured NOT to dismiss this panel,
    and it is not a free first try: `config.Timings.switch_clear_max` records a BACK storm
    that left PGSharp showing NEITHER account with an asterisk - the game logged out of
    both profiles and had to be recovered by hand. A coordinate the tree just reported is
    both more likely to work and strictly safer, so it goes first."""
    out = fsm.step(PANEL_FRAME, recovering(_open_panel()))
    assert kinds(out, Tap)
    assert not kinds(out, Back), "no BACK may be spent while the tree says a panel is up"


@pytest.mark.parametrize("view, why", [
    (None, "no read has happened yet"),
    (replace(_open_panel(), available=False), "the dump could not be read"),
    (replace(_open_panel(), panel_open=False), "the panel is shut"),
    (replace(_open_panel(), close_norm=None), "the dump named no close control"),
])
def test_no_tap_without_a_located_control(view, why):
    """Buttons are LOCATED, never assumed - the rule the v1 menu livelock exists to
    enforce. Each of these is a way the tree can decline to name one, and each must fall
    through to the ladder that was already there rather than invent a coordinate.

    Each case is built to isolate ONE guard. A bare `AccountView(available=False)` also
    defaults `panel_open=False` and `close_norm=None`, so the two later guards answer it
    first and the `available` guard is never the reason it passes - deleting that guard
    left the whole suite green. `available=False` carrying a panel and a close control is
    unreachable in the field, which is the point: unrealistic-by-construction is how a
    guard gets tested on its own (see `panel_with_autowalk` for the same device)."""
    out = fsm.step(PANEL_FRAME, recovering(view))
    assert not kinds(out, Tap), f"tapped something even though {why}"
    assert kinds(out, Back), "the pre-existing BACK rung must still be reachable"


def test_the_rung_is_paced_and_cannot_fire_every_frame():
    """`ctx.ready` on the shared "close" budget, exactly like the optical close rung it
    sits beside, so a stuck panel cannot become a tap storm of its own."""
    c = recovering(_open_panel())
    c.last_action["close"] = c.now
    assert fsm.step(PANEL_FRAME, c) == []


def test_a_visit_that_already_acted_does_not_tap_again():
    """One located tap per RECOVERING visit - the same one-action budget the BACK rung
    uses - which is what bounds how many taps a single (possibly stale) view can spend."""
    c = recovering(_open_panel(), taps_in_state=1)
    assert fsm.step(PANEL_FRAME, c) == []


def test_a_withheld_tap_still_does_not_release_the_back_rung():
    """The empty answer is deliberate: while the tree says a PGSharp panel is in front of
    us, pacing this rung out must not promote BACK to being tried instead."""
    c = recovering(_open_panel())
    c.last_action["close"] = c.now
    assert not kinds(fsm.step(PANEL_FRAME, c), Back)


def test_the_map_still_wins_over_everything():
    """A confirmed map means the panel is gone, whatever a stale view says.

    Asserted against the HANDLER rather than `fsm.step`, because `fsm.desired_state`
    answers `on_map` before any handler runs: driven through `step` this passes with the
    rungs in any order at all, including with the panel close moved above the map check.
    """
    out = fsm.Recovering().step(obs(on_map=True), recovering(_open_panel()))
    tr = kinds(out, Transition)
    assert tr and tr[0].to is BotState.SCANNING
    assert not kinds(out, Tap)


def test_the_handler_writes_nothing_to_the_context():
    """The FSM is pure: (Observation, Context) -> list[Effect]. Only the runner mutates
    the context, which is what keeps a dry run and a live run on the same trajectory.

    Both branches, and each against its OWN context. `on_timeout` is the branch that reads
    `app_restarts`/`app_restart_ts` and answers with a RestartApp, and reaching it needs
    `state_since` in the past - `recovering()` stamps it at `now`, so without that the
    second call is just `step` again, which the first call already covered."""
    import copy
    c = recovering(_open_panel())
    before = copy.deepcopy(c.__dict__)
    fsm.step(PANEL_FRAME, c)
    assert c.__dict__ == before

    t = recovering(_open_panel(), now=10_000.0, last_map_ts=0.0, state_since=0.0)
    before_timeout = copy.deepcopy(t.__dict__)
    assert kinds(fsm.step(PANEL_FRAME, t), RestartApp), "meant to exercise on_timeout"
    assert t.__dict__ == before_timeout


# ------------------------------------------------------- the livelock, end to end

def _drive_recovering(seconds: float, reader=None, cfg=None, dt: float = 0.125,
                      last_map_ts=None):
    """Tick the real Runner through a screen that never changes, at the cadence the live
    loop actually runs (8 inference fps), starting in RECOVERING.

    A single `fsm.step` cannot see this bug at all: the livelock is a CYCLE - RECOVERING
    times out into SCANNING, SCANNING sees no map and comes straight back - and it is the
    per-visit reset of `taps_in_state` that made one BACK per cycle. Same reasoning as
    tests/test_switch_clear.py's own drive loop.
    """
    cfg = cfg or Config()
    r = make_runner(cfg, tree_reader=reader, roster=ROSTER)
    r.stats.account = "TrainerOne"
    r.ctx.now = 1_000.0
    r.ctx.last_map_ts = r.ctx.now if last_map_ts is None else last_map_ts
    r.enter_state(BotState.RECOVERING, IntentOutcome.CARRIED, "wedged behind the panel")
    t0 = r.ctx.now
    while r.ctx.now - t0 < seconds and r.ctx.state is not BotState.HALTED:
        r.ctx.now = round(r.ctx.now + dt, 6)
        r._refresh_accounts(r.ctx.now)
        r.apply(fsm.step(PANEL_FRAME, r.ctx), PANEL_FRAME)
    return r


def test_the_recovering_scanning_loop_is_broken_by_the_located_close():
    """The regression itself, in the shape the trace recorded it."""
    r = _drive_recovering(60.0, reader=FakeTreeReader([_open_panel()]))
    taps = [e for e in r.actuator.applied if isinstance(e, Tap)]
    backs = [e for e in r.actuator.applied if isinstance(e, Back)]
    assert taps, "the panel was open for a minute and nothing was ever tapped"
    assert all((t.x, t.y) == CLOSE_NORM for t in taps)
    assert not backs, f"{len(backs)} BACK press(es) spent on a panel BACK does not close"


def test_without_the_tree_the_run_reproduces_the_measured_back_storm():
    """Red-green: the same minute with no tree reader - which is exactly the state the
    bot was in, since `_refresh_accounts` never read outside SWITCHING - is the livelock
    as measured, one BACK per cycle and nothing else."""
    r = _drive_recovering(60.0, reader=None)
    backs = [e for e in r.actuator.applied if isinstance(e, Back)]
    assert not [e for e in r.actuator.applied if isinstance(e, Tap)]
    assert len(backs) >= 5, (
        f"only {len(backs)} BACK press(es) - the drive loop is not reproducing the "
        f"RECOVERING -> SCANNING cycle this regression lives in")


# --------------------------------------------------------------- the tree refresh

def test_recovering_refreshes_the_tree_at_its_own_slower_cadence():
    """RECOVERING must be able to see the panel, and must not pay the switch's price for
    it: the dump blocks the loop for ~1s, and RECOVERING is entered briefly after every
    ordinary timeout, not only when the bot is wedged."""
    reader = FakeTreeReader([_open_panel()])
    r = make_runner(tree_reader=reader)
    r.ctx.state = BotState.RECOVERING
    real = r._real
    r._refresh_accounts(real)
    assert reader.reads == 1 and r.ctx.accounts is not None
    r._refresh_accounts(real + runner_mod.ACCOUNTS_REFRESH + 0.1)
    assert reader.reads == 1, "the switch cadence must not apply here"
    r._refresh_accounts(real + runner_mod.RECOVER_ACCOUNTS_REFRESH + 0.1)
    assert reader.reads == 2


def test_the_recovering_throttle_is_the_longer_of_the_two():
    assert runner_mod.RECOVER_ACCOUNTS_REFRESH > runner_mod.ACCOUNTS_REFRESH


def test_a_slow_read_still_buys_a_whole_interval_of_live_frames():
    """The throttle is stamped from when the read FINISHED, not when it started.

    `tree_reader.read()` blocks the run loop's own thread. Start-to-start, a read that
    outran its own interval made the next one eligible the instant it returned, so the
    loop could spend most of its wall clock blind - not perceiving, not reading keys, not
    servicing a stop request - and `map_stale_since`, which the restart ladder reads,
    counts throughout. End-to-start makes the interval a floor on time spent looking."""
    class _Slow:
        def __init__(self):
            self.reads = 0

        def read(self):
            self.reads += 1
            time.sleep(0.05)
            return _open_panel()

    reader = _Slow()
    r = make_runner(tree_reader=reader)
    r.ctx.state = BotState.RECOVERING
    real = r._real
    r._refresh_accounts(real)
    assert reader.reads == 1
    r._refresh_accounts(real + runner_mod.RECOVER_ACCOUNTS_REFRESH + 0.02)
    assert reader.reads == 1, "the read's own cost must count against the next interval"
    r._refresh_accounts(real + runner_mod.RECOVER_ACCOUNTS_REFRESH + 0.3)
    assert reader.reads == 2


def test_a_recovering_read_leaves_the_switch_bookkeeping_alone():
    """`_last_seen_active` is the record of who the tree named during THIS switch attempt,
    and `switch_autowalk_active` is a colour read off a menu only SWITCHING opens. Neither
    is a question RECOVERING asked, and answering them from the wrong screen would spend
    evidence `_on_switch_failed` reads."""
    r = make_runner(tree_reader=FakeTreeReader([panel(active="TrainerOne")]))
    r.ctx.state = BotState.RECOVERING
    r._refresh_accounts(r._real)
    assert r.ctx.accounts is not None
    assert r._last_seen_active is None


def test_a_switch_still_refreshes_exactly_as_it_did():
    reader = FakeTreeReader([panel(active="TrainerTwo")])
    r = make_runner(tree_reader=reader)
    r.ctx.state = BotState.SWITCHING
    real = r._real
    r._refresh_accounts(real)
    r._refresh_accounts(real + 0.2)
    assert reader.reads == 1
    r._refresh_accounts(real + runner_mod.ACCOUNTS_REFRESH + 0.1)
    assert reader.reads == 2
    assert r._last_seen_active == "TrainerTwo"


def test_a_closed_panel_read_from_recovering_taps_nothing():
    """What the tree reports on an ordinary run - available, panel shut, no rows - must
    leave the ladder exactly as it was."""
    r = _drive_recovering(30.0, reader=FakeTreeReader([closed_panel()]))
    assert not [e for e in r.actuator.applied if isinstance(e, Tap)]
    assert [e for e in r.actuator.applied if isinstance(e, Back)]


# ------------------------------------------------------------------- the restart

def test_restart_renders_to_one_adb_command_naming_the_configured_package():
    """One invocation, like DoubleTapDrag: a stop that succeeds and a start that never
    runs leaves the phone on the Android launcher, which is worse than being stuck."""
    cfg = Config(app_package="com.example.game", app_activity="com.example.ui.MainActivity")
    cmd = Actuator((1080, 2340), dry_run=True).render(
        RestartApp(cfg.app_package, cfg.app_activity, "recover: restart"))
    assert cmd is not None
    assert cmd.argv[:2] == ("adb", "shell")
    assert len(cmd.argv) == 3, f"expected ONE shell string, got {cmd.argv}"
    shell = cmd.argv[2]
    assert shell.count(cfg.app_package) == 2      # force-stop, then start
    assert "force-stop" in shell and "am start" in shell
    assert cfg.app_activity in shell


def test_the_default_package_is_the_one_verified_on_the_device():
    """PGSharp is a modded build of the SAME package, and with the panel up
    `mCurrentFocus` is still this activity - the panel is an overlay window of this
    process, which is why ending the process ends the panel."""
    cfg = Config()
    assert cfg.app_package == "com.nianticlabs.pokemongo"
    assert cfg.app_activity.endswith("unity.UnityMainActivity")


def test_the_settle_stays_inside_the_adb_timeout():
    """The settle is spent on the device, inside the one command, and ADB_TIMEOUT bounds
    the whole invocation - a settle past it turns every restart into a timed-out command
    and trips the actuator's own failure breaker."""
    assert RestartApp("p", "a", "r").settle_ms / 1000.0 < ADB_TIMEOUT


def test_the_actuator_dispatches_a_restart_at_all():
    """`is_actuation` is the gate `Actuator.apply` checks first; an effect missing from it
    is silently dropped, with `apply` returning False and nothing ever reaching adb."""
    assert is_actuation(RestartApp("p", "a", "r"))
    a = Actuator((1080, 2340), dry_run=True)
    assert a.apply(RestartApp("p", "a", "r"), now=1_000.0) is True
    assert a.stats()["by_budget"] == {"restart": 1}


def test_the_trace_names_the_new_effect(tmp_path):
    """`_write_trace` records effects by type name; a new one must not break the line the
    whole diagnosis of this bug was read out of."""
    r = make_runner(trace_path=tmp_path / "trace.jsonl")
    r._write_trace(PANEL_FRAME, [RestartApp("p", "a", "r")])
    r._trace.flush()
    rec = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[-1])
    assert rec["eff"] == ["RestartApp"]


# --------------------------------------------------------- the restart escalation

def test_a_wedged_run_restarts_the_app_before_it_halts():
    out = fsm.step(PANEL_FRAME, recovering(_open_panel(), now=10_000.0, last_map_ts=0.0,
                                           state_since=0.0))
    restarts = kinds(out, RestartApp)
    assert restarts, "past the watchdog, a restart is what is left to try"
    assert restarts[0].package == Config().app_package
    assert not kinds(out, Halt)


def test_the_restart_hands_control_back_so_it_cannot_re_fire_every_tick():
    """`on_timeout` runs on EVERY tick past the budget - only entering a state restarts
    that clock - so the answer has to include the hand-off, and the hand-off has to come
    AFTER the restart: `Runner.apply` walks the list in order, and a Transition first
    would dispatch the restart with the state already changed."""
    out = fsm.step(PANEL_FRAME, recovering(_open_panel(), now=10_000.0, last_map_ts=0.0,
                                           state_since=0.0))
    names = [type(e).__name__ for e in out]
    assert "RestartApp" in names and "Transition" in names, names
    assert names.index("RestartApp") < names.index("Transition")


def test_the_grace_period_stops_a_loading_app_burning_the_whole_budget():
    """A force-stop plus relaunch is a cold start of tens of seconds, and for every one of
    them the map is still missing - the same condition that authorised the restart. Sixty
    seconds of RECOVERING timing out every 6s must still be ONE restart."""
    r = _drive_recovering(60.0, reader=None, dt=0.5, last_map_ts=0.0)
    restarts = [e for e in r.actuator.applied if isinstance(e, RestartApp)]
    assert len(restarts) == 1, f"{len(restarts)} restarts in 60s - the grace is not held"


def test_the_budget_is_bounded_and_the_run_then_halts():
    """A crash-looping app must not be restarted forever."""
    cfg = Config()
    r = _drive_recovering(400.0, reader=None, dt=0.5, last_map_ts=0.0)
    restarts = [e for e in r.actuator.applied if isinstance(e, RestartApp)]
    assert len(restarts) == cfg.max_app_restarts
    assert r.ctx.state is BotState.HALTED
    assert r._halt_reason
    # The halt line is the one artifact an operator reads afterwards - this whole bug was
    # diagnosed out of one. `ctx.app_restarts` is the count SPENT, and this branch is only
    # reached when it equals the budget, so a message shaped "N of N left to spend" says
    # the opposite of the truth and sends the reader looking for a bug in the gate.
    assert f"{cfg.max_app_restarts} app restart(s) spent" in r._halt_reason, r._halt_reason
    assert "left to spend" not in r._halt_reason, r._halt_reason


def test_the_budget_is_read_from_the_config():
    """A handler that hardcoded the number would pass the test above and ignore this one."""
    cfg = Config(max_app_restarts=1)
    r = _drive_recovering(400.0, reader=None, dt=0.5, cfg=cfg, last_map_ts=0.0)
    assert len([e for e in r.actuator.applied if isinstance(e, RestartApp)]) == 1
    assert r.ctx.state is BotState.HALTED


def test_the_grace_holds_the_whole_ladder_not_only_the_next_restart():
    """A restart in flight must silence `step` too, not just defer the next restart
    decision. `on_timeout` hands control to SCANNING, which sees no map and comes straight
    back, so the grace is ~15 more RECOVERING visits - each with `taps_in_state` freshly
    zeroed, and each of which used to spend a BACK into a Unity cold start."""
    c = recovering(_open_panel(), now=10_000.0, last_map_ts=0.0,
                   app_restarts=1, app_restart_ts=10_000.0 - 5.0)
    assert fsm.step(PANEL_FRAME, c) == []


def test_nothing_is_pressed_while_the_restarted_app_is_still_loading():
    """The same hold through the real loop, which is the only place the cycle exists.

    `config.Timings.switch_clear_max` records where unbounded BACK into this app ends: ~90
    presses into a screen that would not clear left PGSharp showing NEITHER account with
    an asterisk - the game logged out of both profiles and had to be recovered by hand. A
    cold-starting game is exactly that screen, and the bot is the one that caused it."""
    r = _drive_recovering(80.0, reader=FakeTreeReader([_open_panel()]), dt=0.5,
                          last_map_ts=0.0)
    applied = r.actuator.applied
    assert any(isinstance(e, RestartApp) for e in applied), "no restart to wait out"
    first = next(i for i, e in enumerate(applied) if isinstance(e, RestartApp))
    after = applied[first + 1:]
    assert not after, f"{len(after)} action(s) taken into a cold-starting game: {after}"


def test_a_restart_opens_a_settle_window_like_every_other_actuation():
    """`ctx.ready` refuses to act while `settle_until` is ahead of `ctx.now`, and a
    force-stop plus relaunch is by far the longest UI transition the bot can cause.
    Omitted from the settle tuple in `Runner.apply`, a restart would be the one actuation
    the very next tick is free to tap straight through."""
    r = make_runner()
    r.ctx.state = BotState.RECOVERING
    r.ctx.settle_until = 0.0
    r.apply([RestartApp("p", "a", "r")], PANEL_FRAME)
    assert r.ctx.settle_until == r.ctx.now + r.cfg.timings.ui_settle


def test_a_restart_the_actuator_refused_is_not_charged_to_the_budget():
    """`Recovering.on_timeout` is pure and cannot know whether its RestartApp reached the
    device. Counting a refused one would spend a restart that never happened - and stamp a
    90s grace period waiting for an app nobody restarted. Same reasoning as
    `switch_zoom_reps` and `switch_clear_presses`."""
    from tests.test_switch_zoom import _FlakyAct
    r = make_runner()
    r.actuator = _FlakyAct([False, True])
    r.ctx.state = BotState.RECOVERING
    e = RestartApp("p", "a", "r")

    r.apply([e], PANEL_FRAME)
    assert r.ctx.app_restarts == 0 and r.ctx.app_restart_ts == 0.0

    r.apply([e], PANEL_FRAME)
    assert r.ctx.app_restarts == 1 and r.ctx.app_restart_ts == r.ctx.now


class _MapSrc:
    """A capture source that hands back real frames, so `run()` reaches perception. Local,
    like every other runner-shaped fake in this suite."""

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
        return obs(on_map=True, seq=frame.seq, ts=frame.ts)


def test_a_confirmed_map_is_what_refills_the_budget():
    """"Consecutive", not "per run": a restart that actually worked has proved itself and
    must not leave the rest of the run one restart poorer - while an app that crash-loops
    never shows a map and so can never earn another one.

    Driven through the real `Runner.run()` loop, because the claim is about the loop: the
    reset lives beside `last_map_ts` in the read loop, and asserting it anywhere else
    would only be re-stating the line rather than running it."""
    r = make_runner(Config(infer_fps=1_000.0))
    r.source = _MapSrc()
    r.source.runner = r
    r.perceptor = _SeesTheMap()
    r.ctx.app_restarts = 2
    r.ctx.app_restart_ts = 1_234.0

    assert r.run() == 0
    assert r._ticks >= 1, "the loop never completed a tick, so nothing was exercised"
    assert r.ctx.app_restarts == 0, "a confirmed map did not refill the restart budget"
    assert r.ctx.app_restart_ts == 0.0, (
        "a confirmed map refilled the budget but left the cold-start hold in force, so "
        "the ladder stays silent on a screen the bot can already see")


def test_a_confirmed_map_also_ends_the_cold_start_hold():
    """The second half of the chain: clearing `app_restart_ts` actually releases the
    ladder. That the RUNNER clears it on a confirmed map is guarded by
    `test_a_confirmed_map_is_what_refills_the_budget`, which drives the real loop - this
    one would pass even if the runner never cleared it, and is here to prove the field is
    what holds the ladder rather than to guard the line that writes it.

    Both halves are needed because `app_restart_ts` was the one restart counter a map did
    not clear, so the ladder stayed silent for the rest of the 90s window even though the
    screen was already back: measured with the map confirmed 30s after the restart, `step`
    pressed nothing at 35s, 60s and 89s and only resumed at 91s."""
    held = recovering(_open_panel(), now=10_000.0, last_map_ts=0.0,
                      app_restarts=1, app_restart_ts=10_000.0 - 5.0)
    assert fsm.step(PANEL_FRAME, held) == [], "the hold is not in force to begin with"

    # Exactly what the read loop does on a confirmed map - see Runner.run.
    cleared = recovering(_open_panel(), now=10_000.0, last_map_ts=0.0,
                         app_restarts=0, app_restart_ts=0.0)
    taps = [e for e in fsm.step(PANEL_FRAME, cleared) if isinstance(e, Tap)]
    assert taps, "the ladder is still held after the map came back"
    assert (taps[0].x, taps[0].y) == CLOSE_NORM


def test_the_restart_counters_survive_a_state_entry():
    """RECOVERING is entered and left every few seconds while the bot is wedged, so a
    per-entry reset (runner._RESET_ON_ENTRY) would make the bound count to one forever."""
    assert "app_restarts" not in runner_mod._RESET_ON_ENTRY
    assert "app_restart_ts" not in runner_mod._RESET_ON_ENTRY
    r = make_runner()
    r.ctx.app_restarts, r.ctx.app_restart_ts = 2, 1_234.0
    r.enter_state(BotState.RECOVERING, IntentOutcome.CARRIED, "again")
    assert (r.ctx.app_restarts, r.ctx.app_restart_ts) == (2, 1_234.0)
