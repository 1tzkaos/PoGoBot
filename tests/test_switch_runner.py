"""Account switching as the runner drives it: refresh, trigger, select, confirm.

Each of these has a failure mode that a green suite would otherwise hide:

  * the UI-tree read blocks for roughly a second, so it must never happen per frame, and a
    trigger that can never be satisfied must not turn into a per-frame dump either;
  * a switch may only begin from SCANNING with the map confirmed - starting one mid
    encounter abandons a Pokemon mid-throw;
  * only a CONFIRMED switch rolls the session over, so an attempt that times out cannot
    invent a session row or reset a counter;
  * `spins_exhausted` must describe the account we are actually on. Nothing asserted that
    before, which is how one wrong argument at that line could read as good standing;
  * and the decision to switch must survive the shape the tree actually has during a run -
    `available=True, panel_open=False, rows=0` - because the PGSharp panel is closed
    except while SWITCHING is driving it. Deciding from live rows made the whole feature
    inert on the phone: 45 SCANNING ticks, 0 switches.
"""
import json
import logging
import time

import numpy as np
import pytest

from pogobot import fsm
from pogobot import runner as runner_mod
from pogobot.accounts import AccountView, FakeTreeReader
from pogobot.config import DEFAULT
from pogobot.effects import BotState, IntentOutcome, Tap, Transition
from pogobot.frames import Frame
from pogobot.quota import SpinQuota
from tests.factories import obs
from tests.test_switching import panel


def _frame_of(seq):
    return Frame(seq=seq, ts=time.perf_counter(),
                 bgr=np.zeros((1280, 590, 3), dtype=np.uint8))


# Local fakes, matching the pattern every other runner test in this suite uses
# (tests/test_pause.py, tests/test_stats.py). There is deliberately no shared helper:
# each file states the device it is pretending to have.
class _Act:
    def __init__(self):
        self.applied = []

    def apply(self, effect, now=None):
        self.applied.append(effect)
        return True

    def healthy(self):
        return True

    def stats(self):
        return {"sent": len(self.applied)}

    def close(self):
        pass


class _Src:
    def read(self):
        return None

    def healthy(self):
        return True

    def release(self):
        pass


class _Dash:
    """Only the part of the dashboard that matters here: it holds the counters."""

    def __init__(self, stats):
        self.stats = stats


def make_runner(cfg=DEFAULT, **kw):
    return runner_mod.Runner(cfg, _Src(), _Act(), perceptor=None, display=False, **kw)


#: The map is confirmed by `obs()` alone: factories default to screen="Overworld" at 0.99,
#: and `on_map` is `map_ball or screen.is_("Overworld", 0.60)`. So "off the map" needs a
#: different screen - `obs(on_map=False)` is still on the map.
def off_map():
    return obs(screen="Menu", conf=0.99)


def closed_panel():
    """What the tree reports during a normal run, measured on the device: the overlay is
    there, its panel is shut, and no account row is visible to anyone."""
    return AccountView(rows=(), launcher_norm=(0.12, 0.05), accounts_tab_norm=None,
                       close_norm=None, available=True, panel_open=False)


ROSTER = ("TrainerOne", "TrainerTwo")


# ------------------------------------------------------------------ selection

def _selector(quota=None, roster=ROSTER, current="TrainerOne", **kw):
    """A runner asked one question: who is next after `current`?

    Both inputs are the ones production actually has - the roster enumerated at startup
    and the account the session counters belong to. Selection never sees a live view: the
    panel is shut whenever this is asked, so a view could only ever say "no accounts".
    """
    r = make_runner(quota=quota, tree_reader=FakeTreeReader([closed_panel()]), roster=roster,
                    **kw)
    r.stats.account = current
    return r


def test_selection_returns_the_next_usable_account(tmp_path):
    q = SpinQuota(tmp_path / "s.jsonl", limit=10)
    assert _selector(q).choose_next_account() == "TrainerTwo"


def test_selection_costs_no_tree_read():
    """It is asked from SCANNING, where a ~1s blocking dump is a visible stall in an 8fps
    loop - and could not answer anything anyway."""
    r = _selector()
    r.choose_next_account()
    assert r.tree_reader.reads == 0


def test_selection_skips_an_exhausted_account(tmp_path):
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerTwo")
    assert _selector(q).choose_next_account() is None


def test_when_all_accounts_are_capped_it_picks_the_soonest_to_reset(tmp_path):
    now = time.time()
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerOne", now=now - 1 * 3600)      # frees in 23h
    q.record("TrainerTwo", now=now - 23 * 3600)     # frees in 1h
    assert _selector(q).choose_next_account() == "TrainerTwo"


def test_all_capped_but_the_current_one_frees_first_stays_put(tmp_path):
    """Otherwise two capped accounts trade places forever.

    The quota trigger stays due for as long as both are capped, so a selection that always
    names the other account switches, re-triggers, switches back, and the bot spends the
    wait driving the overlay instead of catching.
    """
    now = time.time()
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerOne", now=now - 23 * 3600)     # frees in 1h
    q.record("TrainerTwo", now=now - 1 * 3600)      # frees in 23h
    assert _selector(q).choose_next_account() is None


def test_an_account_with_room_left_is_never_swapped_for_a_capped_one(tmp_path):
    """`usable == []` says every ALTERNATIVE is capped - it says nothing about where we are.

    "Whose oldest spin ages out first" is a meaningless ranking for an account that can
    still spin, and answering it under a clock rotation logs out of an account with room
    onto one that will refuse every stop for hours. Not reachable through the quota
    trigger, where `spins_exhausted` already guarantees the current account is capped,
    which is exactly why nothing else here catches it.
    """
    now = time.time()
    q = SpinQuota(tmp_path / "s.jsonl", limit=2)
    q.record("TrainerOne", now=now - 5 * 3600)       # 1/2 used: room left, frees in 19h
    q.record("TrainerTwo", now=now - 20 * 3600)      # 2/2 used: capped, frees in 4h
    q.record("TrainerTwo", now=now - 20 * 3600)
    assert _selector(q).choose_next_account() is None


def test_a_session_that_never_learned_its_name_is_not_a_round_robin_origin(tmp_path):
    """We do not know where we are, so there is no origin to rotate from and no way to
    tell a capped account from the one that was working. It is also the state a failed
    switch leaves behind, where the wrong guess logs out of an account that was fine."""
    q = SpinQuota(tmp_path / "s.jsonl", limit=10)
    assert _selector(q, current=None).choose_next_account() is None


def test_one_account_is_never_a_switch_target(tmp_path):
    q = SpinQuota(tmp_path / "s.jsonl", limit=10)
    assert _selector(q, roster=("TrainerOne",)).choose_next_account() is None


def test_an_empty_roster_is_never_a_guess():
    """Nothing was ever enumerated, so there is no second account to name."""
    assert _selector(roster=()).choose_next_account() is None


def test_an_account_the_roster_does_not_contain_is_no_origin_to_rotate_from():
    assert _selector(current="SomebodyElse").choose_next_account() is None


def test_the_quota_rules_apply_to_the_cached_roster_too(tmp_path):
    """The roster changes where the names come from, nothing about which one is usable."""
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerTwo")
    assert _selector(q).choose_next_account() is None


# ------------------------------------------------------------------ refresh cost

def test_the_tree_is_never_read_per_frame_outside_a_switch():
    reader = FakeTreeReader([panel()])
    r = make_runner(tree_reader=reader)
    r.ctx.state = BotState.SCANNING
    for _ in range(10):
        r._refresh_accounts(r._real)
    assert reader.reads == 0
    assert r.ctx.accounts is None


def test_during_a_switch_the_tree_is_refreshed_but_throttled():
    reader = FakeTreeReader([panel()])
    r = make_runner(tree_reader=reader)
    r.ctx.state = BotState.SWITCHING
    real = r._real
    r._refresh_accounts(real)
    r._refresh_accounts(real + 0.2)
    assert reader.reads == 1, "the ~1s dump must not run twice in one settle window"
    assert r.ctx.accounts is not None
    r._refresh_accounts(real + runner_mod.ACCOUNTS_REFRESH + 0.1)
    assert reader.reads == 2


def test_a_failed_read_leaves_the_view_alone_rather_than_crashing_the_loop():
    class _Boom:
        def read(self):
            raise RuntimeError("adb went away")

    r = make_runner(tree_reader=_Boom())
    r.ctx.state = BotState.SWITCHING
    r._refresh_accounts(r._real)
    assert r.ctx.accounts is None


def test_an_actuation_during_a_switch_invalidates_the_cached_view():
    """Tapping the launcher TOGGLES the overlay, so deciding twice from one view closes
    the panel it just opened."""
    r = make_runner(tree_reader=FakeTreeReader([panel()]))
    r.ctx.state = BotState.SWITCHING
    r.ctx.accounts = panel()
    r.apply([Tap(0.12, 0.05, "switch: open the PGSharp overlay", budget="switch")], obs())
    assert r.ctx.accounts is None


def test_a_stale_view_cannot_toggle_the_overlay_shut():
    """The same thing through the real handler: one view, several ticks spaced past
    `timings.switch_tap`, must produce exactly one launcher tap. Three taps means open,
    shut, open - and the switch then sits there until its timeout."""
    r = make_runner(tree_reader=FakeTreeReader([panel(open_=False)]))
    r.ctx.state = BotState.SWITCHING
    r.ctx.switch_target, r.ctx.switch_phase = "TrainerTwo", "open"
    r.ctx.accounts = panel(open_=False)
    taps = []
    for t in (0.0, 2.1, 4.2):
        r.ctx.now = 100.0 + t
        effects = fsm.step(obs(), r.ctx)
        r.apply(effects, obs())
        taps += [e for e in effects if isinstance(e, Tap)]
    assert [t.reason for t in taps] == ["switch: open the PGSharp overlay"]


def test_a_begun_switch_forgets_whatever_view_it_had():
    """Every tap in SWITCHING comes from a location the tree reported; a view read before
    the switch started describes a panel that was shut, and its coordinates are a guess by
    the time the handler acts. The handler waits while the view is None."""
    r = make_runner(tree_reader=FakeTreeReader([panel()]), roster=ROSTER)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.accounts = panel()
    r._begin_switch("TrainerTwo")
    assert r.ctx.accounts is None


def test_an_actuation_outside_a_switch_keeps_the_view():
    r = make_runner(tree_reader=FakeTreeReader([panel()]))
    r.ctx.state = BotState.SCANNING
    r.ctx.accounts = panel()
    r.apply([Tap(0.5, 0.6, "target pokemon")], obs())
    assert r.ctx.accounts is not None


# ------------------------------------------------------------------ triggers

def test_switching_stays_off_by_default():
    r = make_runner(tree_reader=FakeTreeReader([panel()]), roster=ROSTER)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.spins_exhausted = True
    r._maybe_switch(obs())
    assert r.ctx.state is BotState.SCANNING


def test_switching_is_never_started_outside_scanning():
    r = make_runner(DEFAULT.scaled(switch_on_quota=True),
                    tree_reader=FakeTreeReader([panel()]), roster=ROSTER)
    r.stats.account = "TrainerOne"
    for state in (BotState.ENCOUNTER, BotState.ROCKET, BotState.POKESTOP,
                  BotState.TARGETING, BotState.POPUP, BotState.RECOVERING):
        r.ctx.state = state
        r.ctx.spins_exhausted = True
        r._maybe_switch(obs())
        assert r.ctx.state is state


def test_a_switch_never_begins_without_the_map():
    """SCANNING is reachable without the map - Recovering gives up into it - and the first
    thing a switch does is tap an overlay we can only trust when the map is up."""
    reader = FakeTreeReader([panel()])
    r = make_runner(DEFAULT.scaled(switch_on_quota=True), tree_reader=reader, roster=ROSTER)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.spins_exhausted = True
    r._maybe_switch(off_map())
    assert r.ctx.state is BotState.SCANNING
    assert reader.reads == 0


def test_a_trigger_fires_from_the_cached_roster_while_the_panel_is_closed():
    """The bug the live run found. The tree only lists accounts while the panel is open,
    and the panel is closed for the whole run - so a decision that needed live rows never
    fired once in 3.5 minutes of SCANNING. The roster comes from the startup read instead,
    and costs no dump at all."""
    reader = FakeTreeReader([closed_panel()])
    r = make_runner(DEFAULT.scaled(switch_on_quota=True), tree_reader=reader, roster=ROSTER)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.accounts = closed_panel()          # what a live refresh would have left behind
    r.ctx.spins_exhausted = True
    r._maybe_switch(obs())
    assert r.ctx.state is BotState.SWITCHING
    assert r.ctx.switch_target == "TrainerTwo"
    assert r.ctx.switch_phase == "open"
    assert reader.reads == 0, "the decision must not pay for a ~1s blocking dump"


def test_no_tree_reader_means_no_switching():
    r = make_runner(DEFAULT.scaled(switch_on_quota=True), roster=ROSTER)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.spins_exhausted = True
    r._maybe_switch(obs())
    assert r.ctx.state is BotState.SCANNING


def test_a_trigger_that_can_never_be_satisfied_never_reads_the_tree(tmp_path):
    """`spins_exhausted` stays true for hours and selection can answer None the whole
    time. That used to force a ~1s blocking dump - throttled to every 30s, but still a
    stall in an 8fps loop - and it never learned anything: the panel is shut."""
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerTwo")                      # the only alternative is capped
    reader = FakeTreeReader([closed_panel()])
    r = make_runner(DEFAULT.scaled(switch_on_quota=True), quota=q, tree_reader=reader,
                    roster=ROSTER)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.spins_exhausted = True
    for _ in range(20):
        r._maybe_switch(obs())
    assert r.ctx.state is BotState.SCANNING
    assert reader.reads == 0


def test_the_clock_trigger_waits_out_its_first_interval():
    reader = FakeTreeReader([closed_panel()])
    r = make_runner(DEFAULT.scaled(switch_every_minutes=1.0), tree_reader=reader,
                    roster=ROSTER)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r._maybe_switch(obs())
    assert r.ctx.state is BotState.SCANNING, "a rotation is not due in the first tick"
    assert reader.reads == 0
    r.ctx.now = r._next_rotation + 0.1
    r._maybe_switch(obs())
    assert r.ctx.state is BotState.SWITCHING


# ------------------------------------------------------------------ the session

def test_a_confirmed_switch_closes_one_session_and_opens_another(tmp_path):
    stats_path = tmp_path / "sessions.jsonl"
    r = make_runner(stats_path=stats_path, tree_reader=FakeTreeReader([panel()]))
    r.stats.account = "TrainerOne"
    r.stats.encounters = 5
    r._on_switch_confirmed("TrainerTwo")
    assert r.stats.account == "TrainerTwo"
    assert r.stats.encounters == 0
    assert "TrainerOne" in stats_path.read_text()


def test_the_confirming_transition_is_what_rolls_the_session_over(tmp_path):
    stats_path = tmp_path / "sessions.jsonl"
    r = make_runner(stats_path=stats_path, tree_reader=FakeTreeReader([panel()]))
    r.stats.account = "TrainerOne"
    r.stats.encounters = 5
    r.ctx.state = BotState.SCANNING
    r._begin_switch("TrainerTwo")
    r.apply([Transition(BotState.SCANNING, IntentOutcome.CONFIRMED,
                        "logged into TrainerTwo")], obs())
    assert r.stats.account == "TrainerTwo" and r.stats.encounters == 0
    assert "TrainerOne" in stats_path.read_text()


def test_an_unconfirmed_switch_changes_no_counters(tmp_path):
    stats_path = tmp_path / "sessions.jsonl"
    r = make_runner(stats_path=stats_path, tree_reader=FakeTreeReader([panel()]))
    r.stats.account = "TrainerOne"
    r.stats.encounters = 5
    r._begin_switch("TrainerTwo")
    assert r.stats.account == "TrainerOne" and r.stats.encounters == 5
    assert not stats_path.exists()


def test_a_switch_that_times_out_writes_no_session_row(tmp_path):
    stats_path = tmp_path / "sessions.jsonl"
    r = make_runner(stats_path=stats_path, tree_reader=FakeTreeReader([panel()]))
    r.stats.account = "TrainerOne"
    r.stats.encounters = 5
    r.ctx.state = BotState.SCANNING
    r._begin_switch("TrainerTwo")
    r.apply([Transition(BotState.RECOVERING, IntentOutcome.EXPIRED, "switch timeout")],
            obs())
    assert r.stats.account == "TrainerOne" and r.stats.encounters == 5
    assert not stats_path.exists()


def test_only_a_confirmed_outcome_rolls_the_session_over(tmp_path):
    """Reaching the map is not the confirmation - the tree naming the target as active is.
    A future exit from SWITCHING that lands on the map without that evidence must not be
    allowed to split the books on it."""
    stats_path = tmp_path / "sessions.jsonl"
    r = make_runner(stats_path=stats_path, tree_reader=FakeTreeReader([panel()]))
    r.stats.account = "TrainerOne"
    r.stats.encounters = 5
    r.ctx.state = BotState.SCANNING
    r._begin_switch("TrainerTwo")
    r.apply([Transition(BotState.SCANNING, IntentOutcome.CARRIED, "map is back")], obs())
    assert r.stats.account == "TrainerOne" and r.stats.encounters == 5
    assert not stats_path.exists()


def test_an_unnamed_outgoing_session_is_still_recorded(tmp_path):
    """A run whose startup tree read failed still did the work. Dropping its row loses those
    hours from the history entirely - the counters do not carry into the new session
    either - and `close()` records an unnamed session anyway, so skipping it here would
    have the two paths disagree about the same object."""
    stats_path = tmp_path / "sessions.jsonl"
    r = make_runner(stats_path=stats_path, tree_reader=FakeTreeReader([panel()]))
    r.stats.encounters = 9
    assert r.stats.account is None
    r._on_switch_confirmed("TrainerTwo")
    rows = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
    assert [(row["account"], row["encounters"]) for row in rows] == [(None, 9)]


def test_a_switch_ends_the_restock_it_interrupts():
    """`restock_stops_at_start` is a mark on the OUTGOING counters. Left behind, `got` goes
    negative against the fresh session and can never reach the target, so the restock only
    ends when its 600s budget expires - logging a stop count below zero on the way out."""
    r = make_runner()
    r.stats.account = "TrainerOne"
    r.stats.stops_collected = 137
    r.ctx.restocking_until = r.ctx.now + 600.0
    r.ctx.restock_stops_at_start = 137
    r._on_switch_confirmed("TrainerTwo")
    assert not r.ctx.restocking
    assert r.ctx.restock_stops_at_start == 0


def test_the_dashboard_follows_the_new_session():
    r = make_runner(tree_reader=FakeTreeReader([panel()]))
    r.dashboard = _Dash(r.stats)
    r.stats.account = "TrainerOne"
    r._on_switch_confirmed("TrainerTwo")
    assert r.dashboard.stats is r.stats, "the TUI would keep rendering the dead session"


def test_a_pause_before_the_switch_does_not_swallow_the_new_uptime():
    """The new session inherits the run's paused total - the FSM clock is derived from it -
    so its start must be on that same clock, or its uptime reads short by every earlier
    pause and every rate it feeds is wrong."""
    r = make_runner(tree_reader=FakeTreeReader([panel()]))
    real = time.perf_counter()
    r.stats.paused_seconds = 100.0
    r.ctx.now = real - 100.0            # the invariant the loop maintains
    r.stats.account = "TrainerOne"
    r._on_switch_confirmed("TrainerTwo")
    assert r.stats.uptime(now=real + 60.0) == pytest.approx(60.0, abs=1.0)


def test_a_confirmed_rotation_restarts_the_clock():
    r = make_runner(DEFAULT.scaled(switch_every_minutes=1.0),
                    tree_reader=FakeTreeReader([panel()]))
    r.ctx.now = r._next_rotation + 0.1
    r._on_switch_confirmed("TrainerTwo")
    assert r._next_rotation == pytest.approx(r.ctx.now + 60.0)


def test_a_switch_does_not_carry_a_pending_tap_claim_into_the_overlay():
    """An Intent claims the screen changed BECAUSE of our tap. A switch takes minutes and
    puts a login screen up, so any answer after it is not evidence of anything."""
    r = make_runner(tree_reader=FakeTreeReader([panel()]))
    r.ctx.state = BotState.SCANNING
    r.ctx.intent = fsm.Intent(ts=r.ctx.now, target_name="pokestop", confidence=0.9,
                              tap_norm=(0.5, 0.6), xywhn=(0.5, 0.6, 0.1, 0.1),
                              expected=BotState.POKESTOP, frame_seq=1)
    r._begin_switch("TrainerTwo")
    assert r.ctx.intent is None


def test_the_abandoned_claim_is_logged_as_a_switch_not_a_pause(caplog):
    """The same line serves both callers, so a hardcoded reason has a switch reporting a
    pause that never happened - in the log a human reads to explain a missing tap."""
    r = make_runner(tree_reader=FakeTreeReader([panel()]))
    r.ctx.state = BotState.SCANNING
    r.ctx.intent = fsm.Intent(ts=r.ctx.now, target_name="pokestop", confidence=0.9,
                              tap_norm=(0.5, 0.6), xywhn=(0.5, 0.6, 0.1, 0.1),
                              expected=BotState.POKESTOP, frame_seq=1)
    with caplog.at_level(logging.INFO, logger="pogobot"):
        r._begin_switch("TrainerTwo")
    line = next(m for m in caplog.messages if "abandoning" in m)
    assert "switch" in line and "paused" not in line


# ------------------------------------------------------------------ failed switches

def _fail_a_switch(r, start, *, tap_login, view=None):
    """Drive one whole attempt through the real Runner and the real FSM, from the trigger
    to the state timeout and back to SCANNING. Returns whether an attempt was started.

    Two shapes of failure, both real:

      * `tap_login=False` - the overlay never opens. The handler taps the launcher, the
        panel stays shut, and the attempt dies without a login ever being tapped.
      * `tap_login=True` - what the phone actually did. The login tap is accepted, PGSharp
        closes its own panel, and the account does not change - so every read of the
        panel keeps naming the OUTGOING account as active.

    `view` overrides what the tree reports, for the case where no read ever names anybody.
    The view is delivered through `_refresh_accounts`, not written onto the Context, so
    what an attempt observed goes through the same path production uses.
    """
    r.ctx.now = start
    r.ctx.state = BotState.SCANNING
    r._maybe_switch(obs())
    if r.ctx.state is not BotState.SWITCHING:
        return False
    r.tree_reader = FakeTreeReader([view if view is not None
                                    else (panel() if tap_login else closed_panel())])
    r._accounts_read_at = 0.0            # the read throttle is not what is under test
    r._refresh_accounts(start)
    r.apply(fsm.step(obs(), r.ctx), obs())
    r.ctx.now = start + r.cfg.timings.switch_timeout + 1.0
    r.apply(fsm.step(obs(on_map=True), r.ctx), obs())
    assert r.ctx.state is BotState.RECOVERING, "the switch must end at its own timeout"
    # RECOVERING gives up INTO scanning, with the map in front of it - which is precisely
    # the condition `_maybe_switch` is waiting for.
    r.apply([Transition(BotState.SCANNING, IntentOutcome.CARRIED, "recovery attempt over")],
            obs())
    return True


def _quota_switcher(**kw):
    r = make_runner(DEFAULT.scaled(switch_on_quota=True),
                    tree_reader=FakeTreeReader([closed_panel()]), roster=ROSTER, **kw)
    r.stats.account = "TrainerOne"
    r.ctx.spins_exhausted = True
    return r


#: A cycle is one attempt plus the recovery that follows it - the fastest the runner can
#: possibly come back for another go.
CYCLE = DEFAULT.timings.switch_timeout + 5.0


def _cycles(r, count, *, t0=None, tap_login=False):
    """Attempt a switch every CYCLE seconds; return the times an attempt actually began.

    Times run forward from the runner's own clock rather than from zero: `ctx.now` is a
    `perf_counter` reading in production, and a switch begun at t=0 would stamp
    `switch_login_ts` with the same 0.0 that means "no login was tapped".
    """
    t0 = r.ctx.now + 1.0 if t0 is None else t0
    return [t for t in (t0 + i * CYCLE for i in range(count))
            if _fail_a_switch(r, t, tap_login=tap_login)]


def test_a_failing_switch_is_not_retried_forever():
    """The defect, driven the way it was found: cycle after cycle through the real Runner.

    `spins_exhausted` stays true for hours, `choose_next_account` keeps naming the same
    account, and RECOVERING gives up straight back into SCANNING - so before the failure
    was recorded anywhere, every single cycle started another attempt. The stuck watchdog
    cannot save this: it refreshes whenever the map is visible, and in this failure mode
    the map IS visible; only the account is wrong.

    The login is deliberately never tapped here, so the account stays known and the ONLY
    thing that can stop the stream is the failure record itself.
    """
    r = _quota_switcher()
    starts = _cycles(r, 24)

    assert len(starts) == runner_mod.SWITCH_MAX_FAILURES, \
        f"24 cycles produced {len(starts)} attempts; unbounded retrying is the bug"
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g >= runner_mod.SWITCH_BACKOFF_BASE for g in gaps), gaps
    assert gaps == sorted(gaps) and gaps[-1] > gaps[0], \
        f"consecutive failures must escalate the wait, not repeat it: {gaps}"


def test_the_login_tapped_failure_gets_the_same_three_attempts(caplog):
    """The failure the phone actually produces, driven end to end. Blanking the account
    here used to destroy `choose_next_account`'s origin, so the run got ONE attempt and
    then nothing - while the log promised a retry in ten minutes that could never come and
    the "giving up" warning never fired."""
    r = _quota_switcher()
    with caplog.at_level(logging.WARNING, logger="pogobot"):
        starts = _cycles(r, 24, tap_login=True)
    assert len(starts) == runner_mod.SWITCH_MAX_FAILURES
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g >= runner_mod.SWITCH_BACKOFF_BASE for g in gaps) and gaps[-1] > gaps[0]
    assert len([m for m in caplog.messages if "giving up" in m]) == 1


def test_the_first_failure_alone_holds_off_the_next_attempt():
    r = _quota_switcher()
    t0 = r.ctx.now + 1.0
    assert _fail_a_switch(r, t0, tap_login=False)
    assert CYCLE < runner_mod.SWITCH_BACKOFF_BASE, "otherwise this proves nothing"
    assert not _fail_a_switch(r, t0 + CYCLE, tap_login=False), "retried inside the backoff"
    assert _fail_a_switch(r, t0 + CYCLE + runner_mod.SWITCH_BACKOFF_BASE, tap_login=False)


def test_the_clock_trigger_is_held_off_by_the_same_record():
    """A missed rotation deadline stays missed - only a CONFIRMED switch moves
    `_next_rotation` - so advancing it could never have covered this on its own, and the
    quota trigger has no deadline to advance at all."""
    r = make_runner(DEFAULT.scaled(switch_every_minutes=1.0),
                    tree_reader=FakeTreeReader([closed_panel()]), roster=ROSTER)
    r.stats.account = "TrainerOne"
    starts = _cycles(r, 24, t0=r._next_rotation + 0.1)
    assert len(starts) == runner_mod.SWITCH_MAX_FAILURES


def test_giving_up_says_so_and_names_the_target(caplog):
    r = _quota_switcher()
    with caplog.at_level(logging.WARNING, logger="pogobot"):
        _cycles(r, 24)
    final = [m for m in caplog.messages if "giving up" in m]
    assert len(final) == 1 and "TrainerTwo" in final[0]


def test_a_confirmed_switch_forgives_the_earlier_failures():
    """One bad patch - a throttle that has since cleared - must not disable switching for
    the rest of the run."""
    r = _quota_switcher()
    t0 = r.ctx.now + 1.0
    assert _fail_a_switch(r, t0, tap_login=False)
    assert r._switch_failures == 1

    r.ctx.state = BotState.SWITCHING
    r._switch_target = "TrainerTwo"
    r.apply([Transition(BotState.SCANNING, IntentOutcome.CONFIRMED,
                        "logged into TrainerTwo")], obs())
    assert r._switch_failures == 0 and r._switch_blocked_until == 0.0

    r.ctx.spins_exhausted = True
    assert _fail_a_switch(r, t0 + CYCLE, tap_login=False), \
        "a proven-working switch must not still be serving the old backoff"


def test_a_timed_out_login_is_attributed_to_whoever_the_overlay_last_named(tmp_path):
    """`switch_login_grace` exists because a login can land late, so an expiry is the one
    case where the outgoing name cannot simply be assumed. It is not a case of knowing
    nothing, though: `verify` re-opens the panel and reads the asterisk right up to the
    timeout - live, it named the outgoing account fourteen times, minutes after the tap.
    Spending that read is what keeps the books, the 24h window and the round-robin origin
    pointing at a real account."""
    q = SpinQuota(tmp_path / "s.jsonl", limit=10)
    r = _quota_switcher(quota=q)
    assert _fail_a_switch(r, r.ctx.now + 1.0, tap_login=True)
    assert r.stats.account == "TrainerOne"

    r.ctx.state = BotState.POKESTOP
    r.apply([Transition(BotState.POPUP, IntentOutcome.CONFIRMED, "stop collected")], obs())
    assert q.state("TrainerOne").used == 1
    assert q.state().used == 0, "an unattributed bucket means the cap is tracked twice"


def test_a_failed_switch_does_not_uncap_the_account(tmp_path, caplog):
    """The regression that matters most. An unknown account reads the EMPTY unattributed
    bucket, so `spins_exhausted` flips True -> False while the real account is still at
    its cap: the FSM resumes targeting stops the game will refuse, and `_explain_refusal`
    goes quiet because the "" bucket is not exhausted. That is the 152-refused-stops
    misdiagnosis `quota.py` exists to prevent, reinstated by a failed switch."""
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerOne")
    r = _quota_switcher(quota=q)
    r._update_spins_exhausted()
    assert r.ctx.spins_exhausted is True

    assert _fail_a_switch(r, r.ctx.now + 1.0, tap_login=True)
    r._update_spins_exhausted()
    assert r.ctx.spins_exhausted is True, "a failed switch un-capped a capped account"

    r.stats.stops_out_of_range = 1          # the every-10th-refusal gate
    with caplog.at_level(logging.WARNING, logger="pogobot"):
        r._explain_refusal()
    assert any("the cap, not distance" in m for m in caplog.messages)


def test_only_a_login_nobody_ever_watched_leaves_the_account_unknown(tmp_path):
    """No read during the attempt named an active account, so there is nothing to spend.
    This is the case `None` is for - and it is the only one."""
    q = SpinQuota(tmp_path / "s.jsonl", limit=10)
    r = _quota_switcher(quota=q)
    assert _fail_a_switch(r, r.ctx.now + 1.0, tap_login=True, view=panel(active=None))
    assert r.stats.account is None

    r.ctx.state = BotState.POKESTOP
    r.apply([Transition(BotState.POPUP, IntentOutcome.CONFIRMED, "stop collected")], obs())
    assert q.state("TrainerOne").used == 0, "booked to an account we cannot vouch for"
    assert q.state().used == 1, "the unattributed bucket is the honest home for it"


def test_an_observation_from_one_attempt_is_not_spent_on_the_next():
    """`_last_seen_active` is scoped to the attempt that observed it: carried forward, an
    attempt that watched nothing would inherit the previous attempt's evidence."""
    r = _quota_switcher()
    assert _fail_a_switch(r, r.ctx.now + 1.0, tap_login=True)
    assert r._last_seen_active == "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r._begin_switch("TrainerTwo")
    assert r._last_seen_active is None


def test_a_switch_that_never_tapped_a_login_keeps_the_account_name():
    """Nothing touched a login button, so the phone is still on the account we started on.
    Blanking it would throw away a fact we do have."""
    r = _quota_switcher()
    assert _fail_a_switch(r, r.ctx.now + 1.0, tap_login=False)
    assert r.stats.account == "TrainerOne"


def test_no_tap_in_a_failing_switch_ever_lands_on_a_delete_button():
    """The failure path is exactly where a fallback coordinate would be tempting, and the
    delete button sits 157px from the login button it would be falling back from. Every
    tap here still has to come from a node the tree just reported."""
    deletes = {r.delete_norm for r in panel().rows if r.delete_norm}
    for tap_login in (False, True):
        r = _quota_switcher()
        _cycles(r, 6, tap_login=tap_login)
        tapped = {(t.x, t.y) for t in r.actuator.applied if isinstance(t, Tap)}
        assert tapped, "an attempt that taps nothing at all proves nothing here"
        assert not (tapped & deletes)


def test_each_attempt_starts_without_the_previous_login_stamp():
    """`_settle` waits out the grace period from `switch_login_ts`. Inherited from an
    earlier attempt it is already satisfied, so attempt 2 could verify against a login tap
    that had not happened yet - and `_on_switch_failed` would read a login into an attempt
    that never tapped one."""
    r = make_runner(tree_reader=FakeTreeReader([closed_panel()]), roster=ROSTER)
    r.ctx.state = BotState.SCANNING
    r.ctx.switch_login_ts = 55.0
    r._begin_switch("TrainerTwo")
    assert r.ctx.switch_login_ts == 0.0


# ------------------------------------------------------------------ the quota flag

def test_the_quota_flag_is_derived_from_this_accounts_window(tmp_path):
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerOne")
    r = make_runner(quota=q)
    r.stats.account = "TrainerOne"
    r._update_spins_exhausted()
    assert r.ctx.spins_exhausted is True


def test_the_quota_flag_stays_clear_while_there_is_room(tmp_path):
    q = SpinQuota(tmp_path / "s.jsonl", limit=10)
    q.record("TrainerOne")
    r = make_runner(quota=q)
    r.stats.account = "TrainerOne"
    r.ctx.spins_exhausted = True
    r._update_spins_exhausted()
    assert r.ctx.spins_exhausted is False


def test_another_accounts_spins_do_not_exhaust_this_one(tmp_path):
    """The bug this suite could not see before: one wrong argument at that line and the
    flag describes somebody else's day."""
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerTwo")
    r = make_runner(quota=q)
    r.stats.account = "TrainerOne"
    r._update_spins_exhausted()
    assert r.ctx.spins_exhausted is False


def test_a_spin_is_recorded_against_the_running_account(tmp_path):
    q = SpinQuota(tmp_path / "s.jsonl", limit=10)
    r = make_runner(quota=q)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.POKESTOP
    r.apply([Transition(BotState.POPUP, IntentOutcome.CONFIRMED, "stop collected")], obs())
    assert q.state("TrainerOne").used == 1
    assert q.state().used == 0, "an unattributed bucket means the cap is tracked twice"


# ------------------------------------------------------------------ dialogue collection

def test_non_map_frames_during_a_switch_are_saved_for_labelling(tmp_path):
    out = tmp_path / "dialogues"
    r = make_runner(tree_reader=FakeTreeReader([panel()]), dialogue_dump=out)
    r.ctx.state = BotState.SWITCHING
    r._collect_dialogue(_frame_of(120), obs(on_map=False, screen="Menu", conf=0.99))
    r._collect_dialogue(_frame_of(121), obs(on_map=True))     # map: not a dialogue
    assert len(list(out.glob("*.png"))) == 1


def test_nothing_is_written_when_the_flag_is_absent(tmp_path):
    r = make_runner(tree_reader=FakeTreeReader([panel()]))
    r.ctx.state = BotState.SWITCHING
    r._collect_dialogue(_frame_of(1), obs(on_map=False, screen="Menu", conf=0.99))      # must not raise


def test_frames_outside_a_switch_are_not_collected(tmp_path):
    out = tmp_path / "dialogues"
    r = make_runner(tree_reader=FakeTreeReader([panel()]), dialogue_dump=out)
    r.ctx.state = BotState.SCANNING
    r._collect_dialogue(_frame_of(1), obs(on_map=False, screen="Menu", conf=0.99))
    assert not out.exists()
