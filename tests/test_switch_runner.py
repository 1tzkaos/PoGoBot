"""Account switching as the runner drives it: refresh, trigger, select, confirm.

Each of these has a failure mode that a green suite would otherwise hide:

  * the UI-tree read blocks for roughly a second, so it must never happen per frame, and a
    trigger that can never be satisfied must not turn into a per-frame dump either;
  * a switch may only begin from SCANNING with the map confirmed - starting one mid
    encounter abandons a Pokemon mid-throw;
  * only a CONFIRMED switch rolls the session over, so an attempt that times out cannot
    invent a session row or reset a counter;
  * `spins_exhausted` must describe the account we are actually on. Nothing asserted that
    before, which is how one wrong argument at that line could read as good standing.
"""
import json
import logging
import time

import pytest

from pogobot import fsm
from pogobot import runner as runner_mod
from pogobot.accounts import AccountView, FakeTreeReader
from pogobot.config import DEFAULT
from pogobot.effects import BotState, IntentOutcome, Tap, Transition
from pogobot.quota import SpinQuota
from tests.factories import obs
from tests.test_switching import panel


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


# ------------------------------------------------------------------ selection

def test_selection_returns_the_next_usable_account(tmp_path):
    q = SpinQuota(tmp_path / "s.jsonl", limit=10)
    r = make_runner(quota=q, tree_reader=FakeTreeReader([panel()]))
    assert r.choose_next_account(panel(active="TrainerOne")) == "TrainerTwo"


def test_selection_skips_an_exhausted_account(tmp_path):
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerTwo")
    r = make_runner(quota=q, tree_reader=FakeTreeReader([panel()]))
    assert r.choose_next_account(panel(active="TrainerOne")) is None


def test_when_all_accounts_are_capped_it_picks_the_soonest_to_reset(tmp_path):
    now = time.time()
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerOne", now=now - 1 * 3600)      # frees in 23h
    q.record("TrainerTwo", now=now - 23 * 3600)     # frees in 1h
    r = make_runner(quota=q, tree_reader=FakeTreeReader([panel()]))
    assert r.choose_next_account(panel(active="TrainerOne")) == "TrainerTwo"


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
    r = make_runner(quota=q, tree_reader=FakeTreeReader([panel()]))
    assert r.choose_next_account(panel(active="TrainerOne")) is None


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
    r = make_runner(quota=q, tree_reader=FakeTreeReader([panel()]))
    assert r.choose_next_account(panel(active="TrainerOne")) is None


def test_a_panel_that_names_no_active_account_is_not_a_round_robin_origin(tmp_path):
    """No asterisk means we do not know where we are, so there is no origin to rotate from
    and no way to tell a capped account from the one that was working."""
    q = SpinQuota(tmp_path / "s.jsonl", limit=10)
    r = make_runner(quota=q, tree_reader=FakeTreeReader([panel()]))
    assert r.choose_next_account(panel(active=None)) is None


def test_one_account_is_never_a_switch_target(tmp_path):
    v = panel()
    single = AccountView(rows=v.rows[:1], launcher_norm=v.launcher_norm,
                         accounts_tab_norm=v.accounts_tab_norm, close_norm=v.close_norm,
                         available=True, panel_open=True)
    r = make_runner(quota=SpinQuota(tmp_path / "s.jsonl", limit=10),
                    tree_reader=FakeTreeReader([single]))
    assert r.choose_next_account(single) is None


def test_a_view_we_could_not_read_is_not_a_reason_to_switch():
    r = make_runner(tree_reader=FakeTreeReader([AccountView(available=False)]))
    assert r.choose_next_account(AccountView(available=False)) is None
    assert r.choose_next_account(None) is None


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
    shut, open - and the switch then sits there until its 120s timeout."""
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


def test_an_actuation_outside_a_switch_keeps_the_view():
    r = make_runner(tree_reader=FakeTreeReader([panel()]))
    r.ctx.state = BotState.SCANNING
    r.ctx.accounts = panel()
    r.apply([Tap(0.5, 0.6, "target pokemon")], obs())
    assert r.ctx.accounts is not None


# ------------------------------------------------------------------ triggers

def test_switching_stays_off_by_default():
    r = make_runner(tree_reader=FakeTreeReader([panel()]))
    r.ctx.state = BotState.SCANNING
    r.ctx.spins_exhausted = True
    r._maybe_switch(obs())
    assert r.ctx.state is BotState.SCANNING


def test_switching_is_never_started_outside_scanning():
    r = make_runner(DEFAULT.scaled(switch_on_quota=True),
                    tree_reader=FakeTreeReader([panel()]))
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
    r = make_runner(DEFAULT.scaled(switch_on_quota=True), tree_reader=reader)
    r.ctx.state = BotState.SCANNING
    r.ctx.spins_exhausted = True
    r._maybe_switch(off_map())
    assert r.ctx.state is BotState.SCANNING
    assert reader.reads == 0


def test_the_quota_trigger_starts_a_switch_from_scanning():
    reader = FakeTreeReader([panel(active="TrainerOne")])
    r = make_runner(DEFAULT.scaled(switch_on_quota=True), tree_reader=reader)
    r.ctx.state = BotState.SCANNING
    r.ctx.spins_exhausted = True
    r._maybe_switch(obs())
    assert r.ctx.state is BotState.SWITCHING
    assert r.ctx.switch_target == "TrainerTwo"
    assert r.ctx.switch_phase == "open"
    assert reader.reads == 1


def test_no_tree_reader_means_no_switching():
    r = make_runner(DEFAULT.scaled(switch_on_quota=True))
    r.ctx.state = BotState.SCANNING
    r.ctx.spins_exhausted = True
    r._maybe_switch(obs())
    assert r.ctx.state is BotState.SCANNING


def test_an_unsatisfiable_trigger_does_not_dump_the_tree_every_frame(tmp_path):
    """`spins_exhausted` stays true for hours, and selection can legitimately answer None
    the whole time. Forcing the ~1s dump on every one of those ticks stalls an 8fps loop."""
    q = SpinQuota(tmp_path / "s.jsonl", limit=1)
    q.record("TrainerTwo")                      # the only alternative is capped
    reader = FakeTreeReader([panel(active="TrainerOne")])
    r = make_runner(DEFAULT.scaled(switch_on_quota=True), quota=q, tree_reader=reader)
    r.ctx.state = BotState.SCANNING
    r.ctx.spins_exhausted = True
    for _ in range(20):
        r._maybe_switch(obs())
    assert r.ctx.state is BotState.SCANNING
    assert reader.reads == 1
    r._real += runner_mod.SWITCH_PROBE_EVERY + 1.0
    r._maybe_switch(obs())
    assert reader.reads == 2, "the probe must resume eventually, just not every frame"


def test_the_clock_trigger_waits_out_its_first_interval():
    reader = FakeTreeReader([panel(active="TrainerOne")])
    r = make_runner(DEFAULT.scaled(switch_every_minutes=1.0), tree_reader=reader)
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
    """An Intent claims the screen changed BECAUSE of our tap. A switch takes up to 120s
    and puts a login screen up, so any answer after it is not evidence of anything."""
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
