"""Counters must match what the bot actually did, and must not overstate it."""
import json

import pytest

from pogobot import runner as runner_mod
from pogobot.config import DEFAULT
from pogobot.effects import BotState, Cooldown, Halt, IntentOutcome, Swipe, Tap, Transition
from pogobot.stats import SessionStats, append_session, lifetime_line, load_lifetime


class _Act:
    """Accepts every actuation, like a live actuator that succeeded."""

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


def _runner():
    return runner_mod.Runner(DEFAULT, _Src(), _Act(), perceptor=None, display=False)


def _apply(r, effects, state=None):
    if state is not None:
        r.ctx.state = state
    r.apply(list(effects), obs=None)


# ---------------------------------------------------------------- event mapping

def test_entering_encounter_counts_once_not_per_tick():
    r = _runner()
    _apply(r, [Transition(BotState.ENCOUNTER, IntentOutcome.CONFIRMED, "x")], BotState.TARGETING)
    # a second Transition while already in ENCOUNTER must not double count
    _apply(r, [Transition(BotState.ENCOUNTER, IntentOutcome.CARRIED, "x")])
    assert r.stats.encounters == 1


def test_balls_thrown_only_counts_gestures_that_were_sent():
    r = _runner()
    _apply(r, [Swipe(0.5, 0.84, 0.5, 0.38, "throw", budget="throw")], BotState.ENCOUNTER)
    _apply(r, [Swipe(0.25, 0.45, 0.75, 0.45, "spin", budget="spin")])
    assert r.stats.balls_thrown == 1, "only the throw budget is a ball"


def test_catch_attempt_is_visible_immediately_not_only_after_the_encounter_ends():
    """A 5 second session that threw a ball must already show the attempt."""
    r = _runner()
    _apply(r, [Transition(BotState.ENCOUNTER, IntentOutcome.CONFIRMED, "x")], BotState.TARGETING)
    _apply(r, [Swipe(0.5, 0.84, 0.5, 0.38, "throw", budget="throw")])
    assert r.stats.catch_attempts == 1, "should not wait for the encounter to end"


def test_multiple_throws_in_one_encounter_are_one_attempt():
    r = _runner()
    _apply(r, [Transition(BotState.ENCOUNTER, IntentOutcome.CONFIRMED, "x")], BotState.TARGETING)
    for _ in range(4):
        _apply(r, [Swipe(0.5, 0.84, 0.5, 0.38, "throw", budget="throw")])
    assert r.stats.balls_thrown == 4 and r.stats.catch_attempts == 1


def test_catch_attempt_requires_a_ball_to_have_been_thrown():
    r = _runner()
    _apply(r, [Transition(BotState.ENCOUNTER, IntentOutcome.CONFIRMED, "x")], BotState.TARGETING)
    _apply(r, [Transition(BotState.SCANNING, IntentOutcome.CARRIED, "back")])
    assert r.stats.encounters == 1
    assert r.stats.catch_attempts == 0, "an encounter we never threw at is not an attempt"

    r2 = _runner()
    _apply(r2, [Transition(BotState.ENCOUNTER, IntentOutcome.CONFIRMED, "x")], BotState.TARGETING)
    _apply(r2, [Swipe(0.5, 0.84, 0.5, 0.38, "throw", budget="throw")])
    _apply(r2, [Transition(BotState.SCANNING, IntentOutcome.CARRIED, "back")])
    assert r2.stats.catch_attempts == 1


def test_stop_outcomes_are_split_by_the_confirm_refute_distinction():
    r = _runner()
    _apply(r, [Transition(BotState.POPUP, IntentOutcome.CONFIRMED, "collected")], BotState.POKESTOP)
    _apply(r, [Transition(BotState.POPUP, IntentOutcome.REFUTED, "out of range")], BotState.POKESTOP)
    assert r.stats.stops_collected == 1
    assert r.stats.stops_out_of_range == 1


def test_a_stop_tap_that_hit_nothing_is_not_counted_as_out_of_range():
    """Leaving POKESTOP because the map is still there says nothing about range.

    The transition is REFUTED (the intent expected POKESTOP and got SCANNING), so counting
    every REFUTED exit put a "Walk closer to interact" number on screen for stops the bot
    never got a range answer about. Only the handler's own exits, both via POPUP, claim one.
    """
    r = _runner()
    _apply(r, [Transition(BotState.SCANNING, IntentOutcome.REFUTED, "tap missed")],
           BotState.POKESTOP)
    _apply(r, [Transition(BotState.ENCOUNTER, IntentOutcome.REFUTED, "it was a Pokemon")],
           BotState.POKESTOP)
    assert r.stats.stops_out_of_range == 0
    assert r.stats.stops_collected == 0
    _apply(r, [Transition(BotState.POPUP, IntentOutcome.REFUTED, "out of range")],
           BotState.POKESTOP)
    assert r.stats.stops_out_of_range == 1


def test_a_missed_stop_tap_through_the_real_fsm_is_not_out_of_range():
    """End to end through fsm.step, because this defect was invisible unit by unit."""
    from pogobot import fsm
    from tests.factories import det, obs

    r = _runner()
    ctx = r.ctx
    ctx.state, ctx.now, ctx.state_since, ctx.last_map_ts = BotState.SCANNING, 100.0, 100.0, 100.0
    on_stop = obs(on_map=True, detections=[det(name="pokestop", conf=0.9)])
    r.apply(fsm.step(on_stop, ctx), obs=on_stop)
    assert ctx.state is BotState.POKESTOP and r.stats.targets_tapped == 1

    ctx.now = ctx.last_map_ts = 102.0            # the tap did nothing; still on the map
    still_map = obs(on_map=True)
    r.apply(fsm.step(still_map, ctx), obs=still_map)
    assert ctx.state is BotState.SCANNING
    assert r.stats.stops_out_of_range == 0, "a missed tap is not a range answer"


def test_rockets_recoveries_halts_and_expiries():
    r = _runner()
    _apply(r, [Transition(BotState.ROCKET, IntentOutcome.CARRIED, "x")], BotState.POPUP)
    _apply(r, [Transition(BotState.RECOVERING, IntentOutcome.CARRIED, "x")], BotState.SCANNING)
    _apply(r, [Transition(BotState.SCANNING, IntentOutcome.EXPIRED, "x")], BotState.TARGETING)
    _apply(r, [Halt("done")], BotState.SCANNING)
    assert (r.stats.rockets_engaged, r.stats.recoveries,
            r.stats.taps_expired, r.stats.halts) == (1, 1, 1, 1)


def test_target_taps_are_counted_but_close_taps_are_not():
    r = _runner()
    _apply(r, [Tap(0.5, 0.6, "target", budget="tap")], BotState.SCANNING)
    _apply(r, [Tap(0.5, 0.89, "close", budget="close")], BotState.POPUP)
    assert r.stats.targets_tapped == 1


def test_nothing_is_counted_when_the_actuator_refuses():
    class _Refuse(_Act):
        def apply(self, effect, now=None):
            return False

    r = runner_mod.Runner(DEFAULT, _Src(), _Refuse(), perceptor=None, display=False)
    _apply(r, [Swipe(0.5, 0.84, 0.5, 0.38, "throw", budget="throw")], BotState.ENCOUNTER)
    _apply(r, [Tap(0.5, 0.6, "target", budget="tap")], BotState.SCANNING)
    assert r.stats.balls_thrown == 0 and r.stats.targets_tapped == 0


# ---------------------------------------------------------------- honesty

def test_confirmed_catches_is_absent_rather_than_guessed():
    """A catch and a flee end identically, so the bot must not report a catch count."""
    s = SessionStats(started=0.0)
    assert s.confirmed_catches is None
    assert "confirmed_catches" not in s.summary(now=60.0)
    assert "catch_attempts" in s.summary(now=60.0)


def test_report_states_the_caveat():
    s = SessionStats(started=0.0)
    assert "confirmed catches" in s.report(now=60.0)


def test_a_dry_run_says_so_in_the_record_and_the_report():
    """dry_run/replay actuators accept every gesture for pacing, so the counters are
    decisions. Nothing downstream may read them as things the bot did."""
    from pogobot.actions import NullActuator

    r = runner_mod.Runner(DEFAULT, _Src(), NullActuator(), perceptor=None, display=False)
    _apply(r, [Transition(BotState.ENCOUNTER, IntentOutcome.CONFIRMED, "x")], BotState.TARGETING)
    _apply(r, [Swipe(0.5, 0.84, 0.5, 0.38, "throw", budget="throw")])
    assert r.stats.dry_run is True
    assert r.stats.summary(now=r.stats.started + 3600)["dry_run"] is True
    assert "DRY RUN" in r.stats.report(now=r.stats.started + 3600)

    live = _runner()
    assert live.stats.dry_run is False
    assert live.stats.summary(now=live.stats.started + 3600)["dry_run"] is False


def test_a_live_run_never_claims_to_be_a_dry_run():
    r = _runner()
    assert "DRY RUN" not in r.stats.report(now=r.stats.started + 3600)


# ---------------------------------------------------------------- rates

def test_rates_are_per_hour():
    s = SessionStats(started=0.0)
    s.encounters = 30
    assert s.per_hour(s.encounters, now=3600.0) == pytest.approx(30.0)
    assert s.per_hour(s.encounters, now=1800.0) == pytest.approx(60.0)


def test_rates_are_unknown_rather_than_extrapolated_from_a_few_seconds():
    """One event 5s in extrapolates to 677/h, which is a fabricated number."""
    from pogobot.stats import RATE_MIN_UPTIME
    s = SessionStats(started=0.0)
    s.encounters = 1
    assert s.per_hour(s.encounters, now=5.0) is None
    assert s.summary(now=5.0)["encounters_per_hour"] is None
    assert "--/h" in s.hud_line(now=5.0)
    assert s.per_hour(s.encounters, now=RATE_MIN_UPTIME + 1) is not None


def test_report_omits_a_rate_it_cannot_state():
    s = SessionStats(started=0.0)
    s.encounters = 1
    assert "/h" not in s.report(now=5.0).split("catch attempts counts")[0]


# ---------------------------------------------------------------- persistence

def test_lifetime_totals_sum_across_sessions(tmp_path):
    p = tmp_path / "sessions.jsonl"
    assert load_lifetime(p) is None
    for enc, stops, up in ((10, 3, 1800.0), (20, 7, 1800.0)):
        s = SessionStats(started=0.0)
        s.encounters, s.stops_collected = enc, stops
        append_session(p, s.summary(now=up))
    total = load_lifetime(p)
    assert total["sessions"] == 2
    assert total["encounters"] == 30 and total["stops_collected"] == 10
    assert total["encounters_per_hour"] == pytest.approx(30.0)
    assert "lifetime over 2 session(s)" in lifetime_line(total)


def test_a_torn_line_from_a_hard_kill_does_not_break_the_history(tmp_path):
    p = tmp_path / "sessions.jsonl"
    s = SessionStats(started=0.0)
    s.encounters = 5
    append_session(p, s.summary(now=3600.0))
    with open(p, "a") as fh:
        fh.write('{"ended": 1, "encounters": 9, "uptime_s"')   # killed mid-write
    total = load_lifetime(p)
    assert total["sessions"] == 1 and total["encounters"] == 5


def test_a_torn_line_does_not_swallow_the_next_session(tmp_path):
    """The record after a hard kill must survive. Appending onto a line with no newline
    glued the two together and load_lifetime skipped both."""
    p = tmp_path / "sessions.jsonl"
    first = SessionStats(started=0.0)
    first.encounters = 5
    append_session(p, first.summary(now=3600.0))
    with open(p, "a") as fh:
        fh.write('{"ended": 1, "encounters": 9, "uptime_s"')   # SIGKILL mid-write
    nxt = SessionStats(started=0.0)
    nxt.encounters = 7
    append_session(p, nxt.summary(now=3600.0))

    total = load_lifetime(p)
    assert total["sessions"] == 2
    assert total["encounters"] == 12


def test_an_unreadable_history_does_not_stop_the_run(tmp_path):
    """load_lifetime only feeds an informational line; it must degrade, not raise."""
    d = tmp_path / "a_directory_where_a_file_belongs"
    d.mkdir()
    assert load_lifetime(d) is None

    binary = tmp_path / "torn_bytes.jsonl"
    binary.write_bytes(b'{"uptime_s": 3600.0, "encounters": 4}\n\xff\xfe half a line\n')
    total = load_lifetime(binary)
    assert total["sessions"] == 1 and total["encounters"] == 4

    junk = tmp_path / "junk.jsonl"
    junk.write_text('null\n"a string"\n{"uptime_s": 3600.0, "encounters": "many"}\n'
                    '{"uptime_s": 3600.0, "encounters": 2}\n')
    total = load_lifetime(junk)
    assert total["sessions"] == 1 and total["encounters"] == 2


def test_concurrent_runs_can_append_to_one_history(tmp_path):
    p = tmp_path / "sessions.jsonl"
    for i in range(50):
        s = SessionStats(started=0.0)
        s.encounters = 1
        append_session(p, s.summary(now=3600.0))
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    assert len(lines) == 50
    assert all(json.loads(l)["encounters"] == 1 for l in lines)


def test_the_session_record_survives_a_cleanup_step_that_throws(tmp_path):
    """A ledger or actuator exception on the way down must not eat the stats write."""
    class _Throws(_Act):
        def close(self):
            raise RuntimeError("adb worker would not join")

        def stats(self):
            raise RuntimeError("worker already gone")

    p = tmp_path / "sessions.jsonl"
    r = runner_mod.Runner(DEFAULT, _Src(), _Throws(), perceptor=None, display=False,
                          stats_path=p)
    r.stats.encounters = 3
    try:
        r.close()
    except RuntimeError:
        pass          # close() may still surface the failure; the record must exist
    assert json.loads(p.read_text().splitlines()[-1])["encounters"] == 3


def test_stop_signals_stay_installed_until_cleanup_is_finished(tmp_path):
    """A second SIGTERM during a slow close() must not hit the default disposition and
    kill the process before the session record is written."""
    import signal

    seen = {}

    class _SlowSrc(_Src):
        def healthy(self):
            return False              # leave the loop on the first tick

        def failure_reason(self):
            return ""

        def release(self):
            seen["handler"] = signal.getsignal(signal.SIGTERM)

    p = tmp_path / "sessions.jsonl"
    r = runner_mod.Runner(DEFAULT, _SlowSrc(), _Act(), perceptor=None, display=False,
                          stats_path=p)
    before = signal.getsignal(signal.SIGTERM)
    assert r.run() == 0
    assert seen["handler"] not in (signal.SIG_DFL, None), \
        "handlers were restored before close() finished"
    assert signal.getsignal(signal.SIGTERM) is before, "handler was not restored after"
    assert p.exists() and p.read_text().strip()


def test_dry_run_sessions_are_excluded_from_lifetime_totals(tmp_path):
    p = tmp_path / "sessions.jsonl"
    live = SessionStats(started=0.0)
    live.encounters = 10
    append_session(p, live.summary(now=1800.0))
    preview = SessionStats(started=0.0, dry_run=True)
    preview.encounters = 999
    append_session(p, preview.summary(now=1800.0))

    total = load_lifetime(p)
    assert total["sessions"] == 1 and total["encounters"] == 10
    assert total["dry_run_sessions"] == 1
    assert "1 dry-run session(s) excluded" in lifetime_line(total)


def test_a_session_with_no_duration_makes_the_lifetime_rate_unknown(tmp_path):
    """Dividing the full counter total by only the KNOWN time invents a rate higher than
    any single session ever achieved (measured: 108 encounters printed as 216/h)."""
    p = tmp_path / "sessions.jsonl"
    with open(p, "a") as fh:
        fh.write(json.dumps({"ended": 1, "encounters": 100}) + "\n")
    s = SessionStats(started=0.0)
    s.encounters = 8
    append_session(p, s.summary(now=1800.0))

    total = load_lifetime(p)
    assert total["encounters"] == 108, "the events are real and must not be dropped"
    assert total["sessions_without_uptime"] == 1
    assert total["encounters_per_hour"] is None
    line = lifetime_line(total)
    assert "--/h" in line and "recorded no duration" in line


def test_the_lifetime_line_never_prints_a_rate_it_cannot_stand_behind(tmp_path):
    p = tmp_path / "sessions.jsonl"
    s = SessionStats(started=0.0)
    s.encounters = 1
    append_session(p, s.summary(now=30.0))        # under RATE_MIN_UPTIME
    total = load_lifetime(p)
    assert total["encounters_per_hour"] is None
    assert "120.0/h" not in lifetime_line(total)
