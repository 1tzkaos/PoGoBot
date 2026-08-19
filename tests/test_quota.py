"""The rolling 24h PokeStop spin cap.

Past the cap a stop refuses in a way that is visually identical to being out of reach.
That ambiguity produced a real misdiagnosis: 152 refused stops in one session were
reported as a positioning problem when the account had simply spun out for the day.
"""
import json
import time

import pytest

from pogobot import fsm
from pogobot.config import DEFAULT
from pogobot.effects import BotState, Tap
from pogobot.quota import DEFAULT_DAILY_LIMIT, WINDOW_SECONDS, SpinQuota
from tests.factories import det, obs as mkobs


def test_a_fresh_account_has_the_whole_allowance():
    q = SpinQuota(None, limit=1200)
    s = q.state()
    assert (s.used, s.remaining, s.exhausted) == (0, 1200, False)


def test_spins_accumulate_and_exhaust():
    q = SpinQuota(None, limit=10)
    for _ in range(10):
        q.record()
    s = q.state()
    assert s.used == 10 and s.remaining == 0 and s.exhausted


def test_spins_older_than_the_window_drop_out():
    """It is a rolling window, not a calendar day."""
    now = time.time()
    q = SpinQuota(None, limit=10)
    for i in range(6):
        q.record(now=now - WINDOW_SECONDS - 100 - i)   # yesterday
    for _ in range(3):
        q.record(now=now)
    s = q.state(now=now)
    assert s.used == 3, "stale spins must age out"
    assert not s.exhausted


def test_it_reports_when_the_oldest_spin_ages_out():
    now = time.time()
    q = SpinQuota(None, limit=5)
    q.record(now=now - WINDOW_SECONDS + 3600)          # ages out in an hour
    s = q.state(now=now)
    assert s.resets_in == pytest.approx(3600, abs=5)
    assert "1h00m" in s.line()


def test_the_exhausted_message_names_the_real_cause():
    q = SpinQuota(None, limit=5)
    q.seed(5, spread_hours=2)
    line = q.state().line()
    assert "QUOTA REACHED" in line
    assert "not distance" in line or "out of range" in line


def test_seeding_records_spins_this_process_did_not_perform():
    """The cap belongs to the account, so spins done by hand still count against it."""
    q = SpinQuota(None, limit=1200)
    q.seed(1200, spread_hours=20)
    assert q.state().exhausted


def test_seed_zero_or_negative_is_a_no_op():
    q = SpinQuota(None, limit=100)
    q.seed(0)
    q.seed(-5)
    assert q.state().used == 0


# ------------------------------------------------------------------ persistence

def test_the_window_survives_a_restart(tmp_path):
    p = tmp_path / "spins.jsonl"
    q = SpinQuota(p, limit=100)
    for _ in range(7):
        q.record()
    assert SpinQuota(p, limit=100).state().used == 7, "the cap spans restarts"


def test_stale_entries_are_dropped_on_load(tmp_path):
    p = tmp_path / "spins.jsonl"
    old = time.time() - WINDOW_SECONDS - 60
    p.write_text("\n".join(json.dumps({"ts": old}) for _ in range(5)) + "\n")
    assert SpinQuota(p, limit=100).state().used == 0


def test_a_torn_line_does_not_lose_the_rest(tmp_path):
    p = tmp_path / "spins.jsonl"
    now = time.time()
    p.write_text(json.dumps({"ts": now}) + "\n" + '{"ts": ' + "\n" +
                 json.dumps({"ts": now}) + "\n")
    # These records have no "account" key, so per the account-keying rework below they
    # are legacy - counted, but not visible via state() until attribute_legacy claims
    # them. legacy_count is the correct place to see the 2 surviving records land.
    assert SpinQuota(p, limit=100).legacy_count == 2


def test_an_unreadable_file_does_not_stop_the_bot(tmp_path):
    d = tmp_path / "spins.jsonl"
    d.mkdir()                       # a directory where a file is expected
    assert SpinQuota(d, limit=100).state().used == 0


def test_recording_never_raises_when_the_path_is_unwritable(tmp_path):
    d = tmp_path / "spins.jsonl"
    d.mkdir()
    q = SpinQuota(d, limit=100)
    q.record()                      # must not raise
    assert q.state().used == 1, "the in-memory count still works"


# ------------------------------------------------------------------ behaviour

def test_stops_are_not_targeted_once_the_cap_is_reached():
    """Tapping a stop past the cap only produces the banner that misled us."""
    c = fsm.Context(cfg=DEFAULT, state=BotState.SCANNING, state_since=0.0, now=10.0)
    c.spins_exhausted = True
    o = mkobs(on_map=True, detections=[det("pokestop", 0.9, cx=0.5, cy=0.64)])
    assert not [e for e in fsm.step(o, c) if isinstance(e, Tap)]


def test_pokemon_are_still_targeted_when_spun_out():
    c = fsm.Context(cfg=DEFAULT, state=BotState.SCANNING, state_since=0.0, now=10.0)
    c.spins_exhausted = True
    o = mkobs(on_map=True, detections=[det("pokemon", 0.9, cx=0.5, cy=0.63)])
    assert [e for e in fsm.step(o, c) if isinstance(e, Tap)], "catching still works"


def test_stops_are_targeted_normally_below_the_cap():
    c = fsm.Context(cfg=DEFAULT, state=BotState.SCANNING, state_since=0.0, now=10.0)
    o = mkobs(on_map=True, detections=[det("pokestop", 0.9, cx=0.5, cy=0.64)])
    assert [e for e in fsm.step(o, c) if isinstance(e, Tap)]


def test_a_zero_limit_disables_the_check():
    q = SpinQuota(None, limit=0)
    for _ in range(5000):
        q.record()
    assert not q.state().exhausted
    assert "no limit configured" in q.state().line()


def test_the_default_limit_is_niantics_documented_cap():
    assert DEFAULT_DAILY_LIMIT == 1200


def test_reset_clears_the_window(tmp_path):
    """A ban that lifts early, or a seed that turns out to be stale, must be correctable:
    an over-stated quota stops the bot targeting stops it can actually use."""
    p = tmp_path / "spins.jsonl"
    q = SpinQuota(p, limit=1200)
    q.seed(1200, spread_hours=20)
    assert q.state().exhausted
    dropped = q.reset()
    assert dropped == 1200
    assert q.state().used == 0 and not q.state().exhausted
    assert SpinQuota(p, limit=1200).state().used == 0, "and it stays cleared across restarts"


def test_reset_on_an_empty_window_is_harmless(tmp_path):
    q = SpinQuota(tmp_path / "spins.jsonl", limit=1200)
    assert q.reset() == 0


# ------------------------------------------------------------------ per-account quotas
#
# The cap belongs to the account, not the phone. With multiple accounts one shared list
# is wrong: two accounts spinning the same stop at different points in the day must not
# exhaust each other's allowance. `account` is optional (defaults to `None`, normalized
# to the "" bucket) so every pre-existing bare call above keeps working unchanged.

def test_windows_are_independent_per_account():
    q = SpinQuota(None, limit=3)
    for _ in range(3):
        q.record("A")
    assert q.state("A").exhausted is True
    assert q.state("B").exhausted is False
    assert q.state("B").remaining == 3


def test_records_round_trip_with_their_account(tmp_path):
    p = tmp_path / "spins.jsonl"
    q = SpinQuota(p, limit=10)
    q.record("A"); q.record("B"); q.record("A")
    assert SpinQuota(p, limit=10).state("A").used == 2
    assert SpinQuota(p, limit=10).state("B").used == 1


def test_legacy_records_are_unattributed_until_told(tmp_path):
    """A record with no "account" key predates this feature. It must not silently count
    against a named account, or an account that never spun could read as exhausted."""
    p = tmp_path / "spins.jsonl"
    p.write_text("\n".join(json.dumps({"ts": time.time()}) for _ in range(5)) + "\n")
    q = SpinQuota(p, limit=10)
    assert q.state("A").used == 0
    assert q.legacy_count == 5
    assert q.attribute_legacy("A") == 5
    assert q.state("A").used == 5


def test_unidentified_runs_do_not_inherit_legacy_spins(tmp_path):
    """The "" bucket (an unidentified-but-tracked run) must stay separate from the
    legacy bucket (records that predate accounts). Otherwise a fresh run that has not
    yet identified its account would read pre-existing legacy spins as its own and
    could make a false exhaustion decision."""
    p = tmp_path / "spins.jsonl"
    p.write_text("\n".join(json.dumps({"ts": time.time()}) for _ in range(5)) + "\n")
    q = SpinQuota(p, limit=10)
    assert q.state().used == 0
    assert q.legacy_count == 5


def test_reset_targets_one_account_or_all(tmp_path):
    q = SpinQuota(tmp_path / "s.jsonl", limit=10)
    q.record("A"); q.record("B")
    assert q.reset("A") == 1
    assert q.state("A").used == 0 and q.state("B").used == 1
    assert q.reset() == 1


def test_soonest_reset_picks_the_account_that_frees_up_first():
    now = time.time()
    q = SpinQuota(None, limit=1)
    q.record("A", now=now - 23 * 3600)      # ages out in ~1h
    q.record("B", now=now - 1 * 3600)       # ages out in ~23h
    assert q.soonest_reset(("A", "B"), now=now) == "A"


def test_soonest_reset_of_an_untouched_account_is_immediate():
    q = SpinQuota(None, limit=1)
    q.record("A")
    assert q.soonest_reset(("A", "B")) == "B", "B has never spun, so it is free right now"


def test_accounts_lists_only_accounts_with_records():
    q = SpinQuota(None, limit=10)
    q.record("B"); q.record("A")
    assert q.accounts() == ("A", "B")


def test_a_wrong_typed_account_fails_loudly_instead_of_reporting_zero():
    """A float key can never match a bucket, so state() would otherwise report
    used=0/exhausted=False - a confident wrong answer indistinguishable from a healthy
    account. That is exactly the misdiagnosis this module exists to prevent."""
    q = SpinQuota(None, limit=10)
    with pytest.raises(TypeError):
        q.state(1234.5)
    with pytest.raises(TypeError):
        q.record(1234.5)
