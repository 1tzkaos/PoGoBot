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
        q.record(now - WINDOW_SECONDS - 100 - i)   # yesterday
    for _ in range(3):
        q.record(now)
    s = q.state(now)
    assert s.used == 3, "stale spins must age out"
    assert not s.exhausted


def test_it_reports_when_the_oldest_spin_ages_out():
    now = time.time()
    q = SpinQuota(None, limit=5)
    q.record(now - WINDOW_SECONDS + 3600)          # ages out in an hour
    s = q.state(now)
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
    assert SpinQuota(p, limit=100).state().used == 2


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
