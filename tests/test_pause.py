"""Pausing must stop input without corrupting the clock.

The trap is resuming: if every deadline aged while the bot was idle, the first tick back
fires every timeout at once. A pause that ends in a recovery storm is worse than no pause.
"""
import time

import pytest

from pogobot import runner as runner_mod
from pogobot.config import DEFAULT
from pogobot.effects import BotState
from pogobot.stats import SessionStats


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


def _runner(**kw):
    return runner_mod.Runner(DEFAULT, _Src(), _Act(), perceptor=None, display=False, **kw)


# ------------------------------------------------------------------ triggers

def test_starts_unpaused():
    assert not _runner().paused


def test_the_toggle_flips_it():
    r = _runner()
    r.toggle_pause()
    assert r._sync_pause() and r.paused
    r.toggle_pause()
    assert not r._sync_pause() and not r.paused


def test_the_pause_file_pauses_and_removing_it_resumes(tmp_path):
    f = tmp_path / "PAUSE"
    r = _runner(pause_file=f)
    assert not r._sync_pause()
    f.touch()
    assert r._sync_pause(), "the file must pause it"
    f.unlink()
    assert not r._sync_pause(), "removing it must resume"


def test_an_unreadable_pause_path_does_not_pause(tmp_path):
    r = _runner(pause_file=tmp_path / "nope" / "deep" / "PAUSE")
    assert not r._sync_pause()


def test_the_file_and_the_toggle_are_independent(tmp_path):
    f = tmp_path / "PAUSE"
    r = _runner(pause_file=f)
    r.toggle_pause()
    assert r._sync_pause()
    f.touch()
    r.toggle_pause()                 # toggle off, but the file still says pause
    assert r._sync_pause(), "the file alone is enough to stay paused"


# ------------------------------------------------------------------ the clock

def test_paused_time_does_not_age_the_state_machine():
    """The whole point: a deadline must not expire while the bot is idle."""
    r = _runner()
    r.toggle_pause()
    r._sync_pause()
    r._paused_at = time.perf_counter() - 300.0        # five minutes paused
    r.toggle_pause()
    r._sync_pause()
    assert r._pause_total == pytest.approx(300.0, abs=2)
    # ctx.now is driven from perf_counter minus paused time, so a deadline stamped before
    # the pause is still the same distance away afterwards.
    before = time.perf_counter() - r.stats.paused_seconds
    assert time.perf_counter() - before == pytest.approx(300.0, abs=2)


def test_paused_time_is_excluded_from_the_rates():
    s = SessionStats(started=0.0, paused_seconds=1800.0)
    s.encounters = 30
    assert s.uptime(now=3600.0) == pytest.approx(1800.0)
    assert s.per_hour(s.encounters, now=3600.0) == pytest.approx(60.0), \
        "an overnight pause must not dilute the rate"


def test_the_summary_reports_paused_time():
    s = SessionStats(started=0.0, paused_seconds=120.0)
    assert s.summary(now=600.0)["paused_s"] == 120.0
    assert "paused" in s.report(now=600.0)


def test_paused_seconds_grows_while_still_paused():
    r = _runner()
    r.toggle_pause()
    r._sync_pause()
    r._paused_at = time.perf_counter() - 10.0
    r._sync_pause()
    assert r.stats.paused_seconds >= 10.0, "a long pause must be visible before it ends"


# ------------------------------------------------------------------ rendering

def test_the_hud_says_paused():
    import numpy as np
    from pogobot import hud
    from tests.factories import obs as mkobs
    img = hud.render(np.zeros((1280, 590, 3), np.uint8), mkobs(on_map=True), DEFAULT,
                     BotState.SCANNING, paused=True)
    plain = hud.render(np.zeros((1280, 590, 3), np.uint8), mkobs(on_map=True), DEFAULT,
                       BotState.SCANNING, paused=False)
    assert not np.array_equal(img, plain), "the paused banner must be visible"
