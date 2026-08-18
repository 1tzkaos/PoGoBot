"""The dashboard must render every counter it claims to, and must not eat the log."""
import logging
import time

import pytest

rich = pytest.importorskip("rich")

from rich.console import Console

from pogobot.effects import BotState
from pogobot.stats import SessionStats
from pogobot.tui import Dashboard, LogPane, available
from tests.factories import det, obs as mkobs


def _dash(width=104, height=34):
    d = Dashboard.__new__(Dashboard)
    d.stats = SessionStats(started=time.perf_counter() - 3600)
    d.lifetime = {"sessions": 2, "encounters": 40, "stops_collected": 5}
    d.console = Console(width=width, height=height, record=True)
    d.pane = LogPane()
    d.pane.setFormatter(logging.Formatter("%(message)s"))
    return d


def _text(d, obs, state=BotState.SCANNING):
    d.console.print(d._render(obs, state, 7.8))
    return d.console.export_text()


def test_available_reports_rich():
    assert available() is True


def test_every_counter_in_the_summary_is_rendered():
    """A counter that exists but is never shown is how encounters_finished got lost."""
    d = _dash()
    for f in ("encounters", "catch_attempts", "balls_thrown", "stops_collected",
              "stops_out_of_range", "rockets_engaged", "targets_tapped",
              "taps_expired", "recoveries", "halts"):
        setattr(d.stats, f, 7)
    out = _text(d, mkobs(on_map=True))
    for label in ("encounters", "catch attempts", "balls thrown", "stops collected",
                  "stops out of range", "rockets engaged", "targets tapped",
                  "taps expired", "recoveries", "halts"):
        assert label in out, f"{label} missing from the dashboard"


def test_all_four_detection_classes_fit_in_the_pane():
    """The pane was fixed at 14 rows and silently truncated the 4th class."""
    d = _dash()
    o = mkobs(on_map=True, detections=[det("pokemon", 0.8), det("pokestop", 0.6),
                                       det("pokestop_rocket", 0.5), det("gym", 0.55)])
    out = _text(d, o)
    for name in ("pokemon", "pokestop", "pokestop_rocket", "gym"):
        assert name in out, f"{name} truncated out of the perception pane"


def test_rates_show_as_unknown_before_the_minimum_uptime():
    d = _dash()
    d.stats.started = time.perf_counter()
    d.stats.encounters = 1
    assert "--/h" in _text(d, mkobs(on_map=True))


def test_renders_before_any_frame_has_arrived():
    d = _dash()
    assert "waiting for a frame" in _text(d, None, BotState.BOOT)


def test_renders_every_state_without_raising():
    for state in BotState:
        d = _dash()
        assert state.value in _text(d, mkobs(on_map=True), state)


def test_log_pane_keeps_records_and_never_raises_on_a_bad_record():
    pane = LogPane(capacity=3)
    pane.setFormatter(logging.Formatter("%(message)s"))
    for i in range(5):
        pane.emit(logging.LogRecord("x", logging.INFO, "f", i, "line %d", (i,), None))
    assert len(pane.records) == 3, "bounded so a long run cannot grow without limit"
    bad = logging.LogRecord("x", logging.INFO, "f", 1, "%d", ("not-an-int",), None)
    pane.emit(bad)   # must not propagate: a logging failure cannot be allowed to kill the bot


def test_a_narrow_terminal_still_renders():
    d = _dash(width=60, height=20)
    assert "PoGoBot" in _text(d, mkobs(on_map=True))
