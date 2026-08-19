from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from pogobot.accounts import (AccountView, FakeTreeReader, UiTreeReader,
                              identify_account, parse_dump)
from pogobot.effects import Tap

FIX = Path(__file__).parent / "fixtures" / "uiautomator"
WH = (1080, 2340)


def view(name: str) -> AccountView:
    return parse_dump((FIX / name).read_bytes(), WH)


def test_open_panel_lists_both_accounts_with_the_active_one_marked():
    v = view("accounts_open.xml")
    assert v.available and v.panel_open
    assert [r.name for r in v.rows] == ["TrainerOne", "TrainerTwo"]
    assert [r.active for r in v.rows] == [True, False]
    assert v.active.name == "TrainerOne"


def test_the_asterisk_is_stripped_from_the_name():
    assert all(not r.name.startswith("*") for r in view("accounts_open.xml").rows)


def test_levels_are_parsed():
    assert [r.level for r in view("accounts_open.xml").rows] == [62, 5]


def test_controls_are_located():
    v = view("accounts_open.xml")
    # Accounts tab spans x 756..1032 of 1080 -> centre ~0.828
    assert v.accounts_tab_norm == pytest.approx((0.8278, 0.1778), abs=0.01)
    assert v.close_norm is not None
    assert v.launcher_norm is not None


def test_login_and_delete_are_distinct_targets():
    row = view("accounts_open.xml").rows[0]
    assert row.login_norm != row.delete_norm
    # login is left of delete; a tap that drifts right destroys an account
    assert row.login_norm[0] < row.delete_norm[0]


def test_closed_overlay_locates_the_launcher_but_lists_nothing():
    v = view("overlay_closed.xml")
    assert v.available is True
    assert v.panel_open is False
    assert v.rows == ()
    assert v.launcher_norm is not None


def test_a_failed_dump_is_unavailable_not_empty():
    v = parse_dump(b"not xml at all", WH)
    assert v.available is False
    assert v.rows == ()
    # the distinction that matters: "could not look" must never read as "no accounts"
    assert v.active is None


def test_by_name_and_names():
    v = view("accounts_open.xml")
    assert v.names == ("TrainerOne", "TrainerTwo")
    assert v.by_name("TrainerTwo").level == 5
    assert v.by_name("Nobody") is None


def test_reader_returns_unavailable_when_adb_fails(monkeypatch):
    def boom(*a, **k):
        raise OSError("adb not found")
    r = UiTreeReader(WH)
    monkeypatch.setattr(r, "_run", boom)
    v = r.read()
    assert v.available is False and v.rows == ()


def test_reader_returns_unavailable_on_uiautomator_error_text(monkeypatch):
    r = UiTreeReader(WH)
    monkeypatch.setattr(r, "_run", lambda *a, **k: b"ERROR: could not get idle state.")
    assert r.read().available is False


def test_reader_parses_a_real_dump(monkeypatch):
    r = UiTreeReader(WH)
    payload = (FIX / "accounts_open.xml").read_bytes()
    monkeypatch.setattr(r, "_run", lambda *a, **k: payload)
    assert r.read().active.name == "TrainerOne"


def test_fake_reader_yields_queued_views_then_repeats_the_last():
    a, b = view("overlay_closed.xml"), view("accounts_open.xml")
    f = FakeTreeReader([a, b])
    assert f.read().panel_open is False
    assert f.read().panel_open is True
    assert f.read().panel_open is True
    assert f.reads == 3


def test_run_deletes_file_before_dumping(monkeypatch):
    """_run() must rm before dump to avoid reading stale files."""
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(stdout=b"")
    monkeypatch.setattr("pogobot.accounts.subprocess.run", mock_run)

    r = UiTreeReader(WH)
    r._run()

    # Verify rm is called before dump. rm call should be first, dump second.
    calls = mock_run.call_args_list
    assert len(calls) >= 2
    rm_call = calls[0]
    dump_call = calls[1]
    # rm should have rm -f DUMP_PATH as args
    assert "rm" in rm_call[0][0]
    assert "-f" in rm_call[0][0]
    # dump should have uiautomator dump as args
    assert "uiautomator" in dump_call[0][0]
    assert "dump" in dump_call[0][0]


def test_run_returns_unavailable_when_dump_produces_no_file(monkeypatch):
    """A dump that succeeds but produces no file should return available=False."""
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(stdout=b"")
    monkeypatch.setattr("pogobot.accounts.subprocess.run", mock_run)

    r = UiTreeReader(WH)
    v = r.read()

    assert v.available is False
    assert v.rows == ()


def test_run_parses_successful_dump_with_fixture_data(monkeypatch):
    """A successful dump that produces valid XML should parse correctly."""
    mock_run = MagicMock()
    payload = (FIX / "accounts_open.xml").read_bytes()
    mock_run.return_value = MagicMock(stdout=payload)
    monkeypatch.setattr("pogobot.accounts.subprocess.run", mock_run)

    r = UiTreeReader(WH)
    v = r.read()

    assert v.available is True
    assert v.active.name == "TrainerOne"
    assert len(v.rows) == 2


def test_run_with_serial_includes_device_selector(monkeypatch):
    """_run() should include -s <serial> in all adb commands when serial is set."""
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(stdout=b"")
    monkeypatch.setattr("pogobot.accounts.subprocess.run", mock_run)

    r = UiTreeReader(WH, serial="device123")
    r._run()

    # All three calls should include -s device123
    calls = mock_run.call_args_list
    for call_obj in calls:
        cmd = call_obj[0][0]
        assert "-s" in cmd
        idx = cmd.index("-s")
        assert cmd[idx + 1] == "device123"


class _Act:
    """Records every effect handed to apply(), like the fake actuator every other test
    file in this suite defines locally (tests/test_switch_runner.py, tests/test_pause.py)
    rather than sharing one - each file states the device it is pretending to have."""

    def __init__(self):
        self.applied = []

    def apply(self, effect, now=None):
        self.applied.append(effect)
        return True


def _taps(act):
    return [e for e in act.applied if isinstance(e, Tap)]


def test_identify_account_opens_reads_and_recloses_the_panel():
    closed, opened = view("overlay_closed.xml"), view("accounts_open.xml")
    tr = FakeTreeReader([closed, opened])
    act = _Act()
    name = identify_account(tr, act, settle=0)
    assert name == "TrainerOne"
    assert [(t.x, t.y) for t in _taps(act)] == [closed.launcher_norm, opened.close_norm]


def test_identify_account_never_taps_a_delete_button():
    """The delete button sits ~24px from login on every row - this is the assertion
    that matters, not just that identification works."""
    closed, opened = view("overlay_closed.xml"), view("accounts_open.xml")
    tr = FakeTreeReader([closed, opened])
    act = _Act()
    identify_account(tr, act, settle=0)
    delete_coords = {r.delete_norm for r in opened.rows if r.delete_norm}
    tapped_coords = {(t.x, t.y) for t in _taps(act)}
    assert not (tapped_coords & delete_coords)


def test_identify_account_taps_nothing_when_the_first_read_is_unavailable():
    tr = FakeTreeReader([AccountView(available=False)])
    act = _Act()
    assert identify_account(tr, act, settle=0) is None
    assert act.applied == []


def test_identify_account_taps_nothing_without_a_located_launcher():
    tr = FakeTreeReader([AccountView(available=True, launcher_norm=None)])
    act = _Act()
    assert identify_account(tr, act, settle=0) is None
    assert act.applied == []


def test_identify_account_still_recloses_the_panel_with_no_account_marked_active():
    """Whatever the second read shows, the panel must be left as it was found - closed -
    if a close control was located, even when no name was found to attribute anything to.

    `accounts_tab_norm=None` here also covers the "wrong tab, but nothing located to fix
    it" branch: zero rows visible, no accounts tab control to tap, so no tab tap is
    attempted - a missing node still means do nothing."""
    closed = view("overlay_closed.xml")
    opened_no_active = AccountView(rows=(), launcher_norm=closed.launcher_norm,
                                   accounts_tab_norm=None, close_norm=(0.05, 0.11),
                                   available=True, panel_open=True)
    tr = FakeTreeReader([closed, opened_no_active])
    act = _Act()
    name = identify_account(tr, act, settle=0)
    assert name is None
    assert [(t.x, t.y) for t in _taps(act)] == [closed.launcher_norm,
                                                opened_no_active.close_norm]


def test_identify_account_closes_nothing_when_the_second_read_fails():
    closed = view("overlay_closed.xml")
    tr = FakeTreeReader([closed, AccountView(available=False)])
    act = _Act()
    assert identify_account(tr, act, settle=0) is None
    # Only the opening tap - never a close guessed without a location for it.
    assert [(t.x, t.y) for t in _taps(act)] == [closed.launcher_norm]


def test_identify_account_selects_the_accounts_tab_when_the_panel_opens_on_another_tab():
    """Measured live: PGSharp remembers the last-viewed tab. The panel opened on Cooldown
    History, not Accounts, so the second read located a close control but zero rows -
    exactly the shape this reproduces: closed -> open on the wrong tab -> open on the
    Accounts tab with rows."""
    closed = view("overlay_closed.xml")
    opened_right_tab = view("accounts_open.xml")
    opened_wrong_tab = replace(opened_right_tab, rows=())
    tr = FakeTreeReader([closed, opened_wrong_tab, opened_right_tab])
    act = _Act()
    name = identify_account(tr, act, settle=0)
    assert name == "TrainerOne"
    assert [(t.x, t.y) for t in _taps(act)] == [
        closed.launcher_norm,
        opened_wrong_tab.accounts_tab_norm,
        opened_right_tab.close_norm,
    ]
    # The row's delete button sits ~24px from its login button - the tab-tap sequence
    # must not land on it either.
    delete_coords = {r.delete_norm for r in opened_right_tab.rows if r.delete_norm}
    tapped_coords = {(t.x, t.y) for t in _taps(act)}
    assert not (tapped_coords & delete_coords)


def test_identify_account_does_not_tap_the_accounts_tab_when_rows_are_already_visible():
    """The common case - the panel already remembers the Accounts tab - must not pay for
    a redundant tab tap and a third read it does not need."""
    closed = view("overlay_closed.xml")
    opened = view("accounts_open.xml")
    tr = FakeTreeReader([closed, opened])
    act = _Act()
    name = identify_account(tr, act, settle=0)
    assert name == "TrainerOne"
    assert [(t.x, t.y) for t in _taps(act)] == [closed.launcher_norm, opened.close_norm]
    assert tr.reads == 2
