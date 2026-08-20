import logging
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

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


def test_rows_alone_prove_the_panel_is_open():
    """`panel_open` used to hinge on one PGSharp resource-id. If `hl_page_close` ever
    moves, `Switching.step` reads an open panel as closed and taps the launcher - which
    TOGGLES the overlay shut, then open again, until the switch times out. Account rows
    are only ever in the tree while the panel is open, so they prove it just as well."""
    xml = (FIX / "accounts_open.xml").read_bytes().replace(b"hl_page_close",
                                                           b"hl_page_dismiss")
    v = parse_dump(xml, WH)
    assert v.close_norm is None, "the close control is deliberately unrecognisable here"
    assert v.rows and v.panel_open is True


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


def test_the_timeout_bounds_the_whole_read_not_each_adb_call(monkeypatch):
    """`timeout` is the length of time the bot is allowed to be blind, so it has to bound
    `read()` itself.

    `Runner._refresh_accounts` calls this on the run loop's own thread: while it blocks,
    no frame is perceived, no key is read and no stop signal is serviced - and
    `Context.map_stale_since`, which the recovery ladder escalates on, counts throughout.
    Applied PER CALL it was no bound at all: three calls at the old 20s default was a 60s
    stall. Each call now gets only what is left of the one budget."""
    import time as _time
    seen = []

    class _Done:
        stdout = b""

    def fake(cmd, **kw):
        seen.append(kw["timeout"])
        _time.sleep(0.02)
        return _Done()

    monkeypatch.setattr("pogobot.accounts.subprocess.run", fake)
    UiTreeReader(WH, timeout=1.0).read()

    assert len(seen) == 3
    assert seen == sorted(seen, reverse=True), f"budget not shared across calls: {seen}"
    assert max(seen) <= 1.0 and seen[-1] < seen[0]


def test_the_default_timeout_is_a_stall_the_run_loop_can_absorb():
    """Five times the ~1s a real read costs, and deliberately shorter than the ~10s
    `uiautomator dump` spends waiting for a window that will not go idle: against a
    rendering game there is no PGSharp panel to find, so giving up first is the answer
    anyway, and `available=False` already means "could not look"."""
    assert UiTreeReader(WH).timeout <= 5.0


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

    def __init__(self, dry_run=False):
        self.applied = []
        # The real Actuator returns True in a dry run and sends nothing, so a caller that
        # wants to know whether the screen actually changed has to read this.
        self.dry_run = dry_run

    def apply(self, effect, now=None):
        self.applied.append(effect)
        return True


def _taps(act):
    return [e for e in act.applied if isinstance(e, Tap)]


def test_identify_account_opens_reads_and_recloses_the_panel():
    closed, opened = view("overlay_closed.xml"), view("accounts_open.xml")
    tr = FakeTreeReader([closed, opened])
    act = _Act()
    read = identify_account(tr, act, settle=0)
    assert read.active.name == "TrainerOne"
    # The roster leaves with it: the panel is shut for the rest of the run, so this one
    # read is the only chance to learn which accounts exist.
    assert read.names == ("TrainerOne", "TrainerTwo")
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
    read = identify_account(tr, act, settle=0)
    assert read.active is None and read.names == ()
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
    read = identify_account(tr, act, settle=0)
    assert read.active.name == "TrainerOne"
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
    read = identify_account(tr, act, settle=0)
    assert read.active.name == "TrainerOne"
    assert [(t.x, t.y) for t in _taps(act)] == [closed.launcher_norm, opened.close_norm]
    assert tr.reads == 2


def test_identify_account_does_not_claim_to_have_opened_a_panel_it_never_opened(caplog):
    """In a dry run every tap is suppressed, so the panel stays shut and the second read
    is the same closed overlay. Reporting that as "overlay opened but no account is marked
    active" asserts an action that never happened."""
    closed = view("overlay_closed.xml")
    tr = FakeTreeReader([closed, closed])
    act = _Act(dry_run=True)
    with caplog.at_level(logging.WARNING, logger="pogobot"):
        read = identify_account(tr, act, settle=0)
    assert read.active is None
    assert not any("overlay opened" in m for m in caplog.messages)
    assert any("did not open" in m and "dry run" in m for m in caplog.messages)
