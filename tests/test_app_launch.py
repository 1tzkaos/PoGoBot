"""Starting the bot with the game closed used to kill it before the FSM ever ran.

    CaptureError: no video on .../stream.mkv within 12.0s. scrcpy returncode=None

scrcpy records the display through a hardware encoder, and an encoder emits frames when
the display CHANGES. On a launcher sitting perfectly still nothing is produced at all, so
`ScrcpySource` waits out its whole start budget and raises. Reproduced exactly that way -
`am force-stop`, start the bot, crash - and the identical start succeeds the moment the
game is on screen, because the game animates constantly.

Note what is NOT the cause, both measured: the game merely being backgrounded is fine
(897KB recorded on the launcher with the game alive behind it), and so is a sleeping screen
(6MB recorded with `mWakefulness=Asleep`). It is specifically a still display.
"""
from __future__ import annotations

import subprocess

import pytest

from pogobot import device


class _Fake:
    """Stands in for subprocess.run, recording every argv it is handed."""

    def __init__(self, focus_after: "int | None" = 0, pid: bool = False):
        self.calls: list[list[str]] = []
        #: dumpsys calls before the game appears in front; None = it never does.
        self.focus_after = focus_after
        self._focus_seen = 0
        self.pid = pid

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        out = b""
        if "dumpsys" in cmd:
            self._focus_seen += 1
            ahead = (self.focus_after is not None
                     and self._focus_seen > self.focus_after)
            name = "com.nianticlabs.pokemongo" if ahead else "com.lge.launcher3"
            out = f"  mCurrentFocus=Window{{a u0 {name}/x.Main}}\n".encode()
        elif "pidof" in cmd:
            out = b"1234\n" if self.pid else b""
        return subprocess.CompletedProcess(cmd, 0, out, b"")

    @property
    def started(self) -> bool:
        return any("am" in c and "start" in c for c in self.calls)


def test_a_game_already_in_front_is_left_alone(monkeypatch):
    """No `am start` when it is already there - restarting a healthy game would throw away
    the very state the preflight is about to configure."""
    fake = _Fake(focus_after=0)
    monkeypatch.setattr(device.subprocess, "run", fake)
    assert device.ensure_app_running("com.nianticlabs.pokemongo", "x.Main")
    assert not fake.started


def test_a_closed_game_is_started_before_capture_opens(monkeypatch):
    fake = _Fake(focus_after=1)
    monkeypatch.setattr(device.subprocess, "run", fake)
    monkeypatch.setattr(device.time, "sleep", lambda *_: None)
    assert device.ensure_app_running("com.nianticlabs.pokemongo", "x.Main")
    assert fake.started, "the game was never started"
    start = next(c for c in fake.calls if "start" in c)
    assert "com.nianticlabs.pokemongo/x.Main" in start


def test_it_gives_up_rather_than_hanging(monkeypatch):
    """A game that never comes up must not wedge startup. False is the answer, not a
    raise: BOOT has its own budget and may still find the map."""
    fake = _Fake(focus_after=None)          # never reaches the foreground
    monkeypatch.setattr(device.subprocess, "run", fake)
    monkeypatch.setattr(device.time, "sleep", lambda *_: None)
    assert device.ensure_app_running("com.nianticlabs.pokemongo", "x.Main",
                                     timeout=1.0) is False


def test_adb_failure_is_not_fatal(monkeypatch):
    """The phone can vanish mid-start; that is the caller's problem to report, not an
    exception out of a helper."""
    def boom(*a, **k):
        raise OSError("adb died")
    monkeypatch.setattr(device.subprocess, "run", boom)
    assert device.ensure_app_running("com.nianticlabs.pokemongo", "x.Main",
                                     timeout=1.0) is False


def test_the_serial_is_threaded_through(monkeypatch):
    fake = _Fake(focus_after=1)
    monkeypatch.setattr(device.subprocess, "run", fake)
    monkeypatch.setattr(device.time, "sleep", lambda *_: None)
    device.ensure_app_running("com.nianticlabs.pokemongo", "x.Main", serial="ABC123")
    assert all(c[:3] == ["adb", "-s", "ABC123"] for c in fake.calls), fake.calls


def test_the_cli_starts_the_game_before_it_opens_capture():
    """Order is the whole point: after ScrcpySource is too late, because the constructor
    is what raises."""
    import inspect
    from pogobot import cli
    src = inspect.getsource(cli.main)
    # The CALL, not the name: `ensure_app_running` also appears in an import inside main,
    # so asserting the bare name passes even with the call deleted (checked by mutation).
    call = "ensure_app_running(cfg.app_package"
    assert call in src, "the cli never starts the game"
    assert src.index(call) < src.index("ScrcpySource(cfg"), \
        "the game must be started BEFORE capture opens"
