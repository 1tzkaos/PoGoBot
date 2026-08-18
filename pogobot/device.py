"""Device facts and slow device queries, kept off the hot loop.

v1 called `adb shell dumpsys input_method` synchronously inside the main loop. Measured on
this machine it takes 0.08-0.11s, so every check stalled the loop for ~10 frames. Worse,
v1's 2-second throttle only advanced when the keyboard was actually found, so in the
common case (no keyboard) the blocking call ran on EVERY iteration. Here it runs on a
background thread and the loop reads a cached Tristate.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Optional

from .observation import Tristate

DEFAULT_WH = (1080, 2340)


def screen_size(adb: str = "adb", serial: Optional[str] = None) -> tuple[int, int]:
    """Logical display size. Override wins: `adb input tap` uses the override space."""
    cmd = [adb] + (["-s", serial] if serial else []) + ["shell", "wm", "size"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=5).stdout.decode()
    except Exception:
        return DEFAULT_WH
    for pattern in (r"Override size:\s*(\d+)x(\d+)", r"Physical size:\s*(\d+)x(\d+)"):
        m = re.search(pattern, out)
        if m:
            return int(m.group(1)), int(m.group(2))
    return DEFAULT_WH


def device_online(adb: str = "adb", serial: Optional[str] = None) -> bool:
    cmd = [adb] + (["-s", serial] if serial else []) + ["get-state"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=5)
        return r.returncode == 0 and b"device" in r.stdout
    except Exception:
        return False


class KeyboardPoller:
    """Publishes a Tristate the loop can read for free.

    UNKNOWN is a real answer: v1 swallowed every failure into False, so an adb error read
    as a confident 'no keyboard is up'.
    """

    def __init__(self, adb: str = "adb", serial: Optional[str] = None, interval: float = 2.0):
        self._cmd = [adb] + (["-s", serial] if serial else []) + ["shell", "dumpsys", "input_method"]
        self._interval = interval
        self._state = Tristate.UNKNOWN
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="keyboard-poller", daemon=True)

    def start(self) -> "KeyboardPoller":
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(self._cmd, capture_output=True, timeout=4).stdout.decode()
                value = Tristate.TRUE if "mInputShown=true" in out else Tristate.FALSE
            except Exception:
                value = Tristate.UNKNOWN
            with self._lock:
                self._state = value
            self._stop.wait(self._interval)

    @property
    def state(self) -> Tristate:
        with self._lock:
            return self._state

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
