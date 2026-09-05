"""The only module in the package permitted to invoke adb.

v1 spread eleven `subprocess` call sites across the file and tested `no_click` at five of
them, so `--no-click` preview mode still tapped the phone, still spun discs and still
threw balls. Here every actuation funnels through one `apply()`, so dry-run is checked
once and cannot be forgotten at a new call site.

v1 also fired every command as `subprocess.Popen(..., stdout=DEVNULL, stderr=DEVNULL)`
and never called `wait()`. Three consequences, all observed:
  * `error: device not found` was byte-for-byte indistinguishable from a successful tap,
    so an unplugged phone looked exactly like a working one for hours.
  * Nothing reaped the children; a 30 Hz loop leaked thousands of zombies per hour.
  * Concurrent Popens raced, so a swipe could reach the device before the tap that was
    supposed to precede it.
A single worker thread fixes all three: commands run in submission order, exit codes and
stderr are inspected, and the hot loop never blocks on a ~100 ms adb round trip.

Coordinates arriving here are NORMALIZED (see effects.py). This module owns the one
conversion to device pixels, so the v1 confusion between stream pixels and device pixels
cannot recur.
"""

from __future__ import annotations

import queue
import re
import subprocess
from pathlib import Path
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Mapping, Optional, Sequence

from .effects import (Back, DoubleTapDrag, Effect, ForegroundApp, Pinch, RestartApp,
                      Swipe, Tap, is_actuation)
from .observation import Tristate

DEFAULT_RESOLUTION = (1080, 2340)

#: stderr an `am start` prints when the activity is already in front. rc is 0 and
#: nothing actually failed, so this must not reach the health breaker - see
#: `Actuator._execute`.
_BENIGN_STDERR = re.compile(r"Activity not started, intent has been delivered")


#: Where the pinch injector lives on the device. /data/local/tmp is the one directory the
#: shell uid can both write and execute from, and is where scrcpy puts its own server jar
#: for the same reason.
_PINCH_REMOTE = "/data/local/tmp/pogobot-pinch.dex"
_PINCH_LOCAL = Path(__file__).resolve().parent / "vendor" / "pinch.dex"

MIN_INTERVAL = 0.25
"""Per-budget floor, in seconds. This is a safety net, not the schedule: the intended
pacing lives in Timings and is applied by Context.ready(). The floor exists so that a
future handler bug cannot machine-gun adb the way v1's unguarded branches could."""

MIN_GLOBAL_GAP = 0.08
"""Floor between any two actuations regardless of budget. v1 emitted a tap and a swipe
in the same tick and the device dropped one of them."""

_CLOCK = time.perf_counter
"""One clock for the whole package. frames.Frame.ts and Runner.ctx.now are perf_counter,
and Runner passes ctx.now into apply(); defaulting to a different clock here would mix
two epochs in _last_any (they are not the same epoch on every platform), and a single
jump forward would rate-limit every future actuation for the life of the process."""

ADB_TIMEOUT = 5.0
QUEUE_DEPTH = 32

_SIZE_RE = re.compile(r"(Override|Physical) size:\s*(\d+)x(\d+)")


@dataclass(frozen=True)
class Command:
    """One realized device command: the argv plus the effect that produced it.

    Kept as data so a dry run, a replay and a live run all describe their intent in the
    same vocabulary and can be diffed against each other.
    """

    argv: tuple[str, ...]
    budget: str
    reason: str
    device_xy: Optional[tuple[int, int]] = None
    ts: float = 0.0

    def __str__(self) -> str:
        return f"{' '.join(self.argv)}  # {self.reason}"


@dataclass
class _Health:
    """Consecutive-failure circuit breaker.

    v1 had no notion of adb health, so a disconnected phone produced an hour of confident
    log lines about targets acquired and discs spun. Once `tripped` is set the actuator
    refuses to act and the runner is expected to Halt rather than flail.
    """

    max_failures: int = 5
    consecutive: int = 0
    tripped: bool = False
    last_error: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_ok(self) -> None:
        with self.lock:
            self.consecutive = 0

    def record_failure(self, detail: str) -> None:
        with self.lock:
            self.consecutive += 1
            self.last_error = detail
            if self.consecutive >= self.max_failures:
                self.tripped = True

    def snapshot(self) -> tuple[int, bool, Optional[str]]:
        with self.lock:
            return self.consecutive, self.tripped, self.last_error

    def reset(self) -> None:
        with self.lock:
            self.consecutive = 0
            self.tripped = False
            self.last_error = None


def _adb_argv(adb: str, serial: Optional[str], *args: str) -> tuple[str, ...]:
    head = (adb,) if serial is None else (adb, "-s", serial)
    return head + args


def device_resolution(adb: str = "adb", serial: Optional[str] = None,
                      timeout: float = 4.0) -> tuple[int, int]:
    """Read `wm size`, preferring the Override line.

    The LG G820 reports `Physical size: 1440x3120` and `Override size: 1080x2340`; input
    events land in the override space, so taking the physical line would misplace every
    tap by 33%.
    """
    try:
        out = subprocess.run(
            _adb_argv(adb, serial, "shell", "wm", "size"),
            capture_output=True, timeout=timeout, check=False,
        ).stdout.decode(errors="replace")
    except (OSError, subprocess.SubprocessError):
        return DEFAULT_RESOLUTION
    sizes = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in _SIZE_RE.finditer(out)}
    return sizes.get("Override") or sizes.get("Physical") or DEFAULT_RESOLUTION


class Actuator:
    """Applies Tap/Swipe/Back to a real device. Everything else is ignored, not an error.

    `apply()` returns whether the effect was actually dispatched to the device, so a
    caller can tell "suppressed" from "sent" without re-deriving dry-run, rate limit or
    breaker state - the three things v1 re-derived, inconsistently, at every call site.
    """

    def __init__(self, screen_wh: tuple[int, int], dry_run: bool = False,
                 adb: str = "adb", serial: Optional[str] = None,
                 max_failures: int = 5,
                 min_interval: float = MIN_INTERVAL,
                 intervals: Optional[Mapping[str, float]] = None,
                 timeout: float = ADB_TIMEOUT,
                 history: int = 64):
        w, h = int(screen_wh[0]), int(screen_wh[1])
        if w <= 0 or h <= 0:
            raise ValueError(f"screen_wh must be positive, got {screen_wh}")
        self.screen_wh = (w, h)
        self.dry_run = bool(dry_run)
        self.adb = adb
        self.serial = serial
        self.timeout = timeout
        self.min_interval = min_interval
        self.intervals = dict(intervals or {})

        self.health = _Health(max_failures=max(1, int(max_failures)))
        self.log: Deque[Command] = deque(maxlen=history)

        self._last_by_budget: dict[str, float] = {}
        self._last_any: float = -1e9
        self._counts = {
            "sent": 0, "ok": 0, "failed": 0, "timed_out": 0,
            "suppressed_dry_run": 0, "rate_limited": 0,
            "dropped_backpressure": 0, "suppressed_unhealthy": 0,
        }
        self._by_budget: dict[str, int] = {}
        self._counts_lock = threading.Lock()

        self._q: "queue.Queue[Optional[Command]]" = queue.Queue(maxsize=QUEUE_DEPTH)
        self._closed = False
        self._worker: Optional[threading.Thread] = None

    # ------------------------------------------------------------ geometry

    def to_device(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        """Normalized 0..1 -> device pixels, clamped inside the screen.

        Clamping is not politeness: v1 computed tap coordinates from stream-pixel
        detections scaled by a resolution it had guessed, and an off-by-one on the right
        edge produced taps at x == width that Android silently discarded, which read as
        "the tap did nothing" and drove the bot into its retry loop.
        """
        w, h = self.screen_wh
        px = int(round(min(max(x_norm, 0.0), 1.0) * (w - 1)))
        py = int(round(min(max(y_norm, 0.0), 1.0) * (h - 1)))
        return min(max(px, 0), w - 1), min(max(py, 0), h - 1)

    def _ensure_pinch(self) -> None:
        """Put the injector on the device once per process.

        Failure is deliberately not raised: the caller is mid-render, and a zoom that
        cannot be pushed should cost the run a zoom, not the run. The command that follows
        will fail on its own and be counted like any other failed actuation.
        """
        if getattr(self, "_pinch_pushed", False) or self.dry_run:
            return
        self._pinch_pushed = True
        try:
            subprocess.run(_adb_argv(self.adb, self.serial, "push",
                                     str(_PINCH_LOCAL), _PINCH_REMOTE),
                           capture_output=True, timeout=self.timeout)
        except Exception:
            # Swallowed on purpose, and this module has no logger by design: the caller is
            # mid-render and the command that follows will fail on its own, be counted in
            # `_counts["failed"]` like any other refused actuation, and show up in
            # `stats()`. Raising here would turn a missing zoom into a dead run.
            pass

    def render(self, effect: Effect) -> Optional[Command]:
        """Effect -> Command, or None if the effect is not an actuation."""
        if isinstance(effect, Tap):
            px, py = self.to_device(effect.x, effect.y)
            return Command(_adb_argv(self.adb, self.serial, "shell", "input", "tap",
                                     str(px), str(py)),
                           effect.budget, effect.reason, (px, py), _CLOCK())
        if isinstance(effect, Swipe):
            x1, y1 = self.to_device(effect.x1, effect.y1)
            x2, y2 = self.to_device(effect.x2, effect.y2)
            ms = max(1, int(effect.duration_ms))
            return Command(_adb_argv(self.adb, self.serial, "shell", "input", "swipe",
                                     str(x1), str(y1), str(x2), str(y2), str(ms)),
                           effect.budget, effect.reason, (x2, y2), _CLOCK())
        if isinstance(effect, Back):
            return Command(_adb_argv(self.adb, self.serial, "shell", "input", "keyevent", "4"),
                           effect.budget, effect.reason, None, _CLOCK())
        if isinstance(effect, ForegroundApp):
            # No force-stop: the game is alive, just behind something else. `am start` on
            # its main activity raises the existing task rather than launching a new one.
            return Command(_adb_argv(self.adb, self.serial, "shell", "am", "start", "-n",
                                     f"{effect.package}/{effect.activity}"),
                           effect.budget, effect.reason, None, _CLOCK())
        if isinstance(effect, Pinch):
            # Pushed lazily rather than at construction: a dry run, a replay and every unit
            # test build an Actuator, and none of them should touch a device. The file is
            # 3.3KB and the push is once per process.
            self._ensure_pinch()
            w, h = self.screen_wh
            px, py = self.to_device(effect.x, effect.y)
            g0 = max(1, int(effect.start_gap * h))
            g1 = max(1, int(effect.end_gap * h))
            shell_cmd = (f"CLASSPATH={_PINCH_REMOTE} app_process / pinch.Pinch "
                         f"{px} {py} {g0} {g1} {int(effect.steps)} {int(effect.duration_ms)}")
            return Command(_adb_argv(self.adb, self.serial, "shell", shell_cmd),
                           effect.budget, effect.reason, (px, py), _CLOCK())
        if isinstance(effect, DoubleTapDrag):
            # Measured on the device: multi-touch is unavailable (sendevent blocked by
            # SELinux, `input motionevent` is single-pointer, two concurrent `input
            # swipe`s do nothing), but a tap immediately followed by a press-and-drag
            # from the same point IS a single pointer, and Android reads it as its
            # one-finger pinch gesture. The two commands MUST be one adb invocation - a
            # separate `subprocess.run` per command would insert adb's own round-trip
            # latency between them and miss the double-tap window `input` needs to chain
            # tap into drag rather than reading them as two independent touches.
            x1, y1 = self.to_device(effect.x1, effect.y1)
            x2, y2 = self.to_device(effect.x2, effect.y2)
            ms = max(1, int(effect.duration_ms))
            shell_cmd = (f"input tap {x1} {y1}; "
                        f"input swipe {x1} {y1} {x2} {y2} {ms}")
            return Command(_adb_argv(self.adb, self.serial, "shell", shell_cmd),
                           effect.budget, effect.reason, (x2, y2), _CLOCK())
        if isinstance(effect, RestartApp):
            # One invocation, for the same reason DoubleTapDrag is one: a separate
            # `subprocess.run` per command would let the worker interleave something else
            # between the stop and the start, and - worse here than there - a failure
            # partway through would leave the game force-stopped with nothing having
            # relaunched it, which is a bot staring at the Android launcher rather than a
            # bot that is merely still stuck.
            #
            # The sleep is the settle described on the effect; `am start` fired straight
            # after `am force-stop` races the teardown. It runs on the DEVICE rather than
            # here so the whole thing stays one command, and it is bounded by ADB_TIMEOUT
            # above, which is what makes a long settle a configuration error rather than a
            # silent hang.
            # `:g` so a whole number of seconds is written as an integer: toybox `sleep`
            # takes fractions, but an integer is what every shell that has ever shipped on
            # Android accepts, and there is nothing to gain from the more exotic form.
            settle = max(0, int(effect.settle_ms)) / 1000.0
            shell_cmd = (f"am force-stop {effect.package}; "
                         f"sleep {settle:g}; "
                         f"am start -n {effect.package}/{effect.activity}")
            return Command(_adb_argv(self.adb, self.serial, "shell", shell_cmd),
                           effect.budget, effect.reason, None, _CLOCK())
        return None

    # ------------------------------------------------------------- dispatch

    def apply(self, effect: Effect, now: Optional[float] = None) -> bool:
        """Dispatch one effect.

        Returns whether the effect was ACCEPTED - i.e. it consumed its budget and, in a
        live run, went to the device. It is deliberately NOT "bytes reached the phone":
        the caller uses this to advance its own pacing state (last_action, settle_until,
        taps_in_state), so a dry run that returned False here would make the FSM re-emit
        the same Tap on every single tick. Measured: 6 taps over 6 ticks under dry_run
        against 1 tap live. A preview that does not follow the live trajectory is the v1
        `--no-click` defect wearing a different hat, so dry_run returns True and the
        *dispatch* question is answered by stats(): `sent`/`ok` vs `suppressed_dry_run`.

        False means suppressed: not an actuation, closed, breaker tripped, rate limited,
        or dropped for backpressure.
        """
        if not is_actuation(effect):
            return False
        now = _CLOCK() if now is None else now

        if self._closed:
            # Without this, _ensure_worker() below would resurrect the worker thread
            # after close() and keep talking to a device the runner has already let go.
            self._bump("suppressed_unhealthy")
            return False

        _, tripped, _ = self.health.snapshot()
        if tripped:
            self._bump("suppressed_unhealthy")
            return False

        cmd = self.render(effect)
        if cmd is None:
            return False

        if not self._allowed(cmd.budget, now):
            self._bump("rate_limited")
            return False

        self.log.append(cmd)
        self._bump_budget(cmd.budget)

        if self.dry_run:
            self._bump("suppressed_dry_run")
            self._mark(cmd.budget, now)
            return True

        # Before the put, not after: a full queue returns early below, so starting the
        # worker afterwards meant a dead worker plus a full queue could never recover.
        self._ensure_worker()
        try:
            self._q.put_nowait(cmd)
        except queue.Full:
            self._bump("dropped_backpressure")
            self.health.record_failure("adb queue full; device is not keeping up")
            return False

        self._mark(cmd.budget, now)
        self._bump("sent")
        return True

    def apply_all(self, effects: Sequence[Effect], now: Optional[float] = None) -> int:
        return sum(1 for e in effects if self.apply(e, now))

    def _allowed(self, budget: str, now: float) -> bool:
        if now - self._last_any < MIN_GLOBAL_GAP:
            return False
        gap = self.intervals.get(budget, self.min_interval)
        return now - self._last_by_budget.get(budget, -1e9) >= gap

    def _mark(self, budget: str, now: float) -> None:
        self._last_by_budget[budget] = now
        self._last_any = now

    # --------------------------------------------------------------- worker

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, name="adb-actuator", daemon=True)
            self._worker.start()

    def _run(self) -> None:
        while True:
            cmd = self._q.get()
            try:
                if cmd is None:
                    return
                try:
                    self._execute(cmd)
                except BaseException as exc:      # noqa: BLE001 - the worker must not die
                    # A worker that dies silently is the v1 failure exactly: commands
                    # vanish while healthy() keeps saying yes. Record it and stay up.
                    self._bump("failed")
                    self.health.record_failure(f"worker error: {exc!r}")
            finally:
                self._q.task_done()

    def _execute(self, cmd: Command) -> None:
        """Run one adb command to completion and judge it.

        Judging on the exit code alone is not enough: some adb builds print
        `error: device not found` and still exit 0, which is precisely the failure v1
        could not see, so a non-empty stderr counts as a failure too.
        """
        try:
            proc = subprocess.run(cmd.argv, capture_output=True,
                                  timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired:
            self._bump("timed_out")
            self._bump("failed")
            self.health.record_failure(f"timed out after {self.timeout:.1f}s: {' '.join(cmd.argv)}")
            return
        except OSError as exc:
            self._bump("failed")
            self.health.record_failure(f"{exc}: {' '.join(cmd.argv)}")
            return

        err = proc.stderr.decode(errors="replace").strip()
        if err and proc.returncode == 0 and _BENIGN_STDERR.search(err):
            # `am start` on an activity that is already top exits 0 and prints "Warning:
            # Activity not started, intent has been delivered to currently running
            # top-most instance." Scored as a failure, every ForegroundApp against a
            # perfectly healthy device counts toward the breaker: measured in logs/run.log,
            # by_budget {'foreground': 3} and all 3 of that run's failures, with last_error
            # carrying exactly that string at rc=0. Five in a row trip `max_failures` and
            # halt a run that had nothing wrong with it.
            err = ""
        if proc.returncode != 0 or err:
            self._bump("failed")
            self.health.record_failure(f"rc={proc.returncode} {err or '(no stderr)'}")
            return

        self._bump("ok")
        self.health.record_ok()

    # ------------------------------------------------------------ reporting

    def _bump(self, key: str, n: int = 1) -> None:
        with self._counts_lock:
            self._counts[key] = self._counts.get(key, 0) + n

    def _bump_budget(self, budget: str) -> None:
        with self._counts_lock:
            self._by_budget[budget] = self._by_budget.get(budget, 0) + 1

    def healthy(self) -> bool:
        _, tripped, _ = self.health.snapshot()
        return not tripped

    def stats(self) -> dict:
        consecutive, tripped, last_error = self.health.snapshot()
        with self._counts_lock:
            counts = dict(self._counts)
            by_budget = dict(self._by_budget)
        counts.update(
            healthy=not tripped,
            consecutive_failures=consecutive,
            max_failures=self.health.max_failures,
            last_error=last_error,
            pending=self._q.qsize(),
            dry_run=self.dry_run,
            screen_wh=self.screen_wh,
            by_budget=by_budget,
        )
        return counts

    def recent(self, n: int = 10) -> list[Command]:
        return list(self.log)[-n:]

    def reset(self) -> None:
        """Re-arm the breaker. Only the runner calls this, and only after it has proved
        the device is back (see `probe`)."""
        self.health.reset()

    def probe(self) -> bool:
        """Synchronous liveness check. Blocking is fine here: it runs on the halt path,
        not in the hot loop."""
        try:
            proc = subprocess.run(_adb_argv(self.adb, self.serial, "get-state"),
                                  capture_output=True, timeout=self.timeout, check=False)
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0 and proc.stdout.decode(errors="replace").strip() == "device"

    def flush(self, timeout: float = 3.0) -> bool:
        """Wait for queued commands to drain. Used at shutdown so the last BACK actually
        reaches the phone before the process exits."""
        deadline = _CLOCK() + timeout
        while self._q.unfinished_tasks and _CLOCK() < deadline:
            time.sleep(0.02)
        return not self._q.unfinished_tasks

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.flush()
        worker = self._worker
        if worker is not None and worker.is_alive():
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass
            worker.join(timeout=self.timeout + 1.0)

    release = close

    def __enter__(self) -> "Actuator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class NullActuator:
    """Records what a real actuator would have done. For replay and unit tests.

    Deliberately not a subclass: nothing it does may reach subprocess, and inheriting
    would make that a one-line mistake away. It renders the same Command objects, so a
    replay transcript is directly comparable with a live run's `recent()`.
    """

    def __init__(self, screen_wh: tuple[int, int] = DEFAULT_RESOLUTION,
                 adb: str = "adb", serial: Optional[str] = None):
        w, h = int(screen_wh[0]), int(screen_wh[1])
        if w <= 0 or h <= 0:
            raise ValueError(f"screen_wh must be positive, got {screen_wh}")
        self.screen_wh = (w, h)
        self.dry_run = True
        self.adb = adb
        self.serial = serial
        self.log: list[Command] = []
        self._by_budget: dict[str, int] = {}

    to_device = Actuator.to_device
    render = Actuator.render

    def apply(self, effect: Effect, now: Optional[float] = None) -> bool:
        """True on acceptance, matching Actuator.apply, so a replayed trajectory is the
        same trajectory. Nothing here can reach subprocess, so True never means a tap."""
        cmd = self.render(effect) if is_actuation(effect) else None
        if cmd is None:
            return False
        self.log.append(cmd)
        self._by_budget[cmd.budget] = self._by_budget.get(cmd.budget, 0) + 1
        return True

    def apply_all(self, effects: Sequence[Effect], now: Optional[float] = None) -> int:
        return sum(1 for e in effects if self.apply(e, now))

    def healthy(self) -> bool:
        return True

    def probe(self) -> bool:
        return True

    def stats(self) -> dict:
        return {
            "sent": 0, "ok": 0, "failed": 0, "timed_out": 0,
            "suppressed_dry_run": len(self.log), "rate_limited": 0,
            "dropped_backpressure": 0, "suppressed_unhealthy": 0,
            "healthy": True, "consecutive_failures": 0, "max_failures": 0,
            "last_error": None, "pending": 0, "dry_run": True,
            "screen_wh": self.screen_wh, "by_budget": dict(self._by_budget),
        }

    def recent(self, n: int = 10) -> list[Command]:
        return self.log[-n:]

    def reset(self) -> None:
        return None

    def flush(self, timeout: float = 0.0) -> bool:
        return True

    def close(self) -> None:
        return None

    release = close

    def __enter__(self) -> "NullActuator":
        return self

    def __exit__(self, *exc) -> None:
        return None
