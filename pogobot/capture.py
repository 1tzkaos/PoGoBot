"""Frame sources: the live scrcpy stream and a deterministic on-disk replay.

The drop-to-latest reader thread from v1 is kept - it is the reason inference never
falls behind the stream. Everything around it is rebuilt, because v1's capture layer
could not express failure:

  * `read()` returned `(True, last_frame.copy())` forever once scrcpy died, so the bot
    kept tapping a phone it could not see. Frames are now stamped with a monotonic seq
    and a `perf_counter` timestamp under the lock, and `read()` returns None once the
    newest frame is older than `cfg.timings.frame_max_age`.
  * The scrcpy process was bound once and never polled again, with stderr to DEVNULL,
    so a dead or never-started stream had no symptom and no diagnosis.
  * `cv2.VideoCapture` on a named FIFO blocks in `open()` until a writer appears. With
    scrcpy missing or the device unauthorized, v1 hung silently and forever.
  * `cleanup()` could run up to four times per exit and released the capture while the
    reader thread was inside `cap.read()`.
  * The FIFO path was one fixed `/tmp` constant shared by every instance, so a second
    run stole the first run's pipe.

ReplaySource is the seam that lets the whole bot - perception, fsm, runner - execute
with no device attached.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import IO, Optional, Sequence

import cv2
import numpy as np

from .config import Config
from .frames import Frame

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

SCRCPY_CANDIDATES = ("scrcpy", "/opt/homebrew/bin/scrcpy", "/usr/local/bin/scrcpy")

_READ_IDLE = 0.002
_EOF_MISSES = 250
_JOIN_TIMEOUT = 2.0


class CaptureError(RuntimeError):
    """Raised at construction when a live stream cannot be established.

    A loud failure at startup replaces v1's silent indefinite block on the FIFO.
    """


def find_scrcpy() -> str:
    """Locate the scrcpy binary before spawning, so 'not installed' is an error message
    rather than an open() that never returns."""
    for candidate in SCRCPY_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise CaptureError(
        "scrcpy not found on PATH or in " + ", ".join(SCRCPY_CANDIDATES[1:])
        + " - install it with `brew install scrcpy`"
    )


def scrcpy_argv(cfg: Config, record_path: Path, serial: Optional[str] = None,
                binary: Optional[str] = None) -> list[str]:
    """Build the scrcpy command line from Config alone.

    v1 hard-coded these flags inside the launcher, so `--max-size` could disagree with
    the resolution the thresholds were tuned against with nothing to compare.
    """
    argv = [binary or find_scrcpy()]
    if serial:
        argv += ["-s", serial]
    argv += [
        "--no-audio",
        "--no-playback",
        f"--max-size={cfg.max_size}",
        f"--max-fps={cfg.max_fps}",
        "--video-bit-rate=8M",
        f"--record={record_path}",
        "--record-format=mkv",
    ]
    return argv


def _poke_fifo(path: Path) -> None:
    """Give a blocked reader a writer so its thread can finish instead of leaking.

    Opening the write end non-blocking succeeds as soon as a reader is parked in open();
    closing it immediately delivers EOF and `cv2.VideoCapture` returns.
    """
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        return
    os.close(fd)


class ScrcpySource:
    """Live capture from scrcpy recording into a private FIFO.

    Only the newest frame is retained; everything the reader thread pulls while
    inference is busy is dropped. That part of v1 was sound and is kept verbatim in
    spirit - the difference is that the newest frame now carries its own age.
    """

    def __init__(self, cfg: Config, serial: Optional[str] = None,
                 start_timeout: float = 12.0, binary: Optional[str] = None) -> None:
        self.cfg = cfg
        self._dir = Path(tempfile.mkdtemp(prefix="pogobot-capture-"))
        self._fifo = self._dir / "stream.mkv"
        self._stderr_path = self._dir / "scrcpy.stderr"

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._release_lock = threading.Lock()
        self._released = False

        self._latest: Optional[np.ndarray] = None
        self._latest_seq = 0
        self._latest_ts = 0.0
        self._seq = 0
        self._reader_error: Optional[str] = None

        self._proc: Optional[subprocess.Popen] = None
        self._stderr: Optional[IO[bytes]] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None

        try:
            os.mkfifo(self._fifo)
            self._spawn(serial, binary)
            self._cap = self._open_capture(start_timeout)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._thread = threading.Thread(target=self._reader, name="pogobot-capture",
                                            daemon=True)
            self._thread.start()
        except BaseException:
            self.release()
            raise

    # ------------------------------------------------------------- FrameSource

    def read(self) -> Optional[Frame]:
        """The newest frame, or None when there is none or it has gone stale."""
        with self._lock:
            bgr, seq, ts = self._latest, self._latest_seq, self._latest_ts
        if bgr is None:
            return None
        if time.perf_counter() - ts > self.cfg.timings.frame_max_age:
            return None
        return Frame(seq=seq, ts=ts, bgr=bgr.copy())

    def healthy(self) -> bool:
        if self._released or self._reader_error is not None:
            return False
        return self._proc is not None and self._proc.poll() is None

    def release(self) -> None:
        """Idempotent teardown in reader-first order.

        v1 could reach `cleanup()` four times per exit and freed the capture out from
        under a thread that was inside `cap.read()`.
        """
        with self._release_lock:
            if self._released:
                return
            self._released = True

        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(_JOIN_TIMEOUT)

        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=_JOIN_TIMEOUT)
            except subprocess.TimeoutExpired:
                self._proc.kill()

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        if self._stderr is not None:
            try:
                self._stderr.close()
            except Exception:
                pass
            self._stderr = None

        shutil.rmtree(self._dir, ignore_errors=True)

    # ------------------------------------------------------------- diagnosis

    @property
    def fifo_path(self) -> Path:
        return self._fifo

    @property
    def returncode(self) -> Optional[int]:
        return None if self._proc is None else self._proc.poll()

    def failure_reason(self) -> str:
        """Why the stream is unhealthy, in words a Note can carry to the log."""
        if self._released:
            return "capture released"
        if self._reader_error is not None:
            return self._reader_error
        code = self.returncode
        if code is not None:
            return f"scrcpy exited with code {code}: {self.stderr_tail()}"
        return ""

    def stderr_tail(self, limit: int = 600) -> str:
        try:
            text = self._stderr_path.read_text(errors="replace").strip()
        except OSError:
            return "<no scrcpy stderr captured>"
        return text[-limit:] if text else "<scrcpy wrote nothing to stderr>"

    def __enter__(self) -> "ScrcpySource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    # ------------------------------------------------------------- internals

    def _spawn(self, serial: Optional[str], binary: Optional[str]) -> None:
        argv = scrcpy_argv(self.cfg, self._fifo, serial=serial, binary=binary)
        self._stderr = open(self._stderr_path, "wb")
        try:
            self._proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                          stderr=self._stderr)
        except OSError as exc:
            raise CaptureError(f"could not start scrcpy ({' '.join(argv)}): {exc}") from exc

    def _open_capture(self, timeout: float) -> cv2.VideoCapture:
        """Open the FIFO off-thread with a bounded wait.

        `cv2.VideoCapture` on a FIFO parks in open() until scrcpy writes the first mkv
        header. When scrcpy never gets that far - not installed, device unauthorized,
        no device - v1 waited forever with no message at all.
        """
        box: list[object] = []

        def opener() -> None:
            try:
                box.append(cv2.VideoCapture(str(self._fifo)))
            except Exception as exc:
                box.append(exc)

        thread = threading.Thread(target=opener, name="pogobot-capture-open", daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            _poke_fifo(self._fifo)
            thread.join(1.0)
            raise CaptureError(
                f"no video on {self._fifo} within {timeout:.0f}s. "
                f"scrcpy returncode={self.returncode}; stderr: {self.stderr_tail()}"
            )

        result = box[0] if box else None
        if isinstance(result, Exception):
            raise CaptureError(f"opening {self._fifo} failed: {result}") from result
        if result is None or not result.isOpened():
            raise CaptureError(
                f"cv2 could not decode {self._fifo}. "
                f"scrcpy returncode={self.returncode}; stderr: {self.stderr_tail()}"
            )
        return result

    def _reader(self) -> None:
        misses = 0
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                return
            try:
                ok, bgr = cap.read()
            except Exception as exc:
                self._reader_error = f"capture read raised: {exc}"
                return
            if not ok or bgr is None:
                misses += 1
                if misses > _EOF_MISSES and (self._proc is None or self._proc.poll() is not None):
                    self._reader_error = (
                        f"stream ended; scrcpy returncode={self.returncode}: "
                        f"{self.stderr_tail()}"
                    )
                    return
                self._stop.wait(_READ_IDLE)
                continue
            misses = 0
            with self._lock:
                self._seq += 1
                self._latest = bgr
                self._latest_seq = self._seq
                self._latest_ts = time.perf_counter()


def _natural_key(path: Path) -> tuple:
    """Sort `shot_2.png` before `shot_10.png`.

    Plain lexicographic order replays a run out of sequence, which turns a regression
    test on recorded frames into noise.
    """
    parts: list = []
    digits = ""
    for ch in path.name:
        if ch.isdigit():
            digits += ch
        else:
            if digits:
                parts.append((1, int(digits), ""))
                digits = ""
            parts.append((0, 0, ch))
    if digits:
        parts.append((1, int(digits), ""))
    return tuple(parts)


class ReplaySource:
    """Frames from a directory of images, in name order.

    Timestamps are stamped at delivery, not at load, so a replayed frame can never be
    reported stale no matter how slow the consumer is - staleness is a property of the
    live stream, and a recorded run must not inherit it.

    `interval` defaults to 0.0, meaning one frame per `read()`: the run is then a pure
    function of the directory contents and completely reproducible. A positive interval
    paces delivery against the wall clock for eyeball debugging, at the cost of that
    determinism.
    """

    def __init__(self, directory: Path | str, interval: float = 0.0, loop: bool = False,
                 suffixes: Sequence[str] = IMAGE_SUFFIXES) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise CaptureError(f"replay directory does not exist: {self.directory}")
        allowed = {s.lower() for s in suffixes}
        self.paths: list[Path] = sorted(
            (p for p in self.directory.iterdir()
             if p.is_file() and p.suffix.lower() in allowed),
            key=_natural_key,
        )
        if not self.paths:
            raise CaptureError(f"no images matching {sorted(allowed)} in {self.directory}")

        self.interval = max(0.0, float(interval))
        self.loop = loop
        self._index = 0
        self._seq = 0
        self._next_due = 0.0
        self._exhausted = False
        self._released = False
        self._current: Optional[Path] = None

    def read(self) -> Optional[Frame]:
        if self._released or self._exhausted:
            return None
        now = time.perf_counter()
        if self.interval > 0.0 and now < self._next_due:
            return None

        while self._index < len(self.paths):
            path = self.paths[self._index]
            self._index += 1
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            self._seq += 1
            self._current = path
            self._next_due = now + self.interval
            if self._index >= len(self.paths):
                if self.loop:
                    self._index = 0
                else:
                    self._exhausted = True
            return Frame(seq=self._seq, ts=time.perf_counter(), bgr=bgr)

        self._exhausted = True
        return None

    def healthy(self) -> bool:
        return not self._released and not self._exhausted

    def release(self) -> None:
        self._released = True

    @property
    def current_path(self) -> Optional[Path]:
        """Which file produced the last frame - the replay equivalent of a log line."""
        return self._current

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def __enter__(self) -> "ReplaySource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
