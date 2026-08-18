"""Rolling 24-hour PokeStop spin quota.

Niantic caps PokeStop spins per rolling 24 hours. Past the cap, stops stop yielding and
the bot sees a stop it cannot use - which is indistinguishable, from the outside, from a
stop that is physically out of reach. That ambiguity already cost us a wrong diagnosis: a
session with 152 refused stops was reported as a positioning problem when the account had
simply spun out for the day.

The window spans restarts, so this is persisted separately from session stats. Timestamps
are wall clock (`time.time`), not `perf_counter`, whose origin resets every run.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Niantic's documented soft cap per rolling 24 hours.
DEFAULT_DAILY_LIMIT = 1200
WINDOW_SECONDS = 24 * 3600
#: Rewrite the file once it holds more than this many stale entries.
COMPACT_AT = 4000


@dataclass
class QuotaState:
    used: int
    limit: int
    remaining: int
    resets_in: Optional[float]     # seconds until the oldest in-window spin ages out
    exhausted: bool

    def line(self) -> str:
        if self.limit <= 0:
            return f"spins in the last 24h: {self.used} (no limit configured)"
        if self.exhausted:
            return (f"SPIN QUOTA REACHED: {self.used}/{self.limit} in the last 24h. "
                    f"Stops will refuse to yield until {_hms(self.resets_in)} from now - "
                    f"this looks identical to 'out of range' on screen.")
        return (f"spins in the last 24h: {self.used}/{self.limit} "
                f"({self.remaining} left" +
                (f", oldest ages out in {_hms(self.resets_in)}" if self.resets_in else "") + ")")


class SpinQuota:
    """Append-only log of spin timestamps, filtered to a rolling window."""

    def __init__(self, path: Optional[Path], limit: int = DEFAULT_DAILY_LIMIT,
                 window: float = WINDOW_SECONDS):
        self.path = Path(path) if path else None
        self.limit = limit
        self.window = window
        self._stamps: list[float] = []
        self._load()

    # ------------------------------------------------------------------ storage

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        stamps: list[float] = []
        total = 0
        try:
            with open(self.path, "r", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        ts = float(json.loads(line)["ts"])
                    except Exception:
                        continue          # a torn line from a hard kill; skip it
                    stamps.append(ts)
        except OSError:
            return                        # unreadable is not fatal; start empty
        self._stamps = sorted(stamps)
        self._prune()
        if total > COMPACT_AT:
            self._compact()

    def _prune(self, now: Optional[float] = None) -> None:
        cutoff = (now if now is not None else time.time()) - self.window
        self._stamps = [t for t in self._stamps if t >= cutoff]

    def _compact(self) -> None:
        """Rewrite the file with only the in-window entries, atomically."""
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                for ts in self._stamps:
                    fh.write(json.dumps({"ts": round(ts, 3)}) + "\n")
            os.replace(tmp, self.path)
        except OSError:
            pass

    # ------------------------------------------------------------------ use

    def record(self, now: Optional[float] = None) -> None:
        ts = time.time() if now is None else now
        self._stamps.append(ts)
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps({"ts": round(ts, 3)}) + "\n")
        except OSError:
            pass                          # losing the log must never stop the bot

    def seed(self, count: int, spread_hours: float = 12.0,
             now: Optional[float] = None) -> None:
        """Backfill spins this bot did not perform, spread over the recent past.

        The cap belongs to the account, not to this process, so spins done by hand or by
        an earlier run still count against it. Without a way to say so, the bot would
        cheerfully report 0/1200 while every stop refused.
        """
        now = time.time() if now is None else now
        if count <= 0:
            return
        step = (spread_hours * 3600.0) / max(count, 1)
        for i in range(count):
            self.record(now - (count - i) * step)

    def reset(self) -> int:
        """Forget every recorded spin. Returns how many were dropped.

        Needed because the window can be wrong in the optimistic direction as well as the
        pessimistic one: a seeded count that turns out to be stale, or a ban that lifted
        earlier than the rolling window implies, would otherwise keep the bot from
        targeting stops it can actually use.
        """
        dropped = len(self._stamps)
        self._stamps = []
        if self.path is not None:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
        return dropped

    def state(self, now: Optional[float] = None) -> QuotaState:
        now = time.time() if now is None else now
        self._prune(now)
        used = len(self._stamps)
        resets = (self._stamps[0] + self.window - now) if self._stamps else None
        return QuotaState(
            used=used,
            limit=self.limit,
            remaining=max(0, self.limit - used) if self.limit > 0 else 0,
            resets_in=resets,
            exhausted=self.limit > 0 and used >= self.limit,
        )

    @property
    def used(self) -> int:
        self._prune()
        return len(self._stamps)


def _hms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"
