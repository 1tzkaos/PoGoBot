"""Rolling 24-hour PokeStop spin quota.

Niantic caps PokeStop spins per rolling 24 hours. Past the cap, stops stop yielding and
the bot sees a stop it cannot use - which is indistinguishable, from the outside, from a
stop that is physically out of reach. That ambiguity already cost us a wrong diagnosis: a
session with 152 refused stops was reported as a positioning problem when the account had
simply spun out for the day.

The cap belongs to the *account*, not to this process or this phone: spins done on the
same account from another device, or by hand, still count against it. So the log is keyed
by account, each with its own independent rolling window. Records written before accounts
were tracked at all (no `"account"` key in the JSON) land in a reserved legacy bucket
until something calls `attribute_legacy` to say who earned them.

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

#: Bucket for records written before accounts were tracked at all. Deliberately not ""
#: (which is where an unidentified-but-tracked run's own spins live) and not a value any
#: real account name could ever collide with.
LEGACY_KEY = "\x00legacy"


def _normalize_account(account: Optional[str]) -> str:
    """`None` means "account unknown", normalized to the "" bucket.

    Anything other than `None` or a `str` is rejected rather than silently accepted: a
    wrong-typed key (e.g. a timestamp passed where an account name was expected) would
    simply never match any bucket, so `state()` would quietly report used=0 /
    exhausted=False - a confident wrong answer that looks exactly like an account in good
    standing. That is the exact failure this module exists to prevent (see module
    docstring), so fail loudly instead of guessing.
    """
    if account is not None and not isinstance(account, str):
        raise TypeError(f"account must be a str or None, got {type(account).__name__}")
    return account if account is not None else ""


@dataclass
class QuotaState:
    used: int
    limit: int
    remaining: int
    resets_in: Optional[float]     # seconds until the oldest in-window spin ages out
    exhausted: bool
    account: str = ""

    def line(self) -> str:
        prefix = f"[{self.account}] " if self.account else ""
        if self.limit <= 0:
            return f"{prefix}spins in the last 24h: {self.used} (no limit configured)"
        if self.exhausted:
            return (f"{prefix}SPIN QUOTA REACHED: {self.used}/{self.limit} in the last 24h. "
                    f"Stops will refuse to yield until {_hms(self.resets_in)} from now - "
                    f"this looks identical to 'out of range' on screen.")
        return (f"{prefix}spins in the last 24h: {self.used}/{self.limit} "
                f"({self.remaining} left" +
                (f", oldest ages out in {_hms(self.resets_in)}" if self.resets_in else "") + ")")


class SpinQuota:
    """Append-only log of spin timestamps, keyed by account and filtered to a rolling window."""

    def __init__(self, path: Optional[Path], limit: int = DEFAULT_DAILY_LIMIT,
                 window: float = WINDOW_SECONDS):
        self.path = Path(path) if path else None
        self.limit = limit
        self.window = window
        self._stamps: dict[str, list[float]] = {}
        self._load()

    # ------------------------------------------------------------------ storage

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        stamps: dict[str, list[float]] = {}
        total = 0
        try:
            with open(self.path, "r", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        rec = json.loads(line)
                        ts = float(rec["ts"])
                    except Exception:
                        continue          # a torn line from a hard kill; skip it
                    # Presence of the "account" key - not its truthiness - decides the
                    # bucket. A record with account="" was written by this code for an
                    # unidentified-but-tracked run and must stay distinct from a truly
                    # legacy record that predates accounts entirely (no key at all); an
                    # `or LEGACY_KEY` truthiness check would fold both into one bucket
                    # and let a fresh, unidentified run inherit someone else's exhausted
                    # quota on the next restart.
                    account = rec["account"] if "account" in rec else LEGACY_KEY
                    stamps.setdefault(account, []).append(ts)
        except OSError:
            return                        # unreadable is not fatal; start empty
        for acct in stamps:
            stamps[acct].sort()
        self._stamps = stamps
        self._prune()
        if total > COMPACT_AT:
            self._rewrite()

    def _prune(self, now: Optional[float] = None) -> None:
        cutoff = (now if now is not None else time.time()) - self.window
        for acct in self._stamps:
            self._stamps[acct] = [t for t in self._stamps[acct] if t >= cutoff]

    def _rewrite(self) -> None:
        """Rewrite the file with only the in-window entries for every account, atomically."""
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                for account, stamps in self._stamps.items():
                    for ts in stamps:
                        rec = {"ts": round(ts, 3)}
                        if account != LEGACY_KEY:
                            rec["account"] = account
                        fh.write(json.dumps(rec) + "\n")
            os.replace(tmp, self.path)
        except OSError:
            pass

    # ------------------------------------------------------------------ use

    def record(self, account: Optional[str] = None, now: Optional[float] = None) -> None:
        account = _normalize_account(account)
        ts = time.time() if now is None else now
        self._stamps.setdefault(account, []).append(ts)
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps({"ts": round(ts, 3), "account": account}) + "\n")
        except OSError:
            pass                          # losing the log must never stop the bot

    def seed(self, count: int, spread_hours: float = 12.0,
             account: Optional[str] = None, now: Optional[float] = None) -> None:
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
            self.record(account, now - (count - i) * step)

    def reset(self, account: Optional[str] = None) -> int:
        """Forget recorded spins for one account, or every account when `account` is
        `None`. Returns how many were dropped.

        Needed because the window can be wrong in the optimistic direction as well as the
        pessimistic one: a seeded count that turns out to be stale, or a ban that lifted
        earlier than the rolling window implies, would otherwise keep the bot from
        targeting stops it can actually use.
        """
        if account is None:
            dropped = sum(len(v) for v in self._stamps.values())
            self._stamps = {}
            if self.path is not None:
                try:
                    self.path.unlink(missing_ok=True)
                except OSError:
                    pass
            return dropped
        dropped = len(self._stamps.pop(account, []))
        if dropped and self.path is not None:
            self._rewrite()
        return dropped

    def state(self, account: Optional[str] = None, now: Optional[float] = None) -> QuotaState:
        account = _normalize_account(account)
        now = time.time() if now is None else now
        self._prune(now)
        stamps = self._stamps.get(account, [])
        used = len(stamps)
        resets = (stamps[0] + self.window - now) if stamps else None
        return QuotaState(
            used=used,
            limit=self.limit,
            remaining=max(0, self.limit - used) if self.limit > 0 else 0,
            resets_in=resets,
            exhausted=self.limit > 0 and used >= self.limit,
            account=account,
        )

    def accounts(self) -> tuple[str, ...]:
        """Named accounts with at least one recorded spin, sorted. Excludes the legacy
        bucket (not an account until `attribute_legacy` says whose it is) and the ""
        bucket (an unidentified-but-tracked run, not a real account name a caller
        iterating "known accounts" should ever see)."""
        return tuple(sorted(k for k in self._stamps if k not in (LEGACY_KEY, "")))

    def soonest_reset(self, names, now: Optional[float] = None) -> Optional[str]:
        """Of `names`, the account whose oldest in-window spin ages out first.

        An account with nothing recorded is already free and wins immediately.
        """
        now = time.time() if now is None else now
        best, best_at = None, None
        for n in names:
            st = self.state(n, now)
            if st.resets_in is None:
                return n                  # nothing recorded: already free
            if best_at is None or st.resets_in < best_at:
                best, best_at = n, st.resets_in
        return best

    @property
    def legacy_count(self) -> int:
        return len(self._stamps.get(LEGACY_KEY, ()))

    def attribute_legacy(self, account: str) -> int:
        """Assign records written before accounts were tracked. Returns how many.

        Called once at startup with whoever the UI tree reports is logged in, which is by
        definition the account that earned them - so the outcome is "attribute them to
        the account that earned them" without baking a name into an open-source repo.
        """
        legacy = self._stamps.pop(LEGACY_KEY, [])
        if not legacy:
            return 0
        self._stamps.setdefault(account, []).extend(legacy)
        self._stamps[account].sort()
        self._rewrite()
        return len(legacy)


def _hms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"
