"""Session counters and rates.

Every counter here is incremented from exactly one place - `Runner.apply`, which already
sees every Effect the state machine emits - so a count cannot drift from what the bot
actually did. `Context.stats` used to be a dict that nothing ever wrote to; this replaces
it with something that is fed by the same chokepoint that performs the actions.

A note on naming, because it matters more than the numbers:

`balls_thrown` and `catch_attempts` are NOT catches. The bot issues a throw and the
encounter later ends, but a successful catch and a Pokemon fleeing both end the same way -
the encounter UI disappears and the map returns. The 5-class screen classifier has no
class for the post-catch award screen, so a confirmed catch is not observable today.
Reporting throws as "catches" would be the same species of error as v1's unverified
training positives: a confident number with nothing behind it. See `confirmed_catches`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional


#: Rates below this much uptime are reported as unknown rather than extrapolated.
RATE_MIN_UPTIME = 120.0


@dataclass
class SessionStats:
    started: float = field(default_factory=time.perf_counter)

    # Verifiable: the bot observed the screen change, or issued the action itself.
    encounters: int = 0            # entered ENCOUNTER (a tap opened a real encounter)
    balls_thrown: int = 0          # throw gestures the actuator accepted (see dry_run)
    catch_attempts: int = 0        # encounters in which at least one ball was thrown

    stops_collected: int = 0       # POI screen confirmed open, then left cleanly
    stops_out_of_range: int = 0    # "Walk closer to interact" seen
    rockets_engaged: int = 0       # entered ROCKET
    targets_tapped: int = 0        # taps issued at a detection
    taps_expired: int = 0          # tap produced no screen change within the timeout
    recoveries: int = 0
    encounters_exhausted: int = 0   # throws did nothing; out of balls or an unwinnable catch
    restocks: int = 0               # times the bot switched to PokeStop-only to refill            # escalated to RECOVERING
    halts: int = 0

    # Not observable yet. Kept so the field exists the day a catch detector lands,
    # rather than silently conflating it with catch_attempts.
    confirmed_catches: Optional[int] = None

    # True when the actuator suppressed every gesture (--dry-run, or --replay's
    # NullActuator). `Actuator.apply` deliberately returns True in that mode so the
    # preview follows the live trajectory, which means `balls_thrown` and
    # `targets_tapped` then count gestures DECIDED, not gestures sent. Recorded here so
    # neither the report nor the lifetime history can pass them off as things the bot did.
    dry_run: bool = False

    _throws_this_encounter: int = 0
    _in_encounter: bool = False

    @property
    def encounters_finished(self) -> int:
        """Derived, so it cannot drift from `encounters`. A resumed encounter is one
        encounter that is still open, not several that each ended."""
        return self.encounters - (1 if self._in_encounter else 0)

    # ------------------------------------------------------------------ time

    def uptime(self, now: Optional[float] = None) -> float:
        return (time.perf_counter() if now is None else now) - self.started

    def per_hour(self, count: int, now: Optional[float] = None) -> Optional[float]:
        """Rate per hour, or None until the session is long enough to mean anything.

        A single event five seconds in extrapolates to 677/h, which is a fabricated
        number. Below RATE_MIN_UPTIME the honest answer is "not yet known".
        """
        up = self.uptime(now)
        if up < RATE_MIN_UPTIME:
            return None
        return count * 3600.0 / up

    # ------------------------------------------------------------------ events

    def on_encounter_start(self) -> None:
        self.encounters += 1
        self._throws_this_encounter = 0
        self._in_encounter = True

    def on_encounter_end(self) -> None:
        """Leaving the screen. The throw tracker is NOT reset here: a resumed encounter
        is the same Pokemon, and resetting it counted a fresh catch attempt each time a
        stuck screen was re-entered - 5 attempts for one Pokemon."""
        self._in_encounter = False

    def on_ball_thrown(self) -> None:
        self.balls_thrown += 1
        self._throws_this_encounter += 1
        if self._throws_this_encounter == 1:
            # Counted on the first throw rather than at the end of the encounter, so an
            # encounter still in progress is already reflected.
            self.catch_attempts += 1

    # ------------------------------------------------------------------ output

    def summary(self, now: Optional[float] = None) -> dict:
        up = self.uptime(now)

        def rate(count: int):
            r = self.per_hour(count, now)
            return None if r is None else round(r, 1)

        out = {
            "uptime_s": round(up, 1),
            "dry_run": bool(self.dry_run),
            "uptime": _hms(up),
            "encounters": self.encounters,
            "encounters_per_hour": rate(self.encounters),
            "catch_attempts": self.catch_attempts,
            "catch_attempts_per_hour": rate(self.catch_attempts),
            "balls_thrown": self.balls_thrown,
            "stops_collected": self.stops_collected,
            "stops_per_hour": rate(self.stops_collected),
            "stops_out_of_range": self.stops_out_of_range,
            "rockets_engaged": self.rockets_engaged,
            "targets_tapped": self.targets_tapped,
            "taps_expired": self.taps_expired,
            "recoveries": self.recoveries,
            "encounters_exhausted": self.encounters_exhausted,
            "restocks": self.restocks,
            "halts": self.halts,
        }
        if self.confirmed_catches is not None:
            out["confirmed_catches"] = self.confirmed_catches
        return out

    def hud_line(self, now: Optional[float] = None) -> str:
        """One compact line for the overlay."""
        def r(count: int) -> str:
            v = self.per_hour(count, now)
            return "--/h" if v is None else f"{v:.0f}/h"
        return (f"{_hms(self.uptime(now))}  "
                f"enc {self.encounters} {r(self.encounters)}  "
                f"try {self.catch_attempts} {r(self.catch_attempts)}  "
                f"stop {self.stops_collected} {r(self.stops_collected)}  "
                f"rkt {self.rockets_engaged}")

    def report(self, now: Optional[float] = None) -> str:
        """Multi-line block for the exit summary."""
        s = self.summary(now)
        def rate(key: str) -> str:
            v = s[key]
            return "" if v is None else f"{v}/h"

        rows = [
            ("uptime", s["uptime"], ""),
            ("encounters", s["encounters"], rate("encounters_per_hour")),
            ("catch attempts", s["catch_attempts"], rate("catch_attempts_per_hour")),
            ("balls thrown", s["balls_thrown"], ""),
            ("stops collected", s["stops_collected"], rate("stops_per_hour")),
            ("stops out of range", s["stops_out_of_range"], ""),
            ("rockets engaged", s["rockets_engaged"], ""),
            ("targets tapped", s["targets_tapped"], ""),
            ("taps that expired", s["taps_expired"], ""),
            ("encounters exhausted", s["encounters_exhausted"], ""),
            ("restocks", s["restocks"], ""),
            ("recoveries", s["recoveries"], ""),
            ("halts", s["halts"], ""),
        ]
        width = max(len(r[0]) for r in rows)
        lines = [f"  {name:<{width}}  {str(value):>7}  {rate}".rstrip()
                 for name, value, rate in rows]
        lines.append("")
        lines.append("  catch attempts counts encounters where a ball was thrown, not")
        lines.append("  confirmed catches - a catch and a flee are indistinguishable to")
        lines.append("  the bot today.")
        lines.append("  stops collected counts stop screens that opened and closed cleanly;")
        lines.append("  the item toast is not read, so the collection itself is inferred.")
        if self.dry_run:
            lines.append("  DRY RUN: no gesture reached the device. balls thrown and targets")
            lines.append("  tapped are decisions, not actions, and this session is excluded")
            lines.append("  from the lifetime totals.")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("_throws_this_encounter", None)
        d.pop("_in_encounter", None)
        d["encounters_finished"] = self.encounters_finished
        return d


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# ---------------------------------------------------------------- persistence

COUNTER_FIELDS = ("encounters", "catch_attempts", "balls_thrown", "stops_collected",
                  "stops_out_of_range", "rockets_engaged", "targets_tapped",
                  "taps_expired", "recoveries", "halts",
                  "encounters_exhausted", "restocks")


def append_session(path, summary: dict) -> None:
    """Append one finished session. One JSON object per line, so a crashed run costs at
    most its own line rather than corrupting the history."""
    import json
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ended": time.time(), **summary}
    # A run killed with SIGKILL can leave a partial line with no terminating newline.
    # Appending straight onto it glues this record to that fragment, and the combined
    # line is unparseable - so the torn-line skip in load_lifetime would silently drop
    # THIS session too. Measured: two appends around one torn write reported 1 session.
    # Written as one write so the O_APPEND atomicity that lets concurrent runs share the
    # file is preserved.
    lead = ""
    try:
        with open(p, "rb") as rh:
            if rh.seek(0, 2) > 0:
                rh.seek(-1, 2)
                if rh.read(1) != b"\n":
                    lead = "\n"
    except FileNotFoundError:
        pass
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(lead + json.dumps(rec) + "\n")


def load_lifetime(path) -> Optional[dict]:
    """Sum every recorded session. Returns None when there is no history yet, or when the
    history cannot be read - this only feeds an informational line, so an unreadable,
    wrong-type or half-written file must never stop the bot from running. It used to:
    a --stats-file that pointed at a directory raised IsADirectoryError out of cli.main
    after the models had loaded, and the run never started.
    """
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    try:
        # errors="replace": a hard kill can tear a line mid-multibyte-character, and a
        # decode error on one byte must not lose the whole history.
        text = p.read_text(errors="replace")
    except OSError:
        return None
    total = {k: 0 for k in COUNTER_FIELDS}
    total["sessions"] = 0
    total["uptime_s"] = 0.0
    total["dry_run_sessions"] = 0
    total["sessions_without_uptime"] = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue          # a torn line from a hard kill; skip it, do not fail
        if not isinstance(rec, dict):
            continue
        try:
            raw_uptime = rec.get("uptime_s")
            uptime = None if raw_uptime is None else float(raw_uptime)
            counts = [int(rec.get(k, 0) or 0) for k in COUNTER_FIELDS]
        except (TypeError, ValueError):
            continue          # a record with a non-numeric counter is skipped, not fatal
        if rec.get("dry_run"):
            # A dry run and a replay decide but send nothing, so their counters are not
            # things the bot did. Counted, not silently discarded.
            total["dry_run_sessions"] += 1
            continue
        total["sessions"] += 1
        if uptime is None:
            # The events are real and are kept, but the duration is not, so the
            # denominator of every rate is now unknown. Dividing the full counter total by
            # only the KNOWN time is how a lifetime rate ends up higher than the best rate
            # any single session ever achieved.
            total["sessions_without_uptime"] += 1
        else:
            total["uptime_s"] += uptime
        for k, v in zip(COUNTER_FIELDS, counts):
            total[k] += v
    if not total["sessions"]:
        return None
    hours = total["uptime_s"] / 3600.0
    total["uptime"] = _hms(total["uptime_s"])
    enough = (total["uptime_s"] >= RATE_MIN_UPTIME and hours > 0
              and not total["sessions_without_uptime"])
    for k in ("encounters", "catch_attempts", "stops_collected"):
        total[f"{k}_per_hour"] = round(total[k] / hours, 1) if enough else None
    return total


def lifetime_line(total: dict) -> str:
    def r(key: str) -> str:
        v = total.get(key)
        return "--/h" if v is None else f"{v}/h"
    line = (f"lifetime over {total['sessions']} session(s), {total['uptime']}: "
            f"encounters {total['encounters']} ({r('encounters_per_hour')}), "
            f"catch attempts {total['catch_attempts']} ({r('catch_attempts_per_hour')}), "
            f"stops {total['stops_collected']} ({r('stops_collected_per_hour')}), "
            f"rockets {total['rockets_engaged']}")
    notes = []
    if total.get("sessions_without_uptime"):
        notes.append(f"{total['sessions_without_uptime']} session(s) recorded no duration, "
                     f"so the rates are unknown")
    if total.get("dry_run_sessions"):
        notes.append(f"{total['dry_run_sessions']} dry-run session(s) excluded")
    return line + (f" [{'; '.join(notes)}]" if notes else "")
