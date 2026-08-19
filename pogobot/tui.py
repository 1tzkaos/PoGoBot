"""A terminal dashboard: live stats, perception, and the log in one screen.

Line logs are fine for a post-mortem but poor for watching a run: the numbers that
matter scroll away. This renders them in place and keeps the log as one pane.

Everything shown here is read from `SessionStats` and the current `Observation`; the
dashboard computes nothing of its own, so it cannot disagree with the exit summary.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from typing import Optional

from .effects import BotState
from .observation import Observation
from .stats import SessionStats

STATE_STYLE = {
    BotState.BOOT: "grey62",
    BotState.SCANNING: "bold green",
    BotState.TARGETING: "bold yellow",
    BotState.ENCOUNTER: "bold magenta",
    BotState.POKESTOP: "bold cyan",
    BotState.ROCKET: "bold red",
    BotState.POPUP: "blue",
    BotState.RECOVERING: "bold dark_orange",
    BotState.HALTED: "bold white on red",
}
LEVEL_STYLE = {"WARNING": "yellow", "ERROR": "bold red", "CRITICAL": "bold white on red"}


class LogPane(logging.Handler):
    """Captures log records for the dashboard instead of letting them corrupt it."""

    def __init__(self, capacity: int = 400):
        super().__init__()
        self.records: deque = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append((record.levelname, self.format(record)))
        except Exception:
            pass


def available() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except Exception:
        return False


class Dashboard:
    """Owns the rich Live display. A no-op if rich is unavailable."""

    def __init__(self, stats: SessionStats, lifetime: Optional[dict] = None,
                 log_capacity: int = 400, quota=None, pause_file=None):
        from rich.console import Console
        from rich.live import Live

        self.stats = stats
        self.lifetime = lifetime
        self.quota = quota
        self.pause_file = pause_file
        self.console = Console()
        self.pane = LogPane(log_capacity)
        self.pane.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        self._live = Live(self._render(None, BotState.BOOT, 0.0),
                          console=self.console, refresh_per_second=8,
                          screen=True, transient=False)
        self._obs: Optional[Observation] = None
        self._state = BotState.BOOT
        self._fps = 0.0
        self._extra: dict = {}
        self._paused = False

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> "Dashboard":
        root = logging.getLogger("pogobot")
        self._removed = [h for h in root.handlers]
        for h in self._removed:
            root.removeHandler(h)
        root.addHandler(self.pane)
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        try:
            self._live.__exit__(*exc)
        finally:
            root = logging.getLogger("pogobot")
            root.removeHandler(self.pane)
            for h in self._removed:
                root.addHandler(h)
            # Replay the log to the real console so a finished run leaves a transcript
            # behind rather than a screen that vanishes with the alternate buffer.
            for level, line in self.pane.records:
                self.console.print(line, style=LEVEL_STYLE.get(level, ""), highlight=False)

    def update(self, obs: Optional[Observation], state: BotState, fps: float,
               extra: Optional[dict] = None, paused: bool = False) -> None:
        self._obs, self._state, self._fps = obs, state, fps
        self._paused = paused
        if extra:
            self._extra = extra
        self._live.update(self._render(obs, state, fps))

    # ------------------------------------------------------------------ rendering

    def _render(self, obs: Optional[Observation], state: BotState, fps: float):
        from rich.layout import Layout
        from rich.panel import Panel

        layout = Layout()
        layout.split_column(
            Layout(self._header(state, fps), size=3),
            Layout(name="middle", size=17),
            Layout(name="log"),
        )
        layout["middle"].split_row(
            Layout(Panel(self._stats_table(), title="session", border_style="green")),
            Layout(Panel(self._perception(obs), title="perception", border_style="cyan")),
        )
        layout["log"].update(Panel(self._log(), title="log", border_style="grey42"))
        return layout

    def _header(self, state: BotState, fps: float):
        from rich.panel import Panel
        from rich.table import Table

        t = Table.grid(expand=True)
        t.add_column(justify="left")
        t.add_column(justify="center")
        t.add_column(justify="right")
        if self.quota is not None:
            q = self.quota.state()
            style = "bold red" if q.exhausted else ("yellow" if q.remaining < 100 else "dim")
            spins = f"[{style}]spins {q.used}/{q.limit}[/]  "
        else:
            spins = ""
        life = ""
        if self.lifetime:
            life = (f"lifetime {self.lifetime['sessions']} runs  "
                    f"enc {self.lifetime['encounters']}  "
                    f"stops {self.lifetime['stops_collected']}")
        if getattr(self, "_paused", False):
            # Say how to get out of it: the pause file is a latch, so a stale one is the
            # likeliest reason a run "will not start".
            how = (f"rm {self.pause_file}" if getattr(self, "pause_file", None)
                   else "press p")
            label = f"[bold black on yellow] PAUSED [/] [dim]{how}[/]"
        else:
            label = f"[{STATE_STYLE.get(state, 'white')}]{state.value}[/]"

        t.add_row(
            label,
            f"[bold]PoGoBot[/]  {self.stats.hud_line()}",
            f"{spins}[dim]{life}[/]  [cyan]{fps:.1f} fps[/]",
        )
        return Panel(t, border_style="grey42")

    def _stats_table(self):
        from rich.table import Table

        s = self.stats.summary()
        t = Table.grid(padding=(0, 2))
        t.add_column(style="grey70")
        t.add_column(justify="right", style="bold")
        t.add_column(justify="right", style="cyan")

        def rate(key: str) -> str:
            v = s.get(key)
            return "--/h" if v is None else f"{v}/h"

        rows = [
            ("encounters", s["encounters"], rate("encounters_per_hour")),
            ("catch attempts", s["catch_attempts"], rate("catch_attempts_per_hour")),
            ("balls thrown", s["balls_thrown"], ""),
            ("stops collected", s["stops_collected"], rate("stops_per_hour")),
            ("stops out of range", s["stops_out_of_range"], ""),
            ("rockets engaged", s["rockets_engaged"], ""),
            ("targets tapped", s["targets_tapped"], ""),
            ("taps expired", s["taps_expired"], ""),
            ("recoveries", s["recoveries"], ""),
            ("halts", s["halts"], ""),
        ]
        for name, value, r in rows:
            t.add_row(name, str(value), r)
        t.add_row("", "", "")
        t.add_row("[dim]uptime[/]", f"[dim]{s['uptime']}[/]", "")
        return t

    def _perception(self, obs: Optional[Observation]):
        from rich.table import Table

        t = Table.grid(padding=(0, 2))
        t.add_column(style="grey70")
        t.add_column(justify="right")
        if obs is None:
            t.add_row("waiting for a frame", "")
            return t

        screen = f"{obs.screen.label} {obs.screen.conf:.2f}" if obs.screen.available else "no classifier"
        red = obs.map_ball.detail.get("red", 0.0)
        orange = obs.map_ball.detail.get("orange", 0.0)

        def flag(name: str, on: bool, note: str = "") -> None:
            mark = "[green]yes[/]" if on else "[grey42]no[/]"
            t.add_row(name, f"{mark} {note}".strip())

        t.add_row("screen", screen)
        flag("map (optical)", obs.map_ball.value, f"[dim]r{red:.2f} o{orange:.2f}[/]")
        flag("on map", obs.on_map)
        flag("close button", obs.close_button_xy is not None)
        flag("action pill", obs.action_pill_xy is not None)
        flag("out of range", obs.stop_out_of_range.value)
        t.add_row("keyboard", obs.keyboard.value.lower())
        t.add_row("frame age", f"{obs.frame_age * 1000:.0f} ms")
        t.add_row("", "")
        counts = Counter(d.name for d in obs.detections)
        t.add_row("[bold]detections[/]", f"[bold]{len(obs.detections)}[/]")
        for name in ("pokemon", "pokestop", "pokestop_rocket", "gym"):
            if counts.get(name):
                best = max((d.conf for d in obs.detections if d.name == name), default=0.0)
                t.add_row(f"  {name}", f"{counts[name]}  [dim]max {best:.2f}[/]")
        return t

    def _log(self):
        from rich.text import Text

        height = max(6, self.console.size.height - 24)
        out = Text()
        for level, line in list(self.pane.records)[-height:]:
            out.append(line + "\n", style=LEVEL_STYLE.get(level, ""))
        return out
