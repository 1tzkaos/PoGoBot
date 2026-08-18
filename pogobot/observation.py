"""What the bot believes about the current frame.

Two rules drive this module:

1. Every optical test reports a *fraction of its ROI area*, never an absolute pixel
   count. The old code used counts on ROIs that scale with `--max-size`; measured on a
   real device frame, `orange_bino` was 5406 px at 1080x2340 but 479 px at
   `--max-size 720`, silently failing its `> 500` test and disabling overworld
   detection entirely. The same signal as a fraction is 0.256 vs 0.235 - stable.
2. Raw scores ride along on the Observation. The old helpers computed six numbers that
   decided everything and threw them away, so nothing could be tuned or diagnosed.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


class Tristate(enum.Enum):
    """A boolean that can also be 'we could not find out'."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

    def is_true(self) -> bool:
        return self is Tristate.TRUE


@dataclass(frozen=True)
class Signal:
    """One optical test: its verdict, the fraction it measured, and the bar it had to clear."""

    value: bool
    score: float
    threshold: float
    detail: dict = field(default_factory=dict)

    @property
    def margin(self) -> float:
        """How far over (positive) or under (negative) the threshold, relative to it."""
        if self.threshold <= 0:
            return 0.0
        return (self.score - self.threshold) / self.threshold

    def __bool__(self) -> bool:
        return self.value


@dataclass(frozen=True)
class ScreenGuess:
    """The classifier's opinion. `available` is False when no classifier is loaded.

    The old code initialised `screen_class = "Overworld"` with `screen_conf = 1.0`, so a
    missing classifier read as a confident assertion that the bot was on the map.
    """

    label: str
    conf: float
    available: bool = True

    def is_(self, *labels: str, min_conf: float = 0.0) -> bool:
        return self.available and self.label in labels and self.conf >= min_conf


@dataclass(frozen=True)
class Detection:
    """One detector box, in stream pixels plus normalized form for training labels."""

    name: str
    conf: float
    xyxy: tuple[int, int, int, int]
    xywhn: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[int, int]:
        """Center in stream pixels - for drawing only."""
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def center_norm(self) -> tuple[float, float]:
        """Center in normalized coords - the only form allowed to reach an actuator."""
        return self.xywhn[0], self.xywhn[1]


@dataclass(frozen=True)
class Observation:
    """Everything perceived from one frame. Pure data - no device, no model handles."""

    seq: int
    ts: float
    stream_wh: tuple[int, int]

    x_button: Signal
    map_ball: Signal
    encounter: Signal
    claim_pill: Signal
    stop_out_of_range: Signal

    screen: ScreenGuess
    detections: tuple[Detection, ...] = ()
    keyboard: Tristate = Tristate.UNKNOWN
    close_button_xy: Optional[tuple[int, int]] = None
    frame_age: float = 0.0

    @property
    def on_map(self) -> bool:
        """Optical map evidence. Deliberately does not consult the classifier.

        The classifier is the least reliable input in the system; the overworld Pokeball
        plus the orange binoculars is direct evidence and must be able to veto it.
        """
        return self.map_ball.value and not self.encounter.value

    @property
    def in_overlay(self) -> bool:
        """A closable overlay is up: an X button is optically present and we are not on the map."""
        return self.x_button.value and not self.map_ball.value and not self.encounter.value


class Smoother:
    """N-of-M temporal vote over a signal, so one bad frame cannot move the machine.

    The old code had no frame history at all: a single misclassified frame could push the
    bot into ENCOUNTER and start swiping at the map.
    """

    def __init__(self, window: int = 5, needed: int = 3):
        if needed > window:
            raise ValueError("needed cannot exceed window")
        self.window = window
        self.needed = needed
        self._buf: Deque[bool] = deque(maxlen=window)

    def push(self, value: bool) -> bool:
        self._buf.append(bool(value))
        return self.value

    @property
    def value(self) -> bool:
        return sum(self._buf) >= self.needed

    @property
    def filled(self) -> bool:
        return len(self._buf) == self.window

    def reset(self) -> None:
        self._buf.clear()
