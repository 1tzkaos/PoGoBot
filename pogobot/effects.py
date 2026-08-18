"""What a state handler is allowed to ask for.

Handlers are pure: they return a list of Effects and never touch adb, disk, or globals.
The runner is the single place that applies dry-run, rate limits, and tracing - which is
why `--no-click` cannot leak here (the v1 bot checked it at 5 of 10 actuation sites).

All coordinates are NORMALIZED (0.0-1.0). The v1 bot mixed stream pixels and device
pixels freely; cooldowns stored device pixels while detections were in stream pixels.
Normalizing at the boundary makes that entire defect class unrepresentable.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Union


class BotState(enum.Enum):
    BOOT = "BOOT"
    SCANNING = "SCANNING"
    TARGETING = "TARGETING"
    ENCOUNTER = "ENCOUNTER"
    POKESTOP = "POKESTOP"
    ROCKET = "ROCKET"
    POPUP = "POPUP"
    RECOVERING = "RECOVERING"
    HALTED = "HALTED"


class IntentOutcome(enum.Enum):
    """How a tap-intent resolved. Required at every transition so scoring cannot be
    silently skipped - v1 cleared `current_intent` at 7 sites, 3 of which scored nothing."""

    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    EXPIRED = "EXPIRED"
    CARRIED = "CARRIED"


@dataclass(frozen=True)
class Tap:
    x: float
    y: float
    reason: str
    budget: str = "tap"


@dataclass(frozen=True)
class Swipe:
    x1: float
    y1: float
    x2: float
    y2: float
    reason: str
    duration_ms: int = 200
    budget: str = "swipe"


@dataclass(frozen=True)
class Back:
    reason: str
    budget: str = "back"


@dataclass(frozen=True)
class Transition:
    to: BotState
    outcome: IntentOutcome
    reason: str


@dataclass(frozen=True)
class SetIntent:
    intent: object


@dataclass(frozen=True)
class Cooldown:
    """Block a normalized screen position for `seconds`."""

    x: float
    y: float
    seconds: float
    reason: str


@dataclass(frozen=True)
class ClearSpatialMemory:
    """Drop all cooldowns. Emitted when the camera rotates, because every remembered
    position refers to a viewport that no longer exists."""

    reason: str


@dataclass(frozen=True)
class Note:
    text: str
    level: str = "info"


@dataclass(frozen=True)
class Halt:
    reason: str


Effect = Union[Tap, Swipe, Back, Transition, SetIntent, Cooldown, ClearSpatialMemory, Note, Halt]

ACTUATIONS = (Tap, Swipe, Back)


def is_actuation(effect: Effect) -> bool:
    return isinstance(effect, ACTUATIONS)
