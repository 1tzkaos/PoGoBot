"""Every threshold and timing constant in the system, in one frozen place.

Thresholds are fractions of their ROI area (see observation.py for why). ROIs are
normalized rectangles (x0, y0, x1, y1) so they survive any capture resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class Rois:
    """Normalized regions of interest. Ported from the tuned values in the v1 bot."""

    bottom_button: Rect = (0.44, 0.85, 0.56, 0.91)
    binoculars: Rect = (0.78, 0.78, 0.94, 0.88)
    throw_ball: Rect = (0.35, 0.76, 0.65, 0.86)
    flee_icon: Rect = (0.05, 0.06, 0.16, 0.12)
    out_of_range_banner: Rect = (0.15, 0.79, 0.85, 0.85)
    claim_button: Rect = (0.25, 0.77, 0.75, 0.83)


@dataclass(frozen=True)
class Thresholds:
    """Fraction-of-ROI bars. Calibrated in tools/calibrate.py against labelled screens."""

    map_ball_red: float = 0.10
    map_bino_orange: float = 0.03
    x_button_mint: float = 0.035
    x_button_teal: float = 0.015
    encounter_ball: float = 0.020
    encounter_flee_white: float = 0.010
    encounter_flee_max_mean: float = 210.0
    out_of_range_pink: float = 0.004
    claim_teal: float = 0.45
    claim_white_text: float = 0.04


@dataclass(frozen=True)
class Timings:
    """Seconds. Every actuator has its own budget - the v1 bot shared one variable
    across eight unrelated actions, so a keyboard dismissal suppressed a disc spin."""

    ui_settle: float = 1.2
    tap_target: float = 1.2
    close_menu: float = 1.0
    throw_ball: float = 3.8
    spin_disc: float = 0.9
    rotate_camera: float = 3.0
    keyboard_check: float = 2.0
    claim_reward: float = 1.5
    rocket_tap: float = 1.6

    targeting_timeout: float = 4.0
    encounter_timeout: float = 25.0
    pokestop_timeout: float = 8.0
    popup_timeout: float = 4.0
    rocket_timeout: float = 150.0
    recovering_timeout: float = 6.0
    scanning_idle_rotate: float = 6.0

    frame_max_age: float = 2.0
    stuck_watchdog: float = 120.0


@dataclass(frozen=True)
class Cooldowns:
    """How long a screen position stays off-limits after an outcome.

    Expressed as a fraction of screen width so it is resolution independent; the v1 bot
    mixed an 80px box and a 140px radius in device pixels with no rationale.
    """

    radius_frac: float = 0.13
    on_success: float = 45.0
    on_refuted: float = 30.0
    on_expired: float = 10.0
    out_of_range: float = 90.0


@dataclass(frozen=True)
class Reach:
    """The interactive ring around the avatar. This model worked well in v1; kept."""

    center_x: float = 0.50
    center_y: float = 0.63
    radius_x: float = 0.38
    radius_y: float = 0.16
    tolerance: float = 1.05


@dataclass(frozen=True)
class Config:
    rois: Rois = field(default_factory=Rois)
    thresholds: Thresholds = field(default_factory=Thresholds)
    timings: Timings = field(default_factory=Timings)
    cooldowns: Cooldowns = field(default_factory=Cooldowns)
    reach: Reach = field(default_factory=Reach)

    det_model: Path = BASE_DIR / "models" / "v3" / "det" / "weights" / "best.pt"
    cls_model: Path = BASE_DIR / "models" / "v3" / "cls" / "weights" / "best.pt"

    # The detector runs at the FLOOR so the learning path can see marginal objects and
    # refuse to write a frame that contains them (they would become background labels).
    # The FSM only acts on detections at or above target_confidence.
    confidence: float = 0.15
    target_confidence: float = 0.30
    imgsz: int = 1024
    infer_fps: float = 8.0
    max_size: int = 1280
    max_fps: int = 30
    device: str = "auto"

    smooth_window: int = 5
    smooth_needed: int = 3
    screen_min_conf: float = 0.60

    catch_mode: str = "throw"
    target_mode: str = "all"
    fight_rockets: bool = True
    auto_rotate: bool = True
    dry_run: bool = False

    range_scale: float = 1.0

    def scaled(self, **kw) -> "Config":
        return replace(self, **kw)


DEFAULT = Config()
