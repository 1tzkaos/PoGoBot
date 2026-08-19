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
    stop_dwell: float = 1.6
    rotate_camera: float = 3.0
    keyboard_check: float = 2.0
    claim_reward: float = 1.5
    rocket_tap: float = 1.6
    # A Rocket battle is a run of screens that look like encounters - a Pokemon, no map,
    # no X button - so the classifier alternates between Rocket and PokemonEncounter and
    # the machine used to follow every flip. Observed live: 6 ROCKET<->ENCOUNTER round
    # trips in 70 seconds, which also inflated both counters. ROCKET now holds until no
    # Rocket screen has been seen for this long.
    rocket_hold: float = 5.0
    # After deliberately leaving an encounter, ignore encounter-looking screens for this
    # long. Without it, fleeing and re-entering the same screen is a livelock: observed
    # live as ENCOUNTER -> RECOVERING -> ENCOUNTER repeating until the watchdog halted.
    encounter_hold: float = 6.0

    targeting_timeout: float = 4.0
    encounter_timeout: float = 25.0
    pokestop_timeout: float = 8.0
    popup_timeout: float = 4.0
    rocket_timeout: float = 150.0
    recovering_timeout: float = 6.0
    scanning_idle_rotate: float = 6.0

    frame_max_age: float = 2.0
    stuck_watchdog: float = 120.0

    #: gap between taps while driving the PGSharp overlay
    switch_tap: float = 2.0
    #: gap between attempts to clear a post-login screen
    switch_clear: float = 2.5
    #: measured: login tap -> post-login modal was ~14s. The budget is generous because
    #: the alternative to waiting is tapping blindly at an unknown screen.
    switch_timeout: float = 120.0


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

    # Scale applied to the reach ellipse for stop targets only.
    #
    # This was 0.55, narrowed because 15 of 16 stop taps came back "Walk closer to
    # interact". That reasoning was wrong: the account had exceeded the rolling 24h spin
    # cap, and a capped stop refuses with the SAME banner as one that is out of reach. The
    # narrowing treated a quota problem as a distance problem and cost real stops - a
    # 2m26s run at 0.55 found only two candidates on a dense map.
    #
    # The cap is now tracked explicitly (see quota.py), so distance can be judged on its
    # own evidence. Back to the ellipse that works for everything else until measurement
    # says otherwise.
    stop_scale: float = 1.0


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

    # Out of Poke Balls is not observable: with one labelled example and no clean
    # positive set, an optical ball detector would be a guessed threshold. It IS
    # observable behaviourally - throwing repeatedly with no resolution means the throws
    # are doing nothing. That covers running out of balls and an unwinnable Pokemon alike.
    max_throws_per_encounter: int = 5
    #: consecutive throw-exhausted encounters before switching to restocking
    restock_after_failures: int = 2
    #: stops to collect before returning to normal targeting
    restock_target_stops: int = 5
    #: give up restocking after this long even if no stop was reachable
    restock_max_seconds: float = 600.0
    auto_rotate: bool = True
    dry_run: bool = False

    #: switch accounts when the current one exhausts its 24h spin cap
    switch_on_quota: bool = False
    #: rotate accounts every N minutes regardless of state (0 disables)
    switch_every_minutes: float = 0.0

    range_scale: float = 1.0

    def scaled(self, **kw) -> "Config":
        return replace(self, **kw)


DEFAULT = Config()
