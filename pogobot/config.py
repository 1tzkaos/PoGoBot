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
    #: Core of the Virtual Go Plus pokeball toggle, top-right of the map (see
    #: config.GoPlusToggle, perception.goplus_signal). Sits entirely inside the icon's
    #: opaque centre - measured zero variance across 12 OFF frames and 8 ON frames
    #: spanning two accounts, three zoom levels and several hours.
    goplus_toggle: Rect = (0.898, 0.200, 0.932, 0.225)


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

    # Virtual Go Plus toggle (Rois.goplus_toggle). Measured in that ROI, in HSV, with
    # green = fraction matching H 40-90, S>=80, V>=80 (perception.goplus_signal):
    #   OFF: V=160.6  S=121.8  green=0.0%   (stable to 1 decimal - 12 map frames, 2
    #                                        accounts, 3 zoom levels, several hours)
    #   ON:  V=253.3  S=166.2  green=25.8%  (stable across 8 consecutive samples)
    # Both states are POSITIVELY matched (every bound below must clear for that state)
    # rather than one being "not the other" - so an unmeasured third appearance, chiefly
    # no Virtual Go Plus at all, reads UNKNOWN/ABSENT rather than being forced into ON or
    # OFF. OFF is bounded on BOTH sides of V and S, not just capped from above: a single
    # ceiling would also match a flat black or white ROI, which is not what was measured
    # and is exactly the kind of unmeasured appearance that must read ABSENT. ON only
    # needs a floor - there is no plausible "too bright/saturated/green to be ON" case,
    # and the S floor alone already rejects achromatic content (black or white). Margins
    # sit roughly halfway between the two measured points on each axis, leaving a dead
    # zone that also reads UNKNOWN.
    #
    # This signal is meaningless off the map: the same ROI read green=100% on a PokeStop
    # reward screen (green background), and 100%/68%/26% on assorted menus and loading
    # screens - the last of those is within noise of the real ON fraction. Callers MUST
    # gate on obs.on_map before trusting the result (see fsm.Switching._goplus).
    goplus_on_v_min: float = 220.0
    goplus_on_s_min: float = 150.0
    goplus_on_green_min: float = 0.15
    goplus_off_v_min: float = 125.0
    goplus_off_v_max: float = 195.0
    goplus_off_s_min: float = 90.0
    goplus_off_s_max: float = 140.0
    goplus_off_green_max: float = 0.08


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
    #: Budget for the whole SWITCHING state, read by `fsm.Switching.timeout`.
    #:
    #: Sized from the one switch that was fully observed end to end: ~2 minutes from the
    #: login tap to an asterisk the overlay would confirm. That is the middle of the
    #: state, not all of it - before the login tap the handler still has to open the
    #: overlay and (sometimes) follow the Accounts tab, and after it the panel has to be
    #: re-opened to read the asterisk, each of those taps gated by `switch_tap` and by the
    #: ~2.5s tree-refresh cadence. `switch_login_grace` (30s) is spent INSIDE this budget
    #: too. At 120s a healthy switch could therefore run out of time with the login
    #: already landed - the worst outcome available, because it is indistinguishable from
    #: the login being refused. 240s covers the observed 2 minutes plus the driving at
    #: both ends with real headroom; the cost of being generous is bounded by the failure
    #: backoff in `runner.py`, which is what stops a genuinely dead switch from repeating.
    switch_timeout: float = 240.0
    #: measured: a login tap reaches the post-login modal in ~14s, but the OUTGOING
    #: account's map can still be on screen for a second or two after the tap - obs.on_map
    #: returning is NOT proof the login has landed. Verify must not act before this many
    #: seconds have passed since the login tap, or it reads the old account's asterisk and
    #: calls a login that is merely still in flight a failure. Real headroom over the 14s,
    #: not a tight bound: waiting a bit too long costs a few seconds out of the switch
    #: budget above; waiting too little burns the whole budget on a false negative.
    switch_login_grace: float = 30.0


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
class ZoomOut:
    """The one-finger zoom-out gesture fired once a confirmed account switch is back on
    the map (see `fsm.Switching._zoom`). Values are read directly off the measurement in
    the task brief, not re-derived: a tap followed by a press-and-drag UP from
    (540, 1170) on a 1080x2340 screen, repeated twice, produced a map-region diff of 43.1
    after the first application and 17.6 after the second, against an 11.1 no-input
    baseline - i.e. the first pass does most of the work and the second lands near
    whatever ceiling the game's zoom-out actually has. A third application was never
    measured to do anything further, so `repeats` stops at the number that was tested,
    not at a round number.
    """

    #: (0.5, 0.5) is screen centre in EITHER orientation and resolution, unlike the
    #: measured device pixel (540, 1170) which is specific to the 1080x2340 screen it was
    #: read on - normalizing to the centre is what makes this gesture resolution
    #: independent, exactly like every other coordinate in the system (see effects.py).
    center_x: float = 0.5
    center_y: float = 0.5
    #: Normalized drag distance. Measured: dragging UP 370px on a 2340px-tall screen.
    #: Expressed as a fraction of screen height so it survives any capture resolution,
    #: the same reasoning `Cooldowns.radius_frac` uses for screen width.
    drag_frac: float = 370.0 / 2340.0
    #: The `input swipe` duration, in ms, used for the drag half of the gesture. 400ms is
    #: what was actually run on the device for both applications that produced the
    #: measurements above; untested durations are not assumed to behave the same.
    duration_ms: int = 400
    #: Two applications were measured; a third was not. Stopping at the tested number
    #: rather than guessing further reductions exist past what was actually observed.
    repeats: int = 2


@dataclass(frozen=True)
class GoPlusToggle:
    """Re-enabling Virtual Go Plus after a confirmed account switch (see
    `fsm.Switching._goplus`). Every number here is measured on the real device, not
    re-derived - see `Thresholds` above for the HSV signatures this acts on.
    """

    #: Normalized tap point - device (989, 496) on 1080x2340.
    tap_x: float = 0.915
    tap_y: float = 0.212
    #: Measured: pressing the toggle takes effect between t+2.2s and t+4.5s (the game
    #: shows "connecting..." meanwhile). Real headroom over the top of that range before
    #: re-checking, not a tight bound - the whole switch is already bounded by
    #: `Timings.switch_timeout` regardless of how this number is tuned.
    press_wait: float = 6.0
    #: Bounds the tap+recheck cycles a single switch may spend on this - it must never
    #: block a switch from confirming. One tap is normally enough; a second covers a
    #: press whose effect landed slowly.
    max_attempts: int = 2


@dataclass(frozen=True)
class Config:
    rois: Rois = field(default_factory=Rois)
    thresholds: Thresholds = field(default_factory=Thresholds)
    timings: Timings = field(default_factory=Timings)
    cooldowns: Cooldowns = field(default_factory=Cooldowns)
    reach: Reach = field(default_factory=Reach)
    zoom: ZoomOut = field(default_factory=ZoomOut)
    goplus: GoPlusToggle = field(default_factory=GoPlusToggle)

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
