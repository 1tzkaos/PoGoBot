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

    # Pokemon GO's own "Do you want to exit Pokemon GO?" confirm dialog (see
    # perception.exit_dialog_signal): a white card centred on a flat teal-green
    # background. Two bands sampled OUTSIDE where the card sits, both required to read
    # flat teal, are more conservative than one - a card that is narrower or taller than
    # expected still leaves at least a sliver of background in either band, but a single
    # band positioned wrong could sit entirely inside the card and read low-teal/high-
    # white on every frame regardless of what is actually on screen.
    exit_dialog_surround_top: Rect = (0.0, 0.02, 1.0, 0.20)
    exit_dialog_surround_bottom: Rect = (0.0, 0.85, 1.0, 0.98)
    #: The card itself - a generic centred-dialog placement, not a per-pixel measurement;
    #: see Thresholds.exit_dialog_card for the honesty note this shares.
    exit_dialog_card: Rect = (0.12, 0.38, 0.88, 0.62)


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

    # Pokemon GO's own exit-confirmation dialog (perception.exit_dialog_signal). Both
    # bars are the rule exactly as measured on the two real positive samples available:
    #   sample 1: teal_surround=0.814  card_white=0.815
    #   sample 2: teal_surround=0.858  card_white=0.815
    # against every labelled class in the 235-frame corpus - one false positive
    # (ExitTrainerBattle, itself a confirm dialog on a flat teal background where BACK is
    # also the correct response) and zero on everything else, including Overworld
    # (max teal 0.35) and PokemonEncounter (max card 1.00, but max teal only 0.44 - the
    # AND of both bars is what separates it, not either alone). Only TWO positive
    # samples exist; these are not a well-sampled threshold the way, say,
    # goplus_on/off_* above are - see the module docstring for the same caveat.
    exit_dialog_teal: float = 0.55
    exit_dialog_card: float = 0.35
    #: "Bright, low-saturation" for the card ROI - not itself a number from the task
    #: brief (only the teal-surround band, H 70-100/S>=60/V>=90, was measured and
    #: reported); chosen to match the qualitative description of a light dialog card
    #: independent of hue, and exercised only against synthetic fixtures, not the real
    #: corpus. See perception.exit_dialog_signal.
    exit_dialog_card_s_max: float = 60.0
    exit_dialog_card_v_min: float = 180.0

    # PGSharp shortcut-menu "AutoWalk" icon glyph colour (perception.autowalk_active_signal,
    # fsm.Switching._autowalk_menu). The user's own report: if the glyph is blue, that
    # account is ALREADY autowalking and must not be tapped again. Measured by sampling
    # every item's own icon box - x=0 to the label's left edge, y over the label's own
    # vertical bounds, both taken from the item's uiautomator node (see
    # accounts.AccountView.autowalk_icon_rect_norm) - on the ONE captured dump of an
    # account that already had AutoWalk running:
    #
    #   item          blue_frac   white_frac
    #   Map               0.141        0.261
    #   7.0 km/h          0.033        0.139
    #   AutoWalk          0.320        0.002     <- active: the glyph is BLUE, not white
    #   Feeds             0.027        0.135
    #   Favorites         0.118        0.172
    #   Teleport          0.213        0.198
    #   Settings          0.213        0.246
    #   Tap to            0.120        0.205
    #
    # (full 1080x2340 capture; the SAME fixture downscaled to the bot's own 590x1280
    # processing resolution reads white=0.001 blue=0.331 for AutoWalk - the signal
    # survives the downscale.) white_frac is the clean discriminator - 0.002 against a
    # 0.135 floor across the seven inactive siblings, a 60x gap - while blue_frac is
    # contaminated by the semi-transparent menu sitting over a blue map: Teleport alone
    # reads blue=0.213 while genuinely inactive, which is exactly why the TRUE bar below
    # requires white too, never blue alone.
    #
    # HONESTY: this is ONE sample of the active state, and there is no clean negative
    # sample of an inactive AutoWalk icon SPECIFICALLY - the seven siblings above are a
    # proxy ("some OTHER icon is white when inactive"), not AutoWalk's own icon caught
    # inactive. An older capture, taken before AutoWalk was ever started on that account,
    # reads white=0.041 blue=0.096 for the very same box - neither clearly white nor
    # clearly blue - so it CANNOT serve as a negative and must not read as confidently
    # inactive; it has to land in the UNKNOWN gap between the two bars below, not be
    # forced into either one. Because the negative side is this thin, "already active"
    # requires BOTH bars to clear, precision-first: misreading an inactive account as
    # already active silently skips a walk the user wanted, while the opposite mistake -
    # trying AutoWalk on an account that already has it running - is exactly today's
    # behaviour and is already known to be safe. The margins below sit strictly between
    # the single active sample (white 0.001-0.002, blue 0.320-0.331) and the ambiguous
    # negative (white 0.041, blue 0.096) - there are no OTHER negative samples to place a
    # margin against.
    autowalk_active_white_max: float = 0.02
    autowalk_active_blue_min: float = 0.20
    #: "Confidently inactive" (FALSE) needs only the white bar - measured the cleaner of
    #: the two above - clearing a floor well under the seven inactive siblings' own
    #: minimum (0.135), so a real white glyph is never mistaken for the UNKNOWN gap that
    #: exists only to keep the single ambiguous sample out of both bars.
    autowalk_inactive_white_min: float = 0.10


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
    #: Bounds the coordinate-free BACK presses `Switching._settle` may fire clearing a
    #: post-login screen, counting only ones the actuator actually accepted - see
    #: `Runner.apply`, the same pattern `switch_zoom_reps`/`switch_goplus_attempts`
    #: already use, and `fsm.Context.switch_clear_presses` for where the count lives.
    #:
    #: Measured on a live run: one BACK dismisses the post-login news modal, but nothing
    #: bounded the presses, and BACK kept firing every `switch_clear` into a Unity
    #: LOADING screen that legitimately runs for minutes. The actuator tally from that
    #: run: `by_budget: {'back': 100, ...}` - about 90 of those went into the loading
    #: screen over four minutes, and nothing else was logged in that window because
    #: every phase was returning `[]` waiting on a map that never came ("account switch
    #: to the target account never confirmed" at +240s, then RECOVERING halted the run at "no
    #: confirmed map for 129s"). Worse than a stall: afterward PGSharp's own account
    #: panel showed NEITHER account with an asterisk - the game had logged out of the
    #: old account and never finished logging into the new one, recovered only by hand.
    #: One or two BACKs clear a news modal; ninety is never right.
    #:
    #: Real headroom over the observed 1-2 presses, not the presses themselves. Once
    #: spent, `_settle` simply waits - `switch_timeout` (240s below) already owns the
    #: outcome, and `_verify` still runs the instant the map returns. A LOCATED close
    #: button (`obs.close_button_xy`) is targeted, not blind, and is not limited by this
    #: bound - only the coordinate-free BACK fallback is.
    switch_clear_max: int = 5
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

    #: Gap between BACK presses aimed at Pokemon GO's own exit-confirmation dialog (see
    #: perception.exit_dialog_signal, fsm.interrupts). Matches RECOVERING's own retry
    #: cadence for the same physical button; this is a repeat-dismissal pace, not a
    #: one-shot timeout - the dialog is not expected to survive even one BACK.
    exit_dialog_back: float = 1.0


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
class AutoWalk:
    """Launching AutoWalk after a confirmed account switch, through PGSharp's floating
    star widget (see `fsm.Switching._autowalk_open` and its neighbours): tap the star,
    pick AutoWalk from the shortcut menu it opens, press CONTINUE LAST if PGSharp offers
    it or OK otherwise, then tap the star again to close the menu it leaves open.

    Bounded by WALL-CLOCK time, unlike `ZoomOut.repeats`/`GoPlusToggle.max_attempts`
    above. Those both act on a FIXED coordinate that cannot fail to be found, so counting
    accepted actuations is enough to bound them. Every step here instead depends on
    LOCATING a uiautomator node - the star, the "AutoWalk" menu entry, a dialog button -
    that can legitimately never appear (a PGSharp update, an unexpected screen), in which
    case zero actuations are ever accepted and a bare attempt count never advances. A
    wall-clock budget is the only bound that still guarantees the ladder gives up rather
    than occupying the screen until `Timings.switch_timeout` itself expires and turns an
    already-successful account switch into a recorded failure.
    """

    #: Real headroom for four settle-and-reread cycles (star, menu, dialog, close) at
    #: `Timings.switch_tap` apart, without eating meaningfully into the switch's own
    #: 240s budget - see the class docstring for why this exists at all.
    budget_s: float = 30.0

    #: Extra time `fsm.Switching._autowalk_deadline` may spend, ON TOP OF `budget_s`,
    #: trying to close a shortcut menu the ladder itself opened before it gives up -
    #: never to keep hoping the ladder will still finish, only to avoid handing SCANNING
    #: a menu left open (see `_autowalk_close`'s own docstring for why that is not
    #: cosmetic: it sits over the reach ellipse SCANNING taps into, and the NEXT switch's
    #: own star tap would toggle it SHUT instead of open, silently killing AutoWalk for
    #: the rest of the run).
    #:
    #: Sized for exactly one "wait for the view, then tap" cycle: `Runner.apply` drops
    #: `ctx.accounts` after every actuation taken while SWITCHING, and only the next
    #: throttled tree refresh (`runner.ACCOUNTS_REFRESH`, 2.5s) puts a usable view back -
    #: this has to survive at least one of those, with headroom for more than one in case
    #: the first lands on `Timings.switch_tap`'s own pacing gate (2.0s) instead of the
    #: star. Confirmation still happens once this is spent too, exactly like `budget_s`
    #: itself - the switch is never held hostage to the cleanup - and the combined total
    #: (40s) stays a small fraction of the 240s `Timings.switch_timeout` that bounds the
    #: whole switch regardless.
    close_grace_s: float = 10.0


@dataclass(frozen=True)
class Config:
    rois: Rois = field(default_factory=Rois)
    thresholds: Thresholds = field(default_factory=Thresholds)
    timings: Timings = field(default_factory=Timings)
    cooldowns: Cooldowns = field(default_factory=Cooldowns)
    reach: Reach = field(default_factory=Reach)
    zoom: ZoomOut = field(default_factory=ZoomOut)
    goplus: GoPlusToggle = field(default_factory=GoPlusToggle)
    autowalk: AutoWalk = field(default_factory=AutoWalk)

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
