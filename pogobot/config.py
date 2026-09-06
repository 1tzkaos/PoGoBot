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
    #: Mint fraction of `Rois.bottom_button` needed to believe a close X is on screen.
    #: This is what `Observation.in_overlay` rides on, and so what routes a closable
    #: screen to POPUP at all - `find_close_button` locating the X is not enough on its
    #: own.
    #:
    #: Lowered from 0.035 after a live wedge on a Gym screen: the X was located at
    #: (0.500, 0.890) the whole time, but the ROI measured mint=0.0321, so `x_button` read
    #: False, `in_overlay` stayed False, and nothing ever routed it to POPUP. Measured at
    #: the resolution the bot actually runs (scrcpy scales the long side to `max_size`, so
    #: a 1080x2340 phone streams at 590x1280); the same frame reads 0.0296 at native
    #: resolution, which is why the fixture is committed at stream size.
    #:
    #: Swept over the 235-frame labelled corpus, counting frames that newly read True:
    #:   0.035  ->  baseline                                     gym screen: MISSED
    #:   0.032  ->  identical in every class                     gym screen: MISSED
    #:   0.030  ->  identical in every class                     gym screen: found  <- chosen
    #:   0.028  ->  Pokestop 1/4 -> 2/4                          gym screen: found
    #: Nothing changes anywhere down to 0.030 - notably PokemonEncounter stays 0/53, which
    #: matters because a true `x_button` VETOES `encounter_signal`, so a false one here
    #: would suppress encounters rather than merely add a stray tap. 0.028 was left on the
    #: table because it is the first value that moves any count at all.
    x_button_mint: float = 0.030
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
    #: How long a fainted party keeps the bot out of ROCKET (see `fsm.desired_state` and
    #: `Rocket.step`). A fainted member is an ACCOUNT fact the bot cannot fix - nothing it
    #: does on screen heals it - so the refusal must not release on anything the screen
    #: does. The real release is a confirmed account switch, which zeroes the stamp.
    #:
    #: Floor is `stuck_watchdog` (120.0) + `app_restart_grace` (90.0) = 210s, under which
    #: the hold can expire mid-recovery and hand the machine straight back to ROCKET. 900
    #: caps declined fights at 4/hour against the measured churn of ~120/hour - one cycle
    #: per 30.1s, paced by `Cooldowns.on_refuted`. Note `switch_every_minutes` defaults to
    #: 0.0, so rotation is opt-in and is NOT a backstop this number may lean on.
    party_fainted_hold: float = 900.0
    #: How long a run may go without an ENCOUNTER before it is declared unproductive and
    #: halted. ARMED ONLY once the session has had its first encounter, so a slow start,
    #: a preflight, or a login is never punished by it.
    #:
    #: The bar is set from measurement, not taste. Over the 303 ENCOUNTER entries in
    #: `logs/run.log` the gap between encounters runs median 17s and p90 50s, and there is
    #: exactly ONE gap above 900s: 31,091s, which is the fainted-party stall itself. There
    #: is nothing in between, so 1800 sits 36x above normal operation and an order of
    #: magnitude below the failure - it cannot fire on a working run.
    #:
    #: This is the only thing that would have ended that run honestly. `Rocket.timeout_s`
    #: fired 173 times and recovered every time; the stale-frame watchdog never fired
    #: because frames kept arriving. Both are per-symptom. This one asks the only question
    #: that matters to an operator: is the bot still catching anything?
    productivity_watchdog: float = 1800.0

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

    #: Budget for the whole PREFLIGHT state, read by `fsm.Preflight.timeout` - the startup
    #: pass that runs a confirmed switch's own zoom-out, Virtual Go Plus and AutoWalk steps
    #: once before the bot plays.
    #:
    #: Derived from the bounds those reused phases already carry, not picked as a round
    #: number: `ZoomOut.repeats` (2) gestures paced by `ui_settle` (1.2s), then at most
    #: `GoPlusToggle.max_attempts` (2) presses `GoPlusToggle.press_wait` (6.0s) apart, then
    #: an AutoWalk ladder that bounds ITSELF at `AutoWalk.budget_s + close_grace_s` (40s) -
    #: about 55s if every one of them runs long. 90s is real headroom over that, and the
    #: cost of the headroom is nil: this state has no failure outcome at all (every exit
    #: goes to SCANNING and plays), so a longer budget only ever buys a slow step the
    #: chance to finish.
    #:
    #: Deliberately kept UNDER `stuck_watchdog` (120s), which is what lets a preflight
    #: reuse those phases without also needing the watchdog credit `Context.switch_exit_ts`
    #: grants a switch: SWITCHING can legitimately hide the map for its whole 240s, while a
    #: preflight cannot occupy the screen long enough to look like a wedged run. Sized
    #: against the tree-read cost too - `accounts.UiTreeReader.read` blocks the run loop for
    #: ~3.0s against the rendering game (measured: 2.96, 3.00, 3.00, 3.00, 4.46), and
    #: `runner.ACCOUNTS_REFRESH` stamps its throttle from when a read FINISHED, so the
    #: AutoWalk ladder's views land ~5.5s apart and its four steps need most of its 30s.
    preflight_timeout: float = 90.0

    #: How long after an app restart (see effects.RestartApp) RECOVERING waits before it
    #: is willing to judge that restart a failure and spend another one.
    #:
    #: A force-stop plus relaunch is a COLD start: the Niantic splash, asset checks and a
    #: login run for tens of seconds before the map can possibly be back, and for every
    #: one of those seconds `map_stale_since` is still older than `stuck_watchdog` (120s)
    #: - the very condition that authorised the restart. Without this the escalation eats
    #: its whole budget in the couple of RECOVERING timeouts (6s each) it takes to notice
    #: the map is still missing, restarting an app that was loading perfectly well.
    #:
    #: Real headroom over "tens of seconds", not a tight bound: the cost of waiting too
    #: long is that a genuinely dead app is left alone for another minute and a half of a
    #: run that is already doing nothing, while the cost of waiting too little is spending
    #: the entire restart budget on one app that simply had not finished starting.
    app_restart_grace: float = 90.0
    #: Gap between attempts to raise the game. Measured: `am start` put Pokemon GO in front
    #: within about a second, so this is a settle window, not a wait for the answer.
    foreground_retry: float = 4.0

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
    """The map zoom-out performed after a confirmed switch and at startup (see
    `fsm.Switching._zoom`). It is a real two-finger pinch - see `effects.Pinch` and
    tools/pinch/ for why nothing else works.

    Every number here is measured on the device.

    The gesture: two fingers on a vertical line through (`center_x`, `center_y`), starting
    `start_gap` apart and closing to `end_gap`. Fingers together zooms out.

    `repeats` is 2 because that is where the bot SEES the most, which is not the same as
    the widest view. Detector yield per frame, pinching out from the game's own post-login
    zoom:

        zoomed in (baseline)   0.2 stops/gyms
        after 1 pinch          0.8
        after 2 pinches        1.2      <- chosen
        after 3 pinches        0.2
        after 4 pinches        0.2

    Past two the map is wide enough that stops and gyms fall below the size the detector
    was trained on, so zooming further makes the bot blind rather than far-sighted. The
    earlier one-finger gesture this replaced was justified by a "map-region diff" - i.e.
    that the screen CHANGED, which a pan and a zoom IN also do - and measured against
    side-by-side captures it never altered the map's scale at all.
    """

    #: (0.5, 0.513) is centre-ish, device (540, 1200) on 1080x2340 - the point the pinch
    #: closes toward. Normalized so the gesture survives any capture resolution.
    center_x: float = 0.5
    center_y: float = 1200.0 / 2340.0
    #: Finger separation at the start and end, as fractions of screen HEIGHT: device 1000px
    #: closing to 150px on a 2340px screen.
    start_gap: float = 1000.0 / 2340.0
    end_gap: float = 150.0 / 2340.0
    #: Intermediate MOVE events. Too few and the game reads a jump rather than a pinch.
    steps: int = 25
    #: Wall-clock length of the gesture.
    duration_ms: int = 700
    #: See the yield table above.
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

    A route that is ALREADY running short-circuits that middle step rather than lengthening
    it: PGSharp answers the menu entry with a "Stop/Pause AutoWalk?" dialog instead of the
    setup one, which the ladder backs out of and then closes the menu, well inside the
    budget below. It used to be indistinguishable from "the setup dialog never appeared",
    which spent the whole of `budget_s` on every single run.

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
class StarSeparation:
    """Dragging PGSharp's floating star clear of its accounts launcher (see
    `fsm.Switching._separate_star` and `accounts.AccountView.overlay_collapsed`).

    The two widgets both float, and a relaunch of the game lays them on top of each other:
    measured immediately after `effects.RestartApp`, the star's clickable rect was
    (0,152)-(108,260) and the launcher's (0,152)-(272,245), leaving the star's own centre
    (54,206) INSIDE the launcher. A tap aimed at the star lands on the accounts launcher
    there, opening the very panel the restart ladder exists to escape - so the restart
    would otherwise cause the wedge it recovers from.

    Only the two numbers that are not derivable from the dump live here. The gesture's
    endpoints are not among them: both come from rects the tree just reported (see
    `accounts.AccountView.star_clear_y_norm`), because a remembered coordinate for a
    widget whose whole problem is that it moves is how this fails silently.
    """

    #: `input swipe` duration for the drag, in ms. 400ms and 500ms were both run on the
    #: device and both moved the star; 500 is the one used here because the slower of two
    #: verified durations is the one less likely to read as a fling. Shorter values were
    #: never tested and are not assumed to behave the same - the same standard
    #: `ZoomOut.duration_ms` states.
    duration_ms: int = 500

    #: Drags one switch attempt may spend before it gives up and skips AutoWalk rather
    #: than tapping a star it cannot trust (see `fsm.Switching._separate_star`).
    #:
    #: More than one is required, not optional: the gesture does NOT land where it is
    #: aimed. Measured, asking for y=626 landed the star's centre at 837, asking for 339
    #: landed at 443, and asking for 356 moved it the other way entirely, to 125. So the
    #: implementation re-reads the tree and judges the result rather than assuming it, and
    #: this is the bound on how long it may keep judging. Three, because the third is the
    #: first that is re-testing a hypothesis two drags have already refuted - the same
    #: reasoning `GoPlusToggle.max_attempts` and `Config.max_app_restarts` state - and
    #: because each drag costs a `Timings.switch_tap` gate plus a tree refresh
    #: (`runner.ACCOUNTS_REFRESH`, 2.5s) out of `AutoWalk.budget_s` (30s), which the rest
    #: of the ladder also has to fit inside.
    max_drags: int = 3


@dataclass(frozen=True)
class TargetWeights:
    """How often each detection class should be tapped, relative to the others.

    What this replaces: `pick_target` ranked `(1 if pokemon else 0, conf)`, a strict
    tiering in which ANY Pokemon outranked ANY stop. On a map with both, stops were never
    tapped at all - not rarely, never - because a single 0.31-confidence Pokemon anywhere
    in reach beat a 0.99 stop. The only escapes were `--target-mode pokestop` and
    restocking, both of which turn Pokemon off entirely. There was no way to say "mostly
    Pokemon, some stops", which is what an operator actually wants.

    These are RATIOS, not probabilities and not thresholds. Only the proportions between
    them matter: {1.0, 0.6} and {10, 6} schedule identically. With the defaults, a map
    showing both classes settles at 5 Pokemon taps for every 3 stops.

    A weight of 0 disables the class - it is skipped exactly the way `fight_rockets=false`
    skips rockets, rather than being ranked last and tapped whenever nothing else is up.

    `pokestop_rocket` defaults to the same weight as `pokestop` because that is what the
    old tiering did: both ranked 0, so neither outranked the other. Raising it is a real
    change in what the bot plays, so it is left to the operator rather than assumed here.
    """

    pokemon: float = 1.0
    pokestop: float = 0.6
    pokestop_rocket: float = 0.6

    def of(self, name: str) -> float:
        """The weight for a detection class; 0 for one that has none.

        Unknown classes weigh nothing rather than defaulting to 1.0: the model gained a
        `gym` class once already, and a new class silently entering the rotation at full
        weight is how the bot starts tapping something nobody chose.
        """
        return float(getattr(self, name, 0.0))

    def __str__(self) -> str:
        return ", ".join(f"{f}={getattr(self, f):g}"
                         for f in ("pokemon", "pokestop", "pokestop_rocket"))


@dataclass(frozen=True)
class BattleParty:
    """Reading whether a Rocket battle party can actually fight (see
    `perception.party_can_battle` and `fsm.Rocket.step`).

    The failure this exists for, measured in `logs/run.log`: a party member had fainted,
    so Pokemon GO refused the fight behind a pink error the bot cannot read. The bot
    pressed USE THIS PARTY, learned nothing, timed out on `Rocket.timeout_s`, recovered,
    saw the invaded stop again and went straight back in. 173 cycles, a median of exactly
    150.0s each - the FULL rocket budget, every time, so not one of them made progress -
    totalling 9.3 hours with `stops_collected: 0`. It did not crash; it looked busy.

    A fainted member is legible before the press: it draws NO HP bar at all. On the live
    stall frame the three cards read [0.0000, 0.6972, 0.6944] green in the bar band. That
    is not a threshold to tune, it is a hole.

    The awkward part is WHERE to look. The HP row sits at frame-y 0.7625 on the 864x1920
    corpus and 0.7402 on the live 1080x2340 device - a 0.0223 spread that a fixed
    frame-relative band does not survive. Measured panel-relative the same rows are 0.5444
    and 0.5546, a spread of 0.0102, so the sheet is located first and everything below is
    read relative to it.
    """

    #: Near-white mask that locates the party sheet. Swept V in {200,215,225,235,245} x
    #: S in {25,40,60}: 8/8 party frames pass and 0/237 other screens do in every cell with
    #: V <= 235; V > 245 admits a Shop frame. 225 sits mid-plateau.
    panel_v_min: int = 225
    panel_s_max: int = 40

    #: Bounds on the sheet's PIXEL area (`cv2.CC_STAT_AREA`), as a fraction of the frame.
    #: Deliberately not the bounding-box area: measured pixel area is 0.2243 (live) and
    #: 0.2533-0.2582 (corpus), while the BBOX area is 0.2995 and 0.3021 - so an `area_max`
    #: of 0.30 read as bbox rejects all five healthy corpus frames and clears the live one
    #: by 0.0005. The two readings are three thousandths apart on the frame that matters,
    #: which is why the unit is named here rather than left to the reader.
    area_min: float = 0.18
    area_max: float = 0.30

    #: The sheet's top edge must sit below this fraction of frame height. The one
    #: load-bearing term in the gate: leave-one-out over the corpus gives 0 false
    #: positives when any OTHER term is dropped, and 3 when this one is.
    top_min: float = 0.45

    #: The three party cards, as fractions of FRAME width. The six dark card borders
    #: measure identically to within 0.002 of frame width across both aspect ratios.
    #: Panel-relative x was measured and rejected: it reduces to the same numbers, and a
    #: single merged white blob would drag all three windows at once.
    cards: tuple = ((0.139, 0.324), (0.406, 0.589), (0.672, 0.856))

    #: Where the HP bar sits, as a fraction of the located PANEL's height. A grid sweep
    #: gives identical separation for every band ending at or below 0.70; the first cell
    #: that breaks is hi = 0.75. The band is a SEARCH WINDOW, not an averaging window -
    #: `party_can_battle` takes the greenest row inside it, so widening this costs
    #: nothing and a band that misses the row entirely is the only real failure.
    bar_band: tuple = (0.50, 0.62)

    #: Green fraction a card's bar band must reach for that member to count as able to
    #: battle. The measured plateau is (0.0000, 0.6132]: a fainted card reads EXACTLY
    #: zero, and the shortest real bar in the corpus is 0.6132 - a member at roughly 89%
    #: health. 0.30 sits 2.04x below that and, at the stream's ~106px card width, still
    #: demands ~32px of green rather than the ~2px that 0.02 would accept.
    #:
    #: The direction of this bar matters more than its value. TRUE - "fight it", today's
    #: stall - is what comes back when every card CLEARS `bar_min`, so lowering this makes
    #: the expensive mistake more likely, not less.
    bar_min: float = 0.30

    #: HONESTY. Exactly one fainted sample exists, and it is a Rhyperior, which is natively
    #: grey - so desaturation is NOT what this measures; the absence of the bar is. Nothing
    #: measured establishes how a bar behaves as it shortens: the only depleted sample is at
    #: ~89%, so "tolerates a member at 3% health" is an assumption, not a finding. The
    #: corpus holds no raid party-select, no GBL party-select and no error-popup frame;
    #: those are the unmeasured near-neighbours.


@dataclass(frozen=True)
class Config:
    rois: Rois = field(default_factory=Rois)
    thresholds: Thresholds = field(default_factory=Thresholds)
    timings: Timings = field(default_factory=Timings)
    cooldowns: Cooldowns = field(default_factory=Cooldowns)
    reach: Reach = field(default_factory=Reach)
    zoom: ZoomOut = field(default_factory=ZoomOut)
    goplus: GoPlusToggle = field(default_factory=GoPlusToggle)
    battle_party: BattleParty = field(default_factory=BattleParty)
    autowalk: AutoWalk = field(default_factory=AutoWalk)
    star_separation: StarSeparation = field(default_factory=StarSeparation)

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
    screen_min_conf: float = 0.6
    #: A much higher bar, used only where the consequence of being wrong is pressing a
    #: button that is not ours. `Rocket.step` taps `action_pill_xy` believing it to be the
    #: fight's affirmative; on Pokemon GO's own SPONSORED interstitial that pill is
    #: "LEARN MORE", and pressing it opens the advertiser's site in a browser - measured
    #: live, `START ... act=VIEW dat=https://www.mlb.com ... cmp=com.android.chrome`,
    #: after which the bot is no longer looking at the game at all and three runs died
    #: there. The screens separate cleanly: every labelled Rocket frame in the corpus
    #: classifies at 1.00, while the ad reads Rocket@0.62 - so 0.6 lets it through and
    #: 0.90 does not, with the whole gap to spare.
    rocket_pill_min_conf: float = 0.900
    #: The bar for believing a Rocket-labelled screen that CARRIES A PILL really is a
    #: Rocket fight (`fsm.rocket_screen`). Applied only when `action_pill_xy` is located;
    #: a pill-absent frame keeps `screen_min_conf`, for the reason below.
    #:
    #: Three screens have livelocked ROCKET the same way: Pokemon GO's exit dialog, its
    #: SPONSORED interstitial, and its news card ("GO Fest 2026: ... Technical Issue").
    #: Each clears `screen_min_conf` (0.6), each carries something pill-shaped where the
    #: affirmative sits, and each costs the full 150s of `Rocket.timeout_s` per visit with
    #: nothing able to act - the press itself is already refused by
    #: `rocket_pill_min_conf`, so the machine enters, can do nothing, and times out. The
    #: news card was measured doing this for 30 minutes: 14 ROCKET entries, 0 stops, until
    #: the productivity watchdog ended the run.
    #:
    #: WHY PILL-CONDITIONAL, which is the whole design. Split by whether a pill is on
    #: screen, live Rocket-labelled frames are two different populations:
    #:
    #:     pill LOCATED   n=67768   median 0.997   >=0.90: 79.1%
    #:     pill ABSENT    n= 3726   median 0.671   >=0.90: 31.0%
    #:
    #: A flat 0.90 bar therefore refuses roughly two thirds of pill-ABSENT frames, and
    #: those are grunt dialogue advances - real fights, mid-fade, with no button to find.
    #: `ROCKET_DIALOGUE_TAP`'s own notes already recorded this and it reproduced here. The
    #: 13 labelled corpus frames all read >=0.998 and hid it completely: they are
    #: hand-picked stills, 11 of 13 are training data, and the two pill-absent ones are
    #: clean frames rather than the fades the live stream is full of.
    #:
    #: HONEST MARGINS, because the first draft of this comment got them badly wrong:
    #:   * The impostor side is a KNIFE EDGE, not a gap. Max observed confidence on a dead
    #:     pill-located frame is 0.899 against a bar of 0.900 - one thousandth, measured
    #:     over 92,714 live frames. This bar is NOT a substitute for the vetoes below it.
    #:   * The exit dialog reads Rocket @ 0.9919-0.9929 (9 frames), ABOVE the bar. It is
    #:     separated entirely by `obs.exit_dialog`, not by this number. Deleting that veto
    #:     because "the bar handles it" would restore the 430s stall it was written for.
    #:   * So this separates two of the three known impostors, not three.
    #:
    #: The corpus covers only ChooseParty, GruntBattleButton, GruntDialogue and
    #: ExitTrainerBattle. There is no labelled frame for the in-battle combat screen, the
    #: charged-attack prompt, a Rocket Leader or Giovanni, the balloon, or any post-battle
    #: screen, so the bar is unmeasured on all of them.
    rocket_route_min_conf: float = 0.900

    catch_mode: str = "throw"
    target_mode: str = "all"
    target_weights: TargetWeights = field(default_factory=TargetWeights)
    #: The PGSharp saved location to jump back to after a confirmed account switch, by
    #: NAME as it appears on the Favorites page. Empty disables it, which is the default:
    #: teleporting is a real action on a real account and nobody should get it by upgrading.
    #:
    #: Matched exact-first, then as a substring, because the real rows carry a flag emoji
    #: and a country an operator will not type - "New York" finds "<flag> New York, USA".
    #: A name that is not on the page taps NOTHING: whether PGSharp keeps favourites per
    #: account or per install is unmeasured, so an account that lacks the entry must do
    #: nothing rather than tap whatever occupies those pixels.
    home_favorite: str = ""
    #: How many recent target taps the share is measured over.
    #:
    #: This bounds the correction. Measured against LIFETIME counts instead, a
    #: class that was off screen for a long stretch comes back owed its whole
    #: absence: 100 Pokemon taps with no stop in sight leaves stops "behind" by 60,
    #: and the next 60 taps are all stops - the same starvation, pointed the other
    #: way. Over a window, the most a returning class can take in a row is its
    #: share of the window.
    target_share_window: int = 20
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

    #: Run the startup preflight - the zoom-out, Virtual Go Plus and AutoWalk steps a
    #: confirmed account switch already performs - once, when the map is first confirmed
    #: (see `fsm.Preflight` and `runner.Runner._maybe_preflight`).
    #:
    #: ON by default, unlike `switch_on_quota`/`switch_every_minutes` beside it. Those two
    #: change WHICH account a run plays; this only re-applies settings a login resets, on
    #: the account the run is already on, and every one of its steps is either a no-op or
    #: already reversible by hand. The failure it prevents is the one that was reported: a
    #: four-hour run played the whole way through zoomed in, with Virtual Go Plus off and
    #: no AutoWalk route, because the only code that set those three ran after a switch
    #: and no switch ever happened. `--no-preflight` turns it off for a run that has been
    #: set up by hand and does not want its camera or its route touched, and `cli` forces
    #: it off under `--replay`: a replay reproduces a recording, and a state the recording
    #: never had would take the screen and ignore the frames it was opened to look at.
    preflight: bool = True

    #: switch accounts when the current one exhausts its 24h spin cap
    switch_on_quota: bool = False
    #: rotate accounts every N minutes regardless of state (0 disables)
    switch_every_minutes: float = 0.0

    range_scale: float = 1.0

    #: The game's own package and main activity, used by RECOVERING's last-resort restart
    #: (see effects.RestartApp and fsm.Recovering.on_timeout). Verified on the device:
    #: PGSharp ships as a modded build of the SAME package, so one name covers both, and
    #: with the accounts panel up `mCurrentFocus` is still this activity - the panel is an
    #: overlay window of this process, which is why ending the process ends the panel.
    #: Here rather than in the effect so a different build is a config edit, not a code
    #: edit; see effects.RestartApp for the rest of that reasoning.
    app_package: str = "com.nianticlabs.pokemongo"
    app_activity: str = "com.nianticproject.holoholo.libholoholo.unity.UnityMainActivity"

    #: Consecutive app restarts RECOVERING may spend before it halts instead (see
    #: `fsm.Recovering.on_timeout`). "Consecutive" because `Runner` zeroes the count on
    #: any confirmed map: a restart that actually worked has proved itself and must not
    #: leave the rest of the run one restart poorer, while an app that crash-loops never
    #: shows a map and so can never refill the budget - which is the property that makes
    #: this a bound at all.
    #:
    #: Two, not three. Each spent restart costs `Timings.app_restart_grace` (90s) of a run
    #: that is doing nothing, and kills whatever the game had in flight. The first tests
    #: "the process is wedged"; the second covers a relaunch that itself landed badly - a
    #: login screen, a cold start that outran the grace. A third would be re-testing a
    #: hypothesis two restarts have already refuted, which is the same reasoning
    #: `runner.SWITCH_MAX_FAILURES` states for switch attempts.
    #: How many times to try raising the game before escalating to a restart. Two, because
    #: a third would mean `am start` is being accepted and ignored, which a relaunch fixes
    #: and a repeat does not.
    max_foreground_attempts: int = 2
    max_app_restarts: int = 2

    def scaled(self, **kw) -> "Config":
        return replace(self, **kw)


DEFAULT = Config()
