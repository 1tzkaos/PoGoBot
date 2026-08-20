"""frame -> Observation. Pure: no adb, no disk, no globals, no GUI.

Design rules enforced here:
  * Every optical test reports a FRACTION of its ROI area. Absolute pixel counts break
    silently when the capture resolution changes.
  * Buttons are LOCATED, never assumed. The v1 bot's close-button fallback of
    (0.50w, 0.8808h) landed inside the overworld main-menu Pokeball, so "close the
    popup" opened the main menu. Every finder here returns Optional and the caller must
    handle None by doing nothing.
  * Nothing in this module decides what to do. That is fsm.py's job.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional, Sequence

import cv2
import numpy as np

from .config import Config, Rect
from .frames import Frame
from .observation import (
    Detection,
    Observation,
    ScreenGuess,
    Signal,
    Tristate,
)

# HSV bands. Ported from the v1 bot's device-tuned values, with the mint/teal overlap
# removed: v1's ring band (75-95) and cross band (80-105) overlapped on H 80-95,
# S 70-255, V 140-170, so a single flat blob satisfied the "ring AND cross" check.
RED_LO_A, RED_HI_A = np.array([0, 100, 100]), np.array([10, 255, 255])
RED_LO_B, RED_HI_B = np.array([170, 100, 100]), np.array([180, 255, 255])
MINT_LO, MINT_HI = np.array([75, 40, 175]), np.array([95, 255, 255])
TEAL_LO, TEAL_HI = np.array([80, 70, 30]), np.array([105, 255, 165])
ORANGE_LO, ORANGE_HI = np.array([10, 130, 150]), np.array([25, 255, 255])
PINK_LO, PINK_HI = np.array([155, 80, 150]), np.array([175, 255, 255])
#: Saturation is capped at 200 (not left at 255) because the level-up screen ("LEVEL
#: n", CLAIM REWARDS) shares the pill's hue with its own teal background: at S<=255
#: pill and background merge into one 590x634 contour (fixture resolution 590x1280)
#: that fails every width/height/aspect check, so the pill—though `claim_pill_signal`
#: sees it fine—is never located. Measured on the fixtures: pill saturation ~142,
#: background saturation ~252; the background covers 86.6-87.0% of the lower ROI. A
#: ceiling between them separates the two without touching hue. Swept against 470
#: labelled corpus frames (state_v3 + state_cls5): S<=220 and S<=200 both locate the
#: pill at (0.499, 0.801) with zero corpus regressions (see tests/test_perception_levelup.py);
#: S<=180 was rejected because the mask starts eating into the pill itself and the
#: located centre drifts to 0.465.
GREEN_PILL_LO, GREEN_PILL_HI = np.array([55, 60, 120]), np.array([95, 200, 255])
#: The Virtual Go Plus toggle's green centre when ON. Exactly the band specified in the
#: task brief (H 40-90, S>=80, V>=80) - not re-derived, see config.Thresholds.
GOPLUS_GREEN_LO, GOPLUS_GREEN_HI = np.array([40, 80, 80]), np.array([90, 255, 255])
#: The exit-confirmation dialog's flat teal/green surround. Exactly the band reported
#: for it (H 70-100, S>=60, V>=90) - see config.Thresholds.exit_dialog_teal for the
#: measured samples and the honesty note about how few of them there are.
EXIT_TEAL_LO, EXIT_TEAL_HI = np.array([70, 60, 90]), np.array([100, 255, 255])
#: PGSharp shortcut-menu icon glyph colours (perception.autowalk_active_signal). Exactly
#: the bands measured and reported - white = S<60 and V>170, blue = H 105-130, S>=120,
#: V>=90 - not re-derived; see config.Thresholds for the measured table and the honesty
#: note about how thin the negative sample is.
AUTOWALK_WHITE_LO, AUTOWALK_WHITE_HI = np.array([0, 0, 171]), np.array([180, 59, 255])
AUTOWALK_BLUE_LO, AUTOWALK_BLUE_HI = np.array([105, 120, 90]), np.array([130, 255, 255])
BALL_BANDS = (
    (np.array([0, 120, 100]), np.array([10, 255, 255])),     # Poke Ball red
    (np.array([170, 120, 100]), np.array([180, 255, 255])),  # Poke Ball red wrap
    (np.array([135, 40, 100]), np.array([175, 255, 255])),   # Premier / Dynamax pink
    (np.array([18, 120, 100]), np.array([32, 255, 255])),    # Ultra ball yellow
    (np.array([100, 120, 100]), np.array([130, 255, 255])),  # Great ball blue
)


def crop(bgr: np.ndarray, rect: Rect) -> np.ndarray:
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = rect
    return bgr[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def mask_frac(hsv: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of the region matching an HSV band. Resolution independent."""
    if hsv.size == 0:
        return 0.0
    m = cv2.inRange(hsv, lo, hi)
    return float(np.count_nonzero(m)) / float(m.size)


def _sig(score: float, threshold: float, **detail) -> Signal:
    return Signal(value=score >= threshold, score=score, threshold=threshold, detail=detail)


def map_ball_signal(bgr: np.ndarray, cfg: Config) -> Signal:
    """Overworld evidence: red Pokeball button AND orange binoculars together."""
    btn = crop(bgr, cfg.rois.bottom_button)
    bino = crop(bgr, cfg.rois.binoculars)
    if btn.size == 0 or bino.size == 0:
        return _sig(0.0, cfg.thresholds.map_ball_red)
    bh = cv2.cvtColor(btn, cv2.COLOR_BGR2HSV)
    red = mask_frac(bh, RED_LO_A, RED_HI_A) + mask_frac(bh, RED_LO_B, RED_HI_B)
    orange = mask_frac(cv2.cvtColor(bino, cv2.COLOR_BGR2HSV), ORANGE_LO, ORANGE_HI)
    ok = red >= cfg.thresholds.map_ball_red and orange >= cfg.thresholds.map_bino_orange
    return Signal(
        value=ok,
        score=min(red / max(cfg.thresholds.map_ball_red, 1e-9),
                  orange / max(cfg.thresholds.map_bino_orange, 1e-9)),
        threshold=1.0,
        detail={"red": red, "orange": orange},
    )


def x_button_signal(bgr: np.ndarray, cfg: Config, map_ball: bool) -> Signal:
    """A mint ring with a darker teal cross inside it: the universal close button."""
    btn = crop(bgr, cfg.rois.bottom_button)
    if btn.size == 0:
        return _sig(0.0, 1.0)
    hsv = cv2.cvtColor(btn, cv2.COLOR_BGR2HSV)
    mint = mask_frac(hsv, MINT_LO, MINT_HI)
    teal = mask_frac(hsv, TEAL_LO, TEAL_HI)
    ok = (
        mint >= cfg.thresholds.x_button_mint
        and teal >= cfg.thresholds.x_button_teal
        and not map_ball
    )
    return Signal(
        value=ok,
        score=min(mint / max(cfg.thresholds.x_button_mint, 1e-9),
                  teal / max(cfg.thresholds.x_button_teal, 1e-9)),
        threshold=1.0,
        detail={"mint": mint, "teal": teal, "vetoed_by_map": bool(map_ball)},
    )


def encounter_signal(bgr: np.ndarray, cfg: Config, x_button: bool) -> Signal:
    """A giant throwable ball low-centre plus the top-left flee icon on a dark backdrop."""
    if x_button:
        return Signal(False, 0.0, 1.0, {"vetoed_by_x": True})
    ball = crop(bgr, cfg.rois.throw_ball)
    flee = crop(bgr, cfg.rois.flee_icon)
    if ball.size == 0 or flee.size == 0:
        return _sig(0.0, 1.0)
    bh = cv2.cvtColor(ball, cv2.COLOR_BGR2HSV)
    ball_frac = sum(mask_frac(bh, lo, hi) for lo, hi in BALL_BANDS)
    fg = cv2.cvtColor(flee, cv2.COLOR_BGR2GRAY)
    flee_white = float(np.count_nonzero(fg > 230)) / float(fg.size)
    outdoor = float(np.mean(fg)) < cfg.thresholds.encounter_flee_max_mean
    ok = (
        ball_frac >= cfg.thresholds.encounter_ball
        and flee_white >= cfg.thresholds.encounter_flee_white
        and outdoor
    )
    return Signal(
        value=ok,
        score=min(ball_frac / max(cfg.thresholds.encounter_ball, 1e-9),
                  flee_white / max(cfg.thresholds.encounter_flee_white, 1e-9)),
        threshold=1.0,
        detail={"ball": ball_frac, "flee_white": flee_white, "outdoor": outdoor},
    )


def out_of_range_signal(bgr: np.ndarray, cfg: Config) -> Signal:
    """The magenta 'Walk closer to interact' banner on a PokeStop."""
    roi = crop(bgr, cfg.rois.out_of_range_banner)
    if roi.size == 0:
        return _sig(0.0, cfg.thresholds.out_of_range_pink)
    pink = mask_frac(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), PINK_LO, PINK_HI)
    return _sig(pink, cfg.thresholds.out_of_range_pink, pink=pink)


def claim_pill_signal(bgr: np.ndarray, cfg: Config) -> Signal:
    """A teal pill with white text where CLAIM REWARDS lives.

    This was the only resolution-invariant check in the v1 bot; its maths is kept.
    """
    roi = crop(bgr, cfg.rois.claim_button)
    if roi.size == 0:
        return _sig(0.0, cfg.thresholds.claim_teal)
    teal = mask_frac(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), MINT_LO, MINT_HI)
    inner = roi[int(roi.shape[0] * 0.2):int(roi.shape[0] * 0.8), :]
    ig = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    white = float(np.count_nonzero(ig > 220)) / float(ig.size)
    ok = teal >= cfg.thresholds.claim_teal and white >= cfg.thresholds.claim_white_text
    return Signal(ok, teal, cfg.thresholds.claim_teal, {"teal": teal, "white": white})


def goplus_signal(bgr: np.ndarray, cfg: Config) -> Tristate:
    """Virtual Go Plus pokeball toggle: TRUE (ON, bright with a green centre), FALSE
    (OFF, dim and desaturated), or UNKNOWN when the ROI matches neither measured
    signature - which is also the honest answer when there is no Virtual Go Plus at all.

    Both states are POSITIVELY identified (see config.Thresholds for the measured
    numbers and margins), not inferred as "not the other one", so an unmeasured third
    appearance in that ROI reads UNKNOWN rather than being forced into ON or OFF.

    Meaningless off the map - the same ROI reads a false 100% green on a PokeStop reward
    screen, and 100%/68%/26% on assorted menus and loading screens (see the docstring on
    Thresholds.goplus_on_v). Callers MUST gate on obs.on_map before trusting this; this
    function has no way to enforce that itself.
    """
    roi = crop(bgr, cfg.rois.goplus_toggle)
    if roi.size == 0:
        return Tristate.UNKNOWN
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = float(np.mean(hsv[:, :, 2]))
    s = float(np.mean(hsv[:, :, 1]))
    green = mask_frac(hsv, GOPLUS_GREEN_LO, GOPLUS_GREEN_HI)
    t = cfg.thresholds
    if v >= t.goplus_on_v_min and s >= t.goplus_on_s_min and green >= t.goplus_on_green_min:
        return Tristate.TRUE
    if (t.goplus_off_v_min <= v <= t.goplus_off_v_max
            and t.goplus_off_s_min <= s <= t.goplus_off_s_max
            and green <= t.goplus_off_green_max):
        return Tristate.FALSE
    return Tristate.UNKNOWN


def autowalk_active_signal(bgr: np.ndarray, icon_rect_norm: Optional[Rect],
                           cfg: Config) -> Tristate:
    """Colour of PGSharp's shortcut-menu "AutoWalk" icon glyph: TRUE when AutoWalk is
    ALREADY running for the current account and must not be tapped again (the user's own
    report, confirmed on the device), FALSE when the glyph is a plain white icon like
    every other menu item, UNKNOWN when neither is clear - see
    `fsm.Switching._autowalk_menu` for the decision this drives.

    `icon_rect_norm` is supplied by the caller, never looked up here: it is the AutoWalk
    entry's own icon box - x=0 to the label's own left edge, y over the label's own
    vertical bounds, both taken from the uiautomator node (see
    `accounts.AccountView.autowalk_icon_rect_norm`) - NEVER a hardcoded rectangle. This
    function supplies only the colour inside it; the uiautomator view supplies the
    bounds, the frame supplies the colour, and neither channel invents the other's job.
    None (the menu, or its AutoWalk entry, has not rendered yet) reads UNKNOWN - there is
    nothing to sample, exactly like a missing node means "do nothing" everywhere else in
    this codebase.

    Measured on the one committed capture of an account that already had AutoWalk
    running (tests/fixtures/{uiautomator,screens}/autowalk_menu_active.{xml,png}),
    sampling every item's OWN icon box the same way:

        item          blue_frac   white_frac
        Map               0.141        0.261
        7.0 km/h          0.033        0.139
        AutoWalk          0.320        0.002     <- active: the glyph is BLUE, not white
        Feeds             0.027        0.135
        Favorites         0.118        0.172
        Teleport          0.213        0.198
        Settings          0.213        0.246
        Tap to            0.120        0.205

    (full 1080x2340 capture; the SAME fixture downscaled to the bot's own 590x1280
    processing resolution reads white=0.001 blue=0.331 for AutoWalk - the signal survives
    the downscale.) white_frac is the clean discriminator here - 0.002 against a 0.135
    floor across the seven inactive siblings, a 60x gap - while blue_frac is contaminated
    by the semi-transparent menu sitting over a blue map: Teleport alone reads blue=0.213
    while genuinely inactive, which is exactly why TRUE below requires white too, not
    blue alone.

    HONESTY: this is ONE sample of the active state, and there is no clean negative
    sample of an inactive AutoWalk icon SPECIFICALLY - the seven siblings above are a
    proxy ("some OTHER icon is white when inactive"), not AutoWalk's own icon caught
    inactive. An older capture, taken before AutoWalk was ever started on that account,
    reads white=0.041 blue=0.096 for the very same box - neither clearly white nor
    clearly blue - so it cannot serve as a negative and must NOT read as confidently
    inactive (FALSE) below; it lands in the UNKNOWN gap between the two bars, which is
    exactly what `cfg.thresholds.autowalk_active_white_max`/`autowalk_inactive_white_min`
    are set to do (see config.Thresholds for the full reasoning). Because the negative
    side is this thin, TRUE requires BOTH bars to clear, precision-first: reading an
    inactive account as "already active" silently skips a walk the user wanted, while the
    opposite mistake - trying AutoWalk on an account that already has it running - is
    exactly today's behaviour and is already known to be safe.
    """
    if icon_rect_norm is None:
        return Tristate.UNKNOWN
    roi = crop(bgr, icon_rect_norm)
    if roi.size == 0:
        return Tristate.UNKNOWN
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = mask_frac(hsv, AUTOWALK_WHITE_LO, AUTOWALK_WHITE_HI)
    blue = mask_frac(hsv, AUTOWALK_BLUE_LO, AUTOWALK_BLUE_HI)
    t = cfg.thresholds
    if white <= t.autowalk_active_white_max and blue >= t.autowalk_active_blue_min:
        return Tristate.TRUE
    if white >= t.autowalk_inactive_white_min:
        return Tristate.FALSE
    return Tristate.UNKNOWN


def exit_dialog_signal(bgr: np.ndarray, cfg: Config) -> Signal:
    """Pokemon GO's own "Do you want to exit Pokemon GO?" confirm dialog.

    Unity-drawn, so it never appears in a uiautomator dump (see accounts.py) - a dump
    taken while it was up returned only PGSharp's own overlay nodes. It must be told
    apart from the map and from every other screen by pixels alone: a flat teal/green
    surround (measured H 70-100, S>=60, V>=90) with a bright, low-saturation card
    centred in the middle band. Both are required - see config.Thresholds.exit_dialog_*
    for the measured samples, the false-positive sweep against the 235-frame labelled
    corpus, and the honesty note that only TWO positive samples exist.

    The response this justifies is BACK (fsm.interrupts), which carries no coordinate at
    all - unlike the OK button this dialog actually has, which sits close enough to the
    fixed point ROCKET taps that a misclassification which instead led to a coordinate
    tap would risk quitting the game outright. That asymmetry is why a two-sample
    threshold is an acceptable trade here: the worst a false positive costs is one BACK
    press RECOVERING would very likely have sent anyway.
    """
    top = crop(bgr, cfg.rois.exit_dialog_surround_top)
    bottom = crop(bgr, cfg.rois.exit_dialog_surround_bottom)
    card = crop(bgr, cfg.rois.exit_dialog_card)
    if top.size == 0 or bottom.size == 0 or card.size == 0:
        return _sig(0.0, cfg.thresholds.exit_dialog_teal)
    teal_top = mask_frac(cv2.cvtColor(top, cv2.COLOR_BGR2HSV), EXIT_TEAL_LO, EXIT_TEAL_HI)
    teal_bottom = mask_frac(cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV), EXIT_TEAL_LO, EXIT_TEAL_HI)
    # The MINIMUM of the two bands, not their average: both are meant to be flat teal
    # background, so a card that spills into (or is mis-placed relative to) one of them
    # must be allowed to sink that band's own score rather than being smoothed away by
    # the other one still reading clean.
    teal = min(teal_top, teal_bottom)
    t = cfg.thresholds
    card_hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    card_mask = cv2.inRange(card_hsv,
                            np.array([0, 0, t.exit_dialog_card_v_min]),
                            np.array([180, t.exit_dialog_card_s_max, 255]))
    card_white = float(np.count_nonzero(card_mask)) / float(card_mask.size)
    ok = teal >= t.exit_dialog_teal and card_white >= t.exit_dialog_card
    return Signal(
        value=ok,
        score=min(teal / max(t.exit_dialog_teal, 1e-9),
                  card_white / max(t.exit_dialog_card, 1e-9)),
        threshold=1.0,
        detail={"teal": teal, "card_white": card_white},
    )


#: Minimum width, as a fraction of frame width, for a contour to be considered the close
#: button. The button is normally found as its whole mint ring: measured across the
#: labelled corpus those blobs run 9-12% of frame width, comfortably over the original
#: 0.05 floor. The "NEW LEVEL UNLOCKS" reward screen paints a cooler, paler X - ring hue
#: 96-107 (past MINT_HI's 95) and a near-white glyph (S~20, under MINT's S>=40) - so the
#: mint mask keeps only the glyph's inner core: round (aspect 1.00) and centred
#: (cx 0.500) but just 3.2% of frame width, which the 0.05 floor discarded. With no
#: button located, POPUP had nothing to tap and the bot cycled POPUP -> RECOVERING until
#: a human intervened.
#:
#: Swept over the 235-frame labelled corpus, counting frames whose located point changes:
#:   0.050  ->  baseline                                  unlocks: MISSED
#:   0.035  ->  0 lost, 0 moved, 8 newly located          unlocks: MISSED
#:   0.030  ->  0 lost, 0 moved, 11 newly located         unlocks: found
#:   0.028  ->  0 lost, 0 moved, 11 newly located         unlocks: found   <- chosen
#:   0.025  ->  0 lost, 0 moved, 16 newly located         unlocks: found
#: Nothing that already worked breaks at any level - no frame loses its button and no
#: frame's tap point moves. 0.028 was chosen over 0.030 purely for margin: the unlocks
#: core measures 3.1-3.2% depending on stream scale, and 0.028 costs nothing that 0.030
#: does not already cost, while 0.025 starts pulling in unrelated blobs.
#:
#: Of the 11 newly located frames, 8 are Overworld/encounter frames where both consumers
#: (Popup.step, Recovering.step) return on `obs.on_map` before ever reaching the tap; the
#: other 3 are Pokedex frames, where locating the close button is the desired behaviour.
#:
#: Rejected alternatives, recorded so nobody re-treads them. Scaling the MORPH_CLOSE
#: kernel with frame width does merge this ring's arcs into one blob, but costs 13 frames
#: that lose their button and 22 whose tap point moves. Widening this locator's hue
#: ceiling to 99 to admit the ring costs 8 lost and moves MainMenu's tap point on 4
#: frames. Only the width floor is free.
CLOSE_MIN_W = 0.028


def find_close_button(bgr: np.ndarray, cfg: Config) -> Optional[tuple[float, float]]:
    """Locate the mint X. Returns normalized centre, or None.

    Returning None is the whole point: v1 fell back to a fixed coordinate that opened
    the main menu, creating a self-sustaining open/close livelock.
    """
    h, w = bgr.shape[:2]
    y_off = int(h * 0.74)
    roi = bgr[y_off:, :]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mint = cv2.inRange(hsv, MINT_LO, MINT_HI)
    mint = cv2.morphologyEx(mint, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < w * CLOSE_MIN_W or cw > w * 0.22:
            continue
        if ch <= 0 or not (0.75 <= cw / ch <= 1.35):   # round
            continue
        cx = x + cw / 2.0
        if not (w * 0.30 <= cx <= w * 0.70):           # bottom-centre only
            continue
        area = cw * ch
        if best is None or area > best[0]:
            best = (area, cx / w, (y_off + y + ch / 2.0) / h)
    return None if best is None else (best[1], best[2])


#: `find_action_pill`'s two white-text gates - a candidate must clear EITHER to be kept.
#: `PILL_WHITE_WIDE` (the original, sole gate) is measured over 70% of the pill's width
#: (x 0.15-0.85) and passes a label that fills most of the pill, like "CLAIM REWARDS".
#: `PILL_WHITE_MIDDLE` is the same white measure taken only over the pill's centred
#: quarter (x 0.38-0.62), so a short CENTRED label - "OK" on the post-login "Stay Aware
#: of Your Surroundings" splash - still clears a gate even though it never fills WIDE's
#: much wider window. Measured on the shape-passing "Stay Aware" candidate (0.51w x
#: 0.063h, so shape and colour both already pass) in the committed fixture at 590x1280:
#: wide=0.0284 (missed - below 0.05), middle=0.0837 (well inside a centred label's own window).
#:
#: Threshold swept against the labelled corpus, counting frames that newly locate a pill:
#:   middle >= 0.080  ->  OK found, new false positives: none
#:   middle >= 0.070  ->  OK found, new false positives: none                 <- chosen
#:   middle >= 0.060  ->  new false positive: ExitTrainerBattle
#:   middle >= 0.040  ->  new false positives: Overworld, ExitTrainerBattle
#: 0.07 was chosen over 0.08 for margin: OK itself measures 0.0837, so 0.08 would leave
#: only 0.0037 of headroom, while 0.07 still shows zero new false positives and the first
#: one does not appear until 0.06.
#:
#: Rejected alternative, recorded so nobody re-treads it: simply lowering
#: PILL_WHITE_WIDE to 0.025 also finds OK, but costs a new Overworld false positive
#: (9 -> 10). The middle-window rule above costs none.
PILL_WHITE_WIDE = 0.05
PILL_WHITE_MIDDLE = 0.07


def find_action_pill(bgr: np.ndarray, cfg: Config) -> Optional[tuple[float, float]]:
    """Locate the wide green affirmative pill (BATTLE / USE THIS PARTY / CLAIM REWARDS /
    the post-login "Stay Aware of Your Surroundings" splash's OK).

    Pokemon GO uses one visual idiom for 'the button that advances', so one finder
    serves every screen that has one. Hardcoding coordinates per screen was rejected:
    measured across the labelled rocket screens they vary by device and aspect ratio.
    """
    h, w = bgr.shape[:2]
    y_off = int(h * 0.45)
    roi = bgr[y_off:, :]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, GREEN_PILL_LO, GREEN_PILL_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < w * 0.28 or ch < h * 0.015 or ch > h * 0.10:
            continue
        if ch <= 0 or not (2.5 <= cw / ch <= 9.0):     # pill shape
            continue
        inner = roi[y + int(ch * 0.25):y + int(ch * 0.75), x + int(cw * 0.15):x + int(cw * 0.85)]
        if inner.size == 0:
            continue
        wide = float(np.count_nonzero(cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) > 215)) / float(inner.size / 3)
        middle = 0.0
        if wide < PILL_WHITE_WIDE:
            # Only worth measuring when WIDE alone did not already pass - see
            # PILL_WHITE_MIDDLE's docstring for why a short centred label needs this
            # narrower window at all.
            mid = roi[y + int(ch * 0.25):y + int(ch * 0.75), x + int(cw * 0.38):x + int(cw * 0.62)]
            if mid.size:
                middle = float(np.count_nonzero(cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY) > 215)) / float(mid.size / 3)
        if wide < PILL_WHITE_WIDE and middle < PILL_WHITE_MIDDLE:  # must carry white label text
            continue
        area = cw * ch
        if best is None or area > best[0]:
            best = (area, (x + cw / 2.0) / w, (y_off + y + ch / 2.0) / h)
    return None if best is None else (best[1], best[2])


class ScreenStabilizer:
    """Requires a screen label to persist before it is believed.

    Measured over 321 real frames, the classifier disagrees with the high-precision
    optical map signal on 2.1% of frames and is confidently wrong (>=0.90) on 0.5%.
    The optical veto already covers the map case; this covers the rest, and means no
    single frame can move the state machine. v1 had no frame history anywhere.
    """

    def __init__(self, window: int = 5, needed: int = 3):
        self.window = window
        self.needed = needed
        self._hist: deque = deque(maxlen=window)
        self._stable: Optional[ScreenGuess] = None

    def push(self, guess: ScreenGuess) -> ScreenGuess:
        if not guess.available:
            return guess
        self._hist.append(guess)
        votes = Counter(g.label for g in self._hist)
        label, count = votes.most_common(1)[0]
        if count >= min(self.needed, len(self._hist)):
            confs = [g.conf for g in self._hist if g.label == label]
            self._stable = ScreenGuess(label, sum(confs) / len(confs), available=True)
        return self._stable or guess


class Perceptor:
    """Holds the models and turns a Frame into an Observation."""

    def __init__(self, cfg: Config, det_model=None, cls_model=None,
                 device: str = "cpu", square_cls_input: bool = True):
        self.cfg = cfg
        self.det = det_model
        self.cls = cls_model
        self.device = device
        self.square_cls_input = square_cls_input
        self.class_names: Sequence[str] = tuple(det_model.names.values()) if det_model else ()
        self.stabilizer = ScreenStabilizer(cfg.smooth_window, cfg.smooth_needed)

    def classify_screen(self, bgr: np.ndarray) -> ScreenGuess:
        """Classify the screen. Squares the frame first when the model was trained that way.

        Ultralytics classification inference is Resize(224) -> CenterCrop(224), which on a
        19.5:9 phone frame keeps only the middle 46% of height - discarding the bottom
        button row and the top flee icon, the two most discriminative regions. Feeding an
        already-square frame makes the crop a no-op so the model sees the whole screen.
        """
        if self.cls is None:
            return ScreenGuess("unknown", 0.0, available=False)
        src = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA) if self.square_cls_input else bgr
        res = self.cls.predict(source=src, imgsz=224, device=self.device, verbose=False)
        if not res or res[0].probs is None:
            return ScreenGuess("unknown", 0.0, available=False)
        p = res[0].probs
        return ScreenGuess(res[0].names[int(p.top1)], float(p.top1conf), available=True)

    def detect(self, bgr: np.ndarray) -> tuple[Detection, ...]:
        if self.det is None:
            return ()
        res = self.det.predict(source=bgr, conf=self.cfg.confidence, imgsz=self.cfg.imgsz,
                               device=self.device, verbose=False)
        out: list[Detection] = []
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cid = int(b.cls[0].item())
                out.append(Detection(
                    name=str(r.names.get(cid, cid)).lower(),
                    conf=float(b.conf[0].item()),
                    xyxy=tuple(int(v) for v in b.xyxy[0].tolist()),
                    xywhn=tuple(float(v) for v in b.xywhn[0].tolist()),
                ))
        return tuple(out)

    def observe(self, frame: Frame, keyboard: Tristate = Tristate.UNKNOWN,
                run_detector: bool = True) -> Observation:
        bgr = frame.bgr
        cfg = self.cfg
        map_ball = map_ball_signal(bgr, cfg)
        x_btn = x_button_signal(bgr, cfg, map_ball.value)
        enc = encounter_signal(bgr, cfg, x_btn.value)
        return Observation(
            seq=frame.seq,
            ts=frame.ts,
            stream_wh=frame.wh,
            map_ball=map_ball,
            x_button=x_btn,
            encounter=enc,
            claim_pill=claim_pill_signal(bgr, cfg),
            stop_out_of_range=out_of_range_signal(bgr, cfg),
            screen=self.stabilizer.push(self.classify_screen(bgr)),
            detections=self.detect(bgr) if run_detector else (),
            keyboard=keyboard,
            close_button_xy=find_close_button(bgr, cfg),
            action_pill_xy=find_action_pill(bgr, cfg),
            frame_age=frame.age(),
            goplus=goplus_signal(bgr, cfg),
            exit_dialog=exit_dialog_signal(bgr, cfg),
        )
