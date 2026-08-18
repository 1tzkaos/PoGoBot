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
GREEN_PILL_LO, GREEN_PILL_HI = np.array([55, 60, 120]), np.array([95, 255, 255])
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
        if cw < w * 0.05 or cw > w * 0.22:
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


def find_action_pill(bgr: np.ndarray, cfg: Config) -> Optional[tuple[float, float]]:
    """Locate the wide green affirmative pill (BATTLE / USE THIS PARTY / CLAIM REWARDS).

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
        white = float(np.count_nonzero(cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) > 215)) / float(inner.size / 3)
        if white < 0.05:                                # must carry white label text
            continue
        area = cw * ch
        if best is None or area > best[0]:
            best = (area, (x + cw / 2.0) / w, (y_off + y + ch / 2.0) / h)
    return None if best is None else (best[1], best[2])


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
            screen=self.classify_screen(bgr),
            detections=self.detect(bgr) if run_detector else (),
            keyboard=keyboard,
            close_button_xy=find_close_button(bgr, cfg),
            frame_age=frame.age(),
        )
