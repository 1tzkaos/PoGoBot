"""One HUD renderer.

v1 had two divergent copies of this code (lines 762-812 and 1143-1166) that disagreed
about when to draw the reach ellipse. It also never displayed the three booleans that
gate the entire bot, nor the scores behind them, so a misbehaving run was undiagnosable.
"""

from __future__ import annotations

from typing import Optional

import cv2

from .config import Config
from .effects import BotState
from .observation import Observation

STATE_COLOR = {
    BotState.BOOT: (200, 200, 200),
    BotState.SCANNING: (0, 255, 0),
    BotState.TARGETING: (0, 200, 255),
    BotState.ENCOUNTER: (255, 100, 255),
    BotState.POKESTOP: (255, 200, 0),
    BotState.ROCKET: (60, 60, 255),
    BotState.POPUP: (50, 100, 255),
    BotState.RECOVERING: (0, 165, 255),
    BotState.HALTED: (0, 0, 255),
}
NAME_COLOR = {
    "pokemon": (0, 255, 0),
    "pokestop": (255, 200, 0),
    "pokestop_rocket": (200, 50, 255),
    "gym": (128, 128, 255),
}


def render(frame_bgr, obs: Observation, cfg: Config, state: BotState,
           fps: float = 0.0, extra: Optional[dict] = None, status: str = ""):
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    if obs.on_map:
        cv2.ellipse(img,
                    (int(w * cfg.reach.center_x), int(h * cfg.reach.center_y)),
                    (int(w * cfg.reach.radius_x * cfg.range_scale),
                     int(h * cfg.reach.radius_y * cfg.range_scale)),
                    0, 0, 360, (255, 255, 100), 1, cv2.LINE_AA)

    for d in obs.detections:
        x1, y1, x2, y2 = d.xyxy
        c = NAME_COLOR.get(d.name, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, f"{d.name} {d.conf:.2f}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    for xy, col, tag in ((obs.close_button_xy, (0, 0, 255), "X"),
                         (obs.action_pill_xy, (0, 255, 255), "GO")):
        if xy is not None:
            px, py = int(xy[0] * w), int(xy[1] * h)
            cv2.circle(img, (px, py), 14, col, 2, cv2.LINE_AA)
            cv2.putText(img, tag, (px + 18, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    cv2.rectangle(img, (0, 0), (w, 86), (18, 18, 18), -1)
    cv2.putText(img, f"{state.value}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                STATE_COLOR.get(state, (255, 255, 255)), 2)
    screen = f"{obs.screen.label} {obs.screen.conf:.2f}" if obs.screen.available else "no-cls"
    cv2.putText(img, f"screen:{screen}  age:{obs.frame_age*1000:.0f}ms  {fps:.1f}fps",
                (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)

    # The gating signals and their raw scores - the thing v1 made impossible to see.
    flags = (f"map:{int(obs.map_ball.value)}({obs.map_ball.detail.get('red', 0):.2f}/"
             f"{obs.map_ball.detail.get('orange', 0):.2f}) "
             f"X:{int(obs.x_button.value)} enc:{int(obs.encounter.value)} "
             f"pill:{int(obs.action_pill_xy is not None)} kbd:{obs.keyboard.value[0]}")
    cv2.putText(img, flags, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 220, 170), 1)

    if status:
        cv2.rectangle(img, (0, h - 34), (w, h), (18, 18, 18), -1)
        cv2.putText(img, status, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (120, 220, 255), 1, cv2.LINE_AA)
    elif extra:
        cv2.putText(img, "  ".join(f"{k}={v}" for k, v in extra.items()),
                    (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 120), 1)
    return img
