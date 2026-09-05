"""Build Observations without a phone, a model, or a frame."""
from pogobot.observation import Observation, ScreenGuess, Signal, Detection, Tristate


def sig(value=False, score=0.0, threshold=1.0):
    return Signal(value=value, score=score, threshold=threshold)


def obs(*, seq=1, ts=0.0, on_map=False, x_button=False, encounter=False,
        claim=False, out_of_range=False, screen="Overworld", conf=0.99,
        screen_available=True, detections=(), keyboard=Tristate.FALSE,
        close_xy=None, pill_xy=None, goplus=Tristate.UNKNOWN, exit_dialog=False,
        promo_xy=None, party=Tristate.UNKNOWN):
    return Observation(
        seq=seq, ts=ts, stream_wh=(590, 1280),
        map_ball=sig(on_map, 1.0 if on_map else 0.0),
        x_button=sig(x_button, 1.0 if x_button else 0.0),
        encounter=sig(encounter, 1.0 if encounter else 0.0),
        claim_pill=sig(claim), stop_out_of_range=sig(out_of_range),
        screen=ScreenGuess(screen, conf, available=screen_available),
        detections=tuple(detections), keyboard=keyboard,
        close_button_xy=close_xy, action_pill_xy=pill_xy, goplus=goplus,
        exit_dialog=sig(exit_dialog, 1.0 if exit_dialog else 0.0),
        promo_save_xy=promo_xy, party_can_battle=party,
    )


def det(name="pokemon", conf=0.8, cx=0.5, cy=0.63, w=0.08, h=0.05):
    x1 = int((cx - w / 2) * 590); x2 = int((cx + w / 2) * 590)
    y1 = int((cy - h / 2) * 1280); y2 = int((cy + h / 2) * 1280)
    return Detection(name=name, conf=conf, xyxy=(x1, y1, x2, y2), xywhn=(cx, cy, w, h))
