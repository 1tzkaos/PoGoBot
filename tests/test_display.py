"""The preview window must never show an un-annotated frame.

Regression: the throttled branch called `_show(..., obs=None)`, which rendered the bare
frame. That path runs on every loop iteration (capped only by `cv2.waitKey(1)`, so ~1 kHz)
while the HUD was rendered once per inference at 8 Hz, so the overlay was visible for
roughly 8 frames in 1000 and the window appeared to strobe.
"""
import time

import numpy as np
import pytest

# Import hud FIRST so it binds the real cv2 at module level. The stub below is inserted
# into sys.modules only to intercept the runner's own window calls; hud.render must keep
# drawing for real or the tests would not be checking anything.
from pogobot import hud as _hud  # noqa: F401
from pogobot import runner as runner_mod
from pogobot.config import DEFAULT
from pogobot.effects import BotState
from pogobot.frames import Frame
from tests.factories import obs as make_obs


class _Src:
    def read(self):
        return None

    def healthy(self):
        return True

    def release(self):
        pass


class _Act:
    def apply(self, effect, now=None):
        return False

    def healthy(self):
        return True

    def stats(self):
        return {"sent": 0}

    def close(self):
        pass


def _runner(monkeypatch):
    shown = []

    class _CV:
        WINDOW_NORMAL = 0

        @staticmethod
        def namedWindow(*a, **k):
            pass

        @staticmethod
        def resizeWindow(*a, **k):
            pass

        @staticmethod
        def imshow(_w, img):
            shown.append(img)

        @staticmethod
        def waitKey(_n):
            return 0

        @staticmethod
        def destroyAllWindows():
            pass

    monkeypatch.setitem(__import__("sys").modules, "cv2", _CV)
    r = runner_mod.Runner(DEFAULT, _Src(), _Act(), perceptor=None, display=True)
    return r, shown


def test_blit_shows_nothing_before_the_first_hud_render(monkeypatch):
    r, shown = _runner(monkeypatch)
    assert r._blit("w") is True
    assert shown == [], "there is no HUD to show yet; must not fall back to a raw frame"


def test_show_renders_a_hud_and_caches_it(monkeypatch):
    r, shown = _runner(monkeypatch)
    frame = Frame(seq=1, ts=time.perf_counter(), bgr=np.zeros((1280, 590, 3), np.uint8))
    r.ctx.now = time.perf_counter()
    r._show("w", frame, make_obs(on_map=True))
    assert r._last_hud is not None
    assert len(shown) == 1


def test_redisplay_reuses_the_cached_hud_and_never_a_raw_frame(monkeypatch):
    r, shown = _runner(monkeypatch)
    frame = Frame(seq=1, ts=time.perf_counter(), bgr=np.zeros((1280, 590, 3), np.uint8))
    r.ctx.now = time.perf_counter()
    r._show("w", frame, make_obs(on_map=True))
    hud_img = shown[0]
    for _ in range(50):
        r._blit("w")
    assert len(shown) == 51
    assert all(img is hud_img for img in shown), "every displayed frame must be the HUD"


def test_hud_is_visually_distinct_from_the_raw_frame(monkeypatch):
    """Guards against a future 'optimisation' that caches the raw frame by mistake."""
    r, shown = _runner(monkeypatch)
    raw = np.zeros((1280, 590, 3), np.uint8)
    r.ctx.now = time.perf_counter()
    r._show("w", Frame(seq=1, ts=time.perf_counter(), bgr=raw), make_obs(on_map=True))
    assert not np.array_equal(shown[0], raw), "the displayed image must carry the overlay"


def test_replay_frames_are_not_consumed_by_the_preview(monkeypatch):
    """A replay directory advances on every read, so the preview must not pull from it.
    scrcpy keeps only the newest frame, so repainting from it is free."""
    from pogobot.capture import ReplaySource, ScrcpySource
    assert ReplaySource.sequential is True
    assert ScrcpySource.sequential is False


def test_render_never_caches_a_bare_frame(monkeypatch):
    r, shown = _runner(monkeypatch)
    raw = np.zeros((1280, 590, 3), np.uint8)
    r._render(Frame(seq=1, ts=0.0, bgr=raw), None)
    assert r._last_hud is None, "a None observation must not overwrite the cached HUD"
