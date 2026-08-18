"""Pausing must stop input without corrupting the clock.

The trap is resuming: if every deadline aged while the bot was idle, the first tick back
fires every timeout at once. A pause that ends in a recovery storm is worse than no pause.
"""
import time

import pytest

from pogobot import runner as runner_mod
from pogobot.config import DEFAULT
from pogobot.effects import BotState
from pogobot.stats import SessionStats


class _Act:
    def __init__(self):
        self.applied = []

    def apply(self, effect, now=None):
        self.applied.append(effect)
        return True

    def healthy(self):
        return True

    def stats(self):
        return {"sent": len(self.applied)}

    def close(self):
        pass


class _Src:
    def read(self):
        return None

    def healthy(self):
        return True

    def release(self):
        pass


def _runner(**kw):
    return runner_mod.Runner(DEFAULT, _Src(), _Act(), perceptor=None, display=False, **kw)


# ------------------------------------------------------------------ triggers

def test_starts_unpaused():
    assert not _runner().paused


def test_the_toggle_flips_it():
    r = _runner()
    r.toggle_pause()
    assert r._sync_pause() and r.paused
    r.toggle_pause()
    assert not r._sync_pause() and not r.paused


def test_the_pause_file_pauses_and_removing_it_resumes(tmp_path):
    f = tmp_path / "PAUSE"
    r = _runner(pause_file=f)
    assert not r._sync_pause()
    f.touch()
    assert r._sync_pause(), "the file must pause it"
    f.unlink()
    assert not r._sync_pause(), "removing it must resume"


def test_an_unreadable_pause_path_does_not_pause(tmp_path):
    r = _runner(pause_file=tmp_path / "nope" / "deep" / "PAUSE")
    assert not r._sync_pause()


def test_the_file_and_the_toggle_are_independent(tmp_path):
    f = tmp_path / "PAUSE"
    r = _runner(pause_file=f)
    r.toggle_pause()
    assert r._sync_pause()
    f.touch()
    r.toggle_pause()                 # toggle off, but the file still says pause
    assert r._sync_pause(), "the file alone is enough to stay paused"


# ------------------------------------------------------------------ the clock

def test_paused_time_does_not_age_the_state_machine():
    """The whole point: a deadline must not expire while the bot is idle."""
    r = _runner()
    r.toggle_pause()
    r._sync_pause()
    r._paused_at = time.perf_counter() - 300.0        # five minutes paused
    r.toggle_pause()
    r._sync_pause()
    assert r._pause_total == pytest.approx(300.0, abs=2)
    # ctx.now is driven from perf_counter minus paused time, so a deadline stamped before
    # the pause is still the same distance away afterwards.
    before = time.perf_counter() - r.stats.paused_seconds
    assert time.perf_counter() - before == pytest.approx(300.0, abs=2)


def test_paused_time_is_excluded_from_the_rates():
    s = SessionStats(started=0.0, paused_seconds=1800.0)
    s.encounters = 30
    assert s.uptime(now=3600.0) == pytest.approx(1800.0)
    assert s.per_hour(s.encounters, now=3600.0) == pytest.approx(60.0), \
        "an overnight pause must not dilute the rate"


def test_the_summary_reports_paused_time():
    s = SessionStats(started=0.0, paused_seconds=120.0)
    assert s.summary(now=600.0)["paused_s"] == 120.0
    assert "paused" in s.report(now=600.0)


def test_paused_seconds_grows_while_still_paused():
    r = _runner()
    r.toggle_pause()
    r._sync_pause()
    r._paused_at = time.perf_counter() - 10.0
    r._sync_pause()
    assert r.stats.paused_seconds >= 10.0, "a long pause must be visible before it ends"


# ------------------------------------------------------------------ rendering

def test_the_hud_says_paused():
    import numpy as np
    from pogobot import hud
    from tests.factories import obs as mkobs
    img = hud.render(np.zeros((1280, 590, 3), np.uint8), mkobs(on_map=True), DEFAULT,
                     BotState.SCANNING, paused=True)
    plain = hud.render(np.zeros((1280, 590, 3), np.uint8), mkobs(on_map=True), DEFAULT,
                       BotState.SCANNING, paused=False)
    assert not np.array_equal(img, plain), "the paused banner must be visible"


# ------------------------------------------------------------------ the loop

"""Everything above tests `_sync_pause` in isolation, which is how the loop's own
behaviour went unchecked. These drive `run()`.

The clock freeze is correct for FSM deadlines and wrong for everything else. Pacing the
loop from the frozen clock deadlocked it: `next_infer` was stamped from a `now` that no
longer advances, so `now < next_infer` stayed true forever. Measured: 1 perception call
in 3 paused seconds instead of 24, and - because the repaint is paced the same way - one
final `waitKey` and then none, so the p key could pause the preview but never resume it
and q could not quit.
"""

import threading
from contextlib import contextmanager

import numpy as np

from pogobot import fsm
from pogobot.frames import Frame
from tests.factories import obs as mkobs, det


class _LoopSrc:
    sequential = False

    def __init__(self):
        self.n = 0

    def read(self):
        self.n += 1
        return Frame(seq=self.n, ts=time.perf_counter(),
                     bgr=np.zeros((64, 32, 3), np.uint8))

    def healthy(self):
        return True

    def release(self):
        pass


class _Perceptor:
    def __init__(self, **kw):
        self.calls = 0
        self.kw = kw

    def observe(self, frame, keyboard=None):
        self.calls += 1
        return mkobs(seq=frame.seq, ts=frame.ts, **self.kw)


class _Ledger:
    def __init__(self):
        self.staged = 0

    def stage(self, frame, obs):
        self.staged += 1

    def resolve(self, intent, outcome, now):
        pass

    def stats(self):
        return {"written": 0}

    def close(self):
        pass


def _loop_runner(perceptor=None, **kw):
    cfg = DEFAULT.scaled(infer_fps=40.0)
    act = _Act()
    r = runner_mod.Runner(cfg, _LoopSrc(), act,
                          perceptor or _Perceptor(on_map=True, detections=(det(),)),
                          display=False, **kw)
    return r, act


@contextmanager
def _running(r):
    """Drive run() on a daemon thread and always ask it to stop.

    Daemon + finally on purpose: a failing assertion in the body must not leave the loop
    spinning and wedge the whole suite at interpreter exit, which is exactly what the
    unfixed loop did - it can no longer be stopped through the preview, so the test that
    proves that has to be able to stop it another way.
    """
    t = threading.Thread(target=r.run, daemon=True)
    t.start()
    try:
        yield t
    finally:
        r._stop = True
        t.join(5)
    assert not t.is_alive(), "run() did not return"


def _run_paused(r, seconds: float):
    """Run the loop paused from the first tick, so anything observed afterwards happened
    while paused and nothing has to be netted off a running prologue."""
    r.toggle_pause()
    with _running(r):
        time.sleep(seconds)


def test_a_paused_loop_keeps_perceiving():
    """The README promises you can watch what it sees while paused. It could not: the
    frozen clock never reached its own next_infer, so perception ran once and stopped."""
    p = _Perceptor(on_map=True)
    r, _ = _loop_runner(p)
    _run_paused(r, 0.3)
    assert p.calls > 5, f"perception stalled while paused ({p.calls} calls in 0.3s)"


def test_nothing_is_actuated_while_paused():
    r, act = _loop_runner()
    _run_paused(r, 0.3)
    assert act.applied == [], "a paused run reached the device"
    assert r._ticks == 0, "the state machine advanced while paused"
    assert r.stats.targets_tapped == 0
    assert r.stats.balls_thrown == 0
    assert r.stats.encounters == 0
    assert r.ctx.state is BotState.BOOT, "state changed while paused"


def test_the_ledger_does_not_stage_paused_frames():
    """A staged frame is claimed by a resolving intent, and nothing resolves while
    paused - so staging only evicts the frame the pending intent still needs."""
    ledger = _Ledger()
    r, _ = _loop_runner(ledger=ledger)
    _run_paused(r, 0.3)
    assert ledger.staged == 0, f"{ledger.staged} paused frames entered the ledger"


def test_the_encounter_ring_does_not_collect_paused_frames(tmp_path):
    """The ring holds the tail of an encounter so a catch award can be labelled. A pause
    inside an encounter must not roll that away and replace it with idle frames."""
    r, _ = _loop_runner(_Perceptor(on_map=False, encounter=True),
                        encounter_dump=tmp_path)
    r.ctx.state = BotState.ENCOUNTER
    _run_paused(r, 0.3)
    assert len(r._enc_ring) == 0, "paused frames were collected as encounter evidence"


def test_the_preview_keeps_repainting_while_paused(monkeypatch):
    """Without this the p and q keys are dead once paused: the repaint is what calls
    waitKey, and it was paced from the frozen clock too."""
    import sys
    import types

    seen = {"waitKey": 0}
    keys: list[int] = []
    fake = types.ModuleType("cv2")
    fake.WINDOW_NORMAL = 0
    for name in ("namedWindow", "resizeWindow", "destroyAllWindows", "imshow"):
        setattr(fake, name, lambda *a, **k: None)

    def _wait(_ms):
        seen["waitKey"] += 1
        return keys.pop(0) if keys else 255

    fake.waitKey = _wait
    monkeypatch.setitem(sys.modules, "cv2", fake)
    monkeypatch.setattr("pogobot.hud.render", lambda *a, **k: "img")

    r, _ = _loop_runner()
    r.display = True
    with _running(r):
        time.sleep(0.1)
        keys.append(ord("p"))        # pause the way the README says you can
        time.sleep(0.25)
        assert r.paused
        painted = seen["waitKey"]
        time.sleep(0.25)
        assert seen["waitKey"] - painted > 3, "the preview froze; p and q are unreachable"
        keys.append(ord("p"))        # ...and resume the same way
        time.sleep(0.25)
        assert not r.paused, "the p key could pause but not resume"


def test_a_pause_does_not_age_a_live_timeout(monkeypatch):
    """The design claim, through the real dispatcher rather than arithmetic.

    TARGETING's real budget is 4s; shortened here so the test costs a second rather than
    five. The value is incidental - what is under test is that `elapsed` does not grow
    across a pause and that the dispatcher therefore does not fire on resume.
    """
    handler = fsm.HANDLERS[BotState.TARGETING]
    monkeypatch.setattr(handler, "timeout_s", 0.5)
    r, _ = _loop_runner(_Perceptor(on_map=False, screen="Overworld", conf=0.1))
    with _running(r):
        time.sleep(0.1)
        r.ctx.state = BotState.TARGETING
        r.ctx.state_since = r.ctx.now
        elapsed_before = r.ctx.elapsed
        r.toggle_pause()
        time.sleep(handler.timeout_s * 3)
        aged = r.ctx.elapsed - elapsed_before
        r.toggle_pause()
        time.sleep(0.15)
        state = r.ctx.state
    assert aged < handler.timeout_s, f"the state machine aged {aged:.2f}s across the pause"
    assert state is BotState.TARGETING, f"resuming fired the timeout ({state})"


def test_the_hud_does_not_discount_paused_time_twice():
    """`SessionStats` subtracts paused_seconds itself, so it needs the REAL clock. Giving
    it ctx.now - which already has paused time removed - subtracted it twice and reported
    0m00s uptime on a run that had been working for minutes."""
    r, _ = _loop_runner()
    r.stats.started = time.perf_counter() - 600.0     # ten minutes ago
    r.stats.paused_seconds = 300.0                     # five of them paused
    r._pause_total = 300.0
    r.ctx.now = time.perf_counter() - r.stats.paused_seconds   # what the loop stamps
    captured = {}

    def _fake_render(*a, **kw):
        captured["status"] = kw.get("status", "")
        return "img"

    import pogobot.hud
    original = pogobot.hud.render
    pogobot.hud.render = _fake_render
    try:
        r._render(Frame(seq=1, ts=time.perf_counter(), bgr=np.zeros((64, 32, 3), np.uint8)),
                  mkobs(on_map=True))
    finally:
        pogobot.hud.render = original
    assert captured["status"].startswith("5m00s"), \
        f"uptime should be 600s real - 300s paused = 5m00s, got {captured['status']!r}"


def test_pausing_abandons_a_tap_whose_answer_we_will_not_see():
    """An Intent is a causal claim, and the frozen clock made it uncheckable.

    `IntentLedger.resolve` gates a corpus row on `now - intent.ts <= causal_max_s` (5s).
    Both terms are the FSM clock, so a pause shrinks the measured latency to the working
    time and the gate stops meaning anything. Measured against the real ledger: a tap
    answered 600.6 real seconds later was written to the corpus as `latency: 0.6,
    CONFIRMED, verified: false` - a human reviewer reads that latency as evidence that
    the tap caused the screen, while in fact the phone sat unwatched for ten minutes and
    a person may have opened it by hand.

    So the claim dies with the pause. EXPIRED is the outcome we can support, and the
    ledger refuses anything that is not CONFIRMED or REFUTED.
    """
    from pogobot.effects import IntentOutcome
    from pogobot.fsm import Intent

    resolved = []

    class _Recording(_Ledger):
        def resolve(self, intent, outcome, now):
            resolved.append((intent.target_name, outcome))

    r = _runner(ledger=_Recording())
    r.ctx.intent = Intent(ts=r.ctx.now, target_name="pokemon", confidence=0.9,
                          tap_norm=(0.5, 0.63), xywhn=(0.5, 0.63, 0.08, 0.05),
                          expected=BotState.ENCOUNTER, frame_seq=1)
    r.toggle_pause()
    r._sync_pause()

    assert r.ctx.intent is None, "a pause left a tap-intent open for the resume to score"
    assert resolved == [("pokemon", IntentOutcome.EXPIRED)], \
        f"the ledger must be told the tap expired, not left to confirm it later: {resolved}"


def test_resuming_cannot_confirm_a_tap_made_before_the_pause():
    """The same defect through run(), because the isolated check above cannot show that
    the confirmation is what a resume would otherwise reach."""
    from pogobot.fsm import Intent

    r, _ = _loop_runner(_Perceptor(on_map=False, screen="PokemonEncounter", conf=0.99))
    with _running(r):
        time.sleep(0.1)
        r.ctx.state = BotState.TARGETING
        r.ctx.state_since = r.ctx.now
        r.ctx.intent = Intent(ts=r.ctx.now, target_name="pokemon", confidence=0.9,
                              tap_norm=(0.5, 0.63), xywhn=(0.5, 0.63, 0.08, 0.05),
                              expected=BotState.ENCOUNTER, frame_seq=1)
        r.toggle_pause()
        time.sleep(0.2)
        held = r.ctx.intent
        r.toggle_pause()
        time.sleep(0.2)
    assert held is None, "the intent survived the pause and was there for the resume to score"
