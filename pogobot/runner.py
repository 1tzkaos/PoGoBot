"""The only module that holds both a FrameSource and an Actuator.

Everything that made v1 hard to reason about is concentrated here on purpose, and kept
small: one place applies effects, one place writes `state`, one place decides that the
run is over. `fsm.step` and `perception.observe` stay pure and testable.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from . import fsm
from .config import Config
from .effects import (
    Back,
    BotState,
    ClearSpatialMemory,
    Cooldown,
    Effect,
    Halt,
    IntentOutcome,
    Note,
    SetIntent,
    Swipe,
    Tap,
    Transition,
    is_actuation,
)
from .frames import Frame, FrameSource
from .observation import Observation, Tristate

log = logging.getLogger("pogobot")

# States whose per-visit bookkeeping must reset on entry.
_RESET_ON_ENTRY = ("spun_disc", "taps_in_state")


class Runner:
    def __init__(self, cfg: Config, source: FrameSource, actuator, perceptor,
                 ledger=None, keyboard=None, trace_path: Optional[Path] = None,
                 display: bool = True):
        self.cfg = cfg
        self.source = source
        self.actuator = actuator
        self.perceptor = perceptor
        self.ledger = ledger
        self.keyboard = keyboard
        self.display = display
        self.ctx = fsm.Context(cfg=cfg, state=BotState.BOOT,
                               state_since=time.perf_counter(), now=time.perf_counter())
        self.ctx.last_map_ts = time.perf_counter()
        self._trace = open(trace_path, "a", buffering=1) if trace_path else None
        self._stop = False
        self._halt_reason: Optional[str] = None
        self._ticks = 0
        self._fps = 0.0

    # ---------------------------------------------------------------- state

    def enter_state(self, to: BotState, outcome: IntentOutcome, reason: str) -> None:
        """The single writer of `state`.

        v1 assigned `state` at 12 sites and stamped the clock at only 9 of them, so three
        transitions handed the next state a stale start time and its timeout fired
        instantly. Resolving the intent here means it can never be silently dropped.
        """
        ctx = self.ctx
        if ctx.intent is not None and outcome is not IntentOutcome.CARRIED:
            self._resolve_intent(ctx.intent, outcome)
            ctx.intent = None
        if ctx.state is not to:
            log.info("%s -> %s (%s)", ctx.state.value, to.value, reason)
        ctx.state = to
        ctx.state_since = ctx.now
        for attr in _RESET_ON_ENTRY:
            setattr(ctx, attr, False if isinstance(getattr(ctx, attr), bool) else 0)

    def _resolve_intent(self, intent, outcome: IntentOutcome) -> None:
        cd = self.cfg.cooldowns
        seconds = {IntentOutcome.CONFIRMED: cd.on_success,
                   IntentOutcome.REFUTED: cd.on_refuted,
                   IntentOutcome.EXPIRED: cd.on_expired}.get(outcome)
        if seconds:
            # v1 never cooled a SUCCESSFUL interaction, so it re-tapped the same PokeStop
            # forever and manufactured the duplicate training corpus.
            x, y = intent.tap_norm
            self.ctx.cooldowns.append((x, y, self.ctx.now + seconds))
        if self.ledger is not None:
            try:
                self.ledger.resolve(intent, outcome, self.ctx.now)
            except Exception:
                log.exception("ledger.resolve failed")

    # ---------------------------------------------------------------- effects

    def apply(self, effects: list[Effect], obs: Observation) -> None:
        """One place applies everything, so dry-run and tracing cannot be forgotten."""
        for e in effects:
            if isinstance(e, Transition):
                self.enter_state(e.to, e.outcome, e.reason)
            elif isinstance(e, SetIntent):
                self.ctx.intent = e.intent
            elif isinstance(e, Cooldown):
                self.ctx.cooldowns.append((e.x, e.y, self.ctx.now + e.seconds))
            elif isinstance(e, ClearSpatialMemory):
                self.ctx.cooldowns.clear()
                log.info("cleared spatial memory: %s", e.reason)
            elif isinstance(e, Note):
                log.log(logging.WARNING if e.level == "warn" else logging.INFO, e.text)
            elif isinstance(e, Halt):
                self._halt_reason = e.reason
                self._stop = True
                self.enter_state(BotState.HALTED, IntentOutcome.CARRIED, e.reason)
            elif is_actuation(e):
                if self.actuator.apply(e, now=self.ctx.now):
                    self.ctx.last_action[getattr(e, "budget", "tap")] = self.ctx.now
                    self.ctx.taps_in_state += 1
                    if isinstance(e, (Tap, Swipe, Back)):
                        self.ctx.settle_until = self.ctx.now + self.cfg.timings.ui_settle

    # ---------------------------------------------------------------- trace

    def _write_trace(self, obs: Observation, effects: list[Effect]) -> None:
        if self._trace is None:
            return
        rec = {
            "seq": obs.seq, "t": round(obs.ts, 3), "state": self.ctx.state.value,
            "screen": obs.screen.label, "conf": round(obs.screen.conf, 3),
            "map": obs.map_ball.value, "x": obs.x_button.value, "enc": obs.encounter.value,
            "red": round(obs.map_ball.detail.get("red", 0.0), 4),
            "orange": round(obs.map_ball.detail.get("orange", 0.0), 4),
            "pill": obs.action_pill_xy is not None,
            "close": obs.close_button_xy is not None,
            "dets": [[d.name, round(d.conf, 2)] for d in obs.detections],
            "eff": [type(e).__name__ for e in effects],
            "age_ms": round(obs.frame_age * 1000, 1),
        }
        self._trace.write(json.dumps(rec) + "\n")

    # ---------------------------------------------------------------- loop

    def run(self) -> int:
        cfg = self.cfg
        next_infer = 0.0
        frames = 0
        t0 = time.perf_counter()
        consecutive_errors = 0
        window = "PoGoBot"
        if self.display:
            import cv2
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 420, 900)

        log.info("running (dry_run=%s, catch=%s, targets=%s, rockets=%s)",
                 cfg.dry_run, cfg.catch_mode, cfg.target_mode, cfg.fight_rockets)
        try:
            while not self._stop:
                now = time.perf_counter()
                self.ctx.now = now

                if not self.source.healthy():
                    self._halt_reason = f"capture source died: {getattr(self.source, 'failure_reason', lambda: 'unknown')()}"
                    break
                if not self.actuator.healthy():
                    self._halt_reason = "actuator circuit breaker tripped (adb failing)"
                    break

                frame: Optional[Frame] = self.source.read()
                if frame is None:
                    # A stale or missing frame must never be treated as a fresh one; v1
                    # served the last good frame forever and tapped a phone it could not see.
                    time.sleep(0.01)
                    if now - self.ctx.last_map_ts > cfg.timings.stuck_watchdog:
                        self._halt_reason = "no usable frames"
                        break
                    continue

                frames += 1
                if now < next_infer:
                    if self.display:
                        self._show(window, frame, None)
                    continue
                next_infer = now + 1.0 / max(cfg.infer_fps, 0.1)

                kbd = self.keyboard.state if self.keyboard else Tristate.UNKNOWN
                try:
                    obs = self.perceptor.observe(frame, keyboard=kbd)
                    consecutive_errors = 0
                except Exception:
                    consecutive_errors += 1
                    log.exception("perception failed (%d consecutive)", consecutive_errors)
                    if consecutive_errors >= 10:
                        self._halt_reason = "perception failing repeatedly"
                        break
                    continue

                if obs.on_map:
                    self.ctx.last_map_ts = now
                if self.ledger is not None:
                    self.ledger.stage(frame, obs)

                effects = fsm.step(obs, self.ctx)
                self.apply(effects, obs)
                self._write_trace(obs, effects)
                self._ticks += 1

                elapsed = now - t0
                if elapsed >= 1.0:
                    self._fps = frames / elapsed
                    frames, t0 = 0, now

                if self.display and not self._show(window, frame, obs):
                    break
        except KeyboardInterrupt:
            log.info("interrupted by user")
        finally:
            self.close()
        if self._halt_reason:
            log.error("HALTED: %s", self._halt_reason)
            return 1
        return 0

    def _show(self, window: str, frame: Frame, obs: Optional[Observation]) -> bool:
        import cv2
        from . import hud
        if obs is None:
            img = frame.bgr
        else:
            stats = self.actuator.stats()
            extra = {"taps": stats.get("applied", 0), "state_s": f"{self.ctx.elapsed:.1f}"}
            if self.ledger is not None:
                extra["saved"] = self.ledger.stats().get("written", 0)
            img = hud.render(frame.bgr, obs, self.cfg, self.ctx.state, self._fps, extra)
        cv2.imshow(window, img)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        for closer in (getattr(self.source, "release", None),
                       getattr(self.actuator, "close", None),
                       getattr(self.keyboard, "stop", None),
                       getattr(self.ledger, "close", None)):
            if closer:
                try:
                    closer()
                except Exception:
                    log.exception("cleanup step failed")
        if self._trace:
            self._trace.close()
        if self.display:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        log.info("stopped after %d ticks; actuator=%s", self._ticks, self.actuator.stats())
