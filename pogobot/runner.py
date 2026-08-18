"""The only module that holds both a FrameSource and an Actuator.

Everything that made v1 hard to reason about is concentrated here on purpose, and kept
small: one place applies effects, one place writes `state`, one place decides that the
run is over. `fsm.step` and `perception.observe` stay pure and testable.
"""

from __future__ import annotations

import json
import logging
import signal
from collections import deque
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
    SetFlag,
    SetIntent,
    Swipe,
    Tap,
    Transition,
    is_actuation,
)
from .frames import Frame, FrameSource
from .observation import Observation, Tristate
from .stats import SessionStats, append_session

log = logging.getLogger("pogobot")

# Preview refresh rate. The inference loop runs slower than this; redisplaying the cached
# HUD in between keeps the window responsive to the q key without re-rendering.
DISPLAY_FPS = 30.0

# How often a headless run logs its counters, so a long session reports progress.
REPORT_EVERY = 300.0

# How much of the tail of an encounter to keep for labelling. The catch sequence -
# ball wobble, "Gotcha!", then the XP/candy/stardust award - runs several seconds.
ENCOUNTER_RING_SECONDS = 6.0

# States whose per-visit bookkeeping must reset on entry.
_RESET_ON_ENTRY = ("spun_disc", "taps_in_state")


class Runner:
    def __init__(self, cfg: Config, source: FrameSource, actuator, perceptor,
                 ledger=None, keyboard=None, trace_path: Optional[Path] = None,
                 display: bool = True, stats_path: Optional[Path] = None,
                 dashboard=None, encounter_dump: Optional[Path] = None):
        self.cfg = cfg
        self.source = source
        self.actuator = actuator
        self.perceptor = perceptor
        self.ledger = ledger
        self.keyboard = keyboard
        self.display = display
        self.stats_path = stats_path
        self.dashboard = dashboard
        self.encounter_dump = encounter_dump
        # A ring of frames from inside an encounter. On exit these are the frames that
        # would show a catch award screen - the evidence a real catch counter needs.
        # Sized in seconds, not frames: at 8 inference fps a 8-frame ring held one second
        # and rolled the award sequence away before the encounter ended.
        self._enc_ring: deque = deque(maxlen=max(8, int(cfg.infer_fps * ENCOUNTER_RING_SECONDS)))
        self.ctx = fsm.Context(cfg=cfg, state=BotState.BOOT,
                               state_since=time.perf_counter(), now=time.perf_counter())
        self.ctx.last_map_ts = time.perf_counter()
        # The actuator, not the config, is the authority on whether anything was actually
        # sent: --replay swaps in a NullActuator regardless of cfg.dry_run.
        self.stats = SessionStats(dry_run=bool(getattr(actuator, "dry_run", False)
                                               or cfg.dry_run))
        self._next_report = self.stats.started + REPORT_EVERY
        self._trace = open(trace_path, "a", buffering=1) if trace_path else None
        self._stop = False
        self._halt_reason: Optional[str] = None
        self._encounter_left_at: Optional[float] = None
        self._ticks = 0
        self._fps = 0.0
        self._last_frame: Optional[Frame] = None
        self._last_hud = None          # last rendered HUD image, reused between inferences
        self._last_obs: Optional[Observation] = None
        self._last_shown = 0.0
        self._shown_hud = 0
        self._shown_raw = 0

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

    def _count_transition(self, e: Transition) -> None:
        """Called before the state changes, so `ctx.state` is still the source state."""
        st = self.stats
        src, dst = self.ctx.state, e.to
        if src is BotState.ENCOUNTER and dst is BotState.RECOVERING \
                and e.outcome is IntentOutcome.EXPIRED:
            # An encounter that outran its budget is ABANDONED, not finished: the screen is
            # still up - that is why it timed out - and `desired_state` outranks RECOVERING,
            # so the next tick reads the same screen and comes straight back. Driving one
            # 100s encounter screen through the real FSM produced 4 encounters, 4 catch
            # attempts and 3 recoveries for one real Pokemon. Record when we left instead of
            # counting an end, so the return trip is recognisable as the same encounter.
            self._encounter_left_at = self.ctx.now
        elif dst is BotState.ENCOUNTER and src is not BotState.ENCOUNTER:
            # It is the same encounter only if the map was never confirmed in between: a
            # recovery that actually worked lands on the map, and a genuinely new encounter
            # can only be reached through it.
            resumed = (src is BotState.RECOVERING
                       and self._encounter_left_at is not None
                       and self.ctx.last_map_ts <= self._encounter_left_at)
            if not resumed:
                self._encounter_left_at = None
                st.on_encounter_start()
        elif src is BotState.ENCOUNTER and dst is not BotState.ENCOUNTER:
            st.on_encounter_end()
            # Only a genuine end: an abandoned encounter still has its screen up, so its
            # frames are not award screens and would mislabel the training set.
            self._dump_encounter_ring()
        if dst is BotState.ROCKET and src is not BotState.ROCKET:
            st.rockets_engaged += 1
        if dst is BotState.RECOVERING and src is not BotState.RECOVERING:
            st.recoveries += 1
        if src is BotState.POKESTOP and dst is BotState.POPUP:
            # Only the PokeStop handler's own two exits carry a claim about the stop: it
            # confirms after a POI screen opened and dwelled, and refutes on "Walk closer
            # to interact". Both leave to POPUP.
            #
            # Every OTHER way out of POKESTOP is also REFUTED, because the intent expected
            # POKESTOP and got something else - the tap missed and the map is still up, or
            # the classifier calls the open POI screen "Overworld" (its Poi class has 8
            # training samples). Counting those as out-of-range put a "Walk closer to
            # interact" number on screen for stops the bot never got a range answer about.
            if e.outcome is IntentOutcome.CONFIRMED:
                st.stops_collected += 1
            elif e.outcome is IntentOutcome.REFUTED:
                st.stops_out_of_range += 1
        if e.outcome is IntentOutcome.EXPIRED \
                and src in (BotState.TARGETING, BotState.POKESTOP):
            # TARGETING and POKESTOP are both post-tap wait states, so an expiry in either
            # is one target tap that never produced the screen it claimed. Counting only
            # TARGETING silently dropped every stop tap whose POI screen never opened.
            st.taps_expired += 1

    def _halt(self, reason: str) -> None:
        """The single place a run is declared halted.

        `halts` was incremented only in the Halt-effect branch, so the four places that
        abort the loop directly - a dead capture source, the actuator circuit breaker, the
        stale-frame watchdog and repeated perception failures - each logged "HALTED",
        returned 1, and then recorded a session with halts=0. The lifetime total counted
        the one halt the FSM can emit and none of the ones the runner raises itself.
        """
        if self._halt_reason is None:
            self.stats.halts += 1
        self._halt_reason = reason

    def _dump_encounter_ring(self) -> None:
        """Write the frames leading up to an encounter ending, for labelling.

        A catch and a flee are indistinguishable to the bot today, so `catch_attempts`
        cannot be promoted to a catch count without evidence. These frames are that
        evidence: the award screen, if there was one, is in here.
        """
        if self.encounter_dump is None or not self._enc_ring:
            return
        try:
            import cv2
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.encounter_dump.mkdir(parents=True, exist_ok=True)
            for i, bgr in enumerate(self._enc_ring):
                cv2.imwrite(str(self.encounter_dump / f"{stamp}_{self._ticks:07d}_{i}.png"), bgr)
        except Exception:
            log.exception("could not write the encounter frames")
        finally:
            self._enc_ring.clear()

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
                self._count_transition(e)
                self.enter_state(e.to, e.outcome, e.reason)
            elif isinstance(e, SetIntent):
                self.ctx.intent = e.intent
            elif isinstance(e, SetFlag):
                setattr(self.ctx, e.name, e.value)
            elif isinstance(e, Cooldown):
                self.ctx.cooldowns.append((e.x, e.y, self.ctx.now + e.seconds))
            elif isinstance(e, ClearSpatialMemory):
                self.ctx.cooldowns.clear()
                log.info("cleared spatial memory: %s", e.reason)
            elif isinstance(e, Note):
                log.log(logging.WARNING if e.level == "warn" else logging.INFO, e.text)
            elif isinstance(e, Halt):
                self._halt(e.reason)
                self._stop = True
                self.enter_state(BotState.HALTED, IntentOutcome.CARRIED, e.reason)
            elif is_actuation(e):
                if self.actuator.apply(e, now=self.ctx.now):
                    budget = getattr(e, "budget", "tap")
                    if budget == "throw":
                        self.stats.on_ball_thrown()
                    elif budget == "tap" and isinstance(e, Tap):
                        self.stats.targets_tapped += 1
                    self.ctx.last_action[budget] = self.ctx.now
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

        # SIGTERM's default action kills the process without unwinding, so `kill`,
        # `timeout` and a system shutdown all skipped the finally block below - losing the
        # session summary and its record in the stats history. Ask for a clean stop instead.
        previous = {}
        def _request_stop(signum, _frame):
            log.info("received %s; finishing the current tick",
                     signal.Signals(signum).name)
            self._stop = True
        # Resolved by name: CPython on Windows has no SIGHUP, and naming it in a tuple
        # raises AttributeError before the try below can catch anything - which killed
        # run() before the loop started, on a platform the README says is supported.
        for _name in ("SIGTERM", "SIGINT", "SIGHUP"):
            sig = getattr(signal, _name, None)
            if sig is None:
                continue
            try:
                previous[sig] = signal.signal(sig, _request_stop)
            except (OSError, ValueError):
                pass

        try:
            while not self._stop:
                now = time.perf_counter()
                self.ctx.now = now

                if not self.source.healthy():
                    reason = getattr(self.source, "failure_reason", lambda: "")() or ""
                    if reason:
                        self._halt(f"capture source died: {reason}")
                    else:
                        log.info("frame source exhausted; finishing")
                    break
                if not self.actuator.healthy():
                    self._halt("actuator circuit breaker tripped (adb failing)")
                    break

                if now < next_infer:
                    # Redisplay the last rendered HUD rather than a bare frame. Passing
                    # obs=None here used to draw the un-annotated frame ~1000x/second
                    # (the rate is capped only by waitKey), while the HUD was drawn once
                    # per inference at 8Hz - so the overlay was visible for roughly 8
                    # frames in 1000 and appeared to strobe.
                    if self.display and self._last_hud is not None \
                            and now - self._last_shown >= 1.0 / DISPLAY_FPS:
                        self._last_shown = now
                        # Repaint the newest frame under the most recent observation, so
                        # the video stays smooth while the overlay updates at infer_fps.
                        # Skipped for a replay directory, where reading consumes a frame.
                        if not getattr(self.source, "sequential", False):
                            fresh = self.source.read()
                            if fresh is not None:
                                self._last_frame = fresh
                                self._render(fresh, self._last_obs)
                        if not self._blit(window):
                            break
                    else:
                        time.sleep(0.002)
                    continue

                frame: Optional[Frame] = self.source.read()
                if frame is None:
                    # A stale or missing frame must never be treated as a fresh one; v1
                    # served the last good frame forever and tapped a phone it could not see.
                    time.sleep(0.01)
                    if now - self.ctx.last_map_ts > cfg.timings.stuck_watchdog:
                        self._halt("no usable frames")
                        break
                    continue

                frames += 1
                self._last_frame = frame
                next_infer = now + 1.0 / max(cfg.infer_fps, 0.1)

                kbd = self.keyboard.state if self.keyboard else Tristate.UNKNOWN
                try:
                    obs = self.perceptor.observe(frame, keyboard=kbd)
                    consecutive_errors = 0
                except Exception:
                    consecutive_errors += 1
                    log.exception("perception failed (%d consecutive)", consecutive_errors)
                    if consecutive_errors >= 10:
                        self._halt("perception failing repeatedly")
                        break
                    continue

                if self.ctx.state is BotState.ENCOUNTER and self.encounter_dump is not None:
                    self._enc_ring.append(frame.bgr.copy())
                if obs.on_map:
                    self.ctx.last_map_ts = now
                if fsm.rocket_screen(obs, cfg):
                    self.ctx.last_rocket_ts = now
                if self.ledger is not None:
                    self.ledger.stage(frame, obs)

                effects = fsm.step(obs, self.ctx)
                self.apply(effects, obs)
                if self.dashboard is not None:
                    try:
                        self.dashboard.update(obs, self.ctx.state, self._fps)
                    except Exception:
                        log.exception("dashboard update failed")
                self._write_trace(obs, effects)
                self._ticks += 1

                if now >= self._next_report:
                    self._next_report = now + REPORT_EVERY
                    log.info("session: %s", self.stats.hud_line(now))

                elapsed = now - t0
                if elapsed >= 1.0:
                    self._fps = frames / elapsed
                    frames, t0 = 0, now

                if self.display and not self._show(window, frame, obs):
                    break
        except KeyboardInterrupt:
            log.info("interrupted by user")
        finally:
            # close() runs BEFORE the handlers are restored, and cleanup is not instant
            # (the actuator flushes its queue, the ledger flushes its writer). Restoring
            # first meant a second SIGTERM during shutdown - a system shutdown, `timeout
            # -k`, an impatient second Ctrl-C - hit the default disposition and killed the
            # process mid-cleanup, losing the session record. Measured: exit 143, no line
            # in sessions.jsonl. While our handler is still installed it only re-sets a
            # flag that is already set.
            try:
                self.close()
            finally:
                for sig, handler in previous.items():
                    try:
                        signal.signal(sig, handler)
                    except Exception:
                        # Never let restoration hide the shutdown it follows; a handler
                        # that was not installed from Python comes back as None, and
                        # signal.signal(sig, None) raises TypeError.
                        log.exception("could not restore the %s handler",
                                      getattr(sig, "name", sig))
        if self._halt_reason:
            log.error("HALTED: %s", self._halt_reason)
            return 1
        return 0

    def _render(self, frame: Frame, obs: Optional[Observation]) -> None:
        """Draw the HUD and cache it. Never caches a bare frame."""
        if obs is None:
            return
        from . import hud
        stats = self.actuator.stats()
        extra = {"taps": stats.get("sent", 0), "state_s": f"{self.ctx.elapsed:.1f}"}
        if self.ledger is not None:
            extra["saved"] = self.ledger.stats().get("written", 0)
        self._last_hud = hud.render(frame.bgr, obs, self.cfg, self.ctx.state, self._fps, extra)
        self._shown_hud += 1

    def _show(self, window: str, frame: Frame, obs: Observation) -> bool:
        """Render the HUD for a fresh observation and display it."""
        self._last_obs = obs
        self._render(frame, obs)
        self._last_shown = self.ctx.now
        return self._blit(window)

    def _blit(self, window: str) -> bool:
        """Push the cached HUD to the window. Returns False when the user presses q."""
        import cv2
        if self._last_hud is None:
            return True
        cv2.imshow(window, self._last_hud)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        # The session record is the durable output of the run, so it is written from a
        # finally: an actuator, ledger or trace that throws on the way down must not eat
        # it. `actuator.stats()` in the log line below is a live call into a component we
        # have just closed, which is exactly the kind of thing that used to.
        try:
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
                try:
                    self._trace.close()
                except Exception:
                    log.exception("could not close the trace file")
            if self.display:
                try:
                    import cv2
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            log.info("stopped after %d ticks (%d HUD renders); actuator=%s",
                     self._ticks, self._shown_hud, self.actuator.stats())
        finally:
            if self.stats_path is not None:
                try:
                    append_session(self.stats_path, self.stats.summary())
                except Exception:
                    log.exception("could not append the session record")
            log.info("session summary:\n%s", self.stats.report())
