"""The state machine. Pure: (Observation, Context) -> list[Effect].

Structure exists to make the v1 failure modes unrepresentable:

  * Handlers cannot write `state`. They return a Transition and the runner applies it
    through one function that also stamps the clock and resolves the intent.
  * The timeout is checked by the dispatcher BEFORE the handler body runs, so a timeout
    can never be shadowed by an earlier branch (v1's CLOSING_POPUP timeout was an `elif`
    below the tap branch and was unreachable while that branch held).
  * Every handler must declare `timeout_s` and `on_timeout`; the registry check at import
    time turns a missing one into a startup error rather than a silent livelock.
  * At most one interrupt fires per tick, and an interrupt never also transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .effects import (
    Back,
    BotState,
    Cooldown,
    ClearSpatialMemory,
    Effect,
    Halt,
    IntentOutcome,
    Note,
    SetFlag,
    SetIntent,
    Swipe,
    Tap,
    Transition,
)
from .observation import Observation, Tristate

TARGETABLE = {"pokemon", "pokestop", "pokestop_rocket"}
ROCKET_TARGETS = {"pokestop_rocket"}
STOP_TARGETS = {"pokestop", "pokestop_rocket"}


@dataclass
class Intent:
    """A tap we made and the claim it implies, held until the screen answers."""

    ts: float
    target_name: str
    confidence: float
    tap_norm: tuple[float, float]
    xywhn: tuple[float, float, float, float]
    expected: BotState
    frame_seq: int


@dataclass
class Context:
    """Mutable runtime state. Handlers read it; only the runner writes it."""

    cfg: Config
    state: BotState = BotState.BOOT
    state_since: float = 0.0
    now: float = 0.0
    intent: Optional[Intent] = None
    cooldowns: list = field(default_factory=list)   # (x, y, expires_at)
    last_action: dict = field(default_factory=dict)  # budget -> ts
    settle_until: float = 0.0
    spun_disc: bool = False
    rotate_dir: str = "left"
    last_map_ts: float = 0.0
    taps_in_state: int = 0
    stats: dict = field(default_factory=lambda: {"spins": 0, "catches": 0, "rockets": 0})

    @property
    def elapsed(self) -> float:
        return self.now - self.state_since

    def ready(self, budget: str, gap: float) -> bool:
        """True when `budget` has been idle for `gap` and the UI is not mid-transition."""
        if self.now < self.settle_until:
            return False
        return self.now - self.last_action.get(budget, -1e9) >= gap

    def is_cool(self, x: float, y: float) -> bool:
        r = self.cfg.cooldowns.radius_frac
        return not any((x - cx) ** 2 + (y - cy) ** 2 <= r * r
                       for cx, cy, exp in self.cooldowns if exp > self.now)


# ---------------------------------------------------------------- observations

def encounter_confirmed(obs: Observation, cfg: Config) -> bool:
    """Encounter needs positive evidence, never 'none of the above'.

    The optical route was measured and rejected. Ball-colour matching does not separate at
    all (Overworld scores HIGHER than encounters: 0.268 vs 0.005 median, because the map is
    full of red and blue objects), and the combined ball+flee test fires on 27% of overworld
    frames while catching only 30% of encounters. So the classifier owns this call, with the
    high-precision optical map signal and the X button as hard vetoes.

    A worked example this protects: a Gym screen classifies as PokemonEncounter @0.59.
    Under the 0.60 gate it is correctly refused, and the bot leaves via POPUP instead of
    throwing balls at a gym.
    """
    if obs.map_ball.value or obs.x_button.value:
        return False
    return obs.screen.is_("PokemonEncounter", min_conf=cfg.screen_min_conf)


def rocket_screen(obs: Observation, cfg: Config) -> bool:
    """A Team GO Rocket step. The affirmative pill finder hits 100% of the labelled
    GruntBattleButton and ChooseParty frames and ~0% of the menu screens."""
    if obs.map_ball.value:
        return False
    return obs.screen.is_("Rocket", min_conf=cfg.screen_min_conf)


def pick_target(obs: Observation, ctx: Context):
    """Best in-reach, not-cooled detection. Pokemon outrank stops, then confidence."""
    cfg = ctx.cfg
    best = None
    for d in obs.detections:
        if d.name not in TARGETABLE:
            continue
        if d.conf < cfg.target_confidence:
            continue
        if cfg.target_mode == "pokemon" and d.name != "pokemon":
            continue
        if cfg.target_mode == "pokestop" and d.name not in STOP_TARGETS:
            continue
        if not cfg.fight_rockets and d.name in ROCKET_TARGETS:
            continue
        x, y = d.center_norm
        scale = cfg.reach.stop_scale if d.name in STOP_TARGETS else 1.0
        rx = max(cfg.reach.radius_x * cfg.range_scale * scale, 1e-6)
        ry = max(cfg.reach.radius_y * cfg.range_scale * scale, 1e-6)
        dx = (x - cfg.reach.center_x) / rx
        dy = (y - cfg.reach.center_y) / ry
        if (dx * dx + dy * dy) ** 0.5 > cfg.reach.tolerance:
            continue
        if not ctx.is_cool(x, y):
            continue
        rank = (1 if d.name == "pokemon" else 0, d.conf)
        if best is None or rank > best[0]:
            best = (rank, d)
    return None if best is None else best[1]


# ---------------------------------------------------------------- handlers

class Handler:
    state: BotState
    timeout_s: float = 30.0

    def on_timeout(self, obs: Observation, ctx: Context) -> list[Effect]:
        return [Transition(BotState.RECOVERING, IntentOutcome.EXPIRED,
                           f"{self.state.value} timed out after {ctx.elapsed:.1f}s")]

    def step(self, obs: Observation, ctx: Context) -> list[Effect]:
        return []


class Boot(Handler):
    state = BotState.BOOT
    timeout_s = 30.0

    def step(self, obs, ctx):
        if obs.on_map:
            return [Transition(BotState.SCANNING, IntentOutcome.CARRIED, "map confirmed")]
        return []

    def on_timeout(self, obs, ctx):
        return [Transition(BotState.RECOVERING, IntentOutcome.CARRIED, "no map within 30s of boot")]


class Scanning(Handler):
    state = BotState.SCANNING
    timeout_s = 1e9   # scanning is the resting state; the stuck-watchdog covers it

    def step(self, obs, ctx):
        cfg = ctx.cfg
        if not obs.on_map:
            # v1's SCANNING had neither an else branch nor a watchdog - the only state
            # with no way out. If we believe we are scanning but no map is visible, we
            # are wrong about something; escalate rather than swiping at whatever is up.
            if ctx.now - ctx.last_map_ts > cfg.timings.popup_timeout:
                return [Transition(BotState.RECOVERING, IntentOutcome.CARRIED,
                                   "scanning but the map is not visible")]
            return []
        target = pick_target(obs, ctx)
        if target is not None:
            if not ctx.ready("tap", cfg.timings.tap_target):
                return []
            x, y = target.center_norm
            # `expected` is the state that would CONFIRM the detection was right, not
            # the state we pass through while waiting. Setting it to TARGETING scored a
            # successful Pokemon tap as REFUTED and cooled a location that worked.
            expected = BotState.POKESTOP if target.name in STOP_TARGETS else BotState.ENCOUNTER
            goto = BotState.POKESTOP if target.name in STOP_TARGETS else BotState.TARGETING
            intent = Intent(ts=ctx.now, target_name=target.name, confidence=target.conf,
                            tap_norm=(x, y), xywhn=target.xywhn, expected=expected,
                            frame_seq=obs.seq)
            return [
                SetIntent(intent),
                Tap(x, y, f"target {target.name} conf={target.conf:.2f}"),
                Transition(goto, IntentOutcome.CARRIED, f"tapped {target.name}"),
            ]
        if cfg.auto_rotate and ctx.ready("rotate", cfg.timings.rotate_camera) \
                and ctx.now - ctx.last_map_ts >= cfg.timings.scanning_idle_rotate:
            y = 0.58
            x1, x2 = (0.58, 0.42) if ctx.rotate_dir == "left" else (0.42, 0.58)
            return [
                Swipe(x1, y, x2, y, f"rotate camera {ctx.rotate_dir}", duration_ms=350, budget="rotate"),
                ClearSpatialMemory("camera rotated; remembered positions no longer valid"),
            ]
        return []


class Targeting(Handler):
    """We tapped something we believed was a Pokemon. Wait for the screen to answer."""

    state = BotState.TARGETING
    timeout_s = 4.0

    def step(self, obs, ctx):
        if obs.on_map:
            return []           # transition not yet visible; wait for timeout
        return []

    def on_timeout(self, obs, ctx):
        x, y = ctx.intent.tap_norm if ctx.intent else (0.5, 0.5)
        return [
            Cooldown(x, y, ctx.cfg.cooldowns.on_expired, "tap produced no screen change"),
            Transition(BotState.SCANNING, IntentOutcome.EXPIRED, "targeting timed out"),
        ]


class Encounter(Handler):
    state = BotState.ENCOUNTER
    timeout_s = 25.0

    def step(self, obs, ctx):
        cfg = ctx.cfg
        if cfg.catch_mode == "manual":
            return []
        if cfg.catch_mode == "flee":
            if ctx.ready("flee", cfg.timings.throw_ball):
                return [Tap(0.09, 0.08, "flee encounter", budget="flee")]
            return []
        if ctx.ready("throw", cfg.timings.throw_ball):
            return [Swipe(0.50, 0.84, 0.50, 0.38, "throw ball", duration_ms=160, budget="throw")]
        return []

    def on_timeout(self, obs, ctx):
        return [
            Note("encounter exceeded its budget; backing out", "warn"),
            Transition(BotState.RECOVERING, IntentOutcome.EXPIRED, "encounter timeout"),
        ]


class Pokestop(Handler):
    """Spin the disc, then leave. A positive is only recorded once the screen confirms
    we actually opened an interactable POI - v1 wrote one 0.8s after the swipe with no
    verification at all, which produced 69% of the poisoned corpus."""

    state = BotState.POKESTOP
    timeout_s = 8.0

    def step(self, obs, ctx):
        cfg = ctx.cfg
        if obs.stop_out_of_range.value:
            x, y = ctx.intent.tap_norm if ctx.intent else (0.5, 0.5)
            return [
                Cooldown(x, y, cfg.cooldowns.out_of_range, "PokeStop out of range"),
                Note("PokeStop out of range; leaving", "info"),
                Transition(BotState.POPUP, IntentOutcome.REFUTED, "out of range"),
            ]
        if not obs.x_button.value:
            return []                       # POI screen not up yet
        if not ctx.spun_disc:
            if not ctx.ready("spin", cfg.timings.spin_disc):
                return []
            return [Swipe(0.25, 0.45, 0.75, 0.45, "spin photo disc", duration_ms=220, budget="spin"),
                    SetFlag("spun_disc", True)]
        if ctx.ready("close", cfg.timings.close_menu):
            return [Transition(BotState.POPUP, IntentOutcome.CONFIRMED, "disc spun; leaving")]
        return []

    def on_timeout(self, obs, ctx):
        return [Transition(BotState.POPUP, IntentOutcome.EXPIRED, "pokestop timeout")]


class Rocket(Handler):
    """Team GO Rocket via the in-game auto-battler: press the affirmative pill when one
    is present, otherwise tap to advance dialogue. No combat vision needed."""

    state = BotState.ROCKET
    timeout_s = 150.0

    def step(self, obs, ctx):
        cfg = ctx.cfg
        if not ctx.ready("rocket", cfg.timings.rocket_tap):
            return []
        if obs.action_pill_xy is not None:
            x, y = obs.action_pill_xy
            return [Tap(x, y, "rocket: affirmative button", budget="rocket")]
        if obs.screen.is_("Rocket", min_conf=cfg.screen_min_conf):
            return [Tap(0.50, 0.62, "rocket: advance dialogue", budget="rocket")]
        return []

    def on_timeout(self, obs, ctx):
        return [
            Note("rocket encounter exceeded its budget", "warn"),
            Transition(BotState.RECOVERING, IntentOutcome.EXPIRED, "rocket timeout"),
        ]


class Popup(Handler):
    """Close a closable overlay. Only ever taps a button it actually located."""

    state = BotState.POPUP
    timeout_s = 4.0

    def step(self, obs, ctx):
        if obs.on_map:
            return [Transition(BotState.SCANNING, IntentOutcome.CARRIED, "overlay closed")]
        if obs.close_button_xy is not None and ctx.ready("close", ctx.cfg.timings.close_menu):
            x, y = obs.close_button_xy
            return [Tap(x, y, "close overlay", budget="close")]
        return []

    def on_timeout(self, obs, ctx):
        return [Transition(BotState.RECOVERING, IntentOutcome.CARRIED, "overlay would not close")]


class Recovering(Handler):
    """Escalating unstick. Never blind-taps a fixed coordinate - that is what created the
    v1 menu loop. BACK first, then a located close button, then halt."""

    state = BotState.RECOVERING
    timeout_s = 6.0

    def step(self, obs, ctx):
        if obs.on_map:
            return [Transition(BotState.SCANNING, IntentOutcome.CARRIED, "recovered to map")]
        if ctx.taps_in_state == 0 and ctx.ready("back", 1.0):
            return [Back("recover: dismiss with BACK")]
        if obs.close_button_xy is not None and ctx.ready("close", ctx.cfg.timings.close_menu):
            return [Tap(*obs.close_button_xy, "recover: located close button", budget="close")]
        return []

    def on_timeout(self, obs, ctx):
        if ctx.now - ctx.last_map_ts > ctx.cfg.timings.stuck_watchdog:
            return [Halt(f"no confirmed map for {ctx.now - ctx.last_map_ts:.0f}s; stopping "
                         f"rather than tapping blindly")]
        return [Transition(BotState.SCANNING, IntentOutcome.CARRIED, "recovery attempt over")]


class Halted(Handler):
    state = BotState.HALTED
    timeout_s = 1e9

    def step(self, obs, ctx):
        return []


HANDLERS = {h.state: h() for h in
            (Boot, Scanning, Targeting, Encounter, Pokestop, Rocket, Popup, Recovering, Halted)}

# Startup contract: a state without a handler, timeout, or on_timeout is a bug, not a livelock.
for _s in BotState:
    if _s not in HANDLERS:
        raise RuntimeError(f"BotState.{_s.name} has no handler")
    _h = HANDLERS[_s]
    if not isinstance(getattr(_h, "timeout_s", None), (int, float)):
        raise RuntimeError(f"{_s.name} handler must declare a numeric timeout_s")
    if _h.on_timeout.__func__ is Handler.on_timeout and _s not in (BotState.HALTED, BotState.SCANNING):
        raise RuntimeError(f"{_s.name} handler must override on_timeout")


# ---------------------------------------------------------------- dispatch

def interrupts(obs: Observation, ctx: Context) -> list[Effect]:
    """At most one fires per tick. Interrupts act but never change state."""
    cfg = ctx.cfg
    if obs.keyboard is Tristate.TRUE and ctx.state is not BotState.ENCOUNTER \
            and ctx.ready("keyboard", cfg.timings.keyboard_check):
        return [Back("soft keyboard is up"), Note("dismissing soft keyboard")]
    if obs.claim_pill.value and not obs.map_ball.value and ctx.ready("claim", cfg.timings.claim_reward):
        if obs.action_pill_xy is not None:
            return [Tap(*obs.action_pill_xy, "claim rewards", budget="claim")]
    return []


def desired_state(obs: Observation, ctx: Context) -> Optional[BotState]:
    """Unambiguous observations that outrank whatever the current handler wants.

    Order matters: rocket screens carry an X button, so if POPUP outranked ROCKET the bot
    would close the grunt dialogue instead of fighting it.
    """
    cfg = ctx.cfg
    if ctx.state is BotState.HALTED:
        return None
    if encounter_confirmed(obs, cfg):
        return BotState.ENCOUNTER
    if cfg.fight_rockets and rocket_screen(obs, cfg):
        return BotState.ROCKET
    if obs.on_map and ctx.state in (BotState.POPUP, BotState.RECOVERING, BotState.BOOT,
                                    BotState.ENCOUNTER):
        return BotState.SCANNING
    if obs.in_overlay and ctx.state in (BotState.SCANNING, BotState.TARGETING):
        return BotState.POPUP
    return None


def step(obs: Observation, ctx: Context) -> list[Effect]:
    """One tick. Pure - returns what should happen, does not make it happen."""
    fired = interrupts(obs, ctx)
    if fired:
        return fired

    want = desired_state(obs, ctx)
    if want is not None and want is not ctx.state:
        outcome = IntentOutcome.CARRIED
        if ctx.intent is not None:
            outcome = (IntentOutcome.CONFIRMED if want is ctx.intent.expected
                       else IntentOutcome.REFUTED)
        return [Transition(want, outcome, f"observation implies {want.value}")]

    handler = HANDLERS[ctx.state]
    if ctx.elapsed > handler.timeout_s:
        return handler.on_timeout(obs, ctx)
    return handler.step(obs, ctx)
