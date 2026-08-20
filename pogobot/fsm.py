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

from .accounts import AccountView
from .config import Config, Timings
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
    last_rocket_ts: float = 0.0
    left_encounter_ts: float = 0.0
    throws_this_encounter: int = 0
    failed_encounters: int = 0
    restocking_until: float = 0.0
    restock_stops_at_start: int = 0
    #: set by the runner from the rolling 24h spin quota
    spins_exhausted: bool = False
    taps_in_state: int = 0
    #: refreshed by the runner from the UI tree; None until first read
    accounts: Optional[AccountView] = None
    switch_target: Optional[str] = None
    switch_phase: str = "open"
    #: ctx.now at the moment the current login tap landed; 0.0 means none has (yet, or at
    #: all - the target was already active and no login was ever tapped)
    switch_login_ts: float = 0.0
    stats: dict = field(default_factory=lambda: {"spins": 0, "catches": 0, "rockets": 0})

    @property
    def elapsed(self) -> float:
        return self.now - self.state_since

    @property
    def restocking(self) -> bool:
        return self.now < self.restocking_until

    def ready(self, budget: str, gap: float, ignore_settle: bool = False) -> bool:
        """True when `budget` has been idle for `gap` and the UI is not mid-transition.

        `ignore_settle` is for actions taken against a button we have optically LOCATED.
        The settle window exists to stop us tapping blind through an animation; when the
        target is visibly on screen right now that reasoning does not apply, and waiting
        loses the button. Measured live: the Rocket BATTLE pill was visible for ~1s and
        the settle window from the preceding close tap swallowed the whole opportunity.
        """
        if self.now < self.settle_until and not ignore_settle:
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
        if ctx.spins_exhausted and d.name in STOP_TARGETS:
            # Past the daily cap a stop cannot yield, and tapping it only produces the
            # "walk closer" banner that made this look like a distance problem.
            continue
        if ctx.restocking and d.name not in STOP_TARGETS:
            # Restocking: ignore Pokemon entirely until the bag is refilled, otherwise
            # every spawn on a dense map outranks the stops we came for.
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

    def timeout(self, ctx: Context) -> float:
        """The budget for this state, in seconds. Overridden where the budget belongs to
        the Config rather than to the handler - see `Switching`."""
        return self.timeout_s

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
                return [Tap(0.095, 0.095, "flee encounter", budget="flee")]
            return []
        if ctx.throws_this_encounter >= cfg.max_throws_per_encounter:
            # Throws are doing nothing: out of balls, or a Pokemon we cannot land. Either
            # way the encounter is over for us. Leave by the flee icon rather than sitting
            # here until the timeout, which is what wedged the bot in a no-ball encounter.
            if ctx.ready("flee", 1.0):
                return [
                    Note(f"{ctx.throws_this_encounter} throws with no result; leaving", "warn"),
                    Tap(0.095, 0.095, "flee: throws exhausted", budget="flee"),
                    Transition(BotState.SCANNING, IntentOutcome.EXPIRED, "throws exhausted"),
                ]
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
    """Open the stop, let the game's auto-spin collect it, then leave.

    No disc swipe: the game has auto-spin built in, so opening the stop is the whole
    interaction. v1 swiped across the screen at y=0.45 on every stop, which on a screen
    that was NOT a PokeStop (a mis-tapped gym, or a tap that missed and left us on the
    map) dragged the map and rotated the camera instead.

    The detection is CONFIRMED once a POI screen actually opens - v1 wrote its positive
    0.8s after the swipe with no verification at all, which produced 69% of the poisoned
    corpus.
    """

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
            return []                       # POI screen has not opened yet
        if not ctx.spun_disc:
            # Mark the visit the moment the screen is confirmed, then dwell briefly so
            # auto-spin can run and the item toast can clear before we close.
            return [SetFlag("spun_disc", True),
                    Note("PokeStop open; letting auto-spin collect", "info")]
        if ctx.now - ctx.state_since >= cfg.timings.stop_dwell:
            return [Transition(BotState.POPUP, IntentOutcome.CONFIRMED, "stop collected; leaving")]
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
        if obs.action_pill_xy is not None:
            # The pill is on screen right now; do not let a generic settle window miss it.
            if ctx.ready("rocket", cfg.timings.rocket_tap, ignore_settle=True):
                x, y = obs.action_pill_xy
                return [Tap(x, y, "rocket: affirmative button", budget="rocket")]
            return []
        if not ctx.ready("rocket", cfg.timings.rocket_tap):
            return []
        if obs.screen.is_("Rocket", min_conf=cfg.screen_min_conf):
            return [Tap(0.50, 0.62, "rocket: advance dialogue", budget="rocket")]
        return []

    def on_timeout(self, obs, ctx):
        return [
            Note("rocket encounter exceeded its budget", "warn"),
            Transition(BotState.RECOVERING, IntentOutcome.EXPIRED, "rocket timeout"),
        ]


class Switching(Handler):
    """Log into another account through the PGSharp overlay.

    Phases advance through `switch_phase` on the Context rather than through nested
    conditionals, so each tick makes exactly one decision from observable state: "open"
    drives the overlay to the target's login button, "settle" waits out whatever the
    login produces until the map is back AND the login has had time to land, and "verify"
    re-opens the overlay to read the asterisk before confirming anything.

    `settle` does NOT identify the screens that appear after a login. It cannot: Willow's
    dialogue classifies as Rocket @0.66, and the optical signal that separates a dialogue
    box fires on 5/5 ChooseParty frames, which is a real Rocket screen. What justifies
    clearing them is context - we just tapped a login button, so whatever is on screen is
    between us and the map. That claim expires with the state timeout.

    The map coming back is not proof the switch worked, either, for two SEPARATE reasons
    that both had to be fixed. First: PGSharp shuts its own panel as part of logging in,
    so every post-login read of the account list comes back `rows=0` regardless of
    whether the login succeeded - `verify` exists to re-open it and read the asterisk,
    the only ground truth for who is actually logged in. Second, and more subtly: the
    OUTGOING account's map can still be on screen for a second or two after the login tap
    - the game has not torn it down yet - so `obs.on_map` can turn true well before the
    login has actually landed (measured: ~14s tap-to-modal). A version that raced this
    re-opened the panel at +6s, correctly read the OUTGOING account still active, and
    concluded the switch had failed - when attempt 2 immediately after confirmed almost
    instantly, proving attempt 1 had simply not been given time to finish. That is why
    `_settle` waits out `Timings.switch_login_grace` before ever handing off to `verify`,
    and why a mismatch inside `verify` is never treated as final (see `_verify`) - "someone
    else is active" at one instant cannot be told apart from "not yet" from that instant
    alone. Only the state timeout (`Timings.switch_timeout`) is allowed to end a switch
    that never confirms, and `Runner` records that expiry so the next attempt waits out a
    backoff instead of re-tapping a control that has just refused us.
    """

    state = BotState.SWITCHING
    #: Declared so the import-time contract below still sees a numeric budget, and kept
    #: equal to the config default so the two can never disagree. `timeout()` is what the
    #: dispatcher actually asks, and it reads the LIVE config - `Timings.switch_timeout`
    #: was configurable in name only while this number was the one that counted.
    timeout_s = Timings().switch_timeout

    def timeout(self, ctx):
        return ctx.cfg.timings.switch_timeout

    def step(self, obs, ctx):
        if ctx.switch_phase == "settle":
            return self._settle(obs, ctx)
        if ctx.switch_phase == "verify":
            return self._verify(obs, ctx)
        cfg = ctx.cfg
        v = ctx.accounts
        if v is None or not v.available:
            return []                    # could not look; the timeout owns the outcome
        if not ctx.ready("switch", cfg.timings.switch_tap):
            return []
        if not v.panel_open:
            if v.launcher_norm is None:
                return []
            return [Tap(*v.launcher_norm, "switch: open the PGSharp overlay", budget="switch")]
        row = v.by_name(ctx.switch_target) if ctx.switch_target else None
        if row is None:
            if v.accounts_tab_norm is None:
                return []
            return [Tap(*v.accounts_tab_norm, "switch: select the Accounts tab", budget="switch")]
        if row.active:
            # Already on the target - no login tap is coming, so there is nothing for
            # the grace period to wait out. `Runner._begin_switch` zeroes
            # `switch_login_ts` at the start of every attempt, so leaving it alone here
            # means `_settle` treats the grace as already satisfied - and means attempt 2
            # can never inherit attempt 1's timestamp.
            return [SetFlag("switch_phase", "settle"),
                    Note(f"already logged into {row.name}; waiting for the map")]
        return [
            SetFlag("switch_phase", "settle"),
            SetFlag("switch_login_ts", ctx.now),
            Note(f"switching to {row.name}", "info"),
            Tap(*row.login_norm, f"switch: log into {row.name}", budget="switch"),
        ]

    def _settle(self, obs, ctx):
        cfg = ctx.cfg
        if not obs.on_map:
            if obs.close_button_xy is not None and ctx.ready("close", cfg.timings.close_menu):
                return [Tap(*obs.close_button_xy, "switch: close a post-login overlay",
                            budget="close")]
            if ctx.ready("back", cfg.timings.switch_clear):
                # Measured: one BACK dismissed the post-login news modal. BACK carries no
                # coordinate at all, which is why it is preferred to tapping a screen whose
                # layout we have exactly one example of.
                return [Back("switch: dismiss a post-login screen")]
            return []
        if ctx.now - ctx.switch_login_ts < cfg.timings.switch_login_grace:
            # The outgoing account's map can reappear before the login has actually
            # landed (see class docstring) - on_map alone is not the signal to act on.
            return []
        # The map is back and the login has had time to land. Hand off to verify in the
        # same tick rather than spending one just to flip the phase - `_verify` reads
        # whatever view is on the Context right now.
        return [SetFlag("switch_phase", "verify")] + self._verify(obs, ctx)

    def _verify(self, obs, ctx):
        """Re-open the overlay and read the asterisk - the only ground truth for who is
        logged in. Never adds its own staleness tracking: the runner drops `ctx.accounts`
        after every tap taken while SWITCHING, so a `None` or an already-stale-looking
        view here just means the next refresh has not landed yet, and doing nothing is
        the correct response either way.

        A mismatch here is never latched as final - see class docstring for why "someone
        else is active" cannot be told apart from "not yet" from a single read. It closes
        what it can and returns, and the next tick tries again from scratch; only the
        state timeout is allowed to end a switch that never confirms.
        """
        cfg = ctx.cfg
        v = ctx.accounts
        if v is None or not v.available:
            return []                    # could not look; wait for the next refresh
        if not ctx.ready("switch", cfg.timings.switch_tap):
            return []
        if not v.panel_open:
            if v.launcher_norm is None:
                return []
            return [Tap(*v.launcher_norm, "switch: re-open the overlay to verify",
                        budget="switch")]
        if not v.rows:
            # PGSharp remembers the last-viewed tab, so the panel can reopen on Cooldown
            # History rather than the account list. identify_account (accounts.py)
            # mirrors this same tab-follow for the same reason; not a third shape.
            if v.accounts_tab_norm is None:
                return []
            return [Tap(*v.accounts_tab_norm, "switch: follow the Accounts tab",
                        budget="switch")]
        if v.active is not None and v.active.name == ctx.switch_target:
            effects = []
            if v.close_norm is not None:
                effects.append(Tap(*v.close_norm, "switch: close overlay after verifying",
                                   budget="switch"))
            effects.append(Transition(BotState.SCANNING, IntentOutcome.CONFIRMED,
                                      f"logged into {ctx.switch_target}"))
            return effects
        # Someone else is active. The login is asynchronous, so this is not evidence of
        # failure by itself - only that it has not landed as of this read. A second login
        # tap here would be blind regardless, so close what we can and try again later;
        # the state timeout, not this check, is what ends a switch that truly never lands.
        effects = [Note(f"switch to {ctx.switch_target} not yet confirmed; "
                        f"{v.active.name if v.active else 'no one'} still active", "info")]
        if v.close_norm is not None:
            effects.append(Tap(*v.close_norm, "switch: close overlay; will re-check",
                               budget="switch"))
        return effects

    def on_timeout(self, obs, ctx):
        return [
            Note(f"account switch to {ctx.switch_target} never confirmed", "warn"),
            Transition(BotState.RECOVERING, IntentOutcome.EXPIRED, "switch timeout"),
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
            (Boot, Scanning, Targeting, Encounter, Pokestop, Rocket, Switching, Popup,
             Recovering, Halted)}

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
    if ctx.state is BotState.SWITCHING:
        # A switch owns the screen until it confirms or times out. Post-login screens
        # look like Rocket and like encounters; following them abandons the switch
        # half-done, logged into neither account cleanly.
        return None
    # While a Rocket fight is in progress, an encounter-looking screen is almost always
    # part of the fight. Only the map may pull us out; the reward encounter is picked up
    # once Rocket screens have stopped for rocket_hold seconds.
    rocket_recent = ctx.now - ctx.last_rocket_ts < cfg.timings.rocket_hold
    if ctx.state is BotState.ROCKET and rocket_recent and not obs.on_map:
        return None
    if encounter_confirmed(obs, cfg):
        # An encounter we deliberately left stays left for a moment. Re-entering it on the
        # next tick is a livelock: observed live as ENCOUNTER -> RECOVERING -> ENCOUNTER
        # repeating on a no-ball screen until the watchdog halted the run.
        held = (ctx.now - ctx.left_encounter_ts < cfg.timings.encounter_hold
                and ctx.last_map_ts <= ctx.left_encounter_ts)
        # Once the map is confirmed the screen really did change, so the next encounter is
        # a different Pokemon and must not be blocked. The hold exists only for the case
        # where we left and the same screen is still up.
        return None if held else BotState.ENCOUNTER
    if cfg.fight_rockets and rocket_screen(obs, cfg):
        return BotState.ROCKET
    if obs.on_map and ctx.state in (BotState.POPUP, BotState.RECOVERING, BotState.BOOT,
                                    BotState.ENCOUNTER, BotState.ROCKET, BotState.POKESTOP):
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
    if ctx.elapsed > handler.timeout(ctx):
        return handler.on_timeout(obs, ctx)
    return handler.step(obs, ctx)
