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
    DoubleTapDrag,
    Effect,
    Halt,
    IntentOutcome,
    Note,
    RestartApp,
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
    #: whether this run has a uiautomator reader at all, set once by `Runner.__init__`
    #: from `tree_reader is not None`. `accounts` cannot answer this: it is legitimately
    #: None between a read and the tap that invalidates it, so "no view right now" and
    #: "there will never be a view" look identical from inside a handler - and the two
    #: want opposite behaviour. `Preflight._autowalk_open` is the one that has to tell
    #: them apart: with no reader its widgets can never be located, so waiting out
    #: `config.AutoWalk.budget_s` buys nothing and costs 30s of a state that owns the
    #: screen. Defaults True because a Context nobody has told otherwise is one where the
    #: phases' own "could not look; wait for the next refresh" handling is the right
    #: answer; the runner overwrites it with the truth before the first tick.
    tree_available: bool = True
    switch_target: Optional[str] = None
    switch_phase: str = "open"
    #: applications of the post-switch zoom-out gesture fired so far in the "zoom" phase.
    #: Reset on every state entry (see runner._RESET_ON_ENTRY), so it starts at 0 both
    #: when a switch begins and again once it leaves SWITCHING - never carries between
    #: attempts, same reasoning as switch_login_ts below.
    switch_zoom_reps: int = 0
    #: coordinate-free BACK presses `_settle` has fired clearing a post-login screen so
    #: far, counting only ones the actuator actually accepted (see
    #: config.Timings.switch_clear_max for the bound and the 90-press storm it guards
    #: against). Reset on every state entry (see runner._RESET_ON_ENTRY), same reasoning
    #: as switch_zoom_reps just above - it must never carry between switch attempts, and
    #: like switch_zoom_reps it is advanced by `Runner.apply`, never by this pure
    #: handler, which cannot know whether a given Back actually reached the device.
    switch_clear_presses: int = 0
    #: ctx.now at the moment the current login tap landed; 0.0 means none has (yet, or at
    #: all - the target was already active and no login was ever tapped)
    switch_login_ts: float = 0.0
    #: tap+recheck cycles spent re-enabling Virtual Go Plus so far in the "goplus" phase.
    #: Reset on every state entry (see runner._RESET_ON_ENTRY), same reasoning as
    #: switch_zoom_reps above - it must never carry between switch attempts.
    switch_goplus_attempts: int = 0
    #: separating drags `Switching._separate_star` has spent on the PGSharp star this
    #: switch attempt, counting only ones the actuator actually accepted (see
    #: `config.StarSeparation.max_drags` for the bound and the measurement that makes more
    #: than one necessary). Reset on every state entry (see runner._RESET_ON_ENTRY):
    #: SWITCHING is entered once per attempt, which is exactly the scope of this bound -
    #: the same reasoning switch_goplus_attempts just above states. Advanced by
    #: `Runner.apply`, never by this pure handler, which cannot know whether a given Swipe
    #: reached the device.
    star_drags: int = 0
    #: ctx.now at the moment the AutoWalk ladder ("autowalk_open" through "autowalk_close"
    #: - see Switching._autowalk_open and neighbours) actually started. 0.0 means it has
    #: not started yet. Zeroed by Runner._begin_switch at the start of every NEW attempt -
    #: same reasoning, and same shape, as switch_login_ts below - never by
    #: runner._RESET_ON_ENTRY, because SWITCHING is entered once per attempt and this must
    #: survive every phase change within that one attempt, not just one state entry.
    switch_autowalk_since: float = 0.0
    #: Colour reading of the shortcut menu's "AutoWalk" icon glyph - TRUE means AutoWalk
    #: is ALREADY running for the target account and must not be tapped again (see
    #: `Switching._autowalk_menu`, `perception.autowalk_active_signal`, and
    #: `config.Thresholds` for the measured table and its honesty note). Refreshed by
    #: `Runner._refresh_accounts` in the SAME call, from the SAME view, that refreshes
    #: `accounts` itself - the uiautomator dump supplies the icon's bounds
    #: (`accounts.AccountView.autowalk_icon_rect_norm`), the CURRENT frame supplies the
    #: colour. This is why it needs no reset of its own in `Runner._begin_switch`, unlike
    #: `switch_autowalk_since` above: the icon box is only ever populated once the
    #: shortcut menu has actually rendered (`_autowalk_menu`), which is well after a new
    #: attempt's very FIRST accounts refresh - during "open" or "settle" - so that first
    #: refresh already overwrites whatever a PRIOR attempt left here with UNKNOWN before
    #: this field could ever be consulted for the new one.
    switch_autowalk_active: Tristate = Tristate.UNKNOWN
    #: ctx.now at the instant the MOST RECENT SWITCHING attempt released the screen -
    #: confirmed or expired, set by runner._count_transition. Deliberately NOT reset on
    #: state entry: unlike switch_zoom_reps/switch_goplus_attempts this is not per-visit
    #: bookkeeping, it is a standing fact ("a switch last let go of the screen at time
    #: T") that the stuck watchdog (Recovering.on_timeout) keeps consulting for as long
    #: as it is the most recent thing that happened. See that method's docstring for why
    #: it exists: a switch can legitimately occupy the screen for up to
    #: Timings.switch_timeout (240s), well past Timings.stuck_watchdog (120s), and
    #: without this a failed switch alone reliably tripped the watchdog the moment it
    #: handed off to RECOVERING.
    switch_exit_ts: float = 0.0
    #: CONSECUTIVE app restarts `Recovering.on_timeout` has spent without the map coming
    #: back, and `ctx.now` at the moment the most recent one was actually accepted by the
    #: actuator (0.0 means none has been). Both are written by `Runner.apply` and
    #: `Runner`'s read loop, never here - a pure handler cannot know whether the
    #: RestartApp it returned reached the device, the same reasoning `switch_zoom_reps`
    #: and `switch_clear_presses` above already state. Deliberately NOT in
    #: `runner._RESET_ON_ENTRY`: RECOVERING is entered and left every few seconds while
    #: the bot is wedged, so a per-entry reset would make the bound (`Config
    #: .max_app_restarts`) count to one forever. The count is zeroed by a confirmed map
    #: instead - see `Config.max_app_restarts` for why that is what makes it a bound.
    app_restarts: int = 0
    app_restart_ts: float = 0.0
    stats: dict = field(default_factory=lambda: {"spins": 0, "catches": 0, "rockets": 0})

    @property
    def elapsed(self) -> float:
        return self.now - self.state_since

    @property
    def restocking(self) -> bool:
        return self.now < self.restocking_until

    @property
    def map_stale_since(self) -> float:
        """The instant the stuck watchdog should measure staleness from.

        `last_map_ts` alone regresses the moment anything legitimately owns the screen
        without the map being visible: a switch can occupy it for up to
        Timings.switch_timeout (240s), well past Timings.stuck_watchdog (120s). Taking
        the LATER of `last_map_ts` and `switch_exit_ts` means a switch's own bounded
        duration is never by itself sufficient to look stuck, while stuckness that
        begins - or continues - after the switch has released the screen is still
        measured from exactly the same origin the watchdog has always used. With no
        switch involved, `switch_exit_ts` stays at its 0.0 default and this reduces to
        `last_map_ts` exactly.

        `app_restart_ts` earns the same credit, for the same reason and a measured
        failure. A relaunch this handler asked for cannot show a map while the game is
        cold-starting, which `Timings.app_restart_grace` puts at tens of seconds, and
        `Recovering._restarting` correctly holds the whole ladder silent for it. Without
        the credit, staleness kept counting from BEFORE the restart, so the run was
        already past `stuck_watchdog` the instant the relaunch began - and the runner's
        frame guard does not measure a run of missing frames, it asks this property and
        fires on the first `frame is None` it happens to read. Measured on the device: a
        restart at 21:42:49, the FSM still correctly logging "waiting out the app
        restart" at 21:42:56, and HALTED "no usable frames" at 21:43:01 - twelve seconds
        into a cold start the bot had itself decided to wait out, with a second restart
        still unspent. The bound does not weaken: restarts are capped by
        `Config.max_app_restarts`, `app_restart_ts` is cleared the moment a map is
        confirmed, and once the budget is gone staleness simply measures from the last
        relaunch, so the halt still arrives one `stuck_watchdog` later.

        The single property exists so every `stuck_watchdog` consumer - the pure FSM
        check in `Recovering.on_timeout` and the runner's own "no usable frames" guard
        in its read loop - shares one definition of staleness instead of two that can
        drift apart.
        """
        return max(self.last_map_ts, self.switch_exit_ts, self.app_restart_ts)

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
    GruntBattleButton and ChooseParty frames and ~0% of the menu screens.

    `exit_dialog` is vetoed here for the same reason `map_ball` already is: Pokemon GO's
    own exit-confirmation dialog classifies as Rocket (measured live) but its buttons
    sit nowhere near ROCKET's fixed dialogue-advance tap, so following it into ROCKET
    means a 150s stall - measured live: 430s consumed for two ROCKET<->RECOVERING round
    trips and nothing else - where interrupts() (the primary defence) would otherwise
    already have dismissed it with BACK. This is the secondary layer: it closes the
    window on ticks where that interrupt's own pacing gate withholds a repeat BACK press
    while the dialog is still on screen.
    """
    if obs.map_ball.value or obs.exit_dialog.value:
        return False
    return obs.screen.is_("Rocket", min_conf=cfg.screen_min_conf)


def reach_distance(cfg: Config, x: float, y: float, scale: float = 1.0) -> float:
    """How far a normalized point is from the avatar's reach, measured in the reach
    ellipse's OWN radii: at or below `cfg.reach.tolerance` is in reach, above it is not.

    Split out of `pick_target` so there is exactly one definition of the ellipse. The
    second consumer is the star-separation check (`Switching._separate_star`), whose whole
    claim is that the place a drag leaves the PGSharp star is somewhere a target detection
    could never be tapped - and a claim like that asserted against a re-typed copy of the
    formula stops meaning anything the moment the real one changes.
    """
    rx = max(cfg.reach.radius_x * cfg.range_scale * scale, 1e-6)
    ry = max(cfg.reach.radius_y * cfg.range_scale * scale, 1e-6)
    dx = (x - cfg.reach.center_x) / rx
    dy = (y - cfg.reach.center_y) / ry
    return (dx * dx + dy * dy) ** 0.5


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
        if reach_distance(cfg, x, y, scale) > cfg.reach.tolerance:
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
    login produces until the map is back AND the login has had time to land, "verify"
    re-opens the overlay to read the asterisk before confirming anything, "zoom" -
    entered only once verify has actually matched - fires the measured one-finger
    zoom-out before the switch is allowed to confirm, so PGoBot is not left driving the
    very-zoomed-in camera the game resets to after every login (see `_zoom`), and
    "goplus" - entered only once every zoom repeat has actually fired - re-enables the
    Virtual Go Plus toggle if it reads OFF (see `_goplus`), since a login turns it off
    every time, and "autowalk_open"/"autowalk_menu"/"autowalk_dialog"/"autowalk_close" -
    entered only once `_goplus` is itself done - drive PGSharp's floating star widget to
    start an AutoWalk route (see `_autowalk_open` and its neighbours), since the user
    wants one running after every switch.

    Everything from "zoom" onward has a SECOND caller: `Preflight` inherits these phases
    and runs exactly that tail once at startup, because a bot pointed at a freshly
    logged-in game needs the same three things a login does. Those methods are therefore
    written to work with no target at all - the only place `switch_target` still appears
    past "verify" is inside message strings, which ask `_label`/`_finished` rather than
    interpolate a None. Anything added below "verify" has to keep that true.

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

    @staticmethod
    def _label(ctx) -> str:
        """"switch" or "preflight" - which of the two runs owns these phases right now.

        `switch_target` is the discriminator because it is the one thing a preflight
        structurally does not have: `Runner._begin_preflight` enters `Preflight` at the
        "zoom" phase with the target left at None precisely so no branch here can name an
        account nothing logged into (see `Preflight`). Every branch from "zoom" onward is
        shared by both runs, so the alternative to asking is a second copy of each of
        them, differing only in a word.
        """
        return "switch" if ctx.switch_target else "preflight"

    @staticmethod
    def _finished(ctx, detail: str = "") -> str:
        """The reason string every exit from the zoom -> goplus -> autowalk chain carries.

        In one place because those phases are driven by BOTH a confirmed account switch
        and the startup preflight, and only the first of the two has an account to name.
        Left as an f-string on `switch_target` it read "logged into None" during a
        preflight - the exact false claim that made the failure this whole change exists
        for so hard to see: logs/sessions.jsonl recorded `account=None` for a four-hour
        run and nothing said why, so an operator-facing line that invents an account name
        is the last thing to hand whoever reads the log next.
        """
        head = (f"logged into {ctx.switch_target}" if ctx.switch_target
                else "startup preflight over")
        return f"{head}; {detail}" if detail else head

    def step(self, obs, ctx):
        if ctx.switch_phase == "settle":
            return self._settle(obs, ctx)
        if ctx.switch_phase == "verify":
            return self._verify(obs, ctx)
        if ctx.switch_phase == "zoom":
            return self._zoom(obs, ctx)
        if ctx.switch_phase == "goplus":
            return self._goplus(obs, ctx)
        if ctx.switch_phase == "autowalk_open":
            return self._autowalk_open(obs, ctx)
        if ctx.switch_phase == "autowalk_menu":
            return self._autowalk_menu(obs, ctx)
        if ctx.switch_phase == "autowalk_dialog":
            return self._autowalk_dialog(obs, ctx)
        if ctx.switch_phase == "autowalk_close":
            return self._autowalk_close(obs, ctx)
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
                # A LOCATED close button is targeted, not blind, so it is never subject
                # to switch_clear_max below - see that constant's docstring.
                return [Tap(*obs.close_button_xy, "switch: close a post-login overlay",
                            budget="close")]
            if ctx.switch_clear_presses >= cfg.timings.switch_clear_max:
                # Bound spent (config.Timings.switch_clear_max) - simply wait rather than
                # keep hammering BACK into what may be a legitimate multi-minute LOADING
                # screen. switch_timeout still owns the outcome, and _verify still runs
                # the instant the map returns.
                return []
            if ctx.ready("switch_clear", cfg.timings.switch_clear):
                # Measured: one BACK dismissed the post-login news modal. BACK carries no
                # coordinate at all, which is why it is preferred to tapping a screen whose
                # layout we have exactly one example of. `switch_clear_presses` is NOT
                # advanced here - this handler is pure and cannot know whether the
                # actuator will actually accept it (rate-limit / backpressure);
                # Runner.apply owns the count, and only for an ACCEPTED press, the same
                # pattern switch_zoom_reps/switch_goplus_attempts already use.
                return [Back("switch: dismiss a post-login screen", budget="switch_clear")]
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
            # Do not transition yet - hand off to the "zoom" phase (SetFlag, since a
            # handler cannot write switch_phase itself) so the confirmed switch gets its
            # camera-reset gesture BEFORE control passes to SCANNING. Confirming here
            # unconditionally, before the panel-close tap has even reached the device,
            # would have made the zoom a SCANNING-time action with no causal tie to the
            # switch that justified it.
            effects = []
            if v.close_norm is not None:
                effects.append(Tap(*v.close_norm, "switch: close overlay after verifying",
                                   budget="switch"))
            effects.append(SetFlag("switch_phase", "zoom"))
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

    def _zoom(self, obs, ctx):
        """Fire the measured one-finger zoom-out, then confirm the switch.

        Only reachable from `_verify`'s match branch, so this phase - and therefore the
        gesture itself - can never run on a switch that failed or merely timed out; a
        mismatch or an expiry never sets `switch_phase` to "zoom" in the first place.

        Waits for `obs.on_map` before ever touching the screen: the close tap `_verify`
        just queued is only just reaching the actuator's queue when this phase is first
        entered, so acting on the same tick would be driving the gesture from a view that
        still shows the account panel, not the map underneath it. One gesture per tick,
        gated by the same settle window (`Timings.ui_settle`) every other actuation in
        the system already paces itself by - no zoom-specific cadence was measured, so
        borrowing the existing one is honest about that rather than inventing a number.

        The CONFIRMED transition is no longer the last thing this phase does - see
        `_goplus`, which it hands off to once `repeats` have actually been applied.
        Deferring THAT hand-off, rather than confirming here and letting the toggle run
        as an ordinary SCANNING-time action, keeps the same causal tie `_verify`'s own
        comment describes: SWITCHING keeps owning the screen for exactly as long as it
        takes to finish everything the switch itself justified, and the whole 240s
        `switch_timeout` still bounds the lot if the map never reappears at all.

        `ctx.switch_zoom_reps` is NOT advanced here. This handler is pure and cannot know
        whether the `DoubleTapDrag` it emits will actually reach the device -
        `Actuator.apply` can legitimately refuse a live command (rate-limit, queue
        backpressure) without raising. `Runner.apply` increments the counter itself, and
        only when that same actuation is the one that was accepted - the same pattern
        `taps_in_state`/`targets_tapped` already use - so a rejected gesture cannot be
        counted as if it had fired and let this phase confirm the switch having sent
        fewer than `repeats` real zoom-outs.
        """
        if not obs.on_map:
            return []
        z = ctx.cfg.zoom
        if ctx.switch_zoom_reps >= z.repeats:
            return [SetFlag("switch_phase", "goplus")] + self._goplus(obs, ctx)
        if not ctx.ready("zoom", 0.0):
            return []                    # let the previous drag's settle window clear
        y2 = z.center_y - z.drag_frac
        # There is nothing to have confirmed during a preflight - no login was tapped -
        # so the clause naming one is omitted rather than filled with None; see
        # `_label`/`_finished` for the rule the whole chain follows.
        who = f" after confirming {ctx.switch_target}" if ctx.switch_target else ""
        return [
            DoubleTapDrag(z.center_x, z.center_y, z.center_x, y2,
                          f"{self._label(ctx)}: zoom out{who} "
                          f"({ctx.switch_zoom_reps + 1}/{z.repeats})",
                          duration_ms=z.duration_ms, budget="zoom"),
        ]

    def _goplus(self, obs, ctx):
        """Re-enable Virtual Go Plus if it reads OFF, once the zoom gesture is done -
        the last thing SWITCHING does before confirming (see `_zoom`'s docstring for why
        the confirmation is deferred this far rather than let this run as a SCANNING-time
        action with no causal tie to the switch that justified it).

        Reachable only from `_zoom`'s completion, so - like the gesture itself - this can
        never run on a switch that failed or merely timed out; a mismatch or an expiry
        never advances `switch_phase` past "verify" in the first place.

        ON or UNKNOWN/ABSENT both fall through immediately: the whole point of the three
        states in `perception.goplus_signal` is that "we do not know, or there is nothing
        there" must never be treated as "so tap it" - that is exactly the missing/ambiguous-
        signal-means-do-nothing rule this module observes everywhere else. Only a
        POSITIVELY read OFF is acted on, and even then `max_attempts` bounds the tap+
        recheck cycles: this must never block a switch from confirming, so the existing
        state timeout (`Timings.switch_timeout`), not a retry limit invented here, owns
        whatever happens if the toggle genuinely will not budge.

        Once done, hands off to "autowalk_open" - the CONFIRMED transition moved there in
        turn (see `_autowalk_close`) for the same reason it moved here from `_zoom`: every
        step SWITCHING still owns the screen gets to run before control is handed back.
        """
        if not obs.on_map:
            return []
        g = ctx.cfg.goplus
        if obs.goplus is not Tristate.FALSE or ctx.switch_goplus_attempts >= g.max_attempts:
            return [SetFlag("switch_phase", "autowalk_open")] + self._autowalk_open(obs, ctx)
        if not ctx.ready("goplus", g.press_wait):
            return []                    # either mid-settle, or waiting for the press to take
        return [Tap(g.tap_x, g.tap_y, f"{self._label(ctx)}: re-enable Virtual Go Plus",
                    budget="goplus")]

    def _autowalk_deadline(self, ctx) -> Optional[list]:
        """Common to every "autowalk_*" phase. Returns the give-up effects once the
        ladder's own wall-clock budget (config.AutoWalk.budget_s) has run out, or None
        while there is still time - see that class's docstring for why a wall clock, not
        an attempt count, is what has to bound this.

        Callers check this ONLY once the node they actually want was not found - a node
        that IS found is always used, however close to (or past) the budget the clock has
        run. The budget exists for "this is never going to appear", not to race a step
        that just this tick actually succeeded.

        Giving up is not the same as walking away. Reaching "autowalk_open"'s OWN
        deadline check means the star was never TAPPED across the whole budget - locating
        it normally taps and advances the phase in the SAME tick (see `_autowalk_open`),
        and the only other way to be here is a star that WAS located and deliberately left
        alone because it is collapsed onto the accounts launcher (`_separate_star`).
        Either way the shortcut menu was never opened, so there is nothing to clean up:
        this confirms immediately, exactly as before this method grew a cleanup step.
        Every OTHER phase is reachable only through that same star tap, so the menu IS
        open by the time any of them can be here - and leaving it that way is not
        cosmetic (see `_autowalk_close`'s own docstring: it sits over the reach
        ellipse SCANNING taps into, and the NEXT switch's own opening tap would toggle it
        SHUT instead of open, silently killing AutoWalk for the rest of the run). So those
        phases get one more chance at a LOCATED star tap before confirming, bounded by
        `config.AutoWalk.close_grace_s` on top of `budget_s` so this can never by itself
        push the switch toward `Timings.switch_timeout`. The AutoWalk dialog needs no
        special case here even though it is the one screen most likely to still be open at
        this point: `_autowalk_dialog` already presses CONTINUE LAST/OK, which is not a
        failure at all but the ladder's own goal, before ever consulting this method
        whenever the dialog and a button are both there - this is only ever reached once
        that phase has nothing left of its own to press.

        `ctx.accounts` is frequently `None` on exactly the tick this first fires -
        `Runner.apply` drops it after every actuation taken while SWITCHING (the star
        tap that opened the menu very much included), and only the next throttled tree
        refresh (`runner.ACCOUNTS_REFRESH`) puts a usable view back. A single check right
        at the budget boundary would see that `None` most of the time and never even
        attempt the closing tap - `close_grace_s` exists so a retry once a real view has
        landed gets a chance, the same reasoning `_autowalk_close`'s own wait already
        uses for the view its normal path needs.
        """
        cfg = ctx.cfg.autowalk
        elapsed = ctx.now - ctx.switch_autowalk_since
        if elapsed <= cfg.budget_s:
            return None
        late = f"AutoWalk did not complete in time ({ctx.switch_phase})"
        if ctx.switch_phase == "autowalk_open":
            return [Transition(BotState.SCANNING, IntentOutcome.CONFIRMED,
                               self._finished(ctx, late))]
        v = ctx.accounts
        if v is not None and v.available and v.star_norm is not None \
                and ctx.ready("switch", ctx.cfg.timings.switch_tap):
            return [
                Tap(*v.star_norm, "autowalk: close the shortcut menu (giving up)",
                    budget="switch"),
                Transition(BotState.SCANNING, IntentOutcome.CONFIRMED,
                           self._finished(ctx, late)),
            ]
        if elapsed <= cfg.budget_s + cfg.close_grace_s:
            return []             # still inside the cleanup allowance; wait for a view
        return [Transition(BotState.SCANNING, IntentOutcome.CONFIRMED,
                           self._finished(ctx, f"{late}; the menu may still be open"))]

    def _separate_star(self, ctx, v):
        """Drag the PGSharp star clear of the accounts launcher when the tree says the two
        floating widgets have collapsed onto each other - before anything taps the star.

        Three answers, and the difference between the last two is the whole point. `None`
        means the tree makes no such claim and the star is safe to tap. A NON-EMPTY list is
        the drag. An EMPTY list means they ARE collapsed and this tick has nothing to spend
        on separating them: the star must not be tapped, and the caller falls through to
        the ladder's own wall-clock deadline rather than returning, so a star that can
        never be safely tapped ends the ladder at `config.AutoWalk.budget_s` instead of
        holding the switch until `Timings.switch_timeout`.

        Why it is worth doing at all: measured immediately after `effects.RestartApp`
        relaunched the game, the star's clickable rect was (0,152)-(108,260) and the
        launcher's (0,152)-(272,245), putting the star's own centre (54,206) INSIDE the
        launcher. `star_norm` IS that centre, so the tap below would be delivered to the
        accounts launcher, which opens the PGSharp accounts panel - the exact screen
        `Recovering._panel_close` and the restart ladder above it exist to escape. The
        restart is what produces the layout, so without this the bot causes the wedge it
        recovers from.

        HERE, at the one place that taps the star, rather than in RECOVERING where the
        restart is decided, for two reasons that are both about what the code can actually
        see:

          * `desired_state` returns SCANNING for RECOVERING the instant `obs.on_map` is
            true, BEFORE any handler runs, so `Recovering.step` only ever executes off the
            map. The collapse becomes consequential exactly when the relaunched app is
            back ON the map - which is precisely when that handler stops running. A rung
            there would be dead in the case it was written for.
          * A drag needs a view that is current in a way RECOVERING's is not.
            `Runner._refresh_accounts` re-reads at `runner.RECOVER_ACCOUNTS_REFRESH` (10s)
            there and nothing drops the view after an actuation, so a drag decided from it
            could start from a coordinate the star has already left - putting a swipe on
            the game's own map, which rotates the camera and silently invalidates every
            cooldown (the failure `ClearSpatialMemory` exists for). During a switch the
            view is refreshed at `runner.ACCOUNTS_REFRESH` (2.5s) AND `Runner.apply` drops
            `ctx.accounts` after every applied effect, so the next drag can only ever be
            decided from a tree read taken AFTER the previous one landed. That is what
            makes "check the result, then try again" structurally true here rather than a
            matter of pacing - which matters because the gesture does not land where it is
            aimed: asking for y=626 landed the star's centre at 837, asking for 339 landed
            at 443, and asking for 356 moved it the other way, to 125.

        Covering this one call site covers every star tap there is: `_autowalk_close` and
        `_autowalk_deadline`'s cleanup tap are reachable only through the tap this rung
        guards.

        Both endpoints come from rects this dump reported - the start is the star rect's
        own centre, the end is `accounts.AccountView.star_clear_y_norm`, derived from the
        launcher's own bottom edge. Nothing here is a remembered coordinate, which for a
        widget whose entire problem is that it moves is the difference between a fix and a
        silent miss. The end is also checked against `reach_distance` before it is used:
        clearing the launcher is not the whole requirement, because a star parked inside
        the ellipse SCANNING taps into would only have moved the collision.
        """
        if not v.overlay_collapsed:
            return None
        cfg = ctx.cfg.star_separation
        if ctx.star_drags >= cfg.max_drags:
            # Bounded, and the bound resolves to skipping AutoWalk rather than to trying
            # the tap anyway. A star that will not move is a star whose tap opens the
            # accounts panel, and a switch that has already confirmed must not be spent
            # re-opening the screen the recovery ladder is for. The Note is the only
            # record an operator gets of why AutoWalk stopped happening.
            return [
                Note(f"the PGSharp star is still collapsed onto the accounts launcher "
                     f"after {ctx.star_drags} drag(s); skipping AutoWalk rather than "
                     f"tapping a control that opens the accounts panel", "warn"),
                Transition(BotState.SCANNING, IntentOutcome.CONFIRMED,
                           self._finished(ctx, "the PGSharp star could not be separated "
                                               "from the accounts launcher")),
            ]
        if v.panel_open:
            # Never drag while the accounts panel is up: during a switch that panel owns
            # the screen, and the star's rect then names a coordinate underneath it.
            return []
        star, clear_y = v.star_rect_norm, v.star_clear_y_norm
        if star is None or clear_y is None:
            # `overlay_collapsed` already proves both rects exist, so the only way to be
            # here is the landing zone refusing to name a place the screen actually has
            # (see `star_clear_y_norm`). Nothing is invented in its place; the caller's
            # own wall-clock deadline ends the phase.
            return []
        x = (star[0] + star[2]) / 2.0
        if reach_distance(ctx.cfg, x, clear_y,
                          max(1.0, ctx.cfg.reach.stop_scale)) <= ctx.cfg.reach.tolerance:
            # Separating the widgets must not simply move the collision. `star_clear_y_norm`
            # only knows about the launcher and the bottom of the screen; it cannot know
            # about the reach ellipse SCANNING taps into, because accounts.py is imported
            # BY this module and so cannot ask it. The check therefore belongs here, and it
            # is not academic: both widgets are user-draggable, and with the device's own
            # measured sizes (star 108x108, launcher 272x93) sitting mid-screen at x=0.5,
            # the landing zone computes to (0.5, 0.500) - 0.81 radii IN, taking a star that
            # was 1.42 radii OUT and putting it exactly where `pick_target` selects. Every
            # SCANNING tap at a detection there would then open PGSharp's shortcut menu
            # over the map, which is the failure `_autowalk_close`'s own docstring
            # describes. Widest ellipse of the two `pick_target` uses, since a landing zone
            # clear of only the narrower one is not clear at all.
            #
            # Refusing rather than inventing somewhere else, the same answer the off-screen
            # branch above gives: this rung knows one landing zone, and if that one is
            # unusable the honest reply is that it has nothing to offer. The star is still
            # not tapped, and the caller's own wall-clock deadline ends the phase.
            return []
        # Paced on the same gate every other step of this ladder uses, on its own budget
        # so a withheld drag and a withheld star tap can never be confused for each other.
        # The runner ticks at frame rate; ungated, this would spend the whole bound inside
        # one tree refresh, deciding every drag from the same view.
        if not ctx.ready("star_drag", ctx.cfg.timings.switch_tap):
            return []
        return [Swipe(x, (star[1] + star[3]) / 2.0, x, clear_y,
                      "autowalk: drag the star clear of the accounts launcher",
                      duration_ms=cfg.duration_ms, budget="star_drag")]

    def _autowalk_open(self, obs, ctx):
        """Tap the PGSharp star to open its shortcut menu (see the module docstring and
        accounts.py's for the widget itself). The star is LOCATED fresh from the current
        dump every time, never a remembered coordinate - it is described as draggable and
        was measured at two different positions hours apart.

        Locating it is not enough to tap it: after an app restart PGSharp lays the star on
        top of the accounts launcher, close enough that the star's own centre falls inside
        the launcher's clickable rect and this tap opens the accounts panel instead. So
        every tick asks `_separate_star` first, and a collapsed pair is dragged apart
        before anything is tapped.

        Only reachable from `_goplus`'s completion, so - like the zoom gesture and the
        Go Plus toggle before it - this can never run on a switch that failed or merely
        timed out.

        Deliberately does NOT gate on `obs.on_map`, unlike `_zoom`/`_goplus`: those two
        fire a BLIND gesture at a fixed screen coordinate and need `on_map` to prove the
        account panel underneath has actually closed before the gesture can land on the
        map rather than the panel. Every autowalk phase instead acts only on a coordinate
        it just read from the LIVE uiautomator tree (`ctx.accounts`, refreshed every tick
        regardless of what is on screen - see `Runner._refresh_accounts`) - the same shape
        `_verify` is in, and for the same reason `_verify` has no such gate either. Gating
        on `on_map` here would be actively wrong: PGSharp's own shortcut menu and the
        AlertDialog it opens sit on top of the map exactly while these phases need to act,
        and a full-screen AlertDialog dims the window behind it - this codebase's own
        exit-dialog fix (`perception.exit_dialog_signal`) is direct evidence that a dialog
        like that can read as NOT on-map. Gating on it would make `_autowalk_deadline`'s
        own escape hatch unreachable while the dialog it is trying to escape is up.
        """
        effects: list = []
        first_tick = ctx.switch_autowalk_since == 0.0
        if first_tick:
            # Start the ladder's own clock rather than checking a deadline of zero
            # elapsed time against it - see _autowalk_deadline.
            effects.append(SetFlag("switch_autowalk_since", ctx.now))
        v = ctx.accounts
        if v is not None and v.available and v.star_norm is not None:
            drag = self._separate_star(ctx, v)
            if drag is None:
                if not ctx.ready("switch", ctx.cfg.timings.switch_tap):
                    return effects
                return effects + [
                    Tap(*v.star_norm, "autowalk: open the PGSharp shortcut menu",
                        budget="switch"),
                    SetFlag("switch_phase", "autowalk_menu"),
                ]
            if drag:
                return effects + drag
            # Collapsed, with nothing to spend on separating them this tick. Deliberately
            # NOT a return: the star may not be tapped, so the only remaining answer is
            # the ladder's own deadline below - see `_separate_star`'s three answers.
        if not first_tick:
            gone = self._autowalk_deadline(ctx)
            if gone is not None:
                return gone
        return effects           # could not look, or the star was not found - wait

    def _autowalk_menu(self, obs, ctx):
        """Pick "AutoWalk" from the shortcut menu the previous phase opened, once it has
        actually rendered. Located by its own text among the menu's items, not by
        position - the same discipline the star lookup above uses.

        No `obs.on_map` gate - see `_autowalk_open`'s docstring; this phase acts on a
        live-located coordinate too, never a blind fixed one.

        Before tapping, checks whether AutoWalk is ALREADY running for this account -
        `ctx.switch_autowalk_active`, a colour reading of this SAME node's icon glyph
        (see `perception.autowalk_active_signal`, refreshed by `Runner._refresh_accounts`
        in lockstep with `ctx.accounts` itself). The user's own report, confirmed on the
        device: a blue glyph means the account is already autowalking, and tapping
        AutoWalk again must not happen. Only a positively-read TRUE skips - FALSE and
        UNKNOWN (including the one ambiguous sample this signal is known not to separate
        cleanly; see config.Thresholds) both fall through to the ordinary tap, because the
        safe failure direction here is "not active": tapping AutoWalk when it is already
        running is exactly today's behaviour, while silently skipping a walk the user
        actually wanted is not.

        Skipping still has to leave the shortcut menu in the state the rest of the ladder
        expects - closed - so it hands off straight to `_autowalk_close` in the same tick
        rather than simply confirming here (see that method's own docstring for why
        leaving the menu open is not cosmetic).
        """
        v = ctx.accounts
        if v is not None and v.available and v.autowalk_menu_norm is not None:
            if ctx.switch_autowalk_active is Tristate.TRUE:
                # No account is named during a preflight rather than one being invented -
                # see `_label`/`_finished`. The rule itself is untouched by which run is
                # driving it: only a positively-read TRUE skips the tap.
                who = f" for {ctx.switch_target}" if ctx.switch_target else ""
                return [
                    Note(f"AutoWalk already active{who}; not tapping it again - "
                         f"closing the shortcut menu instead", "info"),
                    SetFlag("switch_phase", "autowalk_close"),
                ] + self._autowalk_close(obs, ctx)
            if not ctx.ready("switch", ctx.cfg.timings.switch_tap):
                return []
            return [
                Tap(*v.autowalk_menu_norm, "autowalk: select AutoWalk from the shortcut menu",
                    budget="switch"),
                SetFlag("switch_phase", "autowalk_dialog"),
            ]
        gone = self._autowalk_deadline(ctx)
        return gone if gone is not None else []

    def _autowalk_dialog(self, obs, ctx):
        """The 'Auto-Generated GPX' dialog: press CONTINUE LAST when PGSharp offers it,
        otherwise OK (defaults to 50 POIs) - exactly the rule given in the task brief.

        Never touches the POI-count input field or either toggle group: this module has
        no coordinate for any of them (see accounts.py), so there is nothing here capable
        of tapping one even in error. `autowalk_dialog_open` requires the dialog's own
        title text, not merely the presence of button1/2/3 - generic Android AlertDialog
        ids some other dialog could also carry - so this never presses a button that
        belongs to a screen it only coincidentally resembles.

        No `obs.on_map` gate - see `_autowalk_open`'s docstring. This matters most here:
        the "Auto-Generated GPX" AlertDialog is the one screen in this ladder most likely
        to read as off-map (a real AlertDialog dims the window behind it), and this is the
        phase whose job is to act while that dialog is genuinely open.
        """
        v = ctx.accounts
        if v is not None and v.available and v.autowalk_dialog_open:
            target = v.autowalk_continue_last_norm or v.autowalk_ok_norm
            if target is not None:
                if not ctx.ready("switch", ctx.cfg.timings.switch_tap):
                    return []
                reason = ("autowalk: CONTINUE LAST"
                          if v.autowalk_continue_last_norm is not None
                          else "autowalk: OK (default 50 POIs)")
                return [Tap(*target, reason, budget="switch"),
                        SetFlag("switch_phase", "autowalk_close")]
        gone = self._autowalk_deadline(ctx)
        return gone if gone is not None else []

    def _autowalk_close(self, obs, ctx):
        """Tap the star again to close the shortcut menu, which stays open after the
        dialog is dismissed (measured live), then confirm the switch.

        Waits for the view first, exactly like its three siblings, and for the reason
        that makes waiting mandatory rather than merely tidy: `Runner.apply` drops
        `ctx.accounts` after EVERY actuation taken while SWITCHING - the star TOGGLES the
        menu, so a second decision taken from one stale view undoes the first - and only
        the next tree refresh, itself throttled to `runner.ACCOUNTS_REFRESH`, repopulates
        it. A version that confirmed on that `None` view instead of waiting could never
        fire the close tap at any tick rate the runner actually has: driven through the
        real Runner the tap appeared only when a single tick was longer than the whole
        refresh cadence (measured: absent at dt=0.1/0.5/1.0/2.0s, present at dt=3.0s).

        Leaving the menu open is not cosmetic. The menu overlaps the reach ellipse
        SCANNING taps into while the map-ball ROI stays clear of it, so `on_map` remains
        true and SCANNING taps straight into the menu; and the NEXT switch's
        `_autowalk_open` taps the star while that menu is already open, which toggles it
        SHUT - `_autowalk_menu` then finds no "AutoWalk" node, waits out the whole budget
        and confirms without ever starting a route. AutoWalk would work on the first
        switch of a run and silently stop thereafter.

        Bounded by the same `_autowalk_deadline` as its siblings, so the wait can never
        become a hang: a star that never reappears still confirms within
        `config.AutoWalk.budget_s + close_grace_s` rather than holding the switch to the
        full `Timings.switch_timeout` - the same allowance `_autowalk_deadline` itself
        gives `_autowalk_menu`/`_autowalk_dialog` for exactly this same closing tap, since
        by the time any of the three are running the menu is equally open and equally
        worth one more try to shut. Confirmation is still never conditional on the close
        tap SUCCEEDING - only on it having been emitted, or on the allowance being spent.

        No `obs.on_map` gate, for the same reason as its siblings (see `_autowalk_open`'s
        docstring): PGSharp's own menu is up on exactly the ticks this phase must act,
        and a dimmed window behind it can read as NOT on-map.
        """
        v = ctx.accounts
        if v is not None and v.available and v.star_norm is not None:
            if not ctx.ready("switch", ctx.cfg.timings.switch_tap):
                return []
            # Tap before Transition: `Runner.apply` walks the list in order, so the tap is
            # applied while the state is still SWITCHING (or PREFLIGHT, which drives this
            # same ladder) and the confirmation follows it.
            return [
                Tap(*v.star_norm, "autowalk: close the shortcut menu", budget="switch"),
                Transition(BotState.SCANNING, IntentOutcome.CONFIRMED,
                           self._finished(ctx)),
            ]
        gone = self._autowalk_deadline(ctx)
        return gone if gone is not None else []

    def on_timeout(self, obs, ctx):
        return [
            Note(f"account switch to {ctx.switch_target} never confirmed", "warn"),
            Transition(BotState.RECOVERING, IntentOutcome.EXPIRED, "switch timeout"),
        ]


#: The phases a preflight runs, in the order they chain into each other, and the phase
#: `Runner._begin_preflight` therefore enters at. One definition, consulted by both the
#: handler (which refuses any phase outside it) and the runner (which starts the chain), so
#: "which phases are the preflight" cannot drift into two answers that disagree.
PREFLIGHT_PHASES = ("zoom", "goplus", "autowalk_open", "autowalk_menu",
                    "autowalk_dialog", "autowalk_close")

#: What is still unconfirmed when a preflight is abandoned at a given phase, as a COMPLETE
#: clause. The whole value of the warning `Preflight.on_timeout` writes is naming WHICH of
#: the three steps did not happen - "the preflight timed out" alone leaves an operator to
#: guess whether the camera, the toggle or the route is the one that is wrong. Derived from
#: the chain above: a phase is reached only once every phase before it has finished, so
#: everything from the stuck phase onward is what is outstanding.
#:
#: Whole clauses rather than noun phrases slotted into a shared "... did not happen this
#: run" tail, because the last rung is not a subset of the others - by then the three steps
#: HAVE happened and what is outstanding is a menu left open. Slotted into that tail it
#: read "but nothing, but PGSharp's shortcut menu may still be over the map did not happen
#: this run", which is both unreadable and, taken at face value, the opposite of the truth,
#: on the one warning that matters most: a shortcut menu left over the reach ellipse
#: silently kills AutoWalk for the rest of the run (see `Switching._autowalk_close`). A
#: change whose whole premise is that a misleading log line hid a failure for four hours
#: does not get to ship another one.
_PREFLIGHT_UNDONE = {
    "zoom": "the zoom-out, Virtual Go Plus and AutoWalk did not happen this run",
    "goplus": "Virtual Go Plus and AutoWalk did not happen this run",
    "autowalk_open": "AutoWalk did not start this run",
    "autowalk_menu": "AutoWalk did not start this run",
    "autowalk_dialog": "AutoWalk did not start this run",
    # By this phase AutoWalk has been started; only PGSharp's own shortcut menu is still
    # open, which is not cosmetic - see `Switching._autowalk_close`.
    "autowalk_close": "AutoWalk started, but PGSharp's shortcut menu may still be over "
                      "the map",
}

#: The preflight phases that fire a BLIND gesture at a fixed screen coordinate and so
#: refuse to act without `obs.on_map` (`Switching._zoom`, `Switching._goplus`). Named
#: because they are also the phases that own the screen for no good reason once something
#: closable is on it - see `Preflight.step`.
_PREFLIGHT_MAP_PHASES = ("zoom", "goplus")


class Preflight(Switching):
    """The same zoom-out, Virtual Go Plus and AutoWalk steps a confirmed switch runs, once
    at startup, before the bot is allowed to play.

    The user's report is one cause with three symptoms: a four-hour run that "didn't zoom
    out, nor did it continue autowalk nor did it turn on the Pokemon Go Plus". All three
    of those live INSIDE this ladder, which until now ran only after an account switch -
    and that run performed no switch at all (logs/trace.jsonl for it: 115,210 frames, zero
    SWITCHING). A bot that has just been pointed at a freshly-logged-in game is in exactly
    the state a switch leaves behind, so it needs exactly the same three steps.

    A SUBCLASS driving the existing phase methods, rather than a copy of them or a
    `switch_phase` walked through SWITCHING itself. Three things decided that:

      * Nothing is re-implemented. `_zoom`, `_goplus` and the `_autowalk_*` phases come
        from `Switching`, so the measured gesture, the three-answer Go Plus reading
        (ON/UNKNOWN are still never tapped) and the "already autowalking, do not tap it
        again" rule all keep working here by construction rather than by a second author
        remembering them. From "zoom" onward those methods only ever named
        `ctx.switch_target` inside message strings, which is why they are reusable with no
        login and no target at all - and why those strings now ask `_label`/`_finished`
        instead of interpolating a None. The two methods below that are not inherited
        outright still call the inherited one; they add the single thing that genuinely
        differs between the two runs, and `tests/test_preflight.py` refuses any override
        that does not delegate.
      * It is NOT SWITCHING. Reusing that state would have made a preflight
        indistinguishable from a switch in the one artifact this failure was diagnosed
        from - counting SWITCHING frames in the trace is precisely what proved no switch
        ever happened - and would have handed it `Timings.switch_timeout` (240s), the
        switch failure/backoff bookkeeping in `Runner._count_transition`, and the
        "account switch never confirmed" wording, none of which describes a startup check.
      * The import-time contract is satisfied the ordinary way: its own numeric
        `timeout_s`, its own `on_timeout`.

    Bounded twice, and never able to stop the bot playing. `Timings.preflight_timeout`
    bounds the state, and every exit - the ladder finishing, the AutoWalk wall clock
    (`config.AutoWalk.budget_s`) running out, a star that will not separate, no view tree
    to find one in, a closable overlay arriving over the two blind phases, the state
    timeout itself - transitions to SCANNING and plays. The only thing a failure changes
    is what the log says. That budget also sits inside `Timings.stuck_watchdog` (120s), so
    unlike a switch this needs none of the watchdog credit `Context.switch_exit_ts` grants
    SWITCHING: a preflight cannot by itself look like a wedged run.

    It runs ONCE per game start - not once per process, because RECOVERING's last resort
    force-stops and relaunches the game, and a cold relaunch undoes all three of the very
    things this sets. `Runner._maybe_preflight` owns that (a latch, cleared again only by
    an accepted `effects.RestartApp`), for the same reason `_maybe_switch` owns when a
    switch may start: entering a state is the runner's decision, and hanging it off `Boot`
    alone would lose the feature on any startup that took longer than BOOT's 30s budget
    and reached the map through RECOVERING instead - a cold start after `RestartApp` is
    measured in tens of seconds (see `Timings.app_restart_grace`).
    """

    state = BotState.PREFLIGHT
    #: Declared so the import-time contract below still sees a numeric budget, and kept
    #: equal to the config default so the two can never disagree - the same shape, and the
    #: same reason, as `Switching.timeout_s`.
    timeout_s = Timings().preflight_timeout

    def timeout(self, ctx):
        return ctx.cfg.timings.preflight_timeout

    def step(self, obs, ctx):
        if ctx.switch_phase not in PREFLIGHT_PHASES:
            # `Switching.step`'s fall-through is the "open" behaviour: it drives the
            # PGSharp overlay toward a login button for `switch_target`, which a preflight
            # does not have. Refusing outright means no phase this state was never given
            # can ever reach that code and tap the account panel at a target of None.
            return [Transition(BotState.SCANNING, IntentOutcome.CARRIED,
                               f"preflight has no {ctx.switch_phase!r} phase to run")]
        if obs.in_overlay and ctx.switch_phase in _PREFLIGHT_MAP_PHASES:
            # Hand the screen back rather than sit on it. `desired_state` lets this state
            # own the screen (post-login and PGSharp's own windows must not be chased),
            # and its POPUP branch only fires from SCANNING/TARGETING - so while a
            # preflight holds a closable overlay NOTHING closes it, and `_zoom`/`_goplus`
            # cannot advance either, since both refuse to fire their blind fixed-coordinate
            # gesture without `obs.on_map`. Measured through the real Runner with a
            # closable modal arriving one tick in: 4 close attempts over 100s of simulated
            # clock - none of them until the 90s budget expired - against 36 with this
            # yield in place, the first of them a second after the modal appeared.
            #
            # Only these two phases. The autowalk ones act on coordinates read from the
            # live tree, under PGSharp's own menu and AlertDialog, which is exactly the
            # kind of screen that can read as an overlay while the ladder still has to act
            # on it (see `_autowalk_open`'s docstring) - yielding there would abandon a
            # half-driven menu over the map, which is the failure `_autowalk_close` exists
            # to prevent. `in_overlay` and not merely `not on_map`: it demands a smoothed
            # X button AND no map AND no encounter, so a single misread frame during the
            # zoom drag cannot end the preflight, while the modal this is for does.
            #
            # The preflight is not resumed afterwards: the latch in
            # `Runner._maybe_preflight` has already been set. Best-effort is the whole
            # contract of this state - it may never be the reason the bot stops playing.
            return [
                Note(f"a closable overlay came up during the startup preflight's "
                     f"{ctx.switch_phase} step; handing the screen back so it can be "
                     f"closed - {_PREFLIGHT_UNDONE[ctx.switch_phase]}", "warn"),
                Transition(BotState.SCANNING, IntentOutcome.CARRIED,
                           "startup preflight yielding to a closable overlay"),
            ]
        return super().step(obs, ctx)

    def _autowalk_open(self, obs, ctx):
        """The first AutoWalk rung, refused outright when this run has no view tree.

        Every autowalk phase locates its widget - the star, the menu row, the dialog
        button - in the live uiautomator tree, and `cli.prepare_accounts` hands `Runner` a
        reader ONLY when a switch trigger is armed, which is not the default. Without one
        `ctx.accounts` is None forever, so the inherited ladder can do nothing but wait out
        `config.AutoWalk.budget_s`: measured on a controlled clock, two zoom drags and then
        30.4s of empty ticks. That is not merely idle - `desired_state` gives this state
        the screen, so for those 30s an encounter, a Rocket screen or a popup is ignored,
        and `_goplus` may just have switched Virtual Go Plus on, which is what produces
        encounters. Ending here instead costs nothing that was reachable anyway; the runner
        has already said why, once, in `_maybe_preflight`.
        """
        if not ctx.tree_available:
            return [Transition(BotState.SCANNING, IntentOutcome.CONFIRMED,
                               self._finished(ctx, "AutoWalk needs the PGSharp view "
                                                   "tree, which this run has no reader "
                                                   "for"))]
        return super()._autowalk_open(obs, ctx)

    def _autowalk_deadline(self, ctx):
        """The inherited give-up, said out loud.

        "AutoWalk is not running" is one of the three symptoms the user actually reported,
        and every likely way to reach it here - a star that cannot be located, a menu that
        never renders - leaves through this method as a single INFO state-transition line.
        The same change deliberately made a failed identification a WARNING; this is the
        same silence in the same run. `Switching` is left alone: a switch that got as far
        as this has already logged the login it confirmed.
        """
        gone = super()._autowalk_deadline(ctx)
        if gone is None or not any(isinstance(e, Transition) for e in gone):
            return gone            # still inside the ladder's own cleanup allowance
        return [Note(f"the startup preflight could not start AutoWalk within "
                     f"{ctx.cfg.autowalk.budget_s:.0f}s at the {ctx.switch_phase} step - "
                     f"playing anyway, with no route running", "warn")] + gone

    def on_timeout(self, obs, ctx):
        undone = _PREFLIGHT_UNDONE.get(ctx.switch_phase,
                                       "the startup checks did not all happen this run")
        return [
            Note(f"the startup preflight ran out of its {self.timeout(ctx):.0f}s budget "
                 f"at the {ctx.switch_phase} step - playing anyway: {undone}", "warn"),
            # SCANNING, never RECOVERING: nothing here is evidence that the bot is stuck.
            # The screen the preflight was driving is the map (zoom/goplus both wait for
            # it) or PGSharp's own menu over it, and SCANNING's own map check plus the
            # recovery ladder already handle whatever is actually in front of us. Sending
            # a merely-incomplete startup check to RECOVERING would spend a recovery on a
            # bot that is perfectly able to play.
            Transition(BotState.SCANNING, IntentOutcome.CARRIED, "preflight timed out"),
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
    v1 menu loop. The PGSharp panel's own close control, located in the view tree, first;
    then BACK; then an optically located close button; then, once the map has been gone
    past the stuck watchdog, a bounded number of app restarts; and only then halt. A
    restart pauses the whole ladder for as long as the relaunch is cold-starting (see
    `_restarting`) - the one screen where pressing anything is worse than pressing
    nothing."""

    state = BotState.RECOVERING
    timeout_s = 6.0

    def step(self, obs, ctx):
        if obs.on_map:
            return [Transition(BotState.SCANNING, IntentOutcome.CARRIED, "recovered to map")]
        # The restart grace is a HOLD on the whole ladder, not merely a delay on the next
        # restart decision in `on_timeout`. `on_timeout` hands control back to SCANNING,
        # which sees no map and returns here within a tick, so every ~6s of the 90s window
        # is another visit with `taps_in_state` freshly zeroed - roughly fifteen of them,
        # each spending a BACK, into an app that is cold-starting. That is the press
        # `config.Timings.switch_clear_max` exists to bound: ~90 BACKs into a screen that
        # would not clear left PGSharp's panel showing NEITHER account with an asterisk,
        # the game logged out of both profiles and recoverable only by hand. The located
        # rung is no safer here - `ctx.accounts` was read before the process was killed,
        # so its `close_norm` names a window that no longer exists. Nothing on screen
        # during a cold start is ours to press, so nothing is pressed.
        if self._restarting(ctx):
            return []
        panel = self._panel_close(ctx)
        if panel is not None:
            return panel
        if ctx.taps_in_state == 0 and ctx.ready("back", 1.0):
            return [Back("recover: dismiss with BACK")]
        if obs.close_button_xy is not None and ctx.ready("close", ctx.cfg.timings.close_menu):
            return [Tap(*obs.close_button_xy, "recover: located close button", budget="close")]
        return []

    @staticmethod
    def _restarting(ctx):
        """True while a restart this handler asked for is still cold-starting the game.

        One predicate, two consumers - `step` holds the ladder, `on_timeout` holds the
        next restart decision - because two copies of a window this consequential would
        eventually disagree. `app_restart_ts` is 0.0 until `Runner.apply` sees the
        actuator ACCEPT a RestartApp, so a restart nobody performed never opens a window.
        """
        return bool(ctx.app_restart_ts) and \
            ctx.now - ctx.app_restart_ts <= ctx.cfg.timings.app_restart_grace

    def _panel_close(self, ctx):
        """Tap the PGSharp panel's OWN close control when the view tree says that panel is
        what is in front of us.

        `None` means the tree is making no such claim, and the ladder below runs unchanged.
        An EMPTY list is the third answer, and distinct on purpose: the tree DOES say the
        panel is up, but this rung's own pacing withholds the tap this tick. That returns
        nothing rather than falling through, because while a PGSharp panel is known to be
        in front of us the rungs below must not run at all - BACK is the press that logged
        the account out (below), and the optical locator has nothing to find on this
        screen.

        Placed ABOVE the BACK rung, which is the only ordering the measurements support.
        The screen this exists for is the PGSharp accounts panel - a native Android
        overlay listing accounts, drawn over the game. Read at the bot's own resolution it
        reports screen=Menu@0.95, on_map=False, in_overlay=False, x_button=False,
        find_close_button=None, detections=[] - so BOTH rungs below are blind to it: the
        optical locator has nothing to find, and BACK simply does not dismiss it. Measured
        in logs/trace.jsonl, that is 599 of one run's 1520 frames (39%) spent in a
        RECOVERING x47 -> SCANNING x1 cycle at screen=Menu@0.941, one BACK per cycle,
        never recovering.

        Those BACKs are not merely wasted. An unbounded run of them is what
        `config.Timings.switch_clear_max` exists to stop: ~90 presses into a screen that
        would not clear left PGSharp's panel showing NEITHER account with an asterisk -
        the game logged out of both profiles and had to be recovered by hand. So spending
        a coordinate-free guess before a located coordinate is not a neutral first try; it
        is the more dangerous of the two, and it goes second.

        This still honours "buttons are LOCATED, never assumed". `close_norm` is some
        node's own bounds read out of the live view tree (see accounts.py) - exactly the
        source, and exactly the field, `Switching._verify` already taps to close this same
        panel. What it is not is instantaneous: `Runner._refresh_accounts` reads the tree
        on its own throttle (runner.RECOVER_ACCOUNTS_REFRESH), so the view can be up to
        that old, and a panel that closed in between is described here as still open. One
        tap per RECOVERING visit is the bound on what that costs, which is what
        `taps_in_state == 0` buys - the same gate, and the same "one blind action per
        visit" budget, the BACK rung below already uses. The stale tap lands on the
        panel's own close control, the least harmful coordinate of the set, and the next
        refresh corrects the view.
        """
        v = ctx.accounts
        if v is None or not v.available:
            return None
        # Two different PGSharp pages can be in front of us, with different close controls
        # and different ids. The accounts panel is the one this rung was written for. The
        # SETTINGS page is the other, and it wedged a live run the same way: its close
        # control is `hl_st_close` ("OK", top-left), so `close_norm` was None,
        # `panel_open` was False, and this returned None while the optical locator found
        # nothing and BACK did not dismiss it either. Its coordinate is kept in its own
        # field rather than folded into `close_norm`, because `panel_open`/`close_norm`
        # mean "the accounts panel" to `Switching`, which steers by them.
        target = (v.close_norm if v.panel_open and v.close_norm is not None
                  else v.settings_close_norm)
        if target is None:
            return None
        if ctx.taps_in_state != 0 or not ctx.ready("close", ctx.cfg.timings.close_menu):
            return []
        return [Tap(*target,
                    "recover: close the PGSharp panel located in the view tree",
                    budget="close")]

    def on_timeout(self, obs, ctx):
        # `ctx.last_map_ts` alone regressed here: an account switch legitimately owns the
        # screen for up to Timings.switch_timeout (240s), comfortably longer than
        # stuck_watchdog (120s), so `last_map_ts` goes stale while SWITCHING drives the
        # PGSharp overlay whether the switch succeeds or fails. A failed switch reliably
        # tripped this the moment it handed off to RECOVERING - measured on the device,
        # HALTED "no confirmed map for 209s" six seconds after a switch that never
        # confirmed, ending a run over one ordinary, already-handled failure (SWITCHING's
        # own timeout plus the backoff and three-strike give-up in runner.py).
        #
        # `switch_exit_ts` (set by runner._count_transition, never here - handlers cannot
        # write ctx) marks the instant control last legitimately returned from a switch,
        # successful or failed. `ctx.map_stale_since` takes the LATER of it and
        # `last_map_ts`, so a switch's own bounded duration is never by itself sufficient
        # to trip this halt, while stuckness that begins - or continues - AFTER a switch
        # ends is still caught at exactly the same 120s this watchdog has always used:
        # with no switch involved, `switch_exit_ts` stays at its 0.0 default and this is
        # exactly the old check.
        cfg = ctx.cfg
        stale_since = ctx.map_stale_since
        stale_for = ctx.now - stale_since
        if stale_for <= cfg.timings.stuck_watchdog:
            return [Transition(BotState.SCANNING, IntentOutcome.CARRIED,
                               "recovery attempt over")]

        # Past the watchdog, halting is no longer the only thing left. Every rung of
        # `step` acts on the SCREEN, and the failure that produced this ladder is a screen
        # nothing on it can reach - measured, a PGSharp overlay window belonging to the
        # game's own process (`mCurrentFocus` stays the Unity activity while the panel is
        # up), so ending that process ends the panel. Restarting is a bigger hammer than
        # anything above: it kills whatever the game had in flight, which is why it is
        # only ever reached here, after the whole ladder has failed for longer than the
        # watchdog, and never as an ordinary recovery step.
        #
        # A restart is a COLD start, and for the tens of seconds it takes the map is still
        # missing - which is the exact condition that authorised the restart. Without the
        # grace, the next two RECOVERING timeouts (6s each) would read a loading app as a
        # failed restart and spend the whole budget in a few seconds; see
        # `Timings.app_restart_grace`. Handled as a wait, not as a shorter watchdog,
        # because `map_stale_since` legitimately keeps growing throughout: what changed is
        # that we know why. `step` holds the ladder for the same window (see
        # `_restarting`), so the visits this hand-off produces press nothing.
        if self._restarting(ctx):
            return [Transition(BotState.SCANNING, IntentOutcome.CARRIED,
                               "waiting out the app restart")]

        if ctx.app_restarts < cfg.max_app_restarts:
            # The Transition is part of the same answer, not an afterthought: `on_timeout`
            # fires on every tick where `elapsed` exceeds the budget, and only entering a
            # state restarts that clock (`Runner.enter_state`). Returning the restart
            # alone would re-emit it every tick until the grace stamp landed. `Runner
            # .apply` walks the list in order, so the restart is dispatched while the
            # state is still RECOVERING and the hand-off follows it - the same ordering
            # `Switching._autowalk_close` relies on.
            return [
                Note(f"no confirmed map for {stale_for:.0f}s and nothing on screen "
                     f"answered; restarting {cfg.app_package} "
                     f"({ctx.app_restarts + 1}/{cfg.max_app_restarts})", "warn"),
                RestartApp(cfg.app_package, cfg.app_activity,
                           f"recover: restart the app after {stale_for:.0f}s with no map"),
                Transition(BotState.SCANNING, IntentOutcome.CARRIED,
                           "restarted the app; waiting for it to load"),
            ]

        # Reached only once the count EQUALS the budget, so it reports what is spent, not
        # what is left: `ctx.app_restarts` is the tally of restarts taken, and printing it
        # next to "left to spend" told an operator the escalation had never fired at the
        # one moment they read the line. Same standard as
        # test_fsm_livelocks.test_halt_message_reports_the_reason_the_gate_actually_used.
        return [Halt(f"no confirmed map for {stale_for:.0f}s, and all "
                     f"{cfg.max_app_restarts} app restart(s) spent; stopping rather than "
                     f"tapping blindly")]


class Halted(Handler):
    state = BotState.HALTED
    timeout_s = 1e9

    def step(self, obs, ctx):
        return []


HANDLERS = {h.state: h() for h in
            (Boot, Scanning, Targeting, Encounter, Pokestop, Rocket, Switching, Preflight,
             Popup, Recovering, Halted)}

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
    """At most one fires per tick. Interrupts act but never change state.

    Checked first and unconditionally on state: Pokemon GO's own exit-confirmation
    dialog (obs.exit_dialog - see perception.exit_dialog_signal) can appear while
    RECOVERING is unwinding whatever put it there, and needs the fastest, most direct
    response available regardless of what state that happens to be - not routed through
    `desired_state`/a handler's own `step`, both of which run only after this. BACK is
    the whole point: it carries no coordinate at all, where this dialog's own OK button
    sits close enough to ROCKET's fixed dialogue-advance tap (see `rocket_screen`) that a
    coordinate-based response risks quitting the game outright. Given that asymmetry, a
    threshold measured on only two positive samples (see config.Thresholds) is an
    acceptable trade: the worst a false positive costs is one extra BACK press.
    """
    cfg = ctx.cfg
    if obs.exit_dialog.value and ctx.ready("back", cfg.timings.exit_dialog_back):
        return [Back("dismiss Pokemon GO's own exit-confirmation dialog"),
                Note("exit-confirmation dialog detected; pressing BACK rather than "
                     "risking a coordinate tap near its OK button", "warn")]
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
    if ctx.state in (BotState.SWITCHING, BotState.PREFLIGHT):
        # A switch owns the screen until it confirms or times out. Post-login screens
        # look like Rocket and like encounters; following them abandons the switch
        # half-done, logged into neither account cleanly.
        #
        # PREFLIGHT owns it on the same terms and for the same reason: it drives the very
        # same PGSharp overlay (see `Preflight`), whose shortcut menu and AutoWalk
        # AlertDialog are exactly the kind of screen that reads as an overlay, or as
        # not-on-map, while the ladder still needs to act on it. Both are bounded by their
        # own state timeout, which is what keeps "owns the screen" from meaning "forever";
        # PREFLIGHT additionally hands it straight back from its two blind phases when
        # something closable turns up, since those two need the bare map and nothing else
        # would ever close it (see `Preflight.step`).
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
    # ROCKET joins these two only once the fight is demonstrably over. A Rocket screen
    # carries its own X, which is why POPUP must never outrank ROCKET while one is up -
    # closing the grunt dialogue instead of fighting it is the failure the ordering at the
    # top of this function exists to prevent. But the fight ENDS on screens that are not
    # Rocket screens and that `Rocket.step` has nothing to say about: it taps an
    # affirmative pill if one is present, advances dialogue while the classifier still
    # reads Rocket, and otherwise returns nothing at all. The reward and "NEW LEVEL
    # UNLOCKS" screens that follow a fight are neither, so with ROCKET missing from this
    # branch a located X sat unpressed until the 150s handler timeout expired.
    #
    # Measured live, one episode: 1188 frames - 148s - in ROCKET on screen=Menu@0.857 with
    # `close=True` the whole way, emitting not one effect, before `Rocket.on_timeout`
    # handed it to RECOVERING, which pressed the very button that had been located all
    # along. A live fight cannot reach this branch at all: the rocket-hold above returns
    # None while `rocket_recent` and off-map, and an on-map frame is taken by the branch
    # before this one. The `not rocket_recent` term is therefore belt-and-braces -
    # deleting it alone leaves the whole suite green - and is kept only so the
    # precondition is stated where the branch is read, and so loosening that hold cannot
    # silently turn this into a preemption of a fight in progress.
    if obs.in_overlay and (ctx.state in (BotState.SCANNING, BotState.TARGETING)
                           or (ctx.state is BotState.ROCKET and not rocket_recent)):
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
