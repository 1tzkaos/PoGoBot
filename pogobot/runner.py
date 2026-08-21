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
from . import perception
from .config import Config
from .effects import (
    Back,
    BotState,
    ClearSpatialMemory,
    Cooldown,
    Pinch,
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
    is_actuation,
)
from .frames import Frame, FrameSource
from .observation import Observation, Tristate
from . import profiles
from .quota import SpinQuota
from .stats import SessionStats, append_session

from .quota import _hms

log = logging.getLogger("pogobot")

# Preview refresh rate. The inference loop runs slower than this; redisplaying the cached
# HUD in between keeps the window responsive to the q key without re-rendering.
DISPLAY_FPS = 30.0

# How often a headless run logs its counters, so a long session reports progress.
REPORT_EVERY = 300.0

# How much of the tail of an encounter to keep for labelling. The catch sequence -
# ball wobble, "Gotcha!", then the XP/candy/stardust award - runs several seconds.
ENCOUNTER_RING_SECONDS = 6.0

# How often the account list is re-read while a switch is in flight. The dump blocks for
# roughly a second, so this is as often as the loop can afford; it is also why the read
# happens during a switch and nowhere else.
ACCOUNTS_REFRESH = 2.5

# How often the same dump is re-read while RECOVERING, which is the other state that can
# be looking at the PGSharp panel - see fsm.Recovering._panel_close for what reads it and
# why the panel is invisible to every optical signal the bot has.
#
# Four times the switch cadence above, because nothing here is racing a budget. A switch
# has to catch an asterisk inside `Timings.switch_timeout` (240s) and re-reads as often as
# it can afford; RECOVERING's own deadline is the 120s stuck watchdog and the restart
# ladder below it, both of which have room for a dozen reads. Against that, the dump
# blocks this very thread for ~1s - and for up to `accounts.UiTreeReader.timeout` when the
# window will not go idle, which is exactly what a rendering game does - and RECOVERING is
# not only the wedged case: it is entered briefly after every encounter, popup or stop
# that times out, so a switch-rate refresh would buy a stall on each of those to answer a
# question that is almost always "no panel". The stall is bounded twice over: by that
# reader timeout, and by `_refresh_accounts` stamping this throttle from when the read
# FINISHED, so a slow read is always followed by a full interval of live frames.
#
# 10s is also the age the view fed to `_panel_close` can reach, and that bound is the
# reason it is not longer: the observed livelock cycles RECOVERING -> SCANNING about every
# 7s (47 RECOVERING frames at ~8fps plus a single SCANNING one), so a panel that closes is
# described as still open for at most one or two more visits before the next read corrects
# it. The stall only ever lands on a bot that is already not playing: a second lost in
# RECOVERING delays the return to the map by a second, and nothing else.
RECOVER_ACCOUNTS_REFRESH = 10.0

# A switch that expires without confirming is not retried immediately.
#
# Observed live: the login tap is accepted, PGSharp closes its own panel, and the account
# simply does not change - the suspected cause is a login throttle after several switches
# within a few hours. None of that is visible to the stuck watchdog, which only asks
# whether the map is up, and in this failure mode it IS up. So without a record of the
# failure nothing stopped the next tick from starting the same attempt over: six attempts
# in six cycles when it was driven through the real Runner, and a bot that catches nothing
# for the rest of the run because it spends every couple of minutes driving an overlay.
#
# The first wait is 10 minutes. A HEALTHY switch was measured at up to ~2 minutes end to
# end, so anything much shorter is barely longer than one attempt, and retrying an
# external throttle faster than the thing that is throttling us cannot work by
# construction. It doubles per consecutive failure so a throttle that outlives one wait is
# not hammered by the next. Nothing is lost by waiting: the cap blocks stops, not catches,
# so the bot keeps playing on the account it already has.
SWITCH_BACKOFF_BASE = 600.0

# Consecutive failures before the bot stops trying at all and says so. Three attempts
# spread over half an hour that all ended the same way is evidence that the problem is not
# the timing, and each further attempt costs a real screen - up to `switch_timeout` of not
# catching anything - to re-test a hypothesis that has already been refuted twice.
SWITCH_MAX_FAILURES = 3

#: "no profile applied yet", distinct from an account of None (which is a real state: a
#: session that never learned its name still gets the `default` entry).
_UNSET = object()

# States whose per-visit bookkeeping must reset on entry.
_RESET_ON_ENTRY = ("spun_disc", "taps_in_state", "switch_zoom_reps", "switch_goplus_attempts",
                   "switch_clear_presses", "star_drags")


class Runner:
    def __init__(self, cfg: Config, source: FrameSource, actuator, perceptor,
                 ledger=None, keyboard=None, trace_path: Optional[Path] = None,
                 display: bool = True, stats_path: Optional[Path] = None,
                 dashboard=None, encounter_dump: Optional[Path] = None,
                 dialogue_dump: Optional[Path] = None,
                 quota: Optional[SpinQuota] = None,
                 pause_file: Optional[Path] = None, tree_reader=None,
                 roster: tuple[str, ...] = (),
                 account_profiles: Optional[dict] = None):
        self.cfg = cfg
        #: The run's own settings, before any per-account override. Overrides are applied
        #: ON TOP of this rather than on top of each other, so switching A -> B -> A gives
        #: A exactly what it had the first time instead of accumulating B's answers.
        self._base_cfg = cfg
        self.account_profiles = dict(account_profiles or {})
        #: Which account `self.cfg` currently reflects. Deliberately a sentinel rather than
        #: None, because None is a real account state ("we never learned who we are") and
        #: must still get the defaults applied once.
        self._profile_account: object = _UNSET
        self.source = source
        self.actuator = actuator
        self.perceptor = perceptor
        self.ledger = ledger
        self.keyboard = keyboard
        self.display = display
        self.stats_path = stats_path
        self.dashboard = dashboard
        self.encounter_dump = encounter_dump
        self.dialogue_dump = dialogue_dump
        self.quota = quota
        self.pause_file = pause_file
        self.tree_reader = tree_reader
        # The accounts that exist, enumerated once at startup with the panel open. It is
        # not refreshed because it cannot be: outside a switch the panel is shut and the
        # tree lists no rows at all, so a live read would only ever shrink this to nothing.
        self.roster = tuple(roster)
        self._accounts_read_at = 0.0
        #: whether the startup preflight has already been started for the game currently
        #: running (see `_maybe_preflight`). A latch, not a counter: it is set the first
        #: time the attempt is made, so a preflight that could not finish is still not
        #: repeated - PREFLIGHT hands control back to SCANNING either way, and re-entering
        #: it every time the bot returned to SCANNING would be a livelock, not a retry.
        #: Cleared again only by an accepted `effects.RestartApp` (see `apply`), because a
        #: cold relaunch undoes all three of the things the preflight sets.
        self._preflight_done = False
        self._switch_target: Optional[str] = None
        #: consecutive switch attempts that expired without confirming, and the FSM-clock
        #: instant before which no new attempt may start. Reset by a confirmed switch.
        self._switch_failures = 0
        self._switch_blocked_until = 0.0
        #: the account the tree named as active during the CURRENT attempt, if any read
        #: managed to name one. Cleared when an attempt starts, so an observation from one
        #: attempt can never be spent on the conclusions of another.
        self._last_seen_active: Optional[str] = None
        self._paused = False
        self._pause_requested = False    # toggled by SIGUSR1 and by the display key
        self._paused_at = 0.0
        self._pause_total = 0.0
        # A ring of frames from inside an encounter. On exit these are the frames that
        # would show a catch award screen - the evidence a real catch counter needs.
        # Sized in seconds, not frames: at 8 inference fps a 8-frame ring held one second
        # and rolled the award sequence away before the encounter ended.
        self._enc_ring: deque = deque(maxlen=max(8, int(cfg.infer_fps * ENCOUNTER_RING_SECONDS)))
        self.ctx = fsm.Context(cfg=cfg, state=BotState.BOOT,
                               state_since=time.perf_counter(), now=time.perf_counter())
        self.ctx.last_map_ts = time.perf_counter()
        # A standing fact about the run, so a pure handler can tell "no view this tick"
        # from "there will never be a view" - see `fsm.Context.tree_available` and
        # `fsm.Preflight._autowalk_open`, the one place the difference is worth 30s.
        self.ctx.tree_available = tree_reader is not None
        # One full interval in, not on the first tick: `--switch-every 45` means every 45
        # minutes, and rotating out of a fresh account immediately is nobody's intent.
        self._next_rotation = (self.ctx.now + cfg.switch_every_minutes * 60.0
                               if cfg.switch_every_minutes > 0 else 0.0)
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
        self._real = time.perf_counter()   # loop's real-clock sample; paces display + fps
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
            self.ctx.throws_this_encounter = 0
            # It is the same encounter only if the map was never confirmed in between: a
            # recovery that actually worked lands on the map, and a genuinely new encounter
            # can only be reached through it.
            # A NEW encounter is only reachable through the map, so any re-entry with no
            # confirmed map since we left is the same Pokemon - however we got back.
            # Requiring src is RECOVERING was too narrow: with the encounter hold in
            # place the round trip goes through SCANNING instead, and one stuck screen
            # counted as five encounters.
            resumed = (self._encounter_left_at is not None
                       and self.ctx.last_map_ts <= self._encounter_left_at)
            if not resumed:
                self._encounter_left_at = None
                st.on_encounter_start()
        elif src is BotState.ENCOUNTER and dst is not BotState.ENCOUNTER:
            st.on_encounter_end()
            self.ctx.left_encounter_ts = self.ctx.now
            # Remember where we left so a re-entry with no map in between is recognised
            # as the same encounter rather than a new one.
            self._encounter_left_at = self.ctx.now
            self._end_encounter(e)
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
                if self.quota is not None:
                    # The cap belongs to the account, so the spin is booked against the
                    # one that earned it. None normalizes to the unknown-account bucket.
                    self.quota.record(account=self.stats.account)
            elif e.outcome is IntentOutcome.REFUTED:
                st.stops_out_of_range += 1
                self._explain_refusal()
        if e.outcome is IntentOutcome.EXPIRED \
                and src in (BotState.TARGETING, BotState.POKESTOP):
            # TARGETING and POKESTOP are both post-tap wait states, so an expiry in either
            # is one target tap that never produced the screen it claimed. Counting only
            # TARGETING silently dropped every stop tap whose POI screen never opened.
            st.taps_expired += 1
        if src is BotState.SWITCHING and dst is not BotState.SWITCHING:
            # A switch attempt - successful or failed - legitimately occupies the screen
            # for up to `switch_timeout` (240s), well past `stuck_watchdog` (120s), and
            # `last_map_ts` goes stale while SWITCHING drives the PGSharp overlay. Left
            # alone, RECOVERING's own watchdog check reads that staleness as genuine
            # stuckness - measured on the device, HALTED "no confirmed map for 209s" six
            # seconds after a switch that never confirmed. `last_map_ts` itself is left
            # untouched here: Scanning's own popup-timeout check and the
            # encounter-resumed test (below) both depend on it meaning "the map was
            # actually seen". `switch_exit_ts` instead marks the instant control
            # legitimately returned to the FSM, and only `Recovering.on_timeout`
            # consults it - so a switch's own bounded, already-handled failure (backoff +
            # three-strike give-up below) can never by itself trip a halt, while
            # stuckness that begins after a switch ends is still caught at the exact same
            # 120s the watchdog has always used for every other kind of stuckness.
            self.ctx.switch_exit_ts = self.ctx.now
        if src is BotState.SWITCHING and dst is BotState.SCANNING \
                and e.outcome is IntentOutcome.CONFIRMED and self._switch_target:
            # Only a CONFIRMED switch rolls the session over: that outcome means the tree
            # named the target as active AND the map came back. The timeout leaves through
            # RECOVERING as EXPIRED, and an attempt that never landed must not split the
            # books or reset a counter. Last in this method because it REPLACES self.stats,
            # so every count above still lands on the outgoing account.
            self._on_switch_confirmed(self._switch_target)
            self._switch_target = None
        elif src is BotState.SWITCHING and e.outcome is IntentOutcome.EXPIRED:
            # The only other way out of SWITCHING (`Switching.on_timeout`). Recorded here,
            # in the one place transitions are already inspected, because nothing else in
            # the loop can tell a switch that failed from one that was never started.
            self._on_switch_failed(self._switch_target)
            self._switch_target = None

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

    def _frames_starved(self, now: float) -> bool:
        """True once a run of unusable frames has outlasted the stuck watchdog.

        This is a second, independent consumer of `stuck_watchdog` alongside
        `Recovering.on_timeout` - it exists because `fsm.step` (and so every FSM-level
        timeout, including SWITCHING's own `switch_timeout`) can only run once a real
        frame has been read, so a stretch of `frame is None` needs its own backstop.

        A switch is given a LONGER budget here, not excused from the check. The ordinary
        `map_stale_since` accounting cannot judge a switch in flight: SWITCHING drives the
        PGSharp overlay over the map for the whole attempt, so `last_map_ts` is already
        >120s stale on a perfectly on-budget switch, and `switch_exit_ts` still holds
        whatever an earlier switch (or none) left it at. Unguarded, one ordinary transient
        frame-read gap partway through such a switch halted the run outright - exactly the
        failure class `ctx.map_stale_since` was added to stop, reached by a path
        `Recovering.on_timeout` cannot cover, since that only ever runs once SWITCHING has
        already exited.

        Returning False outright was the overcorrection. "It will be caught once the
        switch exits" is not an escape hatch that exists here: SWITCHING is left only
        through `fsm.step`, and the loop reaches `fsm.step` only after a frame has come
        back - so when the thing that has failed IS the frames, that exit never arrives. A
        source that is alive but silent (scrcpy holding the pipe open with the screen off,
        a flaked USB, a sleeping device - see `capture.release`) keeps `healthy()` true and
        the reader thread quiet, so the run loop spun on forever with no halt, no log line
        and no session record, where before it stopped cleanly at 120s. Nothing is tapped
        meanwhile, so it is not unsafe - it is a clean halt replaced by a silent hang.

        `switch_timeout` is the whole budget a switch is entitled to, and `stuck_watchdog`
        on top is the same grace any other state gets before staleness counts as stuckness;
        past their sum there is no reading of the frames that is still consistent with a
        switch merely being slow. `state_since` is the switch's own start (nothing
        re-enters SWITCHING from itself - `_maybe_switch` starts only from SCANNING), so
        this is that switch's clock, not a rolling one.
        """
        if self.ctx.state is BotState.SWITCHING:
            return now - self.ctx.state_since > (self.cfg.timings.switch_timeout
                                                 + self.cfg.timings.stuck_watchdog)
        return now - self.ctx.map_stale_since > self.cfg.timings.stuck_watchdog

    def toggle_pause(self) -> None:
        self._pause_requested = not self._pause_requested

    def _pause_wanted(self) -> bool:
        if self._pause_requested:
            return True
        if self.pause_file is not None:
            try:
                return self.pause_file.exists()
            except OSError:
                return False
        return False

    def _abandon_intent(self, reason: str) -> None:
        """Give up any tap whose answer we will not be there to see.

        `reason` is the caller's, because two things take the screen away from a pending
        tap - a pause and an account switch - and a log line that names the wrong one sends
        whoever reads it looking for a pause that never happened.

        An Intent is a causal claim - "the screen changed BECAUSE of this tap" - and the
        ledger writes a training sample on the strength of it. Freezing the FSM clock keeps
        that claim alive across a pause while making it uncheckable: the latency the ledger
        tests against `causal_max_s` is measured on the frozen clock, so a tap answered ten
        real minutes later is recorded as answered in a fraction of a second. Measured -
        real Runner, real IntentLedger - one corpus row written with `latency: 0.3` for a
        real gap of 601.8s, straight past the 5s causal window that exists to make exactly
        that row impossible, and `verified: false` means a human reads that latency as
        evidence. The same stale claim would score a CONFIRMED against a screen the pause
        invited a human to open by hand.

        So the claim dies with the pause. EXPIRED is what we can actually support - the tap
        got no answer we watched - and it costs only the short on_expired cooldown; the
        ledger rejects a non-CONFIRMED/REFUTED outcome, so nothing reaches the corpus.
        """
        if self.ctx.intent is None:
            return
        log.info("abandoning the pending %s tap: %s",
                 self.ctx.intent.target_name, reason)
        self._resolve_intent(self.ctx.intent, IntentOutcome.EXPIRED)
        self.ctx.intent = None

    def _sync_pause(self, real: Optional[float] = None) -> bool:
        """Enter or leave the paused state. Returns whether we are paused now.

        `real` is the loop's single `perf_counter()` sample. Taking it as an argument keeps
        the pause accounting and `ctx.now` on the *same* instant; sampling the clock twice
        let `ctx.now` creep forward by the gap between the two samples on every iteration.

        The FSM clock is frozen rather than the timers being shifted one by one: `ctx.now`
        is driven from `perf_counter() - _pause_total`, so every stored deadline stays
        comparable and nothing has to know that a pause happened. Shifting each timer
        instead would mean a resume fires every timeout at once - a pause that ends in a
        recovery storm is worse than no pause at all.

        This freeze applies to the FSM clock ONLY. Pacing - the inference schedule, the
        display refresh, the fps window - must stay on the real clock: a frozen `now` never
        reaches its own `next_infer`, so driving the schedule from it stops the loop dead.
        """
        real = time.perf_counter() if real is None else real
        want = self._pause_wanted()      # one stat() of the pause file per iteration
        if want and not self._paused:
            self._paused = True
            self._paused_at = real
            self._abandon_intent("paused before the screen answered")
            log.warning("PAUSED - no taps will be sent. %s",
                        "delete the pause file to resume" if self.pause_file
                        else "send SIGUSR1 again to resume")
        elif not want and self._paused:
            self._paused = False
            self._pause_total += real - self._paused_at
            self.stats.paused_seconds = self._pause_total
            log.info("resumed after %.0fs paused", real - self._paused_at)
        if self._paused:
            self.stats.paused_seconds = self._pause_total + (real - self._paused_at)
        return self._paused

    @property
    def paused(self) -> bool:
        return self._paused

    def _explain_refusal(self) -> None:
        """A refused stop past the daily cap is not a distance problem.

        On screen the two are identical - the same "walk closer to interact" banner - and
        that ambiguity produced a wrong diagnosis once already: 152 refused stops in one
        session were read as bad positioning when the account had spun out for the day.
        """
        if self.quota is None:
            return
        st = self.quota.state(account=self.stats.account)
        if st.exhausted and self.stats.stops_out_of_range % 10 == 1:
            log.warning("stop refused and the 24h spin quota is used up (%d/%d) - this is "
                        "the cap, not distance. Resets in %s.",
                        st.used, st.limit, _hms(st.resets_in))

    def _update_spins_exhausted(self) -> None:
        """Derive the FSM's quota flag from THIS account's rolling 24h window.

        A method rather than three lines in the loop so the derivation can be driven
        directly: the account argument is the easy thing to get wrong here, and getting it
        wrong reports a confident "in good standing" for an account that is spun out - the
        exact ambiguity the quota module exists to remove.
        """
        if self.quota is None:
            return
        # Keyword, not positional: quota.state()'s first positional slot is `account`
        # (per-account quotas), so a bare timestamp here would silently bind to the wrong
        # parameter and never match any bucket.
        self.ctx.spins_exhausted = self.quota.state(account=self.stats.account,
                                                    now=time.time()).exhausted

    def _end_encounter(self, e: Transition) -> None:
        """Track consecutive useless encounters and start restocking after enough of them.

        A single exhausted encounter is normal - a hard Pokemon, a bad throw run. Several
        in a row means the throws themselves are doing nothing, which in practice means an
        empty bag.
        """
        cfg = self.cfg
        exhausted = self.ctx.throws_this_encounter >= cfg.max_throws_per_encounter
        self.ctx.throws_this_encounter = 0
        if not exhausted:
            self.ctx.failed_encounters = 0
            return
        self.ctx.failed_encounters += 1
        self.stats.encounters_exhausted += 1
        if self.ctx.failed_encounters < cfg.restock_after_failures or self.ctx.restocking:
            return
        if self.quota is not None \
                and self.quota.state(account=self.stats.account).exhausted:
            log.warning("throws are doing nothing and the 24h spin quota is used up; "
                        "restocking would be futile - the bag cannot be refilled here")
            return
        self.ctx.restocking_until = self.ctx.now + cfg.restock_max_seconds
        self.ctx.restock_stops_at_start = self.stats.stops_collected
        self.ctx.failed_encounters = 0
        self.stats.restocks += 1
        log.warning("%d encounters ended with throws doing nothing; restocking - "
                    "targeting PokeStops only until %d are collected (or %.0fs)",
                    cfg.restock_after_failures, cfg.restock_target_stops,
                    cfg.restock_max_seconds)

    def _apply_account_profile(self) -> None:
        """Point `cfg` at the settings for whoever we are logged in as.

        Checked every tick rather than hooked onto the places the account changes, because
        there are several - startup, a confirmed switch, and a failed one that hands back
        whatever the overlay last named - and a hook missing from one of them would be a
        silently wrong setting rather than a crash. The comparison is a string identity
        check on the common path.

        Both `self.cfg` and `ctx.cfg` are replaced: the FSM reads `ctx.cfg` (see
        `pick_target` and `desired_state` for the two `fight_rockets` sites), while the
        runner's own logging reads `self.cfg`, and one of them holding a stale answer is
        how "it says rockets are off but it keeps fighting them" happens.
        """
        account = self.stats.account
        if account == self._profile_account:
            return
        self._profile_account = account
        settings = profiles.settings_for(self.account_profiles, account)
        new = self._base_cfg.scaled(**settings) if settings else self._base_cfg
        changed = {k: v for k, v in settings.items()
                   if getattr(self._base_cfg, k) != v}
        self.cfg = new
        self.ctx.cfg = new
        if changed:
            log.info("settings for %s: %s", account or "an unidentified account",
                     ", ".join(f"{k}={str(v).lower()}" for k, v in sorted(changed.items())))

    def _update_restock(self) -> None:
        """Leave restock mode once the bag is plausibly refilled, or the budget expires."""
        ctx = self.ctx
        if not ctx.restocking:
            return
        got = self.stats.stops_collected - ctx.restock_stops_at_start
        if got >= self.cfg.restock_target_stops:
            log.info("restocked from %d stops; resuming normal targeting", got)
            ctx.restocking_until = 0.0
        elif ctx.now >= ctx.restocking_until:
            log.warning("restock window expired after %d stop(s); resuming normal targeting "
                        "- if this repeats, no PokeStop is in range here", got)
            ctx.restocking_until = 0.0

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

    def _collect_dialogue(self, frame, obs) -> None:
        """Save post-login screens for labelling. Never fatal: this is a data-collection
        side effect, and losing a frame must not end a run."""
        if self.dialogue_dump is None or self.ctx.state is not BotState.SWITCHING:
            return
        if obs.on_map:
            return
        try:
            import cv2
            self.dialogue_dump.mkdir(parents=True, exist_ok=True)
            name = f"switch_{int(time.time())}_{frame.seq:06d}.png"
            cv2.imwrite(str(self.dialogue_dump / name), frame.bgr)
        except Exception:
            log.debug("dialogue frame not saved", exc_info=True)

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

    # ---------------------------------------------------------------- accounts

    def _refresh_accounts(self, real: float) -> None:
        """Re-read the UI tree, in the three states that can be looking at PGSharp's own
        windows.

        SWITCHING is the state that OPENS the panel, and reads at ACCOUNTS_REFRESH because
        it is chasing an asterisk inside a bounded budget. PREFLIGHT drives the tail of
        that same ladder at startup (see fsm.Preflight) and shares the cadence for the same
        reason: its AutoWalk steps locate the star, the shortcut menu and a dialog button
        in the live tree, and `config.AutoWalk.budget_s` gives the whole ladder 30s - at
        RECOVER_ACCOUNTS_REFRESH (10s) it would spend that budget waiting for its second
        usable view. RECOVERING is the state that finds a panel already open and cannot see
        it any other way: measured at the bot's own resolution the accounts panel reports
        screen=Menu@0.95, in_overlay=False, x_button=False and no close button, so the tree
        is the only channel that knows it is there (see fsm.Recovering._panel_close). It
        gets its own, longer throttle - RECOVER_ACCOUNTS_REFRESH - because the read blocks
        this thread and RECOVERING is entered briefly after every ordinary timeout, not
        only when the bot is wedged; see that constant for the whole trade.

        What a read costs, since PREFLIGHT now buys them at startup: `UiTreeReader.read`
        blocks THIS thread - the run loop's own - for ~3.0s against the rendering game
        (measured: 2.96, 3.00, 3.00, 3.00, 4.46), bounded by `UiTreeReader.timeout` (5s),
        and the throttle below is stamped from when the read FINISHED, so usable views
        arrive ~5.5s apart. A preflight is therefore a handful of blocked seconds inside
        `Timings.preflight_timeout` (90s), once per run, before the bot starts playing -
        against a run that otherwise plays for hours zoomed in with Virtual Go Plus off.
        Nothing is perceived, no key is read and no signal is serviced while a read blocks,
        which is why this is throttled at all and why RECOVERING's throttle is four times
        longer.

        Everywhere else this still returns without reading: outside these states the panel
        is shut, so a blocking dump could only ever report `rows=()`.

        Paced on the REAL clock, like every other pacing decision in the loop: a paused run
        freezes `ctx.now`, and a frozen clock never reaches its own next deadline.
        """
        switching = self.ctx.state is BotState.SWITCHING
        # The two states that drive PGSharp's overlay themselves, and so need the fast
        # cadence and the AutoWalk icon reading below. `switching` stays separate from it
        # because a preflight has no switch attempt to keep books for - see below.
        driving = switching or self.ctx.state is BotState.PREFLIGHT
        if self.tree_reader is None \
                or not (driving or self.ctx.state is BotState.RECOVERING):
            return
        if real - self._accounts_read_at < (ACCOUNTS_REFRESH if driving
                                            else RECOVER_ACCOUNTS_REFRESH):
            return
        t0 = time.perf_counter()
        try:
            self.ctx.accounts = self.tree_reader.read()
        except Exception:
            log.exception("account tree read failed")
            return
        finally:
            # Stamped from when the read FINISHED, not when it started. The dump blocks
            # this thread, so a start-to-start throttle is no throttle at all once a read
            # runs long: a read that costs more than the interval makes the next one
            # eligible the instant it returns, and the loop never gets a frame back
            # between them. End-to-start guarantees a whole interval of seeing the screen
            # after every interval of not seeing it. Expressed as an offset from `real` -
            # the loop's single clock sample, passed in - rather than a fresh
            # perf_counter, so this method still takes its clock as an argument.
            self._accounts_read_at = real + (time.perf_counter() - t0)
        if not driving:
            # The rest belongs to a run that is driving the overlay itself. `_last_seen_active`
            # is the record of who the tree named during a switch ATTEMPT and is spent by
            # `_on_switch_failed`; the AutoWalk icon colour is read out of a shortcut menu
            # only `Switching`/`Preflight` ever open. Writing either from RECOVERING would be
            # answering a question nobody asked with a reading taken off the wrong screen.
            return
        if switching and self.ctx.accounts.available \
                and self.ctx.accounts.active is not None:
            # The asterisk is ground truth about who is logged in, and `verify` re-reads it
            # every couple of seconds right up to the timeout - on the live failure it
            # named the outgoing account fourteen times, the last of them minutes after the
            # login tap. That is evidence, and `_on_switch_failed` is where it gets spent.
            # Recorded here rather than read off `ctx.accounts` later because the handler
            # drops that view after every tap it takes.
            self._last_seen_active = self.ctx.accounts.active.name
        # Refresh the AutoWalk icon colour reading in lockstep with the view that just
        # supplied its bounds - see perception.autowalk_active_signal,
        # fsm.Switching._autowalk_menu, and fsm.Context.switch_autowalk_active for why
        # this needs no reset of its own between switch attempts. Uses `self._last_frame`,
        # the SAME frame `perceptor.observe` already ran on this tick (set at the top of
        # the read loop, before this method is ever called): the uiautomator dump just
        # read above supplies the icon's bounds, the frame supplies its colour.
        icon_rect = (self.ctx.accounts.autowalk_icon_rect_norm
                    if self.ctx.accounts.available else None)
        self.ctx.switch_autowalk_active = (
            perception.autowalk_active_signal(self._last_frame.bgr, icon_rect, self.cfg)
            if self._last_frame is not None else Tristate.UNKNOWN)

    def choose_next_account(self) -> Optional[str]:
        """Next usable account, round-robin from the current one.

        Named accounts come from the startup roster, and who we are from the session the
        counters belong to, because the live tree can only answer either question while
        the PGSharp panel is open - which, outside a switch, it never is. It took a live
        view as an alternative source once; nothing ever passed one, because the only
        caller decides while the panel is shut, and a branch that cannot run is a branch
        nothing keeps honest.

        A stale roster cannot produce a wrong tap. `Switching.step` looks the target up
        with `by_name` on a view read AFTER the switch began, so an account that is gone
        is simply not found, nothing is tapped, and the attempt times out.

        Everyone capped is not a reason to stop: the cap blocks stops, not catches
        (`pick_target` skips only STOP_TARGETS when `spins_exhausted`). So we move to
        whichever account frees up first and keep catching there while it does.
        """
        names = list(self.roster)
        current = self.stats.account
        # Anything less than a known origin inside a known roster is a guess, and the guess
        # would be to log out of an account that was working: a session that never learned
        # its name (or stopped being able to vouch for it after a failed switch), a name
        # nothing enumerated, or nowhere else to go. A missing signal means do nothing.
        if current is None or current not in names or len(names) < 2:
            return None
        start = names.index(current) + 1
        order = [n for n in names[start:] + names[:start] if n != current]
        if not order:
            return None
        if self.quota is None:
            return order[0]
        usable = [n for n in order if not self.quota.state(account=n).exhausted]
        if usable:
            return usable[0]
        if not self.quota.state(account=current).exhausted:
            # An empty `usable` says every ALTERNATIVE is capped; it says nothing about
            # where we are. Staying on an account that can still spin beats any ranking by
            # "whose oldest spin ages out first", which means nothing for an account that
            # is not waiting on one.
            return None
        # Every alternative is capped too, so the only question is who frees up first - and
        # the account we are already on is a candidate for that. Without it, two capped
        # accounts each name the other, the quota trigger stays due, and the bot spends the
        # whole wait trading places instead of catching. `current` leads because
        # soonest_reset keeps the first name on a tie, which is the "stay put" answer.
        best = self.quota.soonest_reset([current] + order)
        return None if best == current else best

    # ---------------------------------------------------------------- preflight

    def _maybe_preflight(self, obs: Observation) -> None:
        """Run the startup zoom-out / Virtual Go Plus / AutoWalk pass, ONCE, on the first
        confirmed map after the game starts (see `fsm.Preflight` for what those steps are
        and why they are the switch's own methods rather than a second copy).

        Here rather than in `Boot.step` for the same reason `_maybe_switch` is here: which
        state a run enters is the runner's decision, and only the runner can hold the latch
        that makes this happen once per RUN rather than once per visit. Hanging it off BOOT
        alone would also lose the feature on exactly the startups that need it most - BOOT's
        budget is 30s, a cold-started game is measured in tens of seconds (see
        `Timings.app_restart_grace`), and a boot that overruns reaches the map through
        RECOVERING, which never passes through BOOT again.

        Both BOOT and SCANNING are accepted, and the order matters. This is called before
        `fsm.step`, so on the tick BOOT first sees the map the state is still BOOT and the
        preflight starts there - the bot never gets a SCANNING tick in which to tap a target
        first. SCANNING covers the slow-start path above, where BOOT has already timed out.

        `obs.on_map` is required, not merely the state: `_zoom` and `_goplus` both fire at
        FIXED screen coordinates, so starting them against a loading screen or a post-login
        modal would be tapping blind - the whole reason `Switching._zoom` waits for the map
        as well.

        The latch is set BEFORE anything else can fail, so "once" holds however the attempt
        below turns out. Its scope is one game start, not one process: `apply` clears it
        again when RECOVERING's `effects.RestartApp` is accepted, since a cold relaunch
        resets the camera, Virtual Go Plus and the AutoWalk route exactly as a login does -
        without that, the first restart of a long run hands the reported symptom straight
        back for the rest of it. Nothing here can stop the bot playing: with the knob off
        this returns and the run is exactly what it was, and every exit from the state
        itself lands in SCANNING.
        """
        if self._preflight_done or not self.cfg.preflight:
            return
        if self.ctx.state not in (BotState.BOOT, BotState.SCANNING) or not obs.on_map:
            return
        self._preflight_done = True
        if self.tree_reader is None:
            # The zoom-out and the Go Plus toggle are fixed coordinates and an optical
            # reading, so they run regardless; AutoWalk is located in the uiautomator tree
            # and there is nothing to locate it with, so `fsm.Preflight._autowalk_open`
            # ends the state there rather than waiting out `config.AutoWalk.budget_s` for
            # a widget that cannot appear. Said once, plainly, because the alternative is
            # an operator watching two of the three steps happen with no stated cause -
            # which is the class of silence this whole change exists to remove.
            #
            # The trigger is NAMED as the usual cause rather than asserted as the only
            # one: `cli.prepare_accounts` also returns no reader when the pause file is
            # already present at startup, and a claim that is wrong a third of the time is
            # how a log line stops being read at all.
            log.warning("no PGSharp overlay reader this run, so the startup preflight "
                        "will zoom out and set Virtual Go Plus but cannot start AutoWalk "
                        "- the star widget is located in the view tree, which is read "
                        "only when a switch trigger (--switch-on-quota / --switch-every) "
                        "is armed and the run did not start paused")
        self._begin_preflight()

    def _begin_preflight(self) -> None:
        # Same reason `_begin_switch` does this: PREFLIGHT owns the screen for up to
        # `Timings.preflight_timeout` and fills it with gestures of its own, so anything
        # that happens afterwards is not evidence that some earlier tap caused it - and an
        # Intent is exactly that causal claim, which the ledger writes a training sample
        # on the strength of. Reachable only through the SCANNING entry path, and only
        # barely (a tap sets its intent and leaves SCANNING in the same tick), which is
        # precisely why it is one line here rather than an argument in a comment.
        self._abandon_intent("running the startup preflight before the screen answered")
        # Same three resets `_begin_switch` performs, and for the same reasons, minus every
        # one that belongs to a login. `ctx.accounts` goes because whatever view we hold
        # describes a panel that was shut. `switch_target` is pinned to None because that
        # is what tells the shared phase methods they are running a preflight and must not
        # name an account (see `fsm.Switching._label`) - and what stops `Preflight.step`
        # ever reaching the login-driving branch. `switch_autowalk_since` goes because a
        # stale value would let the AutoWalk ladder's wall clock expire before it started;
        # `enter_state` covers the per-visit counters (runner._RESET_ON_ENTRY).
        self.ctx.accounts = None
        self.ctx.switch_target = None
        self.ctx.switch_phase = fsm.PREFLIGHT_PHASES[0]
        self.ctx.switch_autowalk_since = 0.0
        log.info("startup preflight: zoom out, Virtual Go Plus, AutoWalk")
        self.enter_state(BotState.PREFLIGHT, IntentOutcome.CARRIED, "startup preflight")

    def _maybe_switch(self, obs: Observation) -> None:
        """Start a switch if a trigger is due and this is a safe moment to leave.

        SCANNING with the map in front of us is the only such moment: leaving an encounter
        or a Rocket fight abandons a Pokemon mid-throw, and the state alone is not enough
        because RECOVERING gives up INTO scanning without the map ever being confirmed.
        """
        cfg = self.cfg
        if self.tree_reader is None or self.ctx.state is not BotState.SCANNING \
                or not obs.on_map:
            return
        due_quota = cfg.switch_on_quota and self.ctx.spins_exhausted
        due_clock = (cfg.switch_every_minutes > 0
                     and self.ctx.now >= self._next_rotation)
        if not (due_quota or due_clock):
            return
        # A trigger being due says nothing about whether an attempt can succeed. Both
        # triggers stay due indefinitely - `spins_exhausted` for hours, and a missed
        # rotation deadline forever, since only a CONFIRMED switch moves it - so the
        # failure record, not the trigger, is what makes a refused switch stop repeating.
        if self._switch_failures >= SWITCH_MAX_FAILURES:
            return
        if self.ctx.now < self._switch_blocked_until:
            return
        # Decided from the cached roster, with no tree read at all. Probing here used to
        # cost a ~1s blocking dump and could never learn anything: with the panel shut the
        # tree lists no accounts, so the answer was always "nowhere to go".
        target = self.choose_next_account()
        if target is None:
            return
        self._begin_switch(target)

    def _begin_switch(self, name: str) -> None:
        # Any tap still waiting for an answer dies here, for the reason _abandon_intent
        # spells out: SWITCHING owns the screen for a whole `switch_timeout` and fills it
        # with post-login screens, so anything that happens after is not evidence that our
        # tap caused it.
        self._abandon_intent("switching account before the screen answered")
        # Whatever view we last held describes a panel that was shut, or a panel as it
        # looked during some earlier switch. Every tap in SWITCHING comes from a location
        # the tree just reported - `login_norm` sits 157px from `delete_norm` - so the
        # handler starts from nothing and waits for the first refresh of this switch.
        self.ctx.accounts = None
        self.ctx.switch_target = name
        self.ctx.switch_phase = "open"
        # Attempt 2 must not inherit attempt 1's login stamp: `_settle` waits out the
        # grace period from it, and a stale value satisfies that wait immediately - so a
        # verify could run against a login tap that had not happened yet. It is also what
        # tells `_on_switch_failed` whether this attempt ever tapped a login at all.
        self.ctx.switch_login_ts = 0.0
        # Same reasoning, same shape: a stale non-zero value here would make
        # `Switching._autowalk_deadline` believe attempt 2's AutoWalk ladder has already
        # been running since attempt 1's, and could time it out before it ever starts.
        self.ctx.switch_autowalk_since = 0.0
        self._last_seen_active = None
        self._switch_target = name
        log.info("switching account -> %s", name)
        self.enter_state(BotState.SWITCHING, IntentOutcome.CARRIED, f"switch to {name}")

    def _on_switch_failed(self, name: Optional[str]) -> None:
        """A switch expired without the overlay ever naming the target as active.

        Two things have to be recorded, and neither was:

        The attempt must not simply start again. Nothing else in the loop can stop it -
        the trigger that started it is still due, `choose_next_account` still names the
        same account, and the stuck watchdog cannot help because it refreshes on a visible
        map and in this failure mode the map IS visible; the overlay closes and the
        account just does not change. Live, that made the bot re-tap a control that had
        already refused it, every couple of minutes, for the rest of the run.

        And if a login WAS tapped, the outgoing name is no longer something we can simply
        assume: `switch_login_grace` exists because a login can land late, so an expiry is
        exactly the case where the tap may have worked after we stopped looking. But it is
        not a case of knowing nothing either. `verify` re-opens the panel and reads the
        asterisk every couple of seconds right up to the timeout, so this attempt has
        usually WATCHED who is logged in, minutes past the grace period - and that read is
        ground truth, not an assumption. The session is re-attributed to whatever the tree
        last named, which is normally the outgoing account and occasionally the target.

        `None` is reserved for the genuine no-evidence case: a login was tapped and no read
        during the attempt named anybody. It is deliberately not the answer whenever a
        login was tapped, because an unknown account reads the EMPTY unattributed bucket
        for its quota - measured, `spins_exhausted` flipping True -> False while the real
        account sat at its cap, the FSM resuming stop targeting the game would refuse, and
        `_explain_refusal` going quiet because the "" bucket is not exhausted. That is the
        152-refused-stops misdiagnosis `quota.py` was written to prevent. It also destroys
        `choose_next_account`'s origin, so the backoff above would gate a retry that could
        never be attempted anyway.
        """
        self._switch_failures += 1
        wait = SWITCH_BACKOFF_BASE * (2 ** (self._switch_failures - 1))
        self._switch_blocked_until = self.ctx.now + wait
        who = name or "another account"
        # 0.0 is `_begin_switch`'s "this attempt has not tapped a login" sentinel; the FSM
        # clock is a perf_counter reading, so a real stamp is never zero.
        if self.ctx.switch_login_ts:
            self.stats.account = self._last_seen_active
            if self._last_seen_active is None:
                log.warning("the login for %s was tapped, nothing confirmed it, and no "
                            "read during the attempt named an active account - the "
                            "logged-in account is now unknown and spins go to the "
                            "unattributed bucket until a switch confirms one", who)
            else:
                log.warning("the login for %s was tapped but never confirmed; the overlay "
                            "last read %s as the active account, so that is who this "
                            "session is booked to", who, self._last_seen_active)
        if self._switch_failures >= SWITCH_MAX_FAILURES:
            log.warning("%d account switches in a row never confirmed (last target: %s); "
                        "giving up on switching for this run. The overlay accepts the "
                        "login tap and the account does not change, which looks like a "
                        "login throttle - restart once it has cleared.",
                        self._switch_failures, who)
        else:
            # `Switching.on_timeout` has already said the switch never confirmed; this
            # line only has to say what follows from it.
            log.warning("that is switch attempt %d in a row to fail (target: %s); "
                        "holding off the next one for %s",
                        self._switch_failures, who, _hms(wait))

    def _on_switch_confirmed(self, name: str) -> None:
        """Close the outgoing account's session and start the incoming one's.

        Split rather than merged so uptime and rates stay attributable; one row covering
        two accounts describes neither.
        """
        old = self.stats
        if self.stats_path is not None:
            # Written even when the account is unknown: those hours were still worked, the
            # counters do not carry into the new session, and `close()` records an unnamed
            # session anyway - two paths disagreeing about the same object is how a run
            # silently vanishes from the history.
            try:
                append_session(self.stats_path, old.summary())
            except Exception:
                log.exception("could not append the outgoing session")
        # `paused_seconds` has to carry forward: the FSM clock is `real - paused_seconds`,
        # so zeroing it would jump `now` forward by the whole pause total and fire every
        # stored deadline at once. That means `started` must be on the same clock, which is
        # exactly `ctx.now` - with `perf_counter()` instead, the new session's uptime reads
        # short by every pause taken BEFORE the switch, and below RATE_MIN_UPTIME every
        # per-account rate then reports as unknown, which is the whole point of the split.
        self.stats = SessionStats(started=self.ctx.now, dry_run=old.dry_run, account=name)
        self.stats.paused_seconds = old.paused_seconds
        self._next_report = self.stats.started + REPORT_EVERY
        if self.dashboard is not None:
            # The dashboard holds the counters object, not the runner, so a TUI that is not
            # re-pointed keeps rendering the session that just ended.
            self.dashboard.stats = self.stats
        # A restock is a claim about THIS bag - "throws are doing nothing, go collect" - and
        # the new account brings its own. Its progress mark also points at the outgoing
        # counters, so leaving it behind makes `got` negative against a fresh session,
        # unreachable for the target, and the restock ends only when its 600s budget does.
        self.ctx.restocking_until = 0.0
        self.ctx.restock_stops_at_start = 0
        # Any rotation timer restarts here, so a quota switch and a clock switch cannot
        # stack into a second switch moments later.
        if self.cfg.switch_every_minutes > 0:
            self._next_rotation = self.ctx.now + self.cfg.switch_every_minutes * 60.0
        # Switching demonstrably works right now, so whatever earlier attempts ran into is
        # over. Anything else makes one bad patch - a throttle that has since cleared -
        # permanent for the rest of the run.
        self._switch_failures = 0
        self._switch_blocked_until = 0.0

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
                        self.ctx.throws_this_encounter += 1
                    elif budget == "tap" and isinstance(e, Tap):
                        self.stats.targets_tapped += 1
                    elif budget == "zoom" and isinstance(e, Pinch):
                        # Counted here, not by a self-reported SetFlag from _zoom: the
                        # handler is pure and cannot know whether the actuator actually
                        # accepted this gesture (rate-limit / queue backpressure can
                        # legitimately refuse it). Only an accepted application may move
                        # the FSM's repeat count, or `_zoom` could confirm the switch
                        # having sent fewer than `repeats` real zoom-outs.
                        self.ctx.switch_zoom_reps += 1
                    elif budget == "goplus" and isinstance(e, Tap):
                        # Same reasoning as switch_zoom_reps just above: `_goplus` is pure
                        # and cannot know whether its Tap actually reached the device, so
                        # only an ACCEPTED tap may advance the bound that keeps a stuck
                        # toggle from spending unlimited attempts.
                        self.ctx.switch_goplus_attempts += 1
                    elif budget == "switch_clear" and isinstance(e, Back):
                        # Same reasoning as switch_zoom_reps/switch_goplus_attempts just
                        # above: `_settle` is pure and cannot know whether its Back
                        # actually reached the device, so only an ACCEPTED press may
                        # advance the bound that stops it hammering BACK into a
                        # legitimate multi-minute LOADING screen (see
                        # config.Timings.switch_clear_max).
                        self.ctx.switch_clear_presses += 1
                    elif budget == "star_drag" and isinstance(e, Swipe):
                        # Same reasoning as every counter above: `_separate_star` is pure
                        # and cannot know whether its Swipe reached the device, and this
                        # bound is what stops a star that will not move being dragged for
                        # the rest of the switch (config.StarSeparation.max_drags).
                        self.ctx.star_drags += 1
                        # ...and restarting the AutoWalk ladder's own clock is part of the
                        # same fact, not a second decision. `AutoWalk.budget_s` (30s) is
                        # sized for the four settle-and-reread cycles the LADDER needs
                        # (star, menu, dialog, close); making the star tappable is not one
                        # of them. Each drag costs a whole cycle, because the branch below
                        # drops `ctx.accounts` after every applied effect - and a cycle is
                        # not `ACCOUNTS_REFRESH`: `_refresh_accounts` stamps its throttle
                        # from when the read FINISHED, and the dump was measured at ~3.0s
                        # (2.96, 3.00, 3.00, 3.00, 4.46), so usable views are ~5.5s apart.
                        # Three drags is ~16s of a 30s budget. Driven through this Runner
                        # and the real FSM at that cost, a ladder that completes with zero
                        # drags is abandoned part-way with two or three: `_autowalk_deadline`
                        # gives up, and what SCANNING inherits is the shortcut menu sitting
                        # over the reach ellipse it taps into - which `_autowalk_close`'s
                        # own docstring describes as silently killing AutoWalk for the rest
                        # of the run. Only an ACCEPTED drag defers the
                        # deadline, which is what keeps the deadline a backstop: a drag
                        # the actuator refuses moves nothing and buys no time, so a star
                        # that can never be dragged still ends the ladder at `budget_s`
                        # rather than holding the switch to `Timings.switch_timeout`.
                        self.ctx.switch_autowalk_since = self.ctx.now
                    elif budget == "restart" and isinstance(e, RestartApp):
                        # Same reasoning as every counter above, and it matters more here
                        # than anywhere else: `Recovering.on_timeout` is pure and cannot
                        # know whether its RestartApp reached the device. Counting a
                        # rejected one would spend a restart that never happened, and
                        # stamping the grace period for it would then sit out
                        # `app_restart_grace` waiting for an app that was never restarted.
                        # Only an accepted application does either.
                        self.ctx.app_restarts += 1
                        self.ctx.app_restart_ts = self.ctx.now
                        # A cold relaunch resets exactly the three things the preflight
                        # exists to set: the camera zoom goes back to default, Virtual Go
                        # Plus goes off, and there is no AutoWalk route - the same state a
                        # login leaves behind, which is why the switch ladder does all
                        # three. (`Switching._separate_star`'s own measurement is taken
                        # "immediately after effects.RestartApp relaunched the game".) A
                        # run-lifetime latch would therefore hand the reported symptom
                        # straight back: the run that motivated this change logged 215
                        # recoveries. Re-arming makes it once per app start rather than
                        # once per process, which is the scope that matches the cause;
                        # `Config.max_app_restarts` already bounds how many of those there
                        # can be, and `_maybe_preflight`'s own `obs.on_map` gate is what
                        # keeps this from racing the app that is still coming up.
                        self._preflight_done = False
                        log.warning("restarted %s (%d consecutive restart(s) with no "
                                    "confirmed map)", e.package, self.ctx.app_restarts)
                    self.ctx.last_action[budget] = self.ctx.now
                    self.ctx.taps_in_state += 1
                    if isinstance(e, (Tap, Swipe, Back, Pinch, RestartApp)):
                        self.ctx.settle_until = self.ctx.now + self.cfg.timings.ui_settle
                    if self.ctx.state in (BotState.SWITCHING, BotState.PREFLIGHT):
                        # The launcher tap TOGGLES the overlay, so a second decision taken
                        # from the same view closes the panel the first one opened and the
                        # switch stalls until it times out. Drop the view here, where
                        # every applied effect is already seen: the handler does nothing
                        # while it is None, and the next refresh reflects the tap.
                        #
                        # PREFLIGHT is included for the same reason at the other end of the
                        # ladder: the STAR toggles PGSharp's shortcut menu (see
                        # `fsm.Switching._autowalk_close`), so a preflight deciding twice
                        # from one view would close the menu it had just opened and then
                        # find no "AutoWalk" node to pick.
                        self.ctx.accounts = None

    # ---------------------------------------------------------------- trace

    def _write_trace(self, obs: Observation, effects: list[Effect]) -> None:
        if self._trace is None:
            return
        rec = {
            "seq": obs.seq, "t": round(obs.ts, 3), "state": self.ctx.state.value,
            "screen": obs.screen.label, "conf": round(obs.screen.conf, 3),
            "map": obs.map_ball.value, "x": obs.x_button.value, "enc": obs.encounter.value,
            # Recorded as the tristate's own name, not a bool: OFF and ON are separately
            # measured signatures and everything else is UNKNOWN, so collapsing it would
            # erase the only question a live trace can settle - whether real frames land
            # in the unmeasured band between the two (see config.Thresholds). Read it
            # against "map": off the map this ROI is noise, not absence.
            "goplus": obs.goplus.value,
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
        usr1 = getattr(signal, "SIGUSR1", None)
        if usr1 is not None:
            def _toggle(_signum, _frame):
                self.toggle_pause()
            try:
                previous[usr1] = signal.signal(usr1, _toggle)
            except (OSError, ValueError):
                pass
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
                # Two clocks, sampled once. `real` paces the loop (inference, display,
                # fps); `now` is the FSM clock and excludes paused time, so no deadline
                # expires while idle. Pacing off `now` deadlocks the loop: it is frozen,
                # so `now < next_infer` stays true forever and neither perception nor
                # waitKey ever runs again - which also made the p and q keys dead.
                self._real = real = time.perf_counter()
                paused = self._sync_pause(real)
                now = real - self.stats.paused_seconds
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

                if real < next_infer:
                    # Redisplay the last rendered HUD rather than a bare frame. Passing
                    # obs=None here used to draw the un-annotated frame ~1000x/second
                    # (the rate is capped only by waitKey), while the HUD was drawn once
                    # per inference at 8Hz - so the overlay was visible for roughly 8
                    # frames in 1000 and appeared to strobe.
                    if self.display and self._last_hud is not None \
                            and real - self._last_shown >= 1.0 / DISPLAY_FPS:
                        self._last_shown = real
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
                    # See `_frames_starved` for why this is not simply `last_map_ts`.
                    time.sleep(0.01)
                    if self._frames_starved(now):
                        self._halt("no usable frames")
                        break
                    continue

                frames += 1
                self._last_frame = frame
                next_infer = real + 1.0 / max(cfg.infer_fps, 0.1)

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

                if obs.on_map:
                    self.ctx.last_map_ts = now
                    # A confirmed map is the only evidence that a restart worked, so it is
                    # what refills the budget - see Config.max_app_restarts. This makes
                    # the bound "consecutive restarts that did not bring the map back": an
                    # app that crash-loops never reaches this line and so can never earn
                    # another restart, while a wedge cleared hours ago does not leave the
                    # rest of the run one restart poorer.
                    self.ctx.app_restarts = 0
                    # ...and the same evidence closes the cold-start hold. The grace
                    # window exists because nothing on a relaunching game is ours to
                    # press; a confirmed map is proof the relaunch is over, so holding
                    # the ladder for the rest of the 90s would be refusing to recover
                    # from a screen we can already see. Measured before this line
                    # existed: with the map confirmed 30s after a restart, `step`
                    # pressed nothing at 35s, 60s and 89s, and only resumed at 91s.
                    self.ctx.app_restart_ts = 0.0
                if fsm.rocket_screen(obs, cfg):
                    self.ctx.last_rocket_ts = now
                if not paused:
                    # Both of these are evidence stores, and a paused frame is evidence of
                    # nothing the bot did. The encounter ring is sized to hold the tail of
                    # an encounter, so a pause inside one rolls the award screen away and
                    # dumps identical idle frames in its place - a mislabelled corpus, the
                    # exact failure this rewrite exists to prevent. The ledger ring is
                    # claimed by a resolving intent, and no intent can resolve while
                    # paused, so staging only evicts the frame the pending intent needs.
                    if self.ctx.state is BotState.ENCOUNTER and self.encounter_dump is not None:
                        self._enc_ring.append(frame.bgr.copy())
                    if self.ledger is not None:
                        self.ledger.stage(frame, obs)
                    self._collect_dialogue(frame, obs)

                self._update_spins_exhausted()
                if paused:
                    # Perception still runs so the display stays live and the trace keeps
                    # a record, but the machine does not advance and nothing is actuated.
                    self._write_trace(obs, [])
                    if self.dashboard is not None:
                        try:
                            self.dashboard.update(obs, self.ctx.state, self._fps, paused=True)
                        except Exception:
                            log.exception("dashboard update failed")
                    if self.display:
                        self._show(window, frame, obs)
                    continue

                # All three are below the `paused` block on purpose: a paused run must not
                # drive the overlay, and a switch entered while paused would sit in
                # SWITCHING with the FSM clock frozen, so not even its timeout could end
                # it. A preflight is in exactly that position - same overlay, same frozen
                # budget - so it waits for the resume too.
                #
                # The preflight goes FIRST so a trigger that is already due on the very
                # first map frame (an account that starts the run capped, with
                # --switch-on-quota armed) cannot take the screen before the startup checks
                # have run. It cannot delay a switch by more than its own bounded budget:
                # `_maybe_switch` refuses any state but SCANNING, so the trigger simply
                # stays due until the preflight hands the screen back.
                self._apply_account_profile()
                self._refresh_accounts(real)
                self._maybe_preflight(obs)
                self._maybe_switch(obs)
                self._update_restock()
                effects = fsm.step(obs, self.ctx)
                self.apply(effects, obs)
                if self.dashboard is not None:
                    try:
                        self.dashboard.update(obs, self.ctx.state, self._fps, paused=self._paused)
                    except Exception:
                        log.exception("dashboard update failed")
                self._write_trace(obs, effects)
                self._ticks += 1

                if now >= self._next_report:
                    self._next_report = now + REPORT_EVERY
                    log.info("session: %s", self.stats.hud_line())
                    if self.quota is not None:
                        log.info("%s", self.quota.state(account=self.stats.account).line())

                elapsed = real - t0
                if elapsed >= 1.0:
                    self._fps = frames / elapsed
                    frames, t0 = 0, real

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
        # `status` was added to hud.render when the dashboard landed but never passed
        # here, so the counters line was never actually drawn on the preview window.
        # `SessionStats` subtracts paused_seconds itself, so it must be given the REAL
        # clock. Passing ctx.now - which already has paused_seconds removed - subtracted it
        # twice: after a 10-minute pause the HUD read "0m00s" uptime on an 11-minute run,
        # and every rate above it was divided by the wrong denominator.
        self._last_hud = hud.render(frame.bgr, obs, self.cfg, self.ctx.state, self._fps, extra,
                                    status=self.stats.hud_line(),
                                    paused=self._paused)
        self._shown_hud += 1

    def _show(self, window: str, frame: Frame, obs: Observation) -> bool:
        """Render the HUD for a fresh observation and display it."""
        self._last_obs = obs
        self._render(frame, obs)
        self._last_shown = self._real
        return self._blit(window)

    def _blit(self, window: str) -> bool:
        """Push the cached HUD to the window. Returns False when the user presses q."""
        import cv2
        if self._last_hud is None:
            return True
        cv2.imshow(window, self._last_hud)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("p"):
            self.toggle_pause()
        return key != ord("q")

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
        if self.quota is not None:
            log.info("%s", self.quota.state(account=self.stats.account).line())
