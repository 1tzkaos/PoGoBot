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
    DoubleTapDrag,
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

# States whose per-visit bookkeeping must reset on entry.
_RESET_ON_ENTRY = ("spun_disc", "taps_in_state", "switch_zoom_reps")


class Runner:
    def __init__(self, cfg: Config, source: FrameSource, actuator, perceptor,
                 ledger=None, keyboard=None, trace_path: Optional[Path] = None,
                 display: bool = True, stats_path: Optional[Path] = None,
                 dashboard=None, encounter_dump: Optional[Path] = None,
                 dialogue_dump: Optional[Path] = None,
                 quota: Optional[SpinQuota] = None,
                 pause_file: Optional[Path] = None, tree_reader=None,
                 roster: tuple[str, ...] = ()):
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
        self.dialogue_dump = dialogue_dump
        self.quota = quota
        self.pause_file = pause_file
        self.tree_reader = tree_reader
        # The accounts that exist, enumerated once at startup with the panel open. It is
        # not refreshed because it cannot be: outside a switch the panel is shut and the
        # tree lists no rows at all, so a live read would only ever shrink this to nothing.
        self.roster = tuple(roster)
        self._accounts_read_at = 0.0
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
        """Re-read the UI tree. Only during a switch: the dump blocks for ~1s, and it is
        only during a switch that the panel is open for it to see anything.

        Paced on the REAL clock, like every other pacing decision in the loop: a paused run
        freezes `ctx.now`, and a frozen clock never reaches its own next deadline.
        """
        if self.tree_reader is None or self.ctx.state is not BotState.SWITCHING:
            return
        if real - self._accounts_read_at < ACCOUNTS_REFRESH:
            return
        self._accounts_read_at = real
        try:
            self.ctx.accounts = self.tree_reader.read()
        except Exception:
            log.exception("account tree read failed")
            return
        if self.ctx.accounts.available and self.ctx.accounts.active is not None:
            # The asterisk is ground truth about who is logged in, and `verify` re-reads it
            # every couple of seconds right up to the timeout - on the live failure it
            # named the outgoing account fourteen times, the last of them minutes after the
            # login tap. That is evidence, and `_on_switch_failed` is where it gets spent.
            # Recorded here rather than read off `ctx.accounts` later because the handler
            # drops that view after every tap it takes.
            self._last_seen_active = self.ctx.accounts.active.name

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
                    elif budget == "zoom" and isinstance(e, DoubleTapDrag):
                        # Counted here, not by a self-reported SetFlag from _zoom: the
                        # handler is pure and cannot know whether the actuator actually
                        # accepted this gesture (rate-limit / queue backpressure can
                        # legitimately refuse it). Only an accepted application may move
                        # the FSM's repeat count, or `_zoom` could confirm the switch
                        # having sent fewer than `repeats` real zoom-outs.
                        self.ctx.switch_zoom_reps += 1
                    self.ctx.last_action[budget] = self.ctx.now
                    self.ctx.taps_in_state += 1
                    if isinstance(e, (Tap, Swipe, Back, DoubleTapDrag)):
                        self.ctx.settle_until = self.ctx.now + self.cfg.timings.ui_settle
                    if self.ctx.state is BotState.SWITCHING:
                        # The launcher tap TOGGLES the overlay, so a second decision taken
                        # from the same view closes the panel the first one opened and the
                        # switch stalls until it times out. Drop the view here, where
                        # every applied effect is already seen: the handler does nothing
                        # while it is None, and the next refresh reflects the tap.
                        self.ctx.accounts = None

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
                    time.sleep(0.01)
                    if now - self.ctx.last_map_ts > cfg.timings.stuck_watchdog:
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

                # Both are below the `paused` block on purpose: a paused run must not drive
                # the overlay, and a switch entered while paused would sit in SWITCHING
                # with the FSM clock frozen, so not even its timeout could end it.
                self._refresh_accounts(real)
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
