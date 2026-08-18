"""Counters driven through the REAL machine: fsm.step -> Runner.apply, tick by tick.

test_stats.py checks the effect-to-counter mapping by handing `apply` a hand-built list.
That cannot see the defects that live in the SEQUENCE the FSM actually emits - a screen
the machine visits twice, or an outcome that never reaches the branch counting it. Every
case here is a path that was miscounted until it was traced this way, and each one asserts
against the number of real-world events, not the number of state entries.
"""
import logging

import pytest

from pogobot import fsm, runner as runner_mod
from pogobot.config import DEFAULT
from pogobot.effects import BotState
from tests.factories import det, obs


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


class Bot:
    """A Runner with a hand-cranked clock, ticked exactly the way `run()` ticks it."""

    def __init__(self, t0=1000.0, state=BotState.SCANNING):
        self.r = runner_mod.Runner(DEFAULT, _Src(), _Act(), perceptor=None, display=False)
        self.t = t0
        c = self.r.ctx
        c.now = c.state_since = c.last_map_ts = t0
        c.state = state

    def tick(self, o, dt=0.125):
        self.t += dt
        c = self.r.ctx
        c.now = self.t
        if o.on_map:                                  # mirrors Runner.run
            c.last_map_ts = self.t
        if fsm.rocket_screen(o, c.cfg):
            c.last_rocket_ts = self.t
        self.r.apply(fsm.step(o, c), o)

    def ticks(self, o, n, dt=0.125):
        for _ in range(n):
            self.tick(o, dt)

    @property
    def s(self):
        return self.r.stats


MAP = lambda **k: obs(on_map=True, screen="Overworld", **k)
ENCOUNTER = lambda **k: obs(on_map=False, screen="PokemonEncounter", conf=0.95, **k)
STOP = lambda **k: obs(on_map=False, x_button=True, screen="Poi", conf=0.9, **k)
# Not the map, no X button, nothing that outranks the current state: a tap that landed
# somewhere the bot has no opinion about.
NOWHERE = obs(on_map=False, x_button=False, screen="Menu", conf=0.99)


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _pokemon_tap(b):
    b.tick(MAP(detections=[det("pokemon", 0.9, 0.5, 0.63)]))


def _stop_tap(b, name="pokestop"):
    b.tick(MAP(detections=[det(name, 0.9, 0.5, 0.63)]))


# --------------------------------------------------- one encounter is one encounter

def test_an_encounter_screen_that_outlasts_its_budget_is_still_one_encounter():
    """ENCOUNTER times out into RECOVERING, but `desired_state` outranks RECOVERING and the
    same screen is still up, so the machine returns immediately. Before this was counted as
    a round trip, 100s on ONE screen reported 4 encounters and 4 catch attempts."""
    b = Bot()
    _pokemon_tap(b)
    b.ticks(ENCOUNTER(), 800)                      # 100s = three encounter timeouts
    assert b.s.encounters == 1
    assert b.s.catch_attempts == 1
    assert b.s.balls_thrown > 4, "it really did keep throwing"
    assert b.s.recoveries >= 3, "the churn must stay visible; only the double count went"
    # The bot now gives up on a screen that will not resolve rather than only timing out,
    # so it also records the encounter as exhausted. On a device the flee tap ends it;
    # this fixture's screen never changes, which is why the churn continues here.
    assert b.s.encounters_exhausted >= 1


def test_a_resumed_encounter_still_ends_exactly_once():
    b = Bot()
    _pokemon_tap(b)
    b.ticks(ENCOUNTER(), 400)                      # one timeout and the return trip
    b.ticks(MAP(), 4)
    assert (b.s.encounters, b.s.encounters_finished) == (1, 1)


def test_a_new_encounter_after_a_recovery_that_worked_is_counted():
    """The guard is 'was the map confirmed in between', so a real second encounter must
    not be swallowed by it."""
    b = Bot()
    _pokemon_tap(b)
    b.ticks(ENCOUNTER(), 400)
    b.ticks(MAP(), 8)                              # recovery landed on the map
    b.ticks(ENCOUNTER(), 8)
    assert b.s.encounters == 2


def test_an_ordinary_encounter_is_unaffected():
    b = Bot()
    _pokemon_tap(b)
    b.ticks(ENCOUNTER(), 60)
    b.ticks(MAP(), 2)
    assert (b.s.encounters, b.s.encounters_finished, b.s.catch_attempts) == (1, 1, 1)


# --------------------------------------------------- stop taps that answered something else

def test_a_stop_tap_that_missed_is_not_reported_as_out_of_range():
    """The tap did nothing and the map is still up. That is REFUTED, but the bot never
    read 'Walk closer to interact' and must not claim it did."""
    b = Bot()
    _stop_tap(b)
    b.tick(MAP())
    assert b.s.stops_out_of_range == 0
    assert b.s.targets_tapped == 1


def test_a_rocket_stop_engaging_a_grunt_is_not_reported_as_out_of_range():
    b = Bot()
    _stop_tap(b, "pokestop_rocket")
    b.tick(obs(on_map=False, x_button=True, screen="Rocket", conf=0.9, pill_xy=(0.5, 0.88)))
    assert b.s.rockets_engaged == 1
    assert b.s.stops_out_of_range == 0, "one fight used to cost a phantom out-of-range stop"


def test_the_out_of_range_banner_is_still_counted():
    b = Bot()
    _stop_tap(b)
    b.tick(STOP(out_of_range=True))
    assert b.s.stops_out_of_range == 1
    assert b.s.stops_collected == 0


def test_a_collected_stop_is_still_counted():
    b = Bot()
    _stop_tap(b)
    b.ticks(STOP(), 20)                            # open, dwell for auto-spin, leave
    assert b.s.stops_collected == 1
    assert b.s.stops_out_of_range == 0


# --------------------------------------------------- taps that expired

def test_a_stop_tap_whose_screen_never_opened_counts_as_an_expired_tap():
    """POKESTOP timing out is the same real event as TARGETING timing out - a tap that
    never produced the screen it claimed - and only the TARGETING half was counted."""
    b = Bot()
    _stop_tap(b)
    b.ticks(NOWHERE, 80)                           # 10s > pokestop_timeout
    assert b.s.taps_expired == 1


def test_a_pokemon_tap_whose_screen_never_opened_still_counts_once():
    b = Bot()
    _pokemon_tap(b)
    b.ticks(NOWHERE, 40)
    assert b.s.taps_expired == 1


def test_an_encounter_timeout_is_not_an_expired_tap():
    b = Bot()
    _pokemon_tap(b)
    b.ticks(ENCOUNTER(), 400)
    assert b.s.taps_expired == 0, "the tap worked; it was the encounter that ran long"


# --------------------------------------------------- halts the runner raises itself

@pytest.mark.parametrize("kind", ["source", "actuator", "frames"])
def test_every_way_a_run_halts_is_counted(kind):
    """Four paths abort the loop directly. Each logged HALTED and returned 1 while
    recording a session with halts=0, so the lifetime total was missing all of them."""
    src, act = _Src(), _Act()
    if kind == "source":
        src.healthy = lambda: False
        src.failure_reason = lambda: "screenrecord exited"
    elif kind == "actuator":
        act.healthy = lambda: False
    r = runner_mod.Runner(DEFAULT, src, act, perceptor=None, display=False)
    if kind == "frames":
        r.ctx.last_map_ts -= DEFAULT.timings.stuck_watchdog + 1
    rc = r.run()
    assert rc == 1 and r._halt_reason
    assert r.stats.halts == 1
    assert r.stats.summary()["halts"] == 1


def test_a_frame_source_that_simply_ran_out_is_not_a_halt():
    """Replaying a finished directory is a clean end, not a failure."""
    src = _Src()
    src.healthy = lambda: False
    r = runner_mod.Runner(DEFAULT, src, _Act(), perceptor=None, display=False)
    assert r.run() == 0
    assert r.stats.halts == 0
