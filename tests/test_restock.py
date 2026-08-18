"""Running out of Poke Balls, and restocking from PokeStops.

Out of balls is not observable optically - one labelled example and no clean positive
set means any threshold would be guessed. It IS observable behaviourally: throws that
change nothing are throws that are doing nothing.
"""
import pytest

from pogobot import fsm, runner as runner_mod
from pogobot.config import DEFAULT
from pogobot.effects import BotState, IntentOutcome, Swipe, Tap, Transition
from tests.factories import det, obs as mkobs


def ctx(state=BotState.ENCOUNTER, now=100.0, **kw):
    c = fsm.Context(cfg=DEFAULT, state=state, state_since=now - 1, now=now)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def kinds(effects, t):
    return [e for e in effects if isinstance(e, t)]


# ------------------------------------------------------------------ giving up

def test_it_keeps_throwing_while_within_the_budget():
    c = ctx(throws_this_encounter=DEFAULT.max_throws_per_encounter - 1)
    assert kinds(fsm.step(mkobs(screen="PokemonEncounter", conf=0.99), c), Swipe)


def test_it_flees_once_the_throws_are_doing_nothing():
    """The reported symptom: an encounter with no Poke Balls left, which the bot sat on
    until the watchdog halted the run."""
    c = ctx(throws_this_encounter=DEFAULT.max_throws_per_encounter)
    out = fsm.step(mkobs(screen="PokemonEncounter", conf=0.99), c)
    taps = kinds(out, Tap)
    assert taps and taps[0].y < 0.2 and taps[0].x < 0.2, "must tap the flee icon, top left"
    assert not kinds(out, Swipe), "must stop throwing"
    t = kinds(out, Transition)
    assert t and t[0].to is BotState.SCANNING


# ------------------------------------------------------------------ the livelock

def test_an_encounter_we_left_is_not_re_entered_while_the_screen_is_unchanged():
    """ENCOUNTER -> RECOVERING -> ENCOUNTER repeated until the watchdog halted the run."""
    c = ctx(BotState.SCANNING, now=100.0, left_encounter_ts=99.0, last_map_ts=50.0)
    out = fsm.step(mkobs(screen="PokemonEncounter", conf=0.99), c)
    to = [t.to for t in kinds(out, Transition)]
    assert BotState.ENCOUNTER not in to, "must not walk straight back into it"
    # Escalating instead is correct: no map has been confirmed for 50s, so something is
    # wrong and RECOVERING is where that gets handled.
    assert to == [BotState.RECOVERING]


def test_a_new_encounter_after_the_map_is_taken_immediately():
    """The hold must not block a different Pokemon tapped seconds later."""
    c = ctx(BotState.TARGETING, now=100.0, left_encounter_ts=99.0, last_map_ts=99.5)
    t = kinds(fsm.step(mkobs(screen="PokemonEncounter", conf=0.99), c), Transition)
    assert t and t[0].to is BotState.ENCOUNTER


# ------------------------------------------------------------------ restock mode

def test_restocking_ignores_pokemon_and_takes_stops():
    c = ctx(BotState.SCANNING, now=100.0, restocking_until=1e9)
    o = mkobs(on_map=True, detections=[det("pokemon", 0.9, cx=0.5, cy=0.63),
                                       det("pokestop", 0.6, cx=0.52, cy=0.64)])
    out = fsm.step(o, c)
    assert kinds(out, Tap), "it should still tap something"
    t = kinds(out, Transition)
    assert t and t[0].to is BotState.POKESTOP, "a Pokemon must not outrank the stop"


def test_normal_mode_still_prefers_pokemon():
    c = ctx(BotState.SCANNING, now=100.0)
    o = mkobs(on_map=True, detections=[det("pokemon", 0.9, cx=0.5, cy=0.63),
                                       det("pokestop", 0.6, cx=0.52, cy=0.64)])
    t = kinds(fsm.step(o, c), Transition)
    assert t and t[0].to is BotState.TARGETING


def test_restocking_with_no_stop_in_reach_does_not_tap_a_pokemon():
    c = ctx(BotState.SCANNING, now=100.0, restocking_until=1e9)
    o = mkobs(on_map=True, detections=[det("pokemon", 0.95, cx=0.5, cy=0.63)])
    assert not kinds(fsm.step(o, c), Tap)


# ------------------------------------------------------------------ entering and leaving

class _Act:
    def apply(self, effect, now=None):
        return True

    def healthy(self):
        return True

    def stats(self):
        return {"sent": 0}

    def close(self):
        pass


class _Src:
    def read(self):
        return None

    def healthy(self):
        return True

    def release(self):
        pass


def _runner():
    return runner_mod.Runner(DEFAULT, _Src(), _Act(), perceptor=None, display=False)


def _exhausted_encounter(r, now):
    r.ctx.now = now
    r.ctx.state = BotState.TARGETING
    r.apply([Transition(BotState.ENCOUNTER, IntentOutcome.CONFIRMED, "x")], None)
    for _ in range(DEFAULT.max_throws_per_encounter):
        r.apply([Swipe(0.5, 0.84, 0.5, 0.38, "throw", budget="throw")], None)
    r.ctx.last_map_ts = now + 1          # a real end, not a resume
    r.ctx.now = now + 1
    r.apply([Transition(BotState.SCANNING, IntentOutcome.EXPIRED, "exhausted")], None)


def test_one_bad_encounter_does_not_trigger_restocking():
    r = _runner()
    _exhausted_encounter(r, 100.0)
    assert not r.ctx.restocking
    assert r.stats.encounters_exhausted == 1


def test_repeated_useless_encounters_start_restocking():
    r = _runner()
    for i in range(DEFAULT.restock_after_failures):
        _exhausted_encounter(r, 100.0 + i * 10)
    assert r.ctx.restocking, "the bag is empty; go get stops"
    assert r.stats.restocks == 1


def test_a_successful_encounter_resets_the_failure_streak():
    r = _runner()
    _exhausted_encounter(r, 100.0)
    r.ctx.now = 130.0
    r.ctx.state = BotState.TARGETING
    r.apply([Transition(BotState.ENCOUNTER, IntentOutcome.CONFIRMED, "x")], None)
    r.ctx.last_map_ts = 131.0
    r.ctx.now = 131.0
    r.apply([Transition(BotState.SCANNING, IntentOutcome.CARRIED, "caught")], None)
    assert r.ctx.failed_encounters == 0
    _exhausted_encounter(r, 140.0)
    assert not r.ctx.restocking, "the streak must be consecutive"


def test_restocking_ends_once_enough_stops_are_collected():
    r = _runner()
    for i in range(DEFAULT.restock_after_failures):
        _exhausted_encounter(r, 100.0 + i * 10)
    assert r.ctx.restocking
    r.stats.stops_collected = r.ctx.restock_stops_at_start + DEFAULT.restock_target_stops
    r._update_restock()
    assert not r.ctx.restocking


def test_restocking_gives_up_when_no_stop_is_reachable():
    """152 out of 152 stop taps came back 'Walk closer' in a real session; restock must
    not become a permanent mode."""
    r = _runner()
    for i in range(DEFAULT.restock_after_failures):
        _exhausted_encounter(r, 100.0 + i * 10)
    assert r.ctx.restocking
    r.ctx.now = r.ctx.restocking_until + 1
    r._update_restock()
    assert not r.ctx.restocking
