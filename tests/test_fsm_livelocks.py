"""Each test here is a v1 livelock that consumed entire unattended sessions.

Every one is ~5 lines and runs in milliseconds without a device attached - which is
precisely what the old architecture made impossible.
"""
import pytest

from pogobot import fsm
from pogobot.config import DEFAULT
from pogobot.effects import BotState, IntentOutcome, Tap, Transition, Halt, Swipe
from tests.factories import obs, det


def ctx(state=BotState.SCANNING, now=0.0, **kw):
    c = fsm.Context(cfg=DEFAULT, state=state, state_since=0.0, now=now)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def kinds(effects, t):
    return [e for e in effects if isinstance(e, t)]


# --- v1 SM-01: ENCOUNTER had no timeout branch at all -------------------------
def test_encounter_escapes_when_it_never_resolves():
    """v1 threw a ball every 3.8s forever into a post-catch dialog."""
    c = ctx(BotState.ENCOUNTER, now=DEFAULT.timings.encounter_timeout + 1)
    out = fsm.step(obs(screen="PokemonEncounter"), c)
    t = kinds(out, Transition)
    assert t and t[0].to is BotState.RECOVERING


def test_encounter_does_throw_while_within_budget():
    c = ctx(BotState.ENCOUNTER, now=5.0)
    out = fsm.step(obs(screen="PokemonEncounter"), c)
    assert kinds(out, Swipe), "should still be throwing before the timeout"


# --- v1 SM-02: CLOSING_POPUP timeout was an elif below the tap branch ---------
def test_popup_timeout_reachable_even_while_x_button_is_present():
    """v1 could never reach its 4s escape while the X-tap branch held."""
    c = ctx(BotState.POPUP, now=DEFAULT.timings.popup_timeout + 1)
    out = fsm.step(obs(x_button=True, screen="Menu", close_xy=(0.5, 0.88)), c)
    t = kinds(out, Transition)
    assert t and t[0].to is BotState.RECOVERING


# --- v1 SM-03: fallback close coordinate opened the main menu ------------------
def test_never_taps_a_close_button_it_did_not_locate():
    """The v1 fallback (0.50, 0.8808) sat inside the overworld main-menu Pokeball."""
    c = ctx(BotState.POPUP, now=1.0)
    out = fsm.step(obs(x_button=True, screen="Menu", close_xy=None), c)
    assert not kinds(out, Tap), "no located button must mean no tap"


def test_taps_the_button_when_it_is_located():
    c = ctx(BotState.POPUP, now=1.0)
    out = fsm.step(obs(x_button=True, screen="Menu", close_xy=(0.5, 0.88)), c)
    taps = kinds(out, Tap)
    assert taps and taps[0].y == 0.88


# --- v1 Chain A step 2: classifier misfire beat optical ground truth -----------
def test_optical_map_signal_vetoes_a_confident_classifier_misfire():
    """v1 evaluated the classifier branch before the map check, so 'Gym' @0.60 on the
    map dropped it into the open/close menu loop."""
    c = ctx(BotState.SCANNING, now=10.0)
    out = fsm.step(obs(on_map=True, screen="Menu", conf=0.99), c)
    assert not kinds(out, Transition), "must stay SCANNING while optics confirm the map"


def test_map_signal_also_vetoes_a_claimed_encounter():
    c = ctx(BotState.SCANNING, now=10.0)
    o = obs(on_map=True, screen="PokemonEncounter", conf=0.99)
    assert not fsm.encounter_confirmed(o, DEFAULT)


# --- rocket screens carry an X; POPUP must not outrank ROCKET -----------------
def test_rocket_outranks_popup_despite_x_button():
    c = ctx(BotState.SCANNING, now=10.0)
    out = fsm.step(obs(x_button=True, screen="Rocket", conf=0.9, close_xy=(0.5, 0.89)), c)
    t = kinds(out, Transition)
    assert t and t[0].to is BotState.ROCKET


def test_rocket_presses_the_located_affirmative_pill():
    c = ctx(BotState.ROCKET, now=10.0)
    out = fsm.step(obs(screen="Rocket", conf=0.9, pill_xy=(0.5, 0.72)), c)
    taps = kinds(out, Tap)
    assert taps and taps[0].y == 0.72


# --- v1 SM-04: +1.0 written 0.8s after a swipe with zero verification ---------
def test_pokestop_does_not_confirm_before_the_screen_opens():
    c = ctx(BotState.POKESTOP, now=2.0, spun_disc=True)
    out = fsm.step(obs(x_button=False, screen="Overworld"), c)
    assert not [e for e in out if isinstance(e, Transition)
                and e.outcome is IntentOutcome.CONFIRMED]


def test_pokestop_confirms_only_after_a_real_poi_screen_opened():
    c = ctx(BotState.POKESTOP, now=2.0, spun_disc=True)
    out = fsm.step(obs(x_button=True, screen="Poi"), c)
    t = kinds(out, Transition)
    assert t and t[0].outcome is IntentOutcome.CONFIRMED


def test_out_of_range_stop_is_refuted_and_cooled_for_a_long_time():
    from pogobot.effects import Cooldown
    c = ctx(BotState.POKESTOP, now=1.0)
    out = fsm.step(obs(x_button=True, out_of_range=True, screen="Poi"), c)
    cds = kinds(out, Cooldown)
    assert cds and cds[0].seconds == DEFAULT.cooldowns.out_of_range
    assert kinds(out, Transition)[0].outcome is IntentOutcome.REFUTED


# --- the stuck watchdog must stop rather than tap blindly ---------------------
def test_watchdog_halts_instead_of_tapping_forever():
    c = ctx(BotState.RECOVERING, now=1000.0, last_map_ts=0.0)
    out = fsm.step(obs(screen="Menu"), c)
    assert kinds(out, Halt), "a bot that cannot find the map must stop, not keep tapping"


# --- structural guarantees ----------------------------------------------------
def test_every_state_declares_a_timeout_and_an_escape():
    for s in BotState:
        assert s in fsm.HANDLERS
        assert isinstance(fsm.HANDLERS[s].timeout_s, (int, float))


def test_at_most_one_interrupt_per_tick_and_it_never_transitions():
    from pogobot.observation import Tristate
    c = ctx(BotState.SCANNING, now=10.0)
    out = fsm.step(obs(on_map=True, keyboard=Tristate.TRUE), c)
    assert not kinds(out, Transition)


def test_cooldown_blocks_a_target_it_already_failed_on():
    c = ctx(BotState.SCANNING, now=10.0, cooldowns=[(0.5, 0.63, 999.0)])
    out = fsm.step(obs(on_map=True, detections=[det(cx=0.5, cy=0.63)]), c)
    assert not kinds(out, Tap), "a cooled position must not be re-tapped"


def test_target_out_of_reach_is_ignored():
    c = ctx(BotState.SCANNING, now=10.0)
    out = fsm.step(obs(on_map=True, detections=[det(cx=0.95, cy=0.05)]), c)
    assert not kinds(out, Tap)


def test_in_reach_target_is_tapped_and_transitions():
    c = ctx(BotState.SCANNING, now=10.0)
    out = fsm.step(obs(on_map=True, detections=[det(cx=0.5, cy=0.63)]), c)
    assert kinds(out, Tap) and kinds(out, Transition)


# --- v1 #11: SCANNING was the only state with no else and no watchdog ---------
def test_scanning_does_not_act_when_the_map_is_not_visible():
    """A gym screen misclassified as an encounter @0.59 is refused by the confidence
    gate; SCANNING must then not swipe at it either."""
    c = ctx(BotState.SCANNING, now=1.0, last_map_ts=1.0)
    out = fsm.step(obs(on_map=False, screen="PokemonEncounter", conf=0.59), c)
    assert not kinds(out, Swipe) and not kinds(out, Tap)


def test_scanning_escalates_when_the_map_stays_missing():
    c = ctx(BotState.SCANNING, now=30.0, last_map_ts=0.0)
    out = fsm.step(obs(on_map=False, screen="PokemonEncounter", conf=0.59), c)
    t = kinds(out, Transition)
    assert t and t[0].to is BotState.RECOVERING


def test_optical_encounter_signal_decides_nothing():
    """Measured 27% false-positive on overworld and 30% recall; it is trace-only."""
    o = obs(on_map=False, encounter=True, screen="Menu", conf=0.99)
    assert not fsm.encounter_confirmed(o, DEFAULT)


# --- the learning path must actually be reachable ----------------------------
def test_successful_pokemon_tap_scores_confirmed_not_refuted():
    """`expected` must be the state that CONFIRMS the detection, not the waiting state.
    With expected=TARGETING a successful catch scored REFUTED and cooled a good spot."""
    from pogobot.effects import SetIntent
    c = ctx(BotState.SCANNING, now=10.0)
    out = fsm.step(obs(on_map=True, detections=[det(cx=0.5, cy=0.63)]), c)
    intent = kinds(out, SetIntent)[0].intent
    assert intent.expected is BotState.ENCOUNTER
    c.intent = intent
    c.state = BotState.TARGETING
    out2 = fsm.step(obs(screen="PokemonEncounter", conf=0.99), c)
    t = kinds(out2, Transition)
    assert t[0].to is BotState.ENCOUNTER
    assert t[0].outcome is IntentOutcome.CONFIRMED


def test_opening_a_stop_records_the_visit_so_confirm_is_reachable():
    """Nothing set ctx.spun_disc, so POKESTOP could never emit CONFIRMED."""
    from pogobot.effects import SetFlag
    c = ctx(BotState.POKESTOP, now=2.0)
    out = fsm.step(obs(x_button=True, screen="Poi"), c)
    flags = kinds(out, SetFlag)
    assert flags and flags[0].name == "spun_disc" and flags[0].value is True


def test_stop_never_swipes_the_screen():
    """The game auto-spins, so no disc swipe is needed. v1 swiped at y=0.45 on every
    stop, which dragged the map whenever the tap had not actually opened one."""
    for staged in (False, True):
        c = ctx(BotState.POKESTOP, now=2.0, spun_disc=staged)
        assert not kinds(fsm.step(obs(x_button=True, screen="Poi"), c), Swipe)


def test_stop_confirms_after_dwelling_for_auto_spin():
    c = ctx(BotState.POKESTOP, now=DEFAULT.timings.stop_dwell + 0.1, spun_disc=True)
    t = kinds(fsm.step(obs(x_button=True, screen="Poi"), c), Transition)
    assert t and t[0].outcome is IntentOutcome.CONFIRMED


def test_stop_does_not_confirm_before_the_dwell_elapses():
    c = ctx(BotState.POKESTOP, now=0.2, spun_disc=True)
    assert not kinds(fsm.step(obs(x_button=True, screen="Poi"), c), Transition)


def test_detections_below_target_confidence_are_not_tapped():
    """The detector runs at a low floor so the ledger can see marginal objects; the FSM
    must not act on them."""
    c = ctx(BotState.SCANNING, now=10.0)
    out = fsm.step(obs(on_map=True, detections=[det(cx=0.5, cy=0.63, conf=0.20)]), c)
    assert not kinds(out, Tap)
    c2 = ctx(BotState.SCANNING, now=10.0)
    out2 = fsm.step(obs(on_map=True, detections=[det(cx=0.5, cy=0.63, conf=0.40)]), c2)
    assert kinds(out2, Tap)


def test_stop_reach_is_configurable_and_currently_matches_pokemon():
    """This once forced a tighter ellipse for stops, on the belief that "Walk closer to
    interact" meant distance. It meant the 24h spin cap, which refuses with the same
    banner. The scale stays available; its value is now 1.0 until distance evidence that
    is not confounded by the quota says otherwise."""
    far = dict(cx=0.5, cy=0.63 + DEFAULT.reach.radius_y * 0.8)
    for name in ("pokemon", "pokestop"):
        c = ctx(BotState.SCANNING, now=10.0)
        got = bool(kinds(fsm.step(obs(on_map=True, detections=[det(name=name, **far)]), c), Tap))
        assert got is (DEFAULT.reach.stop_scale >= 1.0 or name == "pokemon")


def test_pokestops_close_in_are_still_tapped():
    c = ctx(BotState.SCANNING, now=10.0)
    out = fsm.step(obs(on_map=True, detections=[det(name="pokestop", cx=0.5, cy=0.64)]), c)
    assert kinds(out, Tap)


# --- found in a live run: ROCKET could not be pulled back by the map ----------
def test_rocket_returns_to_scanning_once_the_map_is_visible():
    """Observed live: screen read Overworld@1.00 for 25 consecutive ticks while the state
    stayed ROCKET, because ROCKET was missing from the map pull-back set."""
    c = ctx(BotState.ROCKET, now=20.0)
    t = kinds(fsm.step(obs(on_map=True, screen="Overworld"), c), Transition)
    assert t and t[0].to is BotState.SCANNING


def test_pokestop_returns_to_scanning_once_the_map_is_visible():
    c = ctx(BotState.POKESTOP, now=20.0)
    t = kinds(fsm.step(obs(on_map=True, screen="Overworld"), c), Transition)
    assert t and t[0].to is BotState.SCANNING


def test_a_located_pill_is_tapped_even_inside_the_settle_window():
    """The BATTLE pill was visible for ~1s and the settle window from the preceding close
    tap swallowed the whole opportunity."""
    c = ctx(BotState.ROCKET, now=10.0, settle_until=11.0)
    out = fsm.step(obs(screen="Rocket", conf=0.9, pill_xy=(0.5, 0.72)), c)
    assert kinds(out, Tap), "a visibly located button must not be blocked by settle"


def test_blind_dialogue_taps_still_respect_the_settle_window():
    c = ctx(BotState.ROCKET, now=10.0, settle_until=11.0)
    out = fsm.step(obs(screen="Rocket", conf=0.9, pill_xy=None), c)
    assert not kinds(out, Tap), "a blind tap must still wait for the UI to settle"


# --- found in a live run: ROCKET <-> ENCOUNTER oscillation ---------------------
def test_rocket_is_not_pulled_out_by_an_encounter_looking_screen():
    """A Rocket battle is a run of screens the classifier reads as PokemonEncounter.
    Observed live: 6 ROCKET<->ENCOUNTER round trips in 70s, which also double counted
    both rockets_engaged and encounters."""
    c = ctx(BotState.ROCKET, now=10.0, last_rocket_ts=9.0)
    out = fsm.step(obs(screen="PokemonEncounter", conf=0.99), c)
    assert not kinds(out, Transition), "an encounter screen must not interrupt the fight"


def test_rocket_still_yields_to_the_map():
    c = ctx(BotState.ROCKET, now=10.0, last_rocket_ts=9.9)
    t = kinds(fsm.step(obs(on_map=True, screen="Overworld"), c), Transition)
    assert t and t[0].to is BotState.SCANNING, "the map must always win"


def test_reward_encounter_is_taken_once_rocket_screens_stop():
    """After the fight the shadow Pokemon encounter is real and must be caught."""
    c = ctx(BotState.ROCKET, now=100.0,
            last_rocket_ts=100.0 - DEFAULT.timings.rocket_hold - 1)
    t = kinds(fsm.step(obs(screen="PokemonEncounter", conf=0.99), c), Transition)
    assert t and t[0].to is BotState.ENCOUNTER
