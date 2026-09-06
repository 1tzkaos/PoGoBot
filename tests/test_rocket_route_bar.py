"""Entering ROCKET at all, as opposed to what happens once there.

Three screens have livelocked ROCKET the same way: Pokemon GO's exit dialog, its SPONSORED
interstitial, and its news card ("GO Fest 2026: ... Technical Issue"). Each clears
`screen_min_conf`, each carries something pill-shaped where the affirmative sits, and each
costs the full 150s of `Rocket.timeout_s` per visit with nothing able to act - the press is
already refused by `rocket_pill_min_conf`, so the machine enters, does nothing, times out
and re-enters. The news card was measured doing it for 30 minutes: 14 ROCKET entries, 0
stops, until the productivity watchdog ended the run.

The bar is PILL-CONDITIONAL, and that asymmetry is the whole design. Measured live:

    pill LOCATED   n=67768   median 0.997   >=0.90: 79.1%
    pill ABSENT    n= 3726   median 0.671   >=0.90: 31.0%

so a flat bar refuses two thirds of the pill-absent population, which is grunt dialogue
mid-fade - a real fight with no button to find. Every test below exists because the flat
version passed the whole suite while doing that.
"""
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, "tests")
from factories import obs

from pogobot import fsm
from pogobot.config import DEFAULT as C


def _ctx(cfg=C, now=1000.0):
    c = fsm.Context(cfg=cfg, now=now)
    c.last_map_ts = now
    return c


PILL = (0.5, 0.82)


# ---------------------------------------------------------------- the impostor it was written for

def test_a_pill_bearing_screen_below_the_bar_is_not_a_rocket_fight():
    """The news card: Rocket @ 0.73 with SEE DETAILS located at (0.5, 0.52)."""
    assert fsm.rocket_screen(obs(screen="Rocket", conf=0.73, pill_xy=PILL), C) is False


def test_and_so_the_machine_never_routes_into_rocket_for_it():
    assert fsm.desired_state(obs(screen="Rocket", conf=0.73, pill_xy=PILL),
                             _ctx()) is not fsm.BotState.ROCKET


@pytest.mark.parametrize("conf", [0.62, 0.73, 0.88, 0.899])
def test_the_whole_band_below_the_bar_is_refused_when_a_pill_is_up(conf):
    """0.899 is the highest confidence measured on a dead pill-located frame, against a
    bar of 0.900 - the impostor side is a knife edge, not a gap."""
    assert fsm.rocket_screen(obs(screen="Rocket", conf=conf, pill_xy=PILL), C) is False


def test_a_real_pill_bearing_fight_still_routes():
    assert fsm.rocket_screen(obs(screen="Rocket", conf=0.998, pill_xy=PILL), C) is True
    assert fsm.desired_state(obs(screen="Rocket", conf=0.998, pill_xy=PILL),
                             _ctx()) is fsm.BotState.ROCKET


# ---------------------------------------------------------------- the regression it must not cause

@pytest.mark.parametrize("conf", [0.62, 0.671, 0.73, 0.79, 0.88])
def test_a_dialogue_frame_with_no_pill_still_routes(conf):
    """RED-GREEN for the pill condition, and the reason a flat bar cannot ship.

    Pill-absent Rocket frames run median 0.671 live. Under a flat 0.90 bar every
    confidence here is refused and grunt dialogue stops advancing - a real fight the bot
    would silently stop playing, while still catching Pokemon so every liveness signal
    stays green.
    """
    assert fsm.rocket_screen(obs(screen="Rocket", conf=conf), C) is True


def test_a_flat_bar_would_have_broken_dialogue_and_this_proves_it():
    """Asserts the counterfactual directly, so nobody 'simplifies' the condition away."""
    frame = obs(screen="Rocket", conf=0.671)
    assert fsm.rocket_screen(frame, C) is True
    flat = replace(C, screen_min_conf=C.rocket_route_min_conf)
    assert fsm.rocket_screen(frame, flat) is False


# ---------------------------------------------------------------- the vetoes are still load-bearing

def test_the_exit_dialog_is_refused_by_its_veto_not_by_the_bar():
    """Measured: exit-dialog frames classify Rocket @ 0.9919-0.9929, ABOVE the bar. The
    bar separates two of the three known impostors; this one is the veto's alone."""
    frame = obs(screen="Rocket", conf=0.992, pill_xy=PILL, exit_dialog=True)
    assert fsm.rocket_screen(frame, C) is False
    without = obs(screen="Rocket", conf=0.992, pill_xy=PILL, exit_dialog=False)
    assert fsm.rocket_screen(without, C) is True, "the bar alone would let it through"


def test_the_sponsored_ad_veto_still_carries_its_own_weight():
    """The bar shadows this veto at the ad's real 0.615, which would silently cancel its
    regression coverage. Exercised here above the bar, the way
    tests/test_sponsored_ad.py already does for rocket_pill_min_conf."""
    frame = obs(screen="Rocket", conf=0.99, pill_xy=PILL, promo_xy=(0.866, 0.884))
    assert fsm.rocket_screen(frame, C) is False
    without = obs(screen="Rocket", conf=0.99, pill_xy=PILL)
    assert fsm.rocket_screen(without, C) is True


def test_the_map_veto_still_carries_its_own_weight():
    assert fsm.rocket_screen(obs(screen="Rocket", conf=0.99, pill_xy=PILL, on_map=True),
                             C) is False


# ---------------------------------------------------------------- the constant

def test_the_bar_matches_the_press_gate_it_shares_evidence_with():
    assert C.rocket_route_min_conf == C.rocket_pill_min_conf


def test_the_bar_sits_above_every_measured_impostor_and_below_every_real_frame():
    assert 0.899 < C.rocket_route_min_conf <= 0.998
