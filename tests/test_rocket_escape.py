"""Leaving a Team GO Rocket screen the run has been told not to fight.

`fight_rockets` off has always stopped the bot TAPPING invaded stops (`fsm.pick_target`
skips ROCKET_TARGETS). It never stopped it MEETING the screens - a balloon invasion
arrives unprompted, an invaded stop can be detected as a plain `pokestop` and tapped, and
a per-account profile can flip the setting mid-fight. `fsm.desired_state` gated the
ROCKET route on the same flag and nothing else claimed the screen, so on the majority of
such frames the machine had no route at all.

Measured in logs/trace.jsonl over one 3h27m run with the setting off for that account:
RECOVERING on 7.4% of frames where a healthy run sits at 2-3%, five app restarts, zero
halts, and 3437 frames wedged on a screen classified Rocket. The 90s before three of the
five restarts each read screens {Rocket: 720/720}, states {RECOVERING: 695, SCANNING:
25}, effects {Back: 19, Transition: 37, Tap: 1} - nineteen BACKs into a Rocket dialogue
that BACK does not dismiss, escaped only because the restart ladder relaunched the game.
Zero halts is the point: the safety net held every time, and the thing under it was
broken.

See `fsm.rocket_exit_screen` for the fix and the corpus measurement behind it.
"""
from __future__ import annotations

import collections
import copy
import dataclasses
import pathlib

import pytest

from pogobot import fsm
from pogobot.config import DEFAULT
from pogobot.effects import BotState, Back, Halt, RestartApp, Tap, Transition
from pogobot.observation import ScreenGuess
from tests.factories import obs

#: The run this fix is about: Team GO Rocket turned off for the current account.
OFF = DEFAULT.scaled(fight_rockets=False)


def ctx(cfg=OFF, state=BotState.SCANNING, now=100.0, **kw):
    c = fsm.Context(cfg=cfg, state=state, state_since=0.0, now=now)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def kinds(effects, t):
    return [e for e in effects if isinstance(e, t)]


def rocket(x_button=False, close_xy=(0.50, 0.92), pill_xy=None, conf=0.95):
    """A frame the classifier reads as a Rocket screen.

    `screen="Rocket"` at a confidence over `screen_min_conf` is what makes this genuinely
    off-map: `obs(on_map=False)` alone does not, because the factory's default
    `screen="Overworld"` at conf 0.99 satisfies `Observation.on_map` on its own.

    `pill_xy` defaults to None, so the default frame is one where only a close button was
    located - the shape `fsm.CLOSE_PILL_MIN_DY` has nothing to say about. Frames that
    locate both are built explicitly, since the relationship between the two coordinates is
    the whole subject there.
    """
    return obs(screen="Rocket", conf=conf, x_button=x_button, close_xy=close_xy,
               pill_xy=pill_xy)


# ------------------------------------------------------- rockets off: the escape exists

def test_rockets_off_routes_a_closable_rocket_screen_to_popup():
    """The route that did not exist. POPUP is reached, not ROCKET - the run still refuses
    to fight - and not None, which is what fell through to the BACK loop."""
    assert fsm.desired_state(rocket(), ctx()) is BotState.POPUP


def test_rockets_off_still_routes_the_frames_the_overlay_branch_already_saw():
    """The three-of-thirteen case, unchanged: a Rocket screen that also sets `x_button`
    was already reaching POPUP through `in_overlay`, and must keep doing so.

    `close_xy=None` is what makes this a test of the overlay branch rather than a second
    test of the new one. With a coordinate present the new branch answers first and this
    assertion passes with the overlay branch deleted outright - verified by deleting it,
    which leaves every other test in this file green. With no coordinate
    `rocket_exit_screen` declines and only `in_overlay` can supply the answer."""
    assert fsm.desired_state(rocket(x_button=True, close_xy=None), ctx()) is BotState.POPUP


def test_the_escape_fires_from_targeting_as_well_as_scanning():
    """Both arms of the source set, because only one of them was covered: mutating the
    branch to `ctx.state is BotState.SCANNING` alone left all 805 tests green. TARGETING
    is the state a tap that opened a Rocket screen instead of an encounter lands in, which
    is the live path this route is most often reached by."""
    for state in (BotState.SCANNING, BotState.TARGETING):
        assert fsm.desired_state(rocket(), ctx(state=state)) is BotState.POPUP, state


def test_rockets_off_presses_the_button_it_located():
    """End to end over two ticks, because the press is the point and a Transition alone
    is not one: the route hands the screen to POPUP, and POPUP taps the coordinate
    `find_close_button` returned - never a remembered one."""
    c = ctx()
    tr = kinds(fsm.step(rocket(close_xy=(0.50, 0.92)), c), Transition)
    assert tr and tr[0].to is BotState.POPUP
    c.state, c.state_since = BotState.POPUP, c.now
    taps = kinds(fsm.step(rocket(close_xy=(0.50, 0.92)), c), Tap)
    assert taps and (taps[0].x, taps[0].y) == (0.50, 0.92)


def test_x_button_reading_false_no_longer_blocks_the_escape():
    """The 2-of-13 gained, and the whole defect. `x_button` is a colour fraction over a
    fixed ROI; `find_close_button` is a shaped-contour search that returns a coordinate. On
    the labelled Rocket frames the finder locates a genuine X on 2 frames where `x_button`
    reads False, and `in_overlay` - which consults only `x_button` - is False on every one
    of them, so the pre-existing overlay branch could never reach them."""
    o = rocket(x_button=False, close_xy=(0.50, 0.92))
    assert not o.in_overlay, "meant to exercise the frames in_overlay cannot see"
    assert fsm.desired_state(o, ctx()) is BotState.POPUP


def test_a_rocket_screen_with_nothing_located_degrades_to_the_existing_ladder():
    """"Buttons are LOCATED, never assumed" is not suspended for this route. With no
    coordinate there is nothing to press, so the route declines and the run gets exactly
    the recovery ladder it had before - which is also what the two GruntDialogue frames
    in the corpus get, since neither carries a findable X."""
    o = rocket(close_xy=None)
    assert fsm.desired_state(o, ctx()) is None
    c = ctx(state=BotState.SCANNING, last_map_ts=0.0)
    tr = kinds(fsm.step(o, c), Transition)
    assert tr and tr[0].to is BotState.RECOVERING


def test_the_affirmative_pill_is_never_mistaken_for_an_x():
    """The defect the corpus exposed. `find_close_button` is hue-based and the affirmative
    pill is a green-to-teal gradient, so on the ChooseParty layout the pill's own
    right-hand cap is returned as the close button: (0.627, 0.876) against the pill's
    (0.545, 0.875) on all five labelled frames. Routing that to POPUP presses USE THIS
    PARTY, which STARTS the Team GO Rocket fight the operator turned off - on a run where
    `BotState.ROCKET` is unreachable, so nothing would then drive the battle. A close
    button sharing the pill's row is the pill; see `fsm.CLOSE_PILL_MIN_DY`."""
    pill_cap = rocket(close_xy=(0.627, 0.876), pill_xy=(0.545, 0.875))
    assert not fsm.rocket_exit_screen(pill_cap, OFF)
    assert fsm.desired_state(pill_cap, ctx()) is None
    out = fsm.step(pill_cap, ctx())
    assert not kinds(out, Tap), "would have pressed the button that starts the fight"


def test_a_real_x_below_the_pill_is_still_believed():
    """The other side of that threshold, so the veto cannot pass by refusing everything:
    the tightest genuine pairing in the corpus is GruntBattleButton's X at dy 0.0766 below
    its BATTLE pill, which is 3.8x `fsm.CLOSE_PILL_MIN_DY` and must still route."""
    real_x = rocket(close_xy=(0.500, 0.890), pill_xy=(0.492, 0.813))
    assert fsm.rocket_exit_screen(real_x, OFF)
    assert fsm.desired_state(real_x, ctx()) is BotState.POPUP


def test_the_escape_is_bounded_by_popups_own_timeout():
    """Bounded, not hanging: POPUP owns the screen for `popup_timeout` (4s) and no longer.
    A rocket screen that will not close hands back to RECOVERING on schedule, so the
    ladder underneath keeps running rather than being replaced by this route."""
    c = ctx(state=BotState.POPUP, now=DEFAULT.timings.popup_timeout + 1.0)
    tr = kinds(fsm.step(rocket(), c), Transition)
    assert tr and tr[0].to is BotState.RECOVERING


def test_the_escape_never_fires_from_recovering():
    """RECOVERING is excluded from the source set on purpose. The stuck watchdog and the
    app-restart ladder both live in `Recovering.on_timeout`, which is reached only after
    6s of uninterrupted RECOVERING; a route that pulled RECOVERING into POPUP would reset
    that clock on every frame that happened to locate an X - 16 of 720 in the measured
    window - and could starve the ladder indefinitely, trading a 90s wedge the restart
    eventually cleared for one that nothing clears."""
    assert fsm.desired_state(rocket(x_button=True), ctx(state=BotState.RECOVERING)) is None


def test_the_watchdog_ladder_still_reaches_a_restart_and_then_a_halt():
    """The other half of that claim, stated as behaviour rather than as an absence: on the
    very screen this fix is about, a stuck RECOVERING still escalates exactly as it did -
    a restart while the budget lasts, a halt once it is spent."""
    o = rocket(x_button=True)
    c = ctx(state=BotState.RECOVERING, now=1000.0, last_map_ts=0.0)
    assert kinds(fsm.step(o, c), RestartApp)
    spent = ctx(state=BotState.RECOVERING, now=1000.0, last_map_ts=0.0,
                app_restarts=DEFAULT.max_app_restarts)
    assert kinds(fsm.step(o, spent), Halt)


def test_the_map_still_outranks_the_escape():
    """Unchanged precedence: a confirmed map ends everything, and `rocket_screen`'s own
    map veto means a frame that reads as the map is not a Rocket screen in the first
    place."""
    on_map = obs(on_map=True, screen="Overworld", conf=0.99, close_xy=(0.50, 0.92))
    assert not fsm.rocket_exit_screen(on_map, OFF)
    assert fsm.desired_state(on_map, ctx(state=BotState.RECOVERING)) is BotState.SCANNING


def test_the_exit_confirmation_dialog_is_still_answered_with_back_not_a_tap():
    """Pokemon GO's own "Do you want to exit Pokemon GO?" dialog classifies as Rocket. Its
    OK button sits close enough to a Rocket screen's controls that a coordinate response
    risks quitting the game, which is why `interrupts` answers it with a coordinate-free
    BACK. `rocket_screen`'s exit_dialog veto carries into this route unchanged, so the new
    branch cannot turn that dialog into a tap."""
    o = obs(screen="Rocket", conf=0.99, close_xy=(0.50, 0.92), exit_dialog=True)
    assert not fsm.rocket_exit_screen(o, OFF)
    out = fsm.step(o, ctx())
    assert kinds(out, Back) and not kinds(out, Tap) and not kinds(out, Transition)


# ------------------------------------------------- rockets on: nothing may change at all

def test_rockets_on_still_routes_a_rocket_screen_to_rocket():
    """The property the ordering in `desired_state` exists to protect: a Rocket screen
    carries its own X, so a screen with one located must still be FOUGHT, never closed."""
    o = rocket(x_button=True, close_xy=(0.50, 0.89))
    assert fsm.desired_state(o, ctx(cfg=DEFAULT)) is BotState.ROCKET


def test_rockets_on_is_never_stolen_by_the_new_route():
    """Stated against the predicate itself, so it holds for every frame rather than for
    the one above: with rockets on there is no frame at all this route will claim."""
    for x_button in (False, True):
        for close_xy in (None, (0.50, 0.89), (0.50, 0.92)):
            for pill_xy in (None, (0.50, 0.89), (0.545, 0.733)):
                o = rocket(x_button=x_button, close_xy=close_xy, pill_xy=pill_xy)
                assert not fsm.rocket_exit_screen(o, DEFAULT), (x_button, close_xy, pill_xy)


def test_rockets_on_mid_fight_the_rocket_hold_still_answers_first():
    """A live fight is off-map and looks like a run of encounter and overlay screens; the
    rocket-hold branch returns None outright while `rocket_recent` holds, which is what
    keeps every later branch - including this one - from preempting it."""
    c = ctx(cfg=DEFAULT, state=BotState.ROCKET, now=100.0, last_rocket_ts=100.0)
    post_fight = obs(screen="Menu", conf=0.86, x_button=True, close_xy=(0.50, 0.885))
    assert fsm.desired_state(post_fight, c) is None


def test_a_setting_flipped_mid_fight_does_not_steal_the_screen():
    """The one way rockets-off and state ROCKET can coexist: a per-account profile flips
    `fight_rockets` while a fight is under way. The hold still answers first, and ROCKET
    is absent from the new route's source set as well, so the fight in progress is left
    alone and ends through `Rocket`'s own timeout rather than by having its grunt dialogue
    closed underneath it."""
    c = ctx(state=BotState.ROCKET, now=100.0, last_rocket_ts=100.0)
    assert fsm.desired_state(rocket(x_button=True, close_xy=(0.50, 0.89)), c) is None


# ------------------------------------------------------------------------------- purity

def test_the_route_writes_nothing_to_the_context():
    """The FSM is pure: (Observation, Context) -> list[Effect]. Only the runner mutates
    the context, which is what keeps a dry run and a live run on the same trajectory.
    Every tick this fix can produce, each against its own context: the routing decision,
    the POPUP tap it leads to, and the timeout that hands the screen back."""
    cases = [
        (rocket(x_button=True), ctx()),
        (rocket(close_xy=None), ctx(state=BotState.SCANNING, last_map_ts=0.0)),
        (rocket(), ctx(state=BotState.POPUP, now=1.0)),
        (rocket(), ctx(state=BotState.POPUP, now=DEFAULT.timings.popup_timeout + 1.0)),
        (rocket(x_button=True), ctx(state=BotState.RECOVERING, now=1000.0, last_map_ts=0.0)),
    ]
    for o, c in cases:
        before = copy.deepcopy(c.__dict__)
        fsm.step(o, c)
        assert c.__dict__ == before, c.state


# --------------------------------------------------------- the frames this was measured on

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "datasets" / "state_v3"
#: The corpus the DEPLOYED classifier was trained on, whose `Rocket` directory is the
#: ground truth for "what the bot will call a Rocket screen".
CLS5 = ROOT / "datasets" / "state_cls5"

#: The fine-grained classes that make up that `Rocket` label. Four, not two: the 5-class
#: corpus files GruntBattleButton, GruntDialogue, ChooseParty and ExitTrainerBattle
#: together under `Rocket`, and `test_the_slice_is_exactly_the_class_the_classifier_emits`
#: pins that equivalence by basename so this list cannot drift from it. state_v3 is read
#: rather than state_cls5 only because it keeps the finer label, which is what lets the
#: measurements below say WHICH screens escape and which do not.
ROCKET_CLASSES = ("GruntBattleButton", "GruntDialogue", "ChooseParty", "ExitTrainerBattle")


def _rocket_frames():
    import cv2
    out = []
    for split in ("train", "valid", "test"):
        for cls in ROCKET_CLASSES:
            d = CORPUS / split / cls
            if not d.exists():
                continue
            for p in sorted(d.glob("*.png")):
                im = cv2.imread(str(p))
                if im is not None:
                    out.append((cls, p.name, im))
    return out


def _observe(im):
    """One labelled frame as the bot would see it, except for the class label.

    Every optical field - `x_button`, `close_button_xy`, `map_ball`, `exit_dialog` - comes
    from the real pixels through the same `Perceptor.observe` the runner uses. Only the
    screen label is supplied, from the directory the frame is filed under, because no
    classifier is loaded in a test process; what the classifier itself gets right is
    measured separately in tests/test_perception_golden.py.
    """
    from pogobot.frames import Frame
    from pogobot.perception import Perceptor
    o = Perceptor(DEFAULT).observe(Frame(seq=1, ts=0.0, bgr=im), run_detector=False)
    return dataclasses.replace(o, screen=ScreenGuess("Rocket", 0.95))


@pytest.fixture(scope="module")
def corpus():
    frames = _rocket_frames()
    if not frames:
        pytest.skip("datasets/state_v3 not present (gitignored)")
    return [(cls, name, _observe(im)) for cls, name, im in frames]


def test_the_slice_is_exactly_the_class_the_classifier_emits(corpus):
    """The denominator, pinned to ground truth rather than to a hand-kept list.

    Every measurement below is quoted as "N of 13", and the 13 are only the right frames
    if `ROCKET_CLASSES` really is the whole of what the deployed 5-class model calls
    `Rocket`. Compared by basename, since the two exports carry the same files under
    different labels. The first version of this file counted 7 - GruntBattleButton and
    GruntDialogue only - which silently excluded the ChooseParty frames where the fix went
    wrong, so a wrong denominator here is not a cosmetic problem."""
    if not CLS5.exists():
        pytest.skip("datasets/state_cls5 not present (gitignored)")
    deployed = {p.name for p in CLS5.rglob("*.png") if p.parent.name == "Rocket"}
    assert deployed, "state_cls5 has no Rocket class"
    assert {name for _, name, _ in corpus} == deployed
    assert len(corpus) == 13


def test_every_labelled_rocket_frame_carrying_a_real_x_now_escapes(corpus):
    """The fix, measured on the frames the diagnosis was measured on. Five of the thirteen
    carry an X `find_close_button` can locate and `CLOSE_PILL_MIN_DY` believes, and all
    five now route to POPUP - where before, only the three that also set `x_button` did.
    All five are GruntBattleButton, the one labelled layout that draws a real X."""
    routed = {name for _, name, o in corpus
              if fsm.desired_state(o, ctx()) is BotState.POPUP}
    classes = {cls for cls, name, _ in corpus if name in routed}
    assert len(routed) == 5
    assert classes == {"GruntBattleButton"}


def test_the_frames_the_old_routing_could_not_see_are_exactly_the_ones_gained(corpus):
    """Red-green: `in_overlay` is what the pre-existing overlay branch consults, and the
    two frames it reads False on while a real X is nonetheless located are precisely the
    two this change rescues. Asserted as a set difference so it cannot pass by
    coincidence."""
    gained = {name for _, name, o in corpus
              if not o.in_overlay and fsm.desired_state(o, ctx()) is BotState.POPUP}
    assert len(gained) == 2
    for _, name, o in corpus:
        if name in gained:
            assert not o.x_button.value and o.close_button_xy is not None


def test_the_choose_party_frames_are_refused_because_their_x_is_the_battle_button(corpus):
    """The defect the wider corpus exposed, on the real pixels. On all five ChooseParty
    frames `find_close_button` returns the affirmative pill's own gradient cap, one row
    from `find_action_pill`'s answer - so routing them to POPUP would press USE THIS PARTY
    and START the fight the operator turned off, on a run where `BotState.ROCKET` is
    unreachable and nothing would drive the battle. They get no route and keep the
    recovery ladder, which is the honest answer for a screen whose only located control
    commits to the thing being declined."""
    party = [(name, o) for cls, name, o in corpus if cls == "ChooseParty"]
    assert len(party) == 5
    for name, o in party:
        assert o.close_button_xy is not None, name
        assert o.action_pill_xy is not None, name
        dy = abs(o.action_pill_xy[1] - o.close_button_xy[1])
        assert dy < fsm.CLOSE_PILL_MIN_DY, (name, dy)
        assert not fsm.rocket_exit_screen(o, OFF), name
        assert fsm.desired_state(o, ctx()) is None, name
        assert not kinds(fsm.step(o, ctx()), Tap), name


def test_the_threshold_has_margin_on_both_sides_of_the_real_frames(corpus):
    """`CLOSE_PILL_MIN_DY` is a measured constant, so the measurement is asserted rather
    than described: every frame it must REFUSE and every frame it must KEEP, with the gap
    between them read off the corpus. Nothing sits between 0.0003 and 0.0766, which is
    what makes 0.02 a threshold rather than a guess."""
    pairs = [(cls, abs(o.action_pill_xy[1] - o.close_button_xy[1])) for cls, _, o in corpus
             if o.close_button_xy is not None and o.action_pill_xy is not None]
    refuse = [dy for cls, dy in pairs if cls == "ChooseParty"]
    keep = [dy for cls, dy in pairs if cls != "ChooseParty"]
    assert len(refuse) == 5 and len(keep) == 5
    assert max(refuse) < fsm.CLOSE_PILL_MIN_DY < min(keep)
    assert max(refuse) < 0.001 and min(keep) > 0.07


def test_the_frames_with_no_route_are_left_to_the_ladder(corpus):
    """The eight this route deliberately declines, by the reason it declines each: two
    GruntDialogue carry no signal and no findable X at all, five ChooseParty carry only
    the button that would start the fight, and the ExitTrainerBattle frame is Pokemon GO's
    own exit-confirmation dialog, which `rocket_screen`'s veto keeps on a coordinate-free
    BACK. None of them is a guessed tap, which is the point."""
    stranded = [(cls, name, o) for cls, name, o in corpus
                if fsm.desired_state(o, ctx()) is None]
    assert len(stranded) == 8
    by_class = collections.Counter(cls for cls, _, _ in stranded)
    assert by_class == {"ChooseParty": 5, "GruntDialogue": 2, "ExitTrainerBattle": 1}
    for cls, name, o in stranded:
        if cls == "GruntDialogue":
            assert o.close_button_xy is None and not o.x_button.value, name
        elif cls == "ExitTrainerBattle":
            assert o.exit_dialog.value, name


def test_with_rockets_on_every_labelled_rocket_frame_routes_as_it_always_did(corpus):
    """The no-regression half, on all thirteen frames: with the setting on, the twelve
    Rocket screens are still FOUGHT - including the five carrying a located X and the five
    whose located point is the BATTLE button - and the exit-confirmation frame still gets
    no route, exactly as before this change. Nothing here consults
    `rocket_exit_screen` at all, which is what makes rockets-on untouched."""
    for cls, name, o in corpus:
        want = None if cls == "ExitTrainerBattle" else BotState.ROCKET
        assert fsm.desired_state(o, ctx(cfg=DEFAULT)) is want, name
