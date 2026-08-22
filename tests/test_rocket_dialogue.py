"""The last blind coordinate: ROCKET's dialogue-advance tap.

A Team GO Rocket grunt dialogue advances on a tap ANYWHERE - there is no button to locate -
so `Rocket.step` has one press that cannot follow this codebase's central rule that buttons
are LOCATED, never assumed (`perception.find_close_button`, `fsm.Recovering`). Its only
guard was `screen_min_conf`, 0.60, which is the same bar the SPONSORED interstitial clears
at 0.62 - the screen that opened the advertiser's site in Chrome and cost three consecutive
live runs.

It is not a rare branch. Measured over logs/trace.jsonl: 210,719 ROCKET frames produced
7,507 taps from it.

Raising its bar the way `rocket_pill_min_conf` raises the affirmative branch's was measured
and rejected, and that measurement is the reason this fix reads the frame instead. Over the
same trace, restricted to frames classified Rocket while the machine is in ROCKET:

    frames where a pill IS located (the affirmative branch's domain)   98.1% reach 0.90
    frames where NO pill is located (this branch's domain)             15.5% reach 0.90

so a 0.90 bar costs the affirmative branch nothing and would refuse 84.8% of real dialogue
advances. A bot that cannot finish a fight it has entered is worse off than one that
occasionally taps an advertisement. Nor does a bar set just above the ad work: this
branch's live taps run min 0.600, p05 0.649, median 0.78 against the ad's 0.62, so a 0.63
bar refuses 2.66% of real advances and rests on a sample of one advertisement.

So the separation comes from what the frame SHOWS. See `fsm.rocket_dialogue_screen` for the
predicate and `fsm.ROCKET_DIALOGUE_TAP` for why the coordinate itself is safe - which is a
claim about the geometry of the three finders, proved constructively below, not about the
coordinate having worked before.
"""
from __future__ import annotations

import collections
import copy
import dataclasses
import pathlib

import cv2
import numpy as np
import pytest

from pogobot import fsm
from pogobot import perception as P
from pogobot.config import DEFAULT
from pogobot.effects import BotState, Back, Tap
from pogobot.observation import ScreenGuess
from tests.factories import obs

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "screens"
CORPUS = ROOT / "datasets" / "state_v3"
#: The corpus the DEPLOYED classifier was trained on; its `Rocket` directory is the ground
#: truth for "what the bot will call a Rocket screen". Used only to pin the denominator.
CLS5 = ROOT / "datasets" / "state_cls5"

#: The four fine-grained classes the deployed 5-class model files under `Rocket`. Same list
#: and same reasoning as tests/test_rocket_escape.py, pinned against CLS5 below so it
#: cannot drift: a wrong denominator here would silently exclude the very screens the blind
#: tap goes wrong on.
ROCKET_CLASSES = ("GruntBattleButton", "GruntDialogue", "ChooseParty", "ExitTrainerBattle")

#: The classifier confidence the corpus frames are read at. NOT 1.00, which is what the
#: deployed model returns on frames it was trained on, because at 1.00 the affirmative
#: branch takes every frame carrying a pill and the dialogue branch is never reached - the
#: test would then assert nothing about the branch it is named for. 0.78 is the measured
#: live MEDIAN for this branch's own domain (frames classified Rocket, in ROCKET, with no
#: pill located), and it sits below `rocket_pill_min_conf`, so it is both the honest and
#: the load-bearing value: every frame below is one the affirmative branch has declined and
#: the blind tap used to own.
LIVE_MEDIAN_CONF = 0.78


def ctx(cfg=DEFAULT, state=BotState.ROCKET, now=100.0, **kw):
    c = fsm.Context(cfg=cfg, state=state, state_since=now - 0.5, now=now)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def taps(effects):
    return [e for e in effects if isinstance(e, Tap)]


def rocket_step(o, c=None):
    """The handler branch itself, not the whole tick.

    `fsm.step` runs `interrupts` first, and on an exit-confirmation frame that answers with
    BACK before any handler is consulted - so driving these claims through `fsm.step` would
    measure the interrupt rather than the branch under test. The integration path is
    exercised separately below, including the ticks where the interrupt's pacing gate
    withholds and the handler really does run.
    """
    return fsm.HANDLERS[BotState.ROCKET].step(o, c or ctx())


def old_branch_would_tap(o, cfg=DEFAULT) -> bool:
    """Exactly the condition this change replaced, kept executable so every "now refused"
    claim below is a red-green statement rather than a description: a frame is only
    evidence of a fix if the code before it would have tapped."""
    return o.screen.is_("Rocket", min_conf=cfg.screen_min_conf)


# ------------------------------------------------- the coordinate: what makes it safe now

def _mint_disc(cx, cy, frac=0.10, w=590, h=1280):
    """A round mint blob at a chosen point, in the band `find_close_button` and
    `promo_save_button` both search for. Built in HSV and round-tripped through real cv2
    conversion, the same way tests/test_exit_dialog.py builds its synthetic frames."""
    im = np.full((h, w, 3), 40, np.uint8)
    px = np.zeros((1, 1, 3), np.uint8)
    px[0, 0] = (85, 180, 220)
    colour = tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])
    cv2.circle(im, (int(cx * w), int(cy * h)), int(w * frac / 2), colour, -1)
    return im


def _green_pill(cy, w=590, h=1280):
    """A wide green pill with a white label, at a chosen row."""
    im = np.full((h, w, 3), 40, np.uint8)
    pw, ph = int(w * 0.55), int(h * 0.045)
    x0, y0 = (w - pw) // 2, int(cy * h - ph / 2)
    body = np.full((ph, pw, 3), 0, np.uint8)
    body[:, :] = (85, 180, 220)
    im[y0:y0 + ph, x0:x0 + pw] = cv2.cvtColor(body, cv2.COLOR_HSV2BGR)
    cv2.putText(im, "ADVANCE", (x0 + int(pw * 0.2), y0 + int(ph * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    return im


def test_no_close_button_the_bot_can_locate_can_ever_sit_under_this_tap():
    """Half of why the coordinate is safe, proved by construction rather than asserted.

    `find_close_button` searches only the bottom of the frame, so a control drawn exactly
    under `ROCKET_DIALOGUE_TAP` is invisible to it - which is why a located close button is
    deliberately NOT a veto in `rocket_dialogue_screen`. The same disc lower down IS
    located, so this passes because of the ROI and not because the disc is unfindable."""
    x, y = fsm.ROCKET_DIALOGUE_TAP
    assert P.find_close_button(_mint_disc(x, y), DEFAULT) is None
    for lower in (0.80, 0.89):
        assert P.find_close_button(_mint_disc(x, lower), DEFAULT) is not None, lower


def test_no_promo_save_control_can_ever_sit_under_this_tap():
    """The other structurally-unreachable finder. `promo_save_button` searches y 0.82-0.95
    at x >= 0.72 only; the tap is outside that window on both axes."""
    _, y = fsm.ROCKET_DIALOGUE_TAP
    assert P.promo_save_button(_mint_disc(0.866, y, frac=0.14), DEFAULT) is None
    assert P.promo_save_button(_mint_disc(0.866, 0.884, frac=0.14), DEFAULT) is not None


def test_a_pill_however_CAN_sit_exactly_under_this_tap_which_is_why_it_is_vetoed():
    """The reason the pill veto is load-bearing and not merely defensive: unlike the other
    two finders, `find_action_pill` searches from y 0.45 down, so it can and does return
    the exact point this tap lands on. Every control the tap could press is therefore a
    pill, and refusing a located pill refuses all of them."""
    x, y = fsm.ROCKET_DIALOGUE_TAP
    found = P.find_action_pill(_green_pill(y), DEFAULT)
    assert found is not None
    assert abs(found[0] - x) < 0.02 and abs(found[1] - y) < 0.01, found


def test_the_tap_point_is_unchanged_so_this_is_a_guard_and_not_a_relocation():
    """Stated so the change cannot be mistaken for one that moved the coordinate: the point
    is the same one the bot has always used. What is new is that the frame must be vetted
    before it is pressed."""
    assert fsm.ROCKET_DIALOGUE_TAP == (0.50, 0.62)


# -------------------------------------------------------------- each veto, on its own

def dialogue(**kw):
    """A frame the classifier reads as a Rocket screen at this branch's live median.

    `screen="Rocket"` above `screen_min_conf` is what makes this genuinely off-map:
    `obs(on_map=False)` alone does not, because the factory defaults `screen="Overworld"`
    at 0.99, which satisfies `Observation.on_map` on its own."""
    kw.setdefault("screen", "Rocket")
    kw.setdefault("conf", LIVE_MEDIAN_CONF)
    return obs(**kw)


def test_a_bare_grunt_dialogue_is_still_advanced():
    """The disease must not be traded for something worse. A dialogue frame carries no
    locatable control at all, and it is still tapped, at the same point as before."""
    o = dialogue()
    assert fsm.rocket_dialogue_screen(o, DEFAULT)
    t = taps(rocket_step(o))
    assert t and (t[0].x, t[0].y) == fsm.ROCKET_DIALOGUE_TAP


def test_a_located_pill_refuses_the_blind_tap():
    """The veto that covers ChooseParty's USE THIS PARTY and the advertisement's LEARN
    MORE, and the only one that answers a control the tap could actually land on."""
    o = dialogue(pill_xy=(0.545, 0.875))
    assert old_branch_would_tap(o), "not a red-green case: the old branch declined anyway"
    assert not fsm.rocket_dialogue_screen(o, DEFAULT)
    assert not taps(rocket_step(o))


def test_the_promo_save_control_refuses_the_blind_tap_on_its_own():
    """Independent of the pill veto, because on an advertisement no coordinate is
    defensible - the creative is a link. Asserted with no pill present so it is this term
    and not the one above doing the work."""
    o = dialogue(promo_xy=(0.866, 0.884))
    assert o.action_pill_xy is None
    assert old_branch_would_tap(o)
    assert not fsm.rocket_dialogue_screen(o, DEFAULT)
    assert not taps(rocket_step(o))


def test_the_exit_confirmation_dialog_refuses_the_blind_tap_on_its_own():
    """The third term, and the only defence for a card nothing locates. On the corpus frame
    the "Exit the Trainer Battle?" card spans y 0.397-0.626 with NO at 0.572-0.585 and YES
    at 0.481-0.542; the tap lands at 0.620, inside the card. No pill and no close button is
    located there, so `exit_dialog` is the only signal that can refuse it."""
    o = dialogue(exit_dialog=True)
    assert o.action_pill_xy is None and o.close_button_xy is None
    assert old_branch_would_tap(o)
    assert not fsm.rocket_dialogue_screen(o, DEFAULT)
    assert not taps(rocket_step(o))


def test_a_located_close_button_alone_does_not_refuse_the_tap():
    """The deliberate non-veto, pinned so it reads as a decision rather than an oversight.
    The tap provably cannot reach a located X (see the geometry tests above), and refusing
    on one would cost a further 94 live taps - 1.25% - on frames where nothing has been
    shown to be at risk."""
    o = dialogue(close_xy=(0.50, 0.92))
    assert fsm.rocket_dialogue_screen(o, DEFAULT)
    assert taps(rocket_step(o))


def test_the_ordinary_confidence_bar_still_applies_underneath():
    """The guard is added to the old test, not substituted for it: a screen the classifier
    does not call Rocket at all is still refused."""
    assert not fsm.rocket_dialogue_screen(dialogue(conf=0.50), DEFAULT)
    assert not fsm.rocket_dialogue_screen(dialogue(screen="Menu", conf=0.99), DEFAULT)


def test_the_affirmative_branch_is_untouched_by_this_change():
    """Scope. The pill branch keeps its own bar and its own behaviour - on a confident
    Rocket screen with a pill located, the bot still presses the pill. This change only
    governs what happens when that branch declines."""
    o = dialogue(conf=0.99, pill_xy=(0.545, 0.875))
    t = taps(rocket_step(o))
    assert t and (t[0].x, t[0].y) == (0.545, 0.875)


# ------------------------------------------------- the advertisement, from the fixture

@pytest.fixture(scope="module")
def ad():
    """The committed SPONSORED interstitial, read as the bot reads it.

    Every optical field - the pill, the close button, the save control - comes from the real
    pixels through the same `Perceptor.observe` the runner uses. Only the screen label is
    supplied, at the measured 0.62 the deployed classifier returns for this fixture, since
    no classifier is loaded in this process; that the real model reads 0.62 is pinned
    against the model itself in tests/test_sponsored_ad.py.
    """
    from pogobot.frames import Frame
    im = cv2.imread(str(FIXTURES / "sponsored_ad.png"))
    assert im is not None, "the committed ad fixture is missing"
    o = P.Perceptor(DEFAULT).observe(Frame(seq=1, ts=0.0, bgr=im), run_detector=False)
    return dataclasses.replace(o, screen=ScreenGuess("Rocket", 0.62))


def test_the_fixture_still_carries_both_tells(ad):
    """The trap itself, pinned so nobody can call this screen harmless: the ad puts a pill
    where the affirmative sits AND offers to save itself, and it clears the ordinary
    confidence bar while failing the affirmative branch's."""
    assert ad.action_pill_xy is not None
    assert ad.promo_save_xy is not None and ad.promo_save_xy[0] > 0.8
    assert DEFAULT.screen_min_conf <= ad.screen.conf < DEFAULT.rocket_pill_min_conf


def test_the_advertisement_is_never_tapped_by_the_dialogue_branch(ad):
    """The assertion that matters. Before this change the branch tapped (0.50, 0.62) into
    the ad's own creative - above LEARN MORE, but inside a card that is a link."""
    assert old_branch_would_tap(ad)
    assert not fsm.rocket_dialogue_screen(ad, DEFAULT)
    assert not taps(rocket_step(ad))


def test_the_advertisement_is_refused_by_two_independent_terms(ad):
    """Defence in depth, stated as behaviour: strip either tell from the frame and the
    other still refuses it. A future creative that drops the green pill, or one whose save
    control falls outside the locator's band, is still not tapped."""
    without_promo = dataclasses.replace(ad, promo_save_xy=None)
    assert not fsm.rocket_dialogue_screen(without_promo, DEFAULT), "pill veto did not hold"
    without_pill = dataclasses.replace(ad, action_pill_xy=None)
    assert not fsm.rocket_dialogue_screen(without_pill, DEFAULT), "promo veto did not hold"
    neither = dataclasses.replace(ad, promo_save_xy=None, action_pill_xy=None)
    assert fsm.rocket_dialogue_screen(neither, DEFAULT), (
        "with both tells removed the ad should be tapped again - if not, something else is "
        "refusing it and this fixture is not testing these guards")


def test_the_entry_veto_does_not_cover_a_machine_already_in_rocket(ad):
    """Why the veto had to be repeated inside the handler. `rocket_screen` refuses the ad,
    so `desired_state` never ROUTES to ROCKET on it - but a machine already in ROCKET is
    never asked: the rocket-hold branch answers None while a fight is recent, the state
    holds for `Rocket.timeout_s`, and `Rocket.step` runs on every tick regardless."""
    assert not fsm.rocket_screen(ad, DEFAULT)
    held = ctx(state=BotState.ROCKET, now=100.0, last_rocket_ts=100.0)
    assert fsm.desired_state(ad, held) is None, "nothing redirects the machine off this ad"
    assert fsm.HANDLERS[BotState.ROCKET] is fsm.HANDLERS[held.state]
    assert not taps(fsm.step(ad, held))


# ------------------------------------------------------------- the integration path

def test_the_exit_dialog_interrupt_is_still_the_primary_answer():
    """Unchanged precedence: BACK carries no coordinate at all, which is why it answers
    this dialog before any handler is consulted."""
    out = fsm.step(dialogue(exit_dialog=True), ctx())
    assert [e for e in out if isinstance(e, Back)] and not taps(out)


def test_the_handler_veto_covers_the_ticks_the_interrupt_withholds():
    """The layer beneath the interrupt. `interrupts` gates its repeat BACK on
    `exit_dialog_back`; on the ticks that gate withholds, the handler runs on the very same
    frame. At DEFAULT timings a second, unrelated thing also covers those ticks - see the
    test below - so this state is constructed rather than reached: the point is that the
    predicate refuses the card on its own merits, without help from either pacing gate."""
    c = ctx()
    c.last_action["back"] = c.now                      # BACK just fired; the gate is shut
    o = dialogue(exit_dialog=True)
    assert not [e for e in fsm.interrupts(o, c) if isinstance(e, Back)], (
        "the interrupt still fired; this test is not exercising the fall-through")
    assert not taps(fsm.step(o, c))


def _drive(o, cfg, secs=200.0):
    """The real tick loop, mirroring `Runner.apply`'s pacing bookkeeping: every applied
    effect stamps its budget, and the effects that move the UI open a settle window."""
    c = fsm.Context(cfg=cfg, state=BotState.ROCKET, state_since=0.0, now=0.0,
                    last_rocket_ts=0.0)
    dt, n_taps, n_backs = 1.0 / cfg.infer_fps, 0, 0
    for i in range(int(secs / dt)):
        c.now = i * dt
        c.last_rocket_ts = c.now                 # the rocket-hold branch stays fresh
        for e in fsm.step(o, c):
            if isinstance(e, (Tap, Back)):
                c.last_action[e.budget] = c.now
                c.settle_until = c.now + cfg.timings.ui_settle
            n_taps += isinstance(e, Tap)
            n_backs += isinstance(e, Back)
    return n_taps, n_backs


@pytest.mark.parametrize("ui_settle,back_gap", [
    (1.2, 1.0),     # the defaults: ui_settle > exit_dialog_back, so settle covers the gap
    (1.2, 1.3),     # the gate outlasts the settle window - the handler really does run
    (0.8, 1.0),     # the settle window closes first - likewise
])
def test_the_card_is_refused_however_the_two_pacing_constants_are_ordered(ui_settle, back_gap):
    """Why the `exit_dialog` term is not redundant with the interrupt.

    At default timings nothing reaches this branch on an exit card, but not for any reason
    anyone chose: `ui_settle` (1.2s) merely happens to exceed `exit_dialog_back` (1.0s),
    and the dialogue branch does not pass `ignore_settle`, so the settle window BACK opens
    swallows every tick the BACK gate withholds. Reorder those two plain tuning constants
    and the handler runs on the card - before this guard that meant 60 and 86 blind taps
    into a confirm dialog whose YES forfeits the fight. Safety must not depend on that
    coincidence, so it is asserted under all three orderings.
    """
    cfg = dataclasses.replace(DEFAULT, timings=dataclasses.replace(
        DEFAULT.timings, ui_settle=ui_settle, exit_dialog_back=back_gap))
    n_taps, n_backs = _drive(dialogue(exit_dialog=True), cfg)
    assert n_taps == 0, f"blind tap into the exit card at {ui_settle}/{back_gap}"
    assert n_backs > 0, "BACK must still be the primary answer"


def test_a_real_dialogue_still_advances_through_the_whole_tick():
    """End to end rather than at the branch: nothing above `Rocket.step` refuses a plain
    grunt dialogue, so the fight the bot entered still finishes."""
    t = taps(fsm.step(dialogue(), ctx()))
    assert t and (t[0].x, t[0].y) == fsm.ROCKET_DIALOGUE_TAP


def test_the_pacing_gate_is_untouched():
    """The budget, unchanged: one dialogue tap per `rocket_tap`."""
    c = ctx()
    c.last_action["rocket"] = c.now
    assert not taps(fsm.step(dialogue(), c))
    # A hair past the gap rather than exactly on it: `c.now + 1.6 - c.now` is 1.5999...
    # in binary floating point, so an exact advance lands just UNDER `ready`'s own
    # comparison and would test the arithmetic instead of the gate.
    c.now += DEFAULT.timings.rocket_tap + 0.01
    assert taps(fsm.step(dialogue(), c))


def test_a_refusal_leaves_the_budget_unstamped_so_the_next_frame_retries():
    """Why the guards are cheap. Only `Runner.apply` stamps `last_action`, and only for
    effects it actually applies - so a refused frame costs one frame (~125ms at the default
    infer_fps of 8), not a whole tap interval. A transient false pill on a real dialogue
    therefore delays the fight, it does not stall it."""
    c = ctx()
    assert not taps(rocket_step(dialogue(pill_xy=(0.545, 0.875)), c))
    assert "rocket" not in c.last_action
    assert taps(rocket_step(dialogue(), c)), "the very next frame must be free to act"


# --------------------------------------------------------------------------- purity

def test_no_handler_writes_the_context():
    """The FSM is pure: (Observation, Context) -> list[Effect]. Only the runner mutates the
    context, which is what keeps a dry run and a live run on the same trajectory. Every
    tick this change can produce, each against its own context."""
    cases = [
        (dialogue(), ctx()),
        (dialogue(pill_xy=(0.545, 0.875)), ctx()),
        (dialogue(promo_xy=(0.866, 0.884)), ctx()),
        (dialogue(exit_dialog=True), ctx()),
        (dialogue(close_xy=(0.50, 0.92)), ctx()),
        (dialogue(conf=0.99, pill_xy=(0.545, 0.875)), ctx()),
        (dialogue(), ctx(state=BotState.ROCKET, now=1000.0)),
    ]
    for o, c in cases:
        before = copy.deepcopy(c.__dict__)
        fsm.step(o, c)
        assert c.__dict__ == before, (o.screen, c.state)


def test_the_predicate_reads_nothing_but_the_frame_and_the_config():
    """`rocket_dialogue_screen` is a pure observation test, like `rocket_screen` and
    `rocket_exit_screen` beside it - it takes no Context, so it cannot consult or change
    run state, and the same frame always answers the same way."""
    o = dialogue()
    assert fsm.rocket_dialogue_screen(o, DEFAULT) is fsm.rocket_dialogue_screen(o, DEFAULT)


# --------------------------------------------------- the frames this was measured on

def _rocket_frames():
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


def _observe(im, conf):
    """One labelled frame as the bot would see it, except for the class label.

    Same construction as tests/test_rocket_escape.py: every optical field comes from the
    real pixels through `Perceptor.observe`, and only the screen label is supplied, from the
    directory the frame is filed under, because no classifier is loaded in a test process.
    What the classifier itself gets right is measured in tests/test_perception_golden.py.
    """
    from pogobot.frames import Frame
    o = P.Perceptor(DEFAULT).observe(Frame(seq=1, ts=0.0, bgr=im), run_detector=False)
    return dataclasses.replace(o, screen=ScreenGuess("Rocket", conf))


@pytest.fixture(scope="module")
def corpus():
    frames = _rocket_frames()
    if not frames:
        pytest.skip("datasets/state_v3 not present (gitignored)")
    return [(cls, name, _observe(im, LIVE_MEDIAN_CONF)) for cls, name, im in frames]


def test_the_slice_is_exactly_the_class_the_classifier_emits(corpus):
    """The denominator, pinned to ground truth rather than to a hand-kept list. Every count
    below is quoted as "N of 13", and those are only the right frames if `ROCKET_CLASSES` is
    the whole of what the deployed 5-class model calls `Rocket`."""
    if not CLS5.exists():
        pytest.skip("datasets/state_cls5 not present (gitignored)")
    deployed = {p.name for p in CLS5.rglob("*.png") if p.parent.name == "Rocket"}
    assert deployed, "state_cls5 has no Rocket class"
    assert {name for _, name, _ in corpus} == deployed
    assert len(corpus) == 13


def test_every_one_of_the_thirteen_used_to_be_tapped_blind(corpus):
    """The "before" column, so the "after" means something. At this branch's own live
    confidence the condition it replaced fires on all thirteen Rocket-class frames -
    including the five that would open a swap picker and the one that is a confirm card."""
    assert all(old_branch_would_tap(o) for _, _, o in corpus)


def test_after_the_guard_only_the_grunt_dialogue_frames_are_tapped(corpus):
    """The "after" column. Exactly the two frames that ARE a grunt dialogue keep the blind
    tap; the other eleven are refused. Asserted as a set of class names so it cannot pass by
    landing on the right count with the wrong frames."""
    tapped = {(cls, name) for cls, name, o in corpus if fsm.rocket_dialogue_screen(o, DEFAULT)}
    assert {cls for cls, _ in tapped} == {"GruntDialogue"}
    assert len(tapped) == 2


def test_each_refused_frame_names_the_term_that_refused_it(corpus):
    """Refusals attributed one by one, so no frame is refused for a reason nobody checked.
    Five ChooseParty and five GruntBattleButton carry a located pill; the single
    ExitTrainerBattle frame carries none and is refused by `exit_dialog` alone. No
    Rocket-class frame in the corpus offers to save itself - 0 of 13 - which is what makes
    the promo term a tell for the advertisement and not a tax on real fights."""
    refused = collections.Counter()
    for cls, name, o in corpus:
        if fsm.rocket_dialogue_screen(o, DEFAULT):
            continue
        assert o.promo_save_xy is None, f"{name} appeared to offer a save"
        if o.action_pill_xy is not None:
            refused["pill"] += 1
        elif o.exit_dialog.value:
            refused["exit_dialog"] += 1
        else:
            raise AssertionError(f"{name} refused with no term to explain it")
    assert refused == {"pill": 10, "exit_dialog": 1}


def test_the_choose_party_frames_no_longer_land_on_the_swap_row(corpus):
    """The trap the corpus exposed and the brief did not name. On all five ChooseParty
    frames (0.50, 0.62) falls inside the Recommended Battle Party card, on its "Tap to swap
    Pokemon" row - a tap there opens the swap picker and navigates away from the fight. The
    screen's real control is USE THIS PARTY, located at (0.545, 0.875) on every one of
    them, and pressing THAT is the affirmative branch's job, not this one's."""
    party = [(name, o) for cls, name, o in corpus if cls == "ChooseParty"]
    assert len(party) == 5
    for name, o in party:
        assert o.action_pill_xy is not None, name
        assert abs(o.action_pill_xy[0] - 0.545) < 0.01, name
        assert abs(o.action_pill_xy[1] - 0.875) < 0.01, name
        assert not fsm.rocket_dialogue_screen(o, DEFAULT), name
        assert not taps(rocket_step(o)), name


def test_the_exit_card_frame_is_refused_and_nothing_else_could_have_refused_it(corpus):
    """The frame that proves the `exit_dialog` term is load-bearing: on it, both other
    finders return None, so removing that term alone puts the blind tap back inside a
    confirm card whose NO sits 0.035 above the tap point and whose YES forfeits the
    fight."""
    frames = [(name, o) for cls, name, o in corpus if cls == "ExitTrainerBattle"]
    assert len(frames) == 1
    name, o = frames[0]
    assert o.exit_dialog.value, name
    assert o.action_pill_xy is None and o.promo_save_xy is None, name
    assert not fsm.rocket_dialogue_screen(o, DEFAULT), name
    without_exit = dataclasses.replace(o, exit_dialog=dataclasses.replace(o.exit_dialog,
                                                                         value=False))
    assert fsm.rocket_dialogue_screen(without_exit, DEFAULT), (
        "with the exit term gone this card should be tapped again - if not, some other "
        "term is refusing it and this frame is not testing that guard")


def test_the_two_dialogue_frames_carry_nothing_at_all_to_press(corpus):
    """Why the surviving frames are the right ones to survive: a grunt dialogue is the
    Rocket screen with no control on it. Neither of the two locates a pill, a close button
    or a save control, and neither raises `exit_dialog` - so there is nothing on either
    frame that this tap could be pressing."""
    dlg = [(name, o) for cls, name, o in corpus if cls == "GruntDialogue"]
    assert len(dlg) == 2
    for name, o in dlg:
        assert o.action_pill_xy is None, name
        assert o.close_button_xy is None, name
        assert o.promo_save_xy is None, name
        assert not o.exit_dialog.value, name
        t = taps(rocket_step(o))
        assert t and (t[0].x, t[0].y) == fsm.ROCKET_DIALOGUE_TAP, name


def test_a_confident_fight_still_reaches_its_affirmative_on_the_real_frames(corpus):
    """The other half of "do not break a fight the bot entered", on real pixels: read at the
    confidence the deployed classifier actually returns for them, the ten frames carrying a
    pill are pressed by the affirmative branch - so refusing them here costs the fight
    nothing, it just moves the press to the branch that locates its button."""
    confident = [(cls, name, dataclasses.replace(o, screen=ScreenGuess("Rocket", 1.0)))
                 for cls, name, o in corpus]
    pressed = 0
    for cls, name, o in confident:
        if o.action_pill_xy is None:
            continue
        t = taps(rocket_step(o))
        assert t, name
        assert (t[0].x, t[0].y) == o.action_pill_xy, name
        pressed += 1
    assert pressed == 10, pressed
