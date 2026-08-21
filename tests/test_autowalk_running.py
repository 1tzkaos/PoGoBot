"""AutoWalk that is ALREADY RUNNING when the ladder goes looking for it.

Tapping PGSharp's shortcut-menu "AutoWalk" entry while a route is active does NOT open the
"Auto-Generated GPX" setup dialog. PGSharp answers with a STOP dialog instead - dumped from
the device at the exact moment the ladder was failing and committed verbatim as
tests/fixtures/uiautomator/autowalk_stop_dialog.xml, whose only text nodes are "AutoWalk"
(alertTitle), "Stop/Pause AutoWalk?" (message), "PAUSE" (button2) and "STOP" (button1).

That is why `fsm.Switching._autowalk_dialog` failed on EVERY run. It found
`autowalk_dialog_open` False - the exact-title guard doing its job - pressed nothing, and
waited out the whole of `config.AutoWalk.budget_s`:

    22:12:17 WARNING the startup preflight could not start AutoWalk within 30s at the
                     autowalk_dialog step - playing anyway, with no route running

Two things follow, and they are what this module tests.

The safety property. `autowalk_ok_norm` on that dump was NOT None - button1 is STOP, at
[777,1224][969,1368], centre (0.8083, 0.5538) of 1080x2340 - and button2 is PAUSE. Pressing
either turns the user's route OFF, which is strictly worse than failing to turn it on. So
every assertion here is on the COORDINATE, never on a tap count: a ladder that taps
somewhere else entirely is fine, a ladder that taps there is not.

The consequence. The give-up left PGSharp's shortcut menu over the map. Measured at
1080x2340 its eight entries cover x 12-339, y 563-1331, so a tap aimed underneath lands on
one of them - "Settings" opens the PGSharp settings page (which wedged a live run) and
"Teleport" would move the player. Every exit from the ladder therefore has to leave that
menu shut, or say plainly that it could not.

Three layers, the same split tests/test_autowalk.py uses: `accounts.parse_dump` against the
real committed dump; the `fsm.Switching._autowalk_*` decisions through `fsm.step`; and the
real Runner against a widget model that behaves the way PGSharp does with a route already
running. The end-to-end tests are parametrized over the tick interval for the reason
tests/test_autowalk.py states - `Runner.apply` drops `ctx.accounts` after every actuation
taken while SWITCHING/PREFLIGHT, and only the throttled refresh puts it back, so a test
that ticks slower than the throttle never sees the gap the live loop lives in.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import replace

import pytest

from pogobot import fsm
from pogobot.accounts import parse_dump
from pogobot.config import Config
from pogobot.effects import Back, BotState, IntentOutcome, Note, SetFlag, Tap, Transition
from pogobot.observation import Tristate
from tests.factories import obs
from tests.test_autowalk import (FIX, STAR, TICK_INTERVALS, WH, _drive_until_scanning,
                                awctx, view)
from tests.test_preflight import pctx
from tests.test_switch_runner import ROSTER, make_runner
from tests.test_switching import panel

MAP = obs(on_map=True)
#: The stop dialog's own two buttons, from the committed dump's own bounds - STOP
#: [777,1224][969,1368] and PAUSE [561,1224][777,1368] on a 1080x2340 screen. Written out
#: as the arithmetic rather than as decimals so the fixture, not this file, is the source.
STOP_NORM = ((777 + 969) / 2 / 1080, (1224 + 1368) / 2 / 2340)
PAUSE_NORM = ((561 + 777) / 2 / 1080, (1224 + 1368) / 2 / 2340)
FORBIDDEN = (STOP_NORM, PAUSE_NORM)

BUDGET = Config().autowalk.budget_s
GRACE = Config().autowalk.close_grace_s


# ------------------------------------------------------------------ accounts.py: the dump

def test_the_stop_dialog_parses_as_already_running_and_not_as_the_setup_dialog():
    """The one claim the whole fix rests on. The same dump must NOT satisfy
    `autowalk_dialog_open`, or the ladder would go on to press a button on it."""
    v = view("autowalk_stop_dialog.xml")
    assert v.autowalk_running_dialog_open is True
    assert v.autowalk_dialog_open is False


def test_the_stop_dialogs_own_buttons_are_never_given_a_coordinate():
    """button1 here is STOP and button2 is PAUSE. button2 has never had a coordinate
    anywhere in accounts.py; button1 only ever did because it shares its generic Android id
    with the setup dialog's OK. Both bounds really are in the dump - asserted first - so
    this is a refusal to report them, not an absence of anything to report."""
    raw = (FIX / "autowalk_stop_dialog.xml").read_bytes()
    assert b"[777,1224][969,1368]" in raw and b"[561,1224][777,1368]" in raw
    v = view("autowalk_stop_dialog.xml")
    assert v.autowalk_ok_norm is None
    assert v.autowalk_continue_last_norm is None


def test_the_stop_dialog_is_recognised_by_its_own_message_not_its_title_or_its_buttons():
    """Matched the way `autowalk_dialog_open` is matched on "Auto-Generated GPX": on text
    this dialog owns. Change ONLY the message and the claim is withheld - even though the
    title still reads "AutoWalk" (the feature's name, which any future PGSharp dialog about
    it would carry too) and both generic button ids are still there. That the buttons come
    back is the proof the ids alone never carried the claim."""
    xml = (FIX / "autowalk_stop_dialog.xml").read_bytes().replace(
        b"Stop/Pause AutoWalk?", b"Something Else Entirely")
    assert b'text="AutoWalk"' in xml, "the title must survive, or nothing is being isolated"
    v = parse_dump(xml, WH)
    assert v.autowalk_running_dialog_open is False
    assert v.autowalk_ok_norm == pytest.approx(STOP_NORM)


@pytest.mark.parametrize("name", ["autowalk_dialog_continue_last.xml",
                                  "autowalk_dialog_ok_only.xml"])
def test_the_setup_dialog_is_never_read_as_already_running(name):
    """The other direction of the same separation: the real setup dialogs still parse
    exactly as they did, buttons and all."""
    v = view(name)
    assert v.autowalk_running_dialog_open is False
    assert v.autowalk_dialog_open is True
    assert v.autowalk_ok_norm is not None


@pytest.mark.parametrize("name", ["autowalk_menu.xml", "accounts_open.xml"])
def test_a_dump_with_no_dialog_at_all_claims_neither(name):
    v = view(name)
    assert v.autowalk_running_dialog_open is False
    assert v.autowalk_dialog_open is False


# ------------------------------------------------------------------ fsm: the decisions

def dangerous(**kw):
    """A view carrying the stop dialog AND both of its button coordinates.

    Deliberately MORE dangerous than any dump `accounts.parse_dump` can now produce - it
    withholds both (see the parse tests above), and PAUSE has never had a field at all. The
    FSM gate has to be sufficient on its own, so it is tested against the worst view the
    dataclass can express rather than against the one the parser hands it.
    """
    fields = dict(star_norm=STAR, autowalk_running_dialog_open=True,
                  autowalk_dialog_open=False, autowalk_ok_norm=STOP_NORM,
                  autowalk_continue_last_norm=PAUSE_NORM)
    fields.update(kw)
    return replace(panel(active="TrainerTwo"), **fields)


def taps(effects):
    return [e for e in effects if isinstance(e, Tap)]


def kinds(effects, t):
    return [e for e in effects if isinstance(e, t)]


def strings(effects):
    return ([e.text for e in effects if isinstance(e, Note)]
            + [e.reason for e in effects if hasattr(e, "reason")])


#: Whether the view ALSO claims to be the "Auto-Generated GPX" setup dialog. On a real dump
#: it never can - the two are matched on different text nodes - but the True case is the one
#: that actually bites: it is the only view from which a ladder that consulted the buttons
#: before the stop-dialog flag would reach a button at all.
BOTH_FLAGS = [False, True]


@pytest.mark.parametrize("phase", ["autowalk_menu", "autowalk_dialog", "autowalk_close"])
@pytest.mark.parametrize("elapsed", [0.0, BUDGET + 1.0, BUDGET + GRACE + 1.0])
@pytest.mark.parametrize("also_setup", BOTH_FLAGS)
def test_no_tap_ever_lands_on_pause_or_stop(phase, elapsed, also_setup):
    """THE safety property, asserted on the coordinate. Every phase that can be on screen
    while the stop dialog is up, at every point on the ladder's own clock - before its
    budget, past it, and past the cleanup allowance too, where the give-up paths live."""
    c = awctx(phase=phase, since=100.0,
              accounts=dangerous(autowalk_dialog_open=also_setup))
    c.now = c.switch_autowalk_since + elapsed
    effects = fsm.step(MAP, c)
    for t in taps(effects):
        for bad in FORBIDDEN:
            assert (t.x, t.y) != pytest.approx(bad), t.reason


@pytest.mark.parametrize("elapsed", [0.0, BUDGET + 1.0, BUDGET + GRACE + 1.0])
@pytest.mark.parametrize("also_setup", BOTH_FLAGS)
def test_the_ladder_never_presses_anything_at_all_on_that_dialog(elapsed, also_setup):
    """Stronger than the coordinate check and for the same reason the input field and the
    toggle groups are tested structurally: while the dialog is up the only tap this phase
    may emit is one that closes the shortcut menu - i.e. the star. Nothing else."""
    c = awctx(phase="autowalk_dialog", since=100.0,
              accounts=dangerous(autowalk_dialog_open=also_setup))
    c.now = c.switch_autowalk_since + elapsed
    for t in taps(fsm.step(MAP, c)):
        assert (t.x, t.y) == pytest.approx(STAR), t.reason


def test_back_is_what_dismisses_it_and_the_phase_moves_on_to_closing_the_menu():
    """BACK, verified on the device, and the only dismissal that carries no coordinate at
    all - so it cannot land on PAUSE or STOP however the dialog is laid out."""
    c = awctx(phase="autowalk_dialog", since=100.0, accounts=dangerous())
    effects = fsm.step(MAP, c)
    backs = kinds(effects, Back)
    assert len(backs) == 1
    assert backs[0].budget == "switch"          # the ladder's own pacing gate, not "back"
    assert not taps(effects)
    assert any(isinstance(e, SetFlag) and e.name == "switch_phase"
               and e.value == "autowalk_close" for e in effects)


def test_it_is_reported_as_already_running_not_as_a_failure():
    """SUCCESS: AutoWalk is running, which is the goal state. Said at INFO, in its own
    words, distinctly from the line a route this ladder actually STARTED would produce -
    and nowhere near the "did not complete in time" wording the give-up path uses."""
    c = awctx(phase="autowalk_dialog", since=100.0, accounts=dangerous())
    effects = fsm.step(MAP, c)
    notes = kinds(effects, Note)
    assert len(notes) == 1
    assert notes[0].level == "info"
    assert "already running" in notes[0].text
    assert "PAUSE" in notes[0].text and "STOP" in notes[0].text
    assert not any("did not complete in time" in s for s in strings(effects))
    assert not any(e.level == "warn" for e in notes)


def test_a_switch_says_which_account_the_route_is_already_running_for():
    c = awctx(phase="autowalk_dialog", since=100.0, accounts=dangerous())
    assert "TrainerTwo" in kinds(fsm.step(MAP, c), Note)[0].text


def test_a_preflight_names_no_account_when_it_finds_a_route_already_running():
    """`Preflight` reuses this method with no target at all, so "logged into None" and
    "already running for None" are both one f-string away - see `Switching._label`."""
    c = pctx(phase="autowalk_dialog", accounts=dangerous(), switch_autowalk_since=99.0)
    out = fsm.step(MAP, c)
    assert kinds(out, Back), "nothing was emitted, so no string was actually checked"
    for s in strings(out):
        assert "None" not in s, s


def test_it_acts_even_when_the_alertdialog_reads_off_map():
    """A genuine Android AlertDialog dims the window behind it, so this is the phase most
    likely to read as off-map on the exact tick it must act - the same reasoning the setup
    dialog's own off-map test states."""
    c = awctx(phase="autowalk_dialog", since=100.0, accounts=dangerous())
    assert kinds(fsm.step(obs(on_map=False, screen="Rocket", conf=0.99), c), Back)


def test_it_waits_out_the_ladders_own_pacing_gate_like_every_sibling():
    c = awctx(phase="autowalk_dialog", since=100.0, accounts=dangerous())
    c.last_action["switch"] = c.now
    assert fsm.step(MAP, c) == []


def test_the_stop_dialog_is_checked_before_any_button_is_even_considered():
    """On a real dump only one of the two dialogs can be true - they are matched on
    different text nodes - but if that ever stops holding, the ORDER decides whether the
    ladder backs out or presses something. It backs out."""
    c = awctx(phase="autowalk_dialog", since=100.0,
              accounts=dangerous(autowalk_dialog_open=True))
    effects = fsm.step(MAP, c)
    assert kinds(effects, Back) and not taps(effects)


@pytest.mark.parametrize("reading", [Tristate.FALSE, Tristate.UNKNOWN])
def test_the_dialog_route_works_when_the_icon_colour_reading_has_not_landed(reading):
    """The live case exactly: `AccountView.autowalk_icon_rect_norm` only exists while the
    shortcut menu is open, so the colour reading is frequently UNKNOWN on the tick that
    matters. The dialog is a second, independent route to the same conclusion."""
    c = awctx(phase="autowalk_dialog", since=100.0, accounts=dangerous(),
              switch_autowalk_active=reading)
    assert kinds(fsm.step(MAP, c), Back)


def test_the_existing_icon_colour_skip_still_works_and_is_untouched():
    """The first route, unchanged: a positively-read TRUE icon skips the "select AutoWalk"
    tap outright, so the stop dialog is never even produced. Neither route replaces the
    other - this one acts a whole phase earlier."""
    c = awctx(phase="autowalk_menu", since=100.0,
              accounts=replace(panel(active="TrainerTwo"), star_norm=STAR,
                               autowalk_menu_norm=(0.30, 0.46)),
              switch_autowalk_active=Tristate.TRUE)
    effects = fsm.step(MAP, c)
    assert not any("select AutoWalk" in t.reason for t in taps(effects))
    assert len(taps(effects)) == 1 and (taps(effects)[0].x, taps(effects)[0].y) \
        == pytest.approx(STAR)
    assert any(isinstance(e, Note) and "already active" in e.text for e in effects)
    assert kinds(effects, Transition)[0].outcome is IntentOutcome.CONFIRMED


# ------------------------------------------------------------------ fsm: closing the menu

def test_the_close_phase_backs_out_again_rather_than_confirming_over_a_live_dialog():
    """`_autowalk_dialog` advances the phase in the SAME tick it emits its BACK, and it is
    pure - it cannot know whether the actuator accepted that BACK (rate limit, failure
    breaker). If it did not, the dialog is still in front of the menu, the star tap would
    be swallowed by its scrim, and this phase would confirm with the menu left open: the
    exact outcome the ladder is being fixed for."""
    c = awctx(phase="autowalk_close", since=100.0, accounts=dangerous())
    effects = fsm.step(MAP, c)
    backs = kinds(effects, Back)
    assert len(backs) == 1
    # Same budget as its sibling in `_autowalk_dialog`: it is the gate this rung is paced
    # by (`Timings.switch_tap`), and "back" would stamp a counter nothing here reads.
    assert backs[0].budget == "switch"
    assert not taps(effects)
    assert not kinds(effects, Transition), "confirmed over a dialog that is still up"


@pytest.mark.parametrize("elapsed", [BUDGET + 1, BUDGET + GRACE + 1, 240.0])
def test_the_close_phase_stops_pressing_back_once_the_ladder_is_out_of_time(elapsed):
    """The retry BACK above is the only rung that can repeat WITHOUT advancing anything -
    it neither moves the phase on nor falls through to `_autowalk_deadline` at the bottom
    of the method. So it has to consult that deadline itself, or a dialog BACK does not
    dismiss repeats forever.

    That is not hypothetical: driven through the real Runner at a 3.0s tick (below) an
    unbounded version sent 74 BACKs over 246s and died on `Timings.switch_timeout`, which
    books a switch whose login had already CONFIRMED as a failure and arms the 10-minute
    backoff - the exact thing `config.AutoWalk`'s wall clock is documented to prevent.
    """
    c = awctx(phase="autowalk_close", since=100.0, accounts=dangerous())
    c.now = c.switch_autowalk_since + elapsed
    c.state_since = c.now                       # only the AutoWalk clock may end this
    effects = fsm.step(MAP, c)
    assert kinds(effects, Transition), \
        f"still deciding {elapsed - BUDGET - GRACE:.0f}s past the ladder's own bound"
    for bad in FORBIDDEN:                       # giving up must not become pressing
        assert not [t for t in taps(effects) if (t.x, t.y) == pytest.approx(bad)]


def test_the_close_phase_taps_the_star_once_the_dialog_is_gone():
    c = awctx(phase="autowalk_close", since=100.0,
              accounts=replace(panel(active="TrainerTwo"), star_norm=STAR))
    effects = fsm.step(MAP, c)
    assert len(taps(effects)) == 1
    assert (taps(effects)[0].x, taps(effects)[0].y) == pytest.approx(STAR)
    assert kinds(effects, Transition)[0].outcome is IntentOutcome.CONFIRMED


@pytest.mark.parametrize("phase", ["autowalk_menu", "autowalk_dialog"])
def test_the_deadline_still_closes_the_menu_whenever_a_star_is_located(phase):
    """The give-up paths at the phases where the menu IS open and the phase itself has
    nothing left to try. Unchanged behaviour, restated here because the audit this module
    documents has to cover every exit. "autowalk_close" is absent because a located star
    never reaches its deadline at all - closing the menu IS that phase's own job, and
    `test_the_close_phase_taps_the_star_once_the_dialog_is_gone` above covers it."""
    c = awctx(phase=phase, since=100.0,
              accounts=replace(panel(active="TrainerTwo"), star_norm=STAR,
                               autowalk_menu_norm=None))
    c.now = c.switch_autowalk_since + BUDGET + 1.0
    effects = fsm.step(MAP, c)
    assert len(taps(effects)) == 1
    assert (taps(effects)[0].x, taps(effects)[0].y) == pytest.approx(STAR)
    assert "giving up" in taps(effects)[0].reason
    assert not [n for n in kinds(effects, Note) if n.level == "warn"], \
        "it DID close the menu, so nothing may warn that it did not"
    assert kinds(effects, Transition)[0].outcome is IntentOutcome.CONFIRMED


@pytest.mark.parametrize("phase", ["autowalk_menu", "autowalk_dialog", "autowalk_close"])
def test_the_one_exit_that_cannot_close_the_menu_says_so_at_warn_level(phase):
    """No star located before the cleanup allowance ran out, so there is nothing to close
    it WITH. The CONFIRMED transition line on its own reads like a tidy finish; an operator
    has to be told the menu is still over the map, because that is where a stray tap
    becomes a PGSharp settings page."""
    c = awctx(phase=phase, since=100.0,
              accounts=replace(panel(active="TrainerTwo"), star_norm=None))
    c.now = c.switch_autowalk_since + BUDGET + GRACE + 1.0
    effects = fsm.step(MAP, c)
    assert not taps(effects)
    warns = [n for n in kinds(effects, Note) if n.level == "warn"]
    assert len(warns) == 1
    assert "OPEN" in warns[0].text and "menu" in warns[0].text
    assert kinds(effects, Transition)[0].outcome is IntentOutcome.CONFIRMED


def test_the_open_phase_deadline_claims_nothing_about_a_menu_because_none_was_opened():
    """Reaching "autowalk_open"'s own deadline means the star was never TAPPED across the
    whole budget - locating it always taps and advances the phase in the same tick - so no
    menu was ever opened. A warning here would send an operator looking for a screen that
    is not in front of them."""
    c = awctx(phase="autowalk_open", since=100.0,
              accounts=replace(panel(active="TrainerTwo"), star_norm=None))
    c.now = c.switch_autowalk_since + BUDGET + GRACE + 1.0
    effects = fsm.step(MAP, c)
    assert not taps(effects)
    assert not kinds(effects, Note)
    assert kinds(effects, Transition)[0].outcome is IntentOutcome.CONFIRMED


def test_the_star_separation_giveup_never_opened_a_menu_either():
    """The other exit from "autowalk_open": the star and the accounts launcher are
    collapsed onto each other and will not separate, so the star is deliberately never
    tapped. Nothing to close, and the Note says what it actually skipped."""
    from tests.test_star_separation import COLLAPSED
    c = awctx(phase="autowalk_open", since=100.0, accounts=COLLAPSED,
              star_drags=Config().star_separation.max_drags)
    effects = fsm.step(MAP, c)
    assert not taps(effects)
    assert "skipping AutoWalk" in kinds(effects, Note)[0].text
    assert kinds(effects, Transition)[0].outcome is IntentOutcome.CONFIRMED


def test_a_preflight_with_no_view_tree_never_opened_a_menu_either():
    """`Preflight._autowalk_open` refuses outright when the run has no uiautomator reader:
    the star can never be located, so it can never have been tapped."""
    c = pctx(phase="autowalk_open", tree_available=False)
    effects = fsm.step(MAP, c)
    assert not taps(effects)
    assert "view tree" in kinds(effects, Transition)[0].reason


@pytest.mark.parametrize("phase, mentions_menu", [
    ("autowalk_open", False), ("autowalk_menu", True),
    ("autowalk_dialog", True), ("autowalk_close", True)])
def test_the_state_timeout_names_the_menu_at_exactly_the_phases_where_it_is_open(
        phase, mentions_menu):
    """The last exit in the audit, and the only one with no chance to close anything: the
    STATE timeout. It cannot tap, so all it can do is be honest about what it is leaving
    behind - which is a menu at every autowalk phase except the first."""
    c = pctx(phase=phase, state_since=0.0, now=10_000.0)
    note = kinds(fsm.step(obs(screen="Menu", conf=0.99), c), Note)[0]
    assert note.level == "warn"
    # On the distinctive phrase, never on the bare word "menu": the note interpolates the
    # PHASE name, so "autowalk_menu" satisfies a substring check on "menu" all by itself
    # and this case passed with the wording reverted.
    assert ("shortcut menu may still be over the map" in note.text) is mentions_menu, \
        note.text


# ------------------------------------------------------------------ purity

@pytest.mark.parametrize("phase", ["autowalk_dialog", "autowalk_close"])
@pytest.mark.parametrize("make", [awctx, pctx])
def test_the_handler_writes_nothing_to_the_context(phase, make):
    """The FSM is pure: (Observation, Context) -> list[Effect]. Only the runner mutates the
    context, which is what keeps a dry run and a live run on the same trajectory."""
    c = make(phase=phase, accounts=dangerous())
    c.switch_autowalk_since = 99.0
    before = copy.deepcopy(c.__dict__)
    out = fsm.step(MAP, c)
    assert out, "nothing was emitted, so nothing was actually exercised"
    assert c.__dict__ == before


# ------------------------------------------------------------------ end to end (real Runner)

class PGSharpAlreadyWalking:
    """tests/test_autowalk.PGSharpStar with the ONE difference the device showed.

    A route is already running, so picking "AutoWalk" from the open menu produces the
    Stop/Pause dialog rather than the setup one, and BACK - never a button - is what
    dismisses it. The menu stays up behind it, exactly as it does behind the setup dialog
    (`fsm.Switching._autowalk_close`'s own reason for existing).

    It keeps reporting `autowalk_ok_norm` while that dialog is up, which
    `accounts.parse_dump` no longer does: the FSM gate has to be sufficient on its own, so
    the model hands the ladder the dangerous coordinate anyway and the assertion is that
    nothing ever taps it.

    Driven purely by the effects the actuator actually ACCEPTED, so the model can never run
    ahead of what the FSM really emitted. Needs no device.
    """

    ITEM = (0.30, 0.46)

    def __init__(self, actuator, active="TrainerTwo"):
        self.actuator = actuator
        self.active = active
        self.reads = 0

    def _state(self):
        menu = dialog = False
        for e in self.actuator.applied:
            if isinstance(e, Tap):
                p = (e.x, e.y)
                if p == STAR:
                    menu = not menu
                elif p == self.ITEM and menu:
                    dialog = True          # a route is running: the STOP dialog, not setup
            elif isinstance(e, Back) and dialog:
                dialog = False             # BACK dismisses it - verified on the device
        return menu, dialog

    @property
    def menu_open(self) -> bool:
        return self._state()[0]

    def read(self):
        self.reads += 1
        menu, dialog = self._state()
        return replace(panel(active=self.active), star_norm=STAR,
                       autowalk_menu_norm=self.ITEM if menu else None,
                       autowalk_dialog_open=False,
                       autowalk_running_dialog_open=dialog,
                       autowalk_ok_norm=STOP_NORM if dialog else None)


def _switch_runner(tmp_path):
    r = make_runner(stats_path=tmp_path / "sessions.jsonl", roster=ROSTER)
    widget = PGSharpAlreadyWalking(r.actuator, active="TrainerTwo")
    r.tree_reader = widget
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.now = 1_000.0
    r._begin_switch("TrainerTwo")
    r._accounts_read_at = 0.0
    return r, widget


def _pressed(r, xy):
    return [e for e in r.actuator.applied
            if isinstance(e, Tap) and (e.x, e.y) == pytest.approx(xy)]


@pytest.mark.parametrize("dt", TICK_INTERVALS)
def test_a_switch_finishes_promptly_when_autowalk_is_already_running(tmp_path, dt):
    """The live failure, reproduced end to end and then fixed. Before this change the
    ladder sat in "autowalk_dialog" for the whole 30s budget and handed SCANNING an open
    shortcut menu; now it recognises the dialog, backs out of it, shuts the menu and
    confirms - well inside that budget, at every cadence the runner really has."""
    r, widget = _switch_runner(tmp_path)
    elapsed = _drive_until_scanning(r, MAP, dt)
    assert r.ctx.state is BotState.SCANNING
    assert r.stats.account == "TrainerTwo"          # the switch itself still confirmed
    assert widget.menu_open is False, "the shortcut menu was left over the map"
    assert len(_pressed(r, STAR)) == 2              # open, then close
    assert elapsed < Config().autowalk.budget_s


@pytest.mark.parametrize("dt", TICK_INTERVALS)
def test_nothing_the_switch_ever_sent_landed_on_pause_or_stop(tmp_path, dt):
    """The safety property through the real actuator: not "no extra taps", but no tap at
    either of those two coordinates, at any tick cadence."""
    r, _ = _switch_runner(tmp_path)
    _drive_until_scanning(r, MAP, dt)
    for bad in FORBIDDEN:
        assert not _pressed(r, bad)


def test_back_is_what_the_runner_actually_sent(tmp_path):
    r, _ = _switch_runner(tmp_path)
    _drive_until_scanning(r, MAP, 0.1)
    backs = [e for e in r.actuator.applied if isinstance(e, Back)]
    assert len(backs) == 1
    assert "Stop/Pause" in backs[0].reason


class PGSharpDialogThatBackWillNotShift(PGSharpAlreadyWalking):
    """The same widget, with the one assumption the device could NOT confirm removed: that
    BACK dismisses the dialog. It is verified for TODAY's PGSharp, but the retry rung in
    `_autowalk_close` exists precisely FOR the tick where a BACK does not land, so what it
    does when none of them ever land is the behaviour that has to be bounded."""

    def _state(self):
        menu = dialog = False
        for e in self.actuator.applied:
            if isinstance(e, Tap):
                p = (e.x, e.y)
                if p == STAR:
                    menu = not menu
                elif p == self.ITEM and menu:
                    dialog = True
        return menu, dialog


@pytest.mark.parametrize("dt", TICK_INTERVALS)
def test_a_dialog_that_never_dismisses_still_gives_up_inside_the_ladders_own_budget(
        tmp_path, dt):
    """The ladder is documented to bound ITSELF (`config.AutoWalk`, `Timings
    .preflight_timeout`), so an undismissable dialog must end as a give-up, never by
    running the SWITCH out of time.

    The 3.0s cadence is the one that bites: it is longer than the tree-refresh throttle, so
    `ctx.accounts` is never dropped and the runner-side "view went None, re-read the
    deadline" accident that masks this at frame rate never happens. Unbounded, it measured
    246s / 74 BACKs here and rolled `stats.account` back to TrainerOne.
    """
    r, _ = _switch_runner(tmp_path)
    r.tree_reader = PGSharpDialogThatBackWillNotShift(r.actuator, active="TrainerTwo")
    elapsed = _drive_until_scanning(r, MAP, dt, seconds=400.0)

    assert r.ctx.state is BotState.SCANNING
    assert elapsed < Config().timings.switch_timeout, \
        "the AutoWalk ladder pushed the switch itself into timing out"
    assert r.stats.account == "TrainerTwo", \
        "a confirmed login was rolled back and booked as a failed switch"
    assert elapsed <= BUDGET + GRACE + dt + 5.0, elapsed
    for bad in FORBIDDEN:
        assert not _pressed(r, bad)


def test_the_next_switch_still_finds_the_menu_closed_and_can_open_it(tmp_path):
    """The consequence that makes the closed menu load-bearing rather than tidy: the star
    TOGGLES. A first switch that left the menu open would have the second one's opening tap
    shut it instead, after which no "AutoWalk" node is ever found again."""
    r, widget = _switch_runner(tmp_path)
    _drive_until_scanning(r, MAP, 0.1)
    assert widget.menu_open is False

    widget.active = "TrainerOne"
    r._begin_switch("TrainerOne")
    _drive_until_scanning(r, MAP, 0.1)
    assert r.ctx.state is BotState.SCANNING
    assert r.stats.account == "TrainerOne"
    assert len(_pressed(r, STAR)) == 4              # two opens, two closes
    assert len(_pressed(r, PGSharpAlreadyWalking.ITEM)) == 2, \
        "the second switch never found the menu to pick AutoWalk from"
    assert widget.menu_open is False


@pytest.mark.parametrize("dt", TICK_INTERVALS)
def test_the_startup_preflight_reports_a_running_route_instead_of_failing(tmp_path, dt, caplog):
    """The run in the task brief was a PREFLIGHT, and the same ladder serves both callers.
    The 30s "could not start AutoWalk" warning must be gone, replaced by a plain statement
    that a route is already running - and the menu must still end up shut."""
    from tests.test_preflight import _drive, _runner
    r = _runner()
    widget = PGSharpAlreadyWalking(r.actuator, active="TrainerOne")
    r.tree_reader = widget
    with caplog.at_level(logging.INFO, logger="pogobot"):
        elapsed = _drive(r, obs(on_map=True, goplus=Tristate.UNKNOWN), dt)
    assert elapsed is not None and r.ctx.state is BotState.SCANNING
    assert widget.menu_open is False
    assert len(_pressed(r, STAR)) == 2
    for bad in FORBIDDEN:
        assert not _pressed(r, bad)
    assert not [m for m in caplog.messages if "could not start AutoWalk" in m], caplog.messages
    assert [m for m in caplog.messages if "AutoWalk is already running" in m], caplog.messages
    assert elapsed < Config().timings.preflight_timeout
