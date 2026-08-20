"""AutoWalk after a confirmed account switch: tap PGSharp's floating star widget, pick
AutoWalk from the shortcut menu it opens, press CONTINUE LAST if offered (else OK), then
close the menu again - the ladder `fsm.Switching._autowalk_open` and its neighbours drive
once `_goplus` (tests/test_goplus.py) is itself done.

Two layers, tested separately, the same split tests/test_goplus.py uses:

  * `accounts.parse_dump` is a pure function of one uiautomator dump. Its tests read real
    fixture XML (tests/fixtures/uiautomator/), including one where the star has moved -
    the whole point being that it is LOCATED every time, never a remembered coordinate.
  * `fsm.Switching._autowalk_*` is where the tap-or-not decisions live, and where the
    wall-clock budget (config.AutoWalk.budget_s) is enforced. Its tests drive `fsm.step`
    the same way every other Switching-phase test in this suite does.

The end-to-end tests at the bottom drive the REAL Runner, and they are parametrized over
the tick interval on purpose. Two runner behaviours only interact at short ticks:
`Runner.apply` drops `ctx.accounts` after every actuation taken while SWITCHING, and
`_refresh_accounts` only puts it back every `runner.ACCOUNTS_REFRESH` (2.5s). A test that
advances the clock by more than that per tick never sees the resulting `None` view and can
pass while a phase is silently skipping its own step - which is exactly how the missing
`_autowalk_close` tap survived. The live loop ticks at frame rate; 0.1s is the case that
has to hold.
"""
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from pogobot import fsm
from pogobot import perception as P
from pogobot.accounts import FakeTreeReader, parse_dump
from pogobot.config import AutoWalk, Config
from pogobot.effects import BotState, IntentOutcome, Note, SetFlag, Tap, Transition
from pogobot.frames import Frame
from pogobot.observation import Tristate
from tests.factories import obs
from tests.test_switch_runner import ROSTER, make_runner
from tests.test_switching import budget, ctx as switching_ctx, panel

FIX = Path(__file__).parent / "fixtures" / "uiautomator"
SCREENS = Path(__file__).parent / "fixtures" / "screens"
WH = (1080, 2340)


def view(name: str):
    return parse_dump((FIX / name).read_bytes(), WH)


# ------------------------------------------------------------------ accounts.py: locating the star

def test_the_star_is_located_and_distinct_from_the_cooldown_launcher():
    v = view("accounts_open.xml")
    assert v.star_norm is not None and v.launcher_norm is not None
    assert v.star_norm != v.launcher_norm
    # The exact centres of the two real, measured bounds this fixture carries -
    # [27,488][135,596] for the star, [0,71][272,164] for the cooldown launcher (see the
    # module docstring's own explanation of why bounds differ slightly from the task
    # brief's illustrative numbers).
    assert v.star_norm == pytest.approx(((27 + 135) / 2 / 1080, (488 + 596) / 2 / 2340))
    assert v.launcher_norm == pytest.approx(((0 + 272) / 2 / 1080, (71 + 164) / 2 / 2340))


def test_the_star_is_located_at_its_moved_position_not_a_remembered_one():
    """The star is described as draggable and was measured at two different positions
    hours apart - [27,488][135,596] then [0,188][108,296]. A locator that hardcoded the
    first would silently keep tapping empty space once the widget moved."""
    original = view("overlay_closed.xml")
    moved = view("star_moved.xml")
    assert original.star_norm is not None and moved.star_norm is not None
    assert original.star_norm != moved.star_norm
    assert moved.star_norm == pytest.approx(((0 + 108) / 2 / 1080, (188 + 296) / 2 / 2340))
    # The cooldown launcher did not move in this fixture pair - proof the two anchors
    # (hl_floating_icon vs hl_cd_text) are being told apart, not one lucky coordinate.
    assert moved.launcher_norm == original.launcher_norm


def test_a_dump_with_neither_widget_locates_neither():
    v = parse_dump(b"not xml at all", WH)
    assert v.star_norm is None and v.launcher_norm is None


# ------------------------------------------------------------------ accounts.py: the shortcut menu

def test_autowalk_is_found_among_the_menu_items_by_its_text():
    v = view("autowalk_menu.xml")
    assert v.autowalk_menu_norm is not None
    # All eight items from the task brief are present in the fixture; only "AutoWalk"'s
    # own bounds may ever be returned.
    assert v.autowalk_menu_norm == pytest.approx((0.386574, 0.315385), abs=1e-4)


def test_a_menu_less_dump_does_not_invent_an_autowalk_entry():
    v = view("accounts_open.xml")
    assert v.autowalk_menu_norm is None
    assert v.autowalk_icon_rect_norm is None


def test_the_autowalk_icon_box_is_the_labels_own_left_edge_and_vertical_bounds():
    """x=0 to the label's own left edge, y over the label's own vertical bounds - both
    taken from the SAME node `autowalk_menu_norm` came from
    (tests/fixtures/uiautomator/autowalk_menu.xml: "AutoWalk" bounds [135,688][700,788]
    on this fixture's own 1080x2340 dump), never a hardcoded rectangle."""
    v = view("autowalk_menu.xml")
    assert v.autowalk_icon_rect_norm == pytest.approx(
        (0.0, 688 / 2340, 135 / 1080, 788 / 2340))


# ------------------------------------------------------------------ accounts.py: the AutoWalk dialog

def test_continue_last_is_located_when_present():
    v = view("autowalk_dialog_continue_last.xml")
    assert v.autowalk_dialog_open is True
    assert v.autowalk_continue_last_norm is not None
    assert v.autowalk_ok_norm is not None
    assert v.autowalk_continue_last_norm != v.autowalk_ok_norm


def test_ok_is_located_alone_when_continue_last_is_absent():
    v = view("autowalk_dialog_ok_only.xml")
    assert v.autowalk_dialog_open is True
    assert v.autowalk_continue_last_norm is None
    assert v.autowalk_ok_norm is not None


def test_the_dialog_is_only_recognised_by_its_own_title():
    """button1/2/3 are generic Android AlertDialog ids some other dialog could also use -
    `autowalk_dialog_open` must require the title text, not merely their presence."""
    xml = (FIX / "autowalk_dialog_continue_last.xml").read_bytes().replace(
        b"Auto-Generated GPX", b"Something Else Entirely")
    v = parse_dump(xml, WH)
    assert v.autowalk_dialog_open is False
    # The buttons are still located - only the "is this THE dialog" claim is withheld -
    # exactly like `panel_open` deriving from more than one signal in accounts.py.
    assert v.autowalk_ok_norm is not None


def test_nothing_exposes_a_coordinate_for_the_input_field_or_either_toggle_group():
    """AccountView has no field for hl_aw_input or the toggles at all - this is the
    structural guarantee that makes them untappable, not merely un-tapped. Asserted
    against the real fixture's own dataclass fields rather than by grepping source, so a
    field added later without a corresponding assertion here fails loudly."""
    v = view("autowalk_dialog_continue_last.xml")
    from dataclasses import fields
    names = {f.name for f in fields(v)}
    assert not any("input" in n or "toggle" in n or "stop" in n or "gym" in n
                   or "station" in n or "oneway" in n or "loop" in n for n in names)


# ------------------------------------------------------------------ perception: autowalk_active_signal
#
# The committed fixture pair (tests/fixtures/{uiautomator,screens}/autowalk_menu_active.*)
# is the ONE captured moment PGSharp ever showed AutoWalk already running for an account.
# Real pixels, real bounds - no mocking of either channel.

FIX_ACTIVE_XML = FIX / "autowalk_menu_active.xml"
FIX_ACTIVE_PNG = SCREENS / "autowalk_menu_active.png"


def _active_bgr():
    bgr = cv2.imread(str(FIX_ACTIVE_PNG))
    assert bgr is not None, f"could not load {FIX_ACTIVE_PNG}"
    return bgr


def test_the_active_icon_reads_true_on_the_real_fixture():
    """The one real positive sample: AutoWalk's own icon box, on the dump and screenshot
    taken at the same moment, reads TRUE - a confidently blue glyph, not white."""
    v = view("autowalk_menu_active.xml")
    assert v.autowalk_icon_rect_norm is not None
    assert P.autowalk_active_signal(_active_bgr(), v.autowalk_icon_rect_norm, Config()) \
        is Tristate.TRUE


def test_a_plainly_white_sibling_icon_does_not_read_as_active():
    """A DIFFERENT item's icon box, from the SAME real dump and screenshot - genuinely
    inactive, genuinely white - must read FALSE, not TRUE, and the ladder must proceed
    exactly as today when it does. "Map"'s own label bounds on this fixture's dump are
    [108,303][230,399] - parse_dump only ever exposes AutoWalk's own box, so this is the
    same real measurement, taken by hand the way the task brief's own table was built."""
    icon_rect = (0.0, 303 / 2340, 108 / 1080, 399 / 2340)
    assert P.autowalk_active_signal(_active_bgr(), icon_rect, Config()) is Tristate.FALSE


def test_a_missing_icon_box_reads_unknown_not_active():
    """No node located -> nothing to sample -> UNKNOWN, exactly like a missing node means
    "do nothing" everywhere else in this codebase."""
    assert P.autowalk_active_signal(_active_bgr(), None, Config()) is Tristate.UNKNOWN


def _autowalk_icon_patch(white_frac: float, blue_frac: float, w: int = 100, h: int = 100):
    """A real HSV patch whose white/blue fractions hit the given targets exactly,
    round-tripped through real cv2 HSV<->BGR conversion rather than injected as a mock -
    the same discipline tests/test_goplus.py's synthetic frames use."""
    total = w * h
    n_white = int(round(white_frac * total))
    n_blue = int(round(blue_frac * total))
    flat = np.empty((total, 3), dtype=np.uint8)
    flat[:] = (90, 50, 50)              # neutral: outside both bands
    flat[:n_white] = (0, 10, 200)       # white band: S=10<60, V=200>170
    flat[n_white:n_white + n_blue] = (115, 200, 150)  # blue band: H=115, S=200, V=150
    hsv = flat.reshape(h, w, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_the_ambiguous_older_capture_does_not_read_as_active():
    """The one older capture the task brief itself flags as unusable: white=0.041,
    blue=0.096 for this same box, taken before AutoWalk was ever started on that account -
    neither clearly white nor clearly blue. Synthesized at those exact measured values
    (there is no fixture file for it - see config.Thresholds' honesty note) and must land
    in the UNKNOWN gap, never TRUE."""
    bgr = _autowalk_icon_patch(white_frac=0.041, blue_frac=0.096)
    result = P.autowalk_active_signal(bgr, (0.0, 0.0, 1.0, 1.0), Config())
    assert result is Tristate.UNKNOWN


def test_a_synthetic_active_patch_at_the_measured_values_reads_true():
    bgr = _autowalk_icon_patch(white_frac=0.002, blue_frac=0.320)
    assert P.autowalk_active_signal(bgr, (0.0, 0.0, 1.0, 1.0), Config()) is Tristate.TRUE


def test_a_synthetic_plain_white_patch_reads_false():
    bgr = _autowalk_icon_patch(white_frac=0.30, blue_frac=0.0)
    assert P.autowalk_active_signal(bgr, (0.0, 0.0, 1.0, 1.0), Config()) is Tristate.FALSE


# ------------------------------------------------------------------ config

def test_autowalk_budget_is_a_real_config_knob_not_a_hardcoded_number():
    a = AutoWalk()
    assert a.budget_s == pytest.approx(30.0)
    assert Config().autowalk.budget_s == pytest.approx(a.budget_s)


# ------------------------------------------------------------------ fsm: "autowalk_open"

def _panel_with(**kw):
    return replace(panel(active="TrainerTwo"), **kw)


def awctx(phase="autowalk_open", since=None, **kw):
    c = switching_ctx(phase=phase, target="TrainerTwo", **kw)
    if since is not None:
        c.switch_autowalk_since = since
    return c


def test_autowalk_open_acts_even_when_the_screen_reads_off_map():
    """The critical case this ladder exists to survive: PGSharp's own shortcut menu and
    the AlertDialog it opens sit on top of the map exactly while these phases need to
    act, and a real AlertDialog dims the window behind it - `obs.on_map` reading False
    while the star is genuinely present (per the LIVE uiautomator tree) must not block
    the tap, or the ladder can never get past this phase while its own overlay is up."""
    c = awctx(accounts=_panel_with(star_norm=(0.08, 0.23)))
    effects = fsm.step(obs(on_map=False, screen="Rocket", conf=0.99), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1
    assert (taps[0].x, taps[0].y) == pytest.approx((0.08, 0.23))
    assert any(isinstance(e, SetFlag) and e.name == "switch_phase"
               and e.value == "autowalk_menu" for e in effects)


def test_autowalk_open_waits_off_map_when_the_star_is_not_yet_located():
    """Off-map alone is not evidence the star exists - a missing node still means wait,
    exactly as it does on-map. This is the "no target, off-map, before the deadline"
    half of the off-map story; the "no target, off-map, PAST the deadline" half is
    covered below by test_the_deadline_is_reachable_even_when_the_screen_reads_off_map."""
    c = awctx(accounts=_panel_with(star_norm=None), since=100.0)
    effects = fsm.step(obs(on_map=False, screen="Rocket", conf=0.99), c)
    assert not any(isinstance(e, Tap) for e in effects)
    assert not any(isinstance(e, Transition) for e in effects)


def test_autowalk_open_taps_the_star_and_advances_the_phase():
    c = awctx(accounts=_panel_with(star_norm=(0.08, 0.23)))
    effects = fsm.step(obs(on_map=True), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1
    assert (taps[0].x, taps[0].y) == pytest.approx((0.08, 0.23))
    assert taps[0].budget == "switch"
    assert any(isinstance(e, SetFlag) and e.name == "switch_phase"
               and e.value == "autowalk_menu" for e in effects)
    assert not any(isinstance(e, Transition) for e in effects)


def test_autowalk_open_stamps_its_own_clock_on_first_entry():
    c = awctx(accounts=_panel_with(star_norm=None))
    assert c.switch_autowalk_since == 0.0
    effects = fsm.step(obs(on_map=True), c)
    assert any(isinstance(e, SetFlag) and e.name == "switch_autowalk_since"
               and e.value == c.now for e in effects)


def test_autowalk_open_does_nothing_when_the_star_is_not_located():
    """Missing node -> do nothing, not a guess."""
    c = awctx(accounts=_panel_with(star_norm=None), since=100.0)
    effects = fsm.step(obs(on_map=True), c)
    assert not any(isinstance(e, Tap) for e in effects)
    assert not any(isinstance(e, Transition) for e in effects)


def test_autowalk_open_does_nothing_when_the_view_is_unavailable():
    from pogobot.accounts import AccountView
    c = awctx(accounts=AccountView(available=False), since=100.0)
    assert fsm.step(obs(on_map=True), c) == []


def test_autowalk_open_waits_out_its_own_settle_pace():
    c = awctx(accounts=_panel_with(star_norm=(0.08, 0.23)), since=100.0)
    c.last_action["switch"] = c.now
    assert fsm.step(obs(on_map=True), c) == []


# ------------------------------------------------------------------ fsm: "autowalk_menu"

def test_autowalk_menu_taps_the_located_autowalk_entry():
    c = awctx(phase="autowalk_menu", since=100.0,
              accounts=_panel_with(autowalk_menu_norm=(0.30, 0.46)))
    effects = fsm.step(obs(on_map=True), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1
    assert (taps[0].x, taps[0].y) == pytest.approx((0.30, 0.46))
    assert any(isinstance(e, SetFlag) and e.name == "switch_phase"
               and e.value == "autowalk_dialog" for e in effects)


def test_autowalk_menu_does_nothing_before_the_menu_has_rendered():
    c = awctx(phase="autowalk_menu", since=100.0,
              accounts=_panel_with(autowalk_menu_norm=None))
    assert fsm.step(obs(on_map=True), c) == []


def test_autowalk_menu_skips_the_tap_and_closes_when_already_active():
    """The user's own rule: a blue AutoWalk icon means the account is ALREADY
    autowalking, and it must not be tapped into it again. Skips the "select AutoWalk"
    tap and the whole dialog phase, closes the menu via the located star, and confirms
    the switch - all in the SAME tick, since the fresh view that says "already active" is
    the very view the close tap needs."""
    c = awctx(phase="autowalk_menu", since=100.0,
              accounts=_panel_with(autowalk_menu_norm=(0.30, 0.46), star_norm=(0.08, 0.23)),
              switch_autowalk_active=Tristate.TRUE)
    effects = fsm.step(obs(on_map=True), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert not any("select AutoWalk" in t.reason for t in taps)
    assert len(taps) == 1
    assert (taps[0].x, taps[0].y) == pytest.approx((0.08, 0.23))
    assert "close" in taps[0].reason
    assert any(isinstance(e, Transition) and e.to is BotState.SCANNING
               and e.outcome is IntentOutcome.CONFIRMED for e in effects)
    assert any(isinstance(e, Note) and e.level == "info" and "already active" in e.text
               for e in effects)


@pytest.mark.parametrize("reading", [Tristate.FALSE, Tristate.UNKNOWN])
def test_autowalk_menu_taps_normally_when_not_confidently_active(reading):
    """FALSE (a genuinely white icon) and UNKNOWN (including the one ambiguous sample
    this signal is known not to separate cleanly - see config.Thresholds) both fall
    through to the ordinary tap: the safe failure direction is "not active"."""
    c = awctx(phase="autowalk_menu", since=100.0,
              accounts=_panel_with(autowalk_menu_norm=(0.30, 0.46)),
              switch_autowalk_active=reading)
    effects = fsm.step(obs(on_map=True), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1
    assert (taps[0].x, taps[0].y) == pytest.approx((0.30, 0.46))
    assert any(isinstance(e, SetFlag) and e.name == "switch_phase"
               and e.value == "autowalk_dialog" for e in effects)


# ------------------------------------------------------------------ fsm: "autowalk_dialog"

def test_dialog_prefers_continue_last_when_present():
    c = awctx(phase="autowalk_dialog", since=100.0, accounts=_panel_with(
        autowalk_dialog_open=True, autowalk_continue_last_norm=(0.30, 0.90),
        autowalk_ok_norm=(0.80, 0.90)))
    effects = fsm.step(obs(on_map=True), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1
    assert (taps[0].x, taps[0].y) == pytest.approx((0.30, 0.90))
    assert any(isinstance(e, SetFlag) and e.name == "switch_phase"
               and e.value == "autowalk_close" for e in effects)


def test_dialog_presses_continue_last_even_when_the_alertdialog_reads_off_map():
    """The highest-risk instance of the off-map problem: `autowalk_dialog_open` is a
    genuine Android AlertDialog, which by platform default dims the window behind it -
    so this is the phase most likely to see `obs.on_map` read False on the exact tick it
    most needs to act. The button must still be pressed."""
    c = awctx(phase="autowalk_dialog", since=100.0, accounts=_panel_with(
        autowalk_dialog_open=True, autowalk_continue_last_norm=(0.30, 0.90),
        autowalk_ok_norm=(0.80, 0.90)))
    effects = fsm.step(obs(on_map=False, screen="Rocket", conf=0.99), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1
    assert (taps[0].x, taps[0].y) == pytest.approx((0.30, 0.90))


def test_dialog_uses_ok_when_continue_last_is_absent():
    c = awctx(phase="autowalk_dialog", since=100.0, accounts=_panel_with(
        autowalk_dialog_open=True, autowalk_continue_last_norm=None,
        autowalk_ok_norm=(0.80, 0.90)))
    effects = fsm.step(obs(on_map=True), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1
    assert (taps[0].x, taps[0].y) == pytest.approx((0.80, 0.90))
    assert "OK" in taps[0].reason


def test_dialog_never_touches_the_input_field_or_a_toggle():
    """There is no coordinate anywhere on AccountView for hl_aw_input or the toggle
    groups (see the accounts.py test above); this is the FSM-level half of that
    guarantee - whatever the dialog offers, the only tap this phase can ever emit is one
    of the two button coordinates it was handed."""
    c = awctx(phase="autowalk_dialog", since=100.0, accounts=_panel_with(
        autowalk_dialog_open=True, autowalk_continue_last_norm=(0.30, 0.90),
        autowalk_ok_norm=(0.80, 0.90)))
    effects = fsm.step(obs(on_map=True), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    for t in taps:
        assert (t.x, t.y) in {(0.30, 0.90), (0.80, 0.90)}


def test_dialog_does_nothing_before_it_has_opened():
    c = awctx(phase="autowalk_dialog", since=100.0,
              accounts=_panel_with(autowalk_dialog_open=False))
    assert fsm.step(obs(on_map=True), c) == []


def test_dialog_does_nothing_when_open_but_no_button_is_located_yet():
    c = awctx(phase="autowalk_dialog", since=100.0, accounts=_panel_with(
        autowalk_dialog_open=True, autowalk_continue_last_norm=None, autowalk_ok_norm=None))
    assert fsm.step(obs(on_map=True), c) == []


# ------------------------------------------------------------------ fsm: "autowalk_close"

def test_close_taps_the_star_again_and_confirms():
    c = awctx(phase="autowalk_close", since=100.0,
              accounts=_panel_with(star_norm=(0.08, 0.23)))
    effects = fsm.step(obs(on_map=True), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1
    assert (taps[0].x, taps[0].y) == pytest.approx((0.08, 0.23))
    tr = [e for e in effects if isinstance(e, Transition)]
    assert len(tr) == 1
    assert tr[0].to is BotState.SCANNING and tr[0].outcome is IntentOutcome.CONFIRMED
    assert tr[0].reason.endswith("TrainerTwo")
    # The tap has to be emitted BEFORE the confirmation: `Runner.apply` walks the list in
    # order, and everything it does for an actuation taken while SWITCHING - dropping the
    # stale view, charging the "switch" budget, counting the tap against this state - it
    # only does while `ctx.state` still says SWITCHING.
    assert effects.index(taps[0]) < effects.index(tr[0])


def test_close_waits_for_the_view_instead_of_confirming_without_its_tap():
    """THE regression this phase was rebuilt for.

    `Runner.apply` nulls `ctx.accounts` after EVERY actuation taken while SWITCHING (the
    star toggles the menu, so a second decision from one stale view undoes the first), and
    only the next tree read - throttled to `runner.ACCOUNTS_REFRESH` - puts it back. The
    tick straight after `_autowalk_dialog`'s button press therefore ALWAYS sees `None`.
    A version that confirmed on that view emitted no close tap at any tick rate the runner
    actually has; see the end-to-end test below, which now drives the real Runner at a
    sub-second cadence for exactly this reason."""
    c = awctx(phase="autowalk_close", since=100.0, accounts=None)
    assert fsm.step(obs(on_map=True), c) == []


def test_close_taps_when_the_view_arrives_several_ticks_late():
    """The wait is bounded by the refresh, not a stall. Set up as the live tick really is
    on entry to this phase: the dialog tap just charged the "switch" budget and opened the
    settle window, and `ctx.accounts` is `None` until the next tree read ~2.5s later.
    Ticking at 0.1s throughout, nothing is emitted until that read lands - and the close
    tap fires on the very tick it does."""
    c = awctx(phase="autowalk_close", since=100.0, accounts=None)
    c.last_action["switch"] = c.now
    c.settle_until = c.now + c.cfg.timings.ui_settle
    for _ in range(26):                       # 2.6s, one tree-refresh cadence, 0.1s a tick
        c.now = round(c.now + 0.1, 6)
        assert fsm.step(obs(on_map=True), c) == []
    c.accounts = _panel_with(star_norm=(0.08, 0.23))
    effects = fsm.step(obs(on_map=True), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1
    assert (taps[0].x, taps[0].y) == pytest.approx((0.08, 0.23))
    tr = [e for e in effects if isinstance(e, Transition)]
    assert len(tr) == 1 and tr[0].outcome is IntentOutcome.CONFIRMED


def test_close_confirms_after_the_grace_when_the_star_never_reappears():
    """Waiting must never become hanging. A star that genuinely never comes back (a
    PGSharp update, an id rename, a tree read that keeps failing) leaves the menu open -
    regrettable, and exactly what `_autowalk_deadline` is for - but the switch itself
    succeeded and must still be recorded CONFIRMED. Now bounded by
    `AutoWalk.budget_s + close_grace_s`, not by `budget_s` alone: the extra
    `close_grace_s` is exactly the allowance `_autowalk_deadline` spends trying to find
    the star before it gives that attempt up too - see that method's docstring."""
    c = awctx(phase="autowalk_close", since=100.0, accounts=_panel_with(star_norm=None))
    assert fsm.step(obs(on_map=True), c) == []          # still inside the budget: wait
    c.now = c.switch_autowalk_since + c.cfg.autowalk.budget_s + 1.0
    assert fsm.step(obs(on_map=True), c) == []          # inside the cleanup allowance: still wait
    c.now = (c.switch_autowalk_since + c.cfg.autowalk.budget_s
             + c.cfg.autowalk.close_grace_s + 1.0)
    effects = fsm.step(obs(on_map=True), c)
    assert not any(isinstance(e, Tap) for e in effects)
    tr = [e for e in effects if isinstance(e, Transition)]
    assert len(tr) == 1
    assert tr[0].to is BotState.SCANNING and tr[0].outcome is IntentOutcome.CONFIRMED
    # Bounded, not merely eventual: nowhere near the 240s switch_timeout that would
    # otherwise own this outcome (see config.Timings.switch_timeout).
    assert c.now - c.state_since < c.cfg.timings.switch_timeout


def test_close_confirms_after_the_grace_when_the_view_is_unavailable():
    from pogobot.accounts import AccountView
    c = awctx(phase="autowalk_close", since=100.0, accounts=AccountView(available=False))
    assert fsm.step(obs(on_map=True), c) == []
    c.now = (c.switch_autowalk_since + c.cfg.autowalk.budget_s
             + c.cfg.autowalk.close_grace_s + 1.0)
    tr = [e for e in fsm.step(obs(on_map=True), c) if isinstance(e, Transition)]
    assert tr and tr[0].outcome is IntentOutcome.CONFIRMED


def test_close_taps_and_confirms_even_when_the_screen_reads_off_map():
    """A dialog/menu overlay still dimming the screen on this exact tick must not be able
    to withhold either the close tap or the confirmation - the star is located from the
    LIVE uiautomator tree, so `obs.on_map` has nothing to say about whether it is there
    (see `_autowalk_open`'s docstring)."""
    c = awctx(phase="autowalk_close", since=100.0,
              accounts=_panel_with(star_norm=(0.08, 0.23)))
    effects = fsm.step(obs(on_map=False, screen="Rocket", conf=0.99), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1
    assert (taps[0].x, taps[0].y) == pytest.approx((0.08, 0.23))
    tr = [e for e in effects if isinstance(e, Transition)]
    assert len(tr) == 1
    assert tr[0].to is BotState.SCANNING and tr[0].outcome is IntentOutcome.CONFIRMED


def test_close_waits_out_its_own_settle_pace():
    """Same pacing as every sibling: the located star is not a reason to tap through the
    settle window the dialog press just opened."""
    c = awctx(phase="autowalk_close", since=100.0,
              accounts=_panel_with(star_norm=(0.08, 0.23)))
    c.last_action["switch"] = c.now
    assert fsm.step(obs(on_map=True), c) == []


# ------------------------------------------------------------------ give-up cleanup:
# closing a still-open menu rather than abandoning it (the live defect this task fixes)

def test_deadline_closes_a_still_open_menu_via_the_located_star_stuck_in_autowalk_menu():
    """THE fix. Live failure: the ladder opened the shortcut menu (`_autowalk_open`
    tapped the star), but the screen underneath never let 'AutoWalk' render - a
    post-login story dialogue, on the device, not the map. The budget for THIS phase
    ran out with the menu still open, and the old code confirmed without ever closing
    it - see `_autowalk_close`'s own docstring for why that wedges the run. The located
    star must be tapped shut before the switch confirms."""
    c = awctx(phase="autowalk_menu", since=100.0,
              accounts=_panel_with(star_norm=(0.08, 0.23), autowalk_menu_norm=None))
    c.now = c.switch_autowalk_since + c.cfg.autowalk.budget_s + 1.0
    effects = fsm.step(obs(on_map=True), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1
    assert (taps[0].x, taps[0].y) == pytest.approx((0.08, 0.23))
    assert "giving up" in taps[0].reason
    tr = [e for e in effects if isinstance(e, Transition)]
    assert len(tr) == 1
    assert tr[0].to is BotState.SCANNING and tr[0].outcome is IntentOutcome.CONFIRMED
    assert "autowalk_menu" in tr[0].reason
    # The tap has to land BEFORE the confirmation - Runner.apply walks effects in order
    # and only charges the "switch" budget / drops the stale view while still SWITCHING.
    assert effects.index(taps[0]) < effects.index(tr[0])


def test_deadline_closes_a_still_open_menu_stuck_in_autowalk_dialog_with_no_button():
    """Same fix, from the dialog phase: the AlertDialog opened but offers neither
    CONTINUE LAST nor OK (an unexpected PGSharp screen) - `_autowalk_dialog` itself has
    nothing left to press, so this falls all the way to the shared cleanup, which must
    still close the menu the ladder opened rather than abandon it."""
    c = awctx(phase="autowalk_dialog", since=100.0, accounts=_panel_with(
        star_norm=(0.08, 0.23), autowalk_dialog_open=True,
        autowalk_continue_last_norm=None, autowalk_ok_norm=None))
    c.now = c.switch_autowalk_since + c.cfg.autowalk.budget_s + 1.0
    effects = fsm.step(obs(on_map=True), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1
    assert (taps[0].x, taps[0].y) == pytest.approx((0.08, 0.23))
    tr = [e for e in effects if isinstance(e, Transition)]
    assert len(tr) == 1 and tr[0].outcome is IntentOutcome.CONFIRMED
    assert "autowalk_dialog" in tr[0].reason


def test_deadline_never_taps_the_star_in_autowalk_open_nothing_was_ever_opened():
    """The one phase where a plain confirm - no tap at all - is correct: reaching
    "autowalk_open"'s own deadline means the star was never located across the WHOLE
    budget (finding it always taps and advances the phase in the same tick - see
    `_autowalk_open`), so no menu was ever opened and there is nothing to close. Also
    proves this phase gets no extra `close_grace_s` allowance: it confirms the instant
    the bare `budget_s` runs out, not `budget_s + close_grace_s` like its siblings."""
    c = awctx(phase="autowalk_open", since=100.0,
              accounts=_panel_with(star_norm=None))
    c.now = c.switch_autowalk_since + c.cfg.autowalk.budget_s + 1.0
    effects = fsm.step(obs(on_map=True), c)
    assert not any(isinstance(e, Tap) for e in effects)
    tr = [e for e in effects if isinstance(e, Transition)]
    assert len(tr) == 1 and tr[0].outcome is IntentOutcome.CONFIRMED
    assert tr[0].reason.endswith("(autowalk_open)")


def test_deadline_waits_out_a_none_view_within_the_grace_then_closes_when_it_lands():
    """The exact tick-cadence trap the task brief warns about: `ctx.accounts` is `None`
    on the tick the primary budget first runs out (the star-open tap nulled it, and no
    refresh has landed since), so the cleanup must NOT give up on that one snapshot - it
    has to keep waiting, within `close_grace_s`, until a real view arrives."""
    c = awctx(phase="autowalk_menu", since=100.0, accounts=None)
    c.now = c.switch_autowalk_since + c.cfg.autowalk.budget_s + 1.0
    assert fsm.step(obs(on_map=True), c) == []          # None view: wait, don't guess
    c.now += 1.0
    assert fsm.step(obs(on_map=True), c) == []          # still None: keep waiting
    c.accounts = _panel_with(star_norm=(0.08, 0.23), autowalk_menu_norm=None)
    effects = fsm.step(obs(on_map=True), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1 and (taps[0].x, taps[0].y) == pytest.approx((0.08, 0.23))
    tr = [e for e in effects if isinstance(e, Transition)]
    assert tr and tr[0].outcome is IntentOutcome.CONFIRMED


def test_giveup_close_tap_never_lands_on_a_delete_button():
    """The cleanup tap `_autowalk_deadline` now emits is a NEW tap-emitting path this
    handler did not have before this fix - it deserves the same delete-button guard as
    every other coordinate this module can produce (see
    tests/test_switching.py::test_no_tap_ever_lands_on_a_delete_button, the test that
    matters for every OTHER tap in this handler)."""
    for phase in ("autowalk_menu", "autowalk_dialog", "autowalk_close"):
        c = awctx(phase=phase, since=100.0, accounts=_panel_with(
            star_norm=(0.08, 0.23),
            autowalk_menu_norm=None, autowalk_dialog_open=False,
            autowalk_continue_last_norm=None, autowalk_ok_norm=None))
        c.now = c.switch_autowalk_since + c.cfg.autowalk.budget_s + 1.0
        effects = fsm.step(obs(on_map=True), c)
        taps = [e for e in effects if isinstance(e, Tap)]
        # The star IS located, so the fix taps it - proving the guard is actually
        # exercised rather than vacuously true because nothing was located at all.
        assert len(taps) == 1
        for t in taps:
            for r in c.accounts.rows:
                assert (t.x, t.y) != r.delete_norm


def test_close_grace_stays_a_small_fraction_of_switch_timeout():
    """The bound the task brief requires: the cleanup allowance must not push the switch
    toward `Timings.switch_timeout` (240s) - it is meant to survive one tree-refresh
    cycle (`runner.ACCOUNTS_REFRESH`, 2.5s), not to compete with the state's own
    240s ceiling."""
    total = Config().autowalk.budget_s + Config().autowalk.close_grace_s
    assert total < Config().timings.switch_timeout / 2


# ------------------------------------------------------------------ the wall-clock budget

@pytest.mark.parametrize("phase", ["autowalk_open", "autowalk_menu", "autowalk_dialog",
                                   "autowalk_close"])
def test_a_missing_node_still_lets_the_switch_confirm_once_the_budget_runs_out(phase):
    """The requirement in the task brief, verbatim: a missing node at any step means do
    nothing, and the switch still confirms rather than hanging. Every field that could
    locate something is left None/False - the worst case, not a partial one.

    Past "autowalk_open" the star itself is also never found here, so `_autowalk_deadline`
    has nothing left to try once its own `close_grace_s` cleanup allowance is ALSO spent -
    hence the extra grace added to `c.now` below. "autowalk_open" needs no such allowance:
    reaching its OWN deadline means the star was never found AT ALL, so no menu was ever
    opened and there is nothing to clean up (see `_autowalk_deadline`'s docstring) - it
    still confirms the instant the bare budget_s runs out, which is why adding the grace
    period on top changes nothing for that one phase and is used uniformly here anyway."""
    c = awctx(phase=phase, since=100.0, accounts=_panel_with(
        star_norm=None, autowalk_menu_norm=None, autowalk_dialog_open=False,
        autowalk_continue_last_norm=None, autowalk_ok_norm=None))
    c.now = (c.switch_autowalk_since + c.cfg.autowalk.budget_s
             + c.cfg.autowalk.close_grace_s + 1.0)
    effects = fsm.step(obs(on_map=True), c)
    assert not any(isinstance(e, Tap) for e in effects)
    tr = [e for e in effects if isinstance(e, Transition)]
    assert len(tr) == 1
    assert tr[0].to is BotState.SCANNING and tr[0].outcome is IntentOutcome.CONFIRMED
    assert "TrainerTwo" in tr[0].reason
    assert phase in tr[0].reason         # names which phase was holding - see task brief


@pytest.mark.parametrize("phase", ["autowalk_open", "autowalk_menu", "autowalk_dialog",
                                   "autowalk_close"])
def test_the_deadline_is_reachable_even_when_the_screen_reads_off_map(phase):
    """Regression: an `obs.on_map` gate ahead of `_autowalk_deadline` made the ladder's
    own escape hatch unreachable on exactly the ticks it exists to escape - a menu/dialog
    overlay genuinely up, with its target unlocatable (e.g. a future id rename), reading
    off-map. The deadline must still fire and the switch must still confirm; without the
    fix this returned [] forever and the switch instead expired at the full 240s
    switch_timeout, recording a fully-successful switch as EXPIRED.

    Nothing is ever located here, including the star, so - same reasoning as the test
    above - the extra `close_grace_s` allowance has to run out too before this confirms."""
    c = awctx(phase=phase, since=100.0, accounts=_panel_with(
        star_norm=None, autowalk_menu_norm=None, autowalk_dialog_open=False,
        autowalk_continue_last_norm=None, autowalk_ok_norm=None))
    c.now = (c.switch_autowalk_since + c.cfg.autowalk.budget_s
             + c.cfg.autowalk.close_grace_s + 1.0)
    effects = fsm.step(obs(on_map=False, screen="Rocket", conf=0.99), c)
    assert not any(isinstance(e, Tap) for e in effects)
    tr = [e for e in effects if isinstance(e, Transition)]
    assert len(tr) == 1
    assert tr[0].to is BotState.SCANNING and tr[0].outcome is IntentOutcome.CONFIRMED


@pytest.mark.parametrize("phase", ["autowalk_open", "autowalk_menu", "autowalk_dialog",
                                   "autowalk_close"])
def test_the_budget_does_not_fire_early(phase):
    c = awctx(phase=phase, since=100.0, accounts=_panel_with(
        star_norm=None, autowalk_menu_norm=None, autowalk_dialog_open=False,
        autowalk_continue_last_norm=None, autowalk_ok_norm=None))
    c.now = c.switch_autowalk_since + c.cfg.autowalk.budget_s - 1.0
    effects = fsm.step(obs(on_map=True), c)
    assert effects == []


def test_a_located_node_is_still_used_even_after_the_deadline_has_passed():
    """Ordering within `_autowalk_dialog` (and its siblings): a target that IS present
    is always preferred over giving up, however far past the budget the clock has run -
    the deadline exists for the "never going to appear" case, not to race a real one."""
    c = awctx(phase="autowalk_dialog", since=100.0, accounts=_panel_with(
        autowalk_dialog_open=True, autowalk_continue_last_norm=None,
        autowalk_ok_norm=(0.80, 0.90)))
    c.now = c.switch_autowalk_since + c.cfg.autowalk.budget_s + 1.0   # PAST the deadline
    effects = fsm.step(obs(on_map=True), c)
    taps = [e for e in effects if isinstance(e, Tap)]
    assert len(taps) == 1 and (taps[0].x, taps[0].y) == pytest.approx((0.80, 0.90))
    assert not any(isinstance(e, Transition) for e in effects)


# ------------------------------------------------------------------ never on a failed/timed-out switch

def test_autowalk_is_never_reached_from_a_verify_mismatch():
    """`_autowalk_open` is reachable only through `_goplus`'s completion, itself only
    through `_zoom`'s, itself only through a `_verify` MATCH - so a switch that never
    confirms can never even open the star widget."""
    c = switching_ctx(phase="verify", accounts=panel(active="TrainerOne"))
    effects = fsm.step(obs(on_map=True), c)
    assert not any(isinstance(e, SetFlag) and e.name == "switch_phase"
                   and str(e.value).startswith("autowalk") for e in effects)


@pytest.mark.parametrize("phase", ["autowalk_open", "autowalk_menu",
                                    "autowalk_dialog", "autowalk_close"])
def test_the_state_timeout_still_ends_a_switch_stuck_in_any_autowalk_phase(phase):
    """Every autowalk phase is dispatched underneath `Switching.step`, which the shared
    `fsm.step` dispatcher only ever reaches once `ctx.elapsed <= handler.timeout(ctx)` -
    the timeout check runs BEFORE the phase dispatch, exactly like every other phase, so
    a switch that somehow never even gets as far as its own AutoWalk budget still ends at
    `Timings.switch_timeout`, not by hanging forever."""
    c = switching_ctx(phase=phase, target="TrainerTwo", cfg=budget(17.0),
                      accounts=_panel_with(star_norm=None, autowalk_menu_norm=None,
                                            autowalk_dialog_open=False))
    c.now = c.state_since + 18.0
    effects = fsm.step(obs(on_map=True), c)
    tr = [e for e in effects if isinstance(e, Transition)][0]
    assert tr.to is BotState.RECOVERING and tr.outcome is IntentOutcome.EXPIRED


def test_a_failed_switch_never_taps_the_star():
    """Driven the way test_switch_zoom.py/test_goplus.py drive the equivalent guards for
    their own gestures: a real failed switch, through the real Runner and FSM, produces
    no autowalk-shaped tap anywhere in its applied effects."""
    from tests.test_switch_runner import _fail_a_switch, _quota_switcher
    r = _quota_switcher()
    assert _fail_a_switch(r, r.ctx.now + 1.0, tap_login=True)
    assert not any("autowalk" in getattr(e, "reason", "") for e in r.actuator.applied)


# ------------------------------------------------------------------ end to end (real Runner)

def _drive(r, o):
    r._refresh_accounts(r.ctx.now)
    r.apply(fsm.step(o, r.ctx), o)


def _drive_until_scanning(r, o, dt, seconds=60.0):
    """Tick the real Runner at a FIXED interval until the switch confirms or `seconds` of
    simulated wall clock have gone by, and answer how long it took.

    `dt` is the whole point of these tests, not a detail. `_refresh_accounts` is throttled
    to `runner.ACCOUNTS_REFRESH` (2.5s) and `Runner.apply` drops `ctx.accounts` after every
    actuation taken while SWITCHING, so any tick SHORTER than that throttle leaves the
    ladder looking at a `None` view for several ticks after each of its own taps - which is
    the live loop's real condition, since it ticks at frame rate. A test that advances the
    clock by more than the throttle per tick never sees that gap at all and can pass while
    the ladder is silently skipping a step.

    Bounded by simulated seconds rather than a tick count so the bound means the same thing
    at every cadence, and set well clear of both `AutoWalk.budget_s` (30s) and
    `Timings.switch_timeout` (240s) so it is the FSM, never this loop, that ends the switch.
    """
    t0 = r.ctx.now
    while r.ctx.state is not BotState.SCANNING and r.ctx.now - t0 < seconds:
        r.ctx.now = round(r.ctx.now + dt, 6)
        _drive(r, o)
    return r.ctx.now - t0


def _autowalk_reasons(r):
    return [e.reason for e in r.actuator.applied
            if isinstance(e, Tap) and "autowalk" in e.reason]


LADDER = [
    "autowalk: open the PGSharp shortcut menu",
    "autowalk: select AutoWalk from the shortcut menu",
    "autowalk: CONTINUE LAST",
    "autowalk: close the shortcut menu",
]

STAR = (0.08, 0.23)


def _autowalk_runner(tmp_path, cfg=None, tree_reader=None, active="TrainerTwo",
                     target="TrainerTwo"):
    kw = {"stats_path": tmp_path / "sessions.jsonl", "roster": ROSTER,
          "tree_reader": tree_reader or FakeTreeReader([_full_view(active)])}
    if cfg is not None:
        kw["cfg"] = cfg
    r = make_runner(**kw)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.now = 1_000.0
    r._begin_switch(target)
    r._accounts_read_at = 0.0
    return r


def _full_view(active="TrainerTwo"):
    return replace(panel(active=active), star_norm=STAR,
                   autowalk_menu_norm=(0.30, 0.46), autowalk_dialog_open=True,
                   autowalk_continue_last_norm=(0.30, 0.90), autowalk_ok_norm=(0.80, 0.90))


#: The runner ticks at frame rate, so 0.1s is the cadence the live loop actually has and
#: the case that has to hold. 0.5s and 1.0s are still under the 2.5s tree-refresh throttle
#: and cover the same gap at coarser granularity for free. 3.0s is kept deliberately: it is
#: the ONLY cadence at which a tick is longer than the throttle, so the view is never
#: `None` when a phase looks - and it is exactly the cadence at which the old
#: `_autowalk_close` passed this test while emitting no close tap at any real one.
TICK_INTERVALS = [0.1, 0.5, 1.0, 3.0]


@pytest.mark.parametrize("dt", TICK_INTERVALS)
def test_a_confirmed_switch_runs_autowalk_end_to_end(tmp_path, dt):
    """Driven through the real Runner and real FSM, not constructed by hand: the target
    is already logged in (the "already active" shortcut in `Switching.step`, same as
    test_switch_zoom.py's equivalent test), so the ladder runs open -> settle -> verify
    -> zoom x2 -> goplus (absent) -> autowalk_open -> autowalk_menu -> autowalk_dialog ->
    autowalk_close -> confirmed.

    Parametrized over the tick interval, and that is the load-bearing part: an earlier
    version of this test advanced the clock by 3.0s per tick - longer than the 2.5s
    tree-refresh throttle - so `ctx.accounts` was repopulated before every phase looked at
    it, and all four taps appeared. At every cadence the runner really has (it ticks at
    frame rate) the view was `None` on the tick after the dialog press and the close tap
    never fired at all. The test passed; the bot did not close the menu.

    One view carries every field at once - the star, the open menu, the open dialog -
    which no single real dump could ever show together, but `_refresh_accounts` re-reads
    on every tick regardless of which phase is active (see runner.py), so a queue timed
    to "the right view for the right phase" would depend on exactly how many incidental
    reads the earlier phases (open/settle/zoom/goplus - none of which even consult
    `ctx.accounts`) happen to consume first. What is actually under test here - that the
    ladder advances one step per tick, in order, driven by the real Runner - does not
    need a realistic dump to prove; this module's phase-level tests above already cover
    each step's OWN reaction to a partial view in isolation. The menu-toggle consequence
    of a MISSING close tap does need one, and gets it in the two-switch test below.
    """
    r = _autowalk_runner(tmp_path)
    elapsed = _drive_until_scanning(r, obs(on_map=True), dt)
    assert r.ctx.state is BotState.SCANNING
    assert r.stats.account == "TrainerTwo"          # _on_switch_confirmed actually ran
    assert _autowalk_reasons(r) == LADDER
    star_taps = [(e.x, e.y) for e in r.actuator.applied
                 if isinstance(e, Tap) and "autowalk" in e.reason
                 and (e.x, e.y) == pytest.approx(STAR)]
    assert len(star_taps) == 2         # open, then close
    # And the whole ladder still fits inside its own wall-clock budget with room to
    # spare, at every cadence - the waiting the close phase now does is paid for by the
    # budget that already existed, not by a new one.
    assert elapsed < Config().autowalk.budget_s


def test_the_close_tap_is_the_one_a_sub_second_tick_used_to_lose(tmp_path):
    """Pins the regression to the exact tap, not merely to "four taps happen".

    At 0.1s a tick the old `_autowalk_close` produced the first three taps and confirmed
    the switch with the menu still open. Anyone reintroducing that shape - confirming on a
    `None` view rather than waiting for the refresh - fails here with a diff that names
    the missing step."""
    r = _autowalk_runner(tmp_path)
    _drive_until_scanning(r, obs(on_map=True), 0.1)
    assert "autowalk: close the shortcut menu" in _autowalk_reasons(r)


@pytest.mark.parametrize("dt", TICK_INTERVALS)
def test_a_confirmed_switch_skips_an_already_active_autowalk(tmp_path, dt):
    """The committed fixture (tests/fixtures/{uiautomator,screens}/autowalk_menu_active.*)
    is the ONE captured moment PGSharp ever showed AutoWalk already running for an
    account. Driven through the real Runner: the ladder must skip the "select AutoWalk"
    tap and the whole dialog phase, close the menu via the located star, and still let
    the switch confirm - at every tick cadence the runner really has (the task brief's
    own "watch the tick cadence" warning).

    Only the icon box and the frame it is read from are the REAL fixture; the rest of the
    view (star, account rows, login) is `_full_view`'s own synthetic composite - see that
    helper's docstring for why one view stands in for several real dumps at once here.
    `r._last_frame` is set directly, the same way `tree_reader` is pre-seeded rather than
    simulating a live capture source - `_refresh_accounts` reads it every tick exactly as
    the real loop would (see runner.py), it just never changes here because nothing in
    this harness ever re-reads a camera.
    """
    real_icon_rect = view("autowalk_menu_active.xml").autowalk_icon_rect_norm
    assert real_icon_rect is not None                      # sanity: the fixture parses
    combined = replace(_full_view("TrainerTwo"), autowalk_icon_rect_norm=real_icon_rect)
    r = _autowalk_runner(tmp_path, tree_reader=FakeTreeReader([combined]))
    r._last_frame = Frame(seq=1, ts=0.0, bgr=_active_bgr())
    elapsed = _drive_until_scanning(r, obs(on_map=True), dt)
    assert r.ctx.state is BotState.SCANNING
    assert r.stats.account == "TrainerTwo"                 # _on_switch_confirmed ran
    reasons = _autowalk_reasons(r)
    assert reasons == [
        "autowalk: open the PGSharp shortcut menu",
        "autowalk: close the shortcut menu",
    ]
    star_taps = [(e.x, e.y) for e in r.actuator.applied
                 if isinstance(e, Tap) and "autowalk" in e.reason
                 and (e.x, e.y) == pytest.approx(STAR)]
    assert len(star_taps) == 2                             # open, then close
    assert elapsed < Config().autowalk.budget_s


class PGSharpStar:
    """A tree reader that models the ONE PGSharp behaviour the two-switch regression is
    about, and that no static view can express: the star TOGGLES its shortcut menu.

    Everything it models is already recorded as measured live in `fsm.Switching`'s own
    docstrings - the star toggles the menu (`Runner.apply`'s reason for dropping the view
    after every switch-time actuation), picking "AutoWalk" from an open menu opens the
    dialog, and dismissing the dialog leaves the MENU still up (`_autowalk_close`'s
    reason for existing). Nothing here is invented behaviour, and it is driven purely by
    the taps the actuator actually ACCEPTED, so the model can never run ahead of the
    effects the FSM really emitted. Needs no device.
    """

    ITEM = (0.30, 0.46)
    CONTINUE_LAST = (0.30, 0.90)
    OK = (0.80, 0.90)

    def __init__(self, actuator, active="TrainerTwo"):
        self.actuator = actuator
        self.active = active
        self.reads = 0

    @property
    def menu_open(self) -> bool:
        return self._state()[0]

    def _state(self):
        menu = dialog = False
        for e in self.actuator.applied:
            if not isinstance(e, Tap):
                continue
            p = (e.x, e.y)
            if p == STAR:
                menu = not menu
            elif p == self.ITEM and menu:
                dialog = True
            elif p in (self.CONTINUE_LAST, self.OK) and dialog:
                dialog = False           # the menu stays up behind it - measured live
        return menu, dialog

    def read(self):
        self.reads += 1
        menu, dialog = self._state()
        return replace(
            panel(active=self.active),
            star_norm=STAR,                                  # the widget is always there
            autowalk_menu_norm=self.ITEM if menu else None,
            autowalk_dialog_open=dialog,
            autowalk_continue_last_norm=self.CONTINUE_LAST if dialog else None,
            autowalk_ok_norm=self.OK if dialog else None)


def test_the_next_switch_still_finds_the_menu_closed_and_can_open_it(tmp_path):
    """The consequence that made the missing close tap worth fixing rather than tidying.

    The star toggles. A first switch that leaves the menu OPEN means the second switch's
    `_autowalk_open` taps the star into an already-open menu and toggles it SHUT -
    `_autowalk_menu` then finds no "AutoWalk" node, waits out the whole 30s budget and
    confirms without ever starting a route. AutoWalk would work on the first switch of a
    run and silently stop for the rest of it, with nothing in the logs to say so.

    Driven at the live 0.1s cadence, against a tree reader that actually models the
    toggle, for two switches back to back.
    """
    r = make_runner(stats_path=tmp_path / "sessions.jsonl", roster=ROSTER)
    widget = PGSharpStar(r.actuator, active="TrainerTwo")
    r.tree_reader = widget
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.now = 1_000.0
    o = obs(on_map=True)

    r._begin_switch("TrainerTwo")
    r._accounts_read_at = 0.0
    _drive_until_scanning(r, o, 0.1)
    assert r.ctx.state is BotState.SCANNING
    assert _autowalk_reasons(r) == LADDER
    # The close tap did what it is for: the menu is shut, so it is neither sitting over
    # the reach ellipse SCANNING taps into nor waiting to be toggled shut by the next
    # switch's opening tap.
    assert widget.menu_open is False

    widget.active = "TrainerOne"
    r._begin_switch("TrainerOne")
    _drive_until_scanning(r, o, 0.1)
    assert r.ctx.state is BotState.SCANNING
    assert r.stats.account == "TrainerOne"
    # Both switches ran the full ladder - the second one found the menu to open, which is
    # only true because the first one closed it.
    assert _autowalk_reasons(r) == LADDER * 2
    assert widget.menu_open is False


def test_begin_switch_resets_the_autowalk_clock_between_attempts():
    """Same shape, same reasoning as `switch_login_ts` (fsm.Context's own docstring):
    a stale non-zero value inherited from a PRIOR attempt would make
    `Switching._autowalk_deadline` believe the new attempt's ladder has been running
    since the old one's, and could time it out before it ever starts."""
    r = make_runner(roster=ROSTER)
    r.ctx.switch_autowalk_since = 12_345.0
    r.ctx.state = BotState.SCANNING
    r._begin_switch("TrainerTwo")
    assert r.ctx.switch_autowalk_since == 0.0


@pytest.mark.parametrize("dt", TICK_INTERVALS)
def test_a_confirmed_switch_still_rolls_over_when_autowalk_finds_nothing(tmp_path, dt):
    """The whole point of the wall-clock budget: PGSharp not showing the star (a version
    change, a slow render, anything) must not turn an already-successful account switch
    into a recorded failure. Now that `_autowalk_close` waits for its view like its
    siblings, this has to hold at the sub-second cadences too - "waits" must not have
    become "hangs" at any tick rate."""
    cfg = Config(autowalk=AutoWalk(budget_s=5.0))
    r = _autowalk_runner(tmp_path, cfg=cfg,
                         tree_reader=FakeTreeReader([panel(active="TrainerTwo")]))
    _drive_until_scanning(r, obs(on_map=True), dt)
    assert r.ctx.state is BotState.SCANNING
    assert r.stats.account == "TrainerTwo"
    assert _autowalk_reasons(r) == []


@pytest.mark.parametrize("dt", TICK_INTERVALS)
def test_the_open_menu_is_closed_via_the_located_star_before_the_switch_confirms(tmp_path, dt):
    """THE live failure this task exists to fix, reproduced end to end.

    `_autowalk_open` locates and taps the star (it is always present in `stuck` below,
    exactly like a real PGSharp overlay), opening the shortcut menu - but the screen
    underneath never lets 'AutoWalk' render, the way a post-login story dialogue did on
    the device: `autowalk_menu_norm` stays None forever. The old code's give-up path
    confirmed the switch with the menu still open - on the device this cascaded into
    SCANNING tapping into the menu, RECOVERING never finding the map, and HALTED 88s
    later, over one switch that had actually already succeeded.

    `budget_s`/`close_grace_s` are kept small so the test runs fast, but both are still
    real multiples of `runner.ACCOUNTS_REFRESH` (2.5s) and `Timings.switch_tap` (2.0s) -
    the same shape the live defaults use - so `_autowalk_open` itself gets a fair chance
    at the star and this stays a test of the MENU-STUCK case, not of that different
    (already-acceptable, per the task brief) risk. Driven at every cadence the runner
    really has - the task brief's own "watch the tick cadence" warning, and the reason
    the missing `_autowalk_close` tap survived once already on this branch (see the
    module docstring above).
    """
    cfg = Config(autowalk=AutoWalk(budget_s=8.0, close_grace_s=6.0))
    stuck = replace(panel(active="TrainerTwo"), star_norm=STAR,
                    autowalk_menu_norm=None, autowalk_dialog_open=False)
    r = _autowalk_runner(tmp_path, cfg=cfg, tree_reader=FakeTreeReader([stuck]))
    elapsed = _drive_until_scanning(r, obs(on_map=True), dt, seconds=60.0)
    assert r.ctx.state is BotState.SCANNING
    assert r.stats.account == "TrainerTwo"           # the switch itself still confirmed
    star_taps = [(e.x, e.y) for e in r.actuator.applied
                 if isinstance(e, Tap) and (e.x, e.y) == pytest.approx(STAR)]
    assert len(star_taps) == 2                       # open, then the cleanup close
    close_taps = [e for e in r.actuator.applied if isinstance(e, Tap)
                 and "close the shortcut menu" in e.reason and "giving up" in e.reason]
    assert len(close_taps) == 1
    # Bounded, not merely eventual - the cleanup never pushed anywhere near
    # switch_timeout (240s), regardless of how much quantization overhead a coarse tick
    # adds on top of the ladder's own budget + grace.
    assert elapsed < cfg.timings.switch_timeout / 2
