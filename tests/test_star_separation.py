"""PGSharp's two floating widgets collapse onto each other after an app restart.

Measured on the device immediately after `effects.RestartApp` relaunched the game, and
committed as tests/fixtures/uiautomator/overlay_collapsed.xml:

    star     clickable rect (0,152)-(108,260)   centre (54,206)
    launcher clickable rect (0,152)-(272,245)   centre (136,198)
    overlap  108 x 93 px - very nearly the whole star

The star's own CENTRE lies inside the launcher's clickable rect, and that centre is
precisely what `accounts.AccountView.star_norm` reports and what
`fsm.Switching._autowalk_open` taps. So in this layout a tap meant for the star is
delivered to the accounts launcher, which opens the PGSharp accounts panel - the exact
screen tests/test_panel_recovery.py exists for, and the one that wedged a run into a
RECOVERING -> SCANNING livelock for 39% of its frames. The restart that ladder ends with
is what produces the layout, so without this the bot causes the wedge it recovers from.

Before the restart the same two widgets were well separated (tests/fixtures/uiautomator/
overlay_closed.xml and star_moved.xml), which is what makes this a post-restart layout
change rather than how PGSharp always draws itself.

Two layers, tested separately, the same split tests/test_autowalk.py uses:

  * `accounts.parse_dump` and `AccountView`, against the real fixture XML: the two
    clickable rects, whether they overlap, and where a drag has to land.
  * `fsm.Switching._separate_star`, the rung that emits the drag, plus the runner
    bookkeeping that bounds it.

The drag itself is not assumed to work, and neither are these tests. Verified twice on the
device, `adb shell input swipe` on this overlay does NOT land where it is aimed: asking for
y=626 landed the star's centre at 837, asking for 339 landed at 443, and asking for 356
moved it the other way entirely, to 125. That is why the implementation re-reads the tree
and judges the result, and why `_Device` below only moves the star after a configurable
number of drags rather than after the first.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from pogobot import fsm
from pogobot import runner as runner_mod
from pogobot.accounts import AccountView, parse_dump
from pogobot.actions import Actuator
from pogobot.config import Config
from pogobot.effects import (
    BotState,
    IntentOutcome,
    Note,
    SetFlag,
    Swipe,
    Tap,
    Transition,
)
from tests.factories import det, obs
from tests.test_autowalk import LADDER, _autowalk_reasons, _drive_until_scanning
from tests.test_switch_runner import ROSTER, make_runner
from tests.test_switching import ctx as switching_ctx, panel

FIX = Path(__file__).parent / "fixtures" / "uiautomator"
WH = (1080, 2340)


def view(name: str) -> AccountView:
    return parse_dump((FIX / name).read_bytes(), WH)


#: The two real layouts, read from the committed dumps rather than retyped.
COLLAPSED = view("overlay_collapsed.xml")
SEPARATED = view("overlay_closed.xml")


def swipes(effects):
    return [e for e in effects if isinstance(e, Swipe)]


def taps(effects):
    return [e for e in effects if isinstance(e, Tap)]


def kinds(effects, t):
    return [e for e in effects if isinstance(e, t)]


# --------------------------------------------------- accounts.py: the two rects

def test_the_collapsed_dump_reports_both_widgets_own_rects():
    """Centres are not enough and never were: two points cannot say whether two widgets
    overlap. These are the clickable ancestors' OWN bounds, the same nodes `star_norm` and
    `launcher_norm` are the centres of."""
    assert COLLAPSED.star_rect_norm == pytest.approx((0 / 1080, 152 / 2340,
                                                      108 / 1080, 260 / 2340))
    assert COLLAPSED.launcher_rect_norm == pytest.approx((0 / 1080, 152 / 2340,
                                                          272 / 1080, 245 / 2340))
    # ...and each rect's own centre is exactly the coordinate already published for it,
    # so the two fields can never describe two different widgets.
    for rect, centre in ((COLLAPSED.star_rect_norm, COLLAPSED.star_norm),
                         (COLLAPSED.launcher_rect_norm, COLLAPSED.launcher_norm)):
        assert centre == pytest.approx(((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2))


def test_the_collapsed_dump_reports_the_overlap():
    """The regression itself, off the real dump."""
    assert COLLAPSED.overlay_collapsed is True


def test_the_star_centre_the_bot_would_tap_is_inside_the_launcher():
    """Why the overlap is not cosmetic. `star_norm` is the coordinate
    `_autowalk_open` taps, and in this layout it lands on the accounts launcher."""
    x, y = COLLAPSED.star_norm
    x0, y0, x1, y1 = COLLAPSED.launcher_rect_norm
    assert x0 <= x <= x1 and y0 <= y <= y1


@pytest.mark.parametrize("name", ["overlay_closed.xml", "star_moved.xml",
                                  "accounts_open.xml"])
def test_a_healthy_layout_reports_no_overlap(name):
    """Every other committed dump of this overlay - including the one where the star has
    been dragged to a different position, and the one with the account panel open. A test
    that only ever sees the broken layout cannot tell a detector from a constant."""
    assert view(name).overlay_collapsed is False


@pytest.mark.parametrize("missing", ["hl_floating_icon", "hl_cd_text"])
def test_a_widget_that_was_not_found_is_not_a_separation(missing):
    """"Could not look" must never read as "they are separated" - the same distinction
    `available` draws for the dump as a whole. Renaming one anchor id is how a PGSharp
    update would produce this, and the honest answer is then False in both directions:
    nothing is dragged, and nothing claims the layout is safe either."""
    xml = (FIX / "overlay_collapsed.xml").read_bytes().replace(missing.encode(), b"hl_gone")
    v = parse_dump(xml, WH)
    assert (v.star_rect_norm is None) or (v.launcher_rect_norm is None)
    assert v.overlay_collapsed is False


# ------------------------------------------------- accounts.py: the landing zone

def test_the_landing_zone_is_derived_from_the_launchers_own_bottom_edge():
    """Not a constant: the launcher's own bottom edge, plus half a star so the star's top
    edge is level with it, plus one more whole star height of margin."""
    star, launcher = COLLAPSED.star_rect_norm, COLLAPSED.launcher_rect_norm
    height = star[3] - star[1]
    assert COLLAPSED.star_clear_y_norm == pytest.approx(
        launcher[3] + height / 2.0 + height)
    # In the device pixels the measurement was taken in: launcher bottom 245, star 108
    # tall, so the star's centre goes to 245 + 54 + 108 = 407 and its top edge to 353.
    assert COLLAPSED.star_clear_y_norm * 2340 == pytest.approx(407.0)


def test_the_landing_zone_actually_clears_the_launcher():
    """The claim the arithmetic exists to make, checked as an overlap rather than as a
    number - the same test `overlay_collapsed` itself applies."""
    star, launcher = COLLAPSED.star_rect_norm, COLLAPSED.launcher_rect_norm
    height = star[3] - star[1]
    y = COLLAPSED.star_clear_y_norm
    landed = replace(COLLAPSED,
                     star_rect_norm=(star[0], y - height / 2, star[2], y + height / 2))
    assert landed.overlay_collapsed is False
    # And with a whole star height to spare, which is what makes it survive a drag that
    # falls short - measured, this gesture misses by hundreds of pixels either way.
    assert (y - height / 2) - launcher[3] == pytest.approx(height)


def test_the_landing_zone_follows_the_launcher_rather_than_remembering_a_place():
    """Move the launcher and the answer moves with it. A hardcoded landing coordinate
    passes every test above and this one alone catches it - and the widget it is aiming
    below is itself draggable, so a remembered place is a place it will not be."""
    launcher = COLLAPSED.launcher_rect_norm
    shifted = replace(COLLAPSED,
                      launcher_rect_norm=(launcher[0], launcher[1] + 0.1,
                                          launcher[2], launcher[3] + 0.1))
    assert shifted.star_clear_y_norm == pytest.approx(COLLAPSED.star_clear_y_norm + 0.1)
    assert shifted.star_clear_y_norm != COLLAPSED.star_clear_y_norm


def test_no_landing_zone_is_named_off_the_bottom_of_the_screen():
    """A drag has two endpoints and both have to be somewhere the screen actually has.
    Refusing is what stops this inventing one; `_separate_star` then simply waits, and the
    AutoWalk ladder's own deadline ends the phase."""
    star = (0.0, 0.90, 0.1, 0.96)
    v = AccountView(available=True, star_rect_norm=star,
                    launcher_rect_norm=(0.0, 0.90, 0.25, 0.95))
    assert v.overlay_collapsed is True
    assert v.star_clear_y_norm is None


@pytest.mark.parametrize("field", ["star_rect_norm", "launcher_rect_norm"])
def test_no_landing_zone_without_both_rects(field):
    assert replace(COLLAPSED, **{field: None}).star_clear_y_norm is None


# --------------------------------------------------------- the tap reach ellipse

def test_the_landing_zone_is_outside_the_tap_reach_ellipse():
    """Where the bot leaves the star must be somewhere a target detection can never be
    tapped, or separating the widgets would only have moved the collision.

    Asserted against `fsm.reach_distance` - the same function `pick_target` itself calls,
    not a re-typed copy of the ellipse - so this keeps meaning something if the ellipse is
    ever retuned. Measured: reach centre (0.5, 0.63), radii (0.38, 0.16), tolerance 1.05;
    the landing point is 3.09 radii out.
    """
    cfg = Config()
    star = COLLAPSED.star_rect_norm
    x = (star[0] + star[2]) / 2.0
    d = fsm.reach_distance(cfg, x, COLLAPSED.star_clear_y_norm)
    assert d > cfg.reach.tolerance, f"the star would land {d:.2f} radii out, inside reach"


def test_no_detection_at_the_landing_zone_is_ever_targeted():
    """The same claim through the real selection code rather than through the formula:
    the strongest possible detection, sitting exactly where the star ends up, is not
    picked. `stop_scale` widens the ellipse for stops, so both target kinds are checked."""
    star = COLLAPSED.star_rect_norm
    x = (star[0] + star[2]) / 2.0
    y = COLLAPSED.star_clear_y_norm
    c = fsm.Context(cfg=Config(), now=100.0)
    for name in ("pokemon", "pokestop"):
        assert fsm.pick_target(obs(detections=[det(name=name, conf=0.99, cx=x, cy=y)]),
                               c) is None
    # Control: the same detection at the reach ellipse's own centre IS picked, so the
    # assertion above is about the position and not about some other refusal.
    assert fsm.pick_target(obs(detections=[det(conf=0.99, cx=0.5, cy=0.63)]), c) is not None


def test_reach_distance_is_the_definition_pick_target_uses():
    """One ellipse, one definition. `pick_target` used to inline the arithmetic; a copy of
    it here or anywhere else is how a check drifts away from the rule it checks."""
    cfg = Config()
    inside = fsm.reach_distance(cfg, cfg.reach.center_x, cfg.reach.center_y)
    assert inside == pytest.approx(0.0)
    edge = fsm.reach_distance(cfg, cfg.reach.center_x + cfg.reach.radius_x,
                              cfg.reach.center_y)
    assert edge == pytest.approx(1.0)
    c = fsm.Context(cfg=cfg, now=100.0)
    just_out = cfg.reach.center_y + cfg.reach.radius_y * (cfg.reach.tolerance + 0.05)
    assert fsm.pick_target(obs(detections=[det(cy=just_out)]), c) is None


# ------------------------------------------------------ fsm: the separation rung

def _switch_view(collapsed=True, panel_open=False, **kw):
    """A view as it reads during the AutoWalk phases: the account panel already shut (see
    `_Device`), carrying one of the two REAL widget layouts."""
    src = COLLAPSED if collapsed else SEPARATED
    return replace(panel(active="TrainerTwo"), panel_open=panel_open,
                   launcher_norm=src.launcher_norm,
                   launcher_rect_norm=src.launcher_rect_norm,
                   star_norm=src.star_norm, star_rect_norm=src.star_rect_norm, **kw)


def awctx(collapsed=True, cfg=None, **kw):
    accounts = kw.pop("accounts", None)
    c = switching_ctx(phase="autowalk_open", target="TrainerTwo",
                      accounts=accounts if accounts is not None
                      else _switch_view(collapsed), **({"cfg": cfg} if cfg else {}))
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_a_collapsed_pair_is_dragged_before_the_star_is_tapped():
    out = fsm.step(obs(on_map=True), awctx(collapsed=True))
    assert len(swipes(out)) == 1
    assert not taps(out), "the star must not be tapped while it sits on the launcher"
    assert not any(isinstance(e, SetFlag) and e.name == "switch_phase" for e in out), \
        "the phase must not advance on a star that was never tapped"


def test_a_separated_pair_taps_the_star_and_drags_nothing():
    """The other half, and the one that keeps this from firing on every switch: the
    healthy layout is left completely alone."""
    out = fsm.step(obs(on_map=True), awctx(collapsed=False))
    assert not swipes(out)
    assert len(taps(out)) == 1
    assert (taps(out)[0].x, taps(out)[0].y) == pytest.approx(SEPARATED.star_norm)
    assert any(isinstance(e, SetFlag) and e.name == "switch_phase"
               and e.value == "autowalk_menu" for e in out)


def test_both_drag_endpoints_come_from_the_located_rects():
    """Buttons - and now drag endpoints - are LOCATED, never assumed. The start is the
    star rect's own centre and the end is derived from the launcher rect's own bottom
    edge; neither is a remembered coordinate, which for two widgets that both float is
    the difference between a fix and a silent miss."""
    sw = swipes(fsm.step(obs(on_map=True), awctx(collapsed=True)))[0]
    star, launcher = COLLAPSED.star_rect_norm, COLLAPSED.launcher_rect_norm
    assert (sw.x1, sw.y1) == pytest.approx(((star[0] + star[2]) / 2,
                                            (star[1] + star[3]) / 2))
    assert sw.x2 == sw.x1, "the verified gesture was a straight vertical drag"
    assert sw.y2 == pytest.approx(COLLAPSED.star_clear_y_norm)
    height = star[3] - star[1]
    assert sw.y2 == pytest.approx(launcher[3] + height / 2.0 + height)
    assert sw.duration_ms == Config().star_separation.duration_ms


def test_the_drag_moves_with_the_widgets_rather_than_to_a_remembered_place():
    """Same shape as the landing-zone test above, but through the handler: shift both
    rects and every endpoint shifts with them."""
    def _down(rect):
        return tuple(v + 0.05 if i % 2 else v for i, v in enumerate(rect))

    shifted = replace(_switch_view(collapsed=True),
                      star_rect_norm=_down(COLLAPSED.star_rect_norm),
                      launcher_rect_norm=_down(COLLAPSED.launcher_rect_norm))
    base = swipes(fsm.step(obs(on_map=True), awctx(collapsed=True)))[0]
    moved = swipes(fsm.step(obs(on_map=True), awctx(accounts=shifted)))[0]
    assert moved.y1 == pytest.approx(base.y1 + 0.05)
    assert moved.y2 == pytest.approx(base.y2 + 0.05)


def test_nothing_is_dragged_while_the_accounts_panel_is_open():
    """During a switch the panel owns the screen, and the star's rect then names a
    coordinate underneath it. The star is not tapped either: a collapsed pair means the
    tap is unsafe whatever else is on screen."""
    out = fsm.step(obs(on_map=True), awctx(accounts=_switch_view(True, panel_open=True)))
    assert not swipes(out) and not taps(out)


def test_the_drag_is_paced_and_cannot_fire_every_frame():
    """`ctx.ready` on its own "star_drag" budget. The runner ticks at frame rate, and an
    ungated rung would spend the whole bound in an eighth of a second - before the tree
    has been re-read even once, so every one of those drags would be decided from the
    same view."""
    c = awctx(collapsed=True)
    c.last_action["star_drag"] = c.now
    assert not swipes(fsm.step(obs(on_map=True), c))


def test_a_withheld_drag_still_never_promotes_the_star_tap():
    """The empty answer is deliberate. Pacing this rung out must not fall through to the
    tap it is holding back - that tap is the one that opens the accounts panel."""
    c = awctx(collapsed=True)
    c.last_action["star_drag"] = c.now
    assert not taps(fsm.step(obs(on_map=True), c))


def test_a_star_that_will_not_move_gives_up_instead_of_tapping_it():
    """The bound. A star that has been dragged `max_drags` times and is still on top of
    the launcher is a star whose tap opens the accounts panel, so AutoWalk is skipped and
    the already-confirmed switch is released - never "try the tap anyway"."""
    cfg = Config()
    out = fsm.step(obs(on_map=True), awctx(collapsed=True,
                                           star_drags=cfg.star_separation.max_drags))
    assert not swipes(out) and not taps(out)
    tr = kinds(out, Transition)
    assert len(tr) == 1
    assert tr[0].to is BotState.SCANNING and tr[0].outcome is IntentOutcome.CONFIRMED
    assert "TrainerTwo" in tr[0].reason
    notes = kinds(out, Note)
    assert notes and notes[0].level == "warn", "the operator gets no other record of this"
    assert "accounts launcher" in notes[0].text


def test_the_bound_is_read_from_the_config():
    """A handler that hardcoded the number passes the test above and fails this one."""
    cfg = replace(Config(), star_separation=replace(Config().star_separation, max_drags=1))
    c = awctx(collapsed=True, cfg=cfg, star_drags=0)
    assert swipes(fsm.step(obs(on_map=True), c))
    c = awctx(collapsed=True, cfg=cfg, star_drags=1)
    assert not swipes(fsm.step(obs(on_map=True), c))
    assert kinds(fsm.step(obs(on_map=True), c), Transition)


def test_a_pair_that_never_separates_still_lets_the_switch_confirm():
    """The fall-through, and why the withheld answer is not a `return`: a collapsed star
    that this tick cannot act on must still reach the AutoWalk ladder's own wall-clock
    deadline (`config.AutoWalk.budget_s`), or a switch that has already succeeded hangs
    until `Timings.switch_timeout` (240s) and is recorded as a failure."""
    c = awctx(collapsed=True, switch_autowalk_since=100.0)
    c.now = c.switch_autowalk_since + c.cfg.autowalk.budget_s + 1.0
    c.last_action["star_drag"] = c.now      # paced out on this tick
    out = fsm.step(obs(on_map=True), c)
    assert not swipes(out) and not taps(out)
    tr = kinds(out, Transition)
    assert len(tr) == 1
    assert tr[0].to is BotState.SCANNING and tr[0].outcome is IntentOutcome.CONFIRMED


def test_the_handler_writes_nothing_to_the_context():
    """The FSM is pure: (Observation, Context) -> list[Effect]. Only the runner mutates
    the context, which is what keeps a dry run and a live run on the same trajectory - and
    in particular what stops this rung counting its own drags, which it cannot know landed.

    Both branches, each against its own context: the one that drags, and the one that has
    spent the bound and gives up."""
    for c in (awctx(collapsed=True),
              awctx(collapsed=True, star_drags=Config().star_separation.max_drags)):
        before = copy.deepcopy(c.__dict__)
        fsm.step(obs(on_map=True), c)
        assert c.__dict__ == before


def test_the_drag_renders_to_the_adb_gesture_that_was_verified_on_the_device():
    """The drag has to reach the phone as the command that was actually measured moving
    this widget: `input swipe <x> <y> <x> <target> <ms>` - one x, since the verified
    gesture was straight down, and the duration the effect names.

    Device pixels are the actuator's business, not the handler's, and this is where the
    normalized endpoints become the ones a human could read off `adb shell` - 54 across on
    a 1080-wide screen, 206 down to 407 on a 2340-tall one, which is exactly the
    separation that was performed by hand."""
    star = COLLAPSED.star_rect_norm
    x = (star[0] + star[2]) / 2.0
    e = Swipe(x, (star[1] + star[3]) / 2.0, x, COLLAPSED.star_clear_y_norm,
              "autowalk: drag the star clear of the accounts launcher",
              duration_ms=Config().star_separation.duration_ms, budget="star_drag")
    cmd = Actuator(WH, dry_run=True).render(e)
    assert cmd is not None
    assert cmd.argv[-7:-5] == ("input", "swipe")
    sx, sy, ex, ey, ms = cmd.argv[-5:]
    assert sx == ex, f"the verified gesture was vertical; got {sx} -> {ex}"
    assert (int(sx), int(sy)) == (54, 206)
    assert int(ey) == 407 and int(ey) > int(sy), \
        "the star has to move DOWN, past the launcher"
    assert int(ms) == Config().star_separation.duration_ms
    assert cmd.budget == "star_drag"


def test_the_verified_durations_are_the_only_ones_used():
    """400ms and 500ms were both run on the device and both moved the star. Nothing
    shorter was ever tested, and an `input swipe` short enough to read as a fling is a
    different gesture with a different outcome."""
    assert Config().star_separation.duration_ms in (400, 500)


# ------------------------------------------------------------ runner bookkeeping

def test_only_an_accepted_drag_is_charged_to_the_bound():
    """`_separate_star` is pure and cannot know whether its Swipe reached the device.
    Counting a refused one would spend a drag that never happened and could give up on a
    star nobody ever tried to move. Same reasoning as `switch_zoom_reps`,
    `switch_clear_presses` and `app_restarts`."""
    from tests.test_switch_zoom import _FlakyAct
    r = make_runner()
    r.actuator = _FlakyAct([False, True])
    r.ctx.state = BotState.SWITCHING
    e = Swipe(0.05, 0.09, 0.05, 0.17, "drag the star clear", budget="star_drag")

    r.apply([e], obs(on_map=True))
    assert r.ctx.star_drags == 0

    r.apply([e], obs(on_map=True))
    assert r.ctx.star_drags == 1


def test_the_bound_is_per_switch_attempt():
    """Reset on state entry, like `switch_goplus_attempts` and `switch_zoom_reps`:
    SWITCHING is entered once per attempt, which is exactly the scope of the bound. An
    attempt hours later must not inherit a spent budget from one whose layout has long
    since been dragged apart by hand."""
    assert "star_drags" in runner_mod._RESET_ON_ENTRY
    r = make_runner()
    r.ctx.star_drags = 3
    r._begin_switch("TrainerTwo")
    assert r.ctx.star_drags == 0


def test_an_applied_drag_drops_the_view_it_was_decided_from():
    """The property that makes "check the result, then try again" structural rather than a
    matter of pacing: `Runner.apply` drops `ctx.accounts` after every actuation taken while
    SWITCHING, so the NEXT drag can only ever be decided from a tree read taken after this
    one landed. Nothing here trusts a drag to have worked - measured, this gesture
    overshot by 211px once and moved the star the wrong way entirely another time."""
    r = make_runner()
    r.ctx.state = BotState.SWITCHING
    r.ctx.accounts = _switch_view(collapsed=True)
    r.apply([Swipe(0.05, 0.09, 0.05, 0.17, "drag the star clear", budget="star_drag")],
            obs(on_map=True))
    assert r.ctx.accounts is None


# ------------------------------------------------------------------ end to end

class _Device:
    """The phone, in the two respects this ladder depends on, both measured.

    The accounts panel is shut by the time the AutoWalk phases run - PGSharp closes it as
    part of logging in, and `_verify` closes it again - so `panel_open` follows the phase
    rather than being pinned. And the star moves only once the bot has actually dragged
    it, `drags_needed` times: `input swipe` on this overlay does not land where it is
    aimed (asking for y=626 landed the centre at 837, 339 landed at 443, and 356 went the
    other way to 125), so a device model that separates on the first drag would let an
    implementation that assumes its own drag worked pass.
    """

    def __init__(self, runner, drags_needed=1):
        self.r = runner
        self.drags_needed = drags_needed
        self.reads = 0

    def read(self):
        self.reads += 1
        drags = sum(1 for e in self.r.actuator.applied
                    if isinstance(e, Swipe) and e.budget == "star_drag")
        return _switch_view(
            collapsed=drags < self.drags_needed,
            panel_open=not str(self.r.ctx.switch_phase).startswith("autowalk"),
            autowalk_menu_norm=(0.30, 0.46), autowalk_dialog_open=True,
            autowalk_continue_last_norm=(0.30, 0.90), autowalk_ok_norm=(0.80, 0.90))


def _switch_runner(tmp_path, drags_needed=1, cfg=None):
    r = make_runner(**({"cfg": cfg} if cfg else {}),
                    stats_path=tmp_path / "sessions.jsonl", roster=ROSTER)
    r.tree_reader = _Device(r, drags_needed)
    r.stats.account = "TrainerOne"
    r.ctx.state = BotState.SCANNING
    r.ctx.now = 1_000.0
    r._begin_switch("TrainerTwo")
    r._accounts_read_at = 0.0
    return r


def _star_drags(r):
    return [e for e in r.actuator.applied
            if isinstance(e, Swipe) and e.budget == "star_drag"]


#: The runner ticks at frame rate, so 0.1s is the cadence the live loop actually has.
#: 1.0s is still under the 2.5s tree-refresh throttle and covers the same gap coarsely.
TICK_INTERVALS = [0.1, 1.0]


@pytest.mark.parametrize("dt", TICK_INTERVALS)
@pytest.mark.parametrize("drags_needed", [1, 2])
def test_the_bot_drags_the_star_clear_and_then_runs_autowalk(tmp_path, dt, drags_needed):
    """The whole fix through the real Runner and real FSM: a collapsed overlay is dragged
    apart - retrying when one drag was not enough - and only then is the star tapped.

    Parametrized over `drags_needed` because the retry is the part that cannot be taken on
    trust: with the device separating only on the second drag, an implementation that
    computes one drag and assumes it landed taps a star that is still on the launcher."""
    r = _switch_runner(tmp_path, drags_needed=drags_needed)
    _drive_until_scanning(r, obs(on_map=True), dt)

    assert len(_star_drags(r)) == drags_needed
    assert _autowalk_reasons(r) == LADDER, "the ladder never ran after the separation"
    assert r.ctx.state is BotState.SCANNING
    assert r.stats.account == "TrainerTwo"

    # Every star tap happened after the last drag, and no AutoWalk tap landed inside the
    # launcher's rect - which is the failure this whole file exists to prevent. Only the
    # AutoWalk taps: the switch's own "close overlay after verifying" tap aims at the
    # panel's close control, which really is drawn over the launcher while the panel is
    # up, and is the coordinate the tree named for it.
    applied = r.actuator.applied
    last_drag = max(i for i, e in enumerate(applied) if isinstance(e, Swipe)
                    and e.budget == "star_drag")
    star_taps = [i for i, e in enumerate(applied)
                 if isinstance(e, Tap) and (e.x, e.y) == pytest.approx(SEPARATED.star_norm)]
    assert star_taps and min(star_taps) > last_drag
    x0, y0, x1, y1 = COLLAPSED.launcher_rect_norm
    for e in applied:
        if isinstance(e, Tap) and e.reason.startswith("autowalk"):
            assert not (x0 <= e.x <= x1 and y0 <= e.y <= y1), \
                f"a tap landed inside the accounts launcher: {e.reason}"


def test_a_star_that_never_separates_is_never_tapped_at_all(tmp_path):
    """The bound, end to end. The device refuses to move the star however many times it is
    dragged; the run must spend `max_drags` and then leave it alone entirely rather than
    tapping a control that opens the accounts panel. The switch still confirms - it had
    already succeeded, and AutoWalk is the only thing lost."""
    cfg = Config()
    r = _switch_runner(tmp_path, drags_needed=cfg.star_separation.max_drags + 5)
    _drive_until_scanning(r, obs(on_map=True), 0.1)

    assert len(_star_drags(r)) == cfg.star_separation.max_drags
    assert _autowalk_reasons(r) == [], "the star was tapped even though it never moved"
    assert r.ctx.state is BotState.SCANNING
    assert r.stats.account == "TrainerTwo", "the confirmed switch was thrown away"


# ------------------------------------------- the landing zone vs. the tap reach ellipse

#: A collapsed pair whose landing zone falls INSIDE the reach ellipse. The widget sizes
#: are the device's own measurements (star 108x108, launcher 272x93 on a 1080x2340
#: screen); only the positions differ, and both widgets are draggable by the user - which
#: is the one thing the user is documented as doing with them ("you need to drag one
#: up/down"). Clearing the launcher is necessary and not sufficient: here it would take a
#: star that is 1.42 radii OUT of reach and park it 0.81 radii IN.
def _midscreen_pair():
    def n(x0, y0, x1, y1):
        return (x0 / 1080, y0 / 2340, x1 / 1080, y1 / 2340)
    return n(486, 890, 594, 998), n(400, 915, 672, 1008)


def test_the_midscreen_layout_really_would_move_the_star_into_reach():
    """The premise of the test below, established before it is relied on - otherwise a
    refusal proves nothing about the case it is supposed to be refusing."""
    cfg = Config()
    star, launcher = _midscreen_pair()
    v = AccountView(available=True, star_rect_norm=star, launcher_rect_norm=launcher)
    assert v.overlay_collapsed is True
    x = (star[0] + star[2]) / 2
    before = fsm.reach_distance(cfg, x, (star[1] + star[3]) / 2)
    after = fsm.reach_distance(cfg, x, v.star_clear_y_norm)
    assert before > cfg.reach.tolerance, "the star starts out of reach"
    assert after <= cfg.reach.tolerance, "and the landing zone is inside it"
    # Through the real selection code, not the formula: SCANNING would tap this spot, and
    # every such tap opens PGSharp's shortcut menu over the map (see `_autowalk_close`).
    c = fsm.Context(cfg=cfg, now=100.0)
    assert fsm.pick_target(
        obs(detections=[det(name="pokemon", conf=0.99, cx=x, cy=v.star_clear_y_norm)]),
        c) is not None


def test_a_landing_zone_inside_the_reach_ellipse_is_refused():
    """Separating the widgets must never merely move the collision. `star_clear_y_norm`
    knows about the launcher and the bottom of the screen and nothing else - it cannot
    know about the reach ellipse, because accounts.py is imported BY fsm.py - so the rung
    that acts on it is where the ellipse is consulted. The answer is to refuse, the same
    one the off-screen branch gives: this rung knows one landing zone, and if that one is
    unusable it has nothing to offer. The star is not tapped either."""
    star, launcher = _midscreen_pair()
    v = replace(_switch_view(collapsed=True),
                star_rect_norm=star, launcher_rect_norm=launcher,
                star_norm=((star[0] + star[2]) / 2, (star[1] + star[3]) / 2),
                launcher_norm=((launcher[0] + launcher[2]) / 2,
                               (launcher[1] + launcher[3]) / 2))
    assert v.overlay_collapsed is True
    out = fsm.step(obs(on_map=True), awctx(accounts=v))
    assert not swipes(out), "the star would have been dragged into the tap reach ellipse"
    assert not taps(out), "and a collapsed star must not be tapped either"


def test_the_real_collapsed_layout_is_not_refused_by_that_check():
    """The control. The measured post-restart layout lands 3.09 radii out, so the gate
    above must not be what stops the fix working on the case it was written for."""
    assert swipes(fsm.step(obs(on_map=True), awctx(collapsed=True)))


def test_the_refusal_uses_the_widest_ellipse_pick_target_can_use():
    """`pick_target` scales the ellipse for stop targets (`Reach.stop_scale`), so a
    landing zone clear of only the narrower of the two is not clear at all. Raising
    stop_scale alone must be enough to make a previously-safe landing zone refused."""
    star = COLLAPSED.star_rect_norm
    x = (star[0] + star[2]) / 2.0
    base = Config()
    # Wide enough that the real layout's landing zone falls inside the STOP ellipse only.
    wide = replace(base, reach=replace(base.reach, stop_scale=4.0))
    assert fsm.reach_distance(wide, x, COLLAPSED.star_clear_y_norm) > wide.reach.tolerance
    assert fsm.reach_distance(wide, x, COLLAPSED.star_clear_y_norm,
                              wide.reach.stop_scale) <= wide.reach.tolerance
    assert not swipes(fsm.step(obs(on_map=True), awctx(collapsed=True, cfg=wide)))


# ------------------------------------- the drags are not charged to the AutoWalk ladder

def test_an_accepted_drag_restarts_the_autowalk_ladders_clock():
    """`AutoWalk.budget_s` (30s) is sized for the four settle-and-reread cycles the LADDER
    needs - star, menu, dialog, close. Making the star tappable is not one of them, and it
    is not cheap: `Runner.apply` drops `ctx.accounts` after every applied effect, so each
    drag costs a whole tree-refresh cycle, and a cycle is not `ACCOUNTS_REFRESH` (2.5s) -
    `_refresh_accounts` stamps its throttle from when the read FINISHED and the dump was
    measured at ~3.0s, so usable views are ~5.5s apart. Three drags is ~16s of the 30s the
    ladder itself has to fit inside."""
    r = make_runner()
    r.ctx.state = BotState.SWITCHING
    r.ctx.now = 5_000.0
    r.ctx.switch_autowalk_since = 4_900.0
    r.apply([Swipe(0.05, 0.09, 0.05, 0.17, "drag the star clear", budget="star_drag")],
            obs(on_map=True))
    assert r.ctx.switch_autowalk_since == 5_000.0


def test_a_refused_drag_buys_no_time_at_all():
    """Which is what keeps the deadline a backstop rather than something a stuck rung can
    hold open forever: a drag the actuator never sent moved nothing, so a star that can
    never be dragged still ends the ladder at `budget_s` instead of holding the switch to
    `Timings.switch_timeout` (240s)."""
    from tests.test_switch_zoom import _FlakyAct
    r = make_runner()
    r.actuator = _FlakyAct([False])
    r.ctx.state = BotState.SWITCHING
    r.ctx.now = 5_000.0
    r.ctx.switch_autowalk_since = 4_900.0
    r.apply([Swipe(0.05, 0.09, 0.05, 0.17, "drag the star clear", budget="star_drag")],
            obs(on_map=True))
    assert r.ctx.switch_autowalk_since == 4_900.0
    assert r.ctx.star_drags == 0


def test_the_ladder_still_fits_its_budget_after_the_maximum_number_of_drags(
        tmp_path, monkeypatch):
    """End to end, at the tree-refresh cadence the DEVICE has rather than the one the
    constant names.

    `runner.ACCOUNTS_REFRESH` is 2.5s, but `_refresh_accounts` stamps the throttle from
    when the read FINISHED and `uiautomator dump` was measured at 2.96/3.00/3.00/3.00/4.46s
    against a rendering game - so consecutive usable views are ~5.5s apart, not 2.5s. Every
    other end-to-end test in this file reads instantly and therefore never sees this.

    The assertion is the ladder's OWN budget, not merely "it finished": with the drags
    charged to `AutoWalk.budget_s` the ladder ran 44s of a 30s budget here, which means
    every remaining rung was past its deadline and any node that took one extra read to
    appear would have been abandoned - `_autowalk_deadline` gives up on exactly that. The
    switch is bounded regardless by `Timings.switch_timeout`, which is checked too."""
    monkeypatch.setattr(runner_mod, "ACCOUNTS_REFRESH", 5.5)
    cfg = Config()
    r = _switch_runner(tmp_path, drags_needed=cfg.star_separation.max_drags)
    elapsed = _drive_until_scanning(r, obs(on_map=True), 0.1, seconds=200.0)

    assert len(_star_drags(r)) == cfg.star_separation.max_drags
    assert _autowalk_reasons(r) == LADDER
    assert r.ctx.now - r.ctx.switch_autowalk_since <= cfg.autowalk.budget_s, \
        "the ladder ran past its own wall-clock budget because the drags were charged to it"
    assert elapsed < cfg.timings.switch_timeout


def test_a_landing_zone_that_is_never_usable_still_lets_the_switch_confirm(tmp_path):
    """The two refusals meeting, end to end. A collapsed pair whose only landing zone is
    inside the reach ellipse can never be dragged and must never be tapped, so the only
    thing left to end the phase is the ladder's own wall-clock deadline.

    Which is precisely why only an ACCEPTED drag defers that deadline. If a refusal
    deferred it too, this would hold an already-successful switch until
    `Timings.switch_timeout` (240s) and `_on_switch_failed` would record it as a failure -
    turning a cosmetic overlay problem into a lost account switch."""
    star, launcher = _midscreen_pair()

    class _StuckInReach(_Device):
        def read(self):
            return replace(super().read(), star_rect_norm=star,
                           launcher_rect_norm=launcher,
                           star_norm=((star[0] + star[2]) / 2, (star[1] + star[3]) / 2))

    cfg = Config()
    r = _switch_runner(tmp_path, drags_needed=99)
    r.tree_reader = _StuckInReach(r, 99)
    elapsed = _drive_until_scanning(r, obs(on_map=True), 0.1, seconds=300.0)

    assert not _star_drags(r), "the star was dragged into the tap reach ellipse"
    assert _autowalk_reasons(r) == [], "a collapsed star was tapped"
    assert r.ctx.state is BotState.SCANNING
    assert r.stats.account == "TrainerTwo", "the confirmed switch was thrown away"
    assert elapsed <= cfg.autowalk.budget_s + cfg.autowalk.close_grace_s
