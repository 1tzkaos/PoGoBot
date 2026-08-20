from dataclasses import replace

import pytest

from pogobot import fsm
from pogobot.accounts import AccountRow, AccountView
from pogobot.config import Config
from pogobot.effects import (
    Back,
    BotState,
    DoubleTapDrag,
    IntentOutcome,
    SetFlag,
    Tap,
    Transition,
)
from tests.factories import obs


def row(name, active=False, x=0.74, level=10):
    return AccountRow(name=name, active=active, level=level,
                      login_norm=(x, 0.23), delete_norm=(0.89, 0.23))


def panel(active="TrainerOne", open_=True):
    return AccountView(
        rows=(row("TrainerOne", active == "TrainerOne"),
              row("TrainerTwo", active == "TrainerTwo")),
        launcher_norm=(0.12, 0.05), accounts_tab_norm=(0.83, 0.18),
        close_norm=(0.06, 0.10), available=True, panel_open=open_)


def budget(seconds: float) -> Config:
    """A Config whose switch budget is nothing like the default, so a handler that reads
    it can be told apart from one that hardcodes a number."""
    return Config(timings=replace(Config().timings, switch_timeout=seconds))


def ctx(**kw):
    c = fsm.Context(cfg=kw.pop("cfg", Config()), state=BotState.SWITCHING,
                    now=100.0, state_since=100.0)
    c.switch_target = kw.pop("target", "TrainerTwo")
    c.switch_phase = kw.pop("phase", "open")
    c.accounts = kw.pop("accounts", panel())
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def taps(effects):
    return [e for e in effects if isinstance(e, Tap)]


def test_closed_panel_taps_the_launcher():
    c = ctx(accounts=panel(open_=False))
    assert taps(fsm.step(obs(), c))[0].x == pytest.approx(0.12)


def test_open_panel_without_the_target_taps_the_accounts_tab():
    v = panel()
    c = ctx(accounts=AccountView(rows=(), launcher_norm=v.launcher_norm,
                                 accounts_tab_norm=v.accounts_tab_norm,
                                 close_norm=v.close_norm, available=True, panel_open=True))
    assert taps(fsm.step(obs(), c))[0].x == pytest.approx(0.83)


def test_target_row_present_taps_its_login_button():
    c = ctx()
    effects = fsm.step(obs(), c)
    assert taps(effects)[0].reason.endswith("TrainerTwo")
    assert any(isinstance(e, SetFlag) and e.value == "settle" for e in effects)


def panel_with_autowalk(active="TrainerOne"):
    """`panel()` plus the star/shortcut-menu/AutoWalk-dialog fields, all populated at
    once - a dump that could never really show an open account panel AND an open
    AutoWalk dialog together, but exercising the delete-guard against every coordinate
    this module can ever produce, together, is the point of the test below: it proves
    none of them is ever confused with a delete button, however unrealistic the
    combination that would require."""
    return replace(panel(active), star_norm=(0.08, 0.23),
                   autowalk_menu_norm=(0.30, 0.46), autowalk_dialog_open=True,
                   autowalk_continue_last_norm=(0.30, 0.90),
                   autowalk_ok_norm=(0.80, 0.90))


def test_no_tap_ever_lands_on_a_delete_button():
    """The delete button sits ~24px from login. This is the test that matters.

    Every phase `switch_phase` can actually hold - "tab" and "login" are steps WITHIN
    "open", not phases, and naming them here only ran the "open" case three times. "zoom"
    is included too: it is a screen-centre gesture nowhere near any row, but its endpoints
    are still coordinates this handler emits, and this is the test that matters for those.
    "goplus" needs `obs.goplus=FALSE` forced on top of the loop's own `obs()` call - with
    the suite-wide default of UNKNOWN that phase never taps at all, and a guard that never
    exercises the tap it exists to check proves nothing. The four "autowalk_*" phases need
    the same care: `switch_autowalk_since` has to be non-zero (as it would be by the time
    any phase past "autowalk_open" is reached) or the wall-clock deadline
    (`fsm.Switching._autowalk_deadline`) fires first and no tap - the thing under test -
    is ever emitted. "autowalk_menu" is run TWICE: once as every other phase is, and once
    more with `switch_autowalk_active=Tristate.TRUE` - the "already active" skip chains
    straight into `_autowalk_close` and emits its OWN star tap in the same tick, a
    coordinate this guard did not exercise before that rule existed.
    """
    from pogobot.observation import Tristate
    autowalk_phases = ("autowalk_open", "autowalk_menu", "autowalk_dialog", "autowalk_close")

    def _check(phase, on_map, **extra):
        c = ctx(phase=phase, accounts=panel_with_autowalk(), **extra)
        kw = {"goplus": Tristate.FALSE} if phase == "goplus" else {}
        effects = fsm.step(obs(on_map=on_map, **kw), c)
        points = [(t.x, t.y) for t in taps(effects)]
        points += [(d.x1, d.y1) for d in effects if isinstance(d, DoubleTapDrag)]
        points += [(d.x2, d.y2) for d in effects if isinstance(d, DoubleTapDrag)]
        for x, y in points:
            for row in c.accounts.rows:
                assert (x, y) != row.delete_norm
                assert abs(x - row.delete_norm[0]) > 0.02 or abs(y - row.delete_norm[1]) > 0.02

    for phase in ("open", "settle", "verify", "zoom", "goplus", *autowalk_phases):
        for on_map in (True, False):
            extra = {"switch_autowalk_since": 100.0} if phase in autowalk_phases[1:] else {}
            _check(phase, on_map, **extra)

    for on_map in (True, False):
        _check("autowalk_menu", on_map, switch_autowalk_since=100.0,
              switch_autowalk_active=Tristate.TRUE)


def test_unavailable_view_does_nothing_rather_than_guessing():
    c = ctx(accounts=AccountView(available=False))
    assert fsm.step(obs(), c) == []


def test_settle_presses_back_when_a_post_login_screen_is_up():
    c = ctx(phase="settle", accounts=panel(active="TrainerTwo"))
    # screen="Menu" so on_map is actually False - the factory's default screen guess
    # is "Overworld", which on its own would make on_map True regardless of this param.
    assert any(isinstance(e, Back)
               for e in fsm.step(obs(on_map=False, screen="Menu", conf=0.99), c))


def test_settle_prefers_a_located_close_button_over_back():
    c = ctx(phase="settle", accounts=panel(active="TrainerTwo"))
    effects = fsm.step(obs(on_map=False, screen="Menu", conf=0.99, close_xy=(0.5, 0.9)), c)
    assert taps(effects)[0].x == pytest.approx(0.5)
    assert not any(isinstance(e, Back) for e in effects)


def test_the_clearing_back_press_carries_its_own_budget():
    """`Runner.apply` counts an accepted press by matching this exact budget name (see
    config.Timings.switch_clear_max) - not the shared "back" bucket RECOVERING and the
    exit-dialog/keyboard interrupts also use, so an unrelated Back never eats into the
    bound this guards."""
    c = ctx(phase="settle", accounts=panel(active="TrainerTwo"))
    backs = [e for e in fsm.step(obs(on_map=False, screen="Menu", conf=0.99), c)
             if isinstance(e, Back)]
    assert len(backs) == 1
    assert backs[0].budget == "switch_clear"


def test_settle_stops_pressing_back_once_the_clear_bound_is_spent():
    """The fix for the measured 90-press BACK storm: once `switch_clear_presses` has
    reached `Timings.switch_clear_max`, `_settle` simply waits - it must not keep
    hammering BACK into what may be a legitimate multi-minute LOADING screen."""
    bound = Config().timings.switch_clear_max
    c = ctx(phase="settle", accounts=panel(active="TrainerTwo"),
           switch_clear_presses=bound)
    effects = fsm.step(obs(on_map=False, screen="Menu", conf=0.99), c)
    assert effects == []


def test_a_located_close_button_ignores_the_clear_bound():
    """A located close button is targeted, not blind - unlike the coordinate-free BACK
    fallback it is never subject to switch_clear_max."""
    bound = Config().timings.switch_clear_max
    c = ctx(phase="settle", accounts=panel(active="TrainerTwo"),
           switch_clear_presses=bound)
    effects = fsm.step(
        obs(on_map=False, screen="Menu", conf=0.99, close_xy=(0.5, 0.9)), c)
    assert taps(effects)[0].x == pytest.approx(0.5)
    assert not any(isinstance(e, Back) for e in effects)


def test_settle_still_confirms_once_the_map_returns_after_the_clear_bound_is_spent():
    """The bound only stops the BLIND BACK presses; it must never itself block a switch
    from confirming once the map actually comes back - switch_timeout, not this count,
    owns the outcome of a switch that never confirms."""
    bound = Config().timings.switch_clear_max
    c = ctx(phase="settle", accounts=panel(active="TrainerTwo"),
           switch_clear_presses=bound)
    effects = fsm.step(obs(on_map=True), c)
    assert not [e for e in effects if isinstance(e, Transition)]
    assert any(isinstance(e, SetFlag) and e.value == "zoom" for e in effects)


def test_confirmation_needs_both_the_map_and_the_asterisk():
    on_map_only = ctx(phase="settle", accounts=panel(active="TrainerOne"))
    effects = fsm.step(obs(on_map=True), on_map_only)
    assert not [e for e in effects if isinstance(e, Transition)]
    assert not any(isinstance(e, SetFlag) and e.value == "zoom" for e in effects)
    both = ctx(phase="settle", accounts=panel(active="TrainerTwo"))
    # A match advances to "zoom", not straight to CONFIRMED - see test_switch_zoom.py for
    # why the transition is deferred to the end of that phase.
    effects = fsm.step(obs(on_map=True), both)
    assert not [e for e in effects if isinstance(e, Transition)]
    assert any(isinstance(e, SetFlag) and e.value == "zoom" for e in effects)


def test_settle_reopens_the_panel_once_the_map_is_back():
    """The live-run defect: PGSharp shuts its own panel as part of logging in, so a
    post-login read is rows=0 even though the login worked. The map alone must not
    confirm anything - the state must go looking for the asterisk, not stall."""
    closed = AccountView(rows=(), launcher_norm=(0.12, 0.05), accounts_tab_norm=(0.83, 0.18),
                         close_norm=(0.06, 0.10), available=True, panel_open=False)
    c = ctx(phase="settle", accounts=closed)
    effects = fsm.step(obs(on_map=True), c)
    assert taps(effects)[0].x == pytest.approx(0.12)
    assert any(isinstance(e, SetFlag) and e.value == "verify" for e in effects)
    assert not any(isinstance(e, Transition) for e in effects)


def test_verify_confirms_when_the_reopened_panel_shows_the_target_active():
    """A match closes the panel and hands off to "zoom" - not straight to CONFIRMED.
    See test_switch_zoom.py for what that phase does and why the transition lives there
    instead of here."""
    c = ctx(phase="verify", accounts=panel(active="TrainerTwo"))
    effects = fsm.step(obs(on_map=True), c)
    assert taps(effects)[0].x == pytest.approx(0.06)     # the panel's own close_norm
    assert not any(isinstance(e, Transition) for e in effects)
    assert any(isinstance(e, SetFlag) and e.name == "switch_phase" and e.value == "zoom"
               for e in effects)


def test_a_mismatch_does_not_end_the_switch():
    """The login is asynchronous: 'someone else is active' at one instant means only
    that it has not landed yet, not that it failed. Closing the panel and re-checking -
    not latching a terminal phase - is what makes a later confirm possible."""
    c = ctx(phase="verify", accounts=panel(active="TrainerOne"))
    first = fsm.step(obs(on_map=True), c)
    assert not any(isinstance(e, Transition) for e in first)
    assert not any(isinstance(e, SetFlag) and e.name == "switch_phase" for e in first)
    login_norm = c.accounts.by_name("TrainerTwo").login_norm
    assert not any((t.x, t.y) == login_norm for t in taps(first))
    c.accounts = panel(active="TrainerTwo")          # the login has landed since
    second = fsm.step(obs(on_map=True), c)
    assert not any(isinstance(e, Transition) for e in second)
    assert any(isinstance(e, SetFlag) and e.value == "zoom" for e in second)


def test_verify_waits_out_the_login_grace_period():
    """Regression guard for the exact race the live run hit: the outgoing account's map
    can reappear within a second or two of the login tap, long before the login itself
    has landed (~14s measured). obs.on_map alone must not trigger a verify."""
    c = ctx(phase="settle", switch_login_ts=100.0)
    c.now = 100.0 + Config().timings.switch_login_grace - 1     # inside the grace window
    assert fsm.step(obs(on_map=True), c) == []


def test_a_genuine_failure_still_ends_at_the_timeout():
    """Without a terminal 'failed' phase, a switch that never lands must still be bounded
    by the state timeout rather than re-checking forever."""
    c = ctx(phase="verify", accounts=panel(active="TrainerOne"), cfg=budget(17.0))
    c.now = c.state_since + 18.0
    tr = [e for e in fsm.step(obs(on_map=True), c) if isinstance(e, Transition)][0]
    assert tr.to is BotState.RECOVERING and tr.outcome is IntentOutcome.EXPIRED


def test_the_switch_budget_is_the_configured_one():
    """`Timings.switch_timeout` was dead configuration - the handler hardcoded 120s while
    the config said 120s, and every test computed its deadline FROM the config, so the
    disconnect was invisible. These deadlines come from a budget nothing else in the
    system uses, so a hardcoded number fails them whatever it is."""
    short = ctx(phase="verify", accounts=panel(active="TrainerOne"), cfg=budget(17.0))
    short.now = short.state_since + 16.0
    assert not [e for e in fsm.step(obs(on_map=True), short) if isinstance(e, Transition)]
    short.now = short.state_since + 18.0
    assert [e for e in fsm.step(obs(on_map=True), short) if isinstance(e, Transition)]

    long = ctx(phase="verify", accounts=panel(active="TrainerOne"), cfg=budget(900.0))
    long.now = long.state_since + 300.0
    assert not [e for e in fsm.step(obs(on_map=True), long) if isinstance(e, Transition)]


def test_the_declared_budget_matches_the_config_default():
    """The import-time contract checks `timeout_s`, so the class still has to declare one;
    a declared number that disagrees with the config is the same lie in a smaller font."""
    assert fsm.Switching.timeout_s == Config().timings.switch_timeout


def test_a_rocket_looking_screen_cannot_hijack_a_switch():
    """Willow's post-login dialogue classifies as Rocket @ 0.66."""
    c = ctx(phase="settle", accounts=panel(active="TrainerTwo"))
    assert fsm.desired_state(obs(on_map=False, screen="Rocket", conf=0.66), c) is None


def test_an_encounter_looking_screen_cannot_hijack_a_switch():
    c = ctx(phase="settle", accounts=panel(active="TrainerTwo"))
    assert fsm.desired_state(obs(on_map=False, screen="PokemonEncounter", conf=0.99), c) is None


def test_timeout_escalates_to_recovering():
    c = ctx(phase="settle", cfg=budget(17.0))
    c.now = c.state_since + 18.0
    tr = [e for e in fsm.step(obs(on_map=False), c) if isinstance(e, Transition)][0]
    assert tr.to is BotState.RECOVERING and tr.outcome is IntentOutcome.EXPIRED
