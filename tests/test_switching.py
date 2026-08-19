import pytest

from pogobot import fsm
from pogobot.accounts import AccountRow, AccountView
from pogobot.config import Config
from pogobot.effects import Back, BotState, IntentOutcome, SetFlag, Tap, Transition
from tests.factories import obs


def row(name, active=False, x=0.74, level=10):
    return AccountRow(name=name, active=active, level=level,
                      login_norm=(x, 0.23), delete_norm=(0.89, 0.23),
                      row_norm=(0.04, 0.20, 0.96, 0.26))


def panel(active="TrainerOne", open_=True):
    return AccountView(
        rows=(row("TrainerOne", active == "TrainerOne"),
              row("TrainerTwo", active == "TrainerTwo")),
        launcher_norm=(0.12, 0.05), accounts_tab_norm=(0.83, 0.18),
        close_norm=(0.06, 0.10), available=True, panel_open=open_)


def ctx(**kw):
    c = fsm.Context(cfg=Config(), state=BotState.SWITCHING, now=100.0, state_since=100.0)
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


def test_no_tap_ever_lands_on_a_delete_button():
    """The delete button sits ~24px from login. This is the test that matters."""
    for phase in ("open", "tab", "login", "settle", "verify", "failed"):
        for on_map in (True, False):
            c = ctx(phase=phase)
            for t in taps(fsm.step(obs(on_map=on_map), c)):
                for r in c.accounts.rows:
                    assert (t.x, t.y) != r.delete_norm
                    assert abs(t.x - r.delete_norm[0]) > 0.02 or abs(t.y - r.delete_norm[1]) > 0.02


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


def test_confirmation_needs_both_the_map_and_the_asterisk():
    on_map_only = ctx(phase="settle", accounts=panel(active="TrainerOne"))
    assert not [e for e in fsm.step(obs(on_map=True), on_map_only)
                if isinstance(e, Transition)]
    both = ctx(phase="settle", accounts=panel(active="TrainerTwo"))
    tr = [e for e in fsm.step(obs(on_map=True), both) if isinstance(e, Transition)][0]
    assert tr.to is BotState.SCANNING and tr.outcome is IntentOutcome.CONFIRMED


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
    c = ctx(phase="verify", accounts=panel(active="TrainerTwo"))
    effects = fsm.step(obs(on_map=True), c)
    assert taps(effects)[0].x == pytest.approx(0.06)     # the panel's own close_norm
    tr = [e for e in effects if isinstance(e, Transition)][0]
    assert tr.to is BotState.SCANNING and tr.outcome is IntentOutcome.CONFIRMED


def test_verify_does_not_confirm_or_retap_login_when_someone_else_is_active():
    c = ctx(phase="verify", accounts=panel(active="TrainerOne"))
    effects = fsm.step(obs(on_map=True), c)
    assert not any(isinstance(e, Transition) for e in effects)
    assert any(isinstance(e, SetFlag) and e.value == "failed" for e in effects)
    login_norm = c.accounts.by_name("TrainerTwo").login_norm
    assert not any((t.x, t.y) == login_norm for t in taps(effects))


def test_a_rocket_looking_screen_cannot_hijack_a_switch():
    """Willow's post-login dialogue classifies as Rocket @ 0.66."""
    c = ctx(phase="settle", accounts=panel(active="TrainerTwo"))
    assert fsm.desired_state(obs(on_map=False, screen="Rocket", conf=0.66), c) is None


def test_an_encounter_looking_screen_cannot_hijack_a_switch():
    c = ctx(phase="settle", accounts=panel(active="TrainerTwo"))
    assert fsm.desired_state(obs(on_map=False, screen="PokemonEncounter", conf=0.99), c) is None


def test_timeout_escalates_to_recovering():
    c = ctx(phase="settle")
    c.now = c.state_since + Config().timings.switch_timeout + 1
    tr = [e for e in fsm.step(obs(on_map=False), c) if isinstance(e, Transition)][0]
    assert tr.to is BotState.RECOVERING and tr.outcome is IntentOutcome.EXPIRED
