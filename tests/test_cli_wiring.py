"""The CLI's calls must match the signatures it calls into.

`python3 -m pogobot --tui` crashed with
`Dashboard.__init__() got an unexpected keyword argument 'pause_file'` while the whole
suite was green: every test built these objects directly, so nothing checked that
`cli.main` passes arguments they actually accept. This binds the real call sites in
cli.py against the real signatures, without loading a model or touching a device.
"""
import ast
import inspect
import logging
import pathlib

import pytest

from pogobot import runner as runner_mod
from pogobot import tui
from pogobot.accounts import AccountView, FakeTreeReader, UiTreeReader
from pogobot.cli import build_parser, prepare_accounts
from pogobot.config import DEFAULT
from pogobot.effects import Tap
from pogobot.stats import SessionStats
from tests.test_switching import panel

CLI = pathlib.Path(runner_mod.__file__).parent / "cli.py"

TARGETS = {
    "Dashboard": tui.Dashboard.__init__,
    "Runner": runner_mod.Runner.__init__,
    "SessionStats": SessionStats.__init__,
    "UiTreeReader": UiTreeReader.__init__,
}


def _calls(name: str):
    """Every call to `name` in cli.py, as (keywords, has_star_kwargs)."""
    tree = ast.parse(CLI.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        label = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if label != name:
            continue
        found.append(([k.arg for k in node.keywords if k.arg], len(node.args)))
    return found


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_cli_call_sites_bind_to_the_real_signature(name):
    sig = inspect.signature(TARGETS[name])
    calls = _calls(name)
    assert calls, f"expected cli.py to construct {name}"
    for keywords, n_positional in calls:
        args = ["self"] + ["<pos>"] * n_positional
        try:
            sig.bind(*args, **{k: None for k in keywords})
        except TypeError as exc:
            pytest.fail(f"cli.py calls {name}(...) with arguments it does not accept: {exc}")


def test_cli_hands_the_runner_what_account_switching_needs():
    """Both are startup-only. The roster can never be re-read during a run - the PGSharp
    panel is shut, so a live read lists no accounts - and without the reader a switch
    cannot be driven or confirmed. Dropping either from the call leaves switching inert
    with every other test still green, which is how it shipped inert once already."""
    keywords, _ = _calls("Runner")[0]
    assert "tree_reader" in keywords, "the runner cannot drive a switch without it"
    assert "roster" in keywords, "the runner cannot name a target without it"


def test_every_parser_flag_is_reachable():
    """A flag that main() never reads is a lie in --help."""
    src = CLI.read_text()
    parser = build_parser()
    unread = []
    for action in parser._actions:
        if action.dest in ("help",):
            continue
        # Read either directly (`a.foo`) or by name through the getattr loop that
        # copies overrides onto the Config (`"foo"`).
        if f"a.{action.dest}" not in src and f'"{action.dest}"' not in src:
            unread.append(action.dest)
    assert not unread, f"flags declared but never read by main(): {unread}"


def test_switch_flags_are_wired_through_to_config():
    from pogobot.cli import build_parser, config_from_args
    a = build_parser().parse_args(["--switch-on-quota", "--switch-every", "45"])
    cfg = config_from_args(a)
    assert cfg.switch_on_quota is True
    assert cfg.switch_every_minutes == 45


def test_switching_is_off_unless_asked_for():
    from pogobot.cli import build_parser, config_from_args
    cfg = config_from_args(build_parser().parse_args([]))
    assert cfg.switch_on_quota is False and cfg.switch_every_minutes == 0


def test_account_and_collect_dialogues_flags_parse():
    a = build_parser().parse_args(["--account", "TrainerOne",
                                   "--collect-dialogues", "/tmp/dlg"])
    assert a.account == "TrainerOne"
    assert a.collect_dialogues == pathlib.Path("/tmp/dlg")


def test_account_and_collect_dialogues_default_to_none():
    a = build_parser().parse_args([])
    assert a.account is None
    assert a.collect_dialogues is None


def test_reset_spins_defaults_to_untouched():
    a = build_parser().parse_args([])
    assert a.reset_spins is None


def test_reset_spins_alone_still_means_every_account():
    """`--reset-spins` with no value must keep clearing every account's window, exactly
    as it did before it could also target one - a previously valid invocation must not
    silently start doing something narrower."""
    bare = build_parser().parse_args(["--reset-spins"]).reset_spins
    named = build_parser().parse_args(["--reset-spins", "TrainerOne"]).reset_spins
    assert bare is not None
    assert bare != "TrainerOne"
    assert named == "TrainerOne"


def test_seed_spins_flag_still_parses():
    a = build_parser().parse_args(["--seed-spins", "300"])
    assert a.seed_spins == 300


# ------------------------------------------------------------------ startup accounts

class _Act:
    """Records what would have gone to the phone. Local, like every other runner-shaped
    fake in this suite."""

    def __init__(self):
        self.applied = []
        self.dry_run = False

    def apply(self, effect, now=None):
        self.applied.append(effect)
        return True


def _closed():
    return AccountView(available=True, launcher_norm=(0.12, 0.05), panel_open=False)


def _factory(views):
    """A reader factory that records whether it was ever asked for a reader."""
    made = []

    def make():
        made.append(FakeTreeReader(views))
        return made[-1]

    return make, made


def _prepare(cfg=DEFAULT, requested=None, pause_file=None, views=None, act=None):
    make, made = _factory(views if views is not None else [_closed(), panel(active="TrainerTwo")])
    act = act if act is not None else _Act()
    result = prepare_accounts(cfg, requested=requested, pause_file=pause_file,
                              make_reader=make, actuator=act, settle=0)
    return result, made, act


SWITCHING = DEFAULT.scaled(switch_on_quota=True)


def test_identification_is_skipped_when_no_switch_trigger_is_armed():
    """It costs three taps INTO the panel whose rows carry irreversible delete buttons,
    and a run that will never switch has no use for the roster it produces. Without
    switching, --account is how a run gets attributed and an unnamed one stays in the
    unattributed bucket - exactly the behaviour that predates this feature."""
    (reader, account, roster), made, act = _prepare(requested="TrainerOne")
    assert made == [], "the panel must not even be read"
    assert act.applied == []
    assert (reader, account, roster) == (None, "TrainerOne", ())


def test_an_unswitched_run_without_an_account_stays_unattributed():
    (reader, account, roster), made, act = _prepare()
    assert (reader, account, roster) == (None, None, ())
    assert act.applied == []


def test_identification_is_skipped_while_the_pause_file_exists(tmp_path):
    """Those taps go through the actuator directly, not through `Runner.apply`, so
    nothing else honours the pause. The README's promise has no exceptions: while the file
    exists the bot perceives but sends no input."""
    pause = tmp_path / "PAUSE"
    pause.touch()
    (reader, account, roster), made, act = _prepare(SWITCHING, requested="TrainerOne",
                                                    pause_file=pause)
    assert made == [] and act.applied == []
    assert (reader, account, roster) == (None, "TrainerOne", ())


def test_identification_runs_once_a_trigger_is_armed(tmp_path):
    pause = tmp_path / "PAUSE"          # named but absent: nothing to honour
    (reader, account, roster), made, act = _prepare(SWITCHING, pause_file=pause)
    assert reader is made[0]
    assert account == "TrainerTwo"
    assert roster == ("TrainerOne", "TrainerTwo")

    opened = panel(active="TrainerTwo")
    deletes = {r.delete_norm for r in opened.rows if r.delete_norm}
    tapped = {(t.x, t.y) for t in act.applied if isinstance(t, Tap)}
    assert tapped and not (tapped & deletes), "a startup read must never delete an account"


def test_the_overlay_outranks_a_contradicting_account_flag(caplog):
    """--account is a claim made before the process started; the overlay can see who is
    actually logged in. Believing the flag books every spin to the wrong account,
    under-counts the real one's window - so the bot spins past a cap it cannot see - and
    starts the round-robin from the wrong origin."""
    with caplog.at_level(logging.WARNING, logger="pogobot"):
        (_, account, _), _, _ = _prepare(SWITCHING, requested="TrainerOne")
    assert account == "TrainerTwo"
    warned = [m for m in caplog.messages if "contradicts" in m]
    assert len(warned) == 1
    assert "TrainerOne" in warned[0] and "TrainerTwo" in warned[0]


def test_an_agreeing_account_flag_is_not_a_contradiction(caplog):
    with caplog.at_level(logging.WARNING, logger="pogobot"):
        (_, account, _), _, _ = _prepare(SWITCHING, requested="TrainerTwo")
    assert account == "TrainerTwo"
    assert not [m for m in caplog.messages if "contradicts" in m]


def test_the_account_flag_still_names_a_run_the_overlay_could_not_read():
    """The overlay is preferred where it answers - it is not a requirement."""
    (_, account, roster), _, act = _prepare(SWITCHING, requested="TrainerOne",
                                            views=[AccountView(available=False)])
    assert account == "TrainerOne" and roster == ()
    assert act.applied == [], "an unreadable overlay is never tapped at a guess"
