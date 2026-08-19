"""The CLI's calls must match the signatures it calls into.

`python3 -m pogobot --tui` crashed with
`Dashboard.__init__() got an unexpected keyword argument 'pause_file'` while the whole
suite was green: every test built these objects directly, so nothing checked that
`cli.main` passes arguments they actually accept. This binds the real call sites in
cli.py against the real signatures, without loading a model or touching a device.
"""
import ast
import inspect
import pathlib

import pytest

from pogobot import runner as runner_mod
from pogobot import tui
from pogobot.accounts import UiTreeReader
from pogobot.cli import build_parser
from pogobot.stats import SessionStats

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
