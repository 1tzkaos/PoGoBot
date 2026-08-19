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
from pogobot.cli import build_parser

CLI = pathlib.Path(runner_mod.__file__).parent / "cli.py"

TARGETS = {
    "Dashboard": tui.Dashboard.__init__,
    "Runner": runner_mod.Runner.__init__,
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
