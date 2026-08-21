"""config.json carries the run's usual shape; a typed flag is how one run departs from it.

Keys are command-line options by `dest`, so there is no second vocabulary and no list to
keep in sync - whatever the parser accepts, the file accepts.
"""
from __future__ import annotations

import json

import pytest

from pogobot import userconfig
from pogobot.cli import build_parser, explicit_options


def _apply(raw, argv):
    parser = build_parser()
    ns = parser.parse_args(argv)
    applied = userconfig.apply_run_settings(raw, parser, ns, explicit_options(argv))
    return ns, applied


def test_the_file_sets_options_the_operator_did_not_type():
    ns, applied = _apply({"tui": True, "switch_every": 45}, [])
    assert ns.tui is True
    assert ns.switch_every == 45.0        # --switch-every is type=float
    assert sorted(applied) == ["switch_every=45.0", "tui=true"]


def test_a_typed_flag_beats_the_file():
    """The file is the usual shape; the flag is this run departing from it. An operator
    typing --switch-every should not have to remember what the file says."""
    ns, applied = _apply({"switch_every": 45}, ["--switch-every", "10"])
    assert ns.switch_every == 10
    assert applied == []


def test_a_default_does_not_count_as_typed():
    """The trap this exists for: argparse cannot tell `--max-size 1280` from the 1280
    default, and if a default counted as a choice it would silently beat every line in the
    file."""
    assert "max_size" not in explicit_options([])
    ns, _ = _apply({"max_size": 720}, [])
    assert ns.max_size == 720


def test_a_typed_flag_is_recognised_even_when_it_matches_the_default():
    """Typing a value that happens to equal what the parser would have chosen is still
    typing it, and must still beat the file."""
    parser = build_parser()
    default = parser.parse_args([]).device                   # "auto" - a real default
    assert default == "auto"
    assert "device" in explicit_options(["--device", default])
    # ...and the file must then leave it alone.
    ns, applied = _apply({"device": "cpu"}, ["--device", "auto"])
    assert ns.device == "auto" and applied == []


def test_an_unknown_setting_is_reported_by_name(caplog):
    """In a hand-edited file a typo looks exactly like a line that is working."""
    with caplog.at_level("WARNING"):
        ns, applied = _apply({"swich_every": 45}, [])
    assert applied == []
    assert "swich_every" in caplog.text and "unknown setting" in caplog.text


def test_a_flag_given_a_non_boolean_is_reported(caplog):
    with caplog.at_level("WARNING"):
        _, applied = _apply({"tui": "yes"}, [])
    assert applied == []
    assert "expects true or false" in caplog.text


def test_values_are_coerced_the_way_the_parser_would():
    """JSON has no Path and no argparse types; a string must still land as the parser's
    own type or the rest of main() sees something it never expects."""
    from pathlib import Path
    ns, _ = _apply({"trace": "logs/somewhere.jsonl", "switch_every": "45"}, [])
    assert isinstance(ns.trace, Path)
    assert ns.switch_every == 45.0


def test_a_bad_value_is_reported_and_skipped(caplog):
    with caplog.at_level("WARNING"):
        ns, applied = _apply({"switch_every": "soon"}, [])
    assert applied == []
    assert "could not take" in caplog.text


def test_the_accounts_block_is_not_treated_as_a_run_setting(caplog):
    """It is the other half of the same file, and would otherwise be reported as unknown."""
    with caplog.at_level("WARNING"):
        _, applied = _apply({"accounts": {"TrainerOne": {"fight_rockets": False}}}, [])
    assert applied == []
    assert "unknown setting" not in caplog.text


def test_a_missing_or_broken_file_is_not_fatal(tmp_path):
    assert userconfig.load(tmp_path / "nope.json") == {}
    p = tmp_path / "config.json"
    p.write_text("{oops", encoding="utf-8")
    assert userconfig.load(p) == {}
    p.write_text('["not", "an object"]', encoding="utf-8")
    assert userconfig.load(p) == {}


def test_the_shipped_example_is_accepted_whole(caplog):
    """A sample the operator copies must not teach them a key that does nothing."""
    from pogobot.config import BASE_DIR
    raw = userconfig.load(BASE_DIR / "config.example.json")
    assert raw, "the example should not be empty"
    with caplog.at_level("WARNING"):
        _, applied = _apply(raw, [])
        userconfig.load_profiles(raw, "config.example.json")
    assert applied, "the example sets no run options"
    assert "unknown" not in caplog.text.lower()


def test_the_cli_reads_the_file_before_it_builds_the_config():
    """config_from_args reads the namespace this rewrites, so after it would be too late."""
    import inspect
    from pogobot import cli
    src = inspect.getsource(cli.main)
    assert src.index("apply_run_settings") < src.index("config_from_args(a)")
