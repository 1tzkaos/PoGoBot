"""Per-account settings, so Team GO Rocket can be on for one account and off for another.

`--no-rockets` is a property of the invocation, not of the account it lands on: before
this, a run that rotated through accounts carried one answer across every switch. The
operator edits a JSON file instead, because a flag cannot say "different for that one".
"""
from __future__ import annotations

import json

import pytest

from pogobot import fsm, profiles
from pogobot.config import DEFAULT
from pogobot.effects import BotState
from tests.factories import det, obs
from tests.test_switch_runner import make_runner


def _write(tmp_path, data):
    p = tmp_path / "accounts.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ------------------------------------------------------------------ the file itself

def test_a_missing_file_means_the_run_keeps_its_own_settings(tmp_path):
    """The file is optional, and its absence must be the behaviour that predates it."""
    assert profiles.load_profiles(tmp_path / "nope.json") == {}
    assert profiles.settings_for({}, "TrainerOne") == {}


def test_a_broken_file_never_stops_the_bot(tmp_path):
    p = tmp_path / "accounts.json"
    p.write_text("{not json", encoding="utf-8")
    assert profiles.load_profiles(p) == {}


def test_an_account_overrides_the_default(tmp_path):
    p = _write(tmp_path, {"default": {"fight_rockets": True},
                          "TrainerTwo": {"fight_rockets": False}})
    loaded = profiles.load_profiles(p)
    assert profiles.settings_for(loaded, "TrainerOne") == {"fight_rockets": True}
    assert profiles.settings_for(loaded, "TrainerTwo") == {"fight_rockets": False}


def test_an_unidentified_session_still_gets_the_default(tmp_path):
    """`stats.account` is None when the overlay could not be read. That is a real state,
    not an error, and it must still land on the default rather than on nothing."""
    loaded = profiles.load_profiles(_write(tmp_path, {"default": {"fight_rockets": False}}))
    assert profiles.settings_for(loaded, None) == {"fight_rockets": False}


def test_a_typo_is_reported_rather_than_silently_ignored(tmp_path, caplog):
    """In a hand-edited file `fight_rocket` would otherwise read as "leave it at the
    default" and the operator would be told nothing."""
    p = _write(tmp_path, {"TrainerOne": {"fight_rocket": False}})
    with caplog.at_level("WARNING"):
        loaded = profiles.load_profiles(p)
    assert loaded == {"TrainerOne": {}}
    assert "unknown option" in caplog.text and "fight_rocket" in caplog.text


def test_a_non_boolean_is_reported(tmp_path, caplog):
    p = _write(tmp_path, {"TrainerOne": {"fight_rockets": "no"}})
    with caplog.at_level("WARNING"):
        loaded = profiles.load_profiles(p)
    assert loaded == {"TrainerOne": {}}
    assert "expected true or false" in caplog.text


def test_the_shipped_example_parses_and_uses_only_known_keys():
    """A sample the operator copies must not teach them a key that does nothing."""
    from pogobot.config import BASE_DIR
    loaded = profiles.load_profiles(BASE_DIR / "accounts.example.json")
    assert loaded, "the example should not be empty"
    for account, settings in loaded.items():
        assert set(settings) <= profiles.KNOWN_KEYS, account


# ------------------------------------------------------------------ what it changes

def test_the_runner_applies_the_profile_for_the_current_account():
    r = make_runner()
    r.account_profiles = {"TrainerTwo": {"fight_rockets": False}}
    r.stats.account = "TrainerTwo"
    r._apply_account_profile()
    assert r.ctx.cfg.fight_rockets is False
    assert r.cfg.fight_rockets is False, "the runner's own view must not go stale"


def test_switching_back_restores_the_first_account_settings():
    """Overrides apply to the run's own config, not to each other - A -> B -> A must give
    A exactly what it had the first time."""
    r = make_runner()
    r.account_profiles = {"TrainerTwo": {"fight_rockets": False}}
    for account, expected in (("TrainerOne", True), ("TrainerTwo", False),
                              ("TrainerOne", True)):
        r.stats.account = account
        r._apply_account_profile()
        assert r.ctx.cfg.fight_rockets is expected, account


def test_an_account_with_no_entry_keeps_the_runs_own_setting():
    r = make_runner(DEFAULT.scaled(fight_rockets=False))   # e.g. --no-rockets
    r.account_profiles = {"TrainerTwo": {"fight_rockets": True}}
    r.stats.account = "TrainerOne"
    r._apply_account_profile()
    assert r.ctx.cfg.fight_rockets is False, "the invocation's own answer must survive"


def test_the_fsm_actually_obeys_it():
    """The two places `fight_rockets` decides anything: whether a Rocket screen is entered,
    and whether an invaded stop is a target at all."""
    off = DEFAULT.scaled(fight_rockets=False)
    c = fsm.Context(cfg=off, state=BotState.SCANNING, state_since=0.0, now=10.0)
    rocket_screen = obs(screen="Rocket", conf=0.99)
    assert fsm.desired_state(rocket_screen, c) is not BotState.ROCKET
    c2 = fsm.Context(cfg=off, state=BotState.SCANNING, state_since=0.0, now=10.0)
    assert fsm.pick_target(obs(on_map=True,
                               detections=(det("pokestop_rocket", 0.9, 0.5, 0.63),)), c2) is None


def test_it_is_applied_every_tick_not_only_at_startup():
    """The account changes in three places - startup, a confirmed switch, and a failed one
    that hands back whatever the overlay last named. A hook missing from one of them would
    be a silently wrong setting, so the runner checks rather than being told."""
    import inspect
    src = inspect.getsource(type(make_runner()).run)
    assert "_apply_account_profile()" in src
