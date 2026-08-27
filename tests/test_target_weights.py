"""Weights decide WHICH eligible target is tapped, never WHETHER one may be.

The bug this replaces: `pick_target` ranked `(1 if pokemon else 0, conf)`, so any Pokemon
beat any stop. Not "usually" - always. A 0.31 Pokemon at the edge of reach outranked a 0.99
stop, and on a map with both, stops were tapped exactly never. The only ways out turned
Pokemon off entirely (`--target-mode pokestop`, restocking); there was no way to say
"mostly Pokemon, some stops".

So the properties worth pinning are a pair, and the second is the one a scheduler is likely
to break: the weighted share must hold, AND every hard filter above it must still be hard.
A weight is not permission - an out-of-reach stop, a cooled location, a spent quota and a
class the operator turned off stay untappable at any weight.
"""
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, "tests")

from factories import det, obs

from pogobot import fsm, runner as runner_mod, userconfig
from pogobot.config import DEFAULT, TargetWeights


def _ctx(weights=None, window=(), **cfgkw):
    cfg = DEFAULT
    if weights is not None:
        cfgkw["target_weights"] = weights
    if cfgkw:
        cfg = replace(cfg, **cfgkw)
    c = fsm.Context(cfg=cfg, now=1000.0)
    c.last_map_ts = 1000.0
    c.recent_targets = list(window)
    return c


def _dets(*names, conf=0.9):
    return [det(name=n, conf=conf, cx=0.5, cy=0.63) for n in names]


def _run(ctx, names, n=40):
    """Pick repeatedly, feeding the window the way the runner does on SetIntent."""
    out = []
    for _ in range(n):
        t = fsm.pick_target(obs(on_map=True, detections=_dets(*names), ts=1000.0), ctx)
        if t is None:
            break
        out.append(t.name)
        ctx.recent_targets.append(t.name)
        keep = ctx.cfg.target_share_window
        if len(ctx.recent_targets) > keep:
            del ctx.recent_targets[:-keep]
    return out


def _share(seq, name):
    return seq.count(name) / len(seq)


# ---------------------------------------------------------------- the weighted share

def test_a_map_with_both_settles_on_the_configured_ratio():
    """The whole point: 1.0 against 0.6 is five Pokemon for every three stops."""
    seq = _run(_ctx(), ("pokemon", "pokestop"), n=40)
    assert _share(seq, "pokemon") == pytest.approx(0.625, abs=0.03)
    assert _share(seq, "pokestop") == pytest.approx(0.375, abs=0.03)


def test_stops_are_actually_tapped_on_a_map_full_of_confident_pokemon():
    """The regression this feature exists for. Under the old tiering this was zero."""
    seq = _run(_ctx(), ("pokemon", "pokestop"), n=20)
    assert seq.count("pokestop") > 0


def test_pokemon_go_first():
    """'Weighted above' has to mean the heavier class opens, not merely that it gets more."""
    assert _run(_ctx(), ("pokemon", "pokestop"), n=1) == ["pokemon"]


def test_neither_class_is_ever_starved_by_a_long_run_of_the_other():
    """A correct ratio reached by 25 Pokemon then 15 stops would satisfy the counts and
    still be the starvation this replaces. The largest-remainder rule interleaves."""
    seq = _run(_ctx(), ("pokemon", "pokestop"), n=40)
    longest = 1
    current = 1
    for a, b in zip(seq, seq[1:]):
        current = current + 1 if a == b else 1
        longest = max(longest, current)
    assert longest <= 2, f"ran {longest} of one class in a row: {seq}"


def test_a_map_with_only_stops_gives_every_tap_to_stops():
    """"No Pokemon and a ton of stops" needs no special case: a class that is not on
    screen is not in the denominator."""
    seq = _run(_ctx(), ("pokestop",), n=10)
    assert seq == ["pokestop"] * 10


def test_a_map_with_only_pokemon_gives_every_tap_to_pokemon():
    seq = _run(_ctx(), ("pokemon",), n=10)
    assert seq == ["pokemon"] * 10


def test_the_ratio_follows_the_weights_when_they_are_inverted():
    """A stop-farming account is the same mechanism pointed the other way."""
    w = TargetWeights(pokemon=0.2, pokestop=1.0)
    seq = _run(_ctx(weights=w), ("pokemon", "pokestop"), n=60)
    assert _share(seq, "pokestop") == pytest.approx(1.0 / 1.2, abs=0.05)


def test_only_the_proportions_matter_not_the_magnitudes():
    a = _run(_ctx(weights=TargetWeights(pokemon=1.0, pokestop=0.6)),
             ("pokemon", "pokestop"), n=30)
    b = _run(_ctx(weights=TargetWeights(pokemon=10.0, pokestop=6.0)),
             ("pokemon", "pokestop"), n=30)
    assert a == b


def test_confidence_still_decides_between_two_of_the_same_class():
    """What confidence was always doing usefully. What it no longer does is outrank a
    whole class."""
    ctx = _ctx()
    d_low = det(name="pokemon", conf=0.40, cx=0.50, cy=0.63)
    d_high = det(name="pokemon", conf=0.95, cx=0.52, cy=0.63)
    got = fsm.pick_target(obs(on_map=True, detections=[d_low, d_high], ts=1000.0), ctx)
    assert got is d_high


def test_a_low_confidence_pokemon_no_longer_outranks_a_confident_stop_forever():
    """The exact old failure, stated as a ratio rather than a single pick."""
    ctx = _ctx()
    dets = [det(name="pokemon", conf=0.35, cx=0.50, cy=0.63),
            det(name="pokestop", conf=0.99, cx=0.52, cy=0.63)]
    seq = []
    for _ in range(16):
        t = fsm.pick_target(obs(on_map=True, detections=dets, ts=1000.0), ctx)
        seq.append(t.name)
        ctx.recent_targets.append(t.name)
    assert seq.count("pokestop") >= 5


# ---------------------------------------------------------------- weight 0 disables

def test_a_weight_of_zero_disables_the_class_entirely():
    """Not "ranked last": ranked last still gets tapped whenever nothing else is up."""
    w = TargetWeights(pokemon=1.0, pokestop=0.0, pokestop_rocket=0.0)
    assert _run(_ctx(weights=w), ("pokestop",), n=5) == []
    assert _run(_ctx(weights=w), ("pokemon", "pokestop"), n=6) == ["pokemon"] * 6


def test_a_class_the_weights_do_not_name_is_never_tapped():
    """The model gained a `gym` class once already. A new class must not enter the
    rotation at full weight because nobody thought to name it."""
    assert DEFAULT.target_weights.of("gym") == 0
    assert _run(_ctx(), ("gym",), n=3) == []


# ---------------------------------------------------------------- filters stay hard

def test_reach_still_refuses_a_stop_no_matter_how_heavily_it_is_weighted():
    w = TargetWeights(pokemon=0.01, pokestop=1000.0)
    ctx = _ctx(weights=w)
    far = det(name="pokestop", conf=0.99, cx=0.02, cy=0.02)
    assert fsm.pick_target(obs(on_map=True, detections=[far], ts=1000.0), ctx) is None


def test_confidence_floor_still_applies_at_any_weight():
    w = TargetWeights(pokestop=1000.0)
    ctx = _ctx(weights=w)
    weak = det(name="pokestop", conf=DEFAULT.target_confidence - 0.01, cx=0.5, cy=0.63)
    assert fsm.pick_target(obs(on_map=True, detections=[weak], ts=1000.0), ctx) is None


def test_a_spent_quota_still_refuses_stops_at_any_weight():
    w = TargetWeights(pokemon=0.0, pokestop=1000.0)
    ctx = _ctx(weights=w)
    ctx.spins_exhausted = True
    assert _run(ctx, ("pokestop",), n=3) == []


def test_restocking_still_ignores_pokemon_at_any_weight():
    w = TargetWeights(pokemon=1000.0, pokestop=0.001)
    ctx = _ctx(weights=w)
    ctx.restocking_until = ctx.now + 60
    seq = _run(ctx, ("pokemon", "pokestop"), n=5)
    assert set(seq) == {"pokestop"}


def test_target_mode_still_wins_over_the_weights():
    ctx = _ctx(weights=TargetWeights(pokestop=1000.0), target_mode="pokemon")
    seq = _run(ctx, ("pokemon", "pokestop"), n=5)
    assert set(seq) == {"pokemon"}


def test_rockets_turned_off_are_still_off_at_any_weight():
    ctx = _ctx(weights=TargetWeights(pokestop_rocket=1000.0), fight_rockets=False)
    assert _run(ctx, ("pokestop_rocket",), n=3) == []


def test_a_cooled_location_is_still_refused():
    ctx = _ctx()
    ctx.cooldowns.append((0.5, 0.63, ctx.now + 60))
    assert fsm.pick_target(obs(on_map=True, detections=_dets("pokemon"), ts=1000.0),
                           ctx) is None


# ---------------------------------------------------------------- purity

def test_picking_twice_on_one_frame_does_not_advance_the_schedule():
    """`Context` says handlers read it and only the runner writes it. A scheduler that
    charged itself at pick time would also charge for a tap the runner never issued."""
    ctx = _ctx()
    o = obs(on_map=True, detections=_dets("pokemon", "pokestop"), ts=1000.0)
    first = fsm.pick_target(o, ctx)
    second = fsm.pick_target(o, ctx)
    assert first.name == second.name
    assert ctx.recent_targets == []


def test_the_window_is_read_but_never_written_by_the_fsm():
    ctx = _ctx(window=["pokemon", "pokemon"])
    before = list(ctx.recent_targets)
    fsm.pick_target(obs(on_map=True, detections=_dets("pokemon", "pokestop"), ts=1000.0),
                    ctx)
    assert ctx.recent_targets == before


def test_a_window_full_of_pokemon_hands_the_next_tap_to_the_stop():
    ctx = _ctx(window=["pokemon"] * 8)
    got = fsm.pick_target(obs(on_map=True, detections=_dets("pokemon", "pokestop"),
                              ts=1000.0), ctx)
    assert got.name == "pokestop"


# ---------------------------------------------------------------- the runner's half

class _Act:
    def __init__(self, accept=True):
        self.accept = accept

    def apply(self, effect, now=None):
        return self.accept

    def healthy(self):
        return True

    def stats(self):
        return {}

    def close(self):
        pass


class _Src:
    def read(self):
        return None

    def healthy(self):
        return True

    def release(self):
        pass


def _runner(**kw):
    return runner_mod.Runner(DEFAULT, _Src(), _Act(), perceptor=None, display=False, **kw)


def _intent(name):
    return fsm.Intent(ts=0.0, target_name=name, confidence=0.9, tap_norm=(0.5, 0.6),
                      xywhn=(0.5, 0.6, 0.1, 0.1), expected=fsm.BotState.ENCOUNTER,
                      frame_seq=1)


def test_the_runner_records_the_class_it_chose():
    r = _runner()
    r._record_target(_intent("pokemon"))
    r._record_target(_intent("pokestop"))
    assert r.ctx.recent_targets == ["pokemon", "pokestop"]


def test_the_window_cannot_grow_without_bound():
    """At ~1 target tap a second an unbounded list holds 20k strings by hour six, all but
    the last `target_share_window` of which are never read."""
    r = _runner()
    for _ in range(500):
        r._record_target(_intent("pokemon"))
    assert len(r.ctx.recent_targets) == DEFAULT.target_share_window


def test_a_non_target_intent_is_not_recorded():
    r = _runner()
    r._record_target(_intent("close_button"))
    r._record_target(_intent(None))
    assert r.ctx.recent_targets == []


def test_the_choice_is_recorded_even_when_the_actuator_refuses_the_gesture():
    """A refused gesture was still this class's turn. Recording only accepted taps would
    hand the same class the next turn too, for as long as the actuator kept refusing."""
    r = runner_mod.Runner(DEFAULT, _Src(), _Act(accept=False), perceptor=None,
                          display=False)
    r.apply([fsm.SetIntent(_intent("pokemon"))],
            obs(on_map=True, ts=1000.0))
    assert r.ctx.recent_targets == ["pokemon"]


def test_set_intent_is_the_path_that_records():
    r = _runner()
    r.apply([fsm.SetIntent(_intent("pokestop"))],
            obs(on_map=True, ts=1000.0))
    assert r.ctx.recent_targets == ["pokestop"]


# ---------------------------------------------------------------- per-account config

def _profiles(block):
    return userconfig.load_profiles({"accounts": block})


def test_an_account_can_set_its_own_weights():
    prof = _profiles({"Catcher": {"target_weights": {"pokemon": 1.0, "pokestop": 0.2}}})
    w = prof["Catcher"]["target_weights"]
    assert (w.pokemon, w.pokestop) == (1.0, 0.2)


def test_a_partial_block_is_merged_onto_the_defaults():
    """`{"pokestop": 0.2}` must mean "the usual, but fewer stops", not "Pokemon to zero"."""
    w = _profiles({"A": {"target_weights": {"pokestop": 0.2}}})["A"]["target_weights"]
    assert w.pokemon == DEFAULT.target_weights.pokemon
    assert w.pokestop_rocket == DEFAULT.target_weights.pokestop_rocket
    assert w.pokestop == 0.2


def test_a_misspelled_target_is_reported_by_name(caplog):
    with caplog.at_level("WARNING"):
        prof = _profiles({"A": {"target_weights": {"pokeman": 1.0}}})
    assert prof["A"] == {}
    assert "pokeman" in caplog.text


def test_a_non_object_weights_block_is_refused_with_usable_advice(caplog):
    with caplog.at_level("WARNING"):
        prof = _profiles({"A": {"target_weights": "lots"}})
    assert prof["A"] == {}
    assert "expected an object" in caplog.text


def test_a_negative_weight_is_refused(caplog):
    with caplog.at_level("WARNING"):
        prof = _profiles({"A": {"target_weights": {"pokemon": -1}}})
    assert prof["A"] == {}
    assert "cannot be negative" in caplog.text


def test_true_is_not_accepted_as_a_weight(caplog):
    """`bool` is an `int`, so `{"pokemon": true}` would otherwise weigh 1.0 and look like
    it worked while meaning something nobody wrote."""
    with caplog.at_level("WARNING"):
        prof = _profiles({"A": {"target_weights": {"pokemon": True}}})
    assert prof["A"] == {}
    assert "expected a number" in caplog.text


def test_fight_rockets_still_validates_as_a_boolean(caplog):
    """The per-key dispatch must not have loosened the setting that was already there."""
    with caplog.at_level("WARNING"):
        prof = _profiles({"A": {"fight_rockets": "yes"}})
    assert prof["A"] == {}
    assert "true or false" in caplog.text


def test_a_weights_block_reaches_the_running_config():
    prof = _profiles({"Catcher": {"target_weights": {"pokestop": 0.1}}})
    r = _runner(account_profiles=prof)
    r.stats.account = "Catcher"
    r._apply_account_profile()
    assert r.cfg.target_weights.pokestop == 0.1
    assert r.ctx.cfg.target_weights.pokestop == 0.1, "the FSM reads ctx.cfg, not cfg"
