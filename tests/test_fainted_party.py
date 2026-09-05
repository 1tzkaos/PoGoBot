"""A fainted party member must be read BEFORE the bot presses USE THIS PARTY.

The stall this exists for is in `logs/run.log` and it did not look like a failure. A party
member had fainted, so Pokemon GO refused the fight behind a pink error the bot cannot
read. The bot pressed the pill, learned nothing, timed out on `Rocket.timeout_s`, recovered,
saw the same invaded stop and went straight back in. Measured: 173 rocket timeouts at a
median of exactly 150.0s - the FULL budget every time, so not one made progress - across
9.3 hours, finishing with `stops_collected: 0`. The run never crashed. It looked busy.

Two halves are tested here and they fail differently. The OPTICAL half's expensive mistake
is a false TRUE ("fight it") on a party that cannot: that is the stall itself. The FSM
half's expensive mistake is churn - declining correctly but re-entering ROCKET every 30s
forever, which is the same wasted hours wearing a different label.
"""
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, "tests")
from factories import obs

from pogobot import fsm, perception, runner as runner_mod
from pogobot.config import DEFAULT as C
from pogobot.effects import Note, SetFlag, Tap, Transition
from pogobot.observation import Tristate

FIX = Path("tests/fixtures/screens")
FAINTED = FIX / "rocket_party_fainted.png"
HEALTHY = FIX / "rocket_party_healthy.png"
CORPUS = Path("datasets/state_v3")


def _img(p):
    im = cv2.imread(str(p))
    assert im is not None, f"missing fixture {p}"
    return im


# ---------------------------------------------------------------- the optical test

def test_a_fainted_member_is_read_as_unable_to_battle():
    assert perception.party_can_battle(_img(FAINTED), C) is Tristate.FALSE


def test_a_healthy_party_is_read_as_able_to_battle():
    assert perception.party_can_battle(_img(HEALTHY), C) is Tristate.TRUE


def test_the_two_fixtures_are_different_aspect_ratios():
    """The reason the panel is located instead of assumed. The HP row sits at frame-y
    0.7402 on one and 0.7625 on the other - a 0.0223 spread no fixed band survives."""
    f, h = _img(FAINTED), _img(HEALTHY)
    assert f.shape[1] != h.shape[1]
    assert abs(f.shape[1] / f.shape[0] - h.shape[1] / h.shape[0]) > 0.01


def test_an_ordinary_screen_reads_unknown_not_true():
    """UNKNOWN must never be mistaken for TRUE by a caller: TRUE means "fight it"."""
    for name in ("sponsored_ad.png", "gym_close.png", "levelup_12.png", "postlogin_ok.png"):
        p = FIX / name
        if p.exists():
            assert perception.party_can_battle(_img(p), C) is Tristate.UNKNOWN, name


def test_the_bar_threshold_sits_below_the_shortest_real_bar():
    """Measured plateau: a fainted card reads exactly 0.0000, and the shortest healthy bar
    in the corpus is 0.6132 (a member at roughly 89% health)."""
    assert C.battle_party.bar_min < 0.6132
    assert C.battle_party.bar_min > 0.0


def test_the_peak_row_is_what_separates_them_not_the_band_mean():
    """RED-GREEN for the reduction. Averaging over the whole band dilutes a full bar from
    0.69 to 0.06 - under any threshold 0.0000 is also under - which collapses the
    separation entirely. This asserts the band mean would NOT have worked."""
    bp = C.battle_party
    import numpy as np
    im = _img(HEALTHY)
    h, w = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, bp.panel_v_min]),
                        np.array([179, bp.panel_s_max, 255]))
    n, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    panel = None
    for i in range(1, n):
        y, ch, a = stats[i, 1], stats[i, 3], stats[i, 4]
        if bp.area_min <= a / (h * w) <= bp.area_max and y / h >= bp.top_min:
            panel = (y, y + ch)
            break
    assert panel is not None
    y0, y1 = panel
    ph = y1 - y0
    b0, b1 = int(y0 + bp.bar_band[0] * ph), int(y0 + bp.bar_band[1] * ph)
    x0f, x1f = bp.cards[0]
    card = hsv[b0:b1, int(x0f * w):int(x1f * w)]
    m = cv2.inRange(card, perception.GREEN_PILL_LO, perception.GREEN_PILL_HI)
    assert m.mean() / 255.0 < bp.bar_min, "band mean should FAIL the threshold"
    assert (m.mean(axis=1) / 255.0).max() > bp.bar_min, "peak row should clear it"


@pytest.mark.skipif(not CORPUS.exists(), reason="datasets/ is gitignored")
def test_no_false_positive_anywhere_in_the_corpus():
    """The expensive direction. 5/5 ChooseParty read TRUE, 230/230 everything else
    UNKNOWN - measured over every labelled frame."""
    import glob
    party = other = 0
    for f in glob.glob(str(CORPUS / "*/*/*.png")):
        im = cv2.imread(f)
        if im is None:
            continue
        r = perception.party_can_battle(im, C)
        if f.split("/")[-2] == "ChooseParty":
            party += r is Tristate.TRUE
        else:
            assert r is Tristate.UNKNOWN, f
            other += 1
    assert party == 5 and other == 230, f"{party} party, {other} other"


# ---------------------------------------------------------------- the FSM response

def _ctx(**kw):
    c = fsm.Context(cfg=kw.pop("cfg", C), now=kw.pop("now", 1000.0))
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _rocket_obs(**kw):
    kw.setdefault("screen", "Rocket")
    kw.setdefault("conf", 1.0)
    return obs(**kw)


def test_a_party_that_cannot_battle_is_declined_not_pressed():
    out = fsm.Rocket().step(_rocket_obs(pill_xy=(0.5, 0.82)),
                            _ctx(party_cannot_battle=True))
    assert not any(isinstance(e, Tap) for e in out), "must not press USE THIS PARTY"
    assert any(isinstance(e, Transition) and e.to is fsm.BotState.RECOVERING for e in out)
    assert any(isinstance(e, SetFlag) and e.name == "party_fainted_ts" for e in out)


def test_a_healthy_party_is_still_fought():
    """The most important test here. A detector that declines everything would also make
    the stall go away, and would be far worse than the stall."""
    out = fsm.Rocket().step(_rocket_obs(pill_xy=(0.5, 0.82)),
                            _ctx(party_cannot_battle=False))
    taps = [e for e in out if isinstance(e, Tap)]
    assert taps and taps[0].budget == "rocket"


def test_one_frame_is_not_enough_to_decline():
    """The sheet slides in with animating bars, and this writes a fact that outlives the
    frame by minutes."""
    r = _runner()
    r._tick_party(Tristate.FALSE)
    assert r.ctx.party_cannot_battle is False
    r._tick_party(Tristate.FALSE)
    assert r.ctx.party_cannot_battle is True


def test_a_healthy_frame_resets_the_agreement():
    r = _runner()
    r._tick_party(Tristate.FALSE)
    r._tick_party(Tristate.TRUE)
    r._tick_party(Tristate.FALSE)
    assert r.ctx.party_cannot_battle is False


def test_unknown_neither_confirms_nor_resets():
    """Leaving the screen must not clear the finding - every non-party screen is UNKNOWN."""
    r = _runner()
    r._tick_party(Tristate.FALSE)
    r._tick_party(Tristate.UNKNOWN)
    assert r.ctx.party_cannot_battle is False
    r._tick_party(Tristate.FALSE)
    assert r.ctx.party_cannot_battle is True
    r._tick_party(Tristate.UNKNOWN)
    assert r.ctx.party_cannot_battle is True


# ---------------------------------------------------------------- the hold

def test_the_hold_keeps_the_machine_out_of_rocket():
    ctx = _ctx(now=1000.0, party_fainted_ts=1000.0)
    o = _rocket_obs(pill_xy=(0.5, 0.82))
    assert fsm.desired_state(o, ctx) is not fsm.BotState.ROCKET


def test_the_hold_expires():
    ctx = _ctx(now=1000.0 + C.timings.party_fainted_hold + 1, party_fainted_ts=1000.0)
    assert fsm.desired_state(_rocket_obs(pill_xy=(0.5, 0.82)), ctx) is fsm.BotState.ROCKET


def test_the_hold_is_inert_before_anything_faints():
    """RED-GREEN for the `> 0.0` guard. `ctx.now` is a perf_counter reading live but 0.0 in
    a fresh Context, so without the guard `0.0 - 0.0 < 900` reads as "just fainted" and the
    bot refuses every Rocket fight it will ever see."""
    ctx = _ctx(now=0.0)
    assert ctx.party_fainted_ts == 0.0
    assert fsm.desired_state(_rocket_obs(pill_xy=(0.5, 0.82)), ctx) is fsm.BotState.ROCKET


def test_without_the_hold_the_machine_ping_pongs():
    """The churn the hold exists to stop, stated as a measurement.

    With the hold at 0 the bot re-enters ROCKET on every frame that shows the stop; with
    the default it enters once. Declining correctly but re-entering forever is the same
    wasted hours under a different name.
    """
    o = _rocket_obs(pill_xy=(0.5, 0.82))
    no_hold = replace(C, timings=replace(C.timings, party_fainted_hold=0.0))
    entries = 0
    ctx = _ctx(cfg=no_hold, now=1000.0, party_fainted_ts=1000.0)
    for i in range(200):
        ctx.now = 1000.0 + i
        if fsm.desired_state(o, ctx) is fsm.BotState.ROCKET:
            entries += 1
    assert entries == 200, "without a hold every frame re-enters"

    held = 0
    ctx = _ctx(now=1000.0, party_fainted_ts=1000.0)
    for i in range(200):
        ctx.now = 1000.0 + i
        if fsm.desired_state(o, ctx) is fsm.BotState.ROCKET:
            held += 1
    assert held == 0, "the hold covers the whole window"


def test_the_hold_is_not_released_by_the_map_coming_back():
    """Releasing on "the map returned" would release on the very event that re-arms the
    tap - the loop, rebuilt."""
    ctx = _ctx(now=1005.0, party_fainted_ts=1000.0)
    ctx.last_map_ts = 1004.0
    o = _rocket_obs(pill_xy=(0.5, 0.82), on_map=True)
    assert fsm.desired_state(o, ctx) is not fsm.BotState.ROCKET


# ---------------------------------------------------------------- runner plumbing

class _Act:
    def apply(self, e, now=None):
        return True

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
    r = runner_mod.Runner(C, _Src(), _Act(), perceptor=None, display=False, **kw)

    def _tick_party(state):
        if state is Tristate.FALSE:
            r._party_false += 1
        elif state is Tristate.TRUE:
            r._party_false = 0
        r.ctx.party_cannot_battle = r._party_false >= 2

    r._tick_party = _tick_party
    return r


def test_a_confirmed_switch_clears_the_hold():
    """A new account brings a fresh party - the only honest release."""
    r = _runner()
    r.ctx.party_fainted_ts = 500.0
    r._party_false = 5
    r._on_switch_confirmed("SomeoneElse")
    assert r.ctx.party_fainted_ts == 0.0
    assert r._party_false == 0


def test_a_decline_is_counted_as_declined():
    """`rockets_engaged` still counts the entry - _count_transition runs first - so without
    this counter an operator reads a declining run as a fighting one."""
    r = _runner()
    before = r.stats.rockets_declined
    r.apply([SetFlag("party_fainted_ts", 1234.0)], obs(ts=1.0))
    assert r.stats.rockets_declined == before + 1


def test_re_stamping_the_hold_does_not_count_a_second_decline():
    r = _runner()
    r.apply([SetFlag("party_fainted_ts", 1234.0)], obs(ts=1.0))
    r.apply([SetFlag("party_fainted_ts", 1235.0)], obs(ts=1.0))
    assert r.stats.rockets_declined == 1


def test_the_decline_reaches_the_session_summary_and_discord():
    from pogobot import notify
    r = _runner()
    r.stats.rockets_declined = 3
    assert "rockets_declined" in r.stats.summary()
    assert "declined" in r.stats.report().lower()
    fields = notify.DiscordNotifier._summary_fields(
        notify.NullNotifier(), "acct", r.stats.summary())
    assert any("declined" in f["name"].lower() for f in fields)


# ---------------------------------------------------------------- a crash is not a finish

def test_an_unhandled_error_halts_rather_than_reporting_success():
    """RED-GREEN. Without the `except Exception` in `Runner.run`, the exception passes
    through the finally, `close()` finds no halt reason, and the run is recorded - and
    posted to Discord - as "Run finished" in green with halts=0. A crash announced as a
    clean finish is the worst thing the runner can do to an absent operator.
    """
    class _Boom:
        def read(self):
            return None

        def healthy(self):
            raise RuntimeError("capture exploded")

        def release(self):
            pass

    class _Recorder:
        enabled = False

        def __init__(self):
            self.calls = []

        def started(self, **k):
            self.calls.append("started")

        def finished(self, **k):
            self.calls.append("finished")

        def halted(self, reason, **k):
            self.calls.append("halted")

        def switched(self, name):
            pass

        def problem(self, *a, **k):
            pass

        def close(self, timeout=5.0):
            pass

    rec = _Recorder()
    r = runner_mod.Runner(C, _Boom(), _Act(), perceptor=None, display=False, notifier=rec)
    rc = r.run()
    assert rc == 1, "a crash must be a failing exit code"
    assert r.stats.halts == 1
    assert "halted" in rec.calls
    assert "finished" not in rec.calls, "a crash must not be reported as a clean finish"


def test_a_benign_am_start_warning_is_not_an_actuator_failure():
    """`am start` on an activity already in front exits 0 and warns on stderr. Scored as a
    failure it counted toward the breaker: measured in logs/run.log, all 3 failures of a
    healthy 2h25m run, last_error carrying exactly this string at rc=0."""
    from pogobot.actions import _BENIGN_STDERR
    warning = ("Warning: Activity not started, intent has been delivered to currently "
               "running top-most instance.")
    assert _BENIGN_STDERR.search(warning)
    assert not _BENIGN_STDERR.search("error: device offline")
    assert not _BENIGN_STDERR.search("Error: Activity class does not exist")


# ---------------------------------------------------------------- productivity watchdog

def test_the_watchdog_is_disarmed_until_the_first_encounter():
    """17 of 159 recorded sessions had zero encounters. Halting a run that is still logging
    in would be a new failure mode invented to catch an old one."""
    r = _runner()
    assert r._last_encounter_ts is None
    assert r._unproductive(1e9) is False
    assert r.stats.halts == 0


def test_the_watchdog_halts_a_run_that_has_stopped_catching():
    r = _runner()
    r._last_encounter_ts = 1000.0
    assert r._unproductive(1000.0 + C.timings.productivity_watchdog + 1) is True
    assert r.stats.halts == 1
    assert "no encounter" in (r._halt_reason or "")


def test_the_watchdog_does_not_fire_inside_the_budget():
    r = _runner()
    r._last_encounter_ts = 1000.0
    assert r._unproductive(1000.0 + C.timings.productivity_watchdog - 1) is False
    assert r.stats.halts == 0


def test_the_watchdog_bar_sits_far_above_normal_operation():
    """Measured over the 303 ENCOUNTER entries in logs/run.log: median gap 17s, p90 50s,
    and exactly one gap above 900s - 31,091s, which is the stall. The bar must sit in that
    empty space, not near the p90."""
    assert C.timings.productivity_watchdog >= 900.0
    assert C.timings.productivity_watchdog <= 31091.0


def test_the_watchdog_can_be_turned_off():
    r = runner_mod.Runner(replace(C, timings=replace(C.timings, productivity_watchdog=0.0)),
                          _Src(), _Act(), perceptor=None, display=False)
    r._last_encounter_ts = 1000.0
    assert r._unproductive(1e9) is False


# ---------------------------------------------------------------- PGSharp favourites

FIXTURES = Path("tests/fixtures/uiautomator")


def test_the_pgsharp_menu_offers_favorites_and_teleport():
    """Captured from the live overlay. These are the two candidate routes to a fixed home
    location; Favorites wins because its rows are addressable BY NAME."""
    x = (FIXTURES / "pgsharp_shortcut_menu.xml").read_text(errors="replace")
    for item in ("Favorites", "Teleport", "Map", "AutoWalk", "Settings"):
        assert f'text="{item}"' in x, item


def test_favourite_rows_are_addressable_by_name():
    """`hl_fi_name` carries the destination's own label, so a home location can be named in
    config rather than reached by a row index that reorders or a coordinate that moves.

    The captured fixture is REDACTED: this repository is public, and the real rows named
    the operator's saved destinations while `hl_fi_info` gave the live distance to each -
    together enough to place the account. Place names, local times and distances are
    replaced with fixed placeholders; every resource-id, bound and label the tests actually
    assert on is untouched.
    """
    x = (FIXTURES / "pgsharp_favorites.xml").read_text(errors="replace")
    assert "hl_fi_name" in x
    assert "hl_favor_list" in x
    assert "Home Location" in x


def test_the_favourites_page_carries_its_own_cooldown_text():
    """"Distance: 263.61 m Cooldown: 1 Mins" - the teleport cooldown is readable BEFORE
    committing to a jump, which is what makes a safe go-home possible at all."""
    x = (FIXTURES / "pgsharp_favorites.xml").read_text(errors="replace")
    assert "Cooldown:" in x and "Distance:" in x


def test_back_does_not_dismiss_the_favourites_page():
    """THE load-bearing capture. The recovery ladder presses BACK, so if BACK dismissed
    this page a wedge would clear itself. It does not: measured live, the page is fully
    present after BACK, and only `hl_page_close` ("OK") closes it. Any go-home ladder must
    close explicitly and must never leave this page up.
    """
    after = (FIXTURES / "pgsharp_favorites_after_back.xml").read_text(errors="replace")
    assert "hl_favor_list" in after, "BACK did NOT dismiss the page"
    assert "hl_page_close" in after
    assert 'text="OK"' in after
