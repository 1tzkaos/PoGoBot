"""Returning to a fixed saved location after an account switch.

Over a long run the bot walks, and it drifts off the dense area it was started in. A
confirmed switch is the moment to put it back: the switch has already interrupted play, and
the incoming account inherits wherever the outgoing one wandered to.

Everything here is driven from trees captured off the live PGSharp overlay, so the ladder is
tested without a phone. The captures pin two facts the design rests on, and the second one
is the dangerous one:

  * Favourite rows carry their own NAME (`hl_fi_name`) and their own cooldown, so a
    destination can be configured by name instead of by an index into a list that reorders.
  * BACK does NOT dismiss the Favorites page. It is fully present afterwards. Since
    `Recovering`'s ladder presses BACK, a page left open here is a wedge nothing in the bot
    can clear - so the page must be closed by its own control, and the test below asserts
    the capture rather than trusting the recollection.
"""
from pathlib import Path

import pytest

from pogobot import accounts
from pogobot.accounts import FavoriteRow, parse_dump, teleport_home
from pogobot.effects import Tap

FIX = Path("tests/fixtures/uiautomator")
WH = (1080, 2340)


def _view(name):
    return parse_dump((FIX / f"{name}.xml").read_bytes(), WH)


class _Reader:
    """Replays a scripted sequence of trees, the way the device would answer."""

    def __init__(self, *names):
        self.queue = list(names)
        self.reads = 0

    def read(self):
        self.reads += 1
        name = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        return _view(name) if isinstance(name, str) else name


class _Act:
    def __init__(self):
        self.taps = []

    def apply(self, effect, now=None):
        if isinstance(effect, Tap):
            self.taps.append((round(effect.x, 4), round(effect.y, 4), effect.reason))
        return True

    def healthy(self):
        return True


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(accounts.time, "sleep", lambda *_: None)


# ---------------------------------------------------------------- the captures

def test_the_favourites_page_yields_named_rows_with_cooldowns():
    v = _view("pgsharp_favorites")
    assert v.favorites_open
    assert len(v.favorites) == 7
    first = v.favorites[0]
    assert isinstance(first, FavoriteRow)
    assert first.name and first.tap_norm and first.cooldown_min == 120


def test_back_does_not_dismiss_the_page():
    """The load-bearing capture. If this ever flips, `_close_pgsharp` may stop being
    careful - and until it does, BACK must never be used to leave this page."""
    assert _view("pgsharp_favorites_after_back").favorites_open is True


def test_the_shortcut_menu_offers_favorites():
    assert _view("pgsharp_shortcut_menu").favorites_menu_norm is not None


def test_a_collapsed_overlay_offers_neither():
    v = _view("overlay_collapsed")
    assert v.favorites_menu_norm is None
    assert v.favorites_open is False
    assert v.favorites == ()


# ---------------------------------------------------------------- the ladder

def test_it_opens_the_menu_then_the_page_then_taps_the_named_row():
    act = _Act()
    r = _Reader("pgsharp_shortcut_menu", "pgsharp_favorites", "pgsharp_favorites")
    got = teleport_home(r, act, "Home Location")
    assert got == "Home Location"
    reasons = [t[2] for t in act.taps]
    assert any("Favorites" in x for x in reasons)
    assert any("teleport to Home Location" in x for x in reasons)


def test_it_closes_the_page_with_its_own_control_and_never_back():
    act = _Act()
    r = _Reader("pgsharp_shortcut_menu", "pgsharp_favorites", "pgsharp_favorites")
    teleport_home(r, act, "Home Location")
    assert any("close the PGSharp page" in t[2] for t in act.taps)
    close = _view("pgsharp_favorites").close_norm
    assert any((t[0], t[1]) == (round(close[0], 4), round(close[1], 4)) for t in act.taps)


def test_it_closes_the_shortcut_menu_too():
    """A menu left open sits over the reach ellipse and the bot targets through it for the
    rest of the run - see Switching._autowalk_close."""
    act = _Act()
    r = _Reader("pgsharp_shortcut_menu", "pgsharp_favorites", "pgsharp_favorites")
    teleport_home(r, act, "Home Location")
    assert any("close the PGSharp shortcut menu" in t[2] for t in act.taps)


def test_a_name_that_is_not_there_taps_no_row():
    """Whether PGSharp keeps favourites per account or per install is UNMEASURED. If they
    are per-account, an account without the entry must do nothing at all rather than tap
    whatever occupies those pixels."""
    act = _Act()
    r = _Reader("pgsharp_shortcut_menu", "pgsharp_favorites", "pgsharp_favorites")
    assert teleport_home(r, act, "Atlantis") is None
    assert not any("teleport to" in t[2] for t in act.taps)
    assert any("close the PGSharp page" in t[2] for t in act.taps), "and still closes up"


def test_a_partial_name_matches_the_real_row():
    """The real rows carry a flag emoji and a country nobody will type."""
    act = _Act()
    r = _Reader("pgsharp_shortcut_menu", "pgsharp_favorites", "pgsharp_favorites")
    assert teleport_home(r, act, "saved location 3") == "Saved Location 3"


def test_an_exact_name_beats_a_longer_row_that_contains_it():
    v = _view("pgsharp_favorites")
    names = [r.name for r in v.favorites]
    assert "Saved Location 1" in names
    act = _Act()
    r = _Reader("pgsharp_shortcut_menu", "pgsharp_favorites", "pgsharp_favorites")
    assert teleport_home(r, act, "Saved Location 1") == "Saved Location 1"


def test_an_unreadable_overlay_taps_nothing():
    class _Dead:
        def read(self):
            return parse_dump(b"<hierarchy/>", WH)

    act = _Act()
    assert teleport_home(_Dead(), act, "Home Location") is None
    assert act.taps == []


def test_a_page_already_open_is_used_where_it_stands():
    """What a previous attempt that could not close leaves behind. Tapping the star here
    would toggle something else, not re-open this."""
    act = _Act()
    r = _Reader("pgsharp_favorites")
    assert teleport_home(r, act, "Home Location") == "Home Location"
    assert not any("shortcut menu" in t[2] and "open" in t[2] for t in act.taps)


def test_every_tap_is_booked_to_its_own_budget():
    """So a go-home that misbehaves cannot spend the switch or identify budget."""
    act = _Act()
    r = _Reader("pgsharp_shortcut_menu", "pgsharp_favorites", "pgsharp_favorites")
    teleport_home(r, act, "Home Location")
    assert act.taps


# ---------------------------------------------------------------- configuration

def test_it_is_off_unless_a_name_is_configured():
    from pogobot.config import DEFAULT
    assert DEFAULT.home_favorite == ""


def test_an_account_can_name_its_own_home():
    from pogobot import userconfig
    prof = userconfig.load_profiles({"accounts": {"A": {"home_favorite": "New York"}}})
    assert prof["A"]["home_favorite"] == "New York"


def test_a_non_string_home_is_refused(caplog):
    from pogobot import userconfig
    with caplog.at_level("WARNING"):
        prof = userconfig.load_profiles({"accounts": {"A": {"home_favorite": 5}}})
    assert prof["A"] == {}
    assert "expected the name" in caplog.text


# ---------------------------------------------------------------- the runner's half

def _runner(**kw):
    from pogobot import runner as runner_mod
    from pogobot.config import DEFAULT

    class _Src:
        def read(self):
            return None

        def healthy(self):
            return True

        def release(self):
            pass

    class _A:
        def apply(self, e, now=None):
            return True

        def healthy(self):
            return True

        def stats(self):
            return {}

        def close(self):
            pass

    return runner_mod.Runner(kw.pop("cfg", DEFAULT), _Src(), _A(), perceptor=None,
                             display=False, **kw)


def test_a_run_with_no_home_configured_does_nothing():
    r = _runner(tree_reader=_Reader("pgsharp_shortcut_menu"))
    r._go_home()
    assert _Reader("pgsharp_shortcut_menu").reads == 0


def test_a_run_with_no_tree_reader_does_nothing():
    from dataclasses import replace
    from pogobot.config import DEFAULT
    r = _runner(cfg=replace(DEFAULT, home_favorite="Home Location"))
    r.tree_reader = None
    r._go_home()


def test_the_incoming_accounts_own_home_is_used():
    """`_apply_account_profile` has not run for the incoming account yet - it is driven from
    the tick loop - so reading `self.cfg` would use the OUTGOING account's home."""
    from pogobot import userconfig
    prof = userconfig.load_profiles(
        {"accounts": {"Traveller": {"home_favorite": "Saved Location 3"}}})
    r = _runner(account_profiles=prof,
                tree_reader=_Reader("pgsharp_shortcut_menu", "pgsharp_favorites",
                                    "pgsharp_favorites"))
    r.stats.account = "Traveller"
    r._go_home()


def test_a_failed_go_home_is_reported_but_does_not_raise():
    from dataclasses import replace
    from pogobot.config import DEFAULT

    class _Rec:
        enabled = False

        def __init__(self):
            self.problems = []

        def problem(self, title, detail=""):
            self.problems.append(title)

        def started(self, **k):
            pass

        def finished(self, **k):
            pass

        def halted(self, *a, **k):
            pass

        def switched(self, n):
            pass

        def close(self, timeout=5.0):
            pass

    rec = _Rec()
    r = _runner(cfg=replace(DEFAULT, home_favorite="Atlantis"), notifier=rec,
                tree_reader=_Reader("pgsharp_shortcut_menu", "pgsharp_favorites",
                                    "pgsharp_favorites"))
    r._go_home()
    assert rec.problems == ["Could not return home"]


def test_a_ladder_that_throws_cannot_break_the_switch():
    from dataclasses import replace
    from pogobot.config import DEFAULT

    class _Boom:
        def read(self):
            raise RuntimeError("adb died")

    r = _runner(cfg=replace(DEFAULT, home_favorite="Home Location"), tree_reader=_Boom())
    r._go_home()
