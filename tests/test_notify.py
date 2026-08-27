"""A notifier must never be able to hurt the run it reports on.

Everything here is a variation on that one sentence. Discord is a third party on the far
side of a network the bot does not control: it can be slow, it can 429, it can 404 a
deleted webhook, it can accept a connection and never answer. None of that is allowed to
raise into `Runner`, and none of it is allowed to cost frames - the tick loop has ~125ms at
the default `infer_fps` and a POST can eat all of it.

The other half is the credential. A webhook URL grants posting rights to someone's channel,
so it must never reach a log line, and a URL that is not actually Discord's must never be
POSTed to at all - a mistyped host would otherwise receive this run's account names.
"""
import json
import queue
import threading
import time
import urllib.error

import pytest

from pogobot import notify
from pogobot.notify import DiscordNotifier, NullNotifier, from_settings

GOOD = "https://discord.com/api/webhooks/123456789/tok3n-abcdef"


class _Resp:
    """Minimal stand-in for what urlopen returns."""

    def __init__(self, code=204):
        self.code = code

    def read(self):
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Opener:
    """Records requests instead of making them, and signals when one arrives."""

    def __init__(self, behaviour=None):
        self.requests = []
        self.seen = threading.Event()
        self._behaviour = behaviour or (lambda n: _Resp())

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        try:
            return self._behaviour(len(self.requests))
        finally:
            self.seen.set()

    def payloads(self):
        return [json.loads(r.data.decode("utf-8")) for r in self.requests]


def _notifier(opener, **kw):
    kw.setdefault("min_interval", 0.0)
    return DiscordNotifier(GOOD, opener=opener, **kw)


def _wait(opener, count=1, timeout=2.0):
    """Block until `count` requests have been made, or fail the test."""
    deadline = time.monotonic() + timeout
    while len(opener.requests) < count and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(opener.requests) >= count, (
        f"expected {count} request(s), saw {len(opener.requests)}")


# ---------------------------------------------------------------- the URL is a credential

@pytest.mark.parametrize("url", [
    "https://discord.com/api/webhooks/1/tok",
    "https://discordapp.com/api/webhooks/1/tok",
    "https://ptb.discord.com/api/webhooks/1/tok",
])
def test_real_discord_webhook_urls_are_accepted(url):
    assert notify.valid_url(url)


@pytest.mark.parametrize("url,why", [
    ("http://discord.com/api/webhooks/1/tok", "plaintext http would send the token in clear"),
    ("https://evil.example.com/api/webhooks/1/tok", "a non-Discord host must never be POSTed to"),
    ("https://discord.com/api/v10/users/@me", "a Discord URL that is not a webhook"),
    ("https://discord.com.evil.example/api/webhooks/1/tok", "suffix does not make it Discord"),
    ("", "nothing configured"),
    ("not a url", "unparseable"),
])
def test_anything_that_is_not_a_discord_webhook_is_refused(url, why):
    assert not notify.valid_url(url), why


def test_a_bad_host_never_receives_a_request():
    """The refusal has to happen before the socket, not after.

    This is the case the check exists for: a typo in the host is silent otherwise, and the
    first thing posted carries the account name and the run's statistics.
    """
    opener = _Opener()
    n = from_settings("https://evil.example.com/api/webhooks/1/tok", env={})
    assert isinstance(n, NullNotifier)
    n.started(account="MiniStank")
    assert opener.requests == []


def test_the_token_never_appears_in_the_masked_form():
    masked = notify.masked(GOOD)
    assert "tok3n-abcdef" not in masked
    assert "123456789" in masked, "the id identifies which webhook, and is not secret"


def test_an_invalid_url_is_reported_without_echoing_it(caplog):
    with caplog.at_level("WARNING"):
        assert isinstance(from_settings("https://evil.example.com/api/webhooks/1/sekrit",
                                        env={}), NullNotifier)
    assert "sekrit" not in caplog.text


# ---------------------------------------------------------------- configuration precedence

def test_nothing_configured_yields_a_null_notifier():
    assert isinstance(from_settings(None, env={}), NullNotifier)
    assert isinstance(from_settings("", env={}), NullNotifier)
    assert isinstance(from_settings("   ", env={}), NullNotifier)


def test_the_environment_is_used_when_no_url_was_passed():
    n = from_settings(None, env={notify.ENV_VAR: GOOD})
    try:
        assert isinstance(n, DiscordNotifier)
    finally:
        n.close()


def test_an_explicit_url_beats_the_environment():
    """The file's existing rule - a typed flag wins - extended to the environment."""
    other = "https://discord.com/api/webhooks/999/other"
    n = from_settings(other, env={notify.ENV_VAR: GOOD})
    try:
        assert n._url == other
    finally:
        n.close()


# ---------------------------------------------------------------- the null object

def test_the_null_notifier_answers_every_call_the_real_one_does():
    """Call sites must be able to hold either without a guard.

    A method added to `DiscordNotifier` and forgotten here is a crash on the day it fires,
    which for `halted` means the run's most important notification is the one that breaks.
    """
    public = {m for m in vars(DiscordNotifier) if not m.startswith("_")}
    events = public - {"post", "enabled"}
    missing = [m for m in events if not hasattr(NullNotifier, m)]
    assert missing == [], f"NullNotifier is missing {missing}"


def test_the_null_notifier_does_nothing_loudly():
    n = NullNotifier()
    n.started(account="x")
    n.halted("because", account="x", summary={})
    n.finished(account="x", summary={})
    n.switched("y")
    n.problem("t", "d")
    n.close()
    assert n.enabled is False


# ---------------------------------------------------------------- delivery

def test_an_event_is_delivered_as_a_discord_embed():
    opener = _Opener()
    n = _notifier(opener)
    try:
        n.started(account="MiniStank", settings="rockets=True")
        _wait(opener)
    finally:
        n.close()
    body = opener.payloads()[0]
    embed = body["embeds"][0]
    assert embed["title"] == "Run started"
    assert embed["color"] == notify.COLOR_START
    assert any(f["value"] == "MiniStank" for f in embed["fields"])
    assert opener.requests[0].get_method() == "POST"
    assert opener.requests[0].headers["Content-type"] == "application/json"


def test_a_halt_is_red_and_leads_with_the_reason():
    opener = _Opener()
    n = _notifier(opener)
    try:
        n.halted("capture source died", account="A", summary={"uptime": "1h"})
        _wait(opener)
    finally:
        n.close()
    embed = opener.payloads()[0]["embeds"][0]
    assert embed["color"] == notify.COLOR_HALT
    assert embed["fields"][0]["name"] == "Reason"
    assert "capture source died" in embed["fields"][0]["value"]


def test_the_summary_does_not_relabel_throws_as_catches():
    """`stats.py` is careful that a throw is not a catch. A notification is exactly where
    that care would be undone, because nobody reads an embed sceptically."""
    opener = _Opener()
    n = _notifier(opener)
    try:
        n.finished(account="A", summary={"balls_thrown": 40, "encounters": 12})
        _wait(opener)
    finally:
        n.close()
    text = json.dumps(opener.payloads()[0]).lower()
    assert "balls thrown" in text
    assert "catch" not in text


# ---------------------------------------------------------------- it cannot hurt the run

def test_a_network_failure_never_raises():
    def boom(_n):
        raise urllib.error.URLError("no route to host")
    opener = _Opener(boom)
    n = _notifier(opener)
    try:
        n.started(account="A")
        _wait(opener)
        n.problem("still fine")
    finally:
        n.close()


def test_a_deleted_webhook_never_raises():
    def gone(_n):
        raise urllib.error.HTTPError(GOOD, 404, "Not Found", {}, None)
    opener = _Opener(gone)
    n = _notifier(opener)
    try:
        n.halted("reason", account="A")
        _wait(opener)
    finally:
        n.close()


def test_an_http_error_is_logged_without_the_url(caplog):
    def gone(_n):
        raise urllib.error.HTTPError(GOOD, 404, "Not Found", {}, None)
    opener = _Opener(gone)
    n = _notifier(opener)
    with caplog.at_level("WARNING"):
        try:
            n.started(account="A")
            _wait(opener)
        finally:
            n.close()
    assert "tok3n-abcdef" not in caplog.text
    assert "404" in caplog.text


def test_posting_does_not_block_on_a_slow_discord():
    """The point of the worker thread. A POST that takes a second must not cost the tick
    loop a second - at 8Hz that is eight dropped frames per notification."""
    release = threading.Event()

    def slow(_n):
        release.wait(timeout=5.0)
        return _Resp()

    opener = _Opener(slow)
    n = _notifier(opener)
    try:
        t0 = time.monotonic()
        for _ in range(5):
            n.problem("slow")
        elapsed = time.monotonic() - t0
        assert elapsed < 0.2, f"post() blocked for {elapsed:.3f}s"
    finally:
        release.set()
        n.close()


def test_a_full_queue_drops_instead_of_blocking():
    """An unreachable Discord must not turn into unbounded memory, and must not wedge the
    caller once the backlog is full."""
    release = threading.Event()

    def slow(_n):
        release.wait(timeout=5.0)
        return _Resp()

    opener = _Opener(slow)
    n = _notifier(opener, queue_size=2)
    try:
        t0 = time.monotonic()
        for _ in range(50):
            n.problem("flood")
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, f"post() blocked for {elapsed:.3f}s on a full queue"
        assert n._dropped > 0, "a bounded queue that never drops is not bounded"
    finally:
        release.set()
        n.close()


def test_the_worker_survives_a_failure_and_delivers_the_next_event():
    """A worker that dies on one bad response silently swallows every later notification -
    including the halt that would explain the run."""
    def first_fails(n):
        if n == 1:
            raise urllib.error.URLError("transient")
        return _Resp()

    opener = _Opener(first_fails)
    n = _notifier(opener)
    try:
        n.problem("one")
        n.problem("two")
        _wait(opener, count=2)
    finally:
        n.close()
    assert len(opener.requests) >= 2


# ---------------------------------------------------------------- rate limiting

def test_a_429_is_retried_once_after_the_delay_discord_asks_for():
    class _Body:
        def read(self):
            return json.dumps({"retry_after": 0.01}).encode()

    def limited(n):
        if n == 1:
            err = urllib.error.HTTPError(GOOD, 429, "Too Many Requests", {}, None)
            err.read = _Body().read
            raise err
        return _Resp()

    opener = _Opener(limited)
    n = _notifier(opener)
    try:
        n.problem("x")
        _wait(opener, count=2)
    finally:
        n.close()
    assert len(opener.requests) == 2, "the message should be delivered on the retry"


def test_an_absurd_retry_after_is_dropped_rather_than_waited_on():
    """Parking the worker for minutes holds every later notification behind it."""
    err = urllib.error.HTTPError(GOOD, 429, "Too Many", {}, None)
    err.read = lambda: json.dumps({"retry_after": 9999}).encode()
    assert DiscordNotifier._retry_after(err) is None


def test_a_missing_retry_after_is_not_guessed():
    err = urllib.error.HTTPError(GOOD, 429, "Too Many", {}, None)
    err.read = lambda: b"not json"
    assert DiscordNotifier._retry_after(err) is None


def test_the_worker_paces_itself_under_the_webhook_bucket():
    opener = _Opener()
    n = DiscordNotifier(GOOD, opener=opener, min_interval=0.05)
    try:
        t0 = time.monotonic()
        n.problem("a")
        n.problem("b")
        n.problem("c")
        _wait(opener, count=3, timeout=3.0)
        assert time.monotonic() - t0 >= 0.10, "three posts should span at least two gaps"
    finally:
        n.close()


# ---------------------------------------------------------------- shutdown

def test_close_flushes_what_is_queued():
    opener = _Opener()
    n = _notifier(opener)
    n.finished(account="A", summary={"uptime": "2h"})
    n.close()
    assert len(opener.requests) == 1


def test_close_is_bounded_when_discord_never_answers():
    """`Runner.close` runs on the way out; an unreachable Discord must not hold a shutdown
    open indefinitely."""
    release = threading.Event()

    def never(_n):
        release.wait(timeout=10.0)
        return _Resp()

    opener = _Opener(never)
    n = _notifier(opener)
    try:
        n.problem("x")
        t0 = time.monotonic()
        n.close(timeout=0.2)
        assert time.monotonic() - t0 < 2.0
    finally:
        release.set()


def test_posting_after_close_is_ignored():
    opener = _Opener()
    n = _notifier(opener)
    n.close()
    n.problem("late")
    assert len(opener.requests) == 0


# ---------------------------------------------------------------- wired into the Runner

class _Recorder(NullNotifier):
    """A notifier that records instead of posting, without touching a socket."""

    def __init__(self):
        self.calls = []

    def started(self, **k):
        self.calls.append(("started", k))

    def finished(self, **k):
        self.calls.append(("finished", k))

    def halted(self, reason, **k):
        self.calls.append(("halted", reason))

    def switched(self, name):
        self.calls.append(("switched", name))

    def problem(self, title, detail=""):
        self.calls.append(("problem", title))

    def close(self, timeout=5.0):
        self.calls.append(("close", None))

    def names(self):
        return [c[0] for c in self.calls]


class _Act:
    def apply(self, effect, now=None):
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
    from pogobot import runner as runner_mod
    from pogobot.config import DEFAULT
    return runner_mod.Runner(DEFAULT, _Src(), _Act(), perceptor=None, display=False, **kw)


def test_a_runner_with_no_webhook_configured_still_runs():
    """The default must be a working object, not None, or every call site needs a guard."""
    r = _runner()
    assert isinstance(r.notifier, NullNotifier)
    r._halt("nothing should explode")


def test_a_halt_is_posted_once_no_matter_how_many_times_it_is_declared():
    """`_halt` is reached from four places and the later reasons are consequences of the
    first; a phone buzzing four times says nothing the first buzz did not."""
    rec = _Recorder()
    r = _runner(notifier=rec)
    r._halt("capture source died")
    r._halt("actuator circuit breaker tripped")
    r._halt("and again")
    assert rec.names().count("halted") == 1
    assert rec.calls[0][1] == "capture source died"


def test_a_halted_run_is_not_also_reported_as_finished():
    """"Run finished" arriving after "Run HALTED" reads as a recovery that never happened."""
    rec = _Recorder()
    r = _runner(notifier=rec)
    r._halt("capture source died")
    r.close()
    assert "finished" not in rec.names()
    assert "close" in rec.names()


def test_a_clean_stop_is_reported_as_finished():
    rec = _Recorder()
    r = _runner(notifier=rec)
    r.close()
    assert "finished" in rec.names()
    assert "close" in rec.names()


def test_a_notifier_that_throws_cannot_stop_the_run_from_closing():
    """The whole contract in one test: a broken notifier must not eat the session record."""
    class _Broken(NullNotifier):
        def finished(self, **k):
            raise RuntimeError("discord exploded")

        def close(self, timeout=5.0):
            raise RuntimeError("and again on the way out")

    r = _runner(notifier=_Broken())
    r.close()


def test_the_summary_is_still_posted_when_stats_cannot_be_summarised():
    """`_safe_summary` runs inside `_halt`, the one place that records a run stopping."""
    rec = _Recorder()
    r = _runner(notifier=rec)

    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError("stats are broken")

    r.stats = _Boom()
    with pytest.raises(Exception):
        _ = r.stats.halts
    r.notifier = rec
    assert r._safe_summary() is None


# ---------------------------------------------------------------- the file must not leak it

def test_config_json_reports_the_webhook_as_set_not_as_its_value():
    """`apply_run_settings` logs what the file applied, which for a credential would put
    the token in the log, in terminal scrollback, and in any pasted bug report.

    The value itself must still reach the namespace - redacting the log must not redact
    the setting.
    """
    import argparse
    from pogobot import userconfig

    p = argparse.ArgumentParser()
    p.add_argument("--discord-webhook", default=None)
    p.add_argument("--switch-every", type=float, default=1.0)
    ns = p.parse_args([])

    secret = "https://discord.com/api/webhooks/111/SUPERSECRET"
    applied = userconfig.apply_run_settings(
        {"discord_webhook": secret, "switch_every": 45}, p, ns, set(), "config.json")

    line = ", ".join(applied)
    assert "SUPERSECRET" not in line, "the token reached the startup log"
    assert "discord_webhook=<set>" in line, "it should still say the key was applied"
    assert "switch_every=45.0" in line, "ordinary settings are still reported by value"
    assert ns.discord_webhook == secret, "redacting the log must not redact the setting"


def test_the_secret_key_list_matches_the_flag_that_carries_a_credential():
    from pogobot import userconfig
    assert "discord_webhook" in userconfig.SECRET_KEYS
