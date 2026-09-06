"""Optional Discord notifications for the handful of moments worth interrupting someone.

A run is unattended by design: the operator starts it and walks away. The things that go
wrong are therefore invisible until they come back - a halt at minute three of a six-hour
run reads, from the outside, exactly like a run still going. This posts the few events
that change what the operator would DO, and nothing else.

Three properties matter more than the feature itself, because a notifier that fails badly
is worse than no notifier at all.

**It never stops the bot.** Every public method swallows everything it can raise, the same
contract `userconfig.load` holds and for the same reason: a convenience must not be able to
take down the thing it is reporting on. A webhook that 404s, a DNS failure, a proxy that
hangs, a malformed embed - all of it lands in the log and the run plays on.

**It never blocks the tick loop.** `Runner.run` has a budget of one frame at `infer_fps`
(125ms at the default 8) and spends most of it in inference. A Discord POST is 100-400ms on
a good day and can hang until the socket timeout on a bad one, so posting inline would drop
frames on exactly the events - halts, switch failures - where the next few frames matter
most. `post` therefore only enqueues; a single daemon worker owns the network entirely.

**It cannot flood.** Discord's webhook bucket is roughly 5 requests per 2 seconds, and
exceeding it earns a 429 with a `retry_after` that a naive client turns into a tight retry
loop. The worker paces itself below the bucket, honours `retry_after` when it sees one, and
the queue is bounded: if Discord is unreachable the backlog is dropped and counted rather
than grown without limit. Dropping notifications is the correct failure mode here; holding
memory to deliver a stale one is not.

The URL is a credential - anyone holding it can post to the channel - so it is never
logged, never put in an embed, and never accepted unless it is actually a Discord webhook
endpoint (see `valid_url`). That last check is not pedantry: a mistyped or copy-pasted URL
would otherwise POST this run's account names and statistics to whatever host was named.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional
from urllib.parse import urlparse

log = logging.getLogger("pogobot")

#: Environment variable consulted when no `--discord-webhook` was typed. Overrides
#: `config.json` so the secret can stay out of a file the operator edits by hand.
ENV_VAR = "POGOBOT_DISCORD_WEBHOOK"

#: Hosts Discord actually serves webhooks from. `discordapp.com` is the legacy domain and
#: still works, so URLs saved years ago keep working.
_ALLOWED_HOSTS = frozenset({"discord.com", "discordapp.com",
                            "ptb.discord.com", "canary.discord.com"})

#: Seconds the worker leaves between posts. Discord's webhook bucket is ~5 per 2s; one
#: post per half-second is comfortably inside it and still drains a burst promptly.
MIN_INTERVAL = 0.5

#: How long a single POST may take before it is abandoned. Deliberately short: a hung
#: connection must not hold the queue, and a missed notification costs nothing.
TIMEOUT = 10.0

#: Pending messages held before new ones are dropped. Only reachable when Discord is
#: unreachable, since the events below are rare by construction.
QUEUE_SIZE = 64

#: A 429 asking for longer than this is treated as "Discord does not want us right now"
#: and the message is dropped rather than parked, holding the worker and the queue behind
#: it. Webhook buckets return fractions of a second; minutes means something else is wrong.
MAX_RETRY_AFTER = 30.0

# Embed colours, chosen so severity is readable at a glance in a phone notification.
COLOR_START = 0x5865F2   # blurple  - a run began
COLOR_OK = 0x2ECC71      # green    - a run ended the way it was asked to
COLOR_HALT = 0xED4245    # red      - the run stopped itself and needs attention
COLOR_PROBLEM = 0xF59E0B # amber    - still running, but degraded
COLOR_INFO = 0x949CF7    # pale     - routine progress


def valid_url(url: str) -> bool:
    """True for a real Discord webhook endpoint, over TLS.

    Anything else is refused at construction rather than at post time, so the operator
    learns about a bad URL in the startup banner instead of discovering hours later that
    nothing was ever delivered.
    """
    try:
        u = urlparse(url)
    except ValueError:
        return False
    return (u.scheme == "https"
            and u.hostname in _ALLOWED_HOSTS
            and u.path.startswith("/api/webhooks/"))


def masked(url: str) -> str:
    """A form of the URL safe to put in a log line.

    The webhook id is retained because it is how an operator tells two webhooks apart; the
    token, which is the part that grants posting rights, never appears.
    """
    try:
        parts = urlparse(url).path.rstrip("/").split("/")
        ident = parts[3] if len(parts) > 3 else "?"
    except (ValueError, IndexError):
        ident = "?"
    return f"discord webhook {ident} (token hidden)"


def _field(name: str, value: Any, inline: bool = True) -> dict:
    return {"name": str(name), "value": f"{value}", "inline": bool(inline)}


class NullNotifier:
    """Does nothing, for a run with no webhook configured.

    A null object rather than `Optional[DiscordNotifier]` so no call site has to guard;
    the same reason `NullActuator` exists for `--dry-run`. Every method here mirrors one
    on `DiscordNotifier`.
    """

    enabled = False

    def started(self, *a, **k) -> None: ...
    def finished(self, *a, **k) -> None: ...
    def halted(self, *a, **k) -> None: ...
    def switched(self, *a, **k) -> None: ...
    def heartbeat(self, *a, **k) -> None: ...
    def problem(self, *a, **k) -> None: ...
    def close(self, *a, **k) -> None: ...


class DiscordNotifier:
    """Posts run events to a Discord webhook, off the tick loop.

    Construct with a validated URL; `from_settings` is the ordinary entry point and returns
    a `NullNotifier` when nothing is configured, so callers get a working object either way.
    """

    enabled = True

    def __init__(self, url: str, *, username: str = "PoGoBot",
                 timeout: float = TIMEOUT, queue_size: int = QUEUE_SIZE,
                 min_interval: float = MIN_INTERVAL, opener=None):
        if not valid_url(url):
            raise ValueError("not a Discord webhook URL")
        self._url = url
        self._username = username
        self._timeout = timeout
        self._min_interval = min_interval
        # Injected in tests so the queue/worker contract can be exercised without a socket.
        self._opener = opener or urllib.request.urlopen
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._dropped = 0
        self._sent = 0
        self._failed = 0
        self._lock = threading.Lock()
        self._last_post = 0.0
        self._closed = False
        self._worker = threading.Thread(target=self._drain, name="discord-notify",
                                        daemon=True)
        self._worker.start()

    # ---- public events -------------------------------------------------------------

    def started(self, *, account: Optional[str], settings: str = "") -> None:
        fields = [_field("Account", account or "unknown")]
        if settings:
            fields.append(_field("Settings", settings, inline=False))
        self._embed("Run started", color=COLOR_START, fields=fields)

    def finished(self, *, account: Optional[str], summary: Optional[dict] = None) -> None:
        self._embed("Run finished", color=COLOR_OK,
                    fields=self._summary_fields(account, summary))

    def halted(self, reason: str, *, account: Optional[str],
               summary: Optional[dict] = None) -> None:
        """The one event worth a phone buzzing: the run stopped itself and is not coming
        back. The reason leads, because it is the whole reason to look."""
        fields = [_field("Reason", reason, inline=False)]
        fields += self._summary_fields(account, summary)
        self._embed("Run HALTED", color=COLOR_HALT, fields=fields)

    def heartbeat(self, *, account: Optional[str], summary: Optional[dict] = None,
                  uptime: str = "") -> None:
        """Routine "still working" progress, on a timer.

        Every other event here fires on something going WRONG, which leaves silence
        ambiguous: a healthy six-hour run and a run that died in minute three both say
        nothing at all. This is the one message whose absence is informative.
        """
        title = f"Still running{f' - {uptime}' if uptime else ''}"
        self._embed(title, color=COLOR_INFO,
                    fields=self._summary_fields(account, summary))

    def switched(self, name: str) -> None:
        self._embed("Switched account", color=COLOR_INFO,
                    fields=[_field("Now playing", name)])

    def problem(self, title: str, detail: str = "") -> None:
        """Still running, but degraded - a switch that will not confirm, a spent quota."""
        fields = [_field("Detail", detail, inline=False)] if detail else []
        self._embed(title, color=COLOR_PROBLEM, fields=fields)

    # ---- delivery ------------------------------------------------------------------

    def _summary_fields(self, account: Optional[str],
                        summary: Optional[dict]) -> list[dict]:
        """Render `SessionStats.summary()` as embed fields.

        Only keys that are present are shown, and they keep the names `stats.py` gives
        them. `balls_thrown` is not relabelled "catches" here for the same reason it is not
        called that there: a throw is observable and a catch is not, and a notification is
        exactly where a confident wrong number would do its damage.
        """
        fields = [_field("Account", account or "unknown")]
        if not summary:
            return fields
        for key, label in (("uptime", "Uptime"),
                           ("encounters", "Encounters"),
                           ("balls_thrown", "Balls thrown"),
                           ("stops_collected", "Stops"),
                           ("rockets_engaged", "Rockets"),
                           ("rockets_declined", "Rockets declined"),
                           ("recoveries", "Recoveries"),
                           ("halts", "Halts")):
            if key in summary and summary[key] is not None:
                fields.append(_field(label, summary[key]))
        return fields

    def _embed(self, title: str, *, color: int, fields: list[dict]) -> None:
        payload = {
            "username": self._username,
            "embeds": [{
                "title": title,
                "color": color,
                "fields": fields,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            }],
        }
        self.post(payload)

    def post(self, payload: dict) -> None:
        """Enqueue and return. Never raises, never blocks on the network."""
        if self._closed:
            return
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            # Discord is unreachable and the backlog is stale. Count it; the total is
            # reported once at close rather than a line per drop.
            with self._lock:
                self._dropped += 1
        except Exception:
            log.exception("could not queue a Discord notification")

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                self._pace()
                self._send(item)
            except Exception:
                # The worker must outlive any single failure; a dead worker silently stops
                # every later notification, including the halt that explains the run.
                log.exception("Discord notification worker error")
            finally:
                self._q.task_done()

    def _pace(self) -> None:
        gap = self._min_interval - (time.monotonic() - self._last_post)
        if gap > 0:
            time.sleep(gap)

    def _send(self, payload: dict, _retried: bool = False) -> None:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=body, method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "PoGoBot"})
        self._last_post = time.monotonic()
        try:
            with self._opener(req, timeout=self._timeout) as resp:
                resp.read()
            with self._lock:
                self._sent += 1
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and not _retried:
                wait = self._retry_after(exc)
                if wait is not None:
                    time.sleep(wait)
                    self._last_post = time.monotonic()
                    self._send(payload, _retried=True)
                    return
            with self._lock:
                self._failed += 1
            # The URL is a credential and `exc` may echo it; report the status only.
            log.warning("Discord rejected a notification (HTTP %s)", exc.code)
        except Exception as exc:
            with self._lock:
                self._failed += 1
            log.warning("could not deliver a Discord notification (%s)",
                        type(exc).__name__)

    @staticmethod
    def _retry_after(exc: urllib.error.HTTPError) -> Optional[float]:
        """Seconds Discord asked us to wait, if it named a sane number.

        Both the JSON body and the header are consulted because Discord has used each;
        anything absent, unparseable or longer than `MAX_RETRY_AFTER` means "drop it".
        """
        raw = None
        try:
            raw = json.loads(exc.read().decode("utf-8")).get("retry_after")
        except Exception:
            pass
        if raw is None:
            raw = exc.headers.get("Retry-After") if exc.headers else None
        try:
            wait = float(raw)
        except (TypeError, ValueError):
            return None
        return wait if 0 <= wait <= MAX_RETRY_AFTER else None

    def close(self, timeout: float = 5.0) -> None:
        """Flush what is queued, within a bound, then stop.

        Bounded because this runs on the way out of `Runner.close`: an unreachable Discord
        must not hold a shutdown open. Whatever has not gone by then is reported as dropped
        rather than waited on.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=timeout)
        with self._lock:
            pending = self._q.qsize()
            dropped = self._dropped + max(0, pending - 1)  # -1 for the sentinel
            if dropped or self._failed:
                log.warning("Discord notifications: %d sent, %d failed, %d dropped",
                            self._sent, self._failed, dropped)
            else:
                log.debug("Discord notifications: %d sent", self._sent)


def from_settings(url: Optional[str], *, env: Optional[dict] = None):
    """Build a notifier from the resolved URL, or a `NullNotifier` if there is none.

    Precedence is settled by the caller (`cli`) and follows the file's existing rule that a
    typed flag beats the file; the environment sits between them, so a secret can be kept
    out of `config.json` without having to type it every run.

    A configured-but-invalid URL is a warning, not a failure. The operator asked for
    notifications and will not get them, which is worth saying plainly - but it is not a
    reason to refuse to play.
    """
    import os
    url = url or (env if env is not None else os.environ).get(ENV_VAR) or ""
    url = url.strip()
    if not url:
        return NullNotifier()
    if not valid_url(url):
        log.warning("ignoring the Discord webhook: not an https discord.com "
                    "/api/webhooks/... URL")
        return NullNotifier()
    try:
        n = DiscordNotifier(url)
    except Exception:
        log.exception("could not start Discord notifications")
        return NullNotifier()
    log.info("Discord notifications on (%s)", masked(url))
    return n
