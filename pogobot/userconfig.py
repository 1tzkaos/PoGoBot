"""`config.json`: the settings a run uses, from a file the operator edits by hand.

Two things live here, because they are the same question asked at two scopes: what this
RUN does, and what it does differently depending on which ACCOUNT it is logged into.

    {
      "tui": true,
      "switch_every": 45,
      "accounts": {
        "default":   { "fight_rockets": true },
        "MiniStank": { "fight_rockets": false }
      }
    }

Top level keys are command-line options by their long name with dashes as underscores, so
`--switch-every 45` is `"switch_every": 45` and `--tui` is `"tui": true`. There is no second
vocabulary to learn and no list to keep in sync: whatever the parser accepts, the file
accepts, because the keys ARE the parser's.

A command-line flag still wins. The file is where a run's usual shape lives; the flag is
how one invocation departs from it, and an operator typing `--no-rockets` should not have
to remember what the file says. Only options actually typed override the file - not the
parser's own defaults, which would otherwise silently beat every line in it.

Accounts are not interchangeable, which is the other half. One may be a rocket-hunting
account and another a catching account, and a flag cannot say "different for that account":
a rotating run would need the operator to predict the rotation. Only keys that are
genuinely per-account belong in `accounts`; everything else stays in `Config`, where it is
typed, documented and defaulted in one place.

The file is optional and never fatal. Missing, unreadable and malformed all mean "use the
command line and the defaults", which is exactly the behaviour that predates this module.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("pogobot")

#: Settings an account may override. Anything else in the file is a typo, and typos in a
#: hand-edited file are silent unless something says so - `fight_rocket` would otherwise
#: read as "leave it at the default" and the operator would be told nothing.
KNOWN_KEYS = frozenset({"fight_rockets"})

#: The key whose settings apply to every account that has no entry of its own.
DEFAULT_KEY = "default"

#: Where the per-account block lives inside the file.
ACCOUNTS_KEY = "accounts"


def load(path: Optional[Path]) -> dict[str, Any]:
    """Read the whole file. Never raises: a broken config must not stop the bot playing."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("could not read %s (%s); using the command line and the defaults",
                    p, exc)
        return {}
    if not isinstance(raw, dict):
        log.warning("%s should be a JSON object; ignoring it", p)
        return {}
    return raw


def load_profiles(raw: dict[str, Any], where: str = "config.json"
                  ) -> dict[str, dict[str, Any]]:
    """The `accounts` block: account name -> settings.

    The `default` entry, if present, is kept under its own key and merged by
    `settings_for`.
    """
    block = raw.get(ACCOUNTS_KEY, {})
    if not isinstance(block, dict):
        log.warning('%s: "%s" should be an object mapping account names to settings; '
                    "ignoring it", where, ACCOUNTS_KEY)
        return {}
    p = where
    out: dict[str, dict[str, Any]] = {}
    for account, settings in block.items():
        if not isinstance(settings, dict):
            log.warning("%s: entry for %r should be an object like "
                        '{"fight_rockets": false}; ignoring it', p, account)
            continue
        clean: dict[str, Any] = {}
        for key, value in settings.items():
            if key not in KNOWN_KEYS:
                log.warning("%s: %r sets unknown option %r; known options are %s",
                            p, account, key, ", ".join(sorted(KNOWN_KEYS)))
                continue
            if not isinstance(value, bool):
                log.warning("%s: %r sets %s to %r; expected true or false",
                            p, account, key, value)
                continue
            clean[key] = value
        out[account] = clean
    return out


def settings_for(profiles: dict[str, dict[str, Any]],
                 account: Optional[str]) -> dict[str, Any]:
    """The settings that apply to `account`: the `default` entry, then its own on top.

    An unknown account - or a session that never learned its name - gets the defaults,
    which is the same answer as having no file at all when there is no `default` entry.
    """
    merged = dict(profiles.get(DEFAULT_KEY, {}))
    if account:
        merged.update(profiles.get(account, {}))
    return merged


def describe(profiles: dict[str, dict[str, Any]]) -> str:
    """One line for the startup log, so the operator can see the file was picked up."""
    named = [a for a in profiles if a != DEFAULT_KEY]
    if not named and DEFAULT_KEY not in profiles:
        return "no per-account settings"
    parts = []
    if DEFAULT_KEY in profiles:
        parts.append(_one(DEFAULT_KEY, profiles[DEFAULT_KEY]))
    parts.extend(_one(a, profiles[a]) for a in sorted(named))
    return "per-account settings: " + "; ".join(parts)


def _one(account: str, settings: dict[str, Any]) -> str:
    if not settings:
        return f"{account}: (nothing set)"
    return f"{account}: " + ", ".join(f"{k}={str(v).lower()}"
                                      for k, v in sorted(settings.items()))


def apply_run_settings(raw: dict[str, Any], parser, namespace, explicit: set,
                       where: str = "config.json") -> list[str]:
    """Fold the file's top-level keys into the parsed arguments.

    `explicit` is the set of option dests the operator actually typed; those are left
    alone. Everything else in the file is written onto `namespace`, so the file behaves
    like a set of defaults rather than a second source of truth the CLI has to fight.

    Keys are matched to the parser's own options by `dest`, which is why there is no list
    of supported settings to maintain and no way for one to drift: an option that exists is
    settable, and one that does not is reported by name rather than ignored, because in a
    hand-edited file a typo looks exactly like a line that is working.

    Returns the settings actually applied, for the startup log.
    """
    by_dest = {}
    for action in parser._actions:
        if action.dest not in ("help", argparse.SUPPRESS):
            by_dest[action.dest] = action

    applied: list[str] = []
    for key, value in raw.items():
        if key == ACCOUNTS_KEY:
            continue
        action = by_dest.get(key)
        if action is None:
            log.warning("%s: unknown setting %r; keys are command-line options with "
                        "dashes as underscores, like \"switch_every\" for --switch-every",
                        where, key)
            continue
        if key in explicit:
            log.info("%s: %s is set on the command line, so the file's value is not used",
                     where, key)
            continue
        coerced = _coerce(action, key, value, where)
        if coerced is _BAD:
            continue
        setattr(namespace, key, coerced)
        applied.append(f"{key}={_show(coerced)}")
    return applied


_BAD = object()


def _coerce(action, key: str, value: Any, where: str) -> Any:
    """Make a JSON value look like something the parser would have produced."""
    is_flag = isinstance(getattr(action, "const", None), bool) or (
        action.__class__.__name__ in ("_StoreTrueAction", "_StoreFalseAction"))
    if is_flag:
        if not isinstance(value, bool):
            log.warning("%s: %s expects true or false, got %r", where, key, value)
            return _BAD
        return value
    if value is None:
        return None
    if action.type is not None:
        try:
            return action.type(value)
        except (TypeError, ValueError) as exc:
            log.warning("%s: %s could not take %r (%s)", where, key, value, exc)
            return _BAD
    return value


def _show(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)
