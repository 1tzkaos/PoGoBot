"""Per-account settings, from a JSON file the operator edits by hand.

Accounts are not interchangeable. One may be a rocket-hunting account and another a
catching account, and until now every switch carried the whole run's flags across with it -
`--no-rockets` was a property of the invocation, not of the account it applied to.

This is deliberately a separate file rather than more CLI flags: a flag cannot say
"different for that account", and a run that rotates through several accounts would need
the operator to predict the rotation. It is also deliberately tiny. Only keys that are
genuinely per-account belong here; everything else stays in `Config`, where it is typed,
documented and defaulted in one place.

The file is optional. A missing file, an unreadable one and an empty one all mean "every
account uses the run's own settings", which is exactly the behaviour that predates this
module - so nothing changes for anyone who never creates it.
"""

from __future__ import annotations

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


def load_profiles(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    """Read the file. Never raises: a broken profile must not stop the bot from playing.

    Returns account name -> settings. The `default` entry, if present, is kept under its
    own key and merged by `settings_for`.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("could not read %s (%s); every account will use this run's own "
                    "settings", p, exc)
        return {}
    if not isinstance(raw, dict):
        log.warning("%s should be a JSON object mapping account names to settings; "
                    "ignoring it", p)
        return {}

    out: dict[str, dict[str, Any]] = {}
    for account, settings in raw.items():
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
