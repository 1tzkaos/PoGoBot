"""What a state handler is allowed to ask for.

Handlers are pure: they return a list of Effects and never touch adb, disk, or globals.
The runner is the single place that applies dry-run, rate limits, and tracing - which is
why `--no-click` cannot leak here (the v1 bot checked it at 5 of 10 actuation sites).

All coordinates are NORMALIZED (0.0-1.0). The v1 bot mixed stream pixels and device
pixels freely; cooldowns stored device pixels while detections were in stream pixels.
Normalizing at the boundary makes that entire defect class unrepresentable.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Union


class BotState(enum.Enum):
    BOOT = "BOOT"
    SCANNING = "SCANNING"
    TARGETING = "TARGETING"
    ENCOUNTER = "ENCOUNTER"
    POKESTOP = "POKESTOP"
    ROCKET = "ROCKET"
    SWITCHING = "SWITCHING"
    #: The startup zoom-out / Virtual Go Plus / AutoWalk pass (see `fsm.Preflight`). Its
    #: own state rather than a mode of SWITCHING so a trace can still be read for how many
    #: account switches a run actually performed - the question that diagnosed the failure
    #: this state exists to fix.
    PREFLIGHT = "PREFLIGHT"
    POPUP = "POPUP"
    RECOVERING = "RECOVERING"
    HALTED = "HALTED"


class IntentOutcome(enum.Enum):
    """How a tap-intent resolved. Required at every transition so scoring cannot be
    silently skipped - v1 cleared `current_intent` at 7 sites, 3 of which scored nothing."""

    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    EXPIRED = "EXPIRED"
    CARRIED = "CARRIED"


@dataclass(frozen=True)
class Tap:
    x: float
    y: float
    reason: str
    budget: str = "tap"


@dataclass(frozen=True)
class Swipe:
    x1: float
    y1: float
    x2: float
    y2: float
    reason: str
    duration_ms: int = 200
    budget: str = "swipe"


@dataclass(frozen=True)
class Back:
    reason: str
    budget: str = "back"


@dataclass(frozen=True)
class DoubleTapDrag:
    """A tap immediately followed by a press-and-drag from the same point, delivered as
    ONE adb invocation so the second touch lands inside Android's double-tap window.

    Modeled as the mechanism, not as `ZoomOut`, on purpose: Tap and Swipe already name
    their shape rather than their purpose ("throw ball" and "rotate camera" are both a
    Swipe distinguished only by `reason`), and this gesture happens to be the one-finger
    substitute for pinch-zoom ONLY because multi-touch is unavailable on this device (see
    actions.py). A caller wanting the same tap-then-drag-from-here shape for something
    that is not zoom - a different one-finger gesture PGSharp or the game responds to -
    reuses this and the actuator support behind it for free; a `ZoomOut` effect would have
    made that caller invent a second effect for an identical wire shape.

    (x1, y1) is both the tap point and the drag's start; (x2, y2) is where the drag ends -
    the same field shape as `Swipe`, since underneath it IS a tap plus a swipe from the
    same origin.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    reason: str
    duration_ms: int = 200
    budget: str = "zoom"


@dataclass(frozen=True)
class RestartApp:
    """Force-stop the game and start it again, delivered as ONE adb invocation.

    The last rung of RECOVERING's ladder and the only response left to a process that is
    wedged behind something no button dismisses. Measured: a PGSharp accounts panel that
    BACK does not close and `perception.find_close_button` cannot see held 39% of one
    run's frames in a RECOVERING x47 -> SCANNING x1 cycle, and with the panel up
    `mCurrentFocus` is still the game's own Unity activity - the panel is an overlay
    window of THIS process, so ending the process is what ends the panel.

    Modeled as the mechanism and given the package and activity to act on, rather than
    knowing them: Tap carries a coordinate instead of a purpose for the same reason. The
    values come from Config (`app_package`, `app_activity`), so the PGSharp build being a
    mod of the stock package today - exactly the kind of fact that can change under us -
    is a config edit rather than a code edit.

    `settle_ms` is the pause between the stop and the start, spent inside that one shell
    command (see actions.Actuator.render): `am start` fired immediately after `am
    force-stop` races the process teardown and can land on a package that is still going
    away. It must stay well under `actions.ADB_TIMEOUT` (5s), which bounds the whole
    invocation - a settle longer than that turns every restart into a timed-out command
    and trips the actuator's own failure breaker.
    """

    package: str
    activity: str
    reason: str
    settle_ms: int = 1000
    budget: str = "restart"


@dataclass(frozen=True)
class Pinch:
    """A real two-finger pinch, which `adb shell input` cannot express.

    Zooming the map is the one thing this bot needs that a single pointer cannot do:
    `input swipe` and `input motionevent` each take one x/y, and Pokemon GO's map zoom is a
    pinch. The one-finger double-tap-drag the game documents for humans was measured on the
    device at every distance, duration and anchor and never once changed the map's scale
    when injected; so were `input touchscreen swipe`, two concurrent swipes,
    `KEYCODE_ZOOM_OUT` and a trackball roll. Writing the events directly is refused -
    `sendevent` on /dev/input/eventN is Permission denied under SELinux even though shell
    sits in group `input`, and this is a `user` build with no `su` and `adb root` refused.

    So this goes through the framework instead, the way scrcpy does: `app_process` as the
    shell uid, calling `InputManager.injectInputEvent` with a genuine two-pointer
    MotionEvent (see tools/pinch/). The shell uid already holds INJECT_EVENTS, which is
    what makes scrcpy able to drive the phone at all.

    Gaps are fractions of screen HEIGHT, like every other distance in this system, so the
    gesture survives any capture resolution. `start_gap > end_gap` brings the fingers
    together, which zooms OUT.
    """

    x: float
    y: float
    start_gap: float
    end_gap: float
    reason: str
    steps: int = 25
    duration_ms: int = 700
    budget: str = "zoom"


@dataclass(frozen=True)
class ForegroundApp:
    """Bring the game back to the front without killing it.

    Pokemon GO carries sponsored stops and gyms, and a tap on one opens the sponsor's site
    in a browser: measured live, `ActivityTaskManager: START ... act=VIEW dat=https://
    www.mlb.com ... cmp=com.android.chrome` two seconds after the bot entered ROCKET and
    tapped the fixed dialogue coordinate. The game keeps running; it is simply no longer
    what the screen shows.

    Everything the recovery ladder does is wrong in that state. BACK navigates the BROWSER,
    the optical locators are reading a web page, and the map can never come back while
    another app owns the display - which is how one live run spent 603 frames in ROCKET on
    a cookie banner and then died on the frame guard, because a near-static web page barely
    encodes any frames either.

    Distinct from `RestartApp` on purpose and ordered before it: this is the cheap answer
    that keeps the session, the login and the AutoWalk route intact, where a restart throws
    all three away and costs a cold start.
    """

    package: str
    activity: str
    reason: str
    budget: str = "foreground"


@dataclass(frozen=True)
class Transition:
    to: BotState
    outcome: IntentOutcome
    reason: str


@dataclass(frozen=True)
class SetIntent:
    intent: object


@dataclass(frozen=True)
class Cooldown:
    """Block a normalized screen position for `seconds`."""

    x: float
    y: float
    seconds: float
    reason: str


@dataclass(frozen=True)
class ClearSpatialMemory:
    """Drop all cooldowns. Emitted when the camera rotates, because every remembered
    position refers to a viewport that no longer exists."""

    reason: str


@dataclass(frozen=True)
class SetFlag:
    """Set a per-visit bookkeeping flag on the Context.

    Handlers are pure, so they cannot record "I already spun the disc" themselves.
    Without this the POKESTOP confirm branch was unreachable and every stop scored
    EXPIRED - the learning path was dead code.
    """

    name: str
    value: object = True


@dataclass(frozen=True)
class Note:
    text: str
    level: str = "info"


@dataclass(frozen=True)
class Halt:
    reason: str


Effect = Union[Tap, Swipe, Back, DoubleTapDrag, Pinch, RestartApp, ForegroundApp,
               Transition, SetIntent,
               SetFlag, Cooldown, ClearSpatialMemory, Note, Halt]

ACTUATIONS = (Tap, Swipe, Back, DoubleTapDrag, Pinch, RestartApp, ForegroundApp)


def is_actuation(effect: Effect) -> bool:
    return isinstance(effect, ACTUATIONS)
