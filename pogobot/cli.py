"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import BASE_DIR, Config

#: `--reset-spins` with no value means "every account" - the same thing a bare
#: `--reset-spins` has always meant. An object identity sentinel rather than a string like
#: "__all__" so it can never collide with a real account name read off the overlay.
RESET_ALL = object()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("pogobot", description="Pokemon GO vision bot")
    p.add_argument("--det-model", type=Path, default=None)
    p.add_argument("--cls-model", type=Path, default=None)
    p.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    p.add_argument("--confidence", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--infer-fps", type=float, default=None)
    p.add_argument("--max-size", type=int, default=None)
    p.add_argument("--range-scale", type=float, default=None)
    p.add_argument("--catch-mode", choices=["throw", "flee", "manual"], default=None)
    p.add_argument("--target-mode", choices=["all", "pokemon", "pokestop"], default=None)
    p.add_argument("--no-rockets", action="store_true", help="skip Team GO Rocket stops")
    p.add_argument("--max-throws", type=int, default=None, metavar="N",
                   help="give up on an encounter after N throws change nothing "
                        "(out of balls, or an uncatchable Pokemon)")
    p.add_argument("--restock-after", type=int, default=None, metavar="N",
                   help="switch to PokeStop-only targeting after N useless encounters in a row")
    p.add_argument("--restock-stops", type=int, default=None, metavar="N",
                   help="PokeStops to collect before resuming normal targeting")
    p.add_argument("--max-app-restarts", type=int, default=None, metavar="N",
                   help="consecutive times recovery may force-stop and relaunch the game "
                        "before halting instead (0 never restarts it)")
    p.add_argument("--no-rotate", action="store_true")
    p.add_argument("--no-preflight", action="store_true",
                   help="skip the startup pass that zooms out, turns Virtual Go Plus back "
                        "on and starts AutoWalk once the map is first confirmed")
    p.add_argument("--dry-run", action="store_true",
                   help="perceive and decide, but never touch the device")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--tui", action="store_true",
                   help="live terminal dashboard instead of scrolling log lines")
    p.add_argument("--collect-encounters", type=Path, default=None,
                   help="write the frames around each encounter ending here, for labelling "
                        "a catch detector")
    p.add_argument("--collect-dialogues", type=Path, default=None,
                   help="save post-login screens here, for labelling a Dialogue class")
    p.add_argument("--no-learning", action="store_true", help="do not write training data")
    p.add_argument("--replay", type=Path, default=None,
                   help="run against a directory of frames instead of a phone")
    p.add_argument("--replay-interval", type=float, default=0.0)
    p.add_argument("--serial", default=None, help="adb device serial")
    p.add_argument("--trace", type=Path, default=BASE_DIR / "logs" / "trace.jsonl")
    p.add_argument("--no-trace", action="store_true")
    p.add_argument("--stats-file", type=Path, default=BASE_DIR / "logs" / "sessions.jsonl",
                   help="append each finished session here and report lifetime totals")
    p.add_argument("--no-stats", action="store_true", help="do not record session stats")
    p.add_argument("--quota-file", type=Path, default=BASE_DIR / "logs" / "spins.jsonl",
                   help="rolling 24h PokeStop spin log (spans restarts)")
    p.add_argument("--spin-limit", type=int, default=None, metavar="N",
                   help="PokeStop spins allowed per rolling 24h (0 disables the check)")
    p.add_argument("--account", default=None, metavar="NAME",
                   help="account this run belongs to. Read from the PGSharp overlay "
                        "instead when account switching is on, and the overlay wins if "
                        "the two disagree")
    p.add_argument("--switch-on-quota", action="store_true",
                   help="log into another account when this one exhausts its 24h spin cap")
    p.add_argument("--switch-every", type=float, default=None, metavar="MINUTES",
                   help="rotate accounts every MINUTES regardless of state")
    p.add_argument("--pause-file", type=Path, default=BASE_DIR / "logs" / "PAUSE",
                   help="while this file exists the bot perceives but sends no input; "
                        "also toggled by SIGUSR1, or the p key on the preview window")
    p.add_argument("--reset-spins", nargs="?", const=RESET_ALL, default=None,
                   metavar="ACCOUNT",
                   help="clear the 24h spin window, e.g. once a soft ban has lifted; "
                        "omit ACCOUNT to clear every account's window")
    p.add_argument("--seed-spins", type=int, default=None, metavar="N",
                   help="record N spins the bot did not perform, spread over the last 12h, "
                        "so the quota reflects the account rather than this process "
                        "(targets the identified account, or --account)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def switching_enabled(cfg: Config) -> bool:
    """Whether either account-switch trigger is armed."""
    return bool(cfg.switch_on_quota or cfg.switch_every_minutes > 0)


def _pause_file_present(pause_file) -> bool:
    try:
        return pause_file is not None and pause_file.exists()
    except OSError:
        return False


#: Times `prepare_accounts` will try to read the PGSharp account panel before giving up.
#:
#: One read used to be the whole of it, and a single failure disabled account switching for
#: an entire run with nothing said afterwards: logs/sessions.jsonl recorded `ended 08-20
#: 18:59:47 account=None uptime=4h09m28s`, and because `Runner.choose_next_account` refuses
#: to act without a known origin inside a known roster, the run never attempted a switch -
#: 115,210 frames, zero SWITCHING - so the zoom-out, Virtual Go Plus and AutoWalk steps that
#: live inside a switch never ran either. Run by hand on the same phone minutes later the
#: identical read worked ("logged in as NickStanki (L62), 2 account(s) available", 14.0s),
#: which is what makes the failure SITUATIONAL - most likely the game or the PGSharp overlay
#: was not ready yet at the moment the bot happened to look - and a situational failure is
#: exactly the kind a second look fixes.
#:
#: Three, on the same reasoning `Config.max_app_restarts`, `runner.SWITCH_MAX_FAILURES` and
#: `StarSeparation.max_drags` all state: the first is the ordinary attempt, the second is
#: the one the "not ready yet" hypothesis predicts will succeed, and the third is already
#: re-testing a hypothesis two attempts have refuted. Past that the cause is not timing, and
#: more looks only delay a run that is going to have to be restarted anyway.
IDENTIFY_ATTEMPTS = 3

#: Seconds between those attempts.
#:
#: The gap is not the whole wait - each attempt is itself a multi-second probe of the very
#: thing being waited for: a successful identification was measured at 14.0s end to end, and
#: a failing one still spends up to `accounts.UiTreeReader.timeout` (5s) inside its first
#: blocking dump before it can report anything. So three attempts at this spacing span
#: roughly half a minute of real wall clock, which is the order of a cold app start (see
#: `Timings.app_restart_grace`, sized for "tens of seconds" of exactly that).
#:
#: Real headroom rather than a tight bound, and the asymmetry is why: waiting too long costs
#: a few seconds of startup on a run that lasts hours, while waiting too little costs the
#: whole run's switching - and with it the three post-login steps - with no further attempt
#: and, before this, no further warning either.
IDENTIFY_RETRY_WAIT = 10.0


def prepare_accounts(cfg: Config, *, requested, pause_file, make_reader, actuator,
                     settle: float = 1.0, attempts: int = IDENTIFY_ATTEMPTS,
                     retry_wait: float = IDENTIFY_RETRY_WAIT):
    """Decide who this run belongs to, and hand back what a switch will need.

    Returns `(tree_reader, account, roster)`; a `None` reader means switching is not
    available this run, which `Runner` already treats as "never switch".

    Identifying costs three taps INTO the account panel, and every row there carries a
    delete button ~157px from its login button. That is a fine price for a run that is
    going to drive that panel anyway, and no price a run that will never switch should
    pay - so it happens only when a trigger is armed. Without switching, `--account NAME`
    is how a run gets attributed, and without that the account is simply unknown and the
    spins go to the unattributed bucket, exactly as they did before switching existed.

    It is also skipped while the pause file is present. Those taps go through the actuator
    directly, not through `Runner.apply`, so nothing else in the system would honour the
    pause - and the README's promise is unconditional: while that file exists the bot
    perceives but sends no input.

    When the overlay does answer and contradicts `--account`, the overlay wins. It is the
    only party that can see who is logged in; `--account` is a claim made before the
    process started and goes stale the moment anyone touches the phone. Believing it
    against the evidence books every spin to the wrong account, under-counts the real
    one's 24h window - so the bot spins past a cap it cannot see - and starts the
    round-robin from the wrong origin. `quota._normalize_account` raises rather than guess
    for the same reason; here there is a right answer, so it is used, loudly.

    The read is RETRIED, up to `attempts` times `retry_wait` apart, and its failure is
    announced rather than logged in passing - see IDENTIFY_ATTEMPTS above for the run that
    made both necessary. Nothing is invented when it still fails: no account name is
    guessed and no roster is fabricated, because either would be worse than knowing nothing.
    A guessed name books every spin to an account that may be capped, and an invented roster
    would have `Runner.choose_next_account` name a login target that PGSharp's own panel
    would not find - the tap goes nowhere and the attempt burns `Timings.switch_timeout`.
    """
    log = logging.getLogger("pogobot")
    if not switching_enabled(cfg):
        if not requested:
            log.info("account switching is off, so the PGSharp account panel is left "
                     "alone; spins are recorded unattributed (pass --account NAME to "
                     "book them to an account)")
        # A reader is still handed back when anything else needs to SEE the view tree,
        # even though nothing here will open the panel to identify anyone. Two things do,
        # and both were silently dead on a default invocation because this returned None:
        # `fsm.Preflight._autowalk_open` needs the star's bounds, so AutoWalk - one of the
        # three steps the preflight exists to perform - was skipped on every run without a
        # switch trigger; and `Runner._refresh_accounts` needs a reader before
        # `fsm.Recovering._panel_close` can ever see the PGSharp accounts panel, so the
        # recovery from that panel could not fire either. Constructing the reader is free
        # (it holds a serial and a screen size); what costs is a `read()`, which only
        # those two callers make, on their own throttles.
        return (make_reader() if cfg.preflight else None), requested, ()
    if _pause_file_present(pause_file):
        log.warning("paused at startup (%s exists), so the account panel is not opened "
                    "and account switching is unavailable this run - a switch has no "
                    "roster to pick a target from. Remove the file and restart to "
                    "enable it.", pause_file)
        return None, requested, ()

    from .accounts import identify_account
    # Open the account panel, read who is active, close it again - the one-shot
    # equivalent of what SWITCHING already does one tap at a time. Not required to
    # start: a failed identification just means per-account tracking (the spin quota,
    # session stats, legacy attribution) falls back to the unattributed bucket for this
    # run. `identify_account` logs its own outcome.
    reader = make_reader()
    panel = None
    spent = 0
    dry_run = bool(getattr(actuator, "dry_run", False))
    attempts = max(1, attempts)      # one look is the floor; zero would read nothing at all
    for attempt in range(1, attempts + 1):
        spent = attempt
        panel = identify_account(reader, actuator, settle=settle)
        # An ACTIVE account is what "read it" means here, not merely a panel that answered.
        # Both halves of what a switch needs come from the asterisk: it is the origin
        # `Runner.choose_next_account` rotates from, and without it a roster - even a full
        # one - names no starting point, so the whole feature stays inert exactly as it did
        # on the four-hour run. Retrying a panel that opened on nobody is also the case
        # `Timings.switch_clear_max` records as reachable in the field (PGSharp showing
        # NEITHER account with an asterisk), and one more look costs a few seconds.
        if panel is not None and panel.active is not None:
            if attempt > 1:
                log.info("read the PGSharp account panel on attempt %d of %d",
                         attempt, attempts)
            break
        if dry_run and (panel is None or not panel.panel_open):
            # A dry run suppresses the launcher tap in `Actuator.apply`, so the panel was
            # never opened and no number of further looks can open it. Retrying here would
            # spend the whole budget - `IDENTIFY_RETRY_WAIT` of real `time.sleep` between
            # multi-second blocking dumps - to reach a give-up line that then blames the
            # phone for a suppression this process chose. The retry is for a SITUATIONAL
            # failure; this one is structural.
            break
        if attempt < attempts:
            log.warning("could not read who is logged in (attempt %d of %d); retrying in "
                        "%.0fs - the game or the PGSharp overlay may still be starting up",
                        attempt, attempts, retry_wait)
            if retry_wait:
                time.sleep(retry_wait)
    # The roster is cached from this one read and never re-enumerated: the panel is
    # closed for the rest of the run, so a live read would list no accounts at all.
    # An account added to PGSharp mid-run is therefore not noticed until a restart.
    found = panel.active.name if panel is not None and panel.active else None
    roster = panel.names if panel is not None else ()
    if found is None:
        # Loud, and in the same shape as the paused-at-startup warning above, because it
        # has the same consequence and used to be the only one of the two that was silent
        # about it. What follows from an unknown account is not "slightly worse tracking":
        # `Runner.choose_next_account` returns None for a session with no origin, so NO
        # switch is ever attempted, and the zoom-out, Virtual Go Plus and AutoWalk steps
        # that live inside a switch never run either. Measured: 4h09m, account=None,
        # 115,210 frames, zero SWITCHING, and the operator reasonably assumed a switch had
        # happened. The startup preflight (fsm.Preflight) now covers those three steps
        # once at startup regardless, which is why this says switching rather than
        # everything.
        #
        # The stated CAUSE has to match the run that actually happened: under --dry-run the
        # launcher tap was suppressed by this process, so telling an operator to "restart
        # with both on screen" sends them to look at a phone that is behaving perfectly.
        cause = ("--dry-run suppresses the tap that opens the panel, so it could not be "
                 "read at all; this is expected in a dry run."
                 if dry_run else
                 "This is usually the game or the PGSharp overlay not being up yet when "
                 "the bot started - restart the bot with both on screen.")
        log.warning("ACCOUNT SWITCHING IS DISABLED for this run: the PGSharp account panel "
                    "did not name an active account in %d attempt(s), so there is no "
                    "roster and no account to rotate from, and no switch will be "
                    "attempted. %s %s",
                    spent, cause,
                    f"Spins are booked to --account {requested} meanwhile." if requested
                    else "Spins meanwhile go to the unattributed bucket; --account NAME "
                         "books them to an account.")
    if found and requested and requested != found:
        log.warning("--account %s contradicts the PGSharp overlay, which says %s is "
                    "logged in; going with %s. The overlay is the only thing that can "
                    "see who is actually signed in - drop --account, or pass the name "
                    "the phone is really on.", requested, found, found)
    account = found or requested
    if found is None and account:
        log.info("using --account %s (the overlay did not confirm it)", account)
    return reader, account, roster


def resolve_device(name: str):
    if name != "auto":
        return name
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def config_from_args(a) -> Config:
    cfg = Config()
    overrides = {}
    for name in ("confidence", "imgsz", "infer_fps", "max_size", "range_scale",
                 "catch_mode", "target_mode"):
        v = getattr(a, name, None)
        if v is not None:
            overrides[name] = v
    for cli_name, cfg_name in (("max_throws", "max_throws_per_encounter"),
                               ("restock_after", "restock_after_failures"),
                               ("restock_stops", "restock_target_stops"),
                               ("max_app_restarts", "max_app_restarts")):
        v = getattr(a, cli_name, None)
        if v is not None:
            overrides[cfg_name] = v
    if a.det_model:
        overrides["det_model"] = a.det_model
    if a.cls_model:
        overrides["cls_model"] = a.cls_model
    overrides["dry_run"] = bool(a.dry_run)
    overrides["fight_rockets"] = not a.no_rockets
    overrides["auto_rotate"] = not a.no_rotate
    # Negated flag over a positive one, like --no-rockets and --no-rotate beside it: the
    # feature is on by default (see Config.preflight), so the only thing an invocation ever
    # needs to say is that it wants it off.
    #
    # Never under --replay. A replay exists to reproduce a recorded session, and a preflight
    # is a state the recording never had: it would take the screen on the first on-map frame
    # and, while `desired_state` gives PREFLIGHT the screen, ignore the very encounters and
    # popups the operator opened the recording to look at. There is also nothing for it to
    # do - the actuator is a NullActuator and there is no device to zoom, toggle or route -
    # so this is not a feature being withheld, it is a diagnostic being kept honest.
    overrides["preflight"] = not a.no_preflight and not getattr(a, "replay", None)
    overrides["device"] = a.device
    overrides["switch_on_quota"] = bool(a.switch_on_quota)
    if a.switch_every is not None:
        overrides["switch_every_minutes"] = a.switch_every
    return cfg.scaled(**overrides)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S",
    )
    log = logging.getLogger("pogobot")
    cfg = config_from_args(a)

    from ultralytics import YOLO
    from .actions import Actuator, NullActuator
    from .stats import SessionStats
    from .capture import ReplaySource, ScrcpySource
    from .device import KeyboardPoller, device_online, screen_size
    from .learning import IntentLedger
    from .perception import Perceptor
    from .runner import Runner

    # A locally trained larger detector wins if one is present. Only the compact model
    # is committed; an 85 MB weight file is not something every clone should download.
    if a.det_model is None:
        bigger = BASE_DIR / "models" / "v3" / "det_s" / "weights" / "best.pt"
        if bigger.exists():
            cfg = cfg.scaled(det_model=bigger)
    if not cfg.det_model.exists():
        log.error("detector not found: %s", cfg.det_model)
        return 2
    dev = resolve_device(cfg.device)
    log.info("loading detector %s on %s", cfg.det_model.name, dev)
    det = YOLO(str(cfg.det_model))
    cls = None
    if cfg.cls_model.exists():
        log.info("loading screen classifier %s", cfg.cls_model.name)
        cls = YOLO(str(cfg.cls_model))
    else:
        log.warning("no screen classifier at %s - running on optics only", cfg.cls_model)

    # Class ids always come from the loaded model. Hardcoding them is how v1 would have
    # labelled every Pokemon as a gym once the 4-class dataset shipped (gym=0).
    class_names = [det.names[i] for i in sorted(det.names)]
    log.info("detector classes: %s", class_names)

    tree_reader = None
    account = a.account
    roster: tuple[str, ...] = ()
    if a.replay:
        source = ReplaySource(a.replay, interval=a.replay_interval)
        actuator = NullActuator()
        keyboard = None
        screen_wh = (1080, 2340)
        log.info("REPLAY mode: %s (no device will be touched)", a.replay)
    else:
        if not device_online(serial=a.serial):
            log.error("no adb device. Connect the phone and authorize USB debugging.")
            return 2
        screen_wh = screen_size(serial=a.serial)
        log.info("device screen %dx%d", *screen_wh)
        source = ScrcpySource(cfg, serial=a.serial)
        actuator = Actuator(screen_wh, dry_run=cfg.dry_run, serial=a.serial)
        keyboard = KeyboardPoller(serial=a.serial).start()

        from .accounts import UiTreeReader
        tree_reader, account, roster = prepare_accounts(
            cfg, requested=a.account, pause_file=a.pause_file,
            make_reader=lambda: UiTreeReader(screen_wh, serial=a.serial),
            actuator=actuator)

    perceptor = Perceptor(cfg, det_model=det, cls_model=cls, device=dev,
                          square_cls_input=True)
    ledger = None
    if not a.no_learning and not a.replay:
        ledger = IntentLedger(BASE_DIR / "datasets" / "active_v2", class_names)

    trace = None if a.no_trace else a.trace
    if trace:
        trace.parent.mkdir(parents=True, exist_ok=True)

    stats_path = None if a.no_stats else a.stats_file
    total = None
    if stats_path is not None:
        from .stats import lifetime_line, load_lifetime
        total = load_lifetime(stats_path)
        if total:
            log.info("%s", lifetime_line(total))

    from .quota import DEFAULT_DAILY_LIMIT, SpinQuota
    quota = SpinQuota(a.quota_file,
                      limit=DEFAULT_DAILY_LIMIT if a.spin_limit is None else a.spin_limit)
    if account and quota.legacy_count:
        # Once, at startup, to whoever the tree just said is logged in - by definition the
        # account that earned records written before accounts were tracked at all.
        moved = quota.attribute_legacy(account)
        log.info("attributed %d previously unassigned spin(s) to %s", moved, account)
    if a.reset_spins is not None:
        target = None if a.reset_spins is RESET_ALL else a.reset_spins
        dropped = quota.reset(target)
        log.info("cleared %d spin(s) from the 24h window%s", dropped,
                 f" for {target}" if target else "")
    if a.seed_spins:
        if account:
            quota.seed(a.seed_spins, account=account)
            log.info("seeded %d spins into %s's 24h window", a.seed_spins, account)
        else:
            # A nameless seed would land in the "" bucket, where it silently outlives this
            # run's own tracking and can be mistaken for a real account's history later.
            log.warning("--seed-spins needs an account: pass --account NAME, or run where "
                        "the PGSharp overlay is visible")
    qstate = quota.state(account=account)
    log.log(logging.WARNING if qstate.exhausted else logging.INFO, "%s", qstate.line())

    dashboard = None
    if a.tui:
        from . import tui
        if not tui.available():
            log.warning("--tui needs the 'rich' package; falling back to log lines")
        else:
            dashboard = tui.Dashboard(SessionStats(account=account),
                                      lifetime=total if stats_path else None,
                                      quota=quota, pause_file=a.pause_file)

    runner = Runner(cfg, source, actuator, perceptor, ledger=ledger, keyboard=keyboard,
                    trace_path=trace, display=not a.no_display, stats_path=stats_path,
                    dashboard=dashboard, encounter_dump=a.collect_encounters,
                    dialogue_dump=a.collect_dialogues,
                    quota=quota, pause_file=a.pause_file, tree_reader=tree_reader,
                    roster=roster)
    if dashboard is None:
        # No dashboard means Runner kept the SessionStats it built itself; name it here
        # so the very first session's spins are booked under the identified account rather
        # than the unattributed "" bucket.
        runner.stats.account = account
        return runner.run()
    # The dashboard owns the session counters so the header can render before the first
    # tick; the runner must not create a second set.
    runner.stats = dashboard.stats
    with dashboard:
        return runner.run()


if __name__ == "__main__":
    sys.exit(main())
