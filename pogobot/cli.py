"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import BASE_DIR, Config


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
    p.add_argument("--no-rotate", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="perceive and decide, but never touch the device")
    p.add_argument("--no-display", action="store_true")
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
    p.add_argument("-v", "--verbose", action="store_true")
    return p


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
    if a.det_model:
        overrides["det_model"] = a.det_model
    if a.cls_model:
        overrides["cls_model"] = a.cls_model
    overrides["dry_run"] = bool(a.dry_run)
    overrides["fight_rockets"] = not a.no_rockets
    overrides["auto_rotate"] = not a.no_rotate
    overrides["device"] = a.device
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

    perceptor = Perceptor(cfg, det_model=det, cls_model=cls, device=dev,
                          square_cls_input=True)
    ledger = None
    if not a.no_learning and not a.replay:
        ledger = IntentLedger(BASE_DIR / "datasets" / "active_v2", class_names)

    trace = None if a.no_trace else a.trace
    if trace:
        trace.parent.mkdir(parents=True, exist_ok=True)

    stats_path = None if a.no_stats else a.stats_file
    if stats_path is not None:
        from .stats import lifetime_line, load_lifetime
        total = load_lifetime(stats_path)
        if total:
            log.info("%s", lifetime_line(total))

    runner = Runner(cfg, source, actuator, perceptor, ledger=ledger, keyboard=keyboard,
                    trace_path=trace, display=not a.no_display, stats_path=stats_path)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
