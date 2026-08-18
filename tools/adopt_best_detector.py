#!/usr/bin/env python3
"""Compare trained detectors on the held-out val set and install the winner.

Ranks by class-agnostic localization recall at IoU 0.5, not by mAP. Two reasons:
candidate models may have different class counts (the shipped 3-class model vs the
4-class v3 set), which makes per-class mAP incomparable; and what the bot actually needs
is "did it find the object at all" - the FSM re-checks the class itself.

  python3 tools/adopt_best_detector.py            # compare only
  python3 tools/adopt_best_detector.py --install  # copy the winner into models/v3/det
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent.parent
VAL = BASE / "datasets" / "det_v3" / "valid"
TARGET = BASE / "models" / "v3" / "det" / "weights" / "best.pt"
CANDIDATES = [
    BASE / "models" / "v3" / "det" / "weights" / "best.pt",
    BASE / "models" / "v3" / "det_s" / "weights" / "best.pt",
    BASE / "models" / "pokemongo_yolov8n.pt",
]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def score(model_path: Path, conf: float, imgsz: int) -> tuple[float, int, int]:
    from ultralytics import YOLO
    m = YOLO(str(model_path))
    hit = tot = 0
    for ip in sorted((VAL / "images").glob("*.png")):
        im = cv2.imread(str(ip))
        h, w = im.shape[:2]
        lp = VAL / "labels" / f"{ip.stem}.txt"
        gt = []
        for line in lp.read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
            gt.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        r = m.predict(source=str(ip), conf=conf, imgsz=imgsz, device="mps", verbose=False)[0]
        pred = [tuple(b.xyxy[0].tolist()) for b in (r.boxes or [])]
        for g in gt:
            tot += 1
            if any(iou(g, p) >= 0.5 for p in pred):
                hit += 1
    return (hit / max(1, tot)), hit, tot


def scrub_paths(weights: Path) -> list:
    """Remove absolute build paths from a checkpoint before it can be committed.

    Ultralytics records the training `data` and `project` arguments verbatim, so a
    checkpoint trained locally carries the author's home directory inside it. That is
    invisible to grep because it lives in a pickle, and `torch.load` hands it to anyone
    who downloads the file. TARGET is a tracked path, so scrubbing has to happen here or
    a future --install silently republishes it.
    """
    import torch
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    changed = []
    for field in ("train_args", "args"):
        a = ck.get(field)
        if not isinstance(a, dict):
            continue
        for k, v in list(a.items()):
            if isinstance(v, str) and ("/Users/" in v or "/home/" in v):
                a[k] = Path(v).name
                changed.append(f"{field}.{k}")
    if "git" in ck:
        ck["git"] = None
    if changed:
        torch.save(ck, weights)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=1024)
    a = ap.parse_args()

    if not VAL.exists():
        print(f"no validation set at {VAL}")
        return 2

    results = []
    for c in CANDIDATES:
        if not c.exists():
            print(f"  {c.relative_to(BASE)}: not present, skipping")
            continue
        rec, hit, tot = score(c, a.conf, a.imgsz)
        results.append((rec, c))
        print(f"  {str(c.relative_to(BASE)):44s} recall={hit:3d}/{tot} ({100*rec:5.1f}%)")

    if not results:
        print("nothing to compare")
        return 1
    results.sort(reverse=True, key=lambda t: t[0])
    best_rec, best = results[0]
    print(f"\nwinner: {best.relative_to(BASE)} at {100*best_rec:.1f}%")

    if not a.install:
        print("re-run with --install to copy it into models/v3/det/weights/best.pt")
        return 0
    if best.resolve() == TARGET.resolve():
        print("the winner is already installed")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        shutil.copy2(TARGET, TARGET.with_suffix(".pt.bak"))
    shutil.copy2(best, TARGET)
    scrubbed = scrub_paths(TARGET)
    print(f"installed -> {TARGET.relative_to(BASE)} (previous kept as best.pt.bak)")
    print(f"scrubbed build paths from the installed copy: {scrubbed or 'nothing to scrub'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
