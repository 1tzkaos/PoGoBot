#!/usr/bin/env python3
"""Promote human-reviewed frames from the review queue into a training split.

The bot never writes training data directly. `pogobot/learning.py` curates frames into
`datasets/active_v2/review/` with the detector's own predictions as a starting point,
marked `verified: false` in `ledger.jsonl`. Training on those unverified labels is
self-training, and self-training is what degraded the v1 detector (3.23 -> 2.38
detections per frame over three self-retrain generations).

Workflow:
  1. Run the bot. Frames land in datasets/active_v2/review/{images,labels}/.
  2. Fix the labels by hand (Roboflow, labelImg, whatever) - this is the step that
     actually adds information the model does not already have.
  3. Run this script to move the reviewed pairs into a dated training split.

  python3 tools/promote_reviewed.py --list          # what is waiting, worst first
  python3 tools/promote_reviewed.py --promote       # move reviewed pairs into train/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
REVIEW = BASE / "datasets" / "active_v2"
PRIORITY = {"refuted": 0, "ambiguous": 1, "routine": 2}


def manifest() -> dict:
    out = {}
    j = REVIEW / "ledger.jsonl"
    if j.exists():
        for line in j.read_text().splitlines():
            try:
                rec = json.loads(line)
                out[rec["stem"]] = rec
            except Exception:
                continue
    return out


def pairs():
    imgs = REVIEW / "review" / "images"
    lbls = REVIEW / "review" / "labels"
    if not imgs.exists():
        return []
    return [(p, lbls / f"{p.stem}.txt") for p in sorted(imgs.glob("*.png"))
            if (lbls / f"{p.stem}.txt").exists()]


def cmd_list() -> int:
    m = manifest()
    rows = []
    for img, lbl in pairs():
        rec = m.get(img.stem, {})
        rows.append((PRIORITY.get(rec.get("review_priority", "routine"), 3), img.stem,
                     rec.get("review_priority", "?"), rec.get("outcome", "?"),
                     len([x for x in lbl.read_text().splitlines() if x.strip()]),
                     rec.get("ambiguous", [])))
    rows.sort()
    if not rows:
        print("review queue is empty")
        return 0
    print(f"{len(rows)} frames awaiting review (label these first):\n")
    print(f"{'priority':11s} {'outcome':10s} {'boxes':>5s}  {'ambiguous confs':22s} stem")
    for _, stem, prio, outcome, boxes, amb in rows:
        print(f"{prio:11s} {outcome:10s} {boxes:5d}  {str(amb)[:22]:22s} {stem}")
    print("\nrefuted  = the bot tapped this and was wrong; the label set is probably missing "
          "or mislabelling the thing it actually hit")
    print("ambiguous = the detector was unsure about an object here; the most informative "
          "frames to correct")
    return 0


def cmd_promote(dest: Path, yes: bool) -> int:
    ps = pairs()
    if not ps:
        print("nothing to promote")
        return 0
    print(f"About to move {len(ps)} image/label pairs into {dest}")
    print("Only do this AFTER you have corrected the labels by hand.")
    if not yes:
        print("Re-run with --yes to actually move them.")
        return 0
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)
    for img, lbl in ps:
        shutil.move(str(img), dest / "images" / img.name)
        shutil.move(str(lbl), dest / "labels" / lbl.name)
    print(f"moved {len(ps)} pairs")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the queue, worst first")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--dest", type=Path, default=BASE / "datasets" / "det_v3" / "train")
    ap.add_argument("--yes", action="store_true", help="actually move the files")
    a = ap.parse_args()
    if a.promote:
        return cmd_promote(a.dest, a.yes)
    return cmd_list()


if __name__ == "__main__":
    raise SystemExit(main())
