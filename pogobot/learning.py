"""The only module in the package permitted to write training data.

v1's self-training loop measurably degraded the detector that fed it: 3.23 -> 2.38
detections per frame over the same 40 held-out frames after three self-retrain
generations. Every rule below exists to make one confirmed cause unrepresentable.

  * PARTIAL LABELS. `record_positive` wrote the whole frame with exactly one box. All 266
    pos_*.txt files contain exactly one line, while 37% of the objects visible in those
    frames were unlabelled at the bot's own operating confidence - so every real PokeStop
    that was not the tapped one became a background example. Here a sample carries the
    COMPLETE detection set for its frame or it is not written at all, and any object whose
    confidence lands in the ambiguity band rejects the whole frame rather than being
    guessed at or silently omitted.

  * UNVERIFIED POSITIVES. v1's SPINNING_STOP branch recorded a positive 0.8s after issuing
    the spin swipe, with no evidence the POI screen had ever opened; that branch produced
    69% of the corpus. `resolve` accepts only IntentOutcome.CONFIRMED and only when the
    confirmation lands inside a causal window - too early to be an artefact, too late to
    have been caused by our tap.

  * NEAR-DUPLICATES. 113 of the 265 consecutive positive pairs were written less than 6s
    apart, so the corpus was dominated by a handful of scenes. Writes are deduplicated by
    perceptual hash (dhash) against recent writes, including writes from previous runs
    replayed out of the ledger journal.

  * HARDCODED CLASS IDS. v1 held CLASS_NAMES = [pokemon, pokestop, pokestop_rocket] as a
    module constant while the v3 detector is 4-class [gym, pokemon, pokestop,
    pokestop_rocket] with gym=0; that map labels every Pokemon as a gym. Class ids come
    only from the names of the model actually loaded, passed in at construction, and the
    corpus records those names in its data.yaml. If a later run presents different names,
    the ledger disables itself instead of appending incompatible ids.

  * HARD NEGATIVES. v1 wrote 56 files into hard_negatives/ that nothing ever read. This
    module DOES NOT WRITE THEM, and that is the deliberate choice of the two offered. An
    empty YOLO label file is legitimate background data only for a frame that genuinely
    contains no object of any class; a refuted frame is the opposite case - it provably
    contains a real object (the gym or shop we mistapped, which the 4-class detector has a
    class for). Labelling it empty would train the detector to suppress exactly the objects
    it should be learning, which is the degradation this module exists to prevent. We do
    not know the correct label for a refuted frame, so we write nothing and let the FSM's
    spatial cooldown carry the negative signal instead.

  * DIRTY SPLITS. v1 pointed val at the train images and then wrote live samples into an
    empty val/. The live bot writes to train/ only; val/ is created and left alone so a
    held-out set stays held out.

Writes never happen on the hot loop: `resolve` decides and hands a job to a bounded
background queue, and drops rather than blocks when the writer falls behind.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from ast import literal_eval
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, Sequence

import cv2
import numpy as np

from .effects import IntentOutcome
from .frames import Frame
from .fsm import Intent
from .observation import Detection, Observation

SENTINEL = object()


@dataclass(frozen=True)
class LedgerPolicy:
    """Every bar a sample has to clear. Frozen so a tuning experiment is a new object.

    `causal_min_s` / `causal_max_s` bracket the delay between our tap and the screen
    change that confirms it. Below the floor the change was already under way when we
    tapped; above the ceiling the game moved for some other reason and the frame we would
    label proves nothing. 0.2-5.0s brackets the observed transition latency of the two
    confirmable outcomes (encounter open, POI screen up) with room for a slow load.
    """

    causal_min_s: float = 0.20
    causal_max_s: float = 5.00

    label_min_conf: float = 0.30
    ambiguity_floor: float = 0.15

    dedup_hamming: int = 6
    dedup_memory: int = 4096

    # The ring must outlive the causal window or the window is a lie: a confirmation at
    # +3s would be inside `causal_max_s` and still find its tap frame evicted. `stage` is
    # called once per inference tick, so this has to cover causal_max_s * Config.infer_fps
    # (5.0s * 8fps = 40) with headroom. Raising --infer-fps above ~9 shortens the real
    # retention; `stage` also evicts on age so the ring never exceeds the window in time.
    ring_frames: int = 48
    queue_size: int = 32

    box_match_tol: float = 0.02
    max_boxes: int = 40


@dataclass(frozen=True)
class _Staged:
    """A frame plus the detection set measured on it, held until its intent resolves."""

    seq: int
    ts: float
    bgr: np.ndarray
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class _Job:
    stem: str
    bgr: np.ndarray
    lines: tuple[str, ...]
    record: dict


@dataclass
class _Counters:
    written: int = 0
    queued: int = 0
    dropped: int = 0
    errors: int = 0
    rejected: dict = field(default_factory=dict)
    last_stem: Optional[str] = None
    last_error: Optional[str] = None


def dhash(bgr: np.ndarray, size: int = 8) -> int:
    """64-bit perceptual hash. Cheap enough (~0.1ms) to run inside `resolve`.

    Compares each pixel to its right-hand neighbour on a 9x8 grey thumbnail, so it tracks
    structure rather than exposure - two frames 4s apart on the same street corner hash
    within a few bits of each other even as the day/night tint drifts.
    """
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return int.from_bytes(np.packbits(bits.reshape(-1)).tobytes(), "big")


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _parse_names(text: str) -> Optional[list[str]]:
    """Read `names: [...]` out of a data.yaml without taking a YAML dependency.

    Only the one key matters, and mis-parsing it must fail closed: anything unexpected
    returns None, which the caller treats as an unreadable corpus and refuses to append to.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("names:"):
            continue
        value = line[len("names:"):].strip()
        if not value.startswith("["):
            return None
        try:
            parsed = literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if not isinstance(parsed, (list, tuple)) or not all(isinstance(n, str) for n in parsed):
            return None
        return [str(n) for n in parsed]
    return None


class IntentLedger:
    """Turns confirmed intents into a YOLO detection corpus, or refuses and says why.

    Construct with the names of the detector that is actually loaded
    (`Perceptor.class_names`); an empty name list disables the ledger, because a corpus
    written against a guessed class order is worse than no corpus.

    Usage per tick:  `stage(frame, obs)` while the frame is live, then `resolve(intent,
    outcome, now)` when the FSM resolves that intent. `stats()` is HUD-cheap; `close()`
    drains the writer.
    """

    def __init__(
        self,
        root: Path,
        class_names: Sequence[str],
        *,
        enabled: bool = True,
        policy: LedgerPolicy = LedgerPolicy(),
    ) -> None:
        self.root = Path(root)
        self.policy = policy
        self.class_names = tuple(str(n).lower() for n in class_names)
        self.class_ids = {n: i for i, n in enumerate(self.class_names)}

        self._lock = threading.Lock()
        self._counters = _Counters()
        self._ring: "OrderedDict[int, _Staged]" = OrderedDict()
        self._hashes: Deque[int] = deque(maxlen=policy.dedup_memory)
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=policy.queue_size)
        self._thread: Optional[threading.Thread] = None
        self._serial = 0

        self.enabled = False
        self.disabled_reason: Optional[str] = None
        if not enabled:
            self.disabled_reason = "disabled by caller"
            return
        if not self.class_names:
            self.disabled_reason = "no detector class names; refusing to guess class ids"
            return

        problem = self._bind_corpus()
        if problem is not None:
            self.disabled_reason = problem
            return

        self.enabled = True
        self._thread = threading.Thread(target=self._writer, name="pogobot-ledger", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ corpus

    @property
    def train_images(self) -> Path:
        return self.root / "train" / "images"

    @property
    def train_labels(self) -> Path:
        return self.root / "train" / "labels"

    @property
    def journal(self) -> Path:
        return self.root / "ledger.jsonl"

    def _bind_corpus(self) -> Optional[str]:
        """Attach to the on-disk corpus, or return why we must not write to it.

        The data.yaml is the corpus's record of which class order its label ids mean. v1
        had no such record, so a detector upgrade silently reinterpreted 266 existing
        labels. A mismatch here is fatal to writing, not a warning.
        """
        try:
            for d in (self.train_images, self.train_labels,
                      self.root / "val" / "images", self.root / "val" / "labels"):
                d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"cannot create corpus dirs: {exc}"

        yaml_path = self.root / "data.yaml"
        if yaml_path.exists():
            existing = _parse_names(yaml_path.read_text())
            if existing is None:
                return f"{yaml_path.name} has no readable names: list"
            if [n.lower() for n in existing] != list(self.class_names):
                return (f"corpus classes {existing} != model classes {list(self.class_names)}; "
                        f"appending would corrupt existing label ids")
        else:
            yaml_path.write_text(
                f"path: {self.root}\n"
                "train: train/images\n"
                "val: val/images\n"
                f"nc: {len(self.class_names)}\n"
                f"names: {list(self.class_names)}\n"
            )

        if self.journal.exists():
            self._load_hashes()
        return None

    def _load_hashes(self) -> None:
        """Seed dedup memory from previous runs so a restart cannot re-write the same scene."""
        try:
            lines = self.journal.read_text(errors="replace").splitlines()
        except OSError:
            return
        for line in lines[-self.policy.dedup_memory:]:
            try:
                h = json.loads(line).get("dhash")
            except (ValueError, AttributeError):
                continue
            if isinstance(h, int):
                self._hashes.append(h)

    # ------------------------------------------------------------------ staging

    def stage(self, frame: Frame, obs: Observation) -> None:
        """Hold this frame's pixels and its full detection set until its intent resolves.

        The frame that belongs in the corpus is the one we tapped on, not the one that
        confirmed the tap - the target is only visible in the former. v1 kept it by copying
        into the intent, which meant a 2MB copy on every candidate whether it was ever
        confirmed or not; a small ring keyed by frame seq costs the same per frame but is
        bounded and lets the intent stay pure data.
        """
        if not self.enabled or not obs.detections:
            return
        if obs.seq != frame.seq:
            return
        # The copy is a ~2MB memcpy and stays OUTSIDE the lock: `stats()` runs on the HUD
        # path every displayed frame and the writer thread takes the same lock, so holding
        # it across the copy would put a memcpy on both.
        staged = _Staged(
            seq=frame.seq,
            ts=frame.ts,
            bgr=frame.bgr.copy(),
            detections=tuple(obs.detections),
        )
        horizon = frame.ts - self.policy.causal_max_s
        with self._lock:
            self._ring[frame.seq] = staged
            # Evict on age as well as count: a frame older than the causal window can
            # never be claimed by a resolving intent, so holding it only costs memory.
            while len(self._ring) > 1 and (
                len(self._ring) > self.policy.ring_frames
                or next(iter(self._ring.values())).ts < horizon
            ):
                self._ring.popitem(last=False)

    # ------------------------------------------------------------------ resolve

    def resolve(self, intent: Intent, outcome: IntentOutcome, now: float) -> Optional[str]:
        """Decide whether this resolved intent earns a training sample.

        Returns None when the sample was queued, otherwise the reason it was refused.
        Every early return here is a v1 defect that reached disk 266 times.
        """
        if not self.enabled:
            return self.disabled_reason or "ledger disabled"

        if outcome is not IntentOutcome.CONFIRMED:
            return self._reject(f"outcome {outcome.value}")

        latency = now - intent.ts
        if latency < self.policy.causal_min_s:
            return self._reject("confirmed too fast to be caused by the tap")
        if latency > self.policy.causal_max_s:
            return self._reject("confirmation outside causal window")

        with self._lock:
            staged = self._ring.get(intent.frame_seq)
        if staged is None:
            return self._reject("tap frame no longer staged")

        problem = self._check_labels(staged, intent)
        if problem is not None:
            return self._reject(problem)

        digest = dhash(staged.bgr)
        with self._lock:
            recent = tuple(self._hashes)
        # Scanned outside the lock: up to `dedup_memory` python-level popcounts is long
        # enough to be worth not blocking `stage` on.
        if any(hamming(digest, h) <= self.policy.dedup_hamming for h in recent):
            return self._reject("near-duplicate of a recent write")
        with self._lock:
            self._serial += 1
            serial = self._serial

        wall = time.time()
        stem = (f"{datetime.fromtimestamp(wall).strftime('%Y%m%d_%H%M%S')}"
                f"_{staged.seq:07d}_{serial:04d}_{intent.target_name}")
        lines = tuple(self._label_line(d) for d in staged.detections
                      if d.conf >= self.policy.label_min_conf)
        job = _Job(
            stem=stem,
            bgr=staged.bgr,
            lines=lines,
            record={
                "stem": stem,
                # Wall clock, not `now`: the runner's clock is time.perf_counter(), whose
                # origin is undefined and resets every run, so it cannot date a journal
                # line or be joined against the trace log.
                "ts": round(wall, 3),
                "seq": staged.seq,
                "target": intent.target_name,
                "conf": round(intent.confidence, 4),
                "expected": intent.expected.value,
                "latency": round(latency, 3),
                "boxes": len(lines),
                "dhash": digest,
                "classes": list(self.class_names),
            },
        )
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            with self._lock:
                self._counters.dropped += 1
            return "writer queue full"
        with self._lock:
            # Remembered only once the job is in the writer's hands. Recording it before
            # the enqueue meant a job dropped on a full queue still suppressed its scene
            # as a "near-duplicate of a recent write" that was never written at all.
            self._hashes.append(digest)
            self._counters.queued += 1
        return None

    def _check_labels(self, staged: _Staged, intent: Intent) -> Optional[str]:
        """All-or-nothing label check. A frame we cannot label completely is not written.

        The ambiguity band is the specific fix for v1's measured 37% unlabelled objects: a
        box the detector is unsure about is neither trustworthy ground truth nor safely
        omitted, because omitting it asserts background. Either state poisons the corpus,
        so the frame goes in the bin instead.
        """
        if not staged.detections:
            return "no detections on the tap frame"
        if len(staged.detections) > self.policy.max_boxes:
            return "implausible box count"

        confident: list[Detection] = []
        for d in staged.detections:
            if d.conf >= self.policy.label_min_conf:
                if d.name not in self.class_ids:
                    return f"detector emitted unknown class {d.name!r}"
                confident.append(d)
            elif d.conf >= self.policy.ambiguity_floor:
                return f"ambiguous object {d.name!r} at conf {d.conf:.2f}"
        if not confident:
            return "no confident objects to label"

        tol = self.policy.box_match_tol
        tx, ty = intent.tap_norm
        if not any(abs(d.xywhn[0] - tx) <= tol and abs(d.xywhn[1] - ty) <= tol
                   for d in confident):
            return "tapped box absent from the label set"
        return None

    def _label_line(self, d: Detection) -> str:
        cx, cy, bw, bh = d.xywhn
        return f"{self.class_ids[d.name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    def _reject(self, reason: str) -> str:
        with self._lock:
            self._counters.rejected[reason] = self._counters.rejected.get(reason, 0) + 1
        return reason

    # ------------------------------------------------------------------ writer

    def _writer(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if not isinstance(job, _Job):
                    return
                self._write(job)
            except Exception as exc:                      # a bad write must not kill the thread
                with self._lock:
                    self._counters.errors += 1
                    self._counters.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._queue.task_done()

    def _write(self, job: _Job) -> None:
        """Image and label land together or the pair is removed.

        A .png with no .txt reads to YOLO as a background frame full of unlabelled objects
        - the same poison as a partial label set, arriving by a different route.

        The label goes down FIRST. The unlink below only covers an exception; a SIGKILL or
        a power cut between the two writes is not catchable, and of the two possible
        half-states a .txt with no .png is ignored by the loader while a .png with no .txt
        is the poison above. So the uncatchable interruption is aimed at the harmless one.
        """
        img = self.train_images / f"{job.stem}.png"
        lbl = self.train_labels / f"{job.stem}.txt"
        try:
            lbl.write_text("\n".join(job.lines) + "\n")
            if not cv2.imwrite(str(img), job.bgr):
                raise OSError(f"cv2.imwrite failed for {img}")
        except Exception:
            img.unlink(missing_ok=True)
            lbl.unlink(missing_ok=True)
            raise
        with open(self.journal, "a") as fh:
            fh.write(json.dumps(job.record) + "\n")
        with self._lock:
            self._counters.written += 1
            self._counters.last_stem = job.stem

    # ------------------------------------------------------------------ lifecycle

    def stats(self) -> dict:
        """HUD snapshot. v1 showed only `+N -N`, which stayed green while it wrote garbage;
        the rejection histogram is the part that tells you the loop is behaving."""
        with self._lock:
            c = self._counters
            top = sorted(c.rejected.items(), key=lambda kv: -kv[1])[:3]
            return {
                "enabled": self.enabled,
                "disabled_reason": self.disabled_reason,
                "written": c.written,
                # Errored jobs left the queue without incrementing `written`; counting
                # them as pending would park a phantom backlog on the HUD forever.
                "pending": max(c.queued - c.written - c.errors, 0),
                "dropped": c.dropped,
                "errors": c.errors,
                "rejected": sum(c.rejected.values()),
                "top_rejections": top,
                "staged": len(self._ring),
                "classes": list(self.class_names),
                "last_write": c.last_stem,
                "last_error": c.last_error,
            }

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until the queue drains. Returns False if it did not within `timeout`."""
        if not self.enabled:
            return True
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.02)
        return self._queue.unfinished_tasks == 0

    def close(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self.flush(timeout)
        self.enabled = False
        try:
            # Blocking put, not put_nowait: if the flush timed out the queue is still full,
            # and a dropped sentinel means the writer never returns and the join below just
            # burns the timeout. `enabled` is already False, so no new job can race in.
            self._queue.put(SENTINEL, timeout=max(timeout, 0.1))
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        self._thread = None
        with self._lock:
            self._ring.clear()

    def __enter__(self) -> "IntentLedger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
