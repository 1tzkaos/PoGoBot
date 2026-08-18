# PoGoBot Rework — Design

**Date:** 2026-08-17
**Status:** Approved
**Supersedes:** the monolithic `pokemon_vision_bot.py` (kept runnable, untouched, until parity)

## Problem

A 7-lens adversarial audit of `pokemon_vision_bot.py` confirmed **106 defects**. They dedupe to ~65,
of which **39 are instances of 6 missing structures**. Three self-sustaining failure chains dominate:

**Chain A — menu loop.** `find_close_button_coordinates` has no "not found" path and falls back to
`(0.50w, 0.8808h)` = `(540, 2061)`, which lies *inside* the ROI the same file uses to identify the
overworld main-menu Pokeball (`x 475..604, y 1989..2129`). "Close the popup" therefore *opens the main
menu*. `CLOSING_POPUP`'s 4.0s escape is an `elif` chained below the tap branch, so it is unreachable
while that branch holds. Result: menu opens/closes ~1x/sec forever, detector gated off, nothing logged.

**Chain B — poison loop.** `SPINNING_STOP` writes a `+1.0` training sample 0.8s after swiping with zero
verification a PokeStop opened (69% of the corpus). Each writes a *single-box* label on a multi-object
frame. Measured: 3.23 -> 2.38 detections/frame across 40 frames after three self-retrain generations;
107/165 ingested labels are zero-byte; 40/52 `merged_pokemongo` val images are byte-identical to train.

**Chain C — blind loop.** `read()` returns `(True, last_frame.copy())` forever after stream death and
`scrcpy_proc` is never `poll()`ed, converting the bot into ~3600 blind taps/hour at (540, 2061).

**Root cause found separately:** ultralytics classification inference is `Resize(224) -> CenterCrop(224)`.
On a 19.5:9 frame that keeps only the middle 46% of height (y in 0.268..0.730). The screen classifier the
entire FSM trusts is physically blind to the bottom Pokeball/X button and the top flee icon.

## Architecture

```
pogobot/
  config.py       frozen dataclass: every threshold + timing constant
  frames.py       Frame(seq, ts, bgr) + FrameSource protocol
  capture.py      ScrcpySource (keeps drop-to-latest) | ReplaySource
  perception.py   PURE: frame -> Observation
  observation.py  Observation, Signal, Smoother (N-of-M), confidence policy
  actions.py      Actuator - the only code that touches adb
  fsm.py          PURE: (Observation, Context) -> list[Effect]
  learning.py     IntentLedger - the only code that writes training data
  hud.py          one render()
  runner.py       the only module holding both a FrameSource and an Actuator
  cli.py
```

### Three chokepoints

1. **`enter_state()`** is the only writer of `state`. Stamps `state_start_time`, demands an explicit
   `IntentOutcome` (`CONFIRMED|REFUTED|EXPIRED`), appends the matching cooldown.
2. **`Actuator`** owns `dry_run`, per-action rate limits, adb return codes, circuit breaker. A new call
   site cannot forget the guard.
3. **`IntentLedger.resolve()`** is the only writer of training data. Requires a causal time window, a
   confirming observation, and a dedup check; writes **every box in the frame or none**.

### FSM

Per-state handler objects. The runner checks `now - state_entered > handler.timeout_s` and calls
`on_timeout()` **before** dispatching the handler body, making unreachable-timeout impossible by
construction. Every handler must declare `timeout_s`, `on_timeout`, and `default`; a missing one is a
startup error. Handlers return `list[Effect]`; the runner applies dry-run, rate limits, and tracing in
exactly one place.

States: `BOOT, SCANNING, TARGETING, ENCOUNTER, POKESTOP, ROCKET, POPUP, RECOVERING, HALTED`.

Rocket handling uses the in-game auto-battler: `GruntDialogue` -> tap through; `GruntBattleButton` ->
tap Battle; `ChooseParty` -> confirm party; battle plays itself; `ExitTrainerBattle` -> tap through.
No combat vision required.

### Perception

All thresholds are **fractions of ROI area**, never absolute pixel counts (at `--max-size 720` the old
`orange_bino > 500` measured 479 and overworld detection failed silently). Raw scores ride on the
`Observation` so thresholds are tunable and diagnosable. Optical evidence **vetoes** the classifier
everywhere. `find_close_button_coordinates` returns `Optional` - no confirmed button, no tap.

### Class vocabulary

`CLASS_NAMES` is derived from `model.names` at load time, never hardcoded. The new detector dataset is
4-class `['gym','pokemon','pokestop','pokestop_rocket']` (gym=0), so the old hardcoded `pokemon=0`
mapping would have labelled every Pokemon as a gym.

### Test seam

`perception.observe()` and `fsm.step()` import without torch, adb, cv2 windows, or a phone.
378 on-disk frames become golden perception tests; scripted `Observation` lists become FSM tests.
`ReplaySource` + `NullActuator` give end-to-end runs with no device.

## Models

New clean datasets (verified: zero cross-split leakage, zero intra-train dupes, zero empty labels,
8.04 boxes/image):
- detector: `datasets/det_v3`, 186/21/11, 4 classes -> retrain yolov8n @ imgsz 1024
- classifier: `datasets/state_v3`, 174/40/21, 17 classes -> A/B baseline vs stretch-to-square 224
  to fix the CenterCrop blindness; the winner ships and perception squares frames identically.

Old poisoned buffer `datasets/active_feedback` is quarantined, not deleted.

## Explicitly not changing

Drop-to-latest capture design; scrcpy->FIFO pipeline; the reach-ellipse target model; the two-model
architecture; HSV/ROI heuristics as a category. No multi-device, async, plugin system, config file
format, or database.
