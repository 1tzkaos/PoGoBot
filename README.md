# PoGoBot

A Pokémon GO vision bot: scrcpy video → YOLOv8 detector + screen classifier → a pure,
testable state machine → adb taps.

```bash
python3 -m pogobot                 # run it
python3 -m pogobot --dry-run       # perceive and decide, never touch the phone
python3 -m pogobot --replay <dir>  # run against saved frames, no phone at all
python3 -m pytest tests/ -q        # 35 tests, no device needed, ~4s
```

## Why this exists

A 7-lens adversarial audit of the previous single-file bot (`pokemon_vision_bot.py`, kept
for reference) confirmed **106 defects**. They deduplicated to ~65, of which 39 were
instances of six missing structures. Three of them were self-sustaining loops that
consumed an entire unattended session:

| | what happened |
|---|---|
| **Menu loop** | `find_close_button_coordinates` had no "not found" path and fell back to `(0.50w, 0.8808h)` = `(540, 2061)` — a coordinate that sits *inside* the ROI the same file used to identify the overworld Pokéball. "Close the popup" **opened the main menu**. `CLOSING_POPUP`'s 4s escape was an `elif` below the tap branch, so it was unreachable while that branch held. |
| **Poison loop** | `SPINNING_STOP` wrote a `+1.0` training sample 0.8s after swiping with no verification a PokéStop had opened (69% of the corpus), each with a *single* box label on a multi-object frame. Measured: **3.23 → 2.38 detections/frame** after three self-retrain generations. |
| **Blind loop** | `read()` returned `(True, last_frame.copy())` forever after the stream died, and `scrcpy_proc` was never polled. A cable bump turned the bot into ~3600 blind taps/hour against a phone it could not see. |

## Architecture

```
pogobot/
  config.py       every threshold and timing constant, one frozen dataclass
  frames.py       Frame(seq, ts, bgr) — staleness is expressible
  capture.py      ScrcpySource (drop-to-latest) | ReplaySource
  perception.py   PURE: frame -> Observation
  observation.py  Observation, Signal, ScreenStabilizer
  actions.py      Actuator — the only code that touches adb
  fsm.py          PURE: (Observation, Context) -> list[Effect]
  learning.py     IntentLedger — the only code that writes data
  runner.py       the only module holding both a FrameSource and an Actuator
```

**Three chokepoints**, because 39 of 65 defects were one missing chokepoint each:

1. `enter_state()` is the only writer of `state`. It stamps the clock, demands an explicit
   `IntentOutcome`, and appends the cooldown. v1 assigned `state` at 12 sites and stamped
   the clock at 9.
2. `Actuator` enforces `dry_run`, rate limits, adb return codes and a circuit breaker once.
   v1 checked `no_click` at 5 of 10 actuation sites.
3. `IntentLedger` is the only writer of training data.

**Timeouts are checked by the dispatcher before the handler body runs**, so an unreachable
timeout is impossible. Every handler must declare `timeout_s` and `on_timeout`; a missing
one raises at import, not at 3am.

**All effect coordinates are normalized floats.** v1 mixed stream pixels and device pixels
freely — cooldowns stored device pixels while detections were in stream pixels.

## Perception: measured, not assumed

Every threshold is a **fraction of its ROI area**. v1 used absolute pixel counts on ROIs
that scale with `--max-size`; at `--max-size 720` the binoculars check measured 479 px
against a `> 500` bar and overworld detection silently stopped working.

Calibrated against 235 labelled frames:

| signal | behaviour |
|---|---|
| optical map (red Pokéball **and** orange binoculars) | 79% recall, ~0% false positive → used as a **precision-first veto** |
| red Pokéball alone | 97% recall but 18% false positive on encounters → not used alone |
| `find_close_button` | 93% on menu screens, 1% on encounters |
| `find_action_pill` | 100% on the Rocket BATTLE / USE THIS PARTY screens, 4% on menus |
| optical encounter (ball + flee icon) | **27% false positive, 30% recall — deleted.** It decides nothing; the classifier owns that call behind the map and X-button vetoes |

Buttons are **located, never assumed**. Every finder returns `Optional`; no located button
means no tap. That alone breaks the menu loop.

Over 321 real frames the classifier agrees with the optical map signal 97.9% of the time
and is confidently wrong (≥0.90) on 0.5% — all covered by the veto. An N-of-M stabilizer
means no single frame can move the machine.

## Team GO Rocket

Uses the in-game auto-battler, so no combat vision is needed: tap through `GruntDialogue`,
press the located BATTLE pill, confirm the party, let the battle play itself, tap through
`ExitTrainerBattle`. The Rocket rule outranks the popup rule — those screens carry an X
button, so a popup-first ordering would close the grunt dialogue instead of fighting it.

## Learning: a review queue, not a training split

The bot **never writes training data**. It curates frames into
`datasets/active_v2/review/` with the detector's own predictions as a *proposal*, marked
`verified: false`.

Writing model predictions into a training split is self-training, and self-training is
what degraded v1. Pseudo-labels can only reinforce what the model already believes; they
cannot teach it an object it currently misses.

```bash
python3 tools/promote_reviewed.py --list            # queue, worst first
# ... correct the labels by hand ...
python3 tools/promote_reviewed.py --promote --yes
```

Frames where the bot was **wrong** (`refuted`) and frames containing objects the detector
was **unsure** about are ranked first — those carry the information a human pass can add.

## Models

`datasets/det_v3` (186/21/11, 4 classes) — verified zero cross-split leakage, zero
duplicates, 8.04 boxes/image.

Class-agnostic localization recall on the held-out val set — class-agnostic because the
old model is 3-class and the new one is 4-class, so a per-class comparison would be skewed
by the index shift:

| detector | recall @ IoU 0.5 |
|---|---|
| old `pokemongo_yolov8n.pt` | 23.3% (31/133) |
| new `models/v3/det` | **69.2%** (92/133) |

Per-class mAP50: pokemon 0.809, gym 0.566, pokestop_rocket 0.432, **pokestop 0.169**.
PokéStop is the weak class and the next thing worth labelling.

A larger `yolov8s` is training in the background. When it finishes, compare and adopt:

```bash
python3 tools/adopt_best_detector.py             # compare on the held-out val set
python3 tools/adopt_best_detector.py --install   # install the winner
```

It ranks by class-agnostic recall rather than mAP, because candidates have different
class counts and what the bot needs is "did it find the object at all".

The screen classifier is 5-class (Overworld / PokemonEncounter / Menu / Poi / Rocket),
collapsed from 17 because rare classes had 1–2 samples each. Note the previous 17-class
dataset had **16 train folders but only 10 valid folders**, so ultralytics built different
class indices per split and every validation label was scrambled — its metrics were
meaningless.

## Known limits

- **PokéStop spinning is not yet working.** Over two live soaks the bot opened stops but
  spun zero discs: most taps returned "Walk closer to interact", and PokéStop detection is
  the weakest class (mAP50 0.169). Pokémon catching works well. Labelling more PokéStops
  is the highest-value next step.
- **The Rocket path has not been exercised live** — no Rocket stop appeared during
  testing. It is covered by unit tests and the button finder hits 100% of the labelled
  BATTLE/party screens, but it has not run against the real game.
- **Gym screens can classify as PokemonEncounter** (`Poi` has 8 samples). The confidence
  gate refuses them at 0.59, and the ENCOUNTER timeout bounds any mistake at 25s.
- Turn off the **Pointer location** developer option. It draws white text across the top
  of every frame, which lands in the flee-icon ROI and is baked into saved frames.
