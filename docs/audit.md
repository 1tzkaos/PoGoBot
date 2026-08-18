# Audit of the v1 bot

`legacy/pokemon_vision_bot.py` was a single 1301-line file. A seven-lens adversarial
audit — each lens an independent reviewer, each finding then handed to a second reviewer
whose job was to *refute* it — confirmed **106 defects**. They deduplicate to roughly 65,
of which 39 were instances of six missing structures rather than independent bugs.

This document records what was wrong and which structural decision in `pogobot/` prevents
it, so the reasoning survives the code.

## The three compound failures

Each of these was a self-sustaining loop that consumed an entire unattended session, and
each was assembled from parts found by different lenses.

### Menu loop

`find_close_button_coordinates` had no "not found" path. It fell back to
`(0.50w, 0.8808h)` = `(540, 2061)` on a 1080×2340 device — a coordinate verified to sit
*inside* the region the same file used to identify the overworld Pokéball. "Close the
popup" therefore **opened the main menu**. `CLOSING_POPUP`'s 4-second escape was an `elif`
chained below the tap branch, so it could never be evaluated while that branch held.

*Prevented by:* every finder returns `Optional`, and no located button means no tap. The
dispatcher checks a handler's timeout **before** running its body.

### Poison loop

`SPINNING_STOP` wrote a `+1.0` training sample 0.8 seconds after issuing the spin swipe,
with no evidence a PokéStop screen had ever opened. That branch produced 69% of the
corpus. Each sample was the whole frame plus exactly **one** box label; all 266 label
files contained exactly one line while 37% of the objects visible in those frames went
unlabelled, which teaches a detector that real objects are background.

Measured over the same 40 held-out frames, after three self-retrain generations:
**3.23 → 2.38 detections per frame.** 107 of 165 ingested label files were zero-byte, and
40 of 52 `merged_pokemongo` validation images were byte-identical to training images, so
the metric that would have revealed the decline was itself leaked.

*Prevented by:* `IntentLedger` is the only writer, it requires an explicit outcome inside
a causal time window, it writes every box in a frame or none, and it writes to a review
queue rather than a training split.

### Blind loop

`read()` returned `(True, last_frame.copy())` forever once the stream died, and the
`scrcpy` process was bound once and never polled. A bumped cable turned the bot into
roughly 3600 blind taps per hour at `(540, 2061)` against a phone it could not see, behind
a healthy-looking HUD and an FPS counter that measured loop iterations rather than frames.

*Prevented by:* `Frame` carries `seq` and `ts`; `read()` returns `None` past a maximum
age; the runner polls the process and halts rather than tapping blind.

## Structural causes

| theme | v1 | now |
|---|---|---|
| transition bookkeeping | `state` assigned at 12 sites, clock stamped at 9 | `enter_state()` is the only writer |
| debounce | one shared timer for eight unrelated actions | per-budget timers plus one settle window |
| actuation | 11 adb call sites, `no_click` checked at 5 | `Actuator` enforces it once |
| perception | absolute pixel counts on resolution-scaled regions | fractions of region area |
| liveness | no way to express a stale frame | `Frame(seq, ts)` and a maximum age |
| training writes | 3 call sites, 3 different preconditions | one ledger, one precondition |
| coordinates | stream pixels and device pixels mixed freely | normalized floats only |

## Measurements that changed the design

Several heuristics carried over from v1 were **refuted by measurement** and removed rather
than tuned.

- **Optical encounter detection** (giant ball plus flee icon) fires on 27% of overworld
  frames and catches only 30% of encounters. It now decides nothing and is retained only
  in the trace.
- **Ball-colour matching** does not separate at all: overworld frames score *higher* than
  encounters (0.268 vs 0.005 median) because the map is full of red and blue objects.
- **The red Pokéball alone** reaches 97% recall on the overworld but fires on 18% of
  encounters. Requiring the orange binoculars as well drops recall to 79% and false
  positives to approximately zero, which is the correct trade for a signal used as a veto.
- **Absolute thresholds are resolution traps.** At `--max-size 720` the binoculars check
  measured 479 pixels against a `> 500` bar, silently disabling overworld detection. As a
  fraction the same signal reads 0.235 versus 0.256 at full resolution.

## Dataset defects

The previous screen-state dataset had **16 class folders under `train` but only 10 under
`valid`**. Ultralytics builds class indices from each split's own sorted folder listing,
so `Overworld` was index 9 in training and index 3 in validation: every validation label
was scrambled and the resulting accuracy was worse than chance. Sixteen of its 24
validation images were also byte-identical copies of training images.

The current classifier uses five classes, present in every split, split by perceptual-hash
group so near-duplicate frames cannot straddle a boundary.

## Bugs found in the replacement

Recorded because they are the same species of defect and were caught the same way.

- `IntentOutcome.CONFIRMED` was unreachable in the assembled system: the scanning handler
  set `expected` to the state it passed *through* while waiting rather than the state that
  would confirm the tap, so a **successful** catch scored `REFUTED` and applied a cooldown
  to a location that had just worked. Nothing set the PokéStop visit flag either, so that
  path always expired.
- `ROCKET` was missing from the set of states the map returns to `SCANNING`. Observed
  live: the screen read `Overworld` at full confidence for 25 consecutive ticks while the
  state stayed `ROCKET` until its 150-second timeout.
- The Rocket BATTLE button is visible for roughly one second, and the generic UI-settle
  window from the preceding close tap swallowed the entire opportunity. Actions taken
  against an optically **located** button no longer wait for settle; blind taps still do.
