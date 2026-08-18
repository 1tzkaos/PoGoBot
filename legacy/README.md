# Legacy

Previous generations of this bot, kept for reference. None of it is imported by
`pogobot/` and none of it is maintained.

| file | what it was |
|---|---|
| `pokemon_vision_bot.py` | The v1 vision bot: a single 1301-line file combining capture, perception, a flat if/elif state machine, and adb actuation. Replaced by the `pogobot/` package. A 7-lens audit of this file found 106 confirmed defects; see [`docs/audit.md`](../docs/audit.md). |
| `train_feedback.py` | The v1 retraining pipeline. Superseded by `tools/promote_reviewed.py` and `tools/adopt_best_detector.py`. Its hardcoded 3-class `CLASS_NAMES` predates the 4-class detector and would mislabel every Pokémon as a gym. |
| `main.py` | The original pixel-matching bot, before any ML. Clicked near the screen centre and identified screens by sampling individual pixels. |
| `low-res.py` | A low-resolution variant of `main.py`. |
| `shundoSniper.py` | A standalone 100IV/shundo sniping helper. |
