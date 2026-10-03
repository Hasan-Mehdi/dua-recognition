# The phone model, four times faster: an 8 s audio context (2026-09-30)

On Hasan's Android phone the on-device model took 2.1-2.8 s per 6 s window
(`phone_sessions.md`), so the screen updated every 2-3 s with evidence 2-3 s old. On the
train split, that cadence alone costs about 5 points of line accuracy and most early
identification (see the phone-cadence rows in `display_stability.md`).

## Why a 6 s window costs 30 s of work

Whisper pads every input to 30 s. The encoder runs over all 1500 positions whatever the
audio's length, and the decoder cross-attends to all of them. The live follower only
ever sends 6 s, so four fifths of the encoder's work is padding.

The encoder's positional embeddings are fixed sinusoids. An encoder cut to its first 400
positions is therefore exactly a Whisper built for 8 s inputs: `max_source_positions`
400, the feature extractor's `chunk_length` 8, nothing re-initialized.
`scripts/shorten_context.py` writes such a model. Everything that reads the config then
feeds it 8 s of features: transformers, transformers.js (`preprocessor_config.json`),
and CTranslate2 through faster-whisper.

## Measured, before any fine-tuning

whisper-base-aug-v4 (the phone model), cut to 8 s:

| check | 30 s context | 8 s context |
|---|---:|---:|
| validation windows, CER (200, transformers) | 9.9% | 10.4% |
| the same, CTranslate2 int8 | 10.8% | 11.1% |
| CPU time per window (CTranslate2 int8, batches of 8) | 440 ms | **77 ms** |
| Hasan's 6 sessions, CER against the server model | 42.8% | 42.5% |
| headless Chrome, the page's own worker (ONNX q8, WASM), per update | 2,812 ms | **721 ms** |
| ...updates kept in 60 s of streaming (one per second of new audio) | 21 | **60** |
| ...evidence age when painted, p50 / p95 | 2.83 / 3.15 s | **0.74 / 0.91 s** |

- A second Chrome run on a quieter CPU (03:30): 2,414 ms against **583 ms** per update
  (4.1x), 24 against 60 updates kept, evidence 2.43 against 0.60 s old. The
  synthetic-voice model cut the same way takes 600 ms.
- A 7 s cut is worse: CER 16.7%.
- The desktop run shared the CPU with other jobs, so its absolute times are high. The
  ratio is the result: 3.9x fewer milliseconds per update.
- A phone measurement is still needed. If the ratio holds on Hasan's Android, updates go
  from every 2-3 s to about every second.

`web/models/whisper-base-aug-v4-ctx8` is the same int8 ONNX as the current phone model,
100 MB, with a 20.9 MB encoder instead of 23.1 MB. Open the page with
`?model=whisper-base-aug-v4-ctx8` to try it.

## Reproduce

```
python scripts/shorten_context.py models/whisper-base-aug-v4 --seconds 8   # + -ct2
.venv-export/Scripts/python scripts/export_onnx.py models/whisper-base-aug-v4-ctx8
node scripts/gate_bench.mjs <clip.f32> 120 60 out.json ungated@whisper-base-aug-v4 ungated@whisper-base-aug-v4-ctx8
```

`finetune_whisper.py --context 8` trains at that context.

## Following, and the cadence a phone can keep

Criteria, written before any tracker-level result (`data/cache/reliability/speed-20260930/prereg.md`):
1. At the same cadence (one update a second, 1 s late), line accuracy no more than
   1 point lower and wrong du'a no more than 0.3 points higher.
2. At the cadence each model can keep on the phone, line accuracy no lower than the
   current model's:
   - the 30 s model: every 3 s, shown 2.5 s late (the phone measured 2.1-2.8 s a window);
   - the 8 s model: every second, 1 s late (desktop Chrome measured 0.72 s).
3. Ordinary voices (RetaSy) no more than 1 point worse.

Then one look at test.

| | train: line | wrong du'a | lag | test: line | ±1 | wrong du'a | lag | lines missed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 s model, every 1 s | 81.8% | 0.5% | 0.6 s | 83.3% | 96.3% | 0.4% | 0.7 s | 1.5% |
| 8 s model, every 1 s | 81.3% | 0.4% | 0.6 s | 83.1% | 96.3% | 0.3% | 0.6 s | 1.3% |
| 30 s model, every 3 s (the phone today) | 76.3% | 0.1% | 1.8 s | 79.4% | 96.3% | 0.2% | 1.8 s | 8.3% |

RetaSy, the 388 clips of at most 8 s, each as one live window: 30.5% against 30.6%.

All three criteria pass, on train and on test.
- **At equal cadence** the cut costs 0.2-0.5 points.
- **At the cadence a phone can keep** it gains 3.7 points on test (5.0 on train) and cuts
  the lag from 1.8 to 0.6 s.

`web/app.js` loaded `whisper-base-aug-v4-ctx8` by default from 03:00. At 04:30 it moved on
to `whisper-base-syn-v5-ctx8ft`, the same 8 s context with synthetic voices in training and
one epoch of fine-tuning at 8 s (`synthetic_voices.md`): 580 ms per update in desktop
Chrome, and 85.1% on test at this cadence. `?model=whisper-base-aug-v4` brings back the
30 s model.

The phone's own timing is the one thing still assumed: the first session recorded on
Hasan's Android will show it (`session_report.py`: "Whisper: ... ms p50").
