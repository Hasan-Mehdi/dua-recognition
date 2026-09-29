# Small-CTC comparison: Tilawa FastConformer vs wav2vec2 (2026-09-27)

Run `rel-20260927`, plan Step 4. Development data only: DuaPlayer train reciters.
The preregistered go rule was written on 2026-09-26, before any outcome: no default change
unless word and line quality fall within the follower gates of the incumbent **and** a real
device latency measurement passes.

## Candidate

`fastconformer_full_mixed.onnx` from Tilawa v0.2.0: a fine-tune of NVIDIA's
`stt_ar_fastconformer_hybrid_large_pcd_v1.0` (CC-BY-4.0), int4/int8, 80 ms frames, 1,024 BPE
tokens. It runs through buffered 3 s windows (`src/dua_recognition/ctc_adapter.py`), not
cached streaming. The incumbent is `wav2vec2-quran-dua` (~300M, 20 ms frames).

**Contract check (`small_ctc.py validate`, 27 dev lines):** median line CER 11.5% (mean
20.2%). The tokenization matches the model's own on 86 of 92 probes, and no reference
character falls outside the vocabulary. The model is usable as a follower front end.

## Results (`follow_eval.py --split train`, delays 0.5 / 0.1 s)

| front end | set | word exact | ±1 | jerks/min | line entry median | missed | next line in pause |
|---|---|---:|---:|---:|---:|---:|---:|
| wav2vec2 (incumbent) | flowing, 22 recs | **58.2%** | **92.1%** | **0.77** | **+0.97 s** | **2.9%** | — |
| FastConformer | flowing | 44.2% | 78.3% | 1.54 | +1.43 s | 7.9% | — |
| wav2vec2 (incumbent) | pauses, 21 recs | **62.6%** | **92.5%** | **0.74** | **+0.94 s** | **8.5%** | **2.3%** |
| FastConformer | pauses | 48.3% | 78.8% | 1.60 | +1.40 s | 13.1% | 5.1% |

Page mode, for reference: word exact 35.4% (flowing) and 38.1% (pauses).

- **Latency against the first aligned word** (`follow_latency.py`, 0.1 s compute):
  FastConformer +1.10 s with 6.7% missed, wav2vec2 +0.63 s with 2.1% missed.
- **Desktop CPU cost:** the dump ran at roughly 68 ms per 3 s window with 8 threads on a
  shared CPU. `small_ctc.py bench` was not run, so there is no clean timing.

## Verdict

**Fails the go rule; wav2vec2 stays.** FastConformer is 14 pts worse on word exact, twice as
jerky, misses more line changes (7.9 vs 2.9% flowing, 13.1 vs 8.5% with pauses) and enters lines about 0.46 s *later*. So the
hope that a different acoustic model would emit earlier at the window edge isn't borne out
by this candidate. Its 80 ms frames and buffered windows likely cost it here. No device
measurement was taken: there is nothing to adopt.

Per the plan, the Zipformer branch opens only on a clear unmet need, and this result isn't
one: the incumbent is better on every column. The emission-latency fix, if there is one,
is training-side (delay-penalized CTC on wav2vec2), not a model swap.

## Reproduce

```
python scripts/small_ctc.py validate
python scripts/small_ctc.py dump --threads 8
python scripts/small_ctc.py dump-pauses --threads 8
python scripts/follow_eval.py --split train --ctc fastconformer-tilawa "fw"
python scripts/follow_eval.py --split train --pauses --ctc-dir fastconformer-tilawa --ctc fastconformer-tilawa "fw"
python scripts/follow_latency.py --split train --ctc fastconformer-tilawa --follow-delays 0.1,0.2
```

Artefacts: `data/cache/reliability/rel-20260927/` (`fc_*.txt`, `fe_fc_*.txt`, `lat_fc_flow.txt`,
`small_ctc/validate.json`).
