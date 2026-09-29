# Word follower: which part of the line lag is actionable (2026-09-26)

Run `rel-20260926`, train split only. The round-2 verdict in `follower.md` stands unchanged:
its gate failed on train and the test look was not spent. This report doesn't rerun that
grid and doesn't loosen its criteria. It checks the latency accounting, splits the lag
against three kinds of reference, and decides whether another decision rule is justified.

## Is every live delay counted?

From the code (`scripts/follow_eval.py: follow_updates`):

- **CTC window.** The window scored at step *k* ends at time *t*: its frames never see
  audio after *t* in the normal lane.
  - The "right context" rows of round 2 (A1) use windows that end later. They are
    diagnostics, not live options, and are not repeated here.
- **Tracker anchor.** An anchor is usable only from its window end + the tracker delay
  (0.5 s).
- **Display.** A follower result reaches the screen at *t* + the compute delay.
- **Step cadence.** The 0.2 s step is counted by the 0.1 s tick replay: the screen holds
  the last result between steps.
- **Not counted:**
  - Paint (under one frame).
  - The drop in cadence when compute exceeds the 0.2 s step. The live CPU check had word
    steps at p50 286 ms, so updates came every ~0.35 s. The 0.29 s row below counts that
    step's delay, not the lower rate.

On a GPU the server measured 12 ms p50 / 80 ms p90 per word step, so 0.1 s covers compute
there. On CPU it doesn't.

## Line entry, by reference (`scripts/follow_latency.py --split train`)

22 flowing train recordings. Entry = the first tick within ±4 s showing the new line.

| lane | vs human mark: median | p90 of late | > 0.3 s early | missed | vs first aligned word: median | missed |
|---|---:|---:|---:|---:|---:|---:|
| page mode | +0.12 s | 1.89 s | 30.6% | 2.2% | −0.20 s | 1.6% |
| follower, 0.1 s compute | +0.97 s | 1.56 s | 1.1% | 2.8% | +0.63 s | 2.0% |
| follower, 0.2 s compute | +1.07 s | 1.65 s | 1.0% | 2.9% | +0.73 s | 2.0% |
| follower, 0.29 s compute (live CPU p50) | +1.16 s | 1.74 s | 0.9% | 2.9% | +0.82 s | 2.1% |

Pause set (21 recordings, 4 s room tone after every third line end):

| lane | vs first aligned word: median | missed |
|---|---:|---:|
| page mode | −0.07 s | 4.2% |
| follower, 0.1 s compute | +0.61 s | 2.2% |

On the pause set, "vs human mark" is not meaningful. A human mark can fall before the
inserted pause, 4 s ahead of the words, which shows up as follower "missed" 8.4%.

**Coverage.** Both lanes show something on 97.6–98.0% of ticks, and the follower misses
0.6 pt more transitions than page mode. It does not buy its low early-entry rate by
skipping hard lines.

**Human-audited onsets: none yet.** The review bundle
(`data/cache/reliability/rel-20260926/review/diag1`, 10 "first word after a pause"
excerpts among 60) is prepared but unreviewed. So the ~0.34 s between the human mark and
the aligned first word is **not yet explained**. Human line marks sit on average 0.34 s
before the aligned first word (0.97 vs 0.63 s here; 0.30 s median in round 2). That offset
could mean either of two things:

- **the annotators mark the breath:** then about a third of the "lag" isn't lag at all;
- **the forced aligner is late:** then the follower is later than the automatic numbers say.

Only reviewing those excerpts can tell the two apart.

## What is actionable

- **The decision rule is not late.** Round 2 found the follower moves at the step where the
  line's first letter first beats blank (A3: median 0.00 s), and anchoring on the true word
  changed nothing (A2).
- **Compute is actionable, one-for-one.** Each 0.1 s of word-step delay adds ~0.1 s at every
  line (+0.97 → +1.16 s from GPU-like to live-CPU cost). A CPU server or a slow phone pays
  it in full.
- **The emission is actionable only acoustically.** About 0.3 s of the remainder is wav2vec2
  scoring the window's last frames late (A1). Waiting for right context trades that for the
  same wait. What would reduce it is a model that emits earlier at the window edge, or a
  training-side fix (delay-penalized CTC, label-prior "less peaky" CTC). Neither is a
  display rule.
- **Annotation offset (~0.3 s):** unresolved until the human audit.

**Decision (plan Step 3.4):** the evidence confirms that emissions arrive late and the
follower acts promptly. So no new onset or commit rule is built, and the preview/commit
experiment (Step 3.5) is not run: its precondition, decision/display delay left over
after accounting, isn't met. The round-2 onset rule already showed what an earlier rule
does: it fires inside drawn-out final vowels (3–10 jerks/min). The next experiment is
acoustic (`small_ctc.md`), not another heuristic.

## Hasan's live impression

Hasan tried `?words=ctc` on his own recitation and liked it (2026-09-26). That is a product
signal, and it fits the numbers: the follower is 1–3% early where page mode is ~30% early,
and it holds in pauses (next line 2.2% vs 39.4% on train in round 2). It is not a measured
pass of the round-2 gate. The follower stays opt-in; making it the default in the server is
Hasan's product decision. If he wants that, the honest framing is "later line changes,
far fewer wrong ones", with the numbers above.

## Rerun on the full corpus (`rel-20260927`)

The first run used the 91-text tree (see `browser_gate.md`). On main, with all 505 texts, the
latency table repeats to within 0.1 pt and 0.01 s:

| lane | vs human mark: median | p90 of late | > 0.3 s early | missed | vs first aligned word: median | missed |
|---|---:|---:|---:|---:|---:|---:|
| page mode | +0.12 s | 1.89 s | 30.6% | 2.4% | −0.20 s | 1.8% |
| follower, 0.1 s compute | +0.97 s | 1.56 s | 1.1% | 2.9% | +0.63 s | 2.1% |
| follower, 0.2 s compute | +1.07 s | 1.66 s | 1.0% | 3.0% | +0.73 s | 2.1% |
| follower, 0.29 s compute | +1.16 s | 1.74 s | 0.9% | 3.0% | +0.82 s | 2.1% |

- **Pause set:** page mode −0.06 s (missed 3.9%) and follower +0.61 s (missed 2.3%)
  against the first aligned word.
- **Word following (`follow_eval.py`):**
  - Flowing: follower word exact 58.2% against 35.4% for page mode (±1: 92.1 vs 82.7%), with
    0.77 against 0.86 jerks/min.
  - Pause set: 62.6 vs 38.1%, and the next line shows in a pause 2.3% of the time against 36.3%.
- **Coverage:** shown on 97.4% (flowing) and 97.8% (pauses) of ticks for both lanes.

The conclusions above stand. The human audit is still the open item:
`rel-20260927/review/diag1` (60 diagnostic excerpts, 10 after pauses) and `audit1` (60 random
excerpts) are exported and unreviewed. Coverage (`coverage.txt`): 0 human-reviewed word labels
in any set.

## Reproduce

```
python scripts/follow_latency.py --split train
python scripts/follow_latency.py --split train --pauses
```

Eval sets are cached under `data/cache/follow_items`. The key now covers the tracker config,
reference texts, every input file and the scoring threshold; before this run it covered only
the ASR tag, split, CTC tag and delay.
