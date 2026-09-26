# A CTC word follower (2026-09-26): much better words, later lines

The review of this week's display patches (display lead, five pause rules) traced them to
one cause: the tracker gets a 6 s transcript once a second, with no timing inside it. By the
time a transcript arrives the word it ends on is a second old, and during a pause it can't
tell "stopped at the end of the line" from "about to go on". The patches guess around
that. This experiment gives the display timed evidence instead.

## What it does

`src/dua_recognition/follower.py`, `LocalFollower`. The HMM tracker keeps the jobs it is good
at: which du'a, which line, and re-anchoring. Every 0.2 s, while the tracker is locked:

- take the last `window_s` seconds of a CTC model's frames (wav2vec2, 20 ms per frame);
- score them against the letters of the words around the current one (6 back, 20 ahead)
  with `ctc_align.end_scores`: the best CTC path through that text ending on each letter;
- move to the word whose letter explains the audio best, minus `beta_back` per word moved
  back, at most `max_jump` words forward;
- re-anchor on the tracker if the two disagree by more than a line for `reset_after` s.

When the reciter stops, the frames after their last letter are blanks, and a path can
sit in a blank after that letter. So a pause stays on the last word by construction:
no lead, no step-back rule.

One detail mattered a lot. The follower holds if the recent tail has no letters in it.
Requiring 3 letter frames in the last 0.5 s held it mid-word through every long vowel
(CTC output is peaky): 43% exact with the tracker's anchor replaced by the true word, on
6 train recordings. Requiring 1 frame: 72%.

## How it was tested

`scripts/follow_eval.py`. Two lanes over the same recordings, both scored every 0.1 s
against the forced-aligned words (as `word_eval.py`):

- **hmm**: today's page mode (tracker + lead + pause rules + gliding highlight), from
  whisper-base-quran-dua windows, 0.5 s ASR delay. `word_eval.py`'s tick loop is now shared
  (`hmm_updates`, `replay_ticks`); refactored and old code both give 38.0% exact and
  0.7 jerks/min on test.
- **follow**: the same tracker's position (no lead) anchors the follower. The follower runs on
  CTC dumps (`dump_ctc.py --window 3 --hop 0.2`), and each result is on screen 0.1 s after its
  window ends. Measured cost: 36 ms for a 3 s wav2vec2-large window (8 CPU threads) + 45 ms
  per follower step.

The word truth comes from the same fine-tuned wav2vec2 the follower listens to, which could
flatter it. Two checks don't depend on that. **Line entry** compares with DuaPlayer's
*human* line starts: the median time from a line's start to the display entering it, and
how often it enters more than 0.3 s early. **The stock model** (rabah2026 wav2vec2, not
fine-tuned on du'as) runs as a guard.

Success was defined before looking at test: word exact ≥ baseline + 5 pts; next line
during pauses ≤ 10%; jerks/min ≤ baseline; line-entry median no more than 0.2 s worse and
no more early entries; the same direction with the stock model.

## Tuning (train split only)

A grid of 72 settings (window 1.5/2/3 s, beta_back 1/3/10, max_jump 2/4, temp 1/3, reset
1/3 s) on the flowing train recordings (22) and the train pause set (21 recordings).
Rule: the best flowing word exact among settings with jerks no worse than page mode on
both sets and next line during pauses ≤ 10%. Winner: window 2 s, beta_back 3, max_jump 2,
temp 1, reset 1 s (log: `data/cache/follow_tune_log.txt`). On train it already missed one
criterion. Every setting entered lines 0.9-1.0 s after the human start, against page mode's
+0.12 s.

| train | word exact | ±1 | jerks/min | line entry (median) | > 0.3 s early | next line in pause |
|---|---:|---:|---:|---:|---:|---:|
| page mode, flowing | 35.4% | 82.6% | 0.86 | +0.12 s | 30.6% | |
| follower, flowing | 58.2% | 92.1% | 0.78 | +0.97 s | 1.1% | |
| page mode, pauses | 37.7% | 80.7% | 1.59 | +0.20 s | 29.7% | 39.4% |
| follower, pauses | 62.6% | 92.4% | 0.74 | +0.94 s | 2.5% | 2.2% |

## Test (one look; log `data/cache/follow_test_log.txt`)

15 test recordings (Ashura excluded as usual), the same 15 in the pause set (463 pauses).

| test | word exact | ±1 | mean offset | jerks/min | line entry (median) | > 0.3 s early | next line in pause |
|---|---:|---:|---:|---:|---:|---:|---:|
| page mode, flowing | 38.0% | 87.4% | −0.22 | 0.75 | +0.30 s | 27.4% | |
| **follower**, flowing | **65.2%** | **95.2%** | −0.35 | **0.31** | +0.94 s | **1.1%** | |
| follower, flowing, 0.3 s compute delay | 53.6% | 92.4% | −0.50 | 0.31 | +1.14 s | 0.9% | |
| follower, flowing, stock CTC model | 62.1% | 93.9% | −0.33 | 1.22 | +0.97 s | 4.1% | |
| page mode, pauses | 39.3% | 85.2% | −0.15 | 1.25 | +0.29 s | 30.4% | 32.9% |
| **follower**, pauses | **68.2%** | **95.5%** | −0.27 | **0.37** | +0.93 s | 2.7% | **1.6%** |
| follower, pauses, stock CTC model | 63.3% | 92.2% | −0.19 | 1.76 | +0.93 s | 13.4% | 11.5% |

Against the criteria:

| criterion | result |
|---|---|
| word exact ≥ baseline + 5 pts | **pass**: +27 pts flowing, +29 in the pause set |
| next line during pauses ≤ 10% | **pass**: 1.6% (page mode 32.9%) |
| jerks/min ≤ baseline | **pass**: 0.31 vs 0.75; 0.37 vs 1.25 |
| line entry ≤ 0.2 s worse, no more early entries | **fail** on lag: +0.94 s vs +0.30 s. Early entries pass: 1.1% vs 27.4% |
| same direction with the stock model | **partly**: word exact +24 pts and early entries hold. Jerks don't: 1.22 vs 0.75, 1.76 vs 1.25. Next line in pause 11.5% (vs 32.9%) |

**Verdict: not passed.** So, per the plan, it is not wired into the server or the app.

## What the numbers say

- Within a line the follower is far better, and it does what the pause rules were trying
  to do without any of them.
- It is late into new lines. Human line starts sit ~0.3 s before the aligned first word
  (median; train and test alike), so the follower shows a new line ~0.65 s after the
  reciter starts it. Page mode is on time on median only because its lead guesses ahead,
  which is also why 27-30% of its line changes come more than 0.3 s early, and why it runs
  into the next line during pauses.
- The late entry is the price of waiting for evidence: a new line's first letter has to
  beat "still in the blank after the last one". The 0.3 s compute-delay row shows how it
  scales with latency.
- The jerk result depends on the fine-tuned model. With the stock model the follower hops
  more (still under 2/min).

## If it's pursued

- The obvious hybrid: page mode picks the line (on time), the follower picks the word
  inside it and holds at line ends in pauses. That is a new design, not a tuning of this
  one, and it would need its own train-only tuning and a fresh test look.
- Phone: wav2vec2-quran-dua is a 24-layer, ~300 M-parameter model. On-device would need a
  much smaller CTC model (a separate decision).
- Code: `src/dua_recognition/follower.py`, `scripts/follow_eval.py`, `tests/test_follower.py`,
  `scripts/pause_eval.py build --ctc-model`, `scripts/dump_ctc.py --window 3 --hop 0.2`.
