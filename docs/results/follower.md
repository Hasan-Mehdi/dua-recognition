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

# Round 2 (2026-09-26): fix the late line entry?

Hasan's call after round 1: find where the lag comes from (train only), try to fix it, give
the fix one fresh test look with criteria set beforehand, and, whatever the verdict, put
the follower in the server behind `?words=ctc` so he can try it on his own recitation.

## Where the lag comes from (train)

`scripts/follow_eval.py --split train --diag` (log `data/cache/follow2_diag_log.txt`), the
round-1 follower. **Entry lag** per truth word: the first 0.1 s tick, from 1 s before the
word's forced-aligned start, that shows that word or a later one, minus the start (capped
at +3 s). Line-first words are split by the silence before them.

| train flowing (22 rec.) | word exact | line entry (human) | other words, median / p75 | line-first, gap < 0.3 s | 0.3-1 s | > 1 s |
|---|---:|---:|---:|---:|---:|---:|
| page mode | 35.4% | +0.12 s | +0.57 / +1.17 | +0.31 | −0.10 | −0.43 |
| follower (round 1) | 58.2% | +0.97 s | +0.46 / +0.57 | +0.52 | +0.62 | +0.65 |
| A1: 0.2 s right context | 68.5% | +0.79 s | +0.32 / +0.40 | +0.38 | +0.44 | +0.48 |
| A1: 0.4 s right context | 70.9% | +0.70 s | +0.30 / +0.37 | +0.33 | +0.35 | +0.39 |
| A1: 0.6 s right context | 71.7% | +0.68 s | +0.30 / +0.36 | +0.33 | +0.33 | +0.36 |
| A2: true word as anchor | 58.2% | +0.97 s | +0.46 / +0.56 | +0.52 | +0.62 | +0.65 |
| A4: no compute delay | 65.0% | +0.87 s | +0.36 / +0.47 | +0.42 | +0.52 | +0.55 |

(line-first columns: medians; n = 95 / 1677 / 1132. The pause set gives the same picture:
+0.94 s round 1, +0.66 s with 0.6 s right context, A2 unchanged, A4 +0.84 s.)

- **Human line starts come 0.30 s before the aligned first word** (median; p25 0.16, p75
  0.48; both sets). DuaPlayer's annotators mark the breath/onset, not the first letter.
  About a third of the lag is this offset, and nothing that waits for letters can remove it.
- **The decision rule is not late (A3).** For line-first words, the step where the
  follower moves is the step where the word's first letter first beats blank in a frame
  after its aligned start: median 0.00 s, p25 and p75 also 0.00 (n = 2,988). So a softer
  score (logsumexp instead of max) had nothing to win and wasn't built.
- **The tracker is not the cause (A2).** Anchoring on the true word changes nothing.
- **wav2vec2 is late at the window edge (A1): ~0.3 s.** With right context the letters
  arrive earlier; the gain saturates at ~0.3 s of lag (and +13 pts word exact). Getting it
  means waiting that long, so live it's a trade, not a fix.
- **The compute delay costs its own length (A4):** 0.1 s → +0.1 s.
- **Line starts are only ~0.15 s worse than any other word** (+0.62 vs +0.46 s), and the
  length of the pause before them hardly matters. The follower isn't bad at lines; it is
  evenly ~0.5 s behind every word, and at line starts that adds to the 0.3 s annotation offset.

Budget for a line entry, flowing: 0.30 (human mark → first letter) + ~0.3 (model edge) +
0.1 (compute) + ~0.1 (0.2 s step) + ~0.15 (line-first extra) ≈ the measured +0.97 s.
Page mode is on time only because it guesses ahead (its lead), which is also why 30% of
its line changes come more than 0.3 s early.

## What was tried (train)

All new options are `FollowerConfig` fields that default to round-1 behaviour:

- **Faster scoring**: `ctc_align.end_scores` is numba-compiled (numpy fallback kept; parity
  test on 500 random cases, max relative difference 4e-7). A 40 x 110 scoring call went from
  0.30 ms to 0.004 ms. The round-1 train rows reproduce exactly (58.2% / 0.78 / +0.97 s).
- **Line onset** (`onset_gap`, `onset_window` 0.3 s, `onset_confirm`): on the last word of a
  line, letters in the latest 0.3 s after at least `onset_gap` s of blank move straight to the
  next line's first word. `onset_confirm` also requires the tracker's lead position to be past
  the current line.
- **Hysteresis** (`confirm_steps`): a move back, or more than one word forward, must win this
  many steps in a row. Aimed at the stock model's jerks.
- `soft` (logsumexp scoring) was not added: the plan made it conditional on A3 ≥ 0.2 s, and A3 was 0.

Grid: `onset_gap` {0, 0.15, 0.25, 0.4} x `onset_confirm` {0, 1} x `confirm_steps` {1, 2}
(14 distinct settings) on four train sets: flowing and pauses, each with the fine-tuned and
the stock CTC dumps (`dump_ctc.py --tag w2v-quran-stock --train-only`, `pause_eval.py build
--split train --ctc-model rabah2026/...`). Log: `data/cache/follow2_tune_log.txt`.

| train, fine-tuned CTC | exact (flow / pause) | jerks/min | line entry | > 0.3 s early | next line in pause |
|---|---:|---:|---:|---:|---:|
| page mode | 35.4 / 37.7% | 0.86 / 1.59 | +0.12 / +0.20 s | 30.6 / 29.7% | 39.4% |
| round 1 (onset off, confirm 1) | 58.2 / 62.6% | 0.78 / 0.74 | +0.97 / +0.94 s | 1.1 / 2.5% | 2.2% |
| confirm_steps 2 | 57.3 / 62.0% | 0.57 / 0.53 | +0.99 / +0.96 s | 0.7 / 1.9% | 1.5% |
| onset 0.15 s | 55.0 / 59.0% | 10.35 / 9.18 | −0.54 / −0.32 s | 58.4 / 50.6% | 8.2% |
| onset 0.25 s | 56.8 / 60.6% | 5.43 / 4.86 | +0.43 / +0.55 s | 28.0 / 26.4% | 7.7% |
| onset 0.4 s | 57.5 / 61.3% | 3.41 / 3.08 | +0.77 / +0.76 s | 12.6 / 13.7% | 7.1% |
| onset 0.4 s + confirm (lead) | 58.0 / 62.0% | 1.70 / 1.50 | +0.90 / +0.88 s | 3.3 / 5.0% | 5.5% |
| onset 0.4 s + confirm + confirm_steps 2 | 57.5 / 60.9% | 0.84 / 0.76 | +0.92 / +0.90 s | 2.9 / 4.5% | 11.4% |

| train, stock CTC | exact (flow / pause) | jerks/min | line entry | > 0.3 s early | next line in pause |
|---|---:|---:|---:|---:|---:|
| page mode | 35.4 / 37.7% | 0.86 / 1.59 | +0.12 / +0.20 s | 30.6 / 29.7% | 39.4% |
| round 1 | 55.2 / 58.1% | 1.68 / 2.05 | +1.00 / +0.97 s | 4.1 / 10.6% | 11.7% |
| confirm_steps 2 | 54.1 / 58.0% | 1.05 / 1.15 | +1.03 / +1.00 s | 2.9 / 7.7% | 8.3% |

What each did:

- **The onset rule fires inside words.** Line-final words are often drawn out, and CTC puts
  blanks of 0.15-0.4 s inside them. "Letters after a gap" then fires before the reciter has
  finished the line; ordinary scoring pulls the follower back, and it fires again. Hence the
  early entries and 3-10 jerks/min. A 0.4 s gap cuts the line lag to +0.77 s, but still at
  3.4 jerks/min.
- **Gated on the tracker's lead** (`onset_confirm`), it is safe but gains only 0.05-0.09 s:
  by the time page mode's lead is on the next line, the letters have usually arrived anyway.
- **`confirm_steps` 2 does what it is for.** Jerks drop by about a quarter (fine-tuned) to a
  third (stock). It adds ~0.02 s of lag. But combined with the onset rule it holds the
  onset's own corrections back, and next-line-in-pause rises to 11-20%.

## The go/no-go gate (set before the grid)

Selection rule: qualify on both fine-tuned train sets with jerks ≤ page mode, next line in
pause ≤ 10%, early entries ≤ page mode; then the lowest line-entry median. Only the two
onset-off settings qualify. Every onset setting fails on jerks, except onset 0.4 s + confirm +
confirm_steps 2, which fails next line in pause (11.4%). **Pick: the round-1 config**
(+0.97 / +0.94 s, lower than confirm_steps 2's +0.99 / +0.96).

| gate (train) | needed | pick |
|---|---|---|
| line entry median | ≤ +0.32 s flowing, ≤ +0.40 s pauses | **fail**: +0.97 / +0.94 s |
| word exact | ≥ page mode + 5 pts | pass: +22.8 / +24.9 pts |
| stock model vs page mode: exact, jerks, next line in pause, early | not worse on any | **fail** on jerks: 1.68 / 2.05 vs 0.86 / 1.59 |

**Verdict: gate failed, no test look.** Test was not touched in round 2, so it keeps its one
look for a future design. `FollowerConfig` defaults are unchanged (round 1).

## Why this can't be tuned away

The diagnosis says it plainly. The follower moves as soon as the letters are there (A3 = 0).
The letters come ~0.6 s after the human line start: 0.3 s of annotation offset, plus ~0.3 s
of model-edge latency. To enter a line on time, a display has to move *before* the letters
come. That means guessing, as page mode does, and page mode pays for it with 30% early
entries and 39% next-line-in-pause. Any evidence-driven rule fast enough to fire earlier
reads blanks inside a drawn-out last word as a line break (the onset rows).

That leaves a product choice, not a tuning one: for a reader, is a line change 0.6 s after
the voice worse than one that is 0.3 s early a third of the time and jumps ahead in pauses?
The ±0.3 s criterion treats late as the only failure. `?words=ctc` lets Hasan judge that by
ear and eye.

