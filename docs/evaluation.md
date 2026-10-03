# Evaluation

## Ground truth

Ground truth comes from [DuaPlayer](https://www.duaplayer.org), where reciters upload
recordings together with a hand-recorded start time for every line. That gives
**39 recordings (10 h) of 22 du'as and ziyarat by 12 reciters**, each second labelled
with the line being recited. The tracker searches all 505 texts (DuaPlayer's 91, plus
414 from duas.pro and duas.org), so the 483 without test recordings act as distractors.

## Method

Everything below is measured on **held-out reciters** (6 reciters, 16 recordings, 4 h):
nobody in the test set was used to tune the tracker or to train an ASR model. The split is
by reciter, not by recording ([`splits.py`](../src/dua_recognition/splits.py)). The
evaluation replays each recording exactly as the live system hears it (6 s windows,
1 s hop) and compares the displayed line with the true line once per second.

## Line following

| front end | line | ±1 line | refrain lines | wrong du'a shown | median lag | du'a found @3 s | @10 s |
|---|---|---|---|---|---|---|---|
| baseline: per-window text matching (turbo) | 42.7% | 71.4% | 5.8% | 12.9% | 3.6 s | 66% | 88% |
| large-v3-turbo, stock | 81.5% | 96.1% | 82.6% | 0.4% | 1.4 s | 62% | 91% |
| large-v3-turbo + prompt biasing | 84.8% | 96.2% | 88.1% | 0.4% | 1.2 s | 70% | 93% |
| large-v3-turbo, fine-tuned (server) | 84.0% | 96.3% | 88.7% | 0.4% | 1.3 s | 76% | 92% |
| whisper-small, fine-tuned | 84.4% | 96.3% | 88.4% | 0.3% | 1.3 s | 76% | 92% |
| whisper-base (Quran), fine-tuned, earlier | 85.2% | 96.2% | 89.1% | 0.3% | 1.2 s | 75% | 92% |
| whisper-base (Quran), fine-tuned on more voices | 84.8% | 96.2% | 87.8% | 0.5% | 1.3 s | 72% | 92% |
| **whisper-base (Quran), + synthetic voices, 8 s context (phone)** | **85.4%** | **96.3%** | **88.6%** | 0.5% | 1.2 s | 76% | 93% |

The baseline matches each window's transcript against the corpus on its own, with no
tracker. The last row uses the tracker settings of 2026-09-30
([display_stability.md](results/display_stability.md)). The rows above it predate them;
at this replay, with no display lead, the only one that matters is `null_rate_locked`,
worth +0.2 points on train. *Refrain lines* are lines that recur word for word, so text
alone can't place them. From a cold start mid-recitation, the phone model names the du'a
within 5 s 88% of the time and within 10 s 93%.

The fine-tunes all follow professional reciters about equally well; where they differ is
everyday voices ([how-it-works.md](how-it-works.md#4-run-on-a-cpu-or-a-phone)), which is
why the phone runs the last one. With 91 texts the du'a was found within 3 s about 89% of
the time; 5.5 times as many candidates, many sharing whole phrases, slow that first guess,
while the wrong-du'a rate fell. The front-end comparison
([comparison.md](results/comparison.md)) and `test_small-ft.md` (the first whisper-small
fine-tune) are from the 91-text corpus.

Ziyarat Ashura is excluded from line accuracy (its human timings drift by several lines)
but kept for identification. On the 91-text corpus, a simulated phone-in-a-room (reverb,
15 dB SNR) costs the clean-trained model 8.5 points (85.4% to 76.9%); training with room
augmentation wins almost all of that back, 84.9% in the room
([comparison.md](results/comparison.md)).

## Word following

Lines are half of it; the highlight is on a word. Scored every 0.1 s against forced-aligned
word timings on the test reciters, at the phone's delay (Whisper's result shown 1.2 s after its
window), and on the same recordings with a 4 s pause inserted after every third line
([phone_follower.md](results/phone_follower.md)):

| display | word exact | ±1 word | line steps back /min | next line shown in pauses | line change after the human mark |
|---|---|---|---|---|---|
| Whisper + tracker, predicted forward, gliding (until 2026-09-30) | 34.7% | 83.9% | 0.51 | 53.3% | +0.28 s |
| **CTC word follower on the phone's small CTC model** | **79.3%** | **98.1%** | **0.00** | **1.6%** | +0.63 s |

The follower lights the word whose letters were just heard, so it is never early (0.7% of
line changes more than 0.3 s before the human mark, against 20.7%) and doesn't run on when the
reciter stops; it changes lines a little later, when the new line's first letters arrive.
Over the 224 minutes of test recitation it stepped back a line once; the old display, 114 times.

That table assumes each follower result is shown 0.1 s after its audio. On Hasan's Android it was
nearer 0.25 s (the CTC model shares the CPU with Whisper), which cost it 10 points: 68.6% word
exact on test. Since the evening of 2026-10-01 the phone runs the model on 2 s windows every 0.1 s
with audio in 50 ms chunks: **75.5%** at its own delay, words within a line lit 0.29 s after they
start instead of 0.44 s, the next line shown in 0.8% of pause time instead of 1.5%, and a reader
who says a line again followed back to its start two times in three instead of one in three
([phone_latency.md](results/phone_latency.md)).

## Results and research notes

Per-model tables with per-du'a breakdowns are the `test_*.md` files in
[results/](results/) (the phone model is
[test_whisper-base-syn-v5-ctx8ft.md](results/test_whisper-base-syn-v5-ctx8ft.md); the one
before it, [test_whisper-base-aug-v4.md](results/test_whisper-base-aug-v4.md)).

| topic | write-up |
|---|---|
| ASR speed and accuracy per window | [asr_benchmark.md](results/asr_benchmark.md), [asr_benchmark_turbo.md](results/asr_benchmark_turbo.md) |
| Front ends, prompting and room audio compared | [comparison.md](results/comparison.md) |
| Speed prior and tempo adaptation; a negative result on scoring text from CTC | [speed_prior.md](results/speed_prior.md) |
| Keeping the display on the word being said | [display_lead.md](results/display_lead.md) |
| Waiting when the reciter pauses | [pauses.md](results/pauses.md) |
| Saying "I don't know this one" for du'as outside the corpus | [unknown_dua.md](results/unknown_dua.md) |
| Texts that share a passage | [shared_passages.md](results/shared_passages.md) |
| Findings from phone sessions | [phone_sessions.md](results/phone_sessions.md) |
| The phone app before and after 2026-09-30 (display rules, 8 s context, synthetic voices) | [phone_before_after.md](results/phone_before_after.md) |
| Why the display jumped on ordinary voices, and four rules that stop it | [display_stability.md](results/display_stability.md) |
| The phone model with an 8 s audio context: four times faster | [phone_speed.md](results/phone_speed.md) |
| Synthetic ordinary voices (zero-shot TTS) for the phone model | [synthetic_voices.md](results/synthetic_voices.md) |
| Tilawa's streaming phoneme engine on du'as (negative) | [tilawa_duas.md](results/tilawa_duas.md) |
| A CTC word follower (server, wav2vec2) | [follower.md](results/follower.md) |
| The word follower on the phone: a small CTC model taught by wav2vec2 | [phone_follower.md](results/phone_follower.md) |
| The follower faster on the phone; the last word; repeats | [phone_latency.md](results/phone_latency.md) |
| Lines read out of order: a benchmark, and finding a reader who jumps | [out_of_order.md](results/out_of_order.md) |
| Noha (Urdu, Arabic, Farsi, English): language ID and identification | [noha_lid.md](results/noha_lid.md), [noha_match.md](results/noha_match.md) |

## Reliability checks

A second pass checked the measurements themselves, with each experiment's acceptance
rules written down before its results ([plan](research/claude-execution-plan-2026-09-26.md),
[review](research/current-state-review-2026-09-26.md)):

- [browser_gate.md](results/browser_gate.md): how much speech the phone's speech gate
  discards. `legacy` stays the default; `?gate=energy_assisted` is opt-in.
- [follower_reliability.md](results/follower_reliability.md): where the word follower's
  line lag comes from.
- [small_ctc.md](results/small_ctc.md): a smaller CTC model as the follower's front end.
  wav2vec2 stays.

Human review of word-level labels is still open: the review bundles are exported
([`review_bundle.py`](../scripts/review_bundle.py)) but not yet annotated, so all
word-level numbers rest on automatic alignments.

## Reproduce

```bash
python scripts/fetch_duaplayer.py                         # recordings and line timings (local cache)
python scripts/transcribe_windows.py --model large-v3-turbo
python scripts/evaluate.py                                # test reciters
python scripts/evaluate.py --split train --tune           # tracker grid search (train only)
```
