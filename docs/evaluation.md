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
| **whisper-base (Quran), fine-tuned on more voices (phone)** | **84.8%** | **96.2%** | **87.8%** | 0.5% | 1.3 s | 72% | 92% |

The baseline matches each window's transcript against the corpus on its own, with no
tracker. *Refrain lines* are lines that recur word for word, so text alone can't place
them. From a cold start mid-recitation, the phone model names the du'a within 5 s 88% of
the time and within 10 s 92%.

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

## Results and research notes

Per-model tables with per-du'a breakdowns are the `test_*.md` files in
[results/](results/) (the phone model is
[test_whisper-base-aug-v4.md](results/test_whisper-base-aug-v4.md)).

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
| An experimental CTC word follower | [follower.md](results/follower.md) |
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
