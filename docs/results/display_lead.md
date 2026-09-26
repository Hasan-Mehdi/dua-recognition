# The live display: showing where the reciter is now (2026-09-25)

A tester reading Dua Tawassul into the phone said it felt "always a little behind",
and then, once a lead was added, that it "either goes ahead or doesn't follow along
nicely". Both were right. This file covers what was measured and what changed.

## Word-level ground truth

DuaPlayer's human timings mark where each *line* starts, which can't score a word
highlight. `scripts/word_truth.py` force-aligns each labelled line's known text to its
audio with the fine-tuned wav2vec2 (CTC Viterbi, torchaudio). Knowing the text makes
this far more reliable than recognising it: the median per-frame score is -0.2 to -0.35
on every recording except Ziyarat Ashura, whose line labels are already known to be bad.
That gives start and end times for ~10k words across the 15 test recordings, plus 18k on
the train reciters for tuning. `scripts/word_eval.py` then checks what the screen shows
every 0.1 s against the word being said.

## 1. The display was 1.75-2.25 s behind: show the belief predicted forward

Each update describes the moment the 6 s window ended, but it reaches the screen only
after Whisper has run (0.3-1 s on a phone). Whisper also tends to drop the window's
last word or two. The tracker already models motion (speed prior + tempo), so the
display now shows the belief predicted forward by the measured delay (the audio
captured while ASR ran) plus `TrackerConfig.display_lead` = 0.5 s. The belief itself
is untouched. Silent windows hold the last position shown.

Line level, test reciters, whisper-base-quran-dua (the phone model), scored against the
line being recited *when the result is on screen*. `display_lead` was chosen on the
train split:

| ASR delay | lead | line acc | refrain | line-switch lag (median) | jumps/min |
|---|---|---|---|---|---|
| 0 s | none | 85.4% | 89.0% | 1.25 s | 0.05 |
| 0.5 s | none | 79.4% | 81.7% | 1.75 s | 0.05 |
| 0.5 s | **1.0 s** | **87.2%** | **91.4%** | **0.58 s** | 0.05 |
| 1.0 s | none | 73.5% | 74.7% | 2.25 s | 0.05 |
| 1.0 s | **1.5 s** | **86.8%** | **90.4%** | **0.61 s** | 0.06 |

    python scripts/evaluate.py --asr whisper-base-quran-dua --latency 0.5

## 2. ...but the highlight then jumped ahead: two display rules

At word level the lead helped on average, but visible jerks rose from 0.3 to 4 a
minute. Nearly all of them (3.98 of 4.0) came at line changes: the predicted position
often landed two or three words into the new line, so the highlight skipped its
opening words. Separately, even with no lead, updates arrive once a second while
reciters say 1-2 words a second, so the highlight hopped one or two words at a time.

- **Start a new line at its first word** (`TrackerConfig.enter_line_at_start`). This
  only applies when moving on to the *next* line; a jump elsewhere still lands
  wherever the tracker says.
- **Glide between updates** (`display.py`, `web/display.js`): the highlight moves at
  the reciter's own pace (the tempo estimate x 1.2), eases into each new estimate over
  1 s, never steps back within a line, and never glides past a line's end on its own.
  Both settings were tuned on the train reciters' word timings.

Test reciters, phone model, 0.5 s ASR delay:

| display | on the word being said | within 1 word | mean offset | jerks/min | words skipped/min |
|---|---|---|---|---|---|
| before (no lead, hops) | 24.1% | 73.4% | -0.95 | 0.3 | 12.0 |
| lead only | 33.9% | 85.7% | -0.24 | 4.0 | 16.5 |
| **lead + first word + glide** | **40.1%** | **88.7%** | -0.17 | **0.0** | **6.0** |

At 1.0 s delay: 13.7% / 56.0% before, 36.7% / 86.5% now, with 0.1 jerks and 7.6
skipped words per minute. A jerk is a step back, or a skip of more than two words.

    python scripts/word_eval.py --delay 0.5 "none;cfg enter_line_at_start=0" "fixed;smooth ease=1.0,speed_scale=1.2"

## Negative results

- **A lead from Whisper's word timings.** Aligning each window's transcript to its audio
  (`scripts/dump_word_ends.py`) shows how long ago the last recognised word ended:
  median 0.1-0.6 s, 90th percentile up to 2.3 s. Using that instead of the fixed 0.5 s
  made the word highlight no better, and at line level gained 1.4 points on train at
  4x the jumps. So the phone model's ONNX export doesn't need attention outputs.
- **Faster recovery from the wrong copy of a refrain.** In the tester's clip the
  display followed the wrong copy of Tawassul's five-line refrain (it repeats 14 times),
  then took ~16 s to find the right section. During the refrain the belief was correctly
  spread over all 14 copies. What moved it was "yā abā al-Ḥasan", heard for five
  windows by both the phone model and the fine-tuned turbo, where the text says "yā abā
  Ibrāhīm". Three other sections do contain that phrase. Two fixes were tried:
  - a chance of jumping anywhere within the same du'a: no effect;
  - keeping the sharp search-mode evidence weighting until one *place* holds half
    the mass. That fixed the clip, but on test, listeners joining mid-recitation
    lost 3.5 points on refrain lines.

  Neither was kept. The fine-tuned turbo (server mode) heard "al-Kāẓim" better and
  recovered 3 s sooner than the phone model.
