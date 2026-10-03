# The word follower on the phone: timed evidence instead of a guess (2026-10-01)

Hasan, after a Dua Faraj session on the phone (20261001-023126-797f): "it was fine but
jumpiness towards the end". Lines 6-9 of Faraj are two words each (*wa-qāʾidan wa-nāṣirā*,
*wa-dalīlan wa-ʿaynā*). The page went line 7 → 8 → 7 → 8 → 9 in five seconds, while the
evidence was a line behind.

## Why it jumped

The page's only evidence was a 6 s Whisper transcript about once a second, with no timing
inside it, shown 1.0-1.5 s after its audio. To be on time the page showed where the reciter
would *probably* be by then: the delay plus 0.25 s, at the reciter's pace, about two words. On
a two-word line that is the next line. When the next transcript didn't confirm it, the page
stepped back. Every display rule added this week (the lead, the glide, the pause and stop
rules, back_confirm, retreat) managed the same gap between stale evidence and the present.

A CTC model gives the missing timing: letter probabilities every 20 ms. The word follower
(`follower.py`, round 1 in [follower.md](follower.md)) placed words from wav2vec2 frames on the
server and did far better than the page (test word exact 65% vs 38%), but wav2vec2 has ~300 M
parameters. A phone needs its own model.

## The phone's CTC model

`src/dua_recognition/ctc_student.py`: the encoder of the phone's own Whisper
(whisper-base-syn-v5-ctx8ft, 20 M parameters), reading the latest 3 s, with one linear layer onto
the corpus letters. Whisper's encoder emits one vector per 20 ms, wav2vec2's frame rate.

**Trained from wav2vec2** (`scripts/ctc_teacher.py`, `scripts/train_ctc_student.py`):

- The teacher ran over every training window (68,757 reciter windows, 20,213 synthetic
  ordinary-voice clips, 1,584 crowd-sourced clips): its letter posteriors, and a forced
  alignment of the window's reference text to them (when each letter is said).
- The student hears 3 s crops ending anywhere, often mid-word, sometimes at a session's start.
  Two targets: the teacher's posteriors on the crop's frames (the teacher had the whole 6 s),
  and CTC on the letters said inside the crop. A label's letters count only where the teacher
  hears nothing outside the labelled span: window labels drop words cut at the edges.
- What the student hears is roughened: level, noise, room reverb, vocal tract length and
  SpecAugment as the Whisper fine-tune; from v2 on also a phone's mic path (`phone_channel`:
  gain control that turns up the hiss once the reciter stops, a narrower band, compression) and
  twice the synthetic voices (children's and designed voices included).
- 13-15 minutes on the RTX 5080.

**A teacher for ordinary voices.** wav2vec2-quran-dua had only heard reciters, and the student
copies its frames. One more epoch on the same windows plus the synthetic and crowd-sourced
voices, with room and phone-channel augmentation (`scripts/finetune_ctc.py --extra ... --phone
0.3`, 50 minutes): `wav2vec2-quran-dua-voices`, RetaSy CER (whole clips) 31.2% → 27.0%, Quran-Lab
8.4% → 8.3%, its own validation 25.5% → 21.4%. Student v3 is v2's recipe on its frames.

**Real ordinary voices.** Common Voice 17 Arabic (CC0, `scripts/build_cv_set.py`): people reading
sentences into their own phones and laptops. Its train split is 28,298 clips from only 23
speakers; its test split, 10,423 clips from 968 speakers, trains too (its numbers aren't reported
here); 1,304 clips of its validation speakers (61) are kept back as a check. Read sentences, not
recitation, but the voices and the mics. The teacher hears them well (8.0% CER on the held-out
speakers) and the student badly (v3: 29.9%), so the teacher's frames on them are worth learning
from. Student v4 = v3 plus those clips; v5 and v6 = v4 trained two and four times as long (16
and 32 epochs; 45 minutes and 1.5 hours). Each doubling helped the ordinary voices; a third
(v7, 64 epochs) no longer did on the whole-clip numbers, and was worse on Hasan's sessions (±1
86.4% vs 88.5%, 3.5 jerks a minute vs 2.6), so it wasn't adopted. v6 is the phone's model.

| student | teacher | recipe | 3 s crops, validation CER |
|---|---|---|---:|
| v1 | wav2vec2-quran-dua | base | 26.5% |
| v2 | wav2vec2-quran-dua | + synth v6, phone channel | 26.6% |
| v3 | wav2vec2-quran-dua-voices | as v2 | 24.9% |
| v4 | wav2vec2-quran-dua-voices | as v3 + Common Voice | 25.4% |
| v5 | wav2vec2-quran-dua-voices | as v4, 16 epochs | 24.5% |
| **v6 (phone)** | **wav2vec2-quran-dua-voices** | **as v4, 32 epochs** | **24.0%** |
| tiny v2 | wav2vec2-quran-dua | as v2, whisper-tiny's encoder | 28.6% |
| tiny v3 | wav2vec2-quran-dua-voices | as v3, whisper-tiny's encoder | 28.0% |
| tiny v4 | wav2vec2-quran-dua-voices | as v4, whisper-tiny's encoder | 28.2% |
| tiny v5 | wav2vec2-quran-dua-voices | as v5, whisper-tiny's encoder | 26.8% |
| **tiny v6 (slow phones)** | **wav2vec2-quran-dua-voices** | **as v6, whisper-tiny's encoder** | **26.1%** |

(Validation is reciters' windows; the teacher itself disagrees with these crop labels on 14.8%.)

**Earlier at the edge.** Trained on crops that end mid-word, it emits letters sooner than the
teacher it learnt from: line entries on DuaPlayer test +0.65 s after the human line start,
against the teacher's +0.94 s. Streaming, each 0.2 s taken from the window ending there:

| | RetaSy, streaming CER | RetaSy, ≤ 3 s whole | Quran-Lab, streaming | Quran-Lab, ≤ 3 s whole | Common Voice (held out), streaming | CV, ≤ 3 s whole |
|---|---:|---:|---:|---:|---:|---:|
| wav2vec2 (teacher, server) | 78.0% | 30.1% | 61.9% | 18.0% | | |
| student v1 | 59.3% | 33.0% | 36.0% | 28.0% | | |
| student v2 | 59.2% | 36.3% | 36.0% | 28.0% | 34.0% | 33.0% |
| student v3 | 46.7% | 29.5% | 33.7% | 30.1% | 31.8% | 29.9% |
| student v4 | 45.3% | 27.9% | 32.1% | 26.7% | 24.3% | 16.1% |
| student v5 | 43.7% | 25.1% | 30.7% | 23.9% | 22.1% | 13.3% |
| **student v6 (phone)** | **40.1%** | **22.2%** | **29.2%** | **22.0%** | **20.4%** | **11.6%** |
| tiny v2 | 64.1% | 42.1% | 41.8% | 29.2% | | |
| tiny v3 | 52.4% | 38.3% | 39.7% | 35.1% | 37.7% | 36.3% |
| tiny v4 | 51.6% | 36.7% | 38.5% | 31.7% | 29.8% | 21.3% |
| tiny v5 | 49.1% | 33.6% | 36.8% | 26.1% | 27.9% | 20.6% |
| tiny v6 (slow phones) | 46.8% | 31.0% | 34.3% | 26.1% | 25.8% | 17.5% |

**In the browser**: `web/ctc-worker.js` (transformers.js's Whisper feature extractor, onnxruntime-web,
int8, 20 MB). Desktop Chrome, one 3 s window: 68 ms on one thread, 35 ms on two, 22 ms on four;
the Whisper encoder the phone already runs takes 204 / 105 / 62 ms. The real page replaying a
session ran the follower every 0.25 s.

## The follower's rules

`follower.py` and `web/follower.js` (parity-tested, `tests/test_follower_parity.py`). Round 1's
scoring: every step, the latest 2 s of frames against the words around the current one (the
best CTC path ending on each word), a penalty per word moved back, at most two words forward.
Five additions, each from a failure seen on train, in Hasan's sessions or in the real page:

| rule | what failed without it |
|---|---|
| `back_min` 2, `back_confirm` 2: a one-word step back is ignored; longer ones must win twice in a row | the highlight flicked back a word when a held vowel was rescored |
| `leap_margin` 4: a word beyond reach that beats everything within reach by 4 nats, twice in a row, is gone to | on 797f the model heard *fī hādhihi s-sāʿa* as *fānīsā*; the follower fell 3+ words behind, and every later word was out of its two-word reach: stuck 5 s |
| `word_starts`: a path may start mid-word only in the current word or earlier; a later word must be entered at its first letter | Hasan's *wa-nāṣirā* came out *nās*; the middle of the next line's *wa-dalīlan* fit no worse, and the page entered the next line in the pause |
| `reset_after` 3 s (was 1 s): re-anchor on the tracker only after 3 s of more-than-a-line disagreement | the phone tracker's position is often wrong on an ordinary voice; re-anchoring on it pulled the follower back and forth |
| `ahead_words_reanchor` 4: the tracker 4+ words ahead in the same du'a for 2 s takes the follower forward (never back) | 797f through the real page: the model didn't hear *arḍaka ṭawʿā wa-tumattiʿahu fīhā*, the follower waited on *tuskinahu*, Hasan finished line 9 and stopped, and it stayed on line 8 to the end. Offline the same frames caught up; the page's slightly different windows didn't. Never fired on the sessions or train offline (exact and jerks unchanged) |

Tried and dropped: forcing every path to start at or before the current word (it lagged, then
skipped: exact 54% vs 58% on train); an off-text gap test (no separation between on- and
off-text windows); following through the tracker's lapses (`lapse_hold`, neutral on the sessions);
settling forward in silence (moving up to six words to where the last letters end once the
reciter is silent: it fixed the same ending, but breaths inside drawn-out words look like stops,
and on train jerks went 0.40 → 0.73 a minute and the next line in pauses 1.4 → 2.4%).

The follower's path near a tie is sensitive: the page's int8 model and Python's float model agree
on 99% of frames, yet the follower's words agree on only ~80% of steps over a session, because one
different decision changes the next window's anchor. The rules above are what keeps that from
showing: steps back need confirming, and a follower left behind is brought forward.

## Results

**DuaPlayer test** (held-out reciters; one look, criteria in
`data/cache/overnight/prereg.md`). Tracker on the phone model's windows, shown 1.2 s late as
on the phone; the follower's results 0.1 s after their window:

| flowing recitation | word exact | ±1 | jerks/min | line backs/min | line flickers/min | line entry | > 0.3 s early |
|---|---:|---:|---:|---:|---:|---:|---:|
| page (yesterday's) | 34.7% | 83.9% | 0.53 | 0.51 | 0.23 | +0.28 s | 20.7% |
| follower, student v1 | 78.8% | 98.1% | 0.09 | 0.00 | 0.00 | +0.65 s | 0.7% |
| follower, student v2 | 78.6% | 98.0% | 0.10 | 0.01 | 0.00 | +0.64 s | 0.7% |
| follower, student v3 | 78.4% | 97.9% | 0.10 | 0.00 | 0.00 | +0.65 s | 0.7% |
| follower, student v4 | 78.5% | 97.9% | 0.09 | 0.00 | 0.00 | +0.65 s | 0.6% |
| follower, student v5 | 78.8% | 98.0% | 0.08 | 0.00 | 0.00 | +0.64 s | 0.8% |
| **follower, student v6** | **79.3%** | **98.1%** | **0.08** | **0.00** | **0.00** | +0.63 s | **0.8%** |
| follower, wav2vec2 (server) | 65.6% | 96.0% | 0.17 | 0.03 | 0.01 | +0.95 s | 0.8% |

| pauses (4 s after every third line) | word exact | jerks/min | next line shown in pauses | > 0.3 s early |
|---|---:|---:|---:|---:|
| page | 31.6% | 1.52 | 53.3% | 26.8% |
| follower, student v1 | 80.0% | 0.09 | 1.2% | 1.0% |
| follower, student v2 | 79.6% | 0.09 | 1.5% | 0.9% |
| follower, student v3 | 79.4% | 0.10 | 2.1% | 1.3% |
| follower, student v4 | 79.7% | 0.09 | 1.5% | 1.2% |
| follower, student v5 | 79.8% | 0.09 | 1.8% | 1.7% |
| **follower, student v6** | **80.2%** | **0.09** | **1.6%** | **1.6%** |

Over the 224 minutes of the 15 test recordings, the page stepped back a line 114 times; the
follower (v3 to v6 alike) once. Every criterion passed. The price is line changes about 0.37 s
later: the follower waits for the new line's letters instead of predicting them. v1 was the
model the test was looked at with; v2 to v6 were chosen on Hasan's sessions and
ordinary-voice CER, and each run on test once as a check. On reciters they are the same.

**Hasan's sessions.** Word truth: each whole session forced-aligned to the du'a's text with
wav2vec2 (`scripts/session_eval.py`); the three early sessions where he tapped "I'm here" 4-7
times (he moved around on purpose) break that alignment and are left out. Seven sessions, the
page as the phone drew it against the follower over the same audio and updates:

| | word exact | ±1 | line | > 0.3 s early | ahead of him in pauses | line backs | flickers |
|---|---:|---:|---:|---:|---:|---:|---:|
| page, as recorded | 29.9% | 64.8% | 57.0% | 26% | 21% | 3 | 3 |
| follower, student v1 | 47.2% | 81.6% | 62.3% | 7% | 10% | 2 | 0 |
| follower, student v2 | 56.5% | 85.7% | 65.0% | 5% | 7% | 3 | 1 |
| follower, student v3 | 59.0% | 85.2% | 65.1% | 5% | 7% | 3 | 1 |
| follower, student v4 | 59.3% | 86.9% | 65.7% | 7% | 10% | 3 | 1 |
| follower, student v5 | 59.7% | 86.6% | 65.4% | 7% | 9% | 3 | 1 |
| **follower, student v6** | **59.7%** | **88.5%** | **66.2%** | **7%** | **7%** | 3 | 1 |
| follower, tiny v2 | 45.5% | 77.4% | 60.0% | 11% | 11% | 4 | 1 |
| follower, tiny v3 | 50.7% | 81.8% | 64.3% | 9% | 9% | 2 | 0 |
| follower, tiny v4 | 50.9% | 79.6% | 61.5% | 9% | 9% | 3 | 1 |
| follower, tiny v5 | 56.9% | 85.2% | 65.5% | 7% | 7% | 3 | 1 |
| follower, tiny v6 (slow phones) | 55.5% | 83.6% | 64.6% | 9% | 7% | 3 | 1 |

Word jerks (a step back, or more than two words forward) per minute: page 1.0, v2 2.4, v3 1.7,
v4 2.9, v5 2.6, v6 2.6. On Hasan's voice v3 to v6 are close (seven sessions, one speaker); the
ordinary-voice sets above (~1,600 clips from many speakers) decided, and v6 is also the best
on his sessions within one word (88.5%) and on the line (66.2%).
The page's glide hides one-word corrections the follower shows; its line-level steps back are
the ones Hasan saw.

The truth comes from the teacher, so these flatter the follower somewhat; the page's line
column is less affected (line truth needs only roughly right word times).

**797f through the real page** (`page_replay.mjs`, Whisper held to 1 s, the CTC model to 150 ms):
line 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9, no step back (yesterday: 7 → 8 → 7 → 8 → 9). Within line 8
it waits on *tuskinahu* for six seconds, where the model doesn't hear him, and reaches the last
word of line 9 at 32 s, 1.3 s after he said it. **h649** (the stop that showed line 5 for 8 s
yesterday): the highlight stays on *bi-idhnih*, the next line previewed.

## What changed in the app

- `web/app.js` device engine: the follower is on by default (`?words=off` turns it off). A
  second worker (`ctc-worker.js`) runs the CTC model on the latest 3 s whenever it is free and
  0.2 s of audio has arrived (`?ctchop=`). Its word places the line and word directly;
  Whisper and the tracker still find the du'a, offer the "Is it…?" chips, and anchor the
  follower.
- While the follower places words, Whisper runs every 2 s instead of every second
  (`?anchorhop=`): it only anchors, and the phone's CPU goes to the CTC model.
- The next-line preview follows the follower: on a line's last word with the page hearing
  silence, kept until the word changes.
- When the follower catches up several words at once within a line, the highlight runs through
  them at 70 ms a word instead of snapping over them.
- Picking a du'a or tapping "I'm here" resets the follower to the new anchor.
- Sessions log every follower step (`ctc` events: model time, word, anchor) and its settings;
  `session_report.py` prints its cadence and model time.
- The CTC worker loads after Whisper and runs once on silence before its first step (loading
  both at once slowed Whisper's first windows). With the follower available, the tracker's own
  display shows its evidence, not a lead: before the follower's first step that lead put the page
  a line ahead, and the follower then took it back.
- A phone whose CTC steps take over 400 ms (median of a session's first 20) switches itself to
  the whisper-tiny student from its next session.
- The server (`app/server.py`) runs the same follower on the same student by default
  (`DUA_CTC_MODEL`, `?words=off`); before, `?words=ctc` ran it on wav2vec2.

## Not done

- The phone's own speed: measured on desktop Chrome only. On Hasan's Android the Whisper
  encoder is ~3-5x slower than here; the CTC model should take ~100-200 ms per step there. The
  follower runs whenever the previous step is done, so a slower phone means fewer steps, not a
  queue. A whisper-tiny student (`ctc-student-tiny-v6`) is the fallback if it is too slow.
- Ordinary voices remain the weak point (word exact ~60% on Hasan against 79% on reciters).
  Where the model doesn't hear a phrase (797f: *arḍaka ṭawʿā*), the follower waits on the last
  word it was sure of, 8% of the time two or more words behind, then catches up. More real
  ordinary-voice recordings would help most: Hasan's own sessions, labelled by tapping, are
  the closest thing to the app's users.
- Line entries are later than the old page's guesses, by ~0.35 s on reciters and ~0.4 s on
  Hasan's sessions. Moving on to the next line when the page hears the voice resume after a
  line-end pause would win some of it back, at the cost of a wrong move when the reciter
  repeats; not tried.

## Reproduce

```
python scripts/finetune_ctc.py --base models/wav2vec2-quran-dua --name wav2vec2-quran-dua-voices \
    --data v4 --extra synth_v5,synth_v6,crowd --phone 0.3 --room 0.3 --epochs 1 --lr 1e-5 --warmup 300
python scripts/ctc_teacher.py val_v4 crowd synth_v5 synth_v6 train_v4 --model models/wav2vec2-quran-dua-voices \
    --out data/cache/ctc_student_voices
python scripts/build_cv_set.py && python scripts/build_cv_set.py --split test --name cv_ar_test
python scripts/build_cv_set.py --split validation --name cv_ar_val --files 1      # held out
python scripts/ctc_teacher.py cv_ar cv_ar_test --model models/wav2vec2-quran-dua-voices --out data/cache/ctc_student_voices
python scripts/train_ctc_student.py --name ctc-student-base-v6 --teacher ctc_student_voices \
    --synth synth_v5,synth_v6,cv_ar,cv_ar_test --phone 0.5 --epochs 32
python scripts/train_ctc_student.py --base tarteel-ai/whisper-tiny-ar-quran --name ctc-student-tiny-v6 \
    --teacher ctc_student_voices --synth synth_v5,synth_v6,cv_ar,cv_ar_test --phone 0.5 --lr 3e-4 --epochs 32
python scripts/export_ctc_student.py models/ctc-student-base-v6 --window 3
python scripts/dump_ctc.py --model models/ctc-student-base-v6 --window 3 --hop 0.2
python scripts/pause_eval.py build --tag whisper-base-quran-dua --split test --ctc-model models/ctc-student-base-v6
python scripts/follow_eval.py --split test --asr whisper-base-syn-v5-ctx8ft --ctc ctc-student-base-v6 --delay 1.2 fw
python scripts/follow_eval.py --split test --pauses --ctc ctc-student-base-v6 --ctc-dir ctc-student-base-v6 --delay 1.2 fw
python scripts/session_eval.py --ctc models/ctc-student-base-v6 9puq fcr2 s4pv 0t98 h649 4t6h 797f
python scripts/ctc_voice_eval.py models/ctc-student-base-v6 [--set quranlab]
node scripts/ctc_bench.mjs threads=2 models/ctc-student-base-v6/model_q8.onnx:80x300
node scripts/page_replay.mjs data/sessions/20261001-023126-797f.wav out.wav "" --asr-ms 1000 --ctc-ms 150
```

(v4, v5: v6 with 8 and 16 epochs; v3: v4 without the Common Voice sets; v1 and v2 also without `--teacher`, v1 without
`--synth ... --phone 0.5`; the first teacher pass without `--model`/`--out`.)

Job logs and the pre-registration: `data/cache/overnight/`.
