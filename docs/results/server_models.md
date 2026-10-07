# Bigger models for the server engine (2026-10-06): not adopted

The server engine runs the page's own code with Whisper and the CTC model on the server
(server_engine.md). The server can afford bigger models than a phone, and two were already trained:
the fine-tuned large-v3-turbo (`whisper-turbo-dua`) and the wav2vec2 model the CTC student learned
from (`wav2vec2-quran-dua-voices`). Neither passes the bars written before the run
(`data/cache/server_models/prereg.md`); the server keeps the phone's two models.

## Setup

Candidates against today's server engine (Whisper base `whisper-base-syn-v5-ctx8ft` + CTC student
`ctc-student-base-v6`):

- **W**: base + wav2vec2
- **T**: turbo + student
- **TW**: turbo + wav2vec2

`bench.py score --display stream --split dev --same-text 8`, today's defaults, all 1,373 dev
items. Delays are the server's, not the phone's. Measured one window at a time on this PC's RTX 5080,
median of 40 windows:

| model | per window |
|---|---:|
| Whisper base, 6 s | 37 ms |
| turbo, 6 s (padded to 30 s) | 97 ms |
| CTC student, 2 s | 4 ms |
| wav2vec2, 2 s | 10 ms |

With ~15 ms of round trip, all four are under the pre-registered floors, so every run uses
`--delay 0.15 --follow-delay 0.05`. GPU passes: wav2vec2 43x real time; turbo 11x (74 h of audio
in 5.5 h).

## Dev, all items

| | base + student | W | T | TW |
|---|---:|---:|---:|---:|
| on the reader's line | 90.5% | 88.0% | 91.3% | 88.6% |
| right word | 76.1% | 68.2% | 76.6% | 68.6% |
| du'a found within 10 s* | 94.4% | 94.4% | 96.3% | 96.3% |
| wrong du'a shown | 0.42% | 0.41% | 0.37% | 0.37% |
| jumps /10 min | 1.27 | 1.14 | 1.27 | 1.10 |
| early /10 min | 1.71 | 1.87 | 1.73 | 1.93 |
| lost /10 min | 1.85 | 2.70 | 1.71 | 2.66 |
| go-back etc. followed within 3 s | 77% | 71% | 77% | 70% |
| line-change lag, median | +0.39 s | +0.65 s | +0.39 s | +0.65 s |
| unknown du'a, a known one shown (9 items) | 36.0% | 36.0% | 39.0% | 39.0% |

Bars (any one gain; every guard): word exact +2 pts, on line +1 pt, or found within 10 s* +2 pts.

- **W, TW**: no gain, and most guards fail (line lag median +0.26 s, p90 +0.67 s; right word -8 pts
  in every lane; Hasan's lane on line 81% -> 68%).
- **T**: no gain bar reached (on line +0.8 pt, word +0.5, found +1.9). Five guards fail:
  - majlis stays put -1.1 pt (bar -1);
  - user lane stays put 100% -> 96.3% (one move, in one pause item);
  - user lane wrong du'a 1.33% -> 3.05%;
  - unknown du'as shown 36.0% -> 39.0%;
  - user lane stays put under the absolute 97%.

No candidate passed dev, so none was run on test (the turbo test pass is cached for later:
`data/testbed/asr/whisper-turbo-dua`).

## What the numbers say

**The wav2vec2 model is worse at the edge of the window.** The word follower reads the newest
audio, where a word is still being said. The student was trained on clips that end mid-word; the
wav2vec2 model wasn't. phone_follower.md measured the same thing on streaming windows (letter error
on ordinary voices 78.0% for wav2vec2 against 40.1% for the student). Here it shows as lines entered
0.26 s later at the median and words placed less often.

**Turbo hears noisy rooms much better, and costs a little elsewhere.** By cell:

| | base + student | T |
|---|---:|---:|
| hall: found within 10 s* | 65% | 92% |
| hall: on line | 79% | 86% |
| far: found within 10 s* | 78% | 85% |
| babble: found within 10 s* | 86% | 95% |
| majlis lane: on line | 93.1% | 94.3% |
| harvest lane: on line | 88.7% | 90.6% |

Its wrong-du'a time in Hasan's sessions is all in his two Namaz-e-Wahshat recordings (ticks 212 ->
486 of ~14,600): Wahshat reads Ayat al-Kursi word for word as other texts do (jumps.md, the
shared-passage problem). Turbo was not trained on the synthetic ordinary voices (synthetic voices
cut the base model's error by a quarter), and the tracker was tuned on the base model's
transcripts.

Limits: Hasan's lane is one voice (35 dev items from a few sessions); the unknown-du'a cell is 9
items. The bench's student frames give the encoder 1 s of silence before each 2 s window
(server_engine.md), the wav2vec2 frames are the 2 s alone, as the server serves them.

## Sources

Results `data/testbed/results/srv_{base,w,t,tw}_dev.json`; guard and report output, latency and the
run log in `data/testbed/logs/night/`. `bench.py asr` gained `--split`, and writes each output
under a temporary name first, so a stopped pass leaves no half-written file a rerun would skip.
