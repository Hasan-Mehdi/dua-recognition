# Following an ordinary reader (2026-10-18/19, overnight)

On the bench's reciters the phone is on the reader's line 90% of the time and on their word 74%; on
Hasan's own sessions, 73% and 48% (`bench_sessions.py`, 35 items, phone cadence). This night went after
that gap.

## What the gap is

The page's letter model (`ctc-stream-base-v1`) on the session items, greedy over the final frames of
each item's timed span: 49% of letters wrong, 30 points of it letters left out. On 40 bench `flow`
items: 16%. (The items' truth is the teacher's own forced alignment.)

It is not the phone's microphone:

- Changing the audio before the model changes nothing: a high-pass at 150 or 300 Hz, pre-emphasis,
  a high-pass at 300 Hz with 6 dB more level, 10 dB less level: 49.1-50.1% on the sessions; on the
  bench items 16-17% (10 dB more level clips: 54%).
- 17 of the 35 items were recorded on a desktop, not the phone, and the model does no better there
  (53% of letters wrong, against 41% on the phone's 18).

What differs is the reading. The bench's voices are reciters drawing words out; Hasan reads at
speaking pace, with real silences between phrases, and says some letters the way many non-Arab readers
do. Line by line, the teacher (wav2vec2) still writes most of what he says; the student drops it:

| | Dua Hujjah (session x0fo) | Ayat al-Kursi (session l4wx) |
|---|---|---|
| text | فيهذهالساعهوفيكلساعه | لهمافيالسماواتومافيالارض |
| teacher | فيهارالسءوفيكلالس | لهمافيالسماواتومافيالعاد |
| student | فيهذلسوفيكلسع | ومافيالسماواتومافيالعد |

The letters he says alike are the ones many readers without Arabic as a first language merge: z
for ذ ض ظ, s for ث ص, t for ط, h for ح, a glottal stop or nothing for ع.

Two of the 37 session items were recorded with the microphone captured twice (the start-up bug fixed
on 2026-10-04: audio clock 2.00, every chunk doubled). They are garbled audio, not readings, and are
out of the session bench from tonight (`bench_sessions.py` now skips a session whose audio clock is
over 1.5; the two items are in `data/testbed/sessions/items_retired/double-capture/`). Every session
number below is on the 35.

## Letters said alike, in the decoder

`StreamConfig.accent_fold` (JS `accentFold`; 0 = off) with `accent_groups`: before the stream decoder
reads a step's letter frames, each letter's probability gains `accent_fold` times the probability of the
other letters in its group: ذ ز ض ظ, ث س ص, ت ط, ح ه, ا ع ء, ق ك. A "z" heard where the text has ض
then costs what a "z" heard for ز would, not what an unrelated letter would. Whisper's own transcripts
barely need it (they spell the text: window letter error 46.1% raw, 44.3% folded, `ph-kd2` on the
sessions), so the tracker is unchanged. Python and JavaScript step for step: three new cases in
`tests/test_stream_parity.py` (off, 0.3, and 1.0 inside Tawassul's repeated block).

Hasan's sessions (35 items, phone cadence, against the defaults):

| fold | on line | right word | holds still | early /10 min | lost /10 min |
|---|---:|---:|---:|---:|---:|
| off | 72.5% | 48.3% | 83.7% | 3.86 | 3.57 |
| 0.1 | 73.2% | 51.4% | 85.0% | 3.86 | 2.97 |
| 0.3 | 73.7% | 52.7% | 86.1% | 3.86 | 2.68 |
| 0.5 | 73.7% | 53.0% | 86.1% | 3.86 | 2.38 |
| 1.0 | 73.9% | 53.7% | 88.0% | 3.57 | 2.68 |

Right word at 1.0: +5.4 points [+3.8, +6.9] (paired over items, 95% bootstrap interval). One group
at a time (1.0): ا ع ء +2.9, ت ط +1.1, ح ه +0.8, ذ ز ض ظ +0.4, ث س ص +0.4, ق ك +0.3; adding غ to
ق ك changes nothing. With the server engine's models and profile on the same items, 1.0 gives
right word 60.7% -> 64.9% (+4.2 [+2.7, +5.6]), on line +0.9, early moves 4.16 -> 4.76 /10 min.

The bench's reciters say these letters apart, and there the fold costs holding still. All dev items
(1,317), 1.0 on every group (`s2_dev_fold1` against `s2_dev_base`, which equals 10-10's `taq_new_dev`
to the tick): on line 89.8% -> 89.8%, right word 74.4% -> 74.5%, early moves 1.87 -> 2.01 /10 min, and
five guard bars fail: holding still in the harvest lane (stays put 89.5% -> 87.3%) and in Hasan's dev
lane (98.6% -> 96.7%, which also misses its absolute bar of 97%), majlis lost 0.42 -> 0.52 /10 min, and
unknown du'a shown 25.04% -> 25.05% (one tick in 8,452). His dev lane gains the most: right word 46.1% -> 53.4%.

Folding merges the alphabet from 28 classes to fewer, so talk and translation between lines read a
little more like the text. Weaker folds on fewer groups, on a fixed quarter of the dev items
(`bench_sub.py`, 330 items):

| quarter of dev | on line | right word | holds still | early /10 min | harvest holds still | his right word |
|---|---:|---:|---:|---:|---:|---:|
| off | 90.2% | 74.4% | 89.0% | 1.82 | 89.3% | 44.9% |
| 0.3, all groups | 90.2% | 74.6% | 88.6% | 1.86 | 89.0% | 52.1% |
| 0.5, ا ع ء / ت ط / ح ه | 90.2% | 74.6% | 88.9% | 1.85 | 88.6% | 52.3% |
| 1.0, ا ع ء / ت ط / ح ه | 90.1% | 74.4% | 88.5% | 1.93 | 88.0% | 53.3% |
| 1.0, all groups | 90.1% | 74.5% | 88.1% | 2.02 | 88.1% | 54.3% |

("his right word": the 9 user-lane items in the quarter.) On his 35 sessions, 0.5 on the three groups
gives right word 52.4% (+4.1), on line 73.4%, holds still 83.7% (unchanged), early 4.16 /10 min.

On all dev items, 0.5 on ا ع ء / ت ط / ح ه (`s2_dev_f05top3`):

| dev, 1,317 items | on line | right word | holds still | early /10 min | lost /10 min |
|---|---:|---:|---:|---:|---:|
| off | 89.8% | 74.4% | 88.9% | 1.87 | 1.18 |
| 0.5, three groups | 89.9% | 74.5% | 89.1% | 1.92 | 1.16 |
| harvest lane: off / fold | 88.7% / 88.8% | 72.4% / 72.6% | 89.5% / 88.7% | 1.53 / 1.65 | 1.36 / 1.29 |
| his dev lane: off / fold | 76.6% / 77.8% | 46.1% / 52.1% | 98.6% / 96.7% | 5.22 / 6.27 | 1.74 / 1.74 |

Right word +0.2 [+0.1, +0.2] over all dev items. Guard (`bench.py guard`): harvest and majlis now pass;
two bars still fail, both his dev lane holding still (98.6% -> 96.7%, and its absolute bar of 97%):
two of its 35 items, a pause (ticks on the stopped word 101 -> 74 of 111) and someone talking (36 ->
13 of 36, one move); and unknown du'a shown, by one tick of 8,452 (2,116 -> 2,117). The jump work's
"gain" bars fail as they do for every change this size, and the absolute bar of 78% on his lane's line
fails for the baseline too (76.6% -> 77.8%).

On test (593 items, held-out voices; the test split has no ordinary readers, so this is only the
cost), the same fold (`s2_test_f05top3` against `s2_test_base`, equal to `taq_new_test`):

| test | on line | right word | holds still | early /10 min | lost /10 min | follows a jump ≤3 s |
|---|---:|---:|---:|---:|---:|---:|
| off | 90.5% | 75.9% | 90.9% | 1.90 | 1.01 | 73.0% |
| 0.5, three groups | 90.5% | 76.0% | 90.4% | 1.93 | 1.03 | 70.8% |

Four guard bars fail: studio stays put 88.0% -> 86.3% (three items: a salawat in Munajat Taibeen and
two talk items), majlis 90.0% -> 88.4% (two pauses, Iftitah and Aminallah), following a go-back or jump
within 3 s 73.0% -> 70.8% (the fold makes the line already shown explain the audio a little better: of
927 go-backs, skips, jumps, repeats and restarts, 76 are followed at another moment, 54 of them later,
median +0.1 s, 90% within +0.56 s; 5 cross the 3 s mark, 2 come back under it), and unknown du'a shown
+0.1 point.

Gentler still, on test and his sessions (against the same baselines):

| | his sessions: right word | test: holds still | test: studio holds still | test: majlis holds still | test: follows a jump ≤3 s | dev: his lane holds still |
|---|---:|---:|---:|---:|---:|---:|
| off | 48.3% | 90.9% | 88.0% | 90.0% | 73.0% | 98.6% |
| 0.5, ا ع ء only | 50.8% | 90.7% | 87.0% | 89.6% | | |
| 0.25, three groups | 52.0% | 90.5% | 86.3% | 89.8% | 72.3% | 91.4% |
| 0.5, three groups | 52.4% | 90.4% | 86.3% | 88.4% | 70.8% | 96.7% |

Every strength costs the same three studio items on test; below that, the holding-still counts move on
a handful of items each and not in order of strength (his dev lane holds still worse at 0.25 than at 0.5).

### Decision: on, 0.5 on ا ع ء / ت ط / ح ه (2026-10-19, 02:00)

The default since tonight (`StreamConfig.accent_fold` 0.5, `accent_groups` ("اعء", "تط", "حه"); JS
the same; `SERVER_PROFILE` doesn't pin it, so the server engine takes it too: below). It fails guard bars:
on dev, his lane's holding still (two items) and one tick of unknown du'a; on test, studio and majlis
holding still (five items), following a jump within 3 s (-2.2 points) and 17 ticks of unknown du'a
(7,742 -> 7,759 of 20,708). Taken anyway, because the bench has almost no ordinary readers (his dev
lane is 35 of 1,317 items, test none), and on the only ordinary reader we have the gain is large and
consistent: right word +4.1 [+2.7, +5.5] and on line +0.9 [+0.3, +1.6] on his 35 sessions, 30 items
better and none worse, and +6.0 on his dev lane; with the server engine's models +3.4 [+2.1, +4.6].
On the reciters it is a wash for words (+0.2 dev, +0.0 test) and costs holding still in a few items. `?sc=accentFold:0`
turns it off; `--sc accent_fold=0` on the bench.

The server engine (its Whisper, the window student on 3 s windows every 0.05 s, `SERVER_PROFILE`), dev,
fold against none (`s2_dev_sv_f05top3`, `s2_dev_sv_base`): on line 92.5% both, right word 81.5% both,
holds still 90.0% both, early 1.83 -> 1.88 /10 min; his dev lane right word 58.4% -> 62.4%; guard
fails on majlis holding still (94.9% -> 93.5%) and his lane's (100% -> 96.9%). Not scored on test.
Left unpinned, so the server engine has it too.

The real page (headless Chrome, `page_replay.mjs`, Whisper held to 1.2 s), one of his session items
(Dua Hujjah, 52 s), one play each: the highlight on his word 71% of its 451 steps with the fold, 63%
with `?sc=accentFold:0` (the Python bench, same item: 48% -> 60%; single page plays vary, so this
shows the JavaScript runs in the page and points the same way, not by how much).

New baselines with the fold on, for whatever comes next: `s2_dev_f05top3` and `s2_test_f05top3`
(phone cadence), `s2_sess_f05top3` (the 35 session items), `s2_dev_sv_f05top3` and `s2_sv_f05top3`
(server engine).

## A letter model that has heard more voices

Pre-registered before any candidate was scored (`data/cache/s2/prereg.md`): pass if, on Hasan's 35
session items at phone cadence, right word rises by 3.0 points or more with a paired 95% interval above
0, with no more than -0.5 on line and -1.0 right word on dev and no guard bar newly failing.

`ctc-stream-base-s2`: `ctc-stream-base-v1` trained 3 more epochs (16,272 steps of 48 five-second crops)
on its own sets plus the harvest's 138,614 six-second clips from 2,555 uploaders (the `train_h3c` clips the
phone Whisper was distilled on, 231 h; teacher frames from `wav2vec2-quran-dua-voices`, newly made, 9
to 13 minutes a third) and five sets of ordinary voices (`quranspeech` 3,627 clips, `recerrors` 944, `nahw`
1,093, `fleurs` 2,311, `sawtarabi` 655): 260,354 items, 85 minutes on the GPU while the CPU was busy
scoring. `train_ctc_student.py --add` (new) takes sets whole, `set*k` k times.

| | held-out reciters (val_v4): crop / 6 s letter error | his sessions: letter error | his sessions: on line | right word |
|---|---:|---:|---:|---:|
| `ctc-stream-base-v1` (the page's) | 33.6% / 23.2% | 47.5% | 72.5% | 48.3% |
| `ctc-stream-base-s2` | 33.0% / 22.7% | 45.6% | 72.6% | 49.3% |

Right word +1.0 [+0.1, +2.3]: **it does not pass**, and the page keeps `ctc-stream-base-v1`. With the
fold on both (0.5, three groups), s2 adds +0.6 [+0.0, +1.3] (52.4% -> 53.0%). A thousand more reciters
barely move a letter model on an ordinary reader.

On the bench it is a gain everywhere, though: dev, fold off on both (`s2_dev_s2` against `s2_dev_base`),
on line 89.8% -> 90.0% (+0.2 [+0.1, +0.2]), right word 74.4% -> 75.1% (+0.7 [+0.7, +0.8]), every lane up
(harvest +0.9, studio +0.7, majlis +0.8, mafatih +0.5, his lane +3.7), holds still 88.9% -> 89.1%, jumps
1.26 -> 1.22 and lost 1.18 -> 1.15 /10 min. One guard bar fails: his dev lane holds still 98.6% -> 94.6%
(on his 35 real sessions it holds still better, 83.7% -> 84.5%). With the fold on both (today's
defaults; `s2_dev_s2_f05top3` against `s2_dev_f05top3`) the same: on line +0.2 [+0.1, +0.2], right word
74.5% -> 75.2% (+0.7 [+0.6, +0.7]), lost 1.16 -> 1.12 /10 min; guard fails on holding still in his dev lane
(96.7% -> 87.8%) and Mafatih (93.8% -> 92.8%). On test, fold on both (`s2_test_s2_f05top3` against
`s2_test_f05top3`): on line 90.5% -> 90.7% (+0.2 [+0.1, +0.3]), right word 76.0% -> 76.7% (+0.7 [+0.6,
+0.9]), holds still 90.4% -> 91.4%, early 1.93 -> 1.80 and lost 1.03 -> 1.01 /10 min, jumps 1.49 -> 1.57;
**every guard bar passes**. The int8 export picks the same best letter as PyTorch on 99.8% of final frames
(`web/models/ctc-stream-base-s2`), and it runs in the real page with `?ctc=ctc-stream-base-s2`.

So s2 is a better letter model for reciters and no worse for him, but it was pre-registered as a fix for
his reading and missed that bar; as with h1 on 2026-10-02, it is not switched on after the fact. It is
Hasan's call: one constant (`CTC_MODEL` in web/app.js, `PHONE_MODELS` in app/server.py, `CTC_TAG` in
scripts/bench.py).

### His own sessions in training

`ctc-stream-base-s3a`: s2's recipe plus 277 six-second windows (2 s apart) of his sessions from before
2026-10-03 (19 items, `scripts/session_windows.py`; 9 minutes of distinct audio), each 20 times. Scored
on his 16 items from 2026-10-03 on, which it never heard:

| his later 16 items | letter error | on line | right word |
|---|---:|---:|---:|
| `ctc-stream-base-v1` | 47.4% | 72.7% | 50.5% |
| `ctc-stream-base-s2` | 46.2% | 72.8% | 51.0% |
| `ctc-stream-base-s3a` | 45.0% | 73.8% | 51.9% |

s3a against s2: right word +0.9 [-0.2, +2.5], on line +1.0 [+0.0, +2.6]; with the fold on both +0.3
[-0.6, +1.8]. On the 19 items it trained on, its letter error is 33.9% (s2: 44.9%) and right word +5.7: the
model can learn his voice, but nine minutes of it don't carry over to new sessions. What would help is hours
of ordinary reading, many readers. The debug sessions the page already uploads are that data, and the
pipeline now exists end to end: `bench_sessions.py build` (the teacher times the reading against the text),
`scripts/session_windows.py` (training windows), `ctc_teacher.py` (frames), `train_ctc_student.py --add`.
