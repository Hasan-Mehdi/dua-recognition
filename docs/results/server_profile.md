# The server engine's own models and settings (2026-10-07)

Hasan (2026-10-06, 23:20): "make the cloud model(s) as good as physically possible ... 24 hours."

The server engine runs the page's own code with Whisper and the CTC model on the server
(server_engine.md). Everything here is judged on the scenario bench (`bench.py score --display
stream --same-text 8`) at the server's delays (`--delay 0.15 --follow-delay 0.05`), tuned on the
dev voices (a fixed subset first: every 4th dev item by id, 344 items), then all dev items, then
test once. Bars were written down before the full runs (`data/cache/cloud/prereg.md`).

## Where the server loses, measured first

**Better CTC frames don't help on clean audio.** The stream decoder was given the wav2vec2
teacher's frames computed with the whole recording as context (so they even hear the future)
instead of the student's (as the bench had them: 2 s windows padded with 1 s of silence): on
line 90.5% -> 90.0%, right word 76.1% -> 75.0%, jumps
+0.16 /10 min (dev, all 1,373 items, `ceil_w2vfull_dev`). In the hall cell no word gain either: the
teacher isn't robust to echo. A 300 M wav2vec2 trained for streaming (crops ending anywhere, as
the student) also lost to the 20 M student on the same validation crops (2 s crops: letter error
34.3% against 25.7%; divergence from the teacher on the frames the decoder commits 0.103 against
0.075), so it was stopped after 6,000 steps.

**Most wrong-word time is a steady delay.** Per reading tick, the display shows the previous
word 12.2% of the time. Shown 0.1 s earlier it would be on the right word 82.9% of the time
instead of 76.5%; 0.2 s earlier, 81.5% (subset). A word starting inside a line reaches the
screen a median 0.17 s after its true start (p10 0.12, p90 0.24 s, 23,790 onsets), the same for
every first letter: the step rate, the round trip and a few frames of the CTC model's own
delay. The truth itself is not early: the forced-aligned first word of a line comes a median
0.30 s after the DuaPlayer annotators' line start (5,179 lines, 39 recordings).

**Lines that repeat word for word freeze the highlight.** Dua Tawassul's block "anā tawajjahnā
... yā wajīhan ʿinda-llāh" comes back after each of its 14 names. A reader who starts inside
it can't be placed until the next name; the decoder followed the words correctly but its
probability sat equally on 13-14 copies (~0.075 each), none reached the share needed to move the
highlight, and it stood still for 40 s (dev item `salawat-c04cc32174`). 113 of the 522 texts
have lines that repeat word for word.

## What changed

The server engine now has a profile: settings the server hands the page with `/api/mode`
(`app/server.py` `server_profile()`, `web/app.js` `DeviceEngine(..., profile)`; URL `?sc=`, `?tc=`
and `?ctchop=` still win). The phone's defaults are unchanged. Every new decoder rule below is a
`StreamConfig` setting, off by default, in `stream_follower.py` and `stream-follower.js` alike
(`tests/test_stream_parity.py` has a case for each).

**A bigger Whisper, trained in full.** `whisper-turbo-srv`: whisper-turbo-dua (the LoRA fine-tune
of large-v3-turbo) trained in full for one epoch at an 8 s context (`finetune_whisper.py
--context 8 --optim adamw8bit`: 8-bit AdamW fits the 809 M weights on 16 GB), on train_v4, 67k
harvest windows from 2,555 uploaders and 15% synthetic ordinary voices, with room reverb (RT60
0.3-1.5 s), synthetic halls through a PA (25%), speed, VTLP and SpecAugment. Validation letter
error 23.9% at the cut context before training, 9.3% after. A second round
(`whisper-turbo-srv2`: one more epoch at a lower rate, 5e-6, on train_v4 and 139k harvest windows,
up to 200 per uploader, `build_clip_cache.py --name h3c`) brought validation loss 0.160 -> 0.140
and left letter error at 9.6%; on the bench it equals round 1 overall and helps where a guard
failed (Hasan's sessions: wrong du'a 1.51% -> 1.01%; masjid on line 86.5% -> 89.5%), so the server
uses it. A bench pass runs at 56x real time (the 30 s turbo: 11x). Note: transformers loads
whisper-turbo-dua in float16 (its config's dtype); the fine-tune now asks for float32 master
weights, and it can stream batches from worker processes (`--workers`).

**CTC frames on 3 s windows, every 0.05 s.** The server can afford it (the student takes ~4 ms
per window on the GPU). The phone sends the student 2 s windows (its `-w2` export, to save
compute), but the student was trained on 3 s crops; the bench's old frames hid this by padding
each 2 s window with 1 s of silence to 3 s. Real 2 s windows made line entries slow where lines
open alike (Mujeer, Baha, 'Asharat: p90 1.09 s against 0.80 s); real 3 s windows fixed it and
placed words better (subset: right word 80.0% -> 82.0%; Mujeer/Baha/'Asharat line entry p90
0.65 s). 4 s windows broke the student (right word 38.8%): it never saw them. The decoder's
confirmations count steps, so they're set in steps of 0.05 s (`hop` 0.05, `lineSteps` 3,
`nextSteps` 2, `nextP` 0.7). Found on the way: each step committed `round(dt / 20 ms)` frames,
and 0.05 s is 2.5 frames: Python's rounding dropped a frame every other step and JavaScript's
doubled one. Committed frames are now counted from the session's start in both (`_f_stream`),
which also stops drift on a phone whose steps come at uneven times.

**Repeated lines move the highlight together (`copies`).** Lines with the same words are pooled
for the display: each word is weighed by its probability summed over its copies, and the copy
shown is the one with the highest, or, while none holds half of it, the one nearest the word on
screen. The next-line gate still weighs each line by its own probability.

**A salawat doesn't lose the place (`lapseIntj`).** When the tracker loses the du'a, the decoder
used to start over after 4 s of voice; steps it puts at least half in its salawat chain no longer
count. A 10 s salawat in Dua Baha had sent the display back to the du'a's first line.

**The frames can move the highlight's probability (`ctcPushP`, `ctcSure`, `cFar`).** After a reader jumps far,
the frames name the new line within a second (median 0.9 s), but the probability never got there:
entering a far line costs `cFar` (-13 nats) spread over all line starts, and the tracker needed
a median 5.7 s. Now, when one line beats every other but its neighbours by 4 nats, a fifth of the
probability moves to it; far entries cost -8.

**A shared passage doesn't change the du'a (`sharedWords`).** While the tracker's last 4 words in
another du'a are the same letters as those ending near the word on screen, the du'a doesn't
change. Ayat al-Kursi sits word for word in Namaz-e-Wahshat, Sahifa 54 and an Eid al-Mubahila
text; with the better ear the tracker flipped to another of them inside the passage and the
display followed (Hasan's sessions). The hold starts only once the du'a has been on screen for
10 s (`sharedAfter`): without that, a short shared phrase held a wrong du'a after a fresh
mid-Iftitah start.

**Tracker for turbo's transcripts**: `kappa` 0.2 (from 0.15) and `nullRate` 0.35 (from 0.45):
a better ear makes a wrong text at 0.45 edits per letter look likelier than it is, and unknown
du'as were shown more.

## Results

The final server engine: `whisper-turbo-srv2` (two rounds of the full fine-tune), the phone's
student `ctc-student-base-v6` on 3 s windows every 0.05 s, and the profile in `SERVER_PROFILE`
(`app/server.py`; `data/cache/cloud/profile_D.json` is the same minus the 3 s window and `nextP`).
"Found within 10 s*" clocks from the first 8 words no other text reads (finding.md).
`bench.py score --display stream --same-text 8 --delay 0.15 --follow-delay 0.05`, the CTC frames
from `DUA_BENCH_CTC_WINDOW=3 DUA_BENCH_CTC_HOP=0.05`; today's server engine is `srv_base_*`, the
new one `full_pF2_*`.

| | dev, today | dev, new | **test, today** | **test, new** |
|---|---:|---:|---:|---:|
| items | 1,373 | 1,373 | 617 | 617 |
| on the reader's line | 90.5% | 92.7% | 92.5% | **93.7%** |
| right word | 76.1% | 81.6% | 79.0% | **83.4%** |
| du'a found within 10 s* | 94.4% | 96.7% | 96.9% | **98.6%** |
| jumps /10 min | 1.27 | 1.26 | 1.51 | 1.70 |
| early moves /10 min | 1.71 | 1.79 | 1.82 | 1.84 |
| lost /10 min | 1.85 | 1.11 | 1.36 | **0.82** |
| go-back, skip, jump followed within 3 s | 77% | 82% | 80% | **84%** |
| holds still through pauses, talk, salawat | 88.1% | 90.1% | 91.2% | 91.4% |
| wrong du'a shown | 0.42% | 0.34% | 0.38% | 0.46% |
| unknown du'a: a known one shown | 36% | 33% | 49% | **37%** |
| line-change lag, median / p90 | +0.39 / +0.71 s | +0.35 / +0.72 s | +0.38 / +0.65 s | **+0.31** / +0.68 s |

By lane, on line / right word (dev; test): studio 88.2 -> 90.9 / 74.3 -> 80.4 (90.5 -> 91.7 /
80.8 -> 84.9); harvest 88.7 -> 92.1 / 73.6 -> 80.1 (90.1 -> 92.6 / 73.3 -> 79.8); Mafatih 96.1 ->
96.3 / 83.0 -> 86.4 (95.4 -> 95.7 / 83.7 -> 86.6); majlis 93.1 -> 94.1 / 75.5 -> 79.6 (94.1 -> 93.5
/ 77.1 -> 81.0); Hasan's sessions (dev only) 81.4 -> 86.2 / 54.4 -> 58.4, wrong du'a 1.33 -> 1.01%.

By cell (dev, today -> new), the largest changes:

| dev | on line | right word | found ≤10 s* | lost /10 min |
|---|---|---|---|---|
| hall (synthetic) | 78.9 -> 90.2% | 54.3 -> 70.7% | 65 -> 92% | 5.92 -> 1.93 |
| far from the phone | 84.0 -> 91.1% | 62.5 -> 76.5% | 78 -> 93% | 2.65 -> 1.08 |
| salawat between lines | 72.7 -> 86.4% | 61.2 -> 76.4% | 95 -> 95% | 7.09 -> 1.51 |
| jumping around the du'a | 61.2 -> 74.2% | 49.3 -> 62.7% | 95 -> 97% | 14.29 -> 8.12 |
| plain reading | 95.4 -> 95.8% | 82.1 -> 85.9% | 97 -> 97% | 0.39 -> 0.27 |

(These five rows are the round-1 turbo, `full_pF_dev`; round 2 is within 0.3 pt of them overall.)

**Measured halls and a masjid** (noise_bench.py `rhall`, `masjid`: rooms measured with a
microphone, none used in training; the masjid adds a PA, 1-3 loudspeakers and people talking
12 dB under the voice):

| | on line | right word | found ≤10 s* | lost /10 min |
|---|---|---|---|---|
| masjid, dev: today -> new | 67.4 -> **89.5%** | 49.8 -> **81.3%** | 72.5 -> **90.0%** | 5.49 -> 2.56 |
| masjid, test: today -> new | 56.7 -> **80.7%** | 41.1 -> **73.8%** | 56.2 -> **81.2%** | 3.49 -> 1.74 |
| measured halls, dev | 90.8 -> 93.4% | 72.3 -> 84.0% | 80.5 -> 95.1% | 1.00 -> 0.63 |
| measured halls, test | 91.1 -> 94.7% | 73.0 -> 84.5% | 93.3 -> 100% | 0.39 -> 0.00 |

(40 / 16 masjid items and 41 / 15 hall items, dev / test.)

**The real page**, server engine, the same 20 test items as `pg_server` (server_engine.md),
headless Chrome in real time (`page_check.py` in the session's scratchpad: `bench_page.run_one`
on those items, `page_replay.mjs --server`):

| real page, 20 test items | server engine before | now |
|---|---:|---:|
| on the reader's line | 89.4% | **90.9%** |
| right word | 77.5% | **81.8%** |
| jumps /10 min | 1.86 | 1.60 |
| early moves /10 min | 1.33 | **0.53** |
| lost /10 min | 1.86 | **0.80** |
| go-back etc. followed within 3 s | 69% | **81%** |
| holds still | 75.8% | **88.3%** |
| du'a found within 10 s* | 95% | **100%** |
| wrong du'a shown | 0.51% | 1.05% |
| unknown du'a shown (1 item) | 40% | **17%** |
| line-change lag, median | +0.33 s | **+0.30 s** |

On the page the CTC steps came every 0.05 s, the student's round trip 6 ms at the median (8-24 ms
p90), Whisper's 74-88 ms, the decoder ~1 ms a step: within the pre-registered bar (CTC step at
most 0.1 s). This PC: the server and Chrome on one machine, no network between them.

### Against the bars

The pre-registered gains pass on dev (right word +5.5, on line +2.2, found +2.3) and on test
(right word +4.4, on line +1.2; found +1.7 misses its +2). Guards (`bench.py guard`):
- dev: two fail. Du'a al-'Asharat line-change lag p90 0.55 -> 0.72 s (bar +0.15 s; 7 items); majlis
  wrong du'a 0.19% -> 0.29% (bar +0.1 pt, missed by a rounding hair).
- test: six fail. Jumps rise 1.51 -> 1.70 /10 min overall (no guard on the total); Baha line-change
  p90 0.59 -> 0.87 s; holding still in the studio (87.2 -> 85.4%) and harvest lanes (89.0 -> 87.7%);
  and the majlis lane: on line 94.1 -> 93.5%, lost 0.87 -> 1.47, wrong du'a 0.66 -> 1.50%. The majlis
  losses are one recording (KSIJ Dar es Salaam, Iftitah) in its seven scenario items: each starts
  at Iftitah's "as'aluka qalīlan min kathīr ... wa-ghināka ʿanhu qadīm", which duas.org's Ramadan Day
  16 text reads word for word. The display took the Day 16 text first and, ten seconds on, the
  shared-passage hold kept it until the passage ended. On the page the one wrong-du'a item is
  the same passage. Without the hold, dev shows nothing better and Hasan's Namaz-e-Wahshat goes
  back to 2.08% wrong du'a (holding still 96.2%); a hold only after words unique to the shown
  du'a never fires, since Namaz-e-Wahshat itself is made of passages other texts share. Both
  are starts inside a shared passage; the hold stays, as dev chose.

The test voices were looked at in the jump work (jumps.md) and once more here: the Iftitah
item above was read after the test run, and the variants that followed (no hold, a hold on
unique words) were decided on dev.

## Tried and not adopted

- **A 300 M wav2vec2 CTC model made streaming** (`train_server_ctc.py --arch w2v`, the teacher
  trained on crops ending anywhere): after 6,000 steps it still lost to the 20 M student on the
  same validation crops (above), and the full-context ceiling had shown no headroom on clean
  audio, so it was stopped. The trainer also takes w2v-BERT 2.0 and Whisper encoders
  (`server_ctc.py`); neither was trained to the end.
- **The hall-trained student** (`ctc-student-base-v6-hall`, noise.md) with the new turbo and the
  profile: measured halls and masjid unchanged (right word 80.0% against 79.9%), holding still
  1-2 points lower in the studio, harvest and Hasan's lanes. The server keeps the phone's student.
- **4 s CTC windows** for the 3 s-trained student: right word 38.8% (subset). A student fine-tuned
  on 4 s crops (`ctc-student-base-v6-w4`, 4 epochs from v6) on 4 s windows tied v6 on 3 s (right
  word 82.0% both, subset); not needed.

- **Peak-first CTC student** (targets shifted 3 frames, 60 ms, so letters come out sooner;
  `train_ctc_student.py --shift 3`, 4 epochs from v6 with hall augmentation): no word gain
  (76.3% against 76.5%) and lines entered later (+0.45 s against +0.39 s).
- **Causal WPE in front of both models** (`halls.wpe_online`, cached passes): the bench hall
  78.9% -> 87.7% on line, but plain reading 95.4% -> 69.0%.
- **Holding the du'a through any 4-word overlap** (`sharedWords` 4 without `sharedAfter`): fixed
  Hasan's Namaz-e-Wahshat but kept a wrong text after a fresh start inside a short shared phrase
  (majlis wrong du'a 0.28% -> 0.55%, subset); 8 words never triggered (one spelling differs inside
  Ayat al-Kursi between the texts). The hold now needs the du'a on screen for 10 s.
- **A slower du'a change** (`switchS` 3 s): Hasan's lane 1.51% -> 0.73% wrong du'a, majlis 0.31%
  -> 0.37%.
- **Next-line gate for 0.05 s steps** (`nextMargin` 2.5, `nextSlack` 1.5-2): no change to the slow
  line entries, which came from the 2 s windows (above).
- **Decoder thresholds**: a lower show threshold (0.25) changed nothing; a shorter commit
  lookahead (0.1 s, 0.15 s) and a sharper temperature were worse; one-step line confirmation and
  a shorter next-line hold cut the lag 0.03-0.06 s for +0.14-0.21 jumps and +0.26-0.39 early
  moves per 10 minutes.

## Limits

- A lower `nullRate` (0.30) showed unknown du'as less on dev (33% -> 27%, 9 items) for 0.2 points of
  on line and right word; not taken.
- Whisper every 0.5 s instead of every 1 s was not tried: it doubles the server's GPU time per
  listener for what a faster tracker could add (finding sooner; the decoder now follows far jumps
  from the frames).

- One machine: the server and Chrome ran on the same PC, so the page's round trips have no
  network in them. Over a network each CTC step waits a round trip; the page asks for the next
  window only when the last one is answered, so at 100 ms the steps would come every ~0.1 s, not
  0.05 s (the decoder counts committed frames from the session's start, so uneven steps don't
  drift).
- Hasan's lane is one voice (35 dev items from six recordings); the majlis lane is three venues;
  the measured-room cells are 15-41 items each.
- The profile was tuned on the dev subset and dev, with today's student; the phone keeps its
  defaults. Copies and the salawat-aware lapse also helped at the phone's delays (subset: on line
  +0.7, right word +0.5, lost -0.18) and the student's 2 s windows look like a loss on the phone too;
  neither was changed there.
- Disk: the bench's 3 s / 0.05 s CTC caches take ~70 GB on D: (dev + test + halls).

## Files and commands

- Code: `src/dua_recognition/stream_follower.py` and `web/stream-follower.js` (copies, `lapse_intj`,
  `ctc_push_p`/`ctc_sure`, `shared_words`/`shared_near`/`shared_after`, committed-frame count),
  `tests/test_stream_parity.py` (a case for each), `app/server.py` (`SERVER_ASR`, `EAR_CTC_WINDOW`,
  `SERVER_PROFILE`, `server_profile()`), `web/app.js` (`DeviceEngine(..., profile)`),
  `tests/test_server_ear.py`, `scripts/bench.py` (`DUA_BENCH_CTC_WINDOW`, `DUA_BENCH_CTC_HOP`,
  `--same-within`), `scripts/finetune_whisper.py` (`--optim adamw8bit`, `--workers`, `--eval-every`,
  `--max-steps`, float32 master weights), `scripts/train_ctc_student.py` (`--shift`),
  `scripts/train_server_ctc.py` and `src/dua_recognition/server_ctc.py` (bigger CTC models).
  `finetune_whisper.py` and `train_ctc_student.py` import `scripts/halls.py` (from noise.md, not yet
  committed): it goes in with these.
- Models: `models/whisper-turbo-srv2` (+ `-ct2`), `models/whisper-turbo-srv` (round 1), on D: via
  junctions; `models/ctc-student-base-v6-w4`, `-s3h` (not used).
- Results: `data/testbed/results/{srv_base,full_pF2}_{dev,test}.json`,
  `noise_{srv_base,pF2}_{dev,test}.json`, `ceil_w2vfull_dev.json`, the dev-subset runs `sub_*`,
  `tsub_*`, `tsrv_*`, `fx_*`, `w3_*`; page runs `data/testbed/page/pg_srv_F2` (+ `.json`). Logs
  `data/testbed/logs/cloud/`; pre-registration `data/cache/cloud/prereg.md`.
- Score the server engine as it now runs:

```
DUA_BENCH_CTC_WINDOW=3 DUA_BENCH_CTC_HOP=0.05 python scripts/bench.py score --name X   --asr whisper-turbo-srv2 --display stream --split dev --same-text 8 --delay 0.15 --follow-delay 0.05   --tracker kappa=0.2,null_rate=0.35 --sc hop=0.05,line_steps=3,next_steps=2,next_p=0.7,copies=true,lapse_intj=0.5,ctc_every=10,ctc_push_p=0.2,ctc_sure=4.0,c_far=-8,shared_words=4,shared_after=10
```

(The frames come from `bench.py asr` with `DUA_BENCH_CTC_WINDOW=3 DUA_BENCH_CTC_HOP=0.05` set, or
the scratch `ctc_pass.py`; the Whisper rows from `bench.py asr --model models/whisper-turbo-srv2-ct2`.)
