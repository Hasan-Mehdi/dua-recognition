# The phone engine, as close to the server as it gets (2026-10-07)

Hasan (2026-10-07, 12:10): "make the phone model as close to the cloud model as physically possible,
and get the noisy part (masjid, halls, background noise) working better."

Bars were written down before any candidate was scored on all dev items (`data/cache/phone/prereg.md`).
Tuning used a fixed subset (every 4th dev item by id, 344 items); the chosen configuration was then
scored on all dev items, checked with `bench.py guard`, scored once on test, and played through the
real page.

## Where the phone loses: its compute, more than its models

The bench scored "the phone" with Whisper every 1 s shown 1.2 s late, a CTC step every 0.1 s shown
0.15 s late, and 2 s CTC windows padded with 1 s of silence to the 3 s the student was trained on.
Hasan's phone (Galaxy Z Flip 6, sustained, session `20261006-225939-pu3k`) does less: Whisper takes
1.73 s a window, so it updates every 2 s and is 1.8 s late; the CTC model takes 234 ms a step on
real 2 s windows, so a step comes every 0.25 s and is shown 0.25 s after its audio
([phone_power.md](phone_power.md) has the slowdown: both models get 2-3x slower within a minute).

`bench.py` can now replay that cadence: `DUA_BENCH_ASR_EVERY=2` keeps every 2nd Whisper window,
`DUA_BENCH_CTC_STRIDE=2` every 2nd CTC step (both 1 by default; the tracker's cache key includes
the first). Dev subset, today's models and decoder:

| dev subset (344 items) | on line | right word | found ≤10 s* | line lag |
|---|---:|---:|---:|---:|
| the bench's phone (`psub_base`) | 89% | 69% | 93% | +0.49 s |
| real 2 s windows (`ph_r1`) | 89% | 68% | 93% | +0.51 s |
| **as the phone runs it** (`ph_r2`) | **88%** | **58%** | **80%** | **+0.70 s** |
| ...but CTC every 0.1 s, shown 0.05-0.1 s late (`ph_r5`) | 88% | 73% | 88% | +0.42 s |
| ...and Whisper every 1 s (`ph_r3`) | 90% | 74% | 93% | +0.42 s |
| server engine today (`full_pF2_dev`, all dev) | 93% | 82% | 97% | +0.35 s |

The CTC step rate is worth 15 points of right word on its own; Whisper's 2 s cadence costs du'a
finding (93% vs 88%), and its lateness (1.2 s vs 1.8 s) little. The real page at the same speeds
(`page_check.py`, Whisper held to 1.73 s, the CTC model to 234 ms) agrees: on line 85.2%, right word
56.1%, found 79%, line lag +0.73 s on 20 test items (`pg_ph_today`). (The bench's ticks are 0.1 s
apart, so a result 0.05 s and 0.1 s late score the same.)

## What changed

### A streaming CTC student

`src/dua_recognition/stream_ctc.py`. The window student re-encodes the latest 2 s every step:
each frame is computed twenty times. The stream student has the same Whisper-base encoder and
letter head, with attention a stream can cache:

- layer 0 sees 0.2 s ahead (the right context the stream decoder already waits for before it
  commits a frame) and 3 s back; layers 1-5 see only the past, 3 s back, so their keys and
  values never change once computed;
- no absolute positions: a learnt per-head distance penalty on the attention scores (ALiBi);
- log-mel floored at the running maximum of the last 3 s, not the window's.

A frame is final once 0.2 s after it has arrived; the newest 0.2 s are tentative (computed with
the future there is, not cached), as the window model's newest frames are. The stream decoder
reads it unchanged: the worker hands back the latest 2 s of frames, final then tentative.

- One step on one desktop thread (ONNX int8): **2.9 ms** against 13.3 ms for the 2 s window model
  and 20 ms for the 3 s one. In headless Chrome, the page's own worker: **7 ms** against 28 ms. On
  Hasan's phone that would be ~60 ms a step instead of 234 ms (not measured on the phone).
- Trained from the window student (v6) for 8 epochs on 5 s crops, with the hall, crowd and phone
  augmentations (`train_ctc_student.py --arch stream`). Whole 6 s validation windows: 23.1% letter
  error (the window student, trained on 3 s: 47.6% on 6 s windows).
- Exact: the cached step equals the whole-audio forward to 5e-6 (`tests/test_ctc_stream.py`), and
  the page's cache keeping (`web/ctc-stream.js`) equals its Python mirror step for step in Node.
  The int8 graph picks the same best letter as PyTorch on 99.5% of final frames.
- The page starts it with the microphone (steps of 0.3 s until the du'a is found), so its 3 s of
  context are there when the follower needs them; a slow-phone fallback isn't needed for it.
- The bench stores its frames compactly (final frames once, tentative ones per step: 6.7x smaller,
  the same windows to the bit).

At equal cadence it matches the window model (dev subset, the bench's phone delays: on line 90% vs
89%, right word 68% vs 68%, bench hall right word +6), with slightly more display moves (jumps
+0.12, early +0.18 per 10 min). Its gain is what the phone can afford: a step every 0.1 s.

### A distilled phone Whisper

`whisper-base-ph-kd2`: the phone's Whisper-base (8 s context) trained two more epochs (one at
1.5e-5, then one from it at 7e-6; the first run's second epoch hung at its data loader after a
low-memory stop and was killed) on
train_h3c (train_v4 and 139k harvest windows from 2,555 uploaders) with synthetic ordinary voices,
rooms (RT60 0.3-1.5 s) and halls through a PA with crowds (30%, half of them real people talking),
speed, VTLP and SpecAugment, and **distilled from the server's whisper-turbo-srv2**: the teacher
hears the clean window, the student the augmented one, and half the loss is the KL between their
token distributions (`finetune_whisper.py --teacher --kd 0.5`; large-v3's special tokens sit one id
higher from 50358, so they are mapped). Validation letter error on clean windows 13.5% before,
13.9% after: the gain is in hard audio, not clean.

### Decoder settings for the phone

The server profile's decoder rules (server_profile.md) at the phone's cadence, plus three new ones.
They are now `StreamConfig`'s defaults (py and js); the server engine's profile pins its own values
of the new ones (`sharedNear` 8, `sharedEdits` 0, `cFillNext` -1, `nextIntj` off), not yet scored
with its models.

- `copies`, `lapse_intj` 0.5, the frame push (`ctc_push_p` 0.2, `ctc_sure` 4, `c_far` -8) and the
  shared-passage hold (`shared_words` 4, `shared_after` 10) as on the server;
- **`shared_near` 400, `shared_edits` 2** (new, py+js+parity): the hold looks for the tracker's last
  words anywhere in the shown du'a, and tolerates two letters' difference. Namaz-e-Wahshat spells
  *yaʾūduhu* ييوده where Eid al-Mubahila spells it يوده, so inside Ayat al-Kursi the exact check
  failed and the display switched du'a (Hasan's sessions: wrong du'a 3.5% -> 1.0% with the
  distilled Whisper);
- **`c_fill_next` -2** (from -1): talk between lines is less often taken for the next line's start
  (talk cell early moves -0.76 /10 min, no cost elsewhere).
- **`next_intj`**: the salawat chain as one more alternative in the next-line gate (below).
- Lapses now count the seconds of voice between steps, not a fixed 0.1 s a step (on a slow phone
  the lapse came 2.5x late).

### Holding still through a salawat at 0.1 s steps

At the phone's old 0.25 s steps the display had time to hear the second word of a salawat. At 0.1 s
it committed to the next line on the first: Du'a Baha's lines open "allāhumma innī as'aluka ...",
the salawat "allāhumma ṣalli ...". The studio lane held still 85.2% -> 81.6% (salawat cell 80% ->
72%). Window frames at the same cadence held still better (salawat 81%), so the stream student is
eagerer at the live edge too. What was tried, on all dev items of the salawat, talk, pause and combo
cells: a third confirming step (+1), costlier salawat-to-next-line (no change), no frame push or the
old salawat lapse (no change), a 0.4 s next-line hold (+2, right word -1, lag +0.07 s), training the
stream student with someone talking at the live edge (on 30% or 10% of crops: early moves -0.3 to
-0.4 /10 min but go-backs followed less often, -2 to -9 points, and Hasan's sessions worse), and
**`next_intj`** (new, py+js+parity): the next-line gate counts the salawat chain as one more line
the frames could be reading, its belief with its entry cost given back as the other lines get
theirs, so a line that opens as the salawat does waits for the word where they part: salawat
holding still +4, early moves -0.8 there, lag +0.01 s. Adopted. Together with the 0.4 s hold it
passed the studio bar, but Hasan's sessions held still 98.6% -> 93.6%: not taken.

## Results

The final configuration: `whisper-base-ph-kd2`, `ctc-stream-base-v1` every 0.1 s, the decoder's new
defaults (above, with `next_intj`). Scored at the phone's cadence (Whisper every 2 s, 1.8 s late;
the stream every 0.1 s, 0.1 s late) against the phone as it ran this morning at its cadence (window
student every 0.2 s, 0.25 s late; the old defaults), `--display stream --same-text 8`.

| | dev, phone today | **dev, new** | test, phone today | **test, new** | test, server engine |
|---|---:|---:|---:|---:|---:|
| items | 1,373 | 1,373 | 617 | 617 | 617 |
| on the reader's line | 87.6% | **90.0%** | 89.4% | **90.5%** | 93.7% |
| right word | 56.5% | **74.3%** | 59.7% | **75.7%** | 83.4% |
| du'a found within 10 s* | 78.2% | **93.1%** | 82.9% | **93.9%** | 98.6% |
| lost /10 min | 1.64 | **1.15** | 1.21 | **1.01** | 0.82 |
| go-back, skip, jump followed within 3 s | 75% | **82%** | 78% | **84%** | 84% |
| line-change lag, median | +0.70 s | **+0.41 s** | +0.69 s | **+0.41 s** | +0.31 s |
| jumps /10 min | 1.09 | 1.21 | 1.23 | 1.49 | 1.70 |
| early moves /10 min | 1.55 | 1.81 | 1.65 | 1.88 | 1.84 |
| holds still through pauses, talk, salawat | 88.9% | 88.8% | 92.0% | 91.1% | 91.4% |
| wrong du'a shown | 0.43% | 0.40% | 0.33% | 0.61% | 0.46% |
| unknown du'a: a known one shown | 27% | 25% | 29% | 36% | 37% |

Sources: `full_ph_r2_{dev,test}`, `full_ph_G1_{dev,test}`, `full_pF2_test` (server_profile.md, at the
server's delays). The gap to the server engine on test: on line 4.3 -> 3.2 points, right word 23.7
-> 7.7, found 15.7 -> 4.7.

Dev cells, phone today -> new (on line / right word / found ≤10 s*): plain reading 93/62/81 ->
94/79/93; bench hall 76/44/43 -> 88/68/80; far 81/50/59 -> 86/67/78; people talking 87/55/73 ->
89/73/84; combo 84/50/73 -> 89/69/96; salawat 81/52/81 -> 84/70/90; jumping around 55/33/87 ->
73/58/95.

**Measured halls and a masjid** (noise_bench `rhall`, `masjid`; `nph_r2_*` -> `nph_G1_*`):

| | on line | right word | found ≤10 s* | the server engine (test) |
|---|---|---|---|---|
| masjid, dev (40 items) | 65 -> **78%** | 42 -> **67%** | 48 -> **70%** | |
| masjid, test (16) | 55 -> **67%** | 37 -> **57%** | 38 -> **56%** | 81 / 74 / 81% |
| measured halls, dev (41) | 88 -> **91%** | 57 -> **76%** | 63 -> **85%** | |
| measured halls, test (15) | 88 -> **90%** | 54 -> **74%** | 73 -> **93%** | 95 / 85 / 100% |

**The real page** at the phone's speed (Whisper held to 1.73 s a window, the CTC model to 234 ms a
step for today's window student and 60 ms for the stream student), the 20 test items of the
device-page runs (`page_check.py --like pg_ph_today`):

| real page, 20 test items | today (`pg_ph_today`) | **new (`pg_ph_G1`)** |
|---|---:|---:|
| on the reader's line | 85.2% | **86.4%** |
| right word | 56.1% | **71.8%** |
| lost /10 min | 2.66 | **1.33** |
| go-back etc. followed within 3 s | 62% | **81%** |
| holds still | 76.2% | **88.3%** |
| du'a found within 10 s* | 79% | **84%** |
| line-change lag, median | +0.73 s | **+0.38 s** |
| jumps /10 min | 1.60 | 2.40 |
| early moves /10 min | 0.80 | 1.33 |
| wrong du'a shown | 0.95% | 1.78% |
| unknown du'a shown (1 item) | 31% | 39% |

The stream student's step in that run: 6-8 ms (the window student's: 28 ms) before the floor.

### Against the bars

- Gains (dev, all three needed only one): right word +17.8 (bar +4), on line +2.4 (bar +1), found
  +14.9 (bar +3): pass. Test: +16.0, +1.1, +11.0.
- Equal cadence (the stream student must hold up against the window model, both on the bench's
  phone delays, decoder defaults of the time, all dev): on line 89.5% vs 89.3%, right word 67.6% vs
  67.3% (`full_ph_s1_dev` vs `full_ph_r1_dev`): pass.
- The real page: on line and right word not worse than today's: pass.
- `bench.py guard`, dev: every bar but two passes (lag, follows, wrong du'a, on line, right word,
  lost per lane, the stationary cells' jumps and early moves, unknown du'as). Fails: holding still
  in the studio lane 85.2% -> 83.4% (bar -1 point), and the absolute bar for Hasan's sessions (on
  line at least 78%: 69.7% -> 76.6%; their wrong du'a 2.77% -> 0.59%, holding still 97.5% -> 98.6%).
- Guard, test: fails holding still in the studio (90.5% -> 88.0%) and Mafatih (95.0% -> 92.7%)
  lanes; wrong du'a in the studio (0.37% -> 0.50%), harvest (0.66% -> 0.91%) and majlis lanes
  (0.28% -> 2.17%); majlis on line (-0.7) and lost; unknown du'as shown 29% -> 36%. The majlis
  losses are one recording (KSIJ Dar es Salaam, Du'a al-Iftitah) in its seven scenario items: it
  starts inside a passage duas.org's Ramadan Day 16 text reads word for word, the display takes Day
  16 first, and the shared-passage hold, looking further now, keeps it longer (server_profile.md
  has the same item). On the real page the one wrong-du'a item is the same passage.

The test voices were looked at in the jump work and the server push; here test was scored once, after
the configuration was fixed on dev.

## Tried and not adopted

- Tracker `kappa` 0.2, `null_rate` 0.40 for the distilled Whisper (the server's turbo settings):
  no change to Hasan's Namaz-e-Wahshat flips; found -1, unknown du'as -4.
- For the stream student's extra early moves: next-line margin 4 (no change), judging the next line
  on committed frames only (word -1, lag +0.10 s), a softer temperature 1.2 (talk early +0.95),
  cheaper filler (`fill_cost` 0.7: jumping-around follows -11%), a longer next-line hold 0.5 s
  (early -0.35 but word -1, lag +0.09 s).
- Stream students trained with someone talking at the live edge (`train_ctc_student.py --talk`,
  targets blank after the cut): `ctc-stream-base-v2` (30% of crops, 3 epochs from v1) and `-v3`
  (10%, 2 epochs). Fewer early moves (-0.40 and -0.32 /10 min, dev subset) but go-backs followed less
  often (-9 and -2 points), more lost, and Hasan's sessions worse (v3: on line -6 on 12 items).
- The distilled Whisper's first epoch (`whisper-base-ph-kd`): the second, at half the learning rate,
  was better everywhere it differed (found +1, measured-hall and masjid found +2 to +5).

## How it was run

```bash
# the phone's Whisper, distilled from the server's (two runs: --base the first's output, --lr 7e-6)
python scripts/finetune_whisper.py --base models/whisper-base-syn-v5-ctx8ft --name whisper-base-ph-kd \
    --data h3c --synth 0.15 --context 8 --epochs 2 --lr 1.5e-5 --warmup 300 --batch 32 --room 0.5 \
    --room-rt60 0.3,1.5 --hall 0.3 --hall-voices 0.5 --speed 0.5 --vtlp 0.5 --specaug --workers 5 \
    --eval-every 1500 --teacher models/whisper-turbo-srv2 --kd 0.5
.venv-export/Scripts/python scripts/export_onnx.py models/whisper-base-ph-kd2 --gpu-encoder
# the streaming student
python scripts/train_ctc_student.py --arch stream --window 5 --name ctc-stream-base-v1 \
    --init models/ctc-student-base-v6 --synth synth_v5,synth_v6,cv_ar,cv_ar_test --teacher ctc_student_voices \
    --workers 3 --epochs 8 --hall 0.3 --hall-voices 0.5 --phone 0.3
python scripts/export_stream_ctc.py models/ctc-stream-base-v1      # web/models/, checks int8 against PyTorch
# bench frames (bench.py asr writes the compact cache for a streaming model) and the phone's cadence
python scripts/bench.py asr --split dev
DUA_BENCH_ASR_EVERY=2 python scripts/bench.py score --name full_ph_G1_dev --display stream --split dev \
    --same-text 8 --delay 1.8 --follow-delay 0.1
# the phone as it ran before (window student every 0.2 s, its real 2 s windows: tag ctc-student-base-v6u)
DUA_BENCH_ASR_EVERY=2 DUA_BENCH_CTC_STRIDE=2 python scripts/bench.py score --name full_ph_r2_dev \
    --display stream --split dev --same-text 8 --asr whisper-base-syn-v5-ctx8ft --ctc ctc-student-base-v6u \
    --delay 1.8 --follow-delay 0.25 --sc hop=0.2,copies=false,lapse_intj=2,ctc_push_p=0,c_far=-13,shared_words=0,c_fill_next=-1,next_intj=false
```

(`full_ph_r2_*` were scored before the defaults changed, with `--sc hop=0.2` alone; the settings
spelled out above are the old defaults.) The real-page runs: the job's `page_check.py` (bench_page's
`run_one` on a fixed item list, `--asr-ms 1730 --ctc-ms 234` / `60`). Pre-registration
`data/cache/phone/prereg.md`; logs `data/testbed/logs/phone/`.

## Limits

- Every phone number is emulated from one phone (Hasan's Flip 6) in its throttled state. The stream
  model's ~60 ms step on that phone is scaled from Chrome on this desktop (7 ms vs 28 ms), not
  measured, and Whisper is assumed as slow as today (1.73 s) though the CTC model now takes a
  quarter of the CPU it did.
- ASR every 2 s applies from the first window; a phone still in its fast first minute runs Whisper
  every second while finding the du'a, so du'a finding is somewhat better than these numbers.
- The masjid and measured-hall cells are 40-41 dev and 15-16 test items over ~22 voices; the
  rooms are measured, but none is a masjid.
