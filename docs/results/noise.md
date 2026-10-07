# A phone in a masjid hall (2026-10-06)

Hasan: "how are we going to figure out the lower accuracy part which is in a masjid hall listening
to speaker ... surely there's test files we can use already. And try all of your hypotheses first."

Bars were written before any candidate was scored (`data/cache/noise/prereg.md`). Everything here
is dev voices, today's defaults, phone delays (`score --display stream --same-text 8`); test only
for what passes.

## Summary (dev voices; nothing adopted, test not run)

| measured halls + masjid, pooled | on line | right word | found ≤10 s* |
|---|---:|---:|---:|
| today | 77.4% | 54.2% | 69% |
| the du'a known in advance (no model change) | 88.9% | 61.5% | (given) |
| offline WPE in front of today's models | 81.0% | 58.5% | 80% |
| both | 91.1% | 65.2% | (given) |
| hall-trained CTC student | 79.0% | 59.0% | 69% |
| hall-trained student + Whisper | 80.2% | 59.7% | 70% |
| tracker null_rate 0.65 (rejected: unknown du'as shown 35 -> 81%) | 80.9% | 55.9% | 78% |

The masjid cell alone goes 66% -> 80% on line with the du'a known, 83% with WPE too. The largest
loss in it is people talking in the same hall. Causal WPE (what a phone can run) and a second
training round with real voices as the crowd were stopped by a low-memory stop before they were
scored (see the end).

## The test files

The scenario bench already reverberates its held-out test recordings (`bench.py`: cells `hall`,
RT60 1.4 s at -4 dB direct-to-reverberant, and `far`). Its response is synthetic: a noise tail
with the direct sound on top, the same family the models were trained on (at 0.5 s only). So two
cells were added from rooms measured with a microphone (OpenSLR 28's real responses, none used in
any training), over the same sources (26 studio runs, 30 harvest uploaders):

- `rhall`: a large room alone: Aula Carolina (T20 ~4.5 s), RWCP E2B (~2 s), a lecture hall, the
  REVERB challenge's large room, RWCP E1C;
- `masjid`: the reciter through a PA (band, compressor, saturation), 1-3 loudspeakers 5-60 ms apart
  in one of those rooms, people talking 12 dB under the voice in the same room, a fan at 22 dB.

`scripts/noise_bench.py` builds them in `data/testbed/items_noise/` (apart from the reading grid,
so `bench.py score` without cell names is unchanged) and runs bench.py's own model pass and scorer
on them. `scripts/halls.py` holds the effects.

## Today

| dev | on line | right word | du'a found ≤10 s* | lost /10 min |
|---|---:|---:|---:|---:|
| plain reading (`flow`) | 95% | 74% | 94% | 0.4 |
| room (0.5 s) | 92% | 71% | 90% | 0.9 |
| people talking (`babble`) | 90% | 68% | 86% | 1.2 |
| bench hall | 77% | 48% | 65% | 6.1 |
| far | 82% | 56% | 76% | 3.2 |
| measured halls (`rhall`) | 89% | 65% | 73% | 1.5 |
| **masjid** | **66%** | **44%** | **65%** | **6.1** |

Source: `data/testbed/results/noise_base_dev.json` (761 items).

## Where it's lost

Both models lose it, each its own part:

- **Whisper** (finds the du'a and the line). Letter error on its 6 s windows: plain reading 26.6%,
  bench hall 41.8%, far 38.6%, measured halls 32.7%, masjid 46.3%. It doesn't go quiet in a hall:
  it writes wrong text at nearly full length, which the tracker takes as evidence. With Whisper's
  text replaced by the truth (the bench's `truthear` run), the hall's du'a found ≤10 s goes 57% ->
  92%. Turbo on the server gets 90% (server_models.md): its letter error rises only 23.5% -> 28.3%
  in the hall.
- **The CTC student** (places the word). Its greedy letter error on 2 s windows: plain 42.9%, bench
  hall 60.6%, measured halls 51.0%, masjid 62.9%. With perfect Whisper text the hall's right word
  is still 53% against 75% in plain reading.

Both models were trained with one kind of room: a 0.5 s noise tail with the direct sound on top
(`transcribe_windows.room`; `--room-rt60` was never set).

## Hypotheses

### Knowing the du'a in advance

A room host (or the reader) picks tonight's du'a; the display starts on its first line. Emulated
with the practice mode's pin (`score --practice`, PRACTICE_CFAR=-13 so jumps stay possible as
normal); today's models.

| dev | on line | right word |
|---|---|---|
| bench hall | 77% -> 92% | 48% -> 57% |
| far | 82% -> 94% | 56% -> 63% |
| measured halls | 89% -> 98% | 65% -> 71% |
| masjid | 66% -> 80% | 44% -> 52% |
| plain reading | 95% -> 100% | 74% -> 79% |

The largest gain of any hypothesis, and no model work. It assumes the app is opened as the du'a
starts (the first line is known). Source: `noise_pin_dev.json`.

### Dereverberation before the models (WPE)

Offline WPE (`halls.wpe`: the whole recording at once, 40 taps, delay 2, 512/256 STFT) in front of
today's models. It matches nara_wpe's output on the same input (-0.3 vs -0.3 dB).

| dev | on line | right word | found ≤10 s* |
|---|---|---|---|
| bench hall | 77% -> 87% | 48% -> 62% | 65% -> 85% |
| far | 82% -> 85% | 56% -> 61% | 76% -> 73% |
| measured halls | 89% -> 91% | 65% -> 68% | 73% -> 88% |
| masjid | 66% -> 71% | 44% -> 49% | 65% -> 72% |
| plain reading | 95% -> 95% | 74% -> 74% | 94% -> 94% |

Whisper's letter error with it: hall 41.8 -> 35.2%, measured halls 32.7 -> 30.1%, masjid 46.3 ->
43.7%, plain 26.6 -> 26.4%. Offline WPE sees the future; a phone has to run it causally.

Both together (du'a known, offline WPE), on line / right word: bench hall 98% / 70%, far 97% /
69%, measured halls 99% / 74%, masjid 83% / 57%, plain reading 100% / 79%
(`noise_pin_wpe_dev.json`). In the masjid cell the two add up: 66% -> 80% (du'a known) -> 83%
(and WPE) on line, 44% -> 52% -> 57% right word.

### The tracker's "none of these" state

The tracker explains a window as an unknown text misheard at `null_rate` 0.45 edits per letter.
In the hall Whisper misses 42% of letters, so the right du'a barely beats "none of these", and it
is found late. Raising `null_rate` finds it sooner and costs the unknown du'as:

| dev | 0.45 (today) | 0.55 | 0.65 |
|---|---|---|---|
| measured halls + masjid: on line / word / found ≤10 s* | 77.4% / 54.2% / 69% | 79.7% / 55.4% / 75% | 80.9% / 55.9% / 78% |
| bench hall: found ≤10 s* | 65% | 75% | 82% |
| masjid: on line | 65.7% | 70.3% | 72.0% |
| plain reading: on line | 94.6% | 94.8% | 95.0% |
| unknown du'a: a known one shown | 35% | 52% | 81% |

Not adopted: a fixed higher rate trades one failure for another. If it moves, it moves with the
hearing (a rate that rises when the audio is a hall), which nobody has built yet. Sources:
`noise_null055_dev.json`, `noise_null065_dev.json`.

### What in a masjid costs the most

The masjid cell taken apart: the same items, rooms and seeds, today's models
(`noise_bench.py variant`; `noise_mjparts_dev.json`):

| dev | on line | right word | found ≤10 s* |
|---|---:|---:|---:|
| loudspeakers in the hall (1-3, 5-60 ms apart) | 82.5% | 59.1% | 88% |
| + the PA chain | 77.1% | 56.2% | 75% |
| + people talking 12 dB under the voice, and a fan | 65.7% | 44.0% | 65% |

People talking in the same hall cost the most. Talk without the hall costs little (bench `babble`,
10 dB: 89.8% on line against 94.6% plain); in the hall the talk is smeared over the reciter's
pauses and words alike. By room, the longest (Aula Carolina, T20 ~4.5 s) is the worst: 41% on line
in the masjid cell, 43% with the du'a known (5 items).

### Training the models on halls (round 1)

One more stretch of training for each phone model, with `halls.hall_aug` on 40% of windows: a
synthetic hall (RT60 0.4-3 s, -10..+6 dB direct-to-reverberant, 1-4 loudspeakers 5-60 ms apart, a
PA chain on 60%), speech-shaped crowd noise at 5-25 dB in the same hall. Whisper: one epoch of
the ctx8ft recipe (`whisper-base-syn-v5-hall`). The CTC student: four epochs from v6
(`ctc-student-base-v6-hall`). Measured rooms were never used in training.

| dev | today | Whisper hall | student hall | both |
|---|---|---|---|---|
| measured halls + masjid | 77.4% / 54.2% / 69% | 78.2% / 54.5% / 70% | 79.0% / 59.0% / 69% | 80.2% / 59.7% / 70% |
| bench hall + far | 79.7% / 52.1% / 70% | 81.7% / 53.1% / 78% | 82.8% / 61.1% / 70% | 85.2% / 62.8% / 78% |
| masjid | 65.7% / 44.0% / 65% | 67.4% / 45.0% / 65% | 67.8% / 49.7% / 65% | 70.5% / 51.5% / 65% |
| plain reading | 94.6% / 74.4% / 94% | 94.6% / 74.3% / 94% | 94.6% / 74.4% / 94% | 94.6% / 74.4% / 94% |

(on line / right word / found ≤10 s*.) Room, babble, another recitation, fan and combo stay within
the guards (none more than 0.9 pt lower on line). The student carries the gain: its letter error
on measured halls 51.0 -> 47.2%, masjid 62.9 -> 55.6%, plain 42.9 -> 42.4%. One epoch of Whisper
moved the bench's synthetic hall (letter error 41.8 -> 37.4%, found 65 -> 80%) but not the
measured rooms (32.7 -> 31.1%) or the masjid (46.3 -> 46.0%): what it learnt is the training's own
kind of hall. Sources: `noise_{Wh,Sh,WhSh}_dev.json`.

**Controls** (the same extra training without halls: `whisper-base-syn-v5-ctrl1`,
`ctc-student-base-v6-ctrl4`), measured halls + masjid, on line / right word / found ≤10 s*:
Whisper 77.6% / 54.4% / 69% (hall-trained 78.2% / 54.5% / 70%); student 77.1% / 54.2% / 69%
(hall-trained 79.0% / 59.0% / 69%); both 77.4% / 54.4% / 69% (hall-trained 80.2% / 59.7% / 70%).
The student's gain is the halls'; the extra training alone gives nothing. On the bench's synthetic
hall the Whisper control finds the du'a ≤10 s 72% of the time, the hall-trained one 80% (65% today).
Sources: `noise_{Wc,Sc,WcSc}_dev.json`.

## Not finished

- **Causal WPE** (`halls.wpe_online`: recursive least squares, 20 taps, frame by frame): its model
  outputs for every noise cell are cached (`<cell>_owpe`, dev, 494 items); only the scorer is left
  (`noise_bench.py score <cells>_owpe --name noise_owpe_dev ...`, ~10 min CPU). It decides whether
  the offline gain survives on a phone. It also costs the phone CPU per frame (257 bins x a 20 x 20
  update every 16 ms); not measured on the phone.
- **Round 2** (real voices as the crowd, `--hall 0.5 --hall-voices 0.6`, `data/cache/run/noise_queue4.sh`):
  stopped 2026-10-06 18:55 with the rest when the machine ran low on memory; the student run had
  started (`models/ctc-student-base-v6-hallv` is incomplete).
- **Test voices, the full dev grid and `bench.py guard`, the real page**: none run. The hall-trained
  student is the only model candidate that passed a gain bar (word, measured halls + masjid); it
  needs those steps, and an ONNX export, before it could reach the phone.

## Limits

- No recording from a real masjid: the rooms are measured, but none is a masjid, and the PA chain,
  loudspeaker spread and the crowd's level (12 dB under the voice) are guesses.
- ~40 dev items per cell over ~22 voices; the per-room numbers are 3-11 items each.
- "The du'a known" pins the display to the du'a and the first line the reader starts on: someone
  who opens the app as the du'a starts. Joining late still needs the tracker to find the line.
- Offline WPE sees the whole recording; Whisper's training halls are the same synthetic family as
  the bench's `hall` cell, so that cell flatters them (the measured rooms don't).

## Files

`scripts/halls.py` (hall_rir, pa, crowd_noise, crowd_voices, hall_aug, wpe, wpe_online),
`scripts/noise_bench.py` (cells, front ends, variants; bench.py's asr and score),
`scripts/noise_report.py` (the bars, side by side), `--hall` / `--hall-voices` in
`finetune_whisper.py` and `train_ctc_student.py` (0 = unchanged). Measured responses:
`data/testbed/rirs/real` (OpenSLR 28, Apache 2.0). Queues and logs: `data/cache/run/noise_queue*.sh`,
`data/testbed/logs/noise/`. Pre-registration: `data/cache/noise/prereg.md`.
