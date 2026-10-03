# One bench for every way people read (2026-10-02/03)

Hasan: "I feel like each task I'm making you do is like fix this specific problem I'm having, and then
you're like okay we have reduced error rate by 40%. But then I try again and it's a different problem.
I want to start treating each issue holistically... I want literally every scenario tested... I want
it to be actually legitimately usable."

He was right about why. Every number we had reported came from 16 DuaPlayer recordings by 6
professional reciters, reading in order in clean audio; the pause, repeat and jump benchmarks were the
same 16 with gaps and skips spliced in. Our real-device tests were one voice (his) on two devices. The
word follower had grown to 37 settings, each tuned on its own benchmark, none checked against the others.

## The bench

`scripts/bench.py` builds one grid and scores the phone's display on all of it, in terms a reader would
notice. Data lives in `data/testbed` (a junction to `D:\dua-data\testbed`).

**Voices (lanes)**

| lane | what | voices | du'as |
|---|---|---:|---:|
| studio | DuaPlayer test reciters, human line times | 6 | 12 |
| majlis | du'a nights streamed from other centres (crowd, PA echo) | 4 | 5 |
| harvest | uploaders held out of all training (the harvest export's val side, sha1 % 100 < 3; listed in `test_voices.json`, which the export honours) | 69 | 72 |
| user | Hasan's own phone sessions | 1 | 2 |

Harvest truth is the CTC teacher's forced alignment, confident lines only; spot checks against
whisper-turbo-dua agreed with it where the display disagreed (Mujeer, Tawassul, Sahifa 54).

**Scenarios (24)**: as recorded (`flow`), starting mid-du'a, pauses (4 s, one 15 s), talking between
lines (another voice, 3-6 s), salawat between lines, repeating a line, going back 1-3 lines, skipping
1-4 lines ahead, jumping around the du'a, stumbling and restarting a line, switching du'a, a du'a the
corpus doesn't have, slow (0.7x) and fast (1.3x) reading; and conditions: room, big hall (RT60 1.4 s),
people talking (10 dB), fan, another recitation nearby (-10 dB), far from the phone, overdriven mic,
phone-call codec (AMR-NB), low-bitrate codec (Opus 12k), and a combination. Each item is a program over
source audio (spans, room tone, clips, effects), so the audio is rebuilt exactly; only model outputs
are cached. 1,465 items, 52 h, 84 voices. Voices are split by hash into **dev** (tune on it) and
**test** (report it).

**What is scored** (bars in `bench.BARS`): share of reading time the highlight is on the reader's line;
jumps (from the reader's line to one 2+ away, or another du'a, that the reader isn't on) and early moves
(to the next line before the reader) per 10 minutes; lost episodes (4+ s off the line) per 10 minutes;
wrong-place time; how often a go-back / skip / jump / repeat / restart is followed within 3 s; how much
of a pause, talk or salawat the highlight stays put; du'a found within 10 s; for an unknown du'a, how
often some du'a is on screen anyway; and the line-change lag.

```
python scripts/bench.py sources && python scripts/bench.py build     # sources, programs, truth
python scripts/bench.py asr                                          # GPU: Whisper rows, quiet, CTC frames (~55x real time)
python scripts/bench.py score --name x [--display follower|tracker|oracle|stream] [--split dev] [--sc k=v,..] --compare base
python scripts/bench_page.py --name page_x --query follower=stream  # a sample through the real page in Chrome
```

## Today's phone on the grid

Whisper (syn-v5-ctx8ft) + tracker for the du'a and line, the CTC word follower (ctc-student-base-v6, 2 s
windows every 0.1 s) for the highlight, phone delays. All 1,465 items:

| | on line | jumps /10 min | lost /10 min | follows ≤3 s | stays put | found ≤10 s | unknown du'a shown |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 83% | 3.1 | 4.3 | 58% | 63% | 85% | 16% |
| salawat between lines | 65% | 4.7 | 13.4 | | 28% | | |
| jumping around | 65% | 23.3 | 12.3 | 54% | | | |
| talking between lines | 78% | 3.6 | 7.9 | | 56% | | |
| big hall | 72% | 0.6 | 7.8 | | | 61% | |
| plain reading | 92% | 0.4 | 0.6 | | | 88% | |

Not one cell met every bar. Two ablations say where the failures come from:

- **Tracker alone** (Whisper + HMM, no follower): 71-72% on line, 2.1 s behind, follows 1-17% of moves.
- **Follower with a perfect tracker** (anchored on the true word): 96-99% in plain reading and every
  noise condition, but still 58% of go-backs followed and 9 jumps per 10 minutes when readers go back.

So in plain reading the follower is fine when the tracker is right, and the losses there are the
tracker's (wrong place, slow to find); when readers move around, the follower's own rules fail. One of
them explains the worst new finding: on refrain du'as (Mujeer, Tawassul, Sahifa 54) the display locked
onto the wrong repetition and stepped along with the reader for minutes, because `reset_stuck` (added on
2026-10-01 for one session, 9puq) lets the tracker correct the follower only while it stands still, and
a follower on the wrong repetition keeps hearing the refrain and never stands still.

## The stream decoder

`src/dua_recognition/stream_follower.py`, `web/stream-follower.js` (`?follower=stream`). Instead of
rescoring the last 2 s and keeping one position, it keeps a belief over every position in the du'a and
updates it frame by frame (a CTC forward pass over the du'a's letters), with reading as the transition
model: the text in order is free; starting the line again, going back 1-3 lines, skipping ahead, or a
jump anywhere cost a little; talk (a filler that explains any frame at a small cost per frame) and the
salawat (a short chain said from any word, left for the next line or the same line) are states of their
own. A pause is a blank. A refrain keeps every repetition alive until the words that differ are heard.
The tracker supplies the du'a, the starting belief and gentle evidence at each update.

What each piece bought, on dev voices (all scored on the grid, not on the case that motivated it):

| piece | why | effect |
|---|---|---|
| the decoder itself (untuned) | one model instead of 37 rules | follows +16 pts, lost -0.7/10 min, line lag 0.45 -> 0.30 s; early moves in pauses and talk up |
| letters cost 4 nats while the stop detector hears no voice | room tone makes stray letters, cheapest at the next line's start | on line +5, lost 4.4 -> 1.2 /10 min on the six affected cells |
| next line needs 0.6 of the belief for 2 steps | a stray letter or talk can't flip the line | early moves down; line lag +0.17 s (bought back below) |
| salawat chain | a salawat was a jump or talk | salawat stays put 49 -> 88%, jumps 8.3 -> 2.5 /10 min |
| restarts and go-backs cheaper (-3; -5, -6, -7) | readers do it more than the prior said | repeats followed 78 -> 84%, go-back jumps -0.9 |
| beam (12 lines of ~250-830) + the tracker's and the frames' line proposals | Kumayl took 47 ms a step, Abu Hamza 148 ms (desktop JS) | 3.5-4.9 ms a step; jumping around +3-4 pts |
| hold through a tracker lapse only while silent | an unknown du'a kept the last one on screen | unknown du'a shown 24 -> 14% (dev) |
| 1 nat off the blank column | ordinary voices and echo give faint letters; the belief rested in the blank after the last word | hall +14 pts, far +11, his voice: line lag 0.93 -> 0.55 s; salawat -17 pts (still +7 over today's phone) |
| a du'a change counts after 2 s | texts sharing a passage (Ayat al-Kursi in Namaz-e-Wahshat and Eid-e-Mubahila) flipped the tracker's du'a for a moment | his sessions: wrong du'a 3.5 -> 2.0%, on line 76 -> 78%; a real switch followed 2 s later |
| the filler takes blanks free, pays only on letters | most frames of any speech are blanks: at a flat cost per frame the du'a's own blank states beat the filler all through someone else's talk (it never held more than 2% of the belief) | talk: early moves 10.6 -> 7.7 /10 min, stays put 66 -> 77%; his sessions: jumps 3.8 -> 2.4, early 8.0 -> 5.9 /10 min, stays put 87 -> 98% |
| ...and leaves for anywhere a finished line could (restart, back, skip, far) | back from talk a reader may start anywhere; leaving only for the next line made jumped-to speech look like talk | jumping around: on line 55 -> 58% of the 65% before the filler change; kept at fill cost 1 (2.0 gives that back and loses the talk gains) |

Tried and dropped: weighing the tracker more (0.6, 1.0: no gain, more lost time); a "push" from the
tracker when the decoder stalls (no gain in echo, +3.6 jumps/10 min when readers go back); a lower
next-line threshold (lag -0.12 s but +1.3 early moves/10 min).

Parity: `tests/test_stream_parity.py` (JavaScript equals Python step for step on synthetic readings
with a go-back, a salawat, stray letters and a tracker lapse; the numba pass equals the numpy reference
with or without the beam).

## Result: held-out voices (test split)

Decoder (the defaults above) against today's phone, the same 405 items (799 minutes, 32 voices):

| | today's phone | stream decoder |
|---|---:|---:|
| on the reader's line | 83% | **87%** |
| right word | 61% | **67%** |
| jumps where the reader isn't /10 min | 3.4 | **2.4** |
| early moves /10 min | 3.5 | **2.5** |
| lost (4+ s off the line) /10 min | 4.2 | **1.8** |
| go-back / skip / jump followed within 3 s | 58% | **72%** |
| stays put through pause / talk / salawat | 69% | **88%** |
| line-change lag | 0.43 s | **0.41 s** |
| du'a found within 10 s | 80% | 79% |
| unknown du'a: some du'a shown | 21% | 28% |

By cell (23 with readings to score): time on the line is better in 16, equal in 5 and lower in 2
(starting mid-du'a, -1; jumping around the du'a, -7); lost time is lower or equal everywhere except
jumping around (+2.5 /10 min). Where it isn't better: jumping around the du'a (following within 3 s
-13 pts: the filler that keeps talk from moving the highlight also holds a far jump back a moment),
a few more false jumps in pauses (+0.7 /10 min), the unknown-du'a case (+7 pts), and finding the du'a
in some conditions (the 2 s du'a-change rule: stumble -6, hall -6, phone call -7 pts). Dev voices agree
(on line +4, lost -1.9 /10 min, follows +14, stays put +23, jumps -0.7). On Hasan's own sessions (35
items): on line 67 -> 78%, early moves 11.1 -> 5.9 /10 min, lost 7.7 -> 3.8, jumps 2.8 -> 2.4, follows
47 -> 85%, stays put 78 -> 98%, line lag 0.55 -> 0.51 s.

## Outside the follower

- **Firefox capture bug.** His Waritha and Kumayl sessions of 2026-10-02 (ryzl, df0y, wtrk; Firefox 157,
  Fractal Scape headset) delivered 2.00 s of audio per real second: chunks heard twice, the recitation
  slowed and stuttering, the CTC model nearly deaf (93% of voiced windows <= 2 letters). Firefox with a
  fake microphone (`scripts/capture_check.mjs firefox`) is fine, as was the same headset in Firefox 156,
  so it is that device in that browser. The page now measures audio seconds per real second
  (`checkClock`, logged as `clock`) and tells the reader past 15% off; `session_report.py` prints it.
- **Missing du'as.** As of 2026-10-03 02:30 the harvest has 1,214 recordings (145 h) of texts the app
  doesn't have: Du'a al-Sabah (194 recordings, 37 h, the 19th most-recorded text of 409), Du'a
  al-'Adeelah (68), Du'a al-'Asharat (51), 14 of the 15 Munajat (22-38 each). Their text is in
  `data/harvest/extra_texts.json` (Mafatih). Someone reciting them now gets nothing, or a wrong du'a.
- **fp16 on silence.** The CTC student under GPU autocast returns NaN for a window of digital zeros. It
  only touched the bench's GPU pass (7 items, recomputed in fp32); the phone runs fp32. The decoder
  treats a non-finite frame as no evidence anyway.

## Still failing the bars

- **Finding the du'a**: 80-88% within 10 s, everywhere. The tracker's job; the decoder starts only once
  the tracker has a du'a.
- **Echo and distance** (hall 77-82% on line, far 76-83%): both models hear little; a model problem
  (heavier reverberation in training), not a follower one.
- **Unknown du'as** shown 14-28% of the time; adding the missing popular du'as is the larger fix.
- **Talk between lines**: early moves 10-14 /10 min.
- **Jumping around the du'a**: 64-68% on line.
