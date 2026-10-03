# When the reciter stops, the page should too (2026-09-30)

Hasan, on the phone (session 20260930-214600-h649, Ayat al-Kursi): "fast at recognizing,
but when I stopped reciting it jumped forward 4-5 words ahead." He stopped on *bi-idhnih*,
the last word of line 4, at 21.3 s. From 25 s until he pressed stop, the page showed line 5
with *wa-mā*, its fifth word, highlighted: five words past where he was.

## What happened

Three things added up, and a fourth kept them from being undone.

1. **The update after he stopped predicted he'd go on.** The phone shows each update 1.0-2.5 s
   after its window ends, led by that delay plus 0.25 s (display_lead.md). The window ending
   at 21.5 s still ended in speech, so the lead carried the display into line 5, although
   by the time the update was shown (22.5 s) he had been silent for over a second.
2. **The glide ran on.** Between updates the highlight moves at the reciter's pace (1.8
   words/s here). The next update took 2.5 s, and the highlight ran from word 0 to word 4.
   Within a line it never steps back.
3. **The belief moved for the whole gap.** That update covered 2.5 s. A breath at 24.3 s made
   the window's "seconds silent" 0.06, so the tracker moved the belief 2.5 s worth of words
   forward (to word 3 of line 5), and the retreat rule, which needs a second of silence,
   never came.
4. **The stop detector heard the phone's gain control as recitation.** After he stopped,
   the phone's automatic gain control raised the room hiss from -44 to about -33 dBFS.
   `asr.quiet_at_end` counts a frame as sound if it is 6 dB over the window's quietest
   tenth, and the rising hiss was. For the ten seconds after he stopped it reported 0-1.5 s
   of silence, so neither the hold nor the retreat ever applied.

`scripts/session_stops.py` rebuilds the page 20 times a second from a session's log (the
tracker over each update the page received, when it received it, then the glide) and
reproduces the page's logged word changes exactly (100% on h649). It measures what was on
screen after each stop of at least 1.5 s (Silero over the whole recording) against the last
word said (fine-tuned turbo on the window ending at the stop).

## The stop detector

**Energy counts only near the reciter's voice.** A frame now counts as sound if Silero calls
it speech, or if it is 6 dB over the floor *and* within 15 dB of the reciter's voice level:
the median frame Silero is sure is speech, from at least 16 such frames. The voice level is
remembered from window to window (`asr.QuietMeter`; the page hands it back to the worker
each time), because a window after a stop holds only hiss, and at most a breath or a click
that Silero takes for a few frames of speech at -32 dBFS. A long melodic note is loud; the
hiss sits 20-25 dB under the voice.

Measured on 17,079 mid-word moments of the DuaPlayer reciters (forced-aligned word timings,
where the reciter is sounding) and on every quarter second of Hasan's silences:

| stop detector | reciter mid-word, called silent (train / test) | Hasan silent 0.5-1 s / 1-2 s / 2 s+, called sound |
|---|---|---|
| before: floor + 6 dB | 1.27% / 4.01% | 26.0% / 27.0% / 9.2% |
| voice level of each window alone | 2.11% / 6.81% | 7.8% / 5.7% / 2.6% |
| **remembered voice level, within 15 dB** | **1.27% / 4.04%** | **8.5% / 5.7% / 3.5%** |
| ...within 20 dB | 1.27% / 4.01% | 8.9% / 7.3% / 4.6% |
| Silero alone | 7.2% / 14.4% | 2.0% / 1.5% / 3.9% |

The threshold hardly matters to the reciters (8 dB: 4.07%); 15 dB is where Hasan's two
newest sessions came right (h649 missed 57% of its silence before, under 5% after). What is
left is mostly Silero itself hearing speech in a silence: background voices, and a session
recorded with the TV on (excluded above).

**The page listens too.** The window's quiet is a second or two old by the time its update
is shown. The page holds the audio that arrived since, so it runs its own detector on it
(`gate.js LiveQuiet`, `asr.LiveQuiet`): a 32 ms frame is sound if it comes within 15 dB of
the voice level the worker last measured, 3 frames make a run. Silero is busy in the worker,
so this is loudness alone, and without Silero beside it the floor term hurt:

| live detector | reciter mid-word, called silent (train / test) | Hasan silent 0.3-0.6 s / 0.6-1 s / 1-2 s / 2 s+, called sound |
|---|---|---|
| floor + 6 dB and within 15 dB of the voice | 5.4% / 11.9% | 10.5% / 7.0% / 4.4% / 4.2% |
| **within 15 dB of the voice** | **0.32% / 0.50%** | 11.8% / 7.5% / 5.0% / 4.3% |

## The rules

- **The page's own ear decides "silent now".** `quiet_now` (tracker.py, tracker.js) is the
  live quiet when the page shows an update. When the page has it, it decides whether the
  reciter is silent *now*: for holding still, for letting the lead cross into the next line,
  and for stepping back. The window's quiet still decides whether its transcript has
  anything new.
  - The window ended in speech but they have been silent since (past `still_after`, 0.3 s):
    the lead covers only the part of the delay they were still reciting.
  - Past `retreat_after` (1 s) a display that ran into the next line steps back at once, to
    the end of the line the evidence is on, as the retreat always did.
  - The window ended in a pause they have since recited on from: no reason to hold.
  - The lead may cross into the next line only while the page hears sound. A gap between
    words isn't a stop, so the bar is `still_after`, as for the glide, not the window's 0.1 s.
- **The belief moves only for time spent reciting** (`pause_motion`): every pause of 0.3 s or
  more since the last update is taken off (`QuietMeter.paused`), not only the one at the
  window's end. A breath three seconds after Hasan stopped made the window's trailing quiet
  0.06 s, and the belief, and with it the evidence, walked on into the next line.
- **The glide stops the moment the page hears the reciter stop**, and restarts when they go
  on (`Highlight.pace`; logged as `hush` events).
- **A fix in the glide.** Easing onto a word from below could land on 2.9999999999999996 and
  show the word before. It never mattered while the glide kept moving; once it stops, it did.

Measured and left off (`TrackerConfig`, tracker.js, parity-tested):
- `still_catch_up`: while silent, the display may still move forward to the evidence. It cut
  the time a stopped display rests a word short (25.6% to 20.2% of stop time in Hasan's
  sessions) and changed nothing on DuaPlayer; left off as not needed to pass.
- `retreat_in_line` with the glide's `back`: a highlight that ran ahead in its own line steps
  back after a second of silence. More jerks, no gain.
- A cap on how long the glide runs past an update, and holding the lead's line crossing on
  the live quiet at 0.15 s: no effect on the sessions once the glide stops on the page's ear.

## Results

**Hasan's device sessions**: 14 stops, from 1.5 s after each stop until he went on (the
Python tracker and the glide over each session's own updates):

| | words ahead of the last word said | ≥ 2 ahead | next line on screen | before it | stops ever ≥ 3 ahead | line, while reciting | steps back / flickers |
|---|---:|---:|---:|---:|---:|---:|---:|
| as the phone showed it | -0.84 | 16.2% | 20.5% | 25.9% | 4/14 | 65.4% | 8 / 6 |
| today's settings, before | +0.99 | 23.5% | 25.2% | 5.0% | 6/14 | 78.5% | 4 / 2 |
| **after** | **-0.12** | **1.9%** | **1.9%** | 25.6% | **2/14** | 76.0% | **2 / 1** |
| after, without pause_motion | -0.12 | 1.9% | 4.3% | 27.3% | 2/14 | 77.5% | 2 / 2 |

("As the phone showed it" includes the older settings of the earlier sessions.) On h649
itself: five words ahead for 8 s before; after, the highlight stays on *bi-idhnih*. The two
stops still ≥ 3 ahead are the same under every variant: one is the page following a misheard
repeat of a line, the other (21 words ahead within a 1 s stop) looks like a measurement
artifact and wasn't looked into. The price: the display rests a word short of where they
stopped more often (the transcript's last word arrives a window late, and the lead no longer
guesses it), and line accuracy while reciting is 2.5 points lower, all at breaths between
lines, where the display now keeps the line just finished until they start the next one.
The reference (turbo plus the offline smoother) counts a breath as the next line's.

The real page, headless (`page_replay.mjs`, h649's audio, Whisper held to 1.1 s per update
as on the phone): in two runs it showed line 5's first word for about a second at the moment
he stopped, then *bi-idhnih* to the end. Without `pause_motion` it ended on line 5.

**DuaPlayer, phone model** (whisper-base-syn-v5-ctx8ft), train split, the stop detector and
the live quiet both as the page runs them (data/cache/quiet2):

| | 0.5 s latency: line / ±1 / lag / jumps/min | 1.2 s: line / ±1 / lag / jumps/min |
|---|---|---|
| before | 82.9% / 96.8% / 0.9 s / 0.26 | 84.5% / 96.7% / 0.6 s / 0.35 |
| **after** | **83.8% / 96.7% / 0.7 s / 0.19** | **84.2% / 96.6% / 0.6 s / 0.24** |
| after, without pause_motion | 84.1% / 96.7% / 0.8 s / 0.17 | 84.5% / 96.7% / 0.7 s / 0.22 |

Wrong du'a 0.3% throughout.

**The pause set** (train: 4 s pauses after every third line, 773 of them), as before with
the recording's own room tone, and with that tone turned up by 12 dB over 1.8 s as a phone's
gain control does:

| | flowing: word exact / ±1 | in pauses: next line shown / ≥ 2 words ahead | with gain control: next line / ≥ 2 ahead |
|---|---|---|---|
| 0.5 s latency, before | 34.2% / 80.2% | 27.7% / 18.0% | 56.9% / 32.0% |
| **after** | **37.9% / 82.7%** | **19.0% / 8.5%** | **29.9% / 13.2%** |
| 1.2 s latency, before | 29.8% / 75.3% | 56.1% / 37.3% | 73.0% / 46.9% |
| **after** | **34.0% / 79.1%** | **28.3% / 14.1%** | **40.0% / 16.5%** |

Once they went on after a plain pause, the next line came 0.1-0.8 s sooner. After a
turned-up pause it came 0.2-0.3 s later, because before, the display had never waited at all.

**One look at test** (pre-registered criteria in data/cache/reliability/stops-20260930/prereg.md,
written before any result; the chosen rules were fixed before this ran):

| test split | 0.5 s: line / ±1 / lag / jumps/min | 1.2 s: line / ±1 / lag / jumps/min |
|---|---|---|
| DuaPlayer, before | 84.6% / 96.4% / 1.0 s / 0.22 | 85.5% / 96.4% / 0.7 s / 0.52 |
| **DuaPlayer, after** | **84.9% / 96.4% / 0.8 s / 0.19** | **85.7% / 96.4% / 0.7 s / 0.17** |

| pause set, test (463 pauses) | flowing: word exact / ±1 | in pauses: next line / ≥ 2 ahead | with gain control: next line / ≥ 2 ahead |
|---|---|---|---|
| 0.5 s, before | 35.7% / 84.3% | 24.8% / 11.8% | 43.8% / 19.4% |
| **0.5 s, after** | **40.5% / 87.0%** | **12.4% / 5.9%** | **26.7% / 8.5%** |
| 1.2 s, before | 31.0% / 79.6% | 53.3% / 30.0% | 63.2% / 35.4% |
| **1.2 s, after** | **36.7% / 84.2%** | **21.3% / 8.6%** | **39.2% / 15.2%** |

Wrong du'a 0.4% before and after. Passed on every criterion; the rules are the defaults in
tracker.py and tracker.js since 2026-10-01.

How the choice was made: on train every variant passed; the pre-registration picked the
smallest set (the live quiet without `pause_motion`). Before any test result was read, the
real page replaying h649 with that set drifted into line 5 after a breath three seconds into
the stop, which the pause set's breathless room tone can't show; with `pause_motion` it held.
The A-vs-C test jobs were stopped unfinished and their files deleted unread, and the test
look was run on this set instead.

## Not done

- The display rests a word short of the word they stopped on about a quarter of the time:
  the transcript's last word comes a window late. `still_catch_up` narrows it.
- Silero itself sometimes hears speech in a silence (a television, someone talking nearby);
  the stop detector can't tell those from the reciter.
- 14 stops from one voice on one phone. More sessions, other phones and children will say
  whether 15 dB is the right distance from the voice.

## Reproduce

```
python scripts/session_stops.py [-v] -V "after: quiet=speech live=1 paused=1"   # Hasan's device sessions
python scripts/dump_quiet.py --v2                                           # DuaPlayer: data/cache/quiet2
python scripts/evaluate.py --quick --split train --asr whisper-base-syn-v5-ctx8ft --latency 1.2 --stop2 live paused
python scripts/pause_eval.py quiet2 --tag whisper-base-aug-v4 --split train [--agc 12]
python scripts/pause_eval.py score --tag whisper-base-aug-v4 --split train "base;old" "base;q2;live;paused" "base;q2;live;paused;agc"
node scripts/page_replay.mjs data/sessions/20260930-214600-h649.wav out.wav "" --asr-ms 1100   # the real page, headless
```

Pre-registration and job outputs: `data/cache/reliability/stops-20260930/`.
