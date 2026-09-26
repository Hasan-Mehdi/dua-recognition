# Pauses: the display should wait when the reciter does (2026-09-25)

A tester reading Ziyarat Waritha stopped after *"…wa-l-witra l-mawtūr"* (test3.mp4),
and the highlight carried on one, two, three words into the next line. Professional
reciters flow from line to line, so the benchmark never showed this. People reading
along stop all the time.

## Why it ran ahead

1. **The lead.** The display shows the belief predicted ~1 s past the window's end
   (display_lead.md), so at the moment someone finishes a line it is already
   predicting the next one. Nothing in the audio can say yet whether they'll go on.
2. **The window.** For 6 s after they stop, each window's transcript still ends with
   their last words. The prediction step moves the belief forward every second, and
   evidence that says "ends at *al-mawtūr*" only pulls back softly (κ = 0.15 per
   edit once locked), so the belief drifts on.
3. **The glide.** Between updates the highlight moves at the reciter's pace, whether
   or not they're still reciting.

## A benchmark for it

`scripts/pause_eval.py` inserts a 4 s pause after every third line end of every test
recording (463 pauses; train split: 772 for tuning). The pause is room tone copied from
the recording's own quietest second. The windows overlapping a pause are transcribed
afresh and the rest reuse the cache. The display is scored every 0.1 s against the
forced-aligned word being said (word_truth.py). During a pause, the right answer is
the last word said.

## Knowing when they've stopped

Silero VAD alone (the rule the VAD gate uses) calls long melodic notes silence.
Against the forced-aligned word timings, 19% of test reciters' mid-word moments read
as > 0.3 s quiet. `asr.quiet_at_end` now counts a 32 ms frame as sound if Silero says
speech **or** it is 6 dB above the window's floor (10th-percentile frame energy, and
never below -70 dBFS). That brings the false-silence rate to 4.2% (train: 8.8% to 1.4%).

The -70 dBFS clamp came from the tester's own recording. The headset (or the
browser's noise suppression) outputs exact zeros between words, so against a floor of
digital silence a faint click 0.7 s after they stopped counted as sound. The clamp
changes nothing on the reciter recordings.

## The rules (TrackerConfig)

- `still_after` = 0.3 s: while quiet, the display holds, and the highlight stops gliding.
- `still_motion_after` = 0.3 s: the belief itself moves only for the time they were
  making sound.
- `lead_cross_quiet` = 0.1 s: the lead may carry the display into the next line only
  while there is sound.
- `retreat_after`: once quiet this long, a display that already ran into a line the
  evidence hasn't reached steps back to the end of the line they stopped on. The lead
  crosses at the very moment they stop, before any silence can be heard, so this is
  the only way to undo it.

## Results

**Pauses** (test reciters, 463 inserted 4 s pauses, phone model, 0.5 s ASR delay).
The pause columns cover each pause from 0.5 s in; word accuracy covers everything
else. Chosen on the train split (772 pauses), where the same ordering held:

| rules | next line on screen in a pause | ≥ 2 words ahead | word exact | ±1 word | jerks/min |
|---|---|---|---|---|---|
| none (before) | 84.4% | 54.6% | 34.4% | 82.6% | 0.2 |
| step back after 0.5 s | 22.2% | 9.0% | 36.5% | 85.4% | 1.7 |
| **step back after 1.0 s (page mode)** | **32.9%** | **17.4%** | **36.9%** | **85.6%** | 1.3 |
| step back after 1.5 s | 35.7% | 19.8% | 37.0% | 85.6% | 1.1 |

**Flowing recitation** (the test recordings as they are; word level, 0.5 s delay):

| rules | word exact | ±1 word | jerks/min | words skipped/min |
|---|---|---|---|---|
| none (before) | 40.2% | 88.8% | 0.1 | 1.4 |
| hold + frozen belief only (majlis mode) | 39.6% | 88.5% | 0.1 | 1.4 |
| step back after 0.5 s | 37.5% | 87.0% | 1.3 | 3.3 |
| **step back after 1.0 s (page mode)** | **38.0%** | **87.4%** | **0.7** | 2.4 |
| step back after 1.5 s | 38.2% | 87.4% | 0.6 | 2.2 |

The step back has a price for reciters who breathe between lines. Abu Thar
al-Halawaji pauses over a second at about half his line ends, and each of those
now shows a step back and a step forward again. Hence the two modes: page mode (someone reading along,
the tester's case) waits, while majlis mode (a projector following a professional)
keeps the old run-on behaviour and only holds still and stops the glide.

**The tester's recording** (test3.mp4, the end of Ziyarat Waritha), replayed:

| | what's highlighted 1, 2, 3, 4, 5 s after they stop |
|---|---|
| server pipeline (turbo), before | next line: words 1, 2, 3, 3, 3 |
| server pipeline, page mode | next line word 1, then back to *al-mawtūr* for good |
| server pipeline, majlis mode | next line: words 1, 2, 2, 2, 2 (holds, no walking) |
| phone app (headless Chrome), `?pauses=0` | next line: words 1, 2 |
| phone app, page mode | *al-mawtūr* throughout |

    python scripts/pause_eval.py build --model models/whisper-base-quran-dua-ct2 --tag whisper-base-quran-dua
    python scripts/pause_eval.py score --tag whisper-base-quran-dua "base" "base;still"
    python scripts/word_eval.py --delay 0.5 "fixed;smooth ease=1.0,speed_scale=1.2;still"

## Not done

- Telling a breath from a stop sooner. At the moment someone stops, the audio
  can't say whether they'll go on, so the lead crosses and the step back undoes
  it a second later. A per-reciter pause model looked promising until the data
  showed reciters range from never pausing (Hussein Ghareeb) to pausing at half
  their line ends (al-Halawaji), and the tester flows too until they stop.
