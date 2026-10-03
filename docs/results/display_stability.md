# Why the display jumped around, and four rules that stop it (2026-09-30)

Hasan kept seeing the display jump around on his own recitations, even in the latest runs.
Five of his debug sessions (Ayat al-Kursi on the phone and on desktop Firefox with a
headset) plus one of Dua Faraj show why. The Python tracker re-run over each session's own
transcripts reproduces what the page showed on 100% of updates, so any rule can be tried
on them offline (`scripts/session_replay.py`).

## What went wrong

1. **The lead ran into the next line and came back.** The page shows where the reciter
   will probably be `delay + 0.5 s` from the window's end. Ayat al-Kursi's lines are 3-8
   words, so the prediction crossed into the next line while he was still mid-line. When
   the evidence didn't follow, the display stepped back: 5 → 4 → 5, then 6 → 5 → 6.
2. **Taps were undone within a second.** In one session he tapped "I'm here" on line 8
   six times. Each time, the next update's lead put the display back on line 9, while the
   evidence was still on line 8.
3. **The page froze, then jumped.** The phone model gets about 40-50% of the letters wrong
   on his voice, about as bad as the "not in the corpus" state's 45%. A few such windows
   sent that state to about 100%. The page then held its last line for 5-15 s, and jumped,
   sometimes over a line, when the evidence returned.

A much better ear doesn't fix this. With the server's large model re-transcribing the same
windows (half the phone's character errors), the display stepped back just as often (9
steps back, 7 flickers). The frozen time halved. The jumping comes from the display rules,
which were tuned on DuaPlayer's professional reciters: steady tempo and long lines, where
predicting ahead works.

## Four rules

Each rule is a `TrackerConfig` switch (tracker.py, and the same in tracker.js,
parity-tested):

| rule | what it does |
|---|---|
| `back_confirm = 2` | a step back to an earlier line is shown only once two updates in a row say so (the pause rule's own step back is untouched) |
| `seek_pins_line` | a tapped line stays until the evidence itself moves past it; the lead can't carry the display off it |
| `null_rate_locked = 0.55` | once locked on a du'a, "not in the corpus" must explain a window at 0.55 edits per letter instead of 0.45 |
| `display_lead = 0.25` (was 0.5) | the display runs 0.25 s ahead of the measured delay instead of 0.5 s |

Tried and not kept:
- `lead_cross_words` (the lead may cross only near the line's end): it cut jumps but held
  the display back a line; line accuracy on the sessions fell from 67% to 53%.
- `keep_dua_confidence`: no measurable effect.
- A `null_rate_locked` of 0.6: two tenths of a point more wrong du'a at phone cadence.

## How they were chosen

Criteria, written before any result (`data/cache/reliability/stab-20260929/prereg.md`). Against
the old defaults, a candidate had to meet all of these:

- **DuaPlayer train, phone model, at both cadences** (a desktop at 0.5 s delay, and the
  phone's every 3 s, 2.5 s late):
  - line accuracy no more than 1 point lower, ±1 no more than 0.5 lower;
  - wrong du'a no more than 0.3 points higher;
  - no more jumps.
- **The pause set:** the next line shown during pauses no more than 2 points more often.
- **A du'a missing from the corpus:** some du'a shown no more than 2 points more often.
- **Hasan's sessions:**
  - at least 30% fewer steps back plus flickers;
  - no more taps undone;
  - line accuracy no more than 2 points lower.

Then one look at test.

Hasan's 6 sessions (the Python tracker over each session's own transcripts, scored against
the turbo reference):

| | line | ±1 | page frozen | steps back | flickers | taps undone |
|---|---:|---:|---:|---:|---:|---:|
| old defaults | 67.1% | 94.4% | 13.3% | 9 | 7 | 7/12 |
| **new defaults** | **71.5%** | **98.8%** | **7.9%** | **4** | **2** | **1/12** |
| keep the 0.5 s lead (H) | 72.0% | 94.8% | 8.3% | 5 | 2 | 1/12 |
| no extra lead at all (G) | 73.9% | 98.8% | 7.9% | 3 | 2 | 1/12 |

DuaPlayer, phone model:

| | train, desktop cadence: line / jumps / lag | train, phone cadence: line / jumps | **test**, desktop: line / jumps / lag | **test**, phone: line / jumps |
|---|---|---|---|---|
| old defaults | 81.5% / 0.81 / 0.7 s | 76.3% / 2.87 | 83.9% / 0.74 / 0.8 s | 79.4% / 2.39 |
| **new defaults** | **82.7% / 0.26 / 0.9 s** | **76.8% / 2.18** | **84.6% / 0.23 / 1.0 s** | **79.9% / 1.86** |

On both sets and both cadences, ±1 is unchanged, and wrong du'a is unchanged or 0.1 point lower.

Pauses (4 s of room tone after every third line) and du'as missing from the corpus:

| | next line shown in a pause (train / test) | word exact (train / test) | missing du'a: some du'a shown (train / test) |
|---|---|---|---|
| old defaults | 35.6% / 32.6% | 35.2% / 37.1% | 31.6% / 28.0% |
| new defaults | 27.7% / 24.8% | 34.2% / 35.7% | 33.4% / 29.8% |

- **Passed on train and on test.** The new defaults are in tracker.py and tracker.js.
- **H fell short on one criterion.** It kept the 0.5 s lead, and on the pause set it
  showed the next line 2.1 points more often, against a limit of 2.
- **G did more than was needed.** With no extra lead at all, the word highlight lagged: 5
  points fewer exact words.
- **The price** is 0.2 s more lag at line changes, and a missing du'a shown about 1.8
  points more often.
- **The pause test set** was built from the previous phone model's transcripts
  (whisper-base-quran-dua); the rules don't depend on the model.

`seek_pins_line` only matters with taps, which DuaPlayer recordings don't have. Its whole
effect is in the sessions: 6 of 7 undone taps in one session became 1.

## Reproduce

```
python scripts/session_replay.py -v "old: display_lead=0.5 seek_pins_line=false back_confirm=1 null_rate_locked=null"
python scripts/evaluate.py --quick --split train --asr whisper-base-aug-v4 --latency 0.5 --still [--set k=v ...]
python scripts/evaluate.py --quick --split train --asr whisper-base-aug-v4 --stride 3 --latency 2.5 --still
python scripts/pause_eval.py score --split train --tag whisper-base-aug-v4 "base;still" "x;cfg k=v,...;still"
python scripts/ooc_eval.py --split train --asr whisper-base-aug-v4 --set --set null_rate_locked=0.55
```

`session_replay.py` shows "as recorded" (the settings the page logged; it reproduces the
page 100%) and "defaults" before any variant. `evaluate.py --quick` scores the tracker only,
and `--set` now takes JSON values.
