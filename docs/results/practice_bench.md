# Practice mode: checking a reading line by line

**Where it stands (2026-10-04).** A checker that listens to each line where it stands passes
every pre-registered bar on dev and on its one test run. On test it marks 0.12 read lines per
10 minutes as left out, catches 93% of left-out lines and 97% of forgotten endings, and
decides 1.7 s after the reader goes on (median). See Phase 1 below. The page doesn't use it yet,
and on Hasan's own voice most lines come back not sure.

## Phase 0: can today's display already check a reading?

Phase 0 of [the practice-mode plan](../research/practice-mode-plan-2026-10-04.md): measure before
building. The question is whether the page's display, unchanged, is already good enough to tell
a reader which lines they left out. Bars were written before any run
(`data/cache/practice/prereg.md`, local), with one amendment after the first run.

**Findings in brief**

- **No.** The display used as a checker ("v0") marks 0.83 lines per 10 minutes that were read,
  against a bar of 0.2. It catches 75% of left-out lines, against a bar of 85%. It does meet
  two bars: forgotten endings (88%) and time to decide (median 1.1 s, p90 2.3 s after the line
  read past the gap). By the pre-registered rule, line checking can't ship on the display alone.
  The test split was not used.
- **Clean reading is close.** On plain reading (no talk, salawat or difficult sound) it marks
  0.31 lines per 10 minutes; held-out harvest voices 0.19, Mafatih texts 0.
- **What breaks it:**
  - difficult sound: 49% of the false marks, 1.54 /10 min (hall 6.7, far 3.0)
  - a salawat between lines: 18%, 4.3 /10 min
  - short lines: Hasan's sessions, 36% of whose lines are under 2.5 s, get 1.8 /10 min on
    plain reading
- **Repeated text cuts both ways.** Without credit for identical lines, most false marks were
  the display following the right words at another copy of a repeated block (Tawassul, Mujeer).
  With the credit, the false marks halve, but a left-out refrain line counts as heard because
  its words were heard elsewhere. A checker has to judge a line where it stands, not whether its
  words were heard anywhere.
- **Word level is out of reach for the display.** It catches 66% of cut words but marks
  11 read words per 10 minutes. That was expected: the display moves past words for many
  reasons other than a skip.

## Setup

`scripts/bench.py` adds three mistake scenarios. They are left out of the reading grid unless
asked for: `score --practice` or naming them.

| scenario | what the reader does | items (all splits) |
|---|---|---|
| `skipword` | leaves out one word in every 3rd-5th line. Cut at the midpoints of the gaps around it with 8 ms fades; never a line's first or last word; at least 3 letters; not a word repeated within one line of it | 136, 601 cuts |
| `skipline` | leaves out one line after every 3-5 (a breath, then the line after it) | 136, 476 lines |
| `ending` | stops 1-3 lines before the end it set out to read, then 8 s of room tone | 137 |

Lanes: studio, harvest, mafatih, user. Majlis is crowd audio, not solo practice. A spot check
transcribed 2.5 s either side of 23 cuts with stock turbo; the cut word was heard in none of
them.

False marks are counted on every reading cell except jumps, switch and ooc (a reader moving
around on purpose, or no text). That means 1,468 dev items (3,255 min, 56 voices), the correct
stretches of the mistake items included.

The v0 checker (`practice_v0`) only reads the display's steps; the display and every reading
bar are unchanged (tested):

- A line is marked unheard once the display reaches two lines past it without showing it.
- At the end of the session, every target line not shown is marked.
- A line counts as shown when the display shows any line with the same words (`_same_lines`;
  added after run 1).
- A word is marked when the display moves past it inside its line.
- A mark the display later clears is a flip.

`practice_metrics` scores the marks against the truth, and `bench.py practice --name X` prints
the table.

```
python scripts/bench.py build skipword skipline ending
python scripts/bench.py asr skipword skipline ending
python scripts/bench.py score --display stream --practice --split dev --same-text 8 --name practice_v0b_dev --workers 3
```

## Results (dev)

| run | false lines /10m | ...after the display found the reader | skipline | ending | skip 1-4 | decide med / p90 | line flips /10m | not judged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bar | **≤ 0.2** | | **≥ 85%** | **≥ 85%** | | **≤ 2.5 / 5 s** | | |
| v0 as registered (`practice_v0_dev`) | 1.58 | | 80% | 94% | 86% | 1.1 / 2.3 s | 0.39 | 20% |
| v0 + same-text credit (`practice_v0b_dev`) | 0.83 | 0.78 | 75% | 88% | 80% | 1.1 / 2.3 s | 0.99 | 2% |

Run 1 left 20% of target lines unjudged: lines before the first one the display showed, often
a later copy of a refrain it landed on first. Run 2 judges from the earliest line shown.
Splitting the false marks at the moment the display first shows the reader's line changes
little (0.83 vs 0.78). Finding the reader is not where the false marks come from.

Run 2, by where the reading happens:

| | minutes | false lines /10m | share of false marks |
|---|---:|---:|---:|
| plain reading (flow, pause, repeat, back, stumble, slow, fast, midstart, mistake cells) | 1,867 | 0.31 | 21% |
| talk between lines | 232 | 0.65 | 6% |
| salawat between lines | 113 | 4.34 | 18% |
| difficult sound (room, hall, babble, fan, other recitation, far, clipped, phone call, low bitrate, combo) | 863 | 1.54 | 49% |

Plain reading by lane: harvest 0.19, studio 0.73, user 1.82 (33 min), mafatih 0. Line length
explains much of it:

| lane | median line | lines under 2.5 s |
|---|---:|---:|
| mafatih | 6.6 s | 1% |
| studio | 5.0 s | 4% |
| harvest | 4.5 s | 12% |
| user | 3.1 s | 36% |

A short line read while the display is still catching up is the one it skips over.

## What it means

The display is built to forgive: it holds through talk, waits for evidence before moving on, and
doesn't have to show every short line. A checker needs the opposite: evidence that each line was
said, where it stands. Phase 1's detector should therefore:

- Judge each line on the audio between its neighbours, not on whether the display stopped
  there. A line the display skipped can still have its letters in the CTC frames; a left-out
  refrain line has no frames of its own between its neighbours even though its words occur
  elsewhere.
- Pin the du'a. Practice mode knows which du'a is being read, so it never needs to find it or
  look at another one.
- Treat a salawat between lines as the stream decoder already does (its interjection chain),
  not as a reason to move.
- Be measured on this bench as it stands. The bars stay as registered, and the after-found
  split and the per-condition table are reported alongside.

Difficult sound (hall, far) may stay out of reach: the reading display itself is weakest there
(bench.md). Practice is usually done alone and close to the phone, so a fallback is possible:
the app says when the sound is too poor to check, rather than marking lines.

## Phase 1: a checker that listens to each line (v1)

`src/dua_recognition/practice.py`, mirrored in `web/practice.js`
(`tests/test_practice_parity.py`: the same lines decided at the same moments, scores within
1e-6). The bars are the ones registered for Phase 0; the amendments in
`data/cache/practice/prereg.md` record each change before the run it affected.

**Findings in brief**

- **It passes every bar, on dev and on the one test run:**

  | | dev | test | bar |
  |---|---:|---:|---:|
  | false left-out /10 min | 0.10 | 0.12 | ≤ 0.2 |
  | skipline caught | 95% | 93% | ≥ 85% |
  | ending caught | 95% | 97% | ≥ 85% |
  | time to decide, median / p90 | 1.7 / 3.1 s | 1.7 / 3.8 s | ≤ 2.5 / 5 s |

  Also: 1-4 lines left out at once (`skip`) are caught 89% / 92%, and a left-out line is called
  heard 1% of the time.
- **How: judge each line where it stands.** Once the display has stayed two lines past line k
  for 0.5 s, the frames the follower committed are explained twice: the best path that says line
  k (its first word and half its letters), and the best path without it. Both run between the
  last earlier line not judged left out and the lines up to one past the display. Each of those
  lines may be left out, the ends are free, and the follower's filler covers anything else said.
  The difference in nats is the line's score.
  - **Verdicts:** left out at -4 or below if the display never showed the line, at -20 or below
    if it did; heard at +4 or above; not sure in between.
  - **The sound gate:** a left-out verdict stands only if at least 60% of the last 4 lines
    judged were heard.
- **What it took** (each step measured on a 217-item dev sample, then on all of dev):
  - Pinning the display to the reader's du'a and start, with no jumps across the du'a. A
    repeated block (Tawassul's per-Imam lines) had pulled the highlight to its other copy.
  - Letting the "with k" path end anywhere from line k on: the display can reach k+2 while the
    reader is still in k.
  - Making every line between the context and the reader optional: with three lines left out,
    each neighbour of each one is missing too.
  - A span of at least 20 s: Du'a Baha's lines all open alike, and a display a line ahead had
    cut the span after the line began.
  - The sound gate: without it, false marks are 0.50 /10 min on dev, almost all in hall (7.3)
    and far (5.4), where most lines come back not sure.
- **Where it can't vouch:** Hasan's own sessions (the user lane, dev only, 41 min) come back not
  sure on most lines (45 /10 min). The phone's CTC model hears his voice too loosely for a line's
  letters to beat "something else" by 4 nats. The gate then holds back 3 of the 5 left-out
  lines there (2/5 caught; 4/5 ungated). Real learners on phones are this case, so it is the
  first thing to improve: a CTC model trained on more ordinary voices, as the synthetic voices
  did for Whisper. The left-out lines in his sessions are also few.
- **Words are out of scope.** A line with one word cut out is marked left out 16% of the time,
  and is otherwise heard or not sure. Telling a missing or wrong word needs its own measurement
  (the plan's TTS swaps), and the app should claim only lines until then.
- **Bench truth fix.** `sources.jsonl` kept global word ids from before the Mafatih texts were
  added. 78 of 539 sources named another du'a's words. `cmd_build` now re-derives each line's
  words by du'a, line and place (`fresh_source`). This affected 18 of the 409 mistake items,
  which were rebuilt before the test run. The reading grid's items predate the shift.

### Results by cell (dev, `practice_v1b_dev`)

| cell | min | false left-out /10m | not sure /10m |
|---|---:|---:|---:|
| flow | 232 | 0.04 | 1.55 |
| pause | 246 | 0.08 | 1.83 |
| talk | 232 | 0.09 | 1.81 |
| salawat | 113 | 0.35 | 0.62 |
| repeat | 207 | 0.10 | 2.13 |
| back | 264 | 0.00 | 1.18 |
| skip | 180 | 0.61 | 2.11 |
| slow / fast | 88 / 71 | 0.00 / 0.00 | 8.79 / 0.71 |
| room / fan / clip / phone call / low bitrate | 422 | 0.00-0.11 | 0.8-3.8 |
| babble / other recitation / combo | 281 | 0.11-0.22 | 11.3-12.9 |
| hall / far | 161 | 0.26 / 0.24 | 63.5 / 52.9 |

By lane, false left-out /10 min (left-out lines caught):

| lane | dev | test |
|---|---|---|
| harvest | 0.13 (65/68) | 0.23 (9/14) |
| mafatih | 0.04 (165/167) | 0.07 (92/96) |
| studio | 0.14 (101/112) | 0.04 (14/14) |
| user | 0.00 (2/5) | (none in test) |

The test split is smaller (636 items, 1,356 min, 34 voices) than dev (1,475 items, 3,272 min,
56 voices). Harvest's 9/14 on test is 14 lines.

```
python scripts/bench.py score --display stream --practice --split test --same-text 8 --name practice_v1_test --workers 3
python scripts/bench.py practice --name practice_v1_test --v1 --split test        # --gate 0: no sound gate
```

### On the page (Phase 2)

Practice mode is on the page, for the device engine.

- **Starting (since the evening of 2026-10-04):** "follow · practise" under the seal, chosen as
  on a camera. Practise changes what the seal, the picker and the chips do; each visit starts in
  follow. The seal shows it: a book open on a rehal takes the place of its mic. The seal finds the du'a from the recitation, as in reading mode (about 5 s on the replay
  below), and checking starts at the line where it was found. Lines before that aren't judged. A
  du'a picked ("choose a du'a"; practised ones first, with when and how many lines to work on) is
  checked from line 1, or from a line tapped before the first verdict. "To work on" chips (in
  place of today's du'as) start one line before the first line to work on. "hide the text" sits
  beside the picker. `?practice=<du'a id>` (optional `&from=<line number>`) names the du'a
  instead, `?practice=1` finds it (both for replays), and `?mode=practise` opens in practise.
- **Under the hood:** the du'a is locked and the stream follower runs without far jumps
  (`cFar` -40). `FrameCommitter` and `PracticeChecker` (web/practice.js) run beside it in
  `DeviceEngine._onFrames`.
- **The rosettes:** each line's verse-end rosette gives the verdict. Gilded: heard. Filled in
  rubric: left out (tapping it says "I did say it", logged as `dismiss`). Broken outline: not
  sure. Nothing is gilded just for being passed, and the words of the line being read aren't
  painted as said.
- **Poor sound:** after three lines in a row it can't vouch for, the page says once: "I can't
  hear clearly enough to check every line".
- **Hiding the text:** "Hide the text until it's heard" (home screen while practising, and the
  menu). In the line being read, each word appears as it is said. A line shows whole once the
  reader goes on from it, having said at least half of it word by word. A skipped line appears
  with its verdict, in red. Until then a line is a soft blur, so the page keeps its shape and each
  rosette stays at its line's end. At a 0.24 em blur, short lines could still be read by someone
  who knows the du'a ("يا حي يا قيوم"), so it is 0.55 em, twice. "Next word" (bottom right) shows
  one more word per tap, in gold: the rest of the line being read, then the next line's first.
- **At stop:** lines up to the furthest one reached are judged, and lines never reached aren't
  left out. The furthest line, found left out, stays unjudged (`practice_unfinished`): it is the
  one the reader stopped in. A replay of Du'a Ahd stopped mid-line 6 had marked it left out.
  Stopped by the reader, the text stays, every line shown with its verdict, and the lines left out
  show their transliteration and translation. A card gives the counts ("15 lines checked · 11
  heard · 3 left out · 1 not sure"), the lines left out as rosettes (a tap scrolls to the line),
  the hints used, what is and isn't checked, and "Practise again" (same du'a, same first line).
  Tapping a red line there marks it said (again: undone). A recording that ends goes home with
  the one-line summary, as before (`page_replay.mjs` waits for the home screen). Lines left out
  are remembered per du'a in localStorage (up one each time left out, halved each time heard,
  counted from before the session so a line marked said after stopping counts as heard). They
  get a heavier outline the next time.
- **Logging:** each verdict is logged as `practice` (score, shown, check_ms), and `hint`,
  `dismiss` (while listening) and `practice_unfinished` too. Lines marked said once stopped
  aren't in the log: it ends at stop.

Checked in headless Chrome (`node scripts/page_replay.mjs ... "practice=dua-kumayl&from=220"`)
on a dev skipline reading. With `?practice=1` and the text hidden, the same reading found Kumayl
in 5 s and caught the same 5 lines left out; no line ahead of the reader ever showed. The page's 30 verdicts equal the bench's, line for line, including
the 5 lines left out. The follower's step stays at 1.1 ms median. A hall-echo reading of
Jawshan Sagheer comes back mostly not sure, with the sound notice. `tests/test_practice_parity.py`
also checks that the page cuts a du'a into lines exactly as the bench does.

### What's next

- **Hasan's own phone sessions in practice mode.** Their verdicts and dismissals are the first
  real-learner data, and his voice is the case the checker can least vouch for today.
- **The phone's ear for ordinary voices:** a CTC student trained on more of them.
- **Words, measured before they are claimed:** the TTS swaps and near words of the plan's
  Phase 0.
- **The daily-use pieces of Phase 3:** plans, reminders, the "continue from this line" quiz.
