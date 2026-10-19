# Every model combination side by side (2026-10-09)

Hasan (2026-10-09): "we have a lot of models and random arguments that can be used. can we lay them
all out and see which is best", then "check all combos on the audio data from debug audio and test
data".

Each part had been measured before, but on different days, texts, defaults and cadences, so the old
numbers didn't sit side by side. Here every combination is scored on the same items, today's texts
(after the text audit) and today's defaults.

## What was combined

| part | candidates |
|---|---|
| Whisper (finds the du'a and the line) | phone-sized: `whisper-base-ph-kd2` (the page's), `whisper-base-syn-v5-ctx8ft` (the page's until 10-07), `whisper-base-h2c-ctx8` (harvest retrain, not adopted on 10-03); server-sized: `whisper-turbo-srv2` (the server's), `whisper-turbo-srv` (its first round), `whisper-turbo-dua` (the LoRA turbo); and a perfect ear (the true words of each window) |
| letter model (places the word) | the streaming student `ctc-stream-base-v1` (the page's); the window student `ctc-student-base-v6` (the server's, on 3 s windows; the page's `?ctc=ctc-student-base-v6-w2` on 2 s) |
| word follower (draws the highlight) | the stream decoder (the default); the rule-based follower (`?follower=rules`); none: Whisper and the tracker alone (`?words=off`) |

Two cadences, each with the models that could run there:

- **The phone** (Hasan's Galaxy Z Flip 6, as phone_engine.md measured it): Whisper every 2 s, shown 1.8 s
  late (`DUA_BENCH_ASR_EVERY=2 --delay 1.8`); the streaming student every 0.1 s, shown 0.1 s late; the
  window student on real 2 s windows every 0.2 s, shown 0.25 s late (`DUA_BENCH_CTC_STRIDE=2
  --follow-delay 0.25 --sc hop=0.2`: the 234 ms a step it took on the phone). The phone-sized Whispers only.
- **The server**: its delays (`--delay 0.15 --follow-delay 0.05`); the window student on 3 s windows every
  0.05 s, or the streaming student every 0.1 s; `SERVER_PROFILE`'s decoder settings, and its tracker
  settings (`kappa` 0.2, `null_rate` 0.35) with the turbo Whispers. Every Whisper.

Data:

- **Test**: the bench's test voices, `--split test --display stream|follower|tracker --same-text 8`, 594
  items. The test voices were looked at in the jump work and the two pushes of 10-07.
- **Hasan's debug sessions, as recorded**: `scripts/bench_sessions.py` (new). Of the 90 sessions in
  data/sessions (180 min), every run of at least 4 consecutive lines the teacher (wav2vec2-quran-dua-voices)
  times word by word is an item, played as he recorded it, with every gap of 2 s or more between his words
  a pause the highlight should hold still through: **37 items, 37 min, from 40 sessions**. What's left out:
  one 80-min session left recording (`ka7m`), 14 sessions under 20 s, and readings the teacher can't time
  in order (jumps, talk, misheard stretches). The du'a of a session is the one turbo's transcripts place
  best, among those the page showed or the transcripts point to; where texts share a passage (Ayat
  al-Kursi), the one the teacher times most of in whole lines. The truth is automatic, not human, and
  flatters the letter models (they learned from this teacher). `h2c` and `turbo-srv` aren't scored here:
  their weights were deleted on 10-07, and only the bench's cached outputs remain.

The bench's own user lane (7 of these sessions cut into its scenarios, on dev) is left as it was.

## The phone

Test, 594 items:

| Whisper | letter model | follower | on line | right word | found ≤10 s* | lost /10 min | follows ≤3 s | holds still | line lag |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **ph-kd2** | **streaming** | **stream decoder** | **90.5%** | **75.9%** | **94.0%** | **1.01** | **83%** | **90.9%** | **+0.41 s** |
| h2c | streaming | stream decoder | 90.3% | 75.7% | 90.6% | 1.03 | 83% | 90.7% | +0.41 s |
| syn-v5 | streaming | stream decoder | 89.9% | 75.4% | 91.8% | 1.15 | 83% | 90.8% | +0.41 s |
| perfect ear | streaming | stream decoder | 90.4% | 75.9% | 96.4% | 1.05 | 83% | 91.4% | +0.41 s |
| ph-kd2 | window, 0.25 s steps | stream decoder | 90.0% | 60.5% | 84.7% | 1.13 | 80% | 91.7% | +0.69 s |
| ph-kd2 | streaming | rule-based | 85.3% | 66.9% | 94.0% | 3.93 | 51% | 75.8% | +0.36 s |
| ph-kd2 | window, 0.25 s steps | rule-based | 87.0% | 59.0% | 84.7% | 2.88 | 61% | 74.6% | +0.57 s |
| ph-kd2 | - | none (tracker) | 64.5% | 4.7% | 94.0% | 6.53 | 18% | 74.6% | +2.93 s |
| ph-kd2, **Whisper every 1 s** (1.2 s late) | streaming | stream decoder | **92.2%** | **77.4%** | **95.7%** | 1.03 | 82% | 91.5% | +0.41 s |

Every Whisper ranks the same way across letter models and followers (`c_test_ph_*`: the window student
costs each 15 points of right word, the rule-based follower 9). Paired over items, against today's
(points, 95% interval): syn-v5 found -2.1 [-3.6, -0.7]; h2c found -3.4 [-5.1, -1.8]; the perfect ear
found +2.5 [+1.1, +4.0], on line and right word ±0.0; Whisper every 1 s on line +1.7 [+1.6, +1.9],
right word +1.5 [+1.3, +1.7], found +1.8 [+0.7, +3.0]. At 1 s the unknown du'a is shown as a known one
51% of the time instead of 35% (21 items).

Hasan's sessions, 37 items:

| against today's phone (on line 68.9%, right word 45.5%) | on line, pts | right word, pts |
|---|---|---|
| Whisper every 1 s | **+6.1** [+3.2, +10.1] | **+4.4** [+1.9, +7.8] |
| perfect ear (every 2 s) | +4.7 [+1.7, +8.0] | +3.4 [+1.5, +5.5] |
| syn-v5 | -4.1 [-6.6, -1.6] | -2.2 [-4.0, -0.0] |
| window student, 0.25 s steps | +0.3 [-0.8, +1.4] | -8.9 [-10.9, -6.7] |
| rule-based follower | -7.1 [-11.8, -2.2] | -15.2 [-20.4, -9.8] |
| the server engine (srv2, window student 3 s / 0.05 s, its delays) | +11.4 [+7.2, +17.3] | +15.0 [+11.2, +20.3] |
| ...with the phone's Whisper on the server | +9.2 [+5.7, +14.1] | +12.9 [+9.6, +17.3] |

## The server

Test, 594 items, the server's delays and profile:

| Whisper | letter model | follower | on line | right word | found ≤10 s* | lost /10 min | holds still | wrong du'a | unknown shown |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **turbo-srv2** | **window 3 s / 0.05 s** | **stream decoder** | **93.7%** | **83.5%** | **98.6%** | **0.84** | **91.4%** | **0.48%** | **37%** |
| turbo-srv | window | stream decoder | 93.6% | 83.5% | 98.4% | 0.88 | 91.1% | 0.45% | 37% |
| turbo-dua | window | stream decoder | 93.4% | 83.4% | 97.5% | 0.86 | 91.6% | 0.42% | 34% |
| ph-kd2 | window | stream decoder | 93.6% | 83.5% | 97.9% | 0.88 | 91.2% | 0.55% | 51% |
| syn-v5 | window | stream decoder | 93.2% | 83.2% | 97.2% | 0.97 | 91.1% | 0.50% | 49% |
| h2c | window | stream decoder | 93.6% | 83.5% | 97.3% | 0.85 | 91.0% | 0.48% | 51% |
| perfect ear | window | stream decoder | 93.9% | 83.6% | 98.4% | 0.75 | 92.3% | 0.44% | - |
| turbo-srv2 | streaming, 0.1 s | stream decoder | 93.7% | 79.3% | 98.6% | 0.81 | 88.9% | 0.49% | 37% |
| turbo-srv2 | window | rule-based | 91.0% | 79.9% | 98.8% | 2.70 | 71.4% | 0.27% | 26% |
| turbo-srv2 | - | none (tracker) | 86.7% | 36.0% | 98.8% | 5.06 | 69.5% | 0.27% | 26% |
| turbo-srv2, the phone's values of the five pinned rules | window | stream decoder | 93.4% | 82.6% | 98.6% | 0.93 | 93.5% | 0.63% | 37% |

Paired against today's server: the phone's Whisper on line -0.1 [-0.2, +0.1], right word +0.0, found -0.7
[-1.4, -0.2]; turbo-dua found -1.1 [-2.0, -0.4]; the streaming student right word -4.3 [-4.5, -4.0]; the
phone's five values right word -0.9 [-1.0, -0.8], on line -0.3, but holding still +2.1 and early moves
1.79 -> 1.41 /10 min. On Hasan's sessions: srv2 over turbo-dua on line +3.1 [+1.8, +4.4], over the
phone's Whisper +2.2 [+0.5, +4.2]; the streaming student right word -6.8 [-9.7, -4.7]; the phone's
five values on line -0.7 [-1.7, -0.1], right word -1.0 [-1.6, -0.6].

## What this says

- **Today's combinations are the best of what exists, on both engines and both data sets.** On the
  phone: ph-kd2 + the streaming student + the stream decoder. On the server: turbo-srv2 + the window
  student on 3 s windows every 0.05 s + the stream decoder.
- **The follower and the letter model decide the word; Whisper decides how fast the du'a is found.**
  Any Whisper, even a perfect one, gives the same on-line and right-word numbers within 0.7 points
  on test; between Whispers only du'a finding and unknown du'as move. Between followers the stream
  decoder leads every Whisper by 4-9 points of right word and loses track a quarter to a third as often.
- **Cadence matters more than accuracy.** On the phone, the same Whisper every second gains more than a
  perfect one every two seconds (Hasan's sessions: on line +6.1 against +4.7). Most of the server's lead
  on his sessions (+11.4 on line) is its cadence: the phone's own Whisper there gets +9.2. Whether his
  phone can run Whisper every second (`?anchorhop=1`) isn't known: it took 1.73 s a window on 10-06,
  when the window student took four times the CPU the streaming one does.
- **The letter model must fit the cadence.** On the phone the window student at its affordable 0.25 s
  steps costs 15 points of right word (test) and enters lines 0.28 s later; on the server, where 3 s
  windows every 0.05 s are affordable, it beats the streaming student by 4.3 (test) and 6.8 (sessions).
- **Hasan's own readings are far harder than the bench's voices**: today's phone is on his line 68.9%
  of the time and on his word 45.5% (test: 90.5% and 75.9%). The automatic truth is part of it (his
  phone, his voice, a teacher that may time him worse), but the user lane in bench.py, on the same
  voice, reads the same (76.6%, 46.1%).

### Ruled out by these runs

| | why |
|---|---|
| `whisper-base-syn-v5-ctx8ft` (`?model=`) | found -2.1 (test), on line -4.1 (sessions) |
| `whisper-base-h2c-ctx8` | found -3.4 (test); weights deleted |
| `whisper-turbo-dua` | found -1.1 (test), on line -3.1 (sessions) |
| `whisper-turbo-srv` | found -0.2 (test), otherwise equal; srv2 was better on Hasan's sessions (server_profile.md); weights deleted |
| the window student on the phone (`?ctc=ctc-student-base-v6-w2`) | right word -15.4 (test), -8.9 (sessions); found -9.2 (test) |
| the streaming student on the server | right word -4.3 (test), -6.8 (sessions) |
| the rule-based follower (`?follower=rules`) | on line -5.2, right word -9.0, lost 1.0 -> 3.9 /10 min (test); -7.1 and -15.2 (sessions) |
| no follower (`?words=off`) | right word 5%, 2.9 s behind (phone, test) |
| the phone's values of the server's five pinned rules, together | right word -0.9 (test), -1.0 (sessions); they hold still better (+2.1). One at a time (below), `nextIntj` costs the words |
| `nextIntj` on the server | right word -0.6, line lag on Baha 0.22 -> 0.54 s median (dev) |

### Not settled here

- Whisper every second on the phone: the bench says it's worth having; the phone has to show it can.
  At that cadence `null_rate` 0.35 keeps the unknown du'a at today's rate on dev (below); not yet
  scored on test or on his sessions.
- `cFillNext` -2 on the server (the phone's value; the server pins -1): it passes every guard bar on
  dev, holding still +0.6 (below); not yet scored on test or on his sessions.
- Since about 16:30 on 2026-10-10 the corpus has 528 texts (6 after-salat texts, taqibat.md). Every
  number here is on the 522. The phone's baselines on the 528 exist (`taq_new_dev`, the same as
  `base0909_dev` in every headline column; `taq_new_test`); the server's (`d_srv_base`) and the
  sessions' (`sess_ph_kd2_strm_sd`) would need scoring again.
- `?webgpu` (speed only), `?gate` (the bench runs Whisper on every window) and the tiny student
  (only a fallback for window students, which the page no longer uses by default) aren't on the bench.

## Every setting one step either side (the phone, dev)

The phone's decoder (`StreamConfig`) and tracker (`TrackerConfig`) settings, one at a time, at the phone's
cadence, on a fixed quarter of the dev items (every 4th by id, 330 items; `bench_sub.py`). Each rule
switched off once, each dial a step either side of its default. Changes against the defaults (`s_base`:
on line 90.2%, right word 74.4%, found 93.0%, holds still 89.0%), points or per 10 min; "his" is the user
lane's 9 items here (`sweep_table.py`).

What each rule is worth (switched off):

| off | on line | right word | holds still | jumps | early | other |
|---|---:|---:|---:|---:|---:|---|
| `blank_bias` (1 nat off the blank) | -2.6 | -6.7 | +2.8 | -0.35 | -0.57 | his line -15.8, lost +0.91 |
| `next_margin` (the next-line gate) | +0.2 | +1.4 | **-6.0** | +0.72 | +0.81 | |
| `quiet_pen` (letters cost more in quiet) | +0.2 | +1.1 | **-3.6** | +0.17 | +0.25 | |
| `next_intj` (salawat in the gate) | +0.1 | +0.4 | -2.0 | | | |
| `ctc_push_p` (the frames' push) | -0.2 | -0.2 | +0.2 | | | follows -2.5 |
| `copies` | -0.3 | -0.3 | | | | |
| `shared_words` (the shared-passage hold) | | | | | | his line -2.7 |
| `switch_s` (a du'a change counts after 2 s) | | | -0.1 | +0.10 | | found +0.6, his line -3.5 |
| `stop_p` (catch-up at a stop) | | -0.05 | | | | |
| `lapse_intj` | | | | | | no change |

Every rule earns its place except `lapse_intj`, which changes nothing on these items (it was added for
one 10 s salawat in a Baha session, server_profile.md). The gate, the quiet penalty and the salawat
alternative buy holding still with a point of right word or so each: the trade the jump work chose.

The dials: most move nothing by more than 0.3 points. The ones that do trade one thing for another:

| change | on line | right word | holds still | jumps / early | his word |
|---|---:|---:|---:|---|---:|
| `blank_bias` 1.5 (from 1.0) | +0.3 | +1.3 | -1.0 | +0.14 / +0.11 | +5.1 |
| `next_hold` 0.2 (from 0.3) | | +1.0 | -0.5 | +0.22 / +0.21 | +1.3 |
| `quiet_pen` 2 (from 4) | +0.1 | +0.6 | -0.8 | +0.14 / | +1.5 |
| `fill_cost` 1.5 (from 1.0) | | +0.1 | -0.8 | +0.15 / +0.06 | +3.6 |
| `lookahead` 0.1 (from 0.2) | -0.1 | -0.4 | -0.8 | | -0.9 |
| `temp` 0.8 (from 1.0) | -0.2 | -0.8 | +0.6 | | -3.9 |
| `kappa_search` 1.6 (from 1.2) | +0.5 | +0.3 | +0.3 | | found +0.9, unknown shown +9 |
| `kappa_search` 0.8 | -1.3 | -1.1 | | | found -8.2, unknown shown -8 |
| `popularity` 0 (from 0.5) | -0.2 | -0.2 | | | found -0.9, unknown shown +4.9 |

Four looked like small gains with nothing paid, and went on to all dev items: `lookahead` 0.3 (right
word +0.4, early -0.07), `temp` 1.2 (right word +0.6, his word +5.0; salawat holding still 96.6% -> 92.8%
on its 9 items), the tracker's `null_rate` 0.5 (on line +0.4, found +0.9, his line +4.7; only 2 unknown
du'as in the quarter) and `c_far` -6 (holds still +0.4, jumping around on line +1.9). Their runs on all
dev items (`data/testbed/logs/models/grid4.sh`: each, then the four together, each with `bench.py guard`
against `base0909_dev` and a run on Hasan's sessions) are below: none passes cleanly.

## Follow-ups on all dev items (2026-10-10)

All on the 522 texts. Each run went through `bench.py guard`. The jump work's "gain" bars (jumps and
early moves -30 to -40%) fail for every one and weren't meant for changes this size; so does the
absolute bar of 78% on his line, which today's phone misses too (76.6%). The tables give the rest.

**The server's five pinned rules, one at a time.** The server's profile with one rule at the phone's
value (`d_srv_phone_*`), against the profile as it is (`d_srv_base`: on line 92.5%, right word 81.5%,
holds still 90.0%, early 1.83 /10 min, line lag median 0.35 s):

| rule at the phone's value | on line | right word | holds still | early /10 min | line lag | guard |
|---|---:|---:|---:|---:|---:|---|
| `sharedNear` 400 (server 8) | ±0.0 | ±0.0 | ±0.0 | 1.83 | 0.35 s | PASS; the same on every item |
| `sharedEdits` 2 (server 0) | ±0.0 | ±0.0 | ±0.0 | 1.83 | 0.35 s | PASS; his wrong du'a 1.01% -> 0.52% |
| `cFillNext` -2 (server -1) | ±0.0 | -0.0 | +0.6 | 1.73 | 0.35 s | PASS |
| `nextIntj` on (server off) | -0.1 | -0.6 [-0.6, -0.5] | +0.5 | 1.66 | 0.38 s | FAIL: line lag on Baha 0.22 -> 0.54 s median, 0.64 -> 1.66 s p90; Jawshan Kabir 0.20 -> 0.37 s; majlis holds still -1.6 |
| `stopP` 0.15 (server 0) | ±0.0 | ±0.0 | ±0.0 | 1.83 | 0.35 s | PASS |

So the test split's trade for the five together (right word -0.9, holding still +2.1) is two of them:
`nextIntj` costs the words and enters lines late on the server; `cFillNext` -2 holds still better at no
cost the guard sees. The other three change nothing on dev. `SERVER_PROFILE` is unchanged.

**Whisper every second on the phone.** `d_ph_asr1s*`: Whisper every 1 s, 1.2 s late, against today's
phone (`base0909_dev`: every 2 s, 1.8 s late):

| | on line | right word | found ≤10 s* | lost /10 min | jumps /10 min | unknown shown |
|---|---:|---:|---:|---:|---:|---:|
| every 2 s (today) | 89.8% | 74.4% | 93.0% | 1.18 | 1.26 | 25% |
| every 1 s, `null_rate` 0.45 (default) | 91.3% | 75.5% | 94.3% | 1.37 | 1.38 | 36% |
| every 1 s, `null_rate` 0.40 | 91.1% | 75.3% | 94.3% | 1.38 | 1.42 | 33% |
| every 1 s, `null_rate` 0.35 | 90.9% | 75.1% | 93.6% | 1.40 | 1.39 | 26% |
| every 1 s, the server's tracker (`kappa` 0.2, `null_rate` 0.35) | 90.9% | 75.1% | 93.6% | 1.38 | 1.39 | 26% |

`null_rate` 0.35 brings the unknown du'a back to today's rate and keeps most of the gain (on line +1.1,
right word +0.7). His lane (35 items, 6 recordings) loses track and jumps more at every 1 s: lost
1.74 -> 3.8-4.2 /10 min, jumps 2.4 -> 3.8-4.9, wrong du'a 0.59% -> 1.2-1.9%. On his 37 sessions every
1 s with the default tracker was +6.1 on line (above); the 0.35 run there wasn't made before the corpus
changed.

**The sweep's four candidates (the phone).** Against `base0909_dev` (on line 89.8%, right word 74.4%,
found 93.0%, holds still 88.9%, lost 1.18 /10 min, unknown shown 25.3%), and on his sessions against
today's phone (`sess_ph_kd2_strm_sd`: on line 68.9%, right word 45.5%, holds still 87.4%):

| change | on line | right word | found* | holds still | lost | unknown shown | sessions: on line | sessions: right word | guard |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| `lookahead` 0.3 | +0.2 | +0.3 | ±0.0 | ±0.0 | 1.18 | 25.3% | +0.1 [-0.7, +0.8] | +0.8 [+0.4, +1.2] | FAIL: his lost 1.74 -> 2.09 (one talk item) |
| `temp` 1.2 | +0.1 | +0.5 | ±0.0 | -0.9 | 1.16 | 25.3% | +0.8 [+0.2, +1.8] | +3.1 [+2.0, +4.4] | FAIL: his holds still 98.6% -> 89.9%, his lost 1.74 -> 2.09, majlis lost 0.42 -> 0.58 |
| `null_rate` 0.5 | +0.5 | +0.3 | +0.8 | -0.1 | 1.13 | 28.8% | +2.8 [+1.2, +5.1] | +1.8 [+0.7, +3.3] | FAIL: unknown shown +3.5 pts (9 items) |
| `c_far` -6 | +0.1 | ±0.0 | ±0.0 | +0.4 | 1.16 | 25.3% | -0.0 | -0.1 | PASS but for 1 tick of 8,452 unknown shown |
| all four | +0.8 | +1.1 | +0.8 | -0.4 | 1.07 | 28.8% | +3.6 [+1.5, +6.1] | +5.8 [+3.9, +8.3] | FAIL: his holds still 98.6% -> 88.3%, his lost 1.74 -> 2.79, unknown shown +3.5 |

None is a clear gain:
- `c_far` -6 passes and changes almost nothing.
- `null_rate` 0.5 is the one his sessions like (+2.8 on line), but it shows a du'a for an unknown
  reading 3.5 points more often: the opposite of what Whisper every second needs (0.35).
- `temp` 1.2 gains words on his sessions (+3.1; holds still 87.4% -> 88.7% there). In his dev lane it
  loses half the still time of a pause item and two-thirds of a talk item, both from one recording
  (3ac5478786, Hujjat).

The defaults stay as they are.

## How it was run

```bash
# every combination on test (data/testbed/logs/models/combos.sh: 57 runs, one scorer at a time)
bash data/testbed/logs/models/combos.sh test
# Hasan's sessions: items, model passes, the perfect ear's rows, then the same combinations
python scripts/bench_sessions.py build
python scripts/bench_sessions.py asr --split all --model models/whisper-base-ph-kd2-ct2 --ctc-model models/ctc-stream-base-v1
python scripts/bench_sessions.py asr --split all --model models/whisper-base-syn-v5-ctx8ft-ct2 --ctc-model models/ctc-student-base-v6u
DUA_BENCH_CTC_WINDOW=3 DUA_BENCH_CTC_HOP=0.05 python scripts/bench_sessions.py asr --split all \
    --model models/whisper-turbo-srv2-ct2 --ctc-model models/ctc-student-base-v6     # and whisper-turbo-dua-ct2
python scripts/bench_sessions.py truth
bash data/testbed/logs/models/combos_sess.sh
# one setting at a time on a quarter of dev (grid2.sh: s_* and t_* runs), and the table of changes
DUA_BENCH_ASR_EVERY=2 python data/testbed/logs/models/bench_sub.py score --name s_temp_1.2 --split dev \
    --display stream --same-text 8 --delay 1.8 --follow-delay 0.1 --sc temp=1.2
python data/testbed/logs/models/sweep_table.py
# the follow-ups on all dev items: grid3.sh (the server's rules one at a time, Whisper every 1 s),
# grid4.sh (the four candidates, each with bench.py guard and a run on the sessions)
bash data/testbed/logs/models/grid3.sh; bash data/testbed/logs/models/grid4.sh
python scripts/bench.py guard --name d_srv_phone_next_intj --compare d_srv_base
# tables and paired intervals
python data/testbed/logs/models/summ.py "c_test_ph_*" --lanes
python data/testbed/logs/models/diff_ci.py --root=sessions sess_ph_kd2_strm_sd sess_ph_kd2_strm_sd_asr1s
```

Results: `c_test_*` in data/testbed/results, `sess_*` in data/testbed/sessions/results; logs and the
queue scripts in data/testbed/logs/models. Today's phone on test (`c_test_ph_kd2_strm_sd`) equals
`base0909_test` to the tick, and today's server (`c_test_sv_srv2_win_sd`) is within 0.1 point of `full_pF2_test` (scored before the text audit).

