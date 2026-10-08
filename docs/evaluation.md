# Evaluation

Two kinds of test. The **scenario bench** plays many voices through every way people read and
scores the page's display; since 2026-10-02 it is how changes are judged. The **studio test**,
DuaPlayer's professional recordings with human line timings, is where the models and the
tracker were first measured; its tables are further down.

## The scenario bench

[`bench.py`](../scripts/bench.py) builds one grid and scores the phone's display on all of it
([bench.md](results/bench.md)). Each item is a program over source audio (spans, room tone,
inserted clips, effects), so its audio is rebuilt exactly and only the models' outputs are
cached.

- **Voices**, in lanes: `studio`, the DuaPlayer test reciters (6 voices, human line timings);
  `majlis`, du'a nights streamed from other centres (4; crowds, PA echo); `harvest`, uploaders
  the [data harvest](results/data_harvest.md) keeps out of all training (97; the CTC teacher's
  forced alignment, confident lines only); `mafatih`, readings of the 17 texts added from
  Mafatih (22); `user`, Hasan's own phone sessions (1). 1,990 items, 74 h, 121 voices. Voices
  are split in half by a hash of the voice into **dev** (tune on it) and **test** (report it).
- **Scenarios** (24): read as recorded, starting mid-du'a, pauses (4 s, one of 15 s), another
  voice talking between lines (3-6 s), a salawat between lines, repeating a line, going back
  1-3 lines, skipping 1-4 ahead, jumping around the du'a, stumbling and restarting a line,
  switching du'a, a du'a the app doesn't have, slow (0.7×) and fast (1.3×) reading; and
  conditions: a room, a big hall (RT60 1.4 s), people talking, a fan, another recitation
  nearby, far from the phone, an overdriven mic, a phone-call codec, a low-bitrate codec, and a
  combination.
- **Scored** as a reader would notice it: the share of reading time the highlight is on the
  reader's line, and on the right word; jumps (to a line two or more away, or another du'a,
  that the reader isn't on) and early moves (to the next line before the reader) per 10
  minutes; lost episodes (4 s or more off the line) per 10 minutes; how often a go-back, skip,
  jump, repeat or restart is followed within 3 s; how much of a pause, talk or salawat the
  highlight stays put; the du'a found within 10 s; for a du'a the app doesn't have, how often
  some du'a is on screen anyway; and the line-change lag. Each has a bar for "usable"
  (`bench.BARS`). No lane meets all of them yet.

### How a change is judged

1. **Tune on the dev voices**, on the whole grid, not on the case that motivated the change.
2. **Score with `--same-text 8`.** A display counts as right where it shows the same 8 words
   the reader's du'a reads there, so a correct screen inside a passage several texts share
   isn't called the wrong du'a, and a second clock, *found ≤10 s\**, starts at the first 8
   words no other text reads ([finding.md](results/finding.md)).
3. **Check every bar against a baseline** with `bench.py guard`: the gains and the
   no-regression guards written down before the jump work ([jumps.md](results/jumps.md)), each
   PASS or FAIL, per lane, with Hasan's sessions per recording.
4. **Report the test voices.**
5. **Confirm on the real page.** [`bench_page.py`](../scripts/bench_page.py) plays one held-out
   item per scenario through the actual page in headless Chrome at phone speed (Whisper held to
   at least 1.2 s an update, the CTC model to 150 ms a step), the variants side by side so they
   share the machine's load.

The commands are in [development.md](development.md#judging-a-change).

### Where things stand

The page as it runs since 2026-10-07 (the distilled phone Whisper, the streaming CTC student,
the tracker with the popularity prior, the stream decoder with the phone's rules), on the test
voices with `--same-text 8`, at a phone's real pace: Whisper every 2 s and shown 1.8 s late, a
CTC step every 0.1 s shown 0.1 s late, as Hasan's Galaxy Z Flip 6 runs once it has warmed up
(`DUA_BENCH_ASR_EVERY=2 ... --delay 1.8 --follow-delay 0.1`, `full_ph_G1_test`). 617 items, 22 h,
55 voices. The `user` lane hashes to dev, so it has no test row.

| lane | on line | right word | jumps /10 min | early /10 min | lost /10 min | follows ≤3 s | stays put | found ≤10 s | found ≤10 s\* | wrong du'a | line lag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| studio | 89% | 78% | 1.33 | 1.19 | 1.68 | 63% | 88% | 75% | 99% | 0.5% | +0.41 s |
| majlis | 90% | 72% | 3.46 | 3.98 | 2.34 | 85% | 90% | 100% | 100% | 2.2% | +0.40 s |
| harvest | 89% | 71% | 2.13 | 1.80 | 1.03 | 84% | 90% | 92% | 91% | 0.9% | +0.41 s |
| mafatih | 93% | 80% | 0.48 | 1.78 | 0.36 | 89% | 93% | 93% | 93% | 0.0% | +0.43 s |
| **all** | **91%** | **76%** | **1.49** | **1.88** | **1.01** | **84%** | **91%** | **90%** | **94%** | **0.6%** | **+0.41 s** |
| server engine, all | 94% | 83% | 1.70 | 1.84 | 0.82 | 84% | 91% | 95% | 99% | 0.5% | +0.31 s |

The server engine row is its own models and settings at the server's delays (`full_pF2_test`,
[server_profile.md](results/server_profile.md)). The majlis lane's wrong du'a is one recording, an
Iftitah that starts inside a passage duas.org's Ramadan Day 16 text reads word for word (see
[known limits](#known-limits)).

By scenario, reading as recorded is 94% on line and 81% on the word with 0.15 jumps per 10
minutes; the weakest cells are jumping around the du'a (76% on line, 21.0 jumps per 10 minutes),
switching du'a (81%), far from the phone (85% on line, found within 10 s\* 73% of the time), the
big hall (88%, 88%), and early moves when a reader repeats a line (6.4 per 10 minutes) or someone
talks between lines (3.8). The test voices were looked at while diagnosing jumps and in the two
pushes of 2026-10-07, so they are no longer untouched; the settings were chosen on dev
([jumps.md](results/jumps.md), [phone_engine.md](results/phone_engine.md)).

How it got here, each step on the held-out voices:

- **The stream decoder** in place of the rule-based follower (2026-10-03; 405 items, before
  `--same-text`): on line 83% → 87%, right word 61% → 67%, jumps 3.4 → 2.3, early moves
  3.5 → 2.5 and lost episodes 4.2 → 1.8 per 10 minutes, moves followed within 3 s 58% → 72%,
  stays put 69% → 88%, line lag 0.43 → 0.41 s; a du'a the app doesn't have shown 21% → 28%, and
  readers who jump around followed 13 points less often ([bench.md](results/bench.md)).
- **The popularity prior** at 0.5 (2026-10-03): found within 10 s 88% → 90%, from the first
  distinctive words 93% → 94%, wrong du'a 0.9% → 0.5%, unknown du'as shown +1 point
  ([finding.md](results/finding.md)).
- **The next-line rule** (2026-10-04): jumps 1.94 → 1.49 and early moves 2.37 → 1.75 per 10
  minutes, line lag median 0.42 → 0.48 s; in the majlis and harvest lanes 0.1-0.25 more lost
  episodes per 10 minutes and about a point fewer exact words. It missed three of its
  pre-registered bars by a little and was adopted anyway ([jumps.md](results/jumps.md)).
- **The phone's real pace, and models for it** (2026-10-07). Until then the bench scored the
  phone with Whisper every second, shown 1.2 s late, and the CTC model every 0.1 s on 2 s windows
  padded to 3 s. Hasan's phone runs Whisper every 2 s, 1.8 s late, and took 0.23 s for each CTC
  step. At that pace the page of that morning was on line 89%, on the right word 60% and found the
  du'a within 10 s\* 83% of the time (`full_ph_r2_test`; 91%, 71% and 95% at the old pace). A
  streaming CTC student (each step encodes only the new audio: 7 ms against 28 ms in Chrome), a
  phone Whisper distilled from the server's and the decoder rules the server engine had shown
  (repeated lines, salawat, far jumps, shared passages): on line 91%, right word 76%, found 94%,
  line lag 0.69 → 0.41 s, for 0.27 more jumps and 0.23 more early moves per 10 minutes and unknown
  du'as shown 29% → 36% ([phone_engine.md](results/phone_engine.md)).

With the true words of each window in place of Whisper's transcript (`score --asr truth`, a
perfect ear), the du'a is found within 10 s 91% of the time instead of 86%, and 25-30 points
more often in echo, distance, clipping and babble; jumps and early moves barely change. So
finding the du'a in bad rooms is mostly hearing, and wrong moves are the display's own
([finding.md](results/finding.md)).

## Known limits

- **Du'as the app doesn't have** are often shown as one it does. On the bench's unknown
  texts, some du'a is on screen 25% (dev) to 36% (test) of the time on the phone, 37% (test) on
  the server engine; on an earlier, narrower set it was 14-28%. These are mostly ziyarat that
  share whole lines with texts the app has, and some recordings run into a text it does have,
  so part of it is fair ([bench.md](results/bench.md), [unknown_dua.md](results/unknown_dua.md)).
- **Shared passages.** Ayat al-Kursi is read word for word in three texts (Sahifa 54,
  Namaz-e-Wahshat and an Eid al-Mubahila text). Since 2026-10-07 the display doesn't change du'a
  inside a passage the shown du'a shares with the new one, allowing for a letter or two of
  spelling (Wahshat and Eid al-Mubahila spell one word of Ayat al-Kursi differently). With it
  and the other changes of that day, Hasan's sessions, where Namaz-e-Wahshat used to flip,
  show the wrong du'a 0.6% of the time instead of 2.8%. The cost is a reading that starts
  inside such a passage: it keeps the first text shown until the passage ends. A test Iftitah
  from KSIJ Dar es Salaam starts inside a passage duas.org's Ramadan Day 16 text reads word for
  word, and is shown as Day 16 for it, on the phone and the server engine alike (majlis wrong
  du'a on test: phone 0.3% → 2.2%, server engine 0.7% → 1.5%;
  [phone_engine.md](results/phone_engine.md), [server_profile.md](results/server_profile.md)).
- **Echo and distance.** In a big hall or far from the phone both models hear less (see the
  perfect-ear result above). In a simulated masjid (a PA in a measured hall, people talking in
  it), the phone is on the reader's line 67% of the time and on the word 57%, and names the du'a
  within 10 s\* 56% of the time; the server engine 81%, 74% and 81% (test). The phone Whisper
  distilled with halls and crowds brought most of the phone's gain there (55% → 67% on line);
  dereverberation in front of the models helped offline but its real-time form broke plain
  reading ([noise.md](results/noise.md), [phone_engine.md](results/phone_engine.md)).
- **Hasan's own voice is identified more slowly than reciters'.** In a screening of real-page
  runs from a cold start mid-du'a, 14 of 21 recordings by reciters and congregations named the
  du'a in 6-7 s, while his own phone readings took 16 and 27 s; two of his Kumayl sessions were
  never identified on the phone ([phone_latency.md](results/phone_latency.md)). On the bench
  his lane is a single voice, so how far this holds for other ordinary readers isn't measured.
- **Readers who jump around the du'a** are followed within 3 s about three times in four, and
  the highlight is on their line 76% of the time.
- **Word-level labels are automatic** (see [reliability checks](#reliability-checks)).

## The studio test

### Ground truth

Ground truth comes from [DuaPlayer](https://www.duaplayer.org), where reciters upload
recordings together with a hand-recorded start time for every line. That gives
**39 recordings (10 h) of 22 du'as and ziyarat by 12 reciters**, each second labelled
with the line being recited. The tables in this section were measured with 505 texts
(DuaPlayer's 91, plus 414 from duas.pro and duas.org), so the 483 without test recordings act
as distractors, before the popularity prior and the stream decoder. The 17 texts added from
Mafatih since then leave the bench's existing grid unchanged ([bench.md](results/bench.md)).

### Method

Everything below is measured on **held-out reciters** (6 reciters, 16 recordings, 4 h):
nobody in the test set was used to tune the tracker or to train an ASR model. The split is
by reciter, not by recording ([`splits.py`](../src/dua_recognition/splits.py)). The
evaluation replays each recording exactly as the live system hears it (6 s windows,
1 s hop) and compares the displayed line with the true line once per second.

### Line following

| front end | line | ±1 line | refrain lines | wrong du'a shown | median lag | du'a found @3 s | @10 s |
|---|---|---|---|---|---|---|---|
| baseline: per-window text matching (turbo) | 42.7% | 71.4% | 5.8% | 12.9% | 3.6 s | 66% | 88% |
| large-v3-turbo, stock | 81.5% | 96.1% | 82.6% | 0.4% | 1.4 s | 62% | 91% |
| large-v3-turbo + prompt biasing | 84.8% | 96.2% | 88.1% | 0.4% | 1.2 s | 70% | 93% |
| large-v3-turbo, fine-tuned (server) | 84.0% | 96.3% | 88.7% | 0.4% | 1.3 s | 76% | 92% |
| whisper-small, fine-tuned | 84.4% | 96.3% | 88.4% | 0.3% | 1.3 s | 76% | 92% |
| whisper-base (Quran), fine-tuned, earlier | 85.2% | 96.2% | 89.1% | 0.3% | 1.2 s | 75% | 92% |
| whisper-base (Quran), fine-tuned on more voices | 84.8% | 96.2% | 87.8% | 0.5% | 1.3 s | 72% | 92% |
| whisper-base (Quran), + synthetic voices, 8 s context (phone until 2026-10-07) | 85.4% | 96.3% | 88.6% | 0.5% | 1.2 s | 76% | 93% |
| **the same, distilled from the server's turbo with halls and harvest windows (phone)** | **86.5%** | **96.2%** | **89.8%** | 0.4% | 1.2 s | 84% | 94% |

The baseline matches each window's transcript against the corpus on its own, with no
tracker. The last row uses today's tracker defaults, so it also has the popularity prior of
2026-10-03: the gain over the row above it is the model and the prior together. The row above
it uses the tracker settings of 2026-09-30
([display_stability.md](results/display_stability.md)); the rows above that predate them;
at this replay, with no display lead, the only one that matters is `null_rate_locked`,
worth +0.2 points on train. *Refrain lines* are lines that recur word for word, so text
alone can't place them. From a cold start mid-recitation, the phone model names the du'a
within 5 s 90% of the time and within 10 s 94% (the one before it: 88% and 93%).

The fine-tunes all follow professional reciters about equally well; where they differ is
everyday voices ([how-it-works.md](how-it-works.md#4-run-on-a-cpu-or-a-phone)), which is
why the phone runs the last one. With 91 texts the du'a was found within 3 s about 89% of
the time; 5.5 times as many candidates, many sharing whole phrases, slow that first guess,
while the wrong-du'a rate fell. The front-end comparison
([comparison.md](results/comparison.md)) and `test_small-ft.md` (the first whisper-small
fine-tune) are from the 91-text corpus.

Ziyarat Ashura is excluded from line accuracy (its human timings drift by several lines)
but kept for identification. On the 91-text corpus, a simulated phone-in-a-room (reverb,
15 dB SNR) costs the clean-trained model 8.5 points (85.4% to 76.9%); training with room
augmentation wins almost all of that back, 84.9% in the room
([comparison.md](results/comparison.md)).

### Word following

Lines are half of it; the highlight is on a word. This section is the rule-based follower
(`follower.py`, now `?follower=rules`); the stream decoder that replaced it on the page is
measured on the scenario bench above. Scored every 0.1 s against forced-aligned
word timings on the test reciters, at the phone's delay (Whisper's result shown 1.2 s after its
window), and on the same recordings with a 4 s pause inserted after every third line
([phone_follower.md](results/phone_follower.md)):

| display | word exact | ±1 word | line steps back /min | next line shown in pauses | line change after the human mark |
|---|---|---|---|---|---|
| Whisper + tracker, predicted forward, gliding (until 2026-09-30) | 34.7% | 83.9% | 0.51 | 53.3% | +0.28 s |
| **rule-based CTC word follower on the phone's small CTC model** | **79.3%** | **98.1%** | **0.00** | **1.6%** | +0.63 s |

The follower lights the word whose letters were just heard, so it is never early (0.7% of
line changes more than 0.3 s before the human mark, against 20.7%) and doesn't run on when the
reciter stops; it changes lines a little later, when the new line's first letters arrive.
Over the 224 minutes of test recitation it stepped back a line once; the old display, 114 times.

That table assumes each follower result is shown 0.1 s after its audio. On Hasan's Android it was
nearer 0.25 s (the CTC model shares the CPU with Whisper), which cost it 10 points: 68.6% word
exact on test. Since the evening of 2026-10-01 the phone runs the model on 2 s windows every 0.1 s
with audio in 50 ms chunks: **75.5%** at its own delay, words within a line lit 0.29 s after they
start instead of 0.44 s, the next line shown in 0.8% of pause time instead of 1.5%, and a reader
who says a line again followed back to its start two times in three instead of one in three
([phone_latency.md](results/phone_latency.md)). Hasan's phone, once warm, took 0.23 s a step on
those windows; since 2026-10-07 the page runs a streaming student that encodes only the new audio
at each step, so a step every 0.1 s fits on a phone ([phone_engine.md](results/phone_engine.md)).

## Results and research notes

Per-model tables with per-du'a breakdowns are the `test_*.md` files in
[results/](results/) (the phone model is
[test_whisper-base-ph-kd2.md](results/test_whisper-base-ph-kd2.md); the ones before it,
[test_whisper-base-syn-v5-ctx8ft.md](results/test_whisper-base-syn-v5-ctx8ft.md) and
[test_whisper-base-aug-v4.md](results/test_whisper-base-aug-v4.md)).

| topic | write-up |
|---|---|
| ASR speed and accuracy per window | [asr_benchmark.md](results/asr_benchmark.md), [asr_benchmark_turbo.md](results/asr_benchmark_turbo.md) |
| Front ends, prompting and room audio compared | [comparison.md](results/comparison.md) |
| Speed prior and tempo adaptation; a negative result on scoring text from CTC | [speed_prior.md](results/speed_prior.md) |
| Keeping the display on the word being said | [display_lead.md](results/display_lead.md) |
| Waiting when the reciter pauses | [pauses.md](results/pauses.md) |
| Saying "I don't know this one" for du'as outside the corpus | [unknown_dua.md](results/unknown_dua.md) |
| Texts that share a passage | [shared_passages.md](results/shared_passages.md) |
| Findings from phone sessions | [phone_sessions.md](results/phone_sessions.md) |
| The phone app before and after 2026-09-30 (display rules, 8 s context, synthetic voices) | [phone_before_after.md](results/phone_before_after.md) |
| Why the display jumped on ordinary voices, and four rules that stop it | [display_stability.md](results/display_stability.md) |
| The phone model with an 8 s audio context: four times faster | [phone_speed.md](results/phone_speed.md) |
| Synthetic ordinary voices (zero-shot TTS) for the phone model | [synthetic_voices.md](results/synthetic_voices.md) |
| Tilawa's streaming phoneme engine on du'as (negative) | [tilawa_duas.md](results/tilawa_duas.md) |
| A CTC word follower (server, wav2vec2) | [follower.md](results/follower.md) |
| The word follower on the phone: a small CTC model taught by wav2vec2 | [phone_follower.md](results/phone_follower.md) |
| The follower faster on the phone; the last word; repeats | [phone_latency.md](results/phone_latency.md) |
| When the reciter stops, the page should too | [stops.md](results/stops.md) |
| Lines read out of order: a benchmark, and finding a reader who jumps | [out_of_order.md](results/out_of_order.md) |
| Thousands of hours of uploaded recitations, labelled by their audio | [data_harvest.md](results/data_harvest.md), [data_harvest_numbers.md](results/data_harvest_numbers.md) |
| One bench for every way people read; the stream decoder; texts added from Mafatih | [bench.md](results/bench.md) |
| Finding the du'a: a perfect ear, same-text scoring, the popularity prior, retrains on the harvest | [finding.md](results/finding.md) |
| Fewer jumps and early moves: the next line on evidence; shared passages (not shipped) | [jumps.md](results/jumps.md) |
| A phone in a masjid hall: measured halls, a PA, a crowd; dereverberation | [noise.md](results/noise.md) |
| Bigger models for the server engine (not adopted) | [server_models.md](results/server_models.md) |
| The server engine's own models and settings: a fully fine-tuned turbo, 3 s CTC windows every 0.05 s | [server_profile.md](results/server_profile.md) |
| The phone at its real pace: a streaming CTC student, a distilled Whisper, the phone's decoder rules | [phone_engine.md](results/phone_engine.md) |
| Noha (Urdu, Arabic, Farsi, English): language ID and identification | [noha_lid.md](results/noha_lid.md), [noha_match.md](results/noha_match.md) |

## Reliability checks

A second pass checked the measurements themselves, with each experiment's acceptance
rules written down before its results ([plan](research/claude-execution-plan-2026-09-26.md),
[review](research/current-state-review-2026-09-26.md)):

- [browser_gate.md](results/browser_gate.md): how much speech the phone's speech gate
  discards. `legacy` stays the default; `?gate=energy_assisted` is opt-in.
- [follower_reliability.md](results/follower_reliability.md): where the word follower's
  line lag comes from.
- [small_ctc.md](results/small_ctc.md): Tilawa's FastConformer as the follower's front end.
  wav2vec2 stayed; since 2026-10-01 the follower runs on a small student of it instead
  ([phone_follower.md](results/phone_follower.md)).

Human review of word-level labels is still open: the review bundles are exported
([`review_bundle.py`](../scripts/review_bundle.py)) but not yet annotated, so all
word-level numbers rest on automatic alignments.

## Reproduce

The studio test:

```bash
python scripts/fetch_duaplayer.py                         # recordings and line timings (local cache)
python scripts/transcribe_windows.py --model large-v3-turbo
python scripts/evaluate.py                                # test reciters
python scripts/evaluate.py --split train --tune           # tracker grid search (train only)
```

The scenario bench (its sources come from local caches and the harvest; see
[development.md](development.md#judging-a-change)):

```bash
python scripts/bench.py sources && python scripts/bench.py build   # sources, item programs, truth
python scripts/bench.py asr                                        # GPU: Whisper rows, quiet, CTC frames
python scripts/bench.py score --name now_test --display stream --split test --same-text 8
```
