# Fewer jumps and early moves (2026-10-03/04)

Hasan: "just cuz we will solve the jumping doesn't mean other aspects will become worse. Remember
all the parts that we want to have working." So every change here was held to bars set before
any experiment (pre-registration in `data/cache/jumps/prereg.md`, with its amendments), checked
by `bench.py guard`: tuned on the dev voices, reported on the held-out test voices.

## Where the jumps come from

`scripts/jump_diag.py` sorts every counted jump and early move by what the display did at that
moment. Dev voices, the stream decoder before this work (1,373 items, 52 h):

| | jumps (1.91 /10 min) | early moves (2.44 /10 min) |
|---|---:|---:|
| crossed into the next line | 72% | 95% |
| went back 1-4+ lines | 18% | 4% |
| changed du'a | 4% | 0% |
| skipped ahead, moved inside its line | 6% | 1% |

The crossing into the next line is the display's own rule (stream_follower.py: `next_p` of the
probability on the next line's first word for `next_steps` steps). With the prior "the next line
comes next", that probability is there one or two frames of letters after a line ends, whatever
the reader does next: going back, repeating the line, talking. Perfect hearing doesn't change
these numbers (docs/results/finding.md): the display decides on the prior, not on the letters.

## Waiting longer doesn't do it

| dev | jumps | early | line lag (median) |
|---|---:|---:|---:|
| next_p 0.75, next_steps 3 | -22% | -36% | +0.11 s |
| next_steps 4 | -20% | -34% | +0.20 s |

A flat delay pays its lag on every line and still lets the wrong line through.

## The next line on evidence (shipped)

A move into line k+1 is shown only when the frames prefer it over the lines the reader could be
on instead (line k again, k-1..k-3, k+2..k+4), by `next_margin` nats, with each alternative's
transition cost taken back off, so the prior doesn't decide. While no alternative leads by more
than `next_slack` (1 nat), the move goes through after `next_hold` s anyway. Both decoders
(stream_follower.py `_next_margin`, web/stream-follower.js), parity-tested.

Two things the first version got wrong, found by tracing dev items:

- It compared only the first two words of line k+1. Once the reader was past them, the margin
  fell to -8 and the display stuck for seconds (lost time 2.7 -> 6.7 /10 min). It now compares
  the words of k+1 reached so far with the same words of each alternative.
- Lines that open alike (every line of Baha starts "اللهم إني أسألك", Munajat Kufa "مولاي يا
  مولاي") tie exactly, and the hold, counting only at margin >= 0, never fired. An alternative
  that reads the same letters up to the likeliest word is now left out: listening can't tell them
  apart yet, so the move is shown as before. Refrain du'as keep their speed.

What the margin can separate, measured on 5,655 gated crossings from 285 dev items: right
crossings reach 3 nats within 0.3 s 74% of the time; of the wrong ones a third have frames that
really do prefer the next line (no listening rule stops those), a third are led by an alternative
(blocked), a third are short ties. So the hold sets the trade:

| dev | jumps | early | line lag median / p90 |
|---|---:|---:|---:|
| margin 3, hold 0.5 | -39% | -47% | +0.12 / +0.21 s |
| **margin 3, hold 0.3** | **-28%** | **-33%** | **+0.06 / +0.11 s** |
| margin 2, hold 0.3 | -28% | -32% | +0.05 / +0.08 s |

Hasan chose hold 0.3. Held-out test voices (617 items), against the same scoring:

| test | before | after |
|---|---:|---:|
| jumps /10 min | 1.94 | 1.49 (-23%) |
| early moves /10 min | 2.37 | 1.75 (-26%) |
| line lag median / p90 | 0.42 / 0.68 s | 0.48 / 0.79 s |
| refrain du'as (Baha, Jawshan, 'Asharat, Mujeer), lag p90 | | all within +0.15 s |
| on the reading line | | within 0.3 pt in every lane |
| lost /10 min, majlis / harvest | 0.87 / 2.00 | 1.12 / 2.13 |
| word exact, majlis / harvest | 68.9 / 65.2% | 67.8 / 64.1% |
| follows a reader's jump within 3 s | | unchanged (skip -1.4 pts) |
| found within 10 s, unknown du'as shown | | unchanged |

It misses three pre-registered bars by a little: jumps -23% (bar -30%), lag median +0.06 s (bar
+0.05), and 0.1-0.25 lost per 10 min plus about a point of exact words in majlis and harvest.
Hasan accepted those for about one fewer wrong move per ten minutes. Default since 2026-10-04:
`StreamConfig.next_margin = 3.0, next_hold = 0.3`; `?sc=nextMargin:0` turns it off on the page.

**On the real page** (`bench_page.py`, one test item per scenario, headless Chrome at phone
speed: Whisper 1.2 s and the CTC model 0.15 s a step, both variants played side by side). The
first run showed lines entered 0.13 s later, twice the bench: at phone speed the steps come 0.15-0.2
s apart, and a hold counted in steps lasted 0.45-0.6 s. The hold now counts seconds of audio
(each step adds the audio since the one before; bench steps are 0.1 s, so the bench numbers are
the same, checked item by item). Second run, 20 items, 38 min:

| real page | off | on |
|---|---:|---:|
| early moves /10 min | 1.86 | 1.33 |
| line lag | +0.49 s | +0.51 s |
| lost /10 min | 1.86 | 1.86 |
| follower step (median / p90) | 1.4 / 2.2 ms | 1.4 / 2.2 ms |

(Jumps were equal; a 38-minute sample holds only a handful, and finding moved +5 points by
chance between runs, which the gate can't touch.) Kids mode (`?kids=1`, Hasan's Dua Hujjah): the
lanterns light in order at the same moments with the gate on and off, the celebration comes 1.4
s after the last word in both, and with the gate on the garland didn't step back a line near the
end.

## Shared passages: not shipped

Hasan's Namaz-e-Wahshat sessions jump because Ayat al-Kursi is read word for word in three
texts and the tracker flips between them. `TrackerConfig.switch_confirm` (tracker.js too, off):
another du'a replaces the one on screen only after standing alone on top for 3 updates while
the shown one keeps 5% of the probability, unless it has clearly won (`switch_sure`, 0.9). On dev it
passed every guard and fixed his sessions (wrong du'a 1.1 -> 0.24%, jumps 5 -> 1 in one source).
On the test voices it held Ramadan day 16 over Iftitah through their long shared opening (majlis
wrong du'a 0.55 -> 0.81%): the one update Iftitah stood alone was right there, as the one update
Eid-e-Mubahila stood alone was wrong in Hasan's. The numbers don't tell those apart; the next try
could wait only before switching to a less recited text. Two other rules were tried and removed:
keeping the shown du'a through any shared passage (on a test Iftitah item it held Ramadan day 16
for 37 s of their shared stretch) and holding the old du'a after a change until the new one is
confident (no gain, majlis wrong du'a up).

## Fixed on the way

- The first test run turned up a bug: the tracker's `position()` runs twice an update (on the
  probabilities projected ahead for the display and on the evidence), and the call that still saw the shared
  passage reset the confirmation count, so it never completed (b4a8b07).
- `bench.py --same-text` compared raw words; Eid-e-Mubahila spells Ayat al-Kursi with a detached
  "و" and "ء", so a display showing the reader's very words in that text counted as the wrong
  du'a. Passages are now compared as the tracker compares them (`tracker._passage_keys`). The
  baselines were re-scored (`base_dev_sp`, `base_test_sp`): Hasan's lane wrong du'a 2.2 -> 1.1%,
  majlis 0.43 -> 0.13% on dev, nothing else moved.
- The test split was looked at to find these, so the final test numbers above come from a split
  that is no longer untouched; the fix A settings were chosen on dev before that.

## Tools

- `bench.py guard --name X --compare BASE [--ooc PTS]`: every bar of the pre-registration,
  PASS/FAIL, with Hasan's lane per source recording.
- `scripts/jump_diag.py [scenarios] --split dev [--lane user] [--sc ...] [--show N]`.
- `bench_page.py` reports follow_ms per step and the settings each variant ran with, read back
  from its session log; the page takes `?sc=k:v,...` (stream decoder) and `?tc=k:v,...` (tracker).
- The bench tracker cache key leaves out tracker fields added later while at their neutral value
  (`bench._LATE_FIELDS`): add new fields there, or every cached run is recomputed.
