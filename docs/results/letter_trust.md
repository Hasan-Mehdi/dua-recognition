# Trusting the letter model less for an ordinary reader (2026-10-27/28, overnight): not adopted

The streaming letter model (`ctc-stream-base-v1`) misses about half of an ordinary reader's
letters (letter error 47.5% on Hasan's sessions against 16% on the bench's reciters,
[ordinary_readers.md](ordinary_readers.md)). The stream decoder still weighed its letters as
it does for a reciter. When the letters say little, its own moves (the next word comes next,
backs and skips cost more) should count for more. Before this night only the accent fold (and one blank bias with
it) had been tried on his sessions.

All numbers at the phone's cadence (`DUA_BENCH_ASR_EVERY=2`, `--delay 1.8 --follow-delay 0.1`,
`whisper-base-ph-kd2`, `ctc-stream-base-v1`, `--display stream --same-text 8`), with the
2026-10-19 defaults as the base. Work in `data/cache/n27/` (`sweep.sh`, `bench.sh`, `cmp.py`).

## His 35 session items

`scripts/bench_sessions.py`, each setting against `n27_sess_base` (= `s2_sess_f05top3`); right
word with its paired 95% interval over items.

| setting | right word | on line | jumps /10 min | early /10 min | line lag median | lost /10 min | stays put |
|---|---:|---:|---:|---:|---:|---:|---:|
| today's | 52.4% | 73.4% | 2.08 | 4.16 | 0.55 s | 2.68 | 83.7% |
| `temp` 0.7 | 47.8% (-4.6) | 71.5% | | | | | |
| `temp` 1.25 | 54.7% (+2.3) | 73.9% | 0.89 | 4.16 | 0.51 s | 2.97 | 83.4% |
| `temp` 1.5 | 56.1% (+3.7 [+2.6, +4.7]) | 74.2% | 0.89 | 4.76 | 0.49 s | 2.38 | 85.0% |
| `temp` 2.0 | 55.4% (+3.0) | 73.5% | | | | | |
| `blank_bias` 0.5 | 45.6% (-6.7) | 70.0% | | | | | |
| `blank_bias` 2.0 | 56.9% (+4.5) | 74.7% | 3.27 | 5.05 | 0.49 s | 2.38 | 82.9% |
| `temp` 1.5, `quiet_pen` 2 | 57.0% (+4.7) | 74.5% | 0.59 | 5.05 | 0.47 s | 2.38 | 85.9% |
| **`temp` 1.5, talk and salawat costs / 1.5** | **55.7% (+3.3 [+2.4, +4.3])** | **74.0%** | **0.59** | **4.16** | **0.55 s** | **2.38** | **83.9%** |

`tracker_weight` 0.15 to 0.8, `show_p` 0.25 and `lookahead` 0.3 moved right word by -0.1 to
+1.0 (the last by showing words 0.1 s later). A warmer letter model (`temp`, which divides its
log posteriors) and a dearer blank (`blank_bias`) do the same job, but the dearer blank makes
more jumps (2.08 -> 3.27 /10 min); the warmer model halves them.

## The reciters: talk and salawat

On the bench's dev reciters (1,317 items), `temp` 1.5 alone gave right word +0.7 [+0.6, +0.8], but
the highlight held still worse: stays put 89.1% -> 85.2%, talk between lines 86.4% -> 78.3%,
salawat 83.2% -> 78.2%, early moves 1.92 -> 2.34 /10 min. The temperature divides the letters'
log posteriors, the quiet penalty with them, but not the costs that weigh the du'a against
talk and salawat (the filler's entry and per-letter cost, the interjection chain's), so talk
became cheaper to read as the du'a. Taking those costs down by the same 1.5 (`fill_cost` 0.667,
`c_fill_in` -5.333, `c_fill_word` and `c_fill_next` -1.333, `c_int_in` -2.667, `c_int_next` and
`c_int_word` -0.667) leaves talk and salawat where they stood and the du'a's own moves stronger.
On the dev scenarios where holding still is measured (pause, talk, salawat, repeat, flow; 435 items):

| | stays put | talk | pause | salawat | jumps /10 min | right word |
|---|---:|---:|---:|---:|---:|---:|
| today's | 89.2% | 86.4% | 92.8% | 83.2% | 0.39 | 76.7% |
| `temp` 1.5 | 85.2% | 78.3% | 91.2% | 78.2% | 0.41 | 77.0% |
| `temp` 1.5, `quiet_pen` 6 | 86.1% | 79.2% | 92.3% | 78.6% | 0.40 | 76.4% |
| **`temp` 1.5, talk and salawat costs / 1.5** | **89.0%** | **87.1%** | **93.5%** | **80.1%** | **0.17** | **76.8%** |

Salawat still holds still 3 points less; leaving the interjection costs unscaled changes nothing
there (80.1% / 79.9%).

## The whole bench

`temp` 1.5 with the talk and salawat costs / 1.5, against today's (`n27_dev_base`, `n27_test_base`):

| | dev (1,317 items) | test (593 items) |
|---|---|---|
| on line | 89.9% -> 89.7% (-0.2 [-0.2, -0.1]) | 90.5% -> 90.4% (-0.1 [-0.2, +0.0]) |
| right word | 74.5% -> 74.9% (+0.4 [+0.3, +0.4]) | 76.0% -> 76.5% (+0.5 [+0.3, +0.7]) |
| jumps /10 min | 1.26 -> 1.05 | 1.49 -> 1.30 |
| early /10 min | 1.92 -> 1.92 | 1.93 -> 2.08 |
| lost /10 min | 1.16 -> 1.23 | 1.03 -> 1.00 |
| stays put | 89.1% -> 88.8% | 90.4% -> 90.2% |
| follows a jump within 3 s | 82% -> 81% (median 0.8 -> 1.0 s) | 83% -> 81% (median 0.8 -> 1.0 s) |

Guard bars that fail (`bench.py guard`): on dev, studio lost +0.12 /10 min and stays put 84.7% ->
83.6%, majlis lost 0.47 -> 0.58; on test, skips followed within 3 s 89.0% -> 83.1% and jumps
70.8% -> 66.4%, stays put in studio 86.3% -> 83.0% and majlis 88.4% -> 84.9%. A decoder that
trusts the reading order more makes fewer false jumps and follows a real one later. (The test
voices were looked at in the jump work of 2026-10-04.)

## The real page

`data/cache/n27/page_sessions.py` (`bench_page.py` on the 35 session items; Whisper held to 1.5 s
a step, the CTC model to 0.1 s; both variants of an item played at the same time; the
energy-assisted gate in both, [browser_gate.md](browser_gate.md)). Twice, each time against the
earlier values played alongside:

| real page, 35 session items | on line | right word | stays put (his pauses) | jumps / early / lost |
|---|---:|---:|---:|---:|
| earlier values, play 1 | 75.5% | 51.8% | 92.4% | 4 / 17 / 12 |
| `temp` 1.5, talk and salawat costs / 1.5 | 74.2% | 53.1% | 89.6% | 3 / 15 / 11 |
| earlier values, play 2 | 73.5% | 50.5% | 92.7% | 5 / 14 / 11 |
| the same with `quiet_pen` 6 | 75.2% | 53.7% | 90.0% | 3 / 14 / 8 |

The same settings played twice differ by 2.0 points on line and 1.3 on the word (play 1 against
play 2, an interval over items of -4.0 to -0.4 that says nothing about the next play): one play of
the real page is not enough to judge a point or two. What both plays agree on: the word +1.3 and
+3.2, and holding still in his pauses -2.8 and -2.7, with the quiet penalty as it was (4, which the
temperature takes down to 2.7) or raised to match (6). The Python replay of the same items shows
no such loss (stays put 83.7% -> 83.9%), so the cause is in what the page does that the bench
doesn't (its live stop detector, Whisper's timing on the page); it wasn't found tonight.

With `quiet_pen` 6 the Python bench gets worse too: sessions right word +2.3 instead of +3.3, line
lag 0.55 -> 0.61 s, lost 2.68 -> 2.97 /10 min; dev on line -0.3, right word -0.1; test on line -0.2,
right word +0.0, skips followed within 3 s 89.0% -> 80.9%.

## Decision: not adopted

Holding still when the reader stops is what this page was rebuilt around (the 2026-09-30 stops
fix), and on the real page this costs nearly 3 points of it in his own pauses, on top of the
reciters' guard bars. The defaults stay as they were. For a reader who wants to try it:

`?sc=temp:1.5,fillCost:0.667,cFillIn:-5.333,cFillWord:-1.333,cFillNext:-1.333,cIntIn:-2.667,cIntNext:-0.667,cIntWord:-0.667`

A version for reading mode only (Python and JavaScript defaults, majlis mode and the server
engine on the earlier values, the mode switch mid-session handled) is kept as a patch,
`data/cache/n27/reading_mode_trust.patch`, with the tests passing (183).

What to try next: find why the page moves in his pauses with the warmer model (replay one
session with `?sc=temp:1.5` and read `scripts/session_stops.py` stop by stop), and fix that rather than
the temperature; and judge real-page changes on two or more plays each.

## Reproduce

```
bash data/cache/n27/sweep.sh NAME "temp=1.5 ..."           # session items, against n27_sess_base
bash data/cache/n27/bench.sh dev NAME "temp=1.5,..."        # the scenario bench, then the guard
python data/cache/n27/cmp.py data/testbed/results n27_dev_base NAME... [--scenario talk]
python data/cache/n27/page_sessions.py --split all --per-scenario 99 --variants a:QUERY b:QUERY --jobs 2
```
