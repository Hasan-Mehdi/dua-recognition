# Lines out of order: finding a reader who jumps (2026-10-02)

Hasan: "Curate a test set where the reciter recites random lines out of order and see how well
that tracks. I kinda want everything to work perfect."

## The benchmark

`scripts/jump_eval.py` rebuilds each DuaPlayer recording as a reading that jumps: its first three
lines in order (so the du'a is found), then a jump to a random other line of the same du'a, one to
three lines read on from there, a jump again, for about 2.5 minutes, with a breath of the
recording's own room tone (0.3-1.2 s) before each jump. Lines are cut at their forced-aligned word
boundaries, and only lines whose every word is aligned are used. Jumps go back and forward, near
and far, never to the line that comes next anyway. All the audio is new, so the phone's Whisper
transcribes every tracker window and the phone's CTC model is run over it.

Train: 20 recordings, 255 jumps, 56 minutes. Test (held-out reciters): 14 recordings, 157 jumps, 40 minutes.
`follow_eval.py --jumps` scores it as the other sets, plus the share of time on the right line and,
per jump, whether the display gets to the line jumped to before the next jump, and how long that
takes.

## Before

| train, yesterday's defaults | word exact | right line | jumps found | within 3 s | median time to find |
|---|---:|---:|---:|---:|---:|
| | 33.7% | 33.5% | 28% | 6% | 6.7 s |

The follower's reach is six words back (or the start of this line and the one before) and twenty
ahead, so after a jump it waits for the tracker. The tracker could only jump through its teleport
floor, spread over all 505 texts, so a jump within the du'a was worth almost nothing a priori; and
once it had moved, the follower's re-anchor rules (3 s of more than a line's disagreement) added
their own delay.

## What changed

**The tracker** (`TrackerConfig.p_line_jump` 0.02 a second, `tracker.js` `pLineJump`): a jump to
the first word of any line of the same du'a. Neutral where nothing jumps: line accuracy on the train
reciters 84.3% either way; a du'a missing from the corpus shows some other du'a 34.0% of the time
against 33.8% (`ooc_eval.py`).

**The follower's jump search** (`follower.py` `_jump_search`, `follower.js` `_jumpSearch`): every step
the latest 1.2 s of CTC frames are also scored against the whole du'a. A place outside the
follower's reach that explains them better than every word within reach by `jump_margin` nats, on
`jump_confirm` steps in a row, is gone to; for `jump_hold` (6 s) the tracker can't re-anchor the
follower, since Whisper's word about the jump comes seconds later. Where the text occurs in several
places, the copy nearest the tracker's word. On the phone it costs 1 ms a step for Kumayl and 3 ms
for Jawshan Kabir (desktop; a few times that on a phone), on the page's main thread.

The catch is false jumps in ordinary reading: 1.2 s of audio the small model half-hears can fit a
far place better than the right one nearby. At 5 nats and two steps, 70 of the follower's 98 jumps
in 5.8 hours of ordinary train reading were wrong, a jump to a far line every five minutes; 40 of
them in Hadith al-Kisa, whose lines repeat formulas ("fa-qāla wa-ʿalayka s-salām…").

Tried to tell real jumps from false ones:

| idea | outcome |
|---|---|
| Require the tracker's word near the target (`jump_alone`) | the tracker is 4-6 s behind a real jump: fewer real jumps found, false jumps barely fewer |
| Require the tracker's belief in the target's line (`jump_mass`, `Tracker.line_mass`) | the same: 74% of jumps found against 85%, false jumps 0.76 against 0.81 a minute |
| The follower still before the jump | no: false jumps come when it is already stuck (median 1.2 s still, real ones 0.9 s) |
| Keyword spotting: the target must match from a line's first letter, the audio before it at a fixed cost per frame | worse: 1.5-13 jumps a minute in ordinary reading |
| Two-part scoring: the window explained by the text within reach, then the target from a line's start (`ctc_align.jump_scores`, `jump_seg`) | worse in ordinary reading (0.93-2.7 a minute) |
| Extra margin where the target's phrase (up to three words) occurs elsewhere in the du'a (`jump_repeated`) | **kept**: the landing text of 42% of false jumps recurs, of 9% of real ones |
| More steps in a row, a higher margin | **kept**: the trade below |

(train; jumps/min counts every step back or more than two words forward, so on the jump set it
includes the real jumps.)

Train, with the tracker's line-jump prior on (`data/cache/jumps_eval/final_jump.sh`):

| follower | jump set: jumps found | within 3 s | right line | ordinary reading: jumps/min (0.64 without) |
|---|---:|---:|---:|---:|
| no jump search | 22% | 6% | 29% | 0.64 |
| `jump_margin` 5 | 85% | 74% | 72% | 0.81 |
| 6, four steps, `jump_repeated` 4 | 76% | 65% | 66% | 0.71 |
| **7, four steps, `jump_repeated` 4** | **71%** | **57%** | **63%** | **0.68** |
| 6, six steps, `jump_repeated` 4 | 70% | 55% | 61% | 0.67 |

**Chosen**: `jump_margin` 7, `jump_confirm` 4, `jump_repeated` 4: about one extra jump in half an
hour of ordinary reading, against one every five minutes at 5 nats; most people read in order.

## Results

The defaults now (`data/cache/jumps_eval/final3.sh`), against yesterday's (no jump search, no
line-jump prior), at the phone's delays as in [phone_latency.md](phone_latency.md):

| lines out of order | jumps found | within 3 s | median time to find | right line | word exact |
|---|---:|---:|---:|---:|---:|
| train, before | 28% | 6% | 6.7 s | 33.5% | 33.7% |
| train, now | **71%** | **57%** | **1.4 s** | **63.0%** | **57.6%** |
| test, before | 35% | 6% | 7.3 s | 32.9% | 32.4% |
| test, now | **59%** | **40%** | **1.5 s** | **50.0%** | **47.1%** |

Held-out reciters find fewer jumps than train (59% against 71%): the margin was chosen on train to
keep false jumps rare, and on these readers more real jumps fall under it. A jump not found by the
search is found by the tracker, more slowly (the p75 is 5 s).

Nothing else moved (test): flowing recitation 75.5% word exact (75.5% before), 0.18 jumps a minute
(0.17); pauses 77.8% (77.7%), next line in pauses 0.8% (0.8%); repeats 69.0% (68.7%), back at the
line's start 66% (66%). Hasan's eight sessions score the same with and without the jump search
(57.4% word exact at 0.15 s): it never fires on them. His Kumayl sessions with a repeat (diqq) and
a long run (e85w), replayed from their logs, make the same moves with and without it.

## Not done

- A jump the follower misses waits for the tracker, which takes several seconds to believe it.
- The small CTC model's hearing sets the limit: a better model would let the margin come down.
