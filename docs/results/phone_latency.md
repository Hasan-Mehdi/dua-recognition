# The phone follower, faster: the last word, latency, repeats (2026-10-01, evening)

Hasan, after a Dua Faraj session on the phone with the new word follower (20261001-215013-x0fo):
"It is much better! The only issue I saw was that at the last part I said the final word but it
didn't recognize, so I had to say it again. That, and try to minimize latency while maintaining
or even improving current results." Then, after two Dua Kumayl sessions (l8s3, ukva): "it is
harder to track if the user likes to repeat the previous line".

## The last word

x0fo ends on *wa-tumattiʿahu fīhā ṭawīlā*. The highlight sat on *fīhā* from 36 s, through
*ṭawīlā* at 37.0-37.8 s and four seconds of silence, until he said *ṭawīlā* again at 41.5 s.

- The phone's CTC model heard his *ṭawīlā* as "ti" (ط as ت, the rest below blank). Neither
  wav2vec2 model hears it either: the forced alignment of the session ends on *fīhā*. The
  follower only moves on letters it hears, so it waited.
- Whisper heard it: "و تبطي عظمت فيها تولينا". Its result, 1.3 s late, put the tracker on
  *ṭawīlā* at 40.2 s. But the follower only takes the tracker's word when it is 4+ words ahead
  for 2 s (a rule for a follower that has lost its place), and the tracker was one word ahead.
- The same happened in 0t98 (the last word unlit for 11 s, to the end of the session). The
  session scorer had missed it: it stopped scoring one second after the last word.

**Fixes.** *Quiet catch-up* (`quiet_after` 0.5 s, `quiet_words` 3, follower.py and follower.js):
when the reciter is silent and the tracker's word is ahead of the follower's in the same line, by
at most three words, for half a second, the follower goes there. The page also runs Whisper as
soon as the reciter stops while the follower is following (once per stop), instead of up to 2 s
later, so that anchor comes 1.3 s after the stop.

The real page replaying x0fo's audio (`page_replay.mjs`, Whisper held to 1.3 s): *ṭawīlā* lit at
40.2 s with these two changes alone, 42.5 s on his phone; at **38.1 s** with the latency changes
below, where the 2 s model hears the word itself (0.4 s after he finished it).

## Latency

Where the delay went, from x0fo's log:

- The microphone audio reached the page in 250 ms chunks (`capture-worklet.js`): a word waited
  125 ms on average before any model could hear it.
- The CTC model took **127 ms** a step with Whisper idle and **304 ms** with Whisper running in
  the other worker, which it is 65% of the time while following (1.3 s every 2 s).
- Steps ran every 0.25 s (one per chunk).

How much compute delay costs (`follow_eval.py`, DuaPlayer train reciters, the follower's result
shown `--follow-delay` s after its window): 73.8% word exact at 0.1 s, **59.3% at 0.3 s**.
The morning's 79.3% on the test reciters assumed 0.1 s; at the phone's delays it was 68.6%
(Results, below).

| change | measured |
|---|---|
| audio chunks of 50 ms (`?chunk=`, 800 samples) | a word waits ~25 ms for its chunk instead of ~125 ms |
| a CTC step as soon as the last is done and 0.1 s of audio has come in (`?ctchop=`, was 0.2) | train, same delay: 73.8% → 77.6% word exact, line entry −0.05 s |
| the CTC model on 2 s windows instead of 3 s (`ctc-student-base-v6-w2`: the same weights, exported with `--window 2`) | 45 ms instead of 71 ms a step on one desktop thread (−37%); at equal delay −0.6 pt |

Fine-tuning the student on 2 s crops with a phone noise gate (`train_ctc_student.py --init
--window 2 --gate 0.2`, v8w2: his phone cuts to digital silence between words) was no better on
reciters and worse on his sessions (±1 word 82% vs 88%), so the exported v6 is the phone's model.
Running Whisper less often while following saves compute but hurt his sessions (every 3 s: word
exact 51% → 47%, behind in pauses 10% → 17%), so it stays at every 2 s. Sessions now log the
chunk size, `navigator.hardwareConcurrency` and cross-origin isolation, so the next session will
show how many cores his phone shares between the two models.

**Word lag** (how long after a word's aligned start the display reaches it; the morning's
settings at the phone's delays, 3 s windows every 0.2 s shown 0.25 s late, against the new ones,
2 s every 0.1 s shown 0.15 s late):

| | words within a line, median / p90 | a line's first word, median |
|---|---|---|
| train, before | +0.44 / +0.56 s | +0.46 s |
| train, now | **+0.28 / +0.42 s** | +0.44 s |
| test, before | +0.44 / +0.57 s | +0.46 s |
| test, now | **+0.29 / +0.46 s** | +0.46 s |

(The 50 ms chunks come on top: these assume the audio is there when the step starts. A line's first
word gains nothing on reciters: `line_confirm` and `line_quiet`, below, spend what the faster steps
won there on not showing the next line early.)

## Repeats

None of the benchmarks had a reader going back. `scripts/repeat_eval.py` builds one: after every
third line end of each recording it splices in a breath of the recording's own room tone and then
the line just said, again, then the recording goes on. The tracker's windows over a splice are
transcribed afresh with the phone's Whisper; the CTC model's windows are dumped over the whole
spliced audio; `follow_eval.py --repeats` scores it, with three repeat columns: how often the
display gets back to the line's first or second word during the repeat, how long that takes, and
word exact inside the repeats. Train split: 21 recordings, 769 repeats.

Two rules held the follower off a repeat:

- Going back cost 3 nats a word (`beta_back`), so back to the start of a 6-word line cost 15,
  and the scored text started only 6 words back.
- The forward re-anchor (`ahead_words_reanchor`: the tracker 4+ words ahead for 2 s takes the
  follower there) pulled it off the repeat: the tracker, slower to believe a step back, was still
  ahead.

**Line restarts** (`restart_cost` 3, `restart_lines` 1): back to the first word of the current
line or the line before costs at most 3 nats, however far; the scored text reaches back to them.
People start a line over or say the previous one again; they rarely jump back into the middle of
one. **Stuck-only re-anchors** (`ahead_stuck`, `reset_stuck`): the forward re-anchor, and the
re-anchor after 3 s of more than a line's disagreement, only while the follower's word hasn't
changed in that time. A follower moving word by word is hearing the reader (in 9puq the tracker,
two lines behind, pulled a follower that was on the right word back by two lines). `beta_back` 2,
and `back_confirm` 4 steps: at a step every 0.1 s that is the same 0.4 s as 2 steps at 0.2 s.

**His own repeat** (diqq, Kumayl, 17:56): line 109, then 110, then 109 again at 37 s and a third
time at 57 s. The phone went 109 → 110 → **113** → 111 → 109: at the repeat, the follower leapt to
line 113's *fa-kayfa* (line 109 begins *fa-kayfa ḥtimālī*), and only came back at 58.5 s. Replayed
from his log (frames at the logged step times, the logged anchors) with today's rules, it goes back
to line 109's first word at 38.2 s and follows *iḥtimālī li-balāʾi l-ākhirati wa-jalīli*. One more
rule came from it, **`repeat_hold`** 20 s: after a step back of a line or more, no tracker
re-anchor (forward, line, quiet catch-up) takes the follower forward until it is back where it was
or 20 s have passed. Whisper's tracker rarely believes a repeat; in diqq it moved on into line 111,
and at 42 s, with the follower pausing on his *al-ākhirati* (heard as "illā khīratī"), the forward
re-anchor took the display there. Neutral on flowing recitation, pauses and his sessions; on the
repeat set word exact inside repeats 51.3% → 51.8%, jerks 2.70 → 2.58 a minute. In e85w (Kumayl
39-54, 19:13) the phone jumped from line 46 back to 44 for two seconds: the tracker, two lines
behind, re-anchored a follower that was on the right word. With `reset_stuck` the replay goes
46 → 47 with no step back.

Train split (21 recordings, 769 repeats), each lane at its own phone delay as in the results below:

| | word exact | ±1 | jerks /min | back at the line's start | in | word exact inside repeats |
|---|---:|---:|---:|---:|---:|---:|
| before (3 s windows every 0.2 s, 0.25 s late, the morning's rules) | 52.7% | 77.4% | 3.12 | 31% of repeats | 1.44 s | 26.8% |
| now | **67.6%** | **88.5%** | **2.58** | **64%** | **0.85 s** | **51.8%** |

At the same windows and delay (2 s, 0.1 s, 0.15 s), today's rules alone: 63.5% → 67.6% word
exact, back at the line's start 35% → 64%, word exact inside repeats 32% → 52%. On flowing
recitation and the pause benchmark the restart and re-anchor rules change word exact by less
than 0.3 points; `line_confirm` (next section) costs about one.

Test split (held-out reciters, built the same way after the rules were chosen; one look):

| | word exact | ±1 | jerks /min | back at the line's start | in | word exact inside repeats |
|---|---:|---:|---:|---:|---:|---:|
| before | 58.3% | 82.2% | 2.45 | 34% of repeats | 1.45 s | 27.6% |
| now | **68.7%** | **88.2%** | **2.13** | **66%** | **0.86 s** | **51.8%** |

## Entering the next line during a pause

At a step every 0.1 s on 2 s windows, one window can hear a line's last word as the next line's
first: 0t98's *ṭawʿā* came out "تـوا", matched to *wa-tumattiʿahu*, and line 9 showed in the pause
after line 8. On the pause benchmark the next line was shown in 2.2% of pause time against 1.1%
before. `line_confirm` 2: a step into a later line must win two steps in a row. That brings it
back to 1.0%, for 0.11 s later line entries (within a line, nothing changes). On his sessions, at
three assumed delays, it also kept the follower from getting stuck behind (±1 word 88% at every
delay, against 82-89% without). Waiving it when the new line's letters follow a pause didn't
separate the cases: CTC frames have blank runs between words too, and the false entries in the
pause benchmark come out of the room tone, after a gap.

The held-out pause set disagreed with train: with `line_confirm` 2 alone the next line showed in
3.2% of pause time there, against 1.5% for the morning's follower. Not the 2 s window: at the same
delay 3 s windows do the same on train (0.9% vs 1.0%). **`line_quiet`** 0.3 s: no step into a later
line while the page's stop detector (the voice-relative one from [stops.md](stops.md), `gate.js`
`LiveQuiet`, passed to the follower as `quiet_now`) has heard no voice for 0.3 s. Letters the
model makes up out of room tone can match the next line's first word; a real start of the next
line comes with a voice. Train pauses: next line in pauses 1.0% → 0.6%, line entries 0.03 s later;
with `line_confirm` 1 instead it gives 1.3% at 0.08 s sooner. Both are on: the fewer early lines.
In the benchmarks the detector is the same rule on the recording's audio, with the voice level its
95th-percentile frame (the page remembers it from Whisper's windows); in `session_eval.py` it is
the page's own, rebuilt from the session (`session_stops.renders`). On his sessions it changes
nothing measurable.

## Results

DuaPlayer reciters, the phone's Whisper anchors (shown 1.2 s late). *Before*: the morning's
follower as the phone ran it (3 s windows, a step every 0.2-0.25 s, shown 0.25 s late). *Now*:
2 s windows, a step every 0.1 s, shown 0.15 s late, today's rules.

| flowing recitation | word exact | ±1 | jerks /min | line entry | > 0.3 s early |
|---|---:|---:|---:|---:|---:|
| train, before | 63.0% | 94.3% | 0.46 | +0.79 s | 0.5% |
| train, now | **72.2%** | **95.3%** | 0.63 | +0.78 s | 0.5% |
| test, before | 68.6% | 97.0% | 0.08 | +0.78 s | 0.5% |
| test, now | **75.5%** | **97.4%** | 0.17 | +0.79 s | 0.5% |

| pauses (4 s after every third line) | word exact | jerks /min | line entry | next line shown in pauses | behind in pauses |
|---|---:|---:|---:|---:|---:|
| train, before | 66.9% | 0.40 | +0.78 s | 1.2% | 0.9% |
| train, now | **74.8%** | 0.56 | +0.76 s | **0.6%** | **0.6%** |
| test, before | 71.2% | 0.09 | +0.78 s | 1.5% | 0.4% |
| test, now | **77.7%** | 0.16 | +0.78 s | **0.8%** | **0.1%** |
| (test, with `line_confirm` but before `line_quiet`) | 77.9% | 0.16 | +0.73 s | 3.2% | 0.1% |

Line entries come no sooner than in the morning (+0.78 s against +0.78-0.79 s): what the faster
steps won there, `line_confirm` and `line_quiet` spend on not showing the next line early. Words
within a line come 0.16 s sooner (Word lag, above), and on his voice line entries do come sooner
(+0.84 → +0.58 s, below).

All "now" rows are the final defaults (`data/cache/latency/final2.sh`; the "before" rows
`final.sh` and `final_test.sh`).

The test split was looked at once for flowing recitation and repeats, with everything already
chosen on train and his sessions; the pause set twice: with `line_confirm` alone (3.2%, the row in
brackets), which led to `line_quiet`, chosen on train, then with it. Jerks (a step back, or more than two words forward) rise a little: restarts make a step
back cheaper, and a step every 0.1 s sees more near-ties. On his sessions, steps back a line stay
where they were (0.49 → 0.53 a minute: three over the eight sessions, one where he really went back).

**Hasan's sessions** (8: the 7 from the morning's write-up and x0fo; word truth from the
wav2vec2 model fine-tuned on ordinary voices, which aligns his voice further than the reciters'
one, e.g. x0fo's line 9; scored until 4 s after his last word). Short sessions are path-sensitive
(one decision changes the rest of a session), so each lane is the mean over three assumed phone
delays (before 0.2/0.25/0.3 s, now 0.1/0.15/0.2 s):

| | word exact | ±1 | line | line entry | behind in pauses | ahead in pauses | line backs /min |
|---|---:|---:|---:|---:|---:|---:|---:|
| before | 48.8% | 85.3% | 65.9% | +0.84 s | 15% | 23% | 0.49 |
| now | **57.5%** | **88.5%** | **68.7%** | **+0.58 s** | **5%** | 27% | 0.53 |

**Through the real page** (`page_replay.mjs`, Whisper held to 1.3 s and the CTC step to 180 ms,
about his phone with both models busy): x0fo lines 3 → 9 and 0t98 lines 2 → 9 with no step back,
both with the last word lit; 797f lines 3 → 9 with no step back, line 9 at 29.1 s (the morning's
replay waited on *tuskinahu* and reached line 9's last word at 32 s; the phone showed
7 → 8 → 7 → 8 → 9 the night before).

The "ahead in pauses" column counts the time the display is past the last aligned word in gaps of
1 s or more; with this truth it includes words the aligner left out (x0fo's ṭawīlā, now lit, was
such a word before the truth model change), so it overstates.

## Dua Kumayl: not recognised

l8s3 and ukva are Kumayl lines 109-112 and back to 110. The phone never showed the du'a. Its
Whisper's transcripts were too far off ("و أجهي و قويا أقومي فيها و هو", "مددة و لا يؤمن و لا
يرحم", "مغفرت مغفرت مغفرت"); Kumayl reached 0.695 for one update in l8s3, under the 0.7 bar, and
fell back to "not in the corpus". The other phone-size Whisper models do no better (syn-v6 and
aug-v4 show it for 4-5 of 45 updates), lower bars don't rescue it (0.6: 2 updates of 26), and the
server's fine-tuned turbo places only 54% of l8s3's seconds. The CTC student hears almost nothing
in it ("ين", "ت" from 3 s windows), and scoring its windows against the whole corpus doesn't find
the du'a even in x0fo: three seconds of letters is too little text.

*Verifying* one candidate is easier than searching. Each 3 s window's best CTC path through the
candidate's text (Kumayl lines 107-113), against the same through 150 random passages of the same
length, as a z-score: x0fo against Faraj +3 to +10 in most windows; l8s3 against Kumayl +2.8 to
+4.5 in half its windows; ukva against Kumayl in 2 of 13. But x0fo and 4t6h, which are not
Kumayl, also reach +3 against it in one window or two. A verifier would have to add the evidence
up over time inside the tracker and be calibrated on speech outside the corpus (scripts/ooc_eval.py)
so that it never shows a wrong du'a; not done.

What changed: a du'a once offered as "Is it…?" stays offered, faded, for 12 s after its
likelihood drops (`GUESS_KEEP_MS`). In l8s3 the Kumayl chip was up for 3 s; with it kept, a tap
locks the page to Kumayl, and the follower then needs only to find the line.

## What changed

- `web/capture-worklet.js`, `web/app.js`: 50 ms chunks (`?chunk=`); 2 s CTC models
  (`ctc-student-base-v6-w2`, slow-phone fallback `ctc-student-tiny-v6-w2`, 19 ms vs 30 ms); a CTC
  step every 0.1 s (`?ctchop=`); a stored fallback from before (`ctc-student-tiny-v6`) is ignored;
  Whisper runs at a stop while following; guesses kept 12 s; the session log records `chunk`,
  `cores`, `isolated`.
- `follower.py` / `follower.js` (parity-tested on 14 synthetic cases, and step for step
  on diqq, e85w and x0fo's real frames): `quiet_after`, `restart_cost`, `restart_lines`,
  `ahead_stuck`, `reset_stuck`, `line_confirm`, `repeat_hold`, `line_quiet` (`step(...,
  quiet_now)`, from `app.js`'s `this.ear.quiet`); defaults `beta_back` 2, `back_confirm` 4.
- `pipeline.py` (server): a word step every 0.1 s.
- Tools: `scripts/repeat_eval.py`; `follow_eval.py --repeats --ctc-window --ctc-hop --anchor-every`
  and a "behind in pause" column; `session_eval.py --window --truth-model`, scored to 4 s after
  the last word, with "behind in pauses"; `train_ctc_student.py --init --gate`;
  `session_stops.py` reads the logged chunk size.

## Not done

- His phone's cores and the two workers' threads: each ORT worker takes its default, and the CTC
  step is 2.4× slower while Whisper runs. The next session's log (cores) decides whether to pin
  threads (`?ctcthreads=` exists).
- Kumayl from mid-du'a on his voice (above).
- Server mode (`pipeline.py`) runs the same follower but doesn't pass it a `quiet_now`, so
  `line_quiet` acts only on the phone.
- The phone's CTC model still mishears his ط as ت and drops ع; more of his own sessions,
  labelled, are the most direct data for that.

## Reproduce

```
python scripts/dump_ctc.py --model models/ctc-student-base-v6 --window 2 --hop 0.1
python scripts/export_ctc_student.py models/ctc-student-base-v6 --window 2 --name ctc-student-base-v6-w2
python scripts/export_ctc_student.py models/ctc-student-tiny-v6 --window 2 --name ctc-student-tiny-v6-w2
python scripts/repeat_eval.py build --split train --tag whisper-base-syn-v5-ctx8ft --model models/whisper-base-syn-v5-ctx8ft-ct2
python scripts/repeat_eval.py build --split train --tag whisper-base-syn-v5-ctx8ft --ctc-model models/ctc-student-base-v6 --ctc-window 2 --ctc-hop 0.1
python scripts/pause_eval.py build --tag whisper-base-quran-dua --split train --ctc-model models/ctc-student-base-v6 --ctc-window 2 --ctc-hop 0.1
bash data/cache/latency/final.sh          # the results table
python scripts/session_eval.py --ctc models/ctc-student-base-v6 --window 2 --hop 0.1 --delay 0.15 \
    --truth-model wav2vec2-quran-dua-voices 9puq fcr2 s4pv 0t98 h649 4t6h 797f x0fo
node scripts/ctc_bench.mjs threads=1 models/ctc-student-base-v6/model_q8.onnx:80x300 models/ctc-student-base-v6-w2/model_q8.onnx:80x200
node scripts/page_replay.mjs data/sessions/20261001-215013-x0fo.wav out.wav "" --asr-ms 1300 --ctc-ms 200
```

Logs: `data/cache/latency/`.
