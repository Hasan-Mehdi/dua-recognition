# Finding the du'a, hearing, and the harvest retrains (2026-10-03)

Hasan: "Go ahead with the bigger projects, make smart planned decisions ... focus on big picture
instead of tiny fixes." Everything here is judged on the scenario bench (`scripts/bench.py`, the
stream decoder as the page runs it), tuned on the dev voices and reported on the held-out test
voices.

## Where the losses are: a perfect ear

`bench.py score --asr truth` replaces Whisper's per-second transcript with the true words of the
same 8 s window, everything else unchanged (rows made by the scratch script that wrote
`data/testbed/asr/truth`). All 1,666 items:

| | Whisper (phone model) | perfect ear |
|---|---:|---:|
| on the reading line | 88% | 90% |
| du'a found within 10 s | 86% | 91% |
| found, hall / far (synthetic echo, distance) | 59% / 62% | 89% / 88% |
| jumps / early moves / lost per 10 min | 1.94 / 2.37 / 2.07 | 1.81 / 2.41 / 1.46 |

- **Hearing** costs 5 points of finding overall, but 25-30 in echo, distance, clipping and babble.
- **The search** misses 9% of items even with perfect text. Two reasons, read off traces:
  passages several texts share word for word (Ayat al-Kursi sits in Sahifa 54, Namaz-e-Wahshat
  and an Eid al-Mubahila text: no display can tell them apart, yet the bench called a correct
  screen "wrong du'a"), and readings that open with 20 s of basmala, salawat and "اللهم إني
  أسألك", which hundreds of texts share.
- **Jumps and early moves don't move with perfect hearing.** They come from the display logic
  (the stream decoder and its anchors), not from Whisper: the next lever after finding.

## Scoring fixed: `--same-text 8`

A display counts as right where it shows the same 8 words the reader's du'a reads there (the
tracker's own same-text notion), and a second clock, `found≤10s*`, starts at the first 8-word run
no other text reads. Off by default, so older results stay comparable. With it, the dev voices
are found within 10 s 89% of the time and 91% on the distinctive clock (test: 88% and 93%): part
of the "finding" gap was the test.

## Prior over du'as by how often each is recited (adopted, default 0.5)

`data/dua_popularity.json` (`scripts/dua_popularity.py`): harvest recordings per text, counted only
where a recording reads 3+ lines no other text reads (else the labeller's arbitrary pick among
identical passages would count). Kumayl 794, Ziyarat Ashura 580, Warith 477, ... most duas.org-only
texts 0. `TrackerConfig.popularity`: P(du'a) proportional to (recordings + 1) ** popularity, in
tracker.py and tracker.js (parity-tested).

| | dev: 0.5 | dev: 1.0 | **test: 0.5** |
|---|---:|---:|---:|
| found within 10 s | +1 | +2 | **+2** (88 -> 90%) |
| found from the first distinctive words | +1 | +2 | **+1** (93 -> 94%) |
| on the reading line | 0 | 0 | **+1** (90 -> 91%) |
| wrong du'a | 0.7 -> 0.5% | 0.7 -> 0.4% | **0.9 -> 0.5%** |
| unknown du'a shown | +1 | +4 | **+1** |
| Hasan's sessions (dev only) | even | jumps 3.1 -> 7.3 /10 min | |

1.0 found more, but in Hasan's two Namaz-e-Wahshat sessions it held the more recited text sharing
Ayat al-Kursi after the passage ended. 0.5 is the default since `e8e3589`.

## The phone CTC student retrained on the harvest (h1): not adopted

`ctc-student-base-h1` failed its pre-registration last night on component metrics; on the whole
bench it is no better either: on line equal, words +1, but early moves +0.2 /10 min and "stays
put" -3 pts on held-out voices (-12 on the studio test reciters). The small model's weak spot is
holding still, not hearing more voices.

## The phone Whisper retrained on the harvest (whisper-base-h2c-ctx8): failed its pre-registration

Pre-registered before training (`data/cache/whisper_h2c/prereg.md`): continue the phone model
(`whisper-base-syn-v5-ctx8ft`) for one epoch on train_v4 + 67,325 voice-balanced harvest windows
(2,555 uploaders; `scripts/build_clip_cache.py` keeps them as 1.6 GB of Opus clips instead of 200 GB
of decoded recordings) + synthetic voices 15%, with the room reverb widened to RT60 0.3-1.5 s.
26 min of training.

| test voices (617 items) | syn-v5-ctx8ft | h2c | bar |
|---|---:|---:|---|
| found from the first distinctive words | 93% | 94% (+1) | +2 needed: **fail** |
| on the reading line | 91% | 91% | >= -0.5: pass |
| majlis (real PA echo, crowds) on line | 90% | 90% | >= -1: pass |
| wrong du'a | 0.9% | 0.9% | pass |
| RetaSy crowd voices <= 8 s, CER | 24.1% | 23.7% | pass |
| found, far / hall / clip (synthetic) | 40 / 63 / 74% | 67 / 75 / 87% | (reported) |
| unknown du'a shown | 48% | 51% | (reported) |

On dev the same: hall found +23, on line +7; Hasan's sessions on line +4 but jumps 3.5 -> 5.6 and
wrong du'a 2 -> 4%. The gains sit in the synthetic echo/distance cells, built from the same kind
of reverb the training added; real echo (majlis) shows none. Not adopted; the model stays in
`models/whisper-base-h2c-ctx8` for a real-room test. On top of the popularity prior (test, vs the prior
alone) it adds the same: found +1, far +27, hall +12, unknown du'a shown +2 (results `pop05_test_st8`,
`h2c_pop05_test_st8`).

## The added Mafatih texts, all 17 tested

Lane `mafatih` now has 3+ voices for every added text (uploaders outside the held-out side where
it had none; listed in test_voices.json): 507 items, 22 voices. On the reading line 93-96% for
every text; found within 10 s 100% for 15 of the 17 (Sabah 94%, Munajat Shakeen 71%); wrong du'a
0-0.7%.

## Next

- **Display logic**: jumps (2 /10 min) and early moves (2.4; 7-9 when people talk or repeat) are
  the largest losses perfect hearing doesn't touch.
- **Real echo data**: a real-room check of `whisper-base-h2c-ctx8` (a phone across a hall)
  decides whether its synthetic-echo gains are real.
- **Unknown du'as**: 35-51% shown; the tracker's "not in the corpus" state is the lever.
