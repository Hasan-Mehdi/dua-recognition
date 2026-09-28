# Three phone sessions: a shared passage spelled differently, a lock that never held, taps that did nothing (2026-09-27)

The first on-device test on a real phone (Android 10, Chrome 153, the page's WASM engine,
whisper-base-aug-v4 q8): one Ayat al-Kursi, two Dua Faraj. It "got stuck on a line, was slow
to recognize, and hopped around a bit". The sessions are in data/sessions (mesx, 9puq, yvqf;
private audio, not committed).

## How fast the phone is

| session | Whisper per 6 s window (p50) | one update every | evidence age when shown (p50 / p95) |
|---|---|---|---|
| mesx (Ayat al-Kursi) | 2.07 s | 2.0 s | 2.2 s / 3.1 s |
| 9puq (Faraj), right after | 2.82 s | 3.0 s | 3.1 s / 3.4 s |
| yvqf (Faraj, `?webgpu`) | 20.5 s, one window | - | - |

The page is cross-origin isolated (WASM threads available). `?webgpu` gave one window in 20 s,
and its text was garbage (Latin tokens, broken UTF-8): the q8 weights don't run correctly on
WebGPU. Testing WebGPU needs an fp32/fp16 export.

On the train split, the phone's cadence alone (`--stride 3 --latency 2.5`, phone model) costs
3.4 points of line accuracy (83.3% -> 79.9%) and most of the early identification (at 5 s,
77% -> 55%). Faraj, replayed with the phone model every 1 s: found at 5.3 s; on the phone,
13.8 s.

## What went wrong in the Ayat al-Kursi session

`scripts/session_report.py mesx --timeline`, and the Python tracker re-run over the phone's
own transcripts (identical to what the phone showed):

1. **The same verse, spelled two ways.** The corpus has Ayat al-Kursi on its own
   (`duasorg-namaz-e-wahshat`), inside Sahifa 54, and at the start of "Dua - repentance"
   (`duasorg-eid-e-mubahila-3`). That last one writes "و لا", "و ما", "بشي ء" (and "يوده" for
   "ييوده"). Word for word, its longest identical run with Ayat al-Kursi is 13 words, and
   mostly 1-10, so the shared-passage rule (docs/results/shared_passages.md, 12 identical
   words) grouped it only now and then. The mass split ~55/40 between the two texts, and
   for 20 s neither reached the 0.7 needed to show a line.
2. **So the tracker never counted as locked.** The lock asks for one du'a at 0.95. With the mass split,
   the tracker stayed in search mode (kappa 1.2, 8x the locked 0.15) the whole time, and in
   search mode two misheard windows ("شهور النامين يوم يام") sent the "not in the corpus"
   state from 2% to 60% to 100%. The page kept the last line up: stuck on line 4 for 40 s
   while the reciter went on to line 9.
3. **Taps did nothing.** "I'm here" on lines 6, 7, 8, 9 was logged for debugging only.

The step back at the end (line 10 -> 9) is the pause rule (retreat_after), working as meant.

## Changes

- `same_text_spelling` (TrackerConfig, tracker.js `sameTextSpelling`): passages are compared
  with a lone و joined to the word after it and a lone ء to the word before, words of 4+
  letters one edit apart count as the same, and the two beliefs may peak one word apart (a
  text with a lone و has one more word to put mass on). Ayat al-Kursi vs "Dua - repentance":
  one run through the whole verse.
- `lock_on_passage`: a shared passage counts as one du'a for the lock too, as it already
  did for the display.
- `Tracker.seek(dua, segment)` (py + js), `StreamingRecognizer.seek`, a `seek:` WebSocket
  message: tapping a line while listening puts the tracker and the display there (logged as
  a `seek` event; session_report replays it).

Tried and dropped: measuring the lock without the "not in the corpus" mass (to break the
null -> unlock -> sharper kappa loop). With the du'a given it left the tracker always locked,
never in search mode: du'a-given line accuracy 84.1% -> 81.2%, identification at 3 s
71% -> 50% (train, phone model).

## Measured

Criteria set before any run (STATUS.md): on every configuration, line accuracy no more than
0.3 points lower and the wrong-du'a rate no more than 0.3 points higher than without the
changes; a du'a missing from the corpus shown no more than 2 points more often
(scripts/ooc_eval.py). Then one look at test with the same criteria.

Train (23 recordings, 6 voices, 6.0 h):

| ASR, cadence | | line acc | ±1 | wrong du'a | id @3 s | @5 s | @10 s | @20 s | @30 s |
|---|---|---|---|---|---|---|---|---|---|
| phone model, 1 s | before | 83.3% | 96.2% | 0.4% | 71.1% | 77.4% | 86.7% | 91.3% | 92.0% |
| | after | 83.3% | 96.1% | 0.5% | 70.0% | 76.7% | 86.7% | 90.9% | 92.0% |
| phone model, 3 s, shown 2.5 s late | before | 79.9% | 95.8% | 0.2% | 3.0% | 55.4% | 79.1% | 88.7% | 91.1% |
| | after | 79.9% | 95.9% | 0.1% | 3.0% | 55.2% | 78.9% | 88.7% | 91.1% |
| turbo (server), 1 s | before | 83.5% | 96.8% | 0.4% | 73.9% | 80.0% | 88.7% | 94.8% | 96.1% |
| | after | 83.5% | 96.8% | 0.4% | 72.6% | 79.1% | 88.5% | 94.1% | 95.7% |

Du'a missing from the corpus, some du'a shown: 27.3% before, 27.6% after (train, turbo).

Test (16 recordings, 6 voices, 4.0 h; looked at once, after train passed):

| ASR, cadence | | line acc | ±1 | wrong du'a | id @3 s | @5 s | @10 s | @20 s | @30 s |
|---|---|---|---|---|---|---|---|---|---|
| phone model, 1 s | before | 84.8% | 96.2% | 0.5% | 72.8% | 87.8% | 92.5% | 95.3% | 95.6% |
| | after | 84.8% | 96.2% | 0.5% | 72.8% | 87.8% | 92.5% | 95.3% | 95.6% |
| phone model, 3 s, shown 2.5 s late | before | 83.9% | 96.3% | 0.2% | 3.1% | 47.2% | 85.3% | 93.1% | 96.2% |
| | after | 83.9% | 96.3% | 0.2% | 3.1% | 47.2% | 84.7% | 92.8% | 95.9% |
| turbo (server), 1 s | before | 84.0% | 96.3% | 0.4% | 75.9% | 88.4% | 92.5% | 95.3% | 96.9% |
| | after | 84.0% | 96.3% | 0.4% | 75.9% | 88.4% | 92.5% | 95.3% | 96.9% |

Du'a missing from the corpus, some du'a shown: 27.1% before, 27.4% after (test, phone model).
Passes on both splits.

As with the first shared-passage rule, the DuaPlayer recordings barely touch a long shared
passage, so these sets can only show that nothing else got worse; they can't show the case
that prompted the change. Identification from a random point mid-recitation is 0.2-1.3 points
lower (it isn't one of the criteria): a start inside a shared passage now locks on the group,
and the gentler locked kappa takes longer to single out one text after it.

On the Ayat al-Kursi session itself (phone transcripts through the Python tracker), the du'a
is shown from 15.5 s instead of 19.8 s, and on 16 of 23 updates instead of 5; with the
spelling rule alone, 12 of 23. Still there: at the very end, where the two texts part, the
evidence (the phone heard "يؤون", nearer "يوده") briefly favours the other text and nothing
is shown.

Logs: data/cache/phone_fix/ (log.txt, one .txt/.json per run).
