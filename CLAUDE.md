# dua-recognition

Listens to someone reciting a du'a, works out which of 522 texts it is, and follows it line by
line and word by word: on the phone (web/, everything in the browser) or on a server (app/).
Owner: Hasan. Depth lives in docs/how-it-works.md, docs/evaluation.md, docs/development.md and
one write-up per experiment in docs/results/ (negative results included).

## Layout

- `src/dua_recognition/`: `align.py` (Myers alignment against the whole corpus), `tracker.py`
  (HMM over every word: du'a and line), `stream_follower.py` (the page's word follower),
  `follower.py` (the rule-based word follower), `ctc_student.py` (the small CTC model, on windows),
  `stream_ctc.py` (the same model streaming: the page's),
  `asr.py`, `pipeline.py` (the terminal demo's streaming recognizer), `display.py`.
- `web/`: the app. `tracker.js`, `stream-follower.js`, `follower.js`, `display.js`, `gate.js` and
  `ctc-stream.js` mirror the Python; `app.js` wires them; `corpus.json` comes from
  `scripts/export_web.py`.
- `app/server.py`: FastAPI server: the page, majlis rooms, debug sessions; in the server engine
  (`DUA_ENGINE=server`) also Whisper and the CTC model, which the page asks for over `/ws/ear`
  (`RemoteEar` in app.js). Either engine runs the same page code (`DeviceEngine`).
- `scripts/`: data, harvest, training, export, evaluation; `bench.py` is the scenario bench.
- `data/duas/`: the 522 texts (17 `mafatih-*` added 2026-10-03 by `scripts/mafatih_corpus.py`).
  `data/dua_popularity.json`: harvest recordings per text, the tracker's prior over du'as.

What the page runs by default (since 2026-10-07, docs/results/phone_engine.md): Whisper
`whisper-base-ph-kd2` (the phone's Whisper-base distilled from the server's turbo) on 6 s windows;
the tracker with `popularity` 0.5; the streaming CTC student `ctc-stream-base-v1`
(`stream_ctc.py`: each 0.1 s step encodes only the new audio, `web/ctc-stream.js` keeps its
caches; 50 ms chunks); the stream decoder with the phone profile as `StreamConfig`'s defaults
(copies, salawat lapse, frame push, shared-passage hold with edits, salawat-aware next-line gate).
`?follower=rules` brings back `follower.js`, `?ctc=ctc-student-base-v6-w2` the window student,
`?sc=k:v` and `?tc=k:v` set stream-decoder and tracker options by their JavaScript names, `?webgpu`
runs Whisper's encoder on the GPU (fp16). The server engine runs its own: `whisper-turbo-srv2`
(when present, with a GPU), the window student on 3 s windows every 0.05 s, and `SERVER_PROFILE`
in app/server.py, which the page takes from `/api/mode` (docs/results/server_profile.md).

## Tests

```
.venv\Scripts\python -m pytest          # PowerShell; .venv/Scripts/python.exe -m pytest in Git Bash
```

The parity tests need Node (they skip without it); `tests/test_gate.py` also needs
`onnxruntime-node` under `data/cache/reliability/node`.

## Python and JavaScript stay in step

The tracker, both word followers, the highlight and the stop detector exist in Python (evaluation,
server) and JavaScript (phone). Change both, and keep `tests/test_web_parity.py`,
`tests/test_stream_parity.py`, `tests/test_follower_parity.py` and `tests/test_gate.py` passing:
they require the same moves step for step. A new `TrackerConfig` field also goes into
`bench._LATE_FIELDS` with its neutral value, or the bench's tracker cache key changes and every
cached run is recomputed. After changing `data/duas/` or the popularity counts, rerun
`scripts/export_web.py`.

## Judging a change

A fix for one case is judged on the whole scenario bench, not on the case (docs/results/bench.md;
Hasan asked for every scenario to be tested, not one problem at a time).

1. Tune on dev: `python scripts/bench.py score --name X --display stream --split dev --same-text 8
   [--sc k=v] [--tracker k=v] --compare BASE`. `--display stream` is the page; the default
   `follower` is the rule-based one. Score a fresh baseline with today's defaults first
   (`base_dev_sp`/`base_test_sp` predate the next-line rule). The default delays (Whisper every 1 s,
   1.2 s late) flatter a phone: Hasan's runs Whisper every 2 s, 1.8 s late, so score the phone with
   `DUA_BENCH_ASR_EVERY=2 ... --delay 1.8 --follow-delay 0.1` (phone_engine.md; `full_ph_G1_dev`).
2. `python scripts/bench.py guard --name X --compare BASE`: every pre-registered bar, PASS/FAIL,
   per lane, with Hasan's sessions per recording. `scripts/jump_diag.py` sorts what moves remain.
3. Report on test (`--split test`). The test voices were looked at in the jump work, so say so.
4. Confirm on the real page: `scripts/bench_page.py --variants a:QUERY b:QUERY --jobs 2` (headless
   Chrome at phone speed), or `node scripts/page_replay.mjs IN.wav OUT.wav "QUERY" --asr-ms 1200`.
5. Write it up in docs/results/ with the numbers and their source; for a big change, write the
   acceptance bars down before the run (as in `data/cache/jumps/prereg.md`).

`bench.py asr` is a GPU pass; `score` replays cached model outputs on the CPU.

## Phone sessions

When Hasan says he tested on his phone, the session is in `data/sessions/` (audio + log in one
.wav). Read it with `python scripts/session_report.py [ID] [--timeline]`; `session_stops.py` shows
what the page did after each stop; `page_replay.mjs` plays a session through the real page.

## Machine

- One CUDA job at a time. Other sessions often share the machine and the GPU.
- Watch the Windows commit limit (about 39-42 GB): the machine crashed once under full GPU and
  CPU load near it. Run one bench scorer at a time, and check memory before heavy jobs.
- These are junctions onto `D:\dua-data`: `data/testbed`, `data/harvest`, `data/testsets`,
  `data/untimed`, `data/fatemah`, `data/duasorg_timed`, `data/cache/ctc`, `data/cache/pcm`.

## Writing

- Recitation is never "singing", and never described with notes, melody or sheet music: say
  recite, recitation, "draws a word out".
- In user-facing text the tracker's numbers are probabilities, never "belief".
- Plain, concrete, measured, with numbers and their source; no marketing language or absolutes.
- Demos and pages show the product itself: no eyebrow/headline/subtitle cards, labelled legends,
  footnotes or taglines inside the visual. Context goes in the surrounding text.

## Git

- Other sessions work in the same tree. Never `git add -A`, `git stash` or `git checkout --`;
  stage files by name, and don't revert changes you didn't make. Re-read a file before editing
  it if another session may have touched it.
- Commit or push only when asked.
