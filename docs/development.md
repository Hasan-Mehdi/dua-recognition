# Development

## Setup

```bash
pip install -e ".[web,eval,dev]"
python scripts/fetch_duaplayer.py        # translations, timings, audio -> data/duaplayer/ (~210 MB)
python scripts/fetch_duasorg.py download && python scripts/fetch_duasorg.py lines   # translations for the duas.org texts
python scripts/fetch_duaspro.py --lines                                            # and the duas.pro texts
pytest
```

The fetched data lives in ignored local caches (see [Data](../README.md#license)).
The tests need none of it. The parity tests ([below](#python-and-javascript-in-step)) need
Node and skip without it; the four browser-gate tests also need `onnxruntime-node` installed
under `data/cache/reliability/node`.

Optional extras: `mic` (terminal demo from the microphone), `train` (fine-tuning),
`data` (yt-dlp, for the YouTube harvesting scripts).

## Running

```bash
python app/server.py                     # http://localhost:8000: recite, or replay a recording
python app/demo.py recitation.mp3        # terminal version (no argument = microphone)
```

| environment variable | default | |
|---|---|---|
| `DUA_ENGINE` | `device` once `web/models/` has the [phone models](#on-device-no-server-nothing-leaves-the-phone), else `server` | where the speech models run. `device`: in the browser; the server only serves files, rooms and debug sessions. `server`: on the server, which the page streams its audio to ([server_engine.md](results/server_engine.md)). Either way the page runs the tracker, the stream decoder and everything else |
| `DUA_ASR_MODEL` | `models/whisper-turbo-srv2-ct2` (large-v3-turbo fine-tuned in full at 8 s) if present and there's a GPU, else `models/whisper-base-syn-v5-ctx8ft-ct2` (the page's Whisper) if present, else `large-v3-turbo` | the server engine's Whisper: any faster-whisper model name or CTranslate2 directory ([server_profile.md](results/server_profile.md)) |
| `DUA_ASR_DEVICE` | GPU if available | `cpu` or `cuda` |
| `DUA_CTC_MODEL` | `models/ctc-student-base-v6` | the server engine's CTC model: a student folder, a `server_ctc.py` model, or any Hugging Face CTC model, e.g. `models/wav2vec2-quran-dua-voices` |
| `DUA_CTC_WINDOW` | `3.0` | seconds of audio per CTC window in the server engine (the student was trained on 3 s; the phone sends 2 s) |
| `DUA_SERVER_PROFILE` | `app/server.py` `SERVER_PROFILE` | a JSON file with the server engine's settings for the page: `ctcHop`, stream decoder `sc` and tracker `tc` by their JavaScript names (`{}` = the phone's) |
| `DUA_CTC_DEVICE` | GPU if available | `cpu` or `cuda` for a student (on this PC one 2 s window takes 21 ms on the CPU, 29 ms on the GPU) |
| `DUA_SESSIONS` | `data/sessions` | where uploaded debug sessions go (replays: somewhere else) |

URL options for the web app:

| option | effect |
|---|---|
| `?watch=CODE` | follow another phone's room (majlis mode) |
| `?majlis` | start in majlis mode |
| `?model=NAME` | on-device model under `web/models/` (default `whisper-base-syn-v5-ctx8ft`: synthetic voices, 8 s context; `whisper-base-aug-v4` is the previous one) |
| `?webgpu` | run the on-device model on WebGPU |
| `?words=off` | no word follower: Whisper's lead and the gliding highlight place the word, as before 2026-10-01 |
| `?follower=rules` | the rule-based follower (`follower.js`) instead of the stream decoder (`stream-follower.js`, the default since 2026-10-03) |
| `?sc=k:v,...`, `?tc=k:v,...` | stream-decoder and tracker settings by their JavaScript names, lists as `a;b;c` (`?sc=nextMargin:0` turns off the next-line rule); the session log records them |
| `?ctc=NAME` | on-device: the follower's CTC model under `web/models/` (default `ctc-student-base-v6-w2`; a phone whose steps take over 400 ms switches itself to `ctc-student-tiny-v6-w2` from its next session, and back once that one runs under 120 ms) |
| `?ctchop=S`, `?anchorhop=S` | the follower steps at most every S s (0.1); Whisper runs every S s while it follows (2 on the device, 1 in the server engine) |
| `?chunk=N` | audio reaches the engine in chunks of N samples (800, 50 ms) |
| `?ctcthreads=N` | on-device: WASM threads for the follower's model (default: onnxruntime-web's choice) |
| `?kids=1` | kids mode (`?kids=0` leaves it) |
| `?gate=energy_assisted` | on-device speech gate policy (default `legacy`; see [browser_gate.md](results/browser_gate.md)) |
| `?log=0` | don't record a debug session |
| `?debug` | show the live transcript and latency |
| `?lead=0`, `?pauses=0`, `?glide=0` | turn off display lead, pause handling or the word glide (for comparisons) |

## Judging a change

Changes to the display, the tracker or the models are judged on the scenario bench
([evaluation.md](evaluation.md#the-scenario-bench), [bench.md](results/bench.md)): on the
whole grid, tuned on the dev voices, scored with `--same-text 8`, checked bar by bar with
`guard`, reported on the test voices, and then confirmed on the real page. The bench's
programs, model outputs and results live in `data/testbed` (a junction to
`D:\dua-data\testbed`).

```bash
python scripts/bench.py sources && python scripts/bench.py build     # once: sources, item programs, truth
python scripts/bench.py asr                                          # GPU, once per model: Whisper rows, quiet, CTC frames
python scripts/bench.py score --name base_dev --display stream --split dev --same-text 8   # today's defaults
python scripts/bench.py score --name x_dev --display stream --split dev --same-text 8 \
    --sc next_hold=0.5 --compare base_dev                            # a candidate
python scripts/bench.py guard --name x_dev --compare base_dev        # every pre-registered bar, PASS/FAIL
python scripts/jump_diag.py --split dev [--lane user]                # what the remaining jumps and early moves are
python scripts/bench_page.py --per-scenario 1 --split test --jobs 2 \
    --variants off:sc=nextMargin:0 on:                               # one item per scenario through the real page
```

- `--display stream` is what the page runs. The default, `follower`, is the rule-based
  follower (`--fw` sets its options); `tracker` and `oracle` (the follower anchored on the true
  word) are ablations. `--sc` sets StreamConfig fields, `--tracker` TrackerConfig fields
  (`k=v,...`, tuples as `a;b;c`). `--asr truth` puts each window's true words in place of
  Whisper's transcript (a perfect ear).
- Scenario names and `--lane` narrow the grid while working; the decision is on all of it.
  Results go to `data/testbed/results/NAME.json`. `base_dev_sp` and `base_test_sp` are the
  baselines of the jump work (before the next-line rule went on), so score a fresh baseline
  with today's defaults to compare against.
- `bench_page.py` plays the items through the page in headless Chrome in real time
  (`page_replay.mjs`), with Whisper held to at least 1.2 s an update and the CTC model to 150 ms
  a step (`--asr-ms`, `--ctc-ms`). With `--jobs` equal to the number of variants, an item's
  variants play at the same time and share the machine's load. The page takes the same
  settings by their JavaScript names (`?sc=nextMargin:0`, `?tc=...`) and its session log
  records what each variant ran with.
- A new `TrackerConfig` field must go into `bench._LATE_FIELDS` with its neutral value, or the
  tracker cache key changes and every cached run is recomputed.
- The older per-problem benchmarks (`follow_eval.py` with `--repeats` or `--jumps`,
  `repeat_eval.py`, `jump_eval.py`, `pause_eval.py`, `ooc_eval.py`) still run on the studio
  recordings.

### Python and JavaScript in step

The tracker, both word followers, the gliding highlight and the stop detector exist twice: in
Python for evaluation and the server, and in JavaScript for the phone. A change to one is made
to the other too, and the parity tests run the same inputs through both and require the same
result step for step:

| Python | JavaScript | test |
|---|---|---|
| `tracker.py` | `tracker.js` | `tests/test_web_parity.py` |
| `display.py` (`Highlight`), `asr.LiveQuiet` | `display.js`, `gate.js` | `tests/test_web_parity.py` |
| `stream_follower.py` (and its numba pass against the numpy reference) | `stream-follower.js` | `tests/test_stream_parity.py` |
| `follower.py` | `follower.js` | `tests/test_follower_parity.py` |
| `asr.py` hallucination filter | `gate.js` | `tests/test_gate.py` |

The parity tests skip when Node isn't installed, so run them where it is. The page reads the
corpus from `web/corpus.json` (`scripts/export_web.py`, not committed): regenerate it after
changing `data/duas/` or `data/dua_popularity.json`.

## The phone model

On a CPU-only machine, or for the in-browser version, use a fine-tuned whisper-base
([how it works](how-it-works.md#4-run-on-a-cpu-or-a-phone)):

```bash
python scripts/transcribe_windows.py                      # teacher transcripts of every window
python scripts/build_finetune_set.py --youtube --untimed --version v4   # reference-snapped labels (train voices)
<tts venv>/python scripts/synth_voices.py generate     # synthetic ordinary voices (OmniVoice; see its docstring)
python scripts/synth_voices.py check && python scripts/synth_voices.py build
python scripts/finetune_whisper.py --base tarteel-ai/whisper-base-ar-quran --name whisper-base-syn-v5 \
    --data v4 --room 0.5 --epochs 3 --speed 0.5 --vtlp 0.5 --specaug --synth 0.15
python scripts/finetune_whisper.py --base models/whisper-base-syn-v5 --name whisper-base-syn-v5-ctx8ft \
    --data v4 --synth 0.15 --context 8 --epochs 1 --lr 1e-5 --room 0.5 --speed 0.5 --vtlp 0.5 --specaug
DUA_ASR_MODEL=models/whisper-base-syn-v5-ctx8ft-ct2 DUA_ASR_DEVICE=cpu DUA_ENGINE=server python app/server.py
```

Synthetic ordinary voices take a quarter of the errors off everyday voices
([synthetic_voices.md](results/synthetic_voices.md)). Whisper pads every input to 30 s, while
the live follower only sends 6 s windows, so the phone model runs at an 8 s context: four
times less work per update ([phone_speed.md](results/phone_speed.md)).
`scripts/shorten_context.py MODEL --seconds 8` cuts any checkpoint to 8 s without training,
a little worse than the one-epoch fine-tune above.

### The word follower's model

The phone also runs a small CTC model that hears letters every 20 ms; the stream decoder places
the word from it ([phone_follower.md](results/phone_follower.md),
[bench.md](results/bench.md)). It learns from the server's
wav2vec2, itself given one more epoch on ordinary voices, on the same windows plus the
synthetic, crowd-sourced and Common Voice voices:

```bash
python scripts/finetune_ctc.py --base models/wav2vec2-quran-dua --name wav2vec2-quran-dua-voices \
    --data v4 --extra synth_v5,synth_v6,crowd --phone 0.3 --room 0.3 --epochs 1 --lr 1e-5 --warmup 300
python scripts/build_cv_set.py && python scripts/build_cv_set.py --split test --name cv_ar_test   # Common Voice
python scripts/ctc_teacher.py val_v4 crowd synth_v5 synth_v6 cv_ar cv_ar_test train_v4 \
    --model models/wav2vec2-quran-dua-voices --out data/cache/ctc_student_voices   # frames + letter timings
python scripts/train_ctc_student.py --name ctc-student-base-v6 --teacher ctc_student_voices \
    --synth synth_v5,synth_v6,cv_ar,cv_ar_test --phone 0.5 --epochs 32
python scripts/train_ctc_student.py --base tarteel-ai/whisper-tiny-ar-quran --name ctc-student-tiny-v6 \
    --teacher ctc_student_voices --synth synth_v5,synth_v6,cv_ar,cv_ar_test --phone 0.5 --lr 3e-4 --epochs 32  # slow phones
python scripts/export_ctc_student.py models/ctc-student-base-v6 --window 2 --name ctc-student-base-v6-w2   # web/models/
python scripts/export_ctc_student.py models/ctc-student-tiny-v6 --window 2 --name ctc-student-tiny-v6-w2
```

The page runs them on 2 s windows: a third less compute per step than the 3 s they were trained
on, for 0.6 points at equal delay, and on a phone the compute is most of the delay
([phone_latency.md](results/phone_latency.md)). Repeats have their own benchmark,
`scripts/repeat_eval.py` (`follow_eval.py --repeats`).

### On-device: no server, nothing leaves the phone

```bash
python scripts/export_web.py                                     # web/corpus.json
python scripts/export_onnx.py models/whisper-base-syn-v5-ctx8ft   # web/models/ (see its docstring)
python scripts/export_ctc_student.py models/ctc-student-base-v6 --window 2 --name ctc-student-base-v6-w2   # web/models/ (the word model)
python scripts/export_ctc_student.py models/ctc-student-tiny-v6 --window 2 --name ctc-student-tiny-v6-w2   # its slow-phone fallback
python -m http.server -d web                                     # any static host works
```

## Majlis mode

With `app/server.py` running, tap *Aa → Show on other screens* while following. Other
phones or a projector scan the QR code (or open `?watch=CODE`) and follow along. They run
no speech recognition themselves.

## Debug sessions

On for now (`?log=0` turns them off): every listening session keeps the 16 kHz audio the
recognizer heard and a log of what it made of it (each window's transcript and tracker
state, each line and word shown, taps, scrolls, the phone going to sleep) as one `.wav`,
with the log in a RIFF chunk that players ignore. Served by `app/server.py`, the page
uploads them to `data/sessions/` when a session ends, in either engine (with the phone models
exported it serves the on-device app, with no speech model on the server).
From a static host they stay on the phone until *send* on the home screen (share sheet or
download). While following, tapping the line actually being recited moves the display
there and logs "I'm here".

```bash
python scripts/session_report.py               # newest session: device, mic, speed, what was shown when
python scripts/session_report.py ID --timeline # every update, line move and tap
python scripts/session_report.py ID --score    # vs fine-tuned turbo + offline smoother (GPU)
python scripts/session_report.py ID --score --set kappa=0.2   # what a tracker change would have shown
python scripts/session_stops.py [ID] -v        # what the page showed after each stop, word by word
node scripts/page_replay.mjs SESSION.wav out.wav "" --asr-ms 1100   # the session's audio through the real page, headless
```

The `--score` reference agreed with the human line timings on 88% of seconds (100% within
a line) on a held-out Tawassul clip, and replaying a session's own transcripts through the
Python tracker reproduces the phone's display exactly.

## Code map

```
src/dua_recognition/
  text.py        Arabic normalization (tashkeel, alef/ya/ta-marbuta, alef wasla, Persian letters)
  corpus.py      du'a texts (with how often each is recited, data/dua_popularity.json); labelled recordings
  align.py       corpus index + semi-global alignment (bit-parallel Myers, numba)
  tracker.py     the HMM tracker (speed prior + tempo adaptation, pauses, du'a prior, line jumps,
                 shared passages, "not a known du'a")
  display.py     the gliding word highlight between updates (?words=off)
  stream_follower.py  the stream decoder: CTC forward pass over the whole du'a, reading moves as
                 transitions (the page's word follower)
  follower.py    the rule-based word follower (?follower=rules on the page)
  ctc_student.py the small CTC model (a Whisper encoder with a letter head)
  ctc.py         wav2vec2 CTC posteriors folded onto the corpus alphabet
  ctc_adapter.py sub-word (BPE) CTC models for the follower, pieces kept (small_ctc.md)
  ctc_align.py   scoring text straight from CTC posteriors (experimental; see speed_prior.md)
  offline.py     forward-backward smoother: labels untimed audio (YouTube training data)
  asr.py         batched faster-whisper decoding, hallucination filters, silence timing
  pipeline.py    StreamingRecognizer: audio chunks in, positions out
  splits.py      reciter-disjoint train/test split; held-out test venues
  labels.py      line labels as subtitles for review; agreement between label sets
  provenance.py  where line and word labels came from (automatic or human-reviewed)
  translit.py    one transliteration style, made from the vowelled Arabic
  classify.py, match.py   per-window matching baselines, kept for comparison
scripts/
  data           fetch_duaplayer, fetch_duaspro, fetch_duasorg, fetch_youtube, fetch_testset, find_captioned, speaker_check,
                 mafatih_corpus (the 17 texts added from Mafatih), dua_popularity (the du'a prior's counts)
  harvest        harvest, harvest_queries, harvest_aparat, harvest_web, harvest_sc, harvest_archive (downloads),
                 harvest_label (frames, align, captions, export), harvest_voices, harvest_report,
                 harvest_ctc_cache, build_clip_cache, build_voice_sets (data_harvest.md)
  training       transcribe_windows, align_offline, build_finetune_set, synth_voices (synthetic voices),
                 finetune_whisper, shorten_context (8 s context), export_onnx,
                 ctc_teacher + train_ctc_student + export_ctc_student (the phone's word model)
  bench          bench (the scenario grid: sources, build, asr, score, guard), bench_page (the real page),
                 jump_diag (what the display's jumps are)
  evaluation     evaluate, word_truth, word_eval, follow_eval, pause_eval, repeat_eval, jump_eval, ooc_eval,
                 voice_eval, asr_benchmark, ...
  reliability    reliability_manifest, testset_split, label_coverage, review_bundle, follow_latency,
                 gate_eval (+ gate_replay.mjs, gate_bench.mjs), small_ctc, tilawa_phonemes
  sessions       session_report (the phone's debug sessions, data/sessions/), session_replay (tracker
                 changes replayed over many sessions), session_stops (the display after each stop),
                 session_eval (word-level scores against forced-aligned truth, with the word follower),
                 page_replay.mjs (a recording through the real page in headless Chrome)
  noha research  noha_lid, noha_match, noha_finetune, ...
app/             server.py (FastAPI: the page, the server engine's models over /ws/ear, majlis-mode rooms,
                 debug sessions); demo.py (CLI)
web/             the app, its models in the browser or on the server (transformers.js + tracker.js port; ctc-worker.js, the
                 word model; stream-follower.js, the word follower, and follower.js, the rule-based one);
                 gate.js (the phone's speech gate), session-log.js (debug sessions), kids.js (kids mode),
                 dev/ (benchmark pages)
data/duas/       522 Arabic reference texts, one JSON per du'a
```
