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
The tests need none of it; the four browser-gate tests run only when Node and
`onnxruntime-node` are installed under `data/cache/reliability/node`.

Optional extras: `mic` (terminal demo from the microphone), `train` (fine-tuning),
`data` (yt-dlp, for the YouTube harvesting scripts).

## Running

```bash
python app/server.py                     # http://localhost:8000: recite, or replay a recording
python app/demo.py recitation.mp3        # terminal version (no argument = microphone)
```

| environment variable | default | |
|---|---|---|
| `DUA_ASR_MODEL` | `large-v3-turbo` | any faster-whisper model name or CTranslate2 directory |
| `DUA_ASR_DEVICE` | GPU if available | `cpu` or `cuda` |
| `DUA_ENGINE` | `server` | `device`: the browser runs speech recognition; the server only serves files, rooms and debug sessions |
| `DUA_CTC_MODEL` | `models/wav2vec2-quran-dua` | the CTC model behind `?words=ctc` |

URL options for the web app:

| option | effect |
|---|---|
| `?watch=CODE` | follow another phone's room (majlis mode) |
| `?majlis` | start in majlis mode |
| `?model=NAME` | on-device model under `web/models/` (default `whisper-base-aug-v4`) |
| `?webgpu` | run the on-device model on WebGPU |
| `?words=ctc` | server only: place the word with the CTC word follower (experimental) |
| `?gate=energy_assisted` | on-device speech gate policy (default `legacy`; see [browser_gate.md](results/browser_gate.md)) |
| `?log=0` | don't record a debug session |
| `?debug` | show the live transcript and latency |
| `?lead=0`, `?pauses=0`, `?glide=0` | turn off display lead, pause handling or the word glide (for comparisons) |

## The phone model

On a CPU-only machine, or for the in-browser version, use a fine-tuned whisper-base
([how it works](how-it-works.md#4-run-on-a-cpu-or-a-phone)):

```bash
python scripts/transcribe_windows.py                      # teacher transcripts of every window
python scripts/build_finetune_set.py --youtube --untimed --version v4   # reference-snapped labels (train voices)
python scripts/finetune_whisper.py --base tarteel-ai/whisper-base-ar-quran --name whisper-base-aug-v4 \
    --data v4 --room 0.5 --epochs 3 --speed 0.5 --vtlp 0.5 --specaug
DUA_ASR_MODEL=models/whisper-base-aug-v4-ct2 DUA_ASR_DEVICE=cpu python app/server.py
```

### On-device: no server, nothing leaves the phone

```bash
python scripts/export_web.py                              # web/corpus.json
python scripts/export_onnx.py models/whisper-base-aug-v4   # web/models/ (see its docstring)
python -m http.server -d web                              # any static host works
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
uploads them to `data/sessions/` when a session ends; `DUA_ENGINE=device python
app/server.py` serves the on-device app the same way, with no speech model on the server.
From a static host they stay on the phone until *send* on the home screen (share sheet or
download). While following, tapping the line actually being recited moves the display
there and logs "I'm here".

```bash
python scripts/session_report.py               # newest session: device, mic, speed, what was shown when
python scripts/session_report.py ID --timeline # every update, line move and tap
python scripts/session_report.py ID --score    # vs fine-tuned turbo + offline smoother (GPU)
python scripts/session_report.py ID --score --set kappa=0.2   # what a tracker change would have shown
```

The `--score` reference agreed with the human line timings on 88% of seconds (100% within
a line) on a held-out Tawassul clip, and replaying a session's own transcripts through the
Python tracker reproduces the phone's display exactly.

## Code map

```
src/dua_recognition/
  text.py        Arabic normalization (tashkeel, alef/ya/ta-marbuta, alef wasla, Persian letters)
  corpus.py      du'a texts; labelled recordings
  align.py       corpus index + semi-global alignment (bit-parallel Myers, numba)
  tracker.py     the HMM follower (speed prior + tempo adaptation, pauses, "not a known du'a")
  display.py     the gliding word highlight between updates
  follower.py    word-by-word following from CTC frames between tracker updates (opt-in)
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
  data           fetch_duaplayer, fetch_duaspro, fetch_duasorg, fetch_youtube, fetch_testset, find_captioned, speaker_check
  training       transcribe_windows, align_offline, build_finetune_set, finetune_whisper, export_onnx
  evaluation     evaluate, word_truth, word_eval, follow_eval, pause_eval, ooc_eval, voice_eval, asr_benchmark, ...
  reliability    reliability_manifest, testset_split, label_coverage, review_bundle, follow_latency,
                 gate_eval (+ gate_replay.mjs, gate_bench.mjs), small_ctc
  sessions       session_report (the phone's debug sessions, data/sessions/)
  noha research  noha_lid, noha_match, noha_finetune, ...
app/             server.py (FastAPI + WebSocket; majlis-mode rooms); demo.py (CLI)
web/             the app: server or on-device engine (transformers.js + tracker.js port);
                 gate.js (the phone's speech gate), session-log.js (debug sessions), dev/ (benchmark pages)
data/duas/       505 Arabic reference texts, one JSON per du'a
```
