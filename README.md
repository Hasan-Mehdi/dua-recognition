# dua-recognition

Listen to someone recite a du'a, work out **which du'a it is**, and follow along
**line by line and word by word** in real time, with the Arabic, transliteration and
translation scrolling in sync.

On reciters it was never trained or tuned on, it shows **the right line 85% of the time**
(96% within one line), gets **89% of refrain lines** right (a line that recurs word for word, so text alone can't place it), and shows the wrong du'a
under 1% of the time. From a cold start mid-recitation it names the du'a within
3 s 89% of the time and within 10 s 95%. Those numbers are for a fine-tuned
whisper-base (74 M parameters) small enough to run **in a phone's browser, on-device**.
v0.1 of this repo managed 47% / 7% on the same test.

One phone can also follow for a whole room: **majlis mode** relays the position to any
number of phones or a projector through a QR code.

## Why this is harder than it looks

- **Refrains.** Du'as repeat themselves. In Dua Tawassul, 70 of 115 lines are
  refrains: *"yā wajīhan ʿinda-llāh, ishfaʿ lanā ʿinda-llāh"* comes back after every one of
  the 14 names. A transcript of "the current window" matches all 14 repetitions
  equally well, so matching text alone can't say which one you're on.
- **Shared openings.** Nearly every du'a opens with *bismillāh* and a *ṣalawāt*, and
  several share whole phrases (Dua Baha and Dua Tawassul both have a line starting
  *"allāhumma innī asʾaluka…"*). Committing too early means showing the wrong du'a.
- **Melodic recitation.** A single line can be drawn out for 15 seconds, so a 6-second
  window often holds half a line or less. Off-the-shelf small Whisper models barely
  transcribe it.
- **Real time.** The display has to move when the reciter moves, on hardware people
  actually have.

## How it works

```
mic ─▶ 6 s window every 1 s ─▶ Whisper ─▶ align against every du'a ─▶ HMM follower ─▶ du'a · line · word
                                  │         (semi-global edit        (position prior
                                  │          distance, vectorized)    resolves refrains)
                                  └─ fine-tuned whisper-base: CPU / in-browser
```

**1. Transcribe.** Each hop transcribes the last 6 s (faster-whisper / CTranslate2,
batched offline, batch-of-one live). Whisper's usual silence hallucinations
(*"ترجمة نانسي قنقر"*, *"شكرا"*) and decoding loops are filtered out.

**2. Align against the whole corpus at once** ([`align.py`](src/dua_recognition/align.py)).
All 91 texts become one normalized letter string (tashkeel stripped, alef/ya/ta-marbuta
variants folded, spaces dropped so Whisper's word splitting doesn't matter). For the
window's transcript *h*, a semi-global edit-distance DP gives, for **every** word in the
corpus, the cost of *h* ending exactly there. The row recurrence is vectorized with a
running-minimum trick for the horizontal moves, so one pass over the 34,419-word corpus
takes ~11 ms.

**3. Follow with an HMM** ([`tracker.py`](src/dua_recognition/tracker.py)). This is
*score following*, the technique used to turn sheet-music pages in time with a
performance, applied to recitation. The hidden state is the word being recited,
across all du'as.

- *Predict:* the reciter moved forward a few words since the last hop, may have
  gone back a few (reciters repeat lines), and with a small
  "teleport" probability jumped anywhere, so it recovers if someone skips ahead or
  switches du'a. Silent windows freeze the belief, because pauses between lines aren't
  movement.
- *Speed and tempo:* how far is "a few words" is learned from the human timings
  (a Poisson mixture over the train reciters' speeds), and the tracker keeps a weight
  per speed that adapts to the reciter in front of it: a slow, melodic reciter and a
  brisk one get different predictions. The first version assumed about twice the real
  speed and ran a line ahead on long lines; fixing that was worth +4 points
  ([speed_prior.md](docs/results/speed_prior.md)).
- *Correct:* weight each word by `exp(-κ · cost)` from step 2.
- *Prior:* 30% on the opening words of each du'a and 70% anywhere, uniform over
  du'as. A uniform prior over *words* would make long Kumayl ten times likelier than
  short Faraj before a word is heard.

A refrain matches all its repetitions equally well, and the predicted position is what
separates them. Identification comes for free: the posterior mass inside a du'a is the
probability that it's the one being recited, and nothing is shown until one du'a holds
70% of it. Until then, the likeliest du'as are offered as "Is it…?" chips.

**4. Make it run on a CPU** ([`finetune_whisper.py`](scripts/finetune_whisper.py)).
large-v3-turbo is accurate but needs ~3.3 s per window on CPU. Small models are fast
enough but barely understand recitation out of the box (stock whisper-small: 88% window
CER). So a small model is fine-tuned on train-reciter windows with **reference-snapped
labels**:
turbo transcribes each window, the human line timings bound where in the text it can
be, and the label becomes the *reference* text the transcript aligns to. The teacher
only decides where a window starts and ends in the text, so its spelling mistakes
never reach the labels. Audio is augmented with level changes and noise, to bridge
studio recordings and laptop mics; half the windows also get synthetic room reverb and
noise. The best starting point was [tarteel-ai/whisper-base-ar-quran](https://huggingface.co/tarteel-ai/whisper-base-ar-quran)
(already trained on Quran recitation): 12% window CER after fine-tuning, and it runs
in the browser via ONNX ([`web/`](web/)).

## Evaluation

Ground truth comes from [DuaPlayer](https://www.duaplayer.org), where reciters upload
recordings together with a hand-recorded start time for every line. That gives
**39 recordings (10 h) of 22 du'as and ziyarat by 12 reciters**, each second labelled
with the line being recited. The tracker searches all 91 texts DuaPlayer publishes, so
the 69 without recordings act as distractors.

Everything below is measured on **held-out reciters** (6 reciters, 16 recordings, 4 h): nobody in
the test set was used to tune the tracker or to train an ASR model. The split is by
reciter, not by recording ([`splits.py`](src/dua_recognition/splits.py)). The
evaluation replays each recording exactly as the live system hears it (6 s windows,
1 s hop) and compares the displayed line with the true line once per second.

| front end | line | ±1 line | refrain lines | wrong du'a shown | median lag | du'a found @3 s | @10 s |
|---|---|---|---|---|---|---|---|
| v0.1: per-window matcher (turbo) | 46.7% | 70.4% | 6.7% | 14.6% | 3.3 s | 83% | 91% |
| large-v3-turbo, stock | 81.7% | 96.4% | 82.3% | 0.8% | 1.4 s | 85% | 95% |
| large-v3-turbo + prompt biasing | 84.8% | 96.4% | 88.0% | 0.7% | 1.2 s | 87% | 95% |
| whisper-small, fine-tuned | 84.7% | 96.5% | 88.3% | 0.9% | 1.3 s | 88% | 95% |
| **whisper-base (Quran), fine-tuned** | **85.4%** | **96.4%** | **89.0%** | 1.0% | 1.2 s | **89%** | 95% |

Ziyarat Ashura is excluded from line accuracy (its human timings drift by several
lines) but kept for identification. A simulated phone-in-a-room (reverb, 15 dB SNR)
costs the stock-trained model ~6 points; training with room augmentation wins most of
that back ([comparison.md](docs/results/comparison.md)).

Full tables and per-du'a breakdowns are in [docs/results/](docs/results/): one file
per ASR front end, plus the ASR benchmark, the front-end comparison, and
[speed_prior.md](docs/results/speed_prior.md) (the error analysis behind the speed and
tempo model, and a negative result on scoring text straight from CTC posteriors).

## Run it

```bash
pip install -e ".[web,eval,dev]"
python scripts/fetch_duaplayer.py        # translations, timings, audio -> data/duaplayer/ (~210 MB)
pytest

python app/server.py                     # http://localhost:8000: recite, or replay a recording
python app/demo.py recitation.mp3        # terminal version (no argument = microphone)
```

Pick the ASR model with `DUA_ASR_MODEL` (default `large-v3-turbo`; it runs on GPU when
one is available). On a CPU-only machine, or for the in-browser version, use the
fine-tuned base model:

```bash
python scripts/transcribe_windows.py                      # teacher transcripts of every window
python scripts/build_finetune_set.py                      # reference-snapped labels (train reciters)
python scripts/finetune_whisper.py --base tarteel-ai/whisper-base-ar-quran --name whisper-base-quran-dua --room 0.5
DUA_ASR_MODEL=models/whisper-base-quran-dua-ct2 DUA_ASR_DEVICE=cpu python app/server.py

# on-device: no server, nothing leaves the phone
python scripts/export_web.py                              # web/corpus.json
python scripts/export_onnx.py models/whisper-base-quran-dua   # web/models/ (see its docstring)
python -m http.server -d web                              # any static host works
```

**Majlis mode:** with `app/server.py` running, tap *Aa → Show on other screens* while
following. Other phones or a projector scan the QR code (or open `?watch=CODE`) and
follow along. They run no speech recognition themselves.

Reproduce the numbers:

```bash
python scripts/transcribe_windows.py --model large-v3-turbo
python scripts/evaluate.py                                # test reciters
python scripts/evaluate.py --split train --tune           # tracker grid search (train only)
```

## Layout

```
src/dua_recognition/
  text.py        Arabic normalization (tashkeel, alef/ya/ta-marbuta, alef wasla, Persian letters)
  corpus.py      du'a texts; labelled recordings
  align.py       corpus index + vectorized semi-global alignment
  tracker.py     the HMM follower (speed prior + tempo adaptation)
  ctc_align.py   scoring text straight from CTC posteriors (experimental; see speed_prior.md)
  offline.py     forward-backward smoother: labels untimed audio (YouTube training data)
  asr.py         batched faster-whisper decoding, hallucination filters
  pipeline.py    StreamingRecognizer: audio chunks in, positions out
  splits.py      reciter-disjoint train/test split
  classify.py, match.py   v0.1 baselines (per-window matching), kept for comparison
scripts/         fetch_duaplayer, fetch_youtube, transcribe_windows, evaluate, compare, build_finetune_set,
                 finetune_whisper, finetune_ctc, dump_ctc, asr_benchmark, export_onnx, export_web, ...
app/             server.py (FastAPI + WebSocket; majlis-mode rooms); demo.py (CLI)
web/             the one front end: server or on-device engine (transformers.js + tracker.js port)
data/duas/       Arabic reference texts, one JSON per du'a
```

## Data and licensing

The Arabic texts in `data/duas/` come from [DuaPlayer](https://www.duaplayer.org), a
non-profit, and are the traditional texts of these supplications. Translations,
transliterations, line timings and audio belong to DuaPlayer and its reciters. They
aren't redistributed here: `fetch_duaplayer.py` downloads them into an ignored local
cache for evaluation. Please support DuaPlayer if you find this useful.

Code is MIT-licensed.
