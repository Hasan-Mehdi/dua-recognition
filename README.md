<h1 align="center">dua-recognition</h1>

<p align="center">
  <b>Recognize which du'a is being recited, and follow it line by line in real time.</b><br>
  On a server, or entirely on-device in a phone's browser.
</p>

<p align="center">
  <a href="https://github.com/Hasan-Mehdi/dua-recognition/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/Hasan-Mehdi/dua-recognition/actions/workflows/tests.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
</p>

<p align="center">
  <a href="docs/how-it-works.md">How it works</a> ·
  <a href="docs/evaluation.md">Evaluation</a> ·
  <a href="docs/development.md">Development</a> ·
  <a href="docs/results/">Research notes</a>
</p>

<p align="center">
  <img src="docs/demo.gif" width="720" alt="A phone following Dua Tawassul by a reciter the system never trained on, and a projector following along in majlis mode">
</p>
<p align="center"><i>A phone following Dua Tawassul, recited by a reciter held out of training, with a projector following along in majlis mode.</i></p>

It listens to a recitation, works out which of 505 du'as and ziyarat is being recited, and
shows the Arabic, transliteration and translation, highlighting the line and word being
recited as it goes.

## Highlights

- **Identifies the du'a** from anywhere in a recitation, among 505 du'as and ziyarat,
  usually within 10 seconds.
- **Follows line by line and word by word**, with the Arabic, transliteration and English
  kept in view.
- **Built for how du'as are recited:** refrains repeated word for word, passages shared
  between texts, long melodic notes, and pauses.
- **Runs on-device:** a fine-tuned Whisper model runs in the phone's browser, and no audio
  leaves the phone. A GPU server can run large-v3-turbo instead.
- **Majlis mode:** one phone drives any number of phones or a projector through a QR code.
- **Comfortable to read:** five Arabic typefaces, adjustable text size, word or line
  highlighting, favourites, and suggestions for the day and night.

## Results

Measured on reciters held out of all training and tuning (16 recordings, 4 hours),
replaying each recording exactly as the live system hears it:

| model | runs on | correct line | within one line | refrain lines | wrong du'a shown |
|---|---|---:|---:|---:|---:|
| **whisper-base, fine-tuned** | phone browser | **84.8%** | **96.2%** | **87.8%** | 0.5% |
| large-v3-turbo, fine-tuned | server | 84.0% | 96.3% | 88.7% | 0.4% |
| large-v3-turbo, stock | server | 81.5% | 96.1% | 82.6% | 0.4% |
| per-window text matching (baseline) | server | 42.7% | 71.4% | 5.8% | 12.9% |

Started mid-recitation, the phone model names the du'a within 10 seconds 92% of the time.
On everyday, non-professional voices its character error rate is 29.9% (15.4% for
fine-tuned large-v3-turbo). Methodology, per-du'a tables and every experiment are in
[docs/evaluation.md](docs/evaluation.md).

## Quick start

```bash
git clone https://github.com/Hasan-Mehdi/dua-recognition.git
cd dua-recognition
pip install -e ".[web]"
python scripts/fetch_duaplayer.py    # translations and sample recordings (~210 MB, kept locally)
python app/server.py                 # open http://localhost:8000
```

Recite, or play one of the sample recordings. The server uses Whisper large-v3-turbo
(downloaded on first run) and a GPU when one is available. `python app/demo.py
recitation.mp3` follows a file, or the microphone, in the terminal.

To run with no server at all, fine-tune and export the small model:
[docs/development.md](docs/development.md#the-phone-model).

**Majlis mode:** while following, tap *Aa → Show on other screens*. Other phones or a
projector scan the QR code and follow along without running any speech recognition.

## How it works

```
mic ─▶ 6 s window every 1 s ─▶ Whisper ─▶ align against every du'a ─▶ HMM follower ─▶ du'a · line · word
```

1. **Transcribe** the last 6 seconds of audio every second with Whisper.
2. **Align** the transcript against all 505 texts at once: a bit-parallel edit-distance
   pass scores every word in the 136,503-word corpus in about 3 ms.
3. **Follow** with a hidden Markov model, as score-following systems do for sheet music.
   It predicts where the reciter has moved to, which separates repetitions of a refrain
   that text alone can't tell apart. It adapts to the reciter's tempo and waits through
   pauses.

<p align="center">
  <img src="docs/explainer.gif" width="640" alt="Animated explainer: evidence from one window matches all 14 repetitions of a refrain; multiplying by the tracker's prediction leaves only the right one">
</p>

The full walk-through, including how a 74M-parameter model was fine-tuned to follow
melodic recitation, is in [docs/how-it-works.md](docs/how-it-works.md).

## Documentation

| | |
|---|---|
| [How it works](docs/how-it-works.md) | the problem, the alignment, the tracker and the phone model |
| [Evaluation](docs/evaluation.md) | methodology, full results, and an index of the research notes |
| [Development](docs/development.md) | setup, configuration, training and export, debug sessions, code map |
| [Research notes](docs/results/) | one write-up per experiment, including negative results |

## Project structure

```
src/dua_recognition/   core library: text normalization, corpus alignment, the HMM tracker, streaming pipeline
app/                   FastAPI server (live streaming, majlis rooms) and a terminal demo
web/                   the app: server or on-device engine (Transformers.js, ONNX Runtime Web)
scripts/               data collection, training, export and evaluation
data/duas/             505 reference texts, one JSON file per du'a
docs/                  walk-through, evaluation and research notes
tests/                 pytest suite
```

## Contributing

Issues and pull requests are welcome. Reports of du'as or recitation styles that the app
follows poorly are especially useful; a debug session recorded by the app
([how](docs/development.md#debug-sessions)) shows exactly what it heard and displayed.
Run the tests with `pip install -e ".[dev]"` and `pytest`.

## Acknowledgements

This work builds on the generosity of others:

- **[DuaPlayer](https://www.duaplayer.org)**, a non-profit, and its reciters, whose
  recordings with hand-marked line timings make evaluation possible, along with the texts
  and translations; **[duas.org](https://www.duas.org)** and **[duas.pro](https://duas.pro)**
  ([open API](https://github.com/duas-pro/shia-duas-api)) for further texts, translations
  and recordings.
- [Whisper](https://github.com/openai/whisper), run through
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and
  [CTranslate2](https://github.com/OpenNMT/CTranslate2) on the server and through
  [Transformers.js](https://github.com/huggingface/transformers.js) and
  [ONNX Runtime Web](https://onnxruntime.ai) in the browser.
- [tarteel-ai/whisper-base-ar-quran](https://huggingface.co/tarteel-ai/whisper-base-ar-quran),
  the starting point of the phone model, and
  [wav2vec2-large-xlsr-53-arabic-quran](https://huggingface.co/rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final),
  the base of the word aligner.
- [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection.
- [RetaSy's Quranic audio dataset](https://huggingface.co/datasets/RetaSy/quranic_audio_dataset)
  for evaluating everyday voices.
- [Manim Community](https://www.manim.community/) for the explainer animation, and the
  Amiri, Aref Ruqaa, Scheherazade New, Noto and EB Garamond typefaces.

## Data and license

The Arabic texts in `data/duas/` are the traditional texts of these supplications, as
published by DuaPlayer, duas.pro and duas.org; each file names its source. Translations,
transliterations, line timings and audio belong to those sites and their reciters. They
aren't redistributed here: the `fetch_*.py` scripts download them into an ignored local
cache for evaluation and training. Please support them if you find this useful.

The code is released under the [MIT License](LICENSE).
