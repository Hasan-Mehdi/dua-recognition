# dua-recognition

Listen to a recitation and figure out, in real time, **which du'a it is** and
**where in it the reciter currently is** — then you can follow along with synced
Arabic / transliteration / translation.

The interesting half is passage localization. Du'a Arabic leans on vocabulary
(names, set phrases) that general Arabic ASR transcribes unevenly, and a single
listening window only ever covers part of a line, so exact text matching doesn't
cut it.

## Approach

```
audio window ──▶ which du'a? ──▶ transcribe (Whisper) ──▶ fuzzy-match the line
              (classifier)                              (rapidfuzz over normalized text)
```

- **Identify the du'a** — text baseline today (`TextClassifier`): transcribe and
  pick the closest reference. A melody-aware audio classifier is the planned
  replacement (`AudioClassifier`, `scripts/train_classifier.py`).
- **Transcribe** — faster-whisper, Arabic. Fine-tuning on du'a-specific data is
  scaffolded in `scripts/finetune_whisper.py`; the base model is fine for the
  well-known openings.
- **Locate the line** — normalize Arabic down to bare letters (strip tashkeel,
  fold alef/ya/ta-marbuta), then `partial_ratio` against each segment so a
  fragment still lands on the right line.

## What works now vs. what's stubbed

Working and tested: the Arabic normalizer (`text.py`), the passage matcher
(`match.py`), the corpus loader, the mic sliding-window (`stream.py`), and the
recognizer that wires them together. The two model-training scripts and the
audio classifier are honest skeletons with the interfaces fixed and the
data-dependent parts marked `TODO`.

```
src/dua_recognition/   text, corpus, match, asr, classify, stream, pipeline
scripts/               fetch_audio, preprocess, finetune_whisper, train_classifier
app/demo.py            live mic demo / one-shot file recognition
data/duas/*.json       reference texts (one illustrative excerpt included)
tests/                 normalizer + matcher
```

## Run

```bash
pip install -e ".[dev]"
pytest                       # normalizer + matcher tests

pip install -e ".[demo]"
python app/demo.py clip.wav  # one-shot, or no arg to use the mic
```

Building the corpus: `scripts/fetch_audio.py sources.csv` to pull recitations,
then `scripts/preprocess.py` to resample/segment. Audio and trained models stay
out of git — see `data/README.md`.

## Status

Early. Plumbing runs end to end on the text path; the ASR fine-tune and audio
classifier are next. Reference texts shipped here are minimal placeholders —
replace them with verified sources before relying on the output.
