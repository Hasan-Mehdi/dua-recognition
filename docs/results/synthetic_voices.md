# Synthetic ordinary voices for the phone model (2026-09-30)

The phone model learned from professional reciters, and it does worst on the people who
actually use the app:

- RetaSy's crowd-sourced volunteers: 30.5% CER;
- Hasan's own sessions: 40-50% against the server model.

Hasan's idea, from his work (Qwen-ASR fine-tuned on Qwen-TTS medical audio): synthesize the
missing speech.

## The data

- **TTS:** [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (k2-fsa, April 2026,
  Apache-2.0). It is a zero-shot TTS over 600+ languages, including Standard Arabic
  (`arb`), and clones a voice from a few seconds of audio. Qwen3-TTS, Hasan's choice at
  work, has no Arabic.
- **Voices:** the 339 RetaSy volunteers already in the training set (`crowd.jsonl`), one
  reference clip each (their longest between 2 and 8 s). None of them has an annotated
  test clip, so `voice_eval.py`'s speakers stay unseen.
- **Texts:** spans of 4-16 consecutive words from the vowelled text of all 505 du'as, du'a
  chosen uniformly. Tawassul and Ashura are left out, as they are for the real windows.
  Speed factors run 0.75-1.1: an unhurried reader.
- **Amount:** 12,000 clips, 29.7 h, generated in 2 h 47 min on the RTX 5080. Batches of 4
  were fastest: RTF about 0.1.
- **Round trip:** the fine-tuned turbo transcribed every clip back. Median CER against the
  text 4.8% (quartiles 0 / 4.8 / 15.2%). The CER is flat across speed factors. Kept: CER
  at most 30% and 1-12 s long, which leaves 8,080 clips (15.3 h, 312 voices, 502 du'as,
  half women).

`scripts/synth_voices.py` generates, checks and builds the set, which lands in
`data/cache/finetune/synth.jsonl`. It runs in its own venv (`omnivoice` pins
transformers >= 5.3 and pulls in gradio).

## The models

The current phone recipe, unchanged (`docs/development.md`): the Tarteel whisper-base
checkpoint, the v4 windows, 3 epochs, room 0.5, speed 0.5, VTLP 0.5, SpecAugment.

| model | training data |
|---|---|
| whisper-base-ctrl-v5 | the v4 recipe retrained now, with the same code and environment |
| whisper-base-syn-v5 | the same + the 8,080 synthetic clips (`--synth 0.15`: 10.5% of the set) |
| whisper-base-aug-v4 | the current phone model |

Criteria, written before any result (`data/cache/synth/prereg.md`), against the control:
RetaSy CER at least 1 point lower; Quran-Lab's phone recordings no more than 0.5 points
worse; Hasan's sessions lower; DuaPlayer line accuracy no more than 0.5 points worse.

## Results

| | validation CER | RetaSy crowd | Quran-Lab all | Quran-Lab phone recordings | Hasan's sessions | DuaPlayer train line (1 s) |
|---|---:|---:|---:|---:|---:|---:|
| whisper-base-aug-v4 (current) | — | 29.9% | 12.1% | 17.5% | 42.8% | 81.8% |
| whisper-base-ctrl-v5 (control) | 14.2% | 32.3% | 11.9% | 17.9% | 42.6% | 81.8% |
| **whisper-base-syn-v5 (+ synthetic)** | **13.5%** | **24.7%** | **10.9%** | **16.3%** | **38.3%** | **82.1%** |

How each column was measured:
- *RetaSy crowd*: the 409 correct clips of 70 volunteers, `voice_eval.py`.
- *Quran-Lab phone recordings*: real Tarteel-user phone recordings (`tlog_holdout`).
- *Hasan's sessions*: the 6 device sessions' windows, CER against the server model.
- *DuaPlayer train line*: tracker line accuracy, one update a second, 1 s late.

All four criteria pass against the control, and by a wide margin on ordinary voices:
- **RetaSy:** 7.6 points (a quarter of the errors), and 5.2 points below the current phone
  model.
- **Professional recitation:** no worse (DuaPlayer, and Quran-Lab's professional
  `everyayah_heldout`: 7.6% against 9.3%).
- **Hasan's own voice:** 4.3 points better.

**Is it the voices, or the texts?** Some RetaSy verses also appear inside du'as, and the
synthetic clips read du'a text. The RetaSy clips were therefore split by whether any
synthetic clip shares a 4-word run with them:

| RetaSy clips | control | + synthetic |
|---|---:|---:|
| never read by a synthetic clip (235) | 37.0% | **30.6%** |
| sharing a 4-word run with one (174) | 24.7% | **17.6%** |

The gain is about as large on text no synthetic clip ever said, so the model learned to
hear ordinary voices, not to recall the texts.

**How it compares with the literature.** Published gains from mixing TTS speech into ASR
training are usually a few percent relative:
- up to 14% relative on accented speech ([arXiv 2409.11107](https://arxiv.org/abs/2409.11107));
- 8% on code-switching with CosyVoice2 ([arXiv 2601.00935](https://arxiv.org/pdf/2601.00935)).

This one is 24% relative on RetaSy at a 10% synthetic share. The likely reasons: cloning
the target population's own voices (crowd volunteers, mostly non-native readers), and a
training set that until now held only a few dozen voices, nearly all professional.

## Next

- The phone model is syn-v5 cut to an 8 s context, or fine-tuned at 8 s (`phone_speed.md`).
  Which one, and whether it beats the current default, is below.
- More synthetic data is cheap: 3 hours of GPU per 30 hours of audio. Worth trying next:
  - a higher share;
  - voices beyond RetaSy's 339;
  - children's voices, for the kids' mode (OmniVoice's voice design takes age and gender).

## The phone model: synthetic voices at an 8 s context

The 8 s models, scored on the clips that fit in 8 s, each as one live window
(`voice_eval.py --max-seconds 8`):

| | RetaSy (388 clips) | Quran-Lab (239) | Quran-Lab phone recordings |
|---|---:|---:|---:|
| whisper-base-aug-v4-ctx8 (default since 03:00) | 30.0% | 15.6% | 20.7% |
| whisper-base-ctrl-v5 | 32.4% | 15.9% | 20.1% |
| whisper-base-syn-v5 (30 s) | 24.5% | 13.9% | 18.7% |
| syn-v5 cut to 8 s | 25.6% | 14.6% | 19.4% |
| **syn-v5 fine-tuned 1 epoch at 8 s (`-ctx8ft`)** | **24.1%** | **14.2%** | **18.8%** |

The one-epoch fine-tune at 8 s wins back what the cut lost. On DuaPlayer train it follows as
well as the default (line 83.7% for both; wrong du'a 0.3 against 0.4%).

**The one test look** (criteria in `data/cache/synth/prereg.md`): DuaPlayer test, an update
each second, 1 s late, the new tracker settings.

| | line | ±1 | refrain | wrong du'a |
|---|---:|---:|---:|---:|
| whisper-base-aug-v4-ctx8 | 84.9% | 96.4% | 88.5% | 0.3% |
| **whisper-base-syn-v5-ctx8ft** | **85.1%** | 96.4% | **88.9%** | 0.4% |

Passed. `web/app.js` now loads `whisper-base-syn-v5-ctx8ft`, 100 MB of int8 ONNX like the
model before it, at 580 ms per update in desktop Chrome. At the evaluation's usual replay
(no display lead) it scores 85.4% line and 88.6% refrain lines, and names the du'a within
10 s 93% of the time (`test_whisper-base-syn-v5-ctx8ft.md`). On Hasan's sessions it gets
38.8% CER against the previous model's 42.8%, with the page frozen 3.7% of the time instead
of 7.9%.

## More of it didn't help (v6)

A second round, 12,000 more clips, 22.8 h in 108 min:
- **Voices:** half cloned from the same volunteers reading new spans, half *designed* from
  attributes (OmniVoice's voice design: gender, age including children, pitch, and an accent
  on half of them, Indian the most common).
- **Round trip:** the designed voices came out cleaner (median CER 2.4% against 4.5%; 94%
  kept against 84%), children's included (894 clips, 2.5%).
- **The set:** 17,682 clips, 30.2 h, `synth_v6.jsonl`.
- **Training:** the same recipe at a 20% synthetic share, then the same epoch at 8 s.
- **Labels:** the round made visible a label bug. `build_finetune_set.whisper_style` dropped
  Persian and Urdu letter forms, so یا became ا in a label. That affected about 720 words in
  166 texts, and it is fixed for new labels.

The criteria were written before the run (`data/cache/synth/prereg.md`), against the current
phone model. The 8 s models were scored on clips that fit in 8 s:

| | RetaSy | Quran-Lab | Quran-Lab phone | Hasan's sessions | validation |
|---|---:|---:|---:|---:|---:|
| **syn-v5-ctx8ft (the phone model)** | **24.1%** | **14.2%** | **18.8%** | **38.8%** | 13.5% |
| syn-v6-ctx8ft | 24.8% | 14.8% | 19.3% | 39.2% | 12.9% |

The 30 s models were scored on the full clips:

| | RetaSy | Quran-Lab | Quran-Lab phone |
|---|---:|---:|---:|
| syn-v5 | 24.7% | 10.9% | 16.3% |
| syn-v6 | 24.4% | 11.2% | 16.7% |

Not adopted. Twice the synthetic speech, and voices beyond RetaSy's, did no better on ordinary
voices; only the professional validation set improved. The first 15 hours bought what
synthetic speech can give this model at this share. Directions left untried: a different
TTS, noisier channels, or real ordinary voices.

The children's clips are still worth keeping for a kids' mode, but nothing here can measure
them: there is no recording of a child reciting to test against. A few children's debug
sessions would be the test set.

## Reproduce

```
<tts venv>/python scripts/synth_voices.py generate --n 12000
python scripts/synth_voices.py check
python scripts/synth_voices.py build
python scripts/finetune_whisper.py --base tarteel-ai/whisper-base-ar-quran --name whisper-base-syn-v5 \
    --data v4 --room 0.5 --epochs 3 --speed 0.5 --vtlp 0.5 --specaug --synth 0.15
python scripts/finetune_whisper.py --base models/whisper-base-syn-v5 --name whisper-base-syn-v5-ctx8ft \
    --data v4 --synth 0.15 --context 8 --epochs 1 --lr 1e-5 --room 0.5 --speed 0.5 --vtlp 0.5 --specaug
python scripts/voice_eval.py --device cuda models/whisper-base-syn-v5-ct2   # one model per process
python scripts/voice_eval.py --device cuda --max-seconds 8 models/whisper-base-syn-v5-ctx8ft-ct2
```

The TTS venv: `python -m venv C:\Users\hasan\.cache\tts`, a `.pth` file pointing at the
project's site-packages (for its torch), then `pip install --no-deps omnivoice pydub
tensorboardX webdataset braceexpand audioop-lts`. The checkpoint is k2-fsa/OmniVoice from
the Hugging Face cache.
