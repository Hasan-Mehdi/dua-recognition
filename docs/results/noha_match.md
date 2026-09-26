# Noha identification: can the lyrics decide the language?

Follow-up to [noha_lid.md](noha_lid.md), which found that acoustic language
ID tops out at ~75-79% after 10 s on sung nohas. This measures the whole
chain: transcribe, align against the lyrics of every noha, and run the du'a
HMM tracker unchanged. The question is how often, and how soon, it names the
right noha, and whether it needs to know the language to do so
([`scripts/noha_match.py`](../../scripts/noha_match.py)).

**Findings in brief**

- **The lyrics can pick the language.** Transcribing in the two likeliest
  languages and keeping whichever transcript fits some lyric best comes within
  2-3 points of knowing the language, and it shows a wrong noha less often.
- **Every stock model plateaus** at ~62-68% after 10 s and ~78-81% after 20 s.
  Whisper large-v3, Qwen3-ASR (trained on singing) and Meta's Omnilingual all
  land there. Vocal separation (Demucs) doesn't help, and neither does tracker
  tuning.
- **Fine-tuning a small Whisper on nohas breaks the plateau.** 5,900 windows,
  lyrics-verified labels, 25 minutes on one GPU. whisper-small then finds the
  noha in 36% / 68% / 80% of cold starts after 5 / 10 / 20 s (stock:
  26% / 59% / 76%). Its language ID goes from 69% to 85% after 10 s, so the
  model's own language choice is as good as knowing it. Fine-tuned whisper-base,
  today's browser size, gets 64% / 78% after 10 / 20 s (stock: 40% / 58%).
- **For a real library** (5-35k lyrics per language exist online), an n-gram
  first stage keeps the right noha in its top 20-50 more often than the tracker
  finds it, so the HMM cost stays fixed.

## Setup

- **59 test recordings** (Urdu 15, Arabic 15, Farsi 15, English 14) whose
  YouTube descriptions carry the lyrics. Urdu lyrics are mostly **Roman Urdu**
  (10 of 15), which is how most Urdu noha lyrics are published. Arabic and
  Farsi lyrics are in their own script, English in English.
- **269 lyrics in the corpus**: the 59, plus 210 lyrics-only distractors from
  other noha uploads. Re-uploads of the same noha are merged (5-gram overlap),
  so finding another copy of the right noha isn't scored as an error.
- **Replayed as the app would hear it:** a listener opens the app at 25/45/65/85%
  of the recording (236 cold starts), a 6 s window every 1 s. Each window is
  transcribed forced to each of the four languages.
- **Scored:** is the right noha shown after 3/5/10/20 s, and is a *wrong* noha
  shown (the bracketed numbers). Nothing is shown until one noha holds 70% of
  the tracker's belief. That's the du'a setting, left unchanged.

### A phonetic skeleton instead of a script

To match an Urdu-script transcript against Roman-Urdu lyrics (or a Devanagari
one against Urdu lyrics), both sides are reduced to a Latin consonant
skeleton. The letters Urdu and Farsi pronounce alike are merged (س ص ث → s,
ز ذ ض ظ → z, ت ط → t, ح ہ → h), short vowels are dropped (Arabic script
doesn't write them), and Roman digraphs are folded (kh → x, sh → s). So
*Hussain*, حسین and हुसैन all become `hsn`, and "Uski aawaz nay mara hai
mujhe" and "اس کی آواز نے مارا ہے مجھے" become the same string. The du'a
aligner and tracker then run unchanged on a 17-letter alphabet.

## Results

### Language strategy (whisper-small)

| strategy | @5 s | @10 s | @20 s |
|---|---|---|---|
| right language (oracle) | 29% (2%) | 62% (6%) | 78% (8%) |
| Whisper's own language ID | 24% (3%) | 55% (6%) | 70% (8%) |
| **top-2 LID languages, keep the transcript that fits some lyric best** | 26% (2%) | 59% (5%) | 76% (5%) |
| all 4 languages, best fit | 25% (0%) | 57% (6%) | 76% (4%) |
| a wrong language, forced | 10% (1%) | 30% (6%) | 49% (9%) |

**The lyrics can pick the language.** Transcribing in the top two languages
from Whisper's language ID, then keeping whichever transcript aligns best to
*some* lyric, comes within 2-3 points of knowing the language, and it shows a
wrong noha less often than the oracle does. A wrong language on its own is
far worse (49% at 20 s), so the lyrics can only choose among candidates;
they can't rescue a single wrong guess.

### Gating: what counts as evidence

The du'a pipeline skips windows where Silero VAD hears no speech, and drops
transcripts that Whisper marks no-speech or that loop. On sung nohas:

| gate (whisper-small, oracle) | @10 s | @20 s | wrong @20 s |
|---|---|---|---|
| Silero VAD (the du'a rule) | 51% | 67% | 7% |
| Whisper no-speech < 0.6 | 38% | 67% | 2% |
| none | 61% | 75% | 15% |
| **loop filter only** | **62%** | **78%** | **8%** |

VAD misses singing: it heard speech in about half the windows, and in 1 of
80 for one Farsi music video. Whisper's no-speech probability is high on
music. With no gate at all, one repetitive distractor ("Zainab Zainab
Zainab…", "Abbas haay Abbas") soaked up Whisper's looping output on music
and was shown for 28% of Farsi cold starts. The compression-ratio loop
filter alone fixes that.

### Front ends

Best strategy per model: oracle = language known; best-of = top-2 or all-4
languages, lyrics decide (no language needed).

| model | size | oracle @10 s | oracle @20 s | best-of @10 s | best-of @20 s | browser? |
|---|---|---|---|---|---|---|
| whisper-base | 74 M | 54% (6%) | 73% (7%) | 46% (4%) | 67% (4%) | yes (today's du'a model size) |
| whisper-small | 244 M | 62% (6%) | 78% (8%) | 59% (5%) | 76% (5%) | yes, slower |
| whisper-large-v3-turbo | 809 M | 67% (4%) | 81% (8%) | 62% (5%) | 78% (5%) | no |
| whisper-large-v3 | 1.55 B | 68% (4%) | 80% (8%) | 63% (3%) | 78% (5%) | no |
| **Qwen3-ASR-0.6B** | 0.6 B | 65% (6%) | 80% (7%) | 62% (8%) | 78% (7%) | WebGPU ports exist |
| Qwen3-ASR-1.7B | 1.7 B | 65% (7%) | 79% (7%) | 64% (6%) | 80% (6%) | no |
| Omnilingual CTC-300M (no language input) | 300 M | 47% (6%) | 64% (8%) | = | = | sherpa-onnx WASM |
| Omnilingual CTC-1B (no language input) | 1 B | 44% (3%) | 58% (6%) | = | = | no |
| whisper-small + Omnilingual-1B as a 3rd candidate | | | | 61% (3%) | 78% (6%) | no |
| **whisper-base, fine-tuned on nohas** | 74 M | 64% (7%) | 79% (6%) | 64% (7%)† | 78% (6%)† | yes |
| **whisper-small, fine-tuned on nohas** | 244 M | **69% (5%)** | **80% (8%)** | **68% (3%)** | **80% (6%)** | yes |

† its own language ID (one transcript per window); for base that beats
best-of-2 (59% / 71%).

Per language, at 20 s (oracle):

| model | Urdu | Arabic | Farsi | English |
|---|---|---|---|---|
| whisper-small | 88% | 87% | 68% | 70% |
| large-v3 | **90%** | 85% | 72% | 71% |
| Qwen3-ASR-0.6B (Urdu forced as Hindi) | 88% | 87% | 73% | 71% |
| Omnilingual CTC-1B | 63% | 87% | 72% | 7% |
| whisper-small, fine-tuned on nohas | 88% | 83% | **77%** | 71% |

- **Bigger Whisper barely helps.** large-v3 is 6× whisper-small and gains
  2-6 points. Model size isn't the bottleneck; sung audio is.
- **Qwen3-ASR-0.6B matches large-v3 at 0.6 B.** It is trained on singing.
  It has no Urdu, but forcing Hindi gives Devanagari that the skeleton reads
  like Urdu script: Urdu on par with large-v3. Its own language detection is
  unreliable on nohas (English and Chinese for many Urdu windows), but its
  auto mode still finds 60% / 74% at 10 / 20 s without us picking a language
  (1.7B: 62% / 78%; otherwise no better than 0.6B).
- **Omnilingual needs no language and is strong on Arabic and Farsi**, but
  weak on Urdu and fails on English singing. As a third candidate next to
  whisper-small it adds ~2 points overall (Farsi +8 at 10 s).

### Fine-tuning on nohas

The du'a model went from 47% to 85% line accuracy by fine-tuning a small
Whisper on reference-snapped labels. The noha version
([`scripts/noha_finetune.py`](../../scripts/noha_finetune.py)) has no line
timings, so the lyrics do the checking:

1. **Training audio:** the distractor uploads whose descriptions carry lyrics,
   **from channels that never appear in the test set**. That gave 141
   recordings: Urdu 49, Farsi 57, English 20, Arabic 15.
2. **Labels:** 6 s windows every 3 s, transcribed by large-v3-turbo in the
   recording's language. A window is kept only if its transcript aligns to the
   recording's own lyrics at ≤ 0.35 skeleton edits per letter, with at least 6
   skeleton letters (a one-word transcript fits any lyric). When the lyrics are
   in the language's own script, the label is the matching stretch of the
   lyrics, so the teacher's spelling mistakes never reach it ("در توافع هرمت
   فرصت تکرار بده" → "در طواف حرمت فرصت دیدار بده"). Roman-Urdu lyrics can
   verify an Urdu transcript but not spell it, so there the teacher's text is
   kept. Result: 5,880 training windows (Urdu 2,540, Farsi 1,586, Arabic 886,
   English 868).
3. **Training:** all weights, per-window language token, level/noise/room
   augmentation; 4 epochs of whisper-small in 25 minutes on an RTX 5080. The
   best validation loss (held-out channels) fell from 1.59 to 0.70.

| whisper-small | @5 s | @10 s | @20 s |
|---|---|---|---|
| stock, oracle language | 29% (2%) | 62% (6%) | 78% (8%) |
| stock, best-of-2 | 26% (2%) | 59% (5%) | 76% (5%) |
| fine-tuned, oracle language | 39% (1%) | 69% (5%) | 80% (8%) |
| **fine-tuned, best-of-2** | **36% (1%)** | **68% (3%)** | **80% (6%)** |
| fine-tuned, its own language ID | 38% (1%) | 68% (5%) | 80% (8%) |
| fine-tuned, a wrong language forced | 18% (1%) | 49% (9%) | 65% (12%) |

- It now beats stock large-v3 and Qwen3-ASR at 5 and 10 s.
- The largest gain is Farsi, 48% → 60% at 10 s, the language stock models
  handled worst.
- The language token is fixed as a side effect. Its own LID now equals the
  oracle (see [noha_lid.md](noha_lid.md#update-fine-tuned-whisper-small): 85%
  after 10 s, against 69% stock). A wrong token hurts far less than before
  (49% at 10 s, against 30%), because the model now writes what it hears.
- 141 recordings is a small start. The du'a model's jump came from 10 hours of
  train-reciter audio, and more noha audio with lyrics is easy to find.

### Vocal separation

Separating vocals with Demucs (htdemucs) before whisper-small: 64% / 78% at
10 / 20 s with the language known (+2 / 0) and 54% / 74% with best-of-2
(-5 / -2). Not worth a second model on the device.

### "Not in the corpus" state

The tracker (developed alongside this) can now explain a window as coming
from none of the known texts, at `null_rate` edits per letter. On nohas, with
lengths counted in skeleton letters (fine-tuned whisper-small, best-of-2):

| null_rate | @10 s | @20 s |
|---|---|---|
| off | 68% (3%) | 80% (6%) |
| 0.7 | 67% (3%) | 79% (6%) |
| 0.6 | 65% (2%) | 73% (3%) |
| 0.5 | 60% (1%) | 68% (2%) |

A precision knob: at 0.6 it halves wrong nohas at 20 s for 7 points of
recall. For nohas it matters more than for du'as, since a majlis will hear
many nohas that no lyrics library has.

### Tracker tuning

A grid over the tracker's evidence weights, confidence threshold and start
prior, fit on half the reciters (by channel) and scored on the other half,
barely moves the needle: 58% / 74% at 10 / 20 s held-out against 59% / 76%
for the du'a defaults. The fitted settings raise the confidence threshold
(0.7 → 0.85), which trades about a point of accuracy for fewer wrong nohas
(5% → 3% at 10 s). Faster identification (sharper evidence) costs 9-13%
wrong nohas. **The tracker isn't the bottleneck; the transcripts are.**

### Where it fails

With large-v3 and the right language, 32 of 59 recordings are found from all
four cold starts and 3 from none (all English, where the description's lyrics
fit the audio poorly: 0.5 edits per letter against 0.14 for recordings that
are always found). Most partial failures are Farsi studio music videos
(نماهنگ) with full instrumentation.

### Scaling to a real lyrics library

Alignment against every lyric costs ~33 ms per window for 269 lyrics and
grows linearly (~0.5 s at 2,700), too slow for a phone at the 5-10k lyrics
that exist per language. An n-gram first stage (TF-IDF over skeleton
4-grams of everything heard so far) keeps the right noha in its top 20 for
74% of cold starts at 10 s and its top 50 for 78%. That's above what the
full tracker achieves (59%), so the HMM can run on the top ~50 at a fixed
cost whatever the library size.

| after | top-1 | top-5 | top-20 | top-50 |
|---|---|---|---|---|
| 5 s | 32% | 43% | 53% | 67% |
| 10 s | 54% | 67% | 74% | 78% |
| 20 s | 69% | 76% | 82% | 86% |

## Caveats

- 59 recordings and 269 lyrics is small. A real library of thousands makes
  identification harder (more near-duplicates), and the retrieval numbers
  above are optimistic for the same reason.
- Lyrics come from video descriptions: some are partial, and a few don't
  match what is sung.
- Only cold-start identification is measured. There are no line timings, so
  line accuracy (the du'a headline number) can't be scored yet.
- Channels stand in for reciters; the same reciter can upload on several.

## Reproduce

```bash
python scripts/noha_match.py fetch-lyrics            # test recordings + distractor lyrics
python scripts/noha_match.py transcribe --model small  # also: base, large-v3-turbo, large-v3
python scripts/noha_qwen.py --model Qwen/Qwen3-ASR-0.6B        # separate venv: pip install qwen-asr
python scripts/noha_omni.py --model-dir <sherpa-onnx omnilingual model dir>   # pip install sherpa-onnx
python scripts/noha_match.py eval --model small [--extra omni-ctc-1b]
python scripts/noha_match.py tune --model small
python scripts/noha_match.py retrieval --model small
python scripts/noha_separate.py && python scripts/noha_match.py transcribe --model small --vocals   # pip install demucs
python scripts/noha_finetune.py fetch && python scripts/noha_finetune.py label
python scripts/noha_finetune.py train --base openai/whisper-small
python scripts/noha_match.py transcribe --model models/whisper-small-noha-ct2
python scripts/noha_match.py eval --model models/whisper-small-noha-ct2 [--null 0.6]
```
