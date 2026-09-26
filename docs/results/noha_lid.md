# Can the noha mode detect the language from the audio alone?

A feasibility check before building a noha mode that detects Urdu / Arabic /
Farsi / English itself ([`scripts/noha_lid.py`](../../scripts/noha_lid.py)).

**Short answer: not with stock models; yes after fine-tuning on nohas.** On
sung nohas, the best stock language-ID front end names the right language only
~60% of the time after 5 s and ~75-79% after 10 s, and Whisper as-is gets Farsi
right only 26-43% of the time. A whisper-small fine-tuned on ~5,900 noha windows
(see [noha_match.md](noha_match.md#fine-tuning-on-nohas)) gets 76% after 5 s and
85% after 10 s (Farsi 78%), on reciters it never trained on; see
[*Update: fine-tuned*](#update-fine-tuned-whisper-small) at the end. Even so,
the language is best treated as soft evidence, with the lyrics deciding (see
*What this means for the design*).

## Setup

- **73 YouTube recordings** from 49 channels: Urdu 17, Arabic (Iraqi/Gulf
  latmiyat) 20, Farsi 21, English 15. Found by reciter-name searches in each
  language. Every label was checked against a large-v3 transcript of 20 s from
  the middle: two Punjabi/Saraiki nohas were dropped, one Farsi manqabat that
  the Urdu search found was relabelled, and one "English" video (a Farsi song
  with English subtitles) was dropped. Studio nohas, live majlis recordings and
  music videos (نماهنگ) are all included.
- **Replayed the way the app hears it:** cold starts at 25/45/65/85% of each
  recording (292 trials), a 6 s window every 1 s, windows without speech
  skipped by the tracker's own VAD rule, and log-probabilities summed over the
  speech windows heard so far. The choice is restricted to the four languages,
  with Whisper's Hindi token counted as Urdu.
- **Front ends:** Whisper's language token (base, small, large-v3-turbo,
  large-v3), plus a dedicated LID model,
  [VoxLingua107 ECAPA](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
  (~20 M parameters, trained on YouTube speech).

## Results

Accuracy after *t* seconds of listening. Trials in which the VAD hasn't heard
singing yet count as wrong: that's 23% of trials at 3 s and 7% at 10 s, mostly
music intros and breaks.

| front end | @3 s | @5 s | @10 s | top-2 @10 s | Urdu | Arabic | Farsi | English |
|---|---|---|---|---|---|---|---|---|
| whisper-base | 28% | 36% | 48% | 66% | 59% | 52% | **4%** | 93% |
| whisper-small | 35% | 48% | 65% | 77% | 78% | 71% | **31%** | 90% |
| large-v3-turbo | 43% | 53% | 65% | 77% | 82% | 75% | **26%** | 87% |
| large-v3 | **48%** | **60%** | **73%** | **81%** | 88% | 80% | 43% | 88% |
| VoxLingua107 ECAPA | 47% | 57% | 69% | 80% | 90% | 70% | 61% | **55%** |

(Per-language columns are at 10 s.)

**What goes wrong:**

- **Whisper hears singing as English.** Unrestricted, whisper-base labels sung
  Urdu as Hindi 30%, English 26% and Urdu only 17% of the time. For Farsi,
  every Whisper model's most common wrong answer is English (base guesses
  English 35% of the time and Farsi only 10%). Only English itself benefits.
- **Farsi is the hard language.** Iranian noha production (studio نماهنگ with
  full instrumentation, heavy reverb) plus Farsi's shared vocabulary with Arabic
  and Urdu. Farsi mostly goes to English, then Arabic and Urdu.
- **ECAPA has the opposite bias.** It's the best on Farsi (61%) and the worst
  on English (55%; it pushes English toward Arabic), and it over-predicts Urdu.
- **The first seconds are noisy.** Windows shorter than 3 s mislead. Counting
  only windows with ≥ 3 s of audio adds 2-6 points (large-v3: 60% @5 s, 75% @10 s).

**Fixes tried (fit on half the reciters, tested on the other half):**

| | @5 s | @10 s |
|---|---|---|
| large-v3, windows ≥ 3 s | 60% | 75% |
| + per-language bias correction | 62% | 79% |
| + ECAPA fused | 62% | 79% |
| whisper-small, windows ≥ 3 s | 51% | 69% |
| + per-language bias correction | 57% | 75% |

A simple per-language offset, which mostly un-learns Whisper's pull toward
English, is worth 4-6 points. Fusing ECAPA with Whisper adds nothing once that
offset is applied. With every fix combined, audio-only LID plateaus at ~75-79%
after 10 s.

## What this means for the design

1. **Don't route on the detected language.** A hard "detect, then use the
   Urdu/Farsi/... model" pipeline would start on the wrong language for 1 in 4
   listeners, and for more than half of Farsi listeners with a browser-sized model.
2. **Top-2 is much better than top-1** (77-81% at 10 s against 65-73%), so
   the language can narrow the search without deciding it. Use the LID
   probabilities as a soft prior over nohas by language in the tracker. The
   lyrics alignment makes the real decision, and once the tracker locks onto a
   noha, its language (known from the text) fixes Whisper's language token.
3. **Test next: can the lyrics decide?** The key open question is whether a
   transcript decoded in the *wrong* language still aligns to the right noha
   well enough to recover. That needs lyrics for these recordings, e.g. Urdu
   from urdunohay.com, which publishes Urdu script and Roman Urdu. If it does,
   the whole LID question mostly goes away. If not, the fallback is to decode
   the top-2 languages until the tracker locks (2× ASR cost for the first few
   seconds).
4. **whisper-small, bias-corrected, is the browser candidate.** Its LID is
   within ~4 points of large-v3 once corrected, and fine-tuning on nohas
   (planned anyway, for transcription) should also sharpen its language token,
   since a fine-tune sees the correct language token with every example.
5. **Punjabi/Saraiki nohas are common** (2 of the first 19 "Urdu" results).
   They're worth a fifth label, even if they share the Urdu model.

## Reproduce

```bash
python scripts/noha_lid.py fetch --per-lang 20     # ~75 recordings -> data/noha/ (ignored)
python scripts/noha_lid.py eval --models ecapa base small large-v3-turbo large-v3
```

YouTube search results change, so a re-fetch won't give exactly this set. The
video IDs, labels and per-window scores behind these tables are in
[noha_lid.json](noha_lid.json).

## Update: fine-tuned whisper-small

A whisper-small fine-tuned on nohas with lyrics-verified labels
([`scripts/noha_finetune.py`](../../scripts/noha_finetune.py); 141 recordings
from channels that never appear here) sees the right language token with every
training example. Its language ID was re-measured with the same protocol on
the enlarged set: 99 recordings (Urdu 26, Arabic 30, Farsi 24, English 19),
including the lyrics test set of [noha_match.md](noha_match.md). Stock
whisper-small is shown on the same set.

| front end | @2 s | @3 s | @5 s | @10 s | Urdu | Arabic | Farsi | English |
|---|---|---|---|---|---|---|---|---|
| whisper-small, stock | 30% | 38% | 53% | 69% | 76% | 78% | 32% | 92% |
| **whisper-small, fine-tuned on nohas** | **52%** | **64%** | **76%** | **85%** | **89%** | **84%** | **78%** | 88% |

(Per-language columns at 10 s; "no singing heard yet" still counts as wrong,
7% of trials at 10 s.) Unrestricted, the fine-tuned model's top language on
sung Urdu is Urdu 90% of the time (stock: 37%, with Hindi 28% and English 14%),
and Farsi is Farsi 81% of the time (stock: 40%). The pull toward English on
music is gone. This beats every stock option above, large-v3 and the
dedicated LID model included, at a size that runs in a browser.
