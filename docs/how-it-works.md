# How it works

![Animated explainer: evidence from one window matches all 14 repetitions of a refrain; multiplying by the tracker's prediction leaves only the right one](explainer.gif)

*The heart of a 4½-minute explainer that starts from what a du'a follower is for
([full video](explainer.mp4), [source](anim/explainer.py), made with
[Manim](https://www.manim.community/)). Every curve is the real tracker at one second of a
held-out recording.*

## The problem

Following a du'a is harder than transcribing it:

- **Refrains.** Du'as repeat themselves. In Dua Tawassul, 70 of 115 lines are
  refrains: *"yā wajīhan ʿinda-llāh, ishfaʿ lanā ʿinda-llāh"* comes back after every one of
  the 14 names. A transcript of "the current window" matches all 14 repetitions
  equally well, so matching text alone can't say which one is being recited.
- **Shared openings.** Nearly every du'a opens with *bismillāh* and a *ṣalawāt*, and
  several share whole phrases (Dua Baha and Dua Tawassul both have a line starting
  *"allāhumma innī asʾaluka…"*). Committing too early means showing the wrong du'a.
- **Melodic recitation.** A single line can be drawn out for 15 seconds, so a 6-second
  window often holds half a line or less. Off-the-shelf small Whisper models barely
  transcribe it.
- **Real time, and stopping.** The display has to move when the reciter moves, on
  hardware people actually have, and stay put when they stop: the last 6 s of audio
  still hold their last words for seconds after they fall silent.

## The pipeline

```
mic ─▶ 6 s window every 1 s ─▶ Whisper ─▶ align against every du'a ─▶ HMM follower ─▶ du'a · line · word
                                  │         (semi-global edit        (position prior
                                  │          distance, vectorized)    resolves refrains)
                                  └─ fine-tuned whisper-base: CPU / in-browser
```

### 1. Transcribe

Each hop transcribes the last 6 s (faster-whisper / CTranslate2, batched offline,
batch-of-one live). Whisper's usual silence hallucinations (*"ترجمة نانسي قنقر"*,
*"شكرا"*) and decoding loops are filtered out ([`asr.py`](../src/dua_recognition/asr.py)).

### 2. Align against the whole corpus at once

[`align.py`](../src/dua_recognition/align.py). All 505 texts become one normalized letter
string (tashkeel stripped, alef/ya/ta-marbuta and Urdu-style variants folded, spaces
dropped so Whisper's word splitting doesn't matter). For the window's transcript *h*, a
semi-global edit-distance DP gives, for **every** word in the corpus, the cost of *h*
ending exactly there. It runs as Myers' bit-parallel algorithm (each DP column held as
one bit per letter of *h*, blocked for long fragments): one pass over the 136,503-word
corpus takes ~3 ms in Python (numba) and ~11 ms in the browser, with exactly the DP's
numbers.

### 3. Follow with an HMM

[`tracker.py`](../src/dua_recognition/tracker.py). This is *score following*, the
technique used to turn sheet-music pages in time with a performance, applied to
recitation. The hidden state is the word being recited, across all du'as.

- *Predict:* the reciter moved forward a few words since the last hop, may have
  gone back a few (reciters repeat lines), and with a small "teleport" probability
  jumped anywhere, so it recovers if someone skips ahead or switches du'a.
- *Pauses:* each window also reports how long the reciter has been silent (a voice
  detector, or loudness over the window's floor that comes within 15 dB of the reciter's
  voice: the long melodic notes voice detectors miss count, the room hiss a phone turns up
  once they stop doesn't). The belief moves only for time they were making sound, and the
  display doesn't run on into the next line while they're silent. The page also listens
  to the audio that arrived while Whisper ran, so an update shown after they stopped
  doesn't predict them any further. After a second of silence it steps back if it already
  had. Majlis mode, for a professional who breathes between lines, keeps running on
  ([pauses.md](results/pauses.md), [stops.md](results/stops.md)).
- *Speed and tempo:* how far "a few words" is comes from the human timings
  (a Poisson mixture over the train reciters' speeds), and the tracker keeps a weight
  per speed that adapts to the reciter in front of it: a slow, melodic reciter and a
  brisk one get different predictions. Getting the speed right is worth about 4 points
  of line accuracy ([speed_prior.md](results/speed_prior.md)).
- *Correct:* weight each word by `exp(-κ · cost)` from step 2.
- *Prior:* 30% on the opening words of each du'a and 70% anywhere, uniform over
  du'as. A uniform prior over *words* would make long Kumayl ten times likelier than
  short Faraj before a word is heard.

A refrain matches all its repetitions equally well, and the predicted position is what
separates them. Identification comes for free: the posterior mass inside a du'a is the
probability that it's the one being recited, and nothing is shown until one du'a holds
70% of it. Until then, the likeliest du'as are offered as "Is it…?" chips. Where several
texts share a passage word for word (Ayat al-Kursi inside Sahifa 54), they count as one
until the recitation tells them apart ([shared_passages.md](results/shared_passages.md)).

### 4. Run on a CPU or a phone

[`finetune_whisper.py`](../scripts/finetune_whisper.py). large-v3-turbo is accurate but
needs ~3.3 s per window on CPU. Small models are fast enough but barely understand
recitation out of the box (stock whisper-small: 88% window CER). So a small model is
fine-tuned on train-reciter windows with **reference-snapped labels**: turbo transcribes
each window, the human line timings bound where in the text it can be, and the label
becomes the *reference* text the transcript aligns to. The teacher only decides where a
window starts and ends in the text, so its spelling mistakes never reach the labels.

Windows come from DuaPlayer's train reciters, YouTube, and duas.org recordings (the last
two labelled by an offline forward-backward smoother,
[`offline.py`](../src/dua_recognition/offline.py)), 128 h in all. A speaker-embedding
check keeps every test reciter's voice out, and caught one test recording re-uploaded to
YouTube under another name. Audio is augmented with level changes and noise, speed and
vocal-tract-length perturbation and SpecAugment, and half the windows get synthetic room
reverb and noise. The best starting point was
[tarteel-ai/whisper-base-ar-quran](https://huggingface.co/tarteel-ai/whisper-base-ar-quran)
(already trained on Quran recitation), and the result runs in the browser via ONNX
([`web/`](../web/)).

The voices matter as much as the model. Professional reciters alone don't show how a
model copes with someone reading along, so ordinary voices are scored too, on 409
verified clips from [RetaSy](https://huggingface.co/datasets/RetaSy/quranic_audio_dataset)'s
crowd-sourced recitations ([`voice_eval.py`](../scripts/voice_eval.py)):

| model | character errors, everyday voices |
|---|---|
| large-v3-turbo, fine-tuned (server) | 15.4% |
| large-v3-turbo, stock | 22.8% |
| whisper-small, fine-tuned | 27.2% |
| **whisper-base, fine-tuned + synthetic voices (phone)** | **24.7%** |
| whisper-base, fine-tuned, reciters only | 29.9% |
| whisper-base, earlier fine-tune (fewer voices, no speed/VTLP/SpecAugment) | 38.7% |

The phone model's training set includes 15 hours of synthetic speech: 312 of RetaSy's own
volunteers (none of them in the test clips), cloned by a zero-shot TTS
([OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)) reading the du'as at an unhurried
pace ([synthetic_voices.md](results/synthetic_voices.md)). Retrained without them, the same
recipe scores 32.3%, so the synthetic voices remove about a quarter of the errors on
everyday voices, including on verses no synthetic clip ever read.

Whisper pads every input to 30 s, while the app sends 6 s windows. The phone model is
therefore cut to an 8 s context and fine-tuned there for one epoch: four times less work
per update in the browser, and 24.1% on the RetaSy clips that fit in 8 s
([phone_speed.md](results/phone_speed.md)).

## Word by word

A transcript once a second has no timing inside it, and reaches the screen a second or more
after its audio. So on the phone a second, small model listens too: the encoder of the phone's
Whisper with one layer onto the alphabet, trained from a 300 M-parameter wav2vec2 (itself
tuned on ordinary voices), giving
letter probabilities every 20 ms for the latest 2 s, up to ten times a second
([`ctc_student.py`](../src/dua_recognition/ctc_student.py)). The word follower
([`follower.py`](../src/dua_recognition/follower.py), `web/follower.js`) scores those frames
against the words around the current one and moves to the word whose letters were just heard.
A pause leaves it on the last word said, because the frames after it are silence; nothing is
predicted, so nothing has to be taken back. The tracker still finds the du'a and anchors the
follower. On held-out reciters at the phone's delay, the right word is lit 79% of the time
instead of 35%, and the display practically never steps back a line (once in 224 minutes of
recitation, against 114 times), at the cost of line changes ~0.4 s later
([phone_follower.md](results/phone_follower.md)). On the phone the follower's own compute is
most of its delay; with 2 s windows, a step every 0.1 s and audio in 50 ms chunks, a word is lit
~0.28 s after it starts instead of ~0.44 s. Going back to the start of a line costs it a fixed
amount rather than a price per word, so a reader who says a line again is followed back
([phone_latency.md](results/phone_latency.md)).

The server runs the same follower on the same small model. Without it (`?words=off`), the
highlight glides across the line at the reciter's speed between tracker updates and stops the
moment the page hears them stop ([`display.py`](../src/dua_recognition/display.py),
[display_lead.md](results/display_lead.md), [stops.md](results/stops.md)).
