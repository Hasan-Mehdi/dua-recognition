# How it works

![Animated explainer: evidence from one window matches all 14 repetitions of a refrain; multiplying by the tracker's prediction leaves only the right one](explainer.gif)

*The tracker at one second of a held-out recording. The [full video](explainer.mp4) is a
narrated walkthrough, with subtitles, for someone who knows nothing about it yet. It follows one
recitation of Du'a al-Iftitah, by a reciter the models never trained on, from start to end: how a
phone hears and what the two models are, how the du'a gets named, how the highlight follows the
word, and what the app does when he stops, when someone talks to him and when he goes back two
lines, with the real app shown on the same recording and his recitation heard under the narration
(Abu Thar Al-Halawaji's recording, from [DuaPlayer](https://www.duaplayer.org)). The pictures are the page's own decisions
on that reading, replayed ([source](anim/explainer.py), made with
[Manim](https://www.manim.community/), voiced by [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)).*

## The problem

Following a du'a is harder than transcribing it:

- **Refrains.** Du'as repeat themselves. In Dua Tawassul, 70 of 115 lines are
  refrains: *"yā wajīhan ʿinda-llāh, ishfaʿ lanā ʿinda-llāh"* comes back after every one of
  the 14 names. A transcript of "the current window" matches all 14 repetitions
  equally well, so matching text alone can't say which one is being recited.
- **Shared openings.** Nearly every du'a opens with *bismillāh* and a *ṣalawāt*, and
  several share whole phrases (Dua Baha and Dua Tawassul both have a line starting
  *"allāhumma innī asʾaluka…"*). Committing too early means showing the wrong du'a.
- **Drawn-out recitation.** A single line can be drawn out for 15 seconds, so a 6-second
  window often holds half a line or less. Off-the-shelf small Whisper models barely
  transcribe it.
- **Readers move around.** People reading along go back a line, say a line again, skip
  ahead, stop to talk or add a salawat between lines. Professional recordings hardly ever
  do, so a follower tuned on them alone can fail on readers who do.
- **Real time, and stopping.** The display has to move when the reciter moves, on
  hardware people actually have, and stay put when they stop: the last 6 s of audio
  still hold their last words for seconds after they fall silent.

## The pipeline

```
mic ─┬─▶ 6 s window, every 1-2 s ─▶ Whisper ─▶ align against every du'a ─▶ HMM tracker ─▶ du'a · line
     │                                 │        (semi-global edit          (position and
     │                                 │         distance, vectorized)      du'a priors)
     │                                 └─ fine-tuned whisper-base:             │ du'a, line
     │                                    CPU or in the browser                ▼ probabilities
     └─▶ new audio, every 0.1 s ───▶ small CTC model ─▶ letters, every 20 ms ─▶ stream decoder ─▶ word
                                                                              (reading moves over
                                                                               the whole du'a)
```

Whisper and the tracker find the du'a and the line; the small CTC model and the stream decoder
place the word ([word by word](#word-by-word)). On the phone Whisper runs every second until the
du'a is found and every 2 s once the decoder is placing words. The server engine runs the same
page with its models on the server instead of the phone, and Whisper every second
([server_engine.md](results/server_engine.md)). Since 2026-10-07 the server has its own: a
large-v3-turbo fine-tuned in full, the CTC model on 3 s windows every 0.05 s, and a profile of
decoder and tracker settings the page takes from the server
([server_profile.md](results/server_profile.md)).

### 1. Transcribe

Each hop transcribes the last 6 s (faster-whisper / CTranslate2, batched offline,
batch-of-one live). Whisper's usual silence hallucinations (*"ترجمة نانسي قنقر"*,
*"شكرا"*) and decoding loops are filtered out ([`asr.py`](../src/dua_recognition/asr.py)).

### 2. Align against the whole corpus at once

[`align.py`](../src/dua_recognition/align.py). All 522 texts become one normalized letter
string (tashkeel stripped, alef/ya/ta-marbuta and Urdu-style variants folded, spaces
dropped so Whisper's word splitting doesn't matter). For the window's transcript *h*, a
semi-global edit-distance DP gives, for **every** word in the corpus, the cost of *h*
ending exactly there. It runs as Myers' bit-parallel algorithm (each DP column held as
one bit per letter of *h*, blocked for long fragments): one pass over the corpus takes
~3 ms in Python (numba) and ~11 ms in the browser, with exactly the DP's numbers (measured at
136,503 words; with the Mafatih texts added it is 140,803).

### 3. Follow with an HMM

[`tracker.py`](../src/dua_recognition/tracker.py), and its JavaScript port `web/tracker.js`.
The hidden state is the word being recited, across all du'as. Each update combines two
things: where the reciter should be by now, given where they were, and how well each word
fits what the window heard.

- *Predict:* the reciter moved forward a few words since the last hop, may have
  gone back a few (reciters repeat lines), and with a small "teleport" probability
  jumped anywhere, so it recovers if someone skips ahead or switches du'a.
- *Jumps within the du'a:* 2% per second of audio of going to the start of any line of
  the same du'a (`p_line_jump`). The teleport floor is spread over every text, so on its own
  a reader who skips to a line they know was found only slowly
  ([out_of_order.md](results/out_of_order.md)).
- *Pauses:* each window also reports how long the reciter has been silent (a voice
  detector, or loudness over the window's floor that comes within 15 dB of the reciter's
  voice: the drawn-out words voice detectors miss count, the room hiss a phone turns up
  once they stop doesn't). The probabilities move only for time they were making sound, and
  the display doesn't run on into the next line while they're silent. The page also listens
  to the audio that arrived while Whisper ran, so an update shown after they stopped
  doesn't predict them any further. After a second of silence it steps back if it already
  had. Majlis mode, for a professional who breathes between lines, keeps running on
  ([pauses.md](results/pauses.md), [stops.md](results/stops.md)).
- *Speed and tempo:* how far "a few words" is comes from the human timings
  (a Poisson mixture over the train reciters' speeds), and the tracker keeps a weight
  per speed that adapts to the reciter in front of it: a slow reciter who draws words out
  and a brisk one get different predictions. Getting the speed right is worth about 4 points
  of line accuracy ([speed_prior.md](results/speed_prior.md)).
- *Correct:* weight each word by `exp(-κ · cost)` from step 2.
- *Prior:* 30% on the opening words of each du'a and 70% anywhere. Each du'a gets its
  share first and spreads it over its words: a prior uniform over *words* would make long
  Kumayl ten times likelier than short Faraj before a word is heard.

The share each du'a gets follows how often it is recited (`TrackerConfig.popularity`,
`popularity` in tracker.js): P(du'a) ∝ (recordings + 1)<sup>0.5</sup>, where recordings
counts the uploads in the data harvest that read at least three lines of that text no other
text reads (`data/dua_popularity.json`, built by
[`dua_popularity.py`](../scripts/dua_popularity.py): Kumayl 794, Ziyarat Ashura 580, Warith
477, 193 of the 522 texts none). With the square root, Kumayl starts 28 times likelier than a
text nobody uploaded rather than 795 times. When someone opens with words many texts share
(*allāhumma innī asʾaluka*, *as-salāmu ʿalayka yā abā ʿabdi-llāh*), the texts people actually
recite come first.
On the scenario bench's held-out voices it names the du'a within 10 s 2 points more often
(88% to 90%) and shows the wrong du'a 0.5% of the time instead of 0.9%, while a du'a the app
doesn't have is shown as some other one 1 point more often. An exponent of 1.0 found the du'a
a little more often still, but in Hasan's Namaz-e-Wahshat sessions it held the more recited
text that shares Ayat al-Kursi after the passage had ended (jumps 3.1 to 7.3 per 10 minutes),
so 0.5 is the default ([finding.md](results/finding.md)).

A refrain matches all its repetitions equally well, and the predicted position is what
separates them. Identification comes for free: the posterior mass inside a du'a is the
probability that it's the one being recited, and nothing is shown until one du'a holds
70% of it. Until then, the likeliest du'as are offered as "Is it…?" chips. Where several
texts share a passage word for word (Ayat al-Kursi in Sahifa 54, Namaz-e-Wahshat and an Eid
al-Mubahila text), they count as one until the recitation tells them apart
([shared_passages.md](results/shared_passages.md)). The tracker's likeliest du'a can still
flip between them while the passage lasts; the word display doesn't follow it there (below).

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
| large-v3-turbo, fine-tuned (server, until 2026-10-07) | 15.4% |
| large-v3-turbo, stock | 22.8% |
| whisper-small, fine-tuned | 27.2% |
| whisper-base, fine-tuned + synthetic voices (phone, until 2026-10-07) | 24.7% |
| whisper-base, fine-tuned, reciters only | 29.9% |
| whisper-base, earlier fine-tune (fewer voices, no speed/VTLP/SpecAugment) | 38.7% |
| *at an 8 s context, on the 388 clips that fit in 8 s:* | |
| the same whisper-base + synthetic voices, cut to 8 s | 24.1% |
| **whisper-base, distilled from the server's turbo, halls (phone, since 2026-10-07)** | **21.4%** |
| **large-v3-turbo fine-tuned in full at 8 s (server, since 2026-10-07)** | **11.5%** |

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

A later harvest of uploaded recitations (YouTube, shiavoice.com, Aparat, SoundCloud) gathered
5.8k hours, 1,405 of them labelled line by line from their audio, in about 3,558 voices
([data_harvest.md](results/data_harvest.md)). Retrained on it alone, the small CTC model
(below) and the phone's Whisper both failed the bars set before training
([finding.md](results/finding.md)). What it did help train is a larger teacher: since
2026-10-07 the server's Whisper is large-v3-turbo fine-tuned in full at an 8 s context on the
train windows, about 200k harvest windows, the synthetic voices and simulated halls
([server_profile.md](results/server_profile.md)), and the phone's Whisper is the whisper-base above
trained two more epochs on the train windows and 139k harvest windows from 2,555 uploaders, with
rooms, halls through a PA and crowds, learning half from the transcripts and half from the
turbo's own token probabilities on the clean window (distillation;
[phone_engine.md](results/phone_engine.md)). Its gain is in hard audio: about the same letter
error on clean windows (13.9% against 13.5%), fewer errors on everyday voices (table above) and
in the simulated masjid. The harvest's held-out uploaders are voices on the scenario bench, and
its counts are the du'a prior above.

## Word by word

A transcript once a second has no timing inside it, and reaches the screen a second or more
after its audio. So on the phone a second, small model listens too: the encoder of the phone's
Whisper with one layer onto the alphabet, trained from a 300 M-parameter wav2vec2 (itself
tuned on ordinary voices), giving letter probabilities every 20 ms
([`ctc_student.py`](../src/dua_recognition/ctc_student.py)). On a phone the model's own compute
is most of the delay ([phone_latency.md](results/phone_latency.md)), so since 2026-10-07 the page
runs it streaming ([`stream_ctc.py`](../src/dua_recognition/stream_ctc.py), `ctc-stream-base-v1`):
every 0.1 s it encodes only the audio that is new, its first layer looking 0.2 s ahead and the
others only back, against keys and values it keeps from earlier steps (`web/ctc-stream.js`). A step
costs about a quarter of re-encoding a 2 s window (7 ms against 28 ms in Chrome on a desktop), and
the page still hands the decoder the latest 2 s of frames ([phone_engine.md](results/phone_engine.md)).
The phone's Whisper is distilled from the server's turbo, with halls and crowds in its training.

### The stream decoder

[`stream_follower.py`](../src/dua_recognition/stream_follower.py) and
`web/stream-follower.js`, the page's default since 2026-10-03. It keeps a probability for
every letter of the du'a and updates it frame by frame, a CTC forward pass over the whole
text, with reading itself as the transitions. The text in order is free. Starting the line
again costs 3 nats, going back one to three lines 5-7, skipping two to four lines ahead 8-10,
and anywhere else in the du'a 8 (13 until 2026-10-07), spread over its lines. Talk that isn't
the du'a is a state of its own: a filler that takes blank frames free and pays only for letters,
and that can come back wherever a finished line could, since a reader back from talking may
start anywhere. The
salawat said between lines is a short chain of its own, left for the next line or the same
one. A pause is a blank, so the highlight stays on the last word said rather than running on
at a predicted pace. A refrain keeps every repetition alive until the words that differ are
heard. While the page's stop detector hears no voice, letters cost 4 nats more, so the stray
letters room tone makes are less likely to move the highlight to the next line's start.

A word is shown once it holds 35% of the probability, a word on another line once it holds
half for two steps in a row. A move into the next line also has to be heard: it is shown only
when the frames prefer the words of the new line reached so far over the same words of the
lines the reader could be on instead (the same line again, one to three back, two to four
ahead) by 3 nats, with each alternative's transition cost taken back off so the prior doesn't
decide; or after 0.3 s of audio with no alternative ahead by more than 1 nat. An alternative
that reads the same letters so far (every line of Baha opens with *allāhumma innī asʾaluka*)
can't be told apart by listening yet and is left out. Before this rule, 72% of the display's
jumps and 95% of its early moves were crossings into the next line, made on the prior a frame
or two after a line ended, whatever the reader did next. On the held-out test voices the rule
cut jumps by 23% and early moves by 26%, for lines entered 0.06 s later (median), about a point
fewer exact words and slightly more lost time in the majlis and harvest lanes
([jumps.md](results/jumps.md)). It is on since 2026-10-04; `?sc=nextMargin:0` turns it off.
The salawat counts as one more alternative there, so a line that opens as the salawat does
(*allāhumma …*) waits for the word where they part.

Since 2026-10-07 four more rules are on, first tried on the server engine
([server_profile.md](results/server_profile.md), [phone_engine.md](results/phone_engine.md)):

- **Copies.** Lines with the same letters (the block of Dua Tawassul that comes back after each
  of its 14 names) count as one line for the display. A reader inside such a block can't be
  placed until a line that differs; the probability sat equally on every copy, none reached the
  threshold, and the highlight froze for tens of seconds. Now it follows the words on the copy
  nearest the word on screen.
- **A salawat isn't a lapse.** Time the frames put in the salawat chain doesn't count towards
  letting go of the du'a: a 10 s salawat in Du'a Baha used to send the reader's place back to its
  first line.
- **The frames can call a far jump.** Every half second the latest 1.5 s of letters are scored
  against the whole du'a. When one line beats every other but its neighbours by 4 nats, a fifth
  of the probability moves to just after its best word, so a reader who jumps far is followed
  within about a second instead of when Whisper's tracker gets there, about 6 s later.
- **No du'a change inside a shared passage.** While the tracker's last four words in the other
  du'a are also the shown du'a's words near where the reader is (allowing two letters of
  different spelling), the display stays where it is. Texts that share a passage (Ayat al-Kursi
  in Namaz-e-Wahshat and an Eid al-Mubahila text) no longer switch the title back and forth. It
  waits until the du'a has been on screen for 10 s, as a reader who starts inside a shared
  passage may not be in the right text yet; in a long shared passage that isn't always enough
  ([evaluation.md](evaluation.md#known-limits)).

The tracker still finds the du'a. The decoder starts on the tracker's line probabilities,
takes them in as gentle evidence at each update, changes du'a only once the tracker has held
the new one for 2 s (texts that share a passage flip it for a moment), and lets go of a du'a
the tracker has lost after 20 s of silence, but after 4 s while the reader is making sound,
which is more likely a text the app doesn't have than a lull. To stay fast in long du'as it
updates only the lines that hold some probability, five lines either side of them, and the
lines the tracker or a search of the latest frames proposes: 1-11 ms a step on a desktop in
the longest du'as, from the first step.

Against the rule-based follower it replaced, on the scenario bench's held-out voices (before
the next-line rule): on the reader's line 83% → 87%, the right word 61% → 67%, jumps 3.4 → 2.3
and lost episodes 4.2 → 1.8 per 10 minutes, a reader's go-back, skip or jump followed within
3 s 58% → 72% of the time, and the highlight held through pauses, talk and salawat 69% → 88%.
It is worse when the reader jumps around the du'a (followed within 3 s 13 points less often:
the filler that keeps talk from moving the highlight also holds a far jump back a moment), it
shows some du'a for a text the app doesn't have more often (28% against 21%), and the 2 s
du'a-change rule costs a few points of finding the du'a in some conditions. Played through the
real page in headless Chrome at phone speed, it was on line 81% → 86% and jumped 3.8 → 1.2
times per 10 minutes, agreeing with the replay except on holding still (73% → 60%, resting on
two items) ([bench.md](results/bench.md)).
`tests/test_stream_parity.py` keeps the JavaScript equal to the Python step for step.

### The rule-based follower

[`follower.py`](../src/dua_recognition/follower.py) and `web/follower.js` (`?follower=rules`
on the page) rescore the latest frames against the words
around the current one every step and move to the word whose letters were just heard, with
rules for going back, restarting a line, pauses and jumps. This is what first replaced the
predicted, gliding highlight: on held-out reciters at the phone's delay, the right word was
lit 79% of the time instead of 35%, and the display stepped back a line once in 224 minutes
of recitation against 114 times, at the cost of line changes ~0.4 s later
([phone_follower.md](results/phone_follower.md)). Later it got faster and learned to follow a
reader who says a line again ([phone_latency.md](results/phone_latency.md)) or jumps to
another line ([out_of_order.md](results/out_of_order.md)). By then it had 37 settings, each
tuned on its own benchmark; on the scenario bench, where readers move around, its rules got in
each other's way, and on refrain du'as it could lock onto the wrong repetition for minutes
([bench.md](results/bench.md)).

Until 2026-10-04 the server engine ran this follower, on 3 s windows. Without any follower
(`?words=off`), the highlight glides across the line at the reciter's speed between tracker
updates and stops the moment the page hears them stop
([`display.py`](../src/dua_recognition/display.py),
[display_lead.md](results/display_lead.md), [stops.md](results/stops.md)).
