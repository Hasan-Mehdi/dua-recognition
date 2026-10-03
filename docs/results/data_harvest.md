# Data harvest: thousands of hours of du'a recitations, labelled by their audio

2026-10-01 overnight. Hasan: "get as much data as possible. Any and all sources."

Before tonight the du'a training audio was 364 recordings, about 38 hours: DuaPlayer's
21 train recordings, 257 duas.org files and 86 YouTube videos (`train_v4`). That was
never a limit of what exists. The old YouTube harvester took the first three hits for
the ~25 du'as that already had a DuaPlayer recording, and nothing else looked further.

| | before (`train_v4`) | harvest (2026-10-02 22:30) |
|---|---:|---:|
| recordings | 364 | 21,172 downloaded (5,829 h), 13,967 labelled |
| labelled audio | ~38 h of windows | **1,405 h** of force-aligned lines; export `harvest_v2`: 1.25M windows, ~1,040 h distinct |
| voices (uploaders / reciters) | 309 | **3,558** with usable audio (1,685 with 10+ min); 2,643 in the export |
| texts with audio | 263 | 297 of 505 corpus texts (380 in the export incl. short spans), + 90 h of Mafatih texts the corpus lacks |
| human line timings | 39 DuaPlayer recordings | + 13.2 h of YouTube caption lines (106 recordings) |
| ordinary-voice clip sets | Common Voice, RetaSy crowd | + `quranspeech` (14.2 h, 30 people reciting Quran), `recerrors` (944), `nahw` (1,093 vowelled MSA), `fleurs` (2,311), `sawtarabi` (655) |

Usable hours by platform: shiavoice 661, YouTube 377, SoundCloud 208, Aparat 159 (archive.org's
1,019 h not processed yet: its files are the least likely to be recitations). Left out of the
export: 629 recordings in a test reciter's voice, 1,897 re-uploads, 576 held-out-text spans, 924
recordings that are, or sound like (voice >= 0.85), a validation uploader or a scenario-bench test
recording (data/testbed/test_voices.json). Full tables: [data_harvest_numbers.md](data_harvest_numbers.md).

## Where the audio came from

| source | how | what it holds |
|---|---|---|
| YouTube | 1,601 searches: every corpus text by its Arabic and English name, the widely recited texts under every spelling people title them with (Arabic, Persian, Urdu-style transliteration, English), the 30 Ramadan day du'as and the Sahifa by number, 50 du'a reciters by name, children/families/gatherings; then every channel with 3+ du'a hits crawled whole | professional and amateur reciters from Iraq, the Gulf, Iran, Pakistan/India, the diaspora; majlis streams |
| shiavoice.com | its du'a and ziyarat sections file each recording under the text it holds (one category per du'a: Nudba 202 recordings, Kumayl 187, Warith 145, Aminallah 133, ...); a JSON track list per category | ~4,500 recordings by ~1,200 different reciters, mostly Gulf and Iraqi |
| Aparat | Iran's video site, through its own JSON API (yt-dlp's extractor is broken); Persian queries first | Persian hay'at and mosque recitations, home recordings |
| SoundCloud | yt-dlp search on the widely recited texts | uploads of individual reciters |
| archive.org | search + per-item file lists, files kept only when their own name names a du'a/ziyarat (most "du'a" items there are lectures *about* du'as, Quran or nasheeds) | complete-du'a collections |
| YouTube caption tracks | human Arabic caption tracks fetched with every download (no extra requests) plus the 247 found earlier by find_captioned.py | human line timings, vowelled text, texts outside the corpus too |
| Hugging Face | ordinary voices: `quranspeech` (people reciting the Quran ayah by ayah), `recerrors` (non-professional reciters, MIT; clips with a reviewer-marked error left out), `nahw` (crowd-sourced fully vowelled MSA, CC-BY), `sawtarabi` (its MSA rows), `abjadkids` (children saying letters, numbers, colours, MIT) | voices the phone model is weakest on |

Nothing about an upload is trusted except its audio: titles only decide what is
downloaded.

## From audio to labels

1. **Letter posteriors** (`scripts/harvest_label.py frames`, GPU): one fp16 pass of the
   CTC teacher (`models/wav2vec2-quran-dua-voices`) over each whole file, 30 s chunks
   with 1 s context either side, decoding the next file on a thread meanwhile:
   ~600x real time. Kept as folded letter log posteriors + greedy ids (16 MB per hour).
2. **Which text** (`align`): the greedy text of 6 s windows every 2 s is scored against
   all 505 corpus texts (`CorpusIndex.word_costs`, the Myers kernel); a Viterbi pass over
   {nothing, text 1..505} with a switching penalty cuts the recording into spans of one
   text, or none (talk, translation, nasheed, a text the corpus doesn't have). One file
   can hold several texts (Iftitah recordings carry the Ramadan night du'as after it).
   Stretches no corpus text explains get a second pass against 290 training-only texts
   from Mafatih al-Jinan (`scripts/mafatih_texts.py`: the book's du'a and ziyarat blocks
   whose words aren't mostly in the corpus already, e.g. Ziyarat of Imam Ali on Ghadir,
   Imam Reza's, the fifteen Munajat, Dua Adeela, Asharat). They label training audio
   only; the app's 505 texts are untouched. The decoded book's titles sit one block off
   from their texts in places, so only the texts are used.
3. **Which line, when**: each span goes through the offline smoother (`offline.py`, the
   labeller already validated on DuaPlayer's human timings), then every placed line is
   force-aligned inside its slot with a *local* CTC alignment: audio before the line's
   first letter and after its last is free, so neighbouring lines and pauses don't count
   against it. A line is kept when its letters cost <= 1.5 nats each. Word start/end
   times come out of the same alignment.
4. **Captions** (`captions`): each human caption cue is force-aligned inside its own
   time span the same way. Cues often hold two corpus lines or sit a second off; the
   alignment keeps the cues whose text is really there, with word times.
5. **Voices** (`scripts/harvest_voices.py`): every recording's voice (ECAPA, as
   speaker_check.py) against the test reciters' named recordings and every held-out
   test-set recording; >= 0.7 means it never reaches training.
6. **Export** (`export`): 6 s windows every 3 s over runs of consecutive good lines, text
   = the words whose middle falls inside, in the spelling Whisper writes
   (`build_finetune_set.whisper_style`). Same row format as `train_v4`.

### Rules kept from the existing sets

- **Test reciters** are dropped by name before download (title, channel, uploader;
  Arabic, Persian and Latin spellings: Persian letters fold onto Arabic, so
  مرتضی قریشی matches قريش) and by voice after.
- **Test venues and uploaders** (`splits.is_test_upload`) are never downloaded.
- **Held-out du'as** (Tawassul, Ziyarat Ashura) never train anything: by id, by any id
  whose text is mostly theirs, and for caption tracks by their words.
- **Re-uploads**: one recording on YouTube, Aparat and shiavoice counts once (same main
  text, length within 1.5 s, voice >= 0.9).
- Validation split by uploader (3%), so a voice is on one side only.

## How far the labels can be trusted

- **Which text.** shiavoice files every recording under the text it holds, which nobody
  showed the labeller. On the first 186 recordings in 25 one-text categories, the main
  span named the category's text 98.4% of the time (Kumayl 30/30, Ahl al-Thughur 19/19,
  Iftitah 16/16, Ziyarat Ashura 13/14, Raf' al-Masahif 7/8), never nothing. YouTube and
  Aparat titles agree the same way on spot checks.
- **When.** Where an upload has human caption cues that hold exactly one corpus line,
  the blind line start (smoother + local forced alignment, cue text never seen) is a
  median 0.12 s from the human cue: 78% within 0.5 s, 82% within 1 s (139 lines, 8
  recordings). The rest are mostly refrains placed on another repetition; the caption
  cues themselves are often a second late or hold two lines, so the agreement is a floor.
- **What is exported.** Only lines whose letters force-align at <= 1.5 nats each, and
  only 6 s windows inside runs of such lines. A span can carry the wrong du'a name when
  a short generic du'a shares stock phrases with the audio, but its exported words are
  still the ones that aligned.
- **Who.** 2.8% of recordings matched a test voice and are never exported.

## First use: the phone CTC student (pre-registered, did not pass)

`ctc-student-base-h1`: `ctc-student-base-v6`'s recipe at v6's exact step count (75,406), with
60,000 voice-balanced harvest windows (2,503 uploaders, at most 400 each; teacher frames
sliced from the whole-file posteriors by `scripts/harvest_ctc_cache.py`) added to
`train_v4`'s 68,757 (`train_v4h`). Pre-registration (written before training): pass if
RetaSy streaming CER drops >= 1 pt, with Quran-Lab and Common Voice no more than 1 pt
worse and Hasan's sessions' follow word-exact no more than 2 pt worse. v6 re-run under
the same code (the follower changed since v6's own numbers).

| | v6 | h1 |
|---|---:|---:|
| RetaSy crowd, streaming / whole-clip CER | 40.1 / 22.2% | 40.7 / 23.5% |
| Quran-Lab, streaming / whole-clip CER | 29.2 / 22.0% | 28.3 / 19.9% |
| Common Voice, streaming / whole-clip CER | 20.4 / 11.6% | 20.6 / 12.2% |
| Hasan's 7 sessions, follow word exact / ±1 | 54.5 / 83.6% | 58.2 / 86.9% |
| val_v4 (held-out reciters), crop / 6 s CER | 24.0 / 47.6% | 23.6 / 46.4% |

Fails on its primary criterion: ordinary crowd voices (RetaSy) did not improve. The gains
are where the harvest's voices are: recitation (Quran-Lab, val reciters) and Hasan's own
sessions (+3.7 pt word exact). One seed each, so differences under ~1 pt are noise. Not
adopted. (The battery's DuaPlayer follow_eval runs were skipped: the follower was being
changed by another session that night.) What the harvest is: thousands of reciters and rooms, few ordinary phone readers;
the ordinary-voice sets (`quranspeech`, `recerrors`, `nahw`, `fleurs`) and the harvest's
children/home recordings are the part aimed at the RetaSy gap, and are not in h1. 17% of
h1's harvest windows came from recordings the test-voice check had not reached yet (none
of the four evaluation sets involve those reciters).

## Things found on the way

- **The voice check matters.** By name alone, Farahmand's Tawassul came back three times
  under other uploaders' names (voice 0.99), Hussein Ghareeb's Kumayl as "Dua Kumayl
  Deutsch" (0.95), and Halwachi's Ale Yasin and Warith in captioned re-uploads (0.79-0.86).
- **An existing leak:** the majlis test set holds a recording by Ahmad al-Fatlawi
  (testset dua-iftitah/mj-rRbyKY_), and `train_v4` already has 645 windows of his
  YouTube uploads (`yt:Ahmad Al Fatlawi / أحمد الفتلاوي`). Mahdi Sedghi and Usama
  al-Attar voices also match majlis test recordings. The harvest drops all of them; the
  old set needs a decision.
- **YouTube's bot check** stopped downloads after ~300 videos until yt-dlp got its
  JavaScript challenge solver (`pip install "yt-dlp[default]"` → yt-dlp-ejs). The
  venv's `yt-dlp.exe` launcher is broken; `python -m yt_dlp` works.
- **Caption tracks are loose:** cue starts sit up to 1-2 s off and one cue often holds
  two corpus lines, so captions are used through their own forced alignment, not as is.
- archive.org's du'a-named items are mostly Salafi lectures about tawassul, Quran
  mushafs and nasheeds; the file-name filter keeps the few real recitations.
- **The machine crashed at 02:38** (Kernel-Power 41, no dump, nothing logged before it)
  with the GPU at ~99% for hours, several CPU jobs and memory commit near its limit; it
  was back at 17:38. Downloads had run until then; the labelling loop had been paused
  since 23:40 by another session's pause file that was never removed. Restarted with
  one CUDA job and single-threaded BLAS in the labelling workers.

## Using it

```
python scripts/harvest_label.py export --version v2 --combine h2   # train_h2 = train_v4 + harvest (8 h/text cap)
python scripts/finetune_whisper.py --data h2 ...                     # same recipe as v4/syn-v5
python scripts/harvest_ctc_cache.py harvest_v2 --combine-with train_v4   # CTC student set (6 s clips, on D:)
python scripts/build_voice_sets.py quranspeech recerrors nahw sawtarabi   # ordinary-voice clip sets (--crowd style)
python scripts/harvest_report.py                                     # the numbers above
```

The CTC student needs teacher frames per window (`scripts/ctc_teacher.py harvest_v1`)
before `train_ctc_student.py --train-set harvest_v1`.

Audio stays on D:\dua-data\harvest (junction data/harvest) and is used only to train
models; source URLs are in each platform's `index.jsonl`.
