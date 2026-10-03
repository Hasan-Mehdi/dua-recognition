# Tilawa's streaming phoneme engine on du'as: fast, but it doesn't follow them (2026-09-30)

[Tilawa](https://github.com/yazinsai/tilawa) v0.3.1 (built from source, revision f64a8b1)
follows Quran recitation in the browser. Its engine combines:

- a streaming Zipformer2-CTC phoneme model: 66 MB int8, a 0.5/0.5 interpolation of
  Quran-Lab's zipformer_p-arabic-v3.1 with a fine-tune, licensed Quran-Lab NPL-1.2
  (non-commercial, share-alike);
- a word tracker over its own Quran corpus.

The earlier small-CTC comparison (`small_ctc.md`) was of Tilawa's older FastConformer. This
one tests the whole current engine, run in Node with onnxruntime-node in 480 ms chunks, as
a live microphone would feed it.

## On Hasan's Ayat al-Kursi sessions

Ayat al-Kursi is Quran 2:255-257, so the engine could follow it with its own corpus. The
word positions map onto our 19 lines by word count. Five sessions, against the turbo
reference:

| | line | ±1 | steps back | flickers | found at | compute |
|---|---:|---:|---:|---:|---:|---|
| our app, as recorded | 64% | 97% | 9 | 7 | 10-20 s | ~730 ms per 1 s update (desktop) |
| Tilawa | 60% | 94% | 3 | 6 | 9-16 s | ~15 ms per 0.48 s chunk (RTF 0.035) |

## On du'as

The engine matches against words written in a tajweed-aware phonetic script (ٱللَّهِ ->
للَااهِ), and that table exists only for the Quran. `scripts/tilawa_phonemes.py` writes the
du'as in the same script with about twenty rules: gemination, long vowels, hamzat al-wasl,
the article and sun letters, tanween, ta marbuta, qalqala, madd before hamza, and pausal
forms. On the Quran's own 77,433 words it reproduces Tilawa's table at a 7.2% character
error rate (70.6% of words exact). Most of the misses are ghunna and madd lengths.

The engine's corpus holds exactly 114 surahs. For this test it held the 22 du'as that have
DuaPlayer recordings plus 92 other du'as: a smaller search space than our 505 texts. The
train reciters' 22 recordings (5.8 h) were scored as `evaluate.py` scores the tracker: the
line on screen once a second, against the human timings.

| | line | ±1 | refrain | du'a right | wrong du'a | lag | lines missed | jumps/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tilawa's engine, du'a corpus | 50.4% | 72.3% | 29.0% | 88.0% | 0.6% | 2.0 s | 27.6% | 0.51 |
| ours, phone model, 0.5 s delay (`display_stability.md`) | 81.5% | 96.3% | 75.2% | 96.7% | 0.5% | 0.7 s | 1.8% | 0.81 |

## Verdict

Not a replacement.
- **Where it falls short:** it names the du'a well enough, but it misses more than a
  quarter of the line changes, and a line is right only half the time.
- **Why, in part:** its tracker emits word progress only when words match, and it was tuned
  on Quran recitation against an exact phoneme table. Here it matched rule-made phonemes.
- **What's worth keeping:** the model's cost, about a twentieth of Whisper-base, updating
  every half second.

If the licence suits Hasan, the next test would feed its phoneme posteriors into our own
tracker, as the wav2vec2 word follower does, rather than use Tilawa's tracker.
`scripts/tilawa_phonemes.py` is the piece that makes that possible.

## Reproduce

```
python scripts/tilawa_phonemes.py check --quran zipformer_quran.json
python scripts/tilawa_phonemes.py corpus dua_corpus.json --quran zipformer_quran.json --duas <22 recorded + 92 others>
```

The Node harness and its scorer lived in the session scratchpad; the model, the corpus and
the Quran table come from Tilawa's v0.3.0 release and are not redistributed.
