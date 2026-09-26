# "I don't know this du'a": recitations the corpus doesn't have (2026-09-25)

A tester recited the Friday du'a before Fajr from Mafatih. It isn't one of the
corpus's 91 texts. The app offered "Dua Allahumma Ya Man Yamlik?", then "Dua Night
23 / 25 / Jawshan Kabir?", locked onto Dua Night 24 and switched to Dua Mashlool. The
phrases it heard ("yā qāḍiya ḥawāʾij al-sāʾilīn", "yā man yaʿlamu …") recur almost
word for word in Jawshan Kabir and the Ramadan night du'as. The tracker had no
"none of these", so it had to pick the nearest text.

## The fix

`TrackerConfig.null_rate` adds one more hidden state: *the recitation isn't in the
corpus*. It explains a window as if the transcript were wrong at `null_rate` edits
per letter. A text the corpus has matches far better than that, even through ASR
errors. A different du'a that only shares stock phrases doesn't. The state is
entered and left slowly (`null_enter` 0.002, `null_leave` 0.05 per update). While
it holds most of the mass, no du'a is shown, the "is it…?" guesses disappear, and
after 10 s the page says so.

## How it was measured

Leave-one-out (`scripts/ooc_eval.py`): each recording is replayed against the corpus
*minus its own du'a*, so any du'a shown is wrong. The same settings are also scored
normally, so rejecting more can't hide a worse follower. `null_rate` was chosen on the
train reciters (stock large-v3-turbo). Here are the test reciters with the phone model:

| | unknown du'a: some du'a shown | recordings ever showing one | known du'as: line accuracy |
|---|---|---|---|
| before | 68.1% of the time | 16/16 | 85.4% |
| **null_rate 0.45** | **15.8%** | 16/16 | **85.3%** |
| null_rate 0.5 | 23.8% | 16/16 | 85.3% |

What remains is mostly brief: short stretches where a held-out du'a really does
recite lines another du'a contains.

On the tester's two recordings, played through the real web app (headless Chrome):
- **Friday du'a (not in the corpus):** nothing is shown, where before it showed two wrong du'as.
- **Dua Tawassul (in the corpus):** it locks on at 7 s and follows as before.

## The limit: the speech model

The state is only as good as the transcripts. On the tester's voice the phone model
(whisper-base) gets 54% of characters wrong, where stock large-v3-turbo gets 9.5%. At
that error rate a known du'a can look unknown for a few seconds. The page then holds
the last line rather than blanking, but the fix belongs in the speech model: see the
augmentation experiments.
