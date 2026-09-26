# Texts that share a passage (2026-09-26)

Reciting Ayat al-Kursi, the app never settled on a du'a. The corpus has it on its own
(`duasorg-namaz-e-wahshat`) and word for word inside Sahifa 54, and while the reciter is
in it the evidence can't tell the two apart: the tracker's mass splits between them, and
neither reaches `min_dua_confidence` (0.7), so nothing is shown.

It isn't one pair. 278 of the 505 texts share a run of 10+ identical words with another:
the Ramadan night du'as 22-30 (66-84% of each is shared with the others), the "second
du'a" of Ramadan days 24-29, Ziyarat Waritha and the ziyarat built on it, Dua Ya Mafza'i
inside Abu Hamza, and stock salawat everywhere.

## The rule (tracker.same_text_words)

Du'as whose likeliest positions sit in one run of at least 12 identical words (counting
back and forward from those positions), with at least 3 words of it still to come or the
end of a text reached, count as one du'a for the display threshold. Which one is shown:

- the du'a already on screen, if it was told apart from every other text before the
  passage began (someone reciting Sahifa 54 from its start stays on Sahifa 54);
- otherwise the one the passage sits nearest the start of (someone reciting Ayat
  al-Kursi sees Ayat al-Kursi). This choice is stable, so the display doesn't flip
  between identical texts.

The other texts are reported as `same_as` ("Also in Sahifa 54" under the title). Twelve
words is long enough that a shared opening salawat (7) or bismillah (4) doesn't count:
those still wait until the text becomes distinctive, as before. The rule changes only
what is reported; the belief, its evidence and the lock are untouched.

In a three-text toy corpus (tests/test_tracker.py), reciting the short text showed nothing
at all without the rule, and the right text on every hop but the first with it.

## Measured (phone ASR whisper-base-aug-v4, --latency 0.5)

Criteria set before looking: line accuracy may not fall and the wrong-du'a rate may not
rise by more than 0.2 points; identification should not get worse.

| | line acc | wrong du'a | id @3 s | id @5 s | id @10 s | id @20 s | id @30 s |
|---|---|---|---|---|---|---|---|
| train, rule off | 83.8% | 0.4% | 70.0% | 76.1% | 86.1% | 90.2% | 91.7% |
| train, rule on | 83.8% | 0.4% | 70.9% | 77.0% | 86.7% | 90.7% | 92.2% |
| test, rule off | 86.9% | 0.5% | 72.5% | 87.2% | 91.9% | 95.0% | 95.6% |
| test, rule on | 86.9% | 0.5% | 72.8% | 87.5% | 92.2% | 95.3% | 95.6% |

Following is identical on both splits: none of the DuaPlayer recordings is of a text with
a long shared passage, so this set can't show the case that prompted the change. The gain
is in identification from a random point mid-recitation, where a start inside a shared
passage used to show nothing. Test was looked at once. Logs: data/cache/shared_passage_log.txt,
data/cache/shared_passage_test_log.txt.
