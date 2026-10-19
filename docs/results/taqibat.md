# Ta'qibat: the du'as after salat (2026-10-10)

Hasan asked whether the app has the du'as of duas.org's after-salat PDF
(https://www.duas.org/pdfs/Duaaftersalat.pdf), then for the texts and recordings of the missing ones
(docs/research/after-salat-2026-10-10.md: 8 of about 55 were in), then left the design to me.

## What was added

Six texts, one per prayer, in Mafatih al-Jinan's order, holding only what the corpus lacked. Du'as the
corpus already has as texts of their own (the Maghrib/Isha ta'qibat `-1` to `-14`) stay where they are,
so none is added a second time. Why one text per prayer and
not one per item: the items are mostly one or two lines (C1 is 43 letters), many share their words with
longer texts, and reciters read each prayer's ta'qibat as one sequence; the recordings are cut that way.

| Text | Lines | Words | Holds |
|---|---:|---:|---|
| `duasorg-taqibat-common` Common Ta'qibat (after every prayer) | 160 | 753 | istighfar, "ihdini min 'indik", "ashhadu" (10x), "kullama sabbaha", "waj'al li min amri" (3x), "ya rabba Muhammad" (7x), "ya dhal-jalal" (3x), "bismikal maknun", "ya man la yashghaluhu", "subhana man la ya'tadi", "u'idhu nafsi" (3x), "inna maghfirataka arja", "antas-salam" with the salams, "raditu billahi", "al-'afwa" (3x) |
| `duasorg-taqibat-fajr` Ta'qibat after Fajr | 94 | 466 | Mafatih's Subh section ("wahdini", "al-awsiya'" (10x), "ahyini", the 100x statements, "asbahtu allahumma mu'tasiman", "bismillah la hawla" (10x), "ufawwidu amri", "la hawla tawakkaltu"), then the PDF's tahlil and hamazat (10x each), Imam Kazim's "subhanallahil 'azim … astaghfirullah" (10x) and the aqeeq du'a |
| `duasorg-taqibat-zuhr` Ta'qibat after Zuhr | 51 | 282 | "la ilaha illallahul 'azimul halim", "billahi i'tasamtu" (10x), "in 'azumat dhunubi", "rabbas samawatis sab'" (duas.org lists it under Asr; the PDF under Zuhr), Imam Rida's "ataqarrabu" |
| `duasorg-taqibat-asr` Ta'qibat after Asr | 48 | 273 | the istighfar du'a of Misbah al-Mutahajjid ("tawbata 'abdin dhalil"; Mafatih gives it once), "a'udhu bika min nafsin la tashba'", the Asr tasbih ending "ya man khatama" |
| `duasorg-taqibat-maghrib-isha` Short Ta'qibat after Maghrib and Isha | 26 | 180 | the ten short entries of the Maghrib/Isha page the corpus left out (the salawat verse, "yaf'alu ma yasha'" (3x), "ajirna minan nar", "'afina", "ya rabba Muhammad" (3x) and five more) |
| `duasorg-taqibat-sajda-shukr` Sajdah al-Shukr | 58 | 310 | duas.org's sajda page: "shukran" and "'afwan", "ya mudhilla kulli jabbar", "narin harruha la yutfa", the "ushhiduka" and "anshuduka" du'as |

31 of the 45 items the app lacked are in these texts. Left out: six the PDF gives only in transliteration
with no checked Arabic source (F2, F15, F16, S4, S7, Y3-Y4), two found nowhere (Z4 "arzuqna imanan sadiqan",
and "Zayn al-Muttaqin", which the PDF gives only in English), R3 (garbled on duas.org), and four that are mostly
inside texts the app has (F14, Z0, Y1, Y2).

Built by `scripts/taqibat_corpus.py` from duas.org's ta'qibat pages (local copies in
`data/duasorg_extra/taqibat/`), the corpus's own Maghrib/Isha page, and Mafatih for the tahlil and hamazat.
Long lines are cut into pieces of about 7 words, at punctuation where there is any, and a piece
ends on a particle only where nothing else fits.
A repeat count goes on the first line's English ("(Ten times)"); the line is in the text once. Fixed by hand
from the sources: the Zuhr line missing "wa tajma'u baynal mutafarriq", "sallallahu 'alayhi wa alihi" cut short,
"li-wajhi rabbiyal karim" misspelt, four glued words, zero-width joiners and stray punctuation. English and
transliteration per line are in `data/lines/duasorg-taqibat-*.json` (local, like the other `data/lines`).

## Checked against recordings

255 harvest recordings whose titles name ta'qibat or sajda shukr, and duas.org's 46 ta'qibat mp3s, were
labelled again with `harvest_label.py align` on the corpus with the six texts (teacher
wav2vec2-quran-dua-voices, the labeller's own bar). Usable speech in these recordings: 3.12 h -> 5.19 h.

| Text | Recordings with usable spans | Voices | Lines no recording's alignment placed | Popularity (harvest recordings, ≥3 own lines) |
|---|---:|---:|---|---:|
| fajr | 38 | 36 | 17 of 94: lines 13, 45, 75 and 81-94 (the four PDF additions at the end) | 33 |
| zuhr | 21 | 21 | 5 of 51 | 16 |
| maghrib-isha | 19 | 19 | 13 of 26 | 14 |
| asr | 16 | 16 | 2 of 48 | 13 |
| common | 14 | 13 | 2 of 160 | 6 |
| sajda-shukr | 4 | 4 (duas.org's own mp3s) | 42 of 58 | 0 |

The popularity counts (the tracker's prior over du'as) of five texts the app had move by one or two, because
parts of their recordings are now placed in a ta'qibat text: Iftitah 410 -> 409, Maghrib ta'qibat `-1` 26 -> 25,
Ja'far al-Tayyar `-1` 12 -> 11, the Arafah ziyarat 66 -> 64, 'Adeelah 80 -> 82. Sajda shukr has no harvest
recording, so its prior is the floor.

21-42% of each text's words sit in 6-word runs another text also has (fajr 42%, asr 41%, common 38%, zuhr 34%,
maghrib-isha 30%, sajda 21%), mostly 'Asharat. Those stretches can be told apart only by what comes before
and after them.

## The bench

Phone cadence (`DUA_BENCH_ASR_EVERY=2`, `--delay 1.8 --follow-delay 0.1 --display stream --same-text 8`),
today's texts against the same plus the six (`DUA_CORPUS_DIR` on a staged copy, the staged popularity
counts; `data/cache/after_salat/scripts/bench_staged.py`). Runs `taq_base_dev`, `taq_new_dev`, `taq_base_test`,
`taq_new_test`.

| | dev, 522 texts | dev, 528 | test, 522 | test, 528 |
|---|---:|---:|---:|---:|
| on line | 90% | 90% | 90% | 90% |
| right word | 74% | 74% | 76% | 76% |
| jumps / early / lost per 10 min | 1.26 / 1.87 / 1.18 | same | 1.45 / 1.89 / 1.01 | same |
| found ≤10 s* | 93.0% | 93.0% | 94.0% | 94.0% |
| wrong du'a | 0.4% | 0.4% | 0.7% | 0.7% |
| unknown du'a shown | 25.3% | 25.0% | 35.2% | 40.5% |
| unknown du'a shown, without the Fajr ta'qibat item | 25.3% | 25.0% | 37.1% | 37.4% |

On dev, 2 of 1,317 items change at all: one unknown-du'a reading shows a du'a 20 ticks less, one switch
reading has 11 fewer ticks ahead at a stop. On test, 20 of 594 change: the 13 readings of one Munajat al-Taibeen
recording (12 by one tick, its talk reading 7 ticks less lost); one `dua-amaal-quran` recording (flow and
talk) found 2 s later; one Du'a Arafah reading found 2 s sooner; and four unknown-du'a readings. One of those, `ooc-6bd3791603`, is a
recording titled "تعقیب نماز صبح" (Fajr ta'qibat): with the six texts the app shows a du'a for 1,098 of its
1,127 ticks, against 0 before. That item is no longer an unknown du'a and is retired from the bench at install
(`data/testbed/items_retired/taqibat-2026-10-10/`). The other three (Imam Rida's and Lady Khadija's ziyarat,
Imam Zayn al-Abidin's hirz) show a du'a 10-23 ticks more each: +0.3 points on test, against -0.3 on dev.
`bench.py guard` passes every guard bar on dev and test except the unknown-du'a bar on test, which the Fajr
item fails. The gain bars are the jump work's targets (30-40% fewer jumps) and fail because nothing
moved; the user lane's absolute bar (78% on line) fails in the dev baseline too (76.6%). The test voices were looked at in the jump work; nothing here was tuned on them.

Not measured: how well the app follows the six texts themselves. The bench has no items that read them. A lane
like `mafatih` (3 voices a text, those recordings kept out of training) is the way to measure it. It needs a
change to `bench.py`, a GPU pass for the new items and a fresh baseline for every lane.

## Install

Installed 2026-10-10 at about 16:30, after the other session's bench grid on 522 texts had finished, by
`data/cache/after_salat/scripts/install.py`:
- the six texts are in `data/duas` (528 texts, 142,834 words), byte for byte the ones the bench scored;
- the new labels replace the harvest's labels of the 250 relabelled harvest recordings (old ones in
  `data/cache/after_salat/labels_backup/`);
- `dua_popularity.py` reran: 338 of 528 texts recorded, 15,546 recordings, identical to the staged counts;
- `ooc-6bd3791603` was moved to `data/testbed/items_retired/taqibat-2026-10-10/`;
- then `scripts/export_web.py` (`web/corpus.json`, 528 texts) and the tests (175 passed).

Five aparat recordings the harvest never framed have labels only in the cache, and they stay out, as they did on
the bench. `sources.jsonl` is untouched. A rebuild of the ooc sources would no longer take the Fajr recording,
since its span is now labelled as `duasorg-taqibat-fajr`. Baselines at the phone's cadence on the 528 texts are
`taq_new_dev` and `taq_new_test` (the same corpus and counts; `taq_new_test` still holds the retired item, which
comparisons leave out).

Working files: `data/cache/after_salat/` (scripts, staged corpus, labels, bench logs).
