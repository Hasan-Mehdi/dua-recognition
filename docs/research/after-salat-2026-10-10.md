# Du'as after salat: texts and recordings (2026-10-10)

Hasan asked whether the app has the du'as in duas.org's "Dua'a to be recited after every Namaz" PDF
(https://www.duas.org/pdfs/Duaaftersalat.pdf, a 2007 print of duas.org/namaz.htm), then for a deeper look
at each missing one: where its text is, and what recordings exist.

Working files, the scripts that made them, duas.org's pages and mp3s: `data/cache/after_salat/` (local).
What was then added (six texts, one per prayer) and the bench: docs/results/taqibat.md.

## What the app has

About 55 separate du'as in the PDF (dhikr counts, surahs and nafila left out). The PDF gives transliteration
and English only (its Arabic is images), so each item was matched by an Arabic key phrase against the 522 texts.

- In the app as their own text (8), all from duas.org's Maghrib/Isha page: `duasorg-dua-taqeebat-after-maghrib-isha-prayers-1, -2, -3, -5, -6, -10, -12, -14`.
- A longer form only (1): "Asbahtu billahi mu'minan" (Fajr) is in as `-4`, the evening-and-morning form.
- About 10 only as a phrase inside a longer du'a (Asharat, Iftitah, Adeelah, Mashlool, the Five Tasbih, Dua Wahda,
  the opening of Ziyarat Ashura): the tracker would show the longer du'a.
- About 36 not at all.

## Where the texts are

duas.org has four ta'qibat pages beside the Maghrib/Isha one the corpus came from (all with Arabic, transliteration,
English per line and an mp3 for many du'as): `mobile/fajr-taqibat.html`, `mobile/zuhr-asr-taqibat.html`,
`mobile/common-taqibat.html`, `mobile/sajda-shukr.html` (parsed: `mobile_blocks.json`, 122 blocks). The corpus's
own Maghrib/Isha page (`data/duasorg/dua-taqeebat-after-maghrib-isha-prayers.json`) holds 38 blocks; the short ones
(1-4 lines) never went into the corpus. Mafatih al-Jinan (`D:\dua-data\texts\babe1.json`, chapters 1-2: common
ta'qibat, then Zuhr, Asr, Maghrib, Isha, Subh, sajda shukr) and its Baqiyat al-Salihat appendix
(`Molhaghate2.json`: morning and evening, Maghrib) have most of the rest.

Of the 45 missing items: 31 have a duas.org text, 9 more are in Mafatih or were typed from the PDF's transliteration
(these need a checked source before use), and 2 were found nowhere: the "Zayn al-Muttaqin" du'a (the PDF gives only
English) and "Allahummarzuqna imanan sadiqan" (Zuhr). Per item: `items.json` (`src` = duas.org block or "typed").

## Recordings

Method: the harvest has 255 recordings on disk whose titles name ta'qibat, sajda shukr or Tasbih al-Zahra (16.5 h;
shiavoice's "أدعية الصلوات" category is 44 of its 47 tracks). Only the parts whose text the corpus or
`extra_texts.json` had were ever labelled. They and duas.org's 46 ta'qibat mp3s went through the harvest's teacher
(wav2vec2-quran-dua-voices, 17.9 h in 2 min on the GPU); then each item's lines were found in each recording's
greedy letters (Myers, at most 0.35 edits per letter) and kept when CTC forced alignment scored at most 1.5 per
letter (the labeller's own bar). A recording counts for an item when half its lines (at least 2) are found within
a plausible span. Check: the 8 texts the app has came out at counts close to the labeller's (Maghrib "mujibati"
30 recordings here, 28 in the labels; Isha "laysa li 'ilmun" 21 and 23). Voices = distinct uploaders/reciters.

| Item | Text | Voices in the harvest | duas.org mp3 |
|---|---|---|---|
| F1 Bismillahi wa sallallahu… ufawwidu amri (Fajr) | duas.org Fajr | 21 (53 min) | yes |
| M2 Innallaha wa mala'ikatahu… salawat (Maghrib) | duas.org Maghrib/Isha | 20 | in the Maghrib mp3 |
| Z1 La ilaha illallahul 'azimul halim (Zuhr) | duas.org Zuhr | 17 (31 min) | yes |
| R1 Astaghfirullah… tawbata 'abdin dhalil (Asr) | duas.org Asr | 16 | yes |
| M3 Alhamdulillahil ladhi yaf'alu ma yasha' (Maghrib) | duas.org Maghrib/Isha | 15 | |
| R2 A'udhu bika min nafsin la tashba' (Asr) | duas.org Asr | 13 | in the Asr mp3 |
| F19 Wahdini limakhtulifa fihi (Fajr) | duas.org Fajr | 10 | |
| F20 Al-awsiya' al-radin al-mardiyyin (Fajr) | duas.org Fajr | 8 | yes |
| Z3 In 'azumat dhunubi (Zuhr) | duas.org Zuhr | 5 | yes |
| F6 Ashhadu… ahadan samadan | duas.org Common | 4 | |
| F9 Tahlil, yuhyi wa yumit wa yumitu wa yuhyi | typed | 4 | yes |
| Z2 Billahi i'tasamtu (Zuhr) | duas.org Zuhr | 4 | |
| C1 Astaghfirullah alladhi la ilaha illa huwa | duas.org Common | 3 (+2 that are R1's opening) | yes |
| F5 Allahumma ihdini min 'indik | duas.org Common | 3 | yes |
| F8 La hawla… tawakkaltu 'alal hayy | duas.org Fajr | 3 | yes |
| Z0 Before Zuhr: Subhanallah… lam yattakhidh sahibatan | duas.org Zuhr | 3 | yes |
| E1 Radhitu billahi rabba | duas.org Common | 3 | |
| C2 Bismikal maknun (after Ikhlas 12 times) | duas.org Common | 2; YouTube has 6-8 uploads of it alone | yes |
| F15 Allahu akbar kabira… bukratan wa asila | typed | 2 | |
| F7 U'idhu nafsi… wa ma razaqani rabbi | duas.org Common | 1 | yes |
| F16 Asbahtu wa rabbi mahmud | typed | 1 | |
| Y1 Salam on the Prophet after prayer | typed (Mafatih: ziyarat from afar) | 1 | |
| Z5 Allahumma rabbas samawatis sab' | duas.org (as Asr) | 0 | yes |
| Z6 Ataqarrabu ilayka bijudika wa karamik | duas.org Zuhr | 0 | yes |
| R4 Subhanallah… dhil mulk… ya man khatama (Asr) | duas.org Asr | 0 | yes |
| S5 Ya mudhilla kulli jabbar; S6 narin harruha la yutfa; S8 shukran/'afwan | duas.org Sajda | 0 | yes |
| F2, F10, F13, F17, F18, R3, M4, I2, S1, S3, S4, S7, Y2-Y4, Z4, F4 | see `items.json` | 0 | no |

Recordings follow Mafatih's order for each prayer, not the PDF's: Fajr F19 > F20 > Imam Sadiq's du'a > F1;
Zuhr Z1 > Z2 > Z3; Asr R1 > R2; Maghrib M2 > M3 > "mujibati rahmatik"; common C1 > F5 > F8 > Z0 > F6 > C2 >
"ilahi hadhihi salati" > F7 > E1 (`audio_lines.json`, sequences per recording).

## More audio

- 603 matching titles were found by the harvest but never downloaded (103 h, mostly YouTube; `taqibat_recordings.json`,
  kind `candidates`): about 325 ta'qibat recitations (241 Arabic, 48 Persian, 34 Urdu), 102 Tasbih al-Zahra,
  16 sajda; the rest lectures and news ("تعقيب" also means commentary).
- Metadata-only YouTube searches for the items with no audio (`yt_search.json`): Urdu ta'qibat series per prayer
  (Thilawe Digital: Subh, Asr, Maghrib, Isha; mosaacademy; duas.org's own channel, Fajr), "ziyarat after namaz"
  clips for the short salams (Rah e Babul Ilm, NIJAT TV, Thilawe Digital), one "ya mudhilla kulli jabbar"
  (Sayyid Walid al-Muzaidi), one Urdu du'a with "ghalabat al-dayn". For Z4, M4, I2, F10, F13, F17, R3, R4, Z5 and
  Z6 the searches returned only unrelated recordings.

## What is needed

The bench takes an added text in with 3 voices or more (`bench.MAFATIH_MIN`, as for the Mafatih texts).
17 items have that on disk now. C2 would with the YouTube uploads; F7, F15, F16, Y1 need one or two more.
The rest have one duas.org recording or none: the Urdu-tradition items (Z4, M4, I2, S4, S7, Y2-Y4) are rare in the
Arabic and Persian uploads the harvest is made of, so their voices would come from the Urdu series above or from
new recordings.

Many items are one or two lines (C1 43 letters, Z2 33, M3 39) and share their words with longer texts (C1 opens R1,
F19 is in Iftitah, F9 in Dua Wahda), so as separate texts they will be hard to tell apart. Reciters read each
prayer's ta'qibat as one sequence, and that is how the recordings are cut.
