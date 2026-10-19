# Text audit: where our Arabic differs from what reciters say (2026-10-08)

In Hasan's session 20261008-020516-aq31 (Ziyarat Waritha) the word follower jumped ahead and
then rushed five words in 0.2 s. Line 32 of `ziyarat-warith` reads "bi-abi anta wa ummi ya bna
rasulillah, ya aba Abdillah"; the standard text, and the line's own transliteration and English,
say "bi-abi anta wa ummi" twice. Our Arabic (from DuaPlayer) had dropped the second one, so the
follower had to explain it as the line starting again. A replay with the words put back followed
him word for word. This audit looks for the same kind of error in all 522 texts and proposes
fixes. Nothing in `data/duas/` is changed: the edits are in `data/cache/text_audit/fixes.json`
and `scripts/text_fixes.py` applies them.

## Summary

293 line edits in 123 texts: 64 sure, 195 likely, 34 for Hasan. By kind:

| kind | texts with 20+ recordings | texts with fewer |
|---|---|---|
| wording: a phrase or word missing or extra, a repeat collapsed, a wrong or misspelt word, a duplicated line | 66 (25 sure, 36 likely, 5 Hasan) | 3 likely |
| Dua Nudbah: an alternative reading kept in parentheses beside the recited one | 17 sure | |
| duas.org texts: Mafatih's alternative readings kept in square brackets beside the main one | 10 (8 likely, 2 Hasan) | 165 (138 likely, 27 Hasan) |
| typing: words glued or cut across lines, stray letters, abbreviations, an instruction left in the Arabic | 3 sure | 29 (19 sure, 10 likely) |

The ones that matter most, by how much the text is recited (harvest recordings per text,
`data/dua_popularity.json`):

- **Ziyarat Ashura** (580): line 5 has an extra "al-Zahra", 30 an extra "wa tahayya'at" (Ziyarat
  Warith's word), 38 an extra "ta'ala", 45 an extra "wa ila rasulih"; line 41 lacks the 17 words
  "mimman qatalaka ... wa abra'u ila Allah wa ila rasulih"; "al-salamu alayka ya khiyarat Allah wa
  ibna khiyaratih" is missing before line 3; lines 23-24 hold "wa waliyyun liman walakum wa
  aduwwun liman adakum", which belongs to a later passage. In the harvest, lines 5, 23, 24, 30,
  38 and 45 pass the labeller's alignment check in 0-8% of their placements (line 38: 0 of 357),
  against 90% for the du'a; line 41 in 58%, with 8-12 s of unexplained recitation either side;
  line 3's missing sentence shows as a 7 s gap before it.
- **Ziyarat Waritha** (477): line 32 (the session's line), 59 (extra "al-Zahra"), 35 (extra "wa
  tanaqqabat", Ashura's word), 7 ("waliyy Allah" where most reciters say "alayhi al-salam").
- **Dua Tawassul** (443): line 7 has an extra "ya shafi' al-ummah" (passes in 1 of 466 harvest
  placements); line 60 says "ya aba Ibrahim" where duas.org, Mafatih and 39 of 40 probe clips
  say "ya aba al-Hasan" (the line passes in 0 of 372 placements).
- **Dua Nudbah** (440): 16 lines carry an alternative reading in parentheses that nobody recites
  ("wa nawasha (nahasha)", "al-mahdiyyin (al-muhtadin)" ...), and line 200 lacks "khalaqtahu lana
  ismatan wa maladhan".
- **Dua Iftitah** (410): line 27 repeats "wa hamduhu" from the end of line 26.
- **Abu Hamza** (369): line 65 ends the quotation with "bi-kulli shay'in alima" where reciters say
  "bikum rahima"; 349 has a garbled "an alladhi lqunut"; 389 lacks "bil-taswifi wal-amali"; four
  lines say "wa ala alihi" where reciters say "wa alihi".
- **Hadith al-Kisa** (273), **Makarim al-Akhlaq** (154, line 133 repeats line 132 instead of "wa
  hab li amna yawm al-ma'ad"), **Mashlool** (135, a truncated line, a misspelling, three missing
  words or phrases), **Jawshan Saghir** (124, section openers and two phrases), **Mujeer**,
  **Simaat**, **Munajat Sha'baniyya**, **Yastasheer**, **Muslim ibn Aqil**, the weekday du'as.

The scenario bench barely sees these errors: its harvest lane keeps only runs of lines the
labeller aligned confidently, which a wrong line rarely is, and its studio lane is DuaPlayer's
reciters, who recite DuaPlayer's text (Abu Thar al-Halawaji does say Ashura's "wa tahayya'at").

## Method

Four kinds of evidence, each line judged with all of them in front of it.

1. **The line's own reading and English.** `web/corpus.json`'s reading is not evidence: for every
   line with enough vowel marks it is generated from our own Arabic (`translit.for_display`), and
   the 17 `mafatih-*` texts have no other. The audit uses the sources' own readings instead
   (`data/duaplayer/<id>/slides.json` for the 91 DuaPlayer texts, `data/lines/` for duas.org and
   duas.pro, typed on the same page as the Arabic but separately): 20,271 of 23,484 lines. Our
   Arabic, transliterated by rule, and the source reading were reduced to consonants and aligned;
   a run of 5+ consonants on one side only flagged the line (101 lines; most turned out to be
   readings that are themselves wrong or a row out of step, as in Sahifa Fatimiya's Hareeq).
2. **Second copies.** Every text was aligned word by word against the decoded Mafatih al-Jinan
   (`D:\dua-data\texts`, 1,325 blocks), every duas.org page (`data/duasorg/`) other than the one a
   `duasorg-*` text came from, and the other corpus texts, keeping copies that cover most of
   each other: 168 texts have one (Mafatih 111, duas.org 92, another corpus text 12).
3. **The harvest labels.** 26,398 label files (`data/harvest/labels/`) place 1.7 M lines. For each
   corpus line: how often it passes the forced-alignment check (cost <= 1.5 nats a letter) among
   recordings whose neighbouring lines both pass, against the du'a's median; the mean cost; which
   gap between two words is longest (a missing phrase shows as a 2-3 s gap at the same word in
   most recordings); and the time between consecutive lines (a missing sentence shows as a long
   gap between two lines). 9,951 lines have 10+ such placements. A line that fails where its
   neighbours pass, in many voices, is the strongest single sign.
4. **What reciters say (the probe).** The CTC frames behind the labels are gone, so for each line
   in question the probe cut 16 harvest clips from different uploaders (30-40 to confirm), from
   the previous line's last word to the next line's first word, ran the labelling teacher
   (`wav2vec2-quran-dua-voices`) over them and compared the CTC loss of our line with each
   candidate edit (from the second copies, the readings, or what the clips said). "Probe 15/16"
   below means the edit had the lower loss in 15 of 16 clips; "gain" is the median loss
   difference. The teacher's greedy text of each clip was read for the lines with no candidate.
   1,172 probes, 15,489 clips (one GPU job at a time, about an hour of GPU in all).

The probes covered every flagged line of the 129 texts with 20+ recordings (of the Jawshan
Kabir refrain, 3 of its 100 repeats) and of the 113 with 5-19 (239 probes, fewer clips). The 280 texts with under 5 recordings got the text checks only.

Confidence: **sure** = two independent kinds of evidence agree (a second copy or the line's own
reading, and reciters), or an unambiguous typo; **likely** = reciters clearly prefer it (two
thirds or more of the probe clips) with one source behind it, or a source convention applied
where no recording exists; **needs Hasan** = reciters split, the edit empties a line, or the
source keeps both readings.

Harvest numbers in the tables: "ok a/b (c)" = the line passes in a of b placements whose
neighbours pass, c the du'a's median rate; "harvest p/q" in the impact column = placements and
passes of the line over all labels.

## Fixes, by text

Each row is one entry of `fixes.json` (it holds the full old and new Arabic, in the file's own
spelling). "change" is the word-level difference as the matcher normalises it. Impact: the
line's word count before and after (a changed count drops every bench source that reads the
line, see below), the bench sources that read it (`data/testbed/sources.jsonl`, by lane), its
harvest placements/passes, and human-timed recordings covering it (DuaPlayer, duas.pro,
duas.org, test sets).

**Dua Kumayl** (`dua-kumayl`, 794 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 206 | البارزين → المبادرين | needs Hasan | Ours (and Mafatih) 'fi al-barizin'; duas.org 'fi al-mubadirin'. Harvest ok 298/504 (0.89), cost 1.40. Probe: 'al-mubadirin' 9/16 and 19/30, so the reciters split about 60/40. | 4→4; bench: 4 harvest; harvest 682/401; timed: 4 duaplayer/duas.pro/duas.org, 5 majlis |

**Ziyarat Ashura** (`ziyarat-ashura`, 580 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 3 | +السلام عليك يا خيره الله وابن خيرته | likely | duas.org and Mafatih have 'al-salamu alayka ya khiyarat Allah wa ibna khiyaratih' before this line. Harvest: median 7.2 s between lines 2 and 3, over 3 s in 61% of 417 recordings, and the gap clips hold this sentence. Probe 11/16 and 19/30 (a third of reciters skip it). In the DuaPlayer and duas.org human timings the sentence falls inside line 3's span, so it is prepended to 3. | 6→13; harvest 582/477; timed: 3 duaplayer/duas.pro/duas.org |
| 5 | −الزهراء | sure | 'al-Zahra' is not in the line's tl, duas.org or Mafatih. Harvest ok 20/293 (0.90), cost 1.95. Probe 15/16, 39/40. | 9→8; harvest 566/32; timed: 3 duaplayer/duas.pro/duas.org |
| 23 | −وولي لمن والاكم | needs Hasan | Lines 22-24 read 'wa harbun liman harabakum, wa waliyyun liman walakum, wa aduwwun liman adakum ila yawm al-qiyamah'; duas.org and Mafatih have here only 'wa harbun liman harabakum ila yawm al-qiyamah' (the longer form comes later, in 'inni silmun liman salamakum'). Harvest: lines 23 and 24 pass in 7/337 and 8/352 placements. Probe over 22-24 (clip from 21 to 25): 15/16 prefer the short form, median gain 224 nats. Needs Hasan: it empties segment 23 (with 24). | 3→0; bench: 3 harvest; harvest 337/7; timed: 2 duaplayer/duas.pro/duas.org, 1 majlis |
| 24 | −وعدو لمن عاداكم | needs Hasan | See line 23: with 23 emptied, 24 keeps only 'ila yawm al-qiyamah'. | 6→3; bench: 3 harvest; harvest 352/8; timed: 2 duaplayer/duas.pro/duas.org, 1 majlis |
| 30 | −وتهيات | sure | 'wa tahayya'at' belongs to Ziyarat Warith; not in the tl, duas.org or Mafatih. Harvest ok 20/236 (0.90), cost 1.82. Probe 16/16, 40/40. (DuaPlayer's own reciter Abu Thar al-Halawaji does say it.) | 8→7; harvest 508/21; timed: 3 duaplayer/duas.pro/duas.org, 1 majlis |
| 38 | −تعالي | sure | 'ta'ala' is not in the tl, duas.org or Mafatih. Harvest ok 0/357 (0.90), cost 2.02. Probe 16/16. | 7→6; harvest 519/0; timed: 3 duaplayer/duas.pro/duas.org, 1 majlis |
| 41 | +وبالبراءه ممن قاتلك ونصب لك الحرب; +الظلم والجور عليكم وابرا الي الله والي رسوله ممن اسس اساس | sure | duas.org and Mafatih have the 17 words 'mimman qatalaka ... wa abra'u ila Allah wa ila rasulih' (Mafatih in brackets). Harvest: ok 187/325 (0.90), cost 1.45; median gaps of 7.9 s before and 11.5 s after the line, and the gap clips hold exactly these words. Probe (30 clips) 24/30, median gain 231 nats. | 5→22; harvest 529/283; timed: 3 duaplayer/duas.pro/duas.org, 1 majlis |
| 45 | −والي رسوله | sure | duas.org and Mafatih: 'wa ataqarrabu ila Allah thumma ilaykum'. Harvest: the line passes in 9 of 526 placements. Probe over lines 44-46 (clip from line 43 to 47): 16/16 without 'wa ila rasulih'; with only clips where 44 and 46 already passed, 7/15 (some reciters do say it). | 5→3; harvest 526/9; timed: 3 duaplayer/duas.pro/duas.org, 1 majlis |

**Ziyarat Waritha** (`ziyarat-warith`, 477 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 7 | ولي الله → عليه السلام | likely | Mafatih: '... amir al-mu'minin alayhi al-salam'; duas.org: 'alayhi al-salam [waliyy Allah]'; own tl has both. Probe (30 clips): 'alayhi al-salam' instead of 'waliyy Allah' 19/30, both 5/30. Harvest ok 233/277 (0.92), cost 1.01. | 8→8; bench: 8 harvest; harvest 448/322; timed: 1 user, 6 majlis |
| 32 | +بابي انت و امي | sure | The second 'bi-abi anta wa ummi' is missing. Own tl and en have it twice; duas.org (both Warith pages) and Mafatih have it. Harvest: ok 46/184 (du'a 0.92), cost 1.66, a 2.7 s mean gap after 'rasulillah' (longest gap of the line in 58% of recordings). Probe: 15/16 and 38/40 clips. Hasan's session 20261008-020516-aq31. | 12→16; bench: 2 harvest; harvest 400/53; timed: 5 majlis |
| 35 | −و تنقبت | likely | 'wa tanaqqabat' belongs to Ziyarat Ashura; Warith's own tl/en, duas.org and Mafatih have 'wa tahayya'at li-qitalik' only. Harvest ok 53/193 (0.92), cost 1.89. Probe 10/16 and 19/30: about a third of reciters do add it. | 10→8; bench: 2 harvest; harvest 342/64; timed: 1 majlis |
| 59 | −الزهراء | sure | 'al-Zahra' is not in the line's tl or en, duas.org or Mafatih. Harvest ok 28/86 (0.92), cost 1.70. Probe 11/16, 29/40. | 9→8; bench: 2 harvest; harvest 124/32; timed: 6 majlis |

**Dua Tawassul** (`dua-tawassul`, 443 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 7 | −يا شفيع الامه | sure | 'ya shafi' al-ummah' is not in duas.org or Mafatih (own tl/en have it). Harvest ok 1/26 (0.87), the line passes in 1 of 466 placements, cost 3.13. Probe 16/16, 26/27. | 6→3; harvest 466/1; timed: 3 duaplayer/duas.pro/duas.org |
| 60 | ابراهيم → الحسن | sure | duas.org and Mafatih: 'ya aba al-Hasan, ya Musa bn Ja'far' (ours 'ya aba Ibrahim'). Harvest ok 0/114, the line passes in 0 of 372 placements, cost 2.43. Probe 16/16, 39/40. | 7→7; bench: 1 studio; harvest 372/0; timed: 3 duaplayer/duas.pro/duas.org |

**Dua Nudbah** (`dua-nudbah`, 440 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 200 | −الشايقون; +خلقته لنا عصمه و ملاذا | sure | '( al-sha'iqun )' is an alternative reading, and 'khalaqtahu lana ismatan wa maladhan' (duas.org, Mafatih) is missing before line 201. Probe: without the parenthesis 15/16 (gain 70 nats); with the phrase 16/16 (gain 34 nats). | 12→16; harvest 408/5; timed: 2 duaplayer/duas.pro/duas.org |

**Dua Iftitah** (`dua-iftitah`, 410 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 27 | −وحمده | sure | Line 26 already ends with 'wa hamduhu'; duas.org has it once. Probe 16/16, median gain 33 nats. | 4→3; bench: 1 studio; harvest 372/17; timed: 3 duaplayer/duas.pro/duas.org, 7 majlis |
| 124 | −الزهراء | needs Hasan | duas.org has no 'al-Zahra'. Probe: 9/16 and 18/30 without it, so about 40% of reciters say it. | 5→4; bench: 2 harvest; harvest 349/107; timed: 4 duaplayer/duas.pro/duas.org, 7 majlis |

**Dua Abu Hamza Thumali** (`dua-abu-hamza-thumali`, 369 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 65 | بكل شيء عليما → بكم رحيما | sure | Ours ends the quotation with Quran 4:32's 'bi-kulli shay'in alima'; Mafatih's du'a has 'kana bikum rahima'. Probe 15/16, median gain 56 nats. | 6→5; harvest 222/3; timed: 2 duaplayer/duas.pro/duas.org |
| 237 | وعلي اله → واله | sure | Mafatih and reciters: 'salla Allahu alayhi wa alihi' (ours 'wa ala alihi'). Probe 16/16, gain 16 nats. | 5→4; harvest 217/148; timed: 2 duaplayer/duas.pro/duas.org |
| 349 | الذي لقنوط → القنوط | sure | 'an alladhi lqunut' is a typo for 'an al-qunut' (Mafatih). Probe 16/16, gain 30 nats. | 8→7; harvest 220/116; timed: 2 duaplayer/duas.pro/duas.org |
| 387 | وعلي اله → واله | sure | Mafatih and reciters: 'salla Allahu alayhi wa alihi' (ours 'wa ala alihi'). Probe 16/16, gain 19 nats. | 6→5; harvest 219/43; timed: 2 duaplayer/duas.pro/duas.org |
| 389 | +بالتسويف والامال | sure | Mafatih: 'faqad afnaytu bil-taswifi wal-amali umri'. Probe 16/16, gain 67 nats; the harvest gap after the line (4.8 s median) holds 'bil-taswifi wal-amali umri'. | 7→9; harvest 221/105; timed: 2 duaplayer/duas.pro/duas.org |
| 554 | وعلي اله → واله | sure | Mafatih and reciters: 'salla Allahu alayhi wa alihi' (ours 'wa ala alihi'). Probe 16/16, gain 28 nats. | 9→8; harvest 209/155; timed: 2 duaplayer/duas.pro/duas.org |
| 624 | وعلي اله → واله | sure | Mafatih and reciters: 'salla Allahu alayhi wa alihi' (ours 'wa ala alihi'). Probe 16/16, gain 18 nats. | 5→4; harvest 204/112; timed: 2 duaplayer/duas.pro/duas.org |

**Dua by Holy Prophet(sawa) - 15th Sha'aban** (`duasorg-15-shaban-dua-by-holy-prophet`, 316 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 11 | +بذنوبنا; +يخافك فينا ولا | likely | Reciters: 'wa la tusallit alayna bi-dhunubina man la yakhafuka fina wa la yarhamuna'. Probe 15/20, gain 70 nats. | 6→10; bench: 8 harvest; harvest 1042/334 |

**Hadith Kisa** (`dua-hadithkisa`, 273 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 3 | +عن جابر بن عبد الله الانصاري; +قال سمعت فاطمه | likely | Mafatih has 'qala sami'tu Fatimata annaha qalat'; reciters open with 'an Jabir ibn Abd Allah al-Ansari'. Harvest: median gaps of 6.4 s before line 3 (96% over 3 s) and 5.2 s after it, holding these words. Probe over lines 3-4: 24/29 prefer this (with line 4 below); with Jabir beats without in 23/29, 'qala sami'tu' beats without in 23/29. | 14→23; harvest 279/95; timed: 3 duaplayer/duas.pro/duas.org |
| 4 | −صلي الله عليه واله | likely | Mafatih has no salawat here. Harvest ok 12/74 (0.91), cost 3.35. Probe 14/16 (and 24/29 with line 3 above). Elsewhere in the text reciters do say the salawat (lines 17, 40, 96, 104: 1/16-2/16 drop it). | 13→9; harvest 265/49; timed: 3 duaplayer/duas.pro/duas.org |
| 13 | −عليه السلام | likely | Mafatih has no 'alayhi al-salam' here (nor does our line 24 for al-Husayn). Harvest ok 29/104 (0.91), cost 2.34. Probe 10/16, 25/30. | 11→9; harvest 265/32; timed: 3 duaplayer/duas.pro/duas.org |

**Dua Mujeer** (`dua-mujeer`, 226 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 216 | واجد → واحد | sure | 'ya wajid' (jim) for 'ya wahid' (ha), after 'subhanaka ya majid'; Mafatih 'ya wahid'. Probe 16/16. | 3→3; bench: 1 studio; harvest 162/80; timed: 2 duaplayer/duas.pro/duas.org, 5 majlis |

**Dua Simaat** (`dua-simaat`, 224 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 145 | −يعلم | likely | Mafatih: '... wa la batinaha wa la zahiraha ghayruk' (no second 'ya'lamu' here). Probe 16/16 and, over lines 144-145, 26/26 (adding 'ya'lamu' to line 144 instead: 12/26, no change). | 6→5; harvest 88/1 |

**Munajat Shabaniyah** (`duasorg-munajat-shabaniyah`, 217 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 42 | −الهي قد احسنت الي | likely | Mafatih continues line 41 with 'idh lam tuzhirha ...'; 'ilahi qad ahsanta ilayya' is not there. Probe 13/16, gain 102 nats. | 11→7; harvest 196/38 |

**Ramadan Day 3 · Yā man tuḥallu bihi…** (`duasorg-ramadan-day-03-2`, 176 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 14 | +منها | likely | Reciters: 'wa la yankashifu minha illa ma kashafta'. Probe 30/30. | 6→7; harvest 177/45 |
| 33 | +رحمه و | likely | Reciters: 'wa hab li min ladunka rahmatan wa farajan hani'an'. Probe 30/30, gain 29 nats. | 7→9; harvest 173/3 |

**Dua Makaram al-Akhlaq** (`dua-makaramakhlaq`, 154 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 14 | بالبطر → بالنظر | likely | Mafatih: 'bil-nazar' (ours 'bil-batar'). Probe 12/16, gain 16 nats. | 7→7; bench: 2 harvest; harvest 143/131; timed: 2 duaplayer/duas.pro/duas.org, 1 majlis |
| 133 | وقبل الضلال بالرشاد واكفني موونه معره العباد → وهب لي امن يوم المعاد وامنحني حسن الارشاد | sure | Line 133 repeats line 132 word for word (tl too); Mafatih (Sahifa 20) has 'wa hab li amna yawm al-ma'ad, wa-mnahni husna al-irshad' here. Harvest: the line passes in 0 of 139 placements, cost 5.51. Probe 16/16, gain 186 nats. | 7→8; harvest 139/0; timed: 2 duaplayer/duas.pro/duas.org, 1 majlis |

**Dua - repentance** (`duasorg-eid-e-mubahila-3`, 141 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 17 | −و اعتقدت | likely | Mafatih's alternative reading in square brackets; the line's own transliteration (duas.org) reads without it. No harvest recordings reach this part of the text. | 26→24 |
| 21 | −متعال | likely | Mafatih's alternative reading in square brackets; the line's own transliteration (duas.org) reads without it. No harvest recordings reach this part of the text. | 28→27 |
| 24 | −جوارك و | likely | Mafatih's alternative reading in square brackets; the line's own transliteration (duas.org) reads without it. No harvest recordings reach this part of the text. | 20→18 |
| 43 | −بعطايك | likely | Mafatih's alternative reading in square brackets; the line's own transliteration (duas.org) reads without it. No harvest recordings reach this part of the text. | 14→13 |
| 48 | −تخيلي | likely | Mafatih's alternative reading in square brackets; the line's own transliteration (duas.org) reads without it. No harvest recordings reach this part of the text. | 28→27 |
| 92 | −من | likely | Mafatih's alternative reading in square brackets; the line's own transliteration (duas.org) reads without it. No harvest recordings reach this part of the text. | 17→16 |
| 94 | ق → قياس فارفعني في عليين | sure | Line 94 ends 'qi' and 95 starts 'yasun farfa'ni fi illiyyinaya qabidu': 'qiyasun' is cut in two and 'illiyyin' glued to 'ya qabidu'. The 'ya rafi'' passage goes to 94, 'ya qabidu' starts 95 as every other passage does. | 14→17 |
| 95 | ياس فارفعني في عليينيا → يا | sure | See line 94. The two bracketed words stay (without 'wa la taqbid' the sentence has no verb): needs Hasan's call separately. | 26→23 |
| 102 | −شبه | likely | 'shbh [shabihun]': the line's transliteration reads 'laysa laka shabihun'. | 23→22 |
| 108 | −ذا الفضل | likely | Mafatih's alternative reading in square brackets; the line's own transliteration (duas.org) reads without it. No harvest recordings reach this part of the text. | 5→3; harvest 1/0 |

**Dua Mashlool** (`dua-mashlool`, 135 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 5 | +هو ولا اين | sure | Mafatih: 'wa la kayfa huwa wa la ayna huwa wa la haythu huwa'. Probe 30/30, gain 20 nats. | 8→11; harvest 138/29; timed: 2 duaplayer/duas.pro/duas.org |
| 44 | +يا ظهير | likely | Mafatih: 'ya basiru ya zahiru ya kabiru'. Probe 13/16, gain 21 nats. | 8→10; bench: 1 harvest; harvest 124/81; timed: 2 duaplayer/duas.pro/duas.org |
| 56 | +يا رحمن | sure | Mafatih: 'ya rahimu ya rahmanu ya man huwa kulla yawmin fi sha'n'. Probe 16/16, gain 15 nats. | 9→11; bench: 2 harvest; harvest 123/101; timed: 2 duaplayer/duas.pro/duas.org |
| 125 | +غروبها | sure | The line stops at 'ba'da'; Mafatih: 'ba'da ghurubiha'. Probe 16/16; the harvest gap after the line holds 'ghurubiha'. | 6→7; bench: 2 harvest; harvest 122/110; timed: 2 duaplayer/duas.pro/duas.org |
| 158 | تقظوا → تقنطوا | sure | 'la taqzawu' is a typo for 'la taqnatu' (Quran 39:53, Mafatih). Probe 15/16. | 10→10; bench: 2 harvest; harvest 120/103; timed: 2 duaplayer/duas.pro/duas.org |

**Dua Yastasheer** (`duasorg-dua-yastasheer`, 129 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 16 | والارض → وما في الارض | sure | Mafatih (Quran 20:6): 'lahu ma fi al-samawati wa ma fi al-ard'. Probe 15/16. | 5→7; bench: 3 harvest; harvest 129/107 |

**Dua Jawshan Sagheer** (`dua-jawshan-sagheer`, 124 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 115 | وابيت الا → والا | likely | Mafatih: 'abayta illa in'aman wa-mtinanan, wa illa tatawwulan ya rabbi wa ihsanan' (no second 'abayta'). Probe 16/16, 40/40. | 6→5; harvest 116/9 |
| 261 | −ومولاي | likely | Our text gives every section the same opener 'ilahi wa mawlaya wa sayyidi'; Mafatih varies it, and reciters follow Mafatih. Here 'ilahi wa sayyidi': probe 16/16, gain 51 nats. | 3→2; harvest 103/1 |
| 351 | الهي ومولاي → مولاي | likely | Mafatih: 'mawlaya wa sayyidi'. Probe 16/16, gain 34 nats. | 3→2; harvest 101/1 |
| 373 | الهي → سيدي; −وسيدي | likely | Mafatih: 'sayyidi wa mawlaya'. Probe 15/16, gain 23 nats. | 3→2; harvest 90/1 |
| 398 | −ومولاي | likely | Mafatih: 'ilahi wa sayyidi'. Too few harvest placements to probe (10); same pattern as 261/351/373. | 3→2; harvest 10/0 |
| 418 | −ومولاي | likely | Mafatih: 'ilahi wa sayyidi'. Too few harvest placements to probe (3); same pattern as 261/351/373. | 3→2; harvest 3/1 |
| 434 | لنعمايك → لك من العابدين ولنعمايك | likely | Probe (13 clips): 12/13 say 'wa-j'alni laka min al-abidina wa li-na'ma'ika min al-shakirin', gain 48 nats. | 4→7; harvest 27/1 |

**Izn Dukhool - Permission to Enter the shrine** (`duasorg-ziyarat-muslim-ibn-aqeel-hani-e-urwa-3`, 121 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 10 | +واشهد الله; عليه → به | likely | Reciters: 'ashhadu wa ushhidu Allaha annaka madayta ala ma mada bihi al-badriyyun'. Probe 28/30, gain 49 nats (only adding 'wa ushhidu Allah': gain 26). | 8→10; harvest 124/36 |
| 16 | +الجزاء واوفي | likely | Reciters: 'wa awfara al-jaza'i wa awfa jaza'a ahadin'; the harvest gap before the line (3.4 s median) holds 'wa awfara al-jaza'. Probe 27/30, gain 33 nats. | 6→8; bench: 3 harvest; harvest 125/104 |

**Ziyarat of Muslim ibn Aqeel** (`duasorg-ziyarat-muslim-ibn-aqeel-hani-e-urwa-1`, 117 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 38 | +صلوات الله عليهم | likely | Reciters: '... wa an al-Hasan wa al-Husayn salawatu Allahi alayhim afdala al-jaza'; the harvest gap after the line (3.7 s median) holds it. Probe 25/30, gain 49 nats. | 3→6; bench: 2 harvest; harvest 94/86 |

**Sahifa 54: Removal of worries** (`sahifa-54`, 114 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 18 | +بسم الله الرحمن الرحيم | likely | Reciters say the bismillah before each of the three suras; the harvest gaps before lines 18, 19 and 20 (median 6-34 s, some reciters repeat the sura) open with it. Probe 30/30, gain 70 nats. | 15→19; harvest 338/231 |
| 19 | +بسم الله الرحمن الرحيم | likely | As line 18. Probe 25/30, gain 71 nats. | 23→27; harvest 413/318 |
| 20 | +بسم الله الرحمن الرحيم | likely | As line 18; the gap clips before the line open with the bismillah (no clip passes the span test: it is the text's last line). | 20→24; harvest 425/327 |

**Dua Tawba** (`dua-tawba`, 102 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 38 | يستعصم → يستعظم; −عنه | likely | 'yasta'simu' (sad) for 'yasta'zimu' (za', 'consider great', as the line's own English has it). Harvest ok 27/88 (0.93), cost 1.59. Probe (40 clips) 40/40 for 'yasta'zimu' without the first 'anhu', 35/40 with it. | 8→7; bench: 1 studio, 1 harvest; harvest 99/29; timed: 1 duaplayer/duas.pro/duas.org, 1 majlis |
| 137 | وال محمد → واله | likely | Reciters (and the Sahifa): 'salli ala Muhammadin wa alihi kama hadaytana bihi'. Probe 28/30, gain 32 nats. | 9→8; harvest 103/40; timed: 1 duaplayer/duas.pro/duas.org, 1 majlis |

**Dua al-Ithneen** (`dua-monday`, 84 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 8 | وانحسرت العقول → والعقول | likely | Mafatih: 'kallat al-alsunu an ghayati sifatihi, wa al-uqulu an kunhi ma'rifatihi' (no second verb). Harvest ok 9/61 (0.92), cost 3.42. Probe 26/30, gain 52 nats. | 5→4; harvest 86/10 |
| 25 | −لك | likely | Mafatih: 'thumma lam afi bihi'. Probe 24/30, gain 13 nats. | 5→4; harvest 72/6 |

**Dua al-Khamis** (`dua-thursday`, 83 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 4 | واتاني → وانا في | likely | Mafatih: 'wa kasani diya'ahu wa ana fi ni'matihi'. Probe 27/30, gain 26 nats. | 4→5; bench: 2 harvest; harvest 83/23 |

**Dua Hazeen Recited after Tahajjud** (`duasorg-dua-hazeen-recited-after-tahajjud`, 78 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 21 | المحرب → المهرب | sure | 'al-mahrab' spelt with ح instead of ه ('al-mahrab', escape); Mafatih has ه. Probe 16/16. | 7→7; bench: 1 harvest; harvest 77/61 |

**Allahumma Rabba Shahri Ramadan** (`duasorg-ramadan-every-night-dua-1`, 63 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 6 | +عامي; −العام | likely | Mafatih: 'fi ami hadha wa fi kulli am'. Probe 21/21, gain 56 nats. | 6→6; harvest 75/12 |

**Qunoot Salaat-Eid** (`dua-qunoot-eid`, 51 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 6 | +وشرفا وكرامه | likely | Reciters: 'dhukhran wa sharafan wa karamatan wa mazidan'. Probe 20/25, gain 37 nats. | 7→9; harvest 61/10 |

**Duas of Sayeda Fatima (sa) for Debt Repayment** (`duasorg-sahifa-fatimiya-dua-for-debt-repayment`, 38 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 3 | +فالق الحب و النوي | likely | Reciters: '... rabbana wa rabba kulli shay', faliqa al-habbi wa al-nawa, munzila al-tawrat ...'; harvest gap after the line 3.9 s median (96% over 3 s). Probe 15/17, gain 61 nats. | 5→9; harvest 111/109 |

**Sahifa 1: In praise of God** (`sahifa-01`, 33 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 78 | يبتدرنابعقوبته → يبتدرنا بعقوبته | sure | Missing space. | 5→6; bench: 1 harvest; harvest 31/23 |

**Ziyarat of Imam Jafar al-Sadiq (as)** (`duasorg-ziyarat-of-imam-jafar-al-sadiq-1`, 32 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 8 | +السلام عليك يا مفتاح الخيرات | likely | Harvest: median 8.9 s between lines 8 and 9, over 3 s in all 22 recordings; the gap clips (6/6 sampled) hold 'al-salamu alayka ya miftah al-khayrat'. | 5→10; harvest 33/26 |
| 19 | +الصادقين | likely | Reciters: 'ya za'ima al-sadiqina al-salihin'. Probe 12/15, gain 33 nats. | 5→6; harvest 30/21 |

**Naad E Ali (Kabeer)** (`duasorg-naad-e-ali-2`, 25 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 2 | كل هم وغم → لي; رميت متقاضي → امرته ورميت مقضي; +ظل; يد الله ولي → ويذلل | needs Hasan | Our line opens with 'kulla hammin wa ghammin' (from line 3) and differs from what reciters say; the text here is the teacher's transcription, vowelled by hand. Harvest: line 2 passes in 0 of 74 placements; probe 18/18 for this reading, gain 125 nats. The rest of this text (lines 5-15) also rarely passes; it may want replacing from a better copy. | 18→16; harvest 74/0 |

**Dua after Nafilah Salaat** (`duasorg-ramadan-dua-after-every-salaat-6`, 11 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 1 | −العظيم | likely | Mafatih: 'min al-amri al-mahtum' (no 'al-azim'). Probe 5/5, gain 64 nats. | 9→8; harvest 6/1 |

**Dua of Sayeda Fatima (sa) for Sunday** (`duasorg-sahifa-fatimiya-dua-on-sunday`, 11 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 1 | ا للهم → اللهم | sure | Stray space inside 'Allahumma'. | 7→6; harvest 8/1 |

**Dua Hareeq of Sayeda Fatima after Fajr · Allāhumma ṣalli ‘alā Muḥammadin…** (`duasorg-sahifa-fatimiya-dua-hareeq-ater-fajr-4`, 10 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 152 | الكريمهالعظيمه → الكريمه العظيمه | sure | Missing space. | 16→17; harvest 8/6 |

**Sahifa 24: His prayer for parents** (`sahifa-24`, 10 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 61 | للاباءوالامهات → للاباء والامهات | sure | Missing space. | 6→7; harvest 10/9 |

**Sahifa 44: For the coming of the month of Ramadan** (`sahifa-44`, 10 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 51 | جيراننابالافضال → جيراننا بالافضال | sure | Missing space. | 4→5; harvest 7/6 |

**Dua after Salaat of 17th Rabbi al-Awwal** (`duasorg-amaal-on-17-rabbi-al-awwal`, 9 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 76 | ع → عليه السلام | likely | 'ع' abbreviates 'alayhi al-salam'; the line's transliteration spells it out. | 21→22 |
| 82 | بصرهو → بصره و; −قره | likely | Missing space ('basarahu wa waladahu'). The line's bracket is left for Hasan. Bracketed readings: 'قرة [قَرَّتْ]' -> 'قَرَّتْ' (garbled word, vowelled correction in the bracket). | 25→25 |
| 103 | −و ا | sure | 'wa a' ends line 103 and 'sa'aluka' starts 104: 'wa as'aluka' moves whole to 104. The line's bracket is left for Hasan. | 29→27 |
| 104 | سالك → و اسالك | sure | See line 103. | 40→41 |

**Ramadan Night 23** (`dua-ramadan-23-night`, 8 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 29 | +والتوبه | likely | Mafatih: 'wa al-inabata wa al-tawbata wa al-tawfiqa'. Probe 6/6, gain 17 nats. | 11→12; harvest 8/0 |

**Amaal of Sayeda Fatima (sa) for seeking relief from sorrow and grief** (`duasorg-sahifa-fatimiya-amaal-for-relief-sorrow-grief-1`, 7 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 6 | +عليه السلام | likely | The sign 'ؑ' (alayhi al-salam) drops out of the text the matcher sees; reciters say it (5/5 probe clips). | 12→14; harvest 6/6 |

**Sahifa 30: His supplication for help in repaying debts** (`sahifa-30`, 6 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 23 | الدنياالفانيه → الدنيا الفانيه | sure | Missing space. | 6→7; harvest 6/6 |

**Dua Aaliya Madhaameen** (`duasorg-dua-aliya-mazameen-after-ziyarat-jamea`, 4 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 170 | −كله | likely | 'kullihi ( kullahu )': two case endings of one word. | 6→5; harvest 3/3 |
| 194 | −بجاي اين كلمه نام امامي را كه زيارت مي كند و نام در ان بزر وار را ب ويد | sure | A Persian note in brackets ('say the name of the Imam you visit and his father here'), never recited. | 27→8; harvest 3/0 |

**Thursday Night Dua - Allhumma-aj-alni** (`duasorg-thursday-night-rites-taqeeb-3`, 3 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 25 | حاسداواحفظني → حاسدا واحفظني | sure | Missing space. | 10→11; bench: 1 harvest; harvest 3/3 |

**Dua Noor** (`dua-noor`, 1 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 27 | −يستحب قول الاتي عند اللقمه الاولي من الطعام الذي يفطر به الصايم | likely | 'yustahabbu qawlu al-ati ...' is an instruction (the English says so); instruction lines carry no Arabic elsewhere in the corpus (DuaPlayer's slides, commit e59f89a). Empties the line's Arabic and keeps its English. | 12→0; harvest 1/0 |

**Dua after Fajr 1** (`duasorg-ramadan-eid-7`, 1 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 11 | −ا | sure | 'a' ends line 11 and 'an tusalliya' starts 12. | 8→7; harvest 1/1 |
| 12 | ن → ان; −تقبل; −ما | likely | See line 11. Bracketed readings: '[تَقْبَلَ]' dropped; '[مَا]' dropped. | 36→34; harvest 1/1 |

**Ramadan Day 6 · from Sayyid Ibn Baqi** (`duasorg-ramadan-day-06-4`, 0 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 9 | علانيته → علانيتها | sure | 'alaniyatiha' is cut: its last alif starts line 10. | 9→9 |
| 10 | −ا | sure | See line 9. | 11→10 |

**Ramadan Day 7 · from Muhammad ibn Abi Qurrah** (`duasorg-ramadan-day-07-3`, 0 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 1 | ا → يا | likely | 'a sarikh' for 'ya sarikh al-mustasrikhin'. | 3→3 |

**Ramadan Day 13 · from Muhammad ibn Abi Qurrah** (`duasorg-ramadan-day-13-3`, 0 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 3 | −ا | sure | 'a' ends line 3 and 'dinuka' starts 4 ('adinuka'). | 43→42 |
| 4 | دينك → ادينك; −متكبر; −معني | likely | See line 3. Bracketed readings: '[مُتَكَبِّرٍ]' dropped; '[معنى]' dropped. | 23→21 |

**Ramadan Day 14 · from the Book of Companions** (`duasorg-ramadan-day-14-6`, 0 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 3 | −هو; متفرد → متفردا | likely | 'mutafarrid' ends line 3 and its ending 'an' starts 4. Bracketed readings: '[هُوَ]' dropped. | 12→11 |
| 4 | −ا | sure | 'mutafarrid' ends line 3 and its ending 'an' starts 4. | 11→10 |

**Ramadan Day 20 · from the Prophet (s)** (`duasorg-ramadan-day-20-2`, 0 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 12 | ف انك → فانك | sure | Stray space inside 'fa-innaka'. | 5→4 |

**Ramadan Day 20 · from the Book of Companions** (`duasorg-ramadan-day-20-5`, 0 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 10 | ف انك → فانك | sure | Stray space inside 'fa-innaka'. | 17→16 |

**Ramadan Day 24 · Second du'a** (`duasorg-ramadan-day-24-1`, 0 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 2 | الشهادهالكبير → الشهاده الكبير | sure | Missing space. | 5→6 |

**Ramadan Day 28 · Second du'a** (`duasorg-ramadan-day-28-1`, 0 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 2 | −ا | sure | 'i' ends line 2 and 'lla bima sha'' starts 3 ('illa'). | 23→22 |
| 3 | لا → الا | sure | See line 2. | 16→16 |

**Allahumma arina fihi** (`duasorg-ramadan-dua-daytime-daily-4`, 0 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 2 | −ا | likely | A stray alif before 'ma ya'mulu' (the duas.org transliteration reads 'ma ya'mulu'). | 7→6 |

**Purity in Food and before Eating** (`duasorg-ramadan-iftaar-dua`, 0 recordings)

| line | change (as the matcher sees the words) | confidence | evidence | words; impact |
|---|---|---|---|---|
| 5 | ع → عليه السلام | likely | 'ع' abbreviates 'alayhi al-salam'; the line's transliteration spells it out. | 7→8 |

### Dua Nudbah: readings in parentheses (17 lines, sure)

DuaPlayer's Nudbah keeps a second reading in parentheses beside the recited one, and the
matcher counts both as words to be said. The line's own transliteration reads only the first in
every case, and so do the reciters. Line 200 also lacks "khalaqtahu lana ismatan wa maladhan"
(duas.org, Mafatih; probe 16/16 with it) and is in the table above.

| line | in parentheses (dropped) | probe | words | harvest placed/passed | bench |
|---|---|---|---|---|---|
| 38 | ( به ) | 15/16 for one word there instead of two; the clips say 'wa arajta bi-ruhihi' | 6→5 | 412/162 | 1 studio, 5 harvest |
| 79 | ( ناهش ) | 16/16 | 12→11 | 416/202 | 2 harvest |
| 95 | ( فَلْتًدرِ ) | 15/16 | 14→13 | 410/82 | 1 harvest |
| 115 | ( النِفاقِ ) | 15/16 | 7→6 | 387/218 | 2 harvest |
| 117 | ( الكَذِبِ ) | 16/16 | 7→6 | 385/2 |  |
| 121 | ( الكَلِمِ ) | 16/16 | 6→5 | 408/0 |  |
| 128 | ( المُطالِبُ ) | 13/16 | 6→5 | 464/1 |  |
| 137 | ( المُهْتَدينَ ) | 16/16 | 5→4 | 377/0 |  |
| 140 | ( المُتَطَهْريِِنَ ) | 16/16 | 5→4 | 373/0 |  |
| 142 | (الأكْبَرينَ ) | 16/16 | 5→4 | 406/18 |  |
| 153 | ( المَشْهُورَةِ ) | 16/16 | 5→4 | 406/0 |  |
| 170 | ( لا تُحِيطَ بِِيَ دُونكَ ) | 16/16 | 19→15 | 412/0 |  |
| 172 | ( يَنْزِحُ ) | 16/16 | 8→7 | 404/2 |  |
| 189 | ( فَتَقُرُ عًُيًُوننا ) | 15/16 | 8→6 | 402/1 |  |
| 197 | ( الاُولي ) | 16/16 | 13→12 | 411/96 |  |
| 204 | ( جَنّاتِكَ ) | 16/16 | 9→8 | 405/150 | 1 studio, 1 harvest |

### duas.org texts: readings in square brackets (175 lines)

The duas.org pages built from the Persian Mafatih keep its alternative readings ("nuskhah
badal") in square brackets after the word they replace, sometimes an optional addition. The
matcher counts both. Proposed rule (`bracket_edit` in `assemble.py` in the tools folder, written out
line by line in `fixes.json`): when the word before the bracket has no vowel marks and the bracket does (the page
typed a garbled word and its correction: "khbr [khayra]", "dmt [samat]"), the bracketed reading
replaces it; otherwise the bracket goes and the main reading stays. "likely" when the line's
transliteration reads it that way or there is none to check; "needs Hasan" (27 lines) when the
transliteration keeps the bracket (mostly `amaal-on-17-rabbi-al-awwal` and
`monday-rites-taqeeb-*`), the bracket is the whole line, or the brackets are malformed. Three lines with reversed or
unclosed brackets are left alone (`duasorg-ramadan-day-25-4`:44, `duasorg-thursday-rites-taqeeb-3`:6
and 66). Almost none of these texts has a recording: the evidence is the convention and the
transliterations. Mubahila's (`duasorg-eid-e-mubahila-3`, 141 recordings, but all for the Quran
verses that open it; no clip reaches its brackets) are in the table above.

| text | recordings | likely | needs Hasan |
|---|---|---|---|
| `duasorg-1st-muharram-dua-amaal-2` | 90 | 0 | 2 |
| `duasorg-amaal-on-17-rabbi-al-awwal` | 9 | 2 | 5 |
| `duasorg-monday-rites-taqeeb-5` | 3 | 0 | 3 |
| `duasorg-monday-rites-taqeeb-6` | 1 | 0 | 1 |
| `duasorg-ramadan-day-15-3` | 1 | 3 | 0 |
| `duasorg-ramadan-day-18-5` | 1 | 1 | 0 |
| `duasorg-ramadan-eid-7` | 1 | 7 | 0 |
| `duasorg-arafah-day-dua-amaal-6` | 0 | 1 | 0 |
| `duasorg-arafah-day-dua-amaal-7` | 0 | 1 | 0 |
| `duasorg-dua-ahad-second-1` | 0 | 0 | 2 |
| `duasorg-monday-rites-taqeeb-7` | 0 | 0 | 1 |
| `duasorg-ramadan-day-02-2` | 0 | 2 | 0 |
| `duasorg-ramadan-day-02-3` | 0 | 3 | 0 |
| `duasorg-ramadan-day-02-4` | 0 | 1 | 0 |
| `duasorg-ramadan-day-03-3` | 0 | 1 | 0 |
| `duasorg-ramadan-day-04-3` | 0 | 1 | 0 |
| `duasorg-ramadan-day-04-4` | 0 | 3 | 0 |
| `duasorg-ramadan-day-04-5` | 0 | 1 | 0 |
| `duasorg-ramadan-day-05-1` | 0 | 1 | 0 |
| `duasorg-ramadan-day-05-3` | 0 | 2 | 0 |
| `duasorg-ramadan-day-05-4` | 0 | 1 | 0 |
| `duasorg-ramadan-day-06-2` | 0 | 1 | 0 |
| `duasorg-ramadan-day-06-4` | 0 | 1 | 0 |
| `duasorg-ramadan-day-07-2` | 0 | 3 | 0 |
| `duasorg-ramadan-day-07-4` | 0 | 1 | 0 |
| `duasorg-ramadan-day-08-4` | 0 | 1 | 0 |
| `duasorg-ramadan-day-08-5` | 0 | 1 | 0 |
| `duasorg-ramadan-day-09-4` | 0 | 0 | 1 |
| `duasorg-ramadan-day-10-2` | 0 | 3 | 0 |
| `duasorg-ramadan-day-11-1` | 0 | 3 | 0 |
| `duasorg-ramadan-day-12-2` | 0 | 1 | 0 |
| `duasorg-ramadan-day-12-3` | 0 | 1 | 0 |
| `duasorg-ramadan-day-12-5` | 0 | 1 | 0 |
| `duasorg-ramadan-day-13-2` | 0 | 1 | 0 |
| `duasorg-ramadan-day-13-4` | 0 | 0 | 1 |
| `duasorg-ramadan-day-13-5` | 0 | 0 | 3 |
| `duasorg-ramadan-day-14-1` | 0 | 1 | 0 |
| `duasorg-ramadan-day-14-5` | 0 | 1 | 0 |
| `duasorg-ramadan-day-14-6` | 0 | 4 | 0 |
| `duasorg-ramadan-day-15-1` | 0 | 1 | 0 |
| `duasorg-ramadan-day-15-2` | 0 | 1 | 0 |
| `duasorg-ramadan-day-15-4` | 0 | 1 | 0 |
| `duasorg-ramadan-day-15-6` | 0 | 1 | 0 |
| `duasorg-ramadan-day-16-2` | 0 | 1 | 0 |
| `duasorg-ramadan-day-16-5` | 0 | 2 | 0 |
| `duasorg-ramadan-day-17-3` | 0 | 1 | 0 |
| `duasorg-ramadan-day-17-5` | 0 | 1 | 0 |
| `duasorg-ramadan-day-18-3` | 0 | 5 | 0 |
| `duasorg-ramadan-day-19-1` | 0 | 2 | 1 |
| `duasorg-ramadan-day-19-2` | 0 | 1 | 0 |
| `duasorg-ramadan-day-20-1` | 0 | 1 | 0 |
| `duasorg-ramadan-day-20-2` | 0 | 3 | 0 |
| `duasorg-ramadan-day-20-3` | 0 | 2 | 0 |
| `duasorg-ramadan-day-20-4` | 0 | 0 | 3 |
| `duasorg-ramadan-day-20-5` | 0 | 4 | 0 |
| `duasorg-ramadan-day-21-2` | 0 | 1 | 0 |
| `duasorg-ramadan-day-21-4` | 0 | 4 | 0 |
| `duasorg-ramadan-day-22-3` | 0 | 1 | 0 |
| `duasorg-ramadan-day-22-5` | 0 | 3 | 0 |
| `duasorg-ramadan-day-23-2` | 0 | 1 | 0 |
| `duasorg-ramadan-day-23-5` | 0 | 1 | 0 |
| `duasorg-ramadan-day-24-1` | 0 | 1 | 0 |
| `duasorg-ramadan-day-24-3` | 0 | 1 | 0 |
| `duasorg-ramadan-day-25-3` | 0 | 3 | 1 |
| `duasorg-ramadan-day-25-4` | 0 | 0 | 1 |
| `duasorg-ramadan-day-25-6` | 0 | 1 | 0 |
| `duasorg-ramadan-day-26-3` | 0 | 1 | 0 |
| `duasorg-ramadan-day-26-4` | 0 | 1 | 0 |
| `duasorg-ramadan-day-28-3` | 0 | 1 | 0 |
| `duasorg-ramadan-day-28-4` | 0 | 0 | 2 |
| `duasorg-ramadan-day-30-3` | 0 | 8 | 0 |
| `duasorg-ramadan-eid-11` | 0 | 2 | 0 |
| `duasorg-ramadan-eid-13` | 0 | 20 | 0 |
| `duasorg-ramadan-eid-8` | 0 | 6 | 0 |
| `duasorg-ramadan-first-10-short-duas-8` | 0 | 0 | 1 |
| `duasorg-sahifa-fatimiya-dua-3rd-of-every-month` | 0 | 0 | 1 |

## Looked at, not changed

- **Reciters split.** Kumayl 206 "fi al-barizin" (ours, Mafatih) or "fi al-mubadirin" (duas.org):
  19/30 clips say the second, so it is listed for Hasan, as is Iftitah 124's "al-Zahra" (18/30
  without). Not listed, near 50/50: Ashura 93 "thumma [il'an] al-thani" (17/30), Arafat 299
  (17/30 for repeating line 298), Mujeer 165 "ya ali / ya muta'ali" (17/30), Ziyarat on the Eid
  55 (12/19 for adding "wa ummatan qatalatk"), Sahifa 47:71 "qahira al-arbab" (9/21).
- **Reciters add, the text is right.** Honorifics after names in Hadith al-Kisa (said by 4-5 of
  16 reciters), "Allahumma salli ala Muhammad wa al Muhammad" before du'as, "ya rabb" repeated
  past the text's three (Abu Hamza 12), the bismillah and personal requests between lines, repeated
  suras. The Jawshan Kabir refrain passes the check in about half its placements (the du'a's
  median is 0.96), but the clips say the text as written; the audience reciting along and the drawn-out
  "al-ghawth" make it hard to align, not wrong.
- **The reading is wrong, not the Arabic.** Sahifa Fatimiya's Hareeq (lines 201-246: rows out of
  step), Abu Hamza 10-11 and Ya Uddati 115-116 (reading split differently), Munajat Imam Ali
  15/23/24/30 (doubled words in the reading), Ziyarat Waritha 31 and 46 (reading shorter than
  the Arabic, which is right).
- **One-letter variants**, which the follower absorbs: Jawshan Kabir "la yuhsi / la tuhsi", "la
  yaruddu / la taruddu" (16/16 for t-), hamza spellings everywhere.

## Suspected, not settled

- `duasorg-naad-e-ali-2` (25 recordings): lines 2-15 rarely pass (line 2: 0 of 74); reciters say
  a different version. Line 2's proposal comes from the teacher's transcription; the text may
  want replacing from a better copy.
- Ziyarat Waritha 64-67 say "fa-ya laytani kuntu ma'akum fa-afuza ma'akum" twice; most
  recordings say it once (line 66 is placed 13 times, 0 pass). Removing it would empty two
  lines, so it is not in `fixes.json`.
- Mubahila 95: "wa hifzika [wa la taqbid] yadayya an [kulli] khayrin": without the first bracket
  the sentence has no verb, so the brackets may be additions rather than alternatives.
- Smaller single-clip hints not pursued: Ziyarat Muslim (3) 12 and 21, Tawba 17 "bihi / fihi",
  Salat Ja'far (4) 4, 9 and 13 ("ya muqil al-atharat" between 13 and 14 in 3 of 6 gap clips).
- The 280 texts with under 5 recordings had only the text checks; the Ramadan night du'as
  (22-30) look alike in their frames but are different texts, and the Mafatih blocks for them
  are cut differently, so their second-copy differences are not errors.

## What applying the edits changes

With the default (`sure` + `likely`, 259 lines), 248 lines change their word count. Each such line:

- **drops every bench source that reads it** (`bench.fresh_source` returns None when a line's
  word count differs from the one the source was built with): 41 of 584 sources, 4 of the 26
  studio sources (Iftitah, Nudbah twice, Tawba) and 37 of the 455 harvest ones (Nudbah 11,
  15 Sha'ban 8, Warith 4, Mashlool 3, Yastasheer 3, Muslim ibn Aqil 5, Tawba 2, Thursday 2,
  Sahifa 1, Thursday-night taqeeb 1). The `sure` edits alone drop 24 (3 studio, 21 harvest).
  12 more sources (2 studio, 10 harvest) read a line whose words change but not their number
  (Tawassul 60, Mujeer 216, Warith 7, Makarim 14, Mashlool 158, Hazeen 21): they stay, scored
  against the new words. To get the dropped ones back, rebuild the sources and their items (`bench.py sources`, `build`, `asr`: a GPU pass), and
  rebase every comparison: scores on the old and new grid are not comparable.
- **leaves its harvest labels stale**: 19,752 placements of these lines (5,098 passing) carry
  the old words. `harvest_label.py export` and anything that compares label words with the
  corpus (the probe here, `bench.py sources`) skip or mismatch them until the recordings are
  relabelled, which needs the teacher's frames again (deleted to free D:).
- **leaves word truth stale** for the human-timed recordings that cover it: 93 line-recordings
  in DuaPlayer, duas.pro and duas.org timings and 26 in the majlis test set
  (`scripts/word_truth.py` recomputes them).
- changes `web/corpus.json` only after `python scripts/export_web.py`.

Line timings (DuaPlayer's slide starts) are untouched: no segment is added, removed or
renumbered. Two edits for Hasan empty a line (Ashura 23) or keep only part of one (Ashura 24),
and Dua Noor 27 (likely) empties an instruction line, as the corpus already does for
DuaPlayer's instruction slides.

## Files

- `data/cache/text_audit/fixes.json`: the 293 edits, each with `old_arabic`, `new_arabic`,
  `confidence`, `kind`, `evidence` and `impact` (word counts, bench sources by lane, harvest
  placements/passes, timed recordings).
- `scripts/text_fixes.py`: applies them to `data/duas` (or `--dir COPY`); `--dry-run` prints the
  word-level change of each line, `--confidence sure` limits it, `--only dua:seg` picks lines
  (needed for the `needs Hasan` ones). It refuses a line whose Arabic no longer matches
  `old_arabic`. Tested on a copy (`data/cache/text_audit/duas_test/`, left in place): 259 lines
  change, every other line and every file's formatting stays byte for byte, `corpus.load_all`
  and `CorpusIndex` load it (140,803 words before, 140,580 after).
- `data/cache/text_audit/evidence/`: the probe results (each clip's teacher transcript and
  losses), harvest line statistics and inter-line gaps, the second-copy differences, the
  reading checks and the bracket list. `data/cache/text_audit/tools/`: the scripts that made
  them (not part of the repo; `probe.py`/`probe2.py` run the teacher, `review.py DUA` prints
  every signal for a text).

## Second review (2026-10-09)

An independent check of the 259 sure + likely edits before they are applied, with a glance at
the 34 for Hasan. One entry per edit in `data/cache/text_audit/review.json`: verdict, reason,
and every probe on the line recounted from its clips (rows, and distinct recitations).

**Method.** Each of the 89 edits in texts with 20+ recordings was read against its neighbouring
lines, its own transliteration and English, the second copies (`evidence/ref_diffs.json`, the
decoded Mafatih) and my own knowledge of the texts, named where it is the only source. Its probe
clips were read too: the teacher's transcript of each clip, with the preference recounted from
the losses. The 170 in the long tail were read at the level of the changed words: for the 142
bracket edits, the bracketed words, what is kept, the word diff and the transliteration where
there is one. Every bracket edit that drops three or more words, or swaps in a "garbled-word"
correction, was read in full with its English. Mechanical checks on all 259:

- `old_arabic` matches today's file, and only the intended words change.
- No inserted words repeat the edge of a neighbouring line.
- The cross-line repairs come in pairs, both members in the set, and the text's letters are
  conserved.
- The reviewed set, applied by `scripts/text_fixes.py --dir` to a scratch copy of `data/duas`,
  changes 257 lines in 107 files with nothing skipped. `corpus.load_all` and `CorpusIndex` load
  the copy (140,803 words → 140,570).

**Re-uploads.** "Clips from different uploaders" are not always different recitations: one
recording is often re-uploaded to aparat and YouTube under other names. Rows with the same clip
length (within 0.2 s), a near-identical transcript and the same losses (within 4 nats or 8%) were
merged. Most counts barely move. The exceptions are Ashura 3 (below) and Jafar al-Sadiq 19 (12/15
rows, 4/7 recitations), which stands on its own transliteration and English.

**Counts.** 254 agree, 3 fix, 2 disagree. 127 were checked one by one against sources and clips.
For the other 132 (low-recording bracket edits), the words were checked one by one and the
choice of reading rests on the convention, which holds (below).

**Disagree: leave these out of the sure + likely set.**

- **Ziyarat Ashura 3** (+ "al-salamu alayka ya khiyarat Allah wa ibna khiyaratih"). The text is
  right (Mafatih, duas.org), but reciters split. 10 of the 30 span-probe clips and 5 of the 16
  line clips are one recitation (Ali Fani's), re-uploaded; distinct recitations prefer the edit
  in 9 of 19 and 7 of 12. The gap before line 3 is over 3 s in 61% of 417 recordings, against 2%
  for Ashura's lines in general (median over 89 lines), so at most 55-60% say it. That is the
  60/40 split the audit already sends to Hasan for Kumayl 206 and Iftitah 124.
- **15 Sha'ban, the Prophet's du'a, line 11** (+ "bi-dhunubina", + "man la yakhafuka fina wa
  la"). The long form belongs to the du'a's other version, the one Quran channels recite: "... wa
  la ila al-nari masirana, wa-j'al al-jannata hiya darana, wa la tusallit alayna bi-dhunubina
  ...". The clips that prefer the edit also say the "darana" clause, which our text lacks; the 4
  that say the short form are Shia reciters. Our Arabic, its transliteration and English, and
  Mafatih all have the short form. This text's harvest is almost entirely the other version
  (lines 3, 4 and 5 pass in 6, 10 and 19 of about 1,050 placements). Patching line 11 alone
  gives a text that matches neither version. If those reciters matter, their version wants a
  text of its own. Leaving it out also keeps the 8 harvest bench sources that read the line.

**Fix: new Arabic in review.json.**

- **`duasorg-ramadan-day-04-4` 8 and 9.** Lines 8 and 9 each end in a stray "و", and lines 9 and
  10 start with a bare fatha: "wa as'aluka" is cut across lines, the error the audit repairs
  elsewhere. Dropping line 8's bracket glued "كَيْنُونَتِكَو" into one word. Fixed: 8 drops the
  bracket and the "و"; 9 starts "وَ أَسْأَلُكَ" and keeps its trailing "و" for line 10, which is
  not in fixes.json. If line 10 is edited too, 9 can drop that "و".
- **Mubahila 92.** Dropping "[min]" is right, but "la tahrimni fadlaka" then takes the accusative
  (the edit kept "fadlika"). This affects the display only; the matcher ignores vowels.

**Agreed, but weaker than their label.**

- **Ziyarat Waritha 7** ("alayhi al-salam" for "waliyy Allah") and **35** (− "wa tanaqqabat"):
  the sources agree with the edit, but about a third of reciters say the old wording (18 of 29
  and 19 of 30 distinct recitations). A net gain, but a small one.
- **Jawshan Saghir 434.** Mafatih has the short refrain at this section; the long form follows the
  sick, prisoner and captive sections (our 325, 368, 393, 413). The edit rests on reciters: 10 of
  11 distinct, and in 4 labelled recordings the line fails and is followed by a 4-5 s gap. Lines
  398 and 418 rest on Mafatih and on consistency with 261, 351 and 373. Only 3-10 of 142
  recordings reach lines 398-427 (most go from 397 to 437), and the one clear probe clip there
  is a reciter who says the uniform opener everywhere.
- **Sahifa 54, lines 18-20** (bismillah before each sura): right in substance, but the probe
  population is Quran and adhkar uploads, not Sahifa 54. Lines 9-20 (Ayat al-Kursi and the
  suras) are placed 172-425 times; the du'a's own lines 5-30 times.
- **Where the corpus has no second copy and I rely on my own knowledge of the text:** Ramadan Day
  3 (Sahifa 7: "minha", "rahmatan wa"), Tawba 38 and 137 (Sahifa 31), Makarim 14 (Sahifa 20),
  Qunoot Eid 6, Muslim ibn Aqil (3) 10 and 16, and the Fatimiya debt du'a 3. All also have a
  clear majority of reciters.

**The bracket convention holds.** It keeps the main reading (Mafatih's text; the bracket is its
alternative, "nuskhah badal") and replaces an unvowelled garbled word by its vowelled bracket.
The kept reading is coherent Arabic in every line read. Where a transliteration exists, it reads
the kept reading, with one exception: in `amaal-on-17-rabbi-al-awwal` 80 it keeps both ("mimma
[kama]"), contrary to what the evidence line says. That line's English supports the edit's
"kama". In two lines (`ramadan-day-05-4` 8, `ramadan-day-21-4` 3) the duas.org English follows
the bracket; with no recordings to go by, the convention stands. The rule went wrong
mechanically once (day-04-4, above).

**Evidence lines that are wrong; the edits stand.**

- `ramadan-dua-daytime-daily-4` 2: the transliteration reads "Ama ya'mulu", not "ma ya'mulu".
- `amaal-on-17-rabbi-al-awwal` 80: see above.
- Jawshan Saghir 398 and 418: they say "too few to probe", but a 2-clip and a 1-clip probe exist
  and go the other way.

Every probe fraction and harvest pass count I recounted matches the evidence files: all those in
the popular texts, plus a sample of the rest.

**In passing; none of these is in fixes.json.**

- `ramadan-eid-7` 28 holds an editor's note, "زِيَادَةٌ فِيهِ" ("an addition here:"), followed by
  a variant passage.
- Jawshan Saghir 429, "la bi-stihqaqin minni", fails (cost 4.6-6.6) in all 4 recordings checked.
- Makarim 133's English still repeats line 132.
- 47 edited lines also come out NFC-normalised, with shadda and vowel marks reordered. Nothing
  reads that order: `text.normalize`, `translit` and `web/tracker.js` all apply NFKC first.

**The 34 for Hasan, at a glance.**

- Ashura 23-24: the evidence (Mafatih; span probe 15/16) is as strong as the sure edits, and 35
  lines in 17 texts already have no Arabic, so emptying 23 has precedent.
- Kumayl 206 and Iftitah 124: real splits.
- The 27 bracket edits look applicable as written, except the three that empty or gut a line
  (`ramadan-day-09-4` 26, `ramadan-day-19-1` 10, `ramadan-day-25-3` 4).

**Applying.** The 257 that remain (254 as written, 3 with the fixed Arabic) are safe to apply.
They drop 33 of the 584 bench sources (4 studio, 29 harvest; counted with `bench.fresh_source`
on the patched copy), not the 41 counted above. Labels and word truth go stale as described in
"What applying the edits changes".

## Applied (2026-10-09)

Hasan chose sure + likely after the second review. Applied: 257 edits
(`data/cache/text_audit/fixes_reviewed.json`: the 259 sure + likely, less Ziyarat Ashura 3 and 15
Sha'ban 11, with the review's Arabic for Eid al-Mubahila 92 and Ramadan Day 4 (4) 8-9); 245 change a
line's word count. `data/duas` before the change is kept in `data/cache/text_audit/duas_before/` and
the old `web/corpus.json` in `corpus_before.json`; `scripts/export_web.py` was rerun.

The bench: `scripts/bench_retext.py` carried the items over (each truth word through its line id and
place in the line, old texts to new; 93,070 words land on the same word, 21 on a word an edit
corrected). 92 of 2,410 items read a line whose word count changed and were moved to
`data/testbed/items_retired/text-audit-2026-10-09/`; 778 were rewritten. The tracker cache keys on the
corpus, so the next score recomputes it: score a fresh baseline before comparing with any run from
before 2026-10-09 (`stop_*`, `full_ph_*` and older are on the old texts). `sources.jsonl` is
untouched; a rebuild drops the sources that read a changed line (`bench.fresh_source`). The harvest
labels of the changed lines are stale until relabelled. All tests pass on the new corpus (175).
Fresh baselines on the corrected texts at the phone's cadence, today's defaults (with the stop catch-up,
stop_catchup.md): `base0909_dev` (1,317 items: on line 90%, right word 74%) and `base0909_test` (594 items:
on line 90%, right word 76%).
