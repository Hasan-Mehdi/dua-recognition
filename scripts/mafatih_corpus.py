#!/usr/bin/env python
"""Add widely recited texts the app lacks, from Mafatih al-Jinan: Du'a al-Sabah,
al-'Adeelah, al-'Asharat and Munajat 2-15 (the first, Ta'ibeen, is dua-munajat-taibeen).
The harvest has 1,214 recordings of them, which until now got no du'a or a wrong one.

The text is the decoded book (scripts/mafatih_texts.py -> data/harvest/extra_texts.json),
which keeps the printed page's line ends (mid-phrase) and an Iranian typesetting:
ى for final ي, no hamza seats (اَنْتَ, اِلهى), madda written as shadda (رَجاَّئى), alif
with sukun for a lost hamza (الاْرْض), a separate وَ, some words glued or broken. The
spelling is set to the corpus's, word by word by how the other texts write it (so the
generated reading is right: "fī", not "fā"), and the lines are cut where reciters stop:
the harvest's alignments of these texts give the pause after every word in each
recording (held-out bench voices left out), and a dynamic programme breaks where most
reciters stop, at 3-9 words a line. Where no recording was aligned confidently (each
text's first book line, two stretches of 'Asharat and one of 'Adeelah), and where the
stops split a phrase (a construct, a run of parallel clauses), the lines are set by hand
(MANUAL), read through line by line. No English: none of our sources (DuaPlayer,
duas.org, duas.pro) have these texts.

    python scripts/mafatih_corpus.py            # -> data/duas/<id>.json, prints the lines
    python scripts/mafatih_corpus.py --dry-run  # print only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.text import normalize  # noqa: E402

EXTRA = ROOT / "data" / "harvest" / "extra_texts.json"
LABELS = ROOT / "data" / "harvest" / "labels"
TEST_VOICES = ROOT / "data" / "testbed" / "test_voices.json"
DUAS = ROOT / "data" / "duas"

# (Mafatih block, our id, English name, Arabic name). Each block's opening was checked
# against its title: the decoded book's titles sit one block off in places.
TEXTS = [
    ("mafatih-0037", "mafatih-dua-sabah", "Dua Sabah", "دعاء الصباح"),
    ("mafatih-0044", "mafatih-dua-adeelah", "Dua Adeelah", "دعاء العديلة"),
    ("mafatih-0039", "mafatih-dua-asharat", "Dua Asharat", "دعاء العشرات"),
    ("mafatih-0064", "mafatih-munajat-shakeen", "Munajat Shakeen", "مناجاة الشاكين"),
    ("mafatih-0065", "mafatih-munajat-khaifeen", "Munajat Kha'ifeen", "مناجاة الخائفين"),
    ("mafatih-0066", "mafatih-munajat-rajeen", "Munajat Rajeen", "مناجاة الراجين"),
    ("mafatih-0067", "mafatih-munajat-ragibeen", "Munajat Ragibeen", "مناجاة الراغبين"),
    ("mafatih-0068", "mafatih-munajat-shakireen", "Munajat Shakireen", "مناجاة الشاكرين"),
    ("mafatih-0069", "mafatih-munajat-muteeen", "Munajat Muti'een", "مناجاة المطيعين لله"),
    ("mafatih-0070", "mafatih-munajat-mureedeen", "Munajat Mureedeen", "مناجاة المريدين"),
    ("mafatih-0071", "mafatih-munajat-muhibbeen", "Munajat Muhibbeen", "مناجاة المحبين"),
    ("mafatih-0072", "mafatih-munajat-mutawassileen", "Munajat Mutawassileen", "مناجاة المتوسلين"),
    ("mafatih-0073", "mafatih-munajat-muftaqireen", "Munajat Muftaqireen", "مناجاة المفتقرين"),
    ("mafatih-0074", "mafatih-munajat-arifeen", "Munajat Arifeen", "مناجاة العارفين"),
    ("mafatih-0075", "mafatih-munajat-dhakireen", "Munajat Dhakireen", "مناجاة الذاكرين"),
    ("mafatih-0076", "mafatih-munajat-mutasimeen", "Munajat Mu'tasimeen", "مناجاة المعتصمين"),
    ("mafatih-0077", "mafatih-munajat-zahideen", "Munajat Zahideen", "مناجاة الزاهدين"),
]

# The book's Persian instructions inside the Arabic of 'Asharat ("and ten times:",
# "then say:"). Removed; each place is a line break (the repeated phrases between them
# are lines of their own).
INSTRUCTIONS = [["و", "نيز", "ده", "مرتبه", "مى", "گوئى:"], ["و", "ده", "مرتبه", ":"], ["و", "ده", "مرتبه"],
                ["پس", "مى", "گويى:"]]
# Words the decoder broke in two, and glued together.
BROKEN = {("وَالصّ", "الِحينَ"): "وَالصّالِحينَ", ("وَالاِْعْلا", "نِ"): "وَالإِعْلانِ",
          ("اَنْبِي", "اَّئَكَ"): "أَنْبِياءَكَ", ("اءَ", "اَقْطَعُ"): "أَأَقْطَعُ",
          ("اَنا", "مَتْنِى"): "أَنامَتْنِي", ("وَتَض", "اَّئَلَ"): "وَتَضائَلَ", ("اِكْر", "امِكَ"): "إِكْرامِكَ",
          ("اِيّ", "اىَ"): "إِيّايَ", ("نَعْم", "اَّئِكَ"): "نَعْمائِكَ",
          ("وَوَعَدْ", "تَنا"): "وَوَعَدْتَنا"}
GLUED = {
    "لايُرى": "لا يُرى", "كُلِّشَىءٍ": "كُلِّ شَىْءٍ", "ذَاالطَّوْلِ": "ذَا الطَّوْلِ", "ذَاالْجَلالِ": "ذَا الْجَلالِ",
    "لاحَوْلَ": "لا حَوْلَ", "لاقُوَّهَ": "لا قُوَّهَ", "اِلاّبِاللّهِ": "اِلاّ بِاللّهِ", "ذَاالَّذى": "ذَا الَّذى",
    "وَالْحَمْدُلِلّهِ": "وَالْحَمْدُ لِلّهِ", "يامُجيبُ": "يا مُجيبُ", "لايُغْنيهِ": "لا يُغْنيهِ", "وِ": "وَ",
    "قَدْاَمَرْتَنا": "قَدْ اَمَرْتَنا", "ذُوالنِّعَمِ": "ذُو النِّعَمِ", "اِذَاامْتازَ": "اِذَا امْتازَ", "كُلِّشَىْءٍ": "كُلِّ شَىْءٍ",
    "كُلَّشَىْءٍ": "كُلَّ شَىْءٍ", "مَقْعَدَالصِّدْقِ": "مَقْعَدَ الصِّدْقِ", "يَقِّرُّدُونَ": "يَقَرُّ دُونَ",
    "ذَاالّلِسانِ": "ذَا اللِّسانِ", "اَنْتَذْكُرَنا": "اَنْ تَذْكُرَنا", "الاْبْصارُدُونَ": "الاْبْصارُ دُونَ",
}
# Typos, as the decoder wrote them (after a separate وَ is joined on) -> the word.
FIXED = {
    "فَاْصْفَحِ": "فَاصْفَحِ", "الّلَيْلِ": "اللَّيْلِ", "الَّليْلِ": "اللَّيْلِ", "اللَيْلَ": "اللَّيْلَ", "الِهِ": "آلِهِ",
    "ناِزلاً": "نازِلاً", "وَالْفَنَّاءِ": "وَالْفَناءِ", "وَمَلاَْتَ": "وَمَلَأْتَ", "الْمَاءرِبَ": "الْمَآرِبَ",
    "يَرُِدُّ": "يَرُدُّ", "ساَّئِلَُهُ": "سائِلَهُ", "يُخَيَِّبُ": "يُخَيِّبُ", "امِلَُهُ": "آمِلَهُ",
    "يُسارِعوُنَ": "يُسارِعُونَ", "فَتَقَِرَّ": "فَتَقَرَّ", "لاِِرادَتِكَ": "لإِرادَتِكَ", "عَفِْوكَ": "عَفْوِكَ",
    "مَاءوَى": "مَأْوَى", "يَمْلاَُ": "يَمْلَأُ", "الُمُضِلِّينَ": "الْمُضِلِّينَ", "الْقِيمَةِ": "الْقِيامَةِ",
    "الاّ": "إِلّا", "الرّحِمينَ": "الرّاحِمينَ", "السَّمواتِ": "السَّماواتِ", "سَمواتٍ": "سَماواتٍ",
    "اَلْجَاَتْنِى": "أَلْجَأَتْنِي", "وَاطْمَاَنَّتْ": "وَاطْمَأَنَّتْ", "مَلْجَاَ": "مَلْجَأَ", "اللاَّّئِذينَ": "اللّائِذينَ",
    "يُعِزُّهاَّ": "يُعِزُّها", "ائِمَّةً": "أَئِمَّةً", "وَائِمَّةُ": "وَأَئِمَّةُ", "امَنّا": "آمَنّا", "امينَ": "آمينَ",
    "لِجُرْاَتِها": "لِجُرْأَتِها", "وَاَجْرِ": "وَأَجْرِ", "اَكُفَّ": "أَكُفَّ", "اماقي": "آماقي", "اماقى": "آماقي",
    "الْتَجَاءَ": "الْتَجَأَ", "وَوِقايَهً": "وَوِقايَةً", "عَلَيْهَ": "عَلَيْهِ", "ايجادِ": "إِيجادِ", "اُمَّةِ": "أُمَّةِ",
    "وَاِماما": "وَإِماماً", "اَنِّي": "أَنِّي", "الْعَظيِمِ": "الْعَظيمِ", "واصْطَفْيَتَهُمْ": "وَاصْطَفَيْتَهُمْ",
    "مَلاَ": "مَلَأَ", "بِرَاءْفَتِكَ": "بِرَأْفَتِكَ", "اَلْسِنَةً": "أَلْسِنَةً", "وَوُفّيَتْ": "وَوُفِّيَتْ",
    "اَلْهُو": "أَلْهُو", "اذَنَني": "آذَنَني", "وَالائِكَ": "وَآلائِكَ", "انَسَني": "آنَسَني", "اِكْرامِكَ": "إِكْرامِكَ",
    "الْبَّرُ": "الْبَرُّ", "الائِكَ": "آلائِكَ", "فَالائُكَ": "فَآلائُكَ", "الأَمِلينَ": "الآمِلينَ",
    "الشُّكوُكَ": "الشُّكُوكَ", "الْمَنايِحِ": "الْمَنائِحِ", "فَاِنّا": "فَإِنّا", "وَاَلْحِقْنا": "وَأَلْحِقْنا",
    "وَبَوَّاْتَهُ": "وَبَوَّأْتَهُ", "دَاءْبُهُمُ": "دَأْبُهُمُ", "لاِبْصارِ": "لأَبْصارِ", "وَبوَّاءْتَهُمْ": "وَبَوَّأْتَهُمْ",
    "كَفّى": "كَفّي", "وَشَرايِعَ": "وَشَرائِعَ", "الْمَاءْمُولِ": "الْمَأْمُولِ", "اِقامَةِ": "إِقامَةِ",
    "اِحْصاءِ": "إِحْصاءِ", "اِدْراكِ": "إِدْراكِ", "بِاِدْراكِ": "بِإِدْراكِ", "اِدْراكِها": "إِدْراكِها",
    "وَاِبْعادِكَ": "وَإِبْعادِكَ", "وَاِعْظاماً": "وَإِعْظاماً", "أَلوُذُ": "أَلُوذُ",
}
# Final ى the corpus vote gets wrong or can't decide: alif maqsura (ā).
MAQSURA = {"مُنى", "وَالْمُنى", "اَقْصى", "فَابْتَغى", "اَوى", "الْعِدى", "نَتَمَنّى"}

# Lines set by hand where no recording was aligned confidently, matched in order on
# their normalize()d words; each is a line of its own, the rest is left to the pauses.
MANUAL = {
    "mafatih-dua-sabah": ["اللهم يا من دلع لسان الصباح بنطق تبلجه", "وسرح قطع الليل المظلم بغياهب تلجلجه",
                          "اغفر ذنوبي كلها بحرمه محمد وال محمد", "يا غفار يا غفار يا غفار"],
    "mafatih-dua-adeelah": [
        "شهد الله انه لا اله الا هو", "والملايكه واولوا العلم قايما بالقسط", "لا اله الا هو العزيز الحكيم",
        "علي قامع الكفار", "ومن بعده سيد اولاده الحسن بن علي", "ثم اخوه السبط التابع لمرضات الله الحسين",
        "وافضل الاوصياء المرضيين", "واشهد ان الموت حق ومسايله القبر حق", "والبعث حق والنشور حق والصراط حق",
        # the closing (book lines the aligner never placed)
        "اللهم اني اعوذ بك من العديله عند الموت", "اللهم يا ارحم الراحمين", "اني قد اودعتك يقيني هذا وثبات ديني",
        "وانت خير مستودع",
        "وقد امرتنا بحفظ الودايع", "فرده علي وقت حضور موتي", "رضيت بالله ربا",
        "وبمحمد صلي الله عليه واله نبيا", "وبالاسلام دينا وبالقران كتابا", "وبالكعبه قبله وبعلي وليا واماما",
        "وبالحسن والحسين وعلي بن الحسين", "ومحمد بن علي وجعفر بن محمد", "وموسي بن جعفر وعلي بن موسي",
        "ومحمد بن علي وعلي بن محمد", "والحسن بن علي والحجه بن الحسن", "صلوات الله عليهم ايمه",
        "اللهم اني رضيت بهم ايمه فارضني لهم", "انك علي كل شيء قدير",
    ],
    "mafatih-dua-asharat": [
        "سبحان الله والحمد لله ولا اله الا الله والله اكبر", "ولا حول ولا قوه الا بالله العلي العظيم",
        "سبحان الله اناء الليل واطراف النهار", "سبحان الله بالغدو والاصال", "سبحان الله بالعشي والابكار",
        "سبحان الله حين تمسون وحين تصبحون", "وله الحمد في السماوات والارض وعشيا وحين تظهرون",
        "يخرج الحي من الميت ويخرج الميت من الحي", "ويحيي الارض بعد موتها وكذلك تخرجون",
        "سبحان ربك رب العزه عما يصفون", "وسلام علي المرسلين", "والحمد لله رب العالمين",
        "الملك الحق المهيمن المبين القدوس", "فصل علي محمد واله", "واتمم علي نعمتك وخيرك وبركاتك وعافيتك",
        "بنجاه من النار", "اللهم لك الحمد حمدا سرمدا ابدا", "لا انقطاع له ولا نفاد", "ولك ينبغي واليك ينتهي",
        # the closing praise and the repeated phrases
        "حمدا كثيرا طيبا مباركا فيه", "كما تحب ربنا وترضي", "وكما ينبغي لكرم وجهك وعز جلالك",
        "لا اله الا الله وحده لا شريك له", "له الملك وله الحمد وهو اللطيف الخبير",
        "لا اله الا الله وحده لا شريك له", "له الملك وله الحمد يحيي ويميت ويميت ويحيي",
        "وهو حي لا يموت بيده الخير", "وهو علي كل شيء قدير",
        "استغفر الله الذي لا اله الا هو الحي القيوم واتوب اليه",
        "اللهم اصنع بي ما انت اهله", "ولا تصنع بي ما انا اهله", "فانك اهل التقوي واهل المغفره",
        "وانا اهل الذنوب والخطايا", "فارحمني يا مولاي وانت ارحم الراحمين",
        "لا حول ولا قوه الا بالله", "توكلت علي الحي الذي لا يموت", "والحمد لله الذي لم يتخذ ولدا",
        "ولم يكن له شريك في الملك", "ولم يكن له ولي من الذل وكبره تكبيرا",
    ],
    "mafatih-munajat-shakeen": ["الهي اليك اشكو نفسا بالسوء اماره", "والي الخطييه مبادره", "وبمعاصيك مولعه",
                                "ولسخطك متعرضه", "تسلك بي مسالك المهالك", "وتجعلني عندك اهون هالك",
                                "مملوه بالغفله والسهو", "تسرع بي الي الحوبه وتسوفني بالتوبه",
                                "وشيطانا يغويني", "قد ملا بالوسواس صدري", "واحاطت هواجسه بقلبي",
                                "يعاضد لي الهوي"],
    "mafatih-munajat-khaifeen": ["الهي هل تسود وجوها خرت ساجده لعظمتك",
                                 "او تخرس السنه نطقت بالثناء علي مجدك وجلالتك",
                                 "او تطبع علي قلوب انطوت علي محبتك",
                                 "نجني برحمتك من عذاب النار وفضيحه العار", "اذا امتاز الاخيار من الاشرار",
                                 "وحالت الاحوال وهالت الاهوال", "وقرب المحسنون وبعد المسييون",
                                 "ووفيت كل نفس ما كسبت وهم لا يظلمون"],
    "mafatih-munajat-rajeen": ["يا من اذا سيله عبد اعطاه", "واذا امل ما عنده بلغه مناه",
                               "واذا اقبل عليه قربه وادناه", "وكيف اومل سواك والخلق والامر لك",
                               "ااقطع رجايي منك", "وقد اوليتني ما لم اسيله من فضلك", "وحجابه مرفوع لراجيه",
                               "اسيلك بكرمك", "ان تمن علي من عطايك بما تقر به عيني"],
    "mafatih-munajat-ragibeen": ["الهي ان كان قل زادي في المسير اليك", "الهي استشفعت بك اليك", "واستجرت بك منك",
                                 "اتيتك طامعا في احسانك", "راغبا في امتنانك", "وافدا الي حضره جمالك",
                                 "مريدا وجهك طارقا بابك", "مستكينا لعظمتك وجلالك"],
    "mafatih-munajat-shakireen": ["الهي اذهلني عن اقامه شكرك تتابع طولك", "واعجزني عن احصاء ثنايك فيض فضلك",
                                  "الهي فكما غذيتنا بلطفك وربيتنا بصنعك", "فتمم علينا سوابغ النعم"],
    "mafatih-munajat-muteeen": ["اللهم الهمنا طاعتك", "وجنبنا معصيتك", "ويسر لنا بلوغ ما نتمني من ابتغاء رضوانك",
                                "واحللنا بحبوحه جنانك", "واقشع عن بصايرنا سحاب الارتياب",
                                "واكشف عن قلوبنا اغشيه المريه والحجاب",
                                "الهي اجعلني من المصطفين الاخيار", "والحقني بالصالحين الابرار",
                                "السابقين الي المكرمات المسارعين الي الخيرات", "العاملين للباقيات الصالحات",
                                "الساعين الي رفيع الدرجات", "انك علي كل شيء قدير وبالاجابه جدير",
                                "برحمتك يا ارحم الراحمين"],
    "mafatih-munajat-mureedeen": ["سبحانك ما اضيق الطرق علي من لم تكن دليله", "وما اوضح الحق عند من هديته سبيله"],
    "mafatih-munajat-muhibbeen": ["الهي من ذا الذي ذاق حلاوه محبتك فرام منك بدلا",
                                  "ومن ذا الذي انس بقربك فابتغي عنك حولا",
                                  "الهي فاجعلنا ممن اصطفيته لقربك وولايتك", "واخلصته لودك ومحبتك",
                                  "واعذته من هجرك وقلاك", "وبواته مقعد الصدق في جوارك",
                                  "وخصصته بمعرفتك واهلته لعبادتك", "وهيمت قلبه لارادتك واجتبيته لمشاهدتك",
                                  "واخليت وجهه لك وفرغت فواده لحبك", "ورغبته فيما عندك والهمته ذكرك",
                                  "واوزعته شكرك وشغلته بطاعتك", "وصيرته من صالحي بريتك", "واخترته لمناجاتك",
                                  "وقطعت عنه كل شيء يقطعه عنك", "اللهم اجعلنا ممن دابهم الارتياح اليك والحنين",
                                  "ودهرهم الزفره والانين", "جباههم ساجده لعظمتك", "وعيونهم ساهره في خدمتك",
                                  "ودموعهم سايله من خشيتك", "وقلوبهم متعلقه بمحبتك", "وافيدتهم منخلعه من مهابتك",
                                  "اسيلك حبك وحب من يحبك", "وحب كل عمل يوصلني الي قربك",
                                  "وان تجعلك احب الي مما سواك"],
    "mafatih-munajat-mutawassileen": ["الهي ليس لي وسيله اليك الا عواطف رافتك",
                                      "ولا لي ذريعه اليك الا عوارف رحمتك", "وحط طمعي بفناء جودك",
                                      "فحقق فيك املي", "وبواتهم دار كرامتك",
                                      "واقررت اعينهم بالنظر اليك يوم لقايك"],
    "mafatih-munajat-arifeen": ["الهي قصرت الالسن عن بلوغ ثنايك كما يليق بجلالك",
                                "وعجزت العقول عن ادراك كنه جمالك", "الهي فاجعلنا من الذين",
                                "ترسخت اشجار الشوق اليك في حدايق صدورهم", "وشرايع المصافات يردون",
                                "قد كشف الغطاء عن ابصارهم"],
    "mafatih-munajat-mutasimeen": ["اللهم يا ملاذ اللايذين ويا معاذ العايذين", "ويا منجي الهالكين ويا عاصم البايسين",
                                   "ويا حصن اللاجين", "ان لم اعذ بعزتك فبمن اعوذ", "الهي فلا تخلنا من حمايتك",
                                   "ولا تعرنا من رعايتك", "وذدنا عن موارد الهلكه", "فانا بعينك وفي كنفك ولك",
                                   "اسيلك باهل خاصتك من ملايكتك", "والصالحين من بريتك",
                                   "ان تجعل علينا واقيه تنجينا من الهلكات", "وتجنبنا من الافات",
                                   "وتكننا من دواهي المصيبات", "وان تنزل علينا من سكينتك",
                                   "وان تغشي وجوهنا بانوار محبتك", "وان تووينا الي شديد ركنك",
                                   "وان تحوينا في اكناف عصمتك", "برافتك ورحمتك يا ارحم الراحمين"],
    "mafatih-munajat-muftaqireen": ["وكربي لا يفرجه سوي رحمتك", "وضري لا يكشفه غير رافتك",
                                    "وغلتي لا يبردها الا وصلك", "ويا اكرم الاكرمين ويا ارحم الراحمين",
                                    "لك تخضعي وسوالي", "وتديم علي نعم امتنانك", "وها انا بباب كرمك واقف",
                                    "ولنفحات برك متعرض", "وبحبلك الشديد معتصم", "وبعروتك الوثقي متمسك"],
    "mafatih-munajat-dhakireen": ["الهي لو لا الواجب من قبول امرك", "لنزهتك من ذكري اياك",
                                  "علي ان ذكري لك بقدري لا بقدرك", "وما عسي ان يبلغ مقداري حتي اجعل محلا لتقديسك",
                                  "الهي فالهمنا ذكرك في الخلاء والملاء", "والليل والنهار والاعلان والاسرار",
                                  "وفي السراء والضراء", "وانسنا بالذكر الخفي", "ولا تسكن النفوس الا عند روياك",
                                  "انت المسبح في كل مكان", "والمعبود في كل زمان", "والموجود في كل اوان",
                                  "والمدعو بكل لسان", "والمعظم في كل جنان", "واستغفرك من كل لذه بغير ذكرك",
                                  "ومن كل راحه بغير انسك", "ومن كل سرور بغير قربك", "ومن كل شغل بغير طاعتك"],
    "mafatih-munajat-zahideen": ["الهي اسكنتنا دارا حفرت لنا حفر مكرها", "وعلقتنا بايدي المنايا في حبايل غدرها",
                                 "فانها المهلكه طلابها المتلفه حلالها", "المحشوه بالافات المشحونه بالنكبات",
                                 "الهي فزهدنا فيها", "وتول امورنا بحسن كفايتك", "واوفر مزيدنا من سعه رحمتك",
                                 "واجمل صلاتنا من فيض مواهبك", "واتمم لنا انوار معرفتك",
                                 "واذقنا حلاوه عفوك ولذه مغفرتك"],
}

FATHATAN, FATHA, DAMMA, KASRA, SHADDA, SUKUN = "\u064b", "\u064e", "\u064f", "\u0650", "\u0651", "\u0652"
HARAKAT = "\u064b-\u0652"  # tanween, fatha, damma, kasra, shadda, sukun
DIA = re.compile("[\u064b-\u065f\u0670\u0640\u06e1-\u06ed]")


def _nfc(w: str) -> str:
    """One order for a letter's marks (the decoder writes shadda+fatha and fatha+shadda)."""
    return unicodedata.normalize("NFC", w)


BROKEN = {(_nfc(a), _nfc(b)): v for (a, b), v in BROKEN.items()}
GLUED = {_nfc(k): v for k, v in GLUED.items()}
FIXED = {_nfc(k): v for k, v in FIXED.items()}
MAQSURA = {_nfc(w) for w in MAQSURA}


def _plain(w: str) -> str:
    """Letters only, as written (hamza seats, ى and ة kept)."""
    return DIA.sub("", re.sub("[^\u0600-\u06ff]", "", w))


def _bare(w: str) -> str:
    """Letters only, hamza seats folded, ى and ة kept apart from ي and ه."""
    return _plain(w).translate(str.maketrans("أإآٱ", "اااا"))


def corpus_vote() -> tuple[Counter, Counter]:
    """How often the other texts write each word: by _bare() form, and by _plain() form."""
    bare: Counter = Counter()
    plain: Counter = Counter()
    for p in DUAS.glob("*.json"):
        raw = json.loads(p.read_text(encoding="utf-8"))
        if raw["dua_id"].startswith("mafatih-"):
            continue
        for s in raw["segments"]:
            for w in s["arabic"].split():
                bare[_bare(w)] += 1
                plain[_plain(w)] += 1
    return bare, plain


def _votes(vote: Counter, stem: str, a: str, b: str) -> tuple[int, int]:
    """How often the corpus writes stem+a and stem+b; with a conjunction or preposition
    (و ف ب ل ك) taken off the front while neither is found."""
    for pre in ("", "و", "ف", "ب", "ل", "ك", "وب", "ول", "فب", "فل"):
        if stem.startswith(pre) and (vote[stem[len(pre):] + a] or vote[stem[len(pre):] + b]):
            return vote[stem[len(pre):] + a], vote[stem[len(pre):] + b]
    return 0, 0


def spell(w: str, votes: tuple[Counter, Counter]) -> str:
    """One word in the corpus's spelling."""
    bare, plain = votes
    if w in FIXED:
        return FIXED[w]
    # Madda written as shadda (and a fatha) on a long vowel before hamza, or a madda
    # there: رَجاَّئى, السُّوَّءِ, قآئِما, مآ -> رَجائى, السُّوءِ, قائِما, ما.
    w = re.sub(f"([اوي]){FATHA}?{SHADDA}(?=[ءئؤأ])", r"\1", w)
    w = re.sub("(?<=.)آ(?=[ءئؤ]|$)", "ا", w)
    w = re.sub(f"(?<=[^ل])ا{FATHA}?{SHADDA}", "ا", w)  # an alif takes no shadda (لاّ: the lam's)
    w = re.sub(f"ل{SUKUN}(?=[تثدذرزسشصضطظلن][{FATHA}{DAMMA}{KASRA}]?{SHADDA})", "ل", w)  # assimilated: الْنّار
    # Alif with sukun after the article is a hamza the decoder lost.
    pre = "^((?:وَ|فَ|بِ|وَبِ|فَبِ)?)الا"
    w = re.sub(pre + KASRA + SUKUN, r"\1الإِ", w)
    w = re.sub(pre + DAMMA + SUKUN, r"\1الأُ", w)
    w = re.sub(pre + FATHA + f"(?:{SUKUN})?(?=[^{HARAKAT}])", r"\1الأَ", w)
    w = re.sub(pre + SUKUN + f"(?=[^{HARAKAT}]ا|خِرَ)", r"\1الآ", w)  # الاْمال, الاْخِرَة -> الآمال, الآخِرَة
    w = re.sub(pre + SUKUN + f"(?=ي[^{HARAKAT}])", r"\1الإِ", w)  # الاْيمان -> الإِيمان
    w = re.sub(pre + SUKUN, r"\1الأَ", w)
    w = re.sub(f"(?<=[^ل]{FATHA})ا{SUKUN}", "أ" + SUKUN, w)  # رَاْفَتِكَ -> رَأْفَتِكَ
    # A bare first alif before a vowelled or long letter (not the article's lam) is a lost
    # madda, as wasl alifs come before a sukun: امالُ, امِليهِ -> آمالُ, آمِليهِ (not after وَ: وَاقِياً).
    w = re.sub(f"^ا(?=[\u0628-\u0643\u0645-\u064a]([{FATHA}{DAMMA}{KASRA}{SHADDA}]|ا))", "آ", w)
    # ى with a vowel mark or inside a word is a consonant ya, but not the fatha or
    # tanween the book sets on an alif maqsura at the end (اِلىَ, مَوْلىً).
    w = re.sub(f"(?<![ا{KASRA}])ى{FATHA}$", "ى", w)  # (after an alif or a kasra it is -ya: مَوْلاىَ, دُعِىَ)
    w = re.sub(f"ى(?=[{HARAKAT}\u0621-\u064a])(?!{FATHATAN}$)", "ي", w)
    if w.endswith("ى") and w not in MAQSURA:
        prev = w[-2] if len(w) > 1 else ""
        ya, alif = _votes(bare, _bare(w)[:-1], "ي", "ى")
        if prev == KASRA or (prev != FATHA and ya >= alif):  # after kasra; or no fatha and the corpus says ي
            w = w[:-1] + "ي"
    # Ta marbuta written as ha (قُوَّهَ, وِقايَهً), where the corpus only has the ة form; only
    # with fatha or tanween, which the pronoun -hu/-hi never carries.
    m = re.match(f"(.*[^ا])ه([{FATHATAN}-{FATHA}]{SHADDA}?)$", w)
    if m:
        ta, ha = _votes(bare, _bare(m.group(1)), "ة", "ه")
        if ta > 0 and ha == 0:
            w = m.group(1) + "ة" + m.group(2)
    if w in FIXED:  # a typo keyed by its spelt form
        return FIXED[w]
    return _hamza(w, plain)


def _hamza(w: str, plain: Counter) -> str:
    """The hamza seat on a word's first alif (after و ف ب ل ك): اَنْتَ, وَاِذا -> أَنْتَ, وَإِذا,
    where the corpus writes the word with one; not on a wasl verb (اِغْفِرْ)."""
    m = re.match(f"^(|وَ|فَ|بِ|لِ|كَ|وَبِ|وَلِ|فَبِ|فَلِ)ا([{FATHA}{DAMMA}{KASRA}])(.*)$", w)
    if not m:
        return w
    pre, v, rest = m.groups()
    seat = "إ" if v == KASRA else "أ"
    if v != KASRA and not re.match(f"ل[ل{SUKUN}{SHADDA}]", rest):
        use = True  # an alif with fatha or damma is a hamza, but for the article (and اَللّهُمَّ)
    else:  # إِ or a wasl verb (اِغْفِرْ), أَلْ or the article: as the corpus writes the word
        use = plain[_plain(seat + rest)] > plain[_plain("ا" + rest)]
    return pre + seat + v + rest if use else w


def tokens(block: dict, votes: tuple[Counter, Counter]) -> tuple[list[dict], set[int]]:
    """The text as words in the corpus's spelling, each with the book-word indices
    (normalize()d, as the harvest labels count them) it came from; and the token
    indices after which a line must break."""
    raw, n = [], 0
    for s in block["segments"]:
        for t in s["arabic"].split():
            k = len(normalize(t).split())
            raw.append({"t": _nfc(t), "old": list(range(n, n + k))})
            n += k
    out: list[dict] = []
    forced: set[int] = set()
    i = 0
    while i < len(raw):
        ins = next((p for p in INSTRUCTIONS if [r["t"] for r in raw[i:i + len(p)]] == p), None)
        if ins:
            if out:
                forced.add(len(out) - 1)
            i += len(ins)
            continue
        tok = dict(raw[i])
        pair = (raw[i]["t"], raw[i + 1]["t"]) if i + 1 < len(raw) else None
        if pair in BROKEN:
            tok = {"t": BROKEN[pair], "old": raw[i]["old"] + raw[i + 1]["old"]}
            i += 1
        i += 1
        for j, part in enumerate(GLUED.get(tok["t"], tok["t"]).split()):
            out.append({"t": part, "old": tok["old"], "glued": j > 0})
    # A separate وَ joins the next word, as everywhere else in the corpus.
    joined: list[dict] = []
    forced_j: set[int] = set()
    for k, tok in enumerate(out):
        if joined and joined[-1]["t"] in ("وَ", "و") and (len(joined) - 1) not in forced_j:
            joined[-1] = {"t": "وَ" + tok["t"], "old": joined[-1]["old"] + tok["old"], "glued": joined[-1]["glued"]}
        else:
            joined.append(dict(tok))
        if k in forced:
            forced_j.add(len(joined) - 1)
    for tok in joined:
        tok["t"] = spell(tok["t"], votes)
    return joined, forced_j


PAUSE_S = 0.35  # a pause: a breath or a stop, not the gap between two words
STOP_SHARE = 0.25  # a recording's longest pauses that count as stops (lines are ~5 words)
BREAK_AT = 0.5  # share of reciters stopping above which a break pays
# Words a line doesn't end on (_bare() forms, also after و/ف): particles, a construct's
# first noun, and the calls that open a phrase.
NO_END = {"يا", "لا", "من", "عن", "على", "علي", "الى", "في", "الا", "ان", "لم", "لن", "قد", "ما", "ثم", "او", "ام",
          "بل", "حتى", "اللهم", "الهي", "انا", "اني", "انك", "انه", "بي", "لي", "اله", "بن", "ابن", "كل", "عدد",
          "غير", "ذي", "ذو", "ذا"}
# Words that open a phrase: a break before them is a little likelier.
OPENS = {"يا", "ويا", "فيا", "الهي", "اللهم", "ام", "سيدي", "سبحان"}


def _no_end(w: str) -> bool:
    b = _bare(w)
    return b in NO_END or (b[:1] in "وف" and b[1:] in NO_END - {"بي", "لي"})


def pauses(blocks: list[str]) -> tuple[dict, dict]:
    """block -> book word index -> one flag per recording that read it and the next word
    in confidently placed lines: did it stop there, i.e. a pause among the longest
    quarter of its pauses in this text (slow, melodic readers breathe after every other
    word); and block -> recordings."""
    held = {(r["source"], r["rec"]) for r in json.loads(TEST_VOICES.read_text())["recordings"]}
    want = set(blocks)
    out: dict = {b: defaultdict(list) for b in blocks}
    recs: dict = defaultdict(set)
    words = {}
    for b in json.loads(EXTRA.read_text(encoding="utf-8")):
        if b["dua_id"] in want:
            off, w = {}, []
            for s in b["segments"]:
                off[s["segment_id"]] = len(w)
                w += normalize(s["arabic"]).split()
            words[b["dua_id"]] = (off, w)
    for p in LABELS.glob("*/*.json"):
        if p.name.endswith(".captions.json"):
            continue
        lab = json.loads(p.read_text(encoding="utf-8"))
        for sp in lab.get("spans", []):
            b = sp.get("dua")
            if b not in want or (lab["platform"], lab["id"]) in held:
                continue
            recs[b].add((lab["platform"], lab["id"]))
            off, w = words[b]
            seq = []
            for ln in sp["lines"]:
                o = off.get(ln["seg"])
                if not ln["ok"] or o is None or w[o:o + len(ln["words"])] != ln["words"]:
                    continue
                seq += [(o + j, a, e) for j, (a, e) in enumerate(ln["times"])]
            seq.sort(key=lambda x: x[1])
            g = [(i, a2 - e) for (i, _, e), (i2, a2, _) in zip(seq, seq[1:]) if i2 == i + 1 and 0 <= a2 - e < 30]
            if len(g) < 20:
                continue
            cut = max(PAUSE_S, sorted(x for _, x in g)[int(len(g) * (1 - STOP_SHARE))])
            for i, x in g:
                out[b][i].append(x >= cut)
    return out, recs


def length_cost(n: int) -> float:
    return {1: 3.0, 2: 1.2, 3: 0.3}.get(n, 0.0) + 0.3 * max(0, n - 9)


def _manual(toks: list[dict], manual: list[str]) -> list[tuple[int, int]]:
    """The token span [s, e) of each hand-set line, found in order."""
    words = [normalize(t["t"]) for t in toks]
    out, pos = [], 0
    for line in manual:
        want = normalize(line).split()
        s = next((s for s in range(pos, len(words) - len(want) + 1) if words[s:s + len(want)] == want), None)
        if s is None:
            raise SystemExit(f"hand-set line not in the text (after word {pos}): {line}")
        out.append((s, s + len(want)))
        pos = s + len(want)
    return out


def lines(toks: list[dict], forced: set[int], gaps: dict, manual: list[str]) -> list[list[int]]:
    """Cut the words into lines: maximise the stop shares at the breaks less the cost of
    lines too short or too long."""
    m = len(toks)
    forced = set(forced)
    shares: list[float | None] = []
    for tok in toks:
        g = gaps.get(max(tok["old"]), []) if tok["old"] else []
        shares.append(sum(g) / len(g) if len(g) >= 5 else None)
    # Where no recording was placed confidently the rhyme decides: a word ending like the
    # words reciters do stop after (saj').
    rhyme: dict = defaultdict(list)
    for tok, sh in zip(toks, shares):
        if sh is not None:
            rhyme[_bare(tok["t"])[-3:]].append(sh)
    gain: list[float | None] = []
    for i, tok in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < m else None
        if nxt is not None and nxt["glued"]:
            gain.append(None)  # inside a word the decoder glued: one book word
            continue
        share = shares[i]
        if share is None:
            r = rhyme.get(_bare(tok["t"])[-3:], [])
            share = sum(r) / len(r) if len(r) >= 3 else 0.5
        if _no_end(tok["t"]):
            share -= 1.0  # reciters often hold these, but a line doesn't end on them
        if nxt is not None and _bare(nxt["t"]) in OPENS and not (i > 0 and _bare(toks[i - 1]["t"]) == "يا"):
            share += 0.15  # (not between repeated calls: يا كَريمُ يا كَريمُ)
        gain.append(share - BREAK_AT)
    for s, e in _manual(toks, manual):
        if s > 0:
            forced.add(s - 1)
        forced.add(e - 1)
        for k in range(s, e - 1):
            gain[k] = None
    best = [0.0] + [float("-inf")] * m
    back = [0] * (m + 1)
    for e in range(1, m + 1):  # a line of words [s, e)
        for s in range(e - 1, max(-1, e - 25), -1):
            if any(k in forced for k in range(s, e - 1)):
                break
            if s > 0 and gain[s - 1] is None:
                continue
            v = best[s] + (gain[s - 1] if s > 0 else 0.0) - length_cost(e - s)
            if v > best[e]:
                best[e], back[e] = v, s
    out, e = [], m
    while e > 0:
        out.append(list(range(back[e], e)))
        e = back[e]
    return out[::-1]


def build(gaps: dict, recs: dict, write: bool) -> None:
    blocks = {b["dua_id"]: b for b in json.loads(EXTRA.read_text(encoding="utf-8"))}
    votes = corpus_vote()
    for block_id, dua_id, en, ar in TEXTS:
        toks, forced = tokens(blocks[block_id], votes)
        cut = lines(toks, forced, gaps[block_id], MANUAL.get(dua_id, []))
        segs = [{"segment_id": i + 1, "arabic": " ".join(toks[k]["t"] for k in ln)} for i, ln in enumerate(cut)]
        n = [len(ln) for ln in cut]
        print(f"== {dua_id} ({en}): {len(toks)} words, {len(segs)} lines (median {sorted(n)[len(n) // 2]}, "
              f"max {max(n)}), pauses from {len(recs[block_id])} recordings")
        for s in segs:
            print(f"  {s['segment_id']:3d}  {s['arabic']}")
        if write:
            (DUAS / f"{dua_id}.json").write_text(json.dumps({
                "dua_id": dua_id, "dua_name_en": en, "dua_name_ar": ar,
                "source": "Mafatih al-Jinan (github.com/kazemcodes/MafatihDecoder); lines from reciters' pauses",
                "segments": segs}, ensure_ascii=False, indent=1), encoding="utf-8")


def book_words() -> dict[str, dict]:
    """Mafatih block -> {"dua": our id, "words": the book's words and "off": each book
    line's first (counted as the harvest labels count them), "local": book word -> the
    du'a's own word index (as CorpusIndex counts them)}; for carrying the harvest's
    alignments of these texts over to the corpus's lines (bench.py)."""
    blocks = {b["dua_id"]: b for b in json.loads(EXTRA.read_text(encoding="utf-8"))}
    votes = corpus_vote()
    out = {}
    for block_id, dua_id, *_ in TEXTS:
        toks, _ = tokens(blocks[block_id], votes)
        saved = json.loads((DUAS / f"{dua_id}.json").read_text(encoding="utf-8"))
        if [w for s in saved["segments"] for w in s["arabic"].split()] != [t["t"] for t in toks]:
            raise SystemExit(f"{dua_id}: data/duas differs from this script's text; rerun it")
        off, words = {}, []
        for s in blocks[block_id]["segments"]:
            off[s["segment_id"]] = len(words)
            words += normalize(s["arabic"]).split()
        local, n = {}, 0
        for tok in toks:
            for b in tok["old"]:
                local.setdefault(b, n)
            n += len(normalize(tok["t"]).split())
        out[block_id] = {"dua": dua_id, "words": words, "off": off, "local": local}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build(*pauses([t[0] for t in TEXTS]), write=not args.dry_run)


if __name__ == "__main__":
    main()
