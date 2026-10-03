"""Search queries for the recitation harvest (scripts/harvest.py).

Titles only decide what gets *downloaded*; which text a recording holds is
decided later from its audio (harvest.py identify), so the queries can be
broad. Four groups:

  * every corpus text by its own Arabic and English name;
  * the widely recited du'as and ziyarat under the names people title them
    with in Arabic, Persian, Urdu-style transliteration and English;
  * the daily Ramadan du'as and the Sahifa by number;
  * ordinary voices: children, families, gatherings, phone recordings.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (Arabic, Persian, transliterations...) per widely recited text.
POPULAR: list[list[str]] = [
    ["دعاء كميل", "دعای کمیل", "dua kumayl", "dua kumail", "dua e kumail"],
    ["دعاء التوسل", "دعای توسل", "dua tawassul", "dua e tawassul"],
    ["دعاء الندبة", "دعای ندبه", "dua nudba", "dua e nudba"],
    ["دعاء العهد", "دعای عهد", "dua ahad", "dua e ahad"],
    ["دعاء الفرج الهي عظم البلاء", "دعای فرج الهی عظم البلا", "dua faraj", "dua e faraj ilahi azumal bala"],
    ["دعاء الافتتاح", "دعای افتتاح", "dua iftitah", "dua e iftitah"],
    ["دعاء ابي حمزة الثمالي", "دعای ابوحمزه ثمالی", "dua abu hamza thumali", "dua abu hamza sumali"],
    ["دعاء الجوشن الكبير", "دعای جوشن کبیر", "dua jawshan kabir", "dua e joshan kabeer"],
    ["دعاء الجوشن الصغير", "دعای جوشن صغیر", "dua jawshan saghir", "dua e joshan sagheer"],
    ["دعاء الصباح", "دعای صباح", "dua sabah", "dua e sabah"],
    ["دعاء السمات", "دعای سمات", "dua simat", "dua e samaat"],
    ["دعاء عرفة الامام الحسين", "دعای عرفه امام حسین", "dua arafah imam hussain", "dua e arafa"],
    ["دعاء عرفة الامام السجاد", "دعای عرفه امام سجاد", "sahifa sajjadiya dua 47 arafah"],
    ["دعاء مكارم الاخلاق", "دعای مکارم الاخلاق", "dua makarim al akhlaq"],
    ["دعاء المشلول", "دعای مشلول", "dua mashlool", "dua e mashlool"],
    ["دعاء المجير", "دعای مجیر", "dua mujeer", "dua e mujeer"],
    ["دعاء يستشير", "دعای یستشیر", "dua yastashir"],
    ["دعاء يا من تحل به عقد المكاره", "دعای یا من تحل به عقد المکاره"],
    ["دعاء النور", "دعای نور حضرت زهرا", "dua noor"],
    ["دعاء العديلة", "دعای عدیله", "dua adeela"],
    ["دعاء البهاء", "دعای بهاء سحر", "dua baha", "dua sahar ramadan"],
    ["دعاء السحر", "دعای سحر", "dua sahar"],
    ["دعاء ادريس", "دعای ادریس", "dua nabi idris"],
    ["دعاء الحجة اللهم كن لوليك", "دعای سلامتی امام زمان", "dua salamati imam zamana", "allahumma kun li waliyyika"],
    ["دعاء يا مفزعي عند كربتي", "dua ya mafzai"],
    ["دعاء يا عدتي", "dua ya uddati"],
    ["دعاء علقمة", "دعای علقمه", "dua alqama"],
    ["دعاء يا شاهد كل نجوى", "dua ya shahida kulli najwa"],
    ["دعاء ام داود", "دعای ام داود", "dua umme dawood"],
    ["دعاء اللهم ادخل على اهل القبور السرور", "dua allahumma adkhil ala ahlil quboor"],
    ["دعاء اللهم لك صمت", "dua iftar allahumma laka sumtu"],
    ["دعاء الوداع شهر رمضان", "دعای وداع ماه رمضان", "dua farewell ramadan"],
    ["دعاء رفع المصاحف ليلة القدر", "دعای قرآن به سر گرفتن", "quran sar gereftan", "dua laylatul qadr quran on head"],
    ["دعاء يا علي يا عظيم", "دعای یا علی یا عظیم", "dua ya aliyu ya azeem"],
    ["دعاء اللهم ارزقني توفيق الطاعة", "dua allahummar zuqni"],
    ["دعاء اللهم يا من يملك حوائج السائلين", "دعای اللهم یا من یملک"],
    ["دعاء التوبة الصحيفة السجادية", "dua tawba sahifa"],
    ["دعاء الامام علي في مسجد الكوفة", "مناجات امیرالمومنین در مسجد کوفه", "munajat masjid kufa", "munajat imam ali masjid e kufa"],
    ["المناجاة الشعبانية", "مناجات شعبانیه", "munajat shabaniya", "munajat e shabania"],
    ["مناجاة التائبين", "مناجات تائبین", "munajat taibeen"],
    ["المناجاة الخمس عشرة", "مناجات خمس عشر", "munajat khamsa ashar"],
    ["حديث الكساء", "حدیث کسا", "hadith kisa", "hadees e kisa"],
    ["دعاء يوم الجمعة", "دعای روز جمعه", "dua jummah"],
    ["دعاء يوم السبت", "دعای روز شنبه", "dua saturday"],
    ["دعاء يوم الاحد", "دعای روز یکشنبه", "dua sunday"],
    ["دعاء يوم الاثنين", "دعای روز دوشنبه", "dua monday"],
    ["دعاء يوم الثلاثاء", "دعای روز سه شنبه", "dua tuesday"],
    ["دعاء يوم الاربعاء", "دعای روز چهارشنبه", "dua wednesday"],
    ["دعاء يوم الخميس", "دعای روز پنجشنبه", "dua thursday"],
    ["دعاء الحج", "dua al hajj"],
    ["دعاء الوحدة", "dua wahda"],
    ["تسبيح السحور", "tasbih sahar"],
    ["الصلوات الشعبانية", "صلوات شعبانیه", "salawat shabaniya"],
    ["زيارة عاشوراء", "زیارت عاشورا", "ziyarat ashura", "ziarat e ashura"],
    ["زيارة عاشوراء غير المعروفة", "زیارت عاشورای غیر معروفه", "ziyarat ashura ghair maroof"],
    ["زيارة وارث", "زیارت وارث", "ziyarat warith", "ziarat e waris"],
    ["زيارة امين الله", "زیارت امین الله", "ziyarat aminallah", "ziarat e ameenullah"],
    ["زيارة آل ياسين", "زیارت آل یاسین", "ziyarat aale yasin", "ziarat e aal e yaseen"],
    ["الزيارة الجامعة الكبيرة", "زیارت جامعه کبیره", "ziyarat jamia kabira", "ziarat e jamia kabeera"],
    ["زيارة الجامعة الصغيرة", "زیارت جامعه صغیره"],
    ["زيارة الامام الرضا", "زیارت امام رضا", "ziyarat imam reza"],
    ["زيارة الامام الحسين يوم عرفة", "زیارت امام حسین روز عرفه"],
    ["زيارة الامام الحسين ليلة القدر", "زیارت امام حسین شب قدر"],
    ["زيارة الامام الحسين في العيد", "زیارت امام حسین عید"],
    ["زيارة النصف من شعبان", "زیارت امام حسین نیمه شعبان"],
    ["زيارة رجبية", "زیارت رجبیه"],
    ["زيارة الامام علي يوم الغدير", "زیارت امین الله غدیر", "ziyarat ghadeer"],
    ["زيارة النبي محمد", "زیارت پیامبر", "ziyarat prophet muhammad"],
    ["زيارة الامام الصادق", "زیارت امام صادق"],
    ["زيارة الامام الباقر", "زیارت امام باقر"],
    ["زيارة الامام المهدي", "زیارت امام زمان", "ziyarat imam mahdi"],
    ["الاستغاثة بالامام المهدي", "استغاثه به امام زمان", "istighasa imam mahdi"],
    ["زيارة مسلم بن عقيل", "زیارت مسلم بن عقیل", "ziyarat muslim ibn aqeel"],
    ["زيارة هاني بن عروة"],
    ["زيارة السيدة خديجة", "زیارت حضرت خدیجه"],
    ["زيارة العباس", "زیارت حضرت عباس", "ziyarat abbas"],
    ["زيارة الشهداء", "زیارت شهدای کربلا"],
    ["زيارة يوم السبت", "زیارت روز شنبه"],
    ["زيارة يوم الاحد", "زیارت روز یکشنبه"],
    ["زيارة يوم الاثنين", "زیارت روز دوشنبه"],
    ["زيارة يوم الثلاثاء", "زیارت روز سه شنبه"],
    ["زيارة يوم الاربعاء", "زیارت روز چهارشنبه"],
    ["زيارة يوم الخميس", "زیارت روز پنجشنبه"],
    ["زيارة يوم الجمعة صاحب الزمان", "زیارت روز جمعه امام زمان", "ziyarat friday imam zamana"],
    ["اذن الدخول للحرم", "اذن دخول حرم"],
    ["دعاء الاستخارة الصحيفة", "دعاء ختم القرآن الصحيفة السجادية"],
    ["دعاء ناد علي", "ناد علی", "nade ali", "naad e ali"],
    ["آية الكرسي", "آیت الکرسی", "ayatul kursi"],
    ["صلاة جعفر الطيار", "نماز جعفر طیار", "salat jafar tayyar"],
    ["صلاة الغفيلة", "نماز غفیله"],
    ["صلاة الليل ادعية", "نماز شب دعا", "namaz e shab dua"],
    ["دعاء رؤية الهلال", "دعای رویت هلال"],
    ["دعاء ليلة النصف من شعبان", "دعای شب نیمه شعبان", "dua 15 shaban"],
    ["دعاء اليوم السابع والعشرين من رجب"],
    ["ادعية شهر رجب اللهم يا من ارجوه لكل خير", "دعای رجب یا من ارجوه", "dua rajab ya man arjuhu"],
    ["دعاء كل يوم من شهر رجب", "دعای ماه رجب"],
    ["ادعية شهر رمضان اليومية", "دعای روزهای ماه رمضان", "ramadan daily duas"],
    ["دعاء ليالي القدر", "دعای شب قدر", "dua laylatul qadr"],
    ["دعاء ايام شهر رمضان بصوت", "اللهم اني اسالك من فضلك باحسنه"],
    ["دعاء عيد الفطر", "دعای عید فطر", "eid dua"],
    ["تكبيرات العيد", "تکبیرات عید", "eid takbeer"],
    ["دعاء الامام الكاظم اول شهر رمضان"],
    ["دعاء العشرات", "دعای عشرات", "dua asharat"],
    ["دعاء الحزين", "dua hazeen"],
    ["دعاء صنمي قريش", "dua sanamay quraish"],
    ["دعاء الحريق فاطمة الزهراء"],
    ["تسبيح الزهراء", "تسبیحات حضرت زهرا"],
    ["تعقيبات صلاة المغرب", "تعقیبات نماز مغرب"],
    ["تعقيبات صلاة الفجر", "تعقیبات نماز صبح"],
    ["ادعية الصحيفة الفاطمية", "صحیفه فاطمیه", "sahifa fatimiya"],
    ["الصحيفة العلوية", "صحیفه علویه"],
    ["الصحيفة السجادية كاملة", "صحیفه سجادیه", "sahifa sajjadiya"],
    ["الصحيفة المهدية"],
    ["دعاء التحميد لله الصحيفة السجادية", "الصلاة على محمد وآله الصحيفة السجادية"],
    ["دعاء الامام الحسين يوم عاشوراء"],
    ["دعاء اول محرم", "دعای اول محرم"],
    ["اعمال يوم المباهلة دعاء", "دعای مباهله", "dua mubahila"],
    ["اعمال يوم عرفة", "اعمال روز عرفه"],
    ["دعاء دخول المسجد الحرام", "دعای ورود به مسجد الحرام"],
    ["دعاء الاحرام", "دعای احرام"],
    ["دعاء ختم القرآن", "دعای ختم قرآن"],
    ["دعاء بدء تلاوة القرآن"],
    ["دعاء شهر صفر", "دعای ماه صفر"],
    ["دعاء اول شهر ربيع الاخر"],
    ["دعاء ليلة الجمعة", "دعای شب جمعه", "thursday night dua"],
    ["مفاتيح الجنان", "مفاتیح الجنان", "mafatih al jinan"],
]

# Sahifa Sajjadiya numbering as YouTube titles write it.
AR_ORDINALS = [
    "الأول", "الثاني", "الثالث", "الرابع", "الخامس", "السادس", "السابع", "الثامن", "التاسع", "العاشر",
    "الحادي عشر", "الثاني عشر", "الثالث عشر", "الرابع عشر", "الخامس عشر", "السادس عشر", "السابع عشر",
    "الثامن عشر", "التاسع عشر", "العشرين", "الحادي والعشرين", "الثاني والعشرين", "الثالث والعشرين",
    "الرابع والعشرين", "الخامس والعشرين", "السادس والعشرين", "السابع والعشرين", "الثامن والعشرين",
    "التاسع والعشرين", "الثلاثين",
]
FA_ORDINALS = [
    "اول", "دوم", "سوم", "چهارم", "پنجم", "ششم", "هفتم", "هشتم", "نهم", "دهم", "یازدهم", "دوازدهم",
    "سیزدهم", "چهاردهم", "پانزدهم", "شانزدهم", "هفدهم", "هجدهم", "نوزدهم", "بیستم", "بیست و یکم",
    "بیست و دوم", "بیست و سوم", "بیست و چهارم", "بیست و پنجم", "بیست و ششم", "بیست و هفتم",
    "بیست و هشتم", "بیست و نهم", "سی ام",
]

# Du'a reciters as titles name them. Test reciters (splits.TEST_RECITERS) are
# NOT here: harvest.py's filter drops any upload naming them anyway.
RECITERS = [
    "باسم الكربلائي", "مهدي سماواتي", "مهدی سماواتی", "منصور ارضی", "حسین سازور", "محمود کریمی",
    "عبدالرضا هلالی", "سعید حدادیان", "میثم مطیعی", "حسین طاهری", "محمدحسین پویانفر", "مهدی میرداماد",
    "نزار القطري", "جليل الكربلائي", "علي بوحمد", "ميثم التمار", "حيدر البياتي", "محمد الحجيرات",
    "صالح الدرازي", "مهدي العوامي", "علي الحمادي", "حسين الخياط", "علي الساعدي", "عمار الكناني",
    "محمد باقر الخاقاني", "مصطفى الصراف", "عبد الحي آل قمبر", "علي فاني", "أسامة العطار",
    "محمد الطيب", "ملا علي الكوفي", "حمزة الصغير", "عادل الكربلائي", "مرتضى الحائري", "مهدي الحلي",
    "احمد الساعدي", "حسين فخري", "ابوالفضل بختیاری", "کریم منصوری", "حاج منصور", "حسن خلج",
    "Hasan Khalaj", "Basim Karbalaei", "Mahdi Samavati", "Meysam Motiee", "Nizar Al Qatari",
    "Syed Raza Abbas Zaidi", "Mesum Abbas", "Shadman Raza", "Ali Safdar", "Nadeem Sarwar",
]
RECITER_DUAS = ["دعاء كميل", "دعاء التوسل", "زيارة عاشوراء", "دعاء الندبة", "دعاء الافتتاح", "دعاء العهد",
                "دعای کمیل", "دعای توسل", "زیارت عاشورا", "دعای ندبه", "دعای عهد", "dua kumayl"]

# Ordinary voices: children, families, gatherings, phones.
ORDINARY = [
    "طفل يقرأ دعاء كميل", "طفل يقرأ دعاء الفرج", "طفل يقرأ زيارة عاشوراء", "طفلة تقرأ دعاء",
    "طفل يقرأ دعاء العهد", "طفل يقرأ دعاء التوسل", "طفل يقرأ حديث الكساء", "طفل يقرأ الدعاء بصوت جميل",
    "بصوت طفل دعاء", "بصوت الأطفال دعاء", "بصوت طفلة زيارة", "دعاء بصوت طفل صغير",
    "کودک دعای فرج", "کودک دعای عهد", "کودک زیارت عاشورا", "کودک دعای توسل", "کودک دعای کمیل",
    "دختر بچه دعای فرج", "پسر بچه دعای کمیل", "نوجوان دعای کمیل", "قرائت دعا توسط کودک",
    "kid reciting dua kumail", "child reciting dua faraj", "kids reciting ziyarat ashura",
    "little girl reciting dua", "child reciting hadees e kisa", "kid recites dua tawassul",
    "my son reciting dua", "my daughter reciting dua", "child recites dua ahad",
    "دعاء كميل في البيت", "دعاء التوسل في البيت", "قراءة دعاء كميل جماعي", "دعاء كميل في المسجد",
    "دعای کمیل در خانه", "دعای توسل خانگی", "dua kumail at home", "dua tawassul home recitation",
    "dua kumail majlis live", "dua tawassul majlis", "ziyarat ashura majlis live", "hadees e kisa majlis",
    "dua e kumail recited by", "dua tawassul recited by", "ziarat e ashura recited by",
    "قراءة دعاء الفرج", "قراءة زيارة عاشوراء", "قراءة دعاء العهد", "قراءة حديث الكساء",
    "دعاء كميل بصوت عادي", "تلاوة دعاء", "دعاء مقروء",
]


def _strip(text: str) -> str:
    text = re.sub(r"[ً-ْٰـ‎‏]", "", text or "")
    return re.sub(r"\s+", " ", text).strip()


def corpus_names() -> list[tuple[str, str, str]]:
    """(id, english, arabic) for every corpus text."""
    out = []
    for f in sorted((ROOT / "data" / "duas").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out.append((d["dua_id"], d.get("dua_name_en") or "", _strip(d.get("dua_name_ar") or "")))
    return out


def all_queries() -> list[tuple[str, str]]:
    """(group, query), most productive first, without duplicates."""
    qs: list[tuple[str, str]] = []
    for names in POPULAR:
        for n in names:
            qs.append(("popular", n))
        qs.append(("popular", f"{names[0]} كامل"))
        qs.append(("popular", f"{names[0]} بصوت"))
    for i, (ar, fa) in enumerate(zip(AR_ORDINALS, FA_ORDINALS), 1):
        qs.append(("ramadan", f"دعاء اليوم {ar} من شهر رمضان"))
        qs.append(("ramadan", f"دعای روز {fa} ماه رمضان"))
        qs.append(("ramadan", f"ramadan day {i} dua"))
        qs.append(("ramadan", f"دعاء ليلة {i} من شهر رمضان"))
    for i in range(1, 55):
        qs.append(("sahifa", f"الصحيفة السجادية الدعاء {i}"))
        qs.append(("sahifa", f"sahifa sajjadiya dua {i}"))
    for r in RECITERS:
        for d in RECITER_DUAS[:6] if not r.isascii() else RECITER_DUAS[-1:]:
            qs.append(("reciter", f"{d} {r}"))
        qs.append(("reciter", f"{r} دعاء" if not r.isascii() else f"{r} dua"))
    for q in ORDINARY:
        qs.append(("ordinary", q))
    for dua_id, en, ar in corpus_names():
        if ar and len(ar) >= 6:
            qs.append(("corpus", ar))
        en = re.sub(r"\s*·.*", "", en)
        if en:
            low = en.lower()
            word = "" if any(w in low for w in ("dua", "ziyar", "salaat", "salat", "munajat", "sahifa", "tasbih", "hadith")) else " dua"
            qs.append(("corpus", f"{en}{word}"))
    seen, out = set(), []
    for g, q in qs:
        k = q.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append((g, q.strip()))
    return out


if __name__ == "__main__":
    import collections
    qs = all_queries()
    print(len(qs), collections.Counter(g for g, _ in qs))
