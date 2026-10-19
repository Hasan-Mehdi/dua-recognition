#!/usr/bin/env python
"""The du'as said after the daily prayers (ta'qibat), one text per prayer.

Asked for from duas.org's "Dua'a to be recited after every Namaz" (pdfs/Duaaftersalat.pdf):
the corpus had 8 of its ~55 du'as, all from duas.org's Maghrib/Isha page (docs/research/
after-salat-2026-10-10.md). Reciters read a prayer's ta'qibat as one run, in Mafatih
al-Jinan's order, and the harvest's recordings are cut that way, so each prayer's du'as
the corpus lacks become one text in that order. Du'as the corpus already has as texts
of their own (Dua Wahda, the Maghrib/Isha ones) stay out: two texts holding the same
lines would compete for every reading of them.

Source: duas.org's ta'qibat pages (Arabic, reading and English per line):
mobile/common-taqibat, fajr-taqibat, zuhr-asr-taqibat, sajda-shukr (saved in
data/duasorg_extra/taqibat/, local like the rest of duas.org's English), and the short
blocks of the corpus's own Maghrib/Isha page (data/duasorg/, fetch_duasorg.py) that
went in as no text. Two lines duas.org lacks come from Mafatih (HAND). Lines longer
than LONG words are cut at clause openings (their English stays on the first piece;
the reading the app shows is made from the vowelled Arabic either way). A du'a said
several times is written once, its count before its English.

    python scripts/taqibat_corpus.py                  # -> data/duas/<id>.json + data/lines/<id>.json
    python scripts/taqibat_corpus.py --dry-run        # print only
    python scripts/taqibat_corpus.py --out DIR --lines-out DIR
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.text import normalize  # noqa: E402

DUAS = ROOT / "data" / "duas"
LINES = ROOT / "data" / "lines"  # corpus.LINES_DIR
PAGES = ROOT / "data" / "duasorg_extra" / "taqibat"
MI_PAGE = ROOT / "data" / "duasorg" / "dua-taqeebat-after-maghrib-isha-prayers.json"
SITE = "https://www.duas.org/mobile/{}.html"
LONG = 12  # words: a longer line is cut

TIMES = {3: "(Three times)", 7: "(Seven times)", 10: "(Ten times)", 100: "(A hundred times)"}

# Lines duas.org's pages lack: Mafatih al-Jinan, the end of the ta'qibat chapter (said ten
# times before sunrise and sunset); English from the PDF.
HAND = {
    "tahlil": [
        ("لَا إِلٰهَ إِلَّا اللَّهُ وَحْدَهُ لَا شَرِيكَ لَهُ",
         "There is no god but Allah, the One, without any partner."),
        ("لَهُ الْمُلْكُ وَلَهُ الْحَمْدُ", "To Him belongs the Kingdom and to Him is due all praise."),
        ("يُحْيِي وَيُمِيتُ وَيُمِيتُ وَيُحْيِي", "He gives life and death, and death and life,"),
        ("وَهُوَ حَيٌّ لَا يَمُوتُ", "and He is the Living who does not die."),
        ("بِيَدِهِ الْخَيْرُ وَهُوَ عَلَىٰ كُلِّ شَيْءٍ قَدِيرٌ", "In His hand is the good, and He has power over all things."),
    ],
    "hamazat": [
        ("أَعُوذُ بِاللَّهِ السَّمِيعِ الْعَلِيمِ", "I seek refuge in Allah, the Hearing, the Knowing,"),
        ("مِنْ هَمَزَاتِ الشَّيَاطِينِ", "from the evil suggestions of the devils,"),
        ("وَأَعُوذُ بِاللَّهِ أَنْ يَحْضُرُونِ", "and I seek refuge in Allah from their presence."),
        ("إِنَّ اللَّهَ هُوَ السَّمِيعُ الْعَلِيمُ", "Surely Allah, He is the Hearing, the Knowing."),
    ],
}
# duas.org gives this line no English (keyed by its first words, normalized)
ENGLISH = {
    "سبحان الله العظيم وبحمده استغفر الله":
        "Glory be to Allah the Great, and praise be to Him; I ask Allah's forgiveness and ask Him of His bounty.",
}
# duas.org's Arabic, where it drops what its own reading and English have (keyed by the line, normalized)
FIXES = {
    # "wa tajma'u": "and the scattered are gathered"
    "واسالك باسمك الاعظم الذي به تقوم السموات والارض وبه تحيي الموتي وترزق الاحيا وتفرق بين المجتمع بين المتفرق "
    "وبه احصيت عدد الاجال ووزن الجبال وكيل البحار":
        "وَ اَسْاَلُكَ بِاسْمِكَ الْاَعْظَمِ الَّذِيْ بِهِ تَقُوْمُ السَّمٰوَاتُ وَ الْاَرْضُ وَ بِهٖ تُحْيِي الْمَوْتٰى "
        "وَ تَرْزُقُ الْاَحْيَاءَ وَ تُفَرِّقُ بَيْنَ الْمُجْتَمَعِ وَ تَجْمَعُ بَيْنَ الْمُتَفَرِّقِ "
        "وَ بِهٖ اَحْصَيْتَ عَدَدَ الْآجَالِ وَ وَزْنَ الْجِبَالِ وَ كِيْلَ الْبِحَارِ",
    "صلي الله عليه و": "صَلَّىٰ ٱللَّهُ عَلَيْهِ وَآلِهِ",  # reading: "salla allahu `alayhi wa alihi"
    "سجد وجهي اللييم لوجه ربي ال": "سَجَدَ وَجْهِيَ ٱللَّئِيمُ لِوَجْهِ رَبِّيَ ٱلْكَرِيمِ",  # "... rabbiya alkarimi"
}

# Each text: (page, the block's first words, times said). A page is a duas.org mobile page
# or "mi" (the corpus's Maghrib/Isha page), or "hand" (HAND). Order: Mafatih al-Jinan's,
# then the PDF's du'as from duas.org's pages that Mafatih doesn't list for that prayer.
TEXTS = [
    ("duasorg-taqibat-common", "Common Ta'qibat (after every prayer)", "التعقيبات المشتركة", [
        # Dua Wahda, which comes first, is dua-wahda
        ("common-taqibat", "استغفر الله الذي لا اله الا هو الحي القيوم واتوب", None),
        ("common-taqibat", "اللهم اهدني من عندك", None),
        ("common-taqibat", "اشهد ان لا اله الا الله وحده", 10),
        ("common-taqibat", "سبحان الله كلما سبح الله", None),
        # (Fatiha, Ayat al-Kursi, Shahida, the Mulk and Sakhra verses: Quran, left out)
        ("common-taqibat", "اللهم صل علي محمد وال محمد واجعل لي من امري", 3),
        ("common-taqibat", "يا رب محمد وال محمد صل", 7),
        ("common-taqibat", "يا ذا الجلال والاكرام صل", 3),
        ("common-taqibat", "اللهم اني اسالك باسمك المكنون", None),
        ("common-taqibat", "يا من لا يشغله سمع", None),
        # "Ilahi hadhihi salati" is duasorg-dua-taqeebat-after-maghrib-isha-prayers-10
        ("common-taqibat", "سبحان من لا يعتدي علي", None),
        ("common-taqibat", "اعيذ نفسي وديني", 3),
        ("common-taqibat", "اللهم ان مغفرتك ارجي من", None),
        ("common-taqibat", "اللهم انت السلام", None),
        ("common-taqibat", "رضيت بالله ربا", None),
        ("common-taqibat", "اللهم اني اسالك العفو", 3),
    ]),
    ("duasorg-taqibat-fajr", "Ta'qibat after Fajr", "تعقيب صلاة الصبح", [
        ("fajr-taqibat", "اللهم صل علي محمد وال محمد واهدني", None),
        ("fajr-taqibat", "اللهم صل علي محمد وال محمد الاوصياء", 10),
        ("fajr-taqibat", "اللهم احيني علي ما احييت", None),
        ("fajr-taqibat", "استغفر الله و اتوب اليه", 100),
        ("fajr-taqibat", "صلي الله علي محمد وال محمد سبحان", 100),
        ("fajr-taqibat", "اصبحت اللهم معتصما بذمامك", None),
        ("fajr-taqibat", "سبحان الله العظيم وبحمده ولا حول", 10),
        ("fajr-taqibat", "بسم الله الرحمن الرحيم لا حول", 10),
        # Imam al-Sadiq's "bihaqqi Muhammad" is duasorg-dua-taqeebat-after-maghrib-isha-prayers-3
        ("fajr-taqibat", "بسم الله وصلي الله", None),
        ("fajr-taqibat", "لا حول ولا قوه الا بالله توكلت", None),
        ("hand", "tahlil", 10),
        ("hand", "hamazat", 10),
        ("fajr-taqibat", "سبحان الله العظيم وبحمده استغفر", 10),  # Imam al-Kazim, in the PDF
        ("fajr-taqibat", "امنت بالله وحده لا شريك", None),  # the aqeeq, mornings
    ]),
    ("duasorg-taqibat-zuhr", "Ta'qibat after Zuhr", "تعقيب صلاة الظهر", [
        ("zuhr-asr-taqibat", "لا اله الا الله العظيم الحليم", None),
        ("zuhr-asr-taqibat", "بالله اعتصمت", 10),
        ("zuhr-asr-taqibat", "اللهم ان عظمت ذنوبي", None),
        ("zuhr-asr-taqibat", "اللهم رب السموات السبع", None),  # the PDF's 5th (duas.org lists it for Asr)
        ("zuhr-asr-taqibat", "اللهم اني اتقرب اليك بجودك وكرمك واتقرب اليك بمحمد عبدك", None),  # Imam al-Rida
    ]),
    ("duasorg-taqibat-asr", "Ta'qibat after Asr", "تعقيب صلاة العصر", [
        ("zuhr-asr-taqibat", "استغفر الله الذي لا اله الا هو الحي القيوم الرحمن", None),
        ("zuhr-asr-taqibat", "اللهم اني اعوذ بك من نفس", None),
        # (the PDF's "as'alukat tayyibat wa tarkal munkarat" is left out: duas.org's Arabic and
        # reading of it, "wa fakaka khiyarati", don't make sense and its English reads otherwise)
        ("zuhr-asr-taqibat", "بسم الله الرحمن الرحيم سبحان الله والحمد", None),
    ]),
    ("duasorg-taqibat-maghrib-isha", "Short Ta'qibat after Maghrib and Isha", "تعقيبات المغرب والعشاء", [
        ("mi", "ان الله وملايكته يصلون", None),
        ("mi", "ولا حول ولا قوه الا بالله العلي العظيم", 7),
        ("mi", "الحمد لله الذي يفعل ما يشاء", 3),
        # "Mujibati rahmatik" is duasorg-dua-taqeebat-after-maghrib-isha-prayers-1
        ("mi", "اللهم اجرنا من النار", None),
        ("mi", "سبحانك لا اله الا انت اغفر", None),
        ("mi", "فسبحن الله حين تمسون", 3),
        ("mi", "الحمد لله الذي يولج الليل", None),
        ("mi", "يا مذل كل جبار ومعز", None),  # after the Isha nafila
        ("mi", "اللهم عافنا من كل بلاء", None),
        ("mi", "يا رب محمد عجل فرج", 3),
    ]),
    ("duasorg-taqibat-sajda-shukr", "Sajdah al-Shukr", "سجدة الشكر", [
        ("sajda-shukr", "شكرا شكرا", None),
        ("sajda-shukr", "يا رب الارباب", None),
        ("sajda-shukr", "سبحانك اللهم انت ربي", None),
        ("sajda-shukr", "اعوذ بك من نار حرها", None),
        ("sajda-shukr", "يا رووف يا رحيم", None),
        ("sajda-shukr", "اسالك الراحه عند الموت", None),
        ("sajda-shukr", "سجد وجهي اللييم", None),
        ("sajda-shukr", "اني ظلمت نفسي فاغفر لي", 3),
        ("sajda-shukr", "اللهم مغفرتك اوسع من ذنوبي", None),
        ("sajda-shukr", "يا مذل كل جبار يا معز", None),
        ("sajda-shukr", "يا الله يا رباه يا سيداه", 3),
        ("sajda-shukr", "يا رب ماذا عليك", None),
        ("sajda-shukr", "عظم الذنب من عبدك", None),
        ("sajda-shukr", "الهي عبيدك بفنايك", None),
        ("sajda-shukr", "اللهم اني اشهدك واشهد ملايكتك", None),
        ("sajda-shukr", "اللهم اني انشدك بايوايك", 3),
    ]),
]

_FIXES = {k.replace(" ", ""): v for k, v in FIXES.items()}  # a separate وَ or not
GLUED = {"سَجَدوَجْهِيَ": "سَجَدَ وَجْهِيَ", "ٱللَّهِوَٱلْحَمْدُ": "ٱللَّهِ وَٱلْحَمْدُ", "لِلَّهِوَلاَ": "لِلَّهِ وَلاَ",
         "اللَّهُوَٱللَّهُ": "اللَّهُ وَٱللَّهُ"}
# words a cut line may start with (a clause opening), and may not end with
OPENS = {"يا", "ويا", "اللهم", "الهي", "وان", "ان", "فان", "انك", "انه", "حتي", "ثم", "اسالك", "واسالك", "سبحان",
         "الحمد", "والحمد", "بسم", "اشهد", "واشهد", "صل", "وصل", "اعوذ", "واعوذ", "حسبي", "وبه"}
NO_END = {"يا", "لا", "من", "عن", "علي", "الي", "في", "الا", "ان", "لم", "لن", "قد", "ما", "ثم", "او", "ام", "و", "ب",
          "الذي", "التي", "اللهم", "ابن", "بن", "عبد", "ذي", "ذو", "رب", "وال", "كل"}
STOP = ("،", ".", "؛", ",")


def clean(ar: str) -> str:
    ar = html.unescape(ar).replace("‌", " ")  # ZWNJ: "عليه‌السلام"
    ar = re.sub(r"[*﴿﴾\"“”…۔]|-{2,}", " ", ar)
    ar = re.sub(r"\s+(?=[ً-ْٰ])", "", ar)  # a space before a vowel mark: "ي َا" -> "يَا"
    ar = re.sub(r"\s*([،؛])\s*", r"\1 ", ar)  # "الطَّاغُوْتِ،وَ" -> "الطَّاغُوْتِ، وَ"
    ar = " ".join(GLUED.get(w, w) for w in ar.split())
    ar = re.sub(r"\s+", " ", ar).strip(" ,.،؛")
    return _FIXES.get(normalize(ar).replace(" ", ""), ar)


# ------------------------------------------------------------------ the pages
TOK = re.compile(
    r'(?P<hr><hr\b[^>]*>)|<h[3-6][^>]*>(?P<h>.*?)</h[3-6]>|<b>(?P<b>.*?)</b>'
    r'|<div class="Ara">(?P<ara>.*?)</div>|<div class="Trl">(?P<trl>.*?)</div>|<div class="Tra">(?P<tra>.*?)</div>',
    re.S | re.I)


def _text(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def page(name: str) -> str:
    f = PAGES / f"{name}.mobile.html"
    if not f.exists():
        PAGES.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(SITE.format(name), headers={"User-Agent": "dua-recognition/0.2 (research)"})
        f.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return f.read_text(encoding="utf-8", errors="replace")


def blocks(name: str) -> list[list[dict]]:
    """A mobile page's du'as: runs of lines between rules and headings."""
    out, cur = [], []
    for m in TOK.finditer(page(name)):
        if m.group("hr") or ((m.group("h") is not None or m.group("b") is not None)
                             and _text(m.group("h") or m.group("b") or "") and cur):
            if cur:
                out.append(cur)
            cur = []
        elif m.group("ara") is not None:
            cur.append({"ar": clean(_text(m.group("ara"))), "tl": "", "en": ""})
        elif m.group("trl") is not None and cur:
            cur[-1]["tl"] = _text(m.group("trl"))
        elif m.group("tra") is not None and cur:
            cur[-1]["en"] = _text(m.group("tra"))
    if cur:
        out.append(cur)
    return [b for b in out if any(r["ar"] for r in b)]


def mi_blocks() -> list[list[dict]]:
    d = json.loads(MI_PAGE.read_text(encoding="utf-8"))
    return [[{"ar": clean(s["arabic"]), "tl": (s.get("transliteration") or "").strip(),
              "en": (s.get("translation") or "").strip()} for s in e.get("segments") or [] if (s.get("arabic") or "").strip()]
            for e in d["duas"] if e.get("type") in ("dua", "continue-dua")]


def find(bl: list[list[dict]], opening: str) -> list[dict]:
    want = normalize(opening).split()
    hits = [b for b in bl if normalize(" ".join(r["ar"] for r in b)).split()[:len(want)] == want]
    if not hits:
        raise SystemExit(f"no block opens with {opening!r}")
    return hits[0]  # the page's first: a later one is a repeat further down (common-taqibat's "Twelfth")


# ------------------------------------------------------------------ long lines
def cut(ar: str) -> list[str]:
    """A line of more than LONG words in pieces of about 7: after a comma or full stop
    where there is one, else before a clause opening, else before a وَ; never after a
    word that needs the next."""
    w = ar.split()
    if len(w) <= LONG:
        return [ar]
    n = len(w)
    best = [0.0] + [float("inf")] * n
    back = [0] * (n + 1)
    for e in range(1, n + 1):
        for s in range(max(0, e - 11), e - 2):
            if best[s] == float("inf"):
                continue
            c = best[s] + (e - s - 7) ** 2
            if e < n:
                nxt = (normalize(w[e]).split() or [""])[0]
                if w[e - 1].endswith(STOP) or w[e] in STOP:
                    pass
                elif nxt in OPENS:
                    c += 2
                elif nxt.startswith("و"):
                    c += 8
                else:
                    c += 14
                end = normalize(w[e - 1]).split()[-1:]
                if end and end[0] in NO_END:
                    c += 40
            if c < best[e]:
                best[e], back[e] = c, s
    out, e = [], n
    while e > 0:
        out.append(" ".join(w[back[e]:e]).strip(" ,.،؛"))
        e = back[e]
    return out[::-1]


def build() -> list[tuple[str, dict, dict]]:
    cache = {"mi": mi_blocks()}
    out = []
    for dua_id, en_name, ar_name, parts in TEXTS:
        segs, extra = [], {}
        for pg, opening, times in parts:
            if pg == "hand":
                rows = [{"ar": a, "tl": "", "en": e} for a, e in HAND[opening]]
            else:
                if pg not in cache:
                    cache[pg] = blocks(pg)
                rows = find(cache[pg], opening)
            for k, r in enumerate(rows):
                words = normalize(r["ar"]).split()
                en = r["en"] or next((v for k, v in ENGLISH.items() if words[:len(k.split())] == k.split()), "")
                if k == 0 and times:
                    en = f"{TIMES[times]} {en}".strip()
                pieces = cut(r["ar"])
                for j, piece in enumerate(pieces):
                    sid = len(segs) + 1
                    segs.append({"segment_id": sid, "arabic": piece})
                    row = {}
                    if j == 0 and en:
                        row["en"] = en
                    if len(pieces) == 1 and r["tl"]:
                        row["tl"] = r["tl"]
                    if row:
                        extra[str(sid)] = row
        text = {"dua_id": dua_id, "dua_name_en": en_name, "dua_name_ar": ar_name,
                "source": "https://www.duas.org/regular-after-salaat-taqibat.html (its ta'qibat pages, "
                          "in Mafatih al-Jinan's order; scripts/taqibat_corpus.py)",
                "segments": segs}
        out.append((dua_id, text, extra))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=DUAS)
    ap.add_argument("--lines-out", type=Path, default=LINES)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    for dua_id, text, extra in build():
        n = [len(s["arabic"].split()) for s in text["segments"]]
        print(f"== {dua_id}: {len(n)} lines, {sum(n)} words (median {sorted(n)[len(n) // 2]}, max {max(n)}), "
              f"English on {len(extra)}")
        for s in text["segments"]:
            print(f"  {s['segment_id']:3d}  {s['arabic']}   | {extra.get(str(s['segment_id']), {}).get('en', '')[:60]}")
        if not args.dry_run:
            args.out.mkdir(parents=True, exist_ok=True)
            args.lines_out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{dua_id}.json").write_text(json.dumps(text, ensure_ascii=False, indent=1), encoding="utf-8")
            (args.lines_out / f"{dua_id}.json").write_text(json.dumps(extra, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")


if __name__ == "__main__":
    main()
