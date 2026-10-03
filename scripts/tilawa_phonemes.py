#!/usr/bin/env python
"""Du'a text in Tilawa's phonetic script, so its streaming phoneme model can follow du'as.

Tilawa (github.com/yazinsai/tilawa) follows Quran recitation with a streaming
Zipformer2-CTC phoneme model (Quran-Lab zipformer_p-arabic-v3, NPL-1.2: research
use here, nothing redistributed). It matches what it hears against each word
written in a tajweed-aware phonetic script: ٱللَّهِ -> للَااهِ, ٱلرَّحْمَـٰنِ ->
ررَحمَاانِ, a word at a pause loses its last vowel (ٱلرَّحِيمِ -> ررَحِۦۦۦۦم).
That table exists only for the Quran. This writes the du'as in the same script
with a small rule set ("tajweed-lite": gemination, long vowels, hamzat al-wasl,
the article and sun letters, tanween, ta marbuta, qalqala, madd before hamza,
pausal forms at line ends; no ikhfa, idgham or ghunna lengths), then packs them
into a corpus file Tilawa's engine loads: the du'as as the first surahs, lines
as ayahs, and real surahs after them as distractors (the loader wants 114).

    python scripts/tilawa_phonemes.py check --quran zipformer_quran.json   # rules vs the Quran's table
    python scripts/tilawa_phonemes.py corpus OUT.json --quran zipformer_quran.json [--duas ID ...]

`check` scores the rules on the Quran's own words (vowelled Uthmani in, Tilawa's
phonemes expected), as a character error rate after Tilawa's folding (ۦ -> ي ...).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FATHA, DAMMA, KASRA, SUKUN, SHADDA = "َ", "ُ", "ِ", "ْ", "ّ"
TANWIN = {"ً": FATHA, "ٌ": DAMMA, "ٍ": KASRA}
VOWELS = {FATHA, DAMMA, KASRA}
HAMZA_MARKS = {"ٔ", "ٕ"}
SUN = set("تثدذرزسشصضطظلن")
LONG = {FATHA: "اا", KASRA: "ۦۦ", DAMMA: "ۥۥ"}
HAMZAS = set("أإؤئء")
LETTERS = set("ءابتثجحخدذرزسشصضطظعغفقكلمنهويةىأإؤئآٱ")
QALQALA = set("قطبجد")
PREFIXES = set("وفبكل")
SUPERSCRIPT_ALEF = "ٰ"
SILENT = "۟"  # small high rounded zero: the letter under it isn't spoken
# Marks that aren't pronounced as written: Quranic stops, small high letters, tatweel.
_DROP = re.compile("[ؐ-ؚۖ-ۜ۟-ۤۧ-ۭـ࣓-ࣿ]")
FOLD = {"ۦ": "ي", "ۥ": "و", "ں": "ن", "۾": "م", "ٱ": "ا", "ى": "ي"}


def _letters(word: str) -> list[tuple[str, str]]:
    """(letter, marks) pairs; marks are the diacritics after it. Silent letters are dropped."""
    out: list[list[str]] = []
    for ch in unicodedata.normalize("NFC", word):
        if ch in LETTERS or ch in "ۥۦ":
            out.append([ch, ""])
        elif out:
            out[-1][1] += ch
    out = [[a, b] for a, b in out if SILENT not in b]
    return [(a.replace("ۥ", "و").replace("ۦ", "ي"), _DROP.sub("", b)) for a, b in out]


def _voweled(marks: str) -> bool:
    return any(m in VOWELS or m in HAMZA_MARKS for m in marks)


def _article(ls, i) -> bool:
    """ls[i] is the alef of a definite article, word-initial or after one or two prefix letters."""
    if i + 1 >= len(ls) or ls[i][0] not in "اٱ" or ls[i + 1][0] != "ل" or _voweled(ls[i][1]):
        return False
    return i == 0 or (i <= 2 and all(ls[k][0] in PREFIXES for k in range(i)))


def _is_allah(skel: str) -> bool:
    return skel in ("الله", "ٱلله", "لله", "اللهم", "ٱللهم") or skel.endswith(("لله", "للهم"))


def phonemize(word: str, first: bool, last: bool) -> str:
    """One word. first: starts an utterance (a hamzat al-wasl is spoken); last: ends one (pausal form)."""
    ls = _letters(word)
    if not ls:
        return ""
    skel = "".join(a for a, _ in ls)
    out: list[str] = []
    n = len(ls)
    i = 0
    while i < n:
        ch, marks = ls[i]
        if _article(ls, i):
            lam = ls[i + 1]
            nxt = ls[i + 2] if i + 2 < n else None
            if first and i == 0:
                out.append("ءَ")
            if SHADDA in lam[1]:
                i += 1  # ٱلَّذِى: one written lam, doubled; it's handled as a letter
                continue
            if nxt and nxt[0] in SUN and (SHADDA in nxt[1] or not _voweled(lam[1])) and SUKUN not in lam[1]:
                i += 2  # sun letter: the lam is assimilated (the shadda doubles the next letter)
                continue
            if not _voweled(lam[1]):
                out.append("ل")
                i += 2
                continue
            i += 1
            continue
        if ch == "ٱ" or (ch == "ا" and i == 0 and n > 1 and not _voweled(marks)):
            # hamzat al-wasl: spoken only to start an utterance
            if first and i == 0:
                nv = next((v for v in (ls[2][1] if n > 2 else "") if v in VOWELS), KASRA)
                out.append("ء" + (DAMMA if nv == DAMMA else KASRA))
            i += 1
            continue
        vowel = next((m for m in marks if m in VOWELS), "")
        tanwin = next((TANWIN[m] for m in marks if m in TANWIN), "")
        is_last = i == n - 1
        nxt = ls[i + 1] if i + 1 < n else None
        if ch == "آ":
            out.append("ءَاا" if i == 0 else "اااا")  # mid-word: a madd before hamza or sukun, four counts
            i += 1
            continue
        if ch in HAMZAS:
            ch = "ء"
        elif ch == "ة":
            ch = "ه" if last and is_last else "ت"
        elif ch == "ى":
            if SUPERSCRIPT_ALEF in marks or not vowel:
                i += 1  # alif maqsura: its long a came with the letter before
                continue
            ch = "ي"
        if ch in "وي" and SUPERSCRIPT_ALEF in marks and not vowel:
            out.append("اا")  # ٱلصَّلَوٰةَ: a waw or ya written for a long a
            i += 1
            continue
        if ch == "ا":
            if not vowel and not tanwin:
                if (out and out[-1].endswith(FATHA)) or SUPERSCRIPT_ALEF in marks:
                    out.append("اا")
                i += 1
                continue
            ch = "ء"
        piece = ch * (2 if SHADDA in marks else 1)
        if ch == "ل" and SHADDA in marks and nxt and nxt[0] == "ه" and _is_allah(skel):
            out.append(piece + FATHA + "اا")  # the name of God: a long a after its doubled lam
            i += 1
            continue
        if SUPERSCRIPT_ALEF in marks:
            piece += FATHA + "اا"
        elif vowel:
            piece += vowel
        elif tanwin:
            piece += tanwin
        elif ch in QALQALA and SUKUN in marks:
            piece += "ڇ"
        out.append(piece)
        # A vowel lengthened by the next letter (fatha + ا/ى, kasra + ي, damma + و), unless that one is voweled.
        if nxt and vowel and not tanwin and SUPERSCRIPT_ALEF not in marks:
            nch, nmarks = nxt
            bare = not any(m in VOWELS or m in TANWIN or m == SHADDA or m in HAMZA_MARKS for m in nmarks)
            if bare and ((vowel == FATHA and nch in "اى") or (vowel == KASRA and nch in "يى")
                         or (vowel == DAMMA and nch == "و")):
                after = ls[i + 2][0] if i + 2 < n else ""
                long_ = LONG[vowel] * (2 if after in HAMZAS or after == "آ" else 1)  # madd muttasil
                out[-1] = out[-1][:-1] + vowel + long_
                i += 2
                if nch == "و" and i == n - 1 and ls[i][0] == "ا":
                    i += 1  # the silent alif after waw al-jama'a
                continue
        if tanwin and not (last and is_last):
            out.append("ن")
        i += 1
    s = "".join(out)
    if last:
        # Pausal form: the final short vowel (or tanwin) goes; fathatan becomes a long a.
        if s.endswith("ن") and len(s) >= 2 and s[-2] in VOWELS and any(m in TANWIN for m in ls[-1][1]):
            s = s[:-2] + ("َاا" if s[-2] == FATHA else "")
        elif s and s[-1] in VOWELS:
            s = s[:-1]
            if s and s[-1] in QALQALA:
                s += "ڇ"
        for short, long_ in (("ۦۦ", "ۦۦۦۦ"), ("ۥۥ", "ۥۥۥۥ"), ("اا", "اااا")):
            k = s.rfind(short)
            if k >= 0 and k >= len(s) - len(short) - 2 and not s[k:].startswith(long_):
                s = s[:k] + long_ + s[k + len(short):]
                break
    return s


def _starts_wasl(word: str) -> bool:
    ls = _letters(word)
    return bool(ls) and (ls[0][0] == "ٱ" or _article(ls, 0))


def phonemize_line(words: list[str]) -> list[str]:
    out = [phonemize(w, i == 0, i == len(words) - 1) for i, w in enumerate(words)]
    for i in range(len(out) - 1):
        # Two silences can't meet: a long vowel before a hamzat al-wasl shortens (فِى ٱلسَّمَـٰوَٰتِ -> فِ).
        if _starts_wasl(words[i + 1]):
            for long_ in ("اا", "ۦۦ", "ۥۥ"):
                if out[i].endswith(long_):
                    out[i] = out[i][: -len(long_)]
                    break
    return out


def fold(s: str) -> str:
    return "".join(FOLD.get(c, c) for c in s)


def check(args) -> None:
    import jiwer

    corpus = json.loads(Path(args.quran).read_text(encoding="utf-8"))
    refs, hyps, shown = [], [], 0
    for sura in corpus["surahs"][: args.surahs]:
        for ayah in sura["ayahs"]:
            mine = phonemize_line([w[2] for w in ayah["w"]])
            for w, got in zip(ayah["w"], mine):
                refs.append(fold(w[1]))
                hyps.append(fold(got) or "-")
                if shown < args.show and fold(w[1]) != fold(got):
                    print(f"  {w[2]:>14s}  want {w[1]:<16s} got {got}")
                    shown += 1
    exact = sum(a == b for a, b in zip(refs, hyps)) / len(refs)
    print(f"{len(refs)} Quran words: CER {jiwer.cer(refs, hyps):.1%}, words exact {exact:.1%}")


def build_corpus(args) -> None:
    from dua_recognition.corpus import load_all

    quran = json.loads(Path(args.quran).read_text(encoding="utf-8"))
    duas = load_all()
    ids = (args.duas or sorted(duas))[: 114 - args.keep_surahs]
    surahs, mapping = [], []
    for n, did in enumerate(ids, 1):
        d = duas[did]
        ayahs = []
        for k, seg in enumerate(d.segments, 1):
            words = seg.arabic.split()
            ph = phonemize_line(words)
            ayahs.append({"n": k, "m": "‏", "w": [["‏", p or "ـ", w] for w, p in zip(words, ph)]})
            mapping.append({"surah": n, "ayah": k, "dua": did, "segment": seg.id, "words": len(words)})
        surahs.append({"n": n, "name": d.name_ar, "nameEn": d.name_en, "ayahs": ayahs})
    for extra in quran["surahs"][len(surahs):]:
        surahs.append({**extra, "n": len(surahs) + 1})
    out = Path(args.out)
    out.write_text(json.dumps({"v": 2, "surahs": surahs}, ensure_ascii=False), encoding="utf-8")
    out.with_suffix(".map.json").write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    print(f"{out}: {len(ids)} du'as as surahs 1-{len(ids)}, {len(surahs) - len(ids)} Quran surahs after them")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--quran", required=True, help="Tilawa's zipformer_quran.json")
    c.add_argument("--surahs", type=int, default=114)
    c.add_argument("--show", type=int, default=25)
    b = sub.add_parser("corpus")
    b.add_argument("out")
    b.add_argument("--quran", required=True)
    b.add_argument("--duas", nargs="*")
    b.add_argument("--keep-surahs", type=int, default=0, help="at least this many real surahs as distractors")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"check": check, "corpus": build_corpus}[args.cmd](args)


if __name__ == "__main__":
    main()
