"""One transliteration style for every text, made from its vowelled Arabic.

The sources transliterate in at least three styles (DuaPlayer's letter-by-letter
"alrrahmani", duas.org's capitals and doubled vowels "RAH'MATIKA", macrons), and
415 of the 506 texts have none. Every text is fully vowelled, so the reading can
be spelled out by rule instead, in one plain academic-lite style:

    long vowels ā ī ū, emphatics ḥ ṣ ḍ ṭ ẓ, ‘ for ‘ayn, ’ for hamza,
    the article as al- / ar- (sun letters), l- after a vowel mid-line,
    and the pause at the end of each line (no final short vowel, -ah for ة).

It reads what the vowel marks say; where a source leaves a letter unmarked it
guesses the common reading (an unmarked و after a damma is ū, and so on).
"""
from __future__ import annotations

import re
import unicodedata

FATHA, DAMMA, KASRA, SUKUN, SHADDA = "َ", "ُ", "ِ", "ْ", "ّ"
FATHATAN, DAMMATAN, KASRATAN = "ً", "ٌ", "ٍ"
DAGGER, MADDA, HAMZA_ABOVE, HAMZA_BELOW = "ٰ", "ٓ", "ٔ", "ٕ"
SMALL_I, SMALL_U = "ٖۦ", "ٗۥ"  # subscript alif, small yeh / waw: long vowels
VOWEL = {FATHA: "a", DAMMA: "u", KASRA: "i", FATHATAN: "an", DAMMATAN: "un", KASRATAN: "in"}
MARKS = set(VOWEL) | {SUKUN, SHADDA, DAGGER, MADDA, HAMZA_ABOVE, HAMZA_BELOW} | set(SMALL_I + SMALL_U)

CONSONANT = {
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "ḥ", "خ": "kh", "د": "d", "ذ": "dh",
    "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "ṣ", "ض": "ḍ", "ط": "ṭ", "ظ": "ẓ",
    "ع": "‘", "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ک": "k", "ل": "l", "م": "m",
    "ن": "n", "ه": "h", "ھ": "h", "و": "w", "ي": "y", "ی": "y", "ى": "y", "ة": "t", "ۀ": "t",
    "ء": "’", "أ": "’", "إ": "’", "ؤ": "’", "ئ": "’", "آ": "’",
}
SUN = set("تثدذرزسشصضطظلن")
ALIF, WASLA = "ا", "ٱ"
PUNCT = {"،": ",", "؛": ";", "؟": "?", ".": ".", ",": ",", "!": "!", ":": ":", "?": "?", ";": ";"}
# Everything else that isn't a letter or a mark (stop signs, digits, brackets, tatweel...) goes.
_ARABIC_LETTER = re.compile(r"[ء-ؿف-يٱکیۀھ]")  # letters, not tatweel


def _units(word: str) -> list[tuple[str, set[str]]]:
    """Letters with the marks on each."""
    out: list[tuple[str, set[str]]] = []
    for ch in word:
        if ch in MARKS:
            if out:
                out[-1][1].add(ch)
        elif _ARABIC_LETTER.match(ch):
            out.append((ch, set()))
    return out


# Words the sources often write without the small alif that makes a vowel long:
# skeleton (after any proclitic) -> index of the letter that carries it.
DAGGER_WORDS = {"الرحمن": 4, "رحمن": 3, "اله": 1, "الهي": 1, "الهنا": 1, "هذا": 0, "هذه": 0,
                "هذان": 0, "ذلك": 0, "ذلكم": 0, "لكن": 0, "اولئك": 2, "السموات": 4, "سموات": 2}


def _restore_daggers(units: list[tuple[str, set[str]]]) -> None:
    for start in (0, 1):
        sk = _skeleton(units[start:]).replace("إ", "ا").replace("أ", "ا")
        j = DAGGER_WORDS.get(sk)
        if j is not None and start + j < len(units):
            m = units[start + j][1]
            if not m & {DAGGER, SUKUN, KASRA, DAMMA}:
                m.discard(FATHA)
                m.add(DAGGER)
            return


def _fix_shadda(units: list[tuple[str, set[str]]]) -> None:
    """Source slips around the shadda: on the alif after the doubled letter (إلاّ),
    or with no vowel of its own mid-word (مُحَمّدٍ), where it is nearly always a fatha."""
    for k, (b, m) in enumerate(units):
        if SHADDA in m and b in (ALIF, WASLA) and k > 0:
            m.discard(SHADDA)
            units[k - 1][1].add(SHADDA)
    for k, (b, m) in enumerate(units[:-1]):
        nxt = units[k + 1][0]
        if SHADDA in m and not (m & (set(VOWEL) | {DAGGER, SUKUN})) and nxt not in (ALIF, "ى", "و", "ي"):
            m.add(FATHA)


def _vowel(marks: set[str]) -> str:
    if SUKUN in marks:  # a sukun beside a vowel is a source slip; the sukun wins
        return ""
    for m, v in VOWEL.items():
        if m in marks:
            return v
    return ""


def _skeleton(units) -> str:
    return "".join(b for b, _ in units).replace(WASLA, ALIF)


def _word(units: list[tuple[str, set[str]]], initial: bool, final: bool) -> str:
    """One word. `initial`: first in the line (hamzat al-wasl is voiced);
    `final`: last in the line (pausal form)."""
    out = ""
    i = 0
    if len(units) == 1 and units[0][0] in "وف" and not _vowel(units[0][1]):
        return CONSONANT[units[0][0]] + "a"  # a lone wa / fa written without its fatha
    # Proclitics before an article: wa-, fa-, bi-, li-, ka-.
    pre = ""
    if len(units) > 2 and units[0][0] in "وفبلك" and units[1][0] in (ALIF, WASLA) and units[2][0] == "ل":
        pre = CONSONANT[units[0][0]] + (_vowel(units[0][1]) or "a")
        i = 1
    rest = units[i:]
    sk = _skeleton(rest).replace("أ", "ا").replace("إ", "ا")
    # Allāh and Allāhumma, with whatever case ending the marks give.
    if sk in ("الله", "اللهم") or (sk in ("لله",) and pre == "" and units[0][0] == "ل"):
        if sk == "لله":  # li-llāh: the proclitic lam leans on the name
            return "lillāh" + ("" if final else _vowel(units[-1][1]))
        name = "llāhumma" if sk == "اللهم" else "llāh" + ("" if final else _vowel(rest[-1][1]))
        if sk == "اللهم" and final:
            name = "llāhumm"
        return (pre + name) if pre else "A" + name
    # alladhī, allatī, alladhīna...: the article's lam merged into the relative pronoun's.
    rel = len(rest) > 2 and rest[0][0] in (ALIF, WASLA) and rest[1][0] == "ل" and SHADDA in rest[1][1] \
        and rest[2][0] in "ذت"
    start_vowel = ""
    if rel:
        start_vowel = _vowel(rest[1][1]) or "a"
        out = pre + ("a" if initial and not pre else "") + "ll" + start_vowel
        units = rest[2:]
    elif len(rest) > 2 and rest[0][0] in (ALIF, WASLA) and not _vowel(rest[0][1]) and rest[1][0] == "ل" \
            and HAMZA_ABOVE not in rest[0][1]:
        nxt = rest[2]
        sun = nxt[0] in SUN and (SHADDA in nxt[1] or SUKUN not in rest[1][1] and not _vowel(rest[1][1]))
        sound = CONSONANT[nxt[0]] if sun else "l"
        art = f"{sound}-"
        out = pre + art if pre else (("a" if initial else "") + art)
        # The sun letter is already said once by the article.
        units = rest[2:]
        if sun:
            units = [(units[0][0], units[0][1] - {SHADDA} | {"_sun"})] + units[1:]
    else:
        out = ""
    i = 0

    word_initial = not out or out.endswith("-")  # nor is a hamza right after the article
    last_vowel = start_vowel  # the short vowel the previous letter carried
    n = len(units)
    ta_marbuta_end = False
    for k in range(i, n):
        b, marks = units[k]
        v = _vowel(marks)
        # Tanween belongs on a word's last letter (or before its final alif / ة / hamza).
        if len(v) == 2 and k < n - 1 and units[k + 1][0] not in (ALIF, "ى", "ة", "ء"):
            v = v[0]
        end = k == n - 1
        if b in (ALIF, WASLA) and HAMZA_ABOVE not in marks and HAMZA_BELOW not in marks:
            if k == 0 and word_initial:
                if b == ALIF and MADDA in marks:
                    out += "ā"
                elif v or k + 1 < n and units[k + 1][0] in "وي" and SUKUN in units[k + 1][1]:
                    out += v or "a"  # a hamza the source wrote as a bare alif (او = aw)
                elif out.endswith("-"):
                    pass
                else:  # hamzat al-wasl: i-, or u- before a damma'd third letter
                    third = units[k + 2][1] if k + 2 < n else set()
                    out += "u" if DAMMA in third else "i"
            elif MADDA in marks:
                out += "’ā"
            elif k == 1 and units[0][0] in "وفب" and k + 1 < n and SUKUN in units[k + 1][1]:
                pass  # wa-bna: hamzat al-wasl after a proclitic is silent
            elif last_vowel == "a":
                out = out[:-1] + "ā"
            elif last_vowel in ("an",):
                pass  # the alif that carries -an
            elif end and (out.endswith("ū") or out.endswith("w")):
                pass  # the silent alif after a plural -ū
            elif not last_vowel and v:  # the vowel written on the alif: lā for لاَ
                out += "ā" if v == "a" else v
            elif not last_vowel:
                out += "ā"
            last_vowel = ""
            continue
        if b == "آ":
            if last_vowel == "a" and end:  # yā written يآ
                out = out[:-1] + "ā"
            else:
                out += ("" if k == 0 and word_initial else "’") + "ā"
            last_vowel = ""
            continue
        if b in "أإ" or (b == ALIF and (HAMZA_ABOVE in marks or HAMZA_BELOW in marks)):
            vv = v or ("" if SUKUN in marks else "i" if b == "إ" or HAMZA_BELOW in marks else "a")
            out += ("" if k == 0 and word_initial else "’") + vv
            last_vowel = vv
            continue
        if b == "ى" or (b in "يی" and end and not v and SHADDA not in marks):
            if b != "ى" and SUKUN in marks and last_vowel == "a":
                out += "y"  # the diphthong ay
            elif DAGGER in marks or last_vowel == "a":
                out = out[:-1] + "ā" if last_vowel == "a" else out + "ā"
            elif last_vowel == "i":
                out = out[:-1] + "ī"
            else:
                out += "ā" if b == "ى" else "ī"
            last_vowel = ""
            continue
        # wā, yā before an alif: a consonant (but not the plural -ū with its silent alif)
        before_alif = k + 1 < n and units[k + 1][0] in (ALIF, WASLA, "آ") and not (b == "و" and k + 2 == n)
        if b in "ويی" and not v and SHADDA not in marks and not before_alif:
            long_ = {"و": ("u", "ū", "aw"), "ي": ("i", "ī", "ay"), "ی": ("i", "ī", "ay")}[b]
            if last_vowel == long_[0]:
                out = out[:-1] + long_[1]
                last_vowel = ""
                continue
            if last_vowel == "a" and SUKUN in marks or last_vowel == "a" and not end:
                out += long_[2][1]
                last_vowel = ""
                continue
            if not last_vowel and not end and k > 0:
                out += long_[1]
                last_vowel = ""
                continue
        if b in "ةۀ":
            ta_marbuta_end = end
            if not v:
                out += "h" if last_vowel == "a" or not last_vowel else "t"
                last_vowel = ""
                continue
        c = CONSONANT.get(b, "")
        if SHADDA in marks and not (k == 0 and word_initial):  # word-initial: a tajwid mark
            c = c + c if len(c) == 1 else c + c[-1] if c in ("th", "dh", "sh", "kh", "gh") else c + c
        if c == "’" and (k == 0 and word_initial or out.endswith("-")):
            c = ""  # a word-initial hamza isn't written
        out += c
        if DAGGER in marks:
            out += "ā"
            last_vowel = ""
        elif marks & set(SMALL_I):
            out += "ī"
            last_vowel = ""
        elif marks & set(SMALL_U):
            out += "ū"
            last_vowel = ""
        else:
            out += v
            last_vowel = v
    if final:
        out = _pause(out, ta_marbuta_end)
    return out


def _pause(w: str, ta_marbuta: bool) -> str:
    """The reading at a stop: no final short vowel, -an becomes -ā, ة becomes -ah."""
    if ta_marbuta:
        return re.sub(r"t(an|un|in|a|u|i)?$", "h", w)
    if w.endswith("an") and len(w) > 3:
        return w[:-2] + "ā"
    for tail in ("un", "in"):
        if w.endswith(tail) and len(w) > 3:
            return w[:-2]
    if len(w) > 2 and w[-1] in "aiu" and w[-2] not in "aiuāīūwy":  # but huwa, hiya, liya
        return w[:-1]
    return w


def transliterate(line: str) -> str:
    """One line of vowelled Arabic in the house style."""
    line = unicodedata.normalize("NFKC", line).replace("ـ", "")
    tokens = re.findall(r"[^\s،؛؟.,!:?;]+|[،؛؟.,!:?;]", line)
    words = [(t, _units(t)) for t in tokens]
    for _, u in words:
        if u:
            _restore_daggers(u)
            _fix_shadda(u)
    idx = [k for k, (_, u) in enumerate(words) if u]
    out: list[str] = []
    for k, (tok, units) in enumerate(words):
        if not units:
            if tok in PUNCT and out:
                out[-1] += PUNCT[tok]
            continue
        out.append(_word(units, initial=k == idx[0], final=k == idx[-1]))
    # bismi + Allāh -> bismillāh, ṣallā + Allāhu -> ṣallallāhu: the name joins a vowel before it.
    joined: list[str] = []
    for w in out:
        if w.startswith("Allāh") and joined and joined[-1][-1:] in "aiuāīū" and joined[-1].lower() != "yā":
            prev = joined[-1]
            joined[-1] = prev[:-1] + {"ā": "a", "ī": "i", "ū": "u"}.get(prev[-1], prev[-1]) + w[1:]
        else:
            joined.append(w)
    text = " ".join(w for w in joined if w)
    text = re.sub(r"\bmuḥammad", "Muḥammad", text)
    return _capital(text)


def _capital(t: str) -> str:
    for i, ch in enumerate(t):
        if ch.isalpha():
            return t[:i] + ch.upper() + t[i + 1:]
    return t


def vowelled(line: str, min_share: float = 0.6) -> bool:
    """Whether a line carries enough vowel marks to be read by rule: at least
    `min_share` of its letters marked (long-vowel letters and alifs need none)."""
    units = _units(unicodedata.normalize("NFKC", line))
    letters = [(b, m) for b, m in units if b not in "اٱىوي"]
    if not letters:
        return True
    return sum(1 for _, m in letters if m) / len(letters) >= min_share


def for_display(arabic: str, source: str = "") -> str:
    """The reading the app shows for a line: ours where the Arabic is vowelled
    enough to read by rule, otherwise the source's own (if it has one)."""
    if vowelled(arabic) or not source.strip():
        return transliterate(arabic)
    return source.strip()


# English function words that never occur as words of a transliteration (no "an", "in", "la").
_ENGLISH = {"the", "you", "your", "and", "of", "to", "me", "my", "i", "is", "who", "that", "for", "with",
            "from", "which", "o", "not", "do", "we", "us", "our", "have", "be", "are", "all", "by", "it",
            "he", "his", "him", "they", "them", "their", "what", "upon", "after", "every", "whom", "those"}


def _letters(t: str) -> str:
    """A reading reduced to plain letters, doubled letters single, for comparing styles."""
    t = "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if "a" <= c <= "z")
    return re.sub(r"(.)\1+", r"\1", t)


def looks_english(t: str) -> bool:
    words = re.findall(r"[a-z]+", t.lower())
    return bool(words) and sum(w in _ENGLISH for w in words) / len(words) >= 0.2


def match_readings(arabic: list[str], readings: list[str], reach: int = 3) -> list[int | None]:
    """Which source reading goes with each Arabic line. Scraped rows are sometimes a
    line or two out of step (the reading drifts while the Arabic stays put), so a
    line takes a nearby row when that row matches our own reading clearly better
    than its own does. A vowelled line whose row doesn't match anything gets None;
    an under-vowelled one keeps its row, since our reading of it isn't a fair test."""
    from difflib import SequenceMatcher

    ours = [_letters(transliterate(a)) for a in arabic]
    theirs = [_letters(r) for r in readings]

    def sim(i: int, j: int) -> float:
        if not (0 <= j < len(theirs)) or not theirs[j] or not ours[i]:
            return 0.0
        return SequenceMatcher(None, ours[i], theirs[j], autojunk=False).ratio()

    out: list[int | None] = []
    for i, line in enumerate(arabic):
        here = sim(i, i)
        best = max(range(i - reach, i + reach + 1), key=lambda j: sim(i, j))
        if best != i and sim(i, best) >= 0.5 and sim(i, best) - here >= 0.2:
            out.append(best)
        elif here < 0.3 and vowelled(line):
            out.append(None)
        else:
            out.append(i if i < len(readings) else None)
    return out
