"""Arabic normalization for matching ASR output against reference text.

ASR transcripts rarely carry full tashkeel and disagree on how they render the
alef/ya/ta-marbuta variants, so we strip everything that isn't load-bearing for
a fuzzy comparison and let the matcher do the rest. This is deliberately lossy.
"""
import re
import unicodedata

# Harakat (fathatan..sukun), maddah/hamza marks above and below, the superscript
# alef, Quranic annotation signs, and the tatweel elongation mark.
_TASHKEEL = re.compile(r"[ً-ٰٟۖ-ۭـ]")
# Anything that isn't a (folded) Arabic letter or whitespace.
_NON_ARABIC = re.compile(r"[^ء-ي\s]")

_FOLD = str.maketrans(
    {
        "آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا",  # alef variants, incl. alef wasla
        "ى": "ي", "ی": "ي", "ئ": "ي",            # alef maksura, Persian ya, ya-hamza
        "ؤ": "و",                                # waw-hamza
        "ة": "ه",                                # ta marbuta -> ha
        "ک": "ك",                                # Persian kaf
    }
)


def strip_diacritics(text: str) -> str:
    return _TASHKEEL.sub("", text)


def normalize(text: str) -> str:
    """Fold a string down to bare Arabic letters for fuzzy matching."""
    text = unicodedata.normalize("NFC", text)
    text = strip_diacritics(text).translate(_FOLD)
    text = _NON_ARABIC.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
