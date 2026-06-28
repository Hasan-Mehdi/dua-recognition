"""Arabic normalization for matching ASR output against reference text.

ASR transcripts rarely carry full tashkeel and disagree on how they render the
alef/ya/ta-marbuta variants, so we strip everything that isn't load-bearing for
a fuzzy comparison and let the matcher do the rest. This is deliberately lossy.
"""
import re
import unicodedata

# Harakat (fatha..sukun), the superscript alef, and the tatweel elongation mark.
_TASHKEEL = re.compile(r"[ً-ْٰـ]")
# Anything that isn't an Arabic letter or whitespace, post-normalization.
_NON_ARABIC = re.compile(r"[^ء-ي\s]")

# آ أ إ  ->  ا
_ALEF_VARIANTS = {"آ": "ا", "أ": "ا", "إ": "ا"}


def strip_diacritics(text: str) -> str:
    return _TASHKEEL.sub("", text)


def normalize(text: str) -> str:
    """Fold a string down to bare Arabic letters for fuzzy matching."""
    text = unicodedata.normalize("NFC", text)
    text = strip_diacritics(text)
    for variant, base in _ALEF_VARIANTS.items():
        text = text.replace(variant, base)
    text = text.replace("ى", "ي")  # alef maksura -> ya
    text = text.replace("ة", "ه")  # ta marbuta  -> ha
    text = _NON_ARABIC.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
