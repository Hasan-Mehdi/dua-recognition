"""v0.1's du'a classifier, kept as the baseline in scripts/evaluate.py.

Transcribe, then pick the du'a whose full reference text the transcript
matches best. It looks at each window on its own; the tracker (tracker.py)
identifies the du'a as a by-product of following it instead.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from .asr import transcribe
from .corpus import Dua
from .text import normalize


class TextClassifier:
    """Identify the du'a from a transcript by fuzzy-matching its full text."""

    def __init__(self, duas: dict[str, Dua]):
        self._ref = {k: normalize(" ".join(v.texts)) for k, v in duas.items()}

    def predict_text(self, transcript: str) -> str:
        hyp = normalize(transcript)
        return max(self._ref, key=lambda k: fuzz.partial_ratio(hyp, self._ref[k]))

    def predict(self, audio) -> str:
        return self.predict_text(transcribe(audio))
