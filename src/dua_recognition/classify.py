"""Decide which du'a a clip belongs to.

Each du'a opens with a distinctive text (and melody), so this is the easier half
of the problem. The baseline below is text-only: transcribe, then pick the du'a
whose reference text the transcript matches best. An audio classifier (a small
spectrogram model) would be faster and melody-aware — that's the TODO.
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


class AudioClassifier:
    """Spectrogram-based du'a classifier.

    TODO: train an AST / wav2vec2 head on 5-second mel windows so we can identify
    the du'a without a full transcription pass. See scripts/train_classifier.py.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "audio classifier isn't trained yet — use TextClassifier for now"
        )
