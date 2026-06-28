"""Tie classification and passage matching into one recognizer."""
from __future__ import annotations

from .asr import transcribe
from .classify import TextClassifier
from .corpus import Dua
from .match import PassageMatcher


class Recognizer:
    def __init__(self, duas: dict[str, Dua], classifier: TextClassifier | None = None):
        self.duas = duas
        self.classifier = classifier or TextClassifier(duas)
        self._matchers = {k: PassageMatcher(v) for k, v in duas.items()}

    def recognize(self, audio) -> dict:
        """Transcribe a clip once, identify the du'a, then locate the passage."""
        transcript = transcribe(audio)
        dua_id = self.classifier.predict_text(transcript)
        match = self._matchers[dua_id].locate(transcript)
        return {
            "dua": dua_id,
            "transcript": transcript,
            "segment": match.segment_id if match else None,
            "confidence": round(match.score, 3) if match else 0.0,
        }
