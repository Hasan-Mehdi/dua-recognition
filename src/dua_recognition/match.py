"""v0.1's passage matcher, kept as the baseline in scripts/evaluate.py.

Each window is matched against every line independently, so a refrain that
recurs 14 times in a du'a can land on any of its repetitions. tracker.py
replaces it.
"""
from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from .corpus import Dua
from .text import normalize


@dataclass
class Match:
    segment_id: int
    score: float  # 0..1
    text: str


class PassageMatcher:
    """Fuzzy-match a transcript against the segments of a single du'a.

    partial_ratio is the right tool here: a short listening window usually covers
    only part of a segment, and the reciter may be mid-phrase when we sample.
    """

    def __init__(self, dua: Dua):
        self.dua = dua
        self._norm = [normalize(t) for t in dua.texts]

    def locate(self, transcript: str, min_score: float = 0.6) -> Match | None:
        hyp = normalize(transcript)
        if not hyp:
            return None
        best_idx, best = -1, 0.0
        for i, ref in enumerate(self._norm):
            score = fuzz.partial_ratio(hyp, ref) / 100.0
            if score > best:
                best, best_idx = score, i
        if best < min_score:
            return None
        seg = self.dua.segments[best_idx]
        return Match(seg.id, best, seg.arabic)
