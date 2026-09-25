"""Align a transcript fragment against the whole reference corpus at once.

The corpus is flattened into one letter string (spaces dropped, so Whisper's
word splitting doesn't matter) with a map from each letter back to its word,
segment and du'a. For a fragment h, `end_costs` returns, for every reference
position e, the edit distance between h and the best-matching reference
substring that *ends* at e — semi-global alignment, where h must be consumed
in full but may start anywhere in the reference.

That is exactly the question a follower asks: "if the reciter is here right
now, how well does what we just heard fit?" Repeated refrains score equally
well at every repetition; telling those apart is the tracker's job.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .corpus import Dua
from .text import normalize

_ALPHABET = {chr(c): i + 1 for i, c in enumerate(range(0x0621, 0x064B))}


def encode(text: str) -> np.ndarray:
    """Normalized Arabic -> int codes, spaces dropped."""
    return np.fromiter((_ALPHABET[c] for c in normalize(text) if c in _ALPHABET), dtype=np.int16)


@dataclass
class Word:
    dua: int  # index into CorpusIndex.duas
    segment: int  # segment id within the du'a
    text: str  # normalized
    token: int  # index into the segment's display text, split on whitespace


class CorpusIndex:
    """Every du'a's text as one letter array, plus letter -> word -> segment maps."""

    def __init__(self, duas: dict[str, Dua]):
        self.dua_ids = list(duas)
        self.duas = [duas[k] for k in self.dua_ids]
        self.words: list[Word] = []
        letters, letter_word = [], []
        self.dua_word_span: list[tuple[int, int]] = []
        for di, dua in enumerate(self.duas):
            first = len(self.words)
            for seg in dua.segments:
                for ti, token in enumerate(seg.arabic.split()):
                    for w in normalize(token).split():
                        codes = encode(w)
                        if not codes.size:
                            continue  # punctuation-only token
                        letters.append(codes)
                        letter_word.append(np.full(codes.size, len(self.words), dtype=np.int32))
                        self.words.append(Word(di, seg.id, w, ti))
            self.dua_word_span.append((first, len(self.words)))
        self.letters = np.concatenate(letters)
        self.letter_word = np.concatenate(letter_word)
        self.word_dua = np.array([w.dua for w in self.words], dtype=np.int32)
        self.word_segment = np.array([w.segment for w in self.words], dtype=np.int32)
        # Last letter of each word. Whisper emits whole words, so a fragment
        # "ends at word w" when its alignment ends on w's final letter; scoring
        # anywhere inside w would let the next word's first letter tie with w.
        self._word_ends = np.flatnonzero(np.r_[np.diff(self.letter_word) != 0, True])

    @property
    def n_words(self) -> int:
        return len(self.words)

    def end_costs(self, fragment: str) -> np.ndarray | None:
        """Per-letter semi-global edit distance of `fragment` ending there."""
        h = encode(fragment)
        if not h.size:
            return None
        return semiglobal_end_costs(h, self.letters)

    def word_costs(self, fragment: str) -> np.ndarray | None:
        """Per-word cost: best alignment of `fragment` ending on that word."""
        costs = self.end_costs(fragment)
        if costs is None:
            return None
        # Edit counts are bounded by the fragment length (a few hundred at most),
        # and evaluation keeps one of these per window: int16 keeps that small.
        return costs[self._word_ends].astype(np.int16)

    def ctc_word_costs(self, lp: np.ndarray, clip: float = 300.0) -> np.ndarray | None:
        """Per-word cost from a CTC model's frame log posteriors (ctc_align.py):
        minus the best path log-likelihood of the window ending on that word,
        relative to the best word and clipped (so float16 stays precise where
        it matters)."""
        from .ctc_align import end_scores, has_speech, squeeze_blanks

        if not lp.shape[0] or not has_speech(lp):
            return None
        cost = -end_scores(squeeze_blanks(lp), self.letters)[self._word_ends]
        return np.minimum(cost - cost.min(), clip).astype(np.float16)

    def segment_text(self, dua_index: int, segment_id: int) -> str:
        seg = next(s for s in self.duas[dua_index].segments if s.id == segment_id)
        return seg.arabic

    def text_before(self, word: int, n_words: int = 12) -> str:
        """Reference words leading up to (and including) `word`, same du'a only."""
        lo = max(self.dua_word_span[self.word_dua[word]][0], word - n_words + 1)
        return " ".join(w.text for w in self.words[lo : word + 1])


def semiglobal_end_costs(h: np.ndarray, r: np.ndarray) -> np.ndarray:
    """D[m][e] of the edit-distance DP with a free start in r.

    Row recurrence, vectorized over the reference:
        D[i][e] = min(D[i-1][e-1] + (h_i != r_e), D[i-1][e] + 1, D[i][e-1] + 1)
    The left-neighbour term is a running minimum: D[i][e] = e + cummin(D'[i][k] - k)
    where D' is the row before horizontal moves. O(len(h) * len(r)) numpy work,
    ~20 ms for a 60-letter fragment against the full 12-du'a corpus.
    """
    n = r.size
    idx = np.arange(1, n + 1, dtype=np.int32)
    prev = np.zeros(n + 1, dtype=np.int32)  # row 0: free start anywhere
    for i, c in enumerate(h, start=1):
        cur = np.empty(n + 1, dtype=np.int32)
        cur[0] = i
        diag = prev[:-1] + (r != c)
        up = prev[1:] + 1
        cur[1:] = np.minimum(diag, up)
        # horizontal gaps: cur[e] = min_k<=e cur[k] + (e - k)
        cur[1:] = np.minimum(cur[1:], np.minimum.accumulate(np.r_[cur[0], cur[1:] - idx] )[1:] + idx)
        prev = cur
    return prev[1:]
