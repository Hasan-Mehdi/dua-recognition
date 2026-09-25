"""Growing-buffer streaming front end (LocalAgreement-2), as an alternative to
fixed 6 s windows.

Ported from the author's streaming-asr project (github.com/Hasan-Mehdi,
chunked re-inference + LocalAgreement; Machacek et al. 2023,
https://arxiv.org/abs/2307.14743), adapted for recitation:

  * agreement compares *normalized* Arabic, so two passes that differ only in
    tashkeel or hamza spelling still agree;
  * recitation has no sentence punctuation to cut at, so once the buffer passes
    `trim_after` it is cut at the last committed word, and hard-capped at
    `max_buffer` (long buffers invite Whisper's repetition loops).

Each hop re-transcribes the whole buffer. What the tracker sees is selectable:
the latest full hypothesis (committed + tentative words: more context than a
6 s window at the same freshness) or committed words only (stable, but ~1 pass
later). scripts/transcribe_streaming.py measures both against fixed windows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .asr import SAMPLE_RATE, _is_silent, _looks_hallucinated, load_model
from .text import normalize


@dataclass(frozen=True)
class Word:
    start: float  # seconds, absolute stream time
    end: float
    text: str


class HypothesisBuffer:
    """Commit the longest common prefix of consecutive hypotheses (LocalAgreement-2)."""

    def __init__(self, overlap_tolerance: float = 0.2):
        self.committed: list[Word] = []
        self._pending: list[Word] = []
        self.overlap_tolerance = overlap_tolerance

    @property
    def frontier(self) -> float:
        return self.committed[-1].end if self.committed else 0.0

    def update(self, words: list[Word]) -> list[Word]:
        """Stage a new hypothesis, commit what agrees with the previous one."""
        new = [w for w in words if w.start > self.frontier - self.overlap_tolerance]
        # The buffer still overlaps committed text: drop a repeated head n-gram.
        for n in range(min(4, len(new), len(self.committed)), 0, -1):
            if [normalize(w.text) for w in self.committed[-n:]] == [normalize(w.text) for w in new[:n]]:
                new = new[n:]
                break
        agreed = []
        for prev, cur in zip(self._pending, new):
            if normalize(prev.text) != normalize(cur.text):
                break
            agreed.append(cur)
        self.committed.extend(agreed)
        self._pending = new[len(agreed):]
        return agreed

    @property
    def tentative(self) -> list[Word]:
        return list(self._pending)


class BufferedTranscriber:
    def __init__(self, model: str, trim_after: float = 12.0, max_buffer: float = 20.0):
        self.model = model
        self.trim_after = trim_after
        self.max_buffer = max_buffer
        self.reset()

    def reset(self) -> None:
        self.hyp = HypothesisBuffer()
        self.audio = np.zeros(0, dtype=np.float32)
        self.offset = 0.0  # stream time of audio[0]

    def push(self, chunk: np.ndarray) -> None:
        self.audio = np.concatenate([self.audio, np.asarray(chunk, dtype=np.float32)])

    def process(self) -> tuple[list[Word], list[Word]]:
        """Re-transcribe the buffer; return (all committed words so far, tentative tail)."""
        words: list[Word] = []
        if not _is_silent(self.audio[-SAMPLE_RATE * 6 :]):
            segments, _ = load_model(self.model).transcribe(
                self.audio, language="ar", beam_size=1, word_timestamps=True,
                condition_on_previous_text=False, vad_filter=False, without_timestamps=False,
            )
            for s in segments:
                if s.no_speech_prob > 0.6 or _looks_hallucinated(s.text):
                    continue
                words += [Word(w.start + self.offset, w.end + self.offset, w.word.strip()) for w in (s.words or [])]
        self.hyp.update(words)
        self._trim()
        return self.hyp.committed, self.hyp.tentative

    def _trim(self) -> None:
        dur = len(self.audio) / SAMPLE_RATE
        if dur <= self.trim_after:
            return
        in_buffer = [w for w in self.hyp.committed if w.end > self.offset]
        if in_buffer:
            cut = in_buffer[-1].end
        elif dur > self.max_buffer:
            cut = self.offset + dur - self.trim_after  # nothing agreed (noise, music)
        else:
            return
        drop = int((cut - self.offset) * SAMPLE_RATE)
        if drop > 0:
            self.audio = self.audio[drop:]
            self.offset = cut
