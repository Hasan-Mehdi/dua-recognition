"""Streaming recognizer: audio in, (du'a, line, word) out.

Feed it audio in whatever chunk sizes arrive. Every `hop` seconds it
transcribes the last `window` seconds and moves the tracker. If ASR falls
behind real time, stale hops are skipped rather than queued: only the newest
window matters, and the tracker is told how much time actually passed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import CorpusIndex
from .asr import DEFAULT_MODEL, SAMPLE_RATE, transcribe_batch
from .corpus import Dua
from .tracker import Position, Tracker, TrackerConfig


@dataclass
class Update:
    t: float  # seconds of audio consumed
    transcript: str
    position: Position


class StreamingRecognizer:
    def __init__(
        self,
        duas: dict[str, Dua],
        *,
        model: str = DEFAULT_MODEL,
        window: float = 6.0,
        hop: float = 1.0,
        config: TrackerConfig | None = None,
        prompt_bias: bool = True,
        vad: bool = False,  # hurts in noisy rooms at current thresholds (docs/results/comparison.md)
        index: CorpusIndex | None = None,
    ):
        self.duas = duas
        self.config = config
        self.index = index or CorpusIndex(duas)
        self.tracker = Tracker(self.index, config)
        self.model = model
        self.window = int(window * SAMPLE_RATE)
        self.hop = int(hop * SAMPLE_RATE)
        self.prompt_bias = prompt_bias
        self.vad = vad
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self._buf = np.zeros(0, dtype=np.float32)
        self._total = 0  # samples received
        self._last = 0  # sample count at the last update

    def lock(self, dua_id: str) -> None:
        """The listener said which du'a this is: follow only that one.

        Skips identification entirely (no wait, no chance of a wrong lock).
        The audio buffer is kept, so a lock mid-recitation loses nothing.
        """
        self.index = CorpusIndex({dua_id: self.duas[dua_id]})
        self.tracker = Tracker(self.index, self.config)

    def push(self, samples: np.ndarray) -> None:
        """Buffer 16 kHz float32 samples without running anything."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, samples])[-self.window :]
        self._total += samples.size

    @property
    def due(self) -> bool:
        return self._total - self._last >= self.hop

    def feed(self, samples: np.ndarray) -> Update | None:
        """Add samples; returns an Update when a hop completes."""
        self.push(samples)
        return self.step()

    def step(self) -> Update | None:
        """Transcribe the current window and move the tracker, if a hop is due.

        A caller that pushes audio while a step is running (the web server)
        simply gets one step covering all of it: stale hops are never queued.
        """
        if not self.due:
            return None
        dt = (self._total - self._last) / SAMPLE_RATE
        self._last = self._total
        prompt = self.tracker.prompt() if self.prompt_bias else None
        window = self._buf.copy()
        text = transcribe_batch([window], model=self.model, prompts=[prompt], vad=self.vad)[0]
        pos = self.tracker.update(text, dt)
        return Update(self._total / SAMPLE_RATE, text, pos)


class Recognizer:
    """One-shot: which du'a and line does this clip come from?"""

    def __init__(self, duas: dict[str, Dua], model: str = DEFAULT_MODEL):
        self.duas = duas
        self.model = model
        self.index = CorpusIndex(duas)

    def recognize(self, audio) -> dict:
        from .asr import transcribe

        text = transcribe(audio, model=self.model)
        tracker = Tracker(self.index, TrackerConfig(min_dua_confidence=0.0))
        pos = tracker.update(text, 0.0)
        return {
            "dua": pos.dua,
            "transcript": text,
            "segment": pos.segment,
            "confidence": round(pos.dua_confidence * pos.segment_confidence, 3),
        }
