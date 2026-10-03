"""Streaming recognizer: audio in, (du'a, line, word) out.

Feed it audio in whatever chunk sizes arrive. Every `hop` seconds it
transcribes the last `window` seconds and moves the tracker. If ASR falls
behind real time, stale hops are skipped rather than queued: only the newest
window matters, and the tracker is told how much time actually passed.

With `words="ctc"` (and a ctc.CtcModel), a second, faster loop runs beside it:
every 0.1 s the word follower (follower.py) scores the last 3 s of a CTC
model's frames, anchored on the tracker's position, and says which word is
being recited (word_step). Opt-in: docs/results/follower.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import CorpusIndex
from .asr import DEFAULT_MODEL, SAMPLE_RATE, LiveQuiet, QuietMeter, transcribe_batch
from .corpus import Dua
from .follower import FollowerConfig, LocalFollower
from .tracker import Position, Tracker, TrackerConfig


@dataclass
class Update:
    t: float  # seconds of audio consumed
    transcript: str
    position: Position
    quiet: float = 0.0  # seconds the reciter had been silent at the window's end
    quiet_now: float | None = None  # ...and when the update was ready (asr.LiveQuiet; None: not known)
    paused: float | None = None  # seconds since the last update spent in pauses (asr.QuietMeter)
    voice_db: float | None = None  # the voice level the stop detector compares with


@dataclass
class WordUpdate:
    t: float  # seconds of audio consumed
    dua: str | None  # None: the follower has handed the display back to the tracker
    segment: int | None
    word: int | None  # index into the recognizer's current CorpusIndex.words


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
        lead: bool = True,  # show the predicted current position (docs/results/display_lead.md)
        pauses: bool = True,  # tell the tracker when the reciter stops (docs/results/pauses.md, stops.md)
        index: CorpusIndex | None = None,
        words: str | None = None,  # "ctc": also follow word by word (word_step), with `ctc`
        ctc=None,  # a ctc.CtcModel
        follower: FollowerConfig | None = None,
        word_hop: float = 0.1,
        word_window: float = 3.0,
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
        self.lead = lead
        self.pauses = pauses
        self.words = words if ctc is not None else None
        self.ctc = ctc
        self.follower_config = follower
        self.word_hop = int(word_hop * SAMPLE_RATE)
        self.word_window = int(word_window * SAMPLE_RATE)
        self.follower = LocalFollower(self.index, follower) if self.words == "ctc" else None
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self._buf = np.zeros(0, dtype=np.float32)
        self._total = 0  # samples received
        self._last = 0  # sample count at the last update
        self._wlast = 0  # ...and at the last word step
        self._anchor: int | None = None  # the tracker's evidence position (no lead), for the follower
        self._lead_word: int | None = None  # ...and the word it displays
        # The stop detector: per window (remembering the voice level), and on the audio as it
        # arrives, which goes on while a window is being transcribed.
        self._meter = QuietMeter()
        self._ear = LiveQuiet()
        if self.follower is not None:
            self.follower.reset()

    def lock(self, dua_id: str) -> None:
        """The listener said which du'a this is: follow only that one.

        Skips identification entirely (no wait, no chance of a wrong lock).
        The audio buffer is kept, so a lock mid-recitation loses nothing.
        """
        self.index = CorpusIndex({dua_id: self.duas[dua_id]})
        self.tracker = Tracker(self.index, self.config)
        self._anchor = self._lead_word = None  # word indices of the old index
        if self.follower is not None:
            self.follower = LocalFollower(self.index, self.follower_config)

    def seek(self, dua_id: str, segment: int) -> None:
        """The listener said where they are ("I'm here" on a line): follow on from there."""
        self.tracker.seek(dua_id, segment)
        self._anchor = self._lead_word = None

    def push(self, samples: np.ndarray) -> None:
        """Buffer 16 kHz float32 samples without running anything."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, samples])[-max(self.window, self.word_window) :]
        self._total += samples.size
        if self.pauses:
            self._ear.push(samples)

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
        window = self._buf[-self.window :].copy()
        end = self._total
        text = transcribe_batch([window], model=self.model, prompts=[prompt], vad=self.vad)[0]
        quiet = self._meter(window, dt) if self.pauses else 0.0
        paused = self._meter.paused if self.pauses else None
        # Audio that arrived while transcribing (the web server pushes
        # concurrently) is how far the reciter has moved on since `end`,
        # or how long they have been still.
        delay = (self._total - end) / SAMPLE_RATE
        quiet_now = None
        if self.pauses:
            self._ear.voice_db = self._meter.voice_db
            quiet_now = self._ear.quiet
        pos = self.tracker.update(text, dt, lead=delay + self.tracker.cfg.display_lead if self.lead else 0.0,
                                  quiet=quiet, quiet_now=quiet_now, paused=paused)
        if self.follower is not None:
            now = self.tracker.position()  # the evidence position, as word_eval.hmm_updates anchors it
            self._anchor = now.word if now.dua is not None else None
            self._lead_word = pos.word if pos.dua is not None else None
        return Update(end / SAMPLE_RATE, text, pos, quiet, quiet_now, paused, self._meter.voice_db)

    @property
    def word_due(self) -> bool:
        return self.follower is not None and self._total - self._wlast >= self.word_hop

    def word_step(self) -> WordUpdate | None:
        """One follower step on the last `word_window` s of audio, if one is due.

        Touches only the follower (step() only the tracker), so the server can run
        the two in separate threads; both read their own copy of the buffer."""
        if not self.word_due:
            return None
        end = self._total
        self._wlast = end
        y = self._buf[-self.word_window :].copy()
        w = self.follower.step(self.ctc.window(y), end / SAMPLE_RATE, self._anchor, self._lead_word)
        if w is None:
            return WordUpdate(end / SAMPLE_RATE, None, None, None)
        ix = self.index
        return WordUpdate(end / SAMPLE_RATE, ix.dua_ids[ix.word_dua[w]], int(ix.word_segment[w]), w)


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
