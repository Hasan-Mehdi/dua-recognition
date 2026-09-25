"""Recognize which du'a is being recited and follow it line by line, in real time.

The public surface is intentionally small: load the reference texts, build a
StreamingRecognizer, feed it audio. The heavier pieces (faster-whisper,
sounddevice) are imported lazily so importing the package itself stays cheap.
"""
from .align import CorpusIndex
from .corpus import Dua, Recording, Segment, load_all, load_dua, load_recordings
from .match import Match, PassageMatcher
from .pipeline import Recognizer, StreamingRecognizer, Update
from .tracker import Position, Tracker, TrackerConfig

__all__ = [
    "CorpusIndex",
    "Dua",
    "Match",
    "PassageMatcher",
    "Position",
    "Recognizer",
    "Recording",
    "Segment",
    "StreamingRecognizer",
    "Tracker",
    "TrackerConfig",
    "Update",
    "load_all",
    "load_dua",
    "load_recordings",
]
