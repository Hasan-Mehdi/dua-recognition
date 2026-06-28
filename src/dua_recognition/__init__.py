"""Recognize which du'a is being recited and where in it, in real time.

The public surface is intentionally small: load the reference texts, build a
Recognizer, feed it audio. The heavier pieces (faster-whisper, sounddevice) are
imported lazily so importing the package itself stays cheap.
"""
from .corpus import Dua, Segment, load_dua, load_all
from .match import Match, PassageMatcher
from .pipeline import Recognizer

__all__ = [
    "Dua",
    "Segment",
    "load_dua",
    "load_all",
    "Match",
    "PassageMatcher",
    "Recognizer",
]
