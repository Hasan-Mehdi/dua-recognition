"""Thin wrapper around faster-whisper for Arabic transcription.

Whisper is the backbone for passage localization: transcribe the window, then
fuzzy-match the text. Du'a Arabic carries vocabulary (names, set phrases) that
general Arabic ASR handles unevenly, which is the motivation for fine-tuning
later — see scripts/finetune_whisper.py.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def _model(size: str = "medium", device: str = "auto", compute_type: str = "int8"):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device=device, compute_type=compute_type)


def transcribe(audio, *, size: str = "medium", language: str = "ar") -> str:
    """Transcribe a clip.

    audio: a path to an audio file, or a float32 numpy array sampled at 16 kHz.
    """
    segments, _ = _model(size).transcribe(audio, language=language, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()
