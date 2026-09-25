"""Pull audio off the microphone for streaming recognition."""
from __future__ import annotations

import queue

import numpy as np

SAMPLE_RATE = 16000


def mic_chunks(chunk_s: float = 0.25, samplerate: int = SAMPLE_RATE):
    """Yield float32 mono chunks of ~chunk_s seconds from the default input.

    Blocks until interrupted (Ctrl-C). sounddevice is imported here, not at
    module top, so the rest of the package works without PortAudio installed.
    Windowing is the recognizer's job (pipeline.StreamingRecognizer).
    """
    import sounddevice as sd

    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=samplerate, channels=1, dtype="float32",
        blocksize=int(chunk_s * samplerate), callback=callback,
    ):
        while True:
            yield q.get()
