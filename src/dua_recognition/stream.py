"""Pull overlapping audio windows off the microphone for streaming recognition."""
from __future__ import annotations

import queue

import numpy as np

SAMPLE_RATE = 16000


def mic_windows(window_s: float = 5.0, step_s: float = 2.0, samplerate: int = SAMPLE_RATE):
    """Yield ~window_s float32 windows, advancing ~step_s between yields.

    Blocks on the default input device until interrupted (Ctrl-C). sounddevice is
    imported here, not at module top, so the rest of the package works without
    PortAudio installed.
    """
    import sounddevice as sd

    win = int(window_s * samplerate)
    step = int(step_s * samplerate)
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    buf = np.zeros(0, dtype=np.float32)
    total = 0
    last_emit = 0
    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", callback=callback):
        while True:
            chunk = q.get()
            buf = np.concatenate([buf, chunk])
            total += len(chunk)
            if len(buf) > win:
                buf = buf[-win:]
            if len(buf) >= win and (total - last_emit) >= step:
                last_emit = total
                yield buf.copy()
