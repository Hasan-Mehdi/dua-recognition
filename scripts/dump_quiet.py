#!/usr/bin/env python
"""Seconds since the reciter last made a sound, at the end of every 1 s hop.

Runs the live rule (asr.quiet_at_end: voice detector or energy over the
window's floor) on each 6 s window of each recording and caches the series,
so evaluation can replay the "hold still while they're silent" behaviour.

    python scripts/dump_quiet.py            # -> data/cache/quiet/<audio_id>.json (test + train reciters)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.asr import quiet_at_end  # noqa: E402
from dua_recognition.corpus import load_all, load_recordings  # noqa: E402

OUT = ROOT / "data" / "cache" / "quiet"
SR = 16000


def quiet_series(y: np.ndarray, hop: float = 1.0, window: float = 6.0) -> list[float]:
    """quiet_at_end at the end of every hop, as the live recognizer computes it."""
    return [round(quiet_at_end(y[int(max(0.0, t - window) * SR) : int(t * SR)]), 3)
            for t in ((k + 1) * hop for k in range(int(y.size / SR / hop)))]


def main() -> None:
    from faster_whisper.audio import decode_audio

    OUT.mkdir(parents=True, exist_ok=True)
    for dua in load_all().values():
        for rec in load_recordings(dua):
            f = OUT / f"{rec.audio_id}.json"
            if f.exists():
                continue
            f.write_text(json.dumps(quiet_series(decode_audio(str(rec.path), sampling_rate=SR))))
            print(rec.audio_id, flush=True)


if __name__ == "__main__":
    main()
