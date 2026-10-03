#!/usr/bin/env python
"""Seconds since the reciter last made a sound, at the end of every 1 s hop.

Runs the live rule (asr.quiet_at_end: voice detector or energy over the
window's floor) on each 6 s window of each recording and caches the series,
so evaluation can replay the "hold still while they're silent" behaviour.

    python scripts/dump_quiet.py            # -> data/cache/quiet/<audio_id>.json (test + train reciters)
    python scripts/dump_quiet.py --v2       # -> data/cache/quiet2/<audio_id>.json

--v2 is the stop detector as it is now (docs/results/stops.md): per hop, the
window's quiet with the voice-relative energy rule (asr.QuietMeter, which
remembers the voice level), the seconds of that hop spent in pauses, and the
page's live detector (asr.LiveQuiet) every 0.25 s, its voice level arriving a
second after each window ends (Whisper's time). scripts/evaluate.py --stop2.
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
OUT2 = ROOT / "data" / "cache" / "quiet2"
SR = 16000
CHUNK = 4000  # the page's capture chunks (250 ms)


def quiet_series(y: np.ndarray, hop: float = 1.0, window: float = 6.0) -> list[float]:
    """quiet_at_end at the end of every hop, as the live recognizer computes it."""
    return [round(quiet_at_end(y[int(max(0.0, t - window) * SR) : int(t * SR)]), 3)
            for t in ((k + 1) * hop for k in range(int(y.size / SR / hop)))]


def stop_series(y: np.ndarray, hop: float = 1.0, window: float = 6.0) -> dict:
    """--v2: quiet and pauses per hop (asr.QuietMeter), the live detector every chunk."""
    from dua_recognition.asr import LiveQuiet, QuietMeter

    meter, live = QuietMeter(), LiveQuiet()
    quiet, paused, voice = [], [], []
    ends = [(k + 1) * hop for k in range(int(y.size / SR / hop))]
    for t in ends:
        quiet.append(round(meter(y[int(max(0.0, t - window) * SR) : int(t * SR)], hop), 3))
        paused.append(round(meter.paused, 3))
        voice.append(meter.voice_db)
    series, k = [], 0
    for a in range(0, y.size - CHUNK + 1, CHUNK):
        live.push(y[a : a + CHUNK])
        now = (a + CHUNK) / SR
        while k < len(ends) and ends[k] + 1.0 <= now:
            if voice[k] is not None:
                live.voice_db = voice[k]
            k += 1
        series.append(None if live.quiet is None else round(live.quiet, 3))
    return {"hop": hop, "quiet": quiet, "paused": paused, "live": series, "live_step": CHUNK / SR}


def main() -> None:
    from faster_whisper.audio import decode_audio

    v2 = "--v2" in sys.argv[1:]
    out = OUT2 if v2 else OUT
    out.mkdir(parents=True, exist_ok=True)
    for dua in load_all().values():
        for rec in load_recordings(dua):
            f = out / f"{rec.audio_id}.json"
            if f.exists():
                continue
            y = decode_audio(str(rec.path), sampling_rate=SR)
            f.write_text(json.dumps(stop_series(y) if v2 else quiet_series(y)))
            print(rec.audio_id, flush=True)


if __name__ == "__main__":
    main()
