#!/usr/bin/env python
"""The demo's script: what the server tells the page, second by second, for a real recording.

Streams a held-out DuaPlayer recording through StreamingRecognizer exactly as app/server.py
does (0.25 s chunks, a step whenever a hop is due and the last one has finished, audio that
arrives during a step counted as the reciter moving on) and times every step on this
machine. The page receives each message at window end + measured compute time. Also writes
the recording's loudness, which the demo stage draws beside the phone with what Whisper
heard at each step (the messages' "heard").

    python docs/demo/timeline.py                          # -> docs/demo/timeline.json
    python docs/demo/timeline.py --model models/whisper-turbo-dua-ct2 --start 312 --seconds 46

The default window (Hussein Ghareeb's Tawassul from 312 s) was chosen by scoring 17 start
points of that recording: found in 4.1 s, the line right on 37 of 42 updates and within
one line on all of them. Cold starts inside a refrain do worse, as they must: until a
unique line comes, nothing says which repetition it is.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from dua_recognition.asr import SAMPLE_RATE, quiet_at_end, transcribe_batch  # noqa: E402
from dua_recognition.corpus import load_all, load_recordings  # noqa: E402
from dua_recognition.pipeline import StreamingRecognizer  # noqa: E402

CHUNK = 4000  # the page posts ~250 ms chunks (capture-worklet.js)


def stream(rec: StreamingRecognizer, audio: np.ndarray) -> list[dict]:
    """(arrival s, server message) for every step, as the server would send them."""
    import server  # app/server.py: the message the page receives

    out = []
    busy_until = 0.0
    pos = 0
    while pos < audio.size:
        rec.push(audio[pos : pos + CHUNK])
        pos += CHUNK
        now = pos / SAMPLE_RATE
        if not rec.due or now < busy_until:
            continue
        # StreamingRecognizer.step, with the compute time taken from the clock and the audio
        # that arrives meanwhile pushed before the tracker moves (as the server's threads do).
        dt = (rec._total - rec._last) / SAMPLE_RATE
        rec._last = rec._total
        prompt = rec.tracker.prompt() if rec.prompt_bias else None
        window = rec._buf[-rec.window :].copy()
        end = rec._total
        t0 = time.perf_counter()
        text = transcribe_batch([window], model=rec.model, prompts=[prompt], vad=rec.vad)[0]
        quiet = quiet_at_end(window) if rec.pauses else 0.0
        compute = time.perf_counter() - t0
        while pos < audio.size and pos / SAMPLE_RATE < now + compute:
            rec.push(audio[pos : pos + CHUNK])
            pos += CHUNK
        delay = (rec._total - end) / SAMPLE_RATE
        p = rec.tracker.update(text, dt, lead=delay + rec.tracker.cfg.display_lead, quiet=quiet)
        from dua_recognition.pipeline import Update

        msg = server._message(Update(end / SAMPLE_RATE, text, p, quiet), compute * 1000, rec.index,
                              rec.tracker.speed, rec.tracker.null, rec.tracker.cfg.still_after)
        busy_until = now + compute
        out.append({"at": round(busy_until + 0.02, 3), "msg": msg})  # + a local WebSocket hop
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dua", default="dua-tawassul")
    ap.add_argument("--reciter", default="Hussein Ghareeb")
    ap.add_argument("--model", default=str(ROOT / "models" / "whisper-turbo-dua-ct2"))
    ap.add_argument("--start", type=float, default=312.0)  # line 40, then the refrains 41-45
    ap.add_argument("--seconds", type=float, default=46.0)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("timeline.json"))
    args = ap.parse_args()

    from faster_whisper.audio import decode_audio

    duas = load_all()
    dua = duas[args.dua]
    rec_meta = next(r for r in load_recordings(dua) if r.reciter == args.reciter)
    full = decode_audio(str(rec_meta.path), sampling_rate=SAMPLE_RATE)
    a, b = int(args.start * SAMPLE_RATE), int((args.start + args.seconds) * SAMPLE_RATE)
    audio = full[a:b].astype(np.float32)

    transcribe_batch([audio[: 6 * SAMPLE_RATE]], model=args.model)  # load and warm the model first
    rec = StreamingRecognizer(duas, model=args.model)
    steps = stream(rec, audio)

    # Loudness for the waveform (10 ms RMS) and for the listening star (the 100-4000 Hz band
    # power the page's analyser would see, every 1/60 s).
    hop = SAMPLE_RATE // 100
    rms = np.sqrt(np.mean(audio[: audio.size // hop * hop].reshape(-1, hop) ** 2, axis=1))
    band = []
    n = 1024
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    sel = (freqs >= 100) & (freqs <= 4000)
    for k in range(int(args.seconds * 60)):
        c = int(k / 60 * SAMPLE_RATE)
        x = audio[max(0, c - n) : c]
        if x.size < n:
            x = np.pad(x, (n - x.size, 0))
        spec = np.abs(np.fft.rfft(x * win)) ** 2
        band.append(round(float(10 * np.log10(spec[sel].sum() + 1e-12)), 1))

    found = next((st["at"] for st in steps if st["msg"]["dua"]), None)
    print(f"{len(steps)} steps, compute p50 {np.median([st['msg']['step_ms'] for st in steps]):.0f} ms; "
          f"{dua.name_en} shown after {found} s")
    for st in steps:
        m = st["msg"]
        print(f"  {st['at']:6.2f}  {str(m['dua'])[:14]:14s} line {m['segment']}  tok {m['token']}  "
              f"q {m['quiet']:.1f}  {m['heard'][:40]}")
    args.out.write_text(json.dumps({
        "source": {"dua": args.dua, "name": dua.name_en, "reciter": args.reciter, "audio_id": rec_meta.audio_id,
                   "start": args.start, "seconds": args.seconds, "model": Path(args.model).name},
        "steps": steps, "rms": [round(float(x), 4) for x in rms], "band_db": band,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
