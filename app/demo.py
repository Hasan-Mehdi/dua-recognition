#!/usr/bin/env python
"""Follow a recitation in the terminal, from the mic or from a file.

    python app/demo.py                  # listen on the default mic
    python app/demo.py recitation.mp3   # stream a file through, as if live

Prints a line whenever the recognized du'a or line changes. For the browser
version with synced text, run app/server.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dua_recognition import StreamingRecognizer, load_all  # noqa: E402
from dua_recognition.asr import DEFAULT_MODEL, SAMPLE_RATE  # noqa: E402


def show(duas, update, last):
    p = update.position
    key = (p.dua, p.segment)
    if key == last:
        return last
    if p.dua is None:
        guess = ", ".join(f"{duas[d].name_en} {pr:.0%}" for d, pr in p.candidates)
        print(f"[{update.t:7.1f}s] listening…  ({guess})")
    else:
        seg = next(s for s in duas[p.dua].segments if s.id == p.segment)
        print(f"[{update.t:7.1f}s] {duas[p.dua].name_en} · line {p.segment}  {seg.arabic}")
    return key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", help="audio file; omit to use the mic")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    duas = load_all()
    rec = StreamingRecognizer(duas, model=args.model)
    last = None

    if args.audio:
        from faster_whisper.audio import decode_audio

        y = decode_audio(args.audio, sampling_rate=SAMPLE_RATE)
        chunk = SAMPLE_RATE // 4
        for i in range(0, y.size, chunk):
            update = rec.feed(y[i : i + chunk])
            if update:
                last = show(duas, update, last)
        return

    from dua_recognition.stream import mic_chunks

    print("listening — Ctrl-C to stop")
    try:
        for chunk in mic_chunks():
            update = rec.feed(chunk)
            if update:
                last = show(duas, update, last)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
