#!/usr/bin/env python
"""Live recognition from the mic, or a one-shot run over an audio file.

    python app/demo.py                 # listen on the default mic
    python app/demo.py clip.wav        # recognize a single file

Prints the current du'a, the matched segment, and its text as you recite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dua_recognition import Recognizer, load_all  # noqa: E402


def show(duas, result: dict) -> None:
    seg_id = result["segment"]
    text = ""
    if seg_id is not None:
        seg = next(s for s in duas[result["dua"]].segments if s.id == seg_id)
        text = seg.arabic
    name = duas[result["dua"]].name_en
    print(f"[{name}] seg {seg_id}  ({result['confidence']:.2f})  {text}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="?", help="audio file; omit to use the mic")
    args = ap.parse_args()

    duas = load_all()
    if not duas:
        raise SystemExit("no du'as loaded — add reference texts under data/duas/")
    rec = Recognizer(duas)

    if args.audio:
        show(duas, rec.recognize(args.audio))
        return

    from dua_recognition.stream import mic_windows

    print("listening — Ctrl-C to stop")
    try:
        for window in mic_windows():
            show(duas, rec.recognize(window))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
