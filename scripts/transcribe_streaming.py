#!/usr/bin/env python
"""Run the growing-buffer streaming front end over recordings, for evaluate.py.

Replays each recording hop by hop through BufferedTranscriber and writes, per
hop, the text the tracker would see, in the usual window-cache format, under
two tags:

    <model>+stream       last K committed words + the tentative tail
    <model>+committed    last K committed words only (LocalAgreement as-is)

    python scripts/transcribe_streaming.py --model models/whisper-base-quran-dua-ct2 --tag whisper-base-quran-dua
    python scripts/evaluate.py --asr whisper-base-quran-dua+stream
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from faster_whisper.audio import decode_audio  # noqa: E402
from transcribe_windows import SR, cache_path  # noqa: E402

from dua_recognition.asr import load_model  # noqa: E402
from dua_recognition.corpus import load_all, load_recordings  # noqa: E402
from dua_recognition.splits import is_test  # noqa: E402
from dua_recognition.streaming import BufferedTranscriber  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--context-words", type=int, default=12)
    ap.add_argument("--trim-after", type=float, default=12.0)
    ap.add_argument("--max-buffer", type=float, default=20.0)
    args = ap.parse_args()
    load_model(args.model)

    for dua in load_all().values():
        for rec in load_recordings(dua):
            if not is_test(rec.reciter):
                continue
            outs = {k: cache_path(f"{args.tag}+{k}", rec.audio_id, 6.0, args.hop) for k in ("stream", "committed")}
            if all(p.exists() for p in outs.values()):
                continue
            y = decode_audio(str(rec.path), sampling_rate=SR)
            bt = BufferedTranscriber(args.model, args.trim_after, args.max_buffer)
            hop = int(args.hop * SR)
            rows = {"stream": [], "committed": []}
            t0 = time.time()
            n = int((rec.end_s + 1e-6) // args.hop)
            for k in range(n):
                bt.push(y[k * hop : (k + 1) * hop])
                committed, tentative = bt.process()
                t = round((k + 1) * args.hop, 3)
                tail = [w.text for w in committed[-args.context_words :]]
                rows["committed"].append({"t": t, "text": " ".join(tail)})
                rows["stream"].append({"t": t, "text": " ".join(tail + [w.text for w in tentative])})
            for k, path in outs.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows[k]), encoding="utf-8")
            took = time.time() - t0
            print(f"{dua.id:28s} {rec.reciter[:20]:20s} {n:5d} hops  {took / n * 1000:.0f} ms/hop", flush=True)


if __name__ == "__main__":
    main()
