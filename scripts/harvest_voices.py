#!/usr/bin/env python
"""Voice check of the harvest (scripts/harvest.py) against every test voice.

Names in titles catch most uploads by test reciters; this catches the rest
(re-uploads under another name, unnamed recordings). Each harvested
recording's voice (speaker_check.Embedder: ECAPA, mean of 8 chunks) is scored
against the test reciters' named DuaPlayer/duas.pro recordings and every
recording of the held-out test sets (data/testsets). At >= speaker_check.LEAK
(0.7, where same-reciter pairs start) the recording is marked "test" and the
training export (harvest_label.py export) leaves it out.

    python scripts/harvest_voices.py            # follows new downloads until idle
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from harvest_label import HARVEST, PLATFORMS, read_jsonl, recordings  # noqa: E402
from speaker_check import LEAK, Embedder, canonical, named_recordings  # noqa: E402

from dua_recognition.splits import is_test  # noqa: E402

OUT = HARVEST / "voices.jsonl"
AUDIO = (".mp3", ".m4a", ".webm", ".wav", ".opus", ".ogg", ".flac", ".mp4")


def test_refs(emb: Embedder) -> list[tuple[str, np.ndarray]]:
    refs = []
    for name, p in named_recordings():
        if is_test(canonical(name)):
            e = emb(p)
            if e is not None:
                refs.append((canonical(name), e))
    for p in sorted((ROOT / "data" / "testsets").glob("*/**/*")):
        if p.suffix.lower() in AUDIO:
            e = emb(p)
            if e is not None:
                refs.append((f"testset:{p.parent.name}/{p.stem}", e))
    return refs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idle-exit", type=int, default=60, help="minutes without new recordings before exiting")
    args = ap.parse_args()
    emb = Embedder()
    refs = test_refs(emb)
    names = [n for n, _ in refs]
    R = np.stack([e for _, e in refs])
    print(f"{len(refs)} test voices ({len({n for n in names if not n.startswith('testset:')})} named reciters)",
          flush=True)
    done = {(r["platform"], r["id"]) for r in read_jsonl(OUT)}
    idle, flagged = 0, 0
    while True:
        todo = [(p, m, a) for p, m, a in recordings(PLATFORMS) if (p, m["id"]) not in done]
        if not todo:
            idle += 1
            if idle > args.idle_exit:
                break
            time.sleep(60)
            continue
        idle = 0
        for p, meta, audio in todo:
            row = {"platform": p, "id": meta["id"]}
            try:
                e = emb(audio)
            except Exception as err:  # noqa: BLE001  (ffprobe says "N/A" for some broken files)
                e = None
                row["error"] = str(err)[:100]
            if e is None:
                row.update(verdict="error" if "error" in row else "short", score=None)
            else:
                s = R @ e
                k = int(np.argmax(s))
                row.update(score=round(float(s[k]), 3), nearest=names[k],
                           verdict="test" if s[k] >= LEAK else "ok")
                if row["verdict"] == "test":
                    flagged += 1
                    print(f"  TEST VOICE {p}/{meta['id']}: {names[k]} {s[k]:.2f}  {str(meta.get('title'))[:60]}",
                          flush=True)
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done.add((p, meta["id"]))
        print(f"{len(done)} checked, {flagged} flagged this run", flush=True)


if __name__ == "__main__":
    main()
