#!/usr/bin/env python
"""Harvest training windows as small Opus clips, for training Whisper on the harvest.

finetune_whisper.py's AudioBank decodes every recording a training set names to raw PCM
on disk; the harvest's 6,159 recordings would be ~200 GB, and the disks are full. This
keeps only the windows: a voice-balanced sample of train_h2's harvest rows (at most
--per-uploader windows per uploader, from at most --per-recording windows per
recording), each 6 s window decoded from its recording and written as Opus (~20 KB),
plus a .jsonl whose rows name the clip instead of the recording ("clip": path).

    python scripts/build_clip_cache.py --name h2c            # -> D:\\dua-data\\clips\\h2c\\, train_h2c.jsonl
    python scripts/finetune_whisper.py --data h2c ...        # train_v4 + these clips
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "data" / "cache" / "finetune"
CLIPS = Path(r"D:\dua-data\clips")
SR = 16000
HARVEST = ("yt", "web", "aparat", "soundcloud", "archive")


def decode(path: str, a: float, b: float) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", path,
                          "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16)


def sample(rows: list[dict], per_uploader: int, per_recording: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_up: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_up[r["reciter"]][r["rec"]].append(r)
    out = []
    for up in sorted(by_up):
        recs = sorted(by_up[up])
        rng.shuffle(recs)
        take = []
        for rec in recs:  # spread over the uploader's recordings: different du'as, rooms, days
            ws = by_up[up][rec]
            take += rng.sample(ws, min(per_recording, len(ws)))
            if len(take) >= per_uploader:
                break
        out += take[:per_uploader]
    return out


def build_recording(rows: list[dict], out_dir: Path) -> list[dict]:
    import soundfile as sf

    rows = sorted(rows, key=lambda r: r["start"])
    a, b = rows[0]["start"], rows[-1]["end"]
    y = decode(rows[0]["audio"], a, b) if b - a < 1800 else None
    done = []
    for r in rows:
        name = hashlib.sha1(f"{r['rec']}:{r['start']:.2f}".encode()).hexdigest()[:16]
        f = out_dir / name[:2] / f"{name}.ogg"
        if not f.exists():
            seg = y[int((r["start"] - a) * SR) : int((r["end"] - a) * SR)] if y is not None \
                else decode(r["audio"], r["start"], r["end"])
            if seg.size < SR:  # the file ends early or the window is unreadable
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            sf.write(f, seg.astype(np.float32) / 32767, SR, format="OGG", subtype="OPUS")
        row = {k: v for k, v in r.items() if k not in ("audio", "start", "end")}
        done.append({**row, "clip": str(f), "seconds": round(r["end"] - r["start"], 3)})
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="h2c")
    ap.add_argument("--rows", default="train_h2")
    ap.add_argument("--per-uploader", type=int, default=60)
    ap.add_argument("--per-recording", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(x) for x in (FT / f"{args.rows}.jsonl").read_text(encoding="utf-8").splitlines() if x]
    base = [r for r in rows if r.get("source") not in HARVEST]
    picked = sample([r for r in rows if r.get("source") in HARVEST], args.per_uploader, args.per_recording, args.seed)
    by_rec: dict = defaultdict(list)
    for r in picked:
        by_rec[r["rec"]].append(r)
    out_dir = CLIPS / args.name
    print(f"{len(picked)} harvest windows from {len(by_rec)} recordings, "
          f"{len({r['reciter'] for r in picked})} uploaders; {len(base)} non-harvest rows kept as they are", flush=True)
    clips, n = [], 0
    with ThreadPoolExecutor(args.workers) as pool:  # ffmpeg does the work in its own processes
        for got in pool.map(lambda rs: build_recording(rs, out_dir), by_rec.values()):
            clips += got
            n += 1
            if n % 200 == 0:
                print(f"  {n}/{len(by_rec)} recordings, {len(clips)} clips", flush=True)
    (FT / f"train_{args.name}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in base + clips), encoding="utf-8")
    val = FT / f"val_{args.rows.removeprefix('train_')}.jsonl"
    if val.exists():  # the same validation windows (whole recordings; few)
        (FT / f"val_{args.name}.jsonl").write_text(val.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"{len(base)} + {len(clips)} rows -> train_{args.name}.jsonl", flush=True)


if __name__ == "__main__":
    main()
