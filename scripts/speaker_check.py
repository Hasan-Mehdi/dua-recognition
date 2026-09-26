#!/usr/bin/env python
"""Who is reciting? Match recordings to known reciters by voice.

The train/test split is by reciter, and it's only as good as the reciter
names. Recordings from duas.org or YouTube often have no name, or one spelled
differently, so a test reciter could slip into training. This embeds each
recording's voice (speechbrain ECAPA, averaged over several 8 s chunks) and
compares it with every named reciter in the DuaPlayer / duas.pro sets.

    python scripts/speaker_check.py data/duasorg_timed/*/*.mp3
    python scripts/speaker_check.py --youtube          # every harvested YouTube recording

Prints, per recording, the closest known reciters and their cosine scores.
Same-reciter pairs score far above different-reciter ones (--calibrate shows
both distributions on the named set), so a high score against a test reciter
means: keep this recording out of training.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.splits import is_test  # noqa: E402

CACHE = ROOT / "data" / "cache" / "speaker"
SR = 16000


def decode(path: Path, start: float, dur: float) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(dur), "-i", str(path),
                        "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


class Embedder:
    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir=str(ROOT / "models" / "spkrec-ecapa"),
                                                    run_opts={"device": "cpu"},
                                                    local_strategy=LocalStrategy.COPY)  # no symlinks on Windows

    def __call__(self, path: Path, chunks: int = 8, chunk_s: float = 8.0) -> np.ndarray | None:
        key = CACHE / (hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16] + ".npy")
        if key.exists():
            return np.load(key)
        total = duration(path)
        if total < 20:
            return None
        # Skip the first and last 10% (intros, crowd, closing du'as by someone else).
        starts = np.linspace(0.1 * total, 0.9 * total - chunk_s, chunks)
        embs = []
        for s in starts:
            wav = decode(path, float(s), chunk_s)
            if len(wav) < SR * 2 or np.abs(wav).max() < 1e-3:
                continue
            with torch.no_grad():
                e = self.model.encode_batch(torch.from_numpy(wav.copy()).unsqueeze(0)).squeeze().numpy()
            embs.append(e / np.linalg.norm(e))
        if not embs:
            return None
        v = np.mean(embs, axis=0)
        v /= np.linalg.norm(v)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(key, v)
        return v


def named_recordings() -> list[tuple[str, Path]]:
    out = []
    for meta in sorted(glob.glob(str(ROOT / "data" / "duaplayer" / "*" / "*-*.json"))) + \
            sorted(glob.glob(str(ROOT / "data" / "duaspro" / "*" / "*.json"))):
        if meta.endswith(".repaired.json"):
            continue
        m = json.loads(Path(meta).read_text(encoding="utf-8"))
        mp3 = Path(meta).with_suffix(".mp3")
        if mp3.exists() and m.get("reciter") and m["reciter"] != "Unknown":
            out.append((m["reciter"], mp3))
    return out


def canonical(name: str) -> str:
    # Two spellings of one reciter on DuaPlayer.
    return {"Murtaza Quraish": "Murtada al-Qureish"}.get(name, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--youtube", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="score distributions among the named recordings")
    ap.add_argument("--json", type=Path, help="also write results here")
    args = ap.parse_args()
    emb = Embedder()

    ref = [(canonical(n), p, emb(p)) for n, p in named_recordings()]
    ref = [(n, p, e) for n, p, e in ref if e is not None]
    names = sorted({n for n, _, _ in ref})
    print(f"{len(ref)} named recordings, {len(names)} reciters", flush=True)

    if args.calibrate:
        same, diff = [], []
        for i in range(len(ref)):
            for j in range(i + 1, len(ref)):
                s = float(ref[i][2] @ ref[j][2])
                (same if ref[i][0] == ref[j][0] else diff).append(s)
        for label, xs in (("same reciter", same), ("different reciters", diff)):
            xs = np.array(xs)
            print(f"  {label:18s} n={len(xs):4d}  min {xs.min():.2f}  p5 {np.percentile(xs, 5):.2f}  "
                  f"median {np.median(xs):.2f}  p95 {np.percentile(xs, 95):.2f}  max {xs.max():.2f}")

    paths = [Path(p) for p in args.paths]
    if args.youtube:
        paths += [p for p in sorted((ROOT / "data" / "youtube").glob("*/*"))
                  if p.suffix in (".webm", ".m4a", ".mp3")]
    results = []
    for p in paths:
        e = emb(p)
        if e is None:
            print(f"  {p.name}: too short / silent")
            continue
        best: dict[str, float] = {}
        for n, _, r in ref:
            best[n] = max(best.get(n, -1.0), float(e @ r))
        top = sorted(best.items(), key=lambda kv: -kv[1])[:3]
        flag = " TEST" if is_test(top[0][0]) else ""
        print(f"  {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p}: "
              + ", ".join(f"{n} {s:.2f}" for n, s in top) + flag, flush=True)
        results.append({"path": str(p), "top": top})
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
