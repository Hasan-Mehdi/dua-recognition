#!/usr/bin/env python
"""Turn data/raw/<dua_id>/*.wav into normalized 16 kHz mono segments.

Resample -> peak-normalize -> trim leading/trailing silence -> split into
overlapping windows. Output mirrors the input tree under data/processed/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SR = 16000


def trim_silence(y: np.ndarray, top_db: int = 30) -> np.ndarray:
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt


def windows(y: np.ndarray, window_s: float, step_s: float):
    win, step = int(window_s * SR), int(step_s * SR)
    for start in range(0, max(1, len(y) - win + 1), step):
        yield y[start : start + win]


def process_file(path: Path, dua_id: str, window_s: float, step_s: float) -> int:
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = trim_silence(y)
    if y.size:
        y = y / (np.abs(y).max() + 1e-9)  # peak normalize
    out_dir = PROCESSED / dua_id
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, chunk in enumerate(windows(y, window_s, step_s)):
        sf.write(out_dir / f"{path.stem}_{i:04d}.wav", chunk, SR)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=4.0)
    args = ap.parse_args()

    total = 0
    for wav in RAW.rglob("*.wav"):
        dua_id = wav.parent.name
        total += process_file(wav, dua_id, args.window, args.step)
    print(f"wrote {total} segments to {PROCESSED}")


if __name__ == "__main__":
    main()
