#!/usr/bin/env python
"""Train the audio du'a classifier on processed segments.

Skeleton. Reads data/processed/<dua_id>/*.wav (the labels are the folder names),
turns each clip into a log-mel spectrogram, and trains a small classifier head.
The text-only TextClassifier already works for the recognizer; this is the
faster, melody-aware replacement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def labelled_clips():
    for wav in PROCESSED.rglob("*.wav"):
        yield wav, wav.parent.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="models/dua-classifier")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    clips = list(labelled_clips())
    labels = sorted({label for _, label in clips})
    print(f"{len(clips)} clips across {len(labels)} du'as: {labels}")
    if not clips:
        raise SystemExit("no processed clips — run scripts/preprocess.py first")

    # TODO: log-mel features (librosa) -> a small CNN or an AST head; stratified
    # train/val split by reciter; cross-entropy; save best model + label map to
    # args.out. Report per-class precision/recall and a confusion matrix.


if __name__ == "__main__":
    main()
