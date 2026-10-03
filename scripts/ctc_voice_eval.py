#!/usr/bin/env python
"""Streaming CER of CTC models on ordinary voices, as the phone follower hears them.

Each clip is cut into windows of `--window` s ending every `--hop` s; from each window
only its newest `--hop` s of frames are kept (the frames the follower sees first, with
no audio after them), and the greedy transcript of the stitched frames is scored
against the clip's text. RetaSy's crowd-sourced clips by default (voice_eval.py).

    python scripts/ctc_voice_eval.py models/ctc-student-base models/wav2vec2-quran-dua
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dua_recognition.align import _ALPHABET  # noqa: E402
from voice_eval import cer, load_clips, load_quranlab  # noqa: E402

INV = {v: k for k, v in _ALPHABET.items()}


def greedy(lp: np.ndarray) -> str:
    ids = lp.argmax(-1)
    return "".join(INV[int(i)] for j, i in enumerate(ids) if i != 0 and (j == 0 or i != ids[j - 1]))


def streamed(model, y: np.ndarray, window: float, hop: float) -> np.ndarray:
    n = int(np.ceil(len(y) / 16000 / hop))
    times = [round((k + 1) * hop, 3) for k in range(n)]
    arr, lens, _, _ = model.windows(np.r_[y, np.zeros(int(hop * 16000), np.float32)], times, window, 64)
    keep = int(round(hop / 0.02))
    return np.concatenate([arr[k, max(0, lens[k] - keep) : lens[k]] for k in range(n)]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--set", choices=["retasy", "quranlab", "cv"], default="retasy",
                    help="cv: Common Voice Arabic validation speakers (scripts/build_cv_set.py --split validation)")
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--hop", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--whole-only", action="store_true", help="skip the streaming pass (big models, long windows)")
    args = ap.parse_args()
    from dua_recognition.ctc_student import load_ctc

    if args.set == "cv":
        import json

        text = (ROOT / "data/cache/finetune/cv_ar_val.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(x) for x in text.splitlines() if x]
        clips = [{"y": np.load(ROOT / r["clip"]).astype(np.float32) / 32767, "ref": r["text"]} for r in rows[: args.limit]]
    else:
        clips = load_quranlab(limit=args.limit) if args.set == "quranlab" else load_clips(limit=args.limit)
    print(f"{len(clips)} clips, {sum(len(c['y']) for c in clips) / 16000 / 60:.0f} min")
    for name in args.models:
        m = load_ctc(name)
        tot = {"stream": [0, 0], "whole": [0, 0]}
        for c in clips:
            y = np.asarray(c["y"], dtype=np.float32)
            if not args.whole_only:
                e, n = cer(c["ref"], greedy(streamed(m, y, args.window, args.hop)))
                tot["stream"][0] += e
                tot["stream"][1] += n
            # Clips that fit one window: the whole clip at once, for every model alike.
            if len(y) <= args.window * 16000:
                arr, lens, _, _ = m.windows(y, [len(y) / 16000], len(y) / 16000 + 0.01, 1)
                e, n = cer(c["ref"], greedy(arr[0, : lens[0]].astype(np.float32)))
                tot["whole"][0] += e
                tot["whole"][1] += n
        f = {k: v[0] / max(1, v[1]) for k, v in tot.items()}
        print(f"{name:40s} streaming CER {f['stream']:.1%}   clips <= {args.window:g} s, whole: CER {f['whole']:.1%}",
              flush=True)
        del m


if __name__ == "__main__":
    main()
