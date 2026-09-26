#!/usr/bin/env python
"""How well does each ASR model hear ordinary voices?

The DuaPlayer test reciters are professionals, like the training data. This
scores models on RetaSy's crowd-sourced Quran recitations (1,289 volunteers,
half women, mostly non-Arabic speakers; huggingface.co/datasets/RetaSy/quranic_audio_dataset),
using the clips annotators marked as correct (Quran verses, plus some du'as and
adhkar) against the text each was reciting. CER on normalized letters.

    python scripts/voice_eval.py models/whisper-base-quran-dua-ct2 large-v3-turbo
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.text import normalize  # noqa: E402

DATA = ROOT / "data" / "cache" / "retasy"


def load_clips(labels=("correct",), limit: int | None = None):
    import pyarrow.parquet as pq
    import soundfile as sf

    clips = []
    for f in sorted(DATA.glob("train-*.parquet")):
        t = pq.read_table(f).to_pandas()
        for _, r in t[t.final_label.isin(labels)].iterrows():
            ref = r.Aya  # the text that was to be recited
            if not isinstance(ref, str) or not ref.strip():
                continue
            y, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != 16000:
                from scipy.signal import resample_poly
                y = resample_poly(y, 16000, sr).astype(np.float32)
            clips.append({"y": y, "ref": ref, "reciter": r.reciter_id, "gender": r.reciter_gender})
            if limit and len(clips) >= limit:
                return clips
    return clips


def cer(ref: str, hyp: str) -> tuple[int, int]:
    import jiwer

    r = normalize(ref).replace(" ", "")
    h = normalize(hyp).replace(" ", "")
    if not r:
        return 0, 0
    return round(jiwer.cer(r, h or "-") * len(r)), len(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from faster_whisper import WhisperModel

    clips = load_clips(limit=args.limit)
    print(f"{len(clips)} clips, {len({c['reciter'] for c in clips})} reciters, "
          f"{sum(c['gender'] == 'female' for c in clips)} by women, {sum(len(c['y']) for c in clips) / 16000 / 60:.0f} min")
    for name in args.models:
        m = WhisperModel(name, device=args.device, compute_type="int8" if args.device == "cpu" else "float16")
        tot = {"all": [0, 0], "male": [0, 0], "female": [0, 0]}
        for c in clips:
            segs, _ = m.transcribe(c["y"], language="ar", beam_size=1, condition_on_previous_text=False,
                                   without_timestamps=True)
            e, n = cer(c["ref"], " ".join(s.text for s in segs))
            for k in ("all", c["gender"]):
                if k in tot:
                    tot[k][0] += e
                    tot[k][1] += n
        f = {k: v[0] / max(v[1], 1) for k, v in tot.items()}
        print(f"{name:45s} CER {f['all']:.1%}   men {f['male']:.1%}   women {f['female']:.1%}", flush=True)


if __name__ == "__main__":
    main()
