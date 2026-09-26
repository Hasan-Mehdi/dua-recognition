#!/usr/bin/env python
"""Crowd-sourced training clips: ordinary voices for the fine-tune.

RetaSy's Quranic Audio Dataset (huggingface.co/datasets/RetaSy/quranic_audio_dataset)
has ~6,800 short recitations by 1,289 volunteers, half of them women. Every
reciter with a clip annotators marked "correct" is kept out: those clips are
scripts/voice_eval.py's test set. Of the rest, only unannotated clips are used
(not those marked as mistakes), and only where a strong model's transcript
agrees with the intended text (CER <= --max-cer), since unannotated
recitations can be wrong or cut short.

    python scripts/build_crowd_set.py --filter-model large-v3-turbo --device cuda
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from voice_eval import DATA, cer  # noqa: E402

# Harakat, Quranic annotation marks, tatweel: the training labels are undiacritized.
_MARKS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")


def label(text: str) -> str:
    t = _MARKS.sub("", text).replace("ٱ", "ا")  # alef wasla -> alef
    return " ".join(t.split())


def main() -> None:
    import pyarrow.parquet as pq
    import soundfile as sf
    from faster_whisper import WhisperModel

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter-model", default="large-v3-turbo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-cer", type=float, default=0.25)
    args = ap.parse_args()

    tables = [pq.read_table(f).to_pandas() for f in sorted(DATA.glob("train-*.parquet"))]
    test_reciters = {r for t in tables for r in t[t.final_label == "correct"].reciter_id}
    m = WhisperModel(args.filter_model, device=args.device, compute_type="float16" if args.device == "cuda" else "int8")
    out_dir = DATA / "clips"
    out_dir.mkdir(exist_ok=True)
    rows, seen, kept = [], 0, 0
    for t in tables:
        for i, r in t[t.final_label.isna() & ~t.reciter_id.isin(test_reciters)].iterrows():
            if not isinstance(r.Aya, str) or not r.Aya.strip() or not 1000 <= r.duration_ms <= 29000:
                continue
            seen += 1
            y, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != 16000:
                from scipy.signal import resample_poly
                y = resample_poly(y, 16000, sr).astype(np.float32)
            segs, _ = m.transcribe(y, language="ar", beam_size=1, condition_on_previous_text=False,
                                   without_timestamps=True)
            e, n = cer(r.Aya, " ".join(s.text for s in segs))
            if not n or e / n > args.max_cer:
                continue
            f = out_dir / f"{r.reciter_id}_{i}_{seen}.npy"
            np.save(f, (np.clip(y, -1, 1) * 32767).astype(np.int16))
            rows.append({"clip": str(f.relative_to(ROOT)), "text": label(r.Aya), "reciter": f"crowd:{r.reciter_id}",
                         "gender": r.reciter_gender})
            kept += 1
    (ROOT / "data" / "cache" / "finetune" / "crowd.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    print(f"kept {kept} of {seen} unannotated clips from non-test reciters "
          f"({sum(x['gender'] == 'female' for x in rows)} by women, {len({x['reciter'] for x in rows})} reciters)")


if __name__ == "__main__":
    main()
