#!/usr/bin/env python
"""How well does each ASR model hear ordinary voices?

The DuaPlayer test reciters are professionals, like the training data. This
scores models on RetaSy's crowd-sourced Quran recitations (1,289 volunteers,
half women, mostly non-Arabic speakers; huggingface.co/datasets/RetaSy/quranic_audio_dataset),
using the clips annotators marked as correct (Quran verses, plus some du'as and
adhkar) against the text each was reciting. CER on normalized letters.

    python scripts/voice_eval.py models/whisper-base-quran-dua-ct2 large-v3-turbo
    python scripts/voice_eval.py --set quranlab ...   # Quran-Lab benchmark (below)

--set quranlab: huggingface.co/datasets/Quran-Lab/quranic-asr-benchmark, 600
clips in three groups: real phone recordings from Tarteel users (tlog_holdout),
a held-out professional set (everyayah_heldout) and qul_alnufais. Gated (accept
the terms on the HF account; evaluation use only), cached in data/cache/quranlab.
Scores are reported per group; the phone group is the ordinary-voice number.
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


QURANLAB = ROOT / "data" / "cache" / "quranlab"


def load_quranlab(limit: int | None = None):
    """Quran-Lab clips as {"y", "ref", "reciter", "gender"}; "gender" holds the group name."""
    import json

    import soundfile as sf
    from huggingface_hub import snapshot_download

    snapshot_download("Quran-Lab/quranic-asr-benchmark", repo_type="dataset", local_dir=QURANLAB)
    clips = []
    for meta in sorted(QURANLAB.glob("audio/*/metadata.jsonl")):
        for line in meta.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            ref = next((r[k] for k in ("text", "transcription", "sentence", "aya", "verse_text", "uthmani") if r.get(k)), None)
            f = meta.parent / r.get("file_name", "")
            if not ref or not f.is_file():
                continue
            y, sr = sf.read(f, dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != 16000:
                from scipy.signal import resample_poly
                y = resample_poly(y, 16000, sr).astype(np.float32)
            clips.append({"y": y, "ref": ref, "reciter": r.get("reciter_id") or f.stem, "gender": meta.parent.name})
            if limit and len(clips) >= limit:
                return clips
    if not clips:
        raise SystemExit(f"no clips with text under {QURANLAB}/audio (metadata fields changed?)")
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
    ap.add_argument("--set", choices=["retasy", "quranlab"], default="retasy")
    ap.add_argument("--max-seconds", type=float,
                    help="only clips up to this long, each transcribed as one live window the way the app sends "
                         "them (asr.transcribe_batch): needed for short-context models (scripts/shorten_context.py), "
                         "whose input faster-whisper's transcribe() still pads to 30 s")
    args = ap.parse_args()
    import os

    os.environ.setdefault("DUA_ASR_DEVICE", args.device)
    from faster_whisper import WhisperModel

    if args.set == "quranlab":
        clips = load_quranlab(limit=args.limit)
        groups = sorted({c["gender"] for c in clips})
        print(f"{len(clips)} clips ({', '.join(f'{g} {sum(c['gender'] == g for c in clips)}' for g in groups)}), "
              f"{sum(len(c['y']) for c in clips) / 16000 / 60:.0f} min")
    else:
        clips = load_clips(limit=args.limit)
        groups = ["male", "female"]
        print(f"{len(clips)} clips, {len({c['reciter'] for c in clips})} reciters, "
              f"{sum(c['gender'] == 'female' for c in clips)} by women, {sum(len(c['y']) for c in clips) / 16000 / 60:.0f} min")
    if args.max_seconds:
        clips = [c for c in clips if len(c["y"]) <= args.max_seconds * 16000]
        print(f"  {len(clips)} clips of at most {args.max_seconds:g} s, each as one live window")
    for name in args.models:
        if args.max_seconds:
            from dua_recognition.asr import transcribe_batch

            hyps = [h for i in range(0, len(clips), 16) for h in transcribe_batch([c["y"] for c in clips[i : i + 16]], model=name)]
        else:
            m = WhisperModel(name, device=args.device, compute_type="int8" if args.device == "cpu" else "float16")
        tot = {"all": [0, 0], **{g: [0, 0] for g in groups}}
        for j, c in enumerate(clips):
            if args.max_seconds:
                text = hyps[j]
            else:
                segs, _ = m.transcribe(c["y"], language="ar", beam_size=1, condition_on_previous_text=False,
                                       without_timestamps=True)
                text = " ".join(s.text for s in segs)
            e, n = cer(c["ref"], text)
            for k in ("all", c["gender"]):
                if k in tot:
                    tot[k][0] += e
                    tot[k][1] += n
        f = {k: v[0] / max(v[1], 1) for k, v in tot.items()}
        label = {"male": "men", "female": "women"}
        print(f"{name:45s} CER {f['all']:.1%}   " + "   ".join(f"{label.get(g, g)} {f[g]:.1%}" for g in groups),
              flush=True)


if __name__ == "__main__":
    main()
