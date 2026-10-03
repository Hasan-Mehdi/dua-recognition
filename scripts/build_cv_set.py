#!/usr/bin/env python
"""Ordinary voices reading Arabic: Mozilla Common Voice 17 (CC0) as training clips.

The du'a data is professional reciters plus synthetic and a few crowd-sourced ordinary
voices. Common Voice is thousands of ordinary people reading sentences into whatever mic
they have, phones included: no recitation, but the voices and channels the phone meets.
Clips of 1-10 s whose sentence has Arabic letters only (diacritics and punctuation
removed) and more up- than down-votes.

    python scripts/build_cv_set.py            # train -> data/cache/finetune/cv_ar.jsonl
    python scripts/build_cv_set.py --split validation --name cv_ar_val --files 1   # an eval sample

Common Voice's Arabic train split is many clips from few speakers (23); its validation and
test splits are other speakers, so they train too, except one validation file kept for eval:

    python scripts/build_cv_set.py --split test --name cv_ar_test
    python scripts/build_cv_set.py --split validation --name cv_ar_val2 --skip 1

Clips: data/cache/cv_ar/clips/*.npy (16 kHz int16). Source: the ungated parquet mirror
huggingface.co/datasets/fixie-ai/common_voice_17_0 (ar).
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

OUT = ROOT / "data" / "cache" / "cv_ar"
REPO = "fixie-ai/common_voice_17_0"
_MARKS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # harakat, Quranic marks, tatweel
_NON_ARABIC = re.compile("[^ء-ي ]")


def clean(text: str) -> str:
    t = _MARKS.sub("", text).replace("ٱ", "ا")
    if re.search("[A-Za-z0-9]", t):
        return ""
    return " ".join(_NON_ARABIC.sub(" ", t).split())


def main() -> None:
    import pyarrow.parquet as pq
    from faster_whisper.audio import decode_audio
    from huggingface_hub import HfApi, hf_hub_download

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--name", default="cv_ar")
    ap.add_argument("--files", type=int, default=0, help="only the first N parquet files (an eval sample)")
    ap.add_argument("--skip", type=int, default=0, help="leave out the first N parquet files (kept for eval)")
    args = ap.parse_args()
    clips = OUT / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.startswith(f"ar/{args.split}-") and f.endswith(".parquet"))
    files = files[args.skip :][: args.files] if args.files else files[args.skip :]
    rows, seen = [], 0
    for fn in files:
        t = pq.read_table(hf_hub_download(REPO, fn, repo_type="dataset")).to_pandas()
        for _, r in t.iterrows():
            seen += 1
            text = clean(str(r.get("sentence") or ""))
            if len(text.replace(" ", "")) < 4 or int(r.get("up_votes") or 0) <= int(r.get("down_votes") or 0):
                continue
            y = decode_audio(io.BytesIO(r["audio"]["bytes"]), sampling_rate=16000)
            if not 1.0 <= len(y) / 16000 <= 10.0:
                continue
            name = Path(r["audio"]["path"] or f"{seen}.mp3").stem
            f = clips / f"{name}.npy"
            np.save(f, (np.clip(y, -1, 1) * 32767).astype(np.int16))
            rows.append({"clip": str(f.relative_to(ROOT)), "text": text, "reciter": f"cv:{r.get('client_id')}",
                         "gender": str(r.get("gender") or "")})
        print(f"{fn}: {len(rows)} kept of {seen}", flush=True)
    out = ROOT / "data" / "cache" / "finetune" / f"{args.name}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} clips, {len({r['reciter'] for r in rows})} speakers -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
