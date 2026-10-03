#!/usr/bin/env python
"""More ordinary voices reading Arabic, as training clips (same shape as build_cv_set.py).

Hugging Face sets found in the 2026-10-01 data hunt (downloaded to D:\\dua-data\\hf):

  quranspeech   Yahya-Mohamed/quran_speech_dataset: people reciting the Quran ayah by ayah
  recerrors     sobolev210/quran-recitation-errors (MIT): non-professional reciters, with
                the reviewers' error marks (clips with a marked error are left out)
  nahw          NahwAI/arabic-tashkeel-speech (CC-BY 4.0): crowd-sourced fully vowelled MSA
  sawtarabi     ArabicSpeech/sawtarabi: only its MSA rows (the rest is Egyptian/English)
  fleurs        google/fleurs ar_eg (CC-BY 4.0): Egyptian speakers reading MSA sentences

    python scripts/build_voice_sets.py quranspeech nahw sawtarabi recerrors

Clips: D:\\dua-data\\voices\\<set>\\clips\\*.npy (16 kHz int16; the C: drive is full);
rows: data/cache/finetune/<set>.jsonl with absolute clip paths (AudioBank reads either).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_cv_set import clean  # noqa: E402

HF = Path(r"D:\dua-data\hf")
VOICES = Path(r"D:\dua-data\voices")
SR = 16000


def _parquet_rows(folder: Path):
    import pyarrow.parquet as pq

    # Batches, not whole files: a 300 MB parquet as Python objects is several GB of
    # commit, and the machine's commit limit is shared with GPU jobs.
    for f in sorted(folder.rglob("*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
            yield from batch.to_pylist()


def _save(name: str, key: str, audio_bytes: bytes) -> tuple[Path, float] | None:
    from faster_whisper.audio import decode_audio

    try:
        y = decode_audio(io.BytesIO(audio_bytes), sampling_rate=SR)
    except Exception:  # noqa: BLE001
        return None
    dur = len(y) / SR
    if not 0.8 <= dur <= 20.0:
        return None
    clips = VOICES / name / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    f = clips / f"{key}.npy"
    np.save(f, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    return f, dur


def build(name: str) -> list[dict]:
    rows = []
    if name == "quranspeech":
        for i, r in enumerate(_parquet_rows(HF / "Yahya-Mohamed__quran_speech_dataset")):
            text = clean(r.get("transcription") or "")
            if len(text) < 3:
                continue
            got = _save(name, f"qs{i:05d}_{r['sura']}_{r['aya']}_{r['person']}", r["audio"]["bytes"])
            if got:
                rows.append({"clip": str(got[0]), "text": text, "reciter": f"quranspeech:{r['person']}",
                             "dur": round(got[1], 2)})
    elif name == "nahw":
        for i, r in enumerate(_parquet_rows(HF / "NahwAI__arabic-tashkeel-speech")):
            text = clean(r.get("transcription") or r.get("sentence") or "")
            if len(text) < 3:
                continue
            got = _save(name, f"nahw{i:05d}", r["audio"]["bytes"])
            if got:
                rows.append({"clip": str(got[0]), "text": text, "reciter": f"nahw:{r['speaker_id']}",
                             "dur": round(got[1], 2)})
    elif name == "sawtarabi":
        for i, r in enumerate(_parquet_rows(HF / "ArabicSpeech__sawtarabi")):
            if r.get("dialect") != "MSA":
                continue
            text = clean(r.get("text_diacritized") or r.get("text_not_diacritized") or "")
            if len(text) < 3:
                continue
            got = _save(name, f"st_{r['id']}", r["audio"]["bytes"])
            if got:
                rows.append({"clip": str(got[0]), "text": text, "reciter": "sawtarabi:msa", "dur": round(got[1], 2)})
    elif name == "fleurs":
        for i, r in enumerate(_parquet_rows(HF / "fleurs_ar")):
            text = clean(r.get("raw_transcription") or r.get("transcription") or "")
            if len(text) < 3:
                continue
            got = _save(name, f"fl{i:05d}_{r.get('id')}", r["audio"]["bytes"])
            if got:
                rows.append({"clip": str(got[0]), "text": text, "reciter": f"fleurs:{r.get('gender')}",
                             "dur": round(got[1], 2)})
    elif name == "recerrors":
        root = Path(r"D:\hf\sob")
        meta = [json.loads(x) for x in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        for i, m in enumerate(meta):
            if any(v for v in (m.get("errors") or {}).values()):
                continue  # a reviewer marked a recitation error: the text isn't what was said
            p = root / m.get("file_name", m.get("audio", ""))
            if not p.exists():
                continue
            text = clean(m.get("text") or "")
            got = _save(name, f"re{i:05d}", p.read_bytes()) if len(text) >= 3 else None
            if got:
                rows.append({"clip": str(got[0]), "text": text, "reciter": f"recerrors:{m.get('recording_id', i)}",
                             "dur": round(got[1], 2)})
    out = ROOT / "data" / "cache" / "finetune" / f"{name}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    h = sum(r["dur"] for r in rows) / 3600
    print(f"{name}: {len(rows)} clips, {h:.1f} h, {len({r['reciter'] for r in rows})} voices -> {out.name}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="+", choices=["quranspeech", "nahw", "sawtarabi", "recerrors", "fleurs"])
    args = ap.parse_args()
    for s in args.sets:
        build(s)


if __name__ == "__main__":
    main()
