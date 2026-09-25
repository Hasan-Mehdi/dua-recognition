#!/usr/bin/env python
"""Label recordings that have no line timings, with the offline smoother.

    python scripts/align_offline.py --validate   # agreement with DuaPlayer's human timings
    python scripts/align_offline.py              # label data/youtube/* (needs window caches)

Labelling writes data/youtube/<dua>/<video>.labels.json in the same shape as a
DuaPlayer timing file (segment -> start seconds), plus how much of the
recording the teacher could place at all. Recordings that are mostly unplaced
(wrong du'a, a lecture, heavy music) are marked unusable.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from evaluate import WINDOWS, load_rows  # noqa: E402

from dua_recognition.align import CorpusIndex, encode  # noqa: E402
from dua_recognition.asr import _looks_hallucinated  # noqa: E402
from dua_recognition.corpus import Recording, load_all, load_recordings  # noqa: E402
from dua_recognition.offline import align_recording  # noqa: E402
from dua_recognition.splits import LINE_LABELS_UNRELIABLE  # noqa: E402

YT = ROOT / "data" / "youtube"


def starts_from(times: list[float], segs: np.ndarray, placed: list[bool]) -> list[tuple[float, int]]:
    """Collapse per-step segment ids into (start time, segment) runs, monotone."""
    out, last = [], 0
    for t, s, ok in zip(times, segs, placed):
        if not ok or s <= last:
            continue
        out.append((round(t - 1.0, 2), int(s)))  # a window ending at t first heard it ~1 s ago
        last = s
    return out


def placed_mask(ix: CorpusIndex, rows, costs, max_cost=0.4) -> list[bool]:
    out = []
    for (_, text), c in zip(rows, costs):
        n = encode(text).size if text else 0
        out.append(c is not None and n >= 4 and float(c.min()) / n <= max_cost)
    return out


def validate(asr: str, stride: int) -> None:
    duas = load_all()
    exact = near = n = 0
    per = []
    for dua in duas.values():
        if dua.id in LINE_LABELS_UNRELIABLE:
            continue
        ix = CorpusIndex({dua.id: dua})
        for rec in load_recordings(dua):
            rows = load_rows(asr, rec, 6.0, 1.0)[stride - 1 :: stride]
            if not rows:
                continue
            costs = [ix.word_costs(t) if t else None for _, t in rows]
            segs = align_recording(ix, costs, float(stride))
            e = m = k = 0
            for (t, _), s in zip(rows, segs):
                g = rec.segment_at(t - 1.0)  # the smoother places what was heard ~1 s back
                if g is None:
                    continue
                k += 1
                e += s == g
                m += abs(int(s) - g) <= 1
            exact, near, n = exact + e, near + m, n + k
            per.append((dua.id, rec.reciter, e / max(1, k)))
    print(f"offline aligner vs human timings: exact line {exact / n:.1%}, within ±1 line {near / n:.1%} "
          f"({n} steps at {stride} s)")
    worst = sorted(per, key=lambda x: x[2])[:5]
    print("lowest:", ", ".join(f"{d}/{r} {a:.0%}" for d, r, a in worst))


def label(asr: str, hop: float) -> None:
    duas = load_all()
    for meta_path in sorted(YT.glob("*/*.json")):
        if meta_path.name.endswith(".labels.json"):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dua = duas[meta["dua_id"]]
        cache = WINDOWS / asr / f"yt-{meta['video_id']}_w6_h{hop:g}.jsonl"
        if not cache.exists():
            continue
        rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line]
        rows = [(r["t"], "" if _looks_hallucinated(r["text"]) else r["text"]) for r in rows]
        ix = CorpusIndex({dua.id: dua})
        costs = [ix.word_costs(t) if t else None for _, t in rows]
        segs = align_recording(ix, costs, hop)
        placed = placed_mask(ix, rows, costs)
        starts = starts_from([t for t, _ in rows], segs, placed)
        voiced = [p for (_, text), p in zip(rows, placed) if text]
        coverage = sum(voiced) / max(1, len(voiced))
        lines = len({s for _, s in starts}) / len(dua.segments)
        usable = coverage >= 0.5 and lines >= 0.5
        out = meta_path.with_name(meta_path.stem + ".labels.json")
        out.write_text(json.dumps({
            "video_id": meta["video_id"], "dua_id": dua.id, "reciter": f"yt:{meta.get('channel')}",
            "slide_start_s": {str(s): t for t, s in starts},
            "end_s": rows[-1][0] if rows else 0,
            "placed_fraction": round(coverage, 3), "lines_found": round(lines, 3), "usable": usable,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{dua.id:28s} {meta['video_id']}  placed {coverage:5.1%}  lines {lines:5.1%}  "
              f"{'ok' if usable else 'UNUSABLE'}  {meta.get('title', '')[:50]}")


def load_youtube(dua) -> list[Recording]:
    """YouTube recordings of a du'a with usable silver labels, as Recordings."""
    out = []
    for lab_path in sorted((YT / dua.id).glob("*.labels.json")) if (YT / dua.id).exists() else []:
        lab = json.loads(lab_path.read_text(encoding="utf-8"))
        if not lab["usable"]:
            continue
        meta = json.loads(lab_path.with_name(lab["video_id"] + ".json").read_text(encoding="utf-8"))
        starts = sorted((t, int(s)) for s, t in lab["slide_start_s"].items())
        out.append(Recording(f"yt-{lab['video_id']}", dua.id, lab["reciter"], lab_path.with_name(meta["audio"]),
                             meta.get("duration_s") or lab["end_s"], starts, lab["end_s"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr", default="large-v3-turbo")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--stride", type=int, default=2, help="validation: use every Nth 1 s window")
    ap.add_argument("--hop", type=float, default=2.0, help="labelling: window hop of the YouTube caches")
    args = ap.parse_args()
    if args.validate:
        validate(args.asr, args.stride)
    else:
        label(args.asr, args.hop)


if __name__ == "__main__":
    main()
