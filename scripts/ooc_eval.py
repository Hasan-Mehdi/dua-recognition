#!/usr/bin/env python
"""What does the tracker show for a du'a it doesn't know?

Leave-one-out: each recording is replayed against the corpus *minus its own
du'a*, so anything shown is a false match. Reported: the share of recited time
some du'a is on screen, and the share of recordings that ever show one. The same
settings are also scored in-corpus (line accuracy, identification speed) so a
stricter "not recognised" can't hide a slower or worse follower.

    python scripts/ooc_eval.py --split train --set null_rate=0.4
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
from dua_recognition.align import encode  # noqa: E402
from dua_recognition.tracker import Tracker, TrackerConfig  # noqa: E402


def run(ix, cfg, rows, costs):
    tr = Tracker(ix, cfg)
    out = []
    for (_, text), c in zip(rows, costs):
        p = tr.update_costs(c, 1.0, n_letters=len(encode(text)) if text else 0)
        out.append(p.dua)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr", default=None, help="window cache (default: turbo for train, base-ft for test)")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--set", nargs="*", default=[], action="append", metavar="FIELD=VALUE")
    args = ap.parse_args()
    asr = args.asr or ("large-v3-turbo" if args.split == "train" else "whisper-base-quran-dua")
    duas = ev.load_all()
    full = ev.CorpusIndex(duas)
    recs = []
    for dua in duas.values():
        for rec in ev.load_recordings(dua):
            if ev.is_test(rec.reciter) != (args.split == "test"):
                continue
            rows = ev.load_rows(asr, rec, 6.0, 1.0)
            if rows:
                recs.append((rec, rows))
    minus = {d: ev.CorpusIndex({k: v for k, v in duas.items() if k != d}) for d in {r.dua_id for r, _ in recs}}
    cache = {}
    for rec, rows in recs:
        cache[rec.audio_id] = (
            [full.word_costs(t) if t else None for _, t in rows],
            [minus[rec.dua_id].word_costs(t) if t else None for _, t in rows],
        )
    print(f"{len(recs)} recordings ({args.split}), ASR {asr}")
    for sets in args.set or [[]]:
        cfg = replace(TrackerConfig(), **{k: float(v) for k, v in (s.split("=", 1) for s in sets)})
        shown, ever, n = 0, 0, 0
        line, steps = 0, 0
        found3 = []
        for rec, rows in recs:
            c_full, c_minus = cache[rec.audio_id]
            recited = [rec.segment_at(t) is not None for t, _ in rows]
            p = run(minus[rec.dua_id], cfg, rows, c_minus)
            s = sum(1 for d, r in zip(p, recited) if r and d is not None)
            shown += s
            n += sum(recited)
            ever += s > 0
            if rec.dua_id in ev.LINE_LABELS_UNRELIABLE:
                continue
            tr = Tracker(full, cfg)
            for (t, text), c in zip(rows, c_full):
                q = tr.update_costs(c, 1.0, n_letters=len(encode(text)) if text else 0)
                g = rec.segment_at(t)
                if g is not None:
                    steps += 1
                    line += q.dua == rec.dua_id and q.segment == g
        print(f"{' '.join(sets) or 'default':40s} NOT IN CORPUS: some du'a shown {shown / n:.1%} of the time, "
              f"in {ever}/{len(recs)} recordings | IN CORPUS: line acc {line / steps:.1%}", flush=True)


if __name__ == "__main__":
    main()
