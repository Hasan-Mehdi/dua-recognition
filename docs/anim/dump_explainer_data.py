#!/usr/bin/env python
"""Dump the real tracker internals the explainer animation draws.

One step of Dua Tawassul by a held-out reciter (Hussein Ghareeb), at a moment the
reciter is saying a refrain that recurs 14 times in the du'a: the belief before
the step, the prediction, the evidence (likelihood of the window's transcript
ending at each word) and the posterior. Plus a cold-start identification run.

    python scripts/transcribe_windows.py --model models/whisper-base-quran-dua-ct2 --tag whisper-base-quran-dua --test-only
    python docs/anim/dump_explainer_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402

STEP = 295  # window ending at t = 296 s: "يا وجيه عند الله"


def main() -> None:
    duas = ev.load_all()
    ix = ev.CorpusIndex(duas)
    rec = next(r for r in ev.load_recordings(duas["dua-tawassul"]) if r.reciter == "Hussein Ghareeb")
    rows = ev.load_rows("whisper-base-quran-dua", rec, 6.0, 1.0)
    lo, hi = ix.dua_word_span[ix.dua_ids.index("dua-tawassul")]
    costs = [ix.word_costs(t) if t else None for _, t in rows[: STEP + 8]]

    tr = ev.Tracker(ix)
    seq = []
    for k in range(STEP):
        tr.update_costs(costs[k], 1.0)
        if k >= STEP - 15:
            seq.append(tr.post[lo:hi].copy())
    before = tr.post.copy()
    tr._advance(1.0, locked=tr._locked())
    pred = tr.post.copy()
    kappa = tr.cfg.kappa if tr._locked() else tr.cfg.kappa_search
    c = costs[STEP].astype(float)
    lik = np.exp(-kappa * (c - c.min()))
    post = pred * lik
    post /= post.sum()
    seg = ix.word_segment[lo:hi]

    cold = ev.Tracker(ix)
    ident = []
    for k in range(STEP, STEP + 8):
        cold.update_costs(costs[k], 1.0)
        m = np.bincount(ix.word_dua, weights=cold.post, minlength=len(ix.duas))
        ident.append([[duas[ix.dua_ids[i]].name_en, float(m[i])] for i in np.argsort(-m)[:4]])

    # Per-line views (bar charts): belief mass per line, best evidence per line.
    segs = np.unique(seg)
    per_line = {name: np.array([v[lo:hi][seg == s].sum() for s in segs]) for name, v in
                (("pred_line", pred), ("post_line", post), ("before_line", before))}
    per_line["lik_line"] = np.array([lik[lo:hi][seg == s].max() for s in segs])
    # The audio itself isn't redistributed: only a coarse loudness envelope (50 ms bins).
    from faster_whisper.audio import decode_audio
    t_end = rows[STEP][0]
    y = decode_audio(str(rec.path), sampling_rate=16000)[int((t_end - 16) * 16000) : int(t_end * 16000)]
    env = np.abs(y[: len(y) // 800 * 800]).reshape(-1, 800).max(axis=1)

    np.savez(HERE / "explainer_data.npz", before=before[lo:hi], pred=pred[lo:hi], lik=lik[lo:hi],
             post=post[lo:hi], seg=seg, seq=np.array(seq), segs=segs, env=env / env.max(), **per_line)
    meta = {
        "text": rows[STEP][1],
        "truth_seg": int(rec.segment_at(rows[STEP][0])),
        "argmax_seg": int(seg[post[lo:hi].argmax()]),
        "kappa": kappa,
        "refrain_segs": sorted(int(s) for s in ev.refrain_ids(duas["dua-tawassul"])),
        "ident": ident,
        "lines": {int(x.id): x.arabic for x in duas["dua-tawassul"].segments if 31 <= x.id <= 38},
        "t_end": t_end,
    }
    (HERE / "explainer_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"line {meta['argmax_seg']} (human label {meta['truth_seg']})")


if __name__ == "__main__":
    main()
