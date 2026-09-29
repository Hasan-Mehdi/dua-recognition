#!/usr/bin/env python
"""Line-entry latency of page mode and the word follower, with every live delay counted (plan Step 3).

A thin adapter over follow_eval.py (same eval sets, same lanes, same line_entries rule).
Each online decision uses only audio up to its window end; the result reaches the screen
after the compute delay (--follow-delays: the measured live costs), and the tracker's
anchor after the tracker delay. No right-context/oracle variants here: those are
follow_eval.py --diag diagnostics, reported separately.

Entry lag is measured three ways, per line start:
    human     DuaPlayer's line mark (annotators mark the breath/onset)
    auto      the line's first forced-aligned word (automatic label, same model as the follower)
    reviewed  a human-reviewed first-word onset (scripts/review_bundle.py), where one exists

    python scripts/follow_latency.py --split train
    python scripts/follow_latency.py --split train --ctc fastconformer-tilawa --follow-delays 0.1
"""
from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import follow_eval as fe  # noqa: E402
import word_eval as we  # noqa: E402
from dua_recognition import provenance  # noqa: E402
from dua_recognition.follower import FollowerConfig  # noqa: E402
from dua_recognition.tracker import TrackerConfig  # noqa: E402

RUN = ROOT / "data" / "cache" / "reliability" / os.environ.get("DUA_REL_RUN", "rel-20260927")


def entries(ix, it, shown, ref_starts):
    """Lag of the display entering each line vs `ref_starts` [(t, seg)] (None = never, in +-4 s)."""
    ticks = it["ticks"]
    seg = [None if w is None else int(ix.word_segment[w]) for w in shown]
    out = []
    for t, s in ref_starts:
        a, b = np.searchsorted(ticks, [t - 4.0, t + 4.0])
        lag = None
        for i in range(max(1, a), min(b, len(ticks))):
            if seg[i] == s and seg[i - 1] != s:
                lag = float(ticks[i] - t)
                break
        out.append(lag)
    return out


def refs(ix, it):
    """(human line marks, first aligned word per line, reviewed first-word onsets) after the first line."""
    human = it["lines"][1:]
    first = {}
    for w, a, *_ in it["good"]:
        first.setdefault(int(ix.word_segment[w]), a)
    auto = [(first[s], s) for _, s in human if s in first]
    review = provenance.load_review(it["id"])
    rev = []
    if review:
        lo = ix.dua_word_span[ix.dua_ids.index(it["dua"])][0]
        for w in provenance.human_words(review, lo):
            if fe._line_first(ix, w[0]):
                rev.append((w[1], int(ix.word_segment[w[0]])))
    return {"human": human, "auto": auto, "reviewed": rev}


def stats(lags: list) -> dict:
    found = [x for x in lags if x is not None]
    if not found:
        return {"n": len(lags), "found": 0}
    pos = [x for x in found if x > 0]
    return {"n": len(lags), "missed": 1 - len(found) / len(lags), "median": float(np.median(found)),
            "p90_positive": float(np.percentile(pos, 90)) if pos else 0.0,
            "early_gt_0.3": float(np.mean([x < -0.3 for x in found]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--asr", default="whisper-base-quran-dua")
    ap.add_argument("--ctc", default="wav2vec2-quran-dua")
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--follow-delays", default="0.1,0.2,0.29",
                    help="compute delays to count (s): offline assumption, the 0.2 s step, live CPU p50")
    ap.add_argument("--pauses", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.split == "test":
        sys.exit("test is not looked at in this plan (round 2 kept its one look)")
    cfg = TrackerConfig()
    ix = fe.ev.CorpusIndex(fe.ev.load_all())
    items = fe.load_set(ix, args.asr, args.split, args.ctc, args.ctc, args.pauses, cfg, args.delay)
    lanes = {"page mode": None} | {f"follower, {float(d):g} s compute": float(d) for d in args.follow_delays.split(",")}
    res = {"set": f"{args.split} {'pauses' if args.pauses else 'flowing'}", "ctc": args.ctc, "recordings": len(items),
           "lanes": {}}
    for name, fd in lanes.items():
        lags = {"human": [], "auto": [], "reviewed": []}
        shown = ticks = 0
        for it in items:
            if fd is None:
                r = we.replay_ticks(ix, it["dua"], it["ups"], it["good"], it["ticks"], fe.SMOOTH)
            else:
                ups = fe.follow_updates(ix, it, FollowerConfig(), args.delay, fd)
                r = we.replay_ticks(ix, it["dua"], ups, it["good"], it["ticks"], None)
            shown += sum(w is not None for w in r["shown"])
            ticks += len(r["shown"])
            for k, ref in refs(ix, it).items():
                lags[k] += entries(ix, it, r["shown"], ref)
        res["lanes"][name] = {"shown": shown / max(1, ticks), **{k: stats(v) for k, v in lags.items()}}
        h, a = res["lanes"][name]["human"], res["lanes"][name]["auto"]
        print(f"{name:28s} shown {shown / max(1, ticks):.1%} | human: median {h.get('median', float('nan')):+.2f} "
              f"p90+ {h.get('p90_positive', float('nan')):.2f} early {h.get('early_gt_0.3', float('nan')):.1%} "
              f"missed {h.get('missed', float('nan')):.1%} | auto first word: median {a.get('median', float('nan')):+.2f} "
              f"missed {a.get('missed', float('nan')):.1%} | reviewed n={res['lanes'][name]['reviewed']['n']}",
              flush=True)
    out = args.out or RUN / "follower" / f"latency_{args.split}_{args.ctc}_{'pauses' if args.pauses else 'flowing'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
