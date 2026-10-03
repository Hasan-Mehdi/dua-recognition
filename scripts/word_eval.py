#!/usr/bin/env python
"""Score the word highlight against forced-aligned word timings (scripts/word_truth.py).

Every 0.1 s of each test recording: which word is on screen, and which word is
the reciter saying? Each tracker result reaches the screen `--delay` seconds
after its window ends (ASR time). Also counts visible jerks: the highlight
stepping back, or skipping more than two words at once.

    python scripts/word_eval.py --delay 0.5
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
from dua_recognition.display import Highlight  # noqa: E402
from dua_recognition.tracker import Tracker, TrackerConfig  # noqa: E402

ENDS = ROOT / "data" / "cache" / "word_ends"


def load(tag: str, split: str = "test"):
    duas = ev.load_all()
    ix = ev.CorpusIndex(duas)
    data = []
    for dua in duas.values():
        if dua.id in ev.LINE_LABELS_UNRELIABLE:
            continue
        for rec in ev.load_recordings(dua):
            words = ev.load_word_truth(rec, ix)
            if ev.is_test(rec.reciter) != (split == "test") or not words:
                continue
            rows = ev.load_rows(tag, rec, 6.0, 1.0)
            e = ENDS / tag / f"{rec.audio_id}.json"
            stale = json.loads(e.read_text()) if e.exists() else [None] * len(rows)
            truth = sorted(words, key=lambda w: w[1])
            qf = ROOT / "data" / "cache" / "quiet" / f"{rec.audio_id}.json"
            quiet = json.loads(qf.read_text()) if qf.exists() else [0.0] * len(rows)
            data.append((rec, rows, ev.LazyCosts(ix, [t for _, t in rows]), stale, truth, quiet))
    return ix, data


def hmm_updates(ix, rows, costs, quiet, cfg, delay, lead_mode="fixed", stale=None, still=False, anchors=None,
                letters=None):
    """Replay one recording's windows through the tracker: one entry per window,
    (display time, dua, seg, word, line word range, speed). With `anchors` (a
    list), also append each window's evidence position (no lead), for the
    word follower to anchor on. `letters`: transcript lengths, as pause_eval passes them."""
    tr = Tracker(ix, cfg)
    ups = []
    stale = stale or [None] * len(rows)
    letters = letters or [None] * len(rows)
    for (t, _), c, st, q, n in zip(rows, costs, stale, quiet, letters):
        q = q if still else 0.0
        if lead_mode == "none":
            lead = 0.0
        elif lead_mode == "fixed":
            lead = delay + cfg.display_lead
        else:  # "stale:<extra>"
            lead = delay + (st or 0.0) + float(lead_mode.split(":")[1])
        p = tr.update_costs(c, 1.0, lead, n_letters=n, quiet=q)
        rng = None
        if p.word is not None:
            d = ix.word_dua[p.word]
            lo, hi = ix.dua_word_span[d]
            ws = np.flatnonzero(ix.word_segment[lo:hi] == p.segment) + lo
            rng = (int(ws[0]), int(ws[-1]) + 1)
        ups.append((t + delay, p.dua, p.segment, p.word, rng, 0.0 if q > tr.cfg.still_after else tr.speed))
        if anchors is not None:
            now = tr.position()
            w = now.word if now.dua is not None else None
            # ...and the belief in each line of that du'a, for the follower's jump rule (jump_mass).
            anchors.append((t, w, tr.line_masses(int(ix.word_dua[w])) if w is not None else None))
    return ups


def replay_ticks(ix, dua_id, ups, good, ticks, smooth):
    """What's on screen every tick, and how it compares with the word being said.

    `ups`: (display time, dua, seg, word, line word range, speed), in time order;
    with `smooth` (display.Highlight settings) they drive the gliding highlight,
    without it the word is shown as it arrives. `good`: [word, start, end, score]
    word truth sorted by start. Returns per-tick shown words (None = nothing of
    this du'a) and the offsets, jerks and skipped words."""
    from collections import Counter
    starts = [w[1] for w in good]
    hl = Highlight(**smooth) if smooth is not None else None
    k, prev, shown = 0, None, None
    out = {"shown": [], "offs": [], "jerks": 0, "skipped": 0, "kinds": Counter()}
    for tick in ticks:
        while k < len(ups) and ups[k][0] <= tick:
            _, dua, seg, word, rng, speed = ups[k]
            if hl is not None:
                hl.update(ups[k][0], dua, seg, word, rng or (0, 0), speed)
            else:
                shown = word if dua == dua_id else None
            k += 1
        if hl is not None:
            shown = hl.word(tick) if hl.line and hl.line[0] == dua_id else None
        out["shown"].append(shown)
        i = bisect.bisect_right(starts, tick) - 1
        if i < 0 or shown is None:
            continue
        truth_w = good[i][0]
        out["offs"].append(max(-10, min(10, shown - truth_w)))
        if prev is not None and shown != prev:
            out["skipped"] += abs(shown - prev) - 1 if shown > prev else abs(shown - prev)
        if prev is not None and (shown < prev or shown > prev + 2):
            out["jerks"] += 1
            same = ix.word_segment[shown] == ix.word_segment[prev]
            out["kinds"][("back" if shown < prev else "skip") + (" in line" if same else " across lines")] += 1
        prev = shown
    return out


def run(ix, data, cfg, delay, lead_mode, smooth, min_score=-1.5, still=False):
    from collections import Counter
    offs, jerks, minutes, kinds, skipped = [], 0, 0.0, Counter(), 0
    for rec, rows, costs, stale, truth, quiet in data:
        good = [w for w in truth if w[3] >= min_score]
        if not good:
            continue
        ups = hmm_updates(ix, rows, costs, quiet, cfg, delay, lead_mode, stale, still)
        t_end = min(rec.end_s, good[-1][2] + 2.0)
        ticks = np.arange(good[0][1], t_end, 0.1)
        r = replay_ticks(ix, rec.dua_id, ups, good, ticks, smooth)
        offs += r["offs"]
        jerks += r["jerks"]
        skipped += r["skipped"]
        kinds += r["kinds"]
        minutes += len(ticks) / 600
    o = np.array(offs)
    return {"exact": float(np.mean(o == 0)), "pm1": float(np.mean(np.abs(o) <= 1)), "abs": float(np.abs(o).mean()),
            "mean": float(o.mean()), "jerks_min": jerks / max(minutes, 1e-9),
            "skipped_min": skipped / max(minutes, 1e-9),
            "kinds": {k: round(v / max(minutes, 1e-9), 2) for k, v in kinds.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr", default="whisper-base-quran-dua")
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--source", choices=["duaplayer", *ev.TESTSETS], default="duaplayer",
                    help="a held-out set from corpus.TESTSETS (user, majlis, amateur)")
    ap.add_argument("--tier", choices=["gold", "silver", "all"], default="all",
                    help="gold: human or reviewed line times; silver: auto labels")
    ap.add_argument("variants", nargs="*", default=["none", "fixed"],
                    help="lead[;smooth k=v,...][;cfg k=v,...], lead = none | fixed | stale:<extra s>")
    args = ap.parse_args()
    ev.use_source(args.source, args.tier)
    ix, data = load(args.asr, args.split)
    print(f"{len(data)} recordings ({args.split}), ASR {args.asr}, delay {args.delay:g} s\n"
          f"{ev.describe([d[0] for d in data])}")
    for v in args.variants:
        parts = v.split(";")
        smooth = None
        cfg = TrackerConfig()
        for part in parts[1:]:
            kind, _, kv = part.partition(" ")
            kw = {k: float(x) for k, x in (a.split("=") for a in kv.split(",") if a)}
            if kind == "smooth":
                smooth = kw
            elif kind == "cfg":
                cfg = replace(cfg, **kw)
        r = run(ix, data, cfg, args.delay, parts[0], smooth, still=any(p == "still" for p in parts[1:]))
        print(f"{v:55s} exact {r['exact']:.1%}  ±1 {r['pm1']:.1%}  |off| {r['abs']:.2f}  mean {r['mean']:+.2f}  jerks/min {r['jerks_min']:.1f}  skipped words/min {r['skipped_min']:.1f}",
              flush=True)


if __name__ == "__main__":
    main()
