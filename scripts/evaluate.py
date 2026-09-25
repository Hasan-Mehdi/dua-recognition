#!/usr/bin/env python
"""Score line-following and du'a identification against DuaPlayer's human timings.

Replays cached window transcripts (scripts/transcribe_windows.py) through each
method and compares, once per hop, the line it shows with the line the reciter
is actually on.

    python scripts/evaluate.py                      # test reciters, all methods
    python scripts/evaluate.py --split train --tune # tracker grid search (train only)
    python scripts/evaluate.py --asr base-ft        # another ASR model's cache

Methods
    matcher   v0.1 of this repo: each window classified and located on its own
              (rapidfuzz partial_ratio), holding the last answer when unsure
    tracker   the HMM follower over the whole corpus (identifies the du'a too)
    oracle    the same tracker told which du'a it is — isolates line tracking
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from dua_recognition.align import CorpusIndex  # noqa: E402
from dua_recognition.asr import _looks_hallucinated, speech_in_tail  # noqa: E402
from dua_recognition.classify import TextClassifier  # noqa: E402
from dua_recognition.corpus import Recording, load_all, load_recordings  # noqa: E402
from dua_recognition.match import PassageMatcher  # noqa: E402
from dua_recognition.splits import LINE_LABELS_UNRELIABLE, is_test  # noqa: E402
from dua_recognition.text import normalize  # noqa: E402
from dua_recognition.tracker import Tracker, TrackerConfig  # noqa: E402

WINDOWS = ROOT / "data" / "cache" / "windows"


def load_rows(tag: str, rec: Recording, window: float, hop: float, vad: bool = False) -> list[tuple[float, str]]:
    path = WINDOWS / tag / f"{rec.audio_id}_w{window:g}_h{hop:g}.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if vad and any("speech" not in r for r in rows):
        # Older (clean-audio) caches: run the live VAD over the original audio.
        from faster_whisper.audio import decode_audio

        y = decode_audio(str(rec.path), sampling_rate=16000)
        for r in rows:
            r["speech"] = speech_in_tail(y[int(max(0.0, r["t"] - window) * 16000) : int(r["t"] * 16000)])
    # Re-apply the live filters, so caches built before a filter change still
    # score exactly what the recognizer would do today.
    return [(r["t"], "" if _looks_hallucinated(r["text"]) or (vad and not r.get("speech", True)) else r["text"])
            for r in rows]


def load_ctc_costs(tag: str, rec: Recording, ix: CorpusIndex, window: float, hop: float, rows) -> list | None:
    """Per-window word costs scored straight from cached CTC posteriors (scripts/dump_ctc.py)."""
    path = ROOT / "data" / "cache" / "ctc" / tag / f"{rec.audio_id}_w{window:g}_h{hop:g}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    lp, nf = z["lp"], z["n_frames"]
    # A window the text path silences (VAD gate, empty transcript) is a pause here too.
    return [ix.ctc_word_costs(lp[k, : nf[k]]) if text else None for k, (_, text) in enumerate(rows)]


def refrain_ids(dua) -> set[int]:
    """Segments whose text occurs more than once in the du'a."""
    counts = Counter(normalize(s.arabic) for s in dua.segments)
    return {s.id for s in dua.segments if counts[normalize(s.arabic)] > 1}


# -- methods ----------------------------------------------------------------

def run_matcher(duas, clf, matchers, rows):
    preds, last = [], (None, None)
    for _, text in rows:
        if text:
            d = clf.predict_text(text)
            hit = matchers[d].locate(text)
            if hit:
                last = (d, hit.segment_id)
        preds.append(last)
    return preds


def run_tracker(ix, cfg, costs, hop, start=0, steps=None):
    tr = Tracker(ix, cfg)
    end = len(costs) if steps is None else min(len(costs), start + steps)
    out = []
    for c in costs[start:end]:
        p = tr.update_costs(c, hop)
        out.append((p.dua, p.segment))
    return out


# -- scoring ----------------------------------------------------------------

def score(rec: Recording, rows, preds, refrains) -> dict:
    n = hit = near = dua_ok = wrong = rhit = rn = 0
    jumps, prev = 0, None
    for (t, _), (d, s) in zip(rows, preds):
        g = rec.segment_at(t)
        if g is None:
            continue
        if prev is not None and d is not None and prev[0] == d and s is not None and not (0 <= s - prev[1] <= 1):
            jumps += 1  # the highlight went backwards or skipped lines
        if d is not None and s is not None:
            prev = (d, s)
        n += 1
        right_dua = d == rec.dua_id
        dua_ok += right_dua
        wrong += d is not None and not right_dua
        hit += right_dua and s == g
        near += right_dua and s is not None and abs(s - g) <= 1
        if g in refrains:
            rn += 1
            rhit += right_dua and s == g
    # Lag: how long after a line starts until the display switches to it.
    lags, missed = [], 0
    starts = rec.starts + [(rec.end_s, None)]
    times = [t for t, _ in rows]
    for (s0, k), (s1, _) in zip(starts, starts[1:]):
        if s1 - s0 < 2.0:
            continue  # too short to judge fairly at a 1 s hop
        found = None
        for t, (d, s) in zip(times, preds):
            if t < s0:
                continue
            if t > s1 + 2.0:
                break
            if d == rec.dua_id and s == k:
                found = t - s0
                break
        if found is None:
            missed += 1
        else:
            lags.append(found)
    return {
        "steps": n, "line": hit, "line_pm1": near, "dua": dua_ok, "wrong_dua": wrong,
        "refrain_steps": rn, "refrain_line": rhit,
        "lags": lags, "missed": missed, "boundaries": len(lags) + missed, "jumps": jumps,
    }


def summarize(results: list[dict]) -> dict:
    tot = {k: sum(r[k] for r in results) for k in
           ("steps", "line", "line_pm1", "dua", "wrong_dua", "refrain_steps", "refrain_line", "missed",
            "boundaries", "jumps")}
    lags = [x for r in results for x in r["lags"]]
    return {
        "line_acc": tot["line"] / tot["steps"],
        "line_acc_pm1": tot["line_pm1"] / tot["steps"],
        "dua_acc": tot["dua"] / tot["steps"],
        "wrong_dua_shown": tot["wrong_dua"] / tot["steps"],
        "refrain_acc": tot["refrain_line"] / max(1, tot["refrain_steps"]),
        "lag_median_s": statistics.median(lags) if lags else float("nan"),
        "lines_missed": tot["missed"] / max(1, tot["boundaries"]),
        "hours": tot["steps"] / 3600,
        "jumps_per_min": tot["jumps"] / max(1, tot["steps"]) * 60,
    }


def fmt(name: str, s: dict) -> str:
    return (f"| {name} | {s['line_acc']:.1%} | {s['line_acc_pm1']:.1%} | {s['refrain_acc']:.1%} | "
            f"{s['dua_acc']:.1%} | {s['wrong_dua_shown']:.1%} | {s['lag_median_s']:.1f} s | {s['lines_missed']:.1%} | "
            f"{s['jumps_per_min']:.2f} |")


HEADER = ("| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag "
          "| lines missed | jumps/min |\n|---|---|---|---|---|---|---|---|---|")


# -- identification from a cold start ----------------------------------------

def identification(ix, cfg, clf, recs_rows_costs, hop, n_starts=20, horizon=(3, 5, 10, 20, 30), seed=0):
    """Start listening at a random moment mid-recitation; is the du'a right after h seconds?"""
    rng = random.Random(seed)
    steps = {h: max(1, math.ceil(h / hop)) for h in horizon}
    longest = max(steps.values())
    trk = {h: [] for h in horizon}
    base = {h: [] for h in horizon}
    for rec, rows, costs in recs_rows_costs:
        span = [i for i, (t, _) in enumerate(rows) if rec.segment_at(t) is not None]
        if len(span) <= longest:
            continue
        for _ in range(n_starts):
            i0 = rng.choice(span[:-longest])
            preds = run_tracker(ix, cfg, costs, hop, start=i0, steps=longest)
            for h in horizon:
                n = steps[h]
                trk[h].append(preds[n - 1][0] == rec.dua_id)
                # Baseline: classify everything heard so far as one transcript
                # (non-overlapping windows, so nothing is counted twice).
                every = max(1, round(6 / hop))
                heard = " ".join(x for _, x in rows[i0 : i0 + n][::-1][::every][::-1] if x)
                base[h].append(bool(heard) and clf.predict_text(heard) == rec.dua_id)
    return {h: (float(np.mean(base[h])), float(np.mean(trk[h]))) for h in horizon}, len(trk[horizon[0]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr", default="large-v3-turbo", help="window-cache tag")
    ap.add_argument("--split", choices=["test", "train", "all"], default="test")
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth window: simulates hardware that can only update every N hops")
    ap.add_argument("--tune", action="store_true", help="grid-search tracker settings")
    ap.add_argument("--grid", help='JSON grid for --tune, e.g. {"kappa": [0.1, 0.15]}')
    ap.add_argument("--json", help="write the summary here")
    ap.add_argument("--duas", nargs="*", help="only these du'a ids (the corpus is still all of them)")
    ap.add_argument("--vad", action="store_true", help="apply the live VAD gate (updates with no speech = pause)")
    ap.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE", help="override TrackerConfig fields")
    ap.add_argument("--repaired", action="store_true",
                    help="score line accuracy on repaired labels too (Ziyarat Ashura)")
    ap.add_argument("--ctc", action="store_true",
                    help="score the text against cached CTC posteriors (data/cache/ctc/<asr>) instead of transcripts")
    ap.add_argument("--corpus", choices=["all", "recorded"], default="all",
                    help="recorded: only texts that have recordings (no extra distractors)")
    args = ap.parse_args()

    duas = load_all()
    if args.corpus == "recorded":
        duas = {k: v for k, v in duas.items() if load_recordings(v)}
    ix = CorpusIndex(duas)
    subix = {d: CorpusIndex({d: duas[d]}) for d in duas}
    clf = TextClassifier(duas)
    matchers = {k: PassageMatcher(v) for k, v in duas.items()}

    data = []
    for dua in duas.values():
        if args.duas and dua.id not in args.duas:
            continue
        for rec in load_recordings(dua, repaired=args.repaired):
            if args.split != "all" and is_test(rec.reciter) != (args.split == "test"):
                continue
            rows = load_rows(args.asr, rec, args.window, args.hop, vad=args.vad)[args.stride - 1 :: args.stride]
            if not rows:
                print(f"  (no cache for {dua.id} / {rec.reciter}; skipped)", file=sys.stderr)
                continue
            costs = (load_ctc_costs(args.asr, rec, ix, args.window, args.hop, rows) if args.ctc else
                     [ix.word_costs(text) if text else None for _, text in rows])
            if costs is None:
                print(f"  (no CTC cache for {dua.id} / {rec.reciter}; skipped)", file=sys.stderr)
                continue
            data.append((rec, rows, costs))
    hop = args.hop * args.stride  # seconds between tracker updates
    print(f"{len(data)} recordings, {args.split} split, ASR = {args.asr}, update every {hop:g} s\n")

    base_cfg = replace(TrackerConfig(), **{k: float(v) for k, v in (kv.split("=", 1) for kv in args.set)})
    if args.tune:
        grid = {
            "kappa": [0.15, 0.25, 0.35, 0.5, 0.8],
            "max_speed": [1.5, 2.5, 4.0],
            "p_back": [0.005, 0.02, 0.05],
            "p_teleport": [1e-5, 1e-4, 1e-3],
        }
        if args.grid:
            grid = json.loads(args.grid)
        best = None
        for values in itertools.product(*grid.values()):
            cfg = replace(base_cfg, **dict(zip(grid, values)))
            res = [score(r, rows, run_tracker(ix, cfg, c, hop), refrain_ids(duas[r.dua_id]))
                   for r, rows, c in data if r.dua_id not in LINE_LABELS_UNRELIABLE]
            s = summarize(res)
            ident, _ = identification(ix, cfg, clf, data, hop, n_starts=8, horizon=(5, 10))
            s["id5"], s["id10"] = ident[5][1], ident[10][1]
            # Following matters most; finding the du'a fast mid-recitation second.
            s["objective"] = s["line_acc"] + 0.25 * (s["id5"] + s["id10"]) / 2
            print(f"{dict(zip(grid, values))}  line {s['line_acc']:.3f}  ±1 {s['line_acc_pm1']:.3f}  "
                  f"wrong {s['wrong_dua_shown']:.3f}  jumps/min {s['jumps_per_min']:.2f}  id@5s {s['id5']:.2f}  id@10s {s['id10']:.2f}  "
                  f"lag {s['lag_median_s']:.1f}  obj {s['objective']:.3f}", flush=True)
            if s["wrong_dua_shown"] <= 0.005 and (best is None or s["objective"] > best[1]["objective"]):
                best = (cfg, s)
        print("\nbest:", asdict(best[0]), best[1])
        return

    per_method: dict[str, list[dict]] = {"matcher": [], "tracker": [], "oracle": []}
    per_dua: dict[str, dict[str, list[dict]]] = {}
    for rec, rows, costs in data:
        if rec.dua_id in LINE_LABELS_UNRELIABLE and not args.repaired:
            continue  # still used for identification below
        refrains = refrain_ids(duas[rec.dua_id])
        lo, hi = ix.dua_word_span[ix.dua_ids.index(rec.dua_id)]
        oracle_costs = [None if c is None else c[lo:hi] for c in costs]
        runs = {
            "matcher": run_matcher(duas, clf, matchers, rows),
            "tracker": run_tracker(ix, base_cfg, costs, hop),
            "oracle": run_tracker(subix[rec.dua_id], base_cfg, oracle_costs, hop),
        }
        for name, preds in runs.items():
            s = score(rec, rows, preds, refrains)
            per_method[name].append(s)
            per_dua.setdefault(rec.dua_id, {}).setdefault(name, []).append(s)

    summary = {name: summarize(res) for name, res in per_method.items()}
    print(HEADER)
    labels = {"matcher": "v0.1 per-window matcher", "tracker": "HMM tracker (identifies du'a)",
              "oracle": "HMM tracker, du'a given"}
    for name in per_method:
        print(fmt(labels[name], summary[name]))

    print("\nPer du'a, tracker (matcher):\n\n| du'a | hours | line acc | refrain-line acc |\n|---|---|---|---|")
    for dua_id, m in sorted(per_dua.items()):
        t, b = summarize(m["tracker"]), summarize(m["matcher"])
        n_ref = len(refrain_ids(duas[dua_id]))
        ref = f"{t['refrain_acc']:.1%} ({b['refrain_acc']:.1%})" if n_ref else "—"
        print(f"| {duas[dua_id].name_en} | {t['hours']:.2f} | {t['line_acc']:.1%} ({b['line_acc']:.1%}) | {ref} |")

    ident, n = identification(ix, base_cfg, clf, data, hop)
    print(f"\nIdentification from a random point mid-recitation ({n} starts):\n")
    print("| heard | v0.1 classifier | tracker |\n|---|---|---|")
    for h, (b, t) in ident.items():
        print(f"| {h} s | {b:.1%} | {t:.1%} |")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"asr": args.asr, "split": args.split, "summary": summary,
             "identification": {str(h): {"classifier": b, "tracker": t} for h, (b, t) in ident.items()}},
            indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
