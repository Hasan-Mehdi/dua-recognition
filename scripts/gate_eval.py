#!/usr/bin/env python
"""Browser speech-gate policies (web/gate.js) on development audio: what each discards and costs.

    python scripts/gate_eval.py prepare     # 16 kHz float32 audio + jobs for the Node replay
    node scripts/gate_replay.mjs <run>/gate/jobs.json <run>/gate/decisions
    python scripts/gate_eval.py score       # tables -> <run>/gate/results.json

The gate decisions come from the browser's own code (gate_replay.mjs runs web/gate.js
with the same Silero model). Transcripts come from the phone checkpoint
(whisper-base-aug-v4) through CTranslate2, from the ungated window cache: a different
runtime from the page's ONNX q8 (compared separately in the browser benchmark).
Whisper's text for a hop is used only if the policy lets the hop run; otherwise the
tracker gets "" as the page does. The tracker is the Python one (parity-tested with
web/tracker.js), in page mode with a 0.5 s delay.

Sets (development only; nothing from test):
    flow      DuaPlayer train recordings, at 0 / -10 / -20 dB (the quieter copies: gate
              decisions only; their transcripts are not recomputed)
    pause     the train pause set (4 s of each recording's own room tone after every third
              line end, scripts/pause_eval.py), scored on the phone checkpoint
    sessions  saved debug sessions (unlabelled; gate statistics only)

Labels: human DuaPlayer line starts (line accuracy, onset lag, missed transitions);
automatic forced-aligned words mark "annotated speech" for the discarded-speech measure
(automatic-label diagnostic, not human truth).
"""
from __future__ import annotations

import os
import argparse
import bisect
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
import pause_eval as pe  # noqa: E402
import word_eval as we  # noqa: E402
from dua_recognition.asr import _looks_hallucinated  # noqa: E402
from dua_recognition.tracker import TrackerConfig  # noqa: E402

RUN = ROOT / "data" / "cache" / "reliability" / os.environ.get("DUA_REL_RUN", "rel-20260927")
G = RUN / "gate"
SR = 16000
TAG = "whisper-base-aug-v4"  # the phone checkpoint (web/models/whisper-base-aug-v4)
POLICIES = ["legacy", "energy_assisted", "ungated"]
GAINS = [0, -10, -20]
DELAY = 0.5


def dev_recordings():
    for dua in ev.load_all().values():
        for r in ev._load_recordings(dua, extra=False):
            if not ev._is_test(r.reciter):
                yield dua, r


def prepare(args) -> None:
    from faster_whisper.audio import decode_audio

    pcm = G / "pcm"
    pcm.mkdir(parents=True, exist_ok=True)
    jobs = []

    def put(key: str, y: np.ndarray, gains=(0,), name=None) -> None:
        f = pcm / f"{name or key}.f32"
        if not f.exists():
            y.astype("<f4").tofile(f)
        for g in gains:
            jobs.append({"key": key.format(g=g), "pcm": str(f), "gain_db": g})

    for dua, r in dev_recordings():
        y = decode_audio(str(r.path), sampling_rate=SR)
        put("flow_{g:+d}_" + r.audio_id, y, GAINS, name=f"flow_{r.audio_id}")
    pdir = pe.OUT / f"{TAG}@train"
    recs = {r.audio_id: r for _, r in dev_recordings()}
    for jf in sorted(pdir.glob("*.json")):
        p = json.loads(jf.read_text(encoding="utf-8"))
        r = recs[p["audio_id"]]
        y = decode_audio(str(r.path), sampling_rate=SR)
        P = p["pause_s"]
        at = [s - P * j for j, (s, _) in enumerate(p["pauses"])]  # pause times in the original audio
        put(f"pause_{r.audio_id}", pe.paused_audio(y, at, P))
    for f in sorted((ROOT / "data" / "sessions").glob("*.wav")):
        from session_report import read_session

        y, _ = read_session(f)
        put(f"session_{f.stem}", y)
    (G / "jobs.json").write_text(json.dumps(jobs, indent=0), encoding="utf-8")
    print(f"{len(jobs)} jobs -> {G / 'jobs.json'}")


def decisions(key: str) -> list[dict]:
    f = G / "decisions" / f"{key}.jsonl"
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x] if f.exists() else []


def speech_mask(words, n_hops: int) -> np.ndarray:
    """Seconds of annotated (aligned-word) speech in each hop's new second (t-1, t]."""
    edges = np.zeros(n_hops * 100 + 1)
    for _, a, b, *_ in words:
        i, j = int(round(a * 100)), int(round(b * 100))
        edges[max(0, i): max(0, min(j, n_hops * 100))] = 1
    return edges[: n_hops * 100].reshape(n_hops, 100).sum(1) / 100


def replay_lines(ix, rec, rows, quiet, cfg) -> dict:
    """Page-mode tracker over (t, text) rows; per 0.1 s tick what line/du'a is shown."""
    costs = ev.LazyCosts(ix, [x for _, x in rows])
    ups = we.hmm_updates(ix, rows, costs, quiet, cfg, DELAY, "fixed", None, still=True)
    t0, t1 = rec.starts[0][0], rec.end_s
    ticks = np.arange(t0, t1, 0.1)
    times = [u[0] for u in ups]
    exact = pm1 = wrong = shown_n = 0
    segs = []
    for tick in ticks:
        k = bisect.bisect_right(times, tick) - 1
        dua, seg = (ups[k][1], ups[k][2]) if k >= 0 else (None, None)
        truth = rec.segment_at(tick)
        segs.append(seg if dua == rec.dua_id else None)
        if dua is not None:
            shown_n += 1
            wrong += dua != rec.dua_id
        if truth is None:
            continue
        exact += dua == rec.dua_id and seg == truth
        pm1 += dua == rec.dua_id and seg is not None and abs(seg - truth) <= 1
    n = sum(rec.segment_at(t) is not None for t in ticks)
    # line entries vs the human starts: first tick in +-4 s entering that line
    lags = []
    for t, s in rec.starts[1:]:
        a, b = np.searchsorted(ticks, [t - 4.0, t + 4.0])
        lag = None
        for i in range(max(1, a), min(b, len(ticks))):
            if segs[i] == s and segs[i - 1] != s:
                lag = float(ticks[i] - t)
                break
        lags.append(lag)
    return {"ticks": n, "exact": exact, "pm1": pm1, "wrong": wrong, "all_ticks": len(ticks), "shown": shown_n,
            "lags": lags, "segs": segs, "tick0": float(t0)}


def gated_rows(rows, dec, policy, filt: bool):
    by_t = {round(d["t"]): d for d in dec}
    out = []
    for t, x in rows:
        d = by_t.get(round(t))
        run = d is not None and d[policy]["run"]
        x = x if run else ""
        if filt and x and _looks_hallucinated(x):
            x = ""
        out.append((t, x))
    return out


def quiet_of(rows, dec):
    by_t = {round(d["t"]): d["quiet"] for d in dec}
    return [by_t.get(round(t), 0.0) for t, _ in rows]


def raw_rows(rec) -> list[tuple[float, str]]:
    """The phone checkpoint's windows, unfiltered (the page has no hallucination filter)."""
    f = ev.WINDOWS / TAG / f"{rec.audio_id}_w6_h1.jsonl"
    if not f.exists():
        return []
    return [(r["t"], r["text"]) for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()) if r]


def boot_ci(per_rec: dict, groups: dict, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean per-recording difference, resampling reciters (grouped)."""
    rng = random.Random(seed)
    by_g = defaultdict(list)
    for k, v in per_rec.items():
        by_g[groups[k]].append(v)
    keys = sorted(by_g)
    if len(keys) < 2:
        return float("nan"), float("nan")
    means = []
    for _ in range(n):
        pick = [x for g in (rng.choice(keys) for _ in keys) for x in by_g[g]]
        means.append(float(np.mean(pick)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score(args) -> None:
    duas = ev.load_all()
    ix = ev.CorpusIndex(duas)
    cfg = TrackerConfig()
    res = {"flow": {}, "gain": {}, "pause": {}, "sessions": {}, "per_recording": {}}
    groups = {}
    # -- flowing recordings ------------------------------------------------------
    variants = [(p, False) for p in POLICIES] + [("legacy", True), ("energy_assisted", True), ("ungated", True)]
    agg = {f"{p}{'+filter' if f else ''}": defaultdict(float) for p, f in variants}
    lags = defaultdict(list)
    per = defaultdict(dict)
    for dua, rec in dev_recordings():
        dec = decisions(f"flow_+0_{rec.audio_id}")
        rows = raw_rows(rec)
        words = ev.load_word_truth(rec, ix)
        if not dec or not rows or not words:
            continue
        groups[rec.audio_id] = rec.reciter
        sp = speech_mask(words, len(dec))
        for g in GAINS:
            dg = decisions(f"flow_{g:+d}_{rec.audio_id}")
            for p in POLICIES:
                ran = np.array([d[p]["run"] for d in dg], bool)
                a = res["gain"].setdefault(f"{p} @ {g:+d} dB", defaultdict(float))
                a["speech_s"] += float(sp.sum())
                a["discarded_s"] += float(sp[~ran[: len(sp)]].sum())
                a["hops"] += len(dg)
                a["ran"] += int(ran.sum())
                if g == 0:
                    reasons = defaultdict(int)
                    for d in dg:
                        reasons[d[p]["reason"] or "ran"] += 1
                    for k, v in reasons.items():
                        a[f"reason:{k}"] += v
                per[f"discard {p} @ {g:+d} dB"][rec.audio_id] = float(sp[~ran[: len(sp)]].sum() / max(sp.sum(), 1e-9))
        if dua.id in ev.LINE_LABELS_UNRELIABLE:
            continue
        for p, f in variants:
            name = f"{p}{'+filter' if f else ''}"
            gr = gated_rows(rows, dec, p, f)
            r = replay_lines(ix, rec, gr, quiet_of(rows, dec), cfg)
            a = agg[name]
            for k in ("ticks", "exact", "pm1", "wrong", "all_ticks", "shown"):
                a[k] += r[k]
            a["nonempty_hops"] += sum(1 for _, x in gr if x)
            lags[name] += r["lags"]
            per[f"exact {name}"][rec.audio_id] = r["exact"] / max(1, r["ticks"])
            per[f"wrong {name}"][rec.audio_id] = r["wrong"] / max(1, r["all_ticks"])
        print(f"  {rec.audio_id[:8]} {dua.id:28s} " + "  ".join(
            f"{p[:6]} {per[f'exact {p}'][rec.audio_id]:.1%}" for p in POLICIES), flush=True)
    for name, a in agg.items():
        L = [x for x in lags[name] if x is not None]
        res["flow"][name] = {
            "line_exact": a["exact"] / max(1, a["ticks"]), "line_pm1": a["pm1"] / max(1, a["ticks"]),
            "wrong_dua_exposure": a["wrong"] / max(1, a["all_ticks"]), "shown": a["shown"] / max(1, a["all_ticks"]),
            "onset_lag_median": float(np.median(L)) if L else None,
            "onset_lag_p90": float(np.percentile(L, 90)) if L else None,
            "early_gt_0.3": float(np.mean([x < -0.3 for x in L])) if L else None,
            "missed_transitions": 1 - len(L) / max(1, len(lags[name])), "transitions": len(lags[name]),
            "nonempty_hops": int(a["nonempty_hops"])}
    for k, a in res["gain"].items():
        res["gain"][k] = {**{kk: v for kk, v in a.items()}, "discarded_share": a["discarded_s"] / max(a["speech_s"], 1e-9),
                          "ran_share": a["ran"] / max(1, a["hops"])}
    # paired differences vs legacy, grouped bootstrap over reciters
    for metric in ("exact", "wrong"):
        for p in [v for v in agg if v != "legacy"]:
            base = per[f"{metric} legacy"]
            diff = {k: per[f"{metric} {p}"][k] - base[k] for k in base if k in per[f"{metric} {p}"]}
            lo, hi = boot_ci(diff, groups)
            res["per_recording"][f"{metric} {p} - legacy"] = {
                "mean": float(np.mean(list(diff.values()))) if diff else None, "ci95": [lo, hi],
                "worse": sum(v < 0 for v in diff.values()) if metric == "exact" else sum(v > 0 for v in diff.values()),
                "n": len(diff), "per_rec": {k: round(v, 4) for k, v in diff.items()}}
    for p in POLICIES[1:]:
        base = per["discard legacy @ +0 dB"]
        diff = {k: per[f"discard {p} @ +0 dB"][k] - base[k] for k in base}
        lo, hi = boot_ci(diff, groups)
        res["per_recording"][f"discard {p} - legacy"] = {"mean": float(np.mean(list(diff.values()))), "ci95": [lo, hi],
                                                         "n": len(diff)}
    # -- pause set: nonspeech behaviour ------------------------------------------------
    pdir = pe.OUT / f"{TAG}@train"
    for p, f in variants:
        name = f"{p}{'+filter' if f else ''}"
        a = defaultdict(float)
        for jf in sorted(pdir.glob("*.json")):
            r = json.loads(jf.read_text(encoding="utf-8"))
            dec = decisions(f"pause_{r['audio_id']}")
            if not dec:
                continue
            rows = [(t, x) for t, x in r["rows"]]
            gr = gated_rows(rows, dec, p, f)
            P = r["pause_s"]
            lo = ix.dua_word_span[ix.dua_ids.index(r["dua"])][0]
            truth = [[lo + w, s, e, 0.0] for w, s, e in r["truth"]]
            starts = [w[1] for w in truth]
            costs = ev.LazyCosts(ix, [x for _, x in gr])
            ups = we.hmm_updates(ix, gr, costs, quiet_of(rows, dec), cfg, DELAY, "fixed", None, still=True)
            times = [u[0] for u in ups]
            for s, _ in r["pauses"]:
                # hops whose new second lies inside the pause (pure room tone)
                for t, x in gr:
                    if s + 1.0 <= t <= s + P:
                        a["pause_hops"] += 1
                        a["pause_hops_text"] += bool(x)
                for tick in np.arange(s + 0.5, s + P, 0.1):
                    k = bisect.bisect_right(times, tick) - 1
                    i = bisect.bisect_right(starts, tick) - 1
                    if k < 0 or i < 0 or ups[k][3] is None:
                        continue
                    tw = truth[i][0]
                    w = ups[k][3]
                    a["pause_ticks"] += 1
                    a["next_line"] += bool(ix.word_segment[w] != ix.word_segment[tw] and w > tw)
                    a["wrong_dua"] += ups[k][1] != r["dua"]
        res["pause"][name] = {"transcribed_nonspeech_hops": a["pause_hops_text"] / max(1, a["pause_hops"]),
                              "next_line_in_pause": a["next_line"] / max(1, a["pause_ticks"]),
                              "wrong_dua_in_pause": a["wrong_dua"] / max(1, a["pause_ticks"]),
                              "pause_hops": int(a["pause_hops"])}
    # -- sessions: gate statistics only -------------------------------------------------
    for f in sorted((G / "decisions").glob("session_*.jsonl")):
        dec = decisions(f.stem)
        res["sessions"][f.stem[8:]] = {
            "hops": len(dec), "median_level_db": float(np.median([d["legacy"]["level"] for d in dec])) if dec else None,
            **{f"ran {p}": sum(d[p]["run"] for d in dec) / max(1, len(dec)) for p in POLICIES},
            "below_floor": sum(d["legacy"]["reason"] == "below_floor" for d in dec) / max(1, len(dec))}
    import os

    out = G / os.environ.get("GATE_RESULTS", "results.json")
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("flow", "pause")}, indent=1, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_rec"} for k, v in res["per_recording"].items()},
                     indent=1))
    print(json.dumps({k: {kk: round(v[kk], 4) for kk in ("discarded_share", "ran_share")} for k, v in res["gain"].items()},
                     indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "score"])
    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else score(args)


if __name__ == "__main__":
    main()
