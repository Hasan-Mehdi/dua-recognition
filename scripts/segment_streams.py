#!/usr/bin/env python
"""Find the du'as inside long unlabelled streams, cut them out and label them.

A programme stream (scripts/fetch_fatemah.py) is hours of lecture, majlis and
du'a mixed. The live tracker already knows how to tell those apart: it
identifies the du'a across the whole corpus and has a "not in the corpus"
state for everything else. So: replay the stream's teacher windows
(transcribe_windows.py --stream-dir) through the tracker, keep the spans it
stays locked on one du'a for >= --min-minutes with high confidence, pad them,
cut them to their own audio, and label each with the offline smoother (the
same usable / unusable rule as scripts/align_offline.py). Each span's voice is
checked against the test reciters (scripts/speaker_check.py) so no test voice
reaches training.

    python scripts/segment_streams.py            # -> data/fatemah/spans/<dua>/*.{mp3,json,labels.json}
    python scripts/segment_streams.py --report   # -> data/fatemah/report.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
from align_offline import placed_mask, starts_from  # noqa: E402

from dua_recognition.align import CorpusIndex  # noqa: E402
from dua_recognition.asr import _looks_hallucinated  # noqa: E402
from dua_recognition.offline import align_recording  # noqa: E402
from dua_recognition.splits import is_test  # noqa: E402
from dua_recognition.tracker import Tracker  # noqa: E402

SRC = ROOT / "data" / "fatemah"
AUDIO = SRC / "audio"
SPANS = SRC / "spans"


def load_stream_rows(tag: str, audio_id: str, hop: float) -> list[tuple[float, str]]:
    f = ev.WINDOWS / tag / f"{audio_id}_w6_h{hop:g}.jsonl"
    if not f.exists():
        return []
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x]
    return [(r["t"], "" if _looks_hallucinated(r["text"]) else r["text"]) for r in rows]


def locked_spans(ix, rows, hop: float, conf: float, min_s: float, gap_s: float) -> list[tuple[str, float, float, float]]:
    """(du'a, start s, end s, mean confidence) runs where the tracker shows one du'a at >= conf.
    Short lapses (<= gap_s: a breath, a crowd response) don't end a run."""
    tr = Tracker(ix)
    runs, cur = [], None  # cur = [dua, first t, last t, confs]
    for (t, _), c in zip(rows, ev.LazyCosts(ix, [x for _, x in rows])):
        p = tr.update_costs(c, hop)
        ok = p.dua is not None and p.dua_confidence >= conf
        if ok and cur and p.dua == cur[0]:
            cur[2] = t
            cur[3].append(p.dua_confidence)
        elif ok:  # a new du'a
            if cur:
                runs.append(cur)
            cur = [p.dua, t, t, [p.dua_confidence]]
        elif cur and t - cur[2] > gap_s:  # the lapse outlasted gap_s
            runs.append(cur)
            cur = None
    if cur:
        runs.append(cur)
    # The window ending at t covers [t - 6, t]: a run's audio starts ~6 s before its first window.
    return [(d, max(0.0, a - 6.0), b, float(np.mean(cs))) for d, a, b, cs in runs if b - a + 6.0 >= min_s]


def cut(src: Path, dst: Path, a: float, b: float) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.2f}", "-to", f"{b:.2f}", "-i", str(src),
                    "-vn", "-ac", "1", "-ar", "16000", "-b:a", "96k", str(dst)], check=True)


def route(emb, ref, path: Path) -> tuple[str, str, float]:
    """(split, closest named reciter, score): the untimed duas.org rule (>= 0.7 goes
    with the named reciter; a test voice at 0.55-0.7 is held out; else train)."""
    e = emb(path)
    if e is None:
        return "holdout", "", 0.0
    best: dict[str, float] = {}
    for n, _, r in ref:
        best[n] = max(best.get(n, -1.0), float(e @ r))
    name, s = max(best.items(), key=lambda kv: kv[1])
    test_best = max((v for n, v in best.items() if is_test(n)), default=0.0)
    if s >= 0.7:
        return ("test" if is_test(name) else "train"), name, s
    if test_best >= 0.55:
        return "holdout", max((n for n in best if is_test(n)), key=best.get), test_best
    return "train", name, s


def segment(args) -> None:
    from speaker_check import Embedder, canonical, named_recordings

    duas = ev.load_all()
    ix = CorpusIndex(duas)
    emb = Embedder()
    ref = [(canonical(n), p, emb(p)) for n, p in named_recordings()]
    ref = [(n, p, e) for n, p, e in ref if e is not None]
    for meta_path in sorted(AUDIO.glob("*.json")):
        vid = meta_path.stem
        audio = [p for p in AUDIO.glob(f"{vid}.*") if p.suffix not in (".json", ".part")]
        rows = load_stream_rows(args.tag, f"fa-{vid}", args.hop)
        if not audio or not rows:
            print(f"  {vid}: no audio or no windows yet; skipped", flush=True)
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        spans = locked_spans(ix, rows, args.hop, args.conf, args.min_minutes * 60, args.gap)
        print(f"{vid} ({meta.get('duration', 0) / 60:.0f} min) {meta.get('title', '')[:60]}: {len(spans)} spans", flush=True)
        for dua_id, a, b, conf in spans:
            a, b = max(0.0, a - args.pad), min(rows[-1][0], b + args.pad)
            sid = f"{vid}-{int(a)}"
            d = SPANS / dua_id
            d.mkdir(parents=True, exist_ok=True)
            mp3 = d / f"{sid}.mp3"
            if (d / f"{sid}.labels.json").exists():
                continue
            cut(audio[0], mp3, a, b)
            # Label on the stream's own windows, shifted to the span's clock.
            sub = [(round(t - a, 2), x) for t, x in rows if a + 6.0 <= t <= b]
            dix = CorpusIndex({dua_id: duas[dua_id]})
            costs = [dix.word_costs(x) if x else None for _, x in sub]
            segs = align_recording(dix, costs, args.hop)
            placed = placed_mask(dix, sub, costs)
            starts = starts_from([t for t, _ in sub], segs, placed)
            voiced = [p for (_, x), p in zip(sub, placed) if x]
            coverage = sum(voiced) / max(1, len(voiced))
            lines = len({s for _, s in starts}) / len(duas[dua_id].segments)
            usable = coverage >= 0.5 and lines >= 0.5
            split, who, vs = route(emb, ref, mp3)
            (d / f"{sid}.json").write_text(json.dumps({
                "video_id": sid, "dua_id": dua_id, "audio": mp3.name, "duration_s": round(b - a, 2),
                "stream": vid, "stream_start_s": round(a, 2), "title": meta.get("title", ""),
                "reciter": "Al-Fatemah IC", "split": split, "voice": {"closest": who, "score": round(vs, 3)},
                "tracker_confidence": round(conf, 3)}, ensure_ascii=False, indent=1), encoding="utf-8")
            (d / f"{sid}.labels.json").write_text(json.dumps({
                "video_id": sid, "dua_id": dua_id, "reciter": "fatemah",
                "slide_start_s": {str(s): t for t, s in starts}, "end_s": sub[-1][0] if sub else 0,
                "placed_fraction": round(coverage, 3), "lines_found": round(lines, 3), "usable": usable,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  {dua_id:28s} {a / 60:6.1f}-{b / 60:6.1f} min  conf {conf:.2f}  placed {coverage:5.1%}  "
                  f"lines {lines:5.1%}  {'ok' if usable else 'UNUSABLE'}  voice {split} ({who} {vs:.2f})", flush=True)


def report(args) -> None:
    idx = {json.loads(x)["id"]: json.loads(x) for x in (SRC / "index.jsonl").read_text(encoding="utf-8").splitlines() if x}
    got = [json.loads(p.read_text(encoding="utf-8")) for p in AUDIO.glob("*.json")]
    spans = []
    for lab in sorted(SPANS.glob("*/*.labels.json")):
        m = json.loads(lab.with_name(lab.name.replace(".labels.json", ".json")).read_text(encoding="utf-8"))
        spans.append((m, json.loads(lab.read_text(encoding="utf-8"))))
    duas = ev.load_all()
    per = defaultdict(lambda: [0.0, 0.0, 0])  # hours, usable hours, spans
    splits = defaultdict(lambda: [0, 0.0])
    for m, lab in spans:
        h = m["duration_s"] / 3600
        per[m["dua_id"]][0] += h
        per[m["dua_id"]][2] += 1
        if lab["usable"]:
            per[m["dua_id"]][1] += h
            splits[m["split"]][0] += 1
            splits[m["split"]][1] += h
    total_h = sum(r.get("duration") or 0 for r in got) / 3600
    lines = [
        "# Al-Fatemah IC harvest", "",
        f"Channel index: {len(idx)} videos. Downloaded: {len(got)} streams, {total_h:.1f} h "
        f"(scripts/fetch_fatemah.py; titles ranked by du'a name, no du'a over a quarter of the budget).", "",
        f"Du'a spans found by the tracker (locked >= {args.conf:g} for >= {args.min_minutes:g} min, padded "
        f"±{args.pad:g} s): {len(spans)} spans, {sum(m['duration_s'] for m, _ in spans) / 3600:.1f} h; "
        f"usable after labelling (placed >= 50% and >= 50% of lines found): "
        f"{sum(1 for _, l in spans if l['usable'])} spans, "
        f"{sum(m['duration_s'] for m, l in spans if l['usable']) / 3600:.1f} h.", "",
        "| du'a | spans | hours | usable hours |", "|---|---:|---:|---:|",
    ]
    for d, (h, uh, n) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"| {duas[d].name_en if d in duas else d} (`{d}`) | {n} | {h:.2f} | {uh:.2f} |")
    lines += ["", "Voice routing of usable spans (scripts/speaker_check.py; >= 0.7 goes with the named reciter, "
              "a test voice at 0.55-0.7 is held out):", ""]
    lines += [f"- {k}: {n} spans, {h:.1f} h" for k, (n, h) in sorted(splits.items())]
    usable = [(m, l) for m, l in spans if l["usable"]]
    rng = np.random.default_rng(0)
    pick = [usable[i] for i in rng.choice(len(usable), size=min(5, len(usable)), replace=False)] if usable else []
    lines += ["", "## Spot-check these", "",
              "Each link opens the stream where the span starts (after the padding): does it hold the du'a named?", ""]
    for m, lab in pick:
        t = int(m["stream_start_s"])
        lines.append(f"- {duas[m['dua_id']].name_en if m['dua_id'] in duas else m['dua_id']}: "
                     f"https://www.youtube.com/watch?v={m['stream']}&t={t}s ({m['duration_s'] / 60:.0f} min, "
                     f"placed {lab['placed_fraction']:.0%}, voice {m['split']})")
    lines += ["", "No training on this yet."]
    (SRC / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="large-v3-turbo", help="window cache of the teacher pass")
    ap.add_argument("--hop", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.9, help="du'a confidence that counts as locked")
    ap.add_argument("--min-minutes", type=float, default=3.0)
    ap.add_argument("--gap", type=float, default=20.0, help="seconds of lapse a span survives")
    ap.add_argument("--pad", type=float, default=5.0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report(args) if args.report else segment(args)


if __name__ == "__main__":
    main()
