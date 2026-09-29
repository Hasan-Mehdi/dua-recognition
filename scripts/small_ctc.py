#!/usr/bin/env python
"""One bounded small-CTC comparison (plan Step 4): Tilawa's FastConformer text CTC vs wav2vec2.

    python scripts/small_ctc.py validate            # a few dev lines: transcript, tokenization check
    python scripts/small_ctc.py bench               # size, load, cold/warm window latency, RTF (desktop)
    python scripts/small_ctc.py dump                # 3 s / 0.2 s windows over the dev (train) set
    python scripts/follow_eval.py --split train --ctc fastconformer-tilawa --token-model DIR "fw"

Model: fastconformer_full_mixed.onnx from github.com/yazinsai/tilawa release v0.2.0
(nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0 fine-tune, CC-BY-4.0; int4+int8), kept under
data/cache/ctc/models (D:). Buffered windows, not cached streaming (ctc_adapter.py).
Dumps keep only the columns the follower reads (blank, best non-blank, the du'a's own
tokens): exact for its scoring, and ~30x smaller than the full 1,025-way posteriors.
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
from dua_recognition.ctc_adapter import TokenCtcModel, corpus_surface_words, surface  # noqa: E402

MODEL_DIR = ROOT / "data" / "cache" / "ctc" / "models" / "tilawa-fastconformer-v0.2.0"
TAG = "fastconformer-tilawa"
OUT = ROOT / "data" / "cache" / "ctc" / TAG
RUN = ROOT / "data" / "cache" / "reliability" / os.environ.get("DUA_REL_RUN", "rel-20260927")
SR = 16000


def cer(a: str, b: str) -> float:
    from dua_recognition.text import normalize

    a, b = normalize(a).replace(" ", ""), normalize(b).replace(" ", "")
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / max(1, len(b))


def dev_recs():
    for dua in ev.load_all().values():
        for r in ev._load_recordings(dua, extra=False):
            if not ev._is_test(r.reciter) and dua.id not in ev.LINE_LABELS_UNRELIABLE:
                yield dua, r


def validate(args) -> None:
    from faster_whisper.audio import decode_audio

    m = TokenCtcModel(MODEL_DIR)
    ix = ev.CorpusIndex(ev.load_all())
    surf = corpus_surface_words(ix)
    rng = np.random.default_rng(0)
    recs = list(dev_recs())
    pick = [recs[i] for i in rng.choice(len(recs), size=min(args.recs, len(recs)), replace=False)]
    rows, seg_same, seg_n, missing = [], 0, 0, 0
    for dua, rec in pick:
        y = decode_audio(str(rec.path), sampling_rate=SR)
        di = ix.dua_ids.index(dua.id)
        lo, hi = ix.dua_word_span[di]
        bounds = list(rec.starts) + [(rec.end_s, None)]
        for k in rng.choice(len(bounds) - 1, size=min(args.lines, len(bounds) - 1), replace=False):
            (a, seg), (b, _) = bounds[k], bounds[k + 1]
            if b - a < 1.0 or b - a > 20:
                continue
            ref_words = [surf[w] for w in range(lo, hi) if ix.word_segment[w] == seg]
            lp = m.native(y[int(max(0, a - 0.3) * SR): int((b + 0.3) * SR)])
            toks = m.greedy(lp)
            hyp = m.decode(toks)
            ref = " ".join(ref_words)
            rows.append({"dua": dua.id, "line": seg, "s": round(b - a, 1), "cer": round(cer(hyp, ref), 3),
                         "ref": ref, "hyp": hyp})
            # tokenization check: for hypothesis words that equal a reference word, did the model
            # emit the same pieces our tokenizer assigns?
            words, cur = [], []
            for t in toks:
                p = m.pieces[t]
                if p.startswith("▁") and cur:
                    words.append(cur)
                    cur = []
                cur.append(t)
            if cur:
                words.append(cur)
            refset = set(ref_words)
            for wt in words:
                txt = m.decode(wt)
                if txt in refset:
                    seg_n += 1
                    seg_same += m.tokenize(txt) == wt
            for w in ref_words:
                missing += len(w) - len("".join(m.pieces[t] for t in m.tokenize(w)).replace("▁", ""))
    res = {"model": m.name, "frame_s": m.frame_s, "blank": m.blank, "vocab": len(m.pieces),
           "lines": len(rows), "median_line_cer": float(np.median([r["cer"] for r in rows])),
           "mean_line_cer": float(np.mean([r["cer"] for r in rows])),
           "tokenization_matches_model": f"{seg_same}/{seg_n}", "reference_chars_not_in_vocab": missing,
           "examples": rows[:12]}
    (RUN / "small_ctc").mkdir(parents=True, exist_ok=True)
    (RUN / "small_ctc" / "validate.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, ensure_ascii=False, indent=1))
    for r in rows[:8]:
        print(f"{r['cer']:.2f}  REF {r['ref']}\n      HYP {r['hyp']}")


def bench(args) -> None:
    import os

    import psutil

    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    m = TokenCtcModel(MODEL_DIR, threads=args.threads)
    load = time.perf_counter() - t0
    x = (np.random.default_rng(0).standard_normal(3 * SR) * 0.05).astype(np.float32)
    t0 = time.perf_counter()
    m.native(x)
    cold = time.perf_counter() - t0
    warm = []
    for _ in range(30):
        t0 = time.perf_counter()
        m.native(x)
        warm.append(time.perf_counter() - t0)
    res = {"model": m.name, "bytes": (MODEL_DIR / m.name).stat().st_size, "threads": args.threads,
           "load_s": load, "cold_ms": cold * 1000, "warm_p50_ms": float(np.median(warm)) * 1000,
           "warm_p90_ms": float(np.percentile(warm, 90)) * 1000,
           "rtf_3s_window_every_0.2s": float(np.median(warm)) / 0.2, "rss_mb": (proc.memory_info().rss - rss0) / 2**20,
           "note": "desktop CPU (onnxruntime); not a phone measurement"}
    (RUN / "small_ctc").mkdir(parents=True, exist_ok=True)
    (RUN / "small_ctc" / f"bench_fastconformer_t{args.threads}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def write_meta(m, ix) -> dict:
    """Every corpus word's pieces (keyed by du'a), frame hop, blank, width: what the follower needs.
    Rewritten for the current corpus (DUA_CORPUS_DIR) without touching the dumps."""
    surf = corpus_surface_words(ix)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"model": m.name, "frame_s": m.frame_s, "blank": m.blank, "width": len(m.pieces) + 1,
            "word_tokens": {ix.dua_ids[d]: [m.tokenize(surf[w]) for w in range(*ix.dua_word_span[d])]
                            for d in range(len(ix.dua_ids))}}
    (OUT / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def dump(args) -> None:
    from faster_whisper.audio import decode_audio

    m = TokenCtcModel(MODEL_DIR, threads=args.threads)
    ix = ev.CorpusIndex(ev.load_all())
    meta = write_meta(m, ix)
    for dua, rec in dev_recs():
        f = OUT / f"{rec.audio_id}_w3_h0.2.npz"
        truth = ev.load_word_truth(rec, ix)
        if f.exists() or not truth:
            continue
        good = [w for w in truth if w[3] >= -1.5]
        if not good:
            continue
        y = decode_audio(str(rec.path), sampling_rate=SR)
        # the incumbent's grid (dump_ctc.py --window 3 --hop 0.2): every 0.2 s from the start,
        # up to the end of the follow lane's span (its last good word + 2 s)
        times = np.round(np.arange(0.2, min(rec.end_s, good[-1][2] + 2.0) + 0.2, 0.2), 3)
        t0 = time.time()
        windows_to_npz(m, meta, y, times, dua.id, f)
        print(f"{dua.id:28s} {rec.reciter[:20]:20s} {len(times):5d} windows {time.time() - t0:6.0f} s", flush=True)


def windows_to_npz(m, meta, y, times, dua_id, f) -> None:
    """3 s windows ending at `times`, restricted to the columns the follower reads for this du'a."""
    toks = sorted({t for ws in meta["word_tokens"][dua_id] for t in ws})
    cols = np.array([0, 1] + [m.col(t) for t in toks])
    lps = []
    for t in times:
        w = y[int(max(0.0, t - 3.0) * SR): int(t * SR)]
        lps.append(m.columns(m.native(w))[:, cols].astype(np.float16) if w.size >= 1600 else
                   np.zeros((0, cols.size), np.float16))
    nf = np.array([a.shape[0] for a in lps], np.int32)
    arr = np.full((len(lps), max(1, int(nf.max())), cols.size), -30.0, np.float16)
    for k, a in enumerate(lps):
        arr[k, : len(a)] = a
    np.savez_compressed(f, lp=arr, n_frames=nf, t=np.asarray(times), cols=cols, frame_s=m.frame_s)


def dump_pauses(args) -> None:
    """The train pause set (pause_eval.py build --split train) through the same model, same grid
    as pause_eval's build_ctc, into <pause set>/ctc-fastconformer-tilawa/."""
    import pause_eval as pe
    from faster_whisper.audio import decode_audio

    m = TokenCtcModel(MODEL_DIR, threads=args.threads)
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    recs = {r.audio_id: r for _, r in dev_recs()}
    src = pe.OUT / "whisper-base-quran-dua@train"
    cdir = src / f"ctc-{TAG}"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for jf in sorted(src.glob("*.json")):
        f = cdir / f"{jf.stem}_w3_h0.2.npz"
        r = json.loads(jf.read_text(encoding="utf-8"))
        if f.exists() or r["audio_id"] not in recs:
            continue
        P = r["pause_s"]
        y = decode_audio(str(recs[r["audio_id"]].path), sampling_rate=SR)
        y2 = pe.paused_audio(y, [s - P * j for j, (s, _) in enumerate(r["pauses"])], P)
        end = r["rows"][-1][0]
        times = [round((k + 1) * 0.2, 3) for k in range(int((end + 1e-6) // 0.2))]
        t0 = time.time()
        windows_to_npz(m, meta, y2, times, r["dua"], f)
        print(f"  pauses {jf.stem} {len(times)} windows {time.time() - t0:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["validate", "bench", "dump", "dump-pauses", "meta"])
    ap.add_argument("--recs", type=int, default=5)
    ap.add_argument("--lines", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "meta":
        write_meta(TokenCtcModel(MODEL_DIR), ev.CorpusIndex(ev.load_all()))
        return
    {"validate": validate, "bench": bench, "dump": dump, "dump-pauses": dump_pauses}[args.cmd](args)


if __name__ == "__main__":
    main()
