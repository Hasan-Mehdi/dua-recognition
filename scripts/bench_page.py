#!/usr/bin/env python
"""The bench through the real page: bench items played into web/ in headless Chrome.

bench.py replays the display in Python from cached model outputs. This plays a sample of the
same items into the actual page (scripts/page_replay.mjs: the browser's own Whisper and CTC
models, follower.js, tracker.js, the capture worklet, in real time), reads what the page showed
from its session log (one `ctc` event per follower step, with the word on screen), and scores it
with bench.metrics, so the two can be compared item for item.

    python scripts/bench_page.py --per-scenario 1 --split test --variants page_rules: page_stream:follower=stream
    python scripts/bench_page.py --report --variants page_rules: page_stream:follower=stream

--variants name:query plays every item through each page variant; with --jobs equal to the number of
variants, the variants of one item play at the same time, so they see the same load on the machine
(the page runs in real time: a busy CPU delays its models).

--asr-ms / --ctc-ms make each Whisper / CTC step take at least that long (a phone: Whisper
~1-2.5 s per update on Hasan's Android, the CTC model ~0.1-0.25 s).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench  # noqa: E402

OUT = bench.BENCH / "page"


def run_one(it: dict, args, name: str | None = None, query: str | None = None) -> Path | None:
    import soundfile as sf

    name = name or args.name
    query = args.query if query is None else query
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    wav_in, wav_out = d / f"{it['id']}.in.wav", d / f"{it['id']}.session.wav"
    if wav_out.exists():
        return wav_out
    sf.write(wav_in, bench.render(it), bench.SR, subtype="PCM_16")
    cmd = ["node", str(ROOT / "scripts" / "page_replay.mjs"), str(wav_in), str(wav_out), query]
    if args.asr_ms:
        cmd += ["--asr-ms", str(args.asr_ms)]
    if args.ctc_ms:
        cmd += ["--ctc-ms", str(args.ctc_ms)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    print(p.stdout.strip() or p.stderr.strip()[-300:], flush=True)
    wav_in.unlink(missing_ok=True)
    return wav_out if wav_out.exists() else None


def shown_from_log(it: dict, path: Path, ix) -> tuple[list, float]:
    """Per 0.1 s tick of item time, the page's word on screen, from its `ctc` events; and the
    offset between the session's audio clock and the item (found by cross-correlation)."""
    import session_report as sr

    audio, log = sr.read_session(path)
    y = bench.render(it)
    # The session starts recording a little before the file plays: find the shift.
    a, b = audio[: 30 * bench.SR], y[: 30 * bench.SR]
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    xc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    lag = int(np.argmax(xc[: len(a)]))  # item time 0 is at session sample `lag`
    off = lag / bench.SR
    T = int(it["duration"] * 10)
    shown, k = [], 0
    ev = sorted((e for e in log["events"] if e["type"] == "ctc"), key=lambda e: e["t"])
    cur = None
    for i in range(T):
        t = i / 10 + off
        while k < len(ev) and ev[k]["t"] <= t:
            w = ev[k].get("word")
            cur = None if w is None else (ix.dua_ids[ix.word_dua[w]], int(ix.word_segment[w]), int(w))
            k += 1
        shown.append(cur)
    return shown, off


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="page_test")
    ap.add_argument("--per-scenario", type=int, default=2)
    ap.add_argument("--split", default="test")
    ap.add_argument("--asr-ms", type=int, default=1200)
    ap.add_argument("--ctc-ms", type=int, default=150)
    ap.add_argument("--query", default="", help="page URL query, e.g. 'model=...'")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--report", action="store_true", help="only score what has been played")
    ap.add_argument("--variants", nargs="*", default=None, help="name:query pairs (default: --name with --query)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    items = bench.load_items(split=args.split)
    by = {}
    for it in items:
        by.setdefault(it["scenario"], []).append(it)
    pick = []
    for sc, its in sorted(by.items()):
        rng = random.Random(sc)
        pick += rng.sample(its, min(args.per_scenario, len(its)))
    variants = [tuple(v.split(":", 1)) for v in args.variants] if args.variants else [(args.name, args.query)]
    if not args.report:
        tasks = [(it, n, q) for it in pick for n, q in variants]  # an item's variants side by side
        with ThreadPoolExecutor(args.jobs) as ex:
            list(ex.map(lambda x: run_one(x[0], args, x[1], x[2]), tasks))
    import evaluate as ev

    ix = ev.CorpusIndex(ev.load_all())
    bench._worker_init({}, {}, bench.ASR_TAG, bench.CTC_TAG, 1.2, 0.15)
    res = {}
    for name, _ in variants:
        page = {}
        for it in pick:
            f = OUT / name / f"{it['id']}.session.wav"
            if not f.exists():
                continue
            shown, off = shown_from_log(it, f, ix)
            m = bench.metrics(it, shown, ix)
            m.update(scenario=it["scenario"], lane=it["lane"], voice=it["voice"], split=bench.split_of(it["voice"]),
                     dua=it["dua"], offset=off)
            page[it["id"]] = m
        res[name] = page
        (OUT / f"{name}.json").write_text(json.dumps({"page": page}, ensure_ascii=False))
    both = set.intersection(*(set(r) for r in res.values())) if res else set()
    for name, page in res.items():
        print(f"\n== {name} (real page, headless Chrome): {len(both)} items played by every variant")
        print(bench.table(bench.grid_rows({k: v for k, v in page.items() if k in both})))


if __name__ == "__main__":
    main()


