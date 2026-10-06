#!/usr/bin/env python
"""The one recitation the explainer (round 8) follows from start to end, and the phone's models on it.

Abu Thar Al-Halawaji (a DuaPlayer reciter the models never trained on) reads Du'a al-Iftitah from the
start. The reading is built like a scenario-bench item (bench.Prog): lines 1-6, a pause, lines 7-9,
someone talking, lines 10-12, back to line 10, on to line 16, then quiet. Then the page's models run
on it as `bench.py asr` runs them (Whisper on 6 s windows every second, the stop detector, the CTC
student on 2 s windows every 0.1 s), into the ignored cache: the recording is DuaPlayer's.

    python docs/anim/recitation.py             # -> data/cache/media/explainer8/
    python docs/anim/recitation.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "data" / "cache" / "media" / "explainer8"
SOURCE = "studio:5cff8349-fa2f-49b8-b0ac-08daf2cbe0a2:0"  # Abu Thar Al-Halawaji, Du'a al-Iftitah from line 1
LEAD_IN = 1.5  # s of room tone before he starts (the page is listening, nothing on it yet)


def build(ix) -> dict:
    import bench

    src = next(s for s in bench.load_sources() if s["sid"] == SOURCE)
    src = bench.fresh_source(src, ix)
    by = {ln["seg"]: ln for ln in src["lines"]}
    lines = lambda a, b: [by[k] for k in range(a, b + 1)]  # noqa: E731
    rng = random.Random(8)
    talk = [c for c in bench._talk_clips() if 4.5 <= c["dur"] <= 7.0]
    clip = rng.choice(talk)

    p = bench.Prog(src, ix)
    p.tone(LEAD_IN, "lead")
    p.play(lines(1, 6), event="start")
    p.tone(6.0)  # he stops
    p.play(lines(7, 9))
    p.tone(0.5, "talk")  # someone talks to him
    p.clip("talk", {"clip": clip["clip"], "dur": round(clip["dur"], 3)}, clip["dur"], -2.0)
    p.tone(0.5, "talk")
    p.play(lines(10, 12))
    p.tone(0.8, "breath")
    p.play(lines(10, 16), event="back")  # back two lines
    p.tone(4.0, "end")
    return p.item("explainer", extra={"seed": 8})


def models(it: dict, y: np.ndarray, device: str) -> None:
    import bench
    from dua_recognition.asr import _quiet, _tail_activity, transcribe_batch
    from dua_recognition.ctc_student import load_ctc

    SR = bench.SR
    if not (OUT / "quiet.npz").exists():
        qt, qv = bench.quiet_track(y)
        np.savez(OUT / "quiet.npz", t=qt, q=qv)
    if not (OUT / "asr.json").exists():
        times = list(range(1, int(len(y) / SR) + 1))
        wins = [y[int(max(0, s - bench.WINDOW) * SR) : int(s * SR)] for s in times]
        texts = []
        for b0 in range(0, len(wins), 16):
            texts += transcribe_batch(wins[b0 : b0 + 16], model=str(ROOT / "models" / f"{bench.ASR_TAG}-ct2"))
        quiet = []
        for w in wins:
            probs, db, floor = _tail_activity(w, 3.0)
            quiet.append(3.0 if probs is None else round(_quiet(probs, db, floor, 6.0), 3))
        (OUT / "asr.json").write_text(json.dumps({"rows": [[s, x] for s, x in zip(times, texts)], "quiet": quiet},
                                                 ensure_ascii=False), encoding="utf-8")
        print("whisper done", flush=True)
    if not (OUT / "ctc.npz").exists():
        ctc = load_ctc(str(ROOT / "models" / bench.CTC_TAG))
        end = len(y) / SR
        times = [round((j + 1) * bench.CTC_HOP, 3) for j in range(int((end + 1e-6) // bench.CTC_HOP))]
        arr, lens, _, _ = ctc.windows(y, times, bench.CTC_WINDOW, 32)
        np.savez(OUT / "ctc.npz", lp=arr, n_frames=lens, t=np.array(times))
        print("ctc done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.environ.setdefault("DUA_ASR_DEVICE", args.device)

    import soundfile as sf

    import bench
    import evaluate as ev

    ix = ev.CorpusIndex(ev.load_all())
    OUT.mkdir(parents=True, exist_ok=True)
    it = build(ix)
    old = OUT / "item.json"
    if old.exists() and json.loads(old.read_text(encoding="utf-8"))["ops"] != it["ops"]:
        for f in ("asr.json", "ctc.npz", "quiet.npz"):
            (OUT / f).unlink(missing_ok=True)
    old.write_text(json.dumps(it, ensure_ascii=False), encoding="utf-8")
    y = bench.render(it)
    sf.write(OUT / "recitation.wav", y, bench.SR, subtype="PCM_16")
    print(f"{len(y) / bench.SR:.1f} s; events {it['events']}; still {it['still']}", flush=True)
    models(it, y, args.device)


if __name__ == "__main__":
    main()
