#!/usr/bin/env python
"""The reading the explainer follows, the phone's models on it, and what the page decided at every step.

Hussein Ghareeb (a DuaPlayer reciter the models never trained on) reads Du'a Kumayl from its first line.
The reading is built like a scenario-bench item (bench.Prog) from his recording's forced-aligned words:
lines 1-5, a pause, lines 6-7, someone talking, lines 8-9, a salawat, back two lines to 7, on to 10, quiet.
The page's models run on it as `bench.py asr` runs them: Whisper (whisper-base-ph-kd2) on 6 s windows every
second, the stop detector, the streaming CTC student every 0.1 s. Then the page is replayed at a phone's
pace (Whisper every 2 s, shown 1.8 s late; a CTC step every 0.1 s, shown 0.1 s late): the tracker over the
Whisper rows, the stream decoder over the CTC frames, as `bench.py score --display stream` does.

The audio and model outputs go to the ignored cache (the recording is DuaPlayer's); what the scenes draw
goes to docs/explainer/data/ (numbers only).

    python docs/explainer/reading.py              # CPU is enough: ~2 min of audio
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DUA_ASR_DEVICE", "cpu")
os.environ.setdefault("DUA_BENCH_ASR_EVERY", "2")  # the phone's pace (bench reads it at import)

CACHE = ROOT / "data" / "cache" / "media" / "explainer10"
DATA = HERE / "data"
AUDIO_ID = "4e825159-2053-4d90-8718-08d9bf717fd8"  # Hussein Ghareeb, Du'a Kumayl (DuaPlayer)
DUA = "dua-kumayl"
DELAY, F_DELAY = 1.8, 0.1  # the phone's: Whisper shown 1.8 s late, a CTC step 0.1 s late
LEAD_IN = 1.5


def source(ix) -> dict:
    """His recording as a bench source: whole lines with every word timed, from line 1."""
    import bench
    import evaluate as ev

    ev.use_source("duaplayer", "all")
    duas = ev.load_all()
    rec = next(r for r in ev.load_recordings(duas[DUA]) if r.audio_id == AUDIO_ID)
    words = sorted(ev.load_word_truth(rec, ix) or [], key=lambda w: w[1])
    runs = bench._line_runs(words, ix)
    run = next(r for r in runs if r[0]["seg"] == 1)
    print("lines with every word timed from line 1:", [ln["seg"] for ln in run], flush=True)
    return {"sid": f"studio:{AUDIO_ID}:explainer", "lane": "studio", "audio": str(rec.path), "dua": DUA,
            "voice": "studio:Hussein Ghareeb", "lines": run, "tags": {"condition": rec.condition}}


def build(ix) -> dict:
    import bench

    src = source(ix)
    by = {ln["seg"]: ln for ln in src["lines"]}
    lines = lambda a, b: [by[k] for k in range(a, b + 1)]  # noqa: E731
    rng = random.Random(10)
    talk = [c for c in bench._talk_clips() if 4.5 <= c["dur"] <= 6.5]
    clip = rng.choice(talk)
    salawat = bench._salawat_clips(ix, bench.load_sources())
    own = [c for c in salawat if c["voice"] == src["voice"]] or salawat
    sal = own[0]

    p = bench.Prog(src, ix)
    p.tone(LEAD_IN, "lead")
    p.play(lines(1, 5), event="start")
    p.tone(4.0)  # he stops
    p.play(lines(6, 7))
    p.tone(0.5, "talk")  # someone talks to him
    p.clip("talk", {"clip": clip["clip"], "dur": round(clip["dur"], 3)}, clip["dur"], -2.0)
    p.tone(0.5, "talk")
    p.play(lines(8, 9))
    p.tone(0.4, "salawat")
    p.clip("salawat", {"audio": sal["audio"], "from": sal["from"], "to": sal["to"]}, sal["to"] - sal["from"], 0.0)
    p.tone(0.4, "salawat")
    p.play(lines(7, 10), event="back")  # back two lines
    p.tone(3.0, "end")
    return p.item("explainer10", extra={"seed": 10, "salawat_voice": sal["voice"]})


def models(y: np.ndarray) -> None:
    import bench
    from dua_recognition.asr import _quiet, _tail_activity, transcribe_batch
    from dua_recognition.ctc_student import load_ctc

    SR = bench.SR
    if not (CACHE / "quiet.npz").exists():
        qt, qv = bench.quiet_track(y)
        np.savez(CACHE / "quiet.npz", t=qt, q=qv)
    if not (CACHE / "asr.json").exists():
        times = list(range(1, int(len(y) / SR) + 1))
        wins = [y[int(max(0, s - bench.WINDOW) * SR) : int(s * SR)] for s in times]
        texts = []
        for b0 in range(0, len(wins), 16):
            texts += transcribe_batch(wins[b0 : b0 + 16], model=str(ROOT / "models" / f"{bench.ASR_TAG}-ct2"))
        quiet = []
        for w in wins:
            probs, db, floor = _tail_activity(w, 3.0)
            quiet.append(3.0 if probs is None else round(_quiet(probs, db, floor, 6.0), 3))
        (CACHE / "asr.json").write_text(json.dumps({"rows": [[s, x] for s, x in zip(times, texts)], "quiet": quiet},
                                                   ensure_ascii=False), encoding="utf-8")
        print("whisper done", flush=True)
    if not (CACHE / "ctc.npz").exists():
        ctc = load_ctc(str(ROOT / "models" / bench.CTC_TAG))
        end = len(y) / SR
        times = [round((j + 1) * bench.CTC_HOP, 3) for j in range(int((end + 1e-6) // bench.CTC_HOP))]
        c = ctc.compact(y, times)
        np.savez(CACHE / "ctc.npz", t=np.array(times), **c)
        print("ctc done", flush=True)


def main() -> None:
    import soundfile as sf

    import bench
    import evaluate as ev

    ix = ev.CorpusIndex(ev.load_all())
    CACHE.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    it = build(ix)
    old = CACHE / "item.json"
    if old.exists() and json.loads(old.read_text(encoding="utf-8"))["ops"] != it["ops"]:
        for f in ("asr.json", "ctc.npz", "quiet.npz"):
            (CACHE / f).unlink(missing_ok=True)
    old.write_text(json.dumps(it, ensure_ascii=False), encoding="utf-8")
    y = bench.render(it)
    sf.write(CACHE / "reading.wav", y, bench.SR, subtype="PCM_16")
    print(f"{len(y) / bench.SR:.1f} s; events {it['events']}; still {it['still']}", flush=True)
    models(y)


if __name__ == "__main__":
    main()
