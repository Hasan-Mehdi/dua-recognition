#!/usr/bin/env python
"""Does the display find a reader who jumps around the du'a? A benchmark for lines out of order.

People don't always read a du'a front to back: they skip to the part they know, go back to a
line they like, pick it up again after an interruption somewhere else. This rebuilds each
recording as a reading that jumps: its first three lines in order (so the du'a is found), then
a jump to a random other line of the same du'a, one to three lines read on from there, a jump
again, and so on, with a breath of the recording's own room tone (0.3-1.2 s) before each jump.
Lines are cut at their forced-aligned word boundaries (scripts/word_truth.py), so a line is used
only if all of its words are aligned. Jumps go back and forward, near and far; never to the
line that comes next anyway.

Everything is new audio, so every 6 s tracker window is transcribed with the phone's Whisper,
and the CTC model's windows are dumped over it; follow_eval.py --jumps replays both lanes:

    python scripts/jump_eval.py build --split train --model models/whisper-base-syn-v5-ctx8ft-ct2
    python scripts/jump_eval.py build --split train --ctc-model models/ctc-student-base-v6 --ctc-window 2 --ctc-hop 0.1
    python scripts/follow_eval.py --jumps --split train --asr whisper-base-syn-v5-ctx8ft \\
        --ctc ctc-student-base-v6 --ctc-window 2 --ctc-hop 0.1 --delay 1.2 --follow-delay 0.15 fw

The files have pause_eval.py's layout (rows, quiet, truth, pause_s 0) plus `lines` (the start
of every line read, in the new time), `jumps` ([start, line jumped to, line jumped from]) and
`program` (the source spans and breaths, enough to rebuild the audio: build_audio()).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
from pause_eval import good_words, room_tone  # noqa: E402

OUT = ROOT / "data" / "cache" / "jumps"
SR = 16000
WINDOW = 6.0
TAG = "whisper-base-syn-v5-ctx8ft"


def whole_lines(words, ix) -> list[dict]:
    """Lines whose every word is aligned, in order: segment id, word list (truth rows), span."""
    by_seg: dict[int, list] = {}
    for w in words:
        by_seg.setdefault(int(ix.word_segment[w[0]]), []).append(w)
    out = []
    for seg, ws in by_seg.items():
        ws = sorted(ws, key=lambda w: w[0])
        first, last = ws[0][0], ws[-1][0]
        n_line = int(np.sum(ix.word_segment[ix.dua_word_span[ix.word_dua[first]][0]:ix.dua_word_span[ix.word_dua[first]][1]] == seg))
        if last - first + 1 != len(ws) or len(ws) != n_line or len(ws) < 2:
            continue
        a, b = ws[0][1] - 0.05, ws[-1][2] + 0.1
        if b - a > 15.0:
            continue
        out.append({"seg": seg, "words": ws, "from": a, "to": b})
    out.sort(key=lambda l: l["words"][0][0])
    return out


def make_program(lines: list[dict], rng: random.Random, max_s: float = 150.0) -> list[dict]:
    """Runs of consecutive lines: [{lines: [index into `lines`], breath: s before it}]. The first
    run is the opening three lines in order; each later one starts somewhere else."""
    prog, total = [], 0.0

    def run_from(i: int, n: int) -> list[int]:
        out = [i]
        while len(out) < n and out[-1] + 1 < len(lines) and lines[out[-1] + 1]["seg"] == lines[out[-1]]["seg"] + 1:
            out.append(out[-1] + 1)
        return out

    first = run_from(0, 3)
    prog.append({"lines": first, "breath": 0.0})
    total += sum(lines[i]["to"] - lines[i]["from"] for i in first)
    cur = first[-1]
    while total < max_s:
        choices = [i for i in range(len(lines)) if i not in (cur, cur + 1)]
        if not choices:
            break
        i = rng.choice(choices)
        run = run_from(i, rng.choice([1, 2, 2, 3]))
        prog.append({"lines": run, "breath": round(rng.uniform(0.3, 1.2), 2)})
        total += sum(lines[k]["to"] - lines[k]["from"] for k in run) + prog[-1]["breath"]
        cur = run[-1]
    return prog


def build_audio(y: np.ndarray, spans: list[tuple[float, float, float]]) -> np.ndarray:
    """The jumping reading: for each (from, to, breath), `breath` s of room tone then y[from:to]."""
    tone = room_tone(y, 2.0)
    pieces = []
    for a, b, br in spans:
        pieces += [tone[: int(br * SR)], y[int(a * SR) : int(b * SR)]]
    pieces.append(tone[: int(1.5 * SR)])  # a breath at the end: the reader stops
    return np.concatenate(pieces).astype(np.float32)


def build(args) -> None:
    os.environ.setdefault("DUA_ASR_DEVICE", args.device)
    from faster_whisper.audio import decode_audio

    from dua_recognition.asr import _quiet, _tail_activity, transcribe_batch

    duas = ev.load_all()
    ix = ev.CorpusIndex(duas)
    out_dir = OUT / (TAG + ("" if args.split == "test" else f"@{args.split}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ctc = None
    for dua in duas.values():
        if dua.id in ev.LINE_LABELS_UNRELIABLE:
            continue
        for rec in ev.load_recordings(dua):
            if ev.is_test(rec.reciter) != (args.split == "test"):
                continue
            out = out_dir / f"{rec.audio_id}.json"
            if args.ctc_model:
                if not out.exists():
                    continue
                cdir = out_dir / f"ctc-{Path(args.ctc_model).name}"
                cdir.mkdir(exist_ok=True)
                f = cdir / f"{rec.audio_id}_w{args.ctc_window:g}_h{args.ctc_hop:g}.npz"
                if f.exists():
                    continue
                ctc = ctc or __import__("dua_recognition.ctc_student", fromlist=["load_ctc"]).load_ctc(args.ctc_model)
                r = json.loads(out.read_text(encoding="utf-8"))
                y2 = build_audio(decode_audio(str(rec.path), sampling_rate=SR), [tuple(p) for p in r["program"]])
                end = r["rows"][-1][0]
                times = [round((k + 1) * args.ctc_hop, 3) for k in range(int((end + 1e-6) // args.ctc_hop))]
                arr, lens, _, _ = ctc.windows(y2, times, args.ctc_window, 32)
                np.savez(f, lp=arr, n_frames=lens, t=np.array(times))
                print(f"  ctc {rec.audio_id} {len(times)} windows", flush=True)
                continue
            if out.exists():
                continue
            words = good_words(rec, ix)
            lines = whole_lines(words, ix)
            if len(lines) < 8:
                continue
            rng = random.Random(zlib.crc32(rec.audio_id.encode()))
            prog = make_program(lines, rng)
            if len(prog) < 3:
                continue
            y = decode_audio(str(rec.path), sampling_rate=SR)
            spans, truth, starts, jumps = [], [], [], []
            lo = ix.dua_word_span[ix.dua_ids.index(dua.id)][0]
            t = 0.0
            prev_seg = None
            for k, step in enumerate(prog):
                run, breath = step["lines"], step["breath"]
                a, b = lines[run[0]]["from"], lines[run[-1]]["to"]
                spans.append((round(a, 3), round(b, 3), breath))
                t += breath
                d = t - a  # new time = source time + d
                for i in run:
                    line = lines[i]
                    starts.append([round(line["words"][0][1] + d, 3), line["seg"]])
                    for w in line["words"]:
                        truth.append([w[0] - lo, round(w[1] + d, 3), round(w[2] + d, 3)])
                if k:
                    jumps.append([round(lines[run[0]]["words"][0][1] + d, 3), lines[run[0]]["seg"], prev_seg])
                prev_seg = lines[run[-1]]["seg"]
                t += b - a
            y2 = build_audio(y, spans)
            times = list(range(1, int(len(y2) / SR) + 1))
            wins = [y2[int(max(0, s - WINDOW) * SR) : int(s * SR)] for s in times]
            texts = []
            for b0 in range(0, len(wins), 16):
                texts += transcribe_batch(wins[b0 : b0 + 16], model=args.model)
            quiet = []
            for w in wins:
                probs, db, floor = _tail_activity(w, 3.0)
                quiet.append(3.0 if probs is None else round(_quiet(probs, db, floor, 6.0), 3))
            truth.sort(key=lambda w: w[1])
            rec_out = {
                "audio_id": rec.audio_id, "dua": dua.id, "reciter": rec.reciter, "pause_s": 0.0, "pauses": [],
                "program": [list(s) for s in spans], "jumps": jumps, "lines": starts,
                "rows": [[s, x] for s, x in zip(times, texts)], "quiet": quiet, "truth": truth,
            }
            out.write_text(json.dumps(rec_out, ensure_ascii=False), encoding="utf-8")
            print(f"{dua.id:28s} {rec.reciter[:20]:20s} {len(jumps):3d} jumps  {len(y2) / SR:5.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", default="models/whisper-base-syn-v5-ctx8ft-ct2", help="Whisper model for the windows")
    b.add_argument("--device", default="cuda")
    b.add_argument("--split", choices=["test", "train"], default="train")
    b.add_argument("--ctc-model", help="instead: dump this CTC model's windows over the built audio")
    b.add_argument("--ctc-window", type=float, default=2.0)
    b.add_argument("--ctc-hop", type=float, default=0.1)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    build(args)


if __name__ == "__main__":
    main()
