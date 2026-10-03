#!/usr/bin/env python
"""Does the display follow a reader who goes back and says a line again? A benchmark for repeats.

People reading along repeat: they stumble and start the line over, or say a line twice
because it moved them. Professional recordings hardly ever do, so the follower's rules
(steps back must be confirmed, the tracker pulls a follower that is behind it forward)
were never measured against it. This splices a repeat into each recording after every
third line end: a short breath of the recording's own room tone (so that the inserted
span is a whole number of seconds and the 1 s window grid is kept), then the audio of
the line just finished, again; the recording then goes on with the next line. Word
truth is the forced alignment (word_truth.py) with the repeated line's words a second
time at their new times.

The tracker's 6 s windows that overlap a splice are transcribed afresh (the rest are the
cached ones, shifted), and the CTC model's windows over the whole spliced audio are dumped,
so follow_eval.py can replay both lanes on it:

    python scripts/repeat_eval.py build --split train --tag whisper-base-syn-v5-ctx8ft \\
        --model models/whisper-base-syn-v5-ctx8ft-ct2
    python scripts/repeat_eval.py build --split train --tag whisper-base-syn-v5-ctx8ft \\
        --ctc-model models/ctc-student-base-v6
    python scripts/follow_eval.py --repeats --split train --asr whisper-base-syn-v5-ctx8ft \\
        --ctc ctc-student-base-v6 --delay 1.2 fw

The files have pause_eval.py's layout (rows, quiet, truth, pause_s = 0) plus `repeats`
([start, end, line] of each spoken repeat) and `lines` (human line starts, moved and with
the repeats' starts added), so follow_eval.py reads them with its pause-set code.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
from pause_eval import good_words, room_tone  # noqa: E402

OUT = ROOT / "data" / "cache" / "repeat"
SR = 16000
WINDOW = 6.0


def repeat_points(words, ix, every: int = 3, min_gap_s: float = 15.0) -> list[dict]:
    """Line ends that run straight into the next line, whose whole line is aligned: where to
    insert a repeat (after the line's last word), and the span of the line to say again."""
    out, n, last_t = [], 0, -1e9
    for i, (a, b) in enumerate(zip(words, words[1:])):
        if b[0] != a[0] + 1 or ix.word_segment[a[0]] == ix.word_segment[b[0]]:
            continue
        gap = b[1] - a[2]
        if not 0 <= gap < 3.0:
            continue
        n += 1
        seg = int(ix.word_segment[a[0]])
        k = i
        while k > 0 and words[k - 1][0] == words[k][0] - 1 and int(ix.word_segment[words[k - 1][0]]) == seg:
            k -= 1
        first = words[k][0]
        whole = first == ix.dua_word_span[ix.word_dua[first]][0] or int(ix.word_segment[first - 1]) != seg
        if n % every or a[2] - last_t < min_gap_s or a[2] < 10.0 or not whole or i - k < 1:
            continue
        t = a[2] + min(0.2, gap / 2)
        lo = max(words[k][1] - 0.05, words[k - 1][2] if k else 0.0)
        hi = t
        out.append({"at": round(t, 3), "from": round(lo, 3), "to": round(hi, 3), "seg": seg, "words": [k, i + 1]})
        last_t = t
    return out


def spliced(y: np.ndarray, pts: list[dict]) -> tuple[np.ndarray, list[float]]:
    """`y` with each point's line said again after it, a breath of room tone first, padded so
    each insert is whole seconds long. Returns the audio and each insert's breath length."""
    tone = room_tone(y, 2.0)
    pieces, prev, breaths = [], 0, []
    for p in pts:
        line = y[int(p["from"] * SR) : int(p["to"] * SR)]
        total = math.ceil(line.size / SR + 0.4)
        breath = total * SR - line.size
        pieces += [y[prev : int(p["at"] * SR)], tone[:breath], line]
        breaths.append(breath / SR)
        prev = int(p["at"] * SR)
    pieces.append(y[prev:])
    return np.concatenate(pieces).astype(np.float32), breaths


def build(args) -> None:
    os.environ.setdefault("DUA_ASR_DEVICE", args.device)
    from faster_whisper.audio import decode_audio

    from dua_recognition.asr import _quiet, _tail_activity, transcribe_batch

    duas = ev.load_all()
    ix = ev.CorpusIndex(duas)
    out_dir = OUT / (args.tag + ("" if args.split == "test" else f"@{args.split}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ctc = None
    for dua in duas.values():
        if dua.id in ev.LINE_LABELS_UNRELIABLE:
            continue
        for rec in ev.load_recordings(dua):
            if ev.is_test(rec.reciter) != (args.split == "test"):
                continue
            out = out_dir / f"{rec.audio_id}.json"
            words = good_words(rec, ix)
            if len(words) < 20:
                continue
            lo = ix.dua_word_span[ix.dua_ids.index(dua.id)][0]
            pts = repeat_points(words, ix)
            if not pts:
                continue
            y = decode_audio(str(rec.path), sampling_rate=SR)
            y2, breaths = spliced(y, pts)
            ins = []  # (original time, inserted length, spoken repeat start in y2)
            acc = 0.0
            for p, br in zip(pts, breaths):
                length = br + (p["to"] - p["from"])
                length = round(length)
                ins.append((p["at"], length, p["at"] + acc + br))
                acc += length

            def shift(t: float) -> float:
                return t + sum(L for at, L, _ in ins if at <= t)

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
                end = r["rows"][-1][0]
                times = [round((k + 1) * args.ctc_hop, 3) for k in range(int((end + 1e-6) // args.ctc_hop))]
                arr, lens, _, _ = ctc.windows(y2, times, args.ctc_window, 32)
                np.savez(f, lp=arr, n_frames=lens, t=np.array(times))
                print(f"  ctc {rec.audio_id} {len(times)} windows", flush=True)
                continue
            if out.exists():
                continue
            cached = ev.load_rows(args.tag, rec, WINDOW, 1.0)
            if not cached:
                print(f"  no window cache for {rec.audio_id}; skipped", file=sys.stderr)
                continue
            by_t = {round(t): x for t, x in cached}
            spans = [(shift(at) - L, shift(at)) for at, L, _ in ins]  # inserted spans in y2
            end2 = shift(min(rec.end_s, words[-1][2] + 2.0))
            times = list(range(1, int(end2) + 1))
            texts, fresh = {}, []
            for t2 in times:
                if any(a < t2 and t2 - WINDOW < b for a, b in spans):
                    fresh.append(t2)
                    continue
                k = sum(L for (a, b), (_, L, _) in zip(spans, ins) if b <= t2 - WINDOW)
                texts[t2] = by_t.get(round(t2 - k), "")
            for b in range(0, len(fresh), 16):
                chunk = fresh[b : b + 16]
                wins = [y2[int(max(0, t - WINDOW) * SR) : int(t * SR)] for t in chunk]
                for t, x in zip(chunk, transcribe_batch(wins, model=args.model)):
                    texts[t] = x
            quiet = []
            for t2 in times:
                probs, db, floor = _tail_activity(y2[int(max(0, t2 - WINDOW) * SR) : int(t2 * SR)], 3.0)
                quiet.append(3.0 if probs is None else round(_quiet(probs, db, floor, 6.0), 3))
            truth = [[w[0] - lo, round(shift(w[1]), 3), round(shift(w[2]), 3)] for w in words]
            repeats = []
            for p, (at, L, s2) in zip(pts, ins):
                d = s2 - p["from"]  # repeat time = original time + d
                k0, k1 = p["words"]
                for w in words[k0:k1]:
                    truth.append([w[0] - lo, round(w[1] + d, 3), round(min(w[2] + d, s2 + p["to"] - p["from"]), 3)])
                repeats.append([round(s2 + max(0.0, words[k0][1] - p["from"]), 3), round(s2 + p["to"] - p["from"], 3),
                                p["seg"]])
            truth.sort(key=lambda w: w[1])
            lines = [(shift(t), s) for t, s in rec.starts] + [(a, s) for a, _, s in repeats]
            rec_out = {
                "audio_id": rec.audio_id, "dua": dua.id, "reciter": rec.reciter, "pause_s": 0.0, "pauses": [],
                "repeats": repeats, "lines": sorted(lines), "rows": [[t, texts[t]] for t in times], "quiet": quiet,
                "truth": truth,
            }
            out.write_text(json.dumps(rec_out, ensure_ascii=False), encoding="utf-8")
            print(f"{dua.id:28s} {rec.reciter[:20]:20s} {len(pts):3d} repeats  {len(fresh):5d} new windows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", help="Whisper model for the windows over a splice")
    b.add_argument("--tag", required=True, help="window-cache tag of the same model")
    b.add_argument("--device", default="cuda")
    b.add_argument("--split", choices=["test", "train"], default="train")
    b.add_argument("--ctc-model", help="instead: dump this CTC model's windows over the built spliced audio")
    b.add_argument("--ctc-window", type=float, default=3.0)
    b.add_argument("--ctc-hop", type=float, default=0.2)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    build(args)


if __name__ == "__main__":
    main()
