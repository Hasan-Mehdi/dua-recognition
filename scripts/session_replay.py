#!/usr/bin/env python
"""Replay debug sessions under tracker and display changes, and count what the
listener would have seen: the line on screen, steps back, flickers, taps undone,
and updates spent holding a stale line.

scripts/session_report.py --score looks at one session; this compares variants
across many. Every variant re-runs the Python tracker (parity-tested with
tracker.js) over each session's own hops: its transcripts, delays, pauses and
taps. Taps stay where they were made, although a different display might have
drawn different taps. The reference is session_report's, fine-tuned turbo plus
the offline smoother: automatic, not human truth. It needs the GPU once per
session, then it's cached.

    python scripts/session_replay.py                     # device sessions that found a du'a
    python scripts/session_replay.py bcet s4pv -v "lead0: display_lead=0" -v "pin: seek_pins_line=true"
    python scripts/session_replay.py --asr models/whisper-small-dua-v2-ct2 --prompt

A variant is "name: k=v k=v ...", each k a TrackerConfig field, v JSON
(true, 2, 0.6, null), on top of today's TrackerConfig defaults. Two rows always
run first: "as recorded" (the settings the page logged, so it shows what the
listener saw) and "defaults" (today's settings).

--asr re-transcribes the windows of the hops that produced text on the page
(hops the speech gate skipped stay empty), --prompt with the tracker's
reference text before its position, as the server does.

Columns:
- line / ±1: the line on screen (the page keeps the last one while nothing is
  confident) against the reference, where the reference places the du'a;
- held: updates where nothing was confident and the page kept the last line;
- back, flicker: moves to an earlier line; A -> B -> A within three updates;
- taps undone: taps after which the screen left the tapped line within 2.5 s;
- CER: --asr only, the window transcripts against the reference's.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from session_report import REF_MODEL, SESSIONS, SR, find, read_session, reference, reference_lines, retrack  # noqa: E402

def parse_variant(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    over = {}
    for kv in rest.split():
        k, _, v = kv.partition("=")
        over[k] = json.loads(v) if v != "inf" else float("inf")
    return name.strip(), over


def stability(log: dict, shown: list, segs: np.ndarray, ok: np.ndarray, dua: str) -> dict:
    hops = [e for e in log["events"] if e["type"] == "hop"]
    taps = [e for e in log["events"] if e["type"] == "seek"]
    visible, rows, k = None, [], 0
    for e in log["events"]:
        if e["type"] == "seek":
            visible = (e["dua"], e["seg"])
        elif e["type"] == "hop":
            d, s = shown[k]
            k += 1
            held = d is None and visible is not None
            if d is not None:
                visible = (d, s)
            rows.append((e["t"], held, visible[1] if visible and visible[0] == dua else None))
    n = right = near = 0
    for t, _, s in rows:
        i = min(len(segs) - 1, max(0, round(t)))
        if not ok[i] or s is None:
            continue
        n += 1
        right += s == int(segs[i])
        near += abs(s - int(segs[i])) <= 1
    lines = [s for _, _, s in rows if s is not None]
    step = np.diff(lines) if len(lines) > 1 else np.zeros(0)
    undone = sum(any(s is not None and s != tp["seg"] for t, _, s in rows if tp["t"] < t <= tp["t"] + 2.5)
                 for tp in taps)
    return {"n": n, "line": right / max(1, n), "near": near / max(1, n),
            "held": sum(h for _, h, _ in rows) / max(1, len(hops)), "back": int((step < 0).sum()),
            "flicker": sum(lines[i] != lines[i + 1] and lines[i + 2] == lines[i] for i in range(len(lines) - 2)),
            "undone": undone, "taps": len(taps)}


def cer(ref: str, hyp: str) -> tuple[int, int]:
    import jiwer

    from dua_recognition.text import normalize

    r, h = normalize(ref).replace(" ", ""), normalize(hyp).replace(" ", "")
    return (round(jiwer.cer(r, h or "-") * len(r)), len(r)) if r else (0, 0)


def fmt(m: dict) -> str:
    out = (f"line {m['line']:5.1%}  ±1 {m['near']:5.1%}  held {m['held']:5.1%}  back {m['back']:3d}  "
           f"flicker {m['flicker']:3d}  taps undone {m['undone']}/{m['taps']}")
    return out + (f"  CER {m['cer']:5.1%}" if m.get("cer") is not None else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="session ids or parts of them (default: device sessions that found a du'a)")
    ap.add_argument("--variant", "-v", action="append", default=[], metavar='"NAME: k=v ..."')
    ap.add_argument("--all", action="store_true", help="server sessions too")
    ap.add_argument("--dua", help="the du'a recited (default: the one each session showed most)")
    ap.add_argument("--asr", help="re-transcribe the page's hop windows with this model")
    ap.add_argument("--prompt", action="store_true", help="--asr: prompt with the tracker's reference text")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from dua_recognition.align import CorpusIndex
    from dua_recognition.corpus import load_all

    duas = load_all()
    index = CorpusIndex(duas)
    paths = [p for ref in args.sessions for p in find(ref)] if args.sessions else sorted(SESSIONS.glob("*.wav"))
    variants = [("as recorded", None), ("defaults", {})] + [parse_variant(v) for v in args.variant]
    totals: dict[str, list[dict]] = {name: [] for name, _ in variants}
    for path in paths:
        audio, log = read_session(path)
        found = [e["dua"] for e in log["events"] if e["type"] == "dua"]
        dua = args.dua or log.get("chosen") or (max(set(found), key=found.count) if found else None)
        hops = [e for e in log["events"] if e["type"] == "hop"]
        if dua is None or len(hops) < 10 or (log.get("engine") != "device" and not (args.all or args.sessions)):
            continue
        _, ref_texts = reference(path.stem, audio, str(ROOT / REF_MODEL))
        segs, ok = reference_lines(duas, dua, ref_texts)
        print(f"\n{path.stem}  {log.get('engine')}  {duas[dua].name_en}  ({len(hops)} updates)")
        heard: dict[float, str] = {}

        def hear(e, tracker):
            if not e.get("text"):
                return ""
            key = e["end"]
            if key not in heard or args.prompt:
                from dua_recognition.asr import transcribe_batch

                window = audio[max(0, int((e["end"] - 6) * SR)) : int(e["end"] * SR)]
                heard[key] = transcribe_batch([window], model=args.asr,
                                              prompts=[tracker.prompt() if args.prompt else None])[0]
            return heard[key]

        for name, over in variants:
            shown = retrack(log, duas, over or {}, as_logged=over is None, hear=hear if args.asr else None,
                            index=index)
            m = stability(log, shown, segs, ok, dua)
            if args.asr:
                errs = [cer(ref_texts[min(len(ref_texts) - 1, max(0, round(end) - 1))], t)
                        for end, t in heard.items() if t]
                m["cer"] = sum(e for e, _ in errs) / max(1, sum(n for _, n in errs))
            totals[name].append(m)
            print(f"  {name:24s} {fmt(m)}")
    if sum(map(len, totals.values())) and len(next(iter(totals.values()))) > 1:
        print(f"\nall {len(next(iter(totals.values())))} sessions (rates: mean over sessions; counts: summed)")
        for name, ms in totals.items():
            pooled = {k: float(np.mean([m[k] for m in ms])) for k in ("line", "near", "held")}
            pooled |= {k: sum(m[k] for m in ms) for k in ("back", "flicker", "undone", "taps")}
            pooled["cer"] = float(np.mean([m["cer"] for m in ms])) if args.asr else None
            print(f"  {name:24s} {fmt(pooled)}")


if __name__ == "__main__":
    main()
