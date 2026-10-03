#!/usr/bin/env python
"""What the page shows once the reciter stops, in the recorded debug sessions,
and what a change to the tracker, the stop detector or the word glide would show.

Hasan (2026-09-30): "fast at recognizing, but when I stopped reciting it jumped
forward 4-5 words ahead". A stop here is at least 1.5 s without speech (Silero
over the whole recording, which sees the future and has no window). The last
word said before it comes from the reference ear: fine-tuned turbo on the 6 s
window ending just after the stop, aligned within the du'a near the reference
line (session_report's offline smoother). Both are cached per session.

The page is rebuilt 20 times a second: the Python tracker (parity-tested with
tracker.js) over each update the page received, at the time it received it,
then the word glide between updates (display.py, parity-tested with display.js)
and the line on screen as app.js render() draws it. "as recorded" reproduces
the page: its word changes are checked against the logged ones.

    python scripts/session_stops.py                       # every device session with stops
    python scripts/session_stops.py h649 -v               # ...one, stop by stop
    python scripts/session_stops.py -V "fix: quiet=speech live=1"

A variant is "name: k=v ...": TrackerConfig fields as in session_replay.py (on top of
today's defaults), plus
  quiet=logged|speech|old  the window's quiet: as the page logged it, or recomputed from
                           the audio (asr.QuietMeter, or the rule before 2026-09-30)
  live=1                   the page's own stop detector (asr.LiveQuiet) on the audio it
                           holds when an update arrives; the glide stops while it hears silence
  paused=1                 the tracker gets the pauses since the last update (QuietMeter.paused)
  glide.back_after=S       an update after S s of silence may take the highlight back in its line
  glide.K=V                other display.Highlight settings

Columns, from `--settle` s after each stop (the page is ~1.2 s behind) until
speech resumes or the session ends:
- ahead: words the highlight is past the last word said (mean over that time);
- ≥2 ahead, next line: share of that time the highlight is 2+ words past it, or
  on a later line;
- worst: stops where it was, at some moment, 3+ words past it;
- behind: share of that time it is still before the last word said.
And over the whole session (session_replay's, per update, against the reference
line): line, ±1, back, flicker.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from session_report import (CACHE, REF_MODEL, SESSIONS, SR, find, logged_config, read_session,  # noqa: E402
                            reference, reference_lines)

MIN_STOP = 1.5  # seconds without speech
TICK = 0.05


# -- truth ----------------------------------------------------------------------
def speech_spans(audio: np.ndarray) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    ts = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=400, speech_pad_ms=0,
                                                 min_speech_duration_ms=150))
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


def stops_truth(sid: str, audio: np.ndarray, duas, dua: str, ref_texts: list[str]) -> list[dict]:
    """Stops (speech end, silence length) with the last word said, cached per session."""
    cache = CACHE / f"{sid}.stops.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from dua_recognition.align import CorpusIndex
    from dua_recognition.asr import transcribe_batch

    spans = speech_spans(audio)
    total = len(audio) / SR
    ix = CorpusIndex({dua: duas[dua]})
    segs, ok = reference_lines(duas, dua, ref_texts)
    out = []
    for i, (_, end) in enumerate(spans):
        nxt = spans[i + 1][0] if i + 1 < len(spans) else total
        if nxt - end < MIN_STOP:
            continue
        text = transcribe_batch([audio[max(0, int((end - 6) * SR)) : int((end + 0.3) * SR)]], model=str(ROOT / REF_MODEL))[0]
        k = min(len(segs) - 1, max(0, int(np.ceil(end))))  # the reference window ending just after the stop
        if not text or not ok[k]:
            continue
        costs = ix.word_costs(text)
        near = np.flatnonzero(np.abs(ix.word_segment - int(segs[k])) <= 1)
        w = int(near[np.argmin(costs[near])])
        out.append({"at": round(end, 3), "until": round(nxt, 3), "text": text, "word": w,
                    "seg": int(ix.word_segment[w]), "token": ix.words[w].token, "ref_seg": int(segs[k]),
                    "cost": int(costs[w]), "letters": int(ix.word_costs(text).max())})
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


# -- the page, rebuilt ---------------------------------------------------------
def audio_offset(log: dict) -> float:
    """Page time minus audio time: when a sample reached the page."""
    t0 = datetime.fromisoformat(log["started"].replace("Z", "+00:00")).timestamp() * 1000
    offs = [(e["captured_ms"] - t0) / 1000 - e["end"] for e in log["events"]
            if e["type"] == "hop" and e.get("captured_ms") and e.get("end") is not None]
    return float(np.median(offs)) if offs else 0.1


def parse_variant(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    over = {}
    for kv in rest.split():
        k, _, v = kv.partition("=")
        over[k] = v if k == "quiet" else (float("inf") if v == "inf" else json.loads(v))
    return name.strip(), over


CHUNK = 4000  # samples per chunk from the capture worklet (250 ms; sessions log "chunk" since 2026-10-01)


def renders(log: dict, audio: np.ndarray, duas, index, over: dict) -> tuple[list[dict], list[tuple[float, float]]]:
    """What app.js render() received, in order (one per update and per tap), under
    today's defaults plus `over`; and the page's live quiet after every chunk of
    audio, as (page time, seconds silent)."""
    from dua_recognition import asr
    from dua_recognition.align import CorpusIndex
    from dua_recognition.tracker import RECITER, Tracker, TrackerConfig

    over = dict(over)
    quiet_rule = over.pop("quiet", "logged")
    rel = over.pop("rel", asr.QUIET_REL_DB)
    live = bool(over.pop("live", 0))
    live_rel = over.pop("live_rel", asr.LIVE_REL_DB)
    use_paused = bool(over.pop("paused", 0))
    over = {k: v for k, v in over.items() if not k.startswith("glide.")}
    base = replace(TrackerConfig(), **over)
    cfg = lambda mode: replace(base, **RECITER) if mode == "reciter" else base  # noqa: E731
    mode = log.get("follow", "reading")
    tracker = Tracker(CorpusIndex({log["chosen"]: duas[log["chosen"]]}) if log.get("chosen") else index, cfg(mode))
    logged_lead = (log.get("tracker") or {}).get("displayLead", base.display_lead)
    off = audio_offset(log)
    meter = asr.QuietMeter(rel_db=None if quiet_rule == "old" else rel)
    listener = asr.LiveQuiet(rel_db=live_rel) if live else None
    series: list[tuple[float, float]] = []
    fed = 0
    last_end, out = 0.0, []

    chunk = int(log.get("chunk") or CHUNK)

    def hear_until(t: float) -> None:
        nonlocal fed
        while listener is not None and fed + chunk <= len(audio) and (fed + chunk) / SR + off <= t:
            listener.push(audio[fed : fed + chunk])
            fed += chunk
            if listener.quiet is not None:
                series.append((fed / SR + off, listener.quiet))

    for e in log["events"]:
        t = e["ms"] / 1000
        hear_until(t)
        if e["type"] == "lock":
            tracker = Tracker(CorpusIndex({e["dua"]: duas[e["dua"]]}), cfg(mode))
        elif e["type"] == "follow":
            mode = e["mode"]
            tracker.cfg = cfg(mode)
        elif e["type"] == "seek":
            p = tracker.seek(e["dua"], e["seg"])
            out.append({"t": t, "dua": e["dua"], "seg": e["seg"], "token": _token(tracker, p), "speed": 0.0, "end": None,
                        "anchor": p.word if p is not None and p.dua else None, "ix": tracker.ix, "seek": True})
        elif e["type"] == "hop":
            dt = e.get("dt", e["end"] - last_end)
            last_end = e["end"]
            if "lead" in e:
                lead = e["lead"] + tracker.cfg.display_lead - logged_lead if e["lead"] > 0 else 0.0
            else:
                lead = t - e["end"] + tracker.cfg.display_lead
            window = audio[max(0, int((e["end"] - 6) * SR)) : int(e["end"] * SR)]
            quiet = meter(window, dt)  # also keeps the voice level, which the page's detector uses
            if quiet_rule == "logged":
                quiet = e.get("quiet") or 0.0
            quiet_now = None
            if listener is not None:
                listener.voice_db = meter.voice_db
                quiet_now = listener.quiet
            p = tracker.update(e.get("text") or "", dt, lead=max(0.0, lead), quiet=quiet, quiet_now=quiet_now,
                               paused=meter.paused if use_paused else None)
            still = quiet > tracker.cfg.still_after or (quiet_now is not None and quiet_now > tracker.cfg.still_after)
            now = tracker.position()  # the evidence position: the word follower's anchor
            out.append({"t": t, "dua": p.dua, "seg": p.segment, "token": _token(tracker, p),
                        "speed": 0.0 if still else tracker.speed, "end": e["end"], "quiet": quiet,
                        "quiet_now": quiet_now, "anchor": now.word if now.dua else None, "ix": tracker.ix})
    hear_until(len(audio) / SR + off)
    return out, series


def recorded(log: dict) -> tuple[list[dict], list[tuple[float, float]]]:
    """render() as the page called it, from the log, and its stop detector as it
    stopped and restarted the glide ("hush" events, sessions since 2026-10-01)."""
    out = []
    for e in log["events"]:
        if e["type"] == "hop":
            silent = max(e.get("quiet") or 0.0, e.get("quiet_now") or 0.0)
            out.append({"t": e["ms"] / 1000, "dua": e.get("dua"), "seg": e.get("seg"), "token": e.get("token"),
                        "speed": 0.0 if silent > 0.3 else e.get("speed", 1.0), "end": e.get("end"),
                        "quiet": e.get("quiet") or 0.0, "quiet_now": e.get("quiet_now")})
        elif e["type"] == "seek":
            out.append({"t": e["ms"] / 1000, "dua": e["dua"], "seg": e["seg"], "token": 0, "speed": 0.0, "end": None})
    hush = [(e["ms"] / 1000, 1.0 if e["on"] else 0.0) for e in log["events"] if e["type"] == "hush"]
    return out, hush


def _token(tracker, p) -> int | None:
    return tracker.ix.words[p.word].token if p.word is not None and p.dua is not None else None


def display(rs: list[dict], duas, until: float, glide: dict, live: list[tuple[float, float]] | None = None,
            still_after: float = 0.3, back_after: float | None = None):
    """The line and word on screen every TICK s, as app.js render() and glide() draw them.
    live: the page's stop detector, (page time, seconds silent) as it changes; the glide
    stops while it says they are silent (None: no detector, as before)."""
    from dua_recognition.display import Highlight

    hl = Highlight(**glide)
    dua = seg = None
    speed = 0.0
    out, k, j = [], 0, 0
    stopped = False
    for t in np.arange(rs[0]["t"] if rs else 0.0, until, TICK):
        while live and j < len(live) and live[j][0] <= t:
            now_still = live[j][1] > still_after
            j += 1
            if now_still != stopped:
                stopped = now_still
                hl.pace(t, 0.0 if stopped else speed)
        while k < len(rs) and rs[k]["t"] <= t:
            r = rs[k]
            k += 1
            if r["dua"] is None:
                continue  # the page holds the last place through a lapse
            if r["dua"] != dua:
                dua, seg = r["dua"], None
                hl = Highlight(**glide)
            seg = r["seg"]
            n = len(next(s for s in duas[dua].segments if s.id == seg).arabic.split())
            speed = r["speed"]
            if r["token"] is None:
                continue
            silent = max(r.get("quiet") or 0.0, r.get("quiet_now") or 0.0)
            hl.update(r["t"], dua, seg, r["token"], (0, n), 0.0 if stopped else speed,
                      back=back_after is not None and silent > back_after)
        out.append((t, dua, seg, hl.word(t) if hl.line is not None and (dua, seg) == hl.line else None))
    return out


# -- scoring --------------------------------------------------------------------
def word_index(ix, dua: str, seg: int, token: int) -> int | None:
    d = ix.dua_ids.index(dua)
    lo, hi = ix.dua_word_span[d]
    for w in range(lo, hi):
        if ix.word_segment[w] == seg and ix.words[w].token >= token:
            return w
    return None


def stop_scores(shown, stops, ix, dua: str, off: float, settle: float) -> list[dict]:
    d = ix.dua_ids.index(dua)
    lo = ix.dua_word_span[d][0]
    out = []
    for s in stops:
        truth = lo + s["word"]
        a, b = s["at"] + off + settle, s["until"] + off
        rows = [(t, x) for t, x in ((t, x) for t, *x in shown) if a <= t < b]
        ahead, nxt = [], []
        for t, (sd, sseg, tok) in rows:
            if sd != dua or sseg is None:
                continue
            w = word_index(ix, dua, sseg, tok if tok is not None else 0)
            if w is None:
                continue
            ahead.append(w - truth)
            nxt.append(sseg > s["seg"])
        if not ahead:
            continue
        x = np.array(ahead)
        out.append({"at": s["at"], "len": b - a, "mean": float(np.clip(x, -10, 10).mean()),
                    "ahead2": float(np.mean(x >= 2)), "next": float(np.mean(nxt)), "worst": int(x.max()),
                    "behind": float(np.mean(x < 0)), "final": int(x[-1]), "n": len(x)})
    return out


def line_scores(rs: list[dict], segs: np.ndarray, ok: np.ndarray, dua: str, spans, off: float) -> dict:
    """session_replay's per-update line accuracy, only while the reciter is reciting:
    the reference (turbo windows, offline smoother) keeps moving through a stop on
    the last words still in its windows, the very bias this is about. Plus the
    line moves back and flickers over the whole session."""
    n = right = near = 0
    lines, visible = [], None
    for r in rs:
        if r["dua"] is not None:
            visible = (r["dua"], r["seg"])
        s = visible[1] if visible and visible[0] == dua else None
        if r["end"] is None:  # a tap
            continue
        lines.append(s)
        a = r["t"] - off
        i = min(len(segs) - 1, max(0, round(r["t"])))
        if s is None or not ok[i] or not any(s0 <= a <= e + 1.0 for s0, e in spans):
            continue
        n += 1
        right += s == int(segs[i])
        near += abs(s - int(segs[i])) <= 1
    ls = [x for x in lines if x is not None]
    step = np.diff(ls) if len(ls) > 1 else np.zeros(0)
    return {"n": n, "line": right / max(1, n), "near": near / max(1, n), "back": int((step < 0).sum()),
            "flicker": sum(ls[i] != ls[i + 1] and ls[i + 2] == ls[i] for i in range(len(ls) - 2))}


def check_words(log: dict, shown, off: float) -> float:
    """Share of the page's logged word changes the rebuild reproduces (same word,
    within 0.3 s)."""
    words = [(e["ms"] / 1000, e["token"]) for e in log["events"] if e["type"] == "word"]
    if not words:
        return float("nan")
    ts = np.array([t for t, *_ in shown])
    hit = 0
    for t, tok in words:
        i = np.searchsorted(ts, [t - 0.3, t + 0.3])
        hit += any(x[3] == tok for x in shown[i[0] : i[1] + 1])
    return hit / len(words)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--variant", "-V", action="append", default=[], metavar='"NAME: k=v ..."')
    ap.add_argument("--verbose", "-v", action="store_true", help="stop by stop")
    ap.add_argument("--settle", type=float, default=1.5)
    ap.add_argument("--server", action="store_true", help="server sessions too")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from dua_recognition.align import CorpusIndex
    from dua_recognition.corpus import load_all

    duas = load_all()
    index = CorpusIndex(duas)
    paths = [p for ref in args.sessions for p in find(ref)] if args.sessions else sorted(SESSIONS.glob("*.wav"))
    variants = [("as recorded", None), ("defaults", {})] + [parse_variant(v) for v in args.variant]
    totals: dict[str, list] = {name: [] for name, _ in variants}
    lines_tot: dict[str, list] = {name: [] for name, _ in variants}
    for path in paths:
        audio, log = read_session(path)
        found = [e["dua"] for e in log["events"] if e["type"] == "dua"]
        dua = log.get("chosen") or (max(set(found), key=found.count) if found else None)
        hops = [e for e in log["events"] if e["type"] == "hop"]
        if dua is None or len(hops) < 10 or len(audio) > 600 * SR:
            continue
        if log.get("engine") != "device" and not (args.server or args.sessions):
            continue
        _, ref_texts = reference(path.stem, audio, str(ROOT / REF_MODEL))
        stops = stops_truth(path.stem, audio, duas, dua, ref_texts)
        spans = speech_spans(audio)
        segs, ok = reference_lines(duas, dua, ref_texts)
        off = audio_offset(log) if log.get("engine") == "device" else 0.1
        until = len(audio) / SR + off
        print(f"\n{path.stem}  {log.get('engine')}  {duas[dua].name_en}  ({len(hops)} updates, {len(stops)} stops)")
        for name, over in variants:
            glide = {k[6:]: v for k, v in (over or {}).items() if k.startswith("glide.")}
            back_after = glide.pop("back_after", None)
            if over is None:
                rs, live = recorded(log)
                back_after = 1.0 if live else None  # the page as it is since 2026-10-01
            else:
                rs, live = renders(log, audio, duas, index, over)
            shown = display(rs, duas, until, glide, live or None, back_after=back_after)
            sc = stop_scores(shown, stops, index, dua, off, args.settle)
            totals[name] += sc
            m = line_scores(rs, segs, ok, dua, spans, off)
            lines_tot[name].append(m)
            extra = f"  (rebuild matches {check_words(log, shown, off):.0%} of logged word changes)" if over is None else ""
            print(f"  {name:22s} {fmt(sc)}  | line {m['line']:5.1%} ±1 {m['near']:5.1%} back {m['back']} "
                  f"flicker {m['flicker']}{extra}")
            if args.verbose:
                for s in sc:
                    print(f"      stop at {s['at']:5.1f} s ({s['len']:4.1f} s): ahead {s['mean']:+5.2f}  "
                          f"worst {s['worst']:+d}  final {s['final']:+d}  next line {s['next']:4.0%}")
    print(f"\nall sessions: {len(totals['as recorded'])} stops")
    for name, _ in variants:
        ms = lines_tot[name]
        print(f"  {name:22s} {fmt(totals[name])}  | line {np.mean([m['line'] for m in ms]):5.1%} "
              f"±1 {np.mean([m['near'] for m in ms]):5.1%} back {sum(m['back'] for m in ms)} "
              f"flicker {sum(m['flicker'] for m in ms)}")


def fmt(sc: list[dict]) -> str:
    if not sc:
        return "(no stops)"
    w = np.array([s["n"] for s in sc], float)
    avg = lambda k: float(np.average([s[k] for s in sc], weights=w))  # noqa: E731
    return (f"ahead {avg('mean'):+5.2f}  ≥2 ahead {avg('ahead2'):5.1%}  next line {avg('next'):5.1%}  "
            f"behind {avg('behind'):5.1%}  worst≥3 {sum(s['worst'] >= 3 for s in sc)}/{len(sc)}")


if __name__ == "__main__":
    main()
