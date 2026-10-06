#!/usr/bin/env python
"""Read, summarise and score the debug sessions the web app records.

Every listening session in the page (web/session-log.js) keeps the 16 kHz audio
the recognizer heard, with a log of what it made of it inside the same .wav:
each window's transcript and tracker state, each line and word shown, taps,
scrolls, the page going to the background. app/server.py saves uploads to
data/sessions/; a session sent from the phone by hand can be dropped there too.

    python scripts/session_report.py                     # newest session: summary
    python scripts/session_report.py --all               # one line per session
    python scripts/session_report.py ID --timeline       # every update, line move and tap
    python scripts/session_report.py ID --score          # against a reference (GPU)
    python scripts/session_report.py ID --score --set kappa=0.2   # ...and a tracker change
    python scripts/session_report.py ID --log            # the raw log (JSON)

ID is a file, a session id, or a unique part of one.

--score places every second of the session in the du'a with a stronger ear
than the phone's: the fine-tuned large-v3-turbo transcribes 6 s windows every
second, and the offline forward-backward smoother (offline.py), which sees the
future, turns them into a line per second. It's a reference, not human truth,
so taps ("I'm here" on a line) are checked against it too. The phone's line is
then scored once per update as scripts/evaluate.py scores against DuaPlayer's
timings, and the Python tracker is re-run over the phone's own transcripts
(the same code as tracker.js, tests/test_web_parity.py): with --set, that's
what a tracker change would have shown on this session.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

SESSIONS = ROOT / "data" / "sessions"
CACHE = SESSIONS / ".cache"
SR = 16000
REF_MODEL = "models/whisper-turbo-dua-ct2"


# -- reading --------------------------------------------------------------------
def read_session(path: Path) -> tuple[np.ndarray, dict]:
    """(audio, log) from a session .wav: PCM in "data", the log in a "json" chunk."""
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a WAV file")
    audio, log, pos = np.zeros(0, np.float32), None, 12
    while pos + 8 <= len(data):
        tag, size = data[pos : pos + 4], int.from_bytes(data[pos + 4 : pos + 8], "little")
        body = data[pos + 8 : pos + 8 + size]
        if tag == b"data":
            audio = np.frombuffer(body[: len(body) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
        elif tag == b"json":
            log = json.loads(body)
        pos += 8 + size + (size & 1)
    if log is None:
        raise ValueError(f"{path} has no session log (not recorded by web/session-log.js?)")
    return audio, log


def find(ref: str | None) -> list[Path]:
    files = sorted(SESSIONS.glob("*.wav"))
    if ref is None:
        return files[-1:]
    if Path(ref).is_file():
        return [Path(ref)]
    return [f for f in files if ref in f.stem]


def percentile(xs, q) -> float:
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


class Lines:
    """Segment ids -> line numbers as the page shows them (1-based)."""

    def __init__(self, duas):
        self.duas = duas
        self._idx = {d: {s.id: i for i, s in enumerate(dua.segments)} for d, dua in duas.items()}

    def __call__(self, dua: str | None, seg) -> str:
        if dua is None or seg is None:
            return "-"
        i = self._idx.get(dua, {}).get(seg)
        return "?" if i is None else str(i + 1)

    def name(self, dua: str | None) -> str:
        return self.duas[dua].name_en if dua in self.duas else str(dua)


# -- summary --------------------------------------------------------------------
def summary(path: Path, audio: np.ndarray, log: dict, lines: Lines) -> list[str]:
    ev = log["events"]
    hops = [e for e in ev if e["type"] == "hop"]
    kinds = lambda k: [e for e in ev if e["type"] == k]  # noqa: E731
    mic = log.get("mic") or {}
    out = [
        f"{path.name}  ({len(audio) / SR / 60:.1f} min, ended: {log.get('ended')})",
        f"  started {log.get('started')}  ·  {log.get('engine')} engine, {log.get('model')}  ·  "
        f"{log.get('follow')} mode  ·  source {log.get('source')}"
        + (f"  ·  chose {lines.name(log['chosen'])}" if log.get("chosen") else ""),
        f"  device {log.get('ua', '?')}",
        f"  mic {mic.get('label') or '?'}  {mic.get('rate') or '?'} Hz  agc={mic.get('agc')}  "
        f"echo={mic.get('echo')}  noise={mic.get('noise')}  ·  audio context {log.get('audio_rate')} Hz"
        + (f"  ·  model loaded in {log['load_ms'] / 1000:.1f} s" if log.get("load_ms") else ""),
    ]
    vis = [e for e in ev if e["type"] == "visible" and e.get("shown_ms")]
    if len(vis) > 3 and vis[-1]["shown_ms"] > vis[0]["shown_ms"]:
        # Audio seconds per real second (web/app.js checkClock): 1.00, or the microphone is broken.
        ratio = (vis[-1]["end"] - vis[0]["end"]) / ((vis[-1]["shown_ms"] - vis[0]["shown_ms"]) / 1000)
        out.append(f"  audio clock {ratio:.2f} s per real second"
                   + ("" if abs(ratio - 1) < 0.1 else "  <-- AUDIO ARRIVES AT THE WRONG SPEED: nothing after this is trustworthy"))
    if hops:
        gaps = np.diff([h["t"] for h in hops])
        delay = [h["t"] - h["end"] for h in hops if h.get("end") is not None]
        silent = [h for h in hops if not h.get("text")]
        paused = [h for h in silent if (h.get("quiet") or 0) > 0.3]
        out.append(
            f"  updates: {len(hops)}, one every {percentile(gaps, 50):.1f} s (p90 {percentile(gaps, 90):.1f})  ·  "
            f"ASR {percentile([h['asr_ms'] for h in hops], 50):.0f} ms (p90 {percentile([h['asr_ms'] for h in hops], 90):.0f})"
            f"  ·  delay {percentile(delay, 50):.1f} s  ·  no text {len(silent) / len(hops):.0%} "
            f"({len(paused)} in pauses, {len(silent) - len(paused)} while sound)")
    steps = [e for e in kinds("ctc") if isinstance(e.get("ms"), (int, float))]
    if len(steps) >= 30:
        # The word model's step time (ctc event "ms"), first and last third: a phone that slows as
        # it warms up takes longer at the end.
        ms = [e["ms"] for e in steps]
        k = len(ms) // 3
        out.append(f"  word model: {len(ms)} steps, {percentile(ms, 50):.0f} ms (p90 {percentile(ms, 90):.0f})  ·  "
                   f"first third {percentile(ms[:k], 50):.0f} ms, last third {percentile(ms[-k:], 50):.0f} ms")
    bat = kinds("battery")
    if bat:
        # Drain only over the time it wasn't charging (Chrome reports the level in 1% steps).
        runs = [(a, b) for a, b in zip(bat, bat[1:]) if not a["charging"]]
        used = sum(a["level"] - b["level"] for a, b in runs)
        mins = sum(b["t"] - a["t"] for a, b in runs) / 60
        rate = f" ({used / mins * 60:.0%} an hour)" if mins >= 10 and used > 0 else ""
        out.append(f"  battery {bat[0]['level']:.0%} -> {bat[-1]['level']:.0%}"
                   f"{', charging' if any(e['charging'] for e in bat) else ''}"
                   + (f"  ·  {used:.0%} in {mins:.0f} min unplugged{rate}" if mins > 0 else ""))
    found = kinds("dua")
    if found:
        out.append("  du'a shown: " + ", ".join(f"{lines.name(e['dua'])} at {e['t']:.0f} s" for e in found))
    else:
        out.append("  du'a shown: none")
    guesses = {g for e in kinds("guesses") for g in e["ids"]}
    if guesses:
        out.append("  guesses offered: " + ", ".join(lines.name(g) for g in sorted(guesses)))
    for e in kinds("lock"):
        out.append(f"  locked by tap at {e['t']:.0f} s: {lines.name(e['dua'])}")
    for e in kinds("msg"):
        out.append(f"  said \"{e['text']}\" at {e['t']:.0f} s")
    moves = kinds("line")
    if moves:
        idx = [m["idx"] for m in moves]
        steps = np.diff(idx)
        out.append(f"  line moves: {len(moves)} (lines {min(idx) + 1}-{max(idx) + 1})  ·  back {int((steps < 0).sum())}"
                   f"  ·  skipped ahead {int((steps > 1).sum())}")
    taps = kinds("tap")
    if taps:
        dua = found[-1]["dua"] if found else None
        offs = [(int(lines(dua, e["seg"])) - int(lines(dua, e["shown"])))
                for e in taps if lines(dua, e["seg"]).isdigit() and lines(dua, e["shown"]).isdigit()]
        out.append(f"  taps ('I'm here'): {len(taps)}  ·  shown line was off by " + ", ".join(f"{o:+d}" for o in offs))
    if kinds("scroll"):
        out.append(f"  scrolled by hand: {len(kinds('scroll'))} times")
    hidden = kinds("hidden")
    if hidden:
        out.append(f"  went to the background {len(hidden)} times (first at {hidden[0]['t']:.0f} s)")
    for e in kinds("audio"):
        out.append(f"  audio {e['state']} at {e['t']:.0f} s")
    for e in kinds("error"):
        out.append(f"  ERROR ({e.get('where')}) at {e['t']:.0f} s: {e.get('message', '')[:120]}")
    out += gate_summary(log)
    return out


def gate_summary(log: dict) -> list[str]:
    """Device sessions logged with the gate fields (web/app.js, 2026-09-26): why hops ran or
    were skipped, and how old the evidence was when shown. Older sessions have none: skipped."""
    hops = [e for e in log["events"] if e["type"] == "hop" and "gate" in e]
    if not hops:
        return []
    from collections import Counter

    reasons = Counter(h.get("skip") or "ran, text" for h in hops)
    out = [f"  speech gate ({hops[0]['gate']}): " + ", ".join(f"{k} {v}" for k, v in reasons.most_common())]
    levels = [h["level"] for h in hops if h.get("level") is not None]
    if levels:
        out.append(f"  input level (last 1.5 s): median {statistics.median(levels):.1f} dBFS, "
                   f"{sum(x < -45 for x in levels) / len(levels):.0%} of hops under the -45 dBFS floor")
    steady = [h for h in hops if not h.get("cold")]
    inf = [h["infer_ms"] for h in steady if h.get("infer_ms")]
    cold = [h["infer_ms"] for h in hops if h.get("cold") and h.get("infer_ms")]
    if inf:
        out.append(f"  Whisper: {percentile(inf, 50):.0f} ms p50, {percentile(inf, 90):.0f} ms p90 "
                   f"(first hop {cold[0]:.0f} ms)" if cold else f"  Whisper: {percentile(inf, 50):.0f} ms p50")
    ctc = [e for e in log["events"] if e["type"] == "ctc"]
    if ctc:
        gaps = np.diff([e["t"] for e in ctc])
        kind = ((log.get("words") or {}).get("follower")) or "rules"
        fms = [e["follow_ms"] for e in ctc if e.get("follow_ms") is not None]
        out.append(f"  word follower ({kind}): {len(ctc)} steps, one every {percentile(gaps, 50):.2f} s  ·  "
                   f"model {percentile([e['ms'] for e in ctc], 50):.0f} ms p50, {percentile([e['ms'] for e in ctc], 90):.0f} p90"
                   + (f"  ·  follower {percentile(fms, 50):.1f} ms p50, {percentile(fms, 90):.1f} p90" if fms else "")
                   + f"  ·  shown {percentile([e['delay'] for e in ctc], 50):.2f} s after its audio")
    ages = {e["end"]: e["age_ms"] for e in log["events"] if e["type"] == "visible" and e.get("age_ms") is not None}
    age = [ages[h["end"]] for h in steady if h["end"] in ages]
    if age:
        out.append(f"  evidence age when shown (captured -> painted, steady state): p50 {percentile(age, 50):.0f} ms, "
                   f"p90 {percentile(age, 90):.0f} ms, p95 {percentile(age, 95):.0f} ms")
    return out


def one_line(path: Path, audio: np.ndarray, log: dict, lines: Lines) -> str:
    ev = log["events"]
    hops = [e for e in ev if e["type"] == "hop"]
    found = [e for e in ev if e["type"] == "dua"]
    n = lambda k: sum(e["type"] == k for e in ev)  # noqa: E731
    asr = percentile([h["asr_ms"] for h in hops], 50)
    return (f"{path.stem:28s} {len(audio) / SR / 60:5.1f} min  {log.get('engine', '?'):6s} "
            f"{lines.name(found[-1]['dua']) if found else '(none)':30.30s} "
            f"found {found[0]['t'] if found else float('nan'):5.0f} s  moves {n('line'):4d}  taps {n('tap'):3d}  "
            f"ASR {asr:5.0f} ms")


# -- timeline -------------------------------------------------------------------
def timeline(log: dict, lines: Lines, words: bool = False) -> list[str]:
    out, dua = [], log.get("chosen")
    for e in log["events"]:
        t, k = e["t"], e["type"]
        if k == "hop":
            cand = e.get("cand") or []
            top = f"{cand[0][0]} {cand[0][1]:.2f}" if cand and not e.get("dua") else ""
            ev = f" (evidence {lines(e['dua'], e.get('now_seg'))})" if e.get("now_seg") is not None else ""
            now_q = f"/{e['quiet_now']:3.1f}" if e.get("quiet_now") is not None else ""
            out.append(f"{t:7.1f}  hop   asr {e['asr_ms']:5.0f} ms  delay {t - e['end']:4.1f}  quiet {e.get('quiet') or 0:3.1f}{now_q}  "
                       f"line {lines(e.get('dua'), e.get('seg')):>3s}:{e.get('token') if e.get('token') is not None else '-'}"
                       f"{ev}{'  ' + top if top else ''}  « {e.get('text') or ''} »")
        elif k == "dua":
            dua = e["dua"]
            out.append(f"{t:7.1f}  ══ du'a: {lines.name(dua)}")
        elif k == "line":
            out.append(f"{t:7.1f}  ── line {e['idx'] + 1}")
        elif k == "word":
            if words:
                out.append(f"{t:7.1f}     word {e['token']}")
        elif k == "tap":
            out.append(f"{t:7.1f}  ** TAP: I'm on line {lines(dua, e['seg'])} (showing {lines(dua, e['shown'])})")
        elif k == "guesses":
            out.append(f"{t:7.1f}  ?? guesses: {', '.join(lines.name(g) for g in e['ids']) or '(none)'}")
        elif k == "lock":
            out.append(f"{t:7.1f}  ** chose {lines.name(e['dua'])}")
        elif k == "hush":
            out.append(f"{t:7.1f}     {'stopped: the highlight holds' if e['on'] else 'reciting again'}")
        elif k == "ctc":
            if words:
                out.append(f"{t:7.1f}     ctc {e['ms']:4.0f} ms  word {e.get('word')}  (anchor {e.get('anchor')})")
        elif k == "preview":
            out.append(f"{t:7.1f}     next line {'previewed' if e['on'] else 'preview off'}")
        else:
            rest = {x: v for x, v in e.items() if x not in ("t", "ms", "type")}
            out.append(f"{t:7.1f}  {k}  {json.dumps(rest, ensure_ascii=False) if rest else ''}")
    return out


# -- scoring --------------------------------------------------------------------
def reference(sid: str, audio: np.ndarray, model: str) -> tuple[np.ndarray, list[str]]:
    """Transcripts of the 6 s windows ending every second, cached per session and model."""
    cache = CACHE / f"{sid}.{Path(model).name}.json"
    if cache.exists():
        ref = json.loads(cache.read_text(encoding="utf-8"))
        return np.array(ref["ends"]), ref["texts"]
    from dua_recognition.asr import transcribe_batch

    ends = np.arange(1, len(audio) // SR + 1, dtype=float)
    windows = [audio[max(0, int((e - 6) * SR)) : int(e * SR)] for e in ends]
    texts: list[str] = []
    for i in range(0, len(windows), 16):
        texts += transcribe_batch(windows[i : i + 16], model=model)
        print(f"\r  reference transcripts {len(texts)}/{len(windows)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"model": model, "ends": ends.tolist(), "texts": texts}, ensure_ascii=False),
                     encoding="utf-8")
    return ends, texts


def reference_lines(duas, dua_id: str, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Segment per window (offline smoother) and whether it's trustworthy there:
    the window's text sits in the du'a, or it's a pause after text that did."""
    from dua_recognition.align import CorpusIndex, encode
    from dua_recognition.offline import align_recording

    ix = CorpusIndex({dua_id: duas[dua_id]})
    costs = [ix.word_costs(t) if t else None for t in texts]
    segs = align_recording(ix, costs, 1.0)
    ok, placed = np.zeros(len(texts), bool), False
    for k, (t, c) in enumerate(zip(texts, costs)):
        if t:
            n = encode(t).size
            placed = c is not None and n >= 4 and float(c.min()) / n <= 0.4
        ok[k] = placed
    return segs, ok


# Tracker settings that JSON can't hold as numbers: the page logs Infinity as null.
_INF = {"max_speed", "lead_cross_words", "lead_cross_quiet", "retreat_after", "still_after", "still_motion_after"}


def logged_config(log: dict) -> dict:
    """The tracker settings the page ran with (web/tracker.js, camelCase), as TrackerConfig fields.
    Sessions from before a setting existed simply lack it: it keeps the Python default."""
    import dataclasses
    import re

    from dua_recognition.tracker import TrackerConfig

    fields = {f.name: f for f in dataclasses.fields(TrackerConfig)}
    # Settings added since some sessions were recorded, as they were before they existed
    # (the page then behaved as if they were off), and display_lead's old default for
    # sessions from before it changed (server sessions log no tracker settings at all).
    out = {"lead_cross_words": float("inf"), "seek_pins_line": False, "back_confirm": 1,
           "null_rate_locked": None, "keep_dua_confidence": None,
           "retreat_in_line": False, "still_catch_up": False, "pause_motion": False}
    if (log.get("started") or "") < "2026-09-30T08":
        out["display_lead"] = 0.5
    for k, v in (log.get("tracker") or {}).items():
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", k).lower()
        if name not in fields:
            continue
        if v is None and name in _INF:
            v = float("inf")
        elif v is None and "None" not in str(fields[name].type):
            continue  # not an optional setting: keep the default
        out[name] = tuple(v) if isinstance(v, list) else v
    return out


def retrack(log: dict, duas, overrides: dict, *, as_logged: bool = True, hear=None,
            index=None) -> list[tuple[str | None, int | None]]:
    """The Python tracker over the phone's own transcripts: (du'a, segment) per update.

    as_logged: start from the settings the page ran with (so {} reproduces what it
    showed); False: from today's TrackerConfig defaults. `overrides` go on top.
    hear(hop, tracker) replaces the phone's transcript (scripts/session_replay.py --asr);
    index is a prebuilt CorpusIndex of all `duas`, to save rebuilding it per call.
    """
    from dua_recognition.align import CorpusIndex
    from dua_recognition.tracker import RECITER, Tracker, TrackerConfig

    base = replace(TrackerConfig(), **{**(logged_config(log) if as_logged else {}), **overrides})
    cfg = lambda mode: replace(base, **RECITER) if mode == "reciter" else base  # noqa: E731
    mode = log.get("follow", "reading")
    full = index or CorpusIndex(duas)
    tracker = Tracker(CorpusIndex({log["chosen"]: duas[log["chosen"]]}) if log.get("chosen") else full, cfg(mode))
    last_end, out = 0.0, []
    no_lead = "lead=0" in (log.get("params") or "")
    # Newer sessions log each update's lead (delay + the page's display_lead): swap in
    # the display_lead being tried.
    logged = (log.get("tracker") or {}).get("displayLead", base.display_lead)
    for e in log["events"]:
        if e["type"] == "lock":
            tracker = Tracker(CorpusIndex({e["dua"]: duas[e["dua"]]}), cfg(mode))
        elif e["type"] == "follow":
            mode = e["mode"]
            tracker.cfg = cfg(mode)
        elif e["type"] == "seek":  # "I'm here" moved the tracker (Tracker.seek)
            tracker.seek(e["dua"], e["seg"])
        elif e["type"] == "hop":
            dt = e.get("dt", e["end"] - last_end)
            last_end = e["end"]
            if "lead" in e:
                lead = e["lead"] + tracker.cfg.display_lead - logged if e["lead"] > 0 else 0.0
            else:
                lead = 0.0 if no_lead else e["t"] - e["end"] + tracker.cfg.display_lead
            text = hear(e, tracker) if hear else e.get("text") or ""
            # Sessions since 2026-10-01 also log the page's live quiet and the pauses since the
            # last update (docs/results/stops.md); older ones have neither, as the page then.
            p = tracker.update(text, dt, lead=max(0.0, lead), quiet=e.get("quiet") or 0.0,
                               quiet_now=e.get("quiet_now"), paused=e.get("paused"))
            out.append((p.dua, p.segment))
    return out


def score(shown, hops, ref_segs, ref_ok, dua_id: str, key: str = "t", offset: int = 0) -> dict:
    """Line accuracy once per update against the reference at that moment.

    The reference for moment x is the window ending at x + 1 (it hears what was
    said ~1 s before its end, as in evaluate.py), index x; pass offset=-1 to
    compare a window's own end instead."""
    n = right = near = none = other = 0
    wrong_runs, run = [], None
    for (d, s), h in zip(shown, hops):
        k = min(len(ref_segs) - 1, max(0, round(h[key]) + offset))
        if not ref_ok[k]:
            continue
        g = int(ref_segs[k])
        n += 1
        ok = d == dua_id and s == g
        right += ok
        near += d == dua_id and s is not None and abs(s - g) <= 1
        none += d is None
        other += d is not None and d != dua_id
        if not ok:
            run = run or {"from": h["t"], "shown": [], "truth": [], "text": []}
            run.update(to=h["t"])
            run["shown"].append(s if d == dua_id else None)
            run["truth"].append(g)
            run["text"].append(h.get("text") or "")
        elif run:
            wrong_runs.append(run)
            run = None
    if run:
        wrong_runs.append(run)
    return {"n": n, "right": right / max(1, n), "near": near / max(1, n), "none": none / max(1, n),
            "other": other / max(1, n), "runs": sorted(wrong_runs, key=lambda r: r["from"] - r["to"])}


def lag(shown, hops, ref_segs, ref_ok, dua_id: str) -> list[float]:
    """Seconds from each new line (reference) until the phone shows it."""
    out = []
    truth = [int(ref_segs[min(len(ref_segs) - 1, round(h["t"]))]) for h in hops]
    for i in range(1, len(hops)):
        if truth[i] <= truth[i - 1] or not ref_ok[min(len(ref_ok) - 1, round(hops[i]["t"]))]:
            continue
        for j in range(max(0, i - 5), min(len(hops), i + 30)):
            if shown[j] == (dua_id, truth[i]):
                out.append(hops[j]["t"] - hops[i]["t"])
                break
    return out


def report_score(sid: str, audio, log, duas, lines: Lines, model: str, dua_id: str | None, overrides: dict) -> list[str]:
    hops = [e for e in log["events"] if e["type"] == "hop"]
    found = [e["dua"] for e in log["events"] if e["type"] == "dua"]
    dua_id = dua_id or log.get("chosen") or (max(set(found), key=found.count) if found else None)
    if dua_id is None:
        return ["  --score: the phone never named a du'a; say which it was with --dua"]
    ends, texts = reference(sid, audio, model)
    segs, ok = reference_lines(duas, dua_id, texts)
    out = [f"  reference: {Path(model).name} + offline smoother, placed {ok.mean():.0%} of seconds in "
           f"{lines.name(dua_id)}"]
    phone = [(h.get("dua"), h.get("seg")) for h in hops]
    runs = [("phone", phone)]
    retracked = retrack(log, duas, {})
    runs.append(("python tracker, same transcripts", retracked))
    agree = np.mean([a == b for a, b in zip(phone, retracked)]) if hops else float("nan")
    if overrides:
        runs.append((f"python tracker, {' '.join(f'{k}={v}' for k, v in overrides.items())}",
                     retrack(log, duas, overrides)))
    for name, shown in runs:
        s = score(shown, hops, segs, ok, dua_id)
        lags = lag(shown, hops, segs, ok, dua_id)
        out.append(f"  {name:40s} line {s['right']:5.1%}  ±1 {s['near']:5.1%}  nothing shown {s['none']:5.1%}  "
                   f"wrong du'a {s['other']:4.1%}  median lag {statistics.median(lags) if lags else float('nan'):4.1f} s"
                   f"  ({s['n']} updates)")
    out.append(f"  (the python tracker matches the phone's display on {agree:.0%} of updates)")
    if hops and "now_seg" in hops[0]:
        s = score([(h.get("dua"), h.get("now_seg")) for h in hops], hops, segs, ok, dua_id, "end", -1)
        out.append(f"  {'phone evidence (no lead, at window end)':40s} line {s['right']:5.1%}  ±1 {s['near']:5.1%}")
    taps = [e for e in log["events"] if e["type"] == "tap"]
    if taps:
        agree_taps = [abs(int(segs[min(len(segs) - 1, round(e["t"]))]) - e["seg"]) <= 1 for e in taps]
        out.append(f"  taps agree with the reference (±1 line): {sum(agree_taps)}/{len(taps)}")
    s = score(phone, hops, segs, ok, dua_id)
    if s["runs"]:
        out.append("  longest stretches the phone was off (shown vs reference line):")
        for r in s["runs"][:8]:
            shown = sorted({lines(dua_id, x) for x in r["shown"]}, key=lambda x: (not x.isdigit(), x.zfill(4)))
            truth = sorted({lines(dua_id, x) for x in r["truth"]}, key=lambda x: x.zfill(4))
            said = next((t for t in r["text"] if t), "")
            out.append(f"    {r['from']:6.0f}-{r['to']:<6.0f} showed {','.join(shown):10s} was on {','.join(truth):10s} "
                       f"heard « {said[:60]} »")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", help="file, session id or part of one (default: the newest)")
    ap.add_argument("--all", action="store_true", help="one line per session")
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--words", action="store_true", help="timeline: every word the highlight moved to")
    ap.add_argument("--log", action="store_true", help="print the raw log as JSON")
    ap.add_argument("--score", action="store_true", help="score against a reference transcription (GPU)")
    ap.add_argument("--model", default=REF_MODEL if (ROOT / REF_MODEL).exists() else "large-v3-turbo",
                    help="reference ASR model")
    ap.add_argument("--dua", help="--score: the du'a actually recited (default: the one the phone showed most)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="--score: re-run the tracker with these TrackerConfig changes")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # transcripts are Arabic; the Windows console defaults to cp1252

    files = sorted(SESSIONS.glob("*.wav")) if args.all else find(args.session)
    if not files:
        sys.exit(f"no session matches {args.session!r} in {SESSIONS}" if args.session else f"no sessions in {SESSIONS}")
    from dua_recognition.corpus import load_all

    duas = load_all()
    lines = Lines(duas)
    if args.all:
        for f in files:
            audio, log = read_session(f)
            print(one_line(f, audio, log, lines))
        return
    if len(files) > 1:
        sys.exit("several sessions match: " + ", ".join(f.stem for f in files))
    path = files[0]
    audio, log = read_session(path)
    if args.log:
        print(json.dumps(log, ensure_ascii=False, indent=1))
        return
    print("\n".join(summary(path, audio, log, lines)))
    if args.timeline:
        print()
        print("\n".join(timeline(log, lines, args.words)))
    if args.score:
        overrides = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
        print()
        print("\n".join(report_score(path.stem, audio, log, duas, lines, args.model, args.dua, overrides)))


if __name__ == "__main__":
    main()
