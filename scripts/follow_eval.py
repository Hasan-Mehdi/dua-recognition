#!/usr/bin/env python
"""The CTC word follower (follower.py) against the page-mode display it would replace.

Two lanes over the same recordings:

    hmm       the live display today: tracker + lead + pause rules + gliding
              highlight (word_eval's page mode), from the cached 6 s windows
    follow    the tracker says which du'a and line (its evidence position, no
              lead); the follower moves word by word from CTC frames every
              0.2 s (scripts/dump_ctc.py --window 3 --hop 0.2), each result on
              screen `--follow-delay` s after its window ends

Scored every 0.1 s against forced-aligned word timings (as word_eval.py), plus
a line-entry check against the *human* line starts, which doesn't depend on the
wav2vec2 word truth: how late the display enters each new line, how often it
enters more than 0.3 s early. `--pauses` scores the pause benchmark instead
(scripts/pause_eval.py: 4 s of room tone after every third line end).

    python scripts/follow_eval.py --split train --grid           # tune the follower
    python scripts/follow_eval.py "fw window_s=2,beta_back=3"    # test, one setting
    python scripts/follow_eval.py --pauses "fw ..."
    python scripts/follow_eval.py --split train --diag           # where the line-entry lag comes from

Eval sets are pickled under data/cache/follow_items (--no-cache rebuilds them).
"""
from __future__ import annotations

import argparse
import bisect
import itertools
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
import pause_eval as pe  # noqa: E402
import word_eval as we  # noqa: E402
from dua_recognition.align import encode  # noqa: E402
from dua_recognition.follower import FollowerConfig, LocalFollower  # noqa: E402
from dua_recognition.tracker import TrackerConfig  # noqa: E402

CTC = ROOT / "data" / "cache" / "ctc"
ITEMS = ROOT / "data" / "cache" / "follow_items"
SMOOTH = {"ease": 1.0, "speed_scale": 1.2}  # page mode (web/app.js defaults)


MIN_SCORE = -1.5  # word truth kept for scoring: line score at least this (flowing set)


def _files_sig(*dirs: Path, pattern: str = "*") -> list:
    """(name, size, mtime) of every input file: a changed or added input changes the key."""
    out = []
    for d in dirs:
        if d.exists():
            out += sorted((f"{d.name}/{p.name}", p.stat().st_size, p.stat().st_mtime_ns) for p in d.glob(pattern)
                          if p.is_file())
    return out


def set_key(ix, asr: str, split: str, ctc_tag: str, ctc_dir: str, pauses: bool, cfg, delay) -> str:
    """Everything a cached eval set depends on: the tracker config, the reference texts,
    the window/CTC/word-truth/pause inputs, the scoring threshold and the assumed delay.
    (Until 2026-09-26 the key was only asr/split/ctc/delay: a TrackerConfig or corpus
    change silently reused old HMM replays.)"""
    import hashlib

    ref = hashlib.sha256(" ".join(w.text for w in ix.words).encode("utf-8")).hexdigest()
    if pauses:
        d = pe.OUT / (asr + ("" if split == "test" else f"@{split}"))
        inputs = _files_sig(d, pattern="*.json") + _files_sig(d / f"ctc-{ctc_dir}")
    else:
        inputs = _files_sig(ev.WINDOWS / asr, ev.WORD_TRUTH, CTC / ctc_tag, ROOT / "data" / "cache" / "quiet",
                            ROOT / "data" / "cache" / "stale")
    parts = [repr(cfg), ref, asr, split, ctc_tag, ctc_dir, pauses, f"{delay:g}", MIN_SCORE, inputs]
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:12]


def load_set(ix, asr: str, split: str, ctc_tag: str, ctc_dir: str, pauses: bool, cfg, delay,
             cache: bool = True) -> list[dict]:
    """load_set_uncached, pickled under data/cache/follow_items (the HMM replay takes minutes),
    keyed by set_key so a changed configuration or reference never reuses a stale set."""
    ctc = ctc_dir if pauses else ctc_tag
    key = set_key(ix, asr, split, ctc_tag, ctc_dir, pauses, cfg, delay)
    f = ITEMS / f"{asr}_{split}_{ctc}_{'pauses' if pauses else 'flowing'}_{delay:g}_{key}.pkl"
    if cache and f.exists():
        return pickle.loads(f.read_bytes())
    items = load_set_uncached(ix, asr, split, ctc_tag, ctc_dir, pauses, cfg, delay)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(items))
    return items


def load_set_uncached(ix, asr: str, split: str, ctc_tag: str, ctc_dir: str, pauses: bool, cfg, delay) -> list[dict]:
    """Recordings with everything both lanes need; the HMM lane is replayed here, once."""
    items = []
    if not pauses:
        _, data = we.load(asr, split)
        for rec, rows, costs, stale, truth, quiet in data:
            good = [w for w in truth if w[3] >= MIN_SCORE]
            f = CTC / ctc_tag / f"{rec.audio_id}_w3_h0.2.npz"
            if not good or not f.exists():
                continue
            anchors = []
            ups = we.hmm_updates(ix, rows, costs, quiet, cfg, delay, "fixed", stale, still=True, anchors=anchors)
            t_end = min(rec.end_s, good[-1][2] + 2.0)
            items.append({"id": rec.audio_id, "dua": rec.dua_id, "ups": ups, "anchors": anchors, "good": good,
                          "ticks": np.arange(good[0][1], t_end, 0.1), "ctc": f,
                          "lines": [(t, s) for t, s in rec.starts if t < t_end], "pauses": [], "pause_s": 0.0})
        return items
    recs = {r.audio_id: r for d in ev.load_all().values() for r in ev.load_recordings(d)}
    d = pe.OUT / (asr + ("" if split == "test" else f"@{split}"))
    for jf in sorted(d.glob("*.json")):
        r = json.loads(jf.read_text(encoding="utf-8"))
        f = d / f"ctc-{ctc_dir}" / f"{jf.stem}_w3_h0.2.npz"
        if not f.exists():
            continue
        lo = ix.dua_word_span[ix.dua_ids.index(r["dua"])][0]
        P = r["pause_s"]
        orig = [s - P * j for j, (s, _) in enumerate(r["pauses"])]

        def shift(t: float) -> float:
            return t + P * sum(1 for p in orig if p <= t)

        truth = [[lo + w, a, b, 0.0] for w, a, b in r["truth"]]
        costs = ev.LazyCosts(ix, [x for _, x in r["rows"]])
        letters = [len(encode(x)) if x else 0 for _, x in r["rows"]]
        anchors = []
        ups = we.hmm_updates(ix, r["rows"], costs, r["quiet"], cfg, delay, "fixed", None, still=True,
                             anchors=anchors, letters=letters)
        end = r["rows"][-1][0] + delay
        items.append({"id": r["audio_id"], "dua": r["dua"], "ups": ups, "anchors": anchors, "good": truth,
                      "ticks": np.arange(truth[0][1], end, 0.1), "ctc": f,
                      "lines": [(shift(t), s) for t, s in recs[r["audio_id"]].starts if shift(t) < end],
                      "pauses": [[s, lo + w] for s, w in r["pauses"]], "pause_s": P})
    return items


def follow_updates(ix, it: dict, fcfg: FollowerConfig, hmm_delay: float, f_delay: float, *, rc: int = 0,
                   oracle_anchor: bool = False, steps: list | None = None) -> list[tuple]:
    """The follow lane's screen updates for one recording.

    Diagnostics (--diag): `rc` = n gives each step n hops of right context (window
    k+n minus its last n*10 frames: the same audio, no frame later than t);
    `oracle_anchor` anchors on the true word at t instead of the tracker; `steps`
    (a list) collects (t, word) per step, before the compute delay."""
    z = np.load(it["ctc"])
    lp, nf, ts = z["lp"], z["n_frames"], z["t"]
    fol = LocalFollower(ix, fcfg)
    anchors = it["anchors"]
    at = [a[0] + hmm_delay for a in anchors]  # when each tracker result is available
    lead_at = [u[0] for u in it["ups"]]  # page mode's (lead) position, as it reaches the screen
    starts = [w[1] for w in it["good"]]
    hop_frames = int(round((ts[1] - ts[0]) / 0.02)) if len(ts) > 1 else 10
    ups = []
    for k, t in enumerate(ts):
        j = bisect.bisect_right(at, t) - 1
        hmm_word = anchors[j][1] if j >= 0 else None
        if oracle_anchor:
            i = bisect.bisect_right(starts, t) - 1
            hmm_word = it["good"][i][0] if i >= 0 else None
        j = bisect.bisect_right(lead_at, t) - 1
        lead_word = it["ups"][j][3] if j >= 0 else None
        if rc and k + rc < len(ts):
            x = lp[k + rc, : max(0, nf[k + rc] - rc * hop_frames)]
        else:
            x = lp[k, : nf[k]]
        w = fol.step(x, float(t), hmm_word, lead_word=lead_word)
        if steps is not None:
            steps.append((float(t), w))
        if w is None:
            ups.append((t + f_delay, None, None, None, None, 0.0))
        else:
            ups.append((t + f_delay, ix.dua_ids[ix.word_dua[w]], int(ix.word_segment[w]), w, None, 0.0))
    return ups


def line_entries(ix, it: dict, shown: list) -> list[float | None]:
    """For each human line start after the first: when the display entered that line, relative to it."""
    ticks = it["ticks"]
    seg = [None if w is None else int(ix.word_segment[w]) for w in shown]
    out = []
    for t, s in it["lines"][1:]:
        a, b = np.searchsorted(ticks, [t - 4.0, t + 4.0])
        lag = None
        for i in range(max(1, a), min(b, len(ticks))):
            if seg[i] == s and seg[i - 1] != s:
                lag = float(ticks[i] - t)
                break
        out.append(lag)
    return out


def pause_stats(ix, it: dict, shown: list) -> list[bool]:
    """Per tick inside a pause: is a later line than the one just finished on screen?"""
    starts = [w[1] for w in it["good"]]
    out = []
    for tick, w in zip(it["ticks"], shown):
        if w is None or not any(s + 0.5 <= tick < s + it["pause_s"] for s, _ in it["pauses"]):
            continue
        i = bisect.bisect_right(starts, tick) - 1
        if i < 0:
            continue
        tw = it["good"][i][0]
        out.append(bool(ix.word_segment[w] != ix.word_segment[tw] and w > tw))
    return out


def score(ix, items, lane: str, fcfg=None, hmm_delay=0.5, f_delay=0.1) -> dict:
    offs, jerks, minutes, entries, nxt, covered, n_ticks = [], 0, 0.0, [], [], 0, 0
    for it in items:
        if lane == "hmm":
            r = we.replay_ticks(ix, it["dua"], it["ups"], it["good"], it["ticks"], SMOOTH)
        else:
            ups = follow_updates(ix, it, fcfg, hmm_delay, f_delay)
            r = we.replay_ticks(ix, it["dua"], ups, it["good"], it["ticks"], None)
        offs += r["offs"]
        jerks += r["jerks"]
        minutes += len(it["ticks"]) / 600
        covered += sum(w is not None for w in r["shown"])
        n_ticks += len(r["shown"])
        entries += line_entries(ix, it, r["shown"])
        nxt += pause_stats(ix, it, r["shown"])
    o = np.array(offs)
    found = [x for x in entries if x is not None]
    return {"exact": float(np.mean(o == 0)), "pm1": float(np.mean(np.abs(o) <= 1)), "mean": float(o.mean()),
            "jerks_min": jerks / max(minutes, 1e-9), "shown": covered / max(1, n_ticks),
            "entry_lag": float(np.median(found)) if found else float("nan"),
            "early": float(np.mean([x < -0.3 for x in found])) if found else float("nan"),
            "missed": 1 - len(found) / max(1, len(entries)), "n_lines": len(entries),
            "pause_next": float(np.mean(nxt)) if nxt else float("nan")}


def _line_first(ix, w: int) -> bool:
    return w == ix.dua_word_span[ix.word_dua[w]][0] or ix.word_segment[w] != ix.word_segment[w - 1]


def word_entries(ix, it: dict, shown: list) -> list[tuple]:
    """Per truth word: (first of its line?, gap before it in s, entry lag in s). Entry = the
    first tick from start - 1 s showing that word or a later one (capped at +3 s)."""
    ticks, good = it["ticks"], it["good"]
    out = []
    for i, (w, a, _, _) in enumerate(good):
        gap = a - good[i - 1][2] if i else np.inf
        lo, hi = np.searchsorted(ticks, [a - 1.0, a + 3.0])
        lag = 3.0
        for k in range(lo, min(hi, len(ticks))):
            if shown[k] is not None and shown[k] >= w:
                lag = float(ticks[k] - a)
                break
        out.append((_line_first(ix, w), float(gap), lag))
    return out


def evidence_gaps(ix, it: dict, steps: list) -> list[float]:
    """A3. Per line-first truth word: the step where the follower reaches it, minus the first
    step whose window has the word's first letter beating blank in a frame at or after the
    word's aligned start. Both in step time (before the compute delay)."""
    z = np.load(it["ctc"])
    lp, nf, ts = z["lp"], z["n_frames"], z["t"]
    first_letter = LocalFollower(ix, FollowerConfig())._word_letter
    out = []
    for w, a, _, _ in it["good"]:
        if not _line_first(ix, w):
            continue
        c = int(ix.letters[first_letter[w]])
        ev_t = mv_t = None
        for k in range(int(np.searchsorted(ts, a)), len(ts)):
            if ts[k] > a + 3.0:
                break
            f = lp[k, : nf[k]].astype(np.float32)
            ft = ts[k] - 0.02 * (nf[k] - np.arange(nf[k]))  # frame start times
            sel = ft >= a - 0.02
            if (f[sel, c] > f[sel, 0]).any():
                ev_t = float(ts[k])
                break
        for t, sw in steps:
            if t >= a - 1.0 and sw is not None and sw >= w:
                mv_t = t
                break
        if ev_t is not None and mv_t is not None and mv_t <= a + 3.0:
            out.append(mv_t - ev_t)
    return out


def diag(ix, sets: dict, hmm_delay: float, f_delay: float) -> None:
    """Where does the line-entry lag come from? (--diag; FollowerConfig defaults)."""
    def q(x, p):
        return f"{np.percentile(x, p):+.2f}" if len(x) else "n/a"

    bins = [("gap < 0.3 s", 0.0, 0.3), ("gap 0.3-1 s", 0.3, 1.0), ("gap > 1 s", 1.0, np.inf)]
    variants = [("follower (round 1)", {}, f_delay), ("A1 right context 0.2 s", {"rc": 1}, f_delay),
                ("A1 right context 0.4 s", {"rc": 2}, f_delay), ("A1 right context 0.6 s", {"rc": 3}, f_delay),
                ("A2 oracle anchor", {"oracle_anchor": True}, f_delay), ("A4 follow delay 0", {}, 0.0)]
    for name, items in sets.items():
        off = []
        for it in items:
            for t, s in it["lines"]:
                ws = [g for g in it["good"] if int(ix.word_segment[g[0]]) == s]
                if ws and abs(ws[0][1] - t) < 3.0:
                    off.append(ws[0][1] - t)
        print(f"\n== {name}: {len(items)} recordings; human line start -> aligned first word: "
              f"median {q(off, 50)} s, p25 {q(off, 25)}, p75 {q(off, 75)} (n={len(off)})")
        print("variant | exact | line entry (human) | other words med / p75 | "
              + " | ".join(f"line-first, {b}: med / p75 (n)" for b, _, _ in bins))
        for vname, kw, fd in [("hmm (page mode)", None, None)] + variants:
            offs, ents, wents = [], [], []
            for it in items:
                if kw is None:
                    r = we.replay_ticks(ix, it["dua"], it["ups"], it["good"], it["ticks"], SMOOTH)
                else:
                    ups = follow_updates(ix, it, FollowerConfig(), hmm_delay, fd, **kw)
                    r = we.replay_ticks(ix, it["dua"], ups, it["good"], it["ticks"], None)
                offs += r["offs"]
                ents += [x for x in line_entries(ix, it, r["shown"]) if x is not None]
                wents += word_entries(ix, it, r["shown"])
            other = [g for f, _, g in wents if not f]
            cells = []
            for _, lo, hi in bins:
                x = [g for f, gap, g in wents if f and lo <= gap < hi]
                cells.append(f"{q(x, 50)} / {q(x, 75)} ({len(x)})")
            print(f"{vname} | {np.mean(np.array(offs) == 0):.1%} | {np.median(ents):+.2f} s | "
                  f"{q(other, 50)} / {q(other, 75)} | " + " | ".join(cells), flush=True)
        gaps = []
        for it in items:
            steps = []
            follow_updates(ix, it, FollowerConfig(), hmm_delay, f_delay, steps=steps)
            gaps += evidence_gaps(ix, it, steps)
        print(f"A3 decision minus evidence (line-first words, step time): median {q(gaps, 50)} s, "
              f"p25 {q(gaps, 25)}, p75 {q(gaps, 75)} (n={len(gaps)})", flush=True)


def parse_fw(v: str) -> FollowerConfig:
    kind, _, kv = v.partition(" ")
    kw = {k: float(x) for k, x in (a.split("=") for a in kv.split(",") if a)}
    for k in ("max_jump", "back_words", "ahead_words", "confirm_steps"):
        if k in kw:
            kw[k] = int(kw[k])
    if "onset_confirm" in kw:
        kw["onset_confirm"] = bool(kw["onset_confirm"])
    return replace(FollowerConfig(), **kw)


def fmt(name: str, m: dict) -> str:
    return (f"{name:55s} | {m['exact']:.1%} | {m['pm1']:.1%} | {m['mean']:+.2f} | {m['jerks_min']:.2f} | "
            f"{m['shown']:.1%} | {m['entry_lag']:+.2f} s | {m['early']:.1%} | {m['missed']:.1%} | "
            f"{m['pause_next']:.1%}")


HEADER = ("lane | word exact | ±1 | mean off | jerks/min | shown | line entry median | >0.3 s early | "
          "missed | next line in pause")

GRID = {"window_s": [1.5, 2.0, 3.0], "beta_back": [1.0, 3.0, 10.0], "max_jump": [2, 4], "temp": [1.0, 3.0],
        "reset_after": [1.0, 3.0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr", default="whisper-base-quran-dua", help="window-cache tag for the tracker")
    ap.add_argument("--ctc", default="wav2vec2-quran-dua", help="CTC stream cache tag")
    ap.add_argument("--ctc-dir", help="pause set: ctc-<name> folder (default: --ctc)")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--pauses", action="store_true", help="the pause benchmark (scripts/pause_eval.py)")
    ap.add_argument("--delay", type=float, default=0.5, help="tracker (Whisper) delay, s")
    ap.add_argument("--follow-delay", type=float, default=0.1, help="follower compute delay, s")
    ap.add_argument("--grid", action="store_true", help="also run the tuning grid")
    ap.add_argument("--diag", action="store_true",
                    help="where the line-entry lag comes from: flowing + pause sets of --split")
    ap.add_argument("--no-cache", action="store_true", help="rebuild the pickled eval set")
    ap.add_argument("variants", nargs="*", help='follower settings: "fw k=v,..."')
    args = ap.parse_args()
    cfg = TrackerConfig()
    ix = ev.CorpusIndex(ev.load_all())
    t0 = time.time()
    if args.diag:
        sets = {k: load_set(ix, args.asr, args.split, args.ctc, args.ctc_dir or args.ctc, p, cfg, args.delay,
                            cache=not args.no_cache) for k, p in (("flowing", False), ("pauses", True))}
        print(f"diag ({args.split}), loaded in {time.time() - t0:.0f} s")
        diag(ix, sets, args.delay, args.follow_delay)
        return
    items = load_set(ix, args.asr, args.split, args.ctc, args.ctc_dir or args.ctc, args.pauses, cfg, args.delay,
                     cache=not args.no_cache)
    print(f"{len(items)} recordings ({args.split}{', pauses' if args.pauses else ''}), tracker ASR {args.asr}, "
          f"CTC {args.ctc}, delays {args.delay:g} / {args.follow_delay:g} s  (loaded in {time.time() - t0:.0f} s)")
    print(HEADER, flush=True)
    print(fmt("hmm (page mode)", score(ix, items, "hmm")), flush=True)
    variants = list(args.variants) or (["fw"] if not args.grid else [])
    if args.grid:
        keys = list(GRID)
        variants += ["fw " + ",".join(f"{k}={v:g}" for k, v in zip(keys, vals))
                     for vals in itertools.product(*GRID.values())]
    for v in variants:
        m = score(ix, items, "follow", parse_fw(v), args.delay, args.follow_delay)
        print(fmt(v, m), flush=True)


if __name__ == "__main__":
    main()
