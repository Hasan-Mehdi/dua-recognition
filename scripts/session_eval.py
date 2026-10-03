#!/usr/bin/env python
"""Word-level scores for the debug sessions: the page as it ran, rebuilt, and with the word follower.

Truth: the whole session forced-aligned to the du'a's text with the fine-tuned wav2vec2
(as word_truth.py does for DuaPlayer lines): every frame of the recording against the
lines the reference (session_report: turbo + the offline smoother) says were recited,
one line either side, free to start and end anywhere in that text. Each word gets a
start and an end; a word whose letters the model barely hears is marked unsure, and
the ticks it would decide aren't scored. Automatic, not human: a check on the
follower's own teacher model, so the follower lane is flattered, the page lanes aren't.

Lanes, every 0.05 s of page time (session_stops.py rebuilds the page):
- recorded: the page as the phone drew it (its logged updates, glide and stop detector);
- page: today's tracker and display rules over the same updates;
- follow: the same, plus the CTC word follower (follower.py) on `--ctc` frames, 3 s
  windows every `--hop` s, each shown `--delay` s after its audio; the follower's word
  replaces the page's while its results keep coming (app.js renderWord).

Columns: word exact / ±1 (against the last word begun), line, jerks/min (a step back
or more than two words forward), line back / flicker (A -> B -> A within 3 s) per min,
line entry (median lag after the line's first word begins) and early entries (> 0.3 s
before it), ahead in pauses (share of time in gaps of 1 s+ that the display is past
the last word said).

    python scripts/session_eval.py --ctc models/ctc-student-base
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from session_report import REF_MODEL, SESSIONS, SR, find, read_session, reference, reference_lines  # noqa: E402
from session_stops import TICK, audio_offset, display, recorded, renders, word_index  # noqa: E402

CACHE = ROOT / "data" / "cache" / "session_truth"
TRUTH_MODEL = "wav2vec2-quran-dua"  # --truth-model; another model's truth is cached in session_truth/<model>
CTC_CACHE = ROOT / "data" / "cache" / "session_ctc"
NEG = -1e9
TAIL = 4.0  # seconds scored after the last word said


def ctc_frames(sid: str, audio, model, tag: str, window: float, hop: float):
    """A CTC model's frames over the session, windows of `window` s every `hop` s (cached)."""
    f = CTC_CACHE / tag / f"{sid}_w{window:g}_h{hop:g}.npz"
    if f.exists():
        z = np.load(f)
        return z["lp"], z["n_frames"], z["t"]
    times = [round((k + 1) * hop, 3) for k in range(int(len(audio) / SR / hop))]
    arr, lens, _, _ = model().windows(audio, times, window, 32)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, lp=arr, n_frames=lens, t=np.array(times))
    return arr, lens, np.array(times)


def _align(lp, r, word_of):
    """Semi-global CTC Viterbi: the frames explain a contiguous run of letters r (free start
    and end in r). Returns per-letter first frame (-1 if not on the path) and log prob."""
    T, J = lp.shape[0], r.size
    S = 2 * J + 1
    dp = np.full(S, NEG)
    bp = np.zeros((T, S), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2, 3 fresh start
    for s in range(S):
        dp[s] = lp[0, 0] if s % 2 == 0 else lp[0, r[(s - 1) // 2]]
        bp[0, s] = 3
    for t in range(1, T):
        new = np.full(S, NEG)
        for s in range(S):
            best, arg = dp[s], 0
            if s >= 1 and dp[s - 1] > best:
                best, arg = dp[s - 1], 1
            if s >= 2 and s % 2 == 1 and r[(s - 1) // 2] != r[(s - 3) // 2] and dp[s - 2] > best:
                best, arg = dp[s - 2], 2
            e = lp[t, 0] if s % 2 == 0 else lp[t, r[(s - 1) // 2]]
            new[s] = best + e
            bp[t, s] = arg
        dp = new
    s = int(np.argmax(dp))
    first = np.full(J, -1, dtype=np.int64)
    logp = np.zeros(J)
    for t in range(T - 1, -1, -1):
        if s % 2 == 1:
            j = (s - 1) // 2
            first[j] = t
            logp[j] = max(logp[j] if logp[j] < 0 else NEG, lp[t, r[j]])
        a = bp[t, s]
        if a == 3:
            break
        s -= a
    return first, logp


try:
    from numba import njit

    _align = njit(cache=True)(_align)
except ImportError:
    pass


def truth(sid: str, audio, duas, index, dua: str, ref_segs, ref_ok):
    """[(word index, start s, end s, sure)] for the words of the recited lines."""
    f = (CACHE if TRUTH_MODEL == "wav2vec2-quran-dua" else CACHE / TRUTH_MODEL) / f"{sid}.json"
    if f.exists():
        return json.loads(f.read_text())
    from dua_recognition.ctc import CtcModel

    m = CtcModel(str(ROOT / "models" / TRUTH_MODEL))
    # Whole session in 20 s windows, frames kept from the middle 18 s of each (1 s context either side).
    T = len(audio) / SR
    frames = []
    for a in np.arange(0.0, T, 18.0):
        lo, hi = max(0.0, a - 1.0), min(T, a + 19.0)
        lp = m.window(audio[int(lo * SR) : int(hi * SR)])
        k0 = int(round((a - lo) / 0.02))
        k1 = k0 + int(round((min(T, a + 18.0) - a) / 0.02))
        frames.append(lp[k0:k1])
    lp = np.concatenate(frames).astype(np.float64)
    order = [s.id for s in duas[dua].segments]
    said = sorted({int(ref_segs[k]) for k in range(len(ref_segs)) if ref_ok[k] and int(ref_segs[k]) in order},
                  key=order.index)
    if not said:
        return []
    i0, i1 = max(0, order.index(said[0]) - 1), min(len(order) - 1, order.index(said[-1]) + 1)
    lines = set(order[i0 : i1 + 1])
    d = index.dua_ids.index(dua)
    lo, hi = index.dua_word_span[d]
    words = [w for w in range(lo, hi) if int(index.word_segment[w]) in lines]
    first_letter = np.r_[0, np.flatnonzero(np.diff(index.letter_word) != 0) + 1, index.letters.size]
    r, owner = [], []
    for w in words:
        seg = index.letters[first_letter[w] : first_letter[w + 1]]
        r += [int(c) for c in seg]
        owner += [w] * len(seg)
    r, owner = np.array(r, dtype=np.int64), np.array(owner)
    first, logp = _align(lp, r, owner)
    out = []
    for w in words:
        js = np.flatnonzero(owner == w)
        fr = first[js]
        if (fr < 0).any():
            continue
        sure = bool(np.mean(logp[js]) > np.log(0.05))
        out.append([int(w), float(fr[0] * 0.02), float((fr[-1] + 1) * 0.02), sure])
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def follow_display(rs, duas, until, live, frames, lens, times, off, delay, fcfg):
    """display() with the follower's word over the page's while the follower runs."""
    from dua_recognition.follower import LocalFollower

    page = display(rs, duas, until, {}, live or None, back_after=1.0 if live else None)
    ticks = [p[0] for p in page]
    rs_t = [r["t"] for r in rs]
    out = []
    follower, word, word_at = None, None, -1e9
    fi = 0
    ft = times + off + delay
    live_t = [x[0] for x in live] if live else []
    for (t, dua, seg, tok) in page:
        while fi < len(ft) and ft[fi] <= t:
            k = fi
            fi += 1
            j = bisect.bisect_right(rs_t, t) - 1
            if j < 0:
                continue
            r = rs[j]
            ix, anchor = r["ix"], r.get("anchor")
            if r.get("seek") and follower is not None:
                follower.reset()
            if anchor is None and (follower is None or follower.word is None):
                if follower is not None:
                    follower.reset()
                continue
            if follower is None or follower.ix is not ix:
                follower = LocalFollower(ix, fcfg)
            qn = None
            if live:  # the page's stop detector at this moment (page time, seconds silent)
                q = bisect.bisect_right(live_t, t) - 1
                qn = live[q][1] if q >= 0 else None
            w = follower.step(frames[k, : lens[k]].astype(np.float32), float(times[k]), anchor, quiet_now=qn)
            if w is not None:
                word, word_at, wix = w, t, ix
        if word is not None and t - word_at < 1.0:
            wd = wix.words[word]
            out.append((t, wix.dua_ids[wd.dua], wd.segment, wd.token))
        else:
            out.append((t, dua, seg, tok))
    return out


def score(shown, tr, index, dua: str, off: float) -> dict:
    sure = [w for w in tr if w[3]]
    starts = [w[1] for w in tr]
    if not tr:
        return {}
    # Scored until 4 s after the last word: the silence after a du'a's end counts as a pause.
    t0, t1 = tr[0][1], tr[-1][2] + TAIL
    line_first = {}
    for w, a, _, _ in tr:
        s = int(index.word_segment[w])
        if s not in line_first:
            line_first[s] = a
    offs, line_ok, n_line = [], 0, 0
    jerks = backs = flick = 0
    prev_w, segs = None, []
    pause_ticks = pause_ahead = pause_behind = 0
    entries = {}
    for t, d, s, tok in shown:
        ta = t - off
        if ta < t0 or ta > t1:
            continue
        w = word_index(index, d, s, tok) if d == dua and s is not None and tok is not None else None
        if d == dua and s is not None:
            if not segs or segs[-1][1] != s:
                segs.append((ta, s))
                if s in line_first and s not in entries:
                    entries[s] = ta - line_first[s]
        i = bisect.bisect_right(starts, ta) - 1
        if i >= 0 and tr[i][3]:
            tw = tr[i][0]
            if w is not None:
                offs.append(max(-10, min(10, w - tw)))
            n_line += 1
            line_ok += d == dua and s == int(index.word_segment[tw])
            gap = tr[i + 1][1] - tr[i][2] if i + 1 < len(tr) else np.inf
            if ta > tr[i][2] + 0.5 and gap > 1.0:
                pause_ticks += 1
                pause_ahead += w is not None and w > tw
                pause_behind += w is None or w < tw
        if w is not None and prev_w is not None and (w < prev_w or w > prev_w + 2):
            jerks += 1
        if w is not None:
            prev_w = w
    right_backs = 0
    for k in range(1, len(segs)):
        if segs[k][1] < segs[k - 1][1]:
            backs += 1
            i = bisect.bisect_right(starts, segs[k][0]) - 1
            right_backs += i >= 0 and int(index.word_segment[tr[i][0]]) == segs[k][1]
        if k >= 2 and segs[k][1] == segs[k - 2][1] and segs[k][0] - segs[k - 2][0] <= 3:
            flick += 1
    minutes = (t1 - t0) / 60
    o = np.array(offs) if offs else np.zeros(1)
    ent = [v for v in entries.values()]
    return {"exact": float(np.mean(o == 0)), "pm1": float(np.mean(np.abs(o) <= 1)),
            "behind2": float(np.mean(o <= -2)), "ahead2": float(np.mean(o >= 2)),
            "line": line_ok / max(1, n_line), "jerks": jerks / minutes, "backs": backs / minutes,
            "flick": flick / minutes, "entry": float(np.median(ent)) if ent else float("nan"),
            "early": float(np.mean([e < -0.3 for e in ent])) if ent else float("nan"),
            "pause_ahead": pause_ahead / max(1, pause_ticks), "pause_behind": pause_behind / max(1, pause_ticks),
            "minutes": minutes,
            "n_back": backs, "n_flick": flick, "n_jerk": jerks, "n_back_right": right_backs}


def fmt(name: str, m: dict) -> str:
    return (f"  {name:10s} exact {m['exact']:5.1%}  ±1 {m['pm1']:5.1%}  line {m['line']:5.1%}  jerks/min {m['jerks']:4.1f}  "
            f"line back/min {m['backs']:4.2f}  flicker/min {m['flick']:4.2f}  entry {m['entry']:+5.2f} s  "
            f"early {m['early']:4.0%}  ahead in pauses {m['pause_ahead']:4.0%}  behind in pauses {m['pause_behind']:4.0%}  "
            f"2+ behind {m['behind2']:4.0%}  2+ ahead {m['ahead2']:4.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--ctc", default="models/ctc-student-base")
    ap.add_argument("--hop", type=float, default=0.2)
    ap.add_argument("--window", type=float, default=3.0, help="seconds of audio per CTC window")
    ap.add_argument("--delay", type=float, default=0.15)
    ap.add_argument("--fw", action="append", default=[], help='follower variant "name: k=v,..." (default: round 1)')
    ap.add_argument("--truth-model", default="wav2vec2-quran-dua", help="models/<name>: the forced aligner for word truth")
    ap.add_argument("--tracker", default="", help="TrackerConfig overrides for the rebuilt page, k=v,...")
    args = ap.parse_args()
    tracker_over = {k: json.loads(v) for k, v in (a.split("=") for a in args.tracker.split(",") if a)}
    global TRUTH_MODEL
    TRUTH_MODEL = args.truth_model
    sys.stdout.reconfigure(encoding="utf-8")
    from dua_recognition.align import CorpusIndex
    from dua_recognition.corpus import load_all
    from dua_recognition.ctc_student import load_ctc
    from dua_recognition.follower import FollowerConfig

    duas = load_all()
    index = CorpusIndex(duas)
    paths = [p for r in args.sessions for p in find(r)] if args.sessions else sorted(SESSIONS.glob("*.wav"))
    tag = Path(args.ctc).name
    holder = {}

    def model():
        if "m" not in holder:
            holder["m"] = load_ctc(args.ctc)
        return holder["m"]

    variants = [("follow", FollowerConfig())]
    for v in args.fw:
        name, _, kv = v.partition(":")
        kw = {k: json.loads(x) for k, x in (a.split("=") for a in kv.strip().split(",") if a)}
        variants.append((name.strip(), FollowerConfig(**kw)))
    lanes = ["recorded", "page"] + [n for n, _ in variants]
    tot = {n: [] for n in lanes}
    for path in paths:
        audio, log = read_session(path)
        if log.get("engine") != "device":
            continue
        found = [e["dua"] for e in log["events"] if e["type"] == "dua"]
        dua = log.get("chosen") or (max(set(found), key=found.count) if found else None)
        hops = [e for e in log["events"] if e["type"] == "hop"]
        if dua is None or len(hops) < 10:
            continue
        sid = path.stem
        _, ref_texts = reference(sid, audio, str(ROOT / REF_MODEL))
        ref_segs, ref_ok = reference_lines(duas, dua, ref_texts)
        tr = truth(sid, audio, duas, index, dua, ref_segs, ref_ok)
        if len(tr) < 5:
            continue
        off = audio_offset(log)
        until = len(audio) / SR + off
        frames, lens, times = ctc_frames(sid, audio, model, tag, args.window, args.hop)
        print(f"{sid}  {duas[dua].name_en}  ({len(tr)} words aligned, {sum(w[3] for w in tr)} sure)")
        rs0, live0 = recorded(log)
        shown = {"recorded": display(rs0, duas, until, {}, live0 or None, back_after=1.0 if live0 else None)}
        rs, live = renders(log, audio, duas, index, {"quiet": "speech", "live": 1, "paused": 1, **tracker_over})
        shown["page"] = display(rs, duas, until, {}, live or None, back_after=1.0)
        for name, fcfg in variants:
            shown[name] = follow_display(rs, duas, until, live, frames, lens, times, off, args.delay, fcfg)
        for n in lanes:
            m = score(shown[n], tr, index, dua, off)
            if m:
                tot[n].append(m)
                print(fmt(n, m))
    print(f"\nall {len(tot['page'])} sessions (rates averaged over sessions)")
    for n in lanes:
        ms = tot[n]
        if not ms:
            continue
        avg = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
        print(fmt(n, avg) + f"   (backs {sum(m['n_back'] for m in ms)} [{sum(m['n_back_right'] for m in ms)} right], "
              f"flickers {sum(m['n_flick'] for m in ms)}, "
              f"jerks {sum(m['n_jerk'] for m in ms)})")


if __name__ == "__main__":
    main()
