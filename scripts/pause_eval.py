#!/usr/bin/env python
"""Does the display wait when the reciter stops? A benchmark for pauses.

Professional reciters flow from line to line, so the test recordings hardly
ever stop. People reading along do: they finish a line, breathe, look up, and
the display must not walk on into the next line meanwhile (it did, in the
user's test3.mp4). This inserts a pause after every third line end of each
test recording (room tone copied from the recording's own quietest second),
transcribes the windows that overlap a pause (the rest are the cached ones,
shifted in time), and scores the display word by word against forced-aligned
word timings (scripts/word_truth.py), with the pauses reported separately.

    python scripts/pause_eval.py build --model models/whisper-base-quran-dua-ct2 --tag whisper-base-quran-dua
    python scripts/pause_eval.py score --tag whisper-base-quran-dua "base" "still;cfg still_motion_after=0.3"

A variant is `base[;smooth k=v,...][;cfg k=v,...][;still][;silero]`: `still`
feeds the tracker how long the reciter has been silent (asr.quiet_at_end),
`silero` uses the voice detector alone for that instead of voice-or-energy.

The stop detector as it is now (docs/results/stops.md) needs `quiet2` first:

    python scripts/pause_eval.py quiet2 --tag whisper-base-aug-v4 --split train [--agc 12]

which measures, on the paused audio, the window's quiet with the voice-relative
rule (asr.QuietMeter), the pauses since the last update, and the page's live
detector (asr.LiveQuiet); --agc N turns each inserted pause up by N dB over its
first 1.8 s, as a phone's automatic gain control does once the reciter stops
(stored separately, as "q2@agcN"). Then the flags `q2` (that quiet; `old` the
old rule's on the same audio; `agc` for the turned-up pauses), `paused` (the belief moves only for time not in pauses),
`live` (the tracker hears the live detector at the moment each update is
shown, and the glide stops while it says they are silent) and `back` (after
retreat_after of silence an update may take the highlight back in its line).
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
from dua_recognition.display import Highlight  # noqa: E402
from dua_recognition.tracker import Tracker, TrackerConfig  # noqa: E402

OUT = ROOT / "data" / "cache" / "pause"
SR = 16000
WINDOW = 6.0
MIN_SCORE = -1.5  # forced-alignment line score below which word times aren't trusted


def good_words(rec, ix) -> list[list]:
    words = ev.load_word_truth(rec, ix) or []
    return sorted((w for w in words if w[3] >= MIN_SCORE and w[2] - w[1] < 4.0), key=lambda w: w[1])


def pause_points(words, ix, every: int = 3, min_gap_s: float = 15.0) -> list[tuple[float, int]]:
    """(insert time, last word index) at line ends that run straight into the next line."""
    out, n, last_t = [], 0, -1e9
    for a, b in zip(words, words[1:]):
        if b[0] != a[0] + 1 or ix.word_segment[a[0]] == ix.word_segment[b[0]]:
            continue
        gap = b[1] - a[2]
        if not 0 <= gap < 3.0:
            continue
        n += 1
        if n % every or a[2] - last_t < min_gap_s or a[2] < 10.0:
            continue
        t = a[2] + min(0.2, gap / 2)
        out.append((round(t, 3), a[0]))
        last_t = t
    return out


def room_tone(y: np.ndarray, seconds: float) -> np.ndarray:
    """The recording's quietest second (away from its start and end), tiled."""
    fr = y[: y.size // 512 * 512].reshape(-1, 512)
    db = 10 * np.log10(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)
    k = 31  # ~1 s of frames
    smooth = np.convolve(db, np.ones(k) / k, mode="valid")
    lo, hi = int(10 / 0.032), max(int(10 / 0.032) + 1, smooth.size - int(10 / 0.032))
    i = lo + int(np.argmin(smooth[lo:hi]))
    snip = y[i * 512 : (i + k) * 512]
    reps = int(np.ceil(seconds * SR / snip.size))
    return np.tile(snip, reps)[: int(seconds * SR)].astype(np.float32)


def paused_audio(y: np.ndarray, at: list[float], P: float) -> np.ndarray:
    """`y` with `P` seconds of its own room tone inserted at each time in `at`."""
    tone = room_tone(y, P)
    pieces, prev = [], 0
    for t in at:
        pieces += [y[prev : int(t * SR)], tone]
        prev = int(t * SR)
    pieces.append(y[prev:])
    return np.concatenate(pieces)


def build_ctc(args, out_dir: Path) -> None:
    """CTC posteriors of the paused audio (same pauses as the built set), for the
    word follower: `--ctc-window` s windows every `--ctc-hop` s."""
    from dump_ctc import CtcModel
    from faster_whisper.audio import decode_audio

    recs = {rec.audio_id: rec for dua in ev.load_all().values() for rec in ev.load_recordings(dua)}
    ctag = Path(args.ctc_model).name
    cdir = out_dir / f"ctc-{ctag}"
    cdir.mkdir(parents=True, exist_ok=True)
    m = None
    for f in sorted(out_dir.glob("*.json")):
        out = cdir / f"{f.stem}_w{args.ctc_window:g}_h{args.ctc_hop:g}.npz"
        if out.exists():
            continue
        m = m or __import__('dua_recognition.ctc_student', fromlist=['load_ctc']).load_ctc(args.ctc_model)
        r = json.loads(f.read_text(encoding="utf-8"))
        P = r["pause_s"]
        y = decode_audio(str(recs[r["audio_id"]].path), sampling_rate=SR)
        y2 = paused_audio(y, [s - P * j for j, (s, _) in enumerate(r["pauses"])], P)
        end = r["rows"][-1][0]
        times = [round((k + 1) * args.ctc_hop, 3) for k in range(int((end + 1e-6) // args.ctc_hop))]
        arr, lens, _, _ = m.windows(y2, times, args.ctc_window, 32)
        np.savez(out, lp=arr, n_frames=lens, t=np.array(times))
        print(f"  ctc {f.stem} {len(times)} windows {out.stat().st_size / 1e6:.0f} MB", flush=True)


def build(args) -> None:
    os.environ.setdefault("DUA_ASR_DEVICE", args.device)
    from faster_whisper.audio import decode_audio

    from dua_recognition.asr import _quiet, _tail_activity, transcribe_batch

    duas = ev.load_all()
    ix = ev.CorpusIndex(duas)
    P = args.pause
    out_dir = OUT / (args.tag + ("" if args.split == "test" else f"@{args.split}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.ctc_model:
        build_ctc(args, out_dir)
        return
    for dua in duas.values():
        if dua.id in ev.LINE_LABELS_UNRELIABLE:
            continue
        for rec in ev.load_recordings(dua):
            if ev.is_test(rec.reciter) != (args.split == "test"):
                continue
            out = out_dir / f"{rec.audio_id}.json"
            words = good_words(rec, ix)
            lo = ix.dua_word_span[ix.dua_ids.index(dua.id)][0]
            if out.exists() or len(words) < 20:
                continue
            cached = ev.load_rows(args.tag, rec, WINDOW, 1.0)
            if not cached:
                print(f"  no window cache for {rec.audio_id}; skipped", file=sys.stderr)
                continue
            by_t = {round(t): x for t, x in cached}
            pts = pause_points(words, ix)
            y = decode_audio(str(rec.path), sampling_rate=SR)
            y2 = paused_audio(y, [t for t, _ in pts], P)
            starts2 = [t + P * j for j, (t, _) in enumerate(pts)]  # where each pause begins in y2

            def shift(t: float) -> float:
                return t + P * sum(1 for p, _ in pts if p <= t)

            end2 = shift(min(rec.end_s, words[-1][2] + 2.0))
            times = list(range(1, int(end2) + 1))
            texts, fresh = {}, []
            for t2 in times:
                if any(s < t2 and t2 - WINDOW < s + P for s in starts2):
                    fresh.append(t2)
                    continue
                k = sum(1 for s in starts2 if s + P <= t2 - WINDOW)
                texts[t2] = by_t.get(round(t2 - k * P), "")
            for b in range(0, len(fresh), 16):
                chunk = fresh[b : b + 16]
                wins = [y2[int(max(0, t - WINDOW) * SR) : int(t * SR)] for t in chunk]
                for t, x in zip(chunk, transcribe_batch(wins, model=args.model)):
                    texts[t] = x
            q_mixed, q_silero = [], []
            for t2 in times:
                probs, db, floor = _tail_activity(y2[int(max(0, t2 - WINDOW) * SR) : int(t2 * SR)], 3.0)
                q_mixed.append(3.0 if probs is None else round(_quiet(probs, db, floor, 6.0), 3))
                q_silero.append(3.0 if probs is None else round(_quiet(probs, db, floor, None), 3))
            rec_out = {
                "audio_id": rec.audio_id, "dua": dua.id, "reciter": rec.reciter, "pause_s": P,
                # word numbers count from the du'a's first word, as in data/cache/word_truth
                "pauses": [[s, w - lo] for s, (_, w) in zip(starts2, pts)],
                "rows": [[t, texts[t]] for t in times],
                "quiet": q_mixed, "quiet_silero": q_silero,
                "truth": [[w[0] - lo, round(shift(w[1]), 3), round(shift(w[2]), 3)] for w in words],
            }
            out.write_text(json.dumps(rec_out, ensure_ascii=False), encoding="utf-8")
            print(f"{dua.id:28s} {rec.reciter[:20]:20s} {len(pts):3d} pauses  {len(fresh):5d} new windows", flush=True)


def agc_ramp(n: int, gain_db: float, hold: float = 0.3, rise: float = 1.5) -> np.ndarray:
    """Gain over a pause of n samples: flat for `hold` s, then up by gain_db over `rise` s."""
    t = np.arange(n) / SR
    return (10 ** (np.clip((t - hold) / rise, 0.0, 1.0) * gain_db / 20)).astype(np.float32)


def quiet2(args) -> None:
    """The stop detector's measures on the built paused audio (see the module doc)."""
    from faster_whisper.audio import decode_audio

    from dua_recognition.asr import LiveQuiet, QuietMeter

    recs = {rec.audio_id: rec for dua in ev.load_all().values() for rec in ev.load_recordings(dua)}
    d = OUT / (args.tag + ("" if args.split == "test" else f"@{args.split}"))
    key = "q2" + (f"@agc{args.agc:g}" if args.agc else "")
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if key in r and "quiet_old" in r[key]:
            continue
        P = r["pause_s"]
        y = decode_audio(str(recs[r["audio_id"]].path), sampling_rate=SR)
        y2 = paused_audio(y, [s - P * j for j, (s, _) in enumerate(r["pauses"])], P)
        if args.agc:
            for s, _ in r["pauses"]:
                a = int(s * SR)
                y2[a : a + int(P * SR)] *= agc_ramp(min(int(P * SR), y2.size - a), args.agc)
        meter, old, live = QuietMeter(), QuietMeter(rel_db=None), LiveQuiet()
        quiet, quiet_old, paused, voice = [], [], [], {}
        for t, _ in r["rows"]:
            w = y2[int(max(0, t - WINDOW) * SR) : int(t * SR)]
            quiet.append(round(meter(w, 1.0), 3))
            quiet_old.append(round(old(w), 3))
            paused.append(round(meter.paused, 3))
            voice[t] = meter.voice_db
        series, k, pending = [], 0, sorted(voice)
        chunk = 4000
        for a in range(0, y2.size - chunk + 1, chunk):
            live.push(y2[a : a + chunk])
            now = (a + chunk) / SR
            while k < len(pending) and pending[k] + 1.0 <= now:  # the worker's voice level, a second later
                if voice[pending[k]] is not None:
                    live.voice_db = voice[pending[k]]
                k += 1
            series.append(None if live.quiet is None else round(live.quiet, 3))
        r[key] = {"quiet": quiet, "quiet_old": quiet_old, "paused": paused, "live": series, "live_step": chunk / SR}
        f.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print(f"  {key} {f.stem}", flush=True)


def parse_variant(v: str):
    parts = v.split(";")
    cfg, smooth, flags = TrackerConfig(), {"ease": 1.0, "speed_scale": 1.2}, set()
    for part in parts:
        kind, _, kv = part.partition(" ")
        kw = {k: float(x) for k, x in (a.split("=") for a in kv.split(",") if a)}
        if kind == "smooth":
            smooth = kw
        elif kind == "cfg":
            cfg = replace(cfg, **{k: (bool(x) if isinstance(getattr(cfg, k), bool) else x) for k, x in kw.items()})
        else:
            flags.add(kind)
    return cfg, smooth, flags


def score(ix, data, cfg, smooth, flags, delay: float) -> dict:
    offs, jerks, minutes = [], 0, 0.0
    pause_offs, pause_next, pause_ahead2, lags = [], [], [], []
    for r in data:
        truth = r["truth"]
        starts = [w[1] for w in truth]
        m2 = r.get("q2@agc12" if "agc" in flags else "q2") if flags & {"q2", "paused", "live", "old"} else None
        if flags & {"q2", "paused", "live", "old"} and m2 is None:
            raise SystemExit("run `pause_eval.py quiet2` first (with --agc 12 for `agc`)")
        if "old" in flags:  # the old rule (floor + 6 dB) on the same audio (with `agc`: turned-up pauses)
            quiet = m2["quiet_old"]
        elif "q2" in flags:
            quiet = m2["quiet"]
        else:
            quiet = r["quiet_silero" if "silero" in flags else "quiet"] if "still" in flags else [0.0] * len(r["rows"])
        paused = m2["paused"] if "paused" in flags else [None] * len(r["rows"])

        def live_at(t: float):
            if "live" not in flags:
                return None
            i = int(t / m2["live_step"]) - 1  # the chunks the page holds by then
            return m2["live"][i] if 0 <= i < len(m2["live"]) else None

        tr = Tracker(ix, cfg)
        hl = Highlight(**smooth)
        ups = []
        for (t, _), c, n, q, pz in zip(r["rows"], r["costs"], r["letters"], quiet, paused):
            qn = live_at(t + delay)
            p = tr.update_costs(c, 1.0, delay + cfg.display_lead, n_letters=n, quiet=q, quiet_now=qn, paused=pz)
            rng = None
            if p.word is not None:
                lo, hi = ix.dua_word_span[ix.word_dua[p.word]]
                ws = np.flatnonzero(ix.word_segment[lo:hi] == p.segment) + lo
                rng = (int(ws[0]), int(ws[-1]) + 1)
            still = q > cfg.still_after or (qn is not None and qn > cfg.still_after)
            back = "back" in flags and max(q, qn or 0.0) > cfg.retreat_after
            ups.append((t + delay, p.dua, p.segment, p.word, rng, 0.0 if still else tr.speed, back))
        k, prev = 0, None
        ticks = np.arange(truth[0][1], r["rows"][-1][0] + delay, 0.1)
        pauses = r["pauses"]
        shown_at = []
        stopped, speed_now = False, 0.0
        for tick in ticks:
            if "live" in flags and hl.line is not None:
                qn = live_at(tick)
                now_still = qn is not None and qn > cfg.still_after
                if now_still != stopped:
                    stopped = now_still
                    hl.pace(tick, 0.0 if stopped else speed_now)
            while k < len(ups) and ups[k][0] <= tick:
                _, dua, seg, word, rng, speed, back = ups[k]
                speed_now = speed
                hl.update(ups[k][0], dua, seg, word, rng or (0, 0), 0.0 if stopped else speed, back=back)
                k += 1
            shown = hl.word(tick) if hl.line and hl.line[0] == r["dua"] else None
            shown_at.append(shown)
            i = bisect.bisect_right(starts, tick) - 1
            if i < 0 or shown is None:
                continue
            tw = truth[i][0]
            off = max(-10, min(10, shown - tw))
            in_pause = any(s + 0.5 <= tick < s + r["pause_s"] for s, _ in pauses)
            if in_pause:
                pause_offs.append(off)
                pause_next.append(ix.word_segment[shown] != ix.word_segment[tw] and shown > tw)
                pause_ahead2.append(off >= 2)
            else:
                offs.append(off)
            if prev is not None and (shown < prev or shown > prev + 2):
                jerks += 1
            prev = shown
        for s, w in pauses:
            # After the pause: how long until the next line's first word is on screen?
            resume = s + r["pause_s"]
            lag = None
            for tick, shown in zip(ticks, shown_at):
                if tick < resume:
                    continue
                if tick > resume + 6:
                    break
                if shown is not None and shown > w and ix.word_segment[shown] != ix.word_segment[w]:
                    lag = tick - resume
                    break
            lags.append(lag)
        minutes += len(ticks) / 600
    o, po = np.array(offs), np.array(pause_offs)
    found = [x for x in lags if x is not None]
    return {
        "exact": float(np.mean(o == 0)), "pm1": float(np.mean(np.abs(o) <= 1)), "mean": float(o.mean()),
        "jerks_min": jerks / max(minutes, 1e-9),
        "pause_next_line": float(np.mean(pause_next)), "pause_ahead2": float(np.mean(pause_ahead2)),
        "pause_mean": float(po.mean()), "pause_exact": float(np.mean(po == 0)),
        "resume_lag": float(np.median(found)) if found else float("nan"),
        "resume_missed": 1 - len(found) / max(1, len(lags)), "n_pauses": len(lags),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model")
    b.add_argument("--tag", required=True, help="window-cache tag of the same model")
    b.add_argument("--pause", type=float, default=4.0, help="seconds (whole seconds keep the 1 s hop grid)")
    b.add_argument("--device", default="cpu")
    b.add_argument("--split", choices=["test", "train"], default="test", help="train: for tuning (tag gets @train)")
    b.add_argument("--ctc-model", help="instead: dump this CTC model's posteriors over the already built paused audio")
    b.add_argument("--ctc-window", type=float, default=3.0)
    b.add_argument("--ctc-hop", type=float, default=0.2)
    q = sub.add_parser("quiet2")
    q.add_argument("--tag", required=True)
    q.add_argument("--split", choices=["test", "train"], default="test")
    q.add_argument("--agc", type=float, default=0.0, help="dB the inserted pauses rise by (a phone's AGC)")
    s = sub.add_parser("score")
    s.add_argument("--tag", required=True)
    s.add_argument("--delay", type=float, default=0.5)
    s.add_argument("--split", choices=["test", "train"], default="test")
    s.add_argument("variants", nargs="+")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.cmd == "build":
        build(args)
        return
    if args.cmd == "quiet2":
        quiet2(args)
        return
    ix = ev.CorpusIndex(ev.load_all())
    d = OUT / (args.tag + ("" if args.split == "test" else f"@{args.split}"))
    data = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob("*.json"))]
    from dua_recognition.align import encode

    for r in data:  # align once, score many variants
        lo = ix.dua_word_span[ix.dua_ids.index(r["dua"])][0]
        r["truth"] = [[lo + w, a, b] for w, a, b in r["truth"]]
        r["pauses"] = [[s, lo + w] for s, w in r["pauses"]]
        r["costs"] = ev.LazyCosts(ix, [x for _, x in r["rows"]])  # aligned as read: 500 texts are too big to hold
        r["letters"] = [len(encode(x)) if x else 0 for _, x in r["rows"]]
    print(f"{len(data)} recordings, {sum(len(r['pauses']) for r in data)} pauses, ASR {args.tag}, delay {args.delay:g} s")
    print("variant | word exact | ±1 | mean off | jerks/min || in pause: next line shown | ≥2 words ahead | mean off || resume lag (missed)")
    for v in args.variants:
        cfg, smooth, flags = parse_variant(v)
        m = score(ix, data, cfg, smooth, flags, args.delay)
        print(f"{v:50s} | {m['exact']:.1%} | {m['pm1']:.1%} | {m['mean']:+.2f} | {m['jerks_min']:.1f} || "
              f"{m['pause_next_line']:.1%} | {m['pause_ahead2']:.1%} | {m['pause_mean']:+.2f} || "
              f"{m['resume_lag']:.1f} s ({m['resume_missed']:.0%})", flush=True)


if __name__ == "__main__":
    main()
