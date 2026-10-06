#!/usr/bin/env python
"""Turn harvested recordings (scripts/harvest.py) into labelled training audio.

Nothing about a harvested upload is trusted except its audio. Per recording:

  frames    (GPU) one pass of the CTC teacher (wav2vec2-quran-dua-voices) over
            the whole file: folded letter log posteriors + greedy token ids,
            50 frames/s, data/harvest/frames/<platform>/<id>.npz.
  identify  (CPU) greedy text of 6 s windows every 2 s, scored against all
            corpus texts (align.CorpusIndex, Myers bit-parallel); a Viterbi
            pass over {nothing, text_1..text_N} with a switching penalty cuts
            the recording into spans of one du'a (or none: lecture, music,
            translation, a text the corpus doesn't have).
  align     (CPU) each span through the offline smoother (offline.py, the
            labeller validated against DuaPlayer's human timings), then CTC
            forced alignment of every placed line inside its time slot for
            word timings; lines whose forced path scores badly are dropped.

Output: data/harvest/labels/<platform>/<id>.json, one entry per span with
line and word timings and quality numbers, ready for the training-set builder.

    python scripts/harvest_label.py frames            # loops over new downloads
    python scripts/harvest_label.py identify align
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

HARVEST = ROOT / "data" / "harvest"
FRAMES = HARVEST / "frames"
LABELS = HARVEST / "labels"
PLATFORMS = ("yt", "aparat", "archive", "web", "soundcloud")
TEACHER = ROOT / "models" / "wav2vec2-quran-dua-voices"
SR = 16000
FPS = 50  # wav2vec2: one frame per 320 samples


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def recordings(platforms=PLATFORMS) -> list[tuple[str, dict, Path]]:
    """(platform, meta, audio path) for every downloaded recording: download order
    within a platform, platforms interleaved so none waits for another to finish."""
    per = []
    for p in platforms:
        root = HARVEST / p
        rows, seen = [], set()
        for meta in read_jsonl(root / "index.jsonl"):
            audio = root / "audio" / meta["audio"]
            if audio.exists() and meta["id"] not in seen:
                seen.add(meta["id"])
                rows.append((p, meta, audio))
        per.append(rows)
    out = []
    for i in range(max((len(r) for r in per), default=0)):
        out += [r[i] for r in per if i < len(r)]
    return out


# ---------------------------------------------------------------- frames
class Teacher:
    def __init__(self, model_dir: Path, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        from dua_recognition.ctc import LOG_FLOOR, fold_matrix

        self.torch, self.device, self.floor = torch, device, LOG_FLOOR
        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForCTC.from_pretrained(model_dir).to(device).eval().half()
        vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.proc.tokenizer.pad_token_id
        self.fold = torch.tensor(fold_matrix(vocab, self.blank), device=device)
        self.inv = {i: t for t, i in vocab.items()}

    def run(self, y: np.ndarray, chunk_s: float = 30.0, ctx_s: float = 1.0, batch: int = 8):
        """Folded log posteriors [T, N_COLS] float16 and argmax ids [T] uint8 for all of y.
        Chunks of chunk_s seconds with ctx_s of context either side; only each
        chunk's own frames are kept, so the result is continuous."""
        torch = self.torch
        n_frames = len(y) // 320
        step, ctx = int(chunk_s * SR), int(ctx_s * SR)
        starts = list(range(0, len(y), step))
        lp_out = np.full((n_frames, self.fold.shape[1]), self.floor, dtype=np.float16)
        ids_out = np.full(n_frames, self.blank, dtype=np.uint8)
        with torch.no_grad():
            for b in range(0, len(starts), batch):
                chunk, offs = [], []
                for s in starts[b : b + batch]:
                    lo = max(0, s - ctx)
                    chunk.append(y[lo : min(len(y), s + step + ctx)])
                    offs.append((s, lo))
                inp = self.proc(chunk, sampling_rate=SR, return_tensors="pt", padding=True)
                vals = inp.input_values.to(self.device).half()
                mask = inp.get("attention_mask")
                logits = self.model(vals, attention_mask=mask.to(self.device) if mask is not None else None).logits.float()
                folded = torch.log((logits.softmax(-1) @ self.fold).clamp_min(float(np.exp(self.floor))))
                folded = folded.half().cpu().numpy()
                ids = logits.argmax(-1).to(torch.uint8).cpu().numpy()
                for k, (s, lo) in enumerate(offs):
                    f0 = (s - lo) // 320  # frames of left context to skip
                    g0 = s // 320
                    g1 = min(n_frames, (s + step) // 320)
                    n = min(g1 - g0, folded.shape[1] - f0)
                    if n > 0:
                        lp_out[g0 : g0 + n] = folded[k, f0 : f0 + n]
                        ids_out[g0 : g0 + n] = ids[k, f0 : f0 + n]
        return lp_out, ids_out


def frames_path(platform: str, meta: dict) -> Path:
    return FRAMES / platform / f"{meta['id']}.npz"


def decode_pcm(path) -> np.ndarray:
    """16 kHz mono float32 of the whole file by the ffmpeg CLI, the decoder the clip cutters use.
    PyAV (faster_whisper's decode_audio) stopped early in some archive mp3s and aparat m4as (2026-10-04)."""
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True)
    if p.returncode != 0 and not p.stdout:
        raise RuntimeError(p.stderr.decode(errors="replace")[-200:])
    return np.frombuffer(p.stdout, np.float32).copy()


def ts_span(path) -> float | None:
    """The container's timestamp span (ffprobe), which ffmpeg -ss seeks within. Early yt downloads repeat
    a chunk, so they hold more samples than this span; windows past it cut to nothing."""
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return None


def cmd_frames(args) -> None:
    teacher = Teacher(Path(args.model))
    vocab_ids = {t: i for i, t in teacher.inv.items()}
    (FRAMES / "vocab.json").parent.mkdir(parents=True, exist_ok=True)
    (FRAMES / "vocab.json").write_text(json.dumps(vocab_ids, ensure_ascii=False), encoding="utf-8")
    bad = {r["id"] for r in read_jsonl(FRAMES / "failed.jsonl")}
    idle = 0
    while True:
        todo = [(p, m, a) for p, m, a in recordings(args.platforms)
                if not frames_path(p, m).exists() and m["id"] not in bad]
        if args.limit:
            todo = todo[: args.limit]
        if not todo:
            if not args.follow or idle > args.idle_exit:
                break
            idle += 1
            time.sleep(60)
            continue
        idle = 0
        # Decode the next file on a thread while the GPU works on this one.
        from concurrent.futures import ThreadPoolExecutor

        def _decode(path):
            try:
                return decode_pcm(path), None
            except Exception as e:  # noqa: BLE001
                return None, e

        pool = ThreadPoolExecutor(1)
        nxt = pool.submit(_decode, todo[0][2])
        for i, (p, meta, audio) in enumerate(todo):
            t0 = time.time()
            y, err = nxt.result()
            if i + 1 < len(todo):
                nxt = pool.submit(_decode, todo[i + 1][2])
            if err is not None:
                with (FRAMES / "failed.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": meta["id"], "platform": p, "err": str(err)[:200]}) + "\n")
                bad.add(meta["id"])
                continue
            if len(y) < SR * 5:
                bad.add(meta["id"])
                continue
            lp, ids = teacher.run(y, batch=args.batch)
            out = frames_path(p, meta)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.npz")
            np.savez(tmp, lp=lp, ids=ids)
            tmp.replace(out)
            dt = time.time() - t0
            print(f"{p}/{meta['id']}: {len(y) / SR / 60:5.1f} min in {dt:4.1f} s ({len(y) / SR / max(dt, 1e-3):.0f}x)",
                  flush=True)
        if args.limit:
            break


# ---------------------------------------------------------------- identify + align
WIN, HOP = 6.0, 2.0
MIN_LETTERS = 8      # windows with fewer greedy letters carry no evidence
NONE_COST = 0.42     # per-letter cost of the "nothing from the corpus" state
SWITCH = 2.5         # Viterbi penalty for changing text
MIN_SPAN = 6         # voiced windows (12 s) for a span to count
LINE_MAX = 0.40      # placed_mask threshold (align_offline.py)
FORCE_MAX = 1.5      # forced alignment: max mean -log p per letter for a line to keep


class Vocab:
    def __init__(self):
        v = json.loads((FRAMES / "vocab.json").read_text(encoding="utf-8"))
        self.inv = {i: t for t, i in v.items()}
        self.blank = v.get("<pad>", 0)

    def window_texts(self, ids: np.ndarray, times: list[float]) -> list[str]:
        """Greedy CTC text of the WIN-second window ending at each time."""
        keep = (ids != self.blank) & np.r_[True, ids[1:] != ids[:-1]]
        pos = np.flatnonzero(keep)
        toks = ids[pos]
        out = []
        for t in times:
            a = np.searchsorted(pos, int(max(0.0, t - WIN) * FPS))
            b = np.searchsorted(pos, int(t * FPS))
            out.append("".join(self.inv[int(i)] for i in toks[a:b]).replace("|", " ").strip())
        return out


_STATE: dict = {}


EXTRA = HARVEST / "extra_texts.json"  # scripts/mafatih_texts.py: training-only texts


def load_extra() -> dict:
    """Training-only texts (Mafatih blocks the corpus lacks) as Dua objects."""
    from dua_recognition.corpus import Dua, Segment

    if not EXTRA.exists():
        return {}
    out = {}
    for d in json.loads(EXTRA.read_text(encoding="utf-8")):
        out[d["dua_id"]] = Dua(d["dua_id"], d.get("dua_name_en", ""), d.get("dua_name_ar", ""),
                               [Segment(s["segment_id"], s["arabic"]) for s in d["segments"]])
    return out


def _init_worker() -> None:
    from dua_recognition.align import CorpusIndex
    from dua_recognition.corpus import load_all

    duas = load_all()
    ix = CorpusIndex(duas)
    starts = np.array([a for a, _ in ix.dua_word_span])
    _STATE.update(duas=duas, ix=ix, starts=starts, vocab=Vocab())
    extra = load_extra()
    if extra:
        xix = CorpusIndex(extra)
        _STATE.update(xduas=extra, xix=xix, xstarts=np.array([a for a, _ in xix.dua_word_span]))


def viterbi(E: np.ndarray, switch: float) -> np.ndarray:
    """Cheapest state path through per-window costs E [T, S] with a fixed switching cost."""
    T, S = E.shape
    cost = E[0].copy()
    back = np.zeros((T, S), dtype=np.int32)
    for t in range(1, T):
        j = int(np.argmin(cost))
        move = cost[j] + switch
        stay = cost <= move
        back[t] = np.where(stay, np.arange(S), j)
        cost = np.where(stay, cost, move) + E[t]
    path = np.empty(T, dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def identify(texts: list[str], ix=None, starts=None, only=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per window: state path (0 = none, k = du'a k-1), best per-letter cost, letters.
    `only` (bool per window): score just those windows; the rest are pinned to none."""
    from dua_recognition.align import encode

    ix = ix if ix is not None else _STATE["ix"]
    starts = starts if starts is not None else _STATE["starts"]
    n = len(texts)
    E = np.zeros((n, len(starts) + 1), dtype=np.float32)
    best = np.full(n, np.nan, dtype=np.float32)
    nlet = np.zeros(n, dtype=np.int32)
    for t, text in enumerate(texts):
        k = encode(text).size if text else 0
        nlet[t] = k
        if only is not None and not only[t]:
            E[t, 1:] = 1.0  # taken by the first pass: nothing else may claim it
            continue
        if k < MIN_LETTERS:
            continue  # silence / too little: every state equally likely
        c = ix.word_costs(text)
        per = np.minimum.reduceat(c.astype(np.float32), starts) / k
        E[t, 1:] = np.minimum(per, 1.0)
        E[t, 0] = NONE_COST
        best[t] = per.min()
    return viterbi(E, SWITCH), best, nlet


def _force_align_local(lp, r):
    """CTC Viterbi of letters r (J) inside frames lp (T x C) where the audio before the
    first letter and after the last is free (neighbouring lines, silence). States:
    0 = before, 1..2J+1 = CTC (odd = blank, even = letter (s-2)//2), 2J+2 = after.
    Returns (score of the line's own frames, first/last emission frame per letter)."""
    T = lp.shape[0]
    J = r.size
    S = 2 * J + 3
    NEGI = -1e18
    dp = np.full((T, S), NEGI)
    bp = np.zeros((T, S), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2 (or before->letter 0)
    dp[0, 0] = 0.0
    dp[0, 1] = lp[0, 0]
    dp[0, 2] = lp[0, r[0]]
    for t in range(1, T):
        dp[t, 0] = dp[t - 1, 0]
        for s in range(1, S - 1):
            best = dp[t - 1, s]
            arg = 0
            if dp[t - 1, s - 1] > best:
                best = dp[t - 1, s - 1]
                arg = 1
            if s == 2 and dp[t - 1, 0] > best:  # before -> first letter, skipping the blank
                best = dp[t - 1, 0]
                arg = 2
            elif s >= 4 and s % 2 == 0 and r[(s - 2) // 2] != r[(s - 4) // 2] and dp[t - 1, s - 2] > best:
                best = dp[t - 1, s - 2]
                arg = 2
            e = lp[t, 0] if s % 2 == 1 else lp[t, r[(s - 2) // 2]]
            dp[t, s] = best + e
            bp[t, s] = arg
        # after: from the last letter or the final blank, then free
        a = dp[t - 1, S - 1]
        arg = 0
        if dp[t - 1, S - 2] > a:
            a = dp[t - 1, S - 2]
            arg = 1
        if dp[t - 1, S - 3] > a:
            a = dp[t - 1, S - 3]
            arg = 2
        dp[t, S - 1] = a
        bp[t, S - 1] = arg
    # best end: in "after", or still on the last letter / final blank at T-1
    s = S - 1
    if dp[T - 1, S - 2] > dp[T - 1, s]:
        s = S - 2
    if dp[T - 1, S - 3] > dp[T - 1, s]:
        s = S - 3
    score = dp[T - 1, s]
    first = np.full(J, -1, dtype=np.int32)
    last = np.full(J, -1, dtype=np.int32)
    for t in range(T - 1, -1, -1):
        if s >= 2 and s <= S - 2 and s % 2 == 0:
            j = (s - 2) // 2
            first[j] = t
            if last[j] < 0:
                last[j] = t
        if t == 0:
            break
        a = bp[t, s]
        if s == S - 1:
            s = S - 1 - a  # 0 stay, 1 from final blank, 2 from last letter
        elif s == 2 and a == 2:
            s = 0
        elif s == 0:
            s = 0
        else:
            s -= a
    return score, first, last


try:
    from numba import njit

    force_align_local = njit(cache=True)(_force_align_local)
except ImportError:  # pragma: no cover
    force_align_local = _force_align_local


def force_lines(lp: np.ndarray, dua, starts: list[tuple[float, int]], t_end: float) -> list[dict]:
    """Word timings for each placed line: CTC forced alignment of the line's letters
    inside [its start - 1 s, the next line's start + 1 s]."""
    from dua_recognition.align import encode
    from dua_recognition.text import normalize

    segs = {s.id: s for s in dua.segments}
    out = []
    for i, (t0, sid) in enumerate(starts):
        t1 = starts[i + 1][0] if i + 1 < len(starts) else min(t_end, t0 + 30.0)
        a, b = int(max(0.0, t0 - 1.0) * FPS), int(min(t_end, t1 + 1.0) * FPS)
        seg = segs.get(sid)
        if seg is None or b - a < 10:
            continue
        words = [w for tok in seg.arabic.split() for w in normalize(tok).split() if encode(w).size]
        if not words:
            continue
        codes = [encode(w) for w in words]
        r = np.concatenate(codes).astype(np.int64)
        if b - a < r.size + 2:
            continue
        score, first, last = force_align_local(lp[a:b].astype(np.float64), r)
        if not np.isfinite(score) or score < -1e8:
            continue
        per = -score / r.size
        bounds, k = [], 0
        for c in codes:
            f0, f1 = first[k], last[k + c.size - 1]
            bounds.append([round((a + f0) / FPS, 2), round((a + f1 + 1) / FPS, 2)])
            k += c.size
        out.append({"seg": int(sid), "start": bounds[0][0], "end": bounds[-1][1], "cost": round(float(per), 3),
                    "ok": bool(per <= FORCE_MAX), "words": words, "times": bounds})
    return out


def label_one(job: tuple[str, str, str]) -> dict:
    """Identify and align one recording from its frames file."""
    from align_offline import placed_mask, starts_from
    from dua_recognition.align import CorpusIndex
    from dua_recognition.offline import align_recording

    platform, vid, path = job
    z = np.load(path)
    lp, ids = z["lp"], z["ids"]
    dur = ids.size / FPS
    times = [float(t) for t in np.arange(WIN, dur + 1e-6, HOP)] or [dur]
    texts = _STATE["vocab"].window_texts(ids, times)
    path_, best, nlet = identify(texts)

    def spans_of(path_, best, duas, ix, extra=False):
        out = []
        t = 0
        while t < len(times):
            u = t
            while u + 1 < len(times) and path_[u + 1] == path_[t]:
                u += 1
            k = int(path_[t])
            voiced = [i for i in range(t, u + 1) if nlet[i] >= MIN_LETTERS]
            if k and len(voiced) >= MIN_SPAN:
                dua = duas[ix.dua_ids[k - 1]]
                sub = CorpusIndex({dua.id: dua})
                rows = [(times[i], texts[i]) for i in range(t, u + 1)]
                costs = [sub.word_costs(x) if (x and nlet[i] >= MIN_LETTERS) else None
                         for i, (_, x) in zip(range(t, u + 1), rows)]
                segs = align_recording(sub, costs, HOP)
                placed = placed_mask(sub, rows, costs, LINE_MAX)
                st = starts_from([r[0] for r in rows], segs, placed)
                vflags = [p for (_, x), p in zip(rows, placed) if x]
                coverage = sum(vflags) / max(1, len(vflags))
                lines = force_lines(lp, dua, st, times[u])
                n_ok = sum(1 for ln in lines if ln["ok"])
                out.append({
                    "dua": dua.id, "t0": round(max(0.0, times[t] - WIN), 2), "t1": round(times[u], 2),
                    "mean_cost": round(float(np.nanmean(best[t:u + 1])), 3), "placed": round(coverage, 3),
                    "lines_found": len(st), "lines_total": len(dua.segments), "lines_ok": n_ok,
                    "usable": bool(coverage >= 0.5 and n_ok >= 3), "lines": lines, "extra": extra})
            t = u + 1
        return out

    spans = spans_of(path_, best, _STATE["duas"], _STATE["ix"])
    # Second pass: what the corpus couldn't place, against the training-only texts.
    free = (path_ == 0) & (nlet >= MIN_LETTERS)
    if "xix" in _STATE and free.sum() >= MIN_SPAN:
        xpath, xbest, _ = identify(texts, _STATE["xix"], _STATE["xstarts"], only=free)
        spans += spans_of(xpath, xbest, _STATE["xduas"], _STATE["xix"], extra=True)
        spans.sort(key=lambda s: s["t0"])
    return {"id": vid, "platform": platform, "duration": round(dur, 2), "windows": len(times),
            "voiced_windows": int((nlet >= MIN_LETTERS).sum()), "spans": spans,
            "matched_s": round(sum(sp["t1"] - sp["t0"] for sp in spans), 1),
            "usable_s": round(sum(ln["end"] - ln["start"] for sp in spans if sp["usable"]
                                  for ln in sp["lines"] if ln["ok"]), 1)}


def _label_safe(job: tuple[str, str, str]) -> dict:
    """label_one, except that a frames file that won't load (cut short by a crash) is
    reported for its frames to be redone instead of stopping the pool."""
    try:
        with np.load(job[2]) as z:
            _ = z["lp"].shape, z["ids"].shape
    except Exception as e:  # noqa: BLE001
        return {"platform": job[0], "id": job[1], "path": job[2], "error": f"unreadable frames: {str(e)[:120]}"}
    return label_one(job)


def cmd_align(args) -> None:
    from multiprocessing import Pool

    jobs = []
    for p in args.platforms:
        for f in sorted((FRAMES / p).glob("*.npz")):
            if f.name.endswith(".tmp.npz"):
                continue
            out = LABELS / p / f"{f.stem}.json"
            if out.exists() and not args.redo:
                continue
            jobs.append((p, f.stem, str(f)))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} recordings to label", flush=True)
    t0, tot = time.time(), {"dur": 0.0, "usable": 0.0, "n": 0}
    with Pool(args.workers, initializer=_init_worker) as pool:
        for res in pool.imap_unordered(_label_safe, jobs):
            if "error" in res:  # a frames file cut short (the 2026-10-02 crash): redo its frames
                print(f"{res['platform']}/{res['id']}: {res['error']}", flush=True)
                Path(res["path"]).unlink(missing_ok=True)
                continue
            out = LABELS / res["platform"] / f"{res['id']}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            tot["dur"] += res["duration"]
            tot["usable"] += res["usable_s"]
            tot["n"] += 1
            sp = ", ".join(f"{s['dua']}({s['t1'] - s['t0']:.0f}s,{'ok' if s['usable'] else 'x'})"
                           for s in res["spans"][:3])
            print(f"{res['platform']}/{res['id']}: {res['duration'] / 60:.1f} min, "
                  f"usable {res['usable_s'] / 60:.1f} min  {sp}", flush=True)
            if tot["n"] % 25 == 0:
                print(f"== {tot['n']} done, {tot['dur'] / 3600:.1f} h in, {tot['usable'] / 3600:.1f} h usable, "
                      f"{(time.time() - t0) / 60:.0f} min", flush=True)
    print(f"== total {tot['n']}, {tot['dur'] / 3600:.1f} h in, {tot['usable'] / 3600:.1f} h usable", flush=True)


# ---------------------------------------------------------------- captions
def parse_vtt(path: Path) -> list[tuple[float, float, str]]:
    import re

    out = []
    for blk in path.read_text(encoding="utf-8", errors="replace").replace("\r", "").split("\n\n"):
        m = re.search(r"(\d+):(\d+):([\d.]+) --> (\d+):(\d+):([\d.]+)[^\n]*\n(.+)", blk, re.S)
        if not m:
            m2 = re.search(r"(\d+):([\d.]+) --> (\d+):([\d.]+)[^\n]*\n(.+)", blk, re.S)
            if not m2:
                continue
            a, b, txt = int(m2[1]) * 60 + float(m2[2]), int(m2[3]) * 60 + float(m2[4]), m2[5]
        else:
            a = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
            b = int(m[4]) * 3600 + int(m[5]) * 60 + float(m[6])
            txt = m[7]
        txt = re.sub(r"<[^>]+>", "", txt).replace("\n", " ").strip()
        if txt:
            out.append((a, b, txt))
    return out


def caption_one(job: tuple[str, str, str, str]) -> dict:
    """Human caption cues, each force-aligned inside [cue start - 0.6 s, cue end + 0.6 s]."""
    from dua_recognition.align import encode
    from dua_recognition.text import normalize

    platform, vid, frames, vtt = job
    lp = np.load(frames)["lp"]
    T = lp.shape[0]
    lines = []
    for i, (a_s, b_s, txt) in enumerate(parse_vtt(Path(vtt))):
        tokens = txt.split()
        words, toks = [], []
        for ti, tok in enumerate(tokens):
            for w in normalize(tok).split():
                if encode(w).size:
                    words.append(w)
                    toks.append(tok)
        if not words:
            continue
        codes = [encode(w) for w in words]
        r = np.concatenate(codes).astype(np.int64)
        a, b = int(max(0.0, a_s - 0.6) * FPS), int(min(T / FPS, b_s + 0.6) * FPS)
        if b - a < r.size + 2:
            continue
        score, first, last = force_align_local(lp[a:b].astype(np.float64), r)
        if not np.isfinite(score) or score < -1e8:
            continue
        per = -score / r.size
        bounds, k = [], 0
        for c in codes:
            bounds.append([round((a + first[k]) / FPS, 2), round((a + last[k + c.size - 1] + 1) / FPS, 2)])
            k += c.size
        lines.append({"seg": i, "start": bounds[0][0], "end": bounds[-1][1], "cost": round(float(per), 3),
                      "ok": bool(per <= FORCE_MAX), "words": words, "tokens": toks, "times": bounds,
                      "cue": [a_s, b_s]})
    n_ok = sum(ln["ok"] for ln in lines)
    return {"id": vid, "platform": platform, "cues": len(lines), "cues_ok": n_ok,
            "usable": bool(n_ok >= 5 and n_ok >= 0.3 * max(1, len(lines))), "lines": lines,
            "usable_s": round(sum(ln["end"] - ln["start"] for ln in lines if ln["ok"]), 1)}


def cmd_captions(args) -> None:
    from multiprocessing import Pool

    jobs = []
    for p in ("yt",):
        for vtt in sorted((HARVEST / p / "audio").glob("*.vtt")):
            vid = vtt.name.split(".")[0]
            fr = FRAMES / p / f"{vid}.npz"
            out = LABELS / p / f"{vid}.captions.json"
            if fr.exists() and (args.redo or not out.exists()):
                jobs.append((p, vid, str(fr), str(vtt)))
    print(f"{len(jobs)} captioned recordings", flush=True)
    tot = 0.0
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(caption_one, jobs):
            out = LABELS / res["platform"] / f"{res['id']}.captions.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            tot += res["usable_s"] if res["usable"] else 0
            print(f"{res['id']}: {res['cues_ok']}/{res['cues']} cues ok, {res['usable_s'] / 60:.1f} min"
                  f"{'' if res['usable'] else '  (unusable)'}", flush=True)
    print(f"== captions usable {tot / 3600:.2f} h", flush=True)


# ---------------------------------------------------------------- export
HELD_OUT_DUAS = frozenset({"dua-tawassul", "ziyarat-ashura"})  # build_finetune_set.HELD_OUT_DUAS


def _meta_index() -> dict[tuple[str, str], tuple[dict, Path]]:
    out = {}
    for p in PLATFORMS:
        for meta in read_jsonl(HARVEST / p / "index.jsonl"):
            out[(p, meta["id"])] = (meta, HARVEST / p / "audio" / meta["audio"])
    return out


def _reciter(platform: str, meta: dict) -> str:
    return f"{platform}:{meta.get('channel') or meta.get('channel_id') or meta['id']}"


def _token_lists(dua) -> dict[int, list[str]]:
    """Per segment: the original token behind each normalized word (CorpusIndex order)."""
    from dua_recognition.align import encode
    from dua_recognition.text import normalize

    out = {}
    for seg in dua.segments:
        toks = []
        for tok in seg.arabic.split():
            for w in normalize(tok).split():
                if encode(w).size:
                    toks.append(tok)
        out[seg.id] = toks
    return out


def _windows(lines: list[dict], toks_of, win: float, hop: float, max_gap: float = 4.0):
    """6 s windows over runs of consecutive good lines; text = words whose middle is inside."""
    from build_finetune_set import whisper_style

    good = [ln for ln in lines if ln["ok"]]
    good.sort(key=lambda ln: ln["start"])
    runs, cur = [], []
    for ln in good:
        if cur and (ln["start"] - cur[-1]["end"] > max_gap or ln["start"] < cur[-1]["end"] - 0.5):
            runs.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        runs.append(cur)
    for run in runs:
        words = []
        for ln in run:
            toks = toks_of(ln)
            for (a, b), tok in zip(ln["times"], toks):
                words.append(((a + b) / 2, whisper_style(tok)))
        t0, t1 = run[0]["start"] - 0.3, run[-1]["end"] + 0.3
        if t1 - t0 < 2.0:
            continue
        starts = list(np.arange(t0, max(t0, t1 - win) + 1e-6, hop)) or [t0]
        for s in starts:
            e = min(s + win, t1)
            text = " ".join(w for m, w in words if s <= m < e and w)
            if len(text) >= 4:
                yield round(float(s), 2), round(float(e), 2), text


def cmd_export(args) -> None:
    import hashlib

    from dua_recognition.corpus import load_all

    duas = {**load_all(), **load_extra()}  # extra: training-only texts (mafatih_texts.py)
    toks = {}
    metas = _meta_index()
    rows, stats = [], {"h": 0.0}
    per_platform: dict[str, float] = {}
    per_dua: dict[str, float] = {}
    voices = _voice_flags()
    held = _held_out_grams(duas)
    held_ids = {d for d in duas if _held_share(_dua_words(duas[d]), held) >= 0.5}  # same text, other ids
    dupes = _duplicates(metas)
    held_voice, hv_stats = _voice_holdout(metas, args.val_pct)
    print(f"voice hold-out: {hv_stats}", flush=True)
    skipped = {"test voice": 0, "duplicate": 0, "held-out text": 0, "held-out voice": 0, "past the end": 0}
    for lab_path in sorted(LABELS.glob("*/*.json")):
        res = json.loads(lab_path.read_text(encoding="utf-8"))
        key = (res["platform"], res["id"])
        if key not in metas:
            continue
        meta, audio = metas[key]
        if voices.get(key) in ("test", "error"):  # a voice that matched, or could not be checked
            skipped["test voice"] += 1
            continue
        if key in dupes:
            skipped["duplicate"] += 1
            continue
        if key in held_voice:
            skipped["held-out voice"] += 1
            continue
        reciter = _reciter(res["platform"], meta)
        if lab_path.name.endswith(".captions.json"):
            if not res["usable"]:
                continue
            # Captions bring their own text: keep a held-out du'a out by its words, not its id.
            words = [w for ln in res["lines"] if ln["ok"] for w in ln["words"]]
            if _held_share(words, held) >= 0.2:
                skipped["held-out text"] += 1
                continue
            spans = [{"dua": f"caption:{res['id']}", "lines": res["lines"], "captions": True}]
        else:
            if (LABELS / res["platform"] / f"{res['id']}.captions.json").exists():
                cap = json.loads((LABELS / res["platform"] / f"{res['id']}.captions.json").read_text(encoding="utf-8"))
                if cap.get("usable"):
                    continue  # the human-timed version wins
            spans = [s for s in res["spans"] if s["usable"]]
        # More decoded audio than the platform lists: ask the container where seeks stop.
        end = None
        if res.get("duration", 0) > float(meta.get("duration") or 0) + 1.0:
            end = ts_span(audio)
        for sp in spans:
            dua_id = sp["dua"]
            if dua_id in HELD_OUT_DUAS or dua_id in held_ids:
                skipped["held-out text"] += 1
                continue
            if sp.get("captions"):
                tf = lambda ln: ln["tokens"]  # noqa: E731
            else:
                if dua_id not in toks:
                    toks[dua_id] = _token_lists(duas[dua_id])
                tl = toks[dua_id]
                tf = lambda ln, tl=tl: tl.get(ln["seg"], [])  # noqa: E731
            for s, e, text in _windows(sp["lines"], tf, args.window, args.hop):
                if end is not None and e > end + 0.25:
                    skipped["past the end"] += 1
                    continue
                rows.append({"audio": str(audio.resolve()), "start": s, "end": e, "text": text, "dua": dua_id,
                             "reciter": reciter, "source": res["platform"], "rec": res["id"]})
                d = (e - s) / 3600 * (args.hop / args.window if args.hop < args.window else 1)
                per_platform[res["platform"]] = per_platform.get(res["platform"], 0) + d
                per_dua[dua_id] = per_dua.get(dua_id, 0) + d
                stats["h"] += d
    # Validation by uploader: every window of a channel lands on the same side.
    def side(r):
        return "val" if int(hashlib.sha1(r["reciter"].encode()).hexdigest()[:8], 16) % 100 < args.val_pct else "train"

    out = ROOT / "data" / "cache" / "finetune"
    tr = [r for r in rows if side(r) == "train"]
    va = [r for r in rows if side(r) == "val"]
    for name, part in ((f"harvest_{args.version}", tr), (f"harvest_{args.version}_val", va)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.combine:
        # train_<combine>.jsonl = the existing set + the harvest, capped per text so the
        # few long, popular du'as (Abu Hamza, Kumayl, Jawshan) don't drown the rest;
        # val_<combine>.jsonl = the existing val set unchanged, so val numbers stay comparable.
        import random

        rng = random.Random(0)
        by_text: dict[str, list] = {}
        for r in tr:
            by_text.setdefault(r["dua"], []).append(r)
        cap = int(args.cap_hours * 3600 / args.hop)
        kept = []
        for d, rs in by_text.items():
            rng.shuffle(rs)
            kept += rs[:cap] if not d.startswith("caption:") else rs
        base = [json.loads(x) for x in (out / f"train_{args.base}.jsonl").read_text(encoding="utf-8").splitlines() if x]
        comb = base + kept
        rng.shuffle(comb)
        with (out / f"train_{args.combine}.jsonl").open("w", encoding="utf-8") as f:
            for r in comb:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out / f"val_{args.combine}.jsonl").write_text((out / f"val_{args.base}.jsonl").read_text(encoding="utf-8"),
                                                        encoding="utf-8")
        print(f"train_{args.combine}: {len(base)} train_{args.base} + {len(kept)} harvest windows "
              f"(cap {args.cap_hours} h per text); val_{args.combine} = val_{args.base}", flush=True)
    print(f"{len(rows)} windows ({len(tr)} train / {len(va)} val), ~{stats['h']:.1f} h of distinct audio; "
          f"{len({r['reciter'] for r in rows})} uploaders, {len({r['rec'] for r in rows})} recordings, "
          f"{len({r['dua'] for r in rows if not r['dua'].startswith('caption:')})} corpus texts", flush=True)
    print("by platform (h):", {k: round(v, 1) for k, v in per_platform.items()}, flush=True)
    print(f"left out: {skipped} (held-out texts by id: {sorted(held_ids)})", flush=True)
    top = sorted(per_dua.items(), key=lambda kv: -kv[1])[:25]
    print("top texts (h):", ", ".join(f"{k} {v:.1f}" for k, v in top), flush=True)


def _dua_words(dua) -> list[str]:
    from dua_recognition.text import normalize

    return [w for seg in dua.segments for w in normalize(seg.arabic).split()]


def _held_out_grams(duas) -> set[tuple[str, str, str]]:
    grams = set()
    for d in HELD_OUT_DUAS:
        w = _dua_words(duas[d])
        grams |= set(zip(w, w[1:], w[2:]))
    return grams


def _held_share(words: list[str], grams: set) -> float:
    """Share of a text's word trigrams that occur in a held-out du'a."""
    g = list(zip(words, words[1:], words[2:]))
    return sum(x in grams for x in g) / max(1, len(g))


def _duplicates(metas) -> set[tuple[str, str]]:
    """Re-uploads of one recording (YouTube, Aparat and shiavoice share many): same main
    text, duration within 1.5 s, voice >= 0.9. The first (by platform order) is kept."""
    import hashlib

    spk = ROOT / "data" / "cache" / "speaker"
    info = []
    for (p, vid), (meta, audio) in metas.items():
        lab = LABELS / p / f"{vid}.json"
        if not lab.exists():
            continue
        r = json.loads(lab.read_text(encoding="utf-8"))
        if not r["spans"]:
            continue
        main = max(r["spans"], key=lambda s: s["t1"] - s["t0"])["dua"]
        f = spk / (hashlib.sha1(str(audio.resolve()).encode()).hexdigest()[:16] + ".npy")
        info.append(((p, vid), main, r["duration"], np.load(f) if f.exists() else None))
    by: dict[str, list] = {}
    for k, main, dur, e in info:
        by.setdefault(main, []).append((dur, k, e))
    dupes = set()
    for rs in by.values():
        rs.sort(key=lambda x: x[0])
        for i, (d1, k1, e1) in enumerate(rs):
            if k1 in dupes:
                continue
            for d2, k2, e2 in rs[i + 1:]:
                if d2 - d1 > 1.5:
                    break
                if k2 not in dupes and e1 is not None and e2 is not None and float(e1 @ e2) >= 0.9:
                    dupes.add(k2)
    return dupes


def _is_val(reciter: str, val_pct: int) -> bool:
    import hashlib

    return int(hashlib.sha1(reciter.encode()).hexdigest()[:8], 16) % 100 < val_pct


def _voice_holdout(metas, val_pct: int, thr: float = 0.85) -> tuple[set, dict]:
    """Train-side recordings to drop so the val uploaders (and data/testbed/test_voices.json,
    the scenario bench's test pool) are held out by VOICE, not just by uploader name: the
    same reciter is often uploaded by several channels. Voices are harvest_voices.py's
    ECAPA embeddings (speaker_check's cache).

    thr 0.85, not speaker_check.LEAK's 0.7: on this in-the-wild audio (crowds, reverb,
    music) 1 in 1,000 pairs of different uploaders already scores 0.8, so 0.7 against a
    ~300-voice pool dropped 40% of training recordings, mostly by chance. 0.85 drops the
    strong matches (one reciter on several channels, re-uploads); for a strictly unseen
    test set, data/harvest/heldout_voice_overlap.jsonl gives each held-out recording's
    closest training voice, so the test side can keep only those below 0.7."""
    import hashlib

    spk = ROOT / "data" / "cache" / "speaker"
    emb = {}
    for key, (meta, audio) in metas.items():
        f = spk / (hashlib.sha1(str(audio.resolve()).encode()).hexdigest()[:16] + ".npy")
        if f.exists():
            emb[key] = np.load(f)
    held = {k for k, (m, _) in metas.items() if _is_val(_reciter(k[0], m), val_pct)}
    tb = ROOT / "data" / "testbed" / "test_voices.json"
    bench = set()
    if tb.exists():
        d = json.loads(tb.read_text(encoding="utf-8"))
        rows = d.get("recordings", []) if isinstance(d, dict) else d
        bench = {(r["source"], r["rec"]) for r in rows if isinstance(r, dict) and "source" in r and "rec" in r}
    pool_keys = [k for k in held | bench if k in emb]
    drop = set(bench - held)
    if pool_keys:
        P = np.stack([emb[k] for k in pool_keys])
        for k, e in emb.items():
            if k not in held and k not in bench and float((P @ e).max()) >= thr:
                drop.add(k)
    # The other direction, for the test side: each held-out recording's closest voice among
    # the training recordings that remain.
    train_keys = [k for k in emb if k not in held and k not in bench and k not in drop]
    if pool_keys and train_keys:
        T = np.stack([emb[k] for k in train_keys])
        with (HARVEST / "heldout_voice_overlap.jsonl").open("w", encoding="utf-8") as f:
            for k in pool_keys:
                s = T @ emb[k]
                j = int(np.argmax(s))
                f.write(json.dumps({"source": k[0], "rec": k[1], "max_train_voice": round(float(s[j]), 3),
                                    "nearest_train": list(train_keys[j]),
                                    "reciter": _reciter(k[0], metas[k][0])}, ensure_ascii=False) + "\n")
    stats = {"held-out voices": len(pool_keys), "held-out recordings without a voice": len((held | bench) - set(emb)),
             "train recordings dropped for a held-out voice": len(drop - bench)}
    return drop, stats


def _voice_flags() -> dict[tuple[str, str], str]:
    """(platform, id) -> 'test' for recordings whose voice matched a test reciter."""
    out = {}
    for r in read_jsonl(HARVEST / "voices.jsonl"):
        out[(r["platform"], r["id"])] = r.get("verdict", "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=["frames", "align", "captions", "export"])
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=3.0)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--val-pct", type=int, default=3)
    ap.add_argument("--combine", default="", help="export: also write train_<name>/val_<name> = --base set + harvest")
    ap.add_argument("--base", default="v4", help="export --combine: the existing set to extend")
    ap.add_argument("--cap-hours", type=float, default=8.0, help="export --combine: max hours of windows per text")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--model", default=str(TEACHER))
    ap.add_argument("--platforms", nargs="*", default=list(PLATFORMS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--follow", action="store_true", help="frames: keep waiting for new downloads")
    ap.add_argument("--idle-exit", type=int, default=600, help="frames --follow: minutes idle before exiting")
    args = ap.parse_args()
    for stage in args.stages:
        {"frames": cmd_frames, "align": cmd_align, "captions": cmd_captions, "export": cmd_export}[stage](args)


if __name__ == "__main__":
    main()
