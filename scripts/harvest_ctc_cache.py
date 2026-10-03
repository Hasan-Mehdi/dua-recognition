#!/usr/bin/env python
"""Teacher cache for the phone CTC student from the harvest, without another GPU pass.

scripts/ctc_teacher.py runs the teacher over every training window and keeps its letter
posteriors plus a forced alignment of the window's text. The harvest already has the
same teacher's (wav2vec2-quran-dua-voices) posteriors for whole files
(harvest_label.py frames), so each exported window's frames are sliced out of those and
aligned here, on the CPU, into the same .npz layout train_ctc_student.py reads.

The full export is ~1M overlapping windows (~20 GB of posteriors), more than the
trainer holds in memory, so this keeps a voice-balanced sample: at most --per-reciter
windows per uploader, --max-windows in all, non-overlapping where possible.

    python scripts/harvest_ctc_cache.py harvest_v1 --combine-with train_v4   # -> harvest_v1s, train_v4h
    python scripts/train_ctc_student.py --train-set train_v4h --teacher ctc_student_voices ...
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ctc_teacher import NEG, force_align  # noqa: E402
from harvest_label import FPS, FRAMES  # noqa: E402

from dua_recognition.align import encode  # noqa: E402

FT = ROOT / "data" / "cache" / "finetune"
SR = 16000


def cut_clips(rows: list[dict], clips: Path, jobs: int) -> list[dict]:
    """16 kHz int16 .npy per row (ffmpeg seek + decode); rows get "clip", lose "audio"."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    clips.mkdir(parents=True, exist_ok=True)

    def one(i_r):
        i, r = i_r
        f = clips / f"{i:06d}.npy"
        if not f.exists():
            p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{r['start']:.3f}", "-t", f"{r['end'] - r['start']:.3f}",
                                "-i", r["audio"], "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
            y = np.frombuffer(p.stdout, dtype=np.int16)
            np.save(f, y)
        out = {k: v for k, v in r.items() if k != "audio"}
        return dict(out, clip=str(f), src_audio=r["audio"])

    with ThreadPoolExecutor(jobs) as ex:
        out = list(ex.map(one, enumerate(rows)))
    n_short = sum(1 for r in out if np.load(r["clip"], mmap_mode="r").shape[0] < SR)
    print(f"  {len(out)} clips cut ({n_short} under 1 s)", flush=True)
    return out


def _link(link: Path, target: Path) -> None:
    # A failed trainer run leaves an empty <set> folder behind (unpack() makes it first).
    if link.is_dir() and not link.is_symlink() and not any(link.iterdir()):
        link.rmdir()
    if not link.exists():
        import subprocess

        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


def combine(args, arrays: dict, rows: list[dict], name: str) -> None:
    """<base>h = an existing teacher-cached set (e.g. train_v4) + this harvest sample, as one
    training set (train_ctc_student.py --train-set takes one)."""
    base = ROOT / "data" / "cache" / args.teacher / args.combine_with
    b = {k: np.load(base / f"{k}.npy", mmap_mode="r") for k in arrays}
    out_name = f"{args.combine_with}h"
    d = Path(args.out) / out_name
    d.mkdir(parents=True, exist_ok=True)
    # Frames written through a memory map, a block at a time: several GB, and the machine's
    # commit limit is shared with the GPU job.
    nb, nh = b["frames"].shape[0], arrays["frames"].shape[0]
    fr = np.lib.format.open_memmap(d / "frames.npy", mode="w+", dtype=np.float16,
                                   shape=(nb + nh, b["frames"].shape[1]))
    for i in range(0, nb, 1 << 22):
        j = min(i + (1 << 22), nb)
        fr[i:j] = b["frames"][i:j]
    fr[nb:] = arrays["frames"]
    fr.flush()
    del fr
    np.save(d / "offsets.npy", np.concatenate([b["offsets"], b["offsets"][-1] + arrays["offsets"][1:]]))
    np.save(d / "loff.npy", np.concatenate([b["loff"], b["loff"][-1] + arrays["loff"][1:]]))
    for k in ("letters", "first", "last", "score", "samples"):
        np.save(d / f"{k}.npy", np.concatenate([b[k], arrays[k]]))
    (d / "done").write_text("ok")
    _link(ROOT / "data" / "cache" / args.teacher / out_name, d)
    base_rows = "".join(x + "\n" for x in (FT / f"{args.combine_with}.jsonl").read_text(encoding="utf-8").splitlines()
                        if x.strip())  # train_v4.jsonl has no final newline
    (FT / f"{out_name}.jsonl").write_text(base_rows + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                          encoding="utf-8")
    print(f"{out_name}: {len(b['offsets']) - 1} {args.combine_with} + {len(rows)} harvest windows -> {d}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", help="a harvest export, e.g. harvest_v1 (data/cache/finetune/<set>.jsonl)")
    ap.add_argument("--max-windows", type=int, default=60_000)
    ap.add_argument("-j", "--jobs", type=int, default=6, help="parallel ffmpeg clip cuts")
    ap.add_argument("--per-reciter", type=int, default=400)
    ap.add_argument("--suffix", default="s", help="output set name = <set><suffix>")
    ap.add_argument("--out", default=r"D:\dua-data\ctc_student_harvest", help="where the arrays go")
    ap.add_argument("--teacher", default="ctc_student_voices", help="teacher cache (under data/cache/) to link into")
    ap.add_argument("--combine-with", default="", help="also write <this>h = this teacher-cached set + the sample")
    args = ap.parse_args()
    rows = [json.loads(x) for x in (FT / f"{args.set}.jsonl").read_text(encoding="utf-8").splitlines() if x]
    rng = random.Random(0)
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["reciter"]].append(r)
    picked = []
    for rs in by.values():
        rs.sort(key=lambda r: (r["rec"], r["start"]))
        # every other window: the export's windows overlap by half
        rs = rs[::2]
        rng.shuffle(rs)
        picked += rs[: args.per_reciter]
    rng.shuffle(picked)
    picked = picked[: args.max_windows]
    picked.sort(key=lambda r: (r["source"], r["rec"], r["start"]))
    name = f"{args.set}{args.suffix}"
    print(f"{name}: {len(picked)} windows from {len({r['reciter'] for r in picked})} uploaders", flush=True)

    frames, offsets, letters, loff, first, last, scores, lens, kept = [], [0], [], [0], [], [], [], [], []
    cur, lp_all = None, None
    for i, r in enumerate(picked):
        key = (r["source"], r["rec"])
        if key != cur:
            f = FRAMES / r["source"] / f"{r['rec']}.npz"
            with np.load(f) as z:
                lp_all = z["lp"]
            cur = key
        a, b = int(round(r["start"] * FPS)), int(round(r["end"] * FPS))
        lp = lp_all[a:b]
        codes = encode(r["text"]).astype(np.int64)
        if not lp.shape[0] or not codes.size:
            continue
        sc, fi, la = force_align(lp.astype(np.float64), codes)
        if sc <= NEG / 2:
            continue  # window too short for its text
        frames.append(lp.astype(np.float16))
        offsets.append(offsets[-1] + lp.shape[0])
        letters.append(codes.astype(np.int8))
        loff.append(loff[-1] + codes.size)
        first.append(fi)
        last.append(la)
        scores.append(sc / max(1, lp.shape[0]))
        lens.append(int(round((r["end"] - r["start"]) * SR)))
        kept.append(r)
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{len(picked)}", flush=True)
    # Written unpacked (train_ctc_student.unpack's layout, "done" marker included) on D:,
    # and linked into the teacher cache: C: has no room for a multi-GB cache.
    d = Path(args.out) / name
    d.mkdir(parents=True, exist_ok=True)
    arrays = dict(frames=np.concatenate(frames), offsets=np.array(offsets, dtype=np.int64),
                  letters=np.concatenate(letters), loff=np.array(loff, dtype=np.int64),
                  first=np.concatenate(first), last=np.concatenate(last),
                  score=np.array(scores, dtype=np.float32), samples=np.array(lens, dtype=np.int64))
    for k, v in arrays.items():
        np.save(d / f"{k}.npy", v)
    (d / "done").write_text("ok")
    link = ROOT / "data" / "cache" / args.teacher / name
    _link(link, d)
    # Each window's audio as its own clip: the trainers' AudioBank decodes and caches every
    # referenced source file whole, which for thousands of long harvest files is ~150 GB.
    kept = cut_clips(kept, Path(args.out) / name / "clips", args.jobs)
    # The rows, in the same order, as a finetune set (train_ctc_student.py reads both).
    (FT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                                      encoding="utf-8")
    if args.combine_with:
        combine(args, arrays, kept, name)
    s = np.array(scores)
    print(f"{name}: {len(kept)} windows -> {d} (linked as {link}); per-frame align score median "
          f"{np.median(s):.3f}, p10 {np.percentile(s, 10):.3f}", flush=True)


if __name__ == "__main__":
    main()
