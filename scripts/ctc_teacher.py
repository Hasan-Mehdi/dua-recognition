#!/usr/bin/env python
"""Teacher pass for the phone CTC model: wav2vec2-quran-dua over every training window.

The phone follower needs a CTC model small enough to run several times a second in a
browser (scripts/train_ctc_student.py). It learns from two things this script keeps,
per window of the Whisper fine-tune sets (data/cache/finetune/<set>.jsonl):

- the teacher's per-frame letter posteriors, folded onto the corpus alphabet (ctc.py:
  column 0 = blank), as float16;
- a forced alignment of the window's reference text to those frames (the CTC Viterbi
  path through the reference letters): the frame each letter is emitted on. A student
  trained on a crop of the window then knows exactly which letters the crop holds.

    python scripts/ctc_teacher.py train_v4 val_v4 synth_v5 crowd

Writes data/cache/ctc_student/<set>.npz (offsets into one frame array) per set.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dua_recognition.align import encode  # noqa: E402
from dua_recognition.ctc import LOG_FLOOR, N_COLS, SR, CtcModel  # noqa: E402
from finetune_whisper import AudioBank, load_rows  # noqa: E402

OUT = ROOT / "data" / "cache" / "ctc_student"
NEG = -1e9


def _force_align(lp, r):
    """CTC Viterbi alignment of letters r (J) to frames lp (T x C), all letters used.
    States 0..2J: even = blank, odd = letter (s-1)//2. Returns (score, frame of each
    letter's first emission, frame of its last emission); score -inf if T is too short."""
    T = lp.shape[0]
    J = r.size
    S = 2 * J + 1
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    dp[0, 0] = lp[0, 0]
    if J:
        dp[0, 1] = lp[0, r[0]]
    for t in range(1, T):
        for s in range(S):
            best = dp[t - 1, s]
            arg = 0
            if s >= 1 and dp[t - 1, s - 1] > best:
                best = dp[t - 1, s - 1]
                arg = 1
            if s >= 2 and s % 2 == 1 and r[(s - 1) // 2] != r[(s - 3) // 2] and dp[t - 1, s - 2] > best:
                best = dp[t - 1, s - 2]
                arg = 2
            e = lp[t, 0] if s % 2 == 0 else lp[t, r[(s - 1) // 2]]
            dp[t, s] = best + e
            bp[t, s] = arg
    s = S - 1
    if J and dp[T - 1, S - 2] > dp[T - 1, S - 1]:
        s = S - 2
    score = dp[T - 1, s]
    first = np.full(J, -1, dtype=np.int32)
    last = np.full(J, -1, dtype=np.int32)
    for t in range(T - 1, -1, -1):
        if s % 2 == 1:
            j = (s - 1) // 2
            first[j] = t
            if last[j] < 0:
                last[j] = t
        a = bp[t, s]
        s -= a
    return score, first, last


try:
    from numba import njit

    force_align = njit(cache=True)(_force_align)
except ImportError:
    force_align = _force_align


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--model", default="models/wav2vec2-quran-dua")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT), help="where the .npz files go (one folder per teacher)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = CtcModel(args.model)
    torch = m.torch
    for name in args.sets:
        rows = load_rows(name)
        if args.limit:
            rows = rows[: args.limit]
        bank = AudioBank([r.get("audio") for r in rows])
        t0 = time.time()
        frames, offsets, letters, loff, first, last, scores, lens = [], [0], [], [0], [], [], [], []
        for b in range(0, len(rows), args.batch):
            chunk = rows[b : b + args.batch]
            wavs = [bank.window(r) for r in chunk]
            with torch.no_grad():
                inp = m.proc(wavs, sampling_rate=SR, return_tensors="pt", padding=True)
                vals = inp.input_values.to(m.device).half()
                mask = inp.get("attention_mask")
                logits = m.model(vals, attention_mask=mask.to(m.device) if mask is not None else None).logits.float()
                folded = torch.log((logits.softmax(-1) @ m.fold).clamp_min(np.exp(LOG_FLOOR))).cpu().numpy()
            for k, (r, x) in enumerate(zip(chunk, wavs)):
                f = int(m.model._get_feat_extract_output_lengths(torch.tensor(len(x))).item()) if len(x) else 0
                lp = folded[k, :f].astype(np.float16)
                codes = encode(r["text"]).astype(np.int64)
                sc, fi, la = force_align(lp.astype(np.float64), codes) if f else (NEG, np.zeros(0, np.int32), np.zeros(0, np.int32))
                frames.append(lp)
                offsets.append(offsets[-1] + f)
                letters.append(codes.astype(np.int8))
                loff.append(loff[-1] + codes.size)
                first.append(fi)
                last.append(la)
                scores.append(sc / max(1, f))
                lens.append(len(x))
            if (b // args.batch) % 100 == 0:
                done = b + len(chunk)
                print(f"{name}: {done}/{len(rows)}  {(time.time() - t0) / done * 1000:.1f} ms/window", flush=True)
        np.savez(out / f"{name}.npz", frames=np.concatenate(frames), offsets=np.array(offsets, dtype=np.int64),
                 letters=np.concatenate(letters), loff=np.array(loff, dtype=np.int64),
                 first=np.concatenate(first), last=np.concatenate(last),
                 score=np.array(scores, dtype=np.float32), samples=np.array(lens, dtype=np.int64))
        s = np.array(scores)
        print(f"{name}: {len(rows)} windows in {time.time() - t0:.0f} s; per-frame align score "
              f"median {np.median(s):.3f}, p10 {np.percentile(s, 10):.3f}", flush=True)


if __name__ == "__main__":
    main()
