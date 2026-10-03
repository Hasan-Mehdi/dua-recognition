"""Score the reference text directly against a CTC model's output.

align.py compares a 1-best *transcript* with the corpus. That throws away
everything the recognizer was unsure about: a window where the model hesitates
between two readings becomes one string, right or wrong. With a CTC model we
can skip the transcript and ask the question the follower actually needs:
"if the reciter's last word right now is w, how likely is this audio?"

For a window's per-frame log posteriors (columns = align._ALPHABET codes,
column 0 = blank), `end_scores` runs a CTC Viterbi pass over the whole corpus
letter string with a free start: the window may begin anywhere in the text (even
mid-letter) and must be explained frame by frame up to some end letter e. The
result is, for every e, the best log-likelihood of the window ending there.
Same shape as align.semiglobal_end_costs, so the tracker takes either.
"""
from __future__ import annotations

import numpy as np

NEG = -1e9


def end_scores(lp: np.ndarray, r: np.ndarray, starts: np.ndarray | None = None) -> np.ndarray:
    """Best CTC path log-likelihood of `lp` (T x C) ending on each letter of `r`.

    Two states per reference letter j: L[j] "emitting r[j]" and B[j] "blank after
    r[j]". Moves per frame: stay; L[j-1] or B[j-1] -> L[j] (L[j-1] -> L[j] only when
    the letters differ, as in CTC: a doubled letter needs a blank between).
    Free start: at frame 0 any L[j] or B[j] may be the first state; `starts` (bool per
    letter) limits that to the letters where it is True.
    """
    ok = np.ones(r.size, dtype=np.bool_) if starts is None else np.ascontiguousarray(starts, dtype=np.bool_)
    if _end_scores_jit is not None and r.size:
        return _end_scores_jit(np.ascontiguousarray(lp, dtype=np.float32), np.ascontiguousarray(r, dtype=np.int64), ok)
    return end_scores_numpy(lp, r, ok)


def end_scores_numpy(lp: np.ndarray, r: np.ndarray, starts: np.ndarray | None = None) -> np.ndarray:
    """end_scores, vectorized over letters (the fallback without numba)."""
    lp = lp.astype(np.float32)
    emit = lp[:, r]  # T x J
    blank = lp[:, 0]
    diff = np.r_[True, r[1:] != r[:-1]]
    L = emit[0].copy()
    B = np.full(r.size, blank[0], dtype=np.float32)
    if starts is not None:
        L[~starts] = NEG
        B[~starts] = NEG
    for t in range(1, lp.shape[0]):
        prevL = np.r_[NEG, L[:-1]]
        prevB = np.r_[NEG, B[:-1]]
        enter = np.maximum(prevB, np.where(diff, prevL, NEG))
        newB = np.maximum(B, L) + blank[t]
        L = np.maximum(L, enter) + emit[t]
        B = newB
    return np.maximum(L, B)


def _end_scores_loop(lp, r, starts):
    """end_scores as plain loops, for numba (same float64 sums as the numpy path)."""
    J = r.size
    L = np.empty(J)
    B = np.empty(J)
    for j in range(J):
        L[j] = lp[0, r[j]] if starts[j] else NEG
        B[j] = lp[0, 0] if starts[j] else NEG
    for t in range(1, lp.shape[0]):
        bl = float(lp[t, 0])
        for j in range(J - 1, -1, -1):  # right to left: L[j-1], B[j-1] are still frame t-1's
            enter = NEG
            if j > 0:
                enter = B[j - 1]
                if r[j] != r[j - 1] and L[j - 1] > enter:
                    enter = L[j - 1]
            nb = max(B[j], L[j]) + bl
            L[j] = max(L[j], enter) + float(lp[t, r[j]])
            B[j] = nb
    out = np.empty(J)
    for j in range(J):
        out[j] = max(L[j], B[j])
    return out


def _jump_scores_loop(lp, rl, r, starts):
    """jump_scores as plain loops (numba)."""
    T = lp.shape[0]
    J1, J = rl.size, r.size
    L1 = np.empty(J1)
    B1 = np.empty(J1)
    for j in range(J1):
        L1[j] = lp[0, rl[j]]
        B1[j] = lp[0, 0]
    L = np.empty(J)
    B = np.empty(J)
    for j in range(J):
        L[j] = NEG
        B[j] = NEG
    for t in range(1, T):
        bl = float(lp[t, 0])
        prefix = NEG  # best path through the near text ending at frame t - 1
        for j in range(J1):
            if L1[j] > prefix:
                prefix = L1[j]
            if B1[j] > prefix:
                prefix = B1[j]
        for j in range(J - 1, -1, -1):
            enter = NEG
            if j > 0:
                enter = B[j - 1]
                if r[j] != r[j - 1] and L[j - 1] > enter:
                    enter = L[j - 1]
            if starts[j] and prefix > enter:
                enter = prefix
            nb = max(B[j], L[j]) + bl
            L[j] = max(L[j], enter) + float(lp[t, r[j]])
            B[j] = nb
        for j in range(J1 - 1, -1, -1):
            enter = NEG
            if j > 0:
                enter = B1[j - 1]
                if rl[j] != rl[j - 1] and L1[j - 1] > enter:
                    enter = L1[j - 1]
            nb = max(B1[j], L1[j]) + bl
            L1[j] = max(L1[j], enter) + float(lp[t, rl[j]])
            B1[j] = nb
    out = np.empty(J)
    for j in range(J):
        out[j] = max(L[j], B[j])
    near = NEG
    for j in range(J1):
        near = max(near, L1[j], B1[j])
    return out, near


def jump_scores(lp: np.ndarray, near: np.ndarray, r: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, float]:
    """A jump within the window. The frames up to some point follow the `near` letters (free start,
    as end_scores); from the next frame the path enters `r` at a letter where `starts` is True
    and ends on each letter of `r`. Returns those scores, and the best path through `near` alone
    over the whole window, to compare with: a place in `r` worth jumping to beats it."""
    lp = np.ascontiguousarray(lp, dtype=np.float32)
    rl = np.ascontiguousarray(near, dtype=np.int64)
    r = np.ascontiguousarray(r, dtype=np.int64)
    ok = np.ascontiguousarray(starts, dtype=np.bool_)
    if not r.size or not rl.size or lp.shape[0] < 2:
        return np.full(r.size, NEG), NEG
    return (_jump_scores_jit or _jump_scores_loop)(lp, rl, r, ok)


try:
    from numba import njit

    _end_scores_jit = njit(cache=True, nogil=True)(_end_scores_loop)
    _jump_scores_jit = njit(cache=True, nogil=True)(_jump_scores_loop)
except ImportError:  # numba is optional: end_scores_numpy gives the same answer, slower
    _end_scores_jit = None
    _jump_scores_jit = None


def squeeze_blanks(lp: np.ndarray, p_blank: float = 0.95) -> np.ndarray:
    """Collapse each run of blank-dominated frames into one (their mean).

    CTC output is peaky: ~85% of frames are near-certain blanks, where every
    path does the same thing (sit in a blank or hold a letter). One frame per
    run keeps what matters, including the blank that separates a doubled
    letter, and makes the Viterbi pass several times cheaper.
    """
    is_blank = lp[:, 0] > np.log(p_blank)
    if not is_blank.any():
        return lp
    run_id = np.cumsum(np.r_[True, (is_blank[1:] != is_blank[:-1]) | ~is_blank[1:]])
    out = np.zeros((run_id[-1], lp.shape[1]), dtype=np.float32)
    np.add.at(out, run_id - 1, lp.astype(np.float32))
    return out / np.bincount(run_id - 1)[:, None]


def has_speech(lp: np.ndarray, min_frames: int = 3) -> bool:
    """Any letters at all? Frames where some letter beats blank."""
    return int((lp[:, 1:].max(axis=1) > lp[:, 0]).sum()) >= min_frames
