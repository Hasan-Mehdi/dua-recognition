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


def end_scores(lp: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Best CTC path log-likelihood of `lp` (T x C) ending on each letter of `r`.

    Two states per reference letter j: L[j] "emitting r[j]" and B[j] "blank after
    r[j]". Moves per frame: stay; L[j-1] or B[j-1] -> L[j] (L[j-1] -> L[j] only when
    the letters differ, as in CTC: a doubled letter needs a blank between).
    Free start: at frame 0 any L[j] or B[j] may be the first state.
    """
    lp = lp.astype(np.float32)
    emit = lp[:, r]  # T x J
    blank = lp[:, 0]
    diff = np.r_[True, r[1:] != r[:-1]]
    L = emit[0].copy()
    B = np.full(r.size, blank[0], dtype=np.float32)
    for t in range(1, lp.shape[0]):
        prevL = np.r_[NEG, L[:-1]]
        prevB = np.r_[NEG, B[:-1]]
        enter = np.maximum(prevB, np.where(diff, prevL, NEG))
        newB = np.maximum(B, L) + blank[t]
        L = np.maximum(L, enter) + emit[t]
        B = newB
    return np.maximum(L, B)


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
