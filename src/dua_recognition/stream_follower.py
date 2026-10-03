"""Follow a reciter through a du'a by decoding the CTC frames as one stream.

follower.py rescores the latest 2 s of frames every step and keeps one position, with rules
for everything a reader does that a 2 s window can't see: going back a line or three, skipping,
restarting, pausing, talking in between, a refrain that recurs every few lines. Each rule was
tuned on the case that needed it, and they fight (docs/results/bench.md).

This keeps a belief over every position in the du'a instead and updates it frame by frame
(CTC forward pass over the du'a's letters), with reading itself as the transition model:

    next letter / next word / next line    free (the text in order)
    start the current line again           c_restart
    go back k = 1..3 lines (line start)    c_back[k-1]
    skip ahead to line +2..+4              c_skip[k-2]; the rest of a line, from mid-line: + c_mid
    anywhere else in the du'a              c_far (spread over its lines)
    talk / salawat / anything not in it    filler: enter c_fill_in, any frame at (best column - fill_cost),
                                           leave to a word of the line it left, or the next line's start

A pause is free (blank after the last letter). A refrain keeps every repetition alive until the
words that differ are heard; a step back needs only that the new line's letters fit better
than going on. The tracker (Whisper, slow but sure of the du'a) supplies the du'a, the starting
belief (its line masses) and gentle evidence at each of its updates (tracker_weight).

The frames are a stream: from each window (2 s, every 0.1 s) the hop's worth of frames that end
`lookahead` before the window's end (frames at the very end hear no future and are much worse:
greedy CER 0.25 vs 0.13 at 0.2 s). The newest `lookahead` seconds are decoded tentatively on a
copy for the display, then thrown away.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .align import CorpusIndex

NEG = -1e30

try:  # numba makes the per-frame pass ~50x faster; the numpy fallback is the reference
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


@dataclass
class StreamConfig:
    temp: float = 1.0  # divides the CTC log posteriors
    lookahead: float = 0.2  # s: committed frames end this long before the window's end
    hop: float = 0.1  # s between windows
    frame_s: float = 0.02
    c_restart: float = -3.0
    c_back: tuple = (-5.0, -6.0, -7.0)
    c_skip: tuple = (-8.0, -9.0, -10.0)
    c_mid: float = -3.0  # leaving a line before its last word
    c_rest: float = -3.0  # ...for the next line's start (on top of c_mid)
    c_far: float = -13.0  # all other line starts together
    c_fill_in: float = -8.0
    fill_cost: float = 1.0  # nats per frame below the frame's best column
    c_fill_word: float = -2.0  # filler back to a word of its line (spread over the line's words)
    c_fill_next: float = -1.0  # filler to the next line's start
    # Interjections: the salawat said between lines or after the Prophet's name (optionally with
    # "wa-ajjil farajahum"), entered from any word end at c_int_in and left for the next line's
    # start (c_int_next) or a word of the line it was said in (c_int_word). Without it the decoder
    # has to explain a salawat as a jump to some salawat line of the du'a, or as talk.
    interjections: tuple = ("اللهم صل على محمد وآل محمد|وعجل فرجهم",)
    c_int_in: float = -4.0
    c_int_next: float = -1.0
    c_int_word: float = -1.0
    # The page's stop detector (asr.LiveQuiet: no sound within 15 dB of the voice) as evidence: while
    # it has heard no voice for `quiet_s`, letters cost `quiet_pen` nats more. Room tone makes the
    # model hear a stray letter now and then, and the cheapest place for one is the next line's start.
    quiet_s: float = 0.3
    quiet_pen: float = 4.0
    # Nats off the blank column (0 = none): ordinary voices make faint letters, and a path resting in
    # the blank after the last word soaks up the belief until several are heard (line changes late).
    blank_bias: float = 1.0
    tracker_weight: float = 0.3  # exponent on the tracker's line masses at each of its updates
    tracker_floor: float = 0.02
    show_p: float = 0.35  # a word is shown once it holds this much of the belief...
    line_p: float = 0.5  # ...a word in another line, this much
    line_steps: int = 2  # ...for this many steps in a row
    next_p: float = 0.6  # the next line's first word: this much of the belief...
    next_steps: int = 2  # ...for this many steps
    lapse_s: float = 20.0  # the tracker without a du'a this long (a long pause, a lull): start over
    # ...but while the reader is making sound (stop detector under lapse_quiet s), only lapse_voice s:
    # recitation the tracker can't place is likely a text it doesn't know, not a lull.
    lapse_voice: float = 4.0
    lapse_quiet: float = 1.0
    # A du'a change counts once the tracker has held the new du'a this long (0 = at once). Texts that
    # share a passage (Ayat al-Kursi in Namaz-e-Wahshat and Eid-e-Mubahila) flip the tracker's du'a for a
    # moment, and starting over there showed a line of the other text.
    switch_s: float = 2.0
    # A beam (0 = off: every line every frame, too slow for Kumayl on a phone): after each step only
    # lines holding at least `beam` of the belief, `beam_margin` lines either side (back and skip
    # moves land there), the shown word's line and every line the tracker puts `propose_mass` on
    # (+-1: how a far jump the decoder can't see comes back in) are updated; the rest are dropped.
    beam: float = 1e-6
    beam_margin: int = 5
    propose_mass: float = 0.05
    # The frames propose too: every `ctc_every` steps the latest `ctc_window` s are scored against the
    # whole du'a (ctc_align.end_scores) and up to `ctc_lines` lines whose best end score is within
    # `ctc_margin` nats of the best join the beam (+-1). Only considered: the forward pass decides. A
    # reader who jumps far is otherwise only found once Whisper's tracker believes it, seconds later.
    ctc_every: int = 5
    ctc_window: float = 1.5
    ctc_lines: int = 3
    ctc_margin: float = 4.0
    # The tracker's push: the shown word unchanged `stuck_s` s while the reader makes sound (stop
    # detector under quiet_s), and the tracker `push_words` or more words ahead in the du'a (or on
    # another line): `push_p` of the belief moves to just before the tracker's word. In echo (a hall,
    # a phone far away) the frames blur, the belief can't find the next word and stalls; Whisper,
    # hearing 6 s at a time, still knows where they are. Evidence only re-weights belief already
    # there, so a stalled belief needs this push to get any on the tracker's word at all.
    stuck_s: float = 1.5
    push_words: int = 3
    push_p: float = 0.0  # off: more false jumps when readers go back, no gain in echo (docs/results/bench.md)


@dataclass
class _Dua:
    lo: int  # first word (corpus index)
    r: np.ndarray  # letters
    word_of: np.ndarray  # letter -> word - lo
    line_of_word: np.ndarray  # word - lo -> line number within the du'a
    word_first: np.ndarray  # word - lo -> its first letter
    word_last: np.ndarray  # word - lo -> its last letter
    line_first_word: np.ndarray  # line -> first word - lo
    line_last_word: np.ndarray  # line -> last word - lo
    diff: np.ndarray = field(default=None)  # letter j differs from j-1
    mid_cost: np.ndarray = field(default=None)  # per word: c_mid, NEG for a line's last word
    line_start_letter: np.ndarray = field(default=None)
    log_words_in_line: np.ndarray = field(default=None)  # per word
    line_lo: np.ndarray = field(default=None)  # per line: first letter
    line_hi: np.ndarray = field(default=None)  # ...and one past its last
    start_word: np.ndarray = field(default=None)  # per letter: the word it starts, -1 inside a word


def _lse(a: np.ndarray) -> float:
    m = float(np.max(a)) if a.size else NEG
    return m if m <= NEG / 2 else m + float(np.log(np.sum(np.exp(a - m))))


def _frame_numpy(L, B, F, IL, IB, lp_t, d: _Dua, cfg_arr, ri, diffi, exiti):
    """One frame of the forward pass. L, B: per letter (emitting it / blank after it); F: per line
    (the filler entered from that line); IL, IB: per line x interjection letter (the salawat said
    from that line). Log domain, unnormalized."""
    (c_restart, b1, b2, b3, s2, s3, s4, c_mid, c_rest, c_far, c_fill_in, fill_cost, c_fill_word,
     c_fill_next, c_int_in, c_int_next, c_int_word) = cfg_arr
    nl = d.line_first_word.size
    emit = lp_t[d.r]
    best = float(np.max(lp_t))
    # departures: from each word's end (its last letter, or the blank after it)
    we = np.logaddexp(L[d.word_last], B[d.word_last])
    end_src = we[d.line_last_word]  # the line finished
    mid = we + d.mid_cost  # c_mid for a word before the line's last, NEG for the last
    mid_src = np.logaddexp.reduceat(mid, d.line_first_word)
    line_src = np.logaddexp(end_src, mid_src)
    any_src = np.logaddexp.reduceat(we, d.line_first_word)  # any word end of the line, no c_mid
    # into line starts
    into = line_src + c_restart
    for off, c in ((1, b1), (2, b2), (3, b3)):  # back: line k from line k+off
        into[: nl - off] = np.logaddexp(into[: nl - off], line_src[off:] + c)
    for off, c in ((2, s2), (3, s3), (4, s4)):  # skip: line k from line k-off
        into[off:] = np.logaddexp(into[off:], line_src[: nl - off] + c)
    into[1:] = np.logaddexp(into[1:], mid_src[:-1] + c_rest)  # the rest of a line skipped
    m = float(np.max(line_src))
    if m > NEG / 2:
        far = m + float(np.log(np.sum(np.exp(line_src - m)))) + c_far - np.log(nl)
        into = np.logaddexp(into, far)
    # filler: entered from the line, held at the frame's best column less fill_cost, left again
    F_new = np.logaddexp(F, line_src + c_fill_in) + (best - fill_cost)
    into[1:] = np.logaddexp(into[1:], F[:-1] + c_fill_next)
    # interjection (salawat): entered from any word end of the line, read letter by letter, left again
    K = ri.size
    X = np.full(nl, NEG)
    if K:
        for k in np.flatnonzero(exiti):
            X = np.logaddexp(X, np.logaddexp(IL[:, k], IB[:, k]))
        ie = lp_t[ri]
        IL_prev = np.concatenate([np.full((nl, 1), NEG), IL[:, :-1]], axis=1)
        IB_prev = np.concatenate([np.full((nl, 1), NEG), IB[:, :-1]], axis=1)
        enter_i = np.logaddexp(IB_prev, np.where(diffi[None, :], IL_prev, NEG))
        enter_i[:, 0] = any_src + c_int_in
        IL_new = np.logaddexp(IL, enter_i) + ie[None, :]
        IB_new = np.logaddexp(IL, IB) + lp_t[0]
        into[1:] = np.logaddexp(into[1:], X[:-1] + c_int_next)
    else:
        IL_new, IB_new = IL, IB
    entry = np.full(d.r.size, NEG)
    entry[d.line_start_letter] = into
    fw = np.logaddexp(F[d.line_of_word] + c_fill_word, X[d.line_of_word] + c_int_word) - d.log_words_in_line
    entry[d.word_first] = np.logaddexp(entry[d.word_first], fw)
    prevL = np.empty_like(L)
    prevL[0], prevL[1:] = NEG, L[:-1]
    prevB = np.empty_like(B)
    prevB[0], prevB[1:] = NEG, B[:-1]
    enter = np.logaddexp(prevB, np.where(d.diff, prevL, NEG))
    L_new = np.logaddexp(np.logaddexp(L, enter), entry) + emit
    B_new = np.logaddexp(L, B) + lp_t[0]
    return L_new, B_new, F_new, IL_new, IB_new


def _lae(a, b):
    if a < b:
        a, b = b, a
    if b <= NEG / 2:
        return a
    return a + np.log1p(np.exp(b - a))


def _advance_loop(L, B, F, IL, IB, frames, temp, r, diff, word_first, word_last, line_first_word, line_last_word,
                  line_of_word, mid_cost, line_start_letter, log_words_in_line, cfg_arr, ri, diffi, exiti):
    """_frame_numpy over a run of frames, normalized after each (numba; same sums)."""
    (c_restart, b1, b2, b3, s2, s3, s4, c_mid, c_rest, c_far, c_fill_in, fill_cost, c_fill_word,
     c_fill_next, c_int_in, c_int_next, c_int_word) = cfg_arr
    J = r.size
    nw = word_first.size
    nl = line_first_word.size
    K = ri.size
    we = np.empty(nw)
    mid_src = np.empty(nl)
    line_src = np.empty(nl)
    any_src = np.empty(nl)
    into = np.empty(nl)
    entry = np.empty(J)
    Fn = np.empty(nl)
    X = np.empty(nl)
    backs = (b1, b2, b3)
    skips = (s2, s3, s4)
    for t in range(frames.shape[0]):
        best = NEG
        for c in range(frames.shape[1]):
            v = frames[t, c] / temp
            if v > best:
                best = v
        bl = frames[t, 0] / temp
        for w in range(nw):
            j = word_last[w]
            we[w] = _lae(L[j], B[j])
        for li in range(nl):
            acc = NEG
            acc2 = NEG
            for w in range(line_first_word[li], line_last_word[li] + 1):
                acc = _lae(acc, we[w] + mid_cost[w])
                acc2 = _lae(acc2, we[w])
            mid_src[li] = acc
            any_src[li] = acc2
            line_src[li] = _lae(we[line_last_word[li]], acc)
        m = NEG
        for li in range(nl):
            if line_src[li] > m:
                m = line_src[li]
        far = NEG
        if m > NEG / 2:
            ssum = 0.0
            for li in range(nl):
                ssum += np.exp(line_src[li] - m)
            far = m + np.log(ssum) + c_far - np.log(nl)
        for li in range(nl):
            x = NEG
            for k in range(K):
                if exiti[k]:
                    x = _lae(x, _lae(IL[li, k], IB[li, k]))
            X[li] = x
        for k in range(nl):
            v = line_src[k] + c_restart
            for o in range(3):
                src = k + o + 1
                if src < nl:
                    v = _lae(v, line_src[src] + backs[o])
                src = k - (o + 2)
                if src >= 0:
                    v = _lae(v, line_src[src] + skips[o])
            if k >= 1:
                v = _lae(v, mid_src[k - 1] + c_rest)
                v = _lae(v, F[k - 1] + c_fill_next)
                if K > 0:
                    v = _lae(v, X[k - 1] + c_int_next)
            v = _lae(v, far)
            into[k] = v
        for li in range(nl):
            Fn[li] = _lae(F[li], line_src[li] + c_fill_in) + (best - fill_cost)
        # interjection letters, right to left (k - 1 is still the previous frame's)
        for li in range(nl):
            for k in range(K - 1, -1, -1):
                if k == 0:
                    enter_i = any_src[li] + c_int_in
                else:
                    enter_i = IB[li, k - 1]
                    if diffi[k]:
                        enter_i = _lae(enter_i, IL[li, k - 1])
                nb = _lae(IL[li, k], IB[li, k]) + bl
                IL[li, k] = _lae(IL[li, k], enter_i) + frames[t, ri[k]] / temp
                IB[li, k] = nb
        for j in range(J):
            entry[j] = NEG
        for k in range(nl):
            entry[line_start_letter[k]] = into[k]
        for w in range(nw):
            j = word_first[w]
            li = line_of_word[w]
            if K > 0:
                fw = _lae(F[li] + c_fill_word, X[li] + c_int_word)
            else:
                fw = F[li] + c_fill_word
            entry[j] = _lae(entry[j], fw - log_words_in_line[w])
        mx = NEG
        for j in range(J - 1, -1, -1):  # right to left: L[j-1], B[j-1] are still the previous frame's
            enter = NEG
            if j > 0:
                enter = B[j - 1]
                if diff[j]:
                    enter = _lae(enter, L[j - 1])
            nb = _lae(L[j], B[j]) + bl
            L[j] = _lae(_lae(L[j], enter), entry[j]) + frames[t, r[j]] / temp
            B[j] = nb
            if L[j] > mx:
                mx = L[j]
            if nb > mx:
                mx = nb
        for li in range(nl):
            F[li] = Fn[li]
            if Fn[li] > mx:
                mx = Fn[li]
            for k in range(K):
                if IL[li, k] > mx:
                    mx = IL[li, k]
                if IB[li, k] > mx:
                    mx = IB[li, k]
        for j in range(J):
            L[j] -= mx
            B[j] -= mx
        for li in range(nl):
            F[li] -= mx
            for k in range(K):
                IL[li, k] -= mx
                IB[li, k] -= mx
    return L, B, F, IL, IB


def _advance_beam(L, B, F, IL, IB, frames, temp, r, diff, word_first, word_last, line_first_word, line_last_word,
                  line_of_word, mid_cost, line_start_letter, log_words_in_line, cfg_arr, ri, diffi, exiti,
                  active, line_lo, line_hi, start_word):
    """_advance_loop over the active lines only (a beam): states of other lines are NEG and stay so.
    With every line active it is _advance_loop exactly."""
    (c_restart, b1, b2, b3, s2, s3, s4, c_mid, c_rest, c_far, c_fill_in, fill_cost, c_fill_word,
     c_fill_next, c_int_in, c_int_next, c_int_word) = cfg_arr
    nw = word_first.size
    nl = line_first_word.size
    K = ri.size
    we = np.full(nw, NEG)
    mid_src = np.full(nl, NEG)
    line_src = np.full(nl, NEG)
    any_src = np.full(nl, NEG)
    into = np.full(nl, NEG)
    Fn = np.full(nl, NEG)
    X = np.full(nl, NEG)
    backs = (b1, b2, b3)
    skips = (s2, s3, s4)
    act = np.flatnonzero(active)
    n_act = act.size
    for t in range(frames.shape[0]):
        best = NEG
        for c in range(frames.shape[1]):
            v = frames[t, c] / temp
            if v > best:
                best = v
        bl = frames[t, 0] / temp
        for ai in range(n_act):
            li = act[ai]
            acc = NEG
            acc2 = NEG
            for w in range(line_first_word[li], line_last_word[li] + 1):
                j = word_last[w]
                we[w] = _lae(L[j], B[j])
                acc = _lae(acc, we[w] + mid_cost[w])
                acc2 = _lae(acc2, we[w])
            mid_src[li] = acc
            any_src[li] = acc2
            line_src[li] = _lae(we[line_last_word[li]], acc)
        m = NEG
        for ai in range(n_act):
            if line_src[act[ai]] > m:
                m = line_src[act[ai]]
        far = NEG
        if m > NEG / 2:
            ssum = 0.0
            for ai in range(n_act):
                ssum += np.exp(line_src[act[ai]] - m)
            far = m + np.log(ssum) + c_far - np.log(nl)
        for ai in range(n_act):
            li = act[ai]
            x = NEG
            for k in range(K):
                if exiti[k]:
                    x = _lae(x, _lae(IL[li, k], IB[li, k]))
            X[li] = x
        for ai in range(n_act):
            k = act[ai]
            v = line_src[k] + c_restart
            for o in range(3):
                src = k + o + 1
                if src < nl:
                    v = _lae(v, line_src[src] + backs[o])
                src = k - (o + 2)
                if src >= 0:
                    v = _lae(v, line_src[src] + skips[o])
            if k >= 1:
                v = _lae(v, mid_src[k - 1] + c_rest)
                v = _lae(v, F[k - 1] + c_fill_next)
                if K > 0:
                    v = _lae(v, X[k - 1] + c_int_next)
            v = _lae(v, far)
            into[k] = v
        for ai in range(n_act):
            li = act[ai]
            Fn[li] = _lae(F[li], line_src[li] + c_fill_in) + (best - fill_cost)
        for ai in range(n_act):
            li = act[ai]
            for k in range(K - 1, -1, -1):
                if k == 0:
                    enter_i = any_src[li] + c_int_in
                else:
                    enter_i = IB[li, k - 1]
                    if diffi[k]:
                        enter_i = _lae(enter_i, IL[li, k - 1])
                nb = _lae(IL[li, k], IB[li, k]) + bl
                IL[li, k] = _lae(IL[li, k], enter_i) + frames[t, ri[k]] / temp
                IB[li, k] = nb
        mx = NEG
        for ai in range(n_act - 1, -1, -1):  # lines high to low, letters right to left
            li = act[ai]
            fw_line = _lae(F[li] + c_fill_word, X[li] + c_int_word) if K > 0 else F[li] + c_fill_word
            for j in range(line_hi[li] - 1, line_lo[li] - 1, -1):
                e = NEG  # entry, in _advance_loop's order: the line's start, then a word's start
                if j == line_start_letter[li]:
                    e = into[li]
                w = start_word[j]
                if w >= 0:
                    e = _lae(e, fw_line - log_words_in_line[w])
                enter = NEG
                if j > 0:
                    enter = B[j - 1]
                    if diff[j]:
                        enter = _lae(enter, L[j - 1])
                nb = _lae(L[j], B[j]) + bl
                L[j] = _lae(_lae(L[j], enter), e) + frames[t, r[j]] / temp
                B[j] = nb
                if L[j] > mx:
                    mx = L[j]
                if nb > mx:
                    mx = nb
        for ai in range(n_act):
            li = act[ai]
            F[li] = Fn[li]
            if Fn[li] > mx:
                mx = Fn[li]
            for k in range(K):
                if IL[li, k] > mx:
                    mx = IL[li, k]
                if IB[li, k] > mx:
                    mx = IB[li, k]
        for ai in range(n_act):
            li = act[ai]
            for j in range(line_lo[li], line_hi[li]):
                L[j] -= mx
                B[j] -= mx
            F[li] -= mx
            for k in range(K):
                IL[li, k] -= mx
                IB[li, k] -= mx
    return L, B, F, IL, IB


_advance_jit = _advance_beam_jit = None
if njit is not None:
    _lae = njit(cache=True, nogil=True)(_lae)
    _advance_jit = njit(cache=True, nogil=True)(_advance_loop)
    _advance_beam_jit = njit(cache=True, nogil=True)(_advance_beam)


class StreamFollower:
    def __init__(self, index: CorpusIndex, config: StreamConfig | None = None, use_jit: bool = True):
        self.ix = index
        self.cfg = config or StreamConfig()
        self.use_jit = use_jit
        ix = index
        self._word_letter = np.r_[0, np.flatnonzero(np.diff(ix.letter_word) != 0) + 1, ix.letters.size]
        self._duas: dict[int, _Dua] = {}
        self.reset()

    def reset(self) -> None:
        self.dua: int | None = None
        self.L = self.B = self.F = self.IL = self.IB = None
        self.active = None  # per line: in the beam
        self._proposed = None  # lines the tracker last put propose_mass on
        self._ctc_prop = None  # lines the frames last proposed
        self._steps = 0
        self._shown_since = None  # when the shown word last changed
        self._shown_w = None
        self.word: int | None = None
        self._cand: int | None = None
        self._cand_n = 0
        self._t_stream = None  # end time of the last committed frame
        self._last_anchor_t = None
        self._lapse_since = None
        self._lapse_voice = 0.0
        self._switch_to = None  # a du'a the tracker moved to, and since when
        self._switch_since = None

    def _dua(self, d: int) -> _Dua:
        if d not in self._duas:
            ix = self.ix
            lo, hi = ix.dua_word_span[d]
            la, lb = self._word_letter[lo], self._word_letter[hi]
            r = ix.letters[la:lb].astype(np.int64)
            word_of = (ix.letter_word[la:lb] - lo).astype(np.int64)
            nw = hi - lo
            word_first = np.searchsorted(word_of, np.arange(nw), side="left")
            word_last = np.searchsorted(word_of, np.arange(nw), side="right") - 1
            seg = ix.word_segment[lo:hi]
            new_line = np.r_[True, seg[1:] != seg[:-1]]
            line_of_word = np.cumsum(new_line) - 1
            line_first = np.flatnonzero(new_line)
            line_last = np.r_[line_first[1:] - 1, nw - 1]
            dd = _Dua(lo, r, word_of, line_of_word.astype(np.int64), word_first.astype(np.int64),
                      word_last.astype(np.int64), line_first.astype(np.int64), line_last.astype(np.int64))
            dd.diff = np.r_[True, r[1:] != r[:-1]]
            dd.mid_cost = np.full(nw, self.cfg.c_mid)
            dd.mid_cost[dd.line_last_word] = NEG
            dd.line_start_letter = dd.word_first[dd.line_first_word]
            nwl = np.bincount(dd.line_of_word, minlength=dd.line_first_word.size)
            dd.log_words_in_line = np.log(nwl[dd.line_of_word])
            dd.line_lo = dd.word_first[dd.line_first_word].astype(np.int64)
            dd.line_hi = (dd.word_last[dd.line_last_word] + 1).astype(np.int64)
            dd.start_word = np.full(r.size, -1, dtype=np.int64)
            dd.start_word[dd.word_first] = np.arange(nw)
            self._duas[d] = dd
        return self._duas[d]

    def _cfg_arr(self):
        c = self.cfg
        return (c.c_restart, *c.c_back, *c.c_skip, c.c_mid, c.c_mid + c.c_rest, c.c_far, c.c_fill_in, c.fill_cost,
                c.c_fill_word, c.c_fill_next, c.c_int_in, c.c_int_next, c.c_int_word)

    def _interjection(self):
        """Letters of the interjection chain, which letters differ from the one before, and where it may end
        (after each |-separated part)."""
        if not hasattr(self, "_ichain"):
            from .align import encode
            from .text import normalize

            r, ends = [], []
            for form in self.cfg.interjections:
                parts = form.split("|")
                if r:  # forms after the first aren't supported: one chain
                    break
                for part in parts:
                    r += [int(c) for c in encode(normalize(part).replace(" ", ""))]
                    ends.append(len(r) - 1)
            ri = np.array(r, dtype=np.int64)
            diffi = np.r_[True, ri[1:] != ri[:-1]] if ri.size else np.zeros(0, dtype=bool)
            exiti = np.zeros(ri.size, dtype=np.bool_)
            exiti[ends] = True
            self._ichain = (ri, diffi.astype(np.bool_), exiti)
        return self._ichain

    def _start(self, d: int, hmm_word: int, line_mass=None) -> None:
        """A new du'a: belief spread over word ends, weighted by the tracker's line belief."""
        dd = self._dua(d)
        self.dua = d
        nl = dd.line_first_word.size
        w_line = np.full(nl, self.cfg.tracker_floor)
        if line_mass is not None:
            for li in range(nl):
                w_line[li] += line_mass(int(dd.lo + dd.line_first_word[li]))
        else:
            w_line[dd.line_of_word[hmm_word - dd.lo]] += 1.0
        w_line /= w_line.sum()
        nw_line = np.bincount(dd.line_of_word, minlength=nl)
        self.L = np.full(dd.r.size, NEG)
        self.B = np.full(dd.r.size, NEG)
        self.B[dd.word_last] = np.log(w_line[dd.line_of_word] / nw_line[dd.line_of_word])
        self.F = np.full(nl, NEG)
        K = self._interjection()[0].size
        self.IL = np.full((nl, K), NEG)
        self.IB = np.full((nl, K), NEG)
        self.active = np.ones(nl, dtype=np.uint8)
        self._proposed = None
        self.word = int(hmm_word)

    def _advance(self, L, B, F, IL, IB, frames: np.ndarray):
        dd = self._dua(self.dua)
        cfg = self._cfg_arr()
        ri, diffi, exiti = self._interjection()
        if len(frames) and not np.isfinite(frames).all():  # a broken window (fp16 overflow on silence): no evidence
            frames = np.where(np.isfinite(frames), frames, 0.0)
        if len(frames) and self.cfg.blank_bias:
            frames = np.array(frames, dtype=np.float64)
            frames[:, 0] -= self.cfg.blank_bias
        if not len(frames):
            return L, B, F, IL, IB
        if _advance_beam_jit is not None and self.use_jit and self.cfg.beam > 0:
            return _advance_beam_jit(L.copy(), B.copy(), F.copy(), IL.copy(), IB.copy(),
                                     np.ascontiguousarray(frames, dtype=np.float64), float(self.cfg.temp), dd.r,
                                     dd.diff, dd.word_first, dd.word_last, dd.line_first_word, dd.line_last_word,
                                     dd.line_of_word, dd.mid_cost, dd.line_start_letter, dd.log_words_in_line,
                                     np.array(cfg, dtype=np.float64), ri, diffi, exiti, self.active, dd.line_lo,
                                     dd.line_hi, dd.start_word)
        if _advance_jit is not None and self.use_jit:
            return _advance_jit(L.copy(), B.copy(), F.copy(), IL.copy(), IB.copy(),
                                np.ascontiguousarray(frames, dtype=np.float64), float(self.cfg.temp), dd.r, dd.diff,
                                dd.word_first, dd.word_last, dd.line_first_word, dd.line_last_word, dd.line_of_word,
                                dd.mid_cost, dd.line_start_letter, dd.log_words_in_line,
                                np.array(cfg, dtype=np.float64), ri, diffi, exiti)
        for f in frames:
            L, B, F, IL, IB = _frame_numpy(L, B, F, IL, IB, f / self.cfg.temp, dd, cfg, ri, diffi, exiti)
            m = max(float(np.max(L)), float(np.max(B)), float(np.max(F)), float(np.max(IL)) if IL.size else NEG,
                    float(np.max(IB)) if IB.size else NEG)
            L, B, F, IL, IB = L - m, B - m, F - m, IL - m, IB - m
        return L, B, F, IL, IB

    def _prune(self) -> None:
        """The beam for the next step: lines with belief, their neighbours, the shown word's line, the
        tracker's proposals. Lines leaving it are dropped (NEG)."""
        cfg = self.cfg
        if cfg.beam <= 0:
            return
        dd = self._dua(self.dua)
        nl = dd.line_first_word.size
        st = np.logaddexp(self.L, self.B)
        m = max(float(np.max(st)), float(np.max(self.F)), float(np.max(self.IL)) if self.IL.size else NEG)
        p = np.exp(st - m)
        mass = np.zeros(nl)
        np.add.at(mass, dd.line_of_word[dd.word_of], p)
        mass += np.exp(self.F - m)
        if self.IL.size:
            mass += np.exp(self.IL - m).sum(axis=1) + np.exp(self.IB - m).sum(axis=1)
        keep = mass >= cfg.beam * mass.sum()
        on = np.zeros(nl, dtype=bool)
        k = cfg.beam_margin
        for li in np.flatnonzero(keep):
            on[max(0, li - k) : li + k + 1] = True
        if self.word is not None and self.ix.word_dua[self.word] == self.dua:
            li = dd.line_of_word[self.word - dd.lo]
            on[max(0, li - 1) : li + 2] = True
        for prop in (self._proposed, self._ctc_prop):
            if prop is not None:
                for li in prop:
                    on[max(0, li - 1) : li + 2] = True
        off = (self.active == 1) & ~on
        for li in np.flatnonzero(off):
            a, b = dd.line_lo[li], dd.line_hi[li]
            self.L[a:b] = NEG
            self.B[a:b] = NEG
            self.F[li] = NEG
            if self.IL.size:
                self.IL[li] = NEG
                self.IB[li] = NEG
        self.active = on.astype(np.uint8)

    def _frame_proposals(self, lp: np.ndarray) -> None:
        """Lines whose text explains the latest ctc_window s best (end_scores over the whole du'a)."""
        from .ctc_align import end_scores, squeeze_blanks

        cfg = self.cfg
        dd = self._dua(self.dua)
        sub = squeeze_blanks(lp[-max(1, int(round(cfg.ctc_window / cfg.frame_s))) :])
        sc = end_scores(sub, dd.r)
        per_word = np.maximum.reduceat(sc, dd.word_first)
        per_line = np.maximum.reduceat(per_word, dd.line_first_word)
        best = float(per_line.max())
        order = np.argsort(-per_line, kind="stable")
        self._ctc_prop = [int(li) for li in order[: cfg.ctc_lines] if per_line[li] >= best - cfg.ctc_margin]

    def _push(self, w: int) -> None:
        """Move push_p of the belief to the blank before word w (its line made active)."""
        dd = self._dua(self.dua)
        k = w - dd.lo
        li = int(dd.line_of_word[k])
        if self.active is not None:
            self.active[max(0, li - 1) : li + 2] = 1
        p = self.cfg.push_p
        sl = [self.L, self.B, self.F] + ([self.IL, self.IB] if self.IL.size else [])
        for a in sl:
            a += np.log(1 - p)
        if k > 0:
            j = int(dd.word_last[k - 1])
            self.B[j] = np.logaddexp(self.B[j], np.log(p))
        else:
            j = int(dd.word_first[0])
            self.L[j] = np.logaddexp(self.L[j], np.log(p))

    def _tracker_evidence(self, line_mass) -> None:
        if line_mass is None or self.cfg.tracker_weight <= 0:
            return
        dd = self._dua(self.dua)
        nl = dd.line_first_word.size
        m = np.array([line_mass(int(dd.lo + dd.line_first_word[li])) for li in range(nl)])
        self._proposed = np.flatnonzero(m >= self.cfg.propose_mass)
        g = self.cfg.tracker_weight * np.log(m + self.cfg.tracker_floor)
        g -= g.max()
        self.L = self.L + g[dd.line_of_word[dd.word_of]]
        self.B = self.B + g[dd.line_of_word[dd.word_of]]
        self.F = self.F + g
        self.IL = self.IL + g[:, None]
        self.IB = self.IB + g[:, None]

    def posterior(self, L, B, F, IL=None, IB=None) -> tuple[np.ndarray, float]:
        """Belief per word of the du'a, and the share off the text (filler, interjection)."""
        dd = self._dua(self.dua)
        st = np.logaddexp(L, B)
        off = [F]
        if IL is not None and IL.size:
            off += [IL.ravel(), IB.ravel()]
        offv = np.concatenate(off)
        m = max(float(np.max(st)), float(np.max(offv)))
        p = np.exp(st - m)
        pw = np.bincount(dd.word_of, weights=p, minlength=dd.word_first.size)
        pf = float(np.exp(offv - m).sum())
        z = pw.sum() + pf
        return pw / z, pf / z

    def _quiet(self, frames: np.ndarray, quiet: bool) -> np.ndarray:
        if not quiet or not len(frames) or self.cfg.quiet_pen <= 0:
            return frames
        out = np.array(frames, dtype=np.float64)
        out[:, 1:] -= self.cfg.quiet_pen
        return out

    def step(self, lp: np.ndarray, t: float, hmm_word: int | None, quiet_now: float | None = None,
             line_mass=None, anchor_t: float | None = None, quiet_then: float | None = None) -> int | None:
        w = self._step(lp, t, hmm_word, quiet_now, line_mass, anchor_t, quiet_then)
        self._note_shown(t)
        return w

    def _step(self, lp: np.ndarray, t: float, hmm_word: int | None, quiet_now: float | None = None,
              line_mass=None, anchor_t: float | None = None, quiet_then: float | None = None) -> int | None:
        """One CTC window (frames x columns) ending at t. `quiet_now`: seconds without voice at t
        (the page's stop detector); `quiet_then`: the same `lookahead` earlier (None: quiet_now
        less the lookahead). Returns the word to show (None: nothing)."""
        ix, cfg = self.ix, self.cfg
        if hmm_word is None:  # the tracker has lost the du'a: a lull, a long pause, or a text it doesn't know
            if self.dua is None:
                return None
            if self._lapse_since is None:
                self._lapse_since = t
                self._lapse_voice = 0.0
            if quiet_now is not None and quiet_now < cfg.lapse_quiet:
                self._lapse_voice += cfg.hop
            if t - self._lapse_since > cfg.lapse_s or self._lapse_voice > cfg.lapse_voice:
                self.reset()
                return None
            hmm_word = self.word  # carry on in the du'a it was in
        else:
            self._lapse_since = None
        d = int(ix.word_dua[hmm_word])
        if d == self.dua:
            self._switch_to = self._switch_since = None
        elif self.dua is not None and cfg.switch_s > 0:
            if self._switch_to != d:
                self._switch_to, self._switch_since = d, t
            if t - self._switch_since < cfg.switch_s:
                hmm_word, line_mass, d = self.word, None, self.dua  # not yet: on in the du'a it was in
        if d != self.dua:
            self._start(d, hmm_word, line_mass)
            self._t_stream = None
        if self.dua is None:
            return None
        if anchor_t is not None and anchor_t != self._last_anchor_t:
            self._last_anchor_t = anchor_t
            if d == self.dua:
                self._tracker_evidence(line_mass)
        n = lp.shape[0]
        la = int(round(cfg.lookahead / cfg.frame_s))
        hop = int(round(cfg.hop / cfg.frame_s))
        commit_end = n - la  # frames [.., commit_end) are committed
        if self._t_stream is None:
            new = lp[max(0, commit_end - hop) : commit_end]
        else:
            k = int(round((t - cfg.lookahead - self._t_stream) / cfg.frame_s))
            new = lp[max(0, commit_end - k) : commit_end] if k > 0 else lp[0:0]
        self._t_stream = t - cfg.lookahead
        if quiet_then is None and quiet_now is not None:
            quiet_then = quiet_now - cfg.lookahead
        q_then = quiet_then is not None and quiet_then >= cfg.quiet_s
        q_now = quiet_now is not None and quiet_now >= cfg.quiet_s
        self.L, self.B, self.F, self.IL, self.IB = self._advance(self.L, self.B, self.F, self.IL, self.IB,
                                                                 self._quiet(new, q_then))
        self._steps += 1
        # stalled while the reader sounds, the tracker well ahead (or on another line): push
        if cfg.push_p > 0 and self._shown_since is not None and t - self._shown_since >= cfg.stuck_s                 and quiet_now is not None and quiet_now < cfg.quiet_s and self.word is not None                 and hmm_word is not None and ix.word_dua[hmm_word] == self.dua and self._lapse_since is None:
            dd0 = self._dua(self.dua)
            ahead = hmm_word - self.word >= cfg.push_words
            other = dd0.line_of_word[hmm_word - dd0.lo] != dd0.line_of_word[self.word - dd0.lo] and hmm_word > self.word
            if ahead or other:
                self._push(int(hmm_word))
                self._shown_since = t
        if cfg.ctc_every > 0 and self._steps % cfg.ctc_every == 0:
            if q_now:
                self._ctc_prop = None
            else:
                self._frame_proposals(lp)
        self._prune()
        # the newest frames, tentatively
        L, B, F, IL, IB = self._advance(self.L, self.B, self.F, self.IL, self.IB, self._quiet(lp[commit_end:], q_now))
        pw, pf = self.posterior(L, B, F, IL, IB)
        dd = self._dua(self.dua)
        if pf > 0.5:  # talking, or something else: hold
            self._cand, self._cand_n = None, 0
            return self.word
        best = int(np.argmax(pw))
        w = dd.lo + best
        cur = self.word
        if cur is None or ix.word_dua[cur] != self.dua:
            self.word = w
            return w
        if w == cur:
            self._cand, self._cand_n = None, 0
            return cur
        same_line = dd.line_of_word[best] == dd.line_of_word[cur - dd.lo]
        next_line = dd.line_of_word[best] == dd.line_of_word[cur - dd.lo] + 1 and best == dd.line_first_word[dd.line_of_word[best]]
        if same_line:
            need_p, need_n = cfg.show_p, 1 if best > cur - dd.lo else cfg.line_steps
        elif next_line:
            need_p, need_n = cfg.next_p, cfg.next_steps
        else:
            need_p, need_n = cfg.line_p, cfg.line_steps
        if pw[best] < need_p:
            self._cand, self._cand_n = None, 0
            return cur
        self._cand_n = self._cand_n + 1 if self._cand == w else 1
        self._cand = w
        if self._cand_n >= need_n:
            self.word, self._cand, self._cand_n = w, None, 0
        return self.word

    def _note_shown(self, t: float) -> None:
        if self.word != self._shown_w:
            self._shown_w, self._shown_since = self.word, t
