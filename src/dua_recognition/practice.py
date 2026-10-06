"""Practice mode: did the reader say each line, where it stands?

The reading display forgives: it holds through talk, waits for evidence before moving on, and
needn't show every short line (docs/results/practice_bench.md: used as a checker it marks 0.8
read lines per 10 minutes and misses a quarter of the lines left out). A checker needs the
opposite, evidence that line k was said between its neighbours. So, once the reader is two lines
past it, the CTC frames since just before line k-1 are explained twice:

    with k       line k-1, line k, line k+1
    without k    line k-1, line k+1

each with a free start anywhere in its first line, a free end anywhere in its last, and the
filler (anything else said: talk, a salawat, a stumble; any letter at fill_cost below the
frame's best, a blank free) allowed between lines at c_fill. The difference (nats) says how much
line k's own letters explain that stretch better than leaving it out. It is position-aware: a
refrain line left out has no frames of its own between its neighbours, even though its words
are heard elsewhere; a short line the display skipped over still has its letters there.

    llr >= theta_heard      heard
    llr <= -theta_unheard   left out
    otherwise               not sure

Only the frames the phone would have when it decides (committed, `lookahead` before the step)
are used. The display decides when (it reaches line k+2, or the session ends) and roughly
where (the span starts before the display left line k-2); the frames decide what.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEG = -1e30

try:  # the per-frame pass over a few lines' letters, ~100x faster
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


@dataclass
class PracticeConfig:
    """Defaults tuned on a dev sample of the scenario bench (docs/results/practice_bench.md)."""
    fill_cost: float = 1.5  # the filler: any letter at this many nats below the frame's best, a blank free
    c_fill: float = -8.0  # entering the filler between two lines (stream_follower.c_fill_in)
    c_skip: float = -4.0  # leaving out a line other than the one judged
    max_later: int = 4  # lines after the judged one it may go on into
    settle_s: float = 0.5  # line k is decided once the display has stayed on line k+2 or later this long
    span_s: float | None = None  # the span: this many seconds before the decision (None: from the display's times)
    span_min_s: float = 20.0  # ...from the display's times, but at least this long (a display ahead of the reader)
    min_frac: float = 0.5  # saying line k means at least its first word and this share of its letters
    margin_s: float = 1.5  # the span starts this long before the display left the line before the context
    max_span_s: float = 45.0  # ...but no further back than this
    theta_heard: float = 4.0  # heard at score >= this
    theta_unheard: float = 4.0  # left out at score <= -this, if the display never showed the line...
    theta_shown: float | None = 20.0  # ...and at <= -this even if it did (None: the display isn't asked)
    # the sound has to be good enough to check: a line is left out only if, of the last gate_lines
    # lines decided before it (left-out ones aside), at least gate_heard were heard (hall echo, a
    # phone far away: most lines come back not sure, and a line left out is then not sure either)
    gate_lines: int = 4
    gate_heard: float = 0.6
    gate_min: int = 1


def verdict(score: float, shown: bool, cfg: PracticeConfig) -> str:
    """"heard", "left out" or "not sure" for a line's score (before the sound gate)."""
    if score <= -cfg.theta_unheard and (cfg.theta_shown is None or not shown or score <= -cfg.theta_shown):
        return "left out"
    return "heard" if score >= cfg.theta_heard else "not sure"


def gated(v: str, history: list[str], cfg: PracticeConfig) -> str:
    """The sound gate: a left-out verdict stands only after enough lines heard lately (history: the
    verdicts given so far, in order, left-out ones aside). Updates history."""
    if v == "left out" and cfg.gate_lines > 0:
        recent = history[-cfg.gate_lines :]
        if len(recent) < cfg.gate_min or sum(x == "heard" for x in recent) < cfg.gate_heard * len(recent) - 1e-9:
            v = "not sure"
    if v != "left out":
        history.append(v)
    return v


def gated_verdicts(records: list, cfg: PracticeConfig) -> list[str]:
    """The verdicts the reader is shown for a session's records (line, t, score, shown, ...), in the
    order they were decided."""
    history: list[str] = []
    order = sorted(range(len(records)), key=lambda i: (records[i][1], records[i][0]))
    out = [""] * len(records)
    for i in order:
        out[i] = gated(verdict(records[i][2], records[i][3], cfg), history, cfg)
    return out


def _hyp_score_numpy(lp, r, seg_end, n_first, last_start, fill_cost, c_fill):
    """Best path log-likelihood of frames lp (T x C) through letters r (segments = lines, seg_end
    True on each line's last letter): a free start in the first n_first letters, a free end
    from letter last_start on, the filler after any line's last letter (entered at c_fill)."""
    T, n = lp.shape[0], r.size
    L = np.full(n, NEG)
    B = np.full(n, NEG)
    Fm = np.full(n, NEG)
    e = lp[0]
    fe = max(e[0], e[1:].max() - fill_cost)
    Fpre, Fpost = fe, NEG
    L[:n_first] = e[r[:n_first]]
    same = np.r_[True, r[1:] == r[:-1]]
    for t in range(1, T):
        e = lp[t]
        fe = max(e[0], e[1:].max() - fill_cost)
        enter = np.full(n, NEG)
        enter[1:] = np.maximum(B[:-1], np.where(same[1:], NEG, L[:-1]))
        enter[1:] = np.maximum(enter[1:], np.where(seg_end[:-1], Fm[:-1], NEG))
        enter[:n_first] = np.maximum(enter[:n_first], Fpre)
        end = np.maximum(L, B)
        Fpost = max(Fpost, float(end[last_start:].max())) + fe
        Fm = np.where(seg_end, np.maximum(Fm, end + c_fill) + fe, NEG)
        L, B = np.maximum(L, enter) + e[r], end + e[0]
        Fpre = Fpre + fe
    return max(Fpost, float(np.maximum(L, B)[last_start:].max()))


if njit is not None:
    @njit(cache=True)
    def _hyp_score_jit(lp, r, seg_end, n_first, last_start, fill_cost, c_fill):  # pragma: no cover - same sums
        T, n = lp.shape[0], r.size
        L = np.full(n, NEG)
        B = np.full(n, NEG)
        Fm = np.full(n, NEG)
        nL = np.empty(n)
        nB = np.empty(n)
        nF = np.empty(n)
        e = lp[0]
        best = NEG
        for c in range(1, lp.shape[1]):
            best = max(best, e[c])
        fe = max(e[0], best - fill_cost)
        Fpre, Fpost = fe, NEG
        for j in range(n_first):
            L[j] = e[r[j]]
        for t in range(1, T):
            e = lp[t]
            best = NEG
            for c in range(1, lp.shape[1]):
                best = max(best, e[c])
            fe = max(e[0], best - fill_cost)
            m = NEG
            for j in range(last_start, n):
                m = max(m, max(L[j], B[j]))
            for j in range(n):
                enter = NEG
                if j > 0:
                    enter = B[j - 1]
                    if r[j] != r[j - 1]:
                        enter = max(enter, L[j - 1])
                    if seg_end[j - 1]:
                        enter = max(enter, Fm[j - 1])
                if j < n_first:
                    enter = max(enter, Fpre)
                end = max(L[j], B[j])
                nL[j] = max(L[j], enter) + e[r[j]]
                nB[j] = end + e[0]
                nF[j] = max(Fm[j], end + c_fill) + fe if seg_end[j] else NEG
            Fpost = max(Fpost, m) + fe
            Fpre = Fpre + fe
            for j in range(n):
                L[j], B[j], Fm[j] = nL[j], nB[j], nF[j]
        m = Fpost
        for j in range(last_start, n):
            m = max(m, max(L[j], B[j]))
        return m
else:  # pragma: no cover
    _hyp_score_jit = None


def hyp_score(lp: np.ndarray, lines: list[np.ndarray], fill_cost: float, c_fill: float,
              use_jit: bool = True, end_from: int | None = None) -> float:
    """lp (T x C) explained by these lines' letters in order (see _hyp_score_numpy), ending
    anywhere from line `end_from` on (default: the last line)."""
    if not lines or not lp.shape[0] or any(not x.size for x in lines):
        return NEG
    r = np.concatenate(lines).astype(np.int64)
    seg_end = np.zeros(r.size, dtype=np.bool_)
    seg_end[np.cumsum([x.size for x in lines]) - 1] = True
    k = len(lines) - 1 if end_from is None else end_from
    n_first, last_start = lines[0].size, sum(x.size for x in lines[:k])
    lp = np.ascontiguousarray(lp, dtype=np.float64)
    if _hyp_score_jit is not None and use_jit:
        return float(_hyp_score_jit(lp, r, seg_end, n_first, last_start, float(fill_cost), float(c_fill)))
    return float(_hyp_score_numpy(lp, r, seg_end, n_first, last_start, fill_cost, c_fill))


def _span_score_numpy(lp, r, seg_of, seg_first, seg_last, optional, n_first, end_from, fill_cost, c_fill, c_skip):
    """Best path of frames lp through segments (lines) of letters r in order, where an optional
    segment may be left out (c_skip each) and a mandatory one may not: a free start in the
    first n_first letters, the filler after any segment (c_fill to enter), a free end from letter
    end_from on. seg_of: segment per letter; seg_first/seg_last: each segment's letter range."""
    T, n, S = lp.shape[0], r.size, seg_first.size
    L = np.full(n, NEG)
    B = np.full(n, NEG)
    Fm = np.full(S, NEG)
    e = lp[0]
    fe = max(e[0], e[1:].max() - fill_cost)
    Fpre, Fpost = fe, NEG
    L[:n_first] = e[r[:n_first]]
    same = np.r_[True, r[1:] == r[:-1]]
    first_of = np.zeros(n, dtype=bool)
    first_of[seg_first] = True
    for t in range(1, T):
        e = lp[t]
        fe = max(e[0], e[1:].max() - fill_cost)
        end = np.maximum(L, B)
        exit_s = np.maximum(end[seg_last], Fm)  # segment s finished (or the filler after it)
        into = np.full(S, NEG)  # into segment s' from an earlier one, the optional ones between left out
        for s2 in range(1, S):
            cost = 0.0
            for s in range(s2 - 1, -1, -1):
                into[s2] = max(into[s2], exit_s[s] + cost)
                if not optional[s]:
                    break
                cost += c_skip
        enter = np.full(n, NEG)
        inner = ~first_of
        inner[0] = False
        idx = np.flatnonzero(inner)
        enter[idx] = np.maximum(B[idx - 1], np.where(same[idx], NEG, L[idx - 1]))
        enter[seg_first[1:]] = into[1:]
        enter[:n_first] = np.maximum(enter[:n_first], Fpre)
        Fpost = max(Fpost, float(end[end_from:].max())) + fe
        Fm = np.maximum(Fm, end[seg_last] + c_fill) + fe
        L, B = np.maximum(L, enter) + e[r], end + e[0]
        Fpre = Fpre + fe
    return max(Fpost, float(np.maximum(L, B)[end_from:].max()))


if njit is not None:
    @njit(cache=True)
    def _span_score_jit(lp, r, seg_of, seg_first, seg_last, optional, n_first, end_from, fill_cost, c_fill,
                        c_skip):  # pragma: no cover - same sums
        T, n, S, C = lp.shape[0], r.size, seg_first.size, lp.shape[1]
        L = np.full(n, NEG)
        B = np.full(n, NEG)
        Fm = np.full(S, NEG)
        nL = np.empty(n)
        nB = np.empty(n)
        exit_s = np.empty(S)
        into = np.empty(S)
        e = lp[0]
        best = NEG
        for c in range(1, C):
            best = max(best, e[c])
        Fpre, Fpost = max(e[0], best - fill_cost), NEG
        for j in range(n_first):
            L[j] = e[r[j]]
        for t in range(1, T):
            e = lp[t]
            best = NEG
            for c in range(1, C):
                best = max(best, e[c])
            fe = max(e[0], best - fill_cost)
            for s in range(S):
                exit_s[s] = max(max(L[seg_last[s]], B[seg_last[s]]), Fm[s])
            into[0] = NEG
            for s2 in range(1, S):
                v, cost = NEG, 0.0
                for s in range(s2 - 1, -1, -1):
                    v = max(v, exit_s[s] + cost)
                    if not optional[s]:
                        break
                    cost += c_skip
                into[s2] = v
            m = NEG
            for j in range(end_from, n):
                m = max(m, max(L[j], B[j]))
            for j in range(n):
                if j > 0 and seg_first[seg_of[j]] == j:
                    enter = into[seg_of[j]]
                elif j > 0:
                    enter = B[j - 1]
                    if r[j] != r[j - 1]:
                        enter = max(enter, L[j - 1])
                else:
                    enter = NEG
                if j < n_first:
                    enter = max(enter, Fpre)
                nL[j] = max(L[j], enter) + e[r[j]]
                nB[j] = max(L[j], B[j]) + e[0]
            for s in range(S):
                Fm[s] = max(Fm[s], max(L[seg_last[s]], B[seg_last[s]]) + c_fill) + fe
            Fpost = max(Fpost, m) + fe
            Fpre = Fpre + fe
            for j in range(n):
                L[j], B[j] = nL[j], nB[j]
        m = Fpost
        for j in range(end_from, n):
            m = max(m, max(L[j], B[j]))
        return m
else:  # pragma: no cover
    _span_score_jit = None


def span_score(lp: np.ndarray, segs: list[np.ndarray], optional: list[bool], n_first: int, end_from: int,
               cfg: PracticeConfig, use_jit: bool = True) -> float:
    """lp explained by these lines in order (see _span_score_numpy)."""
    if not segs or not lp.shape[0]:
        return NEG
    r = np.concatenate(segs).astype(np.int64)
    sizes = np.array([x.size for x in segs])
    seg_last = (np.cumsum(sizes) - 1).astype(np.int64)
    seg_first = (seg_last - sizes + 1).astype(np.int64)
    seg_of = np.repeat(np.arange(len(segs)), sizes).astype(np.int64)
    opt = np.array(optional, dtype=np.bool_)
    lp = np.ascontiguousarray(lp, dtype=np.float64)
    args = (lp, r, seg_of, seg_first, seg_last, opt, int(n_first), int(end_from), float(cfg.fill_cost),
            float(cfg.c_fill), float(cfg.c_skip))
    if _span_score_jit is not None and use_jit:
        return float(_span_score_jit(*args))
    return float(_span_score_numpy(*args))


def line_score(lp: np.ndarray, prev: np.ndarray | None, line: np.ndarray, first_word: int, later: list[np.ndarray],
               cfg: PracticeConfig, use_jit: bool = True) -> float:
    """Was line k said, between the last line said before it (prev) and where the reader is now
    (the lines after it, `later`, each of which may have been left out)? The best path that says
    at least k's first word, less the best path without k. Both end anywhere past what they
    must say: a line half said so far counts on its half, and lines the reader hasn't reached
    cost nothing. Without prev (k opens the reading), both start at their first line's start."""
    ctx = [prev] if prev is not None else []
    segs = ctx + [line] + later
    opt = [False] * len(ctx) + [False] + [True] * len(later)
    n_first = prev.size if prev is not None else 1
    with_k = span_score(lp, segs, opt, n_first, sum(x.size for x in ctx) + max(1, first_word) - 1, cfg, use_jit)
    if ctx:
        without = span_score(lp, ctx + later, [False] + [True] * len(later), n_first, 0, cfg, use_jit)
    elif later:  # k opened the reading; without it the reading opens on any later line (at its start)
        without = max(span_score(lp, later[i:], [True] * (len(later) - i), 1, 0, cfg, use_jit)
                      for i in range(len(later)))
    else:  # nothing else to say: the line against the filler alone
        without = float(np.maximum(lp[:, 0], lp[:, 1:].max(axis=1) - cfg.fill_cost).sum())
    return with_k - without


def line_llr(lp: np.ndarray, prev: np.ndarray | None, line: np.ndarray, nxt: np.ndarray | None,
             cfg: PracticeConfig, use_jit: bool = True) -> float:
    """How much better (nats) line k's letters explain the frames between its neighbours than
    leaving it out. With k, the frames may end anywhere from line k on: the display can reach
    line k+2 while the reader is still in line k (it moved on early), and a line half said so far
    is judged on its half."""
    ctx = [x for x in (prev,) if x is not None]
    after = [x for x in (nxt,) if x is not None]
    with_k = hyp_score(lp, ctx + [line] + after, cfg.fill_cost, cfg.c_fill, use_jit, end_from=len(ctx))
    without = hyp_score(lp, ctx + after, cfg.fill_cost, cfg.c_fill, use_jit) if ctx or after else 0.0
    if not ctx and not after:  # a one-line target: the line against the filler alone
        T = lp.shape[0]
        fe = np.maximum(lp[:, 0], lp[:, 1:].max(axis=1) - cfg.fill_cost)
        without = float(fe.sum()) if T else 0.0
    return with_k - without


def committed_frames(lp: np.ndarray, n_frames: np.ndarray, ts: np.ndarray, lookahead: float = 0.2,
                     hop: float = 0.1, frame_s: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """The frame stream the follower commits (stream_follower.StreamFollower._step): from each
    window, the frames since the last step that end `lookahead` before its end. Returns the frames
    (N x C) and each one's end time."""
    la = int(round(lookahead / frame_s))
    h = int(round(hop / frame_s))
    out, times, t_stream = [], [], None
    for k, t in enumerate(ts):
        n = int(n_frames[k])
        commit_end = n - la
        if t_stream is None:
            kk = h
        else:
            kk = int(round((float(t) - lookahead - t_stream) / frame_s))
        t_stream = float(t) - lookahead
        if kk <= 0 or commit_end <= 0:
            continue
        new = lp[k, max(0, commit_end - kk) : commit_end]
        out.append(new)
        times.append(t_stream - frame_s * np.arange(len(new) - 1, -1, -1))
    if not out:
        return np.zeros((0, lp.shape[-1]), np.float32), np.zeros(0)
    f = np.concatenate(out).astype(np.float32)
    # a broken window (fp16 overflow on silence) or a column at exactly zero probability: a floor
    return np.where(np.isfinite(f), np.maximum(f, -50.0), -50.0), np.concatenate(times)


class PracticeChecker:
    """The checker online, as the phone runs it (web/practice.js mirrors it): the frames the
    follower commits arrive with their end times (add_frames), the display's line after each
    step (step), and each call returns the lines decided then, as (line, time, score, shown).

    Line k is decided once the display has stayed on line k+2 or later for settle_s, and lines
    never passed so, at finish(). It is scored (line_score) between the last earlier line not
    judged left out and the lines after it up to one past the display's, on the frames from
    margin_s before the display was last on a line before that context line (but at least
    span_min_s back) to those committed by the decision. first_words: letters in each line's
    first word (the least of k that counts as saying it)."""

    def __init__(self, line_letters: list[np.ndarray], lo: int, hi: int, cfg: PracticeConfig | None = None,
                 first_words: list[int] | None = None, lookahead: float = 0.2, use_jit: bool = True):
        self.letters, self.lo, self.hi = line_letters, lo, hi
        self.cfg = cfg or PracticeConfig()
        self.first_words, self.lookahead, self.use_jit = first_words, lookahead, use_jit
        self._f: list[np.ndarray] = []
        self._ft: list[np.ndarray] = []
        self.verdicts: dict[int, tuple[float, bool]] = {}
        self.shown: set[int] = set()
        self.t_last_on: dict[int, float] = {}  # line -> last time the display was on it (so far)
        self.since: dict[int, float] = {}  # m -> when the display last came onto a line >= m and stayed
        self.furthest = -1
        self.t_first = None  # the first step's time
        self.history: list[str] = []  # the verdicts given, left-out ones aside (the sound gate)
        self.final: dict[int, str] = {}  # line -> the verdict the reader is shown

    def add_frames(self, frames: np.ndarray, times: np.ndarray) -> None:
        if len(frames):
            f = np.asarray(frames, dtype=np.float32)
            self._f.append(np.where(np.isfinite(f), np.maximum(f, -50.0), -50.0))
            self._ft.append(np.asarray(times, dtype=np.float64))
        keep = (float(self._ft[-1][-1]) if self._ft else 0.0) - self.cfg.max_span_s - self.cfg.margin_s - 5.0
        while len(self._ft) > 1 and self._ft[0][-1] < keep:  # older than any span can reach
            self._f.pop(0)
            self._ft.pop(0)

    def _span(self, before_line: int, t: float) -> np.ndarray:
        cfg = self.cfg
        if cfg.span_s is not None:
            t0 = t - self.lookahead - cfg.span_s
        else:
            before = [tt for li, tt in self.t_last_on.items() if li < before_line]
            t0 = (max(before) if before else (self.t_first or 0.0)) - cfg.margin_s
            t0 = max(min(t0, t - self.lookahead - cfg.span_min_s), t - self.lookahead - cfg.max_span_s)
        if not self._f:
            return np.zeros((0, 1), np.float32)
        if len(self._f) > 1:
            self._f, self._ft = [np.concatenate(self._f)], [np.concatenate(self._ft)]
        frames, ftimes = self._f[0], self._ft[0]
        a = int(np.searchsorted(ftimes, t0, side="left"))
        b = int(np.searchsorted(ftimes, t - self.lookahead, side="right"))
        return frames[a:b]

    def _decide(self, k: int, t: float, at: int) -> tuple[int, float, float, bool]:
        cfg, lo = self.cfg, self.lo
        p = k - 1
        while p >= lo and p in self.verdicts and verdict(*self.verdicts[p], cfg) == "left out":
            p -= 1
        prev = self.letters[p] if p >= lo else None
        later = [self.letters[j] for j in range(k + 1, min(self.hi, at + 1, k + cfg.max_later) + 1)]
        fw = self.first_words[k] if self.first_words is not None else self.letters[k].size
        fw = max(fw, int(np.ceil(cfg.min_frac * self.letters[k].size)))
        sc = line_score(self._span(p if p >= lo else k - 1, t), prev, self.letters[k], fw, later, cfg, self.use_jit)
        self.verdicts[k] = (sc, k in self.shown)
        self.final[k] = gated(verdict(sc, k in self.shown, cfg), self.history, cfg)
        return (k, t, sc, k in self.shown)

    def step(self, t: float, li: int | None) -> list[tuple[int, float, float, bool]]:
        if self.t_first is None:
            self.t_first = t
        self._t_last = t
        if li is None:
            return []
        self.shown.add(li)
        self.t_last_on[li] = t
        self.furthest = max(self.furthest, li)
        for m in [m for m in self.since if m > li]:  # back below m: its clock starts again
            del self.since[m]
        for m in range(self.lo + 2, li + 1):
            self.since.setdefault(m, t)
        out = []
        for k in range(self.lo, min(self.hi, li - 2) + 1):
            if k not in self.verdicts and t - self.since[k + 2] >= self.cfg.settle_s - 1e-9:
                out.append(self._decide(k, t, li))
        return out

    def finish(self, t: float | None = None, up_to: int | None = None) -> list[tuple[int, float, float, bool]]:
        """The session is over: every line not decided yet, now (through line up_to only: a reader
        who stops partway meant to, and the lines they never reached aren't left out)."""
        t = getattr(self, "_t_last", 0.0) if t is None else t
        hi = self.hi if up_to is None else min(self.hi, up_to)
        return [self._decide(k, t, max(self.furthest, k)) for k in range(self.lo, hi + 1) if k not in self.verdicts]


def judge_lines(frames: np.ndarray, ftimes: np.ndarray, steps: list[tuple[float, int | None]],
                line_letters: list[np.ndarray], lo: int, hi: int, cfg: PracticeConfig,
                lookahead: float = 0.2, use_jit: bool = True,
                first_words: list[int] | None = None) -> list[tuple[int, float, float, bool]]:
    """PracticeChecker over a whole session at once: frames committed so far (ftimes <= the step's
    time less lookahead) before each step of the display (time, line index or None)."""
    ch = PracticeChecker(line_letters, lo, hi, cfg, first_words, lookahead, use_jit)
    out, a = [], 0
    for t, li in steps:
        b = int(np.searchsorted(ftimes, t - lookahead, side="right"))
        if b > a:
            ch.add_frames(frames[a:b], ftimes[a:b])
            a = b
        out += ch.step(t, li)
    if steps:
        out += ch.finish(steps[-1][0])
    return out
