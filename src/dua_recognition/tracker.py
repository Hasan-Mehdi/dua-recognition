"""Follow a recitation through the corpus: which du'a, which line, which word.

This is score following: an HMM whose hidden state is the word the reciter is
on, across every du'a in the corpus at once. Each update has two halves.

  predict  — the reciter moved forward a little since the last update. Mass
             shifts ahead by a speed prior, with a sliver for going back
             (reciters repeat lines) and a uniform "teleport" floor so the
             tracker can recover, or pick up someone who starts mid-du'a.
  correct  — weight each word by how well the latest transcript ends there
             (align.CorpusIndex.word_costs). A refrain matches equally well at
             every repetition; the predicted position is what separates them.

Identification falls out for free: the posterior mass inside a du'a is the
probability that it is the one being recited.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .align import CorpusIndex


@dataclass
class TrackerConfig:
    # Defaults are the grid-search optimum on the train reciters
    # (scripts/evaluate.py --split train --tune; see docs/RESULTS.md).
    #
    # Log-likelihood per edit. Windows overlap (6 s window, 1 s hop), so each
    # letter is heard several times; kappa is per-hearing and so stays small.
    kappa: float = 0.15
    # Until one du'a holds `lock_confidence` of the mass, weigh evidence more
    # sharply: a listener who joins mid-recitation should be found in seconds.
    # Once locked, the gentler kappa keeps the follower from jumping at every
    # ASR slip. (A single kappa has to pick one; see docs/RESULTS.md.)
    kappa_search: float = 1.2
    lock_confidence: float = 0.95
    # Recitation speed in words/second: most lines run 0.4-1.5 w/s.
    max_speed: float = 4.0
    # How far the reciter moves in dt: a Poisson mixture over these speeds,
    # truncated at max_speed. They are the time-weighted deciles of words per
    # second per line in the train reciters' human timings (0.33-1.33), scaled
    # x1.8 by grid search on the train split: the transcript lags the voice,
    # so the prior has to run a little ahead of the true speed. Empty = the
    # old flat 0..max_speed*dt, which runs far ahead of slow, drawn-out lines
    # (one line ahead was 12% of all steps).
    speeds: tuple[float, ...] = (0.59, 0.99, 1.15, 1.3, 1.44, 1.55, 1.67, 1.84, 2.05, 2.39)
    # Reciters differ several-fold in tempo (0.14-1.2 w/s on average), so
    # rather than averaging the speeds, weigh them by how well each has been
    # predicting this reciter: after every update each speed's weight is
    # multiplied by the evidence its own prediction got (an interacting
    # multiple-model filter). `tempo_memory` < 1 forgets old evidence so a
    # reciter who speeds up is followed. 0 = plain average (no adaptation).
    tempo_memory: float = 0.98
    p_back: float = 0.1  # chance per update of jumping back a few words
    back_words: int = 8
    p_teleport: float = 1e-2  # floor over the whole corpus (see _floor)
    # Once locked on a du'a, jumping elsewhere should take sustained evidence,
    # not one noisy window: a smaller teleport floor while locked.
    p_teleport_locked: float = 1e-2
    # Prior: this much mass on the first `start_words` words of each du'a.
    start_weight: float = 0.3
    start_words: int = 12
    # Reported only once the winning du'a holds this much posterior mass.
    min_dua_confidence: float = 0.7


@dataclass
class Position:
    dua: str | None
    dua_confidence: float
    segment: int | None
    segment_confidence: float
    word: int | None  # index into CorpusIndex.words
    # The reciter has reached the last word of the line (a pause here usually
    # means the next line is coming; the UI previews it without jumping).
    at_line_end: bool = False
    candidates: list[tuple[str, float]] = field(default_factory=list)  # top du'as


class Tracker:
    def __init__(self, index: CorpusIndex, config: TrackerConfig | None = None):
        self.ix = index
        self.cfg = config or TrackerConfig()

        # Where might a listener be when we first hear them? Mostly at the
        # start of some du'a; sometimes they join mid-way. So the prior mixes
        # "the opening words of each du'a" with "anywhere", both uniform over
        # du'as — uniform over words would make a long du'a like Kumayl ten
        # times likelier a priori than a short one like Faraj. The same mix is
        # the teleport target, for when someone switches du'a or skips ahead.
        ix = self.ix
        n_duas = len(ix.dua_word_span)
        anywhere = np.zeros(ix.n_words)
        start = np.zeros(ix.n_words)
        for lo, hi in ix.dua_word_span:
            anywhere[lo:hi] = 1.0 / (n_duas * (hi - lo))
            k = min(self.cfg.start_words, hi - lo)
            start[lo : lo + k] = 1.0 / (n_duas * k)
        w = self.cfg.start_weight
        self._floor = w * start + (1 - w) * anywhere
        self._idx = np.arange(ix.n_words)
        spans = np.array(ix.dua_word_span)[ix.word_dua]
        self._first, self._last = spans[:, 0], spans[:, 1] - 1
        self.reset()

    def reset(self) -> None:
        self.post = self._floor.copy()
        self.tempo = np.full(max(1, len(self.cfg.speeds)), 1.0 / max(1, len(self.cfg.speeds)))
        self._fwd_by_speed: np.ndarray | None = None
        self._last_word: int | None = None

    # -- predict ---------------------------------------------------------
    def _locked(self) -> bool:
        dua_mass = np.bincount(self.ix.word_dua, weights=self.post, minlength=len(self.ix.duas))
        return bool(dua_mass.max() >= self.cfg.lock_confidence)

    def _advance(self, dt: float, locked: bool = False) -> None:
        cfg = self.cfg
        tele = cfg.p_teleport_locked if locked else cfg.p_teleport
        if dt <= 0:
            self.post = (1 - tele) * self.post + tele * self._floor
            return
        n_fwd = int(np.ceil(cfg.max_speed * dt))
        # Flat over 0..n_fwd words ahead: we know the reciter moves forward,
        # not how fast. The observation sorts out the rest. Moves are clamped
        # to the du'a: finishing one doesn't carry you into the next one in
        # the index (that's what the teleport floor is for).
        n = self.post.size
        kernels = self._speed_kernels(dt, n_fwd)  # speeds x (n_fwd + 1)
        shifted = np.stack([np.bincount(np.minimum(self._idx + d, self._last), weights=self.post, minlength=n)
                            for d in range(n_fwd + 1)])
        self._fwd_by_speed = kernels @ shifted if self.cfg.tempo_memory > 0 else None
        fwd = (self.tempo @ kernels) @ shifted
        back = np.zeros_like(self.post)
        for d in range(1, cfg.back_words + 1):
            back += np.bincount(np.maximum(self._idx - d, self._first), weights=self.post, minlength=n)
        back /= cfg.back_words
        p = (1 - cfg.p_back) * fwd + cfg.p_back * back
        self.post = (1 - tele) * p / p.sum() + tele * self._floor

    def _speed_kernels(self, dt: float, n_fwd: int) -> np.ndarray:
        """P(moved d words in dt | speed), one row per speed, d = 0..n_fwd."""
        if not self.cfg.speeds:
            return np.full((1, n_fwd + 1), 1.0 / (n_fwd + 1))
        d = np.arange(n_fwd + 1)
        lam = np.asarray(self.cfg.speeds)[:, None] * dt
        log_fact = np.cumsum(np.r_[0.0, np.log(np.maximum(d[1:], 1))])
        k = np.exp(d * np.log(lam) - lam - log_fact)
        return k / k.sum(axis=1, keepdims=True)

    # -- correct ---------------------------------------------------------
    def update(self, transcript: str, dt: float) -> Position:
        """Advance by dt seconds, then condition on the latest window's text."""
        costs = self.ix.word_costs(transcript) if transcript else None
        return self.update_costs(costs, dt)

    def update_costs(self, costs: np.ndarray | None, dt: float) -> Position:
        """`update` with the alignment already done (evaluation reuses it)."""
        # An empty window is almost always the pause between lines: the
        # reciter isn't moving, so neither does the belief (bar the floors).
        self._advance(dt if costs is not None else 0.0, locked=self._locked())
        if costs is not None:
            kappa = self.cfg.kappa if self._locked() else self.cfg.kappa_search
            lik = np.exp(-kappa * (costs - costs.min()).astype(np.float64))
            if self._fwd_by_speed is not None:
                # Which speed predicted this evidence best? Forget a little first.
                ev = self._fwd_by_speed @ lik
                w = self.tempo ** self.cfg.tempo_memory * ev
                if w.sum() > 0:
                    self.tempo = w / w.sum()
            self.post = self.post * lik
            self.post /= self.post.sum()
        return self._forward_only(self.position())

    def _forward_only(self, pos: Position) -> Position:
        """Within a line, the word highlight only moves forward.

        The line itself may still move back (reciters repeat lines); a word
        pointer sliding back and forth inside one line is just noise.
        """
        last = self._last_word
        if (last is not None and pos.word is not None and pos.segment is not None
                and self.ix.word_segment[last] == pos.segment and self.ix.word_dua[last] == self.ix.word_dua[pos.word]
                and pos.word < last):
            pos.word = last
            pos.at_line_end = last + 1 >= len(self.ix.words) or self.ix.word_segment[last + 1] != pos.segment
        self._last_word = pos.word
        return pos

    def position(self) -> Position:
        ix = self.ix
        dua_mass = np.bincount(ix.word_dua, weights=self.post, minlength=len(ix.duas))
        d = int(dua_mass.argmax())
        conf = float(dua_mass[d])
        top = [(ix.dua_ids[i], float(dua_mass[i])) for i in np.argsort(-dua_mass)[:3]]
        if conf < self.cfg.min_dua_confidence:
            return Position(None, conf, None, 0.0, None, candidates=top)
        lo, hi = ix.dua_word_span[d]
        seg_ids = ix.word_segment[lo:hi]
        seg_mass = np.bincount(seg_ids, weights=self.post[lo:hi])
        s = int(seg_mass.argmax())
        # The word shown must lie in the line shown: the weighted median of the
        # posterior within that line (a lone argmax flickers across a flat line,
        # and taken over the whole du'a it can land in a different line).
        in_seg = lo + np.flatnonzero(seg_ids == s)
        cum = np.cumsum(self.post[in_seg])
        word = int(in_seg[min(int(np.searchsorted(cum, cum[-1] / 2)), len(in_seg) - 1)])
        at_end = word + 1 >= hi or ix.word_segment[word + 1] != ix.word_segment[word]
        return Position(ix.dua_ids[d], conf, s, float(seg_mass[s] / conf), word, at_end, top)

    def prompt(self, n_words: int = 12) -> str | None:
        """Reference text just before the current position, for ASR biasing."""
        pos = self.position()
        if pos.word is None or pos.dua_confidence < 0.9:
            return None
        return self.ix.text_before(pos.word, n_words)
