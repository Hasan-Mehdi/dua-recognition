"""Offline alignment of a whole recording to its du'a: forward-backward.

The live tracker only knows the past. Offline, every window can also see the
future, so the same HMM can be *smoothed*: gamma_t ∝ alpha_t · beta_t, where
beta runs the transition model backwards. That removes the reaction lag and
most of the ambiguity, which makes it a labeller for recordings that come
without line timings (scripts/align_offline.py) — once its agreement with the
human timings has been measured on recordings that have them.
"""
from __future__ import annotations

import numpy as np

from .align import CorpusIndex
from .tracker import Tracker, TrackerConfig


class Smoother(Tracker):
    def _transpose(self, v: np.ndarray, dt: float) -> np.ndarray:
        """Apply the transpose of `_advance` to a backward message."""
        cfg = self.cfg
        tele = cfg.p_teleport * float(self._floor @ v)
        if dt <= 0:
            return (1 - cfg.p_teleport) * v + tele
        n_fwd = int(np.ceil(cfg.max_speed * dt))
        fwd = sum(v[np.minimum(self._idx + d, self._last)] for d in range(n_fwd + 1)) / (n_fwd + 1)
        back = sum(v[np.maximum(self._idx - d, self._first)] for d in range(1, cfg.back_words + 1))
        back = back / cfg.back_words
        return (1 - cfg.p_teleport) * ((1 - cfg.p_back) * fwd + cfg.p_back * back) + tele

    def _lik(self, costs: np.ndarray | None) -> np.ndarray:
        if costs is None:
            return np.ones(self.ix.n_words)
        return np.exp(-self.cfg.kappa * (costs - costs.min()))

    def smooth(self, costs: list[np.ndarray | None], dt: float) -> np.ndarray:
        """Posterior word index (argmax) at every step."""
        self.reset()
        alphas = []
        for c in costs:
            self._advance(dt if c is not None else 0.0)
            self.post = self.post * self._lik(c)
            self.post /= self.post.sum()
            alphas.append(self.post.copy())
        beta = np.ones(self.ix.n_words)
        out = np.empty(len(costs), dtype=np.int64)
        for t in range(len(costs) - 1, -1, -1):
            out[t] = int(np.argmax(alphas[t] * beta))
            if t:
                nxt = costs[t]
                beta = self._transpose(self._lik(nxt) * beta, dt if nxt is not None else 0.0)
                beta /= beta.max()
        return out


def align_recording(index: CorpusIndex, costs: list[np.ndarray | None], dt: float,
                    config: TrackerConfig | None = None) -> np.ndarray:
    """Segment id at every step, for a recording of a single known du'a."""
    cfg = config or TrackerConfig(p_teleport=1e-4, start_weight=0.9)
    words = Smoother(index, cfg).smooth(costs, dt)
    return index.word_segment[words]
