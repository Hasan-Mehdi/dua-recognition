"""Follow the reciter word by word from a CTC model's frames, between tracker updates.

The HMM tracker (tracker.py) hears a 6 s transcript once a second, with no
timing inside it: it knows which du'a and which line well, but by the time a
transcript arrives the word it ends on is a second or more old, and during a
pause it can't tell "stopped at the end of the line" from "about to go on".
The display lead and the pause rules in TrackerConfig work around that.

A CTC model's frames are timed evidence. Every 0.2 s this scores the last
`window_s` seconds of posteriors against the words around the current one
(ctc_align.end_scores: the best path through the reference ending on each
letter) and moves to the word whose letter explains the audio best, with a
penalty for going back and a cap on how far it may jump. When the reciter
stops, the frames after their last letter are blanks, and a blank after a
letter stays on that letter: a pause holds the last word by construction.

The tracker keeps the jobs it is good at: which du'a, which line, and
re-anchoring the follower if the two disagree by more than a line for a while.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import CorpusIndex
from .ctc_align import end_scores, has_speech, squeeze_blanks

FRAME_S = 0.02  # wav2vec2 frame hop


@dataclass
class FollowerConfig:
    # Picked on the train split (scripts/follow_eval.py --split train --grid, flowing and pause
    # sets): best word exact with jerks no worse than page mode (docs/results/follower.md).
    window_s: float = 2.0  # seconds of the latest frames scored each step
    beta_back: float = 3.0  # log-likelihood cost per word moved back (per unit temp)
    max_jump: int = 2  # words it may move forward in one step
    temp: float = 1.0  # divides the CTC log-likelihoods (they are over-confident)
    reset_after: float = 1.0  # seconds of >1 line disagreement before re-anchoring on the tracker
    back_words: int = 6  # reference starts this many words before the current one
    ahead_words: int = 20  # ...and ends this many after
    # Hold if these last seconds hold no letter frame at all (0 = never:
    # trailing blanks keep the last letter anyway). CTC is peaky: requiring
    # 3 letter frames held mid-word through long vowels (oracle-anchored,
    # 6 train recordings: exact 43% with 3 frames, 72% with 1).
    tail_s: float = 0.5
    # Line onset (round 2, docs/results/follower.md): on the last word of a line, letters in
    # the latest `onset_window` s after at least `onset_gap` s of blanks move straight to the
    # next line's first word, without scoring (0 = off). With `onset_confirm`, only if the
    # tracker's lead position (page mode's line) is already past the current line.
    onset_gap: float = 0.0
    onset_window: float = 0.3
    onset_confirm: bool = False
    # A move back, or more than one word forward, must win this many steps in a row.
    confirm_steps: int = 1


class LocalFollower:
    def __init__(self, index: CorpusIndex, config: FollowerConfig | None = None):
        self.ix = index
        self.cfg = config or FollowerConfig()
        ix = index
        # First letter of each word (ix.letters is the whole corpus, word by word).
        self._word_letter = np.r_[0, np.flatnonzero(np.diff(ix.letter_word) != 0) + 1, ix.letters.size]
        new_line = np.r_[True, (np.diff(ix.word_segment) != 0) | (np.diff(ix.word_dua) != 0)]
        self._line_no = np.cumsum(new_line) - 1  # running line number over the corpus
        self.reset()

    def reset(self) -> None:
        self.word: int | None = None
        self._disagree_since: float | None = None
        self._pending: int | None = None  # a big move waiting for confirm_steps
        self._pending_n = 0

    def _onset(self, lp: np.ndarray, lead_word: int | None) -> bool:
        """On the last word of a line: has the next line started (letters after a blank run)?"""
        ix, cfg, w = self.ix, self.cfg, self.word
        if cfg.onset_gap <= 0 or w + 1 >= ix.dua_word_span[ix.word_dua[w]][1] or \
                self._line_no[w + 1] == self._line_no[w]:
            return False
        if cfg.onset_confirm and (lead_word is None or ix.word_dua[lead_word] != ix.word_dua[w]
                                  or self._line_no[lead_word] <= self._line_no[w]):
            return False
        letter = lp[:, 1:].max(axis=1) > lp[:, 0]
        tail = max(1, int(round(cfg.onset_window / FRAME_S)))
        hits = np.flatnonzero(letter[-tail:])
        if not hits.size:
            return False
        k = lp.shape[0] - tail + int(hits[0])  # first letter frame in the tail
        before = np.flatnonzero(letter[:k])
        run = k - (int(before[-1]) + 1 if before.size else 0)  # blank frames just before it
        return run * FRAME_S >= cfg.onset_gap - 1e-9

    def _reference(self, anchor: int) -> tuple[np.ndarray, np.ndarray]:
        """Letters of words [anchor - back, anchor + ahead) in the anchor's du'a, and each letter's word."""
        lo, hi = self.ix.dua_word_span[self.ix.word_dua[anchor]]
        a, b = max(lo, anchor - self.cfg.back_words), min(hi, anchor + self.cfg.ahead_words)
        la, lb = self._word_letter[a], self._word_letter[b]
        return self.ix.letters[la:lb], self.ix.letter_word[la:lb]

    def step(self, lp: np.ndarray, t: float, hmm_word: int | None, lead_word: int | None = None) -> int | None:
        """One CTC window (frames x columns, trimmed to its real length) ending at
        time t; `hmm_word` is the tracker's current word (None: not locked), and
        `lead_word` its lead (display) word, used only by onset_confirm.
        Returns the word to show, or None to leave the display to the tracker."""
        ix, cfg = self.ix, self.cfg
        if hmm_word is None:
            self.reset()
            return None
        if self.word is None or ix.word_dua[self.word] != ix.word_dua[hmm_word]:
            self.word, self._disagree_since = hmm_word, None
        elif abs(int(self._line_no[self.word]) - int(self._line_no[hmm_word])) > 1:
            if self._disagree_since is None:
                self._disagree_since = t
            elif t - self._disagree_since >= cfg.reset_after:
                self.word, self._disagree_since = hmm_word, None
        else:
            self._disagree_since = None
        lp = lp[-max(1, int(round(cfg.window_s / FRAME_S))) :]
        if not lp.size or (cfg.tail_s > 0 and not has_speech(lp[-max(1, int(round(cfg.tail_s / FRAME_S))) :], 1)):
            return self.word  # silence: stay put
        if self._onset(lp, lead_word):
            self.word, self._pending, self._pending_n = self.word + 1, None, 0
            return self.word
        letters, owner = self._reference(self.word)
        if not letters.size:
            return self.word
        sc = end_scores(squeeze_blanks(lp), letters.astype(np.int64)) / cfg.temp
        # Best end letter per word, then the move penalties.
        w0 = int(owner[0])
        best = np.full(int(owner[-1]) - w0 + 1, -np.inf)
        np.maximum.at(best, owner - w0, sc)
        words = np.arange(w0, w0 + best.size)
        cur = self.word
        best -= cfg.beta_back * np.maximum(0, cur - words)
        best[words > cur + cfg.max_jump] = -np.inf
        new = int(words[int(np.argmax(best))])
        if cfg.confirm_steps > 1 and (new < cur or new > cur + 1):
            self._pending_n = self._pending_n + 1 if new == self._pending else 1
            self._pending = new
            if self._pending_n < cfg.confirm_steps:
                return self.word
        self.word, self._pending, self._pending_n = new, None, 0
        return self.word
