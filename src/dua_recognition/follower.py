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
from .ctc_align import end_scores, has_speech, jump_scores, squeeze_blanks

FRAME_S = 0.02  # wav2vec2 frame hop


@dataclass
class FollowerConfig:
    # Scoring picked on the train split in round 1 (docs/results/follower.md); the catch-up leap,
    # the step-back rules, word starts and the re-anchor rules since 2026-10-01, with the phone's
    # CTC model (docs/results/phone_follower.md); the quiet catch-up, line restarts and the
    # stuck-only re-anchor that evening, for a step every 0.1 s (docs/results/phone_latency.md).
    # web/follower.js has the same defaults.
    window_s: float = 2.0  # seconds of the latest frames scored each step
    beta_back: float = 2.0  # log-likelihood cost per word moved back (per unit temp; 3 until 2026-10-01)
    max_jump: int = 2  # words it may move forward in one step
    temp: float = 1.0  # divides the CTC log-likelihoods (they are over-confident)
    reset_after: float = 3.0  # seconds of >1 line disagreement before re-anchoring on the tracker
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
    # While the tracker has lost the du'a (no anchor), keep following on the follower's own
    # word for up to this many seconds (0 = reset at once, as round 1). The phone's Whisper
    # loses confidence on ordinary voices for seconds at a time; the frames often still fit.
    lapse_hold: float = 0.0
    # Steps back (a 1-word step back is usually the model rescoring a held vowel, not the
    # reciter going back): moves back by fewer than `back_min` words are ignored, and a move
    # back must win `back_confirm` steps in a row (1 = at once, as round 1). Repeats of a
    # line or more still go back, a step or two later.
    back_min: int = 2
    back_confirm: int = 4  # steps: 0.4 s at the phone's 0.1 s step (2 at 0.2 s until 2026-10-01)
    # Catch up (0 = off): when a word beyond max_jump explains the audio better than every
    # word within reach by at least this many nats, on `leap_confirm` steps in a row (each
    # time at or past the last such word), go there. A follower that fell behind (a phrase
    # its model didn't hear) otherwise hears only words it isn't allowed to reach, and
    # stalls until the tracker re-anchors it.
    leap_margin: float = 4.0
    leap_confirm: int = 2
    # Word starts: the scored path may begin anywhere in the current word or before it (the
    # window starts mid-word), but in a later word only at its first letter. Off, a window
    # whose last letters fit no word well (a swallowed ending) can sit as well on the middle
    # of the next line's first word as on the word said, and the display enters the line early.
    word_starts: bool = True
    # Forward re-anchor (0 = off): the tracker's word this many words or more ahead, in the same
    # du'a, for `ahead_after` seconds: go there. A follower that fell behind and then heard the
    # reciter stop would otherwise hold there to the end (797f through the page: line 8 while he
    # had finished line 9). Only forward, so a wrong tracker can't pull the display back; the
    # line rule above stays for big disagreements either way.
    ahead_words_reanchor: int = 4
    ahead_after: float = 2.0
    # ...and only while the follower is stuck: its word unchanged for those seconds. A reader
    # saying a line again moves the follower back and along it while the tracker, slower to
    # believe a step back, is still ahead; the re-anchor then pulled the display off the repeat.
    ahead_stuck: bool = True
    # Quiet catch-up (0 = off): the reciter is silent, and the tracker's word is ahead of the
    # follower's in the same line, by at most `quiet_words`, for `quiet_after` seconds: go
    # there. A word the CTC model didn't hear (x0fo: the du'a's last word, ṭawīlā, which Whisper
    # heard) otherwise stays unlit until the reciter says it again.
    quiet_after: float = 0.5
    quiet_words: int = 3
    # Line restarts (0 = off): going back to the first word of the current line, or of one of the
    # `restart_lines` lines before it, costs at most `restart_cost` nats instead of beta_back per
    # word. Readers start a line over, or say the line before again; they rarely jump back into
    # the middle of one. The scored text reaches back to those line starts.
    restart_cost: float = 3.0
    restart_lines: int = 1
    # A step forward into a later line must win this many steps in a row (1 = at once). At a step
    # every 0.1 s one window can hear a line's last word as the next line's first (0t98: ṭawʿā
    # heard "tawā", matched to wa-tumattiʿahu) and show the next line in the pause after it.
    line_confirm: int = 2
    # The line re-anchor (reset_after) only while the follower is stuck, as ahead_stuck: a follower
    # moving word by word is hearing the reciter, and a tracker two lines behind pulled it back
    # (9puq, 20 s).
    reset_stuck: bool = True
    # After a step back of a line or more (a repeat), the tracker's re-anchors may not take the
    # follower forward for this many seconds, or until it is back where it was (0 = off). The
    # tracker rarely believes a repeat: it stays ahead, or moves on into the next lines, and in
    # diqq (Kumayl, line 109 said again) the forward re-anchor pulled a follower that was on the
    # repeat two lines ahead.
    repeat_hold: float = 20.0
    # No step into a later line while the page's stop detector (asr.LiveQuiet: loudness within
    # 15 dB of the reciter's voice) has heard no voice for this many seconds (0 = off; needs
    # `quiet_now` in step). Letters the model makes up out of room tone in a pause can match the
    # next line's first word; a real start of the next line comes with a voice.
    line_quiet: float = 0.3
    # Jumps (0 = off): the latest `jump_window` s of frames are also scored against the whole
    # du'a. When a place outside the follower's own reach explains them better than every word
    # within reach by `jump_margin` nats, on `jump_confirm` steps in a row (at most two words
    # apart), the follower goes there; for `jump_hold` s the tracker can't re-anchor it, since
    # Whisper's result about the jump comes seconds later. Where the text is found in several
    # places (refrains), the one nearest the tracker's word. Readers skip around a du'a; the
    # follower's reach is a few words back and twenty ahead.
    jump_margin: float = 7.0
    jump_window: float = 1.2
    jump_confirm: int = 4
    jump_hold: float = 6.0
    # ...and unless the tracker's word is within a line of the target, it takes `jump_alone` nats
    # (0 = jump_margin either way). In ordinary reading a far place can fit 1.2 s of audio by
    # chance; the tracker hardly ever agrees with such a place, and soon agrees with a real jump.
    jump_alone: float = 0.0
    # Agreement by belief (0 = by the tracker's word, as above): the tracker's posterior mass on
    # the target's line or a neighbour (step's `line_mass`) at least this. A real jump's line gains
    # mass within a Whisper update or two; a chance fit's line has the prior's crumbs.
    jump_mass: float = 0.0
    # Two-part scoring (False = the latest jump_window s against the du'a on their own): the whole
    # window is explained by the text within reach up to some frame, then by the du'a from the
    # first letter of a line on (ctc_align.jump_scores), against the text within reach alone.
    # The audio before a jump counts for the text it belongs to, so the window can be the whole
    # 2 s, and a formula repeated within reach ties with its far copy instead of losing to it.
    jump_seg: bool = False
    # Extra nats needed (0 = none) when the target's phrase (it and up to two words before it in
    # its line) occurs more than once in the du'a: formulaic lines (Hadith al-Kisa) drew most of the
    # false jumps in ordinary reading, and the landing text of 42% of them recurs, against 9% of
    # real jumps' (jump_eval.py, train).
    jump_repeated: float = 4.0


class LocalFollower:
    def __init__(self, index: CorpusIndex, config: FollowerConfig | None = None):
        self.ix = index
        self.cfg = config or FollowerConfig()
        self.frame_s = FRAME_S  # the CTC model's frame hop (ctc_adapter.TokenFollower: 80 ms)
        ix = index
        # First letter of each word (ix.letters is the whole corpus, word by word).
        self._word_letter = np.r_[0, np.flatnonzero(np.diff(ix.letter_word) != 0) + 1, ix.letters.size]
        new_line = np.r_[True, (np.diff(ix.word_segment) != 0) | (np.diff(ix.word_dua) != 0)]
        self._line_no = np.cumsum(new_line) - 1  # running line number over the corpus
        self._line_start = np.flatnonzero(new_line)  # first word of each running line
        self._dua_text: dict[int, tuple] = {}  # du'a -> (first word, letters, letter -> word - first)
        self._dua_starts: dict[int, np.ndarray] = {}  # du'a -> its lines' first letters (jump_seg)
        self._dua_repeated: dict[int, np.ndarray] = {}  # du'a -> per word: its phrase recurs (jump_repeated)
        self.reset()

    def reset(self) -> None:
        self.word: int | None = None
        self._lapse_since: float | None = None
        self._disagree_since: float | None = None
        self._disagree_from: int | None = None
        self._pending: int | None = None  # a big move waiting for confirm_steps
        self._pending_n = 0
        self._leap: int | None = None  # a catch-up target waiting for leap_confirm
        self._leap_n = 0
        self._back: int | None = None  # a step back waiting for back_confirm
        self._back_n = 0
        self._ahead_since: float | None = None
        self._ahead_from: int | None = None
        self._line_to: int | None = None  # a later line waiting for line_confirm
        self._line_n = 0
        self._quiet_since: float | None = None
        self._back_at: float | None = None  # a step back of a line or more: when, and from where
        self._back_from: int | None = None
        self._jump: int | None = None  # a jump target waiting for jump_confirm
        self._jump_n = 0
        self._jumped_at: float | None = None

    def _repeated(self, d: int) -> np.ndarray:
        """Per word of du'a `d`: does its phrase (it and up to two words before it in its line) occur
        more than once in the du'a?"""
        if d not in self._dua_repeated:
            ix = self.ix
            lo, hi = ix.dua_word_span[d]
            keys = []
            for w in range(lo, hi):
                k0 = max(int(self._line_start[self._line_no[w]]), w - 2)
                keys.append(tuple(ix.words[k].text for k in range(k0, w + 1)))
            count: dict[tuple, int] = {}
            for w in range(lo, hi):  # occurrences of each phrase anywhere in the du'a, across lines too
                for n in (1, 2, 3):
                    if w + n <= hi:
                        key = tuple(ix.words[k].text for k in range(w, w + n))
                        count[key] = count.get(key, 0) + 1
            self._dua_repeated[d] = np.array([count.get(k, 0) > 1 for k in keys])
        return self._dua_repeated[d]

    def _jump_search(self, lp: np.ndarray, hmm_word: int, line_mass=None) -> int | None:
        """A word of the du'a outside the follower's reach whose letters explain the latest
        jump_window s far better (jump_margin) than any word within reach; None if there is none."""
        ix, cfg, cur = self.ix, self.cfg, self.word
        d = int(ix.word_dua[cur])
        if d not in self._dua_text:
            lo, hi = ix.dua_word_span[d]
            la, lb = self._word_letter[lo], self._word_letter[hi]
            self._dua_text[d] = (lo, ix.letters[la:lb].astype(np.int64), ix.letter_word[la:lb] - lo)
        lo, letters, owner = self._dua_text[d]
        sub = squeeze_blanks(lp[-max(1, int(round(cfg.jump_window / self.frame_s))) :])
        a, b = self._reference_span(cur)
        if b <= a:
            return None
        if cfg.jump_seg:
            if d not in self._dua_starts:
                first = np.r_[True, owner[1:] != owner[:-1]]  # each word's first letter
                self._dua_starts[d] = first & np.isin(owner + lo, self._line_start)
            near = ix.letters[self._word_letter[a] : self._word_letter[b]]
            sc, here = jump_scores(sub, near, letters, self._dua_starts[d])
        else:
            sc = end_scores(sub, letters)
        best = np.full(int(owner[-1]) + 1, -np.inf)
        np.maximum.at(best, owner, sc)
        if not cfg.jump_seg:
            here = best[a - lo : b - lo].max()
        out = best.copy()
        out[a - lo : b - lo] = -np.inf
        m = out.max()
        if m - here < cfg.jump_margin:
            return None
        cands = np.flatnonzero(out >= m - 0.5) + lo
        ref = hmm_word if ix.word_dua[hmm_word] == d else cur
        target = int(cands[np.argmin(np.abs(cands - ref))])
        if cfg.jump_mass > 0:
            mass = 0.0
            if line_mass is not None and ix.word_dua[hmm_word] == d:
                k = int(self._line_no[target])
                for kk in (k - 1, k, k + 1):
                    if 0 <= kk < self._line_start.size and ix.word_dua[self._line_start[kk]] == d:
                        mass = max(mass, line_mass(int(self._line_start[kk])))
            agree = mass >= cfg.jump_mass
        else:
            agree = ix.word_dua[hmm_word] == d and abs(int(self._line_no[hmm_word]) - int(self._line_no[target])) <= 1
        if not agree and m - here < max(cfg.jump_margin, cfg.jump_alone):
            return None
        if cfg.jump_repeated > 0 and self._repeated(d)[target - lo] and m - here < cfg.jump_margin + cfg.jump_repeated:
            return None
        return target

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
        tail = max(1, int(round(cfg.onset_window / self.frame_s)))
        hits = np.flatnonzero(letter[-tail:])
        if not hits.size:
            return False
        k = lp.shape[0] - tail + int(hits[0])  # first letter frame in the tail
        before = np.flatnonzero(letter[:k])
        run = k - (int(before[-1]) + 1 if before.size else 0)  # blank frames just before it
        return run * self.frame_s >= cfg.onset_gap - 1e-9

    def _restarts(self, cur: int) -> np.ndarray:
        """First words of the current line and of the restart_lines lines before it (same du'a)."""
        lo = self.ix.dua_word_span[self.ix.word_dua[cur]][0]
        k = int(self._line_no[cur])
        out = self._line_start[max(0, k - self.cfg.restart_lines) : k + 1]
        return out[out >= lo]

    def _reference_span(self, anchor: int) -> tuple[int, int]:
        """Words [anchor - back, anchor + ahead) in the anchor's du'a: the follower's reach."""
        lo, hi = self.ix.dua_word_span[self.ix.word_dua[anchor]]
        back = self.cfg.back_words
        if self.cfg.restart_cost > 0:
            back = max(back, anchor - int(self._restarts(anchor)[0]))
        return max(lo, anchor - back), min(hi, anchor + self.cfg.ahead_words)

    def _reference(self, anchor: int) -> tuple[np.ndarray, np.ndarray]:
        """Letters of the words within reach of the anchor (_reference_span), and each letter's word."""
        a, b = self._reference_span(anchor)
        la, lb = self._word_letter[a], self._word_letter[b]
        return self.ix.letters[la:lb], self.ix.letter_word[la:lb]

    def step(self, lp: np.ndarray, t: float, hmm_word: int | None, lead_word: int | None = None,
             quiet_now: float | None = None, line_mass=None) -> int | None:
        """One CTC window (frames x columns, trimmed to its real length) ending at
        time t; `hmm_word` is the tracker's current word (None: not locked), and
        `lead_word` its lead (display) word, used only by onset_confirm; `quiet_now`
        the page's stop detector (seconds without voice, None: unknown), for line_quiet;
        `line_mass(word)` the tracker's belief in that word's line (Tracker.line_mass), for jump_mass.
        Returns the word to show, or None to leave the display to the tracker."""
        ix, cfg = self.ix, self.cfg
        if hmm_word is None:
            if self.word is None or cfg.lapse_hold <= 0:
                self.reset()
                return None
            if self._lapse_since is None:
                self._lapse_since = t
            if t - self._lapse_since > cfg.lapse_hold:
                self.reset()
                return None
            hmm_word = self.word  # follow on from where it is
        else:
            self._lapse_since = None
        if self._back_at is not None and (t - self._back_at > cfg.repeat_hold or self.word >= self._back_from):
            self._back_at = self._back_from = None
        holding = self._back_at is not None and hmm_word > self.word  # a repeat: no pull forward
        if self._jumped_at is not None and t - self._jumped_at >= cfg.jump_hold:
            self._jumped_at = None
        holding = holding or self._jumped_at is not None  # just jumped: the tracker hasn't heard it yet
        if self.word is None or ix.word_dua[self.word] != ix.word_dua[hmm_word]:
            self.word, self._disagree_since = hmm_word, None
            self._back_at = self._back_from = self._jumped_at = None
        elif holding:
            self._disagree_since = None
        elif abs(int(self._line_no[self.word]) - int(self._line_no[hmm_word])) > 1:
            if self._disagree_since is None or (cfg.reset_stuck and self.word != self._disagree_from):
                self._disagree_since, self._disagree_from = t, self.word
            elif t - self._disagree_since >= cfg.reset_after:
                self.word, self._disagree_since = hmm_word, None
        else:
            self._disagree_since = None
        if cfg.ahead_words_reanchor > 0 and not holding and ix.word_dua[hmm_word] == ix.word_dua[self.word] \
                and hmm_word - self.word >= cfg.ahead_words_reanchor:
            if self._ahead_since is None or (cfg.ahead_stuck and self.word != self._ahead_from):
                self._ahead_since, self._ahead_from = t, self.word
            elif t - self._ahead_since >= cfg.ahead_after:
                self.word, self._ahead_since = hmm_word, None
        else:
            self._ahead_since = None
        lp = lp[-max(1, int(round(cfg.window_s / self.frame_s))) :]
        if not lp.size or (cfg.tail_s > 0 and not has_speech(lp[-max(1, int(round(cfg.tail_s / self.frame_s))) :], 1)):
            # Silence: stay put, unless the tracker heard more of this line.
            if cfg.quiet_after > 0 and not holding and ix.word_dua[hmm_word] == ix.word_dua[self.word] \
                    and 0 < hmm_word - self.word <= cfg.quiet_words \
                    and self._line_no[hmm_word] == self._line_no[self.word]:
                if self._quiet_since is None:
                    self._quiet_since = t
                if t - self._quiet_since >= cfg.quiet_after - 1e-9:
                    self.word, self._quiet_since = hmm_word, None
                    self._pending, self._pending_n, self._back, self._back_n = None, 0, None, 0
            else:
                self._quiet_since = None
            return self.word
        self._quiet_since = None
        if self._onset(lp, lead_word):
            self.word, self._pending, self._pending_n = self.word + 1, None, 0
            return self.word
        if cfg.jump_margin > 0:
            target = self._jump_search(lp, hmm_word, line_mass)
            if target is None:
                self._jump, self._jump_n = None, 0
            else:
                near = self._jump is not None and abs(target - self._jump) <= 2
                self._jump_n = self._jump_n + 1 if near else 1
                self._jump = target
                if self._jump_n >= cfg.jump_confirm:
                    self.word, self._jump, self._jump_n, self._jumped_at = target, None, 0, t
                    self._pending, self._pending_n, self._back, self._back_n = None, 0, None, 0
                    self._leap, self._leap_n, self._line_to, self._line_n = None, 0, None, 0
                    self._back_at = self._back_from = None
                    return self.word
        letters, owner = self._reference(self.word)
        if not letters.size:
            return self.word
        starts = None
        if cfg.word_starts:
            starts = (owner <= self.word) | np.r_[True, owner[1:] != owner[:-1]]
        sc = end_scores(squeeze_blanks(lp), letters.astype(np.int64), starts) / cfg.temp
        # Best end letter per word, then the move penalties.
        w0 = int(owner[0])
        best = np.full(int(owner[-1]) - w0 + 1, -np.inf)
        np.maximum.at(best, owner - w0, sc)
        words = np.arange(w0, w0 + best.size)
        cur = self.word
        pen = cfg.beta_back * np.maximum(0, cur - words)
        if cfg.restart_cost > 0:
            r = self._restarts(cur) - w0
            r = r[(r >= 0) & (r < best.size)]
            pen[r] = np.minimum(pen[r], cfg.restart_cost)
        best -= pen
        beyond = words > cur + cfg.max_jump
        far = None
        if cfg.leap_margin > 0 and beyond.any():
            k = int(np.argmax(np.where(beyond, best, -np.inf)))
            if best[k] - np.max(np.where(beyond, -np.inf, best)) >= cfg.leap_margin:
                far = int(words[k])
        best[beyond] = -np.inf
        new = int(words[int(np.argmax(best))])
        if far is None:
            self._leap, self._leap_n = None, 0
        else:
            self._leap_n = self._leap_n + 1 if self._leap is not None and far >= self._leap else 1
            self._leap = far
            if self._leap_n >= cfg.leap_confirm:
                self.word, self._leap, self._leap_n = far, None, 0
                self._pending, self._pending_n, self._back, self._back_n = None, 0, None, 0
                return self.word
        if new < cur:
            if cur - new < cfg.back_min:
                new = cur
            elif cfg.back_confirm > 1:
                self._back_n = self._back_n + 1 if self._back is not None and abs(new - self._back) <= 1 else 1
                self._back = new
                if self._back_n < cfg.back_confirm:
                    return self.word
        if new >= cur:
            self._back, self._back_n = None, 0
        if cfg.confirm_steps > 1 and (new < cur or new > cur + 1):
            self._pending_n = self._pending_n + 1 if new == self._pending else 1
            self._pending = new
            if self._pending_n < cfg.confirm_steps:
                return self.word
        if cfg.line_quiet > 0 and quiet_now is not None and quiet_now >= cfg.line_quiet and new > cur \
                and self._line_no[new] != self._line_no[cur]:
            return self.word  # no voice: not the next line
        if cfg.line_confirm > 1 and new > cur and self._line_no[new] != self._line_no[cur]:
            line = int(self._line_no[new])
            self._line_n = self._line_n + 1 if self._line_to == line else 1
            self._line_to = line
            if self._line_n < cfg.line_confirm:
                return self.word
        self._line_to, self._line_n = None, 0
        self.word, self._pending, self._pending_n = new, None, 0
        if new < cur:
            self._back, self._back_n = None, 0
            if cfg.repeat_hold > 0 and self._line_no[new] < self._line_no[cur]:
                self._back_at, self._back_from = t, cur if self._back_from is None else max(cur, self._back_from)
        return self.word
