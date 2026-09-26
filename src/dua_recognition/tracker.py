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

from dataclasses import dataclass, field, replace

import numpy as np

from .align import CorpusIndex, encode


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
    # "None of these": a state for recitations that aren't in the corpus. It
    # explains a window as if its transcript missed at `null_rate` edits per
    # letter. A du'a the corpus has matches far better than that, even misheard;
    # a different du'a that only shares stock phrases ("yā qāḍiya ḥawāʾij
    # al-sāʾilīn") doesn't, and without this state the tracker had to pick the
    # nearest text anyway. Leave-one-out on the test reciters (scripts/ooc_eval.py):
    # a du'a missing from the corpus showed some other du'a 68% of the time,
    # 16% with this; line accuracy on known du'as 85.4% -> 85.3%. 0 = off.
    null_rate: float = 0.45
    null_enter: float = 0.002  # per update: the reciter moves on to something unknown
    null_leave: float = 0.05  # per update: ...and back to something known
    # Reported only once the winning du'a holds this much posterior mass.
    min_dua_confidence: float = 0.7
    # Texts that share a passage (Ayat al-Kursi inside Sahifa 54, the Ramadan
    # night du'as, stock salawat): while the reciter is inside it the evidence
    # can't tell them apart, and the mass splits between them, often leaving
    # none above min_dua_confidence. So du'as whose most likely positions sit in
    # the same run of at least `same_text_words` identical words, with a few
    # still to come, count as one for that threshold. Shown: the du'a already on
    # screen if it was found before the passage began (someone reciting Sahifa 54
    # from its start), else the one the passage sits nearest the start of
    # (someone reciting Ayat al-Kursi). Long enough that a shared opening salawat
    # or bismillah doesn't count: those still wait for the text to become
    # distinctive. 0 = off.
    same_text_words: int = 12
    same_text_ahead: int = 3
    # The belief is about the end of the audio window, but it reaches the
    # screen later (ASR time), and the transcript's tail lags the voice. So
    # the live display shows the belief predicted forward by the measured
    # delay plus this much (`update(..., lead=delay + display_lead)`). Chosen
    # on the train split; on test it cuts line-switch lag from 1.8 s to 0.6 s
    # at 0.5 s ASR delay and raises line accuracy (docs/results/display_lead.md).
    display_lead: float = 0.5
    # When the display moves on to the next line, start at its first word.
    # The lead's estimate often lands two or three words into a new line and
    # the highlight appeared to skip its opening words (4 jerks/min on test;
    # 0.1 with this, and closer to the word being said: docs/results/display_lead.md).
    enter_line_at_start: bool = True
    # The reciter has been silent this long (update(..., quiet=)): they aren't
    # moving, so neither is the display. The last words stay in the 6 s window
    # for seconds after they stop, and without this the prediction kept walking
    # on through the next line.
    still_after: float = 0.3
    # The lead may move the highlight along the current line but not on to the
    # next one: only evidence starts a new line. Otherwise a reciter who pauses
    # at the end of a line sees the next one light up before they begin it.
    lead_within_line: bool = False
    # ...and past this much silence the belief itself stops moving too: it
    # moves only for the time they were making sound. (With Silero alone this
    # cost 11 points, because it hears long melodic notes as silence;
    # asr.quiet_at_end also counts loudness, and it costs ~0.5.)
    still_motion_after: float = 0.3
    # The lead may carry the display into the next line only while the
    # reciter is making sound (quiet below this). Someone who has just stopped
    # at the end of a line hasn't started the next one. inf = always may.
    lead_cross_quiet: float = 0.1
    # After this much silence, a display that already ran into a line the
    # evidence hasn't reached steps back to the end of the line they stopped
    # on. (The lead crosses at the moment they stop, before any silence can
    # be heard.) inf = never. With 4 s pauses inserted into the test
    # recordings, the next line was on screen for 84% of each pause before
    # these rules and 33% with them; a reciter who flows through breaths sees
    # a step back now and then, which is why majlis mode uses RECITER.
    retreat_after: float = 1.0


# Following a professional reciter (majlis mode): they flow from line to line
# and a breath isn't a stop, so the display may run on into the next line as
# it always did. The hold and the frozen belief still apply (docs/results/pauses.md).
RECITER = {"lead_cross_quiet": float("inf"), "retreat_after": float("inf")}


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
    # Other du'as reading the same words here (TrackerConfig.same_text_words).
    same_as: list[str] = field(default_factory=list)


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
        new_line = np.r_[True, (np.diff(ix.word_segment) != 0) | (np.diff(ix.word_dua) != 0)]
        self._line_first = np.maximum.accumulate(np.where(new_line, self._idx, 0))
        # First word of the following line (n_words past the end of the corpus).
        starts = np.r_[np.flatnonzero(new_line), ix.n_words]
        self._next_line = starts[np.searchsorted(starts, self._idx, side="right")]
        self._spans = np.array(ix.dua_word_span).reshape(-1, 2)
        self._edge_cache: dict[int, np.ndarray] = {}
        self._text = [w.text for w in ix.words]
        self.reset()

    def reset(self) -> None:
        self.post = self._floor.copy()
        self.tempo = np.full(max(1, len(self.cfg.speeds)), 1.0 / max(1, len(self.cfg.speeds)))
        # (speed kernels, shifted beliefs) of the last prediction: row s of
        # kernels @ shifted is the forward prediction under speed s.
        self._fwd_by_speed: tuple[np.ndarray, np.ndarray] | None = None
        self._last_word: int | None = None
        self._shown: Position | None = None
        self.null = 0.5 if self.cfg.null_rate > 0 else 0.0  # P(not in the corpus)
        self._reported: int | None = None  # the du'a last reported (index)
        self._found_alone = False  # ...and it was told apart from every other text

    # -- predict ---------------------------------------------------------
    def _locked(self) -> bool:
        return bool(self._dua_mass().max() >= self.cfg.lock_confidence)

    def _dua_mass(self) -> np.ndarray:
        # post is over the corpus's words; `null` is the rest.
        return (1 - self.null) * np.bincount(self.ix.word_dua, weights=self.post, minlength=len(self.ix.duas))

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
        kernels = self._speed_kernels(dt, n_fwd)  # speeds x (n_fwd + 1)
        cs = np.r_[0.0, np.cumsum(self.post)]
        shifted = np.stack([self._shift(self.post, cs, d) for d in range(n_fwd + 1)])
        self._fwd_by_speed = (kernels, shifted) if self.cfg.tempo_memory > 0 else None
        fwd = (self.tempo @ kernels) @ shifted
        back = np.zeros_like(self.post)
        for d in range(1, cfg.back_words + 1):
            back += self._shift(self.post, cs, -d)
        back /= cfg.back_words
        p = (1 - cfg.p_back) * fwd + cfg.p_back * back
        self.post = (1 - tele) * p / p.sum() + tele * self._floor

    def _shift(self, post: np.ndarray, cs: np.ndarray, d: int) -> np.ndarray:
        """Mass moved d words (back if d < 0), clamped to its own du'a: what
        overshoots a du'a's last (first) word piles up there. `cs` is post's
        cumulative sum with a leading 0. Plain array shifts plus a fix-up at
        each du'a's edge: several times faster than scattering word by word."""
        if d == 0:
            return post.copy()
        out = np.zeros_like(post)
        lo, hi = self._spans[:, 0], self._spans[:, 1]
        if d > 0:
            out[d:] = post[:-d]
            out[self._edges(d)] = 0.0  # what slid in from the previous du'a
            e = hi - 1
            out[e] = cs[e + 1] - cs[np.maximum(lo, e - d)]
        else:
            d = -d
            out[:-d] = post[d:]
            out[self._edges(-d)] = 0.0  # what slid in from the next du'a
            out[lo] = cs[np.minimum(hi, lo + d + 1)] - cs[lo]
        return out

    def _edges(self, d: int) -> np.ndarray:
        """The first d words of every du'a (d > 0), or the last -d (d < 0)."""
        if d not in self._edge_cache:
            k = abs(d)
            lo, hi = self._spans[:, 0], self._spans[:, 1]
            self._edge_cache[d] = np.concatenate(
                [np.arange(a, min(b, a + k)) if d > 0 else np.arange(max(a, b - k), b) for a, b in zip(lo, hi)])
        return self._edge_cache[d]

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
    def update(self, transcript: str, dt: float, lead: float = 0.0, quiet: float = 0.0) -> Position:
        """Advance by dt seconds, then condition on the latest window's text.

        With `lead` > 0 the position returned is the belief predicted `lead`
        seconds past the window's end (see TrackerConfig.display_lead), and a
        silent window holds the last position shown rather than falling back.
        `quiet`: seconds since the reciter last spoke (asr.quiet_at_end); the
        position moves only for the time they were speaking.
        """
        costs = self.ix.word_costs(transcript) if transcript else None
        return self.update_costs(costs, dt, lead, n_letters=len(encode(transcript)) if transcript else 0, quiet=quiet)

    def update_costs(self, costs: np.ndarray | None, dt: float, lead: float = 0.0,
                     n_letters: int | None = None, quiet: float = 0.0) -> Position:
        """`update` with the alignment already done (evaluation reuses it).

        `n_letters`: the transcript's length in letters (align.encode), for the
        "not in the corpus" state. If not given it is estimated as the largest
        cost: no alignment costs more than deleting the whole transcript, and
        across the corpus the worst one comes within a letter of that.
        """
        if n_letters is None and costs is not None:
            n_letters = int(costs.max())
        # An empty window is almost always the pause between lines: the
        # reciter isn't moving, so neither does the belief (bar the floors).
        moving = dt - quiet if quiet > self.cfg.still_motion_after else dt
        self._advance(max(0.0, moving) if costs is not None else 0.0, locked=self._locked())
        if costs is not None:
            kappa = self.cfg.kappa if self._locked() else self.cfg.kappa_search
            c0 = float(costs.min())
            lik = np.exp(-kappa * (costs - c0).astype(np.float64))
            if self._fwd_by_speed is not None:
                # Which speed predicted this evidence best? Forget a little first.
                kernels, shifted = self._fwd_by_speed
                ev = kernels @ (shifted @ lik)
                w = self.tempo ** self.cfg.tempo_memory * ev
                if w.sum() > 0:
                    self.tempo = w / w.sum()
            if self.cfg.null_rate > 0 and n_letters:
                cfg = self.cfg
                known = (1 - self.null) * (1 - cfg.null_enter) + self.null * cfg.null_leave
                in_corpus = known * float(self.post @ lik)
                none = (1 - known) * np.exp(-kappa * min(50.0, max(-50.0, cfg.null_rate * n_letters - c0)))
                self.null = none / (in_corpus + none)
            self.post = self.post * lik
            self.post /= self.post.sum()
        if lead <= 0:
            return self._forward_only(self.position())
        if (costs is None or quiet > self.cfg.still_after) and self._shown is not None:
            now = self.position()
            if quiet > self.cfg.retreat_after:
                self._retreat(now)
            return replace(self._shown, candidates=now.candidates)
        led = self.lookahead(lead)
        if self.cfg.lead_within_line or quiet >= self.cfg.lead_cross_quiet:
            now = self.position()
            if now.dua is not None and (led.dua, led.segment) != (now.dua, now.segment):
                # Stay on the evidence's line, as far along it as the lead reached (its last word).
                last = int(self._next_line[now.word]) - 1
                led = replace(now, word=last, at_line_end=True)
        self._shown = self._forward_only(led)
        return self._shown

    def _retreat(self, now: Position) -> None:
        """The display is in a later line of the same du'a than the evidence: go
        back to the end of the evidence's line."""
        shown = self._shown
        if (now.dua is None or now.word is None or shown.word is None or now.dua != shown.dua
                or shown.word <= now.word or self._line_first[shown.word] == self._line_first[now.word]):
            return
        last = int(self._next_line[now.word]) - 1
        self._shown = replace(now, word=last, at_line_end=True)
        self._last_word = last

    @property
    def speed(self) -> float:
        """Current pace estimate in words/s (the tempo-weighted speed prior)."""
        return float(np.dot(self.tempo, self.cfg.speeds)) if self.cfg.speeds else self.cfg.max_speed / 2

    def lookahead(self, seconds: float) -> Position:
        """Where the reciter probably is `seconds` from now; the belief is unchanged."""
        post, fwd = self.post, self._fwd_by_speed
        self._advance(seconds, locked=self._locked())
        pos = self.position()
        self.post, self._fwd_by_speed = post, fwd
        return pos

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
        elif (last is not None and pos.word is not None and self.cfg.enter_line_at_start
              and self._line_first[pos.word] == self._next_line[last]
              and self.ix.word_dua[pos.word] == self.ix.word_dua[last]):
            # Moved on to the next line: start at its first word.
            pos.word = int(self._line_first[pos.word])
            pos.at_line_end = pos.word + 1 >= self.ix.n_words or self._line_first[pos.word + 1] != pos.word
        self._last_word = pos.word
        return pos

    def _likeliest(self, d: int) -> int:
        lo, hi = self.ix.dua_word_span[d]
        return lo + int(self.post[lo:hi].argmax())

    def _same_passage(self, a: int, b: int) -> bool:
        """Du'as a and b are both, most likely, inside one identical passage."""
        n, ahead = self.cfg.same_text_words, self.cfg.same_text_ahead
        (lo_a, hi_a), (lo_b, hi_b) = self.ix.dua_word_span[a], self.ix.dua_word_span[b]
        wa, wb = self._likeliest(a), self._likeliest(b)
        t = self._text
        fwd = 0
        while fwd < n and wa + fwd < hi_a and wb + fwd < hi_b and t[wa + fwd] == t[wb + fwd]:
            fwd += 1
        back = 0
        while back < n and wa - back - 1 >= lo_a and wb - back - 1 >= lo_b and t[wa - back - 1] == t[wb - back - 1]:
            back += 1
        at_end = wa + fwd == hi_a or wb + fwd == hi_b  # one of them finishes here
        return (fwd >= ahead or at_end) and fwd + back >= n

    def _same_text(self, dua_mass: np.ndarray, order: np.ndarray) -> list[int]:
        """The top du'a, and any of the next likeliest in the same passage as it."""
        group = [int(order[0])]
        if self.cfg.same_text_words <= 0:
            return group
        for o in order[1:8]:
            if dua_mass[o] < 0.01:
                break
            if self._same_passage(group[0], int(o)):
                group.append(int(o))
        return group

    def position(self) -> Position:
        ix = self.ix
        dua_mass = self._dua_mass()
        order = np.argsort(-dua_mass)
        top = [(ix.dua_ids[i], float(dua_mass[i])) for i in order[:3]]
        group = self._same_text(dua_mass, order)
        conf = float(dua_mass[group].sum())
        if conf < self.cfg.min_dua_confidence:
            return Position(None, conf, None, 0.0, None, candidates=top)
        if len(group) == 1:
            d, self._found_alone = group[0], True
        elif self._reported in group and self._found_alone:
            d = self._reported
        else:
            d = min(group, key=lambda g: self._likeliest(g) - ix.dua_word_span[g][0])
            self._found_alone = False
        self._reported = d
        lo, hi = ix.dua_word_span[d]
        seg_ids = ix.word_segment[lo:hi]
        seg_mass = (1 - self.null) * np.bincount(seg_ids, weights=self.post[lo:hi])
        s = int(seg_mass.argmax())
        # The word shown must lie in the line shown: the weighted median of the
        # posterior within that line (a lone argmax flickers across a flat line,
        # and taken over the whole du'a it can land in a different line).
        in_seg = lo + np.flatnonzero(seg_ids == s)
        cum = np.cumsum(self.post[in_seg])
        word = int(in_seg[min(int(np.searchsorted(cum, cum[-1] / 2)), len(in_seg) - 1)])
        at_end = word + 1 >= hi or ix.word_segment[word + 1] != ix.word_segment[word]
        same = [ix.dua_ids[g] for g in group if g != d]
        return Position(ix.dua_ids[d], conf, s, float(seg_mass[s] / dua_mass[d]), word, at_end, top, same)

    def prompt(self, n_words: int = 12) -> str | None:
        """Reference text just before the current position, for ASR biasing."""
        pos = self.position()
        if pos.word is None or pos.dua_confidence < 0.9:
            return None
        return self.ix.text_before(pos.word, n_words)
