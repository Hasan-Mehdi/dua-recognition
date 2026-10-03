// In-browser port of src/dua_recognition/{ctc_align,follower}.py: follow the reciter word by
// word from a CTC model's frames (web/ctc-worker.js), between the tracker's updates.
//
// Every few hundred ms the latest frames are scored against the letters of the words around
// the current one (endScores: the best CTC path through that text ending on each letter), and
// the follower moves to the word whose letter explains the audio best, with a penalty for
// going back and a cap on how far it may jump forward. When the reciter stops, the frames
// after their last letter are blanks, and a path can sit in a blank after that letter: a pause
// holds the last word by construction, with no lead and no step-back rule. The tracker keeps
// what it is good at: which du'a, which line, and re-anchoring the follower when the two
// disagree by more than a line for a while. tests/test_follower_parity.py checks this against
// the Python, move for move.

const NEG = -1e9;
const LOG_P_BLANK = Math.log(0.95);

// The phone default since 2026-10-01 (docs/results/phone_follower.md): round 1's scoring with
// the catch-up leap, the step-back rules, word starts and the re-anchor rules; that evening the
// quiet catch-up, line restarts and the stuck-only forward re-anchor, for a step every 0.1 s
// (docs/results/phone_latency.md).
export const FOLLOW_DEFAULTS = {
  windowS: 2.0, // seconds of the latest frames scored each step
  betaBack: 2.0, // log-likelihood cost per word moved back (3 until 2026-10-01)
  maxJump: 2, // words it may move forward in one step
  temp: 1.0, // divides the CTC log-likelihoods
  resetAfter: 3.0, // seconds of > 1 line disagreement before re-anchoring on the tracker
  backWords: 6,
  aheadWords: 20,
  tailS: 0.5, // hold if these last seconds hold no letter frame at all
  confirmSteps: 1, // a move back, or more than one word forward, must win this many steps in a row
  lapseHold: 0, // while the tracker has lost the du'a, keep following this many seconds (0 = reset)
  // Catch up: a word beyond maxJump that beats every word within reach by leapMargin nats, on
  // leapConfirm steps in a row (each at or past the last), is gone to at once (0 = off).
  leapMargin: 4,
  leapConfirm: 2,
  // Steps back: shorter than backMin words are ignored; a step back must win backConfirm steps in a row.
  backMin: 2,
  backConfirm: 4, // steps: 0.4 s at a step every 0.1 s (2 at 0.2 s until 2026-10-01)
  // The scored path may begin anywhere in the current word or before it, but in a later word only
  // at its first letter (a swallowed ending can't put the display on the next line's first word).
  wordStarts: true,
  // Forward re-anchor: the tracker's word aheadWordsReanchor or more words ahead (same du'a) for
  // aheadAfter s: go there (0 = off). Only forward: a wrong tracker can't pull the display back.
  aheadWordsReanchor: 4,
  aheadAfter: 2.0,
  // ...and only while the follower is stuck (its word unchanged for those seconds): a reader saying a
  // line again moves the follower back along it while the tracker is still ahead.
  aheadStuck: true,
  // Quiet catch-up: the reciter silent and the tracker's word ahead in the same line, by at most
  // quietWords, for quietAfter s: go there (0 = off). A word the CTC model didn't hear (the du'a's
  // last word in x0fo and 0t98, which Whisper heard) otherwise stays unlit until it is said again.
  quietAfter: 0.5,
  quietWords: 3,
  // Line restarts: going back to the first word of the current line, or of one of the restartLines
  // lines before it, costs at most restartCost nats instead of betaBack per word (0 = off).
  restartCost: 3,
  restartLines: 1,
  // A step forward into a later line must win lineConfirm steps in a row (1 = at once): at a step
  // every 0.1 s one window can hear a line's last word as the next line's first.
  lineConfirm: 2,
  // The line re-anchor (resetAfter) only while the follower is stuck too: a follower moving word by
  // word is hearing the reciter, and a tracker two lines behind pulled it back (9puq).
  resetStuck: true,
  // After a step back of a line or more (a repeat), no tracker re-anchor takes the follower forward
  // for repeatHold s, or until it is back where it was: the tracker rarely believes a repeat (diqq).
  repeatHold: 20,
  // No step into a later line while the page's stop detector (gate.js LiveQuiet) has heard no voice
  // for lineQuiet s (0 = off): letters made up out of room tone can match the next line's first word.
  lineQuiet: 0.3,
  // Jumps (0 = off): the latest jumpWindow s of frames are also scored against the whole du'a; a
  // place outside the follower's reach that beats every word within reach by jumpMargin nats, on
  // jumpConfirm steps in a row (at most two words apart), is gone to, and for jumpHold s the tracker
  // can't re-anchor the follower. Refrains: the copy nearest the tracker's word (follower.py).
  jumpMargin: 7,
  jumpWindow: 1.2,
  jumpConfirm: 4,
  jumpHold: 6.0,
  // ...and unless the tracker's word is within a line of the target, it takes jumpAlone nats (0 =
  // jumpMargin either way): a far place can fit 1.2 s by chance; the tracker soon agrees with a real jump.
  jumpAlone: 0,
  // Agreement by belief (0 = by the tracker's word): the tracker's posterior mass on the target's line
  // or a neighbour (step's lineMass, tracker.js lineMass) at least this.
  jumpMass: 0,
  // Two-part scoring (follower.py jump_seg): the window explained by the text within reach up to some
  // frame, then by the du'a from the first letter of a line on (jumpScores), against the text within reach alone.
  jumpSeg: false,
  // Extra nats needed (0 = none) when the target's phrase (it and up to two words before it in its
  // line) occurs more than once in the du'a: formulaic lines drew most false jumps (follower.py).
  jumpRepeated: 4,
};

// Best CTC path log-likelihood of frames lp (T x C, row-major Float32Array) ending on each
// letter of r (ctc_align.end_scores): free start (only where starts[j], if given), CTC topology,
// float64 sums.
export function endScores(lp, T, C, r, starts = null) {
  const J = r.length;
  const L = new Float64Array(J);
  const B = new Float64Array(J);
  for (let j = 0; j < J; j++) {
    const ok = !starts || starts[j];
    L[j] = ok ? lp[r[j]] : NEG;
    B[j] = ok ? lp[0] : NEG;
  }
  for (let t = 1; t < T; t++) {
    const o = t * C;
    const bl = lp[o];
    for (let j = J - 1; j >= 0; j--) {
      let enter = NEG;
      if (j > 0) {
        enter = B[j - 1];
        if (r[j] !== r[j - 1] && L[j - 1] > enter) enter = L[j - 1];
      }
      const nb = Math.max(B[j], L[j]) + bl;
      L[j] = Math.max(L[j], enter) + lp[o + r[j]];
      B[j] = nb;
    }
  }
  const out = new Float64Array(J);
  for (let j = 0; j < J; j++) out[j] = Math.max(L[j], B[j]);
  return out;
}

// A jump within the window (ctc_align.jump_scores): frames up to some point follow the `near` letters
// (free start), then the path enters `r` at a letter where starts[j] and ends on each letter of `r`.
// Returns { out: those scores, here: the best path through `near` alone over the whole window }.
export function jumpScores(lp, T, C, near, r, starts) {
  const J1 = near.length;
  const J = r.length;
  if (!J || !J1 || T < 2) return { out: new Float64Array(J).fill(NEG), here: NEG };
  const L1 = new Float64Array(J1);
  const B1 = new Float64Array(J1);
  for (let j = 0; j < J1; j++) {
    L1[j] = lp[near[j]];
    B1[j] = lp[0];
  }
  const L = new Float64Array(J).fill(NEG);
  const B = new Float64Array(J).fill(NEG);
  for (let t = 1; t < T; t++) {
    const o = t * C;
    const bl = lp[o];
    let prefix = NEG; // best path through the near text ending at frame t - 1
    for (let j = 0; j < J1; j++) {
      if (L1[j] > prefix) prefix = L1[j];
      if (B1[j] > prefix) prefix = B1[j];
    }
    for (let j = J - 1; j >= 0; j--) {
      let enter = NEG;
      if (j > 0) {
        enter = B[j - 1];
        if (r[j] !== r[j - 1] && L[j - 1] > enter) enter = L[j - 1];
      }
      if (starts[j] && prefix > enter) enter = prefix;
      const nb = Math.max(B[j], L[j]) + bl;
      L[j] = Math.max(L[j], enter) + lp[o + r[j]];
      B[j] = nb;
    }
    for (let j = J1 - 1; j >= 0; j--) {
      let enter = NEG;
      if (j > 0) {
        enter = B1[j - 1];
        if (near[j] !== near[j - 1] && L1[j - 1] > enter) enter = L1[j - 1];
      }
      const nb = Math.max(B1[j], L1[j]) + bl;
      L1[j] = Math.max(L1[j], enter) + lp[o + near[j]];
      B1[j] = nb;
    }
  }
  const out = new Float64Array(J);
  for (let j = 0; j < J; j++) out[j] = Math.max(L[j], B[j]);
  let here = NEG;
  for (let j = 0; j < J1; j++) here = Math.max(here, L1[j], B1[j]);
  return { out, here };
}

// Collapse each run of blank-dominated frames into one (their mean), as float32 (ctc_align.squeeze_blanks).
export function squeezeBlanks(lp, T, C) {
  const isBlank = new Uint8Array(T);
  let any = false;
  for (let t = 0; t < T; t++) {
    isBlank[t] = lp[t * C] > LOG_P_BLANK ? 1 : 0;
    any ||= isBlank[t] === 1;
  }
  if (!any) return { lp, T };
  const runOf = new Int32Array(T);
  let n = 0;
  for (let t = 0; t < T; t++) {
    if (t > 0 && (isBlank[t] !== isBlank[t - 1] || !isBlank[t])) n++;
    runOf[t] = n;
  }
  n += 1;
  const sum = new Float32Array(n * C);
  const count = new Int32Array(n);
  for (let t = 0; t < T; t++) {
    const k = runOf[t];
    count[k]++;
    for (let c = 0; c < C; c++) sum[k * C + c] = Math.fround(sum[k * C + c] + lp[t * C + c]);
  }
  const out = new Float32Array(n * C);
  for (let k = 0; k < n; k++) for (let c = 0; c < C; c++) out[k * C + c] = sum[k * C + c] / count[k];
  return { lp: out, T: n };
}

// Any letters at all in frames [from, T)? Frames where some letter beats blank.
export function hasSpeech(lp, T, C, from = 0, minFrames = 1) {
  let n = 0;
  for (let t = Math.max(0, from); t < T; t++) {
    const o = t * C;
    let best = -Infinity;
    for (let c = 1; c < C; c++) if (lp[o + c] > best) best = lp[o + c];
    if (best > lp[o] && ++n >= minFrames) return true;
  }
  return false;
}

export class LocalFollower {
  // index: tracker.js CorpusIndex
  constructor(index, config = {}, frameS = 0.02) {
    this.ix = index;
    this.cfg = { ...FOLLOW_DEFAULTS, ...config };
    this.frameS = frameS;
    const n = index.nWords;
    this.firstLetter = new Int32Array(n + 1);
    for (let w = 0; w < n; w++) this.firstLetter[w + 1] = index.wordEnds[w] + 1;
    this.lineNo = new Int32Array(n);
    const starts = n ? [0] : [];
    for (let w = 1; w < n; w++) {
      const newLine = index.wordSegment[w] !== index.wordSegment[w - 1] || index.wordDua[w] !== index.wordDua[w - 1];
      this.lineNo[w] = this.lineNo[w - 1] + (newLine ? 1 : 0);
      if (newLine) starts.push(w);
    }
    this.lineStart = Int32Array.from(starts); // first word of each running line
    this.lineStartSet = new Set(starts);
    this.duaText = new Map(); // du'a -> { lo, letters, owner (letter -> word - lo) }, for jumps
    this.duaRepeated = new Map(); // du'a -> per word: its phrase recurs in the du'a (jumpRepeated)
    this.reset();
  }

  reset() {
    this.word = null;
    this.lapseSince = null;
    this.disagreeSince = null;
    this.disagreeFrom = null;
    this.pending = null;
    this.pendingN = 0;
    this.leap = null;
    this.leapN = 0;
    this.back = null;
    this.backN = 0;
    this.aheadSince = null;
    this.aheadFrom = null;
    this.quietSince = null;
    this.backAt = null; // a step back of a line or more (a repeat): when, and from where
    this.backFrom = null;
    this.lineTo = null; // a later line waiting for lineConfirm
    this.lineN = 0;
    this.jump = null; // a jump target waiting for jumpConfirm
    this.jumpN = 0;
    this.jumpedAt = null;
  }

  // First words of the current line and of the restartLines lines before it (same du'a).
  _restarts(cur) {
    const lo = this.ix.duaWordSpan[this.ix.wordDua[cur]][0];
    const k = this.lineNo[cur];
    const out = [];
    for (let j = Math.max(0, k - this.cfg.restartLines); j <= k; j++) if (this.lineStart[j] >= lo) out.push(this.lineStart[j]);
    return out;
  }

  // Words [anchor - back, anchor + ahead) in the anchor's du'a: the follower's reach.
  _referenceSpan(anchor) {
    const { ix, cfg } = this;
    const [lo, hi] = ix.duaWordSpan[ix.wordDua[anchor]];
    let back = cfg.backWords;
    if (cfg.restartCost > 0) back = Math.max(back, anchor - this._restarts(anchor)[0]);
    return { a: Math.max(lo, anchor - back), b: Math.min(hi, anchor + cfg.aheadWords) };
  }

  _reference(anchor) {
    const { a, b } = this._referenceSpan(anchor);
    return { a, la: this.firstLetter[a], lb: this.firstLetter[b] };
  }

  // Per word of du'a d: does its phrase (it and up to two words before it in its line) occur more than
  // once in the du'a? (follower.py _repeated)
  _repeated(d) {
    let rep = this.duaRepeated.get(d);
    if (!rep) {
      const { ix } = this;
      const [lo, hi] = ix.duaWordSpan[d];
      const count = new Map();
      for (let w = lo; w < hi; w++) {
        for (let n = 1; n <= 3 && w + n <= hi; n++) {
          const key = ix.words.slice(w, w + n).map((x) => x.text).join(" ");
          count.set(key, (count.get(key) || 0) + 1);
        }
      }
      rep = new Uint8Array(hi - lo);
      for (let w = lo; w < hi; w++) {
        const k0 = Math.max(this.lineStart[this.lineNo[w]], w - 2);
        rep[w - lo] = (count.get(ix.words.slice(k0, w + 1).map((x) => x.text).join(" ")) || 0) > 1 ? 1 : 0;
      }
      this.duaRepeated.set(d, rep);
    }
    return rep;
  }

  // A word of the du'a outside the follower's reach whose letters explain the latest jumpWindow s
  // far better (jumpMargin) than any word within reach, or null (follower.py _jump_search).
  _jumpSearch(lp, n, C, hmmWord, lineMass = null) {
    const { ix, cfg } = this;
    const cur = this.word;
    const d = ix.wordDua[cur];
    let text = this.duaText.get(d);
    if (!text) {
      const [lo, hi] = ix.duaWordSpan[d];
      const la = this.firstLetter[lo];
      const owner = new Int32Array(this.firstLetter[hi] - la);
      for (let w = lo; w < hi; w++) owner.fill(w - lo, this.firstLetter[w] - la, this.firstLetter[w + 1] - la);
      // Each line's first letter (jumpSeg): a word's first letter, in a word that starts a line.
      const starts = new Uint8Array(owner.length);
      for (let k = 0; k < owner.length; k++) {
        starts[k] = (k === 0 || owner[k] !== owner[k - 1]) && this.lineStartSet.has(owner[k] + lo) ? 1 : 0;
      }
      text = { lo, letters: ix.letters.subarray(la, this.firstLetter[hi]), owner, starts };
      this.duaText.set(d, text);
    }
    const { lo, letters, owner, starts } = text;
    const keep = Math.max(1, Math.round(cfg.jumpWindow / this.frameS));
    const s0 = Math.max(0, n - keep);
    const sq = squeezeBlanks(lp.subarray(s0 * C, n * C), n - s0, C);
    const { a, b } = this._referenceSpan(cur);
    if (b <= a) return null;
    let sc;
    let inside = -Infinity;
    if (cfg.jumpSeg) {
      const res = jumpScores(sq.lp, sq.T, C, ix.letters.subarray(this.firstLetter[a], this.firstLetter[b]), letters, starts);
      sc = res.out;
      inside = res.here;
    } else sc = endScores(sq.lp, sq.T, C, letters, null);
    const best = new Float64Array(owner[owner.length - 1] + 1).fill(-Infinity);
    for (let k = 0; k < sc.length; k++) if (sc[k] > best[owner[k]]) best[owner[k]] = sc[k];
    let m = -Infinity;
    for (let w = 0; w < best.length; w++) {
      if (w >= a - lo && w < b - lo) {
        if (!cfg.jumpSeg) inside = Math.max(inside, best[w]);
      } else m = Math.max(m, best[w]);
    }
    if (!(m - inside >= cfg.jumpMargin)) return null;
    const ref = ix.wordDua[hmmWord] === d ? hmmWord : cur;
    let pick = null;
    for (let w = 0; w < best.length; w++) {
      if (w >= a - lo && w < b - lo) continue;
      if (best[w] >= m - 0.5 && (pick == null || Math.abs(w + lo - ref) < Math.abs(pick - ref))) pick = w + lo;
    }
    let agree;
    if (cfg.jumpMass > 0) {
      let mass = 0;
      if (lineMass && ix.wordDua[hmmWord] === d) {
        const k = this.lineNo[pick];
        for (const kk of [k - 1, k, k + 1]) {
          if (kk >= 0 && kk < this.lineStart.length && ix.wordDua[this.lineStart[kk]] === d) {
            mass = Math.max(mass, lineMass(this.lineStart[kk]));
          }
        }
      }
      agree = mass >= cfg.jumpMass;
    } else agree = ix.wordDua[hmmWord] === d && Math.abs(this.lineNo[hmmWord] - this.lineNo[pick]) <= 1;
    if (!agree && !(m - inside >= Math.max(cfg.jumpMargin, cfg.jumpAlone))) return null;
    if (cfg.jumpRepeated > 0 && this._repeated(d)[pick - lo] && !(m - inside >= cfg.jumpMargin + cfg.jumpRepeated)) return null;
    return pick;
  }

  // frames: Float32Array (T x C) of log posteriors ending at time t (s); hmmWord: the tracker's
  // current word (evidence position, null when not locked); quietNow: the page's stop detector
  // (seconds without voice, null: unknown), for lineQuiet; lineMass(w): the tracker's belief in
  // w's line, for jumpMass. Returns the word to show, or null.
  step(frames, T, C, t, hmmWord, quietNow = null, lineMass = null) {
    const { ix, cfg } = this;
    if (hmmWord == null) {
      if (this.word == null || cfg.lapseHold <= 0) {
        this.reset();
        return null;
      }
      this.lapseSince ??= t;
      if (t - this.lapseSince > cfg.lapseHold) {
        this.reset();
        return null;
      }
      hmmWord = this.word; // follow on from where it is
    } else this.lapseSince = null;
    if (this.backAt != null && (t - this.backAt > cfg.repeatHold || this.word >= this.backFrom)) {
      this.backAt = null;
      this.backFrom = null;
    }
    if (this.jumpedAt != null && t - this.jumpedAt >= cfg.jumpHold) this.jumpedAt = null;
    // A repeat: no pull forward. Just jumped: the tracker hasn't heard it yet.
    const holding = (this.backAt != null && hmmWord > this.word) || this.jumpedAt != null;
    if (this.word == null || ix.wordDua[this.word] !== ix.wordDua[hmmWord]) {
      this.word = hmmWord;
      this.disagreeSince = null;
      this.backAt = null;
      this.backFrom = null;
      this.jumpedAt = null;
    } else if (holding) this.disagreeSince = null;
    else if (Math.abs(this.lineNo[this.word] - this.lineNo[hmmWord]) > 1) {
      if (this.disagreeSince == null || (cfg.resetStuck && this.word !== this.disagreeFrom)) {
        this.disagreeSince = t;
        this.disagreeFrom = this.word;
      } else if (t - this.disagreeSince >= cfg.resetAfter) {
        this.word = hmmWord;
        this.disagreeSince = null;
      }
    } else this.disagreeSince = null;
    if (cfg.aheadWordsReanchor > 0 && !holding && ix.wordDua[hmmWord] === ix.wordDua[this.word]
        && hmmWord - this.word >= cfg.aheadWordsReanchor) {
      if (this.aheadSince == null || (cfg.aheadStuck && this.word !== this.aheadFrom)) {
        this.aheadSince = t;
        this.aheadFrom = this.word;
      } else if (t - this.aheadSince >= cfg.aheadAfter) {
        this.word = hmmWord;
        this.aheadSince = null;
      }
    } else this.aheadSince = null;
    const keep = Math.max(1, Math.round(cfg.windowS / this.frameS));
    const t0 = Math.max(0, T - keep);
    const lp = frames.subarray(t0 * C, T * C);
    const n = T - t0;
    if (!n) return this.word;
    if (cfg.tailS > 0 && !hasSpeech(lp, n, C, n - Math.max(1, Math.round(cfg.tailS / this.frameS)), 1)) {
      // Silence: stay put, unless the tracker heard more of this line.
      const d = hmmWord - this.word;
      if (cfg.quietAfter > 0 && !holding && ix.wordDua[hmmWord] === ix.wordDua[this.word] && d > 0 && d <= cfg.quietWords
          && this.lineNo[hmmWord] === this.lineNo[this.word]) {
        this.quietSince ??= t;
        if (t - this.quietSince >= cfg.quietAfter - 1e-9) {
          Object.assign(this, { word: hmmWord, quietSince: null, pending: null, pendingN: 0, back: null, backN: 0 });
        }
      } else this.quietSince = null;
      return this.word;
    }
    this.quietSince = null;
    if (cfg.jumpMargin > 0) {
      const target = this._jumpSearch(lp, n, C, hmmWord, lineMass);
      if (target == null) {
        this.jump = null;
        this.jumpN = 0;
      } else {
        const near = this.jump != null && Math.abs(target - this.jump) <= 2;
        this.jumpN = near ? this.jumpN + 1 : 1;
        this.jump = target;
        if (this.jumpN >= cfg.jumpConfirm) {
          Object.assign(this, { word: target, jump: null, jumpN: 0, jumpedAt: t, pending: null, pendingN: 0, back: null,
            backN: 0, leap: null, leapN: 0, lineTo: null, lineN: 0, backAt: null, backFrom: null });
          return this.word;
        }
      }
    }
    const { a, la, lb } = this._reference(this.word);
    if (lb <= la) return this.word;
    const letters = ix.letters.subarray(la, lb);
    const sq = squeezeBlanks(lp, n, C);
    let starts = null;
    if (cfg.wordStarts) {
      starts = new Uint8Array(letters.length);
      let w = a;
      for (let k = 0; k < letters.length; k++) {
        if (la + k >= this.firstLetter[w + 1]) w++;
        starts[k] = w <= this.word || la + k === this.firstLetter[w] ? 1 : 0;
      }
    }
    const sc = endScores(sq.lp, sq.T, C, letters, starts);
    // Best end letter per word, then the move penalties.
    let word = a;
    let bestWord = a;
    let best = -Infinity;
    let wordBest = -Infinity;
    let farWord = null;
    let farBest = -Infinity;
    const cur = this.word;
    const restarts = cfg.restartCost > 0 ? new Set(this._restarts(cur)) : null;
    const close = (w, v) => {
      let pen = cfg.betaBack * Math.max(0, cur - w);
      if (restarts?.has(w)) pen = Math.min(pen, cfg.restartCost);
      const s = v / cfg.temp - pen;
      if (w > cur + cfg.maxJump) {
        if (s > farBest) [farBest, farWord] = [s, w];
      } else if (s > best) [best, bestWord] = [s, w];
    };
    for (let k = 0; k < letters.length; k++) {
      const w = word + (la + k >= this.firstLetter[word + 1] ? 1 : 0);
      if (w !== word) {
        close(word, wordBest);
        word = w;
        wordBest = -Infinity;
      }
      if (sc[k] > wordBest) wordBest = sc[k];
    }
    close(word, wordBest);
    let next = bestWord;
    const far = cfg.leapMargin > 0 && farWord != null && farBest - best >= cfg.leapMargin ? farWord : null;
    if (far == null) {
      this.leap = null;
      this.leapN = 0;
    } else {
      this.leapN = this.leap != null && far >= this.leap ? this.leapN + 1 : 1;
      this.leap = far;
      if (this.leapN >= cfg.leapConfirm) {
        Object.assign(this, { word: far, leap: null, leapN: 0, pending: null, pendingN: 0, back: null, backN: 0 });
        return this.word;
      }
    }
    if (next < cur) {
      if (cur - next < cfg.backMin) next = cur;
      else if (cfg.backConfirm > 1) {
        this.backN = this.back != null && Math.abs(next - this.back) <= 1 ? this.backN + 1 : 1;
        this.back = next;
        if (this.backN < cfg.backConfirm) return this.word;
      }
    }
    if (next >= cur) {
      this.back = null;
      this.backN = 0;
    }
    if (cfg.confirmSteps > 1 && (next < cur || next > cur + 1)) {
      this.pendingN = next === this.pending ? this.pendingN + 1 : 1;
      this.pending = next;
      if (this.pendingN < cfg.confirmSteps) return this.word;
    }
    if (cfg.lineQuiet > 0 && quietNow != null && quietNow >= cfg.lineQuiet && next > cur
        && this.lineNo[next] !== this.lineNo[cur]) return this.word; // no voice: not the next line
    if (cfg.lineConfirm > 1 && next > cur && this.lineNo[next] !== this.lineNo[cur]) {
      const line = this.lineNo[next];
      this.lineN = this.lineTo === line ? this.lineN + 1 : 1;
      this.lineTo = line;
      if (this.lineN < cfg.lineConfirm) return this.word;
    }
    this.lineTo = null;
    this.lineN = 0;
    this.word = next;
    this.pending = null;
    this.pendingN = 0;
    if (next < cur) {
      this.back = null;
      this.backN = 0;
      if (cfg.repeatHold > 0 && this.lineNo[next] < this.lineNo[cur]) {
        this.backAt = t;
        this.backFrom = this.backFrom == null ? cur : Math.max(cur, this.backFrom);
      }
    }
    return this.word;
  }
}
