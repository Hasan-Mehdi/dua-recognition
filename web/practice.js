// In-browser port of src/dua_recognition/practice.py: practice mode's checker. Did the reader say
// each line, where it stands?
//
// Once the display has stayed two lines past line k, the frames the follower committed since just
// before the line before it are explained twice: the best path that says line k (its first word and
// half its letters), and the best path without it, both between the last earlier line not judged
// left out and the lines after k up to one past the display (each of which may be left out), with
// free ends and the follower's filler for anything else said. The difference (nats) is the line's
// score: heard at >= thetaHeard; left out at <= -thetaUnheard if the display never showed the line,
// at <= -thetaShown if it did; not sure in between. tests/test_practice_parity.py checks this
// against the Python, decision for decision.

const NEG = -1e30;

export const PRACTICE_DEFAULTS = {
  fillCost: 1.5,
  cFill: -8.0,
  cSkip: -4.0,
  maxLater: 4,
  settleS: 0.5,
  spanMinS: 20.0,
  minFrac: 0.5,
  marginS: 1.5,
  maxSpanS: 45.0,
  thetaHeard: 4.0,
  thetaUnheard: 4.0,
  thetaShown: 20.0, // null: the display isn't asked
  // the sound gate: left out only if, of the last gateLines lines decided (left-out ones aside), at
  // least gateHeard were heard
  gateLines: 4,
  gateHeard: 0.6,
  gateMin: 1,
  lookahead: 0.2,
  hop: 0.1,
  frameS: 0.02,
};

export function verdict(score, shown, cfg) {
  if (score <= -cfg.thetaUnheard && (cfg.thetaShown == null || !shown || score <= -cfg.thetaShown)) return "left out";
  return score >= cfg.thetaHeard ? "heard" : "not sure";
}

// The sound gate (practice.py gated): a left-out verdict stands only after enough lines heard
// lately. history: the verdicts given so far, left-out ones aside; updated.
export function gated(v, history, cfg) {
  if (v === "left out" && cfg.gateLines > 0) {
    const recent = history.slice(-cfg.gateLines);
    let heard = 0;
    for (const x of recent) if (x === "heard") heard++;
    if (recent.length < cfg.gateMin || heard < cfg.gateHeard * recent.length - 1e-9) v = "not sure";
  }
  if (v !== "left out") history.push(v);
  return v;
}

// The best path of frames lp (T x C, flat) through lines of letters in order: an optional line may
// be left out (cSkip each), a mandatory one may not; a free start in the first nFirst letters, the
// filler after any line (cFill to enter), a free end from letter endFrom on.
function spanScore(lp, C, segs, optional, nFirst, endFrom, cfg) {
  const T = Math.floor(lp.length / C);
  let n = 0;
  for (const x of segs) n += x.length;
  if (!segs.length || !T || !n) return NEG;
  const S = segs.length;
  const r = new Int32Array(n);
  const segOf = new Int32Array(n);
  const segFirst = new Int32Array(S);
  const segLast = new Int32Array(S);
  let o = 0;
  for (let s = 0; s < S; s++) {
    segFirst[s] = o;
    for (const c of segs[s]) {
      r[o] = c;
      segOf[o] = s;
      o++;
    }
    segLast[s] = o - 1;
  }
  const { fillCost, cFill, cSkip } = cfg;
  let L = new Float64Array(n).fill(NEG);
  let B = new Float64Array(n).fill(NEG);
  const Fm = new Float64Array(S).fill(NEG);
  let nL = new Float64Array(n);
  let nB = new Float64Array(n);
  const exitS = new Float64Array(S);
  const into = new Float64Array(S);
  let best = NEG;
  for (let c = 1; c < C; c++) best = Math.max(best, lp[c]);
  let Fpre = Math.max(lp[0], best - fillCost);
  let Fpost = NEG;
  for (let j = 0; j < nFirst; j++) L[j] = lp[r[j]];
  for (let t = 1; t < T; t++) {
    const b0 = t * C;
    best = NEG;
    for (let c = 1; c < C; c++) best = Math.max(best, lp[b0 + c]);
    const fe = Math.max(lp[b0], best - fillCost);
    for (let s = 0; s < S; s++) exitS[s] = Math.max(Math.max(L[segLast[s]], B[segLast[s]]), Fm[s]);
    into[0] = NEG;
    for (let s2 = 1; s2 < S; s2++) {
      let v = NEG;
      let cost = 0.0;
      for (let s = s2 - 1; s >= 0; s--) {
        v = Math.max(v, exitS[s] + cost);
        if (!optional[s]) break;
        cost += cSkip;
      }
      into[s2] = v;
    }
    let m = NEG;
    for (let j = endFrom; j < n; j++) m = Math.max(m, Math.max(L[j], B[j]));
    for (let j = 0; j < n; j++) {
      let enter;
      if (j > 0 && segFirst[segOf[j]] === j) {
        enter = into[segOf[j]];
      } else if (j > 0) {
        enter = B[j - 1];
        if (r[j] !== r[j - 1]) enter = Math.max(enter, L[j - 1]);
      } else {
        enter = NEG;
      }
      if (j < nFirst) enter = Math.max(enter, Fpre);
      nL[j] = Math.max(L[j], enter) + lp[b0 + r[j]];
      nB[j] = Math.max(L[j], B[j]) + lp[b0];
    }
    for (let s = 0; s < S; s++) Fm[s] = Math.max(Fm[s], Math.max(L[segLast[s]], B[segLast[s]]) + cFill) + fe;
    Fpost = Math.max(Fpost, m) + fe;
    Fpre = Fpre + fe;
    [L, nL] = [nL, L];
    [B, nB] = [nB, B];
  }
  let m = Fpost;
  for (let j = endFrom; j < n; j++) m = Math.max(m, Math.max(L[j], B[j]));
  return m;
}

// Was line k said, between the last line said before it (prev, or null) and where the reader is
// now (later lines, each of which may have been left out)? See practice.py line_score.
export function lineScore(lp, C, prev, line, firstWord, later, cfg) {
  const ctx = prev ? [prev] : [];
  const nFirst = prev ? prev.length : 1;
  const ctxLen = prev ? prev.length : 0;
  const withK = spanScore(lp, C, [...ctx, line, ...later], [...ctx.map(() => false), false, ...later.map(() => true)],
    nFirst, ctxLen + Math.max(1, firstWord) - 1, cfg);
  let without;
  if (ctx.length) {
    without = spanScore(lp, C, [...ctx, ...later], [false, ...later.map(() => true)], nFirst, 0, cfg);
  } else if (later.length) {
    without = NEG;
    for (let i = 0; i < later.length; i++) {
      const segs = later.slice(i);
      without = Math.max(without, spanScore(lp, C, segs, segs.map(() => true), 1, 0, cfg));
    }
  } else {
    without = 0;
    const T = Math.floor(lp.length / C);
    for (let t = 0; t < T; t++) {
      let best = NEG;
      for (let c = 1; c < C; c++) best = Math.max(best, lp[t * C + c]);
      without += Math.max(lp[t * C], best - cfg.fillCost);
    }
  }
  return withK - without;
}

// The frame stream the follower commits (stream-follower.js _step): from each window, the frames
// since the last step that end `lookahead` before its end. add() returns them and their end times.
export class FrameCommitter {
  constructor(cfg = {}) {
    this.cfg = { ...PRACTICE_DEFAULTS, ...cfg };
    this.tStream = null;
  }

  add(frames, T, C, t) {
    const { lookahead, hop, frameS } = this.cfg;
    const la = Math.round(lookahead / frameS);
    const h = Math.round(hop / frameS);
    const commitEnd = T - la;
    const kk = this.tStream == null ? h : Math.round((t - lookahead - this.tStream) / frameS);
    this.tStream = t - lookahead;
    if (kk <= 0 || commitEnd <= 0) return { rows: new Float32Array(0), times: new Float64Array(0) };
    const from = Math.max(0, commitEnd - kk);
    const rows = frames.slice(from * C, commitEnd * C);
    const n = commitEnd - from;
    const times = new Float64Array(n);
    for (let i = 0; i < n; i++) times[i] = this.tStream - frameS * (n - 1 - i);
    return { rows, times };
  }
}

function lowerBound(a, n, x) { // first index with a[i] >= x
  let lo = 0, hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function upperBound(a, n, x) { // first index with a[i] > x
  let lo = 0, hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// The checker online (practice.py PracticeChecker): committed frames arrive with their end times
// (addFrames), the display's line index after each step (step: null when it shows nothing); each
// call returns the lines decided then, [{line, t, score, shown, verdict}].
export class PracticeChecker {
  constructor(lineLetters, lo, hi, cfg = {}, firstWords = null) {
    this.letters = lineLetters;
    this.lo = lo;
    this.hi = hi;
    this.cfg = { ...PRACTICE_DEFAULTS, ...cfg };
    this.firstWords = firstWords;
    this.C = 0;
    this.buf = new Float32Array(0);
    this.times = new Float64Array(0);
    this.n = 0;
    this.verdicts = new Map();
    this.shown = new Set();
    this.tLastOn = new Map();
    this.since = new Map();
    this.furthest = -1;
    this.tFirst = null;
    this.tLast = 0;
    this.history = []; // the verdicts given, left-out ones aside (the sound gate)
  }

  addFrames(rows, C, times) {
    const k = times.length;
    if (!k) return;
    this.C = C;
    const keep = times[k - 1] - this.cfg.maxSpanS - this.cfg.marginS - 5.0;
    let drop = 0;
    while (drop < this.n && this.times[drop] < keep) drop++;
    if (drop < 500) drop = 0; // compact once 10 s of frames are past any span's reach, not every step
    const n = this.n - drop + k;
    if (n * C > this.buf.length || drop) {
      const buf = new Float32Array(Math.max(n * C, 2 * this.buf.length - drop * C));
      buf.set(this.buf.subarray(drop * C, this.n * C));
      const tt = new Float64Array(buf.length / C);
      tt.set(this.times.subarray(drop, this.n));
      this.buf = buf;
      this.times = tt;
      this.n -= drop;
    }
    for (let i = 0; i < rows.length; i++) {
      const v = rows[i];
      this.buf[this.n * C + i] = Number.isFinite(v) ? Math.max(v, -50.0) : -50.0;
    }
    this.times.set(times, this.n);
    this.n += k;
  }

  _span(beforeLine, t) {
    const cfg = this.cfg;
    let t0 = this.tFirst ?? 0;
    let found = false;
    for (const [li, tt] of this.tLastOn) {
      if (li < beforeLine && (!found || tt > t0)) {
        t0 = tt;
        found = true;
      }
    }
    t0 -= cfg.marginS;
    t0 = Math.max(Math.min(t0, t - cfg.lookahead - cfg.spanMinS), t - cfg.lookahead - cfg.maxSpanS);
    const a = lowerBound(this.times, this.n, t0);
    const b = upperBound(this.times, this.n, t - cfg.lookahead);
    return this.buf.subarray(a * this.C, Math.max(a, b) * this.C);
  }

  _decide(k, t, at) {
    const cfg = this.cfg;
    let p = k - 1;
    while (p >= this.lo && this.verdicts.has(p) && verdict(...this.verdicts.get(p), cfg) === "left out") p--;
    const prev = p >= this.lo ? this.letters[p] : null;
    const later = [];
    for (let j = k + 1; j <= Math.min(this.hi, at + 1, k + cfg.maxLater); j++) later.push(this.letters[j]);
    let fw = this.firstWords ? this.firstWords[k] : this.letters[k].length;
    fw = Math.max(fw, Math.ceil(cfg.minFrac * this.letters[k].length));
    const lp = this._span(p >= this.lo ? p : k - 1, t);
    const score = this.C ? lineScore(lp, this.C, prev, this.letters[k], fw, later, cfg) : 0;
    const shown = this.shown.has(k);
    this.verdicts.set(k, [score, shown]);
    return { line: k, t, score, shown, verdict: gated(verdict(score, shown, cfg), this.history, cfg) };
  }

  step(t, li) {
    if (this.tFirst == null) this.tFirst = t;
    this.tLast = t;
    if (li == null) return [];
    this.shown.add(li);
    this.tLastOn.set(li, t);
    this.furthest = Math.max(this.furthest, li);
    for (const m of [...this.since.keys()]) if (m > li) this.since.delete(m);
    for (let m = this.lo + 2; m <= li; m++) if (!this.since.has(m)) this.since.set(m, t);
    const out = [];
    for (let k = this.lo; k <= Math.min(this.hi, li - 2); k++) {
      if (!this.verdicts.has(k) && t - this.since.get(k + 2) >= this.cfg.settleS - 1e-9) out.push(this._decide(k, t, li));
    }
    return out;
  }

  // The session is over: every line not decided yet, now (through line upTo only: a reader who
  // stops partway meant to, and the lines they never reached aren't left out).
  finish(t = null, upTo = null) {
    const tt = t ?? this.tLast;
    const hi = upTo == null ? this.hi : Math.min(this.hi, upTo);
    const out = [];
    for (let k = this.lo; k <= hi; k++) if (!this.verdicts.has(k)) out.push(this._decide(k, tt, Math.max(this.furthest, k)));
    return out;
  }
}

// The lines of du'a d in a tracker.js CorpusIndex, as the checker takes them: each line's segment id
// and letter codes, the letters in its first word, and segment id -> line index.
export function linesOf(ix, d) {
  const [lo, hi] = ix.duaWordSpan[d];
  const segs = [], letters = [], firstWords = [], lineOf = new Map();
  let cur = null;
  for (let w = lo; w < hi; w++) {
    const a = w > 0 ? ix.wordEnds[w - 1] + 1 : 0;
    const codes = ix.letters.subarray(a, ix.wordEnds[w] + 1);
    const seg = ix.wordSegment[w];
    if (seg !== cur) {
      cur = seg;
      lineOf.set(seg, segs.length);
      segs.push(seg);
      letters.push([]);
      firstWords.push(codes.length);
    }
    for (const c of codes) letters.at(-1).push(c);
  }
  return { segs, letters: letters.map((x) => Int32Array.from(x)), firstWords, lineOf };
}
