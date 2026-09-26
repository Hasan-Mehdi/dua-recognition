// In-browser port of src/dua_recognition/{text,align,tracker}.py.
// Same normalization, same semi-global alignment, same HMM, same defaults, so
// the phone version follows a recitation exactly like the server does.
// tests/test_web_parity.py checks this against the Python on real transcripts.

const TASHKEEL = /[ً-ٰٟۖ-ۭـ]/g;
const NON_ARABIC = /[^ء-ي\s]/g;
const FOLD = {
  "آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا", "ى": "ي", "ی": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه", "ک": "ك",
  "ہ": "ه", "ۃ": "ه", "ھ": "ه", "ۂ": "ه", "ے": "ي", "ۓ": "ي",
};

export function normalize(text) {
  let t = text.normalize("NFKC").replace(TASHKEEL, "");
  t = Array.from(t, (c) => FOLD[c] ?? c).join("");
  return t.replace(NON_ARABIC, " ").replace(/\s+/g, " ").trim();
}

function encode(text) {
  const out = [];
  for (const c of normalize(text)) {
    const code = c.codePointAt(0);
    if (code >= 0x0621 && code <= 0x064a) out.push(code - 0x0620);
  }
  return Int16Array.from(out);
}

// D[m][e] of the edit-distance DP with a free start in r (see align.py), by
// Myers' bit-parallel algorithm in 32-bit blocks: each DP column is a set of
// vertical deltas, one bit per fragment letter, advanced one reference letter
// at a time (align.py _myers_blocks is the same with 64-bit blocks). About 30x
// faster than filling the DP, which matters with 500 texts on a phone.
export function semiglobalEndCosts(h, r) {
  const m = h.length;
  const n = r.length;
  const nb = (m + 31) >>> 5;
  const peq = new Uint32Array(64 * nb); // letter codes are < 64
  for (let i = 0; i < m; i++) peq[h[i] * nb + (i >>> 5)] |= 1 << (i & 31);
  const pv = new Uint32Array(nb).fill(0xffffffff);
  const mv = new Uint32Array(nb);
  const last = (1 << ((m - 1) & 31)) >>> 0;
  const TOP = 0x80000000;
  const out = new Int32Array(n);
  let score = m;
  for (let j = 0; j < n; j++) {
    const row = r[j] * nb;
    let hin = 0; // horizontal delta entering the block from above (row 0 is free)
    for (let b = 0; b < nb; b++) {
      const p = pv[b];
      const mm = mv[b];
      const neg = hin < 0 ? 1 : 0;
      const pos = hin > 0 ? 1 : 0;
      const eq0 = peq[row + b];
      const xv = (eq0 | mm) >>> 0;
      const eq = (eq0 | neg) >>> 0;
      const xh = (((((((eq & p) >>> 0) + p) >>> 0) ^ p) | eq) >>> 0);
      let ph = (mm | ~(xh | p)) >>> 0;
      let mh = (p & xh) >>> 0;
      if (b === nb - 1) {
        if (ph & last) score++;
        else if (mh & last) score--;
      }
      hin = ph & TOP ? 1 : mh & TOP ? -1 : 0;
      ph = ((ph << 1) | pos) >>> 0;
      mh = ((mh << 1) | neg) >>> 0;
      pv[b] = (mh | ~(xv | ph)) >>> 0;
      mv[b] = (ph & xv) >>> 0;
    }
    out[j] = score;
  }
  return out;
}

export class CorpusIndex {
  // corpus: [{id, name_en, name_ar, segments: [{id, ar, tl, en}]}]
  constructor(corpus) {
    this.duas = corpus;
    this.duaIds = corpus.map((d) => d.id);
    this.words = []; // {dua, segment, token}
    this.duaWordSpan = [];
    const letters = [];
    const wordEnds = [];
    corpus.forEach((dua, di) => {
      const first = this.words.length;
      for (const seg of dua.segments) {
        seg.ar.split(/\s+/).filter(Boolean).forEach((token, ti) => {
          for (const w of normalize(token).split(" ")) {
            const codes = encode(w);
            if (!codes.length) continue;
            for (const c of codes) letters.push(c);
            wordEnds.push(letters.length - 1);
            this.words.push({ dua: di, segment: seg.id, token: ti, text: w });
          }
        });
      }
      this.duaWordSpan.push([first, this.words.length]);
    });
    this.letters = Int16Array.from(letters);
    this.wordEnds = Int32Array.from(wordEnds);
    this.wordDua = Int32Array.from(this.words, (w) => w.dua);
    this.wordSegment = Int32Array.from(this.words, (w) => w.segment);
  }

  get nWords() {
    return this.words.length;
  }

  wordCosts(fragment) {
    const h = encode(fragment);
    if (!h.length) return null;
    const costs = semiglobalEndCosts(h, this.letters);
    return Int32Array.from(this.wordEnds, (e) => costs[e]);
  }

  textBefore(word, nWords = 12) {
    const lo = Math.max(this.duaWordSpan[this.wordDua[word]][0], word - nWords + 1);
    return this.words.slice(lo, word + 1).map((w) => w.text).join(" ");
  }
}

export const DEFAULTS = {
  kappa: 0.15, kappaSearch: 1.2, lockConfidence: 0.95, maxSpeed: 4.0,
  // Speed prior and tempo adaptation: see TrackerConfig.speeds / tempo_memory in tracker.py.
  speeds: [0.59, 0.99, 1.15, 1.3, 1.44, 1.55, 1.67, 1.84, 2.05, 2.39], tempoMemory: 0.98,
  pBack: 0.1, backWords: 8, pTeleport: 0.01, pTeleportLocked: 0.01, startWeight: 0.3, startWords: 12, minDuaConfidence: 0.7, sameTextWords: 12, sameTextAhead: 3,
  displayLead: 0.5, // seconds shown ahead of the measured delay (see tracker.py)
  enterLineAtStart: true, // moving on to the next line starts at its first word
  // Pauses (TrackerConfig.still_after ... retreat_after in tracker.py): `quiet` is how
  // long the reciter has been silent at the window's end (asr-worker.js quietAtEnd).
  stillAfter: 0.3, leadWithinLine: false, stillMotionAfter: 0.3, leadCrossQuiet: 0.1, retreatAfter: 1.0,
  // "None of these": recitations not in the corpus (TrackerConfig.null_rate in tracker.py).
  nullRate: 0.45, nullEnter: 0.002, nullLeave: 0.05,
};

// Following a professional reciter (majlis mode): see RECITER in tracker.py.
export const RECITER = { leadCrossQuiet: Infinity, retreatAfter: Infinity };

export class Tracker {
  constructor(index, config = {}) {
    this.ix = index;
    this.cfg = { ...DEFAULTS, ...config };
    const n = index.nWords;
    const nDuas = index.duaWordSpan.length;
    this.floor = new Float64Array(n);
    this.first = new Int32Array(n);
    this.last = new Int32Array(n);
    for (const [lo, hi] of index.duaWordSpan) {
      const k = Math.min(this.cfg.startWords, hi - lo);
      for (let w = lo; w < hi; w++) {
        const anywhere = 1 / (nDuas * (hi - lo));
        const start = w - lo < k ? 1 / (nDuas * k) : 0;
        this.floor[w] = this.cfg.startWeight * start + (1 - this.cfg.startWeight) * anywhere;
        this.first[w] = lo;
        this.last[w] = hi - 1;
      }
    }
    // First word of each word's line, and of the line after it.
    this.lineFirst = new Int32Array(n);
    this.nextLine = new Int32Array(n).fill(n);
    for (let w = 0; w < n; w++) {
      const starts = w === 0 || index.wordSegment[w] !== index.wordSegment[w - 1] || index.wordDua[w] !== index.wordDua[w - 1];
      this.lineFirst[w] = starts ? w : this.lineFirst[w - 1];
    }
    for (let w = n - 2; w >= 0; w--) {
      this.nextLine[w] = this.lineFirst[w + 1] === w + 1 ? w + 1 : this.nextLine[w + 1];
    }
    this.reset();
  }

  reset() {
    this.post = Float64Array.from(this.floor);
    this.lastWord = null;
    this.shown = null;
    this.null = this.cfg.nullRate > 0 ? 0.5 : 0; // P(not in the corpus)
    this.reported = null; // the du'a last reported (index)
    this.foundAlone = false; // ...and it was told apart from every other text
    const k = Math.max(1, this.cfg.speeds.length);
    this.tempo = new Float64Array(k).fill(1 / k);
    this.fwdBySpeed = null;
  }

  // P(moved d words in dt | speed), one row per speed, d = 0..nFwd (Poisson, truncated).
  _speedKernels(dt, nFwd) {
    const speeds = this.cfg.speeds.length ? this.cfg.speeds : [null];
    return speeds.map((s) => {
      const row = new Float64Array(nFwd + 1);
      let sum = 0;
      for (let d = 0, logFact = 0; d <= nFwd; d++) {
        if (d > 0) logFact += Math.log(d);
        row[d] = s === null ? 1 : Math.exp(d * Math.log(s * dt) - s * dt - logFact);
        sum += row[d];
      }
      return row.map((x) => x / sum);
    });
  }

  _advance(dt, locked = false) {
    const { cfg, post, floor } = this;
    const n = post.length;
    const tele = locked ? cfg.pTeleportLocked : cfg.pTeleport;
    if (dt <= 0) {
      for (let w = 0; w < n; w++) post[w] = (1 - tele) * post[w] + tele * floor[w];
      return;
    }
    const nFwd = Math.ceil(cfg.maxSpeed * dt);
    const kernels = this._speedKernels(dt, nFwd);
    const shifted = Array.from({ length: nFwd + 1 }, () => new Float64Array(n));
    const back = new Float64Array(n);
    for (let w = 0; w < n; w++) {
      const p = post[w];
      if (!p) continue;
      for (let d = 0; d <= nFwd; d++) shifted[d][Math.min(w + d, this.last[w])] += p;
      for (let d = 1; d <= cfg.backWords; d++) back[Math.max(w - d, this.first[w])] += p;
    }
    // Forward prediction under each speed (kept for the tempo update), and mixed by the tempo weights.
    const bySpeed = kernels.map((k) => {
      const f = new Float64Array(n);
      for (let d = 0; d <= nFwd; d++) if (k[d]) for (let w = 0; w < n; w++) f[w] += k[d] * shifted[d][w];
      return f;
    });
    this.fwdBySpeed = cfg.tempoMemory > 0 ? bySpeed : null;
    let sum = 0;
    for (let w = 0; w < n; w++) {
      let fwd = 0;
      for (let s = 0; s < bySpeed.length; s++) fwd += this.tempo[s] * bySpeed[s][w];
      post[w] = (1 - cfg.pBack) * fwd + cfg.pBack * (back[w] / cfg.backWords);
      sum += post[w];
    }
    for (let w = 0; w < n; w++) post[w] = (1 - tele) * (post[w] / sum) + tele * floor[w];
  }

  _locked() {
    return Math.max(...this._duaMass()) >= this.cfg.lockConfidence;
  }

  _duaMass() {
    const mass = new Float64Array(this.ix.duaWordSpan.length);
    for (let w = 0; w < this.post.length; w++) mass[this.ix.wordDua[w]] += (1 - this.null) * this.post[w];
    return mass;
  }

  // With lead > 0: show the belief predicted `lead` s past the window's end, and hold it through silence.
  // `quiet`: seconds the reciter has been silent at the window's end (see tracker.py update).
  update(transcript, dt, lead = 0, quiet = 0) {
    const costs = transcript ? this.ix.wordCosts(transcript) : null;
    const moving = quiet > this.cfg.stillMotionAfter ? dt - quiet : dt;
    this._advance(costs ? Math.max(0, moving) : 0, this._locked());
    if (costs) {
      const kappa = this._locked() ? this.cfg.kappa : this.cfg.kappaSearch;
      let min = Infinity;
      for (const c of costs) if (c < min) min = c;
      const lik = new Float64Array(costs.length);
      for (let w = 0; w < costs.length; w++) lik[w] = Math.exp(-kappa * (costs[w] - min));
      if (this.fwdBySpeed) {
        // Which speed predicted this evidence best? Forget a little first.
        const wts = this.fwdBySpeed.map((f, s) => {
          let ev = 0;
          for (let w = 0; w < lik.length; w++) ev += f[w] * lik[w];
          return this.tempo[s] ** this.cfg.tempoMemory * ev;
        });
        const tot = wts.reduce((a, b) => a + b, 0);
        if (tot > 0) wts.forEach((x, s) => (this.tempo[s] = x / tot));
      }
      const { nullRate, nullEnter, nullLeave } = this.cfg;
      const nLetters = encode(transcript).length;
      if (nullRate > 0 && nLetters) {
        const known = (1 - this.null) * (1 - nullEnter) + this.null * nullLeave;
        let fit = 0;
        for (let w = 0; w < lik.length; w++) fit += this.post[w] * lik[w];
        const none = (1 - known) * Math.exp(-kappa * Math.min(50, Math.max(-50, nullRate * nLetters - min)));
        this.null = none / (known * fit + none);
      }
      let sum = 0;
      for (let w = 0; w < costs.length; w++) {
        this.post[w] *= lik[w];
        sum += this.post[w];
      }
      for (let w = 0; w < costs.length; w++) this.post[w] /= sum;
    }
    if (lead <= 0) return this.forwardOnly(this.position());
    if ((!costs || quiet > this.cfg.stillAfter) && this.shown) {
      const now = this.position();
      if (quiet > this.cfg.retreatAfter) this._retreat(now);
      return { ...this.shown, candidates: now.candidates };
    }
    let led = this.lookahead(lead);
    if (this.cfg.leadWithinLine || quiet >= this.cfg.leadCrossQuiet) {
      // Only evidence starts a new line: stay at the end of the evidence's line.
      const now = this.position();
      if (now.dua !== null && (led.dua !== now.dua || led.segment !== now.segment)) led = this._lineEnd(now);
    }
    this.shown = this.forwardOnly(led);
    return this.shown;
  }

  _lineEnd(p) {
    const last = this.nextLine[p.word] - 1;
    return { ...p, word: last, token: this.ix.words[last].token, atLineEnd: true };
  }

  // The display ran into a later line than the evidence: back to the end of the evidence's line.
  _retreat(now) {
    const shown = this.shown;
    if (now.dua === null || now.word == null || shown.word == null || now.dua !== shown.dua
        || shown.word <= now.word || this.lineFirst[shown.word] === this.lineFirst[now.word]) return;
    this.shown = this._lineEnd(now);
    this.lastWord = this.shown.word;
  }

  // Current pace estimate in words/s (the tempo-weighted speed prior).
  get speed() {
    const { speeds, maxSpeed } = this.cfg;
    return speeds.length ? speeds.reduce((a, v, s) => a + v * this.tempo[s], 0) : maxSpeed / 2;
  }

  // Where the reciter probably is `seconds` from now; the belief is unchanged.
  lookahead(seconds) {
    const { post, fwdBySpeed } = this;
    this.post = Float64Array.from(post);
    this._advance(seconds, this._locked());
    const p = this.position();
    Object.assign(this, { post, fwdBySpeed });
    return p;
  }

  // Within a line, the word highlight only moves forward (see tracker.py).
  forwardOnly(p) {
    const ix = this.ix;
    const last = this.lastWord;
    if (last !== null && p.word != null && ix.wordSegment[last] === p.segment
        && ix.wordDua[last] === ix.wordDua[p.word] && p.word < last) {
      p.word = last;
      p.token = ix.words[last].token;
      p.atLineEnd = last + 1 >= ix.nWords || ix.wordSegment[last + 1] !== p.segment;
    } else if (last !== null && p.word != null && this.cfg.enterLineAtStart
        && this.lineFirst[p.word] === this.nextLine[last] && ix.wordDua[p.word] === ix.wordDua[last]) {
      // Moved on to the next line: start at its first word (see tracker.py).
      p.word = this.lineFirst[p.word];
      p.token = ix.words[p.word].token;
      p.atLineEnd = p.word + 1 >= ix.nWords || this.lineFirst[p.word + 1] !== p.word;
    }
    this.lastWord = p.word ?? null;
    return p;
  }

  _likeliest(d) {
    const [lo, hi] = this.ix.duaWordSpan[d];
    let w = lo;
    for (let i = lo; i < hi; i++) if (this.post[i] > this.post[w]) w = i;
    return w;
  }

  // Du'as a and b are both, most likely, inside one identical passage (tracker.py).
  _samePassage(a, b) {
    const { sameTextWords: n, sameTextAhead: ahead } = this.cfg;
    const [loA, hiA] = this.ix.duaWordSpan[a];
    const [loB, hiB] = this.ix.duaWordSpan[b];
    const wa = this._likeliest(a);
    const wb = this._likeliest(b);
    const t = (i) => this.ix.words[i].text;
    let fwd = 0;
    while (fwd < n && wa + fwd < hiA && wb + fwd < hiB && t(wa + fwd) === t(wb + fwd)) fwd++;
    let back = 0;
    while (back < n && wa - back - 1 >= loA && wb - back - 1 >= loB && t(wa - back - 1) === t(wb - back - 1)) back++;
    const atEnd = wa + fwd === hiA || wb + fwd === hiB; // one of them finishes here
    return (fwd >= ahead || atEnd) && fwd + back >= n;
  }

  // The top du'a, and any of the next likeliest in the same passage as it: a
  // shared passage counts as one du'a for the threshold (tracker.py, same_text_words).
  _sameText(mass, order) {
    const group = [order[0]];
    if (!(this.cfg.sameTextWords > 0)) return group;
    for (const o of order.slice(1, 8)) {
      if (mass[o] < 0.01) break;
      if (this._samePassage(group[0], o)) group.push(o);
    }
    return group;
  }

  position() {
    const { ix, post } = this;
    const mass = this._duaMass();
    const order = Array.from(mass.keys()).sort((a, b) => mass[b] - mass[a]);
    const candidates = order.slice(0, 3).map((i) => [ix.duaIds[i], mass[i]]);
    const group = this._sameText(mass, order);
    const conf = group.reduce((n, g) => n + mass[g], 0);
    if (conf < this.cfg.minDuaConfidence) return { dua: null, duaConfidence: conf, candidates };
    // Shown: the du'a on screen if it was found before the shared passage began,
    // else the one the passage sits nearest the start of (tracker.py).
    let d;
    if (group.length === 1) {
      d = group[0];
      this.foundAlone = true;
    } else if (group.includes(this.reported) && this.foundAlone) {
      d = this.reported;
    } else {
      const offset = (g) => this._likeliest(g) - ix.duaWordSpan[g][0];
      d = group.reduce((a, b) => (offset(b) < offset(a) ? b : a));
      this.foundAlone = false;
    }
    this.reported = d;
    const [lo, hi] = ix.duaWordSpan[d];
    const segMass = new Map();
    for (let w = lo; w < hi; w++) segMass.set(ix.wordSegment[w], (segMass.get(ix.wordSegment[w]) ?? 0) + (1 - this.null) * post[w]);
    let seg = null;
    let best = -1;
    for (const [s, m] of segMass) if (m > best || (m === best && s < seg)) [seg, best] = [s, m];
    // Weighted median of the posterior within the shown line (see tracker.py).
    const inSeg = [];
    for (let w = lo; w < hi; w++) if (ix.wordSegment[w] === seg) inSeg.push(w);
    let total = 0;
    for (const w of inSeg) total += post[w];
    let cum = 0;
    let word = inSeg[inSeg.length - 1];
    for (const w of inSeg) {
      cum += post[w];
      if (cum >= total / 2) { word = w; break; }
    }
    const atLineEnd = word + 1 >= hi || ix.wordSegment[word + 1] !== ix.wordSegment[word];
    return {
      dua: ix.duaIds[d], duaConfidence: conf, segment: seg, segmentConfidence: best / mass[d],
      word, token: ix.words[word].token, atLineEnd, candidates,
      sameAs: group.filter((g) => g !== d).map((g) => ix.duaIds[g]),
    };
  }

  prompt(nWords = 12) {
    const p = this.position();
    if (p.dua === null || p.duaConfidence < 0.9) return null;
    return this.ix.textBefore(p.word, nWords);
  }
}
