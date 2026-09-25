// In-browser port of src/dua_recognition/{text,align,tracker}.py.
// Same normalization, same semi-global alignment, same HMM, same defaults, so
// the phone version follows a recitation exactly like the server does.
// tests/test_web_parity.py checks this against the Python on real transcripts.

const TASHKEEL = /[ً-ٰٟۖ-ۭـ]/g;
const NON_ARABIC = /[^ء-ي\s]/g;
const FOLD = { "آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا", "ى": "ي", "ی": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه", "ک": "ك" };

export function normalize(text) {
  let t = text.normalize("NFC").replace(TASHKEEL, "");
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

// D[m][e] of the edit-distance DP with a free start in r (see align.py).
export function semiglobalEndCosts(h, r) {
  const n = r.length;
  let prev = new Int32Array(n + 1);
  let cur = new Int32Array(n + 1);
  for (let i = 1; i <= h.length; i++) {
    const c = h[i - 1];
    cur[0] = i;
    let run = i; // running min of cur[k] - k
    for (let e = 1; e <= n; e++) {
      const diag = prev[e - 1] + (r[e - 1] !== c ? 1 : 0);
      const up = prev[e] + 1;
      let v = diag < up ? diag : up;
      const left = run + e;
      if (left < v) v = left;
      cur[e] = v;
      if (v - e < run) run = v - e;
    }
    [prev, cur] = [cur, prev];
  }
  return prev.subarray(1);
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
  pBack: 0.1, backWords: 8, pTeleport: 0.01, pTeleportLocked: 0.01, startWeight: 0.3, startWords: 12, minDuaConfidence: 0.7,
};

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
    this.reset();
  }

  reset() {
    this.post = Float64Array.from(this.floor);
    this.lastWord = null;
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
    for (let w = 0; w < this.post.length; w++) mass[this.ix.wordDua[w]] += this.post[w];
    return mass;
  }

  update(transcript, dt) {
    const costs = transcript ? this.ix.wordCosts(transcript) : null;
    this._advance(costs ? dt : 0, this._locked());
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
      let sum = 0;
      for (let w = 0; w < costs.length; w++) {
        this.post[w] *= lik[w];
        sum += this.post[w];
      }
      for (let w = 0; w < costs.length; w++) this.post[w] /= sum;
    }
    return this.forwardOnly(this.position());
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
    }
    this.lastWord = p.word ?? null;
    return p;
  }

  position() {
    const { ix, post } = this;
    const mass = this._duaMass();
    const order = Array.from(mass.keys()).sort((a, b) => mass[b] - mass[a]);
    const d = order[0];
    const conf = mass[d];
    const candidates = order.slice(0, 3).map((i) => [ix.duaIds[i], mass[i]]);
    if (conf < this.cfg.minDuaConfidence) return { dua: null, duaConfidence: conf, candidates };
    const [lo, hi] = ix.duaWordSpan[d];
    const segMass = new Map();
    for (let w = lo; w < hi; w++) segMass.set(ix.wordSegment[w], (segMass.get(ix.wordSegment[w]) ?? 0) + post[w]);
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
      dua: ix.duaIds[d], duaConfidence: conf, segment: seg, segmentConfidence: best / conf,
      word, token: ix.words[word].token, atLineEnd, candidates,
    };
  }

  prompt(nWords = 12) {
    const p = this.position();
    if (p.dua === null || p.duaConfidence < 0.9) return null;
    return this.ix.textBefore(p.word, nWords);
  }
}
