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

export function encode(text) {
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
  pBack: 0.1, backWords: 8, pTeleport: 0.01, pTeleportLocked: 0.01, pLineJump: 0.02, startWeight: 0.3, startWords: 12, popularity: 0.5, minDuaConfidence: 0.7, sameTextWords: 12, sameTextAhead: 3,
  // Shared passages spelled differently count as one, for the display and the lock
  // (TrackerConfig.same_text_spelling / lock_on_passage in tracker.py).
  sameTextSpelling: true, lockOnPassage: true,
  displayLead: 0.25, // seconds shown ahead of the measured delay (see tracker.py)
  enterLineAtStart: true, // moving on to the next line starts at its first word
  // Pauses (TrackerConfig.still_after ... retreat_after in tracker.py): `quiet` is how
  // long the reciter has been silent at the window's end (asr-worker.js quietAtEnd).
  stillAfter: 0.3, leadWithinLine: false, stillMotionAfter: 0.3, leadCrossQuiet: 0.1, retreatAfter: 1.0,
  // "None of these": recitations not in the corpus (TrackerConfig.null_rate in tracker.py).
  nullRate: 0.45, nullEnter: 0.002, nullLeave: 0.05,
  // From ordinary voices reading short lines (TrackerConfig.lead_cross_words ... keep_dua_confidence
  // in tracker.py): the lead crosses into the next line only near the current one's end; a tapped
  // line stays until the evidence passes it; a step back waits for a second update; once locked,
  // "not in the corpus" needs worse windows; a du'a on screen stays down to a lower bar.
  leadCrossWords: Infinity, seekPinsLine: true, backConfirm: 2, nullRateLocked: 0.55, keepDuaConfidence: null,
  // Shared passages (TrackerConfig.switch_confirm / switch_hold_mass / switch_sure).
  switchConfirm: 0, switchHoldMass: 0.05, switchSure: null,
  // Stops (TrackerConfig.retreat_in_line / still_catch_up in tracker.py): after a second of
  // silence a highlight ahead of the evidence in its own line steps back too; while silent the
  // display may still move forward to the evidence.
  retreatInLine: false, stillCatchUp: false,
  // The belief moves only for time not spent in pauses (TrackerConfig.pause_motion).
  pauseMotion: true,
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
    // Prior over du'as (TrackerConfig.popularity): proportional to (recordings + 1) ** popularity.
    let pd = null;
    if (this.cfg.popularity) {
      pd = index.duas.map((d) => ((d.rec || 0) + 1) ** this.cfg.popularity);
      const sum = pd.reduce((a, b) => a + b, 0);
      pd = pd.map((x) => x / sum);
    }
    index.duaWordSpan.forEach(([lo, hi], d) => {
      const k = Math.min(this.cfg.startWords, hi - lo);
      for (let w = lo; w < hi; w++) {
        const anywhere = pd ? pd[d] / (hi - lo) : 1 / (nDuas * (hi - lo));
        const start = w - lo < k ? (pd ? pd[d] / k : 1 / (nDuas * k)) : 0;
        this.floor[w] = this.cfg.startWeight * start + (1 - this.cfg.startWeight) * anywhere;
        this.first[w] = lo;
        this.last[w] = hi - 1;
      }
    });
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
    // Lines per du'a, for a jump within the du'a (pLineJump: to the first word of any of its lines).
    this.linesInDua = new Float64Array(nDuas);
    for (let w = 0; w < n; w++) if (this.lineFirst[w] === w) this.linesInDua[index.wordDua[w]] += 1;
    Object.assign(this, passageKeys(index.words.map((w) => w.text), index.duaWordSpan));
    this.reset();
  }

  reset() {
    this.post = Float64Array.from(this.floor);
    this.lastWord = null;
    this.shown = null;
    this.pin = null; // last word of a tapped line (seekPinsLine)
    this.backs = 0; // updates in a row that asked to step back a line (backConfirm)
    this.null = this.cfg.nullRate > 0 ? 0.5 : 0; // P(not in the corpus)
    this.reported = null; // the du'a last reported (index)
    this.foundAlone = false; // ...and it was told apart from every other text
    this.nUpdates = 0;
    this.aloneCand = null; // a du'a standing alone on top, not shown yet...
    this.aloneFirst = 0; // ...since this update...
    this.aloneLast = 0; // ...and last seen alone in this one (position() runs more than once an update)
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
    for (let w = 0; w < n; w++) post[w] /= sum;
    if (cfg.pLineJump > 0) {
      // A jump within the du'a, to the first word of any of its lines (TrackerConfig.p_line_jump).
      const pj = Math.min(1, cfg.pLineJump * dt);
      const dua = this.ix.wordDua;
      const mass = new Float64Array(this.linesInDua.length);
      for (let w = 0; w < n; w++) mass[dua[w]] += post[w];
      for (let w = 0; w < n; w++) {
        post[w] = (1 - pj) * post[w] + pj * (this.lineFirst[w] === w ? mass[dua[w]] / this.linesInDua[dua[w]] : 0);
      }
    }
    for (let w = 0; w < n; w++) post[w] = (1 - tele) * post[w] + tele * floor[w];
  }

  // Posterior mass of the line `w` is in (the word follower's jump rule, follower.js jumpMass).
  lineMass(w) {
    let m = 0;
    for (let k = this.lineFirst[w]; k < this.nextLine[w]; k++) m += this.post[k];
    return (1 - this.null) * m;
  }

  _locked() {
    const mass = this._duaMass();
    const top = Math.max(...mass);
    if (top >= this.cfg.lockConfidence || !this.cfg.lockOnPassage) return top >= this.cfg.lockConfidence;
    const order = Array.from(mass.keys()).sort((a, b) => mass[b] - mass[a]);
    return this._sameText(mass, order).reduce((n, g) => n + mass[g], 0) >= this.cfg.lockConfidence;
  }

  _duaMass() {
    const mass = new Float64Array(this.ix.duaWordSpan.length);
    for (let w = 0; w < this.post.length; w++) mass[this.ix.wordDua[w]] += (1 - this.null) * this.post[w];
    return mass;
  }

  // With lead > 0: show the belief predicted `lead` s past the window's end, and hold it through silence.
  // `quiet`: seconds the reciter has been silent at the window's end; `quietNow`: the same when the
  // update is shown (gate.js LiveQuiet; null = not known); `paused`: seconds of the last dt spent in
  // pauses (gate.js stopMeasures; null = only `quiet` counts). See tracker.py update.
  update(transcript, dt, lead = 0, quiet = 0, quietNow = null, paused = null) {
    this.nUpdates += 1;
    const costs = transcript ? this.ix.wordCosts(transcript) : null;
    const moving = paused != null && this.cfg.pauseMotion ? dt - paused
      : quiet > this.cfg.stillMotionAfter ? dt - quiet : dt;
    this._advance(costs ? Math.max(0, moving) : 0, this._locked());
    if (costs) {
      const locked = this._locked();
      const kappa = locked ? this.cfg.kappa : this.cfg.kappaSearch;
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
      const { nullRate, nullEnter, nullLeave, nullRateLocked } = this.cfg;
      const nLetters = encode(transcript).length;
      if (nullRate > 0 && nLetters) {
        const rate = locked && nullRateLocked != null ? nullRateLocked : nullRate;
        const known = (1 - this.null) * (1 - nullEnter) + this.null * nullLeave;
        let fit = 0;
        for (let w = 0; w < lik.length; w++) fit += this.post[w] * lik[w];
        const none = (1 - known) * Math.exp(-kappa * Math.min(50, Math.max(-50, rate * nLetters - min)));
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
    const cfg = this.cfg;
    // How long they have been silent when this is shown: the page's own ear if it has one,
    // else the window's end; a window that ended in a pause they have since recited on
    // from is no reason to hold (tracker.py).
    const heard = quietNow != null;
    const silent = heard ? quietNow : quiet;
    const resumed = heard && quietNow <= cfg.stillAfter;
    if ((!costs || (quiet > cfg.stillAfter && !resumed)) && this.shown) {
      const now = this.position();
      const shown = this.shown;
      if (silent > cfg.retreatAfter) this._retreat(now);
      if (cfg.stillCatchUp && costs && now.word != null && shown.word != null && now.dua === shown.dua
          && now.word > this.shown.word) {
        this.backs = 0;
        this.shown = this.forwardOnly({ ...now });
      }
      return { ...this.shown, candidates: now.candidates };
    }
    let retreat = false;
    if (heard && quietNow > cfg.stillAfter) {
      // They stopped after the window's end: the lead covers only the time they were reciting.
      lead = Math.max(0, lead - cfg.displayLead - quietNow);
      retreat = quietNow > cfg.retreatAfter;
    }
    let led = this.lookahead(lead);
    // Only while they make sound may the lead cross into the next line: under leadCrossQuiet at the
    // window's end; from the page's own ear, a gap between words is not a stop (tracker.py).
    const crossBar = heard ? Math.max(cfg.leadCrossQuiet, cfg.stillAfter) : cfg.leadCrossQuiet;
    if (cfg.leadWithinLine || silent >= crossBar || this.pin !== null || cfg.leadCrossWords !== Infinity) {
      const now = this.position();
      if (this.pin !== null && (now.word == null || this.ix.wordDua[now.word] !== this.ix.wordDua[this.pin]
          || now.word > this.pin)) this.pin = null; // the evidence has left the tapped line
      const hold = cfg.leadWithinLine || silent >= crossBar || this.pin !== null
        || (now.word != null && this.nextLine[now.word] - now.word > cfg.leadCrossWords);
      // Only evidence starts a new line: stay at the end of the evidence's line.
      if (hold && now.dua !== null && (led.dua !== now.dua || led.segment !== now.segment)) led = this._lineEnd(now);
    }
    const shown = this.shown;
    const backALine = !!shown && shown.word != null && led.word != null && led.dua === shown.dua
      && this.lineFirst[led.word] < this.lineFirst[shown.word];
    if (retreat && backALine) led = this._lineEnd(led); // the retreat: they stopped at the end of the evidence's line
    if (cfg.backConfirm > 1 && !retreat && backALine) {
      this.backs += 1;
      if (this.backs < cfg.backConfirm) return { ...shown, candidates: led.candidates }; // not yet: hold the line
    }
    this.backs = 0;
    if (retreat && cfg.retreatInLine && this.lastWord !== null && led.word != null) {
      this.lastWord = Math.min(this.lastWord, led.word); // no forward-only hold past the evidence
    }
    this.shown = this.forwardOnly(led);
    return this.shown;
  }

  _lineEnd(p) {
    const last = this.nextLine[p.word] - 1;
    return { ...p, word: last, token: this.ix.words[last].token, atLineEnd: true };
  }

  // The display ran into a later line than the evidence: back to the end of the evidence's line
  // (retreatInLine: or ahead of it in the same line: back to the evidence's word).
  _retreat(now) {
    const shown = this.shown;
    if (now.dua === null || now.word == null || shown.word == null || now.dua !== shown.dua
        || shown.word <= now.word) return;
    if (this.lineFirst[shown.word] === this.lineFirst[now.word]) {
      if (this.cfg.retreatInLine) {
        this.shown = { ...now };
        this.lastWord = now.word;
      }
      return;
    }
    this.shown = this._lineEnd(now);
    this.lastWord = this.shown.word;
  }

  // The listener says where they are ("I'm here": a tap on line `segment` of du'a
  // `duaId`). The belief restarts over that line, in the corpus; the display starts at
  // its first word (tracker.py seek).
  seek(duaId, segment) {
    const ix = this.ix;
    const d = ix.duaIds.indexOf(duaId);
    if (d < 0) throw new Error(`${duaId} is not in the corpus`);
    const [lo, hi] = ix.duaWordSpan[d];
    const words = [];
    for (let w = lo; w < hi; w++) if (ix.wordSegment[w] === segment) words.push(w);
    if (!words.length) throw new Error(`${duaId} has no line ${segment}`);
    this.post = new Float64Array(ix.nWords);
    for (const w of words) this.post[w] = 1 / words.length;
    this.null = 0;
    this.fwdBySpeed = null;
    this.reported = d;
    this.foundAlone = true;
    this.lastWord = words[0];
    this.pin = this.cfg.seekPinsLine ? this.nextLine[words[0]] - 1 : null;
    this.backs = 0;
    const p = this.position();
    this.shown = { ...p, word: words[0], token: ix.words[words[0]].token, atLineEnd: words.length === 1 };
    return this.shown;
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
    const wa = this._likeliest(a);
    const wb = this._likeliest(b);
    if (!this.cfg.sameTextSpelling) {
      const t = (i) => this.ix.words[i].text;
      return passageRun(t, (x, y) => x === y, wa, wb, this.ix.duaWordSpan[a], this.ix.duaWordSpan[b], n, ahead);
    }
    // Spelled differently, the beliefs can peak a word apart: line them up within a word either way.
    const t = (i) => this.keys[i];
    const [ka, kb] = [this.keyOf[wa], this.keyOf[wb]];
    return [0, -1, 1].some((o) => passageRun(t, sameWord, ka, kb + o, this.keySpan[a], this.keySpan[b], n, ahead));
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
    // Already on screen: keep it down to the lower bar (keepDuaConfidence).
    const keep = this.cfg.keepDuaConfidence != null && group.includes(this.reported);
    if (conf < (keep ? this.cfg.keepDuaConfidence : this.cfg.minDuaConfidence)) return { dua: null, duaConfidence: conf, candidates };
    // Shown: the du'a on screen if it was found before the shared passage began,
    // else the one the passage sits nearest the start of (tracker.py).
    let d;
    const cfg = this.cfg;
    if (group.length === 1) {
      d = group[0];
      const held = cfg.switchConfirm > 0 && this.reported != null && d !== this.reported
        && mass[this.reported] >= cfg.switchHoldMass && (cfg.switchSure == null || conf < cfg.switchSure);
      if (held) {
        if (this.aloneCand !== d || this.nUpdates - this.aloneLast > 1) {
          this.aloneCand = d;
          this.aloneFirst = this.nUpdates;
        }
        this.aloneLast = this.nUpdates;
        if (this.nUpdates - this.aloneFirst + 1 < cfg.switchConfirm) d = this.reported;
        else {
          this.aloneCand = null;
          this.foundAlone = true;
        }
      } else {
        this.aloneCand = null;
        this.foundAlone = true;
      }
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
      sameAs: group.includes(d) ? group.filter((g) => g !== d).map((g) => ix.duaIds[g]) : [],
    };
  }

  prompt(nWords = 12) {
    const p = this.position();
    if (p.dua === null || p.duaConfidence < 0.9) return null;
    return this.ix.textBefore(p.word, nWords);
  }
}

// Words as _samePassage compares them (tracker.py _passage_keys): a lone و joined to
// the word after it, a lone ء to the word before.
function passageKeys(text, spans) {
  const keys = [];
  const keyOf = new Int32Array(text.length);
  const keySpan = [];
  for (const [lo, hi] of spans) {
    const k0 = keys.length;
    for (let w = lo; w < hi; w++) {
      if (text[w] === "ء" && keys.length > k0) {
        keys[keys.length - 1] += text[w];
      } else if (text[w] === "و" && w + 1 < hi) {
        keys.push(text[w] + text[w + 1]);
        keyOf[w] = keys.length - 1;
        w++;
      } else {
        keys.push(text[w]);
      }
      keyOf[w] = keys.length - 1;
    }
    keySpan.push([k0, keys.length]);
  }
  return { keys, keyOf, keySpan };
}

// Word wa of one text and wb of another sit in a run of at least n matching words,
// `ahead` of them still to come unless one text ends there (tracker.py _passage_run).
function passageRun(t, same, wa, wb, [loA, hiA], [loB, hiB], n, ahead) {
  if (wb < loB || wb >= hiB) return false;
  let fwd = 0;
  while (fwd < n && wa + fwd < hiA && wb + fwd < hiB && same(t(wa + fwd), t(wb + fwd))) fwd++;
  let back = 0;
  while (back < n && wa - back - 1 >= loA && wb - back - 1 >= loB && same(t(wa - back - 1), t(wb - back - 1))) back++;
  const atEnd = wa + fwd === hiA || wb + fwd === hiB; // one of them finishes here
  return (fwd >= ahead || atEnd) && fwd + back >= n;
}

// Equal, or both 4+ letters and one edit apart (tracker.py _same_word).
function sameWord(a, b) {
  if (a === b) return true;
  if (Math.min(a.length, b.length) < 4 || Math.abs(a.length - b.length) > 1) return false;
  if (a.length === b.length) {
    let diff = 0;
    for (let i = 0; i < a.length; i++) diff += a[i] !== b[i];
    return diff === 1;
  }
  if (a.length > b.length) [a, b] = [b, a];
  let i = 0;
  while (i < a.length && a[i] === b[i]) i++;
  return a.slice(i) === b.slice(i + 1);
}
