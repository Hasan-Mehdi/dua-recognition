// In-browser port of src/dua_recognition/stream_follower.py: follow the reciter through the du'a
// by decoding the CTC frames (web/ctc-worker.js) as one stream.
//
// A belief over every position in the du'a, updated frame by frame (a CTC forward pass over the
// du'a's letters), with reading itself as the transition model: the text in order is free;
// starting the line again, going back one to three lines, skipping ahead, or a jump anywhere cost
// a little; talk (a filler that explains any frame at a small cost) and the salawat (said between
// lines) are their own states, left again for where the reader was. A pause is a blank. A refrain
// keeps every repetition alive until the words that differ are heard.
//
// From each CTC window (2 s, every 0.1 s) the frames ending `lookahead` before its end are
// committed; the newest `lookahead` seconds are decoded on a copy, for the display. The tracker
// (Whisper) gives the du'a, the starting belief and gentle evidence at each of its updates.
// tests/test_stream_parity.py checks this against the Python, step for step.
import { endScores, squeezeBlanks } from "./follower.js";
import { encode, normalize } from "./tracker.js";

const NEG = -1e30;

export const STREAM_DEFAULTS = {
  temp: 1.0,
  lookahead: 0.2,
  hop: 0.1,
  frameS: 0.02,
  cRestart: -3.0,
  cBack: [-5.0, -6.0, -7.0],
  cSkip: [-8.0, -9.0, -10.0],
  cMid: -3.0,
  cRest: -3.0,
  cFar: -13.0,
  cFillIn: -8.0,
  fillCost: 1.0,
  cFillWord: -2.0,
  cFillNext: -1.0,
  interjection: "اللهم صل على محمد وآل محمد|وعجل فرجهم",
  cIntIn: -4.0,
  cIntNext: -1.0,
  cIntWord: -1.0,
  quietS: 0.3,
  quietPen: 4.0,
  blankBias: 1.0, // nats off the blank column: faint letters (ordinary voices) count for more
  trackerWeight: 0.3,
  trackerFloor: 0.02,
  showP: 0.35,
  lineP: 0.5,
  lineSteps: 2,
  nextP: 0.6,
  nextSteps: 2,
  // Into the next line on evidence (nextMargin 0 = off): see StreamConfig.next_margin.
  nextMargin: 3,
  nextHold: 0.3,
  nextSlack: 1.0,
  gateWords: 2,
  gateTentative: true,
  lapseS: 20.0,
  // ...but while the reader is making sound (stop detector under lapseQuiet s), only lapseVoice s:
  // recitation the tracker can't place is likely a text it doesn't know, not a lull.
  lapseVoice: 4.0,
  lapseQuiet: 1.0,
  // ...counting only the steps the frames themselves put at least this share off the text (talk,
  // salawat, something else; 0 = every step with voice). A reader the CTC model still hears in
  // the du'a while Whisper's transcripts fail (a drawn-out last line, session 5o18) keeps the
  // du'a until lapseS.
  lapseOff: 0.0,
  // ...and not the steps the frames put at least this share in the salawat chain (2 = every step): a long
  // salawat between lines lost the tracker's du'a and, after lapseVoice, the reader's place.
  lapseIntj: 2.0,
  // A du'a change counts once the tracker has held the new du'a this long (0 = at once): texts that
  // share a passage flip the tracker's du'a for a moment.
  switchS: 2.0,
  // ...and none of it counts while the tracker's last sharedWords words in the other du'a are the letters
  // ending at a word within sharedNear words of the shown one (0 = off): the two read the same passage here.
  sharedWords: 0,
  sharedNear: 8,
  sharedAfter: 10.0, // ...once the shown du'a has been on screen this long (s)
  // The beam (0 = off): only lines holding beam of the belief, beamMargin lines either side, the shown
  // word's line and the tracker's proposals (proposeMass, +-1) are updated (stream_follower.py _prune).
  beam: 1e-6,
  beamMargin: 5,
  proposeMass: 0.05,
  // The frames propose too (stream_follower.py _frame_proposals): every ctcEvery steps the latest
  // ctcWindow s are scored against the whole du'a; up to ctcLines lines within ctcMargin nats of the
  // best join the beam. Only considered: the forward pass decides.
  ctcEvery: 5,
  ctcWindow: 1.5,
  ctcLines: 3,
  ctcMargin: 4.0,
  // ...and when one line beats every other but its neighbours by ctcSure nats and isn't the shown line or
  // next to it, ctcPushP of the belief moves to just after its best word (0 = off): after a far jump the
  // frames name the new line within a second, but the belief never got there before the tracker did.
  ctcPushP: 0.0,
  ctcSure: 4.0,
  // The tracker's push (stream_follower.py): the shown word unchanged stuckS s while the reader makes
  // sound, the tracker pushWords or more words ahead (or on a later line): pushP of the belief moves
  // to just before the tracker's word. In echo the frames blur and the belief stalls; Whisper still knows.
  stuckS: 1.5,
  pushWords: 3,
  pushP: 0.0, // off: more false jumps when readers go back, no gain in echo
  // Copies (stream_follower.py): lines with the same words are one line for the display, which weighs
  // each word by the belief summed over its copies and shows the copy nearest the word on screen (the
  // next-line gate still weighs each line by its own belief).
  copies: false,
  copySplit: 0.5, // ...the nearest copy only while no copy holds this share of their summed belief
};

const dd0 = (f) => f._dua(f.dua);

function lae(a, b) {
  if (a < b) {
    const x = a;
    a = b;
    b = x;
  }
  if (b <= NEG / 2) return a;
  return a + Math.log1p(Math.exp(b - a));
}

export class StreamFollower {
  // index: tracker.js CorpusIndex
  constructor(index, config = {}) {
    this.ix = index;
    this.cfg = { ...STREAM_DEFAULTS, ...config };
    const n = index.nWords;
    this.firstLetter = new Int32Array(n + 1);
    for (let w = 0; w < n; w++) this.firstLetter[w + 1] = index.wordEnds[w] + 1;
    this.duas = new Map();
    this.chain = this._chain();
    this.reset();
  }

  reset() {
    this.dua = null;
    this.st = null;
    this.word = null;
    this.cand = null;
    this.candN = 0;
    this.gateS = 0; // audio (s) a move into the next line has leant on the frames, steps in a row
    this.tStep = null; // the last step's time
    this.fStream = null;
    this.lastAnchorT = null;
    this.lapseSince = null;
    this.lapseVoice = 0;
    this.off = 1; // the last step's share off the text (posterior)
    this.intj = 0; // ...and in the salawat chain (with lapseIntj < 1)
    this.duaT = null; // when the decoder started on its du'a
    this.switchTo = null;
    this.switchSince = null;
    this.active = null;
    this.proposed = null;
    this.ctcProp = null;
    this.steps = 0;
    this.shownSince = null;
    this.shownW = null;
  }

  // The interjection's letters, which differ from the one before, and where it may end.
  _chain() {
    const r = [];
    const ends = [];
    for (const part of this.cfg.interjection ? this.cfg.interjection.split("|") : []) {
      for (const c of encode(normalize(part).replace(/ /g, ""))) r.push(c);
      ends.push(r.length - 1);
    }
    const ri = Int32Array.from(r);
    const diffi = new Uint8Array(ri.length);
    for (let k = 0; k < ri.length; k++) diffi[k] = k === 0 || ri[k] !== ri[k - 1] ? 1 : 0;
    const exiti = new Uint8Array(ri.length);
    for (const e of ends) exiti[e] = 1;
    return { ri, diffi, exiti, K: ri.length };
  }

  _dua(d) {
    if (this.duas.has(d)) return this.duas.get(d);
    const ix = this.ix;
    const [lo, hi] = ix.duaWordSpan[d];
    const la = this.firstLetter[lo];
    const lb = this.firstLetter[hi];
    const J = lb - la;
    const nw = hi - lo;
    const r = new Int32Array(J);
    const wordOf = new Int32Array(J);
    const wordFirst = new Int32Array(nw);
    const wordLast = new Int32Array(nw);
    for (let w = 0; w < nw; w++) {
      wordFirst[w] = this.firstLetter[lo + w] - la;
      wordLast[w] = this.firstLetter[lo + w + 1] - 1 - la;
      for (let j = wordFirst[w]; j <= wordLast[w]; j++) {
        r[j] = ix.letters[la + j];
        wordOf[j] = w;
      }
    }
    const lineOfWord = new Int32Array(nw);
    const firsts = [];
    for (let w = 0; w < nw; w++) {
      const newLine = w === 0 || ix.wordSegment[lo + w] !== ix.wordSegment[lo + w - 1];
      if (newLine) firsts.push(w);
      lineOfWord[w] = firsts.length - 1;
    }
    const nl = firsts.length;
    const lineFirstWord = Int32Array.from(firsts);
    const lineLastWord = new Int32Array(nl);
    for (let k = 0; k < nl; k++) lineLastWord[k] = k + 1 < nl ? lineFirstWord[k + 1] - 1 : nw - 1;
    const diff = new Uint8Array(J);
    for (let j = 0; j < J; j++) diff[j] = j === 0 || r[j] !== r[j - 1] ? 1 : 0;
    const midCost = new Float64Array(nw).fill(this.cfg.cMid);
    for (let k = 0; k < nl; k++) midCost[lineLastWord[k]] = NEG;
    const lineStartLetter = new Int32Array(nl);
    for (let k = 0; k < nl; k++) lineStartLetter[k] = wordFirst[lineFirstWord[k]];
    const logWordsInLine = new Float64Array(nw);
    for (let w = 0; w < nw; w++) {
      const k = lineOfWord[w];
      logWordsInLine[w] = Math.log(lineLastWord[k] - lineFirstWord[k] + 1);
    }
    const lineLo = new Int32Array(nl);
    const lineHi = new Int32Array(nl);
    for (let k = 0; k < nl; k++) {
      lineLo[k] = wordFirst[lineFirstWord[k]];
      lineHi[k] = wordLast[lineLastWord[k]] + 1;
    }
    const startWord = new Int32Array(J).fill(-1);
    for (let w = 0; w < nw; w++) startWord[wordFirst[w]] = w;
    // per word: the same word of the first line with its letters (copies)
    const wordCopy = new Int32Array(nw);
    for (let w = 0; w < nw; w++) wordCopy[w] = w;
    const firstOf = new Map();
    let hasCopies = false;
    for (let k = 0; k < nl; k++) {
      let key = Array.prototype.join.call(r.subarray(lineLo[k], lineHi[k]), ",") + "|";
      for (let w = lineFirstWord[k]; w <= lineLastWord[k]; w++) key += `${wordFirst[w] - lineLo[k]},`;
      if (!firstOf.has(key)) firstOf.set(key, k);
      const f = firstOf.get(key);
      if (f !== k) {
        hasCopies = true;
        for (let w = lineFirstWord[k]; w <= lineLastWord[k]; w++) wordCopy[w] = w - lineFirstWord[k] + lineFirstWord[f];
      }
    }
    const dd = { lo, J, nw, nl, r, wordOf, wordFirst, wordLast, lineOfWord, lineFirstWord, lineLastWord, diff,
      midCost, lineStartLetter, logWordsInLine, lineLo, lineHi, startWord, wordCopy, hasCopies };
    this.duas.set(d, dd);
    return dd;
  }

  _start(d, hmmWord, lineMass) {
    const dd = this._dua(d);
    this.dua = d;
    const { nl, nw, J } = dd;
    const K = this.chain.K;
    const wLine = new Float64Array(nl).fill(this.cfg.trackerFloor);
    if (lineMass) for (let k = 0; k < nl; k++) wLine[k] += lineMass(dd.lo + dd.lineFirstWord[k]);
    else wLine[dd.lineOfWord[hmmWord - dd.lo]] += 1.0;
    let z = 0;
    for (let k = 0; k < nl; k++) z += wLine[k];
    const st = { L: new Float64Array(J).fill(NEG), B: new Float64Array(J).fill(NEG), F: new Float64Array(nl).fill(NEG),
      IL: new Float64Array(nl * K).fill(NEG), IB: new Float64Array(nl * K).fill(NEG) };
    for (let w = 0; w < nw; w++) {
      const k = dd.lineOfWord[w];
      const n = dd.lineLastWord[k] - dd.lineFirstWord[k] + 1;
      st.B[dd.wordLast[w]] = Math.log(wLine[k] / z / n);
    }
    this.st = st;
    this.active = new Uint8Array(nl).fill(1);
    this.proposed = null;
    if (this.cfg.beam > 0) {
      // Start on the tracker's lines only (every line of a long du'a for the first second cost 20-80 ms a
      // step on a desktop): the anchor's line and every line it puts proposeMass on; the rest come in by proposals.
      const on = new Uint8Array(nl);
      const k = dd.lineOfWord[hmmWord - dd.lo];
      const mg = this.cfg.beamMargin;
      for (let i = Math.max(0, k - mg); i <= Math.min(nl - 1, k + mg); i++) on[i] = 1;
      if (lineMass) {
        for (let li = 0; li < nl; li++) {
          if (lineMass(dd.lo + dd.lineFirstWord[li]) < this.cfg.proposeMass) continue;
          for (let i = Math.max(0, li - 1); i <= Math.min(nl - 1, li + 1); i++) on[i] = 1;
        }
      }
      for (let li = 0; li < nl; li++) {
        if (on[li]) continue;
        for (let j = dd.lineLo[li]; j < dd.lineHi[li]; j++) {
          st.L[j] = NEG;
          st.B[j] = NEG;
        }
      }
      this.active = on;
    }
    this.cand = null;
    this.candN = 0;
    this.gateS = 0;
    this.word = hmmWord;
  }

  static _copy(st) {
    return { L: st.L.slice(), B: st.B.slice(), F: st.F.slice(), IL: st.IL.slice(), IB: st.IB.slice() };
  }

  // Frames [from, to) of the window (T x C, row-major) into the state, in place: the active lines
  // only (stream_follower.py _advance_beam; every line when the beam is off).
  _advance(st, lp, C, from, to, quiet) {
    const dd = this._dua(this.dua);
    const cfg = this.cfg;
    const { r, diff, wordLast, lineFirstWord, lineLastWord, midCost, lineStartLetter, logWordsInLine, nw, nl,
      lineLo, lineHi, startWord } = dd;
    const { ri, diffi, exiti, K } = this.chain;
    const { L, B, F, IL, IB } = st;
    const temp = cfg.temp;
    const qpen = quiet && cfg.quietPen > 0 ? cfg.quietPen : 0;
    const cRest = cfg.cMid + cfg.cRest;
    const act = [];
    for (let k = 0; k < nl; k++) if (cfg.beam <= 0 || this.active[k]) act.push(k);
    const nAct = act.length;
    const we = new Float64Array(nw).fill(NEG);
    const midSrc = new Float64Array(nl).fill(NEG);
    const lineSrc = new Float64Array(nl).fill(NEG);
    const anySrc = new Float64Array(nl).fill(NEG);
    const jsrc = new Float64Array(nl).fill(NEG);
    const into = new Float64Array(nl).fill(NEG);
    const Fn = new Float64Array(nl).fill(NEG);
    const X = new Float64Array(nl).fill(NEG);
    const em = new Float64Array(C);
    for (let t = from; t < to; t++) {
      const o = t * C;
      for (let c = 0; c < C; c++) {
        const v = lp[o + c];
        // a broken window (fp16 overflow on silence): no evidence, as stream_follower.py
        em[c] = Number.isFinite(v) ? (c > 0 ? v - qpen : v - cfg.blankBias) / temp : (c > 0 ? 0 : -cfg.blankBias) / temp;
      }
      let bestl = NEG; // the frame's best letter
      for (let c = 1; c < C; c++) if (em[c] > bestl) bestl = em[c];
      const bl = em[0];
      const fe = Math.max(bl, bestl - cfg.fillCost); // the filler: any letter at fillCost, a blank free
      for (let ai = 0; ai < nAct; ai++) {
        const k = act[ai];
        let acc = NEG;
        let acc2 = NEG;
        for (let w = lineFirstWord[k]; w <= lineLastWord[k]; w++) {
          we[w] = lae(L[wordLast[w]], B[wordLast[w]]);
          acc = lae(acc, we[w] + midCost[w]);
          acc2 = lae(acc2, we[w]);
        }
        midSrc[k] = acc;
        anySrc[k] = acc2;
        lineSrc[k] = lae(we[lineLastWord[k]], acc);
        jsrc[k] = lae(lineSrc[k], F[k]); // back from talk, a reader resumes anywhere a finished line could
      }
      let m = NEG;
      for (let ai = 0; ai < nAct; ai++) if (jsrc[act[ai]] > m) m = jsrc[act[ai]];
      let far = NEG;
      if (m > NEG / 2) {
        let s = 0;
        for (let ai = 0; ai < nAct; ai++) s += Math.exp(jsrc[act[ai]] - m);
        far = m + Math.log(s) + cfg.cFar - Math.log(nl);
      }
      for (let ai = 0; ai < nAct; ai++) {
        const k = act[ai];
        let x = NEG;
        for (let q = 0; q < K; q++) if (exiti[q]) x = lae(x, lae(IL[k * K + q], IB[k * K + q]));
        X[k] = x;
      }
      for (let ai = 0; ai < nAct; ai++) {
        const k = act[ai];
        let v = jsrc[k] + cfg.cRestart;
        for (let q = 0; q < 3; q++) {
          let src = k + q + 1;
          if (src < nl) v = lae(v, jsrc[src] + cfg.cBack[q]);
          src = k - (q + 2);
          if (src >= 0) v = lae(v, jsrc[src] + cfg.cSkip[q]);
        }
        if (k >= 1) {
          v = lae(v, midSrc[k - 1] + cRest);
          v = lae(v, F[k - 1] + cfg.cFillNext);
          if (K > 0) v = lae(v, X[k - 1] + cfg.cIntNext);
        }
        into[k] = lae(v, far);
      }
      for (let ai = 0; ai < nAct; ai++) {
        const k = act[ai];
        Fn[k] = lae(F[k], lineSrc[k] + cfg.cFillIn) + fe;
      }
      for (let ai = 0; ai < nAct; ai++) {
        const k = act[ai];
        for (let q = K - 1; q >= 0; q--) {
          const i = k * K + q;
          let e;
          if (q === 0) e = anySrc[k] + cfg.cIntIn;
          else {
            e = IB[i - 1];
            if (diffi[q]) e = lae(e, IL[i - 1]);
          }
          const nb = lae(IL[i], IB[i]) + bl;
          IL[i] = lae(IL[i], e) + em[ri[q]];
          IB[i] = nb;
        }
      }
      let mx = NEG;
      for (let ai = nAct - 1; ai >= 0; ai--) { // lines high to low, letters right to left
        const k = act[ai];
        const fwLine = K > 0 ? lae(F[k] + cfg.cFillWord, X[k] + cfg.cIntWord) : F[k] + cfg.cFillWord;
        for (let j = lineHi[k] - 1; j >= lineLo[k]; j--) {
          let e = NEG; // entry, in the same order as the Python: the line's start, then a word's start
          if (j === lineStartLetter[k]) e = into[k];
          const w = startWord[j];
          if (w >= 0) e = lae(e, fwLine - logWordsInLine[w]);
          let enter = NEG;
          if (j > 0) {
            enter = B[j - 1];
            if (diff[j]) enter = lae(enter, L[j - 1]);
          }
          const nb = lae(L[j], B[j]) + bl;
          L[j] = lae(lae(L[j], enter), e) + em[r[j]];
          B[j] = nb;
          if (L[j] > mx) mx = L[j];
          if (nb > mx) mx = nb;
        }
      }
      for (let ai = 0; ai < nAct; ai++) {
        const k = act[ai];
        F[k] = Fn[k];
        if (Fn[k] > mx) mx = Fn[k];
        for (let q = 0; q < K; q++) {
          if (IL[k * K + q] > mx) mx = IL[k * K + q];
          if (IB[k * K + q] > mx) mx = IB[k * K + q];
        }
      }
      for (let ai = 0; ai < nAct; ai++) {
        const k = act[ai];
        for (let j = lineLo[k]; j < lineHi[k]; j++) {
          L[j] -= mx;
          B[j] -= mx;
        }
        F[k] -= mx;
        for (let q = 0; q < K; q++) {
          IL[k * K + q] -= mx;
          IB[k * K + q] -= mx;
        }
      }
    }
  }

  // The beam for the next step (stream_follower.py _prune): lines with belief, their neighbours, the
  // shown word's line, the tracker's proposals; lines leaving it are dropped.
  _prune() {
    const cfg = this.cfg;
    if (cfg.beam <= 0) return;
    const dd = this._dua(this.dua);
    const { nl } = dd;
    const K = this.chain.K;
    const { L, B, F, IL, IB } = this.st;
    let m = NEG;
    for (let k = 0; k < nl; k++) {
      if (!this.active[k]) continue; // dropped lines are NEG
      for (let j = dd.lineLo[k]; j < dd.lineHi[k]; j++) m = Math.max(m, lae(L[j], B[j]));
      m = Math.max(m, F[k]);
      for (let q = 0; q < K; q++) m = Math.max(m, IL[k * K + q]);
    }
    const mass = new Float64Array(nl);
    let tot = 0;
    for (let k = 0; k < nl; k++) {
      if (!this.active[k]) continue;
      for (let j = dd.lineLo[k]; j < dd.lineHi[k]; j++) mass[k] += Math.exp(lae(L[j], B[j]) - m);
      mass[k] += Math.exp(F[k] - m);
      for (let q = 0; q < K; q++) mass[k] += Math.exp(IL[k * K + q] - m) + Math.exp(IB[k * K + q] - m);
      tot += mass[k];
    }
    const on = new Uint8Array(nl);
    const mg = cfg.beamMargin;
    for (let k = 0; k < nl; k++) {
      if (mass[k] < cfg.beam * tot) continue;
      for (let i = Math.max(0, k - mg); i <= Math.min(nl - 1, k + mg); i++) on[i] = 1;
    }
    if (this.word != null && this.ix.wordDua[this.word] === this.dua) {
      const k = dd.lineOfWord[this.word - dd.lo];
      for (let i = Math.max(0, k - 1); i <= Math.min(nl - 1, k + 1); i++) on[i] = 1;
    }
    for (const prop of [this.proposed, this.ctcProp]) {
      if (!prop) continue;
      for (const k of prop) for (let i = Math.max(0, k - 1); i <= Math.min(nl - 1, k + 1); i++) on[i] = 1;
    }
    for (let k = 0; k < nl; k++) {
      if (!this.active[k] || on[k]) continue;
      for (let j = dd.lineLo[k]; j < dd.lineHi[k]; j++) {
        L[j] = NEG;
        B[j] = NEG;
      }
      F[k] = NEG;
      for (let q = 0; q < K; q++) {
        IL[k * K + q] = NEG;
        IB[k * K + q] = NEG;
      }
    }
    this.active = on;
  }

  // Lines whose text explains the latest ctcWindow s best (endScores over the whole du'a).
  _frameProposals(frames, T, C) {
    const cfg = this.cfg;
    const dd = this._dua(this.dua);
    const n = Math.max(1, Math.round(cfg.ctcWindow / cfg.frameS));
    const from = Math.max(0, T - n);
    const sq = squeezeBlanks(frames.subarray(from * C, T * C), T - from, C);
    const sc = endScores(sq.lp, sq.T, C, dd.r);
    const perLine = new Float64Array(dd.nl).fill(-Infinity);
    for (let j = 0; j < dd.J; j++) {
      const k = dd.lineOfWord[dd.wordOf[j]];
      if (sc[j] > perLine[k]) perLine[k] = sc[j];
    }
    let best = -Infinity;
    for (let k = 0; k < dd.nl; k++) if (perLine[k] > best) best = perLine[k];
    const order = Array.from({ length: dd.nl }, (_, k) => k).sort((a, b) => perLine[b] - perLine[a]);
    this.ctcProp = order.slice(0, cfg.ctcLines).filter((k) => perLine[k] >= best - cfg.ctcMargin);
    if (cfg.ctcPushP > 0) this._framePush(dd, sc, perLine);
  }

  // cfg.ctcPushP: the frames' sure line gets that share of the belief (stream_follower.py _frame_push).
  _framePush(dd, sc, perLine) {
    const cfg = this.cfg;
    const nl = dd.nl;
    for (let k = 0; k < nl; k++) if (!Number.isFinite(perLine[k])) return; // a broken window: no push
    let bl = 0;
    for (let k = 1; k < nl; k++) if (perLine[k] > perLine[bl]) bl = k;
    let other = -Infinity;
    let any = false;
    for (let k = 0; k < nl; k++) {
      if (Math.abs(k - bl) <= 1) continue;
      any = true;
      if (perLine[k] > other) other = perLine[k];
    }
    if (any && perLine[bl] - other < cfg.ctcSure) return;
    if (this.word != null && this.ix.wordDua[this.word] === this.dua
        && Math.abs(bl - dd.lineOfWord[this.word - dd.lo]) <= 1) return;
    // the line's best word: the highest end score over its letters, word by word (first best)
    let bw = dd.lineFirstWord[bl];
    let bs = -Infinity;
    for (let w = dd.lineFirstWord[bl]; w <= dd.lineLastWord[bl]; w++) {
      let ws = -Infinity;
      for (let j = dd.wordFirst[w]; j <= dd.wordLast[w]; j++) if (sc[j] > ws) ws = sc[j];
      if (ws > bs) {
        bs = ws;
        bw = w;
      }
    }
    const j = dd.wordLast[bw];
    const parts = [this.st.L, this.st.B, this.st.F, this.st.IL, this.st.IB];
    let m = -Infinity;
    for (const x of parts) for (let i = 0; i < x.length; i++) if (x[i] > m) m = x[i];
    let z = 0;
    for (const x of parts) for (let i = 0; i < x.length; i++) z += Math.exp(x[i] - m);
    const logz = m + Math.log(z);
    const p = cfg.ctcPushP;
    const lq = Math.log(1 - p);
    for (const x of parts) for (let i = 0; i < x.length; i++) x[i] += lq;
    this.st.B[j] = lae(this.st.B[j], Math.log(p) + logz);
    if (this.active) for (let i = Math.max(0, bl - 1); i <= Math.min(nl - 1, bl + 1); i++) this.active[i] = 1;
  }

  // Move pushP of the belief to the blank before word w (its line made active).
  _push(w) {
    const dd = this._dua(this.dua);
    const k = w - dd.lo;
    const li = dd.lineOfWord[k];
    if (this.active) for (let i = Math.max(0, li - 1); i <= Math.min(dd.nl - 1, li + 1); i++) this.active[i] = 1;
    const p = this.cfg.pushP;
    const lq = Math.log(1 - p);
    for (const a of [this.st.L, this.st.B, this.st.F, this.st.IL, this.st.IB]) for (let i = 0; i < a.length; i++) a[i] += lq;
    if (k > 0) {
      const j = dd.wordLast[k - 1];
      this.st.B[j] = lae(this.st.B[j], Math.log(p));
    } else {
      const j = dd.wordFirst[0];
      this.st.L[j] = lae(this.st.L[j], Math.log(p));
    }
  }

  _trackerEvidence(lineMass) {
    if (!lineMass || this.cfg.trackerWeight <= 0) return;
    const dd = this._dua(this.dua);
    const { nl, J } = dd;
    const K = this.chain.K;
    const g = new Float64Array(nl);
    let gm = -Infinity;
    this.proposed = [];
    for (let k = 0; k < nl; k++) {
      const mk = lineMass(dd.lo + dd.lineFirstWord[k]);
      if (mk >= this.cfg.proposeMass) this.proposed.push(k);
      g[k] = this.cfg.trackerWeight * Math.log(mk + this.cfg.trackerFloor);
      if (g[k] > gm) gm = g[k];
    }
    for (let k = 0; k < nl; k++) g[k] -= gm;
    const { L, B, F, IL, IB } = this.st;
    for (let j = 0; j < J; j++) {
      const k = dd.lineOfWord[dd.wordOf[j]];
      L[j] += g[k];
      B[j] += g[k];
    }
    for (let k = 0; k < nl; k++) {
      F[k] += g[k];
      for (let q = 0; q < K; q++) {
        IL[k * K + q] += g[k];
        IB[k * K + q] += g[k];
      }
    }
  }

  // Belief per word of the du'a ({pw}), and the share off the text (filler, interjection: pf).
  // cfg.sharedWords: the letters of the words ending at `other` (another du'a) also end at a word near `cur`
  // in the shown du'a (stream_follower.py _shared_here).
  _sharedHere(cur, other) {
    const ix = this.ix;
    const cfg = this.cfg;
    const fl = this.firstLetter;
    const db = ix.wordDua[other];
    if (other - cfg.sharedWords + 1 < ix.duaWordSpan[db][0]) return false;
    const a = fl[other - cfg.sharedWords + 1];
    const b = fl[other + 1];
    const n = b - a;
    const [lo, hi] = ix.duaWordSpan[this.dua];
    for (let w = Math.max(lo, cur - cfg.sharedNear); w < Math.min(hi, cur + cfg.sharedNear + 1); w++) {
      const e = fl[w + 1];
      if (e - n < fl[lo]) continue;
      let same = true;
      for (let i = 0; i < n && same; i++) same = ix.letters[e - n + i] === ix.letters[a + i];
      if (same) return true;
    }
    return false;
  }

  // The share of the belief in the interjection chain (the salawat between lines).
  _interjShare(st) {
    const dd = this._dua(this.dua);
    const { L, B, F, IL, IB } = st;
    const K = this.chain.K;
    if (!K) return 0;
    let m = NEG;
    for (let j = 0; j < L.length; j++) m = Math.max(m, lae(L[j], B[j]));
    for (let k = 0; k < F.length; k++) m = Math.max(m, F[k]);
    for (let i = 0; i < IL.length; i++) m = Math.max(m, IL[i], IB[i]);
    let z = 0;
    let pi = 0;
    for (let j = 0; j < L.length; j++) z += Math.exp(lae(L[j], B[j]) - m);
    for (let k = 0; k < F.length; k++) z += Math.exp(F[k] - m);
    for (let i = 0; i < IL.length; i++) pi += Math.exp(IL[i] - m) + Math.exp(IB[i] - m);
    z += pi;
    return z > 0 ? pi / z : 0;
  }

  posterior(st) {
    const dd = this._dua(this.dua);
    const { L, B, F, IL, IB } = st;
    const K = this.chain.K;
    const all = this.cfg.beam <= 0 || !this.active;
    let m = NEG;
    for (let k = 0; k < dd.nl; k++) {
      if (!all && !this.active[k]) continue; // dropped lines are NEG: nothing to add
      for (let j = dd.lineLo[k]; j < dd.lineHi[k]; j++) m = Math.max(m, lae(L[j], B[j]));
      m = Math.max(m, F[k]);
      for (let q = 0; q < K; q++) m = Math.max(m, IL[k * K + q], IB[k * K + q]);
    }
    const pw = new Float64Array(dd.nw);
    let z = 0;
    let pf = 0;
    for (let k = 0; k < dd.nl; k++) {
      if (!all && !this.active[k]) continue;
      for (let j = dd.lineLo[k]; j < dd.lineHi[k]; j++) {
        const p = Math.exp(lae(L[j], B[j]) - m);
        pw[dd.wordOf[j]] += p;
        z += p;
      }
      pf += Math.exp(F[k] - m);
      for (let q = 0; q < K; q++) pf += Math.exp(IL[k * K + q] - m) + Math.exp(IB[k * K + q] - m);
    }
    z += pf;
    for (let w = 0; w < dd.nw; w++) pw[w] /= z;
    return { pw, pf: pf / z };
  }

  // One CTC window (T x C, row-major) ending at t. quietNow: seconds without voice at t (the page's
  // stop detector); quietThen: the same lookahead earlier (null: quietNow less the lookahead).
  // lineMass(word): the tracker's belief in that word's line; anchorT: when the tracker's latest
  // result arrived. Returns the word to show (null: nothing).
  step(frames, T, C, t, hmmWord, quietNow = null, lineMass = null, anchorT = null, quietThen = null) {
    const w = this._step(frames, T, C, t, hmmWord, quietNow, lineMass, anchorT, quietThen);
    if (this.word !== this.shownW) {
      this.shownW = this.word;
      this.shownSince = t;
    }
    return w;
  }

  _step(frames, T, C, t, hmmWord, quietNow = null, lineMass = null, anchorT = null, quietThen = null) {
    const ix = this.ix;
    const cfg = this.cfg;
    // the audio this step adds (steps come further apart on a slow phone)
    const dt = this.tStep == null ? cfg.hop : Math.min(Math.max(t - this.tStep, 0), 1);
    this.tStep = t;
    if (hmmWord == null) {
      if (this.dua == null) return null;
      if (this.lapseSince == null) {
        this.lapseSince = t;
        this.lapseVoice = 0;
      }
      if (quietNow != null && quietNow < cfg.lapseQuiet && this.off >= cfg.lapseOff && this.intj < cfg.lapseIntj) {
        this.lapseVoice += cfg.hop;
      }
      if (t - this.lapseSince > cfg.lapseS || this.lapseVoice > cfg.lapseVoice) {
        this.reset();
        return null;
      }
      hmmWord = this.word;
      lineMass = null;
    } else this.lapseSince = null;
    let d = ix.wordDua[hmmWord];
    if (d === this.dua) {
      this.switchTo = null;
      this.switchSince = null;
    } else if (this.dua != null && cfg.switchS > 0 && cfg.sharedWords && this.word != null
        && this.duaT != null && t - this.duaT >= cfg.sharedAfter && this._sharedHere(this.word, hmmWord)) {
      this.switchTo = null;
      this.switchSince = null;
      hmmWord = this.word;
      lineMass = null;
      d = this.dua;
    } else if (this.dua != null && cfg.switchS > 0) {
      if (this.switchTo !== d) {
        this.switchTo = d;
        this.switchSince = t;
      }
      if (t - this.switchSince < cfg.switchS) { // not yet: on in the du'a it was in
        hmmWord = this.word;
        lineMass = null;
        d = this.dua;
      }
    }
    if (d !== this.dua) {
      this._start(d, hmmWord, lineMass);
      this.fStream = null;
      this.duaT = t;
    }
    if (anchorT != null && anchorT !== this.lastAnchorT) {
      this.lastAnchorT = anchorT;
      if (lineMass) this._trackerEvidence(lineMass);
    }
    const la = Math.floor(cfg.lookahead / cfg.frameS + 0.5);
    const hop = Math.floor(cfg.hop / cfg.frameS + 0.5);
    const commitEnd = T - la;
    // committed frames counted from the session's start (a 0.05 s step is 2.5 frames: stream_follower.py)
    const fNow = Math.floor((t - cfg.lookahead) / cfg.frameS + 1e-6);
    let from;
    if (this.fStream == null) from = Math.max(0, commitEnd - hop);
    else {
      const k = fNow - this.fStream;
      from = k > 0 ? Math.max(0, commitEnd - k) : commitEnd;
    }
    this.fStream = fNow;
    if (quietThen == null && quietNow != null) quietThen = quietNow - cfg.lookahead;
    const qThen = quietThen != null && quietThen >= cfg.quietS;
    const qNow = quietNow != null && quietNow >= cfg.quietS;
    if (commitEnd > from) this._advance(this.st, frames, C, from, commitEnd, qThen);
    this.steps++;
    // stalled while the reader sounds, the tracker well ahead (or on a later line): push
    if (cfg.pushP > 0 && this.shownSince != null && t - this.shownSince >= cfg.stuckS && quietNow != null
        && quietNow < cfg.quietS && this.word != null && ix.wordDua[hmmWord] === this.dua && this.lapseSince == null) {
      const ahead = hmmWord - this.word >= cfg.pushWords;
      const other = dd0(this).lineOfWord[hmmWord - dd0(this).lo] !== dd0(this).lineOfWord[this.word - dd0(this).lo]
        && hmmWord > this.word;
      if (ahead || other) {
        this._push(hmmWord);
        this.shownSince = t;
      }
    }
    if (cfg.ctcEvery > 0 && this.steps % cfg.ctcEvery === 0) {
      if (qNow) this.ctcProp = null;
      else this._frameProposals(frames, T, C);
    }
    this._prune();
    const tent = StreamFollower._copy(this.st);
    if (T > commitEnd) this._advance(tent, frames, C, Math.max(0, commitEnd), T, qNow);
    const { pw, pf } = this.posterior(tent);
    this.off = pf;
    if (cfg.lapseIntj < 1) this.intj = this._interjShare(tent);
    const dd = this._dua(this.dua);
    if (pf > 0.5) {
      this.cand = null;
      this.candN = 0;
      this.gateS = 0;
      return this.word;
    }
    const cur = this.word;
    let pwd = pw; // the display's belief per word (copies pooled)
    if (cfg.copies && dd.hasCopies) pwd = StreamFollower._poolCopies(dd, pw);
    let best = 0;
    for (let w = 1; w < dd.nw; w++) if (pwd[w] > pwd[best]) best = w;
    if (cfg.copies && dd.hasCopies) {
      // the copy with the most belief, unless none holds copySplit of it (split, as through a repeated
      // block): then of the copies holding at least half the best's, the nearest the word on screen
      const c = dd.wordCopy[best];
      let n = 0;
      let top = -1;
      for (let w = 0; w < dd.nw; w++) {
        if (dd.wordCopy[w] !== c) continue;
        n++;
        if (top < 0 || pw[w] > pw[top]) top = w;
      }
      best = top;
      if (cur != null && ix.wordDua[cur] === this.dua && n > 1 && pw[top] < cfg.copySplit * pwd[top]) {
        const rel = cur - dd.lo;
        let bestScore = Infinity;
        for (let w = 0; w < dd.nw; w++) {
          if (dd.wordCopy[w] !== c || !(pw[w] >= 0.5 * pw[top])) continue;
          const score = Math.abs(w - rel) * 2 + (w < rel ? 1 : 0); // ties: ahead
          if (score < bestScore) {
            bestScore = score;
            best = w;
          }
        }
      }
    }
    const w = dd.lo + best;
    if (cur == null || ix.wordDua[cur] !== this.dua) {
      this.word = w;
      return w;
    }
    if (w === cur) {
      this.cand = null;
      this.candN = 0;
      this.gateS = 0;
      return cur;
    }
    const lineB = dd.lineOfWord[best];
    const lineC = dd.lineOfWord[cur - dd.lo];
    const sameLine = lineB === lineC;
    const intoNext = lineB === lineC + 1;
    const nextLine = intoNext && best === dd.lineFirstWord[lineB];
    let needP;
    let needN;
    if (sameLine) {
      needP = cfg.showP;
      needN = best > cur - dd.lo ? 1 : cfg.lineSteps;
    } else if (nextLine) {
      needP = cfg.nextP;
      needN = cfg.nextSteps;
    } else {
      needP = cfg.lineP;
      needN = cfg.lineSteps;
    }
    if (pwd[best] < needP) {
      this.cand = null;
      this.candN = 0;
      this.gateS = 0;
      return cur;
    }
    this.candN = this.cand === w ? this.candN + 1 : 1;
    this.cand = w;
    let gate = true;
    if (intoNext && cfg.nextMargin > 0) {
      const pg = cfg.gateTentative ? pw : this.posterior(this.st).pw; // each line by its own belief (copies)
      const m = this._nextMargin(dd, lineC, pg, best);
      this.gateS = m >= -cfg.nextSlack ? this.gateS + dt : 0;
      gate = m >= cfg.nextMargin || this.gateS >= cfg.nextHold - 1e-3;
    } else {
      this.gateS = 0;
    }
    if (this.candN >= needN && gate) {
      this.word = w;
      this.cand = null;
      this.candN = 0;
      this.gateS = 0;
    }
    return this.word;
  }

  // Each word's belief summed over its copies (cfg.copies), summed in word order as numpy's bincount does.
  static _poolCopies(dd, pw) {
    const pc = new Float64Array(dd.nw);
    for (let w = 0; w < dd.nw; w++) pc[dd.wordCopy[w]] += pw[w];
    const out = new Float64Array(dd.nw);
    for (let w = 0; w < dd.nw; w++) out[w] = pc[dd.wordCopy[w]];
    return out;
  }

  // Nats by which the frames prefer the words of line k+1 read so far (to upTo) over the same
  // words of the lines they could be instead (stream_follower.py _next_margin): Infinity when they
  // all read the same words, 0 when none that differs is in the beam.
  _nextMargin(dd, k, pw, upTo) {
    const cfg = this.cfg;
    const seen = upTo - dd.lineFirstWord[k + 1] + 1; // words of k+1 reached
    const span = Math.max(cfg.gateWords, seen);
    const words = (li, n = span) => [dd.lineFirstWord[li], Math.min(dd.lineFirstWord[li] + n, dd.lineLastWord[li] + 1)];
    const start = (li) => {
      const [a, b] = words(li);
      let s = 0;
      for (let w = a; w < b; w++) s += pw[w];
      return s;
    };
    // the same letters as line k+1 over the words reached: can't be told apart by listening yet
    const same = (li) => {
      const [a, b] = words(li, seen);
      const [a1, b1] = words(k + 1, seen);
      const i0 = dd.wordFirst[a];
      const n = dd.wordLast[b - 1] + 1 - i0;
      const j0 = dd.wordFirst[a1];
      if (n !== dd.wordLast[b1 - 1] + 1 - j0) return false;
      for (let i = 0; i < n; i++) if (dd.r[i0 + i] !== dd.r[j0 + i]) return false;
      return true;
    };
    const alts = [[k, cfg.cRestart]];
    cfg.cBack.forEach((c, i) => alts.push([k - 1 - i, c]));
    cfg.cSkip.forEach((c, i) => alts.push([k + 2 + i, c]));
    let best = -Infinity;
    let differ = false;
    for (const [li, c] of alts) {
      if (li >= 0 && li < dd.nl && !same(li)) {
        differ = true;
        const mass = start(li);
        if (mass > 1e-300) best = Math.max(best, Math.log(mass) - c);
      }
    }
    if (!differ) return Infinity; // every alternative reads the same words: no gate
    if (best === -Infinity) return 0;
    return Math.log(Math.max(start(k + 1), 1e-300)) - best;
  }
}
