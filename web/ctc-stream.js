// A streaming CTC model's state between steps (src/dua_recognition/stream_ctc.py, StreamStep):
// each step hands the ONNX graph only the audio of the frames that are new, with every cache,
// and keeps what comes back. The page still sends the latest window and its end; this returns
// the latest 2 s of frames, final ones then the newest `lookahead` tentative ones, as the window
// model's rows. ctc-worker.js runs it in the browser, tests/test_ctc_stream.py in Node.

const SR = 16000;
const HOP = 160;
const LOG_FLOOR = -30;
const OUT_FRAMES = 100; // 2 s of frames back to the page
const STATES = ["mel_c", "top_c", "x0q", "k0c", "v0c", "x0p", "kc", "vc", "hp"];

// Encoder frames whose inputs have all arrived after n samples (stream_ctc.frames_ready).
export const framesReady = (n) => Math.max(0, Math.floor((n - 2 * HOP - 200) / 320) + 1);

export class CtcStream {
  constructor(ort, session, meta) {
    Object.assign(this, { ort, session, meta, s: null });
  }

  freshState() {
    const st = {};
    for (const k of STATES) {
      const dims = this.meta.states[k];
      const size = dims.reduce((a, b) => a * b, 1);
      const fill = k === "mel_c" ? -1.5 : k === "top_c" ? -30 : k === "x0p" || k === "hp" ? -1e6 : 0;
      st[k] = new this.ort.Tensor("float32", new Float32Array(size).fill(fill), dims);
    }
    return st;
  }

  // Start over at this window: audio[0] is sample `end - audio.length` of the session.
  coldStart(audio, end) {
    const start = end - audio.length;
    // the first new frame's audio starts 40 samples before its first mel frame's centre
    const f0 = Math.max(0, Math.ceil((start + 40) / 320));
    this.s = { st: this.freshState(), f0, end, histStart: start, hist: audio.slice(), finals: [], first: f0, tent: null };
  }

  reset() {
    this.s = null;
  }

  // audio: the latest window; end: its end, in samples since the session's start.
  async step(audio, end) {
    let s = this.s;
    if (!s || end <= s.end || end - s.end > audio.length) {
      this.coldStart(audio, end);
      s = this.s;
    } else {
      const fresh = audio.subarray(audio.length - (end - s.end));
      const hist = new Float32Array(s.hist.length + fresh.length);
      hist.set(s.hist);
      hist.set(fresh, s.hist.length);
      s.hist = hist;
      s.end = end;
    }
    const C = this.meta.n_cols;
    const R = this.meta.stream.lookahead;
    const F = framesReady(end);
    const n = F - s.f0;
    if (n > 0) {
      const a = HOP * (2 * s.f0 + 1) - 200;
      const b = HOP * 2 * F + 200;
      const chunk = new Float32Array(b - a); // zeros before what the stream has heard
      const from = Math.max(a, s.histStart);
      chunk.set(s.hist.subarray(from - s.histStart, b - s.histStart), from - a);
      // positions count from the stream's first frame: the frames before it (no audio) come out
      // with negative positions, which the graph keeps out of its caches (only distances matter)
      const feeds = { audio: new this.ort.Tensor("float32", chunk, [chunk.length]),
        f0: new this.ort.Tensor("float32", new Float32Array([s.f0 - s.first]), [1]), ...s.st };
      const out = await this.session.run(feeds);
      for (const k of STATES) s.st[k] = out[k + "_out"];
      const lf = out.lp_final.data;
      // final rows are frames f0 - R .. F - R - 1; before the stream's first frame they mean nothing
      for (let i = 0; i < n; i++) {
        if (s.f0 - R + i >= s.first) s.finals.push(lf.slice(i * C, (i + 1) * C));
      }
      if (s.finals.length > OUT_FRAMES) s.finals.splice(0, s.finals.length - OUT_FRAMES);
      s.tent = out.lp_tent.data.slice();
      s.f0 = F;
      // keep the audio the next step can need (from 40 samples before its first mel frame's centre)
      const keepFrom = HOP * (2 * F + 1) - 200;
      if (keepFrom > s.histStart) {
        s.hist = s.hist.slice(keepFrom - s.histStart);
        s.histStart = keepFrom;
      }
    }
    // the latest frames: final ones, then the tentative newest (none yet: blank)
    const fin = s.finals.slice(-(OUT_FRAMES - R));
    const T = fin.length + R;
    const frames = new Float32Array(T * C);
    fin.forEach((row, i) => frames.set(row, i * C));
    for (let i = 0; i < R; i++) {
      const at = (fin.length + i) * C;
      if (s.tent) frames.set(s.tent.subarray(i * C, (i + 1) * C), at);
      else {
        frames.fill(LOG_FLOOR, at, at + C);
        frames[at] = 0;
      }
    }
    return { frames, T, C, F };
  }
}

export { SR };
