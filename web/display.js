// The word highlight between tracker updates: it glides forward at the
// reciter's pace, eases into each new estimate, never steps back within a
// line and never glides past its end. Same rule as src/dua_recognition/display.py.
export const HIGHLIGHT = { ease: 1.0, speedScale: 1.2 }; // tuned on the train reciters (display.py)

export class Highlight {
  constructor(cfg = {}) {
    Object.assign(this, HIGHLIGHT, cfg);
    this.line = null;
    this.t0 = 0;
    this.x0 = 0;
    this.target = 0;
    this.speed = 0;
    this.lo = 0;
    this.hi = 0;
  }

  // A new position arrived at time t (seconds). `line` is any key for the line
  // shown; [lo, hi) the positions in it; `speed` positions per second.
  update(t, line, pos, lo, hi, speed) {
    if (line == null || pos == null) {
      this.line = null;
      return;
    }
    if (line !== this.line) {
      Object.assign(this, { line, x0: pos, lo, hi });
    } else {
      this.x0 = this.position(t);
    }
    Object.assign(this, { t0: t, target: pos, speed: speed * this.speedScale });
  }

  position(t) {
    const dt = Math.max(0, t - this.t0);
    const e = this.ease > 0 ? Math.min(1, dt / this.ease) : 1;
    const x = this.x0 + (this.target - this.x0) * e + this.speed * dt;
    return Math.min(Math.max(this.x0, x), this.hi - 1 + 0.999);
  }

  word(t) {
    return this.line == null ? null : Math.floor(this.position(t));
  }
}
