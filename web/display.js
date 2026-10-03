// The word highlight between tracker updates: it glides forward at the
// reciter's pace, eases into each new estimate, never steps back within a
// line (unless told the reciter has stopped: `back`), never glides past its
// end, and stops or restarts between updates on the page's own stop detector
// (pace). Same rule as src/dua_recognition/display.py.
export const HIGHLIGHT = { ease: 1.0, speedScale: 1.2 }; // tuned on the train reciters (display.py)

export class Highlight {
  constructor(cfg = {}) {
    Object.assign(this, HIGHLIGHT, cfg);
    this.line = null;
    this.t0 = 0;
    this.x0 = 0;
    this.target = 0;
    this.speed = 0;
    this.floor = 0; // it doesn't go below this (x0, or the target when going back)
    this.lo = 0;
    this.hi = 0;
  }

  // A new position arrived at time t (seconds). `line` is any key for the line
  // shown; [lo, hi) the positions in it; `speed` positions per second. back: the
  // reciter has stopped a while, and `pos` may be behind the highlight.
  update(t, line, pos, lo, hi, speed, back = false) {
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
    this.floor = back ? Math.min(this.target, this.x0) : this.x0;
  }

  // Between updates: glide on at `speed` from now, or stop (0).
  pace(t, speed) {
    if (this.line == null) return;
    const x = this.position(t);
    const settling = this.floor < this.x0; // on its way back to the target: carry on
    Object.assign(this, { x0: x, t0: t, speed: speed * this.speedScale });
    if (!settling) this.floor = x;
  }

  position(t) {
    const dt = Math.max(0, t - this.t0);
    const e = this.ease > 0 ? Math.min(1, dt / this.ease) : 1;
    // (so that e = 1 lands on the target exactly: see display.py)
    const x = this.target + (this.x0 - this.target) * (1 - e) + this.speed * dt;
    return Math.min(Math.max(this.floor, x), this.hi - 1 + 0.999);
  }

  word(t) {
    return this.line == null ? null : Math.floor(this.position(t));
  }
}
