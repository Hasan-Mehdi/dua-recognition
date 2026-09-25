// Resample whatever the AudioContext runs at down to 16 kHz mono and post
// ~250 ms Float32 chunks to the page, which forwards them over the WebSocket.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0; // fractional read position into the incoming stream
    this.prev = 0; // last sample of the previous block, for interpolation
    this.out = new Float32Array(4000);
    this.n = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input.length) return true;
    const ch = input[0];
    // Mix down if stereo (a replayed file usually is).
    let x = ch;
    if (input.length > 1) {
      x = new Float32Array(ch.length);
      for (let c = 0; c < input.length; c++) {
        const src = input[c];
        for (let i = 0; i < ch.length; i++) x[i] += src[i] / input.length;
      }
    }
    while (this.pos < x.length) {
      const i = Math.floor(this.pos);
      const f = this.pos - i;
      const a = i === 0 ? this.prev : x[i - 1];
      const b = x[i];
      this.out[this.n++] = a + (b - a) * f;
      if (this.n === this.out.length) {
        this.port.postMessage(this.out.slice(0));
        this.n = 0;
      }
      this.pos += this.ratio;
    }
    this.pos -= x.length;
    this.prev = x[x.length - 1];
    return true;
  }
}

registerProcessor("capture", Capture);
