// Resample whatever the AudioContext runs at down to 16 kHz mono and post Float32 chunks
// ({x, t}) to the page, which forwards them over the WebSocket. Chunks are `chunk` samples
// (processorOptions; 4000 = 250 ms until 2026-10-01): every word the page reacts to waits for
// its chunk to fill, on average half a chunk.
class Capture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0; // fractional read position into the incoming stream
    this.prev = 0; // last sample of the previous block, for interpolation
    this.out = new Float32Array(options?.processorOptions?.chunk || 4000);
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
        // t: AudioContext time of the chunk's last sample (the audio clock; the page maps
        // it to performance time with ctx.getOutputTimestamp()).
        this.port.postMessage({ x: this.out.slice(0), t: currentTime + (i + 1) / sampleRate });
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
