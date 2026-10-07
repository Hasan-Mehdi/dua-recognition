"""The streaming CTC model: the cache changes nothing, and the page's stream (web/ctc-stream.js,
run in Node through onnxruntime-node) gives the frames its Python mirror (stream_ctc.PageStream)
gives, step for step. A small random model stands in for the trained one."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from dua_recognition.ctc_student import WhisperCTC  # noqa: E402
from dua_recognition.stream_ctc import (CausalLogMel, PageStream, StreamCTC, StreamStep, _check_stream,  # noqa: E402
                                        export_step, frames_ready, run_steps)

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
ORT = ROOT / "data" / "cache" / "reliability" / "node" / "node_modules" / "onnxruntime-node"
SR = 16000


def tiny_model(seed: int = 0) -> StreamCTC:
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    torch.manual_seed(seed)
    cfg = WhisperConfig(d_model=64, encoder_layers=3, encoder_attention_heads=4, encoder_ffn_dim=128,
                        decoder_layers=1, decoder_attention_heads=4, decoder_ffn_dim=64, num_mel_bins=80,
                        max_source_positions=400)
    w = WhisperForConditionalGeneration(cfg)
    m = StreamCTC(WhisperCTC(w.model.encoder), lookahead=4, left=20, left0=15)
    with torch.no_grad():
        m.slopes.mul_(torch.rand_like(m.slopes) + 0.5)
    return m.eval()


def signal(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    y = 0.05 * rng.standard_normal(t.size) + 0.3 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0)
    return y.astype(np.float32)


def test_cache_changes_nothing():
    assert _check_stream(tiny_model(), seconds=4.0) < 1e-4


def test_step_matches_whole_audio():
    m = tiny_model(1)
    y = np.r_[np.zeros(4000, np.float32), signal(5.0, 1)]
    with torch.no_grad():
        x0 = m.frontend(CausalLogMel()(torch.tensor(y)[None]))[0]
        hs = m.final_states(x0, seg=37)
        final = m.head(m.layer_norm(hs[-1])).log_softmax(-1).numpy()
    R = m.lookahead
    for F, lf, lt in run_steps(StreamStep(m), y, [int(SR * t) for t in np.arange(0.5, 5.2, 0.1)]):
        f0 = F - lf.shape[0]
        idx = np.arange(f0 - R, F - R)
        ok = idx >= 0
        assert np.abs(lf[ok] - final[idx[ok]]).max() < 1e-4
        with torch.no_grad():
            assert np.abs(lt - m.tentative(hs, [F])[0].numpy()).max() < 1e-4


def test_frames_ready():
    assert frames_ready(32000) == 99 and frames_ready(80000) == 249 and frames_ready(100) == 0


RUNNER = """
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [root, dir, pcm, ends] = process.argv.slice(2);
const require = createRequire(root + "/data/cache/reliability/node/package.json");
const ort = require("onnxruntime-node");
const { CtcStream } = await import(pathToFileURL(root + "/web/ctc-stream.js").href);
const meta = JSON.parse(readFileSync(dir + "/meta.json", "utf8"));
const session = await ort.InferenceSession.create(dir + "/model.onnx");
const s = new CtcStream(ort, session, meta);
const buf = readFileSync(pcm);
const y = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const W = Math.round(meta.window_s * 16000);
const out = [];
for (const e of JSON.parse(ends)) {
  const audio = new Float32Array(W);
  const a = e - W;
  for (let i = Math.max(0, a); i < e; i++) audio[i - a] = y[i];
  const r = await s.step(audio, e);
  out.push(Array.from(r.frames));
}
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None or not ORT.exists(), reason="node + onnxruntime-node (data/cache/reliability/node)")
def test_page_stream_parity(tmp_path):
    m = tiny_model(2)
    meta = export_step(m, tmp_path / "model.onnx")
    meta["window_s"] = 3.0
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    y = signal(9.0, 2)
    pcm = tmp_path / "y.f32"
    y.astype("<f4").tofile(pcm)
    # steps every 0.1 s from 4 s in, one late (a slow step), then a gap longer than the window (cold start)
    ends = [int(SR * t) for t in list(np.arange(4.0, 6.0, 0.1)) + [6.35, 6.4, 6.5] + list(np.arange(8.0, 9.0, 0.1))]
    js = tmp_path / "run.mjs"
    js.write_text(RUNNER, encoding="utf-8")
    res = subprocess.run([NODE, str(js), ROOT.as_posix(), tmp_path.as_posix(), str(pcm), json.dumps(ends)],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout)
    ps = PageStream(StreamStep(m))
    W = 3 * SR
    C = meta["n_cols"]
    for e, g in zip(ends, got):
        audio = np.zeros(W, np.float32)
        a = e - W
        audio[max(0, a) - a:] = y[max(0, a): e]
        want = ps.step(audio, e)
        g = np.array(g, np.float32).reshape(-1, C)
        assert g.shape == want.shape
        assert np.abs(g - want).max() < 1e-3
