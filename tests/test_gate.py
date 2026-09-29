"""The browser speech gate (web/gate.js), run in Node with the real Silero model.

These exercise the page's own code, not a Python copy of its decision logic:
each case builds a 6 s window, runs gate.decide / quietAtEnd through
onnxruntime-node (data/cache/reliability/node) and checks the branch taken.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
ORT = ROOT / "data" / "cache" / "reliability" / "node" / "node_modules" / "onnxruntime-node"
SR = 16000

pytestmark = pytest.mark.skipif(NODE is None or not ORT.exists(),
                                reason="node + onnxruntime-node (npm i in data/cache/reliability/node) needed")

RUNNER = """
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [root, pcm] = process.argv.slice(2);
const require = createRequire(root + "/data/cache/reliability/node/package.json");
const ort = require("onnxruntime-node");
const gate = await import(pathToFileURL(root + "/web/gate.js").href);
const s = await ort.InferenceSession.create(root + "/web/vad/silero_vad_v6.onnx");
const vad = gate.makeVadProbs((f) => s.run(f), ort.Tensor);
const buf = readFileSync(pcm);
const y = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const out = { quiet: await gate.quietAtEnd(y, vad) };
for (const p of gate.POLICIES) out[p] = await gate.decide(y, p, vad);
out.halluc = [await gate.looksHallucinated("شكرا"), await gate.looksHallucinated("الله ".repeat(20)),
              await gate.looksHallucinated("اللهم اني اسالك برحمتك التي وسعت كل شيء")];
console.log(JSON.stringify(out));
"""


def run_gate(tmp_path, y: np.ndarray) -> dict:
    js = tmp_path / "run.mjs"
    js.write_text(RUNNER, encoding="utf-8")
    pcm = tmp_path / "x.f32"
    y.astype("<f4").tofile(pcm)
    out = subprocess.run([NODE, str(js), ROOT.as_posix(), str(pcm)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def tone(seconds: float, db: float, f: float = 220.0, seed: int = 0) -> np.ndarray:
    """A steady harmonic tone (a drawn-out sung vowel, roughly) at `db` dBFS RMS."""
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 6))
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (db / 20)).astype(np.float32)


def noise(seconds: float, db: float, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal(int(seconds * SR))
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (db / 20)).astype(np.float32)


def test_quiet_input_is_below_the_floor_for_every_policy(tmp_path):
    r = run_gate(tmp_path, noise(6, -60))
    for p in ("legacy", "energy_assisted", "ungated"):
        assert r[p]["run"] is False and r[p]["reason"] == "below_floor"
    assert r["quiet"] == pytest.approx(3.0)  # true quiet is reported as quiet


def test_new_sound_rejected_by_silero_reaches_the_energy_branch(tmp_path):
    # 4.5 s of quiet room, then 1.5 s of a steady loud tone: sound, but not speech to Silero.
    y = np.concatenate([noise(4.5, -50), tone(1.5, -20)])
    r = run_gate(tmp_path, y)
    assert r["legacy"]["vad"]["pass"] is False
    assert r["legacy"]["run"] is False and r["legacy"]["reason"] == "vad_reject"
    assert r["energy_assisted"]["run"] is True and r["energy_assisted"]["via"] == "energy"
    assert r["ungated"]["run"] is True and r["ungated"]["vad"] is None  # Silero never consulted
    assert r["quiet"] < 0.5  # quietAtEnd already saw the sound


def test_stationary_noise_stays_gated_under_energy_assist(tmp_path):
    # A fan at a steady -30 dBFS: above the floor, no rise over its own 10th percentile.
    r = run_gate(tmp_path, noise(6, -30, seed=1))
    assert r["legacy"]["reason"] == "vad_reject"
    assert r["energy_assisted"]["run"] is False and r["energy_assisted"]["energy"]["pass"] is False
    assert r["ungated"]["run"] is True


def test_hallucination_check_matches_python(tmp_path):
    from dua_recognition.asr import _looks_hallucinated

    r = run_gate(tmp_path, noise(6, -60))
    texts = ["شكرا", "الله " * 20, "اللهم اني اسالك برحمتك التي وسعت كل شيء"]
    assert r["halluc"] == [_looks_hallucinated(t) for t in texts]
