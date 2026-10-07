#!/usr/bin/env python
"""Export a streaming CTC student (stream_ctc.py) for the browser: one ONNX step, int8.

Each call takes the audio of the new frames and every cache, and returns the new final frames,
the newest tentative ones and the caches (stream_ctc.StreamStep has the shapes). ctc-worker.js
keeps the caches between calls (ctc-stream.js); the page itself is unchanged: it still sends the
latest window and gets back the latest 2 s of frames.

    python scripts/export_stream_ctc.py models/ctc-stream-base-v1
    python scripts/export_stream_ctc.py --base models/whisper-base-syn-v5-ctx8ft --name ctc-stream-speedtest

Writes web/models/<name>/{model.onnx, model_q8.onnx, meta.json} and checks the int8 model's
frames against PyTorch's on a recording from the bench.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from dua_recognition.stream_ctc import StreamCTC, StreamStep, StreamStudent, export_step, run_steps  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("student", nargs="?", help="trained streaming student folder (student.pt)")
    ap.add_argument("--base", help="no student: a Whisper encoder, untrained head (speed tests)")
    ap.add_argument("--name", help="web/models/<name> (default: the student folder's name)")
    ap.add_argument("--window", type=float, default=3.0, help="seconds of audio the page sends each step (the "
                                                               "worker uses the new part; at a cold start, all of it)")
    args = ap.parse_args()
    if args.student:
        st = StreamStudent(args.student, device="cpu")
        model, meta = st.model, dict(st.meta)
    else:
        model = StreamCTC.from_whisper(args.base).eval()
        meta = {"base": args.base, "untrained": True}
    model.eval()
    name = args.name or Path(args.student).name
    out = ROOT / "web" / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    meta = {**meta, **export_step(model, out / "model.onnx", out / "model_q8.onnx"), "window_s": args.window}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    for f in ("model.onnx", "model_q8.onnx"):
        print(f"{f}: {(out / f).stat().st_size / 1e6:.1f} MB")

    # the int8 graph against PyTorch, step by step on 20 s of a bench recording
    import onnxruntime as ort

    import bench

    step = StreamStep(model).eval()
    it = sorted(bench.load_items(split="dev"), key=lambda i: i["id"])[3]
    y = bench.render(it)[: 16000 * 20].astype(np.float32)
    ends = [int(16000 * t) for t in np.arange(3.0, 20.0, 0.1)]
    ref = run_steps(step, y, ends)
    for f in ("model.onnx", "model_q8.onnx"):
        sess = ort.InferenceSession(str(out / f), providers=["CPUExecutionProvider"])
        got = run_steps(step, y, ends, run=lambda feeds: sess.run(None, feeds))
        df = max(float(np.abs(a[1] - b[1]).max()) for a, b in zip(ref, got))
        agree = np.mean([float((a[1].argmax(-1) == b[1].argmax(-1)).mean()) for a, b in zip(ref, got)])
        print(f"{f}: max |dlogp| {df:.3f}, same best letter {agree:.1%} of final frames")


if __name__ == "__main__":
    main()
