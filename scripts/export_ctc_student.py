#!/usr/bin/env python
"""Export the phone CTC model (ctc_student.py) for the browser: ONNX, int8.

The graph takes a fixed-length log-mel window (Whisper's features, computed in the
page by transformers.js's WhisperFeatureExtractor) and returns the folded letter log
posteriors, one row per 20 ms:

    input_features [1, 80, 100 * window_s]  ->  logp [1, 50 * window_s, 43]

    python scripts/export_ctc_student.py models/ctc-student-base --window 3
    python scripts/export_ctc_student.py --base models/whisper-base-syn-v5-ctx8ft --name ctc-speedtest

Writes web/models/<name>/{model.onnx, model_q8.onnx, preprocessor_config.json, meta.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from dua_recognition.ctc_student import SR, WhisperCTC  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("student", nargs="?", help="trained student folder (student.pt)")
    ap.add_argument("--base", help="no student: a Whisper model's encoder with an untrained head (speed tests)")
    ap.add_argument("--name", help="web/models/<name> (default: the student folder's name)")
    ap.add_argument("--window", type=float, default=3.0)
    args = ap.parse_args()
    meta = {}
    if args.student:
        ck = torch.load(Path(args.student) / "student.pt", map_location="cpu", weights_only=False)
        meta = ck["meta"]
        model = WhisperCTC.from_whisper(meta["base"], meta.get("max_positions"))
        model.load_state_dict(ck["state"])
    else:
        model = WhisperCTC.from_whisper(args.base)
        meta = {"base": args.base, "untrained": True}
    model.eval()
    name = args.name or Path(args.student).name
    out = ROOT / "web" / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    frames = int(round(args.window * 100))
    x = torch.zeros(1, 80, frames)
    onnx = out / "model.onnx"
    torch.onnx.export(model, (x,), str(onnx), input_names=["input_features"], output_names=["logp"],
                      opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(onnx), str(out / "model_q8.onnx"), weight_type=QuantType.QUInt8)
    (out / "preprocessor_config.json").write_text(json.dumps({
        "feature_extractor_type": "WhisperFeatureExtractor", "feature_size": 80, "sampling_rate": SR,
        "hop_length": 160, "n_fft": 400, "chunk_length": args.window, "n_samples": int(args.window * SR),
        "nb_max_frames": frames, "padding_side": "right", "padding_value": 0.0, "return_attention_mask": False,
    }, indent=1))
    meta = {**meta, "window_s": args.window, "frames_in": frames, "frames_out": frames // 2, "frame_s": 0.02}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    for f in ("model.onnx", "model_q8.onnx"):
        print(f"{f}: {(out / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
