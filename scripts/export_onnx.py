#!/usr/bin/env python
"""Export a fine-tuned Whisper for the in-browser follower (web/).

Runs optimum's ONNX export, int8-quantizes the encoder and the merged decoder,
and lays the files out the way transformers.js loads them:

    web/models/<name>/{config,generation_config,preprocessor_config,tokenizer*}.json
    web/models/<name>/onnx/{encoder_model,decoder_model_merged}_quantized.onnx

optimum-onnx pins its own transformers version, so run this from a separate
environment:

    python -m venv .venv-export
    .venv-export/Scripts/pip install "optimum-onnx[onnxruntime]"
    .venv-export/Scripts/python scripts/export_onnx.py models/whisper-base-quran-dua
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = ["config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json",
        "merges.txt", "normalizer.json"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="fine-tuned Hugging Face checkpoint directory")
    ap.add_argument("--name", help="output name under web/models/ (default: the model's)")
    args = ap.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path(args.model).resolve()
    out = ROOT / "web" / "models" / (args.name or src.name)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [str(Path(sys.executable).with_name("optimum-cli")), "export", "onnx", "-m", str(src),
             "--task", "automatic-speech-recognition-with-past", "--device", "cpu", tmp],
            check=True,
        )
        tmp = Path(tmp)
        (out / "onnx").mkdir(parents=True, exist_ok=True)
        for name in ("encoder_model", "decoder_model_merged"):
            # The merged decoder hides both decoders in If-subgraphs; without
            # EnableSubgraph they stay fp32 and the download triples.
            quantize_dynamic(tmp / f"{name}.onnx", out / "onnx" / f"{name}_quantized.onnx",
                             weight_type=QuantType.QUInt8, extra_options={"EnableSubgraph": True})
        for f in KEEP:
            for base in (tmp, src):
                if (base / f).exists():
                    shutil.copy(base / f, out / f)
                    break

    # transformers.js needs the multilingual fields (lang_to_id, task_to_id...);
    # take them from the checkpoint's own generation config.
    gen = json.loads((src / "generation_config.json").read_text(encoding="utf-8"))
    if "lang_to_id" not in gen:
        raise SystemExit("generation_config.json lacks lang_to_id; fine-tune with scripts/finetune_whisper.py")
    (out / "generation_config.json").write_text(json.dumps(gen, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in (out / "onnx").iterdir()) / 1e6
    print(f"wrote {out.relative_to(ROOT)} ({size:.0f} MB of int8 ONNX)")


if __name__ == "__main__":
    main()
