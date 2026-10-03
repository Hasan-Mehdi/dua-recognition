#!/usr/bin/env python
"""Give a fine-tuned Whisper a shorter audio context, for the phone.

Whisper pads every input to 30 s, and the encoder's cost follows that length,
not the audio's: a 6 s window costs as much as a 30 s one, and the decoder
cross-attends to all 1500 frames too. The live follower only ever sends 6 s.
The encoder's positional embeddings are fixed sinusoids, so an encoder cut to
the first 400 positions (8 s) is exactly a Whisper built for 8 s inputs:
max_source_positions 400, the feature extractor's chunk_length 8. Everything
that reads the config (transformers, transformers.js, CTranslate2 via
faster-whisper's preprocessor_config.json) then feeds it 8 s of features.

Measured before any fine-tuning (docs/results/phone_speed.md): whisper-base-aug-v4
on 200 validation windows, CPU, CER 9.9% at 30 s, 10.4% at 8 s, 16.7% at 7 s;
3.4x less time per window at 8 s.

    python scripts/shorten_context.py models/whisper-base-aug-v4 --seconds 8
    -> models/whisper-base-aug-v4-ctx8 (+ -ct2); then scripts/export_onnx.py on it
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def shorten(model, proc, seconds: int) -> None:
    """In place: model and processor now take `seconds` of audio."""
    import torch
    from transformers import WhisperFeatureExtractor

    n = seconds * 50  # encoder positions: 2 mel frames of 10 ms each
    enc = model.model.encoder
    rows = enc.embed_positions.weight.data[:n].clone()
    enc.embed_positions = torch.nn.Embedding(n, rows.shape[1]).to(rows.device)
    enc.embed_positions.weight.data.copy_(rows)
    enc.embed_positions.requires_grad_(False)
    enc.max_source_positions = model.config.max_source_positions = n
    fe = proc.feature_extractor
    proc.feature_extractor = WhisperFeatureExtractor(feature_size=fe.feature_size, sampling_rate=fe.sampling_rate,
                                                     hop_length=fe.hop_length, chunk_length=seconds, n_fft=fe.n_fft)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Hugging Face Whisper checkpoint directory")
    ap.add_argument("--seconds", type=int, default=8)
    ap.add_argument("--name", help="output name under models/ (default: <model>-ctx<seconds>)")
    ap.add_argument("--no-ct2", action="store_true")
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    src = Path(args.model)
    out = ROOT / "models" / (args.name or f"{src.name}-ctx{args.seconds}")
    model = WhisperForConditionalGeneration.from_pretrained(src)
    proc = WhisperProcessor.from_pretrained(src)
    shorten(model, proc, args.seconds)
    model.save_pretrained(out)
    proc.save_pretrained(out)
    # transformers 5 saves the feature extractor inside processor_config.json; the ONNX
    # export (optimum, an older transformers in .venv-export) looks for this file.
    proc.feature_extractor.to_json_file(out / "preprocessor_config.json")
    print(f"{out}: max_source_positions {model.config.max_source_positions}, "
          f"chunk_length {proc.feature_extractor.chunk_length}")
    if args.no_ct2:
        return
    ct2 = out.with_name(out.name + "-ct2")
    subprocess.run([sys.executable, "-m", "ctranslate2.converters.transformers", "--model", str(out),
                    "--output_dir", str(ct2), "--force", "--quantization", "float16"], check=True)
    shutil.copy(out / "tokenizer.json", ct2 / "tokenizer.json")
    proc.feature_extractor.to_json_file(ct2 / "preprocessor_config.json")
    print(f"{ct2}")


if __name__ == "__main__":
    main()
