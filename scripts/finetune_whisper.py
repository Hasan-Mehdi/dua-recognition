#!/usr/bin/env python
"""LoRA fine-tune of Whisper on aligned (audio, text) du'a pairs.

Skeleton. The data loader is the missing piece — it depends on the forced
alignment step (segment-level text + timestamps) which isn't wired up yet.
General Arabic ASR already does an okay job on the well-known openings; the win
from fine-tuning is on the du'a-specific vocabulary (names, set phrases).

Needs: transformers, peft, datasets, accelerate.
"""
from __future__ import annotations

import argparse


def build_dataset(aligned_dir: str):
    """Load aligned clips into a HF dataset of {audio, text}.

    TODO: read the alignment output (per-segment text + timestamps), slice the
    source audio, and yield {"audio": np.ndarray, "text": str}. Hold out ~20% of
    recordings (by reciter, not by clip) for a clean eval split.
    """
    raise NotImplementedError("alignment output -> dataset loader is the next step")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="openai/whisper-medium")
    ap.add_argument("--aligned", default="data/aligned")
    ap.add_argument("--out", default="models/whisper-dua-lora")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        ),
    )
    model.print_trainable_parameters()

    _ = build_dataset(args.aligned)  # TODO
    # TODO: Seq2SeqTrainer with a WhisperProcessor collator; report WER on the
    # held-out split; save the adapter to args.out.


if __name__ == "__main__":
    main()
