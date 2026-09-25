#!/usr/bin/env python
"""Fine-tune a wav2vec2-style CTC model on the same windows as the Whisper fine-tune.

CTC is the other route to a fast recognizer: no autoregressive decoder, so a
window costs one encoder pass (no 30 s padding either), and it can't fall into
the decoding loops Whisper sometimes does. The starting point is a model
already tuned on Quran recitation, which is the closest domain on the Hub.

    python scripts/finetune_ctc.py --base rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final

Writes models/<name>/ (Hugging Face; load with DUA_ASR_MODEL=ctc:models/<name>).
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from finetune_whisper import AudioBank, augment, cer, load_rows  # noqa: E402

SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final")
    ap.add_argument("--name", default="wav2vec2-quran-dua")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCTC, AutoProcessor, get_linear_schedule_with_warmup

    out = ROOT / "models" / args.name
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    train, val = load_rows("train"), load_rows("val")
    print(f"{len(train)} train / {len(val)} val windows", flush=True)
    bank = AudioBank([r["audio"] for r in train + val])

    proc = AutoProcessor.from_pretrained(args.base)
    model = AutoModelForCTC.from_pretrained(args.base, ctc_loss_reduction="mean").cuda()
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()

    def batch_of(rows, train_mode):
        wav = [bank.window(r) for r in rows]
        if train_mode:
            wav = [augment(x, rng) for x in wav]
        inp = proc(wav, sampling_rate=SR, return_tensors="pt", padding=True)
        labels = proc.tokenizer([r["text"] for r in rows], padding=True, return_tensors="pt")
        ids = labels.input_ids.masked_fill(labels.attention_mask == 0, -100)
        mask = inp.get("attention_mask")
        return inp.input_values.cuda(), (mask.cuda() if mask is not None else None), ids.cuda()

    @torch.no_grad()
    def evaluate(rows, limit=400):
        model.eval()
        refs, hyps = [], []
        for i in range(0, min(len(rows), limit), args.batch):
            chunk = rows[i : i + args.batch]
            x, mask, _ = batch_of(chunk, False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, attention_mask=mask).logits
            hyps += proc.batch_decode(logits.argmax(-1).cpu())
            refs += [r["text"] for r in chunk]
        model.train()
        return cer(refs, hyps)

    cer0 = evaluate(val)
    print(f"before: val CER {cer0:.1%}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / (args.batch * args.accum))
    sched = get_linear_schedule_with_warmup(opt, args.warmup, steps)
    model.train()
    step, t0, best = 0, time.time(), None
    for epoch in range(args.epochs):
        rng.shuffle(train)
        for k, i in enumerate(range(0, len(train), args.batch)):
            x, mask, ids = batch_of(train[i : i + args.batch], True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, attention_mask=mask, labels=ids).loss / args.accum
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            if (k + 1) % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step}/{steps}  loss {loss.item() * args.accum:.3f}  "
                      f"{(time.time() - t0) / step:.2f} s/step", flush=True)
        vcer = evaluate(val)
        print(f"epoch {epoch + 1}: val CER {vcer:.1%}", flush=True)
        if best is None or vcer < best:
            best = vcer
            model.save_pretrained(out)
            proc.save_pretrained(out)
    print(f"best val CER {best:.1%} (from {cer0:.1%}) -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
