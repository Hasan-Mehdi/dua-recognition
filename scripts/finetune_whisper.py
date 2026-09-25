#!/usr/bin/env python
"""Fine-tune a small Whisper on train-reciter windows, then export for faster-whisper.

Goal: follow a recitation in real time on a plain CPU. large-v3-turbo is
accurate but takes ~3 s per window on CPU; whisper-small/base are fast enough
but, off the shelf, barely understand melodic recitation. This closes the gap.

Data comes from scripts/build_finetune_set.py (reference-snapped labels on
6 s windows, train reciters only, two du'as held out entirely).

    python scripts/finetune_whisper.py --base openai/whisper-small
    python scripts/finetune_whisper.py --base openai/whisper-base --lr 3e-5

Writes models/<name>/ (Hugging Face) and models/<name>-ct2/ (CTranslate2,
loadable with DUA_ASR_MODEL=models/<name>-ct2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dua_recognition.text import normalize  # noqa: E402

DATA = ROOT / "data" / "cache" / "finetune"
SR = 16000


def load_rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line]


class AudioBank:
    """Decode each recording once to 16 kHz int16 on disk, then memory-map it.

    Tens of hours of audio held in RAM would count against the Windows commit
    limit on top of CUDA's own; file-backed pages don't.
    """

    def __init__(self, paths):
        from faster_whisper.audio import decode_audio

        pcm = ROOT / "data" / "cache" / "pcm"
        pcm.mkdir(parents=True, exist_ok=True)
        self.audio = {}
        for p in sorted(set(paths)):
            f = pcm / (hashlib.sha1(p.encode("utf-8")).hexdigest()[:16] + ".npy")
            if not f.exists():
                y = decode_audio(p, sampling_rate=SR)
                np.save(f, (np.clip(y, -1, 1) * 32767).astype(np.int16))
            self.audio[p] = np.load(f, mmap_mode="r")

    def window(self, row) -> np.ndarray:
        y = self.audio[row["audio"]]
        return y[int(row["start"] * SR) : int(row["end"] * SR)].astype(np.float32) / 32767


def augment(x: np.ndarray, rng: random.Random, room_p: float = 0.0) -> np.ndarray:
    """Studio recordings in, phones in a room out: level, noise, and (with
    probability room_p) reverb plus room noise at 5-25 dB SNR. Training on
    noisy audio beats denoising at inference for modern ASR
    (arxiv.org/abs/2512.17562)."""
    if room_p and rng.random() < room_p:
        from transcribe_windows import room

        return room(x, rng.uniform(5, 25), seed=rng.randrange(1 << 30))
    x = x * rng.uniform(0.3, 1.5)
    if rng.random() < 0.6:
        snr_db = rng.uniform(10, 35)
        power = float(np.mean(x**2)) + 1e-10
        x = x + np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, math.sqrt(power / 10 ** (snr_db / 10)), x.shape
        ).astype(np.float32)
    return np.clip(x, -1, 1)


def cer(refs: list[str], hyps: list[str]) -> float:
    import jiwer

    refs = [normalize(r).replace(" ", "") or "-" for r in refs]
    hyps = [normalize(h).replace(" ", "") or "-" for h in hyps]
    return jiwer.cer(refs, hyps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="openai/whisper-small")
    ap.add_argument("--name", help="output name (default: whisper-<size>-dua)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--room", type=float, default=0.0, help="share of training windows given room reverb + noise")
    ap.add_argument("--lora", type=int, default=0, metavar="RANK",
                    help="train LoRA adapters of this rank instead of all weights (for large models)")
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

    name = args.name or f"{args.base.split('/')[-1]}-dua"
    out = ROOT / "models" / name
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    train, val = load_rows("train"), load_rows("val")
    print(f"{len(train)} train / {len(val)} val windows", flush=True)
    bank = AudioBank([r["audio"] for r in train + val])

    proc = WhisperProcessor.from_pretrained(args.base, language="arabic", task="transcribe")
    tok = proc.tokenizer
    tok.set_prefix_tokens(language="arabic", task="transcribe", predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(args.base).cuda()
    model.config.use_cache = False
    # Non-reentrant checkpointing also works when the base weights are frozen (LoRA).
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not hasattr(model.generation_config, "lang_to_id"):
        # Older fine-tunes (e.g. tarteel-ai's) ship a pre-multilingual generation config.
        from transformers import GenerationConfig

        size = {384: "tiny", 512: "base", 768: "small", 1024: "medium"}[model.config.d_model]
        model.generation_config = GenerationConfig.from_pretrained(f"openai/whisper-{size}")
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    start_id = model.config.decoder_start_token_id
    if args.lora:
        # large-v3-turbo's 809M weights plus AdamW state don't fit in 16 GB;
        # low-rank adapters on every attention and MLP projection do.
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=args.lora, lora_alpha=2 * args.lora, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
        model.print_trainable_parameters()

    def batch_of(rows, train_mode):
        wav = [bank.window(r) for r in rows]
        if train_mode:
            wav = [augment(x, rng, args.room) for x in wav]
        feats = proc.feature_extractor(wav, sampling_rate=SR, return_tensors="pt").input_features
        labels = tok([r["text"] for r in rows], padding=True, return_tensors="pt")
        ids = labels.input_ids.masked_fill(labels.attention_mask == 0, -100)
        if (ids[:, 0] == start_id).all():
            ids = ids[:, 1:]  # the model prepends decoder_start itself
        return feats.cuda(), ids.cuda()

    @torch.no_grad()
    def evaluate(rows, limit=400):
        model.eval()
        model.config.use_cache = True
        refs, hyps, losses = [], [], []
        for i in range(0, min(len(rows), limit), args.batch):
            chunk = rows[i : i + args.batch]
            feats, ids = batch_of(chunk, False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses.append(model(input_features=feats, labels=ids).loss.item())
                gen = model.generate(feats, max_new_tokens=96)
            hyps += tok.batch_decode(gen, skip_special_tokens=True)
            refs += [r["text"] for r in chunk]
        model.train()
        model.config.use_cache = False
        return float(np.mean(losses)), cer(refs, hyps)

    loss0, cer0 = evaluate(val)
    print(f"before: val loss {loss0:.3f}  val CER {cer0:.1%}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    sched = get_linear_schedule_with_warmup(opt, args.warmup, steps)
    model.train()
    step, t0, best = 0, time.time(), None
    for epoch in range(args.epochs):
        rng.shuffle(train)
        for i in range(0, len(train), args.batch):
            feats, ids = batch_of(train[i : i + args.batch], True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=feats, labels=ids).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step}/{steps}  loss {loss.item():.3f}  "
                      f"{(time.time() - t0) / step:.2f} s/step", flush=True)
        vloss, vcer = evaluate(val)
        print(f"epoch {epoch + 1}: val loss {vloss:.3f}  val CER {vcer:.1%}", flush=True)
        if best is None or vcer < best:
            best = vcer
            model.save_pretrained(out / "adapter" if args.lora else out)
            proc.save_pretrained(out)

    if args.lora:
        # Merge the best adapter into full weights so the export below is an
        # ordinary Whisper checkpoint.
        from peft import PeftModel

        base = WhisperForConditionalGeneration.from_pretrained(args.base)
        merged = PeftModel.from_pretrained(base, out / "adapter").merge_and_unload()
        merged.generation_config = model.generation_config
        merged.save_pretrained(out)

    print(f"best val CER {best:.1%} (from {cer0:.1%}); converting to CTranslate2", flush=True)
    ct2 = out.with_name(out.name + "-ct2")
    subprocess.run(
        [sys.executable, "-m", "ctranslate2.converters.transformers", "--model", str(out),
         "--output_dir", str(ct2), "--force", "--quantization", "float16"],
        check=True,
    )
    # faster-whisper looks for these two next to the weights.
    shutil.copy(out / "tokenizer.json", ct2 / "tokenizer.json")
    proc.feature_extractor.to_json_file(ct2 / "preprocessor_config.json")
    print(f"done: DUA_ASR_MODEL={ct2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
