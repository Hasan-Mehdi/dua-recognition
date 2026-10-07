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
        for p in sorted({p for p in paths if p}):
            f = pcm / (hashlib.sha1(p.encode("utf-8")).hexdigest()[:16] + ".npy")
            if not f.exists():
                y = decode_audio(p, sampling_rate=SR)
                np.save(f, (np.clip(y, -1, 1) * 32767).astype(np.int16))
            self.audio[p] = np.load(f, mmap_mode="r")

    def window(self, row) -> np.ndarray:
        if "clip" in row and row["clip"].endswith(".ogg"):  # a harvest window (scripts/build_clip_cache.py)
            import soundfile as sf

            return sf.read(row["clip"], dtype="float32")[0]
        if "clip" in row:  # a whole crowd-sourced clip (scripts/build_crowd_set.py)
            return np.load(ROOT / row["clip"]).astype(np.float32) / 32767
        y = self.audio[row["audio"]]
        return y[int(row["start"] * SR) : int(row["end"] * SR)].astype(np.float32) / 32767


def augment(x: np.ndarray, rng: random.Random, room_p: float = 0.0, rt60: tuple = (0.5, 0.5)) -> np.ndarray:
    """Studio recordings in, phones in a room out: level, noise, and (with
    probability room_p) reverb plus room noise at 5-25 dB SNR. Training on
    noisy audio beats denoising at inference for modern ASR
    (arxiv.org/abs/2512.17562)."""
    if room_p and rng.random() < room_p:
        from transcribe_windows import room

        snr, seed = rng.uniform(5, 25), rng.randrange(1 << 30)
        return room(x, snr, seed=seed, rt60=rng.uniform(*rt60) if rt60[0] != rt60[1] else rt60[0])
    x = x * rng.uniform(0.3, 1.5)
    if rng.random() < 0.6:
        snr_db = rng.uniform(10, 35)
        power = float(np.mean(x**2)) + 1e-10
        x = x + np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, math.sqrt(power / 10 ** (snr_db / 10)), x.shape
        ).astype(np.float32)
    return np.clip(x, -1, 1)


def speed_perturb(x: np.ndarray, rng: random.Random, lo: float = 0.9, hi: float = 1.1) -> np.ndarray:
    """Play faster or slower, pitch moving with it (Kaldi-style speed perturbation)."""
    f = rng.uniform(lo, hi)
    n = max(1, int(len(x) / f))
    return np.interp(np.arange(n) * f, np.arange(len(x)), x).astype(np.float32)


def vtlp(feats: torch.Tensor, rng: random.Random, p: float, lo: float = 0.85, hi: float = 1.15) -> torch.Tensor:
    """Vocal tract length perturbation on log-mel features: stretch the
    frequency axis by alpha, as a longer or shorter vocal tract would (a
    different speaker: women, children, other men)."""
    n_mels = feats.shape[1]
    out = feats.clone()
    for b in range(feats.shape[0]):
        if rng.random() >= p:
            continue
        a = rng.uniform(lo, hi)
        src = torch.clamp(torch.arange(n_mels, dtype=torch.float32) / a, 0, n_mels - 1)
        i0 = src.floor().long()
        i1 = torch.clamp(i0 + 1, max=n_mels - 1)
        w = (src - i0).view(-1, 1)
        out[b] = feats[b, i0] * (1 - w) + feats[b, i1] * w
    return out


def spec_augment(feats: torch.Tensor, rng: random.Random, n_freq: int = 2, f_max: int = 12,
                 n_time: int = 2, t_max: int = 40) -> torch.Tensor:
    """SpecAugment: blank a few mel bands and short stretches of time (masked
    to each clip's mean), so no single cue is indispensable. Time masks stay
    short (<=0.4 s) so a whole word is rarely hidden from its label."""
    out = feats.clone()
    for b in range(feats.shape[0]):
        fill = out[b].mean()
        for _ in range(n_freq):
            w = rng.randint(0, f_max)
            f0 = rng.randint(0, feats.shape[1] - w)
            out[b, f0 : f0 + w, :] = fill
        # only mask where there is audio (features are padded to 30 s)
        used = int((feats[b] > feats[b].min() + 1e-3).any(0).nonzero().max().item()) + 1 if (feats[b] > feats[b].min() + 1e-3).any() else feats.shape[2]
        for _ in range(n_time):
            w = rng.randint(0, t_max)
            t0 = rng.randint(0, max(0, used - w))
            out[b, :, t0 : t0 + w] = fill
    return out


def cer(refs: list[str], hyps: list[str]) -> float:
    import jiwer

    refs = [normalize(r).replace(" ", "") or "-" for r in refs]
    hyps = [normalize(h).replace(" ", "") or "-" for h in hyps]
    return jiwer.cer(refs, hyps)


def whisper_processor(base: str, context: float):
    """The processor for --base, its feature extractor cut to --context seconds as main() cuts the model."""
    from transformers import WhisperFeatureExtractor, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(base, language="arabic", task="transcribe")
    proc.tokenizer.set_prefix_tokens(language="arabic", task="transcribe", predict_timestamps=False)
    if context < 30:
        fe = proc.feature_extractor
        proc.feature_extractor = WhisperFeatureExtractor(feature_size=fe.feature_size, sampling_rate=SR,
                                                         hop_length=fe.hop_length, chunk_length=int(context),
                                                         n_fft=fe.n_fft)
    return proc


def make_batch(rows, train_mode, args, bank, proc, rng, start_id, tproc=None):
    """Audio (augmented when training), log-mel features and label ids for one batch, on the CPU.
    With `tproc` (the --teacher's processor), also the teacher's features of the clean audio."""
    from halls import hall_aug

    wav = [bank.window(r) for r in rows]
    tfeats = tproc.feature_extractor(wav, sampling_rate=SR, return_tensors="pt").input_features if tproc else None
    if train_mode:
        wav = [speed_perturb(x, rng) if rng.random() < args.speed else x for x in wav]
        wav = [hall_aug(x, rng, args.hall_voices) if args.hall and rng.random() < args.hall else
               augment(x, rng, args.room, args.room_rt60) for x in wav]
    feats = proc.feature_extractor(wav, sampling_rate=SR, return_tensors="pt").input_features
    if train_mode and args.vtlp:
        feats = vtlp(feats, rng, args.vtlp)
    if train_mode and args.specaug:
        feats = spec_augment(feats, rng)
    labels = proc.tokenizer([r["text"] for r in rows], padding=True, return_tensors="pt")
    ids = labels.input_ids.masked_fill(labels.attention_mask == 0, -100)
    if (ids[:, 0] == start_id).all():
        ids = ids[:, 1:]  # the model prepends decoder_start itself
    return (feats, ids) if tproc is None else (feats, ids, tfeats)


class _Batches(torch.utils.data.Dataset):
    """Training batches made in worker processes (--workers): each opens the audio and processor itself."""

    def __init__(self, chunks, args, start_id, seed):
        self.chunks, self.args, self.start_id, self.seed = chunks, args, start_id, seed
        self._open = None

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, b):
        if self._open is None:
            paths = sorted({r.get("audio") for c in self.chunks for r in c if r.get("audio")})
            tproc = teacher_processor(self.args.teacher) if self.args.teacher else None
            self._open = (AudioBank(paths), whisper_processor(self.args.base, self.args.context), tproc)
        bank, proc, tproc = self._open
        rng = random.Random((self.seed << 20) ^ b)
        return make_batch(self.chunks[b], True, self.args, bank, proc, rng, self.start_id, tproc)


def teacher_processor(path: str):
    """The --teacher's processor, its feature extractor as saved (a cut model saves its own context)."""
    from transformers import WhisperProcessor

    proc = WhisperProcessor.from_pretrained(path, language="arabic", task="transcribe")
    proc.tokenizer.set_prefix_tokens(language="arabic", task="transcribe", predict_timestamps=False)
    return proc


# Whisper large-v3 added <|yue|> at 50358: its special tokens from there on sit one id above
# those of the earlier models; text tokens and <|endoftext|> (50257) are the same.
V3_SHIFT_FROM = 50358
N_TEXT = 50258  # ids 0..50257: text and <|endoftext|>, the tokens the distillation compares


def kd_loss(student_logits, teacher_logits, ids, temp: float = 1.0):
    """KL(teacher || student) per label token that is text or the end, over the shared ids,
    at temperature `temp` (times temp**2, as Hinton et al.)."""
    keep = (ids >= 0) & (ids < N_TEXT)
    if not keep.any():
        return student_logits.sum() * 0.0
    s = (student_logits[keep].float() / temp).log_softmax(-1)[:, :N_TEXT]
    t = (teacher_logits[keep].float() / temp).log_softmax(-1)[:, :N_TEXT]
    return (t.exp() * (t - s)).sum(-1).mean() * temp**2


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
    ap.add_argument("--room-rt60", type=lambda v: tuple(float(x) for x in v.split(",")), default=(0.5, 0.5),
                    metavar="LO,HI", help="the room's reverberation time, uniform in [LO, HI] s (halls: 0.3,1.5)")
    ap.add_argument("--hall", type=float, default=0.0,
                    help="share of training windows heard from a seat in a hall through a PA, with a crowd "
                         "(halls.hall_aug: RT60 0.4-3 s, 1-4 loudspeakers) instead of --room")
    ap.add_argument("--hall-voices", type=float, default=0.0,
                    help="share of --hall crowds that are real people talking (Common Voice, training side)")
    ap.add_argument("--data", default="", help="training set version suffix, e.g. v2 -> train_v2.jsonl / val_v2.jsonl")
    ap.add_argument("--speed", type=float, default=0.0, help="share of windows speed-perturbed (0.9-1.1x)")
    ap.add_argument("--vtlp", type=float, default=0.0, help="share of windows given vocal tract length perturbation")
    ap.add_argument("--specaug", action="store_true", help="SpecAugment frequency and time masks")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="train only the decoder: the acoustic model keeps what it learnt from many voices")
    ap.add_argument("--crowd", type=float, default=0.0,
                    help="add crowd-sourced clips (data/cache/finetune/crowd.jsonl) as this share of the training set")
    ap.add_argument("--synth", type=float, default=0.0,
                    help="add synthetic ordinary voices (data/cache/finetune/synth.jsonl, scripts/synth_voices.py) "
                         "as up to this share of the training set")
    ap.add_argument("--synth-data", default="synth", help="which synthetic set: data/cache/finetune/<name>.jsonl")
    ap.add_argument("--lora", type=int, default=0, metavar="RANK",
                    help="train LoRA adapters of this rank instead of all weights (for large models)")
    ap.add_argument("--context", type=float, default=30.0, metavar="SECONDS",
                    help="audio context: Whisper pads every input to 30 s, so a 6 s window costs the encoder "
                         "five times its length. Shorter = a smaller Whisper (max_source_positions = 50 per "
                         "second, the feature extractor's chunk_length to match) that the phone runs faster")
    ap.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw",
                    help="adamw8bit (bitsandbytes): a quarter of AdamW's state, so turbo trains in full on 16 GB")
    ap.add_argument("--workers", type=int, default=0, help="processes preparing batches (0: in the training loop)")
    ap.add_argument("--eval-every", type=int, default=0, help="also evaluate (and keep the best) every N steps")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (the schedule still spans --epochs)")
    ap.add_argument("--teacher", default="", help="distil from this Whisper (models/whisper-turbo-srv2): it hears the "
                                                   "clean window, the student the augmented one")
    ap.add_argument("--kd", type=float, default=0.5, help="weight of the teacher's token distributions (--teacher)")
    ap.add_argument("--kd-temp", type=float, default=1.0, help="temperature of the distillation")
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

    from halls import hall_aug

    name = args.name or f"{args.base.split('/')[-1]}-dua"
    out = ROOT / "models" / name
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    sfx = f"_{args.data}" if args.data else ""
    train, val = load_rows("train" + sfx), load_rows("val" + sfx)
    if args.crowd:
        crowd = load_rows("crowd")
        rng.shuffle(crowd)
        n = min(len(crowd), int(len(train) * args.crowd / (1 - args.crowd)))
        train += crowd[:n]
        print(f"+{n} crowd-sourced clips", flush=True)
    if args.synth:
        synth = load_rows(args.synth_data)
        rng.shuffle(synth)
        n = min(len(synth), int(len(train) * args.synth / (1 - args.synth)))
        train += synth[:n]
        print(f"+{n} synthetic clips", flush=True)
    if args.context < 30:
        # A clip longer than the context would lose audio its label still names.
        def secs(r):
            if "seconds" in r:
                return r["seconds"]
            return r["end"] - r["start"] if "clip" not in r else np.load(ROOT / r["clip"], mmap_mode="r").shape[0] / SR

        n0 = len(train)
        train = [r for r in train if secs(r) <= args.context - 0.2]
        val = [r for r in val if secs(r) <= args.context - 0.2]
        print(f"context {args.context:g} s: dropped {n0 - len(train)} training clips longer than that", flush=True)
    print(f"{len(train)} train / {len(val)} val windows", flush=True)
    bank = AudioBank([r.get("audio") for r in train + val])

    proc = WhisperProcessor.from_pretrained(args.base, language="arabic", task="transcribe")
    tok = proc.tokenizer
    tok.set_prefix_tokens(language="arabic", task="transcribe", predict_timestamps=False)
    # float32 master weights whatever the checkpoint was saved in (whisper-turbo-dua is float16)
    model = WhisperForConditionalGeneration.from_pretrained(args.base, dtype=torch.float32)
    n_pos = int(round(args.context * 50))  # encoder frames: 2 mel frames (10 ms each) per position
    if n_pos < model.config.max_source_positions:
        # The encoder's positional embeddings are fixed sinusoids, so the first n_pos rows
        # are exactly those of an n_pos-long encoder: nothing is re-initialized.
        from transformers import WhisperFeatureExtractor

        enc = model.model.encoder
        rows = enc.embed_positions.weight.data[:n_pos].clone()
        enc.embed_positions = torch.nn.Embedding(n_pos, rows.shape[1])
        enc.embed_positions.weight.data.copy_(rows)
        enc.embed_positions.requires_grad_(False)
        enc.max_source_positions = model.config.max_source_positions = n_pos
        fe = proc.feature_extractor
        proc.feature_extractor = WhisperFeatureExtractor(feature_size=fe.feature_size, sampling_rate=SR,
                                                         hop_length=fe.hop_length, chunk_length=int(args.context),
                                                         n_fft=fe.n_fft)
        print(f"encoder context {args.context:g} s ({n_pos} positions)", flush=True)
    model = model.cuda()
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
    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    if args.lora:
        # large-v3-turbo's 809M weights plus AdamW state don't fit in 16 GB;
        # low-rank adapters on every attention and MLP projection do.
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=args.lora, lora_alpha=2 * args.lora, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
        model.print_trainable_parameters()

    teacher, tproc = None, None
    if args.teacher:
        tproc = teacher_processor(args.teacher)
        teacher = WhisperForConditionalGeneration.from_pretrained(args.teacher, dtype=torch.bfloat16).cuda().eval()
        teacher.requires_grad_(False)
        print(f"teacher {args.teacher}: kd {args.kd} at temperature {args.kd_temp}", flush=True)

    def teacher_logits(tfeats, ids):
        from transformers.models.whisper.modeling_whisper import shift_tokens_right

        dec = shift_tokens_right(ids, model.config.pad_token_id, start_id)
        dec = torch.where(dec >= V3_SHIFT_FROM, dec + 1, dec)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return teacher(input_features=tfeats.to(torch.bfloat16), decoder_input_ids=dec).logits

    def batch_of(rows, train_mode):
        feats, ids = make_batch(rows, train_mode, args, bank, proc, rng, start_id)
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

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    sched = get_linear_schedule_with_warmup(opt, args.warmup, steps)
    model.train()
    step, t0, best = 0, time.time(), None

    def keep_if_best(tag):
        nonlocal best
        vloss, vcer = evaluate(val)
        print(f"{tag}: val loss {vloss:.3f}  val CER {vcer:.1%}", flush=True)
        if best is None or vcer < best:
            best = vcer
            model.save_pretrained(out / "adapter" if args.lora else out)
            proc.save_pretrained(out)
            proc.feature_extractor.to_json_file(out / "preprocessor_config.json")  # for export_onnx.py
            print(f"  saved -> {out.relative_to(ROOT)}", flush=True)

    stop = False
    for epoch in range(args.epochs):
        rng.shuffle(train)
        chunks = [train[i : i + args.batch] for i in range(0, len(train), args.batch)]
        if args.workers:
            loader = torch.utils.data.DataLoader(
                _Batches(chunks, args, start_id, args.seed * 1000 + epoch), batch_size=None, shuffle=False,
                num_workers=args.workers, prefetch_factor=2,
                timeout=900)  # a worker that never starts: an error, not a hang
        else:
            loader = (make_batch(c, True, args, bank, proc, rng, start_id, tproc) for c in chunks)
        for batch in loader:
            feats, ids = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res = model(input_features=feats, labels=ids)
            loss = res.loss
            if teacher is not None:
                kd = kd_loss(res.logits, teacher_logits(batch[2].cuda(non_blocking=True), ids), ids, args.kd_temp)
                loss = (1 - args.kd) * loss + args.kd * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                kd_txt = f" (ce {res.loss.item():.3f}, kd {kd.item():.3f})" if teacher is not None else ""
                print(f"epoch {epoch + 1} step {step}/{steps}  loss {loss.item():.3f}{kd_txt}  "
                      f"{(time.time() - t0) / step:.2f} s/step", flush=True)
            if args.eval_every and step % args.eval_every == 0:
                keep_if_best(f"step {step}")
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        keep_if_best(f"epoch {epoch + 1}")
        if stop:
            break

    if args.lora:
        # Merge the best adapter into full weights so the export below is an
        # ordinary Whisper checkpoint.
        from peft import PeftModel

        base = WhisperForConditionalGeneration.from_pretrained(args.base, dtype=torch.float32)
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
