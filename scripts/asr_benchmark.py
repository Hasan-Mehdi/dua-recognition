#!/usr/bin/env python
"""Character error rate of candidate ASR models on test-reciter windows.

Windows and references come from `build_finetune_set.py --split test`: 6 s
cuts from held-out reciters, each labelled with the reference text it covers.
(Those references were placed using large-v3-turbo's transcripts, so window
edges slightly favour turbo; comparisons among the other models are fair.)

    python scripts/asr_benchmark.py large-v3-turbo small models/whisper-small-dua-ct2
    python scripts/asr_benchmark.py hf-ctc:jonatasgrosman/wav2vec2-large-xlsr-53-arabic

Model names: anything faster-whisper loads, or `hf-ctc:<repo>` for a
Hugging Face wav2vec2-style CTC model, or `hf-whisper:<repo>` for a
Transformers Whisper checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jiwer  # noqa: E402
import numpy as np  # noqa: E402
from faster_whisper.audio import decode_audio  # noqa: E402

from dua_recognition.asr import transcribe_batch  # noqa: E402
from dua_recognition.text import normalize  # noqa: E402

SR = 16000


def windows(limit: int, seed: int = 0):
    rows = [json.loads(line) for line in (ROOT / "data/cache/finetune/test.jsonl").read_text(encoding="utf-8").splitlines()]
    random.Random(seed).shuffle(rows)
    rows = rows[:limit]
    cache = {}
    out = []
    for r in rows:
        y = cache.get(r["audio"])
        if y is None:
            y = cache[r["audio"]] = decode_audio(r["audio"], sampling_rate=SR)
        out.append((y[int(r["start"] * SR) : int(r["end"] * SR)], r["text"]))
    return out


def make_hf_ctc(repo: str, device: str):
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    proc = AutoProcessor.from_pretrained(repo)
    model = AutoModelForCTC.from_pretrained(repo).to(device).eval()
    if device == "cuda":
        model = model.half()

    @torch.no_grad()
    def run(batch):
        inp = proc([x for x in batch], sampling_rate=SR, return_tensors="pt", padding=True)
        vals = inp.input_values.to(device)
        if device == "cuda":
            vals = vals.half()
        logits = model(vals, attention_mask=inp.get("attention_mask", None).to(device)
                       if inp.get("attention_mask", None) is not None else None).logits
        return proc.batch_decode(logits.argmax(-1))
    return run


def make_hf_whisper(repo: str, device: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(repo)
    model = WhisperForConditionalGeneration.from_pretrained(repo).to(device).eval()
    try:
        model.generation_config.lang_to_id  # noqa: B018
    except AttributeError:
        # Older fine-tunes ship a pre-multilingual generation config.
        from transformers import GenerationConfig

        model.generation_config = GenerationConfig.from_pretrained(f"openai/whisper-{_size(model)}")

    @torch.no_grad()
    def run(batch):
        feats = proc.feature_extractor(list(batch), sampling_rate=SR, return_tensors="pt").input_features.to(device)
        gen = model.generate(feats, language="arabic", task="transcribe", max_new_tokens=96)
        return proc.batch_decode(gen, skip_special_tokens=True)
    return run


def _size(model) -> str:
    return {384: "tiny", 512: "base", 768: "small", 1024: "medium", 1280: "large-v3"}[model.config.d_model]


def make_hf_seq2seq(repo: str, device: str):
    """Any Transformers speech seq2seq model with no language token (e.g. Moonshine)."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    proc = AutoProcessor.from_pretrained(repo)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(repo).to(device).eval()

    @torch.no_grad()
    def run(batch):
        out = []
        for x in batch:  # variable-length inputs; Moonshine has no 30 s padding
            inp = proc(x, sampling_rate=SR, return_tensors="pt").to(device)
            # ~6.5 tokens per second of audio is Moonshine's own guidance for max length
            gen = model.generate(**inp, max_new_tokens=int(len(x) / SR * 13) + 8)
            out.append(proc.batch_decode(gen, skip_special_tokens=True)[0])
        return out
    return run


def score(refs, hyps) -> float:
    r = [normalize(x).replace(" ", "") or "-" for x in refs]
    h = [normalize(x).replace(" ", "") or "-" for x in hyps]
    return jiwer.cer(r, h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = windows(args.limit)
    refs = [t for _, t in data]
    print(f"{len(data)} test windows\n\n| model | window CER |\n|---|---|")
    for name in args.models:
        if name.startswith("hf-ctc:"):
            run = make_hf_ctc(name.split(":", 1)[1], args.device)
        elif name.startswith("hf-seq2seq:"):
            run = make_hf_seq2seq(name.split(":", 1)[1], args.device)
        elif name.startswith("hf-whisper:"):
            run = make_hf_whisper(name.split(":", 1)[1], args.device)
        else:
            run = lambda b, n=name: transcribe_batch(list(b), model=n)  # noqa: E731
        t0 = time.time()
        hyps = []
        for i in range(0, len(data), args.batch):
            hyps += run([x for x, _ in data[i : i + args.batch]])
        print(f"| {name} | {score(refs, hyps):.1%} |   ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
