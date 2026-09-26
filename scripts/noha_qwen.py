#!/usr/bin/env python
"""Qwen3-ASR as the noha front end, cached in scripts/noha_match.py's format.

Qwen3-ASR is trained for song and singing recognition and does its own
language ID, but has no Urdu: spoken Urdu is forced as Hindi (Devanagari),
which noha_match's phonetic skeleton reads like Urdu script or Roman Urdu.
Needs `pip install qwen-asr` (pins transformers 4.57; use a separate venv).

    python scripts/noha_qwen.py --model Qwen/Qwen3-ASR-0.6B
    python scripts/noha_match.py eval --model qwen3-asr-0.6b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import noha_match as N  # noqa: E402

FORCE = {"ur": "Hindi", "ar": "Arabic", "fa": "Persian", "en": "English"}
DETECTED = {"hindi": "ur", "urdu": "ur", "arabic": "ar", "persian": "fa", "english": "en"}


def main() -> None:
    import torch
    from qwen_asr import Qwen3ASRModel

    from dua_recognition.asr import _is_silent, speech_in_tail
    from noha_lid import load_audio

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--batch", type=int, default=40)
    args = ap.parse_args()
    model = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0", max_new_tokens=96)
    cache = N.OUT / "cache" / args.model.split("/")[-1].lower()
    cache.mkdir(parents=True, exist_ok=True)

    def run(wavs, language):
        out = []
        for lo in range(0, len(wavs), args.batch):
            out += model.transcribe([(w, N.SR) for w in wavs[lo: lo + args.batch]], language=language)
        return out

    for m in N.test_items():
        path = cache / f"{m['video_id']}.json"
        if path.exists():
            continue
        audio = load_audio(Path(m["audio_path"]))
        trials = []
        for s, wins in N._windows(audio):
            n = len(wins)
            live = [i for i, w in enumerate(wins) if w.size and not _is_silent(w)]
            t = {"start": s, "texts": {l: [""] * n for l in N.LANGS}, "nsp": {l: [0.0] * n for l in N.LANGS},
                 "vad": [False] * n, "lid": [[0.0] * 4] * n, "auto": [""] * n, "auto_lang": [""] * n}
            ws = [wins[i] for i in live]
            if ws:
                for i in live:
                    t["vad"][i] = bool(speech_in_tail(wins[i]))
                for i, r in zip(live, run(ws, None)):
                    t["auto"][i], t["auto_lang"][i] = r.text, r.language or ""
                    code = DETECTED.get((r.language or "").split(",")[0].strip().lower())
                    # One-hot "log-probs" from the detected language (uniform if outside the four).
                    t["lid"][i] = [0.0 if code == l else -5.0 for l in N.LANGS] if code else [0.0] * 4
                for lang, name in FORCE.items():
                    for i, r in zip(live, run(ws, name)):
                        t["texts"][lang][i] = r.text
            trials.append(t)
        path.write_text(json.dumps({"video_id": m["video_id"], "lang": m["lang"], "trials": trials},
                                   ensure_ascii=False), encoding="utf-8")
        mid = trials[1]
        print(f"{m['lang']} {m['video_id']}: [{mid['auto_lang'][9]}] {mid['texts'][m['lang']][9][:60]}", flush=True)


if __name__ == "__main__":
    main()
