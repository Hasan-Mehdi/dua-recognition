#!/usr/bin/env python
"""Meta's Omnilingual ASR (CTC, 1600+ languages) as the noha front end.

Language-free: no language token to get wrong, and it writes whichever
script it hears. Runs through sherpa-onnx's int8 export (348 MB for 300M,
CPU; sherpa-onnx also builds for WebAssembly). Cached in
scripts/noha_match.py's format, with the one transcript under every language.

    pip install sherpa-onnx
    python scripts/noha_omni.py --model-dir <sherpa-onnx-omnilingual-asr-...-300M-ctc-int8-2025-11-12>
    python scripts/noha_match.py eval --model omni-ctc-300m
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import noha_match as N  # noqa: E402


def main() -> None:
    import sherpa_onnx

    from dua_recognition.asr import _is_silent, speech_in_tail
    from noha_lid import load_audio

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--name", default="omni-ctc-300m")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    d = Path(args.model_dir)
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    recs = [sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=threads)
        for _ in range(args.workers)]
    cache = N.OUT / "cache" / args.name
    cache.mkdir(parents=True, exist_ok=True)

    def decode(job):
        k, w = job
        rec = recs[k % len(recs)]
        st = rec.create_stream()
        st.accept_waveform(N.SR, w)
        rec.decode_stream(st)
        return st.result.text.strip()

    for m in N.test_items():
        path = cache / f"{m['video_id']}.json"
        if path.exists():
            continue
        audio = load_audio(Path(m["audio_path"]))
        trials = []
        for s, wins in N._windows(audio):
            n = len(wins)
            live = [i for i, w in enumerate(wins) if w.size and not _is_silent(w)]
            with ThreadPoolExecutor(args.workers) as ex:
                texts = list(ex.map(decode, [(j, wins[i]) for j, i in enumerate(live)]))
            text = [""] * n
            for i, t in zip(live, texts):
                text[i] = t
            vad = [bool(i in live and speech_in_tail(wins[i])) for i in range(n)]
            trials.append({"start": s, "texts": {l: text for l in N.LANGS}, "nsp": {l: [0.0] * n for l in N.LANGS},
                           "vad": vad, "lid": [[0.0] * 4] * n})
        path.write_text(json.dumps({"video_id": m["video_id"], "lang": m["lang"], "trials": trials},
                                   ensure_ascii=False), encoding="utf-8")
        print(f"{m['lang']} {m['video_id']}: {trials[1]['texts']['ur'][12][:70]}", flush=True)


if __name__ == "__main__":
    main()
