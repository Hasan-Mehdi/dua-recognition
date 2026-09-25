#!/usr/bin/env python
"""Transcribe windows *with the tracker in the loop*, prompting Whisper with the
reference text just before where the tracker currently is.

This can't use the independent-window cache, since each prompt depends on
everything heard so far. All requested recordings advance in lockstep, one
batched ASR call per hop, and the texts land in the usual cache format under
`<model>+prompt`, so scripts/evaluate.py scores them like any other ASR.

    python scripts/transcribe_prompted.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from faster_whisper.audio import decode_audio  # noqa: E402
from transcribe_windows import SR, cache_path  # noqa: E402

from dua_recognition.align import CorpusIndex  # noqa: E402
from dua_recognition.asr import load_model, transcribe_batch  # noqa: E402
from dua_recognition.corpus import load_all, load_recordings  # noqa: E402
from dua_recognition.splits import is_test  # noqa: E402
from dua_recognition.tracker import Tracker  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--split", choices=["test", "train", "all"], default="test")
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--prompt-words", type=int, default=12)
    args = ap.parse_args()
    tag = f"{Path(args.model).name}+prompt"
    load_model(args.model)

    duas = load_all()
    ix = CorpusIndex(duas)
    recs = [r for d in duas.values() for r in load_recordings(d)
            if args.split == "all" or is_test(r.reciter) == (args.split == "test")]
    recs = [r for r in recs if not cache_path(tag, r.audio_id, args.window, args.hop).exists()]
    audio = {r.audio_id: decode_audio(str(r.path), sampling_rate=SR) for r in recs}
    trackers = {r.audio_id: Tracker(ix) for r in recs}
    rows = {r.audio_id: [] for r in recs}
    n_steps = {r.audio_id: int((r.end_s + 1e-6) // args.hop) for r in recs}
    t0 = time.time()
    for k in range(max(n_steps.values(), default=0)):
        live = [r for r in recs if k < n_steps[r.audio_id]]
        t = round((k + 1) * args.hop, 3)
        wins = [audio[r.audio_id][int(max(0.0, t - args.window) * SR) : int(t * SR)] for r in live]
        prompts = [trackers[r.audio_id].prompt(args.prompt_words) for r in live]
        texts = transcribe_batch(wins, model=args.model, prompts=prompts)
        for r, text in zip(live, texts):
            trackers[r.audio_id].update(text, args.hop)
            rows[r.audio_id].append({"t": t, "text": text})
        if k % 300 == 0:
            print(f"step {k}  ({time.time() - t0:.0f} s)", flush=True)
    for r in recs:
        out = cache_path(tag, r.audio_id, args.window, args.hop)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows[r.audio_id]), encoding="utf-8")
    print(f"done: {len(recs)} recordings -> {tag}")


if __name__ == "__main__":
    main()
