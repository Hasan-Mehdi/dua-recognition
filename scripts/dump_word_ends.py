#!/usr/bin/env python
"""When did the last word of each cached window transcript end?

Aligns each cached window's transcript (scripts/transcribe_windows.py) to its
audio with Whisper's cross-attention and stores how long before the window's
end its last word finished ("staleness"). The tracker's belief is about where
the transcript ends, which is that much before the window does.

    python scripts/dump_word_ends.py --model models/whisper-base-quran-dua-ct2 --tag whisper-base-quran-dua --test-only
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

import evaluate as ev  # noqa: E402
from dua_recognition.asr import load_model, _tokenizer  # noqa: E402

OUT = ROOT / "data" / "cache" / "word_ends"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--audio", help="one audio file instead of the corpus (rows from --rows json)")
    ap.add_argument("--rows", help="json [[t, text], ...] for --audio")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from faster_whisper.audio import decode_audio, pad_or_trim

    m = load_model(args.model)
    tok = _tokenizer(args.model)
    jobs = []
    if args.audio:
        rows = [(float(t), x) for t, x in json.loads(Path(args.rows).read_text(encoding="utf-8"))]
        jobs.append((Path(args.audio), rows, Path(args.rows).with_suffix(".ends.json")))
    else:
        for dua in ev.load_all().values():
            for rec in ev.load_recordings(dua):
                if (args.test_only and not ev.is_test(rec.reciter)) or (args.train_only and ev.is_test(rec.reciter)):
                    continue
                rows = ev.load_rows(args.tag, rec, 6.0, 1.0)
                if rows:
                    jobs.append((rec.path, rows, OUT / args.tag / f"{rec.audio_id}.json"))
    for path, rows, out in jobs:
        if out.exists():
            continue
        y = decode_audio(str(path), sampling_rate=16000)
        ends = [None] * len(rows)
        todo = [i for i, (_, x) in enumerate(rows) if x]
        # align() takes one frame count per batch, so batch windows of equal length
        # (all but the first few seconds of a recording are a full 6 s).
        by_len: dict[int, list[int]] = {}
        for i in todo:
            by_len.setdefault(int(min(rows[i][0], 6.0) * 100), []).append(i)
        for n, group in by_len.items():
            for b in range(0, len(group), args.batch):
                idx = group[b : b + args.batch]
                wins = [y[int(max(0.0, rows[i][0] - 6.0) * 16000) : int(rows[i][0] * 16000)] for i in idx]
                enc = m.encode(np.stack([pad_or_trim(m.feature_extractor(w)) for w in wins]))
                toks = [tok.encode(" " + rows[i][1].strip()) for i in idx]
                for i, w, words in zip(idx, wins, m.find_alignment(tok, toks, enc, max(1, len(wins[0]) // 160))):
                    if words:
                        ends[i] = round(len(w) / 16000 - float(words[-1]["end"]), 3)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ends), encoding="utf-8")
        st = [e for e in ends if e is not None]
        print(f"{out.stem:40s} {len(st):5d} windows  staleness median {np.median(st):.2f} s  p90 {np.percentile(st, 90):.2f} s",
              flush=True)


if __name__ == "__main__":
    main()
