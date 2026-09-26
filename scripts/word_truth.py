#!/usr/bin/env python
"""Word-level timings for the test recordings, by CTC forced alignment.

DuaPlayer's human timings mark where each *line* starts. To score the word
highlight too, force-align each labelled line's known text to its audio with
the fine-tuned wav2vec2 (models/wav2vec2-quran-dua) and keep when every word
starts and ends. Knowing the text makes this far more reliable than
recognising it; lines that align badly (the reciter repeated or skipped
words) are flagged by their per-frame score.

    python scripts/word_truth.py            # test reciters -> data/cache/word_truth/<audio_id>.json
    python scripts/word_truth.py --train    # train reciters (for tuning the display)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dump_ctc import LOG_FLOOR, fold_matrix  # noqa: E402

import evaluate as ev  # noqa: E402
from dua_recognition.align import encode  # noqa: E402

OUT = ROOT / "data" / "cache" / "word_truth"
SR = 16000
PAD = 0.3  # seconds of context either side of a line


def main() -> None:
    import argparse

    import torch
    from faster_whisper.audio import decode_audio
    from torchaudio.functional import forced_align
    from transformers import AutoModelForCTC, AutoProcessor

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="train reciters instead of test")
    args = ap.parse_args()
    model_dir = str(ROOT / "models" / "wav2vec2-quran-dua")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForCTC.from_pretrained(model_dir).to(device).eval()
    if device == "cuda":
        model = model.half()
    fold = torch.tensor(fold_matrix(proc.tokenizer.get_vocab(), proc.tokenizer.pad_token_id), device=device)
    stride = 320 / SR  # wav2vec2 frame hop

    duas = ev.load_all()
    ix = ev.CorpusIndex(duas)
    for dua in duas.values():
        di = ix.dua_ids.index(dua.id)
        lo, hi = ix.dua_word_span[di]
        for rec in ev.load_recordings(dua):
            if ev.is_test(rec.reciter) == args.train:
                continue
            out = OUT / f"{rec.audio_id}.json"
            if out.exists():
                continue
            y = decode_audio(str(rec.path), sampling_rate=SR)
            bounds = rec.starts + [(rec.end_s, None)]
            words = []  # [word index (global), start s, end s, line score]
            for (s0, seg), (s1, _) in zip(bounds, bounds[1:]):
                ws = [w for w in range(lo, hi) if ix.word_segment[w] == seg]
                if not ws or s1 - s0 < 0.5:
                    continue
                codes = [encode(ix.words[w].text) for w in ws]
                letters = np.concatenate(codes)
                owner = np.concatenate([np.full(len(c), w) for w, c in zip(ws, codes)])
                a = max(0.0, s0 - PAD)
                x = y[int(a * SR) : int((s1 + PAD) * SR)]
                with torch.no_grad():
                    inp = proc(x, sampling_rate=SR, return_tensors="pt")
                    v = inp.input_values.to(device)
                    logits = model(v.half() if device == "cuda" else v).logits.float()
                    lp = torch.log((logits.softmax(-1) @ fold).clamp_min(np.exp(LOG_FLOOR)))[0].cpu()
                T = lp.shape[0]
                need = len(letters) + int(np.sum(letters[1:] == letters[:-1]))
                if T < need:
                    continue
                path, scores = forced_align(lp[None], torch.tensor(letters[None], dtype=torch.int32), blank=0)
                path, scores = path[0].numpy(), scores[0].numpy()
                # frame -> letter index: each non-blank run is one letter, in order
                li, first, last = -1, {}, {}
                for t, c in enumerate(path):
                    if c == 0:
                        continue
                    if t == 0 or c != path[t - 1]:
                        li += 1  # a doubled letter always has a blank between its two runs
                    w = int(owner[li])
                    first.setdefault(w, t)
                    last[w] = t
                score = float(scores[path != 0].mean()) if (path != 0).any() else -99.0
                for w in ws:
                    if w in first:
                        words.append([w - lo, round(a + first[w] * stride, 3), round(a + (last[w] + 1) * stride, 3), round(score, 3)])
            OUT.mkdir(parents=True, exist_ok=True)
            # Word numbers count from the du'a's first word (evaluate.load_word_truth).
            out.write_text(json.dumps({"dua": dua.id, "words": words}), encoding="utf-8")
            sc = np.array([w[3] for w in words])
            print(f"{dua.id:28s} {rec.reciter[:20]:20s} {len(words):5d} words  line score median {np.median(sc):.2f}", flush=True)


if __name__ == "__main__":
    main()
