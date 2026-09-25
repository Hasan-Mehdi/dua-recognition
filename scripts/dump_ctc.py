#!/usr/bin/env python
"""Run a CTC model over every recording the way the live recognizer hears it.

Same windows as scripts/transcribe_windows.py (6 s, 1 s hop, growing buffer at
the start), but instead of one transcript per window this keeps the model's
per-frame letter posteriors, folded down to the corpus alphabet
(align._ALPHABET: normalized letters, column 0 = blank, which also absorbs
diacritics, word delimiters and specials). The tracker can then score "how
well does the text ending at word w explain this audio" directly
(ctc_align.py) instead of going through a 1-best transcript.

Also writes the greedy transcript of each window to the ordinary window cache
(data/cache/windows/<tag>/), so the same model can be evaluated on the text
path as well.

    python scripts/dump_ctc.py --model models/wav2vec2-quran-dua --test-only
    python scripts/dump_ctc.py --model rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final --tag w2v-quran-stock
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from faster_whisper.audio import decode_audio  # noqa: E402

from dua_recognition.align import _ALPHABET  # noqa: E402
from dua_recognition.asr import speech_in_tail  # noqa: E402
from dua_recognition.corpus import load_all, load_recordings  # noqa: E402
from dua_recognition.splits import is_test  # noqa: E402
from dua_recognition.text import normalize  # noqa: E402
from transcribe_windows import cache_path, room  # noqa: E402

SR = 16000
CTC_CACHE = ROOT / "data" / "cache" / "ctc"
N_COLS = len(_ALPHABET) + 1
LOG_FLOOR = -30.0


def ctc_cache_path(tag: str, audio_id: str, window: float, hop: float) -> Path:
    return CTC_CACHE / tag / f"{audio_id}_w{window:g}_h{hop:g}.npz"


def fold_matrix(vocab: dict[str, int], blank_id: int) -> np.ndarray:
    """[V, N_COLS] 0/1: which corpus column each model token's mass goes to."""
    m = np.zeros((max(vocab.values()) + 1, N_COLS), dtype=np.float32)
    for tok, i in vocab.items():
        n = normalize(tok) if len(tok) == 1 else ""
        m[i, _ALPHABET.get(n, 0) if i != blank_id else 0] = 1.0
    return m


def main() -> None:
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/wav2vec2-quran-dua")
    ap.add_argument("--tag", help="cache name (defaults to the model's last path part)")
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--duas", nargs="*", help="limit to these du'a ids")
    ap.add_argument("--room", type=float, metavar="SNR_DB")
    args = ap.parse_args()
    tag = (args.tag or Path(args.model).name) + (f"+room{args.room:g}" if args.room is not None else "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model).to(device).eval()
    if device == "cuda":
        model = model.half()
    vocab = proc.tokenizer.get_vocab()
    blank = proc.tokenizer.pad_token_id
    fold = torch.tensor(fold_matrix(vocab, blank), device=device)
    inv = {i: t for t, i in vocab.items()}

    for dua in load_all().values():
        if args.duas and dua.id not in args.duas:
            continue
        for rec in load_recordings(dua):
            if args.test_only and not is_test(rec.reciter):
                continue
            out = ctc_cache_path(tag, rec.audio_id, args.window, args.hop)
            txt = cache_path(tag, rec.audio_id, args.window, args.hop)
            if out.exists() and txt.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            txt.parent.mkdir(parents=True, exist_ok=True)
            y = decode_audio(str(rec.path), sampling_rate=SR)
            if args.room is not None:
                y = room(y, args.room, seed=zlib.crc32(rec.audio_id.encode()))
            t0 = time.time()
            n = int((rec.end_s + 1e-6) // args.hop)
            times = [round((k + 1) * args.hop, 3) for k in range(n)]
            wins = [y[int(max(0.0, t - args.window) * SR) : int(t * SR)] for t in times]
            lps, lens, texts = [], [], []
            with torch.no_grad():
                for b in range(0, n, args.batch):
                    chunk = wins[b : b + args.batch]
                    inp = proc(chunk, sampling_rate=SR, return_tensors="pt", padding=True)
                    vals = inp.input_values.to(device)
                    mask = inp.get("attention_mask")
                    logits = model(vals.half() if device == "cuda" else vals,
                                   attention_mask=mask.to(device) if mask is not None else None).logits.float()
                    probs = logits.softmax(-1)
                    folded = torch.log((probs @ fold).clamp_min(np.exp(LOG_FLOOR)))
                    ids = logits.argmax(-1).cpu().numpy()
                    for k, x in enumerate(chunk):
                        f = int(model._get_feat_extract_output_lengths(torch.tensor(len(x))).item()) if len(x) else 0
                        lps.append(folded[k, :f].cpu().numpy().astype(np.float16))
                        lens.append(f)
                        # greedy CTC decode: collapse repeats, drop blanks
                        seq = [i for j, i in enumerate(ids[k, :f]) if i != blank and (j == 0 or i != ids[k, j - 1])]
                        texts.append("".join(inv[i] for i in seq).replace("|", " ").strip())
            fmax = max(lens) if lens else 0
            arr = np.full((n, fmax, N_COLS), LOG_FLOOR, dtype=np.float16)
            for k, a in enumerate(lps):
                arr[k, : len(a)] = a
            np.savez(out, lp=arr, n_frames=np.array(lens, dtype=np.int32), t=np.array(times))
            speech = [speech_in_tail(w) for w in wins]
            rows = [{"t": t, "text": x, "speech": sp} for t, x, sp in zip(times, texts, speech)]
            txt.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
            took = time.time() - t0
            print(f"{dua.id:28s} {rec.reciter[:20]:20s} {n:5d} windows  {took:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
