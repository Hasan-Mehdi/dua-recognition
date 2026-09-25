#!/usr/bin/env python
"""Transcribe every recording the way the live recognizer hears it.

Slides a `--window`-second window over each recording with a `--hop`-second
step and transcribes each window independently, caching the text. Evaluation
(scripts/evaluate.py) then replays these transcripts through the tracker, so
tuning the tracker never needs a GPU.

    python scripts/transcribe_windows.py --model large-v3-turbo
    python scripts/transcribe_windows.py --model models/whisper-base-dua-ct2 --tag base-ft
"""
from __future__ import annotations

import argparse
import json
import zlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faster_whisper.audio import decode_audio  # noqa: E402

from dua_recognition.asr import load_model, speech_in_tail, transcribe_batch  # noqa: E402
from dua_recognition.corpus import Recording, load_all, load_recordings  # noqa: E402
from dua_recognition.splits import is_test  # noqa: E402

CACHE = ROOT / "data" / "cache" / "windows"
SR = 16000


def cache_path(tag: str, audio_id: str, window: float, hop: float) -> Path:
    return CACHE / tag / f"{audio_id}_w{window:g}_h{hop:g}.jsonl"


def room(y: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Studio recording -> something like a phone in a room: a synthetic
    reverb tail (RT60 ~0.5 s) plus pink-ish noise at `snr_db`, including the
    pauses between lines, where real rooms are never silent."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.5 * SR)) / SR
    rir = rng.standard_normal(t.size) * np.exp(-6.9 * t / 0.5)
    rir[0] = 1.0
    h = rir / np.abs(rir).sum() * 4
    n = 1 << int(np.ceil(np.log2(y.size + h.size - 1)))
    wet = np.fft.irfft(np.fft.rfft(y, n) * np.fft.rfft(h, n), n)[: y.size]  # FFT convolution
    white = rng.standard_normal(y.size)
    pink = np.convolve(white, np.ones(32) / 32, mode="same")  # crude low-pass: fan/hum-like (short kernel: cheap)
    noise = 0.6 * pink / pink.std() + 0.4 * white
    speech_power = np.mean(wet[np.abs(wet) > 0.02] ** 2) if np.any(np.abs(wet) > 0.02) else np.mean(wet**2)
    noise *= np.sqrt(speech_power / 10 ** (snr_db / 10) / np.mean(noise**2))
    out = wet + noise
    return (out / (np.abs(out).max() + 1e-9) * 0.5).astype(np.float32)  # phones apply AGC


def _youtube(dua) -> list[Recording]:
    """Downloaded YouTube recordings of a du'a, unlabelled (no line timings yet)."""
    out = []
    for meta_path in sorted((ROOT / "data" / "youtube" / dua.id).glob("*.json")):
        if meta_path.name.endswith(".labels.json"):
            continue
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        dur = float(m.get("duration_s") or 0)
        out.append(Recording(f"yt-{m['video_id']}", dua.id, f"yt:{m.get('channel')}",
                             meta_path.with_name(m["audio"]), dur, [], dur))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--tag", help="cache name (defaults to the model name)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--duas", nargs="*", help="limit to these du'a ids")
    ap.add_argument("--test-only", action="store_true", help="only held-out (test) reciters")
    ap.add_argument("--constrain", action="store_true",
                    help="decode as if the du'a were locked: only its words' tokens allowed (tag gets +lex)")
    ap.add_argument("--room", type=float, metavar="SNR_DB",
                    help="simulate a phone in a room: reverb + noise at this SNR (tag gets +room<SNR>)")
    ap.add_argument("--youtube", action="store_true",
                    help="transcribe data/youtube/* instead (teacher pass for align_offline.py)")
    args = ap.parse_args()
    tag = args.tag or Path(args.model).name
    if args.constrain:
        tag += "+lex"
    if args.room is not None:
        tag += f"+room{args.room:g}"
    load_model(args.model, args.device)

    duas = load_all()
    for dua in duas.values():
        if args.duas and dua.id not in args.duas:
            continue
        recs = _youtube(dua) if args.youtube else load_recordings(dua)
        if args.test_only:
            recs = [r for r in recs if is_test(r.reciter)]
        for rec in recs:
            out = cache_path(tag, rec.audio_id, args.window, args.hop)
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            y = decode_audio(str(rec.path), sampling_rate=SR)
            if args.room is not None:
                y = room(y, args.room, seed=zlib.crc32(rec.audio_id.encode()))
            t0 = time.time()
            # Start with a growing buffer, like the live recognizer does.
            n = int((rec.end_s + 1e-6) // args.hop)
            times = [round((k + 1) * args.hop, 3) for k in range(n)]
            texts: list[str] = []
            for b in range(0, n, args.batch):
                chunk = times[b : b + args.batch]
                wins = [y[int(max(0.0, t - args.window) * SR) : int(t * SR)] for t in chunk]
                texts += transcribe_batch(wins, model=args.model,
                                          constrain_to=" ".join(dua.texts) if args.constrain else None)
            # Record whether the newest audio held speech, so evaluate.py --vad can
            # apply the live VAD gate to these texts after the fact.
            speech = [speech_in_tail(y[int(max(0.0, t - args.window) * SR) : int(t * SR)]) for t in times]
            rows = [{"t": t, "text": x, "speech": sp} for t, x, sp in zip(times, texts, speech)]
            tmp = out.with_suffix(".tmp")
            tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
            tmp.replace(out)
            took = time.time() - t0
            print(f"{dua.id:28s} {rec.reciter[:20]:20s} {len(rows):5d} windows  "
                  f"{took:6.1f}s  ({took / len(rows) * 1000:.0f} ms/window)", flush=True)


if __name__ == "__main__":
    main()
