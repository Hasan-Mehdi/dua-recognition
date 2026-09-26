#!/usr/bin/env python
"""Turn a real user's recording into a labelled test recording (auto labels, to review).

The test reciters are professionals; the people who will use the app aren't.
This takes one recording (any format ffmpeg reads, phone video included) and
writes it in the DuaPlayer shape, so every evaluation script reads it with
`--source user`:

    data/usertest/<dua_id>/<audio_id>.mp3     the audio (16 kHz mono)
    data/usertest/<dua_id>/<audio_id>.json    DuaPlayer meta: line start times ("auto": true)
    data/usertest/<dua_id>/<audio_id>.srt     the line texts on those times, to check in VLC
    data/cache/word_truth/<audio_id>.json     forced-aligned word timings (as scripts/word_truth.py)
    data/cache/windows/<tag>/<audio_id>_w6_h1.jsonl   the windows used (reused by evaluate.py)

Steps: large-v3-turbo windows (6 s, 1 s hop) -> which du'a (the tracker over
the whole corpus, unless --dua is given) -> line starts from the offline
smoother -> word timings from the wav2vec2 forced aligner, line by line.
Labels are machine-made: open the .srt next to the audio and fix what's off
before trusting a number from it. The audio is a person's voice: data/usertest
is gitignored and must stay that way.

    python scripts/user_label.py my_test.mp4 --reciter hasan
    python scripts/user_label.py take.m4a --dua dua-tawassul --reciter guest3
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
from align_offline import placed_mask, starts_from  # noqa: E402
from transcribe_windows import cache_path  # noqa: E402

from dua_recognition.align import CorpusIndex  # noqa: E402
from dua_recognition.asr import _looks_hallucinated, speech_in_tail, transcribe_batch  # noqa: E402
from dua_recognition.offline import align_recording  # noqa: E402
from dua_recognition.tracker import Tracker  # noqa: E402

SR = 16000
OUT = ROOT / "data" / "usertest"


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def identify(ix: CorpusIndex, costs) -> tuple[str | None, float]:
    """The du'a the live tracker is locked on for most of the recording."""
    tr = Tracker(ix)
    seen = Counter()
    for c in costs:
        p = tr.update_costs(c, 1.0)
        if p.dua is not None:
            seen[p.dua] += 1
    if not seen:
        return None, 0.0
    dua, n = seen.most_common(1)[0]
    return dua, n / len(costs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--dua", help="du'a id, if known (skips identification)")
    ap.add_argument("--reciter", default="user", help="a name or code for the person (stays local)")
    ap.add_argument("--model", default="large-v3-turbo", help="ASR for the teacher windows")
    ap.add_argument("--tag", default="large-v3-turbo", help="window-cache tag of --model")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from faster_whisper.audio import decode_audio

    from dua_recognition.asr import load_model

    stem = re.sub(r"[^A-Za-z0-9]+", "-", args.audio.stem).strip("-").lower()
    audio_id = f"user-{re.sub(r'[^A-Za-z0-9]+', '', args.reciter).lower()}-{stem}"
    tmp = OUT / "_incoming" / f"{audio_id}.mp3"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.audio), "-vn", "-ac", "1", "-ar", str(SR),
                    "-b:a", "96k", str(tmp)], check=True)
    y = decode_audio(str(tmp), sampling_rate=SR)
    dur = y.size / SR

    win = cache_path(args.tag, audio_id, 6.0, 1.0)
    if win.exists():
        rows = [json.loads(line) for line in win.read_text(encoding="utf-8").splitlines() if line]
    else:
        load_model(args.model, args.device)
        times = [float(k + 1) for k in range(int(dur))]
        texts = []
        for b in range(0, len(times), 16):
            chunk = times[b : b + 16]
            texts += transcribe_batch([y[int(max(0.0, t - 6.0) * SR) : int(t * SR)] for t in chunk], model=args.model)
        rows = [{"t": t, "text": x, "speech": speech_in_tail(y[int(max(0.0, t - 6.0) * SR) : int(t * SR)])}
                for t, x in zip(times, texts)]
        win.parent.mkdir(parents=True, exist_ok=True)
        win.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    rows = [(r["t"], "" if _looks_hallucinated(r["text"]) else r["text"]) for r in rows]

    duas = ev.load_all()
    full = CorpusIndex(duas)
    if args.dua:
        dua_id, share = args.dua, 1.0
    else:
        dua_id, share = identify(full, ev.LazyCosts(full, [x for _, x in rows]))
        if dua_id is None:
            sys.exit("could not tell which du'a this is; pass --dua")
    dua = duas[dua_id]
    print(f"{audio_id}: {dur:.0f} s, du'a {dua_id} ({dua.name_en}){'' if args.dua else f', locked {share:.0%} of the time'}")

    ix = CorpusIndex({dua_id: dua})
    costs = [ix.word_costs(x) if x else None for _, x in rows]
    segs = align_recording(ix, costs, 1.0)
    placed = placed_mask(ix, rows, costs)
    starts = starts_from([t for t, _ in rows], segs, placed)
    voiced = [p for (_, x), p in zip(rows, placed) if x]
    coverage = sum(voiced) / max(1, len(voiced))
    print(f"  {len(starts)} line starts; teacher placed {coverage:.0%} of voiced windows")
    if not starts:
        sys.exit("no lines placed: wrong du'a, or the recording is too hard for the teacher")

    from word_truth import align_words, load_aligner

    words = align_words(load_aligner(), full, y, starts, dur, full.dua_ids.index(dua_id))
    good = [w for w in words if w[3] >= -1.5]
    print(f"  {len(words)} words aligned, {len(good)} with a line score >= -1.5")

    d = OUT / dua_id
    d.mkdir(parents=True, exist_ok=True)
    mp3 = d / f"{audio_id}.mp3"
    tmp.replace(mp3)
    meta = {"audio_id": audio_id, "dua_id": dua_id, "reciter": f"user:{args.reciter}", "duration_ms": int(dur * 1000),
            "slide_start_ms": {str(s): int(t * 1000) for t, s in starts}, "auto": True,
            "source_file": args.audio.name, "teacher": args.tag, "placed_fraction": round(coverage, 3)}
    (d / f"{audio_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    seg_text = {s.id: s.arabic for s in dua.segments}
    bounds = starts + [(dur, None)]
    cues = [f"{i}\n{srt_time(t0)} --> {srt_time(t1)}\n{s}. {seg_text[s]}\n"
            for i, ((t0, s), (t1, _)) in enumerate(zip(bounds, bounds[1:]), 1)]
    (d / f"{audio_id}.srt").write_text("\n".join(cues), encoding="utf-8")
    ev.WORD_TRUTH.mkdir(parents=True, exist_ok=True)
    (ev.WORD_TRUTH / f"{audio_id}.json").write_text(json.dumps({"dua": dua_id, "words": words}), encoding="utf-8")
    print(f"  wrote {d / audio_id}.{{mp3,json,srt}} and word truth")


if __name__ == "__main__":
    main()
