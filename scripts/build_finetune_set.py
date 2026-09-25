#!/usr/bin/env python
"""Build (window, text) training pairs from the train reciters.

The labels are reference-snapped pseudo-labels. For each cached window:

  1. the human slide timings say which lines the window overlaps;
  2. the teacher's transcript of that window (large-v3-turbo, from
     scripts/transcribe_windows.py) is aligned against just those lines;
  3. the label is the *reference* text between the aligned start and end.

So the teacher only decides where the window's audio begins and ends in the
text — its spelling mistakes never reach the labels — and the targets look
exactly like what the live recognizer is asked to transcribe: arbitrary 6 s
cuts, mid-word starts and all. Windows the teacher can't place confidently
are dropped.

    python scripts/build_finetune_set.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from dua_recognition.align import encode, semiglobal_end_costs  # noqa: E402
from dua_recognition.corpus import load_all, load_recordings  # noqa: E402
from dua_recognition.splits import is_test  # noqa: E402
from dua_recognition.text import normalize, strip_diacritics  # noqa: E402
from align_offline import load_youtube  # noqa: E402
from evaluate import WINDOWS, load_rows  # noqa: E402

from dua_recognition.asr import _looks_hallucinated  # noqa: E402
from dua_recognition.splits import LINE_LABELS_UNRELIABLE  # noqa: E402

OUT = ROOT / "data" / "cache" / "finetune"
# Never trained on, by anyone: measures whether the fine-tuned model learned to
# hear recitation or just memorized these texts. Both have test reciters.
HELD_OUT_DUAS = frozenset({"dua-tawassul", "ziyarat-ashura"})
_NOT_LETTER = re.compile(r"[^ء-ي\s]")
YT_HOP = 2.0


def load_yt_rows(teacher: str, rec) -> list[tuple[float, str]]:
    path = WINDOWS / teacher / f"{rec.audio_id}_w6_h{YT_HOP:g}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [(r["t"], "" if _looks_hallucinated(r["text"]) else r["text"]) for r in rows]


def whisper_style(token: str) -> str:
    """Reference token -> the undiacritized spelling Whisper writes."""
    t = strip_diacritics(token).replace("ٱ", "ا")
    return _NOT_LETTER.sub("", t)


def snap(hyp: str, words: list[str]) -> tuple[int, int, float] | None:
    """Best reference word span [i, j] for hyp, and cost per letter."""
    h = encode(hyp)
    if h.size < 4:
        return None
    codes = [encode(w) for w in words]
    letter_word = np.concatenate([np.full(c.size, i) for i, c in enumerate(codes)])
    r = np.concatenate(codes)
    end_costs = semiglobal_end_costs(h, r)
    e = int(end_costs.argmin())
    # Start: the same alignment run backwards over the reversed strings.
    start_costs = semiglobal_end_costs(h[::-1], r[: e + 1][::-1])
    s = e - int(start_costs.argmin())
    return int(letter_word[s]), int(letter_word[e]), float(end_costs[e]) / h.size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="large-v3-turbo")
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=2, help="keep every Nth window")
    ap.add_argument("--max-cost", type=float, default=0.35, help="edits per letter")
    ap.add_argument("--val", type=float, default=0.04)
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="test: build an ASR benchmark (windows + snapped references) from test reciters")
    ap.add_argument("--youtube", action="store_true", help="also use YouTube recordings (silver labels)")
    args = ap.parse_args()
    test_mode = args.split == "test"

    rng = random.Random(0)
    rows_out, stats = [], {"kept": 0, "empty": 0, "unplaced": 0}
    for dua in load_all().values():
        if dua.id in HELD_OUT_DUAS and not test_mode:
            continue
        if dua.id in LINE_LABELS_UNRELIABLE:
            continue
        seg_tokens = {s.id: s.arabic.split() for s in dua.segments}
        recs = [(r, args.hop) for r in load_recordings(dua) if is_test(r.reciter) == test_mode]
        if args.youtube and not test_mode:
            recs += [(r, YT_HOP) for r in load_youtube(dua)]
        for rec, hop in recs:
            rows = load_yt_rows(args.teacher, rec) if rec.audio_id.startswith("yt-") else                 load_rows(args.teacher, rec, args.window, hop)[:: args.every]
            for t, text in rows:
                t0 = max(0.0, t - args.window)
                if not text:
                    stats["empty"] += 1
                    continue
                # Lines the window overlaps, padded by one each side for label slop.
                first, last = rec.segment_at(t0), rec.segment_at(min(t, rec.end_s - 1e-3))
                if first is None or last is None:
                    continue
                ids = [i for i in range(first - 1, last + 2) if i in seg_tokens]
                tokens = [tok for i in ids for tok in seg_tokens[i] if whisper_style(tok)]
                norm = [normalize(whisper_style(tok)) for tok in tokens]
                placed = snap(text, norm)
                if placed is None or placed[2] > args.max_cost:
                    stats["unplaced"] += 1
                    continue
                i, j, _ = placed
                label = " ".join(whisper_style(tok) for tok in tokens[i : j + 1])
                rows_out.append({"audio": str(rec.path), "start": round(t0, 3), "end": t,
                                 "text": label, "dua": dua.id, "reciter": rec.reciter})
                stats["kept"] += 1

    rng.shuffle(rows_out)
    n_val = 0 if test_mode else int(len(rows_out) * args.val)
    OUT.mkdir(parents=True, exist_ok=True)
    parts = (("test", rows_out),) if test_mode else (("val", rows_out[:n_val]), ("train", rows_out[n_val:]))
    for name, part in parts:
        (OUT / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in part), encoding="utf-8")
    hours = sum(r["end"] - r["start"] for r in rows_out) / 3600
    print(f"{stats} -> {len(rows_out) - n_val} {args.split} / {n_val} val windows ({hours:.1f} h of audio)")


if __name__ == "__main__":
    main()
