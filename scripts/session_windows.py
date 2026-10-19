#!/usr/bin/env python
"""Training windows for the letter model from Hasan's debug sessions, as the harvest export makes them.

The session bench (scripts/bench_sessions.py) times each reading word by word with the teacher. This cuts
those items into 6 s windows every 2 s, the text of each being the words whose middle falls inside, and
leaves out windows that cut a word at either edge. Clips are int16 .npy; the rows go to
data/cache/finetune/<name>.jsonl, ready for scripts/ctc_teacher.py and train_ctc_student.py --add.

    python scripts/session_windows.py --before 20261003 --name sess_a     # sessions before a date
    python scripts/session_windows.py --since 20261003 --name sess_b

2026-10-19 (docs/results/ordinary_readers.md): 9 minutes of his sessions, 20 times over, took the stream
student from 44.9% to 33.9% letter error on those sessions but only 46.2% to 45.0% on later ones.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import bench  # noqa: E402
from dua_recognition.align import CorpusIndex  # noqa: E402
from dua_recognition.corpus import load_all  # noqa: E402

SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="data/cache/finetune/<name>.jsonl")
    ap.add_argument("--items", default=str(ROOT / "data/testbed/sessions/items/session"))
    ap.add_argument("--before", default="", help="only sessions recorded before this date (YYYYMMDD)")
    ap.add_argument("--since", default="", help="only sessions recorded on or after this date")
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=2.0)
    args = ap.parse_args()
    ix = CorpusIndex(load_all())
    did = {d: i for i, d in enumerate(ix.dua_ids)}
    out = ROOT / "data" / "cache" / "session_clips" / args.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(str(Path(args.items) / "*.json"))):
        it = json.loads(Path(f).read_text(encoding="utf-8"))
        sid = it["sid"].split(":")[1]
        if (args.before and sid >= args.before) or (args.since and sid < args.since):
            continue
        y = bench.render(it)
        words = [(ix.words[ix.dua_word_span[did[d]][0] + int(w)].text, a, b) for d, w, a, b in it["words"]]
        t, k = 0.0, 0
        while t + 1.0 < len(y) / SR:
            e = min(t + args.window, len(y) / SR)
            inside = [w for w, a, b in words if t <= (a + b) / 2 < e]
            cut = any(a < t < b or a < e < b for w, a, b in words if not t <= (a + b) / 2 < e)
            if inside and not cut:
                p = out / f"{it['id']}_{k:03d}.npy"
                np.save(p, (np.clip(y[int(t * SR) : int(e * SR)], -1, 1) * 32767).astype(np.int16))
                rows.append({"clip": str(p.relative_to(ROOT)), "text": " ".join(inside), "reciter": it.get("voice", "user"),
                             "sid": sid, "dua": it["dua"], "dur": round(e - t, 2)})
                k += 1
            t += args.hop
    (ROOT / "data" / "cache" / "finetune" / f"{args.name}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{args.name}: {len(rows)} windows, {len({r['sid'] for r in rows})} sessions, "
          f"{sum(r['dur'] for r in rows) / 60:.1f} min of windows")


if __name__ == "__main__":
    main()
