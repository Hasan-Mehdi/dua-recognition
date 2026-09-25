#!/usr/bin/env python
"""Flag recordings whose slide timings don't match the text they label.

DuaPlayer's timings were recorded against a version of each du'a's slides; if
the text was later re-split, the labels drift. A tracker error doesn't look
like a *constant* offset held for minutes, but a timing-file mismatch does. So
for each recording we follow it with the du'a given and report the median
(predicted line - labelled line) over each quarter of the recording.

    python scripts/audit_labels.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import load_rows, run_tracker  # noqa: E402

from dua_recognition.align import CorpusIndex  # noqa: E402
from dua_recognition.corpus import load_all, load_recordings  # noqa: E402
from dua_recognition.tracker import TrackerConfig  # noqa: E402


def audit(asr: str = "large-v3-turbo") -> list[tuple[str, str, list[float]]]:
    duas = load_all()
    out = []
    for dua in duas.values():
        ix = CorpusIndex({dua.id: dua})
        for rec in load_recordings(dua):
            rows = load_rows(asr, rec, 6.0, 1.0)
            if not rows:
                continue
            costs = [ix.word_costs(t) if t else None for _, t in rows]
            preds = run_tracker(ix, TrackerConfig(), costs, 1.0)
            diffs = [(t, s - rec.segment_at(t)) for (t, _), (_, s) in zip(rows, preds)
                     if s is not None and rec.segment_at(t) is not None]
            q = len(diffs) // 4
            quarters = [statistics.median(d for _, d in diffs[i * q : (i + 1) * q]) for i in range(4)] if q else []
            out.append((dua.id, rec.reciter, quarters))
    return out


def repair(dua_id: str, asr: str = "large-v3-turbo") -> None:
    """Keep the human boundary times; re-derive which line starts at each.

    For every timing key, the offline smoother says which line is recited
    between it and the next key; the mapping is forced monotone. Written next
    to the original as <audio>.repaired.json, used only on request.
    """
    import collections
    import json

    from dua_recognition.offline import align_recording

    dua = load_all()[dua_id]
    ix = CorpusIndex({dua.id: dua})
    for rec in load_recordings(dua):
        meta_path = rec.path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        keys = sorted((int(k), v / 1000) for k, v in meta["slide_start_ms"].items())
        rows = load_rows(asr, rec, 6.0, 1.0)
        costs = [ix.word_costs(t) if t else None for _, t in rows]
        segs = align_recording(ix, costs, 1.0)
        heard = [(t - 1.0, int(s)) for (t, _), s in zip(rows, segs)]
        starts, last = {}, 0
        for (_, t0), (_, t1) in zip(keys, keys[1:] + [(None, rec.duration_s)]):
            inside = [s for t, s in heard if t0 <= t < t1]
            line = collections.Counter(inside).most_common(1)[0][0] if inside else last
            if line > last:
                starts[line] = round(t0 * 1000)
                last = line
        starts[len(dua.segments) + 1] = round(keys[-1][1] * 1000) if keys[-1][0] > len(dua.segments) else round(rec.duration_s * 1000)
        meta_path.with_name(meta_path.stem + ".repaired.json").write_text(
            json.dumps({"slide_start_ms": {str(k): v for k, v in sorted(starts.items())}}, indent=1), encoding="utf-8")
        print(f"repaired {dua_id} / {rec.reciter}: {len(starts) - 1} of {len(dua.segments)} lines have a start")


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--repair":
        repair(sys.argv[2])
        return
    print("| du'a | reciter | median offset per quarter (lines) | verdict |\n|---|---|---|---|")
    for dua_id, reciter, quarters in audit():
        # Sustained: the same non-zero offset in two consecutive quarters.
        bad = any(a == b != 0 for a, b in zip(quarters, quarters[1:]))
        verdict = "**labels drift**" if bad else "ok"
        print(f"| {dua_id} | {reciter} | {' / '.join(f'{x:+.0f}' for x in quarters)} | {verdict} |")


if __name__ == "__main__":
    main()
