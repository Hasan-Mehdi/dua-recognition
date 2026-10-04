"""Where the display's jumps and early moves come from (docs/results/jumps.md).

Replays bench items with the stream decoder (bench.py score --display stream, same scoring) and
sorts every counted jump and early move by what the display did at that moment: crossed into the
next line (its first word or a later one), moved inside its line, went back, skipped ahead, or
changed du'a. The decoder takes the same branch as the move's shape (stream_follower.py _step:
same line / next line / any other line; a du'a change comes from _start), so this is the branch.

    python scripts/jump_diag.py --split dev                       # shipped settings
    python scripts/jump_diag.py --split dev --sc next_margin=3    # a candidate
    python scripts/jump_diag.py --split dev --lane user --show 20 # list the moments
"""

from __future__ import annotations

import argparse
import collections
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402


def _kv(spec: str, ints=()) -> dict:
    import json

    out = {}
    for a in (x for x in spec.split(",") if x):
        k, v = a.split("=")
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        elif ";" in v:
            out[k] = tuple(float(x) for x in v.split(";"))
        else:
            out[k] = int(v) if k in ints else json.loads(v)
    return out


def shape(ix, a, b) -> str:
    """The display's move from shown word a to shown word b (global word indices)."""
    if a is None:
        return "appear"
    if ix.word_dua[a] != ix.word_dua[b]:
        return "dua change"
    ln = bench._line_no(ix)
    d = int(ln[b] - ln[a])
    if d == 0:
        return "same line"
    if d == 1:
        return "next line, 1st word" if ln[b - 1] != ln[b] else "next line, later word"
    if d < 0:
        return f"back {-d}" if d >= -3 else "back 4+"
    return f"skip {d - 1}" if d <= 4 else "skip 4+"


def _one(it: dict):
    shown = bench.replay(it)
    if shown is None:
        return None
    m = bench.metrics(it, shown, bench._W["ix"])
    ix = bench._W["ix"]
    out = []
    for i, kind in m.get("moves", []):
        if kind == "far":
            continue
        prev = next((shown[j][2] for j in range(i - 1, -1, -1) if shown[j] is not None), None)
        out.append((kind, shape(ix, prev, shown[i][2]), round(i * 0.1, 1)))
    return it["id"], it["scenario"], it["lane"], it["sid"], it["duration"], out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="*")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--lane")
    ap.add_argument("--sc", default="", help="StreamConfig overrides k=v,... (as bench.py score)")
    ap.add_argument("--tracker", default="", help="TrackerConfig overrides k=v,...")
    ap.add_argument("--same-text", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--show", type=int, default=0, help="list this many moments (item, time, kind, shape)")
    args = ap.parse_args()

    items = bench.load_items(args.scenarios or None, split=args.split)
    if args.lane:
        items = [i for i in items if i["lane"] == args.lane]
    sc_kw = _kv(args.sc, ints=("line_steps", "next_steps", "gate_words"))
    init = (_kv(args.tracker), {}, bench.ASR_TAG, bench.CTC_TAG, 1.2, 0.15, "stream", sc_kw, args.same_text)
    rows = []
    with Pool(args.workers, initializer=_init, initargs=init) as pool:
        for r in pool.imap_unordered(_one, items, chunksize=1):
            if r is not None:
                rows.append(r)
    minutes = sum(r[4] for r in rows) / 60
    by = {k: collections.Counter() for k in ("jump", "early")}
    by_cell = collections.defaultdict(collections.Counter)
    moments = []
    for iid, scen, lane, sid, _, moves in rows:
        for kind, shp, t in moves:
            by[kind][shp] += 1
            by_cell[(kind, scen)][shp] += 1
            moments.append((iid, t, kind, shp))
    print(f"{len(rows)} items, {minutes:.0f} min, sc {sc_kw or 'shipped'}, tracker {args.tracker or 'shipped'}")
    for kind in ("jump", "early"):
        n = sum(by[kind].values())
        print(f"\n{kind}: {n} ({n / max(minutes, 1e-9) * 10:.2f} /10 min), by what the display did")
        for shp, c in by[kind].most_common():
            print(f"  {shp:22s} {c:5d}  {c / max(n, 1):5.0%}")
    print("\nper scenario (jump / early: the commonest shape)")
    for scen in sorted({s for _, s in by_cell}):
        parts = []
        for kind in ("jump", "early"):
            c = by_cell.get((kind, scen))
            if c:
                shp, k = c.most_common(1)[0]
                parts.append(f"{kind} {sum(c.values())} ({k} {shp})")
        print(f"  {scen:10s} " + ", ".join(parts))
    for iid, t, kind, shp in sorted(moments)[: args.show]:
        print(f"  {iid:24s} t={t:6.1f}  {kind:5s} {shp}")


def _init(*a):
    bench._worker_init(*a)
    bench._W["keep_moves"] = True


if __name__ == "__main__":
    main()
