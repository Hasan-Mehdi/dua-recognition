#!/usr/bin/env python
"""Carry the scenario bench's items over a change to the du'a texts (data/duas).

Bench items (data/testbed/items/<cell>/*.json) name the reader's words by their place in the du'a
("words": [du'a, word, start, end]; events' and practice errors' "word": [du'a, word]). A line
that gains or loses words shifts every later word of that du'a, so after a text change the items
would score against the wrong words. This maps each word through (line id, place in the line) from
the old texts to the new ones, rewrites the items whose places moved, and moves aside the items
that read a line whose word count changed (their timings were cut for the old words): to
data/testbed/items_retired/<tag>/<cell>/, not deleted. Line ids never change (text_fixes.py edits
one line's Arabic in place), so line-level fields (skipline, ending, target) stay as they are.

    python scripts/bench_retext.py --old data/cache/text_audit/duas_before --dry-run
    python scripts/bench_retext.py --old data/cache/text_audit/duas_before --tag text-audit-2026-10-09

The tracker cache keys on the corpus words, so the next score recomputes the tracker (about an hour
for dev): score a fresh baseline before comparing anything to runs made before the change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.align import CorpusIndex  # noqa: E402
from dua_recognition.corpus import load_all  # noqa: E402

ITEMS = ROOT / "data" / "testbed" / "items"
RETIRED = ROOT / "data" / "testbed" / "items_retired"


def word_maps(old: CorpusIndex, new: CorpusIndex) -> dict[str, dict[int, int | None]]:
    """Per du'a whose words moved: old word (place in the du'a) -> new word, None in a line whose word count changed."""
    out = {}
    for dua in old.dua_ids:
        if dua not in new.dua_ids:
            continue
        lines = []
        for ix in (old, new):
            lo, hi = ix.dua_word_span[ix.dua_ids.index(dua)]
            segs = [int(s) for s in ix.word_segment[lo:hi]]
            first, count = {}, {}
            for w, s in enumerate(segs):
                first.setdefault(s, w)
                count[s] = count.get(s, 0) + 1
            lines.append((segs, first, count))
        (so, fo, co), (_, fn, cn) = lines
        m = {}
        for w, s in enumerate(so):
            m[w] = fn[s] + (w - fo[s]) if s in cn and cn[s] == co[s] else None
        if any(v != k for k, v in m.items()):
            out[dua] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path, required=True, help="a copy of data/duas from before the change")
    ap.add_argument("--new", type=Path, default=ROOT / "data" / "duas", help="the texts after it (default data/duas)")
    ap.add_argument("--tag", default="retext", help="folder name under items_retired/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    maps = word_maps(CorpusIndex(load_all(args.old)), CorpusIndex(load_all(args.new)))
    print(f"{len(maps)} du'as whose words moved: {', '.join(sorted(maps))}")
    moved = rewritten = 0
    by_cell: dict[str, int] = {}
    for f in sorted(ITEMS.glob("*/*.json")):
        it = json.loads(f.read_text(encoding="utf-8"))
        refs = [w for w in it.get("words", [])]
        refs += [e["word"] for e in it.get("events", []) if e.get("word")]
        refs += [e["word"] for e in it.get("errors", []) if e.get("word")]
        if not any(r[0] in maps for r in refs):
            continue
        if any(r[0] in maps and maps[r[0]].get(int(r[1])) is None for r in refs):
            moved += 1
            by_cell[f.parent.name] = by_cell.get(f.parent.name, 0) + 1
            if not args.dry_run:
                dest = RETIRED / args.tag / f.parent.name / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                f.replace(dest)
            continue
        for r in refs:  # the lists inside `it`, changed in place
            if r[0] in maps:
                r[1] = maps[r[0]][int(r[1])]
        rewritten += 1
        if not args.dry_run:
            tmp = f.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(it, ensure_ascii=False), encoding="utf-8")
            tmp.replace(f)
    verb = "would move" if args.dry_run else "moved"
    print(f"{verb} {moved} items aside ({', '.join(f'{c} {n}' for c, n in sorted(by_cell.items()))}); "
          f"{'would rewrite' if args.dry_run else 'rewrote'} {rewritten}")


if __name__ == "__main__":
    main()
