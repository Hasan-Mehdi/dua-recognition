#!/usr/bin/env python
"""Apply the text audit's fixes (docs/results/text_audit.md) to the du'a texts.

data/cache/text_audit/fixes.json lists one edit per line: {dua, segment_id, old_arabic,
new_arabic, confidence, evidence}. Each edit replaces one segment's Arabic and nothing else:
no segment is added, removed or renumbered (segment ids key DuaPlayer's timings and the bench's
truth). An edit whose old_arabic no longer matches the file (someone changed the line since the
audit) is skipped and reported.

    python scripts/text_fixes.py --dry-run                 # what would change, word by word
    python scripts/text_fixes.py                           # apply "sure" and "likely"
    python scripts/text_fixes.py --confidence sure         # only "sure"
    python scripts/text_fixes.py --dir COPY                # apply to a copy of data/duas
    python scripts/text_fixes.py --only ziyarat-warith:32  # one line (dua:segment_id, repeatable)

"needs Hasan" edits are applied only when named with --only. After applying to data/duas:
rerun scripts/export_web.py (web/corpus.json). A line whose word count changes drops the bench
sources that read it (bench.fresh_source) and leaves its word truth stale; fixes.json's
"impact" field says which.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.align import encode  # noqa: E402
from dua_recognition.text import normalize  # noqa: E402

FIXES = ROOT / "data" / "cache" / "text_audit" / "fixes.json"
LEVELS = {"sure": 0, "likely": 1, "needs Hasan": 2}


def words(arabic: str) -> list[str]:
    """A line's words as the corpus index counts them (align.CorpusIndex)."""
    return [w for tok in arabic.split() for w in normalize(tok).split() if encode(w).size]


def load_fixes(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["fixes"] if isinstance(data, dict) else data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixes", type=Path, default=FIXES)
    ap.add_argument("--dir", type=Path, default=ROOT / "data" / "duas", help="the texts to edit (default data/duas)")
    ap.add_argument("--confidence", choices=["sure", "likely"], default="likely",
                    help="apply edits at least this confident (default: sure and likely)")
    ap.add_argument("--only", action="append", default=[], metavar="DUA:SEG",
                    help="apply just these lines, whatever their confidence")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # (Arabic on a Windows console)

    fixes = load_fixes(args.fixes)
    only = {(o.rsplit(":", 1)[0], int(o.rsplit(":", 1)[1])) for o in args.only}
    if only:
        chosen = [f for f in fixes if (f["dua"], f["segment_id"]) in only]
        missing = only - {(f["dua"], f["segment_id"]) for f in chosen}
        for d, s in sorted(missing):
            print(f"not in {args.fixes.name}: {d}:{s}")
    else:
        chosen = [f for f in fixes if LEVELS.get(f["confidence"], 9) <= LEVELS[args.confidence]]

    by_file: dict[str, list[dict]] = {}
    for f in chosen:
        by_file.setdefault(f["dua"], []).append(f)
    applied = skipped = 0
    count_changes = []
    for dua, fs in sorted(by_file.items()):
        path = args.dir / f"{dua}.json"
        if not path.exists():
            print(f"{dua}: no {path}")
            skipped += len(fs)
            continue
        text = path.read_text(encoding="utf-8")
        raw = json.loads(text)
        segs = {s["segment_id"]: s for s in raw["segments"]}
        changed = False
        for f in sorted(fs, key=lambda f: f["segment_id"]):
            seg = segs.get(f["segment_id"])
            tag = f"{dua}:{f['segment_id']} [{f['confidence']}]"
            if seg is None:
                print(f"{tag}: no such segment, skipped")
                skipped += 1
                continue
            if seg["arabic"] == f["new_arabic"]:
                print(f"{tag}: already applied")
                continue
            if seg["arabic"] != f["old_arabic"]:
                print(f"{tag}: the line changed since the audit, skipped\n   now:      {seg['arabic']}\n"
                      f"   expected: {f['old_arabic']}")
                skipped += 1
                continue
            wo, wn = words(f["old_arabic"]), words(f["new_arabic"])
            print(f"{tag}: {len(wo)} -> {len(wn)} words\n   - {' '.join(wo)}\n   + {' '.join(wn)}")
            if len(wo) != len(wn):
                count_changes.append(tag)
            seg["arabic"] = f["new_arabic"]
            changed = True
            applied += 1
        if changed and not args.dry_run:
            out = json.dumps(raw, ensure_ascii=False, indent=1)
            if text.endswith("\n"):
                out += "\n"
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(out, encoding="utf-8")
            tmp.replace(path)
    verb = "would apply" if args.dry_run else "applied"
    print(f"\n{verb} {applied}, skipped {skipped}; {len(count_changes)} change a line's word count")
    if applied and not args.dry_run:
        print("next: python scripts/export_web.py   (web/corpus.json), then the bench and word truth as fixes.json's"
              " impact notes say")


if __name__ == "__main__":
    main()
