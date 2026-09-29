#!/usr/bin/env python
"""Freeze development vs final-test groups for the held-out sets, before scoring any candidate.

The DuaPlayer test reciters have informed many decisions: they stay a historical
regression suite, not fresh evidence (Dwork et al. 2015, adaptive holdout reuse).
The new sets (corpus.TESTSETS: user, majlis, amateur) are split by *group*: the
majlis venue (a YouTube channel, so its re-uploads and every stream from it stay
together) or, for ordinary voices, the uploader/venue if known, else the person.
A group is assigned once and never moved; new groups are appended. Groups already
examined during development (e.g. Hasan's own clips) go to development.

    python scripts/testset_split.py           # assign new groups, print the split
    python scripts/testset_split.py --show

Writes data/testsets/split.json (gitignored with the audio: it names private uploaders).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.corpus import TESTSETS, load_all, load_recordings  # noqa: E402
from dua_recognition.splits import TEST_CHANNELS  # noqa: E402

SPLIT = ROOT / "data" / "testsets" / "split.json"
DEV_SHARE = 0.4  # of groups per set; the rest is final test
# Looked at while building the app (STATUS.md: my_test.mp4, test3.mp4): never final test.
EXAMINED = {("user", "person:user:hasan")}


def group_of(set_name: str, rec) -> str:
    """Majlis spans: their YouTube channel (segment_streams.py writes channel_id), so every
    stream and re-upload of a venue stays together; the venue name if no id was stored."""
    meta_f = Path(rec.path).with_suffix(".json")
    meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
    if meta.get("channel_id"):
        return f"venue:{meta['channel_id']}"
    if set_name == "majlis":
        return f"venue:{rec.venue or rec.reciter}"
    return f"venue:{rec.venue}" if rec.venue else f"person:{rec.reciter}"


def rank(set_name: str, group: str) -> str:
    return hashlib.sha256(f"{set_name}|{group}".encode()).hexdigest()


def assign(split: dict, set_name: str, groups: list[str]) -> list[str]:
    """Assign the groups not yet in `split`: keep each set's dev share near DEV_SHARE,
    new groups in a fixed hash order (no outcome can influence it)."""
    s = split.setdefault(set_name, {})
    new = sorted((g for g in groups if g not in s), key=lambda g: rank(set_name, g))
    for g in new:
        if (set_name, g) in EXAMINED:
            s[g] = {"part": "dev", "why": "examined during development"}
            continue
        n_dev = sum(v["part"] == "dev" for v in s.values())
        part = "dev" if n_dev < DEV_SHARE * (len(s) + 1) - 1e-9 else "final"
        s[g] = {"part": part, "why": "hash order", "assigned": datetime.date.today().isoformat()}
    return new


def load_split() -> dict:
    return json.loads(SPLIT.read_text(encoding="utf-8")) if SPLIT.exists() else {}


def part_of(set_name: str, rec, split: dict | None = None) -> str | None:
    """'dev', 'final', or None (group not assigned yet: score nothing on it)."""
    split = load_split() if split is None else split
    g = split.get(set_name, {}).get(group_of(set_name, rec))
    return g["part"] if g else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    split = load_split()
    duas = load_all()
    recs = {name: [r for d in duas.values() for r in load_recordings(d, cache_dir=path, extra=False)]
            for name, path in TESTSETS.items() if path.exists()}
    if not args.show:
        # Majlis venues are known in advance: assign them before any span exists.
        assign(split, "majlis", [f"venue:{v}" for v in TEST_CHANNELS])
        for name, rs in recs.items():
            added = assign(split, name, sorted({group_of(name, r) for r in rs}))
            if added:
                print(f"{name}: assigned {len(added)} new group(s)")
        SPLIT.parent.mkdir(parents=True, exist_ok=True)
        SPLIT.write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, groups in split.items():
        for g, v in sorted(groups.items()):
            n = [r for r in recs.get(name, []) if group_of(name, r) == g]
            print(f"{name:8s} {v['part']:5s} {g[:60]:60s} {len(n):3d} rec {sum(r.end_s for r in n) / 3600:5.2f} h "
                  f"({v['why']})")


if __name__ == "__main__":
    main()
