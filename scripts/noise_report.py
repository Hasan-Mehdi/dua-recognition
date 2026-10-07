#!/usr/bin/env python
"""The noise work's bars (data/cache/noise/prereg.md), candidates side by side against a baseline.

    python scripts/noise_report.py noise_base_dev noise_base_dev:wpe noise_WhSh_dev [--split dev]

Rows: the decisive pool (measured halls + masjid), the synthetic pool (bench hall + far), then
each cell. Columns per result: on line, right word, du'a found <=10 s*, lost /10 min. The first
name is the baseline; a candidate's gain bars and screening guards are marked PASS/FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

POOLS = {"measured halls + masjid": ("rhall", "masjid"), "bench hall + far": ("hall", "far")}
GUARD_CELLS = ("room", "babble", "bgrecite", "fan")


FRONTS = ("_wpe", "_owpe")


def load(name: str, split: str) -> dict[str, list[dict]]:
    """name, or name:front (e.g. noise_base_dev:wpe): that run's cells without a front end, or only
    its copies through that front end, under the plain cell's name."""
    name, _, front = name.partition(":")
    sfx = f"_{front}" if front else ""
    items = json.loads((bench.BENCH / "results" / f"{name}.json").read_text(encoding="utf-8"))["items"]
    by: dict[str, list[dict]] = {}
    for m in items.values():
        sc = m["scenario"]
        if split != "all" and m.get("split") != split:
            continue
        if sfx:
            if not sc.endswith(sfx):
                continue
            sc = sc[: -len(sfx)]
        elif sc.endswith(FRONTS):
            continue
        by.setdefault(sc, []).append(m)
    return by


def stats(ms: list[dict]) -> dict:
    a = bench.aggregate(ms) if ms else {}
    return {k: a.get(k, float("nan")) for k in ("on_line", "word_exact", "found_d10s", "lost_10")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    runs = {n: load(n, args.split) for n in args.names}
    base = runs[args.names[0]]
    cells = [c for c in ("flow", "room", "babble", "bgrecite", "fan", "combo", "hall", "far", "rhall", "masjid",
                         "masjid_room", "masjid_pa") if any(c in r for r in runs.values())]
    rows = [(k, v) for k, v in POOLS.items()] + [(c, (c,)) for c in cells]
    head = "| | " + " | ".join(args.names) + " |"
    print(head)
    print("|---|" + "---:|" * len(args.names))
    for label, cs in rows:
        out = []
        for n in args.names:
            ms = [m for c in cs for m in runs[n].get(c, [])]
            if not ms:
                out.append("")
                continue
            s = stats(ms)
            out.append(f"{s['on_line']:.1%} / {s['word_exact']:.1%} / {s['found_d10s']:.0%} / {s['lost_10']:.1f}")
        print(f"| {label} | " + " | ".join(out) + " |")
    print("\n(on line / right word / found <=10 s* / lost per 10 min)\n")
    for n in args.names[1:]:
        r = runs[n]
        def pool(run, cs):
            return stats([m for c in cs for m in run.get(c, [])])
        b, c = pool(base, POOLS["measured halls + masjid"]), pool(r, POOLS["measured halls + masjid"])
        gains = {"on line +3": c["on_line"] - b["on_line"] >= 0.03, "word +3": c["word_exact"] - b["word_exact"] >= 0.03,
                 "found* +5": c["found_d10s"] - b["found_d10s"] >= 0.05}
        bs, cs_ = pool(base, POOLS["bench hall + far"]), pool(r, POOLS["bench hall + far"])
        same_dir = cs_["on_line"] >= bs["on_line"] - 0.005 and cs_["word_exact"] >= bs["word_exact"] - 0.005
        guards = {}
        if "flow" in r and "flow" in base:
            fb, fc = stats(base["flow"]), stats(r["flow"])
            guards["flow on line -0.5"] = fc["on_line"] >= fb["on_line"] - 0.005
            guards["flow word -1"] = fc["word_exact"] >= fb["word_exact"] - 0.01
            guards["flow found* -1"] = fc["found_d10s"] >= fb["found_d10s"] - 0.01
        for g in GUARD_CELLS:
            if g in r and g in base:
                guards[f"{g} on line -1"] = stats(r[g])["on_line"] >= stats(base[g])["on_line"] - 0.01
        print(f"{n} vs {args.names[0]}: gains " + ", ".join(f"{k} {'PASS' if v else 'fail'}" for k, v in gains.items())
              + f"; synthetic pool same direction {'PASS' if same_dir else 'FAIL'}; guards "
              + (", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in guards.items()) or "none scored"))


if __name__ == "__main__":
    main()
