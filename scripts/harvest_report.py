#!/usr/bin/env python
"""How much the harvest holds, and how far its labels can be trusted.

    python scripts/harvest_report.py [--md docs/results/data_harvest_numbers.md]

  totals      per platform: downloaded, letter posteriors done, labelled, matched
              to a corpus text, usable (aligned lines that passed the forced-alignment
              check), voice-flagged (a test reciter's voice: never exported)
  identify    shiavoice files every recording under the text it holds; for the
              categories that are one corpus text, how often the labeller's main
              span names that text
  timing      recordings with human caption cues AND blind labels: blind line
              starts vs the human cue starts (good lines only)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from harvest_label import FRAMES, HARVEST, LABELS, PLATFORMS, read_jsonl  # noqa: E402

# shiavoice category -> corpus texts that count as right (same text under two ids too).
SV_TRUTH = {
    "دعاء كميل": {"dua-kumayl"}, "دعاء الندبة": {"dua-nudbah"}, "زيارة وارث": {"ziyarat-warith"},
    "زيارة أمين الله": {"ziyarat-aminallah-imam-ali", "duasorg-ziyarat-ameenallah-2"},
    "دعاء الافتتاح": {"dua-iftitah"}, "دعاء البهاء": {"dua-baha"}, "دعاء ياعدتي": {"dua-ya-uddati"},
    "المناجاة الشعبانية": {"duasorg-munajat-shabaniyah"}, "دعاء العهد": {"dua-aahad"},
    "دعاء السمات": {"dua-simaat"}, "حديث الكساء": {"dua-hadithkisa"},
    "دعاء أبي حمزة الثمالي": {"dua-abu-hamza-thumali"}, "زيارة عاشوراء": {"ziyarat-ashura"},
    "دعاء الجوشن الكبير": {"dua-jawshan-kabir"}, "دعاء عرفة": {"dua-arafat"},
    "زيارة آل ياسين": {"ziyarat-aal-yaseen"}, "الزيارة الجامعة الكبيرة": {"duasorg-ziyarat-jamia-kabeera"},
    "دعاء مكارم الأخلاق": {"dua-makaramakhlaq"}, "دعاء التوسل": {"dua-tawassul"},
    "الصلوات الشعبانية": {"duasorg-salawat-shabaniyah"}, "دعاء المجير": {"dua-mujeer"},
    "دعاء يستشير": {"duasorg-dua-yastasheer"}, "دعاء المشلول": {"dua-mashlool"}, "دعاء التوبة": {"dua-tawba"},
    "دعاء رفع المصاحف": {"dua-amaal-quran"}, "دعاء علقمه": {"duasorg-dua-alqama-after-ziyarat-ashura-imam-husain"},
    "دعاء أم داود": {"duasorg-aamaal-umm-e-dawood-1"}, "دعاء أهل الثغور": {"sahifa-27"},
    "دعاء يوم السبت": {"dua-saturday"}, "دعاء يوم الجمعة": {"dua-friday"}, "دعاء يوم الأحد": {"dua-sunday"},
    "دعاء يوم الإثنين": {"dua-monday"}, "دعاء يوم الثلاثاء": {"dua-tuesday"}, "دعاء يوم الأربعاء": {"dua-wednesday"},
    "دعاء يوم الخميس": {"dua-thursday"},
}


def _hours(xs) -> float:
    return sum(xs) / 3600


def totals(lines: list[str]) -> None:
    voices = {(r["platform"], r["id"]): r.get("verdict") for r in read_jsonl(HARVEST / "voices.jsonl")}
    lines += ["| platform | recordings | hours | posteriors | labelled | matched h | usable h | test voice |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    grand = Counter()
    for p in PLATFORMS:
        idx = {}
        for m in read_jsonl(HARVEST / p / "index.jsonl"):
            idx.setdefault(m["id"], m)
        if not idx:
            continue
        dur = _hours(float(m.get("duration") or 0) for m in idx.values())
        fr = len(list((FRAMES / p).glob("*.npz"))) if (FRAMES / p).exists() else 0
        labs = [json.loads(f.read_text(encoding="utf-8")) for f in (LABELS / p).glob("*.json")
                if not f.name.endswith(".captions.json")] if (LABELS / p).exists() else []
        flagged = sum(1 for k in idx if voices.get((p, k)) == "test")
        matched = _hours(r["matched_s"] for r in labs)
        usable = _hours(r["usable_s"] for r in labs if voices.get((p, r["id"])) != "test")
        lines.append(f"| {p} | {len(idx)} | {dur:.0f} | {fr} | {len(labs)} | {matched:.0f} | {usable:.0f} | {flagged} |")
        grand.update(n=len(idx), h=dur, fr=fr, lab=len(labs), m=matched, u=usable, t=flagged)
    lines.append(f"| **all** | {grand['n']} | {grand['h']:.0f} | {grand['fr']} | {grand['lab']} | {grand['m']:.0f} | "
                 f"{grand['u']:.0f} | {grand['t']} |")
    caps = [json.loads(f.read_text(encoding="utf-8")) for f in LABELS.glob("*/*.captions.json")]
    lines.append(f"\nHuman caption tracks: {len(caps)} recordings, {sum(c['usable'] for c in caps)} usable, "
                 f"{_hours(c['usable_s'] for c in caps if c['usable']):.1f} h of human-timed lines.")


def identify(lines: list[str]) -> None:
    idx = {m["id"]: m for m in read_jsonl(HARVEST / "web" / "index.jsonl")}
    per = defaultdict(Counter)
    for f in (LABELS / "web").glob("*.json") if (LABELS / "web").exists() else []:
        r = json.loads(f.read_text(encoding="utf-8"))
        m = idx.get(r["id"])
        if not m:
            continue
        cat = m["title"].split(" / ")[0]
        if cat not in SV_TRUTH:
            continue
        if not r["spans"]:
            per[cat]["none"] += 1
            continue
        main = max(r["spans"], key=lambda s: s["t1"] - s["t0"])
        per[cat]["right" if main["dua"] in SV_TRUTH[cat] else "wrong"] += 1
        if main["dua"] in SV_TRUTH[cat] and main["usable"]:
            per[cat]["usable"] += 1
    tot = Counter()
    for c in per.values():
        tot.update(c)
    n = sum(tot[k] for k in ("right", "wrong", "none"))
    if not n:
        return
    lines += ["", f"Identification on shiavoice ({n} recordings in {len(per)} one-text categories): "
                  f"main span names the category's text {tot['right'] / n:.1%}, another text {tot['wrong'] / n:.1%}, "
                  f"nothing {tot['none'] / n:.1%}.", "",
              "| category | n | right | wrong | none |", "|---|---:|---:|---:|---:|"]
    for cat, c in sorted(per.items(), key=lambda kv: -sum(kv[1].values()))[:20]:
        k = c["right"] + c["wrong"] + c["none"]
        lines.append(f"| {cat} | {k} | {c['right']} | {c['wrong']} | {c['none']} |")


def timing(lines: list[str]) -> None:
    errs = []
    n_rec = 0
    for cf in LABELS.glob("*/*.captions.json"):
        cap = json.loads(cf.read_text(encoding="utf-8"))
        bf = cf.with_name(cf.name.replace(".captions.json", ".json"))
        if not cap["usable"] or not bf.exists():
            continue
        blind = json.loads(bf.read_text(encoding="utf-8"))
        # One-to-one pairs only: a cue holding the same words as the corpus line (caption
        # authors often put two corpus lines in one cue), whose own text force-aligns there.
        cues = [(ln["cue"][0], set(ln["words"]), len(ln["words"])) for ln in cap["lines"] if ln["ok"]]
        k = 0
        for sp in blind["spans"]:
            for ln in sp["lines"]:
                if not ln["ok"]:
                    continue
                w = set(ln["words"])
                cand = [c for c in cues if len(w & c[1]) >= 0.8 * max(len(w), len(c[1]))]
                if cand:
                    best = min(cand, key=lambda c: abs(c[0] - ln["start"]))
                    errs.append(ln["start"] - best[0])
                    k += 1
        n_rec += bool(k)
    if not errs:
        return
    e = np.abs(np.array(errs))
    lines += ["", f"Blind line starts vs human caption cues ({len(errs)} lines in {n_rec} recordings): "
                  f"median |error| {np.median(e):.2f} s, within 0.5 s {np.mean(e < 0.5):.1%}, "
                  f"within 1 s {np.mean(e < 1):.1%}."]


def texts(lines: list[str]) -> None:
    voices = {(r["platform"], r["id"]): r.get("verdict") for r in read_jsonl(HARVEST / "voices.jsonl")}
    per = Counter()
    recs = Counter()
    for f in LABELS.glob("*/*.json"):
        if f.name.endswith(".captions.json"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        if voices.get((r["platform"], r["id"])) == "test":
            continue
        for sp in r["spans"]:
            if sp["usable"] and not sp.get("extra"):
                per[sp["dua"]] += sum(ln["end"] - ln["start"] for ln in sp["lines"] if ln["ok"])
                recs[sp["dua"]] += 1
    # Voices: distinct uploaders (channels / reciters) with usable audio, and the ordinary-voice searches.
    up, groups = Counter(), Counter()
    for p in PLATFORMS:
        idx = {m["id"]: m for m in read_jsonl(HARVEST / p / "index.jsonl")}
        for f in (LABELS / p).glob("*.json") if (LABELS / p).exists() else []:
            if f.name.endswith(".captions.json"):
                continue
            r = json.loads(f.read_text(encoding="utf-8"))
            m = idx.get(r["id"])
            if not m or voices.get((p, r["id"])) == "test" or not r["usable_s"]:
                continue
            up[f"{p}:{m.get('channel') or m.get('channel_id')}"] += r["usable_s"]
            groups[m.get("group", "?")] += r["usable_s"]
    extra = sum(sum(ln["end"] - ln["start"] for ln in sp["lines"] if ln["ok"])
                for f in LABELS.glob("*/*.json") if not f.name.endswith(".captions.json")
                for sp in json.loads(f.read_text(encoding="utf-8"))["spans"] if sp.get("extra") and sp["usable"])
    lines += ["", f"Distinct uploaders with usable audio: {len(up)} ({sum(1 for v in up.values() if v >= 600)} with "
                  f"10+ min). Usable hours by search group: "
                  + ", ".join(f"{g} {s / 3600:.1f}" for g, s in groups.most_common()) + ".",
              f"Usable audio of texts outside the corpus (Mafatih second pass): {extra / 3600:.1f} h."]
    lines += ["", f"Corpus texts with usable aligned audio: {len(per)} of 505.", "",
              "| text | recordings | usable h |", "|---|---:|---:|"]
    for d, s in per.most_common(30):
        lines.append(f"| {d} | {recs[d]} | {s / 3600:.1f} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    lines: list[str] = []
    totals(lines)
    identify(lines)
    timing(lines)
    texts(lines)
    out = "\n".join(lines)
    print(out)
    if args.md:
        args.md.write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
