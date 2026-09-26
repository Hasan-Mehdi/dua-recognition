#!/usr/bin/env python
"""Harvest du'a recitations from the Al-Fatemah Islamic Centre YouTube channel.

The channel streams long programmes (amaal nights, Ramadan, Muharram) with
many different reciters. There are no captions, so this is unlabelled audio:
scripts/segment_streams.py finds the du'a spans and the offline smoother
labels them later.

    python scripts/fetch_fatemah.py index      # flat list of /videos + /streams -> index.jsonl
    python scripts/fetch_fatemah.py download   # best-matching titles, audio only, <= --hours

Both stages are resumable (state in data/fatemah/, a junction to D:). Pacing
and rate-limit handling come from find_captioned.py: one request at a time.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_captioned import Pacer, _append, _read_jsonl, classify, is_test_upload, ydl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "fatemah"
INDEX = OUT / "index.jsonl"
AUDIO = OUT / "audio"
CHANNEL = "https://www.youtube.com/@AlFatemahIC"

# Du'a / ziyarat groups (English, Urdu and Arabic spellings). A title scores 3
# per group it names, +2 for an amaal night; programme words alone (Muharram,
# Ramadan, majlis) are mostly lectures and don't qualify.
GROUPS = {
    "kumayl": ["dua kumail", "dua-e-kumail", "dua e kumail", "dua-e-kumayl", "dua kumayl", "kumail dua",
               "dua-i-kumail", "دعاء كميل", "دعای کمیل"],
    "tawassul": ["tawassul", "tawasul", "توسل"], "iftitah": ["iftitah", "iftetah", "افتتاح"],
    "abu_hamza": ["abu hamza", "abu hamzah", "ابي حمزة", "أبي حمزة"], "jawshan": ["jawshan", "joshan", "jaushan", "جوشن"],
    "arafah": ["arafah", "arafa ", "arafat", "عرفة"], "nudba": ["nudba", "nudbah", "ندبة", "ندبه"],
    "ahd": ["dua ahd", "dua-e-ahd", "dua e ahd", "عهد"], "ziyarat_ashura": ["ziyarat ashura", "ziyarat-e-ashura",
    "ziyarat e ashura", "ziarat ashura", "زيارة عاشوراء", "aamaal e aashur", "amaal of ashura", "aamaal of ashura"],
    "warith": ["warith", "waritha", "وارث"], "aminallah": ["aminallah", "amin allah", "ameenallah", "امين الله", "أمين الله"],
    "sahar": ["dua sahar", "dua-e-sahar", "سحر"], "mujeer": ["mujeer", "mujir", "مجير"],
    "jamia": ["jamia kabira", "jamea kabira", "jamia kabeera", "جامعة"], "ale_yasin": ["ale yasin", "aale yasin", "آل يس"],
    "alqama": ["alqama", "alqamah", "علقمة"], "sabah": ["dua sabah", "dua-e-sabah", "صباح"],
    "simaat": ["simaat", "samaat", "سمات"], "mashlool": ["mashlool", "mashlul", "مشلول"],
    "baha": ["dua baha", "dua-e-baha", "بهاء"], "shabaniya": ["shabaniya", "sha'baniya", "شعبانية"],
    "makarim": ["makarim", "مكارم"], "nahiya": ["ziyarat nahiya", "ziyarat-e-nahiya", "ناحية"],
    "ziyarat": ["ziyarat", "ziarat", "زيارة"], "munajat": ["munajat", "munajaat", "مناجاة"],
    "qadr": ["shab-e-qadr", "shab e qadr", "laylatul qadr", "lailatul qadr", "laylat al-qadr", "ليلة القدر"],
}
AMAAL = re.compile(r"a+'?a+mal|amaal|a'amaal|اعمال|أعمال", re.I)
SKIP = re.compile(r"(lecture|khutbah?|sermon|q&a|nasheed|qasida|noha|nauha|latmiya|matam|speech|class|"
                  r"reflection|talk|interview|quiz|kids|children|introduction|tafseer|jashn|jashan|"
                  r"wiladat|wilaadat|lessons?|soz|explained|explanation|meaning)", re.I)


def groups(title: str) -> list[str]:
    t = " ".join(title.lower().replace("-", " ").split())
    t2 = title.lower()
    return [g for g, al in GROUPS.items() if any(a.replace("-", " ") in t or a in t2 for a in al)]


def score(title: str) -> int:
    g = groups(title)
    if SKIP.search(title) or not (g or AMAAL.search(title)):
        return 0
    return 3 * len(g) + (2 if AMAAL.search(title) else 0)


def cmd_index(args) -> None:
    seen = {r["id"] for r in _read_jsonl(INDEX)}
    pacer = Pacer(args.gap)
    for tab in ("videos", "streams"):
        while True:
            try:
                with ydl(args, extract_flat="in_playlist") as y:
                    info = y.extract_info(f"{CHANNEL}/{tab}", download=False)
                pacer.ok()
                break
            except Exception as e:  # noqa: BLE001
                if classify(str(e)) == "rate" and pacer.cool_down(str(e)):
                    continue
                print(f"  index {tab} failed: {str(e)[:160]}", flush=True)
                info = {"entries": []}
                break
        n = 0
        for e in info.get("entries") or []:
            if not e or e.get("id") in seen:
                continue
            seen.add(e["id"])
            _append(INDEX, {"id": e["id"], "title": e.get("title") or "", "duration": e.get("duration"),
                            "date": e.get("upload_date") or e.get("release_timestamp"), "tab": tab,
                            "score": score(e.get("title") or "")})
            n += 1
        print(f"  {tab}: {n} new entries", flush=True)
        pacer.wait()
    rows = _read_jsonl(INDEX)
    print(f"index: {len(rows)} videos, {sum(1 for r in rows if r['score'] > 0)} matching titles", flush=True)


def selection(rows: list[dict], hours: float, min_min: float) -> list[dict]:
    for r in rows:
        r["score"] = score(r["title"])
    ok = [r for r in rows if r["score"] > 0 and (r.get("duration") or 0) >= min_min * 60]
    ok.sort(key=lambda r: (-r["score"], -(r.get("duration") or 0)))
    out, total, per = [], 0.0, {}
    for r in ok:
        # No single du'a gets more than a quarter of the budget (Kumayl would take all of it).
        key = (groups(r["title"]) or ["amaal"])[0]
        if per.get(key, 0) + r["duration"] > hours * 3600 / 4:
            continue
        # Very long streams (> 5 h) are mostly lectures around a short du'a: skip them.
        if (r.get("duration") or 0) > 5 * 3600:
            continue
        if total + r["duration"] > hours * 3600:
            continue
        out.append(r)
        total += r["duration"]
        per[key] = per.get(key, 0) + r["duration"]
    return out


def cmd_download(args) -> None:
    # Training venue: never a test venue, and no video a test set holds out.
    rows = [r for r in _read_jsonl(INDEX) if not is_test_upload({"id": r["id"], "channel_url": CHANNEL})]
    rows = selection(rows, args.hours, args.min_minutes)
    print(f"selected {len(rows)} videos, {sum(r['duration'] for r in rows) / 3600:.1f} h", flush=True)
    AUDIO.mkdir(parents=True, exist_ok=True)
    pacer = Pacer(args.gap)
    for i, r in enumerate(rows):
        vid = r["id"]
        if (AUDIO / f"{vid}.json").exists():
            continue
        opts = dict(outtmpl=str(AUDIO / "%(id)s.%(ext)s"),
                    format="bestaudio[abr<=96]/bestaudio")
        ok = False
        while True:
            try:
                with ydl(args, **opts) as y:
                    y.params["skip_download"] = False
                    y.download([f"https://www.youtube.com/watch?v={vid}"])
                pacer.ok()
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                if classify(str(e)) == "rate" and pacer.cool_down(str(e)):
                    continue
                print(f"  download failed {vid}: {str(e)[:160]}", flush=True)
                break
        if ok:
            (AUDIO / f"{vid}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [{i + 1}/{len(rows)}] {'ok' if ok else 'FAILED'} {vid} {r['duration'] / 60:.0f} min: "
              f"{r['title'][:70]}", flush=True)
        pacer.wait()
    print("download done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["index", "download"])
    ap.add_argument("--cookies", type=Path, help="Netscape cookies.txt (YouTube domains only)")
    ap.add_argument("--gap", type=float, default=30, help="seconds between requests (jittered)")
    ap.add_argument("--hours", type=float, default=40, help="download: audio budget")
    ap.add_argument("--min-minutes", type=float, default=10, help="download: shortest video")
    args = ap.parse_args()
    {"index": cmd_index, "download": cmd_download}[args.stage](args)


if __name__ == "__main__":
    main()
