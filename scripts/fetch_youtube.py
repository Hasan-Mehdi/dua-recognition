#!/usr/bin/env python
"""Harvest extra recitations from YouTube, for fine-tuning only.

For each du'a in the corpus: search YouTube in Arabic, keep results whose
length is plausible for a full recitation, drop anything by a reciter already
in the DuaPlayer set, and download the audio. None of these carry line
timings; scripts/align_offline.py labels them with the offline aligner, whose
accuracy is measured against DuaPlayer's human timings first.

Reciters already in the DuaPlayer set are excluded by name (Arabic and
English, as they appear in titles/channels): test reciters so the held-out
numbers can't leak, train reciters so the extra hours bring new voices.

    python scripts/fetch_youtube.py --per-dua 3

Audio stays in the ignored data/youtube/ cache with a manifest of source URLs.
It's used locally to train a model; it isn't redistributed.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.corpus import load_all, load_recordings  # noqa: E402
from dua_recognition.text import strip_diacritics  # noqa: E402

OUT = ROOT / "data" / "youtube"
YTDLP = str(Path(sys.executable).with_name("yt-dlp"))

KNOWN_RECITERS = [
    # test
    "حلواجي", "الحلواجي", "halwachi", "halawaji", "abather", "abu thar", "أباذر", "اباذر",
    "غريب", "ghareeb", "ghreeb", "gharib",
    "الأكرف", "الاكرف", "akraf",
    "قريش", "qureish", "quraish", "qurayshi",
    "فرهمند", "farahmand",
    # train
    "فاني", "fani", "قمبر", "qambar", "kambar", "العطار", "attar", "بوماد", "boumad",
    "رسولي", "rasouli", "رضوي", "rizvi",
]
NOT_A_RECITATION = ["شرح", "تفسير", "محاضرة", "lecture", "explained", "tafseer", "reaction", "مقطع", "shorts"]


def search(query: str, n: int) -> list[dict]:
    out = subprocess.run(
        [YTDLP, f"ytsearch{n}:{query}", "--flat-playlist", "--dump-json", "--no-warnings"],
        capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


def _fold(text: str) -> str:
    # Titles write names with diacritics ("غريّب"); compare bare letters.
    return strip_diacritics(text).translate(str.maketrans("أإآ", "ااا")).lower()


def excluded(v: dict) -> bool:
    text = _fold(f"{v.get('title', '')} {v.get('channel', '')} {v.get('uploader', '')}")
    return any(_fold(name) in text for name in KNOWN_RECITERS + NOT_A_RECITATION)


def acceptable(v: dict, expected_s: float) -> bool:
    if excluded(v):
        return False
    d = v.get("duration") or 0
    return 0.6 * expected_s <= d <= 1.8 * expected_s


def download(v: dict, dua_id: str) -> Path | None:
    out_dir = OUT / dua_id
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{v['id']}.json"
    if meta_path.exists():
        return meta_path
    r = subprocess.run(
        [YTDLP, "-f", "bestaudio[abr<=96]/bestaudio", "--no-playlist", "--no-warnings",
         "-o", str(out_dir / "%(id)s.%(ext)s"), f"https://www.youtube.com/watch?v={v['id']}"],
        capture_output=True, text=True, encoding="utf-8", timeout=900,
    )
    audio = [p for p in out_dir.glob(f"{v['id']}.*") if p.suffix not in (".json", ".part")]
    if r.returncode != 0 or not audio:
        print(f"    download failed: {v['id']} {r.stderr.strip()[-200:]}")
        return None
    meta_path.write_text(json.dumps({
        "video_id": v["id"], "url": f"https://www.youtube.com/watch?v={v['id']}",
        "title": v.get("title"), "channel": v.get("channel") or v.get("uploader"),
        "duration_s": v.get("duration"), "dua_id": dua_id, "audio": audio[0].name,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta_path


def prune() -> None:
    """Delete earlier downloads that today's filters would reject."""
    for meta_path in sorted(OUT.glob("*/*.json")):
        if meta_path.name.endswith(".labels.json"):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not excluded({"title": meta.get("title"), "channel": meta.get("channel")}):
            continue
        vid = meta["video_id"]
        for p in list(meta_path.parent.glob(f"{vid}.*")) + list((ROOT / "data/cache/windows").glob(f"*/yt-{vid}_*")):
            p.unlink()
        print(f"  pruned {meta['dua_id']}/{vid}: {meta.get('title', '')[:60]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-dua", type=int, default=3)
    ap.add_argument("--search", type=int, default=20, help="results to consider per query")
    ap.add_argument("--duas", nargs="*")
    args = ap.parse_args()

    prune()
    for dua in load_all().values():
        if args.duas and dua.id not in args.duas:
            continue
        recs = load_recordings(dua)
        if not recs:
            continue
        expected = statistics.median(r.duration_s for r in recs)
        have = len([p for p in (OUT / dua.id).glob("*.json") if not p.name.endswith(".labels.json")])             if (OUT / dua.id).exists() else 0
        name = re.sub(r"\s+", " ", dua.name_ar).strip()
        seen, kept = set(), have
        for query in (name, f"{name} كامل"):
            if kept >= args.per_dua:
                break
            for v in search(query, args.search):
                if kept >= args.per_dua:
                    break
                if v["id"] in seen or not acceptable(v, expected):
                    continue
                seen.add(v["id"])
                print(f"  {dua.id}: {v.get('title', '')[:70]} ({v.get('duration')} s)", flush=True)
                if download(v, dua.id):
                    kept += 1
        print(f"{dua.id}: {kept} recordings", flush=True)


if __name__ == "__main__":
    main()
