#!/usr/bin/env python
"""Build the corpus from DuaPlayer (duaplayer.org): texts, recordings, line timings.

DuaPlayer shows each du'a as numbered slides, and its reciters upload audio with
a hand-recorded start time for every slide. That gives us, per recording, a
human label of which line is being recited at every moment — the ground truth
the tracker is evaluated against.

    python scripts/fetch_duaplayer.py              # the default set below
    python scripts/fetch_duaplayer.py dua/kumayl   # just one

Writes:
    data/duas/<id>.json                    Arabic segments (committed)
    data/duaplayer/<id>/slides.json        + English / transliteration (local)
    data/duaplayer/<id>/<audio_id>.json    reciter, duration, slide timings (local)
    data/duaplayer/<id>/<audio_id>.mp3     the recording (local)

Only the Arabic text is committed. Translations, timings and audio belong to
DuaPlayer and its reciters and stay in the ignored data/duaplayer/ cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUAS = ROOT / "data" / "duas"
CACHE = ROOT / "data" / "duaplayer"
BASE = "https://www.duaplayer.org"

# Every du'a / ziyarat on DuaPlayer with an approved Arabic recitation
# (September 2026). Whether a recording is train or test is decided by its
# reciter (src/dua_recognition/splits.py), not by the du'a.
DEFAULT = [
    "dua/kumayl", "dua/tawassul", "dua/hadithkisa", "dua/iftitah", "dua/baha",
    "dua/aahad", "dua/hujjat", "dua/makaramakhlaq", "dua/Mashlool", "dua/nudbah",
    "ziyarat/ashura", "ziyarat/aminallah-imam-ali",
    "dua/abu-hamza-thumali", "dua/mujeer", "dua/tawba", "dua/simaat",
    "dua/munajat-ali-kufa", "dua/munajat-taibeen", "dua/amaal-quran",
    "dua/ya-aliyu-ya-azeem", "dua/wahda", "dua/ramadan-1",
]


def _get(path: str):
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "dua-recognition/0.2"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _dua_id(kind: str, route: str) -> str:
    return f"{kind}-{route.lower()}"


def fetch(kind: str, route: str, audio: bool = True) -> None:
    dua_id = _dua_id(kind, route)
    langs = urllib.parse.quote("ar|en|tl")
    info = _get(f"/api/{kind}/{route}?langs={langs}&getSlides=true")
    slides = info.get("slides") or {}
    slides.update(_get(f"/api/slides/{kind}/{route}?langs={langs}&afterUniversal=true"))
    ids = sorted((int(k) for k in slides), key=int)
    if not ids:
        print(f"  {dua_id}: no slides, skipped")
        return

    names = info.get("duaName", {})
    segments = [
        {"segment_id": i, "arabic": slides[str(i)].get("ar", "").strip()}
        for i in ids
    ]
    DUAS.mkdir(parents=True, exist_ok=True)
    (DUAS / f"{dua_id}.json").write_text(
        json.dumps(
            {
                "dua_id": dua_id,
                "dua_name_en": names.get("tl") or names.get("en") or route,
                "dua_name_ar": names.get("ar", ""),
                "source": f"{BASE}/en/{kind}/{route}",
                "segments": segments,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    out = CACHE / dua_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "slides.json").write_text(
        json.dumps({str(i): slides[str(i)] for i in ids}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    recs = [
        a for a in _get(f"/api/audio/list/{kind}/{route}?rnd=1")
        if a.get("approvedState") == 1 and "ar" in a.get("languages", [])
    ]
    for a in recs:
        detail = _get(f"/api/audio/{a['audioId']}?rnd=1")
        meta = {
            "audio_id": a["audioId"],
            "dua_id": dua_id,
            "reciter": detail["reciter"]["reciterName"].get("tl", ""),
            "duration_ms": detail["duration"],
            # slide id -> start time in ms. A key past the last slide, when
            # present, marks where the recitation ends.
            "slide_start_ms": {int(k): v for k, v in detail["slideTiming"].items()},
        }
        (out / f"{a['audioId']}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        mp3 = out / f"{a['audioId']}.mp3"
        if audio and not mp3.exists():
            urls = detail.get("urls", {})
            url = urls.get("64") or urls.get(str(min(map(int, urls))))
            urllib.request.urlretrieve(url, mp3)
        time.sleep(0.5)  # be polite to a small non-profit's API
    print(f"  {dua_id}: {len(segments)} segments, {len(recs)} recordings")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("routes", nargs="*", default=DEFAULT, help="kind/route, e.g. dua/kumayl")
    ap.add_argument("--no-audio", action="store_true", help="texts and timings only")
    ap.add_argument("--all", action="store_true",
                    help="every du'a and ziyarat in DuaPlayer's catalogue, recorded or not")
    args = ap.parse_args()
    routes = args.routes
    if args.all:
        routes = [f"{kind}/{x['r']}" for kind in ("dua", "ziyarat") for x in _get(f"/api/browse/{kind}")["result"]]
    for r in routes:
        kind, route = r.split("/", 1)
        try:
            fetch(kind, route, audio=not args.no_audio)
        except Exception as e:  # keep going; one bad route shouldn't sink the batch
            print(f"  {r}: failed ({e})", file=sys.stderr)


if __name__ == "__main__":
    main()
