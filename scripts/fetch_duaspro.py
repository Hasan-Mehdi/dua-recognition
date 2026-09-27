#!/usr/bin/env python
"""Add du'as and timed recordings from duas.pro's open API.

duas.pro (github.com/duas-pro/shia-duas-api, "openly accessible for anyone to
use in their own projects") stores du'as line by line, and each recitation has
a start time for every line, like DuaPlayer's slide timings. Its texts for the
du'as DuaPlayer also has are the same line splits, so timings line up with our
segment ids; beyond those it adds most of al-Sahifa al-Sajjadiyya and a few
more du'as and ziyarat. Most of its recordings are DuaPlayer's own uploads;
only the ones we don't already have are fetched.

    python scripts/fetch_duaspro.py              # new texts + new recordings
    python scripts/fetch_duaspro.py --no-audio   # texts and timings only
    python scripts/fetch_duaspro.py --lines      # translations + readings of our duas.pro texts

Writes:
    data/duas/<id>.json                   Arabic segments (committed)
    data/duaspro/<id>/<uuid>.json         reciter, duration, line timings (local)
    data/duaspro/<id>/<uuid>.mp3          the recording (local)
    data/lines/<id>.json                  English + transliteration per line (--lines, local)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.translit import match_readings  # noqa: E402

DUAS = ROOT / "data" / "duas"
CACHE = ROOT / "data" / "duaspro"
DUAPLAYER = ROOT / "data" / "duaplayer"
LINES = ROOT / "data" / "lines"  # corpus.LINES_DIR
NEW_TEXTS = DUAS
BASE = "https://dhreftlcbkiqbsprhjqh.supabase.co/functions/v1"
SITE = "https://duas.pro"

# Reciters we can't place in the train/test split don't get downloaded: an
# unnamed voice could be one of the test reciters.
SKIP_RECITERS = {"Unknown", ""}


def _get(path: str):
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "dua-recognition/0.2"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def dua_id(slug: str) -> str:
    """Our id for a duas.pro slug. Sahifa entries are numbered ("47-his-...")."""
    m = re.match(r"(\d+)-", slug)
    return f"sahifa-{int(m.group(1)):02d}" if m else slug.lower()


def catalogue() -> list[dict]:
    out, page = [], 1
    while True:
        rows = _get(f"/duas?languages=ar&size=50&page={page}")["data"]
        out += rows
        if len(rows) < 50:
            break
        page += 1
    seen, uniq = set(), []
    for d in out:  # the paginated listing repeats a few entries
        if d["slug"] not in seen:
            seen.add(d["slug"])
            uniq.append(d)
    return uniq


def have_recording(did: str, duration_s: float) -> bool:
    """A DuaPlayer recording of the same du'a with the same length is the same upload."""
    for meta in (DUAPLAYER / did).glob("*-*.json"):
        if meta.name.endswith(".repaired.json"):
            continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        if abs(m["duration_ms"] / 1000 - duration_s) < 2.0:
            return True
    return False


def _bare(text: str) -> str:
    """Letters only: no diacritics, one form of alif/ya/ta marbuta, Persian letters folded."""
    text = re.sub(r"[ً-ٰٟـۡ-ۭ]", "", text)
    text = text.translate(str.maketrans("أإآٱىةیکؤئ",
                                        "اااايهيكوي"))
    return " ".join(re.sub(r"[^ء-ي ]", " ", text).split())


def _opening(segments: list[str], skip_basmala: bool = True) -> str:
    segs = [s for s in map(_bare, segments) if s]
    if skip_basmala and segs and segs[0].startswith("بسم الله"):
        segs = segs[1:]
    return " ".join(segs[:3])[:120]


def known_openings() -> dict[str, str]:
    """Opening words of every text we have, so a du'a listed under another slug isn't added twice."""
    out = {}
    for p in DUAS.glob("*.json"):
        raw = json.loads(p.read_text(encoding="utf-8"))
        out[_opening([s["arabic"] for s in raw["segments"]])] = raw["dua_id"]
    return out


def fetch(slug: str, audio: bool, openings: dict[str, str]) -> tuple[int, int]:
    did = dua_id(slug)
    d = _get(f"/duas/{slug}?languages=ar")["data"]
    lines = [(l["line_number"], (l["translations"] or [{}])[0].get("text", "").strip()) for l in d["lines"]]
    lines = [(n, t) for n, t in lines if t]
    if not lines:
        return 0, 0
    new_text = 0
    path = DUAS / f"{did}.json"
    if not path.exists():
        path = NEW_TEXTS / f"{did}.json"
    same = openings.get(_opening([t for _, t in lines]))
    if not path.exists() and same:
        print(f"  {slug}: same text as {same}, skipped")
        return 0, 0
    if not path.exists():
        path.write_text(json.dumps({
            "dua_id": did,
            "dua_name_en": slug.replace("-", " ").title() if did == slug else f"Sahifa {did[-2:]}: {slug.split('-', 1)[1].replace('-', ' ')}",
            "dua_name_ar": (d.get("title") or {}).get("ar", ""),
            "source": f"{SITE}/{slug}",
            "segments": [{"segment_id": n, "arabic": t} for n, t in lines],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        new_text = 1

    new_recs = 0
    for r in d.get("recitations") or []:
        reciter = (r.get("reciters") or {}).get("full_name_tl", "")
        dur = r["duration_in_ms"] / 1000
        if reciter in SKIP_RECITERS or have_recording(did, dur):
            continue
        out = CACHE / did
        meta_path = out / f"{r['uuid']}.json"
        if meta_path.exists():
            continue
        a = _get(f"/audios/{r['uuid']}")["data"]
        starts = a.get("startTimes") or []
        if len(starts) != len(d["lines"]):
            print(f"  {did}/{reciter}: {len(starts)} start times for {len(d['lines'])} lines, skipped")
            continue
        out.mkdir(parents=True, exist_ok=True)
        timing = {l["line_number"]: t for l, t in zip(d["lines"], starts)}
        mp3 = out / f"{r['uuid']}.mp3"
        if audio and not mp3.exists():
            urllib.request.urlretrieve(a["audio_low_quality_url"], mp3)
        meta_path.write_text(json.dumps({
            "audio_id": r["uuid"],
            "dua_id": did,
            "reciter": reciter,
            "duration_ms": r["duration_in_ms"],
            "slide_start_ms": timing,
            "source": f"{SITE}/{slug}",
        }, indent=1), encoding="utf-8")
        new_recs += 1
        print(f"  {did}: + {reciter} ({dur / 60:.0f} min, {len(timing)} lines)", flush=True)
        time.sleep(0.5)
    return new_text, new_recs


def save_lines(slug: str) -> tuple[int, int]:
    """English and transliteration for each line of one of our duas.pro texts
    (readings a row out of step put back: translit.match_readings)."""
    did = dua_id(slug)
    d = _get(f"/duas/{slug}?languages=ar,en,translit")["data"]
    rows = [{t["language"]: (t.get("text") or "").strip() for t in l["translations"] or []} for l in d["lines"]]
    ours = {s["segment_id"]: s["arabic"] for s in json.loads((DUAS / f"{did}.json").read_text(encoding="utf-8"))["segments"]}
    arabic = [ours.get(l["line_number"], r.get("ar", "")) for l, r in zip(d["lines"], rows)]
    tl = [r.get("translit", "") for r in rows]
    out = {}
    for l, r, j in zip(d["lines"], rows, match_readings(arabic, tl)):
        e = {"tl": tl[j] if j is not None else "", "en": r.get("en", "")}
        if e["en"] or e["tl"]:
            out[str(l["line_number"])] = e
    LINES.mkdir(parents=True, exist_ok=True)
    (LINES / f"{did}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(out), len(d["lines"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slugs", nargs="*", help="only these duas.pro slugs")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--lines", action="store_true", help="only fetch translations + readings for texts we have")
    ap.add_argument("--texts-dir", type=Path, default=DUAS,
                    help="where new texts go (stage them elsewhere while other evaluations run)")
    args = ap.parse_args()
    global NEW_TEXTS
    NEW_TEXTS = args.texts_dir
    NEW_TEXTS.mkdir(parents=True, exist_ok=True)
    texts = recs = 0
    openings = known_openings()
    if args.lines:
        # By each text's own source slug: the paginated listing drops a few entries.
        sources = [json.loads(p.read_text(encoding="utf-8")).get("source", "") for p in DUAS.glob("*.json")]
        ours = [src.removeprefix(SITE + "/") for src in sources if src.startswith(SITE + "/")]
        n = found = lines = 0
        for slug in ours:
            if args.slugs and slug not in args.slugs:
                continue
            try:
                f, t = save_lines(slug)
            except Exception as e:  # noqa: BLE001
                print(f"  {slug}: failed ({e})", file=sys.stderr)
                continue
            n, found, lines = n + 1, found + f, lines + t
            time.sleep(0.3)
        print(f"{n}/{len(ours)} texts, {found}/{lines} lines")
        return
    for d in catalogue():
        if args.slugs and d["slug"] not in args.slugs:
            continue
        try:
            t, r = fetch(d["slug"], audio=not args.no_audio, openings=openings)
        except Exception as e:  # keep going; one bad entry shouldn't sink the batch
            print(f"  {d['slug']}: failed ({e})", file=sys.stderr)
            continue
        texts += t
        recs += r
        time.sleep(0.3)
    print(f"{texts} new texts, {recs} new recordings")


if __name__ == "__main__":
    main()
