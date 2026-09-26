#!/usr/bin/env python
"""Find YouTube recitations that carry a human-written Arabic caption track.

Some uploaders (Ali Fani's channel, for one) attach real caption files, one
du'a line per cue, on fully diacritized text. That's the same kind of label
DuaPlayer gives us: which line is being recited at every moment, by a person.
Auto-generated captions don't count and are ignored.

Stages, each resumable (state lives in data/captioned/):

    python scripts/find_captioned.py search     # YouTube searches -> candidates.jsonl
    python scripts/find_captioned.py scan       # which candidates have manual ar captions
    python scripts/find_captioned.py channels   # crawl uploads of channels that caption du'as
    python scripts/find_captioned.py fetch      # captions + audio for the hits

Pacing follows what worked for the music-player harvest: one request at a
time, ~2.5 s apart with jitter, escalating cool-down (2 min .. 30 min) on a
rate limit, and a per-video "sign in" lock is logged and skipped rather than
treated as an IP block. --cookies raises YouTube's per-IP ceiling.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

import yt_dlp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "captioned"
CANDIDATES = OUT / "candidates.jsonl"
SCANNED = OUT / "scanned.jsonl"

# Titles that promise text on screen or in captions get scanned first.
HINTS = re.compile(r"sub|caption|cc\b|lyrics|text|مترجم|ترجمة|الكلمات|مع الكلمات|مكتوب|زیرنویس|متن|ترجمه", re.I)
DUA_WORDS = re.compile(r"dua|du'a|duaa|ziyar|munaj|دعا|زيار|زیار|مناجا|صحيفة|صحیفه|sahifa|tasbih|تسبيح", re.I)

# Du'as and ziyarat outside DuaPlayer's 91 texts that are widely recited. A
# captioned recording of one brings its own text, so these widen coverage too.
EXTRA_NAMES = [
    "الزيارة الجامعة الكبيرة", "زيارة آل يس", "دعاء علقمة", "زيارة وارث", "دعاء الجوشن الكبير",
    "دعاء الجوشن الصغير", "دعاء عرفة", "دعاء الصباح", "دعاء السمات", "دعاء الندبة", "دعاء المشلول",
    "زيارة الناحية المقدسة", "زيارة عاشوراء غير المعروفة", "دعاء يستشير", "دعاء العديلة",
    "دعاء الفرج", "دعاء الغريق", "دعاء النور", "مناجاة أمير المؤمنين", "المناجاة الشعبانية",
    "مناجاة الخمسة عشر", "الصحيفة السجادية", "دعاء مكارم الأخلاق", "دعاء أبي حمزة الثمالي",
    "زيارة الأربعين", "زيارة عاشوراء", "دعاء التوسل", "دعاء كميل", "دعاء الافتتاح", "دعاء العهد",
    "زيارة أمين الله", "دعاء الحجة", "دعاء يوم الثلاثاء", "أدعية أيام الأسبوع", "دعاء أيام شهر رمضان",
    "دعاء السحر", "دعاء البهاء", "زيارة الإمام الحسين", "زيارة الإمام الرضا", "زيارة السيدة زينب",
    "زيارة السيدة المعصومة", "زيارة أم البنين", "زيارة العباس", "دعاء الإمام الحسين يوم عرفة",
]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Pacer:
    """One request at a time, jittered sleeps, escalating cool-down on rate limits."""

    def __init__(self, gap: float, max_cooldown: float = 1800, max_cooldowns: int = 6):
        self.gap, self.max_cooldown, self.max_cooldowns = gap, max_cooldown, max_cooldowns
        self.cooldowns = 0

    def wait(self) -> None:
        time.sleep(self.gap * random.uniform(0.7, 1.6))

    def ok(self) -> None:
        self.cooldowns = 0

    def cool_down(self, why: str) -> bool:
        self.cooldowns += 1
        if self.cooldowns > self.max_cooldowns:
            return False
        s = min(120 * 2 ** (self.cooldowns - 1), self.max_cooldown)
        print(f"  rate limited ({why[:80]}); cooling down {s} s", flush=True)
        time.sleep(s)
        return True


def classify(err: str) -> str:
    s = err.lower()
    if any(x in s for x in ("429", "too many requests", "rate", "try again later", "page needs to be reloaded")):
        return "rate"
    if "sign in to confirm" in s or "not a bot" in s:
        return "auth"
    return "fail"


def ydl(args, **extra) -> yt_dlp.YoutubeDL:
    opts = dict(quiet=True, no_warnings=True, skip_download=True,
                js_runtimes={"node": {}}, **extra)
    if args.cookies:
        opts["cookiefile"] = str(args.cookies)
    return yt_dlp.YoutubeDL(opts)


def queries() -> list[tuple[str | None, str]]:
    q: list[tuple[str | None, str]] = []
    # Read the texts directly: this script runs in whichever Python has yt-dlp,
    # which needn't have the package's dependencies.
    for path in sorted((ROOT / "data" / "duas").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        ar = re.sub(r"[ً-ْٰ]", "", d.get("dua_name_ar", ""))
        ar = re.sub(r"\s+", " ", ar).strip()
        en = d.get("dua_name_en", "")
        for s in (f"{ar} مع الكلمات", f"{ar} مترجم", f"{ar} مكتوب", f"{en} arabic subtitles", f"{en} AR SUB"):
            if s.split()[0]:
                q.append((d["dua_id"], s))
    for name in EXTRA_NAMES:
        q += [(None, f"{name} مع الكلمات"), (None, f"{name} مترجم")]
    return q


def cmd_search(args) -> None:
    done = {r["query"] for r in _read_jsonl(OUT / "searched.jsonl")}
    seen = {r["id"] for r in _read_jsonl(CANDIDATES)}
    pacer = Pacer(args.gap)
    todo = [x for x in queries() if x[1] not in done]
    print(f"{len(todo)} queries to run, {len(seen)} candidates so far", flush=True)
    with ydl(args, extract_flat=True) as y:
        for dua_id, q in todo:
            while True:
                try:
                    r = y.extract_info(f"ytsearch{args.n}:{q}", download=False)
                    pacer.ok()
                    break
                except Exception as e:  # noqa: BLE001
                    if classify(str(e)) == "rate" and pacer.cool_down(str(e)):
                        continue
                    print(f"  search failed: {q}: {str(e)[:120]}", flush=True)
                    r = None
                    break
            new = 0
            for e in (r or {}).get("entries") or []:
                if not e or e["id"] in seen or (e.get("duration") or 0) < 60:
                    continue
                seen.add(e["id"])
                new += 1
                _append(CANDIDATES, {"id": e["id"], "title": e.get("title"), "channel": e.get("channel"),
                                     "channel_id": e.get("channel_id"), "duration": e.get("duration"),
                                     "query_dua": dua_id, "query": q})
            _append(OUT / "searched.jsonl", {"query": q, "new": new})
            pacer.wait()
    print(f"{len(seen)} candidates", flush=True)


def _priority(c: dict) -> tuple:
    t = f"{c.get('title') or ''}"
    return (not HINTS.search(t), not DUA_WORDS.search(t), c.get("duration") or 0)


def cmd_scan(args) -> None:
    scanned = {r["id"] for r in _read_jsonl(SCANNED)}
    todo = sorted((c for c in _read_jsonl(CANDIDATES) if c["id"] not in scanned), key=_priority)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} to scan ({len(scanned)} done)", flush=True)
    pacer, hits, auth_streak = Pacer(args.gap), 0, 0
    with ydl(args) as y:
        for i, c in enumerate(todo, 1):
            while True:
                try:
                    info = y.extract_info(f"https://www.youtube.com/watch?v={c['id']}", download=False)
                    status, pacer_ok = "ok", True
                    break
                except Exception as e:  # noqa: BLE001
                    kind = classify(str(e))
                    if kind == "rate":
                        if pacer.cool_down(str(e)):
                            continue
                        print("giving up: still rate limited after repeated cool-downs", flush=True)
                        return
                    info, status, pacer_ok = None, kind, False
                    break
            if pacer_ok:
                pacer.ok()
                auth_streak = 0
            elif status == "auth":
                auth_streak += 1
                if auth_streak >= 10 and not pacer.cool_down("10 sign-in locks in a row"):
                    return
            subs = sorted(k for k in ((info or {}).get("subtitles") or {}) if k != "live_chat")
            has_ar = any(k == "ar" or k.startswith("ar-") for k in subs)
            hits += has_ar
            _append(SCANNED, {"id": c["id"], "status": status, "subs": subs, "has_ar": has_ar,
                              "title": (info or {}).get("title") or c.get("title"),
                              "channel": (info or {}).get("channel") or c.get("channel"),
                              "channel_id": (info or {}).get("channel_id") or c.get("channel_id"),
                              "duration": (info or {}).get("duration") or c.get("duration"),
                              "query_dua": c.get("query_dua")})
            if has_ar:
                print(f"  HIT {c['id']} {(info or {}).get('channel')}: {(info or {}).get('title', '')[:70]}", flush=True)
            if i % 50 == 0:
                print(f"  scanned {i}/{len(todo)}, {hits} hits", flush=True)
            pacer.wait()
    print(f"done: {hits} hits", flush=True)


def cmd_channels(args) -> None:
    """Uploaders who caption one du'a usually caption the rest: list their uploads."""
    hits = [r for r in _read_jsonl(SCANNED) if r["has_ar"] and r.get("channel_id")]
    crawled = {r["channel_id"] for r in _read_jsonl(OUT / "channels.jsonl")}
    seen = {r["id"] for r in _read_jsonl(CANDIDATES)}
    pacer = Pacer(args.gap)
    with ydl(args, extract_flat=True) as y:
        for ch in sorted({r["channel_id"] for r in hits} - crawled):
            try:
                r = y.extract_info(f"https://www.youtube.com/channel/{ch}/videos", download=False)
            except Exception as e:  # noqa: BLE001
                print(f"  channel {ch} failed: {str(e)[:120]}", flush=True)
                r = None
            new = 0
            for e in (r or {}).get("entries") or []:
                if not e or e["id"] in seen or (e.get("duration") or 0) < 60:
                    continue
                if not DUA_WORDS.search(e.get("title") or ""):
                    continue
                seen.add(e["id"])
                new += 1
                _append(CANDIDATES, {"id": e["id"], "title": e.get("title"), "channel": (r or {}).get("channel"),
                                     "channel_id": ch, "duration": e.get("duration"),
                                     "query_dua": None, "query": f"channel:{ch}"})
            _append(OUT / "channels.jsonl", {"channel_id": ch, "new": new})
            print(f"  channel {(r or {}).get('channel')}: {new} new du'a uploads", flush=True)
            pacer.wait()


def cmd_fetch(args) -> None:
    hits = [r for r in _read_jsonl(SCANNED) if r["has_ar"]]
    # The same recording is often uploaded once per subtitle language: keep one
    # per (channel, duration to the nearest 3 s).
    uniq: dict[tuple, dict] = {}
    for r in hits:
        uniq.setdefault((r.get("channel_id"), round((r.get("duration") or 0) / 3)), r)
    pacer = Pacer(args.gap)
    audio_dir = OUT / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for r in uniq.values():
        vid = r["id"]
        if (audio_dir / f"{vid}.json").exists():
            continue
        # Captions are tiny; audio waits until the disk has room (other jobs share it).
        want_audio = args.audio and shutil.disk_usage(OUT).free > args.min_free_gb * 1e9
        opts = dict(writesubtitles=True, subtitleslangs=["ar.*", "ar"], subtitlesformat="json3/srt/vtt",
                    outtmpl=str(audio_dir / "%(id)s.%(ext)s"), skip_download=not want_audio,
                    format="bestaudio[abr<=96]/bestaudio")
        while True:
            try:
                with ydl(args, **opts) as y:
                    y.params["skip_download"] = not want_audio
                    y.download([f"https://www.youtube.com/watch?v={vid}"])
                pacer.ok()
                break
            except Exception as e:  # noqa: BLE001
                if classify(str(e)) == "rate" and pacer.cool_down(str(e)):
                    continue
                print(f"  fetch failed {vid}: {str(e)[:120]}", flush=True)
                break
        (audio_dir / f"{vid}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  fetched {vid}: {(r.get('title') or '')[:70]}", flush=True)
        pacer.wait()
    print(f"{len(uniq)} unique captioned recordings ({len(hits)} uploads)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["search", "scan", "channels", "fetch"])
    ap.add_argument("--cookies", type=Path, help="Netscape cookies.txt (YouTube domains only)")
    ap.add_argument("--gap", type=float, default=2.5, help="seconds between requests (jittered)")
    ap.add_argument("-n", type=int, default=25, help="results per search query")
    ap.add_argument("--limit", type=int, default=0, help="scan at most this many")
    ap.add_argument("--audio", action="store_true", help="fetch: download audio as well as captions")
    ap.add_argument("--min-free-gb", type=float, default=15, help="fetch: skip audio below this much free disk")
    args = ap.parse_args()
    {"search": cmd_search, "scan": cmd_scan, "channels": cmd_channels, "fetch": cmd_fetch}[args.stage](args)


if __name__ == "__main__":
    main()
