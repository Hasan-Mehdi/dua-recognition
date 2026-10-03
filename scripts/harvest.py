#!/usr/bin/env python
"""Harvest du'a recitations at scale: find, download, then let the audio say what it is.

fetch_youtube.py took the first few search hits per du'a that already had a
DuaPlayer recording (86 videos). This casts the net wide instead: every
corpus text by name in Arabic, Persian, transliteration and English, the
daily Ramadan du'as and the Sahifa by number, du'a reciters by name, ordinary
voices (children, families, gatherings), then the uploads of every channel
those searches turn up. Titles only decide what is downloaded; which corpus
text a recording holds (if any) is decided from the audio later, so a
mistitled or off-topic upload costs a download, not a wrong label.

Stages (state in data/harvest/<platform>/, a junction to D:\\dua-data\\harvest):

    python scripts/harvest.py discover            # searches -> candidates.jsonl
    python scripts/harvest.py channels            # crawl channels with du'a hits
    python scripts/harvest.py download -j 3       # audio -> audio/<id>.<ext>, index.jsonl

Rules carried over from the other harvesters: test reciters (by name, in
title/channel), test venues and test uploaders (splits.is_test_upload) are
never downloaded. Voices are checked again after download (speaker_check.py).
Audio stays local and is used only to train models; it is not redistributed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARVEST = ROOT / "data" / "harvest"
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str, path: Path):
    # By path: this runs in whichever Python has yt-dlp, which needn't have the
    # package's dependencies (dua_recognition/__init__ imports them).
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


splits = _load("splits", ROOT / "src" / "dua_recognition" / "splits.py")
text = _load("text", ROOT / "src" / "dua_recognition" / "text.py")

# Test reciters only: train reciters' other recordings are fine to train on.
TEST_NAMES = ["حلواجي", "الحلواجي", "halwachi", "halawaji", "halawachi", "abather", "abu thar", "abathar",
              "أباذر", "اباذر", "ابوذر", "أبوذر", "غريب", "ghareeb", "ghreeb", "gharib", "غريّب",
              "الأكرف", "الاكرف", "akraf", "قريش", "qureish", "quraish", "qurayshi", "qoraishi", "ghoraishi",
              "ghoreishi", "qureshi", "فرهمند", "farahmand", "حلواچی", "halvachi", "halwaji"]
NOT_A_RECITATION = ["شرح", "تفسير", "محاضرة", "lecture", "explained", "explanation", "tafseer", "tafsir",
                    "reaction", "#shorts", "تحلیل", "سخنرانی", "khutba", "خطبة", "بيان معاني", "podcast",
                    "بودكاست", "interview", "مقابلة", "tutorial", "تعليم القراءة", "how to recite"]
EXCLUDED_VIDEOS = {"qxuDk75kkBI"}  # Halwachi's DuaPlayer Hujjat test recording re-uploaded (fetch_youtube.py)
DUA_HINT = re.compile(
    r"دعا|زيار|زیار|مناجا|صحيف|صحیف|تسبيح|تسبیح|كساء|کسا|اعمال|أعمال|تعقيب|تعقیب|صلوات|صلاة|نماز|"
    r"dua|du'a|doa|duaa|ziyar|ziar|zyarat|munaj|sahifa|kisa|tasbih|salawat|amaal|kumail|kumayl|komail|"
    r"tawassul|nudba|nudbah|iftitah|jawshan|joshan|mashlool|faraj|ahad|ahd|ashura|warith|waris|aminallah|"
    r"ameenallah|yasin|yaseen|jamia|kursi|کمیل|كميل|توسل|ندب|عهد|فرج|افتتاح|جوشن|عاشورا|وارث|امین|أمين|"
    r"ياسين|یاسین|جامعة|جامعه|الكرسي|الکرسی", re.I)


def _fold(s: str) -> str:
    # Persian and Urdu letter forms fold onto Arabic ones (قریشی -> قريشي), so a
    # name written either way matches; چ -> ج and ژ -> ز for Persian spellings.
    s = text.strip_diacritics(s or "").translate(text._FOLD).translate(str.maketrans("چژگپ", "جزكب"))
    return s.translate(str.maketrans("أإآ", "ااا")).lower()


def excluded(v: dict) -> str:
    """Why an upload must not be harvested ('' if it may)."""
    if v.get("id") in EXCLUDED_VIDEOS:
        return "excluded video"
    if splits.is_test_upload(v):
        return "test venue/uploader"
    blob = _fold(f"{v.get('title', '')} {v.get('channel', '')} {v.get('uploader', '')}")
    for name in TEST_NAMES:
        if _fold(name) in blob:
            return f"test reciter ({name})"
    for w in NOT_A_RECITATION:
        # Children and families post their recitations as Shorts: keep those for the
        # ordinary-voice queries (the audio decides later what they hold).
        if w == "#shorts" and v.get("group") == "ordinary":
            continue
        if _fold(w) in blob:
            return f"not a recitation ({w})"
    return ""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut short by a killed run
    return out


_lock = threading.Lock()


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Pacer:
    """Jittered gaps; escalating cool-down on rate limits (shared by threads)."""

    def __init__(self, gap: float, max_cooldown: float = 1800, max_cooldowns: int = 8):
        self.gap, self.max_cooldown, self.max_cooldowns = gap, max_cooldown, max_cooldowns
        self.cooldowns = 0
        self.until = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        time.sleep(self.gap * random.uniform(0.6, 1.5))
        while time.time() < self.until:
            time.sleep(5)

    def ok(self) -> None:
        self.cooldowns = 0

    def cool_down(self, why: str) -> bool:
        with self.lock:
            if time.time() < self.until:  # another thread already cooling down
                return True
            self.cooldowns += 1
            if self.cooldowns > self.max_cooldowns:
                return False
            s = min(120 * 2 ** (self.cooldowns - 1), self.max_cooldown)
            print(f"  rate limited ({why[:100]}); cooling down {s} s", flush=True)
            self.until = time.time() + s
        while time.time() < self.until:
            time.sleep(5)
        return True


def classify(err: str) -> str:
    s = err.lower()
    if "sign in to confirm" in s or "not a bot" in s:
        return "auth"
    if any(x in s for x in ("429", "too many requests", "try again later", "page needs to be reloaded")):
        return "rate"
    if any(x in s for x in ("private video", "unavailable", "removed", "members-only", "copyright", "age",
                            "not available", "terminated", "blocked")):
        return "gone"
    return "fail"


def ydl(cookies: Path | None, **extra):
    import yt_dlp
    opts = dict(quiet=True, no_warnings=True, skip_download=True, noprogress=True, js_runtimes={"node": {}})
    opts.update(extra)
    if cookies:
        opts["cookiefile"] = str(cookies)
    return yt_dlp.YoutubeDL(opts)


def _entry(e: dict, group: str, query: str) -> dict:
    return {"id": e.get("id"), "title": e.get("title"), "channel": e.get("channel") or e.get("uploader"),
            "channel_id": e.get("channel_id") or e.get("uploader_id"), "duration": e.get("duration"),
            "views": e.get("view_count"), "group": group, "query": query}


# ---------------------------------------------------------------- discover
def cmd_discover(args) -> None:
    from harvest_queries import all_queries
    out = HARVEST / "yt"
    searched = {r["query"] for r in read_jsonl(out / "searched.jsonl")}
    todo = [(g, q) for g, q in all_queries() if q not in searched]
    if args.groups:
        todo = [(g, q) for g, q in todo if g in args.groups]
    print(f"{len(todo)} queries to search ({len(searched)} done)", flush=True)
    pacer = Pacer(args.gap)
    for i, (group, query) in enumerate(todo, 1):
        while True:
            try:
                with ydl(args.cookies, extract_flat="in_playlist") as y:
                    info = y.extract_info(f"ytsearch{args.n}:{query}", download=False)
                pacer.ok()
                break
            except Exception as e:  # noqa: BLE001
                if classify(str(e)) in ("rate", "auth") and pacer.cool_down(str(e)):
                    continue
                print(f"  search failed {query!r}: {str(e)[:120]}", flush=True)
                info = None
                break
        entries = [x for x in (info or {}).get("entries") or [] if x and x.get("id")]
        for e in entries:
            append(out / "candidates.jsonl", _entry(e, group, query))
        append(out / "searched.jsonl", {"query": query, "group": group, "hits": len(entries)})
        if i % 25 == 0:
            print(f"  {i}/{len(todo)} searched", flush=True)
        pacer.wait()


# ---------------------------------------------------------------- channels
def cmd_channels(args) -> None:
    """Crawl the uploads of channels whose search hits were mostly du'as."""
    out = HARVEST / "yt"
    cands = read_jsonl(out / "candidates.jsonl")
    done = {r["channel_id"] for r in read_jsonl(out / "channels.jsonl")}
    hits: dict[str, Counter] = defaultdict(Counter)
    names = {}
    for c in cands:
        ch = c.get("channel_id")
        if not ch or c.get("group") == "channel":
            continue
        names[ch] = c.get("channel")
        hits[ch]["all"] += 1
        if DUA_HINT.search(c.get("title") or "") and not excluded(c):
            hits[ch]["dua"] += 1
    ranked = sorted((ch for ch in hits if hits[ch]["dua"] >= args.min_hits and ch not in done),
                    key=lambda ch: -hits[ch]["dua"])
    ranked = [ch for ch in ranked if not splits.is_test_upload({"channel_id": ch})]
    print(f"{len(ranked)} channels to crawl", flush=True)
    pacer = Pacer(args.gap)
    for ch in ranked[: args.limit or None]:
        n_new = 0
        for tab in ("videos", "streams"):
            url = f"https://www.youtube.com/channel/{ch}/{tab}"
            try:
                with ydl(args.cookies, extract_flat="in_playlist", playlistend=args.max_per_channel) as y:
                    info = y.extract_info(url, download=False)
                pacer.ok()
            except Exception as e:  # noqa: BLE001
                if classify(str(e)) in ("rate", "auth"):
                    pacer.cool_down(str(e))
                continue
            for e in (info or {}).get("entries") or []:
                if not e or not e.get("id"):
                    continue
                if not DUA_HINT.search(e.get("title") or ""):
                    continue
                e.setdefault("channel_id", ch)
                e.setdefault("channel", names.get(ch))
                append(out / "candidates.jsonl", _entry(e, "channel", f"channel:{ch}"))
                n_new += 1
            pacer.wait()
        append(out / "channels.jsonl", {"channel_id": ch, "channel": names.get(ch), "dua_hits": hits[ch]["dua"],
                                        "new": n_new})
        print(f"  {names.get(ch)}: {n_new} du'a uploads", flush=True)


# ---------------------------------------------------------------- captioned
def cmd_import_captioned(args) -> None:
    """Queue the uploads find_captioned.py found with a manual Arabic track."""
    out = HARVEST / "yt"
    have = {c["id"] for c in read_jsonl(out / "candidates.jsonl") if c.get("group") == "captioned"}
    n = 0
    for r in read_jsonl(ROOT / "data" / "captioned" / "scanned.jsonl"):
        if not r.get("has_ar") or r["id"] in have or not DUA_HINT.search(r.get("title") or ""):
            continue
        append(out / "candidates.jsonl", {"id": r["id"], "title": r.get("title"), "channel": r.get("channel"),
                                          "channel_id": r.get("channel_id"), "duration": r.get("duration"),
                                          "views": None, "group": "captioned", "query": "find_captioned"})
        have.add(r["id"])
        n += 1
    print(f"{n} captioned uploads queued")


# ---------------------------------------------------------------- download
GROUP_RANK = {"captioned": -1, "ordinary": 0, "popular": 1, "corpus": 2, "ramadan": 3, "sahifa": 3, "reciter": 4, "channel": 5}


def plan(args) -> list[dict]:
    out = HARVEST / "yt"
    have = {r["id"] for r in read_jsonl(out / "index.jsonl")}
    # Transient failures (YouTube's 403s on media URLs) get two more tries in later rounds.
    fails = Counter(r["id"] for r in read_jsonl(out / "failed.jsonl"))
    gone = {r["id"] for r in read_jsonl(out / "failed.jsonl") if r.get("kind") == "gone"}
    gone |= {vid for vid, n in fails.items() if n >= 3}
    old = set()
    for p in (ROOT / "data" / "youtube").glob("*/*.json"):
        old.add(p.stem.replace(".labels", ""))
    best: dict[str, dict] = {}
    queries: dict[str, set] = defaultdict(set)
    for c in read_jsonl(out / "candidates.jsonl"):
        vid = c["id"]
        queries[vid].add(c["query"])
        if vid not in best or GROUP_RANK.get(c["group"], 9) < GROUP_RANK.get(best[vid]["group"], 9):
            best[vid] = c
    rows, why = [], Counter()
    for vid, c in best.items():
        d = c.get("duration") or 0
        if vid in have or vid in gone or vid in old:
            continue
        r = excluded(c)
        if r:
            why[r.split(" (")[0]] += 1
            continue
        min_s = 12 if c["group"] == "ordinary" else args.min_s  # a child's Dua Faraj is ~20 s
        if not (min_s <= d <= args.max_s):
            why["duration"] += 1
            continue
        if c["group"] == "channel" and not DUA_HINT.search(c.get("title") or ""):
            why["channel off-topic"] += 1
            continue
        if c["group"] in ("corpus", "popular", "reciter", "ramadan", "sahifa") and not DUA_HINT.search(
                f"{c.get('title') or ''} {c['query']}"):
            why["off-topic"] += 1
            continue
        c = dict(c, n_queries=len(queries[vid]))
        rows.append(c)
    # Search hits before whole-channel crawls (the crawls are 4x larger and less often
    # du'as); within each, channels interleaved round-robin, so the first hours of
    # downloading cover as many voices as possible.
    order = []
    for tier in (lambda c: c["group"] != "channel", lambda c: c["group"] == "channel"):
        by_ch: dict[str, list] = defaultdict(list)
        for c in sorted((c for c in rows if tier(c)),
                        key=lambda c: (GROUP_RANK.get(c["group"], 9), -c["n_queries"], -(c.get("views") or 0))):
            by_ch[c.get("channel_id") or c["id"]].append(c)
        while any(by_ch.values()):
            for ch in list(by_ch):
                if by_ch[ch]:
                    order.append(by_ch[ch].pop(0))
                else:
                    del by_ch[ch]
    print(f"plan: {len(order)} to download, {sum((c.get('duration') or 0) for c in order) / 3600:.0f} h; "
          f"skipped {dict(why)}", flush=True)
    return order


def cmd_download(args) -> None:
    out = HARVEST / "yt"
    audio = out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    order = plan(args)
    if args.limit:
        order = order[: args.limit]
    pacer = Pacer(args.gap)
    q = list(reversed(order))
    stats = Counter()
    t0 = time.time()

    def worker() -> None:
        while True:
            with _lock:
                if not q:
                    return
                c = q.pop()
            if shutil.disk_usage(audio).free < args.min_free_gb * 1e9:
                print("  disk nearly full: stopping", flush=True)
                return
            vid = c["id"]
            # Human Arabic caption tracks come along for free: one line per cue,
            # vowelled, timed by a person (auto captions are not requested).
            opts = dict(format="bestaudio[abr<=96]/bestaudio[ext=m4a]/bestaudio", skip_download=False,
                        outtmpl=str(audio / "%(id)s.%(ext)s"), retries=3, socket_timeout=60,
                        overwrites=False, continuedl=True, writesubtitles=True,
                        subtitleslangs=["ar", "ar-.*", "ar_.*"], subtitlesformat="vtt/srt/best")
            info = None
            tries = 0
            while True:
                tries += 1
                try:
                    with ydl(args.cookies if pacer.cooldowns or args.always_cookies else None, **opts) as y:
                        info = y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
                    pacer.ok()
                    break
                except Exception as e:  # noqa: BLE001
                    kind = classify(str(e))
                    if kind in ("rate", "auth") and pacer.cool_down(str(e)):
                        continue
                    if "403" in str(e) and tries < 3:
                        time.sleep(3)
                        continue
                    append(out / "failed.jsonl", {"id": vid, "kind": kind, "err": str(e)[:300]})
                    stats[kind] += 1
                    break
            if info:
                files = [p for p in audio.glob(f"{vid}.*") if p.suffix not in (".part", ".ytdl", ".json", ".vtt", ".srt")]
                subs = sorted(p.name for p in audio.glob(f"{vid}.*") if p.suffix in (".vtt", ".srt"))
                if files:
                    meta = {**c, "audio": files[0].name, "duration": info.get("duration") or c.get("duration"),
                            "upload_date": info.get("upload_date"), "uploader_id": info.get("uploader_id"),
                            "channel_id": info.get("channel_id") or c.get("channel_id"),
                            "channel": info.get("channel") or c.get("channel"),
                            "description": (info.get("description") or "")[:500],
                            "language": info.get("language"), "tags": (info.get("tags") or [])[:20], "subs": subs,
                            "url": f"https://www.youtube.com/watch?v={vid}"}
                    # A late check with the full metadata (uploader ids, description).
                    r = excluded({**meta, "title": f"{meta['title']} {meta['description'][:200]}"})
                    if r.startswith("test"):
                        files[0].unlink()
                        append(out / "failed.jsonl", {"id": vid, "kind": "gone", "err": r})
                        stats["excluded late"] += 1
                    else:
                        append(out / "index.jsonl", meta)
                        stats["ok"] += 1
                        stats["s"] += meta["duration"] or 0
            n = stats["ok"]
            if n and n % 20 == 0:
                print(f"  {n} downloaded, {stats['s'] / 3600:.1f} h, {dict((k, v) for k, v in stats.items() if k != 's')}, "
                      f"{(time.time() - t0) / 60:.0f} min", flush=True)
            pacer.wait()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"done: {dict(stats)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["discover", "channels", "download", "plan", "import-captioned"])
    ap.add_argument("--cookies", type=Path, help="Netscape cookies.txt (used after a rate limit / bot check)")
    ap.add_argument("--always-cookies", action="store_true")
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("-n", type=int, default=40, help="results per search")
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--min-hits", type=int, default=3, help="channels: du'a hits needed to crawl")
    ap.add_argument("--max-per-channel", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-j", "--jobs", type=int, default=2)
    ap.add_argument("--min-s", type=float, default=30)
    ap.add_argument("--max-s", type=float, default=3 * 3600)
    ap.add_argument("--min-free-gb", type=float, default=40)
    args = ap.parse_args()
    if args.stage == "plan":
        plan(args)
    else:
        {"discover": cmd_discover, "channels": cmd_channels, "download": cmd_download,
         "import-captioned": cmd_import_captioned}[args.stage](args)


if __name__ == "__main__":
    main()
