#!/usr/bin/env python
"""Fetch audio for the held-out test sets: majlis nights and ordinary voices.

Every test voice today is a professional reciter in a studio. These sets are
what the app actually faces: a du'a night in a centre (a PA, a crowd
answering), and ordinary people reading on a phone. Nothing here is ever
used for training: the venues are splits.TEST_CHANNELS, and the private
uploaders' channel and video ids go to data/testsets/<set>/holdout.json, which
every training harvester reads (splits.is_test_upload).

    python scripts/fetch_testset.py majlis                 # index + download the test venues' du'a nights
    python scripts/fetch_testset.py slides                 # a few frames per venue: are line slides projected?
    python scripts/fetch_testset.py amateur-search         # -> data/testsets/amateur/candidates.md, to tick
    python scripts/fetch_testset.py amateur                # download the ticked candidates (+ --urls FILE)

Majlis: each venue channel's /streams and /videos are indexed and titles are
ranked by du'a name (fetch_fatemah.GROUPS); at most --venue-hours of raw
stream per venue is downloaded, spread over du'as, to data/testsets/majlis/streams.
segment_streams.py --all-test then cuts the du'a spans out.

Ordinary voices: YouTube searches in English, Urdu, Arabic and Persian for
personal, school and ladies' majlis recitations (women's and children's voices
on purpose: the test set has neither), 3-40 min, few views, no known reciter.
The candidates are for a person to tick; nothing is downloaded unticked.
Direct audio links (e.g. erfan.ir mp3s) can be listed in --urls too.

Pacing and rate-limit handling come from find_captioned.py: one YouTube
client at a time. Runs in whichever Python has yt-dlp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_fatemah import AMAAL, groups  # noqa: E402
from find_captioned import Pacer, _append, _read_jsonl, _splits, classify, ydl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "testsets"
MAJLIS, AMATEUR = OUT / "majlis", OUT / "amateur"

# A du'a night, not a lecture or a latmiya that happens to name a du'a.
SKIP = re.compile(r"(lecture|khutbah?|sermon|q&a|nasheed|qasida|noha|nauha|latmiya|matam|speech|class|"
                  r"reflection|talk|interview|quiz|introduction|tafseer|jashn|wiladat|lessons?|explained|"
                  r"explanation|meaning|shorts|#shorts|madrasa graduation)", re.I)

AMATEUR_QUERIES = [
    # English
    "my dua kumayl recitation", "dua kumail recited by me", "dua tawassul recitation practice",
    "dua kumail recitation by sister", "ladies majlis dua kumail", "dua kumayl recited by student",
    "kids reciting dua tawassul", "child recites dua kumail", "girl reciting dua kumail",
    "dua ahad recitation by student", "ziyarat ashura recited at home", "dua e nudba recitation home",
    "madrasa dua competition recitation", "dua kumail recitation competition", "dua tawassul by my son",
    "dua iftitah recitation home ramadan", "ziyarat warith recitation", "ziyarat aminullah recited by",
    # Urdu (the same, in Urdu script and Roman Urdu)
    "دعائے کمیل خواتین مجلس", "دعائے توسل بچے", "بچی دعائے کمیل پڑھ رہی ہے", "dua e kumail by baji",
    "dua e tawassul by bachi", "dua e kumail ladies majlis", "zyarat e ashura ladies",
    # Arabic
    "دعاء كميل بصوت طفل", "دعاء التوسل بصوت طفلة", "دعاء كميل بصوت امرأة", "دعاء كميل بصوتي",
    "قراءة دعاء التوسل في البيت", "دعاء العهد بصوت طالب", "مسابقة قراءة الدعاء",
    # Persian
    "دعای کمیل با صدای کودک", "دعای توسل با صدای دختر", "دعای کمیل با صدای خودم", "قرائت دعای عهد دانش آموز",
    "دعای توسل خانم", "مسابقه قرائت دعا دانش آموزی",
]
# Professionals the view-count filter can miss (beyond splits.KNOWN_RECITERS).
PROFESSIONALS = ["karbalaei", "كربلائي", "کربلایی", "samavati", "سماواتی", "mazeedi", "المزيدي", "maliki",
                 "المالكي", "abdul hay", "عبدالحي", "ansariyan", "انصاریان", "haj mahdi", "sibt", "nadeem sarwar",
                 "ali safdar", "mir hasan mir", "abbas haider", "farhan ali waris", "official"]


def _fold(s: str) -> str:
    return re.sub(r"[ً-ْٰ]", "", s or "").translate(str.maketrans("أإآ", "ااا")).lower()


def _download(args, url: str, out_dir: Path, vid: str) -> bool:
    """Audio only, into out_dir/<vid>.<ext>; YouTube via yt-dlp, anything else as a plain file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if "youtube.com" not in url and "youtu.be" not in url:
        ext = Path(url.split("?")[0]).suffix or ".mp3"
        urllib.request.urlretrieve(url, out_dir / f"{vid}{ext}")
        return True
    pacer = Pacer(args.gap)
    while True:
        try:
            with ydl(args, outtmpl=str(out_dir / "%(id)s.%(ext)s"), format="bestaudio[abr<=96]/bestaudio") as y:
                y.params["skip_download"] = False
                y.download([url])
            return True
        except Exception as e:  # noqa: BLE001
            if classify(str(e)) == "rate" and pacer.cool_down(str(e)):
                continue
            print(f"  download failed {vid}: {str(e)[:160]}", flush=True)
            return False


def _holdout(root: Path, channels=(), videos=()) -> None:
    """Record test uploads so training harvesters skip them (splits.is_test_upload)."""
    f = root / "holdout.json"
    h = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"channels": [], "videos": []}
    h["channels"] = sorted(set(h["channels"]) | {c for c in channels if c})
    h["videos"] = sorted(set(h["videos"]) | {v for v in videos if v})
    root.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(h, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- majlis

def majlis_score(title: str) -> int:
    g = [x for x in groups(title) if x not in ("ziyarat", "munajat", "qadr")] or groups(title)
    if SKIP.search(title) or not (g or AMAAL.search(title)):
        return 0
    return 3 * len(g) + (1 if AMAAL.search(title) else 0)


def cmd_majlis(args) -> None:
    venues = _splits().TEST_CHANNELS
    channels = args.channel or [f"https://www.youtube.com/channel/{c}" for c in venues]
    index = MAJLIS / "index.jsonl"
    seen = {r["id"] for r in _read_jsonl(index)}
    pacer = Pacer(args.gap)
    for url in channels:
        cid = url.rstrip("/").rsplit("/", 1)[-1]
        if any(r.get("channel_id") == cid for r in _read_jsonl(index)) and not args.reindex:
            continue
        for tab in ("streams", "videos"):
            while True:
                try:
                    with ydl(args, extract_flat="in_playlist") as y:
                        info = y.extract_info(f"{url}/{tab}", download=False)
                    pacer.ok()
                    break
                except Exception as e:  # noqa: BLE001
                    if classify(str(e)) == "rate" and pacer.cool_down(str(e)):
                        continue
                    print(f"  {cid} /{tab}: {str(e)[:120]}", flush=True)
                    info = {"entries": []}
                    break
            n = 0
            for e in info.get("entries") or []:
                if not e or e.get("id") in seen:
                    continue
                seen.add(e["id"])
                _append(index, {"id": e["id"], "title": e.get("title") or "", "duration": e.get("duration"),
                                "channel_id": cid, "venue": venues.get(cid, info.get("channel") or cid), "tab": tab,
                                "score": majlis_score(e.get("title") or "")})
                n += 1
            print(f"  {venues.get(cid, cid)} /{tab}: {n} entries", flush=True)
            pacer.wait()
    rows = _read_jsonl(index)
    # Per venue: the best-titled du'a nights, round-robin over du'as, within the raw budget.
    streams = MAJLIS / "streams"
    picked = []
    for cid in sorted({r["channel_id"] for r in rows}):
        ok = [r for r in rows if r["channel_id"] == cid and r["score"] > 0
              and args.min_minutes * 60 <= (r.get("duration") or 0) <= args.max_minutes * 60]
        by: dict[str, list[dict]] = {}
        for r in sorted(ok, key=lambda r: -r["score"]):
            by.setdefault((groups(r["title"]) or ["amaal"])[0], []).append(r)
        total, order = 0.0, []
        while any(by.values()):
            for k in list(by):
                if by[k]:
                    order.append(by[k].pop(0))
        for r in order:
            if total + r["duration"] > args.venue_hours * 3600:
                continue
            total += r["duration"]
            picked.append(r)
        print(f"  {venues.get(cid, cid)}: {len(ok)} du'a nights indexed, {sum(1 for r in picked if r['channel_id'] == cid)} "
              f"picked ({total / 3600:.1f} h)", flush=True)
    for i, r in enumerate(picked, 1):
        if (streams / f"{r['id']}.json").exists():
            continue
        ok = _download(args, f"https://www.youtube.com/watch?v={r['id']}", streams, r["id"])
        if ok:
            (streams / f"{r['id']}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [{i}/{len(picked)}] {'ok' if ok else 'FAILED'} {r['venue'][:20]} {r['id']} "
              f"{r['duration'] / 60:.0f} min: {r['title'][:60]}", flush=True)
        pacer.wait()
    _holdout(MAJLIS, channels=venues, videos=[r["id"] for r in picked])


def cmd_slides(args) -> None:
    """A few 360p frames from two streams per venue, to see whether line slides are projected."""
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((MAJLIS / "streams").glob("*.json"))]
    frames = MAJLIS / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    per: dict[str, int] = {}
    pacer = Pacer(args.gap)
    for r in rows:
        if per.get(r["channel_id"], 0) >= 2:
            continue
        per[r["channel_id"]] = per.get(r["channel_id"], 0) + 1
        for frac in (0.3, 0.5, 0.7):
            t = int(r["duration"] * frac)
            out = frames / f"{re.sub(r'[^A-Za-z]+', '', r['venue'])[:12]}_{r['id']}_{t}.jpg"
            if out.exists():
                continue
            try:
                with ydl(args, format="best[height<=360]/worst") as y:
                    info = y.extract_info(f"https://www.youtube.com/watch?v={r['id']}", download=False)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", info["url"],
                                "-frames:v", "1", str(out)], check=True, timeout=120)
                print(f"  {out.name}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  frame failed {r['id']} @{t}: {str(e)[:120]}", flush=True)
            pacer.wait()


# ---------------------------------------------------------------- ordinary voices

def cmd_amateur_search(args) -> None:
    known = [_fold(x) for x in _splits().KNOWN_RECITERS + PROFESSIONALS]
    cands = AMATEUR / "candidates.jsonl"
    done = {r["query"] for r in _read_jsonl(AMATEUR / "searched.jsonl")}
    seen = {r["id"] for r in _read_jsonl(cands)}
    pacer = Pacer(args.gap)
    with ydl(args, extract_flat=True) as y:
        for q in AMATEUR_QUERIES:
            if q in done:
                continue
            while True:
                try:
                    res = y.extract_info(f"ytsearch{args.n}:{q}", download=False)
                    pacer.ok()
                    break
                except Exception as e:  # noqa: BLE001
                    if classify(str(e)) == "rate" and pacer.cool_down(str(e)):
                        continue
                    print(f"  search failed: {q}: {str(e)[:120]}", flush=True)
                    res = {"entries": []}
                    break
            n = 0
            for e in res.get("entries") or []:
                if not e or e.get("id") in seen:
                    continue
                text = _fold(f"{e.get('title', '')} {e.get('channel', '')} {e.get('uploader', '')}")
                dur, views = e.get("duration") or 0, e.get("view_count")
                if not 180 <= dur <= 2400 or (views is not None and views > args.max_views):
                    continue
                if any(k in text for k in known) or SKIP.search(e.get("title") or ""):
                    continue
                seen.add(e["id"])
                _append(cands, {"id": e["id"], "title": e.get("title"), "channel": e.get("channel") or e.get("uploader"),
                                "channel_id": e.get("channel_id"), "duration": dur, "views": views, "query": q})
                n += 1
            _append(AMATEUR / "searched.jsonl", {"query": q, "new": n})
            print(f"  {q}: {n} new", flush=True)
            pacer.wait()
    rows = _read_jsonl(cands)
    lines = ["# Ordinary-voice candidates", "",
             "Tick `[x]` the ones that are one ordinary person (not a professional, not a studio track), reciting a "
             "du'a in the corpus. Edit the condition word if you can tell (phone / headset / room / majlis). "
             "Then run `fetch_testset.py amateur`. These links stay local (gitignored).", ""]
    for r in sorted(rows, key=lambda r: r["query"]):
        lines.append(f"- [ ] phone | https://www.youtube.com/watch?v={r['id']} | {r['duration'] // 60} min | "
                     f"{r['views'] if r['views'] is not None else '?'} views | {(r['channel'] or '')[:30]} | "
                     f"{(r['title'] or '')[:80]} | q: {r['query']}")
    old = AMATEUR / "candidates.md"
    if old.exists():  # keep earlier ticks
        ticked = {m.group(2): m.group(1) for m in re.finditer(r"- \[x\] (\w+) \| https://www\.youtube\.com/watch\?v=([\w-]{11})",
                                                              old.read_text(encoding="utf-8"))}
        lines = [re.sub(r"^- \[ \] phone \| (https://www\.youtube\.com/watch\?v=([\w-]{11}))",
                        lambda m: f"- [x] {ticked[m.group(2)]} | {m.group(1)}" if m.group(2) in ticked else m.group(0), x)
                 for x in lines]
    old.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} candidates -> {old}", flush=True)


def cmd_amateur(args) -> None:
    todo = []  # (url, condition)
    md = AMATEUR / "candidates.md"
    if md.exists():
        todo += [(m.group(2), m.group(1)) for m in re.finditer(
            r"- \[x\] (\w+) \| (https://www\.youtube\.com/watch\?v=[\w-]{11})", md.read_text(encoding="utf-8"))]
    if args.urls:
        for line in args.urls.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and parts[0].startswith("http"):
                todo.append((parts[0], parts[1] if len(parts) > 1 else "phone"))
    by_id = {r["id"]: r for r in _read_jsonl(AMATEUR / "candidates.jsonl")}
    raw = AMATEUR / "raw"
    pacer = Pacer(args.gap)
    for url, cond in todo:
        m = re.search(r"v=([\w-]{11})", url)
        vid = m.group(1) if m else "x" + hashlib.sha1(url.encode()).hexdigest()[:10]
        if (raw / f"{vid}.json").exists():
            continue
        ok = _download(args, url, raw, vid)
        c = by_id.get(vid, {})
        if ok:
            (raw / f"{vid}.json").write_text(json.dumps({
                "id": vid, "url": url, "condition": cond, "title": c.get("title", ""), "channel": c.get("channel", ""),
                "channel_id": c.get("channel_id", ""), "duration": c.get("duration")}, ensure_ascii=False, indent=1),
                encoding="utf-8")
            _holdout(AMATEUR, channels=[c.get("channel_id")], videos=[vid] if m else [])
        print(f"  {'ok' if ok else 'FAILED'} {vid} ({cond}): {c.get('title', url)[:70]}", flush=True)
        if m:
            pacer.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["majlis", "slides", "amateur-search", "amateur"])
    ap.add_argument("--cookies", type=Path, help="Netscape cookies.txt (YouTube domains only)")
    ap.add_argument("--gap", type=float, default=20, help="seconds between requests (jittered)")
    ap.add_argument("--channel", nargs="*", help="majlis: venue channel URLs (default: splits.TEST_CHANNELS)")
    ap.add_argument("--reindex", action="store_true", help="majlis: index channels again")
    ap.add_argument("--venue-hours", type=float, default=5, help="majlis: raw stream hours per venue")
    ap.add_argument("--min-minutes", type=float, default=15)
    ap.add_argument("--max-minutes", type=float, default=180)
    ap.add_argument("-n", type=int, default=30, help="amateur-search: results per query")
    ap.add_argument("--max-views", type=int, default=20000, help="amateur-search: skip more-viewed uploads")
    ap.add_argument("--urls", type=Path, help="amateur: extra 'URL [condition]' lines (YouTube or direct audio)")
    args = ap.parse_args()
    {"majlis": cmd_majlis, "slides": cmd_slides, "amateur-search": cmd_amateur_search,
     "amateur": cmd_amateur}[args.stage](args)


if __name__ == "__main__":
    main()
