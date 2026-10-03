#!/usr/bin/env python
"""Aparat (aparat.com) side of the recitation harvest: Iran's video site,
where most Persian du'a recitations live (Kumayl, Tawassul, Nudba, Ziyarat
Ashura by mosque and hay'at reciters, plus home recordings).

yt-dlp's Aparat extractor is broken, so this talks to Aparat's own JSON API:
search pages (10 videos each, a `next` link) and the video page's file list,
from which the smallest MP4 is fetched and its audio track kept (AAC, 22 kHz
mono at 144p: plenty for 16 kHz models).

    python scripts/harvest_aparat.py discover
    python scripts/harvest_aparat.py download -j 4

Same rules as scripts/harvest.py (its excluded() filter: test reciters by
name in either script, test venues, lectures). State in data/harvest/aparat/.
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from harvest import DUA_HINT, HARVEST, append, excluded, read_jsonl  # noqa: E402
from harvest_queries import all_queries  # noqa: E402

OUT = HARVEST / "aparat"
API = "https://www.aparat.com/api/fa/v1"
HEAD = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36",
        "Accept": "application/json"}
PERSIAN = set("پچژگکی")


def get(url: str, params=None, tries: int = 4):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEAD, timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                time.sleep(30 * (k + 1))
                continue
            return None
        except Exception:  # noqa: BLE001
            time.sleep(5 * (k + 1))
    return None


def cmd_discover(args) -> None:
    searched = {r["query"] for r in read_jsonl(OUT / "searched.jsonl")}
    qs = [(g, q) for g, q in all_queries() if not q.isascii()]
    # Persian spellings first: that's what Aparat titles use.
    qs.sort(key=lambda gq: 0 if PERSIAN & set(gq[1]) else 1)
    todo = [(g, q) for g, q in qs if q not in searched]
    print(f"{len(todo)} queries ({len(searched)} done)", flush=True)
    for i, (group, query) in enumerate(todo, 1):
        url, params, n = f"{API}/video/video/search/text/{query}", {"type_search": "search"}, 0
        for _page in range(args.pages):
            d = get(url, params)
            if not d:
                break
            for x in d.get("included") or []:
                if x.get("type") != "Video":
                    continue
                a = x["attributes"]
                append(OUT / "candidates.jsonl", {
                    "id": a.get("uid"), "num_id": a.get("id"), "title": a.get("title"),
                    "description": (a.get("description") or "")[:300], "channel": a.get("sender_name"),
                    "channel_id": a.get("username"), "duration": int(a.get("duration") or 0),
                    "views": int(a.get("visit_cnt_int") or 0), "group": group, "query": query})
                n += 1
            try:
                url, params = d["data"][0]["attributes"]["link"]["next"], None
            except (KeyError, IndexError, TypeError):
                break
            if not url:
                break
            time.sleep(args.gap * random.uniform(0.5, 1.5))
        append(OUT / "searched.jsonl", {"query": query, "group": group, "hits": n})
        if i % 25 == 0:
            print(f"  {i}/{len(todo)} searched", flush=True)
        time.sleep(args.gap * random.uniform(0.5, 1.5))


def plan(args) -> list[dict]:
    have = {r["id"] for r in read_jsonl(OUT / "index.jsonl")}
    fails = Counter(r["id"] for r in read_jsonl(OUT / "failed.jsonl"))
    best: dict[str, dict] = {}
    nq: dict[str, set] = defaultdict(set)
    for c in read_jsonl(OUT / "candidates.jsonl"):
        if not c.get("id"):
            continue
        nq[c["id"]].add(c["query"])
        best.setdefault(c["id"], c)
    rows, why = [], Counter()
    for vid, c in best.items():
        if vid in have or fails[vid] >= 2:
            continue
        r = excluded({**c, "title": f"{c.get('title') or ''} {c.get('description') or ''}"})
        if r:
            why[r.split(" (")[0]] += 1
            continue
        if not (args.min_s <= (c.get("duration") or 0) <= args.max_s):
            why["duration"] += 1
            continue
        if not DUA_HINT.search(f"{c.get('title') or ''} {c.get('description') or ''}"):
            why["off-topic"] += 1
            continue
        rows.append(dict(c, n_queries=len(nq[vid])))
    by_ch: dict[str, list] = defaultdict(list)
    for c in sorted(rows, key=lambda c: (-c["n_queries"], -(c.get("views") or 0))):
        by_ch[c.get("channel_id") or c["id"]].append(c)
    order = []
    while by_ch:
        for ch in list(by_ch):
            order.append(by_ch[ch].pop(0))
            if not by_ch[ch]:
                del by_ch[ch]
    print(f"plan: {len(order)} to download, {sum(c['duration'] for c in order) / 3600:.0f} h; skipped {dict(why)}",
          flush=True)
    return order


def fetch(c: dict) -> str:
    """Download one video's audio; '' on success, else the reason."""
    d = get(f"{API}/video/video/show/videohash/{c['id']}", {"pr": 1, "mf": 1})
    try:
        files = d["data"]["attributes"]["file_link_all"] or []
    except (KeyError, TypeError):
        return "no metadata"
    if not files:
        return "no files"
    url = files[0]["urls"][0]  # lowest quality first
    audio = OUT / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    tmp = audio / f"{c['id']}.part.mp4"
    try:
        with requests.get(url, headers=HEAD, timeout=60, stream=True) as r:
            if r.status_code != 200:
                return f"http {r.status_code}"
            with tmp.open("wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return f"download: {str(e)[:100]}"
    out = audio / f"{c['id']}.m4a"
    p = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-vn", "-c:a", "copy", str(out)],
                       capture_output=True, text=True)
    tmp.unlink(missing_ok=True)
    if p.returncode != 0 or not out.exists():
        out.unlink(missing_ok=True)
        return f"ffmpeg: {p.stderr[-100:]}"
    return ""


def cmd_download(args) -> None:
    order = plan(args)
    if args.limit:
        order = order[: args.limit]
    q = list(reversed(order))
    lock = threading.Lock()
    stats = Counter()
    t0 = time.time()

    def worker():
        while True:
            with lock:
                if not q:
                    return
                c = q.pop()
            why = fetch(c)
            if why:
                append(OUT / "failed.jsonl", {"id": c["id"], "err": why})
                stats["fail"] += 1
            else:
                append(OUT / "index.jsonl", {**c, "audio": f"{c['id']}.m4a",
                                             "url": f"https://www.aparat.com/v/{c['id']}"})
                stats["ok"] += 1
                stats["s"] += c["duration"]
                if stats["ok"] % 20 == 0:
                    print(f"  {stats['ok']} downloaded, {stats['s'] / 3600:.1f} h, fail {stats['fail']}, "
                          f"{(time.time() - t0) / 60:.0f} min", flush=True)
            time.sleep(args.gap * random.uniform(0.5, 1.5))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"done: {dict(stats)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["discover", "download", "plan"])
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-s", type=float, default=30)
    ap.add_argument("--max-s", type=float, default=3 * 3600)
    args = ap.parse_args()
    {"discover": cmd_discover, "download": cmd_download, "plan": plan}[args.stage](args)


if __name__ == "__main__":
    main()
