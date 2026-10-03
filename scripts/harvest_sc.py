#!/usr/bin/env python
"""SoundCloud side of the recitation harvest (yt-dlp's scsearch + downloads).

    python scripts/harvest_sc.py discover
    python scripts/harvest_sc.py download -j 3

Queries: the widely recited texts (harvest_queries.POPULAR, every spelling) and
the reciter queries. Same filters as scripts/harvest.py. State in
data/harvest/soundcloud/.
"""
from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from harvest import DUA_HINT, HARVEST, append, excluded, read_jsonl, ydl  # noqa: E402
from harvest_queries import POPULAR, RECITER_DUAS, RECITERS  # noqa: E402

OUT = HARVEST / "soundcloud"


def queries() -> list[str]:
    qs = [n for names in POPULAR for n in names]
    qs += [f"{d} {r}" for r in RECITERS[:20] for d in RECITER_DUAS[:2]]
    seen, out = set(), []
    for q in qs:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def cmd_discover(args) -> None:
    done = {r["query"] for r in read_jsonl(OUT / "searched.jsonl")}
    todo = [q for q in queries() if q not in done]
    print(f"{len(todo)} queries", flush=True)
    for i, q in enumerate(todo, 1):
        try:
            with ydl(None, extract_flat="in_playlist") as y:
                info = y.extract_info(f"scsearch{args.n}:{q}", download=False)
        except Exception as e:  # noqa: BLE001
            print(f"  failed {q!r}: {str(e)[:100]}", flush=True)
            info = None
        n = 0
        for e in (info or {}).get("entries") or []:
            if not e or not e.get("url"):
                continue
            append(OUT / "candidates.jsonl", {
                "id": f"sc-{e.get('id')}", "url": e.get("url"), "title": e.get("title"),
                "channel": e.get("uploader"), "channel_id": f"sc:{e.get('uploader_id') or e.get('uploader')}",
                "duration": e.get("duration") or 0, "views": e.get("view_count"), "group": "popular", "query": q})
            n += 1
        append(OUT / "searched.jsonl", {"query": q, "hits": n})
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
        time.sleep(args.gap * random.uniform(0.5, 1.5))


def cmd_download(args) -> None:
    have = {r["id"] for r in read_jsonl(OUT / "index.jsonl")}
    fails = Counter(r["id"] for r in read_jsonl(OUT / "failed.jsonl"))
    best, why = {}, Counter()
    for c in read_jsonl(OUT / "candidates.jsonl"):
        best.setdefault(c["id"], c)
    rows = []
    for c in best.values():
        if c["id"] in have or fails[c["id"]] >= 2:
            continue
        r = excluded(c)
        if r:
            why[r.split(" (")[0]] += 1
            continue
        if not (args.min_s <= (c.get("duration") or 0) <= args.max_s):
            why["duration"] += 1
            continue
        if not DUA_HINT.search(f"{c.get('title') or ''} {c['query']}"):
            why["off-topic"] += 1
            continue
        rows.append(c)
    by = defaultdict(list)
    for c in rows:
        by[c["channel_id"]].append(c)
    order = []
    while by:
        for k in list(by):
            order.append(by[k].pop(0))
            if not by[k]:
                del by[k]
    if args.limit:
        order = order[: args.limit]
    print(f"plan: {len(order)}, {sum(c['duration'] for c in order) / 3600:.0f} h; skipped {dict(why)}", flush=True)
    audio = OUT / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    q = list(reversed(order))
    lock = threading.Lock()
    stats = Counter()

    def worker():
        while True:
            with lock:
                if not q:
                    return
                c = q.pop()
            try:
                with ydl(None, skip_download=False, format="bestaudio/best",
                         outtmpl=str(audio / f"{c['id']}.%(ext)s")) as y:
                    y.extract_info(c["url"], download=True)
                files = [p for p in audio.glob(f"{c['id']}.*") if p.suffix not in (".part", ".ytdl")]
                if not files:
                    raise RuntimeError("no file")
            except Exception as e:  # noqa: BLE001
                append(OUT / "failed.jsonl", {"id": c["id"], "err": str(e)[:200]})
                stats["fail"] += 1
                continue
            append(OUT / "index.jsonl", {**c, "audio": files[0].name})
            with lock:
                stats["ok"] += 1
                stats["s"] += c["duration"]
                if stats["ok"] % 25 == 0:
                    print(f"  {stats['ok']} tracks, {stats['s'] / 3600:.1f} h, fail {stats['fail']}", flush=True)
            time.sleep(args.gap * random.uniform(0.5, 1.5))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"done: {dict(stats)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["discover", "download"])
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("-j", "--jobs", type=int, default=3)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-s", type=float, default=30)
    ap.add_argument("--max-s", type=float, default=3 * 3600)
    args = ap.parse_args()
    {"discover": cmd_discover, "download": cmd_download}[args.stage](args)


if __name__ == "__main__":
    main()
