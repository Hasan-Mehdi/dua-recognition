#!/usr/bin/env python
"""Shia audio libraries on the open web: the recitation harvest's third front.

shiavoice.com files its du'a and ziyarat recordings by text (one category per
du'a: "دعاء أبي حمزة الثمالي" holds 90 reciters) and serves every category's
track list as JSON (/cat-N/tracks: title, reciter, duration, /stream-ID mp3).
The crawl walks the category tree under /ed3ie (الأدعية والمناجيات) and
/zeyarat (الزيارات); what each recording really holds is still decided from
its audio (harvest_label.py).

    python scripts/harvest_web.py discover
    python scripts/harvest_web.py download -j 4

Same filters as scripts/harvest.py (test reciters by name, lectures).
State in data/harvest/web/ (platform "web").
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from harvest import HARVEST, append, excluded, read_jsonl  # noqa: E402

OUT = HARVEST / "web"
SV = "https://shiavoice.com"
HEAD = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"}


def _get(url, **kw):
    for k in range(4):
        try:
            r = requests.get(url, headers={**HEAD, **kw.pop("headers", {})}, timeout=60, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 503):
                time.sleep(20 * (k + 1))
                continue
            return None
        except Exception:  # noqa: BLE001
            time.sleep(5 * (k + 1))
    return None


def _cats(page_html: str) -> list[tuple[str, str]]:
    out = []
    for m in re.finditer(r'<a[^>]+href="(?:https://shiavoice\.com)?/(cat-\d+)"[^>]*>(.*?)</a>', page_html, re.S):
        t = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m[2]))).strip()
        out.append((m[1], t))
    return out


def _dur(s: str) -> float:
    try:
        parts = [float(x) for x in str(s).split(":")]
    except ValueError:
        return 0.0
    v = 0.0
    for p in parts:
        v = v * 60 + p
    return v


def cmd_discover(args) -> None:
    """Breadth-first over categories reachable from the du'a and ziyarat sections."""
    seen_cat = {r["cat"] for r in read_jsonl(OUT / "cats.jsonl")}
    have = {r["id"] for r in read_jsonl(OUT / "candidates.jsonl")}
    frontier = []
    for section in ("/ed3ie", "/zeyarat"):
        r = _get(SV + section)
        if r:
            frontier += [(c, t, section) for c, t in _cats(r.text)]
    # Category pages list their sub-categories; the side bars list unrelated ones
    # (latmiyat releases), so only follow links whose page is in a du'a section.
    depth = {c: 0 for c, _, _ in frontier}
    n_tracks = 0
    while frontier:
        cat, title, section = frontier.pop(0)
        if cat in seen_cat:
            continue
        seen_cat.add(cat)
        r = _get(f"{SV}/{cat}/tracks", headers={"Accept": "application/json"})
        tracks = []
        try:
            tracks = r.json() if r else []
        except ValueError:
            tracks = []
        for t in tracks:
            uid = f"sv-{t['uniqid']}"
            if uid in have:
                continue
            have.add(uid)
            append(OUT / "candidates.jsonl", {
                "id": uid, "title": f"{title} / {t.get('title')}", "channel": t.get("artist"),
                "channel_id": f"sv:{t.get('artistHref')}", "duration": _dur(t.get("duration")),
                "stream": SV + t["url"], "url": SV + t.get("href", ""), "group": section.strip("/"),
                "query": cat, "site": "shiavoice"})
            n_tracks += 1
        append(OUT / "cats.jsonl", {"cat": cat, "title": title, "section": section, "tracks": len(tracks)})
        print(f"  {cat} {title[:40]}: {len(tracks)} tracks (total new {n_tracks})", flush=True)
        if depth.get(cat, 0) < 2:
            page = _get(f"{SV}/{cat}")
            if page:
                for c, t in _cats(page.text):
                    if c not in seen_cat and c not in depth:
                        depth[c] = depth.get(cat, 0) + 1
                        frontier.append((c, t, section))
        time.sleep(args.gap)
    print(f"done: {n_tracks} new tracks", flush=True)


def cmd_download(args) -> None:
    have = {r["id"] for r in read_jsonl(OUT / "index.jsonl")}
    fails = Counter(r["id"] for r in read_jsonl(OUT / "failed.jsonl"))
    rows, why = [], Counter()
    for c in read_jsonl(OUT / "candidates.jsonl"):
        if c["id"] in have or fails[c["id"]] >= 2:
            continue
        r = excluded(c)
        if r:
            why[r.split(" (")[0]] += 1
            continue
        if c["duration"] and not (args.min_s <= c["duration"] <= args.max_s):
            why["duration"] += 1
            continue
        rows.append(c)
    # Interleave reciters so the first hours cover as many voices as possible.
    by: dict[str, list] = {}
    for c in rows:
        by.setdefault(c.get("channel_id") or c["id"], []).append(c)
    order = []
    while by:
        for k in list(by):
            order.append(by[k].pop(0))
            if not by[k]:
                del by[k]
    if args.limit:
        order = order[: args.limit]
    print(f"plan: {len(order)} tracks, {sum(c['duration'] for c in order) / 3600:.0f} h; skipped {dict(why)}",
          flush=True)
    audio = OUT / "audio"
    audio.mkdir(parents=True, exist_ok=True)
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
            dst = audio / f"{c['id']}.mp3"
            try:
                with requests.get(c["stream"], headers=HEAD, timeout=90, stream=True) as r:
                    if r.status_code not in (200, 206):
                        raise RuntimeError(f"http {r.status_code}")
                    tmp = dst.with_suffix(".part")
                    with tmp.open("wb") as f:
                        for block in r.iter_content(1 << 20):
                            f.write(block)
                    tmp.replace(dst)
            except Exception as e:  # noqa: BLE001
                append(OUT / "failed.jsonl", {"id": c["id"], "err": str(e)[:200]})
                stats["fail"] += 1
                continue
            append(OUT / "index.jsonl", {**c, "audio": dst.name})
            with lock:
                stats["ok"] += 1
                stats["s"] += c["duration"]
                if stats["ok"] % 50 == 0:
                    print(f"  {stats['ok']} tracks, {stats['s'] / 3600:.1f} h, fail {stats['fail']}, "
                          f"{(time.time() - t0) / 60:.0f} min", flush=True)
            time.sleep(args.gap)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"done: {dict(stats)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["discover", "download"])
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-s", type=float, default=20)
    ap.add_argument("--max-s", type=float, default=3 * 3600)
    args = ap.parse_args()
    {"discover": cmd_discover, "download": cmd_download}[args.stage](args)


if __name__ == "__main__":
    main()
