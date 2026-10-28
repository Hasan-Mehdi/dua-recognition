#!/usr/bin/env python
"""Who used the hosted page: the server's visits log (data/sessions/visits.jsonl, app/server.py
LogVisits) summed up per visitor (address + browser).

    python scripts/visits.py                  # everyone who opened the page, newest last
    python scripts/visits.py --since 2026-10-24
    python scripts/visits.py --all            # also the one-request visitors (link previews, bots)

opened: page loads. listened: taps on start (the speech model asked for: asr-worker.js on the phone
link, /ws/ear on the server engine's, with the minutes it stayed connected). sessions: debug sessions
that came in (scripts/session_report.py ID).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISITS = ROOT / "data" / "sessions" / "visits.jsonl"
LINKS = {8443: "phone", 8444: "server"}


def short_ua(ua: str) -> str:
    """'Android Chrome 155', 'Windows Chrome 155', 'iPhone Safari 18', ..."""
    os_ = next((name for key, name in [("iPhone", "iPhone"), ("iPad", "iPad"), ("Android", "Android"),
                                       ("Windows", "Windows"), ("Mac OS X", "Mac"), ("Linux", "Linux")]
                if key in ua), "?")
    for key, name in [("Edg/", "Edge"), ("SamsungBrowser/", "Samsung"), ("CriOS/", "Chrome"), ("FxiOS/", "Firefox"),
                      ("Firefox/", "Firefox"), ("Chrome/", "Chrome"), ("Version/", "Safari")]:
        m = re.search(re.escape(key) + r"(\d+)", ua)
        if m:
            return f"{os_} {name} {m.group(1)}"
    return f"{os_} {ua[:40]}" if ua else "(no browser named)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", help="YYYY-MM-DD[THH:MM]: only visits from then on")
    ap.add_argument("--all", action="store_true", help="include visitors who only made one request")
    ap.add_argument("--file", type=Path, default=VISITS)
    args = ap.parse_args()

    if not args.file.exists():
        raise SystemExit(f"{args.file}: no visits logged yet (app/server.py writes it since 2026-10-24)")
    who: dict[tuple, dict] = {}
    for line in args.file.read_text(encoding="utf-8").splitlines():
        try:
            v = json.loads(line)
        except ValueError:
            continue
        if args.since and v["t"] < args.since:
            continue
        w = who.setdefault((v["ip"], v["ua"]), {"first": v["t"], "requests": 0, "opened": 0, "listened": 0,
                                                "ear_s": 0.0, "sessions": [], "links": set()})
        w["last"] = v["t"]
        w["requests"] += 1
        w["links"].add(LINKS.get(v.get("port"), str(v.get("port"))))
        path, ok = v["path"], (v["status"] or 0) < 400
        if v["method"] == "GET" and path in ("/", "/index.html") and ok:
            w["opened"] += 1
        elif path == "/asr-worker.js" and ok:
            w["listened"] += 1
        elif path == "/ws/ear" and v["status"] == 101:
            w["listened"] += 1
            w["ear_s"] += v.get("s", 0)
        m = re.fullmatch(r"/api/sessions/([\w-]+?)(?:\.wav|/0)", path)
        if v["method"] == "POST" and m and ok and m.group(1) not in w["sessions"]:
            w["sessions"].append(m.group(1))

    shown = sorted(((k, w) for k, w in who.items() if args.all or w["requests"] > 1), key=lambda kw: kw[1]["last"])
    rows = [w for _, w in shown]
    for (ip, ua), w in shown:
        when = w["first"].replace("T", " ")[:16] + ("" if w["last"][:16] == w["first"][:16]
                                                     else " -> " + w["last"].replace("T", " ")[5:16])
        ear = f" ({w['ear_s'] / 60:.1f} min on the server)" if w["ear_s"] else ""
        print(f"{when}  {ip}  {short_ua(ua)}  [{', '.join(sorted(w['links']))}]")
        print(f"    opened {w['opened']}, listened {w['listened']}{ear}, sessions {len(w['sessions'])}"
              + (": " + " ".join(w["sessions"]) if w["sessions"] else ""))
    hidden = len(who) - len(rows)
    print(f"\n{len(rows)} visitors" + (f", {hidden} one-request visitors not shown (--all)" if hidden else "")
          + f", {sum(w['listened'] > 0 for w in rows)} listened, "
          f"{sum(len(w['sessions']) for w in rows)} sessions")


if __name__ == "__main__":
    main()
