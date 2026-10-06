"""Re-encode harvested audio to 16 kHz mono Opus at 32 kbps, the format GigaSpeech stores its
audio in, to save disk space. On a 14-recording test the labels came out the same (du'a,
usable seconds 99.9%, line starts within 0.04 s).

Left as they are: the bench's recordings (data/testbed/sources.jsonl), recordings whose voice
matched a test set (voices.jsonl verdict test/error), the held-out voice pool
(sha1(reciter) % 100 < 3), and files already at 48 kbps or less.

.webm and .m4a keep their names (both containers hold Opus). Other formats become <id>.webm;
the platform's index.jsonl and every jsonl/json under data/cache that names the old path are
rewritten at the end (--finalize), and only then are the old files deleted.
Each file: encode to harvest/reencode_tmp, decode the result in full, compare its length with
the label's (the original as the pipeline decoded it), then replace. One line per file in
harvest/logs/reencode.jsonl; a rerun picks up where it stopped.

    python scripts/harvest_reencode.py --limit 20        # a few, then look at the log
    python scripts/harvest_reencode.py -j 8              # everything, then --finalize
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import harvest_label as hl  # noqa: E402

HARVEST = Path(os.path.realpath(hl.HARVEST))
TMP = HARVEST / "reencode_tmp"
LOG = HARVEST / "logs" / "reencode.jsonl"
SAME_NAME = {".webm": "webm", ".m4a": "mp4"}
AUDIO = {".webm", ".m4a", ".mp3", ".wma", ".ogg", ".opus", ".wav", ".flac", ".aac", ".amr", ".mp4"}
MAX_KBPS = 48


def read_log() -> dict:
    done = {}
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["platform"], r["id"])] = r
    return done


def append_log(row: dict) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def skip_sets() -> tuple[set, set, set]:
    bench = set()
    for line in (ROOT / "data" / "testbed" / "sources.jsonl").read_text(encoding="utf-8").splitlines():
        sid = json.loads(line)["sid"]
        if sid.startswith("harvest:"):
            _, p, rid = sid.split(":")[:3]
            bench.add((p, rid))
    voices = {(r["platform"], r["id"]) for r in hl.read_jsonl(HARVEST / "voices.jsonl")
              if r.get("verdict") in ("test", "error")}
    held = set()
    for p, meta, _ in hl.recordings():
        if hl._is_val(hl._reciter(p, meta), 3):
            held.add((p, meta["id"]))
    return bench, voices, held


def ref_duration(p: str, rid: str, meta: dict) -> tuple[float | None, str]:
    lab = HARVEST / "labels" / p / f"{rid}.json"
    if lab.exists():
        try:
            return float(json.loads(lab.read_text(encoding="utf-8"))["duration"]), "label"
        except Exception:  # noqa: BLE001
            pass
    if meta.get("duration"):
        return float(meta["duration"]), "index"
    return None, "none"


def decoded_seconds(path: Path) -> float:
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "null", "-",
                          "-progress", "pipe:1", "-nostats"], capture_output=True, text=True, check=True).stdout
    us = [int(m) for m in re.findall(r"out_time_us=(\d+)", out)]
    return (us[-1] / 1e6) if us else 0.0


def work(job: dict) -> dict:
    src = Path(job["src"])
    ext = src.suffix.lower()
    fmt = SAME_NAME.get(ext, "webm")
    dst = src if ext in SAME_NAME else src.with_suffix(".webm")
    tmp = TMP / f"{job['platform']}_{job['id']}.{'m4a' if fmt == 'mp4' else 'webm'}"
    row = {k: job[k] for k in ("platform", "id", "ref_s", "ref_from")}
    row.update(src=src.name, dst=dst.name, src_bytes=src.stat().st_size)
    t0 = time.time()
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map", "0:a:0", "-vn",
                        "-map_metadata", "-1", "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "32k",
                        "-f", fmt, str(tmp)], capture_output=True, text=True, check=True)
        got = decoded_seconds(tmp)
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        return {**row, "status": "fail", "err": (e.stderr or "")[-300:]}
    row.update(new_s=round(got, 3), dst_bytes=tmp.stat().st_size, sec=round(time.time() - t0, 1))
    ref = job["ref_s"]
    if ref is not None:
        tol = (0.3 + 0.001 * ref) if job["ref_from"] == "label" else (2.0 + 0.01 * ref)
        if abs(got - ref) > tol:
            tmp.unlink(missing_ok=True)
            return {**row, "status": "mismatch"}
    elif got < 5:
        tmp.unlink(missing_ok=True)
        return {**row, "status": "mismatch"}
    if dst.exists() and dst != src:
        tmp.unlink(missing_ok=True)
        return {**row, "status": "fail", "err": f"{dst.name} already exists"}
    os.replace(tmp, dst)
    return {**row, "status": "ok" if dst == src else "ok-renamed"}


def cmd_run(args) -> None:
    TMP.mkdir(exist_ok=True)
    done = read_log()
    bench, voices, held = skip_sets()
    jobs, skipped = [], {"bench": 0, "test voice": 0, "held-out voice": 0, "low bitrate": 0, "done": 0,
                         "not audio": 0}
    for p, meta, audio in hl.recordings():
        key = (p, meta["id"])
        if key in done and done[key]["status"].startswith("ok"):
            skipped["done"] += 1
            continue
        if audio.suffix.lower() not in AUDIO:
            skipped["not audio"] += 1
            continue
        if key in bench:
            skipped["bench"] += 1
            continue
        if key in voices:
            skipped["test voice"] += 1
            continue
        if key in held:
            skipped["held-out voice"] += 1
            continue
        ref, src_of = ref_duration(p, meta["id"], meta)
        size = audio.stat().st_size
        if ref and size * 8 / ref / 1000 <= MAX_KBPS:
            skipped["low bitrate"] += 1
            continue
        jobs.append({"platform": p, "id": meta["id"], "src": str(Path(os.path.realpath(audio))),
                     "ref_s": ref, "ref_from": src_of, "bytes": size})
    if args.limit:
        jobs = jobs[: args.limit]
    gb = sum(j["bytes"] for j in jobs) / 1e9
    print(f"{len(jobs)} files to re-encode ({gb:.0f} GB); skipped {skipped}", flush=True)
    t0, n, saved, bad = time.time(), 0, 0, 0
    with ProcessPoolExecutor(args.jobs) as pool:
        futs = [pool.submit(work, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            append_log(r)
            n += 1
            if r["status"].startswith("ok"):
                saved += r["src_bytes"] - r["dst_bytes"]
            else:
                bad += 1
                print(f"{r['platform']}/{r['id']}: {r['status']} {r.get('err', '')} "
                      f"ref {r.get('ref_s')} new {r.get('new_s')}", flush=True)
            if n % 200 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"== {n}/{len(jobs)}, saved {saved / 1e9:.1f} GB, {bad} not replaced, "
                      f"{el / 60:.0f} min, ~{el / n * (len(jobs) - n) / 60:.0f} min left", flush=True)


def cmd_finalize(args) -> None:
    """Point index.jsonl and data/cache manifests at the renamed files, then delete the old ones."""
    log = read_log()
    ren = {k: r for k, r in log.items() if r["status"] == "ok-renamed"}
    print(f"{len(ren)} renamed files", flush=True)
    by_platform: dict[str, dict] = {}
    for (p, rid), r in ren.items():
        by_platform.setdefault(p, {})[r["src"]] = r["dst"]
    # 1. index.jsonl: the "audio" field is the bare file name.
    for p, names in by_platform.items():
        idx = HARVEST / p / "index.jsonl"
        lines, changed = [], 0
        for line in idx.read_text(encoding="utf-8").splitlines():
            try:
                m = json.loads(line) if line.strip() else None
            except json.JSONDecodeError:  # e.g. a run of NULs from the 2026-10-02 crash: keep it as it is
                m = None
            if m is None:
                lines.append(line)
                continue
            if m.get("audio") in names:
                m["audio"] = names[m["audio"]]
                changed += 1
                line = json.dumps(m, ensure_ascii=False)
            lines.append(line)
        tmp = idx.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, idx)
        print(f"{p}/index.jsonl: {changed} lines", flush=True)
    # 2. Manifests that store absolute paths (JSON-escaped backslashes).
    pat = re.compile(r'harvest(\\\\|/)(' + "|".join(map(re.escape, by_platform)) + r')(\\\\|/)audio(\\\\|/)([^"\\/]+)"')

    def sub(m):
        new = by_platform[m.group(2)].get(m.group(5))
        return m.group(0) if new is None else m.group(0)[: -len(m.group(5)) - 1] + new + '"'

    roots = [ROOT / "data" / "cache", Path(r"D:\dua-data\clips")]
    for root in roots:
        for f in list(root.rglob("*.jsonl")) + list(root.rglob("*.json")):
            if f.is_symlink() or f.stat().st_size > 4e9:
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            new, k = pat.subn(sub, text)
            if new != text:
                tmp = f.with_name(f.name + ".tmp")
                tmp.write_text(new, encoding="utf-8")
                os.replace(tmp, f)
                print(f"{f}: {k} paths", flush=True)
    # 3. The old files, now that nothing points at them.
    gone = 0
    for (p, rid), r in ren.items():
        old = HARVEST / p / "audio" / r["src"]
        if old.exists() and (HARVEST / p / "audio" / r["dst"]).exists():
            old.unlink()
            gone += 1
    print(f"deleted {gone} originals", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    cmd_finalize(args) if args.finalize else cmd_run(args)


if __name__ == "__main__":
    main()
