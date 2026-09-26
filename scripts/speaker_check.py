#!/usr/bin/env python
"""Who is reciting? Match recordings to known reciters by voice.

The train/test split is by reciter, and it's only as good as the reciter
names. Recordings from duas.org or YouTube often have no name, or one spelled
differently, so a test reciter could slip into training. This embeds each
recording's voice (speechbrain ECAPA, averaged over several 8 s chunks) and
compares it with every named reciter in the DuaPlayer / duas.pro sets.

    python scripts/speaker_check.py data/duasorg_timed/*/*.mp3
    python scripts/speaker_check.py --youtube          # every harvested YouTube recording
    python scripts/speaker_check.py --testset majlis   # leak check of a held-out set (below)

Prints, per recording, the closest known reciters and their cosine scores.
Same-reciter pairs score far above different-reciter ones (--calibrate shows
both distributions on the named set), so a high score against a test reciter
means: keep this recording out of training.

--testset NAME checks a held-out set (corpus.TESTSETS) the other way round:
against a bank of every voice used in training (named reciters, YouTube,
duas.org untimed train, Al-Fatemah train spans, captioned audio, the RetaSy
crowd clips; data/cache/voice_bank.npz). It flags
  - a test recording at >= 0.7 against a training voice (a leak to decide on),
  - re-uploads: the same du'a, length within 3% and voice >= 0.9, test vs
    train and within the test set (the rule that catches qxuDk75kkBI).
Decisions go in data/testsets/NAME/leak_decisions.json ({audio_id: "keep: why"
or "drop: why"}); leak_report.md next to it lists what's still open.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.corpus import TESTSETS  # noqa: E402
from dua_recognition.splits import is_test  # noqa: E402

CACHE = ROOT / "data" / "cache" / "speaker"
BANK = ROOT / "data" / "cache" / "voice_bank.npz"
LEAK, DUPE = 0.7, 0.9
SR = 16000


def decode(path: Path, start: float, dur: float) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(dur), "-i", str(path),
                        "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


class Embedder:
    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir=str(ROOT / "models" / "spkrec-ecapa"),
                                                    run_opts={"device": "cpu"},
                                                    local_strategy=LocalStrategy.COPY)  # no symlinks on Windows

    def __call__(self, path: Path, chunks: int = 8, chunk_s: float = 8.0) -> np.ndarray | None:
        key = CACHE / (hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16] + ".npy")
        if key.exists():
            return np.load(key)
        total = duration(path)
        if total < 20:
            return None
        # Skip the first and last 10% (intros, crowd, closing du'as by someone else).
        starts = np.linspace(0.1 * total, 0.9 * total - chunk_s, chunks)
        embs = []
        for s in starts:
            wav = decode(path, float(s), chunk_s)
            if len(wav) < SR * 2 or np.abs(wav).max() < 1e-3:
                continue
            with torch.no_grad():
                e = self.model.encode_batch(torch.from_numpy(wav.copy()).unsqueeze(0)).squeeze().numpy()
            embs.append(e / np.linalg.norm(e))
        if not embs:
            return None
        v = np.mean(embs, axis=0)
        v /= np.linalg.norm(v)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(key, v)
        return v


def named_recordings() -> list[tuple[str, Path]]:
    out = []
    for meta in sorted(glob.glob(str(ROOT / "data" / "duaplayer" / "*" / "*-*.json"))) + \
            sorted(glob.glob(str(ROOT / "data" / "duaspro" / "*" / "*.json"))):
        if meta.endswith(".repaired.json"):
            continue
        m = json.loads(Path(meta).read_text(encoding="utf-8"))
        mp3 = Path(meta).with_suffix(".mp3")
        if mp3.exists() and m.get("reciter") and m["reciter"] != "Unknown":
            out.append((m["reciter"], mp3))
    return out


def _meta(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _audio_of(meta_path: Path, meta: dict) -> Path | None:
    if meta.get("audio") and (meta_path.parent / meta["audio"]).exists():
        return meta_path.parent / meta["audio"]
    found = [p for p in meta_path.parent.glob(meta_path.stem + ".*")
             if p.suffix in (".mp3", ".webm", ".m4a", ".opus", ".wav")]
    return found[0] if found else None


def training_audio() -> list[dict]:
    """Every recording whose voice could reach training: {source, name, path, dua, duration, test}.
    Named reciters on the test side are kept too: a test set repeating a DuaPlayer
    test voice isn't a leak, but it isn't a new voice either."""
    out = []
    metas = sorted(glob.glob(str(ROOT / "data" / "duaplayer" / "*" / "*-*.json")))
    metas += sorted(glob.glob(str(ROOT / "data" / "duaspro" / "*" / "*.json")))
    metas += sorted(glob.glob(str(ROOT / "data" / "duasorg_timed" / "*" / "*.json")))
    for meta in metas:
        if meta.endswith(".repaired.json"):
            continue
        m, mp3 = _meta(meta), Path(meta).with_suffix(".mp3")
        if mp3.exists() and m.get("reciter") and m["reciter"] != "Unknown":
            out.append({"source": "named", "name": canonical(m["reciter"]), "path": mp3, "dua": Path(meta).parent.name,
                        "duration": m.get("duration_ms", 0) / 1000, "test": is_test(m["reciter"])})
    for src, pattern, name_key in (("youtube", "youtube/*/*.json", None),
                                   ("untimed", "untimed/duasorg/*/*.json", "reciter"),
                                   ("fatemah", "fatemah/spans/*/*.json", None)):
        for meta in sorted((ROOT / "data").glob(pattern)):
            if meta.name.endswith(".labels.json"):
                continue
            m = _meta(meta)
            if m.get("split", "train") != "train":
                continue  # held out already
            audio = _audio_of(meta, m)
            if audio:
                out.append({"source": src, "name": f"{src}:{m.get(name_key) if name_key else meta.stem}",
                            "path": audio, "dua": m.get("dua_id") or meta.parent.name,
                            "duration": m.get("duration_s") or 0, "test": False})
    for meta in sorted((ROOT / "data" / "captioned" / "audio").glob("*.json")):
        m = _meta(meta)
        audio = _audio_of(meta, m)
        if audio:
            out.append({"source": "captioned", "name": f"captioned:{m.get('channel') or meta.stem}", "path": audio,
                        "dua": m.get("query_dua") or "", "duration": m.get("duration") or 0, "test": False})
    return out


def crowd_voices(emb: Embedder) -> list[tuple[str, np.ndarray]]:
    """One mean embedding per RetaSy volunteer in the crowd fine-tune set (clips are a few seconds each)."""
    f = ROOT / "data" / "cache" / "finetune" / "crowd.jsonl"
    if not f.exists():
        return []
    by: dict[str, list[np.ndarray]] = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        clip = ROOT / Path(r["clip"].replace("\\", "/"))
        if not clip.exists() or len(by.get(r["reciter"], [])) >= 8:
            continue
        wav = np.load(clip).astype(np.float32) / 32767
        if len(wav) < SR * 2:
            continue
        with torch.no_grad():
            e = emb.model.encode_batch(torch.from_numpy(wav).unsqueeze(0)).squeeze().numpy()
        by.setdefault(r["reciter"], []).append(e / np.linalg.norm(e))
    out = []
    for name, es in by.items():
        v = np.mean(es, axis=0)
        out.append((name, v / np.linalg.norm(v)))
    return out


def voice_bank(emb: Embedder, rebuild: bool = False) -> tuple[list[dict], np.ndarray]:
    """(rows, embeddings) of training_audio() plus the crowd voices, cached in BANK
    (rebuilt when the set of training files changes)."""
    items = training_audio()
    key = sorted(str(it["path"]) for it in items)
    if BANK.exists() and not rebuild:
        z = np.load(BANK, allow_pickle=False)
        rows = json.loads(str(z["rows"]))
        if sorted(r["key"] for r in rows if r["source"] != "crowd") == key:
            return rows, z["emb"]
    rows, embs = [], []
    for i, it in enumerate(items, 1):
        e = emb(it["path"])
        if e is not None:
            rel = it["path"].relative_to(ROOT) if it["path"].is_relative_to(ROOT) else it["path"]
            rows.append({**it, "path": str(rel), "key": str(it["path"])})
            embs.append(e)
        if i % 100 == 0:
            print(f"  voice bank: {i}/{len(items)}", flush=True)
    for name, e in crowd_voices(emb):
        rows.append({"source": "crowd", "name": name, "path": "", "key": "", "dua": "", "duration": 0, "test": False})
        embs.append(e)
    # Files with no usable voice (too short, silent) still count toward the cache key.
    have = {r["key"] for r in rows}
    rows += [{"source": "none", "name": "", "path": "", "key": k, "dua": "", "duration": 0, "test": False}
             for k in key if k not in have]
    embs += [np.zeros(len(embs[0]) if embs else 192, dtype=np.float32)] * (len(rows) - len(embs))
    BANK.parent.mkdir(parents=True, exist_ok=True)
    np.savez(BANK, emb=np.array(embs, dtype=np.float32), rows=json.dumps(rows, ensure_ascii=False))
    return rows, np.array(embs, dtype=np.float32)


def _clusters(embs: list[np.ndarray], thr: float) -> int:
    """Connected components of the >= thr similarity graph: about how many voices."""
    parent = list(range(len(embs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            if float(embs[i] @ embs[j]) >= thr:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(len(embs))})


def _same_upload(dua_a: str, dur_a: float, dua_b: str, dur_b: float) -> bool:
    return bool(dua_a) and dua_a == dua_b and dur_a > 0 and abs(dur_a - dur_b) <= 0.03 * dur_a


def check_testset(name: str, emb: Embedder, rebuild: bool = False) -> list[dict]:
    """Leak and re-upload flags for one held-out set; writes leak_report.md/.json into its dir."""
    root = TESTSETS[name]
    rows, bank = voice_bank(emb, rebuild)
    live = [r["source"] for r in rows if r["source"] != "none"]
    print(f"voice bank: {len(live)} training voices ("
          + ", ".join(f"{k} {v}" for k, v in sorted(Counter(live).items())) + ")", flush=True)
    tests = []
    for meta in sorted(root.glob("*/*-*.json")):
        m, mp3 = _meta(meta), meta.with_suffix(".mp3")
        e = emb(mp3) if mp3.exists() else None
        if e is not None:
            tests.append({"id": m["audio_id"], "dua": m["dua_id"], "duration": m["duration_ms"] / 1000, "emb": e})
    decisions = _meta(root / "leak_decisions.json") if (root / "leak_decisions.json").exists() else {}
    flags = []
    for t in tests:
        sims = bank @ t["emb"]
        for j in np.flatnonzero(sims >= LEAK):
            r, s = rows[j], float(sims[j])
            if s >= DUPE and _same_upload(t["dua"], t["duration"], r["dua"], r["duration"]):
                flags.append({"id": t["id"], "kind": "re-upload of a training recording", "other": r["path"], "score": s})
            else:
                kind = "same voice as a DuaPlayer test reciter" if r["test"] else "voice heard in training"
                flags.append({"id": t["id"], "kind": kind, "other": r["name"], "score": s})
    for i, a in enumerate(tests):
        for b in tests[i + 1:]:
            s = float(a["emb"] @ b["emb"])
            if s >= DUPE and _same_upload(a["dua"], a["duration"], b["dua"], b["duration"]):
                flags.append({"id": b["id"], "kind": "duplicate inside the test set", "other": a["id"], "score": s})
    seen, uniq = set(), []  # one flag per (recording, kind, other), at its best score
    for f in sorted(flags, key=lambda f: -f["score"]):
        k = (f["id"], f["kind"], f["other"])
        if k not in seen:
            seen.add(k)
            uniq.append({**f, "score": round(f["score"], 3), "decision": decisions.get(f["id"], "")})
    open_ = [f for f in uniq if not f["decision"] and f["kind"] != "same voice as a DuaPlayer test reciter"]
    voices = _clusters([t["emb"] for t in tests], LEAK)
    lines = [f"# Leak check: {name}", "",
             f"{len(tests)} recordings against {len(live)} training voices; about {voices} distinct voices in the set "
             f"(ECAPA >= {LEAK} counts as one voice).", "",
             f"Open flags: {len(open_)} (decide each in leak_decisions.json: \"keep: why\" or \"drop: why\").", "",
             "| recording | flag | against | score | decision |", "|---|---|---|---:|---|"]
    lines += [f"| {f['id']} | {f['kind']} | {f['other']} | {f['score']:.2f} | {f['decision']} |" for f in uniq]
    (root / "leak_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "leak_report.json").write_text(json.dumps({"recordings": len(tests), "voices": voices, "open": len(open_),
                                                       "flags": uniq}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return uniq


def canonical(name: str) -> str:
    # Two spellings of one reciter on DuaPlayer.
    return {"Murtaza Quraish": "Murtada al-Qureish"}.get(name, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--youtube", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="score distributions among the named recordings")
    ap.add_argument("--json", type=Path, help="also write results here")
    ap.add_argument("--testset", choices=list(TESTSETS), help="leak check of a held-out set against the training voices")
    ap.add_argument("--rebuild", action="store_true", help="--testset: rebuild the voice bank")
    args = ap.parse_args()
    emb = Embedder()
    if args.testset:
        check_testset(args.testset, emb, args.rebuild)
        return

    ref = [(canonical(n), p, emb(p)) for n, p in named_recordings()]
    ref = [(n, p, e) for n, p, e in ref if e is not None]
    names = sorted({n for n, _, _ in ref})
    print(f"{len(ref)} named recordings, {len(names)} reciters", flush=True)

    if args.calibrate:
        same, diff = [], []
        for i in range(len(ref)):
            for j in range(i + 1, len(ref)):
                s = float(ref[i][2] @ ref[j][2])
                (same if ref[i][0] == ref[j][0] else diff).append(s)
        for label, xs in (("same reciter", same), ("different reciters", diff)):
            xs = np.array(xs)
            print(f"  {label:18s} n={len(xs):4d}  min {xs.min():.2f}  p5 {np.percentile(xs, 5):.2f}  "
                  f"median {np.median(xs):.2f}  p95 {np.percentile(xs, 95):.2f}  max {xs.max():.2f}")

    paths = [Path(p) for p in args.paths]
    if args.youtube:
        paths += [p for p in sorted((ROOT / "data" / "youtube").glob("*/*"))
                  if p.suffix in (".webm", ".m4a", ".mp3")]
    results = []
    for p in paths:
        e = emb(p)
        if e is None:
            print(f"  {p.name}: too short / silent")
            continue
        best: dict[str, float] = {}
        for n, _, r in ref:
            best[n] = max(best.get(n, -1.0), float(e @ r))
        top = sorted(best.items(), key=lambda kv: -kv[1])[:3]
        flag = " TEST" if is_test(top[0][0]) else ""
        print(f"  {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p}: "
              + ", ".join(f"{n} {s:.2f}" for n, s in top) + flag, flush=True)
        results.append({"path": str(p), "top": top})
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
