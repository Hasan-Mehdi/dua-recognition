#!/usr/bin/env python
"""Freeze the starting point of a reliability run (docs/research/claude-execution-plan-2026-09-26.md).

Writes data/cache/reliability/<run>/manifest.json: source-file and corpus hashes,
checkpoints, dependency versions, current defaults, which recordings exist in
which split, and the experiment choices + acceptance rules, written *before*
any new outcome table is looked at. A commit id alone doesn't pin this tree:
other sessions leave work in it, so every file that decides a number is hashed.

    python scripts/reliability_manifest.py rel-20260926
    python scripts/reliability_manifest.py rel-20260926 --check   # what changed since
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data" / "cache" / "reliability"

# Files whose content decides the numbers of this plan.
SOURCES = [
    "src/dua_recognition/" + f for f in (
        "align.py", "asr.py", "corpus.py", "ctc.py", "ctc_align.py", "display.py", "follower.py",
        "labels.py", "offline.py", "pipeline.py", "splits.py", "streaming.py", "text.py", "tracker.py")
] + [
    "scripts/" + f for f in (
        "evaluate.py", "word_eval.py", "follow_eval.py", "pause_eval.py", "word_truth.py", "user_label.py",
        "dump_ctc.py", "transcribe_windows.py", "session_report.py")
] + ["web/" + f for f in ("asr-worker.js", "app.js", "tracker.js", "display.js", "session-log.js")] + [
    "app/server.py"]

# Checkpoints: hash the small files that identify them, size the weights.
CHECKPOINTS = ["models/wav2vec2-quran-dua", "models/whisper-base-aug-v4-ct2", "models/whisper-turbo-dua-ct2",
               "models/whisper-base-quran-dua-ct2", "web/models/whisper-base-aug-v4", "web/vad"]

PACKAGES = ["numpy", "torch", "torchaudio", "transformers", "faster_whisper", "ctranslate2", "onnxruntime",
            "numba", "fastapi"]

# Written before any new outcome is looked at (plan, Step 0.5). Round-2's own
# preregistration (docs/results/follower.md) is not touched.
PREREGISTRATION = {
    "written": "2026-09-26, before any Step 2-4 outcome table",
    "development_data": {
        "duaplayer_train": "DuaPlayer train reciters (splits.TEST_RECITERS complement): the tuning set, "
                           "also the development set for gate/follower/small-CTC comparisons",
        "duaplayer_test": "historical regression suite only (examined repeatedly; not fresh validation)",
        "testsets": "majlis/amateur/user split into development vs final by person/venue group "
                    "(scripts/testset_split.py) before any candidate is scored on them",
    },
    "browser_gate": {
        "policies": ["legacy", "energy_assisted", "ungated"],
        "default": "legacy (unchanged unless the selection rule passes and a device measurement exists)",
        "fixed_factor": "-45 dBFS tail floor held fixed in the first comparison",
        "energy_assisted_rule": "run Whisper if the legacy Silero rule passes, OR the last 1.5 s holds a "
                                "run of >= 7 consecutive 32 ms frames at >= floor + 6 dB, where floor = "
                                "max(-70 dBFS, 10th-percentile frame energy of the 6 s window) (asr.quiet_at_end's "
                                "energy evidence, with the speech gate's 7-frame run length)",
        "ungated_rule": "skip Silero; keep the -45 dBFS floor and every existing output safeguard",
        "filter": "the browser has no hallucination filter; the Python one (asr._looks_hallucinated) is a "
                  "separate ablation, never changed together with the gate",
        "selection_rule": "among policies that reduce discarded annotated speech: line accuracy (exact) no "
                          "more than 1.0 pt below legacy, wrong-du'a/nonspeech exposure no more than 0.5 pt "
                          "above legacy, p95 evidence age no more than 0.2 s worse. Engineering tolerances, "
                          "not statistical guarantees. Paired per-recording results; if unresolved, keep "
                          "legacy and leave the candidate opt-in.",
    },
    "follower": {
        "rule": "no new decision rule unless the latency accounting shows decision/display delay remains "
                "after emission latency is accounted for (plan Step 3.4-3.5)",
        "gates_if_a_candidate_is_built": {
            "word_exact_vs_page_mode": ">= +5 pts where valid labels exist",
            "next_line_in_pause": "<= 10%",
            "jerks_and_early_entries": "no worse than page mode",
            "human_onset_lag": "median no more than 0.2 s worse than page mode, no more missed transitions",
            "also_vs": "round-2 incumbent (round-1 FollowerConfig)",
        },
    },
    "small_ctc": {
        "candidates": ["incumbent wav2vec2-quran-dua (~300M, 20 ms frames)",
                       "one text-CTC candidate: Tilawa's fastconformer_full_mixed.onnx (nvidia "
                       "stt_ar_fastconformer_hybrid_large_pcd_v1.0, CC-BY-4.0), if its contract is usable"],
        "go_rule": "no default change unless word/line quality is within the follower gates of the incumbent "
                   "AND a real device latency measurement passes; desktop numbers are provisional",
        "zipformer_branch": "only if the text-CTC comparison leaves a clear unmet need and the NPL-1.2 terms fit",
    },
}


def sha(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(limit) if limit else f.read())
    return h.hexdigest()[:16]


def tree_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def checkpoint(rel: str) -> dict:
    d = ROOT / rel
    if not d.exists():
        return {"missing": True}
    files = [p for p in d.rglob("*") if p.is_file()]
    small = [p for p in files if p.stat().st_size < 5_000_000]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files),
            "config_hash": tree_hash(small) if small else None,
            # first 64 MB of the largest file: enough to tell two checkpoints apart, cheap to read
            "weights_head_hash": sha(max(files, key=lambda p: p.stat().st_size), 64 << 20) if files else None}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def recordings() -> dict:
    from dua_recognition.corpus import TESTSETS, load_all, load_recordings
    from dua_recognition.splits import LINE_LABELS_UNRELIABLE, is_test

    duas = load_all()
    out = {"duaplayer": {"train": [], "test": []}}
    for dua in duas.values():
        for r in load_recordings(dua, extra=False):
            out["duaplayer"]["test" if is_test(r.reciter) else "train"].append(
                {"id": r.audio_id, "dua": r.dua_id, "reciter": r.reciter, "s": round(r.end_s, 1),
                 "line_labels_unreliable": r.dua_id in LINE_LABELS_UNRELIABLE})
        for name, d in TESTSETS.items():
            if d.exists():
                for r in load_recordings(dua, cache_dir=d, extra=False):
                    out.setdefault(name, []).append({"id": r.audio_id, "dua": r.dua_id, "tier": r.tier,
                                                     "condition": r.condition, "venue": r.venue,
                                                     "s": round(r.end_s, 1)})
    return out


def build(run: str) -> dict:
    duas = sorted((ROOT / "data" / "duas").glob("*.json"))
    versions = {}
    for m in PACKAGES:
        try:
            versions[m] = importlib.import_module(m).__version__
        except Exception as e:  # noqa: BLE001
            versions[m] = f"unavailable: {type(e).__name__}"
    try:
        versions["node"] = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
    except OSError:
        versions["node"] = "unavailable"
    from dua_recognition.follower import FollowerConfig
    from dua_recognition.tracker import TrackerConfig

    return {
        "run": run,
        "git": {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": git("status", "--porcelain").splitlines()},
        "platform": {"os": platform.platform(), "python": sys.version.split()[0], "packages": versions},
        "corpus": {"texts": len(duas), "hash": tree_hash(duas)},
        "sources": {s: sha(ROOT / s) for s in SOURCES if (ROOT / s).exists()},
        "checkpoints": {c: checkpoint(c) for c in CHECKPOINTS},
        "defaults": {
            "server_asr": "whisper-turbo-dua (ct2) + prompt biasing (asr.DEFAULT_MODEL)",
            "phone_asr": "web/models/whisper-base-aug-v4 (ONNX q8, web/app.js)",
            "browser_gate": "legacy: speechInTail (Silero run >= 7 frames >= 0.35, peak >= 0.5, tail 1.5 s, "
                            "-45 dBFS floor) must pass before Whisper runs (web/asr-worker.js)",
            "server_gate": "Recognizer vad=False (pipeline.py)",
            "tracker": repr(TrackerConfig()),
            "follower": repr(FollowerConfig()),
            "words_ctc": "opt-in ?words=ctc (server only), round-1 FollowerConfig",
        },
        "recordings": recordings(),
        "owned_elsewhere": {
            "data/cache/run/testsets_queue.sh": "held-out test-set queue (majlis/amateur labelling, baseline "
                                                "windows); no scoring until review. Do not touch its outputs.",
            "data/cache/run/quranlab_eval.sh": "Quran-Lab CER run (CPU)",
            "data/cache/run/caption_chain.sh": "caption harvest (YouTube lane)",
            "app/server.py (running process)": "Hasan's phone server on :8443 (GPU); not restarted",
        },
        "reused": [
            "follower rounds 1-2 (docs/results/follower.md, data/cache/follow*_log.txt, CTC dumps data/cache/ctc)",
            "word truth data/cache/word_truth (automatic, wav2vec2-quran-dua)",
            "pause benchmark data/cache/pause (inserted 4 s room tone)",
            "window caches data/cache/windows/<tag> (CTranslate2, ungated)",
            "held-out test-set tooling (corpus.TESTSETS, user_label.py, segment_streams.py)",
        ],
        "preregistration": PREREGISTRATION,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--check", action="store_true", help="compare the tree with a saved manifest")
    ap.add_argument("--note", default="", help="why this manifest exists (e.g. a rerun after data loss)")
    args = ap.parse_args()
    f = OUT / args.run / "manifest.json"
    if args.check:
        old = json.loads(f.read_text(encoding="utf-8"))
        new = build(args.run)
        for key in ("sources", "checkpoints"):
            for k in sorted(set(old[key]) | set(new[key])):
                if old[key].get(k) != new[key].get(k):
                    print(f"changed: {k}")
        if old["corpus"] != new["corpus"]:
            print("changed: corpus")
        return
    if f.exists():
        sys.exit(f"{f} exists: a manifest is frozen once (use --check, or a new run name)")
    f.parent.mkdir(parents=True, exist_ok=True)
    m = build(args.run)
    if args.note:
        m["note"] = args.note
    f.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
