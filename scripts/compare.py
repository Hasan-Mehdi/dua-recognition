#!/usr/bin/env python
"""One table for every front end x setting on the test reciters.

Runs scripts/evaluate.py for each configuration below whose caches exist and
collects the headline numbers (tracker, identifying the du'a itself).

    python scripts/compare.py > docs/results/comparison.md
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "whisper-base-quran-dua"
CONFIGS = [
    # (label, asr tag, extra evaluate args)
    ("6 s windows (current)", BASE, []),
    ("+ VAD", BASE, ["--vad"]),
    ("+ VAD + sticky lock", BASE, ["--vad", "--set", "p_teleport_locked=0.0001"]),
    ("streaming, full hypothesis", f"{BASE}+stream", []),
    ("streaming, committed only", f"{BASE}+committed", []),
    ("noisy room: 6 s windows", f"{BASE}+room15", []),
    ("noisy room: + VAD + sticky lock", f"{BASE}+room15", ["--vad", "--set", "p_teleport_locked=0.0001"]),
    ("noisy room: room-trained model", f"{BASE}-room+room15", []),
    ("noisy room: room-trained + VAD + sticky", f"{BASE}-room+room15", ["--vad", "--set", "p_teleport_locked=0.0001"]),
    ("clean: room-trained model", f"{BASE}-room", []),
    ("clean: turbo fine-tuned (server)", "whisper-turbo-dua", []),
    ("noisy room: turbo fine-tuned + VAD + sticky", "whisper-turbo-dua+room15", ["--vad", "--set", "p_teleport_locked=0.0001"]),
    ("clean: base v3 (more data + room)", "whisper-base-quran-dua-v3", []),
    ("noisy room: base v3 + VAD + sticky", "whisper-base-quran-dua-v3+room15", ["--vad", "--set", "p_teleport_locked=0.0001"]),
]


def main() -> None:
    print("| configuration | line acc | ±1 | refrain | wrong du'a | lag | jumps/min | found @3 s | @10 s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, tag, extra in CONFIGS:
        if not (ROOT / "data" / "cache" / "windows" / tag).exists():
            print(f"| {label} | (not run) |||||||| ", file=sys.stderr)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.json"
            subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--asr", tag, "--json", str(out), *extra],
                           check=True, capture_output=True)
            r = json.loads(out.read_text())
        t = r["summary"]["tracker"]
        ident = r["identification"]
        print(f"| {label} | {t['line_acc']:.1%} | {t['line_acc_pm1']:.1%} | {t['refrain_acc']:.1%} | "
              f"{t['wrong_dua_shown']:.1%} | {t['lag_median_s']:.1f} s | {t['jumps_per_min']:.2f} | "
              f"{ident['3']['tracker']:.0%} | {ident['10']['tracker']:.0%} |", flush=True)


if __name__ == "__main__":
    main()
