"""Session logs with and without the gate/latency fields (web/app.js, 2026-09-26)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import session_report as sr  # noqa: E402


def test_old_logs_have_no_gate_section():
    old = {"events": [{"t": 1.0, "ms": 1, "type": "hop", "end": 1.0, "asr_ms": 300, "text": "", "quiet": 0}]}
    assert sr.gate_summary(old) == []


def test_new_logs_separate_skips_from_empty_text_and_report_age():
    ev = []
    for k in range(20):
        skip = [None, "vad_reject", "below_floor", "empty"][k % 4]
        ev.append({"t": k + 1.0, "ms": k, "type": "hop", "end": float(k + 1), "gate": "legacy", "skip": skip,
                   "ran": skip in (None, "empty"), "level": -30.0 if skip != "below_floor" else -52.0,
                   "infer_ms": 400.0 if skip in (None, "empty") else None, "cold": k == 0})
        ev.append({"t": k + 1.4, "ms": k, "type": "visible", "end": float(k + 1), "age_ms": 500.0 + 10 * k})
    out = "\n".join(sr.gate_summary({"events": ev}))
    assert "vad_reject 5" in out and "below_floor 5" in out and "empty 5" in out and "ran, text 5" in out
    assert "25% of hops under the -45 dBFS floor" in out
    assert "p95" in out and "evidence age" in out
