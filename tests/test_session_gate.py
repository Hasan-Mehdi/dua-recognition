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


def test_retrack_swaps_in_the_display_lead_being_tried(monkeypatch):
    """Sessions log each update's lead (delay + the page's display_lead). A tried
    display_lead must replace the logged one, not be ignored (it was, before 2026-09-29)."""
    from dua_recognition import tracker as T
    from dua_recognition.corpus import load_all

    duas = {k: v for k, v in load_all().items() if k in ("duasorg-namaz-e-wahshat", "dua-aahad")}
    seen = []
    real = T.Tracker.update

    def spy(self, transcript, dt, lead=0.0, quiet=0.0, **kw):
        seen.append(round(lead, 3))
        return real(self, transcript, dt, lead, quiet, **kw)

    monkeypatch.setattr(T.Tracker, "update", spy)
    log = {"tracker": {"displayLead": 0.5}, "events": [
        {"t": 2.8, "type": "hop", "end": 2.0, "dt": 1.0, "lead": 1.3, "text": "الله لا إله إلا هو"},
        {"t": 3.8, "type": "hop", "end": 3.0, "dt": 1.0, "lead": 1.3, "text": "الحي القيوم"}]}
    sr.retrack(log, duas, {})  # as logged: the page's own leads
    sr.retrack(log, duas, {"display_lead": 0.0})
    sr.retrack(log, duas, {"display_lead": 0.25}, as_logged=False)
    assert seen == [1.3, 1.3, 0.8, 0.8, 1.05, 1.05]


def test_logged_config_reads_the_pages_settings():
    log = {"tracker": {"kappaSearch": 1.2, "displayLead": 0.5, "leadCrossQuiet": None, "backConfirm": 1,
                       "nullRateLocked": None, "speeds": [0.5, 1.0], "notAField": 3}}
    cfg = sr.logged_config(log)
    assert cfg == {"kappa_search": 1.2, "display_lead": 0.5, "lead_cross_quiet": float("inf"), "back_confirm": 1,
                   "null_rate_locked": None, "speeds": (0.5, 1.0),
                   # not in this log: the page didn't have them yet, so they were off
                   "lead_cross_words": float("inf"), "seek_pins_line": False, "keep_dua_confidence": None,
                   "retreat_in_line": False, "still_catch_up": False, "pause_motion": False}
