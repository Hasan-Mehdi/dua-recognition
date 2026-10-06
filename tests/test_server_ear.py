"""The server engine's ear (app/server.py /ws/ear): the page streams audio and names each window
by the sample it ends at; the server cuts it from what it was sent and answers as the page's
workers do (asr-worker.js, ctc-worker.js)."""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
MODELS = [ROOT / "models" / "whisper-base-syn-v5-ctx8ft-ct2", ROOT / "models" / "ctc-student-base-v6" / "student.pt"]


@pytest.fixture(scope="module")
def server():
    os.environ.setdefault("DUA_ENGINE", "server")
    sys.path.insert(0, str(ROOT / "app"))
    return importlib.import_module("server")


def test_window_is_cut_by_sample_number(server):
    a = server.EarAudio()
    x = np.arange(1, 5001, dtype=np.float32)
    a.push(x[:3000])
    a.push(x[3000:])
    assert np.array_equal(a.window(5000, 1000), x[4000:5000])
    assert np.array_equal(a.window(3000, 1000), x[2000:3000])  # audio sent after the request is not in it
    w = a.window(500, 1000)  # before the stream began: zeros, as the page's buffer starts
    assert np.array_equal(w[:500], np.zeros(500)) and np.array_equal(w[500:], x[:500])


def test_window_after_a_reconnect(server):
    a = server.EarAudio()
    a.push(np.ones(2000, np.float32))
    a.at(10_000)  # audio from 2000 to 10000 was lost: the server hears silence there
    a.push(np.full(1000, 2.0, np.float32))
    w = a.window(11_000, 3000)
    assert np.array_equal(w[:2000], np.zeros(2000)) and np.array_equal(w[2000:], np.full(1000, 2.0))


def test_window_keeps_the_last_30_s(server):
    a = server.EarAudio()
    a.push(np.ones(server.EarAudio.KEEP + 16000, np.float32))
    assert a.buf.size == server.EarAudio.KEEP
    assert a.window(a.total, 32000).sum() == 32000


@pytest.mark.skipif(not all(p.exists() for p in MODELS), reason="the page's Whisper (CTranslate2) and CTC student needed")
def test_transcribe_and_frames(server):
    from fastapi.testclient import TestClient

    rng = np.random.default_rng(0)
    y = (0.1 * np.sin(2 * np.pi * 220 * np.arange(3 * SR) / SR) + 0.01 * rng.normal(size=3 * SR)).astype(np.float32)
    with TestClient(server.app) as client, client.websocket_connect("/ws/ear") as ws:
        ws.send_text(json.dumps({"ch": "ctc", "type": "load"}))
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["meta"]["window_s"] == 2.0 and ready["gen"] is None
        ws.send_text(json.dumps({"ch": "ear", "type": "at", "total": 0}))
        for i in range(0, y.size, 800):
            ws.send_bytes((y[i : i + 800] * 32768).astype("<i2").tobytes())
        ws.send_text(json.dumps({"ch": "ctc", "type": "frames", "id": y.size, "gen": 3}))
        head = ws.receive_json()
        assert (head["type"], head["id"], head["gen"], head["T"], head["C"]) == ("frames", y.size, 3, 100, 43)
        lp = np.frombuffer(ws.receive_bytes(), "<f2").reshape(100, 43).astype(np.float32)
        assert np.isfinite(lp).all() and np.allclose(np.logaddexp.reduce(lp, axis=1), 0, atol=0.05)
        ws.send_text(json.dumps({"ch": "asr", "type": "transcribe", "id": y.size, "gen": 3, "voiceDb": None, "dt": 1.0}))
        text = ws.receive_json()
        assert text["type"] == "text" and text["id"] == y.size and text["gen"] == 3
        assert text["gate"]["policy"] == "legacy" and text["quiet"] is not None and "paused" in text
