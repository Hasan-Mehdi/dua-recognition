"""Debug sessions on the server (app/server.py): pieces sent while a session is recorded
(web/session-log.js SessionLog.stream) become the same .wav the page uploads whole, and every
request is logged in visits.jsonl."""
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from session_report import read_session  # noqa: E402

HEAD = {"id": "20261024-120000-abcd", "started": "2026-10-24T12:00:00.000Z", "engine": "device", "ended": "unknown"}


@pytest.fixture
def server(tmp_path, monkeypatch):
    os.environ.setdefault("DUA_ENGINE", "server")
    sys.path.insert(0, str(ROOT / "app"))
    s = importlib.import_module("server")
    monkeypatch.setattr(s, "SESSIONS", tmp_path)
    return s


@pytest.fixture
def client(server):
    from fastapi.testclient import TestClient

    return TestClient(server.app)  # no `with`: no startup, so no models and no sweeper


def piece(pcm, events, **head):
    j = "\n".join(json.dumps(e) for e in [{**HEAD, **head}, *events]).encode()
    return len(j).to_bytes(4, "little") + j + pcm.astype("<i2").tobytes()


def test_pieces_make_the_session(server, client, tmp_path):
    a, b = np.arange(16000, dtype=np.int16), np.arange(8000, dtype=np.int16) * -1
    sid = HEAD["id"]
    assert client.post(f"/api/sessions/{sid}/0", content=piece(a, [{"t": 0, "type": "start"}])).json() == {"next": 1}
    assert not (tmp_path / f"{sid}.wav").exists()
    client.post(f"/api/sessions/{sid}/1", content=piece(b, [{"t": 1.5, "type": "end", "reason": "stop"}], ended="stop"))
    audio, log = read_session(tmp_path / f"{sid}.wav")
    assert np.array_equal(np.round(audio * 32768).astype(np.int16), np.concatenate([a, b]))
    assert log["ended"] == "stop" and log["engine"] == "device"
    assert [e["type"] for e in log["events"]] == ["start", "end"]
    assert not (tmp_path / ".live" / sid).exists()


def test_a_piece_sent_twice_or_missing(server, client):
    sid, x = HEAD["id"], np.zeros(100, np.int16)
    client.post(f"/api/sessions/{sid}/0", content=piece(x, []))
    assert client.post(f"/api/sessions/{sid}/0", content=piece(x, [])).json() == {"next": 1}  # kept once
    assert client.post(f"/api/sessions/{sid}/2", content=piece(x, [])).status_code == 409
    assert client.post(f"/api/sessions/other/3", content=piece(x, [])).status_code == 409


def test_a_killed_tab_is_built_once_quiet(server, client, tmp_path):
    sid = HEAD["id"]
    client.post(f"/api/sessions/{sid}/0", content=piece(np.ones(320, np.int16), [{"t": 0, "type": "start"}]))
    server.sweep_live()
    assert not (tmp_path / f"{sid}.wav").exists()  # still coming in
    server.sweep_live(time.time() + server.LIVE_IDLE_S + 1)
    audio, log = read_session(tmp_path / f"{sid}.wav")
    assert audio.size == 320 and log["ended"] == "unknown"
    # it may still go on (a phone that slept): more pieces, built again
    client.post(f"/api/sessions/{sid}/1", content=piece(np.ones(320, np.int16), [], ended="stop"))
    assert read_session(tmp_path / f"{sid}.wav")[0].size == 640


def test_the_whole_file_replaces_the_pieces(server, client, tmp_path):
    sid = HEAD["id"]
    client.post(f"/api/sessions/{sid}/0", content=piece(np.ones(320, np.int16), []))
    whole = (tmp_path / "w.wav")
    server.sweep_live(time.time() + server.LIVE_IDLE_S + 1)
    whole.write_bytes((tmp_path / f"{sid}.wav").read_bytes())
    assert client.post(f"/api/sessions/{sid}.wav", content=whole.read_bytes()).status_code == 200
    assert not (tmp_path / ".live" / sid).exists()


def test_visits_are_logged(server, client, tmp_path):
    client.get("/api/mode", headers={"User-Agent": "Mozilla/5.0 (Linux; Android 10; K) Chrome/155.0.0.0"})
    client.get("/nothing-here")
    v = [json.loads(line) for line in (tmp_path / "visits.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(x["method"], x["path"], x["status"]) for x in v] == [("GET", "/api/mode", 200), ("GET", "/nothing-here", 404)]
    assert "Android" in v[0]["ua"] and v[0]["t"][:4].isdigit()
