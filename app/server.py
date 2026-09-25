#!/usr/bin/env python
"""Server mode for the web front end (web/): recite into the browser mic, or
replay a recording, and the du'a follows the line and word being recited.

    python app/server.py                                  # http://localhost:8000
    DUA_ASR_MODEL=models/whisper-base-quran-dua-ct2 DUA_ASR_DEVICE=cpu python app/server.py

The browser streams 16 kHz mono float32 over a WebSocket; the server answers
each hop with the tracker's position. ASR runs off the event loop, and audio
that arrives meanwhile is folded into the next step, so a slow machine lags
gracefully instead of queueing.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from dua_recognition.align import CorpusIndex  # noqa: E402
from dua_recognition.asr import DEFAULT_MODEL, load_model  # noqa: E402
from dua_recognition.corpus import load_all, load_recordings  # noqa: E402
from dua_recognition.pipeline import StreamingRecognizer  # noqa: E402

WEB = ROOT / "web"  # the one front end; it detects this server via /api/mode
MODEL = os.environ.get("DUA_ASR_MODEL", DEFAULT_MODEL)

DUAS = load_all()
INDEX = CorpusIndex(DUAS)
RECORDINGS = {r.audio_id: r for d in DUAS.values() for r in load_recordings(d)}

app = FastAPI(title="dua-recognition")


@app.on_event("startup")
def _warm() -> None:
    load_model(MODEL)  # first request shouldn't pay for the model load


@app.get("/api/mode")
def mode():
    return {"mode": "server", "model": MODEL}


@app.get("/corpus.json")
@app.get("/api/duas")
def duas():
    return [
        {
            "id": d.id,
            "name_en": d.name_en,
            "name_ar": d.name_ar,
            "segments": [
                {"id": s.id, "ar": s.arabic, "tl": s.transliteration, "en": s.translation}
                for s in d.segments
            ],
        }
        for d in DUAS.values()
    ]


@app.get("/api/recordings")
def recordings():
    return [
        {"id": r.audio_id, "dua": r.dua_id, "dua_name": DUAS[r.dua_id].name_en, "reciter": r.reciter}
        for r in RECORDINGS.values()
    ]


@app.get("/audio/{audio_id}")
def audio(audio_id: str):
    rec = RECORDINGS.get(audio_id)
    if rec is None:
        raise HTTPException(404)
    return FileResponse(rec.path, media_type="audio/mpeg")


def _message(update, step_ms: float, index: CorpusIndex) -> dict:
    p = update.position
    word = index.words[p.word] if p.word is not None else None
    return {
        "t": round(update.t, 2),
        "heard": update.transcript,
        "dua": p.dua,
        "dua_confidence": round(p.dua_confidence, 3),
        "segment": p.segment,
        "segment_confidence": round(p.segment_confidence, 3),
        "token": word.token if word and p.dua else None,
        # Reached the end of the line and the latest window was silent: the
        # next line is probably coming, so the UI previews it.
        "pause_at_line_end": bool(p.at_line_end and not update.transcript),
        "candidates": [
            {"id": d, "name": DUAS[d].name_en, "p": round(prob, 3)} for d, prob in p.candidates
        ],
        "step_ms": round(step_ms),
    }


@app.websocket("/ws")
async def follow(ws: WebSocket):
    await ws.accept()
    rec = StreamingRecognizer(DUAS, model=MODEL, index=INDEX)
    running: asyncio.Task | None = None

    async def run_step():
        t0 = time.perf_counter()
        update = await asyncio.to_thread(rec.step)
        if update is not None:
            await ws.send_json(_message(update, (time.perf_counter() - t0) * 1000, rec.index))

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text") or ""
            if text.startswith("lock:") and text[5:] in DUAS:
                if running:
                    await running
                rec.lock(text[5:])
                continue
            if msg.get("text") == "reset":
                if running:
                    await running
                rec.reset()
                continue
            data = msg.get("bytes")
            if not data:
                continue
            rec.push(np.frombuffer(data, dtype=np.float32))
            if rec.due and (running is None or running.done()):
                running = asyncio.create_task(run_step())
    except WebSocketDisconnect:
        pass
    finally:
        if running:
            running.cancel()


# -- majlis mode ----------------------------------------------------------------
# One phone listens; everyone else in the room follows on their own screen (or a
# projector) by opening ?watch=<code>. The host relays the position updates it
# already renders, so this works whichever engine the host runs (server or
# on-device) and viewers cost nothing but a WebSocket.

class Room:
    def __init__(self) -> None:
        self.host: WebSocket | None = None
        self.viewers: set[WebSocket] = set()
        self.last: str | None = None

    async def tell_host(self) -> None:
        if self.host is not None:
            try:
                await self.host.send_json({"viewers": len(self.viewers)})
            except Exception:
                pass


ROOMS: dict[str, Room] = {}


@app.websocket("/ws/room/{code}")
async def room(ws: WebSocket, code: str, role: str = "view"):
    code = code.upper()[:8]
    await ws.accept()
    rm = ROOMS.setdefault(code, Room())
    try:
        if role == "host":
            rm.host = ws  # the latest host wins (e.g. after a reconnect)
            await rm.tell_host()
            while True:
                rm.last = await ws.receive_text()
                dead = []
                for v in list(rm.viewers):
                    try:
                        await v.send_text(rm.last)
                    except Exception:
                        dead.append(v)
                rm.viewers.difference_update(dead)
        else:
            rm.viewers.add(ws)
            await rm.tell_host()
            if rm.last:
                await ws.send_text(rm.last)
            while True:
                await ws.receive_text()  # viewers only listen; this waits for the disconnect
    except WebSocketDisconnect:
        pass
    finally:
        if rm.host is ws:
            rm.host = None
        rm.viewers.discard(ws)
        await rm.tell_host()
        if rm.host is None and not rm.viewers:
            ROOMS.pop(code, None)


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


if __name__ == "__main__":
    # Browsers only allow the microphone on https (or localhost). To use a
    # phone on the same network: HOST=0.0.0.0 SSL_CERT=cert.pem SSL_KEY=key.pem
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 8000)),
        ssl_certfile=os.environ.get("SSL_CERT"),
        ssl_keyfile=os.environ.get("SSL_KEY"),
    )
