#!/usr/bin/env python
"""Server mode for the web front end (web/): recite into the browser mic, or
replay a recording, and the du'a follows the line and word being recited.

    python app/server.py                                  # http://localhost:8000
    DUA_ASR_MODEL=models/whisper-base-quran-dua-ct2 DUA_ASR_DEVICE=cpu python app/server.py
    DUA_ENGINE=device python app/server.py                # the phone runs the model; this serves
                                                          # the page, majlis rooms and debug sessions

The browser streams 16 kHz mono float32 over a WebSocket; the server answers
each hop with the tracker's position. ASR runs off the event loop, and audio
that arrives meanwhile is folded into the next step, so a slow machine lags
gracefully instead of queueing.

?words=ctc (off by default) adds the CTC word follower (docs/results/follower.md):
every 0.2 s a {"type": "word"} message says which word is being recited. The CTC
model (DUA_CTC_MODEL, default models/wav2vec2-quran-dua) loads on first use.

Debug sessions the page records (web/session-log.js) are uploaded to
data/sessions/, one .wav each with its log inside (scripts/session_report.py).
"""
from __future__ import annotations

import asyncio
import functools
import os
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from dua_recognition.align import CorpusIndex  # noqa: E402
from dua_recognition.asr import DEFAULT_MODEL, load_model  # noqa: E402
from dua_recognition.corpus import load_all, load_recordings, web_json  # noqa: E402
from dua_recognition.pipeline import StreamingRecognizer  # noqa: E402
from dua_recognition.tracker import RECITER, TrackerConfig  # noqa: E402

WEB = ROOT / "web"  # the one front end; it detects this server via /api/mode
MODEL = os.environ.get("DUA_ASR_MODEL", DEFAULT_MODEL)
ENGINE = os.environ.get("DUA_ENGINE", "server")  # "device": the browser runs speech recognition itself
SESSIONS = ROOT / "data" / "sessions"
CTC_MODEL = os.environ.get("DUA_CTC_MODEL", str(ROOT / "models" / "wav2vec2-quran-dua"))
_ctc = None
_ctc_lock = threading.Lock()


def ctc_model():
    """The CTC model for ?words=ctc, loaded once, on first use."""
    global _ctc
    with _ctc_lock:
        if _ctc is None:
            from dua_recognition.ctc import CtcModel

            _ctc = CtcModel(CTC_MODEL)
        return _ctc

DUAS = load_all()
INDEX = CorpusIndex(DUAS)
RECORDINGS = {r.audio_id: r for d in DUAS.values() for r in load_recordings(d)}

app = FastAPI(title="dua-recognition")


@app.on_event("startup")
def _warm() -> None:
    if ENGINE == "server":
        load_model(MODEL)  # first request shouldn't pay for the model load


@app.get("/api/mode")
def mode():
    return {"mode": ENGINE, "model": MODEL if ENGINE == "server" else None, "sessions": True}


@app.post("/api/sessions/{name}")
async def save_session(name: str, request: Request):
    """A debug session from the page: its audio, with the log in a RIFF chunk."""
    if not re.fullmatch(r"[\w-]{1,80}\.wav", name):
        raise HTTPException(400)
    SESSIONS.mkdir(parents=True, exist_ok=True)
    part = SESSIONS / (name + ".part")
    with part.open("wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    part.replace(SESSIONS / name)
    return {"saved": name}


@app.get("/corpus.json")
@app.get("/api/duas")
@functools.cache
def duas():
    return web_json(DUAS)  # built once: every line's reading is worked out here


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


def _message(update, step_ms: float, index: CorpusIndex, speed: float, unknown: float = 0.0,
             still_after: float = 0.3) -> dict:
    p = update.position
    still = update.quiet > still_after  # the reciter has stopped: so does the gliding highlight
    word = index.words[p.word] if p.word is not None else None
    return {
        "t": round(update.t, 2),
        "heard": update.transcript,
        "dua": p.dua,
        "dua_confidence": round(p.dua_confidence, 3),
        "segment": p.segment,
        "segment_confidence": round(p.segment_confidence, 3),
        "token": word.token if word and p.dua else None,
        "speed": 0.0 if still else round(speed, 3),  # words/s: the page glides the highlight at this pace
        "quiet": round(update.quiet, 2),  # seconds the reciter had been silent
        "unknown": round(unknown, 3),  # P(the recitation isn't in the corpus)
        # Reached the end of the line and the latest window was silent: the
        # next line is probably coming, so the UI previews it.
        "pause_at_line_end": bool(p.at_line_end and (not update.transcript or still)),
        "candidates": [
            {"id": d, "name": DUAS[d].name_en, "p": round(prob, 3)} for d, prob in p.candidates
        ],
        # Other texts reading the same words here (tracker.same_text_words): "also in ...".
        "same_as": p.same_as,
        "step_ms": round(step_ms),
    }


def _follow(mode: str) -> TrackerConfig:
    return TrackerConfig(**RECITER) if mode == "reciter" else TrackerConfig()


@app.websocket("/ws")
async def follow(ws: WebSocket):
    await ws.accept()
    # ?lead=0 / ?pauses=0 switch those off, for comparing by feel. ?follow=reciter:
    # majlis mode, where the display runs on through a reciter's breaths. ?words=ctc: the
    # word follower (an experiment, off by default).
    words = "ctc" if ws.query_params.get("words") == "ctc" else None
    rec = StreamingRecognizer(DUAS, model=MODEL, index=INDEX, lead=ws.query_params.get("lead") != "0",
                              pauses=ws.query_params.get("pauses") != "0",
                              config=_follow(ws.query_params.get("follow", "reading")),
                              words=words, ctc=await asyncio.to_thread(ctc_model) if words else None)
    running: asyncio.Task | None = None
    word_running: asyncio.Task | None = None

    async def idle():
        for task in (running, word_running):
            if task:
                await task

    async def run_word():
        t0 = time.perf_counter()
        wu = await asyncio.to_thread(rec.word_step)
        if wu is not None:
            token = rec.index.words[wu.word].token if wu.word is not None else None
            await ws.send_json({"type": "word", "t": round(wu.t, 2), "dua": wu.dua, "segment": wu.segment,
                                "token": token, "step_ms": round((time.perf_counter() - t0) * 1000)})

    async def run_step():
        t0 = time.perf_counter()
        update = await asyncio.to_thread(rec.step)
        if update is not None:
            await ws.send_json(_message(update, (time.perf_counter() - t0) * 1000, rec.index, rec.tracker.speed,
                                        rec.tracker.null, rec.tracker.cfg.still_after))

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text") or ""
            if text.startswith("lock:") and text[5:] in DUAS:
                await idle()
                rec.lock(text[5:])
                continue
            if text.startswith("seek:"):  # "I'm here": the listener tapped the line they're on
                dua, _, seg = text[5:].rpartition(":")
                if dua in rec.index.dua_ids and seg.lstrip("-").isdigit():
                    await idle()
                    try:
                        rec.seek(dua, int(seg))
                    except ValueError:  # no such line
                        pass
                continue
            if text.startswith("follow:"):  # the page switched between page and majlis mode
                await idle()
                rec.config = rec.tracker.cfg = _follow(text[7:])
                continue
            if msg.get("text") == "reset":
                await idle()
                rec.reset()
                continue
            data = msg.get("bytes")
            if not data:
                continue
            rec.push(np.frombuffer(data, dtype=np.float32))
            if rec.due and (running is None or running.done()):
                running = asyncio.create_task(run_step())
            if rec.word_due and (word_running is None or word_running.done()):
                word_running = asyncio.create_task(run_word())
    except WebSocketDisconnect:
        pass
    finally:
        for task in (running, word_running):
            if task:
                task.cancel()


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
