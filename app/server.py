#!/usr/bin/env python
"""The web app's server (web/): the page, majlis rooms, debug sessions and, in the server
engine, the speech models.

    python app/server.py                                  # http://localhost:8000
    DUA_ENGINE=server python app/server.py                # Whisper and the CTC model run here
    DUA_ENGINE=server DUA_ASR_DEVICE=cpu DUA_CTC_DEVICE=cpu python app/server.py   # ...on the CPU

Either way the page runs the same code (web/app.js DeviceEngine): the tracker, the stream
decoder, the highlight, practice mode. With the phone models exported to web/models/
(docs/development.md) the browser runs the models as well ("device", then the default), and this
serves the page, majlis rooms and debug sessions. In the server engine the page streams its audio
here (16 kHz int16 over /ws/ear) and asks for each Whisper and CTC window by the sample it ends at
(docs/results/server_engine.md). DUA_ASR_MODEL: Whisper (default the page's, as CTranslate2);
DUA_CTC_MODEL: the CTC model (default the page's student, on the page's 2 s windows; any Hugging
Face CTC model works, e.g. models/wav2vec2-quran-dua-voices).

Debug sessions the page records (web/session-log.js) are uploaded to data/sessions/ (or
DUA_SESSIONS), one .wav each with its log inside (scripts/session_report.py).
"""
from __future__ import annotations

import asyncio
import functools
import json
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

from dua_recognition.asr import (  # noqa: E402
    DEFAULT_MODEL, SAMPLE_RATE, QuietMeter, load_model, speech_in_tail, transcribe_batch)
from dua_recognition.corpus import load_all, load_recordings, web_json  # noqa: E402

WEB = ROOT / "web"  # the one front end; it detects this server via /api/mode
# The page's Whisper (web/app.js), as a CTranslate2 conversion: the tracker was tuned on its transcripts.
PAGE_MODEL = ROOT / "models" / "whisper-base-syn-v5-ctx8ft-ct2"
# The server's own Whisper: large-v3-turbo fine-tuned in full at an 8 s context, in two rounds (finetune_whisper.py;
# docs/results/server_profile.md), used when it's here and there's a GPU (on a CPU it takes seconds a window).
SERVER_ASR = ROOT / "models" / "whisper-turbo-srv2-ct2"


def _gpu() -> bool:
    if os.environ.get("DUA_ASR_DEVICE", "").lower() == "cpu":
        return False
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


# The page's own models (web/app.js: the Whisper default and CTC_MODEL): with them exported the
# browser runs speech recognition itself ("device"); otherwise this server does ("server").
PHONE_MODELS = [WEB / "models" / "whisper-base-syn-v5-ctx8ft" / "onnx", WEB / "models" / "ctc-student-base-v6-w2"]
ENGINE = os.environ.get("DUA_ENGINE") or ("device" if all(p.is_dir() for p in PHONE_MODELS) else "server")
MODEL = os.environ.get("DUA_ASR_MODEL") or (
    str(SERVER_ASR) if ENGINE == "server" and SERVER_ASR.is_dir() and _gpu()
    else str(PAGE_MODEL) if PAGE_MODEL.is_dir() else DEFAULT_MODEL)
SESSIONS = Path(os.environ.get("DUA_SESSIONS") or ROOT / "data" / "sessions")  # (replays: elsewhere)
MAX_SESSION_BYTES = 256 << 20  # 16 kHz int16 mono: over two hours
CTC_MODEL = os.environ.get("DUA_CTC_MODEL", str(ROOT / "models" / "ctc-student-base-v6"))
_ctc_lock = threading.Lock()


# The CTC windows: 3 s, the length the student was trained on. The phone sends it 2 s to save compute
# (web/models/ctc-student-base-v6-w2); on the server 3 s placed the word better and entered lines that
# open alike sooner (docs/results/server_profile.md). The page's decoder reads the frames that end 0.2 s
# before the window's end, whatever its length. DUA_CTC_WINDOW: another length (4 s broke the student).
EAR_CTC_WINDOW = float(os.environ.get("DUA_CTC_WINDOW", "3.0"))
_ear_ctc = None


def ear_ctc():
    global _ear_ctc
    with _ctc_lock:
        if _ear_ctc is None:
            from dua_recognition.ctc_student import StudentCtc, load_ctc

            # DUA_CTC_DEVICE=cpu: one 2 s window took 21 ms on this PC's CPU against 29 ms on its GPU,
            # and CUDA's setup costs 0.6 GB more memory.
            device = os.environ.get("DUA_CTC_DEVICE")
            student = (Path(CTC_MODEL) / "student.pt").exists()
            m = StudentCtc(CTC_MODEL, device) if device and student else load_ctc(CTC_MODEL)
            m.window_s = EAR_CTC_WINDOW
            m.window(np.zeros(int(EAR_CTC_WINDOW * SAMPLE_RATE), np.float32))  # the first run is the slow one
            _ear_ctc = m
        return _ear_ctc


DUAS = load_all()
RECORDINGS = {r.audio_id: r for d in DUAS.values() for r in load_recordings(d)}

app = FastAPI(title="dua-recognition")


@app.on_event("startup")
def _warm() -> None:
    if ENGINE == "server":
        if not os.environ.get("DUA_ENGINE"):
            print("web/models/ has no phone models: Whisper and the CTC model run here "
                  "(docs/development.md)", file=sys.stderr)
        load_model(MODEL)  # first request shouldn't pay for the model load
        transcribe_batch([np.zeros(SAMPLE_RATE, np.float32) + 0.01], model=MODEL)
        try:
            ear_ctc()
        except Exception as e:  # the page follows without it: Whisper and the tracker alone
            print(f"no CTC model ({CTC_MODEL}: {e}): no word-by-word following", file=sys.stderr)


def server_profile() -> dict:
    """The server engine's profile for the page (web/app.js DeviceEngine): the CTC step (s) and the
    stream decoder (`sc`) and tracker (`tc`) settings by their JavaScript names, as tuned for the
    server's models and delays. DUA_SERVER_PROFILE: a JSON file in its place ({} = the phone's)."""
    f = os.environ.get("DUA_SERVER_PROFILE")
    return json.loads(Path(f).read_text(encoding="utf-8")) if f else dict(SERVER_PROFILE)


# Tuned on the scenario bench for the server's models and delays (docs/results/server_profile.md): CTC
# steps every 0.05 s, the stream decoder's rules for repeated lines, salawat, far jumps and shared
# passages; the tracker's settings only with the turbo transcripts they were tuned on.
SERVER_PROFILE: dict = {
    "ctcHop": 0.05,
    "sc": {"hop": 0.05, "lineSteps": 3, "nextSteps": 2, "nextP": 0.7, "copies": True, "lapseIntj": 0.5,
           "ctcEvery": 10, "ctcPushP": 0.2, "ctcSure": 4.0, "cFar": -8, "sharedWords": 4, "sharedAfter": 10},
    **({"tc": {"kappa": 0.2, "nullRate": 0.35}} if Path(MODEL).resolve() == SERVER_ASR.resolve() else {}),
}


@app.get("/api/mode")
def mode():
    if ENGINE != "server":
        return {"mode": ENGINE, "model": None, "sessions": True}
    return {"mode": ENGINE, "model": Path(MODEL).name.removesuffix("-ct2"), "sessions": True,
            "ctc": f"{Path(CTC_MODEL).name}-w{EAR_CTC_WINDOW:g}", "profile": server_profile()}


@app.post("/api/sessions/{name}")
async def save_session(name: str, request: Request):
    """A debug session from the page: its audio, with the log in a RIFF chunk."""
    if not re.fullmatch(r"[\w-]{1,80}\.wav", name):
        raise HTTPException(400)
    if int(request.headers.get("content-length") or 0) > MAX_SESSION_BYTES:
        raise HTTPException(413)
    SESSIONS.mkdir(parents=True, exist_ok=True)
    part = SESSIONS / (name + ".part")
    try:
        size = 0
        with part.open("wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_SESSION_BYTES:
                    raise HTTPException(413)  # no or a false Content-Length: cap what's written
                f.write(chunk)
        part.replace(SESSIONS / name)
    finally:
        part.unlink(missing_ok=True)  # a dropped or refused upload leaves nothing behind
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


# -- the page's ear -------------------------------------------------------------
# The server engine: the page streams its audio here and asks for what web/asr-worker.js and
# web/ctc-worker.js give it on the device (a window's transcript and stop measures, a window's CTC
# frames); everything that follows the recitation (tracker.js, the stream decoder, the highlight,
# practice mode) runs in the page, the same code in either engine. A window is named by the sample
# it ends at, from the audio already sent, so audio crosses the network once.

class EarAudio:
    """The audio one page has streamed, kept by sample number."""

    KEEP = 30 * SAMPLE_RATE

    def __init__(self) -> None:
        self.at(0)

    def at(self, total: int) -> None:
        """The stream (re)starts at sample `total`: a new session, or a reconnect after audio was lost."""
        self.buf = np.zeros(0, np.float32)
        self.total = total

    def push(self, x: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, x])[-self.KEEP :]
        self.total += x.size

    def window(self, end: int, n: int) -> np.ndarray:
        """The n samples before sample `end`, zeros where none were heard (the page's buffer starts zeroed)."""
        first = self.total - self.buf.size  # sample number of buf[0]
        a, b = end - n - first, end - first
        out = np.zeros(n, np.float32)
        lo, hi = max(a, 0), min(b, self.buf.size)
        if hi > lo:
            out[lo - a : hi - a] = self.buf[lo:hi]
        return out


WINDOW = 6 * SAMPLE_RATE  # the page's Whisper window (web/app.js DeviceEngine.window)


def _gate(window: np.ndarray) -> dict:
    """web/gate.js decide(), policy "legacy": Whisper runs if the last 1.5 s reach -45 dBFS and
    Silero hears a run of speech in them (asr.speech_in_tail, the same rule)."""
    tail = window[-int(1.5 * SAMPLE_RATE) :]
    level = 20 * np.log10(np.sqrt(np.mean(np.square(tail, dtype=np.float64))) + 1e-12) if tail.size else -240.0
    out = {"policy": "legacy", "level": round(float(level), 1), "run": True, "reason": None}
    if level < -45:
        return {**out, "run": False, "reason": "below_floor"}
    if not speech_in_tail(window):
        return {**out, "run": False, "reason": "vad_reject"}
    return out


def _hear(window: np.ndarray, voice_db: float | None, dt: float | None) -> dict:
    """What asr-worker.js answers a "transcribe" with: the stop measures (gate.js stopMeasures is
    asr.QuietMeter on one window, the voice level handed back by the page) and, past the gate,
    Whisper's transcript. transcribe_batch also drops Whisper's no-speech and hallucinated outputs,
    as on the bench the tracker was tuned on."""
    t0 = time.perf_counter()
    meter = QuietMeter()
    meter.voice_db = voice_db
    quiet = meter(window, dt)
    gate = _gate(window)
    text, infer_ms = "", None
    if gate["run"]:
        t1 = time.perf_counter()
        text = transcribe_batch([window], model=MODEL)[0]
        infer_ms = round((time.perf_counter() - t1) * 1000, 1)
    return {"type": "text", "text": text, "quiet": quiet, "paused": meter.paused, "voiceDb": meter.voice_db,
            "gate": gate, "skip": None if text else gate["reason"] or "empty", "hallucinated": False,
            "infer_ms": infer_ms, "server_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.websocket("/ws/ear")
async def ear(ws: WebSocket):
    """Binary messages: 16 kHz mono int16 audio. Text: JSON requests, each with its channel ("asr" or
    "ctc", as the worker it stands in for) and the page's session number `gen`, echoed back:
    {"type": "at", "total"}: the stream is at this sample (a new session, or audio lost to a reconnect);
    {"type": "load"}: answered "ready" once the model is loaded;
    {"type": "transcribe", "id", "voiceDb", "dt"}: the window ending at sample id;
    {"type": "frames", "id"}: the CTC frames of the 2 s ending there, as a JSON header
    {"type": "frames", "id", "T", "C", ...} followed by T x C float16 log probabilities."""
    await ws.accept()
    audio = EarAudio()
    sending = asyncio.Lock()
    tasks: set[asyncio.Task] = set()

    async def answer(req: dict, reply: dict, data: bytes | None = None) -> None:
        reply = {"ch": req["ch"], "id": req.get("id"), "gen": req.get("gen"), **reply}
        async with sending:  # a header and its frames go out together
            await ws.send_json(reply)
            if data is not None:
                await ws.send_bytes(data)

    async def transcribe(req: dict, window: np.ndarray) -> None:
        try:
            reply = await asyncio.to_thread(_hear, window, req.get("voiceDb"), req.get("dt"))
        except Exception as e:  # as the worker does: the page carries on with no transcript
            reply = {"type": "text", "text": "", "quiet": None, "paused": None, "voiceDb": req.get("voiceDb"),
                     "gate": None, "skip": "error", "hallucinated": False, "error": str(e)}
        await answer(req, reply)

    async def frames(req: dict, window: np.ndarray) -> None:
        t0 = time.perf_counter()
        try:
            lp = await asyncio.to_thread(lambda: ear_ctc().window(window))
        except Exception as e:
            return await answer(req, {"type": "error", "message": str(e)})
        await answer(req, {"type": "frames", "T": lp.shape[0], "C": lp.shape[1],
                           "server_ms": round((time.perf_counter() - t0) * 1000, 1)}, lp.astype("<f2").tobytes())

    async def load(req: dict) -> None:
        try:
            if req["ch"] == "asr":
                await asyncio.to_thread(load_model, MODEL)
                return await answer(req, {"type": "ready"})
            await asyncio.to_thread(ear_ctc)
            await answer(req, {"type": "ready", "meta": {"window_s": EAR_CTC_WINDOW}})
        except Exception as e:
            await answer(req, {"type": "error", "message": str(e)})

    def spawn(coro) -> None:
        task = asyncio.create_task(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                audio.push(np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768)
                continue
            try:
                req = json.loads(msg.get("text") or "")
                kind, ch = req.get("type"), req.get("ch")
            except (ValueError, AttributeError):
                continue
            if kind == "at":
                audio.at(int(req.get("total") or 0))
            elif kind == "load" and ch in ("asr", "ctc"):
                spawn(load(req))
            # The window is cut now, from the audio received so far: the page sent it before asking.
            elif kind == "transcribe" and ch == "asr":
                end = int(req["id"])
                spawn(transcribe(req, audio.window(end, min(WINDOW, max(end, 0)))))
            elif kind == "frames" and ch == "ctc":
                spawn(frames(req, audio.window(int(req["id"]), int(EAR_CTC_WINDOW * SAMPLE_RATE))))
    except WebSocketDisconnect:
        pass
    finally:
        for task in list(tasks):
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
