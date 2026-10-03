"""Arabic transcription of short windows with faster-whisper's CTranslate2 model.

Whisper is the backbone for passage localization: transcribe the window, then
align the text. Anything that loads as a faster-whisper model works here — a
stock size ("large-v3-turbo", "base", ...) or a CTranslate2 conversion of the
fine-tuned model (scripts/finetune_whisper.py).

We call the encoder/decoder directly rather than going through
`WhisperModel.transcribe`: windows are at most a few seconds, so its long-form
machinery (VAD, seeking, timestamp segmentation) buys nothing, and going direct
lets many windows share one batched forward pass. The live path and the
offline evaluation use this same function, batch of one or batch of many.
"""
from __future__ import annotations

import os
import zlib
from functools import lru_cache
from typing import Sequence

import numpy as np

from .text import strip_diacritics

DEFAULT_MODEL = os.environ.get("DUA_ASR_MODEL", "large-v3-turbo")
SAMPLE_RATE = 16000

# Whisper's stock hallucinations on silence and music. Arabic YouTube subtitles
# taught it to "caption" quiet audio with credits; none of these occur in a du'a.
_HALLUCINATIONS = ("اشترك", "ترجمة", "نانسي", "المترجم", "للمشاهدة", "موسيقى")
# Whole-window outputs it produces in pauses. "شكر" does occur in du'as, so
# these only count when they are the entire transcript.
_HALLUCINATED_WHOLE = {"شكرا", "شكرا لكم", "شكرا جزيلا", "شكرا لك"}
_SILENCE_DBFS = -45.0


def _default_device() -> str:
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


@lru_cache(maxsize=4)
def load_model(name: str = DEFAULT_MODEL, device: str | None = None, compute_type: str | None = None):
    from faster_whisper import WhisperModel

    device = device or os.environ.get("DUA_ASR_DEVICE") or _default_device()
    compute_type = compute_type or ("float16" if device == "cuda" else "int8")
    return WhisperModel(name, device=device, compute_type=compute_type)


@lru_cache(maxsize=4)
def _tokenizer(name: str):
    from faster_whisper.tokenizer import Tokenizer

    m = load_model(name)
    return Tokenizer(m.hf_tokenizer, m.model.is_multilingual, task="transcribe", language="ar")


def _is_silent(x: np.ndarray) -> bool:
    if not x.size:
        return True
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20 * np.log10(rms + 1e-12) < _SILENCE_DBFS


def speech_in_tail(window: np.ndarray, tail_s: float = 1.5) -> bool:
    """Silero VAD (bundled with faster-whisper) over the window's last tail_s.

    A room is never silent: between lines a phone mic hears fans, breath and
    echo, which Whisper happily "transcribes" into words that match some other
    line. If nothing in the newest audio is speech, the update should be a
    pause (the tracker holds still), whatever Whisper would have said.

    Rule (mirrored exactly in web/asr-worker.js): speech means a run of at
    least 7 frames (7 x 32 ms) whose probability stays >= 0.35 and peaks >= 0.5.
    """
    from faster_whisper.vad import get_vad_model

    tail = window[-int(tail_s * SAMPLE_RATE) :]
    if _is_silent(tail):
        return False
    tail = np.pad(tail, (0, (-tail.size) % 512)).astype(np.float32)
    probs = np.ravel(get_vad_model()(tail)) if tail.size else np.zeros(0)
    return _speech_run(probs)


VAD_FRAME_S = 512 / SAMPLE_RATE


def speech_runs(probs: np.ndarray, min_frames: int = 7, on: float = 0.5, off: float = 0.35) -> list[tuple[int, int]]:
    """Frame spans [start, end) that count as speech under the rule above."""
    runs, start, peak = [], None, 0.0
    for i, p in enumerate(list(probs) + [0.0]):
        if p >= off:
            start = i if start is None else start
            peak = max(peak, p)
        else:
            if start is not None and i - start >= min_frames and peak >= on:
                runs.append((start, i))
            start, peak = None, 0.0
    return runs


# The stop detector (quiet_at_end): energy counts as sound only within this many dB
# of the reciter's voice, and a window's voice level is trusted from this many
# frames of speech. Tuned in docs/results/stops.md.
QUIET_REL_DB: float | None = 15.0
VOICE_MIN_FRAMES = 16
LIVE_REL_DB = 15.0  # the page's live detector (LiveQuiet), energy alone


def quiet_at_end(window: np.ndarray, tail_s: float = 3.0, energy_db: float | None = 6.0,
                 rel_db: float | None = QUIET_REL_DB, voice_db: float | None = None) -> float:
    """Seconds since the reciter last made a sound (tail_s if not in the last tail_s).

    The transcript of a 6 s window keeps the last words for seconds after the
    reciter stops. This says how much of that is silence, so the tracker only
    moves the position for time the reciter was actually speaking.

    A 32 ms frame counts as sound if Silero calls it speech (p >= 0.35) or it
    is `energy_db` above the window's floor (its 10th-percentile frame
    energy, at least -70 dBFS); sound is a run of at least 3 frames. Silero alone misses long
    melodic notes: against forced-aligned word timings it called 19% of the
    test reciters' mid-word moments silent for > 0.3 s, and the energy term
    brings that to 4% (docs/results/pauses.md). None = Silero only.

    ...and the energy must also come within `rel_db` of the reciter's voice
    (`voice_db`, from earlier windows: QuietMeter; None = this window's own
    voice_level). A phone's automatic gain control turns the room up by 10-15
    dB within a second of the reciter stopping, and against the window's floor
    that hiss read as someone still reciting: in Hasan's sessions a quarter of
    the moments after he stopped (docs/results/stops.md). A long note is loud;
    the hiss sits 20-25 dB under the voice. With no voice level known, energy
    alone counts for nothing. rel_db None = the floor alone decides, as before.
    Mirrored in web/gate.js (quietAtEnd).
    """
    probs, db, floor = _tail_activity(window, tail_s)
    if probs is None:
        return tail_s
    return _quiet(probs, db, floor, energy_db, rel_db, voice_db)


def voice_level(probs: np.ndarray, db: np.ndarray, min_frames: int = VOICE_MIN_FRAMES) -> float | None:
    """The reciter's voice in dBFS: the median frame Silero is sure is speech
    (p >= 0.5), from at least `min_frames` of them. A breath or a click after
    they stop can be "speech" for a few frames, and it is 20 dB quieter."""
    voiced = db[probs >= 0.5]
    return float(np.median(voiced)) if voiced.size >= max(1, min_frames) else None


class QuietMeter:
    """quiet_at_end over one stream's windows in order, remembering the voice
    level from the last window that had enough speech to measure it: after the
    reciter stops, the window holds only hiss, which has no voice to compare
    with. Mirrored in web/app.js (the page keeps voiceDb and hands it to the
    worker's quietAtEnd).

    With `dt` it also says how much of the window's last dt seconds went on
    pauses (silences of min_pause or more, anywhere in them, not only at the
    end): `paused`. The tracker's belief moves only for the rest. A breath or
    a click a second after the reciter stopped otherwise made the whole gap
    since the last update count as reciting (docs/results/stops.md).
    """

    def __init__(self, tail_s: float = 3.0, energy_db: float | None = 6.0, rel_db: float | None = QUIET_REL_DB,
                 min_pause: float = 0.3):
        self.tail_s, self.energy_db, self.rel_db, self.min_pause = tail_s, energy_db, rel_db, min_pause
        self.voice_db: float | None = None
        self.paused: float | None = None

    def __call__(self, window: np.ndarray, dt: float | None = None) -> float:
        probs, db, floor = _tail_activity(window, self.tail_s)
        if probs is None:
            self.paused = None if dt is None else dt
            return self.tail_s
        v = voice_level(probs, db)
        if v is not None:
            self.voice_db = v
        runs = _sound_runs(probs, db, floor, self.energy_db, self.rel_db, self.voice_db)
        self.paused = None if dt is None else min(dt, _paused(runs, probs.size, dt, self.min_pause))
        return (probs.size - runs[-1][1]) * VAD_FRAME_S if runs else probs.size * VAD_FRAME_S


class LiveQuiet:
    """Seconds the reciter has been silent, on the audio the page holds right now.

    The window's quiet (quiet_at_end) is a second or two old by the time the
    page shows its update: Whisper was running while the reciter went on, or
    stopped. The page hears that audio as it arrives, so it can tell at once.
    Silero is in the worker, busy with Whisper, so this goes by loudness
    alone: a 32 ms frame is sound if it comes within `rel_db` of the voice
    level the worker last measured (QuietMeter.voice_db); sound is a run of
    at least 3 frames. Until a voice level is known it says nothing (None).

    Unlike quiet_at_end there is no floor term (6 dB over the quietest tenth
    of the last seconds): without Silero beside it, it called 12% of the
    test reciters' mid-word moments silent, against 0.5% without it, and it
    caught hardly any more of Hasan's stops (docs/results/stops.md).
    Mirrored in web/gate.js.
    """

    FRAME = 512

    def __init__(self, rel_db: float = LIVE_REL_DB):
        self.rel_db = rel_db
        self.rest = np.zeros(0, np.float32)
        self.voice_db: float | None = None
        self.n = 0  # frames heard
        self.run = 0  # frames in the current run of sound
        self.last_end = 0  # frames heard when the last run of 3+ was last sounding

    def push(self, x: np.ndarray) -> None:
        x = np.concatenate([self.rest, np.asarray(x, np.float32)])
        k = x.size // self.FRAME
        self.rest = x[k * self.FRAME :]
        frames = x[: k * self.FRAME].reshape(k, self.FRAME).astype(np.float64)
        for d in 10 * np.log10(np.mean(frames * frames, axis=1) + 1e-12):
            self.n += 1
            sound = self.voice_db is not None and d >= self.voice_db - self.rel_db
            self.run = self.run + 1 if sound else 0
            if self.run >= 3:
                self.last_end = self.n

    @property
    def quiet(self) -> float | None:
        if self.voice_db is None:
            return None
        return (self.n - self.last_end) * self.FRAME / SAMPLE_RATE


def _tail_activity(window: np.ndarray, tail_s: float):
    """Silero probabilities and frame energies (dBFS) over the window's last tail_s,
    in 512-sample frames ending exactly at the window's end, plus the whole
    window's energy floor. (None, ...) if the tail is digital silence."""
    from faster_whisper.vad import get_vad_model

    x = np.asarray(window, dtype=np.float32)
    x = x[x.size % 512 :]
    n_tail = min(x.size, int(tail_s * SAMPLE_RATE) // 512 * 512)
    if not n_tail or _is_silent(x[-n_tail:]):
        return None, None, None
    frames = x.reshape(-1, 512).astype(np.float64)
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    floor = float(np.percentile(db, 10))
    # (a copy: faster-whisper's VAD zeroes the last 64 samples of what it's given, in place)
    probs = np.ravel(get_vad_model()(x[-n_tail:].copy()))[: n_tail // 512]
    return probs, db[-probs.size :], floor


def _quiet(probs: np.ndarray, db: np.ndarray, floor: float, energy_db: float | None,
           rel_db: float | None = None, voice_db: float | None = None) -> float:
    runs = _sound_runs(probs, db, floor, energy_db, rel_db, voice_db)
    if not runs:
        return probs.size * VAD_FRAME_S
    return (probs.size - runs[-1][1]) * VAD_FRAME_S


def _sound_runs(probs: np.ndarray, db: np.ndarray, floor: float, energy_db: float | None,
                rel_db: float | None = None, voice_db: float | None = None) -> list[tuple[int, int]]:
    """Frame spans [start, end) of sound, by quiet_at_end's rule."""
    if energy_db is None:
        return speech_runs(probs)
    # The floor is never taken below -70 dBFS: phones and headsets with noise
    # suppression output exact zeros between words, and against a floor of
    # digital silence a faint click read as "still reciting".
    bar = max(floor, -70.0) + energy_db
    if rel_db is not None:
        voice = voice_db if voice_db is not None else voice_level(probs, db)
        bar = max(bar, voice - rel_db) if voice is not None else np.inf
    active = (probs >= 0.35) | (db >= bar)
    runs, start = [], None
    for i, a in enumerate(list(active) + [False]):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start >= 3:
                runs.append((start, i))
            start = None
    return runs


def _paused(runs: list[tuple[int, int]], n: int, seconds: float, min_pause: float) -> float:
    """Seconds of the last `seconds` of n frames spent in silences of at least
    `min_pause` (between runs of sound, or before the first or after the last)."""
    lo = max(0, n - int(round(seconds / VAD_FRAME_S)))
    edges = [0] + [x for r in runs for x in r] + [n]
    out = 0
    for a, b in zip(edges[::2], edges[1::2]):  # the gaps
        if (b - a) * VAD_FRAME_S >= min_pause:
            out += max(0, b - max(a, lo))
    return out * VAD_FRAME_S


def _speech_run(probs: np.ndarray, min_frames: int = 7, on: float = 0.5, off: float = 0.35) -> bool:
    run, peak = 0, 0.0
    for p in probs:
        if p >= off:
            run += 1
            peak = max(peak, p)
            if run >= min_frames and peak >= on:
                return True
        else:
            run, peak = 0, 0.0
    return False


def _looks_hallucinated(text: str) -> bool:
    bare = text.strip(" .!؟?،,")
    if bare in _HALLUCINATED_WHOLE or any(h in text for h in _HALLUCINATIONS):
        return True
    raw = text.encode("utf-8")
    # A decoding loop ("الله الله الله ...") compresses far better than speech.
    return len(raw) > 40 and len(raw) / len(zlib.compress(raw)) > 2.4


@lru_cache(maxsize=64)
def _suppress_outside(model: str, text: str) -> tuple[int, ...]:
    """Every text token that never occurs when writing `text`, to suppress.

    Once the listener's du'a is known, Whisper should only be able to write
    words from it (as the Gurbani captioning system does with a lexicon trie;
    this is the token-level version CTranslate2 supports). Each word is
    tokenized both word-initial and mid-line, in the spelling Whisper uses.
    """
    import re

    tok = _tokenizer(model)
    words = {re.sub(r"[^ء-ي]", "", strip_diacritics(w).replace("ٱ", "ا")) for w in text.split()}
    allowed: set[int] = set()
    for w in filter(None, words):
        allowed.update(tok.encode(w))
        allowed.update(tok.encode(" " + w))
    return tuple(t for t in range(tok.eot) if t not in allowed)


def transcribe_batch(
    windows: Sequence[np.ndarray],
    *,
    model: str = DEFAULT_MODEL,
    prompts: Sequence[str | None] | None = None,
    max_new_tokens: int = 96,
    vad: bool = False,
    constrain_to: str | None = None,
) -> list[str]:
    """Transcribe float32 16 kHz windows (each <= 30 s) in one forward pass.

    prompts[i], if given, is fed to the decoder as preceding text: pass the
    reference just before where the tracker thinks the reciter is, and Whisper
    leans toward the right vocabulary.
    """
    if model.startswith("ctc:"):
        return _transcribe_ctc(windows, model[4:])

    from faster_whisper.audio import pad_or_trim

    m = load_model(model)
    tok = _tokenizer(model)
    prompts = list(prompts) if prompts is not None else [None] * len(windows)
    out = [""] * len(windows)
    live = [i for i, w in enumerate(windows) if not _is_silent(w) and (not vad or speech_in_tail(w))]
    if not live:
        return out

    # To the model's own context: 30 s, or less for scripts/shorten_context.py's models.
    frames = m.feature_extractor.nb_max_frames
    feats = np.stack([pad_or_trim(m.feature_extractor(windows[i]), frames) for i in live])
    encoded = m.encode(feats)
    token_prompts = []
    for i in live:
        p = []
        if prompts[i]:
            p = [tok.sot_prev] + tok.encode(" " + prompts[i].strip())[-(m.max_length // 2 - 1) :]
        token_prompts.append(p + list(tok.sot_sequence) + [tok.no_timestamps])

    # CTranslate2 needs <|startoftranscript|> at the same position across a
    # batch, so windows whose prompts differ in length decode in groups.
    results = [None] * len(live)
    by_len: dict[int, list[int]] = {}
    for k, p in enumerate(token_prompts):
        by_len.setdefault(len(p), []).append(k)
    for n, ks in by_len.items():
        enc = encoded if len(ks) == len(live) else _select(encoded, ks)
        for k, res in zip(ks, m.model.generate(
            enc,
            [token_prompts[k] for k in ks],
            beam_size=1,
            max_length=n + max_new_tokens,
            return_no_speech_prob=True,
            suppress_blank=True,
            suppress_tokens=[-1, *(_suppress_outside(model, constrain_to) if constrain_to else ())],
        )):
            results[k] = res
    for i, res in zip(live, results):
        tokens = [t for t in res.sequences_ids[0] if t < tok.eot]
        text = tok.decode(tokens).strip()
        if res.no_speech_prob > 0.6 or _looks_hallucinated(text):
            continue
        out[i] = text
    return out


@lru_cache(maxsize=2)
def _ctc(repo: str):
    """A wav2vec2-style CTC model (optional backend; needs torch + transformers)."""
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    proc = AutoProcessor.from_pretrained(repo)
    model = AutoModelForCTC.from_pretrained(repo).eval()
    if torch.cuda.is_available():
        model = model.half().cuda()
    else:
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return proc, model


def _transcribe_ctc(windows: Sequence[np.ndarray], repo: str) -> list[str]:
    """CTC has no decoder: one encoder pass per window, greedy decoding."""
    import torch

    proc, model = _ctc(repo)
    out = [""] * len(windows)
    live = [i for i, w in enumerate(windows) if not _is_silent(w)]
    if not live:
        return out
    inp = proc([windows[i] for i in live], sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    x = inp.input_values
    mask = inp.get("attention_mask")
    if next(model.parameters()).is_cuda:
        x, mask = x.half().cuda(), (mask.cuda() if mask is not None else None)
    with torch.no_grad():
        ids = model(x, attention_mask=mask).logits.argmax(-1).cpu()
    for i, text in zip(live, proc.batch_decode(ids)):
        out[i] = text.strip()
    return out


def _select(encoded, rows: list[int]):
    """Rows of a CTranslate2 StorageView (encoder output), as a new one."""
    import ctranslate2

    arr = np.asarray(encoded.to_device(ctranslate2.Device.cpu) if encoded.device == "cuda" else encoded)
    sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(arr[rows]))
    return sv.to_device(ctranslate2.Device.cuda) if encoded.device == "cuda" else sv


def transcribe(audio, *, model: str = DEFAULT_MODEL, prompt: str | None = None) -> str:
    """Transcribe one window: a path, or float32 samples at 16 kHz."""
    if not isinstance(audio, np.ndarray):
        from faster_whisper.audio import decode_audio

        audio = decode_audio(audio, sampling_rate=SAMPLE_RATE)
    return transcribe_batch([audio], model=model, prompts=[prompt])[0]
