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

    feats = np.stack([pad_or_trim(m.feature_extractor(windows[i])) for i in live])
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
