"""The explainer's voice-over: each beat's audio and its subtitles (the words are in lines.py).

The narrator speaks English with an Arabic accent: OmniVoice (k2-fsa, Apache-2.0) clones narrator.flac, a voice
it designed itself speaking Iraqi Arabic (no real person's voice), into English. Islamic words in the English
are written in Arabic script (lines.TTS_SPELLING), so the same voice says them natively, and the Arabic phrases,
marked in a line as [[ar:ARABIC|transliteration]], are spoken in Arabic between sentences. Every sentence is
voiced up to a few times and the take Whisper transcribes closest to the text is kept, cut cleanly after its
last word, so a garbled take or a trailing hum never reaches the video.

Each line is cached by its text in data/cache/explainer_voice/ as a .wav and a .json listing its sentences with
their start and end, which the subtitles use. Editing a line re-voices only that line.

    <venv with omnivoice>/python docs/explainer/voice.py [KEY ...]   # GPU

The takes are voiced first (OmniVoice alone, in its own process), then chosen (Whisper alone): the two together
pushed the machine's commit past its limit. EXPLAINER_PICK_DEVICE=cpu EXPLAINER_PICK_MODEL=openai/whisper-small
when memory is short; EXPLAINER_TAKES_LIMIT=N voices N takes per process (OmniVoice's commit grows as it goes).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lines import CAPTION_NUMBERS, LINES, SLOW, TTS_SPELLING  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("EXPLAINER_VOICE_CACHE", ROOT / "data" / "cache" / "explainer_voice"))
# The narrator: OmniVoice's own designed voice ("male, middle-aged, moderate pitch") reading NARRATOR_TEXT
# in Iraqi Arabic.
NARRATOR = Path(__file__).parent / "narrator.flac"
NARRATOR_TEXT = "الناس يتعلمون الدعاء بالبيت وبالجامع، ويقرونه سوه بالليالي المباركة. وهذا التطبيق يسمع القارئ، ويتابعه كلمة كلمة."
SPEED = 0.9  # unhurried (1.0, its natural pace, was too fast; the old explainer used 0.92)
ENGINE = f"omnivoice|narrator-iraqi|speed{SPEED}"
GAP, AR_GAP = 0.5, 0.55  # seconds between sentences, and around an Arabic phrase
SLOW_SPEED, SLOW_GAP = 0.86, 0.5
TAKES = {"en": 4, "ar": 8}  # tries per segment; Arabic phrases are short and less steady
GOOD = {"en": 0.93, "ar": 0.92}  # a take this close to the text is kept at once

_SPELL = re.compile("|".join(re.escape(k) + r"(?![a-z])" for k in sorted(TTS_SPELLING, key=len, reverse=True)))
_AR = re.compile(r"\[\[ar:([^|\]]+)\|([^\]]+)\]\]")


def segments(text: str, key: str = "") -> list[dict]:
    """A line's parts in order: English sentences and Arabic phrases, each with what is shown
    (`show`) and what is voiced (`say`, `lang`, and `speed` where the line is said slower)."""
    out, at = [], 0
    for m in list(_AR.finditer(text)) + [None]:
        chunk = text[at : m.start() if m else len(text)]
        for s in re.split(r"(?<=[.?!])\s+", chunk.strip()):
            if s:
                out.append({"lang": "en", "show": s, "say": _SPELL.sub(lambda k: TTS_SPELLING[k.group()], s)})
                if key in SLOW:
                    out[-1]["speed"] = SLOW_SPEED
        if m:
            out.append({"lang": "ar", "show": m.group(2), "say": m.group(1)})
            at = m.end()
    return out


def caption(text: str) -> str:
    for words, digits in CAPTION_NUMBERS:
        text = text.replace(words, digits)
    return text


def _tag(key: str) -> str:
    pace = f"|slow{SLOW_SPEED},{SLOW_GAP}" if key in SLOW else ""
    return hashlib.sha1(f"{ENGINE}|{LINES[key]}|{sorted(TTS_SPELLING.items())}{pace}".encode()).hexdigest()[:10]


def clip(key: str) -> tuple[Path, float, list[dict]]:
    """The voiced line, its length in seconds and its parts with their times (for subtitles)."""
    path = CACHE / f"{key}-{_tag(key)}.wav"
    if not path.exists() and os.environ.get("EXPLAINER_STANDIN") == "1":
        return standin(key)
    if not path.exists():
        raise FileNotFoundError(f"{path.name}: voice the narration first (python docs/explainer/voice.py)")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return path, meta["duration"], meta["segments"]


def standin(key: str) -> tuple[Path, float, list[dict]]:
    """Silence as long as the line will roughly take to say (2.6 words a second), with its sentences
    timed in proportion: EXPLAINER_STANDIN=1 renders the scene's layout before the voicing."""
    import numpy as np
    import soundfile as sf

    parts, t = segments(LINES[key], key), 0.0
    for i, seg in enumerate(parts):
        t += (GAP if i else 0.0)
        d = len(seg["show"].split()) / (2.6 * (SLOW_SPEED / SPEED if key in SLOW else 1.0)) + 0.3
        seg.update(start=round(t, 3), end=round(t + d, 3))
        t += d
    path = CACHE / "standin" / f"{key}-{_tag(key)}.wav"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.zeros(int(t * 16000), np.float32), 16000)
    return path, round(t, 3), parts


# -- voicing (needs omnivoice, transformers and a GPU) ------------------------------------------------
def _plain(s: str, lang: str) -> str:
    """Letters only, for scoring a take against its text: digits and number words dropped
    (Whisper writes 115 for "a hundred and fifteen"), Arabic without its vowel marks."""
    s = s.lower()
    if lang == "ar":
        s = re.sub(r"[ً-ْٰـ]", "", s).replace("أ", "ا").replace("إ", "ا").replace("ة", "ه")
        return re.sub(r"[^ء-ي]", "", s)
    nums = (r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
            r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
            r"thousand|million|percent|point|half|and a)\b")
    s = re.sub(nums, " ", re.sub(r"[\d%]+", " ", s))
    return re.sub(r"[^a-z]", "", s)


def _trim(y, sr, top_db: float = 40.0, keep: float = 0.05):
    import numpy as np

    env = np.abs(y)
    on = np.flatnonzero(env > env.max() * 10 ** (-top_db / 20))
    if not on.size:
        return y
    a, b = max(0, on[0] - int(keep * sr)), min(len(y), on[-1] + int(keep * sr))
    return y[a:b]


def _speech_end(y, sr, last_word: float, reach: float = 0.5) -> int:
    """Where to cut after the last word Whisper heard: once the voice has died away (30 dB under
    its peak), but no later than `reach` seconds past the word, so a hum or rumble is left out."""
    import numpy as np

    hop = int(0.02 * sr)
    floor = np.abs(y).max() * 10 ** (-30 / 20)
    end = int((last_word + 0.1) * sr)
    limit = min(len(y), int((last_word + reach) * sr))
    while end < limit and np.sqrt(np.mean(y[end : end + hop] ** 2)) > floor * 0.5:
        end += hop
    return min(end, limit, len(y))


def _fade(y, sr, secs: float = 0.06):
    import numpy as np

    n = min(len(y), int(secs * sr))
    y = y.copy()
    y[len(y) - n :] *= np.linspace(1.0, 0.0, n, dtype=y.dtype)
    return y


def _seg_path(seg: dict) -> Path:
    speed = f"|speed{seg['speed']}" if "speed" in seg else ""
    tag = hashlib.sha1(f"{ENGINE}|{seg['lang']}|{seg['say']}{speed}".encode()).hexdigest()[:12]
    return CACHE / "segments" / f"{seg['lang']}-{tag}.wav"


def _todo(lang: str, keys: list[str]) -> dict[str, dict]:
    segs = {str(_seg_path(s)): s for k in keys for s in segments(LINES[k], k) if s["lang"] == lang}
    return {p: s for p, s in segs.items() if not Path(p).exists()}


def _take_path(path: str, take: int) -> Path:
    return CACHE / "takes" / f"{Path(path).stem}-{take}.wav"


def take_pass(lang: str, keys: list[str]) -> None:
    """Voice every not-yet-voiced segment in one language TAKES[lang] times (OmniVoice alone: with
    Whisper beside it, the two pushed the machine's commit past its limit)."""
    import gc

    import numpy as np
    import soundfile as sf
    import torch

    segs = _todo(lang, keys)
    todo = [(p, s, k) for p, s in segs.items() for k in range(TAKES[lang]) if not _take_path(p, k).exists()]
    print(f"{lang}: {len(segs)} segments to voice, {len(todo)} takes", flush=True)
    if not todo:
        return
    (CACHE / "takes").mkdir(parents=True, exist_ok=True)
    import librosa
    from omnivoice import OmniVoice

    # EXPLAINER_TTS_DEVICE=cpu: slower, but on Windows the GPU's memory counts against the machine's
    # commit too, and with other jobs running the GPU pass can push it past the limit.
    cpu = os.environ.get("EXPLAINER_TTS_DEVICE") == "cpu"
    tts = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu" if cpu else "cuda:0",
                                    dtype=torch.bfloat16 if cpu else torch.float16)
    sr = tts.sampling_rate
    # The narrator's voice prompt, made once and kept: making it costs about 3 GB more than voicing.
    tag = hashlib.sha1(NARRATOR.read_bytes() + NARRATOR_TEXT.encode()).hexdigest()[:10]
    keep = CACHE / f"narrator_prompt-{tag}.pt"
    if keep.exists():
        prompt = torch.load(keep, weights_only=False)
    else:
        ref, _ = librosa.load(NARRATOR, sr=sr)
        prompt = tts.create_voice_clone_prompt((torch.from_numpy(ref)[None], sr), ref_text=NARRATOR_TEXT)
        torch.save(prompt, keep)
    gc.collect()
    limit = int(os.environ.get("EXPLAINER_TAKES_LIMIT", "0")) or len(todo)  # a few, to time them
    for path, seg, take in todo[:limit]:
        torch.manual_seed(take)
        y = tts.generate(text=seg["say"], language="en" if lang == "en" else "arb", voice_clone_prompt=prompt,
                         speed=seg.get("speed", SPEED), num_step=32)[0]
        sf.write(_take_path(path, take), _trim(np.asarray(y, dtype=np.float32), sr), sr)
    print(f"{lang}: {min(limit, len(todo))} takes voiced", flush=True)


def pick_pass(lang: str, keys: list[str]) -> None:
    """Keep each segment's best take: the one Whisper transcribes closest to the text, cut cleanly
    after its last word."""
    import difflib
    import gc

    import soundfile as sf
    import torch
    from transformers import pipeline

    segs = _todo(lang, keys)
    if not segs:
        return
    (CACHE / "segments").mkdir(parents=True, exist_ok=True)
    # Whisper through transformers, not faster-whisper: CTranslate2's CUDA beside torch's crashes the process.
    # EXPLAINER_PICK_DEVICE=cpu: as EXPLAINER_TTS_DEVICE (the GPU's memory counts against the commit too).
    cpu = os.environ.get("EXPLAINER_PICK_DEVICE") == "cpu"
    # EXPLAINER_PICK_MODEL: a smaller Whisper when even turbo on the CPU is too much for the machine's commit.
    model = os.environ.get("EXPLAINER_PICK_MODEL", "openai/whisper-large-v3-turbo")
    whisper = pipeline("automatic-speech-recognition", model=model, dtype=torch.float32 if cpu else torch.float16,
                       device="cpu" if cpu else "cuda")
    gc.collect()
    for path, seg in segs.items():
        target = _plain(seg["say"] if lang == "ar" else seg["show"], lang)
        best = (-9.0, None, "", None)
        for take in range(TAKES[lang]):
            if not _take_path(path, take).exists():
                continue
            y, sr = sf.read(_take_path(path, take), dtype="float32")
            out = whisper({"raw": _resample(y, sr), "sampling_rate": 16000}, return_timestamps="word",
                          generate_kwargs={"language": lang, "task": "transcribe"})
            heard = out["text"].strip()
            # The Arabic model sometimes hums or rumbles on after the last word: cut 0.15 s after the
            # last word Whisper heard, and fade out.
            ends = [c["timestamp"][1] for c in out.get("chunks", []) if c["timestamp"][1]]
            if ends:
                y = _fade(y[: _speech_end(y, sr, max(ends))], sr, 0.08)
            got = _plain(heard, lang)
            score = difflib.SequenceMatcher(None, target, got).ratio()
            # A take may also run on past the text (TTS models sometimes add words): penalise length.
            score -= 0.5 * max(0.0, len(got) / max(1, len(target)) - 1.15)
            if lang == "ar":  # and in time: a held hum fools the transcript but not the clock
                score -= 0.5 * max(0.0, len(y) / sr / (0.13 * len(target) + 0.5) - 1.3)
            if score > best[0]:
                best = (score, y, heard, sr)
            if score >= GOOD[lang]:
                break
        score, y, heard, sr = best
        if y is None:
            print(f"{lang} no takes | {seg['show'][:60]}", flush=True)
            continue
        sf.write(path, y, sr)
        Path(path).with_suffix(".json").write_text(json.dumps({"score": round(score, 3), "heard": heard},
                                                              ensure_ascii=False), encoding="utf-8")
        flag = "" if score >= 0.85 else "   <-- check"
        print(f"{lang} {score:.2f} | {seg['show'][:60]} | heard: {heard[:60]}{flag}", flush=True)


def assemble(keys: list[str]) -> None:
    """Join each line's voiced segments with pauses; write the line's .wav and its timings."""
    import numpy as np
    import soundfile as sf

    for k in keys:
        parts, pieces, t, sr = segments(LINES[k], k), [], 0.0, None
        for i, seg in enumerate(parts):
            y, sr = sf.read(_seg_path(seg), dtype="float32")
            info = json.loads(_seg_path(seg).with_suffix(".json").read_text(encoding="utf-8"))
            if i:
                gap = AR_GAP if "ar" in (seg["lang"], parts[i - 1]["lang"]) else SLOW_GAP if k in SLOW else GAP
                pieces.append(np.zeros(int(gap * sr), np.float32))
                t += gap
            seg.update(start=round(t, 3), end=round(t + len(y) / sr, 3), **info)
            pieces.append(y)
            t += len(y) / sr
        path = CACHE / f"{k}-{_tag(k)}.wav"
        sf.write(path, np.concatenate(pieces), sr)
        path.with_suffix(".json").write_text(json.dumps({"duration": round(t, 3), "segments": parts},
                                                        ensure_ascii=False, indent=1), encoding="utf-8")


def voice_all(keys: list[str] | None = None) -> None:
    """Each language, and the voicing and the choosing of takes, in a process of its own (the TTS
    model and Whisper don't fit in the machine's memory together)."""
    import subprocess
    import sys

    keys = keys or list(LINES)
    for lang in ("en", "ar"):
        for step in ("--takes", "--pick"):
            subprocess.run([sys.executable, __file__, step, lang, *keys], check=True)
    assemble(keys)


def _resample(y, sr: int):
    import numpy as np
    from scipy.signal import resample_poly

    g = np.gcd(sr, 16000)
    return resample_poly(y, 16000 // g, sr // g).astype(np.float32)


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["--takes"]:
        take_pass(sys.argv[2], sys.argv[3:])
        sys.exit()
    if sys.argv[1:2] == ["--pick"]:
        pick_pass(sys.argv[2], sys.argv[3:])
        sys.exit()
    keys = sys.argv[1:] or list(LINES)
    voice_all(keys)
    print(f"{len(keys)} lines, {sum(clip(k)[1] for k in keys) / 60:.1f} min of speech")
