"""The explainer's voice-over: what is said in each beat, its audio, and its subtitles.

The narrator speaks English with an Arabic accent: OmniVoice (k2-fsa, Apache-2.0) clones
narrator-iraqi.flac, a voice it designed itself speaking Iraqi Arabic (no real person's voice), into
English. Islamic words in the English are written in Arabic script (TTS_SPELLING), so the
same voice says them natively, and the Arabic phrases, marked in a line as
[[ar:ARABIC|transliteration]], are spoken in Arabic between sentences. Every sentence is voiced
up to a few times and the take Whisper transcribes closest to the text is kept, cut cleanly
after its last word, so a garbled take or a trailing hum never reaches the video.

Each line is cached by its text in data/cache/explainer_voice/ as a .wav and a .json listing
its sentences with their start and end, which the subtitles use. Editing a line re-voices
only that line.

    <venv with omnivoice>/python docs/anim/narration.py   # GPU

The takes are voiced first (OmniVoice alone, in its own process), then chosen (Whisper alone): the two
together pushed the machine's commit past its limit. With other jobs running, voicing from the saved
voice prompt on the GPU still fits; choosing needed EXPLAINER_PICK_DEVICE=cpu
EXPLAINER_PICK_MODEL=openai/whisper-small (2026-10-05, peak commit 39.8 GB).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("EXPLAINER_VOICE_CACHE", ROOT / "data" / "cache" / "explainer_voice"))
# The narrator: OmniVoice's own designed voice ("male, middle-aged, moderate pitch") reading NARRATOR_TEXT
# in Iraqi Arabic.
NARRATOR = Path(__file__).parent / "narrator-iraqi.flac"
NARRATOR_TEXT = "الناس يتعلمون الدعاء بالبيت وبالجامع، ويقرونه سوه بالليالي المباركة. وهذا التطبيق يسمع القارئ، ويتابعه كلمة كلمة."
SPEED = 0.92
ENGINE = f"omnivoice|{NARRATOR.stem}|speed{SPEED}"
GAP, AR_GAP = 0.3, 0.4  # seconds between sentences, and around an Arabic phrase
# The letter model and the decoder bring the most new ideas per second: said slower, with longer
# pauses between sentences.
SLOW = {"word2", "word3", "dec2", "dec5"}
SLOW_SPEED, SLOW_GAP = 0.84, 0.6
TAKES = {"en": 4, "ar": 8}  # tries per segment; Arabic phrases are short and less steady
GOOD = {"en": 0.93, "ar": 0.92}  # a take this close to the text is kept at once

LINES = {
    # -- what it's for: two questions, two listeners
    "open1": "At a gathering, one person usually recites the du'a aloud, while everyone else reads along, from a book "
             "or a phone. Many du'as are long, this one, Dua Tawassul, is a hundred and fifteen lines, and it's easy "
             "to lose your place.",
    "open2": "This app listens to the reciter and tries to keep your place for you. To do that, it has two questions to "
             "answer. Which du'a is this, out of five hundred and twenty-two du'as and ziyarat, the greetings recited "
             "at shrines? And where in it is he right now, down to the word?",
    "open3": "Two models listen at the same time, one for each question. Whisper, a speech recognizer, turns the last "
             "few seconds into text every second or two, and that text is looked for in every du'a. A much smaller "
             "letter model hears letters ten times a second, and follows him through the du'a Whisper found.",
    "open4": "Let's take one recording through both. It's Hussein Ghareeb, about five minutes into Dua Tawassul, at "
             "line thirty-eight. Neither model heard him in training.",

    # -- which du'a: Whisper
    "hear1": "Every second or so, the app takes the last six seconds of sound, called a window, and draws it as a "
             "spectrogram. Time runs left to right, pitch from low to high, and the brighter a spot, the louder.",
    "asr1": "Whisper turns that into text. The app's version is small enough to run on a phone. Out of the box, it "
            "barely understands recitation, where one word can be drawn out for seconds, so it was trained further, "
            "on over a hundred hours of recited du'as, and on synthetic voices of ordinary people.",
    "asr3": "Here's what it heard. [[ar:يَا وَجِيهْ عِنْدَ الله|Yā wajīh ‘inda-llāh.]]",
    "asr4": "What he actually recited was this. [[ar:يَا وَجِيهًا|Yā wajīhan.]] Close, but not exact, and only half "
            "the line, because he draws every word out. So nothing after this point takes the text as certain.",
    "match1": "The app compares the heard text with every du'a it knows. Vowel marks, spellings and spaces are "
              "simplified on both sides. Then it counts the fewest letters to change, add, or remove, to turn one "
              "into the other. Here it's one, a missing alif.",
    "match2": "It does this ending at every one of a hundred and forty thousand words, in about a hundredth of a "
              "second. Here are the costs across Dua Tawassul: the shorter the bar, the closer the fit.",
    "dua1": "Every word of every text gets a probability that he's there right now. Before anything is heard, it's "
            "spread over all the texts, with a little more on the ones people recite most. With each window, places "
            "that fit well become more likely, and places that fit badly much less.",
    "dua2": "Add up the probabilities inside each du'a. If you'd walked in at exactly this moment, after one window, "
            "Dua Tawassul would be ahead, at twenty-nine percent. The app names a du'a only at seventy percent, and "
            "until then, it offers its best guesses for you to tap.",
    "dua3": "A second later, it's at ninety-nine point eight percent, and the app names it.",
    "dua4": "The same probabilities also say roughly where he is in the du'a. That's where the second model takes "
            "over.",

    # -- where: the letter model and the decoder
    "word1": "Whisper alone can't keep up with the word. Its text comes only every second or two, with no timing "
             "inside it, and describes sound that's already a second old.",
    "word2": "So the letter model listens alongside. It's the listening half of the same Whisper, with one small "
             "layer added that maps what it hears onto the alphabet. Ten times a second, it takes the last two "
             "seconds of sound, and for every fiftieth of a second, gives a probability for each letter, or for no "
             "letter at all.",
    "word3": "Here it is as he carries on. [[ar:اِشْفَعْ لَنَا عِنْدَ|Ishfa‘ lanā ‘inda.]] Each row is one letter of "
             "these words, and time runs left to right. The brighter a square, the surer the model is that it heard "
             "that letter, at that moment.",
    "dec1": "Two seconds hold a word or two, and many du'as share words, so the letter model doesn't search for the "
            "du'a. Instead, a decoder follows him through the one Whisper found. It keeps a probability on every "
            "letter of the du'a, not just the line he's on, and updates it every fiftieth of a second.",
    "dec2": "Each update asks two things, much as you'd keep your place yourself. Where could he be now, given where "
            "he was? Reading on costs nothing. Starting the line again costs a little, going back a line or three "
            "costs more, skipping ahead more still, and anywhere else in the du'a the most. And which of those places "
            "explains the letters just heard?",
    "dec3": "Talk that isn't the du'a, and the salawat people say between lines, have places of their own, so they "
            "don't drag the highlight along. A pause is simply no letters, so the highlight waits on the last word "
            "said.",
    "dec4": "Here's what that looks like when a reader goes back. We cut this recording so that after line "
            "forty-three, he returns to line forty-one, as people reading along often do.",
    "dec5": "While he takes a breath, nothing new is heard, so the probability stays at the end of line forty-three. "
            "When he starts again, the first sounds fit several places, and for half a second, the highlight goes to "
            "the start of line forty-three. A few letters later, line forty-one explains them best, and the highlight "
            "follows him there, about a second after he began.",
    "dec6": "Whisper keeps listening as a check, every two seconds on a phone. If he moves to a different du'a, "
            "Whisper notices first, and the decoder follows once Whisper has held the new one for two seconds.",

    # -- all of it, and how well it works
    "sum1": "So, Whisper and the probabilities over every du'a find which du'a it is, and roughly where. The letter "
            "model and the decoder follow him through it, word by word. All of it runs inside the phone's browser, "
            "so the recitation never has to leave the phone.",
    "res1": "It was tested on voices kept out of training, reading the way people do: in flow, pausing, talking in "
            "between, going back, skipping ahead, in quiet rooms and echoing halls.",
    "res2": "It names the du'a within ten seconds ninety percent of the time. The highlight is on the reader's line "
            "eighty-seven percent of the time, and on the exact word two times in three. When a reader goes back or "
            "skips ahead, it follows within three seconds nearly three times in four.",
    "limits": "It still makes mistakes. When two du'as share a passage word for word, it can't tell them apart until "
              "they part ways. It does worse in echoing halls, and when a reader jumps around the du'a. And for a "
              "du'a it doesn't have, it too often shows one it does.",
    "end": "That's the whole path, from the microphone to the highlighted word.",
}

# Islamic words, written for the voice in Arabic script so it says them as an Arabic speaker would.
TTS_SPELLING = {
    "Dua Tawassul": "دعاء التوسل",
    "Tawassul": "التوسل",
    "du'a": "دعاء",
    "ziyarat": "زيارات",
    "Hussein Ghareeb": "حسين غريب",
    "salawat": "صلوات",
}
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


# Subtitles show numbers as digits; the voice says the words.
CAPTION_NUMBERS = [
    ("five hundred and twenty-two", "522"), ("a hundred and forty thousand", "140,000"),
    ("all hundred and fifteen", "all 115"), ("a hundred and fifteen", "115"), ("sixteen thousand", "16,000"),
    ("four hundred and eighty", "480"), ("thirty milliseconds", "30 milliseconds"),
    ("seventy-four million", "74 million"), ("a hundred hours", "100 hours"), ("thirty-eight", "38"),
    ("thirty-seven", "37"), ("Fourteen words", "14 words"), ("thirty-one", "31"), ("forty-five", "45"),
    ("eleven more", "11 more"), ("seventy-six percent", "76%"), ("seventy-nine percent", "79%"),
    ("sixty-nine percent", "69%"), ("one point two", "1.2"), ("zero point one five", "0.15"),
    ("zero point six two", "0.62"), ("zero point six", "0.6"), ("zero point five four", "0.54"),
    ("twenty-two percent", "22%"), ("forty-three percent", "43%"), ("under one percent", "under 1%"),
    ("eleven percent", "11%"), ("fifty-four percent", "54%"), ("a hundred percent", "100%"),
    ("a hundred and twenty times", "120 times"), ("fourteen times", "14 times"), ("twenty-nine percent", "29%"),
    ("ninety-nine point eight percent", "99.8%"), ("seventy percent", "70%"), ("ten seconds", "10 seconds"),
    ("ninety-three percent", "93%"), ("less than one percent", "less than 1%"), ("eighty-five percent", "85%"),
    ("within one line, ninety-six", "within one line, 96%"), ("forty-seven percent", "47%"),
    ("six and a half percent", "6.5%"), ("against eighty-nine", "against 89%"),
    ("forty-three", "43"), ("forty-one", "41"), ("ninety percent", "90%"), ("eighty-seven percent", "87%"),
]


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
    if not path.exists():
        raise FileNotFoundError(f"{path.name}: voice the narration first (python docs/anim/narration.py)")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return path, meta["duration"], meta["segments"]


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
