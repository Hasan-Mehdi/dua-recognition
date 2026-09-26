#!/usr/bin/env python
"""Can the lyrics decide the language? Noha identification across scripts.

scripts/noha_lid.py showed that acoustic language ID tops out at ~75-79%
after 10 s on sung nohas. This asks the next question: if the language is
wrong, or never decided, can alignment against the lyrics still find the
right noha?

Lyrics come from the YouTube descriptions of the data/noha/ recordings
(Urdu mostly in Roman Urdu, Arabic and Farsi in their own script, English in
English). To compare a transcript with lyrics in another script, both are
reduced to a *phonetic skeleton*: a Latin consonant string in which the
letters Urdu/Farsi pronounce alike are merged (س ص ث -> s, ز ذ ض ظ -> z, ...),
short vowels are dropped (Arabic script doesn't write them) and Roman
digraphs are folded (kh -> x, sh -> s ...). "Hussain", "حسین" and "حُسَين"
all become "hsn".

    python scripts/noha_match.py lyrics          # extract + check lyrics
    python scripts/noha_match.py transcribe --model small
    python scripts/noha_match.py eval --model small
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from dua_recognition.align import CorpusIndex, Word, semiglobal_end_costs  # noqa: E402
from dua_recognition.corpus import Dua, Segment  # noqa: E402

OUT = ROOT / "data" / "noha"
LANGS = ("ur", "ar", "fa", "en")

# -- phonetic skeleton ---------------------------------------------------------
# Arabic-script letters -> Latin class. Empty = dropped (vowel carriers, ain,
# hamza, aspiration marks). Urdu pronunciation merges the Arabic emphatics, and
# Roman Urdu spells them by sound, so they share a class.
_AR2LAT = {
    "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ط": "t", "ث": "s", "س": "s", "ص": "s", "ش": "s",
    "ج": "j", "چ": "c", "ح": "h", "ه": "h", "ہ": "h", "ۃ": "", "ة": "", "خ": "x", "د": "d", "ڈ": "d",
    "ذ": "z", "ز": "z", "ض": "z", "ظ": "z", "ژ": "z", "ر": "r", "ڑ": "r", "ع": "", "غ": "g", "گ": "g",
    "ف": "f", "ق": "k", "ک": "k", "ك": "k", "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "", "ؤ": "",
    "ی": "", "ي": "", "ى": "", "ئ": "", "ے": "", "ۓ": "", "ا": "", "أ": "", "إ": "", "آ": "", "ٱ": "",
    "ء": "", "ھ": "",
}
# Roman Urdu / English: digraphs first, then single letters. Vowels and the
# semivowels y/w (Arabic script writes them as the vowel carriers above) drop.
_LAT_DIGRAPHS = [("kh", "x"), ("gh", "g"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("dh", "d"),
                 ("ph", "f"), ("bh", "b"), ("jh", "j"), ("zh", "z"), ("ck", "k"), ("q", "k"),
                 ("c", "k"), ("v", ""), ("w", ""), ("y", "")]
_LAT_KEEP = set("bcdfghjklmnprstxz")


# Devanagari (a Hindi transcript of spoken Urdu) -> the same classes. Vowel
# signs, virama and independent vowels drop; a nukta turns the letter into its
# Perso-Arabic sound (ज़ = ز = z).
_DEV2LAT = {
    "क": "k", "ख": "k", "ग": "g", "घ": "g", "च": "c", "छ": "c", "ज": "j", "झ": "j", "ट": "t", "ठ": "t",
    "ड": "d", "ढ": "d", "ण": "n", "त": "t", "थ": "t", "द": "d", "ध": "d", "न": "n", "प": "p", "फ": "f",
    "ब": "b", "भ": "b", "म": "m", "य": "", "र": "r", "ल": "l", "व": "", "श": "s", "ष": "s", "स": "s",
    "ह": "h", "ड़": "r", "ढ़": "r", "क़": "k", "ख़": "x", "ग़": "g", "ज़": "z", "फ़": "f",
}
_DEV_NUKTA = {"क": "k", "ख": "x", "ग": "g", "ज": "z", "फ": "f", "ड": "r", "ढ": "r"}


def _key_devanagari(word: str) -> str:
    chars = unicodedata.normalize("NFD", word)
    out = []
    for i, ch in enumerate(chars):
        nukta = i + 1 < len(chars) and chars[i + 1] == "़"
        if nukta and ch in _DEV_NUKTA:
            out.append(_DEV_NUKTA[ch])
        elif ch in "ंँ":
            out.append("n")
        else:
            out.append(_DEV2LAT.get(ch, ""))
    return "".join(out)


def _key_word(word: str) -> str:
    if re.search(r"[ऀ-ॿ]", word):
        return re.sub(r"(.)\1+", r"\1", _key_devanagari(word))
    word = unicodedata.normalize("NFKD", word)
    word = "".join(ch for ch in word if not unicodedata.combining(ch))
    if re.search(r"[؀-ۿ]", word):
        # Word-final heh after a letter is the vowel -a/-e (سکینہ, فاطمه), not h.
        word = re.sub(r"(?<=.)[هہ]$", "", word)
        out = "".join(_AR2LAT.get(ch, "") for ch in word)
    else:
        w = word.lower()
        w = re.sub(r"[^a-z]", "", w)
        # "c" is k except in the digraph ch, which is handled first.
        for a, b in _LAT_DIGRAPHS[:10]:
            w = w.replace(a, b.upper())
        for a, b in _LAT_DIGRAPHS[10:]:
            w = w.replace(a, b)
        w = w.lower()
        # Final -h after a vowel is written, not said (Allah is the exception
        # that doesn't matter): "Fatimah" = "Fatima".
        w = re.sub(r"(?<=[aeiou])h$", "", w)
        out = "".join(ch for ch in w if ch in _LAT_KEEP)
    return re.sub(r"(.)\1+", r"\1", out)  # doubled letters: "Abbas" = "Abas" = عباس


def key(text: str) -> str:
    """Phonetic skeleton, words separated by spaces (empty words dropped)."""
    return " ".join(filter(None, (_key_word(w) for w in re.split(r"[\s\-_/.,،؛;:!?؟\"'()\[\]]+", text))))


_ALPHA = {c: i + 1 for i, c in enumerate("bcdfghjklmnprstxz")}


def encode_key(text: str) -> np.ndarray:
    return np.fromiter((_ALPHA[c] for c in key(text) if c in _ALPHA), dtype=np.int16)


class SkeletonIndex(CorpusIndex):
    """CorpusIndex over phonetic skeletons instead of normalized Arabic."""

    def __init__(self, duas: dict[str, Dua]):
        self.dua_ids = list(duas)
        self.duas = [duas[k] for k in self.dua_ids]
        self.words: list[Word] = []
        letters, letter_word = [], []
        self.dua_word_span = []
        for di, dua in enumerate(self.duas):
            first = len(self.words)
            for seg in dua.segments:
                for ti, token in enumerate(seg.arabic.split()):
                    codes = encode_key(token)
                    if not codes.size:
                        continue
                    letters.append(codes)
                    letter_word.append(np.full(codes.size, len(self.words), dtype=np.int32))
                    self.words.append(Word(di, seg.id, key(token), ti))
            self.dua_word_span.append((first, len(self.words)))
        self.letters = np.concatenate(letters)
        self.letter_word = np.concatenate(letter_word)
        self.word_dua = np.array([w.dua for w in self.words], dtype=np.int32)
        self.word_segment = np.array([w.segment for w in self.words], dtype=np.int32)
        self._word_ends = np.flatnonzero(np.r_[np.diff(self.letter_word) != 0, True])

    def end_costs(self, fragment: str):
        h = encode_key(fragment)
        if not h.size:
            return None
        return semiglobal_end_costs(h, self.letters)


# -- lyrics from video descriptions -----------------------------------------------
_CREDIT = re.compile(
    r"(https?://|www\.|@|#|©|\||^[^:]{1,30}:\s|poetry|poet|reciter|recited|composer|composition|studio|audio|video|"
    r"director|chorus|record|rights|subscribe|follow|facebook|instagram|twitter|youtube|tiktok|whatsapp|"
    r"production|produced|edit|special thanks|team|engineer|graphic|camera|shoot|dop|album|lyrics|"
    r"channel|download|official|copyright|management|organi[sz]|mix|master|kalam|nauha khwan|noha khwan|"
    r"كلمات|أداء|اداء|ألحان|الحان|إصدار|اصدار|تصوير|مونتاج|هندسة|استوديو|قناة|اشترك|حقوق|انتاج|إنتاج|"
    r"الشاعر|الرادود|شعر|مداح|متن|تنظیم|آهنگساز|میکس|کانال|سرود|شاعر|تدوین|تصویربرداری|"
    r"شاعری|کلام|نوحہ خواں|ریکارڈ)", re.I)
_HONORIFIC = re.compile(r"\(\s*(a\.?\s*s|s\.?\s*a|saww?|s\.?a\.?w\.?w?|pbuh|ajtf|atfs|ra)\.?\s*\)|[ؑؐؒ]", re.I)
_MARKS = re.compile(r"(\d{1,2}:\d{2}(:\d{2})?|\(x?\d+x?\)|x\d+|\[[^\]]*\]|[●♪♫♩♬♭─┤├*•~=_]+|\d+)")


def _script(line: str) -> str | None:
    ar = len(re.findall(r"[؀-ۿ]", line))
    lat = len(re.findall(r"[A-Za-z]", line))
    if ar >= 4 and ar >= 3 * lat:
        return "arab"
    if lat >= 4 and lat >= 3 * ar:
        return "latn"
    return None


def extract_lyrics(desc: str) -> tuple[str, list[str]] | None:
    """The longest run of lyric-looking lines in one script. Credits,
    hashtags, links and timing marks are skipped; blank lines don't break a run."""
    runs: dict[str, list[list[str]]] = {"arab": [[]], "latn": [[]]}
    for raw in desc.splitlines():
        raw = _HONORIFIC.sub(" ", raw)
        line = _MARKS.sub(" ", raw).strip(" .…،,-–—\t")
        if not re.search(r"[A-Za-z؀-ۿ]", line):
            continue  # blank, "//", "*****": stanza breaks, not the end of the lyrics
        s = _script(line)
        if s is None or _CREDIT.search(line) or len(line.split()) > 16:
            for r in runs.values():
                if r[-1]:
                    r.append([])
            continue
        runs[s][-1].append(re.sub(r"\s+", " ", line))
        other = runs["latn" if s == "arab" else "arab"]
        if other[-1]:
            other.append([])
    best = None
    for s, rs in runs.items():
        for r in rs:
            if len(r) >= 8 and (best is None or len(r) > len(best[1])):
                best = (s, r)
    return best


def load_items() -> list[dict]:
    items = []
    for lang in LANGS:
        for p in sorted((OUT / lang).glob("*.json")):
            if p.name.count(".") > 1:
                continue  # <id>.lyrics.json and other sidecars
            m = json.loads(p.read_text(encoding="utf-8"))
            if m.get("exclude"):
                continue
            m["audio_path"] = str(p.with_name(m["audio"]))
            items.append(m)
    return items


def cmd_lyrics() -> None:
    n = defaultdict(lambda: defaultdict(int))
    for m in load_items():
        got = extract_lyrics(m.get("description") or "")
        path = OUT / m["lang"] / f"{m['video_id']}.lyrics.json"
        if got is None:
            n[m["lang"]]["none"] += 1
            path.unlink(missing_ok=True)
            continue
        script, lines = got
        n[m["lang"]][script] += 1
        path.write_text(json.dumps({"script": script, "lines": lines}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{m['lang']} {m['video_id']} {script} {len(lines):3d} lines | {lines[0][:50]} / {lines[len(lines)//2][:50]}")
    for lang, c in n.items():
        print(lang, dict(c))


# -- more recordings (and distractor lyrics) from videos that carry lyrics ------------
LYRIC_QUERIES = {
    "ur": ["noha lyrics urdu", "nohay 2025 lyrics", "nohay 2024 lyrics", "noha with lyrics abbas",
           "noha lyrics sakina", "noha lyrics ali asghar", "nauha lyrics imam hussain", "noha lyrics bibi zainab",
           "manqabat lyrics urdu", "noha written urdu", "نوحہ لیرکس", "نوحہ کلام",
           "Mir Hasan Mir noha lyrics", "Shadman Raza noha lyrics", "Farhan Ali Waris lyrics",
           "Nadeem Sarwar noha lyrics", "Ali Shanawar lyrics", "Syed Raza Abbas Zaidi lyrics"],
    "ar": ["لطمية مع الكلمات", "كلمات لطمية حسينية", "قصيدة حسينية كلمات", "لطميات محرم مكتوبة",
           "باسم الكربلائي كلمات", "حسين فيصل قصيدة", "مسلم الوائلي كلمات", "علي بوحمد لطمية",
           "صالح الدرازي لطمية", "عبدالله القرمزي قصيدة", "محمد الحلفي لطمية", "كلمات القصيدة الرادود"],
    "fa": ["متن نوحه", "نوحه با متن", "متن مداحی محرم", "نوحه سینه زنی متن", "متن زمینه محرم",
           "متن نوحه حسین طاهری", "متن نوحه محمود کریمی", "متن نوحه بنی فاطمه", "متن شور محرم"],
    "en": ["english noha lyrics", "english latmiya lyrics", "english nauha lyrics", "english noha abbas lyrics",
           "english latmiya hussain", "english noha zainab", "english noha ali asghar", "english noha 2025"],
}


def _video_json(vid: str) -> dict | None:
    import subprocess

    ytdlp = str(Path(sys.executable).with_name("yt-dlp"))
    r = subprocess.run([ytdlp, "--skip-download", "--dump-json", "--no-warnings",
                        f"https://www.youtube.com/watch?v={vid}"], capture_output=True, text=True, encoding="utf-8")
    try:
        return json.loads(r.stdout)
    except ValueError:
        return None


def cmd_fetch_lyrics(target: int, per_query: int) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import noha_lid

    seen = {p.stem.split(".")[0] for p in OUT.rglob("*.json")}
    for lang, queries in LYRIC_QUERIES.items():
        have = sum(1 for p in (OUT / lang).glob("*.lyrics.json"))
        dis_dir = OUT / "distractors" / lang
        dis_dir.mkdir(parents=True, exist_ok=True)
        for q in queries:
            for v in noha_lid.search(q, per_query):
                if v["id"] in seen:
                    continue
                seen.add(v["id"])
                j = _video_json(v["id"])
                got = extract_lyrics((j or {}).get("description") or "")
                if not j or not got or len(got[1]) < 12:
                    continue
                script, lines = got
                title = f"{j.get('title', '')} {j.get('channel', '')}".lower()
                clash = any(w in title for o, ws in noha_lid.LANG_WORDS.items() if o != lang for w in ws)
                test = (have < target and not clash and noha_lid.acceptable(v, lang)
                        and noha_lid.download(v, lang, q))
                if test:
                    meta_path = OUT / lang / f"{v['id']}.json"
                    m = json.loads(meta_path.read_text(encoding="utf-8"))
                    m["description"] = j.get("description", "")
                    meta_path.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
                    (OUT / lang / f"{v['id']}.lyrics.json").write_text(
                        json.dumps({"script": script, "lines": lines}, ensure_ascii=False, indent=1), encoding="utf-8")
                    have += 1
                else:
                    (dis_dir / f"{v['id']}.lyrics.json").write_text(json.dumps(
                        {"script": script, "lines": lines, "title": j.get("title"), "query": q},
                        ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"  {lang} {'TEST' if test else 'dist'} {script} {len(lines):3d} {j.get('title', '')[:60]}",
                      flush=True)
        print(f"{lang}: {have} test recordings with lyrics, "
              f"{sum(1 for _ in dis_dir.glob('*.json'))} distractors", flush=True)


# -- transcribe every window in every language ---------------------------------------
STARTS = (0.25, 0.45, 0.65, 0.85)
HORIZON = 20
SR = 16000


def test_items() -> list[dict]:
    out = []
    for m in load_items():
        p = OUT / m["lang"] / f"{m['video_id']}.lyrics.json"
        if p.exists():
            m["lyrics"] = json.loads(p.read_text(encoding="utf-8"))
            out.append(m)
    return out


def _windows(audio: np.ndarray) -> list[tuple[float, list[np.ndarray]]]:
    """Cold starts at STARTS of the recording; window t ends t s after the start."""
    dur = audio.size / SR
    out = []
    for frac in STARTS:
        s = frac * dur
        out.append((s, [audio[int(max(s, s + t - 6) * SR): int((s + t) * SR)] for t in range(1, HORIZON + 1)]))
    return out


def transcribe_all(model_name: str, windows: list[np.ndarray], langs=LANGS, signals: bool = True) -> dict:
    """Greedy transcripts of each window forced to each language, unfiltered,
    with the signals a gate could use: Silero VAD on the tail (the du'a
    tracker's rule), Whisper's no-speech probability per language, and the
    restricted LID log-probs (as scripts/noha_lid.py). Encoder runs once."""
    import noha_lid
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.tokenizer import Tokenizer

    from dua_recognition.asr import _is_silent, load_model, speech_in_tail

    m = load_model(model_name)
    n_w = len(windows)
    out = {"texts": {l: [""] * n_w for l in langs}, "nsp": {l: [1.0] * n_w for l in langs},
           "vad": [False] * n_w, "lid": [[0.0] * len(noha_lid.LANGS)] * n_w}
    live = [i for i, w in enumerate(windows) if w.size and not _is_silent(w)]
    if not live:
        return out
    if signals:  # VAD and language ID; labelling training data needs neither
        for i in live:
            out["vad"][i] = bool(speech_in_tail(windows[i]))
        lp, _ = noha_lid.lid_logprobs(m, [windows[i] for i in live])
        for k, i in enumerate(live):
            out["lid"][i] = lp[k].round(3).tolist()
    feats = np.stack([pad_or_trim(m.feature_extractor(windows[i])) for i in live])
    for lo in range(0, len(live), 16):
        enc = m.encode(feats[lo: lo + 16])
        n = min(16, len(live) - lo)
        for lang in langs:
            tok = Tokenizer(m.hf_tokenizer, True, task="transcribe", language=lang)
            prompt = list(tok.sot_sequence) + [tok.no_timestamps]
            res = m.model.generate(enc, [prompt] * n, beam_size=1, max_length=len(prompt) + 96,
                                   return_no_speech_prob=True, suppress_blank=True, suppress_tokens=[-1],
                                   repetition_penalty=1.0, no_repeat_ngram_size=0)
            for k, r in enumerate(res):
                i = live[lo + k]
                out["texts"][lang][i] = tok.decode([t for t in r.sequences_ids[0] if t < tok.eot]).strip()
                out["nsp"][lang][i] = round(float(r.no_speech_prob), 3)
    return out


def _vocal_windows(video_id: str) -> list[tuple[float, list[np.ndarray]]] | None:
    """The same cold-start windows from Demucs vocals (scripts/noha_separate.py)."""
    path = OUT / "vocals" / f"{video_id}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return [(float(k), [z[f"s{k}"][int(max(0, t - 6) * SR): int(t * SR)] for t in range(1, HORIZON + 1)])
            for k in range(len(STARTS))]


def cmd_transcribe(model_name: str, vocals: bool = False) -> None:
    import noha_lid

    cache = OUT / "cache" / (model_name.replace("/", "_") + ("+vocals" if vocals else ""))
    cache.mkdir(parents=True, exist_ok=True)
    for m in test_items():
        path = cache / f"{m['video_id']}.json"
        if path.exists():
            continue
        wins_by_start = _vocal_windows(m["video_id"]) if vocals else             _windows(noha_lid.load_audio(Path(m["audio_path"])))
        if wins_by_start is None:
            continue
        trials = []
        for s, wins in wins_by_start:
            trials.append({"start": s, **transcribe_all(model_name, wins)})
        path.write_text(json.dumps({"video_id": m["video_id"], "lang": m["lang"], "trials": trials},
                                   ensure_ascii=False), encoding="utf-8")
        print(f"{m['lang']} {m['video_id']}: {trials[1]['texts'][m['lang']][9][:60]}", flush=True)


# -- identification -------------------------------------------------------------
def build_corpus(distractors: bool = True) -> dict[str, Dua]:
    duas = {}

    def add(did: str, name: str, lines: list[str]):
        segs = [Segment(i + 1, line) for i, line in enumerate(lines) if encode_key(line).size]
        if len(segs) >= 4:
            duas[did] = Dua(did, name, "", segs)

    def grams(lines):
        k = "".join(key(" ".join(lines)).split())
        return {k[i: i + 5] for i in range(len(k) - 4)}

    kept = []  # 5-gram sets of everything added so far

    def novel(lines):
        """Another upload of the same noha isn't a distractor: counting it as a
        wrong answer would punish finding the right noha."""
        g = grams(lines)
        if not g or any(len(g & h) / min(len(g), len(h)) > 0.3 for h in kept):
            return False
        kept.append(g)
        return True

    for m in test_items():
        if novel(m["lyrics"]["lines"]):
            add(m["video_id"], m["title"], m["lyrics"]["lines"])
    if distractors:
        for p in sorted((OUT / "distractors").glob("*/*.lyrics.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            if novel(d["lines"]):
                add("x-" + p.name.split(".")[0], d.get("title") or p.name, d["lines"])
    return duas


# Whisper's stock outputs for music and silence, in the four languages.
_JUNK = re.compile(r"(اشترك|ترجمة|نانسي|المترجم|موسيقى|موسیقی|زیرنویس|subscribe|thank(s| you) for watching|"
                   r"subtitles by|amara\.org|♪|music\)|\[music|تھینک یو|شکریہ)", re.I)


def _looped(text: str) -> bool:
    """A decoding loop compresses far better than lyrics (asr._looks_hallucinated's rule)."""
    import zlib

    raw = text.encode("utf-8")
    return len(raw) > 40 and len(raw) / len(zlib.compress(raw)) > 2.4


def _usable(text: str) -> bool:
    return bool(text) and not _JUNK.search(text)


def tracker_config(null_rate: float = 0.0, **kw):
    """The du'a defaults, pinned: the tracker is being developed alongside, and
    its "not in the corpus" state (null_rate) counts letters in the Arabic
    alphabet unless told otherwise, so it is off unless asked for."""
    from dataclasses import fields, replace

    from dua_recognition.tracker import TrackerConfig

    cfg = TrackerConfig()
    if "null_rate" in {f.name for f in fields(cfg)}:
        cfg = replace(cfg, null_rate=null_rate)
    return replace(cfg, **kw)


def _update(tracker, costs, text: str):
    """tracker.update_costs with the transcript's length in skeleton letters
    (for the null state, when the tracker has one)."""
    try:
        return tracker.update_costs(costs, 1.0, n_letters=encode_key(text).size if text else 0)
    except TypeError:  # a tracker without the null state
        return tracker.update_costs(costs, 1.0)


def cmd_eval(model_name: str, distractors: bool, gate: str, checkpoints=(3, 5, 10, 20), extra: str | None = None,
             only: list[str] | None = None, null_rate: float = 0.0) -> None:
    from dua_recognition.tracker import Tracker

    duas = build_corpus(distractors)
    ix = SkeletonIndex(duas)
    cache = OUT / "cache" / model_name.replace("/", "_")
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cache.glob("*.json"))]
    runs = [r for r in runs if r["video_id"] in duas]
    if extra:  # a second model's transcripts, offered to best-of as one more candidate
        ecache = OUT / "cache" / extra.replace("/", "_")
        other = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in ecache.glob("*.json")}
        runs = [r for r in runs if r["video_id"] in other]
        for r in runs:
            for trial, o in zip(r["trials"], other[r["video_id"]]["trials"]):
                trial["extra"] = o["texts"]["ur"] if "auto" not in o else o["auto"]
    print(f"corpus: {len(duas)} lyrics ({ix.n_words} words); model {model_name}; gate {gate}; null {null_rate}; "
          f"{len(runs)} recordings x {len(STARTS)} cold starts")
    tracker = Tracker(ix, tracker_config(null_rate))
    memo: dict[str, tuple] = {}

    def costs(text):
        """(word costs, per-letter fit) of a transcript, memoized."""
        if text not in memo:
            c = ix.word_costs(text)
            memo[text] = (c, float(c.min()) / max(encode_key(text).size, 1) if c is not None else np.inf)
        return memo[text]

    def open_(trial, t, lang):
        """Whether window t's transcript in `lang` counts as evidence. `gate`
        is "+"-joined: vad, nsp<th>, loop (compression-ratio filter), fit<th>
        (best per-letter alignment cost anywhere in the corpus must be <= th:
        a transcript that fits no lyric is noise, not a clue)."""
        text = trial["texts"][lang][t]
        if not _usable(text):
            return False
        for g in gate.split("+"):
            if g == "vad" and not trial["vad"][t]:
                return False
            if g.startswith("nsp") and trial["nsp"][lang][t] >= float(g[3:] or 0.6):
                return False
            if g == "loop" and _looped(text):
                return False
            if g.startswith("fit") and costs(text)[1] > float(g[3:]):
                return False
        return True

    def best_of(trial, t, langs):
        cands = [trial["texts"][l][t] for l in langs if open_(trial, t, l)]
        return min(cands, key=lambda x: costs(x)[1]) if cands else ""

    def strategy(name, trial, true_lang):
        lid = np.array(trial["lid"])
        heard = []
        for t in range(HORIZON):
            if any(open_(trial, t, l) for l in LANGS):
                heard.append(t)
            order = np.argsort(-lid[heard].sum(0)) if heard else np.arange(len(LANGS))
            if name == "oracle":
                yield trial["texts"][true_lang][t] if open_(trial, t, true_lang) else ""
            elif name.startswith("forced-"):
                yield trial["texts"][name[7:]][t] if open_(trial, t, name[7:]) else ""
            elif name == "lid":
                lang = LANGS[order[0]]
                yield trial["texts"][lang][t] if open_(trial, t, lang) else ""
            elif name == "best-of-2":
                yield best_of(trial, t, [LANGS[k] for k in order[:2]])
            elif name == "best-of-4":
                yield best_of(trial, t, LANGS)
            elif name == "best-of-2+extra":
                cands = [trial["texts"][LANGS[k]][t] for k in order[:2] if open_(trial, t, LANGS[k])]
                e = trial["extra"][t]
                if _usable(e) and not ("loop" in gate and _looped(e)):
                    cands.append(e)
                yield min(cands, key=lambda x: costs(x)[1]) if cands else ""
            elif name == "auto":  # the model's own language choice (Qwen3-ASR)
                text = trial["auto"][t]
                yield text if _usable(text) else ""

    names = ["oracle", "lid", "best-of-2", "best-of-4", "wrong"] + (["auto"] if "auto" in runs[0]["trials"][0] else [])         + (["best-of-2+extra"] if extra else [])
    names = [n for n in names if not only or n in only]
    res = defaultdict(lambda: defaultdict(list))
    for r in runs:
        true = r["video_id"]
        for trial in r["trials"]:
            for name in names:
                subs = [f"forced-{l}" for l in LANGS if l != r["lang"]] if name == "wrong" else [name]
                for sname in subs:
                    tracker.reset()
                    shown = []
                    for text in strategy(sname, trial, r["lang"]):
                        pos = _update(tracker, costs(text)[0] if text else None, text)
                        shown.append(pos.dua)
                    res[name][r["lang"]].append([shown[t - 1] == true for t in checkpoints]
                                                + [shown[t - 1] not in (None, true) for t in checkpoints])
    k = len(checkpoints)
    print("noha identified after t s (wrong noha shown)")
    print("strategy     lang   n    " + "  ".join(f"@{t:>2}s      " for t in checkpoints))
    summary = {}
    for name in names:
        for lang in (*LANGS, "all"):
            rows = [x for l, xs in res[name].items() if lang in ("all", l) for x in xs]
            if not rows:
                continue
            a = np.mean(rows, axis=0)
            summary[f"{name}/{lang}"] = a.round(4).tolist()
            print(f"{name:12} {lang:4} {len(rows):4}  " + "  ".join(f"{a[i]:4.0%} ({a[k + i]:3.0%})" for i in range(k)))
        print()
    return summary


def cmd_tune(model_name: str, gate: str) -> None:
    """Grid-search the tracker for nohas (best-of-2 front end), 2-fold by channel
    so each half's settings are scored on reciters they weren't chosen on."""
    import itertools
    from dataclasses import replace as dc_replace

    from dua_recognition.tracker import Tracker, TrackerConfig

    duas = build_corpus(True)
    ix = SkeletonIndex(duas)
    cache = OUT / "cache" / model_name.replace("/", "_")
    meta = {m["video_id"]: m for m in test_items()}
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cache.glob("*.json"))]
    runs = [r for r in runs if r["video_id"] in duas]
    memo = {}

    def fit(text):
        if text not in memo:
            c = ix.word_costs(text)
            memo[text] = (c, float(c.min()) / max(encode_key(text).size, 1) if c is not None else np.inf)
        return memo[text]

    def ok(trial, t, lang):
        text = trial["texts"][lang][t]
        return _usable(text) and not ("loop" in gate and _looped(text))

    # The best-of-2 transcript sequence doesn't depend on the tracker: precompute.
    seqs = []
    for r in runs:
        for trial in r["trials"]:
            lid, heard, seq = np.array(trial["lid"]), [], []
            for t in range(HORIZON):
                if any(ok(trial, t, l) for l in LANGS):
                    heard.append(t)
                order = np.argsort(-lid[heard].sum(0)) if heard else np.arange(4)
                cands = [trial["texts"][LANGS[k]][t] for k in order[:2] if ok(trial, t, LANGS[k])]
                seq.append(fit(min(cands, key=lambda x: fit(x)[1]))[0] if cands else None)
            seqs.append((r["video_id"], seq))

    grid = {"kappa_search": [0.6, 1.2, 2.0, 3.0], "min_dua_confidence": [0.5, 0.7, 0.85],
            "kappa": [0.15, 0.3], "start_weight": [0.0, 0.3]}
    cps = (5, 10, 20)
    table = {}
    for vals in itertools.product(*grid.values()):
        cfg = tracker_config(**dict(zip(grid, vals)))
        tr = Tracker(ix, cfg)
        per = {}
        for vid, seq in seqs:
            tr.reset()
            shown = [tr.update_costs(c, 1.0).dua for c in seq]
            per.setdefault(vid, []).append([shown[t - 1] == vid for t in cps] + [shown[t - 1] not in (None, vid) for t in cps])
        table[vals] = per
        print(dict(zip(grid, vals)), " ".join(f"{x:.0%}" for x in np.mean([x for v in per.values() for x in v], 0)),
              flush=True)

    def score(per, vids):
        a = np.mean([x for v in vids for x in per[v]], 0)
        return a[:3].mean() - 3 * a[3:].mean(), a

    chans = sorted({meta[v]["channel"] for v in {vid for vid, _ in seqs}})
    rng = np.random.default_rng(0)
    rng.shuffle(chans)
    folds = [{v for v in meta if meta[v]["channel"] in set(chans[i::2]) and v in duas} for i in range(2)]
    base = tuple(getattr(TrackerConfig(), k) for k in grid)
    print(f"\ndefault {dict(zip(grid, base))}: " + " ".join(f"{x:.0%}" for x in score(table[base], set.union(*folds))[1]))
    held = []
    for i in range(2):
        best = max(table, key=lambda v: score(table[v], folds[i])[0])
        s_, a = score(table[best], folds[1 - i])
        held.append(np.array([a * len(folds[1 - i])]))
        print(f"fit on fold {i}: {dict(zip(grid, best))} -> held-out " + " ".join(f"{x:.0%}" for x in a))
    tot = sum(len(f) for f in folds)
    print("held-out, both folds: " + " ".join(f"{x:.0%}" for x in (sum(h[0] for h in held) / tot)))
    print("(columns: found @" + ",".join(map(str, cps)) + " s, then wrong noha shown @ the same)")


class NgramRetriever:
    """First stage for a large corpus: TF-IDF over skeleton letter n-grams
    (spaces dropped, like the aligner). The HMM then only has to align
    against the top-K lyrics, whatever the corpus size."""

    def __init__(self, duas: dict[str, Dua], n: int = 4):
        from collections import Counter

        self.n, self.ids = n, list(duas)
        self.tf = []
        df = Counter()
        for d in duas.values():
            c = Counter(self._grams(" ".join(s.arabic for s in d.segments)))
            self.tf.append(c)
            df.update(c.keys())
        self.idf = {g: np.log(len(duas) / k) for g, k in df.items()}

    def _grams(self, text: str) -> list[str]:
        k = "".join(key(text).split())
        return [k[i: i + self.n] for i in range(len(k) - self.n + 1)]

    def rank(self, text: str) -> list[str]:
        q = set(self._grams(text))
        scores = [sum(self.idf.get(g, 0.0) for g in q if g in tf) for tf in self.tf]
        return [self.ids[i] for i in np.argsort(-np.array(scores))]


def cmd_retrieval(model_name: str, gate: str, ks=(1, 5, 20, 50), cps=(5, 10, 20)) -> None:
    """Recall@K of the n-gram first stage, fed the best-of-2 transcripts heard so far."""
    duas = build_corpus(True)
    ix = SkeletonIndex(duas)
    ret = NgramRetriever(duas)
    cache = OUT / "cache" / model_name.replace("/", "_")
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cache.glob("*.json"))]
    runs = [r for r in runs if r["video_id"] in duas]
    memo = {}

    def fit(text):
        if text not in memo:
            c = ix.end_costs(text)
            memo[text] = float(c.min()) / max(encode_key(text).size, 1) if c is not None else np.inf
        return memo[text]

    hits = defaultdict(list)
    for r in runs:
        for trial in r["trials"]:
            lid, heard, seq = np.array(trial["lid"]), [], []
            for t in range(HORIZON):
                ok = [l for l in LANGS if _usable(trial["texts"][l][t]) and not _looped(trial["texts"][l][t])]
                if ok:
                    heard.append(t)
                order = np.argsort(-lid[heard].sum(0)) if heard else np.arange(4)
                cands = [trial["texts"][LANGS[k]][t] for k in order[:2] if LANGS[k] in ok]
                seq.append(min(cands, key=fit) if cands else "")
                if t + 1 in cps:
                    # Overlapping windows repeat words; the n-gram *set* doesn't mind.
                    ranked = ret.rank(" ".join(seq))
                    pos = ranked.index(r["video_id"])
                    hits[t + 1].append([pos < k for k in ks])
    print(f"n-gram retrieval over {len(duas)} lyrics, {model_name} best-of-2 transcripts so far")
    print("after   " + "  ".join(f"top-{k:<3}" for k in ks))
    for t in cps:
        print(f"{t:>3} s   " + "  ".join(f"{x:6.0%}" for x in np.mean(hits[t], 0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lyrics")
    f = sub.add_parser("fetch-lyrics")
    f.add_argument("--target", type=int, default=15, help="test recordings with lyrics per language")
    f.add_argument("--per-query", type=int, default=25)
    t = sub.add_parser("transcribe")
    t.add_argument("--model", default="small")
    t.add_argument("--vocals", action="store_true", help="use Demucs vocals (scripts/noha_separate.py)")
    e = sub.add_parser("eval")
    e.add_argument("--model", default="small")
    e.add_argument("--no-distractors", action="store_true")
    e.add_argument("--gate", default="loop", help="none | vad | nsp<threshold>, e.g. nsp0.6")
    e.add_argument("--extra", help="second model's cache, offered to best-of-2 as another candidate")
    e.add_argument("--only", nargs="*", help="strategies to run")
    e.add_argument("--null", type=float, default=0.0, help="tracker null_rate (skeleton edits/letter); 0 = off")
    sub.add_parser("retrieval").add_argument("--model", default="small")
    u = sub.add_parser("tune")
    u.add_argument("--model", default="small")
    u.add_argument("--gate", default="loop")
    args = ap.parse_args()
    if args.cmd == "retrieval":
        return cmd_retrieval(args.model, "loop")
    if args.cmd == "tune":
        return cmd_tune(args.model, args.gate)
    if args.cmd == "transcribe":
        cmd_transcribe(args.model, args.vocals)
    elif args.cmd == "eval":
        cmd_eval(args.model, not args.no_distractors, args.gate, extra=args.extra, only=args.only, null_rate=args.null)
    elif args.cmd == "lyrics":
        cmd_lyrics()
    elif args.cmd == "fetch-lyrics":
        cmd_fetch_lyrics(args.target, args.per_query)


if __name__ == "__main__":
    main()
