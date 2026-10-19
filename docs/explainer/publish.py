#!/usr/bin/env python
"""The six rendered parts (scenes.py), joined into docs/explainer.mp4 with their sound and subtitles.

Each part logged where its narration starts, which stretches of the reading play under it, and where a shot
of the real page goes (data/cache/media/explainer10/log/P*.json). Here:
- the parts are joined end to end;
- the narration (voice.py) is laid at its times and levelled (speech at about -19 dBFS, peaks held under
  -2 dBFS by a look-ahead limiter);
- the reading (reading.py; Hussein Ghareeb's recording from DuaPlayer) is laid in where a part asked for it,
  a little under the narrator's level on its own and 14 dB lower while the narrator speaks;
- the film of the real page (docs/demo/film.mjs, data/cache/media/explainer10/phone.mp4) is laid over the
  picture at each shot, fading in and out;
- subtitles from the narration's sentence timings (numbers as digits, clauses packed into two balanced lines,
  Arabic phrases in italics) are burned in with libass in Latin Modern, and written as an .srt beside the video.
  Said during a shot of the app, they sit at the left, clear of the phone.

    python docs/explainer/publish.py [--quality 1440p60 | draft] [--out docs/explainer.mp4] [--no-subs]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import voice  # noqa: E402

CACHE = ROOT / "data" / "cache" / "media" / "explainer10"
READING = CACHE / "reading.wav"
FILM = CACHE / "phone.mp4"
PARTS = ["P1", "P2", "P3", "P4", "P5", "P6"]
SR = 48000

WIDTH = 44  # characters per subtitle line
SIDE_WIDTH = 26  # beside the phone, while a shot of the app is up
ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Narration,LM Roman 10,50,&H00F2F2F2,&H00F2F2F2,&H00000000,&H64000000,0,0,0,0,100,100,0.3,0,1,2.4,0,2,160,160,36,1
Style: Side,LM Roman 10,44,&H00F2F2F2,&H00F2F2F2,&H00000000,&H64000000,0,0,0,0,100,100,0.3,0,1,2.4,0,4,80,1240,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def cues_of(text: str, width: int = WIDTH) -> list[str]:
    """A sentence as subtitle cues of at most two lines: clauses (split after , ; :) are packed into cues; a
    clause too long for one cue is split between words into pieces of about equal length."""
    packed, cur = [], ""
    for clause in re.split(r"(?<=[,;:])\s+", text):
        cand = f"{cur} {clause}".strip()
        if cur and len(cand) > 2 * width:
            packed.append(cur)
            cur = clause
        else:
            cur = cand
    packed.append(cur)
    out = []
    for c in packed:
        k = -(-len(c) // (2 * width))
        while k > 1:
            cut = min((i for i, ch in enumerate(c) if ch == " "), key=lambda i: abs(i - len(c) / k))
            out.append(c[:cut])
            c = c[cut + 1:]
            k -= 1
        out.append(c)
    return [wrap(c, width) for c in out if c]


def wrap(c: str, width: int) -> str:
    if len(c) <= width:
        return c
    spaces = [i for i, ch in enumerate(c) if ch == " "]
    keep_number = [i for i in spaces if not c[i - 1].isdigit()] or spaces  # "10 seconds" stays on one line
    i = min(keep_number, key=lambda i: abs(i - len(c) / 2))
    return c[:i] + "\n" + c[i + 1:]


def side_wrap(c: str, width: int = SIDE_WIDTH) -> str:
    lines = [""]
    for w in c.replace("\n", " ").split():
        if lines[-1] and len(lines[-1]) + 1 + len(w) > width:
            lines.append(w)
        else:
            lines[-1] = f"{lines[-1]} {w}".strip()
    return "\n".join(lines)


def subtitles(said: list[dict]) -> list[tuple[float, float, str, str]]:
    """(start, end, text, lang) cues; a sentence's time is shared among its cues by length, and each cue stays up
    a little past its words (never into the next cue)."""
    cues = []
    for v in said:
        parts = cues_of(voice.caption(v["text"]))
        n = sum(len(p) for p in parts)
        t = v["start"]
        for p in parts:
            d = (v["end"] - v["start"]) * len(p) / n
            cues.append((t, t + d, p, v["lang"]))
            t += d
    return [(a, min(b + 0.6, cues[i + 1][0]) if i + 1 < len(cues) else b + 0.6, c, lang)
            for i, (a, b, c, lang) in enumerate(cues)]


def write_subtitles(cues, srt: Path, ass: Path, shots=()) -> None:
    def beside(a: float, b: float) -> bool:
        return any(s["start"] <= (a + b) / 2 <= s["start"] + s["dur"] for s in shots)

    def srt_t(t):
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    def ass_t(t):
        cs = int(round(t * 100))
        return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"

    srt.write_text("\n".join(f"{i}\n{srt_t(a)} --> {srt_t(b)}\n{c}\n" for i, (a, b, c, _) in enumerate(cues, 1)),
                   encoding="utf-8")
    events = []
    for a, b, c, lang in cues:
        side = beside(a, b)
        text = (side_wrap(c) if side else c).replace("\n", "\\N")
        events.append(f"Dialogue: 0,{ass_t(a)},{ass_t(b)},{'Side' if side else 'Narration'},,0,0,0,,"
                      f"{'{\\i1}' if lang == 'ar' else ''}{text}")
    ass.write_text(ASS_HEAD + "\n".join(events) + "\n", encoding="utf-8")


def level(x: np.ndarray, rms_db: float = -19.0, peak_db: float = -2.0) -> np.ndarray:
    speech = np.abs(x) > 1e-3
    return limit(x * 10 ** (rms_db / 20) / np.sqrt(np.mean(x[speech] ** 2)), peak_db)


def limit(y: np.ndarray, peak_db: float = -2.0) -> np.ndarray:
    need = np.minimum(1.0, 10 ** (peak_db / 20) / np.maximum(np.abs(y), 1e-9))
    w = int(0.006 * SR)
    gain = np.minimum(uniform_filter1d(minimum_filter1d(need, 2 * w + 1), w), need)
    return (y * gain).astype(np.float32)


def load(path: Path) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    g = np.gcd(sr, SR)
    return resample_poly(y, SR // g, sr // g).astype(np.float32)


def recitation(spans: list[dict], n: int, rms_db: float = -22.0, fade: float = 0.25) -> np.ndarray:
    """The reading as a track n samples long: each span's seconds at its video second, with short fades."""
    y = load(READING)
    y *= 10 ** (rms_db / 20) / np.sqrt(np.mean(y[np.abs(y) > 1e-3] ** 2))
    out = np.zeros(n, np.float32)
    for s in spans:
        piece = y[int(s["from"] * SR): int(s["to"] * SR)].copy() * 10 ** (s.get("gain_db", 0.0) / 20)
        k = min(len(piece) // 2, int(fade * SR))
        ramp = np.linspace(0, 1, k, dtype=np.float32)
        piece[:k] *= ramp
        piece[len(piece) - k:] *= ramp[::-1]
        at = int(s["at"] * SR)
        m = max(0, min(len(piece), n - at))
        out[at: at + m] += piece[:m]
    return out


def ducking(said: list[dict], n: int, duck_db: float = -14.0, bridge: float = 1.0) -> np.ndarray:
    """A gain for the reading: duck_db while the narrator speaks (pauses under `bridge` s count as speaking)."""
    spans = sorted((v["start"] - 0.2, v["end"] + 0.15) for v in said)
    on = np.zeros(n, np.float32)
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] < bridge:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for a, b in merged:
        on[max(0, int(a * SR)): min(n, int(b * SR))] = 1.0
    on = uniform_filter1d(on, int(0.3 * SR))
    return (10 ** (duck_db * on / 20)).astype(np.float32)


def overlays(shots: list[dict], size: str, fade: float = 0.5) -> list[str]:
    """Filter chains laying shot i (input 1 + i: the film, seeked to the shot) over input 0's picture: [v0] .. [vN-1].
    Each shot reads its own slice of the film, so nothing waits in memory for a later shot."""
    chains = []
    for i, s in enumerate(shots):
        a, d = s["start"], s["dur"]
        chains.append(f"[{1 + i}:v]setpts=PTS-STARTPTS+{a}/TB,scale={size},format=yuva420p,"
                      f"fade=t=in:st={a}:d={fade}:alpha=1,fade=t=out:st={a + d - fade:.3f}:d={fade}:alpha=1[s{i}]")
        chains.append(f"[{'0:v' if i == 0 else f'v{i - 1}'}][s{i}]overlay=eof_action=pass:"
                      f"enable='between(t,{a},{a + d:.3f})'[v{i}]")
    return chains


def probe(path: Path, what: str) -> str:
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", what, "-of",
                           "csv=p=0:s=x", str(path)], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="final", help="final (render/) or draft (draft/)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "explainer.mp4"))
    ap.add_argument("--no-subs", action="store_true")
    args = ap.parse_args()
    src = CACHE / ("render" if args.quality == "final" else "draft")
    out = Path(args.out)

    # the parts end to end, and where each starts
    logs, offset, said, spans, shots = [], 0.0, [], [], []
    durs = [float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                  str(src / f"{p}.mp4")], capture_output=True, text=True, check=True).stdout)
            for p in PARTS]
    for p, dur in zip(PARTS, durs):
        log = json.loads((CACHE / "log" / f"{p}.json").read_text(encoding="utf-8"))
        for v in log["voice"]:
            for seg in v["segs"]:
                said.append({"start": offset + v["at"] + seg["start"], "end": offset + v["at"] + seg["end"],
                             "text": seg["show"], "lang": seg["lang"], "key": v["key"], "line_at": offset + v["at"]})
        spans += [dict(r, at=offset + r["at"]) for r in log["recite"]]
        shots += [dict(s, start=offset + s["start"]) for s in log["shots"]]
        logs.append((p, offset, log))
        offset += dur
    total = offset
    n = int(total * SR) + SR

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "parts.txt").write_text("".join(f"file '{(src / f'{p}.mp4').as_posix()}'\n" for p in PARTS))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", "parts.txt", "-c",
                        "copy", "video.mp4"], check=True, cwd=tmp)

        # the narration, line by line at its times
        voice_track = np.zeros(n, np.float32)
        for _, off, log in logs:
            for v in log["voice"]:
                path, _, _ = voice.clip(v["key"])
                y = load(path)
                at = int((off + v["at"]) * SR)
                m = max(0, min(len(y), n - at))
                voice_track[at: at + m] += y[:m]
        mix = level(voice_track)
        mix = limit(mix + recitation(spans, n) * ducking(said, n))
        sf.write(tmp / "sound.wav", mix[: int(total * SR)], SR, subtype="FLOAT")

        size = probe(tmp / "video.mp4", "stream=width,height").replace("x", ":")
        graph, last = (overlays(shots, size), f"[v{len(shots) - 1}]") if shots else ([], "[0:v]")
        if not args.no_subs:
            cues = subtitles(said)
            write_subtitles(cues, out.with_suffix(".srt"), tmp / "subs.ass", shots)
            (tmp / "fonts").mkdir()
            for name in ("lmroman10-regular.otf", "lmroman10-italic.otf"):
                found = subprocess.run(["kpsewhich", name], capture_output=True, text=True, check=True).stdout.strip()
                shutil.copy(found, tmp / "fonts" / name)
            graph.append(f"{last}ass=subs.ass:fontsdir=fonts[subbed]")
            last = "[subbed]"
        graph.append(f"{last}format=yuv420p[out]")
        film = [x for s in shots for x in ("-ss", f"{s['from']:.3f}", "-t", f"{s['dur']:.3f}", "-i", str(FILM))]
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", "video.mp4", *film, "-i", "sound.wav",
                        "-filter_complex", ";".join(graph), "-map", "[out]", "-map", f"{1 + len(shots)}:a",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-tune", "animation", "-pix_fmt", "yuv420p",
                        "-threads", "6", "-x264-params", "rc-lookahead=20",
                        "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", "out.mp4"],
                       check=True, cwd=tmp)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(tmp / "out.mp4", out)
    timeline = {"parts": [(p, round(o, 3)) for p, o, _ in logs], "duration": round(total, 3), "shots": shots,
                "recite": spans}
    (CACHE / "timeline.json").write_text(json.dumps(timeline, indent=1), encoding="utf-8")
    print(f"wrote {out} ({total / 60:.2f} min)")


if __name__ == "__main__":
    main()
