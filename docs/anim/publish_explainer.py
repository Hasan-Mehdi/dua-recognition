#!/usr/bin/env python
"""Turn Manim's render of the explainer into docs/explainer.mp4, with subtitles.

The shots of the real app (media/explainer_shots.json: when, and from where in the film of it,
data/cache/media/explainer_phone_back.mp4) are laid over the render, fading in and out.

The voice is levelled (speech at about -19 dBFS, peaks held under -2 dBFS by a look-ahead
limiter). The subtitles come from the sentence timings the scene wrote
(media/explainer_voice.json): numbers as digits, cues broken at commas into at most two
balanced lines, Arabic phrases in italics. They are burned in with libass in Latin Modern (TeX's
face, so they match the video's own text; found with kpsewhich), and also written as an .srt
next to the video for players and sites that take a caption file.

    python docs/anim/publish_explainer.py [--media MEDIA_DIR] [--out docs/explainer.mp4] [--no-subs] [--phone FILM]
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

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from narration import caption  # noqa: E402

WIDTH = 44  # characters per subtitle line
SIDE_WIDTH = 26  # beside the phone, while a shot of the app is up
ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Narration,LM Roman 10,52,&H00F2F2F2,&H00F2F2F2,&H00000000,&H64000000,0,0,0,0,100,100,0.3,0,1,2.4,0,2,160,160,42,1
Style: Side,LM Roman 10,44,&H00F2F2F2,&H00F2F2F2,&H00000000,&H64000000,0,0,0,0,100,100,0.3,0,1,2.4,0,4,80,1240,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def cues_of(text: str, width: int = WIDTH) -> list[str]:
    """A sentence as subtitle cues of at most two lines: clauses (split after , ; :) are packed
    into cues; a clause too long for one cue is split between words; each cue is wrapped into
    two lines of similar length."""
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
        # Into k pieces of about equal length, so no one-word cue is left over at the end.
        k = -(-len(c) // (2 * width))
        while k > 1:
            cut = min((i for i, ch in enumerate(c) if ch == " "), key=lambda i: abs(i - len(c) / k))
            out.append(c[:cut])
            c = c[cut + 1 :]
            k -= 1
        out.append(c)
    return [wrap(c, width) for c in out if c]


def wrap(c: str, width: int) -> str:
    if len(c) <= width:
        return c
    spaces = [i for i, ch in enumerate(c) if ch == " "]
    keep_number = [i for i in spaces if not c[i - 1].isdigit()] or spaces  # "10 seconds" stays on one line
    i = min(keep_number, key=lambda i: abs(i - len(c) / 2))
    return c[:i] + "\n" + c[i + 1 :]


def subtitles(voice: list[dict]) -> list[tuple[float, float, str, str]]:
    """(start, end, text, lang) cues; a sentence's time is shared among its cues by length, and
    each cue stays up a little past its words (never into the next cue)."""
    cues = []
    for v in voice:
        parts = cues_of(caption(v["text"]))
        n = sum(len(p) for p in parts)
        t = v["start"]
        for p in parts:
            d = (v["end"] - v["start"]) * len(p) / n
            cues.append((t, t + d, p, v["lang"]))
            t += d
    return [(a, min(b + 0.6, cues[i + 1][0]) if i + 1 < len(cues) else b + 0.6, c, lang)
            for i, (a, b, c, lang) in enumerate(cues)]


def side_wrap(c: str, width: int = SIDE_WIDTH) -> str:
    """A cue for beside the phone: lines of at most `width` characters, broken between words."""
    lines = [""]
    for w in c.split():
        if lines[-1] and len(lines[-1]) + 1 + len(w) > width:
            lines.append(w)
        else:
            lines[-1] = f"{lines[-1]} {w}".strip()
    return "\n".join(lines)


def write_subtitles(cues, srt: Path, ass: Path, shots=()) -> None:
    """Cues said while a shot of the app is up go beside the phone: its own text is at the bottom."""
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
    events = [f"Dialogue: 0,{ass_t(a)},{ass_t(b)},{'Side' if beside(a, b) else 'Narration'},,0,0,0,,"
              f"{'{\\i1}' if lang == 'ar' else ''}" + (side_wrap(c) if beside(a, b) else c).replace("\n", "\\N")
              for a, b, c, lang in cues]
    ass.write_text(ASS_HEAD + "\n".join(events) + "\n", encoding="utf-8")


def overlays(shots: list[dict], fade: float = 0.5) -> list[str]:
    """Filter-graph chains laying each shot of input 2 over input 0's picture: [v0] .. [vN-1]."""
    chains = ["[2:v]split=" + str(len(shots)) + "".join(f"[p{i}]" for i in range(len(shots)))]
    for i, s in enumerate(shots):
        a, d = s["start"], s["dur"]
        chains.append(f"[p{i}]trim=start={s['from']}:duration={d},setpts=PTS-STARTPTS+{a}/TB,format=yuva420p,"
                      f"fade=t=in:st={a}:d={fade}:alpha=1,fade=t=out:st={a + d - fade:.3f}:d={fade}:alpha=1[s{i}]")
        chains.append(f"[{'0:v' if i == 0 else f'v{i - 1}'}][s{i}]overlay=eof_action=pass:"
                      f"enable='between(t,{a},{a + d:.3f})'[v{i}]")
    return chains


def level(x: np.ndarray, sr: int, rms_db: float = -19.0, peak_db: float = -2.0) -> np.ndarray:
    speech = np.abs(x) > 1e-3
    y = x * 10 ** (rms_db / 20) / np.sqrt(np.mean(x[speech] ** 2))
    need = np.minimum(1.0, 10 ** (peak_db / 20) / np.maximum(np.abs(y), 1e-9))
    w = int(0.006 * sr)
    gain = np.minimum(uniform_filter1d(minimum_filter1d(need, 2 * w + 1), w), need)  # never above the ceiling
    return (y * gain).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", default="media", help="Manim's media directory")
    ap.add_argument("--out", default=str(ROOT / "docs" / "explainer.mp4"))
    ap.add_argument("--no-subs", action="store_true")
    ap.add_argument("--phone", default=str(ROOT / "data" / "cache" / "media" / "explainer_phone_back.mp4"),
                    help="film.mjs's film of the real app, for the shots")
    args = ap.parse_args()
    media = Path(args.media)
    video = media / "videos" / "explainer" / "1080p30" / "Explainer.mp4"
    out = Path(args.out)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "48000",
                        "-c:a", "pcm_f32le", str(tmp / "raw.wav")], check=True)
        x, sr = sf.read(tmp / "raw.wav", dtype="float32")
        sf.write(tmp / "voice.wav", level(x, sr), sr, subtype="FLOAT")
        shots_file = media / "explainer_shots.json"
        shots = json.loads(shots_file.read_text(encoding="utf-8")) if shots_file.exists() else []
        graph, last = (overlays(shots), f"[v{len(shots) - 1}]") if shots else ([], "[0:v]")
        if not args.no_subs:
            cues = subtitles(json.loads((media / "explainer_voice.json").read_text(encoding="utf-8")))
            write_subtitles(cues, out.with_suffix(".srt"), tmp / "subs.ass", shots)
            (tmp / "fonts").mkdir()
            for name in ("lmroman10-regular.otf", "lmroman10-italic.otf"):
                found = subprocess.run(["kpsewhich", name], capture_output=True, text=True, check=True).stdout.strip()
                shutil.copy(found, tmp / "fonts" / name)
            graph.append(f"{last}ass=subs.ass:fontsdir=fonts[subbed]")
            last = "[subbed]"
        graph.append(f"{last}format=yuv420p[out]")
        phone = ["-i", str(Path(args.phone).resolve())] if shots else []
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video.resolve()), "-i", "voice.wav", *phone,
                        "-filter_complex", ";".join(graph), "-map", "[out]", "-map", "1:a", "-c:v", "libx264",
                        "-preset", "slow", "-crf", "26", "-tune", "animation", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-af", "apad",
                        "-shortest", "-movflags", "+faststart", "out.mp4"], check=True, cwd=tmp)
        shutil.move(tmp / "out.mp4", out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
