"""What every scene of the explainer shares: the palette, the type, the data, and a Scene that keeps each
beat in step with its line of narration and logs what the publish step lays under the picture (the voice,
the reading's own audio, the shots of the real app).

One colour per idea, the whole video through: gold is Whisper and the words it writes, teal the letter
model and its letters, blue probability, white the texts. Latin Modern (TeX's face) for English, Amiri for
Arabic.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import manimpango
import numpy as np
from manimlib import *  # noqa: F401,F403

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / "data" / "cache" / "media" / "explainer10"
IMG = CACHE / "img"
LOG = CACHE / "log"
sys.path.insert(0, str(HERE))
import voice  # noqa: E402

for f in (HERE / "fonts" / "Amiri-Regular.ttf",
          Path(os.environ.get("APPDATA", "")) / "TinyTeX/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
          Path(os.environ.get("APPDATA", "")) / "TinyTeX/texmf-dist/fonts/opentype/public/lm/lmroman10-italic.otf"):
    if f.exists():
        manimpango.register_font(str(f))

def _sharper_writer() -> None:
    """ManimGL pipes frames to x264 at its default quality; the publish step encodes again, so the parts are
    written nearly lossless (crf 8) and thin lines survive both passes."""
    import subprocess
    import types

    from manimlib.scene import scene_file_writer as sfw

    def popen(command, *a, **kw):
        if "rawvideo" in command:
            command = command[:-1] + ["-crf", "8", "-preset", "fast", command[-1]]
        return subprocess.Popen(command, *a, **kw)

    sfw.sp = types.SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT, call=subprocess.call)


_sharper_writer()

PACE = 1.25  # every fixed animation length, x this (the first cut moved too fast)


def T(seconds: float) -> float:
    return seconds * PACE


BG = "#101114"
GOLD = "#F2B84B"  # Whisper
TEAL = "#4FD1B5"  # the letter model
BLUE = "#58C4DD"  # probability
INK = "#ECEFF1"  # the texts
SOFT = "#9AA3AD"
DIM = "#4A5058"
FAINT = "#2A2E35"
WARN = "#F07C6C"  # the weak spots, used sparingly
SERIF = "Latin Modern Roman 10"
ARABIC = "Amiri"

# the film of the real page (film.mjs): the reading starts at film second FILM0
FILM0 = 5.8


def film_at(reading_s: float) -> float:
    return FILM0 + reading_s


_DATA = {}


def data() -> tuple[dict, dict]:
    if not _DATA:
        _DATA["z"] = dict(np.load(HERE / "data" / "reading.npz"))
        _DATA["m"] = json.loads((HERE / "data" / "reading.json").read_text(encoding="utf-8"))
    return _DATA["z"], _DATA["m"]


def en(s: str, size: float = 30, color=INK, **kw) -> Text:
    return Text(s, font=SERIF, font_size=size, fill_color=color, **kw)


def ar(s: str, size: float = 40, color=INK, **kw) -> Text:
    return Text(s, font=ARABIC, font_size=size, fill_color=color, **kw)


def spec(name: str, height: float = 2.0, **kw) -> ImageMobject:
    return ImageMobject(str(IMG / f"{name}.png"), height=height, **kw)


def wave(t0: float, t1: float, width: float, height: float, color=SOFT, stroke: float = 1.2) -> VMobject:
    """The reading's waveform between t0 and t1 (s): its min and max every 10 ms, as one filled outline."""
    z, _ = data()
    a, b = int(t0 * 100), int(t1 * 100)
    lo, hi = z["wave_lo"][a:b].astype(float), z["wave_hi"][a:b].astype(float)
    peak = max(1e-4, np.percentile(np.abs(np.r_[lo, hi]), 99.5))
    xs = np.linspace(-width / 2, width / 2, len(lo))
    top = [np.array([x, height / 2 * min(1.0, h / peak), 0]) for x, h in zip(xs, hi)]
    bot = [np.array([x, height / 2 * max(-1.0, l / peak), 0]) for x, l in zip(xs[::-1], lo[::-1])]
    m = VMobject()
    m.set_points_as_corners(top + bot + [top[0]])
    m.set_fill(color, 0.85).set_stroke(color, stroke, 0.9)
    return m


def chip(text: str, color=GOLD, size: float = 22, pad: float = 0.12, font=SERIF) -> VGroup:
    t = Text(text, font=font, font_size=size, fill_color=color)
    box = RoundedRectangle(width=t.get_width() + 2 * pad, height=t.get_height() + 2 * pad, corner_radius=0.08)
    box.set_stroke(color, 1.2, 0.8).set_fill(color, 0.08)
    box.move_to(t)
    return VGroup(box, t)


def disc(params_m: float, color, label: str, scale: float = 0.055) -> VGroup:
    """A disc whose area is the parameter count (millions), with its count under it."""
    r = scale * np.sqrt(params_m)
    c = Circle(radius=r).set_fill(color, 0.28).set_stroke(color, 2)
    t = en(label, 22, color).next_to(c, DOWN, buff=0.15)
    return VGroup(c, t)


def bar(height: float, width: float = 0.3, color=BLUE, opacity: float = 0.85) -> Rectangle:
    r = Rectangle(width=width, height=max(height, 1e-3))
    return r.set_fill(color, opacity).set_stroke(width=0)


class Said:
    """A line of narration being spoken: its start, length and sentences."""

    def __init__(self, scene: "VoiceScene", key: str, t0: float, dur: float, segs: list[dict]):
        self.scene, self.key, self.t0, self.dur, self.segs = scene, key, t0, dur, segs

    def at(self, i: int) -> float:
        """Scene time when sentence i starts."""
        return self.t0 + self.segs[min(i, len(self.segs) - 1)]["start"]

    def end_of(self, i: int) -> float:
        return self.t0 + self.segs[min(i, len(self.segs) - 1)]["end"]

    def until(self, i: int, offset: float = 0.0) -> None:
        """Wait until sentence i starts (+offset)."""
        rest = self.at(i) + offset - self.scene.time
        if rest > 1 / 120:
            self.scene.wait(rest)

    def left(self, i: int | None = None) -> float:
        """Seconds until sentence i ends (default: the line ends), at least a beat."""
        end = self.t0 + self.dur if i is None else self.end_of(i)
        return max(0.4, end - self.scene.time)


class VoiceScene(Scene):
    """A scene timed to the narration: `with self.say("key") as s:` plays the line from that moment (the
    publish step lays the audio there) and waits out the rest of it at the end of the block."""

    def setup(self):
        super().setup()
        self.log = {"voice": [], "recite": [], "shots": []}

    @contextmanager
    def say(self, key: str, pad: float = 0.6):
        _, dur, segs = voice.clip(key)
        t0 = self.time
        self.log["voice"].append({"key": key, "at": round(t0, 4), "dur": dur, "segs": segs})
        s = Said(self, key, t0, dur, segs)
        yield s
        rest = t0 + dur + pad - self.time
        if rest > 1 / 120:
            self.wait(rest)

    def recite(self, a: float, b: float, gain_db: float = 0.0) -> None:
        """The reading's audio from a to b (its seconds), starting now."""
        self.log["recite"].append({"from": a, "to": b, "at": round(self.time, 4), "gain_db": gain_db})

    def shot(self, reading_from: float, dur: float, recite: bool = True) -> None:
        """The film of the real page from the moment it heard reading second `reading_from`, for `dur` s, laid
        over the picture by the publish step, with the reading's audio under it."""
        self.log["shots"].append({"from": round(film_at(reading_from), 3), "start": round(self.time, 4), "dur": dur})
        if recite:
            self.recite(reading_from, reading_from + dur)

    def tear_down(self):
        LOG.mkdir(parents=True, exist_ok=True)
        self.log["duration"] = round(self.time, 4)
        (LOG / f"{type(self).__name__}.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
        super().tear_down()
