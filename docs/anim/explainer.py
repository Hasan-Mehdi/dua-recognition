"""The explainer: one recitation, followed from start to end (Manim Community + LaTeX, voiced by OmniVoice
in an Arabic-accented voice).

Abu Thar Al-Halawaji (a reciter the models never trained on) reads Du'a al-Iftitah from the start; on
the way he stops for six seconds, someone talks to him, and he goes back two lines (recitation.py builds
the reading). Before the app's part, how a phone hears at all and what the two models are; then
everything on screen is what the page did with this reading, replayed by dump_recitation.py
(recitation.npz / recitation.json): what Whisper heard, the probability of every du'a, the letters the
letter model heard every 20 ms, the probability of every line, the word shown.

The recording runs along the top with a playhead; each picture below is drawn from the data at the
playhead's moment, so the reading can be slowed down, held, or run fast between the moments that
matter. Whenever it runs at his own speed, his recitation is heard: the scene logs where
(media/explainer_recitation.json) and publish_explainer.py mixes it in under the narration. What is said
is in narration.py, which voices it first. The real app is shown four times (film.mjs, one phone: see
SHOTS); publish_explainer.py lays those shots in.

    pip install manim soundfile               # plus a LaTeX install with dvisvgm
    manim -qh --fps 30 --disable_caching docs/anim/explainer.py Explainer
    python docs/anim/publish_explainer.py     # shots, level the voice, subtitles -> docs/explainer.mp4
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import manimpango
import numpy as np
from manim import (
    BLACK, DOWN, GREY_B, GREY_C, GREY_D, LEFT, RIGHT, TEAL, UP, WHITE, YELLOW, ArcBetweenPoints, Arrow, Circle,
    Create, Dot, FadeIn, FadeOut, GrowArrow, ImageMobject, Line, MarkupText, Rectangle, ReplacementTransform, RoundedRectangle,
    Scene, Tex, Transform, UpdateFromAlphaFunc, ValueTracker, VGroup, VMobject, config, linear, smooth,
)
from manim.constants import RESAMPLING_ALGORITHMS

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from narration import clip  # noqa: E402

Z = np.load(HERE / "recitation.npz")
M = json.loads((HERE / "recitation.json").read_text(encoding="utf-8"))
manimpango.register_font(str(HERE / "fonts" / "Amiri-Regular.ttf"))
config.background_color = BLACK

# The reading (recitation.py's, in the ignored data/cache/media/: the recording is DuaPlayer's).
MEDIA = ROOT / "data" / "cache" / "media" / "explainer8"
# The real app on the same reading, filmed by docs/demo/film.mjs from a cold start:
#   node docs/demo/capture.mjs --file data/cache/media/explainer8/recitation.wav --start 0 --seconds 161 \
#        --credit "Abu Thar Al-Halawaji" --out data/cache/media/explainer8/runs/1-iftitah.json
#   node docs/demo/film.mjs --runs data/cache/media/explainer8/runs --follow 161 --fps 30 \
#        --out data/cache/media/explainer8/phone.mp4
# The phone starts listening PHONE_LEAD s into the film: film second f is recording second f - PHONE_LEAD.
PHONE = MEDIA / "phone.mp4"
PHONE_LEAD = float(os.environ.get("EXPLAINER_PHONE_LEAD", "5.83"))
# Where each shot of the real app starts, in recording seconds, and how long it lasts. The first is the
# page following lines 4-6, ahead of the story; "found" carries on from his "aftatiḥu th-thanā'a" to the
# name coming up. The last one ends at 158 s: about 3 s after he stopped, the page moved on to line 17,
# which he never read (the replay stays on line 16).
SHOTS = {"open": (26.3, None), "found": (23.3, 5.5), "drop": (89.5, 7.0), "end": (149.5, 8.5)}
# His phrases the narration leaves to him (find2, find3): recording seconds and their transliteration.
PHRASES = {"inni": (18.75, 20.31, "Allāhumma innī"), "aftatihu": (20.61, 22.97, "aftatiḥu th-thanā'a")}
# Two sounds in the basmala's picture (recording seconds): the s of bism, and the drawn-out a of raḥmān.
HISS, VOWEL = (1.62, 1.80), (3.35, 4.35)

DUR = float(M["duration"])
TALK = "#d08c4a"  # talk, and the share the app gives "not reading"
DIM = "#2a2a2a"
LETTERS = {i + 1: chr(c) for i, c in enumerate(range(0x0621, 0x064B))}  # align._ALPHABET, inverted
COL = {int(s): i for i, s in enumerate(Z["lines"])}
SAID = [tuple(s) for s in M["said"]]  # (line, word, start, end), recording seconds
WORD_LINE = {w: int(ln) for ln, ws in M["line_words"].items() for w in ws}
IFTITAH = M["dua_ids"].index(M["dua"])
NAMES = {"dua-tawassul": "Du'a Tawassul", "dua-baha": "Du'a Baha", "dua-jawshan-kabir": "Du'a Jawshan Kabir",
         "dua-kumayl": "Du'a Kumayl", "dua-hujjat": "Du'a Hujjah", "dua-hadithkisa": "Hadith al-Kisa",
         "dua-iftitah": "Du'a al-Iftitah", "dua-ramadan-24": "Ramadan day 24", "dua-nudbah": "Du'a Nudbah"}
CTC_COLORS = [(0, 0, 0), (0, 70, 70), (0, 160, 150), (120, 240, 220), (255, 255, 255)]
MAGMA = [(0, 0, 4), (40, 17, 90), (114, 31, 129), (183, 55, 121), (241, 96, 93), (254, 176, 120), (252, 253, 191)]
RESULTS = [("named the du'a within 10 s", "91"), ("highlight on the reader's line", "91"),
           ("on the exact word", "71"), ("a go-back, skip or jump followed within 3 s", "79"),
           ("held still through pauses, talk and salawat", "92")]
LIMITS = [("an echoing hall: named within 10 s", "75"), ("phone far from the reader: on the line", "77"),
          ("reader jumping around the du'a: on the line", "62"),
          ("a du'a the app doesn't have, shown as another", "49")]


def dua_name(dua_id: str, limit: int = 30) -> str:
    """A du'a's English name for a label (LaTeX): ours where we have one, else the corpus's, honorifics
    dropped and long ones cut at a word."""
    if dua_id in NAMES:
        return NAMES[dua_id]
    name = re.sub(r"\s*\((a\.?s\.?|atfs|sa|s\.?a\.?w?\.?a?|pbuh)\)", "", M["names"].get(dua_id, dua_id), flags=re.I)
    name = name.split(" · ")[0].strip().replace("&", r"\&")
    return name if len(name) <= limit else name[:limit].rsplit(" ", 1)[0] + r"\ldots"


def occurrences() -> list[tuple[int, float, float]]:
    """Each run of one line as he read it: (line, start, end)."""
    out = []
    for ln, _, a, b in SAID:
        if out and out[-1][0] == ln:
            out[-1] = (ln, out[-1][1], b)
        else:
            out.append((ln, a, b))
    return out


OCC = occurrences()
TALK_SPAN = (min(a for a, b, k in M["still"] if k == "talk"), max(b for a, b, k in M["still"] if k == "talk"))


# -- the data at a moment ------------------------------------------------------------------------
def step(t: float) -> int:
    """The decoder step on screen at recording second t."""
    return max(0, int(np.searchsorted(Z["t"], t, side="right")) - 1)


def update(t: float) -> int:
    """The Whisper update on screen at recording second t (-1: none yet)."""
    return int(np.searchsorted(Z["dua_t"], t, side="right")) - 1


def shown(t: float) -> tuple[int, int] | None:
    """(line, word index in the line) the page highlights at t, or None."""
    w = int(Z["word"][step(t)])
    ln = WORD_LINE.get(w) if w >= 0 else None
    return None if ln is None else (ln, M["line_words"][str(ln)].index(w))


def line_of(t: float) -> int | None:
    s = shown(t)
    return None if s is None else s[0]


def colormap(v: np.ndarray, stops) -> np.ndarray:
    stops = np.asarray(stops, dtype=float)
    x = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


# -- type ----------------------------------------------------------------------------------------
def arabic(s: str, size: float = 34, color=WHITE) -> MarkupText:
    # MarkupText, not Text: Text drops short Arabic strings whose shaped glyphs don't match its
    # characters one to one.
    return MarkupText(html.escape(s, quote=False), font="Amiri", font_size=size, color=color)


def tex(s: str, size: float = 24, color=WHITE) -> Tex:
    return Tex(s, font_size=size, color=color)


def pct(p: float) -> str:
    return rf"{round(100 * p):d}\%"


def word_groups(line: MarkupText, n: int) -> list[VGroup]:
    """The glyphs of each word of a typeset Arabic line, first word (rightmost) first: glyphs
    clustered by the gaps between them, the smallest gap that leaves n clusters."""
    glyphs = sorted(line.submobjects, key=lambda g: g.get_left()[0])
    spans = [(g.get_left()[0], g.get_right()[0]) for g in glyphs]

    def clusters(gap: float) -> list[list[int]]:
        out, right = [[0]], spans[0][1]
        for i in range(1, len(spans)):
            if spans[i][0] - right > gap:
                out.append([])
            out[-1].append(i)
            right = max(right, spans[i][1])
        return out

    for gap in np.linspace(0.02, 0.6, 59) * line.height:
        cs = clusters(gap)
        if len(cs) == n:
            return [VGroup(*[glyphs[i] for i in c]) for c in reversed(cs)]
    raise ValueError(f"could not split the line into {n} words")


_CACHE: dict = {}


def cached(key, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


# -- the recording ---------------------------------------------------------------------------------
TOP_Y, SX0, SX1 = 3.3, -6.5, 6.5


class Recording(VGroup):
    """The recording as its loudness, from t0 to t1: heard so far in grey, still to come dim, talk in
    its own colour, line numbers over where each line starts, the playhead at the clock. window marks
    what a model hears at the playhead: "whisper" (the last 6 s) or "letters" (the last 2 s)."""

    def __init__(self, t0=0.0, t1=DUR, x0=SX0, x1=SX1, y=TOP_Y, h=0.5, numbers=True, preview=False):
        super().__init__()
        self.t0, self.t1, self.x0, self.x1, self.y, self.h, self.preview = t0, t1, x0, x1, y, h, preview
        env = Z["env"]
        self.ts = (np.arange(len(env)) + 0.5) * 0.1
        self.v = np.maximum(np.clip((env + 31.0) / 20.0, 0.0, 1.0) ** 1.3 * h / 2, 0.012)
        self.body, self.labels, self.win = VGroup(), VGroup(), VGroup()
        self.spans = []
        if numbers:
            for ln, a, b in OCC:
                if t0 <= a <= t1:
                    self.labels.add(tex(str(ln), 15 * max(1.0, h / 0.5) ** 0.5, GREY_D)
                                    .move_to([self.x(a) + 0.08, y + h / 2 + 0.17, 0], aligned_edge=LEFT))
                    self.spans.append((a, b))
        self.head = Line([0, y - h / 2 - 0.12, 0], [0, y + h / 2 + 0.12, 0], stroke_width=2.5, color=WHITE)
        self.window = None
        self.add(self.body, self.labels, self.win, self.head)
        self.set_time(0.0)

    def x(self, s):
        return self.x0 + (self.x1 - self.x0) * (np.asarray(s) - self.t0) / (self.t1 - self.t0)

    def _shape(self, a: float, b: float, color, opacity: float):
        i = (self.ts >= max(a, self.t0)) & (self.ts <= min(b, self.t1))
        if i.sum() < 2:
            return None
        xs, hs = self.x(self.ts[i]), self.v[i]
        pts = [[x, self.y + q, 0] for x, q in zip(xs, hs)] + [[x, self.y - q, 0] for x, q in zip(xs[::-1], hs[::-1])]
        return VMobject(stroke_width=0, fill_color=color, fill_opacity=opacity).set_points_as_corners(pts + [pts[0]])

    def set_time(self, t: float):
        self.t = t
        heard_to = self.t1 if self.preview else t
        cuts = sorted({self.t0, self.t1, min(max(heard_to, self.t0), self.t1), *TALK_SPAN})
        parts = []
        for a, b in zip(cuts, cuts[1:]):
            talk = TALK_SPAN[0] <= a and b <= TALK_SPAN[1]
            past = b <= heard_to + 1e-6
            s = self._shape(a, b, TALK if talk else (GREY_B if past else GREY_D), 0.9 if past else 0.45)
            if s is not None:
                parts.append(s)
        self.body.submobjects = parts
        for lab, (a, b) in zip(self.labels, self.spans):
            lab.set_color(WHITE if a <= t <= b + 0.5 else (GREY_B if a <= heard_to else GREY_D))
        on = self.t0 <= t <= self.t1
        self.head.put_start_and_end_on([self.x(t), self.y - self.h / 2 - 0.12, 0], [self.x(t), self.y + self.h / 2 + 0.12, 0])
        self.head.set_stroke(opacity=1.0 if on and not self.preview else 0.0)
        if self.window and on:
            span, color = (6.0, YELLOW) if self.window == "whisper" else (2.0, TEAL)
            a = max(t - span, self.t0)
            self.win.submobjects = [Rectangle(width=max(self.x(t) - self.x(a), 0.02), height=self.h + 0.14,
                                              stroke_color=color, stroke_width=2).move_to([(self.x(a) + self.x(t)) / 2, self.y, 0])]
        else:
            self.win.submobjects = []
        return self


class Windows(VGroup):
    """What each model hears of a Recording at the clock: Whisper, its last three 6 s windows (one a
    second, the newest brightest); or the letter model, its last 2 s and a tick for every run (ten a
    second)."""

    def __init__(self, rec: Recording, kind: str):
        super().__init__()
        self.rec, self.kind = rec, kind
        self.name = tex("Whisper", 26, YELLOW) if kind == "whisper" else tex("letters", 26, TEAL)
        self.parts = VGroup()
        self.add(self.parts, self.name)

    def set_time(self, t: float):
        x, top = self.rec.x, self.rec.y + self.rec.h / 2
        if self.kind == "whisper":
            k = update(t)
            ends = [float(Z["dua_t"][j]) for j in (k, k - 1, k - 2) if j >= 0]
            self.parts.submobjects = [
                Rectangle(width=x(e) - x(max(e - 6, self.rec.t0)), height=0.24, stroke_color=YELLOW, stroke_width=2,
                          stroke_opacity=(1.0, 0.45, 0.25)[i])
                .move_to([(x(max(e - 6, self.rec.t0)) + x(e)) / 2, top + 0.4 + 0.32 * i, 0]) for i, e in enumerate(ends)]
            left = x(max(ends[0] - 6, self.rec.t0)) if ends else x(t)
            self.name.next_to([left, top + 0.4, 0], LEFT, 0.3)
        else:
            y = self.rec.y - self.rec.h / 2 - 0.45
            a = max(t - 2.0, self.rec.t0)
            runs = np.arange(np.ceil(a * 10) / 10, t + 1e-6, 0.1)
            self.parts.submobjects = [
                Rectangle(width=max(x(t) - x(a), 0.02), height=0.24, stroke_color=TEAL, stroke_width=2).move_to([(x(a) + x(t)) / 2, y, 0]),
                *[Line([x(r), y + 0.2, 0], [x(r), y + 0.42, 0], stroke_width=1.5, color=TEAL) for r in runs]]
            self.name.next_to([x(a), y, 0], LEFT, 0.3)
        return self


# -- which du'a ----------------------------------------------------------------------------------
FIELD_COLS, FIELD_GAP = 29, 0.19


def field_cells() -> list[tuple[int, int]]:
    """A cell per du'a: the texts people recite most nearer the middle."""
    rec = np.array(M["recordings"])
    order = np.argsort(-rec, kind="stable")
    rows = int(np.ceil(len(order) / FIELD_COLS))
    cells = sorted(((r - (rows - 1) / 2) ** 2 + ((c - (FIELD_COLS - 1) / 2) * 0.62) ** 2, r, c)
                   for r in range(rows) for c in range(FIELD_COLS))
    pos = [None] * len(order)
    for k, (_, r, c) in zip(order, cells):
        pos[k] = (r, c)
    return pos


class Field(VGroup):
    """Every du'a as a dot, as bright and large as its probability; the likeliest named on the right."""

    def __init__(self, center=(-0.6, -0.15, 0), labels: int = 3):
        super().__init__()
        self.cells = field_cells()
        rows = int(np.ceil(len(self.cells) / FIELD_COLS))
        self.cx = center[0] - (FIELD_COLS - 1) / 2 * FIELD_GAP
        self.cy = center[1] + (rows - 1) / 2 * FIELD_GAP
        self.n_labels = labels
        self.dots = VGroup(*[Dot([self.cx + c * FIELD_GAP, self.cy - r * FIELD_GAP, 0], radius=0.026, color=GREY_D,
                                 fill_opacity=0.55) for r, c in self.cells])
        self.names = VGroup()
        self.add(self.dots, self.names)
        self.k = -2
        self.set_time(0.0)

    def set_time(self, t: float):
        k = update(t)
        if k == self.k:
            return self
        self.k = k
        p = Z["dua_mass"][k] if k >= 0 else np.zeros(len(self.cells))
        for d, dot in enumerate(self.dots):
            q = float(p[d])
            c = dot.get_center()
            dot.scale_to_fit_width(2 * (0.026 + 0.09 * np.sqrt(q)))
            dot.move_to(c)
            if q > 0.002:
                dot.set_fill(TEAL, opacity=min(1.0, 0.35 + 2.5 * np.sqrt(q)))
            else:
                dot.set_fill(GREY_D, opacity=0.55)
        names = []
        if self.n_labels and k >= 0:
            top = [d for d in np.argsort(-p)[: self.n_labels] if p[d] >= 0.01]
            right = self.cx + (FIELD_COLS - 1) * FIELD_GAP + 0.5
            for i, d in enumerate(sorted(top, key=lambda d: self.cells[d])):
                name = dua_name(M["dua_ids"][d])
                lab = tex(rf"{name}\enspace {pct(p[d])}", 24).move_to([right, self.cy - 1.0 - i * 0.62, 0], aligned_edge=LEFT)
                names += [Line(self.dots[d].get_center(), lab.get_left() + 0.1 * LEFT, stroke_width=1, color=GREY_B), lab]
        self.names.submobjects = names
        return self


class Heard(VGroup):
    """What Whisper wrote for the window on screen."""

    def __init__(self, pos=(-3.6, 2.35, 0), size: float = 40, max_w: float = 6.5):
        super().__init__()
        self.pos, self.size, self.max_w, self.k = pos, size, max_w, -2

    def set_time(self, t: float):
        k = update(t)
        if k == self.k:
            return self
        self.k = k
        text = M["updates"][k]["heard"] if k >= 0 else ""
        if not text:
            self.submobjects = []
            return self
        m = cached(("heard", text, self.size), lambda: arabic(text, self.size, YELLOW)).copy()
        if m.width > self.max_w:
            m.scale_to_fit_width(self.max_w)
        self.submobjects = [m.move_to(self.pos)]
        return self


class Share(VGroup):
    """One du'a's probability (Iftitah's by default) as a bar, with the 70% the page waits for."""

    def __init__(self, x: float = 6.0, y0: float = -2.3, hgt: float = 4.0, dua: int = IFTITAH, null: bool = False):
        super().__init__()
        self.x, self.y0, self.hgt, self.dua, self.null = x, y0, hgt, dua, null
        frame = Rectangle(width=0.34, height=hgt, stroke_color=GREY_C, stroke_width=1.5).move_to([x, y0 + hgt / 2, 0])
        yt = y0 + 0.7 * hgt
        self.fill = Rectangle(width=0.34, height=hgt, stroke_width=0, fill_color=TEAL, fill_opacity=1)
        self.num = VGroup()
        self.add(frame, self.fill, Line([x - 0.3, yt, 0], [x + 0.3, yt, 0], stroke_width=2, color=WHITE),
                 tex(r"70\%", 20, GREY_B).next_to([x + 0.3, yt, 0], RIGHT, 0.1), self.num)
        self.k = -2
        self.set_time(0.0)

    def set_time(self, t: float):
        k = update(t)
        if k == self.k:
            return self
        self.k = k
        p = float(Z["dua_mass"][k, self.dua]) if k >= 0 else 0.0
        h = max(self.hgt * p, 0.001)
        self.fill.stretch_to_fit_height(h).move_to([self.x, self.y0 + h / 2, 0])
        self.fill.set_fill(opacity=1.0 if p > 0.004 else 0.0)
        self.num.submobjects = [cached(("pct", pct(p), p >= 0.7), lambda: tex(pct(p), 26, TEAL if p >= 0.7 else WHITE)).copy()
                                .next_to([self.x, self.y0 + self.hgt, 0], UP, 0.18)]
        return self


# -- following -------------------------------------------------------------------------------------
LINE_X, LINE_Y = -1.9, 2.05


def line_text(ln: int, hi: int | None, x: float = LINE_X, y: float = LINE_Y, size: float = 44) -> VGroup:
    """Line ln of Iftitah, word hi lit as the page lights it, its number and its English."""
    words = M["lines"][str(ln)]["ar"].split()
    t = arabic(M["lines"][str(ln)]["ar"], size)
    if t.width > 8.6:
        t.scale_to_fit_width(8.6)
    t.move_to([x, y, 0])
    if hi is not None:
        try:
            word_groups(t, len(words))[hi].set_color(TEAL)
        except ValueError:
            pass
    en = tex(M["lines"][str(ln)]["en"].replace("&", r"\&"), 22, GREY_B).next_to(t, DOWN, 0.28)
    return VGroup(t, tex(str(ln), 22, GREY_B).next_to(t, RIGHT, 0.35), en)


class Page(VGroup):
    """The line the page shows at the clock, its word lit; nothing when the page shows nothing."""

    def __init__(self, x: float = LINE_X, y: float = LINE_Y, size: float = 44):
        super().__init__()
        self.x, self.y, self.size, self.key = x, y, size, "-"

    def set_time(self, t: float):
        s = shown(t)
        if s == self.key:
            return self
        self.key = s
        self.submobjects = [] if s is None else [cached(("line", s, self.x, self.y, self.size),
                                                        lambda: line_text(s[0], s[1], self.x, self.y, self.size)).copy()]
        return self


class Letters(VGroup):
    """The 2 s the letter model heard last, its likeliest letter every 20 ms (a dot where none), right
    to left like the text: the newest on the left."""

    N = 100

    def __init__(self, x: float = LINE_X, y: float = -0.35, w: float = 8.4, h: float = 1.0):
        super().__init__()
        self.x, self.y, self.w = x, y, w
        self.box = RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=GREY_C, stroke_width=1.5).move_to([x, y, 0])
        self.right, self.dx = x + w / 2 - 0.2, (w - 0.4) / self.N
        self.dots = VGroup(*[Dot([self.right - (j + 0.5) * self.dx, y - 0.02, 0], radius=0.011, color=GREY_D)
                             for j in range(self.N)])  # dots[0] is the newest frame
        self.glyphs = VGroup()
        self.add(self.box, self.dots, self.glyphs)
        self.set_time(0.0)

    def set_time(self, t: float):
        i1 = int(round(t / 0.02))
        let, lp = Z["letter"], Z["letter_p"]
        glyphs, prev = [], 0
        for j in range(self.N - 1, -1, -1):  # oldest first
            i = i1 - 1 - j
            c = int(let[i]) if 0 <= i < len(let) else 0
            self.dots[j].set_opacity(1.0 if c == 0 else 0.0)
            if c and c != prev:
                q = float(lp[i])
                g = cached(("letter", c), lambda: arabic(LETTERS[c], 30)).copy()
                glyphs.append(g.set_opacity(0.35 + 0.65 * min(1.0, q)).move_to([self.right - (j + 0.5) * self.dx, self.y + 0.05, 0]))
            prev = c
        self.glyphs.submobjects = glyphs
        return self


ROW0, ROW_GAP, BAR_X, BAR_W = 2.45, 0.268, 5.5, 1.55


def row_y(ln: int) -> float:
    return ROW0 - (ln - 1) * ROW_GAP


OFF_Y = row_y(16) - 0.5


def speech_icon(color=TALK) -> VGroup:
    """A speech bubble: someone talking, not reading."""
    b = RoundedRectangle(corner_radius=0.12, width=0.62, height=0.4, stroke_color=color, stroke_width=2.5)
    tail = VMobject(stroke_color=color, stroke_width=2.5).set_points_as_corners(
        [b.get_bottom() + 0.12 * LEFT, b.get_bottom() + 0.2 * LEFT + 0.16 * DOWN, b.get_bottom() + 0.02 * LEFT])
    return VGroup(b, tail)


class Lines(VGroup):
    """Lines 1-16 of Iftitah, top to bottom, each with the probability the app gives it; the line it
    shows in white. talk: the share it gives "not reading", below."""

    def __init__(self, talk: bool = False):
        super().__init__()
        self.bars, self.nums = VGroup(), VGroup()
        tracks = VGroup()
        for ln in range(1, 17):
            yy = row_y(ln)
            tracks.add(Line([BAR_X - BAR_W, yy, 0], [BAR_X, yy, 0], stroke_width=1, color=DIM))
            self.bars.add(Rectangle(width=BAR_W, height=0.15, stroke_width=0, fill_color=TEAL, fill_opacity=0.95))
            self.nums.add(tex(str(ln), 17, GREY_C).move_to([BAR_X + 0.32, yy, 0]))
        self.add(tracks, self.bars, self.nums)
        self.talk = None
        if talk:
            self.talk = Rectangle(width=BAR_W, height=0.15, stroke_width=0, fill_color=TALK, fill_opacity=0.95)
            self.add(Line([BAR_X - BAR_W, OFF_Y, 0], [BAR_X, OFF_Y, 0], stroke_width=1, color=DIM), self.talk,
                     speech_icon().scale(0.5).move_to([BAR_X + 0.32, OFF_Y, 0]))
        self.set_time(0.0)

    @staticmethod
    def _bar(bar, p: float, yy: float):
        w = max(BAR_W * p, 0.002)
        bar.stretch_to_fit_width(w).move_to([BAR_X - w / 2, yy, 0])
        bar.set_fill(opacity=0.95 if p > 0.004 else 0.0)

    def set_time(self, t: float):
        k = step(t)
        cur = line_of(t)
        for ln in range(1, 17):
            self._bar(self.bars[ln - 1], float(Z["line_p"][k, COL[ln]]), row_y(ln))
            self.nums[ln - 1].set_color(WHITE if cur == ln else GREY_C)
        if self.talk is not None:
            self._bar(self.talk, float(Z["off"][k]), OFF_Y)
        return self


def moves(src: int, targets: dict[int, float]) -> VGroup:
    """Arcs from line src to other lines, right of the numbers, as thick as the move is cheap; a loop
    for reading the line again."""
    g = VGroup()
    x = BAR_X + 0.62
    for dst, wgt in targets.items():
        if dst == src:
            g.add(Circle(radius=0.11, stroke_width=wgt, color=GREY_B).move_to([x + 0.13, row_y(src), 0]))
            continue
        bend = 0.9 + 0.12 * abs(dst - src)
        g.add(ArcBetweenPoints([x, row_y(src), 0], [x, row_y(dst), 0],
                               angle=-bend * np.pi / 2 if dst > src else bend * np.pi / 2, stroke_width=wgt, color=GREY_B))
    return g


def ladder(src: int, lo: int, hi: int, x: float = 1.2, y0: float = 2.0, gap: float = 0.78):
    """Lines lo..hi of Iftitah large, one under the other (number, then the line), line src in white; and
    the moves out of src drawn right of the numbers: (rows, the free move to the next line, the costly
    ones: again, back one or two, ahead two or three)."""
    yy = {ln: y0 - (ln - lo) * gap for ln in range(lo, hi + 1)}
    rows = VGroup()
    for ln in range(lo, hi + 1):
        color = WHITE if ln == src else GREY_C
        text = arabic(M["lines"][str(ln)]["ar"], 30, color)
        if text.width > 7.4:
            text.scale_to_fit_width(7.4)
        rows.add(tex(str(ln), 26, color).move_to([x, yy[ln], 0]), text.next_to([x - 0.35, yy[ln], 0], LEFT, 0))
    ax = x + 0.45

    def arc(dst, w, color=GREY_B):
        if dst == src:
            return Circle(radius=0.16, stroke_width=w, color=color).move_to([ax + 0.18, yy[src], 0])
        bend = 0.8 + 0.1 * abs(dst - src)
        return ArcBetweenPoints([ax, yy[src], 0], [ax, yy[dst], 0], angle=(-1 if dst > src else 1) * bend * np.pi / 2,
                                stroke_width=w, color=color)

    free = arc(src + 1, 7.0, TEAL)
    costly = VGroup(*[arc(d, w) for d, w in ((src, 2.4), (src - 1, 2.0), (src - 2, 1.5), (src + 2, 1.5), (src + 3, 1.1))
                      if lo <= d <= hi])
    return rows, free, costly


# -- how a phone hears -----------------------------------------------------------------------------
def basmala_words() -> list[tuple[str, float, float]]:
    return [(M["word_text"][str(w)], a, b) for ln, w, a, b in SAID if ln == 1]


def waveform(x0: float, x1: float, y: float, h: float, color=GREY_B) -> VMobject:
    """The basmala as it reached the phone: the lowest and highest sample every 2 ms."""
    lo, hi = Z["basmala_lo"].astype(float), Z["basmala_hi"].astype(float)
    s = h / 2 / max(np.abs(lo).max(), hi.max())
    xs = np.linspace(x0, x1, len(lo))
    pts = [[x, y + v * s, 0] for x, v in zip(xs, hi)] + [[x, y + v * s, 0] for x, v in zip(xs[::-1], lo[::-1])]
    return VMobject(stroke_width=0.6, stroke_color=color, fill_color=color, fill_opacity=1).set_points_as_corners(pts + [pts[0]])


def spectrogram(w: float, h: float) -> ImageMobject:
    """Whisper's picture of the basmala: 80 bands, low at the bottom, a column every 10 ms."""
    mel = Z["basmala_mel"][::-1].astype(float) / 255
    img = ImageMobject(colormap(mel ** 1.5, MAGMA))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    return img.stretch_to_fit_width(w).stretch_to_fit_height(h)


def mel_crop(a: float, b: float, h: float = 1.0) -> ImageMobject:
    """The picture of recording seconds a..b of the basmala."""
    b0 = float(Z["basmala_span"][0])
    mel = Z["basmala_mel"][::-1, int((a - b0) * 100) : int((b - b0) * 100)].astype(float) / 255
    img = ImageMobject(colormap(mel ** 1.5, MAGMA))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    return img.stretch_to_fit_height(h).stretch_to_fit_width(h * (b - a) * 1.4)


def network(x: float, y: float, w: float = 4.2, h: float = 2.6, sizes=(6, 9, 9, 9, 6), seed: int = 0) -> VGroup:
    """A small drawing of a network: columns of units, every unit joined to the next column's; how
    much each link counts is how bright it is."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(x - w / 2, x + w / 2, len(sizes))
    cols = [VGroup(*[Dot([cx, yy, 0], radius=0.06, color=GREY_B) for yy in np.linspace(y - h / 2 * n / max(sizes),
                                                                                        y + h / 2 * n / max(sizes), n)])
            for cx, n in zip(xs, sizes)]
    links = VGroup()
    for a, b in zip(cols, cols[1:]):
        for p in a:
            for q in b:
                links.add(Line(p.get_center(), q.get_center(), stroke_width=1, color=GREY_B,
                               stroke_opacity=float(rng.uniform(0.05, 0.6))))
    return VGroup(links, *cols)


def renudge(net: VGroup, seed: int) -> VGroup:
    """The same network with its links' weights nudged."""
    rng = np.random.default_rng(seed)
    out = net.copy()
    for ln in out[0]:
        ln.set_stroke(opacity=float(np.clip(ln.get_stroke_opacity() + rng.normal(0, 0.18), 0.04, 0.8)))
    return out


def letter_grid(t: float, w: float, h: float) -> tuple[ImageMobject, list[int]]:
    """The letter model's last 2 s before t: a row for no letter (on top) and one for each letter it
    heard in them, a column per 20 ms (newest on the left), brighter the likelier; and those letters."""
    i1 = int(round(t / 0.02))
    p = np.exp(Z["letters_lp"][i1 - 100 : i1].astype(float)).T[:, ::-1]  # [columns of the model, frames]
    used = [int(c) for c in np.flatnonzero(p[1:].max(1) > 0.3) + 1]
    img = ImageMobject(colormap(p[[0] + used] ** 0.45, CTC_COLORS))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return img.stretch_to_fit_width(w).stretch_to_fit_height(h), used


def phone_icon(w: float = 1.6, h: float = 3.0) -> VGroup:
    return VGroup(RoundedRectangle(corner_radius=0.22, width=w, height=h, stroke_color=GREY_B, stroke_width=2.5),
                  Line([-0.18, h / 2 - 0.16, 0], [0.18, h / 2 - 0.16, 0], stroke_width=2.5, color=GREY_B))


class Explainer(Scene):
    def construct(self):
        self._subs, self._shots, self._rec = [], [], []
        self.clock = ValueTracker(0.0)
        only = os.environ.get("EXPLAINER_ONLY", "").split(",")  # some parts alone, to check their layout
        for part in ("opening", "hearing", "models", "finding", "following", "readers", "ending"):
            if only == [""] or part in only:
                getattr(self, part)()
        self.write_subtitles()
        (Path(config.media_dir) / "explainer_shots.json").write_text(json.dumps(self._shots, indent=1), encoding="utf-8")

    # -- helpers -------------------------------------------------------------------------------------
    def get_moving_and_static_mobjects(self, animations):
        # Everything is drawn afresh every frame, from the scene's top-level mobjects: the views swap
        # their parts as the clock moves, and Manim's own list (every part, flattened when an
        # animation starts) would keep drawing the parts they swapped out.
        return list(self.mobjects) + [m for m in self.foreground_mobjects if m not in self.mobjects], []

    def sound_at_now(self, path: Path, gain: float = 0.0):
        # Not Scene.add_sound: it does nothing right after a play call served from Manim's cache.
        self.renderer.file_writer.add_sound(str(path), self.renderer.time, gain)

    @contextmanager
    def voice(self, key: str, pad: float = 0.4):
        """Speak one narration line; the block's animations play under it, and the scene waits for
        the line to finish."""
        path, dur, parts = clip(key)
        start = self.renderer.time
        self._cue = (start, parts)
        self.sound_at_now(path)
        self._subs += [(start + p["start"], start + p["end"], p["show"], p["lang"]) for p in parts]
        yield dur
        left = start + dur - self.renderer.time
        if left > 0.02:
            self.wait(left)
        if pad:
            self.wait(pad)

    def cue(self, i: int, lead: float = 0.0):
        """Wait until sentence i of the line being spoken starts (`lead` seconds early)."""
        start, parts = self._cue
        t = start + parts[min(i, len(parts) - 1)]["start"] - lead
        if t > self.renderer.time + 0.02:
            self.wait(t - self.renderer.time)

    def left_in(self, i: int) -> float:
        """Seconds from now to the end of sentence i of the line being spoken."""
        start, parts = self._cue
        return max(0.1, start + parts[min(i, len(parts) - 1)]["end"] - self.renderer.time)

    def until(self, i: int) -> float:
        """Seconds from now to the start of sentence i of the line being spoken."""
        start, parts = self._cue
        return max(0.1, start + parts[min(i, len(parts) - 1)]["start"] - self.renderer.time)

    def shot(self, name: str, dur: float | None = None):
        """The real app over the frame from the shot's recording second, for dur seconds; the clock goes
        to that second, and the caller runs it at real speed under the shot (so his voice, in the local
        copy, matches the film). publish_explainer.py lays the picture in."""
        at, d = SHOTS[name]
        d = dur if dur is not None else d
        start = self.renderer.time
        self._shots.append({"start": round(start, 3), "from": round(at + PHONE_LEAD, 3), "dur": round(d, 3)})
        self.clock.set_value(at)  # under the shot; run() at real speed plays his voice with it
        return d

    def run(self, t1: float, run_time: float, *anims, rate=linear):
        """Move the clock to recording second t1 over run_time; at real speed his voice is heard (logged for
        publish_explainer.py, which mixes it in)."""
        t0 = self.clock.get_value()
        if t1 > t0 and rate is linear and abs(run_time - (t1 - t0)) < 0.05:
            self._rec.append({"at": round(self.renderer.time, 3), "from": round(t0, 3), "to": round(t1, 3)})
        self.play(self.clock.animate.set_value(t1), *anims, run_time=run_time, rate_func=rate)

    def at_speed(self, t1: float, within: float, first: bool = True):
        """Get the clock to t1 in `within` seconds: at his own speed (heard) when that fits, the time left
        over spent holding before (first) or after; faster than him (not heard) when it doesn't."""
        d = t1 - self.clock.get_value()
        if d <= 0:
            self.wait(max(within, 0.02))
        elif d > within + 0.02:
            self.run(t1, max(within, 0.1))
        else:
            if first and within - d > 0.02:
                self.wait(within - d)
            self.run(t1, d)
            if not first and within - d > 0.02:
                self.wait(within - d)

    def phrase(self, key: str):
        """Run the clock at his speed through one of his phrases (PHRASES), with it as a subtitle."""
        a, b, said = PHRASES[key]
        lead = max(0.0, a - self.clock.get_value())
        self._subs.append((self.renderer.time + lead, self.renderer.time + lead + (b - a), said, "ar", "reciter"))
        self.run(b + 0.25, b + 0.25 - self.clock.get_value())

    def cue_at(self, i: int, frac: float):
        """Wait until `frac` of the way through sentence i of the line being spoken."""
        start, parts = self._cue
        p = parts[min(i, len(parts) - 1)]
        t = start + p["start"] + frac * (p["end"] - p["start"])
        if t > self.renderer.time + 0.02:
            self.wait(t - self.renderer.time)

    def follow(self, *views):
        """Keep views drawn at the clock."""
        for v in views:
            v.set_time(self.clock.get_value())
            # Only the view itself: an animation's copies of it (a fade's start and end) carry the
            # updater too, and redrawn mid-animation their parts no longer match the view's.
            v.add_updater(lambda m, v=v: m.set_time(self.clock.get_value()) if m is v else None)
        return views

    def hide(self, *views, others=(), run_time: float = 0.5):
        """Fade views out. Their redraw stops first: a fade out lets the view update once at its end and
        then puts its starting shape back onto whatever parts the view has by then."""
        for v in views:
            v.clear_updaters()
        self.play(*[FadeOut(v) for v in views], *others, run_time=run_time)

    def show(self, *views, others=(), run_time: float = 0.5):
        """Fade views in, drawn at the clock and following it again."""
        for v in views:
            v.clear_updaters()
            if hasattr(v, "key"):
                v.key = "-"
            if hasattr(v, "k"):
                v.k = -2
        self.follow(*views)
        self.play(*[FadeIn(v) for v in views], *others, run_time=run_time)

    def write_subtitles(self):
        """When each sentence of the narration, and each of his phrases it leaves to him, is said (for
        publish_explainer.py's subtitles, and for when it lowers his voice under the narrator's); and
        where his recitation plays."""
        subs = [{"start": round(s[0], 3), "end": round(s[1], 3), "text": s[2], "lang": s[3],
                 "who": s[4] if len(s) > 4 else "narrator"} for s in sorted(self._subs)]
        media = Path(config.media_dir)
        (media / "explainer_voice.json").write_text(json.dumps(subs, ensure_ascii=False, indent=0), encoding="utf-8")
        (media / "explainer_recitation.json").write_text(json.dumps(self._rec, indent=0), encoding="utf-8")

    def clear(self, keep=(), run_time: float = 0.8):
        gone = [m for m in self.mobjects if m not in keep and m is not self.clock]
        for m in gone:
            m.clear_updaters()
        if gone:
            self.play(*[FadeOut(m) for m in gone], run_time=run_time)
        self.wait(0.2)

    # -- 1. the reading --------------------------------------------------------------------------------
    def opening(self):
        # The real app first, following him word by word, and him alone for a few seconds; the narration
        # comes in over it.
        at, lead, dur = SHOTS["open"][0], 3.5, clip("open1")[1]
        d = self.shot("open", lead + dur + 0.8)
        self.run(at + lead, lead)
        with self.voice("open1", pad=0.0):
            self.run(at + lead + dur, dur)
        self.run(at + d, d - lead - dur)
        # The whole reading, ahead of him, from his first word.
        self.clock.set_value(0.0)
        big = Recording(y=0.2, h=1.6, x0=-6.6, x1=6.6, preview=True)
        big.set_time(0.0)
        with self.voice("ahead1"):
            self.play(FadeIn(big.labels, lag_ratio=0.05), Create(big.body), run_time=2.4, rate_func=smooth)
        self.top = Recording()
        self.top.set_time(0.0)
        self.play(ReplacementTransform(big, self.top), run_time=1.4)
        self.follow(self.top)

    def ensure_top(self):
        if not hasattr(self, "top"):
            self.top = Recording()
            self.add(*self.follow(self.top))

    # -- 2. how a phone hears ------------------------------------------------------------------------
    def hearing(self):
        self.ensure_top()
        b0, b1 = float(Z["basmala_span"][0]), float(Z["basmala_span"][1])
        x0, x1, y = -6.0, 6.0, 0.9
        sx = lambda s: x0 + (x1 - x0) * (s - b0) / (b1 - b0)  # noqa: E731
        wave = waveform(x0, x1, y, 2.0)
        words = VGroup(*[arabic(w, 36).move_to([sx((a + b) / 2), -0.55, 0]) for w, a, b in basmala_words()])
        # 20 ms inside "allah", sample by sample.
        s = Z["basmala_samples"].astype(float)
        at = float(Z["basmala_at"][0])
        mark = Rectangle(width=max(sx(at + 0.02) - sx(at), 0.05), height=2.3, stroke_color=TEAL, stroke_width=2).move_to(
            [(sx(at) + sx(at + 0.02)) / 2, y, 0])
        px0, px1, py, ph = -6.0, 2.2, -1.75, 1.3
        xs = np.linspace(px0, px1, len(s))
        scale = ph / 2 / np.abs(s).max()
        stems = VGroup(*[Line([x, py, 0], [x, py + v * scale, 0], stroke_width=1.2, color=GREY_C) for x, v in zip(xs, s)])
        dots = VGroup(*[Dot([x, py + v * scale, 0], radius=0.022, color=TEAL) for x, v in zip(xs, s)])
        nums = VGroup(*[tex(f"{v:+.3f}".replace("-", r"$-$"), 20, GREY_B) for v in s[150:159]]).arrange(DOWN, buff=0.07)
        nums.next_to([5.0, -0.95, 0], DOWN, 0, aligned_edge=LEFT)  # clear of the words, right of the samples
        dots_more = tex(r"\dots", 24, GREY_B).next_to(nums, DOWN, 0.08)
        # First, the basmala as he says it, a playhead crossing its waveform.
        head = Line([0, y - 1.15, 0], [0, y + 1.15, 0], stroke_width=2.5, color=WHITE)
        head.add_updater(lambda m: m.move_to([sx(min(max(self.clock.get_value(), b0), b1)), y, 0]))
        self.play(FadeIn(wave), FadeIn(words), run_time=1.0)
        self.run(b0, 0.3, rate=smooth)
        self.add(head)
        self.run(b1, b1 - b0)
        head.clear_updaters()
        self.play(FadeOut(head), run_time=0.4)
        with self.voice("sound1"):
            self.cue(1, 0.2)  # "sixteen thousand times a second"
            self.play(Create(mark), run_time=0.6)
            self.play(Transform(mark.copy(), Rectangle(width=px1 - px0 + 0.3, height=ph + 0.4, stroke_color=TEAL,
                                                       stroke_width=2).move_to([(px0 + px1) / 2, py, 0]), remover=True),
                      FadeIn(stems), FadeIn(dots), run_time=1.2)
            self.cue(2, 0.1)  # "seventy-five thousand numbers"
            self.play(FadeIn(nums, lag_ratio=0.15), FadeIn(dots_more), run_time=1.8)
        # The same sound as a picture: a column every hundredth of a second.
        spec = spectrogram(x1 - x0, 2.4).move_to([0, y, 0])
        cover = Rectangle(width=x1 - x0 + 0.02, height=2.45, stroke_width=0, fill_color=BLACK, fill_opacity=1).move_to(spec)
        with self.voice("sound2"):
            self.play(FadeOut(stems), FadeOut(dots), FadeOut(nums), FadeOut(dots_more), FadeOut(mark), run_time=0.8)
            self.cue(1, 0.3)  # "Every hundredth of a second..."
            self.add(spec, cover, words)
            self.remove(wave)
            self.play(UpdateFromAlphaFunc(cover, lambda m, a: m.stretch_to_fit_width(max((x1 - x0) * (1 - a), 0.001))
                                          .move_to([x1 - (x1 - x0) * (1 - a) / 2, y, 0])),
                      run_time=min(5.0, self.left_in(1)), rate_func=linear)
            self.remove(cover)
            self.cue(2, 0.2)  # "each sound as a shape of its own"
            for w in words:
                self.play(w.animate.set_color(TEAL), run_time=0.35)
                self.play(w.animate.set_color(WHITE), run_time=0.25)
            marks = VGroup(*[Rectangle(width=sx(b) - sx(a), height=2.42, stroke_color=TEAL, stroke_width=2.5)
                             .move_to([(sx(a) + sx(b)) / 2, y, 0]) for a, b in (HISS, VOWEL)])
            self.cue(3, 0.15)  # "The s is a faint haze reaching to the top,"
            self.play(Create(marks[0]), run_time=0.6)
            self.cue_at(3, 0.45)  # "and a vowel he draws out is a stack of bright stripes"
            self.play(Create(marks[1]), run_time=0.6)
        self.play(FadeOut(marks), run_time=0.5)
        self.picture = spec
        self.words = words

    # -- 3. the two models -----------------------------------------------------------------------------
    def models(self):
        self.ensure_top()
        if not hasattr(self, "picture"):
            self.picture = spectrogram(12.0, 2.4).move_to([0, 0.9, 0])
            self.words = VGroup(*[arabic(w, 36) for w, _, _ in basmala_words()]).arrange(LEFT, buff=1.4).move_to([0, -0.55, 0])
            self.add(self.picture, self.words)
        spec, words = self.picture, self.words
        small = spectrogram(2.6, 1.3).move_to([-5.2, -0.3, 0])
        net = network(-0.4, -0.3, w=5.4, h=3.4)
        out = arabic("بسم الله الرحمن الرحيم", 32).move_to([4.9, -0.3, 0])
        arrows = VGroup(Arrow(small.get_right(), net.get_left() + 0.1 * LEFT, buff=0.15, stroke_width=2.5, color=GREY_B,
                              max_tip_length_to_length_ratio=0.12),
                        Arrow(net.get_right() + 0.1 * RIGHT, out.get_left(), buff=0.15, stroke_width=2.5, color=GREY_B,
                              max_tip_length_to_length_ratio=0.12))
        # Training pairs: a stretch of sound and what was said in it.
        pairs = [(mel_crop(a, b, 1.0), arabic(w, 38)) for w, a, b in basmala_words()]
        with self.voice("model1"):
            self.play(FadeOut(words), FadeOut(spec), FadeIn(small), run_time=1.2)
            self.play(FadeIn(net, lag_ratio=0.02), GrowArrow(arrows[0]), run_time=1.5)
            self.cue(1, 0.2)  # "a long chain of simple sums, with millions of adjustable numbers"
            self.play(Transform(net, renudge(net, 1)), run_time=1.6)
            self.cue(2, 0.2)  # "In training, it's shown recordings together with what was said"
            self.play(FadeOut(small), GrowArrow(arrows[1]), run_time=0.5)
            for i, (img, txt) in enumerate(pairs):
                img.move_to([-5.2, -0.3, 0])
                txt.move_to([4.9, -0.3, 0])
                self.play(FadeIn(img), FadeIn(txt), run_time=0.45)
                self.play(Transform(net, renudge(net, 10 + i)), run_time=0.9)
                self.play(FadeOut(img), FadeOut(txt), run_time=0.35)
            self.play(FadeOut(arrows[1]), run_time=0.3)
        # Whisper: a listening half and a writing half.
        name = tex("Whisper", 34, YELLOW).move_to([-0.4, 2.0, 0])
        split = Line([-0.4, -2.1, 0], [-0.4, 1.5, 0], stroke_width=2, color=GREY_D)
        small = spectrogram(2.6, 1.3).move_to([-5.2, -0.3, 0])
        written = VGroup(*[arabic(w, 32) for w, _, _ in basmala_words()]).arrange(LEFT, buff=0.22).move_to([4.9, -0.3, 0])
        with self.voice("whisper1"):
            self.play(FadeIn(name), FadeIn(small), run_time=1.0)
            self.cue(1, 0.2)  # "Its first half listens to the picture."
            self.play(Create(split), net[1].animate.set_color(YELLOW), net[2].animate.set_color(YELLOW),
                      run_time=1.0)
            self.cue(2, 0.2)  # "Its second half writes down what was said, one word at a time."
            self.play(net[4].animate.set_color(YELLOW), net[5].animate.set_color(YELLOW), GrowArrow(arrows[1]), run_time=0.8)
            for w in written:
                self.play(FadeIn(w, shift=0.1 * LEFT), run_time=0.5)
        # The phone's version.
        phone = phone_icon(2.2, 3.8).move_to([-0.6, -0.3, 0])
        with self.voice("whisper2"):
            self.play(VGroup(net, split).animate.scale(0.33).move_to([-0.6, -0.3, 0]), FadeIn(phone),
                      name.animate.move_to([-0.6, 1.95, 0]), FadeOut(arrows), FadeOut(small), FadeOut(written), run_time=1.4)
            self.cue(1, 0.2)  # "Out of the box it barely understands recitation, so it was trained further"
            crops = [mel_crop(a, b, 0.75) for _, a, b in basmala_words()] * 2
            for i, c in enumerate(crops[:7]):
                self.play(FadeIn(c.move_to([3.4 + 0.14 * i, 0.6 - 0.14 * i, 0])), run_time=0.22)
            self.play(Transform(net, renudge(net, 30)), run_time=1.0)
        # How often each one hears, at his own speed: Whisper's 6 s once a second, the letters' 2 s ten
        # times a second.
        z = Recording(0.0, 22.0, x0=-5.0, x1=6.6, y=0.0, h=1.2)  # room on the left for the names
        whis, letters = Windows(z, "whisper"), Windows(z, "letters")
        self.follow(z, whis)
        gone = [m for m in self.mobjects if m is not self.top and m is not self.clock]
        for m in gone:
            m.clear_updaters()
        with self.voice("listen1"):
            self.play(*[FadeOut(m) for m in gone], FadeIn(z), FadeIn(whis), run_time=0.9)
            self.at_speed(self.clock.get_value() + self.until(1) - 0.2, self.until(1) - 0.2, first=False)
            self.follow(letters)
            self.run(self.clock.get_value() + 0.4, 0.4, FadeIn(letters))
            self.at_speed(min(17.4, self.clock.get_value() + self.left_in(2)), self.left_in(2), first=False)  # not into "Allāhumma innī"
        self.clear(keep=[self.top])

    # -- 4. which du'a -------------------------------------------------------------------------------
    def finding(self):
        self.ensure_top()
        heard, field = Heard(), Field()
        with self.voice("find1"):
            self.play(self.clock.animate.set_value(3.2), run_time=1.0, rate_func=smooth)  # back to the basmala
            self.top.window = "whisper"
            self.follow(heard, field)
            self.play(FadeIn(heard), FadeIn(field), run_time=0.8)
            self.at_speed(17.3, self.until(3))  # at his speed, reaching 17 s as "After seventeen seconds" starts
        # His "Allāhumma innī", then what it does: Whisper's update at 21.2 s has it.
        self.wait(0.3)
        self.phrase("inni")
        with self.voice("find2"):
            self.run(21.25, 0.6, rate=smooth)
        share = Share()
        self.follow(share)
        # His "aftatiḥu th-thanā'a", from where it starts; Whisper's update at 23.2 s has it.
        self.run(PHRASES["aftatihu"][0] - 0.05, 0.5, FadeIn(share), rate=smooth)
        self.phrase("aftatihu")
        with self.voice("find3"):
            self.cue(1)
            self.run(23.3, 0.2)
            field.n_labels, field.k = 1, -2
            self.wait(self.left_in(2) - 0.3)
        # The real page, from where he is: its name comes up.
        d = self.shot("found")
        self.run(SHOTS["found"][0] + d, d)
        self.top.window = None
        self.clear(keep=[self.top])

    # -- 5. which word -------------------------------------------------------------------------------
    def following(self):
        self.ensure_top()
        if self.clock.get_value() < 25.0:
            self.clock.set_value(27.5)
        page = Page()
        self.follow(page)
        with self.voice("word1"):
            self.play(FadeIn(page), run_time=0.8)
            self.at_speed(min(35.0, self.clock.get_value() + self.left_in(2)), self.left_in(2), first=False)
        # The letter model: Whisper's listening half, one layer, a probability per letter per 20 ms.
        t = self.clock.get_value()
        net = network(-4.6, -0.6, w=3.0, h=2.0)
        cut = Line([-4.6, -1.8, 0], [-4.6, 0.6, 0], stroke_width=2, color=GREY_D)
        layer = Rectangle(width=0.16, height=2.0, stroke_width=0, fill_color=TEAL, fill_opacity=0.9).move_to([-2.55, -0.6, 0])
        grid, used = letter_grid(t, 6.6, 2.8)
        grid.move_to([2.15, -0.6, 0])
        n_rows = len(used) + 1
        rows = VGroup(Dot([grid.get_right()[0] + 0.28, grid.get_top()[1] - 0.5 * grid.height / n_rows, 0], radius=0.03,
                          color=GREY_B))
        for i, c in enumerate(used):
            yy = grid.get_top()[1] - (i + 1.5) * grid.height / n_rows
            rows.add(arabic(LETTERS[c], 22, GREY_B).move_to([grid.get_right()[0] + 0.28, yy, 0]))
        teacher = network(2.15, 1.75, w=5.6, h=0.75, sizes=(7, 10, 10, 10, 10, 7), seed=5)
        teacher[0].set_stroke(opacity=0.12)
        for col in teacher[1:]:
            col.set_opacity(0.3)
        with self.voice("ctc1"):
            self.hide(page, others=[FadeIn(net[0]), *[FadeIn(c) for c in net[1:]]], run_time=1.0)
            self.play(Create(cut), *[c.animate.set_color(TEAL) for c in net[1:3]], run_time=0.8)
            self.play(*[FadeOut(c) for c in net[4:]], run_time=0.8)
            self.play(FadeIn(layer), run_time=0.6)
            self.cue(1, 0.2)  # "For every fiftieth of a second ... a probability for each Arabic letter"
            self.play(FadeIn(grid), FadeIn(rows), run_time=1.6)
            self.cue(2, 0.2)  # "It learned by copying a far bigger model"
            self.play(FadeIn(teacher), run_time=1.2)
            self.play(grid.animate.set_opacity(0.55), run_time=0.6)
            self.play(grid.animate.set_opacity(1.0), run_time=0.6)
        letters = Letters()
        with self.voice("ctc2"):
            self.play(FadeOut(teacher), FadeOut(net), FadeOut(cut), FadeOut(layer), FadeOut(rows), FadeOut(grid),
                      run_time=1.0)
            self.show(letters, page, run_time=0.6)
            self.top.window = "letters"
            self.at_speed(37.2, self.left_in(2))
        # Every line's probability, and the moves: drawn large on a few lines, then back to the reading.
        lines = Lines()
        self.follow(lines)
        s = line_of(self.clock.get_value())
        rows, free, costly = ladder(s, s - 2, s + 3)
        with self.voice("lines1"):
            self.play(FadeIn(lines), run_time=1.0)
            self.cue(1, 0.5)  # "Carrying on costs nothing."
            self.hide(page, letters, others=[FadeIn(rows)], run_time=0.5)
            self.play(Create(free), run_time=0.8)
            self.cue(2, 0.2)  # "Saying the line again, going back a few lines, or skipping ahead ... each costs a little"
            self.play(Create(costly, lag_ratio=0.3), run_time=1.8)
            self.cue(3, 0.4)
            self.play(FadeOut(rows), FadeOut(free), FadeOut(costly), run_time=0.5)
            self.show(page, letters, run_time=0.4)
            self.run(46.4, 46.4 - self.clock.get_value())  # at his speed to the pause, even past the sentence
        self.views = (page, letters, lines)

    # -- 6. what readers do ------------------------------------------------------------------------
    def readers(self):
        self.ensure_top()
        if not hasattr(self, "views"):
            self.clock.set_value(46.4)
            self.views = self.follow(Page(), Letters(), Lines())
            self.add(*self.views)
            self.top.window = "letters"
        page, letters, lines = self.views
        # The pause, at his speed: silence, then him starting line 7.
        with self.voice("pause1"):
            self.at_speed(52.3, self.left_in(1), first=False)
        self.run(54.6, 2.3)
        with self.voice("pause2"):
            self.at_speed(54.6 + self.left_in(0), self.left_in(0))
        # On to the talk, quickly: the letters would only flicker.
        self.hide(letters, run_time=0.4)
        self.run(81.5, 4.0, rate=smooth)
        talk = Lines(talk=True)
        lines.clear_updaters()
        self.show(letters, talk, others=[FadeOut(lines)], run_time=0.8)
        # The talk as it happened, then again, slowly.
        self.run(88.6, 88.6 - 81.5)
        self.wait(0.4)
        self.run(81.5, 1.2, rate=smooth)
        with self.voice("talk1"):
            self.run(83.8, self.until(2))
            self.run(85.2, self.left_in(2))
            self.run(88.0, self.left_in(3))
        # Whisper's window is still talk: the app wonders if this is a du'a it doesn't have.
        heard = Heard(pos=(-1.9, 0.45, 0), size=36)
        share = Share(x=-6.2, y0=-2.3, hgt=2.6)
        ring = VGroup(Circle(radius=0.5, stroke_color=GREY_B, stroke_width=2))
        fill = Circle(radius=0.5, stroke_width=0, fill_color=GREY_B, fill_opacity=0.8)
        q = tex("?", 34, BLACK)
        ring.add(fill, q).move_to([-4.6, -1.65, 0])

        def null_at(m):
            k = update(self.clock.get_value())
            v = M["updates"][k]["null"] if k >= 0 else 0.0
            m[1].scale_to_fit_width(max(1.0 * np.sqrt(v), 0.001)).move_to(m[0])
            m[1].set_fill(opacity=0.8 if v > 0.01 else 0.0)

        self.follow(heard, share)
        ring.add_updater(null_at)
        self.top.window = "whisper"
        with self.voice("drop1"):
            self.hide(letters, others=[FadeIn(heard), FadeIn(share), FadeIn(ring)], run_time=0.8)
            self.run(88.15, self.until(1))
            self.run(89.4, self.until(2))  # the shot: the real page holds line 10 while the doubt lasts
            d = self.shot("drop")
            self.run(SHOTS["drop"][0] + d, d)
        self.top.window = "letters"
        for m in (heard, share, ring):
            m.clear_updaters()
        self.play(FadeOut(heard), FadeOut(share), FadeOut(ring), run_time=0.6)
        self.run(104.4, 2.0, rate=smooth)
        self.show(letters, run_time=0.4)
        # The go-back as it happened, then again, slowly.
        self.run(109.4, 5.0)
        self.wait(0.4)
        self.run(105.6, 1.2, rate=smooth)
        arcs = moves(12, {13: 4.0, 10: 2.0})
        with self.voice("back1"):
            self.run(107.1, self.until(1))
            self.play(Create(arcs), run_time=0.8)
            self.run(107.45, 1.5)  # line 13 at 97%, the highlight held on line 12
            self.cue(3, 0.2)
            self.run(107.9, 1.2)  # line 10's letters: line 10
            self.play(FadeOut(arcs), run_time=0.5)
            self.run(108.6, self.left_in(3))
        self.views = (page, letters, talk)

    # -- 7. the rest, and beyond this reading ----------------------------------------------------------
    def ending(self):
        self.ensure_top()
        if not hasattr(self, "views"):
            self.clock.set_value(108.0)
            self.views = self.follow(Page(), Letters(), Lines(talk=True))
            self.add(*self.views)
        page, letters, talk = self.views
        with self.voice("end1"):
            self.hide(letters, run_time=0.4)
            self.run(155.7, self.left_in(0), rate=smooth)
        self.top.window = None
        self.clear(keep=[self.top])

        def rows(items, y0, color):
            g = VGroup()
            for i, (name, v) in enumerate(items):
                yy = y0 - i * 0.5
                g.add(tex(name, 28, color).move_to([-5.6, yy, 0], aligned_edge=LEFT),
                      tex(v + r"\%", 28, color).move_to([4.6, yy, 0], aligned_edge=RIGHT))
            return g

        good, weak = rows(RESULTS, 2.15, WHITE), rows(LIMITS, -0.65, GREY_B)
        with self.voice("held1"):
            self.cue(1, 0.2)
            self.play(FadeIn(good, lag_ratio=0.15), run_time=2.0)
            self.cue(2, 0.2)
            self.play(FadeIn(weak[:6], lag_ratio=0.15), run_time=1.5)
            self.cue(3, 0.2)
            self.play(FadeIn(weak[6:], lag_ratio=0.15), run_time=0.8)
        self.wait(1.0)
        self.clear(keep=[self.top])
        d = self.shot("end")
        self.run(SHOTS["end"][0] + d, d)
        self.top.clear_updaters()
        self.play(FadeOut(self.top), run_time=1.2)
        self.wait(0.5)

