"""How the app finds a du'a and then follows it, from the microphone to the highlighted word: a
narrated walkthrough for someone who knows nothing about it yet (Manim Community + LaTeX, voiced by
OmniVoice in an Arabic-accented voice).

Two models listen at once, one per question. Whisper, every second or two, turns the last 6 s into
text, which is matched against every du'a: that finds the du'a. The letter model, ten times a
second, hears letters, and the stream decoder follows the reader through the du'a with them: that
places the line and the word. The video gives each its share, on real data:

- Whisper's part: Dua Tawassul recited by a held-out reciter (Hussein Ghareeb) around t = 296 s,
  dumped by docs/anim/dump_explainer_data.py.
- The decoder's part: the scenario bench's go-back item for the same recording (he reads lines
  39-43, then goes back to line 41), replayed by docs/anim/dump_follow_data.py exactly as the
  page's stream decoder runs.

What is said is in narration.py, which voices it first. Between the steps, the real app is shown
following the go-back recording (film.mjs, one phone: see SHOTS); publish_explainer.py lays those
shots in.

    pip install manim soundfile               # plus a LaTeX install with dvisvgm
    manim -qh --fps 30 docs/anim/explainer.py Explainer
    python docs/anim/publish_explainer.py     # shots, level the voice, subtitles -> docs/explainer.mp4
"""
from __future__ import annotations

import html
import importlib.util
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import manimpango
import numpy as np
from manim import (
    BLACK, BLUE, DOWN, GREY_B, GREY_D, LEFT, ORIGIN, PI, RIGHT, TEAL, UP, WHITE, YELLOW, Arc, Arrow, Brace, Circle,
    Create, CurvedArrow, DashedLine, Dot, FadeIn, FadeOut, GrowArrow, GrowFromEdge, Group, ImageMobject, Indicate,
    Line, MarkupText, MathTex, MovingCameraScene, NumberLine, Rectangle, ReplacementTransform,
    RoundedRectangle, Square, SurroundingRectangle, Tex, Transform, UpdateFromAlphaFunc, VGroup, Write,
    config, linear,
)
from manim.constants import RESAMPLING_ALGORITHMS

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from narration import clip  # noqa: E402

# The app's own text normalization, loaded alone (the package imports its whole pipeline).
_spec = importlib.util.spec_from_file_location("dua_text", ROOT / "src" / "dua_recognition" / "text.py")
_text = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_text)
normalize, strip_diacritics = _text.normalize, _text.strip_diacritics

DATA = np.load(HERE / "explainer_data.npz")
META = json.loads((HERE / "explainer_meta.json").read_text(encoding="utf-8"))
FOLLOW = np.load(HERE / "explainer_follow.npz")
FMETA = json.loads((HERE / "explainer_follow.json").read_text(encoding="utf-8"))
manimpango.register_font(str(HERE / "fonts" / "Amiri-Regular.ttf"))

config.background_color = BLACK
# DUA_EXPLAINER_AUDIO=1: also play the real recitation (cut by the dump scripts into the ignored
# data/cache/media/). The published render has the narration only: the recitation belongs to
# DuaPlayer and its reciters.
MEDIA = ROOT / "data" / "cache" / "media"
AUDIO = os.environ.get("DUA_EXPLAINER_AUDIO") == "1"
# The real app on the go-back recording (dump_follow_data.py's explainer_back.wav), filmed by
# docs/demo/film.mjs from a cold start:
#   node docs/demo/capture.mjs --file data/cache/media/explainer_back.wav --start 0 --seconds 52 \
#        --out data/cache/media/explainer_runs/1-tawassul-back.json
#   node docs/demo/film.mjs --runs data/cache/media/explainer_runs --follow 44 --fps 30 \
#        --out data/cache/media/explainer_phone_back.mp4
# The page named the du'a 5.94 s in (after 3.4 s on line 67, which opens like line 39), went back to
# line 41 1.1 s after he did. The phone starts listening 5.83 s into the film (stage.html's opening):
# film second f is recording second f - PHONE_LEAD.
PHONE = MEDIA / "explainer_phone_back.mp4"
PHONE_LEAD = float(os.environ.get("EXPLAINER_PHONE_LEAD", "5.83"))
PHONE_FOUND = float(os.environ.get("EXPLAINER_PHONE_FOUND", str(5.83 + 5.94)))  # film second the du'a is named
# Where each shot starts, in recording seconds: following lines 41-43 (the opening), the cold start
# (after the du'a is named), going back from line 43 to 41 (after the decoder), and the end.
SHOTS = {"open": 15.5, "found": 0.0, "back": 32.0, "end": 39.5}
PRIOR, EVID, POST = BLUE, YELLOW, TEAL
SEGS = DATA["segs"]  # line ids 1..115
N_TEXTS = META["n_texts"]
LETTERS = {i + 1: chr(c) for i, c in enumerate(range(0x0621, 0x064B))}  # align._ALPHABET, inverted
# The decoder's scenes: these lines of Dua Tawassul, and the data's per-line columns for them.
FLINES = list(range(38, 46))
FCOL = {int(s): i for i, s in enumerate(FOLLOW["lines"])}
# Held-out voices on the scenario bench (docs/results/bench.md, test split; finding.md for the du'a).
RESULTS = [("du'a named within 10 s", 90), ("highlight on the reader's line", 87), ("on the exact word", 67),
           ("a go-back, skip or jump followed within 3 s", 72), ("held through pauses, talk and salawat", 88)]
CTC_COLORS = [(0, 0, 0), (0, 70, 70), (0, 160, 150), (120, 240, 220), (255, 255, 255)]


def arabic(s: str, size: float = 34, color=WHITE) -> MarkupText:
    # MarkupText, not Text: Text drops short Arabic strings (a letter, a word) whose shaped
    # glyphs don't match its characters one to one.
    return MarkupText(html.escape(s, quote=False), font="Amiri", font_size=size, color=color)


def word_groups(line: MarkupText, n: int) -> list[VGroup]:
    """The glyphs of each word of a typeset Arabic line, first word (rightmost) first.

    Glyphs are clustered by the gaps between them: a space is wider than the gap a
    non-joining letter leaves inside a word, so the smallest gap that leaves n clusters
    splits the line into its words, on the line's own baseline."""
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

    em = line.height
    for gap in np.linspace(0.02, 0.6, 59) * em:
        cs = clusters(gap)
        if len(cs) == n:
            return [VGroup(*[glyphs[i] for i in c]) for c in reversed(cs)]
    raise ValueError(f"could not split the line into {n} words")


def short_name(name: str, limit: int = 30) -> str:
    """A du'a's English name for a label: honorifics dropped, long ones cut."""
    name = re.sub(r"\s*\((a\.?s\.?|atfs|s\.?a\.?w?\.?|pbuh)\)", "", name, flags=re.I).strip()
    name = re.sub(r"^Dua - ", "Dua ", name)
    return name if len(name) <= limit else name[: limit - 1].rstrip() + "..."


def end_costs(h: str, r: str) -> np.ndarray:
    """Semi-global edit distance of h ending at each letter of r (align.semiglobal_end_costs)."""
    prev = np.zeros(len(r) + 1, dtype=int)
    for i, c in enumerate(h, start=1):
        cur = np.empty_like(prev)
        cur[0] = i
        for j in range(1, len(r) + 1):
            cur[j] = min(prev[j - 1] + (c != r[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[1:]


def align(h: str, r: str) -> list[tuple[int | None, int | None]]:
    """Global edit-distance alignment of h against r: (h index | None, r index | None) pairs."""
    d = np.zeros((len(h) + 1, len(r) + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(len(h) + 1), np.arange(len(r) + 1)
    for i in range(1, len(h) + 1):
        for j in range(1, len(r) + 1):
            d[i, j] = min(d[i - 1, j - 1] + (h[i - 1] != r[j - 1]), d[i - 1, j] + 1, d[i, j - 1] + 1)
    i, j, out = len(h), len(r), []
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (h[i - 1] != r[j - 1]):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif j and d[i, j] == d[i, j - 1] + 1:
            out.append((None, j - 1))
            j -= 1
        else:
            out.append((i - 1, None))
            i -= 1
    return out[::-1]


def colormap(v: np.ndarray, stops) -> np.ndarray:
    """v in [0, 1] -> RGB uint8 through evenly spaced colour stops."""
    stops = np.asarray(stops, dtype=float)
    x = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


MAGMA = [(0, 0, 4), (40, 17, 90), (114, 31, 129), (183, 55, 121), (241, 96, 93), (254, 176, 120), (252, 253, 191)]


def mic_icon() -> VGroup:
    """A microphone: capsule, the holder around it, a stand."""
    return VGroup(RoundedRectangle(corner_radius=0.16, width=0.32, height=0.62, stroke_color=WHITE, stroke_width=2.5)
                  .move_to([0, 0.18, 0]),
                  Arc(radius=0.27, start_angle=np.pi, angle=np.pi, stroke_color=WHITE, stroke_width=2.5).move_to([0, -0.1, 0]),
                  Line([0, -0.37, 0], [0, -0.58, 0], stroke_width=2.5), Line([-0.18, -0.58, 0], [0.18, -0.58, 0], stroke_width=2.5))


def spectrogram(w: float, h: float) -> ImageMobject:
    mel = DATA["mel"][::-1].astype(float) / 255
    img = ImageMobject(colormap(mel ** 1.4, MAGMA))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    return img.stretch_to_fit_width(w).stretch_to_fit_height(h)


def letter_grid(w: float, h: float) -> ImageMobject:
    """The letter model's frames for "ishfa' lana 'inda": one row per letter, one column per 20 ms."""
    heat = np.stack([DATA["ctc"].astype(float)[:, c] for c in DATA["ctc_letters"]])
    img = ImageMobject(colormap(heat ** 0.6, CTC_COLORS))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return img.stretch_to_fit_width(w).stretch_to_fit_height(h)


def follow_at(t: float) -> int:
    """The decoder step on screen at item second t."""
    return max(0, int(np.searchsorted(FOLLOW["t"], t, side="right")) - 1)


def word_line(w: int) -> tuple[int, int]:
    """(line, word within the line) of a word of the du'a (index from its first word)."""
    for ln, ws in FMETA["line_words"].items():
        if w in ws:
            return int(ln), ws.index(w)
    return -1, -1


def reading_at(t: float) -> tuple[int, int] | None:
    """(line, word within it) the reader is on at item second t: the last word he started."""
    said = [s for s in FMETA["said"] if s[2] <= t]
    if not said:
        return None
    ln, w, _, _ = said[-1]
    return int(ln), FMETA["line_words"][str(ln)].index(w)


class Explainer(MovingCameraScene):
    def construct(self):
        self._subs, self._shots = [], []
        self.intro()
        self.sound()
        self.whisper()
        self.matching()
        self.which_dua()
        self.letters()
        self.decoder()
        self.whole()
        self.results()
        self.ending()
        self.write_subtitles()
        (Path(config.media_dir) / "explainer_shots.json").write_text(json.dumps(self._shots, indent=1), encoding="utf-8")

    # -- helpers -------------------------------------------------------------------
    def sound_at_now(self, path: Path):
        # Not Scene.add_sound: it does nothing right after a play call served from Manim's
        # cache (an identical pause on a blank screen, say), and the line would go missing.
        self.renderer.file_writer.add_sound(str(path), self.renderer.time)

    @contextmanager
    def voice(self, key: str, pad: float = 0.4):
        """Speak one narration line; the block's animations play under it, and the scene
        waits for the line to finish."""
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

    def cue_end(self, i: int, lead: float):
        """Wait until `lead` seconds before sentence i of the line being spoken ends."""
        start, parts = self._cue
        t = start + parts[min(i, len(parts) - 1)]["end"] - lead
        if t > self.renderer.time + 0.02:
            self.wait(t - self.renderer.time)

    def left_in(self, i: int) -> float:
        """Seconds from now to the end of sentence i of the line being spoken."""
        start, parts = self._cue
        return start + parts[min(i, len(parts) - 1)]["end"] - self.renderer.time

    def say(self, key: str, pad: float = 0.4):
        with self.voice(key, pad):
            pass

    def recite(self, name: str) -> bool:
        """The real recitation, in the local render only."""
        if AUDIO and (MEDIA / name).exists():
            import soundfile as sf

            self.sound_at_now(MEDIA / name)
            self.wait(sf.info(MEDIA / name).duration + 0.3)
            return True
        return False

    def shot(self, at: float, dur: float, gain: float = -2.0, talk: float | None = None):
        """The real app over the frame, from recording second `at` for `dur` seconds (the scene goes
        on underneath: keep it dark). publish_explainer.py lays the picture in; with the recitation,
        its sound plays here, 10 dB down from `talk` seconds in, where the narration comes in."""
        at = at + PHONE_LEAD  # film seconds
        start = self.renderer.time
        self._shots.append({"start": round(start, 3), "from": round(at, 3), "dur": round(dur, 3)})
        if AUDIO and PHONE.exists():
            import soundfile as sf

            wav = PHONE.with_suffix(".wav")
            if not wav.exists() or wav.stat().st_mtime < PHONE.stat().st_mtime:
                import subprocess

                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(PHONE), "-vn", "-ac", "1", str(wav)], check=True)
            sr = sf.info(wav).samplerate
            y, _ = sf.read(wav, dtype="float32", start=int(at * sr), frames=int(dur * sr))
            n = int(0.5 * sr)
            y[:n] *= np.linspace(0, 1, n, dtype=np.float32)
            y[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
            if talk is not None:
                duck = np.ones(len(y), dtype=np.float32)
                k = int((talk - 0.4) * sr)
                duck[k:] = 10 ** (-10 / 20)
                duck[k : k + int(0.4 * sr)] = np.linspace(1, 10 ** (-10 / 20), len(duck[k : k + int(0.4 * sr)]))
                y *= duck
            path = MEDIA / f"explainer_shot{len(self._shots)}.wav"
            sf.write(path, y, sr)
            self.renderer.file_writer.add_sound(str(path), start, gain)

    def write_subtitles(self):
        """When each sentence of the narration is said, for publish_explainer.py's subtitles."""
        subs = [{"start": round(a, 3), "end": round(b, 3), "text": text, "lang": lang} for a, b, text, lang in self._subs]
        (Path(config.media_dir) / "explainer_voice.json").write_text(json.dumps(subs, ensure_ascii=False, indent=0),
                                                                     encoding="utf-8")

    def clear(self, keep=()):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=0.9)
        self.wait(0.3)

    def page(self, which: dict, size: float = 27) -> tuple[VGroup, VGroup]:
        """Lines of Dua Tawassul, set right-aligned like a printed page, with line numbers."""
        lines = VGroup(*[arabic(t, size) for t in which.values()]).arrange(DOWN, buff=0.16, aligned_edge=RIGHT)
        nums = VGroup(*[Tex(str(k), font_size=24, color=GREY_B).next_to(ln, RIGHT, buff=0.25)
                        for k, ln in zip(which, lines)])
        return lines, nums

    def mini_wave(self, k: int, color, w: float = 1.5, h: float = 0.26) -> VGroup:
        """A small waveform: a stretch of the recording's loudness, a different stretch for each k."""
        env, n = DATA["env"], 60
        a = env[(k * 41) % (len(env) - n):][:n]
        return VGroup(*[Line([x, -h / 2 * v - 0.01, 0], [x, h / 2 * v + 0.01, 0], stroke_width=1.6, color=color)
                        for x, v in zip(np.linspace(-w / 2, w / 2, n), a / a.max())])

    def glow(self, mob, color=TEAL, opacity=0.16):
        return SurroundingRectangle(mob, color=color, buff=0.08, stroke_width=0, fill_opacity=opacity)

    def follow_words(self, lines: VGroup, which: list[int], until: float, step: float = 0.42):
        """Light the words of the given lines one after another, with a glow on the line, until
        the scene clock reaches `until`."""
        glow = None
        texts = list(META["lines"].values())
        for li in which:
            try:
                ws = word_groups(lines[li], len(texts[li].split()))
            except ValueError:
                ws = [lines[li]]
            g = self.glow(lines[li])
            self.play(FadeIn(g) if glow is None else Transform(glow, g), run_time=0.4)
            glow = glow or g
            for w in ws:
                if self.renderer.time + step > until:
                    return glow
                self.play(w.animate.set_color(TEAL), run_time=step * 0.5)
                self.wait(step * 0.5)
            self.play(lines[li].animate.set_color(WHITE), run_time=0.2)
        return glow

    def node(self, label: str, icon, x: float, y: float, w: float = 2.3, h: float = 1.55) -> Group:
        """A box with a small picture and its name under it."""
        r = RoundedRectangle(corner_radius=0.15, width=w, height=h, stroke_color=WHITE, stroke_width=2).move_to([x, y, 0])
        if icon.width > w - 0.5:
            icon.scale_to_fit_width(w - 0.5)
        if icon.height > h - 0.95:
            icon.scale_to_fit_height(h - 0.95)
        icon.move_to(r.get_center() + 0.2 * UP)
        return Group(r, icon, Tex(label, font_size=24).move_to(r.get_center() + (h / 2 - 0.3) * DOWN))

    # -- 1. what it's for: two questions, two listeners ----------------------------------------------
    def intro(self):
        # The real app first, following him; the narration comes in over it.
        lead = 2.5
        self.shot(SHOTS["open"], lead + clip("open1")[2][0]["end"] + 0.9, talk=lead)
        self.wait(lead)
        lines, nums = self.page(META["lines"])
        ids = list(map(int, META["lines"]))
        page = VGroup(lines, nums).scale_to_fit_height(4.9).move_to(0.2 * LEFT)
        # Where these eight lines sit in the whole du'a.
        track = Line(2.5 * UP, 2.5 * DOWN, color=GREY_D, stroke_width=3).next_to(page, RIGHT, buff=0.55)
        a, b = (ids[0] - 1) / META["n_lines"], ids[-1] / META["n_lines"]
        thumb = Line(track.point_from_proportion(a), track.point_from_proportion(b), color=GREY_B, stroke_width=7)
        ends = VGroup(Tex("1", font_size=22, color=GREY_B).next_to(track, UP, buff=0.12),
                      Tex(str(META["n_lines"]), font_size=22, color=GREY_B).next_to(track, DOWN, buff=0.12))
        with self.voice("open1"):
            self.cue_end(0, -0.9)  # the shot is over
            self.play(FadeIn(page, lag_ratio=0.1), run_time=1.5)
            self.play(Create(track), Create(thumb), FadeIn(ends), run_time=1.2)

        # Question one: which of all the texts is this? Question two: where in it, to the word?
        book = VGroup(page, track, thumb, ends)
        cells = VGroup(*[Square(0.16, stroke_width=0, fill_color=GREY_B, fill_opacity=0.35) for _ in range(N_TEXTS)])
        cells.arrange_in_grid(cols=29, buff=0.06).move_to(3.3 * RIGHT)
        me = cells[META["dua_index"]]
        with self.voice("open2") as d:
            self.cue(2)  # "Which du'a is this, out of 522"
            self.play(book.animate.scale(0.62).move_to(3.6 * LEFT), run_time=1.0)
            self.play(FadeIn(cells, lag_ratio=0.002), run_time=1.6)
            self.play(me.animate.set_fill(TEAL, opacity=1).scale(1.6), run_time=0.8)
            link = DashedLine(me.get_left(), book.get_right() + 0.1 * RIGHT, color=TEAL, stroke_width=2)
            self.play(Create(link), run_time=0.6)
            self.cue(3)  # "And where in it is he right now, down to the word?"
            self.play(FadeOut(cells), FadeOut(link), book.animate.scale(1 / 0.62).move_to(0.2 * LEFT), run_time=1.0)
            glow = self.follow_words(lines, [5, 6], until=self.renderer.time + self.left_in(3) + 0.3)
        self.play(FadeOut(glow), lines.animate.set_color(WHITE), run_time=0.4)
        self.clear()

        # Two listeners, one per question, at their own pace.
        mic = mic_icon().move_to([-6.0, 0, 0])
        whisper = RoundedRectangle(corner_radius=0.15, width=2.4, height=1.1, stroke_color=WHITE, stroke_width=2).move_to([-2.6, 1.55, 0])
        w_name = Tex("Whisper", font_size=32).move_to(whisper)
        small = RoundedRectangle(corner_radius=0.12, width=1.6, height=0.8, stroke_color=WHITE, stroke_width=2).move_to([-2.6, -1.55, 0])
        s_name = Tex("letter model", font_size=24).move_to(small)
        feeds = VGroup(*[Arrow(mic.get_right(), box.get_left(), buff=0.2, stroke_width=3, color=GREY_B,
                               max_tip_length_to_length_ratio=0.08) for box in (whisper, small)])
        # How often each one hears: the same 4 s, once a second and ten times a second.
        x0, x1 = -1.0, 3.0

        def ticks(y: float, every: float, h: float, color) -> VGroup:
            return VGroup(*[Line([x, y - h / 2, 0], [x, y + h / 2, 0], stroke_width=3 if every >= 1 else 1.6, color=color)
                            for x in np.arange(x0, x1 + 1e-6, every * (x1 - x0) / 4)])

        slow, fast = ticks(1.55, 1.0, 0.7, YELLOW), ticks(-1.55, 0.1, 0.4, TEAL)
        q1 = VGroup(*[Square(0.09, stroke_width=0, fill_color=GREY_B, fill_opacity=0.35) for _ in range(N_TEXTS)])
        q1.arrange_in_grid(cols=29, buff=0.03).move_to([4.9, 1.55, 0])
        q1[META["dua_index"]].set_fill(TEAL, opacity=1).scale(1.8)
        row = arabic(META["line38"]["arabic"], 26).move_to([4.9, -1.55, 0])
        if row.width > 3.0:
            row.scale_to_fit_width(3.0)
        q2 = VGroup(self.glow(row), row)
        try:
            word_groups(row, 8)[2].set_color(TEAL)
        except ValueError:
            pass
        with self.voice("open3"):
            self.play(Create(mic), run_time=0.6)
            self.play(GrowArrow(feeds[0]), FadeIn(whisper), FadeIn(w_name), GrowArrow(feeds[1]), FadeIn(small), FadeIn(s_name),
                      run_time=1.0)
            self.cue(1)  # "Whisper ... every second or two ... looked for in every du'a"
            self.play(Create(slow, lag_ratio=1.0), run_time=2.4, rate_func=linear)
            self.play(FadeIn(q1, lag_ratio=0.001), run_time=1.0)
            self.cue(2)  # "A much smaller letter model hears letters ten times a second"
            self.play(Create(fast, lag_ratio=1.0), run_time=2.4, rate_func=linear)
            self.play(FadeIn(q2), run_time=0.8)
        self.clear()

        # The recording: line 38, five minutes in.
        lines, nums = self.page(META["lines"])
        page = VGroup(lines, nums).scale_to_fit_height(4.9).move_to(ORIGIN)
        with self.voice("open4"):
            self.play(FadeIn(page), run_time=0.8)
            self.cue(1)  # "at line 38"
            g = self.glow(lines[-1])
            self.play(FadeIn(g), *[w.animate.set_color(TEAL) for w in word_groups(lines[-1], 8)[:4]], run_time=1.2)
        self.recite("explainer_hook.wav")
        self.clear()

    # -- 2. which du'a: sound -> a picture ---------------------------------------------------------
    def sound(self):
        env = DATA["env"]
        x0, x1, y = -6.4, 6.4, 1.7
        xs = np.linspace(x0, x1, len(env))
        wave = VGroup(*[Line([x, y - 0.75 * a - 0.02, 0], [x, y + 0.75 * a + 0.02, 0], stroke_width=2.2, color=BLUE)
                        for x, a in zip(xs, env)])
        # The spectrogram of the newest 6 s: the window everything below is about.
        img = spectrogram(9.6, 3.0).move_to([0.3, -0.95, 0])
        xw = x1 - (x1 - x0) * 6 / 16
        win = Rectangle(width=x1 - xw + 0.08, height=1.75, stroke_color=YELLOW, stroke_width=3).move_to([(xw + x1) / 2, y, 0])
        rays = VGroup(DashedLine(win.get_corner(DOWN + LEFT), img.get_corner(UP + LEFT), color=GREY_D, stroke_width=1.5),
                      DashedLine(win.get_corner(DOWN + RIGHT), img.get_corner(UP + RIGHT), color=GREY_D, stroke_width=1.5))
        t_arrow = Arrow(img.get_corner(DOWN + LEFT) + 0.25 * DOWN, img.get_corner(DOWN + RIGHT) + 0.25 * DOWN, buff=0,
                        stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.02)
        f_arrow = Arrow(img.get_corner(DOWN + LEFT) + 0.25 * LEFT, img.get_corner(UP + LEFT) + 0.25 * LEFT, buff=0,
                        stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.06)
        t_lbl = Tex("time", font_size=24, color=GREY_B).next_to(t_arrow, RIGHT, buff=0.12)
        f_lbl = Tex("pitch", font_size=24, color=GREY_B).rotate(np.pi / 2).next_to(f_arrow, LEFT, buff=0.08)
        with self.voice("hear1"):
            self.play(Create(wave, lag_ratio=0.01), run_time=2.0)
            self.play(Create(win), run_time=0.8)  # "called a window"
            self.play(Create(rays), FadeIn(img, shift=0.2 * DOWN), run_time=1.4)
            self.cue(1)  # "Time runs left to right, pitch from low to high"
            self.play(Create(t_arrow), FadeIn(t_lbl), run_time=0.8)
            self.play(Create(f_arrow), FadeIn(f_lbl), run_time=0.8)
        self.play(FadeOut(VGroup(t_arrow, f_arrow, t_lbl, f_lbl, rays)), run_time=0.5)
        self.wave, self.win, self.img = wave, win, img

    # -- 3. Whisper: sound -> text ------------------------------------------------------------
    def whisper(self):
        img = self.img
        box = RoundedRectangle(corner_radius=0.2, width=3.0, height=2.2, stroke_color=WHITE, stroke_width=2).move_to([0, 0.9, 0])
        name = Tex("Whisper", font_size=36).next_to(box.get_top(), DOWN, buff=0.22)
        layers = [np.linspace(-0.55, 0.55, n) for n in (4, 5, 5, 4)]
        nodes = [[box.get_center() + np.array([-0.9 + 0.6 * k, yy - 0.3, 0]) for yy in ys] for k, ys in enumerate(layers)]
        edges = VGroup(*[Line(a, b, stroke_width=0.6, color=GREY_D) for k in range(3) for a in nodes[k] for b in nodes[k + 1]])
        dots = VGroup(*[Dot(p, radius=0.05, color=GREY_B) for col in nodes for p in col])
        net = VGroup(box, name, edges, dots)
        size = Tex("74 million parameters", font_size=24, color=GREY_B).next_to(box, DOWN, buff=0.2)
        # What it was trained further on: recited du'as, and synthetic voices of ordinary people.
        recited = VGroup(*[self.mini_wave(k, BLUE) for k in range(3)]).arrange(DOWN, buff=0.06)
        voices = VGroup(*[self.mini_wave(k + 3, c) for k, c in enumerate((TEAL, YELLOW, GREY_B))]).arrange(DOWN, buff=0.06)
        train = VGroup(VGroup(recited, Tex(r"100 hours of recited du'as", font_size=22, color=GREY_B)).arrange(DOWN, buff=0.15),
                       VGroup(voices, Tex(r"synthetic voices", font_size=22, color=GREY_B)).arrange(DOWN, buff=0.15))
        train[0].move_to([-2.6, -1.75, 0])
        train[1].move_to([2.6, -1.75, 0])
        feeds = VGroup(*[Arrow(g[0].get_top(), box.get_corner(DOWN + side) + 0.35 * -side, buff=0.12, stroke_width=2.5, color=GREY_B,
                               max_tip_length_to_length_ratio=0.12) for g, side in ((train[0], LEFT), (train[1], RIGHT))])
        with self.voice("asr1"):
            self.play(FadeOut(self.wave), FadeOut(self.win), img.animate.stretch_to_fit_width(4.2).stretch_to_fit_height(1.9)
                      .move_to([-4.6, 0.9, 0]), run_time=1.2)
            brace = Brace(img, UP, buff=0.08, color=GREY_B)
            six = Tex("6 seconds", font_size=26, color=GREY_B).next_to(brace, UP, buff=0.06)
            a1 = Arrow(img.get_right(), box.get_left(), buff=0.15, color=WHITE, stroke_width=3)
            self.play(GrowFromEdge(brace, DOWN), FadeIn(six), Create(a1), FadeIn(net), run_time=1.2)
            self.cue(1)  # "small enough to run on a phone"
            self.play(FadeIn(size), run_time=0.8)
            self.cue(2)  # "... trained further, on over 100 hours of recited du'as, and on synthetic voices"
            self.wait(min(2.5, max(0.0, self.left_in(2) - 4.0)))
            self.play(FadeIn(train[0], shift=0.15 * UP), GrowArrow(feeds[0]), run_time=1.0)
            self.wait(1.0)
            self.play(FadeIn(train[1], shift=0.15 * UP), GrowArrow(feeds[1]), run_time=1.0)

        heard = arabic(META["text"], 44, YELLOW).move_to([4.5, 1.05, 0])
        tr = Tex(r"\emph{y\=a waj\={\i}h `inda-ll\=ah}", font_size=28, color=YELLOW).next_to(heard, DOWN, buff=0.18)
        a2 = Arrow(box.get_right(), heard.get_left() + 0.1 * LEFT, buff=0.15, color=WHITE, stroke_width=3)
        with self.voice("asr3"):
            self.play(FadeOut(train), FadeOut(feeds), Create(a2), FadeIn(heard, shift=0.1 * RIGHT), run_time=1.0)
            self.play(FadeIn(tr), run_time=0.6)

        l38 = META["line38"]
        line = arabic(l38["arabic"], 40).move_to([0, -1.3, 0])
        words = word_groups(line, 8)
        tl = Tex(r"\emph{y\=a waj\={\i}han `inda-ll\=ah, ishfa` lan\=a `inda-ll\=ah}", font_size=28).next_to(line, DOWN, buff=0.2)
        en = Tex(r"O you who are honoured by God, intercede for us with God", font_size=24, color=GREY_B).next_to(tl, DOWN, buff=0.14)
        num = Tex("38", font_size=24, color=GREY_B).next_to(line, RIGHT, buff=0.3)
        inwin = Brace(VGroup(*words[:4]), UP, buff=0.12, color=YELLOW)
        with self.voice("asr4"):
            self.play(FadeOut(size), FadeIn(line), FadeIn(num), FadeIn(tl), FadeIn(en), run_time=1.2)
            self.cue(2)  # "Close, but not exact, and only half the line"
            self.play(Indicate(words[1], color=YELLOW, scale_factor=1.15), run_time=1.4)
            self.play(GrowFromEdge(inwin, DOWN), *[w.animate.set_color(YELLOW) for w in words[:4]], run_time=1.2)
        # A breath: the window once more, as he recited it (only the local render has his voice).
        if not self.recite("explainer_window.wav"):
            self.wait(0.6)
        self.clear()

    # -- 4. text -> where in every du'a -------------------------------------------------------------
    def letter_row(self, letters: str, cols: list[int], x_right: float, y: float, w: float, color=WHITE, size=30):
        """Letters in boxes, right to left, letter k in column cols[k]."""
        g = VGroup()
        for c, ch in zip(cols, letters):
            sq = Square(w, stroke_color=GREY_D, stroke_width=1.5).move_to([x_right - (c + 0.5) * w, y, 0])
            g.add(VGroup(sq, arabic(ch, size, color).move_to(sq)))
        return g

    def matching(self):
        half = " ".join(META["line38"]["arabic"].split()[:4])
        ref_txt, heard_txt = normalize(half).replace(" ", ""), normalize(META["text"]).replace(" ", "")
        ref = arabic(half, 52).move_to([0, 1.7, 0])
        heard = arabic(META["text"], 52, YELLOW).move_to([0, -1.1, 0])
        tags = VGroup(Tex("line 38", font_size=26, color=GREY_B).next_to(ref, LEFT, buff=0.8),
                      Tex("heard", font_size=26, color=GREY_B).next_to(heard, LEFT, buff=0.8))
        bare = arabic(strip_diacritics(half), 52).move_to(ref)
        pairs = align(heard_txt, ref_txt)
        w = 0.62
        xr = len(pairs) * w / 2
        ref_cols = [k for k, (_, j) in enumerate(pairs) if j is not None]
        heard_cols = [k for k, (i, _) in enumerate(pairs) if i is not None]
        ref_row = self.letter_row(ref_txt, ref_cols, xr, 1.2, w)
        heard_row = self.letter_row(heard_txt, heard_cols, xr, -0.6, w, YELLOW)
        # Spaces gone, the letters close up; the gap for the missing letter opens only when edits come in.
        ref_flat = self.letter_row(ref_txt, list(range(len(ref_txt))), xr, 1.2, w)
        heard_flat = self.letter_row(heard_txt, list(range(len(heard_txt))), xr, -0.6, w, YELLOW)
        links = VGroup(*[Line(ref_row[ref_cols.index(k)][0].get_bottom(), heard_row[heard_cols.index(k)][0].get_top(),
                              color=TEAL, stroke_width=2.5)
                         for k, (i, j) in enumerate(pairs) if i is not None and j is not None and heard_txt[i] == ref_txt[j]])
        gaps = [k for k, (i, _) in enumerate(pairs) if i is None]
        holes = VGroup(*[Square(w, stroke_color=YELLOW, stroke_width=2.5).move_to([xr - (k + 0.5) * w, -0.6, 0]) for k in gaps])
        cost = Tex(r"cost $=1$", font_size=40, color=YELLOW).next_to(heard_row, DOWN, buff=0.6)
        with self.voice("match1"):
            self.play(FadeIn(ref), FadeIn(heard), FadeIn(tags), run_time=1.0)
            self.cue(1)  # "Vowel marks, spellings and spaces are simplified on both sides."
            self.play(Transform(ref, bare), run_time=1.2)
            self.play(FadeOut(ref), FadeOut(heard), FadeIn(ref_flat, lag_ratio=0.04), FadeIn(heard_flat, lag_ratio=0.04),
                      tags[0].animate.next_to(ref_row, LEFT, buff=0.4), tags[1].animate.next_to(heard_row, LEFT, buff=0.4),
                      run_time=1.2)
            self.cue(2)  # "Then it counts the fewest letters to change, add, or remove"
            self.play(ReplacementTransform(ref_flat, ref_row), ReplacementTransform(heard_flat, heard_row), run_time=1.0)
            self.play(Create(links, lag_ratio=0.1), run_time=1.6)
            self.cue(3)  # "Here it's one, a missing alif."
            self.play(Create(holes), Indicate(ref_row[ref_cols.index(gaps[0])], color=YELLOW, scale_factor=1.3), run_time=1.0)
            self.play(Write(cost), run_time=0.8)
        self.play(FadeOut(VGroup(links, holes, cost, tags, ref_row, heard_row)), run_time=0.7)

        # Slide the heard text along lines 37-38 and note the cost of it ending at each letter.
        strip_txt = "".join(normalize(META["lines"][k]).replace(" ", "") for k in ("37", "38"))
        costs = end_costs(heard_txt, strip_txt)
        ws = 0.25
        xr = len(strip_txt) * ws / 2
        strip = self.letter_row(strip_txt, list(range(len(strip_txt))), xr, -0.2, ws, size=17)
        probe = self.letter_row(heard_txt, list(range(len(heard_txt))), 0, 0.75, ws, YELLOW, size=17)
        probe_w = len(heard_txt) * ws
        marks = VGroup(*[Tex(str(c), font_size=20, color=TEAL if c == costs.min() else GREY_B)
                         .move_to([xr - (k + 0.5) * ws, -0.75, 0]) for k, c in enumerate(costs)])
        for m in marks:
            m.set_opacity(0)

        def slide(p, a):
            e = a * (len(strip_txt) - 1)
            p.move_to([xr - (e + 0.5) * ws + probe_w / 2 - ws / 2, 0.75, 0])
            for k, m in enumerate(marks):
                m.set_opacity(1 if k <= e else 0)

        n37 = len(normalize(META["lines"]["37"]).replace(" ", ""))
        which = VGroup(Tex("line 37", font_size=22, color=GREY_B).move_to([xr - n37 * ws / 2, -1.2, 0]),
                       Tex("line 38", font_size=22, color=GREY_B).move_to([xr - (n37 + len(strip_txt)) * ws / 2, -1.2, 0]))
        split = DashedLine([xr - n37 * ws, 0.05, 0], [xr - n37 * ws, -1.3, 0], color=GREY_D, stroke_width=1.5)
        best = int(costs.argmin())
        ring = SurroundingRectangle(VGroup(strip[best], marks[best]), color=TEAL, buff=0.05)
        cost_bars, ends = self.cost_chart()
        with self.voice("match2"):
            self.play(FadeIn(strip), FadeIn(which), Create(split), run_time=0.8)
            self.add(marks)
            slide(probe, 0)
            self.play(FadeIn(probe), run_time=0.4)
            self.play(UpdateFromAlphaFunc(Group(probe, marks), lambda m, al: slide(probe, al)),
                      run_time=max(2.5, self.left_in(0) - 1.0), rate_func=linear)
            self.play(Create(ring), FadeOut(probe), run_time=0.6)
            self.cue(1)  # "Here are the costs across Dua Tawassul"
            self.play(FadeOut(VGroup(strip, marks, ring, which, split)), run_time=0.5)
            self.play(Create(cost_bars, lag_ratio=0.002), FadeIn(ends), run_time=2.2)
            low = VGroup(*[b for b, c in zip(cost_bars, DATA["cost"]) if c == DATA["cost"].min()])
            dots = VGroup(*[Dot(b.get_bottom() + 0.18 * DOWN, radius=0.05, color=TEAL) for b in low])
            # "the shorter the bar, the closer the fit"
            self.play(low.animate.set_color(TEAL), FadeIn(dots, lag_ratio=0.1), run_time=1.0)
        self.clear()

    def cost_chart(self) -> tuple[VGroup, VGroup]:
        """The heard text's cost ending at each word of Dua Tawassul."""
        cost_w, seg = DATA["cost"], DATA["seg"]
        xs = np.linspace(-6.2, 6.2, len(cost_w))
        base = -1.6
        vals = cost_w / cost_w.max()
        bars = VGroup(*[Line([x, base, 0], [x, base + 0.02 + 3.4 * v, 0], stroke_width=1.6, color=GREY_B)
                        for x, v in zip(xs, vals)])
        ends = VGroup(Tex("line 1", font_size=22, color=GREY_B).next_to([xs[0], base, 0], DOWN, buff=0.15),
                      Tex(f"line {seg.max()}", font_size=22, color=GREY_B).next_to([xs[-1], base, 0], DOWN, buff=0.15))
        return bars, ends

    # -- 5. which du'a? -------------------------------------------------------------------------
    def which_dua(self):
        prior = np.array(META["prior_all"])
        cells = VGroup(*[Square(0.15, stroke_width=0, fill_color=TEAL, fill_opacity=0.25) for _ in range(N_TEXTS)])
        cells.arrange_in_grid(cols=27, buff=0.05).move_to([-4.3, -0.7, 0])
        me = META["dua_index"]
        # One square is one text; inside it, a probability for every one of its words.
        cost = DATA["cost"].astype(float)
        xs = np.linspace(-0.6, 6.2, len(cost))
        base = -0.9
        words = VGroup(*[Line([x, base, 0], [x, base + 0.28, 0], stroke_width=0.6, color=TEAL) for x in xs])
        zoom = VGroup(DashedLine(cells[me].get_corner(UP + RIGHT), words.get_corner(UP + LEFT), color=GREY_D, stroke_width=1.5),
                      DashedLine(cells[me].get_corner(DOWN + RIGHT), words.get_corner(DOWN + LEFT), color=GREY_D, stroke_width=1.5))

        def shade(p: float) -> float:
            return float(np.clip(0.03 + 0.67 * p / prior.max(), 0.03, 1.0))

        # Each word's probability times its fit (the best match anywhere fits 1; kappa 1.2 while searching).
        eq = MathTex(r"P_{\text{new}}(w)", r"\;\propto\;", r"P(w)", r"\;\times\;", r"\text{fit}(w)", font_size=46)
        eq.to_edge(UP, buff=0.45)
        eq[0].set_color(POST)
        eq[4].set_color(EVID)
        fit = np.exp(-1.2 * (cost - cost.min()))
        tall = VGroup(*[Line([x, base, 0], [x, base + 0.02 + 2.3 * f, 0], stroke_width=1.0, color=POST) for x, f in zip(xs, fit)])
        with self.voice("dua1"):
            self.play(FadeIn(cells, lag_ratio=0.002), run_time=1.6)
            self.play(cells[me].animate.set_fill(opacity=0.9), Create(zoom), run_time=0.8)
            self.play(Create(words, lag_ratio=0.002), run_time=1.4)
            self.cue(1)  # "... spread over all the texts, with a little more on the ones people recite most"
            self.play(*[c.animate.set_fill(opacity=shade(p)) for c, p in zip(cells, prior)], run_time=1.5)
            self.cue(2)  # "With each window, places that fit well become more likely"
            self.play(Write(eq), run_time=1.2)
            self.play(Transform(words, tall), run_time=2.0)

        names = [n for n, _ in META["ident"][0]]
        labels = VGroup(*[Tex(short_name(n, 26), font_size=26) for n in names]).arrange(DOWN, buff=0.42, aligned_edge=RIGHT)
        labels.move_to([2.1 - labels.width / 2, -0.7, 0])
        x0 = labels.get_right()[0] + 0.3
        width = 3.0

        def share(v: float) -> MathTex:
            s = f"{v:.1%}".replace(".0%", "%") if v >= 0.001 else f"{v:.2%}"
            return MathTex(s.replace("%", r"\%"), font_size=26)

        # Before anything is heard: the prior, a little more for the du'as people recite most.
        bars = VGroup(*[Rectangle(width=max(0.01, width * META["prior"][n]), height=0.34, stroke_width=0, fill_color=GREY_B,
                                  fill_opacity=0.9).move_to([x0, lab.get_center()[1], 0], aligned_edge=LEFT)
                        for n, lab in zip(names, labels)])
        nums = VGroup(*[share(META["prior"][n]).next_to(b, RIGHT, buff=0.12) for n, b in zip(names, bars)])

        def step(k):
            probs = dict(META["ident"][k])
            anims = []
            for i, n in enumerate(names):
                v = probs.get(n, 0.0)
                nb = Rectangle(width=max(0.01, width * v), height=0.34, stroke_width=0,
                               fill_color=POST if i == 0 else GREY_B, fill_opacity=0.9).move_to([x0, bars[i].get_center()[1], 0], aligned_edge=LEFT)
                anims += [Transform(bars[i], nb), Transform(nums[i], share(v).next_to(nb, RIGHT, buff=0.12))]
            # The other texts share what the top four leave, each in proportion to its prior.
            rest = (1 - sum(probs.values())) / (1 - sum(META["prior"][n] for n in probs))
            anims += [c.animate.set_fill(opacity=shade(p * rest)) for i, (c, p) in enumerate(zip(cells, prior)) if i != me]
            anims.append(cells[me].animate.set_fill(opacity=0.25 + 0.75 * probs["Dua Tawassul"]))
            return anims

        line70 = DashedLine([x0 + width * 0.7, labels.get_top()[1] + 0.3, 0], [x0 + width * 0.7, labels.get_bottom()[1] - 0.25, 0],
                            color=WHITE, stroke_width=2)
        l70 = MathTex(r"70\%", font_size=24, color=WHITE).next_to(line70, UP, buff=0.08)
        with self.voice("dua2"):
            self.play(FadeOut(VGroup(words, zoom, eq)), run_time=0.5)
            self.play(FadeIn(labels, lag_ratio=0.1), FadeIn(bars), FadeIn(nums), run_time=1.0)
            self.cue(1)  # "after one window, Dua Tawassul would be ahead, at 29%"
            self.play(*step(0), run_time=1.5)
            self.cue(2)  # "The app names a du'a only at 70%"
            self.play(Create(line70), FadeIn(l70), run_time=1.0)
            self.cue_end(2, 2.4)  # "... it offers its best guesses for you to tap"
            self.play(Indicate(labels, color=WHITE, scale_factor=1.04), run_time=1.2)
        with self.voice("dua3"):
            self.wait(0.3)
            self.play(*step(1), run_time=1.5)
            self.cue_end(0, 1.4)  # "... and the app names it"
            self.play(Indicate(labels[0], color=TEAL, scale_factor=1.15), run_time=1.0)
            self.play(labels[0].animate.set_color(TEAL), run_time=0.4)
        self.clear()

        # Up for air: the real app, from listening to naming the du'a; roughly where, said over it.
        lead = PHONE_FOUND + 1.8 - (SHOTS["found"] + PHONE_LEAD)
        self.shot(SHOTS["found"], lead + clip("dua4")[1] + 1.2, talk=lead)
        self.wait(lead)
        self.say("dua4", pad=1.2)

    # -- 6. where: the letter model --------------------------------------------------------------
    def letters(self):
        # Whisper's view: a 6 s window, shown a second or so after it closes, with no timing inside.
        t_end, words = META["t_end"], META["words38"]
        ax = NumberLine(x_range=[288, 300, 1], length=12, include_ticks=True, tick_size=0.05, color=GREY_B).move_to([0, -0.4, 0])
        stamps = VGroup(*[Tex(f"{t // 60}:{t % 60:02d}", font_size=20, color=GREY_B).next_to(ax.n2p(t), DOWN, buff=0.15)
                          for t in range(288, 301, 2)])
        win = Rectangle(width=ax.get_unit_size() * 6, height=0.5, stroke_color=YELLOW, stroke_width=2.5, fill_color=YELLOW,
                        fill_opacity=0.12).move_to(ax.n2p(t_end - 3) + 0.4 * UP)
        heard = arabic(META["text"], 32, YELLOW).next_to(win, UP, buff=0.2)
        now_t = 297.7
        now = Line(ax.n2p(now_t) + 0.3 * DOWN, ax.n2p(now_t) + 2.6 * UP, color=TEAL, stroke_width=3)
        now_l = Tex("now", font_size=26, color=TEAL).next_to(now, UP, buff=0.1)
        saying = next(w for _, w, s, e in words if s <= now_t + 0.3 and e >= now_t)
        said = arabic(saying, 32, TEAL).next_to(now, RIGHT, buff=0.15).shift(0.9 * UP)
        late = Brace(Line(ax.n2p(t_end), ax.n2p(now_t)), DOWN, buff=0.55, color=GREY_B)
        with self.voice("word1"):
            self.play(Create(ax), FadeIn(stamps), run_time=0.8)
            self.play(FadeIn(win), FadeIn(heard), run_time=1)
            self.cue(1)  # "Its text comes only every second or two, with no timing inside it"
            self.play(Create(now), FadeIn(now_l), FadeIn(said, shift=0.1 * LEFT), run_time=1.2)
            self.play(Indicate(heard, color=YELLOW, scale_factor=1.1), run_time=1.2)
            self.cue_end(1, 2.4)  # "... sound that's already a second old"
            self.play(GrowFromEdge(late, UP), run_time=1.0)

        # The letter model: Whisper's listening half, one small layer onto the alphabet.
        def half(label: str, x: float, color) -> VGroup:
            r = RoundedRectangle(corner_radius=0.12, width=1.9, height=1.0, stroke_color=color, stroke_width=2).move_to([x, 2.55, 0])
            return VGroup(r, Tex(label, font_size=24, color=color).move_to(r))

        ear, pen = half("listening half", -4.6, WHITE), half("writing half", -2.6, GREY_D)
        w_name = Tex("Whisper", font_size=26, color=GREY_B).next_to(VGroup(ear, pen), UP, buff=0.12)
        layer = RoundedRectangle(corner_radius=0.08, width=0.35, height=1.0, stroke_color=TEAL, stroke_width=2,
                                 fill_color=TEAL, fill_opacity=0.25).move_to([-2.95, 2.55, 0])
        alphabet = arabic("ا ب ت ث ج ح خ د ذ ر ز ...", 26, TEAL).next_to(layer, RIGHT, buff=0.35)
        to_abc = Arrow(ear.get_right(), layer.get_left(), buff=0.06, stroke_width=2.5, color=GREY_B, max_tip_length_to_length_ratio=0.3)
        model = VGroup(ear, layer, to_abc, alphabet)

        # A 2 s window every 0.1 s; each 20 ms frame, a probability per letter.
        ctc = DATA["ctc"].astype(float)
        fa = ord("ف") - 0x0621 + 1
        f = int(ctc[:, fa].argmax())
        top = [int(c) for c in np.argsort(-ctc[f])[:5]]
        if 0 not in top:  # the empty sign is always shown: it is said aloud
            top[-1] = 0
        small = VGroup(*[Rectangle(width=ax.get_unit_size() * 2, height=0.32, stroke_color=TEAL, stroke_width=2)
                         .move_to(ax.n2p(now_t - 1 - 0.1 * (3 - k)) + 1.15 * UP) for k in range(4)])
        rows = VGroup()
        for c in top:
            lab = MathTex(r"\varnothing", font_size=36, color=GREY_B) if c == 0 else arabic(LETTERS[c], 36)
            lab.move_to(ORIGIN)
            bar = Rectangle(width=max(0.02, 3.2 * ctc[f, c]), height=0.3, stroke_width=0, fill_color=TEAL, fill_opacity=0.9)
            bar.move_to([0.45, 0, 0], aligned_edge=LEFT)
            rows.add(VGroup(lab, bar))
        rows.arrange(DOWN, buff=0.16, aligned_edge=LEFT).move_to([-3.4, 1.85, 0])
        ms20 = Tex("20 ms", font_size=24, color=GREY_B).next_to(rows, UP, buff=0.2)
        with self.voice("word2"):
            self.play(FadeOut(VGroup(heard, said, late)), win.animate.set_opacity(0.3), run_time=0.6)
            self.cue(1)  # "It starts from the listening half of the same Whisper"
            self.play(FadeIn(w_name), FadeIn(ear), FadeIn(pen), run_time=0.8)
            self.play(pen.animate.set_opacity(0.15), run_time=0.6)
            self.play(FadeOut(pen), GrowArrow(to_abc), FadeIn(layer), run_time=0.8)
            self.play(FadeIn(alphabet, shift=0.1 * RIGHT), run_time=0.8)
            self.cue(2)  # "Ten times a second, it takes the last two seconds of sound"
            self.play(FadeOut(model), FadeOut(w_name), run_time=0.5)
            for k in range(4):
                self.play(FadeIn(small[k]), run_time=0.3)
                if k < 3:
                    self.play(FadeOut(small[k]), run_time=0.12)
            self.play(FadeIn(ms20), FadeIn(rows, lag_ratio=0.15), run_time=1.4)
            self.cue_end(2, 1.6)  # "... or for no letter at all"
            self.play(Indicate(rows[top.index(0)][0], color=WHITE, scale_factor=1.6), run_time=1.2)
        self.clear()

        # "ishfa' lana 'inda": letters x 20 ms frames, as the model heard them.
        line = arabic(META["line38"]["arabic"], 40).move_to([0, 2.85, 0])
        lw = word_groups(line, 8)
        a, b = META["said_words"]
        for w in lw[:a]:
            w.set_color(GREY_B)  # said before this stretch
        said_w = [w for _, w, _, _ in words[a:b]]
        letters = DATA["ctc_letters"]
        path = DATA["ctc_path"]
        gw, gh = 10.6, 3.6
        img = letter_grid(gw, gh).move_to([0.6, -0.55, 0])
        n_l, n_f = len(letters), ctc.shape[0]
        cell_h = gh / n_l
        x_l, y_t = img.get_left()[0], img.get_top()[1]
        row_lbl = VGroup(*[arabic(LETTERS[int(c)], 22, GREY_B).move_to([x_l - 0.3, y_t - (i + 0.5) * cell_h, 0])
                           for i, c in enumerate(letters)])
        bounds = np.cumsum([len(normalize(w).replace(" ", "")) for w in said_w])[:-1]
        seps = VGroup(*[Line([x_l - 0.5, y_t - k * cell_h, 0], [x_l + gw, y_t - k * cell_h, 0], color=GREY_D, stroke_width=1)
                        for k in bounds])
        secs = VGroup(*[Tex(f"{s} s", font_size=20, color=GREY_B).move_to([x_l + gw * s / (n_f * 0.02), y_t - gh - 0.25, 0])
                        for s in range(0, int(n_f * 0.02) + 1)])
        frame_box = Rectangle(width=gw, height=gh, stroke_color=GREY_D, stroke_width=1).move_to(img)
        cover = Rectangle(width=gw, height=gh + 0.02, stroke_width=0, fill_color=BLACK, fill_opacity=1).move_to(img)
        entry = [int(np.flatnonzero(path >= k)[0]) for k in range(n_l)]
        lit_at = [entry[k] for k in np.r_[0, bounds]]  # frame where each said word's first letter is reached
        head = Line([x_l, y_t, 0], [x_l, y_t - gh, 0], color=WHITE, stroke_width=2)

        def light(t: float):
            for wi, fr in enumerate(lit_at):
                lw[a + wi].set_color(TEAL if t >= fr else WHITE)

        def reveal(m, alpha):
            t = alpha * n_f
            frac = t / n_f
            m.stretch_to_fit_width(max(gw * (1 - frac), 1e-3)).move_to([x_l + gw * frac + gw * (1 - frac) / 2, img.get_center()[1], 0])
            head.move_to([x_l + gw * frac, img.get_center()[1], 0])
            light(t)

        with self.voice("word3", pad=1.0) as d:
            self.play(FadeIn(line), run_time=0.8)
            self.add(img, cover)
            self.play(FadeIn(row_lbl, lag_ratio=0.05), Create(seps), FadeIn(secs), Create(frame_box), run_time=1.5)
            self.add(head)
            # Everything the updater changes is in the animated group, so Manim redraws it every frame.
            self.play(UpdateFromAlphaFunc(Group(cover, head, line), lambda m, al: reveal(cover, al)),
                      run_time=max(4.6, d - 3.0), rate_func=linear)
        self.remove(cover)
        self.play(FadeOut(head), run_time=0.3)
        if AUDIO and (MEDIA / "explainer_words.wav").exists():
            # Once more in real time, with the recitation.
            light(-1)
            self.sound_at_now(MEDIA / "explainer_words.wav")
            sweep = Line(head.get_start(), head.get_end(), color=WHITE, stroke_width=2)

            def replay(m, alpha):
                m.move_to([x_l + gw * alpha, img.get_center()[1], 0])
                light(alpha * n_f)

            self.add(sweep)
            self.play(UpdateFromAlphaFunc(Group(sweep, line), lambda m, al: replay(sweep, al)), run_time=n_f * 0.02,
                      rate_func=linear)
            self.remove(sweep)
        self._grid = Group(img, row_lbl, seps, secs, frame_box, line)

    # -- 7. where: the decoder ------------------------------------------------------------------------
    def follow_page(self) -> dict:
        """Lines 38-45 as a page, with a bar beside each line for the decoder's probability on it, one
        for off the text (talk, salawat), the shown word in teal on a glowing line, and the line the
        reader is on with its number in yellow and a mark under his word."""
        texts = {k: FMETA["lines"][str(k)] for k in FLINES}
        lines, nums = self.page(texts, size=27)
        page = VGroup(lines, nums).scale_to_fit_height(5.2)
        if page.width > 7.0:
            page.scale_to_fit_width(7.0)
        page.move_to([0.7 - page.width / 2, 0.3, 0])
        words = []
        for k, ln in zip(FLINES, lines):
            n = len(FMETA["line_words"][str(k)])
            try:
                words.append(word_groups(ln, n))
            except ValueError:
                words.append([ln] * n)
        bx, bw, bh = 1.15, 2.6, 0.28
        rails = VGroup(*[Line([bx, n.get_center()[1], 0], [bx + bw, n.get_center()[1], 0], stroke_width=1, color=GREY_D)
                         for n in nums])
        bars = VGroup(*[Rectangle(width=1e-3, height=bh, stroke_width=0, fill_color=POST, fill_opacity=0.9)
                        .move_to([bx, n.get_center()[1], 0], aligned_edge=LEFT) for n in nums])
        y_off = lines.get_bottom()[1] - 0.55
        off_lbl = Tex("talk, salawat", font_size=22, color=GREY_B).move_to([bx - 0.2, y_off, 0], aligned_edge=RIGHT)
        off_rail = Line([bx, y_off, 0], [bx + bw, y_off, 0], stroke_width=1, color=GREY_D)
        off_bar = Rectangle(width=1e-3, height=bh, stroke_width=0, fill_color=GREY_B, fill_opacity=0).move_to([bx, y_off, 0], aligned_edge=LEFT)
        glow = self.glow(lines[0]).set_opacity(0)
        mark = Line(ORIGIN, 0.3 * RIGHT, color=YELLOW, stroke_width=4).set_opacity(0)
        v = {"lines": lines, "nums": nums, "page": page, "words": words, "bars": bars, "rails": rails, "off": off_bar,
             "off_rail": off_rail, "off_lbl": off_lbl, "glow": glow, "mark": mark, "bx": bx, "bw": bw, "bh": bh,
             "show_off": False}
        return v

    def follow_set(self, v: dict, t: float, reader: bool = True):
        """Draw the decoder's state at item second t onto follow_page()'s mobjects."""
        k = follow_at(t)
        p = FOLLOW["line_p"][k]
        for i, ln in enumerate(FLINES):
            w = max(1e-3, v["bw"] * float(p[FCOL[ln]]))
            v["bars"][i].become(Rectangle(width=w, height=v["bh"], stroke_width=0, fill_color=POST, fill_opacity=0.9)
                                .move_to([v["bx"], v["nums"][i].get_center()[1], 0], aligned_edge=LEFT))
        v["off"].become(Rectangle(width=max(1e-3, v["bw"] * float(FOLLOW["off"][k])), height=v["bh"], stroke_width=0,
                                  fill_color=GREY_B, fill_opacity=0.9 if v["show_off"] else 0)
                        .move_to(v["off_rail"].get_start(), aligned_edge=LEFT))
        for ws in v["words"]:
            for w in ws:
                w.set_color(WHITE)
        shown = int(FOLLOW["word"][k])
        ln, wi = word_line(shown) if shown >= 0 else (-1, -1)
        if ln in FLINES:
            li = FLINES.index(ln)
            v["glow"].become(self.glow(v["lines"][li]))
            v["words"][li][wi].set_color(TEAL)
        else:
            v["glow"].set_opacity(0)
        for i, n in enumerate(v["nums"]):
            n.set_color(GREY_B)
        r = reading_at(t) if reader else None
        if r and r[0] in FLINES:
            li = FLINES.index(r[0])
            v["nums"][li].set_color(YELLOW)
            wg = v["words"][li][r[1]]
            v["mark"].become(Line(wg.get_corner(DOWN + LEFT) + 0.07 * DOWN, wg.get_corner(DOWN + RIGHT) + 0.07 * DOWN,
                                  color=YELLOW, stroke_width=4))
        else:
            v["mark"].set_opacity(0)

    def follow_play(self, v: dict, t0: float, t1: float, run_time: float, reader: bool = True):
        """The decoder from item second t0 to t1, over run_time seconds of video."""
        moving = Group(v["bars"], v["off"], v["glow"], v["mark"], v["lines"], v["nums"])
        self.play(UpdateFromAlphaFunc(moving, lambda m, al: self.follow_set(v, t0 + (t1 - t0) * al, reader)),
                  run_time=run_time, rate_func=linear)

    def decoder(self):
        v = self.follow_page()
        lines, nums = v["lines"], v["nums"]
        # The whole du'a as a strip of its 115 lines, with these eight marked.
        n = META["n_lines"]
        strip = VGroup(*[Line([-6.65, 2.6 - 5.2 * (k + 0.5) / n, 0], [-6.35, 2.6 - 5.2 * (k + 0.5) / n, 0], stroke_width=1.2,
                              color=GREY_D) for k in range(n)])
        sel = VGroup(*[strip[k - 1] for k in FLINES])
        bracket = VGroup(DashedLine(sel.get_corner(UP + RIGHT), lines.get_corner(UP + LEFT) + 0.2 * LEFT, color=GREY_D, stroke_width=1.2),
                         DashedLine(sel.get_corner(DOWN + RIGHT), lines.get_corner(DOWN + LEFT) + 0.2 * LEFT, color=GREY_D, stroke_width=1.2))
        t_start = 30.0  # well into line 43
        with self.voice("dec1"):
            self.play(FadeOut(self._grid), run_time=0.8)
            self.cue(1)  # "Instead, a decoder follows him through the one Whisper found."
            self.play(FadeIn(strip, lag_ratio=0.01), run_time=0.8)
            self.play(sel.animate.set_color(WHITE), Create(bracket), FadeIn(v["page"]), run_time=1.2)
            self.cue(2)  # "It keeps a probability on every letter of the du'a"
            self.follow_set(v, t_start, reader=False)
            self.play(FadeIn(v["rails"]), FadeIn(v["bars"]), FadeIn(v["glow"]), run_time=1.0)
            self.follow_play(v, t_start, t_start + 2.0, 2.0, reader=False)
        self.play(FadeOut(strip), FadeOut(bracket), run_time=0.5)

        # The moves the reading can make from the end of line 43, and what each costs.
        cur = FLINES.index(43)
        lx = v["bx"] + 0.35
        pts = [np.array([lx, n.get_center()[1], 0]) for n in nums]
        stops = VGroup(*[Circle(radius=0.08, stroke_color=GREY_B, stroke_width=2) for _ in pts])
        for c, p in zip(stops, pts):
            c.move_to(p)
        here = Dot(pts[cur], radius=0.1, color=TEAL)

        def hop(i: int, width: float, opacity: float, color=GREY_B) -> CurvedArrow:
            up = pts[i][1] > pts[cur][1]
            arc = CurvedArrow(pts[cur] + 0.1 * RIGHT, pts[i] + 0.1 * RIGHT, angle=(1 if up else -1) * 2.0,
                              stroke_width=width, color=color, tip_length=0.1)
            arc.set_stroke(opacity=opacity)
            arc.tip.set_fill(opacity=opacity)
            return arc

        nxt = hop(cur + 1, 4.0, 1.0, TEAL)
        again = Arc(radius=0.15, arc_center=pts[cur] + 0.3 * RIGHT, start_angle=0.75 * PI, angle=-1.5 * PI, stroke_width=3,
                    color=GREY_B).add_tip(tip_length=0.09)
        back = VGroup(*[hop(cur - k, 2.8 - 0.5 * (k - 1), 0.85 - 0.12 * (k - 1)) for k in (1, 2, 3)])
        ahead = VGroup(hop(cur + 2, 1.6, 0.55))
        far = VGroup(*[DashedLine(pts[cur] + 0.15 * RIGHT, pts[cur] + np.array([1.2, s * 2.3, 0]), stroke_width=1.2,
                                  color=GREY_B, dash_length=0.08).set_stroke(opacity=0.45) for s in (1, -1)])
        col = lx + 0.85

        def lbl(text: str, y: float, color=GREY_B) -> Tex:
            return Tex(text, font_size=24, color=color).move_to([col, y, 0], aligned_edge=LEFT)

        labels = VGroup(lbl("reading on", pts[cur + 1][1], TEAL), lbl("again", pts[cur][1]), lbl("back", pts[cur - 2][1]),
                        lbl("ahead", pts[cur + 2][1]),
                        Tex("anywhere", font_size=24, color=GREY_B).next_to(far[0].get_end(), RIGHT, buff=0.1))
        heard = arabic(FMETA["lines"]["44"].split()[0], 30, YELLOW)
        with self.voice("dec2"):
            self.play(FadeOut(v["bars"]), FadeOut(v["rails"]), FadeIn(stops), FadeIn(here), run_time=0.8)
            self.cue(1)  # "Where could he be now, given where he was?"
            self.play(Create(nxt), Create(again), *[Create(a) for a in back], *[Create(a) for a in ahead],
                      *[Create(f) for f in far], run_time=1.6)
            self.cue(2)  # "Reading on costs nothing."
            self.play(Indicate(nxt, color=TEAL, scale_factor=1.1), FadeIn(labels[0]), run_time=1.0)
            self.cue(3)  # "Starting the line again costs a little, ..."
            for i, m in ((1, again), (2, back), (3, ahead), (4, far)):
                self.play(FadeIn(labels[i]), Indicate(m, color=WHITE, scale_factor=1.1), run_time=0.9)
            self.cue(4)  # "And which of those places explains the letters just heard?"
            heard.next_to(labels[0], RIGHT, buff=0.35)
            self.play(FadeIn(heard, shift=0.1 * LEFT), run_time=0.8)
            self.play(nxt.animate.set_color(TEAL), lines[cur + 1].animate.set_color(TEAL), here.animate.move_to(pts[cur + 1]),
                      run_time=1.0)
        self.play(FadeOut(heard), lines[cur + 1].animate.set_color(WHITE), here.animate.move_to(pts[cur]), run_time=0.6)

        # Talk and salawat have places of their own; a pause is no letters.
        moves = VGroup(nxt, again, back, ahead, far, labels)
        talk = RoundedRectangle(corner_radius=0.12, width=1.3, height=0.55, stroke_color=GREY_B, stroke_width=2)
        talk.move_to([lx + 2.4, pts[cur - 2][1], 0])
        sal = RoundedRectangle(corner_radius=0.12, width=3.0, height=0.55, stroke_color=GREY_B, stroke_width=2)
        sal.move_to([lx + 3.0, pts[cur + 2][1], 0])
        talk_l = Tex("talk", font_size=24, color=GREY_B).move_to(talk)
        sal_l = arabic("اللَّهُمَّ صَلِّ عَلَى مُحَمَّدٍ وَآلِ مُحَمَّدٍ", 20, GREY_B).move_to(sal)
        outs = VGroup(CurvedArrow(pts[cur] + 0.12 * RIGHT, talk.get_left(), angle=-0.5, stroke_width=2, color=GREY_B, tip_length=0.1),
                      CurvedArrow(pts[cur] + 0.12 * RIGHT, sal.get_left(), angle=0.5, stroke_width=2, color=GREY_B, tip_length=0.1))
        backs = VGroup(CurvedArrow(talk.get_bottom(), pts[cur + 1] + 0.12 * RIGHT, angle=-0.6, stroke_width=2, color=GREY_B,
                                   tip_length=0.1),
                       CurvedArrow(sal.get_top() + 0.6 * LEFT, pts[cur + 1] + 0.12 * RIGHT, angle=0.6, stroke_width=2,
                                   color=GREY_B, tip_length=0.1))
        pause = VGroup(*[Rectangle(width=0.08, height=0.32, stroke_width=0, fill_color=WHITE, fill_opacity=1) for _ in range(2)])
        pause.arrange(RIGHT, buff=0.08)
        with self.voice("dec3"):
            self.play(FadeOut(moves), run_time=0.6)
            self.play(FadeIn(talk), FadeIn(talk_l), FadeIn(sal), FadeIn(sal_l), Create(outs), run_time=1.2)
            self.play(Create(backs), run_time=0.8)
            self.cue(1)  # "A pause is simply no letters"
            self.play(FadeOut(VGroup(talk, talk_l, sal, sal_l, outs, backs)), run_time=0.6)
            li = FLINES.index(43)
            pause.next_to(v["words"][li][-1], LEFT, buff=0.25)
            self.play(FadeIn(pause), Indicate(v["words"][li][-1], color=TEAL, scale_factor=1.15), run_time=1.2)
        self.play(FadeOut(pause), FadeOut(stops), FadeOut(here), run_time=0.5)

        # A reader going back: lines 41-43, then back to 41. The bars are the decoder's own, step by step.
        v["show_off"] = True
        self.follow_set(v, 15.2)
        with self.voice("dec4") as d:
            self.play(FadeIn(v["rails"]), FadeIn(v["bars"]), FadeIn(v["off_rail"]), FadeIn(v["off_lbl"]), FadeIn(v["off"]),
                      FadeIn(v["mark"]), run_time=0.8)
            self.follow_play(v, 15.2, 33.0, max(5.0, d - 1.4))
        with self.voice("dec5"):
            self.follow_play(v, 33.0, 35.3, 3.0)
            self.cue(1)  # "When he starts again, the first sounds fit several places"
            self.follow_play(v, 35.3, 36.2, max(3.0, self.left_in(1) - 0.3))
            self.cue(2)  # "A few letters later, line 41 explains them best"
            self.follow_play(v, 36.2, 37.6, 3.0)
        # Whisper, every two seconds, as a check on the decoder.
        chip = RoundedRectangle(corner_radius=0.12, width=1.6, height=0.6, stroke_color=YELLOW, stroke_width=2)
        chip.move_to([v["bx"] + v["bw"] + 1.6, v["lines"].get_top()[1] + 0.2, 0])
        chip_l = Tex("Whisper", font_size=24, color=YELLOW).move_to(chip)
        feed = Arrow(chip.get_bottom(), [v["bx"] + v["bw"] / 2, v["bars"][0].get_top()[1] + 0.05, 0], buff=0.08,
                     stroke_width=2.5, color=YELLOW, max_tip_length_to_length_ratio=0.12)
        pings = [t for t, w in zip(FOLLOW["t"], FOLLOW["whisper"]) if w and 37.6 < t < 45.0][::2]  # every 2 s on a phone
        with self.voice("dec6") as d:
            self.play(FadeIn(chip), FadeIn(chip_l), GrowArrow(feed), run_time=0.8)
            t, span = 37.6, max(4.0, d - 1.6)
            end = min(45.0, t + span)
            for p in pings + [end]:
                if p > end or p <= t:
                    continue
                self.follow_play(v, t, p, (p - t) * span / (end - t))
                if p < end:
                    self.play(Indicate(chip, color=YELLOW, scale_factor=1.15), Indicate(feed, color=YELLOW), run_time=0.01 + 0.3)
                t = p
        self.clear()

        # Up for air: the real app on the same go-back.
        self.shot(SHOTS["back"], 8.5)
        self.wait(8.5)

    # -- 8. the whole picture ---------------------------------------------------------------------
    def whole(self):
        whisper_icon = Group(spectrogram(1.4, 0.5), arabic(META["text"], 18, YELLOW)).arrange(RIGHT, buff=0.15)
        cells = VGroup(*[Square(0.05, stroke_width=0, fill_color=GREY_B, fill_opacity=0.3) for _ in range(N_TEXTS)])
        cells.arrange_in_grid(cols=29, buff=0.02)
        cells[META["dua_index"]].set_fill(TEAL, opacity=1).scale(2.0)
        lo = list(SEGS).index(29)
        post = DATA["post_line"][lo : lo + 19]
        where_icon = VGroup(*[Rectangle(width=0.06, height=max(0.01, 0.6 * v / post.max()), stroke_width=0, fill_color=POST,
                                        fill_opacity=0.9) for v in post]).arrange(RIGHT, buff=0.03, aligned_edge=DOWN)
        grid_icon = letter_grid(1.6, 0.6)
        k = follow_at(37.0)
        lp = [float(FOLLOW["line_p"][k][FCOL[ln]]) for ln in FLINES]
        dec_icon = VGroup(*[Rectangle(width=max(0.01, 0.9 * p), height=0.07, stroke_width=0, fill_color=POST, fill_opacity=0.9)
                            for p in lp]).arrange(DOWN, buff=0.03, aligned_edge=LEFT)
        mic = mic_icon().move_to([-6.25, 0, 0])
        top = [self.node("Whisper", whisper_icon, -4.0, 1.5, 2.1), self.node("every du'a", cells, -1.3, 1.5, 2.1),
               self.node("probabilities", where_icon, 1.4, 1.5, 2.1)]
        bot = [self.node("letter model", grid_icon, -4.0, -1.5, 2.1), self.node("decoder", dec_icon, 1.4, -1.5, 2.1)]
        out_top = VGroup(Tex("Dua Tawassul", font_size=26, color=TEAL), Tex("roughly where", font_size=24, color=GREY_B))
        out_top.arrange(DOWN, buff=0.12).move_to([4.9, 1.5, 0])
        out_bot = VGroup(Tex("line 41", font_size=26, color=TEAL), arabic(FMETA["lines"]["41"].split()[1], 34, TEAL))
        out_bot.arrange(DOWN, buff=0.12).move_to([4.9, -1.5, 0])

        def arrow(p, q):
            return Arrow(p, q, buff=0.12, stroke_width=3, color=GREY_B, max_tip_length_to_length_ratio=0.15)

        arrows_top = VGroup(arrow(mic.get_right(), top[0].get_left()), arrow(top[0].get_right(), top[1].get_left()),
                            arrow(top[1].get_right(), top[2].get_left()), arrow(top[2].get_right(), out_top.get_left()))
        arrows_bot = VGroup(arrow(mic.get_right(), bot[0].get_left()), arrow(bot[0].get_right(), bot[1].get_left()),
                            arrow(bot[1].get_right(), out_bot.get_left()))
        handoff = arrow(top[2].get_bottom(), bot[1].get_top())
        with self.voice("sum1"):
            self.play(Create(mic), run_time=0.5)
            self.play(Create(arrows_top[0]), FadeIn(top[0]), run_time=0.7)
            self.play(Create(arrows_top[1]), FadeIn(top[1]), run_time=0.7)
            self.play(Create(arrows_top[2]), FadeIn(top[2]), Create(arrows_top[3]), FadeIn(out_top), run_time=0.9)
            self.cue(1)  # "The letter model and the decoder follow him through it, word by word."
            self.play(Create(arrows_bot[0]), FadeIn(bot[0]), run_time=0.7)
            self.play(Create(arrows_bot[1]), FadeIn(bot[1]), Create(handoff), run_time=0.8)
            self.play(Create(arrows_bot[2]), FadeIn(out_bot), run_time=0.6)
            self.cue(2)  # "All of it runs inside the phone's browser"
            everything = Group(mic, *top, *bot, out_top, out_bot, arrows_top, arrows_bot, handoff)
            phone = RoundedRectangle(corner_radius=0.4, width=everything.width + 0.7, height=everything.height + 0.8,
                                     stroke_color=GREY_B, stroke_width=2).move_to(everything)
            self.play(Create(phone), run_time=1.5)
        self.clear()

    # -- 9. results ---------------------------------------------------------------------------------
    def results(self):
        setup = VGroup(Tex(r"32 voices kept out of training, 13 hours", font_size=34),
                       Tex(r"reading in flow, pausing, talking in between, going back, skipping ahead", font_size=30, color=GREY_B),
                       Tex(r"quiet rooms, echoing halls, noise, phone calls", font_size=30, color=GREY_B)).arrange(DOWN, buff=0.4)
        with self.voice("res1"):
            for s in setup:
                self.play(FadeIn(s, shift=0.1 * UP), run_time=0.8)
                self.wait(1.2)
        self.play(FadeOut(setup), run_time=0.6)
        labels = VGroup(*[Tex(r[0], font_size=34) for r in RESULTS]).arrange(DOWN, buff=0.38, aligned_edge=RIGHT).move_to([-1.0, 0, 0])
        values = VGroup(*[Tex(f"{r[1]}" + r"\%", font_size=34, color=TEAL).next_to(lab, RIGHT, buff=0.6) for r, lab in zip(RESULTS, labels)])
        values.align_to(values[0], LEFT)
        with self.voice("res2"):
            self.play(FadeIn(labels[0]), FadeIn(values[0]), run_time=0.8)
            self.cue(1)  # "The highlight is on the reader's line 87% of the time, and on the exact word"
            self.play(FadeIn(labels[1]), FadeIn(values[1]), run_time=0.8)
            self.play(FadeIn(labels[2]), FadeIn(values[2]), run_time=0.8)
            self.cue(2)  # "When a reader goes back or skips ahead, it follows within three seconds"
            self.play(FadeIn(labels[3]), FadeIn(values[3]), run_time=0.8)
            self.play(FadeIn(labels[4]), FadeIn(values[4]), run_time=0.8)
        with self.voice("limits"):
            self.play(VGroup(labels, values).animate.set_opacity(0.35), run_time=1.0)
        self.wait(0.4)
        self.clear()

    # -- 10. the path, once more ------------------------------------------------------------------
    def ending(self):
        """The whole path as a chain of pictures, then the real app once more."""
        w, gap = 1.5, 0.32
        x0 = -(7 * w + 6 * gap) / 2 + w / 2
        centers = [np.array([x0 + i * (w + gap), 0.3, 0]) for i in range(7)]
        env = DATA["env"][::5]
        wave = VGroup(*[Line([i * w / len(env), -0.5 * a - 0.02, 0], [i * w / len(env), 0.5 * a + 0.02, 0], stroke_width=2,
                             color=BLUE) for i, a in enumerate(env)])
        spec = spectrogram(w, 1.0)
        text = arabic(META["text"], 30, YELLOW)
        cells = VGroup(*[Square(0.035, stroke_width=0, fill_color=GREY_B, fill_opacity=0.3) for _ in range(N_TEXTS)])
        cells.arrange_in_grid(cols=29, buff=0.014)
        cells[META["dua_index"]].set_fill(TEAL, opacity=1).scale(2.2)
        grid = letter_grid(w, 0.8)
        k = follow_at(37.0)
        lp = [float(FOLLOW["line_p"][k][FCOL[ln]]) for ln in FLINES]
        dec = VGroup(*[Rectangle(width=max(0.01, w * p), height=0.09, stroke_width=0, fill_color=POST, fill_opacity=0.9)
                       for p in lp]).arrange(DOWN, buff=0.04, aligned_edge=LEFT)
        word = arabic(FMETA["lines"]["41"].split()[1], 44, TEAL)
        panels = [wave, spec, text, cells, grid, dec, word]
        for p, c in zip(panels, centers):
            if p.width > w:
                p.scale_to_fit_width(w)
            p.move_to(c)
        arrows = [Arrow(centers[i] + (w / 2 + 0.02) * RIGHT, centers[i + 1] - (w / 2 + 0.02) * RIGHT, buff=0, stroke_width=2.5,
                        color=GREY_B, max_tip_length_to_length_ratio=0.35) for i in range(6)]
        with self.voice("end"):
            self.play(FadeIn(wave), run_time=0.4)
            for i in range(1, 7):
                self.play(GrowArrow(arrows[i - 1]), FadeIn(panels[i], shift=0.1 * RIGHT), run_time=0.45)
        self.wait(0.6)
        self.clear()
        # The real app, once more.
        self.shot(SHOTS["end"], 8.0)  # the camera pulls back about 2 s after this
        self.wait(8.0)
        end = VGroup(Tex(r"dua-recognition", font_size=48),
                     Tex(r"github.com/Hasan-Mehdi/dua-recognition", font_size=28, color=GREY_B)).arrange(DOWN, buff=0.35)
        self.play(FadeIn(end, shift=0.2 * UP), run_time=1.2)
        self.wait(4)
