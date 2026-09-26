"""How the follower works, told the way 3Blue1Brown would (Manim Community + LaTeX).

It starts from what a du'a follower is for and why it's hard, then builds each
idea in words and pictures before its equation, paced for reading.

Every number drawn is real: the tracker's belief, prediction and evidence at one
second of Dua Tawassul recited by a held-out reciter (Hussein Ghareeb, t = 296 s),
dumped by docs/anim/dump_explainer_data.py.

    pip install manim            # plus a LaTeX install with dvisvgm
    manim -qh docs/anim/explainer.py Explainer
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import manimpango
import numpy as np
from manim import (
    BLACK, BLUE, DOWN, GREY_B, GREY_D, LEFT, RIGHT, TEAL, UP, WHITE, YELLOW, Arrow, Brace, Create, FadeIn,
    FadeOut, GrowFromEdge, Indicate, Line, MathTex, MovingCameraScene, NumberLine, Rectangle, ReplacementTransform,
    SurroundingRectangle, Tex, Text, Transform, VGroup, Write, config,
)

HERE = Path(__file__).parent
DATA = np.load(HERE / "explainer_data.npz")
META = json.loads((HERE / "explainer_meta.json").read_text(encoding="utf-8"))
manimpango.register_font(str(HERE / "fonts" / "Amiri-Regular.ttf"))

config.background_color = BLACK
# DUA_EXPLAINER_AUDIO=1: play the real recitation (cut by dump_explainer_data.py into
# the ignored data/cache/media/). The published render is silent: the audio belongs to
# DuaPlayer and its reciters.
MEDIA = HERE.parents[1] / "data" / "cache" / "media"
AUDIO = os.environ.get("DUA_EXPLAINER_AUDIO") == "1"
PRIOR, EVID, POST = BLUE, YELLOW, TEAL
SEGS = DATA["segs"]  # line ids 1..115
N_TEXTS = META.get("n_texts", 506)
# Test reciters, phone model (scripts/evaluate.py; docs/results/): v0.1's per-window matcher -> now.
RESULTS = META.get("results", {"line": (47, 85), "refrain": (6, 89), "wrong": (13, 0.3)})


def arabic(s: str, size: float = 34, color=WHITE) -> Text:
    return Text(s, font="Amiri", font_size=size, color=color)


def short_name(name: str, limit: int = 30) -> str:
    """A du'a's English name for a label: honorifics dropped, long ones cut."""
    name = re.sub(r"\s*\((a\.?s\.?|atfs|s\.?a\.?w?\.?|pbuh)\)", "", name, flags=re.I).strip()
    return name if len(name) <= limit else name[: limit - 1].rstrip() + "..."


def reading_time(*tex: str) -> float:
    """Seconds to read a caption comfortably (~15 characters a second, 2.2 s at least)."""
    plain = re.sub(r"\\[a-zA-Z]+|[{}$`'\\]", "", "".join(tex))
    return max(2.2, len(plain) / 15)


class Explainer(MovingCameraScene):
    def construct(self):
        self.title_card()
        self.intro()
        self.hook()
        self.abstract()
        self.evidence()
        self.bayes()
        self.motion()
        self.identify()
        self.results()

    # -- helpers -------------------------------------------------------------------
    def sound(self, name: str):
        if AUDIO and (MEDIA / name).exists():
            self.add_sound(str(MEDIA / name))

    def say(self, *tex: str, size: float = 34, hold: float | None = None):
        """Caption at the bottom; stays up long enough to read."""
        new = Tex(*tex, font_size=size).to_edge(DOWN, buff=0.4)
        if new.width > config.frame_width - 1:
            new.scale_to_fit_width(config.frame_width - 1)
        old = getattr(self, "_say", None)
        if old is not None:
            self.play(FadeOut(old, shift=0.15 * UP), run_time=0.5)
        self.play(FadeIn(new, shift=0.15 * UP), run_time=0.8)
        self._say = new
        self.wait(reading_time(*tex) if hold is None else hold)
        return new

    def unsay(self):
        if getattr(self, "_say", None) is not None:
            self.play(FadeOut(self._say), run_time=0.5)
            self._say = None

    def clear(self, keep=()):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=1.0)
        self._say = None
        self.wait(0.6)

    def heading(self, text: str, size: float = 44) -> Tex:
        h = Tex(text, font_size=size).to_edge(UP, buff=0.5)
        self.play(Write(h), run_time=1.4)
        self.wait(0.6)
        return h

    def line_axis(self, y: float, length: float = 12.0) -> NumberLine:
        return NumberLine(x_range=[0.5, SEGS.max() + 0.5, 1], length=length, include_ticks=False,
                          stroke_width=2, color=GREY_B).move_to([0, y, 0])

    def bars(self, ax: NumberLine, values: np.ndarray, color, height: float = 1.4, opacity: float = 0.9) -> VGroup:
        v = values / (values.max() or 1)
        w = ax.get_unit_size() * 0.8
        g = VGroup()
        for s, h in zip(SEGS, v):
            r = Rectangle(width=w, height=max(h * height, 0.001), stroke_width=0, fill_color=color, fill_opacity=opacity)
            r.move_to(ax.n2p(s), aligned_edge=DOWN)
            g.add(r)
        return g

    def page(self) -> tuple[VGroup, VGroup, list[int]]:
        """Lines 31-38 of Dua Tawassul, set right-aligned like a printed page, with line numbers."""
        lines = VGroup(*[arabic(t, 27) for t in META["lines"].values()]).arrange(DOWN, buff=0.16, aligned_edge=RIGHT)
        nums = VGroup(*[Tex(str(k), font_size=24, color=GREY_B).next_to(ln, RIGHT, buff=0.25)
                        for k, ln in zip(META["lines"], lines)])
        return lines, nums, list(map(int, META["lines"]))

    # -- 0. title ------------------------------------------------------------------------
    def title_card(self):
        t = Tex(r"Following a du'a, line by line", font_size=62)
        sub = Tex(r"how an app can listen to a recitation and keep your place", font_size=32, color=GREY_B)
        sub.next_to(t, DOWN, buff=0.45)
        self.play(Write(t), run_time=2)
        self.play(FadeIn(sub, shift=0.2 * UP), run_time=1)
        self.wait(2.5)
        self.play(FadeOut(t), FadeOut(sub), run_time=1)
        self.wait(0.5)

    # -- 1. what it's for, and why it's hard ------------------------------------------------
    def intro(self):
        lines, nums, ids = self.page()
        page = VGroup(lines, nums).scale_to_fit_height(4.6).move_to([0, 0.55, 0])
        self.play(FadeIn(page, lag_ratio=0.1), run_time=2)
        self.say(r"A du'a is a long prayer, often recited aloud by one person", r" while everyone else reads along.")
        self.say(r"It's easy to lose your place: Dua Tawassul alone runs to 115 lines.")

        # What we want: the line being recited stays highlighted, by listening alone.
        glow = SurroundingRectangle(lines[0], color=TEAL, buff=0.07, stroke_width=0, fill_opacity=0.18)
        self.play(FadeIn(glow), run_time=0.8)
        self.say(r"The goal: an app that listens, and keeps the line being recited highlighted.", hold=0.5)
        for k in (1, 2, 3):
            self.play(glow.animate.become(SurroundingRectangle(lines[k], color=TEAL, buff=0.07, stroke_width=0,
                                                               fill_opacity=0.18)), run_time=1.1)
            self.wait(0.7)
        self.wait(1)
        self.play(FadeOut(glow), FadeOut(self._say), page.animate.scale(0.8).to_edge(RIGHT, buff=0.5), run_time=1.2)
        self._say = None

        why = Tex(r"Why that's harder than it sounds", font_size=40).to_corner(UP + LEFT, buff=0.6)
        self.play(Write(why), run_time=1.4)
        self.wait(0.6)
        heard_lbl = Tex(r"heard:", font_size=26, color=GREY_B)
        heard = arabic(META["text"], 34, YELLOW)
        heard_row = VGroup(heard_lbl, heard).arrange(RIGHT, buff=0.25)
        points = VGroup(
            Tex(r"1. It hears only a few seconds at a time,\\ \hspace*{1.1em}and it mishears.", font_size=30),
            heard_row,
            Tex(r"2. Lines repeat, word for word.", font_size=30),
            Tex(r"3. People join partway through.", font_size=30),
        ).arrange(DOWN, buff=0.5, aligned_edge=LEFT).next_to(why, DOWN, buff=0.6).to_edge(LEFT, buff=0.7)
        heard_row.shift(0.45 * RIGHT)
        points[2:].shift(0.15 * DOWN)

        # 1. a fragment, misheard
        self.play(FadeIn(points[0], shift=0.2 * RIGHT))
        self.play(FadeIn(heard_lbl), FadeIn(heard, shift=0.1 * UP))
        self.say(r"A speech recognizer hears a few seconds, and gets some of it wrong:",
                 r" here, half a line, with the case endings lost.")

        # 2. repeats
        self.play(FadeIn(points[2], shift=0.2 * RIGHT))
        boxes = VGroup(*[SurroundingRectangle(lines[ids.index(k)], color=YELLOW, buff=0.05) for k in (31, 38)])
        self.play(Create(boxes), run_time=1)
        self.say(r"Lines 31 and 38 are the same words. In this du'a", r" that refrain comes back 14 times.")

        # 3. joining late
        self.play(FadeIn(points[3], shift=0.2 * RIGHT))
        arrive = Arrow(page.get_left() + 1.2 * LEFT + 0.3 * DOWN, lines[4].get_left() + 0.1 * LEFT, buff=0.1,
                       color=TEAL, stroke_width=4)
        self.play(Create(arrive), run_time=0.8)
        self.say(r"And someone who walks in halfway can't be counted from the start.")
        self.say(r"No single clue is enough. The trick is to combine them.", r" Let's build it up one idea at a time.")
        self.clear()

    # -- 2. the window ---------------------------------------------------------------------
    def hook(self):
        title = self.heading(r"Every second, listen to the last 6 seconds")
        env = DATA["env"]
        xs = np.linspace(-6.2, -0.6, len(env))
        wave = VGroup(*[Line([x, 1.3 - 0.55 * a - 0.02, 0], [x, 1.3 + 0.55 * a + 0.02, 0], stroke_width=2, color=BLUE)
                        for x, a in zip(xs, env)])
        self.sound("explainer_hook.wav")  # ends on the refrain, as the window closes
        self.play(Create(wave, lag_ratio=0.01), run_time=2.2)
        x6 = xs[-1] - (xs[-1] - xs[0]) * 6 / 16
        win = Rectangle(width=xs[-1] - x6 + 0.1, height=1.45, stroke_color=YELLOW, stroke_width=3).move_to([(x6 + xs[-1]) / 2, 1.3, 0])
        brace = Brace(win, UP, buff=0.05, color=GREY_B)
        six = Tex("last 6 seconds", font_size=28, color=GREY_B).next_to(brace, UP, buff=0.05)
        self.play(Create(win), GrowFromEdge(brace, DOWN), FadeIn(six), run_time=1.2)
        self.say(r"This is the recitation as sound. The yellow box is the newest 6 seconds.")
        heard = arabic(META["text"], 40, YELLOW).move_to([-3.4, -0.9, 0])
        heard_lbl = Tex("what the speech recognizer heard", font_size=26, color=GREY_B).next_to(heard, DOWN, buff=0.15)
        arrow = Arrow(win.get_bottom(), heard.get_top() + 0.15 * UP, buff=0.15, color=YELLOW, stroke_width=3)
        self.play(Create(arrow), FadeIn(heard, shift=0.2 * DOWN), FadeIn(heard_lbl), run_time=1.2)
        self.say(r"Whisper, a speech recognizer, turns it into text: \emph{y\=a waj\={\i}h `inda-ll\=ah}.")

        lines, nums, ids = self.page()
        page = VGroup(lines, nums).scale_to_fit_height(4.2).move_to([3.4, -0.55, 0])
        self.play(FadeIn(page, lag_ratio=0.08), run_time=1.8)
        boxes = VGroup(*[SurroundingRectangle(lines[ids.index(k)], color=YELLOW, buff=0.06) for k in (31, 38)])
        self.play(Create(boxes), run_time=1)
        self.say(r"Where in the du'a could that be? It fits line 31 and line 38 equally well.")
        self.page, self.boxes = page, boxes
        self.play(*[FadeOut(m) for m in (wave, win, brace, six, arrow, heard_lbl, title)],
                  heard.animate.scale(0.8).to_corner(UP + RIGHT, buff=0.4), run_time=1.2)
        self.heard = heard

    # -- 3. text -> a number line ---------------------------------------------------------
    def abstract(self):
        ax = self.line_axis(-0.6)
        refrains = set(META["refrain_segs"])
        ticks = VGroup(*[Line(ax.n2p(s) + 0.1 * DOWN, ax.n2p(s) + 0.1 * UP, stroke_width=2.5,
                              color=YELLOW if s in refrains else GREY_D) for s in SEGS])
        self.say(r"To reason about every line at once, lay all 115 out in order, left to right.", hold=0.5)
        self.play(ReplacementTransform(self.page, ticks), FadeOut(self.boxes), run_time=2.2)
        self.play(Create(ax), run_time=1)
        labels = VGroup(*[Tex(str(k), font_size=24, color=GREY_B).next_to(ax.n2p(k), DOWN, buff=0.2) for k in (1, 31, 38, 115)])
        self.play(FadeIn(labels))
        self.wait(1.5)
        self.say(rf"Yellow ticks are refrain lines: {len(refrains)} of the 115.")
        self.ax, self.ticks, self.ax_labels = ax, ticks, labels

    # -- 4. clue one: the evidence ----------------------------------------------------------------
    def evidence(self):
        lik = self.bars(self.ax, DATA["lik_line"], EVID, height=2.2)
        self.say(r"First clue: what we just heard.")
        self.say(r"For each line, ask: if the reciter were just finishing this line,",
                 r" how well would these words fit?", hold=0.5)
        self.sound("explainer_window.wav")  # the 6 s the recognizer heard
        self.play(FadeOut(self.ticks), GrowFromEdge(lik, DOWN), run_time=2.5)
        self.wait(2.5)
        self.say(r"A tall bar is a good fit.")
        peaks = [s for s, v in zip(SEGS, DATA["lik_line"]) if v > 0.99]
        self.play(*[Indicate(lik[list(SEGS).index(s)], color=WHITE, scale_factor=1.2) for s in peaks], run_time=1.8)
        self.say(rf"A perfect fit at {len(peaks)} different lines. On its own, this clue is a coin toss.")
        term = MathTex(r"P(\text{heard} \mid \text{line})", color=EVID, font_size=44).to_edge(UP, buff=0.5)
        self.say(r"Call it the \emph{evidence}: the chance of hearing this, if they're at that line.", hold=0.3)
        self.play(Write(term), run_time=1.5)
        self.wait(2.5)
        self.lik, self.term = lik, term

    # -- 5. clue two, and Bayes ------------------------------------------------------------
    def bayes(self):
        rows_y = (1.55, -0.35, -2.25)
        axes = [self.line_axis(y, length=10.0).shift(1.3 * RIGHT) for y in rows_y]
        lik = self.bars(axes[1], DATA["lik_line"], EVID, height=1.4)
        self.say(r"Second clue: where we thought they were a moment ago.", hold=0.3)
        self.play(FadeOut(self.ax_labels), ReplacementTransform(self.ax, axes[1]), ReplacementTransform(self.lik, lik),
                  FadeOut(self.heard), self.term.animate.scale(0.62).next_to(axes[1], LEFT, buff=0.3).shift(0.6 * UP),
                  run_time=1.5)
        tag_e = self.term

        prior_before = self.bars(axes[0], DATA["before_line"], PRIOR, height=1.4)
        prior = self.bars(axes[0], DATA["pred_line"], PRIOR, height=1.4)
        tag_p = MathTex(r"P(\text{line})", color=PRIOR, font_size=44).scale(0.62).next_to(axes[0], LEFT, buff=0.3).shift(0.6 * UP)
        self.play(Create(axes[0]), GrowFromEdge(prior_before, DOWN), FadeIn(tag_p), run_time=1.5)
        self.wait(1)
        self.say(r"That's our belief from one second ago: they were almost certainly around here.")
        self.say(r"They've kept reciting since, so push that belief forward by one second.", hold=0.3)
        self.play(Transform(prior_before, prior), run_time=2)
        self.wait(2)

        post = self.bars(axes[2], DATA["post_line"], POST, height=1.4)
        tag_q = MathTex(r"P(\text{line} \mid \text{heard})", color=POST, font_size=44).scale(0.62)
        tag_q.next_to(axes[2], LEFT, buff=0.3).shift(0.6 * UP)
        times = MathTex(r"\times", font_size=48).next_to(tag_e, UP, buff=0.3)
        self.say(r"Now combine the two clues: multiply them, line by line.", hold=0.3)
        self.play(Create(axes[2]), FadeIn(times))
        self.play(ReplacementTransform(VGroup(prior_before.copy(), lik.copy()), post), FadeIn(tag_q), run_time=2.5)
        self.wait(2)
        self.say(r"Only one line is both a good fit \emph{and} where they could plausibly be.")

        # Zoom in on the answer.
        k = int(np.argmax(DATA["post_line"]))
        target = post[k]
        frame = self.camera.frame
        saved = frame.copy()
        self.play(frame.animate.scale(0.25).move_to(target.get_center() + 0.15 * UP),
                  lik.animate.set_fill(opacity=0), run_time=2.2)
        lbl = Tex(rf"line {META['argmax_seg']}", font_size=16, color=POST).next_to(target, UP, buff=0.06)
        truth = Tex(rf"human label: line {META['truth_seg']}", font_size=11, color=GREY_B).next_to(lbl, UP, buff=0.04)
        pct = Tex(f"{DATA['post_line'][k]:.0%}".replace("%", r"\%"), font_size=13, color=WHITE).next_to(target, RIGHT, buff=0.05).shift(0.2 * DOWN)
        self.play(FadeIn(lbl, shift=0.05 * DOWN), FadeIn(truth), FadeIn(pct))
        self.wait(3)
        self.play(frame.animate.become(saved), lik.animate.set_fill(opacity=0.9), run_time=2)
        self.say(r"The evidence said ``one of 14''. The belief says which one.")

        # The rule behind it, now that the picture is there.
        eq = MathTex(r"P(\text{line} \mid \text{heard})", r"\;\propto\;", r"P(\text{heard} \mid \text{line})",
                     r"\;\cdot\;", r"P(\text{line})", font_size=44).to_edge(UP, buff=0.35)
        eq[0].set_color(POST)
        eq[2].set_color(EVID)
        eq[4].set_color(PRIOR)
        self.say(r"Written down, that's Bayes' rule. The answer becomes next second's belief,", r" and it repeats.",
                 hold=0.3)
        self.play(ReplacementTransform(tag_q.copy(), eq[0]), FadeIn(eq[1]), ReplacementTransform(tag_e.copy(), eq[2]),
                  FadeIn(eq[3]), ReplacementTransform(tag_p.copy(), eq[4]), run_time=2)
        self.wait(4)
        self.clear()

    # -- 6. how far does the belief move? --------------------------------------------------------
    def motion(self):
        title = self.heading(r"How far is ``forward''?")
        self.say(r"In one second a slow, melodic reciter might move one word. A brisk one, three.", hold=0.3)
        d = np.arange(0, 7)

        def pois(lam):
            k = np.exp(d * np.log(lam) - lam - np.cumsum(np.r_[0, np.log(np.maximum(d[1:], 1))]))
            return k / k.sum()

        charts = VGroup()
        for speed, name, col in ((0.6, r"slow, melodic reciter", BLUE), (2.0, r"brisk reciter", TEAL)):
            ax = NumberLine(x_range=[-0.5, 6.5, 1], length=4.2, include_ticks=False, color=GREY_B)
            bs = VGroup(*[Rectangle(width=0.45, height=max(2.4 * p, 0.01), stroke_width=0, fill_color=col,
                                    fill_opacity=0.9).move_to(ax.n2p(i), aligned_edge=DOWN) for i, p in enumerate(pois(speed))])
            nums = VGroup(*[MathTex(str(i), font_size=26, color=GREY_B).next_to(ax.n2p(i), DOWN, buff=0.12) for i in d])
            cap = Tex(name, font_size=28).next_to(nums, DOWN, buff=0.2)
            charts.add(VGroup(ax, bs, nums, cap))
        charts.arrange(RIGHT, buff=1.4, aligned_edge=DOWN).shift(0.3 * DOWN)
        words = Tex(r"words moved in one second", font_size=24, color=GREY_B).next_to(charts, UP, buff=0.35)
        for c in charts:
            self.play(Create(c[0]), GrowFromEdge(c[1], DOWN), FadeIn(c[2]), FadeIn(c[3]), run_time=1.4)
            self.wait(1)
        self.play(FadeIn(words))
        self.wait(2.5)
        self.say(r"The app keeps several paces in mind, taken from human timings,", r" and learns which fits this reciter.")
        eq = MathTex(r"P(\text{moved } d \text{ words})", r"=", r"\sum_{s}", r"\pi_s", r"\,\mathrm{Pois}(d;\, s\,\Delta t)",
                     font_size=40).next_to(title, DOWN, buff=0.35)
        eq[3].set_color(YELLOW)
        self.play(Write(eq), run_time=1.8)
        self.say(r"Each pace $s$ gets a weight $\pi_s$, updated every second by how well it predicted the evidence.")
        self.play(Indicate(eq[3], color=YELLOW), run_time=1.4)
        self.wait(2)
        self.clear()

    # -- 7. which du'a? ------------------------------------------------------------------------
    def identify(self):
        title = self.heading(r"We never told it which du'a this is")
        self.say(rf"The belief spreads over all {N_TEXTS} du'as and ziyarat at once.", r" Add it up inside each one.")
        eq = MathTex(r"P(\text{du'a} \mid \text{heard})", r"=", r"\sum_{\text{line} \,\in\, \text{du'a}}",
                     r"P(\text{line} \mid \text{heard})", font_size=40).next_to(title, DOWN, buff=0.35)
        eq[0].set_color(YELLOW)
        eq[3].set_color(POST)
        self.play(Write(eq), run_time=1.8)
        self.wait(1.5)
        names = [n for n, _ in META["ident"][0]]
        labels = VGroup(*[Tex(short_name(n), font_size=30) for n in names]).arrange(DOWN, buff=0.45, aligned_edge=RIGHT)
        labels.move_to([-1.0 - labels.width / 2, -1.0, 0])
        bars = VGroup(*[Rectangle(width=0.01, height=0.38, stroke_width=0, fill_color=GREY_B, fill_opacity=0.9)
                        .next_to(lab, RIGHT, buff=0.3) for lab in labels])
        pct = VGroup(*[MathTex("0\\%", font_size=28).next_to(b, RIGHT, buff=0.15) for b in bars])
        self.play(FadeIn(labels, lag_ratio=0.1), FadeIn(bars), FadeIn(pct))
        for k in range(2):
            probs = dict(META["ident"][k])
            anims = []
            for i, n in enumerate(names):
                v = probs.get(n, 0.0)
                nb = Rectangle(width=max(0.01, 5.5 * v), height=0.38, stroke_width=0,
                               fill_color=POST if i == 0 else GREY_B, fill_opacity=0.9).next_to(labels[i], RIGHT, buff=0.3)
                anims += [Transform(bars[i], nb),
                          Transform(pct[i], MathTex(f"{v:.0%}".replace("%", r"\%"), font_size=28).next_to(nb, RIGHT, buff=0.15))]
            self.play(*anims, run_time=1.5)
            self.say(rf"after {k + 1} second{'s' if k else ''} of listening", hold=2.2)
        self.say(r"The same belief that follows the line names the du'a, for free.")
        self.clear()

    # -- 8. results ---------------------------------------------------------------------------------
    def results(self):
        title = Tex(r"On reciters it never trained on", font_size=40, color=GREY_B).to_edge(UP, buff=0.9)

        def num(v):
            return (f"{v:g}" + r"\%")

        rows = VGroup(*[
            Tex(label, r"\quad " + num(a), r"$\;\longrightarrow\;$", num(b), font_size=46)
            for label, (a, b) in ((r"right line", RESULTS["line"]), (r"refrain lines", RESULTS["refrain"]),
                                  (r"wrong du'a shown", RESULTS["wrong"]))
        ]).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        for r in rows:
            r[3].set_color(TEAL)
        foot = Tex(r"matching each window on its own \quad$\longrightarrow$\quad following with a belief",
                   font_size=28, color=GREY_B).next_to(rows, DOWN, buff=0.7)
        self.play(FadeIn(title))
        for r in rows:
            self.play(Write(r), run_time=1.3)
            self.wait(0.8)
        self.play(FadeIn(foot))
        self.wait(4)
        self.play(*[FadeOut(m) for m in (title, rows, foot)], run_time=1)
        end = VGroup(Tex(r"dua-recognition", font_size=48),
                     Tex(r"github.com/Hasan-Mehdi/dua-recognition", font_size=28, color=GREY_B)).arrange(DOWN, buff=0.35)
        self.play(FadeIn(end, shift=0.2 * UP), run_time=1.2)
        self.wait(3)
