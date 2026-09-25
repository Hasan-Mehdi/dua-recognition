"""How the follower works, told the way 3Blue1Brown would (Manim Community + LaTeX).

Every number drawn is real: the tracker's belief, prediction and evidence at one
second of Dua Tawassul recited by a held-out reciter (Hussein Ghareeb, t = 296 s),
dumped by docs/anim/dump_explainer_data.py.

    pip install manim            # plus a LaTeX install with dvisvgm
    manim -qh docs/anim/explainer.py Explainer
"""
from __future__ import annotations

import json
from pathlib import Path

import manimpango
import numpy as np
from manim import (
    BLACK, BLUE, DOWN, GREY_B, GREY_D, LEFT, ORIGIN, RIGHT, TEAL, UP, WHITE, YELLOW, Arrow, Brace, Create,
    DashedLine, FadeIn, FadeOut, GrowFromEdge, Indicate, Line, MathTex, MovingCameraScene, NumberLine, Rectangle,
    ReplacementTransform, SurroundingRectangle, Tex, Text, Transform, TransformMatchingTex, VGroup, Write, config,
    rate_functions,
)

HERE = Path(__file__).parent
DATA = np.load(HERE / "explainer_data.npz")
META = json.loads((HERE / "explainer_meta.json").read_text(encoding="utf-8"))
manimpango.register_font(str(HERE / "fonts" / "Amiri-Regular.ttf"))

config.background_color = BLACK
PRIOR, EVID, POST = BLUE, YELLOW, TEAL
SEGS = DATA["segs"]  # line ids 1..115


def arabic(s: str, size: float = 34, color=WHITE) -> Text:
    return Text(s, font="Amiri", font_size=size, color=color)


class Explainer(MovingCameraScene):
    def construct(self):
        self.hook()
        self.abstract()
        self.evidence()
        self.bayes()
        self.motion()
        self.identify()
        self.results()

    # -- helpers -------------------------------------------------------------------
    def say(self, *tex: str, size: float = 34):
        new = Tex(*tex, font_size=size).to_edge(DOWN, buff=0.4)
        old = getattr(self, "_say", None)
        if old is not None:
            self.play(FadeOut(old, shift=0.15 * UP), run_time=0.35)
        self.play(FadeIn(new, shift=0.15 * UP), run_time=0.5)
        self._say = new
        return new

    def clear(self, keep=()):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=0.8)
        self._say = None

    def line_axis(self, y: float, length: float = 12.0) -> NumberLine:
        ax = NumberLine(x_range=[0.5, SEGS.max() + 0.5, 1], length=length, include_ticks=False,
                        stroke_width=2, color=GREY_B).move_to([0, y, 0])
        return ax

    def bars(self, ax: NumberLine, values: np.ndarray, color, height: float = 1.4, opacity: float = 0.9) -> VGroup:
        v = values / (values.max() or 1)
        w = ax.get_unit_size() * 0.8
        g = VGroup()
        for s, h in zip(SEGS, v):
            r = Rectangle(width=w, height=max(h * height, 0.001), stroke_width=0, fill_color=color, fill_opacity=opacity)
            r.move_to(ax.n2p(s), aligned_edge=DOWN)
            g.add(r)
        return g

    # -- 1. the problem, with the real text ----------------------------------------------
    def hook(self):
        title = Tex("Someone is reciting. Which line are they on?", font_size=44).to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1.5)

        env = DATA["env"]
        xs = np.linspace(-6.2, -0.6, len(env))
        wave = VGroup(*[Line([x, 1.3 - 0.55 * a - 0.02, 0], [x, 1.3 + 0.55 * a + 0.02, 0], stroke_width=2, color=BLUE)
                        for x, a in zip(xs, env)])
        self.play(Create(wave, lag_ratio=0.01), run_time=1.6)
        x6 = xs[-1] - (xs[-1] - xs[0]) * 6 / 16
        win = Rectangle(width=xs[-1] - x6 + 0.1, height=1.45, stroke_color=YELLOW, stroke_width=3).move_to([(x6 + xs[-1]) / 2, 1.3, 0])
        brace = Brace(win, UP, buff=0.05, color=GREY_B)
        six = Tex("last 6 seconds", font_size=28, color=GREY_B).next_to(brace, UP, buff=0.05)
        self.play(Create(win), GrowFromEdge(brace, DOWN), FadeIn(six))
        heard = arabic(META["text"], 40, YELLOW).move_to([-3.4, -0.4, 0])
        heard_lbl = Tex("what the speech recognizer heard", font_size=26, color=GREY_B).next_to(heard, DOWN, buff=0.15)
        arrow = Arrow(win.get_bottom(), heard.get_top(), buff=0.1, color=YELLOW, stroke_width=3)
        self.play(Create(arrow), FadeIn(heard, shift=0.2 * DOWN), FadeIn(heard_lbl))

        lines = VGroup(*[arabic(t, 27) for t in META["lines"].values()]).arrange(DOWN, buff=0.13, aligned_edge=RIGHT)
        nums = VGroup(*[Tex(str(k), font_size=24, color=GREY_B).next_to(ln, RIGHT, buff=0.25)
                        for k, ln in zip(META["lines"], lines)])
        page = VGroup(lines, nums).scale_to_fit_height(4.7).move_to([3.4, -0.55, 0])
        self.play(FadeIn(page, lag_ratio=0.08), run_time=1.6)
        ids = list(map(int, META["lines"]))
        boxes = VGroup(*[SurroundingRectangle(lines[ids.index(k)], color=YELLOW, buff=0.06) for k in (31, 38)])
        self.play(Create(boxes), run_time=1)
        self.say(r"Lines 31 and 38 are the same words.", r" The text alone can't tell them apart.")
        self.wait(2)
        self.say(r"And in Dua Tawassul this refrain comes back 14 times.")
        self.wait(1.5)
        self.page, self.boxes, self.title = page, boxes, title
        self.play(*[FadeOut(m) for m in (wave, win, brace, six, arrow, heard_lbl, title)], heard.animate.scale(0.8).to_corner(UP + RIGHT, buff=0.4))
        self.heard = heard

    # -- 2. text -> a number line ---------------------------------------------------------
    def abstract(self):
        ax = self.line_axis(-0.6)
        refrains = set(META["refrain_segs"])
        ticks = VGroup(*[Line(ax.n2p(s) + 0.1 * DOWN, ax.n2p(s) + 0.1 * UP, stroke_width=2.5,
                              color=YELLOW if s in refrains else GREY_D) for s in SEGS])
        self.say(r"Squash the du'a down to its 115 lines, in order.")
        self.play(ReplacementTransform(self.page, ticks), FadeOut(self.boxes), run_time=2)
        self.play(Create(ax), run_time=0.8)
        labels = VGroup(*[Tex(str(k), font_size=24, color=GREY_B).next_to(ax.n2p(k), DOWN, buff=0.2) for k in (1, 31, 38, 115)])
        self.play(FadeIn(labels))
        self.say(r"Yellow ticks are refrain lines: 70 of the 115.")
        self.wait(2)
        self.ax, self.ticks, self.ax_labels = ax, ticks, labels

    # -- 3. the evidence ------------------------------------------------------------------
    def evidence(self):
        lik = self.bars(self.ax, DATA["lik_line"], EVID, height=2.2)
        term = MathTex(r"P(\text{heard} \mid \text{line})", color=EVID, font_size=44).to_edge(UP, buff=0.5)
        self.play(Write(term))
        self.say(r"How well does what we heard fit the end of each line?")
        self.play(FadeOut(self.ticks), GrowFromEdge(lik, DOWN), run_time=2)
        self.wait(1)
        peaks = [s for s, v in zip(SEGS, DATA["lik_line"]) if v > 0.99]
        self.say(rf"A perfect fit at {len(peaks)} different lines. On its own, this is a coin toss.")
        self.play(*[Indicate(lik[list(SEGS).index(s)], color=WHITE, scale_factor=1.2) for s in peaks], run_time=1.5)
        self.wait(1.5)
        self.lik, self.term = lik, term

    # -- 4. Bayes: prediction x evidence ------------------------------------------------------
    def bayes(self):
        eq = MathTex(r"P(\text{line} \mid \text{heard})", r"\;\propto\;", r"P(\text{heard} \mid \text{line})",
                     r"\;\cdot\;", r"P(\text{line})", font_size=44).to_edge(UP, buff=0.5)
        eq[0].set_color(POST)
        eq[2].set_color(EVID)
        eq[4].set_color(PRIOR)
        self.say(r"But we aren't starting from nothing. A second ago, we had a belief.")
        self.play(TransformMatchingTex(self.term, eq), run_time=1.5)

        # Three stacked rows: prior, evidence, posterior.
        rows_y = (1.55, -0.35, -2.25)
        axes = [self.line_axis(y, length=10.0).shift(1.3 * RIGHT) for y in rows_y]
        lik = self.bars(axes[1], DATA["lik_line"], EVID, height=1.4)
        self.play(
            FadeOut(self.ax_labels),
            ReplacementTransform(self.ax, axes[1]),
            ReplacementTransform(self.lik, lik),
            FadeOut(self.heard),
        )
        prior_before = self.bars(axes[0], DATA["before_line"], PRIOR, height=1.4)
        prior = self.bars(axes[0], DATA["pred_line"], PRIOR, height=1.4)
        tag_p = eq[4].copy().scale(0.62).next_to(axes[0], LEFT, buff=0.3).shift(0.6 * UP)
        tag_e = eq[2].copy().scale(0.62).next_to(axes[1], LEFT, buff=0.3).shift(0.6 * UP)
        self.play(Create(axes[0]), GrowFromEdge(prior_before, DOWN), ReplacementTransform(eq[4].copy(), tag_p),
                  ReplacementTransform(eq[2].copy(), tag_e))
        self.say(r"Where we thought they were, pushed forward by one second of reciting.")
        self.play(Transform(prior_before, prior), run_time=1.5)
        self.wait(1)

        post = self.bars(axes[2], DATA["post_line"], POST, height=1.4)
        tag_q = eq[0].copy().scale(0.62).next_to(axes[2], LEFT, buff=0.3).shift(0.6 * UP)
        times = MathTex(r"\times", font_size=48).next_to(tag_e, UP, buff=0.3)
        self.say(r"Multiply the two, line by line.")
        self.play(Create(axes[2]), FadeIn(times))
        self.play(ReplacementTransform(VGroup(prior_before.copy(), lik.copy()), post),
                  ReplacementTransform(eq[0].copy(), tag_q), run_time=2)
        self.wait(0.5)

        # Zoom in on the answer.
        k = int(np.argmax(DATA["post_line"]))
        target = post[k]
        frame = self.camera.frame
        saved = frame.copy()
        self.play(frame.animate.scale(0.25).move_to(target.get_center() + 0.15 * UP),
                  lik.animate.set_fill(opacity=0), run_time=2)
        lbl = Tex(rf"line {META['argmax_seg']}", font_size=16, color=POST).next_to(target, UP, buff=0.06)
        truth = Tex(rf"human label: line {META['truth_seg']}", font_size=11, color=GREY_B).next_to(lbl, UP, buff=0.04)
        pct = Tex(f"{DATA['post_line'][k]:.0%}".replace("%", r"\%"), font_size=13, color=WHITE).next_to(target, RIGHT, buff=0.05).shift(0.2 * DOWN)
        self.play(FadeIn(lbl, shift=0.05 * DOWN), FadeIn(truth), FadeIn(pct))
        self.wait(2)
        self.play(frame.animate.become(saved), lik.animate.set_fill(opacity=0.9), run_time=1.6)
        self.say(r"The evidence said ``one of 14''. The belief says which one.")
        self.wait(2.5)
        self.clear()

    # -- 5. how far does the belief move? --------------------------------------------------------
    def motion(self):
        title = Tex(r"How far do we push the belief forward?", font_size=44).to_edge(UP, buff=0.5)
        eq = MathTex(r"P(\text{moved } d \text{ words})", r"=", r"\sum_{s}", r"\pi_s", r"\,\mathrm{Pois}(d;\, s\,\Delta t)",
                     font_size=42).next_to(title, DOWN, buff=0.4)
        eq[3].set_color(YELLOW)
        self.play(Write(title))
        self.play(Write(eq), run_time=1.5)
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
        charts.arrange(RIGHT, buff=1.4).shift(0.9 * DOWN)
        for c in charts:
            self.play(Create(c[0]), GrowFromEdge(c[1], DOWN), FadeIn(c[2]), FadeIn(c[3]), run_time=1.2)
        self.say(r"Speeds $s$ come from human line timings. The weights $\pi_s$ adapt to the reciter as they go.")
        self.play(Indicate(eq[3], color=YELLOW), run_time=1.2)
        self.wait(1.5)
        self.say(r"The first version assumed twice the real speed, and ran a line ahead on long lines.")
        self.wait(2.5)
        self.clear()

    # -- 6. which du'a? ------------------------------------------------------------------------
    def identify(self):
        title = Tex(r"Which du'a is it?", font_size=44).to_edge(UP, buff=0.5)
        eq = MathTex(r"P(\text{du'a} \mid \text{heard})", r"=", r"\sum_{\text{line} \,\in\, \text{du'a}}",
                     r"P(\text{line} \mid \text{heard})", font_size=40).next_to(title, DOWN, buff=0.35)
        eq[0].set_color(YELLOW)
        eq[3].set_color(POST)
        self.play(Write(title))
        self.play(Write(eq), run_time=1.5)
        self.say(r"The belief covers all 91 texts at once. Add it up inside each one.")
        names = [n for n, _ in META["ident"][0]]
        labels = VGroup(*[Tex(n.replace("'", "'"), font_size=30) for n in names]).arrange(DOWN, buff=0.45, aligned_edge=RIGHT)
        labels.move_to([-3.2, -1.0, 0])
        base = [Line(ORIGIN, ORIGIN) for _ in names]
        bars = VGroup()
        for lab in labels:
            bars.add(Rectangle(width=0.01, height=0.38, stroke_width=0, fill_color=GREY_B, fill_opacity=0.9)
                     .next_to(lab, RIGHT, buff=0.3))
        pct = VGroup(*[MathTex("0\\%", font_size=28).next_to(b, RIGHT, buff=0.15) for b in bars])
        self.play(FadeIn(labels, lag_ratio=0.1), FadeIn(bars), FadeIn(pct))
        for k in range(2):
            probs = dict(META["ident"][k])
            anims = []
            for i, n in enumerate(names):
                v = probs.get(n, 0.0)
                nb = Rectangle(width=max(0.01, 7.0 * v), height=0.38, stroke_width=0,
                               fill_color=POST if i == 0 else GREY_B, fill_opacity=0.9).next_to(labels[i], RIGHT, buff=0.3)
                anims += [Transform(bars[i], nb),
                          Transform(pct[i], MathTex(f"{v:.0%}".replace("%", r"\%"), font_size=28).next_to(nb, RIGHT, buff=0.15))]
            self.play(*anims, run_time=1.2)
            self.say(rf"after {k + 1} second{'s' if k else ''} of listening")
            self.wait(1.2)
        self.wait(1)
        self.clear()

    # -- 7. results ---------------------------------------------------------------------------------
    def results(self):
        title = Tex(r"On reciters it never trained on", font_size=40, color=GREY_B).to_edge(UP, buff=0.9)
        rows = VGroup(
            Tex(r"right line", r"\quad 47\%", r"$\;\longrightarrow\;$", r"85\%", font_size=46),
            Tex(r"refrain lines", r"\quad 7\%", r"$\;\longrightarrow\;$", r"89\%", font_size=46),
            Tex(r"wrong du'a shown", r"\quad 15\%", r"$\;\longrightarrow\;$", r"1\%", font_size=46),
        ).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        for r in rows:
            r[3].set_color(TEAL)
        foot = Tex(r"v0.1 matched each window on its own \quad$\longrightarrow$\quad now: score following",
                   font_size=28, color=GREY_B).next_to(rows, DOWN, buff=0.7)
        self.play(FadeIn(title))
        for r in rows:
            self.play(Write(r), run_time=1.1)
        self.play(FadeIn(foot))
        self.wait(3)
