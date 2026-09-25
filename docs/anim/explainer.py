"""How the follower works, in the style of 3Blue1Brown (Manim Community).

Every curve is real: the tracker's belief, prediction and evidence at one step of
Dua Tawassul recited by a held-out reciter (Hussein Ghareeb, t = 296 s), dumped by
docs/anim/dump_explainer_data.py.

    pip install manim
    manim -qh docs/anim/explainer.py Explainer
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, Axes, Create, FadeIn, FadeOut, GrowFromCenter, Line, Rectangle, RoundedRectangle, Scene,
    Text, Transform, VGroup, VMobject, Write, config, rate_functions,
)

HERE = Path(__file__).parent
DATA = np.load(HERE / "explainer_data.npz")
META = json.loads((HERE / "explainer_meta.json").read_text(encoding="utf-8"))

BG = "#0f1115"
INK = "#efe8da"
MUTED = "#8d8676"
GOLD = "#d8b36a"
BLUE = "#58c4dd"
TEAL = "#5cd0b3"
RED = "#fc6255"
FONT = "Segoe UI"

config.background_color = BG


def txt(s: str, size: float = 30, color: str = INK, **kw) -> Text:
    return Text(s, font=FONT, font_size=size, color=color, **kw)


class Explainer(Scene):
    def construct(self):
        self.seg = DATA["seg"]
        self.n = len(self.seg)
        self.title()
        axes, bands = self.corpus()
        self.listen()
        lik = self.evidence(axes)
        self.memory(axes, lik)
        self.march(axes)
        self.play(*[FadeOut(m) for m in self.mobjects])
        self.identify()
        self.tempo()
        self.result()

    # -- helpers ---------------------------------------------------------------
    def caption(self, s: str, size: float = 30) -> Text:
        c = txt(s, size).to_edge(DOWN, buff=0.45)
        if getattr(self, "_cap", None) is not None:
            self.play(FadeOut(self._cap, shift=0.2 * UP), run_time=0.3)
            self.play(FadeIn(c, shift=0.2 * UP), run_time=0.4)
        else:
            self.play(FadeIn(c, shift=0.2 * UP), run_time=0.6)
        self._cap = c
        return c

    def area(self, axes: Axes, values: np.ndarray, color: str, height: float = 1.0) -> VMobject:
        v = values / (values.max() or 1) * height
        pts = [axes.c2p(0, 0)] + [axes.c2p(i, y) for i, y in enumerate(v)] + [axes.c2p(self.n - 1, 0)]
        m = VMobject(stroke_color=color, stroke_width=2.5, fill_color=color, fill_opacity=0.35)
        m.set_points_as_corners([*pts, pts[0]])
        return m

    # -- scenes ------------------------------------------------------------------
    def title(self):
        t = txt("How does it know where you are?", 46)
        s = txt("following a du'a, line by line", 28, MUTED).next_to(t, DOWN)
        self.play(Write(t), run_time=1.5)
        self.play(FadeIn(s, shift=0.2 * UP))
        self.wait(1.2)
        self.play(FadeOut(t), FadeOut(s))

    def corpus(self):
        axes = Axes(x_range=[0, self.n, 100], y_range=[0, 1.1, 1], x_length=12, y_length=3.2,
                    axis_config={"color": MUTED, "include_ticks": False}, tips=False).shift(0.2 * DOWN)
        self.axes = axes
        label = txt("Dua Tawassul: 637 words, 115 lines", 26, MUTED).next_to(axes, UP, buff=0.35).to_edge(LEFT, buff=0.8)
        refr = set(META["refrain_segs"])
        bands = VGroup()
        for s in np.unique(self.seg):
            idx = np.flatnonzero(self.seg == s)
            x0, x1 = axes.c2p(idx[0], 0)[0], axes.c2p(idx[-1] + 1, 0)[0]
            r = Rectangle(width=max(x1 - x0 - 0.01, 0.01), height=0.14, stroke_width=0,
                          fill_color=GOLD if s in refr else MUTED, fill_opacity=0.9 if s in refr else 0.35)
            r.move_to([(x0 + x1) / 2, axes.c2p(0, 0)[1] - 0.2, 0])
            bands.add(r)
        self.play(Create(axes.x_axis), FadeIn(label), run_time=1)
        self.play(FadeIn(bands, lag_ratio=0.01), run_time=2)
        self.caption("Gold lines are refrains: 70 of the 115 lines repeat")
        self.wait(1.5)
        self.label, self.bands = label, bands
        return axes, bands

    def listen(self):
        # A schematic waveform above the text; a 6 s window slides by 1 s.
        rng = np.random.default_rng(3)
        xs = np.linspace(-6, 6, 400)
        env = 0.25 + 0.2 * np.abs(np.sin(xs * 0.9)) + 0.05 * rng.standard_normal(400)
        wave = VGroup(*[Line([x, 2.35 - h, 0], [x, 2.35 + h, 0], stroke_width=2, color=BLUE)
                        for x, h in zip(xs, np.clip(env, 0.03, None))])
        win = RoundedRectangle(corner_radius=0.08, width=3.0, height=1.3, stroke_color=INK, stroke_width=3).move_to([-1.5, 2.35, 0])
        wl = txt("last 6 s", 20, INK).next_to(win, UP, buff=0.08)
        self.play(FadeOut(self.label), FadeIn(wave, lag_ratio=0.002), run_time=1.2)
        self.caption("Every second, transcribe the last six seconds of audio")
        self.play(Create(win), FadeIn(wl))
        for _ in range(3):
            self.play(win.animate.shift(0.5 * RIGHT), wl.animate.shift(0.5 * RIGHT), run_time=0.5)
        heard = txt(META["text"], 40, INK).next_to(win, DOWN, buff=0.25)
        self.play(FadeIn(heard, shift=0.3 * DOWN))
        self.wait(0.8)
        self.wave, self.win, self.wl, self.heard = wave, win, wl, heard

    def evidence(self, axes):
        self.caption("Where in the du'a could that text end?  exp(-κ · edit distance)")
        self.play(FadeOut(self.wave), FadeOut(self.win), FadeOut(self.wl),
                  self.heard.animate.scale(0.8).to_corner(UP + RIGHT, buff=0.5), run_time=0.9)
        lik = self.area(axes, DATA["lik"], GOLD, 1.0)
        self.play(Create(lik), run_time=2.2, rate_func=rate_functions.linear)
        self.caption("It fits every repetition of the refrain equally well")
        self.wait(2)
        return lik

    def memory(self, axes, lik):
        self.caption("But we already had a belief about where the reciter was")
        self.play(lik.animate.set_fill(opacity=0.12).set_stroke(opacity=0.35))
        before = self.area(axes, DATA["before"], BLUE, 1.0)
        self.play(FadeIn(before), run_time=1)
        self.wait(1)
        self.caption("Predict: one second later, they've moved forward a little")
        pred = self.area(axes, DATA["pred"], BLUE, 1.0)
        self.play(Transform(before, pred), run_time=1.5)
        self.wait(0.6)
        self.caption("Correct: multiply the prediction by the evidence")
        post = self.area(axes, DATA["post"], TEAL, 1.0)
        prod = txt("prediction × evidence", 26, TEAL).to_corner(UP + LEFT, buff=0.5)
        self.play(FadeIn(prod), Transform(before, post), lik.animate.set_fill(opacity=0.06), run_time=1.8)
        peak = int(DATA["post"].argmax())
        arrow_x = axes.c2p(peak, 0)[0]
        mark = txt(f"line {META['argmax_seg']}", 26, TEAL).move_to([arrow_x, axes.c2p(0, 1.1)[1] + 0.25, 0])
        truth = txt(f"human label: line {META['truth_seg']}", 22, MUTED).next_to(mark, DOWN, buff=0.12)
        self.play(GrowFromCenter(mark), FadeIn(truth))
        self.caption("Only the repetition we expected survives")
        self.wait(2.5)
        self.play(FadeOut(mark), FadeOut(truth), FadeOut(prod), FadeOut(lik))
        self.belief = before

    def march(self, axes):
        self.caption("Second by second, the belief walks through the du'a")
        for frame in DATA["seq"][::2]:
            self.play(Transform(self.belief, self.area(axes, frame, TEAL, 1.0)), run_time=0.35, rate_func=rate_functions.linear)
        self.play(Transform(self.belief, self.area(axes, DATA["post"], TEAL, 1.0)), run_time=0.35)
        self.wait(0.8)

    def identify(self):
        self._cap = None
        head = txt("Which du'a is it?", 40).to_edge(UP, buff=0.6)
        self.play(Write(head))
        sub = txt("The belief spans all 91 texts at once. Sum it inside each one.", 26, MUTED).next_to(head, DOWN)
        self.play(FadeIn(sub))
        names = [n for n, _ in META["ident"][0]]
        steps = [[dict(META["ident"][k]).get(n, 0.0) for n in names] for k in range(2)]
        rows = VGroup()
        bars = []
        for i, n in enumerate(names):
            y = 0.9 - i * 0.85
            lab = txt(n, 26, INK if i == 0 else MUTED).move_to([-3.2, y, 0]).align_to([-1.2, 0, 0], RIGHT)
            bar = Rectangle(width=0.02, height=0.45, stroke_width=0, fill_color=TEAL if i == 0 else MUTED, fill_opacity=0.9)
            bar.move_to([-0.9, y, 0], aligned_edge=LEFT)
            rows.add(lab)
            bars.append(bar)
        pct = [txt("", 24) for _ in names]
        self.play(FadeIn(rows), *[FadeIn(b) for b in bars])
        for k, vals in enumerate(steps):
            anims = []
            for i, v in enumerate(vals):
                w = max(0.02, 6.0 * v)
                nb = Rectangle(width=w, height=0.45, stroke_width=0, fill_color=bars[i].get_fill_color(), fill_opacity=0.9)
                nb.move_to(bars[i].get_left(), aligned_edge=LEFT)
                anims.append(Transform(bars[i], nb))
                np_ = txt(f"{v:.0%}", 24, INK).next_to(nb, RIGHT, buff=0.15)
                anims.append(Transform(pct[i], np_))
            when = txt(f"after {k + 1} s of listening", 26, GOLD).to_edge(DOWN, buff=0.6)
            anims.append(Transform(self._cap, when) if self._cap else FadeIn(when))
            self._cap = self._cap or when
            self.play(*anims, run_time=1.2)
            self.wait(1)
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects])

    def tempo(self):
        head = txt("How far is \"a little\"?", 40).to_edge(UP, buff=0.6)
        self.play(Write(head))
        d = np.arange(0, 6)

        def kernel(speed):
            k = np.exp(d * np.log(speed) - speed - np.cumsum(np.r_[0, np.log(np.maximum(d[1:], 1))]))
            return k / k.sum()

        groups = VGroup()
        for j, (speed, name) in enumerate([(0.6, "slow, melodic reciter"), (2.0, "brisk reciter")]):
            g = VGroup()
            for i, p in enumerate(kernel(speed)):
                r = Rectangle(width=0.5, height=max(0.02, 3.2 * p), stroke_width=0,
                              fill_color=BLUE if j == 0 else GOLD, fill_opacity=0.85)
                r.move_to([i * 0.62, 0, 0], aligned_edge=DOWN)
                g.add(r)
            g.add(txt(name, 24, MUTED).next_to(g, DOWN, buff=0.3))
            g.add(txt("words moved in 1 s  →", 18, MUTED).next_to(g, DOWN, buff=0.1))
            groups.add(g)
        groups.arrange(RIGHT, buff=1.6).shift(0.2 * DOWN)
        self.play(FadeIn(groups[0], lag_ratio=0.1))
        self.play(FadeIn(groups[1], lag_ratio=0.1))
        c = txt("Learned from human line timings, and re-weighted for each reciter as they go", 26).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(c))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

    def result(self):
        head = txt("On reciters it never trained on", 34, MUTED).to_edge(UP, buff=1.0)
        a = txt("right line   47%  →  85%", 44)
        b = txt("refrain lines   7%  →  89%", 44).next_to(a, DOWN, buff=0.5)
        c = txt("v0.1 (match each window alone)  →  now", 22, MUTED).next_to(b, DOWN, buff=0.6)
        self.play(FadeIn(head))
        self.play(Write(a), run_time=1.2)
        self.play(Write(b), run_time=1.2)
        self.play(FadeIn(c))
        self.wait(3)
