"""The explainer, in six scenes (ManimGL, 3Blue1Brown's own Manim). Each is timed to its narration
(lines.py) and drawn from the real reading's data (reading.py, dump.py, images.py).

    cd docs/explainer
    manimgl scenes.py P1 -w -r 2560x1440 --video_dir ../../data/cache/media/explainer10/render    # P1 ... P6
    python publish.py                                     # the parts, the voice, the reading, the subtitles

The bottom of the frame (below y = -2.8) is kept clear for the subtitles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit import *  # noqa: E402,F401,F403
from kit import (T, BG, BLUE, CACHE, DIM, FAINT, GOLD, IMG, INK, SERIF, SOFT, TEAL, WARN, VoiceScene, ar, bar,  # noqa: E402
                 chip, data, disc, en, spec, voice, wave)

OTHER = "#B49CE0"  # a second recording's points, a cloned voice


def num(value: float, size: float = 36, color=INK, places: int = 0, **kw) -> DecimalNumber:
    return DecimalNumber(value, num_decimal_places=places, font_size=size, color=color,
                         text_config={"font": SERIF}, **kw)


def peak_dots(img: Mobject, peaks: np.ndarray, n_frames: int, color=INK, r: float = 0.05, glow: bool = True):
    """Shazam's points on a spectrogram picture: (frame, band, value) rows placed on the image."""
    W, H = img.get_width(), img.get_height()
    x0, y0 = img.get_left()[0], img.get_bottom()[1]
    dots = VGroup()
    for t, f, _ in peaks:
        p = np.array([x0 + (t + 0.5) / n_frames * W, y0 + (f + 0.5) / 80 * H, 0])
        dots.add(Dot(p, radius=r).set_fill(color, 1).set_stroke(BG, 0.8))
    halo = Group(*(GlowDot(d.get_center(), radius=r * 4.5, color=color, glow_factor=2.0).set_opacity(0.5)
                   for d in dots)) if glow else Group()
    return halo, dots


def reveal(img: Mobject, run_time: float = 1.6):
    """Draw a picture in from left to right."""
    cover = Rectangle(width=img.get_width() + 0.05, height=img.get_height() + 0.05)
    cover.set_fill(BG, 1).set_stroke(width=0).move_to(img)
    return cover, cover.animate.stretch(0.001, 0, about_edge=RIGHT).set_anim_args(run_time=run_time)


def mic_icon(color=INK, h: float = 1.1) -> VGroup:
    head = RoundedRectangle(width=0.36, height=0.62, corner_radius=0.18).set_stroke(color, 2.5)
    cup = Arc(start_angle=PI, angle=PI, radius=0.3).set_stroke(color, 2.5).next_to(head, DOWN, buff=-0.2)
    stem = Line(cup.get_bottom(), cup.get_bottom() + 0.22 * DOWN).set_stroke(color, 2.5)
    base = Line(LEFT * 0.18, RIGHT * 0.18).set_stroke(color, 2.5).next_to(stem, DOWN, buff=0)
    return VGroup(head, cup, stem, base).set_height(h)


def person(color=INK, h: float = 0.9) -> VGroup:
    head = Circle(radius=0.16).set_fill(color, 1).set_stroke(width=0)
    body = Arc(start_angle=0, angle=PI, radius=0.32).set_stroke(color, 0).set_fill(color, 1)
    body.next_to(head, DOWN, buff=0.06)
    return VGroup(head, body).set_height(h)


def page_icon(n: int = 7, w: float = 2.3, h: float = 3.0, color=SOFT) -> VGroup:
    box = RoundedRectangle(width=w, height=h, corner_radius=0.14).set_stroke(color, 2).set_fill(BG, 1)
    rows = VGroup(*(Line(LEFT, RIGHT).set_width(w * (0.78 if i % 3 else 0.58)) for i in range(n)))
    rows.arrange(DOWN, buff=h / (n + 2)).move_to(box).set_stroke(DIM, 4)
    for r in rows:
        r.align_to(box.get_right() + LEFT * 0.22, RIGHT)
    return VGroup(box, rows)


def song_wave(width: float, height: float, color=SOFT) -> VMobject:
    xs = np.linspace(-width / 2, width / 2, 900)
    t = np.linspace(0, 6, 900)
    env = 0.55 + 0.45 * np.abs(np.sin(np.pi * t * 2)) ** 3
    y = env * (0.6 * np.sin(2 * np.pi * 7.3 * t) + 0.4 * np.sin(2 * np.pi * 19.1 * t + 1)) * height / 2
    top = [np.array([x, abs(v), 0]) for x, v in zip(xs, y)]
    bot = [np.array([x, -abs(v), 0]) for x, v in zip(xs[::-1], y[::-1])]
    m = VMobject()
    m.set_points_as_corners(top + bot + [top[0]])
    return m.set_fill(color, 0.85).set_stroke(color, 1.2, 0.9)


def mini_wave(rng, width: float = 1.1, height: float = 0.32, color=SOFT, n: int = 40) -> VMobject:
    xs = np.linspace(-width / 2, width / 2, n)
    ys = rng.uniform(0.15, 1.0, n) * np.hanning(n) * height / 2
    m = VMobject().set_points_as_corners([np.array([x, y, 0]) for x, y in zip(xs, ys)] +
                                         [np.array([x, -y, 0]) for x, y in zip(xs[::-1], ys[::-1])])
    return m.set_fill(color, 0.8).set_stroke(color, 1, 0.8)


def box_label(text: str, color, w: float = 2.2, h: float = 0.9, size: float = 28) -> VGroup:
    b = RoundedRectangle(width=w, height=h, corner_radius=0.12).set_stroke(color, 2.5).set_fill(color, 0.1)
    t = en(text, size, color).move_to(b)
    return VGroup(b, t)


def fold(text: str) -> str:
    """The letters the matching compares: vowel marks dropped, alef and ya forms folded, no spaces."""
    import unicodedata
    out = []
    for c in unicodedata.normalize("NFC", text):
        if "ً" <= c <= "ْ" or c in "ٰـ ":
            continue
        out.append({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}.get(c, c))
    return "".join(out)


def letter_tiles(s: str, size: float = 30, color=GOLD, w: float = 0.48, h: float = 0.6) -> VGroup:
    tiles = VGroup()
    for c in s:
        box = RoundedRectangle(width=w, height=h, corner_radius=0.08).set_stroke(color, 1.5, 0.8).set_fill(color, 0.08)
        tiles.add(VGroup(box, ar(c, size, color).move_to(box)))
    return tiles.arrange(LEFT, buff=0.05)


def arabic_line(text: str, size: float = 40, color=INK, buff: float = 0.2) -> VGroup:
    """A line of Arabic as its words, right to left, so a highlight can sit on one word."""
    return VGroup(*(ar(w, size, color) for w in text.split())).arrange(LEFT, buff=buff, aligned_edge=DOWN)


class P1(VoiceScene):
    """What a du'a recognizer is: the real app, Shazam, why its points don't carry over, Tarteel, what we took."""

    def construct(self):
        z, m = data()
        pk = np.load(IMG / "peaks.npz")

        # -- cold open: the real page (laid over this by the publish step), his voice alone, then the narration
        # From the quiet before line 2 (his salawat has died away by 12.2 s) to the pause after line 5 (32.0 s):
        # he is heard alone first, the app names the du'a at 20.5 s, then it follows lines 3-5.
        start, end, lead = 12.25, 32.0, 4.0
        self.shot(start, end - start)
        w = wave(end - 12, end, 12.5, 1.8).move_to(UP * 0.4)
        self.add(w)
        self.wait(lead)
        with self.say("open1"):
            pass
        self.wait(max(0.3, (end - start) - self.time))

        # -- Shazam: what it does for you, then how: a picture of the sound, its brightest points, codes, one song
        logo_s = ImageMobject(str(IMG / "logo_shazam.png"), height=1.7).move_to(UP * 2.4)
        with self.say("shazam1") as s:
            self.play(FadeIn(logo_s, scale=0.85), Transform(w, song_wave(12.5, 1.8).move_to(DOWN * 0.9)), run_time=T(1.6))
            s.until(1)
            # a phone held up to a song: it listens, then names it
            phone = RoundedRectangle(width=1.6, height=2.9, corner_radius=0.22).set_stroke(INK, 2.5)
            phone.move_to(RIGHT * 4.9 + DOWN * 0.4)
            self.play(logo_s.animate.set_height(0.85).move_to(phone.get_center() + UP * 0.45), ShowCreation(phone),
                      w.animate.set_width(8.6, stretch=True).move_to(LEFT * 1.5 + DOWN * 0.4), run_time=T(1.1))
            rings = VGroup(*(Circle(radius=0.55 + 0.28 * k).set_stroke(BLUE, 2.5, 0.8 - 0.2 * k).move_to(logo_s)
                             for k in range(3)))
            self.play(LaggedStart(*(GrowFromCenter(r) for r in rings), lag_ratio=0.3), run_time=T(1.0))
            self.play(FadeOut(rings), run_time=T(0.4))
            card = VGroup(RoundedRectangle(width=1.15, height=0.16, corner_radius=0.06).set_fill(INK, 0.95),
                          RoundedRectangle(width=0.75, height=0.12, corner_radius=0.05).set_fill(SOFT, 0.95))
            card.set_stroke(width=0).arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to(phone.get_center() + DOWN * 0.65)
            self.play(FadeIn(card, shift=UP * 0.15), run_time=max(T(0.5), s.left(1)))
            s.until(2)
            img = spec("song", 3.5).set_width(10.4, stretch=True).move_to(LEFT * 0.75 + DOWN * 0.85)
            self.play(FadeOut(VGroup(phone, card)), logo_s.animate.set_height(0.8).to_corner(UL, buff=0.35),
                      w.animate.set_height(0.7, stretch=True).set_width(10.4, stretch=True).move_to(LEFT * 0.75 + UP * 2.65),
                      run_time=T(1.0))
            cover, sweep = reveal(img, 2.0)
            self.add(img, cover)
            self.play(sweep)
            self.remove(cover)
            halo, dots = peak_dots(img, pk["s"], int(pk["n_s"]))
            self.play(img.animate.set_opacity(0.2), LaggedStartMap(FadeIn, dots, lag_ratio=0.012), FadeIn(halo),
                      run_time=s.left(2))
            s.until(3)
            pts = sorted(dots, key=lambda d: d.get_center()[0])
            anchor = pts[len(pts) // 5]
            ax, ay = anchor.get_center()[:2]
            zone = [d for d in pts if 0.25 < d.get_center()[0] - ax < 1.7 and abs(d.get_center()[1] - ay) < 0.95][:4]
            box = Rectangle(width=1.5, height=1.9).set_stroke(GOLD, 1.5, 0.7).move_to([ax + 1.0, ay, 0])
            links = VGroup(*(Line(anchor.get_center(), d.get_center()).set_stroke(GOLD, 2.5) for d in zone))
            self.play(anchor.animate.scale(2.4).set_fill(GOLD), FadeIn(box), run_time=T(0.6))
            self.play(LaggedStartMap(ShowCreation, links, lag_ratio=0.2), *(d.animate.set_fill(GOLD).scale(1.6) for d in zone),
                      run_time=T(0.9))
            index = VGroup(*(Line(LEFT, RIGHT).set_width(1.3).set_stroke(DIM, 2.5) for _ in range(30)))
            index.arrange(DOWN, buff=0.1).move_to([5.95, -0.85, 0])
            codes = VGroup(*(chip(c, GOLD, 24) for c in ("3a1f·7c", "3a1f·92", "40e2·1b", "40e2·66")[: len(zone)]))
            codes.arrange(DOWN, buff=0.18).next_to(index, LEFT, buff=0.35)
            self.play(FadeIn(index, lag_ratio=0.02), run_time=T(0.5))
            self.play(LaggedStart(*(TransformFromCopy(VGroup(anchor.copy(), l, d.copy()), c)
                                    for l, d, c in zip(links, zone, codes)), lag_ratio=0.2), run_time=T(1.1))
            hits = [index[i] for i in (4, 5, 15, 16, 24)][: len(codes)]
            self.play(*(c.animate.scale(0.4).set_opacity(0).move_to(h) for c, h in zip(codes, hits)),
                      *(h.animate.set_stroke(GOLD, 5) for h in hits), run_time=s.left(3))
            s.until(4)
            rng = np.random.default_rng(2)
            heights = rng.uniform(0.2, 0.8, 26)
            heights[17] = 3.6
            axis = Line(LEFT * 4.6, RIGHT * 4.6).set_stroke(SOFT, 1.5).move_to(DOWN * 2.45)
            bars = VGroup(*(bar(h, 0.28, GOLD if i == 17 else DIM, 1) for i, h in enumerate(heights)))
            bars.arrange(RIGHT, buff=0.07, aligned_edge=DOWN).next_to(axis, UP, buff=0)
            hist = VGroup(axis, bars)
            self.play(FadeOut(Group(img, halo, dots, box, links, index, anchor)), w.animate.move_to(UP * 2.65), run_time=T(0.6))
            self.play(ShowCreation(axis), LaggedStart(*(GrowFromEdge(b, DOWN) for b in bars), lag_ratio=0.03),
                      run_time=s.left())

        # -- the same line, two reciters: their points don't coincide; their words do
        a = spec("line2_a", 1.7).set_width(10.2, stretch=True).move_to(UP * 2.0)
        b = spec("line2_b", 1.7).set_width(10.2, stretch=True).move_to(DOWN * 0.95)
        ha, da = peak_dots(a, pk["a"], int(pk["n_a"]), INK, r=0.06, glow=False)
        hb, db = peak_dots(b, pk["b"], int(pk["n_b"]), OTHER, r=0.06, glow=False)
        with self.say("catch1") as s:
            self.play(FadeOut(Group(w, hist, logo_s), shift=LEFT * 2), run_time=T(0.7))
            self.play(FadeIn(a), run_time=T(0.6))
            self.play(a.animate.set_opacity(0.15), LaggedStartMap(FadeIn, da, lag_ratio=0.01), FadeIn(ha),
                      run_time=s.left(0))
            s.until(1)
            self.play(FadeIn(b), run_time=T(0.6))
            self.play(b.animate.set_opacity(0.15), LaggedStartMap(FadeIn, db, lag_ratio=0.01), FadeIn(hb), run_time=T(0.9))
            ghost = db.copy()
            self.play(ghost.animate.shift(a.get_center() - b.get_center()), run_time=s.left(1))
            s.until(2)
            line2 = m["lines"]["2"]["ar"]
            ta = ar(line2, 38).next_to(a, DOWN, buff=0.2)
            tb = ar(line2, 38).next_to(b, DOWN, buff=0.2)
            self.play(FadeIn(ta, shift=UP * 0.2), FadeIn(tb, shift=UP * 0.2), run_time=T(0.9))
            self.play(FadeOut(Group(a, b, da, db, ha, hb, ghost)), tb.animate.move_to(UP * 0.4), ta.animate.move_to(UP * 0.4),
                      run_time=s.left())
            self.remove(tb)
        words = ta

        # -- Tarteel: an app for reciting the Quran that listens for the words
        logo_t = ImageMobject(str(IMG / "logo_tarteel.png"), height=1.7).move_to(UP * 2.4)
        with self.say("tarteel1") as s:
            self.play(FadeIn(logo_t, scale=0.85), words.animate.scale(0.95).move_to(LEFT * 0.75 + DOWN * 0.25),
                      run_time=T(1.1))
            self.play(logo_t.animate.set_height(0.8).to_corner(UL, buff=0.35), run_time=max(T(0.6), s.left(0)))
            s.until(1)
            mic = mic_icon(INK, 1.3).move_to(LEFT * 6.0 + DOWN * 0.25)
            page = page_icon().move_to(RIGHT * 5.55 + DOWN * 0.25)
            a1 = Arrow(mic.get_right(), words.get_left(), buff=0.25).set_stroke(SOFT, 3)
            a2 = Arrow(words.get_right(), page.get_left(), buff=0.25).set_stroke(SOFT, 3)
            lit = page[1][3]
            self.play(FadeIn(mic), run_time=T(0.5))
            self.play(GrowArrow(a1), words.animate.set_fill(GOLD), run_time=T(0.9))
            self.play(GrowArrow(a2), FadeIn(page), run_time=T(1.0))
            cells = VGroup(*(Line(LEFT, RIGHT).set_width(lit.get_width() / 5 - 0.07) for _ in range(5)))
            cells.arrange(LEFT, buff=0.07).move_to(lit).set_stroke(INK, 4)
            self.play(lit.animate.set_stroke(INK, 4), page[0].animate.set_stroke(INK), run_time=T(0.5))
            hl = cells[0].copy().set_stroke(GOLD, 9)
            self.play(FadeIn(hl), run_time=T(0.3))
            for c in cells[1:]:
                self.play(hl.animate.move_to(c), run_time=T(0.4))
            s.until(2)
            # the page hidden: each word shows only once it is recited; a missed one is marked
            self.play(*(r.animate.set_stroke(opacity=0.1) for r in page[1]), FadeOut(hl), FadeOut(cells),
                      run_time=T(0.6))
            shown = cells.copy().set_stroke(INK, 4, 1)
            miss = shown[3].copy().set_stroke(WARN, 6)
            for i, c in enumerate(shown):
                self.play(ShowCreation(miss if i == 3 else c), run_time=T(0.45))
            self.wait(max(0.05, s.left() - 0.1))
        with self.say("tarteel2") as s:
            self.play(FadeOut(Group(mic, a1, words, a2, page, *shown[:3], *shown[4:], miss)), run_time=T(0.6))
            # an earlier app, on the phone's own speech recognition
            ph = RoundedRectangle(width=1.6, height=2.9, corner_radius=0.22).set_stroke(INK, 2.5).move_to(LEFT * 4.9 + UP * 0.3)
            pm = mic_icon(SOFT, 0.75).move_to(ph.get_center() + DOWN * 0.55)
            self.play(ShowCreation(ph), FadeIn(pm), run_time=max(T(0.8), s.left(0)))
            s.until(1)
            # good enough to find the ayah, but more mistakes than the reciter
            said = arabic_line(m["lines"]["2"]["ar"], 34, INK, 0.22).move_to(RIGHT * 1.7 + UP * 1.3)
            heard_ = said.copy().move_to(RIGHT * 1.7 + UP * 0.1)
            l1 = en("recited", 26, SOFT).next_to(said, LEFT, buff=0.45)
            l2 = en("heard", 26, SOFT).next_to(heard_, LEFT, buff=0.45).align_to(l1, LEFT)
            self.play(FadeIn(said), FadeIn(l1), run_time=T(0.7))
            self.play(TransformFromCopy(said, heard_), FadeIn(l2), run_time=T(0.9))
            wrong = [heard_[2], heard_[5], heard_[6]]
            self.play(*(w_.animate.set_fill(WARN) for w_ in wrong), run_time=max(T(0.6), s.left(1)))
            s.until(2)
            # crowdsourced recordings, curated, and models of their own
            self.play(FadeOut(VGroup(ph, pm, said, heard_, l1, l2)), run_time=T(0.5))
            rng = np.random.default_rng(5)
            tiles = VGroup(*(mini_wave(rng, 0.55, 0.26, SOFT, 22) for _ in range(40)))
            tiles.arrange_in_grid(4, 10, h_buff=0.2, v_buff=0.14).move_to(UP * 0.6)
            mins = num(0, 44).move_to(DOWN * 1.35 + LEFT * 0.6)
            mins_l = en("minutes", 30, SOFT)
            mins_l.add_updater(lambda l: l.next_to(mins, RIGHT, buff=0.15, aligned_edge=DOWN))
            self.add(mins, mins_l)
            self.play(LaggedStartMap(FadeIn, tiles, lag_ratio=0.03), ChangeDecimalToValue(mins, 75000), run_time=T(1.8))
            mins_l.clear_updaters()
            model = RoundedRectangle(width=2.2, height=1.1, corner_radius=0.14).set_stroke(INK, 2.5).move_to(UP * 0.6)
            self.play(LaggedStart(*(t.animate.move_to(model).scale(0.3).set_opacity(0) for t in tiles), lag_ratio=0.01),
                      FadeIn(model), run_time=max(T(1.1), s.left(2)))
            s.until(3)
            card = chip("whisper-base-ar-quran", INK, 28).move_to(model)
            self.play(ReplacementTransform(model, card[0]), FadeIn(card[1]), FadeOut(VGroup(mins, mins_l)), run_time=T(0.9))
            self.play(card.animate.scale(0.75).move_to([5.0, -2.35, 0]), run_time=s.left())
        tarteel = Group(logo_t, card)

        # -- what we took from each, and what du'as add
        with self.say("took1") as s:
            self.play(FadeOut(tarteel), run_time=T(0.6))
            idx = VGroup(*(Line(LEFT, RIGHT).set_width(1.5).set_stroke(DIM, 3) for _ in range(14))).arrange(DOWN, buff=0.11)
            for i in (3, 9):
                idx[i].set_stroke(GOLD, 5)
            tall = VGroup(*(bar(h, 0.17, GOLD if i == 4 else DIM, 1) for i, h in enumerate([.3, .4, .2, .35, 1.9, .3, .4])))
            tall.arrange(RIGHT, buff=0.05, aligned_edge=DOWN).next_to(idx, RIGHT, buff=0.6, aligned_edge=DOWN)
            shz = VGroup(idx, tall).move_to(LEFT * 3.6 + UP * 1.55)
            ls = ImageMobject(str(IMG / "logo_shazam.png"), height=0.75).next_to(shz, UP, buff=0.25)
            self.play(FadeIn(ls, scale=0.8), LaggedStartMap(FadeIn, idx, lag_ratio=0.04),
                      LaggedStart(*(GrowFromEdge(b_, DOWN) for b_ in tall)), run_time=s.left(0))
            s.until(1)
            tw = arabic_line(m["lines"]["2"]["ar"], 36, INK, 0.16).move_to(RIGHT * 3.5 + UP * 1.6)
            under = Line(LEFT, RIGHT).set_width(tw[0].get_width()).set_stroke(GOLD, 6).next_to(tw[0], DOWN, buff=0.12)
            lt = ImageMobject(str(IMG / "logo_tarteel.png"), height=0.75).next_to(tw, UP, buff=0.4)
            self.play(FadeIn(lt, scale=0.8), FadeIn(tw), run_time=T(0.6))
            self.play(ShowCreation(under), run_time=T(0.4))
            for wd in tw[1:4]:
                self.play(under.animate.set_width(wd.get_width()).next_to(wd, DOWN, buff=0.12), run_time=T(0.35))
            s.until(2)
            dots = VGroup(*(Dot(radius=0.042) for _ in range(522))).arrange_in_grid(18, 29, buff=0.085)
            dots.set_fill(DIM, 1).move_to(LEFT * 3.6 + DOWN * 1.25)
            share = np.zeros(522, bool)
            share[np.random.default_rng(1).choice(522, 412, replace=False)] = True
            self.play(FadeIn(dots, lag_ratio=0.002), run_time=T(0.7))
            self.play(*(d.animate.set_fill(INK) for d, s_ in zip(dots, share) if s_), run_time=T(0.9))
            rows = VGroup(*(Line(LEFT, RIGHT).set_width(4.0).set_stroke(DIM, 5) for _ in range(5)))
            rows.arrange(DOWN, buff=0.42).move_to(RIGHT * 3.6 + DOWN * 1.25)
            dot = Dot(radius=0.11).set_fill(GOLD, 1).move_to(rows[1].get_right())
            self.play(FadeIn(rows), FadeIn(dot), run_time=T(0.5))
            self.play(dot.animate.move_to(rows[1].get_center()), run_time=T(0.7))
            self.play(Indicate(dot, scale_factor=1.5, color=GOLD), run_time=T(0.7))
            self.play(dot.animate.move_to(rows[0].get_right()), run_time=T(0.6))
            bubble = VGroup(RoundedRectangle(width=0.8, height=0.48, corner_radius=0.2).set_stroke(SOFT, 2.5),
                            Polygon(ORIGIN, RIGHT * 0.14, DOWN * 0.18 + RIGHT * 0.02).set_stroke(SOFT, 2.5))
            bubble[1].next_to(bubble[0], DOWN, buff=-0.02).shift(LEFT * 0.18)
            bubble.next_to(rows, LEFT, buff=0.35).shift(UP * 0.5)
            self.play(FadeIn(bubble, scale=0.6), run_time=max(0.4, s.left() - 0.5))
            self.play(FadeOut(Group(shz, tw, under, dots, rows, dot, bubble, ls, lt)), run_time=T(0.5))


class P2(VoiceScene):
    """The data: the texts, the timed recitations, the harvest and how it was labelled, ordinary voices."""

    def construct(self):
        z, m = data()
        rng = np.random.default_rng(7)
        xs = [-4.7, 0.0, 4.7]
        top = 2.35

        with self.say("data1") as s:
            words = z["dua_words"].astype(float)
            hs = 0.2 + 1.3 * np.log1p(words) / np.log1p(words.max())
            sky = VGroup(*(bar(h, 3.7 / 522, INK, 0.9) for h in hs)).arrange(RIGHT, buff=0, aligned_edge=DOWN)
            sky.move_to([xs[0], top, 0])
            texts_n = en("522 texts", 30, SOFT).move_to([xs[0], 1.25, 0])
            count = num(0, 40).move_to([xs[0] - 0.55, 0.7, 0])
            wl = en("words", 28, SOFT)
            wl.add_updater(lambda w_: w_.next_to(count, RIGHT, buff=0.15, aligned_edge=DOWN))
            self.play(LaggedStart(*(GrowFromEdge(b_, DOWN) for b_ in sky), lag_ratio=0.002), run_time=T(1.5))
            self.add(wl)
            self.play(FadeIn(texts_n), FadeIn(count), run_time=T(0.4))
            self.play(ChangeDecimalToValue(count, 140803), run_time=max(1.0, s.left(1) * 0.8))
            wl.clear_updaters()
            s.until(2)
            wv = wave(1.5, 31.9, 3.9, 1.1).move_to([xs[1], top, 0])
            first = {}
            for ln, _, a_, _ in m["said"]:
                if a_ <= 31.9:
                    first.setdefault(ln, a_)
            ticks = VGroup()
            for ln, a_ in first.items():
                x = wv.get_left()[0] + (a_ - 1.5) / (31.9 - 1.5) * wv.get_width()
                ticks.add(Line([x, top + 0.75, 0], [x, top - 0.75, 0]).set_stroke(GOLD, 2.5))
            dp = en("DuaPlayer, timed by hand", 28, SOFT).move_to([xs[1], 1.25, 0])
            self.play(FadeIn(wv), run_time=T(0.7))
            self.play(LaggedStartMap(ShowCreation, ticks, lag_ratio=0.15), FadeIn(dp), run_time=s.left())

        with self.say("data2") as s:
            names = VGroup(*(en(n, 30, INK) for n in ("YouTube", "Shiavoice", "Aparat", "SoundCloud", "Internet Archive")))
            names.arrange(DOWN, buff=0.22, aligned_edge=LEFT).move_to([-5.6, -1.55, 0])
            funnel = VGroup(Line([-3.95, -0.45, 0], [-2.35, -1.35, 0]), Line([-3.95, -2.75, 0], [-2.35, -1.85, 0]))
            funnel.set_stroke(SOFT, 2.5)
            self.play(LaggedStartMap(FadeIn, names, shift=RIGHT * 0.2, lag_ratio=0.12), ShowCreation(funnel), run_time=T(1.3))
            recs = num(0, 34).move_to([-3.2, 0.25, 0])
            recs_l = en("recordings,", 28, SOFT)
            hrs = num(0, 34)
            hrs_l = en("hours", 28, SOFT)
            recs_l.add_updater(lambda l: l.next_to(recs, RIGHT, buff=0.12, aligned_edge=DOWN))
            hrs.add_updater(lambda h: h.next_to(recs_l, RIGHT, buff=0.22, aligned_edge=DOWN))
            hrs_l.add_updater(lambda l: l.next_to(hrs, RIGHT, buff=0.12, aligned_edge=DOWN))
            stream = VGroup(*(Dot(radius=0.035).set_fill(SOFT, 0.9).move_to(names[rng.integers(5)].get_right() + RIGHT * 0.2)
                              for _ in range(90)))
            self.add(recs, recs_l, hrs, hrs_l)
            self.play(LaggedStart(*(d.animate.move_to([-2.45, -1.6 + rng.uniform(-0.1, 0.1), 0]).set_opacity(0)
                                    for d in stream), lag_ratio=0.02),
                      ChangeDecimalToValue(recs, 21340), ChangeDecimalToValue(hrs, 5890), run_time=s.left(0))
            for mob in (recs_l, hrs, hrs_l):
                mob.clear_updaters()
            s.until(1)
            box = box_label("letter model", TEAL, 2.1, 1.0, 26).move_to([-0.9, -1.6, 0])
            tile = mini_wave(rng, 2.6, 0.7, SOFT, 90).move_to([2.25, -1.6, 0])
            self.play(FadeIn(box), run_time=T(0.4))
            letters = VGroup(*(ar(c, 28, TEAL) for c in "اللهماني")).arrange(LEFT, buff=0.06).next_to(box, UP, buff=0.2)
            self.play(LaggedStartMap(FadeIn, letters, lag_ratio=0.1), run_time=T(0.7))
            target = sky[int(m["kumayl"])]
            self.play(letters.animate.scale(0.4).move_to(target.get_top()).set_opacity(0),
                      target.animate.set_fill(TEAL, 1).stretch(4, 0), run_time=T(1.0))
            self.play(FadeIn(tile, shift=LEFT * 0.3), run_time=T(0.5))
            marks = VGroup(*(Line(UP * 0.45, DOWN * 0.45).set_stroke(GOLD, 2.5).move_to(tile.get_left() + RIGHT * x)
                             for x in np.linspace(0.15, 2.45, 7)))
            junk = VGroup(*(mini_wave(rng, 0.32, 0.3, DIM, 12).move_to(tile.get_right() + RIGHT * (0.35 + 0.36 * k))
                            for k in range(3)))
            self.play(LaggedStartMap(ShowCreation, marks, lag_ratio=0.1), FadeIn(junk), run_time=T(0.9))
            self.play(junk.animate.shift(DOWN * 1.0).set_opacity(0), run_time=T(0.6))
            usable = num(0, 40, TEAL).move_to([5.0, -1.25, 0])
            usable_l = en("hours", 28, SOFT)
            vcs = num(0, 40, TEAL).move_to([5.0, -1.95, 0])
            vcs_l = en("voices", 28, SOFT)
            usable_l.add_updater(lambda l: l.next_to(usable, RIGHT, buff=0.12, aligned_edge=DOWN))
            vcs_l.add_updater(lambda l: l.next_to(vcs, RIGHT, buff=0.12, aligned_edge=DOWN))
            self.add(usable, usable_l, vcs, vcs_l)
            self.play(ChangeDecimalToValue(usable, 1400), ChangeDecimalToValue(vcs, 3500), run_time=s.left())
            usable_l.clear_updaters()
            vcs_l.clear_updaters()
        harvest = VGroup(names, funnel, recs, recs_l, hrs, hrs_l, box, tile, marks, usable, usable_l, vcs, vcs_l)

        with self.say("data3") as s:
            self.play(harvest.animate.set_opacity(0.3), run_time=T(0.6))
            people = VGroup(*(mini_wave(rng, 0.85, 0.3, c, 26) for c in
                              [INK, SOFT, "#C9B79C", "#9CC9B7", "#B79CC9", "#C99CA8"] * 2))
            people.arrange_in_grid(4, 3, h_buff=0.3, v_buff=0.2).move_to([xs[2], top, 0])
            ov = en("ordinary voices", 28, SOFT).move_to([xs[2], 1.25, 0])
            self.play(LaggedStartMap(FadeIn, people, lag_ratio=0.05), FadeIn(ov), run_time=T(1.2))
            src = people[4]
            clone = src.copy()
            self.play(Indicate(src, scale_factor=1.25, color=OTHER), run_time=T(0.6))
            line3 = ar(m["lines"]["3"]["ar"], 30, OTHER).move_to([xs[2], 0.05, 0])
            self.play(clone.animate.scale(1.4).move_to([xs[2], 0.65, 0]).set_fill(OTHER).set_stroke(OTHER),
                      FadeIn(line3), run_time=s.left(0))
            s.until(1)
            out = [people[1], people[8], tile]
            crosses = VGroup(*(Cross(p).set_stroke(WARN, 4).scale(0.7) for p in out))
            self.play(tile.animate.set_opacity(1), LaggedStartMap(ShowCreation, crosses, lag_ratio=0.2), run_time=T(0.9))
            self.play(*(FadeOut(VGroup(p, c), shift=DOWN * 0.4) for p, c in zip(out, crosses)), run_time=s.left())
        self.play(FadeOut(Group(*self.mobjects)), run_time=T(0.6))


class P3(VoiceScene):
    """The two models side by side, phone and server, and how they were trained."""

    def construct(self):
        z, m = data()
        top = spec("win_b", 1.15).set_width(6.0, stretch=True).move_to(UP * 3.0)
        cards = []
        for x, col, name in ((-3.6, GOLD, "Whisper"), (3.6, TEAL, "letter model")):
            c = RoundedRectangle(width=6.7, height=4.85, corner_radius=0.18).set_stroke(col, 2.5).set_fill(col, 0.035)
            c.move_to([x, -0.3, 0])
            t = en(name, 34, col).next_to(c.get_corner(UL), DR, buff=0.22)
            link = Line(top.get_bottom() + RIGHT * x * 0.3, c.get_top()).set_stroke(col, 2, 0.6)
            cards.append(VGroup(c, t, link))
        gold, teal = cards
        G, TC = gold[0].get_center(), teal[0].get_center()

        with self.say("models1") as s:
            self.play(FadeIn(top), run_time=T(0.8))
            self.play(LaggedStart(*(FadeIn(c[0], shift=UP * 0.3) for c in cards), lag_ratio=0.25),
                      *(ShowCreation(c[2]) for c in cards), *(FadeIn(c[1]) for c in cards), run_time=s.left())

        with self.say("whisper1") as s:
            enc = VGroup(*(RoundedRectangle(width=0.3, height=1.35, corner_radius=0.06).set_stroke(GOLD, 1.8)
                           .set_fill(GOLD, 0.18) for _ in range(6))).arrange(RIGHT, buff=0.08)
            dec = enc.copy()
            enc.move_to(G + LEFT * 1.55 + UP * 0.75)
            dec.move_to(G + RIGHT * 1.45 + UP * 0.75)
            arr = Arrow(enc.get_right(), dec.get_left(), buff=0.12).set_stroke(GOLD, 3)
            listens = en("listens", 26, SOFT).next_to(enc, UP, buff=0.12)
            writes = en("writes", 26, SOFT).next_to(dec, UP, buff=0.12)
            bracket = Rectangle(width=6.0, height=1.25).set_stroke(GOLD, 3.5).move_to(top)
            self.play(ShowCreation(bracket), run_time=T(0.7))
            self.play(TransformFromCopy(bracket, enc), FadeIn(listens), run_time=T(1.0))
            self.play(GrowArrow(arr), FadeIn(dec, lag_ratio=0.1), FadeIn(writes), run_time=s.left(1))
            s.until(2)
            text = next(u["heard"] for u in m["updates"] if "برحمتك" in u["heard"])
            ws = VGroup(*(ar(w_, 38, GOLD) for w_ in text.split())).arrange(LEFT, buff=0.22)
            ws.move_to(G + DOWN * 0.75).set_max_width(6.0)
            for w_ in ws:
                self.play(FadeIn(w_, shift=LEFT * 0.15), run_time=T(0.32))
            s.until(3)
            d_phone = disc(74, GOLD, "74 M · phone", 0.028).move_to(G + LEFT * 2.35 + DOWN * 1.55)
            d_srv = disc(809, GOLD, "809 M · server", 0.028).move_to(G + RIGHT * 2.05 + DOWN * 1.35)
            card = chip("whisper-base-ar-quran", INK, 22).next_to(d_phone[0], RIGHT, buff=0.3)
            self.play(FadeOut(ws), GrowFromCenter(d_phone[0]), FadeIn(d_phone[1]), run_time=T(0.7))
            self.play(GrowFromCenter(d_srv[0]), FadeIn(d_srv[1]), run_time=T(0.7))
            self.play(FadeIn(card, shift=LEFT * 0.3), run_time=s.left())

        with self.say("ctc1") as s:
            self.play(bracket.animate.set_stroke(TEAL), run_time=T(0.4))
            # the frames of "bi-rahmatika": the likeliest letter every 20 ms (a long run of one letter shown short),
            # blanks as dots
            wb = [x for x in m["said"] if x[0] == 2][3]
            a_, b_ = int(wb[2] / 0.02) - 3, int(wb[3] / 0.02) + 3
            ids = [int(i) for i in z["letter"][a_:b_]]
            runs = []
            for i in ids:
                if runs and runs[-1][0] == i:
                    runs[-1][1] += 1
                else:
                    runs.append([i, 1])
            seq = [i for i, n in runs for _ in range(min(n, 3 if i else 2))]
            cols = VGroup(*((ar(chr(0x0621 + i - 1), 32, TEAL) if i > 0 else Dot(radius=0.045).set_fill(SOFT, 0.9))
                            for i in seq))
            step = min(0.36, 6.0 / max(1, len(cols)))
            for k, c in enumerate(cols):
                c.move_to(TC + np.array([(len(cols) - 1) / 2 * step - k * step, 1.0, 0]))
            ticks = VGroup(*(Line(UP * 0.1, DOWN * 0.1).set_stroke(TEAL, 1.5, 0.5).move_to(c.get_center() + DOWN * 0.42)
                             for c in cols))
            s.until(1)
            self.play(LaggedStart(*(FadeIn(c, shift=DOWN * 0.1) for c in cols), lag_ratio=0.04), FadeIn(ticks),
                      run_time=s.left(1))
            s.until(2)
            keep, prev = [], None
            for c, i in zip(cols, seq):
                if i > 0 and i != prev:
                    keep.append(c)
                prev = i
            gone = [c for c in cols if c not in keep]
            spelled = "".join(chr(0x0621 + i - 1) for i, _ in runs if i > 0)
            joined = ar(spelled, 52, TEAL).move_to(TC + DOWN * 0.05)
            self.play(*(FadeOut(g, scale=0.5) for g in gone), FadeOut(ticks), run_time=T(0.7))
            self.play(*(c.animate.move_to(joined.get_center()) for c in keep), run_time=T(0.7))
            self.play(FadeOut(VGroup(*keep)), FadeIn(joined), run_time=s.left(2))
            s.until(3)
            encT = enc.copy().set_stroke(TEAL).set_fill(TEAL, 0.18).move_to(TC + LEFT * 1.6 + DOWN * 1.55)
            layer = RoundedRectangle(width=0.16, height=1.35, corner_radius=0.05).set_stroke(TEAL, 2).set_fill(TEAL, 0.5)
            layer.next_to(encT, RIGHT, buff=0.3)
            alpha = VGroup(*(ar(chr(c), 18, TEAL) for c in range(0x0628, 0x063B, 3))).arrange(DOWN, buff=0.0)
            alpha.set_height(1.35).next_to(layer, RIGHT, buff=0.25)
            d20 = disc(20, TEAL, "20 M · both", 0.028).move_to(TC + RIGHT * 2.3 + DOWN * 1.55)
            self.play(joined.animate.move_to(TC + UP * 0.85), run_time=T(0.4))
            self.play(TransformFromCopy(enc, encT), run_time=T(0.9))
            self.play(FadeIn(layer, shift=RIGHT * 0.2), FadeIn(alpha, lag_ratio=0.05), run_time=T(0.7))
            self.play(GrowFromCenter(d20[0]), FadeIn(d20[1]), run_time=s.left())

        with self.say("train1") as s:
            self.play(FadeOut(Group(*self.mobjects)), run_time=T(0.5))
            clean = spec("aug_clean", 1.15).set_width(3.3, stretch=True).move_to(UP * 2.9)
            label = ar(m["lines"]["2"]["ar"], 32, INK).next_to(clean, DOWN, buff=0.2)
            self.play(FadeIn(clean), run_time=T(0.6))
            self.play(FadeIn(label, shift=DOWN * 0.2), run_time=s.left(0))
            s.until(1)
            names = [("aug_faster", "faster"), ("aug_higher", "higher voice"), ("aug_noise", "noise"),
                     ("aug_room", "a room"), ("aug_hall", "hall, loudspeakers, people"), ("aug_masked", "stripes blanked")]
            grid = Group()
            for img, lab in names:
                im = spec(img, 1.05).set_width(3.3, stretch=True)
                grid.add(Group(im, en(lab, 26, SOFT).next_to(im, DOWN, buff=0.1)))
            grid.arrange_in_grid(2, 3, h_buff=0.5, v_buff=0.3).move_to(DOWN * 0.95)
            for g in grid:
                src = clean.copy()
                self.add(src)
                self.play(FadeTransform(src, g[0]), FadeIn(g[1]), run_time=T(0.7))
            self.wait(max(0.05, s.left() - 0.05))

        with self.say("distil1") as s:
            self.play(FadeOut(Group(*self.mobjects)), run_time=T(0.45))
            big = disc(809, GOLD, "809 M", 0.033).move_to(LEFT * 3.0 + UP * 1.75)
            small = disc(74, GOLD, "74 M", 0.033).move_to(RIGHT * 3.6 + UP * 1.75)
            hear_big = spec("aug_clean", 0.85).set_width(2.0, stretch=True).next_to(big[0], LEFT, buff=0.45)
            hear_small = spec("aug_hall", 0.85).set_width(2.0, stretch=True).next_to(small[0], RIGHT, buff=0.45)
            p_t = np.array([0.62, 0.18, 0.09, 0.05, 0.04, 0.02])
            p_s = np.array([0.25, 0.22, 0.2, 0.15, 0.1, 0.08])
            bt = VGroup(*(bar(1.6 * p, 0.3, GOLD) for p in p_t)).arrange(RIGHT, buff=0.08, aligned_edge=DOWN)
            bs = VGroup(*(bar(1.6 * p, 0.3, GOLD, 0.45) for p in p_s)).arrange(RIGHT, buff=0.08, aligned_edge=DOWN)
            bt.move_to([big.get_x(), -0.25, 0], aligned_edge=DOWN)
            bs.move_to([small.get_x(), -0.25, 0], aligned_edge=DOWN)
            arrow = Arrow(big[0].get_right(), small[0].get_left(), buff=0.3).set_stroke(GOLD, 3)
            self.play(GrowFromCenter(big[0]), FadeIn(big[1]), FadeIn(hear_big), run_time=T(0.7))
            self.play(GrowArrow(arrow), GrowFromCenter(small[0]), FadeIn(small[1]), FadeIn(hear_small), run_time=T(0.7))
            self.play(LaggedStart(*(GrowFromEdge(b_, DOWN) for b_ in bt), lag_ratio=0.08),
                      LaggedStart(*(GrowFromEdge(b_, DOWN) for b_ in bs), lag_ratio=0.08), run_time=T(0.8))
            target = bt.copy().move_to(bs, aligned_edge=DOWN).set_fill(GOLD, 0.85)
            self.play(Transform(bs, target), run_time=T(1.3))
            t300 = disc(300, TEAL, "300 M", 0.033).move_to(LEFT * 3.0 + DOWN * 1.9)
            t20 = disc(20, TEAL, "20 M", 0.033).move_to(RIGHT * 3.6 + DOWN * 1.9)
            arrow2 = Arrow(t300[0].get_right(), t20[0].get_left(), buff=0.3).set_stroke(TEAL, 3)
            self.play(FadeIn(t300), GrowArrow(arrow2), FadeIn(t20), run_time=s.left())
        self.play(FadeOut(Group(*self.mobjects)), run_time=T(0.5))


class Ribbon(VGroup):
    """All 522 texts end to end, each as a stretch proportional to its words."""

    def __init__(self, width: float = 13.0, height: float = 0.3, y: float = 0.0):
        z, m = data()
        words = z["dua_words"].astype(float)
        self.edges = np.r_[0, np.cumsum(words)] / words.sum() * width - width / 2
        segs = VGroup()
        for i in range(len(words)):
            r = Rectangle(width=max(self.edges[i + 1] - self.edges[i], 0.004), height=height)
            r.set_fill(INK if i % 2 else SOFT, 0.5).set_stroke(width=0)
            r.move_to([(self.edges[i] + self.edges[i + 1]) / 2, y, 0])
            segs.add(r)
        super().__init__(segs)
        self.segs, self.y_, self.width_, self.height_ = segs, y, width, height

    def x_of_text(self, i: int) -> float:
        return float((self.edges[i] + self.edges[i + 1]) / 2)


def fit_spikes(rib: Ribbon, depth: float = 1.5, bins: int = 900, k: float = 0.7) -> VMobject:
    """How well the heard letters fit, ending at each stretch of the texts: exp(-k * extra letters to change)."""
    z, _ = data()
    c = z["cost"].astype(float)
    edges = np.linspace(0, len(c), bins + 1).astype(int)
    lo = np.array([c[a:b].min() for a, b in zip(edges[:-1], edges[1:])])
    fit = np.exp(-k * (lo - lo.min()))
    xs = np.linspace(-rib.width_ / 2, rib.width_ / 2, bins)
    y0 = rib.y_ - rib.height_ / 2 - 0.08
    pts = []
    for x, f in zip(xs, fit):
        pts += [np.array([x, y0, 0]), np.array([x, y0 - depth * f, 0]), np.array([x, y0, 0])]
    return VMobject().set_points_as_corners(pts).set_stroke(GOLD, 2.2)


class P4(VoiceScene):
    """Recognition: Whisper's words, matched against every text, the tracker, results."""

    def construct(self):
        z, m = data()
        heard = "اللهم إني اسالك برحمتك التي"  # what the page's Whisper heard in the window ending at 16.97 s

        strip = wave(9.0, 21.0, 12.6, 1.1).move_to(UP * 2.85)
        x_of = lambda t: strip.get_left()[0] + (t - 9.0) / 12.0 * strip.get_width()  # noqa: E731
        win = Rectangle(width=6.0 / 12.0 * strip.get_width(), height=1.45).set_stroke(GOLD, 3.5)
        win.move_to([x_of(16.97 - 3.0), 2.85, 0])
        with self.say("rec1") as s:
            self.play(FadeIn(strip), run_time=s.left())
        # Line 2 whole, from the quiet after his salawat (12.25 s) to the quiet after the line (19.7 s). As the
        # playhead reaches 16.97 s, Whisper takes the last six seconds and writes them down.
        h0, hw, h1 = 12.25, 16.97, 19.7
        head = Line(UP * 0.72, DOWN * 0.72).set_stroke(INK, 2.5).move_to([x_of(h0), 2.85, 0])
        self.recite(h0, h1)
        self.add(head)
        self.play(head.animate.move_to([x_of(hw), 2.85, 0]), run_time=hw - h0, rate_func=linear)
        words = VGroup(*(ar(w_, 48, GOLD) for w_ in heard.split())).arrange(LEFT, buff=0.32).move_to(UP * 1.0)
        self.play(head.animate.move_to([x_of(h1), 2.85, 0]), ShowCreation(win),
                  LaggedStart(*(FadeIn(w_, shift=DOWN * 0.2) for w_ in words), lag_ratio=0.15), run_time=h1 - hw,
                  rate_func=linear)
        self.remove(head)
        with self.say("rec1b") as s:
            tiles = letter_tiles(fold(heard), 36, GOLD, 0.52, 0.66).move_to(UP * 1.0)
            self.play(FadeOut(words, shift=UP * 0.2), LaggedStart(*(FadeIn(t, shift=DOWN * 0.1) for t in tiles),
                                                                  lag_ratio=0.03), run_time=s.left())

        rib = Ribbon(13.0, 0.3, y=0.15)
        spikes = fit_spikes(rib)
        with self.say("rec2") as s:
            self.play(FadeOut(VGroup(strip, win)), tiles.animate.scale(0.62).move_to(UP * 2.9), run_time=T(0.7))
            self.play(LaggedStart(*(FadeIn(r) for r in rib.segs), lag_ratio=0.001), run_time=T(0.9))
            slider = tiles.copy().scale(0.55).next_to(rib, UP, buff=0.12).align_to(rib, LEFT)
            self.play(TransformFromCopy(tiles, slider), run_time=T(0.5))
            self.play(slider.animate.align_to(rib, RIGHT), ShowCreation(spikes), run_time=max(1.5, s.left(0)),
                      rate_func=linear)
            self.play(FadeOut(slider), run_time=T(0.3))
            s.until(1)
            a_, b_ = "برحمتك", "بسمك"
            n, k = len(a_), len(b_)
            D = np.zeros((n + 1, k + 1), int)
            D[:, 0], D[0, :] = np.arange(n + 1), np.arange(k + 1)
            for i in range(1, n + 1):
                for j in range(1, k + 1):
                    D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a_[i - 1] != b_[j - 1]))
            cell = 0.4
            grid = VGroup()
            for i in range(n + 1):
                for j in range(k + 1):
                    sq = Square(cell).set_stroke(DIM, 1).move_to([-j * cell, -i * cell, 0])
                    grid.add(VGroup(sq, en(str(D[i, j]), 22, INK).move_to(sq)))
            rows = VGroup(*(ar(c, 26, GOLD).move_to([cell, -(i + 1) * cell, 0]) for i, c in enumerate(a_)))
            colh = VGroup(*(ar(c, 26, INK).move_to([-(j + 1) * cell, cell, 0]) for j, c in enumerate(b_)))
            gbox = VGroup(grid, rows, colh).move_to(LEFT * 3.0 + DOWN * 1.5)
            self.play(rib.segs.animate.shift(UP * 1.45).set_fill(opacity=0.2),
                      spikes.animate.shift(UP * 1.45).set_stroke(opacity=0.3), run_time=T(0.5))
            self.play(FadeIn(rows), FadeIn(colh), run_time=T(0.4))
            order = sorted(range(len(grid)), key=lambda q: (q // (k + 1)) + (q % (k + 1)))
            self.play(LaggedStart(*(FadeIn(grid[q]) for q in order), lag_ratio=0.04), run_time=T(1.5))
            self.play(grid[-1][0].animate.set_stroke(GOLD, 3.5), run_time=T(0.3))
            bits = VGroup(*(Square(0.17).set_stroke(TEAL, 1.2).set_fill(TEAL, 0.1) for _ in range(64)))
            bits.arrange_in_grid(4, 16, buff=0.05).move_to(RIGHT * 3.3 + DOWN * 1.3)
            self.play(TransformFromCopy(VGroup(*grid[0:: k + 1]), VGroup(*bits[: n + 1])),
                      FadeIn(VGroup(*bits[n + 1:]), lag_ratio=0.01), run_time=T(0.7))
            for f in np.random.default_rng(9).random((4, 64)) < 0.5:
                self.play(*(b.animate.set_fill(TEAL, 0.85 if ff else 0.1) for b, ff in zip(bits, f)), run_time=T(0.2))
            ms = en("whole library: 3 ms", 32, TEAL).next_to(bits, DOWN, buff=0.3)
            self.play(FadeIn(ms), run_time=s.left())
        self.play(FadeOut(VGroup(gbox, bits, ms)), rib.segs.animate.shift(DOWN * 1.45).set_fill(opacity=0.5),
                  spikes.animate.shift(DOWN * 1.45).set_stroke(opacity=1), run_time=T(0.5))

        counter = Integer(522, font_size=60, text_config={"font": SERIF}).move_to(UP * 2.8 + RIGHT * 5.8)
        phrase = arabic_line(heard, 46, GOLD, 0.32).move_to(UP * 2.75 + LEFT * 0.5)
        with self.say("rec3") as s:
            self.play(FadeOut(tiles), FadeIn(phrase), FadeIn(counter), run_time=T(0.6))
            self.play(phrase.animate.set_opacity(0.35), run_time=T(0.4))
            names = ["Allāhumma", "Allāhumma innī", "Allāhumma innī asʾaluka"]
            box = lab = None
            for k_, (_, cnt, ids) in enumerate(m["prefix"][:3]):
                if k_ == 0:
                    s.until(1)
                if k_ == 1:
                    s.until(3)
                part = VGroup(*phrase[: k_ + 1])
                nb = SurroundingRectangle(part, buff=0.12).set_stroke(GOLD, 3)
                nl = Text(names[k_], font=SERIF, font_size=30, fill_color=GOLD, slant=ITALIC).next_to(nb, DOWN, buff=0.15)
                keep = set(ids)
                anims = [part.animate.set_opacity(1), ChangeDecimalToValue(counter, cnt),
                         *(r.animate.set_fill(opacity=0.5 if i in keep else 0.06) for i, r in enumerate(rib.segs))]
                anims += [ShowCreation(nb), FadeIn(nl)] if box is None else [Transform(box, nb), Transform(lab, nl)]
                self.play(*anims, run_time=T(0.9) if k_ != 1 else T(0.5))
                if box is None:
                    box, lab = nb, nl
            s.until(5)
            # Jawshan Kabir's sections open with almost these words: only the fourth differs
            jaw = arabic_line("اللَّهُمَّ إِنِّي أَسْألُكَ بِاسْمِكَ", 40, INK, 0.32)
            jaw.align_to(phrase, RIGHT).set_y(1.45)
            for w_, p_ in zip(jaw[:3], phrase[:3]):
                w_.set_x(p_.get_x())
            jaw[3].next_to(jaw[2], LEFT, buff=0.32)
            j = m["dua_ids"].index("dua-jawshan-kabir")
            lo_x, hi_x = rib.edges[j], rib.edges[j + 1]
            focus = Rectangle(width=hi_x - lo_x + 0.12, height=2.3).set_stroke(GOLD, 2.5).move_to([(lo_x + hi_x) / 2, -0.6, 0])
            name = en("Du'a Jawshan Kabir", 30, GOLD).next_to(focus, UP, buff=0.2)
            jl = en("Jawshan Kabir", 26, SOFT).next_to(jaw, LEFT, buff=0.4)
            self.play(FadeOut(lab), phrase[3].animate.set_opacity(1), FadeIn(jaw), FadeIn(jl), ShowCreation(focus),
                      FadeIn(name), run_time=T(0.9))
            d1 = SurroundingRectangle(phrase[3], buff=0.1).set_stroke(WARN, 3)
            d2 = SurroundingRectangle(jaw[3], buff=0.1).set_stroke(WARN, 3)
            self.play(ShowCreation(d1), ShowCreation(d2), spikes.animate.set_stroke(GOLD, 3), run_time=max(T(0.6), s.left()))
        words_up = VGroup(phrase, box, jaw, jl, d1, d2)

        with self.say("rec4") as s:
            self.play(FadeOut(VGroup(focus, name, spikes, counter, words_up)), rib.animate.move_to(DOWN * 2.3),
                      run_time=T(0.6))
            # a map keeps the GPS dot on roads the car could have reached
            road = Line(LEFT * 4.5, RIGHT * 4.5).set_stroke(DIM, 12).move_to(UP * 0.6)
            other = Line(LEFT * 4.5, RIGHT * 4.5).set_stroke(DIM, 12).move_to(UP * 1.75)
            fixes = VGroup(*(Dot(radius=0.07).set_fill(SOFT, 0.95).move_to([x, 0.6 + dy, 0])
                             for x, dy in zip(np.linspace(-3.6, 3.0, 7), [0.5, 0.7, 0.4, 0.75, 0.6, 0.8, 0.55])))
            car = Dot(radius=0.15).set_fill(BLUE, 1).move_to([-3.6, 0.6, 0])
            trail = TracedPath(car.get_center, stroke_color=BLUE, stroke_width=4)
            self.play(ShowCreation(road), ShowCreation(other), run_time=T(0.5))
            self.add(trail, car)
            for f in fixes:
                self.play(FadeIn(f, scale=0.5), car.animate.move_to([f.get_x(), 0.6, 0]), run_time=T(0.36))
            s.until(1)
            trail.clear_updaters()
            self.play(FadeOut(VGroup(road, other, fixes, car, trail)), run_time=T(0.4))
            base = rib.get_top()[1]
            scale_h = 3.6

            def bars_for(p):
                return VGroup(*(bar(scale_h * max(float(p[i]), 0.0), max(rib.edges[i + 1] - rib.edges[i], 0.03), BLUE, 0.9)
                                .move_to([rib.x_of_text(i), base, 0], aligned_edge=DOWN) for i in range(len(rib.segs))))

            line70 = DashedLine(LEFT * 6.5, RIGHT * 6.5).set_stroke(BLUE, 1.8, 0.8).move_to([0, base + 0.7 * scale_h, 0])
            l70 = en("70%", 28, BLUE).next_to(line70, UP, buff=0.08).align_to(line70, RIGHT)
            pb = bars_for(z["prior"].astype(float) * (0.18 / float(z["prior"].max())))
            self.play(LaggedStart(*(GrowFromEdge(b_, DOWN) for b_ in pb), lag_ratio=0.001),
                      *(r.animate.set_fill(opacity=0.4) for r in rib.segs), run_time=T(0.9))
            self.play(ShowCreation(line70), FadeIn(l70), run_time=T(0.5))
            page = {u["at"]: u["u"] for u in json.loads((CACHE / "runs" / "1-kumayl.json").read_text(encoding="utf-8"))["updates"]
                    if u["u"].get("candidates")}
            for at in (18.766, 20.538):
                p = np.zeros(len(rib.segs))
                for c in page[at]["candidates"]:
                    p[m["dua_ids"].index(c["id"])] = c["p"]
                self.play(Transform(pb, bars_for(p)), run_time=T(0.9))
                if at < 20:
                    jx = rib.x_of_text(m["dua_ids"].index("dua-jawshan-kabir"))
                    jw = rib.edges[m["dua_ids"].index("dua-jawshan-kabir") + 1]
                    tag = en("Jawshan Kabir · 70%", 26, BLUE).next_to(np.array([jw, base + 0.7 * scale_h - 0.3, 0]), RIGHT, buff=0.2)
                    self.play(FadeIn(tag), run_time=T(0.4))
                    s.until(2, 0.3)
                    self.play(FadeOut(tag), run_time=T(0.3))
            kx = rib.x_of_text(int(m["kumayl"]))
            kt = en("Du'a Kumayl · 100%", 30, BLUE).move_to([kx + 0.6, base + scale_h + 0.3, 0])
            self.play(FadeIn(kt, shift=UP * 0.2), run_time=s.left())
        state = VGroup(rib, pb, line70, l70, kt)
        # the real page from the quiet before line 3 (19.65 s, "Is it...?" still up) to the quiet after it (25.6 s):
        # the name comes at 20.5 s
        self.shot(19.65, 25.6 - 19.65)
        self.wait(25.6 - 19.65)

        with self.say("rec5") as s:
            self.play(FadeOut(state), run_time=T(0.4))
            axis = Line(LEFT * 1.6, RIGHT * 1.6).set_stroke(SOFT, 1.5).move_to(LEFT * 3.4 + DOWN * 1.6)
            hb = VGroup(*(bar(h, 0.26, GOLD if i == 6 else DIM, 1) for i, h in
                          enumerate([.35, .55, .25, .45, .3, .4, 3.0, .35, .25]))).arrange(RIGHT, buff=0.07, aligned_edge=DOWN)
            hb.next_to(axis, UP, buff=0)
            kb = bar(3.0, 0.6, BLUE).move_to(RIGHT * 3.4 + DOWN * 1.6, aligned_edge=DOWN)
            kname = en("Du'a Kumayl", 30, BLUE).next_to(kb, UP, buff=0.2)
            self.play(ShowCreation(axis), LaggedStart(*(GrowFromEdge(b_, DOWN) for b_ in hb), lag_ratio=0.05), run_time=T(0.9))
            self.play(TransformFromCopy(hb[6], kb), FadeIn(kname), run_time=T(1.1))
            tline = ar(m["lines"]["2"]["ar"], 34, INK).move_to(UP * 2.6)
            self.play(FadeIn(tline, shift=DOWN * 0.2), run_time=max(0.4, s.left() - 0.45))
            self.play(FadeOut(VGroup(axis, hb, kb, kname, tline)), run_time=T(0.4))

        scen = sorted({k.split("|")[0] for k in m["grid"]})
        lanes = ["studio", "majlis", "harvest", "mafatih"]
        with self.say("results1") as s:
            cells = VGroup()
            size = 0.44
            for i, sc in enumerate(scen):
                for j, ln in enumerate(lanes):
                    v = m["grid"].get(f"{sc}|{ln}")
                    sq = Square(size * 0.9).set_stroke(width=0)
                    sq.set_fill(BLUE, 0.15 + 0.85 * max(0.0, v - 0.5) / 0.5 if v else 0.0)
                    if v is None:
                        sq.set_stroke(FAINT, 1)
                    sq.move_to([-4.35 + i * size, 2.65 - j * size, 0])
                    sq.v = v
                    cells.add(sq)
            lane_names = VGroup(*(en(t, 24, SOFT) for t in ("studio", "du'a nights", "uploaded", "Mafatih")))
            for j, t in enumerate(lane_names):
                t.move_to([-4.35 - 0.4, 2.65 - j * size, 0], aligned_edge=RIGHT)
            self.play(LaggedStart(*(FadeIn(c, scale=0.5) for c in cells), lag_ratio=0.004), FadeIn(lane_names),
                      run_time=s.left(0))
            s.until(1)
            heads = VGroup(*(en(t, 30, SOFT) for t in ("on the line", "on the word", "du'a named within 10 s")))
            heads.arrange(RIGHT, buff=1.0).move_to(DOWN * 0.15 + RIGHT * 0.6)
            vals = VGroup(*(Integer(0, font_size=80, text_config={"font": SERIF}, unit="%") for _ in range(3)))
            for v_, h in zip(vals, heads):
                v_.next_to(h, DOWN, buff=0.35)
            phone = en("phone", 30, SOFT).next_to(vals, LEFT, buff=0.9)
            self.play(FadeIn(heads), FadeIn(phone), FadeIn(vals), run_time=T(0.4))
            for v_, target in zip(vals, (91, 76, 94)):
                self.play(ChangeDecimalToValue(v_, target), run_time=T(0.8))
            self.wait(max(0.05, s.left() - 0.05))
        results = VGroup(cells, lane_names, heads, vals, phone)

        with self.say("results2") as s:
            weak = [c for c in cells if c.v is not None and c.v < 0.8]
            self.play(*(c.animate.set_stroke(WARN, 3) for c in weak), run_time=T(0.7))
            s.until(1)
            self.play(FadeOut(results), run_time=T(0.4))
            parts = [
                (person(INK, 0.8), "reciter"),
                (mic_icon(INK, 0.8), "microphone"),
                (box_label("PA", SOFT, 0.9, 0.6, 26), "amplifier"),
                (VGroup(*(Polygon(LEFT * 0.16 + UP * 0.12, LEFT * 0.16 + DOWN * 0.12, RIGHT * 0.16 + DOWN * 0.26,
                                  RIGHT * 0.16 + UP * 0.26).set_stroke(SOFT, 2.5) for _ in range(3))).arrange(DOWN, buff=0.25),
                 "loudspeakers"),
                (VGroup(*(Arc(radius=0.3 + 0.25 * k, start_angle=-PI / 4, angle=PI / 2).set_stroke(SOFT, 2.5, 0.9 - 0.2 * k)
                          for k in range(4))), "a measured hall"),
                (VGroup(*(person(DIM, 0.42) for _ in range(6))).arrange_in_grid(2, 3, buff=0.12), "people talking"),
            ]
            row = VGroup()
            for icon, lab in parts:
                row.add(VGroup(icon, en(lab, 24, SOFT).next_to(icon, DOWN, buff=0.18)))
            row.arrange(RIGHT, buff=0.55, aligned_edge=DOWN).move_to([-1.6, 1.3, 0])
            for p_ in row:
                self.play(FadeIn(p_, shift=RIGHT * 0.15), run_time=T(0.42))
            s.until(2)
            b1 = bar(0.55 * 3.0, 0.75, BLUE, 0.55).move_to([4.6, -2.6, 0], aligned_edge=DOWN)
            b2 = bar(0.81 * 3.0, 0.75, BLUE, 0.9).move_to([6.0, -2.6, 0], aligned_edge=DOWN)
            v1 = Integer(55, font_size=40, text_config={"font": SERIF}, unit="%")
            v2 = Integer(81, font_size=40, text_config={"font": SERIF}, unit="%").next_to(b2, UP, buff=0.1)
            v1.add_updater(lambda v: v.next_to(b1, UP, buff=0.1))
            t1 = en("phone", 26, SOFT).next_to(b1, DOWN, buff=0.1)
            t2 = en("server", 26, SOFT).next_to(b2, DOWN, buff=0.1)
            ol = en("on the line", 26, SOFT).move_to([5.3, 1.1, 0])
            self.play(GrowFromEdge(b1, DOWN), FadeIn(v1), FadeIn(t1), FadeIn(ol), row.animate.shift(LEFT * 0.4), run_time=T(0.6))
            self.play(Transform(b1, bar(0.67 * 3.0, 0.75, BLUE, 0.9).move_to(b1, aligned_edge=DOWN)),
                      ChangeDecimalToValue(v1, 67), run_time=T(1.1))
            v1.clear_updaters()
            self.play(GrowFromEdge(b2, DOWN), FadeIn(v2), FadeIn(t2), run_time=s.left())
        self.play(FadeOut(Group(*self.mobjects)), run_time=T(0.5))


class P5(VoiceScene):
    """Following: letters place the word; moves, pauses, talk; how often; what it costs."""

    def construct(self):
        z, m = data()
        st_t, st_line, st_word = z["step_t"], z["step_line"], z["step_word"]
        lines = [6, 7, 8, 9, 10]
        ys = [2.75, 1.75, 0.75, -0.25, -1.25]
        rows = VGroup()
        for ln, y in zip(lines, ys):
            r = arabic_line(m["lines"][str(ln)]["ar"], 44, INK, 0.24)
            r.move_to([0, y, 0]).align_to(RIGHT * 5.6, RIGHT)
            rows.add(r)
        nums = VGroup(*(en(str(ln), 26, SOFT).move_to([6.05, y, 0]) for ln, y in zip(lines, ys)))
        clock = ValueTracker(35.75)
        run = json.loads((CACHE / "runs" / "1-kumayl.json").read_text(encoding="utf-8"))
        page, seg = [], None
        for e in run["shown"]:
            if e["type"] == "line":
                seg = e["seg"]
            elif e["type"] == "word" and seg is not None:
                page.append((e["t"], seg, e["token"]))
        page_t = np.array([p[0] for p in page])

        def shown(t):
            i = int(np.searchsorted(page_t, t, side="right")) - 1
            if i < 0 or page[i][1] not in lines:
                return None, 0
            return page[i][1], page[i][2]

        hl = Rectangle(width=1.0, height=0.7).set_stroke(width=0).set_fill(GOLD, 0.25)

        def place(r):
            ln, i = shown(clock.get_value())
            if ln is None:
                r.set_fill(opacity=0)
                return
            wd = rows[lines.index(ln)][i]
            r.set_width(wd.get_width() + 0.16, stretch=True).set_height(0.78, stretch=True).move_to(wd).set_fill(GOLD, 0.25)

        probs = VGroup(*(Rectangle(width=0.14, height=0.02).set_fill(BLUE, 0.9).set_stroke(width=0) for _ in lines))

        def set_probs(g):
            i = max(0, int(np.searchsorted(st_t, clock.get_value(), side="right")) - 1)
            for b_, ln, y in zip(g, lines, ys):
                b_.set_height(0.02 + 0.7 * float(st_line[i, ln - 1]), stretch=True).move_to([6.55, y - 0.35, 0],
                                                                                           aligned_edge=DOWN)

        hl.add_updater(place)
        probs.add_updater(set_probs)
        # what Whisper last wrote, shown when the page had it: a beat behind the highlight
        ups = [(u["at"], u["u"]["heard"]) for u in run["updates"] if 34.0 <= u["at"] <= 49.0 and u["u"].get("heard")]
        gold_txt = VGroup(*(ar(h, 30, GOLD).move_to([-3.4, -2.35, 0]) for _, h in ups))
        up_t = np.array([t for t, _ in ups])

        def whisper_text(g):
            k = int(np.searchsorted(up_t, clock.get_value(), side="right")) - 1
            for i, sub in enumerate(g):
                sub.set_fill(opacity=1.0 if i == k else 0.0)

        with self.say("follow1") as s:
            # lines 6 and 7 whole: from the quiet after his pause (35.75 s) to the end of line 7 (47.75 s)
            f0, f1 = 35.75, 47.75
            self.add(hl, gold_txt)
            self.play(FadeIn(rows, lag_ratio=0.1), FadeIn(nums), FadeIn(probs), run_time=T(0.8))
            self.recite(f0, f1)
            gold_txt.add_updater(whisper_text)
            self.play(clock.animate.set_value(f1), run_time=f1 - f0, rate_func=linear)
            gold_txt.clear_updaters()
            self.play(FadeOut(gold_txt), run_time=T(0.3))

        with self.say("follow2") as s:
            hl.clear_updaters()
            here = rows[2]  # line 8
            self.play(hl.animate.move_to(here[-1]).set_width(here[-1].get_width() + 0.16, stretch=True), run_time=T(0.5))
            on = Arrow(here.get_right() + UP * 0.45, here.get_left() + UP * 0.45, buff=0).set_stroke(TEAL, 4)
            rail = min(r.get_left()[0] for r in rows) - 0.45
            y8 = here.get_y()
            again = CurvedArrow(np.array([rail, y8 - 0.18, 0]), np.array([rail, y8 + 0.18, 0]), angle=-PI * 1.4)
            back = CurvedArrow(np.array([rail, y8, 0]), np.array([rail, rows[0].get_y(), 0]), angle=PI / 2.2)
            ahead = CurvedArrow(np.array([rail, y8, 0]), np.array([rail, rows[4].get_y(), 0]), angle=-PI / 2.2)
            ladder = VGroup()
            for k, (lab, c, arc) in enumerate([("again", 3, again), ("back", 6, back), ("ahead", 9, ahead)]):
                arc.set_stroke(SOFT, 2.5)
                stp = bar(0.17 * c, 0.5, WARN, 0.85).move_to([-5.6 + 0.85 * k, -2.55, 0], aligned_edge=DOWN)
                tl = en(lab, 26, SOFT).next_to(stp, UP, buff=0.1)
                ladder.add(VGroup(arc, stp, tl))
            free = VGroup(bar(0.04, 0.5, TEAL, 0.9).move_to([-6.45, -2.55, 0], aligned_edge=DOWN))
            free.add(en("on", 26, TEAL).next_to(free[0], UP, buff=0.1))
            self.play(GrowArrow(on), FadeIn(free), run_time=T(0.7))
            self.play(LaggedStart(*(AnimationGroup(ShowCreation(l[0]), GrowFromEdge(l[1], DOWN), FadeIn(l[2]))
                                    for l in ladder), lag_ratio=0.45), run_time=s.left(1))
            s.until(2)
            self.play(FadeOut(VGroup(ladder, on, free)), run_time=T(0.4))
        # he goes back two lines, heard at his own pace, the highlight following
        back_a, back_b = 74.2, 80.65  # from the quiet after his salawat to the quiet after line 7, real time
        clock.set_value(back_a)
        hl.add_updater(place)
        probs.add_updater(set_probs)
        self.recite(back_a, back_b)
        self.play(clock.animate.set_value(back_b), run_time=back_b - back_a, rate_func=linear)
        hl.clear_updaters()
        probs.clear_updaters()

        with self.say("follow3") as s:
            frames = VGroup(*(Dot(radius=0.045).set_fill(SOFT, 0.85) for _ in range(30))).arrange(LEFT, buff=0.17)
            frames.move_to(DOWN * 2.35 + RIGHT * 1.0)
            self.play(LaggedStart(*(FadeIn(f, shift=LEFT * 0.1) for f in frames), lag_ratio=0.03),
                      Indicate(hl, scale_factor=1.06, color=GOLD), run_time=s.left(0))
            s.until(1)
            talk = box_label("talk", SOFT, 1.7, 0.85, 30).move_to(LEFT * 5.4 + DOWN * 0.6)
            sal = Circle(radius=0.5).set_stroke(TEAL, 2.5).move_to(LEFT * 5.4 + UP * 1.6)
            sal_t = en("salawat", 26, TEAL).next_to(sal, DOWN, buff=0.12)
            self.play(FadeIn(talk), run_time=T(0.5))
            self.play(frames[:12].animate.move_to(talk).scale(0.3).set_opacity(0), run_time=T(0.8))
            self.play(FadeIn(VGroup(sal, sal_t)), run_time=T(0.5))
            dot = Dot(radius=0.09).set_fill(TEAL, 1).move_to(sal.point_from_proportion(0))
            self.play(MoveAlongPath(dot, sal), run_time=T(1.0))
            self.play(FadeOut(dot), run_time=T(0.2))
            line = rows[1]
            self.play(hl.animate.move_to(line[-1]).set_width(line[-1].get_width() + 0.16, stretch=True), run_time=T(0.5))
            nxt = rows[2][0]
            self.play(Indicate(nxt, color=GOLD, scale_factor=1.1), run_time=T(0.7))
            self.play(hl.animate.move_to(nxt).set_width(nxt.get_width() + 0.16, stretch=True), run_time=s.left())

        with self.say("follow4") as s:
            self.play(FadeOut(Group(*self.mobjects)), run_time=T(0.45))
            a = Integer(0, font_size=120, text_config={"font": SERIF}, unit="%").move_to(LEFT * 3.3 + UP * 0.8)
            b = Integer(0, font_size=120, text_config={"font": SERIF}, unit="%").move_to(RIGHT * 3.3 + UP * 0.8)
            la = en("went back or skipped:\nfollowed within 3 s", 30, SOFT).next_to(a, DOWN, buff=0.45)
            lb = en("pauses, talk, salawat:\nheld still", 30, SOFT).next_to(b, DOWN, buff=0.45)
            self.add(a)
            self.play(ChangeDecimalToValue(a, 84), FadeIn(la), run_time=T(1.3))
            s.until(0, max(0.0, s.end_of(0) - s.at(0) - 2.4))
            self.add(b)
            self.play(ChangeDecimalToValue(b, 91), FadeIn(lb), run_time=s.left())

        with self.say("cost1") as s:
            self.play(FadeOut(VGroup(a, b, la, lb)), run_time=T(0.4))
            phone = RoundedRectangle(width=3.0, height=5.4, corner_radius=0.35).set_stroke(INK, 3).move_to(LEFT * 4.6 + UP * 0.3)
            blocks = VGroup(box_label("Whisper", GOLD, 2.3, 0.85, 28), box_label("letter model", TEAL, 2.3, 0.85, 28),
                            box_label("decoder", BLUE, 2.3, 0.85, 28)).arrange(DOWN, buff=0.45).move_to(phone)
            self.play(ShowCreation(phone), LaggedStartMap(FadeIn, blocks, lag_ratio=0.2), run_time=s.left(0))
            s.until(1)
            t0x, w_ = -2.0, 8.6
            g_row, t_row = 1.6, -0.2
            golds = VGroup(*(Rectangle(width=1.73 / 10 * w_, height=0.55).set_fill(GOLD, 0.55).set_stroke(GOLD, 1.5)
                             .move_to([t0x + (2 * k + 0.865) / 10 * w_, g_row, 0]) for k in range(5)))
            ticks = VGroup(*(Line(UP * 0.22, DOWN * 0.22).set_stroke(TEAL, 2.5)
                             .move_to([t0x + (k + 0.5) / 100 * w_, t_row, 0]) for k in range(100)))
            g_l = en("Whisper: 1.7 s a window, every 2 s", 28, GOLD).next_to(golds, UP, buff=0.25).align_to(golds, LEFT)
            t_l = en("letter model: 7 ms a step, every 0.1 s", 28, TEAL).next_to(ticks, DOWN, buff=0.3).align_to(golds, LEFT)
            d_l = en("decoder: 1-11 ms a step", 28, BLUE).next_to(t_l, DOWN, buff=0.25).align_to(golds, LEFT)
            axis = Line([t0x, -2.3, 0], [t0x + w_, -2.3, 0]).set_stroke(SOFT, 1.5)
            secs = en("10 s", 24, SOFT).next_to(axis, DOWN, buff=0.1).align_to(axis, RIGHT)
            self.play(ShowCreation(axis), FadeIn(secs), LaggedStart(*(GrowFromEdge(g, LEFT) for g in golds), lag_ratio=0.2),
                      FadeIn(g_l), run_time=T(1.5))
            self.play(LaggedStartMap(ShowCreation, ticks, lag_ratio=0.01), FadeIn(t_l), run_time=T(1.3))
            self.play(FadeIn(d_l), run_time=s.left())
        self.play(FadeOut(Group(*self.mobjects)), run_time=T(0.5))


class P6(VoiceScene):
    """The whole thing once, the real app, and the credits."""

    def construct(self):
        z, m = data()
        y1, y2 = 1.25, -1.35
        src = wave(12.3, 18.0, 1.6, 0.9).move_to([-6.15, -0.05, 0])
        top_row = [box_label("Whisper", GOLD, 1.75, 0.8, 26), box_label("words", GOLD, 1.5, 0.8, 26),
                   box_label("522 texts", INK, 1.8, 0.8, 26), box_label("tracker", BLUE, 1.6, 0.8, 26),
                   box_label("Du'a Kumayl", BLUE, 2.1, 0.8, 26)]
        bot_row = [box_label("letter model", TEAL, 2.1, 0.8, 26), box_label("letters", TEAL, 1.5, 0.8, 26),
                   box_label("decoder", BLUE, 1.6, 0.8, 26), box_label("the line, the word", GOLD, 2.6, 0.8, 26)]
        VGroup(*top_row).arrange(RIGHT, buff=0.42).move_to([0.75, y1, 0])
        VGroup(*bot_row).arrange(RIGHT, buff=0.42).move_to([0.25, y2, 0])
        a1 = VGroup(*(Arrow(a.get_right(), b.get_left(), buff=0.06).set_stroke(SOFT, 2.5) for a, b in zip(top_row, top_row[1:])))
        a2 = VGroup(*(Arrow(a.get_right(), b.get_left(), buff=0.06).set_stroke(SOFT, 2.5) for a, b in zip(bot_row, bot_row[1:])))
        f1 = Arrow(src.get_right(), top_row[0].get_left(), buff=0.12).set_stroke(SOFT, 2.5)
        f2 = Arrow(src.get_right(), bot_row[0].get_left(), buff=0.12).set_stroke(SOFT, 2.5)
        down = Arrow(top_row[3].get_bottom(), bot_row[2].get_top(), buff=0.1).set_stroke(BLUE, 2.5)
        cad1 = en("every 2 s", 24, GOLD).next_to(f1, UP, buff=0.05).shift(LEFT * 0.2)
        cad2 = en("50 times a second", 24, TEAL).next_to(src, DOWN, buff=0.25)
        cad2.set_x(max(cad2.get_x(), -6.9 + cad2.get_width() / 2))

        with self.say("recap1") as s:
            self.play(FadeIn(src), run_time=T(0.4))
            self.play(GrowArrow(f1), FadeIn(cad1), FadeIn(top_row[0]), run_time=T(0.5))
            s.until(1)
            for a, b in zip(a1[:2], top_row[1:3]):
                self.play(GrowArrow(a), FadeIn(b), run_time=T(0.8))
            for a, b in zip(a1[2:], top_row[3:]):
                self.play(GrowArrow(a), FadeIn(b), run_time=T(0.8))
            s.until(2)
            self.play(GrowArrow(f2), FadeIn(cad2), FadeIn(bot_row[0]), run_time=T(0.5))
            self.play(GrowArrow(a2[0]), FadeIn(bot_row[1]), run_time=T(0.45))
            self.play(GrowArrow(a2[1]), GrowArrow(down), FadeIn(bot_row[2]), run_time=T(0.5))
            self.play(GrowArrow(a2[2]), FadeIn(bot_row[3]), run_time=s.left())
        self.shot(85.9, 91.7 - 85.9)  # the app itself on line 9, his recitation alone
        self.wait(91.7 - 85.9)
        self.remove(*self.mobjects)

        credits = [
            ("Recitations and texts", "DuaPlayer and its reciters (the reading here: Hussein Ghareeb) · duas.org · duas.pro · "
                                      "shiavoice.com · the uploaders on YouTube, Aparat, SoundCloud and the Internet Archive"),
            ("Models", "Whisper (OpenAI) · whisper-base-ar-quran (Tarteel AI) · wav2vec2 XLSR-53 (Meta AI) and its Quran "
                       "fine-tune by rabah2026 · Silero VAD · SpeechBrain ECAPA · OmniVoice (k2-fsa), voices and narration"),
            ("Data", "RetaSy's Quranic audio dataset · Mozilla Common Voice · OpenSLR 28's measured rooms"),
            ("Runtimes", "faster-whisper · CTranslate2 · Transformers.js · ONNX Runtime Web"),
            ("Ideas", "Avery Wang, Shazam's audio search (2003) · Gene Myers, bit-vector string matching (1999) · "
                      "Graves et al., CTC (2006) · Hinton, Vinyals, Dean, distillation (2015) · Newson and Krumm, "
                      "HMM map matching (2009)"),
            ("Logos", "the Shazam (Apple) and Tarteel (Tarteel Inc.) app icons, shown to name the apps"),
            ("Made with", "ManimGL (3Blue1Brown) · Amiri · Latin Modern"),
        ]
        block = VGroup()
        for head, body in credits:
            h = en(head, 30, GOLD)
            lines_, cur = [], ""
            for p in body.split(" · "):
                cand = f"{cur} · {p}" if cur else p
                if len(cand) > 92 and cur:
                    lines_.append(cur)
                    cur = p
                else:
                    cur = cand
            lines_.append(cur)
            b = VGroup(*(Text(l, font=SERIF, font_size=24, fill_color=INK) for l in lines_)).arrange(DOWN, buff=0.08,
                                                                                                aligned_edge=LEFT)
            block.add(VGroup(h, b).arrange(DOWN, buff=0.1, aligned_edge=LEFT))
        block.arrange(DOWN, buff=0.28, aligned_edge=LEFT).move_to(UP * 0.35)
        block.set_max_height(6.4)
        with self.say("credits1") as s:
            self.recite(91.7, 97.75, gain_db=-4.0)  # line 10, whole
            self.play(LaggedStart(*(FadeIn(b_, shift=UP * 0.15) for b_ in block), lag_ratio=0.25), run_time=s.left())
        self.wait(3.5)
        self.play(FadeOut(block), run_time=T(1.4))
