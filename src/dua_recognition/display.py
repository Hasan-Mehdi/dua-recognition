"""What the screen shows between tracker updates: a smoothly moving word highlight.

The tracker updates about once a second, and its word estimate wobbles by a
word or two from update to update. Drawn as-is, the highlight snaps ahead,
sits, snaps again, and sometimes waits for the reciter to catch up. Instead:

- between updates the highlight glides forward at the reciter's own pace
  (the tracker's tempo estimate), so it moves while they do;
- a new estimate is eased into over `ease` seconds rather than jumped to;
- within a line it never steps back: if it has run ahead, it holds until the
  reciter's predicted position catches up;
- it never glides past the end of the line on its own: a new line is shown
  only when the tracker says so (then it jumps, as it should).

web/app.js implements the same rule (tests/test_web_parity.py).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Highlight:
    ease: float = 1.0  # seconds to close the gap to a new estimate
    speed_scale: float = 1.2  # multiplies the tracker's pace estimate

    def __post_init__(self) -> None:
        self.line: tuple[str, int] | None = None
        self.t0 = 0.0
        self.x0 = 0.0  # position (word index, fractional) shown at t0
        self.target = 0.0
        self.speed = 0.0
        self.lo = self.hi = 0  # word index range of the shown line

    def update(self, t: float, dua: str | None, segment: int | None, word: int | None,
               line_words: tuple[int, int], speed: float) -> None:
        """A new tracker position arrived at time t (seconds, display clock)."""
        if dua is None or word is None:
            self.line = None
            return
        line = (dua, segment)
        if line != self.line:
            self.line, self.x0 = line, float(word)
            self.lo, self.hi = line_words
        else:
            self.x0 = self.position(t)
        self.t0, self.target, self.speed = t, float(word), speed * self.speed_scale

    def position(self, t: float) -> float:
        dt = max(0.0, t - self.t0)
        e = min(1.0, dt / self.ease) if self.ease > 0 else 1.0
        x = self.x0 + (self.target - self.x0) * e + self.speed * dt
        return min(max(self.x0, x), self.hi - 1 + 0.999)

    def word(self, t: float) -> int | None:
        return None if self.line is None else int(self.position(t))
