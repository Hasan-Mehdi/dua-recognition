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
  only when the tracker says so (then it jumps, as it should);
- `pace(t, speed)` stops or restarts the glide between updates: the page
  hears the reciter stop before the next update does (asr.LiveQuiet);
- an update with `back` (they have been silent a while) may take it back
  within the line: the tracker's word is then the place they stopped, and
  a highlight that glided past it would otherwise stay there.

web/display.js implements the same rule (tests/test_web_parity.py).
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
        self.floor = 0.0  # it doesn't go below this (x0, or the target when going back)
        self.lo = self.hi = 0  # word index range of the shown line

    def update(self, t: float, dua: str | None, segment: int | None, word: int | None,
               line_words: tuple[int, int], speed: float, back: bool = False) -> None:
        """A new tracker position arrived at time t (seconds, display clock). back: the
        reciter has stopped a while, and `word` may be behind the highlight."""
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
        self.floor = min(self.target, self.x0) if back else self.x0

    def pace(self, t: float, speed: float) -> None:
        """Between updates: glide on at `speed` (words/s, before speed_scale) from now, or stop (0)."""
        if self.line is None:
            return
        x = self.position(t)
        settling = self.floor < self.x0  # on its way back to the target: carry on
        # The ease towards the target carries on from here.
        self.x0, self.t0, self.speed = x, t, speed * self.speed_scale
        self.floor = self.floor if settling else x

    def position(self, t: float) -> float:
        dt = max(0.0, t - self.t0)
        e = min(1.0, dt / self.ease) if self.ease > 0 else 1.0
        # (written so that e = 1 lands on the target exactly: from below, x0 + (target - x0)
        # can come to 2.9999999999999996, a word short once the glide has stopped)
        x = self.target + (self.x0 - self.target) * (1 - e) + self.speed * dt
        return min(max(self.floor, x), self.hi - 1 + 0.999)

    def word(self, t: float) -> int | None:
        return None if self.line is None else int(self.position(t))
