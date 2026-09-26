"""Line labels as subtitles, for review, and how far two sets of labels agree.

Auto-labelled test recordings (scripts/user_label.py, segment_streams.py) are
reviewed in a subtitle editor with a waveform (Subtitle Edit): one cue per
line, "<line id>. <text>", from its start to the next line's start. Reading
the corrected .srt back promotes the recording from silver to gold.
"""
from __future__ import annotations

import bisect
import re

_TIME = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})")
_ARROW = re.compile(rf"{_TIME.pattern}\s*-->\s*{_TIME.pattern}")


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _secs(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def to_srt(starts: list[tuple[float, int]], end_s: float, text: dict[int, str]) -> str:
    """starts: (seconds, line id) ascending; each cue runs to the next start (the last to end_s)."""
    bounds = list(starts) + [(end_s, None)]
    cues = [f"{i}\n{srt_time(t0)} --> {srt_time(t1)}\n{s}. {text.get(s, '')}\n"
            for i, ((t0, s), (t1, _)) in enumerate(zip(bounds, bounds[1:]), 1)]
    return "\n".join(cues)


def from_srt(srt: str) -> tuple[list[tuple[float, int]], float]:
    """(starts, end of the last cue). The line id is the number before the first
    dot of each cue's text; cue numbering and the rest of the text are ignored."""
    starts, end = [], 0.0
    for block in re.split(r"\n\s*\n", srt.replace("\r\n", "\n").strip()):
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            m = _ARROW.search(line)
            if m:
                body = " ".join(lines[i + 1:]).strip()
                sid = re.match(r"\s*(\d+)\s*\.", body)
                if sid:
                    starts.append((round(_secs(*m.groups()[:4]), 3), int(sid.group(1))))
                    end = max(end, _secs(*m.groups()[4:]))
                break
    starts.sort()
    return starts, round(end, 3)


def meta_starts(meta: dict, n_lines: int) -> tuple[list[tuple[float, int]], float]:
    """(starts, end) from a DuaPlayer-shape meta; a key past the last line marks the end."""
    timing = {int(k): v / 1000 for k, v in meta["slide_start_ms"].items()}
    starts = sorted((t, k) for k, t in timing.items() if k <= n_lines)
    after = [t for k, t in timing.items() if k > n_lines]
    return starts, min(after) if after else meta["duration_ms"] / 1000


def set_starts(meta: dict, starts: list[tuple[float, int]], end_s: float, n_lines: int) -> dict:
    """The meta with these line times (and an end marker when the labels stop early)."""
    out = dict(meta)
    ms = {str(s): int(round(t * 1000)) for t, s in starts}
    if end_s * 1000 < meta["duration_ms"] - 1:
        ms[str(n_lines + 1)] = int(round(end_s * 1000))
    out["slide_start_ms"] = ms
    return out


def line_at(starts: list[tuple[float, int]], t: float) -> int | None:
    i = bisect.bisect_right([s for s, _ in starts], t) - 1
    return starts[i][1] if i >= 0 else None


def agreement(a: list[tuple[float, int]], b: list[tuple[float, int]], end_s: float, step: float = 1.0) -> float:
    """Share of the seconds (from the earlier first start to end_s) on which two label sets name the same line."""
    if not a or not b:
        return 0.0
    t0 = min(a[0][0], b[0][0])
    ticks = [t0 + step * (k + 0.5) for k in range(max(1, int((end_s - t0) / step)))]
    return sum(line_at(a, t) == line_at(b, t) for t in ticks) / len(ticks)
