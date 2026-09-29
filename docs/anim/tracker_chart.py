"""The README's results chart: per-window matching vs the HMM tracker, on the same windows.

Reads the phone model's test results and writes a light and a dark SVG, which the README
switches between with <picture>. Standard library only.

    python docs/anim/tracker_chart.py
"""
from __future__ import annotations

import json
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
SOURCE = DOCS / "results" / "test_whisper-base-aug-v4.json"

METRICS = [  # key, label, note under the label
    ("line_acc", "Correct line", ""),
    ("line_acc_pm1", "Within one line", ""),
    ("refrain_acc", "Refrain lines", "repeated word for word"),
    ("wrong_dua_shown", "Wrong du'a shown", "lower is better"),
]

# Series colours are validated for colour-blind separation on GitHub's light and dark pages;
# the rest follows GitHub's own text and border colours.
THEMES = {
    "light": dict(page="#ffffff", ink="#1f2328", ink2="#59636e", grid="#e6eaef", link="#c8d1da",
                  tracker="#2a78d6", matcher="#eb6834"),
    "dark": dict(page="#0d1117", ink="#f0f6fc", ink2="#9198a1", grid="#21262d", link="#3d444d",
                 tracker="#3987e5", matcher="#d95926"),
}

W, X0, X1 = 720, 230, 690  # width, plot left and right
ROW0, ROW_H = 132, 50
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def svg(summary: dict, c: dict) -> str:
    x = lambda v: X0 + v * (X1 - X0)  # noqa: E731
    bottom = ROW0 + ROW_H * (len(METRICS) - 1) + 28
    h = bottom + 16
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
        f'font-family="{FONT}" role="img" aria-labelledby="t d">',
        '<title id="t">Same windows, with and without the tracker</title>',
        '<desc id="d">' + "; ".join(
            f"{label}: {pct(summary['matcher'][k])} per-window matching, {pct(summary['tracker'][k])} HMM tracker"
            for k, label, _ in METRICS) + "</desc>",
        f'<text x="0" y="22" font-size="16" font-weight="600" fill="{c["ink"]}">'
        "Same audio, same model: what the tracker adds</text>",
        f'<text x="0" y="44" font-size="13" fill="{c["ink2"]}">'
        "Phone model on held-out reciters (16 recordings, 4 h), scored once per second</text>",
    ]
    # legend
    lx = 0
    for key, name in (("tracker", "HMM tracker"), ("matcher", "Per-window text matching, no tracker")):
        out.append(f'<circle cx="{lx + 6}" cy="74" r="6" fill="{c[key]}"/>')
        out.append(f'<text x="{lx + 18}" y="78.5" font-size="13" fill="{c["ink"]}">{name}</text>')
        lx += 150 if key == "tracker" else 0
    # grid and axis
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{x(t):.1f}" y1="{ROW0 - 30}" x2="{x(t):.1f}" y2="{bottom - 8}" '
                   f'stroke="{c["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x(t):.1f}" y="{bottom + 8}" font-size="12" fill="{c["ink2"]}" '
                   f'text-anchor="middle" style="font-variant-numeric:tabular-nums">{int(t * 100)}%</text>')
    # rows
    for i, (k, label, note) in enumerate(METRICS):
        y = ROW0 + i * ROW_H
        ly = y + 5 if not note else y
        out.append(f'<text x="{X0 - 22}" y="{ly}" font-size="14" fill="{c["ink"]}" text-anchor="end">{label}</text>')
        if note:
            out.append(f'<text x="{X0 - 22}" y="{y + 17}" font-size="12" fill="{c["ink2"]}" '
                       f'text-anchor="end">{note}</text>')
        a, b = summary["matcher"][k], summary["tracker"][k]
        out.append(f'<line x1="{x(a):.1f}" y1="{y}" x2="{x(b):.1f}" y2="{y}" stroke="{c["link"]}" '
                   'stroke-width="2" stroke-linecap="round"/>')
        for v, key, weight, ink in ((a, "matcher", 400, c["ink2"]), (b, "tracker", 600, c["ink"])):
            out.append(f'<circle cx="{x(v):.1f}" cy="{y}" r="7" fill="{c[key]}" stroke="{c["page"]}" stroke-width="2"/>')
            out.append(f'<text x="{x(v):.1f}" y="{y - 13}" font-size="13" font-weight="{weight}" fill="{ink}" '
                       f'text-anchor="middle" style="font-variant-numeric:tabular-nums">{pct(v)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    summary = json.loads(SOURCE.read_text(encoding="utf-8"))["summary"]
    for name, colours in THEMES.items():
        path = DOCS / f"tracker-{name}.svg"
        path.write_text(svg(summary, colours), encoding="utf-8")
        print("wrote", path.relative_to(DOCS.parent))


if __name__ == "__main__":
    main()
