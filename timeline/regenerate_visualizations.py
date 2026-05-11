#!/usr/bin/env python3
"""Refresh inline SVG diagrams for the deep-learning timeline.

The original timeline stores each visualization directly inside the HTML
reports. This script keeps that simple publishing model, while stamping every
diagram with export-size metadata, an accessible label, and a shared canvas
layer that is styled from style.css.
"""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SVG_RE = re.compile(r"<svg\b[^>]*>.*?</svg>", re.DOTALL)
HEADING_RE = re.compile(r"<h[12][^>]*>(.*?)</h[12]>", re.DOTALL)
VIEWBOX_RE = re.compile(r'viewBox="([^"]+)"')


def clean_heading(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return " ".join(text.split())


def title_before(content: str, start: int) -> str:
    section_start = content.rfind("<section", 0, start)
    context = content[section_start if section_start != -1 else 0 : start]
    headings = HEADING_RE.findall(context)
    if headings:
        return clean_heading(headings[-1])
    return "Deep Learning Timeline"


def parse_viewbox(svg: str) -> tuple[float, float, float, float]:
    match = VIEWBOX_RE.search(svg)
    if not match:
        return 0.0, 0.0, 520.0, 240.0
    parts = [float(part) for part in match.group(1).replace(",", " ").split()]
    if len(parts) != 4:
        return 0.0, 0.0, 520.0, 240.0
    return parts[0], parts[1], parts[2], parts[3]


def fmt_num(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def ensure_visual_class(opening: str) -> str:
    class_match = re.search(r'class="([^"]*)"', opening)
    if not class_match:
        return opening[:-1] + ' class="timeline-visual">'

    classes = class_match.group(1).split()
    if "timeline-visual" not in classes:
        classes.append("timeline-visual")
    return (
        opening[: class_match.start()]
        + f'class="{" ".join(classes)}"'
        + opening[class_match.end() :]
    )


def strip_generated_bits(body: str) -> str:
    return re.sub(
        r'^\s*<title>.*?</title>\s*<desc>.*?</desc>\s*<rect class="viz-canvas"[^>]*/>\s*',
        "\n",
        body,
        flags=re.DOTALL,
    )


def upgrade_svg(svg: str, title: str) -> str:
    opening_end = svg.find(">")
    opening = svg[: opening_end + 1]
    body = strip_generated_bits(svg[opening_end + 1 : -6])

    x, y, width, height = parse_viewbox(svg)
    opening = re.sub(
        r'\s(?:width|height|style|role|aria-label|preserveAspectRatio|shape-rendering|text-rendering|data-visual-quality|data-export-width|data-export-height)="[^"]*"',
        "",
        opening,
    )
    opening = ensure_visual_class(opening)

    label = html.escape(f"{title} timeline visualization", quote=True)
    attrs = (
        ' style="width: 100%; height: auto;"'
        ' role="img"'
        f' aria-label="{label}"'
        ' preserveAspectRatio="xMidYMid meet"'
        ' shape-rendering="geometricPrecision"'
        ' text-rendering="optimizeLegibility"'
        ' data-visual-quality="retina"'
        f' data-export-width="{fmt_num(width * 2)}"'
        f' data-export-height="{fmt_num(height * 2)}"'
    )
    opening = opening[:-1] + attrs + ">"

    pad = max(4.0, min(width, height) * 0.018)
    radius = min(22.0, max(12.0, height * 0.075))
    canvas = (
        f'<rect class="viz-canvas" x="{fmt_num(x + pad)}" y="{fmt_num(y + pad)}" '
        f'width="{fmt_num(width - (pad * 2))}" height="{fmt_num(height - (pad * 2))}" '
        f'rx="{fmt_num(radius)}" ry="{fmt_num(radius)}"/>'
    )

    title_node = f"<title>{html.escape(title)}</title>"
    desc_node = (
        "<desc>Higher-resolution SVG diagram regenerated with a shared "
        "timeline visual treatment.</desc>"
    )
    return f"{opening}\n  {title_node}\n  {desc_node}\n  {canvas}\n  {body.lstrip()}</svg>"


def refresh_file(path: Path) -> int:
    original = path.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        title = title_before(original, match.start())
        return upgrade_svg(match.group(0), title)

    updated = SVG_RE.sub(replace, original)
    if updated != original:
        path.write_text(updated, encoding="utf-8")
    return len(SVG_RE.findall(updated))


def main() -> None:
    total = 0
    for path in sorted(ROOT.glob("*.html")):
        total += refresh_file(path)
    print(f"Regenerated {total} inline SVG diagrams.")


if __name__ == "__main__":
    main()
