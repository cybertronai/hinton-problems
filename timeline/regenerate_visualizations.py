#!/usr/bin/env python3
"""Regenerate the deep-learning timeline's inline SVG illustrations.

The timeline publishes static HTML with each SVG embedded directly in the page.
This script treats those SVGs as generated artifacts and replaces every one
with a fresh, illustrative scene tailored to the milestone.
"""

from __future__ import annotations

import hashlib
import html
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SVG_RE = re.compile(r"<svg\b[^>]*>.*?</svg>", re.DOTALL)
HEADING_RE = re.compile(r"<h[12][^>]*>(.*?)</h[12]>", re.DOTALL)
PERIOD_RE = re.compile(r'<p class="period">([^<]+)</p>', re.DOTALL)
SECTION_RE = re.compile(r'<section class="milestone" id="([^"]+)">')

W = 760
H = 430
INK = "#17202b"
MUTED = "#667085"


@dataclass(frozen=True)
class Scene:
    accent: str
    secondary: str
    renderer: str


SCENES: dict[str, Scene] = {
    "1950s-adjoint": Scene("#2f6fdd", "#d64b8d", "adjoint"),
    "1970-reverse-ad": Scene("#4d7cff", "#d6538c", "reverse_ad"),
    "1974-werbos": Scene("#6d5bd3", "#2aa889", "werbos"),
    "1982-hopfield": Scene("#6d5bd3", "#e35c4f", "hopfield"),
    "1983-boltzmann": Scene("#3858d6", "#d45f8f", "boltzmann"),
    "1986-backprop": Scene("#2f6fdd", "#d64b8d", "backprop"),
    "1989-cnn": Scene("#2878c7", "#2aa889", "cnn"),
    "1991-vanishing": Scene("#d45f8f", "#f59d35", "vanishing"),
    "1991-chunker": Scene("#4775d1", "#24a0a8", "chunker"),
    "1997-lstm": Scene("#2a9d8f", "#6d5bd3", "lstm"),
    "2003-bengio-nlm": Scene("#4978e8", "#f09c3b", "language_model"),
    "2002-cd-rbm": Scene("#3b82d6", "#d64b8d", "rbm"),
    "2006-dbn": Scene("#4d72d8", "#2aa889", "dbn"),
    "2009-bengio-review": Scene("#7459c9", "#f09c3b", "pyramid"),
    "2011-gpu-cnn": Scene("#2f75d6", "#2aa889", "gpu"),
    "2012-alexnet": Scene("#256fd1", "#ef7b45", "alexnet"),
    "2012-dropout": Scene("#5067d8", "#d64b8d", "dropout"),
    "2013-word2vec": Scene("#2d9f88", "#6d5bd3", "word2vec"),
    "2014-seq2seq": Scene("#4d72d8", "#f09c3b", "seq2seq"),
    "2015-attention": Scene("#5b63d6", "#d64b8d", "attention"),
    "2014-gan": Scene("#d45f8f", "#2f75d6", "gan"),
    "2015-batchnorm": Scene("#2aa889", "#f09c3b", "batchnorm"),
    "2015-resnet": Scene("#2f75d6", "#e35c4f", "resnet"),
    "2015-nature-review": Scene("#5f68d8", "#2aa889", "dominant"),
    "2016-alphago": Scene("#253f8f", "#2aa889", "alphago"),
    "2017-transformer": Scene("#6952d5", "#2fb3c6", "transformer"),
    "2018-bert": Scene("#4f70d9", "#f0a33f", "bert"),
    "2018-neural-ode": Scene("#2a9d8f", "#5b63d6", "neural_ode"),
    "2020-scaling-laws": Scene("#356fd3", "#f09c3b", "scaling"),
    "2020-gpt3": Scene("#6754d8", "#2aa889", "gpt3"),
    "2020-diffusion": Scene("#d64b8d", "#2fb3c6", "diffusion"),
    "2021-clip": Scene("#3b82d6", "#2aa889", "clip"),
    "2021-alphafold2": Scene("#2a9d8f", "#6d5bd3", "alphafold"),
    "2022-rlhf": Scene("#5268d8", "#f09c3b", "rlhf"),
    "2022-chatgpt": Scene("#2aa889", "#5b63d6", "chatgpt"),
    "2023-gpt4": Scene("#5d67d8", "#2fb3c6", "gpt4"),
    "2024-gpt4o": Scene("#2f75d6", "#d64b8d", "gpt4o"),
    "2024-o1-reasoning": Scene("#6952d5", "#f09c3b", "reasoning"),
    "2024-nobel": Scene("#c7922b", "#5b63d6", "nobel"),
    "2025-deepseek": Scene("#2f75d6", "#e35c4f", "deepseek"),
    "2025-agentic": Scene("#5d67d8", "#2aa889", "agentic"),
}


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def clean_heading(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return " ".join(text.split())


def last_match(pattern: re.Pattern[str], content: str, end: int) -> str | None:
    matches = list(pattern.finditer(content, 0, end))
    if not matches:
        return None
    return clean_heading(matches[-1].group(1))


def slug_before(content: str, end: int, fallback: str) -> str:
    matches = list(SECTION_RE.finditer(content, 0, end))
    if matches:
        return matches[-1].group(1)
    return fallback


def slug_prefix(slug: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", slug)


def deterministic_palette(slug: str) -> tuple[str, str]:
    palettes = [
        ("#2f6fdd", "#d64b8d"),
        ("#5b63d6", "#2aa889"),
        ("#7459c9", "#f09c3b"),
        ("#2a9d8f", "#6d5bd3"),
        ("#d45f8f", "#2fb3c6"),
    ]
    index = hashlib.sha1(slug.encode("utf-8")).digest()[0] % len(palettes)
    return palettes[index]


def wrap(value: str, width: int = 44) -> list[str]:
    return textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=False) or [value]


def tx(
    x: float,
    y: float,
    value: str,
    size: int = 16,
    fill: str = INK,
    weight: int | str = 500,
    anchor: str = "start",
    opacity: float | None = None,
    family: str | None = None,
) -> str:
    attrs = [
        f'x="{x:g}"',
        f'y="{y:g}"',
        f'font-size="{size}"',
        f'fill="{fill}"',
        f'font-weight="{weight}"',
        f'text-anchor="{anchor}"',
    ]
    if opacity is not None:
        attrs.append(f'opacity="{opacity:g}"')
    if family:
        attrs.append(f'font-family="{family}"')
    return f"<text {' '.join(attrs)}>{esc(value)}</text>"


def multiline(x: float, y: float, lines: list[str], size: int = 16, fill: str = INK, weight: int | str = 500, anchor: str = "start", gap: int = 21) -> str:
    return "\n".join(tx(x, y + i * gap, line, size, fill, weight, anchor) for i, line in enumerate(lines))


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: float = 14, opacity: float | None = None, sw: float = 1.2, extra: str = "") -> str:
    attrs = f'x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx:g}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}"'
    if opacity is not None:
        attrs += f' opacity="{opacity:g}"'
    if extra:
        attrs += f" {extra}"
    return f"<rect {attrs}/>"


def circle(x: float, y: float, r: float, fill: str, stroke: str = "none", sw: float = 1.2, opacity: float | None = None) -> str:
    attrs = f'cx="{x:g}" cy="{y:g}" r="{r:g}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}"'
    if opacity is not None:
        attrs += f' opacity="{opacity:g}"'
    return f"<circle {attrs}/>"


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, sw: float = 2, opacity: float | None = None, marker: str | None = None, dash: str | None = None) -> str:
    attrs = f'x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{stroke}" stroke-width="{sw:g}" fill="none"'
    if opacity is not None:
        attrs += f' opacity="{opacity:g}"'
    if marker:
        attrs += f' marker-end="{marker}"'
    if dash:
        attrs += f' stroke-dasharray="{dash}"'
    return f"<line {attrs}/>"


def path(d: str, stroke: str = "none", fill: str = "none", sw: float = 2, opacity: float | None = None, marker: str | None = None, dash: str | None = None, extra: str = "") -> str:
    attrs = f'd="{d}" stroke="{stroke}" fill="{fill}" stroke-width="{sw:g}"'
    if opacity is not None:
        attrs += f' opacity="{opacity:g}"'
    if marker:
        attrs += f' marker-end="{marker}"'
    if dash:
        attrs += f' stroke-dasharray="{dash}"'
    if extra:
        attrs += f" {extra}"
    return f"<path {attrs}/>"


def pill(x: float, y: float, label: str, fill: str, color: str = "#ffffff", w: float | None = None) -> str:
    width = w if w is not None else max(72, len(label) * 7.8 + 28)
    return "\n".join([
        rect(x, y, width, 29, fill, "none", 14),
        tx(x + width / 2, y + 20, label, 12, color, 700, "middle"),
    ])


def card(x: float, y: float, w: float, h: float, label: str, fill: str, stroke: str, sub: str | None = None) -> str:
    parts = [rect(x, y, w, h, fill, stroke, 18, extra='filter="url(#shadow)"')]
    parts.append(tx(x + w / 2, y + 30, label, 15, INK, 700, "middle"))
    if sub:
        parts.append(tx(x + w / 2, y + 53, sub, 12, MUTED, 500, "middle"))
    return "\n".join(parts)


def dot_grid() -> str:
    dots = []
    for x in range(70, 721, 70):
        for y in range(130, 382, 56):
            if (x + y) % 3 == 0:
                dots.append(circle(x, y, 1.8, "#8aa3c5", opacity=0.18))
    return "\n".join(dots)


def scene_defs(prefix: str, accent: str, secondary: str) -> str:
    return f"""<defs>
  <linearGradient id="{prefix}-bg" x1="0" x2="1" y1="0" y2="1">
    <stop offset="0%" stop-color="#fafdff"/>
    <stop offset="52%" stop-color="#eef5ff"/>
    <stop offset="100%" stop-color="#fdf3f8"/>
  </linearGradient>
  <linearGradient id="{prefix}-panel" x1="0" x2="1" y1="0" y2="1">
    <stop offset="0%" stop-color="#ffffff" stop-opacity="0.97"/>
    <stop offset="100%" stop-color="#f7fbff" stop-opacity="0.94"/>
  </linearGradient>
  <linearGradient id="{prefix}-accent-soft" x1="0" x2="1">
    <stop offset="0%" stop-color="{accent}" stop-opacity="0.18"/>
    <stop offset="100%" stop-color="{secondary}" stop-opacity="0.18"/>
  </linearGradient>
  <linearGradient id="{prefix}-accent" x1="0" x2="1">
    <stop offset="0%" stop-color="{accent}"/>
    <stop offset="100%" stop-color="{secondary}"/>
  </linearGradient>
  <filter id="{prefix}-shadow" x="-20%" y="-20%" width="140%" height="150%">
    <feDropShadow dx="0" dy="16" stdDeviation="18" flood-color="#22304a" flood-opacity="0.16"/>
  </filter>
  <filter id="{prefix}-small-shadow" x="-30%" y="-30%" width="160%" height="180%">
    <feDropShadow dx="0" dy="6" stdDeviation="7" flood-color="#22304a" flood-opacity="0.18"/>
  </filter>
  <marker id="{prefix}-arrow-a" markerWidth="12" markerHeight="12" refX="10" refY="5.5" orient="auto">
    <path d="M0,0 L11,5.5 L0,11 Z" fill="{accent}"/>
  </marker>
  <marker id="{prefix}-arrow-b" markerWidth="12" markerHeight="12" refX="10" refY="5.5" orient="auto">
    <path d="M0,0 L11,5.5 L0,11 Z" fill="{secondary}"/>
  </marker>
  <marker id="{prefix}-arrow-ink" markerWidth="12" markerHeight="12" refX="10" refY="5.5" orient="auto">
    <path d="M0,0 L11,5.5 L0,11 Z" fill="{INK}"/>
  </marker>
</defs>"""


def base_open(slug: str, title: str, period: str, accent: str, secondary: str) -> tuple[str, str]:
    prefix = slug_prefix(slug)
    title_lines = wrap(title, 42)[:2]
    header = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" class="timeline-visual illustrative-visual" style="width: 100%; height: auto;" role="img" aria-label="{esc(title)} illustrative timeline visualization" preserveAspectRatio="xMidYMid meet" shape-rendering="geometricPrecision" text-rendering="optimizeLegibility" data-visual-quality="illustrative-retina" data-export-width="{W * 2}" data-export-height="{H * 2}">',
        f"  <title>{esc(title)}</title>",
        "  <desc>Completely redrawn illustrative SVG scene for the deep-learning timeline.</desc>",
        scene_defs(prefix, accent, secondary),
        f'  <rect x="0" y="0" width="{W}" height="{H}" fill="url(#{prefix}-bg)"/>',
        dot_grid(),
        f'  <circle cx="672" cy="72" r="86" fill="{accent}" opacity="0.08"/>',
        f'  <circle cx="93" cy="372" r="96" fill="{secondary}" opacity="0.08"/>',
        '  <rect x="24" y="24" width="712" height="382" rx="34" fill="url(#' + prefix + '-panel)" stroke="#dce8f5" filter="url(#' + prefix + '-shadow)"/>',
        pill(50, 48, period, accent, "#ffffff"),
        multiline(50, 104 - (len(title_lines) - 1) * 13, title_lines, 23, INK, 800, "start", 27),
    ]
    return prefix, "\n".join(header)


def base_close() -> str:
    return "</svg>"


def small_label(x: float, y: float, value: str, color: str = MUTED) -> str:
    return tx(x, y, value.upper(), 11, color, 800, "start")


def renderer_adjoint(prefix: str, a: str, b: str) -> str:
    nodes = [(128, 224, "x0"), (260, 190, "x1"), (402, 214, "x2"), (546, 174, "xT")]
    parts = [
        path("M92,254 C170,172 250,184 332,208 C415,232 492,130 604,164", a, "none", 6, 0.7, f"url(#{prefix}-arrow-a)"),
        path("M604,286 C514,340 420,308 338,294 C250,280 174,338 94,304", b, "none", 5, 0.75, f"url(#{prefix}-arrow-b)"),
        small_label(86, 162, "forward state trajectory", a),
        small_label(424, 340, "backward adjoint sensitivities", b),
    ]
    for x, y, label in nodes:
        parts += [circle(x, y, 29, "#ffffff", a, 2.4, 1), tx(x, y + 6, label, 16, INK, 800, "middle")]
        parts.append(line(x, y + 39, x, 288, b, 1.6, 0.35, dash="5 7"))
        parts.append(circle(x, 288, 18, "#fff3f8", b, 1.8))
    return "\n".join(parts)


def renderer_reverse_ad(prefix: str, a: str, b: str) -> str:
    nodes = [(118, 230, "a"), (118, 304, "b"), (278, 206, "u=a*b"), (278, 304, "v=a+b"), (452, 254, "y"), (604, 254, "J")]
    edges = [(147, 230, 248, 206), (147, 304, 248, 206), (147, 230, 248, 304), (147, 304, 248, 304), (330, 206, 422, 254), (330, 304, 422, 254), (481, 254, 574, 254)]
    parts = [rect(84, 166, 560, 192, "#ffffff", "#d8e5f3", 26, extra='filter="url(#small-shadow)"'), small_label(104, 190, "value tape and reverse accumulation", a)]
    for e in edges:
        parts.append(line(*e, a, 2.2, 0.55, f"url(#{prefix}-arrow-a)"))
    for x, y, label in nodes:
        fill = "#eef5ff" if label in ("a", "b") else "#ffffff"
        stroke = b if label == "J" else a
        parts += [rect(x - 38, y - 23, 76, 46, fill, stroke, 13), tx(x, y + 5, label, 14, INK, 800, "middle")]
    parts += [
        path("M604,148 C512,118 420,126 338,160 C252,196 176,178 100,146", b, "none", 4, 0.85, f"url(#{prefix}-arrow-b)", dash="9 8"),
        tx(352, 142, "reverse pass: chain rule flows against the arrows", 15, b, 800, "middle"),
    ]
    return "\n".join(parts)


def renderer_werbos(prefix: str, a: str, b: str) -> str:
    parts = [
        rect(74, 180, 190, 132, "#eef5ff", a, 22, extra='filter="url(#small-shadow)"'),
        rect(496, 180, 190, 132, "#fff3f8", b, 22, extra='filter="url(#small-shadow)"'),
        tx(169, 218, "Control theory", 20, INK, 800, "middle"),
        tx(591, 218, "Neural nets", 20, INK, 800, "middle"),
        tx(169, 250, "adjoints, DP,", 14, MUTED, 600, "middle"),
        tx(169, 270, "optimal feedback", 14, MUTED, 600, "middle"),
        tx(591, 250, "weights, layers,", 14, MUTED, 600, "middle"),
        tx(591, 270, "credit assignment", 14, MUTED, 600, "middle"),
        path("M262,252 C338,170 424,170 498,252", a, "none", 10, 0.22),
        path("M262,252 C338,170 424,170 498,252", f"url(#{prefix}-accent)", "none", 4.5, 1, f"url(#{prefix}-arrow-a)"),
        circle(380, 188, 44, "#ffffff", "#d8e5f3", 1.4, 1),
        tx(380, 183, "Werbos", 17, INK, 800, "middle"),
        tx(380, 205, "bridge", 13, MUTED, 700, "middle"),
    ]
    return "\n".join(parts)


def renderer_hopfield(prefix: str, a: str, b: str) -> str:
    parts = [
        path("M76,330 C118,190 172,314 228,230 C276,158 336,338 394,224 C454,112 504,330 598,194 C630,154 662,180 688,154 L688,356 L76,356 Z", a, "#eef5ff", 3, 0.92),
        path("M104,210 C158,176 192,222 220,246", b, "none", 4, 0.8, f"url(#{prefix}-arrow-b)"),
        small_label(88, 164, "energy landscape with attractor basins", a),
    ]
    for x, y, label in [(226, 246, "memory A"), (398, 230, "memory B"), (604, 196, "memory C")]:
        parts += [circle(x, y, 14, b, "#ffffff", 3), tx(x, y + 40, label, 13, INK, 800, "middle")]
    for x, y in [(148, 292), (304, 326), (522, 312)]:
        parts.append(path(f"M{x-26},{y} Q{x},{y-20} {x+28},{y}", "#ffffff", "none", 2, 0.7))
    return "\n".join(parts)


def renderer_boltzmann(prefix: str, a: str, b: str) -> str:
    visible = [(150, 302), (238, 302), (326, 302), (414, 302)]
    hidden = [(194, 206), (282, 206), (370, 206)]
    parts = [rect(92, 160, 390, 196, "#ffffff", "#d8e5f3", 28, extra='filter="url(#small-shadow)"'), small_label(110, 184, "stochastic energy model", a)]
    for vx, vy in visible:
        for hx, hy in hidden:
            parts.append(line(vx, vy, hx, hy, "#8ea9cd", 1.2, 0.4))
    for i, (x, y) in enumerate(visible, 1):
        parts += [circle(x, y, 22, "#ffffff", a, 2.2), tx(x, y + 5, f"v{i}", 13, INK, 800, "middle")]
    for i, (x, y) in enumerate(hidden, 1):
        parts += [circle(x, y, 22, "#fff0f6", b, 2.2), tx(x, y + 5, f"h{i}", 13, INK, 800, "middle")]
    parts += [
        card(516, 178, 126, 66, "+ phase", "#effaf6", "#bde7d8", "data clamped"),
        card(516, 278, 126, 66, "- phase", "#fff3f8", "#f0bfd3", "model dreams"),
        path("M580,244 C548,238 548,218 580,214", a, "none", 3, 0.85, f"url(#{prefix}-arrow-a)"),
    ]
    return "\n".join(parts)


def layered_net(prefix: str, a: str, b: str, x0: int = 130, y0: int = 170) -> list[str]:
    layers = [[(x0, y0 + i * 46) for i in range(4)], [(x0 + 150, y0 + 23 + i * 46) for i in range(3)], [(x0 + 300, y0 + 23 + i * 46) for i in range(3)], [(x0 + 450, y0 + 46 + i * 46) for i in range(2)]]
    parts: list[str] = []
    for left, right in zip(layers, layers[1:]):
        for x1, y1 in left:
            for x2, y2 in right:
                if abs(y1 - y2) < 90:
                    parts.append(line(x1, y1, x2, y2, "#8db2df", 1.15, 0.55))
    for li, layer in enumerate(layers):
        for x, y in layer:
            fill = "#ffffff" if li < 3 else "#fff3f8"
            stroke = a if li < 3 else b
            parts.append(circle(x, y, 18, fill, stroke, 2))
    return parts


def renderer_backprop(prefix: str, a: str, b: str) -> str:
    parts = layered_net(prefix, a, b)
    parts += [
        small_label(86, 155, "activation stream", a),
        line(104, 352, 630, 352, a, 5, 0.82, f"url(#{prefix}-arrow-a)"),
        tx(366, 374, "forward activations", 14, a, 800, "middle"),
        path("M626,142 C510,92 384,102 274,128 C186,148 132,126 94,108", b, "none", 5, 0.86, f"url(#{prefix}-arrow-b)"),
        tx(376, 112, "backward error signal", 14, b, 800, "middle"),
    ]
    return "\n".join(parts)


def renderer_cnn(prefix: str, a: str, b: str) -> str:
    parts = [small_label(76, 152, "local filters build visual hierarchies", a)]
    for i in range(5):
        parts.append(rect(78 + i * 3, 182 - i * 3, 92, 92, "#eef5ff", a, 12, opacity=0.95))
    for i in range(4):
        parts.append(rect(252 + i * 10, 174 - i * 8, 92, 92, "#ffffff", "#8db2df", 10))
    for i in range(3):
        parts.append(rect(424 + i * 10, 194 - i * 8, 64, 64, "#ffffff", "#8fcfbf", 8))
    parts += [
        line(178, 226, 246, 220, a, 3, 0.8, f"url(#{prefix}-arrow-a)"),
        line(356, 222, 416, 222, a, 3, 0.8, f"url(#{prefix}-arrow-a)"),
        line(508, 224, 588, 224, a, 3, 0.8, f"url(#{prefix}-arrow-a)"),
        rect(600, 178, 76, 96, "#fff3f8", b, 16),
        tx(638, 217, "7", 42, b, 900, "middle"),
        tx(124, 298, "image", 13, MUTED, 700, "middle"),
        tx(306, 298, "feature maps", 13, MUTED, 700, "middle"),
        tx(468, 298, "pool", 13, MUTED, 700, "middle"),
    ]
    return "\n".join(parts)


def renderer_vanishing(prefix: str, a: str, b: str) -> str:
    parts = [
        small_label(88, 153, "long chains multiply gradients", b),
        line(96, 342, 650, 342, "#a8b7cc", 2),
        line(96, 342, 96, 162, "#a8b7cc", 2),
        path("M112,178 C202,218 278,252 346,278 C438,314 526,332 650,338", a, "none", 5, 0.9),
        path("M112,338 C214,314 310,270 404,208 C490,150 572,128 650,112", b, "none", 3.5, 0.55, dash="10 7"),
    ]
    for i, r in enumerate([22, 17, 12, 8, 5]):
        x = 160 + i * 95
        y = 202 + i * 28
        parts += [circle(x, y, r, "#ffffff", a, 2), tx(x, y + 5, "∂", max(10, int(r)), a, 800, "middle")]
    parts += [tx(626, 322, "vanish", 14, a, 800, "end"), tx(628, 126, "explode", 14, b, 800, "end")]
    return "\n".join(parts)


def renderer_chunker(prefix: str, a: str, b: str) -> str:
    parts = [small_label(92, 154, "surprises rise to a slower timescale", a)]
    xs = [106, 156, 206, 256, 306, 356, 406, 456]
    for i, x in enumerate(xs):
        fill = "#fff3f8" if i in (2, 5) else "#eef5ff"
        parts.append(rect(x, 284, 34, 48, fill, b if i in (2, 5) else a, 8))
    parts += [
        rect(90, 214, 394, 44, "#ffffff", "#d8e5f3", 14),
        tx(286, 242, "lower predictor consumes ordinary events", 14, INK, 700, "middle"),
        rect(208, 164, 266, 44, "#ffffff", b, 14),
        tx(341, 192, "upper predictor sees only residual surprise", 14, INK, 700, "middle"),
    ]
    for x in [223, 373]:
        parts += [line(x, 284, x, 210, b, 2.5, 0.85, f"url(#{prefix}-arrow-b)"), path(f"M{x-12},268 L{x},246 L{x+12},268", b, "none", 2.2)]
    return "\n".join(parts)


def renderer_lstm(prefix: str, a: str, b: str) -> str:
    parts = [
        small_label(86, 154, "constant error carousel with gates", a),
        path("M92,234 C196,184 300,184 404,234 C508,284 612,284 688,234", a, "none", 8, 0.28),
        path("M92,234 C196,184 300,184 404,234 C508,284 612,284 688,234", a, "none", 4.5, 1, f"url(#{prefix}-arrow-a)"),
    ]
    for x, label in [(174, "input"), (316, "forget"), (458, "output"), (600, "cell")]:
        parts += [rect(x - 44, 284, 88, 58, "#ffffff", "#cfe0f3", 16, extra='filter="url(#small-shadow)"'), circle(x, 284, 20, "#fff3f8", b, 2), tx(x, 322, label, 13, INK, 800, "middle")]
        parts.append(line(x, 284, x, 238, b, 2.4, 0.7, f"url(#{prefix}-arrow-b)"))
    parts += [tx(390, 224, "memory highway", 18, a, 800, "middle")]
    return "\n".join(parts)


def renderer_language_model(prefix: str, a: str, b: str) -> str:
    words = ["the", "cat", "sat"]
    parts = [small_label(88, 154, "words become distributed coordinates", a)]
    for i, word in enumerate(words):
        parts += [rect(92, 184 + i * 58, 84, 36, "#ffffff", a, 10), tx(134, 208 + i * 58, word, 15, INK, 800, "middle"), line(176, 202 + i * 58, 238, 248, a, 2, 0.6, f"url(#{prefix}-arrow-a)")]
    parts += [rect(246, 170, 146, 170, "#eef5ff", a, 20, extra='filter="url(#small-shadow)"'), tx(319, 206, "embedding", 18, INK, 800, "middle"), tx(319, 230, "matrix C", 15, MUTED, 700, "middle")]
    for i in range(7):
        parts.append(rect(275 + i * 12, 258, 7, 46 - i * 4, b if i % 2 else a, "none", 3, opacity=0.75))
    parts += [line(392, 254, 472, 254, a, 3, 0.75, f"url(#{prefix}-arrow-a)"), card(488, 190, 150, 122, "softmax", "#fff3f8", b, "next word distribution")]
    return "\n".join(parts)


def renderer_rbm(prefix: str, a: str, b: str) -> str:
    parts = [small_label(92, 154, "one-step reconstruction carousel", a)]
    centers = [(142, 254, "data v"), (302, 206, "hidden h"), (462, 254, "recon v'"), (302, 316, "hidden h'")]
    for i, (x, y, label) in enumerate(centers):
        parts.append(card(x - 58, y - 36, 116, 72, label, "#ffffff" if i % 2 == 0 else "#eef5ff", a if i % 2 == 0 else b))
    parts += [
        path("M200,246 C236,206 252,202 244,208", a, "none", 3, 0.85, f"url(#{prefix}-arrow-a)"),
        path("M354,208 C390,210 424,226 462,252", a, "none", 3, 0.85, f"url(#{prefix}-arrow-a)"),
        path("M462,288 C420,322 376,334 302,316", b, "none", 3, 0.85, f"url(#{prefix}-arrow-b)"),
        path("M244,316 C190,314 154,296 142,254", b, "none", 3, 0.85, f"url(#{prefix}-arrow-b)"),
        tx(302, 376, "positive phase - negative phase", 15, INK, 800, "middle"),
    ]
    return "\n".join(parts)


def renderer_dbn(prefix: str, a: str, b: str) -> str:
    parts = [small_label(94, 154, "greedy layer-wise pretraining ladder", a)]
    widths = [360, 300, 240, 180]
    for i, w in enumerate(widths):
        x = 200 + (360 - w) / 2
        y = 318 - i * 62
        fill = "#eef5ff" if i % 2 == 0 else "#fff3f8"
        parts += [rect(x, y, w, 44, fill, a if i % 2 == 0 else b, 16, extra='filter="url(#small-shadow)"'), tx(380, y + 28, f"RBM layer {i+1}", 15, INK, 800, "middle")]
        if i < len(widths) - 1:
            parts.append(line(380, y, 380, y - 18, b, 2.5, 0.75, f"url(#{prefix}-arrow-b)"))
    parts += [rect(78, 308, 92, 54, "#ffffff", "#d8e5f3", 16), tx(124, 341, "pixels", 14, MUTED, 800, "middle"), line(170, 336, 198, 340, a, 2.5, 0.8, f"url(#{prefix}-arrow-a)"), tx(592, 198, "fine-tune", 16, b, 800, "middle"), path("M540,186 C584,162 626,172 660,210", b, "none", 3.5, 0.7, f"url(#{prefix}-arrow-b)")]
    return "\n".join(parts)


def renderer_pyramid(prefix: str, a: str, b: str) -> str:
    parts = [small_label(94, 154, "features compose into abstractions", a)]
    levels = [("edges", 104, 328, 552, 48, "#eef5ff", a), ("parts", 154, 268, 452, 48, "#ffffff", "#8db2df"), ("objects", 214, 208, 332, 48, "#fff3f8", b), ("concepts", 282, 148, 196, 48, "#fff8e8", "#f09c3b")]
    for label, x, y, w, h, fill, stroke in levels:
        parts += [rect(x, y, w, h, fill, stroke, 18, extra='filter="url(#small-shadow)"'), tx(x + w / 2, y + 30, label, 17, INK, 800, "middle")]
    for x in [214, 282, 350, 418, 486]:
        parts.append(line(x, 328, 360, 196, a, 1.6, 0.22))
    return "\n".join(parts)


def renderer_gpu(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "parallel compute turns deep nets practical", a)]
    parts += [rect(90, 184, 190, 142, "#182338", "none", 22, extra='filter="url(#small-shadow)"'), rect(122, 216, 126, 78, "#263b62", a, 12)]
    for i in range(8):
        parts.append(line(104 + i * 22, 170, 104 + i * 22, 184, a, 2, 0.75))
        parts.append(line(104 + i * 22, 326, 104 + i * 22, 342, a, 2, 0.75))
    parts += [tx(185, 263, "GPU", 34, "#ffffff", 900, "middle"), path("M284,250 C360,210 424,212 496,248", a, "none", 4, 0.85, f"url(#{prefix}-arrow-a)")]
    for i, x in enumerate([508, 554, 600]):
        parts.append(rect(x, 204 + i * 18, 72, 72, "#ffffff", "#8db2df", 10, opacity=0.92))
    parts += [tx(590, 330, "CNN contests", 15, INK, 800, "middle"), path("M628,180 L652,226 L704,232 L666,264 L676,316 L630,290 L584,316 L594,264 L556,232 L608,226 Z", "#c7922b", "#fff4c2", 2.2)]
    return "\n".join(parts)


def renderer_alexnet(prefix: str, a: str, b: str) -> str:
    parts = [small_label(88, 154, "imagenet scoreboard flips", a)]
    for i, x in enumerate([92, 160, 228, 296, 364]):
        parts.append(rect(x, 240 - i * 12, 60, 60, "#ffffff", a, 9, opacity=0.94))
    parts += [line(430, 244, 500, 244, a, 3, 0.9, f"url(#{prefix}-arrow-a)"), card(512, 174, 150, 138, "ImageNet 2012", "#fff8e8", "#f09c3b", "top-5 error drops")]
    for i, (label, width, color) in enumerate([("AlexNet", 104, b), ("runner-up", 62, "#9aa8b7"), ("old guard", 44, "#c7d2df")]):
        y = 220 + i * 30
        parts += [tx(534, y + 13, label, 12, MUTED, 700), rect(596, y, width, 16, color, "none", 8)]
    parts += [tx(244, 330, "two GPUs + ReLUs + dropout", 15, INK, 800, "middle")]
    return "\n".join(parts)


def renderer_dropout(prefix: str, a: str, b: str) -> str:
    parts = [small_label(86, 154, "randomly train thinned subnetworks", a)]
    coords = [(150, 210), (150, 266), (150, 322), (288, 198), (288, 254), (288, 310), (426, 226), (426, 282), (568, 254)]
    dropped = {(288, 254), (426, 226)}
    for x1, y1 in coords:
        for x2, y2 in coords:
            if x2 > x1 and x2 - x1 < 160 and (x1, y1) not in dropped and (x2, y2) not in dropped:
                parts.append(line(x1, y1, x2, y2, "#8db2df", 1, 0.4))
    for x, y in coords:
        if (x, y) in dropped:
            parts += [circle(x, y, 19, "#f3f6fb", "#cbd5e1", 1.5, 0.7), line(x - 12, y - 12, x + 12, y + 12, b, 2), line(x + 12, y - 12, x - 12, y + 12, b, 2)]
        else:
            parts.append(circle(x, y, 19, "#ffffff", a, 2))
    parts += [tx(360, 366, "many masks approximate an ensemble", 15, INK, 800, "middle")]
    return "\n".join(parts)


def renderer_word2vec(prefix: str, a: str, b: str) -> str:
    parts = [small_label(86, 154, "semantic geometry appears in vector space", a)]
    points = [(194, 284, "king", a), (346, 236, "queen", b), (174, 340, "man", "#94a3b8"), (326, 334, "woman", "#94a3b8"), (516, 232, "royal", "#f09c3b")]
    parts += [line(120, 350, 620, 350, "#b8c8db", 1.8), line(120, 350, 120, 180, "#b8c8db", 1.8)]
    for x, y, label, color in points:
        parts += [circle(x, y, 18, "#ffffff", color, 2.2, 1), tx(x, y - 26, label, 13, INK, 800, "middle")]
    parts += [
        line(194, 284, 174, 340, a, 2.2, 0.65, f"url(#{prefix}-arrow-a)"),
        line(174, 340, 326, 334, b, 2.2, 0.65, f"url(#{prefix}-arrow-b)"),
        line(326, 334, 346, 236, b, 2.2, 0.65, f"url(#{prefix}-arrow-b)"),
        tx(378, 184, "king - man + woman ≈ queen", 18, b, 800, "middle"),
    ]
    return "\n".join(parts)


def renderer_seq2seq(prefix: str, a: str, b: str) -> str:
    source = ["Je", "suis", "ici"]
    target = ["I", "am", "here"]
    parts = [small_label(84, 154, "encoder compresses, decoder unfolds", a)]
    for i, word in enumerate(source):
        parts += [rect(92 + i * 74, 248, 60, 38, "#eef5ff", a, 10), tx(122 + i * 74, 273, word, 14, INK, 800, "middle")]
        if i < 2:
            parts.append(line(152 + i * 74, 267, 166 + i * 74, 267, a, 2, 0.75, f"url(#{prefix}-arrow-a)"))
    parts += [circle(354, 267, 45, "#ffffff", b, 2.4, 1), tx(354, 263, "thought", 15, INK, 800, "middle"), tx(354, 283, "vector", 13, MUTED, 700, "middle")]
    for i, word in enumerate(target):
        x = 480 + i * 68
        parts += [rect(x, 248, 56, 38, "#fff3f8", b, 10), tx(x + 28, 273, word, 14, INK, 800, "middle")]
        if i < 2:
            parts.append(line(x + 56, 267, x + 68, 267, b, 2, 0.75, f"url(#{prefix}-arrow-b)"))
    parts += [line(240, 267, 306, 267, a, 3, 0.8, f"url(#{prefix}-arrow-a)"), line(400, 267, 480, 267, b, 3, 0.8, f"url(#{prefix}-arrow-b)")]
    return "\n".join(parts)


def renderer_attention(prefix: str, a: str, b: str) -> str:
    src = ["the", "cat", "sat", "down"]
    parts = [small_label(84, 154, "soft alignment replaces one fixed vector", a)]
    for i, word in enumerate(src):
        x = 108 + i * 96
        parts += [rect(x, 298, 70, 38, "#eef5ff", a, 10), tx(x + 35, 323, word, 13, INK, 800, "middle")]
    parts += [rect(520, 180, 112, 54, "#fff3f8", b, 16, extra='filter="url(#small-shadow)"'), tx(576, 213, "decoder step", 15, INK, 800, "middle")]
    weights = [0.25, 0.7, 0.42, 0.16]
    for i, wt in enumerate(weights):
        x = 143 + i * 96
        parts.append(path(f"M576,234 C520,{260 - i*18} {x+22},{254 - i*10} {x},298", b, "none", 1.5 + wt * 5, 0.26 + wt * 0.52))
    parts += [tx(334, 202, "attention weights", 18, b, 800, "middle")]
    return "\n".join(parts)


def renderer_gan(prefix: str, a: str, b: str) -> str:
    parts = [small_label(86, 154, "two networks in an adversarial studio", b)]
    parts += [
        card(84, 210, 150, 88, "generator", "#fff3f8", b, "noise → sample"),
        card(304, 210, 150, 88, "discriminator", "#eef5ff", a, "real or fake?"),
        card(524, 210, 118, 88, "feedback", "#ffffff", "#d8e5f3", "minimax"),
        line(234, 254, 304, 254, b, 3.2, 0.85, f"url(#{prefix}-arrow-b)"),
        line(454, 254, 524, 254, a, 3.2, 0.85, f"url(#{prefix}-arrow-a)"),
        path("M584,298 C480,366 216,366 154,300", a, "none", 3.2, 0.65, f"url(#{prefix}-arrow-a)", dash="9 7"),
    ]
    return "\n".join(parts)


def renderer_batchnorm(prefix: str, a: str, b: str) -> str:
    parts = [small_label(86, 154, "normalize, then learn scale and shift", a)]
    for i, h in enumerate([32, 74, 48, 104, 58, 34]):
        parts.append(rect(108 + i * 20, 318 - h, 12, h, b, "none", 5, opacity=0.65))
    parts += [tx(164, 340, "wild activations", 13, MUTED, 700, "middle"), line(252, 260, 334, 260, a, 3, 0.8, f"url(#{prefix}-arrow-a)"), circle(386, 260, 52, "#ffffff", a, 2.4), tx(386, 253, "μ=0", 17, INK, 800, "middle"), tx(386, 276, "σ=1", 17, INK, 800, "middle"), line(438, 260, 520, 260, a, 3, 0.8, f"url(#{prefix}-arrow-a)")]
    for i, h in enumerate([45, 50, 42, 55, 48, 44]):
        parts.append(rect(544 + i * 20, 318 - h, 12, h, a, "none", 5, opacity=0.72))
    parts += [tx(604, 340, "γ, β retuned", 13, MUTED, 700, "middle")]
    return "\n".join(parts)


def renderer_resnet(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "identity highway over residual blocks", a)]
    xs = [148, 270, 392, 514]
    for i, x in enumerate(xs):
        parts += [rect(x - 44, 224, 88, 68, "#ffffff", a, 16, extra='filter="url(#small-shadow)"'), tx(x, 264, f"F{i+1}(x)", 15, INK, 800, "middle")]
        if i < len(xs) - 1:
            parts.append(line(x + 44, 258, xs[i + 1] - 44, 258, a, 3, 0.8, f"url(#{prefix}-arrow-a)"))
    parts += [
        path("M102,258 C130,160 488,160 542,224", b, "none", 5, 0.78, f"url(#{prefix}-arrow-b)"),
        tx(334, 184, "skip connection: y = x + F(x)", 17, b, 800, "middle"),
        circle(574, 258, 27, "#fff3f8", b, 2.4),
        tx(574, 265, "+", 28, b, 900, "middle"),
        line(602, 258, 668, 258, b, 3, 0.75, f"url(#{prefix}-arrow-b)"),
    ]
    return "\n".join(parts)


def renderer_dominant(prefix: str, a: str, b: str) -> str:
    parts = [small_label(88, 154, "one paradigm spreads across domains", a)]
    parts += [circle(380, 252, 62, f"url(#{prefix}-accent)", "#ffffff", 4), tx(380, 248, "deep", 21, "#ffffff", 900, "middle"), tx(380, 272, "learning", 18, "#ffffff", 900, "middle")]
    domains = [(178, 188, "speech"), (572, 188, "vision"), (164, 304, "NLP"), (596, 306, "RL"), (304, 350, "genomics"), (466, 350, "drug discovery")]
    for x, y, label in domains:
        parts += [line(380, 252, x, y, "#8db2df", 1.8, 0.45), circle(x, y, 38, "#ffffff", a if x < 380 else b, 2.2, 1), tx(x, y + 5, label, 12, INK, 800, "middle")]
    return "\n".join(parts)


def renderer_alphago(prefix: str, a: str, b: str) -> str:
    parts = [small_label(86, 154, "policy, value, search, self-play", a), rect(92, 178, 210, 210, "#f8e4b7", "#c7922b", 16)]
    for i in range(1, 9):
        parts.append(line(92 + i * 21, 178, 92 + i * 21, 388, "#9a6b2f", 1, 0.65))
        parts.append(line(92, 178 + i * 21, 302, 178 + i * 21, "#9a6b2f", 1, 0.65))
    for x, y, fill in [(155, 241, INK), (197, 241, "#ffffff"), (218, 283, INK), (176, 304, "#ffffff"), (239, 325, INK)]:
        parts.append(circle(x, y, 9, fill, "#1a1a1a", 1))
    parts += [card(366, 188, 118, 66, "policy net", "#eef5ff", a), card(366, 292, 118, 66, "value net", "#fff3f8", b), line(302, 260, 366, 222, a, 2.5, 0.8, f"url(#{prefix}-arrow-a)"), line(302, 304, 366, 326, b, 2.5, 0.8, f"url(#{prefix}-arrow-b)")]
    parts += [path("M532,324 C512,274 526,226 572,204 C628,180 668,220 650,272", a, "none", 3.5, 0.75, f"url(#{prefix}-arrow-a)"), tx(608, 326, "MCTS tree", 16, INK, 800, "middle")]
    return "\n".join(parts)


def renderer_transformer(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "tokens attend to every other token in parallel", a)]
    tokens = ["Q", "K", "V", "softmax", "heads"]
    for i, tok in enumerate(tokens):
        x = 100 + i * 118
        parts += [rect(x, 300, 84, 38, "#ffffff", a if i < 3 else b, 11), tx(x + 42, 325, tok, 14, INK, 800, "middle")]
    for i in range(5):
        for j in range(5):
            op = 0.18 + (i == j) * 0.38 + ((i + j) % 3 == 0) * 0.22
            parts.append(rect(166 + j * 38, 178 + i * 28, 32, 22, a if i <= j else b, "none", 5, opacity=op))
    parts += [tx(262, 374, "attention map", 15, INK, 800, "middle"), path("M430,188 C500,156 580,174 628,224 C662,260 650,304 602,328", b, "none", 4, 0.72, f"url(#{prefix}-arrow-b)"), tx(564, 188, "multi-head prism", 15, b, 800, "middle")]
    return "\n".join(parts)


def renderer_bert(prefix: str, a: str, b: str) -> str:
    parts = [small_label(86, 154, "bidirectional pretraining by masking words", a)]
    tokens = ["the", "[MASK]", "learns", "context"]
    for i, tok in enumerate(tokens):
        x = 98 + i * 128
        fill = "#fff3f8" if tok == "[MASK]" else "#ffffff"
        stroke = b if tok == "[MASK]" else a
        parts += [rect(x, 218, 98, 44, fill, stroke, 12), tx(x + 49, 246, tok, 14, INK, 800, "middle")]
    parts += [path("M148,218 C208,168 366,168 530,218", a, "none", 3, 0.42), path("M530,262 C456,324 264,324 148,262", b, "none", 3, 0.42), card(292, 306, 174, 58, "fine-tune", "#eef5ff", a, "QA · NLI · tagging")]
    return "\n".join(parts)


def renderer_neural_ode(prefix: str, a: str, b: str) -> str:
    parts = [small_label(86, 154, "depth becomes a continuous trajectory", a)]
    for i in range(7):
        x = 118 + i * 82
        parts.append(line(x, 334, x + 24, 286 - i * 13, "#9fb2cb", 1.2, 0.4, f"url(#{prefix}-arrow-ink)"))
    parts += [path("M106,324 C194,262 230,326 312,252 C408,166 496,214 638,158", a, "none", 5, 0.92, f"url(#{prefix}-arrow-a)"), path("M638,196 C540,262 472,218 378,288 C282,360 196,300 106,354", b, "none", 3.5, 0.68, f"url(#{prefix}-arrow-b)", dash="8 7"), tx(400, 188, "dz/dt = f(z,t)", 18, a, 800, "middle"), tx(388, 338, "adjoint solves gradients backward", 15, b, 800, "middle")]
    return "\n".join(parts)


def renderer_scaling(prefix: str, a: str, b: str) -> str:
    parts = [small_label(86, 154, "predictable power laws over scale", a), line(112, 340, 650, 340, "#a8b7cc", 2), line(112, 340, 112, 174, "#a8b7cc", 2)]
    parts += [path("M126,192 C224,218 318,248 420,286 C506,318 586,332 644,338", a, "none", 4.5, 0.9), path("M126,192 L644,338", b, "none", 2, 0.35, dash="7 7")]
    for x, y, label in [(154, 200, "small"), (310, 246, "medium"), (484, 308, "large"), (620, 336, "frontier")]:
        parts += [circle(x, y, 9, "#ffffff", a, 2), tx(x, y - 18, label, 12, MUTED, 700, "middle")]
    parts += [tx(402, 182, "loss ∝ compute^-α", 18, b, 900, "middle"), tx(638, 360, "scale", 12, MUTED, 700, "end"), tx(96, 184, "loss", 12, MUTED, 700, "end")]
    return "\n".join(parts)


def renderer_gpt3(prefix: str, a: str, b: str) -> str:
    parts = [small_label(86, 154, "few-shot prompts steer a giant model", a), rect(92, 192, 188, 160, "#ffffff", "#d8e5f3", 20, extra='filter="url(#small-shadow)"')]
    prompts = ["translate: chat →", "sentiment: great →", "code: fib(n) →"]
    for i, p in enumerate(prompts):
        parts.append(tx(116, 230 + i * 36, p, 14, INK, 700))
    parts += [line(280, 272, 374, 272, a, 3.4, 0.8, f"url(#{prefix}-arrow-a)"), circle(466, 272, 82, f"url(#{prefix}-accent)", "#ffffff", 4), tx(466, 264, "175B", 34, "#ffffff", 900, "middle"), tx(466, 294, "parameters", 16, "#ffffff", 800, "middle"), line(548, 272, 640, 272, b, 3.4, 0.8, f"url(#{prefix}-arrow-b)"), tx(648, 278, "task solved", 16, b, 800)]
    return "\n".join(parts)


def renderer_diffusion(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "learn to reverse a noising process", a)]
    xs = [110, 244, 378, 512, 646]
    shades = ["#ffffff", "#e8eef6", "#b9c5d3", "#738195", "#273142"]
    for i, x in enumerate(xs):
        parts += [rect(x - 38, 220, 76, 76, shades[i], "#d6e1ee", 12), tx(x, 318, ["clean", "t=100", "t=500", "t=800", "noise"][i], 12, MUTED, 700, "middle")]
        if i < len(xs) - 1:
            parts.append(line(x + 44, 258, xs[i + 1] - 44, 258, a, 2.8, 0.75, f"url(#{prefix}-arrow-a)"))
    parts += [path("M648,188 C514,152 370,152 222,184 C152,200 120,180 92,162", b, "none", 4, 0.76, f"url(#{prefix}-arrow-b)"), tx(378, 176, "reverse denoising model", 17, b, 800, "middle")]
    return "\n".join(parts)


def renderer_clip(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "images and captions share an embedding space", a), card(92, 190, 150, 74, "image encoder", "#eef5ff", a), card(92, 298, 150, 74, "text encoder", "#fff3f8", b), line(242, 226, 330, 252, a, 3, 0.75, f"url(#{prefix}-arrow-a)"), line(242, 334, 330, 292, b, 3, 0.75, f"url(#{prefix}-arrow-b)")]
    for i in range(5):
        for j in range(5):
            fill = "#eaf0f8"
            op = 0.9
            if i == j:
                fill = f"url(#{prefix}-accent)"
                op = 0.86
            parts.append(rect(408 + j * 35, 184 + i * 35, 30, 30, fill, "#ffffff", 5, opacity=op))
    parts += [tx(494, 378, "contrastive positives on the diagonal", 14, INK, 800, "middle"), circle(342, 274, 42, "#ffffff", "#d8e5f3", 2), tx(342, 279, "shared", 15, INK, 800, "middle")]
    return "\n".join(parts)


def renderer_alphafold(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "sequence attention folds into 3D structure", a)]
    for i in range(6):
        parts.append(rect(92 + i * 38, 196 + (i % 2) * 28, 30, 72, "#eef5ff", a, 7, opacity=0.82))
    parts += [tx(202, 302, "MSA + pair representation", 14, MUTED, 800, "middle"), line(312, 248, 398, 248, a, 3, 0.82, f"url(#{prefix}-arrow-a)"), card(408, 206, 132, 84, "Evoformer", "#ffffff", a, "attention blocks"), line(540, 248, 596, 248, b, 3, 0.82, f"url(#{prefix}-arrow-b)")]
    ribbon = "M604,286 C574,236 618,194 662,224 C704,256 666,326 612,330 C556,334 556,276 604,286"
    parts += [path(ribbon, b, "none", 8, 0.7), path("M606,286 C632,258 642,248 664,226", a, "none", 5, 0.8), tx(644, 354, "protein structure", 14, INK, 800, "middle")]
    return "\n".join(parts)


def renderer_rlhf(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "human preferences become a reward signal", a)]
    stages = [(108, "SFT", "demos"), (306, "Reward", "rank A vs B"), (512, "RL policy", "optimize")]
    for x, title, sub in stages:
        parts.append(card(x, 214, 140, 86, title, "#ffffff", a if title != "Reward" else b, sub))
    parts += [line(248, 257, 306, 257, a, 3, 0.8, f"url(#{prefix}-arrow-a)"), line(446, 257, 512, 257, b, 3, 0.8, f"url(#{prefix}-arrow-b)"), rect(300, 322, 156, 40, "#fff8e8", "#f09c3b", 12), tx(378, 348, "human preference", 14, INK, 800, "middle"), line(378, 322, 378, 300, "#f09c3b", 2.5, 0.8, f"url(#{prefix}-arrow-ink)")]
    return "\n".join(parts)


def renderer_chatgpt(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "multi-turn conversation becomes the interface", a)]
    bubbles = [(92, 188, 300, 44, "Explain self-attention intuitively.", "#eef5ff", a), (286, 252, 350, 66, "Sure. Each token asks what it should look at, then mixes information from the answers.", "#fff3f8", b), (96, 342, 280, 40, "Can you make it visual?", "#eef5ff", a)]
    for x, y, w, h, label, fill, stroke in bubbles:
        parts += [rect(x, y, w, h, fill, stroke, 20, extra='filter="url(#small-shadow)"'), multiline(x + 22, y + 28, wrap(label, 43), 13, INK, 700, "start", 17)]
    parts += [circle(648, 222, 38, f"url(#{prefix}-accent)", "#ffffff", 3), tx(648, 228, "LLM", 18, "#ffffff", 900, "middle")]
    return "\n".join(parts)


def renderer_gpt4(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "frontier model reads mixed inputs", a), card(92, 184, 132, 76, "image", "#eef5ff", a, "diagram/photo"), card(92, 296, 132, 76, "text", "#fff3f8", b, "prompt"), circle(386, 276, 78, f"url(#{prefix}-accent)", "#ffffff", 4), tx(386, 270, "GPT-4", 30, "#ffffff", 900, "middle"), tx(386, 298, "reasoning", 16, "#ffffff", 800, "middle"), line(224, 222, 318, 262, a, 3, 0.75, f"url(#{prefix}-arrow-a)"), line(224, 334, 318, 294, b, 3, 0.75, f"url(#{prefix}-arrow-b)"), card(548, 228, 116, 96, "answer", "#ffffff", "#d8e5f3", "exam-grade")]
    parts.append(line(464, 276, 548, 276, a, 3, 0.85, f"url(#{prefix}-arrow-a)"))
    return "\n".join(parts)


def renderer_gpt4o(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "audio, vision, text in one real-time loop", a), circle(380, 260, 72, f"url(#{prefix}-accent)", "#ffffff", 4), tx(380, 254, "omni", 28, "#ffffff", 900, "middle"), tx(380, 284, "model", 18, "#ffffff", 800, "middle")]
    inputs = [(154, 184, "voice"), (130, 260, "vision"), (154, 336, "text")]
    outputs = [(604, 184, "voice"), (630, 260, "vision"), (604, 336, "text")]
    for x, y, label in inputs:
        parts += [circle(x, y, 32, "#ffffff", a, 2.2), tx(x, y + 5, label, 12, INK, 800, "middle"), line(x + 34, y, 318, 260, a, 2.5, 0.55, f"url(#{prefix}-arrow-a)")]
    for x, y, label in outputs:
        parts += [line(442, 260, x - 34, y, b, 2.5, 0.55, f"url(#{prefix}-arrow-b)"), circle(x, y, 32, "#fff3f8", b, 2.2), tx(x, y + 5, label, 12, INK, 800, "middle")]
    return "\n".join(parts)


def renderer_reasoning(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "spend inference compute exploring solution paths", a)]
    roots = [(380, 188, "problem")]
    level2 = [(230, 260, "try A"), (380, 260, "try B"), (530, 260, "try C")]
    level3 = [(304, 332, "check"), (456, 332, "proof")]
    for x, y, label in roots + level2 + level3:
        fill = "#fff8e8" if label in ("check", "proof") else "#ffffff"
        stroke = b if label in ("check", "proof") else a
        parts += [rect(x - 54, y - 22, 108, 44, fill, stroke, 12), tx(x, y + 5, label, 13, INK, 800, "middle")]
    for x2, y2, _ in level2:
        parts.append(line(380, 210, x2, y2 - 22, a, 2, 0.55, f"url(#{prefix}-arrow-a)"))
    for x1, y1, _ in [level2[1], level2[2]]:
        for x2, y2, _ in level3:
            if abs(x1 - x2) < 120:
                parts.append(line(x1, y1 + 22, x2, y2 - 22, b, 2.5, 0.7, f"url(#{prefix}-arrow-b)"))
    parts += [circle(590, 332, 32, b, "#ffffff", 3), tx(590, 338, "✓", 26, "#ffffff", 900, "middle")]
    return "\n".join(parts)


def renderer_nobel(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "deep learning recognized as fundamental science", a)]
    for x, title, sub, color in [(230, "Physics", "Hopfield + Hinton", a), (530, "Chemistry", "proteins + prediction", b)]:
        parts += [circle(x, 250, 78, "#fff4c2", "#c7922b", 5, 1), circle(x, 250, 58, "#ffe99b", "#c7922b", 2), tx(x, 238, "2024", 21, "#805a18", 900, "middle"), tx(x, 266, title, 18, "#805a18", 900, "middle"), tx(x, 350, sub, 14, INK, 800, "middle")]
    parts += [path("M308,250 C366,204 432,204 452,250", "#c7922b", "none", 3, 0.6, f"url(#{prefix}-arrow-ink)")]
    return "\n".join(parts)


def renderer_deepseek(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "verifiable rewards forge reasoning behavior", a), card(90, 220, 142, 84, "base model", "#eef5ff", a, "pretrained"), line(232, 262, 318, 262, a, 3, 0.8, f"url(#{prefix}-arrow-a)"), rect(320, 190, 166, 144, "#fff8e8", "#f09c3b", 22, extra='filter="url(#small-shadow)"'), tx(403, 228, "RL forge", 22, INK, 900, "middle"), tx(403, 260, "math answer ✓", 13, MUTED, 800, "middle"), tx(403, 282, "code tests ✓", 13, MUTED, 800, "middle"), line(486, 262, 566, 262, b, 3, 0.8, f"url(#{prefix}-arrow-b)"), card(566, 220, 122, 84, "reasoner", "#fff3f8", b, "R1/distill")]
    return "\n".join(parts)


def renderer_agentic(prefix: str, a: str, b: str) -> str:
    parts = [small_label(84, 154, "models become systems that operate tools", a), circle(380, 260, 70, f"url(#{prefix}-accent)", "#ffffff", 4), tx(380, 254, "agent", 28, "#ffffff", 900, "middle"), tx(380, 282, "orchestrator", 14, "#ffffff", 800, "middle")]
    tools = [(180, 180, "browser"), (582, 178, "code"), (160, 288, "memory"), (596, 292, "files"), (286, 352, "tools"), (486, 352, "calendar")]
    for x, y, label in tools:
        parts += [line(380, 260, x, y, "#8db2df", 1.8, 0.45), rect(x - 48, y - 24, 96, 48, "#ffffff", a if x < 380 else b, 14), tx(x, y + 5, label, 13, INK, 800, "middle")]
    return "\n".join(parts)


RENDERERS: dict[str, Callable[[str, str, str], str]] = {
    "adjoint": renderer_adjoint,
    "reverse_ad": renderer_reverse_ad,
    "werbos": renderer_werbos,
    "hopfield": renderer_hopfield,
    "boltzmann": renderer_boltzmann,
    "backprop": renderer_backprop,
    "cnn": renderer_cnn,
    "vanishing": renderer_vanishing,
    "chunker": renderer_chunker,
    "lstm": renderer_lstm,
    "language_model": renderer_language_model,
    "rbm": renderer_rbm,
    "dbn": renderer_dbn,
    "pyramid": renderer_pyramid,
    "gpu": renderer_gpu,
    "alexnet": renderer_alexnet,
    "dropout": renderer_dropout,
    "word2vec": renderer_word2vec,
    "seq2seq": renderer_seq2seq,
    "attention": renderer_attention,
    "gan": renderer_gan,
    "batchnorm": renderer_batchnorm,
    "resnet": renderer_resnet,
    "dominant": renderer_dominant,
    "alphago": renderer_alphago,
    "transformer": renderer_transformer,
    "bert": renderer_bert,
    "neural_ode": renderer_neural_ode,
    "scaling": renderer_scaling,
    "gpt3": renderer_gpt3,
    "diffusion": renderer_diffusion,
    "clip": renderer_clip,
    "alphafold": renderer_alphafold,
    "rlhf": renderer_rlhf,
    "chatgpt": renderer_chatgpt,
    "gpt4": renderer_gpt4,
    "gpt4o": renderer_gpt4o,
    "reasoning": renderer_reasoning,
    "nobel": renderer_nobel,
    "deepseek": renderer_deepseek,
    "agentic": renderer_agentic,
}


def render_scene(slug: str, title: str, period: str) -> str:
    scene = SCENES.get(slug)
    if scene is None:
        accent, secondary = deterministic_palette(slug)
        scene = Scene(accent, secondary, "dominant")

    prefix, opening = base_open(slug, title, period, scene.accent, scene.secondary)
    body = RENDERERS[scene.renderer](prefix, scene.accent, scene.secondary)
    svg = "\n".join([opening, body, base_close()])
    return svg.replace("url(#shadow)", f"url(#{prefix}-shadow)").replace(
        "url(#small-shadow)", f"url(#{prefix}-small-shadow)"
    )


def refresh_file(path: Path) -> int:
    original = path.read_text(encoding="utf-8")
    fallback_slug = path.stem

    def replace(match: re.Match[str]) -> str:
        slug = slug_before(original, match.start(), fallback_slug)
        title = last_match(HEADING_RE, original, match.start()) or "Deep Learning Timeline"
        period = last_match(PERIOD_RE, original, match.start()) or "Timeline"
        return render_scene(slug, title, period)

    updated = SVG_RE.sub(replace, original)
    if updated != original:
        path.write_text(updated, encoding="utf-8")
    return len(SVG_RE.findall(updated))


def main() -> None:
    total = 0
    for path in sorted(ROOT.glob("*.html")):
        total += refresh_file(path)
    print(f"Regenerated {total} illustrative SVG scenes.")


if __name__ == "__main__":
    main()
