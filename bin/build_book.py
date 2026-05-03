#!/usr/bin/env python3
"""
Build src/ for mdBook from per-stub folders + top-level docs.

mdBook requires:
- book.toml at repo root (already present)
- src/ with chapter .md files referenced by src/SUMMARY.md

This script:
1. Resets src/
2. Copies README.md -> src/index.md
3. Copies RESULTS.md -> src/results.md
4. Copies each stub folder -> src/<slug>/ (READMEs + viz/ + .gif)
5. Generates src/SUMMARY.md grouped by decade

Usage:
    python3 bin/build_book.py

CI runs this before `mdbook build`. src/ is gitignored.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo",
    ".cache", "*.npz", "*.tar.gz", "*.gz",
)

# Decade grouping for SUMMARY.md. Order within each decade is curated.
DECADES = [
    ("1980s — Foundations", [
        "encoder-4-2-4",  # worked example
        "encoder-3-parity",
        "encoder-4-3-4",
        "encoder-8-3-8",
        "encoder-40-10-40",
        "encoder-backprop-8-3-8",
        "xor",
        "n-bit-parity",
        "symmetry",
        "negation",
        "binary-addition",
        "t-c-discrimination",
        "recurrent-shift-register",
        "sequence-lookup-25",
        "distributed-to-local-bottleneck",
        "shifter",
        "grapheme-sememe",
        "family-trees",
        "riser-spectrogram",
        "fast-weights-rehearsal",
    ]),
    ("1990s — Unsupervised & Helmholtz", [
        "vowel-mixture-experts",
        "random-dot-stereograms",
        "sunspots",
        "spline-images-factorial-vq",
        "dipole-position",
        "dipole-3d-constraint",
        "dipole-what-where",
        "helmholtz-shifter",
        "bars",
    ]),
    ("2000s — RBMs & deep belief", [
        "bars-rbm",
        "transforming-pairs",
        "bouncing-balls-2",
        "bouncing-balls-3",
    ]),
    ("2010s — Capsules, distillation, attention", [
        "transforming-autoencoders",
        "deep-lambertian-spheres",
        "rnn-pathological",
        "distillation-mnist-omitted-3",
        "air-multimnist",
        "air-3d-primitives",
        "fast-weights-associative-retrieval",
        "multi-level-glimpse-mnist",
        "catch-game",
        "affnist",
        "multimnist-capsnet",
        "smallnorb-novel-viewpoint",
        "constellations",
    ]),
    ("2020s — Subclass, GLOM, Forward-Forward", [
        "mnist-2x5-subclass",
        "geo-flow-capsules",
        "ellipse-world",
        "ff-hybrid-mnist",
        "ff-label-in-input",
        "ff-recurrent-mnist",
        "ff-cifar-locally-connected",
        "ff-aesop-sequences",
    ]),
]


def stub_title(slug: str) -> str:
    """Pretty title for nav. Mark the worked example."""
    if slug == "encoder-4-2-4":
        return "encoder-4-2-4 ★ (worked example)"
    return slug


def main() -> None:
    if SRC.exists():
        shutil.rmtree(SRC)
    SRC.mkdir()

    # Top-level pages
    shutil.copy(ROOT / "README.md", SRC / "index.md")
    shutil.copy(ROOT / "RESULTS.md", SRC / "results.md")

    # Per-stub folders
    all_stubs: list[str] = []
    for _, slugs in DECADES:
        all_stubs.extend(slugs)

    missing: list[str] = []
    for slug in all_stubs:
        src_dir = ROOT / slug
        if not src_dir.exists():
            missing.append(slug)
            continue
        dst_dir = SRC / slug
        shutil.copytree(src_dir, dst_dir, ignore=IGNORE)

    if missing:
        print(f"WARNING: {len(missing)} stub folders missing: {missing}")

    # Generate SUMMARY.md
    summary = ["# Summary", ""]
    summary.append("[Home](index.md)")
    summary.append("[Results catalog](results.md)")
    summary.append("")
    for decade, slugs in DECADES:
        summary.append(f"# {decade}")
        summary.append("")
        for slug in slugs:
            if slug in missing:
                continue
            summary.append(f"- [{stub_title(slug)}]({slug}/README.md)")
        summary.append("")

    (SRC / "SUMMARY.md").write_text("\n".join(summary) + "\n")

    n_chapters = len(all_stubs) - len(missing)
    print(f"Built {SRC} with {n_chapters} stub chapters + 2 top-level pages")


if __name__ == "__main__":
    main()
