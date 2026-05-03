#!/usr/bin/env python3
"""
Build docs/ directory for MkDocs from per-stub folders.

mkdocs requires docs_dir to be a child directory of the config file,
so we can't point at the repo root directly. This script copies the
relevant content into docs/ before mkdocs build.

Usage:
    python3 bin/build_docs.py

CI runs this before `mkdocs build`. docs/ is gitignored.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo",
    ".cache", "*.npz", "*.tar.gz", "*.gz",
)


def is_stub_folder(p: Path) -> bool:
    """A stub folder has README.md and at least one .py file at top level."""
    if not p.is_dir():
        return False
    if p.name.startswith(".") or p.name in {"docs", "bin", "site"}:
        return False
    if not (p / "README.md").exists():
        return False
    return any(child.suffix == ".py" for child in p.iterdir())


def main() -> None:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir()

    # Top-level docs
    for name in ("README.md", "RESULTS.md"):
        src = ROOT / name
        if src.exists():
            shutil.copy(src, DOCS / name)

    # Per-stub folders
    stubs = sorted(p.name for p in ROOT.iterdir() if is_stub_folder(p))
    print(f"Found {len(stubs)} stub folders")
    for stub in stubs:
        src = ROOT / stub
        dst = DOCS / stub
        shutil.copytree(src, dst, ignore=IGNORE)

    print(f"Built {DOCS} with {len(stubs) + 2} top-level entries")


if __name__ == "__main__":
    main()
