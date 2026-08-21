"""Fresh-copy helper for the immutable master reference dataset.

The master dataset under ``fixture/master/`` is NEVER mutated.  Tests (or
manual runs) that need real FITS inputs copy fresh images out of the master
into a scratch directory (``tmp_path`` in pytest, or a named target).

Usage::

    from fixture.copy_fresh import copy_master_to

    copied = copy_master_to(tmp_path)               # all 10 images
    copied = copy_master_to(tmp_path, n=4)          # first 4 images only
"""

from __future__ import annotations

import shutil
from pathlib import Path

MASTER_DIR = Path(__file__).resolve().parent / "master"


def master_paths() -> list[Path]:
    """All master FITS, in deterministic (sorted) order."""
    if not MASTER_DIR.is_dir():
        raise FileNotFoundError(
            f"Master fixture dir not found: {MASTER_DIR}. "
            "Run `python fixture/generate_master.py` first."
        )
    return sorted(MASTER_DIR.glob("Light_*.fit"))


def copy_master_to(target: Path, n: int | None = None) -> list[Path]:
    """Copy fresh master images into ``target`` and return the new paths.

    ``target`` is created if needed.  ``n`` limits the number of images
    (first ``n`` in sorted order); ``None`` copies all.
    """
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    sources = master_paths()
    if n is not None:
        sources = sources[:n]
    out: list[Path] = []
    for src in sources:
        dst = target / src.name
        shutil.copy2(src, dst)
        out.append(dst)
    return out


if __name__ == "__main__":
    import sys

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fixture/tmp-scratch")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    copied = copy_master_to(dest, n=n)
    print(f"Copied {len(copied)} master images to {dest}")
    for p in copied:
        print(f"  {p}")
