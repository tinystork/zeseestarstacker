"""C12 — bounded recursive masters scan + flatten for non-recursive admission.

ZeCalibrator's ``open_session_library`` scans only the **top level** of the
chosen masters folder.  Real libraries (e.g. M74) are nested several levels deep
(bias/…, dark/…, Flat/…).  This module provides a Zsss-side **bounded recursive
scan** (FITS-only, depth-capped, excluding known output dirs) plus a **flatten**
step that materialises the candidates into a single flat directory so the
existing non-recursive admission can consume them.  Admission and role
identification stay ZeCalibrator's: this module only *finds* FITS files, never
decides a role or a scientific fact.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List, Optional

FITS_SUFFIXES = (".fits", ".fit", ".fts")

# Bounded recursion depth (documented default): enough for the 3-4 level M74
# layout (bias/temp -10/…, dark/…/dark/…), never unbounded.
DEFAULT_MAX_DEPTH = 4

# Known output directories to skip so past run outputs are never ingested as
# master candidates.
DEFAULT_EXCLUDED_DIR_NAMES = frozenset({"stacked", "calibrated", "processed", "output"})


def scan_masters_recursive(
    root,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    exclude_dir_names: Iterable[str] = DEFAULT_EXCLUDED_DIR_NAMES,
) -> List[str]:
    """Recursively scan a masters folder for FITS files (bounded depth).

    Returns a deterministic (sorted) list of absolute paths.  FITS-only; skips
    known output directories by (lower-cased) name; never recurses past
    ``max_depth``.  Read-only: the source folder is never modified.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    excluded = {str(n).lower() for n in exclude_dir_names}
    out: List[str] = []

    def _walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.is_dir():
                if entry.name.lower() in excluded:
                    continue
                _walk(entry, depth + 1)
            elif entry.is_file() and entry.suffix.lower() in FITS_SUFFIXES:
                out.append(str(entry))

    _walk(root, 0)
    return out


def flatten_masters(candidates: Iterable[str], flat_dir) -> str:
    """Copy candidate FITS into a single top-level directory.

    Returns the flat directory path.  Name collisions are disambiguated with a
    numeric suffix; a missing/unreadable candidate is skipped (never fatal).  The
    copy is a full materialisation (works across filesystems, unlike hardlinks).
    """
    flat = Path(flat_dir)
    flat.mkdir(parents=True, exist_ok=True)
    used: set = set()
    for cand in candidates:
        src = Path(cand)
        if not src.is_file():
            continue
        name = src.name
        stem, suffix = src.stem, src.suffix
        i = 1
        while name.lower() in used:
            name = f"{stem}_{i}{suffix}"
            i += 1
        used.add(name.lower())
        try:
            shutil.copy2(str(src), str(flat / name))
        except OSError:
            continue
    return str(flat)


__all__ = [
    "DEFAULT_EXCLUDED_DIR_NAMES",
    "DEFAULT_MAX_DEPTH",
    "FITS_SUFFIXES",
    "flatten_masters",
    "scan_masters_recursive",
]
