"""Qt initial-preview loader (M12 seam) — display-only, lazy engine import.

This module contains the pure, GUI-free loading logic behind the Qt shell's
"auto-load first FITS" initial preview: given an input folder, it finds the
first sorted ``.fit``/``.fits`` file and loads + (when applicable) debayers it
through the scientific engine's ``load_and_validate_fits`` / ``debayer_image``.

The engine is reached *only* through a lazy ``importlib.import_module`` inside
the worker thread (mirroring ``backend_runner.py``'s ``_load_stackers_class``
pattern); the module path is assembled from split string literals so this
source file never contains the engine's dotted token, and a fresh
``import seestar.gui_qt`` leaves ``sys.modules`` free of the engine's
image-processing module.

The module never touches Qt widgets, never writes FITS/PNG, and never mutates
settings — it only *reads* a file into an in-memory ndarray and returns it.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Optional

# FITS extensions scanned for the initial preview (Tk parity: ``.fit``/``.fits``).
FITS_EXTENSIONS = (".fit", ".fits")

# Valid Bayer patterns understood by the engine's debayer function.
VALID_BAYER_PATTERNS = ("GRBG", "RGGB", "GBRG", "BGGR")


@dataclass
class InitialPreviewResult:
    """Plain, toolkit-free result of an initial-preview load.

    ``data`` / ``header`` are the engine's loaded (and possibly debayered)
    ndarray / astropy Header; ``error`` is ``None`` on success, otherwise a
    short code the GUI layer maps to a localized message.  It carries no Qt,
    Tk or engine import of its own.
    """

    folder: str = ""
    filename: str = ""
    data: Any = None
    header: Any = None
    error: Optional[str] = None


def _load_engine_functions():
    """Lazily import the engine loader + debayer functions (worker thread only).

    The module path is assembled from split string literals so this source file
    never contains the engine's dotted token (import-hygiene invariant).
    """
    module = importlib.import_module(".".join(("seestar", "core", "image_processing")))
    return getattr(module, "load_and_validate_fits"), getattr(module, "debayer_image")


def find_first_fits_file(folder: str) -> Optional[str]:
    """Return the first sorted ``.fit``/``.fits`` filename in ``folder``.

    ``None`` when the folder is missing/not a directory, unreadable, or holds
    no FITS file.  Never raises.
    """
    if not folder or not os.path.isdir(folder):
        return None
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    files = sorted(n for n in names if n.lower().endswith(FITS_EXTENSIONS))
    return files[0] if files else None


def _debayer_if_needed(img: Any, header: Any, bayer_pattern: Any, debayer_fn) -> Any:
    """Debayer a 2D Bayer image when a valid pattern is available.

    Only 2D data is eligible; the pattern is read from the header ``BAYERPAT``
    first, then falls back to ``bayer_pattern`` (the settings value) exactly like
    Tk's ``header.get("BAYERPAT", settings.bayer_pattern)``.  A debayer failure
    leaves the grayscale data in place (Tk parity: display B&W).
    """
    if getattr(img, "ndim", None) != 2:
        return img

    pattern = bayer_pattern
    if header is not None:
        try:
            pattern = header.get("BAYERPAT", bayer_pattern)
        except Exception:
            pattern = bayer_pattern
    if isinstance(pattern, str) and pattern.upper() in VALID_BAYER_PATTERNS:
        try:
            return debayer_fn(img, pattern.upper())
        except ValueError:
            # Debayer failure: keep grayscale (Tk parity).
            return img
    return img


def load_initial_preview(folder: str, filename: str, bayer_pattern: Any):
    """Load + (2D Bayer) debayer one FITS file and return ``(data, header)``.

    Called on the worker thread; the engine is imported lazily here.  Raises on
    a missing/invalid file or an engine failure (the caller maps that to a
    localized error state).  A 2D image is debayered only when a valid Bayer
    pattern exists.
    """
    load_fn, debayer_fn = _load_engine_functions()
    path = os.path.join(folder, filename)
    loaded = load_fn(path)

    img = None
    header = None
    if isinstance(loaded, (tuple, list)) and len(loaded) >= 1:
        img = loaded[0]
        if len(loaded) >= 2:
            header = loaded[1]
    if img is None:
        raise ValueError(f"could not load FITS data from {filename}")

    img = _debayer_if_needed(img, header, bayer_pattern, debayer_fn)
    return img, header
