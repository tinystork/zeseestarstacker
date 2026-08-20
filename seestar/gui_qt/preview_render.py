"""Qt preview image renderer (M8 seam) — strictly display-only.

This module converts *image-like* preview data carried by a
:class:`~seestar.gui_qt.backend_runner.BackendPreviewPayload` into a copied Qt
:class:`QImage` for the Preview tab.  It is the first real rendering seam for
the Qt shell and stays strictly display-only:

* it never touches science output or the Tk GUI,
* it imports PySide6 Qt image classes at module import time (cheap, and always
  available whenever the Qt shell is in use),
* it imports numpy *lazily*, inside the conversion path only, so a fresh
  ``import seestar.gui_qt`` never pulls numpy into ``sys.modules``; when numpy
  is unavailable (or the data is not image-like) the renderer returns ``None``
  without raising.

Input duck-typing (from ``BackendPreviewPayload.data``):

* a ``tuple``/``list`` is treated as the real stacker's
  ``(display_array, hist_array, ...)`` shape and the *first* image-like
  element is rendered;
* 2D arrays render as grayscale, 3D arrays with 3/4 channels render as
  RGB/RGBA (channels-last layout);
* float data is clipped and scaled from ``[0, 1]`` to ``uint8``; ``uint8``
  data passes through; other integer data is clipped to the 8-bit display
  range (no stretch controls yet — that is a later milestone);
* missing/invalid data yields ``None`` (no image) and never raises.

The returned :class:`QImage` is always a deep copy, so no reference to a
temporary numpy buffer (or the payload's data) outlives the call.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtGui import QImage


def _load_numpy():
    """Lazily import numpy (the module object, or ``None`` when unavailable)."""
    try:
        import importlib

        return importlib.import_module("numpy")
    except Exception:
        return None


def _is_array_like(np: Any, obj: Any) -> bool:
    """True for ndarrays and duck-typed array-likes (but not str/bytes)."""
    if isinstance(obj, np.ndarray):
        return True
    if isinstance(obj, (str, bytes, bytearray, memoryview)):
        return False
    return hasattr(obj, "shape") and hasattr(obj, "__array__")


def _extract_array(np: Any, data: Any):
    """Return the first array candidate inside ``data``, or ``None``.

    A tuple/list is treated as the real stacker's
    ``(display_array, hist_array, ...)`` carrier: the first array-like element
    wins.  When no element is array-like, the sequence itself is treated as a
    (nested-list) image so plain Python lists of lists also render.
    """
    if data is None:
        return None
    if isinstance(data, (tuple, list)):
        for item in data:
            if _is_array_like(np, item):
                try:
                    return np.asarray(item)
                except Exception:
                    return None
        try:
            return np.asarray(data)
        except Exception:
            return None
    if _is_array_like(np, data):
        try:
            return np.asarray(data)
        except Exception:
            return None
    return None


def _iter_array_candidates(np: Any, data: Any):
    """Yield array candidates from preview ``data`` without raising.

    Tuple/list payloads are common for the real stacker.  We try each
    array-like element independently so a leading non-image helper array (for
    example histogram data) does not prevent a later image-like element from
    rendering.  If no element is array-like, the whole sequence is tried as a
    nested-list image.
    """
    if data is None:
        return
    if isinstance(data, (tuple, list)):
        found_array_like = False
        for item in data:
            if not _is_array_like(np, item):
                continue
            found_array_like = True
            try:
                yield np.asarray(item)
            except Exception:
                continue
        if not found_array_like:
            try:
                yield np.asarray(data)
            except Exception:
                return
        return
    arr = _extract_array(np, data)
    if arr is not None:
        yield arr


def _to_uint8(np: Any, arr: Any):
    """Normalize ``arr`` to a ``uint8`` display array, or ``None`` if unsupported."""
    try:
        arr = np.asarray(arr)
    except Exception:
        return None
    if arr.ndim not in (2, 3) or arr.size == 0:
        return None
    try:
        kind = arr.dtype.kind
        if kind == "f":
            # Float data: assume a display-normalized [0, 1] array.
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        elif kind == "b":
            arr = arr.astype(np.uint8) * 255
        elif kind == "u" and arr.dtype == np.uint8:
            arr = arr
        elif kind in ("u", "i"):
            # Other integer data: clip to the 8-bit display range.  Values
            # beyond 255 saturate to white — acceptable for this display-only
            # MVP (stretch controls are a later milestone).
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            return None
    except Exception:
        return None
    return arr


def _array_to_qimage(np: Any, arr: Any) -> Optional[QImage]:
    """Convert a 2D/3D array to a copied ``QImage`` (channels-last RGB/RGBA)."""
    arr = _to_uint8(np, arr)
    if arr is None:
        return None
    try:
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 2:
            height, width = arr.shape
            fmt = QImage.Format.Format_Grayscale8
            bytes_per_line = width
        elif arr.shape[2] == 3:
            height, width = arr.shape[0], arr.shape[1]
            fmt = QImage.Format.Format_RGB888
            bytes_per_line = width * 3
        elif arr.shape[2] == 4:
            height, width = arr.shape[0], arr.shape[1]
            fmt = QImage.Format.Format_RGBA8888
            bytes_per_line = width * 4
        else:
            return None
        image = QImage(arr.tobytes(), width, height, bytes_per_line, fmt)
    except Exception:
        return None
    # Deep copy so no reference to the temporary buffer outlives this call.
    return image.copy()


def render_preview_image(data: Any) -> Optional[QImage]:
    """Render image-like preview ``data`` into a copied ``QImage`` (or ``None``).

    This is the single public entry point of the renderer.  It never raises:
    missing, malformed, or unsupported data returns ``None`` so the caller can
    fall back to a metadata-only update (or clear a previous image).
    """
    np = _load_numpy()
    if np is None:
        return None
    for arr in _iter_array_candidates(np, data):
        image = _array_to_qimage(np, arr)
        if image is not None:
            return image
    return None
