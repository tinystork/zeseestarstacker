"""Qt preview display-adjustment helpers (M10) — strictly display-only.

White-balance (per-channel R/G/B gain), display stretch (linear / asinh / log /
auto) and a simple display histogram for the Preview surface.  Like the rest of
the Qt shell this module imports only PySide6 Qt classes at module import time
and imports numpy *lazily*, inside the conversion path only, so a fresh
``import seestar.gui_qt`` never pulls numpy (or Tk / the engine) into
``sys.modules``.

Every helper operates on a *copy* of the input :class:`QImage` and never
mutates it, so the stored original display image (``MainWindow._preview_source``)
stays pristine for later recomputation.  Neutral settings (WB == ``(1.0, 1.0,
1.0)`` and stretch == ``"linear"``) return a plain copy of the source that is
byte-identical to the unadjusted render, preserving the M5/M8 display behaviour
exactly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtGui import QColor, QImage, QPainter, QPixmap

# Display-stretch modes (UI vocabulary, backend-agnostic).
STRETCH_MODES = ("linear", "asinh", "log", "auto")
DEFAULT_STRETCH = "linear"

# Neutral white-balance gains (R, G, B).  1.0 = no change.
NEUTRAL_WB = (1.0, 1.0, 1.0)
DEFAULT_WB = NEUTRAL_WB

# Channel-name -> histogram bar colour for the display histogram pixmap.
_HISTOGRAM_COLORS = {
    "L": QColor(225, 225, 225),
    "R": QColor(225, 70, 70),
    "G": QColor(70, 200, 70),
    "B": QColor(70, 120, 225),
}


def _load_numpy():
    """Lazily import numpy (the module object, or ``None`` when unavailable)."""
    try:
        import importlib

        return importlib.import_module("numpy")
    except Exception:
        return None


def _image_to_array(np: Any, image: Optional[QImage]):
    """Return a uint8 array (H,W) or (H,W,3/4) for ``image``, or ``None``.

    Handles the exact formats emitted by ``preview_render`` (Grayscale8 /
    RGB888 / RGBA8888) and falls back to RGB888 for any other format.  Scanline
    padding (4-byte alignment) is accounted for via ``bytesPerLine``.
    """
    if image is None or image.isNull():
        return None
    fmt = image.format()
    if fmt == QImage.Format.Format_Grayscale8:
        channels = 1
    elif fmt == QImage.Format.Format_RGB888:
        channels = 3
    elif fmt == QImage.Format.Format_RGBA8888:
        channels = 4
    else:
        image = image.convertToFormat(QImage.Format.Format_RGB888)
        channels = 3
    height = image.height()
    width = image.width()
    if height <= 0 or width <= 0:
        return None
    bytes_per_line = image.bytesPerLine()
    try:
        raw = np.frombuffer(bytes(image.bits()), dtype=np.uint8)
    except Exception:
        return None
    expected = height * bytes_per_line
    if raw.size < expected:
        return None
    raw = raw[:expected]
    width_bytes = width * channels
    if bytes_per_line == width_bytes:
        if channels == 1:
            return raw.reshape(height, width)
        return raw.reshape(height, width, channels)
    # Padded scanlines: drop the padding tail of each row.
    try:
        raw2d = raw.reshape(height, bytes_per_line)[:, :width_bytes]
    except Exception:
        return None
    if channels == 1:
        return raw2d.reshape(height, width)
    return raw2d.reshape(height, width, channels)


def _array_to_qimage(np: Any, arr) -> Optional[QImage]:
    """Convert a uint8 2D/3D array to a copied ``QImage`` (Grayscale/RGB/RGBA)."""
    try:
        arr = np.ascontiguousarray(arr)
    except Exception:
        return None
    if arr.ndim == 2:
        height, width = arr.shape
        image = QImage(arr.tobytes(), width, height, width, QImage.Format.Format_Grayscale8)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        height, width = arr.shape[0], arr.shape[1]
        image = QImage(arr.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        height, width = arr.shape[0], arr.shape[1]
        image = QImage(arr.tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888)
    else:
        return None
    return image.copy()


def _apply_wb(np: Any, f, wb):
    """Multiply RGB channels by ``(r, g, b)`` gains; grayscale/alpha untouched."""
    if f.ndim != 3 or f.shape[2] < 3:
        return f
    r, g, b = (float(wb[0]), float(wb[1]), float(wb[2]))
    out = f.copy()
    out[..., 0] = np.clip(f[..., 0] * r, 0.0, 1.0)
    out[..., 1] = np.clip(f[..., 1] * g, 0.0, 1.0)
    out[..., 2] = np.clip(f[..., 2] * b, 0.0, 1.0)
    return out


def _apply_stretch(np: Any, f, mode: str):
    """Apply a monotonic display-stretch tone curve to float ``[0, 1]`` data."""
    f = np.clip(f, 0.0, 1.0)
    if mode == "linear":
        return f
    if mode == "log":
        return np.log1p(f) / np.log(2.0)
    if mode == "asinh":
        k = 10.0
        return np.arcsinh(f * k) / np.arcsinh(k)
    if mode == "auto":
        mn = float(np.min(f))
        mx = float(np.max(f))
        if mx - mn <= 1e-12:
            return f
        return np.clip((f - mn) / (mx - mn), 0.0, 1.0)
    # Unknown mode degrades to linear (never raises).
    return f


def apply_wb_stretch(
    source: Optional[QImage],
    wb=NEUTRAL_WB,
    stretch: str = DEFAULT_STRETCH,
) -> Optional[QImage]:
    """Return a WB + stretch transformed copy of ``source`` (or ``None``).

    Neutral settings return a plain copy (byte-identical to the source) so the
    default preview behaviour matches the pre-M10 render exactly.  The source
    image is never mutated.
    """
    if source is None or source.isNull():
        return None
    wb = tuple(float(x) for x in wb)
    if stretch == DEFAULT_STRETCH and wb == NEUTRAL_WB:
        return source.copy()
    np = _load_numpy()
    if np is None:
        return source.copy()
    arr = _image_to_array(np, source)
    if arr is None:
        return source.copy()
    f = arr.astype(np.float64) / 255.0
    f = _apply_wb(np, f, wb)
    f = _apply_stretch(np, f, stretch)
    out = np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)
    return _array_to_qimage(np, out)


def compute_histogram(image: Optional[QImage], bins: int = 256) -> Optional[Dict[str, Any]]:
    """Return ``{channel: int64 histogram}`` for the display image, or ``None``.

    Grayscale images yield a single ``"L"`` channel; RGB/RGBA yield ``"R"`` /
    ``"G"`` / ``"B"``.  Non-image/malformed input returns ``None`` (no crash).
    """
    np = _load_numpy()
    if np is None:
        return None
    arr = _image_to_array(np, image)
    if arr is None:
        return None
    if arr.ndim == 2:
        planes = {"L": arr}
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        planes = {"R": arr[..., 0], "G": arr[..., 1], "B": arr[..., 2]}
    else:
        return None
    bins = max(1, int(bins))
    result: Dict[str, Any] = {}
    for name, plane in planes.items():
        hist, _ = np.histogram(plane, bins=bins, range=(0, 256))
        result[name] = hist.astype(np.int64)
    return result


def compute_histogram_stats(image: Optional[QImage]) -> Optional[str]:
    """Return a compact min–max summary string for ``image``, or ``None``."""
    np = _load_numpy()
    if np is None:
        return None
    arr = _image_to_array(np, image)
    if arr is None:
        return None
    if arr.ndim == 2:
        return f"L {int(arr.min())}–{int(arr.max())}"
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        return (
            f"R {int(r.min())}–{int(r.max())} · "
            f"G {int(g.min())}–{int(g.max())} · "
            f"B {int(b.min())}–{int(b.max())}"
        )
    return None


def render_histogram_pixmap(
    image: Optional[QImage],
    width: int = 256,
    height: int = 64,
) -> Optional[QPixmap]:
    """Draw a small per-channel histogram bar chart, or ``None`` if not renderable."""
    width = max(1, int(width))
    height = max(1, int(height))
    hist = compute_histogram(image, bins=width)
    if not hist:
        return None
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(22, 22, 24))
    painter = QPainter(pixmap)
    try:
        max_count = max(int(h.max()) for h in hist.values())
        if max_count <= 0:
            max_count = 1
        for name, h in hist.items():
            color = _HISTOGRAM_COLORS.get(name, QColor(225, 225, 225))
            painter.setPen(color)
            n = len(h)
            for i in range(n):
                bar = int(round(float(h[i]) / max_count * (height - 1)))
                if bar <= 0:
                    continue
                x = i * width // n
                painter.drawLine(x, height - 1, x, height - 1 - bar)
    finally:
        painter.end()
    return pixmap
