"""Qt preview display-adjustment helpers (M10/M13) — strictly display-only.

White-balance (per-channel R/G/B gain), the display stretch (linear / asinh /
log / auto), black/white points, gamma, brightness/contrast/saturation and a
simple display histogram for the Preview surface.  Like the rest of the Qt
shell this module imports only PySide6 Qt classes at module import time and
imports numpy *lazily*, inside the conversion path only, so a fresh
``import seestar.gui_qt`` never pulls numpy (or Tk / the engine) into
``sys.modules``.

Every helper operates on a *copy* of the input :class:`QImage` and never
mutates it, so the stored original display image (``MainWindow._preview_source``)
stays pristine for later recomputation.  The adjustment math (WB, stretch,
gamma, brightness/contrast/saturation and auto-WB) reproduces the Tk GUI's
``PreviewManager.process_image`` pipeline (``seestar/gui/preview.py``) and its
``seestar.tools.stretch`` tone curves / auto-WB, reimplemented here in pure
numpy so the Qt shell never imports the engine or the Tk tooling.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtGui import QColor, QImage, QPainter, QPixmap

# Display-stretch modes (UI vocabulary, backend-agnostic).  ``auto`` is a
# Qt-only convenience mode (Tk has an "Auto Stretch" *button* that computes
# black/white points, not a stretch mode); the three Tk modes are linear /
# asinh / log.
STRETCH_MODES = ("linear", "asinh", "log", "auto")

# Defaults aligned with the Tk GUI (``seestar/gui/main_window.py``):
# stretch "Asinh", black point 0.01, white point 0.99, gamma 1.0, WB 1/1/1,
# brightness/contrast/saturation 1.0.
DEFAULT_STRETCH = "asinh"
DEFAULT_BLACK_POINT = 0.01
DEFAULT_WHITE_POINT = 0.99
DEFAULT_GAMMA = 1.0

# Neutral white-balance gains (R, G, B).  1.0 = no change.
NEUTRAL_WB = (1.0, 1.0, 1.0)
DEFAULT_WB = NEUTRAL_WB

DEFAULT_BRIGHTNESS = 1.0
DEFAULT_CONTRAST = 1.0
DEFAULT_SATURATION = 1.0

# Slider ranges / steps, taken from the Tk ``_create_slider_spinbox_group``
# calls in ``seestar/gui/main_window.py`` (they must match Tk exactly).
WB_MIN, WB_MAX, WB_STEP = 0.1, 5.0, 0.01
BLACK_POINT_MIN, BLACK_POINT_MAX, BLACK_POINT_STEP = 0.0, 1.0, 0.001
WHITE_POINT_MIN, WHITE_POINT_MAX, WHITE_POINT_STEP = 0.0, 1.0, 0.001
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP = 0.1, 5.0, 0.01
BRIGHTNESS_MIN, BRIGHTNESS_MAX, BRIGHTNESS_STEP = 0.1, 3.0, 0.01
CONTRAST_MIN, CONTRAST_MAX, CONTRAST_STEP = 0.1, 3.0, 0.01
SATURATION_MIN, SATURATION_MAX, SATURATION_STEP = 0.0, 3.0, 0.01

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


# --------------------------------------------------------------------------
# Tk-parity tone-curve / colour math (pure numpy, float [0, 1] in/out).
# These reproduce ``seestar/tools/stretch.py`` exactly for the paths the Tk
# preview uses, so the Qt display matches the Tk display.
# --------------------------------------------------------------------------
def _apply_wb(np: Any, f, wb):
    """Multiply RGB channels by ``(r, g, b)`` gains (``ColorCorrection.white_balance``)."""
    if f.ndim != 3 or f.shape[2] < 3:
        return f
    r, g, b = (float(wb[0]), float(wb[1]), float(wb[2]))
    out = f.copy()
    out[..., 0] = np.clip(f[..., 0] * r, 0.0, 1.0)
    out[..., 1] = np.clip(f[..., 1] * g, 0.0, 1.0)
    out[..., 2] = np.clip(f[..., 2] * b, 0.0, 1.0)
    return out


def _stretch_linear(np: Any, f, black_point, white_point):
    white_point = max(float(white_point), float(black_point) + 1e-6)
    black_point = float(black_point)
    return np.clip((f - black_point) / (white_point - black_point), 0.0, 1.0)


def _stretch_log(np: Any, f, scale, black_point):
    data = np.nan_to_num(f)
    shifted = data - float(black_point)
    clipped = np.maximum(shifted, 1e-10)
    max_val = float(np.nanmax(clipped))
    if max_val <= 0:
        return np.zeros_like(data)
    denom = np.log1p(float(scale) * max_val)
    if denom < 1e-10:
        return np.zeros_like(data)
    return np.clip(np.log1p(float(scale) * clipped) / denom, 0.0, 1.0)


def _stretch_asinh(np: Any, f, scale, black_point):
    data = np.nan_to_num(f)
    shifted = data - float(black_point)
    clipped = np.maximum(shifted, 0.0)
    max_val = float(np.nanmax(clipped))
    if max_val <= 0:
        return np.zeros_like(data)
    denom = np.arcsinh(float(scale) * max_val)
    if denom < 1e-10:
        return np.zeros_like(data)
    return np.clip(np.arcsinh(float(scale) * clipped) / denom, 0.0, 1.0)


def _apply_stretch(np: Any, f, mode: str, black_point: float, white_point: float):
    """Apply the Tk display-stretch tone curve to float ``[0, 1]`` data."""
    f = np.clip(f, 0.0, 1.0)
    if mode == "linear":
        return _stretch_linear(np, f, black_point, white_point)
    if mode == "log":
        return _stretch_log(np, f, 10.0, black_point)
    if mode == "asinh":
        scale = (
            10.0 / max(0.01, white_point - black_point)
            if white_point > black_point
            else 10.0
        )
        return _stretch_asinh(np, f, scale, black_point)
    if mode == "auto":
        mn = float(np.min(f))
        mx = float(np.max(f))
        if mx - mn <= 1e-12:
            return f
        return np.clip((f - mn) / (mx - mn), 0.0, 1.0)
    # Unknown mode degrades to linear (never raises).
    return f


def _apply_gamma(np: Any, f, gamma):
    gamma = float(gamma)
    if abs(gamma - 1.0) < 1e-6:
        return f
    corrected = np.power(np.maximum(f, 1e-10), gamma)
    return np.clip(corrected, 0.0, 1.0)


def _apply_brightness_contrast_saturation(
    np: Any, f, brightness, contrast, saturation
):
    """Apply the Tk image-adjustment enhancements (brightness/contrast/saturation).

    Reproduces the Tk ``process_image`` use of the image-enhancement library:
    brightness multiplies, contrast blends toward the mean of the first band
    (as a scalar applied to every channel), and saturation blends each channel
    toward the luma.  Grayscale data skips the saturation step (the Tk path
    only applies saturation to RGB images).
    """
    if abs(float(brightness) - 1.0) > 1e-3:
        f = f * float(brightness)
    if abs(float(contrast) - 1.0) > 1e-3:
        if f.ndim == 3 and f.shape[2] >= 3:
            mean = float(np.mean(f[..., 0]))
        else:
            mean = float(np.mean(f))
        f = mean * (1.0 - float(contrast)) + f * float(contrast)
    if f.ndim == 3 and f.shape[2] >= 3 and abs(float(saturation) - 1.0) > 1e-3:
        luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        f = luma[..., None] * (1.0 - float(saturation)) + f * float(saturation)
    return f


def apply_preview_adjustments(
    source: Optional[QImage],
    *,
    wb=NEUTRAL_WB,
    stretch: str = DEFAULT_STRETCH,
    black_point: float = DEFAULT_BLACK_POINT,
    white_point: float = DEFAULT_WHITE_POINT,
    gamma: float = DEFAULT_GAMMA,
    brightness: float = DEFAULT_BRIGHTNESS,
    contrast: float = DEFAULT_CONTRAST,
    saturation: float = DEFAULT_SATURATION,
) -> Optional[QImage]:
    """Return a fully adjusted copy of ``source`` (or ``None``), display-only.

    The pipeline order matches the Tk ``PreviewManager.process_image``:
    white balance -> stretch -> gamma -> brightness/contrast/saturation.  The
    input ``QImage`` is never mutated.  A full identity setting (neutral WB,
    linear stretch with 0/1 black/white, unit gamma and unit B/C/S) returns a
    plain byte-identical copy so the pre-M10 default display behaviour is
    preserved when the user selects exactly that state.
    """
    if source is None or source.isNull():
        return None
    wb = tuple(float(x) for x in wb)
    black_point = float(black_point)
    white_point = float(white_point)
    gamma = float(gamma)
    brightness = float(brightness)
    contrast = float(contrast)
    saturation = float(saturation)

    identity = (
        wb == NEUTRAL_WB
        and stretch == "linear"
        and abs(black_point) < 1e-9
        and abs(white_point - 1.0) < 1e-9
        and abs(gamma - 1.0) < 1e-6
        and abs(brightness - 1.0) < 1e-3
        and abs(contrast - 1.0) < 1e-3
        and abs(saturation - 1.0) < 1e-3
    )
    if identity:
        return source.copy()

    np = _load_numpy()
    if np is None:
        return source.copy()
    arr = _image_to_array(np, source)
    if arr is None:
        return source.copy()

    f = arr.astype(np.float64) / 255.0
    f = _apply_wb(np, f, wb)
    f = _apply_stretch(np, f, stretch, black_point, white_point)
    f = np.clip(f, 0.0, 1.0)
    f = _apply_gamma(np, f, gamma)
    f = np.clip(f, 0.0, 1.0)
    f = _apply_brightness_contrast_saturation(np, f, brightness, contrast, saturation)
    out = np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)
    return _array_to_qimage(np, out)


def compute_auto_wb(image: Optional[QImage]) -> tuple:
    """Compute auto white-balance gains from a display image (Tk parity).

    Mirrors ``seestar.tools.stretch.apply_auto_white_balance``: for a
    3-channel image each channel's mode (from a 256-bin histogram over the
    [0.5, 99.5] percentile range) is equalised toward the green channel's mode,
    with the R/B gains clipped to [0.2, 5.0].  Non-colour, missing, or invalid
    data returns neutral ``(1.0, 1.0, 1.0)`` and never raises.
    """
    np = _load_numpy()
    if np is None:
        return NEUTRAL_WB
    arr = _image_to_array(np, image)
    if arr is None or arr.ndim != 3 or arr.shape[2] < 3:
        return NEUTRAL_WB

    f = arr.astype(np.float64) / 255.0
    modes = []
    for i in range(3):
        channel = f[..., i].ravel()
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            return NEUTRAL_WB
        lo, hi = np.percentile(finite, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1e-5
        hist, edges = np.histogram(finite, bins=256, range=(lo, hi))
        idx = int(np.argmax(hist))
        mode = (edges[idx] + edges[idx + 1]) / 2.0
        mode = max(mode, 1e-5)
        modes.append(mode)

    mode_r, mode_g, mode_b = modes
    gain_r = mode_g / mode_r if mode_r > 1e-9 else 1.0
    gain_g = 1.0
    gain_b = mode_g / mode_b if mode_b > 1e-9 else 1.0
    gain_r = float(np.clip(gain_r, 0.2, 5.0))
    gain_b = float(np.clip(gain_b, 0.2, 5.0))
    return (gain_r, gain_g, gain_b)


# --------------------------------------------------------------------------
# Display histogram (unchanged M10 surface).
# --------------------------------------------------------------------------
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
