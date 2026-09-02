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

**Unit contract (PHI-R3.1).**  Two display paths exist:

* the *legacy QImage chain* (``apply_preview_adjustments`` /
  ``apply_preview_wb``, single-array payloads) keeps the historical Tk-parity
  ``[0, 1]`` display-level semantics bit-identical;
* the *Option-A float display path* (:func:`render_analysis_display`) renders
  the visible preview from the preserved float analysis/WB source: the
  black/white points are **analysis units** over ``[0, upper]``,
  ``upper = max(1.0, finite max)`` — a white point above ``1`` is a
  first-class value that recovers preserved headroom — and the final
  ``uint8``/``QImage`` conversion is the only clipping boundary of the
  display path.  The BP/WP validation seams (:func:`normalize_bp_wp` /
  :func:`quantize_bp_wp` / :func:`clamp_bp_wp_edit`) take an explicit
  ``max_value`` domain so the same deterministic code serves both paths.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

from PySide6.QtGui import QImage

from seestar.utils.phi_trace import phi_trace_enabled, phi_trace_stage

logger = logging.getLogger(__name__)

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

# Deterministic BP/WP minimum separation: exactly one black/white slider+spin
# step, so the enforced gap is always representable by the control resolution
# (BLACK_POINT_STEP == WHITE_POINT_STEP == 0.001).  This is the single
# authoritative separation used by MainWindow, the stretch sliders/spins and
# the histogram BP/WP handles.
BP_WP_MIN_SEPARATION = BLACK_POINT_STEP

# Control-domain maximum used by the legacy [0, 1] display controls.  The
# Option-A float path re-scopes the same seams to the preserved analysis
# domain upper ``max(1.0, finite max)`` (PHI-R3.1) by passing ``max_value``;
# the legacy QImage path never does, so its historical [0, 1] semantics are
# bit-identical.
BP_WP_LEGACY_MAX = 1.0


def normalize_bp_wp(bp, wp, min_separation=BP_WP_MIN_SEPARATION, max_value=BP_WP_LEGACY_MAX):
    """Normalize a black/white point pair to a valid ``0 <= BP < WP <= max_value`` pair.

    ``max_value`` is the current control domain upper bound: ``1.0`` for the
    legacy QImage display path (historical semantics), or the preserved
    analysis range upper (``max(1.0, finite max)``, PHI-R3.1) for Option-A
    float previews so a white point above ``1`` is a first-class value.
    Non-finite or non-numeric inputs fall back to the deterministic neutral
    defaults.  Finite values are clamped to ``[0, max_value]`` and, when they
    overlap or invert, separated by at least ``min_separation`` with the black
    point taking priority (the white point is pushed up, or the black point
    pulled down when the white point would overflow ``max_value``).  Always
    returns a valid pair and never raises.
    """
    try:
        bp = float(bp)
        wp = float(wp)
    except (TypeError, ValueError):
        return DEFAULT_BLACK_POINT, DEFAULT_WHITE_POINT
    if not (math.isfinite(bp) and math.isfinite(wp)):
        return DEFAULT_BLACK_POINT, DEFAULT_WHITE_POINT
    max_value = float(max_value)
    if not (math.isfinite(max_value) and max_value > 0.0):
        max_value = BP_WP_LEGACY_MAX
    bp = min(max(bp, 0.0), max_value)
    wp = min(max(wp, 0.0), max_value)
    if wp - bp < min_separation:
        wp = min(max_value, bp + min_separation)
        if wp - bp < min_separation:
            bp = max(0.0, wp - min_separation)
    return bp, wp


def quantize_bp_wp(value, step=BP_WP_MIN_SEPARATION, max_value=BP_WP_LEGACY_MAX):
    """Snap a single BP/WP value to the shared control-resolution grid.

    Returns the nearest ``k * step`` grid point (``k`` an integer) expressed as
    ``k / factor`` where ``factor = round(1.0 / step)`` — i.e. the *exact*
    double the stretch spinbox stores for its ``decimals=3`` resolution, so a
    value quantized here round-trips through ``QDoubleSpinBox`` bit-identically
    (grid points above ``1`` — analysis-domain white points, PHI-R3.1 — stay
    on the same grid).

    Non-finite or non-numeric input falls back to the neutral black-point
    default and every result is clamped to ``[0, max_value]``.  This is the
    single quantization seam shared by the histogram handle drag and the
    MainWindow stretch controls so all surfaces agree *during* a live drag,
    not just on release.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_BLACK_POINT
    if not math.isfinite(value):
        return DEFAULT_BLACK_POINT
    max_value = float(max_value)
    if not (math.isfinite(max_value) and max_value > 0.0):
        max_value = BP_WP_LEGACY_MAX
    value = min(max(value, 0.0), max_value)
    factor = int(round(1.0 / step))
    return round(value * factor) / factor


def clamp_bp_wp_edit(driver, value, other, min_separation=BP_WP_MIN_SEPARATION, max_value=BP_WP_LEGACY_MAX):
    """Return the valid ``(bp, wp)`` pair after editing one endpoint to ``value``.

    ``driver`` is ``"bp"`` or ``"wp"``; ``other`` is the current value of the
    un-edited endpoint.  The edited endpoint is clamped into ``[0, max_value]``
    (``max_value`` = current control-domain upper, PHI-R3.1) and, if it would
    cross the other endpoint, clamped to preserve the deterministic
    ``min_separation`` gap (the un-edited endpoint is never moved).  Non-finite
    input falls back to the neutral defaults.  Always returns a valid pair and
    never raises.
    """
    try:
        value = float(value)
        other = float(other)
    except (TypeError, ValueError):
        return DEFAULT_BLACK_POINT, DEFAULT_WHITE_POINT
    if not (math.isfinite(value) and math.isfinite(other)):
        return DEFAULT_BLACK_POINT, DEFAULT_WHITE_POINT
    max_value = float(max_value)
    if not (math.isfinite(max_value) and max_value > 0.0):
        max_value = BP_WP_LEGACY_MAX
    other = min(max(other, 0.0), max_value)
    if driver == "bp":
        bp = min(max(value, 0.0), max_value)
        wp = min(other, max_value)
        if bp >= wp - min_separation:
            bp = max(0.0, wp - min_separation)
    else:
        wp = min(max(value, 0.0), max_value)
        bp = min(other, max_value)
        if wp <= bp + min_separation:
            wp = min(max_value, bp + min_separation)
    # Final invariant enforcement (also guards pathological ``other`` values).
    return normalize_bp_wp(bp, wp, min_separation, max_value)


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
    if phi_trace_enabled():
        # PHI-R2: this is the QImage-derived DISPLAY buffer (uint8 -> float
        # [0,1]) entering the display tone chain — it is NOT the analysis
        # ``_wb_only_float`` diagnostic buffer (that one is traced as
        # ``wb_only`` in main_window).  Relabelled ``display_input`` so the
        # stage attribution is unambiguous.
        phi_trace_stage(
            logger,
            route="qt",
            stage="display_input",
            arr=f,
            stretch=stretch,
            bp=f"{black_point:g}",
            wp=f"{white_point:g}",
        )
    f = _apply_wb(np, f, wb)
    f = _apply_stretch(np, f, stretch, black_point, white_point)
    f = np.clip(f, 0.0, 1.0)
    f = _apply_gamma(np, f, gamma)
    f = np.clip(f, 0.0, 1.0)
    f = _apply_brightness_contrast_saturation(np, f, brightness, contrast, saturation)
    out = np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)
    if phi_trace_enabled():
        # Final display-stage output (uint8 [0,255]) — display buffer, not an
        # analysis buffer.  ``one_n`` for uint8 counts the == 255 saturated
        # pixels (see ``phi_trace._stage_stats`` dtype-aware counters).
        phi_trace_stage(
            logger,
            route="qt",
            stage="display_output",
            arr=out,
            stretch=stretch,
        )
    return _array_to_qimage(np, out)


def apply_preview_wb(
    source: Optional[QImage],
    *,
    wb=NEUTRAL_WB,
) -> Optional[QImage]:
    """Return a WB-only copy of ``source`` (white balance applied, no stretch).

    The Tk GUI computes its preview histogram from the *WB-only* image
    (``PreviewManager.image_data_wb``), i.e. after white balance but *before*
    the display stretch / gamma / brightness-contrast-saturation.  This helper
    reproduces exactly that intermediate image so the Qt histogram can share
    the Tk data source (M14 histogram-source alignment).  Grayscale data is
    unaffected by white balance, so it returns a plain copy.  The input image
    is never mutated.
    """
    if source is None or source.isNull():
        return None
    wb = tuple(float(x) for x in wb)
    if wb == NEUTRAL_WB:
        return source.copy()
    np = _load_numpy()
    if np is None:
        return source.copy()
    arr = _image_to_array(np, source)
    if arr is None:
        return source.copy()
    f = arr.astype(np.float64) / 255.0
    f = _apply_wb(np, f, wb)
    f = np.clip(f, 0.0, 1.0)
    out = np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)
    return _array_to_qimage(np, out)


# --------------------------------------------------------------------------
# PHI-R3.1 — Option-A float display path (analysis-unit tone curves)
#
# Unit contract: for Option-A previews the *visible display* is rendered from
# the preserved float analysis/WB source (never from a pre-quantized uint8
# copy), with the user-selected black/white points expressed in the same
# analysis units the float histogram shows (``[0, upper]``,
# ``upper = max(1.0, finite max)`` — white points above ``1`` are first-class
# values that recover preserved headroom).  The black point is the analysis
# value mapped to display black (0), the white point the analysis value mapped
# to display white (1); values above the white point saturate — the final
# uint8/QImage conversion is the only clipping boundary of the display path.
# The legacy QImage chain (``apply_preview_adjustments``) keeps its historical
# [0, 1] Tk-parity semantics unchanged for single-array payloads.
# --------------------------------------------------------------------------

# Fixed gain of the analysis-domain asinh/log curves (matches the legacy
# ``10 / (wp - bp)`` scale evaluated at the full bp..wp window: the window
# maps into the ``0..1`` display range with the same asymptotic character).
_ANALYSIS_STRETCH_GAIN = 10.0


def _analysis_stretch(np: Any, f, mode: str, black_point: float, white_point: float):
    """Fixed-reference analysis-unit stretch of the float display path.

    ``f`` is the WB-only analysis buffer (finite, ``>= 0``, may exceed ``1`` =
    preserved headroom).  ``black_point``/``white_point`` are analysis units:
    ``shift = max(f - bp, 0)`` and ``win = wp - bp``, then:

    * ``linear`` — ``clip(shift / win, 0, 1)``;
    * ``asinh``  — ``clip(asinh(G*shift/win) / asinh(G), 0, 1)``;
    * ``log``    — ``clip(log1p(G*shift/win) / log1p(G), 0, 1)``;
    * ``auto``   — min/max fill-range normalisation over the finite buffer
      (marker-agnostic convenience mode, unchanged character);

    with ``G = _ANALYSIS_STRETCH_GAIN``.  ``bp`` maps to display black, ``wp``
    to display white, everything above ``wp`` saturates to white — the *fixed
    reference* is the white point (unlike the legacy adaptive chain which
    normalises by the data maximum), so raising the white point above ``1``
    visibly recovers preserved headroom instead of leaving it white-clipped.
    Returns a fresh array; the input is never mutated.
    """
    bp = float(black_point)
    wp = float(white_point)
    win = wp - bp
    if win <= 0.0:
        win = BP_WP_MIN_SEPARATION
    if mode == "auto":
        mn = float(np.min(f))
        mx = float(np.max(f))
        if mx - mn <= 1e-12:
            return f
        return np.clip((f - mn) / (mx - mn), 0.0, 1.0)
    shift = f - bp
    if mode == "linear":
        return np.clip(shift / win, 0.0, 1.0)
    if mode == "asinh":
        s = np.maximum(shift, 0.0)
        return np.clip(
            np.arcsinh(_ANALYSIS_STRETCH_GAIN * s / win)
            / np.arcsinh(_ANALYSIS_STRETCH_GAIN),
            0.0,
            1.0,
        )
    if mode == "log":
        s = np.maximum(shift, 0.0)
        return np.clip(
            np.log1p(_ANALYSIS_STRETCH_GAIN * s / win)
            / np.log1p(_ANALYSIS_STRETCH_GAIN),
            0.0,
            1.0,
        )
    # Unknown mode degrades to linear (never raises).
    return np.clip(shift / win, 0.0, 1.0)


def render_analysis_display(
    analysis,
    *,
    stretch: str = DEFAULT_STRETCH,
    black_point: float = DEFAULT_BLACK_POINT,
    white_point: float = DEFAULT_WHITE_POINT,
    gamma: float = DEFAULT_GAMMA,
    brightness: float = DEFAULT_BRIGHTNESS,
    contrast: float = DEFAULT_CONTRAST,
    saturation: float = DEFAULT_SATURATION,
) -> Optional[QImage]:
    """Render the Option-A float display from an analysis buffer (PHI-R3.1).

    The visible display is derived from the preserved float analysis/WB
    source — the user-selected stretch (black/white points in analysis units,
    possibly above ``1``), gamma and brightness/contrast/saturation are
    applied in float **before** the final ``uint8``/``QImage`` conversion,
    which is the only clipping boundary of the display path.  A white point
    above ``1`` visibly recovers preserved headroom instead of leaving it
    white-clipped (contract: no premature clamp at anchor mapping / WB /
    QImage ingest).

    Input ``analysis`` is the WB-only float buffer (finite, non-negative,
    headroom allowed).  Non-finite values are floored to ``0`` defensively.
    Returns a deep-copied ``QImage`` (uint8) or ``None`` on unusable input;
    never raises; never mutates the input.
    """
    np = _load_numpy()
    if np is None:
        return None
    try:
        arr = np.asarray(analysis, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim not in (2, 3) or arr.size == 0:
        return None
    f = np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
    black_point = float(black_point)
    white_point = float(white_point)
    gamma = float(gamma)
    brightness = float(brightness)
    contrast = float(contrast)
    saturation = float(saturation)
    if phi_trace_enabled():
        # Option-A display chain input: the preserved float analysis/WB buffer
        # (may carry headroom > 1 — this is analysis data entering the tone
        # chain, NOT a pre-quantized uint8-derived [0,1] buffer).
        phi_trace_stage(
            logger,
            route="qt",
            stage="display_input",
            arr=f,
            stretch=stretch,
            bp=f"{black_point:g}",
            wp=f"{white_point:g}",
        )
    f = _analysis_stretch(np, f, stretch, black_point, white_point)
    f = np.clip(f, 0.0, 1.0)
    f = _apply_gamma(np, f, gamma)
    f = np.clip(f, 0.0, 1.0)
    f = _apply_brightness_contrast_saturation(
        np, f, brightness, contrast, saturation
    )
    out = np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)
    if phi_trace_enabled():
        # Final display-stage output (uint8 [0,255]) — bounded screen domain.
        phi_trace_stage(
            logger,
            route="qt",
            stage="display_output",
            arr=out,
            stretch=stretch,
        )
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


def compute_auto_stretch(image: Optional[QImage]) -> tuple:
    """Compute auto-stretch black/white points (0-1) from a WB-only image.

    Mirrors ``seestar.tools.stretch.apply_auto_stretch`` plus the normalisation
    performed in ``SeestarStackerGUI.apply_auto_stretch``: the luminance is
    built from the (WB-only) image, the black/white points are the 1st / 99th
    percentiles of the finite luminance, then both are mapped through the
    full-image min/max into the ``[0, 1]`` UI scale the stretch sliders use
    (with a ``1e-4`` minimum separation).  Missing/non-image/too-small input
    returns the neutral defaults and never raises.
    """
    np = _load_numpy()
    if np is None:
        return (DEFAULT_BLACK_POINT, DEFAULT_WHITE_POINT)
    arr = _image_to_array(np, image)
    if arr is None:
        return (DEFAULT_BLACK_POINT, DEFAULT_WHITE_POINT)
    f = arr.astype(np.float64) / 255.0
    if f.ndim == 3 and f.shape[2] >= 3:
        luminance = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    elif f.ndim == 2:
        luminance = f
    else:
        return (DEFAULT_BLACK_POINT, DEFAULT_WHITE_POINT)

    finite_lum = luminance[np.isfinite(luminance)]
    if finite_lum.size < 20:
        return (DEFAULT_BLACK_POINT, DEFAULT_WHITE_POINT)

    bp_calc = float(np.percentile(finite_lum, 1.0))
    wp_calc = float(np.percentile(finite_lum, 99.0))
    min_separation = 1e-4
    bp_calc = float(np.clip(bp_calc, 0.0, 1.0 - min_separation))
    wp_calc = float(np.clip(wp_calc, bp_calc + min_separation, 1.0))

    # Normalise through the full-image min/max (Tk maps the percentile values
    # into the UI 0-1 slider scale using the whole WB image's data range).
    min_data_val = float(np.nanmin(f))
    max_data_val = float(np.nanmax(f))
    range_data = max_data_val - min_data_val
    if range_data < 1e-9:
        range_data = 1.0
    bp_ui = float(np.clip((bp_calc - min_data_val) / range_data, 0.0, 1.0))
    wp_ui = float(np.clip((wp_calc - min_data_val) / range_data, 0.0, 1.0))
    if wp_ui <= bp_ui + 1e-4:
        wp_ui = min(1.0, bp_ui + 1e-4)
    if bp_ui >= wp_ui - 1e-4:
        bp_ui = max(0.0, wp_ui - 1e-4)
    return (bp_ui, wp_ui)


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


def compute_histogram_percentile(
    image: Optional[QImage], percentile: float = 99.5
) -> Optional[float]:
    """Return the ``percentile`` pixel level (0-1) of the image, or ``None``.

    Used by the interactive histogram's auto-zoom / zoom actions (Tk
    ``HistogramWidget.zoom_histogram`` takes the 99.5th percentile of the
    flattened pixel data as the zoomed right edge).  Computed over the finite
    pixels of every channel (grayscale included); empty/invalid input returns
    ``None`` and never raises.
    """
    np = _load_numpy()
    if np is None:
        return None
    arr = _image_to_array(np, image)
    if arr is None:
        return None
    f = arr.astype(np.float64) / 255.0
    finite = f[np.isfinite(f)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, float(percentile)))


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
