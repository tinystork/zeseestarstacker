"""Qt preview view transforms (M5 seam) — strictly display-only.

Zoom / rotation / fit helpers that operate on a copied display :class:`QImage`
and produce the :class:`QPixmap` actually shown in the Preview tab.  Like the
rest of the Qt shell, this module imports only PySide6 Qt classes and never
touches the Tk GUI or the scientific engine.

Rotation is expressed in *clockwise* degrees (multiples of 90, accumulated by
the caller and reduced modulo 360).  Zoom is one of the user-visible labels
``Fit`` / ``100%`` / ``200%`` / ``50%``:

* ``Fit`` scales the rotated image to fit ``target_size``, preserving aspect
  ratio (``Qt.AspectRatioMode.KeepAspectRatio`` + smooth transformation).
* percent labels scale the rotated image by the literal factor (1.0 / 2.0 /
  0.5) from the rotated image's native size, without ever mutating the source
  image.

M18 adds a *continuous* zoom path (``MIN_ZOOM``/``MAX_ZOOM``/``ZOOM_STEP``
mirrored from the Tk ``PreviewManager``) plus a pan offset: ``render_view``
accepts an optional ``zoom_factor`` + ``pan_offset`` so the shell can reproduce
the Tk mouse-wheel zoom and left-drag pan as a pure view transform layered on
top of the same render path.

The source :class:`QImage` is never mutated: every helper returns a fresh
image/pixmap, so the stored original display image stays pristine for
subsequent zoom/rotation recomputation.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPainter, QPixmap, QTransform

# User-visible zoom labels (order matches the ``zoom_combo`` items).
ZOOM_LABELS = ("Fit", "100%", "200%", "50%")

# Percent-zoom factor per label.  ``Fit`` is handled separately (aspect-fit).
ZOOM_FACTORS = {"100%": 1.0, "200%": 2.0, "50%": 0.5}

# Tk ``PreviewManager`` pan/zoom parity (M18): the continuous zoom-factor bounds
# and the multiplicative mouse-wheel step, mirrored 1:1 from
# ``seestar/gui/preview.py`` (``MAX_ZOOM`` / ``MIN_ZOOM`` / ``zoom_factor``).
MIN_ZOOM = 0.05
MAX_ZOOM = 15.0
ZOOM_STEP = 1.15


def normalized_rotation(degrees: int) -> int:
    """Return ``degrees`` reduced to the equivalent ``0``/``90``/``180``/``270``."""
    return int(degrees) % 360


def rotated_image(source: QImage, degrees: int) -> QImage:
    """Return a copy of ``source`` rotated clockwise by ``degrees``.

    Non-multiple-of-90 inputs are tolerated by reducing modulo 360 (the UI only
    ever produces multiples of 90).  A zero rotation returns a plain copy so
    callers can rely on a fresh image without depending on transform semantics.
    """
    degrees = normalized_rotation(degrees)
    if degrees == 0:
        return source.copy()
    return source.transformed(
        QTransform().rotate(degrees), Qt.TransformationMode.SmoothTransformation
    )


def percent_scaled_image(source: QImage, zoom: str) -> QImage:
    """Scale ``source`` by the percent-zoom factor for ``zoom``.

    ``100%`` returns the image unchanged; ``Fit`` and unknown labels fall back
    to native size so a bad label can never produce garbage.
    """
    factor = ZOOM_FACTORS.get(zoom, 1.0)
    if factor == 1.0:
        return source
    width = max(1, int(round(source.width() * factor)))
    height = max(1, int(round(source.height() * factor)))
    return source.scaled(
        width,
        height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def downsampled_image(source: QImage, factor: int) -> QImage:
    """Return ``source`` scaled down by ``factor`` (display-only, never mutates).

    This is the local, display-only seam for the Tk preview-resolution cycle
    button (``Res 1/1``..``Res 1/4``).  The Tk button drives the engine's
    ``preview_downsample_factor`` (a backend re-render); the Qt shell cannot
    touch the engine, so it approximates the effect by down-scaling the already
    adjusted display image.  ``factor <= 1`` (or a non-positive factor) returns
    the image unchanged; the source is never mutated (a fresh image is produced).
    """
    factor = int(factor)
    if factor <= 1 or source.isNull():
        return source
    width = max(1, int(round(source.width() / factor)))
    height = max(1, int(round(source.height() / factor)))
    return source.scaled(
        width,
        height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def fit_pixmap(source: QImage, target_size: QSize) -> QPixmap:
    """Scale ``source`` to fit ``target_size``, preserving aspect ratio.

    A null source or a degenerate target size returns the source at native size
    (so a not-yet-laid-out label can never produce a 0x0 pixmap).
    """
    pixmap = QPixmap.fromImage(source)
    if pixmap.isNull() or source.isNull():
        return pixmap
    if target_size is None or target_size.width() <= 0 or target_size.height() <= 0:
        return pixmap
    return pixmap.scaled(
        target_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def clamp_zoom_factor(factor: float) -> float:
    """Clamp a continuous zoom factor to the Tk ``[MIN_ZOOM, MAX_ZOOM]`` range."""
    return max(MIN_ZOOM, min(MAX_ZOOM, float(factor)))


def preset_label_for_factor(factor: float) -> Optional[str]:
    """Return the combo label matching ``factor`` exactly, else ``None``.

    Used to re-sync ``zoom_combo`` after a continuous wheel zoom: an exact
    preset match shows that preset, anything else shows a blank (custom) combo
    entry so the combo never lies about a non-preset zoom.
    """
    factor = float(factor)
    for label in ("100%", "200%", "50%"):
        if abs(factor - ZOOM_FACTORS[label]) < 1e-6:
            return label
    return None


def scaled_image(source: QImage, factor: float) -> QImage:
    """Scale ``source`` by a continuous ``factor`` (display-only, never mutates).

    ``factor == 1.0`` returns ``source`` unchanged (same contract as
    ``percent_scaled_image`` for the ``100%`` label); a non-positive factor also
    returns ``source`` unchanged so a degenerate input can never produce a 0x0
    image.  The source is never mutated (a fresh image is produced on scale).
    """
    if source.isNull():
        return source
    factor = float(factor)
    if factor <= 0 or abs(factor - 1.0) < 1e-9:
        return source
    width = max(1, int(round(source.width() * factor)))
    height = max(1, int(round(source.height() * factor)))
    return source.scaled(
        width,
        height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def compose_panned_pixmap(
    pixmap: QPixmap, target_size: QSize, pan_offset=(0.0, 0.0)
) -> QPixmap:
    """Draw ``pixmap`` centred + ``pan_offset`` into a ``target_size`` canvas.

    This is the Qt equivalent of the Tk ``_redraw_canvas`` image placement: the
    image is drawn centred on the viewport (``target_size``) plus the pan
    offset, and the canvas clips it.  A null pixmap or a degenerate
    ``target_size`` returns ``pixmap`` unchanged so a not-yet-laid-out label can
    never produce an empty view.
    """
    if pixmap.isNull():
        return pixmap
    if (
        target_size is None
        or target_size.width() <= 0
        or target_size.height() <= 0
    ):
        return pixmap
    canvas = QPixmap(target_size)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    try:
        ox, oy = pan_offset
        x = int(round((target_size.width() - pixmap.width()) / 2 + ox))
        y = int(round((target_size.height() - pixmap.height()) / 2 + oy))
        painter.drawPixmap(x, y, pixmap)
    finally:
        painter.end()
    return canvas


def zoomed_image_size(
    source: QImage, rotation: int, downsample_factor: int, zoom_factor: float
):
    """Return the ``(width, height)`` of the numeric-zoom scaled image.

    Mirrors the ``render_view`` numeric path (rotate → downsample → continuous
    scale) but returns the size without materialising the composite, so callers
    can render an accurate resolution label even while the view is panned.
    """
    if source is None or source.isNull():
        return 0, 0
    base = downsampled_image(rotated_image(source, rotation), downsample_factor)
    scaled = scaled_image(base, clamp_zoom_factor(zoom_factor))
    return scaled.width(), scaled.height()


def fit_scale(
    source: QImage, rotation: int, downsample_factor: int, target_size: QSize
) -> float:
    """Return the aspect-preserving fit scale for ``source`` into ``target_size``.

    Used when a wheel zoom exits "Fit" mode so the continuous factor continues
    from the current fit scale (Tk ``zoom_fit`` sets ``zoom_level`` to exactly
    this value).  Degenerate inputs return ``1.0``.
    """
    if source is None or source.isNull():
        return 1.0
    base = downsampled_image(rotated_image(source, rotation), downsample_factor)
    if (
        base.width() <= 0
        or base.height() <= 0
        or target_size is None
        or target_size.width() <= 0
        or target_size.height() <= 0
    ):
        return 1.0
    return min(
        target_size.width() / base.width(), target_size.height() / base.height()
    )


def render_view(
    source: Optional[QImage],
    rotation: int,
    zoom: str,
    target_size: QSize,
    downsample_factor: int = 1,
    *,
    zoom_factor: Optional[float] = None,
    pan_offset=(0.0, 0.0),
) -> Optional[QPixmap]:
    """Render the preview pixmap for ``source`` under ``rotation`` and ``zoom``.

    Returns ``None`` when there is no renderable source (so the caller can
    clear the label); otherwise returns the pixmap to display.  ``downsample_factor``
    (``Res 1/N``) scales the rotated image down by that factor before the zoom
    is applied — a display-only approximation of the Tk engine-coupled preview
    downsample.  The source image is never mutated.

    ``zoom_factor`` (M18) selects the continuous numeric zoom path: when given
    (and ``zoom`` is not ``Fit``) the rotated/downsampled image is scaled by
    ``zoom_factor`` (clamped to ``[MIN_ZOOM, MAX_ZOOM]``) instead of the
    discrete percent label.  ``pan_offset`` is the viewport pan offset applied
    on top; a non-zero offset composes the scaled image into a ``target_size``
    canvas (centred + offset, clipped).  Omitting ``zoom_factor`` keeps the
    original discrete-zoom behaviour for backward compatibility.
    """
    if source is None or source.isNull():
        return None
    rotated = rotated_image(source, rotation)
    rotated = downsampled_image(rotated, downsample_factor)
    if zoom == "Fit":
        return fit_pixmap(rotated, target_size)
    if zoom_factor is None:
        return QPixmap.fromImage(percent_scaled_image(rotated, zoom))
    scaled = scaled_image(rotated, clamp_zoom_factor(zoom_factor))
    ox, oy = pan_offset
    if ox or oy:
        return compose_panned_pixmap(
            QPixmap.fromImage(scaled), target_size, (ox, oy)
        )
    return QPixmap.fromImage(scaled)
