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

The source :class:`QImage` is never mutated: every helper returns a fresh
image/pixmap, so the stored original display image stays pristine for
subsequent zoom/rotation recomputation.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap, QTransform

# User-visible zoom labels (order matches the ``zoom_combo`` items).
ZOOM_LABELS = ("Fit", "100%", "200%", "50%")

# Percent-zoom factor per label.  ``Fit`` is handled separately (aspect-fit).
_ZOOM_FACTORS = {"100%": 1.0, "200%": 2.0, "50%": 0.5}


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
    factor = _ZOOM_FACTORS.get(zoom, 1.0)
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


def render_view(
    source: Optional[QImage],
    rotation: int,
    zoom: str,
    target_size: QSize,
    downsample_factor: int = 1,
) -> Optional[QPixmap]:
    """Render the preview pixmap for ``source`` under ``rotation`` and ``zoom``.

    Returns ``None`` when there is no renderable source (so the caller can
    clear the label); otherwise returns the pixmap to display.  ``downsample_factor``
    (``Res 1/N``) scales the rotated image down by that factor before the zoom
    is applied — a display-only approximation of the Tk engine-coupled preview
    downsample.  The source image is never mutated.
    """
    if source is None or source.isNull():
        return None
    rotated = rotated_image(source, rotation)
    rotated = downsampled_image(rotated, downsample_factor)
    if zoom == "Fit":
        return fit_pixmap(rotated, target_size)
    return QPixmap.fromImage(percent_scaled_image(rotated, zoom))
