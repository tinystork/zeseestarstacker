"""Qt interactive histogram view (M14 / ZSSS-OTPUX-HIST-H1) — display-only.

Replaces the former static ``QLabel`` histogram pixmap with a real ``QWidget``
that reproduces the Tk ``HistogramWidget`` *interactions* (auto-zoom, reset
view, zoom, reset zoom and BP/WP line dragging) without importing Tk, the
scientific engine, or any plotting/image library.

Since H1 the widget has two data inputs:

* ``set_model`` — the *authoritative* immutable result of
  :func:`seestar.gui_qt.preview_analysis.compute_histogram_float`: a 512-bin
  float-domain model over ``[0, 1]`` with per-channel ``counts`` /
  ``log_counts`` / ``stats`` and a robust plotted X range plus the explicit
  full ``[0, 1]`` range.  This is what production Option-A previews feed; no
  ``QImage`` round-trip is involved for the histogram or its statistics.
* ``set_data`` — the legacy single-array compatibility path, which still takes
  a WB-only ``QImage`` and computes the historical 256-bin ``uint8`` histogram
  via :mod:`seestar.gui_qt.preview_adjust`.  It is retained only so old
  producers keep working; the Option-A model is authoritative.

Rendering draws the model bars from ``log_counts`` (log-space heights) so fine
tonal detail stays readable, overlays the R/G/B (or L) channels on the same
axes, labels the plotted X domain, and draws the BP/WP lines in the normalised
``[0, 1]`` level space shared with the stretch sliders.  BP/WP line dragging
emits ``rangeChanged`` **live/coalesced** at ~25 ms during the drag and an
exact final emission on release (the Qt equivalent of Tk's
``update_stretch_from_histogram``).

The widget never mutates any image: it only holds a computed model/histogram
and a percentile level.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .preview_adjust import (
    BP_WP_MIN_SEPARATION,
    compute_histogram,
    compute_histogram_percentile,
    normalize_bp_wp,
    quantize_bp_wp,
)

# Channel-name -> bar colour (matches the earlier M10 display-histogram
# palette so the surface stays visually consistent).
_CHANNEL_COLORS: Dict[str, QColor] = {
    "L": QColor(225, 225, 225),
    "R": QColor(225, 70, 70),
    "G": QColor(70, 200, 70),
    "B": QColor(70, 120, 225),
}

_BACKGROUND = QColor(22, 22, 24)
_BLACK_POINT_COLOR = QColor(255, 170, 170)
_WHITE_POINT_COLOR = QColor(170, 170, 255)
_EMPTY_TEXT_COLOR = QColor(120, 120, 120)
_AXIS_TEXT_COLOR = QColor(150, 150, 150)

# Default black/white line positions (Tk stretch defaults).
_DEFAULT_BLACK_POINT = 0.01
_DEFAULT_WHITE_POINT = 0.99
# Single authoritative BP/WP minimum separation (one slider/spin step), shared
# with the MainWindow stretch controls via ``preview_adjust``.
_MIN_SEPARATION = BP_WP_MIN_SEPARATION

# Live-drag coalescing cadence (ZSSS-OTPUX §5.7): intermediate BP/WP emissions
# during a drag are timer-coalesced to roughly this interval; the release
# emission is never dropped.
_LIVE_DRAG_INTERVAL_MS = 25

# Bottom strip reserved for the plotted X-domain labels.
_AXIS_MARGIN_BOTTOM = 14.0

# Deterministic minimum zoom width for the robust plotted X range (matches the
# legacy ``max(0.02, p99.5)`` lower bound so a degenerate-but-valid robust
# range never collapses to a sliver).
_MIN_ZOOM_WIDTH = 0.02


def _validated_x_range(x_range) -> tuple:
    """Return a validated robust plotted X range ``(lo, hi)`` or full ``(0, 1)``.

    Accepts the model ``x_range`` metadata; rejects non-sequence values,
    non-finite bounds, out-of-domain bounds and degenerate ``hi <= lo`` ranges
    by falling back to the explicit full ``[0, 1]`` range.  A valid range that
    is narrower than :data:`_MIN_ZOOM_WIDTH` is widened deterministically
    (centred, clamped to ``[0, 1]``) so zooming never collapses to a sliver.
    """
    if not isinstance(x_range, (tuple, list)) or len(x_range) != 2:
        return (0.0, 1.0)
    try:
        lo = float(x_range[0])
        hi = float(x_range[1])
    except (TypeError, ValueError):
        return (0.0, 1.0)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return (0.0, 1.0)
    if not (0.0 <= lo < hi <= 1.0):
        return (0.0, 1.0)
    if hi - lo < _MIN_ZOOM_WIDTH:
        mid = 0.5 * (lo + hi)
        lo = mid - 0.5 * _MIN_ZOOM_WIDTH
        hi = mid + 0.5 * _MIN_ZOOM_WIDTH
        if lo < 0.0:
            hi += -lo
            lo = 0.0
        if hi > 1.0:
            lo -= hi - 1.0
            hi = 1.0
        lo = max(0.0, lo)
        hi = min(1.0, hi)
    return (lo, hi)


def format_histogram_stats(stats: Optional[Dict[str, Dict[str, float]]]) -> Optional[str]:
    """Return a deterministic per-channel stats summary in the ``[0, 1]`` domain.

    Each channel reports ``min``/``max``/``median``/``mean``/``std`` (the five
    ratified stats), labelled ``R``/``G``/``B`` (or ``L`` for mono), joined by
    "·".  Returns ``None`` when ``stats`` is empty/``None``.
    """
    if not stats:
        return None
    parts = []
    for name in ("R", "G", "B", "L"):
        s = stats.get(name)
        if s is None:
            continue
        parts.append(
            f"{name} {s['min']:.3f}–{s['max']:.3f} "
            f"med {s['median']:.3f} mean {s['mean']:.3f} std {s['std']:.3f}"
        )
    if not parts:
        return None
    return " · ".join(parts)


class HistogramView(QWidget):
    """A small interactive per-channel histogram with BP/WP lines and zoom.

    ``set_model`` feeds the authoritative float-domain model (512 bins, RGB
    overlay or L, log-space heights, exact stats); ``set_data`` is the legacy
    QImage compatibility path.  ``set_range`` positions the black/white lines;
    ``zoom_histogram`` / ``reset_histogram_view`` / ``reset_zoom`` reproduce
    the Tk zoom behaviours; dragging a line emits ``rangeChanged`` live
    (coalesced ~25 ms) and exactly once on release.
    """

    # Emitted with the current (0-1) black/white points: live/coalesced during
    # a drag and exactly once (final, authoritative) on release.
    rangeChanged = Signal(float, float)
    # The inline compact view uses this as an obvious double-click expansion
    # seam.  Detached views may ignore it; emitting never computes data.
    expandRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Authoritative float model (compute_histogram_float result) or None.
        self._model: Optional[Dict[str, Any]] = None
        # Legacy 256-bin counts dict (``{channel: int64 array}``) or None.
        self._histogram: Optional[Dict[str, Any]] = None
        self._percentile_99_5: float = 1.0
        # Validated robust plotted X range from the float model (``(lo, hi)``)
        # or ``None`` while on the legacy path (which zooms ``[0, p99.5]``).
        self._x_range: Optional[tuple] = None
        self._black_point: float = _DEFAULT_BLACK_POINT
        self._white_point: float = _DEFAULT_WHITE_POINT
        self._view_min: float = 0.0
        self._view_max: float = 1.0
        # Manual-zoom window ``(view_min, view_max)`` preserved across data
        # refreshes (Tk ``freeze_x_range`` / ``_stored_xlim`` semantics), or
        # ``None`` while the view tracks the full data range.  Cleared by
        # ``reset_histogram_view`` / ``reset_zoom`` and when data is cleared.
        self._frozen_range: Optional[tuple] = None
        self.auto_zoom_enabled: bool = False
        self._drag_line: Optional[str] = None
        # Live-drag coalescing (single-shot, ~25 ms).  A pending intermediate
        # emission is dropped/replaced; the release emit is never dropped.
        self._live_drag_timer = QTimer(self)
        self._live_drag_timer.setSingleShot(True)
        self._live_drag_timer.setInterval(_LIVE_DRAG_INTERVAL_MS)
        self._live_drag_timer.timeout.connect(self._emit_live_drag)
        self.setMinimumSize(256, 64)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ data
    def set_model(self, model: Optional[Dict[str, Any]]) -> None:
        """Feed the authoritative float-domain histogram model.

        ``model`` is the immutable result of
        :func:`~seestar.gui_qt.preview_analysis.compute_histogram_float`
        (``bins``/``range``/``channels``/``counts``/``log_counts``/``stats``/
        ``x_range``/``full_range``).  Passing the same model object again only
        repaints (cheap), so a refresh that merely moves the BP/WP markers does
        not reset a manual zoom.
        """
        if model is not None and model is self._model:
            # Same model object: cheap repaint (no frozen-zoom reset).  A
            # ``None`` model always falls through so ``clear()`` also drops any
            # legacy histogram left behind by a prior ``set_data``.
            self.update()
            return
        self._model = model
        self._histogram = None
        if model is None:
            self._percentile_99_5 = 1.0
            self._x_range = None
            self._frozen_range = None
        else:
            x_range = model.get("x_range")
            self._x_range = _validated_x_range(x_range)
            self._percentile_99_5 = self._x_range[1]
        self._apply_view_after_data()
        self.update()

    def set_data(self, image: Optional[QImage]) -> None:
        """Legacy compatibility path: feed a WB-only display ``QImage``.

        Keeps the historical 256-bin ``uint8`` histogram for old single-array
        producers.  Production Option-A previews must use :meth:`set_model`.
        """
        self._model = None
        if image is None or image.isNull():
            self._histogram = None
            self._percentile_99_5 = 1.0
            self._x_range = None
            self._frozen_range = None
        else:
            self._histogram = compute_histogram(image, bins=256)
            p99_5 = compute_histogram_percentile(image, 99.5)
            self._percentile_99_5 = 1.0 if p99_5 is None else p99_5
            self._x_range = None
        self._apply_view_after_data()
        self.update()

    def set_legacy_data(self, histogram: Optional[Dict[str, Any]], percentile_99_5: float = 1.0) -> None:
        """Inject a pre-computed legacy 256-bin histogram + percentile (no recompute).

        Used by the legacy path's cache: ``MainWindow`` computes the histogram
        and percentile once per ``(source, WB)`` revision and feeds them here on
        refresh without a ``QImage`` round-trip recompute.  Mirrors
        :meth:`set_data` but skips ``compute_histogram`` /
        ``compute_histogram_percentile`` entirely.
        """
        self._model = None
        self._histogram = histogram
        self._percentile_99_5 = float(percentile_99_5)
        self._x_range = None
        self._apply_view_after_data()
        self.update()

    def _apply_view_after_data(self) -> None:
        """Re-apply auto/manual zoom state after a data (model) change."""
        if self.auto_zoom_enabled and self.has_data:
            self.zoom_histogram()
        elif self._frozen_range is not None:
            # Preserve a manual zoom across the refresh (Tk freeze_x_range).
            self._view_min, self._view_max = self._frozen_range
        else:
            self._view_min = 0.0
            self._view_max = 1.0

    def clear(self) -> None:
        """Clear the histogram surface (no preview)."""
        self.set_model(None)

    @property
    def has_data(self) -> bool:
        """True while a model or legacy histogram is present."""
        return self._model is not None or self._histogram is not None

    @property
    def model(self) -> Optional[Dict[str, Any]]:
        """The authoritative float-domain model, or ``None``."""
        return self._model

    @property
    def histogram(self) -> Optional[Dict[str, Any]]:
        """Per-channel bin counts ``{channel: int64 array}`` (model or legacy)."""
        if self._model is not None:
            return self._model.get("counts")
        return self._histogram

    @property
    def stats(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Per-channel ``{min, max, median, mean, std}`` stats, or ``None``."""
        if self._model is None:
            return None
        return self._model.get("stats")

    # -------------------------------------------------------------- BP/WP
    @property
    def black_point(self) -> float:
        """Current black-point line position in the 0-1 level scale."""
        return self._black_point

    @property
    def white_point(self) -> float:
        """Current white-point line position in the 0-1 level scale."""
        return self._white_point

    def set_range(self, bp: float, wp: float) -> None:
        """Position the black/white lines (0-1 UI values, Tk ``set_range``).

        Enforces ``0 <= BP < WP <= 1`` with the shared deterministic minimum
        separation via :func:`preview_adjust.normalize_bp_wp` (the same seam
        used by the MainWindow stretch controls, so the handles always agree
        with the slider/spin state).  Non-finite inputs fall back to the
        neutral defaults deterministically.  A no-op while a drag is active so
        a live-drag echo from the sliders never clobbers the authoritative
        in-flight line position (avoids jitter / rounding feedback).
        """
        if self._drag_line is not None:
            return
        self._black_point, self._white_point = normalize_bp_wp(bp, wp)
        self.update()

    # ----------------------------------------------------------- zoom/view
    @property
    def view_range(self) -> tuple:
        """Current zoom window ``(view_min, view_max)`` in 0-1 level space."""
        return (self._view_min, self._view_max)

    def zoom_histogram(self) -> None:
        """Zoom the X axis to the authoritative data range (Tk ``zoom_histogram``).

        The Option-A float model zooms to its validated robust plotted X range
        (both ``lo`` and ``hi``); the legacy path keeps the historical
        ``[0, max(0.02, p99.5)]`` window.  Invalid/degenerate model metadata
        already fell back to the full ``[0, 1]`` range in :meth:`set_model`.
        """
        self._view_min, self._view_max = self._zoom_window()
        # A manual zoom is frozen across refreshes unless auto-zoom is active
        # (Tk stores ``_stored_xlim`` only while ``freeze_x_range`` is set).
        if not self.auto_zoom_enabled:
            self._frozen_range = (self._view_min, self._view_max)
        self.update()

    def _zoom_window(self) -> tuple:
        """Return the zoom window ``(view_min, view_max)`` for the current data.

        Model data uses the validated robust ``x_range``; legacy data uses the
        historical ``[0, max(0.02, p99.5)]`` window.
        """
        if self._x_range is not None:
            return self._x_range
        return (0.0, max(0.02, self._percentile_99_5))

    def reset_histogram_view(self) -> None:
        """Reset the X axis to the full ``[0, 1]`` range (Tk ``reset_histogram_view``)."""
        self._view_min = 0.0
        self._view_max = 1.0
        self._frozen_range = None
        self.update()

    def reset_zoom(self) -> None:
        """Reset the X axis to the full data range (Tk ``reset_zoom``)."""
        self._view_min = 0.0
        self._view_max = 1.0
        self._frozen_range = None
        self.update()

    def set_view_range(self, view_min: float, view_max: float) -> None:
        """Apply a shared/frozen X window supplied by the owning controller.

        Used only to initialize/synchronize another presentation surface.  It
        never changes the histogram model or performs analysis.
        """
        validated = _validated_x_range((view_min, view_max))
        self._view_min, self._view_max = validated
        self._frozen_range = validated
        self.update()

    # -------------------------------------------------------- drag plumbing
    # The mouse handlers below are thin wrappers around these small methods so
    # the interaction logic is unit-testable without synthesising mouse events.
    def _plot_rect(self) -> QRectF:
        margin = 4.0
        return QRectF(
            margin,
            margin,
            max(1.0, self.width() - 2.0 * margin),
            max(1.0, self.height() - margin - _AXIS_MARGIN_BOTTOM),
        )

    def _level_to_x(self, level: float) -> float:
        rect = self._plot_rect()
        span = self._view_max - self._view_min
        if span <= 0.0:
            return rect.left()
        return rect.left() + (level - self._view_min) / span * rect.width()

    def _x_to_level(self, x: float) -> float:
        rect = self._plot_rect()
        if rect.width() <= 0.0:
            return 0.0
        frac = (x - rect.left()) / rect.width()
        return self._view_min + frac * (self._view_max - self._view_min)

    def _start_drag_at(self, x: float) -> Optional[str]:
        """Pick the nearest line to ``x`` (pixel) and start dragging it."""
        if not self.has_data:
            return None
        pick = max(5.0, 0.02 * self.width())
        d_bp = abs(x - self._level_to_x(self._black_point))
        d_wp = abs(x - self._level_to_x(self._white_point))
        if d_bp <= pick and d_bp <= d_wp:
            self._drag_line = "min"
        elif d_wp <= pick:
            self._drag_line = "max"
        else:
            self._drag_line = None
        return self._drag_line

    def _drag_at(self, x: float) -> None:
        """Move the active line to pixel ``x`` (clamped + kept separated).

        Schedules a coalesced live ``rangeChanged`` emission (the intermediate
        emission is timer-coalesced; raw mouse events never each emit).  The
        in-flight handle position is snapped to the shared control-resolution
        grid *before* it is stored/emitted, so the handle, the stretch
        sliders/spins and the authoritative MainWindow state agree exactly
        during the live drag, not only after release.
        """
        if not self._drag_line or not self.has_data:
            return
        level = min(max(self._x_to_level(x), 0.0), 1.0)
        if self._drag_line == "min":
            self._black_point = quantize_bp_wp(
                min(level, self._white_point - _MIN_SEPARATION)
            )
        else:
            self._white_point = quantize_bp_wp(
                max(level, self._black_point + _MIN_SEPARATION)
            )
        self.update()
        self._schedule_live_drag()

    def _schedule_live_drag(self) -> None:
        """Arm the coalescing timer (no-op if already armed)."""
        if not self._live_drag_timer.isActive():
            self._live_drag_timer.start()

    def _emit_live_drag(self) -> None:
        """Emit one coalesced live BP/WP update (timer slot / test seam)."""
        self._live_drag_timer.stop()
        if self._drag_line is not None and self.has_data:
            self.rangeChanged.emit(self._black_point, self._white_point)

    def _end_drag(self) -> None:
        """Finish a drag, emitting the exact final BP/WP on release."""
        self._live_drag_timer.stop()
        if self._drag_line and self.has_data:
            self._drag_line = None
            self.rangeChanged.emit(self._black_point, self._white_point)
        self._drag_line = None

    # ------------------------------------------------------------- events
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_drag_at(event.position().x())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_line is not None:
            self._drag_at(event.position().x())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._end_drag()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.expandRequested.emit()
        super().mouseDoubleClickEvent(event)

    # ------------------------------------------------------------- drawing
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKGROUND)
        if not self.has_data:
            painter.setPen(_EMPTY_TEXT_COLOR)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No preview")
            painter.end()
            return

        rect = self._plot_rect()
        if rect.width() <= 1.0 or rect.height() <= 1.0:
            painter.end()
            return

        if self._model is not None:
            # 512-bin RGB overlay / L from log1p counts (readable heights).
            self._paint_bars(
                painter,
                rect,
                self._model["log_counts"],
            )
        else:
            # Legacy 256-bin linear counts.
            self._paint_bars(painter, rect, self._histogram)

        self._draw_line(painter, rect, self._black_point, _BLACK_POINT_COLOR)
        self._draw_line(painter, rect, self._white_point, _WHITE_POINT_COLOR)
        self._paint_axis_labels(painter, rect)
        painter.end()

    def _paint_bars(
        self,
        painter: QPainter,
        rect: QRectF,
        heights_by_name: Dict[str, Any],
    ) -> None:
        """Draw per-channel bars, normalising over the *visible* bins.

        Channels are composited *additively* so single R/G/B regions remain
        distinguishable and overlapping distributions remain visibly composite
        (pair overlaps blend toward yellow/magenta/cyan, the triple overlap
        toward white) instead of the last channel masking the earlier ones.
        The painter composition state is saved before the bars and restored
        before the BP/WP markers and axis labels are drawn.
        """
        max_h = 1.0
        for h in heights_by_name.values():
            n = len(h)
            for i in range(n):
                center = (i + 0.5) / n
                if self._view_min <= center <= self._view_max:
                    max_h = max(max_h, float(h[i]))

        painter.save()
        try:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Plus
            )
            for name, h in heights_by_name.items():
                color = _CHANNEL_COLORS.get(name, QColor(225, 225, 225))
                painter.setPen(color)
                n = len(h)
                for i in range(n):
                    value = float(h[i])
                    if value <= 0.0:
                        continue
                    center = (i + 0.5) / n
                    if center < self._view_min or center > self._view_max:
                        continue
                    bar = int(round(value / max_h * (rect.height() - 1.0)))
                    x = self._level_to_x(center)
                    painter.drawLine(
                        QPointF(x, rect.bottom()),
                        QPointF(x, rect.bottom() - bar),
                    )
        finally:
            painter.restore()

    def _paint_axis_labels(self, painter: QPainter, rect: QRectF) -> None:
        """Label the plotted X domain (view window) under the bars."""
        label_h = max(1.0, self.height() - rect.bottom() - 2.0)
        if label_h < 8.0:
            return
        painter.setPen(_AXIS_TEXT_COLOR)
        font = painter.font()
        if font.pointSize() > 7:
            font.setPointSize(font.pointSize() - 2)
        painter.setFont(font)
        y = rect.bottom() + 1.0
        painter.drawText(
            QRectF(rect.left(), y, 48.0, label_h),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{self._view_min:.2f}",
        )
        painter.drawText(
            QRectF(rect.right() - 48.0, y, 48.0, label_h),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{self._view_max:.2f}",
        )

    def _draw_line(self, painter: QPainter, rect: QRectF, level: float, color: QColor) -> None:
        x = self._level_to_x(level)
        pen = QPen(color, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
