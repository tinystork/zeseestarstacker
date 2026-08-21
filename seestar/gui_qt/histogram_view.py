"""Qt interactive histogram view (M14) — display-only, single live surface.

Replaces the former static ``QLabel`` histogram pixmap with a real ``QWidget``
that reproduces the Tk ``HistogramWidget`` *interactions* (auto-zoom, reset
view, zoom, reset zoom and BP/WP line dragging) without importing Tk, the
scientific engine, or any plotting/image library.  The per-channel histogram
is computed by the numpy-lazy helpers in
:mod:`seestar.gui_qt.preview_adjust`; all drawing here is pure ``QPainter``.

Coordinate space: the widget works in a normalised "level" space ``[0, 1]`` so
the black/white-point lines share the exact axis of the stretch sliders.  This
mirrors the Tk ``HistogramWidget.set_range`` / ``update_stretch_from_histogram``
contract (BP/WP are 0-1 UI values); the ``rangeChanged`` signal is the Qt
equivalent of Tk's ``range_change_callback``.

The widget never mutates any image: it only holds a computed histogram and a
percentile level.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .preview_adjust import compute_histogram, compute_histogram_percentile

# Channel-name -> bar colour (matches the previous ``render_histogram_pixmap``
# palette so the new surface stays visually consistent with M10).
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

# Default black/white line positions (Tk stretch defaults).
_DEFAULT_BLACK_POINT = 0.01
_DEFAULT_WHITE_POINT = 0.99
_MIN_SEPARATION = 1e-4


class HistogramView(QWidget):
    """A small interactive per-channel histogram with BP/WP lines and zoom.

    ``set_data`` feeds a new (WB-only) display image; ``set_range`` positions
    the black/white lines; ``zoom_histogram`` / ``reset_histogram_view`` /
    ``reset_zoom`` reproduce the Tk zoom behaviours; dragging a line emits
    ``rangeChanged`` on release (the Qt ``update_stretch_from_histogram``).
    """

    # Emitted on BP/WP line release with the new (0-1) black/white points.
    rangeChanged = Signal(float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._histogram: Optional[Dict[str, Any]] = None
        self._percentile_99_5: float = 1.0
        self._black_point: float = _DEFAULT_BLACK_POINT
        self._white_point: float = _DEFAULT_WHITE_POINT
        self._view_min: float = 0.0
        self._view_max: float = 1.0
        self.auto_zoom_enabled: bool = False
        self._drag_line: Optional[str] = None
        self.setMinimumSize(256, 64)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ data
    def set_data(self, image: Optional[QImage]) -> None:
        """Feed a new display image (WB-only for the Tk histogram source)."""
        if image is None or image.isNull():
            self._histogram = None
            self._percentile_99_5 = 1.0
        else:
            self._histogram = compute_histogram(image, bins=256)
            p99_5 = compute_histogram_percentile(image, 99.5)
            self._percentile_99_5 = 1.0 if p99_5 is None else p99_5
        if self.auto_zoom_enabled and self._histogram:
            self.zoom_histogram()
        else:
            self.reset_zoom()
        self.update()

    def clear(self) -> None:
        """Clear the histogram surface (no preview)."""
        self.set_data(None)

    @property
    def has_data(self) -> bool:
        """True while a histogram is present."""
        return bool(self._histogram)

    @property
    def histogram(self) -> Optional[Dict[str, Any]]:
        """The current per-channel histogram (``{channel: int64 array}``)."""
        return self._histogram

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
        """Position the black/white lines (0-1 UI values, Tk ``set_range``)."""
        bp = float(bp)
        wp = float(wp)
        if wp <= bp + _MIN_SEPARATION:
            wp = min(1.0, bp + _MIN_SEPARATION)
        if bp >= wp - _MIN_SEPARATION:
            bp = max(0.0, wp - _MIN_SEPARATION)
        self._black_point = min(max(bp, 0.0), 1.0)
        self._white_point = min(max(wp, 0.0), 1.0)
        self.update()

    # ----------------------------------------------------------- zoom/view
    @property
    def view_range(self) -> tuple:
        """Current zoom window ``(view_min, view_max)`` in 0-1 level space."""
        return (self._view_min, self._view_max)

    def zoom_histogram(self) -> None:
        """Zoom the X axis to ``[0, max(0.02, p99.5)]`` (Tk ``zoom_histogram``)."""
        self._view_min = 0.0
        self._view_max = max(0.02, self._percentile_99_5)
        self.update()

    def reset_histogram_view(self) -> None:
        """Reset the X axis to the full ``[0, 1]`` range (Tk ``reset_histogram_view``)."""
        self._view_min = 0.0
        self._view_max = 1.0
        self.update()

    def reset_zoom(self) -> None:
        """Reset the X axis to the full data range (Tk ``reset_zoom``)."""
        self._view_min = 0.0
        self._view_max = 1.0
        self.update()

    # -------------------------------------------------------- drag plumbing
    # The mouse handlers below are thin wrappers around these small methods so
    # the interaction logic is unit-testable without synthesising mouse events.
    def _plot_rect(self) -> QRectF:
        margin = 4.0
        return QRectF(
            margin, margin, max(1.0, self.width() - 2.0 * margin),
            max(1.0, self.height() - 2.0 * margin),
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
        if not self._histogram:
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
        """Move the active line to pixel ``x`` (clamped + kept separated)."""
        if not self._drag_line or not self._histogram:
            return
        level = min(max(self._x_to_level(x), 0.0), 1.0)
        if self._drag_line == "min":
            self._black_point = min(level, self._white_point - _MIN_SEPARATION)
        else:
            self._white_point = max(level, self._black_point + _MIN_SEPARATION)
        self.update()

    def _end_drag(self) -> None:
        """Finish a drag, emitting ``rangeChanged`` with the final BP/WP."""
        if self._drag_line and self._histogram:
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

    # ------------------------------------------------------------- drawing
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKGROUND)
        if not self._histogram:
            painter.setPen(_EMPTY_TEXT_COLOR)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No preview")
            painter.end()
            return

        rect = self._plot_rect()
        if rect.width() <= 1.0 or rect.height() <= 1.0:
            painter.end()
            return

        # Normalise bar heights over the *visible* bins so zooming reveals detail.
        max_count = 1
        for h in self._histogram.values():
            n = len(h)
            for i in range(n):
                center = (i + 0.5) / n
                if self._view_min <= center <= self._view_max:
                    max_count = max(max_count, int(h[i]))

        for name, h in self._histogram.items():
            color = _CHANNEL_COLORS.get(name, QColor(225, 225, 225))
            painter.setPen(color)
            n = len(h)
            for i in range(n):
                count = int(h[i])
                if count <= 0:
                    continue
                center = (i + 0.5) / n
                if center < self._view_min or center > self._view_max:
                    continue
                bar = int(round(count / max_count * (rect.height() - 1.0)))
                x = self._level_to_x(center)
                painter.drawLine(
                    QPointF(x, rect.bottom()), QPointF(x, rect.bottom() - bar)
                )

        self._draw_line(painter, rect, self._black_point, _BLACK_POINT_COLOR)
        self._draw_line(painter, rect, self._white_point, _WHITE_POINT_COLOR)
        painter.end()

    def _draw_line(self, painter: QPainter, rect: QRectF, level: float, color: QColor) -> None:
        x = self._level_to_x(level)
        pen = QPen(color, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
