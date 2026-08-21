"""Qt preview image surface (M18): a ``QLabel`` that forwards wheel-zoom and
left-drag pan gestures to the owning :class:`~seestar.gui_qt.main_window.MainWindow`.

Pure view widget: it converts native mouse events into semantic Qt signals and
never touches image data, the engine, or the Tk GUI.  The owner turns those
signals into view-transform state (a continuous zoom factor plus a pan offset)
and re-renders through :func:`seestar.gui_qt.preview_view.render_view`.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel


class PreviewImageView(QLabel):
    """Preview surface emitting wheel-zoom and pan gestures.

    Signals
    -------
    wheelZoom(int, float, float)
        ``+1`` for zoom-in / ``-1`` for zoom-out (Tk ``Button-4``/``Button-5`` /
        ``event.delta`` parity), followed by the cursor ``(x, y)`` in widget
        coordinates so the owner can anchor the zoom at the cursor.
    panDelta(float, float)
        the viewport ``(dx, dy)`` since the previous mouse-move while the left
        button is held (left-drag pan, Tk ``B1-Motion`` parity).
    """

    wheelZoom = Signal(int, float, float)
    panDelta = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(256, 256)
        self._panning = False
        self._last_pos = None

    def wheelEvent(self, event) -> None:
        """Forward a wheel turn as a ``wheelZoom`` signal (zoom in/out)."""
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        pos = event.position()
        self.wheelZoom.emit(1 if delta > 0 else -1, pos.x(), pos.y())
        event.accept()

    def mousePressEvent(self, event) -> None:
        """Begin a left-drag pan on left-button press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._last_pos = event.position()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Emit the pan delta while dragging with the left button held."""
        if self._panning and self._last_pos is not None:
            pos = event.position()
            dx = pos.x() - self._last_pos.x()
            dy = pos.y() - self._last_pos.y()
            self._last_pos = pos
            self.panDelta.emit(float(dx), float(dy))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """End the left-drag pan on left-button release."""
        if event.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self._last_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)
