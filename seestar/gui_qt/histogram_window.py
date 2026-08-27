"""Detached presentation surface for the authoritative OTPUX histogram.

This module deliberately contains no histogram computation and owns no worker.
The window is a second, larger :class:`HistogramView` fed by ``MainWindow``
with the exact same cached model object and display state as the inline view.
All actions are emitted back to ``MainWindow``, which remains the sole owner of
BP/WP, zoom policy, Auto Stretch, Auto WB and the H1/H2 worker lifecycle.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .histogram_view import HistogramView


class DetachedHistogramWindow(QDialog):
    """Large non-modal mirror of the inline histogram.

    Closing hides the dialog (Qt's default for a non-deleting dialog); the
    object, model reference and view state remain owned by ``MainWindow`` for a
    later reopen.  No action here can touch processing or scientific buffers.
    """

    rangeChanged = Signal(float, float)
    autoZoomToggled = Signal(bool)
    resetViewRequested = Signal()
    zoomRequested = Signal()
    resetZoomRequested = Signal()
    autoStretchRequested = Signal()
    autoWbRequested = Signal()
    liveAutoStretchToggled = Signal(bool)
    liveAutoWbToggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.resize(960, 640)

        layout = QVBoxLayout(self)
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        self.histogram_view = HistogramView(self)
        self.histogram_view.setMinimumSize(720, 400)
        layout.addWidget(self.histogram_view, 1)

        toolbar = QWidget(self)
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(0, 0, 0, 0)
        self.auto_zoom_check = QCheckBox()
        self.reset_view_button = QPushButton()
        self.zoom_button = QPushButton()
        self.reset_zoom_button = QPushButton("R")
        self.reset_zoom_button.setToolTip("Reset zoom")
        self.auto_stretch_button = QPushButton()
        self.auto_wb_button = QPushButton()
        self.live_auto_stretch_check = QCheckBox()
        self.live_auto_wb_check = QCheckBox()
        self.close_button = QPushButton()
        for widget in (
            self.auto_zoom_check,
            self.reset_view_button,
            self.zoom_button,
            self.reset_zoom_button,
            self.auto_stretch_button,
            self.auto_wb_button,
            self.live_auto_stretch_check,
            self.live_auto_wb_check,
        ):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(self.close_button)
        layout.addWidget(toolbar)

        self.histogram_view.rangeChanged.connect(self.rangeChanged.emit)
        self.auto_zoom_check.toggled.connect(self.autoZoomToggled.emit)
        self.reset_view_button.clicked.connect(self.resetViewRequested.emit)
        self.zoom_button.clicked.connect(self.zoomRequested.emit)
        self.reset_zoom_button.clicked.connect(self.resetZoomRequested.emit)
        self.auto_stretch_button.clicked.connect(self.autoStretchRequested.emit)
        self.auto_wb_button.clicked.connect(self.autoWbRequested.emit)
        self.live_auto_stretch_check.toggled.connect(
            self.liveAutoStretchToggled.emit
        )
        self.live_auto_wb_check.toggled.connect(self.liveAutoWbToggled.emit)
        self.close_button.clicked.connect(self.close)

    def set_texts(self, translate) -> None:
        """Refresh localized visible strings without rebuilding the dialog."""
        self.setWindowTitle(translate("histogram_window_title"))
        self.auto_zoom_check.setText(translate("histo_auto_zoom"))
        self.reset_view_button.setText(translate("histo_reset"))
        self.zoom_button.setText(translate("histo_zoom"))
        self.auto_stretch_button.setText(translate("auto_stretch"))
        self.auto_wb_button.setText(translate("auto_wb"))
        self.live_auto_stretch_check.setText(translate("live_auto_stretch"))
        self.live_auto_wb_check.setText(translate("live_auto_wb"))
        self.close_button.setText(translate("close"))

    def set_model(self, model: Optional[Dict[str, Any]]) -> None:
        """Mirror the already-computed authoritative model object verbatim."""
        self.histogram_view.set_model(model)

    def set_legacy_data(
        self,
        histogram: Optional[Dict[str, Any]],
        percentile_99_5: float = 1.0,
    ) -> None:
        """Mirror the inline legacy cache without recomputing it."""
        self.histogram_view.set_legacy_data(histogram, percentile_99_5)

    def set_histogram_actions_enabled(self, enabled: bool) -> None:
        """Enable data-dependent actions; live-auto toggles remain available."""
        for widget in (
            self.auto_zoom_check,
            self.reset_view_button,
            self.zoom_button,
            self.reset_zoom_button,
            self.auto_stretch_button,
            self.auto_wb_button,
        ):
            widget.setEnabled(bool(enabled))
