"""Minimal PySide6 main-window shell for ZeSeestarStacker.

This is a *non-default* GUI shell.  The default entry point remains the Tk
``seestar.main:main`` application; this module is an architectural foothold for
the later parity migration and deliberately does NOT:

* launch any real stacking,
* import the Tk GUI,
* import the scientific engine,
* spawn background workers.

Threading rule (enforced by design, not just convention): this shell spawns no
threads at all, so nothing can mutate Qt widgets off the GUI thread.  Any
future worker must communicate with this window exclusively through Qt queued
signals — never by touching widgets directly.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

DEFAULT_TITLE = "ZeSeestarStacker — PySide6 shell"

# Placeholder tab labels used by the smoke test and future parity migration.
TAB_STACK = "Stack"
TAB_SETTINGS = "Settings"
TAB_PREVIEW = "Preview"
TAB_LOG = "Log"


class MainWindow(QMainWindow):
    """Minimal side-by-side Qt main window used for offscreen smoke tests.

    The window owns only presentation state: a title, a tab widget with
    placeholder panels, a progress bar, a status bar, and inert start/stop
    controls.  It never touches the scientific engine and never starts a real
    run.
    """

    # Emitted when the inert start/stop controls change state.  These exist so
    # a future worker/controller can be wired in via queued connections without
    # mutating widgets from another thread.
    started = Signal()
    stopped = Signal()

    def __init__(self, title: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(title if title is not None else DEFAULT_TITLE)
        self._running: bool = False
        self._shutdown_called: bool = False

        self._build_central()
        self._build_status_bar()
        self._wire_controls()

    # ------------------------------------------------------------------ UI
    def _build_central(self) -> None:
        self.tabs = QTabWidget()

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        self.tabs.addTab(self._build_stack_tab(), TAB_STACK)
        self.tabs.addTab(self._placeholder_panel("Settings placeholder"), TAB_SETTINGS)
        self.tabs.addTab(self._placeholder_panel("Preview placeholder"), TAB_PREVIEW)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.tabs.addTab(self.log_view, TAB_LOG)

        self.setCentralWidget(self.tabs)

    def _build_stack_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        label = QLabel("Stacking pipeline placeholder — no real work is performed.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        layout.addStretch(1)

        return panel

    def _placeholder_panel(self, text: str) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return panel

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        status.addPermanentWidget(self.progress)
        self.setStatusBar(status)
        self.statusBar().showMessage("Idle")

    def _wire_controls(self) -> None:
        self.start_button.clicked.connect(self._on_start)
        self.stop_button.clicked.connect(self._on_stop)
        self._update_run_state()

    # ------------------------------------------------------------ controls
    def _on_start(self) -> None:
        if self._running:
            return
        self._running = True
        self.started.emit()
        self._update_run_state()
        self.statusBar().showMessage("Running (placeholder)…")
        self.log("Started (placeholder — no real stacking).")

    def _on_stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.stopped.emit()
        self._update_run_state()
        self.statusBar().showMessage("Stopped.")
        self.log("Stopped.")

    def _update_run_state(self) -> None:
        self.start_button.setEnabled(not self._running)
        self.stop_button.setEnabled(self._running)

    # ------------------------------------------------------------- helpers
    def log(self, message: str) -> None:
        """Append a line to the read-only log tab."""
        self.log_view.append(message)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def shutdown_called(self) -> bool:
        return self._shutdown_called

    def shutdown(self) -> None:
        """Idempotent teardown hook.

        Called automatically by :meth:`closeEvent` and safe to call directly
        (e.g. from an application-level ``aboutToQuit`` handler).  Subclasses
        may extend it for real teardown; the base implementation only stops the
        inert run state.
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True
        if self._running:
            self._on_stop()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override name
        self.shutdown()
        event.accept()
