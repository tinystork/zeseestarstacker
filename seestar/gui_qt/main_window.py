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

The Stack tab exposes a *minimal* subset of the real settings controls
(input/output/temp/filename, batch size, stacking mode, drizzle, local solver).
Those controls feed a :class:`~seestar.gui_qt.settings_state.QtSettingsState`
model, which :meth:`MainWindow.build_run_request` turns into a validated,
immutable :class:`~seestar.gui_qt.run_bridge.RunRequest` — without ever starting
the backend.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .run_bridge import RunRequest, build_run_request as _build_run_request
from .settings_state import QtSettingsState

DEFAULT_TITLE = "ZeSeestarStacker — PySide6 shell"

# Placeholder tab labels used by the smoke test and future parity migration.
TAB_STACK = "Stack"
TAB_SETTINGS = "Settings"
TAB_PREVIEW = "Preview"
TAB_LOG = "Log"

# Minimal visible subset of stacking modes, drizzle modes and local-solver
# preferences.  Values are the *backend* keys, not display labels.
STACKING_MODES = [
    "kappa-sigma",
    "classic",
    "mean",
    "median",
    "winsorized-sigma-clip",
    "linear-fit-clip",
]
DRIZZLE_MODES = ["Final", "Incremental"]
SOLVER_PREFERENCES = ["none", "astap", "zesolver"]


class MainWindow(QMainWindow):
    """Minimal side-by-side Qt main window used for offscreen smoke tests.

    The window owns presentation state (title, tabs, progress bar, status bar)
    plus a small settings form whose values are mirrored into a
    :class:`QtSettingsState` model.  It never touches the scientific engine and
    never starts a real run.
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
        self.settings_state: QtSettingsState = QtSettingsState()

        self._build_central()
        self._build_status_bar()
        self._wire_controls()
        self._wire_settings_controls()
        self._sync_state_from_controls()

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

        # --- Minimal settings form (visible subset only) ---
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.temp_edit = QLineEdit()
        self.output_filename_edit = QLineEdit()

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(-1, 1_000_000)
        self.batch_spin.setValue(0)
        self.batch_spin.setToolTip("Batch size (-1 = auto).")

        self.stacking_mode_combo = QComboBox()
        self.stacking_mode_combo.addItems(STACKING_MODES)
        self.stacking_mode_combo.setCurrentText("kappa-sigma")

        self.drizzle_check = QCheckBox("Enable drizzle")
        self.drizzle_check.setChecked(False)

        self.drizzle_mode_combo = QComboBox()
        self.drizzle_mode_combo.addItems(DRIZZLE_MODES)
        self.drizzle_mode_combo.setCurrentText("Final")

        self.drizzle_group_spin = QSpinBox()
        self.drizzle_group_spin.setRange(1, 1_000_000)
        self.drizzle_group_spin.setValue(50)

        self.solver_combo = QComboBox()
        self.solver_combo.addItems(SOLVER_PREFERENCES)
        self.solver_combo.setCurrentText("none")

        form = QFormLayout()
        form.addRow("Input folder", self.input_edit)
        form.addRow("Output folder", self.output_edit)
        form.addRow("Temp folder", self.temp_edit)
        form.addRow("Output filename", self.output_filename_edit)
        form.addRow("Batch size", self.batch_spin)
        form.addRow("Stacking mode", self.stacking_mode_combo)
        form.addRow("", self.drizzle_check)
        form.addRow("Drizzle mode", self.drizzle_mode_combo)
        form.addRow("Drizzle group size", self.drizzle_group_spin)
        form.addRow("Local solver", self.solver_combo)
        layout.addLayout(form)

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

    def _wire_settings_controls(self) -> None:
        """Mirror every settings widget into ``self.settings_state`` on change."""
        self.input_edit.textChanged.connect(self._sync_state_from_controls)
        self.output_edit.textChanged.connect(self._sync_state_from_controls)
        self.temp_edit.textChanged.connect(self._sync_state_from_controls)
        self.output_filename_edit.textChanged.connect(self._sync_state_from_controls)
        self.batch_spin.valueChanged.connect(self._sync_state_from_controls)
        self.stacking_mode_combo.currentIndexChanged.connect(
            self._sync_state_from_controls
        )
        self.drizzle_check.stateChanged.connect(self._sync_state_from_controls)
        self.drizzle_mode_combo.currentIndexChanged.connect(
            self._sync_state_from_controls
        )
        self.drizzle_group_spin.valueChanged.connect(self._sync_state_from_controls)
        self.solver_combo.currentIndexChanged.connect(self._sync_state_from_controls)

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

    # ------------------------------------------------- settings collection
    def _sync_state_from_controls(self, *_ignored) -> None:
        """Copy the current widget values into the settings model.

        Accepts (and ignores) any signal payload so it can be connected to
        ``textChanged``/``valueChanged``/``currentIndexChanged``/``stateChanged``
        directly.
        """
        state = self.settings_state
        state.input_folder = self.input_edit.text()
        state.output_folder = self.output_edit.text()
        state.temp_folder = self.temp_edit.text()
        state.output_filename = self.output_filename_edit.text()
        state.batch_size = self.batch_spin.value()
        state.stacking_mode = self.stacking_mode_combo.currentText()
        state.use_drizzle = self.drizzle_check.isChecked()
        state.drizzle_mode = self.drizzle_mode_combo.currentText()
        state.drizzle_group_size = self.drizzle_group_spin.value()
        state.local_solver_preference = self.solver_combo.currentText()

    def collect_settings_state(self) -> QtSettingsState:
        """Return the settings model, freshly synced from the current widgets.

        This is the single choke point between Qt widgets and the run-request
        builder: it reads *plain* values only, never widgets, so downstream code
        (and tests) never need a live Qt object.
        """
        self._sync_state_from_controls()
        return self.settings_state

    def build_run_request(
        self,
        *,
        initial_additional_folders: Optional[List[str]] = None,
        auto_chunk_size: Optional[int] = None,
        special_single: bool = False,
    ) -> RunRequest:
        """Build an immutable :class:`RunRequest` from the current settings.

        This does **not** start the backend: it only collects the visible
        controls into a :class:`QtSettingsState` and forwards it to the
        Qt/Tk-independent ``run_config.build_run_request``.
        """
        state = self.collect_settings_state()
        return _build_run_request(
            state,
            initial_additional_folders=initial_additional_folders,
            auto_chunk_size=auto_chunk_size,
            special_single=special_single,
        )

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
