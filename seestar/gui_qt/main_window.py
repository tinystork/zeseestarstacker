"""Minimal PySide6 main-window shell for ZeSeestarStacker.

This is a *non-default* GUI shell.  The default entry point remains the Tk
``seestar.main:main`` application; this module is an architectural foothold for
the later parity migration and deliberately does NOT:

* launch any real stacking,
* import the Tk GUI,
* import the scientific engine.

Threading rule (enforced by design, not just convention): the shell starts a
run on a dedicated ``QThread`` via a
:class:`~seestar.gui_qt.run_controller.RunController`, but nothing mutates Qt
widgets off the GUI thread.  The worker communicates with this window
exclusively through Qt queued signals — never by touching widgets directly, and
never by receiving or storing widget references.

The Stack tab exposes a *minimal* subset of the real settings controls
(input/output/temp/filename, batch size, stacking mode, drizzle, local solver).
Those controls feed a :class:`~seestar.gui_qt.settings_state.QtSettingsState`
model, which :meth:`MainWindow.build_run_request` turns into a validated,
immutable :class:`~seestar.gui_qt.run_bridge.RunRequest`, which the Start button
hands to the lifecycle controller.  The default backend remains simulated;
real ``SeestarQueuedStacker`` activation is explicit opt-in only.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtCore import Qt
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

from .backend_runner import BaseRunBackend, SeestarQueuedStackerBackend
from .run_bridge import RunRequest, build_run_request as _build_run_request
from .run_controller import RunController
from .settings_state import QtSettingsState

DEFAULT_TITLE = "ZeSeestarStacker — PySide6 shell"

# Backend selection modes understood by the shell's Start button.
BACKEND_MODES = ("simulated", "seestar")
DEFAULT_BACKEND_MODE = "simulated"

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
    :class:`QtSettingsState` model.  It never touches the scientific engine at
    import time.  The default Start path begins a *simulated* run through its
    :class:`RunController`; an explicit backend factory or ``backend_mode`` can
    opt into another backend.  All widget updates happen in GUI-thread slots
    connected to the controller's queued signals.
    """

    def __init__(
        self,
        title: Optional[str] = None,
        parent: Optional[QWidget] = None,
        *,
        backend_factory: Optional[Callable[[], BaseRunBackend]] = None,
        backend_mode: str = DEFAULT_BACKEND_MODE,
    ):
        super().__init__(parent)
        if backend_mode not in BACKEND_MODES:
            raise ValueError(
                f"backend_mode must be one of {BACKEND_MODES!r}, got {backend_mode!r}"
            )
        if backend_factory is not None and not callable(backend_factory):
            raise TypeError("backend_factory must be callable or None")
        self.backend_factory = backend_factory
        self.backend_mode = backend_mode
        self.setWindowTitle(title if title is not None else DEFAULT_TITLE)
        self._running: bool = False
        self._shutdown_called: bool = False
        self.settings_state: QtSettingsState = QtSettingsState()
        self.controller = RunController(self)

        self._build_central()
        self._build_status_bar()
        self._wire_controls()
        self._wire_settings_controls()
        self._wire_controller()
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

    def _wire_controller(self) -> None:
        """Connect the lifecycle controller to this window's GUI-thread slots.

        All of these are queued signal deliveries from the worker thread
        (relayed by the controller), so every slot below runs on the GUI thread
        and is the only place widgets may be updated.
        """
        self.controller.started.connect(self._on_run_started)
        self.controller.progress_changed.connect(self._on_progress)
        self.controller.log_message.connect(self.log)
        self.controller.finished.connect(self._on_run_finished)
        self.controller.failed.connect(self._on_run_failed)
        self.controller.cancelled.connect(self._on_run_cancelled)

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
        request = self.build_run_request()
        backend = self.resolve_backend()
        if backend is None:
            self.controller.start(request)
        else:
            self.controller.start(request, backend=backend)

    def _on_stop(self) -> None:
        if not self._running:
            return
        self.controller.cancel()
        self.statusBar().showMessage("Cancelling…")
        self.log("Stop requested.")

    # ------------------------------------------------- lifecycle callbacks
    def _on_run_started(self) -> None:
        self._running = True
        self._update_run_state()
        self.statusBar().showMessage("Running…")

    def _on_progress(self, percent: int) -> None:
        self.progress.setValue(int(percent))

    def _on_run_finished(self) -> None:
        self._running = False
        self._update_run_state()
        self.progress.setValue(100)
        self.statusBar().showMessage("Finished.")
        self.log("Run finished.")

    def _on_run_failed(self, message: str) -> None:
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage(f"Failed: {message}")
        self.log(f"Run failed: {message}")

    def _on_run_cancelled(self) -> None:
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage("Cancelled.")
        self.log("Run cancelled.")

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

    def resolve_backend(self) -> Optional[BaseRunBackend]:
        """Return the run backend for a Start, or ``None`` for the default.

        The default (``backend_factory is None`` and
        ``backend_mode == "simulated"``) returns ``None`` so
        :meth:`RunController.start` falls back to its built-in
        :class:`SimulatedRunBackend`.  A non-default selection produces an
        explicit backend:

        * ``backend_factory`` (when set) is called to produce a fresh backend,
        * ``backend_mode == "seestar"`` lazily constructs a
          :class:`SeestarQueuedStackerBackend` — the real engine is imported
          only when that backend's ``run()`` is invoked, never here.
        """
        if self.backend_factory is not None:
            return self.backend_factory()
        if self.backend_mode == "seestar":
            return SeestarQueuedStackerBackend()
        return None

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
        (e.g. from an application-level ``aboutToQuit`` handler).  It requests
        stop of any live run, waits for the worker QThread to finish, and
        resets UI state.  Safe to call multiple times.
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self.controller.shutdown()
        self._running = False
        self._update_run_state()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override name
        self.shutdown()
        event.accept()
