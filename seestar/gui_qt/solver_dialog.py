"""Solver configuration dialog (Qt view/controller) for the PySide6 shell.

This is the first *real* Qt solver-configuration dialog — the reasonable parity
equivalent of the Tk ``LocalSolverSettingsWindow``.  It is a pure view/
controller over the existing solver boundaries: it never re-implements the
solver, never touches ZeSolver private APIs, and never mutates the ZeSoftware
contract.  The scientific semantics stay in the engine; this dialog only reads
and writes the *plain* solver settings carried by
:class:`~seestar.gui_qt.settings_state.QtSettingsState`.

Import hygiene: like the rest of :mod:`seestar.gui_qt`, this module never
imports the scientific engine or the Tk GUI at import time.  The public solver
boundary is reached lazily through :mod:`seestar.gui_qt.solver_service`, whose
engine module paths are assembled from split string literals.

Deliberate divergence from the Tk reference (documented): instead of a modal
message box, validation and configuration failures are surfaced through an
in-dialog, non-blocking error label.  This is functionally equivalent (a clear
UI result) but keeps the dialog headless-testable without needing to drive a
blocking modal message box.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .settings_state import QtSettingsState
from .solver_service import (
    check_zesolver_readiness as _default_check_readiness,
    open_zesolver_configuration as _default_open_configuration,
    zesolver_session_refresh_action as _default_session_refresh,
    zesolver_ui_state as _default_ui_state,
)

# Solver preference identifiers (exact, stable — mirrors Tk + backend keys).
SOLVER_NONE = "none"
SOLVER_ASTAP = "astap"
SOLVER_ZESOLVER = "zesolver"
SOLVER_CHOICES = (SOLVER_NONE, SOLVER_ASTAP, SOLVER_ZESOLVER)

# Interval (ms) between lifecycle polls while a ZeSolver configuration session
# is still running (single check per tick, never a busy loop).
_ZESOLVER_REFRESH_TICK_MS = 250


class _FallbackUiState:
    """Conservative, engine-free UI state used when the boundary raises."""

    label = "ZeSolver : inutilisable"
    status_color = "gray"
    show_configuration_button = False


class SolverSettingsDialog(QDialog):
    """Modal solver-configuration dialog (None / ASTAP / ZeSolver).

    Construct with the current :class:`QtSettingsState`; on OK the collected
    values are available via :meth:`values` and can be written back into a
    state object via :meth:`apply_to`.  The ZeSolver status label and
    configuration button are driven by the injected (or default, lazy) public
    boundary so the dialog never imports the engine itself.
    """

    def __init__(
        self,
        parent: Optional[QWidget],
        initial: QtSettingsState,
        *,
        check_readiness: Optional[Callable[[], Any]] = None,
        ui_state_fn: Optional[Callable[[Any], Any]] = None,
        open_configuration: Optional[Callable[[], Any]] = None,
        session_refresh_action: Optional[Callable[[Any], str]] = None,
        refresh_tick_ms: int = _ZESOLVER_REFRESH_TICK_MS,
    ):
        super().__init__(parent)
        self._check_readiness = check_readiness or _default_check_readiness
        self._ui_state_fn = ui_state_fn or _default_ui_state
        self._open_configuration = (
            open_configuration or _default_open_configuration
        )
        self._session_refresh_action = (
            session_refresh_action or _default_session_refresh
        )
        self._refresh_tick_ms = refresh_tick_ms
        self._result: Optional[dict] = None

        self.setWindowTitle("Local Astrometry Solvers Configuration")
        self.setModal(True)

        self._initial_choice = self._normalise_choice(
            getattr(initial, "local_solver_preference", SOLVER_NONE)
        )
        self._initial_astap_path = str(getattr(initial, "astap_path", "") or "")
        self._initial_astap_data_dir = str(
            getattr(initial, "astap_data_dir", "") or ""
        )
        self._initial_search_radius = self._as_float(
            getattr(initial, "astap_search_radius", 3.0), 3.0
        )
        self._initial_downsample = self._as_int(
            getattr(initial, "astap_downsample", 1), 1
        )
        self._initial_sensitivity = self._as_int(
            getattr(initial, "astap_sensitivity", 100), 100
        )

        self._build_ui()
        self._refresh_zesolver_status()
        self._update_astap_enabled()

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _normalise_choice(value: Any) -> str:
        text = str(value or SOLVER_NONE).strip().lower()
        return text if text in SOLVER_CHOICES else SOLVER_NONE

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------- UI build
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Solver choice (None / ASTAP / ZeSolver) ---
        choice_group = QGroupBox("Solver")
        choice_layout = QVBoxLayout(choice_group)
        self.none_radio = QRadioButton("No solver")
        self.astap_radio = QRadioButton("ASTAP")
        self.zesolver_radio = QRadioButton("ZeSolver")
        for radio in (self.none_radio, self.astap_radio, self.zesolver_radio):
            choice_layout.addWidget(radio)
            radio.toggled.connect(self._on_choice_changed)
        layout.addWidget(choice_group)

        # Initial selection (radios share a parent -> mutually exclusive).
        {
            SOLVER_NONE: self.none_radio,
            SOLVER_ASTAP: self.astap_radio,
            SOLVER_ZESOLVER: self.zesolver_radio,
        }[self._initial_choice].setChecked(True)

        # --- ZeSolver status display (read-only) + configure action ---
        self.zesolver_status_label = QLabel("")
        layout.addWidget(self.zesolver_status_label)

        self.configure_zesolver_button = QPushButton("Configurer ZeSolver")
        self.configure_zesolver_button.clicked.connect(self._on_configure_zesolver)
        layout.addWidget(self.configure_zesolver_button)

        # --- ASTAP fallback configuration ---
        self.astap_frame = QGroupBox("ASTAP — fallback autonome")
        astap_layout = QVBoxLayout(self.astap_frame)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Executable:"))
        self.astap_path_edit = QLineEdit(self._initial_astap_path)
        path_row.addWidget(self.astap_path_edit, 1)
        self.astap_path_browse = QPushButton("Browse...")
        self.astap_path_browse.clicked.connect(self._browse_astap_path)
        path_row.addWidget(self.astap_path_browse)
        astap_layout.addLayout(path_row)

        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("Data Dir:"))
        self.astap_data_edit = QLineEdit(self._initial_astap_data_dir)
        data_row.addWidget(self.astap_data_edit, 1)
        self.astap_data_browse = QPushButton("Browse...")
        self.astap_data_browse.clicked.connect(self._browse_astap_data_dir)
        data_row.addWidget(self.astap_data_browse)
        astap_layout.addLayout(data_row)

        numeric_form = QFormLayout()
        self.search_radius_spin = QDoubleSpinBox()
        self.search_radius_spin.setRange(0.1, 90.0)
        self.search_radius_spin.setSingleStep(0.5)
        self.search_radius_spin.setDecimals(1)
        self.search_radius_spin.setValue(self._initial_search_radius)
        numeric_form.addRow("ASTAP Search Radius (deg):", self.search_radius_spin)

        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 8)
        self.downsample_spin.setValue(self._initial_downsample)
        numeric_form.addRow("Downsample:", self.downsample_spin)

        self.sensitivity_spin = QSpinBox()
        self.sensitivity_spin.setRange(10, 1000)
        self.sensitivity_spin.setSingleStep(5)
        self.sensitivity_spin.setValue(self._initial_sensitivity)
        numeric_form.addRow("Sensitivity:", self.sensitivity_spin)

        astap_layout.addLayout(numeric_form)
        layout.addWidget(self.astap_frame)

        # --- Non-blocking error / result label (replaces Tk messagebox) ---
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        # --- OK / Cancel ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self._on_ok)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    # ------------------------------------------------------- choice / enable
    def _current_choice(self) -> str:
        if self.astap_radio.isChecked():
            return SOLVER_ASTAP
        if self.zesolver_radio.isChecked():
            return SOLVER_ZESOLVER
        return SOLVER_NONE

    def _on_choice_changed(self, *_args) -> None:
        self._update_astap_enabled()

    def _update_astap_enabled(self) -> None:
        """Enable the ASTAP frame for ``astap`` and ``zesolver`` (fallback),
        disable it for ``none`` — mirroring the Tk radiobutton gating."""
        if not hasattr(self, "astap_frame"):
            # Called from a radio ``toggled`` fired during construction, before
            # the ASTAP frame exists; the final state is set at the end of
            # ``__init__``.
            return
        enabled = self._current_choice() in (SOLVER_ASTAP, SOLVER_ZESOLVER)
        self.astap_frame.setEnabled(enabled)

    # ------------------------------------------------------- ZeSolver status
    def _safe_ui_state(self):
        try:
            discovery = self._check_readiness()
            return self._ui_state_fn(discovery)
        except Exception:
            return _FallbackUiState()

    def _refresh_zesolver_status(self) -> None:
        ui = self._safe_ui_state()
        self.zesolver_status_label.setText(ui.label)
        self.zesolver_status_label.setStyleSheet(f"color: {ui.status_color};")
        visible = bool(ui.show_configuration_button)
        self.configure_zesolver_button.setVisible(visible)
        self.configure_zesolver_button.setEnabled(visible)

    def _on_configure_zesolver(self) -> None:
        """Launch the public ZeSolver configuration UI (never raises)."""
        try:
            ok, payload = self._open_configuration()
        except Exception as exc:  # defensive: injected fake may raise
            ok, payload = False, f"{type(exc).__name__}: {exc}"

        if not ok:
            self._show_error(
                str(payload or "Impossible de configurer ZeSolver.")
            )
            self._refresh_zesolver_status()
            return
        self._refresh_zesolver_status()
        # Deferred refresh: re-evaluate readiness once the opaque session ends
        # (no-op for API v1.1, which returns no handle).
        self._schedule_zesolver_refresh(payload)

    def _schedule_zesolver_refresh(self, handle) -> None:
        """Poll the config-session lifecycle and refresh once it ends.

        Mirrors the Tk recursive ``after()`` using a re-armed ``QTimer``: one
        check per tick, never a busy loop.  The pure decision lives in
        ``zesolver_session_refresh_action``; an unobservable handle yields
        ``"none"`` and no work.
        """
        try:
            action = self._session_refresh_action(handle)
        except Exception:
            action = "none"
        if action == "refresh":
            self._refresh_zesolver_status()
        elif action == "wait":
            QTimer.singleShot(
                self._refresh_tick_ms,
                lambda: self._schedule_zesolver_refresh(handle),
            )

    # ------------------------------------------------------------- browsing
    def _browse_astap_path(self) -> None:
        current = self.astap_path_edit.text().strip()
        if current and os.path.exists(os.path.dirname(current)):
            start_dir = os.path.dirname(current)
        else:
            start_dir = os.path.expanduser("~")
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select ASTAP Executable", start_dir
        )
        if filepath:
            self.astap_path_edit.setText(filepath)

    def _browse_astap_data_dir(self) -> None:
        current = self.astap_data_edit.text().strip()
        start_dir = current if current and os.path.isdir(current) else os.path.expanduser("~")
        dirpath = QFileDialog.getExistingDirectory(
            self, "Select ASTAP Star Index Data Directory", start_dir
        )
        if dirpath:
            self.astap_data_edit.setText(dirpath)

    # --------------------------------------------------------------- OK/cancel
    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)

    def _on_ok(self) -> None:
        """Validate and, if valid, capture the values and accept the dialog.

        ASTAP primary with an empty executable path is rejected in-dialog
        (matching the Tk behaviour) with a clear, non-blocking UI result.
        """
        self.error_label.setText("")
        choice = self._current_choice()
        astap_path = self.astap_path_edit.text().strip()

        if choice == SOLVER_ASTAP and not astap_path:
            self._show_error(
                "ASTAP is selected, but the executable path is missing."
            )
            return

        self._result = {
            "local_solver_preference": choice,
            "astap_path": astap_path,
            "astap_data_dir": self.astap_data_edit.text().strip(),
            "astap_search_radius": self.search_radius_spin.value(),
            "astap_downsample": self.downsample_spin.value(),
            "astap_sensitivity": self.sensitivity_spin.value(),
        }
        self.accept()

    def values(self) -> dict:
        """Return the captured values (valid only after the dialog was accepted)."""
        if self._result is None:
            raise RuntimeError("values() called before the dialog was accepted")
        return dict(self._result)

    def apply_to(self, state: QtSettingsState) -> None:
        """Write the captured values into a :class:`QtSettingsState`."""
        if self._result is None:
            raise RuntimeError("apply_to() called before the dialog was accepted")
        for key, value in self._result.items():
            setattr(state, key, value)
