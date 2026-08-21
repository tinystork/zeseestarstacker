"""PySide6 (Qt) GUI shell package — non-default.

This package contains the minimal PySide6 shell used as an architectural
foothold for the Tk → Qt parity migration.  It is kept separate from
:mod:`seestar.gui` (the Tk GUI) so the two toolkits are never mixed in a
single module.

Importing this package pulls in PySide6.  It does NOT import the Tk GUI and
does NOT touch the scientific engine.  The run-request builder
(``run_config.py``) is reached through :mod:`seestar.gui_qt.run_bridge`, which
re-exports the canonical :mod:`seestar.gui.run_config` module.  Because the
``seestar`` and ``seestar.gui`` package inits are lazy, neither the engine nor
Tk is ever pulled in.  The real stacker engine is reachable *only* through the
lazy :class:`~seestar.gui_qt.backend_runner.SeestarQueuedStackerBackend`, which
imports it on first ``run()``, not at import time.
"""

from __future__ import annotations

from .app import create_application, run_qt_app
from .backend_runner import (
    BackendPreviewPayload,
    BackendRunResult,
    BaseRunBackend,
    SeestarQueuedStackerBackend,
    SimulatedRunBackend,
)
from .main_window import (
    BACKEND_MODES,
    DEFAULT_BACKEND_MODE,
    DEFAULT_TITLE,
    DRIZZLE_MODES,
    SOLVER_PREFERENCES,
    STACKING_MODES,
    MainWindow,
)
from .run_controller import RunController
from .run_worker import RunStatus, RunWorker
from .settings_validation import normalize_batch_size, validate_settings_for_backend
from .settings_state import QtSettingsState
from .solver_probe import probe_zesolver_operational

__all__ = [
    "MainWindow",
    "BACKEND_MODES",
    "DEFAULT_BACKEND_MODE",
    "DEFAULT_TITLE",
    "QtSettingsState",
    "RunController",
    "RunStatus",
    "RunWorker",
    "BaseRunBackend",
    "BackendPreviewPayload",
    "BackendRunResult",
    "SimulatedRunBackend",
    "SeestarQueuedStackerBackend",
    "validate_settings_for_backend",
    "normalize_batch_size",
    "probe_zesolver_operational",
    "STACKING_MODES",
    "DRIZZLE_MODES",
    "SOLVER_PREFERENCES",
    "create_application",
    "run_qt_app",
]
