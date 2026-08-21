"""QApplication lifecycle helpers for the non-default PySide6 shell."""

from __future__ import annotations

import sys
from typing import Callable, Optional, Sequence

from PySide6.QtWidgets import QApplication

from .backend_runner import BaseRunBackend
from .main_window import DEFAULT_BACKEND_MODE, MainWindow
from .settings_persistence import resolve_settings_path


def create_application(argv: Optional[Sequence[str]] = None) -> QApplication:
    """Return the process-wide QApplication, creating it if necessary.

    Safe to call multiple times (returns the existing instance).  Passing an
    explicit ``argv`` is useful in tests; otherwise ``sys.argv`` is used.
    """
    app = QApplication.instance()
    if app is None:
        args = list(argv) if argv is not None else list(sys.argv)
        app = QApplication(args)
    return app


def run_qt_app(
    argv: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    backend_factory: Optional[Callable[[], BaseRunBackend]] = None,
    backend_mode: str = DEFAULT_BACKEND_MODE,
    settings_path: Optional[str] = None,
) -> int:
    """Build the shell window and enter the Qt event loop.

    ``backend_mode`` (``"simulated"`` or ``"seestar"``) and/or
    ``backend_factory`` select how the window's Start button resolves a run
    backend; the default is the safe simulated runner.  Returns the
    :meth:`QApplication.exec` exit code.  Closing the window ends the loop
    (Qt's ``quitOnLastWindowClosed`` default).  ``settings_path`` selects the
    settings/geometry JSON file; ``None`` (the default) resolves the
    platform-aware user-config path (migrating a legacy CWD
    ``seestar_settings.json`` non-destructively) so the non-default Qt shell
    persists settings across launches regardless of the working directory.
    Pass an explicit path to override.
    """
    app = create_application(argv)
    if settings_path is None:
        settings_path = resolve_settings_path()
    window = MainWindow(
        title=title,
        backend_factory=backend_factory,
        backend_mode=backend_mode,
        settings_path=settings_path,
    )
    window.show()
    return app.exec()
