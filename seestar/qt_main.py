"""Official Qt (PySide6) entry point for ZeSeestarStacker.

This module is the ``gui-scripts`` console entry point
(``zeseestarstacker = seestar.qt_main:main``).  The default backend is the real
``seestar`` engine; ``--backend simulated`` remains an explicit dev/test option
that runs the safe no-engine simulated runner.

::

    zeseestarstacker                                  # real backend (default)
    python -m seestar.qt_main --backend seestar       # real backend (explicit)
    python -m seestar.qt_main --backend simulated     # safe, no engine

``--backend`` selects how the Qt shell resolves the run backend for its Start
button.  The default is ``seestar``, which configures the lazy
:class:`~seestar.gui_qt.backend_runner.SeestarQueuedStackerBackend`; the real
engine is still imported only when a run is actually started.

A tiny env-only startup witness (``ZSSS_QT_STARTUP_WITNESS=1``) is available
for CI / smoke testing: it creates the application and window offscreen, proves
the title/version, the resolved default backend class and the packaged visual
resources, prints ``ZSSS_QT_STARTUP_WITNESS_*`` markers and returns ``0``
without entering the Qt event loop.  Normal launches are unaffected.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Tuple

from .gui_qt.app import run_qt_app

BACKEND_CHOICES = ("simulated", "seestar")
DEFAULT_BACKEND_MODE = "seestar"


def parse_qt_args(argv: Optional[Sequence[str]] = None) -> Tuple[str, list]:
    """Extract ``--backend {simulated|seestar}`` from ``argv``.

    Returns ``(backend_mode, remaining_argv)``.  The mode defaults to the real
    ``seestar`` backend; ``--backend simulated`` is the explicit dev/test
    option.  The ``--backend``/``--backend=...`` tokens are removed so the
    remaining arguments can be handed straight to :class:`QApplication`.
    Raises :class:`SystemExit` on a missing or unknown value.
    """
    args = list(sys.argv if argv is None else argv)
    mode = DEFAULT_BACKEND_MODE
    remaining: list = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--backend":
            i += 1
            if i >= len(args):
                raise SystemExit("--backend requires a value: simulated|seestar")
            mode = args[i]
        elif arg.startswith("--backend="):
            mode = arg.split("=", 1)[1]
        else:
            remaining.append(arg)
        i += 1
    if mode not in BACKEND_CHOICES:
        raise SystemExit(
            f"invalid --backend value {mode!r} (choose simulated|seestar)"
        )
    return mode, remaining


def _run_startup_witness() -> int:
    """Env-only offscreen smoke test (``ZSSS_QT_STARTUP_WITNESS=1``).

    Creates the :class:`QApplication` and a real ``MainWindow`` on the offscreen
    platform, proves the window title/version, the resolved default backend
    class and the packaged visual resources, prints ``ZSSS_QT_STARTUP_WITNESS_*``
    markers, shuts down and returns ``0`` without ever entering the Qt event
    loop.  Normal launches never reach this path.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from .gui_qt.app import create_application
    from .gui_qt.main_window import MainWindow, product_version
    from .gui_qt.resources import load_empty_preview_pixmap, load_window_icon

    app = create_application([])
    window = MainWindow(backend_mode=DEFAULT_BACKEND_MODE)
    try:
        backend = window.resolve_backend()
        backend_class = type(backend).__name__ if backend is not None else "None"
        icon = load_window_icon()
        preview = load_empty_preview_pixmap()
        print(f"ZSSS_QT_STARTUP_WITNESS_TITLE={window.windowTitle()!r}")
        print(f"ZSSS_QT_STARTUP_WITNESS_VERSION={product_version()!r}")
        print(f"ZSSS_QT_STARTUP_WITNESS_BACKEND_MODE={DEFAULT_BACKEND_MODE!r}")
        print(f"ZSSS_QT_STARTUP_WITNESS_BACKEND_CLASS={backend_class}")
        print(
            "ZSSS_QT_STARTUP_WITNESS_ICON="
            f"{'present' if icon is not None and not icon.isNull() else 'absent'}"
        )
        print(
            "ZSSS_QT_STARTUP_WITNESS_PREVIEW="
            f"{'present' if preview is not None and not preview.isNull() else 'absent'}"
        )
    finally:
        window.shutdown()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    if os.environ.get("ZSSS_QT_STARTUP_WITNESS") == "1":
        return _run_startup_witness()
    backend_mode, remaining = parse_qt_args(argv)
    return run_qt_app(remaining, backend_mode=backend_mode)


if __name__ == "__main__":
    raise SystemExit(main())
