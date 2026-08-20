"""Non-default PySide6 launch helper.

Run explicitly with::

    python -m seestar.qt_main --backend simulated      # default, safe
    python -m seestar.qt_main --backend seestar        # real backend (opt-in)

This does NOT change the default ``zeseestarstacker = seestar.main:main``
entry point; it exists so the experimental Qt shell can be launched during the
migration without disturbing the Tk application.

``--backend`` selects how the Qt shell resolves the run backend for its Start
button.  The default is ``simulated`` (no engine, no real work).  ``seestar``
is an explicit opt-in that configures the lazy
:class:`~seestar.gui_qt.backend_runner.SeestarQueuedStackerBackend`; the real
engine is still imported only when a run is actually started.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence, Tuple

from .gui_qt.app import run_qt_app

BACKEND_CHOICES = ("simulated", "seestar")
DEFAULT_BACKEND_MODE = "simulated"


def parse_qt_args(argv: Optional[Sequence[str]] = None) -> Tuple[str, list]:
    """Extract ``--backend {simulated|seestar}`` from ``argv``.

    Returns ``(backend_mode, remaining_argv)``.  The mode defaults to
    ``simulated``; the ``--backend``/``--backend=...`` tokens are removed so the
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    backend_mode, remaining = parse_qt_args(argv)
    return run_qt_app(remaining, backend_mode=backend_mode)


if __name__ == "__main__":
    raise SystemExit(main())
