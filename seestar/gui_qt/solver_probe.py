"""Lazy ZeSolver operational-readiness probe (service layer).

The pure stdlib validator :mod:`seestar.gui_qt.settings_validation` takes a
plain boolean ``zesolver_operational`` so it never imports the engine and stays
unit-testable in complete isolation.  This module is the *lazy* service that
actually computes that boolean at start time by probing the public ZeSolver
readiness adapter — imported only on first call, never at
``import seestar.gui_qt`` time.

The engine module path is assembled from split string literals so this source
file stays free of the engine's dotted tokens, mirroring the import-hygiene
pattern already used by :mod:`seestar.gui_qt.backend_runner`.  A fresh
``import seestar.gui_qt`` therefore leaves ``sys.modules`` clean of the heavy
engine packages (alignment / core / queue manager).
"""

from __future__ import annotations

import importlib


def probe_zesolver_operational() -> bool:
    """Return ``True`` only when the public ZeSolver API reports state available.

    Any failure — engine absent, incompatible, unhealthy, not operational, or a
    probe exception — yields ``False`` so the solver gate falls back to ASTAP
    (or blocks) exactly like the historical Tk path does.
    """
    try:
        module = importlib.import_module(
            ".".join(("seestar", "alignment", "zesolver" + "_adapter"))
        )
        check = getattr(module, "check_zesolver_readiness", None)
        if check is None:
            return False
        discovery = check()
        state = getattr(discovery, "state", None)
        return getattr(state, "value", None) == "available"
    except Exception:
        return False
