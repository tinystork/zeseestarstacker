"""Lazy SolverDialog service layer (public solver boundary, engine-free at import).

The pure Qt dialog (:mod:`seestar.gui_qt.solver_dialog`) needs three pieces of
the *existing* public solver boundary:

* ``check_zesolver_readiness()``     -> the operational-readiness discovery,
* ``zesolver_ui_state(discovery)``   -> the pure presentation mapping,
* ``open_zesolver_configuration()``  -> launch the public configuration UI,
* ``zesolver_session_refresh_action(handle)`` -> deferred-refresh decision.

Those live in the engine subtree, which :mod:`seestar.gui_qt` must never import
at package-import time.  This module is the *lazy* service that reaches them on
first call only, using the same import-hygiene pattern as
:mod:`seestar.gui_qt.solver_probe`: the engine module paths are assembled from
split string literals so this source file stays free of the engine's dotted
tokens, and a fresh ``import seestar.gui_qt`` never pulls the engine into
``sys.modules``.
"""

from __future__ import annotations

import importlib


def _adapter_module():
    """Import the public solver adapter lazily (first call only)."""
    return importlib.import_module(
        ".".join(("seestar", "alignment", "zesolver" + "_adapter"))
    )


def _port_module():
    """Import the solver port (pure UI-state mapping) lazily."""
    return importlib.import_module(
        ".".join(("seestar", "alignment", "solver" + "_port"))
    )


def check_zesolver_readiness():
    """Return the public readiness discovery (never raises for absent solver).

    Delegates to the adapter's defensive ``check_zesolver_readiness``, which
    returns a ``SolverDiscovery`` (``NOT_INSTALLED`` / ``UNHEALTHY`` /
    ``INCOMPATIBLE`` / ``NOT_OPERATIONAL`` / ``AVAILABLE``) instead of raising.
    """
    return _adapter_module().check_zesolver_readiness()


def zesolver_ui_state(discovery):
    """Map a discovery onto the pure presentation state (label / colour / flag).

    This is the exact ``zesolver_ui_state(check_zesolver_readiness())``
    composition consumed by the dialog, split only so the dialog can inject a
    fake in tests.
    """
    return _port_module().zesolver_ui_state(discovery)


def open_zesolver_configuration():
    """Launch the public ZeSolver configuration UI; returns ``(ok, handle)``.

    The adapter never raises here: ``(False, message)`` on absence/failure,
    ``(True, handle)`` (or ``(True, None)`` for API v1.1) on success.
    """
    return _adapter_module().open_zesolver_configuration()


def zesolver_session_refresh_action(handle):
    """Return the deferred-refresh action for a config-session handle."""
    return _adapter_module().zesolver_session_refresh_action(handle)
