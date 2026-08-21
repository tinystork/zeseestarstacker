"""Engine-free bridge to the canonical run-request builder.

``seestar.gui.run_config`` is the Qt/Tk-independent run-request builder.  It
lives inside the Tk GUI package tree but is pure stdlib (``dataclasses``,
``types``, ``typing``) and imports nothing GUI- or engine-related.

The normal dotted import path is now safe: ``seestar/__init__.py`` and
``seestar/gui/__init__.py`` are lazy and import neither the scientific engine
nor Tk, so this bridge re-exports the *canonical* module/class instead of
loading a private copy under a flat name.

This matters: :class:`RunRequest` here MUST be the same class object as
``seestar.gui.run_config.RunRequest`` so that Qt controller/worker boundaries
compare ``is``-identical types with the Tk side.
"""

from __future__ import annotations

from seestar.gui.run_config import (
    RunRequest,
    build_backend_kwargs,
    build_run_request,
    compute_align_on_disk,
    split_backend_kwargs,
)

__all__ = [
    "RunRequest",
    "build_run_request",
    "build_backend_kwargs",
    "compute_align_on_disk",
    "split_backend_kwargs",
]
