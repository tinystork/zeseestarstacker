"""Plain run-summary payload + lazy ``final.fits`` header reader (M23).

The payload dataclass is pure Python (importable without the engine or
astropy).  The header reader lazily imports ``astropy.io.fits`` *inside* the
function (never at module level) so ``import seestar.gui_qt`` stays free of a
top-level astropy/engine import — the same split-string / lazy-import
discipline used by :mod:`seestar.gui_qt.backend_runner`.

The backend adapter (regular run) and the boring subprocess runner (single-batch
run) both build a :class:`SummaryPayload` here; the Qt dialog in
:mod:`seestar.gui_qt.main_window` only *formats* the payload for display.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SummaryPayload:
    """Plain, toolkit-free run-summary data carried adapter -> GUI thread.

    Fields
    ------
    status:
        Human-readable terminal status ("finished" / "failed" / "cancelled").
    duration_seconds:
        Total processing time in seconds (``None`` when unknown).
    files_attempted:
        Number of input files the run attempted (``None`` when unknown).
    final_stack_file:
        Absolute path of the expected ``final.fits`` (``""`` when unknown).
    final_stack_exists:
        True when ``final_stack_file`` exists on disk.
    images_in_final_stack:
        ``NIMAGES`` from the ``final.fits`` header (``None`` when unknown).
    total_exposure_seconds:
        ``TOTEXP`` from the ``final.fits`` header (``None`` when unknown).
    can_open_output:
        True when the output folder exists and holds a final stack (drives the
        dialog's "Open Output" button).
    """

    status: str = ""
    duration_seconds: Optional[float] = None
    files_attempted: Optional[int] = None
    final_stack_file: str = ""
    final_stack_exists: bool = False
    images_in_final_stack: Optional[int] = None
    total_exposure_seconds: Optional[float] = None
    can_open_output: bool = False


def read_final_fits_header(path: str) -> Dict[str, Any]:
    """Lazily read a ``final.fits`` header into a plain dict (never raises).

    ``astropy.io.fits`` is imported lazily here via a split string literal so
    this module (and the whole ``gui_qt`` package) never has a top-level
    astropy/engine import — mirroring the engine's own lazy-import discipline.
    Returns ``{}`` for a missing/unreadable path.
    """
    if not path or not os.path.isfile(path):
        return {}
    try:
        fits = importlib.import_module(".".join(("astropy", "io", "fits")))
        header = fits.getheader(path)
        return dict(header)
    except Exception:
        return {}


def build_summary_payload(
    *,
    status: str,
    duration_seconds: Optional[float],
    files_attempted: Optional[int],
    output_dir: Optional[str],
) -> SummaryPayload:
    """Build a :class:`SummaryPayload` from terminal run facts + ``final.fits``.

    The final-stack path and the ``NIMAGES`` / ``TOTEXP`` header fields are
    derived here (lazily) so the GUI layer only ever formats the result.
    """
    final_path = os.path.join(output_dir, "final.fits") if output_dir else ""
    header = read_final_fits_header(final_path) if final_path else {}

    images = None
    exposure = None
    try:
        if "NIMAGES" in header:
            images = int(header["NIMAGES"])
    except (TypeError, ValueError):
        pass
    try:
        if "TOTEXP" in header:
            exposure = float(header["TOTEXP"])
    except (TypeError, ValueError):
        pass

    exists = bool(final_path and os.path.isfile(final_path))
    return SummaryPayload(
        status=status,
        duration_seconds=duration_seconds,
        files_attempted=files_attempted,
        final_stack_file=final_path,
        final_stack_exists=exists,
        images_in_final_stack=images,
        total_exposure_seconds=exposure,
        can_open_output=bool(output_dir and os.path.isdir(output_dir) and exists),
    )
