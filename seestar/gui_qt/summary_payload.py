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
    final_stack_path: Optional[str] = None,
) -> SummaryPayload:
    """Build a :class:`SummaryPayload` from terminal run facts + the final stack.

    The final-stack path is the *source of truth* for the run's product.  When
    ``final_stack_path`` is provided (the real FITS the engine wrote, e.g.
    ``stack_final_drizzle_final.fit``), every derived field --
    :attr:`SummaryPayload.final_stack_file`, ``final_stack_exists``,
    ``images_in_final_stack``, ``total_exposure_seconds`` and
    ``can_open_output`` -- is computed from that path.  The GUI must never
    maintain a fragile list of output filenames.

    When ``final_stack_path`` is omitted (legacy callers, the boring runner),
    it falls back to the historical ``<output_dir>/final.fits`` convention.
    """
    if final_stack_path:
        final_path = final_stack_path
    else:
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
    # The openable folder is derived from the final-stack path when present
    # (the file may live outside the requested output dir), else from the
    # requested output dir (legacy ``final.fits`` fallback).
    open_dir = os.path.dirname(final_path) if final_path else (output_dir or "")
    return SummaryPayload(
        status=status,
        duration_seconds=duration_seconds,
        files_attempted=files_attempted,
        final_stack_file=final_path,
        final_stack_exists=exists,
        images_in_final_stack=images,
        total_exposure_seconds=exposure,
        can_open_output=bool(open_dir and os.path.isdir(open_dir) and exists),
    )


def derive_terminal_status(payload: Optional[SummaryPayload]) -> str:
    """Derive the presentation-layer terminal status from a summary payload.

    Returns ``"success"`` when the run produced a final stack with at least one
    accumulated image *and* an openable output folder; returns ``"empty"``
    otherwise (no payload at all, a missing ``final.fits``, zero images, or an
    unopenable output folder).

    This keeps :attr:`SummaryPayload.status` free to carry the backend's raw
    terminal label (``"finished"``) while the presentation layer decides
    whether that ``"finished"`` is a real success or an empty/no-output run.
    The helper is pure stdlib so it stays importable and unit-testable without
    Qt, the engine, or astropy.
    """
    if payload is None:
        return "empty"
    if not payload.final_stack_exists:
        return "empty"
    if payload.images_in_final_stack in (None, 0):
        return "empty"
    if not payload.can_open_output:
        return "empty"
    return "success"
