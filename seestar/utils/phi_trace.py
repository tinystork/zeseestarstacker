"""PHI-R2 debug-gated preview-stage trace helper (display-only).

Instrumentation contract (PHI-R2, ``docs/phi_viewer_archaeology.md`` section 7):

* one clearly documented debug gate: the ``ZSSS_PHI_TRACE`` environment
  variable.  Any non-empty value other than ``0`` / ``false`` / ``no`` /
  ``off`` (case-insensitive) enables tracing; default (unset) is **disabled**,
  and every function is then a no-op that returns without importing numpy or
  touching any array;
* when enabled, :func:`phi_trace_stage` emits one *compact single-line*
  record prefixed ``PREVIEW_STAGE`` at ``logger.debug`` level.  The record
  identifies at least: source route, stage, dtype, shape, min, p01, median,
  p99, max, plus caller-supplied fields (preview factor, source buffer
  identity, monotonic sequence, ...).  Numbers use ``%.6g`` formatting;
* stage statistics are computed on a deterministic fixed-stride subsample
  (at most ``_MAX_STATS_SAMPLE`` elements) so tracing never copies a whole
  array and never logs per-pixel data;
* the helper is deliberately pure and reusable: no science imports, no Qt
  imports, no array mutation — it only reads.  numpy is imported *lazily*,
  inside the functions, and only when the gate is on.

The exact string format is not a contract; the fields and the ``PREVIEW_STAGE``
prefix are.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

_PHI_TRACE_ENV = "ZSSS_PHI_TRACE"
_PHI_TRACE_PREFIX = "PREVIEW_STAGE"
_MAX_STATS_SAMPLE = 1_000_000

_FALSY = ("", "0", "false", "no", "off")


def phi_trace_enabled() -> bool:
    """Return ``True`` when the ``ZSSS_PHI_TRACE`` debug gate is enabled."""
    return os.environ.get(_PHI_TRACE_ENV, "").strip().lower() not in _FALSY


def _load_numpy():
    """Lazily import numpy (only called when the gate is on)."""
    try:
        import importlib

        return importlib.import_module("numpy")
    except Exception:
        return None


def _stage_stats(np, arr) -> Optional[Dict[str, str]]:
    """Return compact deterministic stats for ``arr`` (or ``None``).

    ``arr`` is a 2D/3D numeric array; stats are computed on a fixed-stride
    subsample (``arr.flat[::stride]``, at most ``_MAX_STATS_SAMPLE`` elements)
    so no whole-array copy is made for tracing.  Values are compact strings.
    """
    if arr is None or not hasattr(arr, "ndim"):
        return None
    try:
        if arr.ndim not in (2, 3) or arr.size == 0:
            return None
        stride = max(1, arr.size // _MAX_STATS_SAMPLE)
        sample = arr.flat[::stride]
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            return None
        p01, median, p99 = np.percentile(finite, [1.0, 50.0, 99.0])
        return {
            "dtype": str(arr.dtype),
            "shape": "x".join(str(s) for s in arr.shape),
            "min": f"{float(np.min(finite)):.6g}",
            "p01": f"{float(p01):.6g}",
            "median": f"{float(median):.6g}",
            "p99": f"{float(p99):.6g}",
            "max": f"{float(np.max(finite)):.6g}",
        }
    except Exception:
        return None


def phi_trace_stage(logger, *, route: str, stage: str, arr: Any = None, **extra) -> None:
    """Emit one compact ``PREVIEW_STAGE`` record at ``logger.debug`` (gated).

    Parameters
    ----------
    logger:
        The calling module's ``logging.Logger`` (records propagate normally).
    route:
        Source route label (``classic`` / ``drizzle`` / ``qt`` /
        ``legacy_drizzle`` ...).
    stage:
        Pipeline stage name (``source`` / ``pre_resize`` / ``post_resize`` /
        ``payload_arrive`` / ``raw_source`` / ``anchor_mapped`` / ``wb_only`` /
        ``stretch_input`` / ``stretch_output`` / ``qimage`` ...).
    arr:
        Optional array to summarize (sampled, never copied, never mutated).
    extra:
        Additional ``key=value`` fields appended verbatim (e.g. ``factor=2``,
        ``src=SUM/W``, ``src_id=...``, ``seq=...``, ``identity=...``, ``res=...``).

    No-op when the gate is disabled; never raises when enabled.
    """
    if not phi_trace_enabled():
        return
    try:
        parts = [f"{_PHI_TRACE_PREFIX} route={route} stage={stage}"]
        np = _load_numpy()
        if np is not None:
            stats = _stage_stats(np, arr)
            if stats is not None:
                parts.extend(f"{k}={v}" for k, v in stats.items())
        parts.extend(f"{k}={v}" for k, v in extra.items())
        logger.debug(" ".join(parts))
    except Exception:
        pass  # tracing is best-effort: it must never perturb production behaviour
