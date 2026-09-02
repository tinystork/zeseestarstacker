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
  array and never logs per-pixel data.  Counters are computed on that same
  bounded sample: ``n`` (finite sample size), ``under_n`` / ``over_n`` /
  ``zero_n`` / ``one_n`` with dtype-aware bounds — for float arrays the
  canonical *analysis* domain is ``[0, 1]`` for counter purposes (under
  ``< 0``, over ``> 1``, zero ``== 0``, one ``== 1``); for integer arrays the
  bounds are the dtype range (unsigned: ``[0, max]``, so for a uint8 display
  buffer ``one_n`` counts the ``== 255`` saturated pixels and
  ``over_n``/``under_n`` are always 0);

  PHI-R3 semantics: analysis stages no longer hard-clip to ``[0, 1]`` — the
  anchor mapping and the WB derivation preserve finite out-of-range float
  headroom, so ``over_n > 0`` at ``anchor_mapped``/``wb_only``/``raw_source``
  means **preserved analysis headroom**, not a clip artifact, and ``one_n``
  counts only *exact* ``== 1.0`` values.  Only the final display-rendering
  boundary is bounded: the uint8 display stages carry ``one_n`` as the
  ``== 255`` saturated-pixel count (display saturation), so a witness can
  always distinguish analysis headroom from display-domain saturation;
* arrival records carry the PHI-R3 monotonic acceptance outcome: an accepted
  payload_arrive record has no ``drop`` field, a payload refused by the
  run-scoped sequence gate carries ``drop=stale`` (older emission) or
  ``drop=duplicate`` (repeated emission);
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

    Counters (all on the bounded finite sample, so unambiguous and bounded):

    * ``n``        — number of finite sampled values;
    * ``under_n``  — count below the domain lower bound (float: ``< 0``;
      unsigned int: always 0);
    * ``over_n``   — count above the domain upper bound (float: ``> 1``;
      integer: ``> dtype max``, i.e. 0 for uint8);
    * ``zero_n``   — count ``== 0``;
    * ``one_n``    — count equal to the domain upper bound (float: ``== 1``;
      uint8: ``== 255`` saturated display pixels).

    For float ``[0, 1]`` analysis stages ``under_n``/``over_n``/``zero_n``/
    ``one_n`` directly answer the plateau/headroom question: headroom is
    ``over_n > 0`` at ``raw_source``, and the first stage with ``one_n > 0``
    (and ``over_n == 0``) is the first clip site.
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
        kind = getattr(arr.dtype, "kind", "f")
        if kind in ("u", "i"):
            info = np.iinfo(arr.dtype)
            lo, hi = int(info.min), int(info.max)
            under_n = int(np.count_nonzero(finite < lo))
            over_n = int(np.count_nonzero(finite > hi))
            zero_n = int(np.count_nonzero(finite == 0))
            one_n = int(np.count_nonzero(finite == hi))
        else:  # float / complex / bool-ish: canonical [0, 1] domain
            under_n = int(np.count_nonzero(finite < 0.0))
            over_n = int(np.count_nonzero(finite > 1.0))
            zero_n = int(np.count_nonzero(finite == 0.0))
            one_n = int(np.count_nonzero(finite == 1.0))
        return {
            "dtype": str(arr.dtype),
            "shape": "x".join(str(s) for s in arr.shape),
            "min": f"{float(np.min(finite)):.6g}",
            "p01": f"{float(p01):.6g}",
            "median": f"{float(median):.6g}",
            "p99": f"{float(p99):.6g}",
            "max": f"{float(np.max(finite)):.6g}",
            "n": str(int(finite.size)),
            "under_n": str(under_n),
            "over_n": str(over_n),
            "zero_n": str(zero_n),
            "one_n": str(one_n),
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
        ``req=2``, ``cap=1``, ``src=SUM/W``, ``src_id=...``, ``seq=...``,
        ``identity=...``, ``res=...``, ``preq=...``, ``pres=...``, ``pcap=...``,
        ``drop=stale|duplicate`` for gated payload drops).

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
