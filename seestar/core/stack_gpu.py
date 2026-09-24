"""CuPy reduction kernels for the sorting-based stacking reductions.

GPU twins of the CPU scientific reference in :mod:`seestar.core.stack_methods`
(which stays untouched).  These kernels reproduce the EXACT CPU algorithm:

* NaN == missing sample,
* median / std via ``cp.nanmedian`` / ``cp.nanstd`` (ddof=0, NaN excluded),
* identical ``mask = valid & (arr >= low) & (arr <= high)`` (kappa-sigma) or
  ``valid & (|residuals - med_res| <= sigma * std_res)`` (linear-fit clip),
* identical ``1e-6`` division floor (weighted) / all-masked ``0.0`` column
  fix-up (unweighted),
* identical ``_rejected_pct`` formula computed over the valid samples,
* identical ``count_nonzero``-based weight map.

Only kappa-sigma, linear-fit-clip, median and winsorized-sigma are
accelerated: they are the sorting-heavy reductions that profit on GPU
(measured).  Mean stays CPU.

CuPy is imported lazily (module import never requires cupy); every public
function returns plain NumPy arrays (``cp.asnumpy`` before return) — a CuPy
array never crosses the module boundary.

Winsorized-sigma twin (M3, Track B): ``stack_winsorized_sigma_gpu`` is an
exact reproduction of the CPU ``_stack_winsorized_sigma_iter`` NumPy/bottleneck
path (``USE_SCIPY_WINSOR=0``): the same iterative winsorize-and-clip loop,
the same rank-based winsorization (``argsort`` + inverse-permutation rank +
``floor(limit * n_valid)`` inclusive bounds), the same NaN handling (missing
samples stay NaN and are excluded from order statistics), the same
``apply_rewinsor`` survivor-bound substitution, and the same final weighted /
unweighted reduction with the ``1e-6`` floor.  ``seestar/core/stack_methods.py``
is the scientific authority and is never modified.

Small-N exact fast path (Track P1, phase C): for every reduction the per-pixel
valid count satisfies ``n_valid <= N_batch`` (the ACTUAL current reduction
population), so when ``floor(low * N_batch) == floor(high * N_batch) == 0``
the winsor rank is zero on every pixel of every iteration: winsorization is
the identity and the survivor rewinsor bounds degenerate to the survivor
min/max.  In that regime the per-iteration ``argsort``/``take_along_axis``/
inverse-rank ``argsort``/replacement work is skipped entirely (mean/std are
computed straight on the masked stack) and the final ``apply_rewinsor`` bound
pass uses an exact min/max reduction (CPU degenerate behavior preserved:
all-invalid survivor columns keep the ``+inf`` sentinel on both sides).  The
fast path is provably bit-identical to the slow path in that regime and is
only selected on the population bound ``N_batch`` (never on the frozen
``B_resolved``); every other input keeps the original slow path untouched.

Rank/order-temporary elimination (Track P2, phase D): the slow path
(``_winsorize_axis0_cp`` / ``_winsorize_bounds_cp``) no longer materializes
the two full-size int64 temporaries ``order = argsort(sort_key)`` and
``rank = argsort(order)`` (the largest VRAM consumers of the slow path).
Sorted values come from a direct ``cp.sort`` (bitwise the sorted multiset of
``argsort`` + ``take_along_axis``) and replacement is expressed as VALUE
clipping instead of rank replacement:

* the CPU replaces exactly the samples strictly below the ``lowidx`` order
  statistic with ``low_bound`` and strictly above the ``keep_idx`` order
  statistic with ``high_bound`` (rank ``>= upidx``); samples equal to a
  boundary order statistic that the CPU would "replace" keep the identical
  bit pattern, so rank replacement and value clipping agree bit-for-bit;
* the CPU's extreme-limit ``IndexError`` (``lowidx >= N``, e.g. ``(1.0, 0.0)``
  on an all-valid column) is preserved verbatim (explicit out-of-bounds
  check, same message);
* the only case value clipping cannot reproduce is the degenerate OVERLAP
  corner where both sides are active and the two replaced rank ranges
  intersect on a column (``floor(low * n) + floor(high * n) > n`` ->
  ``lowidx > upidx``): there the sequential low-then-high rank replacement
  can split a boundary tie group between the two bounds, which no value
  test can express.  Such inputs fall back to the exact pre-Phase-D
  rank-based implementation (``_winsorize_axis0_rank_path_cp``, retained
  verbatim) so the CPU-defined behavior is preserved for every input;
* ``_winsorize_bounds_cp`` (no rank, clamped ``lowidx``/``highidx``) gets the
  same direct-sort swap with its ``max_idx`` clamp semantics unchanged.

The Phase C zero-rank fast path is untouched: in that regime both functions
are skipped entirely, so there is no interaction.
"""

from __future__ import annotations

import math as _math
import os as _os
import time as _time

import numpy as np

import logging as _logging

logger = _logging.getLogger(__name__)

__all__ = [
    "stack_kappa_sigma_gpu",
    "stack_linear_fit_clip_gpu",
    "stack_median_gpu",
    "stack_winsorized_sigma_gpu",
    "stack_winsorized_sigma_gpu_tiled",
]

_cupy_module = None


# ---------------------------------------------------------------------------
# Opt-in stage profiler (env ZSSS_GPU_PROFILE=1) - recording ONLY.
#
# Every ``_p_*`` call is a hard no-op when the env var is unset, so the
# kernels behave exactly as before (same ops, same order, same values).
# When enabled the probes never alter data, dtypes, control flow or return
# contracts: they only record (a) host monotonic wall stamps for pure-host /
# synchronization segments and (b) stream-ordered CuPy events for device
# stage durations (elapsed between consecutive events after a final sync).
# The standalone profiler driver (profiling/) additionally drives memory
# snapshots through ``snap_mem``.  Nothing here can change reduction
# results; the driver proves that bit-for-bit (instrumented vs plain).
# ---------------------------------------------------------------------------

_PROBE = None


class _StageProbe:
    """Collector for opt-in stage observations (single-threaded use)."""

    def __init__(self, cp):
        self.cp = cp
        self.marks = []  # [(name, wall_seconds, cp.cuda.Event or None)]
        self.wall = []  # [(name, start_s, end_s)] host/sync segments
        self.mem = []  # [(name, free, total, p_used, p_free, p_total)]
        self.notes = []  # free-form run facts (iterations, n_rej, ...)
        self.snap_mem = False
        self._wall_open = None

    def reset(self):
        self.marks = []
        self.wall = []
        self.mem = []
        self.notes = []
        self.snap_mem = False
        self._wall_open = None

    def event(self, name):
        """Record a stream-ordered event marking the end of device work up
        to this point (plus a host wall stamp)."""
        ev = self.cp.cuda.Event()
        ev.record()
        self.marks.append((name, _time.perf_counter(), ev))
        if self.snap_mem:
            self._snapshot(name)

    def wall_start(self, name):
        self._wall_open = (name, _time.perf_counter())

    def wall_end(self):
        if self._wall_open is not None:
            name, t0 = self._wall_open
            self._wall_open = None
            self.wall.append((name, t0, _time.perf_counter()))

    def note(self, text):
        self.notes.append(str(text))

    def _snapshot(self, name):
        """Synchronize and snapshot driver + pool memory state."""
        try:
            self.cp.cuda.Stream.null.synchronize()
            free, total = self.cp.cuda.runtime.memGetInfo()
            pool = self.cp.get_default_memory_pool()
            self.mem.append(
                (
                    name,
                    int(free),
                    int(total),
                    int(pool.used_bytes()),
                    int(pool.free_bytes()),
                    int(pool.total_bytes()),
                )
            )
        except Exception:
            self.mem.append((name, -1, -1, -1, -1, -1))


def _ensure_probe():
    """Create the probe lazily when ZSSS_GPU_PROFILE=1 (GPU required)."""
    global _PROBE
    if _PROBE is None and _os.getenv("ZSSS_GPU_PROFILE") == "1":
        _PROBE = _StageProbe(_get_cupy())
    return _PROBE


def _p_event(name):
    probe = _PROBE
    if probe is not None:
        probe.event(name)


def _p_wall_start(name):
    probe = _PROBE
    if probe is not None:
        probe.wall_start(name)


def _p_wall_end():
    probe = _PROBE
    if probe is not None:
        probe.wall_end()


def _p_note(text):
    probe = _PROBE
    if probe is not None:
        probe.note(text)


def _get_cupy():
    """Import CuPy once, lazily.  Raises ImportError when cupy is absent."""
    global _cupy_module
    if _cupy_module is None:
        import cupy as cp  # noqa: PLC0415 - deliberate lazy import

        _cupy_module = cp
    return _cupy_module


def _stacked(images):
    """Stack images onto the device, mirroring the CPU kernel's first step.

    CPU: ``arr = np.stack([im for im in images], axis=0).astype(np.float32)``
    then transfer once; identical values, identical dtype.
    """
    cp = _get_cupy()
    return cp.asarray(
        np.stack([im for im in images], axis=0).astype(np.float32)
    )


def _broadcast_weights_cp(cp, arr, weights):
    """Device twin of ``stack_methods._broadcast_weights``."""
    w = cp.asarray(weights, dtype=cp.float32)
    shape = (arr.shape[0],) + (1,) * (arr.ndim - 1)
    return w.reshape(shape)


def _rejected_pct_cp(cp, mask, valid):
    """Device twin of ``stack_methods._rejected_pct`` (global float)."""
    n_valid = int(cp.count_nonzero(valid))
    if n_valid == 0:
        return 0.0
    n_surv = int(cp.count_nonzero(mask))
    return 100.0 * (n_valid - n_surv) / float(n_valid)


def _weighted_result(cp, mask, arr, w):
    """Weighted reduction body shared by kappa-sigma and linear-fit clip."""
    wm = cp.where(mask, w, cp.float32(0.0))
    sum_w = cp.sum(wm, axis=0, dtype=cp.float32)
    arr0 = cp.where(mask, arr, cp.float32(0.0))
    sum_d = cp.sum(arr0 * wm, axis=0, dtype=cp.float32)
    result = cp.where(
        sum_w > 1e-6,
        sum_d / cp.maximum(sum_w, 1e-6),
        cp.zeros_like(sum_d),
    )
    return result, sum_w


def _unweighted_result(cp, mask, arr):
    """Unweighted (nanmean over survivors) reduction body."""
    result = cp.nanmean(cp.where(mask, arr, cp.nan), axis=0)
    result = cp.where(cp.any(mask, axis=0), result, cp.float32(0.0))
    sum_w = cp.count_nonzero(mask, axis=0).astype(cp.float32)
    return result, sum_w


def stack_kappa_sigma_gpu(
    images,
    weights=None,
    sigma_low=3.0,
    sigma_high=3.0,
    return_weights=False,
):
    """CuPy twin of ``stack_methods._stack_kappa_sigma``.

    Returns ``(result, rejected_pct)`` or, with ``return_weights=True``,
    ``(result, sum_w, rejected_pct)`` — all arrays NumPy float32, exactly like
    the CPU kernel.
    """
    cp = _get_cupy()
    arr = _stacked(images)
    valid = ~cp.isnan(arr)
    med = cp.nanmedian(arr, axis=0)
    std = cp.nanstd(arr, axis=0)
    low = med - sigma_low * std
    high = med + sigma_high * std
    mask = valid & (arr >= low) & (arr <= high)
    if weights is not None:
        result, sum_w = _weighted_result(
            cp, mask, arr, _broadcast_weights_cp(cp, arr, weights)
        )
    else:
        result, sum_w = _unweighted_result(cp, mask, arr)
    rejected_pct = _rejected_pct_cp(cp, mask, valid)
    result_np = cp.asnumpy(result.astype(cp.float32))
    if return_weights:
        return result_np, cp.asnumpy(sum_w.astype(cp.float32)), rejected_pct
    return result_np, rejected_pct


def stack_linear_fit_clip_gpu(images, weights=None, sigma=3.0, return_weights=False):
    """CuPy twin of ``stack_methods._stack_linear_fit_clip``.

    Return contract identical to the CPU kernel (see
    :func:`stack_kappa_sigma_gpu`).
    """
    cp = _get_cupy()
    arr = _stacked(images)
    valid = ~cp.isnan(arr)
    median = cp.nanmedian(arr, axis=0)
    residuals = arr - median
    med_res = cp.nanmedian(residuals, axis=0)
    std_res = cp.nanstd(residuals, axis=0)
    mask = valid & (cp.abs(residuals - med_res) <= sigma * std_res)
    if weights is not None:
        result, sum_w = _weighted_result(
            cp, mask, arr, _broadcast_weights_cp(cp, arr, weights)
        )
    else:
        result, sum_w = _unweighted_result(cp, mask, arr)
    rejected_pct = _rejected_pct_cp(cp, mask, valid)
    result_np = cp.asnumpy(result.astype(cp.float32))
    if return_weights:
        return result_np, cp.asnumpy(sum_w.astype(cp.float32)), rejected_pct
    return result_np, rejected_pct


def stack_median_gpu(images, weights=None, return_weights=False):
    """CuPy twin of ``stack_methods._stack_median``.

    ``weights`` is accepted and IGNORED, exactly like the CPU kernel (median
    has no weighted form); the weight map is the per-pixel valid-sample count.
    """
    cp = _get_cupy()
    arr = _stacked(images)
    valid = ~cp.isnan(arr)
    result = cp.nanmedian(arr, axis=0)
    result = cp.where(cp.any(valid, axis=0), result, cp.float32(0.0))
    result_np = cp.asnumpy(result.astype(cp.float32))
    sum_w = cp.count_nonzero(valid, axis=0).astype(cp.float32)
    if return_weights:
        return result_np, cp.asnumpy(sum_w.astype(cp.float32)), 0.0
    return result_np, 0.0


# ---------------------------------------------------------------------------
# Winsorized sigma clip (M3, Track B) — exact CuPy twin of the CPU
# ``_stack_winsorized_sigma_iter`` NumPy/bottleneck path.
#
# CPU reference (scientific authority, NEVER modified):
#   seestar/core/stack_methods.py::_stack_winsorized_sigma_iter
#       -> _winsorize_axis0_numpy / _winsorize_bounds
# This file reproduces that algorithm step by step; every deviation below
# would be a scientific bug, not an "optimization".
# ---------------------------------------------------------------------------


def _winsorize_bounds_cp(cp, arr, limits):
    """CuPy twin of ``stack_methods._winsorize_bounds``.

    Returns ``(low_bound, high_bound)`` per column over non-NaN samples only:
    ``low_bound`` = order statistic ``floor(low * n_valid)``, ``high_bound`` =
    order statistic ``n_valid - 1 - floor(high * n_valid)`` (scipy
    ``inclusive=(True, True)`` index convention), with the CPU's own
    ``max_idx = max(n_valid - 1, 0)`` clamp.

    Track P2 (phase D): the sorted values come from a direct ``cp.sort`` of
    the NaN-sentinel ``sort_key`` instead of ``argsort`` + ``take_along_axis``
    (the int64 ``order`` temporary is gone).  ``sorted_vals`` is the same
    sorted multiset, bit-for-bit, so ``lowidx``/``highidx`` (small HxW int64,
    clamped to ``max_idx`` exactly like the CPU) read the same order
    statistics as before.
    """
    low, high = limits
    valid = ~cp.isnan(arr)
    n_valid = cp.count_nonzero(valid, axis=0)
    _p_event("bounds_valid_count")
    sort_key = cp.where(valid, arr, cp.float32(cp.inf))
    _p_event("bounds_sort_key")
    sorted_vals = cp.sort(sort_key, axis=0)
    _p_event("bounds_direct_sort")

    max_idx = cp.maximum(n_valid - 1, 0)
    lowidx = cp.clip(cp.floor(low * n_valid).astype(cp.int64), 0, max_idx)
    highidx = cp.clip(
        n_valid - 1 - cp.floor(high * n_valid).astype(cp.int64), 0, max_idx
    )
    _p_event("bounds_idx")

    low_b = cp.take_along_axis(sorted_vals, lowidx[cp.newaxis], axis=0)
    high_b = cp.take_along_axis(sorted_vals, highidx[cp.newaxis], axis=0)
    _p_event("bounds_lo_hi")
    return low_b, high_b


def _winsor_zero_rank_regime(limits, n_batch):
    """True iff the winsor rank is provably zero on every pixel/iteration.

    Pure host-side predicate (no device work).  The per-pixel winsor rank on
    one side is ``floor(limit * n_valid)`` with ``n_valid <= n_batch`` (the
    ACTUAL current reduction population: every per-pixel valid count is at
    most the frame count of this reduction, in every iteration and in the
    survivor rewinsor pass).  Since ``floor(limit * n)`` is non-decreasing
    in ``n``, checking the population bound ``n_batch`` proves the zero-rank
    regime globally: winsorization is the identity and the survivor rewinsor
    bounds degenerate to the survivor min/max.

    The arithmetic mirrors the CPU exactly (``floor(limit * n_valid)`` in
    float64), so the boundary agrees with the reference at every
    ``n_valid <= n_batch``.  Negative limits never qualify: the CPU treats a
    negative limit as *no winsorization on that side* (``if low > 0``) and
    the slow path reproduces that branch for bit-level symmetry.  Non-finite
    limits never qualify either (``math.floor(inf)`` would raise; the CPU
    hits its own degenerate branches on those inputs, mirrored by the slow
    path).
    """
    low, high = limits
    # NaN-safe: ``NaN >= 0`` is False, so NaN limits stay on the slow path.
    if not (low >= 0.0 and high >= 0.0):
        return False
    if not (np.isfinite(low) and np.isfinite(high)):
        return False
    n = int(n_batch)
    return _math.floor(float(low) * n) == 0 and _math.floor(float(high) * n) == 0


def _winsor_zero_rank_cols_cp(cp, limits, n_valid_col):
    """CuPy twin of ``stack_methods._winsor_zero_rank_cols`` (per column).

    ``floor(low * n) == 0`` and ``floor(high * n) == 0`` for ``n =
    n_valid_col`` (the ACTUAL per-column valid population).  Negative /
    non-finite limits never qualify (no column zero-rank).
    """
    low, high = float(limits[0]), float(limits[1])
    if not (low >= 0.0 and high >= 0.0):
        return cp.zeros_like(n_valid_col, dtype=bool)
    if not (np.isfinite(low) and np.isfinite(high)):
        return cp.zeros_like(n_valid_col, dtype=bool)
    return (cp.floor(low * n_valid_col) == 0) & (
        cp.floor(high * n_valid_col) == 0
    )


def _gross_outlier_keep_cp(cp, arr, valid):
    """CuPy twin of ``stack_methods._gross_outlier_keep`` (one-pass guard).

    Same per-column extreme-gap-vs-survivor-consensus rule (gap factor 100) AND
    local relative deviation rule (``|extreme - median| > 3*|median|``, median
    != 0), computed on the device (never a CPU reduction / ``cp.asnumpy``).
    Both criteria are invariant under positive multiplicative scaling (no
    absolute unit floor, no unit-dependent epsilon).  Returns a boolean
    keep-mask shaped like ``arr``; missing (NaN) samples are excluded.
    """
    n_valid = cp.count_nonzero(valid, axis=0)
    if not bool(cp.any(n_valid >= 3)):
        # No column has enough valid samples to guard (N < 3 or all columns
        # n_valid <= 2): no safe rejection, keep everything.
        return valid
    # Sort valid samples ascending; NaN -> +inf (kept out of the extremes).
    sort_key = cp.where(valid, arr, cp.float32(cp.inf))
    sorted_vals = cp.sort(sort_key, axis=0)  # (N, ...)
    # Free the sort key early: only the sorted values are needed downstream, and
    # releasing the (N, H, W) key lets the CuPy pool reuse that block (avoids the
    # full-frame OOM on 2 GB cards).
    del sort_key

    # Extract every value needed downstream as independent (H, W) arrays (min1
    # is copied to break the slice view) so the (N, H, W) sort output can be
    # released immediately, keeping the peak footprint low on 2 GB cards.
    min1 = sorted_vals[0].copy()
    min2 = sorted_vals[1]
    max1_idx = cp.clip(n_valid - 1, 0, None)
    max2_idx = cp.clip(n_valid - 2, 0, None)
    max1 = cp.take_along_axis(sorted_vals, max1_idx[cp.newaxis], axis=0)[0]
    max2 = cp.take_along_axis(sorted_vals, max2_idx[cp.newaxis], axis=0)[0]
    lo = cp.clip((n_valid - 1) // 2, 0, None)
    hi = cp.clip(n_valid // 2, 0, None)
    lo_val = cp.take_along_axis(sorted_vals, lo[cp.newaxis, ...], axis=0)[0]
    hi_val = cp.take_along_axis(sorted_vals, hi[cp.newaxis, ...], axis=0)[0]

    guardable = n_valid >= 3

    # Mixed tiles can contain guardable and all-invalid columns.  Compute the
    # gaps/spans on SAFE finite operands: ``cp.where`` zeros the +inf sentinels
    # on the non-guardable columns so no inf-inf NaN is ever produced (no
    # ``cp.errstate``, which CuPy 14.2 does not support), then mask the
    # rejection with ``guardable``.  Mirrors the CPU exactly.
    max1_s = cp.where(guardable, max1, cp.float32(0.0))
    max2_s = cp.where(guardable, max2, cp.float32(0.0))
    min1_s = cp.where(guardable, min1, cp.float32(0.0))
    min2_s = cp.where(guardable, min2, cp.float32(0.0))
    del sorted_vals, min2  # release the (N, H, W) sort output (min1 is a copy)
    high_gap = max1_s - max2_s
    high_span = max2_s - min1_s
    low_gap = min2_s - min1_s
    low_span = max1_s - min2_s

    high_a = high_gap > cp.float32(100.0) * high_span
    low_a = low_gap > cp.float32(100.0) * low_span

    # Local relative deviation criterion (scale-invariant).  ``median == 0``
    # has no finite relative scale -> conservative (no deviation rejection).
    # Median derived DIRECTLY from sorted_vals (no second nanmedian
    # masked-array/sort allocation): sorted_vals[0:n_valid] hold the valid
    # samples ascending, so the median is the middle value (odd n_valid) or the
    # mean of the two middle values (even n_valid) — exact cp.nanmedian
    # semantics for every valid-count case.
    median = cp.where(
        (n_valid % 2) == 1, lo_val, (lo_val + hi_val) * cp.float32(0.5)
    )
    median = cp.where(n_valid == 0, cp.float32(0.0), median)
    median_abs = cp.abs(median)
    has_scale = median_abs > 0
    dev_threshold = cp.float32(3.0) * median_abs
    high_b = has_scale & (cp.abs(max1 - median) > dev_threshold)
    low_b = has_scale & (cp.abs(min1 - median) > dev_threshold)

    reject_high = high_a & high_b & guardable
    reject_low = low_a & low_b & guardable

    is_max1 = valid & (arr == max1[cp.newaxis, ...])
    is_min1 = valid & (arr == min1[cp.newaxis, ...])
    reject = (is_max1 & reject_high[cp.newaxis, ...]) | (
        is_min1 & reject_low[cp.newaxis, ...]
    )
    keep = valid & ~reject

    n_rej = int(cp.count_nonzero(reject))
    if n_rej:
        logger.debug(
            "ROBUST_SMALL_N mode=gross_outlier_guard (gpu) rejected=%d", n_rej
        )
    return keep


def _winsorize_bounds_minmax_cp(cp, arr, mask):
    """Zero-rank twin of ``_winsorize_bounds_cp``: survivor min/max, no sort.

    Only valid inside the zero-rank regime (survivor counts are ``<= N_batch``
    so their rank floors are zero too): the CPU bound order statistics
    degenerate to the survivor minimum (low) and maximum (high).  The CPU's
    pathological all-invalid-column behavior is preserved EXACTLY: an empty
    survivor column yields ``+inf`` on BOTH sides (the reference sorts NaN to
    ``+inf`` and reads the sentinel), NOT NaN, so ``clip(arr, +inf, +inf)``
    on the rejected valid samples of that column reproduces the reference
    bit-for-bit.  Ties, single-valid columns, ``+/-inf`` samples and NaN
    handling are all preserved (values are compared, never arithmetically
    combined; NaN samples are excluded through ``mask``).
    """
    # low: smallest survivor, or +inf when the column has no survivor.
    low_b = cp.min(cp.where(mask, arr, cp.float32(cp.inf)), axis=0)
    # high: largest survivor; an empty column reads the CPU +inf sentinel.
    high_b = cp.max(cp.where(mask, arr, cp.float32(-cp.inf)), axis=0)
    high_b = cp.where(cp.any(mask, axis=0), high_b, cp.float32(cp.inf))
    return low_b, high_b


def _winsorize_axis0_rank_path_cp(cp, arr, limits):
    """Exact pre-Phase-D rank-based winsorization (retained verbatim).

    Used ONLY by ``_winsorize_axis0_cp`` for the degenerate OVERLAP corner
    (both winsor sides active with ``floor(low * n) + floor(high * n) > n``
    on some column, i.e. the replaced rank ranges intersect): there the
    sequential low-then-high rank replacement can split a boundary tie group
    between the two bounds, which no value comparison can express, so the
    original ``argsort``/inverse-rank machinery is kept for those inputs to
    preserve the CPU-defined behavior exactly.  This is the verbatim
    pre-Phase-D implementation (``order`` and ``rank`` full-size int64
    temporaries included) and stays bit-identical to it.
    """
    low, high = limits
    arr = arr.astype(cp.float32, copy=False)
    result = arr.copy()
    _p_event("winsor_copy_result")

    valid = ~cp.isnan(arr)
    n_valid = cp.count_nonzero(valid, axis=0)
    _p_event("winsor_valid_count")

    _p_wall_start("sync_any_nvalid")
    _has_valid = bool(cp.any(n_valid > 0))
    _p_wall_end()
    if not _has_valid:
        _p_note("winsorize_all_invalid_column_set")
        return result

    # Sort ascending with NaN pushed to the end (NaN -> +inf).
    sort_key = cp.where(valid, arr, cp.float32(cp.inf))
    _p_event("winsor_sort_key")
    order = cp.argsort(sort_key, axis=0)
    _p_event("winsor_argsort")
    sorted_vals = cp.take_along_axis(sort_key, order, axis=0)
    _p_event("winsor_take_along")

    # ``rank`` is the inverse permutation of ``order``: ``rank[i, ...]`` is the
    # sorted position (0 = smallest) of original sample ``i`` along axis 0.
    rank = cp.argsort(order, axis=0)
    _p_event("winsor_rank_argsort")

    if low > 0:
        lowidx = cp.clip(cp.floor(low * n_valid).astype(cp.int64), 0, None)
        # Mirror NumPy exactly: an index >= the axis length makes
        # ``np.take_along_axis`` raise IndexError (e.g. ``low=1.0`` on an
        # all-valid column -> floor(1.0 * n_valid) == n_valid == N).  CuPy
        # silently wraps out-of-range indices, so the twin must raise
        # explicitly to reproduce the CPU reference failure instead of
        # diverging into wrapped-index garbage.
        _p_wall_start("sync_lowidx_oob_check")
        _lowidx_oob = bool(cp.any(lowidx >= arr.shape[0]))
        _p_wall_end()
        if _lowidx_oob:
            raise IndexError(
                f"index {int(cp.max(lowidx))} is out of bounds for axis 0 "
                f"with size {arr.shape[0]}"
            )
        low_bound = cp.take_along_axis(
            sorted_vals, lowidx[cp.newaxis], axis=0
        )
        _p_event("winsor_low_bound")
        low_sel = valid & (rank < lowidx[cp.newaxis])
        _p_wall_start("sync_low_sel")
        _has_low_sel = bool(cp.any(low_sel))
        _p_wall_end()
        if _has_low_sel:
            result[low_sel] = cp.broadcast_to(
                low_bound, result.shape
            )[low_sel]
        _p_event("winsor_low_replace")

    if high > 0:
        highidx = cp.clip(cp.floor(high * n_valid).astype(cp.int64), 0, None)
        upidx = cp.clip(n_valid - highidx, 0, None)
        keep_idx = cp.clip(upidx - 1, 0, None)
        high_bound = cp.take_along_axis(
            sorted_vals, keep_idx[cp.newaxis], axis=0
        )
        _p_event("winsor_high_bound")
        high_sel = valid & (rank >= upidx[cp.newaxis])
        _p_wall_start("sync_high_sel")
        _has_high_sel = bool(cp.any(high_sel))
        _p_wall_end()
        if _has_high_sel:
            result[high_sel] = cp.broadcast_to(
                high_bound, result.shape
            )[high_sel]
        _p_event("winsor_high_replace")

    _p_event("winsor_return")
    return result


def _winsorize_axis0_cp(cp, arr, limits):
    """CuPy twin of ``stack_methods._winsorize_axis0_numpy``.

    Vectorized winsorization along the first axis with the exact CPU index
    semantics (``floor(limit * n_valid)`` truncation, NaN samples never
    touched / preserved as NaN).  Track P2 (phase D): replacement is
    expressed as VALUE clipping against the order-statistic bounds instead of
    rank-based scatter, and the sorted values come from a direct ``cp.sort``
    of the NaN-sentinel key -- the two full-size int64 temporaries
    ``order = argsort(sort_key)`` and ``rank = argsort(order)`` (the largest
    VRAM consumers of the slow path) are gone.

    Equivalence proof (rank replacement == value clipping): the CPU replaces
    the samples with ``rank < lowidx`` by ``low_bound = sorted[lowidx]`` and
    the samples with ``rank >= upidx`` by ``high_bound = sorted[keep_idx]``
    (``keep_idx = upidx - 1``).  A sample with ``x < low_bound`` always has
    ``rank < lowidx`` and vice versa up to the tie group AT the boundary,
    whose members the CPU would "replace" with their own bit pattern (a
    no-op); symmetrically ``x > high_bound`` iff ``rank >= upidx`` up to
    same-bit boundary ties.  NaN samples never compare True, so missing
    samples are untouched either way.  Hence the value tests ``valid &
    (arr < low_bound)`` / ``valid & (arr > high_bound)`` select exactly the
    samples whose replacement changes the value, and the writes are
    bit-identical to the rank-based scatter (ties at the boundary included;
    the only residual difference class is a mixed ``-0.0``/``+0.0`` boundary
    tie, which flips a zero's sign bit only -- numerically zero, and the CPU
    tie order there is itself backend-arbitrary).

    The CPU's extreme-limit ``IndexError`` (``lowidx >= N``, e.g. ``low =
    1.0`` on an all-valid column, where NumPy's ``take_along_axis`` raises)
    is preserved verbatim via the explicit out-of-bounds check below -- the
    twin never silently clips ``lowidx``.  The ONLY case value clipping
    cannot reproduce is the degenerate OVERLAP corner (both sides active
    with ``floor(low * n) + floor(high * n) > n`` on some column -> the
    replaced rank ranges intersect): those inputs fall back to
    ``_winsorize_axis0_rank_path_cp`` (the exact pre-Phase-D implementation)
    so the CPU-defined behavior is preserved for every input.
    """
    low, high = limits
    arr = arr.astype(cp.float32, copy=False)
    result = arr.copy()
    _p_event("winsor_copy_result")

    valid = ~cp.isnan(arr)
    n_valid = cp.count_nonzero(valid, axis=0)
    _p_event("winsor_valid_count")

    _p_wall_start("sync_any_nvalid")
    _has_valid = bool(cp.any(n_valid > 0))
    _p_wall_end()
    if not _has_valid:
        _p_note("winsorize_all_invalid_column_set")
        return result

    # Sort ascending with NaN pushed to the end (NaN -> +inf).  Direct value
    # sort: no int64 ``order`` permutation is materialized; ``sorted_vals``
    # is bitwise the multiset ``argsort`` + ``take_along_axis`` would gather.
    sort_key = cp.where(valid, arr, cp.float32(cp.inf))
    _p_event("winsor_sort_key")
    sorted_vals = cp.sort(sort_key, axis=0)
    _p_event("winsor_direct_sort")

    if low > 0:
        lowidx = cp.clip(cp.floor(low * n_valid).astype(cp.int64), 0, None)
        # Mirror NumPy exactly: an index >= the axis length makes
        # ``np.take_along_axis`` raise IndexError (e.g. ``low=1.0`` on an
        # all-valid column -> floor(1.0 * n_valid) == n_valid == N).  CuPy
        # silently wraps out-of-range indices, so the twin must raise
        # explicitly to reproduce the CPU reference failure instead of
        # diverging into wrapped-index garbage.  This check runs BEFORE the
        # overlap fallback, exactly like the CPU's low branch precedes the
        # high branch.
        _p_wall_start("sync_lowidx_oob_check")
        _lowidx_oob = bool(cp.any(lowidx >= arr.shape[0]))
        _p_wall_end()
        if _lowidx_oob:
            raise IndexError(
                f"index {int(cp.max(lowidx))} is out of bounds for axis 0 "
                f"with size {arr.shape[0]}"
            )
        low_bound = cp.take_along_axis(
            sorted_vals, lowidx[cp.newaxis], axis=0
        )
        _p_event("winsor_low_bound")
    else:
        lowidx = None
        low_bound = None

    if high > 0:
        highidx = cp.clip(cp.floor(high * n_valid).astype(cp.int64), 0, None)
        upidx = cp.clip(n_valid - highidx, 0, None)
        keep_idx = cp.clip(upidx - 1, 0, None)
        high_bound = cp.take_along_axis(
            sorted_vals, keep_idx[cp.newaxis], axis=0
        )
        _p_event("winsor_high_bound")
    else:
        upidx = None
        high_bound = None

    # Degenerate overlap corner: with both sides active, the replaced rank
    # ranges ``[0, lowidx)`` and ``[upidx, n_valid)`` intersect on a column
    # exactly when ``lowidx > upidx`` (``floor(low * n) + floor(high * n) >
    # n``).  There the sequential low-then-high replacement can give two
    # members of one boundary tie group DIFFERENT bounds, which no value
    # comparison can express -- fall back to the exact pre-Phase-D
    # rank-based implementation for the whole call.
    if low > 0 and high > 0:
        _p_wall_start("sync_overlap_check")
        _overlap = bool(cp.any(lowidx > upidx))
        _p_wall_end()
        if _overlap:
            _p_note("winsorize_overlap_rank_fallback")
            return _winsorize_axis0_rank_path_cp(cp, arr, limits)

    if low_bound is not None:
        # Value-clip low side: replace exactly the samples strictly below the
        # ``lowidx`` order statistic with it.  Boundary-tie members are not
        # written (their replacement would be a same-bit no-op).  NaN
        # samples never compare True, so missing samples stay NaN.
        low_sel = valid & (arr < low_bound)
        _p_wall_start("sync_low_sel")
        _has_low_sel = bool(cp.any(low_sel))
        _p_wall_end()
        if _has_low_sel:
            result[low_sel] = cp.broadcast_to(
                low_bound, result.shape
            )[low_sel]
        _p_event("winsor_low_replace")

    if high_bound is not None:
        # Value-clip high side: replace exactly the samples strictly above
        # the ``keep_idx`` order statistic (``rank >= upidx``) with it.
        high_sel = valid & (arr > high_bound)
        _p_wall_start("sync_high_sel")
        _has_high_sel = bool(cp.any(high_sel))
        _p_wall_end()
        if _has_high_sel:
            result[high_sel] = cp.broadcast_to(
                high_bound, result.shape
            )[high_sel]
        _p_event("winsor_high_replace")

    _p_event("winsor_return")
    return result


def _winsorized_rejected_pct_cp(cp, mask, valid) -> float:
    """CuPy twin of ``stack_methods._rejected_pct`` (global float)."""
    n_valid = int(cp.count_nonzero(valid))
    if n_valid == 0:
        return 0.0
    n_surv = int(cp.count_nonzero(mask))
    return 100.0 * (n_valid - n_surv) / float(n_valid)


def stack_winsorized_sigma_gpu(
    images,
    weights=None,
    kappa=3.0,
    winsor_limits=(0.05, 0.05),
    apply_rewinsor=True,
    max_iters=5,
    kappa_decay=0.9,
    return_weights=False,
):
    """CuPy scientific twin of ``_stack_winsorized_sigma_iter`` (NumPy path).

    Reproduces the CPU iterative winsorized sigma clipping EXACTLY (same
    winsorization index semantics, same sigma loop, same early exit, same
    ``apply_rewinsor`` survivor-bound substitution, same final weighted /
    unweighted reduction with the ``1e-6`` division floor).  NaN samples are
    missing samples: excluded from order statistics and preserved as NaN in
    the winsorized result; ``rejected_pct`` is computed over the ORIGINAL
    valid samples.

    Exact small-N fast path (Track P1): when the winsor ranks are provably
    zero for the ACTUAL current reduction population (``floor(low * N_batch)
    == floor(high * N_batch) == 0`` with ``N_batch = len(images)`` — e.g. any
    limits with both floors zero at N_batch = 19 for the ``(0.05, 0.05)``
    default) winsorization is the identity on every pixel of every iteration
    and the per-iteration order-statistics work (``argsort`` /
    ``take_along_axis`` / inverse-rank ``argsort`` / replacement) plus the
    survivor-bound sort are skipped: mean/std are computed directly on the
    masked stack and the ``apply_rewinsor`` bounds are the survivor
    min/max.  The output is bit-identical to the slow path in that regime
    (verified by the phase C tests) and remains CPU-exact within the
    documented parity tolerance.  Every other input (including N_batch = 20
    at the default limits, where the rank may be 1) takes the untouched
    slow path.

    Returns ``(result, rejected_pct)`` or, with ``return_weights=True``,
    ``(result, sum_w, rejected_pct)`` — all arrays NumPy float32
    (``cp.asnumpy`` before return), rejected_pct a Python float.
    """
    cp = _get_cupy()
    _ensure_probe()
    _p_event("gpu_fn_start")
    if _PROBE is None:
        arr = _stacked(images)
    else:
        # Probe path: same two steps as ``_stacked`` split so the host
        # stack-packing wall time and the H2D device copy can be measured
        # separately.  Identical values, identical dtype, identical result.
        _p_wall_start("host_stack_pack")
        _host_stack = np.stack([im for im in images], axis=0).astype(
            np.float32
        )
        _p_wall_end()
        _p_event("h2d_transfer")
        arr = cp.asarray(_host_stack)
    _p_event("h2d_arr_ready")

    # Missing samples are excluded from the very first iteration.
    mask = ~cp.isnan(arr)
    valid = mask
    _p_event("mask_prep")

    # Exact small-N fast-path selection (Track P1).  N_batch is the ACTUAL
    # current reduction population of THIS call (the final partial batch of
    # a frozen B_resolved is passed here as-is), never the frozen B_resolved:
    # when floor(low*N_batch) == floor(high*N_batch) == 0, every per-pixel
    # valid count (<= N_batch, every iteration, every column) floors to zero
    # and the winsor rank is zero everywhere -> winsorization is the
    # identity.  All other inputs keep the exact slow path below.
    n_batch = int(arr.shape[0])

    # Per-column zero-rank classification from the ORIGINAL valid population
    # (registration NaNs / support mask), never the nominal N_batch.
    n_valid_col = cp.count_nonzero(valid, axis=0)
    zero_rank_cols = _winsor_zero_rank_cols_cp(cp, winsor_limits, n_valid_col)
    rank_cols = ~zero_rank_cols
    n_zero_rank = int(cp.count_nonzero(zero_rank_cols))
    _p_note("gross_guard_zero_rank_cols=%d n_batch=%d" % (n_zero_rank, n_batch))
    if n_zero_rank:
        logger.debug(
            "ROBUST_SMALL_N mode=gross_outlier_guard zero_rank_cols=%d "
            "rank_cols=%d n_batch=%d",
            n_zero_rank,
            int(cp.count_nonzero(rank_cols)),
            n_batch,
        )

    # One-pass gross-outlier guard (zero-rank columns only).
    guard_keep = (
        _gross_outlier_keep_cp(cp, arr, valid) if n_zero_rank else valid
    )

    # Historical Winsor on rank-sufficient columns only (zero-rank columns
    # NaN-masked out so they don't feed the schedule / kappa decay).
    # All-zero-rank fast path: no rank columns -> the Winsor loop is skipped
    # entirely (the guard alone decides the zero-rank columns).
    rank_cols3 = rank_cols[cp.newaxis, ...]
    if bool(cp.any(rank_cols)):
        arr_w = cp.where(rank_cols3, arr, cp.float32(cp.nan))
        mask_w = ~cp.isnan(arr_w)

        kappa_iter = float(kappa)
        for itr in range(int(max_iters)):
            _p_event("iter%d_loop_top" % itr)
            arr_masked = cp.where(mask_w, arr_w, cp.float32(cp.nan))
            _p_event("iter%d_masked" % itr)
            arr_w_data = _winsorize_axis0_cp(cp, arr_masked, winsor_limits)
            _p_event("iter%d_winsorized" % itr)

            mu_w = cp.nanmean(arr_w_data, axis=0)
            _p_event("iter%d_nanmean" % itr)
            sigma_w = cp.nanstd(arr_w_data, axis=0, ddof=1)
            _p_event("iter%d_nanstd" % itr)

            # Columns with <= 1 valid sample have undefined ddof=1 std ->
            # treat as no-rejection identity (sigma == 0), identical to the
            # CPU guard.
            n_valid_col = cp.count_nonzero(mask_w, axis=0)
            sigma_w = cp.where(
                n_valid_col <= 1, cp.float32(0.0), sigma_w
            )
            _p_event("iter%d_sigma_guard" % itr)

            low = mu_w - cp.float32(kappa_iter) * sigma_w
            high = mu_w + cp.float32(kappa_iter) * sigma_w
            new_mask = mask_w & (arr_w >= low) & (arr_w <= high)
            _p_event("iter%d_new_mask" % itr)
            _p_wall_start("iter%d_sync_nrej" % itr)
            n_rej = int(cp.count_nonzero(mask_w)) - int(cp.count_nonzero(new_mask))
            _p_wall_end()
            _p_note("iter=%d n_rej=%d" % (itr, n_rej))
            _p_event("iter%d_nrej_sync" % itr)
            mask_w = new_mask
            if n_rej == 0:
                _p_note("early_exit_iteration=%d" % itr)
                break
            if kappa_decay < 1.0:
                kappa_iter = kappa * (kappa_decay ** (itr + 1))
        else:
            _p_note("max_iters_reached=%d" % int(max_iters))
    else:
        mask_w = valid & rank_cols3  # all-False (no rank columns)

    # Combine: zero-rank columns use the frozen guard mask; rank-sufficient
    # columns use the iterative Winsor mask.
    mask = cp.where(rank_cols3, mask_w, guard_keep)

    if apply_rewinsor:
        # Rejected-but-valid samples are substituted with the winsorized bound
        # of the SURVIVOR distribution; survivors are preserved exactly;
        # missing samples remain NaN.  Same as the CPU branch.  The bounds
        # are per-column (survivor order statistics); zero-rank columns
        # degenerate to survivor min/max exactly like the CPU.
        low_b, high_b = _winsorize_bounds_cp(
            cp, cp.where(mask, arr, cp.float32(cp.nan)), winsor_limits
        )
        _p_event("rewinsor_bounds_done")
        clipped = cp.clip(arr, low_b, high_b)
        _p_event("rewinsor_clip")
        arr_final = cp.where(
            mask, arr, cp.where(valid, clipped, cp.float32(cp.nan))
        )
        _p_event("rewinsor_final")
    else:
        arr_final = cp.where(mask, arr, cp.float32(cp.nan))
        _p_event("no_rewinsor_final")

    # arr_final is NaN exactly where a sample does NOT contribute to the mean.
    contrib = ~cp.isnan(arr_final)
    _p_event("contrib")

    if weights is not None:
        w = _broadcast_weights_cp(cp, arr, weights)
        sum_w = cp.nansum(
            cp.where(contrib, w, cp.float32(0.0)), axis=0, dtype=cp.float32
        )
        sum_d = cp.nansum(arr_final * w, axis=0, dtype=cp.float32)
        result = cp.where(
            sum_w > 1e-6,
            sum_d / cp.maximum(sum_w, 1e-6),
            cp.zeros_like(sum_d),
        )
        _p_event("final_reduce_weighted")
    else:
        result = cp.nanmean(arr_final, axis=0)
        result = cp.where(
            cp.any(contrib, axis=0), result, cp.float32(0.0)
        )
        sum_w = cp.count_nonzero(contrib, axis=0).astype(cp.float32)
        _p_event("final_reduce_unweighted")

    _p_wall_start("sync_rejected_pct")
    rejected_pct = _winsorized_rejected_pct_cp(cp, mask, valid)
    _p_wall_end()
    _p_event("rejected_pct")

    _p_event("result_astype")
    result_np = cp.asnumpy(result.astype(cp.float32))
    _p_event("d2h_done")
    if return_weights:
        return result_np, cp.asnumpy(sum_w.astype(cp.float32)), rejected_pct
    return result_np, rejected_pct


# ---------------------------------------------------------------------------
# Exact-N_batch SPATIAL GPU tiling (Track P3, phase E).
#
# DEDICATED tiling seam for the Winsorized reduction.  This is NOT the CPU
# ``_combine_hq_by_tiles`` subgroup loop (queue_manager, §19): that path
# derives ``group_size < N_batch`` and reduces NONLINEARLY over stack-axis
# subgroups, which is a bounded-memory *approximation*.  The seam below
# NEVER splits the scientific stack axis: every tile retains the FULL
# ``N_batch`` population (all frames, all per-image scalar weights, all
# NaN/mask patterns) and only the SPATIAL dimensions are tiled, so every
# output pixel is reduced over exactly the same N_batch samples as the
# untiled GPU twin (and the CPU authority).
#
# Global-iteration coordination (why a plain per-tile call is NOT exact)
# ---------------------------------------------------------------------
# The reference loop early-exits on the FIRST iteration whose GLOBAL
# rejection count is zero, and the sigma band narrows (``kappa_decay``)
# after every rejecting iteration.  A per-pixel rejection decision is
# purely column-local, but the NUMBER of executed iterations and the kappa
# sequence are GLOBAL properties: a tile may stop rejecting locally at
# iteration ``i`` and still lose pixels at iteration ``i+1`` under the
# narrower global band (measured divergence: up to 215 ADU in a synthetic
# two-region stack).  Per-tile independent early exit is therefore NOT
# exact.  The tiled driver reproduces the global schedule exactly with two
# coordinated tile passes:
#
#   pass 1 (schedule discovery): every tile runs the DETERMINISTIC kappa
#     schedule ``kappa * decay**i`` (i = 0 .. max_iters-1) with no early
#     exit, recording its local rejection count per iteration; the counts
#     are summed over tiles.  Up to the global stop iteration every
#     iteration rejects somewhere, so the reference's kappa sequence is
#     exactly the deterministic schedule; pass 1 therefore yields the
#     reference's per-iteration GLOBAL counts.  ``z`` = first iteration
#     with global count 0 (``z_eff = max_iters`` when the run exhausts the
#     schedule without a zero-rejection iteration; note ``max_iters == 1``
#     always has ``z_eff == max_iters``).
#   pass 2 (exact replay): every tile re-runs exactly ``z_eff`` schedule
#     iterations (no counts, no early exit) and the per-tile survivor mask
#     is finalised exactly like the untiled twin (``apply_rewinsor`` bounds
#     over the tile survivors, weighted / unweighted final reduction with
#     the ``1e-6`` division floor).
#
# The reference's own mask after the run is ``M_z``: iterations 0..z-1 all
# rejected somewhere (they decay the kappa, which the deterministic
# schedule reproduces bit-for-bit) and iteration z changed nothing (a
# no-op replay of the mask after z iterations is the same mask), so pass 2
# with ``z_eff`` iterations is bit-identical to the untiled run per column.
#
# No-halo proof
# -------------
# Every operation of the reduction is independent per aligned
# (spatial, channel) position along axis 0: winsorization (per-column
# order statistics of the tile's OWN axis-0 samples), location/scale
# (``nanmean`` / ``nanstd`` along axis 0), the sigma band test and mask
# update (elementwise per column), the survivor ``apply_rewinsor`` bounds
# (per-column order statistics / min-max), the weighted/unweighted final
# sums (axis-0 reductions with the scalar-per-image weights) and the
# per-pixel rejection counting.  No operation reads a neighbouring
# spatial/channel position, so no halo, no overlap blend, no feather and
# no tile-order dependence can exist; reconstruction is EXACT PLACEMENT
# only.  Verified on the device: every axis-0 primitive used here
# (``cp.sort``, ``cp.nanmean``, ``cp.nanstd``, ``cp.nansum`` with the
# ``(N,)`` weights, count reductions) is bitwise identical when applied to
# a spatial slice of the frame or to the same columns inside the full
# frame for every tile geometry whose per-tile spatial output count stays
# out of the CuPy micro-reduction band (measured on this stack: mono
# band S <= 26, RGB band S <= 30; row bands of any height on realistic
# widths are always far above it).  For MICRO tiles inside that band CuPy
# selects a different axis-0 reduction kernel whose per-column float32
# accumulation order can differ by ~1 float32 ulp (measured worst
# ~2.4e-7 relative, ~2.4e-4 absolute at 1000 ADU scale) with BITWISE-
# identical survivor masks and EXACT global ``rejected_pct`` — four
# orders of magnitude below the documented CPU-parity tolerance; the
# suite pins this residual class explicitly (see
# tests/test_stack_gpu_winsorized_tiled.py) so no divergence beyond it is
# ever accepted.
#
# Every tile keeps the full ``N_batch`` population, so the Phase C
# zero-rank fast path (``_winsor_zero_rank_regime`` on ``N_batch``) and
# the Phase D clip/sort slow path (``_winsorize_axis0_cp`` /
# ``_winsorize_bounds_cp``, including the per-call overlap rank fallback)
# apply unchanged per tile.
#
# ``rejected_pct`` is GLOBAL: pass 2 accumulates ``n_valid`` and
# ``n_survivor`` per tile and returns
# ``100 * (sum n_valid - sum n_survivor) / sum n_valid`` — the sum/sum
# formula, never an average of per-tile percentages.
# ---------------------------------------------------------------------------


def _winsor_tile_slices(frame_shape, tile_shape):
    """Spatial slices of one frame for a tiling geometry (exact placement).

    ``frame_shape``: ``(H, W)``.  ``tile_shape``:

    * ``None`` -> the whole frame ``(0, H, 0, W)``;
    * ``int`` or ``(tile_h,)`` -> full-width row bands of ``tile_h`` rows
      (last band partial);
    * ``(tile_h, tile_w)`` -> rectangular ``tile_h x tile_w`` tiles in
      row-major order (last row band and last column of every band
      partial).

    Never splits the stack axis: every slice is purely spatial.  Returns a
    list of ``(y0, y1, x0, x1)`` tuples.
    """
    H, W = int(frame_shape[0]), int(frame_shape[1])
    if tile_shape is None:
        return [(0, H, 0, W)]
    if isinstance(tile_shape, (tuple, list)):
        if len(tile_shape) == 1:
            tile_h, tile_w = int(tile_shape[0]), W
        else:
            tile_h, tile_w = int(tile_shape[0]), int(tile_shape[1])
    else:
        tile_h, tile_w = int(tile_shape), W
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError(
            "tile_shape dimensions must be positive, got %r" % (tile_shape,)
        )
    tile_h = min(tile_h, H)
    tile_w = max(1, min(tile_w, W))
    slices = []
    for y0 in range(0, H, tile_h):
        y1 = min(y0 + tile_h, H)
        for x0 in range(0, W, tile_w):
            x1 = min(x0 + tile_w, W)
            slices.append((y0, y1, x0, x1))
    return slices


def _winsor_schedule_kappas(kappa, kappa_decay, n_iters):
    """Kappa of every scheduled iteration, bitwise as in the reference.

    The reference narrows kappa ONLY when ``kappa_decay < 1.0`` and only
    after a rejecting iteration, so along a run that never stopped early
    iteration ``i`` uses ``kappa * kappa_decay**i`` (``i = 0`` uses the
    bare ``float(kappa)`` initialisation, bitwise identical to the
    ``* 1.0`` product).  With ``kappa_decay >= 1.0`` the band never
    narrows and every iteration uses ``float(kappa)``.
    """
    k = float(kappa)
    if not (float(kappa_decay) < 1.0):
        return [k] * int(n_iters)
    return [k * (float(kappa_decay) ** i) for i in range(int(n_iters))]


def _winsorized_tile_iterations_cp(
    cp,
    arr,
    kappa,
    winsor_limits,
    n_iters,
    kappa_decay,
    collect_counts,
    tile_tag,
):
    """Run ``n_iters`` schedule iterations on one device tile (full stack
    axis preserved) with NO early exit; mirror of the reference loop body.

    Per-column zero-rank classification from the ORIGINAL valid population:
    zero-rank columns get the one-pass gross-outlier guard (frozen, never
    consuming the global schedule); rank-sufficient columns run the historical
    Winsor iterations (their per-iteration rejection counts feed the
    schedule).  Returns ``(final_mask, counts)``.
    """
    valid = ~cp.isnan(arr)
    n_valid_col = cp.count_nonzero(valid, axis=0)
    zero_rank_cols = _winsor_zero_rank_cols_cp(cp, winsor_limits, n_valid_col)
    rank_cols = ~zero_rank_cols
    rank_cols3 = rank_cols[cp.newaxis, ...]

    # One-pass gross-outlier guard (zero-rank columns only).
    guard_keep = (
        _gross_outlier_keep_cp(cp, arr, valid)
        if bool(cp.any(zero_rank_cols))
        else valid
    )

    # Historical Winsor iterations on rank-sufficient columns only (the
    # all-zero-rank fast path skips the loop entirely).
    if bool(cp.any(rank_cols)):
        arr_w = cp.where(rank_cols3, arr, cp.float32(cp.nan))
        mask_w = ~cp.isnan(arr_w)
        kappas = _winsor_schedule_kappas(kappa, kappa_decay, n_iters)
        counts = [] if collect_counts else None
        for itr in range(int(n_iters)):
            kappa_iter = kappas[itr]
            _p_event("tiled_%s_iter%d" % (tile_tag, itr))
            arr_masked = cp.where(mask_w, arr_w, cp.float32(cp.nan))
            arr_w_data = _winsorize_axis0_cp(cp, arr_masked, winsor_limits)
            mu_w = cp.nanmean(arr_w_data, axis=0)
            sigma_w = cp.nanstd(arr_w_data, axis=0, ddof=1)
            n_valid_col = cp.count_nonzero(mask_w, axis=0)
            sigma_w = cp.where(n_valid_col <= 1, cp.float32(0.0), sigma_w)
            low = mu_w - cp.float32(kappa_iter) * sigma_w
            high = mu_w + cp.float32(kappa_iter) * sigma_w
            new_mask = mask_w & (arr_w >= low) & (arr_w <= high)
            if collect_counts:
                _p_wall_start("tiled_%s_nrej_sync" % tile_tag)
                n_rej = int(cp.count_nonzero(mask_w)) - int(
                    cp.count_nonzero(new_mask)
                )
                _p_wall_end()
                counts.append(n_rej)
            mask_w = new_mask
    else:
        mask_w = valid & rank_cols3  # all-False (no rank columns)
        counts = [0] * int(n_iters) if collect_counts else None

    final_mask = cp.where(rank_cols3, mask_w, guard_keep)
    return final_mask, counts


def _winsorized_tile_finalize_cp(
    cp, arr, mask, valid, weights, apply_rewinsor, winsor_limits
):
    """Final tail of one tile: survivor ``apply_rewinsor`` substitution,
    contribution mask and the weighted / unweighted reduction — the exact
    mirror of the untiled twin's tail on the tile's columns.  Returns
    ``(result_t, sum_w_t)`` device arrays shaped like the tile's spatial
    extent ``(tile_h, tile_w[, C])``.
    """
    if apply_rewinsor:
        low_b, high_b = _winsorize_bounds_cp(
            cp, cp.where(mask, arr, cp.float32(cp.nan)), winsor_limits
        )
        clipped = cp.clip(arr, low_b, high_b)
        arr_final = cp.where(
            mask, arr, cp.where(valid, clipped, cp.float32(cp.nan))
        )
    else:
        arr_final = cp.where(mask, arr, cp.float32(cp.nan))
    contrib = ~cp.isnan(arr_final)
    if weights is not None:
        w = _broadcast_weights_cp(cp, arr, weights)
        sum_w = cp.nansum(
            cp.where(contrib, w, cp.float32(0.0)), axis=0, dtype=cp.float32
        )
        sum_d = cp.nansum(arr_final * w, axis=0, dtype=cp.float32)
        result = cp.where(
            sum_w > 1e-6,
            sum_d / cp.maximum(sum_w, 1e-6),
            cp.zeros_like(sum_d),
        )
    else:
        result = cp.nanmean(arr_final, axis=0)
        result = cp.where(
            cp.any(contrib, axis=0), result, cp.float32(0.0)
        )
        sum_w = cp.count_nonzero(contrib, axis=0).astype(cp.float32)
    return result, sum_w


def stack_winsorized_sigma_gpu_tiled(
    images,
    weights=None,
    kappa=3.0,
    winsor_limits=(0.05, 0.05),
    apply_rewinsor=True,
    max_iters=5,
    kappa_decay=0.9,
    return_weights=False,
    tile_shape=None,
    _tile_order="rowmajor",
):
    """Exact-N_batch SPATIAL GPU tiling of the Winsorized reduction.

    Same scientific twin, same parameters, same return contract as
    :func:`stack_winsorized_sigma_gpu` — ``(result, rejected_pct)`` or,
    with ``return_weights=True``, ``(result, sum_w, rejected_pct)`` (NumPy
    float32 arrays + Python float) — but the SPATIAL dimensions are
    processed in tiles so the per-tile device working set is
    ``N_batch x tile_h x tile_w x C`` instead of the full frame, while the
    full ``N_batch`` stack population is preserved for EVERY output pixel
    (the stack axis is never split; no hierarchical nonlinear reduction).

    ``tile_shape`` is the Phase F planner seam: ``None`` (or a geometry
    covering the whole frame in one tile) delegates to the untiled twin;
    ``int`` / ``(tile_h,)`` selects full-width row bands; ``(tile_h,
    tile_w)`` selects rectangular tiles.  ``_tile_order`` is a test-only
    hook (``"rowmajor"`` or ``"reversed"``) proving tile-order invariance.

    Global-iteration coordination: see the module section above — pass 1
    discovers the reference's global stop iteration ``z_eff`` from
    per-tile rejection counts summed over tiles; pass 2 replays exactly
    ``z_eff`` schedule iterations per tile and finalises it.  The output
    is bit-identical to the untiled twin (reconstruction is exact
    placement only, no blend/feather/halo) and ``rejected_pct`` is the
    global sum/sum formula.
    """
    cp = _get_cupy()
    _ensure_probe()
    _p_event("tiled_fn_start")
    n_batch = int(len(images))
    if n_batch == 0:  # pragma: no cover - degenerate, mirrors CPU failure
        raise ValueError("tiled winsorized sigma requires at least one image")
    _p_wall_start("tiled_host_stack_pack")
    host = np.stack([im for im in images], axis=0).astype(np.float32)
    _p_wall_end()
    frame = host.shape[1:]
    H, W = int(frame[0]), int(frame[1])
    color = host.ndim == 4
    spatial = _winsor_tile_slices((H, W), tile_shape)
    if len(spatial) == 1:
        # Untiled / full-frame geometry: the untiled twin IS the reference
        # implementation of this reduction; delegate for guaranteed
        # bitwise identity (and its single-pass early-exit loop).
        return stack_winsorized_sigma_gpu(
            images,
            weights,
            kappa=kappa,
            winsor_limits=winsor_limits,
            apply_rewinsor=apply_rewinsor,
            max_iters=max_iters,
            kappa_decay=kappa_decay,
            return_weights=return_weights,
        )
    if _tile_order == "reversed":
        spatial = list(reversed(spatial))
    _p_note(
        "tiled n_batch=%d n_tiles=%d tile_shape=%s"
        % (n_batch, len(spatial), tile_shape)
    )

    # ---- pass 1: schedule discovery (deterministic kappa schedule,
    # no early exit, per-iteration LOCAL rejection counts summed globally)
    # With max_iters == 1 the reference loop always runs its single
    # iteration to the end (a zero-rejection iteration would only shorten
    # it by a mask no-op), so z_eff == max_iters is provable a priori and
    # the discovery pass (and the z search over its counts) is skipped.
    if int(max_iters) == 1:
        z_eff = 1
    else:
        global_counts = [0] * int(max_iters)
        for t, (y0, y1, x0, x1) in enumerate(spatial):
            arr_t = cp.asarray(host[:, y0:y1, x0:x1])
            _p_event("tiled_p1_tile%d" % t)
            _, counts = _winsorized_tile_iterations_cp(
                cp,
                arr_t,
                kappa,
                winsor_limits,
                int(max_iters),
                kappa_decay,
                collect_counts=True,
                tile_tag="p1_t%d" % t,
            )
            for i, c in enumerate(counts):
                global_counts[i] += c
        z_eff = int(max_iters)
        for i, c in enumerate(global_counts):
            if c == 0:
                z_eff = i
                break
    _p_note(
        "tiled z_eff=%d global_counts=%s"
        % (z_eff, global_counts if int(max_iters) > 1 else "skipped")
    )

    # ---- pass 2: exact replay of z_eff schedule iterations per tile and
    # exact-placement reconstruction + GLOBAL rejection accounting
    out_shape = (H, W) + ((int(frame[2]),) if color else ())
    result = np.empty(out_shape, dtype=np.float32)
    sum_w = np.empty(out_shape, dtype=np.float32)
    n_valid_total = 0
    n_surv_total = 0
    for t, (y0, y1, x0, x1) in enumerate(spatial):
        arr_t = cp.asarray(host[:, y0:y1, x0:x1])
        _p_event("tiled_p2_tile%d" % t)
        valid_t = ~cp.isnan(arr_t)
        # Always run the tile helper: it applies the one-pass gross-outlier
        # guard to zero-rank columns AND replays ``z_eff`` Winsor iterations
        # on rank-sufficient columns (``z_eff == 0`` -> guard only, no Winsor).
        mask_t, _ = _winsorized_tile_iterations_cp(
            cp,
            arr_t,
            kappa,
            winsor_limits,
            z_eff,
            kappa_decay,
            collect_counts=False,
            tile_tag="p2_t%d" % t,
        )
        res_t, sumw_t = _winsorized_tile_finalize_cp(
            cp,
            arr_t,
            mask_t,
            valid_t,
            weights,
            apply_rewinsor,
            winsor_limits,
        )
        _p_wall_start("tiled_tile_counts_sync")
        n_valid_total += int(cp.count_nonzero(valid_t))
        n_surv_total += int(cp.count_nonzero(mask_t))
        _p_wall_end()
        _p_event("tiled_p2_tile%d_final" % t)
        result[y0:y1, x0:x1] = cp.asnumpy(res_t.astype(cp.float32))
        sum_w[y0:y1, x0:x1] = cp.asnumpy(sumw_t.astype(cp.float32))
        _p_event("tiled_tile_placed")

    if n_valid_total == 0:
        rejected_pct = 0.0
    else:
        rejected_pct = (
            100.0 * (n_valid_total - n_surv_total) / float(n_valid_total)
        )
    _p_note("tiled rejected_pct=%.12f" % rejected_pct)
    _p_event("tiled_fn_done")
    if return_weights:
        return result, sum_w, rejected_pct
    return result, rejected_pct
