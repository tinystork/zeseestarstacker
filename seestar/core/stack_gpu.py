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
"""

from __future__ import annotations

import math as _math
import os as _os
import time as _time

import numpy as np

__all__ = [
    "stack_kappa_sigma_gpu",
    "stack_linear_fit_clip_gpu",
    "stack_median_gpu",
    "stack_winsorized_sigma_gpu",
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
    """
    low, high = limits
    valid = ~cp.isnan(arr)
    n_valid = cp.count_nonzero(valid, axis=0)
    _p_event("bounds_valid_count")
    sort_key = cp.where(valid, arr, cp.float32(cp.inf))
    _p_event("bounds_sort_key")
    order = cp.argsort(sort_key, axis=0)
    _p_event("bounds_argsort")
    sorted_vals = cp.take_along_axis(sort_key, order, axis=0)
    _p_event("bounds_take_along")

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


def _winsorize_axis0_cp(cp, arr, limits):
    """CuPy twin of ``stack_methods._winsorize_axis0_numpy``.

    Vectorized winsorization along the first axis with the exact CPU index
    semantics (``floor(limit * n_valid)`` truncation, rank-based replacement,
    NaN samples never touched / preserved as NaN).  The CPU also uses
    ``np.where(valid, arr, inf)`` then sorts: sorted_vals rows at/after
    ``n_valid`` are ``inf`` (missing samples), and the rank-based masks
    exclude invalid samples via ``valid``, so missing values are never
    replaced.  This twin reproduces the same index arithmetic bit-for-bit
    (identical floor/clip/rank ops); only the float reduction order may
    differ by ULPs (covered by the parity tolerances).
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
    zero_rank = _winsor_zero_rank_regime(winsor_limits, n_batch)
    _p_note("zero_rank_fastpath=%s n_batch=%d" % (zero_rank, n_batch))
    if zero_rank:
        _p_event("fastpath_decided")
    kappa_iter = float(kappa)

    for itr in range(int(max_iters)):
        _p_event("iter%d_loop_top" % itr)
        arr_masked = cp.where(mask, arr, cp.float32(cp.nan))
        _p_event("iter%d_masked" % itr)
        if zero_rank:
            # Zero-rank regime: winsorization is the identity, so the
            # per-iteration sort block of _winsorize_axis0_cp (argsort +
            # take_along_axis + inverse-rank argsort + replacement) is
            # skipped and the location/scale statistics are computed
            # directly on the masked stack.  The slow path would return an
            # exact copy of arr_masked here, so mu_w/sigma_w and everything
            # downstream are bit-identical.
            arr_w_data = arr_masked
            _p_event("iter%d_winsor_skipped_zero_rank" % itr)
        else:
            arr_w_data = _winsorize_axis0_cp(cp, arr_masked, winsor_limits)
            _p_event("iter%d_winsorized" % itr)

        mu_w = cp.nanmean(arr_w_data, axis=0)
        _p_event("iter%d_nanmean" % itr)
        sigma_w = cp.nanstd(arr_w_data, axis=0, ddof=1)
        _p_event("iter%d_nanstd" % itr)

        # Columns with <= 1 valid sample have undefined ddof=1 std -> treat as
        # no-rejection identity (sigma == 0), identical to the CPU guard.
        n_valid_col = cp.count_nonzero(mask, axis=0)
        sigma_w = cp.where(
            n_valid_col <= 1, cp.float32(0.0), sigma_w
        )
        _p_event("iter%d_sigma_guard" % itr)

        low = mu_w - cp.float32(kappa_iter) * sigma_w
        high = mu_w + cp.float32(kappa_iter) * sigma_w
        new_mask = mask & (arr >= low) & (arr <= high)
        _p_event("iter%d_new_mask" % itr)
        _p_wall_start("iter%d_sync_nrej" % itr)
        n_rej = int(cp.count_nonzero(mask)) - int(cp.count_nonzero(new_mask))
        _p_wall_end()
        _p_note("iter=%d n_rej=%d" % (itr, n_rej))
        _p_event("iter%d_nrej_sync" % itr)
        mask = new_mask
        if n_rej == 0:
            _p_note("early_exit_iteration=%d" % itr)
            break
        if kappa_decay < 1.0:
            kappa_iter = kappa * (kappa_decay ** (itr + 1))
    else:
        _p_note("max_iters_reached=%d" % int(max_iters))

    if apply_rewinsor:
        # Rejected-but-valid samples are substituted with the winsorized bound
        # of the SURVIVOR distribution; survivors are preserved exactly;
        # missing samples remain NaN.  Same as the CPU branch.
        if zero_rank:
            # Survivor counts are also <= N_batch, so the survivor rank
            # floors are zero too: the CPU _winsorize_bounds order
            # statistics degenerate to the survivor min (low) / max (high).
            # The min/max reduction preserves the reference degenerate
            # behavior exactly (empty survivor column -> +inf sentinel on
            # both sides); no sort is performed.
            low_b, high_b = _winsorize_bounds_minmax_cp(cp, arr, mask)
            _p_event("rewinsor_bounds_minmax")
        else:
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
