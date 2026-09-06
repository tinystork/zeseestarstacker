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
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "stack_kappa_sigma_gpu",
    "stack_linear_fit_clip_gpu",
    "stack_median_gpu",
    "stack_winsorized_sigma_gpu",
]

_cupy_module = None


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
    sort_key = cp.where(valid, arr, cp.float32(cp.inf))
    order = cp.argsort(sort_key, axis=0)
    sorted_vals = cp.take_along_axis(sort_key, order, axis=0)

    max_idx = cp.maximum(n_valid - 1, 0)
    lowidx = cp.clip(cp.floor(low * n_valid).astype(cp.int64), 0, max_idx)
    highidx = cp.clip(
        n_valid - 1 - cp.floor(high * n_valid).astype(cp.int64), 0, max_idx
    )

    low_b = cp.take_along_axis(sorted_vals, lowidx[cp.newaxis], axis=0)
    high_b = cp.take_along_axis(sorted_vals, highidx[cp.newaxis], axis=0)
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

    valid = ~cp.isnan(arr)
    n_valid = cp.count_nonzero(valid, axis=0)

    if not bool(cp.any(n_valid > 0)):
        return result

    # Sort ascending with NaN pushed to the end (NaN -> +inf).
    sort_key = cp.where(valid, arr, cp.float32(cp.inf))
    order = cp.argsort(sort_key, axis=0)
    sorted_vals = cp.take_along_axis(sort_key, order, axis=0)

    # ``rank`` is the inverse permutation of ``order``: ``rank[i, ...]`` is the
    # sorted position (0 = smallest) of original sample ``i`` along axis 0.
    rank = cp.argsort(order, axis=0)

    if low > 0:
        lowidx = cp.clip(cp.floor(low * n_valid).astype(cp.int64), 0, None)
        low_bound = cp.take_along_axis(
            sorted_vals, lowidx[cp.newaxis], axis=0
        )
        low_sel = valid & (rank < lowidx[cp.newaxis])
        if bool(cp.any(low_sel)):
            result[low_sel] = cp.broadcast_to(
                low_bound, result.shape
            )[low_sel]

    if high > 0:
        highidx = cp.clip(cp.floor(high * n_valid).astype(cp.int64), 0, None)
        upidx = cp.clip(n_valid - highidx, 0, None)
        keep_idx = cp.clip(upidx - 1, 0, None)
        high_bound = cp.take_along_axis(
            sorted_vals, keep_idx[cp.newaxis], axis=0
        )
        high_sel = valid & (rank >= upidx[cp.newaxis])
        if bool(cp.any(high_sel)):
            result[high_sel] = cp.broadcast_to(
                high_bound, result.shape
            )[high_sel]

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

    Returns ``(result, rejected_pct)`` or, with ``return_weights=True``,
    ``(result, sum_w, rejected_pct)`` — all arrays NumPy float32
    (``cp.asnumpy`` before return), rejected_pct a Python float.
    """
    cp = _get_cupy()
    arr = _stacked(images)

    # Missing samples are excluded from the very first iteration.
    mask = ~cp.isnan(arr)
    valid = mask
    kappa_iter = float(kappa)

    for itr in range(int(max_iters)):
        arr_masked = cp.where(mask, arr, cp.float32(cp.nan))
        arr_w_data = _winsorize_axis0_cp(cp, arr_masked, winsor_limits)

        mu_w = cp.nanmean(arr_w_data, axis=0)
        sigma_w = cp.nanstd(arr_w_data, axis=0, ddof=1)

        # Columns with <= 1 valid sample have undefined ddof=1 std -> treat as
        # no-rejection identity (sigma == 0), identical to the CPU guard.
        n_valid_col = cp.count_nonzero(mask, axis=0)
        sigma_w = cp.where(
            n_valid_col <= 1, cp.float32(0.0), sigma_w
        )

        low = mu_w - cp.float32(kappa_iter) * sigma_w
        high = mu_w + cp.float32(kappa_iter) * sigma_w
        new_mask = mask & (arr >= low) & (arr <= high)
        n_rej = int(cp.count_nonzero(mask)) - int(cp.count_nonzero(new_mask))
        mask = new_mask
        if n_rej == 0:
            break
        if kappa_decay < 1.0:
            kappa_iter = kappa * (kappa_decay ** (itr + 1))

    if apply_rewinsor:
        # Rejected-but-valid samples are substituted with the winsorized bound
        # of the SURVIVOR distribution; survivors are preserved exactly;
        # missing samples remain NaN.  Same as the CPU branch.
        low_b, high_b = _winsorize_bounds_cp(
            cp, cp.where(mask, arr, cp.float32(cp.nan)), winsor_limits
        )
        clipped = cp.clip(arr, low_b, high_b)
        arr_final = cp.where(
            mask, arr, cp.where(valid, clipped, cp.float32(cp.nan))
        )
    else:
        arr_final = cp.where(mask, arr, cp.float32(cp.nan))

    # arr_final is NaN exactly where a sample does NOT contribute to the mean.
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

    rejected_pct = _winsorized_rejected_pct_cp(cp, mask, valid)

    result_np = cp.asnumpy(result.astype(cp.float32))
    if return_weights:
        return result_np, cp.asnumpy(sum_w.astype(cp.float32)), rejected_pct
    return result_np, rejected_pct
