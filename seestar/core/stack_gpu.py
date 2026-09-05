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

Only kappa-sigma, linear-fit-clip and median are accelerated: they are the
sorting-heavy reductions that profit on GPU (measured).  Mean and
winsorized-sigma stay CPU.

CuPy is imported lazily (module import never requires cupy); every public
function returns plain NumPy arrays (``cp.asnumpy`` before return) — a CuPy
array never crosses the module boundary.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "stack_kappa_sigma_gpu",
    "stack_linear_fit_clip_gpu",
    "stack_median_gpu",
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
