# Stacking algorithms duplicated from ZeMosaic

import os
import numpy as np
import logging
import warnings
from typing import Optional, Sequence, Tuple

USE_SCIPY_WINSOR = os.getenv("SEESTAR_USE_SCIPY_WINSOR", "0") == "1"
if USE_SCIPY_WINSOR:
    try:
        from scipy.stats.mstats import winsorize as _scipy_winsorize
        SCIPY_AVAILABLE = True
    except Exception:  # pragma: no cover - optional dependency
        _scipy_winsorize = None
        SCIPY_AVAILABLE = False
else:  # Prefer the NumPy fallback for better performance
    _scipy_winsorize = None
    SCIPY_AVAILABLE = False


try:  # optional acceleration
    import bottleneck as bn  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    bn = None

NANMEAN = bn.nanmean if bn else np.nanmean
NANSTD = bn.nanstd if bn else np.nanstd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provenance contract
# ---------------------------------------------------------------------------
#
# ``NaN`` marks a *missing* (spatially invalid) sample.  Nonlinear reduction
# kernels (median / kappa-sigma / linear-fit clip / winsorized sigma) must never
# treat ``NaN`` as a numeric observation.
#
# When ``return_weights=True`` the kernel returns ``(result, W, rejected_pct)``
# where ``W`` is the effective per-pixel / per-channel denominator such that
# ``result * W`` equals the numerator of the reduction.  Two reduced groups
# compose exactly via ``sum(result * W) / sum(W)`` for the linear weighted mean
# family.  For the nonlinear (rejection / median) family this is the *defined*
# bounded-memory hierarchical algorithm, not a global-exact statistic.
#
# ``W`` has the same trailing (spatial / channel) shape as ``result``:
# ``(H, W)`` for grey and ``(H, W, C)`` for colour inputs.


def _broadcast_weights(arr: np.ndarray, weights) -> np.ndarray:
    """Broadcast a scalar-per-image weight vector to the array shape."""
    w = np.asarray(weights, dtype=np.float32)
    shape = (arr.shape[0],) + (1,) * (arr.ndim - 1)
    return w.reshape(shape)


def _rejected_pct(mask: np.ndarray, valid: np.ndarray) -> float:
    """Percentage of *valid* samples rejected (missing samples excluded)."""
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return 0.0
    n_surv = int(np.count_nonzero(mask))
    return 100.0 * (n_valid - n_surv) / float(n_valid)


def _winsorize_axis0_numpy(arr: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
    """Vectorized winsorization along the first axis using NumPy.

    Matches ``scipy.stats.mstats.winsorize`` with the default
    ``inclusive=(True, True)`` semantics: the number of samples replaced on
    each side is ``floor(n * limit)`` (truncation) of the *valid* (non-NaN)
    samples per column.  ``NaN`` marks a *missing* sample: missing values are
    excluded from the order statistics (the low/high fractions are computed
    over the number of valid samples per column) and are preserved as ``NaN``
    in the result.
    """

    low, high = limits
    arr = arr.astype(np.float32, copy=False)
    result = arr.copy()

    valid = ~np.isnan(arr)
    n_valid = np.count_nonzero(valid, axis=0)

    if not np.any(n_valid > 0):
        return result

    # Sort ascending with NaN pushed to the end (NaN -> +inf).
    sort_key = np.where(valid, arr, np.inf)
    order = np.argsort(sort_key, axis=0)
    sorted_vals = np.take_along_axis(sort_key, order, axis=0)

    # ``rank`` is the inverse permutation of ``order``: ``rank[i, ...]`` is the
    # sorted position (0 = smallest) of the original sample ``i`` along axis 0.
    # Winsorization must replace samples by *ordered rank*, never by their
    # original memory position.  Invalid (NaN) samples sort last, so their rank
    # is >= n_valid and they are additionally excluded by ``valid`` below.
    rank = np.argsort(order, axis=0)

    if low > 0:
        # floor/truncation of ``low * n_valid`` (inclusive=True semantics),
        # matching scipy's ``int(low * n)``.
        lowidx = np.clip(np.floor(low * n_valid).astype(int), 0, None)
        low_bound = np.take_along_axis(sorted_vals, lowidx[np.newaxis], axis=0)
        low_sel = valid & (rank < lowidx[np.newaxis])
        if np.any(low_sel):
            result[low_sel] = np.broadcast_to(low_bound, result.shape)[low_sel]

    if high > 0:
        highidx = np.clip(np.floor(high * n_valid).astype(int), 0, None)
        upidx = np.clip(n_valid - highidx, 0, None)
        keep_idx = np.clip(upidx - 1, 0, None)
        high_bound = np.take_along_axis(sorted_vals, keep_idx[np.newaxis], axis=0)
        high_sel = valid & (rank >= upidx[np.newaxis])
        if np.any(high_sel):
            result[high_sel] = np.broadcast_to(high_bound, result.shape)[high_sel]

    return result


def _winsorize_bounds(arr: np.ndarray, limits: Tuple[float, float]):
    """Return ``(low_bound, high_bound)`` winsorized quantiles per column,
    computed over non-NaN samples only.

    ``low_bound`` is the value at the ``floor(low * n_valid)`` order statistic
    and ``high_bound`` the value at the ``n_valid - 1 - floor(high * n_valid)``
    order statistic of the valid samples along axis 0 — the same index
    convention as ``scipy.stats.mstats.winsorize`` with
    ``inclusive=(True, True)``.
    """
    low, high = limits
    valid = ~np.isnan(arr)
    n_valid = np.count_nonzero(valid, axis=0)
    sort_key = np.where(valid, arr, np.inf)
    order = np.argsort(sort_key, axis=0)
    sorted_vals = np.take_along_axis(sort_key, order, axis=0)

    max_idx = np.maximum(n_valid - 1, 0)
    lowidx = np.clip(np.floor(low * n_valid).astype(int), 0, max_idx)
    highidx = np.clip(
        n_valid - 1 - np.floor(high * n_valid).astype(int), 0, max_idx
    )

    low_b = np.take_along_axis(sorted_vals, lowidx[np.newaxis], axis=0)
    high_b = np.take_along_axis(sorted_vals, highidx[np.newaxis], axis=0)
    return low_b, high_b


def _stack_mean(images, weights=None, return_weights=False):
    arr = np.stack([im for im in images], axis=0).astype(np.float32)
    valid = ~np.isnan(arr)
    if weights is not None:
        w = _broadcast_weights(arr, weights)
        wv = np.where(valid, w, np.float32(0.0))
        sum_w = np.sum(wv, axis=0, dtype=np.float32)
        arr0 = np.where(valid, arr, np.float32(0.0))
        sum_d = np.sum(arr0 * wv, axis=0, dtype=np.float32)
        result = np.divide(
            sum_d, sum_w, out=np.zeros_like(sum_d), where=sum_w > 1e-9
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.nanmean(arr, axis=0)
        result = np.where(np.any(valid, axis=0), result, np.float32(0.0))
        sum_w = np.count_nonzero(valid, axis=0).astype(np.float32)
    result = result.astype(np.float32)
    if return_weights:
        return result, sum_w.astype(np.float32), 0.0
    return result, 0.0


def _stack_median(images, _weights=None, return_weights=False):
    arr = np.stack([im for im in images], axis=0).astype(np.float32)
    valid = ~np.isnan(arr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(arr, axis=0)
    result = np.where(np.any(valid, axis=0), result, np.float32(0.0))
    result = result.astype(np.float32)
    sum_w = np.count_nonzero(valid, axis=0).astype(np.float32)
    if return_weights:
        return result, sum_w, 0.0
    return result, 0.0


def _stack_kappa_sigma(
    images, weights=None, sigma_low=3.0, sigma_high=3.0, return_weights=False
):
    arr = np.stack([im for im in images], axis=0).astype(np.float32)
    valid = ~np.isnan(arr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(arr, axis=0)
        std = np.nanstd(arr, axis=0)
    low = med - sigma_low * std
    high = med + sigma_high * std
    mask = valid & (arr >= low) & (arr <= high)
    if weights is not None:
        w = _broadcast_weights(arr, weights)
        wm = np.where(mask, w, np.float32(0.0))
        sum_w = np.sum(wm, axis=0, dtype=np.float32)
        arr0 = np.where(mask, arr, np.float32(0.0))
        sum_d = np.sum(arr0 * wm, axis=0, dtype=np.float32)
        result = np.divide(
            sum_d, sum_w, out=np.zeros_like(sum_d), where=sum_w > 1e-6
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.nanmean(np.where(mask, arr, np.nan), axis=0)
        result = np.where(np.any(mask, axis=0), result, np.float32(0.0))
        sum_w = np.count_nonzero(mask, axis=0).astype(np.float32)
    rejected_pct = _rejected_pct(mask, valid)
    result = result.astype(np.float32)
    if return_weights:
        return result, sum_w.astype(np.float32), rejected_pct
    return result, rejected_pct


def _stack_linear_fit_clip(images, weights=None, sigma=3.0, return_weights=False):
    arr = np.stack([im for im in images], axis=0).astype(np.float32)
    valid = ~np.isnan(arr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(arr, axis=0)
        residuals = arr - median
        med_res = np.nanmedian(residuals, axis=0)
        std_res = np.nanstd(residuals, axis=0)
    mask = valid & (np.abs(residuals - med_res) <= sigma * std_res)
    if weights is not None:
        w = _broadcast_weights(arr, weights)
        wm = np.where(mask, w, np.float32(0.0))
        sum_w = np.sum(wm, axis=0, dtype=np.float32)
        arr0 = np.where(mask, arr, np.float32(0.0))
        sum_d = np.sum(arr0 * wm, axis=0, dtype=np.float32)
        result = np.divide(
            sum_d, sum_w, out=np.zeros_like(sum_d), where=sum_w > 1e-6
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.nanmean(np.where(mask, arr, np.nan), axis=0)
        result = np.where(np.any(mask, axis=0), result, np.float32(0.0))
        sum_w = np.count_nonzero(mask, axis=0).astype(np.float32)
    rejected_pct = _rejected_pct(mask, valid)
    result = result.astype(np.float32)
    if return_weights:
        return result, sum_w.astype(np.float32), rejected_pct
    return result, rejected_pct


def _stack_winsorized_sigma_iter(
    images: Sequence[np.ndarray],
    weights: Optional[np.ndarray],
    kappa: float = 3.0,
    winsor_limits: Tuple[float, float] = (0.05, 0.05),
    apply_rewinsor: bool = True,
    max_iters: int = 5,
    kappa_decay: float = 0.9,
    max_mem_bytes: int = int(os.getenv("SEESTAR_MAX_MEM", 2_000_000_000)),
    return_weights: bool = False,
) -> Tuple[np.ndarray, float]:

    """Iterative Winsorized sigma clipping.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        List or array of images ``(N, H, W)`` or ``(N, H, W, 3)``.  ``NaN``
        marks a missing (spatially invalid) sample.
    weights : Optional[np.ndarray]
        Optional weight array of shape ``(N,)``.
    kappa : float, optional
        Sigma clipping threshold. Defaults to ``3.0``.
    winsor_limits : Tuple[float, float], optional
        Fractional limits for Winsorization ``(low, high)``.
    apply_rewinsor : bool, optional
        Replace rejected pixels with their winsorized value if ``True`` (they
        remain in the mean with their weight), otherwise exclude them (``NaN``).
    max_iters : int, optional
        Maximum number of iterations. Defaults to ``5``.
    kappa_decay : float, optional
        Multiplicative decay for ``kappa`` at each iteration.
    max_mem_bytes : int, optional
        Abort if stacking would exceed this memory usage.
    return_weights : bool, optional
        When ``True`` return ``(result, W, rejected_pct)`` where ``W`` is the
        effective denominator (matching the ``apply_rewinsor`` definition).

    Returns
    -------
    Tuple[np.ndarray, float] or Tuple[np.ndarray, np.ndarray, float]
        Stacked image and rejection percentage (plus ``W`` when requested).
    """

    logger.debug(
        "Winsorized sigma clip start: kappa=%s limits=%s apply_rewinsor=%s",
        kappa,
        winsor_limits,
        apply_rewinsor,
    )

    shape = images[0].shape
    exp_bytes = len(images) * np.prod(shape) * 4
    if exp_bytes > max_mem_bytes:
        raise MemoryError("Stack exceeds max_mem_bytes")

    arr = np.stack([im.astype(np.float32, copy=False) for im in images], axis=0)

    # Missing samples are excluded from the very first iteration.
    mask = ~np.isnan(arr)
    valid = mask
    kappa_iter = float(kappa)

    for itr in range(max_iters):
        if SCIPY_AVAILABLE:
            arr_masked = np.ma.array(arr, mask=~mask)
            arr_w = _scipy_winsorize(
                arr_masked,
                limits=winsor_limits,
                axis=0,
                inclusive=(True, True),
            )
            arr_w_data = np.asarray(arr_w.filled(np.nan), dtype=np.float32)
            # scipy.stats.mstats.winsorize clears the mask and overwrites the
            # masked (missing / previously rejected) entries with the high
            # winsor bound.  Restore the current iteration mask so those
            # entries stay NaN and remain excluded from the location/scale
            # statistics computed below (matches the NumPy fallback).
            arr_w_data[~mask] = np.nan
        else:
            arr_masked = np.where(mask, arr, np.nan)
            arr_w_data = _winsorize_axis0_numpy(arr_masked, winsor_limits)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu_w = NANMEAN(arr_w_data, axis=0)
            sigma_w = NANSTD(arr_w_data, axis=0, ddof=1)

        # A column with fewer than two valid samples has an undefined sample
        # standard deviation (``nanstd(..., ddof=1)`` is NaN for a single valid
        # sample, and NaN for zero).  A NaN sigma would make ``low``/``high``
        # NaN and reject the only valid sample; rewinsorization over the
        # resulting empty survivor set then yields non-finite bounds.  Since
        # statistical clipping is undefined for <=1 valid sample, treat those
        # columns as a no-rejection identity: sigma == 0 makes low == high == mu
        # so the lone valid sample is always kept.  Zero-valid columns already
        # have mask all-False and remain non-contributing regardless.
        n_valid_col = np.count_nonzero(mask, axis=0)
        sigma_w = np.where(n_valid_col <= 1, np.float32(0.0), sigma_w)

        low = mu_w - kappa_iter * sigma_w
        high = mu_w + kappa_iter * sigma_w
        new_mask = mask & (arr >= low) & (arr <= high)
        n_rej = np.count_nonzero(mask) - np.count_nonzero(new_mask)
        logger.debug(
            "WinsorSig iter=%d : rej=%d (%.2f%%)",
            itr + 1,
            n_rej,
            100.0 * n_rej / max(mask.size, 1),
        )
        mask = new_mask
        if n_rej == 0:
            break
        if kappa_decay < 1.0:
            kappa_iter = kappa * (kappa_decay ** (itr + 1))

    if apply_rewinsor:
        # Rejected (valid but clipped) samples are substituted with the
        # nearest winsorized bound of the *survivor* distribution; missing
        # samples remain NaN and *survivors are preserved exactly*.  The
        # bounds come from the survivors only, so outliers do not contaminate
        # the substituted value.  A run with no rejection therefore returns
        # the original data unchanged — ``apply_rewinsor=True`` must never
        # alter surviving samples.
        low_b, high_b = _winsorize_bounds(
            np.where(mask, arr, np.nan), winsor_limits
        )
        clipped = np.clip(arr, low_b, high_b)
        arr_final = np.where(mask, arr, np.where(valid, clipped, np.nan))
    else:
        arr_final = np.where(mask, arr, np.nan)

    # arr_final is NaN exactly where a sample does NOT contribute to the mean.
    contrib = ~np.isnan(arr_final)

    if weights is not None:
        w = _broadcast_weights(arr, weights)
        sum_w = np.nansum(np.where(contrib, w, np.float32(0.0)), axis=0)
        sum_d = np.nansum(arr_final * w, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.divide(
                sum_d,
                sum_w,
                out=np.zeros_like(sum_d),
                where=sum_w > 1e-6,
            )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = NANMEAN(arr_final, axis=0)
        result = np.where(np.any(contrib, axis=0), result, np.float32(0.0))
        sum_w = np.count_nonzero(contrib, axis=0).astype(np.float32)

    rejected_pct = _rejected_pct(mask, valid)
    logger.debug("WinsorSig done : total rej=%.2f%%", rejected_pct)

    result = result.astype(np.float32)
    if return_weights:
        return result, sum_w.astype(np.float32), rejected_pct
    return result, rejected_pct


def _stack_winsorized_sigma(
    images: Sequence[np.ndarray],
    weights: Optional[np.ndarray],
    kappa: float = 3.0,
    winsor_limits: Tuple[float, float] = (0.05, 0.05),
    apply_rewinsor: bool = True,
    max_mem_bytes: Optional[int] = None,
    return_weights: bool = False,
) -> Tuple[np.ndarray, float]:
    """Compatibility wrapper for iterative Winsorized sigma clipping."""
    return _stack_winsorized_sigma_iter(
        images,
        weights,
        kappa=kappa,
        winsor_limits=winsor_limits,
        apply_rewinsor=apply_rewinsor,
        max_mem_bytes=max_mem_bytes
        if max_mem_bytes is not None
        else int(os.getenv("SEESTAR_MAX_MEM", 2_000_000_000)),
        return_weights=return_weights,
    )
