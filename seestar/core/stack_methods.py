# Stacking algorithms duplicated from ZeMosaic

import numpy as np


def _stack_mean(images, weights=None):
    arr = np.stack([im for im in images], axis=0).astype(np.float32)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)[:, None, None]
        if arr.ndim == 4:
            w = w[..., None]
        sum_w = np.sum(w, axis=0)
        sum_d = np.sum(arr * w, axis=0)
        result = np.divide(sum_d, sum_w, out=np.zeros_like(sum_d), where=sum_w > 1e-9)
    else:
        result = np.mean(arr, axis=0)
    return result.astype(np.float32), 0.0


def _stack_median(images, _weights=None):
    arr = np.stack([im for im in images], axis=0).astype(np.float32)
    result = np.median(arr, axis=0)
    return result.astype(np.float32), 0.0


def _stack_kappa_sigma(images, weights=None, sigma_low=3.0, sigma_high=3.0):
    arr = np.stack([im for im in images], axis=0).astype(np.float32)
    med = np.median(arr, axis=0)
    std = np.std(arr, axis=0)
    low = med - sigma_low * std
    high = med + sigma_high * std
    mask = (arr >= low) & (arr <= high)
    arr_clip = np.where(mask, arr, np.nan)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)[:, None, None]
        if arr.ndim == 4:
            w = w[..., None]
        sum_w = np.nansum(w * mask, axis=0)
        sum_d = np.nansum(arr_clip * w, axis=0)
        result = np.divide(sum_d, sum_w, out=np.zeros_like(sum_d), where=sum_w > 1e-6)
    else:
        result = np.nanmean(arr_clip, axis=0)
    rejected_pct = 100.0 * (mask.size - np.count_nonzero(mask)) / float(mask.size)
    return result.astype(np.float32), rejected_pct


def _stack_linear_fit_clip(images, weights=None, sigma=3.0):
    arr = np.stack([im for im in images], axis=0).astype(np.float32)
    median = np.median(arr, axis=0)
    residuals = arr - median
    med_res = np.median(residuals, axis=0)
    std_res = np.std(residuals, axis=0)
    mask = np.abs(residuals - med_res) <= sigma * std_res
    arr_clip = np.where(mask, arr, np.nan)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)[:, None, None]
        if arr.ndim == 4:
            w = w[..., None]
        sum_w = np.nansum(w * mask, axis=0)
        sum_d = np.nansum(arr_clip * w, axis=0)
        result = np.divide(sum_d, sum_w, out=np.zeros_like(sum_d), where=sum_w > 1e-6)
    else:
        result = np.nanmean(arr_clip, axis=0)
    rejected_pct = 100.0 * (mask.size - np.count_nonzero(mask)) / float(mask.size)
    return result.astype(np.float32), rejected_pct


def _stack_winsorized_sigma(
    images,
    weights,
    kappa=3.0,
    winsor_limits=(0.05, 0.05),
    apply_rewinsor=True,
):
    """Winsorized sigma clip stacking used by the queue manager."""
    from scipy.stats.mstats import winsorize
    from astropy.stats import sigma_clipped_stats

    arr = np.stack([im for im in images], axis=0).astype(np.float32)
    arr_w = winsorize(arr, limits=winsor_limits, axis=0)
    try:
        _, med, std = sigma_clipped_stats(arr_w, sigma=3.0, axis=0, maxiters=5)
    except TypeError:
        _, med, std = sigma_clipped_stats(
            arr_w, sigma_lower=3.0, sigma_upper=3.0, axis=0, maxiters=5
        )
    low = med - kappa * std
    high = med + kappa * std
    mask = (arr >= low) & (arr <= high)
    if apply_rewinsor:
        arr_clip = np.where(mask, arr, arr_w)
    else:
        arr_clip = np.where(mask, arr, np.nan)
    if weights is not None:
        w = np.asarray(weights)[:, None, None]
        if arr.ndim == 4:
            w = w[..., None]
        sum_w = np.nansum(w * mask, axis=0)
        sum_d = np.nansum(arr_clip * w, axis=0)
        result = np.divide(sum_d, sum_w, out=np.zeros_like(sum_d), where=sum_w > 1e-6)
    else:
        result = np.nanmean(arr_clip, axis=0)
    rejected_pct = 100.0 * (mask.size - np.count_nonzero(mask)) / float(mask.size)
    return result.astype(np.float32), rejected_pct

