# Stacking algorithms duplicated from ZeMosaic

import os
import numpy as np
import logging
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


def _winsorize_axis0_numpy(arr: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
    """Vectorized winsorization along the first axis using NumPy.

    This replicates the behaviour of ``scipy.stats.mstats.winsorize`` with
    ``inclusive=(False, False)`` which matches the default of SciPy. The
    implementation is significantly faster for large arrays than the SciPy
    version which relies on ``apply_along_axis``.
    """

    low, high = limits
    arr = arr.astype(np.float32, copy=False)
    result = arr.copy()

    n = arr.shape[0]
    order = np.argsort(arr, axis=0)

    if low > 0:
        lowidx = int(round(low * n))
        low_values = np.take_along_axis(arr, order, axis=0)[lowidx]
        idx = order[:lowidx]
        np.put_along_axis(result, idx, low_values, axis=0)

    if high > 0:
        upidx = n - int(round(high * n))
        up_values = np.take_along_axis(arr, order, axis=0)[upidx - 1]
        idx = order[upidx:]
        np.put_along_axis(result, idx, up_values, axis=0)

    return result


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
        result = NANMEAN(arr_clip, axis=0)
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
        result = NANMEAN(arr_clip, axis=0)
    rejected_pct = 100.0 * (mask.size - np.count_nonzero(mask)) / float(mask.size)
    return result.astype(np.float32), rejected_pct


def _stack_winsorized_sigma_iter(
    images: Sequence[np.ndarray],
    weights: Optional[np.ndarray],
    kappa: float = 3.0,

    winsor_limits: Tuple[float, float] = (0.05, 0.05),
    apply_rewinsor: bool = True,
    max_iters: int = 5,
    kappa_decay: float = 0.9,
    max_mem_bytes: int = int(os.getenv("SEESTAR_MAX_MEM", 2_000_000_000)),
) -> Tuple[np.ndarray, float]:

    """Iterative Winsorized sigma clipping.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        List or array of images ``(N, H, W)`` or ``(N, H, W, 3)``.
    weights : Optional[np.ndarray]
        Optional weight array of shape ``(N,)``.
    kappa : float, optional
        Sigma clipping threshold. Defaults to ``3.0``.

    winsor_limits : Tuple[float, float], optional

        Fractional limits for Winsorization ``(low, high)``.
    apply_rewinsor : bool, optional
        Replace rejected pixels with their winsorized value if ``True``,
        otherwise with ``NaN``.
    max_iters : int, optional
        Maximum number of iterations. Defaults to ``5``.
    kappa_decay : float, optional
        Multiplicative decay for ``kappa`` at each iteration.

    max_mem_bytes : int, optional
        Abort if stacking would exceed this memory usage. Defaults to the value
        of the ``SEESTAR_MAX_MEM`` environment variable (2 GB if unset).

    Returns
    -------
    Tuple[np.ndarray, float]

        Stacked image and rejection percentage.

    Examples
    --------
    >>> import numpy as np
    >>> from seestar.core.stack_methods import _stack_winsorized_sigma
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(0, 1, size=(5, 4, 4)).astype(np.float32)
    >>> data[0, 0, 0] = 50
    >>> out, pct = _stack_winsorized_sigma(data, None)
    >>> out.shape
    (4, 4)
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

    mask = np.ones_like(arr, dtype=bool)
    kappa_iter = float(kappa)

    for itr in range(max_iters):
        if SCIPY_AVAILABLE:
            arr_masked = np.ma.array(arr, mask=~mask)
            arr_w = _scipy_winsorize(arr_masked, limits=winsor_limits, axis=0)
            arr_w_data = np.asarray(arr_w.filled(np.nan), dtype=np.float32)
        else:
            arr_masked = np.where(mask, arr, np.nan)
            arr_w_data = _winsorize_axis0_numpy(arr_masked, winsor_limits)

        mu_w = NANMEAN(arr_w_data, axis=0)
        sigma_w = NANSTD(arr_w_data, axis=0, ddof=1)

        low = mu_w - kappa_iter * sigma_w
        high = mu_w + kappa_iter * sigma_w
        new_mask = mask & (arr >= low) & (arr <= high)
        n_rej = np.count_nonzero(mask) - np.count_nonzero(new_mask)
        logger.debug(
            "WinsorSig iter=%d : rej=%d (%.2f%%)",
            itr + 1,
            n_rej,
            100.0 * n_rej / mask.size,
        )
        mask = new_mask
        if n_rej == 0:
            break
        if kappa_decay < 1.0:
            kappa_iter = kappa * (kappa_decay ** (itr + 1))

    if SCIPY_AVAILABLE:
        arr_masked = np.ma.array(arr, mask=~mask)
        arr_w_final = _scipy_winsorize(arr_masked, limits=winsor_limits, axis=0)
        arr_w_final = np.asarray(arr_w_final.filled(np.nan), dtype=np.float32)
    else:
        arr_masked = np.where(mask, arr, np.nan)
        arr_w_final = _winsorize_axis0_numpy(arr_masked, winsor_limits)

    arr_nan = np.where(mask, arr, np.nan)
    if apply_rewinsor:
        arr_final = np.where(mask, arr, arr_w_final)
    else:
        arr_final = arr_nan

    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)[:, None, None]
        if arr.ndim == 4:
            w = w[..., None]
        sum_w = np.nansum(w * mask, axis=0)
        sum_d = np.nansum(arr_nan * w, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):

            result = np.divide(
                sum_d,
                sum_w,
                out=np.zeros_like(sum_d),
                where=sum_w > 1e-6,
            )
    else:
        result = NANMEAN(arr_final, axis=0)


    rejected_pct = 100.0 * (mask.size - np.count_nonzero(mask)) / float(mask.size)
    logger.debug("WinsorSig done : total rej=%.2f%%", rejected_pct)

    return result.astype(np.float32), rejected_pct


def _stack_winsorized_sigma(
    images: Sequence[np.ndarray],
    weights: Optional[np.ndarray],
    kappa: float = 3.0,
    winsor_limits: Tuple[float, float] = (0.05, 0.05),
    apply_rewinsor: bool = True,
    max_mem_bytes: Optional[int] = None,
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
    )

