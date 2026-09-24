# Stacking algorithms duplicated from ZeMosaic

import os
import math
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
# Small-N gross-outlier guard (zero-rank regime) — constants
# ---------------------------------------------------------------------------
#
# When a column's winsor rank is zero (``floor(low * n_valid) ==
# floor(high * n_valid) == 0``), winsorization is the identity and the
# canonical mean/std clip band is dominated by a single huge outlier: it lets
# e.g. ``[100, 101, 99, 102, 63000]`` at N=5 through.  Those columns get a
# ONE-PASS translation/scale-conservative gross-outlier guard (not a sigma
# clip, never consuming kappa-decay iterations) — see ``_gross_outlier_keep``.
#
# The guard rejects only the lowest/highest extreme of a column, and only when
# BOTH (a) its gap to the survivor consensus exceeds GAP_FACTOR x the survivor
# span (a pure ratio: Gaussian noise has a top-order gap ~0.1x the survivor span
# regardless of mean/sigma, while a gross isolated outlier has one in the
# hundreds to thousands) AND (b) its deviation from the median exceeds
# DEV_FACTOR x |median| — a LOCAL RELATIVE scale.  Both criteria are invariant
# under positive multiplicative scaling (no ``max(|median|, 1)`` unit floor, no
# unit-dependent epsilon): scaling the whole column scales |median|, the gaps
# and the spans identically, so the decision is unchanged (normalized [0,1]
# floats and raw 16-bit counts alike).  Criterion (b) is conservative at
# median == 0 (there is no finite relative scale, so no deviation rejection) —
# that is what keeps a small lone value over a flat zero consensus (e.g.
# ``[0]*9 + [2]``) while still rejecting a gross outlier over a flat nonzero
# consensus (``[100]*4 + [63000]``, deviation ~629x the median).
_SMALL_N_GROSS_GAP_FACTOR = np.float32(100.0)
_SMALL_N_GROSS_DEV_FACTOR = np.float32(3.0)


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


def _winsor_schedule_kappas(kappa, kappa_decay, n_iters):
    """Kappa of every scheduled iteration, bitwise as in the reference.

    The reference narrows kappa ONLY when ``kappa_decay < 1.0`` and only
    after a rejecting iteration, so along a run that never stopped early
    iteration ``i`` uses ``kappa * kappa_decay**i`` (``i = 0`` uses the
    bare ``float(kappa)`` initialisation, bitwise identical to the
    ``* 1.0`` product).  With ``kappa_decay >= 1.0`` the band never
    narrows and every iteration uses ``float(kappa)``.

    This is the neutral schedule seam used by the exact-N spatial tiled
    driver (pass 1 schedule discovery / pass 2 replay); the FULL_CPU loop
    consumes the same schedule with its historical early-exit semantics.
    """
    k = float(kappa)
    if not (float(kappa_decay) < 1.0):
        return [k] * int(n_iters)
    return [k * (float(kappa_decay) ** i) for i in range(int(n_iters))]


def _winsorized_sigma_iteration_body(arr, mask, kappa_iter, winsor_limits):
    """One neutral Winsorized-sigma schedule iteration (no early exit).

    Verbatim mirror of the canonical loop body: winsorize (NumPy fallback or
    SciPy opt-in path) -> NANMEAN/NANSTD with the documented
    ``n_valid_col <= 1 -> sigma = 0`` rule -> ``low/high`` band -> mask
    update.  Returns ``(new_mask, n_rej)``; ``n_rej`` is the LOCAL per-
    iteration rejection count over this array (int).  The FULL_CPU loop
    keeps its historical early-exit and calls this helper; the exact-N
    spatial driver calls it on every tile with the full ``N`` stack axis
    preserved.
    """
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
    n_rej = int(np.count_nonzero(mask)) - int(np.count_nonzero(new_mask))
    return new_mask, n_rej


def _winsor_zero_rank_regime(limits, n):
    """Pure rank-0 detector (local equivalent of the planner predicate).

    ``floor(limit * n) == 0`` on both sides => winsorization is the identity
    on every pixel/iteration (per-pixel ``n_valid <= n``).  Negative or
    non-finite limits never qualify (mirrors the canonical helper: a
    negative limit means *no winsorization on that side*).  Weighted /
    unweighted does not change the regime (weights do not alter valid
    counts).  Mathematically equivalent to
    ``seestar.core.cpu_memory_planner.winsor_zero_rank_regime``.
    """
    low, high = float(limits[0]), float(limits[1])
    if not (low >= 0.0 and high >= 0.0):
        return False
    if not (np.isfinite(low) and np.isfinite(high)):
        return False
    n_int = int(n)
    return math.floor(low * n_int) == 0 and math.floor(high * n_int) == 0


def _winsor_zero_rank_cols(limits, n_valid_col):
    """Per-column zero-rank mask (vectorized ``_winsor_zero_rank_regime``).

    ``floor(low * n) == 0`` and ``floor(high * n) == 0`` for ``n =
    n_valid_col`` (the ACTUAL valid population of each column, from the
    original registration-NaN/support mask — never the nominal N_batch).
    Negative or non-finite limits never qualify (no column is zero-rank).
    Returns a boolean array shaped like ``n_valid_col``.
    """
    low, high = float(limits[0]), float(limits[1])
    if not (low >= 0.0 and high >= 0.0):
        return np.zeros_like(n_valid_col, dtype=bool)
    if not (np.isfinite(low) and np.isfinite(high)):
        return np.zeros_like(n_valid_col, dtype=bool)
    return (np.floor(low * n_valid_col) == 0) & (
        np.floor(high * n_valid_col) == 0
    )


def _gross_outlier_keep(arr, valid):
    """One-pass translation/scale-conservative gross-outlier guard (keep mask).

    Rejects only the lowest / highest EXTREME of each column, and only when it
    is BOTH (a) isolated by an extreme gap from the survivor consensus AND
    (b) far from the median in a LOCAL RELATIVE sense:

        high_gap = max1 - max2 ; high_survivor_span = max2 - min1
        low_gap  = min2 - min1 ; low_survivor_span  = max1 - min2
        reject extreme iff
            extreme_gap > GAP_FACTOR * survivor_span   AND
            |extreme - median| > DEV_FACTOR * |median|   (median != 0)

    Both criteria are invariant under positive multiplicative scaling (no
    ``max(|median|, 1)`` unit floor, no unit-dependent epsilon): the gap/span
    ratio and the deviation/|median| ratio scale identically with the data.
    Criterion (b) is conservative at ``median == 0`` (no finite relative scale,
    so no deviation rejection) — that keeps a small lone value over a flat zero
    consensus (``[0]*9 + [2]``) while still rejecting a gross outlier over a
    flat nonzero consensus (``[100]*4 + [63000]``).  One pass only;
    ``n_valid <= 2`` -> no safe rejection; ties (no unique extreme) -> no
    rejection.
    """
    n_valid = np.count_nonzero(valid, axis=0)
    if not np.any(n_valid >= 3):
        # No column has enough valid samples to guard (N < 3 or all columns
        # n_valid <= 2): no safe rejection, keep everything.
        return valid
    # Sort valid samples ascending; NaN -> +inf (kept out of the extremes).
    sort_key = np.where(valid, arr, np.float32(np.inf))
    sorted_vals = np.sort(sort_key, axis=0)  # (N, ...)
    del sort_key

    # Extract every value needed downstream as independent (H, W) arrays (min1
    # is copied to break the slice view) so the (N, H, W) sort output can be
    # released immediately, keeping the peak footprint low.
    min1 = sorted_vals[0].copy()
    min2 = sorted_vals[1]
    max1_idx = np.clip(n_valid - 1, 0, None)
    max2_idx = np.clip(n_valid - 2, 0, None)
    max1 = np.take_along_axis(sorted_vals, max1_idx[np.newaxis], axis=0)[0]
    max2 = np.take_along_axis(sorted_vals, max2_idx[np.newaxis], axis=0)[0]
    lo = np.clip((n_valid - 1) // 2, 0, None)
    hi = np.clip(n_valid // 2, 0, None)
    lo_val = np.take_along_axis(sorted_vals, lo[np.newaxis, ...], axis=0)[0]
    hi_val = np.take_along_axis(sorted_vals, hi[np.newaxis, ...], axis=0)[0]

    guardable = n_valid >= 3

    # Mixed tiles can contain both guardable columns and columns with no valid
    # samples.  The latter use +inf sentinels above; compute the gaps/spans on
    # SAFE finite operands (``np.where`` zeros the +inf sentinels on the
    # non-guardable columns) so no inf-inf NaN is ever produced, then mask the
    # rejection with ``guardable``.
    max1_s = np.where(guardable, max1, np.float32(0.0))
    max2_s = np.where(guardable, max2, np.float32(0.0))
    min1_s = np.where(guardable, min1, np.float32(0.0))
    min2_s = np.where(guardable, min2, np.float32(0.0))
    del sorted_vals, min2  # release the (N, H, W) sort output (min1 is a copy)
    high_gap = max1_s - max2_s
    high_span = max2_s - min1_s
    low_gap = min2_s - min1_s
    low_span = max1_s - min2_s

    high_a = high_gap > _SMALL_N_GROSS_GAP_FACTOR * high_span
    low_a = low_gap > _SMALL_N_GROSS_GAP_FACTOR * low_span

    # Local relative deviation criterion (scale-invariant): the extreme must be
    # far from the median relative to |median|.  ``median == 0`` has no finite
    # relative scale -> conservative (no deviation rejection).  The per-column
    # median is derived DIRECTLY from the already-sorted valid values (no second
    # nanmedian masked-array/sort allocation): the valid samples are
    # ``sorted_vals[0:n_valid]`` ascending, so the median is the middle value
    # (odd n_valid) or the mean of the two middle values (even n_valid) —
    # exactly ``np.nanmedian`` semantics for every valid-count case.
    median = np.where(
        (n_valid % 2) == 1, lo_val, (lo_val + hi_val) * np.float32(0.5)
    )
    median = np.where(n_valid == 0, np.float32(0.0), median)
    median_abs = np.abs(median)
    has_scale = median_abs > 0
    dev_threshold = _SMALL_N_GROSS_DEV_FACTOR * median_abs
    with np.errstate(invalid="ignore"):
        high_b = has_scale & (np.abs(max1 - median) > dev_threshold)
        low_b = has_scale & (np.abs(min1 - median) > dev_threshold)

    reject_high = high_a & high_b & guardable
    reject_low = low_a & low_b & guardable

    # A tie (max1 == max2, or min1 == min2) has a zero gap, so reject_* is
    # already False; the equality match below is harmless in that case.
    is_max1 = valid & (arr == max1[np.newaxis, ...])
    is_min1 = valid & (arr == min1[np.newaxis, ...])
    reject = (is_max1 & reject_high[np.newaxis, ...]) | (
        is_min1 & reject_low[np.newaxis, ...]
    )
    keep = valid & ~reject

    n_rej = int(np.count_nonzero(reject))
    if n_rej:
        logger.debug(
            "ROBUST_SMALL_N mode=gross_outlier_guard rejected=%d", n_rej
        )
    return keep



def _stack_winsorized_sigma_iter(
    images: Sequence[np.ndarray],
    weights: Optional[np.ndarray],
    kappa: float = 3.0,
    winsor_limits: Tuple[float, float] = (0.05, 0.05),
    apply_rewinsor: bool = True,
    max_iters: int = 5,
    kappa_decay: float = 0.9,
    max_mem_bytes: Optional[int] = None,
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
        Abort if stacking would exceed this memory usage.  ``None`` (the
        default) resolves to ``int(os.getenv("SEESTAR_MAX_MEM",
        2_000_000_000))`` — a backward-compatible default intended ONLY for
        non-production standalone callers (e.g. ``streaming_stack`` and
        direct tests).  The production queue path always resolves one byte
        budget in the queue Winsorized wrapper and threads it through the
        worker tuple verbatim (stage C budget contract), so it can never
        silently trigger this fallback.
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

    budget = (
        int(max_mem_bytes)
        if max_mem_bytes is not None
        else int(os.getenv("SEESTAR_MAX_MEM", 2_000_000_000))
    )

    shape = images[0].shape
    exp_bytes = len(images) * np.prod(shape) * 4
    if exp_bytes > budget:
        raise MemoryError("Stack exceeds max_mem_bytes")

    arr = np.stack([im.astype(np.float32, copy=False) for im in images], axis=0)

    # Missing samples are excluded from the very first iteration.
    mask = ~np.isnan(arr)
    valid = mask
    kappas = _winsor_schedule_kappas(kappa, kappa_decay, max_iters)

    # Per-column zero-rank classification from the ORIGINAL valid population
    # (registration NaNs / support mask), never the nominal N_batch.  A
    # column with floor(low*n_valid)==floor(high*n_valid)==0 is zero-rank and
    # gets the one-pass gross-outlier guard; a rank-sufficient column keeps
    # the historical iterative Winsor path (even if its survivor count later
    # drops below the boundary).
    n_valid_col = np.count_nonzero(valid, axis=0)
    zero_rank_cols = _winsor_zero_rank_cols(winsor_limits, n_valid_col)
    rank_cols = ~zero_rank_cols
    n_zero_rank = int(np.count_nonzero(zero_rank_cols))
    if n_zero_rank:
        logger.debug(
            "ROBUST_SMALL_N mode=gross_outlier_guard zero_rank_cols=%d "
            "rank_cols=%d n_batch=%d",
            n_zero_rank,
            int(np.count_nonzero(rank_cols)),
            len(images),
        )

    # One-pass gross-outlier guard for the zero-rank columns (frozen once;
    # never consumes kappa-decay iterations).
    guard_keep = _gross_outlier_keep(arr, valid) if n_zero_rank else valid

    # Historical iterative Winsor on the rank-sufficient columns only: the
    # zero-rank columns are NaN-masked out so they do not contribute to the
    # iteration count / early exit / kappa decay.  All-zero-rank fast path:
    # no rank columns -> the Winsor loop is skipped entirely (the guard alone
    # decides the zero-rank columns).
    rank_cols3 = rank_cols[np.newaxis, ...]  # (1, H, W[, C]) broadcast
    if np.any(rank_cols):
        arr_w = np.where(rank_cols3, arr, np.nan)
        mask_w = ~np.isnan(arr_w)  # == valid & rank_cols

        for itr in range(int(max_iters)):
            new_mask, n_rej = _winsorized_sigma_iteration_body(
                arr_w, mask_w, kappas[itr], winsor_limits
            )
            logger.debug(
                "WinsorSig iter=%d : rej=%d (%.2f%%)",
                itr + 1,
                n_rej,
                100.0 * n_rej / max(mask_w.size, 1),
            )
            mask_w = new_mask
            if n_rej == 0:
                break
    else:
        mask_w = valid & rank_cols3  # all-False (no rank columns)

    # Combine: zero-rank columns use the frozen guard mask; rank-sufficient
    # columns use the iterative Winsor mask.
    mask = np.where(rank_cols3, mask_w, guard_keep)

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
