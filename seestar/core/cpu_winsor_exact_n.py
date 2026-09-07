"""Exact-N SPATIAL CPU tiling of the Winsorized sigma reduction (8.4.0 stage C).

Mission ``zsss-840-prew80-20260907``.  This module is the CPU twin of the
GPU driver ``seestar.core.stack_gpu.stack_winsorized_sigma_gpu_tiled``: it
processes the SPATIAL dimensions in tiles while the full ``N`` stack
population is preserved for EVERY output pixel (the stack axis is never
split; no hierarchical nonlinear reduction).  It reuses the SAME canonical
per-iteration helper (``stack_methods._winsorized_sigma_iteration_body``),
the same ``_winsorize_bounds`` / ``NANMEAN`` / ``NANSTD`` primitives and the
same deterministic kappa schedule as the untiled reference, so the output is
bitwise-identical to ``_stack_winsorized_sigma_iter`` (axis-0 accumulation is
over the identical N elements per column; there is no CPU micro-tile
kernel-switch caveat).

Global-iteration coordination (proven two-pass schedule, mirror of the GPU
driver):
* pass 1 (schedule discovery): every tile runs the DETERMINISTIC kappa
  schedule ``kappa * decay**i`` (i = 0 .. max_iters-1) with no early exit
  and returns its LOCAL per-iteration rejection counts; counts are SUMMED
  across tiles; ``z_eff`` = first iteration with global count 0 (or
  ``max_iters`` when the run exhausts the schedule without a zero-rejection
  iteration; ``max_iters == 1`` always has ``z_eff == max_iters``).
* pass 2 (exact replay): every tile re-runs exactly ``z_eff`` schedule
  iterations (no counting), finalises the survivor mask, applies
  ``apply_rewinsor`` (per-column bounds over the TILE's own survivors) and
  the weighted/unweighted final reduction, then EXACT PLACEMENT into the
  output arrays (no blend/feather/halo).

``rejected_pct`` is the GLOBAL sum/sum formula
(``100 * (sum(n_valid) - sum(n_surv)) / sum(n_valid)`` over all tiles), never
an average of per-tile percentages.

Refusal / retry semantics
-------------------------
* The driver never reduces N and never returns an empty success.  It reuses
  the pure planner's vocabulary: ``CPU_MIN_TILE_OUT`` (minimum viable spatial
  outputs per tile) and the stable planner refusal reasons
  (``REASON_MIN_TILE_EXCEEDS_BUDGET`` / ``REASON_NO_VALID_TILE`` /
  ``REASON_BUDGET_NEGATIVE``) via :class:`CpuWinsorMemoryRefused`.
* When ``max_mem_bytes`` is given, each candidate geometry is preflight-checked
  against the conservative stage-B working-set model (planner factors +
  scratch); candidates that cannot fit are refused before any allocation.
* On a catchable ``MemoryError`` while running a candidate geometry, the
  driver retries with a strictly smaller SPATIAL tile (tile_h / tile_w only —
  N, reducer, kappa, winsor limits, normalization and weights are never
  retry-mutable). ``max_retries`` means smaller-geometry attempts AFTER the
  initial planned attempt, so total execution attempts are bounded by
  ``max_retries + 1``; refusal follows exhaustion of that bound or of the
  admissible geometry sequence.

Fast path: the pure rank-0 detector
(``cpu_memory_planner.winsor_zero_rank_regime``) selects the fast MEMORY
envelope used by the planner; the canonical NumPy helper allocates sort
temporaries even in the rank-0 regime, so the reducer takes NO rank-0
shortcut (documented in the stage-B evidence; the driver math is identical
for both regimes).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from seestar.core.stack_methods import (
    NANMEAN,
    _broadcast_weights,
    _winsor_schedule_kappas,
    _winsorize_bounds,
    _winsorized_sigma_iteration_body,
)
from seestar.core.cpu_memory_planner import (
    CPU_MIN_TILE_OUT,
    CPU_WINSOR_FAST_FACTOR,
    CPU_WINSOR_FAST_SCRATCH_BYTES,
    CPU_WINSOR_SLOW_FACTOR,
    CPU_WINSOR_SLOW_SCRATCH_BYTES,
    REASON_BUDGET_NEGATIVE,
    REASON_MIN_TILE_EXCEEDS_BUDGET,
    REASON_NO_VALID_TILE,
    winsor_zero_rank_regime,
)

__all__ = [
    "stack_winsorized_sigma_cpu_tiled",
    "CpuWinsorMemoryRefused",
    "winsor_tile_slices",
    "cpu_tile_demand_bytes",
    "CPU_MIN_TILE_OUT",
]


class CpuWinsorMemoryRefused(MemoryError):
    """Engine refusal carrying the truthful planner reason.

    Raised when no scientifically valid spatial geometry fits the available
    budget (never a reduced-N fallback, never an empty success).
    """

    def __init__(self, reason, *, details=None):
        self.reason = str(reason)
        self.details = dict(details or {})
        super().__init__(
            f"CPU winsorized memory refusal: {self.reason} "
            f"(N never reduced; details={self.details})"
        )


def winsor_tile_slices(frame_shape, tile_shape):
    """Spatial slices of one frame for a tiling geometry (exact placement).

    Mirror of the GPU helper ``_winsor_tile_slices``:
    ``frame_shape`` = ``(H, W)``; ``tile_shape`` = ``None`` (whole frame),
    ``int`` / ``(tile_h,)`` (full-width row bands, last band partial) or
    ``(tile_h, tile_w)`` (rectangular tiles, row-major, last row band and
    last column partial).  Never splits the stack axis.  Returns a list of
    ``(y0, y1, x0, x1)`` tuples.
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


def cpu_tile_demand_bytes(n, tile_h, tile_w, channels, itemsize=4,
                          winsor_limits=(0.05, 0.05), scipy_backend=False):
    """Conservative incremental working-set demand of one CPU tile (stage B).

    Uses the measured planner envelopes: numpy slow 13.0x + 16 MiB, numpy
    rank-0 fast 12.0x + 16 MiB, scipy opt-in 17.5x + 32 MiB.  The demand is
    the incremental working set ABOVE the already-resident full input
    (resident observations counted once elsewhere).
    """
    n = int(n)
    if scipy_backend:
        factor = 17.5
        scratch = 32 * 1024 * 1024
    elif winsor_zero_rank_regime(winsor_limits, n):
        factor = CPU_WINSOR_FAST_FACTOR
        scratch = CPU_WINSOR_FAST_SCRATCH_BYTES
    else:
        factor = CPU_WINSOR_SLOW_FACTOR
        scratch = CPU_WINSOR_SLOW_SCRATCH_BYTES
    base = n * int(tile_h) * int(tile_w) * int(channels) * int(itemsize)
    return int(base * factor) + scratch, factor, scratch


def _tile_full_cell_outputs(frame_shape, tile_shape, min_tile_out):
    """Validate a geometry's full-cell spatial outputs >= min_tile_out.

    Returns ``(ok, reason, full_cell_outputs)``.  A geometry whose FULL cells
    (first tile_h x tile_w block) fall below the minimum viable tile is
    rejected with the planner reason (mirrors the GPU planner's minimum-tile
    contract; partial edge tiles are allowed because every FULL cell is >=
    the floor and partials only exist at frame edges).
    """
    H, W = int(frame_shape[0]), int(frame_shape[1])
    if tile_shape is None:
        return True, None, H * W
    if isinstance(tile_shape, (tuple, list)) and len(tile_shape) >= 2:
        tile_h, tile_w = int(tile_shape[0]), int(tile_shape[1])
    else:
        tile_h = int(tile_shape[0]) if isinstance(tile_shape, (tuple, list)) else int(tile_shape)
        tile_w = W
    tile_h = min(max(1, tile_h), H)
    tile_w = min(max(1, tile_w), W)
    full_out = tile_h * tile_w
    if full_out < int(min_tile_out):
        return False, REASON_NO_VALID_TILE, full_out
    return True, None, full_out


def _shrink_geometry(frame_shape, tile_shape, min_tile_out):
    """Strictly smaller admissible spatial geometry, or ``None``.

    Spatial-only retry vocabulary: halves tile_h first, then tile_w, never
    below 1; refuses to produce a geometry whose full-cell outputs fall below
    ``min_tile_out``.  N / kappa / limits / normalization / weights are not
    part of the geometry and are therefore never touched by a retry.
    """
    H, W = int(frame_shape[0]), int(frame_shape[1])
    if tile_shape is None:
        return None
    if isinstance(tile_shape, (tuple, list)) and len(tile_shape) >= 2:
        tile_h, tile_w = int(tile_shape[0]), int(tile_shape[1])
    else:
        tile_h = int(tile_shape[0]) if isinstance(tile_shape, (tuple, list)) else int(tile_shape)
        tile_w = W
    candidates = []
    h2 = max(1, tile_h // 2)
    if h2 != tile_h:
        candidates.append((h2, tile_w))
    w2 = max(1, tile_w // 2)
    if w2 != tile_w:
        candidates.append((tile_h, w2))
    if h2 != tile_h and w2 != tile_w:
        candidates.append((h2, w2))
    for th, tw in candidates:
        th = min(th, H)
        tw = min(max(1, tw), W)
        if th * tw >= int(min_tile_out):
            return (th, tw)
    return None


def _winsorized_tile_iterations_np(
    arr_t, kappa, winsor_limits, n_iters, kappa_decay, collect_counts
):
    """Run ``n_iters`` schedule iterations on one tile (full stack axis kept).

    Verbatim canonical per-iteration body (no early exit).  Returns
    ``(final_mask, counts)`` where ``counts`` is a list of local per-iteration
    rejection counts when ``collect_counts`` else ``None``.
    """
    mask = ~np.isnan(arr_t)
    kappas = _winsor_schedule_kappas(kappa, kappa_decay, n_iters)
    counts = [] if collect_counts else None
    for itr in range(int(n_iters)):
        new_mask, n_rej = _winsorized_sigma_iteration_body(
            arr_t, mask, kappas[itr], winsor_limits
        )
        if collect_counts:
            counts.append(n_rej)
        mask = new_mask
    return mask, counts


def _winsorized_tile_finalize_np(
    arr_t, mask_t, valid_t, weights, apply_rewinsor, winsor_limits
):
    """Final tail of one tile: rewinsor substitution + weighted/unweighted
    reduction (canonical tail on the tile's columns)."""
    if apply_rewinsor:
        low_b, high_b = _winsorize_bounds(
            np.where(mask_t, arr_t, np.nan), winsor_limits
        )
        clipped = np.clip(arr_t, low_b, high_b)
        arr_final = np.where(
            mask_t, arr_t, np.where(valid_t, clipped, np.nan)
        )
    else:
        arr_final = np.where(mask_t, arr_t, np.nan)
    contrib = ~np.isnan(arr_final)
    if weights is not None:
        w = _broadcast_weights(arr_t, weights)
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
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            result = NANMEAN(arr_final, axis=0)
        result = np.where(
            np.any(contrib, axis=0), result, np.float32(0.0)
        )
        sum_w = np.count_nonzero(contrib, axis=0).astype(np.float32)
    return np.asarray(result, dtype=np.float32), np.asarray(sum_w, dtype=np.float32)


def _materialize_spatial_tile(images, y0, y1, x0, x1):
    """Materialize exactly one ``N x tile_h x tile_w [x C]`` float32 cube.

    ``np.asarray`` on an ndarray or memmap is a zero-copy view; spatial
    slicing therefore happens before ``np.stack`` allocates the tile cube.
    In particular, this helper never asks NumPy to stack complete frames.
    """
    views = [np.asarray(image)[y0:y1, x0:x1, ...] for image in images]
    return np.stack(views, axis=0).astype(np.float32, copy=False)


def _run_tiled_geometry(
    images,
    frame_shape,
    spatial,
    weights,
    kappa,
    winsor_limits,
    apply_rewinsor,
    max_iters,
    kappa_decay,
):
    """Two-pass exact-N spatial reduction over ``spatial`` slices.

    ``images`` contains the already-resident observations.  Only the current
    spatial tile is stacked, independently in each pass.  Returns
    ``(result, sum_w, rejected_pct, z_eff)`` with exact placement.
    """
    H, W = int(frame_shape[0]), int(frame_shape[1])
    trailing_shape = tuple(frame_shape[2:])
    out_shape = (H, W) + trailing_shape
    result = np.empty(out_shape, dtype=np.float32)
    sum_w = np.empty(out_shape, dtype=np.float32)

    # ---- pass 1: schedule discovery (deterministic kappa schedule, no
    # early exit, LOCAL rejection counts summed globally)
    if int(max_iters) == 1:
        z_eff = 1
    else:
        global_counts = [0] * int(max_iters)
        for (y0, y1, x0, x1) in spatial:
            arr_t = _materialize_spatial_tile(images, y0, y1, x0, x1)
            _, counts = _winsorized_tile_iterations_np(
                arr_t, kappa, winsor_limits, int(max_iters), kappa_decay,
                collect_counts=True,
            )
            for i, c in enumerate(counts):
                global_counts[i] += c
        z_eff = int(max_iters)
        for i, c in enumerate(global_counts):
            if c == 0:
                z_eff = i
                break

    # ---- pass 2: exact replay of z_eff schedule iterations per tile and
    # exact-placement reconstruction + GLOBAL rejection accounting
    n_valid_total = 0
    n_surv_total = 0
    for (y0, y1, x0, x1) in spatial:
        arr_t = _materialize_spatial_tile(images, y0, y1, x0, x1)
        valid_t = ~np.isnan(arr_t)
        if z_eff == 0:
            mask_t = valid_t  # reference iteration 0 rejected nothing
        else:
            mask_t, _ = _winsorized_tile_iterations_np(
                arr_t, kappa, winsor_limits, z_eff, kappa_decay,
                collect_counts=False,
            )
        res_t, sumw_t = _winsorized_tile_finalize_np(
            arr_t, mask_t, valid_t, weights, apply_rewinsor, winsor_limits
        )
        n_valid_total += int(np.count_nonzero(valid_t))
        n_surv_total += int(np.count_nonzero(mask_t))
        result[y0:y1, x0:x1] = res_t
        sum_w[y0:y1, x0:x1] = sumw_t

    if n_valid_total == 0:
        rejected_pct = 0.0
    else:
        rejected_pct = (
            100.0 * (n_valid_total - n_surv_total) / float(n_valid_total)
        )
    return result, sum_w, rejected_pct, z_eff


def stack_winsorized_sigma_cpu_tiled(
    images,
    weights=None,
    kappa=3.0,
    winsor_limits=(0.05, 0.05),
    apply_rewinsor=True,
    max_iters=5,
    kappa_decay=0.9,
    return_weights=False,
    tile_shape=None,
    max_mem_bytes=None,
    min_tile_out=CPU_MIN_TILE_OUT,
    max_retries=4,
    _tile_order="rowmajor",
    _retry_callback: Optional[Callable[..., None]] = None,
):
    """Exact-N SPATIAL CPU tiling of the Winsorized sigma reduction.

    Same scientific twin, same parameters, same return contract as
    :func:`seestar.core.stack_methods._stack_winsorized_sigma_iter` —
    ``(result, rejected_pct)`` or, with ``return_weights=True``,
    ``(result, sum_w, rejected_pct)`` (float32 arrays + Python float) — but
    the SPATIAL dimensions are processed in tiles so the per-tile working set
    is ``N x tile_h x tile_w x C`` while the full ``N`` stack population is
    preserved for every output pixel.

    ``tile_shape``: ``None`` (or a geometry covering the whole frame in one
    tile) delegates to the untiled twin; ``int`` / ``(tile_h,)`` selects
    full-width row bands; ``(tile_h, tile_w)`` selects rectangular tiles.
    ``_tile_order`` is a test-only hook (``"rowmajor"`` / ``"reversed"``)
    proving tile-order invariance.  ``max_mem_bytes`` enables the
    conservative preflight + bounded spatial retry on MemoryError.
    ``max_retries`` is the number of smaller-geometry execution attempts
    allowed after the initial planned attempt (total attempts are therefore
    at most ``max_retries + 1``).  ``_retry_callback`` is an internal
    provenance seam called for every allocation recovery transition.
    """
    n = int(len(images))
    if n == 0:  # pragma: no cover - degenerate, mirrors CPU failure
        raise ValueError("tiled winsorized sigma requires at least one image")
    first = np.asarray(images[0])
    if first.ndim not in (2, 3):
        raise ValueError(
            "tiled winsorized sigma expects HxW or HxWxC observations"
        )
    frame = first.shape
    H, W = int(frame[0]), int(frame[1])
    C = int(frame[2]) if first.ndim == 3 else 1
    budget = int(max_mem_bytes) if max_mem_bytes is not None else None
    spatial = winsor_tile_slices((H, W), tile_shape)
    if len(spatial) == 1:
        # Untiled / full-frame geometry: the untiled twin IS the reference
        # implementation; delegate for guaranteed bitwise identity.  When a
        # budget is supplied the conservative model preflight applies first so
        # an oversized FULL geometry refuses with the planner reason instead
        # of allocating (never reduces N, never an empty success).
        if budget is not None:
            if budget <= 0:
                raise CpuWinsorMemoryRefused(
                    REASON_BUDGET_NEGATIVE, details={"max_mem_bytes": budget}
                )
            demand, _f, _s = cpu_tile_demand_bytes(
                n, H, W, C, 4, winsor_limits
            )
            if demand > budget:
                raise CpuWinsorMemoryRefused(
                    REASON_MIN_TILE_EXCEEDS_BUDGET,
                    details={
                        "tile_shape": "full",
                        "demand_bytes": demand,
                        "max_mem_bytes": budget,
                    },
                )
        from seestar.core.stack_methods import _stack_winsorized_sigma_iter

        return _stack_winsorized_sigma_iter(
            images,
            weights,
            kappa=kappa,
            winsor_limits=winsor_limits,
            apply_rewinsor=apply_rewinsor,
            max_iters=max_iters,
            kappa_decay=kappa_decay,
            max_mem_bytes=budget,
            return_weights=return_weights,
        )
    if _tile_order == "reversed":
        spatial = list(reversed(spatial))

    ok, reason, _full_out = _tile_full_cell_outputs(
        (H, W), tile_shape, min_tile_out
    )
    if not ok:
        raise CpuWinsorMemoryRefused(reason, details={"min_tile_out": int(min_tile_out)})

    if budget is not None and budget <= 0:
        raise CpuWinsorMemoryRefused(
            REASON_BUDGET_NEGATIVE, details={"max_mem_bytes": budget}
        )

    def _preflight(tile_shape_candidate):
        if budget is None:
            return True
        cand_spatial = winsor_tile_slices((H, W), tile_shape_candidate)
        if len(cand_spatial) == 1:
            return True
        if isinstance(tile_shape_candidate, (tuple, list)) and len(tile_shape_candidate) >= 2:
            th0, tw0 = int(tile_shape_candidate[0]), int(tile_shape_candidate[1])
        else:
            th0 = int(tile_shape_candidate[0]) if isinstance(tile_shape_candidate, (tuple, list)) else int(tile_shape_candidate)
            tw0 = W
        demand, _f, _s = cpu_tile_demand_bytes(
            n, min(th0, H), min(max(1, tw0), W), C, 4, winsor_limits
        )
        return demand <= budget

    # Candidate geometries: initial, then strictly smaller spatial tiles.
    candidates = []
    cand = tile_shape
    seen = set()
    while cand is not None:
        key = tuple(cand) if isinstance(cand, (tuple, list)) else (int(cand), W)
        if key not in seen:
            seen.add(key)
            candidates.append(cand)
        cand = _shrink_geometry((H, W), cand, min_tile_out)
    if not candidates:
        raise CpuWinsorMemoryRefused(
            REASON_NO_VALID_TILE, details={"min_tile_out": int(min_tile_out)}
        )

    executable_candidates = [cand for cand in candidates if _preflight(cand)]
    if not executable_candidates:
        raise CpuWinsorMemoryRefused(
            REASON_MIN_TILE_EXCEEDS_BUDGET,
            details={"max_mem_bytes": budget, "min_tile_out": int(min_tile_out)},
        )

    last_memory_error = None
    attempts = 0
    retry_budget = max(0, int(max_retries))
    max_attempts = retry_budget + 1
    attempted_shapes = []
    bounded_candidates = executable_candidates[:max_attempts]
    for candidate_index, cand in enumerate(bounded_candidates):
        attempts += 1
        attempted_shapes.append(
            tuple(cand) if isinstance(cand, (tuple, list)) else (int(cand),)
        )
        try:
            spatial_cand = winsor_tile_slices((H, W), cand)
            if _tile_order == "reversed" and spatial_cand:
                spatial_cand = list(reversed(spatial_cand))
            result, sum_w, rejected_pct, z_eff = _run_tiled_geometry(
                images,
                frame,
                spatial_cand,
                weights,
                kappa,
                winsor_limits,
                apply_rewinsor,
                max_iters,
                kappa_decay,
            )
            if attempts > 1 and _retry_callback is not None:
                _retry_callback(
                    attempt=attempts,
                    old_tile_shape=bounded_candidates[candidate_index - 1],
                    new_tile_shape=cand,
                    reason="allocation_failure",
                    outcome="recovered",
                )
            if return_weights:
                return result, sum_w, rejected_pct
            return result, rejected_pct
        except MemoryError as mem_err:
            last_memory_error = mem_err
            next_cand = (
                bounded_candidates[candidate_index + 1]
                if candidate_index + 1 < len(bounded_candidates)
                else None
            )
            if _retry_callback is not None:
                _retry_callback(
                    attempt=attempts,
                    old_tile_shape=cand,
                    new_tile_shape=next_cand,
                    reason="allocation_failure",
                    outcome="retrying" if next_cand is not None else "exhausted",
                )
            continue

    if isinstance(last_memory_error, CpuWinsorMemoryRefused):
        raise last_memory_error
    raise CpuWinsorMemoryRefused(
        REASON_MIN_TILE_EXCEEDS_BUDGET,
        details={
            "attempts": attempts,
            "max_retries": retry_budget,
            "attempted_tile_shapes": attempted_shapes,
            "n": n,
        },
    )
