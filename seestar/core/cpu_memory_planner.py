"""Pure CPU memory planner for the Winsorized sigma reduction (8.4.0, stage B).

Mission ``zsss-840-prew80-20260907``.  Stage B delivers ONLY the planner and
its evidence — nothing here wires the planner into production execution
(wiring is stage C/E).  This module mirrors the shape of the GPU planner
(``seestar.core.gpu_vram_planner``) but for the canonical CPU path
(``seestar.core.stack_methods._stack_winsorized_sigma_iter``): it returns
pure execution geometry, never touches devices, and structurally cannot
change the scientific ``N``.

Decision vocabulary
-------------------
* policy mode: ``AUTO`` / ``OVERRIDE`` (the caller's ceiling semantics);
* execution strategy: ``FULL_CPU`` / ``SPATIAL_TILED_CPU`` /
  ``CPU_MEMORY_REFUSAL``.

Design rules
------------
* PURE PLANNER: no device access, no psutil import, no CUDA/GPU/model-name
  logic, no runtime memory query.  All memory state is injected as plain
  integers (``available_ram_bytes``, ``policy_ceiling_bytes``,
  ``reserve_bytes``).
* FROZEN N: the planner takes ``n`` and never reduces or splits it.  The
  decision carries the input ``n`` verbatim and exposes **no** field that
  could encode a reduced stack population.  Tiles are spatial only.
* CONSERVATIVE model (measured stage-B evidence, see
  ``.a2a-reports/zsss-840-prew80-20260907/stage-b-evidence/cpu_winsor_rss_evidence.md``):
  incremental working set above already-resident input =
  ``FACTOR x input_cube_bytes + SCRATCH``, with envelopes chosen so that the
  model is >= the observed process-peak delta on every measured row:
    - NumPy slow path (winsor rank > 0):  13.0 x + 16 MiB
    - NumPy fast path (winsor rank == 0): 12.0 x + 16 MiB
      (the canonical NumPy helper still allocates sort_key / int64 argsort
      order / inverse-rank argsort / sorted values / bool+float workspaces in
      the rank-0 regime — measured rank-0 rows sit in the same ratio band, so
      the fast envelope is a conservative floor above every measured rank-0
      row, not an assumed shortcut that the helper does not implement.)
    - SciPy opt-in backend:                 17.5 x + 32 MiB
  ``apply_rewinsor`` and weighted/unweighted made no measurable difference
  (identical deltas within measurement noise on same-cube rows), so the
  envelopes above cover both flag values.
* Already-resident input observations are counted ONCE: they are part of the
  process baseline and of runtime free RAM at decision time.  The planner
  demand models only the *incremental* working set and never subtracts the
  resident input twice from available RAM.
* Reserve is named, documented and tested: the caller may inject any
  ``reserve_bytes``; :func:`recommended_reserve_bytes` provides the default
  policy (fixed minimum + proportional component) and accounts for process
  pool duplication/import overhead and the future output/serialization copy
  as explicit named quantities.

Geometry rules
--------------
* Full-frame first: if the full-frame incremental model fits the effective
  budget, the decision is ``FULL_CPU``.
* Otherwise prefer full-width horizontal bands (``(tile_h,)``); rectangular
  ``(tile_h, tile_w)`` geometry is only chosen when even one full-width row
  cannot fit.
* Every tile must keep at least ``CPU_MIN_TILE_OUT`` spatial outputs (the
  documented lower bound; mirrors the GPU planner's bitwise-regime floor so a
  tile is never degenerate and per-tile overhead stays amortized).  If the
  smallest admissible tile still exceeds the effective budget ->
  ``CPU_MEMORY_REFUSAL`` (never reduce N, never an empty success).
* Retry vocabulary: a ``SPATIAL_TILED_CPU`` decision may retry with a
  *smaller spatial tile* (``tile_h`` / ``tile_w`` only).  No decision field
  exposes N / reducer / kappa / winsor / normalization / weights as retry
  knobs — the decision dataclass simply has no such fields.

Runtime re-evaluation contract (concept only in stage B)
--------------------------------------------------------
``effective_budget_bytes = min(policy_ceiling_bytes,
available_ram_bytes - reserve_bytes)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

MODE_AUTO = "AUTO"
MODE_OVERRIDE = "OVERRIDE"

FULL_CPU = "FULL_CPU"
SPATIAL_TILED_CPU = "SPATIAL_TILED_CPU"
CPU_MEMORY_REFUSAL = "CPU_MEMORY_REFUSAL"

# Stable refusal reasons.
REASON_BUDGET_NEGATIVE = "cpu_budget_negative"
REASON_NO_VALID_TILE = "cpu_no_valid_tile"
REASON_MIN_TILE_EXCEEDS_BUDGET = "cpu_min_tile_exceeds_budget"
REASON_INVALID_INPUT = "cpu_invalid_input"

# ---------------------------------------------------------------------------
# Measured memory-model constants (stage B evidence; conservative direction)
# ---------------------------------------------------------------------------

# Incremental working-set factor over the N x spatial-tile float32 stack.
CPU_WINSOR_SLOW_FACTOR = 13.0     # numpy, winsor rank > 0
CPU_WINSOR_FAST_FACTOR = 12.0     # numpy, winsor rank == 0
CPU_SCIPY_WINSOR_FACTOR = 17.5    # scipy opt-in backend

CPU_WINSOR_SLOW_SCRATCH_BYTES = 16 * 1024 * 1024
CPU_WINSOR_FAST_SCRATCH_BYTES = 16 * 1024 * 1024
CPU_SCIPY_WINSOR_SCRATCH_BYTES = 32 * 1024 * 1024

# Spatial-output lower bound per tile (documented minimum viable tile).
CPU_MIN_TILE_OUT = 96

# Explicit named policy quantities (never silently ignored).
# Measured isolated-process baseline RSS (interpreter + seestar/astropy/numpy
# imports) was 336-420 MiB per worker; 448 MiB per extra worker is the rounded
# conservative policy constant applied when a process pool is actually used.
CPU_POOL_WORKER_OVERHEAD_BYTES = 448 * 1024 * 1024
# Future output/serialization copy: result + W float32 frames.
CPU_OUTPUT_SERIALIZATION_FRAMES = 2

# Reserve policy (named, documented, tested).
CPU_RESERVE_FIXED_MIN_BYTES = 256 * 1024 * 1024
CPU_RESERVE_FRACTION = 0.02


def winsor_zero_rank_regime(limits: Sequence[float], n: int) -> bool:
    """Pure rank-regime detector: ``floor(limit * n) == 0`` on both sides.

    Mathematically equivalent to the canonical behavior used by the reducer:
    per-pixel winsor rank is ``floor(limit * n_valid)`` with
    ``n_valid <= n``, so the population bound ``n`` proves the zero-rank
    regime globally.  Negative / non-finite limits never qualify (mirrors the
    canonical helper, which treats a negative limit as *no winsorization on
    that side*).  Weighted/unweighted does not change the regime: weights do
    not alter the valid-sample count.
    """
    low, high = float(limits[0]), float(limits[1])
    if not (low >= 0.0 and high >= 0.0):
        return False
    if not (math.isfinite(low) and math.isfinite(high)):
        return False
    n_int = int(n)
    return math.floor(low * n_int) == 0 and math.floor(high * n_int) == 0


def recommended_reserve_bytes(
    available_ram_bytes: int,
    frame_bytes: int,
    pool_workers: int = 1,
) -> int:
    """Named, documented reserve policy (never negative).

    ``max(CPU_RESERVE_FIXED_MIN_BYTES, CPU_RESERVE_FRACTION x available_ram)``
    plus, when a process pool is actually used (``pool_workers > 1``), the
    explicit per-extra-worker import/duplication overhead
    (``CPU_POOL_WORKER_OVERHEAD_BYTES`` per additional worker) plus the future
    output/serialization copy (``CPU_OUTPUT_SERIALIZATION_FRAMES x
    frame_bytes``).  The fixed minimum + proportional component protects the
    process' other live allocations (headers, weights, quality arrays, OS
    page cache noise); the output copy and pool overhead are named policy
    quantities, never silently ignored.
    """
    available = max(0, int(available_ram_bytes))
    prop = int(CPU_RESERVE_FRACTION * available)
    reserve = max(int(CPU_RESERVE_FIXED_MIN_BYTES), prop)
    workers = max(1, int(pool_workers))
    if workers > 1:
        reserve += (workers - 1) * int(CPU_POOL_WORKER_OVERHEAD_BYTES)
    reserve += int(CPU_OUTPUT_SERIALIZATION_FRAMES) * max(0, int(frame_bytes))
    return reserve


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class CpuMemoryDecision:
    """One pure planner decision (immutable).

    Structural exact-N contract: ``n`` mirrors the planner input verbatim and
    NO field of this dataclass can encode a reduced/split stack population
    (there is no ``n_tile``/``n_per_tile`` field, and the retry vocabulary is
    spatial-only).  All byte fields are diagnostics (modeled incremental peak
    / effective budget) for logging and stage-C wiring.

    ``tile_shape``: ``None`` for FULL_CPU / CPU_MEMORY_REFUSAL;
    ``(tile_h,)`` for full-width horizontal bands; ``(tile_h, tile_w)`` for
    rectangular tiles (only used when a full-width band cannot fit).
    """

    mode: str
    strategy: str
    reason: Optional[str]
    n: int
    frame_shape: Tuple[int, int]
    channels: int
    fast_path: bool
    backend: str
    tile_shape: Optional[Tuple[int, ...]]
    n_tiles: int
    tile_outputs: int
    minimum_tile: Tuple[int, int]
    estimated_peak_bytes: int
    per_tile_peak_bytes: int
    effective_budget_bytes: int
    reserve_bytes: int
    available_ram_bytes: int
    policy_ceiling_bytes: int
    retry_tile_shape_allowed: bool
    details: dict

    @property
    def is_refusal(self) -> bool:
        return self.strategy == CPU_MEMORY_REFUSAL


def _refusal(
    reason,
    *,
    n,
    frame_shape,
    channels,
    fast_path,
    backend,
    estimated_full,
    reserve,
    effective_budget,
    available_ram,
    ceiling,
    details,
):
    return CpuMemoryDecision(
        mode="",
        strategy=CPU_MEMORY_REFUSAL,
        reason=reason,
        n=n,
        frame_shape=tuple(frame_shape),
        channels=channels,
        fast_path=fast_path,
        backend=backend,
        tile_shape=None,
        n_tiles=0,
        tile_outputs=0,
        minimum_tile=(0, 0),
        estimated_peak_bytes=estimated_full,
        per_tile_peak_bytes=0,
        effective_budget_bytes=effective_budget,
        reserve_bytes=reserve,
        available_ram_bytes=available_ram,
        policy_ceiling_bytes=ceiling,
        retry_tile_shape_allowed=False,
        details=details,
    )


def plan_cpu_winsor_execution(
    *,
    n: int,
    frame_shape: Sequence[int],
    channels: int = 1,
    dtype_itemsize: int = 4,
    winsor_limits: Tuple[float, float] = (0.05, 0.05),
    apply_rewinsor: bool = True,
    weighted: bool = False,
    scipy_backend: bool = False,
    available_ram_bytes: int,
    reserve_bytes: Optional[int] = None,
    policy_ceiling_bytes: Optional[int] = None,
    mode: str = MODE_AUTO,
    pool_workers: int = 1,
    min_tile_out: int = CPU_MIN_TILE_OUT,
) -> CpuMemoryDecision:
    """Plan the CPU execution strategy for one frozen-N Winsorized reduction.

    Parameters
    ----------
    n : int
        Scientific stack population of THIS reduction (never modified).
    frame_shape : (H, W)
        Spatial shape of one prepared observation.
    channels : int
        Channel count (1 mono, 3 RGB).
    dtype_itemsize : int
        Itemsize in bytes of the float32 stack (4).
    winsor_limits : (low, high)
        Winsor limits; selects the fast (rank-0) vs slow memory envelope via
        :func:`winsor_zero_rank_regime` (the same pure predicate family the
        canonical reducer uses).
    apply_rewinsor, weighted : bool
        Accepted for vocabulary completeness; measured evidence shows no
        material envelope difference for either flag, so they do not change
        the chosen constants (documented in the evidence markdown).
    scipy_backend : bool
        When True the SciPy opt-in envelope is used (measured separately).
    available_ram_bytes : int
        Runtime free/available RAM injected by the caller (plain integer).
    reserve_bytes : int | None
        Safety reserve; ``None`` -> :func:`recommended_reserve_bytes`.
    policy_ceiling_bytes : int | None
        Policy ceiling (AUTO ceiling or OVERRIDE requested budget); ``None``
        -> ``available_ram_bytes``.
    mode : str
        ``AUTO`` or ``OVERRIDE`` (caller semantics; only recorded here).
    pool_workers : int
        Number of pool workers (1 = in-process).  >1 applies the documented
        per-extra-worker import/duplication overhead to the reserve.
    min_tile_out : int
        Minimum viable spatial outputs per tile.

    Returns a frozen :class:`CpuMemoryDecision`.  Raises ``ValueError`` only
    for caller bugs (invalid N / frame / itemsize / negative memory state).
    """
    if int(n) <= 0:
        raise ValueError("plan_cpu_winsor_execution requires n >= 1")
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must be (H, W)")
    H, W = int(frame_shape[0]), int(frame_shape[1])
    if H <= 0 or W <= 0:
        raise ValueError("frame_shape must be (H, W) with H, W >= 1")
    if int(channels) <= 0 or int(dtype_itemsize) <= 0:
        raise ValueError("channels and dtype_itemsize must be positive")
    if int(min_tile_out) <= 0:
        raise ValueError("min_tile_out must be positive")
    if int(available_ram_bytes) < 0:
        raise ValueError("available_ram_bytes must be non-negative")

    n = int(n)
    C = int(channels)
    isz = int(dtype_itemsize)
    available = int(available_ram_bytes)
    ceiling = (
        int(policy_ceiling_bytes)
        if policy_ceiling_bytes is not None
        else available
    )
    reserve = (
        int(reserve_bytes)
        if reserve_bytes is not None
        else recommended_reserve_bytes(available, H * W * C * isz, pool_workers)
    )
    reserve = max(0, reserve)

    fast = winsor_zero_rank_regime(winsor_limits, n)
    if scipy_backend:
        factor = CPU_SCIPY_WINSOR_FACTOR
        scratch = CPU_SCIPY_WINSOR_SCRATCH_BYTES
    elif fast:
        factor = CPU_WINSOR_FAST_FACTOR
        scratch = CPU_WINSOR_FAST_SCRATCH_BYTES
    else:
        factor = CPU_WINSOR_SLOW_FACTOR
        scratch = CPU_WINSOR_SLOW_SCRATCH_BYTES

    s_full = H * W
    base_full = n * s_full * C * isz
    # Incremental working-set model above already-resident input (counted once;
    # never subtracted twice from available RAM).
    estimated_full = int(base_full * factor) + scratch
    effective_budget = min(ceiling, available - reserve)

    details = {
        "factor": factor,
        "scratch": scratch,
        "mode": mode,
        "pool_workers": int(pool_workers),
        "resident_input_bytes": base_full,
        "input_cube_bytes": base_full,
    }

    def _full():
        return CpuMemoryDecision(
            mode=mode,
            strategy=FULL_CPU,
            reason=None,
            n=n,
            frame_shape=(H, W),
            channels=C,
            fast_path=fast,
            backend="scipy" if scipy_backend else "numpy",
            tile_shape=None,
            n_tiles=1,
            tile_outputs=s_full,
            minimum_tile=(min(1, H), min(W, max(1, min_tile_out))),
            estimated_peak_bytes=estimated_full,
            per_tile_peak_bytes=estimated_full,
            effective_budget_bytes=effective_budget,
            reserve_bytes=reserve,
            available_ram_bytes=available,
            policy_ceiling_bytes=ceiling,
            retry_tile_shape_allowed=False,
            details=details,
        )

    if effective_budget <= 0:
        return _refusal(
            REASON_BUDGET_NEGATIVE,
            n=n, frame_shape=(H, W), channels=C, fast_path=fast,
            backend="scipy" if scipy_backend else "numpy",
            estimated_full=estimated_full, reserve=reserve,
            effective_budget=effective_budget,
            available_ram=available, ceiling=ceiling, details=details,
        )

    # 1) Whole-frame decision.
    if estimated_full <= effective_budget:
        return _full()

    # 2) Spatial tiling: only the tile's incremental working set must fit
    #    (the resident full input is already counted in the process baseline).
    #    Largest per-tile spatial output count whose modeled demand fits.
    s_cap = int((effective_budget - scratch) / (factor * n * C * isz))
    if s_cap < min_tile_out:
        return _refusal(
            REASON_MIN_TILE_EXCEEDS_BUDGET,
            n=n, frame_shape=(H, W), channels=C, fast_path=fast,
            backend="scipy" if scipy_backend else "numpy",
            estimated_full=estimated_full, reserve=reserve,
            effective_budget=effective_budget,
            available_ram=available, ceiling=ceiling, details=details,
        )
    s_cap = min(s_cap, s_full)

    def _tile_peak(tile_h, tile_w):
        return int(n * tile_h * tile_w * C * isz * factor) + scratch

    # 2a) Prefer full-width horizontal bands: one row of W outputs minimum.
    if W >= min_tile_out:
        band_h = max(1, s_cap // W)  # largest full-width band that fits
        if band_h >= 1:
            tile_h = min(H, band_h)
            per_tile = _tile_peak(tile_h, W)
            if per_tile <= effective_budget:
                n_tiles = _ceil_div(H, tile_h)
                return CpuMemoryDecision(
                    mode=mode,
                    strategy=SPATIAL_TILED_CPU,
                    reason=None,
                    n=n,
                    frame_shape=(H, W),
                    channels=C,
                    fast_path=fast,
                    backend="scipy" if scipy_backend else "numpy",
                    tile_shape=(tile_h,),
                    n_tiles=n_tiles,
                    tile_outputs=tile_h * W,
                    minimum_tile=(1, W) if W >= min_tile_out else (1, W),
                    estimated_peak_bytes=estimated_full,
                    per_tile_peak_bytes=per_tile,
                    effective_budget_bytes=effective_budget,
                    reserve_bytes=reserve,
                    available_ram_bytes=available,
                    policy_ceiling_bytes=ceiling,
                    retry_tile_shape_allowed=True,
                    details=dict(
                        details,
                        s_cap=s_cap,
                        per_tile_base_bytes=n * tile_h * W * C * isz,
                        tile_w=W,
                    ),
                )

    # 2b) Rectangular fallback: full-width band cannot fit (s_cap < W).
    #     Scan column counts; each tile >= min_tile_out outputs.
    max_c = W // min_tile_out
    for c in range(2, max_c + 2):
        tile_w = min(s_cap, (W - min_tile_out) // (c - 1))
        if tile_w < _ceil_div(W, c):
            continue
        tile_w = min(tile_w, W)
        tile_h = min(H, max(1, s_cap // tile_w))
        per_tile = _tile_peak(tile_h, tile_w)
        if per_tile <= effective_budget and tile_w * tile_h >= min_tile_out:
            n_tiles = _ceil_div(H, tile_h) * _ceil_div(W, tile_w)
            return CpuMemoryDecision(
                mode=mode,
                strategy=SPATIAL_TILED_CPU,
                reason=None,
                n=n,
                frame_shape=(H, W),
                channels=C,
                fast_path=fast,
                backend="scipy" if scipy_backend else "numpy",
                tile_shape=(tile_h, tile_w),
                n_tiles=n_tiles,
                tile_outputs=tile_h * tile_w,
                minimum_tile=(1, min_tile_out),
                estimated_peak_bytes=estimated_full,
                per_tile_peak_bytes=per_tile,
                effective_budget_bytes=effective_budget,
                reserve_bytes=reserve,
                available_ram_bytes=available,
                policy_ceiling_bytes=ceiling,
                retry_tile_shape_allowed=True,
                details=dict(
                    details,
                    s_cap=s_cap,
                    per_tile_base_bytes=n * tile_h * tile_w * C * isz,
                    full_width_band_impossible=True,
                ),
            )

    return _refusal(
        REASON_NO_VALID_TILE,
        n=n, frame_shape=(H, W), channels=C, fast_path=fast,
        backend="scipy" if scipy_backend else "numpy",
        estimated_full=estimated_full, reserve=reserve,
        effective_budget=effective_budget,
        available_ram=available, ceiling=ceiling, details=details,
    )
