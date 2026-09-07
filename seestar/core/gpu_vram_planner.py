"""Adaptive VRAM execution planner for the Winsorized GPU reduction (Track P4).

Phase F wiring target: replaces the coarse whole-stack Winsorized VRAM
yes/no guard (``_GPU_FOOTPRINT_FACTOR_WINSORIZED`` in queue_manager) with an
algorithm-aware decision producing one of

* ``FULL_GPU``          -- run the untiled GPU twin,
* ``TILED_GPU``         -- run the exact-N_batch SPATIAL GPU tiling
  ``stack_winsorized_sigma_gpu_tiled`` with a chosen ``tile_shape``,
* ``CPU_FALLBACK``      -- reason-coded CPU execution.

Design rules (mission Track P4):

* PURE PLANNER: no device access, no CuPy import, no GPU model-name tables,
  no Windows shared-memory heuristics.  All memory state is injected as
  plain integers (``driver_free_bytes`` from ``memGetInfo``, ``pool_free_bytes``
  from the CuPy default memory pool).  Unit tests need no device.
* FROZEN N_batch: the planner takes ``n_batch`` as an input and never
  reduces it -- tiles are SPATIAL only (the phase E seam never splits the
  scientific stack axis).  ``decision.n_batch`` mirrors the input; no
  field of the decision can represent a stack-axis split.
* Memory model (measured phase D / E evidence on the optimized clip/sort
  path, NOT the blind 6.0 and NOT hardware names):

  * slow path (winsor rank > 0): the per-tile peak device footprint is
    ``N_batch * tile_h * tile_w * C * itemsize`` (one tile's float32 stack)
    times a sort/live factor plus a sort-scratch / retention constant.
    Measured optimized slow path: peak LIVE == 7.26--7.79 x the stack at
    480x270 (phase D N=20/32/50) and 7.53--8.15 x at 1080p (phase E), while
    the total POOL demand (live + radix-sort scratch + per-column index
    temporaries + size-class retention) fits ``~8.9 x stack + ~151 MiB`` on
    the phase E 1080p rows (least-squares over the seven executed
    geometries: full N=20 and tiled N=20/32/50 rows, 28--158 MiB stack).
    The model uses 9.5 x + 200 MiB, which is >= every measured pool demand
    of the executed phase E rows with 50--200 MiB margin (conservative
    direction; validated on-device: the phase E N=32 1080p untiled OOM and
    the phase F boundary probe both sit just above the 9.5 x envelope,
    while every planner-selected geometry executed).
  * fast path (phase C zero-rank regime, winsor identity: no per-iteration
    sort temporaries): factor 3.0 x + 64 MiB (conservative; measured
    phase C/E fast-path runs fit far below the slow path).
* Budget: ``driver_free + pool_free``; a decision (full or per-tile) is
  admissible iff its modeled demand + the EXPLICIT RESERVE fits the budget.
  Query failures never reach this module: the wiring converts them into
  ``CPU_FALLBACK`` decisions with the stable reasons below (never a crash).
* Bitwise-regime constraint: every spatial tile of a chosen geometry must
  keep at least ``min_tile_out`` (96) spatial outputs -- above the measured
  CuPy micro-reduction band (mono S <= 26, RGB S <= 30) so no tile silently
  drops the phase E bitwise guarantee.  If no scientifically valid geometry
  fits -> ``CPU_FALLBACK(vram_no_valid_tile)``.
* No AutoBatch participation: this module only picks a GPU execution
  strategy for a frozen ``n_batch``; it never resolves batch sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from seestar.core.stack_gpu import _winsor_zero_rank_regime

# ---------------------------------------------------------------------------
# Decision vocabulary
# ---------------------------------------------------------------------------

FULL_GPU = "FULL_GPU"
TILED_GPU = "TILED_GPU"
CPU_FALLBACK = "CPU_FALLBACK"

# Stable CPU fallback reasons (wiring + diagnostics contract).
REASON_VRAM_NO_VALID_TILE = "vram_no_valid_tile"
REASON_PLANNER_FAILURE = "planner_failure"
REASON_POOL_QUERY_FAILURE = "pool_query_failure"
REASON_MEMINFO_FAILURE = "meminfo_failure"

# ---------------------------------------------------------------------------
# Memory model constants (measured phase D/E, see module docstring)
# ---------------------------------------------------------------------------

# Spatial outputs per tile floor: every tile of a TILED geometry must have
# >= this many outputs so the reduction stays out of the CuPy micro band
# (mono S <= 26 / RGB S <= 30 measured; 96 is the tested fully-bitwise
# rectangular floor of the phase E suite).
WINSOR_MIN_TILE_OUT = 96

# Optimized clip/sort SLOW path (winsor rank > 0): peak footprint factor
# over the N x tile float32 stack.  Phase D measured live 7.26--7.79x at
# 480x270; phase E measured 7.53--8.15x live but ~8.9x + ~151 MiB total
# POOL demand at 1080p (least-squares over the executed rows); 9.5x is
# conservative against every measured row.
WINSOR_SLOW_SORT_FACTOR = 9.5

# Additive term of the slow path: cupy radix-sort scratch + per-column
# index temporaries + pool size-class retention (phase D/E pool-total minus
# ~8.9 x stack fitted ~150--200 MiB at 1080p; 200 MiB is conservative).
WINSOR_SLOW_SCRATCH_BYTES = 200 * 1024 * 1024

# Phase C zero-rank fast path (winsorization is the identity: the
# per-iteration sort block and the survivor-bound sort are skipped):
# conservative working factor over the N x tile stack.
WINSOR_FAST_SORT_FACTOR = 3.0

WINSOR_FAST_SCRATCH_BYTES = 64 * 1024 * 1024

# Explicit reserve applied by the wiring on top of the modeled demand
# (other-device users / context overhead / model uncertainty).
WINSOR_PLANNER_DEFAULT_RESERVE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class WinsorExecDecision:
    """One planner decision (immutable).

    ``kind`` is FULL_GPU / TILED_GPU / CPU_FALLBACK.  For TILED_GPU,
    ``tile_shape`` is the phase E seam input: ``int``/``(tile_h,)`` full-width
    row bands or ``(tile_h, tile_w)`` rectangular tiles.  For CPU_FALLBACK,
    ``reason`` is a stable code from the catalog above.  All byte fields are
    diagnostics (modeled demand / effective budget) for logging.
    """

    kind: str
    reason: Optional[str] = None
    tile_shape: Optional[Tuple[int, ...]] = None
    n_batch: int = 0
    frame_shape: Tuple[int, int] = (0, 0)
    channels: int = 1
    fast_path: bool = False
    demand_full_bytes: int = 0
    demand_tile_bytes: int = 0
    reserve_bytes: int = 0
    effective_budget_bytes: int = 0
    tile_outputs: int = 0
    n_tiles: int = 0
    details: dict = field(default_factory=dict)

    @property
    def is_gpu(self) -> bool:
        return self.kind in (FULL_GPU, TILED_GPU)


def _cpu_fallback(reason, n_batch, frame_shape, channels, fast_path,
                  demand_full, reserve, effective_budget):
    return WinsorExecDecision(
        kind=CPU_FALLBACK,
        reason=reason,
        n_batch=n_batch,
        frame_shape=tuple(frame_shape),
        channels=channels,
        fast_path=fast_path,
        demand_full_bytes=demand_full,
        demand_tile_bytes=0,
        reserve_bytes=reserve,
        effective_budget_bytes=effective_budget,
    )


def _ceil_div(a, b):
    return -(-a // b)


def _choose_rect_geometry(H, W, s_cap, min_out):
    """Largest rectangular geometry (tile_h, tile_w) with every tile
    >= ``min_out`` spatial outputs and every full cell <= ``s_cap`` outputs.

    Only called when a single full-width row does NOT fit (``s_cap < W``)
    and ``W >= min_out``.  Scans the column count c = ceil(W / tile_w); for
    each c the widest admissible ``tile_w`` is
    ``min(s_cap, (W - min_out) // (c - 1))`` provided it still leaves the
    last (partial) column at least ``min_out`` wide and c columns cover W.
    Returns ``(tile_h, tile_w)`` or None.  Row-wise partial bands are always
    valid here because ``tile_w >= min_out`` (a partial band is at least one
    row tall, i.e. >= tile_w outputs).
    """
    # c columns need tile_w in [ceil(W/c), min(s_cap, (W - min_out)//(c-1))]
    max_c = W // min_out  # tile_w >= min_out caps the column count
    for c in range(2, max_c + 2):
        upper = min(s_cap, (W - min_out) // (c - 1))
        lower = max(min_out, _ceil_div(W, c))
        if upper < lower:
            continue
        tile_w = upper
        tile_h = min(H, max(1, s_cap // tile_w))
        return (tile_h, tile_w)
    return None


def _choose_row_band(H, W, s_cap, min_out):
    """Largest full-width row-band height whose full cells fit ``s_cap``.

    ``W >= min_out`` case: every band (including a partial last band) has at
    least ``W >= min_out`` outputs, so any height is valid.  Returns the
    height or None when not even one full row fits (``s_cap < W``).
    """
    if s_cap < W:
        return None
    tile_h = min(H, max(1, s_cap // W))
    return tile_h


def _choose_narrow_band(H, W, s_cap, min_out):
    """Row-band search for narrow frames (``W < min_out``).

    A band of ``tile_h`` rows has ``tile_h * W`` outputs; full cells need
    ``tile_h >= ceil(min_out / W)`` and a partial last band of ``r`` rows is
    valid only when ``r == 0`` or ``r * W >= min_out``.  Returns the largest
    admissible height or None.
    """
    t_min = _ceil_div(min_out, W)
    if s_cap < t_min * W:
        # Even the smallest admissible full cell does not fit.
        return None
    tile_h_max = min(H, s_cap // W)
    for tile_h in range(tile_h_max, t_min - 1, -1):
        r = H % tile_h
        if r == 0 or r >= t_min:
            return tile_h
    return None


def plan_winsorized_gpu_execution(
    *,
    n_batch: int,
    frame_shape: Sequence[int],
    channels: int = 1,
    dtype_itemsize: int = 4,
    winsor_limits: Tuple[float, float] = (0.05, 0.05),
    driver_free_bytes: int,
    pool_free_bytes: int = 0,
    reserve_bytes: int = WINSOR_PLANNER_DEFAULT_RESERVE_BYTES,
    min_tile_out: int = WINSOR_MIN_TILE_OUT,
) -> WinsorExecDecision:
    """Plan the GPU execution strategy for one frozen N_batch reduction.

    Parameters
    ----------
    n_batch : int
        Frozen stack population of THIS reduction (never modified).
    frame_shape : (H, W)
        Spatial frame shape of one image.
    channels : int
        Channel count of the stack axis-0 trailing dims (1 mono, 3 RGB).
    dtype_itemsize : int
        Itemsize in bytes of the float stack (4 for float32).
    winsor_limits : (low, high)
        Winsor limits; select the phase C zero-rank fast-path memory model
        via ``_winsor_zero_rank_regime`` (same pure predicate as the kernel).
    driver_free_bytes : int
        ``cuda.runtime.memGetInfo()`` free bytes (device memory still
        allocatable).  NOT a nominal capacity: this is the *effective*
        allocatable memory, so "24 GiB nominal / 2 GiB allocatable" arrives
        here as 2 GiB -- no GPU-name rules anywhere.
    pool_free_bytes : int
        CuPy default memory-pool reusable free bytes (R2-F4 semantics).
    reserve_bytes : int
        Explicit reserve folded into the modeled demand.
    min_tile_out : int
        Bitwise-regime floor of spatial outputs per tile (default 96).

    Returns a WinsorExecDecision.  Raises ValueError only for caller bugs
    (empty batch / non-positive frame); the wiring converts any exception
    into CPU_FALLBACK(planner_failure).
    """
    if int(n_batch) <= 0:
        raise ValueError("plan_winsorized_gpu_execution requires n_batch >= 1")
    if len(frame_shape) < 2 or int(frame_shape[0]) <= 0 or int(frame_shape[1]) <= 0:
        raise ValueError("frame_shape must be (H, W) with H, W >= 1")
    if int(channels) <= 0 or int(dtype_itemsize) <= 0:
        raise ValueError("channels and dtype_itemsize must be positive")
    if int(min_tile_out) <= 0:
        raise ValueError("min_tile_out must be positive")
    if int(driver_free_bytes) < 0 or int(pool_free_bytes) < 0:
        raise ValueError("memory-state bytes must be non-negative")

    n = int(n_batch)
    H, W = int(frame_shape[0]), int(frame_shape[1])
    C = int(channels)
    isz = int(dtype_itemsize)
    s_full = H * W
    reserve = int(reserve_bytes)
    budget = int(driver_free_bytes) + int(pool_free_bytes)

    fast = bool(_winsor_zero_rank_regime(winsor_limits, n))
    if fast:
        factor = WINSOR_FAST_SORT_FACTOR
        scratch = WINSOR_FAST_SCRATCH_BYTES
    else:
        factor = WINSOR_SLOW_SORT_FACTOR
        scratch = WINSOR_SLOW_SCRATCH_BYTES

    base_full = n * s_full * C * isz
    demand_full = int(base_full * factor) + scratch
    effective_budget = budget - reserve

    def _full():
        return WinsorExecDecision(
            kind=FULL_GPU,
            n_batch=n,
            frame_shape=(H, W),
            channels=C,
            fast_path=fast,
            demand_full_bytes=demand_full,
            demand_tile_bytes=demand_full,
            reserve_bytes=reserve,
            effective_budget_bytes=effective_budget,
            tile_outputs=s_full,
            n_tiles=1,
            details={"factor": factor, "scratch": scratch},
        )

    # 1) Whole-frame (untiled) decision.
    if demand_full + reserve <= budget:
        return _full()

    # 2) Spatial tiling search: no scientifically valid tile -> CPU_FALLBACK.
    max_tile_base = budget - reserve - scratch  # bytes for N x tile stack
    if max_tile_base <= 0:
        return _cpu_fallback(
            REASON_VRAM_NO_VALID_TILE, n, (H, W), C, fast,
            demand_full, reserve, effective_budget,
        )
    # Largest per-tile spatial output count whose modeled demand fits.
    s_cap = int((max_tile_base / factor) // (n * C * isz))
    if s_cap < min_tile_out:
        return _cpu_fallback(
            REASON_VRAM_NO_VALID_TILE, n, (H, W), C, fast,
            demand_full, reserve, effective_budget,
        )
    s_cap = min(s_cap, s_full)

    if W >= min_tile_out:
        tile_h = _choose_row_band(H, W, s_cap, min_tile_out)
        if tile_h is not None:
            tile_shape = (tile_h,)
            per_tile_out = tile_h * W
        else:
            # Not even one full row fits -> rectangular split of the width.
            geom = _choose_rect_geometry(H, W, s_cap, min_tile_out)
            if geom is None:
                return _cpu_fallback(
                    REASON_VRAM_NO_VALID_TILE, n, (H, W), C, fast,
                    demand_full, reserve, effective_budget,
                )
            tile_shape = tuple(geom)
            per_tile_out = int(geom[0]) * int(geom[1])
    else:
        tile_h = _choose_narrow_band(H, W, s_cap, min_tile_out)
        if tile_h is None:
            return _cpu_fallback(
                REASON_VRAM_NO_VALID_TILE, n, (H, W), C, fast,
                demand_full, reserve, effective_budget,
            )
        tile_shape = (tile_h,)
        per_tile_out = tile_h * W

    tile_h, tile_w = (tile_shape[0], W) if len(tile_shape) == 1 else tile_shape
    per_tile_base = n * tile_h * tile_w * C * isz
    demand_tile = int(per_tile_base * factor) + scratch
    n_rows = _ceil_div(H, tile_h)
    n_cols = _ceil_div(W, tile_w)
    return WinsorExecDecision(
        kind=TILED_GPU,
        tile_shape=tile_shape,
        n_batch=n,
        frame_shape=(H, W),
        channels=C,
        fast_path=fast,
        demand_full_bytes=demand_full,
        demand_tile_bytes=demand_tile,
        reserve_bytes=reserve,
        effective_budget_bytes=effective_budget,
        tile_outputs=per_tile_out,
        n_tiles=n_rows * n_cols,
        details={
            "factor": factor,
            "scratch": scratch,
            "s_cap": s_cap,
            "per_tile_base_bytes": per_tile_base,
        },
    )
