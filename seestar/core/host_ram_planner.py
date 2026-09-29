"""Pure conjoint host-RAM planner for the Winsorized reduction (Lot D, low-RAM).

Mission ``ZSSS-MEMORY-GPU-20260929``.  This module mirrors the shape of the
GPU VRAM planner and the CPU memory planner, but decides the HOST strategy for
one frozen-N Winsorized reduction: whether the already-aligned/normalised
inputs and the full-frame SCI/WHT outputs can stay in RAM, or must be spilled
to run-scoped scratch disk / memmapped, and — crucially — the **host tile cap**
(a bound on per-tile spatial outputs) that a tiled geometry must respect so
its host working set fits alongside everything else.

The GPU VRAM planner already bounds a tile by device memory; a tile is only
admissible when it fits **both** sides.  This module produces the host-side
cap, which the wiring combines with the VRAM cap (the GPU planner receives it
as ``host_tile_outputs_cap``; the CPU planner receives it as ``max_tile_out``
via ``min_tile_out`` / a budget re-derivation).

Design rules
------------
* PURE PLANNER: no device access, no psutil import, no disk access, no GPU
  model-name logic.  All memory state is injected as plain integers.
* FROZEN N: the planner takes ``n`` and never reduces or splits it; the tile
  cap is spatial only.
* NO DOUBLE COUNTING: ``resident_input_bytes`` are the frames+masks ALREADY
  resident at decision time (part of the process baseline).  They are counted
  ONCE.  The incremental host working set is ``tile_cube + output + overhead``;
  the resident inputs are never subtracted twice from available RAM.
* ``spill_inputs`` frees the resident inputs (they move to scratch disk and are
  re-opened as memmap), so the effective budget for NEW allocations grows by
  ``resident_input_bytes``.  ``memmap_outputs`` moves the SCI/WHT outputs to
  disk, removing them from the RAM working set (the working set then only
  needs the per-tile cube + overhead).  ``spill_and_memmap`` does both.
* Named reserve and overhead are explicit policy quantities, never an
  unexplained magic number.

Decision vocabulary
-------------------
``in_memory | spill_inputs | memmap_outputs | spill_and_memmap | refuse``.

Strategy selection is a fixed, documented priority: the FIRST strategy that
admits a valid tile (tile cap >= ``min_tile_out``) wins.  A strategy only
"admits" a tile when the tile's host working set fits the strategy's budget.

* ``in_memory``          — inputs resident, outputs in RAM.
* ``spill_inputs``       — inputs spilled (freed), outputs in RAM.
* ``memmap_outputs``     — inputs resident, outputs on disk.
* ``spill_and_memmap``   — inputs spilled, outputs on disk.
* ``refuse``             — even the minimum tile does not fit under the least
                           demanding strategy (never reduce N, never an empty
                           success).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

HOST_IN_MEMORY = "in_memory"
HOST_SPILL_INPUTS = "spill_inputs"
HOST_MEMMAP_OUTPUTS = "memmap_outputs"
HOST_SPILL_AND_MEMMAP = "spill_and_memmap"
HOST_REFUSE = "refuse"

# Stable refusal reasons.
REASON_HOST_BUDGET_NEGATIVE = "host_budget_negative"
REASON_HOST_NO_VALID_TILE = "host_no_valid_tile"

# Named per-tile overhead (reducer coordination temporaries beyond the cube
# itself — mask planes, per-column index arrays, schedule scratch).  The
# caller may inject a backend-specific value; for the GPU host path the only
# host materialisation is the tile cube itself (the device temporaries live on
# VRAM), so the default is zero.
HOST_TILE_OVERHEAD_BYTES = 0


@dataclass(frozen=True)
class HostRamDecision:
    """One frozen host-RAM decision (immutable).

    ``strategy`` is one of the vocabulary above.  ``host_tile_outputs_cap`` is
    the INCLUSIVE upper bound on per-tile spatial outputs (``None`` means the
    whole frame fits under the chosen strategy — i.e. the cap is the full
    frame and does not constrain tiling).  ``spill_input_bytes`` /
    ``output_memmap_bytes`` are the named disk costs (0 when not required).
    All byte fields are diagnostics for the wiring and provenance.
    """

    strategy: str
    reason: Optional[str]
    n: int
    frame_shape: Tuple[int, int]
    channels: int
    available_ram_bytes: int
    reserve_bytes: int
    resident_input_bytes: int
    output_bytes: int
    overhead_bytes: int
    tile_working_factor: float
    tile_scratch_bytes: int
    effective_budget_bytes: int
    host_tile_outputs_cap: Optional[int]
    spill_input_bytes: int
    output_memmap_bytes: int
    details: dict = field(default_factory=dict)

    @property
    def is_refusal(self) -> bool:
        return self.strategy == HOST_REFUSE

    @property
    def spills_inputs(self) -> bool:
        return self.strategy in (HOST_SPILL_INPUTS, HOST_SPILL_AND_MEMMAP)

    @property
    def memmaps_outputs(self) -> bool:
        return self.strategy in (HOST_MEMMAP_OUTPUTS, HOST_SPILL_AND_MEMMAP)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def winsorized_tile_from_cap(frame_shape, cap, min_tile_out):
    """Derive a spatial tile geometry from a per-tile spatial-output cap.

    Prefers a full-width horizontal band ``(tile_h,)`` whose full cells keep
    ``min_tile_out`` outputs; falls back to a rectangular ``(tile_h, tile_w)``
    when the frame is narrower than ``min_tile_out``.  Returns ``None`` when no
    valid geometry exists under the cap (never a reduced N, never an empty
    success).  Pure and deterministic.
    """
    H, W = int(frame_shape[0]), int(frame_shape[1])
    cap = int(cap)
    min_tile_out = int(min_tile_out)
    if cap < min_tile_out or cap <= 0:
        return None
    if W >= min_tile_out:
        tile_h = max(1, min(H, cap // W))
        if tile_h * W >= min_tile_out:
            return (tile_h,)
        return None
    # Narrow frame (W < min_tile_out): a band of tile_h rows has tile_h * W
    # outputs, so the full-cell tile must be tall enough; a partial last band
    # of r rows is valid only when r == 0 or r * W >= min_tile_out.
    t_min = _ceil_div(min_tile_out, W)
    tile_h = max(t_min, min(H, cap // W))
    if tile_h * W < min_tile_out:
        return None
    return (tile_h,)


def _tile_cube_bytes(n, s, channels, itemsize):
    return int(n) * int(s) * int(channels) * int(itemsize)


def _max_tile_outputs_for_budget(
    n, channels, itemsize, factor, scratch, budget, min_tile_out, s_full
):
    """Largest per-tile spatial output count whose working set fits ``budget``,
    or ``None`` when even ``min_tile_out`` does not fit."""
    if budget <= 0:
        return None
    # budget >= tile_cube(s) * factor + scratch
    avail_for_cube = budget - scratch
    if avail_for_cube <= 0:
        return None
    s_cap = int(avail_for_cube / (factor * n * channels * itemsize))
    if s_cap < int(min_tile_out):
        return None
    return min(s_cap, int(s_full))


def plan_host_ram_execution(
    *,
    n: int,
    frame_shape: Sequence[int],
    channels: int = 1,
    dtype_itemsize: int = 4,
    available_ram_bytes: int,
    reserve_bytes: int,
    resident_input_bytes: Optional[int] = None,
    output_bytes: Optional[int] = None,
    overhead_bytes: int = HOST_TILE_OVERHEAD_BYTES,
    tile_working_factor: float = 1.0,
    tile_scratch_bytes: int = 0,
    min_tile_out: int = 96,
) -> HostRamDecision:
    """Plan the host-RAM strategy + host tile cap for one frozen-N reduction.

    Parameters
    ----------
    n, frame_shape, channels, dtype_itemsize : as in the other planners.
    available_ram_bytes : live RAM available NOW (plain integer, injected).
    reserve_bytes : named safety reserve (already computed by the caller).
    resident_input_bytes : frames+masks ALREADY resident (counted once).
        ``None`` -> ``n * H * W * C * itemsize + n * H * W`` (frames + masks).
    output_bytes : SCI+WHT full-frame float32 outputs.  ``None`` ->
        ``2 * H * W * C * itemsize``.
    overhead_bytes : named per-tile overhead (reducer temporaries beyond the
        cube).  Conservative floor; the caller may pass a backend value.
    tile_working_factor : host working-set factor over the tile cube.  The GPU
        host path only materialises the cube (``1.0``); the CPU numpy path's
        working set is larger (pass the CPU planner factor).  A tile's host
        working set = ``factor * tile_cube + tile_scratch_bytes``.
    tile_scratch_bytes : additive per-tile working-set scratch.
    min_tile_out : minimum viable spatial outputs per tile.

    Returns a frozen :class:`HostRamDecision`.  Raises ``ValueError`` only for
    caller bugs (invalid N / frame / itemsize / negative memory state).
    """
    if int(n) <= 0:
        raise ValueError("plan_host_ram_execution requires n >= 1")
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must be (H, W)")
    H, W = int(frame_shape[0]), int(frame_shape[1])
    if H <= 0 or W <= 0:
        raise ValueError("frame_shape must be (H, W) with H, W >= 1")
    if int(channels) <= 0 or int(dtype_itemsize) <= 0:
        raise ValueError("channels and dtype_itemsize must be positive")
    if int(min_tile_out) <= 0:
        raise ValueError("min_tile_out must be positive")
    if int(available_ram_bytes) < 0 or int(reserve_bytes) < 0:
        raise ValueError("memory-state bytes must be non-negative")

    n = int(n)
    C = int(channels)
    isz = int(dtype_itemsize)
    s_full = H * W
    available = int(available_ram_bytes)
    reserve = int(reserve_bytes)

    resident = (
        int(resident_input_bytes)
        if resident_input_bytes is not None
        else n * s_full * C * isz + n * s_full
    )
    output = (
        int(output_bytes)
        if output_bytes is not None
        else 2 * s_full * C * isz
    )
    overhead = int(overhead_bytes)
    factor = float(tile_working_factor)
    scratch = int(tile_scratch_bytes)

    # Effective budget for NEW allocations (the resident inputs are baseline,
    # already reflected in ``available``; they are never subtracted again).
    # ``spill_inputs`` frees them, growing the budget by ``resident``.
    budget_in_memory = available - reserve
    budget_spilled = available - reserve + resident

    details = {
        "resident_input_bytes": resident,
        "output_bytes": output,
        "overhead_bytes": overhead,
        "tile_working_factor": factor,
        "tile_scratch_bytes": scratch,
        "budget_in_memory": budget_in_memory,
        "budget_spilled": budget_spilled,
    }

    def _decide():
        # Priority order (ascending disk cost, documented): in_memory ->
        # memmap_outputs -> spill_inputs -> spill_and_memmap -> refuse.
        # ``memmap_outputs`` precedes ``spill_inputs`` because for the
        # winsorized stack N >= 2 the outputs (2 frames) are no larger than
        # the inputs (N frames), so memmapping outputs is the cheaper disk
        # trade.  Each strategy admits a valid tile when its working set
        # (cube + output-in-RAM + overhead) fits its budget.
        # 1) in_memory: inputs resident, outputs in RAM.
        if budget_in_memory <= 0:
            cap_in = None
        else:
            cap_in = _max_tile_outputs_for_budget(
                n, C, isz, factor, scratch,
                budget_in_memory - output - overhead,
                min_tile_out, s_full,
            )
        if cap_in is not None:
            return _mk(HOST_IN_MEMORY, None, cap_in, 0, 0, budget_in_memory)

        # 2) memmap_outputs: inputs resident, outputs on disk.
        cap_mo = _max_tile_outputs_for_budget(
            n, C, isz, factor, scratch,
            budget_in_memory - overhead,
            min_tile_out, s_full,
        ) if budget_in_memory > 0 else None
        if cap_mo is not None:
            return _mk(HOST_MEMMAP_OUTPUTS, None, cap_mo, 0, output,
                       budget_in_memory)

        # 3) spill_inputs: inputs freed (budget grows by ``resident``),
        # outputs still in RAM.
        if budget_spilled <= 0:
            cap_spill = None
        else:
            cap_spill = _max_tile_outputs_for_budget(
                n, C, isz, factor, scratch,
                budget_spilled - output - overhead,
                min_tile_out, s_full,
            )
        if cap_spill is not None:
            return _mk(HOST_SPILL_INPUTS, None, cap_spill, resident, 0,
                       budget_spilled)

        # 4) spill_and_memmap: both.
        cap_sm = _max_tile_outputs_for_budget(
            n, C, isz, factor, scratch,
            budget_spilled - overhead,
            min_tile_out, s_full,
        ) if budget_spilled > 0 else None
        if cap_sm is not None:
            return _mk(HOST_SPILL_AND_MEMMAP, None, cap_sm, resident, output,
                       budget_spilled)

        # 5) refuse: nothing admits the minimum tile.
        reason = (
            REASON_HOST_BUDGET_NEGATIVE
            if budget_spilled <= 0
            else REASON_HOST_NO_VALID_TILE
        )
        return HostRamDecision(
            strategy=HOST_REFUSE,
            reason=reason,
            n=n,
            frame_shape=(H, W),
            channels=C,
            available_ram_bytes=available,
            reserve_bytes=reserve,
            resident_input_bytes=resident,
            output_bytes=output,
            overhead_bytes=overhead,
            tile_working_factor=factor,
            tile_scratch_bytes=scratch,
            effective_budget_bytes=max(budget_in_memory, budget_spilled),
            host_tile_outputs_cap=None,
            spill_input_bytes=0,
            output_memmap_bytes=0,
            details=dict(details, min_tile_out=int(min_tile_out)),
        )

    def _mk(strategy, reason, cap, spill, memmap, budget):
        return HostRamDecision(
            strategy=strategy,
            reason=reason,
            n=n,
            frame_shape=(H, W),
            channels=C,
            available_ram_bytes=available,
            reserve_bytes=reserve,
            resident_input_bytes=resident,
            output_bytes=output,
            overhead_bytes=overhead,
            tile_working_factor=factor,
            tile_scratch_bytes=scratch,
            effective_budget_bytes=budget,
            host_tile_outputs_cap=cap,
            spill_input_bytes=spill,
            output_memmap_bytes=memmap,
            details=dict(details, min_tile_out=int(min_tile_out)),
        )

    return _decide()
