"""Automatic CPU memory policy engine (8.4.0, stage E1).

Mission ``zsss-840-prew80-20260907``.  Stage E1 delivers the ZSSS-owned
automatic CPU memory POLICY layer (resolution + provenance record builders)
and its production wiring: normal product execution resolves the CPU memory
policy automatically from actual machine/workload state (AUTO), never from
the legacy user HQ RAM value and never from GPU model/VRAM/CUDA name; a
single resolved byte budget flows through the CPU Winsorized chain; the
exact-N spatial CPU tiled driver (stage C) and the pure planner (stage B)
are actually used.

This module is PURE, like the stage-B planner: no device access, no psutil
at import, no runtime memory query, no GPU/model-name logic.  All memory
state is injected as plain integers by the wiring layer (queue_manager).

Vocabulary
----------
* policy mode: ``AUTO`` / ``OVERRIDE`` (ceiling semantics — never mixed with
  the execution strategy);
* execution strategy: ``FULL_CPU`` / ``SPATIAL_TILED_CPU`` /
  ``CPU_MEMORY_REFUSAL`` (reused verbatim from
  :mod:`seestar.core.cpu_memory_planner`).

Policy rules
------------
* AUTO: policy ceiling resolved at run preflight from the then-available RAM
  minus the named safety reserve
  (``ceiling = max(0, available_ram_preflight - reserve_preflight)``).  The
  ceiling is a POLICY/capability statement, never a frozen promise of future
  bytes: at every CPU execution the budget is re-evaluated against the RAM
  available *at that moment*.
* OVERRIDE: an explicit expert budget (env ``ZSSS_CPU_MEMORY_OVERRIDE_BYTES``
  or the internal test seam) becomes the ceiling; provenance records
  ``mode=override`` and ``requested_budget_bytes``.
* Runtime re-evaluation at CPU execution time:
  ``effective_budget_bytes = min(policy_ceiling_bytes,
  available_ram_now_bytes - reserve_now_bytes)`` with the named, documented
  reserve from the stage-B planner (:func:`recommended_reserve_bytes`:
  fixed minimum 256 MiB + 2 % proportional + explicit pool/output
  quantities) — no unexplained magic number.
* CPU RAM is NEVER derived from GPU model / VRAM / CUDA device name.

Delegation: :func:`resolve_cpu_winsor_decision` computes the execution
decision by delegating to the stage-B pure planner
(:func:`plan_cpu_winsor_execution`) with the frozen scientific ``N`` — only
FULL_CPU / SPATIAL_TILED_CPU geometry / CPU_MEMORY_REFUSAL may differ
between simulations.

Provenance record builders (pure dicts, JSON-safe scalars only — no
image/mask arrays): one builder per required event kind
(MEMORY_POLICY / CPU_WINSOR_MEMORY_DECISION / CPU_WINSOR_MEMORY_RETRY /
CPU_WINSOR_MEMORY_REFUSAL).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from seestar.core.cpu_memory_planner import (
    CPU_MIN_TILE_OUT,
    CPU_MEMORY_REFUSAL,
    FULL_CPU,
    MODE_AUTO,
    MODE_OVERRIDE,
    REASON_BUDGET_NEGATIVE,
    REASON_MIN_TILE_EXCEEDS_BUDGET,
    REASON_NO_VALID_TILE,
    SPATIAL_TILED_CPU,
    CpuMemoryDecision,
    plan_cpu_winsor_execution,
    recommended_reserve_bytes,
)

__all__ = [
    "MODE_AUTO",
    "MODE_OVERRIDE",
    "FULL_CPU",
    "SPATIAL_TILED_CPU",
    "CPU_MEMORY_REFUSAL",
    "CpuMemoryPolicyPreflight",
    "resolve_cpu_policy_preflight",
    "resolve_cpu_winsor_decision",
    "cpu_memory_policy_tokens",
    "cpu_winsor_decision_tokens",
    "cpu_winsor_retry_tokens",
    "cpu_winsor_refusal_tokens",
]

# Env override honoured by the wiring layer (documented expert/CI/test seam).
CPU_MEMORY_OVERRIDE_ENV = "ZSSS_CPU_MEMORY_OVERRIDE_BYTES"


@dataclass(frozen=True)
class CpuMemoryPolicyPreflight:
    """One frozen per-run CPU memory POLICY statement (pure).

    ``mode``: AUTO or OVERRIDE.  ``policy_ceiling_bytes`` bounds every
    runtime effective budget of the run.  ``requested_budget_bytes`` is set
    only in OVERRIDE mode (the explicit expert budget).  ``reserve_bytes`` is
    the named reserve recommended at preflight (informational; the runtime
    re-evaluation recomputes the reserve from the RAM available at execution
    time — this record never promises fixed future bytes).
    """

    mode: str
    total_ram_bytes: Optional[int]
    available_ram_preflight_bytes: int
    reserve_bytes: int
    policy_ceiling_bytes: int
    requested_budget_bytes: Optional[int] = None

    @property
    def is_override(self) -> bool:
        return self.mode == MODE_OVERRIDE


def resolve_cpu_policy_preflight(
    *,
    available_ram_preflight_bytes: int,
    total_ram_bytes: Optional[int] = None,
    mode: str = MODE_AUTO,
    requested_budget_bytes: Optional[int] = None,
    frame_bytes: int = 0,
    pool_workers: int = 1,
) -> CpuMemoryPolicyPreflight:
    """Resolve the once-per-run CPU memory POLICY (pure).

    AUTO ceiling: ``max(0, available_ram_preflight - reserve_preflight)``.
    OVERRIDE ceiling: the explicit requested budget verbatim.  The reserve
    comes from the stage-B named policy (:func:`recommended_reserve_bytes`),
    never an unexplained constant.
    """
    if int(available_ram_preflight_bytes) < 0:
        raise ValueError("available_ram_preflight_bytes must be non-negative")
    mode = MODE_OVERRIDE if mode == MODE_OVERRIDE else MODE_AUTO
    reserve = recommended_reserve_bytes(
        int(available_ram_preflight_bytes), int(frame_bytes), int(pool_workers)
    )
    if mode == MODE_OVERRIDE:
        if requested_budget_bytes is None:
            raise ValueError(
                "OVERRIDE mode requires an explicit requested_budget_bytes"
            )
        requested = int(requested_budget_bytes)
        if requested <= 0:
            raise ValueError("requested_budget_bytes must be positive")
        ceiling = requested
    else:
        requested = None
        ceiling = max(0, int(available_ram_preflight_bytes) - reserve)
    return CpuMemoryPolicyPreflight(
        mode=mode,
        total_ram_bytes=(
            int(total_ram_bytes) if total_ram_bytes is not None else None
        ),
        available_ram_preflight_bytes=int(available_ram_preflight_bytes),
        reserve_bytes=reserve,
        policy_ceiling_bytes=ceiling,
        requested_budget_bytes=requested,
    )


def resolve_cpu_winsor_decision(
    *,
    mode: str,
    n: int,
    frame_shape: Sequence[int],
    channels: int,
    dtype_itemsize: int = 4,
    winsor_limits: Tuple[float, float] = (0.05, 0.05),
    apply_rewinsor: bool = True,
    weighted: bool = False,
    scipy_backend: bool = False,
    available_ram_bytes: int,
    policy_ceiling_bytes: Optional[int] = None,
    reserve_bytes: Optional[int] = None,
    pool_workers: int = 1,
    min_tile_out: int = CPU_MIN_TILE_OUT,
) -> CpuMemoryDecision:
    """One CPU execution decision from (workload, RAM now, reserve, ceiling,
    mode) — pure, deterministic, delegating to the stage-B planner.

    Runtime re-evaluation: ``effective_budget = min(policy_ceiling,
    available_ram_now - reserve)`` is computed inside the planner exactly like
    this (the planner receives the runtime available RAM and the ceiling; when
    ``reserve_bytes`` is omitted it applies the named
    :func:`recommended_reserve_bytes` policy).  The frozen scientific ``N``
    is passed verbatim; only FULL_CPU / SPATIAL_TILED_CPU geometry /
    CPU_MEMORY_REFUSAL may differ between RAM simulations.
    """
    return plan_cpu_winsor_execution(
        n=n,
        frame_shape=frame_shape,
        channels=channels,
        dtype_itemsize=dtype_itemsize,
        winsor_limits=winsor_limits,
        apply_rewinsor=apply_rewinsor,
        weighted=weighted,
        scipy_backend=scipy_backend,
        available_ram_bytes=int(available_ram_bytes),
        reserve_bytes=(
            int(reserve_bytes) if reserve_bytes is not None else None
        ),
        policy_ceiling_bytes=(
            int(policy_ceiling_bytes)
            if policy_ceiling_bytes is not None
            else None
        ),
        mode=mode,
        pool_workers=int(pool_workers),
        min_tile_out=int(min_tile_out),
    )


# ---------------------------------------------------------------------------
# Provenance record builders (pure dicts, JSON-safe scalars; bounded — they
# never carry image/mask arrays or nested payloads).
# ---------------------------------------------------------------------------

def cpu_memory_policy_tokens(preflight: CpuMemoryPolicyPreflight) -> dict:
    """MEMORY_POLICY record tokens (mode, RAM, reserve, ceiling, override)."""
    tokens = {
        "mode": preflight.mode,
        "total_ram_bytes": preflight.total_ram_bytes,
        "available_ram_preflight_bytes": (
            preflight.available_ram_preflight_bytes
        ),
        "reserve_bytes": preflight.reserve_bytes,
        "policy_ceiling_bytes": preflight.policy_ceiling_bytes,
    }
    if preflight.is_override and preflight.requested_budget_bytes is not None:
        tokens["requested_budget_bytes"] = preflight.requested_budget_bytes
    return tokens


def cpu_winsor_decision_tokens(
    decision: CpuMemoryDecision,
    available_ram_runtime_bytes: int,
) -> dict:
    """CPU_WINSOR_MEMORY_DECISION record tokens for one CPU execution."""
    if decision.strategy == FULL_CPU:
        strategy = "full"
    elif decision.strategy == SPATIAL_TILED_CPU:
        strategy = "spatial_tiled"
    else:
        strategy = "refused"
    tokens = {
        "scientific_n": decision.n,
        "frame_shape": "x".join(str(d) for d in decision.frame_shape),
        "available_ram_runtime_bytes": int(available_ram_runtime_bytes),
        "effective_budget_bytes": decision.effective_budget_bytes,
        "estimated_peak_bytes": decision.estimated_peak_bytes,
        "strategy": strategy,
        "mode": decision.mode,
    }
    if decision.tile_shape is not None:
        tokens["tile_shape"] = ",".join(str(d) for d in decision.tile_shape)
        tokens["tile_count"] = decision.n_tiles
    return tokens


def cpu_winsor_retry_tokens(
    *,
    old_tile_shape,
    new_tile_shape,
    reason: str = "allocation_failure",
    attempt: Optional[int] = None,
    outcome: Optional[str] = None,
) -> dict:
    """CPU_WINSOR_MEMORY_RETRY record tokens (only on a bounded allocation
    retry; spatial tile shapes only)."""
    def _shape_token(shape):
        if shape is None:
            return None
        if isinstance(shape, (tuple, list)):
            return ",".join(str(int(d)) for d in shape)
        return str(int(shape))

    tokens = {
        "old_tile_shape": _shape_token(old_tile_shape),
        "new_tile_shape": _shape_token(new_tile_shape),
        "reason": reason,
    }
    if attempt is not None:
        tokens["attempt"] = int(attempt)
    if outcome is not None:
        tokens["outcome"] = str(outcome)
    return tokens


def cpu_winsor_refusal_tokens(
    *,
    scientific_n: int,
    effective_budget_bytes: int,
    minimum_estimated_bytes: int,
    reason: str,
) -> dict:
    """CPU_WINSOR_MEMORY_REFUSAL record tokens (truthful refusal)."""
    return {
        "scientific_n": int(scientific_n),
        "effective_budget_bytes": int(effective_budget_bytes),
        "minimum_estimated_bytes": int(minimum_estimated_bytes),
        "reason": reason,
    }
