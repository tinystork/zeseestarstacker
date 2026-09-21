# GPU residency — architectural decision record (ZSSS)

- **status:** DECIDED — **retain host-centric architecture**
- **date:** 2026-09-21
- **scope:** architectural position only; **no implementation, no branch change, no CuPy change**
- **related:** `docs/gpu_acceleration_architecture.md` (the acceleration architecture this decision
  confirms), `seestar/core/gpu.py`, `seestar/core/stack_gpu.py`, `seestar/core/gpu_vram_planner.py`

## Question that was asked

> Would a GPU-**resident** execution model materially improve ZeSeestarStacker itself, independently
> of any other product, while preserving its scientific invariants, memory policy, CPU parity and
> operational reliability?

This was asked independently of any consumer. A future consumer/beneficiary (e.g. ZeCalibrator) is
**not** a justification for changing ZSSS and did not influence this decision.

## Evidence (verified in code and documentation)

* **Every GPU touchpoint is a single reduction call**: `_gpu_reduce` performs
  `cp.asarray(host_stack)` → CuPy sorting-based kernel → `cp.asnumpy(result)`; results are "always
  returned as NumPy arrays". There is **no device-resident state between stages**.
* **GPU-eligible scope** is deliberately narrow: the sorting-based reductions (kappa-sigma,
  linear-fit-clip, median, winsorized-sigma-clip), implemented as exact twins of the CPU reference
  algorithms.
* **CPU-only by design**: mean, alignment, drizzle, reprojection, feathering, quality metrics,
  streaming, and the **tiled / HQ path**. The tiled/HQ routing decision happens **before**
  `_gpu_reduce` is reached, so a tiled stack runs on CPU regardless of GPU size; the tiled path is
  documented as intentionally not redesigned.
* **VRAM policy**: per call, dynamic — the estimated footprint (stack + sorting peak, ~4× the
  float32 stack) must fit in **60 % of currently free VRAM**, else CPU. No fixed threshold.
* **Measured reference (MX150 2 GB, 1080×1920 float32, documented, not generalisable)**:
  N=20 ⇒ CPU ≈1.92 s vs GPU ≈0.99 s (~1.9×); N=30 ⇒ 4.01 s vs 1.46 s (~2.7×); N=50 ⇒ real OOM,
  guard routes to CPU. (Independent check: 8.29 MB/plane ⇒ N=20 ≈663 MB peak ✓, N=30 ≈995 MB ✓,
  N=50 ≈1.66 GB > 1.2 GB budget ✓ — the guard is coherent.)
* **Failure semantics**: any GPU failure/rejection ⇒ logged warning + automatic CPU rerun. The CPU
  reference is authoritative.

## Decision

**Retain the host-centric architecture.** Reasons:

1. The GPU already accelerates the part it suits, through a per-call, dynamic, fallback-safe policy
   with measured wins on the eligible subset.
2. The remaining CPU-only stages are where the pipeline's **memory strategy** lives (tiling/HQ,
   streaming, exact-N policy); making them device-resident is a rewrite, not an optimisation.
3. Residency would require device state across stages, a second (VRAM) memory/tiling planner,
   end-to-end parity obligations and structurally harder fallback — while being **unavailable
   precisely for the largest workloads** (tiled/large stacks).
4. Nothing in ZSSS's own requirements forces it; the host-centric model is a deliberate,
   documented engineering position.

## Reopen triggers (the decision is reversible, not vague)

Reconsider only if one of these appears:

1. **Hardware/workload shift** — representative hardware or workloads materially change, especially
   where the current VRAM guard or tiled routing becomes the limiting factor.
2. **Measured stage dominance** — representative profiling shows the CPU-only stages (alignment /
   reprojection / drizzle / related) dominate end-to-end performance enough to justify
   reconsideration (the target would then be *those stages*, not residency as such).
3. **Product requirement** — a genuine requirement for an uninterrupted device-resident pipeline
   that the host-centric model cannot satisfy.

## Mandatory preconditions if residency is ever reopened

* CPU remains the **canonical** reference;
* stage-level **and** end-to-end CPU/GPU scientific parity defined **before** any migration;
* exact-N / memory-policy invariants preserved;
* VRAM planning/tiling made explicit;
* fallback/recovery semantics designed before implementation;
* no silent scientific-mode change;
* integration with any other product is a possible **beneficiary**, never the justification.

## Explicitly out of scope

No implementation was authorised by this decision; no branch was modified; no CuPy behaviour was
changed; `docs/gpu_acceleration_architecture.md` remains the authoritative description of the
shipped acceleration architecture.
