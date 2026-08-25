# RF-2 — registration-reference architecture (state)

- Branch: `feature/registration-field-rotation` @ `cf8d25c` (research gate) +
  RF2-02 production implementation (working tree, uncommitted).
- Gate: **RF2-02 ACCEPTED** after independent architect review (production change,
  working tree intentionally uncommitted).
- Full research report: `docs/registration_reference_rf2.md`
- Production contract + diagnostics schema: `docs/registration_reference_rf2_production.md`

## Verdicts (summary)

- **BATCH_INVARIANCE** — only the **immutable selected reference** is invariant;
  geometry measured on per-frame transforms (max |ΔM| = 0 across bs 1/5/10,
  unitless matrix-element difference).  The previous "first-batch freeze"
  candidate is batch-dependent and REJECTED.
- **ORDER_INVARIANCE** — only the immutable target is order-invariant (max |ΔM| = 0
  natural vs reversed); evolving and first-batch-freeze are order-dependent.
- **BIAS_OBSERVABILITY** — representable bias (translation) is *corrected*;
  non-representable bias (radial) is *hidden* by any absorbing target; only the
  immutable single-frame reference keeps fit residual == true error.  Bounded to
  the tested systematic-bias construction (translation / rotation / quadratic
  radial).
- **RESUME_CONTRACT** — reproject/drizzle/mosaic fail closed (only plain classic
  SUM/W resumable).  Unchanged by RF2-02.
- **DRIZZLE_PREWARP** — the computed `warpAffine` result is **DEAD** on the
  standard Drizzle path.  Removed in RF2-02 via the `transform_only` contract.

## Decision — accepted and implemented (RF2-02)

**`STABLE REGISTRATION TARGET REQUIRED`** — hold the initially-selected reference
frame (`_get_reference_image`, manual or auto-best) immutable for the whole run.

Production implementation (`seestar/core/alignment.py`,
`seestar/queuep/queue_manager.py`, new `seestar/core/registration_diagnostics.py`):

1. **Immutable target** — `reference_image_data_for_global_alignment` is never
   reassigned to a cumulative stack at the batch/finalize seams; only the
   cumulative-stack WCS/grid update side effect (`_solve_cumulative_stack`) is
   preserved.
2. **Passive diagnostics** — versioned JSON-Lines `registration_diagnostics.jsonl`
   in the output folder; fail-open; never affects science.
3. **Drizzle dead pre-warp removed** — `_align_image(transform_only=True)`
   skips the warp, returns the same 2x3 tf; classic path unchanged.

## Validation (RF2-02, exact commands recorded in the RF2-02 report)

- New RF2-02 tests (`tests/test_rf2_production_impl.py`): 11 passed.
- RF2 research tests (2 files) + RF1 four files: 71 passed (includes the updated
  audit test pinning the now-empty worker-target mutation set).
- Drizzle (5 files): 33 passed.  HSI closure (6 files): 111 passed.
- Resume: `test_resume.py` 125 passed, `test_qt_last_stack_resume_m23.py` 21 passed
  (run separately — a pre-existing `seestar.gui` stub cross-file isolation issue
  makes the two fail to collect when run in that order; unrelated to RF2-02).
- Extra directly-affected checks: `test_queue_manager_reproject.py` 28 passed,
  `test_hierarchical_stacking_integrity.py` 37 passed, reliability/drizzle
  (5 files) 18 passed.

No commit / push / merge / tag / history rewrite. Research files preserved.

## Independent architect review — 2026-08-25

- Diff and ownership seams inspected against `cf8d25c`; no unexpected file or
  scientific-contract change found.
- Independent gates: 82 RF1/RF2, 98 Drizzle/reproject/HSI-integrity, 111 HSI
  closure, 125 resume, and 21 Qt last-stack/resume tests passed (**437 total**).
- Post-review focused rerun: 11 RF2 production tests passed; production seam
  witness exited 0; production modules compile; `git diff --check` clean.
- Review-only corrections clarified that target provenance is session-local in
  the header returned by `_get_reference_image` (the temporary
  `reference_image.fit` omits that card) and removed stale pre-warp comments.
- Verdict: **ACCEPT**.  No commit or remote action performed.

Next scientific gate remains multi-session temperature/focus evidence before
any Euclidean-to-Similarity model change; RF2-02 does not retain scale.
