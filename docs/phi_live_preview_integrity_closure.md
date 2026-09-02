# PHI-LIVE-PREVIEW-INTEGRITY-V2 — Project Closure

**Status:** CLOSED (PHI-R1 → PHI-R5 all accepted; this document is the durable
project-local closure evidence).  Working tree on
`fix/live-preview-histogram-v2` @ `63e9b875e4db140333f2cb8691235f54bc629f4f`
remains uncommitted by design (review chain); no commit/push/merge/tag/deploy
was ever performed by the PHI missions.

**Durable reports (index):** `/home/tristan/.openclaw/workspace/.a2a-reports/` —
`PHI-R3.md`, `PHI-R3.1.md`, `PHI-LIVE-PREVIEW-INTEGRITY-V2.nono.r2.md`,
`PHI-LIVE-PREVIEW-INTEGRITY-V2.coco.r2.md`,
`PHI-LIVE-PREVIEW-INTEGRITY-V2.nono.r3.md`,
`PHI-LIVE-PREVIEW-INTEGRITY-V2.coco.r3.md`,
`PHI-LIVE-PREVIEW-INTEGRITY-V2.coco.r4.md`,
`PHI-LIVE-PREVIEW-INTEGRITY-V2.coco.r5.md`,
`PHI-LIVE-PREVIEW-INTEGRITY-V2.coco.closure.md` (this closure).

---

## 1. User-visible resolution

For the Qt live preview (Option-A payloads from the active Classic SUM/W and
standard Drizzle producers), the viewer now:

1. **Preserves finite float headroom end-to-end.**  The anchor mapping and the
   white-balance derivation never hard-clip the bright tail to `1.0`; only the
   final `uint8`/`QImage` display conversion bounds values.  No NaN/Inf ever
   propagates into the analysis buffers.
2. **Shows an explicit analysis-range histogram.**  512 bins over
   `[0, upper]` with `upper = max(1.0, finite max)`: identical to the legacy
   `[0,1]` model when no headroom exists, extended above `1` when HDR/WB
   headroom is present; counts and stats always describe the same finite
   non-negative sample.
3. **Makes both black point and white point genuinely usable in analysis
   units** — values above `1` are first-class on the histogram markers, the
   sliders/spins, the drag conversion and the display stretch; all surfaces
   (inline + detached) stay synchronized, and a white point above `1` visibly
   recovers preserved headroom.
4. **Renders the visible Option-A display from the preserved float
   analysis/WB source** with the user black/white points applied *before* the
   final `uint8` conversion (the only clipping boundary); the legacy
   single-array QImage path keeps its historical `[0,1]` Tk-parity semantics.
5. **Rejects stale/duplicate/foreign previews on the active producers** with a
   trace-independent monotonic acceptance gate: `PREV_SEQ` + durable
   `PREV_RUN` (run/session identity) are stamped on every active-producer
   payload regardless of the `ZSSS_PHI_TRACE` debug gate, so an older or
   duplicate emission — or a late payload of a previous producer session —
   can never replace the current display or poison the new run's sequence
   high-water mark.
6. **Retired the dead legacy preview machinery (R5).**  The M3-D OBSOLETE
   LEGACY incremental-Drizzle preview/process chain
   (`drizzle_batch_worker`, `_start_drizzle_process`,
   `_process_incremental_drizzle_batch`, `_update_preview_incremental_drizzle`,
   `_wait_drizzle_processes`), its sole-purpose state (incl. the retired
   `drizzle_executor` and the `cumulative_drizzle_data*` carrier) and the dead
   reproject/master preview carrier were removed after Tristan's explicit
   human-gate approval; supported Classic, standard Drizzle, reproject and
   final-output science paths are untouched.

## 2. Acceptance matrix (R1 → R5)

| Phase | Objective | Evidence / tests (key) | Accepted result |
|---|---|---|---|
| **R1 — investigation/archaeology** | Rank live-preview integrity hypotheses (premature clipping, histogram domain mismatch, ordering, cumulative resampling, legacy route reachability) before any behaviour change. | `docs/phi_viewer_archaeology.md` (hypotheses 1-5 + decision tests). | Investigation complete; decisions gated to R2+ (accepted baseline `9cd8e85`). |
| **R2 — instrumentation & witnesses** | Debug-gated compact `PREVIEW_STAGE` telemetry (stages, deterministic counters, producer sequence/resolution metadata) and deterministic reproduction witnesses; no behavioural change. | `docs/phi_preview_instrumentation_r2.md` §1-6; `tests/test_phi_preview_pipeline.py` (witnesses W1-W4, producer isolation, counters, truthful PREV_REQ/RES/CAP, gate-off no-op) — 18 PHI tests at R2 closure. | Committed baseline `63e9b87` + REWORK-1/1.2/1.3 corrections accepted. |
| **R3 — behavioural integrity correction** | Preserve finite float headroom through analysis/WB/histogram; clamp only at display; monotonic producer-sequence gate; explicit analysis-range histogram semantics. | Rewritten W1 headroom witness; R3 gate tests; `test_preview_analysis.py` semantics (40); `docs/phi_preview_instrumentation_r2.md` §7. | Delivered; review chain led to R3.1 (REJECT → rework). |
| **R3.1 — analysis-unit BP/WP + float display** | Both black/white points genuinely adjustable in analysis units (incl. > 1); Option-A visible display rendered from preserved float with BP/WP before final uint8; legacy path stays `[0,1]`; deterministic reconcile on range change. | `test_phi_r31_*` (6 tests: range>1, WP>1 set/drag synced everywhere, display recovers headroom, extended validation, legacy `[0,1]`, shrink reconcile); view/controls suites. | Delivered; review chain led to R3.2 (REJECT → rework). |
| **R3.2 — trace-independent producer identity + frozen-range reconcile** | Producer ordering/run identity emitted independently of the debug gate (production gate); durable run/session id bound at the run lifecycle; foreign old-run payloads cannot poison the new run; frozen histogram view range reconciled on model-domain change. | `test_phi_r32_*` (8 tests incl. real-backend stamping, foreign-run arrival orders, gate-off metadata contract); `docs/phi_preview_instrumentation_r2.md` §9. | Delivered; review chain led to R3.3 (REJECT → rework). |
| **R3.3 — detached policy mirror + model→legacy reset** | Inline/detached histogram surfaces share the frozen/unfrozen view policy (no artificial detached freeze); model→legacy transitions reset to a valid `[0,1]` view; real `SeestarQueuedStackerBackend` stamping covered under trace off. | `test_phi_r33_*` (4 tests incl. Nono F2/F3 repros + real-backend stamping). | **Accepted** (Nono review-4 + Junior). |
| **R4 — route reachability & integrity parity** | Live route/dispatch inventory (Classic, standard Drizzle, legacy incremental Drizzle) with exact predicates; legacy incremental Drizzle proved dead/guarded (no speculative removal before the human gate). | `test_phi_r4_*` (3 dispatch-predicate tests; the R4 AST dead-subgraph guard test was superseded in R5 by the retirement regression `test_phi_r5_legacy_machinery_retired_no_supported_dispatch_invokes_it`); `docs/phi_preview_instrumentation_r2.md` §11. | **Accepted** (R5 baseline). |
| **R5 — approved retirement** | Remove the approved unreachable legacy preview/process/carrier machinery; migrate forensic coverage to positive supported-path invariants; regression-test that no supported dispatch invokes the removed legacy carrier. | Retirement regression `test_phi_r5_legacy_machinery_retired_no_supported_dispatch_invokes_it`; migrated `test_supported_drizzle_preview_and_final_save_invariants`; boring/worker/save-final test migrations; `docs/phi_preview_instrumentation_r2.md` §12. | **Accepted** (Nono review-r5 + Junior). |

## 3. Remaining non-blockers (outside PHI closure — no concealment)

- **No real hardware/GPU run was performed.**  All evidence is offscreen-Qt +
  deterministic fixtures.  Real-run cadence tracing of the supported Classic /
  standard Drizzle preview routes under load remains a possible future (R6)
  activity; PHI makes no claim of hardware validation.
- **Known unrelated baseline test failures (pre-existing, proven by stash
  probes against the accepted baseline; none are PHI regressions):**
  1. `tests/test_resume.py` — `test_start_processing_valid_resume_pins_reference_and_queue`,
     `test_start_processing_quality_weighted_captures_q_ref_once_fresh`,
     `test_start_processing_quality_weighted_resume_skips_recomputation`,
     `test_repeated_start_binds_fresh_canonical_config`;
  2. `tests/test_reproject_zm_wcs_fix.py::test_mode0_final_keeps_science_with_artifact_reference_header`;
  3. `tests/test_boring_drizzle_boundary.py::test_classic_memmaps_use_fixed_reference_grid_shape`;
  4. `tests/test_save_final_stack.py::test_save_final_stack_radec_from_reference_header`;
  5. `tests/test_worker_incremental_drizzle.py::test_worker_calls_add_frame_to_drizzle_accumulators`;
  6. `tests/test_reliability_source_immutability_r1.py` — `test_worker_minimal_run_keeps_source_when_move_stacked_false`,
     `test_worker_minimal_run_moves_source_by_default`;
  7. `tests/test_exposure_metadata_contract.py` — `test_resume_continuation_no_double_count`,
     `test_drizzle_post_admission_side_effect_failure_keeps_lockstep`,
     `test_drizzle_failed_add_increments_neither`.
  These live in the separate accepted checkpoint-startup / resume-contract /
  reference-materialization work area and are not repaired or concealed by
  PHI.
- **Historical docs may be stale.**  R1/R2-era statements that predate the
  accepted behavioural corrections (e.g. pre-R3 "[0,1] histogram" contract
  wording, R2 gate-off/no-metadata statements, R4 "recorded, not removed"
  wording for the legacy routes) remain as *historical records*; the
  living instrumentation document (`docs/phi_preview_instrumentation_r2.md`)
  marks superseded statements and is the navigation anchor (see §4).  No
  historical claim was silently rewritten.

## 4. Navigation

- Living PHI document (route inventory, per-route contract, per-phase
  records R2→R5): **`docs/phi_preview_instrumentation_r2.md`** — see its
  header status and §§7-12.
- Investigation history: `docs/phi_viewer_archaeology.md`.
- Display/analysis target contract + amendments: `docs/output_truthfulness_preview_audit.md` (§5.2/§5.3 amendments).
- Durable mission reports: `/home/tristan/.openclaw/workspace/.a2a-reports/` (index in §0 above).
