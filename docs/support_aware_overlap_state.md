# Support-aware overlap photometry — mission state
Mission: zsss-support-overlap-p1-20260907
Base/branch: ede9d98ba46aa8bf2075e50b5719293d01f659a0 / beta

- [x] Junior read-only archaeology completed; see support_aware_overlap_phase0.md.
- [x] Targeted baseline: 285 passed / 5 pre-existing failures (exact IDs/log in archaeology).
- [ ] Coco bounded Phase 1 implementation and validation.
- [ ] Junior independent diff/science/tests review.
- [ ] Nono independent review ACCEPT.
- [ ] Literal required automated gates GREEN (baseline debt must remain explicit).
- [ ] Windows W80 A/B/C/D instructions and fresh-copy witness, fixed (1268) reference.
- [ ] Numerical W80 analysis + human visual acceptance.
- [ ] Phase 2 separately scoped/reviewed only AFTER Phase 1 and W80 acceptance.

## Phase 1 implementation status (Coco, 2026-09-07, iteration r1 / REWORK-1)

Addressed with evidence (module + tests green; targeted gates 375 pass / 5
pre-existing baseline failures only, no regressions):
- Loader HWC1 wrong-axis invalidity bug fixed (spatial reduction over the
  final channel axis; (H,W) mask for every HWC layout); docs aligned; loader
  roundtrip tests (default/opt-in science identity, CHW/HWC/mono/nonfinite,
  failure arity).
- Geometry validation: missing/malformed/non-finite/singular M now raises an
  explicit ValueError in the strict helper and returns (None, stable reason)
  in geometry_support_mask_or_none; seam-level sky_mean/linear_fit geometry
  estimators return neutral with reason; never identity/full-support guess.
- Sampling: count-first bounded coordinate generation (no full-frame
  np.nonzero before the budget), finite positive max_samples validation.
- Corrected original-bug witnesses (true numeric-zero unsupported padding,
  >25% unsupported, legacy P25 collapse, new recovery <=2e-6) and actual
  production-warp constant/structured scenes at rotations 0/22.5/45/70/100
  and +/- translations, coverage bands, legitimate zero, nonfinite.
- Bayer influence radius validated empirically against the REAL debayer for
  GRBG/RGGB/GBRG/BGGR (Chebyshev radius 1, incl. borders).
- Plain-classic batch_size==1 hidden sky subtraction bypassed (BS1).
- Resume: minimal scalar Classic normalization-science contract marker
  written to the manifest and validated headless (pre/post-fix mixing
  fail-closed; legacy-unchanged accepted only when science unchanged).
- queue_manager Classic _process_file requests M (return_M=True,
  transform_only=False) for plain-classic sky/linear sessions.

NOT yet integrated (remaining seam, bounded for rework-2):
- Per-source support carrier (M + content validity) through the worker
  classic item path into _stack_batch normalization; support-aware
  replacement of the full-frame percentile helpers inside
  _normalize_sources_against_reference; neutral missing-support behaviour in
  all callers; reference content-validity capture at the reference seam;
  HSI fixtures migrated to explicit known-support carriers with analytic
  paired truth where they encode legacy Classic offsets; diagnostics sink;
  resume marker tests (clean checkpoint equivalence, dirty refusal);
  complete tests/ rerun with baseline differential.

Evidence: .a2a-reports/zsss-support-overlap-p1-20260907/ (baseline-*,
rework1-targeted.log, junit). Targeted after REWORK-1: 375 passed /
5 failed (same 5 pre-existing baseline IDs).

Phase 0 amendment: bypass hidden batch_size=1 sky subtraction for plain Classic; carry explicit invalidity before loader/warp repairs; protect checkpoint science across algorithm change. Do not alter upstream preparation or registration output.
Mosaic NOT AFFECTED by proven batch-index-0 path. Weighting audit only; RGB variance follow-up separate.
No commit/push/merge/tag/release/deploy performed or requested for this dispatch.
Expected initial worker report: /home/tristan/.openclaw/workspace/.a2a-reports/zsss-support-overlap-p1-20260907.coco.r0.md
Next single step: Coco REWORK-1 in SAME mission/session (no reset), completing authorized production seam and correcting Junior findings; callback-driven, no polling.

## Junior r0 review / continuation decisions

2026-09-07: 84 primitive+Drizzle tests independently passed; full baseline JUnit verified (2730 pass/36 fail/2 skip). Production normalization NOT wired. No human approval required for already-authorized seam/HSI expectation/Resume scalar decisions. REWORK-1 addresses loader HWC1 shape bug, singular-M fabricated support, unbounded coordinate allocation, incorrect zero-padding witness and missing end-to-end Phase1 integration. Exact review: /home/tristan/.openclaw/workspace/.a2a-reports/zsss-support-overlap-p1-20260907.junior.r0.md
Expected next callback: zsss-support-overlap-p1-20260907.coco.r1.md, phase rework-1. No commit or publication.

## Junior r1 review — REWORK-2

132 primitive+Drizzle tests independently passed (5.32s), diff-check clean. Still no production overlap normalization: old full-canvas call remains active. No new technical decision blocker; Coco reported execution-budget exhaustion. Next action is SAME-session REWORK-2 with integration-first priority, then end-to-end none/order/batch/resume/diagnostics validation and full-suite differential. Exact review /home/tristan/.openclaw/workspace/.a2a-reports/zsss-support-overlap-p1-20260907.junior.r1.md. Expected report zsss-support-overlap-p1-20260907.coco.r2.md, phase rework-2. No Phase1 acceptance, Nono not yet dispatched.

## REWORK-2 completion — production seam integrated (Coco r2)

2026-09-07. The production seam is now WIRED and validated:

- `_process_file`: plain-classic sky/linear sessions load with the opt-in
  loader invalidity report (science bit-identical), derive the truthful
  content-valid mask (conservative Bayer dilation when CFA-debayered) and
  publish the per-frame transient support carrier `(M, content_valid,
  src_shape)` into a THREAD-LOCAL slot immediately before the successful
  return (cleared at entry and by the consuming worker site; race-free).
- Worker classic item path consumes the carrier and appends it as the 6th
  element of the batch tuple (disk/BS1 path keeps the carrier in memory next
  to the saved image/mask paths); `_stack_batch` accepts >=5-tuples, keeps
  carriers in lockstep with the accepted reduction list, and passes them to
  the normalization seam.
- `_normalize_sources_against_reference`: per-frame paired-overlap estimators
  (`estimate_sky_mean_from_geometry` / `estimate_linear_fit_from_geometry`)
  on real M + truthful content + session reference content; formulas
  byte-identical (P25/P90 model and aligned-luminance sky), only the sample
  support changed. Missing carrier / content / M => NEUTRAL with a stable
  reason (never the legacy full-frame helpers, never an all-valid guess);
  missing session reference still fails closed (RuntimeError); shape mismatch
  raises loudly. `none` remains a strict image no-op.
- Reference seam: `_p1_reference_content_mask` resolves the fixed
  reference's original FITS from header provenance and captures its truthful
  loader-original content validity once per session (no all-valid fallback).
- Geometry wrappers no longer default missing content masks to all-valid:
  explicit neutral reasons `no_source_content_validity` /
  `no_reference_content_validity` added to the estimator entrypoints.
- Diagnostics: bounded per-frame scalar records (frame/method/geometry/
  effective/common counts/fractions/estimator/reason/offset); no canvas
  masks retained past reduction; fail-open by design.
- Module/harness witnesses: duplicate shadowed constant-sky test removed;
  structured-sky scene rebuilt from a shared analytic sky evaluated at the
  forward transform coordinates (physically valid; warp-vs-analytic verified
  to ~6e-8); HSI closure + backend-parity + pinned-reference fixtures now
  carry EXPLICIT known support (identity M + known-finite content) with the
  same reference-content seeding; new `tests/test_p1_seam_witnesses.py`
  covers loader truthfulness/bit-identity, seam normalization, neutral
  missing-support semantics, supported-zero vs padding-zero pixels, carrier
  thread-local semantics, and the scalar manifest marker (contract matrix,
  scalar-only persistence with no M/mask serialization, resume gate
  refusal/accept).
- Validation: targeted group 509 passed / 5 failed = the exact 5 pre-existing
  baseline IDs (rf2 M16 witness data + 4 resume env); complete `tests/`
  2838 passed / 35 failed / 2 skipped / 179 warnings (baseline 2730/36/2/179)
  — zero NEW failures, one pre-existing failure no longer reproduced,
  ~108 new tests added.

Evidence: .a2a-reports/zsss-support-overlap-p1-20260907/rework2-full.log,
rework2-failures.txt vs baseline-failures.txt. Report:
.a2a-reports/zsss-support-overlap-p1-20260907.coco.r2.md.

## Junior r2 review — ACCEPT, Nono dispatched

2026-09-07: full suite completed 2838 pass/35 fail/2 skip (916s); differential vs baseline 2730/36/2 => +107 new tests, ZERO new failures (35 = subset of 36, one flaky cov06b passed). Targeted mission scope 388 pass/4 fail (4 = pre-existing resume IDs). Production seam verified directly. Junior ACCEPT; Nono independent review dispatched scope-limited. Expected .a2a-reports/zsss-support-overlap-p1-20260907.nono.r0.md, phase review-1. W80 preparation only after Nono ACCEPT.

## Nono review — FINDINGS -> Junior REWORK-3

Nono: FINDINGS (F1 medium stacked-ref content resolution gap breaking clean-resume equivalence; F2 missing session WARN; F3 unused diagnostics summary; F4 style). 10/10 acceptance criteria MET, no brightness/safety defect. Junior confirmed F1 independently. Dispatch final corrective REWORK-3 (F1+F2+test, optional F3), same Coco session. Expected report zsss-support-overlap-p1-20260907.coco.r3.md, phase rework-3. Review record .a2a-reports/...nono.r0.md and .junior.nono-review.md.

## REWORK-3 completion — Nono F1/F2/F3 (Coco r3)

2026-09-07. Final corrective iteration under the 3-iteration limit:

- F1: `_p1_reference_content_mask` now re-derives the reference content
  evidence through the SAME identity resolution as the pinned reference:
  persisted resume-reference identity (`_resume_reference_identity` ->
  `_resolve_source_path`, original -> verified `stacked/` counterpart via
  size+mtime `_identity_matches`) first, then header provenance over each
  search dir's original AND `stacked/` subfolder, each verified by the same
  identity machinery.  A clean resume whose reference was moved to stacked/
  therefore re-derives the content mask; post-resume frames keep the same
  correction as pre-resume frames (no silent NEUTRAL discontinuity).
- F2: session-level WARN (logger.warning + update_progress) emitted when the
  OVERLAP contract (plain-classic sky_mean/linear_fit) requires reference
  content evidence that cannot be re-derived; fail-open preserved (never
  aborts the run; deterministic neutral + stable reason unchanged).
- F3: `_p1_diagnostics_summary` is now wired into end-of-run reporting via
  the worker release seam (`_release_norm_reference`): bounded accepted/
  neutral summary logged when frames were recorded, session-level WARN when
  any frame was neutral; state cleared as before.
- F4: reason-ordering unchanged (documented style).
- Regression tests added (tests/test_p1_seam_witnesses.py W6): stacked/
  rediscovery on resume (identity + header-provenance orders, truthful
  invalid-patch mask, post-resume correction equivalence), unresolvable
  WARN path, and release-seam summary/WARN wiring.
- Validation (owner scope, NO full suite): mission files + touched resume
  test = 236 passed / 7 warnings / EXIT=0 (rework3-mission.log); Nono's six
  files alone = 232 passed (was 232 at Nono review; +3 net new W6 tests in
  the witness file, 1 unpack fix).  Exact baseline differential unchanged:
  the 4 pre-existing test_resume.py baseline failures were NOT run/touched.

No commit/push/merge/tag/release. Report: .a2a-reports/zsss-support-overlap-p1-20260907.coco.r3.md

## REWORK-3 accepted by Junior — Nono re-review pending

F1/F2/F3 resolved and independently verified (236 pass/7 warnings/EXIT=0). Final corrective iteration (3/3) done. Remaining gate: Nono ACCEPT on r3 delta. Record .a2a-reports/...junior.r3.md. Expected Nono re-review report zsss-support-overlap-p1-20260907.nono.r1.md, phase review-2. W80 prep only after Nono ACCEPT.

## Phase 1 COMPLETE — ACCEPTED

Nono re-review ACCEPT (F1/F2/F3 resolved, 10/10 criteria, 236 mission-scope pass, no residual findings). Phase-1 gate GREEN. W80 witness instructions written docs/w80_witness_instructions.md. Next: Tristan executes W80 A/B/C/D on Windows (fresh copies, pinned 1268 reference); then numerical+visual acceptance; Phase 2 (Reproject) gated separately. No commit/push/merge/tag/release performed.
