# ZSSS Support-Aware Overlap Photometry — Phase 0 archaeology
Date: 2026-09-07. Architect: Junior.
Mission: zsss-support-overlap-p1-20260907.
Repository: /home/tristan/.openclaw/workspace/projects/zeseestarstacker
Verified initial branch: beta. HEAD: ede9d98ba46aa8bf2075e50b5719293d01f659a0 (8.3.0).
Initial and post-baseline git status: clean. No production changes during archaeology.
Line references below describe this exact base, not future edited line numbers.

## Findings (mandatory ten questions)
1. CONFIRMED: queuep/queue_manager.py:14433 _normalize_sources_against_reference prepends private _norm_reference and calls core/normalization.py helpers. sky_mean uses full aligned luminance P25, no support. Reference immutable capture at 7558, helper 14410. Preserve it.
2. CONFIRMED: core/alignment.py:92 _align_cpu uses INTER_LINEAR, BORDER_CONSTANT, NaN padding; _align_image:367 converts NaN to numeric zero, then retains existing clipping. This occurs before queue normalization. Do not change that image warp science.
3. CONFIRMED: core/normalization.py linear_fit uses full-frame P25/P90 (per-channel RGB) and a=where(delta_src>1e-5,delta_ref/max(delta_src,1e-9),1), b=ref_low-a*src_low. Preserve this model/formula, change only corresponding sample support.
4. NO: _normalize_sources_against_reference receives arrays only. _stack_batch already has valid_pixel_masks_for_coverage, but _process_file:10906 derives them from luminance thresholds (or all True for very dark arrays). They are not truthful geometry; simply forwarding them is insufficient.
5. core/alignment.py:134 exposes return_M=True; with return_diagnostics=True returns image,success,M,diag. Classic caller 10856 currently omits return_M and drops M. Request return_M=True, transform_only=False on this Classic call only. Keep same features, acceptance, Euclidean scale=1, interpolation, canvas, reference, clipping. Drizzle caller 10840 continues transform_only=True.
6. core/drizzle_background.py:316 robust_location = finite float64 samples, median, MAD*1.4826, sigma=3, 3 clipping iterations, degenerate-scale <=1e-12 break; exact result/used-count contract. Estimator :559 samples deterministic regular grid <=250000, maps original pixel centers via affine or native/reference WCS, bilinearly samples immutable anchor; overlap requires in-grid, positive source validity/weight and finite source+anchor in all channels. Minimum 200 samples; insufficient => zero correction and reason. Per-channel paired differences, subtract offsets before deposit. Extract ONLY truly generic math with bit-consistency witnesses, not Drizzle geometry/model.
7. Standard non-mosaic, non-reproject Drizzle routes at queue_manager:8403 directly to _add_frame_to_drizzle_accumulators and never _stack_batch/_normalize_sources_against_reference. Preserve it. Do not overgeneralize to the historical drizzle+reproject flag combination, which dispatches earlier through reproject_between_batches at 8286.
8. PROVEN AFFECTED: non-mosaic Classic with reproject_between_batches=True collects aligned arrays into five-tuples and calls _stack_batch at 8314 (tail at 8842). Classic with reproject_coadd_final=True also reaches _stack_batch via normal Classic batching, but _is_plain_classic (14340) returns False. At 13723, non-plain multi-image batches call legacy helpers with THIS batch index 0; singleton mean skips it. Phase 2 must address precisely these paths and inspect any enabled hybrid separately. Reproject resume currently unsupported; do not invent support in Phase 1.
9. Mosaic NOT AFFECTED by that specific batch-local normalization path. Worker branches at 8142 / 8195 append panels to all_aligned_files_with_info_for_mosaic, finalization at 8780 calls _finalize_mosaic_processing (9917), which uses reproject_and_coadd per channel, not _stack_batch. The non-plain comment mentioning Mosaic is not reachability proof. No Mosaic implementation changes authorized here.
10. Classic clean resume uses committed cumulative SUM/WHT (+ existing coverage support accumulators), ledger/reference identity/plan, atomic manifest dirty-before-mutation and clean-after-commit. Reference is resolved/pinned again and normalization anchor re-captured. Per-frame M/footprint is transient through registration->normalization->reduction and need not be persisted. Existing batch=1 disk intermediates carry aligned image/mask and must preserve semantics. Future transforms/support can be recomputed after a clean boundary. docs/resume_contract_v2.md authoritative; dirty checkpoints remain refused.

## Additional findings / design amendments before implementation
- queue_manager:10958 has a SECOND additive sky subtraction solely for batch_size==1, independent of normalize_method. This contradicts end-to-end none and order/batch invariance. Bound fix: bypass this competing branch for plain Classic only; all Classic correction must come from immutable-anchor normalization. Leave other paths unchanged. This is necessary to satisfy the requested contract, not a new background model.
- core/image_processing.py:62 loader already repairs raw nonfinite values to zero BEFORE normalizing each image by its own min/max; debayer/WB/hot-pixel preparation then runs. Checking finite(aligned) alone cannot recover original invalid provenance, and geometry must not be inferred from repaired zeros. Carry original explicit invalidity opt-in/transiently where needed through BOTH source and reference preparation, without changing existing image values, loading normalization, registration, star detection, WB or hot-pixel science. Account conservatively for interpolation/debayer influence, with tests. Legitimate raw/prepared zero is valid.
- OpenCV NaN border behavior can contaminate neighbors even at zero interpolation weight. Test the EXACT production warp against support, including identity/boundary/fractional translations; a simple thresholded ones warp without boundary validation is not automatically sufficient. Conservative documented interior/erosion belongs to reliability, not aggressive output cropping.
- Keep scientific normalization domain clear: synthetic offset truth is at prepared/aligned source input; raw FITS loading already min/max-normalizes. Do not redesign this upstream science.
- Same-version pre-fix and post-fix checkpoints must not silently mix normalization/support semantics. Inspect current version/fingerprint validation and bind a minimal scalar scientific contract identifier if needed; no M or full masks added to checkpoint. Maintain canonical configuration/fingerprint agreement and fail-closed Resume v2.
- Phase 0 invalidates any assumption that changing only two percentile helpers suffices. Amended plan above handles proven Classic boundaries; Phase 2 remains gated.

## Weighting audit (audit only)
core/weights.py:28/78 estimate statistics over whole image (support-blind).
FWHM: scalar min_fwhm/median_fwhm -> constant map -> divide by its own max => map=1 for ordinary finite positive estimates, monochrome AND RGB; fallback branches also ones. Pathological nonfinite maps are not proof of an intended algorithm.
Variance mono: var/var => constant 1; invalid variance gives uniform 1e-6 then max-normalizes to 1. Thus ordinary effective scalar remains 1.
Variance RGB: per-channel inverse variances normalized by maximum of whole RGB map retain relative channel ratios. queue_manager:13739 then nanmean(map) yields a genuine per-image scalar applied to quality_weights. Support contamination can therefore affect effective RGB weight. This is not repaired here.
Numerical read-only probe (NumPy RNG seed 12, 128x128, channel std ratio 1:2:3):
mono effective scalar 1.0; RGB channel means [1.0,0.25,0.11111111664489118], effective scalar 0.45370370554828643.
Singleton coverage/support code separately consumes these scalars (13559); do not silently redesign weighting or harmonize unrelated weight science.
Follow-up candidate: support-aware RGB variance weights and explicit weighting intent, separately approved. No weights.py changes in this mission.

## Baseline validation actually run
Command: QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_drizzle_background.py tests/test_hsi_closure_normalization.py tests/test_coverage_support_classic.py tests/test_rf2_production_impl.py tests/test_rf2_production_seam.py tests/test_resume.py
Result: 285 passed, 5 failed, 7 warnings in 19.93s; exit 1.
Log: /home/tristan/.openclaw/workspace/.a2a-reports/zsss-support-overlap-p1-20260907/baseline-targeted.log
Failures at untouched base:
- test_rf2_production_seam.py::test_target_policy_witness_integration
- test_resume.py::test_start_processing_valid_resume_pins_reference_and_queue
- test_resume.py::test_start_processing_quality_weighted_captures_q_ref_once_fresh
- test_resume.py::test_start_processing_quality_weighted_resume_skips_recomputation
- test_resume.py::test_repeated_start_binds_fresh_canonical_config
Trace includes stale reference fixtures (No SIMPLE card / frozen registration reference state is missing).
These are baseline evidence, not waived green gates. Full baseline suite still required; classify exact failures and any regression independently.

## Approved bounded Phase 1 design / acceptance
- Truthful geometry from actual Classic M; independent content validity; effective support; finite same-coordinate common samples.
- Neutral core overlap primitive reuses exact Drizzle robust-location math; lock Drizzle before extracting, preserve legacy imports. No competing copied robust estimator.
- sky_mean scalar paired luminance I-R robust location, correction I-offset, same scalar for RGB; no gain/surface.
- linear_fit same per-channel P25/P90 formula on common reliable positions.
- none strict normalization no-op; no batch=1 competing subtraction in plain Classic.
- missing/unreliable support/insufficient overlap => deterministic neutral correction + passive bounded provenance, never legacy statistics.
- Immutable reference; per-source correction invariant under permutation/batches; transient footprint lifetime, disk/tiled routes included.
- Synthetic rotations 0/22.5/45/70/100, translations +/-X,+/-Y, coverage near 95/80/60/50%, >25% unsupported, legitimate zero, NaN/Inf, +/-known offsets, mono/RGB, structured sky (paired positions). Explicit float tolerance required. Resume clean checkpoint equivalence and dirty rejection required.
- Logs fail-open. Maintain GPU log/provenance durability.
- Full targeted gates and complete suite evidence + Junior verification + Nono ACCEPT before W80 preparation/execution; no claiming baseline failures are green.
- Do not change reducer/Winsor/GPU/COV/registration/weights/Drizzle science. No Phase 2 implementation or Windows master mutation.
