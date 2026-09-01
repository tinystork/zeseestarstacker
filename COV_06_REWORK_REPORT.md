# COV-06 REWORK REPORT

<!-- project: path:/home/tristan/.openclaw/workspace/projects/zeseestarstacker -->

Closure pass over the coverage-aware peripheral reconstruction feature.

## Provenance

* Starting HEAD: `3a3d825c8d3779216f18c4e3d43d752048b35c9f`
* Ending HEAD: `d66d7d9218edf6b0ab9319558ed95ba5aca3cede`
* Branch: `feature/coverage-aware-peripheral-reconstruction`
* Implemented by: Junior (took over after Coco delivery failure x2)
* Baseline: `501eb9b5031a1ea81f09e6f01687a4cc349de879`

## Blocker A — footprint taper EDT boundary bug — FIXED

`make_footprint_taper` now pads the mask with invalid (`False`) support before
the Euclidean distance transform and crops it back, so a footprint that fills
the array feathers symmetrically from all four boundaries.

Before (all-valid 32x32, feather_px=5): TL=0.2 TR=1.0 BL=1.0 BR=1.0 (asymmetric).
After: TL=TR=BL=BR=0.2, centre=1.0, top==bottom, left==right (symmetric).
The pure-NumPy chamfer fallback uses the same padded convention.

Tests added: all-valid rect (4 corners/4 edges symmetric), mask touching one /
several boundaries, internal invalid island, fallback convention parity.

## Blocker B — legacy inverse-WHT render feather — RETIRED FROM DEFAULT

`apply_feathering` default changed `True -> False` in the Tk settings
(`seestar/gui/settings.py`) and the Qt settings state
(`seestar/gui_qt/settings_state.py`).  `feather_by_weight_map` (WHT-derived
gain [0.5, 2.0]) is marked DEPRECATED and no longer the default edge treatment.
Both legacy feather and new coverage-aware render are now OFF by default until
the 1602-image Windows witness.

## Blocker C — mean singleton vs multi-image inconsistency — FIXED

The singleton (`batch_size=1`, mean) path now applies the SAME footprint-aware
taper (`self._footprint_taper_map`) to its coverage map instead of the
historical radial weight, so a singleton batch and the same exposure inside a
multi-image batch share identical geometric support semantics.

Witness added: `test_singleton_vs_multi_decomposition_with_taper` asserts
support (SUP_W1/SUP_W2) and scientific (SUM/WHT) equality between one batch of
N and N singleton batches with feathering enabled.

## Condition D — reliable overlap: WHT-fraction, documented deviation

COV-03's `_ibn_reliable_fraction` thresholds the batch coverage weight (a
relative fraction of peak weight).  This is NOT equivalent to positive support
confidence for nonlinear/Drizzle modes and is documented as such (see
`docs/coverage_confidence_contract.md` COV-03 section).  No false semantic
equivalence is claimed; the estimate uses the batch coverage that the IBN layer
already carries, not SUP_W1/SUP_W2.

## Condition E — Drizzle reaches the render path — FIXED

`_derive_neff_support_for_render` now falls back to the Drizzle positive
support pair (`_drizzle_support_n_eff`, from `drizzle_sup_w1`/`drizzle_sup_w2`)
when the Classic support memmaps are absent.  It never reads the signed native
Lanczos WHT.

## Condition F — duplicate per-frame EDT — REDUCED

`_footprint_taper_map` now caches by mask identity within one batch, and the
mean `weigh_one` reuses the raw boolean mask (no redundant `astype(bool)` copy),
so the scientific weighting and support accumulation share one distance
transform per exposure.  Cache cleared at the start of each `_stack_batch`.

Benchmark (scipy EDT + pad, this machine):
* 1080x1920: ~309 ms/exposure -> ~309 s / 1k (~5.15 min), ~3,090 s / 10k
  (~51.5 min), ~30,900 s / 100k (~8.58 h).  COV-06B corrects the previous
  report's erroneous x1000 scaling; the real cost is intentionally not hidden.
* 2160x3840: ~2.0s/exposure.

No new EDT optimization is part of COV-06B.  A future candidate is to compute
the taper once in source space and transport it with the already-established
registration geometry, subject to an independent scientific-equivalence gate.

## Boring verification — SUPPORT_FULLY_TRACKED

`seestar/gui/boring_stack.py` instantiates the regular
`SeestarQueuedStacker` (line 817) and calls
`start_processing(use_drizzle=False, ...)` (lines 888/908), so Boring routes
through the classic SUM/W path that accumulates positive support in COV-01B.

## Resulting defaults (post REWORK)

* `apply_feathering` (legacy inverse-WHT): **OFF**
* `apply_coverage_render` (new confidence render): **OFF**
* `apply_batch_feathering` (coverage-support taper gate): unchanged (True).
  It gates the COV footprint taper in mean/support paths while retaining the
  established radial behavior in other reducers; COV-06B changes only the GUI
  terminology, not those reducer-specific mathematics.
* Scientific FITS never receives any cosmetic treatment.

## Real Linux witness (evidence, not renderer validation)

A real `winsorized-sigma-clip` run completed successfully with 160 input
entries in 54 batches (`RUN_SUCCEEDED`), writing a float32 scientific FITS and
a preview PNG.  The observed final WHT range was approximately 0.60 to 158.95,
which demonstrates a substantial centre/periphery coverage disparity and the
original product problem: central noise improves much faster than poorly
covered peripheral noise.

The Expert GUI still showed legacy `Feathering=ON`, `Batch feathering=ON`, and
`Low-weight mask=OFF` for that run.  It therefore proves only that the COV
branch did not break this real 160-entry winsorized run.  It does **not** prove
the efficacy of the new coverage-aware final reconstruction.

Many input names used the `image.fit` / `image_dup_....fit` pattern.  The 160
entries are consequently acceptable as a smoke/stress witness but must not be
treated as 160 statistically independent noise realizations.
`N_eff_support` remains the **support effective exposure count**, never a claim
of true independent-sample N_eff for a duplicated dataset.

## Tests run (this machine)

* Coverage + HSI closure suite: **266 passed, 0 failed**
* test_resume / test_save_final_stack / test_boring_drizzle_boundary /
  test_qt_settings_surface: 208 passed, 6 failed — all 6 independently
  confirmed pre-existing at baseline `501eb9b` (4 resume frozen-reference,
  1 save-final radec, 1 boring shape assert).
* `git diff --check` clean.

## Files changed

* `seestar/enhancement/weight_utils.py` (EDT pad + fallback)
* `seestar/enhancement/stack_enhancement.py` (feather deprecation)
* `seestar/gui/settings.py`, `seestar/gui_qt/settings_state.py` (defaults)
* `seestar/queuep/queue_manager.py` (singleton taper, render Drizzle fallback,
  taper cache)
* `tests/test_coverage_taper.py`, `tests/test_coverage_support_classic.py`
  (new witnesses)
