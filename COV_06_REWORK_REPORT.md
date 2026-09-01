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
* 1080x1920: ~309 ms/exposure -> ~0.3s/1k, ~3.1s/10k, ~30.9s/100k (per this
  machine's EDT cost; a downsampled EDT is a future optimization if 100k
  full-res runs are required).
* 2160x3840: ~2.0s/exposure.

## Boring verification — SUPPORT_FULLY_TRACKED

`seestar/gui/boring_stack.py` instantiates the regular
`SeestarQueuedStacker` (line 817) and calls
`start_processing(use_drizzle=False, ...)` (lines 888/908), so Boring routes
through the classic SUM/W path that accumulates positive support in COV-01B.

## Resulting defaults (post REWORK)

* `apply_feathering` (legacy inverse-WHT): **OFF**
* `apply_coverage_render` (new confidence render): **OFF**
* `apply_batch_feathering` (footprint taper gate): unchanged (True)
* Scientific FITS never receives any cosmetic treatment.

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
