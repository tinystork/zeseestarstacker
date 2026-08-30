# Coverage-Aware Peripheral Reconstruction — Architecture Archaeology (COV-00)

<!-- project: path:/home/tristan/.openclaw/workspace/projects/zeseestarstacker -->

Gate: **COV-00** (architecture archaeology and characterization only — no code changes).
Repo: `/home/tristan/.openclaw/workspace/projects/zeseestarstacker`
Branch: `feature/coverage-aware-peripheral-reconstruction`
Baseline HEAD: `501eb9b5031a1ea81f09e6f01687a4cc349de879` (== origin/main at start)
Product baseline: 8.2.2 Phoenix consedit.

> Every claim below is an *observed* fact with file/function/line evidence, not an
> assumption. Anything not directly observed is labelled UNKNOWN / hypothesis and
> paired with a resolving test in §12.

---

## 1. Executive summary

1. The pipeline has **four finalization strategies** (one mode == one accumulation
   strategy), decided by `_decide_finalization_mode` (`queue_manager.py:1069`):
   `mosaic`, `drizzle`, `reproject_coadd`, `classic_sumw`. These are the
   *finalization* modes. The per-batch *reducer* (mean / median / kappa-sigma /
   winsorized-sigma / linear-fit-clip) is a separate axis selected by
   `stacking_mode` / `stack_reject_algo`.

2. **The single most important scientific-risk finding**: `feather_by_weight_map`
   (`stack_enhancement.py:378`) computes a WHT-derived **brightness gain map**
   `gain = blurred_wht / max(wht, floor)` clipped to **[0.5, 2.0]** and multiplies
   the image by it. Low-WHT regions therefore get a **gain > 1 (up to 2×)** — a direct
   contradiction of the frozen invariant "Low WHT is lower confidence, never a reason
   for brightness gain". It is currently applied **only** on the cosmetic path
   (preview + final-save cosmetics), NOT on the scientific FITS pixel data (§9), but it
   is a live hazard for any future "cosmetic reconstruction".

3. **Batch-level radial weighting** (`make_radial_weight_map`,
   `weight_utils.py:3`) is applied to *coverage/denominator* maps in multiple
   scientific accumulation paths (§6.4). Whether it cancels in the final SUM/W divide
   depends on the accumulation semantics; for single-grid classic it cancels per-pixel,
   for mosaics/reproject it does not. This is the second scientific-risk area.

4. The **scientific output is preserved separately** from cosmetic processing in the
   final save: FITS pixel data is written from `raw_adu_data_for_ui_histogram`
   (raw linear SUM/W divide or native drizzle `out_img`), while feather/low-WHT-mask
   cosmetics only mutate `data_after_postproc` (UI preview + PNG). This is the
   natural seam for COV-01 (§9, §11).

5. Drizzle already has a sound, signed-WHT-aware core (`drizzle_core.py`) with an
   explicit resume/checkpoint seam. The classic SUM/W path has its own memmap +
   manifest resume state. Both are mapped in §10.

---

## 2. Canonical mode keys and aliases

### 2.1 Backend stacking-mode keys (Qt shell)

`seestar/gui_qt/main_window.py:315-322`:

```python
STACKING_MODES = [
    "kappa-sigma",
    "classic",
    "mean",
    "median",
    "winsorized-sigma-clip",
    "linear-fit-clip",
]
```

### 2.2 Legacy Tk `stack_method` (reducer axis)

`seestar/gui/settings.py:1540-1544`: `"mean"`, `"median"`, `"kappa_sigma"`,
`"winsorized_sigma_clip"`, `"linear_fit_clip"`.

Derivation: `settings.py:284` and `settings.py:895` —

```python
if getattr(self, "stacking_mode", "") != "classic":
    self.stacking_mode = self.stack_method.replace("_", "-")
```

So `stacking_mode` is `stack_method` with `_` → `-`, **except** the literal
`"classic"` mode which is preserved verbatim.

### 2.3 Rejection-algo / final-combine keys

- `stack_reject_algo` (`settings.py:1600-1614`): `"none"`, `"kappa_sigma"`,
  `"winsorized_sigma_clip"`, `"linear_fit_clip"`.
- `stack_final_combine` (`settings.py:266-275`): `"mean"`, `"median"`,
  `"reproject"`, `"reproject_coadd"`; `reproject_between_batches =
  (== "reproject")`, `reproject_coadd_final = (== "reproject_coadd")`.

### 2.4 Aliases normalized at the reducer boundary

`queue_manager.py:1160-1205` (`_is_winsorized_mode`, `_is_linear_fit_clip_mode`):

- Winsorized aliases: `"winsorized-sigma"`, `"winsorized-sigma-clip"`,
  `"winsorized_sigma"`, `"winsorized_sigma_clip"`. Canonical scientific key is
  `"winsorized-sigma"`.
- Linear-fit-clip aliases: `"linear_fit_clip"`, `"linear-fit-clip"`. Canonical
  scientific key is `"linear_fit_clip"`.

### 2.5 Canonical run-contract field names

`seestar/run_contract.py:332-397` — `stacking_mode`,
`stack_kappa_low`/`stack_kappa_high`, `winsor_limits`, `normalize_method`,
`weighting_method`, `use_quality_weighting`, `apply_feathering`,
`apply_batch_feathering`, `feather_blur_px`, `apply_low_wht_mask`,
`low_wht_percentile`, `low_wht_soften_px`, `use_drizzle`, `drizzle_mode`,
etc. All feather/crop/low-WHT fields are `_FP_CLASSIC` fingerprint-domain.

---

## 3. Finalization-mode single source of truth

`queue_manager.py:353-356`:

```python
FINALIZATION_MODE_MOSAIC          = "mosaic"
FINALIZATION_MODE_DRIZZLE         = "drizzle"
FINALIZATION_MODE_REPROJECT_COADD = "reproject_coadd"
FINALIZATION_MODE_CLASSIC_SUMW    = "classic_sumw"
```

`_decide_finalization_mode` (`queue_manager.py:1069-1094`), in priority order:

1. `is_mosaic_run` → `mosaic`
2. `drizzle_active_session` → `drizzle`
3. `reproject_between_batches` → `classic_sumw` (reprojected batches summed into memmaps)
4. `reproject_coadd_final` → `reproject_coadd`
5. otherwise → `classic_sumw`

This mode is resolved once at accumulation init (`queue_manager.py:4075`) and passed
explicitly to `_save_final_stack` (never re-derived as a fallback).

---

## 4. Complete mode matrix

Columns: Mode (user-visible) | Canonical key | Scientific reducer | Coverage source |
WHT semantics | Current edge treatment | Resume state | Proposed treatment.

| Mode | Canonical key | Scientific reducer | Coverage source | WHT semantics | Current edge treatment | Resume state | Proposed treatment |
|---|---|---|---|---|---|---|---|
| Classic / default | `classic` (`stacking_mode`) + `classic_sumw` (finalization) | falls through to weighted arithmetic mean (reducer `else` branch, `queue_manager.py:12784-12811`) | `cumulative_wht_memmap` (SUM/W accumulator) | effective denominator W (per-pixel/per-channel), `max(wht,0)` on finalize | `_feather_batch_coverage` radial taper on batch coverage (`11896`); interbatch feather on weights (`3262`) | `memmap_accumulators/{cumulative_SUM.npy,cumulative_WHT.npy,resume_manifest.json}` (`13084-13775`) | Final-only cosmetic reconstruction; scientific SUM/W unchanged |
| Mean | `mean` | `_stack_mean` (`stack_methods.py:107`) | sum of per-image weights (broadcast scalar quality weight) | W = sum(valid w); `result = Σ(d·w)/Σw` | `_feather_batch_coverage` on batch denominator | classic SUM/W manifest | same as classic |
| Median | `median` | `_stack_median` (`stack_methods.py:131`) — NaN-excluded median; **not** redefined | count of valid samples (W = n_valid) | W = count(non-NaN) | `_feather_batch_coverage` (`queue_manager.py:12735`) | classic SUM/W manifest | same as classic |
| Kappa-sigma | `kappa-sigma` (backend) / `kappa_sigma` (reject_algo) | `_stack_kappa_sigma` (`stack_methods.py:148`) | W = sum of weights of accepted samples (or count) | W = Σw over in-range samples | `_feather_batch_coverage` (`12701`) | classic SUM/W manifest | same as classic |
| Winsorized sigma clip | `winsorized-sigma-clip` / `winsorized_sigma_clip` (aliases) | `_stack_winsorized_sigma_iter` (`stack_methods.py:262`) | W = sum of weights of contributing samples (rewinsor ⇒ rejected keep weight) | W = effective denominator matching `apply_rewinsor` | `_feather_batch_coverage` (`12664`) | classic SUM/W manifest | same as classic |
| Linear-fit clip | `linear-fit-clip` / `linear_fit_clip` | `_stack_linear_fit_clip` (`stack_methods.py:204`) | W = sum of weights of accepted samples | W = Σw over in-range residual samples | `_feather_batch_coverage` (`12767`) | classic SUM/W manifest | same as classic |
| Boring (CLI front-end) | (not a reducer) — calls `start_processing` with `stacking_mode="winsorized-sigma"`, `use_drizzle=False` (`boring_stack.py:891,908`) | winsorized-sigma (or final-combine override) | classic SUM/W memmaps | classic denominator | classic radial taper via batch feathering | classic manifest | same as classic |
| Reproject (between batches) | `reproject` (`stack_final_combine`) → `classic_sumw` | per-batch reducer (unchanged) on *reprojected* batches | `cumulative_wht_memmap` from `_reproject_worker` weight maps | reprojected footprint × radial (`_reproject_worker`, `1357`) | footprint mask + radial in worker; IBN (`2861`) | classic manifest | same |
| Reproject coadd (final) | `reproject_coadd` (`stack_final_combine`) → `reproject_coadd` | per-batch reducer + explicit `reproject_and_coadd` SCI/WHT | `reproject_and_coadd` coverage (`reproject_utils.py`) | footprint-derived coverage (scalar/array weights reprojected) | background matching (`match_background`) + radial on classic batch WHT (`16330`) | classic manifest + reproject SCI/WHT transient | same |
| Drizzle | `drizzle` (`use_drizzle=True`) | native drizzle weighted mean (`DrizzleAccumulator.finalize("divide")` → native `out_img`) | native `out_wht` (signed for Lanczos) | **signed** WHT; valid iff finite & `wht > WEIGHT_EPSILON` (`drizzle_core.py:59`) | `wht_relative_threshold` (`drizzle_core.py:715`) effective 0.0 for Lanczos; no gain | `.m3d_checkpoint/checkpoint.json` + `from_native_state` (`drizzle_core.py:291`, `drizzle_checkpoint.py`) | keep native signed WHT; cosmetics final-only |

Notes:
- "classic" is a **finalization** concept (`classic_sumw`), not a distinct reducer;
  it falls through the reducer dispatch to the weighted mean. UNKNOWN: whether
  "classic" intends any ccdproc-`combine` path distinct from `_stack_mean`
  (`ccdproc_combine` is imported at `queue_manager.py:239` but the batch reducer
  uses the ZeMosaic-style accumulators — see `queue_manager.py:12186`). Resolving
  test in §12.1.
- "reproject" and "reproject coadd" are distinct finalization modes (memmap SUM/W vs
  explicit `reproject_and_coadd`) but share the same per-batch reducers.

---

## 5. Reducer provenance contract

`stack_methods.py:1` "Stacking algorithms duplicated from ZeMosaic". The kernels
return `(result, W, rejected_pct)` under `return_weights=True` where
`result * W == numerator` for the linear family. This is the *defined* bounded-memory
hierarchical algorithm for the nonlinear family. `NaN` marks missing samples and is
never a numeric observation. The returned `W` is the effective per-pixel denominator
that downstream code then *feathers* (radial taper) — see §6.4.

---

## 6. Edge/coverage pipeline — call-path map

### 6.1 `feather_by_weight_map` (COSMETIC gain — hazard)

- Definition: `seestar/enhancement/stack_enhancement.py:378`.
- Imported into `queue_manager.py` at `1799` with a fail-open fallback stub
  (`1814`) guarded by `_FEATHERING_AVAILABLE`.
- Math: `wht_blurred = blur(wht)`; `gain = wht_blurred / max(wht, wht_min_for_gain)`;
  `gain_clipped = clip(gain, 0.5, 2.0)`; `gain_blurred = blur(gain_clipped)`;
  `out = img * gain_blurred`.
- Call sites (all cosmetic):
  - `_update_preview_sum_w` (`queue_manager.py:5013`) — preview only.
  - `_save_final_stack` (`queue_manager.py:17351`) — `data_after_postproc` only.
- **Risk**: gain up to 2× in low-WHT regions. See §8.1.

### 6.2 `apply_low_wht_mask` (COSMETIC median-fill)

- Definition: `seestar/enhancement/stack_enhancement.py:414`.
- Imported into `queue_manager.py` at `1880` with fallback stub `1891`.
- Math: adaptive threshold on `wht` (percentile, capped at median, floored at
  `min_threshold=1e-5`); binary mask → Gaussian soft mask (`soften_px`);
  fill = median color; `out = img·mask + fill·(1-mask)`; aborts if masked fraction
  > `max_mask_fraction=0.50` or resulting dynamic range < 0.05.
- Call sites (all cosmetic):
  - `_update_preview_sum_w` (`5028`).
  - `_save_final_stack` (`17370`).
- **Note**: this *attenuates* low-WHT regions toward a neutral median fill; it does
  **not** introduce a brightness gain. It is the "right shape" of edge treatment but is
  still cosmetic-only and median-fill (not a science-preserving reconstruction).

### 6.3 `_feather_batch_coverage` (SCIENTIFIC radial taper on denominator)

- Definition: `queue_manager.py:11896-11908`.
- Behavior: if `apply_batch_feathering` (default True), multiply the batch coverage
  map by `make_radial_weight_map(h, w)` (feather_fraction=0.92, floor=0.10).
- Call sites: `queue_manager.py:12664` (winsorized), `12701` (kappa), `12735`
  (median), `12767` (linear-fit), `12811` (mean `sum_weights`). Also the
  single-image mean fast path at `12419`.
- **Semantics**: applied to the returned `batch_coverage_map_2d` *after* the batch
  numerator has already been divided by the un-feathered denominator. This is a
  denominator-only reweighting (see §8.2).

### 6.4 `make_radial_weight_map` (radial falloff primitive)

- Definition: `seestar/enhancement/weight_utils.py:3`; fallback copy at
  `queue_manager.py:219`.
- Math: weight 1.0 for `r < feather_fraction`, linear ramp to `floor` at `r=1`.
  Defaults `feather_fraction=0.92`, `floor=0.10`.
- Call sites (scientific):
  - `_reproject_worker` (`1357`) — footprint/weight map × radial.
  - `_interbatch_apply_feather` (`3269`, feather_fraction=0.98) — interbatch weights.
  - mosaic panel footprints (`8792`).
  - `_feather_batch_coverage` (`11902`) and single-image path (`12419`).
  - classic batch WHT per-channel (`15411`) and reproject-coadd/batch-size-1 path (`16330`).

### 6.5 Interbatch normalization (photometric + radial)

- `_apply_interbatch_normalization` (`queue_manager.py:2861`): background 2D model
  subtraction (`estimate_background_2d`), then photometric gain/offset from overlap
  median ratio (`_interbatch_compute_scales`, `3286`), then
  `weights = _interbatch_apply_feather(weights)` (radial taper).
- The gain is `data = data * scale + offset` per channel (RGB) or scalar.
- Triggered on reprojected batches (`2544-2545`) and on classic final-combine
  mini-stacks (`_apply_final_combine_interbatch_normalization`, `3063`), for
  `stack_final_combine in {"mean", "winsorized_sigma_clip"}` (`_should_use_final_combine_ibn`, `3049`).
- **Risk**: multiplicative photometric gain is applied to *science* (batch pixel
  values) — this is a legit photometric normalization, distinct from the WHT-derived
  cosmetic gain in §6.1. The radial taper on weights biases which pixels define the
  overlap statistics.

### 6.6 Reproject input/output weights

- `seestar/core/reprojection.py` — `reproject_to_reference_wcs` (interp, per-channel),
  `resolve_all_wcs` (multiprocessing).
- `seestar/enhancement/reproject_utils.py` — `reproject_and_coadd` (astropy reference
  with local fallback), `reproject_and_coadd_from_paths`, `streaming_reproject_and_coadd`,
  `compute_final_output_grid` (`find_optimal_celestial_wcs`), background matching
  (`_estimate_background_corrections`, `solve_corrections_sgd`).
- Weight handling in the local coadd: `weight_proj = reproject(weight)·footprint`; the
  accumulator does `sum += img·weight; cov += weight` (coadd body in `reproject_utils.py`).
- `REPROJECT_FORCE_LOCAL=1` env forces the local accumulator (avoids astropy's internal
  normalization) — relevant for classic reproject of already-stacked batches.

### 6.7 Drizzle native WHT handling (signed Lanczos)

- `seestar/core/drizzle_core.py`: `WEIGHT_EPSILON = 1e-9` (`59`),
  `LANCZOS_KERNELS = {"lanczos2","lanczos3"}` (`69`), `DrizzleAccumulator` (`236`).
- drizzle 2.2.0 stores the *weighted mean* in `out_img` and the signed weight in
  `out_wht`; `finalize("divide")` returns native `out_img` gated by
  `finite & (wht > WEIGHT_EPSILON)`.
- `wht_relative_threshold` (`715`) is a *coverage/support policy* (fraction of a
  spatially-supported robust max), scale-invariant, effective **0.0 for Lanczos**.
- `support_integrity_violations` (`517`) fails closed if nonzero science sits on
  invalid native WHT support.
- Legacy `drizzle_utils.drizzle_finalize` (`drizzle_utils.py:8`) still assumes
  `sci = flux*weight` and divides by `wht` — **obsolete vs. the M3 core** (see §7).

### 6.8 Preview postprocessing

- `_update_preview_sum_w` (`queue_manager.py:4960-5160`): `avg = sum/wht` →
  optional `feather_by_weight_map` (`5013`) → optional `apply_low_wht_mask`
  (`5028`) → min/max 0-1 normalization → downsample. Captures an immutable
  `raw_linear_fullres` copy as display-analysis data (never science).

### 6.9 Final save postprocessing

- `_save_final_stack` (`queue_manager.py:16704`): resolve `final_image_initial_raw`
  per mode → WHT threshold (M3 relative for drizzle; legacy raw-absolute for others,
  `17212`) → `raw_adu_data_for_ui_histogram = nan_to_num(raw)` → percentile 0-1
  normalization (skipped when `preserve_linear_output`) → `data_after_postproc` =
  copy → optional feather (`17351`) + low-WHT-mask (`17370`) → FITS write from
  `raw_adu_data_for_ui_histogram` (`17595`, float32) or uint16 scaling of the same
  (`17610+`) → PNG from `data_after_postproc` (`17729`).

### 6.10 Boring routing

- `seestar/gui/boring_stack.py` is strictly the classic single-batch SUM/W memmap
  path (M3-D boundary, `boring_stack.py:85-98`): `use_drizzle=False`, no drizzle
  session, no per-group preview. `main() → _run_stack() → start_processing(
  stacking_mode="winsorized-sigma", use_drizzle=False, ...)` (`891-908`).
  Final-combine resolved by `_resolve_final_combine` (`692`), mapping
  `"reproject and coadd"` → `reproject_coadd`, etc.

---

## 7. Candidate path classification

- **ACTIVE_VALID**
  - `DrizzleAccumulator` + `drizzle_stream` + `wht_relative_threshold` +
    `support_integrity_violations` (`drizzle_core.py`) — the M3 standard drizzle
    finalization (`queue_manager.py:16882-16997`).
  - `_stack_mean/_median/_kappa_sigma/_linear_fit_clip/_winsorized_sigma`
    (`stack_methods.py`) — all live reducers.
  - `_feather_batch_coverage` + `make_radial_weight_map` — live batch coverage taper.
  - `_apply_interbatch_normalization` — live photometric/background normalization.
  - `reproject_and_coadd` family (`reproject_utils.py`) — live for mosaic and
    reproject_coadd finalization.
  - `feather_by_weight_map` / `apply_low_wht_mask` — live but **cosmetic-only**
    (preview + final-save cosmetics).
  - `_save_final_stack` — single finalization seam (all four modes).
  - `_load_classic_batch_wht` (`queue_manager.py:2091`) — live per-channel classic
    WHT sidecar loader (versioned `HSIVER==2`, `WHTSEM=="EFF_DENOM"`).

- **ACTIVE_OBSOLETE_REPLACED**
  - `drizzle_utils.drizzle_finalize` (`drizzle_utils.py:8`) — superseded by
    `DrizzleAccumulator.finalize`; still referenced by `livestack_mode.py:352` and a
    drizzle preview path (`queue_manager.py:10895-10902`). Its "divide" semantics
    (`sci/wht` where `sci=flux·wht`) no longer match the M3 native `out_img`
    (weighted-mean) convention — a latent trap if reused for finalization.

- **COMPATIBILITY_ONLY**
  - `DrizzleProcessor` / `DrizzleIntegrator` (`enhancement/drizzle_integration.py`)
    — legacy per-file drizzle; imported by `stack_enhancement` and `mosaic_processor`
    with fail-open stubs. Not on the M3 standard-drizzle finalization path.
  - `simple_stacker.create_master_tile` (`core/simple_stacker.py`) — documented
    "simplified local replacement", mean-only, ignores most params.
  - `reproject_to_reference_wcs` / `resolve_all_wcs` (`core/reprojection.py`) —
    thin wrappers still used for classic reproject batch finalization.
  - `StackEnhancer` (`enhancement/stack_enhancement.py`) — normalization/CLAHE/edge
    crop; `_normalize_images_*` duplicated from `core/normalization.py`.
  - `streaming_stack.stack_disk_streaming` / `streaming_reproject_and_coadd` — disk
    streaming variants.

- **DEAD** (no live caller on the scientific path; retained for lifecycle only)
  - `_wait_drizzle_processes` (`boring_stack.py:96` documents it as a "M3-D legacy
    no-op", historical submitter marked OBSOLETE LEGACY and never called).

- **UNKNOWN**
  - Whether `ccdproc_combine` (`queue_manager.py:239`) is used by any *live*
    reducer, or is dead import surface. (grep finds only `5340`
    `reproject_and_combine` — needs §12.2.)
  - Whether `livestack_mode.py` (`LiveStackController`, `queuep/livestack_mode.py`)
    is a user-reachable mode or a standalone prototype (docstring says "drop it under
    `seestar/modes/`"). No import found from `queue_manager`.

---

## 8. Scientific-risk findings

### 8.1 Post-normalization gain derived from WHT (CRITICAL)

`feather_by_weight_map` (`stack_enhancement.py:378-411`) multiplies the image by a
WHT-ratio gain map clipped to **[0.5, 2.0]**:

```python
gain_map = wht_blurred / np.maximum(wht_f32, wht_min_for_gain)
gain_map_clipped = np.clip(gain_map, min_gain, max_gain)   # [0.5, 2.0]
feathered_image = img_f32 * gain_map_blurred[..., None]
```

Where `wht` is locally low but the blurred neighbourhood is higher, `gain > 1` —
i.e. **low-coverage regions are brightened up to 2×**. This directly violates the frozen
invariant "Low WHT is lower confidence, never a reason for brightness gain". It is
currently cosmetic-only (never written to the scientific FITS pixels), but any COV
reconstruction that reuses this function as-is would inherit the violation.

### 8.2 Batch-level radial weighting of the denominator (HIGH)

`_feather_batch_coverage` multiplies the returned batch denominator
`batch_coverage_map_2d` by a radial map **after** the batch numerator was already
divided by the un-feathered denominator. Consequences:

- Single-grid classic (all batches share the same output shape): the radial factor is
  identical per-pixel across batches, so it cancels in the global
  `Σ(batch·W)/Σ(W)` divide — **no net bias**, but the stored coverage map itself is
  radially attenuated (affects WHT threshold / display / low-WHT mask).
- Mosaic / reproject (different batches cover different regions): the radial factor
  does **not** cancel; edge pixels are down-weighted by each contributing panel's own
  radial falloff — a real, geometry-dependent edge bias.
- The interbatch feather (`_interbatch_apply_feather`, `3262`, fraction 0.98)
  additionally biases which pixels define the overlap scale/offset statistics.

Resolving tests: §12.3, §12.4.

### 8.3 Drizzle signed WHT is correctly preserved (LOW / positive)

`DrizzleAccumulator` keeps native signed `out_wht` (`wht` property returns the
un-clipped copy), validity is `finite & (wht > WEIGHT_EPSILON)`, and Lanczos
thresholds are forced to 0.0. No `abs(wht)`, no huge-value clip, no percentile hiding.
This aligns with the frozen invariants. The only gap is the legacy
`drizzle_utils.drizzle_finalize` semantics mismatch (§7).

### 8.4 Median is not silently redefined (LOW / positive)

`_stack_median` (`stack_methods.py:131`) uses `np.nanmedian` with NaN excluded and
returns W = valid count — consistent with the invariant "Median must not be silently
redefined".

---

## 9. Scientific output vs preview / render / final-save boundaries

| Boundary | Data | Notes |
|---|---|---|
| Scientific accumulator | `cumulative_sum_memmap` / `cumulative_wht_memmap` (classic) or `DrizzleAccumulator._out_img/_out_wht` (drizzle) | raw SUM/W or native drizzle; no cosmetics |
| Scientific final pixels | `final_image_initial_raw` → `raw_adu_data_for_ui_histogram` | WHT threshold NaN→0 applied; **this** is what FITS writes |
| FITS primary HDU | `raw_adu_data_for_ui_histogram` (float32 raw ADU, or uint16-scaled) | `queue_manager.py:17595,17610+`; feather/low-WHT-mask **not** applied |
| FITS companion WHT | `final_wht_hwc` (native signed, drizzle only) | `_write_companion_wht_fits`, only when `save_drizzle_wht` |
| UI preview (live) | `preview_data_normalized` (+ `raw_linear_fullres` second element) | `_update_preview_sum_w`; feather+low-WHT-mask+minmax applied |
| UI preview (final) | `last_saved_data_for_preview = data_after_postproc` | cosmetic 0-1, feather+low-WHT-mask applied |
| PNG | `data_after_postproc` | `_save_final_preview_png` |

**Key seam**: the scientific FITS pixels and the cosmetic preview/PNG diverge exactly at
`data_after_postproc`. COV-01's "cosmetic reconstruction must be final-only and
separate from scientific output" can be implemented by adding a new post-processing
stage *after* `raw_adu_data_for_ui_histogram` is captured, feeding only
`data_after_postproc` / preview / PNG — never the FITS pixels.

---

## 10. Resume / checkpoint state (for SUP_W1 / SUP_W2)

### 10.1 Classic SUM/W

- `memmap_accumulators/cumulative_SUM.npy` + `cumulative_WHT.npy` +
  `resume_manifest.json` (`queue_manager.py:655,13084-13775`).
- `_RESUME_MANIFEST_FILENAME = "resume_manifest.json"`; schema-versioned; atomic
  temp + `os.replace` commit; `_RESUME_STATE_CLEAN` on success.

### 10.2 Drizzle (M3)

- `<output>/.m3d_checkpoint/checkpoint.json` (single commit point, `drizzle_checkpoint.py:184-186`).
- Writer is **write-only**; reader `read_drizzle_checkpoint` validates the *entire*
  checkpoint before `DrizzleAccumulator.from_native_state` (`drizzle_core.py:291`)
  rebuilds around the exact native `out_img`/`out_wht` (bit-identical continuation).
- `total_exptime` must be restored alongside the buffers (upstream drizzle rejects
  `exptime==0` with non-empty weights).

### 10.3 Drizzle background anchor

- `drizzle_background.py`: `BackgroundAnchor` (`362`) — immutable anchor state
  (`anchor_data`, `tf`, `reference_shape_hw`), `to_metadata/from_metadata`
  scalar-contract serialization (not a full checkpoint), `estimate_background_offsets`
  (`551`) / `apply_background_offsets` (`740`).

---

## 11. Concrete seams for COV-01 (identified, not implemented)

1. **`_save_final_stack` post-processing stage** (`queue_manager.py:17332-17390`):
   the single place where cosmetic edge treatment (`feather_by_weight_map`,
   `apply_low_wht_mask`) is applied after `raw_adu_data_for_ui_histogram` is
   captured. COV-01 can add a *new* final-only reconstruction function here, fed by
   `final_wht_map_for_postproc`, and route it only into `data_after_postproc`.
2. **`feather_by_weight_map`**: either replace with a non-gaining reconstruction
   (attenuation-only) or add a `max_gain=1.0` mode for COV. It is the prototype of
   the exact behaviour the frozen invariants forbid.
3. **`_feather_batch_coverage`** (`11896`): the single choke-point for batch-level
   radial denominator taper; COV-01 can gate the scientific taper independently from
   any final-only reconstruction.
4. **`apply_low_wht_mask`**: the existing attenuation+median-fill primitive is the
   closest existing analogue to a "cosmetic reconstruction"; COV-01 can generalize its
   fill policy (median → a reconstructed edge value).
5. **`make_radial_weight_map`** (`weight_utils.py:3`): the shared radial primitive;
   any COV change to edge weights must be audited against all six call sites in §6.4.
6. **Preview tuple contract** (`_update_preview_sum_w` `5152`, and the Qt
   `raw_linear` second element `20465-20548`): already carries a raw-linear element
   separate from the displayed image — reuse it to prove reconstruction never touches
   science.

---

## 12. Uncertainties and resolving tests

1. **"classic" reducer identity**: does `classic` intend a ccdproc-`combine` path
   distinct from `_stack_mean`? — Test: run classic vs mean on identical inputs with
   identical settings; compare batch `STK_NOTE` and pixels. (Evidence gap:
   `ccdproc_combine` import at `239` but no live reducer call found.)
2. **`ccdproc_combine` liveness**: grep only finds `5340` (`reproject_and_combine`).
   — Test: trace callers of `reproject_and_combine`; confirm it is/isn't on any
   user-facing path.
3. **Radial cancellation in single-grid classic**: confirm radial feather cancels in the
   final divide (hypothesis) vs. introduces a bias. — Test: two runs, identical images,
   `apply_batch_feathering=True/False`; diff final FITS pixels.
4. **Mosaic/reproject edge bias**: quantify edge down-weighting from per-panel radial
   falloff. — Test: synthetic two-panel mosaic with known overlap flux; compare edge vs
   centre photometry.
5. **`feather_by_weight_map` gain magnitude**: measure real gain range on a low-WHT
   edge. — Test: instrument a run; log `gain_map_clipped` min/max.
6. **`livestack_mode` reachability**: is it wired into any UI/CLI? — Test: grep
   imports of `livestack_mode` / `LiveStackController` across `seestar` and
   `main.py`/`qt_main.py`.

---

## 13. Frozen-invariant conformance (observed)

| Invariant | Observed status |
|---|---|
| SCI conceptually SUM/WHT | Holds (classic `cumulative_sum/wht`; drizzle native `out_img/out_wht`) |
| Low WHT never a brightness-gain reason | **VIOLATED by `feather_by_weight_map` gain∈[0.5,2.0]** (cosmetic-only today) |
| Scientific WHT ≠ positive support confidence | Holds for drizzle (signed, `WEIGHT_EPSILON`); classic WHT is `max(wht,0)` on finalize |
| Native drizzle Lanczos WHT may be signed, stays signed | Holds (`DrizzleAccumulator.wht` returns un-clipped; threshold forced 0.0 for Lanczos) |
| Median not silently redefined | Holds (`_stack_median` = nanmedian) |
| Reproject APPROXIMATE BY DESIGN | Holds (`reproject_interp` interp; no proof of exactness claimed) |
| Cosmetic reconstruction final-only & separate | Partially: cosmetics already separate from FITS pixels; "reconstruction" not yet implemented |
| All existing modes remain | Holds (no mode deleted; dispatch chain intact) |

---

*Prepared by Coco (COV-00). Evidence from source inspection of
`seestar/queuep/queue_manager.py`, `seestar/core/*`, `seestar/enhancement/*`,
`seestar/gui/*`, `seestar/gui_qt/*` at HEAD `501eb9b`. No production code changed.*
