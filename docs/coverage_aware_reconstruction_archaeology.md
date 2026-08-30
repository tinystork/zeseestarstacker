# Coverage-Aware Peripheral Reconstruction — Architecture Archaeology (COV-00)

<!-- project: path:/home/tristan/.openclaw/workspace/projects/zeseestarstacker -->

Gate: **COV-00** (architecture archaeology and characterization only — no code changes).
Repo: `/home/tristan/.openclaw/workspace/projects/zeseestarstacker`
Branch: `feature/coverage-aware-peripheral-reconstruction`
Baseline HEAD: `501eb9b5031a1ea81f09e6f01687a4cc349de879` (== origin/main at start)
Product baseline: 8.2.2 Phoenix consedit.

> Every claim below is an *observed* fact with file/function/line evidence, not an
> assumption. Anything not directly observed is labelled UNKNOWN / hypothesis and paired
> with a resolving test in §12. This revision (R1) corrects §1, §4, §6, §7, §9, §10,
> §11, §13 per Junior's rejection notes.

---

## 1. Executive summary

1. The pipeline has **four finalization strategies** (one mode == one accumulation
   strategy), decided by `_decide_finalization_mode` (`queue_manager.py:1069`):
   `mosaic`, `drizzle`, `reproject_coadd`, `classic_sumw`. These are the
   *finalization* modes. The per-batch *reducer* (mean / median / kappa-sigma /
   winsorized-sigma / linear-fit-clip) is a separate axis selected by
   `stacking_mode` / `stack_reject_algo`.

2. **No distinct per-original-exposure positive support accumulator exists at
   baseline.** `rg "SUP_W1|SUP_W2|positive support|support confidence"` over
   `seestar/**` returns nothing. What exists today is (a) the classic effective
   denominator WHT (an *estimator* denominator, not a support-confidence domain),
   (b) the Drizzle native *signed* WHT, and (c) a *derived positive display/postprocess
   proxy* built from the positive portion of the signed Drizzle WHT
   (`final_wht_map_for_postproc += max(native_wht, 0)`,
   `queue_manager.py:16896-16926`). None of these is SUP_W1/SUP_W2 or a
   per-original-exposure `N_eff_support`. See §8.2.

3. **Scientific-risk finding (independently verified, retained)**: `feather_by_weight_map`
   (`stack_enhancement.py:378`) computes a WHT-derived **brightness gain map**
   `gain = blurred_wht / max(wht, floor)` clipped to **[0.5, 2.0]** and multiplies the
   image by it — low-WHT regions are brightened up to 2×. This violates the frozen
   invariant "Low WHT is lower confidence, never a reason for brightness gain". It is
   currently applied **only** on the cosmetic path (preview + final-save cosmetics),
   NOT on the scientific FITS pixel data (§9).

4. **Batch-level radial weighting** (`make_radial_weight_map`, `weight_utils.py:3`)
   is applied to *coverage/denominator* maps at six scientific sites (§6.4). It is
   ACTIVE today; its cancellation in the final SUM/W divide is geometry-dependent
   (§8.3) and it is only a COV-02 *replacement candidate*, not yet proven replaceable.

5. The **scientific output is preserved separately** from cosmetic processing in the
   final save (§9). One stale source docstring (`queue_manager.py:16718`) claims the
   uint16 FITS save uses "cosmetic [0,1] data"; the actual code (`17597+`) scales
   `raw_adu_data_for_ui_histogram` — recorded as debt in §9.2.

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

Derivation (`settings.py:284`, `settings.py:895`):

```python
if getattr(self, "stacking_mode", "") != "classic":
    self.stacking_mode = self.stack_method.replace("_", "-")
```

### 2.3 Rejection-algo / final-combine keys

- `stack_reject_algo` (`settings.py:1600-1614`): `"none"`, `"kappa_sigma"`,
  `"winsorized_sigma_clip"`, `"linear_fit_clip"`.
- `stack_final_combine` (`settings.py:266-275`): `"mean"`, `"median"`,
  `"reproject"`, `"reproject_coadd"`; `reproject_between_batches=(=="reproject")`,
  `reproject_coadd_final=(=="reproject_coadd")`.

### 2.4 Aliases normalized at the reducer boundary

`queue_manager.py:1160-1205` (`_is_winsorized_mode`, `_is_linear_fit_clip_mode`):

- Winsorized aliases: `"winsorized-sigma"`, `"winsorized-sigma-clip"`,
  `"winsorized_sigma"`, `"winsorized_sigma_clip"`. Canonical scientific key
  `"winsorized-sigma"`.
- Linear-fit-clip aliases: `"linear_fit_clip"`, `"linear-fit-clip"`. Canonical
  scientific key `"linear_fit_clip"`.

### 2.5 Canonical run-contract field names

`seestar/run_contract.py:332-397` — `stacking_mode`, `stack_kappa_low/high`,
`winsor_limits`, `normalize_method`, `weighting_method`, `use_quality_weighting`,
`apply_feathering`, `apply_batch_feathering`, `feather_blur_px`,
`apply_low_wht_mask`, `low_wht_percentile`, `low_wht_soften_px`, `use_drizzle`,
`drizzle_mode`, etc. All feather/crop/low-WHT fields are `_FP_CLASSIC` domain.

---

## 3. Finalization-mode single source of truth

`queue_manager.py:353-356`:

```python
FINALIZATION_MODE_MOSAIC          = "mosaic"
FINALIZATION_MODE_DRIZZLE         = "drizzle"
FINALIZATION_MODE_REPROJECT_COADD = "reproject_coadd"
FINALIZATION_MODE_CLASSIC_SUMW    = "classic_sumw"
```

`_decide_finalization_mode` (`queue_manager.py:1069-1094`), priority order:

1. `is_mosaic_run` → `mosaic`
2. `drizzle_active_session` → `drizzle`
3. `reproject_between_batches` → `classic_sumw` (reprojected batches → memmaps)
4. `reproject_coadd_final` → `reproject_coadd`
5. otherwise → `classic_sumw`

Resolved once at accumulation init (`queue_manager.py:4075`), passed explicitly to
`_save_final_stack` (never re-derived as a fallback).

---

## 4. Complete mode matrix

Columns: Mode | Canonical key | Scientific reducer | Coverage source | WHT semantics |
Current edge treatment | Resume state | Proposed support treatment (COV-01..05).

| Mode | Canonical key | Scientific reducer | Coverage source | WHT semantics | Current edge treatment | Resume state | Proposed support treatment |
|---|---|---|---|---|---|---|---|
| Classic / default | `classic` + `classic_sumw` | weighted arithmetic mean (reducer `else` branch, `queue_manager.py:12784-12811`) | `cumulative_wht_memmap` | effective denominator W; `max(wht,0)` on finalize | `_feather_batch_coverage` radial taper (`11896`); interbatch feather (`3262`) | `memmap_accumulators/{SUM,WHT,resume_manifest.json}` (v2) | COV-01: per-original-exposure positive support s_i (SUP_W1+=s_i, SUP_W2+=s_i²) before mini-stack reduction |
| Mean | `mean` | `_stack_mean` (`stack_methods.py:107`) | per-image weight sum | `Σ(d·w)/Σw` | `_feather_batch_coverage` | classic manifest | same as classic (per-exposure support accumulation) |
| Median | `median` | `_stack_median` (`stack_methods.py:131`) — unchanged | valid-sample count | W = count(non-NaN) | `_feather_batch_coverage` (`12735`) | classic manifest | median science **unchanged**; per-exposure s_i support independent of the estimator denominator |
| Kappa-sigma | `kappa-sigma`/`kappa_sigma` | `_stack_kappa_sigma` (`stack_methods.py:148`) | accepted-sample weight sum | `Σw` over in-range | `_feather_batch_coverage` (`12701`) | classic manifest | original geometric/quality support via s_i (independent of rejection survivors), NOT exact estimator N_eff |
| Winsorized sigma clip | `winsorized-sigma-clip`/`winsorized_sigma_clip` | `_stack_winsorized_sigma_iter` (`stack_methods.py:262`) | contributing-sample weight sum | W matches `apply_rewinsor` | `_feather_batch_coverage` (`12664`) | classic manifest | original geometric/quality support via s_i (independent of rejection survivors), NOT exact estimator N_eff |
| Linear-fit clip | `linear-fit-clip`/`linear_fit_clip` | `_stack_linear_fit_clip` (`stack_methods.py:204`) | accepted-sample weight sum | `Σw` over in-range residual | `_feather_batch_coverage` (`12767`) | classic manifest | original geometric/quality support via s_i (independent of rejection survivors), NOT exact estimator N_eff |
| Boring (CLI) | not a reducer — `stacking_mode="winsorized-sigma"`, `use_drizzle=False` (`boring_stack.py:891,908`) | winsorized-sigma (or final-combine override) | classic SUM/W memmaps | classic denominator | classic radial taper | classic manifest | routes through the same backend support contract (no separate boring path) |
| Reproject (between batches) | `reproject` → `classic_sumw` | per-batch reducer on reprojected batches | `cumulative_wht_memmap` via `_reproject_worker` | reprojected footprint × radial (`1357`) | footprint mask + radial; IBN (`2861`) | classic manifest | support evaluated/transformed per original exposure onto the output grid, then SUP_W1+=R(s_i), SUP_W2+=R(s_i)²; never use mini-stack count as N_eff |
| Reproject coadd (final) | `reproject_coadd` → `reproject_coadd` | per-batch reducer + `reproject_and_coadd`/incremental master | `initialize_master`/`reproject_and_combine` (`incremental_reprojection.py`) + `reproject_and_coadd` | `R(V)*R(W)` separate transport (approx) | background matching; radial on classic WHT (`16330`) | classic manifest + transient master_sum/cov | per-exposure transformed support accumulation (SUP_W1+=R(s_i), SUP_W2+=R(s_i)²); R(V)·R(W) stays APPROXIMATE BY DESIGN |
| Drizzle | `drizzle` (`use_drizzle=True`) | native drizzle weighted mean (`finalize("divide")` → native `out_img`) | native `out_wht` (signed for Lanczos) | **signed** WHT; valid iff finite & `wht>WEIGHT_EPSILON` (`drizzle_core.py:59`) | `wht_relative_threshold` (`715`), 0.0 for Lanczos; no gain | `.m3d_checkpoint/checkpoint.json` + `from_native_state` | keep native signed WHT; accumulate **separate** positive support (s_i, s_i² per original exposure) |

Notes:
- "classic" is a *finalization* concept (`classic_sumw`), not a distinct reducer; it
  falls through the reducer dispatch to the weighted mean. UNKNOWN: whether "classic"
  intends a ccdproc-`combine` path distinct from `_stack_mean` (resolving test §12.1).
- "reproject" and "reproject coadd" are distinct finalization modes but share per-batch reducers.
- Proposed-support column reflects the COV gate sequence (§11); it does **not** claim any
  exact estimator N_eff for nonlinear reducers, and does **not** bless the current radial
  weighting.

---

## 5. Reducer provenance contract

`stack_methods.py:1` "Stacking algorithms duplicated from ZeMosaic". Kernels return
`(result, W, rejected_pct)` under `return_weights=True` where `result*W == numerator`
for the linear family; for the nonlinear family W is the *defined* bounded-memory
hierarchical denominator. `NaN` marks missing samples (never a numeric observation).
The returned W is an **estimator denominator** — it is NOT a positive-support-confidence
domain and must not be promoted to one (see §8.2).

---

## 6. Edge/coverage pipeline — call-path map

### 6.1 `feather_by_weight_map` (COSMETIC gain — hazard)

- Definition: `stack_enhancement.py:378`; imported `queue_manager.py:1799` (fallback `1814`).
- Math: `gain = blur(wht)/max(wht, floor)`, clipped **[0.5, 2.0]**, blurred, then
  `out = img * gain`. Low-WHT ⇒ gain > 1 (up to 2×).
- Call sites (all cosmetic): `_update_preview_sum_w` (`5013`); `_save_final_stack`
  (`17351`, `data_after_postproc` only).
- See §8.1.

### 6.2 `apply_low_wht_mask` (COSMETIC median-fill)

- Definition: `stack_enhancement.py:414`; imported `queue_manager.py:1880` (fallback `1891`).
- Math: adaptive WHT threshold → soft mask → median-color fill `img·mask + fill·(1-mask)`.
  Aborts if masked fraction > 0.50 or dynamic range < 0.05. Attenuates (no gain); median-fill,
  not a science-preserving reconstruction.
- Call sites (all cosmetic): `5028`, `17370`.

### 6.3 `_feather_batch_coverage` (SCIENTIFIC radial taper on denominator — ACTIVE, candidate for COV-02 replacement)

- Definition: `queue_manager.py:11896-11908`. If `apply_batch_feathering` (default True),
  multiply batch coverage by `make_radial_weight_map(h,w)` (0.92/0.10).
- Call sites: `12664`, `12701`, `12735`, `12767`, `12811`, single-image `12419`.
- Applied to the denominator **after** the batch numerator was already divided by the
  un-feathered denominator → denominator-only reweighting (§8.3).

### 6.4 `make_radial_weight_map` (radial falloff primitive)

- Definition: `weight_utils.py:3`; fallback `queue_manager.py:219`. Weight 1.0 for
  `r<fraction`, ramp to `floor` at `r=1`.
- Scientific call sites: `_reproject_worker` (`1357`), `_interbatch_apply_feather`
  (`3269`, 0.98), mosaic footprints (`8792`), `_feather_batch_coverage` (`11902`) +
  single-image (`12419`), classic batch WHT (`15411`), reproject-coadd/BS=1 (`16330`).

### 6.5 Interbatch normalization (photometric + radial)

- `_apply_interbatch_normalization` (`2861`): background 2D subtraction
  (`estimate_background_2d`) → overlap-median gain/offset (`_interbatch_compute_scales`
  `3286`) → `weights = _interbatch_apply_feather(weights)` (radial).
- `data = data*scale + offset` (per channel or scalar). Triggered on reprojected batches
  (`2544`) and classic final-combine mini-stacks (`3063`) for
  `stack_final_combine in {"mean","winsorized_sigma_clip"}` (`3049`).
- This multiplicative gain is a legitimate *photometric* normalization, distinct from the
  WHT-derived cosmetic gain in §6.1. The radial taper on weights biases the overlap statistics.

### 6.6 Reproject input/output weights

- `core/reprojection.py`: `reproject_to_reference_wcs` (interp, per-channel), `resolve_all_wcs`.
- `enhancement/reproject_utils.py`: `reproject_and_coadd` (astropy reference + local
  fallback), `reproject_and_coadd_from_paths`, `streaming_reproject_and_coadd`,
  `compute_final_output_grid`, background matching (`_estimate_background_corrections`).
- Local coadd: `weight_proj = R(W)·footprint`; `sum += R(V)·weight_proj`;
  `cov += weight_proj`. `REPROJECT_FORCE_LOCAL=1` forces the local accumulator.

### 6.7 `incremental_reprojection.py` (reproject-coadd SCI/WHT master)

- Module: `seestar/core/incremental_reprojection.py` — three functions, all imported at
  `queue_manager.py:292-295`:
  - `initialize_master(batch_img, batch_cov, batch_wcs, ref_wcs)` (`~17`): reprojects the
    first batch's image **and** coverage separately via `reproject_to_reference_wcs`
    (`reproject_interp`), then returns `master_sum = R(img)·R(cov)` and
    `master_cov = R(cov)`.
  - `reproject_and_combine(master_sum, master_cov, ...)` (`~88`): accumulates
    `master_sum += R(img)·R(cov)`, `master_cov += R(cov)`.
  - `reproject_and_coadd_batch(...)` (`~148`): delegates to `reproject_and_coadd`.
- Live call sites: `initialize_master` at `queue_manager.py:5336`; `reproject_and_combine`
  at `5340`, in the `reproject_coadd_final` accumulation path (guarded by
  `reproject_coadd_final` at `5330-5334`).
- **WHT/coverage transport semantics**: V (batch image) and W (batch coverage) are
  reprojected **separately** then multiplied — `R(V)·R(W)` — not `R(V·W)`. This is the
  documented approximation: for linear R, `R(V)·R(W) − R(V·W) = −Σ aₖaₗ(Vₖ−Vₗ)(Wₖ−Wₗ)`,
  nonzero under fractional-pixel shift. Characterized (not mocked) by
  `tests/test_hsi_closure_reprojection.py` as **APPROXIMATE BY DESIGN** (reproduces the
  astropy `reproject_and_coadd` reference; exact SUM/WHT composability is reserved for the
  non-reproject path).
- **HSI classification**: this is `FINALIZATION_MODE_REPROJECT_COADD`. The SCI/WHT pair
  (`master_sum`/`master_coverage`) is finalized by `_save_final_stack` under
  `is_classic_reproject_mode`.
- **Future positive support domain through this path (R2-2)**: positive support must be
  evaluated/transformed **per original exposure** onto the applicable output grid *before*
  accumulation whenever required to preserve the contract — accumulate transformed `s_i`
  into `SUP_W1` and transformed `s_i²` into `SUP_W2`. Transporting a *batch aggregate*
  `SUP_W2` through `R` is **not** generally equivalent to the sum of squared per-exposure
  transformed supports and can become batch-dependent. No design may use the number of
  mini-stacks/batches as an effective exposure count. Scientific `R(V)·R(W)` remains
  APPROXIMATE BY DESIGN, but support-decomposition invariance across the *same* original
  exposures is mandatory.

### 6.8 Drizzle native WHT handling (signed Lanczos)

- `drizzle_core.py`: `WEIGHT_EPSILON=1e-9` (`59`), `LANCZOS_KERNELS={"lanczos2","lanczos3"}`
  (`69`), `DrizzleAccumulator` (`236`). drizzle 2.2.0 stores weighted mean in
  `out_img` and signed weight in `out_wht`; `finalize("divide")` returns native
  `out_img` gated by `finite & (wht>WEIGHT_EPSILON)`.
- `wht_relative_threshold` (`715`) is a coverage/support *policy* (fraction of a
  spatially-supported robust max), scale-invariant, effective 0.0 for Lanczos.
- `support_integrity_violations` (`517`) fails closed on nonzero science over invalid WHT.
- **Derived positive display proxy** (`queue_manager.py:16896-16926`): in the drizzle
  finalization, `final_wht_map_for_postproc += max(native_wht, 0)` per channel then
  `*= 1/3`. This is **display/postprocess bookkeeping only** — NOT a distinct support
  accumulator, NOT per-original-exposure N_eff_support (§8.2).

### 6.9 Preview postprocessing

- `_update_preview_sum_w` (`4960-5160`): `avg=sum/wht` → optional feather (`5013`) →
  optional low-WHT-mask (`5028`) → min/max 0-1 → downsample; captures immutable
  `raw_linear_fullres` as display-analysis data (never science).

### 6.10 Final save postprocessing

- `_save_final_stack` (`16704`): per-mode `final_image_initial_raw` → WHT threshold
  (M3 relative for drizzle; legacy raw-absolute otherwise, `17212`) →
  `raw_adu_data_for_ui_histogram = nan_to_num(raw)` → percentile 0-1 (skipped when
  `preserve_linear_output`) → `data_after_postproc` copy → optional feather (`17351`) +
  low-WHT-mask (`17370`) → FITS from `raw_adu_data_for_ui_histogram` (`17595` float32,
  or uint16 scaling of the same at `17597+`) → PNG from `data_after_postproc` (`17729`).

### 6.11 Boring routing

- `boring_stack.py` is strictly the classic single-batch SUM/W memmap path (M3-D boundary,
  `85-98`): `use_drizzle=False`, no drizzle session, no per-group preview.
  `main() → _run_stack() → start_processing(stacking_mode="winsorized-sigma", use_drizzle=False)`
  (`891-908`). Final-combine via `_resolve_final_combine` (`692`).

---

## 7. Candidate path classification

Classification vocabulary (corrected per R6):
- **ACTIVE_VALID** — live/currently valid on a user-facing scientific/cosmetic path
  today, with no demonstrated replacement. This is a *liveness* classification only — NOT a
  permanent scientific endorsement; a path may be ACTIVE_VALID and still be a COV
  replacement candidate (e.g. the radial taper is ACTIVE_VALID today but a COV-02 target).
- **ACTIVE_OBSOLETE_REPLACED** — live OR legacy, but only *after* a proven replacement
  exists (never asserted in advance).
- **COMPATIBILITY_ONLY** — legacy surface retained for compatibility; not on the primary
  path; may be deletable later.
- **DEAD** — no live caller on the scientific path.
- **UNKNOWN** — liveness/semantics not yet established.

- **ACTIVE_VALID**
  - `DrizzleAccumulator` + `drizzle_stream` + `wht_relative_threshold` +
    `support_integrity_violations` (`drizzle_core.py`) — M3 drizzle finalization
    (`queue_manager.py:16882-16997`).
  - `_stack_mean/_median/_kappa_sigma/_linear_fit_clip/_winsorized_sigma`
    (`stack_methods.py`) — all live reducers.
  - `_feather_batch_coverage` + `make_radial_weight_map` — live batch coverage taper
    (COV-02 *replacement candidate*, NOT yet obsolete).
  - `initialize_master` / `reproject_and_combine` / `reproject_and_coadd_batch`
    (`incremental_reprojection.py`) — live `reproject_coadd` accumulation.
  - `_apply_interbatch_normalization` — live photometric/background normalization.
  - `reproject_and_coadd` family (`reproject_utils.py`) — live mosaic + reproject_coadd.
  - `feather_by_weight_map` / `apply_low_wht_mask` — live but cosmetic-only.
  - `_save_final_stack` — single finalization seam.
  - `_load_classic_batch_wht` (`queue_manager.py:2091`) — live per-channel classic WHT
    sidecar loader (HSIVER==2, WHTSEM=="EFF_DENOM").

- **COMPATIBILITY_ONLY**
  - `drizzle_utils.drizzle_finalize` (`drizzle_utils.py:8`): legacy `sci/wht` finalizer.
    **NOT yet ACTIVE_OBSOLETE_REPLACED** — still called by the drizzle preview path
    (`queue_manager.py:10895-10902`) and by `livestack_mode.py:352`. Its "divide"
    semantics (`sci=flux·wht`) no longer match the M3 native `out_img` convention — a
    latent trap, but it remains *live* on preview/livestack, so classification stays
    COMPATIBILITY_ONLY until a proven replacement removes those call sites.
  - `DrizzleProcessor` / `DrizzleIntegrator` (`enhancement/drizzle_integration.py`) —
    legacy per-file drizzle; imported with fail-open stubs by `stack_enhancement` and
    `mosaic_processor`.
  - `simple_stacker.create_master_tile` (`core/simple_stacker.py`) — mean-only simplified.
  - `reproject_to_reference_wcs` / `resolve_all_wcs` (`core/reprojection.py`) — thin
    wrappers still used for classic reproject.
  - `StackEnhancer` (`stack_enhancement.py`) — normalization/CLAHE/edge crop.
  - `streaming_stack` disk-streaming variants.

- **DEAD**
  - `_wait_drizzle_processes` (`boring_stack.py:96` documents it as a "M3-D legacy no-op").

- **UNKNOWN**
  - `ccdproc_combine` (`queue_manager.py:239`) — only grep hit is `5340`-adjacent
    `reproject_and_combine`; whether any live reducer uses it is unproven (§12.2).
  - `livestack_mode.py` (`LiveStackController`) — no import found from `queue_manager`
    (§12.6).

---

## 8. Scientific-risk findings

### 8.1 Post-normalization gain derived from WHT (CRITICAL, cosmetic-only)

`feather_by_weight_map` (`stack_enhancement.py:378-411`) multiplies the image by a
WHT-ratio gain clipped to **[0.5, 2.0]**, so low-WHT regions are brightened up to 2× —
violating "low WHT is lower confidence, never a reason for brightness gain". Currently
cosmetic-only; any COV reconstruction reusing it would inherit the violation.

### 8.2 Support-confidence status (corrected)

Baseline has **no distinct per-original-exposure positive support accumulator** (SUP_W1 /
SUP_W2 / `N_eff_support` absent — `rg` returns nothing). Concretely:

- Classic WHT is an **effective estimator denominator** (sum of W over completed batches),
  frequently reused as a coverage proxy, but it carries *estimator* semantics — it is not a
  positive support-confidence domain.
- Drizzle WHT is the native **signed** scientific weight; it must stay signed. A separately
  accumulated *positive* support domain is **absent**.
- `final_wht_map_for_postproc` (derived from `max(native_wht, 0)`,
  `queue_manager.py:16896-16926`) is **display/postprocess bookkeeping**, not SUP_W1/SUP_W2
  and not per-original-exposure `N_eff_support`.
- **COV-01 must not feed the final scientific WHT into the support domain**; the support
  accumulator must be generated from original exposures before irreversible mini-stack
  reduction (§11).

**Proposed support-domain contract (COV-01 target, not implemented).** For each original
exposure `i`, define a positive per-pixel support
`s_i = valid_geometric_support · optional_quality_significance · optional_spatial_support_taper`,
accumulate `SUP_W1 += s_i` and `SUP_W2 += s_i²`, and define
`N_eff_support = SUP_W1² / SUP_W2` where defined. `SUP_W1` is **not** a raw exposure count
except in the restricted unit-weight case (`s_i ∈ {0,1}`). For rejection reducers, support
confidence describes the *original geometric/quality* support and may deliberately differ
from the surviving estimator WHT; rejection masks are not used for support unless a
separately named, justified future metric is introduced. Median science remains unchanged
and its support remains independent.

### 8.3 Batch-level radial weighting of the denominator (HIGH — replacement candidate)

`_feather_batch_coverage` multiplies the returned batch denominator by a radial map after
the numerator was already divided by the un-feathered denominator. Single-grid classic:
radial cancels per-pixel in the global divide (no net bias) but the stored coverage map is
radially attenuated (affects WHT threshold/display/low-WHT-mask). Mosaic/reproject: does
not cancel → geometry-dependent edge bias. Interbatch feather (`3262`, 0.98) additionally
biases overlap statistics. COV-02 replaces this global radial taper with a
transformed-real-footprint taper — only after replacement proof does this become obsolete.

### 8.4 Drizzle signed WHT preserved (positive)

`DrizzleAccumulator` keeps native signed `out_wht`; validity `finite & (wht>WEIGHT_EPSILON)`;
Lanczos threshold forced 0.0; no `abs(wht)`, no huge-value clip, no percentile hiding.
Gap: legacy `drizzle_utils.drizzle_finalize` semantics (§7).

### 8.5 Median not redefined; reproject approximate (positive)

`_stack_median` = `np.nanmedian` (NaN excluded, W=valid count). Reprojection is
APPROXIMATE BY DESIGN (`test_hsi_closure_reprojection.py`), reserved to non-reproject
composability.

---

## 9. Scientific output vs preview / render / final-save boundaries

| Boundary | Data | Notes |
|---|---|---|
| Scientific accumulator | `cumulative_{sum,wht}_memmap` (classic) / `DrizzleAccumulator._out_img/_out_wht` (drizzle) / `master_sum/master_coverage` (reproject_coadd) | raw SUM/W, native drizzle, or `R(V)·R(W)` master |
| Scientific final pixels | `final_image_initial_raw` → `raw_adu_data_for_ui_histogram` | WHT threshold NaN→0; this is what FITS writes |
| FITS primary HDU | `raw_adu_data_for_ui_histogram` (float32 raw ADU, or uint16-scaled) | `17595` (float32), `17597+` (uint16); feather/low-WHT-mask **not** applied |
| FITS companion WHT | `final_wht_hwc` (native signed, drizzle only) | `_write_companion_wht_fits`, only when `save_drizzle_wht` |
| UI preview (live) | `preview_data_normalized` (+ `raw_linear_fullres` 2nd element) | feather+low-WHT-mask+minmax applied |
| UI preview (final) | `last_saved_data_for_preview = data_after_postproc` | cosmetic 0-1 |
| PNG | `data_after_postproc` | `_save_final_preview_png` |

**Key seam**: FITS pixels and cosmetic preview/PNG diverge at `data_after_postproc`.
COV-04's final-only render seam belongs *after* `raw_adu_data_for_ui_histogram` is
captured, feeding only `data_after_postproc`/preview/PNG — never FITS pixels.

### 9.1 Stale source docstring (debt)

`queue_manager.py:16718` (docstring) claims: "La sauvegarde FITS reste basée sur
`self.raw_adu_data_for_ui_histogram` (si float32) ou les données cosmétiques [0,1] (si
uint16)." The **code** at `17597+` (uint16 branch) uses
`raw_data = self.raw_adu_data_for_ui_histogram` and scales that raw ADU to 0-65535 — it
does **not** use the cosmetic `data_after_postproc`. Code behavior: both float32 and
uint16 FITS writes use the raw linear data. The docstring is stale; record as debt for a
later docs-only cleanup (not this gate).

---

## 10. Resume / checkpoint state (for SUP_W1 / SUP_W2)

### 10.1 Classic SUM/W — schema, seams, cleanup

- Constants (`queue_manager.py:653-660`): `_RESUME_MANIFEST_VERSION = 2` (current),
  `_RESUME_MANIFEST_VERSION_MIN = 1` (legacy), `_RESUME_STATE_CLEAN="clean"`,
  `_RESUME_STATE_DIRTY="dirty"`, `_RESUME_MODE_CLASSIC_SUMW="classic_sumw"`.
- **Schema v1** manifests (legacy engine hash + HSI fields only) remain *readable and
  resumable* under their exact fingerprint contract; **never upgraded in place** (`662-663`).
- Manifest semantics (v2, `13719-13724`): `sum` = "HWC numerator: sum V·W", `wht` =
  "HWC effective denominator: sum W", `final` = "per-channel SUM/WHT".
- **Write** (`_write_resume_manifest`, `13662+`): v2 builds `_canonical_run_config()`
  and refuses (fail-closed) if `classic_fingerprint(cfg) != engine fingerprint`; atomic
  temp + `os.replace`; state clean/dirty.
- **Read** (`_read_resume_manifest`, `13970+`): refuses if manifest missing, corrupt,
  schema ∉ {1,2}, state != clean, mode != classic_sumw, or fingerprint mismatch; v2 also
  validates canonical config consistency; quality-weighted resumes require a positive
  finite `quality_reference_scale`.
- **Failed-start cleanup (ownership-safe)**:
  - `_ATTEMPT_CREATED_CHECKPOINT_ARTIFACTS` allowlist (`667-672`):
    `cumulative_SUM.npy`, `cumulative_WHT.npy`, `resume_manifest.json`,
    `resume_manifest.json.tmp` (plus `batches_count.txt` in cleanup).
  - `_snapshot_existing_state` (`2599`) records pre-existing `memmap_accumulators`
    (+contents), `batches_count.txt`, `run_config.cfg`, `.m3d_checkpoint`.
  - `_remove_attempt_created_state` (`2636`) prunes only allowlisted names absent from
    the snapshot; removes the dir only if attempt-created and now empty.
  - `_cleanup_failed_start` (`2725`) is idempotent/ownership-safe: stops autotuner if
    this attempt started it, releases norm-reference memory, closes memmap handles (also on
    resume false starts — without touching files), removes attempt-created artifacts
    (fresh-only).
- **Implication for COV-01**: any new SUP_W1/SUP_W2 artifact must be added to
  `_ATTEMPT_CREATED_CHECKPOINT_ARTIFACTS` and to `_snapshot_existing_state` /
  transaction ownership, or a failed fresh start could either (a) be falsely refused by its
  own leftovers, or (b) leak SUP files. See §11.

### 10.2 Drizzle (M3) — writer/reader/namespace cleanup

- `<output>/.m3d_checkpoint/checkpoint.json` (`drizzle_checkpoint.py:184-186`) is the
  single commit point. Writer is **write-only**; `read_drizzle_checkpoint` validates the
  *entire* checkpoint before `DrizzleAccumulator.from_native_state`
  (`drizzle_core.py:291`) rebuilds the native `out_img`/`out_wht`. `total_exptime` must be
  restored alongside the buffers.
- **Exactness (honest, per R2-3)**: the restore seam can be byte/array exact — unit
  witnesses `tests/test_drizzle_resume_continuation.py` and
  `tests/test_drizzle_resume_backend.py::test_stop_resume_backend_is_bit_identical_and_commits_n_plus_1`
  assert `np.array_equal` / `max_abs_diff == 0.0` for the native buffers across the `.npy`
  roundtrip and the backend Stop→Resume; `read_drizzle_checkpoint` also pins the
  drizzle/numpy library versions so continuation is refused across a rounding-behaviour
  change (`drizzle_checkpoint.py:1701`). However, known production E2E Stop→Resume evidence
  on this baseline showed SCI and WCS **bit-identical** while native WHT continuation may
  differ at float32 **ULP-level** (historically 1–2 ULP in bounded witnesses). Do not claim
  blanket bit identity. COV support state must target the strongest reproducibility
  contract and record exact evidence (bit-exact where proven, bounded-ULP otherwise).
- Failed-start cleanup treats `.m3d_checkpoint` as a whole-directory artifact: it is
  removed only if absent from the pre-existing snapshot (`2699-2713`). A
  failed-first-checkpoint (fresh run) therefore leaves no stale `.m3d_checkpoint` to
  falsely refuse a retry.
- **Implication for COV-01**: a positive-support domain for drizzle needs its own
  persistence (a new checkpoint field or sibling artifact) validated by the same
  read-entirely-then-restore discipline; it is **absent** today.

### 10.3 Drizzle background anchor

- `drizzle_background.py`: `BackgroundAnchor` (`362`) — immutable anchor state
  (`anchor_data`, `tf`, `reference_shape_hw`), scalar-contract
  `to_metadata/from_metadata`, `estimate_background_offsets` (`551`) /
  `apply_background_offsets` (`740`).

### 10.4 Evidence tests (not run at this docs-only gate)

`tests/test_resume.py`, `tests/test_drizzle_resume_backend.py`,
`tests/test_drizzle_resume_continuation.py`, `tests/test_qt_last_stack_resume_m23.py`,
`tests/test_resume_intent_contract_rsm2.py`, `tests/test_qt_resume_selector_rsm2_02c.py`,
`tests/test_hsi_closure_reprojection.py`, and the `test_hsi_closure_*` family.

### 10.5 Missing support state → future legacy contract (NOT implemented today)

If a checkpoint (classic v2 or drizzle) lacks positive-support state: **science may
resume** (SUM/WHT or native out_img/out_wht are sufficient), but a
**confidence-aware render must be disabled** — do not fabricate SUP_W2. This is a required
future legacy contract, not current behavior.

---

## 11. COV seams by mission gate

> These are *identification* only (this gate changes no code). COV-01 is **not** final
> cosmetic reconstruction; final-only render is COV-04.

- **COV-01 — original-exposure support generation.** Generate positive support **before**
  irreversible mini-stack reduction. Per original exposure define
  `s_i = valid_geometric_support · optional_quality_significance · optional_spatial_taper`
  (NOT a raw count except unit-weight), then `SUP_W1 += s_i`, `SUP_W2 += s_i²`, and
  `N_eff_support = SUP_W1²/SUP_W2`. Backend-neutral, separate SUP_W1/SUP_W2 state. For
  reproject modes the support is evaluated/transformed per original exposure onto the
  output grid before accumulation (never transported as a batch aggregate, never
  batch-count-as-N_eff). Rejection reducers: support describes original geometric/quality
  support and may deliberately differ from the surviving estimator WHT; rejection masks
  are not used for support. Median science unchanged. Seams: the batch reducer loop
  (`queue_manager.py:12400-12830`) where per-image masks are consumed *before*
  `_feather_batch_coverage`; classic memmap init/load/commit (`4400-4410`, `13662-14040`)
  and drizzle `DrizzleAccumulator`/`from_native_state` for persistence. New SUP artifacts
  must join the cleanup allowlist (§10.1).
- **COV-02 — transformed real-footprint taper.** Replace the global
  `make_radial_weight_map` taper (§6.3/§6.4) with a per-footprint transformed taper.
  Seam: `_feather_batch_coverage` (`11896`) and the six radial call sites.
- **COV-03 — reliable-overlap normalization.** Harden `_interbatch_compute_scales`
  (`3286`) overlap/scale estimation against edge-bias from the radial taper.
- **COV-04 — final-only render seam.** Insert a cosmetic reconstruction stage after
  `raw_adu_data_for_ui_histogram` capture (`17198+`), feeding only
  `data_after_postproc`/preview/PNG. Replace/augment `feather_by_weight_map` (remove or
  cap the gain) and `apply_low_wht_mask` (median-fill → reconstructed edge).
- **COV-05 — cleanup.** Delete/reclassify now-obsolete paths only after replacement proof
  (e.g. `drizzle_utils.drizzle_finalize`, radial weighting) is demonstrated.

Note: a `max_gain=1.0` option for `feather_by_weight_map` is **not** the COV-01 support
architecture; it is at most a temporary cosmetic safety idea.

---

## 12. Uncertainties and resolving tests

1. **"classic" reducer identity** — ccdproc-`combine` vs `_stack_mean`. Test: identical
   inputs, compare batch `STK_NOTE` + pixels.
2. **`ccdproc_combine` liveness** — grep only finds `5340`-adjacent. Test: trace
   `reproject_and_combine` callers.
3. **Radial cancellation (single-grid)** — Test: `apply_batch_feathering=True/False`,
   diff final FITS.
4. **Mosaic/reproject edge bias + reproject support decomposition** — Test: synthetic
   two-panel mosaic; measure edge vs centre photometry; and prove that accumulating
   per-exposure transformed supports (`SUP_W2 = Σ R(s_i)²`) is invariant to batch grouping
   for the same original exposures, versus the (batch-dependent) `R(Σ s_i²)` aggregate
   transport.
5. **`feather_by_weight_map` gain magnitude** — Test: instrument, log `gain_map_clipped`.
6. **`livestack_mode` reachability** — Test: grep imports of `LiveStackController`.

---

## 13. Frozen-invariant conformance (observed)

| Invariant | Observed status |
|---|---|
| SCI conceptually SUM/WHT | Holds (classic `cumulative_sum/wht`; drizzle native `out_img/out_wht`) |
| Low WHT never a brightness-gain reason | **VIOLATED by `feather_by_weight_map` gain∈[0.5,2.0]** (cosmetic-only today) |
| Distinct positive support confidence vs scientific WHT | **ABSENT at baseline** — classic WHT is an estimator denominator; drizzle WHT is signed scientific; only a derived display proxy exists (`16896-16926`). Not an implemented separation. |
| Native drizzle Lanczos WHT may be signed, stays signed | Holds (`DrizzleAccumulator.wht` un-clipped; Lanczos threshold 0.0) |
| Median not silently redefined | Holds (`_stack_median` = nanmedian) |
| Reproject APPROXIMATE BY DESIGN | Holds (separate `R(V)·R(W)` transport; `test_hsi_closure_reprojection.py`) |
| Cosmetic reconstruction final-only & separate | Partially: cosmetics already separate from FITS pixels; COV-04 reconstruction not yet implemented |
| All existing modes remain | Holds (no mode deleted; dispatch chain intact) |

---

*Prepared by Coco (COV-00, revision R1). Evidence from source inspection of
`seestar/queuep/queue_manager.py`, `seestar/core/*` (incl. `incremental_reprojection.py`),
`seestar/enhancement/*`, `seestar/gui/*`, `seestar/gui_qt/*`, `tests/*` at HEAD
`501eb9b`. No production code changed.*
