# Registration & field rotation — RF-1 research report (corrective iteration C)

**Task:** RF-1 (corrective C, iteration 2/3) — corrected scale statistics, corrected global-reference (coordinate-frame vs target-image) audit, and a new batching-dependence POC.
**Baseline:** `61291aa035562f09a2bd7cf7971ef8192e1585bf`
**Branch:** `feature/registration-field-rotation`
**Scope:** production-external. **No production code under `seestar/` was changed.**

This document replaces the RF-1B report.  RF-1B was rejected after independent
review for two material scientific statements; both are corrected here from
*executed* code and *measured* results, and the previously-missing
batching-dependence experiment is added.

---

## 0. Corrected model verdict

**`MODEL_DECISION: FURTHER DATA REQUIRED`**

The executed classic/Drizzle registration path fits an astroalign **similarity**
transform and then **discards the scale, forcing scale = 1.0** (an **Euclidean**
rotation+translation model).  Two measured facts, and one unmeasured one,
determine the verdict:

1. **Synthetic (measured):** under an injected 0.3% uniform scale drift, the
   current Euclidean model leaves a **2.75 px corner residual** (centre 0.77 /
   edge 1.74 / corner 2.75 px).  So the current model is **not** universally
   sufficient — it *would* break if real frames carried a comparable scale
   drift.
2. **Real M16 (measured, one session):** the astroalign similarity scale is
   **consistent with 1.0 within noise** (see §7 for the *corrected* statistics:
   `median(|scale−1|) = 74.1 ppm` ≈ **0.08 px** at the frame corner, below the
   ~0.15 px held-out centroid noise).  Retaining scale gives a **negligible,
   signed-± held-out improvement** (median ~0.001 px, negative on some frames)
   — the scale is *fit noise* on this session, not a coherent signal.
3. **Unmeasured:** cross-session / temperature-dependent scale drift.  M16 is
   **one session** (2025-05-30, ~23 min, one focal state).  It cannot establish
   whether a different session/temperature would show the 0.3% drift that the
   synthetic case shows is damaging.

Because the synthetic case proves the current model can leave a 2.75 px corner
error *if* real scale drift exists, and the single real session cannot rule it
out, the honest gate is **FURTHER DATA REQUIRED** — not "CURRENT MODEL IS
SUFFICIENT" (RF-1A's unsupported claim) and not a model change without data.

**Correction vs RF-1B (defect #1):** the RF-1B label "median |scale−1| =
24.5 ppm" was wrong — the code computed `abs(median(scale)−1)`.  The two
quantities are different; this report now reports both (plus the mean), with
the corner-pixel implication restated using the correct statistic.  See §7.

---

## 1. `CURRENT_REGISTRATION_MODEL`

| Field | Value |
|---|---|
| Star detection / matching | `astroalign.find_transform` (SEP source extraction + triangle invariant matching + RANSAC) |
| Transform **returned** by astroalign | `SimilarityTransform` (rotation + uniform scale + translation) — 4 dof |
| Transform **retained** by production | `SimilarityTransform(rotation=θ, translation=(tx,ty))` with **scale=1** — 3 dof (Euclidean) |
| Discarded information | uniform scale `s` (astroalign fits similarity only — no shear) |
| Applied warp | `cv2.warpAffine` 2×3 (CPU) / `cv2.cuda.warpPerspective` 3×3 (CUDA), `INTER_LINEAR` |
| Affine/projective assumption | none — the model is rigid rotation + translation only |

**Code anchors** (verified from installed source)

* `astroalign` **2.6.2**, `_MatchTransform.fit` (`astroalign.py:198-211`):
  `estimate_transform("similarity", self.source[s], self.target[d])` — the
  returned object is always a **similarity**.
* `find_transform` returns `(best_t, (source_controlp[s], target_controlp[d]))`
  (`astroalign.py:412-414`) — the matched star pairs are recoverable and are
  used as the real-data witness (§7).
* `seestar/core/alignment.py:220-226` — `transform_skimage_obj, (source_matches,
  target_matches) = aa.find_transform(source=..., target=...)`.  The matches are
  used only to log a count; the transform is what is kept.
* `seestar/core/alignment.py:228-237` — scale discard:

```python
a, b = transform_skimage_obj.params[0, 0], transform_skimage_obj.params[1, 0]
theta = np.arctan2(b, a)
tx = transform_skimage_obj.params[0, 2]
ty = transform_skimage_obj.params[1, 2]
transform_no_scale = SimilarityTransform(rotation=theta, translation=(tx, ty))
cv2_M = transform_no_scale.params[:2, :]
```

For a similarity matrix `[[s·cosθ, -s·sinθ, tx],[s·sinθ, s·cosθ, ty],…]`,
`θ = arctan2(s·sinθ, s·cosθ) = θ` is recovered exactly and `s` is dropped.
Rotation and translation are therefore correct *when the true geometry is
similarity*; only scale is lost.

---

## 2. `MODEL_COMPARISON` (synthetic table — unchanged from RF-1B)

Deterministic synthetic star fields (`1920×1080` px, corner radius 1101.5 px,
centroid noise σ = 0.05 px, 100 stars, 70/30 fit/hold-out, fixed seeds).
Models: `translation` (2), `euclidean` (3, **current**), `similarity` (4),
`affine` (6), `projective` (8), `poly3` (20 dof, order-3 polynomial).

**Held-out RMS residual (px)** — lower is better; `FAIL` = underdetermined fit.

| scenario | translation | euclidean | similarity | affine | projective | poly3 |
|---|---|---|---|---|---|---|
| translation | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.07 |
| rotation (30°) | 368.07 | **0.08** | 0.08 | 0.08 | 0.08 | 0.08 |
| rotation_translation (18°) | 182.91 | **0.08** | 0.08 | 0.08 | 0.08 | 0.08 |
| large_rotation (120°) | 1.0e+03 | **0.09** | 0.08 | 0.08 | 0.09 | 0.09 |
| scale (0.3%) | 11.24 | 1.89 | **0.06** | 0.06 | 0.06 | 0.07 |
| affine (shear) | 2.92 | 2.53 | 0.94 | **0.08** | 0.08 | 0.08 |
| projective | 1.34 | 1.20 | 0.76 | 0.55 | **0.07** | 0.07 |
| smooth_local (radial r³) | 47.77 | 0.50 | 0.28 | 0.26 | 0.25 | **0.08** |
| partial_overlap | 117.01 | **0.06** | 0.06 | 0.06 | 0.06 | 0.07 |
| outliers (15%) | 183.63 | **0.06** | 0.06 | 0.06 | 0.06 | 0.08 |
| degenerate (3 stars) | 0.09 | 0.06 | 0.06 | FAIL | FAIL | FAIL |

**Scale scenario detail (0.3% uniform drift)** — the only scenario where the
discarded scale matters:

| model | centre | edge | corner | recovered scale |
|---|---|---|---|---|
| euclidean (current) | 0.77 | 1.74 | 2.75 | 1.0 (forced) |
| similarity | 0.06 | 0.06 | 0.04 | 1.00300 |

**Smooth (radial r³) correction:** `poly3` now reaches the noise floor
(hold RMS 0.08 px), *proving* that an order-3 polynomial (which spans `x³` and
`x·y²`) represents the injected r³ field.  RF-1A's `poly2` left 0.25 px —
precisely the unrepresentable cubic part.  `poly3`'s robustness cost is real
(20 free parameters), but under the same closed-form + MAD protocol it still
recovers to the noise floor on the 15%-outlier scenario here (0.08 px); a
flexible model still demands a proper RANSAC in production, and this synthetic
result must not be read as robustness evidence for a real estimator.

**Outliers (15% false matches, deterministic MAD rejection, `k=5`)**: all
rigid/low-order models recover to the noise floor; this is a *relative*
comparison only, not astroalign's internal triangle+RANSAC estimator.

**Failure behaviour** (`degenerate`, 3 stars → 2 in fit set): `translation`,
`euclidean`, `similarity` fit; `affine` (≥3), `projective` (≥4), `poly3` (≥10)
correctly refuse.  Minimum-point guards are respected.

**Runtime (corrected, non-decisive):** measured min/median/max fit wall-time
per model across the scenarios (µs, machine-dependent; regenerated at report
time, so the exact values drift between runs):

| model | min | median | max | fitted scenarios |
|---|---|---|---|---|
| translation | 55.9 | 67.7 | 716.2 | 11 |
| euclidean | 389.7 | 528.7 | 2081.4 | 11 |
| similarity | 432.6 | 561.5 | 1792.3 | 11 |
| affine | 921.6 | 1290.2 | 4920.9 | 10 |
| projective | 924.9 | 1318.2 | 4771.4 | 10 |
| poly3 | 277.0 | 353.5 | 1629.5 | 10 |

Model complexity is still not a runtime driver relative to pixel resampling.

---

## 3. `RESIDUAL_ANALYSIS` (unchanged from RF-1B)

### 3.1 Fit vs held-out

On rigid scenarios the fit and held-out residuals are both ≈ the 0.05 px noise
floor for every rotation-aware model (70 points ≫ dof, no overfitting).  On the
`scale` scenario the Euclidean model shows a large **fit** residual
(1.82 px RMS) that does not shrink on held-out data — the signature of *model
bias* (a missing degree of freedom), not noise.

### 3.2 Detector-coordinate (radial) structure

| model | Spearman(r, resid) | slope px/1000px | hold_rms px |
|---|---|---|---|
| translation | +0.961 | +81.73 | 47.77 |
| euclidean (current) | +0.851 | +1.32 | 0.50 |
| similarity | −0.244 | −0.12 | 0.28 |
| affine | −0.087 | −0.03 | 0.26 |
| projective | −0.075 | −0.06 | 0.25 |
| poly3 | +0.324 | +0.04 | 0.08 |

The current Euclidean model leaves a **strong monotonic radial residual**
(Spearman +0.85) when a smooth radial distortion is present — the diagnostic
signature of an unmodeled detector-coordinate distortion.  `poly3` removes it
(0.08 px, no monotonic slope).  This is *synthetic by construction*; it does
not imply M16 has such distortion.  A real stable radial distortion would show
this same signature in RF-2.

### 3.3 Centre / edge / corner

On `scale`, the Euclidean residual grows centre 0.77 → edge 1.74 → corner
2.75 px (unmodeled uniform scale).  On rigid scenarios there is no spatial
gradient (centre ≈ edge ≈ corner ≈ noise).

---

## 4. `GLOBAL_REFERENCE_AUDIT` (corrected — RF-1B conflated coordinate frame, target image, and provenance)

RF-1B's headline "direct vs chained mapping" mixed up three distinct questions.
This section separates them and answers each exactly.  The *executed-flow*
facts (which branch reassigns the reference variable) are unchanged from RF-1B
and are still correct; the *language* about geometry/provenance is corrected.

### 4.1 Where `reference_image_data_for_global_alignment` is assigned

All assignments, from `seestar/queuep/queue_manager.py` (pinned by the AST
test `tests/test_global_reference_audit.py`):

| line | value | meaning |
|---|---|---|
| `5235` | `= None` | initialiser |
| `5419-5423` | `= self.aligner._get_reference_image(...)` | **initial reference** (a single frame, manual or auto-best by `median/(1.4826·MAD)`, `alignment.py:375+`) |
| `5928-5936` | `= self._flush_current_batch(...)` | batch-plan flush **seam** — returns its input reference **unchanged** (see 4.3) |
| `6252` | `= stack_img` (`_solve_cumulative_stack()`) | **mutation** (reproject, worker path) |
| `6259` | `= stacked_np.astype(...)` | **mutation** (reproject fallback) |
| `6266` | `= stacked_np.astype(...)` | **mutation** (reproject, inner `else` — dead code, still inside the positive outer guard) |
| `6749` | `= stack_img` | **mutation** (reproject, finalize last-batch) |
| `6754` | `= stacked_np.astype(...)` | **mutation** (reproject finalize fallback) |
| `6759` | `= stacked_np.astype(...)` | **mutation** (reproject finalize, inner `else` — dead code, still inside the positive outer guard) |

Every mutation (6252/6259/6266/6749/6754/6759) is lexically inside a
**positive** `if self.reproject_between_batches` guard (worker path guarded at
6147 and 6243; finalize path at 6695 and 6746).  Lines 6266 and 6759 sit in an
inner `else` whose condition is the same already-true guard, so they are dead
code — but they are still *inside* the positive outer guard, not in the classic
path.  There is **no** mutation site in the plain-classic, M3-Drizzle,
batch-plan-flush, or `reproject_coadd_final` paths.

### 4.2 Exact executed conditions (unchanged from RF-1B)

* **Plain classic** (`reproject_between_batches=False`, `drizzle_active_session=False`):
  reference stays the immutable initial frame.  → **direct** source→immutable.
* **M3 Drizzle standard** (`drizzle_active_session=True`, `reproject_between_batches=False`):
  same — reference stays immutable; `_process_file` returns the original prepared
  frame + a 2×3 `tf`, which `pixmap_from_alignment` turns into a pixel map.
  → **direct** source→immutable (via tf).
* **`reproject_between_batches`** (classic *or* drizzle): after each batch is
  stacked, WCS-solved, and combined, the worker calls `_solve_cumulative_stack()`
  and **replaces** the reference with `stack_img` (or `stacked_np`) at 6243-6272
  (and again for the last partial batch at 6746-6772).
* **Batch-plan flush** (`use_batch_plan`, `_BATCH_BREAK_TOKEN`): the flush seam
  at 5928-5936 reassigns the reference to `_flush_current_batch(...)`'s return,
  which in the *active* classic branch returns its input **unchanged**
  (`queue_manager.py:9067-9174`; the reproject branch inside is dead code
  `if False and ...`).  → **no-op reassignment**, reference stays immutable.
* **`reproject_coadd_final`** (`reproject_coadd_final=True`,
  `reproject_between_batches=False`): never touches the reference variable.
  → immutable.

### 4.3 What `_solve_cumulative_stack()` returns as the new reference

`_solve_cumulative_stack` (`queue_manager.py:11877+`) returns the cumulative
`sum / wht` stack (float32) and its header.  With
`freeze_reference_wcs = self.reproject_between_batches` (`2987`), it **skips
re-solving** after the first solve and reuses the frozen reference WCS — so the
returned stack **grows** (more batches → more SNR) while its WCS grid stays
frozen.

### 4.4 Coordinate frame vs target image vs provenance — corrected statement

Three distinct facts, previously collapsed into "chained":

1. **Coordinate frame (grid) — FROZEN.**  `freeze_reference_wcs` keeps the
   reference WCS grid fixed across batches.  A per-frame astroalign matrix
   fitted against the current cumulative image still maps **original source
   pixels directly into the same frozen global pixel coordinate system**.
   There is **no explicit transform composition** source→batch-grid→global-grid;
   the per-frame matrix is a single direct source→global-grid map.  **Direct
   coordinate mapping to the frozen global grid: YES.**

2. **Target image (data) — EVOLVES.**  In `reproject_between_batches` mode the
   reference **image data** fed to `find_transform` is the cumulative stack,
   which changes each batch (moving SNR).  The **immutable target image: NO** in
   `reproject_between_batches` (the initial single-frame reference is
   overwritten after the first batch).  What changes with batching is the
   *noisy registration target image / centroid field*, which may make the
   fitted transforms **batch-history dependent**.

3. **Provenance — NOT RETAINED (classic reproject path).**  The per-frame
   `matrix_M_val` / `M_astroalign` is **not persisted** in the classic reproject
   path; only the aligned pixels / batch products remain.  The original fit
   **cannot be reconstructed** post-hoc.  (A source→frozen-grid map *is*
   available in the sense that the grid is frozen, but it was fitted against
   the moving stack, and the specific matrix used per frame is not stored.)
   **Transform reproducibility/provenance after processing: NO** (classic
   reproject path).

4. **Accumulated/batch-dependent centroid bias — HYPOTHESIS, now tested.**  The
   assignment mutation alone does **not** prove any bias; it only establishes
   that the target evolves.  Whether the evolving target actually introduces
   batch-dependent/accumulated centroid bias is an **experimental** question,
   answered by the new batching-dependence POC (§8).

### 4.5 Answer to the global invariant (exactly)

> Is there a single immutable global reference, with each frame mapped directly
> to it, reproducibly?

* **Coordinate frame:** the global *grid* is immutable (frozen WCS), and each
  frame maps **directly** into it.  **Yes.**
* **Target image:** the reference *image data* is **not** immutable in
  `reproject_between_batches` mode (it is the evolving cumulative stack).
  **No.**
* **Provenance:** the per-frame transform is **not** retained in the classic
  reproject path, so the fit is **not** reconstructible after processing.
  **No.**
* **Independence/bias:** whether the moving target causes batch-dependent or
  accumulated centroid bias is **measured** in §8, not asserted from the
  mutation sites.

### 4.6 Architecture finding (decision for Jarvis — not implemented)

The audit reveals a **moving-target design** in `reproject_between_batches`
mode: a frozen coordinate grid with an evolving reference image.  It is
plausibly intentional (higher-SNR reference, frozen WCS grid), but it means
per-frame transforms are fitted against different targets and their provenance
is discarded.  The **separate architecture finding** (independent of the scale
decision) is: **direct fixed-grid mapping exists, but the evolving target may
be batch-history dependent** — a hypothesis the batching-dependence POC (§8)
now tests and partially confirms.  If reproducibility/independence matters, the
smallest RF-2 POC (§9.3) is to retain an immutable high-SNR target or persist
each frame's registration solution.

---

## 5. `RESAMPLING_AUDIT` (unchanged from RF-1B)

**Terminology:** *interpolation* (per-pixel value reconstruction during a
geometric warp) and *resampling/deposition* (flux redistribution onto the
output grid) are distinct.  Drizzle does **not** pre-interpolate source data;
its flux redistribution **is** the single sampling/deposition stage.

### 5.1 Every warp/reproject path (production files)

| Path | Model | Data operation |
|---|---|---|
| `alignment._align_cpu` — `cv2.warpAffine`, `INTER_LINEAR` | 2×3 (Euclidean) | interpolation |
| `alignment._align_cuda` — `cv2.cuda.warpPerspective`, `INTER_LINEAR` | 3×3 (Euclidean) | interpolation |
| `fast_aligner_module.warp_image` — `cv2.warpAffine`, `INTER_LINEAR` | 2×3 (similarity) | interpolation |
| `livestack_mode._align` — `aa.apply_transform` | similarity | interpolation |
| `reprojection.reproject_to_reference_wcs` — `reproject_interp` | WCS→WCS | interpolation |
| `incremental_reprojection.initialize_master/reproject_and_combine` | WCS→WCS | interpolation |
| `drizzle_core.pixmap_from_alignment` | 2×3 tf → pixel-centre map | none (coordinate mapping) |
| `DrizzleAccumulator.add` → `drizzle.resample` | pixmap | deposition (not interpolation) |

### 5.2 Number of geometric sampling stages per source frame, by mode

| Mode | interpolation stages | deposition (drizzle) stages |
|---|---|---|
| Classic stacking | 1 (per-frame `warpAffine` INTER_LINEAR) | 0 |
| M3 Drizzle standard | 0 on data | 1 (drizzle flux redistribution) |
| Classic + `reproject_between_batches` | 2 — per-frame warp + batch-stack WCS reprojection | 0 |
| Mosaic local_fast | 1 | 0 |
| Livestack | 1 | 0 |

### 5.3 Unused computed warp (the M3 Drizzle waste)

In M3 Drizzle standard, `_process_file` requests the matrix via `return_M=True`
and `_align_image` fully executes `cv2.warpAffine`, then `_process_file`
replaces the returned data with the *original* prepared frame and feeds it
through `pixmap_from_alignment` — no warp on data.  The computed `aligned_img`
is an unused, thrown-away warp (geometrically harmless; computationally
wasteful).

### 5.4 Measured single-star effect (σ=1.5 px, peak 1000, 0.3 px diagonal shift)

| method | flux | peak | FWHM (px) | ecc | centroid |
|---|---|---|---|---|---|
| identity (no resampling) | 14137.17 | 1000.00 | 3.532 | 0.000 | (32.00,32.00) |
| warpAffine INTER_LINEAR (production CPU) | 14137.17 | 879.34 | 3.697 | 0.000 | (32.31,32.31) |
| warpAffine INTER_CUBIC | 14137.17 | 969.80 | 3.461 | 0.000 | (32.34,32.34) |
| warpAffine INTER_LANCZOS4 | 14137.17 | 954.41 | 3.532 | 0.000 | (32.33,32.33) |
| drizzle pixmap deposition (square) | 14137.17 | 884.02 | 3.694 | 0.000 | (32.30,32.30) |

Flux is conserved by every method.  `INTER_LINEAR` blurs a sharp star
(peak −12 %, FWHM +4.7 %).  Kernel/PSF study is deferred to RF-2.

---

## 6. `MODEL_DECISION`

**`FURTHER DATA REQUIRED`** (as stated in §0).

Rationale (evidence-gated):

1. The dominant physically-expected geometry for a Seestar S50 (alt-az mount)
   is field rotation + drift translation — the Euclidean model.  Synthetic:
   Euclidean reaches the 0.05 px noise floor on every rigid scenario.
2. **Translation-only is ruled out** (100–1000 px held-out residuals on any
   rotation) — the current model must keep rotation.
3. **The current model is not universally sufficient**: the synthetic 0.3%
   scale case leaves a 2.75 px corner residual, so a real scale drift of that
   size would be damaging.  This is a *measured possibility*, not an assumption.
4. **Real data does not resolve it**: the single M16 session shows scale ≈ 1.0
   within noise (§7), so on *this* session the current model is fine — but one
   session cannot establish cross-session/temperature behaviour.
5. The highest-leverage discarded quantity is the **similarity scale**; the
   decision requires measuring it across **multiple sessions / focal states**.

---

## 7. M16 real-data witness (corrected — complementary evidence)

Script: `research/registration_field_rotation/m16_scale_witness.py` (no
`seestar` import; faithful reimplementation of the production preparation basis).

### 7.1 Preprocessing method (corrected — defect: RF-1B hardcoded GRBG and omitted hot-pixel correction)

The witness now replicates the production preparation **exactly**, verified
bit-identical against the production helpers (see §10 commands):

1. **Load + normalize** (`load_and_validate_fits` equivalent): first image HDU,
   min/max normalized to float32 [0,1], non-finite values zeroed.
2. **Variance gate** (`std < 0.0005` → reject) for reference candidates.
3. **Debayer with header `BAYERPAT`** (fallback `GRBG`), not a hardcoded
   pattern.  All 20 M16 frames carry `BAYERPAT='GRBG'`, so this fix does not
   change the numbers, but it is now faithful.
4. **Hot-pixel correction** — `detect_and_correct_hot_pixels` (threshold 3.0,
   neighbourhood 5), replicated bit-identically on the CPU path
   (`cv2.medianBlur` median + `cv2.blur` box-filter mean/mean_sq +
   `hot = channel > median + threshold*std` + replace by median).  The only
   production branch not replicated is the CUDA box filter (unavailable here);
   CUDA-vs-CPU box filtering differs only at float rounding, far below the
   scale conclusion's noise.
5. **White balance** is green-invariant (production scales only R and B), and
   the reference-selection metric is computed without it, so it is correctly
   omitted for the green-channel alignment.

**Reference selection result changed** (this is the measurable effect of the
hot-pixel fix): the production-quality metric
`median/(1.4826·MAD)` now selects **`Light_M 16_10.0s_LP_20250530-035421.fit`**
(metric 24.113) instead of RF-1B's `…_040549.fit` (23.832).  The A/B below
shows this does not alter the scale conclusion.

### 7.2 Per-frame table (astroalign returned scale; 19/19 aligned, 0 failures)

| frame | matches | a.a. scale | rot(deg) | tx | ty | eucl hold RMS | sim hold RMS | eucl corner | sim corner |
|---|---|---|---|---|---|---|---|---|---|
| …_035203 | 48 | 1.000057 | −0.006 | −52.5 | −16.8 | 0.139 | 0.137 | 0.056 | 0.119 |
| …_035214 | 48 | 1.000083 | −0.010 | −54.7 | −19.0 | 0.161 | 0.157 | 0.131 | 0.123 |
| …_035225 | 48 | 1.000138 | −0.009 | −56.6 | −20.9 | 0.170 | 0.146 | 0.150 | 0.132 |
| …_035236 | 48 | 1.000020 | −0.009 | −57.6 | −23.0 | 0.131 | 0.130 | 0.063 | 0.062 |
| …_035248 | 48 | 1.000031 | −0.009 | −59.7 | −25.1 | 0.158 | 0.167 | 0.134 | 0.149 |
| …_035307 | 48 | 0.999965 | −0.008 | −65.7 | −25.8 | 0.145 | 0.143 | 0.086 | 0.082 |
| …_035329 | 48 | 0.999993 | −0.010 | −74.1 | −29.9 | 0.174 | 0.176 | 0.113 | 0.127 |
| …_035352 | 48 | 1.000086 | −0.013 | −81.5 | −34.1 | 0.344 | 0.347 | 0.375 | 0.378 |
| …_035410 | 48 | 1.000048 | 0.002 | 1.2 | 1.9 | 0.178 | 0.177 | 0.140 | 0.134 |
| …_035444 | 49 | 1.000133 | −0.001 | 0.3 | −4.1 | 0.134 | 0.117 | 0.153 | 0.077 |
| …_035455 | 48 | 1.000094 | 0.004 | 1.1 | −6.2 | 0.135 | 0.134 | 0.140 | 0.128 |
| …_035516 | 47 | 0.999986 | −0.011 | −87.9 | −104.1 | 0.178 | 0.177 | 0.107 | 0.106 |
| …_035601 | 48 | 0.999985 | −0.020 | −122.7 | −10.5 | 0.151 | 0.151 | 0.085 | 0.091 |
| …_035612 | 48 | 1.000074 | −0.018 | −125.1 | −12.3 | 0.136 | 0.124 | 0.080 | 0.059 |
| …_035631 | 48 | 1.000097 | −0.006 | −63.1 | −15.3 | 0.138 | 0.147 | 0.145 | 0.224 |
| …_040538 | 48 | 1.000003 | −0.001 | −32.3 | −52.2 | 0.125 | 0.125 | 0.106 | 0.106 |
| …_040549 | 48 | 1.000087 | −0.001 | −32.1 | −54.1 | 0.149 | 0.148 | 0.090 | 0.100 |
| …_040600 | 49 | 1.000077 | −0.003 | −31.3 | −56.1 | 0.165 | 0.159 | 0.110 | 0.086 |
| …_040621 | 48 | 1.000146 | −0.011 | −119.8 | 41.6 | 0.155 | 0.136 | 0.166 | 0.109 |

(`…` = `Light_M 16_10.0s_LP_20250530`; full precision in the script output.)

### 7.3 Aggregate scale statistics — corrected (defect #1)

| n | min | median | max | MAD | range |
|---|---|---|---|---|---|
| 19 | 0.999965 | 1.000074 | 1.000146 | 4.35e-5 | 1.80e-4 |

| statistic | value (ppm) | corner error @1101.5 px (px) |
|---|---|---|
| \|median(scale) − 1\| | 74.1 | 0.0817 |
| median(\|scale − 1\|) | 74.1 | 0.0817 |
| mean(\|scale − 1\|) | 65.6 | 0.0722 |

RF-1B's "median |scale−1| = 24.5 ppm" was `abs(median(scale)−1)` with the
wrong label.  The corrected quantities differ from each other and from RF-1B's
number: `|median(scale)−1| = 74.1 ppm`, `median(|scale−1|) = 74.1 ppm`,
`mean(|scale−1|) = 65.6 ppm`.  (On RF-1B's *uncorrected* preprocessing the same
three quantities were 24.5 / 46.2 / 53.4 ppm — Jarvis's independent computation
— confirming the label/code defect.)  The **corner-pixel implication** is now
stated with the correct "typical per-frame" statistic: `median(|scale−1|) ≈
74 ppm` → **≈ 0.08 px** at the 1101.5 px corner — still far below the ~0.15 px
held-out centroid noise.  The conclusion (scale is fit noise on this session)
is unchanged.

### 7.4 A/B — preprocessing fix does not alter the scale conclusion

| configuration | reference frame | \|median(scale)−1\| (ppm) | median\|s−1\| (ppm) | mean\|s−1\| (ppm) | hold-RMS improvement (median px) |
|---|---|---|---|---|---|---|
| RF-1B (GRBG hardcoded, no hot-pixel) | …_040549 | 24.5 | 46.2 | 53.4 | 0.0018 |
| corrected (header BAYERPAT + hot-pixel) | …_035421 | 74.1 | 74.1 | 65.6 | 0.0011 |

Both configurations give a scale deviation far below the held-out centroid
noise and a negligible, signed-± held-out improvement.  The preprocessing fix
changes the reference frame and the exact ppm, but **not** the conclusion that
the scale is fit noise on this single session.

### 7.5 Held-out improvement and residual levels

| metric | min | median | max | mean |
|---|---|---|---|---|
| hold RMS improvement (eucl − sim, px) | −0.0094 | 0.0011 | 0.0235 | 0.0036 |
| corner improvement (px) | −0.0792 | 0.0008 | 0.0762 | 0.0020 |

| model | median hold RMS | max hold RMS | median hold P95 | max hold P95 |
|---|---|---|---|---|
| euclidean (current) | 0.1513 | 0.3439 | 0.2511 | 0.5781 |
| similarity | 0.1471 | 0.3465 | 0.2444 | 0.5848 |

### 7.6 Hold-out limitation (stated explicitly)

astroalign's matched-pair selection / RANSAC ran on **all** detected stars
**before** the 70/30 fit/hold-out split.  The held-out residuals are therefore
a **model-fit hold-out** (the fit never sees the held-out pairs), **not** a
fully independent correspondence-selection validation: the correspondences were
chosen by astroalign using every star.  This witness measures whether a
similarity scale is *supported by the matched pairs astroalign produced*; it
does not independently validate the matcher.

### 7.7 Diagnosis

The scale residuals are **fit noise**, not a coherent signal: `median(|scale−1|)
= 74.1 ppm` (≈ 0.08 px at the corner, below the ~0.15 px held-out centroid
noise); the held-out improvement from retaining scale is ~0.001 px median and
**negative on some frames** (similarity occasionally *worse* — the classic
noise signature).  Rotation is tiny (−0.021°…+0.004°) and translation drifts
smoothly, consistent with alt-az field rotation + drift over ~23 min —
geometry the Euclidean model already handles.  **Dataset limitation:** one
session only; it cannot establish cross-session behaviour or a stable
thermal scale drift.

---

## 8. Batching-dependence POC (new — the architecture-risk experiment)

Module: `research/registration_field_rotation/batch_dependence_poc.py`;
tests: `tests/test_batch_dependence_poc.py`.

**Question:** does the evolving registration target in `reproject_between_batches`
behave differently from an immutable one?  This is the *behavioural* counterpart
to the structural AST guard (§4), which only proves the variable is reassigned.

**Design:** deterministic synthetic star catalogues on a fixed global grid with
known per-frame similarity transforms (rotation 0→3°, translation drift), 200
stars (70/30 fit/hold-out), centroid noise σ = 0.05 px, closed-form similarity
fit (astroalign's RANSAC matcher abstracted away).  Two strategies on the same
frames/noise:

* **A — immutable target:** every frame registered to one fixed noisy reference
  catalogue `P_true + noise`.
* **B — evolving target:** frames registered sequentially; the reference
  catalogue is rebuilt every `batch_size` frames as the mean of all
  already-registered frames mapped onto the frozen global grid.

**Metrics (global grid):** `hold_resid_target` (fit residual vs the target),
`hold_resid_true` (source→true-global transform error vs ground truth),
`ref_bias` (drift of the reference catalogue from ground truth), plus
centre/edge/corner splits.

### 8.1 Zero-mean centroid noise — no bias propagation

| strategy | batch | hold_resid_target (mean/last) | hold_resid_true (mean/last) | ref_bias (mean/last) |
|---|---|---|---|---|
| A immutable | n/a | 0.092/0.096 | 0.063/0.066 | 0.062/0.062 |
| B evolving (bs=1) | 1 | 0.067/0.066 | 0.063/0.066 | 0.022/0.012 |
| B evolving (bs=5) | 5 | 0.070/0.066 | 0.063/0.066 | 0.026/0.012 |
| B evolving (bs=30, never updates) | 30 | 0.092/0.096 | 0.063/0.066 | 0.062/0.062 |

With zero-mean noise the evolving target **does not** introduce bias: its
reference *converges* to ground truth (ref_bias 0.062 → 0.012 px) and its
true-global error equals the immutable target's (0.063 px).  `batch_size = N`
degenerates to the immutable case (never updates).

### 8.2 Adversarial systematic radial centroid bias (c=4 px at corner) — bias is absorbed and hidden

A barrel-like radial displacement `|bias| = c·(r/r_max)²` (not representable by
a similarity fit) is applied to every frame's centroids.

| strategy | batch | hold_resid_target (mean/last) | hold_resid_true (mean/last) | ref_bias (mean/last) | corner true (last) |
|---|---|---|---|---|---|
| A immutable | n/a | 0.308/0.304 | 0.303/0.301 | 0.062/0.062 | 0.365 |
| B evolving | 1 | 0.074/0.066 | 0.303/0.301 | 0.280/0.287 | 0.365 |
| B evolving | 5 | 0.105/0.066 | 0.303/0.301 | 0.249/0.287 | 0.365 |
| B evolving | 10 | 0.145/0.065 | 0.303/0.301 | 0.212/0.287 | 0.365 |
| B evolving | 30 | 0.308/0.304 | 0.303/0.301 | 0.062/0.062 | 0.365 |

* **A exposes** the bias: its fit residual is 0.308 px (the similarity fit
  cannot absorb the radial field), and its reference stays unbiased (0.062).
* **B absorbs and hides** it: its reference drifts (ref_bias 0.062 → 0.287 px),
  its fit residual collapses to the noise floor (0.066 px), while the
  source→true-global error is **unchanged** (0.303 px).  The bias is hidden,
  not removed.
* **Batch-size dependence:** the drift rate depends on the update cadence
  (ref_bias mean 0.280 px at bs=1 → 0.212 px at bs=10 → 0.062 px at bs=30),
  while the final true-global error is identical.  The trajectory is
  batch-history dependent.
* **Reference-drift trajectory (B, bs=1):** ref_bias = 0.062 (frame 0) →
  0.287 (frames 5, 9, 10, 15, 20, 29) — the drift appears at the first batch
  update and stays.

### 8.3 Order dependence — a transient first-batch bias matters only for B

Only the first 10 frames (one batch at bs=10) carry the bias; the rest are
clean.  Compare natural vs reversed processing order.

| strategy | order | hold_resid_true (mean) | hold_resid_true (last frame) | ref_bias (last) |
|---|---|---|---|---|
| A immutable | natural | 0.142 | 0.066 | 0.062 |
| A immutable | reversed | 0.142 | 0.308 | 0.062 |
| B evolving | natural | 0.142 | 0.066 | 0.144 |
| B evolving | reversed | 0.142 | 0.308 | 0.015 |

The immutable target is **order-independent** (reference stays clean, ref_bias
0.062 both orders).  The evolving target is **order-dependent**: the same
biased frames contaminate the reference when processed first (ref_bias 0.144)
but not when processed last (ref_bias 0.015).  With a similarity fit the
contamination *dilutes* over subsequent clean batches (the unrepresentable bias
cannot be perpetuated by clean frames, which map back to ground truth), so it
does **not** compound without bound — a stated limit, not a proof of unbounded
drift.

### 8.4 POC conclusions and limits

1. **Zero-mean noise does not propagate** through an evolving target.
2. **A systematic (unrepresentable) centroid bias is absorbed into the evolving
   reference and hidden from the fit residual**, making the per-frame fit
   residual an unreliable proxy for true-global accuracy in reproject mode.
3. **The evolving target is batch-size and order dependent** in its reference
   trajectory.
4. **Limit:** with a closed-form similarity fit, an *unrepresentable* bias
   dilutes rather than compounds; a bias the estimator *can* represent (e.g. a
   pure translation/scale, or whatever the Euclidean model absorbs) would be
   perpetuated instead.  The POC proves the architecture is **batch-history
   dependent** and that fit residuals can be misleading, not that a specific
   bias compounds without bound.  It is a POC of the architecture risk, not a
   production worker replacement.

---

## 9. Limitations, RF-2 gate, ZeAlfie impact

### 9.1 Limitations

* Synthetic-only for the model *comparison* and the batching POC; real data is
  one session.
* Robustness protocol is a closed-form + MAD pass (model POC) / closed-form
  similarity fit (batching POC), not astroalign's triangle+RANSAC estimator.
* Resampling study is a single-star sanity check; kernel/PSF/ringing/
  edge-footprint deferred to RF-2.
* Global-reference audit is a static/AST structural path guard (§4) plus the
  batching POC (§8); it does not execute the production `_worker` (see §9.3).
* Microbenchmarks are non-decisive and machine-dependent.

### 9.2 ZeAlfie impact

**ZeAlfie integration impact: NONE.**  No ZeAlfie code read/imported; no
boundary changed.  The only
cross-cutting invariant affected by any *future* scale change is the **HSI
contract** (untouched here): retaining scale would change pixel-level outputs
of the classic warp and the Drizzle footprint, so HSI "plain-classic exactness"
expectations and the documented reprojection approximations must be re-verified
if RF-2 ever retains scale.  The reproject-mode moving-target finding (§4)
likewise has no HSI interaction today, but retaining an immutable reference or
persisting per-frame transforms is a prerequisite for any future HSI closure
over per-frame maps.

### 9.3 Exact RF-2 gate (recommendation — smallest change, not a rewrite)

**Scale (FURTHER DATA REQUIRED path):**
1. **Measurement (no behaviour change):** log `transform_skimage_obj.scale` per
   frame in `alignment.py:228-237`.
2. **Cross-session evidence:** run the M16-scale witness (§7) on ≥2 more
   sessions spanning temperature/focus; establish the scale range and whether
   it is coherent or noise.
3. **Gated minimal change (only if |scale−1| consistently ≳ 1e-4):** stop
   forcing `scale = 1.0`.  ~4 lines in `alignment.py`.  No Drizzle change.

**Moving-target (from §4/§8, if independence/provenance is required):**
4. **Smallest RF-2 production POC:** retain an **immutable high-SNR reference
   target** for alignment (e.g. keep the initial reference image for all frames,
   or a first-batch high-SNR stack frozen as the alignment target), **or**
   persist each frame's registration solution (record the per-frame matrix and
   the reference identity used).  This is bookkeeping / a target-selection
   change, **not** a rewrite.  Do **not** proceed to a full reproject re-design.
5. **Behavioural test:** with `reproject_between_batches=True` and ≥2 batches,
   drive `_worker` on 2 synthetic batches (or mock `_process_file`/`_stack_batch`)
   and assert each frame's alignment target identity (`initial_reference` vs
   `cumulative_stack_N`); with `reproject_between_batches=False`, assert all
   frames use `initial_reference`.  This directly proves the moving vs immutable
   behaviour without a brittle source-structure test.

---

## 10. Commands, results, dependencies

Environment (`.venv`): astroalign **2.6.2**, scikit-image **0.26.0**, opencv
**5.0.0**, numpy **2.5.2**, astropy **8.0.1**, drizzle **2.2.0**, reproject
**0.21.0**, sep **1.4.1**, scipy **1.18.1**.

| Command | Result |
|---|---|
| `python research/registration_field_rotation/model_selection_poc.py` | full measured report (tables in §2–§5), exit 0 |
| `python research/registration_field_rotation/m16_scale_witness.py` | M16 witness report (§7), exit 0 |
| `python research/registration_field_rotation/batch_dependence_poc.py` | batching POC report (§8), exit 0 |
| production-vs-witness A/B (`_load_normalized`, `debayer_image`, `detect_and_correct_hot_pixels`) | bit-identical (maxdiff 0.0) |
| `python -m pytest tests/test_registration_model_selection_poc.py -q` | **15 passed** |
| `python -m pytest tests/test_global_reference_audit.py -q` | **11 passed** |
| `python -m pytest tests/test_m16_scale_witness.py -q` | **7 passed** |
| `python -m pytest tests/test_batch_dependence_poc.py -q` | **9 passed** |
| `python -m pytest tests/test_drizzle_core.py tests/test_interbatch_classic.py tests/test_solver_port.py -q` | **20 passed** (baseline unchanged) |

Skipped / unavailable: full test suite (not run, per instructions); ZeSolver
private API and sibling repositories (not in scope); M16 is read-only.

---

## 11. Files changed

* `research/registration_field_rotation/m16_scale_witness.py` (modified) —
  corrected scale statistics (defect #1), header-BAYERPAT + hot-pixel
  correction (faithful, verified bit-identical), A/B, explicit hold-out
  limitation, `scale_statistics()` helper.
* `research/registration_field_rotation/batch_dependence_poc.py` (new) —
  batching-dependence POC (direct vs evolving target).
* `research/registration_field_rotation/model_selection_poc.py` (label only) —
  "corrective B" → "corrective C".
* `tests/test_global_reference_audit.py` (modified) — branch-polarity fix,
  exact mutation-line pin, snippet tests; described as a structural path guard.
* `tests/test_m16_scale_witness.py` (new) — corrected-statistic + faithful-
  preprocessing tests + full-witness integration (skipped if data absent).
* `tests/test_batch_dependence_poc.py` (new) — pins the batching POC conclusions.
* `docs/registration_field_rotation_research.md` (this file, rewritten).
* `docs/registration_field_rotation_state.md` (updated).

**No file under `seestar/` was modified.**
