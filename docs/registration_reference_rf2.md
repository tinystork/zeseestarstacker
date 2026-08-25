# RF-2 — registration-reference architecture audit + bounded production-path POC

**Task:** RF2-01 corrective iteration C1 (research gate only) — measured
evidence for the registration-reference architecture *before* any production
change.
**Branch:** `feature/registration-field-rotation`
**HEAD:** `cf8d25ceeb72f0feade8289da8b405606dbecb5b` (clean, equal to origin)
**Parent HSI scientific contract:** `61291aa035562f09a2bd7cf7971ef8192e1585bf`
**Scope:** production-external. **No file under `seestar/` was changed.**

This report is a *research gate*, not a production decision.  The architectural
recommendation in §11 is **explicitly a recommendation for Jarvis review, not
closure**.

---

## 1. BASELINE

| Field | Value |
|---|---|
| Branch | `feature/registration-field-rotation` |
| HEAD | `cf8d25ceeb72f0feade8289da8b405606dbecb5b` |
| vs origin | clean, equal |
| Parent (HSI contract) | `61291aa035562f09a2bd7cf7971ef8192e1585bf` |
| Working tree at start | clean (only RF-2 additions at end) |
| RF-1 status | research-only in `cf8d25c`; history not rewritten |

Environment (`.venv`): astroalign **2.6.2**, scikit-image **0.26.0**, opencv
**5.0.0**, numpy **2.5.2**, astropy **8.0.1**, drizzle **2.2.0**, python **3.13.5**.

---

## 2. CURRENT_TARGET_LIFECYCLE (evidenced, not inferred)

### 2.1 Coordinate-system / target-image / accumulator / WCS separation

Four distinct objects are kept separate throughout the worker; they must not be
conflated:

| Concept | What it is | Where | Immutable? |
|---|---|---|---|
| **Coordinate grid (WCS)** | the frozen pixel<->sky grid | `reference_wcs_object`, `ref_wcs_header`, `reference_header_for_wcs` | **YES** in reproject mode (`freeze_reference_wcs`) |
| **Registration target image** | image data fed to `find_transform` | `reference_image_data_for_global_alignment` | **NO** in reproject mode (replaced by cumulative stack) |
| **Stack accumulator** | `sum` / `wht` memmaps of aligned pixels | `cumulative_sum_memmap`, `cumulative_wht_memmap` | grows monotonically |
| **WCS/astrometric reference** | the solved astrometric solution | `reference_wcs_object` | **YES** once frozen |

### 2.2 Reference lifecycle table (exact anchors)

| line (queue_manager.py) | event |
|---|---|
| `5235` | `reference_image_data_for_global_alignment = None` initialiser |
| `5419-5426` | `= self.aligner._get_reference_image(...)` — **initial reference** (single auto/manual frame) |
| `2987` | `self.freeze_reference_wcs = self.reproject_between_batches` |
| `5927-5936` | batch-plan `_BATCH_BREAK_TOKEN` → `_flush_current_batch(...)` **no-op reassignment** (classic branch returns input unchanged; reproject branch inside is dead `if False and ...`) |
| `6147` / `6243` | worker-path `if self.reproject_between_batches:` (positive guard) |
| `6247-6252` | `reference_image_data_for_global_alignment = stack_img` (from `_solve_cumulative_stack`) |
| `6259` / `6266` | `= stacked_np.astype(...)` (fallback / dead inner-else) |
| `6746-6759` | finalize last-partial-batch path, same replacement (6749/6754/6759) |
| `11877+` | `_solve_cumulative_stack()` returns `sum/wht` mean; **skips re-solve** when `freeze_reference_wcs` and `reference_wcs_object` present |
| `17739+` | `_add_frame_to_drizzle_accumulators(original_data, header, tf, weight_map, native_wcs)` |

### 2.3 Mode table

| Mode | flags | Reference target | Per-frame fit | Transform persisted? |
|---|---|---|---|---|
| Plain classic | `reproject_between_batches=False`, `drizzle_active_session=False` | initial single frame (immutable) | `_align_image` (Euclidean) | no (classic) |
| M3 Drizzle standard | `drizzle_active_session=True`, `reproject_between_batches=False` | initial single frame (immutable) | `_align_image(return_M=True)` → `tf` fed to drizzle | **yes, the `tf` is consumed immediately** (not persisted to disk) |
| `reproject_between_batches` (classic or drizzle) | `reproject_between_batches=True` | **evolving cumulative stack** (replaced each batch) | `_align_image` (Euclidean) | no (classic reproject path) |
| `reproject_coadd_final` | `reproject_coadd_final=True`, `reproject_between_batches=False` | initial single frame (immutable) | `_align_image` (Euclidean) | no |

### 2.4 The two facts that matter

1. **Grid frozen, data evolving.**  `freeze_reference_wcs` keeps the WCS grid
   fixed, but `reference_image_data_for_global_alignment` is replaced by the
   cumulative stack at every batch boundary (lines 6252 / 6749).  Each frame is
   still fit **directly** to the *current* target, so the fitted per-frame
   transform maps original source pixels straight into the frozen grid — there
   is no explicit transform composition — but the *target* (and therefore the
   matched centroid field) changes every batch.

2. **Per-frame transform not persisted** in the classic reproject path; the
   Drizzle standard path computes the 2×3 `tf` (via `return_M=True`) but uses it
   immediately (never written), and the classic path never retains it.

**Key consequence (the scientific contract at stake):** the *evolving* target's
identity is a function of the batch stream — it depends on `batch_size` and on
*which* frames arrive in which order.  A target with that property cannot yield
organization-independent per-frame transforms.

---

## 3. POC_DESIGN (candidate selection / rejection)

### 3.1 Candidate set

| Candidate | Description | Organization independence | Verdict |
|---|---|---|---|
| **`immutable` (selected reference)** | the initially-selected single frame from `_get_reference_image` (manual or auto-best), held constant for the whole run | **YES by construction** — identity is a function of `(frame selection)` only, never of `batch_size` or order | **SELECTED (primary stable candidate)** |
| `evolving` (current production) | cumulative stack rebuilt at every batch boundary | NO — identity depends on the batch stream | retained as the current baseline |
| `freeze_first_batch` (previous "stable high-SNR") | freeze the mean of the first `batch_size` frames | NO — identity depends on `batch_size` and first-batch content | **REJECTED (explored)** |

### 3.2 Why the previous candidate was rejected

The previous iteration proposed a "stable first-batch high-SNR reference" and
argued it was batch-size invariant "after freeze".  That is **false** as a
batch/order-invariance claim: the reference's *identity and bias* depend on
`batch_size` and on which frames happen to arrive first.  The transient-radial
table in §4.4 proves it directly — the same candidate shows `ref_bias` of
**2.928 px** (biased-first) vs **0.021 px** (biased-last).  "Constant after
freeze" is a *within-run* property; it is not *across-decomposition* invariance.
A target whose build reads the batch identity cannot be organization-independent.

### 3.3 Experiment construction (production-path-bounded)

Four layers of evidence, each bounded to what it can prove:

1. **`registration_lifecycle.py`** — deterministic closed-form harness reproducing
   the worker's registration-reference *data contracts* 1:1: the production
   **Euclidean estimator** (similarity fit → discard scale, replicating
   `alignment.py:228-237`), the reference-replacement seam, and the 2×3/3×3
   source→global transform contract.  The heavy astroalign matching and WCS
   solving are abstracted (stated limit); the scale-discard arithmetic is
   **faithful/equivalent** (not claimed bit-identical — see §14).  Strategies:
   `immutable` (selected), `evolving` (production), `freeze_first_batch`
   (rejected).  Metrics: `fit_resid`, `true_err` (vs ground truth), `ref_bias`,
   centre/edge/corner, runtime, failure rate, **and per-frame transforms keyed by
   frame ID** (the geometry evidence).

2. **`production_seam_witness.py`** — runs the **real** `SeestarAligner._align_image`
   with only `astroalign.find_transform` monkeypatched to a known transform,
   proving the real scale-discard and the real `return_M` 2×3 contract.

3. **`m16_target_policy_witness.py`** *(new, corrective C1)* — bounded real M16
   witness that compares the **immutable selected reference** vs the **evolving
   target** on **actual prepared M16 pixels** using the **real astroalign matcher**
   and the **production Euclidean conversion**, across ≥3 batch sizes and ≥2
   deterministic orders, measuring per-frame transform displacement at
   centre/edge/corner.  **Observational only — no ground truth** (see §5 and §14).

4. **`m16_witness_rf2.py`** — bounded real M16 diversity/noise witness (geometry
   spread + observable-noise level), explicitly **not** ground truth.

### 3.4 Bias scenarios

* `zero_mean` — no systematic bias (propagation test).
* `translation` (representable, 0.5 px) — does the Euclidean model correct it?
* `radial` (non-representable, c=4 px quadratic) — the hidden-bias test.
* `transient radial` (first 10 frames only) — order dependence + representability.

---

## 4. SYNTHETIC_RESULTS

Deterministic; N=30 frames, M=200 stars, 70/30 fit/hold-out, σ=0.05 px, fixed
seed.  Production Euclidean estimator in all tables.  The **immutable selected
reference** is the primary stable candidate; `freeze_first_batch` is the
explored-rejected candidate.

### 4.1 Scenario 1 — zero-mean noise (batch sizes on the same observations)

| strategy | batch | fit P50 | fit P95 | true P50 | true P95 | ref_bias last |
|---|---|---|---|---|---|---|
| immutable (selected ref) | n/a | 0.093 | 0.102 | 0.063 | 0.072 | 0.062 |
| freeze_first_batch (rejected) | 1 | 0.084 | 0.093 | 0.063 | 0.072 | 0.060 |
| freeze_first_batch (rejected) | 5 | 0.071 | 0.092 | 0.063 | 0.072 | 0.029 |
| freeze_first_batch (rejected) | 10 | 0.069 | 0.098 | 0.063 | 0.072 | 0.021 |
| evolving | 1 | 0.066 | 0.081 | 0.063 | 0.072 | 0.012 |
| evolving | 5 | 0.067 | 0.092 | 0.063 | 0.072 | 0.012 |
| evolving | 10 | 0.068 | 0.098 | 0.063 | 0.072 | 0.013 |
| evolving | 30 | 0.093 | 0.102 | 0.063 | 0.072 | 0.062 |

### 4.2 Scenario 2 — representable translation bias (0.5 px, every frame)

| strategy | fit P50/P95 | true P50/P95 | true mean | ref_bias last |
|---|---|---|---|---|
| immutable | 0.093/0.102 | 0.063/0.072 | 0.063 | 0.062 |
| freeze_first_batch (rejected) | 0.084/0.093 | 0.063/0.072 | 0.063 | 0.060 |
| evolving | 0.066/0.081 | 0.063/0.072 | 0.063 | 0.012 |

### 4.3 Scenario 3 — non-representable radial bias (c=4 px, every frame)

| strategy | fit P50/P95 | true P50/P95 | ref_bias last | centre/edge/corner true (last) |
|---|---|---|---|---|
| immutable | 2.714/2.728 | 2.715/2.729 | 0.062 | 2.729/2.610/2.998 |
| freeze_first_batch (rejected) | 0.084/0.097 | 2.718/2.732 | 2.931 | 2.735/2.614/3.004 |
| evolving | 0.066/0.081 | 2.718/2.732 | 2.930 | 2.735/2.614/3.004 |

### 4.4 Scenario 4 — transient first-batch radial bias, natural vs reversed order

| strategy | order | fit P50 | true P50 | true mean | ref_bias last |
|---|---|---|---|---|---|
| immutable | natural | 0.098 | 0.067 | 0.948 | 0.062 |
| immutable | reversed | 0.098 | 0.067 | 0.948 | 0.062 |
| freeze_first_batch (rejected) | natural | 2.716 | 0.073 | 0.951 | 2.928 |
| freeze_first_batch (rejected) | reversed | 0.095 | 0.067 | 0.948 | 0.021 |
| evolving | natural | 2.708 | 0.070 | 0.950 | 1.464 |
| evolving | reversed | 0.095 | 0.067 | 0.948 | 0.014 |

### 4.5 Scenario 5 — BATCH_INVARIANCE (transforms, not image similarity)

Same preselected reference identity; only the batch decomposition varies.
Geometry invariance measured on per-frame transform matrices keyed by frame ID.

**ΔM units:** ``max |ΔM|`` is the maximum absolute element difference of the
3×3 transform matrices — a unitless matrix-element quantity, **not**
pixel-equivalent.  Pixel-space displacement is reported separately for the real
M16 witness in §5.2 (centre/edge/corner, px).

| strategy | batch sizes | max \|ΔM\| across batch sizes | invariant? |
|---|---|---|---|
| immutable (selected ref) | 1 / 5 / 10 | 0.000e+00 | **YES** |
| freeze_first_batch (rejected) | 1 / 5 / 10 | 1.170e-02 | NO |

### 4.6 Scenario 6 — ORDER_INVARIANCE (transforms keyed by frame ID)

Same preselected reference identity; natural vs reversed order.

| strategy | orders | max \|ΔM\| natural vs reversed | invariant? |
|---|---|---|---|
| immutable (selected ref) | natural / reversed | 0.000e+00 | **YES** |
| freeze_first_batch (rejected) | natural / reversed | 3.181e-02 | NO |
| evolving | natural / reversed | 3.183e-02 | NO |

### 4.7 Runtime & failure rate

| strategy | scenario | runtime_s | failure_rate |
|---|---|---|---|
| immutable (zero-mean) | — | 0.025 | 0.0 |
| freeze_first_batch (zero-mean) | — | 0.033 | 0.0 |
| evolving (zero-mean) | — | 0.026 | 0.0 |
| evolving (radial) | — | 0.033 | 0.0 |
| freeze_first_batch (radial) | — | 0.038 | 0.0 |

---

## 5. M16_TARGET_POLICY_RESULTS (real pixels, observational — no ground truth)

Real astroalign matcher + production Euclidean conversion (`alignment.py:228-237`).
Immutable target = the auto-selected reference frame (production quality metric
`median/(1.4826·MAD)` = 24.113).  Evolving target = cumulative mean of warped
green channels (deviation from production RGB memmap `sum/wht` documented in §14).
19 non-reference frames, green shape 1920×1080, batch sizes {1,5,10}, orders
{natural, reversed}.

### 5.1 Per-configuration summary

| policy | order | bs | aligned | failed | runtime_s | target rebuilds | target-fit residual P50/P95 (px) |
|---|---|---|---|---|---|---|---|
| evolving | natural | 1 | 19 | 0 | 16.4 | 18 | 0.112/0.245 |
| evolving | natural | 5 | 19 | 0 | 12.9 | 3 | 0.126/0.239 |
| evolving | natural | 10 | 19 | 0 | 11.7 | 1 | 0.120/0.261 |
| evolving | reversed | 1 | 19 | 0 | 19.2 | 18 | 0.114/0.280 |
| evolving | reversed | 5 | 19 | 0 | 15.7 | 3 | 0.114/0.264 |
| evolving | reversed | 10 | 19 | 0 | 13.2 | 1 | 0.113/0.256 |
| immutable | natural | n/a | 19 | 0 | 10.0 | 0 | 0.129/0.251 |
| immutable | reversed | n/a | 19 | 0 | 10.5 | 0 | 0.129/0.251 |

### 5.2 Organisation sensitivity (point displacement, px; P50/P95/max over frames)

Displacement of the fitted per-frame transform at canonical points across
configurations of the *same* policy.  Zero = organization-independent target.

| comparison | centre P50/P95/max | edge P50/P95/max | corner P50/P95/max |
|---|---|---|---|
| immutable × batch sizes | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 |
| immutable × order | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 |
| evolving × batch sizes | 0.018/0.070/0.101 | 0.018/0.082/0.105 | 0.021/0.069/0.101 |
| evolving × order | 0.030/0.075/0.090 | 0.034/0.090/0.097 | 0.033/0.076/0.091 |
| immutable vs evolving (same frame set) | 0.000/0.043/0.052 | 0.000/0.049/0.053 | 0.000/0.048/0.053 |

### 5.3 astroalign determinism floor

The immutable policy's reversed pass recomputes every transform against the same
fixed reference; any non-zero dispersion there would be the RANSAC repeatability
floor.  It is **exactly zero** on this session (astroalign is empirically
deterministic on these frames despite its unseeded `default_rng` shuffle).

### 5.4 M16 diversity/noise complement (`m16_witness_rf2.py`)

| quantity | value |
|---|---|
| aligned OK / frames | 19 / 20 |
| scale range | 180.3 ppm |
| median \|scale−1\| | 74.1 ppm |
| rotation span | 0.0237 deg |
| median \|rotation\| | 0.0086 deg |
| translation span (x / y) | 126.3 / 145.7 px |
| held-out (target-fit) residual median / max | 0.151 / 0.344 px |

---

## 6. BATCH_INVARIANCE verdict

**Only the immutable selected reference is batch-size invariant; geometry is
measured on transforms, not final-image similarity.**

* The immutable target's per-frame transforms are **identical** (max |ΔM| = 0,
  unitless matrix-element difference — not px)
  across batch sizes 1/5/10 — its identity does not reference `batch_size`.
* The `freeze_first_batch` target's transforms differ across batch sizes
  (max |ΔM| = 1.17e-2 synthetic, unitless) because its identity *is* a function of
  `batch_size`.
* On real M16 pixels, the evolving target shows non-zero batch sensitivity
  (corner P50 0.021 px, max 0.101 px) while the immutable target shows exactly
  zero (§5.2).
* The **true-global error is batch-size independent in every strategy** — it is a
  per-frame property of the estimator, not of the reference.

---

## 7. ORDER_INVARIANCE verdict

**Only the immutable selected reference is order-invariant.**

* Immutable: per-frame transforms keyed by frame ID are identical across natural
  vs reversed order (max |ΔM| = 0, unitless matrix-element difference), synthetic
  and real.
* `freeze_first_batch`: order-dependent through the first batch (3.18e-2 synthetic).
* Evolving: order-dependent through the whole history (3.18e-2 synthetic; corner
  P50 0.033 px on real M16).
* Mean true-global error is identical across orders for every strategy (same set
  of biased frames).

---

## 8. BIAS_OBSERVABILITY verdict

**The fit residual against a drifting/frozen target is an unreliable proxy for
true-global accuracy; only the immutable single-frame target keeps fit residual
== true error.**

* **Representable bias (translation) is *corrected*** by the Euclidean model
  (true error stays at the 0.063 px noise floor) — not a risk.
* **Non-representable bias (radial) is *hidden*, not removed, by any target that
  absorbs it**: the reference drifts to absorb it (ref_bias → ~2.9 px), the fit
  residual collapses to the noise floor (0.066 px), while the true-global error
  stays at the bias magnitude (~2.7 px).  The immutable target exposes it
  (fit residual == true error == 2.7 px).

**Bounding caveat:** the observability conclusion above is bounded to the
*tested* systematic-bias construction (a quadratic radial distortion shared by
all frames).  It does **not** claim that *every* distortion common to reference
and sources becomes observable via the fit residual; only the representable vs
non-representable distinction under the Euclidean estimator was tested (three
bias shapes: translation, rotation, quadratic radial — see §14).

This is why the immutable target's *slightly higher* target-fit residual on M16
(0.129 px vs ~0.11–0.12 px) is **not a cost**: it is the honest signal, not an
accuracy loss.  The single-frame reference is noisier than a stack, but that
noise is *observable*, whereas the stack's lower residual can hide absorbed bias.

---

## 9. RESUME_CONTRACT finding

Resume is **fail-closed for every non-plain-classic mode** (`HSI-2B`):

* `_is_plain_classic()` (queue_manager.py:11991+) returns True **only** for
  plain classic SUM/W (no mosaic, no drizzle, no reproject_between_batches, no
  reproject_coadd_final).
* `initialize()` (3527-3535) fails closed when resume artifacts are present in a
  non-resumable mode ("Reprise impossible … Utilisez un dossier de sortie vide"),
  instead of silently recreating/overwriting the accumulators.
* `_solve_cumulative_stack()` (11877+) skips re-solving under `freeze_reference_wcs`
  (grid stays frozen) — the *grid* is resumable-stable, but the *target image*
  is not covered by the two memmaps, which is exactly why reproject is not
  resumable.

**Finding:** the reproject path's scientific state (the evolving target image +
its provenance) is not captured by the resume manifest, so resume correctly
refuses it.  This is consistent with the §8 finding that the target's evolution
is the lossy part of the state.

---

## 10. DRIZZLE_PREWARP_AUDIT — classification DEAD

**Overall classification of the computed pre-warp image: `DEAD`.**

The Drizzle standard path computes a `cv2.warpAffine` (`aligned_img_astroalign`)
that is **thrown away before the frame is fed to the Drizzle kernel**.  The
`return_M` 2×3 `tf`, the original prepared pixels, and the original validity mask
are each **separately REQUIRED**; the warped image is required by none of them.

Evidence (exact anchors in `queue_manager.py`):

1. `8631` — `_want_M = is_drizzle_or_mosaic_mode and not self.is_mosaic_run`.
2. `8632-8645` — `_align_image(..., return_M=True)` **fully executes
   `cv2.warpAffine`** (`alignment.py` `_align_cpu`/`_align_cuda`) to produce
   `aligned_img_astroalign`, *and* returns the 2×3 `M_astroalign`.
3. `8663` — `data_final_pour_retour = aligned_img_astroalign.astype(np.float32)`
   (the warped image is initially kept).
4. `8689` — `matrice_M_calculee = M_astroalign` (the tf is retained).
5. `8856-8858` — `data_final_pour_retour =
   image_for_alignment_or_drizzle_input.astype(np.float32)` — the warped image is
   **replaced by the original prepared pixels** before return.
6. `6305-6310` — worker feeds `_add_frame_to_drizzle_accumulators(processed_data,
   header_orig, matrix_M_val, valid_mask_val, native_wcs=...)`; `17739+` consumes
   **original pixels + tf + original validity mask** (no warp on data).

**Classification of each piece:**

| Piece | Classification | Evidence |
|---|---|---|
| `cv2.warpAffine` result (`aligned_img_astroalign`) in Drizzle standard | **DEAD** (computed, then discarded at 8856-8858) | `drizzle_core.pixmap_from_alignment` maps original pixel centres through `tf`; data is never warped |
| `return_M=True` → 2×3 `tf` | **REQUIRED** (consumed by `pixmap_from_alignment` → `drizzle.resample`) | `_add_frame_to_drizzle_accumulators` (17739+) |
| original prepared pixels | **REQUIRED** (the drizzle input) | 8856-8858 + `DrizzleAccumulator.add` |
| original validity mask | **REQUIRED** (weight map) | `_process_file` mask computed on original (8708-8741) |

**Scope statement:** this audit covers the standard Drizzle branch
(`is_drizzle_or_mosaic_mode` and not `self.is_mosaic_run`, i.e. the
non-mosaic Drizzle path) and its *alternative* return paths — the classic
alignment branch (`else` at 8647-8657, `return_M=False`) never computes the
`_want_M` warp, and the mosaic branch (`is_mosaic_run=True`) does not set
`_want_M` (8631), so the warp-then-discard pattern is specific to the standard
Drizzle branch.  The **reproject/native-WCS** alternative inside the mosaic
`astrometry_per_panel` path uses `_calculate_M_from_wcs` (8514) rather than a
warp; it is a different code path and does not produce a discarded warp.  A full
mosaic-branch audit is out of scope for this gate (noted in §14).

The dead warp was a research finding at this gate; it has since been removed in
the RF2-02 production implementation via the ``transform_only`` / skip-resampling
contract (see ``docs/registration_reference_rf2_production.md``).  This research
report's §10 remains the audit evidence; the removal itself is a production
change outside this (research) gate.

---

## 11. ARCHITECTURAL_DECISION (accepted and implemented in RF2-02)

**Decision label: `STABLE REGISTRATION TARGET REQUIRED`** — accepted by Jarvis
after independent review and implemented as the RF2-02 production change (see
``docs/registration_reference_rf2_production.md``).

The decision turns on **reproducibility, observability and provenance**, not
accuracy — the measurements show changing the target does **not** change
true-global accuracy (the error is a per-frame property of the Euclidean
estimator).

* The **immutable selected reference** is the **only** organization-independent
  target: per-frame transforms are identical across batch sizes and orders
  (max |ΔM| = 0, unitless, synthetic §4.5–4.6 and real M16 §5.2).  A rerun can
  therefore be reconstructed from the retained target + sources + settings, but
  the per-frame transforms are **not persisted** to disk (they are recomputed per
  run); the passive RF2 diagnostics record the applied rotation/translation and
  (optionally) the residual, not a resumable transform store (the prerequisite
  RF-1 §9.3 flagged for HSI).
* It is the **only** target that preserves bias observability (fit residual ==
  true error, §8).
* Its real-witness behaviour is acceptable: 19/19 frames align, 0 failures,
  target-fit residual 0.129 px (only ~0.017 px above the evolving target's —
  an honest single-frame-noise signal, not an accuracy loss).

**Rejected:** `CURRENT EVOLVING TARGET RETAINED` — the evolving target has no
surviving advantage: it keeps changing the reference (batch/order dependent) and
its fit residual is the least reliable diagnostic.
**Rejected:** `HYBRID STRATEGY REQUIRED` — no hybrid was tested in this gate
(see §14 follow-up 1); a hybrid must not be recommended untested.
**Not selected:** `FURTHER EVIDENCE REQUIRED` — the invariance question is
resolved (immutable is uniquely invariant, and its real-witness behaviour is
acceptable); the *residual* open question (scale/temperature diversity across
sessions) is a separate RF-1 question that does **not** block the target-identity
decision.

**Documented residual (must be tracked):** the immutable target does **not**
restore bias observability for *hidden* non-representable distortions — it
*exposes* them (fit residual == true error), which is exactly the observability
we want, but a drift that a richer model could correct would still show up as a
large residual.  That is an *estimator* issue (Euclidean vs richer models — an
explicit RF-2 non-goal).

---

## 12. CHANGED_FILES

**Total: 8 files** (the 7 pre-existing RF-2 files + 1 newly added witness).

Modified (6):

* `research/registration_reference_rf2/registration_lifecycle.py` — corrected:
  immutable selected reference promoted to primary stable candidate;
  `freeze_first_batch` retained as rejected candidate; per-frame transforms
  keyed by frame ID added; batch/order invariance scenarios (§4.5–4.6);
  scale-discard arithmetic relabelled faithful/equivalent.
* `research/registration_reference_rf2/m16_witness_rf2.py` — docstring/role
  corrected (diversity/noise complement; language softened).
* `tests/test_rf2_lifecycle_poc.py` — rewritten (13 → 18 tests) with corrected
  strategy names and the new transform-invariance tests.
* `tests/test_rf2_production_seam.py` — extended (5 → 10 tests) with the M16
  target-policy witness helper + integration tests.
* `docs/registration_reference_rf2.md` — this report (corrected).
* `docs/registration_reference_rf2_state.md` — summary updated.

Added (1):

* `research/registration_reference_rf2/m16_target_policy_witness.py` — **new**
  real M16 target-policy witness (immutable vs evolving, real matcher/estimator).

Unchanged (1, retained):

* `research/registration_reference_rf2/production_seam_witness.py` — no change
  needed (real `_align_image` scale-discard + `return_M` witness).

**No `seestar/` file touched; no commit; no push.**

---

## 13. TESTS_RUN (exact commands and outcomes)

| Command | Result |
|---|---|
| `python research/registration_reference_rf2/registration_lifecycle.py` | exit 0, §4 tables |
| `python research/registration_reference_rf2/m16_target_policy_witness.py` | exit 0, §5 tables (~2m04s) |
| `python research/registration_reference_rf2/m16_witness_rf2.py` | exit 0, §5.4 (~25s) |
| `python research/registration_reference_rf2/production_seam_witness.py` | exit 0, scale→1.0/det→1.0/rot→2.0°/t preserved |
| `python -m pytest tests/test_rf2_lifecycle_poc.py tests/test_rf2_production_seam.py -q` | **28 passed** |
| `python -m pytest tests/test_batch_dependence_poc.py tests/test_global_reference_audit.py tests/test_m16_scale_witness.py tests/test_registration_model_selection_poc.py -q` | **42 passed** |

Full suite: **not run** (per instructions).  No failures; only pre-existing
SciPy `scipy.ndimage.filters` deprecation warnings from `colour_demosaicing`
(unrelated to RF-2).

Final `git status`: only the eight RF-2 files above; **no `seestar/` modification
and no unrelated edits**.

---

## 14. LIMITATIONS / FOLLOW-UPS

* Synthetic catalogues + closed-form similarity fit; the astroalign
  triangle/RANSAC matcher and ASTAP WCS solving are abstracted in the harness
  (the matcher cannot change the scale-discard or the reference-evolution
  *contract*, which are what the harness pins).
* **Scale-discard arithmetic is faithful/equivalent, not claimed bit-identical**
  (corrective defect #7): production applies it to `float32` astroalign params;
  the harness applies the same arithmetic to `float64` closed-form params, which
  can differ at floating-point rounding.  The real-seam witness
  (`production_seam_witness.py`) exercises the actual `float32` path.
* The M16 evolving-target emulation rebuilds the target as the **mean of warped
  green channels**, not the production RGB memmap `sum/wht` + WCS re-solve; it
  preserves the identity semantics (cumulative aligned stack) but is **not
  worker-equivalent** (documented deviation).
* M16 is one session: **no ground truth**, no cross-session/temperature scale
  data.  The real-pixel results are observational organization-sensitivity
  proxies, not accuracy measurements.  They are correctly distinguished from the
  ground-truth synthetic results.
* Only three bias shapes explored (translation, rotation, quadratic radial).
* Drizzle audit is scoped to the standard (non-mosaic) Drizzle branch; the mosaic
  `astrometry_per_panel` WCS path is noted but not audited in full.

**Follow-ups (recommended, not part of this gate):**

1. If a hybrid is ever considered (e.g. immutable target + a secondary
   never-used-for-warp diagnostic anchor), it must be built and measured before
   recommendation — do not recommend an untested hybrid.
2. Run the M16 target-policy witness on ≥2 more sessions spanning temperature/
   focus to confirm the evolving target's organization sensitivity is
   representative (the scale question, RF-1 §9.3, remains open).
3. If a `return_M`-only alignment path is ever wanted, remove the dead
   `warpAffine` on the Drizzle standard path (§10) — a production change outside
   this gate.
