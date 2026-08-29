# M3 Drizzle Photometric Integrity & Coverage Contract (ZSSS-DPIC-01)

Status: **implemented, uncommitted (awaiting architect review)** — 8.1.0 / Phoenix consedit.

This document is the focused architecture reference for the M3 Drizzle
*original-sample photometric integrity* and *coverage* contract.  It describes
the design implemented by task ZSSS-DPIC-01, its exact terminology, its
scientific justification, and its explicit boundaries (in particular what is
**not** implemented: full persistent Drizzle SCI/WHT checkpoint/resume — later superseded by RSM2).

---

## 1. Original-sample lifecycle

Every accepted frame is processed exactly once, in a fixed order, through the
single per-channel Drizzle accumulator path (`_add_frame_to_drizzle_accumulators`):

```
original pose (load -> debayer -> hot pixels -> ADU rescale)
   │   (H, W, 3) float32, never warped / reprojected
   ▼
[1] additive background match (this task)   <- constant per-channel offset
   ▼
[2] single Drizzle deposition               <- DrizzleAccumulator.add
   ▼
SCI / WHT (immutable contribution)
```

* The **geometry** of a frame is a 2x3 affine `tf` (rotation + translation,
  scale 1.0) mapping ORIGINAL pixel centres to the reference grid.
* `pixmap_from_alignment` applies `tf` only to pixel *centres* (through the
  reference WCS into the output WCS).  No `warpAffine` is applied to the data.
* The deposited science samples therefore stay on their original pixel grid.
  Only the *measurement* path (estimating a correction) may interpolate.

---

## 2. Immutable additive background matching (not generic subtraction)

`seestar/core/drizzle_background.py` implements a **constant per-channel
additive** correction only.  It is deliberately *not* generic background
subtraction, *not* a spatial background model, and *not* a reprojection.

Goal: remove the spatial "mean step" that Drizzle's weighted mean develops when
the set of contributing frames changes (partial overlap, dithering, field
rotation) and those frames carry different additive sky backgrounds.

Correction model, per channel `c`:

```
frame_corrected[c] = frame[c] - offset[c]
```

`offset[c]` is estimated so that the frame's background matches the anchor's
background; a constant shift therefore leaves every gradient / source structure
(nebula, extended object, star wings) untouched — it only moves the zero level.

### 2.1 The anchor

`BackgroundAnchor` is captured **once** per run and never mutated:

* from the **immutable registration reference** returned by
  `aligner._get_reference_image` in `_worker`, captured **before any frame
  deposition** (`_capture_reference_drizzle_bg_anchor`);
* the anchor is labelled `reference:<id>` with `id` = the exact
  `HIERARCH SEESTAR REF SRCFILE` identifier (never derived from frame
  arrival/admission order);
* it stores the anchor pixel data `(H, W, C)` (float32) plus its geometry
  `tf` (anchor pixels -> reference grid; **identity** for the registration
  reference) and the reference-grid shape.

The reference returned by `_get_reference_image` is a debayered,
hot-pixel-corrected RGB image in `[0, 1]`.  Crucially, `_get_reference_image`
does **not** apply the per-frame "WB basique" R/B gains that `_process_file`
applies after debayering, so the anchor capture applies the **same shared
helper** (`apply_wb_basique`, a verbatim extraction of the `_process_file`
inline block — R/B medians pulled toward G by `clip(med_g/med_ch, 0.5, 2.0)`
when `med_g > 1e-6`) and then rescales into the ADU domain via the shared
`rescale_01_to_adu` helper (`* 65535` when the max is in `[0,1]`, then
`clip >= 0`).  Both the anchor and the deposited frames therefore carry the
exact same multiplicative R/B gains, so the anchor and frames share one RGB
photometric domain and the additive-only relative-match model holds (the two
cannot drift multiplicatively in R/B).  A 2-D grayscale reference skips the WB
step but is still rescaled.

`BackgroundAnchor` owns a **private float32 copy** of the anchor pixels:
constructing it never shares storage with, nor makes read-only, the caller's
array (the private copy is the only array whose write flag is cleared).

### 2.2 Estimator, masks and fallback

`estimate_background_offsets(frame, weight, tf, anchor, *, native_wcs=None,
reference_wcs=None, ...)` is a pure function of (immutable anchor state +
current frame + geometry):

1. deterministically sample frame pixel centres (bounded budget, `<= 250000`);
2. map each sampled centre to the reference grid via `tf` (affine) **or** via
   `native_wcs.all_pix2world` -> `reference_wcs.all_world2pix` (celestial, for
   the astrometry-single path where `tf is None`);
3. sample the anchor at those positions (bilinear, **measurement-only**);
4. `delta = frame - anchor_sample` per channel;
5. valid samples = `weight > 0` AND in reference grid AND finite everywhere;
6. per channel, robustly estimate the constant offset as the
   **iterative sigma-clipped median** of `delta` (MAD-based scale, 3 iters,
   3σ);
7. fall back to a **neutral (zero) correction** with a structured reason when
   overlap is insufficient or the geometry is degenerate / pathological.

`apply_background_offsets` performs the per-channel constant subtraction in a
**private float32 copy** (float32 offsets subtracted in-place on that copy and
returned as float32): the caller's input is never mutated and no full-frame
float64 temporary (result or intermediate) is ever materialised, bounding the
correction transient to one float32 frame copy.

Robustness rationale (conservative estimator):

* the **median** is insensitive to bright sources and to moderate diffuse
  structure (both are minority outliers around the sky level);
* **sigma-clipping** rejects stars, hot pixels and gradient tails;
* NaN/Inf, invalid-weight and out-of-grid border samples are excluded;
* comparison happens only on **geometrically corresponding overlap samples** —
  translated/rotated frames are never compared at unrelated raw pixel
  coordinates (the `native_wcs` path maps through WCS, never through a false
  identity `tf`).

Fallback vocabulary (stable strings): `accepted`, `insufficient_overlap`,
`no_valid_samples`, `no_anchor_data`, `degenerate_geometry`, `invalid_wcs`.
Diagnostics additionally report `n_candidate` / `stride` / `max_samples` (the
sampling budget actually used).

### 2.3 Diagnostics / observability

The per-frame diagnostics (`dR/dG/dB`, overlap/used/candidate sample counts,
accepted/fallback reason, confidence, provenance) are logged at DEBUG for
accepted frames and INFO for fallbacks (bounded: one line per frame).  The last
diagnostics and the anchor are also retained on the stacker
(`_drizzle_bg_last_diag`, `_drizzle_bg_anchor`) for tests/observability.  The
reference frame yields ~zero correction by construction (`delta = 0` when the
frame *is* the anchor).  If a malformed path has **no** run-level anchor, the
frame is deposited with a deterministic neutral correction (`reason:
"no_anchor"`) — never by creating a new arrival-dependent anchor.

### 2.4 Incremental immutability

A frame's correction is resolved against the immutable anchor **before** its one
and only deposition.  Once deposited into SCI/WHT, that contribution is
immutable; later frames cannot mutate an earlier frame's correction.  There is
no evolving/running master background.

### 2.5 Serialization boundary (explicit)

`BackgroundAnchor.to_metadata()` / `from_metadata()` reconstruct the **scalar
anchor contract** (version, provenance, geometry, per-channel background) for
identity verification.  The reconstructed anchor is pixel-less (`sample()`
raises) but keeps its documented anchor shape inspectable (`shape` reads a
separately persisted `anchor_shape_hwc`, never a `None` array).  They
deliberately do **not** serialize the per-pixel anchor or the Drizzle
accumulators — this task does **not** add a full persistent Drizzle SCI/WHT
checkpoint/resume architecture.

---

## 3. Persistent resume boundary (exact, 8.1.0 baseline)

The existing persistent resume contract (HSI-2B, `_resume_manifest.json` +
`cumulative_SUM/WHT.npy`) covers **plain classic SUM/W only**.  Drizzle, mosaic
and reproject finalization remain **explicitly non-resumable**:

* a run that selects Drizzle refuses to start if resume artifacts are present
  (`_save_final_stack`/`initialize` fail closed, unchanged by this task);
* DPIC-01 does **not** expand that contract to serialize the Drizzle
  accumulators, the background anchor pixel data, or the intermediate WCS/grids.

The only new persistence-adjacent artifacts added by DPIC-01 are:
* the **companion WHT FITS** (a final *product*, not a checkpoint), and
* in-memory anchor + diagnostics (never written to disk as a resume state).

---

## 4. Native signed WHT and the corrected relative threshold

`WHT` (weight) is the **native mathematical Drizzle kernel weight/count map**
(`out_wht`), exposure-scaled: `sci = sum(w · f)`, `wht = sum(w · expscale)`.
The native *intermediate weighted mean* that users consume is `out_img`
(`drizzle` 2.2.0 stores the weighted mean there, **not** the weighted flux), so
`finalize("divide")` returns `out_img` directly rather than reconstructing
`(out_img · out_wht) / max(wht, 1e-9)`.

For the `square`, `point`, `turbo` and `gaussian` kernels `WHT` is effectively
positive coverage.  For the **Lanczos kernels** (`lanczos2`, `lanczos3`) the
native `WHT` is **signed**: the sinc lobes are negative near coverage edges, so
`out_wht` contains genuine negative values.  A signed WHT is a *mathematical*
weight map, **not** a positive physical coverage map, and must never be
treated as one.

A sample is **valid** only when its native `out_img` and native `WHT` are both
finite AND `WHT > WEIGHT_EPSILON` (`1e-9`).  Invalid samples (zero, near-zero
positive, negative, or non-finite WHT; or non-finite science) become `0.0` —
never `abs(wht)`, never a huge-value clip, never percentile hiding.

The public `WHT Threshold %` (float `0..1`) is a **coverage / support policy**,
not a raw absolute weight.

`wht_relative_threshold(wht, fraction, tile_size=8, tile_support_min=4,
n_phase_offsets=2)` in `seestar/core/drizzle_core.py` uses a **spatially
supported (block) robust maximum** reference:

* the positive WHT footprint is partitioned into fixed-size `tile_size` (8 px)
  squares under `n_phase_offsets` (2) deterministic half-tile phase positions
  per axis (i.e. 4 tile grids: `(0,0)`, `(0,4)`, `(4,0)`, `(4,4)` for the
  default 8 px tile), removing base-tile boundary sensitivity;
* a tile *supports* a level only when it contains at least `tile_support_min`
  (4) positive pixels; its supported level is its `tile_support_min`-th largest
  positive value (an O(n) partition per tile);
* `reference_support` = the maximum supported level across all tiles and
  phases — a **spatially supported robust maximum**;
* a single (or a few) isolated pathological maxima can never define the
  reference (no tile contains `tile_support_min` such pixels), while a compact
  full-support plateau (e.g. `2%` of the footprint) *is* recovered because it
  populates at least one whole tile;
* a spatially *scattered* population of outliers (`> 0.5%` globally but fewer
  than `tile_support_min` in every tile) does **not** define the reference —
  the reference stays at the surrounding supported background level;
* it operates only on *positive* values within tiles, so sparse-positive
  kernels/pixfrac need no positive geometric neighbours;
* it is **scale-invariant** under exposure scaling (`wht -> k*wht` leaves the
  mask unchanged and scales the reference by `k`);
* `cutoff = fraction * reference_support`;
* `mask = (wht > 0) & (wht >= cutoff)` — **zero-weight pixels are always
  invalid** when the policy is applied;
* 3-D WHT is reduced to 2-D by the per-pixel channel **mean** (the same
  reduction the finalizer uses for display/post-processing).

Degenerate edge behaviour: when no tile anywhere reaches `tile_support_min`
positive pixels (a footprint smaller than the minimum supported population, or
a purely isolated-pixel layout), the reference collapses to the *minimum*
positive value (a deterministic keep-everything choice) with reason
`no_supported_tile`.

Logs include `fraction`, `reference_support`, `cutoff`, `masked_fraction`,
`n_valid`, `n_positive`, `tile_size`, `tile_support_min`, `n_phase_offsets`,
`reason`.  This policy is **not** the photometric fix (that is §2): it only
defines which final coverage is kept.

**Scope**: the relative policy applies **only** to M3 Drizzle finalization.
Classic / Mosaic / Reproject finalization keeps the legacy raw-absolute
threshold block byte-for-byte (Classic is the control; non-M3 behaviour is out
of scope), and never receives an M3 `WhtThresholdResult` policy object.

---

## 5. Kernel / pixfrac propagation

`initialize` now normalizes `self.drizzle_kernel` / `self.drizzle_pixfrac` with
`validate_drizzle_kernel` / `validate_drizzle_pixfrac` (in
`drizzle_core.py`) and passes the validated values to every scientific
`DrizzleAccumulator` constructor:

* `kernel` must be one of the drizzle-engine set
  (`square|gaussian|point|turbo|lanczos2|lanczos3`); unknown values fall back to
  `square` (the flux-conserving default) with a warning;
* `pixfrac` must be finite in `(0, 1]`; invalid values fall back to `1.0` with
  a warning;
* the defaults remain `square` / `1.0`;
* the accumulator passes `pixfrac` to the engine at each `add_image` call
  (`DrizzleAccumulator.add` -> `pixfrac=self.pixfrac`).

This is a deterministic **well-logged fallback** (consistent with the existing
settings coercion), not a hard start refusal: invalid runtime values can never
reach the Drizzle engine.

**Lanczos effective-runtime policy (ZSSS-DNOW-01 R1).**  Upstream drizzle
2.2.0 **ignores `pixfrac`** for the Lanczos kernels (it is assumed `1.0`) and
its Lanczos kernels do **not** conserve flux.  At the M3 accumulator boundary
the *effective* values are therefore forced:

* `kernel` -> the validated Lanczos name (unchanged);
* `pixfrac` -> **`1.0`** (what the engine actually uses); the requested value
  is retained only as optional provenance (`DRZPFREQ`), never mislabelled as
  effective;
* relative `WHT Threshold` -> **`0.0`** (a signed WHT must never be
  relative-thresholded); the requested value is retained only as optional
  provenance (`DRZWTHRQ`).

Positive kernels (`square`/`gaussian`/`point`/`turbo`) keep the requested
`pixfrac` and relative `WHT Threshold` unchanged.  `drizzle_scale` and
`pixel_scale_ratio` are **not** changed by R1: the current code omits
`pixel_scale_ratio`, so the engine actually uses `1.0` (upstream recommends
`pixel_scale_ratio` for `turbo`/`gaussian`/`lanczos`); this is documented as a
known limitation and a separate qualification concern.

At accumulator initialization one concise line is emitted to the durable Qt run
log (`update_progress`) and the logger:

```
DRIZZLE_CONFIG kernel=<effective> pixfrac=<effective> scale=<effective> wht_threshold=<effective>
```

For Lanczos, an adjacent bounded message records the requested -> effective
explanation (pixfrac and threshold).

**GUI-name caveat**: the Qt/Tk UI and settings still list `tophat`, but the
underlying drizzle 2.2.0 engine **rejects** it.  `validate_drizzle_kernel` does
not claim every GUI name is engine-supported: `tophat` (and any other
GUI-only name) is deterministically coerced to `square` at the runtime
boundary, with an explicit warning and test coverage.  The GUI list itself is
out of scope for this corrective task.

---

## 6. Companion WHT product

For actual M3 Drizzle finalization, `_save_final_stack` writes a **separate**
companion FITS (`<name>_wht.fit`) carrying the per-channel **native signed**
mathematical Drizzle WHT (the raw `out_wht`, never `max(raw_wht, 0)`):

* aligned and cropped **exactly** with the final scientific output (same bounding
  box, same WCS/header, CHW layout);
* the primary HDU shape/extension layout is **never** changed
  (compatibility-safe);
* the header records the **signed** `WHTMIN/WHTMAX/WHTMEAN` plus the
  `WHTNEG/WHTZERO/WHTPOS` sample counts and `COVFRAC` (the fraction of
  *positive-weight* samples, accurately named); when a (positive-kernel)
  threshold policy was applied it also records
  `WHTFRAC/WHTREF/WHTCUT/WHTMASK` plus the spatial parameters
  `WHTTILE/WHTSUPP/WHTNPH`;
* `EXTNAME` stays `WHT` (compatibility) but `WHTTYPE` is set to `NATIVE`
  (a short explicit FITS discriminator) and the `HISTORY`/`COMMENT` label the
  product truthfully as a *native mathematical Drizzle WHT*, never a positive
  coverage map;
* the companion is always a float32 **native WHT product** (never a positive
  coverage product): `BITPIX` is forced to `-32` and any inherited
  integer-primary `BSCALE`/`BZERO` pseudo-unsigned scaling is stripped, so it
  never claims unsigned-int semantics;
* the companion is written **only after the primary FITS write has succeeded**
  (`fits_write_success`); on primary failure no companion is created and
  `_companion_wht_path` stays `None`;
* the write is **fail-open**: a companion failure is logged as an explicit
  diagnostic and never corrupts the primary scientific output;
* `_companion_wht_path` and `_drizzle_wht_policy_result` are reset at the start
  of every finalization, so a stale product/policy can never leak across runs.

The display/post-processing code path keeps a **separate clipped-positive 2-D
reduction** (`max(wht, 0)` then channel-mean) used *only* where crop/display
mechanics require positive support; this reduction is **not** the physical
coverage map for Lanczos.  A future independent positive coverage map is an
explicit non-goal for this task.

M3 finalization memory/transient behaviour: the final science HWC and the
companion native-WHT HWC are **preallocated once** and filled one channel at a
time (`finalize("divide")` and `wht` temporaries die each iteration).  The
display/post-processing map is accumulated as a 2-D positive-support reduction
with a **single reusable 2-D scratch**; a full-frame clipped-positive HWC cube
(or a list of clipped-positive channels) is never materialised.  For the real
3840×2160×3 output this bounds the M3 finalization transient to the two
unavoidable HWC products (~200 MB) plus ~130 MB of scratch/one-channel
temporaries, versus the prior list-plus-`np.stack` approach which kept three
channel lists and three stacked HWC cubes alive (~500 MB+).

The temporary investigation NPZ witness is **not** saved.

Every M3 primary FITS and its companion now record the **effective** runtime
provenance: `DRZKERNEL`, `DRZPIXFR` (effective, so `1.0` for Lanczos),
`DRZSCALE`, `DRZMODE` (`"M3"`), and `DRZWTHT` (the effective WHT threshold, a
`<= 8`-char FITS key).  Requested `pixfrac`/`WHT Threshold` are recorded
separately (`DRZPFREQ`/`DRZWTHRQ`) only when they differ from effective, so
requested and effective are never confused.  (`DRZKERNEL` is 9 characters and
is written by astropy as a `HIERARCH` card, read back under the same name —
this matches the pre-existing `_update_header_for_drizzle_final` convention.)

---

## 7. Noisy low-coverage behaviour: scientific footprint vs presentation

* **Scientific footprint**: below `min_overlap_samples`, the correction is
  neutral (zero) with reason `insufficient_overlap`; pixels with zero weight
  never contribute; the relative WHT threshold drops low-coverage pixels only
  via the *coverage policy* (this is presentation/masking, not a photometric
  claim).
* **Presentation**: the relative WHT threshold masks low-coverage output pixels
  (NaN in the science image → excluded from the percentile stretch / preview).
* Neither mechanism invents flux: neutral correction + zero-weight exclusion
  guarantee that low-coverage regions degrade gracefully (fewer samples, no
  fabricated offsets).

---

## 8. HSI interaction / non-interaction

* The additive background match is applied **before** Drizzle deposition; it
  does **not** change the HSI (hierarchical stacking integrity) semantics —
  HSI remains a plain-classic concern.
* The correction is a per-frame constant; it does **not** alter SCI/WHT
  accumulation order, weight formulas, or the `sci / wht` contract.
* The relative WHT threshold and companion WHT are **finalization-time only**
  and do not feed back into accumulation.
* Resume (§3) remains plain-classic-only; DPIC-01 adds no HSI-visible state.

---

## 9. Tests

* `tests/test_drizzle_background.py` — unit tests: zero/constant/RGB offsets,
  star & NaN robustness, insufficient-overlap neutral fallback, purity &
  anchor-state immutability, metadata round-trip (incl. shape inspectability and
  the private-copy/read-only guarantee), WB-basique helper equivalence and
  asymmetric-RGB gain extraction, WHT block-support semantics (adversarial 2%
  compact plateau + isolated outlier, scattered `>0.5%` outliers rejected,
  compact boundary-straddling cluster accepted via phase offsets, exposure
  scaling, zeros, near-threshold, channel reduction, sparse-positive layout,
  tiny-footprint degenerates), native-WCS mapping (shifted/rotated), invalid-WCS
  and degenerate-geometry fallbacks, bounded sampling, float32 offset
  application (dtype / immutability / tolerance / no float64 full-frame),
  kernel/pixfrac validation (incl. the `tophat` GUI-name fallback) + engine spy.
* `tests/test_drizzle_photometric_integrity.py` — synthetic integration:
  translated partial-overlap RGB frames with additive backgrounds (100/110/90)
  + common stars (mean step suppressed while support change remains), broad
  extended source spanning a coverage boundary (structure preserved, not
  flattened), group-size invariance, SCI/WHT coherence, and the queue_manager
  boundary (reference-derived anchor capture, arrival-order invariance,
  reference-zero-correction, no-anchor neutral fallback, kernel/pixfrac
  wiring + fallbacks).
* `tests/test_save_final_stack.py` — save-final witnesses: relative WHT policy
  is M3-only (non-M3 keeps raw semantics and no policy object), companion WHT
  shape/WCS/metadata, threshold metadata (spatial `WHTTILE/WHTSUPP/WHTNPH`),
  real zero-WHT-border crop (primary and companion cropped identically with the
  exact `x0/y0` CRPIX shift), integer-primary companion (float32 WHT with no
  inherited `BZERO`/`BSCALE`, weights unchanged), companion-failure leaves
  primary intact, primary-failure creates no companion, non-M3 creates no
  companion, effective DRZ provenance on primary + companion, native signed WHT
  companion retention, and the fail-closed support-integrity gate.

---

## 10. Native-signed-WHT finalization & support gate (ZSSS-DNOW-01 R1)

`DrizzleAccumulator.finalize("divide")` returns a **private float32 copy of the
native `out_img`** (the weighted-mean image), gated by the validity predicate
`finite(out_img) & finite(out_wht) & (out_wht > WEIGHT_EPSILON)`.  Invalid
samples become `0.0`.  The `sci` property (weighted flux
`out_img · out_wht`) is retained **only as derived bookkeeping**, never as the
native final science.  The `wht` property exposes the native **signed** WHT
verbatim (no clipping).

Other `finalize` modes stay coherent and compatibility-safe on the same
validity gate: `"none"` returns the gated weighted flux, `"max"` returns
`out_img · max(valid wht)`, `"n_images"` returns `out_img · mean(valid wht)`.

**Support-integrity gate.**  `support_integrity_violations(sci, wht)` (finite /
support logic, not an arbitrary ADU limit) reports any sample whose science is
nonzero/non-finite on *invalid* native WHT support.  At the M3 finalization
seam `_save_final_stack` fails closed with a clear diagnostic if any violation
is found (the final stack is not written).  The all-zero / no-support check is
based on `finite(wht) & (wht > WEIGHT_EPSILON)`, never `abs(wht)` or a clipped
negative reduction.

M3 final and live preview both consume the same `finalize("divide")` path, so
the preview shares the native-safe finalization (no huge signed-WHT artefacts).
