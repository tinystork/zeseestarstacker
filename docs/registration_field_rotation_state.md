# Registration & field rotation — project state

- Baseline: `61291aa035562f09a2bd7cf7971ef8192e1585bf`
- Working branch: `feature/registration-field-rotation`
- Current milestone: RF-1 (corrective C, iteration 2/3) — corrected scale
  statistics + corrected global-reference audit (coordinate frame vs target
  image vs provenance) + new batching-dependence POC
- HSI contract: immutable; plain-classic exactness scope and documented
  reprojection approximations must not be reinterpreted
- Baseline validation: `20 passed` from `test_drizzle_core.py`,
  `test_interbatch_classic.py`, and `test_solver_port.py`

## RF-1 corrective C — findings (replaces RF-1B)

- **Model decision: `FURTHER DATA REQUIRED`** (unchanged).
  - Current registration model (classic + M3 Drizzle standard): astroalign
    **similarity** fit → scale **discarded** → **Euclidean** (rotation +
    translation, scale=1).  Anchors: `seestar/core/alignment.py:220-237`.
  - Synthetic (measured): a 0.3% uniform scale drift leaves a **2.75 px corner
    residual** under the current model — not universally sufficient.
  - Real M16 (one session, measured, *corrected*): scale ≈ 1.0 within noise —
    `median(|scale−1|) = 74.1 ppm` ≈ **0.08 px** at the corner, below the
    ~0.15 px held-out centroid noise; held-out improvement from retaining scale
    is negligible (median ~0.001 px, signed ±).  Fit noise, not a coherent signal.
  - Cross-session/temperature scale drift is **unmeasured**; hence FURTHER DATA
    REQUIRED (not a model change, not "sufficient").
- **Statistic correction (defect #1):** RF-1B's "median |scale−1| = 24.5 ppm"
  was `abs(median(scale)−1)` with a wrong label.  Now reported as three distinct
  quantities: `|median(scale)−1| = 74.1 ppm`, `median(|scale−1|) = 74.1 ppm`,
  `mean(|scale−1|) = 65.6 ppm`; corner implication restated with
  `median(|scale−1|)`.
- **M16 preprocessing correction:** the witness now reads header `BAYERPAT` and
  applies `detect_and_correct_hot_pixels` (threshold 3.0, neighbourhood 5),
  verified **bit-identical** to the production helpers on the CPU path.  The
  reference frame changes (…_035421, metric 24.113) but the A/B shows the scale
  conclusion is unchanged.  Hold-out limitation stated explicitly (astroalign
  RANSAC ran on all stars before the fit/hold-out split → model-fit holdout,
  not independent correspondence-selection validation).
- **Global-reference audit (corrected — defect #2):** three distinct facts,
  previously conflated:
  1. **Coordinate frame (grid) — FROZEN** (`freeze_reference_wcs`); per-frame
     matrix maps source directly into the frozen global grid.  Direct mapping:
     **YES**.
  2. **Target image (data) — EVOLVES** in `reproject_between_batches` (cumulative
     stack).  Immutable target image: **NO** in reproject mode.
  3. **Provenance — NOT RETAINED** in the classic reproject path (per-frame M
     not persisted; aligned pixels/batch products remain).  Transform
     reconstruction: **NO**.
  - Accidental/accumulated centroid bias was a **hypothesis**, now tested by the
    batching POC.
- **Batching-dependence POC (new):** compares immutable vs evolving reference
  catalogue on a fixed global grid with known transforms.
  - Zero-mean noise → **no** bias propagation (reference converges to truth).
  - Adversarial systematic radial bias → evolving target **absorbs and hides**
    the bias (reference drifts 0.062→0.287 px, fit residual collapses to noise,
    true-global error unchanged) — fit residual is an unreliable proxy for
    true-global accuracy in reproject mode.
  - **Batch-size and order dependence** of the reference trajectory confirmed.
  - Limit: an *unrepresentable* bias dilutes (does not compound) under a
    similarity fit; a representable bias would be perpetuated.  POC, not a
    production worker replacement.
- **AST test (fixed + reframed):** `_inside_reproject_guard` is now
  branch-aware (positive body only; `else`/`elif`-negative and `not`-guards
  rejected), pins the **exact** mutation line set, and is unit-tested on small
  snippets.  Described as a **structural path guard**, not experimental proof.

## Deliverables (RF-1 corrective C)

- `research/registration_field_rotation/m16_scale_witness.py` (corrected stats +
  faithful preprocessing + A/B + holdout limitation).
- `research/registration_field_rotation/batch_dependence_poc.py` (new).
- `research/registration_field_rotation/model_selection_poc.py` (label only).
- `tests/test_global_reference_audit.py` (branch-polarity fix, 11 tests).
- `tests/test_m16_scale_witness.py` (new, 7 tests).
- `tests/test_batch_dependence_poc.py` (new, 9 tests).
- `tests/test_registration_model_selection_poc.py` (unchanged, 15 tests).
- `docs/registration_field_rotation_research.md` (full corrected report).
- `docs/registration_field_rotation_state.md` (this file).

## Open items / gates

- RF-1 architect gate: **ACCEPTED** after independent file/math/source review,
  three POC reruns, production-preprocessing equivalence witness
  (`raw_maxdiff=0`, `prepared_maxdiff=0`), and **62 passed** targeted tests.
- Accepted decision: **FURTHER DATA REQUIRED**. No production model migration
  is scientifically justified from one real session.
- Next gate (RF-2 candidate, not authorized yet):
  1. Log the discarded astroalign scale per frame (no behaviour change).
  2. Run the M16 scale witness on ≥2 more sessions spanning temperature/focus.
  3. Only if |scale−1| is consistently ≳ 1e-4, make the ~4-line change in
     `alignment.py` to stop forcing scale=1 (Drizzle core needs no change).
  4. If per-frame independence/provenance is required, retain an **immutable
     high-SNR reference target** (or persist each frame's registration solution
     + reference identity) — a bookkeeping/target-selection change, not a
     rewrite — and add the behavioural worker test in the research doc §9.3.
- RF-1 is closed as a research gate. RF-2 remains blocked on additional
  cross-session/temperature/focus data; no production implementation was made.
