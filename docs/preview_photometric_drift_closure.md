# ZSSS-PREVIEW-DRIFT-01 — Live preview/histogram artificial saturation under photometric drift

**Task:** investigate, reproduce, minimally fix, test and document the Qt live
preview/histogram artificial saturation (white-out) under legitimate 2x-3x
photometric drift.

**Branch:** `fix/preview-photometric-drift`
**Baseline HEAD:** `4f05a44c0bf6d4bf0d8acd452c35eb65a26cf8b1` (clean worktree)

---

## 1. Root cause

The Option-A display pipeline is (see `docs/output_truthfulness_preview_audit.md`
§5.2, §3.2):

```
queue_manager Option-A payload (legacy_normalized, raw_linear)
   → main_window._ingest_option_a_preview
       raw = extract_raw_linear(data)
       if first valid preview:  anchors = compute_anchors(raw)   # p0.5 / p99.5
       mapped = map_raw_linear(raw, anchors)                     # clip((raw-lo)/(hi-lo), 0, 1)
       _pristine_float = mapped
   → histogram / Auto Stretch / Auto WB (all consume _pristine_float / WB-only)
```

The anchors were **frozen for the whole run/context** (only recomputed on the
first valid preview). The mapping `clip((raw − lo)/(hi − lo), 0, 1)` therefore
hard-clips any pixel whose raw-linear value exceeds the first frame's `p99.5`.

`raw_linear` is the *linear* accumulator average (classic SUM/W `SUM/WHT`, or
Drizzle `finalize("divide")`). This average does **not** mechanically grow
merely because more frames accumulate: each new result is a weighted mean, and
a homogeneous incoming population keeps it in the same absolute range. What
*does* legitimately shift the raw-linear distribution is a change
in the **cumulative population** that the average is taken over — e.g.
heterogeneous exposures (10 s frames early in a run, 30 s frames later),
changing sky / throughput, or per-batch normalisation. That shift moves the
robust `p0.5`/`p99.5` percentiles of the current frame. Under a 2x-3x global
evolution of that population, `p99.5` of the current frame sits ~2x-3x above
the frozen anchor, so the *entire* image maps to `≥ 1.0` and clips to `1.0` —
artificial saturation / white-out, and a histogram whose top bin (value `1.0`)
collapses the whole population.

This is a **display-only defect**: the scientific accumulators, FITS output and
finalization math are never touched (see §Scientific isolation). The frozen
anchor was the correct answer to *per-preview min/max pumping*; it simply had
no mechanism to accommodate genuine large drift.

## 2. Quantitative evidence (synthetic, deterministic)

Reproduced with a fixed-seed synthetic raw-linear frame
`uniform(100, 200, (400, 400, 3))` (seed `20260828`) and an unchanged reference
pixel `raw = 150.0` at index `(5, 7)`. All numbers below are exact outputs of
the accepted core (`compute_anchors`, `adapt_anchors_for_drift`,
`map_raw_linear`, `compute_histogram_float`); histograms are quoted on the R
channel of the authoritative 512-bin float model.

"Baseline" rows use the pre-fix frozen anchors (no adaptation); "fixed" rows use
the hysteretic ratchet. Raw and anchor columns are absolute linear flux units;
mapped columns are in the display domain `[0, 1]`.

### 2.1 Case A — modest drift (+10%, whole frame scales)

| Frame | raw min | raw max | raw p0.5 | raw median | raw p99.5 | anchors (lo, hi) | mapped min | mapped max | mapped median | frac `==0` | frac `==1` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frame 1 (initial) | 100.00082 | 199.99991 | 100.50135 | 149.95975 | 199.51047 | (100.50135, 199.51047) | 0.00000 | 1.00000 | 0.49953 | 0.0050 | 0.0050 |
| frame A (+10%), baseline | 110.00091 | 219.99991 | 110.55149 | 164.95573 | 219.46152 | (100.50135, 199.51047) | 0.09595 | 1.00000 | 0.65099 | 0.0000 | 0.1862 |
| frame A (+10%), fixed | 110.00091 | 219.99991 | 110.55149 | 164.95573 | 219.46152 | (100.50135, 199.51047) | 0.09595 | 1.00000 | 0.65099 | 0.0000 | 0.1862 |

The +10% drift stays inside the `0.25` hysteresis band, so the anchors are
**bit-identical** in both baseline and fixed (anti-pumping preserved). The
`~18.6%` top-tail clip is the honest, documented cost of strict anti-pumping on
small changes, and is *not* the defect (which is *majority* `1.0`).

### 2.2 Case B — 2x then 3x global drift (whole frame scales)

| Frame | raw min | raw max | raw p0.5 | raw median | raw p99.5 | anchors (lo, hi) | mapped min | mapped max | mapped median | frac `==0` | frac `==1` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frame 2x, baseline | 200.00165 | 399.99982 | 201.00270 | 299.91949 | 399.02094 | (100.50135, 199.51047) | 1.00000 | 1.00000 | 1.00000 | 0.0000 | 1.0000 |
| frame 2x, fixed | 200.00165 | 399.99982 | 201.00270 | 299.91949 | 399.02094 | (100.50135, 399.02094) | 0.33331 | 1.00000 | 0.66802 | 0.0000 | 0.0050 |
| frame 3x, baseline | 300.00247 | 599.99976 | 301.50406 | 449.87927 | 598.53137 | (100.50135, 199.51047) | 1.00000 | 1.00000 | 1.00000 | 0.0000 | 1.0000 |
| frame 3x, fixed | 300.00247 | 599.99976 | 301.50406 | 449.87927 | 598.53137 | (100.50135, 598.53138) | 0.40058 | 1.00000 | 0.70152 | 0.0000 | 0.0050 |

Baseline collapses the *entire* population to `1.0` (`frac ==1 == 1.0`). The
fixed ratchet widens only the high anchor (`lo` stays frozen at `100.50135`),
restoring an in-domain distribution with only the natural ~0.5% robust high tail
saturated.

### 2.3 Case C — unchanged reference pixel (`raw = 150.0`), rest of frame drifts

The reference pixel's raw value is held at `150.0` while the surrounding frame
scales by `+10%`, `2x`, `3x`.

| Frame | raw min | raw max | raw p0.5 | raw median | raw p99.5 | anchors (lo, hi) | mapped median | frac `==0` | frac `==1` | mapped ref (fixed) | mapped ref (baseline) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ref unchanged, +10% | 110.00091 | 219.99991 | 110.55149 | 164.95525 | 219.46152 | (100.50135, 199.51047) | 0.65099 | 0.0000 | 0.1862 | 0.49994 | 0.49994 |
| ref unchanged, 2x | 150.00000 | 399.99982 | 201.00197 | 299.91864 | 399.02094 | (100.50135, 399.02094) | 0.66802 | 0.0000 | 0.0050 | 0.16581 | 0.49994 |
| ref unchanged, 3x | 150.00000 | 599.99976 | 301.50296 | 449.87793 | 598.53137 | (100.50135, 598.53138) | 0.70152 | 0.0000 | 0.0050 | 0.09939 | 0.49994 |

For `+10%` (no adaptation warranted) the reference pixel maps **identically**
(`0.49994` — stability preserved). For `2x`/`3x` (adaptation warranted) its
mapped value changes in a **bounded, one-way** manner (`0.16581`, `0.09939`)
because the range widened to stop the surrounding white-out; the baseline frozen
anchor would have left it at `0.49994` while *everything else* clipped to `1.0`
(which is exactly the artificial-saturation defect, not a desirable stability).

## 3. Chosen design and trade-off

**Hysteretic monotonic anchor expansion (a ratchet), display-only.**

New toolkit-free core `preview_analysis.adapt_anchors_for_drift(anchor_lo,
anchor_hi, raw_linear, hysteresis=0.25, sep=1e-4)`:

1. Compute the new frame's robust percentile range (`p0.5`, `p99.5`) over the
   deterministic finite-positive sample (the same sample `compute_anchors`
   uses).
2. Keep the anchors **bit-identical** while the new robust range stays within a
   hysteresis band `0.25 × (hi − lo)` around the frozen range → anti-pumping
   for small frame-to-frame evolution.
3. When the robust range escapes the band, **widen monotonically outward**:
   `hi = max(hi, cur_hi)` on bright drift, `lo = min(lo, cur_lo)` on dark
   drift. The expansion covers exactly the new robust range — no overshoot, no
   per-frame percentile re-normalization.
4. Never shrink: a transient dimmer frame after a bright adaptation does not
   "zoom back in" (strong temporal anti-pumping). Degenerate / non-finite
   input carries no drift information and leaves anchors unchanged.

`main_window._ingest_option_a_preview` now calls `adapt_anchors_for_drift` on
every *successive* preview (first preview still uses `compute_anchors`).

**Trade-offs (explicit):**

* *Stability vs accommodation.* Stability is now **conditional**: exact
  unchanged-pixel mapping is guaranteed only while no adaptation is warranted
  (drift within the 0.25x band). Once drift exceeds the band, the mapping
  widens (a bounded, one-way step) — this is the necessary cost of not
  white-outing. This replaces the previous *unconditional* frozen-anchor
  guarantee.
* *Modest evolution still clips the top tail.* A +10% evolution keeps frozen
  anchors and therefore clips ~18.6% of the top tail to `1.0`. This is the
  honest cost of strict anti-pumping on small changes, and it is *not* the
  defect (which was *majority* `1.0`). Raising `hysteresis` would reduce
  re-anchoring churn but push more small-drift clipping; lowering it would
  re-anchor more eagerly (closer to per-frame pumping). `0.25` was chosen so a
  +10% evolution stays frozen while 2x/3x adapts.
* *Ratchet never shrinks within a context.* A transient bright outlier frame
  (covering >0.5% of pixels, enough to move `p99.5`) can permanently widen the
  range for the rest of the run. Mitigated by the robust `p0.5`/`p99.5`
  (insensitive to <0.5% outliers) and accepted as a cosmetic-only, display-only
  effect.

**Rejected alternatives:** per-frame percentile re-normalization (re-introduces
the pumping defect this architecture removed); symmetric full re-anchor on every
drift step (less anti-pumping than the ratchet); backend-side mapping (blurs the
display/backend boundary). **Deferred** (documented, not implemented): a full
decoupling of the raw→display map from the histogram/Auto controls via a
dedicated display-lut layer — larger than the minimal fix and not required to
close the drift defect.

## 4. Files changed

* `seestar/gui_qt/preview_analysis.py` — add `ANCHOR_DRIFT_HYSTERESIS` constant
  and pure `adapt_anchors_for_drift()`; extend `map_raw_linear` docstring.
* `seestar/gui_qt/main_window.py` — import + call `adapt_anchors_for_drift` on
  successive Option-A previews in `_ingest_option_a_preview` (narrow
  display-state / Option-A ingestion path only).
* `tests/test_preview_analysis.py` — replace the unconditional frozen-anchor
  witness with the bounded-stability invariant + drift-accommodation tests.
* `tests/test_qt_display_state_otpux.py` — reframe the frozen-anchor test as
  bounded stability, add `test_2x_3x_drift_accommodated_no_whiteout`.
* `tests/test_qt_histogram_otpux_drift.py` — **new** histogram regression
  witness (see §5).
* `docs/preview_photometric_drift_closure.md` — this document.

No scientific/backend/alignment/registration/RF2/HSI/rejection/SUM/WHT/FIT/
drizzle/solver/analyser/file-movement code was modified. `queue_manager.py` was
inspected read-only only.

## 5. Tests added/changed

`tests/test_preview_analysis.py`:

* `test_unchanged_reference_pixel_stable_when_no_adaptation_warranted`
* `test_adapt_anchors_modest_evolution_stays_frozen`
* `test_adapt_anchors_2x_3x_drift_no_whiteout`
* `test_adapt_anchors_dark_drift_widens_low_anchor`
* `test_adapt_anchors_ratchet_never_shrinks`
* `test_adapt_anchors_degenerate_and_invalid_inputs_safe`
* (replaced) `test_fixed_anchor_successive_preview_witness`

`tests/test_qt_display_state_otpux.py`:

* `test_2x_3x_drift_accommodated_no_whiteout` (new)
* (reframed) `test_frozen_anchors_across_successive_previews`

`tests/test_qt_histogram_otpux_drift.py` (new — the requested histogram
regression witness):

* `test_successive_drift_does_not_collapse_histogram_top_bin` — feeds a
  deterministic channels-last HWC **RGB** Option-A sequence through
  `MainWindow`, waits for the async histogram worker, and inspects the
  **authoritative** float model (`win._histogram_model` is
  `win.right_histogram_view.model`). It asserts the applied model reports the
  authoritative channels `["R", "G", "B"]`, that the three first-frame channel
  medians are distinct (three healthy channel distributions), and that after a
  2x and a 3x drift the top bin (x = 1) fraction stays `<< 1` and the
  per-channel `median`/`mean` stay well inside `(0, 1)` for **every** R/G/B
  channel. The display mapping (anchors → `[0, 1]` map → float histogram) is
  shared by mono and RGB, so the RGB channels prove the exact R/G/B drift
  symptom directly rather than via a mono `L` reduction.

  Synthetic RGB frame: three independent `uniform` bands
  `R ~ U(110, 210)`, `G ~ U(90, 190)`, `B ~ U(130, 230)` over `(400, 400)`
  pixels each, stacked HWC, seed `20260828`. First-frame anchors (pooled
  p0.5 / p99.5) are `(91.521384, 228.486819)`; the fixed ratchet widens only
  the high anchor, to `456.973637` (2x) and `685.460456` (3x).

  Before/after per-channel numbers (median / mean / top-bin frac; the top bin
  is `x == 1`, the 512th of 512 bins; baseline == fixed on the first frame
  because no adaptation has been warranted yet):

  | drift | channel | baseline (frozen) median / mean / top-bin frac | fixed median / mean / top-bin frac |
  |---|---|---|---|
  | 1x (initial) | R | n/a | 0.499007 / 0.499547 / 0.000000 |
  | 1x (initial) | G | n/a | 0.354019 / 0.354156 / 0.000000 |
  | 1x (initial) | B | n/a | 0.645957 / 0.645614 / 0.017756 (2841/160000) |
  | 2x | R | 1.000000 / 0.998687 / 0.958712 | 0.624472 / 0.624877 / 0.000000 |
  | 2x | G | 1.000000 / 0.957087 / 0.758231 | 0.515794 / 0.515833 / 0.000000 |
  | 2x | B | 1.000000 / 1.000000 / 1.000000 | 0.734621 / 0.734364 / 0.018606 (2977/160000) |
  | 3x | R | 1.000000 / 1.000000 / 1.000000 | 0.653405 / 0.653779 / 0.000000 |
  | 3x | G | 1.000000 / 1.000000 / 1.000000 | 0.553100 / 0.553136 / 0.000000 |
  | 3x | B | 1.000000 / 1.000000 / 1.000000 | 0.755068 / 0.754830 / 0.018950 (3032/160000) |

The removed unconditional witness is not "deleted without replacement": it is
replaced by the explicit **bounded stability invariant** — exact stability
*when no adaptation is warranted*, bounded ratchet behaviour *when it is* — and
the trade-off is documented above and in §3.

## 6. Validation commands and results (R2 — RGB witness)

```
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_qt_histogram_otpux_drift.py -q
    → 1 passed   (RGB histogram witness)
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest \
    tests/test_qt_histogram_otpux_h1.py tests/test_qt_histogram_otpux_h2.py \
    tests/test_qt_histogram_m14.py -q
    → 47 passed
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest \
    tests/test_preview_analysis.py tests/test_qt_display_state_otpux.py -q
    → 53 passed
.venv/bin/python -m py_compile seestar/gui_qt/main_window.py \
    seestar/gui_qt/preview_analysis.py tests/test_preview_analysis.py \
    tests/test_qt_display_state_otpux.py tests/test_qt_histogram_otpux_drift.py
    → OK (no output)
git diff --check
    → OK (no output)
```

Total focused R2 run: **101 passed, 0 failed** (1 + 47 + 53; no full
1615-image workload was run). The architect's wider viewer/preview regression
run, including initial preview, Drizzle preview, ergonomics, pan/zoom and view
reconciliation, passed **261 tests** with 4 third-party SciPy deprecation
warnings.

### 6.1 Baseline-failure proof (behavioural, not ImportError)

In a clean temporary worktree at `4f05a44c…` with only
`tests/test_qt_histogram_otpux_drift.py` copied in:

```
QT_QPA_PLATFORM=offscreen <venv>/python -m pytest tests/test_qt_histogram_otpux_drift.py -q
    → FAILED tests/...::test_successive_drift_does_not_collapse_histogram_top_bin
        AssertionError: scale=2.0 R: histogram top bin collapsed (frac=0.9587)
```

The test fails **behaviourally** (top-bin collapse: at 2x the R channel already
has `median == 1.0` with a `0.958712` top-bin fraction and the B channel is
fully collapsed to `median == mean == 1.0` / `top-bin == 1.0`; at 3x all three
R/G/B channels collapse to `median == mean == 1.0` / `top-bin == 1.0`), not via
`ImportError`: it imports only stable public seams that exist on the baseline.
The existing
`test_qt_display_state_otpux.py::test_2x_3x_drift_accommodated_no_whiteout` and
the `adapt_anchors_for_drift` unit tests additionally fail on baseline
(`assert win._anchor_hi > hi1`, and the `ANCHOR_DRIFT_HYSTERESIS` import does
not exist) — but the histogram witness is the one that fails on the *actual
collapse behaviour*.

## 7. Existing controls covered

* Manual BP/WP, WB, gamma, BCS survive a new preview unchanged — covered by
  `test_manual_adjustments_survive_new_backend_preview` and
  `test_no_automatic_auto_stretch_or_autowb_on_successive_updates` (still pass;
  the fix only changes the *source mapping*, never the control state).
* Auto Stretch / Auto WB use the float buffers and never mutate anchors — covered
  by the `test_explicit_auto_stretch_matches_float_core_…` /
  `test_explicit_auto_wb_matches_float_core_…` tests (still pass).
* Live Auto Stretch / Live Auto WB fire once per new batch — covered by
  `test_qt_otpux_stable_a/b` (still pass).
* Histogram auto-zoom/reset controls — covered by `test_qt_histogram_m14.py`
  (still pass).
* Context reset (run start / new folder / clear / invalid payload) — covered by
  `test_run_start_resets_anchors_and_buffers`, `test_new_folder_resets_…`,
  `test_clear_preview_resets_…`, `test_invalid_payload_clears_…` (still pass).

## 8. Real FIT witness

The interrupted-run artifacts were supplied read-only under
`/media/tristan/X10 Pro/M16/out`. No stacking run was launched. The run log
confirms batches 1–11 used 10 s sources, the first 30 s sources entered just
before `stack_batch012.fit`, and shutdown completed normally after saving
`stack_batch023.fit`: `FINAL_PREVIEW_SAVE_RETURNED success=True` followed by
`ENGINE_PROCESSING_RETURNING error=False`. The log records 440 accepted 10 s
frames and 457 accepted 30 s frames (897 cumulative images).

`stack_batch023.fit` SHA-256 is
`03bb5c349e9476c2fe7b6f73f048a52f1802cce7016ebf61a1c1c8e1f93ef35a`.
It is a float32 `(3, 1920, 1080)` FIT with 100% finite pixels. Per-channel
linear statistics are:

| channel | finite min | median | p99 | p99.5 | max |
|---|---:|---:|---:|---:|---:|
| R | 0.01198465 | 0.02228578 | 0.02759879 | 0.03027090 | 0.64649820 |
| G | 0.01196289 | 0.02229312 | 0.02715891 | 0.02997326 | 0.49200463 |
| B | 0.01197778 | 0.02232207 | 0.02699615 | 0.02933633 | 0.31103739 |

This is a healthy, non-saturated raw-linear distribution. The stopped-run
final PNG is also a normal astronomical image, independently confirming the
science/display split.

Only batch 23 was retained, so the exact batch-1 anchors cannot be recovered.
For a deterministic real-data drift witness, the early 10 s context is
modelled as `stack_batch023 / 3`, matching the documented 10 s→30 s exposure
transition. This is explicitly an anchor surrogate, not a claim that an
archived batch-1 FIT was measured. It yields initial anchors
`(0.00692557, 0.00996306)`.

Applying the 8.2.0 frozen mapping to the **real batch-23 pixels** reproduces
the failure exactly: all three channels have mapped min/median/max `1.0`,
histogram mean/median `1.0`, and 100% of pixels in the final bin. With the fix,
the high anchor widens to `0.02988918` and the same unmodified FIT produces:

| channel | mapped median | histogram mean | histogram median | exact `1.0` | final-bin fraction |
|---|---:|---:|---:|---:|---:|
| R | 0.668894 | 0.682304 | 0.668869 | 0.5466% | 0.5360% |
| G | 0.669213 | 0.677462 | 0.669195 | 0.5083% | 0.4952% |
| B | 0.670474 | 0.677431 | 0.670443 | 0.4520% | 0.4366% |

The real FIT was also fed through the actual offscreen `MainWindow` Option-A
ingestion and asynchronous authoritative RGB histogram path. Auto WB remained
neutral; Auto Stretch selected `asinh`, BP `0.633`, WP `0.983`; the rendered
preview is visually usable and shows M16. Diagnostic artifacts are retained in
`review/zsss_preview_photometric_drift_evidence_20260828/`:

* `stack_batch023_old_stale_mapping.png` — baseline white-out;
* `stack_batch023_fixed_adaptive_mapping.png` — usable fixed raw mapping;
* `stack_batch023_mainwindow_fixed_controls.png` — real MainWindow rendering
  after Auto WB / Auto Stretch.

## 9. Scientific isolation

The change is confined to the display-only mapping (`preview_analysis.py` +
`main_window._ingest_option_a_preview`). No alignment, registration, RF2, HSI,
rejection, SUM/WHT, FIT output/header/finalization, drizzle, solver, analyser,
file-movement or backend stacking logic is touched. `preview_analysis.py`
remains toolkit-free with lazy numpy and no forbidden imports (the existing
science-isolation tests still pass). Backend Option-A payloads remain
`(legacy_normalized, raw_linear)`; the second element stays raw/finite — verified
by `test_preview_raw_linear_producer.py` (still passes).

## 10. Remaining risk / deferred work

* Hysteresis `0.25` is a single tuning constant; a different dead-band would
  shift the modest-vs-drift boundary (documented trade-off, §3).
* The ratchet never shrinks within a context; a >0.5%-coverage transient
  bright frame can widen the range for the rest of the run (cosmetic-only).
* Larger decoupling (a dedicated display-LUT between the raw map and the
  histogram/Auto controls) is deferred — not required to close this defect.
* The exact batch-1 anchor pair is unavailable because only batch 23 was
  retained. The real-data before/after therefore uses an explicitly labelled
  10 s/30 s scale surrogate, supported by the run log; the post-fix MainWindow
  display itself uses the real batch-23 pixels unchanged.

## 11. Git and delivery state

The implementation, tests and this closure report are delivered as one
coherent local commit on `fix/preview-photometric-drift`, based directly on
`4f05a44`. The final HEAD, status, diff stat and recent log are captured in the
review bundle. `git diff --check` and Python compilation are clean. No push,
merge, tag, release or version change has occurred.

## 12. Acceptance verdict

**ACCEPT.** G1–G12 are closed within the mission's display-only scope:

- the frozen-anchor clipping root cause is independently demonstrated;
- the RGB regression fails behaviourally on clean `4f05a44` and passes after
  the fix;
- 2x/3x drift no longer whites out either the mapped buffer or live histogram;
- bounded small-change stability, context reset and all requested controls stay
  protected;
- the real batch-23 FIT is healthy and renders correctly through `MainWindow`;
- 261 relevant viewer/preview tests pass;
- the source FIT hash is unchanged and no scientific subsystem was modified.

The only evidence limitation is explicit: batch 1 was not retained, so the
real-data old-anchor replay uses the logged 10 s/30 s ratio as a surrogate.
This does not affect the post-fix real-FIT display witness or the automated
baseline failure proof.
