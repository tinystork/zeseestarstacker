# ZeSeestarStacker — Hierarchical Stacking Integrity (HSI) Closure Report

**Final closure report — current worktree.**

| | |
| --- | --- |
| Branch | `feature/post-phoenix-polish` |
| HEAD | `64a4c6ae86f66b7520f90e1b581143b5e7e37ef5` |
| Worktree state | Accepted HSI changes remain **uncommitted**; worktree is dirty by design |
| Scope | Hierarchical stacking integrity (SUM / WHT / V) of the plain-classic path |
| Status | HSI technical closure **accepted by architect** on 2026-08-24. **Not a release claim.** |

This document is the consolidated, final-current description of hierarchical
stacking integrity. It separates **historical baseline defects** (deterministic
counterexamples observed *before* the corrective work) from the **accepted
final-current behavior**, consolidates the closure phases P1–P5 with their
verdicts and independent acceptance evidence, states limitations honestly, and
records the ZeAlfie ecosystem boundary.

The report rewrite itself was documentation-only: it changed no production code
or tests. Independent architect review has now accepted the bounded HSI
technical closure described here. This does **not** declare the project or a
release complete.

---

## 1. Executive summary

Hierarchical stacking integrity (HSI) is the property that a stack of source
frames may be split into sub-groups, each group reduced separately, and the
group results combined to the same scientific value that a single monolithic
reduction of all frames would produce — independent of batch boundaries, group
size, and (for the supported path) memory-reduction strategy.

The sufficient representation for a composable reduction is the pair

```text
SUM  = Σ_i (valid_i · weight_i · value_i)     (numerator)
WHT  = Σ_i (valid_i · weight_i)               (effective denominator)
V    = SUM / WHT                               (displayed value, per channel)
```

Two intermediates compose exactly as `SUM' = SUM1 + SUM2`, `WHT' = WHT1 + WHT2`,
`V' = SUM' / WHT'`. This is the **only** universally composable contract.

**What is exact:**

- Plain/classic weighted and unweighted arithmetic mean — fully composable via
  `SUM / WHT`, including the uniform mean, the quality-weighted mean (with
  batch-independent weights), and per-channel `SUM / WHT` accumulation through
  RAM / tiled-HQ / memmap reducers.
- Per-channel `WHT` retention through classic-batch persistence and
  reprojection loading (version-2 sidecars).
- The resume contract for the **plain classic non-drizzle / non-mosaic /
  non-reproject** path (versioned manifest, fail-closed).

**What is approximate by design:**

- Median — **not composable**. A median plus a count is not a sufficient
  statistic for a global median; the bounded-memory contract is a
  count-weighted combination of local medians and may vary with batch
  boundaries.
- Kappa-sigma, linear-fit clipping, winsorized sigma — local hierarchical
  rejection is **approximate by design**: the locally discarded distribution is
  no longer present, so a local clip cannot generally reproduce a single global
  clip. Their **effective `WHT`** (survivors / substituted winsorized samples)
  is nonetheless exact after the accepted corrections.
- Weighted-intermediate reprojection — `R(V)` and `R(W)` are transported
  separately and multiplied, rather than the canonical numerator `R(V·W)`. This
  matches the documented/Astropy coadd semantics and is **approximate by
  design**; the plain classic non-reproject `SUM / WHT` path is unaffected.

**What is out of scope (not claimed):**

- Drizzle, mosaic, and inter-batch / final reprojection are **not** part of the
  resumable or exact-composability claim for the plain-classic path (drizzle is
  a structurally separate accumulator; see §8 and §9).
- Global clipping parity, full GUI / end-to-end / scientific image-quality
  validation, and byte-identity claims where evidence is tolerance-based.

---

## 2. Baseline-before vs final-current verdict table

The **Baseline** column records deterministic counterexamples observed against
the pre-correction code. The **Final** column records the accepted current
behavior. "BASELINE" labels historical observations only; they do not describe
the current tree.

| Family | Baseline (before correction) | Final current (accepted) | Classification |
| --- | --- | --- | --- |
| Uniform mean / plain classic | exact only when masks honored and no batch-dependent normalization/post-processing intervenes | Composable `SUM/WHT`; decomposition invariant to float32 reduction order (`none` path) | **EXACT** |
| Quality-weighted mean / `min_weight` | batch-local weight normalization changed cross-batch ratios (`17.27` vs `39.33` vs `36.67`) | `max(q/q_ref, min_weight)`, batch-independent; composes exactly through `SUM/WHT` | **EXACT** (composability); user-setting semantic drift was a **DEFECT**, now corrected |
| Source normalization (`linear_fit` / `sky_mean`) | batch-local index-0 reference; batch/order dependent (`linear_fit` Δ=149.094, `sky_mean` Δ=86.603); singleton batches bypassed normalization | fixed immutable session reference; decomposition invariant (max Δ=5.09e-5) | **DEFECT → EXACT** |
| Automatic IBN (inter-batch normalization) | auto-started on plain classic; decomposition-dependent (`none` became Δ=134.317 under IBN) | gated off for plain classic; preserved unchanged for non-plain paths | **DEFECT → EXACT (gated)** |
| Median | local median intrinsically non-composable; masked `10` became `5` (invalid zero bias) | NaN-as-missing; count-weighted local-median approximation, documented boundary dependence | **NOT COMPOSABLE (by design)** |
| Kappa-sigma | local clipping approximate; survivor mass lost (`3.3333` vs `2.5`) | effective `WHT` = survivors; non-associativity documented (`V=200,W=10` vs `V=100,W=9`) | **APPROXIMATE BY DESIGN**, correct effective WHT |
| Linear-fit clipping (incl. spelling dispatch) | GUI `linear-fit-clip` spelling fell through to arithmetic mean (`V=200,W=10` instead of clipped `V=100,W=9`) | `_is_linear_fit_clip_mode` accepts both spellings in RAM/tiled/memmap dispatch | **APPROXIMATE BY DESIGN** + **spelling DEFECT corrected** |
| Winsorized sigma (incl. singleton/NaN hardening) | weighted/unweighted `W` inconsistency; SciPy re-masked leak; singleton `ddof=1` produced non-finite (`+inf`, `W=1`); rank-vs-position indexing bug | canonical floor/truncation semantics; missing samples excluded from stats; `n_valid ≤ 1` no-rejection identity; correct substitution `WHT` | **APPROXIMATE BY DESIGN** + **hardening DEFECTS corrected** |
| RAM / tiled-HQ / memmap | tiled/HQ merged subgroups by geometric coverage; hidden-subgroup boundaries affected results | all reducers consume the same aligned/normalized arrays and compose by `SUM/WHT`; parity `dV=0, dW=0, dSUM=0` for `group_size ≥ N` | **EXACT parity** |
| Per-channel WHT / persistence / reprojection | 3-D WHT broke `_save_final_stack`; reproject collapsed WHT to 2-D; all three sidecars held the same 2-D map | HWC WHT end-to-end; version-2 per-channel sidecars; legacy collapsed sidecar refused scientifically | **EXACT (per-channel)** |
| Weighted-intermediate reprojection | `R(V)` and `R(W)` transported separately (`R(V)·R(W)`), not `R(V·W)` | unchanged; documented approximation (max\|δ\|=2.726528 over 324 interior pixels) | **APPROXIMATE BY DESIGN** |
| Drizzle | structurally separate; already retained per-channel weighted flux/weight | unchanged; group size is a resource policy | **EXACT (separate accumulator)** |
| Resume / checkpoint | production init did not restore persisted SUM/WHT; skip logic could skip after state reset | versioned manifest + exact source ledger; fail-closed refusals; plain-classic only | **DEFECT → EXACT (scoped)** |
| `min_weight` config transport | Qt / settings / backend disagreed below `0.01` (settings accepted `0.005`, later clamped) | Tk/Qt/settings/backend share default `0.01`, range `[0.01, 1.0]`; NaN/Inf/non-numeric → default | **DEFECT → EXACT (aligned)** |

---

## 3. Canonical final mathematical contracts

### 3.1 Plain mean `SUM / WHT`

For the plain (weighted or unweighted) arithmetic mean, every pixel and channel
preserves

```text
S = Σ_i (valid_i · w_i · v_i)
W = Σ_i (valid_i · w_i)
V = S / W   (where W > 0; V = 0 where W == 0)
```

Pairs compose exactly by component-wise addition at arbitrary depth and
grouping. A normalized image alone, `NIMAGES` alone, or geometric coverage
alone is **not** sufficient. The frame weight `w_i` must be batch-independent.

### 3.2 Effective per-channel WHT after rejection / substitution

For per-channel rejection families, `W` is the **effective** denominator, i.e.
the surviving (or, for `apply_rewinsor=True`, substituted-winsorized) samples
per pixel and channel:

- Kappa-sigma / linear-fit clip: `W = Σ survivor · w` (or survivor count when
  unweighted).
- Winsorized sigma (`apply_rewinsor=True`): a valid-but-rejected sample is kept
  with a substituted winsorized value and **still contributes its weight**;
  missing samples remain NaN and never contribute; survivors are preserved
  exactly. `apply_rewinsor=False` excludes rejected samples entirely.
- `W` is `~isnan(arr_final)` — the exact set that contributes to the mean.

Geometric presence *before* rejection is **not** that denominator.

### 3.3 Immutable source-normalization reference

For plain-classic `linear_fit` / `sky_mean`, every aligned observation is
normalized against **one immutable session reference** — the real global
classic-alignment reference image captured once per worker (a private float32
copy, never the first remaining source, never an intermediate stack). The
reference is placed at index 0 and discarded from the output, so singleton
batches are normalized too. A missing required reference **fails closed**
(`RuntimeError`) before any reduction; `normalize_method = none` is a strict
no-op. The reference is released in worker cleanup on every exit.

### 3.4 `q(source) / q_ref` relative quality weights

A frame's scalar quality weight is

```text
w(source) = max( q(source) / q_ref , min_weight )
```

where `q_ref` is the immutable prepared session-reference quality scale pinned
exactly once before worker launch, and `min_weight` is a **relative** floor
expressed as a fraction of `q_ref` (the reference itself receives weight `1.0`
before the floor). The weight is a deterministic function of each frame's own
metrics and the pinned reference — independent of batch companions, order, and
splitting. `q_ref` is finite / required / persisted / restored fail-closed:

- `q(scores)` is finite and positive for adversarial NaN/Inf/overflow inputs;
  positive overflow saturates high (float32 max), never collapses to the floor.
- A missing/malformed `q_ref` raises `_QualityReferenceError` rather than
  silently restoring the pre-P5 absolute domain.
- Plain-classic checkpoints persist `quality_reference_scale`; quality-weighted
  resume validates it before mutation and restores it verbatim; missing /
  malformed / non-finite / non-positive values fail closed.

### 3.5 Nonlinear local-hierarchy limitation

Once a local rejection decision is made, the locally reduced value can compose
correctly *within the defined hierarchical algorithm* — provided the
intermediate preserves the numerator and denominator of the surviving (or
explicitly substituted) samples per pixel and channel. This is sufficient to
compose already-reduced groups, but it **cannot** generally make a local
sigma-clip equal a single global clip: the discarded distribution is no longer
present. Exact global clipping would require retaining the original samples or
a distributional summary strong enough to reproduce the global decision, which
the current algorithms do not define. Kappa-sigma, linear-fit clipping, and
winsorized sigma are therefore **approximate by design** under local
hierarchical rejection, with correct effective `WHT` after the accepted
corrections.

### 3.6 Weighted reproject approximation `R(V)·R(W)` vs `R(V·W)`

Weighted-intermediate reprojection transports value and weight separately —
`R(V)` and `R(W)` — and then forms the numerator as `R(V) · R(W)`. It does **not**
transport the canonical numerator as `R(V · W)`. These are not equal for a
spatially varying weight field under a non-identity transform. This matches the
documented separate spatial transform and Astropy coadd semantics and is
**approximate by design**; the exact plain-classic non-reproject `SUM / WHT`
path is unaffected. Direct `R(V·W)` transport is **not implemented**.

---

## 4. Final source-path description

Current file/function names (line numbers intentionally omitted — they drift;
function names are the stable reference). All paths below are the
**accepted current** state.

### `seestar/core/stack_methods.py` (scientific kernels)

- `_stack_mean` — plain weighted/unweighted mean; NaN-as-missing; returns
  `(result, W, rejected_pct)` when `return_weights=True`.
- `_stack_median` — unweighted `nanmedian`; `W = valid count`.
- `_stack_kappa_sigma` — NaN-as-missing median/σ clip; `W` = survivors.
- `_stack_linear_fit_clip` — median-of-residuals fit clip; `W` = survivors.
- `_stack_winsorized_sigma` / `_stack_winsorized_sigma_iter` — iterative
  winsorized sigma; SciPy path uses explicit `inclusive=(True, True)`, then
  restores the iteration mask (`arr_w_data[~mask] = np.nan`) so missing samples
  never re-enter the location/scale statistics; `n_valid ≤ 1` columns are a
  no-rejection identity; substitution preserves survivors exactly.
- `_winsorize_axis0_numpy`, `_winsorize_bounds` — NumPy fallback; `np.floor`
  (canonical truncation) and rank-based (not position-based) selection.
- `_broadcast_weights`, `_rejected_pct` — helpers.

### `seestar/queuep/queue_manager.py` (dispatch / accumulation / persistence / resume)

- Mode dispatch aliases: `_is_winsorized_mode`, `_is_linear_fit_clip_mode`
  (accept both `linear_fit_clip` and GUI/Qt `linear-fit-clip`).
- Reduction / accumulation: `_stack_batch`, `_combine_batch_result`,
  `_combine_hq_by_tiles`, `_stack_worker`.
- Normalization: `_is_plain_classic`, `_should_capture_norm_reference`,
  `_capture_normalization_reference`, `_release_norm_reference`,
  `_normalize_sources_against_reference`.
- Quality weighting: `_capture_quality_reference`, `_calculate_weights`,
  `_pinned_quality_reference_scale`, `_raw_quality_metric`,
  `_calculate_quality_metrics`, `_QualityReferenceError`.
- Persistence: `_hsi_validate_wht`, `_hsi_wht_channel`, `_hsi_crop_wht`,
  `_hsi_finite_nonneg`, `_hsi_persist_sci_wht_pair`,
  `_save_and_solve_classic_batch`, `_load_classic_batch_wht`,
  `_reproject_classic_batches`, `_reproject_classic_batches_zm`.
- Resume / checkpoint: `_source_identity`, `_scan_queue_decomposition`,
  `_validate_resume_headless`, `_early_resume_preflight`,
  `_validate_and_open_resume`, `_checkpoint_mark_dirty`,
  `_checkpoint_commit_batch`, `_filter_queue_by_resume_ledger`,
  `_RESUME_FINGERPRINT_ATTRS` (authoritative fingerprint attribute list),
  `_ResumeCheckpointError`, `_normalize_min_weight`, `start_processing`,
  `_worker`.

### `seestar/gui/settings.py` (config transport)

- `DEFAULT_MIN_WEIGHT = 0.01`, `_MIN_WEIGHT_LOWER = 0.01`,
  `_MIN_WEIGHT_UPPER = 1.0`, `normalize_min_weight` (NaN/Inf/non-numeric →
  default; finite → clamp to `[0.01, 1.0]`). Mirrors
  `seestar.queuep.queue_manager._normalize_min_weight`.

### `seestar/gui_qt/main_window.py` / `seestar/gui/main_window.py`

- Qt field `min_weight` range `[0.01, 1.0]` default `0.01`; Tk
  `min_weight_var` default `0.01`.

---

## 5. Closure phases P1–P5

### P1 — normalization / batch-boundary invariance audit + P1-FIX

- **Finding (audit):** `linear_fit` and `sky_mean` were batch- and
  order-dependent under a batch-local index-0 reference (max Δ=149.094 /
  86.603), singleton batches bypassed normalization entirely, and
  `start_processing` auto-started IBN on plain classic, making even
  `normalize_method=none` decomposition-dependent (max Δ=134.317).
- **Correction:** immutable pinned session reference; fixed-reference helper
  running before the singleton fast path and before any reducer/weight
  estimator; backend parity (`_stack_batch` passes aligned+normalized arrays to
  tiled/HQ and `use_memmap` reducers); IBN gated off for plain classic and
  preserved unchanged for non-plain paths. C1 corrected three defects (out-of-
  scope non-plain semantic change, session lifecycle leak, fail-closed missing
  reference) and scoped reference capture.
- **Acceptance:** `tests/test_hsi_closure_normalization.py`,
  `tests/test_hsi_closure_ibn.py`; decomposition invariance for
  `linear_fit`/`sky_mean`/`none`, RAM/tiled/memmap parity with no
  `/tmp/hq_batch*.dat` leak, reference immutability, plain-classic no-IBN.
  Independent post-fix witnesses: max decomposition Δ=5.09e-5 across
  `none`/`linear_fit`/`sky_mean`; irregular `[ABCD]` vs `[AB]+[CD]` Δ=3.05e-5.

### P2 — RAM / tiled-HQ / memmap scientific parity

- **Finding:** tiled/HQ merged subgroups by geometric coverage; hidden-subgroup
  boundaries could change nonlinear results; dispatch reachability of the
  reducers was uncertain.
- **Correction:** reducers compose by effective `SUM/WHT`; parity demonstrated
  for `group_size ≥ N`.
- **Acceptance:** independent P2 suite `14 passed`; P1 normalization + HSI
  suites `66 passed`; RAM vs tiled and RAM vs memmap maximum deltas
  `dV=0, dW=0, dSUM=0`; normalized value error vs pinned reference
  `6.103515625e-05`.

### P3 — non-associativity witnesses, family classification, two defect corrections

- **Finding (classification):** mean `EXACT`; median `NOT COMPOSABLE`;
  kappa-sigma, linear-fit clipping, winsorized sigma `APPROXIMATE BY DESIGN`
  under local hierarchical rejection, with correct effective `WHT` after the
  two production defects were corrected. Global `[100×9, 1100]` → `V=100,W=9`;
  local split composes to `V=200,W=10` (outlier survives locally).
- **Defect 1 (spelling dispatch):** GUI/settings produced `linear-fit-clip`
  but backend matched only `linear_fit_clip`, falling through to mean
  (`V=200,W=10` instead of clipped `V=100,W=9`). **Corrected (P3-C1):**
  `_is_linear_fit_clip_mode` alias predicate in RAM/tiled/memmap dispatch.
- **Defect 2 (winsorized singleton non-finite):** a production-reachable
  winsorized singleton produced non-finite output (`+inf`, `W=1`) because
  `ddof=1` emptied the survivor set. **Corrected (P3-C2):** `n_valid ≤ 1`
  columns are a no-rejection identity inside the real kernel.
- **Acceptance:** rejection witnesses `10 passed` then `12` (C1) then `15`
  (C2); HSI + backend parity `53 passed`; rewinsor + HSI `38 passed`; backend
  parity `16 passed`.

### P4 — weighted-intermediate reprojection witness and exact verdict

- **Finding:** all weighted-intermediate reprojection paths source-traced to
  transport value and weight separately (`R(V)` and `R(W)`), then form
  `R(V) · R(W)`; they do not transport `R(V·W)`.
- **Verdict:** **APPROXIMATE BY DESIGN** for reproject/mosaic weighted
  intermediates. A real bilinear `reproject_interp` witness (fractional-pixel
  TAN shift, spatially varying `V`/`W`) gives `max|δ|=2.726528`,
  `mean|δ|=1.786013` over 324 fully covered interior pixels; constant-W and
  identity-transform controls agree to interpolation tolerance; the exact
  plain-classic non-reproject `SUM/WHT` path is unaffected.
- **Acceptance:** reprojection witness `6 passed`; HSI invariants `37 passed`;
  broader queue-manager reprojection suite `28 passed` (with `shapely`
  installed in the environment).

### P5 — `min_weight` compatibility audit + P5-FIX

- **Finding (audit):** verdict A (HSI composability) **EXACT** — the
  absolute-floor implementation is batch-independent and composes exactly
  through `SUM/WHT` (singleton/multi-image, unequal decompositions, and
  RAM/tiled/memmap agree). Verdict B (user-setting compatibility) **DEFECT** —
  removing batch-local normalization changed the meaning of the same user
  setting; Qt / settings validation / backend disagreed below `0.01`.
- **Correction (P5-FIX C1):** reference-relative floor `max(q/q_ref,
  min_weight)` with immutable `q_ref`; `q_ref` finite/required/persisted/
  restored fail-closed; all config seams aligned to default `0.01` and range
  `[0.01, 1.0]`.
- **Acceptance:** P5 `31 passed`; full resume `125 passed`; focused
  capture/continuation seams `5 passed`; Qt resume `21 passed`; run-config
  `17 passed`; combined normalization/backend/HSI/rejection/reprojection
  `103 passed`.

---

## 6. Resume contract and exact supported scope

Resume is supported **only** for the plain classic non-drizzle, non-mosaic,
non-reproject `SUM / WHT` path. That state is exactly two HWC float32
accumulators — numerator `SUM = Σ(V·W)` and effective denominator `WHT = ΣW`,
`final = SUM/WHT` per channel — plus the ordered set of source observations
whose contributions are already inside them.

### On-disk manifest (`memmap_accumulators/resume_manifest.json`, version 1)

Written atomically (temp file + `os.replace`), deterministically (sorted keys),
JSON-safe primitives only. Binds `schema_version`, `state`
(`clean`/`dirty`), `mode` (`classic_sumw` only), `semantics`,
`shape` `[H, W, 3]`, `dtype_sum`/`dtype_wht` (`float32`), a scientific
configuration `fingerprint` (SHA-256 over `_RESUME_FINGERPRINT_ATTRS`),
`stacked_batches_count`, and `completed_sources` (an exact ordered ledger).

### Session / input / reference / plan binding

The manifest also binds the checkpoint to the dataset, not just settings:
`session.input_roots`, `session.reference` (identity via `path`/`name`/`size`/
`mtime_ns`), `session.plan` (`sources` + `decomposition`),
`images_in_cumulative_stack` / `total_exposure_seconds` (restored scientific
counters), and `cumulative_header` (NIMAGES/TOTEXP/SUMWGHTS + reference
metadata). Changing input roots, reference identity, or the plan refuses resume.

### Fail-closed binding

- **Artifacts / session / reference / plan / dtype / q_ref** are validated
  before any mutation. The early read-only preflight
  (`_early_resume_preflight` → `_validate_resume_headless` +
  `_validate_checkpoint_artifacts_readonly` + `_probe_reference_shape_hwc`)
  runs before `_get_reference_image`, so a malformed session can never write
  `temp_processing/reference_image.fit/.png`, the manifest, or SUM/WHT.
- **dtype contract:** manifest `dtype_sum`/`dtype_wht` must equal the
  runtime-configured scientific accumulator dtype (`float32` by default), and
  on-disk dtypes must match; int64/Unicode/complex/bool checkpoints are refused
  before reference preparation (never raising from `np.isfinite`).
- **Ledger:** a completed source is `(normcase(abspath(path)), size,
  mtime_ns)`; the queue filter skips only exact three-field matches. A
  same-path/different-identity item raises instead of silently skipping.
- **Source resolution:** each completed source must exist at its original path
  or its moved-to-stacked location (`<src_dir>/stacked/<basename>`); a missing
  or replaced source refuses.

### Dirty / transaction-end clean protocol

`_checkpoint_mark_dirty` runs before mutating SUM/WHT and **raises**
(`_ResumeCheckpointError`) on persist failure — never warns-and-continues.
`_checkpoint_commit_batch` runs at the real end of `_combine_batch_result`,
after SUM/WHT flush and after counters/header update; clean is never claimed
without a durable manifest.

### Honest crash-atomicity limitation

This is a conservative boundary protocol, **not** a journal / double-buffer. It
implements exact restart from clean committed batch boundaries and refuses
dirty crash state. It does **not** promise recovery of an in-flight batch, adds
no database, and makes no full-memmap copies per batch. A crash inside a batch
leaves the manifest dirty; automatic resume is then refused, with all artifacts
preserved for inspection. There is no automatic rollback or reprocessing from
that dirty state. Unsupported modes (drizzle / mosaic / inter-batch / final
reproject) and legacy/unversioned/incomplete state likewise **fail closed**
without modifying or deleting any artifact. A fresh run requires a clean output
folder (there is no explicit "start over vs resume" user-intent flag; an invalid
existing output directory fails closed rather than guessing destructive
authority).

---

## 7. Complete current test evidence

All commands were run in this worktree against the current (uncommitted)
accepted HSI changes. Suites are reported separately; the numbers are **not**
summed into a global figure.

| Command (`.venv/bin/python -m pytest … -q`) | Result |
| --- | --- |
| `tests/test_hsi_closure_min_weight.py` | **31 passed** (5.12s) |
| `tests/test_hsi_closure_normalization.py tests/test_hsi_closure_backend_parity.py tests/test_hierarchical_stacking_integrity.py tests/test_hsi_closure_rejection.py tests/test_hsi_closure_reprojection.py` | **103 passed** (50.78s) |
| `tests/test_resume.py` | **125 passed** (7.74s) |
| `tests/test_qt_last_stack_resume_m23.py` | **21 passed** (1.96s) |
| `tests/test_run_config.py` | **17 passed** (0.06s) |
| `tests/test_queue_manager_reproject.py` | **28 passed** (9.86s) |
| `git diff --check` | clean (exit 0) |

---

## 8. Known limitations / non-claims

- **Median is not composable.** A median + count is not a sufficient statistic
  for a global median; the bounded-memory contract is a count-weighted
  combination of local medians and may vary with batch boundaries.
- **Local rejection is approximate by design.** Kappa-sigma, linear-fit
  clipping, and winsorized sigma cannot generally reproduce a single global
  clip from local groups; their effective `WHT` is exact, the global decision
  is not claimed.
- **Reproject transport is approximate by design.** `R(V)·R(W)` is transported,
  not `R(V·W)`; direct `R(V·W)` transport is not implemented.
- **Non-plain resume is unsupported.** Drizzle / mosaic / inter-batch / final
  reproject are not resumable and fail closed if resume artifacts are
  presented.
- **No global clipping parity claim.** No claim that hierarchical clipping
  equals single-pass global clipping.
- **No full GUI / end-to-end / scientific image-quality validation.** These
  were not run as part of HSI closure.
- **Dirty / uncommitted worktree.** Accepted HSI changes remain uncommitted; no
  merge, push, tag, release, or cleanup was performed or authorized.

---

## 9. ZeAlfie ecosystem boundary

No runtime dependency on ZeAlfie was introduced. HSI is local to
ZeSeestarStacker: the `SUM / WHT` contract, the quality-weight reference scale,
the normalization reference, the per-channel persistence sidecars, and the
resume manifest are all computed, persisted, and validated inside
`seestar/queuep/queue_manager.py` and `seestar/core/stack_methods.py` with no
ZeAlfie import, call, or data handoff.

Conservative interoperability consequence: because HSI does not read or write
any ZeAlfie artifact, there is no API integration to describe — ZeAlfie (and
ZeSolver / ZeAnalyser, likewise untouched) neither consumes nor supplies the
HSI intermediate representation. Any future cross-tool handoff would require a
separate, explicitly designed interface and is not part of this work.

---

## 10. Final acceptance checklist and closure verdict

- [x] Baseline-before vs final-current verdict table complete (mean, weighted
      mean, normalization/IBN, median, kappa-sigma, linear-fit incl. spelling
      dispatch, winsorized sigma incl. singleton/NaN hardening, RAM/tiled-HQ/
      memmap, per-channel WHT/persistence/reprojection, weighted-intermediate
      reprojection, drizzle, resume/checkpoint, `min_weight` transport).
- [x] Canonical final mathematical contracts stated (mean `SUM/WHT`, effective
      per-channel WHT, immutable normalization reference, `q/q_ref` weights,
      nonlinear local-hierarchy limitation, `R(V)·R(W)` vs `R(V·W)`).
- [x] Final source-path description uses current function names, no stale line
      numbers.
- [x] Closure phases P1–P5 consolidated with finding / correction / acceptance
      evidence.
- [x] Resume contract scoped to plain classic non-drizzle/non-mosaic/
      non-reproject, fail-closed bindings, honest crash-atomicity limitation.
- [x] Test evidence reported per-invocation (no summed global number);
      the previously `shapely`-blocked queue-manager suite now `28 passed`.
- [x] Limitations / non-claims stated.
- [x] ZeAlfie ecosystem boundary stated with a conservative interoperability
      consequence and no invented API integration.
- [x] No commit / push / merge / tag / release / cleanup performed.
- [x] Independent architect review completed against source and current
      worktree; closure suites rerun successfully.

**Architect verdict: ACCEPT.** The bounded HSI technical closure described by
this report is complete. This verdict is limited to the stated scope and
evidence; it does **not** declare the wider project release-ready. The dirty
worktree remains uncommitted, and any commit, push, merge, tag, release, or
deployment is a separate gate.

ZeAlfie integration impact: NONE
