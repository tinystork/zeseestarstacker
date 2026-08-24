# HSI Closure — Project State

Last update: 2026-08-24

## Baseline

- Branch: `feature/post-phoenix-polish`
- Base HEAD: `64a4c6ae86f66b7520f90e1b581143b5e7e37ef5`
- Input review snapshot: `ZeSeestarStacker_HSI_review_64a4c6a_dirty_20260824T040344Z`
- Worktree: accumulated, accepted HSI changes are intentionally uncommitted.
- Remote actions: no push, merge, squash, rebase, or history rewrite authorised.

## Accepted invariants — preserve unless disproved

- [x] Hierarchical representation by effective `SUM / WHT`.
- [x] Effective surviving contribution after rejection.
- [x] Per-channel contribution weights.
- [x] Deterministic frame-quality weights.
- [x] Versioned scientific intermediates and fail-closed legacy handling.
- [x] Explicit non-associativity of median/rejection families.

## Closure gates

- [x] P1 — normalization / batch-boundary invariance audit, including the
      production-auto-started inter-batch normalizer.
- [x] P1-FIX — minimum correction for demonstrated normalization defects.
- [x] P2 — RAM / tiled-HQ / memmap scientific parity.
- [x] P3 — executable non-associativity witnesses, family classification, and
      bounded corrections for the two demonstrated defects.
- [x] P4 — real weighted-intermediate reprojection witness and exact verdict.
- [x] P5-AUDIT — `min_weight` compatibility audit.
- [x] P5-FIX — restore meaningful relative-floor semantics without batch-local
      normalization.
- [x] FINAL — rewrite the HSI report as baseline-before vs final-current model,
      complete verdict tables, regression evidence, limitations, and required
      ZeAlfie ecosystem line.

## Active task

None. `ZSSS-HSI-CLOSURE-FINAL` was architect-accepted on 2026-08-24 after
source review and independent reruns of the documented closure suites. The HSI
technical closure is complete within its stated scope; the worktree remains
uncommitted and release actions remain separate human gates.

## Accepted P5 corrective evidence

- Quality weights now use `max(q(source) / q_ref, min_weight)`, where `q_ref`
  is captured exactly once from the immutable prepared session reference before
  worker launch. No batch-local normalization or first-item anchoring remains.
- `q(scores)` is finite and positive for adversarial NaN/Inf/overflow inputs;
  positive overflow saturates high instead of collapsing to the minimum.
- Missing or malformed `q_ref` aborts singleton, multi-image and checkpoint
  creation paths instead of falling back to raw or uniform weighting.
- Plain-classic checkpoints persist `quality_reference_scale`; quality-weighted
  resume validates it before mutation and restores it verbatim. Missing,
  malformed, non-finite and non-positive values fail closed.
- A quality-weighted uninterrupted-vs-resumed witness with a binding floor
  agrees on final `SUM`, per-channel `WHT` and `V`; the post-resume source uses
  the restored reference scale.
- Tk, Qt, settings validation and backend transport now share default `0.01`
  and range `[0.01, 1.0]`; NaN/Inf/nonnumeric values normalize explicitly to
  the default.
- Independent review: P5 `31 passed`; full resume `125 passed`; focused
  capture/continuation seams `5 passed`; Qt resume `21 passed`; run-config
  `17 passed`; combined normalization/backend/HSI/rejection/reprojection
  `103 passed`; `git diff --check` clean.

## Accepted P5 audit evidence

- The current absolute-floor implementation is batch-independent and composes
  exactly through `SUM/WHT`: singleton and multi-image seams agree, unequal
  decompositions agree, and RAM/tiled/memmap preserve identical `V`, effective
  per-channel `WHT`, and `V * W` (up to float32 reduction order).
- Current formula is `max(snr**a * stars**b, min_weight)`. With typical SNR
  `[10, 50, 100]`, the entire configured `min_weight` range `[0.01, 1]` is
  inert. Rescaling the same 1:5:10 metrics to `[0.001, 0.005, 0.01]` flattens
  the weights to `[0.01, 0.01, 0.01]`.
- Pre-P1 used a mean-1 relative domain; therefore the same user setting changed
  meaning when batch-local normalization was removed. The old batch-local
  algorithm is not acceptable, but the user-facing semantic drift is a real
  compatibility defect.
- Verdict A (HSI composability): **EXACT**. Verdict B (user-setting
  compatibility): **DEFECT**.
- `min_weight` is already part of the resume scientific fingerprint, but Qt,
  settings validation and the backend disagree below `0.01` (`0.005` can pass
  settings validation and is later clamped to `0.01`).
- Independent review: P5 witnesses `18 passed`; normalization/backend/HSI
  regressions `82 passed`; run-config `17 passed`; focused resume
  `113 passed`; `git diff --check` clean.

## Accepted P4 audit evidence

- All weighted-intermediate reprojection paths source-traced in P4 transport
  value and weight separately: `R(V)` and `R(W)`, then form the numerator
  `R(V) * R(W)`; they do not transport the canonical numerator as `R(V * W)`.
- A real bilinear `reproject_interp` witness with a fractional-pixel TAN shift
  and spatially varying `V`/`W` gives `max|delta| = 2.726528` and
  `mean|delta| = 1.786013` across 324 fully covered interior pixels. Constant-W
  and identity-transform controls agree to interpolation tolerance.
- Real `initialize_master`, the local `reproject_and_coadd` fallback, and the
  Astropy reference branch all reproduce the separate-transport result and
  remain clearly distinct from direct `R(V * W)` transport.
- Classification: **APPROXIMATE BY DESIGN** for reproject/mosaic weighted
  intermediates. This matches the documented separate spatial transform and
  Astropy coadd semantics; the exact plain-classic non-reproject `SUM/WHT` path
  is unaffected.
- Independent review: reprojection witness `6 passed`; HSI invariants
  `37 passed`; environment-restoration witness passed; `git diff --check`
  clean. The broader queue-manager reprojection suite is now `28 passed`
  (the previously documented single `shapely`-missing environment failure was
  resolved by installing `shapely` into the environment).

## Accepted P3 audit evidence

- Mean is exactly composable through `SUM/WHT`; median is not composable.
- Kappa-sigma and linear-fit clipping are non-associative under local bounded-
  memory rejection, but their effective WHT correctly excludes rejected
  samples. Global `[100x9,1100]` gives `V=100,W=9`; the demonstrated local
  split composes to `V=200,W=10` because the outlier survives locally.
- Winsorized sigma is likewise non-associative, while its substitution contract
  intentionally retains valid-sample WHT. The witness gives global `V=3.25,W=8`
  versus hierarchical `V=15.125,W=8`.
- Per-channel rejection and missing-sample coverage propagate independently;
  the witness denominator is `[9,10,10]` for a one-channel outlier.
- Independent audit suites: rejection witnesses `10 passed`; HSI + backend
  parity regressions `53 passed`; `git diff --check` clean.
- Defect 1: GUI/settings produces `linear-fit-clip`, but backend dispatch accepts
  only `linear_fit_clip`; the user-visible spelling falls through to mean
  (`V=200,W=10` instead of clipped `V=100,W=9`).
- Defect 2: a production-reachable winsorized singleton produces non-finite
  output (`+inf`, `W=1`) because `ddof=1` empties the survivor set. This requires
  a second bounded correction after C1.

## Accepted P3-C1 result

- Added one backend-boundary linear-fit-clip alias predicate accepting the
  canonical `linear_fit_clip` and GUI-produced `linear-fit-clip` spellings.
- `_stack_worker`, tiled/HQ reduction, and RAM `_stack_batch` all use the same
  predicate, including `stack_reject_algo` compatibility.
- Independent witness on `[100x9,1100]`: both spellings give `V=100,W=9,
  rejection=10%`; mean remains observably different at `V=200,W=10`.
- Independent production-backend evidence covers RAM, tiled/HQ and memmap;
  rejection suite `12 passed`, HSI + backend parity `53 passed`, no memmap
  artifacts, and `git diff --check` clean.
- `stack_disk_streaming()` remains a separate public core API with its own
  documented canonical underscore vocabulary; it is not called by the audited
  GUI/QueueManager pipeline and is not evidence against this pipeline fix.

## Accepted P3-C2 and final P3 result

- Winsorized clipping now treats every pixel/channel with at most one current
  valid contribution as a no-rejection identity; the guard is inside the real
  kernel and therefore also covers partial spatial/channel coverage.
- Independent witnesses: singleton `V=100,W=1,rejection=0`; weighted singleton
  `V=100,W=3.5`; mixed grayscale coverage `W=[1,1,2]`; mixed colour coverage
  `W=[1,2,3]`; zero-valid columns remain `V=0,W=0`.
- Ordinary winsorized substitution and non-associativity are unchanged:
  `[100x9,1100] -> V=100,W=10`; `[0..6,100]` remains global `V=3.25,W=8`
  versus hierarchical `V=15.125,W=8`.
- Independent suites: rejection `15 passed`; rewinsor + HSI `38 passed`;
  backend parity `16 passed`; no memmap artifacts; `git diff --check` clean.
- Final P3 classification: mean `EXACT`; median `NOT COMPOSABLE`; kappa-sigma,
  linear-fit clipping, and winsorized sigma `APPROXIMATE BY DESIGN` under local
  hierarchical rejection, with correct effective WHT propagation.

## Accepted P2 audit and corrective evidence

- Independently reproduced: P2 suite `14 passed`; P1 normalization + HSI
  suites `66 passed`; `git diff --check` clean.
- Source and executable dispatch agree: median, kappa-sigma,
  linear-fit-clip, and winsorized-sigma reach tiled/HQ and memmap; `mean` does
  not, even when tile/memmap conditions are forced.
- With `group_size >= N`, RAM/tiled/memmap parity is demonstrated for all four
  non-mean families; mean's tiled reducer primitive also agrees numerically but
  is not reachable from production dispatch.
- Independent ad-hoc production witness confirms normalized kappa-sigma parity
  for both `linear_fit` and `sky_mean`: maximum deltas in `V`, `W`, and `V*W`
  were all exactly zero; normalized value differed from the pinned reference by
  at most `6.11e-5`.
- The initial evidence gap was corrected: the mean normalization tests now say
  explicitly that mean has no production tiled/memmap dispatch, while new
  `linear_fit` and `sky_mean` witnesses run through production kappa-sigma and
  spy on the actual `_combine_hq_by_tiles` dispatch for both tiled and memmap.
- Those witnesses assert `group_size >= N`, use nonuniform deterministic
  quality weights and partial coverage, and compare final `V`, effective `WHT`,
  and `SUM = V*W`.
- Independent corrective review: RAM vs tiled and RAM vs memmap maximum deltas
  were exactly `dV=0`, `dW=0`, `dSUM=0` for both normalizers; maximum normalized
  value error against the immutable reference was `6.103515625e-05`.
- Independent suites: backend parity `16 passed`; normalization + HSI
  invariants `66 passed`.
- Small-group kappa and median witnesses prove expected nonlinear grouping
  dependence. A probed winsorized singleton subgroup produces NaN; classification
  and any minimum hardening remain reserved for P3 rejection review.

## Known baseline observations to verify, not assume

- `_stack_batch` calls normalization only for multi-image batches and selects
  index 0 of that batch as reference.
- the one-image mean fast path appears to bypass normalization;
- tiled/HQ may use already-normalized arrays in some modes but reload source
  paths in others, potentially bypassing normalization.

These baseline observations were converted into deterministic evidence by P1.

## Accepted P1 evidence

- `none` + mean is decomposition invariant; independently reproduced maximum
  difference: `4.07e-5` (float32 reduction order only).
- `linear_fit` is batch- and order-dependent under the historical batch-local
  index-0 reference; maximum reproduced difference: `149.094`.
- `sky_mean` is batch- and order-dependent, including pure-offset inputs;
  maximum reproduced affine-case difference: `86.603`.
- the defect is not only the singleton shortcut: `[A,B] + [C,D]` differs from
  `[A,B,C,D]` with no singleton batch because each batch selects its own first
  frame as reference.
- one-frame batches bypass normalization completely.
- continuous `SUM += V*W ; WHT += W` does not repeat normalization.
- RAM mean and tiled/HQ in-memory paths consume normalized arrays; tiled/HQ
  with `batch_size != 1` reloads raw `_current_batch_paths` and discards the
  already-normalized/aligned samples. `use_memmap=True` changes storage only.

Evidence: `tests/test_hsi_closure_normalization.py` (15 passing characterization
tests before correction). Production HSI files remained byte-identical to the
accepted review bundle during this audit.

## P1 audit correction required

- `start_processing()` unconditionally calls `_interbatch_start_session()`.
- `_stack_batch()` subsequently applies `_apply_interbatch_normalization()` or
  `_apply_final_combine_interbatch_normalization()` to each mini-stack.
- The first P1 harness set `interbatch_norm_active = False`; its claim that the
  layer was merely optional/default-off is therefore not a valid description
  of the real production session.
- No P1 implementation correction will begin until this second layer is
  experimentally classified, so the closure gate cannot be satisfied by a
  test that bypasses part of the production path.

## Accepted P1 corrective-audit evidence

- IBN is not user-configurable: `start_processing()` auto-starts it and reads
  no setting.
- With `normalize_method = none`, the plain weighted mean is invariant with
  IBN disabled (maximum `1.22e-4`) but becomes decomposition-dependent with
  IBN active (maximum `134.317`).
- IBN does not repair `linear_fit` or `sky_mean`; it shrinks some gaps and
  enlarges others (maximum observed `239.274` and `124.383` respectively).
- The IBN master is effectively the first mini-stack (`_ibn_master_min = 1`),
  gain is skipped below 10,000 overlap pixels, and singleton mean batches skip
  IBN entirely.
- Its radial feather rescales WHT for multi-image batches but not singleton
  batches, creating a second decomposition-dependent relative-weight change.
- Independent reviewer run: normalization + IBN + HSI tests, `65 passed`; the
  exact numeric witness matrix was independently reproduced.

P1 architectural verdict: automatic IBN on plain-classic `SUM/WHT` is a
demonstrated defect, not an accepted approximation. Source normalization must
be defined per observation against the immutable session reference. IBN may
remain unchanged only for non-plain spatial/reprojection paths outside P1.

## Accepted P1-FIX result

- Plain classic `linear_fit` / `sky_mean` normalize each aligned observation
  against one immutable pinned session reference, including singleton batches.
- Missing required reference fails closed before reduction; `none` is unchanged.
- Plain classic tiled/HQ and memmap reducers consume the same aligned,
  normalized arrays as RAM; raw-path reload is retained only outside this path.
- Automatic IBN is disabled for plain classic and preserved unchanged for
  mosaic/drizzle/reproject; their historical batch-local source normalization
  is likewise preserved.
- The one-frame reference is captured only when required and released in worker
  cleanup.
- Independent post-fix witnesses: maximum decomposition delta `5.09e-5` across
  `none`, `linear_fit`, and `sky_mean`; irregular `[ABCD]` vs `[AB]+[CD]`
  maximum `3.05e-5`.
- Independent regression runs: P1 `43 passed`; HSI+resume `150 passed`;
  stacking-core `7 passed`; Qt resume `21 passed`; `git diff --check` clean.
