# ZSSS Support-Aware Overlap Photometry — Windows W80 witness instructions
Mission zsss-support-overlap-p1-20260907 · Phase 1 ACCEPTED (Junior + Nono) · date 2026-09-07

## Status
Phase-1 Classic fix complete and independently accepted. The production code is
an UNCOMMITTED worktree on branch `beta` at base `ede9d98ba46aa8bf2075e50b5719293d01f659a0`
(8.3.0). No commit/push/merge/tag/release has been performed (mission forbids it).
These instructions describe how to execute and preserve the W80 physical witness.
Do NOT run against the master copy; create fresh copies per comparison.

## Canonical dataset
- Directory: `D:\tulip\W80_v1` (IMMUTABLE master).
- 80 FITS total: 40 ALTAZ 20 s + 40 EQ 20 s (EQ from two sessions).
- ALTAZ frames deliberately cover ~0°, 22.5°, 45°, 70°, 100° rotation.
- Pinned reference for EVERY comparison run:
  `ALTAZLight_SH 2- 101_20.0s_LP_20241028-195131 (1268).fit`
  (do NOT let automatic reference selection vary between runs).

## Fresh copies
For each run A/B/C/D, copy the whole 80-file set into a fresh folder (e.g.
`D:\tulip\W80_runs\A`, `B`, `C`, `D`) so the application's possible
move-to-stacked behavior never touches the master.

## Code basis per run
- A — LEGACY SKY_MEAN: run against a CLEAN checkout of base `ede9d98` (no
  uncommitted Phase-1 changes), normalize_method = sky_mean.
- B — NORMALIZATION NONE: current worktree (or base — none is a strict no-op in
  both), normalize_method = none.
- C — OVERLAP-AWARE SKY_MEAN: current uncommitted worktree, normalize_method = sky_mean.
- D — OVERLAP-AWARE LINEAR_FIT: current uncommitted worktree, normalize_method = linear_fit.

## Fixed scientific settings (identical across A/B/C/D except normalize_method)
- stacking mode / reducer: winsorized-sigma-clip
- batch size: 20
- kappa low: 3, kappa high: 3
- winsor: 0.05 / 0.05
- reference: manually pinned (1268).fit
- quality weighting and all other scientific parameters identical to baseline config
- avoid optional cosmetic post-processing for causal comparison

## Evidence to preserve per run
- final SCI FITS
- WHT / support outputs where applicable
- run_config
- main log
- registration diagnostics
- normalization diagnostics (passive/fail-open P1 diagnostics)
- reference identity (pinned path)
- Git SHA of the code actually run

## Recommended numerical analysis
Measure background in multiple sky/background regions, not a single whole-frame
mean. Per run, produce:
1. background median by coverage zone,
2. background spread between zones,
3. large-scale residual map,
4. correlation between background level and footprint/coverage.

Expected: C and D strongly reduce the artificial `background level ↔ geometric
coverage` dependency vs A, without flattening genuine astrophysical structure.
Do not optimize for visual flatness; do not declare closure from screenshots
alone (synthetic + automated + numerical + human visual all required).

## Acceptance gate
- Phase 1 gate GREEN (this): Classic/HSI/Winsorized/GPU/COV/Drizzle/Resume + new
  overlap synthetic tests green + Nono ACCEPT.
- After W80: A/B/C/D comparison + numerical background/coverage analysis +
  human visual inspection → ACCEPT/REWORK recommendation.
- Phase 2 (Reproject batch-local normalization) is gated SEPARATELY and must not
  start before W80 acceptance. Mosaic documented NOT AFFECTED by the proven path.

## Pre-existing baseline debt (NOT caused by this mission — do not treat as gate blockers)
Full suite baseline (ede9d98): 2730 pass / 36 fail / 2 skip. Post-Phase-1 full
suite: 2838 pass / 35 fail / 2 skip (0 new failures; one flaky cov06b render
test passed). Remaining failures are environmental/out-of-scope: stale
frozen-registration-reference fixtures (4 in test_resume.py + 1 rf2_seam),
missing M16 dataset, zesolver not installed, reproject shapely absent, Qt
settings seeding. Report them honestly; do not relabel as green.
