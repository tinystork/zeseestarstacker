# research/drizzle_scientific_closure_p2/p2a — Phase P2-A deposition-truth harness

Mission: `zsss-drizzle-scientific-closure-p2-20260910`, phase **P2-A** (bounded
"direct cancellation proof"). Research only — no production change, no
threshold, no conditioning rule.

Branch: `feature/drizzle-scientific-closure-p2` · starting HEAD
`39826eed1146a7ec6e160e242b87aaaca787b800` · product version 8.4.0 (unchanged).

## What is here

- `p2a_deposition_truth.py` — bounded, deterministic harness.
  - **Block 1 (delivered engine-level truth).** Reads the frozen physical
    artifacts of the completed runs under
    `/home/tristan/Téléchargements/out` (`drizzle_science_diagnostics.json`,
    the `.m3d_checkpoint` native `out_img`/`out_wht` buffers, the delivered
    final FITS) and records, for a curated physical sample:
    delivered normalized SCI, the signed native-WHT denominator sum
    (`out_wht`), the signed numerator contribution sum (`SCI × out_wht`), the
    reconstructed normalized SCI (`N / D`) and its relative error, plus the
    Phase-1 normalized-WHT proxies (`wht_over_local_ref`, `n_eff`, ...).
    These are *observed facts from the delivered buffers*, not an independent
    per-contribution replay.
  - **Block 2 (frozen-geometry replay gate).** Rebuilds the run geometry from
    the frozen artifacts (per-frame affine from
    `registration_diagnostics.jsonl`, ZSSS `pixmap_from_alignment` convention
    `pixmap = scale·(tf @ [x,y,1])`, scale = 3, pixfrac = 1) and re-deposits
    the same frames through the *installed* `drizzle` 2.2.0 engine to test
    whether the reconstructed **native WHT** reproduces the delivered native
    WHT. A per-contribution signed-weight decomposition
    (positive/negative/absolute sums, direct cancellation metric) is only
    meaningful if this gate passes.
- `test_p2a_deposition_truth.py` — narrow hermetic tests for the
  delivered-truth extraction (synthetic run; no physical artifacts, no replay).
- `artifacts/p2a_deposition_truth.json` — produced artifact (deterministic).

## Run

    .venv/bin/python research/drizzle_scientific_closure_p2/p2a/p2a_deposition_truth.py
    .venv/bin/python -m pytest research/drizzle_scientific_closure_p2/p2a/test_p2a_deposition_truth.py -q

## Headline findings (2026-09-10)

Delivered engine-level identity (Block 1) is exact at every sampled pixel:
`reconstructed SCI = (SCI·WHT)/WHT` reproduces the delivered SCI with
relative error `0.0` (float32). Catastrophic pixels are denominator-tiny
(e.g. Lanczos3 SCI −1.895e5 at native WHT 6.06e-5; STOP Lanczos2 SCI +8.214e5
at native WHT 1.04e-5) while stable/benign controls carry native WHT ≈ 88
(Square) / 92 (Lanczos2 robust-high).

**Block 2 gate FAILS.** The reconstructed native WHT does not reproduce the
delivered native WHT: zero-shift correlation ≈ 0.13 / 0.11 / 0.03 for
Lanczos2 / Lanczos3 / STOP-Lanczos2, and ≤ 0.15 even at the best integer 2-D
shift (≈ −59…−52 rows). Even the flux-conserving Square control needs an
unexplained systematic ≈ −58-row shift (corr 0.74 → 0.96) and still does not
reach 1.0. The exact per-frame output pixmaps (sub-pixel phase) are therefore
**not reconstructible** from the preserved artifacts. Consequence: the signed
per-contribution positive/negative/absolute sums and the direct cancellation
metric at the physical pixels are **not** produced here — see the durable
P2-A report for the precise blocker and the decision needed.
