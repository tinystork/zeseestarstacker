# research/drizzle_contract/rework1 — corrected archaeology harness (r1)

Corrects the r0 harness defects (Junior review). Research only; no production
patch, no commit.

Corrections:
- pixmaps built through the REAL ZSSS `build_output_grid` +
  `pixmap_from_alignment` on a deterministic TAN WCS, with the production
  output-WCS shape fix (pixel_shape/array_shape/_naxis) — empirical formula
  `out = s*x + (s-1)` proven for scales 1-4 (geo.py::prove_formula).
- pixel_scale_ratio candidates: current omitted (1.0), explicit 1/s, upstream
  None-estimate (0.3335), and s as a wrongness diagnostic.
- true translated / rotated / rotated+translated partial footprints.
- transform-dithered identical-scene accumulation (N = 1/2/4/16/64, scale 3).
- iscale alternatives with area-normalised photometry; mixed-exposure and
  non-uniform quality-weighting witnesses; independent square-kernel support
  pair; resume x3 through the production `DrizzleAccumulator` /
  `from_native_state` seam.
- honest input-domain excursion metrics (raw negative fraction abandoned).

Key result: genuine Lanczos3 ×3 excursion at rotated partial footprints —
native SCI −1109…+7796 on an input bounded [0,200] with WHT > 1e-9
(artifacts/fp_lanczos3_rotated.npz).

Run: `.venv/bin/python research/drizzle_contract/rework1/run_rework.py`
Artifacts: `artifacts/{rows.json, rows.csv, manifest.json, <cell>.npz}`.
