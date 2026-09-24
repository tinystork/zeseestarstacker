# research/drizzle_contract — ZSSS Drizzle kernel/scale scientific contract archaeology

Mission: `zsss-drizzle-contract-20260909`. Research only — NO production
patch, NO commit. All probes execute the REAL installed `drizzle` 2.2.0 engine
(`.venv`), no mocks.

## What is here

- `probe_engine.py` — deterministic probe library mirroring the ZSSS Standard
  M3 deposit convention (`seestar/core/drizzle_core.py::DrizzleAccumulator` +
  `queue_manager._add_frame_to_drizzle_accumulators`): output grid
  `round(H*scale) x round(W*scale)` (ZSSS `build_output_grid` CDELT/scale &
  CRPIX*scale convention ⇒ output pixel index = `scale*input index`), pixmap
  `scale*(i+0.5)`, `Drizzle(out_img,out_wht,kernel,fillval)` +
  `add_image(data, exptime, pixmap, weight_map, in_units='counts', pixfrac,
  wht_scale=exptime)`; like ZSSS, NO `iscale` and NO `pixel_scale_ratio`
  unless an A/B cell passes them. Inputs A–E, metric functions, WHT-bin
  analysis, NPZ/CSV/JSON sinks.
- `matrix.py` — mandatory real-engine matrix: kernels
  square/lanczos2/lanczos3/point/turbo/gaussian × scales 1/2/3/4 on constant
  (A) and star (B) fields; C edge & D partial-footprint at scale 3; E
  identical-frame repeats (N=1,2,4,16,64); current-vs-explicit
  `pixel_scale_ratio` at scale 3 (A/B); weighting OFF/ON; exposure
  10/20/30 s; continuous == checkpoint+resume parity.
- `probe_high_contrast.py` — high-contrast / partial-overlap / dither hunt
  for the catastrophic Lanczos regime (|SCI| > 1e3 cells saved + WHT-bin
  correlation).
- `reproducer_lanczos.py` — broader pathology scan (JSON of every cell).
- `artifacts/` — per-cell `*.npz` (sci/wht/mask), `metrics.csv`,
  `metrics.json`, `lanczos_pathology_scan.json`, `lanczos_high_contrast.json`,
  catastrophic `hc_*_bins.json`.

## Run

    .venv/bin/python research/drizzle_contract/matrix.py
    .venv/bin/python research/drizzle_contract/reproducer_lanczos.py
    .venv/bin/python research/drizzle_contract/probe_high_contrast.py

## Key upstream facts pinned (drizzle 2.2.0, resample.py)

- `Drizzle(kernel=..., fillval=..., out_img=..., out_wht=..., exptime=...,
  disable_ctx=...)`; `add_image(data, exptime, pixmap, data2=None, dq=None,
  scale=<deprecated>, iscale=1.0, pixel_scale_ratio=1.0, weight_map=None,
  wht_scale=1.0, pixfrac=1.0, in_units='cps', xmin/xmax/ymin/ymax)`.
- `iscale` rescales data (data2 by iscale²); deprecated `scale` ⇒
  `iscale=scale², pixel_scale_ratio=scale`.
- `pixel_scale_ratio` sizes turbo/gaussian/lanczos kernels from their nominal
  INPUT-pixel size into the OUTPUT coordinate system; `None` ⇒ estimated from
  pixmap.
- `in_units='counts'` ⇒ data rescaled by `expscale=exptime`; ZSSS passes
  `in_units='counts'` + `wht_scale=exptime` (exposure folded into weight once
  — no double scaling; exposure witness confirms rate semantics).
- Upstream warning: gaussian and lanczos2/3 do NOT conserve flux.
- ZSSS effective runtime forcing for Lanczos: `pixfrac -> 1.0` (upstream
  ignores pixfrac for Lanczos anyway) and relative WHT threshold -> 0.0.
- ZSSS `DrizzleAccumulator.add` passes NO `iscale`/`pixel_scale_ratio`
  (upstream defaults 1.0/1.0) on every reachable direct-accumulation path.

## Headline witness numbers (see report + metrics.csv/json)

- Constant fully-covered fields: SCI ≈ input value for every kernel × scale
  1..4; native WHT negative for Lanczos (wht_min -0.17 at s3) — benign for
  uniform flux.
- Square scale 1..4: SCI max ~= star peak at s>=2; ~48% "negative" pixels are
  float32 accumulation ripple of magnitude ~2e-5 relative (|min| ~0.02 on a
  1000-peak field), NOT ringing.
- High-contrast Lanczos x3 (ZSSS wiring): |SCI| > 1e4 (up to ~1.0e5) at star
  cores with small positive native WHT ((1e-3,1e-1] and (1e-1,1] bins) —
  signed-WHT denominator amplification; negative-WHT pixels also carry large
  |SCI| (would be excluded by the ZSSS WEIGHT_EPSILON support mask).
- Explicit `pixel_scale_ratio=scale` changes the Lanczos/gaussian coverage
  geometry (negative-WHT fraction drops ~0.48 -> ~0.22 at scale 3) but does
  not by itself remove the peak amplification in these cells.
- Continuous == checkpoint+resume: bit-identical sci/wht for square x3 and
  lanczos2 x3.
- Exposure 10/20/30 s square: SCI = counts/exptime rate (no double scaling).

See `docs/`-adjacent durable report
`/home/tristan/.openclaw/workspace/.a2a-reports/zsss-drizzle-contract-20260909.coco.r0.md`.
