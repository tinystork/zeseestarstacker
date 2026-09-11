# Changelog

All notable changes to ZeSeestarStacker are documented in this file.

## [8.5.0] — Phoenix consedit

- corrected Drizzle pixel-scale geometry
- Lanczos catastrophic cancellation closure without a conditioning threshold
- truthful pixfrac kernel-aware UX
- signed float histogram/display fixes
- final scientific histogram
- adaptive histogram zoom
- default viewer-compatible reversible float32 FITS export

## [8.4.0] — Phoenix consedit

Candidate / beta release.  These changes have NOT yet been validated on the
Windows W80 acceptance runs (no W80 / incident-witness execution is claimed
here).

- support-aware Classic overlap normalization: accepted/neutral
  `sky_mean` / `linear_fit` events (incl. fallbacks) are attributed to the
  REAL original FITS identity of each exposure
- durable per-run normalization diagnostics: bounded, append-only
  `normalization_diagnostics_<token>.jsonl` per run (real FITS basenames,
  allowlisted scalar records, fail-open writer)
- automatic CPU memory policy (AUTO): the engine resolves the Winsorized CPU
  memory budget at execution from the actual machine/workload state;
  the legacy "HQ RAM Limit (GB)" control is removed from the normal GUI and
  replaced by a read-only AUTO status; an expert override remains available
  ONLY through the explicit CLI/env seam
  (`ZSSS_CPU_MEMORY_OVERRIDE_BYTES`, provenance-visible)
- exact-N spatial CPU Winsorized fallback: memory pressure changes the tile
  GEOMETRY only — the scientific batch N is never reduced or subdivided
  (the historical N-subgroup heuristic is removed from the Winsorized path);
  the spatial driver slices every observation first and materializes only the
  current `N × tile_h × tile_w × C` cube, never a second full-frame N-cube
- end-to-end resolved memory-budget propagation through the CPU Winsorized
  chain (policy → batch → queue wrapper → worker → core): no hidden 1/2 GiB
  deep default on the production path
- transactional failed-batch handling: a failed Classic reduction is fatal
  and truthful — sources are never moved/consumed and committed state is
  never advanced (counter / count-file / partial / resume ledger / meta)
- improved memory/GPU provenance: `MEMORY_POLICY` per run and
  `CPU_WINSOR_MEMORY_DECISION` / `_RETRY` / `_REFUSAL` per CPU execution,
  alongside the existing truthful GPU execution records; allocation retries
  are explicitly bounded and record both recovery and exhaustion outcomes

## [8.3.0] — Phoenix consedit

- run provenance closure: the durable `DRIZZLE_CONFIG` log line now records the
  requested kernel/scale/pixfrac/WHT threshold alongside the runtime-effective
  values, so any requested ≠ effective divergence is visible in one
  machine-readable record
- Drizzle-kernel consistency: the Qt and Tk GUIs and the kernel allowlists no
  longer offer `tophat` (unsupported by drizzle 2.2.0); every user-facing list
  now matches the engine's `VALID_DRIZZLE_KERNELS` exactly
- Winsorized GPU qualification: the CuPy winsorized-sigma-clip reducer is
  implemented as an exact twin of the CPU reference and wired into the
  production Classic stacking dispatch (8.3.0 feature lineage).  GPU
  eligibility is workload/VRAM-dependent (same memory model admits larger
  stacks on larger GPUs); the CPU reference remains authoritative and any
  per-batch VRAM rejection falls back to the CPU automatically.  Drizzle
  accumulation itself is not GPU-accelerated, and the provenance records the
  truth (``stacking_mode_effective=drizzle_direct_accumulation`` /
  ``GPU_DECISION execution=not_executed`` on Drizzle runs).

## [Unreleased]

### Removed

- removed the obsolete `seestar.apply_denoise` (OpenCV-CUDA non-local-means
  denoising) public export as dead-surface cleanup — GPU acceleration is now
  CuPy-only for the sorting-based stacking reductions (kappa-sigma /
  linear-fit-clip / median / winsorized-sigma-clip since 8.3.0); it was never
  a supported public contract.

## [8.2.3] — Phoenix consedit

- histogram: preserve the complete HDR tail distribution with a dual-domain
  model — Reset/Full now shows the full sampled range (sparse extreme tails
  genuinely binned) instead of widening the axis around robust-only bars
- histogram: explicit persistent view state (auto / full / manual) so a
  Reset/Full choice survives successive previews and a manual zoom reconciles
  safely when the analysis domain shrinks
- histogram: documented the end-of-run lifecycle (the final histogram is the
  last live preview, never a silent FITS readback)
- retire the abandoned LiveStack mode (no production reachability)
- restore Light-theme readability by completing the disabled palette so
  disabled controls render dimmed and legible in both Light and Dark themes

## [8.2.2] — Phoenix consedit

- fix ZeSolver RGB reference WCS: canonicalize the transported WCS to its
  celestial 2D component so a solved NAXIS=3 cube is no longer rejected as
  non-celestial (no false ASTAP fallback)

## [8.2.1] — Phoenix consedit

- fixed long-run live-preview analysis drift
- fixed Classic/Reproject processing of inputs without pre-existing WCS
- preserve and propagate the solved immutable reference WCS across aligned batches
- ZeSolver can be used for Reproject without requiring ASTAP when operational

## [8.2.0] — Phoenix consedit

- hardened Drizzle photometric normalization across changing frame coverage
- native Drizzle science finalization with safe signed-weight handling
- qualified Lanczos2 and Lanczos3 signed-WHT behavior
- truthful requested/effective Drizzle kernel, pixfrac and WHT provenance
- optional Drizzle WHT companion export, disabled by default
- clarified WHT threshold policy: zero default and N/A for signed Lanczos kernels
- hardened full pytest-suite import isolation

## [8.1.0] — Phoenix consedit
- hierarchical stacking integrity and effective SUM/WHT semantics
- immutable registration reference and registration diagnostics
- stabilized Classic, Reproject and Drizzle preview architecture
- truthful exposure metadata including resume
- float RGB histogram with detachable interactive view
- deterministic Auto Stretch and Auto White Balance with live updates
- persistent zoom, rotation, pan and preview resolution across stack updates
- hardened PySide6 lifecycle, persistent diagnostics and actionable startup refusal UX
- validated with real Boring, resume and Drizzle witnesses

## [8.1.0b2]
- truthful exposure metadata across Classic, Reproject, Drizzle and resume
- stable raw-linear preview and float RGB histogram pipeline
- detachable interactive histogram with synchronized black/white points
- deterministic Auto Stretch and Auto White Balance with live preview updates
- persistent rotation, zoom, pan and preview resolution across stack updates
- hardened startup-refusal propagation and localized output-folder guidance

## [8.1.0b1]
- hierarchical stacking integrity and effective SUM/WHT semantics
- immutable registration reference
- passive registration diagnostics
- Drizzle registration/pre-warp cleanup
- hardened PySide6 run lifecycle
- persistent per-run diagnostics
- actionable startup refusal UX
- real M16 80-frame classic/Drizzle validation

## [8.0.0] — Phoenix consedit

- PySide6 GUI replaces Tkinter as the primary interface
- Tkinter retained temporarily as explicit fallback
- Drizzle UI parity restored
- Qt backend lifecycle and processing summary hardened
- settings moved to platform user-data paths
- System tab / theme / language integration

## [7.1.1] - 2026-08-22

### Fixed

- Version consistency: the product display version (`seestar.gui.settings` save
  and the DRZ batch debug string) is now derived from the package source of
  truth (`seestar.__version__` + `seestar.__codename__`) instead of a
  hardcoded literal.
