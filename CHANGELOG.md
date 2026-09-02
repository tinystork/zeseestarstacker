# Changelog

All notable changes to ZeSeestarStacker are documented in this file.

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
