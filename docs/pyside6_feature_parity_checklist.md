# PySide6 Feature-Parity Checklist — Tk → Qt

- Repository: `tinystork/zeseestarstacker`
- Branch: `feature/pyside6-migration`
- Purpose: live, concise tracker for Tk → PySide6 parity. `[x]` = implemented and
  test-covered; `[ ]` = not yet implemented (do not assume it exists).
- Companion docs: `docs/pyside6_migration_audit_coco.md`,
  `docs/pyside6_migration_p0_report.md` (full audit + milestone plan).

## How to read

Only mark `[x]` what a lot actually delivered. A row that is partially done must
stay `[ ]` (or be split) so the matrix never over-claims. The authoritative
backend contract lives in `seestar/gui/run_config.py`; this checklist tracks the
*Qt shell* surface (`seestar/gui_qt/*`).

---

## 1. Batch size contract

| # | Item | Status | Notes |
|---|---|---|---|
| 1.1 | UI spinbox may display `0` | `[x]` | `batch_spin` range `-1..1_000_000`, default `0` |
| 1.2 | `0` → Auto sentinel `-1` before RunRequest/backend | `[x]` | `normalize_batch_size`; applied in `_effective_settings_state` |
| 1.3 | `0` + "Reproject and coadd" keeps `0` (batch-zero) | `[x]` | `reproject_coadd_final` → `allow_mode_zero` |
| 1.4 | `1` → boring/single-batch path, not refused in preflight | `[x]` | preflight no longer rejects `batch_size == 1` |
| 1.5 | `>= 2` → explicit batch | `[x]` | passes through unchanged |
| 1.6 | `1` → boring_stack subprocess / `stack_plan.csv` routing | `[x]` | `boring_route` (CSV parse + command) + `boring_runner` (QProcess); `batch_size == 1` routes to boring, never `RunController.start` |

## 2. Final combination + interdependencies

| # | Item | Status | Notes |
|---|---|---|---|
| 2.1 | Final-combine choices (mean / median / winsorized_sigma_clip / reproject / reproject_coadd) | `[x]` | `final_combine_combo` (Stacking tab) with the five historical Tk keys |
| 2.2 | `reproject_between_batches` ⇄ `reproject_coadd_final` mutual exclusion | `[x]` | single source of truth: `stack_final_combine` drives both flags via `final_combine_flags` |
| 2.3 | Drizzle / reproject / boring-thread interdependencies (button gating) | `[x]` | boring checkbox ↔ batch-size sync + spinbox lock; drizzle controls disabled/unchecked while boring mode active |

## 3. Solver dialog / readiness

| # | Item | Status | Notes |
|---|---|---|---|
| 3.1 | Solver choices `none` / `astap` / `zesolver` | `[x]` | `SOLVER_PREFERENCES` + combo |
| 3.2 | Solver gate truth table (ZeSolver operational / ASTAP fallback / block) | `[x]` | `settings_validation` reproduces `resolve_solver_gate` |
| 3.3 | ZeSolver operational-readiness probe (lazy, engine-free at import) | `[x]` | `solver_probe.probe_zesolver_operational`; injected into preflight |
| 3.4 | ASTAP selected but absent → block | `[x]` | preflight rejects empty `astap_path` |
| 3.5 | Solver settings *dialog* (Local Solvers window, readiness refresh UI) | `[x]` | `solver_dialog.SolverSettingsDialog` + `solver_service` (lazy public boundary); ASTAP frame gating, ZeSolver status/configure + deferred readiness refresh |
| 3.6 | ASTAP/ZeSolver config persistence via `solver_config` (engine bridge) | `[x]` | M21: on dialog accept the Qt shell persists the solver fields into the engine `solver_config` (`seestar_config.json`) via a lazy, accept-time import of `seestar.core.solver_config` (`load_config` → overlay → `save_config`, Tk-identical merge/legacy-migration semantics); solver choice stays a settings-surface value (no engine key). Qt JSON surface (§18, M19) is unchanged |

## 4. Browse / paths

| # | Item | Status | Notes |
|---|---|---|---|
| 4.1 | Browse input folder | `[x]` | `QFileDialog.getExistingDirectory` → `input_edit` |
| 4.2 | Browse output folder | `[x]` | `QFileDialog.getExistingDirectory` → `output_edit` |
| 4.3 | Reference image path | `[x]` | Stacking-tab field + FITS `QFileDialog.getOpenFileName` browse |
| 4.4 | Temp folder | `[x]` | Stack tab field + browse |
| 4.5 | Last-stack / last output path persistence | `[x]` | `last_stack_path` + input/output/temp/reference/filename persisted to the settings JSON on close/shutdown and re-applied on launch (M8) |

## 5. Inputs / folders / output / analyzer

| # | Item | Status | Notes |
|---|---|---|---|
| 5.1 | View inputs | `[x]` | non-backend Qt dialog (main + staged folders) |
| 5.2 | Add folder | `[x]` | stage + validate (input/output/subfolder-of-output) |
| 5.3 | Open output | `[x]` | `QDesktopServices.openUrl` (user-triggered) |
| 5.4 | Analyze (ZeAnalyser launch) | `[x]` | stdlib-only `seestar/gui_qt/analyzer_launch.py` seam + `MainWindow._on_analyse`; button enabled only for an existing input dir; non-blocking launch with `ZEANALYSER_COMMAND_FILE`; reference-return *consumption* seam + GUI-thread `QTimer` watcher (M25.5-A) that consumes a late `REFERENCE=`, updates the reference, prepares output and starts processing — Tk-parity, no busy-loop/thread |

## 6. Preview

| # | Item | Status | Notes |
|---|---|---|---|
| 6.1 | Preview image rendering (array → `QImage`) | `[x]` | `preview_render` (display-only) |
| 6.2 | White-balance controls | `[x]` | `preview_adjust` per-channel R/G/B gains (neutral 1.0) shown as slider + numeric spinbox pairs, `Auto WB` (mode-based gains from the display image, Tk parity) + `Reset`; Tk range 0.1–5.0 step 0.01; display-only, re-renders the derived preview immediately (M10/M13) |
| 6.3 | Stretch controls (linear / asinh / log / auto) | `[x]` | `stretch_combo` (Asinh default, Tk parity) + black/white/gamma slider + numeric spinbox controls (Tk defaults 0.01/0.99/1.0, ranges/steps 0–1/0.001 and 0.1–5/0.01) + `Auto Stretch` (percentile bp/wp from the WB-only image → Asinh) + `Reset Stretch`; `preview_adjust` tone curves reproduce the Tk `StretchPresets` math; display-only, re-renders the derived preview (M10/M13/M14) |
| 6.4 | Histogram | `[x]` | single live surface in the persistent right panel (`HistogramView` QWidget) fed by the *WB-only* derived image (Tk `image_data_wb` source); reproduces the Tk auto-zoom / reset-view / zoom / reset-zoom interactions + BP/WP line dragging → `rangeChanged`; localized stats label; the M10 duplicated Preview-controls-tab histogram was removed (M10/M14) |
| 6.5 | Zoom (Fit / 100% / 200% / 50%) | `[x]` | display-only `preview_view` + `MainWindow` view controls; percent zoom scales from the rotated native size, Fit preserves aspect ratio |
| 6.6 | Rotation (left / right 90°) | `[x]` | cumulative ±90° modulo 360; preserves source image; zoom reapplies after rotation |
| 6.7 | Pan | `[x]` | continuous mouse-wheel zoom (Tk `zoom_factor` 1.15, `MIN_ZOOM` 0.05 / `MAX_ZOOM` 15.0) + left-drag pan (viewport offset, unbounded) layered on the existing `preview_view` render path via `PreviewImageView` + `render_view(zoom_factor=…, pan_offset=…)`; display-only, never mutates `_preview_source` (M18, see §17) |
| 6.8 | Auto-load first FITS of the input folder (initial preview) | `[x]` | `MainWindow._try_show_first_input_image` + `seestar/gui_qt/initial_preview.py` (M12): folder restored from settings or chosen via Browse Input non-blockingly loads the first sorted `.fit`/`.fits` on a daemon thread (lazy `importlib` engine import), debayers 2D Bayer data (header `BAYERPAT` else `settings.bayer_pattern`), and delivers the image back via a queued Qt signal; missing/empty folder clears the preview with a localized message; redundant-reload guard skips an unchanged folder |
| 6.9 | Brightness / contrast / saturation + reset (display-only) | `[x]` | slider + numeric spinbox pairs with Tk defaults 1.0/1.0/1.0 and ranges/steps 0.1–3.0/0.01, 0.1–3.0/0.01, 0.0–3.0/0.01 + `Reset Adjust.`; pure-numpy reproduction of the Tk image-enhancement behaviour; display-only (M13) |

## 7. Progress / log / copy

| # | Item | Status | Notes |
|---|---|---|---|
| 7.1 | Progress bar (0..100) | `[x]` | queued `progress_changed` |
| 7.2 | Log view | `[x]` | read-only `QTextEdit` |
| 7.3 | Copy log to clipboard | `[x]` | `Copy Log` button next to the log area; copies the full plain-text log via `QApplication.clipboard()`, disabled while the log is empty, armed on the first log line, never mutates the log or run state |
| 7.4 | Elapsed / remaining (ETA) time surface | `[x]` | `elapsed_label` / `remaining_label` driven only by the existing progress lifecycle signals + an injectable monotonic clock; `progress_time` helper (`format_duration` + naive `estimate_remaining_seconds`, no divide-by-zero at 0%) |

## 8. Localisation

| # | Item | Status | Notes |
|---|---|---|---|
| 8.1 | FR / EN language switch | `[x]` | `language_combo` enabled + user-triggered; switching updates `_current_language_code()` (feeds ZeAnalyser `--lang`) and re-labels visible Qt strings in place (tabs, path/action buttons, progress/log labels, preview/View/Histogram/Actions group titles, section titles + representative Settings/Mosaic field labels) without rebuilding the window; English default, French ↔ English round-trip (M9) |
| 8.2 | `Localization` key parity | `[x]` | Qt-local `seestar/gui_qt/localization.py` (pure stdlib) holds a compact FR/EN mapping for the Qt shell surface; a parity guard asserts every registered key has both `en` and `fr`, and missing-key/unknown-language fallback never raises. Delta: the full Tk `Localization` dictionaries are intentionally NOT imported (kept Qt/Tk/engine-free); the remaining ~unmapped Settings field labels stay English |

## 9. Settings / geometry persistence

| # | Item | Status | Notes |
|---|---|---|---|
| 9.1 | Settings surface (full `QtSettingsState` mirror) | `[x]` | grouped, scrollable Settings tab |
| 9.2 | Window geometry save/restore | `[x]` | `saveGeometry()` → base64 `window_geometry` JSON key; `restoreGeometry()` on load; offscreen round-trip tested (M8) |
| 9.3 | Settings persistence (`seestar_settings.json` / XDG) | `[x]` | `settings_persistence` (pure stdlib) + `QtSettingsState.from_dict` coercion; platform-aware user-config default (Windows `%APPDATA%`, macOS `~/Library/Application Support`, Linux `$XDG_CONFIG_HOME`/`~/.config`, all under `ZeSeestarStacker/`) with non-destructive migration of a legacy CWD `seestar_settings.json` (M25.5-B), injectable path for tests (M8) |

## 10. Last stack / resume

| # | Item | Status | Notes |
|---|---|---|---|
| 10.1 | Last stack display / resume | `[x]` | M23: last-stack → output pre-fill parity (Tk `_on_last_stack_changed` guard, browse/manual/persisted-load), run-end Processing Summary dialog (backend-computed payload via worker→controller→MainWindow signal; lazy `final.fits` NIMAGES/TOTEXP, never astropy at gui_qt module level; shown after regular + boring runs), and engine auto-resume seam verified (Qt run path forwards `output_folder` → `start_processing` `output_dir`, so `_can_resume` sees the memmap/batches_count artifact set) |

## 11. Entry point

| # | Item | Status | Notes |
|---|---|---|---|
| 11.1 | Non-default Qt entry point (`seestar.qt_main`) | `[x]` | `--backend simulated|seestar` |
| 11.2 | Official entry point switched to Qt | `[ ]` | default stays Tk `seestar.main:main` |

## 12. Run lifecycle robustness

| # | Item | Status | Notes |
|---|---|---|---|
| 12.1 | RunController worker/thread lifecycle (queued signals) | `[x]` | M3/M4 seams |
| 12.2 | Cancel / stop | `[x]` | `cancel()` → worker/backend |
| 12.3 | Shutdown retains live thread on timeout (no destroy-while-running) | `[x]` | `shutdown()` returns `bool`, defers cleanup |
| 12.4 | Backend activation (`simulated` / `seestar`) | `[x]` | lazy `SeestarQueuedStackerBackend` |
| 12.5 | Real backend (`seestar`) progress/log bridge drains the engine's deferred `gui_event_queue` | `[x]` | `SeestarQueuedStackerBackend.run` drains the stacker's thread-safe queue (the Tk GUI drains it from a periodic `after` loop; the Qt bridge had no equivalent) so progress/log/preview callbacks reach the Qt GUI (M11) |

## 13. Shell topology (Tk layout parity)

| # | Item | Status | Notes |
|---|---|---|---|
| 13.1 | Left/right `QSplitter` (control panel + persistent preview/action panel) | `[x]` | `_build_central` + `_build_left_panel` / `_build_right_panel` |
| 13.2 | Scrollable left panel (language + tabs + progress/log) | `[x]` | `QScrollArea` wrapping language combo, `QTabWidget`, progress, log |
| 13.3 | Left tabs `Stacking` / `Expert` / `Preview controls` | `[x]` | replaces former `Stack`/`Settings`/`Preview`/`Log` top-level tabs |
| 13.4 | Persistent right panel (preview + metadata + view + histogram + actions) | `[x]` | stays visible across left-tab switches; the right-panel histogram surface (`right_histogram_group` / `right_histogram_status` / `right_histogram_view`) is the *single* live histogram surface (the duplicated Preview-controls-tab histogram was removed in M14) |
| 13.5 | Action buttons Start / Stop / Analyse / Solver / View Inputs / Add Folder / Open Output | `[x]` | Start/Stop/Solver/View Inputs/Add Folder/Open Output functional; Analyse disabled |
| 13.6 | Zoom / resolution / rotation controls (real interactivity) | `[x]` | zoom (Fit/100/200/50), resolution label (orig → displayed + zoom + rotation), rotate left/right; display-only, offscreen-tested |
| 13.7 | Language switch (FR/EN) | `[x]` | placeholder combo is now enabled and participates in the FR/EN switch (M9) |

---

## 14. Expert tab content parity (M15)

Per-control Tk `tab_expert` → Qt parity (the Tk controls live in
`seestar/gui/main_window.py` ~l.1520-1942; the Qt surface is the `Expert` tab
built by `MainWindow._settings_tab`).  `[x]` = delivered this lot (present +
correct label/default/range/enabler/reset); `[ ]` = gap with a note.

| # | Tk control (label → var) | Range / default | Qt equivalent | Status |
|---|---|---|---|---|
| 14.1 | `warning_label` ("Expert Settings!", red italic) | — | `expert_warning_label` (`expert_warning_text`) | `[x]` |
| 14.2 | "Feather inter-batch (radial blend)" → `apply_batch_feathering_var` | bool, `True` | `apply_batch_feathering` checkbox | `[x]` |
| 14.3 | "Enable Feathering" → `apply_feathering_var` | bool, `True` | `apply_feathering` checkbox (enabler) | `[x]` |
| 14.4 | "Blur (px)" → `feather_blur_px_var` | 32–512 step 16, `256` | `feather_blur_px` spinbox | `[x]` |
| 14.5 | "Apply Low WHT Mask" → `apply_low_wht_mask_var` | bool, `False` | `apply_low_wht_mask` checkbox (enabler) | `[x]` |
| 14.6 | "Percentile" → `low_wht_pct_var` | 1–100 step 1, `5` | `low_wht_percentile` spinbox | `[x]` |
| 14.7 | "Soften (px)" → `low_wht_soften_px_var` | 32–512 step 16, `128` | `low_wht_soften_px` spinbox | `[x]` |
| 14.8 | "Enable BN" → `apply_bn_var` | bool, `True` | `apply_bn` checkbox (enabler) | `[x]` |
| 14.9 | "Grid Size" → `bn_grid_size_str_var` | combo 8x8…64x64, `24x24` | `bn_grid_size_str` combo | `[x]` |
| 14.10 | "BG Perc. Low" → `bn_perc_low_var` | 0–40 step 1, `5` | `bn_perc_low` spinbox | `[x]` |
| 14.11 | "BG Perc. High" → `bn_perc_high_var` | 10–95 step 1, `40` | `bn_perc_high` spinbox | `[x]` |
| 14.12 | "BG Std Factor" → `bn_std_factor_var` | 0.5–5.0 step 0.1, `1.5` | `bn_std_factor` spinbox | `[x]` |
| 14.13 | "Min Gain" → `bn_min_gain_var` | 0.1–2.0 step 0.1, `0.2` | `bn_min_gain` spinbox | `[x]` |
| 14.14 | "Max Gain" → `bn_max_gain_var` | 1.0–10.0 step 0.1, `7.0` | `bn_max_gain` spinbox | `[x]` |
| 14.15 | "Enable Edge/Chroma Correction" → `apply_cb_var` | bool, `True` | `apply_cb` checkbox (enabler) | `[x]` |
| 14.16 | "Border Size (px)" → `cb_border_size_var` | 5–150 step 5, `25` | `cb_border_size` spinbox | `[x]` |
| 14.17 | "Blur Radius (px)" → `cb_blur_radius_var` | 0–50 step 1, `8` | `cb_blur_radius` spinbox | `[x]` |
| 14.18 | "Min Blue Factor" → `cb_min_b_factor_var` | 0.1–1.0 step 0.05, `0.4` | `cb_min_b_factor` spinbox | `[x]` |
| 14.19 | "Max Blue Factor" → `cb_max_b_factor_var` | 1.0–3.0 step 0.05, `1.5` | `cb_max_b_factor` spinbox | `[x]` |
| 14.20 | "Enable Final Cropping" → `apply_final_crop_var` | bool, `True` | `apply_final_crop` checkbox (enabler) | `[x]` |
| 14.21 | "Edge Crop (%)" → `final_edge_crop_percent_var` | 0.0–25.0 step 0.5, `2.0` | `final_edge_crop_percent` spinbox | `[x]` |
| 14.22 | "Crop master tiles" → `apply_master_tile_crop_var` (Stacking tab in Tk) | bool, `False` | `apply_master_tile_crop` checkbox (enabler) | `[x]` |
| 14.23 | "Crop % per side" → `master_tile_crop_percent_var` | 0.0–25.0 step 0.5, `18.0` | `master_tile_crop_percent` spinbox | `[x]` |
| 14.24 | "Enable Photutils 2D Bkg Subtraction" → `apply_photutils_bn_var` | bool, `False` | `apply_photutils_bn` checkbox (enabler) | `[x]` |
| 14.25 | "Box Size (px)" → `photutils_bn_box_size_var` | 16–1024 step 16, `128` | `photutils_bn_box_size` spinbox | `[x]` |
| 14.26 | "Filter Size (px, odd)" → `photutils_bn_filter_size_var` | 1–15 step 2, `11` | `photutils_bn_filter_size` spinbox | `[x]` |
| 14.27 | "Sigma Clip Value" → `photutils_bn_sigma_clip_var` | 1.0–5.0 step 0.1, `3.0` | `photutils_bn_sigma_clip` spinbox | `[x]` |
| 14.28 | "Exclude Brightest (%)" → `photutils_bn_exclude_percentile_var` | 0.0–100.0 step 1.0, `95.0` | `photutils_bn_exclude_percentile` spinbox | `[x]` |
| 14.29 | "Save final FITS as float32" → `save_as_float32_var` | bool, `False` | `save_final_as_float32` checkbox | `[x]` |
| 14.30 | "Preserve linear output" → `preserve_linear_output_var` | bool, `False` | `preserve_linear_output` checkbox | `[x]` |
| 14.31 | `reset_expert_button` ("Reset Expert Settings") | — | `reset_expert_button` (`reset_expert_button`) | `[x]` |

Notes / gaps:

- **Enablers gate sub-options exactly like Tk** (14.3/14.5/14.8/14.15/14.20/
  14.22/14.24): unchecked disables the gated widgets via
  `_update_expert_enabler_states` (the Qt equivalent of Tk
  `_update_*_options_state` / `_update_master_tile_crop_state`).  `[x]`
- **Reset-to-defaults** (14.31) restores the BN / CB / master-tile-crop /
  final-crop / feathering / batch-feathering / low-weight-mask / Photutils-BN
  widgets to their `QtSettingsState` defaults (`_reset_expert_settings`, GUI
  state only).  Output-format fields (`save_final_as_float32` /
  `preserve_linear_output`) are deliberately **not** reset, matching the Tk
  button.  *Deviation:* the Tk button also omits the Low WHT Mask group; the
  Qt button resets it too so "Reset Expert Settings" actually restores the
  whole Expert surface (justified, documented).  `[x]`
- **Engine-coupled status.**  The numeric BN / CB / crop / Photutils /
  feathering parameters already reach the backend through the existing M10
  `settings_state` → `build_backend_kwargs` path, so they are wired, not
  display-only; whether the engine *applies* them during a run is the
  pre-existing backend E2E scope and is unchanged by this lot.  The three
  enabler flags `apply_bn` / `apply_cb` / `apply_final_crop` are **display-only
  now** (gating + persistence): `build_backend_kwargs` does not consume them
  today (verified), so wiring them to the engine is deferred to a later
  backend E2E milestone if needed.  `[ ]` backend E2E for enabler flags.
- **Default-value divergence (Tk init vars vs `SettingsManager`).**  The Tk
  `init_variables` seeds a few Expert vars with values that differ from
  `SettingsManager.get_default_values` (e.g. `bn_grid_size_str` `"16x16"` vs
  `"24x24"`, `bn_perc_high` `30` vs `40`, `bn_std_factor` `1.0` vs `1.5`,
  `photutils_bn_filter_size` `5` vs `11`, `photutils_bn_exclude_percentile`
  `98.0` vs `95.0`).  The Qt shell seeds every widget from `QtSettingsState`
  (aligned with `SettingsManager`), which is also what the Tk reset/apply path
  uses; the transient Tk init-var values are not reproduced.  `[x]` (documented
  divergence, canonical default = `SettingsManager`).

---

## 15. Stacking tab content parity (M16)

Per-control Tk `tab_stacking` → Qt parity (the Tk controls live in
`seestar/gui/main_window.py` ~l.957-1516; the Qt surface is the `Stacking` tab
built by `MainWindow._build_stacking_tab` plus the `Expert` tab "Stacking /
Paths", "Drizzle Advanced", "Calibration / Hot Pixels" and "Colour /
Post-processing" sections, which host the Tk Stacking-tab sub-options moved
there by M10/M15).  `[x]` = delivered this lot (present + correct
label/default/range/enabler); `[ ]` = gap with a note.

| # | Tk control (label → var) | Range / default | Qt equivalent | Status |
|---|---|---|---|---|
| 15.1 | "Input:" → `input_path` | str | `input_edit` + Browse (Stacking tab) | `[x]` |
| 15.2 | "Output:" → `output_path` | str | `output_edit` + Browse | `[x]` |
| 15.3 | "Filename:" → `output_filename_var` | str | `output_filename_edit` | `[x]` |
| 15.4 | "Reference" → `reference_image_path` | str | `reference_edit` + Browse | `[x]` |
| 15.5 | "Last stack:" → `last_stack_path` | str | `last_stack_edit` + Browse | `[x]` |
| 15.6 | "Temporary:" → `temp_folder_path` | str | `temp_edit` + Browse | `[x]` |
| 15.7 | "Crop master tiles" → `apply_master_tile_crop_var` | bool, `False` | `apply_master_tile_crop` checkbox (enabler, Expert tab) | `[x]` (M15) |
| 15.8 | "Crop % per side" → `master_tile_crop_percent_var` | 0.0–25.0 step 0.5, `18.0` | `master_tile_crop_percent` (Expert tab) | `[x]` (M15) |
| 15.9 | "Normalization:" → `stack_norm_method_var` | combo none/linear_fit/sky_mean, `none` | `stack_norm_method` combo (Expert tab) | `[x]` |
| 15.10 | "Weighting:" → `stack_weight_method_var` | combo none/noise_variance/noise_fwhm/snr/stars, `none` | `stack_weight_method` combo (Expert tab) | `[x]` |
| 15.11 | "Kappa Low:" → `stacking_kappa_low_var` | 0.1–10 step 0.1, `3.0` | `stack_kappa_low` (Expert tab, 0.0–10.0) | `[x]` (range diff) |
| 15.12 | "Kappa High:" → `stacking_kappa_high_var` | 0.1–10 step 0.1, `3.0` | `stack_kappa_high` (Expert tab, 0.0–10.0) | `[x]` (range diff) |
| 15.13 | "Winsor Limits:" → `stacking_winsor_limits_str_var` | str `0.05,0.05` | `stack_winsor_limits` (Expert tab) | `[x]` |
| 15.14 | "Final Combine:" → `stack_final_combine_var` | combo mean/median/winsorized_sigma_clip/reproject/reproject_coadd, `mean` | `final_combine_combo` (Stacking tab) | `[x]` |
| 15.15 | "HQ RAM limit (GB)" → `max_hq_mem_var` | 1–64 step 1, `8.0` | `max_hq_mem_spin` (Stacking tab) | `[x]` display-only, `[x]` backend (M20) |
| 15.16 | "Method:" → `stack_method_var` | combo mean/median/kappa_sigma/winsorized_sigma_clip/linear_fit_clip, `kappa_sigma` | `stacking_mode_combo` (Stacking tab, backend keys) | `[x]` (label diff) |
| 15.17 | "Apply Final SCNR (Green)" → `apply_final_scnr_var` | bool, `False` | `apply_final_scnr` checkbox (enabler, Expert tab) | `[x]` (enabler; default diff) |
| 15.18 | SCNR amount → `final_scnr_amount_var` | 0–1 step 0.05, `0.8` | `final_scnr_amount` (Expert tab) | `[x]` (default diff) |
| 15.19 | "Preserve Luminosity (SCNR)" → `final_scnr_preserve_lum_var` | bool, `True` | `final_scnr_preserve_luminosity` (Expert tab) | `[x]` |
| 15.20 | "Batch Size:" → `batch_size` | 0–9999 step 1, `10` | `batch_spin` (Stacking tab, `-1..1_000_000`, `0`=Auto) | `[x]` (contract §1) |
| 15.21 | "Threaded Boring Stack" → `boring_thread_var` | bool, `False` | `boring_check` (Stacking tab) | `[x]` |
| 15.22 | "Enable Drizzle" → `use_drizzle_var` | bool, `False` | `drizzle_check` (Stacking tab) | `[x]` |
| 15.23 | "Drizzle processing:" → `drizzle_mode_var` | radio Standard(`Final`)/Large dataset(`Incremental`), `Final` | `drizzle_mode_combo` (`Final`/`Incremental`) | `[x]` (radio vs combo) |
| 15.24 | "Preview group size:" → `drizzle_group_size_var` | 1–100000 step 10, `50` | `drizzle_group_spin` (1–100000 step 10, `50`) | `[x]` |
| 15.25 | Drizzle policy hint (grey, wrapped) | — | `drizzle_policy_hint` (Stacking tab) | `[x]` (this lot) |
| 15.26 | "Scale:" → `drizzle_scale_var` | radio x2/x3/x4, `2` | `drizzle_scale` (Expert tab, int 1–10) | `[x]` (radio vs spinbox) |
| 15.27 | "WHT Threshold %:" → `drizzle_wht_display_var`→`drizzle_wht_threshold_var` | 10–100 step 5 (%), `0.7` | `drizzle_wht_threshold` (Expert tab, float 0–1) | `[x]` (% vs float) |
| 15.28 | "Kernel:" → `drizzle_kernel_var` | combo 7 kernels, `square` | `drizzle_kernel` combo (Expert tab) | `[x]` |
| 15.29 | "Pixfrac:" → `drizzle_pixfrac_var` | 0.01–2.00 step 0.05, `1.0` | `drizzle_pixfrac` (Expert tab) | `[x]` |
| 15.30 | "Use GPU" → `use_gpu_var` | bool, `False` | `use_gpu_check` (Stacking tab) | `[x]` display-only, `[x]` backend (M20) |
| 15.31 | "Correct hot pixels" → `correct_hot_pixels` | bool, `True` | `correct_hot_pixels` checkbox (Expert tab) | `[x]` |
| 15.32 | "Threshold:" → `hot_pixel_threshold` | 1–10 step 0.1, `3.0` | `hot_pixel_threshold` (Expert tab, 0.5–10) | `[x]` (range diff) |
| 15.33 | "Neighborhood:" → `neighborhood_size` | 3–15 step 2, `5` | `neighborhood_size` (Expert tab, 1–20 step 1) | `[x]` (range/step diff) |
| 15.34 | "Cleanup temporary files" → `cleanup_temp_var` | bool, `True` | `cleanup_temp` checkbox (Expert tab) | `[x]` |
| 15.35 | "Edge Enhance" → `apply_chroma_correction_var` | bool, `True` | `apply_chroma_correction` checkbox (Expert tab) | `[x]` |

Notes / gaps:

- **Drizzle enabler gating mirrors Tk** (15.22–15.30): the Enable-drizzle flag
  gates the drizzle mode combo, the group-size spin, the new Use-GPU checkbox
  and the Expert-tab "Drizzle Advanced" sub-options (scale / WHT threshold /
  kernel / pixfrac) via `_update_drizzle_gating` (the Qt equivalent of Tk
  `_update_drizzle_options_state`).  **Group-size gate** (M3-D): the group-size
  spinbox is enabled only when drizzle is on *and* the mode is
  `Incremental` (Large dataset), matching the Tk policy.  The existing M4
  boring-mode gate (`_update_boring_gating`) now delegates to the drizzle gate
  so boring mode still force-disables+unchecks drizzle while the Tk-parity
  drizzle flag takes over the sub-option enablement.  `[x]`
- **SCNR enabler gating** (15.17–15.19): `apply_final_scnr` now gates
  `final_scnr_target_channel` / `final_scnr_amount` /
  `final_scnr_preserve_luminosity` (added to `EXPERT_ENABLER_GATES`), mirroring
  the Tk `_update_final_scnr_options_state`.  `[x]`
- **Engine-coupled handoff (15.15 / 15.30 backend, M20).**
  `use_gpu` (15.30) and `max_hq_mem_gb` (15.15) are added to `QtSettingsState`
  and surfaced on the Stacking tab (persisted/collected like Tk).  M20 wires
  them into the Qt run path as *seam-only* fields on the `RunRequest` (the
  canonical `build_backend_kwargs` is deliberately left unchanged, so the Tk
  flow stays byte-identical): `MainWindow.build_run_request` attaches them via
  `run_handoff.attach_run_settings`, and `SeestarQueuedStackerBackend` applies
  `use_gpu` → `stacker.use_gpu` and `max_hq_mem_gb` →
  `stacker.max_hq_mem` (bytes) before `start_processing`, never as
  `start_processing` kwargs (see §19).  The boring single-batch subprocess
  route forwards the same `max_hq_mem_gb` as `--max-mem` since M25 (see §20.4),
  so the configured HQ-RAM limit reaches both run paths.  `[x]` backend.
- **Kappa / Winsor visibility (Tk `_toggle_kappa_visibility`) is not reproduced.**
  The Tk Stacking tab hides the Kappa Low/High and Winsor-Limits controls unless
  the stacking method or final-combine is `kappa_sigma` /
  `winsorized_sigma_clip`; the Qt Expert tab shows them always (they are always
  present in `build_backend_kwargs`, so this is purely cosmetic).  `[ ]`
- **Control-shape deltas** (present, but different widget): drizzle mode is a
  combo (`Final`/`Incremental`) instead of the Tk "Standard"/"Large dataset /
  incremental" radio pair (15.23); drizzle scale is an int spinbox 1–10 instead
  of the Tk x2/x3/x4 radio (15.26); the WHT threshold is a raw 0–1 float instead
  of a 10–100 % display that converts to 0–1 (15.27).  The stored backend keys /
  defaults are identical.
- **Default-value divergence (Tk init vars vs `SettingsManager`).**  The Tk
  `init_variables` seeds `apply_final_scnr_var=False` / `final_scnr_amount_var=
  0.8` / `max_hq_mem_var=8` (DoubleVar) / `use_gpu_var=False`, while the
  canonical `SettingsManager.get_default_values` seeds `apply_final_scnr=True` /
  `final_scnr_amount=0.6` / `max_hq_mem_gb=8` / `use_gpu=False`.  The Qt shell
  seeds every widget from `QtSettingsState` (aligned with `SettingsManager`),
  matching the M15 convention; the transient Tk init-var values are not
  reproduced.  `[x]` (documented divergence).
- **Reset behaviour.**  The Tk Stacking tab has no reset button (only the
  Expert tab does, via `reset_expert_button`, M15); the Qt Stacking tab
  likewise has none.  `[x]` (N/A, verified).

---

## 16. Right panel / preview toolbar parity (M17)

Per-control Tk right-panel / preview-toolbar → Qt parity (the Tk controls live
in `seestar/gui/main_window.py` ~l.2113-2320 plus the `PreviewManager` canvas
interactions in `seestar/gui/preview.py`; the Qt surface is
`MainWindow._build_right_panel` plus `preview_view` / `preview_adjust` /
`histogram_view`).  `[x]` = delivered this lot (present + correct label /
behaviour); `[ ]` = gap with a note.  "engine-coupled" marks controls whose Tk
behaviour reaches the stacking engine (`queued_stacker` / run backend), which
the Qt shell reproduces either as a functional action or as a display-only
equivalent with the backend part deferred.

| # | Tk control (label → target) | Behaviour / coupling | Qt equivalent | Status |
|---|---|---|---|---|
| 16.1 | `start_button` "Start" → `start_processing` | starts a run (engine-coupled) | `start_button` | `[x]` |
| 16.2 | `stop_button` "Stop" → `stop_processing` | cancels a run (engine-coupled) | `stop_button` | `[x]` |
| 16.3 | `analyze_folder_button` "Analyze Input Folder" → `_launch_folder_analyzer` | external ZeAnalyser launch | `analyse_button` (M7; label "Analyse") | `[x]` (label diff) |
| 16.4 | `local_solver_button` "Local Solvers..." → `_open_local_solver_settings_window` | solver dialog | `solver_button` (M2; label "Solver") | `[x]` (label diff) |
| 16.5 | `open_output_button` "Open Output" → `_open_output_folder` | open output dir | `open_output_button` | `[x]` |
| 16.6 | `add_files_button` "Add Folder" → `file_handler.add_folder` | stage a folder | `add_folder_button` | `[x]` |
| 16.7 | `show_folders_button` "View Inputs" → `_show_input_folder_list` | list staged folders | `view_inputs_button` | `[x]` |
| 16.8 | `histogram_widget` (HistogramWidget) | live histogram + BP/WP drag | `right_histogram_view` (single surface, M14) | `[x]` |
| 16.9 | `auto_zoom_histo_check` "Auto zoom histogram" | auto-zoom toggle | `auto_zoom_histo_check` | `[x]` |
| 16.10 | `hist_reset_view_btn` "Reset Histogram" | reset view | `hist_reset_view_button` | `[x]` |
| 16.11 | `hist_zoom_btn` "Zoom Histogram" | zoom into histogram | `hist_zoom_button` | `[x]` |
| 16.12 | `hist_reset_btn` "R" | reset zoom | `hist_reset_button` | `[x]` |
| 16.13 | `preview_canvas` (Canvas) | preview image surface | `preview_image_label` (QLabel) | `[x]` |
| 16.14 | `zoom_100_button` "Zoom 100%" / `zoom_fit_button` "Zoom Fit" | discrete zoom (engine-free) | `zoom_combo` (`Fit`/`100%`/`200%`/`50%`) | `[x]` (shape diff) |
| 16.15 | `preview_res_button` "Res 1/1..1/4" → `_cycle_preview_resolution` | cycles preview resolution (engine-coupled: `set_preview_downsample_factor` + `refresh_preview`) | `preview_res_button` | `[x]` display-only, `[x]` backend E2E live control (M22), `[x]` display reconciliation (M24) |
| 16.16 | `rotate_left_button` / `rotate_right_button` | rotate preview ±90° (engine-free) | `rotate_left_button` / `rotate_right_button` | `[x]` |
| 16.17 | Pan (left-drag / scroll) via `PreviewManager` | canvas pan + scroll-zoom (engine-free) | `PreviewImageView` + `render_view(zoom_factor=…, pan_offset=…)` | `[x]` (M18, checklist 6.7, §17) |
| 16.18 | Save / export | no dedicated right-panel button; the run saves FITS/PNG via the engine | run backend + `open_output_button` | `[x]` (N/A) |

Notes / gaps:

- **Res-cycle button (16.15).**  The Qt right panel has a `preview_res_button`
  that cycles factors 1→2→3→4→1 with the Tk `Res 1/N` label (localized
  `Res`/`Rés`), persisted in GUI state (`_preview_res_factor`), and — via the
  display-only `downsampled_image` / `render_view(..., downsample_factor=...)`
  seam — re-renders the local preview at the new factor; it never mutates
  `_preview_source`.  **Engine coupling (M22, delivered):** during an active
  run the button now also forwards the factor through a thread-safe control
  channel (GUI → `RunController` → `RunWorker` → backend →
  `stacker.set_preview_downsample_factor` + `refresh_preview`, applied on the
  worker thread), matching the Tk `_cycle_preview_resolution` engine coupling;
  idle clicks stay display-only.  **Display reconciliation (M24, delivered):**
  the on-screen preview now reflects the engine factor 1:1 during an active
  run — the render path (`_refresh_preview_view` /
  `_effective_preview_downsample_factor`) uses factor `1` while a run is active
  (the engine already pushed frames at its own `preview_downsample_factor`
  resolution, Tk parity: Tk renders the engine-pushed frames directly with no
  second downsample) and restores the local display-only factor
  (`_preview_res_factor`) on run end; idle (no run) display-only local factor
  behaviour is preserved byte-identically (M17/M18).  No new preview-frame
  pipeline, no IPC/protocol change, no Res-button label/cycle change.
  **Default deviation:** the Qt default factor is `1` (native) because the Qt
  shell's display-only preview is never engine-downsampled and the existing
  preview tests lock native rendering; the Tk initial factor is `2` (engine
  `preview_downsample_factor` default).  `[x]` display-only (M17), `[x]`
  backend E2E live control (M22), `[x]` display reconciliation (M24).
- **Kappa/winsor visibility (M16 gap closed).**  `_toggle_kappa_visibility` now
  mirrors the Tk `_toggle_kappa_visibility` rule exactly: `stack_kappa_low` /
  `stack_kappa_high` show for stacking method `kappa-sigma` /
  `winsorized-sigma-clip` or final-combine `winsorized_sigma_clip`;
  `stack_winsor_limits` shows for stacking method `winsorized-sigma-clip` or
  final-combine `winsorized_sigma_clip`.  Purely cosmetic: the values stay in
  the shared model and `build_backend_kwargs` always passes
  `stack_kappa_low` / `stack_kappa_high` / `winsor_limits` regardless of
  visibility.  The standalone `kappa` field is **not** part of the Tk kappa
  frame and stays always visible.  `[x]`
- **Pan (16.17, checklist 6.7).**  Tk pan/scroll-zoom is a pure display
  interaction inside `PreviewManager` (mouse-wheel zoom, left-drag pan on the
  Canvas) with **no** engine coupling.  Delivered in M18 (§17): the Qt preview
  surface (`preview_image_label`) is now a `PreviewImageView` that emits
  wheel-zoom / pan-delta signals, and `render_view` layers a continuous
  `zoom_factor` + `pan_offset` on top of the existing render path.  `[x]`
- **Solver-config persistence (3.6).**  The Qt-side JSON persistence is
  delivered (M19, §18): the ASTAP / ZeSolver fields live in `QtSettingsState`
  and round-trip through the M8 settings surface (dialog prefill ⇄ accept →
  JSON).  The *engine* `solver_config` bridge is now delivered too (M21): on
  dialog accept the Qt shell also writes the engine `seestar_config.json` via
  a lazy import of `seestar.core.solver_config` (`load_config` → overlay →
  `save_config`), so the engine config becomes the runtime-consumed solver
  source while the Qt JSON surface stays the display/state source.  `[x]` Qt
  JSON surface (M19), `[x]` engine bridge (M21).

---

## 17. Preview pan / zoom parity (M18)

Per-behaviour Tk `PreviewManager` pan/zoom → Qt parity.  The Tk interaction
source is `PreviewManager` in `seestar/gui/preview.py` (mouse-wheel zoom +
left-drag pan over `preview_canvas`, a pure view transform); the Qt surface is
`preview_image_view.PreviewImageView` (a `QLabel` emitting `wheelZoom` /
`panDelta`) plus `preview_view.render_view(zoom_factor=…, pan_offset=…)`.
`[x]` = delivered this lot (present + correct semantics); "engine-coupled"
marks behaviours that reach the stacking engine — none are expected here.

| # | Tk `PreviewManager` behaviour | Semantics | Qt equivalent | Status |
|---|---|---|---|---|
| 17.1 | `_zoom_on_scroll` (`<MouseWheel>`/`<Button-4>`/`<Button-5>`) | wheel zoom ×1.15 (up) / ÷1.15 (down), clipped to `[0.05, 15.0]`, cursor-anchored | `PreviewImageView.wheelZoom` → `MainWindow._on_wheel_zoom` (uses `ZOOM_STEP`/`MIN_ZOOM`/`MAX_ZOOM` + cursor-anchored pan shift) | `[x]` |
| 17.2 | `_start_pan`/`_pan_image`/`_stop_pan` (`<ButtonPress-1>`/`<B1-Motion>`) | left-drag pan, viewport offset, **no clamping** | `PreviewImageView.panDelta` → `MainWindow._on_pan_delta` (unbounded `_view_offset_x/y`) | `[x]` |
| 17.3 | `reset_zoom_and_pan` | zoom → 1.0, pan → (0,0) | `MainWindow._reset_view_transform` (on new preview image / clear) | `[x]` |
| 17.4 | `zoom_fit` | fit to canvas + reset pan | `zoom_combo` "Fit" (fit render + pan reset) | `[x]` |
| 17.5 | `zoom_full_size` | 100% (`zoom_level = 1.0`) | `zoom_combo` "100%" preset (`_preview_zoom_factor = 1.0`) | `[x]` (shape diff: combo) |
| 17.6 | `rotate_left` / `rotate_right` reset pan | rotation resets pan (aspect flip) | `_on_rotate_left/right` reset `_view_offset_x/y`, keep zoom | `[x]` |
| 17.7 | engine coupling | **none** — pure view transform | **none** — no engine/Tk/backend import or call | `[x]` (confirmed) |

Notes / rules:

- **Single continuous zoom factor.**  `MainWindow._preview_zoom_factor` (default
  `1.0`, range `[0.05, 15.0]`) is the single source of truth for the numeric
  zoom, mirroring Tk `zoom_level`.  The `zoom_combo` percent presets
  (`100%`/`200%`/`50%`) and the mouse-wheel both set this same factor; "Fit" is
  a combo *mode* (not a numeric factor) handled by the aspect-fit render path.
- **Wheel zoom ↔ combo interaction rule.**  A wheel turn multiplies/divides the
  continuous factor by `ZOOM_STEP` (1.15) and re-syncs the combo: an exact
  preset match shows that preset, anything else shows a blank (custom) combo so
  the combo never lies about a non-preset zoom.  Picking any combo preset
  returns to that preset **and recentres** (resets pan to 0).  Wheeling from
  "Fit" exits Fit and continues from the current fit scale (Tk `zoom_fit` sets
  `zoom_level` to the fit scale).  This is the documented "wheel zoom sets a
  level outside the presets; the combo returns to a preset" option.
- **Pan is unbounded** (Tk applies no clamping); the offset is a pure viewport
  translation, and `compose_panned_pixmap` clips the scaled image to the
  viewport.  When not panned, the pixmap stays the exact scaled image size
  (existing M5/M13/M17 dimension tests are preserved); a non-zero pan composes
  into a viewport-sized canvas.
- **Reset semantics.**  Pan/zoom reset to `100%` + centred on every new preview
  image (`_on_preview` / `_on_initial_preview_result` / `_clear_preview`), and
  "Zoom Fit" (combo "Fit") recentres (resets pan).  Rotation resets pan but
  keeps the zoom factor (Tk parity).
- **Display-only / engine-coupled.**  `[x]` confirmed engine-free: `_preview_source`
  is never mutated (all helpers copy), no FITS/PNG writes, no settings-file /
  backend changes, and `preview_image_view.py` / `preview_view.py` contain no
  engine/Tk/zesolver import path (asserted by the M18 tests + the existing
  import-hygiene tests).

---

## 18. Solver settings persistence (Qt JSON surface) (M19)

Settings-only persistence of the solver-dialog fields through the M8 JSON
settings surface.  The ASTAP / ZeSolver values are held in `QtSettingsState`
and round-trip through `settings_persistence` (the Qt shell's own
`seestar_settings.json`).  From M21 the *same* accepted values are also bridged
into the engine `solver_config` (`seestar_config.json`); the Qt JSON surface
remains the display/state source and the engine config becomes the
runtime-consumed source (see the M21 note below).  `[x]` = delivered this lot
(M19: Qt JSON surface; M21: engine bridge).

| Field | Tk GUI default (`SettingsManager`) | Engine default (`solver_config.DEFAULT_CONFIG`) | Qt default (`QtSettingsState`) | Persisted key | Round-trip test |
|---|---|---|---|---|---|
| solver preference | `none` | (n/a — legacy `ansvr`/`astrometry` map to `zesolver`) | `none` | `local_solver_preference` | `test_solver_fields_round_trip_through_settings_persistence` |
| ASTAP executable | `""` | `astap_executable_path` = `""` | `""` | `astap_path` | same |
| ASTAP data dir | `""` | `astap_data_directory_path` = `""` | `""` | `astap_data_dir` | same |
| search radius (deg) | `3.0` | `astap_default_search_radius` = `3.0` | `3.0` | `astap_search_radius` | same |
| downsample | `1` | `astap_default_downsample` = `2` | `1` | `astap_downsample` | same |
| sensitivity | `100` | `astap_default_sensitivity` = `100` | `100` | `astap_sensitivity` | same |

Notes / decisions:

- **Default-value divergence (documented).**  The Qt `astap_downsample` default
  is `1` (matching the Tk `SettingsManager` default and the Qt solver dialog),
  while the engine `solver_config` `astap_default_downsample` default is `2`.
  The Qt shell keeps `1` because it is the canonical Tk GUI default (enforced
  by `test_defaults_aligned_with_settings_manager`).  Since M21 the Qt shell
  *does* write the engine config on accept, so an unmodified default accept
  propagates `astap_default_downsample = 1` into the engine config — a
  documented, self-consistent consequence of the GUI default.  `[x]`
  (documented divergence).
- **Engine bridge (M21).**  Persistence goes through the M8 JSON surface *and*
  (on accept) through the engine `solver_config` via a lazy, function-scoped
  import in `seestar/gui_qt/solver_config_bridge.py`; the engine module path is
  assembled from split string literals so `import seestar.gui_qt` stays
  engine-free (fresh-process hygiene test).  `check_zesolver_readiness` /
  `probe_zesolver_operational` / `open_zesolver_configuration` are still never
  called for persistence, and `seestar/core/solver_config.py` is untouched
  (asserted by `test_no_writes_to_engine_solver_config` for the JSON surface
  and by `tests/test_qt_solver_bridge_m21.py` for the bridge).  `[x]`.
- **Dialog wiring.**  `MainWindow._on_solver` opens
  `SolverSettingsDialog(self, self.collect_settings_state())` (prefill from the
  current `QtSettingsState`); on accept it writes the dialog values back into
  the live Qt controls via `_apply_solver_dialog_values`, which fold into
  `settings_state` and thus survive the `_save_persisted_settings` save →
  `_load_persisted_settings` load round-trip, and then calls
  `solver_config_bridge.write_solver_config(values)` to persist the mapped
  values into the engine config (Tk `_on_ok` timing; cancel/ESC writes
  nothing).  `[x]`.

---

## 19. Qt run-settings handoff — `use_gpu` / `max_hq_mem_gb` (M20)

Backend E2E part 1: the Qt run flow consumes the Qt-collected `use_gpu` (15.30)
and `max_hq_mem_gb` (15.15) settings.  The canonical shared builder
`seestar.gui.run_config.build_backend_kwargs` is deliberately **unchanged** (so
the Tk flow stays byte-identical); the Qt shell attaches the two values to its
`RunRequest` as *seam-only* fields (the same pattern as `stack_final_combine`)
and the Qt backend adapter applies them to the stacker *instance*.

| Setting | Qt control | Collected field | Consumed where | Test |
|---|---|---|---|---|
| Use GPU (drizzle) | `use_gpu_check` (Stacking tab) | `QtSettingsState.use_gpu` | `run_handoff.attach_run_settings` → `RunRequest.backend_kwargs["use_gpu"]` → `SeestarQueuedStackerBackend._apply_seam_kwargs` → `stacker.use_gpu` | `test_qt_run_settings_handoff_m20.py` |
| HQ RAM limit (GB) | `max_hq_mem_spin` (Stacking tab) | `QtSettingsState.max_hq_mem_gb` | `run_handoff.attach_run_settings` → `RunRequest.backend_kwargs["max_hq_mem_gb"]` → `SeestarQueuedStackerBackend._apply_seam_kwargs` → `stacker.max_hq_mem` (bytes) | `test_qt_run_settings_handoff_m20.py` |

Mechanics:

* `MainWindow.build_run_request` collects `QtSettingsState` →
  `run_config.build_run_request` (canonical) → `attach_run_settings(request,
  use_gpu=..., max_hq_mem_gb=...)`, producing a still-immutable `RunRequest`
  whose `backend_kwargs` carry the two seam fields.
* `seestar.gui.run_config.SEAM_ONLY_KWARGS` gains `use_gpu` / `max_hq_mem_gb`
  (additive, default-preserving): `split_backend_kwargs` filters them out of the
  `start_processing` kwargs, so the engine never receives them as keywords.
* `SeestarQueuedStackerBackend._apply_seam_kwargs` applies `use_gpu` →
  `stacker.use_gpu` and `max_hq_mem_gb` → `stacker.max_hq_mem =
  int(gb * 1024**3)`; the GB→bytes conversion lives in the adapter, not the
  snapshot.
* Fallback: a bare surface (no persisted settings / untouched controls) attaches
  the Qt/Tk defaults (`use_gpu=False`, `max_hq_mem_gb=8.0`), so behaviour equals
  today's defaults.

Explicitly out of scope this lot (deferred gaps, unchanged `[ ]`):
last-stack resume (10.1) and the official Qt entry point (11.2).  (The engine
solver bridge (3.6) was closed by M21, the Res-factor E2E (16.15) by M22, and
last-stack resume (10.1) by M23, all after this lot.)  The boring single-batch
subprocess `--max-mem` now forwards the configured `max_hq_mem_gb` (see §20.4,
M25; it was a pre-existing M4 delta).

## 20. Last-stack display / resume parity (M23)

Closes checklist 10.1 across three sub-items, all without touching the Tk GUI or
the engine.

**20.1 — Last-stack → output pre-fill parity.**  The Qt shell now mirrors Tk
`_on_last_stack_changed` for *every* last-stack change path — Browse, manual
edit, and persisted-load at launch.  `MainWindow._on_last_stack_changed`
(connected to `last_stack_edit.textChanged` and invoked at the end of
`_apply_state_to_controls`) pre-fills the output folder with the last-stack
file's parent directory **only when the output folder is empty** (the exact Tk
`if not output_path.get()` guard); a non-empty output is never clobbered.  The
browse path no longer pre-fills separately (it relies on the same connected
handler).  M8 settings persistence round-trips green: the pre-filled output is
persisted on the next save, matching Tk.

**20.2 — Run-end Processing Summary dialog.**  The Qt shell now shows a
"Processing Summary" dialog at the end of a run (regular and boring), matching
the Tk summary (Status, Total Processing Time, Files Attempted, Final Stack
File, Images in Final Stack, Total Exposure (Final Stack), plus an "Open
Output" button when applicable).  The summary **data** is computed outside the
widget layer: a new pure dataclass
`seestar.gui_qt.summary_payload.SummaryPayload` + lazy
`read_final_fits_header` / `build_summary_payload` (astropy imported lazily via
`importlib`, never at module level).  The regular-run backend adapter
`SeestarQueuedStackerBackend.run` and the boring runner
`QProcessBoringRunner._on_finished` each build the payload and emit it through a
new `summary(object)` signal (`RunWorker` → `RunController.summary_updated` →
`MainWindow`, and `BoringRunnerBase.summary` → `MainWindow`).  The Qt dialog
only *formats* the payload — no astropy/engine import at `gui_qt` module level
and no new engine import.

**20.3 — Engine auto-resume verification.**  The Qt run path already forwards
`output_folder` as `start_processing(output_dir=...)` (the canonical
`build_backend_kwargs` emits `output_dir`, `split_backend_kwargs` leaves it in
`start_kwargs`, and `SeestarQueuedStackerBackend.run` passes it verbatim), so
`SeestarQueuedStacker.start_processing` sets `self.output_folder` and
`_can_resume(Path(self.output_folder))` sees the exact memmap_accumulators /
`batches_count.txt` artifact set.  No new resume UI is built; the seam is now
covered by tests (`test_qt_last_stack_resume_m23.py`) and documented here.

**20.4 — Boring `--max-mem` parity (M25).**  The boring single-batch route now
forwards the user-configured HQ-RAM limit instead of hardcoding `8.0`.
`MainWindow._start_boring_route` passes `max_mem_gb=float(state.max_hq_mem_gb)`
into `boring_route.build_boring_request`, and `build_boring_request` keeps its
`max_mem_gb: float = 8.0` default so callers that pass nothing still get `8.0`.

**Parity statement (Tk reference vs Qt boring now).**  The Tk boring branch
emits `--max-mem str(getattr(self.settings, "max_hq_mem_gb", 8))` with
`max_hq_mem_gb` a float read from the `max_hq_mem_var` `tk.DoubleVar` (default
`8.0`) — i.e. Tk *always* forwards the configured value and never omits
`--max-mem` (the engine default rule only applies when the CLI arg is absent,
which never happens in the Tk flow).  The Qt boring route now emits the
identical argv (`--max-mem str(float(state.max_hq_mem_gb))`, default `8.0`), so
the two surfaces are value-for-value identical for both the default and any
configured value (the engine parses `--max-mem` as `float`, so `"8.0"` / `"4.0"`
… encode the same limit).  No Tk/engine change; the regular (non-boring) run
path is unchanged (it already read the same `max_hq_mem_gb` source via M20).

Explicitly out of scope this lot (unchanged `[ ]`): the official Qt entry point
(11.2).  (The M22 live-preview-frame display reconciliation was closed by M24,
and the boring single-batch `--max-mem` 8.0 GB delta by M25, both after this
lot.)

---

## Last updated

- **2026-08-21 — lot ZSSS-QT-FP-M1**: delivered items 2.1, 2.2 and section 13
  (Tk-like splitter topology + persistent right panel + final-combination
  business selector).  Detailed solver dialog, browse actions, preview
  WB/stretch/histogram and zoom/rotation interactivity remain `[ ]`.
- **2026-08-21 — lot ZSSS-QT-FP-P0**: delivered items 1.1–1.5, 3.1–3.4, 12.3
  (batch-size contract, solver gate + readiness probe injection, shutdown
  robustness). Everything else remains `[ ]`.
- **2026-08-21 — lot ZSSS-QT-FP-M2**: delivered item 3.5 (first real Qt solver
  configuration dialog: None/ASTAP/ZeSolver choice, ASTAP path/data/numeric
  fields with Browse, ZeSolver status label + configuration button driven by the
  public `zesolver_ui_state(check_zesolver_readiness())` contract, in-dialog
  ASTAP-path validation, OK→`MainWindow` state round-trip, lazy engine-free
  import). Solver *persistence* via `solver_config` (item 3.6) remains `[ ]`.
- **2026-08-21 — lot ZSSS-QT-FP-M3**: delivered items 4.1–4.4, 5.1–5.3
  (browse input/output/temp/reference/last-stack via `QFileDialog`, reference +
  last-stack controls surfaced on the Stacking tab, View Inputs non-backend
  dialog, Open Output via `QDesktopServices.openUrl`, Add Folder staging +
  validation with `initial_additional_folders` passed through `_on_start`).
  Last-stack persistence (4.5), full resume semantics (10.1), Analyze external
  launch (5.4) and full settings persistence (9.3) remain `[ ]`.
- **2026-08-21 — lot ZSSS-QT-FP-M4**: delivered items 1.6 and 2.3 (boring /
  single-batch CSV route + gating).  `batch_size == 1` now routes to the boring
  stack subprocess (`seestar/gui/boring_stack.py`, launched by filesystem path —
  never imported) instead of `RunController.start`, with a conservative CSV
  parser (`boring_route.parse_stack_plan_csv` replicating the Tk `read_rows`
  rules) and a QProcess-based runner (`boring_runner.QProcessBoringRunner`).
  Missing/empty/invalid `stack_plan.csv` or a listed missing FITS blocks start
  with a clear preflight error and leaves the UI idle.  A visible
  "Threaded Boring Stack" checkbox synchronises with the batch-size spinbox
  (check ⇄ `batch_size == 1`, spinbox locked while checked) and gates the
  incompatible drizzle controls.  Remaining deltas: `--max-mem` is fixed at the
  8.0 GB default (no `max_hq_mem_gb` field in `QtSettingsState` yet); the real
  QProcess lifecycle is implemented but **not** exercised by automated tests
  (tests inject a fake runner and never spawn the subprocess); the CSV weight
  column is parsed but the command builder does not consume it (the subprocess
  re-reads the CSV itself).
- **2026-08-21 — lot ZSSS-QT-FP-M5**: delivered items 6.5, 6.6 and 13.6
  (first real preview view interactivity).  The right-panel View controls now
  drive a display-only zoom (`Fit`/`100%`/`200%`/`50%`), a resolution label
  (original → displayed size + zoom + rotation), and left/right rotation
  (±90° cumulative modulo 360) applied only to the displayed preview pixmap.
  The rendered `QImage` is stored as the copied transform source and never
  mutated; invalid/missing data clears the source, pixmap, rotation and
  disables the view controls.  New Qt-only helper `seestar/gui_qt/preview_view.py`
  keeps the transform math out of `main_window.py`.  White-balance / stretch /
  histogram (6.2/6.3/6.4) and pan (6.7) remain `[ ]`.
- **2026-08-21 — lot ZSSS-QT-FP-M6**: delivered items 7.3 and 7.4 (Copy Log +
  elapsed/remaining time surface).  A `Copy Log` button next to the log area
  copies the full plain-text log to the system clipboard via
  `QApplication.clipboard()` (disabled while empty, armed on first log line,
  never mutating log or run state).  Visible `Elapsed` / `Remaining` labels are
  driven only by the existing progress lifecycle signals and an injectable
  monotonic clock: elapsed starts at 0 on run start, remaining is `—` until a
  usable percent arrives, updates from a naive
  `elapsed * (100 − percent) / percent` estimate for 1..99, becomes `0:00` on
  finish/100, and `failed`/`cancelled` on terminal non-success.  Boring
  (single-batch CSV) runs feed the same log/copy surface and show honest time
  labels (elapsed visible, remaining unknown throughout, `0:00`/`failed`/
  `cancelled` at the end).  New Qt-only helper
  `seestar/gui_qt/progress_time.py` keeps the math out of `main_window.py` and
  is deterministic (no real sleeping in tests).
- **2026-08-21 — lot ZSSS-QT-FP-M7**: delivered item 5.4 (Analyze /
  ZeAnalyser launch seam).  New stdlib-only seam
  `seestar/gui_qt/analyzer_launch.py` mirrors the public-process-contract
  behaviour of the Tk helper without importing Tk or the engine; every
  side-effecting dependency (executable detection, popen, temp-dir, pid) is
  injectable so tests never spawn a real ZeAnalyser.  `MainWindow._on_analyse`
  validates the input folder, detects ZeAnalyser (`zeanalyser` entry point,
  `python -m zeanalyser` fallback), creates the command-file path, and launches
  a non-blocking subprocess with `ZEANALYSER_COMMAND_FILE` set — never marking
  a run active.  The Analyse button is enabled only when the input folder is a
  non-empty existing directory (updated on path changes).  A single-shot,
  Qt-safe command-file *consumption* seam (`_check_analyzer_command_file`)
  parses `REFERENCE=<path>`, updates the reference field only for a non-empty
  reference, and deletes the file best-effort.  Remaining delta: the periodic
  re-arming reference-return *watcher* is intentionally deferred (no QTimer
  polling yet); the launch seam and one-shot consumption are covered by
  `tests/test_qt_analyzer_launch.py`.
- **2026-08-21 — lot ZSSS-QT-FP-M8**: delivered items 4.5, 9.2 and 9.3
  (bounded Qt settings/geometry persistence).  New pure-stdlib helper
  `seestar/gui_qt/settings_persistence.py` loads/saves a UTF-8 JSON dict
  (missing/corrupt file → defaults, never raises, deterministic `sort_keys`);
  `QtSettingsState.from_dict` + `_coerce_value` coerce known fields, ignore
  unknown keys and fill missing keys from defaults.  `MainWindow` accepts an
  injectable `settings_path` (default `None` = persistence disabled for bare
  tests; the `run_qt_app` entry point defaults to CWD `seestar_settings.json`,
  matching Tk), applies persisted settings to all visible controls on
  construction (paths, filename, batch/stacking/final-combine/drizzle/solver +
  the full Settings/Mosaic surface), refreshes path-action enablement, and
  saves `collect_settings_state()` + base64 `saveGeometry()` on completed
  shutdown.  `stack_final_combine` stays the single source of truth (the
  derived `reproject_*` flags are re-derived from the combo on apply), no
  seam-only key is passed to the backend, and no Tk/engine/scientific code was
  touched.  Solver config persistence (item 3.6) remains `[ ]`.
- **2026-08-21 — lot ZSSS-QT-FP-M9**: delivered items 8.1, 8.2 and 13.7
  (Qt FR/EN language switch + localization surface).  The previously disabled
  `language_combo` is now enabled and user-triggered: switching re-labels the
  visible Qt shell strings in place (tabs, Stacking-tab path labels and
  Browse/checkbox labels, Start/Stop/Analyse/Solver/View Inputs/Add Folder/Open
  Output/Copy Log, progress/log + elapsed/remaining labels, Preview/View/
  Histogram/Actions group titles, the 12 Expert section titles and the Mosaic
  sub-labels plus a representative subset of Settings field labels) without
  rebuilding the window, and `_current_language_code()` feeds the ZeAnalyser
  launch `--lang` argument.  A new pure-stdlib helper
  `seestar/gui_qt/localization.py` holds the compact FR/EN mapping with safe
  key/fallback logic; `QtSettingsState` gains a `language: str = "en"` field
  normalised on load (unknown/corrupt → English) and persisted through the
  existing M8 settings JSON round-trip.  Remaining deltas: the remaining
  Settings field labels stay English (no full mapping yet), the backend-mode
  notice and transient log/status/dialog strings stay English, and the
  full Tk `Localization` dictionaries are intentionally not imported (kept
  Qt/Tk/engine-free).  Covered by `tests/test_qt_localization.py`.
- **2026-08-21 — lot ZSSS-QT-FP-M10**: delivered items 6.2, 6.3 and 6.4
  (display-only preview WB / stretch / histogram).  The left "Preview controls"
  tab is no longer a placeholder: it now holds a White-balance group (per-channel
  R/G/B `QDoubleSpinBox` gains, neutral 1.0, plus a `Reset` button), a Stretch
  group (`linear` / `asinh` / `log` / `auto`, linear default) and a Histogram
  group (a `QPixmap` bar-chart surface + a localized empty/stats status label).
  All three act strictly on a *derived* display image produced by the new
  Qt-local helper `seestar/gui_qt/preview_adjust.py` (lazy numpy, no Tk/engine);
  the stored `_preview_source` `QImage` is never mutated, the raw preview payload
  contract and the scientific backend are untouched, and zoom/rotation still
  apply cleanly on top of the WB/stretch result (and vice versa).  Neutral
  settings (WB `(1.0, 1.0, 1.0)` + `linear`) return a byte-identical copy, so the
  default preview behaviour matches M5/M8 exactly.  The right-panel "Histogram"
  surface is preserved, not removed: the persistent right panel keeps a live
  histogram group (`right_histogram_group` / `right_histogram_status` /
  `right_histogram_view`) fed by the same pixmap/stats as the Preview controls
  tab histogram, so both surfaces update and clear together (checklist item
  13.4 stays true).  Preview-control values are intentionally **not** persisted
  (kept out of `QtSettingsState` to avoid touching M8) and reset to neutral on
  each window construction.  New FR/EN keys (`wb_*`, `stretch_*`,
  `histogram_empty`, `histogram_stats`) were added with full parity.  Pan (6.7)
  remains `[ ]`.
  Covered by `tests/test_qt_preview.py` and `tests/test_qt_localization.py`.
- **2026-08-21 — lot ZSSS-QT-FP-M11**: delivered item 12.5 (real backend
  progress/log bridge).  A real end-to-end witness (`--backend seestar` on the
  mini M16 dataset) showed the engine's `SeestarQueuedStacker.update_progress`
  never calls the installed callback directly — it pushes closures onto
  `stacker.gui_event_queue` (a thread-safe `Queue`) and relies on the GUI layer
  to drain it.  The Tk GUI drains it from a periodic `root.after` loop
  (`_poll_gui_events`); the Qt bridge had no such loop, so the progress bar and
  log view were starved (0 numeric progress / 8 stray aligner log lines) while
  the backend still produced the final FITS + preview.  Fix: a minimal Qt-only
  change in `SeestarQueuedStackerBackend.run` that drains `gui_event_queue` on
  the worker thread both while polling and once the processing thread finishes
  (final flush), tolerating malformed items.  Verified with a programmatic
  offscreen `MainWindow(backend_mode="seestar")` witness: progress
  `[5, 25, 50, 75, 100]` (monotonic), 35 log lines (reference selection →
  final FITS/PNG save), 1 final preview payload rendered to the Qt preview pane
  (right-panel histogram updated), `finished` terminal state, `_thread`/
  `_worker` reaped, Start/Stop/Analyse/View/Add/Open re-enabled, `shutdown()`
  → `True` with no live thread.  No scientific code touched.  Covered by
  `tests/test_qt_backend_runner.py` (queue-drain + malformed-item resilience);
  see the witness report under `~/.openclaw/workspace/review/`
  `zsss-pyside6-m11-real-backend-e2e/`.
- **2026-08-21 — lot ZSSS-QT-FP-M13**: completed the Tk preview-controls
  parity (display-only).  White balance (6.2) now includes an `Auto WB` action
  (mode-based R/G/B gains computed from the display image, mirroring
  `apply_auto_white_balance`) plus slider + numeric-spinbox pairs for R/G/B
  (Tk range 0.1–5.0, step 0.01); stretch (6.3) gains black/white/gamma
  slider + spinbox controls (Tk defaults 0.01/0.99/1.0, ranges/steps 0–1/
  0.001 and 0.1–5/0.01) plus a `Reset Stretch`, and the default stretch is
  now `Asinh` (Tk parity); a new image-adjustments surface (6.9) adds
  brightness/contrast/saturation sliders + reset (Tk defaults 1.0, ranges/steps
  0.1–3.0/0.01, 0.1–3.0/0.01, 0.0–3.0/0.01).  All adjustment math
  (WB, stretch, gamma, B/C/S and auto-WB) is reimplemented in pure numpy inside
  `preview_adjust` (lazy numpy, no Tk/engine/PIL import) and reproduces the Tk
  `PreviewManager.process_image` pipeline exactly; the stored `_preview_source`
  `QImage` is never mutated.  M12 hardening: `_on_initial_preview_result` now
  drops a stale `InitialPreviewResult` whose absolute `folder` no longer matches
  the selected input folder, so a fast folder switch cannot let the old folder's
  image overwrite the new folder's preview.  Known gap left for later: the Tk
  preview resolution-cycle button (`Res 1/1..1/4`) drives the backend's
  `preview_downsample_factor` (not display-only) and is intentionally out of
  scope; pan (6.7) remains `[ ]`.  Covered by
  `tests/test_qt_preview_controls.py` (+ updated `tests/test_qt_preview.py`
  defaults) and the existing import-hygiene tests.
- **2026-08-21 — lot ZSSS-QT-FP-M12**: delivered item 6.8 (initial-preview
  parity: auto-load first FITS).  The Qt shell now reproduces the Tk GUI's
  initial-preview behaviour: when the input folder is set (restored from
  settings at startup, or chosen via Browse Input) it non-blockingly loads the
  first sorted `.fit`/`.fits` file and displays it in the right preview panel
  with the histogram and view/preview controls enabled.  The folder is
  validated and scanned synchronously (cheap); only the FITS load + debayer
  runs on a daemon `threading.Thread`, which lazily imports the engine
  (`load_and_validate_fits` / `debayer_image` via `importlib`, mirroring
  `backend_runner.py`'s split-string discipline) and delivers the in-memory
  image back through a *queued* Qt signal — no widget is ever touched off the
  GUI thread, and a slow/absent folder never freezes the shell.  2D Bayer data
  is debayered only when a valid pattern exists (header `BAYERPAT` else
  `settings.bayer_pattern`, default `GRBG`), matching Tk.  Missing/empty folder
  states clear the preview with localized messages (`preview_no_input_folder`,
  `preview_no_fits`, `preview_loading`, `preview_loaded`, `preview_error`), and
  a redundant-reload guard skips an unchanged folder so repeated settings
  restores never reload the same image.  Qt-only: no Tk/engine/settings changes
  and no FITS/PNG writes.  Covered by `tests/test_qt_initial_preview.py` (real
  FITS load, missing/empty folder, 2D Bayer debayer → 3 channels, GUI-thread
  responsiveness smoke, redundant-reload guard).
- **2026-08-21 — lot ZSSS-QT-FP-M14**: histogram surface consolidation +
  Auto Stretch action + histogram-source alignment (display-only).  The
  duplicated Preview-controls-tab histogram was **removed** (not merely
  hidden); the persistent right-panel histogram is now the single live
  surface (checklist item 13.4 stays true).  A new Qt-only
  `seestar/gui_qt/histogram_view.py` `HistogramView` (pure `QPainter`, lazy
  numpy) reproduces the Tk `HistogramWidget` interactions — auto-zoom
  (`Auto zoom histogram` checkbox), `Reset Histogram`, `Zoom Histogram` and
  `Reset zoom` (`R`) plus BP/WP line dragging that emits `rangeChanged` (the
  Qt equivalent of Tk `update_stretch_from_histogram`
  `V_HistoCallbackRefreshLight_1`), which mirrors the dragged values back into
  the black/white slider+spin controls.  The stretch group gains an
  `Auto Stretch` button (Tk `apply_auto_stretch` parity): it computes the
  black/white points from the *WB-only* derived image (percentile 1%/99% of
  the luminance, normalised into the 0-1 slider scale via the WB image's
  min/max) and switches the stretch method to `Asinh`, updating the bp/wp
  slider+spin values and refreshing the display.  Histogram source aligned to
  Tk: the right-panel histogram is now computed from the WB-only derived
  image (`apply_preview_wb`, Tk `image_data_wb`) instead of the fully-stretched
  display image, so it reacts to white balance but not to stretch / gamma /
  brightness-contrast-saturation.  No engine/Tk/settings changes; no FITS/PNG
  writes; `_preview_source` is never mutated (all helpers operate on copies).
  Covered by `tests/test_qt_histogram_m14.py` (+ re-pointed
  `tests/test_qt_preview.py`, `tests/test_qt_localization.py` and
  `tests/test_qt_shell.py`), and the existing import-hygiene tests.
- **2026-08-21 — lot ZSSS-QT-FP-M15**: Expert tab content parity + closure +
  M14 leftovers.  Delivered the full Tk `tab_expert` → Qt per-control parity
  (new section 14): every Expert control now exists with the Tk label (fully
  FR/EN localised via `localization`), the Tk range/step and the canonical
  `SettingsManager` default; the BN / CB / final-crop / master-tile-crop /
  Photutils / feathering / low-weight enabler checkboxes gate their sub-option
  widgets exactly like the Tk `_update_*_options_state` methods; a new
  "Reset Expert Settings" button restores the whole Expert surface to model
  defaults (GUI-only).  `bn_grid_size_str` is now a combo (was a free-text
  field).  `QtSettingsState` gains the three gating flags `apply_bn` /
  `apply_cb` / `apply_final_crop` (persisted like Tk, but not consumed by
  `build_backend_kwargs` → "display-only now, backend E2E later").  M14
  leftovers: the persistent histogram now freezes a manual X zoom across
  `set_data` refreshes (Tk `freeze_x_range` semantics) and resets it on
  reset-view / reset-zoom / auto-zoom; the dead `render_histogram_pixmap`
  helper (and its unused colour palette) was removed from `preview_adjust`.
  No Tk/engine/backend/settings-file changes; `_preview_source` never mutated.
  Covered by `tests/test_qt_expert_m15.py` (10 tests) + the existing import
  hygiene tests; `git diff --check` clean.
- **2026-08-21 — lot ZSSS-QT-FP-M16**: Stacking tab content parity + closure
  (new section 15).  Delivered the full Tk `tab_stacking` → Qt per-control
  parity: the previously missing Stacking-tab items are now present — the
  drizzle **Use GPU** checkbox (`use_gpu_check`), the **HQ RAM limit (GB)**
  spinbox (`max_hq_mem_spin`) and the grey **drizzle policy hint**
  (`drizzle_policy_hint`), all FR/EN localised via `localization`.  The
  drizzle enabler now gates its sub-options exactly like Tk
  `_update_drizzle_options_state` (mode combo + group-size spin + Use-GPU +
  the Expert-tab "Drizzle Advanced" scale/WHT/kernel/pixfrac), with the M3-D
  **group-size gate** (enabled only for drizzle + `Incremental` Large-dataset
  mode); the boring-mode gate (`_update_boring_gating`) now delegates to the
  drizzle gate.  `apply_final_scnr` was added to `EXPERT_ENABLER_GATES` so the
  SCNR amount / target-channel / preserve-luminosity sub-options gate like Tk
  `_update_final_scnr_options_state`.  `QtSettingsState` gains `use_gpu` and
  `max_hq_mem_gb` (persisted/collected like Tk) but they are **not** consumed
  by `build_backend_kwargs` → "display-only now, backend E2E later" (the boring
  CLI `--max-mem` stays fixed at 8.0 GB; no `use_gpu` backend kwarg).  No
  Tk/engine/backend/settings-file changes; `_preview_source` never mutated.
  One legitimate test re-point: `tests/test_qt_boring_route.py`
  `test_boring_mode_gates_drizzle_controls` now reflects the Tk-parity drizzle
  gating (sub-options re-enable only after re-checking drizzle, not merely on
  un-checking boring).  Covered by `tests/test_qt_stacking_m16.py` (11 tests)
  + the existing import-hygiene tests; `git diff --check` clean.
- **2026-08-21 — lot ZSSS-QT-FP-M17**: right-panel preview ergonomics parity
  closure (new section 16).  Delivered the Res 1/1..1/4 preview-resolution
  cycle button (`preview_res_button`) on the right panel: it cycles factors
  1→2→3→4→1 with the Tk `Res 1/N` label (localized `Res`/`Rés`), persists the
  factor in GUI state (`_preview_res_factor`), and re-renders the local preview
  at the new factor via a display-only `downsampled_image` /
  `render_view(..., downsample_factor=...)` seam — never mutating
  `_preview_source` and never importing/using the engine.  The Tk button's
  engine coupling (`set_preview_downsample_factor` + `refresh_preview`) is a
  documented backend-E2E gap.  Closed the M16 kappa/winsor gap:
  `_toggle_kappa_visibility` mirrors the Tk show/hide rule exactly (cosmetic
  only — `build_backend_kwargs` still always carries `stack_kappa_low` /
  `stack_kappa_high` / `winsor_limits`).  Default deviation (documented): the
  Qt Res factor defaults to `1` (native) vs the Tk engine default `2`.  Pan
  (6.7) and solver-config persistence (3.6) remain `[ ]` with feasibility
  notes.  Covered by `tests/test_qt_preview_ergonomics_m17.py` (12 tests) +
  the existing import-hygiene tests; `git diff --check` clean.
- **2026-08-21 — lot ZSSS-QT-FP-M18**: preview pan/zoom parity (checklist 6.7,
  new section 17) — display-only.  Reproduced the Tk `PreviewManager`
  mouse-wheel zoom + left-drag pan as a pure Qt view transform on top of the
  existing `preview_view` render path.  The preview surface is now a
  `PreviewImageView` (`seestar/gui_qt/preview_image_view.py`, a `QLabel`
  emitting `wheelZoom`/`panDelta`); `MainWindow` keeps a single continuous
  `_preview_zoom_factor` (default 1.0, `[MIN_ZOOM 0.05, MAX_ZOOM 15.0]`) and a
  `_view_offset_x/y` pan offset (unbounded, Tk clamps nothing), and
  `render_view` gained an optional `zoom_factor` + `pan_offset` (plus
  `scaled_image` / `compose_panned_pixmap` / `clamp_zoom_factor` /
  `preset_label_for_factor` / `zoomed_image_size` / `fit_scale` helpers).  Wheel
  zoom mirrors the Tk 1.15 step with cursor anchoring; the `zoom_combo` presets
  set the same factor and recentre, while a non-preset wheel zoom shows a blank
  (custom) combo — the documented interaction rule (wheel zoom sets a level
  outside the presets; the combo returns to a preset).  Pan/zoom reset on every
  new preview image and on "Zoom Fit"; rotation resets pan but keeps zoom (Tk
  parity).  Strictly display-only: `_preview_source` is never mutated, no
  FITS/PNG writes, no engine/Tk/backend/settings-file changes, and the new
  modules are engine/Tk/zesolver-free (asserted by the M18 tests + the existing
  import-hygiene tests).  Engine-coupled gaps: none.  Gaps left for later:
  solver-config persistence (3.6) and backend-E2E items.  Covered by
  `tests/test_qt_preview_pan_zoom_m18.py` (28 tests) + the existing
  import-hygiene tests; `git diff --check` clean.
- **2026-08-21 — lot ZSSS-QT-FP-M19**: solver settings persistence via the Qt
  JSON settings surface (checklist 3.6 → §18) — settings-only.  The six
  solver-dialog fields (`local_solver_preference`, `astap_path`,
  `astap_data_dir`, `astap_search_radius`, `astap_downsample`,
  `astap_sensitivity`) were already present in `QtSettingsState` and the
  `Expert`-tab "Solver" section, and already round-trip through the generic M8
  `settings_persistence` surface; M19 hardens and documents that contract and
  closes the remaining checklist gap.  `MainWindow._on_solver` prefills
  `SolverSettingsDialog` from `collect_settings_state()` and writes accepted
  values back into the live controls (`_apply_solver_dialog_values`), so
  accepted values survive a `_save_persisted_settings` →
  `_load_persisted_settings` round-trip.  Documented default divergence:
  `astap_downsample` defaults to `1` (Tk `SettingsManager` + Qt dialog) vs the
  engine `astap_default_downsample` `2`; the Qt shell keeps `1` and never
  touches the engine config.  **Engine bridge (checklist 3.6) stays `[ ]`**:
  no read/write of `seestar.core.solver_config`, and no
  `check_zesolver_readiness` / `probe_zesolver_operational` /
  `open_zesolver_configuration` is invoked for persistence; `solver_config.py`
  is untouched (asserted).  No Tk/engine/backend files changed;
  `_preview_source` never mutated.  Covered by
  `tests/test_qt_solver_persistence_m19.py` (new) + the existing
  import-hygiene tests; `git diff --check` clean.
- **2026-08-21 — lot ZSSS-QT-FP-M20**: backend E2E part 1 — Qt run-settings
  handoff for `use_gpu` (15.30) and `max_hq_mem_gb` (15.15).  The canonical
  shared builder `seestar.gui.run_config.build_backend_kwargs` stays unchanged
  (Tk flow byte-identical); the Qt shell attaches the two collected values to
  its `RunRequest` as seam-only fields via a new engine/Tk-free
  `seestar/gui_qt/run_handoff.py` (`attach_run_settings`), and
  `SeestarQueuedStackerBackend._apply_seam_kwargs` applies `use_gpu` →
  `stacker.use_gpu` and `max_hq_mem_gb` → `stacker.max_hq_mem` (bytes) before
  `start_processing` — never as `start_processing` kwargs.
  `seestar.gui.run_config.SEAM_ONLY_KWARGS` gains the two names (additive,
  default-preserving) so `split_backend_kwargs` filters them out.  Fallback
  degrades to the Qt/Tk defaults.  The boring single-batch subprocess `--max-mem`
  stays fixed at 8.0 GB (pre-existing M4 delta); solver bridge (3.6), Res-factor
  E2E (16.15), last-stack resume (10.1) and the Qt entry point (11.2) remain
  `[ ]`.  Re-pointed one M16 assertion (`test_engine_coupled_items_are_wired_into_run_request`
  replaces the old "display-only" assertion).  Covered by
  `tests/test_qt_run_settings_handoff_m20.py` (new, 14 tests) + the existing
  import-hygiene tests; `git diff --check` clean.
- **2026-08-21 — lot ZSSS-QT-FP-M21**: backend E2E part 2 — engine solver
  bridge (checklist 3.6 → `[x]`).  On `SolverSettingsDialog` accept,
  `MainWindow._on_solver` now also calls the new engine/Tk-free
  `seestar/gui_qt/solver_config_bridge.py`, which lazily imports
  `seestar.core.solver_config` (module path assembled from split string
  literals, accept-time only) and persists the five mapped ASTAP fields
  (`astap_path`→`astap_executable_path`, `astap_data_dir`→
  `astap_data_directory_path`, `astap_search_radius`→
  `astap_default_search_radius`, `astap_downsample`→`astap_default_downsample`,
  `astap_sensitivity`→`astap_default_sensitivity`) via the engine's own
  `load_config` → overlay → `save_config`, so merge/legacy-migration semantics
  are byte-identical to the Tk `_on_ok` write.  The solver *choice*
  (`local_solver_preference`) is deliberately not mapped — the engine config
  has no solver-choice key (Tk keeps it in `SettingsManager` /
  `seestar_settings.json`).  Cancel/ESC writes nothing; the Qt JSON surface
  (M19) is unchanged and remains the display/state source while the engine
  config becomes the runtime-consumed source.  No Tk/engine/`solver_config.py`
  changes; `_preview_source` never mutated.  Re-pointed two accept-path tests
  (M19 `test_dialog_accept_values_survive_save_load_round_trip` and
  `test_on_solver_opens_dialog_and_applies`) to isolate the engine config path.
  Covered by `tests/test_qt_solver_bridge_m21.py` (new, 11 tests) + the
  existing import-hygiene tests; `git diff --check` clean.  Remaining gaps:
  Res-factor E2E (16.15), last-stack resume (10.1), Qt entry point (11.2), and
  the boring single-batch `--max-mem` 8.0 GB delta.
- **2026-08-21 — lot ZSSS-QT-FP-M22**: backend E2E part 3 — live Res-factor
  control (checklist 16.15 backend half → `[x]`).  The Qt `preview_res_button`
  now drives the live engine `preview_downsample_factor` during an active run
  via a thread-safe control channel: `MainWindow._on_preview_res_cycle` →
  `RunController.set_preview_downsample_factor` →
  `RunWorker.set_preview_downsample_factor` →
  `SeestarQueuedStackerBackend.set_preview_downsample_factor` (enqueues the
  factor on a thread-safe `queue.Queue`) → drained by the backend's run loop on
  the worker thread → `stacker.set_preview_downsample_factor(factor)` +
  `stacker.refresh_preview()` (Tk `_cycle_preview_resolution` parity, both
  best-effort).  Idle clicks stay display-only (label + local display downsample
  unchanged); the factor mapping is identity (`Res 1/1` → 1, `Res 1/2` → 2,
  `Res 1/4` → 4).  No Tk/engine changes; the control channel is additive and
  thread-safe (all stacker mutations happen on the worker thread, never the GUI
  thread).  **Display-path gap:** the engine factor changes the resolution of
  the preview *data* the engine pushes (via its preview callback), while the
  shell still applies its own local display downsample — the on-screen preview
  does not yet reconcile the two; the engine-side application is verifiable on
  the stacker instance, and the live-preview-frame display reconciliation
  remains `[ ]` (no new preview-frame pipeline built).  Covered by
  `tests/test_qt_res_live_m22.py` (new, 11 tests) + the existing
  import-hygiene tests; `git diff --check` clean.  Remaining gaps: last-stack
  resume (10.1), Qt entry point (11.2), and the boring single-batch `--max-mem`
  8.0 GB delta.
- **2026-08-21 — lot ZSSS-QT-FP-M23**: last-stack display / resume parity
  (checklist 10.1 → `[x]`, new section 20).  (a) last-stack → output pre-fill
  parity: `MainWindow._on_last_stack_changed` (connected to
  `last_stack_edit.textChanged` and invoked at the end of
  `_apply_state_to_controls`) pre-fills the output folder from the last-stack
  parent dir for *all* change paths (browse, manual edit, persisted-load) with
  the exact Tk "only when output empty" guard; the browse path now delegates to
  the same handler and M8 persistence round-trips green.  (b) run-end
  "Processing Summary" dialog: a new engine/Tk-free
  `seestar/gui_qt/summary_payload.py` (`SummaryPayload` + lazy
  `read_final_fits_header`/`build_summary_payload`, astropy imported lazily via
  `importlib`, never at module level); the backend adapter
  (`SeestarQueuedStackerBackend.run`) and the boring runner
  (`QProcessBoringRunner._on_finished`) each build the payload and emit it
  through a new `summary(object)` signal
  (`RunWorker`→`RunController.summary_updated`→`MainWindow`, and
  `BoringRunnerBase.summary`→`MainWindow`); the Qt dialog only formats the
  payload (Status / Total Processing Time / Files Attempted / final.fits
  NIMAGES/TOTEXP / Open Output) and shows after both regular and boring runs.
  (c) engine auto-resume verified: the Qt run path forwards `output_folder` as
  `start_processing(output_dir=...)`, so `_can_resume` sees the memmap/
  batches_count artifact set — no new resume UI (documented + tested).  No
  Tk/engine changes; `_preview_source` never mutated; no FITS/PNG writes.
  Covered by `tests/test_qt_last_stack_resume_m23.py` (new, 15 tests) + the
  existing import-hygiene tests; `git diff --check` clean.  Remaining gaps:
  Qt entry point (11.2), the boring single-batch `--max-mem` 8.0 GB delta, and
  the M22 live-preview-frame display reconciliation.
- **2026-08-21 — lot ZSSS-QT-FP-M24**: preview display reconciliation (closes
  the M22 caveat; checklist 16.15 display half → `[x]`).  The on-screen preview
  now reflects the engine factor 1:1 during an active run: a new
  `MainWindow._effective_preview_downsample_factor()` makes the engine factor
  the single source of truth at render time — while `_running` it returns `1`
  (the engine already pushed frames at its own `preview_downsample_factor`
  resolution, so no local display downsample is applied on top; Tk parity: Tk
  renders the engine-pushed frames directly), and when idle it returns the
  existing local display-only `_preview_res_factor` (M17/M18 behaviour
  byte-identical).  The render path (`_refresh_preview_view`) and the Fit-exit
  `fit_scale` in `_on_wheel_zoom` both consume the effective factor, so
  resolution label, zoom, and pan all stay consistent during a run.  Run end
  (finished/failed/cancelled) restores idle behaviour automatically because the
  factor is derived from `_running` alone — no separate "engine factor" state is
  stored, so nothing can go stale.  The Res-button cycle/label semantics and
  the M22 control channel are unchanged; no new preview-frame pipeline, no new
  IPC/protocol, no Tk/engine changes.  Covered by
  `tests/test_qt_preview_reconcile_m24.py` (new, 8 tests) + the M17/M18 suites
  + the existing import-hygiene tests; `git diff --check` clean.  Remaining
  gaps: Qt entry point (11.2) and the boring single-batch `--max-mem` 8.0 GB
  delta.
- **2026-08-21 — lot ZSSS-QT-FP-M25**: boring single-batch `--max-mem` delta
  (pre-existing M4) closed.  `MainWindow._start_boring_route` now forwards the
  user-configured HQ-RAM limit by passing `max_mem_gb=float(state.max_hq_mem_gb)`
  into `boring_route.build_boring_request`; `build_boring_request` keeps its
  `max_mem_gb: float = 8.0` default so callers that pass nothing still get
  `8.0`.  **Parity statement:** the Tk boring branch emits
  `--max-mem str(getattr(self.settings, "max_hq_mem_gb", 8))` with
  `max_hq_mem_gb` a float (default `8.0`) — Tk always forwards the configured
  value and never omits `--max-mem`; the Qt boring route now emits the identical
  argv (`--max-mem str(float(state.max_hq_mem_gb))`, default `8.0`), so the two
  surfaces are value-for-value identical for the default and any configured
  value.  No Tk/engine change; the regular run path is unchanged (same
  `max_hq_mem_gb` source, M20).  Covered by
  `tests/test_qt_boring_mem_m25.py` (new, 6 tests) + the M4/M20 suites + the
  existing import-hygiene tests; `git diff --check` clean.  Remaining gap: Qt
  entry point (11.2).
- **2026-08-21 — lot ZSSS-QT-FP-M25.5-A**: ZeAnalyser reference-return watcher
  (closes the M7 deferral — "periodic analyzer reference-return watcher
  deferred" → `[x]`).  The Qt shell now reproduces the complete historical Tk
  workflow: `MainWindow._on_analyse` arms a GUI-thread `QTimer`
  (`_analyzer_watch_timer`, interval 1000 ms — the exact Tk
  `after(1000, ...)` cadence) right after a non-blocking launch, and each
  `_analyzer_watch_tick` polls the command file once, reusing the existing
  single-shot `_check_analyzer_command_file` consumption seam.  When
  `REFERENCE=<path>` arrives it (a) updates the reference field/state, (b)
  prepares a default output folder when empty (`<input>/stack_output_analyzer`,
  Tk parity), and (c) starts processing via `_on_start` (the Qt equivalent of
  Tk `start_processing()`); it then stops.  The watcher also stops when a run
  is already active (`_running`, Tk `self.processing` parity) or on
  shutdown/close (`shutdown()` stops the timer; the timer is parented to the
  window so Qt's object tree also destroys it — no zombie callback into a
  destroyed `MainWindow`).  Consumption is idempotent (the command file is
  deleted on read, so one reference is consumed once).  No new thread, no busy
  loop, no Tk/engine/`analyzer_launch.py` change.  Covered by
  `tests/test_qt_analyzer_watcher_m255a.py` (new, 6 tests) + the M7
  `test_qt_analyzer_launch.py` suite + shell/file-actions/localization/worker-
  lifecycle suites + the import-hygiene tests; `git diff --check` clean.
  Remaining gap: Qt entry point (11.2).
- **2026-08-21 — lot ZSSS-QT-FP-M25.5-B**: settings independent of the CWD
  (closes the 9.3 "CWD `seestar_settings.json`" convention for the official Qt
  shell).  `settings_persistence.default_settings_path()` now resolves a
  platform-aware per-user location — Windows `%APPDATA%/ZeSeestarStacker`,
  macOS `~/Library/Application Support/ZeSeestarStacker`, Linux
  `$XDG_CONFIG_HOME/ZeSeestarStacker` (default `~/.config`) — replicated with
  pure `os`/`platform` (no engine import), so `cd /tmp && python -m
  seestar.qt_main` finds the *same* settings as any other launch directory.
  New `resolve_settings_path()` implements the documented priority: (1) the
  platform file already exists → new wins (legacy untouched); (2) else a legacy
  CWD `seestar_settings.json` exists → migrated non-destructively into the
  platform location (directories created, legacy file **preserved**, whole JSON
  document copied verbatim so recognised *and* unknown keys survive; filtering
  stays a `QtSettingsState.from_dict` model concern); (3) else → the platform
  path with the directory created eagerly (defaults until first save).  Failure
  mode: an unwritable user-config location returns `None` (never raises), so
  `run_qt_app` disables persistence and the GUI opens with code defaults.
  `save_settings_json` now creates its parent directory and keeps returning
  `False` (never raising) on any write failure.  Tk and the engine are
  untouched; the Tk `SettingsManager` remains CWD-based, so this is a
  deliberate improvement over Tk parity, not a parity change.  Covered by
  `tests/test_qt_settings_cwd_m255b.py` (new, 14 tests) + the re-pointed
  `test_qt_settings_persistence.py` default-path test + the existing
  persistence/state/surface/validation/window/localization suites + the
  import-hygiene tests; `git diff --check` clean.  Remaining gap: Qt entry
  point (11.2).
