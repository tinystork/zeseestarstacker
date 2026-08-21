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
| 3.6 | ASTAP/ZeSolver config persistence via `solver_config` | `[ ]` | |

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
| 5.4 | Analyze (ZeAnalyser launch) | `[x]` | stdlib-only `seestar/gui_qt/analyzer_launch.py` seam + `MainWindow._on_analyse`; button enabled only for an existing input dir; non-blocking launch with `ZEANALYSER_COMMAND_FILE`; reference-return *consumption* seam present, periodic re-arming watcher deferred |

## 6. Preview

| # | Item | Status | Notes |
|---|---|---|---|
| 6.1 | Preview image rendering (array → `QImage`) | `[x]` | `preview_render` (display-only) |
| 6.2 | White-balance controls | `[x]` | `preview_adjust` per-channel R/G/B gains (neutral 1.0) shown as slider + numeric spinbox pairs, `Auto WB` (mode-based gains from the display image, Tk parity) + `Reset`; Tk range 0.1–5.0 step 0.01; display-only, re-renders the derived preview immediately (M10/M13) |
| 6.3 | Stretch controls (linear / asinh / log / auto) | `[x]` | `stretch_combo` (Asinh default, Tk parity) + black/white/gamma slider + numeric spinbox controls (Tk defaults 0.01/0.99/1.0, ranges/steps 0–1/0.001 and 0.1–5/0.01) + `Auto Stretch` (percentile bp/wp from the WB-only image → Asinh) + `Reset Stretch`; `preview_adjust` tone curves reproduce the Tk `StretchPresets` math; display-only, re-renders the derived preview (M10/M13/M14) |
| 6.4 | Histogram | `[x]` | single live surface in the persistent right panel (`HistogramView` QWidget) fed by the *WB-only* derived image (Tk `image_data_wb` source); reproduces the Tk auto-zoom / reset-view / zoom / reset-zoom interactions + BP/WP line dragging → `rangeChanged`; localized stats label; the M10 duplicated Preview-controls-tab histogram was removed (M10/M14) |
| 6.5 | Zoom (Fit / 100% / 200% / 50%) | `[x]` | display-only `preview_view` + `MainWindow` view controls; percent zoom scales from the rotated native size, Fit preserves aspect ratio |
| 6.6 | Rotation (left / right 90°) | `[x]` | cumulative ±90° modulo 360; preserves source image; zoom reapplies after rotation |
| 6.7 | Pan | `[ ]` | |
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
| 9.3 | Settings persistence (`seestar_settings.json` / XDG) | `[x]` | `settings_persistence` (pure stdlib) + `QtSettingsState.from_dict` coercion; CWD `seestar_settings.json` default (Tk convention), injectable path for tests (M8) |

## 10. Last stack / resume

| # | Item | Status | Notes |
|---|---|---|---|
| 10.1 | Last stack display / resume | `[ ]` | |

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
