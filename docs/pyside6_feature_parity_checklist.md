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
| 4.5 | Last-stack / last output path persistence | `[ ]` | browse + `last_stack_path` state field added; persistence later |

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
| 6.2 | White-balance controls | `[ ]` | |
| 6.3 | Stretch controls (linear / asinh / log / auto) | `[ ]` | |
| 6.4 | Histogram | `[ ]` | |
| 6.5 | Zoom (Fit / 100% / 200% / 50%) | `[x]` | display-only `preview_view` + `MainWindow` view controls; percent zoom scales from the rotated native size, Fit preserves aspect ratio |
| 6.6 | Rotation (left / right 90°) | `[x]` | cumulative ±90° modulo 360; preserves source image; zoom reapplies after rotation |
| 6.7 | Pan | `[ ]` | |

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
| 8.1 | FR / EN language switch | `[ ]` | |
| 8.2 | `Localization` key parity | `[ ]` | |

## 9. Settings / geometry persistence

| # | Item | Status | Notes |
|---|---|---|---|
| 9.1 | Settings surface (full `QtSettingsState` mirror) | `[x]` | grouped, scrollable Settings tab |
| 9.2 | Window geometry save/restore | `[ ]` | |
| 9.3 | Settings persistence (`seestar_settings.json` / XDG) | `[ ]` | |

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

## 13. Shell topology (Tk layout parity)

| # | Item | Status | Notes |
|---|---|---|---|
| 13.1 | Left/right `QSplitter` (control panel + persistent preview/action panel) | `[x]` | `_build_central` + `_build_left_panel` / `_build_right_panel` |
| 13.2 | Scrollable left panel (language + tabs + progress/log) | `[x]` | `QScrollArea` wrapping language combo, `QTabWidget`, progress, log |
| 13.3 | Left tabs `Stacking` / `Expert` / `Preview controls` | `[x]` | replaces former `Stack`/`Settings`/`Preview`/`Log` top-level tabs |
| 13.4 | Persistent right panel (preview + metadata + view + histogram + actions) | `[x]` | stays visible across left-tab switches |
| 13.5 | Action buttons Start / Stop / Analyse / Solver / View Inputs / Add Folder / Open Output | `[x]` | Start/Stop/Solver/View Inputs/Add Folder/Open Output functional; Analyse disabled |
| 13.6 | Zoom / resolution / rotation controls (real interactivity) | `[x]` | zoom (Fit/100/200/50), resolution label (orig → displayed + zoom + rotation), rotate left/right; display-only, offscreen-tested |
| 13.7 | Language switch (FR/EN) | `[ ]` | placeholder combo (disabled) |

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
