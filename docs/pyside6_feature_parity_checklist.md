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
| 1.6 | `1` → boring_stack subprocess / `stack_plan.csv` routing | `[ ]` | Qt routing of the CSV path is a later milestone |

## 2. Final combination + interdependencies

| # | Item | Status | Notes |
|---|---|---|---|
| 2.1 | Final-combine choices (mean / median / winsorized_sigma_clip / reproject / reproject_coadd) | `[x]` | `final_combine_combo` (Stacking tab) with the five historical Tk keys |
| 2.2 | `reproject_between_batches` ⇄ `reproject_coadd_final` mutual exclusion | `[x]` | single source of truth: `stack_final_combine` drives both flags via `final_combine_flags` |
| 2.3 | Drizzle / reproject / boring-thread interdependencies (button gating) | `[ ]` | |

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
| 4.1 | Browse input folder | `[ ]` | `input_edit` is a plain `QLineEdit`, no dialog |
| 4.2 | Browse output folder | `[ ]` | |
| 4.3 | Reference image path | `[x]` | Settings surface field only (no dialog) |
| 4.4 | Temp folder | `[x]` | Stack tab field |
| 4.5 | Last-stack / last output path persistence | `[ ]` | |

## 5. Inputs / folders / output / analyzer

| # | Item | Status | Notes |
|---|---|---|---|
| 5.1 | View inputs | `[ ]` | |
| 5.2 | Add folder | `[ ]` | |
| 5.3 | Open output | `[ ]` | |
| 5.4 | Analyze (ZeAnalyser launch) | `[ ]` | |

## 6. Preview

| # | Item | Status | Notes |
|---|---|---|---|
| 6.1 | Preview image rendering (array → `QImage`) | `[x]` | `preview_render` (display-only) |
| 6.2 | White-balance controls | `[ ]` | |
| 6.3 | Stretch controls (linear / asinh / log / auto) | `[ ]` | |
| 6.4 | Histogram | `[ ]` | |
| 6.5 | Zoom / pan | `[ ]` | |
| 6.6 | Rotation | `[ ]` | |

## 7. Progress / log / copy

| # | Item | Status | Notes |
|---|---|---|---|
| 7.1 | Progress bar (0..100) | `[x]` | queued `progress_changed` |
| 7.2 | Log view | `[x]` | read-only `QTextEdit` |
| 7.3 | Copy log to clipboard | `[ ]` | |

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
| 13.5 | Action buttons Start / Stop / Analyse / Solver / View Inputs / Add Folder / Open Output | `[x]` | Start/Stop functional; the rest are disabled topology stubs |
| 13.6 | Zoom / resolution / rotation controls (real interactivity) | `[ ]` | basic disabled placeholders only |
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
