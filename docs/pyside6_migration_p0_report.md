# P0 Audit Report — Tkinter GUI → PySide6 Migration

- Repository: `tinystork/zeseestarstacker`
- Local path: `/home/tristan/.openclaw/workspace/projects/zeseestarstacker`
- Branch: `feature/pyside6-migration`
- Required baseline: `0d9af8bf28d508c86ef63d0e471411077e224b58`
- Snapshot: `zsss_beta_0d9af8b_src.zip`
- Verdict: **READY TO IMPLEMENT**
- Gate: **STOP after this report** until human validation. No PySide6 migration has started.

## 1. Baseline Git verified

Verified locally:

- `git rev-parse HEAD` → `0d9af8bf28d508c86ef63d0e471411077e224b58`
- `git rev-parse --abbrev-ref HEAD` → `feature/pyside6-migration`
- `git status --short` after audit docs → only:
  - `?? docs/pyside6_migration_audit_coco.md`
  - `?? docs/pyside6_migration_p0_report.md`
- No source-code diff outside audit documentation.

## 2. Evidence and validation

Read-only audit evidence is recorded in `docs/pyside6_migration_audit_coco.md`.

Independent validation rerun by Jarvis:

```bash
.venv/bin/python -m pytest \
  tests/test_solver_gate.py \
  tests/test_solver_config.py \
  tests/test_analyzer_launch.py \
  tests/test_zesolver_adapter.py \
  tests/test_packaging.py \
  tests/test_threads.py \
  tests/test_m3d_settings.py \
  tests/test_drizzle_preview.py -q
```

Result: **87 passed, 4 warnings**.

Note: an initial shortened pytest command without `tests/` prefixes failed with `file or directory not found: test_solver_config.py`; corrected command above passed.

## 3. Tkinter module inventory

Tk GUI inventory:

- `seestar/gui/main_window.py` — `SeestarStackerGUI`, main window, Tk variables, layout, start/stop, callbacks, summary.
- `seestar/gui/settings.py` — `SettingsManager`, JSON persistence, `update_from_ui`, `apply_to_ui`, `validate_settings`, `get_default_values`, `save_settings`, `load_settings`.
- `seestar/gui/preview.py` — `PreviewManager`, `tk.Canvas`, `ImageTk.PhotoImage`, zoom/pan/rotate/stretch display.
- `seestar/gui/histogram_widget.py` — `HistogramWidget(ttk.Frame)`, `matplotlib.use('TkAgg')`, `FigureCanvasTkAgg`, async histogram calculation.
- `seestar/gui/progress.py` — `ProgressManager`, status text, progress bar, timers.
- `seestar/gui/local_solver_gui.py` — `LocalSolverSettingsWindow(tk.Toplevel)`, ASTAP/ZeSolver config and readiness refresh.
- `seestar/gui/mosaic_gui.py` — `MosaicSettingsWindow(tk.Toplevel)`, mosaic settings and duplicated solver controls.
- `seestar/gui/file_handling.py` — file/folder/reference/temp dialogs.
- `seestar/gui/boring_stack.py` — CLI subprocess path for `batch_size == 1`, plus CSV readers.
- `seestar/gui/analyzer_launch.py` — GUI-free ZeAnalyser subprocess boundary.
- `seestar/gui/ui_utils.py` — `ToolTip`.
- `seestar/gui/__init__.py` — lazy exports.
- `seestar/main.py` — entry point for `zeseestarstacker = "seestar.main:main"`.

Measured audit inventory: GUI Python files total roughly **14,017 LOC**; `seestar/queuep/queue_manager.py` roughly **16,047 LOC**.

## 4. Responsibility map

Classification:

- UI pure: layout/widgets in `create_layout`, widget references, `ToolTip`, histogram rendering, preview canvas rendering, progress widgets, solver/mosaic dialogs.
- Presentation orchestration: option-state toggles, language refresh, histogram/stretch/white-balance controls, preview refresh, folder counters.
- State/settings: `seestar/gui/settings.py.SettingsManager`; separate solver config in `seestar/core/solver_config.py`.
- Engine call: `main_window.py:start_processing`, `_prepare_single_batch_if_needed`, `_run_boring_stack_process`, `stop_processing`.
- Engine callback: `update_progress_gui`, `update_preview_from_stacker`, `_processing_finished`, `_show_summary_dialog`, `_refresh_final_preview_and_histo`.
- Thread boundary: `GuiEventQueue`, `_poll_gui_events`, `BackendStarter`, `GUI_ProgressTracker`, `BoringStackWorker`, backend `processing_thread`, histogram executor.
- External interop: `analyzer_launch.py`, `zesolver_adapter.py`, `solver_port.py`, `solver_config.py`.
- Legacy/confusing: `seestar/core/settings.py.Settings` appears near-dead versus live `SettingsManager`; `seestar/main.py` contains legacy launch scaffolding; `_LEGACY_SOLVER_PREFERENCES` supports soft migration.

## 5. GUI entry points and startup/shutdown

Startup:

- Console entry: `zeseestarstacker = "seestar.main:main"`.
- Main app constructs `SeestarStackerGUI` and a `SeestarQueuedStacker` instance.
- `seestar/gui/__init__.py` uses lazy attributes and should remain import-light.

Shutdown:

- `_on_closing` handles processing-active confirmation.
- `stop_processing` delegates to `queued_stacker.stop()` or terminates `boring_proc`.
- `_processing_finished` resets UI state, finalizes progress/timer, preview/histogram refresh, summary.

PySide6 target:

- `QApplication` + `QMainWindow`/central widget.
- `closeEvent` replaces `_on_closing`.
- `QTimer` replaces recurring `root.after` polling.
- `QObject` signals replace queued Tk callables.

## 6. Behavioral parity matrix

| Behavior | Current Tk path | Critical state/backend | PySide6 target | Parity gate |
|---|---|---|---|---|
| Boot/window geometry | `main_window.py __init__`, `root.geometry` | `window_geometry` | `QMainWindow.resize/restoreGeometry` | construct offscreen, assert geometry/settings |
| Layout/tabs | `create_layout`, ttk frames/notebook | widget references | `QWidget`, `QSplitter`, `QScrollArea`, `QTabWidget` | headless widget tree smoke |
| Settings model | `tk.StringVar/IntVar/DoubleVar/BooleanVar` | `SettingsManager` | plain model + Qt widget adapter | defaults/roundtrip test |
| Start processing | `start_processing` | `backend_kwargs`, `queued_stacker.start_processing(**backend_kwargs)` | controller slot + worker start | kwargs snapshot identical |
| Progress | `update_progress_gui` | `set_progress_callback` | queued Qt signal | message/level/percent forwarded |
| Preview | `update_preview_from_stacker` | `set_preview_callback` | queued Qt signal → `QImage/QPixmap` | array/header metadata forwarded |
| Histogram | `HistogramWidget` | async histogram math | `FigureCanvasQTAgg` + `QThreadPool` | identical bins/range on fixture |
| Stop/cancel | `stop_processing` | `queued_stacker.stop`, `boring_proc.terminate` | cancel slot/QProcess terminate | stop flags observed |
| Finish | `_processing_finished` | summary/preview/final state | queued finish slot | summary fields identical |
| `batch_size == 1` | `_run_boring_stack_process` | `boring_stack.py`, `stack_plan.csv` | `QProcess` or subprocess worker | stdout parse/retcode/final path |
| Solver settings | `LocalSolverSettingsWindow` | `save_config`, `resolve_solver_gate` | `QDialog` | config persistence + gate truth table |
| ZeAnalyser | `_launch_folder_analyzer`, `_check_analyzer_command_file` | `ZEANALYSER_COMMAND_FILE` | `QProcess`/subprocess wrapper | command-file contract |
| Language | `update_ui_language` | `Localization`, `language` | retranslate | EN/FR key parity |

## 7. Critical settings: batch semantics

Preserve exactly:

- `batch_size < 0` / `-1`: auto mode; QueueManager estimates batch size when no stack plan is present.
- `batch_size = 0`: explicit in-memory single-batch/reproject-coadd workflow; may force/adjust `reproject_coadd_final`, `stack_final_combine`, `freeze_reference_wcs`, and disable inter-batch reprojection as current behavior dictates.
- `batch_size = 1`: special CSV-driven single-batch/boring stack mode via `_prepare_single_batch_if_needed`; requires `stack_plan.csv`; missing CSV raises `FileNotFoundError` after log/message `"Batch size 1 without CSV – aborting"`.
- `batch_size >= 2`: normal batched workflow; GUI starter sets `queued_stacker.align_on_disk = True`.

CSV identifiers to preserve:

- `stack_plan.csv`
- `file_path`, `order`, `file`, `filename`, `path`, `index`
- Messages:
  - `"Batch size 1 without CSV – aborting"`
  - `"Stack plan CSV detected at '{csv_path}'. Preparing single batch"`
  - `"Stack plan CSV is empty"`

## 8. Critical settings: drizzle and output

Preserve:

- `drizzle_processing_label`
- `drizzle_processing_standard`
- `drizzle_processing_incremental`
- Drizzle modes: `Final`, `Incremental`
- `drizzle_group_size` default/coercion, currently default 50.
- Drizzle Standard/Large dataset semantics: mode is resource/preview policy, not a science-mode fork.
- Output strings/settings:
  - `output_format_frame_title`
  - `save_as_float32_label`
  - `preserve_linear_output_label`
  - `save_final_as_float32`
  - `preserve_linear_output`
  - `stack_final_combine`

## 9. Critical settings: solver / ASTAP / ZeSolver

Preserve:

- `local_solver_preference`: `none`, `astap`, `zesolver`.
- Legacy solver preferences: `ansvr`, `astrometry` map into ZeSolver compatibility path.
- `resolve_solver_gate` behavior and exact block labels:
  - `zesolver_unavailable_no_astap`
  - `astap_not_configured`
  - `no_solver_configured`
- ASTAP settings:
  - `astap_path`
  - `astap_data_dir`
  - `astap_search_radius`
  - `astap_downsample`
  - `astap_sensitivity`
- ZeSolver public API-only boundary:
  - `_ZESOLVER_API_MODULE = "zesolver.api.v1"`
  - `SUPPORTED_API_MAJOR`
  - `REQUIRED_CAPABILITIES`
  - `SOLVE_BACKEND_CAPABILITIES`
  - `check_zesolver_readiness`
  - `open_zesolver_configuration`
  - `zesolver_session_refresh_action`
  - `zesolver_ui_state`
  - `_ZESOLVER_REFRESH_TICK_MS`

No private ZeSolver imports, no sibling checkout, no CWD/sys.path hack, no mandatory ZeAlfie dependency.

## 10. Critical settings: language, geometry, paths, FITS/expert options

Preserve:

- Language: `language`, FR/EN localization keys, `reset_expert_button`, `tab_expert_title`.
- Geometry: `window_geometry` / current persistence behavior.
- Paths: input, output, reference, temp, last-stack style settings.
- Expert/FITS options: hot pixels, Bayer pattern, BN/CB/photutils BN, feathering, low-WHT mask, SCNR, GPU flag, matching background, crop settings, `save_final_as_float32`, `preserve_linear_output`.

Risk: `seestar_settings.json` is CWD-relative, unlike XDG-based `seestar_config.json` in `solver_config.py`.

## 11. Threading, workers, callbacks

Current model:

- Backend worker: `SeestarQueuedStacker._worker` running inside `processing_thread`.
- GUI starter: `BackendStarter` daemon thread.
- Progress tracker: `GUI_ProgressTracker` daemon thread.
- Boring mode worker: `BoringStackWorker` daemon thread plus subprocess.
- Histogram worker: `_HIST_EXECUTOR = ThreadPoolExecutor(max_workers=1)`.
- Backend quality/drizzle paths may use process pools/subprocesses.

Current callback bridge:

- Backend uses `set_progress_callback` and `set_preview_callback`.
- GUI uses `GuiEventQueue` of callables drained by `_poll_gui_events` with `root.after(50)`.
- `update_progress_gui` and `update_preview_from_stacker` guard with `root.after(0)` if called off main thread.

PySide6 replacement:

- `QObject` signals with queued connections for progress/preview/finished.
- `QTimer` for polling/debounce where polling remains necessary.
- `QThread` or `QThreadPool` for starter/progress/histogram jobs.
- Keep backend callback signatures unchanged initially.

## 12. Cross-thread access findings

Observed risks:

- `_track_processing_progress` reads `queued_stacker.*` counters without a lock; accepted today as snapshot semantics.
- `queue_manager.update_progress` throttles globally and may drop GUI messages when queue size exceeds 500.
- `SettingsManager.update_from_ui`, `validate_settings`, and `apply_to_ui` can be invoked from the `_starter` thread; this mutates the settings model and may touch Tk-backed variables off the GUI thread.

Migration rule:

- In PySide6, collect and validate settings on the GUI thread, then pass an immutable snapshot to the worker/controller.
- No worker should mutate Qt widgets or Qt-bound state directly.

## 13. GUI ↔ engine dependency map

Current direct imports from GUI into engine/core:

- `SeestarQueuedStacker` from `seestar/queuep/queue_manager.py`
- `APP_VERSION` alias from QueueManager debug version constant
- `load_and_validate_fits`, `debayer_image` from `seestar/core/image_processing.py`
- `downsample_image` from `seestar/core/utils.py`
- `load_config`, `resolve_solver_gate` from `seestar/core/solver_config.py`
- `check_zesolver_readiness` from `seestar/alignment/zesolver_adapter.py`
- `read_paths` from `seestar/gui/boring_stack.py`

Recommendation:

- Keep engine Qt-independent.
- Introduce a thin GUI-side `BackendController` facade later, but do not refactor scientific code.
- Preserve `SeestarQueuedStacker.start_processing(**kwargs)`, `set_progress_callback`, `set_preview_callback`, `stop`, `is_running`, `add_folder`, `get_estimated_total_images` during first migration milestones.

## 14. Preview / histogram / stretch separation

Current separation is acceptable:

- Science stacking remains in `queue_manager`, `core`, `enhancement`, `alignment`.
- Display-only processing lives in `PreviewManager.process_image`, `StretchPresets`, `ColorCorrection`, and `HistogramWidget`.
- Preview applies WB/stretch/gamma/brightness/contrast/saturation for display only.
- Histogram uses downsampled preview/hist data and does not write science output.

PySide6 target:

- Replace `ImageTk.PhotoImage` with `QImage/QPixmap`.
- Replace `FigureCanvasTkAgg` with `FigureCanvasQTAgg`.
- Do not move display stretch into backend science path.

## 15. Solver / ZeSolver / ZeAnalyser boundaries

Boundaries are already clean and must be preserved verbatim unless a specific independent bug is demonstrated:

- `seestar/alignment/zesolver_adapter.py`: lazy public import `zesolver.api.v1`, public API v1 compatibility, no private imports, no `sys.path` hacks.
- `seestar/alignment/solver_port.py`: transport-neutral solver state/capability model.
- `seestar/core/solver_config.py`: solver config persistence and `resolve_solver_gate`.
- `seestar/gui/analyzer_launch.py`: GUI-free subprocess contract for ZeAnalyser using `ZEANALYSER_COMMAND_FILE`, `REFERENCE=`, `TIMESTAMP=`.

ASTAP remains fallback.

## 16. Packaging / resources / installed-CWD independence

Findings:

- Main entry point is declared via `pyproject.toml`: `zeseestarstacker = "seestar.main:main"`.
- No PySide6/PyQt6 main GUI code exists yet.
- Pre-existing PyQt5 is limited to `seestar/tools/visu.py` and optional tools dependencies; do not cross-import it into the PySide6 main app.
- `histogram_widget.py` currently has `matplotlib.use('TkAgg')`, a global backend side effect to remove/replace during Qt port.
- `seestar/__init__.py` transitively requires `cv2` through tools/stretch import path; headless import tests must use `.venv/bin/python` with OpenCV available.
- `seestar_settings.json` CWD-relative persistence is a packaging/install risk.

## 17. Risks and blockers

Risks to handle before/during implementation:

1. CWD-relative `seestar_settings.json`.
2. Two settings models: `seestar/core/settings.py.Settings` vs `seestar/gui/settings.py.SettingsManager`.
3. Two config stores: GUI settings vs solver config.
4. `astap_search_radius` discrepancy: `3.0` default in settings vs `30.0` Tk dialog initialization.
5. Non-GUI-thread mutation of settings in `_starter`.
6. 60+ key untyped `backend_kwargs` dict.
7. Global `matplotlib.use('TkAgg')` side effect.
8. Existing PyQt5 standalone tool must remain isolated from PySide6 main GUI.
9. Launch/import scaffolding in `seestar/main.py` is fragile.
10. Stale build/design artifacts should not be treated as spec.

No hard blocker found that prevents starting implementation after human validation.

## 18. Out of scope for P0

Not done and intentionally not started:

- No PySide6 widget implementation.
- No Tk removal.
- No engine/science refactor.
- No Drizzle/Reproject/Solver/QueueManager/boring_stack functional change.
- No ZeAlfie dependency.
- No private ZeSolver import.
- No sibling checkout/CWD/sys.path inter-project hack.
- No push, merge, tag, release, or deployment.

## 19. Proposed milestones and gates

M0 — Boundary freeze / parity harness:

- Add typed/schema contract for `backend_kwargs` or facade snapshot.
- Move settings collection/validation to GUI thread.
- Decide handling for CWD-relative settings.
- Gate: existing 87-test subset remains green.

M1 — Settings/view decoupling:

- Refactor `SettingsManager` toward plain model + Tk adapter.
- Gate: defaults/collect/validate/apply roundtrip tests with fake view and Tk adapter.

M2 — UI pure PySide6 skeleton:

- Port layout, progress, preview display sink, histogram canvas, dialogs.
- Gate: `QT_QPA_PLATFORM=offscreen` construction smoke.

M3 — Threading/callback migration:

- Replace `GuiEventQueue`/`root.after` bridge with Qt signals and `QTimer`.
- Gate: cross-thread signal forwarding tests equivalent to `tests/test_threads.py`.

M4 — Behavioral parity:

- Compare Tk and Qt backend-surface snapshots: settings, `backend_kwargs`, solver gate, ZeAnalyser command, boring stack handling, preview metadata, language switch.
- Gate: parity tests pass without pixel-level visual claims.

## 20. Final verdict

**READY TO IMPLEMENT** after Tristan validates this P0 audit.

The migration should start with M0/M1 boundary and settings work, not direct widget replacement. The safest first implementation step is to create a testable settings/backend snapshot boundary while keeping Tk live and the engine untouched.

STOP GATE: no PySide6 migration work has been started. Awaiting human GO.
