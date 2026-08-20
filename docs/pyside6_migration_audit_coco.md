# P0 Audit — Tkinter GUI → PySide6 Migration Evidence (READ-ONLY)

- Author: Coco (subagent, Mission ZSSS-PYSIDE6 / P0 audit)
- Date: 2026-08-20
- Repository: `/home/tristan/.openclaw/workspace/projects/zeseestarstacker`
- Branch: `feature/pyside6-migration` (HEAD == `0d9af8b` — baseline, **0 commits ahead**; no PySide6 code exists yet)
- Scope: READ-ONLY. No source modified. Only this report file was created under `docs/`.

---

## 1. Evidence commands run and key outputs

| Command | Key output |
|---|---|
| `git branch --show-current && git log --oneline -5` | `feature/pyside6-migration`; HEAD `0d9af8b Fix M3-D drizzle preview group-size propagation` |
| `git rev-list --count 0d9af8b..HEAD` | `0` (branch is at baseline, nothing committed) |
| `git diff --stat 0d9af8b..HEAD` | empty (no diff) |
| `wc -l seestar/gui/*.py` | total **14,017** lines; `main_window.py` 6,860; `settings.py` 2,728; `boring_stack.py` 1,280; `histogram_widget.py` 794; `mosaic_gui.py` 623; `preview.py` 565; `local_solver_gui.py` 507; `progress.py` 227; `file_handling.py` 192; `analyzer_launch.py` 125; `ui_utils.py` 78; `__init__.py` 38 |
| `wc -l seestar/queuep/queue_manager.py` | **16,047** lines (backend `SeestarQueuedStacker`) |
| `grep -rin "pyside\|pyqt6" ...` | No PySide6/PyQt6 anywhere. **PyQt5 exists only** in `seestar/tools/visu.py` (standalone FITS viewer, optional `[tools]` extra) + `requirements.txt`/`pyproject.toml` `[optional-dependencies].tools` + stale `build/` artifact |
| `grep -rln "import tkinter\|from tkinter" seestar/gui/*.py` | 10 modules: boring_stack, file_handling, histogram_widget, local_solver_gui, main_window, mosaic_gui, preview, progress, settings, ui_utils |
| `grep -n "TkAgg\|backend_tkagg" seestar/gui/*.py` | `histogram_widget.py:9` `matplotlib.use('TkAgg')`; `:10` `FigureCanvasTkAgg` |
| `grep -c ":" seestar/localization/{fr,en}.py` | fr **359** keys, en **360** keys (1-key asymmetry) |
| `.venv/bin/python -c "<probe>"` (pure boundary modules) | `resolve_solver_gate` truth table verified (see §9); `SUPPORTED_API_MAJOR=1`, `REQUIRED_CAPABILITIES=('wcs_write',)`, `SOLVE_BACKEND_CAPABILITIES=('near_solve','blind_solve')`; `detect_analyzer_command()` → `['<venv>/python','-m','zeanalyser']` (ZeAnalyser installed in this venv); `parse_reference_from_command_file` → `/tmp/ref.fit`; lazy `import seestar.gui` exposes all 6 lazy attrs without creating a Tk root |
| `.venv/bin/python -m pytest tests/test_solver_gate.py test_solver_config.py test_analyzer_launch.py test_zesolver_adapter.py test_packaging.py test_threads.py test_m3d_settings.py test_drizzle_preview.py -q` | **87 passed** in 4.45s (4 warnings, colour_demosaicing deprecations only) |
| `grep -n "from \|import " main_window.py | grep queuep/core/alignment/enhancement/solver` | see §7 — GUI imports `SeestarQueuedStacker`, `APP_VERSION` (queue_manager), `load_and_validate_fits`/`downsample_image`/`debayer_image` (core), `load_config`/`resolve_solver_gate` (solver_config), `check_zesolver_readiness` (zesolver_adapter) |

Note: system `python3` lacks `cv2`; the probe must use `.venv/bin/python` (which contains the full stack). `seestar/__init__.py` imports `seestar.tools` → `stretch.py` → `cv2` at package import time, so **importing `seestar` requires OpenCV**.

---

## 2. Tk GUI module inventory

| Module | LOC | Tk role |
|---|---|---|
| `seestar/gui/main_window.py` | 6,860 | `SeestarStackerGUI` — main window, all `tk.Var` state, layout, callbacks, start/stop, summary |
| `seestar/gui/settings.py` | 2,728 | `SettingsManager` — full settings model + JSON persistence (Tk-coupled via `update_from_ui`/`apply_to_ui`) |
| `seestar/gui/boring_stack.py` | 1,280 | CLI `main()` for batch_size==1; also exports `read_rows`/`read_paths` consumed by GUI |
| `seestar/gui/histogram_widget.py` | 794 | `HistogramWidget(ttk.Frame)` — matplotlib `TkAgg` embedded, async hist calc |
| `seestar/gui/mosaic_gui.py` | 623 | `MosaicSettingsWindow(tk.Toplevel)` — mosaic params + duplicated solver/ASTAP config |
| `seestar/gui/preview.py` | 565 | `PreviewManager` — `tk.Canvas` image display, zoom/pan/rotate, WB/stretch/B/C/S |
| `seestar/gui/local_solver_gui.py` | 507 | `LocalSolverSettingsWindow(tk.Toplevel)` — ZeSolver/ASTAP config + readiness UI |
| `seestar/gui/progress.py` | 227 | `ProgressManager` — progress bar + status text + timers |
| `seestar/gui/file_handling.py` | 192 | `FileHandlingManager` — `filedialog` browse input/output/reference/temp, add_folder |
| `seestar/gui/analyzer_launch.py` | 125 | **GUI-free** ZeAnalyser subprocess helpers (no Tk/PIL/cv2 import) |
| `seestar/gui/ui_utils.py` | 78 | `ToolTip` |
| `seestar/gui/__init__.py` | 38 | lazy `__getattr__` accessors (6 symbols) |
| `seestar/localization/localization.py` + `fr.py`/`en.py` | — | `Localization`; translation catalogues |
| `seestar/main.py` | — | console entry point (`zeseestarstacker = seestar.main:main`) |

---

## 3. Responsibility map

Classification of every component (UI pure / presentation-orchestration / state-settings / engine call / engine callback / thread boundary / external interop / legacy-dead):

**UI pure (widget construction + event binding, no engine state):**
- `create_layout` (`main_window.py:879`–2288), `_create_slider_spinbox_group` (`:2942`), `_store_widget_references` (`:3004`)
- `ui_utils.ToolTip`
- `HistogramWidget` rendering (`plot_histogram`, `_on_press/_motion/_scroll`, zoom/reset)
- `PreviewManager` canvas drawing (`_redraw_canvas`, `_zoom_on_scroll`, `_start_pan/_pan_image/_stop_pan`, `zoom_fit/zoom_full_size`)
- `ProgressManager` (pure display + timer)
- dialog windows `MosaicSettingsWindow`, `LocalSolverSettingsWindow` (widgets + browse)

**Presentation-orchestration (glue between vars, widgets, and preview/histogram):**
- `_update_drizzle_options_state` (`:634`), `_update_final_scnr_options_state` (`:706`), `_update_feathering/_low_wht_mask/_photutils_bn/_bn/_cb/_crop_options_state` (`:2322`–2472)
- `update_ui_language` (`:3298`), `change_language` (`:3289`)
- `update_histogram_lines_from_sliders` (`:3417`), `update_stretch_from_histogram` (`:3428`), `_debounce_refresh_preview` (`:3506`)
- `apply_auto_stretch` (`:3730`), `apply_auto_white_balance` (`:4124`), `reset_*` (`:4177`–4203)
- `update_image_info` (`:3906`), `_try_show_first_input_image` (`:3960`)
- `update_remaining_files` (`:4383`), `update_additional_folders_display` (`:4414`), `update_add_folder_button_state` (`:4965`)

**State-settings (model + persistence):**
- `SettingsManager` (`gui/settings.py`) — the *de facto* settings model; `update_from_ui`/`apply_to_ui`/`validate_settings`/`get_default_values`/`save_settings`/`load_settings`/`export_run_settings`
- `seestar/core/settings.py` `Settings` dataclass — **minimal/legacy** (only `apply_batch_feathering`); effectively unused by the GUI (the GUI uses `gui/settings.py.SettingsManager`, not this). Flagged as confusing duplication.
- `seestar/core/solver_config.py` — second config store (`seestar_config.json` in XDG user dir), separate from `SettingsManager`

**Engine call (GUI → backend):**
- `start_processing` (`main_window.py:6388`) — builds `backend_kwargs` dict and calls `queued_stacker.start_processing(**backend_kwargs)`; also `_prepare_single_batch_if_needed` (`:6271`), `_get_auto_chunk_size` (`:6372`)
- `_run_boring_stack_process` (`:4481`) — spawns `boring_stack.py` as subprocess
- `stop_processing` (`:4773`) — `queued_stacker.stop()` / `boring_proc.terminate()`
- `SeestarQueuedStacker` instantiated in `__init__` (`:297`–313) with `settings=self.settings`

**Engine callback (backend → GUI):**
- `update_progress_gui` (`:5369`) — main-thread-guarded, parses `folder_count_update:`/`ETA_UPDATE:`/`UNALIGNED_INFO:` sentinel prefixes
- `update_preview_from_stacker` (`:3630`) — main-thread-guarded preview callback
- `_processing_finished` (`:5516`) — end-of-run orchestration (summary, preview/histo, auto-stretch)
- `_show_summary_dialog` (`:6092`), `_refresh_final_preview_and_histo` (`:5060`)/`_direct` (`:6015`)

**Thread boundary:**
- `gui_event_queue` (`GuiEventQueue`, `queue_manager.py:47`) — `queue.Queue` of callables, drained by `_poll_gui_events` (`main_window.py:5040`) via `root.after(50,...)`
- `_track_processing_progress` (`:4264`) — `GUI_ProgressTracker` daemon thread
- `_starter` (inside `start_processing`) — `BackendStarter` daemon thread
- `_worker` (inside `_run_boring_stack_process`) — `BoringStackWorker` daemon thread
- `_HIST_EXECUTOR = ThreadPoolExecutor(max_workers=1)` (`histogram_widget.py:25`)
- backend `SeestarQueuedStacker._worker` (`queue_manager.py:4486`) — `processing_thread`

**External interop (boundary to preserve):**
- `analyzer_launch.py` — ZeAnalyser subprocess (env `ZEANALYSER_COMMAND_FILE`, `REFERENCE=`/`TIMESTAMP=` contract)
- `zesolver_adapter.py` — ZeSolver public `zesolver.api.v1` only
- `solver_port.py` — internal transport-neutral solver contract + `zesolver_ui_state`
- `solver_config.py` — config contract (legacy `zemosaic_config.json` soft migration)

**Legacy-dead code (candidates, verify before touching):**
- `seestar/core/settings.py` `Settings` dataclass — near-dead duplicate of `gui/settings.py`
- `seestar/main.py` `check_dependencies()` and the massive `sys.path`/`__package__`/`importlib.util` debug scaffolding — legacy launch hacks (still *executed*, so "legacy" but not dead)
- `_to_slug` GUI-value→slug map (`main_window.py:76`) — small, still used by boring_stack launch
- `_LEGACY_SOLVER_PREFERENCES = ("ansvr","astrometry")` — soft-migration only

---

## 4. Behavioral parity matrix (function → Tk → PySide6 target → parity test)

| Function | Tk module/widget | Settings/state | Backend call | Callback | Thread | PySide6 target | Parity test |
|---|---|---|---|---|---|---|---|
| Window boot, icon, geometry | `main_window.py __init__` (`root=tk.Tk()`, `iconphoto`, `root.geometry`, `root.minsize`) | `window_geometry` | — | — | main | `QMainWindow`/`QWidget`, `setWindowIcon`, `resize`, `setMinimumSize` | instantiate headless; assert settings applied |
| Layout/panels | `create_layout` (ttk frames, notebook, scrollbars) | — | — | — | main | `QWidget`/`QSplitter`/`QScrollArea`/`QTabWidget` | widget tree build without display |
| Tk var state | `init_variables` (`tk.StringVar/IntVar/DoubleVar/BooleanVar`) | mirrors `SettingsManager` attrs | — | — | main | QML-free: `QSettings`-backed model or plain attrs + signals | default values equal `get_default_values()` |
| Start | `start_processing` (`:6388`) | `update_from_ui`→`validate_settings`→`apply_to_ui` | `queued_stacker.start_processing(**backend_kwargs)` | — | `BackendStarter` | QThread or `QThreadPool` start | kwargs dict snapshot identical |
| Progress | `update_progress_gui` (`:5369`) | — | — | `set_progress_callback` → `gui_event_queue.put` | main (via queue) | Qt signal/slot (queued connection) | message+percent forwarded |
| Preview | `update_preview_from_stacker` (`:3630`) | `current_preview_data` etc. | — | `set_preview_callback` | main | Qt queued signal → `QPixmap`/`QImage` | preview array → display |
| Histogram | `HistogramWidget.update_histogram` (`:161`) | — | — | — | `_HIST_EXECUTOR` | `FigureCanvasQTAgg` + `QThreadPool` | async hist result identical |
| Stop | `stop_processing` (`:4773`) | — | `queued_stacker.stop()` / `boring_proc.terminate()` | — | main | slot → cancel | stop flag observed |
| Finish | `_processing_finished` (`:5516`) | reads `queued_stacker.*` | — | — | main (via queue) | queued slot | summary fields equal |
| boring (batch=1) | `_run_boring_stack_process` (`:4481`) | `output_path` | subprocess `boring_stack.py` | stdout regex parse | `BoringStackWorker` | `QProcess` | retcode + final.fits |
| Solver config | `LocalSolverSettingsWindow` | `local_solver_preference`, `astap_*` | `save_config` | — | main | QDialog | settings persisted |
| ZeAnalyser launch | `_launch_folder_analyzer` (`:2602`), `_check_analyzer_command_file` (`:2685`) | `input_folder`, `lang` | `launch_analyzer` (subprocess) | — | main | `QProcess` | cmd/env contract |
| Language | `update_ui_language` (`:3298`) | `language` | — | — | main | retranslate | en/fr key parity |

---

## 5. Critical settings/mappings inventory

### batch_size semantics (validated `gui/settings.py:1368`–1400; used `main_window.py:6388`+)
- `<=0` (or `-1`) → **Auto**: QM estimates dynamically (`_estimate_batch_size`); `validate_settings` forces sentinel `-1`
- `0` → allowed **only** for `reproject_coadd_final` / final-combine `reproject_coadd` → "single batch in-memory" (`allow_mode_zero` logic `:1376`–1383)
- `1` → **boring_stack.py subprocess** + `stack_plan.csv` single batch; also `_prepare_single_batch_if_needed` (`:6271`) forces `stack_final_combine=mean`, disables drizzle/reproject, raises `FileNotFoundError` if CSV missing; `_toggle_boring_thread` (`:740`) sets `boring_thread_var` and disables batch spinbox; `_on_batch_size_changed` (`:761`) keeps the two in sync
- `>=2` → `queued_stacker.align_on_disk = True` (`:670` in `_starter`)
- `enable_preview = False` when `batch_size == 1` (`start_processing` `:6405`)

### drizzle_mode (`gui/settings.py:1721`–1735; valid `["Final","Incremental"]`)
- M3-D: `Final`/`Incremental` are a **resource/preview policy**, not two sciences. Scale/WHT/kernel/pixfrac stay enabled in both; only `drizzle_group_size` is enabled when `Incremental` (see `_update_drizzle_options_state` `:634`).

### drizzle_group_size (`gui/settings.py:1780`–1793)
- default `50`; coerce int, `<1` → reset to default. Passed to backend as `drizzle_group_size` (`start_processing` kwargs). `_coerce_drizzle_group_size` exists in queue_manager (`:371`).

### solver selection
- `local_solver_preference` ∈ `["none","astap","zesolver"]` (validated `gui/settings.py:2232`–2249)
- legacy `ansvr`/`astrometry` → treated as `zesolver` (both `resolve_solver_gate` `solver_config.py:217` and `AstrometrySolver._migrate_legacy_preference`)
- two UIs share this setting: `local_solver_gui.py` (dedicated window) **and** `mosaic_gui.py` (duplicated solver radio, `:176`–197)

### ASTAP
- `astap_path`, `astap_data_dir` (settings, `seestar_settings.json`); `astap_search_radius` (default 3.0 in code / 30.0 shown in some UI init — **note discrepancy**: `get_default_values` `:1287` = `3.0`, but `local_solver_gui.py:89` and `mosaic_gui.py:83` initialize `DoubleVar(30.0)`)
- `astap_downsample` (default 1 in settings vs config default 2), `astap_sensitivity` (100) stored in `seestar_config.json` (solver_config), written via `save_config` from `local_solver_gui._on_ok`

### ZeSolver readiness / setup refresh
- `check_zesolver_readiness()` (adapter) → `SolverDiscovery`; UI state via `zesolver_ui_state()` (`solver_port.py`); deferred refresh loop `zesolver_session_refresh_action` + `_schedule_zesolver_refresh` (250 ms tick) in `local_solver_gui.py`

### paths
- input/output/reference/last_stack/temp folders (`StringVar` + settings); `save_settings` writes CWD-relative `seestar_settings.json` (**CWD-dependence risk**, see §10)

### expert options
- hot pixels, neighborhood, BN (grid/perc/std/gain), CB, master-tile crop, photutils BN, feathering, batch feathering, low-WHT mask, SCNR, save-as-float32, preserve-linear-output, use_gpu, matching background for final, reproject_between_batches, reproject_coadd_final, mosaic_settings dict

### language
- `language` ∈ `["en","fr"]`; `Localization` (default `en`, fallback `en`); persisted in both `SettingsManager.language` and `solver_config.language`

### output settings
- `output_filename`, `save_final_as_float32` (uint16 default), `preserve_linear_output`, `stack_final_combine` (`mean` default; `_to_slug` maps `Reproject and coadd→reproject_coadd`, `Reproject→reproject`, `Mean→mean`, `Reject→reject`, `None→none`)

---

## 6. Threading / callback / cross-thread access analysis

**Threads spawned:**
1. `BackendStarter` (daemon, `start_processing` `:6827`) — reads UI settings off the Tk thread *by design* (snapshot), builds `backend_kwargs`, starts `_track_processing_progress`, calls `start_processing`.
2. `GUI_ProgressTracker` (daemon, `:4264`) — polls `queued_stacker.is_running()`, reads `processed_files_count`, `aligned_files_count`, `files_in_queue`, computes ETA; **all widget writes are queued** via `gui_event_queue.put(_gui_update)`.
3. `BoringStackWorker` (daemon, `:4730`) — runs `boring_stack.py` subprocess, parses stdout, queues GUI updates.
4. Backend `SeestarQueuedStacker._worker` (`queue_manager.py:4486`) — the actual `processing_thread`.
5. `_HIST_EXECUTOR` (`ThreadPoolExecutor(1)`) — histogram bin computation.
6. Backend `ProcessPoolExecutor` (quality metrics `_get_quality_executor` `:4375`) + drizzle subprocesses.

**Cross-thread mechanism:** a single `gui_event_queue` (`GuiEventQueue`, `queue.Queue` subclass) of *callables*; drained on the Tk main loop by `_poll_gui_events` (`:5040`, re-arms `root.after(50,...)`). Registration in `__init__`:
- `set_progress_callback(lambda m,p=None: gui_event_queue.put(lambda: update_progress_gui(m,p)))`
- `set_preview_callback(lambda *a,**k: gui_event_queue.put(lambda: update_preview_from_stacker(*a,**k)))`

**Guards:** `update_progress_gui` (`:5369`) and `update_preview_from_stacker` (`:3630`) both re-dispatch via `root.after(0, ...)` if not on `threading.main_thread()`. `HistogramWidget.update_histogram`/`_apply_histogram` and `PreviewManager.display_processed_image`/`update_preview` use `after_idle` guards.

**Risks observed (for parity design):**
- `_track_processing_progress` reads many `queued_stacker.*` attributes **without a lock** (snapshot semantics, tolerated but racy). `update_additional_folders_display` (`:4414`) does take `folders_lock` (timeout 0.15s).
- `update_progress` (`queue_manager.py:2913`) throttles via module-global `_QM_LAST_GUI_PUSH`/`_QM_DEBOUNCE` and **drops** messages when `gui_event_queue.qsize() > 500` (`put_nowait`, catch `Full`).
- `settings.update_from_ui`/`validate_settings`/`apply_to_ui` are called **both** on the main thread (before start) **and** from `_starter` thread — the model is mutated from a non-GUI thread while Tk vars may still reference it. PySide6 port must keep the settings snapshot model thread-local or move all of it into the GUI thread (recommended).
- `SeestarQueuedStacker.__getstate__/__setstate__` (`queue_manager.py:1092`/`1125`) exist → the backend is pickle-aware (relevant if port uses `QProcess`/pickle boundaries).

---

## 7. Dependencies GUI ↔ engine and Qt boundary recommendations

**Current GUI→engine imports (direct, not via a facade):**
- `from ..queuep.queue_manager import SeestarQueuedStacker` (`main_window.py:90`) and `GLOBAL_DRZ_BATCH_VERSION_STRING_ULTRA_DEBUG as APP_VERSION` (`:43`)
- `from ..core.image_processing import debayer_image, load_and_validate_fits` (`:115`)
- `from ..core.utils import downsample_image` (`:116`)
- `from seestar.core.solver_config import load_config, resolve_solver_gate` (`:41`)
- `from ..alignment.zesolver_adapter import check_zesolver_readiness` (`:49`)
- `from .boring_stack import read_paths` (`:50`)

**The clean engine boundary already exists** on the backend side: `SeestarQueuedStacker` exposes a public surface of `start_processing(**kwargs)`, `set_progress_callback`, `set_preview_callback`, `stop`, `is_running`, `add_folder`, `get_estimated_total_images`, and a documented `backend_kwargs` dict. The Qt port should **not** import engine internals beyond this surface.

**Qt boundary recommendations (design, not implemented):**
1. Keep `queue_manager`/`core`/`enhancement`/`alignment` **untouched**; expose a thin `BackendController` façade with the same `start_processing(**kwargs)` + signal emissions, replacing `gui_event_queue` with Qt queued signals/slots (or a `QObject`-based worker with `QThread`).
2. Replace `root.after(...)` recursion (`_poll_gui_events`, debounce timers) with `QTimer`; replace `after_idle` guards with `QMetaObject.invokeMethod(..., Qt.QueuedConnection)`.
3. Replace `matplotlib.use('TkAgg')`/`FigureCanvasTkAgg` with `FigureCanvasQTAgg` (note `tools/visu.py` already uses `backend_qt5agg`, proving the Qt matplotlib backend is available in this env).
4. Replace `ImageTk.PhotoImage` with `QImage`/`QPixmap` (Pillow already used for processing — keep Pillow as the image pipeline, swap only the display sink).
5. Keep `analyzer_launch.py` and `zesolver_adapter.py`/`solver_port.py`/`solver_config.py` **as-is** (they are GUI-free by design); they are the interop boundary.
6. Decouple `SettingsManager` from Tk: `update_from_ui`/`apply_to_ui` currently read/write `tk.Var` objects — refactor to a plain-attribute model + `collect()`/`apply()` against a widget-agnostic view interface.
7. Resolve the two-settings-model duplication (`core/settings.py.Settings` vs `gui/settings.py.SettingsManager`) and two-config-store split (`seestar_settings.json` CWD vs `seestar_config.json` XDG) as part of the migration (see §10).

---

## 8. Preview / histogram / stretch data-flow and display/science separation

**Data flow (preview):**
1. Backend computes `last_saved_data_for_preview` (cosmetic `[0,1]` non-stretched) and `raw_adu_data_for_ui_histogram` (ADU); stored as `queued_stacker` attributes; read in `_processing_finished` (`:5588`–5600).
2. During run, `set_preview_callback` → `update_preview_from_stacker(preview_array, stack_header, ...)` (`:3630`); tuple `(preview_display, preview_hist)` unpacked.
3. `PreviewManager.process_image` (`preview.py:167`) applies **display-only** WB (`white_balance`), stretch (`linear`/`asinh`/`log` via `StretchPresets`), gamma, then PIL `ImageEnhance` brightness/contrast/saturation; returns `(pil_img, hist_data)`. **Science pixels are never written here.**
4. `HistogramWidget.update_histogram` (`histogram_widget.py:161`) downsamples `[::4,::4]`, submits `_calculate_hist_data` to `_HIST_EXECUTOR`, then `_apply_histogram` plots on main thread. Auto-range logic (`data_min/max_for_current_plot`, `freeze_x_range`) lives in `_calculate_hist_data` (`:232`).
5. Stretch controls (`update_histogram_lines_from_sliders` `:3417`, `update_stretch_from_histogram` `:3428`, `apply_auto_stretch` `:3730`, `apply_auto_white_balance` `:4124`) mutate only preview `tk.DoubleVar`s (`preview_black_point`, `preview_white_point`, `preview_gamma`, `preview_r/g/b_gain`, brightness/contrast/saturation) → never touch backend output.

**Separation assessment:** Good. Display math is in `seestar/tools/stretch.py` (`StretchPresets`, `ColorCorrection`) and `PreviewManager`; histogram math in `HistogramWidget`; science stack math entirely in `queue_manager`/`core`. The only display/science coupling points are (a) the backend also returns `raw_adu_data_for_ui_histogram` (a *copy* for UI, fine), and (b) `_processing_finished` calls `load_and_validate_fits(final_stack_path)` + `downsample_image` in the GUI thread for the final preview — a science util reused for display (acceptable, but should be moved behind the façade in Qt).

---

## 9. Solver / ZeSolver / ZeAnalyser boundary findings

**ZeSolver (optional, public API v1 only) — `zesolver_adapter.py`:**
- Only `zesolver.api.v1` imported (string `_ZESOLVER_API_MODULE`), lazily inside `discover_zesolver()`; **no** sibling checkout scan, **no** `sys.path` mutation, **no** private imports.
- Compatibility decided on `API_MAJOR` (==1) + declared/negotiated capabilities, never Git/product version.
- `REQUIRED_CAPABILITIES=("wcs_write",)`, `SOLVE_BACKEND_CAPABILITIES=("near_solve","blind_solve")`, optional `("cancel","gpu")` (`solver_port.py`).
- `check_zesolver_readiness()` layers `v1.readiness()` on top of discovery; returns `SolverDiscovery` with `operational`/`configuration_needed`.
- `open_zesolver_configuration()` returns opaque handle (v1.2+) or `None` (v1.1); `zesolver_session_refresh_action` drives non-busy `after()` poll.
- Adapter `solve()` **never raises** (returns `SolverOutcome`); least-destructive `WritePolicy` preferred (never overwrites input); `network_policy=DISABLED` enforced.
- Verified truth table of `resolve_solver_gate` (see §1): `zesolver+operational→allow`; `zesolver+not-operational+astap_configured→allow`; `zesolver+not-operational+no_astap→block "zesolver_unavailable_no_astap"`; `astap+configured→allow`; `astap+no→block "astap_not_configured"`; `none→block "no_solver_configured"`; legacy `ansvr`/`astrometry`→`zesolver`.

**ASTAP:** local executable fallback; configured via `astap_path`/`astap_data_dir`/`astap_search_radius`; gate `astap_configured = bool(astap_path.strip())` (`main_window.py:6449`).

**ZeAnalyser (external process contract) — `analyzer_launch.py`:**
- Discovered at runtime (not a declared dependency): `shutil.which("zeanalyser")` → `python -m zeanalyser`.
- Command file protocol: env `ZEANALYSER_COMMAND_FILE`; file lines `REFERENCE=<path>` / `TIMESTAMP=<...>` (protocol v1; locked by `tests/test_process_contract.py`).
- `launch_analyzer` is non-blocking `subprocess.Popen`, `popen` injectable (tested).
- GUI consumption: `_launch_folder_analyzer` (`main_window.py:2602`) + `_check_analyzer_command_file` surveillance (`:2685`) + `consume_command_file`.
- Note: `detect_analyzer_command()` returned the venv `python -m zeanalyser` path in this environment (ZeAnalyser installed here) — confirm this is the *sibling product*, not a `seestar` internal, before assuming.

**Boundary integrity:** all three boundaries are already clean and GUI-free; **preserve verbatim** during the Qt migration.

---

## 10. Risks / blockers / out-of-scope discoveries

1. **CWD-dependence of `SettingsManager`**: `SETTINGS_FILENAME = "seestar_settings.json"` opened as a **relative** path (`gui/settings.py:27`, `open(self.settings_file,...)`); writes/reads depend on the process CWD. `solver_config.py` already fixed this with an XDG user dir — the settings file should follow suit. (Out of migration scope but a parity hazard.)
2. **Two settings models**: `core/settings.py.Settings` (near-dead) vs `gui/settings.py.SettingsManager` (live). Migration must not silently switch which model the backend reads.
3. **Two config stores**: `seestar_settings.json` (GUI settings, CWD) vs `seestar_config.json` (solver config, XDG). Solver/ASTAP downsample/sensitivity live in the *latter*; everything else in the former.
4. **`astap_search_radius` default discrepancy**: `get_default_values` = `3.0` but `local_solver_gui.py:89` / `mosaic_gui.py:83` initialize the Tk var to `30.0`. Verify which is authoritative before porting.
5. **`SettingsManager.update_from_ui`/`validate_settings` called from a non-GUI thread** (`_starter` in `start_processing`) — mutates shared model concurrently with Tk-var access. Qt port should move validation to the GUI thread (recommended) or snapshot fully.
6. **`backend_kwargs` is a 60+ key hand-built dict** (`main_window.py:6717`–6830) with no schema/typing — the single highest-risk coupling point; a schema/typing contract or façade is strongly recommended for parity.
7. **Pre-existing PyQt5** in `seestar/tools/visu.py` (standalone viewer, `[tools]` extra). The PySide6 migration of the *main* GUI is independent of this tool; ensure the two Qt stacks (PyQt5 tool vs PySide6 app) are not cross-imported.
8. **`matplotlib.use('TkAgg')` is a global side effect** on import of `histogram_widget.py` — the Qt port must switch to `QtAgg` and remove the global call.
9. **`seestar/__init__.py` imports `cv2` transitively** (via `tools.stretch`), so `import seestar` requires OpenCV — affects headless import tests for the Qt port.
10. **`seestar/main.py` sys.path/`__package__`/`importlib.util` scaffolding** is launch-specific and fragile; the Qt entry point should use the `gui-scripts` console entry (`seestar.main:main`) and drop the debug scaffolding.
11. **Stale `build/` and `C:/data/...` artifacts** and `seestar/requirements.txt` (duplicate of root `requirements.txt`) are present in-tree — out of scope, but should be cleaned/handled to avoid package-data surprises.
12. **`review/` and `docs/M3D_*.md`** contain in-flight design notes (M3-D / W-1 / mode-0 reproject) — do not treat as stable spec without Jarvis confirmation.
13. **No PySide6/PyQt6 code exists yet** on the branch (0 commits ahead of baseline). Any test that expects Qt widgets cannot pass until migration begins — plan parity gates as *behavioral* (settings/kwargs/backend surface), not visual, initially.

---

## 11. Proposed milestones / tests / gates

**M0 — Boundary freeze (no Qt yet):**
- Add a typed/schema `backend_kwargs` contract (or façade) so GUI→backend call is testable headless.
- Move `SettingsManager.update_from_ui/validate_settings` fully onto the GUI thread; add thread-safety test.
- Move `seestar_settings.json` to XDG user dir (mirror `solver_config`).
- Gate: existing 87-test subset (solver_gate, solver_config, analyzer_launch, zesolver_adapter, packaging, threads, m3d_settings, drizzle_preview) stays green.

**M1 — Settings/view decoupling:**
- Refactor `SettingsManager` to plain-attribute model + `collect(view)`/`apply(view)` with a Tk view adapter (and later a Qt view adapter).
- Gate: new test `test_settings_model_roundtrip` (defaults ↔ collect ↔ validate ↔ apply) passing on both Tk and a fake view.

**M2 — Widget port (UI pure):**
- Port `main_window` layout + `PreviewManager` + `HistogramWidget` + `ProgressManager` + dialogs to PySide6, keeping manager logic shared.
- Gate: `test_gui_constructs_headless` (offscreen QPA) + visual smoke via `QT_QPA_PLATFORM=offscreen`.

**M3 — Threading/callbacks:**
- Replace `gui_event_queue` + `root.after` recursion with Qt queued signals/`QTimer`; keep `set_progress_callback`/`set_preview_callback` signatures.
- Gate: `test_threads.py`-equivalent + a new `test_qt_signal_forward` asserting cross-thread delivery.

**M4 — Parity acceptance:**
- Side-by-side behavioral parity: settings snapshot, `backend_kwargs` dict, preview data-flow, solver gate, ZeAnalyser launch, boring_stack subprocess, language switch.
- Gate: a `test_parity_*` suite comparing Tk and Qt runs against the *backend surface only* (not pixels), plus the frozen solver/analyser contract tests.

**Explicit non-goals:** no new functional Qt/PySide6 code in this audit; no widget replacement now; no Tk removal now; engine/science (`Drizzle`/`Reproject`/`Solver`/`QueueManager`/`boring_stack`) frozen; ZeSoftware interop (no ZeAlfie dependency, no private ZeSolver imports, no sibling checkout/CWD/sys.path hacks) preserved.
