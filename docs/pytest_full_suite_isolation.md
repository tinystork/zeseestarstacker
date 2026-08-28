# pytest full-suite isolation (PYISO-01)

## Purpose

The full `pytest tests/` run must be **order-independent**: no test module may
leave a fake *canonical internal* `seestar.*` module behind in `sys.modules`
that another test module later imports under the same canonical name.

Canonical internal names are the dotted production import paths that other
tests import for real — `seestar`, `seestar.gui`, `seestar.gui.settings`,
`seestar.gui.histogram_widget`, `seestar.queuep.queue_manager`,
`seestar.alignment.astrometry_solver`, `seestar.enhancement.reproject_utils`,
etc.  A synthetic module installed under one of these names **persists for the
rest of the session** and is order-dependent: whichever module is collected
first "wins", and a later module that needs a real attribute (e.g.
`normalize_min_weight`, `_canonicalize_wcs_scale`, `TILE_HEIGHT`) breaks.

Non-canonical names (`seestar_solver_config`, `seestar_settings_manager`,
`seestar_analyzer_launch`, …) cannot collide with a production import path and
are therefore safe.  Third-party dependency stubs (`cv2`, `astroalign`,
`ccdproc`, `drizzle`) are also out of scope: they are not `seestar.*` and are
installed under `find_spec(...) is None` guards.

## Reproduction history

| Tree | Collection result | Notes |
|------|-------------------|-------|
| `9fe668d` (release 8.1.0) | **11 collection errors** | synthetic `seestar` package trees installed at module scope in the two poison-source tests |
| `3d363a4` (beta, `fix: harden drizzle photometric…`) | **12 collection errors** | same two sources, plus a `test_wb_basique_domain.py` cascade (the extra 12th) |
| `3d363a4` + 2 substantive repairs ("two-file tree") | clean; `1540 passed, 32 failed, 1 skipped` | see "Intermediate result" below |

The 12th collection error at beta was `test_wb_basique_domain.py`
(`from seestar.queuep.queue_manager import SeestarQueuedStacker` + `from
seestar.core.drizzle_background import …`) — a downstream casualty of the
partially-initialized `seestar.queuep.queue_manager` left behind by the
`test_preserve_linear_output` poison chain (below).

## Verified poison chains

### 1. `queue_manager` chain

`tests/test_preserve_linear_output.py` (pre-fix) built a hand-rolled fake
`seestar` package tree (`seestar.alignment.astrometry_solver`,
`seestar.enhancement.reproject_utils`, `seestar.enhancement.stack_enhancement`,
`seestar.enhancement.color_correction`, …) and then executed the **real**
`seestar/queuep/queue_manager.py` under the canonical name via
`spec.loader.exec_module(queue_manager)`.

As production evolved (`reproject_utils` gained `ensure_wcs_pixel_shape` and
friends), the synthetic tree went stale: `exec_module` raised mid-import and a
**partially-initialized** `seestar.queuep.queue_manager` stayed in
`sys.modules`.  Every later test doing
`from seestar.queuep.queue_manager import SeestarQueuedStacker` imported the
broken half-module → collection error cascade (including the `wb_basique`
cascade at beta).

### 2. `astrometry_solver` chain

`tests/test_load_wcs_ignore_missing_simple.py` (pre-fix) installed a synthetic
tree including `seestar.alignment.astrometry_solver` with only
`AstrometrySolver = object`, and `seestar.reproject_utils`,
`seestar.utils.wcs_utils`, etc.  That persisted and broke
`tests/test_reproject_zm_wcs_fix.py`, which imports
`_canonicalize_wcs_scale` from the real `astrometry_solver`.

## SYS.MODULES AUDIT

Audit pattern: every `sys.modules["seestar…"] = …` / `ModuleType("seestar…")`
occurrence across `tests/*.py` (module scope vs fixture scope classified with
an AST walk that stops at function/class boundaries).

### Category A — SAFE / LOCAL (non-canonical names, no fix needed) — 6 files

These install modules under underscore names that cannot collide with a
production `seestar.*` import:

| File | Key(s) |
|------|--------|
| `tests/test_analyzer_launch.py` | `seestar_analyzer_launch` |
| `tests/test_qt_settings_state.py` | `seestar_settings_manager_qt` |
| `tests/test_run_config.py` | `seestar_run_config`, `seestar_settings_manager` |
| `tests/test_solver_config.py` | `seestar_solver_config` |
| `tests/test_solver_gate.py` | `seestar_solver_gate_config` |
| `tests/test_solver_port.py` | `seestar_solver_port` |

### Category B — FIXTURE-SCOPED (monkeypatch / save-restore inside tests) — 10 files

These create fake `seestar.*` modules **inside test functions/fixtures** and
restore them (via `monkeypatch.setitem`, `_restore_module`, or a captured
`_saved` snapshot).  No persistent pollution:

`test_astap_wcs_padding.py`, `test_finalize_continuous_stack.py`,
`test_m3d_settings.py`, `test_queue_manager_reproject.py`,
`test_rewinsorization.py`, `test_rf2_production_impl.py`,
`test_rf2_production_seam.py`, `test_version_consistency.py`,
`test_zsss_final_save_truthfulness.py`, `test_zsss_startup_refusal_qm.py`.

### Category C — EXPLICITLY RESTORED at module scope — 2 files

These install a fake `seestar`/`seestar.gui`/`seestar.gui.settings`/
`seestar.gui.histogram_widget` at module scope, import `queue_manager`, then
**restore** the prior `sys.modules` entries (captured `_saved_sys_modules`).
No fake survives after import:

`test_reliability_source_immutability_r1.py`,
`test_reliability_stacked_restore_b.py`.

### Category D — UNSAFE FIXED (module-level persistent canonical fake → real import) — 27 files

These previously installed persistent canonical fakes at module scope
(`if "seestar.gui" not in sys.modules:` … `sys.modules["seestar.*"] = fake`).
The stub block was **removed**; the module now imports the real internal
modules directly.  Removed keys per file:

| File | Removed canonical keys |
|------|------------------------|
| `test_auto_stretch.py` | `seestar`, `seestar.gui`, `seestar.gui.settings`, `seestar.gui.histogram_widget` |
| `test_autotuner.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_boring_drizzle_boundary.py` | `seestar`, `seestar.gui`, `seestar.gui.settings`, `seestar.gui.histogram_widget` |
| `test_boring_thread.py` | `seestar`, `seestar.gui`, `seestar.gui.settings`, `seestar.gui.histogram_widget` |
| `test_hierarchical_stacking_integrity.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_hsi_closure_backend_parity.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_hsi_closure_ibn.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_hsi_closure_min_weight.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_hsi_closure_normalization.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_hsi_closure_rejection.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_incremental_reprojection.py` | `seestar`, `seestar.gui`, `seestar.gui.settings`, `seestar.gui.histogram_widget` (+ removed `reproject_utils` spec-reload) |
| `test_interbatch_background.py` | `seestar`, `seestar.gui`, `seestar.gui.settings`, `seestar.gui.histogram_widget` |
| `test_m3d_group_size_propagation.py` | `seestar`, `seestar.gui`, `seestar.gui.settings`, `seestar.gui.histogram_widget` |
| `test_m3d_policy.py` | `seestar`, `seestar.gui`, `seestar.gui.settings`, `seestar.gui.histogram_widget` |
| `test_preview_raw_linear_producer.py` | `seestar`, `seestar.gui`, `seestar.gui.settings`, `seestar.gui.histogram_widget` |
| `test_quality_executor_persistent.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_quality_executor_recreate.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_quality_fallback.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_quality_parallel.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_quality_pool_size.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_reliability_drizzle_scale_r2.py` | `seestar.gui.settings`, `seestar.gui.histogram_widget` |
| `test_reproject_mode_consistency.py` | `seestar`, `seestar.gui`, `seestar.gui.settings`, `seestar.gui.histogram_widget` |
| `test_resume.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_save_final_stack.py` | `seestar`, `seestar.gui`, `seestar.gui.settings`, `seestar.gui.histogram_widget` |
| `test_single_batch_csv.py` | `seestar`, `seestar.gui`, `seestar.gui.settings`, `seestar.gui.histogram_widget` |
| `test_threads.py` | `seestar`, `seestar.gui`, `seestar.gui.settings` |
| `test_worker_incremental_drizzle.py` | `seestar`, `seestar.gui`, `seestar.gui.settings`, `seestar.gui.histogram_widget` |

### Category E — SUBSTANTIVE REAL-IMPORT REPAIRS (preserved) — 2 files

The two original poison sources were already rewritten in the prior iteration
to import real internal modules; both are preserved in this iteration:

* `test_preserve_linear_output.py` — now `from seestar.queuep import
  queue_manager`; only the two file-writing ops are monkeypatched; added
  `finalization_mode = queue_manager.FINALIZATION_MODE_MOSAIC`.
* `test_load_wcs_ignore_missing_simple.py` — now
  `from seestar.gui.boring_stack import _load_wcs_header_only`; the synthetic
  `seestar` tree (incl. minimal `astrometry_solver`) is gone.

### Unresolved canonical pollution

**None.**  After this change there is no remaining module-scope, persistent
fake of a canonical internal `seestar.*` name anywhere in `tests/`.

## Repair strategy

1. Delete every module-scope `if "seestar.gui" not in sys.modules:` (and
   `if "seestar.gui.settings" not in sys.modules:`) stub block that installed
   fake `seestar`/`seestar.gui`/`seestar.gui.settings`/
   `seestar.gui.histogram_widget` modules, and let the module import the real
   internal modules (the `.venv` ships `matplotlib`, `opencv-python-headless`,
   `Pillow`, `tkinter`, … so the real GUI/settings/histogram modules import
   cleanly).
2. Keep the real-import repairs in the two substantive files (Category E).
3. Keep the guard-protected third-party stubs (`cv2`/`astroalign`/`ccdproc`/
   `drizzle`) — they are non-canonical and no-op when the dep is installed.
4. Leave Category B (fixture-scoped) and Category C (explicitly restored)
   unchanged: neither persists pollution beyond its scope.
5. Classify Category A (non-canonical underscore names) as safe/local.

## Validation

Fast gates only (full suite is the architect's gate, not run here):

* `pytest --collect-only -q` — **clean** twice: `1573 tests collected`, 0 errors
  (both runs).
* Order pair `test_preserve_linear_output.py` + `test_progress_callback.py`
  (both orders): identical — `2 passed, 1 failed` (the failure is the
  pre-existing `test_finalize_streaming_with_simple_callback`, listed in the
  baseline `pytest_full_20260828T093548Z.log`).
* Astrometry pair `test_astrometry_solver.py` + `test_reproject_zm_wcs_fix.py`
  (both orders): identical — `15 passed, 1 failed` (pre-existing
  `test_mode0_final_keeps_science_with_artifact_reference_header`).
* `test_load_wcs_ignore_missing_simple.py` + `test_reproject_zm_wcs_fix.py`
  (both orders): identical — `7 passed, 1 failed` (same pre-existing failure).
* Implicated files individually: `test_preserve_linear_output` (1 passed),
  `test_load_wcs_ignore_missing_simple` (1 passed),
  `test_reproject_zm_wcs_fix` (6 passed / 1 pre-existing failure),
  `test_progress_callback` (1 passed / 1 pre-existing failure),
  `test_wb_basique_domain` (5 passed).  All 27 Category-D files were also run
  individually: all pass or hit only pre-existing baseline failures/skips.
* `git diff --check` — clean (exit 0).
* `git diff HEAD -- seestar/` — empty (no production change).

## Intermediate result (not the final tree)

`pytest_full_20260828T093548Z.log` (`1540 passed, 32 failed, 1 skipped`) was
launched against the **intermediate two-file tree** — `3d363a4` plus only the
two substantive real-import repairs (`test_preserve_linear_output.py`,
`test_load_wcs_ignore_missing_simple.py`) — **before** the additional 25 stub
edits were applied.  It is evidence for that two-file tree, **not** for this
final tree.  The final-tree result is for the architect's full-suite run.

## Final full-suite result

The exact final test-only tree was run in one normal Python process:

```text
pytest_full_isolation_20260828T083627Z.log
11 failed, 1561 passed, 1 skipped, 65 warnings in 521.49s
PYTEST_EXIT=1
```

Collection completed normally and all 1573 tests executed.  There were no
collection errors and no import-order cascade.  The same eleven failures were
then run against the clean detached 8.1.0 worktree at `9fe668d`; all eleven
reproduced there (`pytest_stable_9fe668d_failure_repro_20260828T0849Z.log`).
They therefore predate both this isolation repair and the Drizzle beta commit.

### Remaining failure classification

* **Environment-specific fixture drift (1):**
  `test_m16_scale_witness.py::test_full_witness_scale_within_noise` expects 19
  aligned files, while the external `/home/tristan/M16/quick` fixture now
  contains enough duplicate inputs to produce 32.
* **Historical test expectations / stale lightweight fixtures (2):**
  `test_progress_callback.py::test_finalize_streaming_with_simple_callback`
  constructs a `SimpleNamespace` without the now-required
  `_get_final_match_background`; and
  `test_save_final_stack.py::test_save_final_stack_radec_from_reference_header`
  constructs a `Dummy` without `logger`, aborting save before
  `final_stacked_path` is assigned.
* **Historical real production defects (8):** seven tests in
  `test_reproject_utils.py` expose runtime references to undefined
  `_estimate_mem_gb` and `ReprojectCoaddResult` in
  `seestar.enhancement.reproject_utils` (including the Boring-stack import
  path), and
  `test_reproject_zm_wcs_fix.py::test_mode0_final_keeps_science_with_artifact_reference_header`
  shows `_reproject_classic_batches_zm` returning `False` for the historical
  artifact-header witness.  These are independently present at `9fe668d` and
  require separate product/test-debt missions; they were deliberately not
  changed here.

No failure was hidden, skipped, xfailed, reordered, or repaired outside the
test-isolation scope.

## Closure verdict

**ACCEPT for PYISO-01.**  The historical collection blockade is removed, the
complete suite now coexists and executes in one process, representative orders
are invariant, the accepted Drizzle targeted gate remains unchanged, and the
production diff is empty.  The eleven genuine historical failures above remain
open and explicitly classified for follow-up.
