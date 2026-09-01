# COV-06B Independent Review — Nono

**Repository:** /home/tristan/.openclaw/workspace/projects/zeseestarstacker
**Branch:** feature/coverage-aware-peripheral-reconstruction
**Reviewed HEAD:** b0134f37c287125ee8a71aeba636be9381ffff1e
**COV-06B starting SHA:** 7d9306589998fa58e550f0167f4cb4b89ba95eb0
**COV-06B implementation SHA:** adb98eeffb25291a6c7bdc2f60f7e148144faca8
**Scientific baseline:** 501eb9b5031a1ea81f09e6f01687a4cc349de879 (== origin/main == origin/beta)
**Date:** 2026-09-01
**Reviewer:** Nono (independent; source, commits, diffs, and tests inspected directly; implementer report audited as evidence, not truth)

## VERDICT: ACCEPT

All ten COV-06B objectives verified. All eighteen mandatory questions answered with
source/line/test evidence. No COV-06B regression found. The six failing tests observed
at HEAD are known baseline failures, each reproduced unchanged at the starting SHA
7d93065 in an independent temp extraction. The diff is tightly scoped (19 files,
+682/-58), `git diff --check` is clean, and repository status is fully explained.

---

## Independent commands and results

Environment: project venv `.venv/bin/python` (Python 3.13, pytest 8.3.5, astropy 7.0.1,
numpy 2.2.4); Qt tests run offscreen (`QT_QPA_PLATFORM=offscreen`).

| Command | Result |
|---|---|
| `git rev-parse HEAD` | b0134f37c287125ee8a71aeba636be9381ffff1e |
| `git branch --show-current` | feature/coverage-aware-peripheral-reconstruction |
| `git rev-parse origin/main origin/beta` | 501eb9b5... / 501eb9b5... (baseline, unchanged) |
| `git diff --stat 7d93065..HEAD` | 19 files, 682 insertions(+), 58 deletions(-) |
| `git diff --check 7d93065..HEAD` | CLEAN (no output) |
| `git log 7d93065..HEAD` | adb98ee fix(COV-06B), b0134f3 docs(COV-06B) — exactly 2 commits |
| `python -m compileall -q seestar/` | OK |
| pytest: cov06b render-ab + settings-migration + coverage-render + run-config | 32 passed |
| pytest: qt_expert_m15 + qt_settings_surface + qt_backend_runner | 45 passed |
| pytest: coverage support/taper/overlap/classic/reproject + resume | 247 passed, 4 failed (resume baseline, see below) |
| pytest: save_final_stack radec + boring_drizzle_boundary | 2 failed (baseline, see below) |
| pytest: HSI closure (6 files) + drizzle signed-weights + drizzle finalize | 129 passed |
| pytest: winsorized_sigma + rewinsorization + boring_thread + boring_thread_regex + coverage_drizzle | 23 passed |
| pytest: full COV/Qt/settings/backend aggregate (11 files) | 145 passed |
| Baseline repro at 7d93065 (temp `git archive` extraction) | 4 resume failures + 2 others: IDENTICAL set and causes |
| Screenshot `review/COV06B_qt_expert.png` | exists, 631,006 B, SHA256 ac903a9e... matches report claim |

### Baseline-failure reproduction (independent)

Failing at HEAD and reproduced **unchanged** at starting SHA 7d93065 (extracted to
`/tmp/zsss_cov06b_start`, run with the same project venv):

1. `test_resume.py::test_start_processing_valid_resume_pins_reference_and_queue`
2. `test_resume.py::test_start_processing_quality_weighted_captures_q_ref_once_fresh`
3. `test_resume.py::test_start_processing_quality_weighted_resume_skips_recomputation`
4. `test_resume.py::test_repeated_start_binds_fresh_canonical_config`
5. `test_save_final_stack.py::test_save_final_stack_radec_from_reference_header`
   (test dummy lacks `logger` -> AttributeError, then astropy `Empty filename`)
6. `test_boring_drizzle_boundary.py::test_classic_memmaps_use_fixed_reference_grid_shape`

`test_resume.py` is untouched in the COV-06B range (verified by
`git diff --name-only 7d93065..HEAD | grep resume` -> empty). These are therefore
pre-existing baseline failures, correctly distinguished from regressions (objective 10,
N15).

---

## Mandatory questions (N1-N18)

**N1. Is legacy inverse-WHT Feathering OFF after migration of an old config? - YES**
`seestar/settings_migration.py:39-47`: `migrate_settings_data` forces
`migrated["apply_feathering"] = False` for every schema < 1 mapping and stamps
`settings_schema_version = 1`. Wired into both loaders: Tk
`seestar/gui/settings.py:2690` (`settings_data, settings_migrated = migrate_settings_data(...)`,
persisted via `self.save_settings()` when `settings_migrated`), Qt
`seestar/gui_qt/main_window.py:4784-4789` (migration then immediate
`save_settings_json` when changed). Witness tests:
`tests/test_cov06b_settings_migration.py::test_old_apply_feathering_true_migrates_to_safe_state`,
`test_tk_loader_migrates_and_records_schema_once`,
`tests/test_qt_expert_m15.py::test_cov06b_qt_old_config_migrates_once_and_round_trips`
— all passed in my run.

**N2. Can an old apply_feathering=true silently reactivate the deprecated path? - NO**
The value-level migration runs on every settings load in both shells
(Tk settings.py:2690; Qt main_window.py:4784) and forces the flag OFF before any state is
built (`QtSettingsState.from_dict(data)` at main_window.py:4791 runs on the migrated
data). Backend default is also safe: `start_processing(..., apply_feathering=False)`
(queue_manager.py:19795) and the render/feather gates use
`hasattr(...) and self.apply_feathering` (queue_manager.py:5143, 18461-18462). The
file-layer `migrate_legacy_settings` (settings_persistence.py:139-152) copies legacy JSON
verbatim, but value-level migration always runs at the subsequent load — verified by the
Qt round-trip test. A schema-v1 file with `apply_feathering=true` can only be produced by
deliberate manual edit (migration never downgrades future schemas by design); that is not
a silent reactivation path.

**N3. Is the deprecated legacy control removed/clearly isolated from normal GUI? - YES**
Qt normal Expert surface: `apply_feathering` and `feather_blur_px` are removed from
`SETTINGS_SECTIONS` (main_window.py:479-486), `LOCALIZED_SETTINGS_FIELD_KEYS`
(main_window.py:587-588), `EXPERT_ENABLER_GATES` (main_window.py:642),
`EXPERT_RESET_ATTRS` (main_window.py:681-682). The section is renamed
`Coverage / Edge Reconstruction`. The Tk compatibility shell retains the control but
only under explicit "Legacy / Deprecated inverse-WHT Feathering" labels and tooltips
(localization/en.py:145-148, 388-395; fr.py:146-152, 390-395) with default OFF
(settings.py:1354) — consistent with objective 9.

**N4. Does "Batch feathering" terminology now reflect its real current semantics? - YES**
Qt label: `_field("apply_batch_feathering", "Coverage support taper", "bool")`
(main_window.py:482); localized
`field_apply_batch_feathering` = "Coverage support taper" /
"Adoucissement du support de couverture" (gui_qt/localization.py:362-366) and Tk
`feather_inter_batch_label` / tooltips explicitly state the reducer-specific mathematics
is unchanged (localization/en.py:150, 395). The implementation docs
(COV_06_REWORK_REPORT.md, docs/coverage_confidence_contract.md) record that
`apply_batch_feathering` gates the COV footprint taper in mean/support paths and keeps
established radial behavior in other reducers; no math was changed (see N5/N13). Only an
internal docstring `_feather_batch_coverage` (queue_manager.py:12805) still says "radial
batch feathering" — informational only, not user-visible; documented as a separate
cleanup candidate.

**N5. Is the validated footprint support taper still active and unchanged mathematically? - YES**
`apply_batch_feathering` default True in Tk (settings.py:1356), Qt
(settings_state.py:261) and backend (queue_manager.py:19794, 20135). It still gates the
COV footprint taper at queue_manager.py:11893, 12806 (radial weight map), 13347, 13741.
The COV-06B diff touches queue_manager only to add the `apply_coverage_render` keyword
(+2 lines); no taper/support math was modified. `tests/test_coverage_taper.py` (264
lines, part of the 247-pass group) and the full coverage-support suites pass.

**N6. Is coverage-aware final reconstruction visible in Qt? - YES**
`_field("apply_coverage_render", "Coverage-aware final reconstruction", "bool")`
(main_window.py:483) under section "Coverage / Edge Reconstruction"; localized
`field_apply_coverage_render` (gui_qt/localization.py:367-369). Verified live by
`tests/test_qt_expert_m15.py::test_cov06b_modern_coverage_surface_and_backend_wiring`
(group title present, QCheckBox present, unchecked by default) and by the offscreen
screenshot `review/COV06B_qt_expert.png` (SHA256 matches the report).

**N7. Does its setting persist correctly across save/reload? - YES**
`QtSettingsState.apply_coverage_render` (settings_state.py:262) round-trips through
`to_dict()`/`from_dict` (settings_state.py:364-369 includes every non-transient field);
Qt save writes it (main_window.py:4808-4812) with the schema marker; Tk save writes it
(settings.py:2567-2569). Witness tests:
`test_cov06b_settings_migration.py::test_coverage_render_state_round_trip` and
`test_qt_expert_m15.py::test_cov06b_qt_old_config_migrates_once_and_round_trips`
(second MainWindow load restores apply_coverage_render=True; file byte-value stable) —
both passed.

**N8. Does the setting actually propagate to the backend? - YES**
Complete chain verified: QCheckBox -> `collect_settings_state` -> `build_run_request` ->
`build_backend_kwargs` (`"apply_coverage_render": getattr(settings, "apply_coverage_render", False)`,
run_config.py:249) -> `start_processing` keyword-only param default False
(queue_manager.py:19812) -> `self.apply_coverage_render = bool(apply_coverage_render)`
(queue_manager.py:20136) -> final render gate (queue_manager.py:18499-18515). Verified by
`test_qt_backend_runner.py::test_seestar_backend_maps_request_to_stackers`
(`stacker.start_kwargs["apply_coverage_render"] is True`) and the M15 wiring test
(`request.backend_kwargs["apply_coverage_render"] is True`) — both passed.

**N9. Is coverage render OFF by default? - YES**
Tk defaults (settings.py:1357), Qt state (settings_state.py:262), backend signature
default (queue_manager.py:19812), and the defensive read in build_backend_kwargs
(run_config.py:249) are all False. `test_fresh_installation_has_safe_cov_defaults`
passed.

**N10. Does render ON leave scientific FITS data unchanged? - YES**
In `_save_final_stack`, the coverage render is applied only to
`data_after_postproc` — the cosmetic preview branch (queue_manager.py:18506-18515,
`self.last_saved_data_for_preview = data_after_postproc.copy()` at 18526, PNG at 18861).
The scientific FITS primary HDU is written from `raw_adu_data_for_ui_histogram`
(queue_manager.py:18710/18757, 18814-18819), and the optional WHT companion from
`final_wht_hwc` (18809-18813) — never from the rendered buffer.
`tests/test_cov06b_render_ab.py::test_render_on_changes_only_cosmetic_product` asserts
exact SCI array equality, exact full-header byte equality, and that only the preview
product differs — passed.

**N11. Are SCI/WHT/support/WCS unaffected by render ON/OFF? - YES**
Same A/B witness asserts input SCI, WHT, SUP_W1 and SUP_W2 arrays are exactly unchanged
(`np.array_equal` before/after for both runs) and that the eight WCS cards
(CTYPE/CRPIX/CRVAL/CDELT) are preserved in the final header; also
`test_drizzle_render_confidence_uses_positive_support_not_signed_wht` shows the render
confidence path never consults a signed native WHT. Passed.

**N12. Is Drizzle positive support still separate from signed native WHT? - YES**
`_derive_neff_support_for_render` (queue_manager.py:11918-11943) uses only the positive
Classic `coverage_sup_w1/w2` memmaps (N_eff = (w1/sqrt(max(w2,0)))^2, clamped >= 0) or the
Drizzle positive-support accumulator (`_drizzle_support_n_eff`); the signed native
Lanczos WHT is never consulted. The A/B test constructs a deliberately signed
`native_wht` and asserts it is never used. No SUP_W1/SUP_W2 math changed in COV-06B
(objective 8). Passed.

**N13. Has no reducer/HSI/resume mathematical contract been unintentionally changed? - YES**
The COV-06B diff touches only: run_config.py (1 line), settings.py, new
settings_migration.py, gui_qt (localization/main_window/settings_persistence/settings_state),
localization en/fr, queue_manager (+2 lines: keyword param + attribute assignment), tests,
and docs. No reducer files, no HSI files, no registration/GAR code, no scientific
SUM/WHT/Drizzle-WHT math, no resume math were modified. `apply_coverage_render` is
deliberately absent from the run-contract fingerprint (grep of run_contract.py -> no hit),
consistent with its purely cosmetic role. HSI closure suites (129 passed) and
coverage/drizzle suites (145 passed) confirm.

**N14. Is the benchmark scaling now arithmetically correct? - YES**
COV_06_REWORK_REPORT.md: `309 ms/exposure -> 309 s/1k (~5.15 min), 3,090 s/10k
(~51.5 min), 30,900 s/100k (~8.58 h)`. 309 ms x 1,000 = 309,000 ms = 309 s ✓;
x 10,000 = 3,090 s ✓; x 100,000 = 30,900 s ✓; conversions to minutes/hours ✓. The
erroneous x1000-downscaling of the previous report is corrected and the correction is
explicitly documented.

**N15. Are known baseline failures distinguished from regressions? - YES**
Six failures at HEAD, all reproduced unchanged at starting SHA 7d93065 in an independent
temp extraction (see table above): 4 x test_resume.py (untouched by COV-06B), 1 x
test_save_final_stack (dummy lacks `logger`), 1 x test_boring_drizzle_boundary (obsolete
source-text shape assertion). The implementer report documents the same six with matching
causes — I verified the causes directly (e.g., `AttributeError: 'Dummy' object has no
attribute 'logger'`). Zero COV-06B-specific failures across all my runs.

**N16. Is the final diff tightly scoped to COV-06B? - YES**
`git diff --stat 7d93065..HEAD` = 19 files, all directly serving the ten objectives:
migration module + both loaders, Qt Expert surface + localization + persistence,
backend keyword plumbing, two new witness test files, updates to 4 existing tests, and
the implementation/benchmark docs. The second commit (b0134f3) is docs-only
(COV_06B_IMPLEMENTATION_REPORT.md, +205).

**N17. Is git diff --check clean? - YES**
`git diff --check 7d93065..HEAD` returned no output (exit 0). `compileall` also clean.

**N18. Is repository status fully explained? - YES**
`git status`: branch `feature/coverage-aware-peripheral-reconstruction` at HEAD
b0134f3, with only three untracked files, all pre-existing and preserved:
`COV_NONO_REVIEW.md` (previous review artifact, dated before this audit),
`FETCH_HEAD` (empty), `main` (empty) — the last two are empty stray files predating
COV-06B (timestamps 30 Aug). origin/main and origin/beta remain at the baseline
501eb9b. No stray tracked modifications.

---

## Findings by severity

**Critical / High:** none.

**Medium:** none.

**Low / informational:**
1. Internal docstring `_feather_batch_coverage` (queue_manager.py:12805) still uses the
   words "radial batch feathering". Not user-visible (GUI labels/tooltips are corrected),
   math unchanged; the docs explicitly list this legacy radial behavior as a separate
   future cleanup candidate. No action required for COV-06B.
2. `migrate_legacy_settings` (settings_persistence.py) copies legacy JSON verbatim at
   the file layer; safety relies on the value-level migration at load. This is
   intentional (file layer does not filter) and fully covered by tests; noted for
   future readers.
3. Tk shell still exposes the legacy inverse-WHT control (labeled Legacy/Deprecated,
   default OFF). Deliberate per objective 9 (Tk/non-Qt compatibility); Qt normal surface
   is clean. Product-owner visibility only.

## Repository status (final)

- HEAD: b0134f37c287125ee8a71aeba636be9381ffff1e (branch
  feature/coverage-aware-peripheral-reconstruction), 2 commits ahead of starting SHA.
- Tracked worktree: clean; `git diff --check` clean; compileall OK.
- Untracked: COV_NONO_REVIEW.md, FETCH_HEAD, main (pre-existing, preserved, explained).
- No push/merge/tag/release/reset performed; no other files created or modified.

## Conclusion

COV-06B closes the coverage-reconstruction product loop correctly: versioned one-time
migration retires persisted legacy inverse-WHT Feathering in both shells, the deprecated
control is absent from the normal Qt surface, terminology now matches semantics,
coverage-aware final reconstruction is visible in Qt (default OFF), persists, and
propagates end-to-end to the backend, the render is provably cosmetic-only
(A/B exact-equality witness), Drizzle positive support stays distinct from signed native
WHT, no scientific contract was touched, the benchmark arithmetic is corrected, and the
six failing tests are all verified pre-existing baseline failures. Verdict: **ACCEPT**.
