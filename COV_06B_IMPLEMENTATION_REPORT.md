# COV-06B Implementation Report

## Provenance

- Branch: `feature/coverage-aware-peripheral-reconstruction`
- Starting full SHA: `7d9306589998fa58e550f0167f4cb4b89ba95eb0`
- Ending implementation SHA: `adb98eeffb25291a6c7bdc2f60f7e148144faca8`
- Original scientific baseline: `501eb9b5031a1ea81f09e6f01687a4cc349de879`
- `origin/main`: `501eb9b5031a1ea81f09e6f01687a4cc349de879`
- `origin/beta`: `501eb9b5031a1ea81f09e6f01687a4cc349de879`

Implementation commit:

```text
adb98ee fix(COV-06B): close coverage controls and legacy migration
```

This report is committed after the implementation SHA so that the recorded
implementation endpoint is immutable and auditable.

## Legacy migration design

`seestar/settings_migration.py` introduces the smallest shared versioned
migration mechanism:

- schema marker: `settings_schema_version`
- current schema: `1`
- a pre-versioned or malformed legacy schema is copied, has
  `apply_feathering` forced to `false`, and receives schema version 1;
- schema 1 reloads are not mutated;
- future schemas are preserved and never downgraded.

Both Qt and Tk load through this migration.  When migration occurs, the
settings are immediately saved on the normal settings path.  Every normal Qt
or Tk save records schema version 1.

Old-settings witness:

```json
{"apply_feathering": true, "batch_size": 7}
```

loads and is persisted as:

```json
{"apply_feathering": false, "batch_size": 7, "settings_schema_version": 1}
```

The second load leaves the saved mapping byte-for-value stable.  The Tk
compatibility loader has the same once-only witness.

Fresh/modern witness:

- legacy inverse-WHT Feathering: OFF
- coverage support taper: ON
- coverage-aware final reconstruction: OFF
- coverage-render enable/save/reload: remains ON

The deprecated inverse-WHT controls are absent from the normal Qt Expert
surface.  The historical Tk compatibility surface retains them only with
explicit **Legacy / Deprecated** labels and tooltips.

## GUI and backend wiring

Final Qt Expert section:

```text
Coverage / Edge Reconstruction
  Coverage support taper                         ON
  Coverage-aware final reconstruction            OFF
  Low-weight mask                                unchanged default (OFF)
  Low-weight percentile                          unchanged
  Low-weight soften                              unchanged
```

`apply_batch_feathering` remains the backward-compatible serialized key.
Inspection showed that it gates the validated COV footprint taper in the
mean/positive-support path and retains established radial behavior in other
reducers.  COV-06B therefore changes the product label only; it does not alter
any reducer-specific mathematics.

Complete render-control path:

```text
QCheckBox
 -> QtSettingsState.apply_coverage_render
 -> JSON save/reload
 -> build_backend_kwargs()
 -> RunRequest
 -> SeestarBackend
 -> SeestarQueuedStacker.start_processing(keyword-only, default False)
 -> self.apply_coverage_render
 -> downstream final preview/PNG renderer
```

The setting is intentionally absent from the scientific run-contract
fingerprint because it is a downstream cosmetic choice.

GUI evidence was generated from the real offscreen Qt `MainWindow`, with the
Expert tab and the new control visible:

```text
/home/tristan/.openclaw/workspace/review/COV06B_qt_expert.png
SHA256 ac903a9ef0defc2d947a69b94a56c47860cfa5ad6220816d0c64cd1ca1275a0d
```

## Render OFF/ON scientific equivalence

`tests/test_cov06b_render_ab.py` performs deterministic OFF and ON finalizer
runs from identical SCI, WHT, support and WCS inputs.

Evidence:

- final FITS SCI arrays are exactly equal (`numpy.array_equal`);
- complete final FITS headers are byte-string equal;
- input SCI and WHT arrays remain exactly unchanged;
- SUP_W1 and SUP_W2 remain exactly unchanged;
- WCS cards remain unchanged;
- the OFF preview and ON preview differ;
- Drizzle render confidence is read from positive Drizzle support and not from
  a deliberately signed native WHT.

The renderer remains downstream of scientific reconstruction.

## Benchmark correction and performance scope

`COV_06_REWORK_REPORT.md` now reports the arithmetic honestly:

- 309 ms/exposure x 1,000 = 309 s = 5.15 min
- 309 ms/exposure x 10,000 = 3,090 s = 51.5 min
- 309 ms/exposure x 100,000 = 30,900 s = 8.58 h

No EDT optimization or registration/support redesign was performed.  A future
candidate is a precomputed source-space taper transported with registration
geometry, subject to its own scientific-equivalence gate.

The report also preserves the real Linux witness as a 160-entry, 54-batch,
`winsorized-sigma-clip` success with WHT approximately 0.60--158.95.  It
explicitly records that legacy Feathering was ON and duplicated filenames were
present.  The run is therefore smoke/stress evidence, not final-render efficacy
proof or proof of 160 independent noise realizations.  The term remains
**support effective exposure count**.

## Tests executed

Final COV/Qt/settings/backend group:

```text
290 passed, 1 skipped, 0 failed
```

Established HSI/COV focused group:

```text
150 passed, 0 failed
```

Adjacent scientific FITS, winsorized/mean smoke, Boring path, Drizzle
confidence and resume-sensitive group:

```text
150 passed, 2 failed
```

The two failures were reproduced unchanged at the starting SHA:

- `tests/test_save_final_stack.py::test_save_final_stack_radec_from_reference_header`
  (test dummy lacks `logger`)
- `tests/test_boring_drizzle_boundary.py::test_classic_memmaps_use_fixed_reference_grid_shape`
  (obsolete source-text assertion for `shape=(H,W)`)

Full `tests/test_resume.py`:

```text
149 passed, 4 failed
```

All four failures were reproduced unchanged at the starting SHA:

- `test_start_processing_valid_resume_pins_reference_and_queue`
- `test_start_processing_quality_weighted_captures_q_ref_once_fresh`
- `test_start_processing_quality_weighted_resume_skips_recomputation`
- `test_repeated_start_binds_fresh_canonical_config`

Mission aggregate, without double-counting focused reruns:

```text
739 passed, 1 skipped, 6 known starting-HEAD failures, 0 COV-06B regressions
```

An initial invocation through the system `python` produced seven collection
errors because that interpreter lacked PySide6/OpenCV.  The repository
`.venv/bin/python` invocation is the valid project environment and produced
the results above; the collection errors are not counted as product tests.

`python -m compileall` (project venv) completed successfully.

## Final implementation-tree checks

- `git diff --check`: clean
- tracked worktree: clean after implementation commit
- preserved pre-existing untracked files:
  `COV_NONO_REVIEW.md`, `FETCH_HEAD`, `main`
- no push, merge, tag, release, version bump or remote update performed
- Coco was not reset, contacted, or used
