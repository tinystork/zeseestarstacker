# Coverage-Aware Reconstruction — Legacy Edge-Treatment Cleanup (COV-05)

<!-- project: path:/home/tristan/.openclaw/workspace/projects/zeseestarstacker -->

Gate: **COV-05** — classify (do not blindly delete) the historical edge/coverage
paths, and remove only genuinely DEAD code after evidence.

## Classification vocabulary

* **ACTIVE_VALID** — live on a user-facing path today.
* **ACTIVE_OBSOLETE_REPLACED** — still live but superseded by the COV coverage
  architecture on the path it was introduced for (removal candidate only after
  product-owner validation, since it changes default cosmetic behaviour).
* **COMPATIBILITY_ONLY** — legacy surface retained for compatibility; still
  called, so not deletable.
* **DEAD** — no live caller.
* **UNKNOWN** — liveness not conclusively established.

## Result (call-site evidence)

| Candidate | Classification | Evidence
| --- | --- | --- |
| `feather_by_weight_map` (WHT-derived gain [0.5, 2.0]) | ACTIVE_VALID (hazard) | callers `queue_manager.py:5147` (preview) and `18439` (`_save_final_stack`) |
| `apply_low_wht_mask` (median-fill) | ACTIVE_VALID | GUI-exposed (`settings.py`, `run_contract.py`, `main_window.py`); `queue_manager.py:5162/18458` |
| `_feather_batch_coverage` (radial) | ACTIVE_VALID | 4 remaining reducer call sites `queue_manager.py:13585/13622/13656/13688` (winsor/kappa/linear-fit/median); mean path already migrated to the COV-02 footprint taper |
| `make_radial_weight_map` | ACTIVE_VALID | reproject (`1370`), IBN feather (`3302`), mosaic, reducer coverage, single-image |
| `_wait_drizzle_processes` | COMPATIBILITY_ONLY | "M3-D legacy no-op" but still called in `_cleanup_stacker` (`17832`) and `boring_stack.py:104`; many tests pin it |
| `drizzle_utils.drizzle_finalize` | COMPATIBILITY_ONLY | still used by incremental-drizzle preview (`queue_manager.py:11036`) and `livestack_mode.py:352` |
| `ccdproc_combine` import (`queue_manager.py:266`) | UNKNOWN | import present, no live reducer caller found |
| `simple_stacker` / `create_master_tile_simple` | COMPATIBILITY_ONLY | mean-only simplified path retained; no live primary-path caller found |

## Removals performed

None.  No candidate was proven **DEAD** with a live-caller search: every flagged
symbol still has at least one caller or a pinned compatibility contract.

## Deferred (product-owner gate)

* `feather_by_weight_map` is the one candidate whose WHT-derived **gain**
  contradicts the frozen invariant "low WHT is never a reason for brightness
  gain".  It is cosmetic-only today.  COV-04's `coverage_aware_render` is the
  gain-free replacement, but removing/neutralising `feather_by_weight_map` would
  change the long-standing default preview, so it is left ACTIVE and flagged
  for an explicit product decision (per mission §22).
* `_feather_batch_coverage` radial for the rejection/median reducers remains a
  candidate for the same footprint-taper migration done for mean in COV-02; it
  cancels for single-grid classic and does not alter the order-statistics
  estimator, so it is left ACTIVE and documented.

All user-facing stacking modes (Boring, Mean, Median, Kappa-sigma, Winsorized
sigma, Linear-fit clip, Reproject, Drizzle) are preserved.
