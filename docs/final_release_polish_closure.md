# Final Release Polish — Closure Report

**Product:** ZeSeestarStacker
**Baseline:** `22e86238f4458ca07cfdee052fdf25009b62caae` (origin/beta)
**Release branch:** `fix/final-release-polish`
**Final HEAD:** `3092000` (see commit list below)
**Previous version:** 8.2.2 — Phoenix consedit
**New version:** 8.2.3 — Phoenix consedit

---

## Summary

The final release polish closed four bounded workstreams. The scientific engine
is **bit / numerically unchanged**; only presentation defects and dead code were
addressed.

| Workstream | Scope | Verdict |
|---|---|---|
| FRP-H1 | histogram full-tail / viewport / lifecycle | ACCEPT (Nono) |
| FRP-D1 | retire abandoned LiveStack mode | ACCEPT (Nono) |
| FRP-L1 | light-theme readability | ACCEPT (Nono) |
| FRP-R2 | version bump 8.2.2 → 8.2.3 | ACCEPT (final Nono gate) |

Commits (oldest → newest, all on `fix/final-release-polish`, no push):

```
aa88798 fix(histogram): preserve full HDR tail distribution and semantic view state
eaa7a0a fix(histogram): clean default window on clear and restore marker coverage
801a04c refactor: remove abandoned livestack mode
22bc07e fix(gui): restore light-theme readability via complete disabled palette
3092000 chore: bump ZeSeestarStacker release version to 8.2.3
```

---

## FRP-H1 — Histogram

### Root cause
The histogram model (`compute_histogram_float`) binned 512 bins only over a
robust plot high `bin_hi` (so a sparse extreme HDR tail never stretched the bins
into a few spikes). Values above `bin_hi` were counted as `overflow` but their
**distribution** was not binned. `reset_histogram_view`/`reset_zoom` widened the
X axis to the full analysis range but the bars still only existed in `bin_range`
— "Reset/Full" did not actually show the full distribution. The view state was
ad-hoc (`auto_zoom_enabled` bool + `_frozen_range`), so a Full/Reset choice was
transient (a new model silently fell back to the default window).

### After
- **Dual-domain histogram** — `compute_histogram_float` now also produces
  `full_counts`/`full_log_counts`/`full_hist_range` (512 bins over the true
  analysis maximum `(0, upper)`), from the **exact same** in-domain sample (no
  second image traversal).
- **Persistent view state** — `HistogramView` drives behaviour from an explicit
  `_view_mode` (`default`/`auto`/`full`/`manual`): AUTO re-zooms to each new
  model's robust range; FULL (set by Reset/Full) persists across new models and
  paints the full-domain bars; MANUAL preserves a frozen zoom (reconciled/clamped
  on domain shrink, never silently returning to AUTO or `[0,1]`).
- **End-of-run lifecycle** — verified and documented (`_on_run_finished`
  docstring): the final visible histogram represents the **last accepted live
  preview**, never the saved FITS (no FITS readback / new disk I/O).

### Count-conservation evidence
Per channel: `sum(counts) + overflow == sampled_count` AND
`sum(full_counts) == sampled_count`; per-channel `stats.max` stays the true
analysis max; dense high-end populations make `full_counts == counts`
(degenerate-identical). Covered by `tests/test_phi_full_histogram_frp_h1.py`.

---

## FRP-D1 — LiveStack retirement

### Reachability proof
`seestar/queuep/livestack_mode.py` (`LiveStackController` + `compute_snr`/
`stretch_01`/`save_png16`/`start_livestack_cli`) had **zero** production
importers (no `import livestack`, no `LiveStackController` usage outside the
file). The `_FakeLiveStacker` classes in `test_qt_preview_reconcile_m24.py` and
`test_qt_res_live_m22.py` are self-contained active-stacker test doubles (not
imports). No tracked entry point or packaging file referenced the module.

### Files removed
- `seestar/queuep/livestack_mode.py` (392 lines) — the only deletion.

### Intentionally retained shared helpers (LIVE elsewhere)
- `seestar/core/drizzle_utils.py::drizzle_finalize` — still unit-tested and
  referenced by docs/examples (production-orphaned after this deletion, but
  retained per mission §14 conservative dead-code policy).
- `seestar/core/incremental_reprojection.py` (`reproject_and_combine`,
  `reproject_and_coadd_batch`, etc.) — exported in `seestar/core/__init__.py`
  and exercised by dedicated tests.

Historical archaeology docs retain their livestack references (per mission §13).

---

## FRP-L1 — Light theme readability

### Root cause
The theme is QPalette-only (`_dark_palette()` / `_light_palette()` in
`main_window.py`). Both palettes set the Active group's core roles but omitted
the **Disabled** color group's `WindowText` (and `Base`/`Button`/`Highlight`/
`HighlightedText`), so disabled labels/group-box titles/tab labels/status text
rendered full-contrast (black in light, white in dark) — indistinguishable from
enabled, and on native dark platforms unset roles could leak dark-theme colors
into the light theme.

### Files changed
- `seestar/gui_qt/main_window.py` — new `_complete_disabled_group()` helper
  wired into both palettes: Disabled text roles (`WindowText`/`Text`/
  `ButtonText`/`PlaceholderText`) use the dimmed `disabled_text` gray; Disabled
  structural roles mirror their Active values. Every Active-group color is
  byte-identical.
- `tests/test_qt_system_tab_m255c.py` — 3 new palette tests (dimming + contrast).

### Dark / Light / System validation
- Light: Disabled text dimmed to `#7f7f7f`, contrast ≥ 3.51:1 vs `Window`,
  4.00:1 vs `Base` — the defect removed. Active unchanged.
- Dark: Active unchanged; Disabled text dimmed (shared bug fix, allowed by §17).
- System: unchanged path (`standardPalette()`); existing restore test green.
- Histogram surface left deliberately dark and readable (not forced light).

---

## Tests

### Targeted (FRP areas, all green)
- Histogram / preview / OTPUX / theme / shared-helper set:
  **304 passed** (Nono final-gate run), **214 passed** (histogram-focused),
  **94 passed** (theme/Qt), **29 passed** (drizzle-helper + LiveStack doubles),
  **4 passed** (version consistency).

### Broad / full suite
- **2274 collected → 2242 passed, 30 failed, 2 skipped.**
- The **30 failures reproduce byte-identically at baseline `22e86238`** (verified
  via a clean baseline worktree; Nono independently spot-checked a 14-failure
  subset and confirmed byte-identical FAILED lists). All 30 are pre-existing
  environment/baseline failures (resume, reproject-streaming, zesolver
  discovery, solver-gate, Drizzle boundary/exposure-metadata, M16 witness,
  progress callback, reliability-drizzle-scale/source-immutability) — **none in
  FRP-touched files, zero FRP regressions.**

### Scientific non-regression
The diff touches only Qt display modules (`histogram_view.py`, `main_window.py`
palette + docstring, `preview_analysis.py` histogram metadata), the orphaned
`livestack_mode.py`, tests, and version/CHANGELOG. No change to SUM/WHT/N_eff/
COV/registration/Drizzle/Classic/HSI/Resume/final-FITS paths, Auto Stretch/WB
math, or `preview_adjust.py`. Scientific behavior is BIT / NUMERICALLY UNCHANGED.

### Real witness
The previously completed real Drizzle ×3 witness (80 images, Winsorized Sigma
Clip, Drizzle Final, Lanczos2, scale 3, pixfrac 1.0, 5760×3240, Coverage Render
APPLIED, FITS+PNG PASS, GUI completion PASS) is cited as release evidence; the
FRP changes are display-only and do not require a scientific rerun.

---

## Version

- **Previous:** 8.2.2 — Phoenix consedit
- **New:** 8.2.3 — Phoenix consedit (patch; unambiguous — all changes are fixes/polish)
- **Authoritative source:** `seestar/__init__.py` `__version__` (pyproject uses
  dynamic `attr = "seestar.__version__"`; `test_version_consistency.py` passes).
- **CHANGELOG:** `## [8.2.3] — Phoenix consedit` entry added.

---

## Nono

Final release review verdict: **ACCEPT** (all 7 final-release criteria verified;
no critical/high/medium findings). Per-workstream reviews also ACCEPT.

---

## Repository

- Branch `fix/final-release-polish`, HEAD `3092000`.
- `git status` clean; `git diff --check 22e86238..3092000` clean.
- No push / merge / tag / release performed.

---

## Recommended promotion sequence (after final human acceptance)

```
fix/final-release-polish → beta   (fast-forward, then push origin/beta)
beta                      → main  (after beta acceptance)
```
