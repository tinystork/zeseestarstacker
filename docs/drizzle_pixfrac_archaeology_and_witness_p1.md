# Drizzle pixfrac archaeology + Phase-1 physical witness

Mission: `zsss-drizzle-scientific-closure-p1-20260910` (Phase 1 only).
Scope: audit/documentation + passive instrumentation. **No science behaviour is
changed, no pixfrac control/setting/migration is touched, and >1 is never
endorsed.**

## 1. What the engine actually does (drizzle 2.2.0, installed 2.2.0)

Executable probes live in `tests/test_drizzle_pixfrac_archaeology_p1.py`
(real `drizzle.resample.Drizzle.add_image` at pixfrac 0.5 / 0.8 / 1.0 / 1.5).

| Kernel | pixfrac active? | Observations |
|---|---|---|
| `square` | yes | `<1` concentrates the footprint; `>1` spreads it (more nonzero output pixels). |
| `turbo` | yes | same footprint/`>1` spread behaviour as `square`. |
| `gaussian` | yes | changing pixfrac changes the weight distribution (active). |
| `point` | no | bit-identical output for every pixfrac. |
| `lanczos2` | no | **pixel_fraction is ignored and assumed 1.0** (upstream warning); output bit-identical for every pixfrac. |
| `lanczos3` | no | same as `lanczos2`. |

So the only kernels for which the GUI pixfrac value can have any effect are
`square`, `turbo`, `gaussian`. For `lanczos2/3` the effective pixfrac is
always `1.0` regardless of what the user requests.

## 2. User-facing selectors

| Surface | File | Bounds / default / step |
|---|---|---|
| Qt standard drizzle spin | `seestar/gui_qt/main_window.py` (`drizzle_pixfrac_spin`) | `QDoubleSpinBox` range `0.01..2.0`, step `0.05`, decimals 2, value from `settings_state.drizzle_pixfrac` (default `1.0`). |
| Qt mosaic form spec | `seestar/gui_qt/main_window.py` | `("pixfrac", "Pixfrac", "float", (0.01, 2.0, 0.05, 2))`. |
| Tk standard drizzle spin | `seestar/gui/main_window.py` (`drizzle_pixfrac_spinbox`) | `ttk.Spinbox from_=0.01 to=2.00 increment=0.05 format="%.2f"`; var default `1.0`. |
| Tk mosaic spin | `seestar/gui/mosaic_gui.py` | `from_=0.01, to=2.00, increment=0.05`; var default from settings `'pixfrac'` (default `0.8`). |
| Tk settings default / migration | `seestar/gui/settings.py` | default `drizzle_pixfrac = 1.0`; out-of-range values are clipped to `[0.01, 2.0]` (a log message is emitted); non-numeric resets to default. Mosaic block default `"pixfrac": 0.8`. |
| Qt settings default | `seestar/gui_qt/settings_state.py` | `drizzle_pixfrac: float = 1.0`; mosaic default `"pixfrac": 0.8`. |

**Conclusion (audit only):** every user-facing selector visibly permits values
up to `2.0`, i.e. `>1`. Legacy settings that stored e.g. `1.5` survive the
clip and are restored unchanged.

## 3. Production normalization / coercion / persistence / resume seams

| Seam | File | Behaviour for pixfrac |
|---|---|---|
| Backend validation | `seestar/core/drizzle_core.py::validate_drizzle_pixfrac` | values outside `(0, 1]` (including `>1`, `0`, NaN/Inf, non-numeric) are normalized to `1.0` with a reason. This is the boundary that keeps the real engine from ever receiving a `>1` value. |
| M3 init policy | `seestar/queuep/queue_manager.py` | `pixfrac_eff = 1.0 if kernel in LANCZOS_KERNELS else pixfrac_requested`; the requested value is preserved on `drizzle_pixfrac_requested` and in the run log. |
| Run contract | `seestar/run_contract.py` | `drizzle_pixfrac_requested` and `drizzle_pixfrac_effective` (scientific section; `drizzle_pixfrac_effective = 1.0` for Lanczos). |
| Checkpoint writer/reader | `seestar/core/drizzle_checkpoint.py` | per-channel `pixfrac` must be in `(0, 1]`; the support accumulators require `pixfrac == 1.0`. |
| Resume locator | `seestar/resume_locator.py` | accepts an *effective* pixfrac in `0.01 <= pixfrac <= 2.0` (i.e. the historical bound is still `2.0`). This is an archaeology fact, **not** an endorsement of `>1`. |
| FITS provenance | `seestar/queuep/queue_manager.py` | `DRZPIXFR` (effective) and `DRZPFREQ` (requested, only when it differs — i.e. the Lanczos case). |

**Net effect today:** the GUI can request `>1`, the settings layer persists it,
the backend coerces it to `1.0` before the engine, and Lanczos additionally
forces `1.0`. The only path on which a `>1` value could reach the engine is a
direct API call bypassing `validate_drizzle_pixfrac` — none exists in
production.

## 4. Phase-1 diagnostic artifact

Each Standard-M3 run now writes one passive, atomic artifact
`drizzle_science_diagnostics.json` into the run output folder (see
`seestar/core/drizzle_science_diagnostics.py`). It carries the resolved
geometry (WCS-derived candidate), the effective `add_image` contract, the final
bbox crop bounds and the effective WHT-threshold policy, per-channel
pre-stretch SCI stats, native signed-WHT diagnostics, a diagnostic-only
threshold sweep (physical-support vs positive-native-WHT denominators kept
explicit), support conditioning (N_eff over valid support only), per-channel
spatial boundary bins, conditioning candidates, and the bounded
SUPPORT_LIFECYCLE ring. It is fail-open and never touches science.

Rework-1 additions:

* the SCI/WHT/support/threshold sections are computed **after** the final bbox
  crop and the effective WHT policy, so artifact row/col/index and support
  statistics address the final SCI FITS grid (`crop` + `sections_meta`);
* per-channel output (no channel-mean cancellation); physical support is
  `SUP_W1 > 0` when the pair exists, otherwise an explicitly-labelled
  native-WHT-derived fallback (`support_source`, and the per-channel
  `threshold_sweep.support_source`/`support_denominator_label`);
* the summary is **memory-bounded** (row-chunked float32 streaming, bounded
  percentile samples and extrema candidates, zero-copy SUP views);
* lifecycle retention coalesces repetitive `checkpoint_save` events (count +
  first/last generation) while always retaining terminal evidence; the
  successful artifact is rewritten **after** cleanup, finalization-returned and
  FITS save so terminal events are persisted, not just held in memory; every
  terminal failure route (invalid accumulators / no support / support-integrity
  / FITS failure) persists `artifact_final` with an explicit
  `outcome`/`success`/`reason`.

Rework-2 additions (real x3/x4 boundedness):

* the boundary analysis streams **row tiles with a bounded halo (>= 17 px)** and
  a transient tile EDT — no full-frame distance map and no float64 upcast of a
  resident float32 map.  The four global array edges are treated as support
  boundaries (false padding); interior continuations are padded True so no
  artificial window boundary appears.  Per-extreme boundary distances are
  queried locally (bounded window); unresolved distances are reported as
  `distance=null` with `distance_resolved=false` (a truthful `>16`);
* local positive-WHT references use bounded tile samples; the SUP views are
  taken zero-copy from the already-resident private native `_out_wht` buffers
  (never the copying `.wht` property), sliced after the crop and never mutated;
* measured incremental peak (isolated process, in-place resident inputs,
  `VmHWM` after minus before): **x3 3240x5760 ~92 MiB (target <=256)**, **x4
  4320x7680 ~147 MiB (target <=384)** — versus rework-1 ~606 / ~1077 MiB.
  Runtime ~29 s (x3) / ~75 s (x4) synchronous at the finalization seam, fully
  fail-open.

## 5. Physical witness instructions (run by Tristan after local acceptance)

Purpose: obtain real physical evidence that the diagnostics faithfully describe
a physical run and to compare the benign control against the unsafe kernels.
**Do not run this as part of Phase 1; it is an operator step.**

Preconditions
- Fresh Standard Drizzle run (not resume, not mosaic).
- Same dataset, same reference, same subset for all variants.
- Quality weighting unchanged; everything except the kernel/pixfrac below
  identical.
- 20–50 accepted frames.
- STOP is allowed and relevant — a mid-run STOP is itself useful evidence.

Variants
- **A — control:** kernel `Square`, scale ×3, pixfrac `1`.
- **B — under test:** kernel `Lanczos2`, scale ×3, pixfrac `1`.
- **C — optional:** kernel `Lanczos3`, scale ×3, pixfrac `1`.

Preserve for each variant
- the run log;
- `run_config.cfg`;
- registration diagnostics (`registration_diagnostics.jsonl`);
- the final SCI FITS;
- the preview PNG;
- `drizzle_science_diagnostics.json`;
- checkpoint metadata (generation files / manifest).

Report back: the three artifacts plus any observed difference in the SCI FITS
and the diagnostics' SCI/WHT/support sections. Do not launch a long physical
run from within Phase 1.
