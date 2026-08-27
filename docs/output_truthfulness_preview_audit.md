# ZSSS-OTPUX-AUDIT-01 — Exposure Metadata & Preview/Histogram Lifecycle Audit

> **C3 corrective iteration (3 of max 3 — final).** C1 was rejected after independent review;
> C2 closed six defects but left the preview architecture as an open two-option choice and left
> Auto Stretch/Auto WB too broad. C3 finalizes the document by (1) ratifying **one stable
> preview architecture (Option A only)** — §5.2; (2) specifying an **exact background-population
> Auto Stretch** — §5.5; (3) specifying an **exact true-background-band Auto WB** — §5.6; (4)
> ratifying **histogram details** (512 bins, RGB overlay, log1p counts, full stats, ~25 ms live
> drag, worker generation token) — §5.3/§5.7; (5) cleaning up current-facts labels — §2.2; and
> (6) converting the open "architect decisions needed" list into **ratified architect
> decisions** — §11. No open architectural choice remains.

- **Mission:** ZSSS-OTPUX-AUDIT-01 (audit/documentation only — no production code)
- **Corrective iteration:** C3 (this doc only; supersedes C1/C2)
- **Branch:** `feature/output-truthfulness-preview-ux`
- **HEAD:** `a7fc8d6a6c16f4c5d724152ebcc16255cac78093` (== `origin/beta` at mission start)
- **Document status:** AUDIT-01 is **accepted pending parent (Jarvis/Tristan) verification** —
  it is *not* a project-wide acceptance and changes no project state.
- **Implementation closure (2026-08-27):** the contracts subsequently implemented
  in this same worktree, including STABLE-A/B/C, are **accepted locally by the
  architect** after independent diff review and consolidated validation; see
  §14.  The original audit statements above and §§1-11 remain the historical
  pre-implementation snapshot, not the current production-state verdict.
- **Scope of this doc:** current-behaviour audit + evidence-backed *proposed contracts* for the
  deliverables `EXPOSURE_METADATA_CONTRACT`, `CURRENT_PREVIEW_AUDIT`, `HISTOGRAM_CONTRACT`,
  `DISPLAY_STATE_CONTRACT`, `AUTO_STRETCH_CONTRACT`, `AUTO_WB_CONTRACT`.
- **Non-goals honored:** no widget redesign, no production code, no HSI/Drizzle math change,
  no version/push/branch/merge change, only this one doc created.

---

## 1. Executive finding

1. **The scientific stacking pixels are clean and independent of display state.** The Qt shell's
   preview/histogram/adjustment pipeline is strictly display-only and never feeds back into the
   SUM/WHT memmaps or Drizzle accumulators. The invariant "display state never feeds stacking"
   **holds today** — every adjustment helper in `preview_adjust.py` operates on a copied `QImage`
   and the backend preview callbacks only *read* accumulator state (`queue_manager.py`).

2. **Exposure metadata has two distinct defects, not one.**
   - **`TOTEXP` is silently wrong for Drizzle.** `total_exposure_seconds` is incremented *only*
     in the classic `_combine_batch_result` path (`queue_manager.py:10756`). The Drizzle M3 path
     never increments it, yet `_save_final_stack` writes `TOTEXP = round(total_exposure_seconds, 2)`
     for *all* modes (`15948`). A fresh Drizzle run therefore writes `TOTEXP = 0.0`, and the GUI
     summary shows "Total Exposure: 0 s".
   - **`NIMAGES` counts *aligned* files, not *accepted/contributing* frames, for Drizzle.**
     `_worker` increments `aligned_files_count` *before* calling
     `_add_frame_to_drizzle_accumulators` (`6424` → `6447`). If the Drizzle add fails (returns
     `False`), only `failed_stack_count` is bumped (`6468`); `aligned_files_count` is **not**
     rolled back. Because Drizzle never touches `images_in_cumulative_stack`, `_save_final_stack`
     falls back to `aligned_files_count` (`15918-15922`), so a failed accumulator add inflates the
     final `NIMAGES`. (`failed_stack_count` is a diagnostic counter only; it is never written to
     the final header.)

3. **The current Qt histogram is WB-only, 8-bit, 256 bins, synchronous, release-only drag.**
   It is computed from a WB-only **uint8** `QImage` (not the backend's float32 `[0,1]` data),
   over 256 bins with range `(0,256)`, recomputed synchronously on the GUI thread inside every
   `_refresh_preview_view`, and `rangeChanged` is emitted **only on mouse release** — so the
   preview/stretch does not live-update during a BP/WP line drag, and the 8-bit quantization
   collapses fine tonal detail before binning.

4. **Exposure read/normalization sites are divergent and mix two unrelated concerns.** There are
   at least four distinct sites that read `EXPTIME` with different fallbacks (classic batch sum,
   single-image batch, legacy incremental-drizzle `add_image`, M3 accumulator `add`), plus a dead
   `_update_header_for_drizzle_final`. Crucially, the M3 Drizzle per-frame read
   (`float(header.get("EXPTIME", 1.0))`, `18085`) is a **scientific accumulator-scaling fallback**
   (used to scale counts into the Drizzle accumulator via `in_units="counts"`), *not* a truthful
   metadata-seconds value. These two must be documented separately: the scientific fallback is
   legitimate for pixel math; the metadata-seconds value must never silently substitute `1.0`
   (or `0.0`) for a missing/unknown exposure in `TOTEXP`.

5. **`_save_final_stack` never *writes* a per-frame `EXPTIME`/`EXPOSURE` keyword, but it *copies*
   `current_stack_header` (`15928`), so a final header can *inherit* a stale per-frame `EXPTIME`.**
   `_save_final_stack` itself writes only `NIMAGES`/`TOTEXP` (+ WCS/pointing) (`15944-15950`).
   The standard classic SUM/W path fills `current_stack_header` from an explicit `keys_to_copy`
   list that **excludes** `EXPTIME` (`10774-10788`); the standard M3 Drizzle path leaves it
   without `EXPTIME` (fresh `fits.Header()` at `10035-10067`, or the `fits.Header()` fallback at
   `15928-15931`). However, some finalization paths do `self.current_stack_header = hdr.copy()`
   from a batch file header that *carries* `EXPTIME` — e.g. `_finalize_single_classic_batch`
   (`15263`), whose `hdr` is the classic-batch file header written from
   `stack_info_header = reference_header_for_wcs.copy()` (`_stack_batch` `11921-11925`) — so a
   single-batch reproject/coadd final save **can inherit a stale `EXPTIME`**. Target (§5.1 point
   7): final save must explicitly set a uniform `EXPTIME` or delete any stale/inherited
   `EXPTIME`/`EXPOSURE` for mixed/unknown.

6. **Duplicate/redundant work in the refresh path.** `_refresh_preview_view` applies WB twice
   (once for the histogram's `wb_only` image at `3044`, once inside `apply_preview_adjustments` at
   `3045-3055`), and the histogram path runs three independent `QImage → ndarray` conversions per
   refresh (`compute_histogram`, `compute_histogram_percentile`, `compute_histogram_stats`).

The **proposed contracts (§5)** pin the current behaviour explicitly, mark the Drizzle
`TOTEXP=0.0` divergence, the Drizzle `NIMAGES` aligned-vs-accepted inflation, and the 8-bit
histogram as *known current deviations*, and define a minimal, reversible implementation path
(§7) that does not touch science pixels. **Existing passing tests are baseline only (§9); none
of them prove the target contracts.**

---

## 2. Evidence map (file:line for every key claim)

### 2.1 Qt is the current production GUI
- `ZeSeestarStacker.egg-info/entry_points.txt`: `zeseestarstacker = seestar.qt_main:main`
- `seestar/qt_main.py:1-17` (module docstring): official PySide6 entry point; default backend
  `seestar`.

### 2.2 Preview data domain & normalization (backend → GUI)

Four preview producers exist. The **standard live classic SUM/W** path is
`_update_preview_sum_w` and the **standard live M3 Drizzle** path is
`_update_preview_drizzle_accumulator`; both currently re-normalize **per preview** (the mapping
is recomputed from the *current* accumulator content, so it drifts as the stack evolves). A
third producer, `_update_preview`, is a **legacy incremental-reprojection/coadd path** (plus a
folder-add refresh), *not* the standard classic/M3 live path. The fourth is an **end-of-run**
final-save representation, *not* part of the live preview mapping at all.

- **`_update_preview` (legacy incremental-reprojection/coadd + folder-add refresh — NOT the
  standard classic/M3 live path)** `queue_manager.py:3889-3933`: sends
  `(current_stack_data.copy(), current_stack_data_raw.copy() or current_stack_data.copy())` — a
  **tuple of two arrays**. It performs **no normalization itself**; it relies on whoever last
  populated `current_stack_data`/`current_stack_data_raw`. Its live setters are:
  - `_update_preview_master` (`4527-4544`): `raw = avg_cropped`; `current_stack_data =
    clip((raw - nanmin(raw))/(nanmax(raw) - nanmin(raw)), 0, 1)` — **min/max per preview**.
  - legacy incremental-drizzle batch (`10111-10125`): same **min/max** normalization.
- **`_update_preview_sum_w` (classic SUM/W)** `queue_manager.py:4180-4400`: reads SUM/WHT as
  float64 → `avg = SUM / max(WHT, 1e-9)` (NaN/Inf→0) → optional feathering/low-WHT-mask →
  `(avg - nanmin(avg))/(nanmax(avg) - nanmin(avg))` → `clip(0,1)` → `float32` (`4290-4320`) →
  `cv2.resize` INTER_AREA by `eff_factor` (param > `preview_downsample_factor` > default `2`,
  clamped `1..4`, only if new `H,W > 10`) → sends a **single** array `preview_data_to_send`
  (`4340-4390`). **min/max per preview.**
- **`_update_preview_drizzle_accumulator` (M3 Drizzle)** `queue_manager.py:18184-18270`: per
  channel `acc.finalize("divide")` → HWC stack → **percentile stretch** `lo,hi =
  nanpercentile(preview_hwc, [1.0, 99.0])`; `(x - lo)/(hi - lo)` (else 0.5 / 0) → `clip(0,1)`
  `float32` (`18206-18216`) → downsample if `max(side) > _MAX_PREVIEW_SIDE_PX` (= `1000`, defined
  at `293`) → GUI downsample factor `1..4` → sends a **single** array `preview_to_send`
  (`18220-18260`). **1%/99% percentile per preview.**
- **Final save preview (end-of-run, separate representation — not the live mapping contract)**
  `_save_final_stack` (`15911`): sets
  `self.last_saved_data_for_preview = data_after_postproc.copy()` (the post-processed `[0,1]`
  image, "sans stretch cosmétique du backend" — docstring `15323`); the final preview PNG uses
  `save_preview_image(..., apply_stretch=True, enhanced_stretch=False)` (`365-412`). This is a
  **one-shot** post-processed export produced at the end of a run; it is **not** a live producer
  and does not participate in the per-preview normalization/mapping contract of §5.2.

**Summary of the normalization divergence (this is the C2 defect #2):**
- classic SUM/W uses **min/max**, M3 Drizzle uses **1%/99% percentile** — two different,
  incompatible mappings for the same nominal "preview".
- both recompute their reference points on **every** preview from the *current* accumulator
  content, so **the same physical pixel can change numeric value between successive previews**
  purely because the stack's extrema moved (no fixed reference pixel stays constant).
- neither path carries the linear float source to the GUI: `render_preview_image`
  (`preview_render.py:177`) quantizes to `uint8` via `_to_uint8` (`119-155`) and, for the legacy
  tuple, `_iter_array_candidates` (`80-100`) returns the **first** array-like element that
  renders — the already-normalized `current_stack_data` — so `current_stack_data_raw` (the linear
  average) is **discarded** at the Qt boundary.

- Payload mapping (toolkit-free carrier): `backend_runner.py:66-99` `BackendPreviewPayload` —
  `data`/`header`/`stack_name`/`image_count`/`total_images`/`current_batch`/`total_batches`/
  `extra`; `backend_runner.py:385-421` `_map_preview_payload` (positional `data` = arg 0,
  `header` = arg 1, `stack_name` = arg 2, …; args beyond 7 → `extra`). **`BackendPreviewPayload`
  preserves the data verbatim (tuple stays a tuple, single array stays single) and discards
  nothing; it performs no normalization and no interpretation.**

### 2.3 Preview buffer → display (Qt)
- `main_window.py:2595-2632` `_on_preview`: `render_preview_image(payload.data)` at `2619`,
  stores deep-copied `QImage` into `_preview_source` at `2623`, then `_refresh_preview_view()`.
- `main_window.py:2696-2732` `_on_initial_preview_result`: same render path for the first-FITS
  initial preview.
- `preview_render.py:119-155` `_to_uint8`: **numeric domain decision** — `float` → `clip(0,1)`
  → `*255` → `uint8`; `bool` → `*255`; `uint8` passthrough; other ints clip to `[0,255]`.
  This is where the backend `float32 [0,1]` data is quantized to **8-bit**.
- `preview_render.py:177-196` `render_preview_image`: single public entry, returns a deep copy.

### 2.4 Display decisions / authoritative state
- `main_window.py:814` `self._preview_source` (the pristine display source `QImage`).
- `main_window.py:843-849`: `_stretch`, `_black_point`, `_white_point`, `_gamma`,
  `_brightness`, `_contrast`, `_saturation` — the authoritative display-adjustment state.
- `main_window.py:836-842` (comment) + `_wb` set in `_on_wb_changed` (`2886-2893`).
- `main_window.py:815-835`: `_preview_rotation`, `_preview_res_factor`, `_preview_zoom_factor`,
  `_view_offset_x/_view_offset_y` (view transform state).
- `main_window.py:3020-3109` `_refresh_preview_view`: the single display refresh path.
  - `wb_only = apply_preview_wb(source, wb=self._wb)` at `3044` (histogram source).
  - `adjusted = apply_preview_adjustments(source, wb=..., stretch=..., ...)` at `3045-3055`.
  - `render_view(adjusted, ...)` at ~`3057-3067`.
  - `_refresh_histogram(wb_only)` at `3094`.

### 2.5 Histogram contract (current)
- `main_window.py:3165-3181` `_refresh_histogram`: feeds the WB-only image,
  `right_histogram_view.set_data(image)` (`3178`), `set_range(bp, wp)` (`3179`),
  `compute_histogram_stats(image)` (`3180`).
- `histogram_view.py:81-104` `set_data`: `compute_histogram(image, bins=256)` + `p99.5`.
- `histogram_view.py:60` `rangeChanged = Signal(float, float)`.
- `histogram_view.py:218-224` `_end_drag`: emits `rangeChanged` **on release only**.
- `histogram_view.py:236-239` `mouseReleaseEvent`: the only path that calls `_end_drag`.
- `histogram_view.py:225-234` `mouseMoveEvent` → `_drag_at`: updates the line visually
  (`_drag_at` at `208-216`) **without** emitting — so no live preview update during drag.
- `preview_adjust.py:432-456` `compute_histogram`: `np.histogram(plane, bins=bins, range=(0,256))`
  over a **uint8** array (`range=(0,256)` at `453`); grayscale → `"L"`, RGB → `"R"/"G"/"B"`.
- `preview_adjust.py:458-476` `compute_histogram_percentile`: `percentile` over `arr/255.0`.

### 2.6 Auto Stretch / Auto WB (current)
- `main_window.py:2935-2951` `_on_auto_stretch`: `wb_only = apply_preview_wb(...)` (`2945`) →
  `compute_auto_stretch(wb_only)` → writes BP/WP spins → `setCurrentText("asinh")`.
- `preview_adjust.py:378-430` `compute_auto_stretch`: luminance = `0.299R+0.587G+0.114B`,
  BP/WP = `percentile(..., 1.0)` / `percentile(..., 99.0)` (`405-406`), then normalized through
  **full-image min/max** into `[0,1]` UI scale (`417-425`), `1e-4` min separation. (The
  full-image absolute min/max normalization is the behaviour the architect flagged for removal.)
- `main_window.py:2906-2919` `_on_auto_wb`: `compute_auto_wb(self._preview_source)` (the **pre-WB**
  source, *not* `wb_only`) — this is already the deterministic/idempotent input; the target
  contract formalizes it and hardens the valid-pixel selection (§5.6).
- `preview_adjust.py:337-375` `compute_auto_wb`: per-channel mode from a 256-bin histogram over
  `[0.5, 99.5]` percentile range (`363`), equalized toward green mode, R/B gains clipped
  `[0.2, 5.0]` (`372-374`); non-color returns neutral `(1,1,1)`. It computes the mode over the
  full per-channel `[0.5, 99.5]` percentile range **without excluding borders/clipped/saturated
  pixels** — the robust-selection gap the target addresses.

### 2.7 Exposure metadata flow (source → final)
- **Source read (header preserved):** `core/image_processing.py:62-205` `load_and_validate_fits`
  copies the HDU header verbatim (`header = hdu_img.header.copy()`, ~`120`); EXPTIME/EXPOSURE are
  not stripped. `queue_manager.py:8440-...` `_process_file` sets
  `header_final_pour_retour = header_from_load.copy()` (~`8500`) and adds `_SRCFILE`; this header
  travels to alignment and back as `item_result_tuple[1]` = `header_orig`
  (`queue_manager.py:6428-6433`, `6315-6323`).
- **Batch header `TOTEXP` generation (two sites):**
  - `queue_manager.py:11566-11573` single-image mean batch:
    `tot_exp = float(stack_info_header.get("EXPTIME", stack_info_header.get("EXPOSURE", 0.0)))`
    at `11570` — **falls to `0.0` if both keys absent**.
  - `queue_manager.py:11944-11956` multi-image batch: loops `valid_headers_for_ccdproc`, sums the
    first of `("EXPTIME","EXPOSURE")` present per header; headers with neither contribute `0`.
- **Classic accumulation:** `queue_manager.py:10600`
  `batch_exposure = float(stack_info_header.get("TOTEXP", 0.0))`; `10756`
  `self.total_exposure_seconds += batch_exposure`.
- **Classic accepted-frame counter:** `queue_manager.py:10620`
  `self.images_in_cumulative_stack += num_physical_images_in_batch` — incremented **only after**
  the shape/coverage gates in `_combine_batch_result` (`10470-10616`); a rejected classic batch
  returns before this line (e.g. `10512`, `10550`, `10577`, `10593`, `10616`) and bumps only
  `failed_stack_count`. So classic `NIMAGES` is approximately "batches that passed the gate".
- **Classic final header:** `queue_manager.py:10826-10833` `current_stack_header["NIMAGES"]` and
  `current_stack_header["TOTEXP"] = round(self.total_exposure_seconds, 2)` (`10830`).
- **Final FITS header (all modes):** `queue_manager.py:15918-15950`
  - `final_header = self.current_stack_header.copy() if self.current_stack_header else
    fits.Header()` (`15928-15931`).
  - `effective_image_count = images_in_cumulative_stack if >0 else aligned_files_count`
    (`15918-15922`; fallback applies when Drizzle/mosaic — `images_in_cumulative_stack` stays 0
    for Drizzle).
  - `final_header["NIMAGES"] = effective_image_count` (`15944`).
  - `final_header["TOTEXP"] = round(self.total_exposure_seconds, 2)` (`15948`).
  - **`EXPTIME`/`EXPOSURE` are never *written* here** — but they are **not stripped either**, so
    they survive iff they are already in `current_stack_header`.
- **`EXPTIME` inheritance (C2 correction of C1's "no final `EXPTIME` at all"):**
  - Classic SUM/W: `current_stack_header` is built in `_combine_batch_result` from an explicit
    `keys_to_copy` list (`10774-10788`) that **does not include `EXPTIME`** ⇒ no `EXPTIME`.
  - M3 Drizzle: `current_stack_header` is a fresh `fits.Header()` (init `10035-10067`; fallback
    `15931`) with no `EXPTIME`.
  - **But** `_finalize_single_classic_batch` (`15263`) does
    `self.current_stack_header = hdr.copy()` where `hdr` is the classic-batch file header
    (`15250`), itself written from `stack_info_header = reference_header_for_wcs.copy()`
    (`_stack_batch` `11921-11925`) — and `reference_header_for_wcs` originates from a
    solved/loaded source header that carries `EXPTIME` (`12122`, `12153`, `13546`, `14743`,
    `17463`). So the **single-classic-batch reproject/coadd** final save inherits a stale
    `EXPTIME`.
  - Net: "no final `EXPTIME`" is **false as a blanket claim**; it is true only for the standard
    classic multi-batch SUM/W and M3 Drizzle paths.

### 2.8 Drizzle exposure divergence (the core bugs)
- Drizzle M3 path adds frames directly to `drizzle_accumulators` and **never** calls
  `_combine_batch_result`: `queue_manager.py:6421-6462` (`aligned_files_count += 1` at `6424`,
  `_increment_aligned_counter()` at `6425`, `_add_frame_to_drizzle_accumulators(...)` at `6447`).
  - **Order matters:** `aligned_files_count` is incremented *before* the add attempt; on failure
    only `failed_stack_count += 1` (`6468`) — no rollback of `aligned_files_count`.
- `_add_frame_to_drizzle_accumulators` (`queue_manager.py:18018-18107`) reads
  `exptime = float(header.get("EXPTIME", 1.0))` (`18085`) with fallback `1.0`, sanitizes
  non-finite/`<=0` to `1.0` (`18087-18088`), and always `in_units="counts"` (`18098`). This is
  the **scientific scaling fallback** (counts → accumulated drizzle units); it does **not** touch
  `total_exposure_seconds` and must not be conflated with metadata seconds.
- `total_exposure_seconds` is incremented **only** at `queue_manager.py:10756` (classic
  `_combine_batch_result`). Full-file grep confirms no other `+=`.
- **There is already an accepted-frame counter** — `_drizzle_frame_count` — incremented exactly
  once per successful add: it is bumped inside `_drizzle_group_tick()` (`18146-18148`), which the
  M3 worker calls **only inside the `if added:` branch** (`6454-6465`). `_save_final_stack` does
  **not** use it; it falls back to `aligned_files_count` (incremented *before* the add at `6424`,
  never rolled back) — this is the C2 counter-placement defect (a correct counter already exists;
  the final save reads the wrong one).
- Therefore, for a fresh Drizzle run:
  - `total_exposure_seconds == 0.0` → `_save_final_stack` writes `TOTEXP = 0.0` (`15948`).
  - `images_in_cumulative_stack == 0` → `NIMAGES = aligned_files_count` (`15922`), which includes
    frames whose Drizzle add **failed** (`6424` before `6447`, no rollback) — even though
    `_drizzle_frame_count` holds the correct accepted value.
- GUI impact: `summary_payload.py:110-111` reads `TOTEXP` from `final.fits` →
  `total_exposure_seconds` shown as `0`.

### 2.9 Resume persistence / restore (classic only)
- Manifest schema/persistence: `queue_manager.py:12626-12698` `_write_resume_manifest` persists
  `images_in_cumulative_stack` (`12677-12679`), `total_exposure_seconds` (`12680-12682`),
  `cumulative_header` (`12683`), `stacked_batches_count`, session identity, ledger.
- `cumulative_header` serialization: `queue_manager.py:12422-12433` `_serialize_cumulative_header`
  (skips `HISTORY`/`COMMENT`); rebuild `12435-12444` `_header_from_serialized`.
- Restore: `queue_manager.py:13205-13238` reads `images_in`, `totexp`, `header_data`, sets
  `self.total_exposure_seconds = totexp` (`13237`) and
  `self.current_stack_header = self._header_from_serialized(header_data)` (`13238`).
- Resume is **limited to plain classic SUM/W**: `queue_manager.py:12760`
  (`manifest.get("mode") != _RESUME_MODE_CLASSIC_SUMW` → refused); header/mode context at
  `queue_manager.py:445-513`.

### 2.10 Divergent / dead EXPTIME sites
- Legacy incremental-drizzle `add_image` path (older, still present): `queue_manager.py:9817-9846`
  (EXPTIME valid → `in_units="counts"`; else `exptime=1.0`, `in_units="cps"`).
- `enhancement/drizzle_integration.py:314-315`: `exposure_time = max(1e-6, float(EXPTIME))`.
- Dead code: `queue_manager.py:7760-7859` `_update_header_for_drizzle_final` — defined but
  **never called** anywhere in the repo (grep confirms only the definition + internal log lines).
  **Left untouched per this mission; documented as dead only (see §6-R6, §7-D).**
- Reference temp-save key copy: `core/alignment.py:625` `safe_keys_to_copy = ['DATE-OBS',
  'EXPTIME', 'FILTER', 'INSTRUME', 'OBJECT']` (reference_image.fit only, not the batch path).

---

## 3. Pre-change maps

### 3.1 Exposure metadata map

```
source FITS (EXPTIME/EXPOSURE in HDU header)
   │ load_and_validate_fits (core/image_processing.py:62)  — header copied verbatim
   ▼
_process_file (queue_manager.py:8440)  — header_orig = loaded header + _SRCFILE
   │  alignment (6-tuple: data, header, scores, wcs, matrix_M, valid_mask)
   ▼
diverge:
  ├─ classic SUM/W ──▶ _stack_batch (11323)
  │     batch header TOTEXP = Σ EXPTIME|EXPOSURE  (11944-11956, single 11570)
  │     └─▶ _combine_batch_result (10345):
  │           shape/coverage gates (10470-10616) → reject ⇒ failed_stack_count only
  │           batch_exposure = TOTEXP (10600)
  │           images_in_cumulative_stack += NIMAGES (10620)   ◀── accepted (post-gate)
  │           total_exposure_seconds += batch_exposure (10756)   ◀── ONLY increment site
  │           current_stack_header NIMAGES/TOTEXP (10826-10833)
  │     └─▶ final_header NIMAGES=images_in_cumulative_stack; TOTEXP=total_exposure_seconds (15944-15948)
  │
  └─ Drizzle (M3) ──▶ aligned_files_count += 1  (6424, BEFORE add, never rolled back)
        └─▶ _add_frame_to_drizzle_accumulators (18018)
              exptime = EXPTIME|1.0 (18085) → accs[ch].add(exptime, in_units="counts")  [SCIENCE fallback]
              returns True/False → False ⇒ failed_stack_count += 1 (6468), aligned_files_count unchanged
        images_in_cumulative_stack: NOT updated (stays 0)
        total_exposure_seconds: NOT updated  ◀── DIVERGENCE → TOTEXP == 0.0
        └─▶ final_header NIMAGES = aligned_files_count (15922)  ◀── includes failed adds
             final_header TOTEXP = total_exposure_seconds (15948) == 0.0
```

Resume (classic only) persists/restores `total_exposure_seconds` + `cumulative_header`
(`12680-12683`, `13205-13238`).

### 3.2 Preview / histogram map (Qt)

```
backend (float32 [0,1] HWC, possibly downsampled)
   │ preview_callback → BackendPreviewPayload.data (backend_runner.py:385-421)
   ▼
MainWindow._on_preview (2595) [GUI thread]
   │ render_preview_image (preview_render.py:177) → _to_uint8 (119) → uint8 QImage (deep copy)
   ▼
_preview_source (814)  — pristine uint8 QImage   (float [0,1] ndarray NOT retained today)
   │
   ▼
_refresh_preview_view (3020)  [GUI thread, synchronous]
   ├─ wb_only = apply_preview_wb(source, wb)          (3044)  → histogram source (uint8, WB-only)
   ├─ adjusted = apply_preview_adjustments(source, ...)(3045)  → WB→stretch→gamma→BCS (uint8)
   ├─ render_view(adjusted, rot, zoom, pan)            (~3057) → QPixmap
   └─ _refresh_histogram(wb_only)                      (3094)
         └─ HistogramView.set_data(wb_only) (81) → compute_histogram(bins=256, range=(0,256))
              + compute_histogram_percentile(99.5)   [synchronous, full-image]
```

Authoritative display state lives only in `MainWindow` (`_wb/_stretch/_black_point/_white_point/
_gamma/_brightness/_contrast/_saturation` + view transform fields). `HistogramView` holds a
*derived* histogram and its own BP/WP line positions (`_black_point/_white_point`), which are
re-synced from the sliders on every `set_data`.

**Target domain separation (C3 — Option A, replaces C1's pre-WB choice):**

```
backend scientific current stack (SUM/WHT memmaps or Drizzle accumulators)   ← mutating
   │ callback sends tuple (legacy_normalized, raw_linear)
   │   legacy_normalized → legacy-consumer compatibility only (first element)
   │   raw_linear        → linear float source (same geometry, downsampled/capped)
   ▼
Qt extracts/copies raw_linear; computes anchors ONCE per run / first valid preview
   │  anchor_lo/anchor_hi = p0.5/p99.5 (finite min/max fallback only if degenerate)
   │  maps every later raw_linear → pristine normalized [0,1] through the same anchors
   │  builds production _preview_source/QImage from that mapped copy
   ▼
immutable linear preview representation (float [0,1], frozen mapping)   ← Qt-owned
   │ apply authoritative WB gains  →  derived WB-only linear [0,1] analysis buffer
   ▼
WB-only analysis domain  ── histogram / stats / Auto Stretch / BP/WP
   │                       (AutoWB alone uses the pristine pre-WB source)
   │ apply WB → stretch → gamma → BCS (display transform)
   ▼
display transform  →  uint8 QImage  →  view (zoom/pan/rotation)
```

The anchors are reset only at run start / a new initial preview context; Manual/Auto Stretch and
AutoWB never change them. The WB-only analysis buffer is recomputed **only on a new raw source or
a WB change** (§5.3 point 7); BP/WP/stretch/gamma/BCS/zoom/pan/rotation operate *after* it and
never move it.

---

## 4. Current-behaviour contracts (as-built)

### 4.1 `EXPOSURE_METADATA_CONTRACT` (as-built)

| Aspect | Current behaviour | Evidence |
|---|---|---|
| Source of truth per frame | original FITS header `EXPTIME` (else `EXPOSURE`) | `image_processing.py:120`, `queue_manager.py:11570`, `11948` |
| Batch TOTEXP (multi) | Σ over valid batch headers of first-present `EXPTIME`/`EXPOSURE`; missing → contributes `0` | `11944-11956` |
| Batch TOTEXP (single mean) | single `EXPTIME`→`EXPOSURE`→`0.0` | `11566-11573` |
| Classic accepted counter | `images_in_cumulative_stack` incremented **after** shape/coverage gates | `10620` vs `10470-10616` |
| Classic total | cumulative `total_exposure_seconds` += batch TOTEXP | `10600`, `10756` |
| Classic final TOTEXP | `round(total_exposure_seconds, 2)` | `10830`, `15948` |
| Drizzle accepted counter | **none** — `images_in_cumulative_stack` never incremented | Drizzle path `6421-6462` never calls `_combine_batch_result` |
| Drizzle final NIMAGES | `aligned_files_count` fallback (incl. failed adds; `6424` before `6447`, no rollback) | `15918-15922` |
| Drizzle final TOTEXP | `round(total_exposure_seconds, 2)` = **0.0 (bug)** | `15948` vs no increment in `18018-18107` |
| Drizzle per-frame exptime | `EXPTIME`→`1.0` (sanitized finite/>0), `in_units="counts"` — **science scaling**, not metadata | `18085`, `18087-18088`, `18098` |
| Final per-input `EXPTIME` keyword | **never *written* by `_save_final_stack`; can be *inherited* via `current_stack_header.copy()` (`15928`) in single-classic-batch paths (`_finalize_single_classic_batch` `15263`)** | `15928-15950`; classic `keys_to_copy` excludes it (`10774-10788`); M3 Drizzle header omits it (`10035-10067`) |
| Rejected/failed representation | `failed_stack_count += …` only (diagnostic; never written to final header) | `6468`, `10512/10549/10576/10592`, `10889` |
| Resume persistence | `total_exposure_seconds` + serialized `cumulative_header` | `12680-12683`, restore `13205-13238` |
| Resume scope | classic SUM/W only | `12760` |

### 4.2 `CURRENT_PREVIEW_AUDIT` (as-built)

- **Pixel buffer:** copied `QImage` (uint8, Grayscale8/RGB888/RGBA8888) stored as
  `MainWindow._preview_source` (`814`, `2623`).
- **Numeric domain:** backend sends `float32 [0,1]`; renderer quantizes to `uint8` (`_to_uint8`,
  `preview_render.py:119-155`). WB/stretch/gamma/BCS all operate on `uint8` → `float64/255` →
  `uint8` (`preview_adjust.py:69-133`, `242-330`). **The float `[0,1]` source is not retained
  after quantization.**
- **Before/after WB/stretch:** histogram = WB-only (pre-stretch); displayed image = WB→stretch→
  gamma→BCS (`3020-3094`).
- **Bit depth:** 8-bit at render time; histogram bins 8-bit values.
- **Bins:** 256 (`histogram_view.py:81`, `preview_adjust.py:453`).
- **Recomputation triggers:** every `_refresh_preview_view` (slider change, stretch change, WB
  change, auto-WB/auto-stretch, wheel zoom, pan, Res cycle, new preview).
- **Thread:** GUI thread, synchronous (no worker offload).
- **Duplicate work:** WB applied twice per refresh (`3044` + inside `3045`); three independent
  `_image_to_array` conversions per histogram refresh (`compute_histogram`,
  `compute_histogram_percentile`, `compute_histogram_stats`).

### 4.3 `HISTOGRAM_CONTRACT` (as-built)

- Single live surface: `HistogramView` (right panel); tab duplicate removed (module docstring
  `histogram_view.py:1-14`, `main_window.py:1354-1429`).
- Source: WB-only `QImage` (`main_window.py:3044`, `3165-3179`).
- 256 bins over `range=(0,256)` of uint8 channel planes (`preview_adjust.py:432-456`).
- BP/WP lines in normalized `[0,1]` level space; `set_range` clamps with `1e-4` min separation
  (`histogram_view.py:140-153`).
- `rangeChanged` emitted **on release only** (`histogram_view.py:218-239`); drag moves the line
  visually without emitting.
- BP/WP drag → mirrored to stretch sliders via `_on_hist_range_changed`
  (`main_window.py:2953-2957`, wired at `1813`).

### 4.4 `DISPLAY_STATE_CONTRACT` (as-built)

- Authoritative state = `MainWindow` fields (`_wb`, `_stretch`, `_black_point`, `_white_point`,
  `_gamma`, `_brightness`, `_contrast`, `_saturation`; view: `_preview_rotation`,
  `_preview_res_factor`, `_preview_zoom_factor`, `_view_offset_x/y`).
- `_preview_source` is never mutated; every adjustment/view helper returns a fresh image
  (`preview_adjust.py:17-19`, `preview_view.py:24-26`).
- Display state never feeds stacking (no write path from `MainWindow` → engine except the M22
  `set_preview_downsample_factor` display-resolution control, `main_window.py:2857-2899`, which
  only re-renders preview resolution, not science).

### 4.5 `AUTO_STRETCH_CONTRACT` (as-built)

- Input: WB-only image (`apply_preview_wb(source, wb)`), `main_window.py:2945`.
- Algorithm: luminance percentiles 1%/99% → map through **full-image absolute min/max** into
  `[0,1]` (`preview_adjust.py:378-430`); min separation `1e-4`.
- Trigger: Auto Stretch button (`main_window.py:1797`, `2935`).
- Side effect: switches stretch mode to `asinh` and writes BP/WP spins (which trigger refresh).
- Lifecycle: no memory of whether auto-stretch already ran; the button recomputes on each click;
  backend preview updates do not automatically re-run auto-stretch (BP/WP values are user state).

### 4.6 `AUTO_WB_CONTRACT` (as-built)

- Input: `_preview_source` (pre-WB, pre-stretch), `main_window.py:2906-2919`.
- Algorithm: per-channel mode from 256-bin histogram over `[0.5, 99.5]` percentile range,
  equalize toward green, R/B gains clip `[0.2, 5.0]` (`preview_adjust.py:337-375`).
- Exclusions: non-color / <3 channels / missing data → neutral `(1,1,1)`.
- **No valid-pixel pre-selection** (no border/zero/clip/saturation exclusion) — the mode is
  estimated over the raw `[0.5, 99.5]` percentile range of each full channel.
- Trigger: Auto WB button (`main_window.py:1794`, `2906`).
- Idempotency: re-running Auto WB re-derives from `_preview_source` (pre-WB), so it is already
  deterministic per source (target formalizes and hardens this — §5.6).

---

## 5. Proposed target contracts

> These are *contract proposals*, not implementations. They document the desired truthfulness
> semantics so Jarvis can define the first implementation mission without guessing. Names in
> caps are the deliverable contract identifiers. This C3 revision incorporates all architect
> feedback; items 1–10 of the C1 review are mapped explicitly (markers `[A1]`…`[A10]`).

### 5.1 `EXPOSURE_METADATA_CONTRACT` (proposed)

**Core definition (mode-independent) `[A1]`:**
> An **accepted / contributing frame** is a frame whose *scientific contribution was
> successfully admitted to the final accumulator/batch*. For classic SUM/W this is a batch that
> passed the `_combine_batch_result` shape/coverage gates and was summed into the memmaps; for
> Drizzle M3 this is a frame for which `_add_frame_to_drizzle_accumulators` returned `True`.

1. **`NIMAGES` = number of accepted/contributing frames** (exact). It must count *successful
   admission*, never "aligned" or "attempted" frames:
   - classic batch failures (shape/coverage gate rejections),
   - failed alignment,
   - quality pre-rejection,
   - failed Drizzle accumulator adds (`_add_frame_to_drizzle_accumulators` → `False`),
   
   must **not** increase `NIMAGES`. Concretely, an accepted-frame counter **already exists** —
   `_drizzle_frame_count`, bumped in `_drizzle_group_tick()` (`18146-18148`) which the worker
   calls only inside the `if added:` branch (`6454-6465`). The fix is to **read that counter** in
   `_save_final_stack` for Drizzle instead of the `aligned_files_count` fallback (`15918-15922`),
   and to add a matching accepted-exposure sum in the *same* success path. `aligned_files_count`
   (`6424`) remains a pre-admission "attempted" counter and must not feed `NIMAGES`.
1b. **Counter/exposure placement (C2 — single atomic update):** every accepted-frame counter and
   accepted-exposure sum is updated **exactly once, only after successful scientific admission**,
   and never more than once per frame. The contract does **not** prescribe a single code
   location; either of these is acceptable **provided** the update is atomic with admission, is
   executed exactly once per admitted frame, and the same value feeds both `NIMAGES` and
   `TOTEXP` (no double count, no pre-admission increment):
   - *worker-side after `True`* — increment in the `if added:` branch (the existing
     `_drizzle_group_tick()` pattern, `6454-6465`); or
   - *accumulator-side atomic result* — the accumulator returns an atomic success result and the
     counter/exposure sum are derived from it.
2. **`TOTEXP` = exact nominal sum** `Σ EXPTIME` (seconds) over the *same* accepted frame set as
   `NIMAGES`, `round(..., 2)`, **when complete** (see point 5 for unknown). Classic and Drizzle
   must derive it from the same accepted-frame counter.
3. **One canonical metadata read site** for per-frame exposure: parse `EXPTIME` first, else
   `EXPOSURE`, else *unknown*. A value is **valid iff it is finite and `> 0`**; anything else
   (absent, non-numeric, `NaN`, `inf`, `<= 0`) is **unknown**.
4. **Scientific vs truthful separation `[A2]`:** the Drizzle accumulator's existing
   `float(header.get("EXPTIME", 1.0))` → sanitize → `in_units="counts"` (`18085-18098`) is a
   **scientific scaling fallback** for pixel math and is *left as-is* for the accumulator `add`.
   It must **not** feed `TOTEXP`. Metadata seconds for `TOTEXP` come from the canonical parser
   (point 3), which never fabricates `1.0` or `0.0`.
5. **Unknown exposure handling `[A2]`:** if *any* accepted contributor has unknown exposure,
   the scalar `TOTEXP` must be **omitted / marked unknown**, never written as `0.0` or a
   fabricated `1.0`. Track an explicit related counter — **`NEXPUNK`** (ratified; count of
   accepted contributors with unknown exposure). `NIMAGES` remains exact regardless (unknown
   exposure does not affect image count).
6. **Batching/resume composition `[A2]`:** the unknown count `NEXPUNK` (and the per-batch
   "any-unknown" flag) must compose across batches and survive the resume manifest
   (`12626-12698` / restore `13205-13238`), so a resumed run can still emit the correct
   known/unknown `TOTEXP` state. Keep `NIMAGES` and the known-exposure sum in lockstep with the
   same frame set.
7. **Final `EXPTIME` semantics `[A2]` (architect preference, adopted):** a final stack has no
   single per-frame exposure when inputs differ. Therefore:
   - **Set** the final `EXPTIME` keyword **only when all accepted per-frame exposures are known
     and uniform** (equal within rounding tolerance), with a clear comment such as
     `"per-input exposure; all accepted frames"`.
   - **Delete (not merely omit)** any `EXPTIME`/`EXPOSURE` keyword that was inherited through
     `current_stack_header.copy()` (`15928`) for **mixed or unknown** exposure — `_save_final_stack`
     must explicitly remove/replace stale per-input exposure keywords so they can never leak into
     `final.fits` (C2 strengthens C1's "omit" into an explicit `del`/replacement, required by the
     inheritance paths in §2.7/§4.1).
   - **Rationale:** `EXPTIME` is defined as a per-observation (per-input-frame) keyword. Writing a
     uniform value when inputs differ would be a fabricated scalar; writing the sum would
     duplicate `TOTEXP`. `TOTEXP` remains the exact nominal sum when complete, and is the correct
     keyword for total on-sky integration. (The standard classic SUM/W and M3 Drizzle baselines
     write no final `EXPTIME`; the single-classic-batch reproject/coadd path can inherit one —
     §2.7/§4.1 — which is why the delete step above is mandatory.)
8. **Nominal integrated exposure limitation `[A1]`:** `TOTEXP` is a *scalar nominal* sum of
   per-frame metadata exposures. Pixel-local rejection/coverage (Drizzle per-pixel weight,
   classic coverage map, quality weighting) modulates *which pixels* receive contribution and is
   deliberately **not** promoted into a local/per-pixel exposure map. No new per-pixel exposure
   product is introduced by this contract.

**Alternatives rejected:**
- *Keep the per-batch `Σ EXPTIME` then re-sum in `total_exposure_seconds`* — rejected because it
  double-depends on header completeness and the single-image branch already diverges
  (`EXPTIME→EXPOSURE→0.0`).
- *Leave Drizzle `TOTEXP=0.0` as "unknown"* — rejected because the GUI summary reads `TOTEXP`
  verbatim and would display a false `0 s`; the contract instead writes the real nominal sum when
  complete, else omits `TOTEXP` and lets the UI surface `NEXPUNK`.
- *Reuse the scientific `1.0` fallback for metadata* — rejected: it would fabricate exposure and
  silently corrupt `TOTEXP`.
- *Move exposure to HSI/registration math* — rejected (out of scope; exposure is a metadata
  counter, not a pixel quantity).

### 5.2 `CURRENT_PREVIEW_AUDIT` (proposed → frozen baseline)

Document §4.2 as the frozen baseline; the contract records, as invariants:
- Display-only, never feeds science.
- `_preview_source` pristine; derived images are copies.
- Backend preview data domain is `float32 [0,1]` (HWC or HxW).
- Known deviations to resolve: 8-bit quantization at render boundary (`_to_uint8`), and the
  non-retention of the float `[0,1]` source needed by §5.3/§5.5/§5.6, plus the **per-preview
  normalization remapping** (§2.2) that makes the same physical pixel change value as the stack
  evolves.

**Normalization stabilization (C3 — ratified single architecture: Option A only).** The target
separates four stages:
1. **scientific current stack** (SUM/WHT memmaps or Drizzle accumulators — mutating),
2. **immutable linear preview representation** (the float average/`divide` result, before any
   WB/stretch),
3. **WB-only analysis domain** (the representation after applying authoritative WB gains —
   histogram/stats/Auto Stretch/BP/WP live here),
4. **display transform** (WB → stretch → gamma → BCS → uint8 `QImage` → view).

**Ratified Option A — backend carries two arrays; Qt owns the mapping.** There is exactly one
preview architecture; no alternative remains.

- **Payload shape (all producers):** the preview callback sends a **tuple of two arrays**
  `(legacy_normalized, raw_linear)`:
  - **`legacy_normalized` (first)** — the existing normalized display array, retained **for
    callback/backward compatibility** with legacy consumers. Existing legacy renderers may still
    choose the first element, but production Qt must not use it for the live image once the
    fixed-anchor path is available (otherwise per-preview pumping remains).
  - **`raw_linear` (second)** — a new **immutable raw-linear preview array**, downsampled/capped
    to the **same geometry as `legacy_normalized`** before the callback, carrying the true linear
    float source for analysis.
- **`raw_linear` per mode:**
  - **classic SUM/W** — the `SUM / WHT` divide preview representation **after any explicitly
    display-only preview masks** (feathering / low-WHT mask), **before** the min/max
    normalization.
  - **M3 Drizzle** — the `finalize("divide")` HWC stack, **before** the 1%/99% percentile
    normalization.
- **Qt owns the mapping; backend never stores GUI display state.** Qt extracts/copies the second
  `raw_linear` array (never mutating the backend or science data). On the **first valid preview
  of a run**, Qt computes **fixed normalization anchors** from a **deterministic finite positive
  sample** of `raw_linear`: `anchor_lo = percentile(sample, 0.5)`, `anchor_hi =
  percentile(sample, 99.5)`; **fall back to finite min/max only when the percentile sample is
  degenerate** (empty, non-finite, or `anchor_hi ≤ anchor_lo + sep`), with separation epsilon
  `sep = 1e-4`.
- **Anchors are reset only at run start / a new initial preview context**, never on every
  preview update. Qt maps every later raw preview through the **same** anchors into the pristine
  normalized `[0,1]` analysis source:
  `mapped = clip((raw_linear − anchor_lo) / (anchor_hi − anchor_lo), 0, 1)`.
  Consequently **the same unchanged linear pixel maps identically** across successive previews.
- **Production Qt builds `_preview_source` from `mapped`, not from `legacy_normalized`.** It
  converts a copy of `mapped` to `QImage`, then applies WB → stretch → gamma → BCS and the
  existing view transforms. The `QImage`/`render_view` machinery is preserved while its input
  uses the fixed mapping; the visible preview and the float analysis domain therefore cannot
  silently follow the backend's per-update min/max or percentile remap.
- **Anchors are display-analysis state only.** Manual Stretch, Auto Stretch, and Auto WB operate
  on the mapped `[0,1]` analysis source and **never change the anchors**. Scientific accumulators
  (SUM/WHT memmaps, Drizzle accumulators) are untouched.
- **Downsampling / memory bounds.** `raw_linear` is downsampled **before** the callback to bound
  memory: the Drizzle `_MAX_PREVIEW_SIDE_PX = 1000` cap and the configured GUI downsample factor
  (1..4, default **2**) are retained and applied to `raw_linear` at the **same geometry** as
  `legacy_normalized`; the **2× default downsample is explicitly covered** (both arrays are
  produced at the same final preview resolution, and percentile-anchor sampling is
  **deterministic and capped** via a fixed stride over the finite positive sample).

**Regression test (C3):** successive previews of an evolving stack must **not** remap a fixed
reference pixel — a pixel whose linear value does not change must keep the same `[0,1]` analysis
value across previews (only its *contribution* changes via WB, never the reference mapping). See
§7 step 7b.

**Alternatives rejected:** moving WB/stretch into the backend (rejected — breaks the
display/backend separation and the "display never feeds stacking" invariant); rendering directly
from float in Qt (rejected as a widget redesign, out of scope for the first mission);
re-normalizing on every preview (rejected — the defect this contract removes); backend-side
normalization (backend sends normalized + fixed metadata) — rejected in C3: it puts the reference
mapping in the backend, which must then be communicated back to the GUI and duplicated in the
summary, blurring the backend/GUI boundary that the "display never feeds stacking" invariant
depends on.

### 5.3 `HISTOGRAM_CONTRACT` (proposed) `[A5][A7]`

> **C2 domain decision (reverses C1):** the analysis domain is **WB-only pre-stretch**, *not*
> pre-WB. Rationale: the displayed image is WB → stretch → gamma → BCS, and BP/WP are stretch
> controls that operate on the WB-only image; a pre-WB histogram would plot a domain that never
> matches what the user sees, and it would still move under AutoWB. The property C1 wanted
> (stability while dragging stretch controls) is satisfied by the **WB-only pre-stretch** domain,
> because stretch is applied *after* it.

1. **Source:** two **parallel immutable preview-analysis buffers** retained alongside
   `_preview_source` (the `QImage` used for the view pipeline): (a) the **pristine pre-WB linear
   source** (for AutoWB only), and (b) the **derived WB-only linear `[0,1]` analysis buffer**,
   re-derived from (a) using the authoritative WB gains whenever WB changes. The `QImage`
   rendering/view pipeline is preserved, but `_preview_source` is constructed from the fixed-
   anchor mapped float source rather than `legacy_normalized`; the float buffers also feed the
   histogram and auto computations.
2. **Domain:** histogram/stats/Auto Stretch/BP/WP are all computed over the **derived WB-only
   linear `[0,1]` pre-stretch** buffer. RGB distributions/statistics are computed **per channel**
   on that same `[0,1]` domain, and are plotted as an **RGB overlay** (the three channel
   histograms drawn on the same axes). WB is applied *once* to produce the analysis buffer; it is
   not part of the plotted display transform for histogram purposes. The pristine pre-WB buffer
   is used **only** by AutoWB (§5.6).
3. **Bins:** **`512` bins over `[0,1]` (float domain)** — ratified, no alternative.
4. **Sampling:** deterministic sampling **cap/stride** over the analysis buffer to bound cost on
   large previews (e.g. sample at most `N` pixels via a fixed stride computed from the buffer
   size, deterministic and documented). Statistics must be computed on the **same sampled set**
   used for the plotted histogram.
5. **Stats (plotted domain `[0,1]`):** per channel report **`min`, `max`, `median`, `mean`,
   `std`** (all five, no optionals) in the plotted `[0,1]` domain, computed on the **same WB-only
   sampled float domain** as the histogram. Use `log1p` on the Y axis for the bar heights (count
   visualization in log space, preserving empty-bin = 0 behaviour).
6. **X range:** robust auto X range (e.g. percentile-based on the finite sample, guarded against
   empty/degenerate samples) **plus** an explicit full-range `[0,1]` toggle.
7. **Recompute triggers (C3):** recompute the analysis buffer **only on a new raw source** (a new
   `raw_linear` preview, i.e. a new analysis source) **or on a WB change** (which re-derives the
   WB-only buffer). Do **not** recompute on BP/WP/stretch/gamma/BCS/zoom/pan/rotation mouse moves
   — those only re-render the `QImage` view and re-draw the already-computed histogram (BP/WP
   line positions are just marks on the same underlying histogram). Note: BP/WP and stretch
   changes do **not** change the WB-only analysis buffer (stretch is applied *after* the analysis
   domain).
8. **Threading/staleness (§5.8):** histogram analysis runs off the GUI thread (or bounded),
   guarded by a generation/version token.

**Alternatives rejected:**
- *Histogram from the fully stretched image* — rejected (would churn the histogram when the user
  drags BP/WP and break the stable-domain requirement).
- *256 bins over uint8 forever* — rejected (8-bit binning loses tonal fidelity that the backend
  already carries as float32; the "truthfulness" goal requires binning the true `[0,1]` domain).
- *Pre-WB histogram source (C1's choice)* — rejected in C2: a pre-WB domain never matches the
  displayed WB→stretch image and still moves under AutoWB; the WB-only pre-stretch domain is what
  the user actually controls with BP/WP. The *current* WB-only `QImage` behaviour (§4.3) is the
  correct *domain*; only the 8-bit quantization and the recompute triggers change.
- *Recompute on every slider tick* — rejected (unnecessary; only a new analysis source or a WB
  change alters the analysis buffer).

### 5.4 `DISPLAY_STATE_CONTRACT` (proposed)

1. `MainWindow` remains the single authoritative owner of WB/stretch/BP/WP/gamma/BCS + view
   transform state. `HistogramView` holds **derived** BP/WP positions that are reconciled to the
   sliders (current `set_range` already does this).
2. Every refresh path recomputes **once** from `_preview_source`; eliminate the double-WB and
   triple-`_image_to_array` redundancy by computing `wb_only` once and reusing it (both for the
   histogram and as the pre-stretch input to `apply_preview_adjustments`, which should accept an
   already-WB'd buffer or skip re-applying WB). `wb_only` here is the §5.3 WB-only analysis
   buffer; histogram stats derive from it (float), not from a fresh `QImage` conversion.
   `_preview_source` itself is the `QImage` copy constructed from the fixed-anchor mapped raw
   source (§5.2), never from the per-preview `legacy_normalized` array.
3. No state is stored in the engine; display state is never serialized into FITS.

**Alternatives rejected:** centralizing display state in a shared `QtSettingsState` (rejected —
  the existing `_on_*` handlers already own it and the redundancy fix is local); a separate
  "PreviewManager" port (rejected — larger refactor than the truthfulness mission requires).

### 5.5 `AUTO_STRETCH_CONTRACT` (proposed) `[A4]`

1. **Domain:** compute BP/WP **directly in the normalized `[0,1]` control domain** from the §5.3
   **WB-only mapped float** analysis buffer. The current full-image absolute min/max
   normalization (`preview_adjust.py:417-425`) is **removed**; **no min/max renormalization
   exists** — the input is already the pristine `[0,1]` anchor mapping (§5.2).
2. **Exact algorithm (C3 — single ratified spec, background-population):**
   1. **Input `S`** = the WB-only mapped float `[0,1]` values; keep pixels that are **finite**,
      have `mask[pixel] > 0` when a weight/validity mask is available, and **exclude exact
      clipped values** `0.0` and `1.0`.
   2. If `|S| < 20`, return the deterministic neutral defaults `(0.01, 0.99)`.
   3. `p005 = percentile(S, 0.5)`, `p60 = percentile(S, 60)`, `p995 = percentile(S, 99.5)`.
   4. **Background population** `B = { s ∈ S : s ≤ p60 }`; `bg = median(B)`;
      `sigma = 1.4826 · MAD(B)` (MAD = median of absolute deviations from `bg`).
   5. `bp = clip(max(p005, bg − 2.8·sigma), 0, 1 − sep)`.
   6. `wp = clip(max(p995, bg + 8·sigma, bp + sep), bp + sep, 1)`.
   7. `sep = 1e-4`. **Deterministic degenerate fallback:** if `B` is empty/non-finite or
      `sigma == 0`, set `bp = p005`, `wp = p995`, then clamp/separate as in 5–6.
   8. **Constants:** percentiles `0.5 / 60 / 99.5`, black-point background spread `2.8`, white-point
      background spread `8`, MAD scale `1.4826`, `sep = 1e-4`, `min sample = 20`, fallback
      `(0.01, 0.99)`. **No full-image min/max renormalization step exists.**
3. **Robust intent:** `bp` is driven by the low tail (`p005`) but **pulled up toward the
   background** (`bg − 2.8σ`), so a faint background pedestal is not clipped to black; `wp` is
   driven by the bright tail (`p995`) but **never below the background + 8σ** (so a low-contrast
   frame still stretches); the `60`-percentile split defines the background population robustly
   against a bright minority, effectively rejecting the upper ~40% (stars/emission) from the
   background estimate. Because `B` is a median/MAD statistic over a lower-half population, the
   result is **insensitive to outlier pixels** (bright stars or hot pixels do not move `bg`/`σ`)
   and **repeatable** for a fixed input (deterministic percentiles and median/MAD).
4. **Lifecycle:**
   - The **first valid preview may initialize BP/WP once** (auto-initialization).
   - Thereafter BP/WP parameters **remain stable across backend preview updates** (a new preview
     does not silently re-run auto-stretch).
   - **Explicit Auto Stretch recomputes** from the current analysis source.
   - **Manual adjustment locks/preserves state** (user drags/spins are not overwritten by any
     automatic recalculation).
   - **No silent continual recalculation** on any non-explicit trigger.
5. Trigger: Auto Stretch button; side effect: set stretch mode to `asinh` (unchanged).

**Alternatives rejected:** the C2 `1%/99%` percentile pair plus a *separate* spread guard
(rejected — two competing mechanisms with no single spec); computing from the
stretched image (rejected — would double-apply stretch); retaining the full-image min/max
normalization (rejected — non-robust to outliers and moves the control domain); any
"percentiles *or* background-MAD" two-branch recipe (rejected — C3 specifies exactly the single
background-population algorithm above).

### 5.6 `AUTO_WB_CONTRACT` (proposed) `[A3]`

1. **Input:** estimate gains from the **pristine pre-WB mapped float `[0,1]`** source (the §5.3
   pristine pre-WB linear source, i.e. the anchored `[0,1]` mapping before WB is applied), so
   **repeated explicit AutoWB is deterministic/idempotent**. **Do not** estimate from already-WB
   pixels.
2. **Exact robust sample (C3 — single ratified spec, true background band):**
   1. Take the **common finite RGB set**: pixels where all three channels are finite and the
      array has ≥3 channels (else neutral `(1,1,1)`).
   2. **Every channel strictly positive and below saturation:** require `0 < R, G, B < 0.98`
      (excludes zero/dark borders and near-saturated/clipped stars).
   3. **True background luminance band:** `lum = 0.299·R + 0.587·G + 0.114·B` over the surviving
      pixels; keep only pixels with `p5 ≤ lum ≤ p60`, where `p5 = percentile(lum, 5)` and
      `p60 = percentile(lum, 60)` (guarded against degenerate `p60 ≤ p5`).
   4. **Per-channel centre = median of the exact same selected pixels:** for each channel compute
      `centre_ch = median(channel[selected])` on the **same** band-selected pixel set — no
      per-channel re-selection, no histogram mode.
   5. **Gains relative green, clipped:** `gain_r = clip(centre_g / centre_r, 0.2, 5.0)`,
      `gain_g = 1.0`, `gain_b = clip(centre_g / centre_b, 0.2, 5.0)`; guard: any
      `centre_ch ≤ 1e-6` → neutral `(1.0, 1.0, 1.0)`.
   6. **Fallback:** if fewer than **64** valid pixels remain after exclusion, return neutral
      `(1.0, 1.0, 1.0)`.
   7. **Deterministic sampling cap** (shared with the histogram when one is used) bounds the
      sample before the percentile/median computations.
   8. **Constants:** channel validity `(0.0, 0.98)`, luminance band `p5`/`p60`, R/B clip
      `[0.2, 5.0]`, centre floor `1e-6`, min sample `64`, neutral `(1.0, 1.0, 1.0)`.
3. **Robust intent:** the `p5..p60` band deliberately rejects saturated stars, zero/dark borders,
   and the upper ~40% bright/emission population, so gains are estimated from the **true
   background** — the neutral-grey sky pedestal — rather than from bright colored emission. The
   per-channel **median** (not mode) over the exact same pixels gives a stable centre that is
   insensitive to outliers in each channel.
4. Exclusions unchanged: non-color / <3 channels / missing data → neutral `(1,1,1)`.
5. **Display-only:** gains feed the WB display transform only; never serialized or fed to science.

**Alternatives rejected:** Auto WB from the stretched image (rejected — stretch would bias the
centre estimate); Auto WB from the WB-only image (rejected — violates the idempotent pre-WB requirement);
resetting WB to neutral before computing (rejected — wasteful; compute from pre-WB directly);
the C2 broad `[0.5, 99.5]` luminance band + per-channel 256-bin histogram **mode** (rejected in
C3 — the wide band admits saturated stars and bright emission, and a per-channel histogram mode is
noisier and non-deterministic at the band boundary); "prefer a neutral/background sample" as an
unspecified recipe (rejected — C3 specifies the exact `p5..p60` band + per-channel median above).

### 5.7 Live drag `[A6]`

- `HistogramView` emits **live/coalesced** BP/WP state **during** drag at a target cadence of
  **~25 ms** (≈40 Hz), with a **final emit on release** (authoritative value).
- The live emit triggers a preview re-render **from the cached `_preview_source` only** (the
  already-derived stretch path), and **no histogram recompute, no FITS read, no stacking work**.
- Coalescing: intermediate emits may be dropped/timer-coalesced so the GUI thread is not flooded;
  the release emit is never dropped.

### 5.8 Performance / threading `[A7]`

- Histogram analysis (binning + stats + auto estimators) runs **off the GUI thread** (worker) or
  is **bounded** so the GUI does not block.
- A **generation/version token** is attached to each analysis request; stale results (older token)
  are **discarded** and never overwrite a newer preview's histogram.
- `QImage` creation, widget updates, and `paintEvent`-visible state changes occur **only on the
  GUI thread** (results marshalled back via signal/slot).

### 5.9 Histogram worker lifecycle (H2) — bounded latest-wins

Implemented in `seestar/gui_qt/histogram_worker.py` (coordinator) and wired into
`MainWindow` (`_refresh_histogram_float` schedules; `_on_histogram_result` applies).

**Bounded latest-wins.** A single long-lived worker `QThread` runs the authoritative
`compute_histogram_float`. The GUI-thread `HistogramCoordinator` enforces:

* at most **one** computation running at any time;
* at most **one** pending request retained;
* a newer request **replaces** the older pending request (coalescing) — there is no unbounded
  QThread/QThreadPool queue under rapid preview/WB updates.

**Generation token.** Every `schedule()` assigns the next explicit monotonic generation token;
`invalidate()` (reset / new context / shutdown) also bumps it. A worker result is applied only
when its token still equals the coordinator's newest token; otherwise it is **discarded** and the
latest pending request runs next. Lifecycle: `source generation N -> request N -> worker result N
-> GUI-thread result slot -> generation/source check -> apply or discard`; if `N+1` arrives before
`N` completes, `N` is discarded and can never overwrite `N+1`.

**GUI-thread-only presentation.** The worker only *reads* the analysis buffer and never touches
widgets; results travel back via a queued signal. `MainWindow._on_histogram_result` (GUI thread)
re-checks the source token `(analysis_generation, wb_only_revision)` against the current state and
the `_shutting_down` flag before applying the model/status to `HistogramView`.

**Recompute contract.** A request is scheduled only when the WB-only buffer is re-derived (a new
raw source or an actual WB change) — never on BP/WP / stretch / gamma / BCS / zoom / pan /
rotation / Auto Stretch. Manual WB and AutoWB/WB-reset schedule exactly one request when the WB
actually changes, zero when idempotent.

**Input ownership.** The WB-only float64 analysis buffer is shared with the worker by reference,
not copied: `MainWindow` *replaces* (never mutates) it on re-derivation, and
`compute_histogram_float` is pure/read-only, so the shared buffer is immutable-by-convention for
the duration of the request. `_pristine_float`, `_raw_linear`, the raw payload arrays and the
scientific buffers are never mutated.

**Clean shutdown.** `MainWindow.shutdown()` invalidates the coordinator, disconnects the result
channel, `quit()`s and `wait()`s the worker thread, and joins it with the run controller; a
retryable `False` return retains the thread references so a still-running thread is never
destroyed. No widget update can occur after shutdown completes.

**Instrumentation seams.** The coordinator exposes `requests_scheduled` (generations requested),
`jobs_started` (actual worker computations), `pending_replaced` (coalesced pending),
`stale_discarded` (results discarded), `latest_applied` (latest results applied) and
`last_latency_ms`/`max_latency_ms`/`recent_latencies` (latency). Tests inject a slow/controlled
compute callable via `MainWindow(histogram_compute_fn=...)` — no monkeypatching of global
scientific state.

---

## 6. Unresolved risks

- **R1 — Drizzle `TOTEXP=0.0` and `NIMAGES` inflation are user-visible today.** Both surface in
  `summary_payload.py:110-111`. Fixing requires (a) incrementing an accepted-frame exposure sum
  in the M3 Drizzle path *on successful add only*, and (b) making `_save_final_stack` read the
  **already-correct** `_drizzle_frame_count` (bumped in `_drizzle_group_tick()` inside
  `if added:`, `6454-6465`/`18146-18148`) instead of the `aligned_files_count` fallback
  (`15918-15922`). This touches `_add_frame_to_drizzle_accumulators`/`_worker`/
  `_save_final_stack`, which is science-adjacent (pixels stay untouched; only the counters
  change). Needs Jarvis sign-off that the Drizzle `exptime` in seconds of the original frame
  (`EXPTIME`) is the correct per-frame exposure for `TOTEXP`, and that the accepted counter/sum
  update happens exactly once (worker-side `if added:` or accumulator-side atomic result — §5.1
  point 1b).
- **R2 — Float analysis buffer without changing the view pipeline.** `_preview_source` is today a
  `QImage` used by `render_view`. The target adds parallel float `[0,1]` buffers (pristine
  pre-WB + derived WB-only) retained alongside it; the `QImage` pipeline stays untouched.
  **Ratified (§5.2):** retain the `raw_linear` float ndarray from the payload tuple *before*
  `_to_uint8` quantization; **Qt owns the fixed anchor mapping and constructs production
  `_preview_source` from the mapped raw copy**. `legacy_normalized` remains compatibility data
  only — Option A, no backend-side mapping.
- **R3 — `rangeChanged` during drag** changes interaction frequency; risk of redundant full
  refreshes on drag. **Ratified (§5.7):** coalesce with a ~25 ms timer; final emit on release;
  live rerender uses cached `_preview_source` only.
- **R4 — Missing `EXPTIME`/`EXPOSURE` semantics.** The contract now mandates: valid iff
  finite `>0`; else unknown; any unknown among accepted frames ⇒ omit `TOTEXP` + track `NEXPUNK`.
  A silent `0.0`/`1.0` in `TOTEXP` is eliminated. `NIMAGES` stays exact. **Ratified (§11.2):** the
  keyword is `NEXPUNK`; its resume-manifest composition follows §5.1 point 6.
- **R5 — Resume manifest carries two exposure copies** (`total_exposure_seconds` and
  `cumulative_header["TOTEXP"]`). They are written from the same value today, but a future fix
  must keep them in lockstep (and now also carry `NEXPUNK`/known-unknown state) or reduce to a
  single source.
- **R6 — Dead `_update_header_for_drizzle_final`** (`queue_manager.py:7760-7859`) is never called
  and misleads future readers about the Drizzle header path. **Per this mission's scope it is
  documented as dead and left untouched** — no removal, no edits, to avoid unrelated cleanup
  (`[A10]`). Flagged for a *separate*, explicitly authorized cleanup mission if Jarvis wants it
  removed later.
- **R7 — Per-preview normalization remapping (C2 defect #2).** Classic (`_update_preview_sum_w`,
  min/max) and Drizzle (`_update_preview_drizzle_accumulator`, 1%/99%) re-derive their reference
  points on every preview, so the same physical pixel changes `[0,1]` value as the stack evolves.
  **Ratified resolution (§5.2):** backend carries a two-element tuple `(legacy_normalized,
  raw_linear)`; Qt freezes the anchor mapping once per run/first valid preview from `raw_linear`
  (p0.5/p99.5, finite min/max fallback only when degenerate) and maps every later preview through
  the same anchors — Option A, unified across classic/Drizzle.
- **R8 — Final `EXPTIME` inheritance (C2 correction of C1's blanket claim).** `_save_final_stack`
  copies `current_stack_header` (`15928`); single-classic-batch finalization
  (`_finalize_single_classic_batch` `15263`) sets `current_stack_header = hdr.copy()` from a batch
  header that carries `EXPTIME`, so a stale per-frame `EXPTIME` can reach `final.fits`. Resolution
  (§5.1 point 7): final save must explicitly set uniform `EXPTIME` or delete inherited
  `EXPTIME`/`EXPOSURE` for mixed/unknown.

---

## 7. Minimal implementation decomposition

> Decomposed into four independent workstreams per architect feedback `[A8]`. Ordered within each
> workstream by risk/reversibility; each step is independently verifiable and leaves science
> pixels untouched. This is a *plan*, not implementation.

### A. Exposure truthfulness (implementation + tests)

1. **Canonical exposure parser** `_frame_exposure_seconds(header) -> float | None`: `EXPTIME` →
   `EXPOSURE` → `None`; valid iff finite and `> 0`. Route the classic batch sum (`11944-11956`),
   the single-image branch (`11570`), and the metadata side of the Drizzle path (`18085`) through
   it. Keep the scientific `1.0` fallback *inside* `_add_frame_to_drizzle_accumulators` for the
   accumulator `add` (`in_units="counts"`) only.
2. **Accepted-frame counter/exposure placement (Drizzle).** Reuse the existing accepted-frame
   counter `_drizzle_frame_count` (bumped in `_drizzle_group_tick()` only inside the `if added:`
   branch, `6454-6465`/`18146-18148`) and add an accepted-exposure sum updated in the **same**
   success path (worker-side after `True` **or** accumulator-side atomic result — §5.1 point 1b).
   Update `_save_final_stack` to read `_drizzle_frame_count` (and the accepted sum) for Drizzle
   instead of the `aligned_files_count` fallback (`15918-15922`). Do **not** increment anything at
   the pre-add `aligned_files_count += 1` (`6424`). Classic
   `images_in_cumulative_stack`/`total_exposure_seconds` unchanged.
3. **`TOTEXP`/`NEXPUNK`/final-`EXPTIME` semantics.** Write `TOTEXP = Σ known` only when all
   accepted exposures are known; else omit `TOTEXP` and write `NEXPUNK`. Set final `EXPTIME` only
   when all accepted exposures are known *and uniform*, with the per-input comment; **else
   explicitly delete any inherited `EXPTIME`/`EXPOSURE`** in `_save_final_stack` (§5.1 point 7).
   Extend the resume manifest to persist the known/unknown state and `NEXPUNK`.
4. **Tests (A):** fresh Drizzle run → `NIMAGES == _drizzle_frame_count` (a forced failed add must
   not inflate `NIMAGES`), `TOTEXP == Σ EXPTIME` when complete, `TOTEXP` omitted + `NEXPUNK`
   present when any exposure unknown, final `EXPTIME` present only when uniform **and absent
   (deleted) when mixed/unknown even on single-classic-batch paths**, classic unchanged,
   `tests/test_resume.py` still green.

### B. Pure preview analysis / display-model helpers (implementation + tests)

5. **Analysis-buffer retention (Option A).** Backend sends a two-element payload tuple
   `(legacy_normalized, raw_linear)`; Qt retains the **`raw_linear`** element as the pristine
   pre-WB linear source (before `_to_uint8` quantization, never mutating backend/science), and
   derives the WB-only linear `[0,1]` analysis buffer on WB change. Qt computes and freezes the
   normalization anchors once per run/first valid preview (p0.5/p99.5; finite min/max fallback
   only when degenerate) and maps every later `raw_linear` through them (§5.2).
6. **Pure float helpers** in `preview_adjust.py` (or a new analysis module), domain `[0,1]`,
   **WB-only pre-stretch**: `compute_histogram_float` (512 bins, `[0,1]`, per-channel, `log1p` Y),
   `compute_histogram_stats_float` (min/max/median/mean/std), `compute_auto_stretch_float`
   (exact §5.5 background-population algorithm, **no min/max normalization**, deterministic min
   separation), `compute_auto_wb_float` (pre-WB pristine, exact §5.6 true-background-band
   algorithm). Deterministic sampling cap/stride shared by histogram + stats.
7. **Tests (B):** pure-function unit tests — bin domain/512, per-channel stats, sampling
   determinism, auto-stretch **outlier insensitivity** (bright stars/hot pixels do not move
   `bg`/`σ` or BP/WP) and **repeatability** (same input ⇒ same BP/WP), and auto-WB tests for
   **known scaling** (grey pedestal ⇒ near-neutral gains), **saturation** (saturated stars
   excluded), **zero/dark borders** (excluded), **strong colored bright emission** (excluded from
   the `p5..p60` band), and **idempotence** (re-run on same source ⇒ same gains).
7b. **Tests (B, normalization):** successive previews of an evolving stack must **not remap a
   fixed reference pixel** — a pixel whose linear value is unchanged keeps the same `[0,1]`
   analysis value across previews (the anchor mapping is frozen at first valid preview; Option A).

### C. Qt integration / interactions / tests / performance

8. **De-duplicate refresh path.** Compute `wb_only` once per refresh; reuse for histogram + as
   pre-stretch input; compute stats from the §5.3 buffer (not triple `_image_to_array`).
9. **Recompute triggers (C3).** Recompute the analysis buffer/histogram only on a **new raw
   source** (`raw_linear` change) **or a WB change**; on BP/WP/stretch/gamma/BCS/zoom/pan/rotation
   moves only re-render the `QImage` view (BP/WP line marks move on the same histogram; stretch
   is applied *after* the analysis domain and does not change it).
10. **Live drag.** Emit live/coalesced `rangeChanged` (~25 ms) during drag + final emit on
    release; rerender preview from cached `_preview_source` only (no histogram recompute / FITS
    read / stack).
11. **Threading/staleness.** Move histogram analysis off the GUI thread (or bound it); generation/
    version token discards stale results; `QImage`/widget updates on GUI thread only.
12. **Tests (C):** offscreen Qt tests for drag emission cadence/release finality, histogram
    recompute-on-source-only, stale-result discard, and a perf smoke (refresh under drag does not
    exceed a bounded latency).

### D. Real witnesses + scientific separation

13. **Witness fixtures** (real or synthetic FITS) exercising every path: classic multi/single
    batch, Drizzle success + forced failed add, missing/non-numeric/`<=0` exposure, mixed exposure
    uniformity, resume restore. Assert the *final header* output (the witness), not just internal
    counters.
14. **Scientific separation proof.** Assert that the display/adjustment pipeline never mutates
    SUM/WHT memmaps or Drizzle accumulators (regression guard), and that the exposure counters do
    not change pixel math (bit-exact pixel output before/after the metadata change).
15. **Dead code:** document `_update_header_for_drizzle_final` as dead (§2.10, §6-R6) — **no
    removal in this mission** (`[A10]`).

---

## 8. Focused test seams (identify only — do NOT implement here)

- **Histogram / preview adjustments:** `seestar/gui_qt/preview_adjust.py` —
  `compute_histogram`, `compute_histogram_percentile`, `compute_auto_stretch`, `compute_auto_wb`,
  `apply_preview_wb`, `apply_preview_adjustments`. Existing coverage:
  `tests/test_qt_histogram_m14.py`, `tests/test_qt_preview.py`, `tests/test_auto_stretch.py`.
- **Histogram widget interactions:** `seestar/gui_qt/histogram_view.py` —
  `set_data`/`set_range`/`_drag_at`/`_end_drag`/`mouseReleaseEvent`/`rangeChanged`. Covered by
  `tests/test_qt_histogram_m14.py` (offscreen).
- **Preview render boundary:** `seestar/gui_qt/preview_render.py` — `_to_uint8`,
  `render_preview_image` (the 8-bit quantization seam). Covered indirectly by
  `tests/test_qt_preview.py`.
- **Exposure flow:** `seestar/queuep/queue_manager.py` —
  `_stack_batch` (11570, 11944-11956), `_combine_batch_result` (10600, 10620, 10756, 10830),
  `_worker` Drizzle add order (6424/6447/6468), `_add_frame_to_drizzle_accumulators` (18085),
  `_save_final_stack` (15918-15948), `_write_resume_manifest` (12626-12698). Existing coverage:
  `tests/test_resume.py` (125 tests), `tests/test_single_batch_csv.py`,
  `tests/test_save_final_stack.py`, `tests/test_drizzle_integration_qm.py`.
- **GUI summary exposure display:** `seestar/gui_qt/summary_payload.py` — `build_summary_payload`
  reads `TOTEXP` (`110-111`). Pure-stdlib, easy to unit-test with a synthetic `final.fits`.

Likely *new* test files (future mission, not now): a `test_exposure_metadata_*.py` for the
Drizzle `TOTEXP`/`NIMAGES`/`NEXPUNK` regression and read-site consolidation; updates to
`tests/test_qt_histogram_m14.py` for float-domain bins, drag-emission frequency, and
source-only recompute; and a `test_preview_normalization_stability.py` (or equivalent) asserting
that successive previews do **not** remap a fixed reference pixel (§5.2 regression, §7 step 7b).

---

## 9. Exact commands / tests run — baseline evidence only

> `[A9]` These results are **baseline observations only**. They prove the *current* behaviour is
> exercised by the existing suite; they do **not** prove the target contracts, which are not yet
> implemented. The target acceptance is defined by §7 per-step acceptance (Tests A/B/C/D + 7b),
> which require new/updated tests.

```bash
# repo state (read-only)
cd /home/tristan/.openclaw/workspace/projects/zeseestarstacker
git status
git branch --show-current
git rev-parse HEAD
git log --oneline -5

# static evidence (read-only greps) — see §2 line references

# focused existing tests (read-only, offscreen)
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_qt_histogram_m14.py tests/test_qt_preview.py -q -p no:cacheprovider
#   → 45 passed, 1 warning (AstropyDeprecationWarning in reproject_utils) in 11.40s

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_resume.py -q -p no:cacheprovider
#   → 125 passed, 4 warnings in 13.93s

# NOTE: tests/test_qt_last_stack_resume_m23.py fails at COLLECTION (not run):
#   ModuleNotFoundError: No module named 'seestar.gui.run_config'
#   (pre-existing, unrelated to this audit — see §10)
```

---

## 10. Git state / diff summary

- **Branch:** `feature/output-truthfulness-preview-ux` (unchanged).
- **HEAD:** `a7fc8d6a6c16f4c5d724152ebcc16255cac78093` (unchanged, == `origin/beta`).
- **Commit created:** NO. **Push performed:** NO. **Version/branch/merge/tag changes:** NONE.
- **Files changed:** exactly one — `docs/output_truthfulness_preview_audit.md` (new).
- **Worktree:** clean before this mission; after writing the doc, only the new doc is untracked.

```bash
$ git status --short
?? docs/output_truthfulness_preview_audit.md
```

- **Pre-existing (observed, not caused by this audit):**
  `tests/test_qt_last_stack_resume_m23.py` fails to collect
  (`ModuleNotFoundError: No module named 'seestar.gui.run_config'`). Left untouched; flagged for
  Jarvis as an unrelated pre-existing test-collection issue.

---

## 11. Ratified architect decisions

> These are the final ratified decisions (C3). There is no open architectural choice remaining.

1. **Drizzle `TOTEXP` semantics (ratified).** Drizzle `TOTEXP` = `Σ EXPTIME` (seconds) of
   *accepted* frames; the accepted counter/sum updates **exactly once after successful admission**
   (worker-side `if added:` or accumulator-side atomic result — §5.1 point 1b); `_save_final_stack`
   reads the existing `_drizzle_frame_count` instead of the `aligned_files_count` fallback
   (`15918-15922`); the pre-add `aligned_files_count += 1` (`6424`) never feeds `NIMAGES`
   (pixels untouched). `[A1]`
2. **Missing-exposure semantics (ratified).** `EXPTIME`→`EXPOSURE`, valid iff finite `>0`, else
   unknown; any unknown among accepted frames ⇒ omit scalar `TOTEXP` + track **`NEXPUNK`**;
   `NIMAGES` stays exact; resume-manifest composition of the known/unknown state per §5.1 point 6.
   `[A2]`
3. **Final `EXPTIME` semantics (ratified).** Set final `EXPTIME` only when all accepted exposures
   are known and uniform (with the per-input comment); **explicitly delete** any inherited
   `EXPTIME`/`EXPOSURE` for mixed/unknown, per the inheritance paths in §2.7/§4.1. `[A2]`
4. **Float analysis domain (ratified).** Analysis domain = **WB-only pre-stretch**
   (histogram/stats/Auto Stretch/BP/WP), with a separate pristine pre-WB buffer used only by
   AutoWB; **512 bins** over `[0,1]`, per-channel **RGB overlay**, `log1p` counts, full stats
   (`min/max/median/mean/std`); `QImage` view pipeline unchanged. `[A5]`
5. **Live drag + threading/staleness (ratified).** `rangeChanged` live-drag coalesced at **~25 ms**
   plus a final release emit; histogram analysis off the GUI thread (or bounded) with a generation/
   version token; `QImage`/widget updates on the GUI thread only. `[A6][A7]`
6. **Preview normalization stabilization (ratified — Option A only).** Backend sends the tuple
   `(legacy_normalized, raw_linear)`; `raw_linear` is the linear `SUM/W` divide (classic, after
   display-only masks) or the Drizzle `finalize("divide")` HWC (before 1%/99%), downsampled to the
   same geometry as `legacy_normalized` before the callback; **Qt owns the mapping** and freezes
   anchors once per run/first valid preview (p0.5/p99.5, finite min/max fallback only when
   degenerate), and production Qt builds `_preview_source` from that fixed-anchor mapped raw
   copy. `legacy_normalized` is compatibility-only. Backend-side normalization is rejected;
   classic (min/max) and Drizzle (1%/99%) are unified onto this one stable reference. `[A5]`
7. **Decomposition (ratified).** The A/B/C/D workstream split (§7) is the mission boundary for the
   first implementation mission. `[A8]`
8. **Dead code (ratified).** `_update_header_for_drizzle_final` is documented-as-dead and left
   untouched in this mission; removal (if desired) is deferred to a separate, explicitly
   authorized cleanup mission. `[A10]`

---

## 12. H3 test-harness hardening note (collection-order + version expectations)

**ZSSS-OTPUX-H3-TEST-ORDER-VERSION** (test-only; no production change).

- **Collection-order pollution fixed.** `tests/test_exposure_metadata_contract.py`
  previously installed synthetic `seestar`, `seestar.gui` (empty `__path__`),
  `seestar.gui.settings` and `seestar.gui.histogram_widget` modules into
  process-global `sys.modules` at import/collection time and never restored them.
  Collected first, this made later `seestar.gui_qt` imports fail with
  `ModuleNotFoundError: No module named 'seestar.gui.run_config'`. The stubs are
  no longer necessary: `queue_manager`'s `from seestar.gui.settings import
  SettingsManager, TILE_HEIGHT` is already wrapped in a try/except fallback
  (`SettingsManager = object`), so the test now imports the real `queue_manager`
  directly and installs no synthetic module. Nothing needs restoring.
- **Stale `8.0.0` product-version assertions removed.** All hard-coded `8.0.0`
  expectations in tests were replaced with exact values derived from the
  authoritative source (`seestar.__version__` + `seestar.__codename__` =
  `8.1.0b1` / `Phoenix consedit`): `test_qt_resources_m255d.py` (product
  version / window title), `test_qt_backend_activation.py` (startup witness),
  `test_zsss_lifecycle_reliability.py` (run-metadata `product_version`) and
  `test_packaging.py` (PEP 440 validity, no version pin). Product version and
  production behaviour are unchanged.

---

## 13. Final UX addendum — detachable histogram + batch-boundary live auto

**Status:** implemented on the accepted H1/H2 and exposure/science contracts;
no histogram-analysis or scientific architecture change.

### 13.1 Detachable histogram

The persistent inline histogram now has an explicit Expand action and a
double-click expansion seam.  The lazily-created non-modal
`DetachedHistogramWindow` is a presentation mirror only:

- it owns a second `HistogramView`, but receives the **same authoritative model
  object** already computed by the single H1/H2 `HistogramCoordinator`;
- it owns no coordinator, worker, compute callable or analysis buffer;
- opening/reopening schedules zero histogram computations;
- model, statistics text, BP/WP, auto-zoom and shared view range are mirrored;
- BP/WP drags in either view route through the same `MainWindow` state seam;
- its Auto Stretch / Auto WB buttons call the existing deterministic one-shot
  actions; its live-auto toggles mirror the same two booleans;
- close hides the non-deleting dialog and changes no processing/display state.

### 13.2 Live auto per displayed scientific preview (STABLE-A)

`MainWindow` owns two independent, explicit display-only states, enabled by
default: `Live Auto Stretch` and `Live Auto WB`.  Live auto is keyed by a
**stable scientific-preview identity** derived from engine metadata — never the
GUI `drizzle_group_spin` cadence, which was a widget knob and not a scientific
freshness authority:

- **Classic** — `("batch", current_batch)`: a positive new
  `BackendPreviewPayload.current_batch` (with `PREV_SRC="SUM/W Accumulators"`)
  is the batch identity.
- **Reproject** — `("batch", current_batch)`: the legacy `PREV_SRC`-less
  incremental-reprojection/coadd path, keyed by the same `current_batch`
  counter.
- **Drizzle** — `("drizzle", image_count)`: the engine accepted-frame counter
  (`_drizzle_frame_count`, i.e. `[A1]`'s "accepted/contributing frame") is the
  identity.  Every delivered scientific preview (standard policy: one per
  accepted frame; incremental policy: one per group) has a distinct
  `image_count`, so a displayed preview N can never carry live-auto parameters
  computed for a different identity.

Classic and Reproject deliberately share one identity family (`"batch"`) keyed
by `current_batch` (`stacked_batches_count`).  The `PREV_SRC` header is **not**
an identity dimension within the batch family: `queue_manager._update_preview`
(legacy incremental reproject/coadd) emits no `PREV_SRC`, while
`queue_manager.refresh_preview` routes *every* non-Drizzle session through
`_update_preview_sum_w` with `PREV_SRC="SUM/W Accumulators"` — both key on the
same `stacked_batches_count`.  A Reproject batch N's resolution refresh may
therefore truthfully arrive re-labelled `"SUM/W Accumulators"` while remaining
the same scientific batch N, and must stay inert for live auto and
`raw_revision`.  (The witness `preview_mode` still reports the truthful route
label `classic`/`reproject`/`drizzle` for instrumentation; it is not an identity
dimension.)

The full identity is `(run_context_id, family, counter)`; `run_context_id`
increments once per run so identities never collide across runs.  A duplicate
callback / repaint / resolution refresh carrying the same identity performs no
duplicate AutoWB/AutoStretch (`[A4]`: no silent recalculation on a non-new
preview — while the *one-shot* Auto Stretch button remains a separate, stable,
explicit action that never auto-re-runs).

At one new identity the live-auto application is atomic and ordered: AutoWB is
computed first from pristine pre-WB float data (idempotent per identity), Auto
Stretch then consumes the corresponding WB-only float buffer, and the ordinary
`_on_preview` tail performs the single render/histogram scheduling pass.  No
intermediate refresh or second histogram job is created.

Direct manual BP/WP edits (spin/slider or either histogram view) disable only
Live Auto Stretch.  Direct manual WB-gain edits disable only Live Auto WB.
One-shot Auto Stretch / Auto WB buttons remain available and do not toggle the
live states.  Input payload arrays, SUM/WHT and Drizzle accumulators are never
mutated by any of these operations.

**Instrumentation (witness seams):** `MainWindow` exposes read-only
`preview_mode`, `preview_identity`, `displayed_identity`, `raw_revision`,
`wb_revision`, `live_target_identity` and `live_bp_wp`, so tests can assert
"preview N -> auto target N" and "duplicate N -> no duplicate work" without
touching science.

### 13.3 Preview view-state persistence (STABLE-B)

**Status:** implemented on the accepted STABLE-A identity / live-auto and
display-isolation contracts; no display-analysis, histogram, science or engine
change.

A *valid* backend preview replaces only the displayed scientific content.  The
user's accumulated **view state** — the continuous zoom factor (including a
custom wheel zoom and `Fit` mode), the pan offsets and the accumulated
rotation — is independent of content freshness and survives:

- a valid successive scientific preview N → N+1;
- a same-scientific-identity duplicate callback and an engine
  resolution-refresh re-render of the same identity;
- a genuinely new scientific preview N+1 (content changes; view state does
  not).

The `_on_preview` valid-preview branch no longer zeroes `_preview_rotation` and
no longer calls `_reset_view_transform()` unconditionally.  View state is
reset **only** at explicit lifecycle boundaries, each still starting at the
established defaults (rotation 0°, 100% zoom, centred):

- `_on_run_started` — a new run resets rotation/zoom/pan exactly once, so the
  next run's first preview never inherits the previous run's view state.  The
  reset is atomic from the user's perspective: if a prior valid preview is
  retained, its displayed pixmap and resolution label are immediately
  re-rendered to agree with the reset state (0° / 100% / centred) via a pure
  view reconciliation (`_refresh_preview_view(histogram=False)`) that never
  touches `raw_revision`/identity or schedules a histogram recompute for the
  unchanged source;
- a successful new initial-folder preview (`_on_initial_preview_result`);
- `_clear_preview` — explicit clear resets view state;
- an invalid / unrenderable payload — resets view state and disables the view
  controls exactly as before.

Existing explicit user actions keep their semantics: a preset pick recentres
pan, rotation resets pan (Tk disorientation parity), and clear resets.
`Fit` mode remains a combo *mode* (the fit scale is recomputed per render, not
persisted), so it re-fits the new source automatically while staying selected.

Live auto / STABLE-A identity semantics are unchanged: view persistence never
touches `_record_preview_identity`, `_apply_live_auto_for_batch`,
`raw_revision`, `wb_revision` or the live-auto counters, so a new batch still
fires AutoWB/AutoStretch exactly once and a duplicate/resolution-refresh still
performs no duplicate work.  No payload mutation and no scientific, engine or
settings state change.

### 13.4 Unmistakable terminal-failure presentation (STABLE-C)

**Status:** implemented on the accepted STABLE-A/STABLE-B presentation state;
no status/log wording, summary, controller, runner, science or engine change.

Processing failures are now surfaced through a genuine, owned, non-blocking
`QMessageBox` in addition to the existing (unchanged, truthful) status-bar and
log text.  Four failure surfaces present exactly one box each:

- **Normal backend terminal failure** — `MainWindow._on_run_failed(message)`
  shows a **Critical** box with localized title `error_box_run_failed_title`
  and the raw `message` as plain-text body.
- **Boring terminal failure** — `MainWindow._on_boring_failed(message)` shows a
  **Critical** box (`error_box_boring_failed_title` + raw `message`).
- **Preflight failure** — `MainWindow._report_preflight_failure(prefix, errors)`
  (normal and Boring) shows one **Warning** box
  (`error_box_preflight_title` + the joined `prefix + errors` message): a
  validation attempt yields exactly one box, never one per error string.
- **Structured startup refusal** — `MainWindow._on_run_refused(payload)` shows a
  **Warning** box reusing the existing localized title/body from
  `_format_refusal` (no new wording; the known `OUTPUT_STATE_INCOMPATIBLE`
  code maps through the existing EN/FR keys).

**Ownership / lifecycle design:** the box is a real `QMessageBox` parented to
`MainWindow`, retained on `self._error_message_box` so it can never be
garbage-collected, and exposed read-only as `MainWindow.error_message_box`
with a presentation counter `MainWindow.error_box_count` (display-only test
instrumentation).  A new failure **reuses and replaces** the same box
(re-setting icon/title/body) — repeated signals never pile boxes up; the
previous content is deterministically replaced, not appended.

**Non-blocking guarantee:** the box is shown via `QWidget.show()` with
`Qt.WindowModality.WindowModal` — never `exec()` and never a static
`QMessageBox.critical/warning/information`.  The handler therefore returns
immediately, so the controller-owned terminal `RETURNED`/run-log close and
`QThread` teardown always run.  The body is always rendered with
`Qt.TextFormat.PlainText`, so error detail is never interpreted as rich text.

**Negative contract:** SUCCESS, EMPTY/NO OUTPUT and CANCELLED show **no** error
box (summary-dialog behaviour unchanged); a validation/preflight attempt shows
exactly one box; a single terminal signal shows exactly one box.  `shutdown()`
closes/hides any outstanding box cleanly and idempotently (no stale
window-modal dialog after a normal window close).

---

## 14. Architect closure review — 2026-08-27

**Verdict: ACCEPTED LOCALLY / TECHNICAL MISSION COMPLETE.**  The implemented
exposure-truthfulness, stable preview/display analysis, H1/H2 histogram,
detached-histogram/live-auto UX and STABLE-A/B/C contracts satisfy their
ratified acceptance criteria.  The final STABLE-A identity correction and the
STABLE-B atomic new-run reconciliation were independently reproduced before
acceptance; STABLE-C was inspected for non-blocking controller semantics and
single-box ownership.

Final independent validation on branch
`feature/output-truthfulness-preview-ux`, HEAD
`a7fc8d6a6c16f4c5d724152ebcc16255cac78093`:

- all Qt, preview-analysis, raw-linear producer and exposure-contract tests:
  **750 passed** (749 in the consolidated run plus the isolated TkAgg witness);
- HSI closure: **16 passed**;
- ZSSS lifecycle reliability: **24 passed**;
- packaging: **4 passed**;
- focused STABLE-C lifecycle matrix: **99 passed** and STABLE-A/B/UX/histogram/
  localization matrix: **102 passed** (overlapping focused gates, not added to
  the unique 794-test consolidated total);
- `compileall` and `git diff --check`: clean.

Warnings are limited to known third-party SciPy deprecations and Astropy FITS
comment truncation.  No production version, branch, baseline commit or remote
state changed.  The worktree is intentionally uncommitted; no commit, push,
merge, tag, release or deployment was performed.  Those publication actions
remain explicit human gates.

---

## 15. MICRO CLOSURE — startup-refusal propagation contract (2026-08-27)

Human manual test finding: with an output directory containing previous
processing state and an incompatible run configuration, the run ended on the
generic technical failure path — Critical "Run failed" carrying
`SeestarQueuedStacker.start_processing() reported it did not start` — instead
of the structured `OUTPUT_STATE_INCOMPATIBLE` handling.  This closure is a
classification/propagation/explanation fix only: resume semantics, refusal
behaviour and artifact handling are unchanged.

**Audit finding (exact identity-loss chain).**

1. Structured carrier exists: `queue_manager.py` `StartupRefusal` (class),
   reset per start attempt and set at the early resume preflight site
   (`self.startup_refusal = self._build_startup_refusal(early_result)`).
2. Identity lost at the classification guard `_build_startup_refusal`: it
   only emitted the stable code when `_resume_requested and not
   _is_plain_classic()`.  A resume-requested refusal on a *plain-classic*
   session (scientific fingerprint mismatch, missing/corrupt/legacy manifest,
   dtype mismatch, incompatible reference shape, invalid quality reference
   scale) returned `None`, leaving `startup_refusal = None`.
3. Flattening: `backend_runner.py` `run()` — `start_processing()` returned
   `False` with no carrier, so it raised the plain
   `RuntimeError("...reported it did not start")`.
4. `RunWorker` emitted `failed(str(exc))`; `RunController` relayed it;
   `MainWindow._on_run_failed` presented the generic Critical box.

**Fix (narrowest boundary).** `_build_startup_refusal` now classifies *every*
early refusal of a resume-requested session as `OUTPUT_STATE_INCOMPATIBLE`
(the engine already knows the refusal is caused by the previous output state;
the precise reason stays in `technical_detail`, never parsed).  This is a
structured-reason surfacing only: the same runs are still refused, nothing is
written, processing behaviour is unchanged.  Genuinely unknown false starts
(no resume artifacts, no structured carrier) keep the generic Critical path.

**User-facing wording (mode-independent, EN/FR, proper FR apostrophes).**

- Title EN `Output folder already in use` / FR `Dossier de sortie déjà utilisé`.
- Body: the folder contains data from a previous processing run; to resume,
  make sure the selected mode supports resume; to start a new stack, select a
  new or empty output folder.  It never says the folder cannot be reused and
  never names a mode.  The old mode-specific keys
  (`startup_refusal_output_state_incompatible_body_generic`,
  `startup_refusal_mode_mosaic`, `startup_refusal_mode_reproject`) and
  `MainWindow._refusal_mode_label` were removed.
- The dialog presents only the localized guidance (Warning, owned, non-blocking
  — STABLE-C pattern unchanged); the status bar shows the localized title and
  the run log keeps the engine's technical detail as a secondary line.

**Contract tests.** `tests/test_qt_otpux_micro_closure.py` covers: the full
end-to-end witness (real adapter + fake refusing engine → exactly one Warning
box, no Critical, old output byte-identical, retry in a new folder succeeds);
generic unknown false start stays Critical; EN/FR parity and exact wording;
FR end-to-end box; handler state-preservation matrix; adapter-level structured
`StartupRefusedError` vs generic `RuntimeError`.  `test_zsss_startup_refusal_qm.py`
now asserts a plain-classic resume refusal carries the structured code and a
non-resume refusal stays generic; `test_zsss_lifecycle_reliability.py`
C-section asserts the mode-independent wording.
