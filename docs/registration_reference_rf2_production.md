# RF2-02 — production contract: immutable registration target + passive diagnostics

**Task:** RF2-02 minimal production implementation (after accepted research gate).
**Branch:** `feature/registration-field-rotation` @ `cf8d25c` (working tree, uncommitted).
**Parent HSI scientific contract:** `61291aa`.

This is the concise production/diagnostics contract.  The measurement evidence is
in `docs/registration_reference_rf2.md`; this document records what the code now
does and the guarantees a reviewer/consumer can rely on.

---

## 1. Registration target lifecycle (immutable)

* The registration target is the processed reference image returned by
  `SeestarAligner._get_reference_image` (manual or auto-best), held in
  `reference_image_data_for_global_alignment`.
* It is **immutable for the entire run**, including
  `reproject_between_batches`: the batch-completion and finalize seams no longer
  reassign it to the cumulative stack (`sum/wht` mean).
* The cumulative stack, WCS/grid, previews, batch output and science are all
  preserved.  `_solve_cumulative_stack()` is still called at the reproject seams
  **for its WCS/grid side effect only**; its return value is no longer used as a
  target.
* The batch-plan flush seam (`_flush_current_batch`) remains a no-op reassignment
  (classic path returns its input unchanged).

**Ownership / provenance (unambiguous):**

| Concept | Identifier in code | Lifecycle |
|---|---|---|
| Registration target image | `reference_image_data_for_global_alignment` | immutable |
| Coordinate grid / WCS | `reference_wcs_object`, `ref_wcs_header`, `reference_header_for_wcs` | frozen by `freeze_reference_wcs` |
| Stack accumulator | `cumulative_sum_memmap` / `cumulative_wht_memmap` | grows monotonically |
| Batch lifecycle | `current_batch_items_with_masks_for_stack_batch` | per-batch |

Target provenance is identified from the `HIERARCH SEESTAR REF SRCFILE`
basename carried by the header returned from `_get_reference_image`.  A tiny
in-memory field, `self._registration_target_provenance_id`, carries that basename
for the diagnostics record.  The temporary saved `reference_image.fit`
deliberately omits this card; **no persistence-format migration was added**.

---

## 2. Passive diagnostics (observational, fail-open)

* **Path:** `<output_folder>/registration_diagnostics.jsonl` (JSON Lines).
* **Schema version:** `1.0` (`schema_version` field; consumers key on it).
* **Fail-open:** any build/serialize/write error is caught and logged at debug
  level; it can never alter registration success or any scientific result.
* **Privacy-safe:** only basenames / provenance identifiers; no full paths.

Record fields:

```
schema_version, ts, session_id, event="registration", frame (basename),
target_policy="immutable_selected_reference", reference_provenance (basename),
model="euclidean", success, raw_scale, applied_rotation_deg,
applied_translation [tx, ty], match_count, residual_px {p50, p95, rms} | null,
error | null, diagnostic_only=["raw_scale", "residual_px"]
```

* `raw_scale` — the raw astroalign similarity scale **before** discard
  (`hypot(a, b)` from the returned params).  Diagnostic only.
* `residual_px` — residual of the returned match pairs under the **applied**
  Euclidean matrix (rotation + translation, scale forced to 1.0).  Computed only
  when ≥2 matches are available; otherwise `null`.  Diagnostic only.
* `applied_rotation_deg` / `applied_translation` — the applied Euclidean transform.
* `match_count` — number of matched pairs returned by astroalign (no invented
  inlier count; astroalign does not expose one here).

Per-call diagnostic state is rebuilt on every attempt: on failure the diagnostics
dict is `None` (no stale frame attribution).

---

## 3. Drizzle dead pre-warp removal (transform-only contract)

`SeestarAligner._align_image` gained two backward-compatible kwargs:

* `transform_only=False` — runs the exact same matcher and Euclidean
  scale-discard, computes the exact same 2x3 `cv2_M_final`, but **skips the
  `cv2.warpAffine` resampling** and returns the original (unwarped) image as the
  first element.
* `return_diagnostics=False` — returns a 4-tuple `(img, success, M, diag)`.

The standard (non-mosaic) Drizzle path in `_process_file` now calls
`_align_image(..., return_M=True, transform_only=True, return_diagnostics=True)`.
The classic path calls `_align_image(..., return_diagnostics=True)` (still warps).

**Guarantees:**

* The returned `tf` is bit-identical to the regular alignment `tf` for a
  deterministic matcher (tested).
* The warp backend (`_align_cpu`/`_align_cuda` → `cv2.warpAffine`) is **not**
  invoked on the Drizzle transform-only path (tested).
* Original prepared data, original validity mask, native-WCS fallback, quality
  score, header metadata and Drizzle accumulator inputs are unchanged.
* Classic alignment still warps exactly once (tested).
* `align_on_disk`/memmap cleanup is preserved (the transform-only path flushes/
  closes/removes its temporary input memmap and never deletes data still in use).

---

## 4. Resume contract (unchanged)

Reproject / drizzle / mosaic remain **fail-closed** for resume; only plain
classic SUM/W is resumable.  No additional target persistence was added: the
registration target is recreated and its provenance is identified in memory on
each fresh run; no resume manifest field was needed or introduced.
Transforms are **not persisted**; a rerun recomputes them from the retained
target + sources + settings (the diagnostics record the applied matrix values
observationally, not as a resumable store).

---

## 5. HSI / classic invariants (unchanged)

* HSI weighting/rejection/normalization/RAM-tiled-memmap parity: unchanged.
* Classic resampling: unchanged (single warp per frame).
* Drizzle science: original pixels + geometric mapping (`tf`) + original
  mask/weights, unchanged.
* No Euclidean→Similarity change; raw scale is diagnostic only.  No
  Homography/TPS/projective/distortion engine, no framework rewrite.

---

## 6. Changed files

* `seestar/core/alignment.py` — `_align_image` transform-only + diagnostics.
* `seestar/core/registration_diagnostics.py` — **new** small helper (schema,
  `build_record`, fail-open `append_record`).
* `seestar/queuep/queue_manager.py` — immutable target seams, diagnostics
  recording, provenance/session fields.
* `tests/test_global_reference_audit.py` — empty mutation set (RF2 decision).
* `tests/test_drizzle_integration_qm.py` — fake aligner updated to the new contract.
* `tests/test_rf2_production_impl.py` — **new** production tests.
* `docs/registration_reference_rf2.md` / `_state.md` — corrected to accepted
  decision + implementation.
