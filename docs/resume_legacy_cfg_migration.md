# Legacy 5.x run-config `.cfg` migration (Run CFG v2)

Status: **contract / migration layer only** — no engine wiring, no manifest, no
Qt restore, no `run_config.cfg` writes in real runs yet.

This document describes how `seestar/run_contract.py` recognises and migrates
the legacy JSON `.cfg` witness produced by product version **5.6.0**
(`_stack_*.cfg`).  The reference witness is a JSON object of 3809 bytes
(SHA-256 `0985a53e…`) containing **95 top-level keys**, one of which is a
secret (`astrometry_api_key`).

> **Secret handling.** `astrometry_api_key` — and any password/token/
> credential-like key — is classified *unsafe* and is **never** serialised,
> fingerprinted, or digested.  Diagnostics may name the *key* only, never its
> value.  No secret value or personal path appears in this document, the
> committed fixture, the tests, or any output.

## Schema v2 target

```json
{
  "schema_version": 2,
  "product_version": "<string>",
  "scientific_config": { "…": "…" },
  "execution_config":  { "…": "…" },
  "provenance":       { "…": "…" }
}
```

Deterministic compact JSON (sorted keys, UTF-8), `.cfg` extension, atomic
write to an explicit caller path only.  The single source of truth for every
canonical field is `FIELD_DEFS` in `seestar/run_contract.py`; serialisation,
the scientific fingerprint, the run digest, the settings/backend mappings and
this migration all read from it.

## Classification

Every legacy key is classified into exactly one of six buckets:

| Bucket | Count (witness) | Meaning |
|---|---|---|
| `mapped` | 70 | legacy key == canonical name, migrated 1:1 |
| `renamed` | 11 | legacy key → canonical field under a different name |
| `obsolete` | 4 | recognised but retired/ignored (no canonical target) |
| `nonscientific` | 9 | UI preference (geometry / language / preview stretch-WB) |
| `unsafe` | 1 | secret/credential — never serialised |
| `unknown` | 0 | not recognised at all |

Total: **95**.

### `renamed` (11)

| Legacy key | Canonical target | Reason |
|---|---|---|
| `version` | `product_version` (top-level) | legacy product-version string |
| `stack_norm_method` | `scientific_config.normalize_method` | name alignment |
| `stack_weight_method` | `scientific_config.weighting_method` | name alignment |
| `stack_reject_algo` | `scientific_config.stacking_mode` | rejection = stacking method (`_` vs `-`) |
| `stack_method` | `scientific_config.stacking_mode` | legacy duplicate of `stacking_mode` |
| `stack_winsor_limits` | `scientific_config.winsor_limits` | `"a,b"` → `[a, b]` floats |
| `drizzle_scale` | `scientific_config.drizzle_scale_requested` | requested vs effective |
| `drizzle_wht_threshold` | `scientific_config.drizzle_wht_threshold_requested` | requested vs effective |
| `drizzle_kernel` | `scientific_config.drizzle_kernel_requested` | requested vs effective |
| `drizzle_pixfrac` | `scientific_config.drizzle_pixfrac_requested` | requested vs effective |
| `save_final_as_float32` | `execution_config.save_as_float32` | name alignment |

> `stacking_mode` / `stack_method` / `stack_reject_algo` are three spellings of
> the *same* resume-critical rejection method.  The migration normalises `-`/`_`
> and case, and **fails closed** (`AmbiguousLegacyError`) if they disagree.

### `obsolete` (4)

Retired solver/backend keys with no canonical target; ignored with a bounded
diagnostic.

| Legacy key | Reason |
|---|---|
| `use_third_party_solver` | superseded by `local_solver_preference` |
| `local_ansvr_path` | ANSVR solver retired (ZeSolver owns the strategy) |
| `ansvr_host_port` | ANSVR solver retired |
| `astrometry_solve_field_dir` | Astrometry.net backend retired |

### `nonscientific` (9)

UI presentation preferences, deliberately excluded from scientific config and
not serialised.

| Legacy key | Reason |
|---|---|
| `preview_stretch_method` | preview stretch |
| `preview_black_point` | preview stretch |
| `preview_white_point` | preview stretch |
| `preview_gamma` | preview stretch |
| `preview_r_gain` | preview white balance |
| `preview_g_gain` | preview white balance |
| `preview_b_gain` | preview white balance |
| `language` | UI language |
| `window_geometry` | window geometry |

### `unsafe` (1)

| Legacy key | Handling |
|---|---|
| `astrometry_api_key` | classified `unsafe`; value never read, serialised, fingerprinted or digested; key name only in diagnostics |

### `mapped` (70)

Migrated 1:1 to the same canonical name.  Section shown in parentheses.

**Paths / output** — `execution_config`:
`input_folder`, `output_folder`, `output_filename`, `reference_image_path`,
`last_stack_path`, `temp_folder`.

**Stacking / rejection** — `scientific_config`:
`stacking_mode`, `kappa`, `stack_kappa_low`, `stack_kappa_high`,
`stack_final_combine`.

**Normalisation / weighting** — `scientific_config`:
`use_quality_weighting`, `weight_by_snr`, `weight_by_stars`, `snr_exponent`,
`stars_exponent`, `min_weight`.

**Hot-pixel / debayer** — `scientific_config`:
`bayer_pattern`, `correct_hot_pixels`, `hot_pixel_threshold`, `neighborhood_size`.

**Batch / resource / temp** — `scientific_config` / `execution_config`:
`batch_size`, `max_hq_mem_gb`, `use_gpu`, `cleanup_temp`.

**Drizzle** — `scientific_config`:
`use_drizzle`, `drizzle_mode`, `drizzle_double_norm_fix`.

**Post-processing** — `scientific_config`:
`apply_chroma_correction`, `apply_final_scnr`, `final_scnr_target_channel`,
`final_scnr_amount`, `final_scnr_preserve_luminosity`, `apply_bn`,
`bn_grid_size_str`, `bn_perc_low`, `bn_perc_high`, `bn_std_factor`,
`bn_min_gain`, `bn_max_gain`, `apply_cb`, `cb_border_size`, `cb_blur_radius`,
`cb_min_b_factor`, `cb_max_b_factor`, `apply_final_crop`,
`final_edge_crop_percent`, `apply_photutils_bn`, `photutils_bn_box_size`,
`photutils_bn_filter_size`, `photutils_bn_sigma_clip`,
`photutils_bn_exclude_percentile`.

**Feather / crop / low-weight mask** — `scientific_config`:
`apply_feathering`, `feather_blur_px`, `apply_batch_feathering`,
`apply_master_tile_crop`, `master_tile_crop_percent`, `apply_low_wht_mask`,
`low_wht_percentile`, `low_wht_soften_px`.

**Mosaic** — `execution_config`:
`mosaic_mode_active`, `mosaic_settings`.

**Solver / geometry** — `execution_config`:
`local_solver_preference`, `astap_path`, `astap_data_dir`,
`astap_search_radius`, `use_radec_hints`.

**Output format / reprojection** — `execution_config`:
`preserve_linear_output`, `reproject_between_batches`, `reproject_coadd_final`.

## Contract guarantees

* **Determinism.** `to_canonical_bytes()` / `scientific_fingerprint()` /
  `full_digest()` are byte-stable for equal configuration; the fingerprint only
  tracks the `fingerprint=True` (resume-critical) fields.
* **Fail closed.** A resume-critical field with conflicting legacy aliases
  (`AmbiguousLegacyError`), an uncoercible value (`ValidationError`) or an
  unsafe value (`UnsafeLegacyError`) aborts the migration.
* **Never resumable from CFG alone.** `migrate_legacy` returns a configuration
  restoration/reproducibility object with `resumable=False`; it can never
  authorise a scientific checkpoint/resume.
* **No secret, no personal value, no I/O.** Construction performs no I/O;
  paths are stored verbatim (no normalisation, no existence checks); secrets
  never enter any serialised or digestable payload.
