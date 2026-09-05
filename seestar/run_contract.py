"""Canonical per-run configuration contract (Run CFG v2).

This module is the toolkit-independent, engine-independent source of truth for
the *versioned* per-run configuration (the ``.cfg`` produced at run start) and
for the bounded migration of the legacy 5.x JSON ``.cfg`` witness format.

It deliberately imports **only the Python standard library** — no ``numpy``, no
``astropy``, no ``Qt``/``PySide6``, no ``Tk``, no engine module — so it can be
imported and unit-tested in complete isolation (verified by
``tests/test_run_config_v2.py`` with ``numpy``/``astropy``/``PySide6``/
``tkinter`` blocked in ``sys.modules``).

Design goals
------------

* **One field-definition source.** :data:`FIELD_DEFS` is the single registry
  of every canonical field: its canonical name, section, coercion kind, the
  current Qt/settings attribute name, the backend/engine name where different,
  restore eligibility, resume/fingerprint relevance, presence (always vs
  checkpoint vs optional) and legacy aliases.  Serialization, the scientific
  fingerprint, the run digest, the settings/backend mappings and the legacy
  migration all read from this one table — there are no parallel hard-coded
  field lists.

* **Schema v2.** A deterministic JSON object:

  .. code-block:: json

      {
        "schema_version": 2,
        "product_version": "<string>",
        "scientific_config": { ... },
        "execution_config": { ... },
        "provenance": { ... }
      }

  ``.cfg`` extension, UTF-8, atomic write to an explicit caller path only.

* **Two explicit fingerprint domains.** :func:`classic_fingerprint` reproduces
  the legacy manifest-v1 engine hash
  (``SeestarQueuedStacker._scientific_fingerprint``) byte-for-byte — every
  legacy fingerprint attribute is present, ``None`` where unavailable, keyed by
  the exact legacy attribute name with the documented percent->decimal
  transform.  :func:`drizzle_fingerprint` is the effective-contract science
  hash for the drizzle deposition path: it requires every effective field
  (fail closed, never a partial payload) and embeds a stable domain token so
  the two modes can never collide.  There is no implicit classic fallback.

* **Secrets never serialise.** ``astrometry_api_key`` and any password/token/
  credential-like key are classified *unsafe* by the legacy migration and are
  never written to, fingerprinted from, or digested from.  Diagnostics may name
  the *key* only, never its value.

* **Legacy CFG is a restoration object, never a resumable claim.** Migrating a
  legacy 5.x ``.cfg`` yields a configuration restoration/reproducibility object
  only.  It can never, by itself, authorise a scientific resume.

No file is written at import or at model-construction time; the only writes are
the explicit atomic :func:`write_cfg` (caller-supplied path) and no I/O happens
inside :class:`RunConfig` construction or :func:`migrate_legacy`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "SCHEMA_VERSION",
    "Section",
    "FingerprintDomain",
    "LegacyClass",
    "FieldDef",
    "FIELD_DEFS",
    "field_def",
    "is_unsafe_key",
    "is_obsolete_legacy_key",
    "is_nonscientific_legacy_key",
    "classify_legacy_key",
    "RunConfig",
    "ConfigError",
    "ValidationError",
    "AmbiguousLegacyError",
    "UnsafeConfigError",
    "UnsafeLegacyError",
    "collect_from_settings",
    "collect_from_backend",
    "apply_to_settings",
    "map_to_backend",
    "scientific_fingerprint",
    "classic_fingerprint",
    "drizzle_fingerprint",
    "full_digest",
    "canonical_bytes",
    "diff_configs",
    "DiffResult",
    "write_cfg",
    "read_cfg",
    "ReadReport",
    "parse_legacy_cfg",
    "migrate_legacy",
    "LegacyMigrationResult",
    "classic_fingerprint_names",
    "fingerprint_field_defs",
    "required_fingerprint_names",
]


SCHEMA_VERSION = 2


class Section:
    """Canonical section names (also the top-level schema keys)."""

    SCIENTIFIC = "scientific_config"
    EXECUTION = "execution_config"
    PROVENANCE = "provenance"
    # Pseudo-section for the top-level ``product_version`` field (not nested
    # inside any of the three config sections).
    TOP = "__top__"

    ALL = (SCIENTIFIC, EXECUTION, PROVENANCE)


class FingerprintDomain:
    """Explicit scientific-fingerprint domains (mode separation).

    The scientific fingerprint is always computed for one explicit domain;
    there is no implicit classic fallback.  ``CLASSIC_SUMW`` reproduces the
    legacy manifest-v1 engine hash
    (``SeestarQueuedStacker._scientific_fingerprint``) byte-for-byte.
    ``DRIZZLE`` is the effective-contract science hash for the drizzle
    deposition path.
    """

    CLASSIC_SUMW = "classic_sumw"
    DRIZZLE = "drizzle"

    ALL = (CLASSIC_SUMW, DRIZZLE)


class LegacyClass:
    """Bounded classification for every key found in a legacy JSON ``.cfg``."""

    MAPPED = "mapped"              # legacy key == canonical name, 1:1
    RENAMED = "renamed"            # legacy key -> canonical field, different name
    OBSOLETE = "obsolete"          # recognised but retired/ignored
    NONSCIENTIFIC = "nonscientific"  # UI preference (geometry/language/preview)
    UNKNOWN = "unknown"            # not recognised at all
    UNSAFE = "unsafe"              # secret/credential — never serialise

    ALL = (MAPPED, RENAMED, OBSOLETE, NONSCIENTIFIC, UNKNOWN, UNSAFE)


class ConfigError(Exception):
    """Base error for the run-config contract."""


class ValidationError(ConfigError):
    """A canonical value failed validation/coercion or JSON safety."""


class AmbiguousLegacyError(ConfigError):
    """Two legacy keys map to one resume-critical field with conflicting values."""


class UnsafeConfigError(ConfigError):
    """A secret/credential-like key was found where it must never appear."""


class UnsafeLegacyError(UnsafeConfigError):
    """A resume-critical field received a secret/unsafe value."""


# ---------------------------------------------------------------------------
# Field definition registry — the single source of truth
# ---------------------------------------------------------------------------

# Coercion kinds understood by :func:`_coerce`.
KIND_STR = "str"
KIND_INT = "int"
KIND_FLOAT = "float"
KIND_BOOL = "bool"
KIND_INT_OR_NONE = "int_or_none"
KIND_FLOAT_OR_NONE = "float_or_none"
KIND_BOOL_OR_NONE = "bool_or_none"
KIND_STR_OR_NONE = "str_or_none"
KIND_WINSOR = "winsor_limits"   # "0.05,0.05" | (0.05, 0.05) -> [lo, hi]
KIND_LIST = "list"              # list of JSON-safe scalars/strings
KIND_DICT = "dict"              # nested JSON object

# Legacy fingerprint value transforms (applied only when building the
# classic v1-compatible payload; canonical serialisation is never touched).
TRANSFORM_PERCENT_TO_DECIMAL = "percent_to_decimal"

# Presence: a field is serialised whenever it has a value; ``presence`` governs
# *where* a value is expected to originate.
PRESENCE_ALWAYS = "always"       # part of the user/request config
PRESENCE_CHECKPOINT = "checkpoint"  # runtime-effective value, required at checkpoint
PRESENCE_OPTIONAL = "optional"   # may be absent everywhere


@dataclass(frozen=True)
class FieldDef:
    """Definition of one canonical run-config field.

    Attributes
    ----------
    name:
        Canonical key used inside the serialized section (and in
        ``scientific_config``/``execution_config``/``provenance``).
    section:
        One of :class:`Section` (``scientific_config``/``execution_config``/
        ``provenance``) or ``Section.TOP`` for ``product_version``.
    kind:
        Coercion kind (one of the ``KIND_*`` constants).
    qt_source:
        Attribute name on the Qt/settings state object
        (``QtSettingsState`` / ``SettingsManager``) that feeds this field.
    backend_name:
        Backend/engine keyword-argument or attribute name where it differs from
        ``name``.  ``None`` means "same as canonical name".
    backend_mapped:
        Whether this field has a backend keyword-argument equivalent at all
        (used by :func:`map_to_backend`).  Defaults to ``True`` for always-present
        scientific/execution fields, ``False`` for provenance / runtime-effective /
        derived / GUI-only fields.
    restore_eligible:
        Whether the value may be written back to a settings/UI state.
    fingerprint_domains:
        The explicit fingerprint domains this field participates in (subset of
        :class:`FingerprintDomain.ALL`).  Empty means "never fingerprinted".
        A field may belong to several domains (e.g. a Bayer/weight field shared
        by the classic SUM/W and drizzle contracts).
    legacy_fingerprint_key:
        Exact legacy engine attribute name used in the classic v1-compatible
        payload when it differs from the canonical ``name`` (e.g.
        ``master_tile_crop_percent`` -> ``master_tile_crop_percent_decimal``).
        ``None`` means "same as ``name``".
    legacy_fingerprint_transform:
        Optional value transform applied when building the classic
        v1-compatible payload (e.g. ``percent_to_decimal``).  ``None`` means
        "no transform".
    presence:
        One of the ``PRESENCE_*`` constants.
    derived_from:
        Canonical name of another field from which this field's value is
        derived (``drizzle_processing_policy`` from ``drizzle_mode``).
    legacy_aliases:
        Legacy 5.x JSON ``.cfg`` keys that map onto this canonical field.
        A key equal to ``name`` is classified *mapped*; any other alias is
        *renamed*.
    doc:
        Human-readable note (units, divergence, contract token).
    """

    name: str
    section: str
    kind: str
    qt_source: Optional[str] = None
    backend_name: Optional[str] = None
    backend_mapped: bool = True
    restore_eligible: bool = True
    fingerprint_domains: Tuple[str, ...] = ()
    legacy_fingerprint_key: Optional[str] = None
    legacy_fingerprint_transform: Optional[str] = None
    presence: str = PRESENCE_ALWAYS
    derived_from: Optional[str] = None
    legacy_aliases: Tuple[str, ...] = ()
    doc: str = ""


def _f(
    name: str,
    section: str,
    kind: str,
    qt: Optional[str] = None,
    backend: Optional[str] = None,
    backend_mapped: Optional[bool] = None,
    restore: bool = True,
    fp: Tuple[str, ...] = (),
    legacy_fp_key: Optional[str] = None,
    legacy_fp_transform: Optional[str] = None,
    presence: str = PRESENCE_ALWAYS,
    derived: Optional[str] = None,
    legacy: Tuple[str, ...] = (),
    doc: str = "",
) -> FieldDef:
    if backend_mapped is None:
        backend_mapped = (
            presence == PRESENCE_ALWAYS
            and section != Section.PROVENANCE
            and derived is None
        )
    return FieldDef(
        name=name,
        section=section,
        kind=kind,
        qt_source=qt,
        backend_name=backend,
        backend_mapped=backend_mapped,
        restore_eligible=restore,
        fingerprint_domains=fp,
        legacy_fingerprint_key=legacy_fp_key,
        legacy_fingerprint_transform=legacy_fp_transform,
        presence=presence,
        derived_from=derived,
        legacy_aliases=legacy,
        doc=doc,
    )


# Fingerprint-domain tuples used throughout the field table.
_FP_CLASSIC = (FingerprintDomain.CLASSIC_SUMW,)
_FP_BOTH = (FingerprintDomain.CLASSIC_SUMW, FingerprintDomain.DRIZZLE)
_FP_DRIZZLE = (FingerprintDomain.DRIZZLE,)


# The canonical field table.  Order is documentation order, not serialisation
# order (serialisation sorts keys).
FIELD_DEFS: Tuple[FieldDef, ...] = (
    # --- top-level ---
    _f("product_version", Section.TOP, KIND_STR, legacy=("version",),
       doc="Product display version recorded in the legacy 'version' key."),

    # --- scientific: stacking / rejection family (classic fingerprint) ---
    _f("stacking_mode", Section.SCIENTIFIC, KIND_STR, qt="stacking_mode",
       legacy=("stacking_mode", "stack_method", "stack_reject_algo"), fp=_FP_CLASSIC,
       doc="Rejection/stacking method; legacy 'stack_method'/'stack_reject_algo' "
           "spell the same value with '_' instead of '-'."),
    _f("kappa", Section.SCIENTIFIC, KIND_FLOAT, qt="kappa", fp=_FP_CLASSIC),
    _f("stack_kappa_low", Section.SCIENTIFIC, KIND_FLOAT, qt="stack_kappa_low",
       fp=_FP_CLASSIC),
    _f("stack_kappa_high", Section.SCIENTIFIC, KIND_FLOAT, qt="stack_kappa_high",
       fp=_FP_CLASSIC),
    _f("winsor_limits", Section.SCIENTIFIC, KIND_WINSOR, qt="stack_winsor_limits",
       backend="winsor_limits", fp=_FP_CLASSIC, legacy=("stack_winsor_limits",),
       doc="Winsorised-clip limits; canonical form [lo, hi] floats."),

    # --- scientific: normalisation / weighting (classic + drizzle fingerprint) ---
    _f("normalize_method", Section.SCIENTIFIC, KIND_STR, qt="stack_norm_method",
       fp=_FP_CLASSIC, legacy=("stack_norm_method",)),
    _f("weighting_method", Section.SCIENTIFIC, KIND_STR, qt="stack_weight_method",
       fp=_FP_BOTH, legacy=("stack_weight_method",)),
    _f("use_quality_weighting", Section.SCIENTIFIC, KIND_BOOL,
       qt="use_quality_weighting", fp=_FP_BOTH),
    _f("weight_by_snr", Section.SCIENTIFIC, KIND_BOOL, qt="weight_by_snr",
       fp=_FP_BOTH),
    _f("weight_by_stars", Section.SCIENTIFIC, KIND_BOOL, qt="weight_by_stars",
       fp=_FP_BOTH),
    _f("snr_exponent", Section.SCIENTIFIC, KIND_FLOAT, qt="snr_exponent",
       backend="snr_exp", fp=_FP_BOTH),
    _f("stars_exponent", Section.SCIENTIFIC, KIND_FLOAT, qt="stars_exponent",
       backend="stars_exp", fp=_FP_BOTH),
    _f("min_weight", Section.SCIENTIFIC, KIND_FLOAT, qt="min_weight",
       backend="min_w", fp=_FP_BOTH),

    # --- scientific: hot-pixel / debayer (classic + drizzle fingerprint) ---
    _f("correct_hot_pixels", Section.SCIENTIFIC, KIND_BOOL,
       qt="correct_hot_pixels", fp=_FP_BOTH),
    _f("hot_pixel_threshold", Section.SCIENTIFIC, KIND_FLOAT,
       qt="hot_pixel_threshold", fp=_FP_BOTH),
    _f("neighborhood_size", Section.SCIENTIFIC, KIND_INT,
       qt="neighborhood_size", fp=_FP_BOTH),
    _f("bayer_pattern", Section.SCIENTIFIC, KIND_STR, qt="bayer_pattern",
       fp=_FP_BOTH),

    # --- scientific: batch decomposition (classic fingerprint) ---
    _f("batch_size", Section.SCIENTIFIC, KIND_INT, qt="batch_size", fp=_FP_CLASSIC),
    _f("chunk_size", Section.SCIENTIFIC, KIND_INT_OR_NONE, backend="chunk_size",
       backend_mapped=True, fp=_FP_CLASSIC, presence=PRESENCE_OPTIONAL,
       doc="Auto chunk size (batch_size==1 non-CSV path); runtime-derived."),

    # --- scientific: feathering / crop applied before accumulation (classic) ---
    _f("apply_feathering", Section.SCIENTIFIC, KIND_BOOL, qt="apply_feathering",
       fp=_FP_CLASSIC),
    _f("apply_batch_feathering", Section.SCIENTIFIC, KIND_BOOL,
       qt="apply_batch_feathering", fp=_FP_CLASSIC),
    _f("apply_coverage_render", Section.EXECUTION, KIND_BOOL,
       qt="apply_coverage_render",
       doc="Optional final-only cosmetic coverage reconstruction."),
    _f("feather_blur_px", Section.SCIENTIFIC, KIND_INT, qt="feather_blur_px",
       fp=_FP_CLASSIC),
    _f("apply_master_tile_crop", Section.SCIENTIFIC, KIND_BOOL,
       qt="apply_master_tile_crop", fp=_FP_CLASSIC),
    _f("master_tile_crop_percent", Section.SCIENTIFIC, KIND_FLOAT,
       qt="master_tile_crop_percent", fp=_FP_CLASSIC,
       legacy=("master_tile_crop_percent",),
       legacy_fp_key="master_tile_crop_percent_decimal",
       legacy_fp_transform=TRANSFORM_PERCENT_TO_DECIMAL,
       doc="Canonical unit: percent. Engine fingerprint attribute "
           "'master_tile_crop_percent_decimal' = value/100."),
    _f("apply_low_wht_mask", Section.SCIENTIFIC, KIND_BOOL,
       qt="apply_low_wht_mask", fp=_FP_CLASSIC),
    _f("low_wht_percentile", Section.SCIENTIFIC, KIND_INT,
       qt="low_wht_percentile", fp=_FP_CLASSIC),
    _f("low_wht_soften_px", Section.SCIENTIFIC, KIND_INT,
       qt="low_wht_soften_px", fp=_FP_CLASSIC),

    # --- scientific: drizzle contract (M3) ---
    _f("use_drizzle", Section.SCIENTIFIC, KIND_BOOL, qt="use_drizzle"),
    _f("drizzle_mode", Section.EXECUTION, KIND_STR, qt="drizzle_mode",
       legacy=("drizzle_mode",), doc="'Final'/'Incremental' policy source."),
    _f("drizzle_processing_policy", Section.SCIENTIFIC, KIND_STR,
       derived="drizzle_mode",
       doc="Derived: Final->'standard', Incremental->'incremental'."),
    _f("drizzle_group_size", Section.EXECUTION, KIND_INT,
       qt="drizzle_group_size", legacy=("drizzle_group_size",),
       doc="Preview/progression policy, not science."),
    _f("drizzle_scale_requested", Section.SCIENTIFIC, KIND_FLOAT,
       qt="drizzle_scale", backend="drizzle_scale", legacy=("drizzle_scale",)),
    _f("drizzle_scale_effective", Section.SCIENTIFIC, KIND_FLOAT,
       presence=PRESENCE_CHECKPOINT, fp=_FP_DRIZZLE,
       doc="Runtime-effective scale."),
    _f("drizzle_kernel_requested", Section.SCIENTIFIC, KIND_STR,
       qt="drizzle_kernel", backend="drizzle_kernel", legacy=("drizzle_kernel",)),
    _f("drizzle_kernel_effective", Section.SCIENTIFIC, KIND_STR,
       presence=PRESENCE_CHECKPOINT, fp=_FP_DRIZZLE,
       doc="Runtime-effective kernel (tophat coerces to square)."),
    _f("drizzle_pixfrac_requested", Section.SCIENTIFIC, KIND_FLOAT,
       qt="drizzle_pixfrac", backend="drizzle_pixfrac", legacy=("drizzle_pixfrac",)),
    _f("drizzle_pixfrac_effective", Section.SCIENTIFIC, KIND_FLOAT,
       presence=PRESENCE_CHECKPOINT, fp=_FP_DRIZZLE,
       doc="Runtime-effective pixfrac (1.0 for Lanczos)."),
    _f("drizzle_wht_threshold_requested", Section.SCIENTIFIC, KIND_FLOAT,
       qt="drizzle_wht_threshold", backend="drizzle_wht_threshold",
       legacy=("drizzle_wht_threshold",)),
    _f("drizzle_wht_threshold_effective", Section.SCIENTIFIC, KIND_FLOAT,
       presence=PRESENCE_CHECKPOINT, fp=_FP_DRIZZLE,
       doc="Runtime-effective relative WHT threshold."),
    _f("drizzle_wht_policy", Section.SCIENTIFIC, KIND_STR, presence=PRESENCE_OPTIONAL,
       fp=_FP_DRIZZLE,
       doc="Coverage-threshold policy token (e.g. 'relative_coverage_v1')."),
    _f("drizzle_fillval", Section.SCIENTIFIC, KIND_STR, presence=PRESENCE_CHECKPOINT,
       fp=_FP_DRIZZLE,
       doc="Drizzle fillval; runtime-effective value retained by "
           "DrizzleAccumulator and restored on resume."),
    _f("drizzle_double_norm_fix", Section.SCIENTIFIC, KIND_BOOL,
       qt="drizzle_double_norm_fix", legacy=("drizzle_double_norm_fix",),
       fp=_FP_DRIZZLE),

    # --- scientific: final-output post-processing (not fingerprint) ---
    _f("apply_chroma_correction", Section.SCIENTIFIC, KIND_BOOL,
       qt="apply_chroma_correction"),
    _f("apply_final_scnr", Section.SCIENTIFIC, KIND_BOOL, qt="apply_final_scnr"),
    _f("final_scnr_target_channel", Section.SCIENTIFIC, KIND_STR,
       qt="final_scnr_target_channel"),
    _f("final_scnr_amount", Section.SCIENTIFIC, KIND_FLOAT, qt="final_scnr_amount"),
    _f("final_scnr_preserve_luminosity", Section.SCIENTIFIC, KIND_BOOL,
       qt="final_scnr_preserve_luminosity"),
    _f("apply_bn", Section.SCIENTIFIC, KIND_BOOL, qt="apply_bn"),
    _f("bn_grid_size_str", Section.SCIENTIFIC, KIND_STR, qt="bn_grid_size_str"),
    _f("bn_perc_low", Section.SCIENTIFIC, KIND_INT, qt="bn_perc_low"),
    _f("bn_perc_high", Section.SCIENTIFIC, KIND_INT, qt="bn_perc_high"),
    _f("bn_std_factor", Section.SCIENTIFIC, KIND_FLOAT, qt="bn_std_factor"),
    _f("bn_min_gain", Section.SCIENTIFIC, KIND_FLOAT, qt="bn_min_gain"),
    _f("bn_max_gain", Section.SCIENTIFIC, KIND_FLOAT, qt="bn_max_gain"),
    _f("apply_cb", Section.SCIENTIFIC, KIND_BOOL, qt="apply_cb"),
    _f("cb_border_size", Section.SCIENTIFIC, KIND_INT, qt="cb_border_size"),
    _f("cb_blur_radius", Section.SCIENTIFIC, KIND_INT, qt="cb_blur_radius"),
    _f("cb_min_b_factor", Section.SCIENTIFIC, KIND_FLOAT, qt="cb_min_b_factor"),
    _f("cb_max_b_factor", Section.SCIENTIFIC, KIND_FLOAT, qt="cb_max_b_factor"),
    _f("apply_photutils_bn", Section.SCIENTIFIC, KIND_BOOL, qt="apply_photutils_bn"),
    _f("photutils_bn_box_size", Section.SCIENTIFIC, KIND_INT,
       qt="photutils_bn_box_size"),
    _f("photutils_bn_filter_size", Section.SCIENTIFIC, KIND_INT,
       qt="photutils_bn_filter_size"),
    _f("photutils_bn_sigma_clip", Section.SCIENTIFIC, KIND_FLOAT,
       qt="photutils_bn_sigma_clip"),
    _f("photutils_bn_exclude_percentile", Section.SCIENTIFIC, KIND_FLOAT,
       qt="photutils_bn_exclude_percentile"),
    _f("apply_final_crop", Section.SCIENTIFIC, KIND_BOOL, qt="apply_final_crop"),
    _f("final_edge_crop_percent", Section.SCIENTIFIC, KIND_FLOAT,
       qt="final_edge_crop_percent"),

    # --- scientific: background-match / grid / registration contracts ---
    _f("match_background_for_final", Section.SCIENTIFIC, KIND_BOOL_OR_NONE,
       qt="match_background_for_final", backend_mapped=True,
       presence=PRESENCE_OPTIONAL),
    _f("background_match_contract", Section.SCIENTIFIC, KIND_STR,
       presence=PRESENCE_OPTIONAL,
       fp=_FP_DRIZZLE,
       doc="Background-match contract token (placeholder until runtime pins it)."),
    _f("background_match_contract_version", Section.SCIENTIFIC, KIND_INT,
       presence=PRESENCE_OPTIONAL, fp=_FP_DRIZZLE),
    _f("output_grid_contract", Section.SCIENTIFIC, KIND_STR,
       presence=PRESENCE_OPTIONAL,
       fp=_FP_DRIZZLE,
       doc="Output-grid contract placeholder (runtime-effective values required "
           "at checkpoint)."),
    _f("output_grid_contract_version", Section.SCIENTIFIC, KIND_INT,
       presence=PRESENCE_OPTIONAL, fp=_FP_DRIZZLE),
    _f("registration_contract", Section.SCIENTIFIC, KIND_STR,
       presence=PRESENCE_OPTIONAL,
       fp=_FP_DRIZZLE,
       doc="Registration contract placeholder (runtime-effective values required "
           "at checkpoint)."),
    _f("registration_contract_version", Section.SCIENTIFIC, KIND_INT,
       presence=PRESENCE_OPTIONAL, fp=_FP_DRIZZLE),

    # --- execution: paths (verbatim; no I/O, no normalisation on construction) ---
    _f("input_folder", Section.EXECUTION, KIND_STR, qt="input_folder",
       backend="input_dir"),
    _f("output_folder", Section.EXECUTION, KIND_STR, qt="output_folder",
       backend="output_dir"),
    _f("output_filename", Section.EXECUTION, KIND_STR, qt="output_filename"),
    _f("temp_folder", Section.EXECUTION, KIND_STR, qt="temp_folder"),
    _f("reference_image_path", Section.EXECUTION, KIND_STR,
       qt="reference_image_path", backend="reference_path_ui"),
    _f("last_stack_path", Section.EXECUTION, KIND_STR, qt="last_stack_path",
       backend_mapped=False, restore=True,
       doc="Last completed stack; GUI parity, never resume intent."),

    # --- execution: batch / resource / mode ---
    _f("cleanup_temp", Section.EXECUTION, KIND_BOOL, qt="cleanup_temp",
       backend="perform_cleanup"),
    _f("stack_final_combine", Section.EXECUTION, KIND_STR,
       qt="stack_final_combine"),
    _f("max_hq_mem_gb", Section.EXECUTION, KIND_FLOAT, qt="max_hq_mem_gb"),
    _f("num_processing_workers", Section.EXECUTION, KIND_INT_OR_NONE,
       presence=PRESENCE_OPTIONAL, doc="Engine worker count (-1 auto)."),
    _f("use_gpu", Section.EXECUTION, KIND_BOOL, qt="use_gpu",
       doc="GPU acceleration intent (request_gpu)."),
    _f("save_as_float32", Section.EXECUTION, KIND_BOOL, qt="save_final_as_float32",
       backend="save_as_float32", legacy=("save_final_as_float32",)),
    _f("preserve_linear_output", Section.EXECUTION, KIND_BOOL,
       qt="preserve_linear_output"),
    _f("mosaic_mode_active", Section.EXECUTION, KIND_BOOL, qt="mosaic_mode_active",
       backend="is_mosaic_run"),
    _f("mosaic_settings", Section.EXECUTION, KIND_DICT, qt="mosaic_settings"),
    _f("reproject_between_batches", Section.EXECUTION, KIND_BOOL,
       qt="reproject_between_batches"),
    _f("reproject_coadd_final", Section.EXECUTION, KIND_BOOL,
       qt="reproject_coadd_final"),

    # --- execution: solver backend / geometry-relevant ---
    _f("local_solver_preference", Section.EXECUTION, KIND_STR,
       qt="local_solver_preference"),
    _f("astap_path", Section.EXECUTION, KIND_STR, qt="astap_path"),
    _f("astap_data_dir", Section.EXECUTION, KIND_STR, qt="astap_data_dir"),
    _f("astap_search_radius", Section.EXECUTION, KIND_FLOAT,
       qt="astap_search_radius"),
    _f("astap_downsample", Section.EXECUTION, KIND_INT, qt="astap_downsample"),
    _f("astap_sensitivity", Section.EXECUTION, KIND_INT, qt="astap_sensitivity"),
    _f("use_radec_hints", Section.EXECUTION, KIND_BOOL, qt="use_radec_hints",
       backend_mapped=False, doc="Solver hint kept in settings, not a backend kwarg."),

    # --- provenance: software / algorithm contract versions ---
    _f("producer", Section.PROVENANCE, KIND_STR, presence=PRESENCE_OPTIONAL,
       doc="Producer identifier (e.g. 'zeseestarstacker')."),
    _f("producer_version", Section.PROVENANCE, KIND_STR, presence=PRESENCE_OPTIONAL),
    _f("algorithm_contract_version", Section.PROVENANCE, KIND_INT,
       presence=PRESENCE_OPTIONAL,
       doc="Scientific algorithm contract version."),
    _f("drizzle_lib_version", Section.PROVENANCE, KIND_STR,
       presence=PRESENCE_CHECKPOINT, doc="drizzle library version at write time."),
)

_FIELD_BY_NAME: Dict[str, FieldDef] = {f.name: f for f in FIELD_DEFS}
assert len(_FIELD_BY_NAME) == len(FIELD_DEFS), "duplicate canonical field names"

# Legacy keys explicitly recognised but retired / ignored.
_LEGACY_OBSOLETE_KEYS: Tuple[str, ...] = (
    "use_third_party_solver",
    "local_ansvr_path",
    "ansvr_host_port",
    "astrometry_solve_field_dir",
)

# Legacy UI-preference keys (never scientific, never serialised here).
_LEGACY_NONSCIENTIFIC_KEYS: Tuple[str, ...] = (
    "preview_stretch_method",
    "preview_black_point",
    "preview_white_point",
    "preview_gamma",
    "preview_r_gain",
    "preview_g_gain",
    "preview_b_gain",
    "language",
    "window_geometry",
)

# Explicit secret keys.  Never serialise, fingerprint or digest.
_UNSAFE_EXACT_KEYS = frozenset({"astrometry_api_key"})

# Substring hints for password/token/credential-like keys (name-only matching,
# never value inspection).
_UNSAFE_SUBSTRINGS = (
    "api_key",
    "apikey",
    "token",
    "password",
    "passwd",
    "secret",
    "credential",
    "private_key",
    "auth",
)


def field_def(name: str) -> FieldDef:
    """Return the :class:`FieldDef` for a canonical field name."""
    return _FIELD_BY_NAME[name]


def is_unsafe_key(name: str) -> bool:
    """Return ``True`` when a key is secret/credential-like.

    Matches on the key *name* only — never on its value — so secret material is
    never read, compared or reported.
    """
    n = str(name).strip().lower()
    if n in _UNSAFE_EXACT_KEYS:
        return True
    return any(hint in n for hint in _UNSAFE_SUBSTRINGS)


def is_obsolete_legacy_key(name: str) -> bool:
    return name in _LEGACY_OBSOLETE_KEYS


def is_nonscientific_legacy_key(name: str) -> bool:
    return name in _LEGACY_NONSCIENTIFIC_KEYS


# Reverse index: legacy key -> (FieldDef | classification sentinel).
_LEGACY_INDEX: Dict[str, Tuple[Optional[FieldDef], str]] = {}
for _fd in FIELD_DEFS:
    for _alias in _fd.legacy_aliases:
        if _alias == _fd.name:
            _LEGACY_INDEX[_alias] = (_fd, LegacyClass.MAPPED)
        else:
            _LEGACY_INDEX[_alias] = (_fd, LegacyClass.RENAMED)
# Implicit 1:1 mapping for every canonical field name (legacy key == canonical
# name).  Explicit aliases above win; ``Section.TOP`` fields are excluded
# because their legacy key is ``version`` (an explicit alias), not their name.
for _fd in FIELD_DEFS:
    if _fd.section == Section.TOP:
        continue
    if _fd.name not in _LEGACY_INDEX:
        _LEGACY_INDEX[_fd.name] = (_fd, LegacyClass.MAPPED)
for _key in _LEGACY_OBSOLETE_KEYS:
    _LEGACY_INDEX[_key] = (None, LegacyClass.OBSOLETE)
for _key in _LEGACY_NONSCIENTIFIC_KEYS:
    _LEGACY_INDEX[_key] = (None, LegacyClass.NONSCIENTIFIC)


def classify_legacy_key(name: str) -> Tuple[Optional[FieldDef], str]:
    """Classify a legacy key, returning ``(target FieldDef or None, class)``.

    Unsafe (secret) keys are classified before the field table so a key can
    never be promoted to a serialised field by accident.
    """
    if is_unsafe_key(name):
        return (None, LegacyClass.UNSAFE)
    entry = _LEGACY_INDEX.get(name)
    if entry is not None:
        return entry
    return (None, LegacyClass.UNKNOWN)


# ---------------------------------------------------------------------------
# Coercion and JSON safety
# ---------------------------------------------------------------------------


def _coerce(kind: str, value: Any) -> Any:
    """Coerce ``value`` to the canonical form of ``kind``.

    Raises :class:`ValidationError` when the value cannot be coerced.  Explicit
    ``None`` is preserved for the nullable kinds (``*_or_none``); the strictly
    typed kinds (``KIND_STR``/``KIND_INT``/``KIND_FLOAT``/``KIND_BOOL``/
    ``KIND_WINSOR``/``KIND_LIST``/``KIND_DICT``) reject ``None``.
    """
    if value is None:
        if kind in (
            KIND_INT_OR_NONE,
            KIND_FLOAT_OR_NONE,
            KIND_BOOL_OR_NONE,
            KIND_STR_OR_NONE,
        ):
            return None
        raise ValidationError(f"null is not permitted for {kind!r} fields")
    if kind == KIND_STR:
        return value if isinstance(value, str) else str(value)
    if kind == KIND_STR_OR_NONE:
        return value if isinstance(value, str) else None
    if kind == KIND_INT:
        if isinstance(value, bool):
            raise ValidationError("bool is not an int")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"not an int: {value!r}") from exc
    if kind == KIND_INT_OR_NONE:
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if kind == KIND_FLOAT:
        if isinstance(value, bool):
            raise ValidationError("bool is not a float")
        try:
            f = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"not a float: {value!r}") from exc
        if not math.isfinite(f):
            raise ValidationError("non-finite float")
        return f
    if kind == KIND_FLOAT_OR_NONE:
        if isinstance(value, bool):
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None
    if kind == KIND_BOOL:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("true", "1"):
                return True
            if s in ("false", "0"):
                return False
        raise ValidationError(f"not a bool: {value!r}")
    if kind == KIND_BOOL_OR_NONE:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        return None
    if kind == KIND_WINSOR:
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",")]
            if len(parts) != 2:
                raise ValidationError(f"bad winsor limits: {value!r}")
            return [_coerce(KIND_FLOAT, p) for p in parts]
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValidationError(f"bad winsor limits: {value!r}")
            return [_coerce(KIND_FLOAT, v) for v in value]
        raise ValidationError(f"bad winsor limits: {value!r}")
    if kind == KIND_LIST:
        if not isinstance(value, (list, tuple)):
            raise ValidationError(f"not a list: {value!r}")
        out = []
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                raise ValidationError("nested container in list field")
            out.append(_json_scalar(item))
        return out
    if kind == KIND_DICT:
        if not isinstance(value, dict):
            raise ValidationError(f"not a dict: {value!r}")
        return _json_safe(value)
    raise ValidationError(f"unknown kind {kind!r}")


def _json_scalar(value: Any) -> Any:
    """Return a JSON-safe scalar, rejecting NaN/Inf and non-scalar types."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        f = float(value) if isinstance(value, float) else value
        if isinstance(f, float) and not math.isfinite(f):
            raise ValidationError("non-finite number")
        return value
    if isinstance(value, (list, tuple, dict)):
        raise ValidationError("container where scalar expected")
    raise ValidationError(f"non-JSON value {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    """Deep-copy ``value`` into a strictly JSON-safe structure.

    Rejects ``NaN``/``Inf`` (both bare floats and nested), ``bytes``, sets, and
    any non-JSON type.  Tuples become lists for deterministic serialisation.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    raise ValidationError(f"non-JSON value of type {type(value).__name__}")


def _normalize_mode_token(value: Any) -> str:
    """Normalise a mode/rejection token for alias-conflict comparison."""
    return str(value).strip().lower().replace("-", "_")


# ---------------------------------------------------------------------------
# The canonical model
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Immutable-by-convention canonical run configuration.

    ``scientific`` / ``execution`` / ``provenance`` map canonical field names to
    coerced, JSON-safe values.  Construction performs no I/O.
    """

    product_version: str = ""
    scientific: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_sections(
        cls,
        *,
        product_version: str = "",
        scientific: Optional[Mapping[str, Any]] = None,
        execution: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "RunConfig":
        """Build from canonical-name section mappings (values already coerced).

        Unknown canonical names are ignored (allowlist).  The derived fields
        are computed and appended.
        """
        cfg = cls(product_version=str(product_version or ""))
        if scientific:
            cfg.scientific.update(_filter_section(Section.SCIENTIFIC, scientific))
        if execution:
            cfg.execution.update(_filter_section(Section.EXECUTION, execution))
        if provenance:
            cfg.provenance.update(_filter_section(Section.PROVENANCE, provenance))
        _finalize_derived(cfg)
        return cfg

    # -------------------------------------------------------------- serialise
    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return the deterministic schema-v2 object (no secrets, no I/O)."""
        out: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "product_version": str(self.product_version),
        }
        out[Section.SCIENTIFIC] = _sorted_section(self.scientific)
        out[Section.EXECUTION] = _sorted_section(self.execution)
        out[Section.PROVENANCE] = _sorted_section(self.provenance)
        return out

    def to_canonical_bytes(self) -> bytes:
        """Deterministic canonical JSON bytes (compact, sorted keys, UTF-8)."""
        return json.dumps(
            self.to_canonical_dict(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    def scientific_fingerprint(self, domain: str) -> str:
        """SHA-256 of the scientific fields of one explicit domain."""
        return scientific_fingerprint(self, domain=domain)

    def classic_fingerprint(self) -> str:
        """v1-compatible classic SUM/W fingerprint (legacy engine parity)."""
        return classic_fingerprint(self)

    def drizzle_fingerprint(self) -> str:
        """Effective Drizzle scientific-contract fingerprint."""
        return drizzle_fingerprint(self)

    def full_digest(self) -> str:
        """SHA-256 of the whole canonical JSON document."""
        return full_digest(self)

    def get(self, section: str, name: str) -> Any:
        return self._section(section).get(name)

    def _section(self, section: str) -> Dict[str, Any]:
        if section == Section.SCIENTIFIC:
            return self.scientific
        if section == Section.EXECUTION:
            return self.execution
        if section == Section.PROVENANCE:
            return self.provenance
        raise KeyError(section)


def _filter_section(section: str, values: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, value in values.items():
        fd = _FIELD_BY_NAME.get(name)
        if fd is None or fd.section != section:
            continue
        out[name] = _json_safe(value)
    return out


def _sorted_section(section: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: _json_safe(v) for k, v in sorted(section.items())}


def _finalize_derived(cfg: RunConfig) -> None:
    """Compute derived fields (e.g. ``drizzle_processing_policy``)."""
    for fd in FIELD_DEFS:
        if not fd.derived_from:
            continue
        src_fd = _FIELD_BY_NAME.get(fd.derived_from)
        if src_fd is None:
            continue
        src_value = cfg.get(src_fd.section, src_fd.name)
        if src_value is None:
            continue
        cfg._section(fd.section)[fd.name] = _derive(fd.name, src_value)


def _derive(name: str, source: Any) -> Any:
    if name == "drizzle_processing_policy":
        mode = _normalize_mode_token(source)
        return "incremental" if mode == "incremental" else "standard"
    return None


# ---------------------------------------------------------------------------
# Fingerprint / digest / diff
# ---------------------------------------------------------------------------


def canonical_bytes(cfg: RunConfig) -> bytes:
    """Deterministic canonical JSON bytes for a :class:`RunConfig`."""
    return cfg.to_canonical_bytes()


def _classic_fingerprint_payload(cfg: RunConfig) -> Dict[str, Any]:
    """Build the exact legacy v1 classic SUM/W fingerprint payload.

    Every classic fingerprint attribute is present, using ``None`` where the
    value is unavailable, keyed by the exact legacy engine attribute name
    (``legacy_fingerprint_key``) and value-transformed where documented
    (``legacy_fingerprint_transform``).
    """
    payload: Dict[str, Any] = {}
    for fd in FIELD_DEFS:
        if FingerprintDomain.CLASSIC_SUMW not in fd.fingerprint_domains:
            continue
        key = fd.legacy_fingerprint_key or fd.name
        value = cfg.get(fd.section, fd.name)
        if value is None:
            payload[key] = None
        else:
            payload[key] = _legacy_fingerprint_value(fd, value)
    return payload


def _legacy_fingerprint_value(fd: FieldDef, value: Any) -> Any:
    value = _json_safe(value)
    if fd.legacy_fingerprint_transform == TRANSFORM_PERCENT_TO_DECIMAL:
        return float(value) / 100.0
    return value


def _drizzle_fingerprint_payload(cfg: RunConfig) -> Dict[str, Any]:
    """Build the effective Drizzle contract payload, failing closed on any
    missing required effective field."""
    payload: Dict[str, Any] = {}
    missing: List[str] = []
    for fd in FIELD_DEFS:
        if FingerprintDomain.DRIZZLE not in fd.fingerprint_domains:
            continue
        value = cfg.get(fd.section, fd.name)
        if value is None:
            missing.append(fd.name)
            continue
        payload[fd.name] = _json_safe(value)
    if missing:
        missing.sort()
        raise ValidationError(
            "drizzle fingerprint missing required effective field(s): "
            + ", ".join(missing)
        )
    # Stable domain token prevents any Classic/Drizzle payload collision and
    # makes the mode explicit in the hashed bytes.
    payload["fingerprint_domain"] = FingerprintDomain.DRIZZLE
    return payload


def classic_fingerprint(cfg: RunConfig) -> str:
    """SHA-256 hex digest byte-for-byte identical to the legacy manifest-v1
    engine hash (``SeestarQueuedStacker._scientific_fingerprint``).

    Includes every legacy fingerprint attribute, using ``None`` where
    unavailable, serialised with ``json.dumps(sort_keys=True,
    separators=(',',':'))`` semantics.  A hash cannot be used to reconstruct
    configuration.
    """
    payload = _classic_fingerprint_payload(cfg)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def drizzle_fingerprint(cfg: RunConfig) -> str:
    """SHA-256 hex digest of the effective Drizzle scientific contract.

    Requires every Drizzle fingerprint field (fail closed, never a partial
    payload) and embeds the stable Drizzle domain token.
    """
    payload = _drizzle_fingerprint_payload(cfg)
    blob = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def scientific_fingerprint(cfg: RunConfig, *, domain: str) -> str:
    """SHA-256 hex digest over the scientific fields of one explicit domain.

    The domain is required: there is no implicit classic fallback.  Prefer the
    clearly named :func:`classic_fingerprint` (v1-compatible) or
    :func:`drizzle_fingerprint` (effective contract), or pass a
    :class:`FingerprintDomain` value.
    """
    if domain == FingerprintDomain.CLASSIC_SUMW:
        return classic_fingerprint(cfg)
    if domain == FingerprintDomain.DRIZZLE:
        return drizzle_fingerprint(cfg)
    raise ValidationError(f"unknown fingerprint domain {domain!r}")


def full_digest(cfg: RunConfig) -> str:
    """SHA-256 hex digest over the whole canonical JSON document."""
    return hashlib.sha256(cfg.to_canonical_bytes()).hexdigest()


def classic_fingerprint_names() -> frozenset:
    """Canonical names of the classic v1 SUM/W fingerprint fields."""
    return frozenset(
        fd.name for fd in FIELD_DEFS
        if FingerprintDomain.CLASSIC_SUMW in fd.fingerprint_domains
    )


def fingerprint_field_defs(domain: str) -> Tuple[FieldDef, ...]:
    """Field definitions participating in ``domain`` (in FIELD_DEFS order)."""
    if domain not in FingerprintDomain.ALL:
        raise ValidationError(f"unknown fingerprint domain {domain!r}")
    return tuple(fd for fd in FIELD_DEFS if domain in fd.fingerprint_domains)


def required_fingerprint_names(domain: str) -> frozenset:
    """Canonical names that must be present to hash ``domain``.

    Classic includes every attribute (``None`` where absent); Drizzle requires
    every effective field (fail closed, never a partial payload).
    """
    return frozenset(fd.name for fd in fingerprint_field_defs(domain))


def _is_classic_fingerprint(fd: FieldDef) -> bool:
    """``True`` when a field participates in the classic resume-critical set."""
    return FingerprintDomain.CLASSIC_SUMW in fd.fingerprint_domains


@dataclass(frozen=True)
class DiffEntry:
    field: str
    section: str
    checkpoint: Any
    current: Any


@dataclass
class DiffResult:
    """Bounded field-level diff (checkpoint vs current), never a giant dump."""

    diffs: List[DiffEntry]
    truncated: bool
    total: int


def diff_configs(
    checkpoint: RunConfig,
    current: RunConfig,
    *,
    limit: int = 40,
    sections: Optional[Iterable[str]] = None,
) -> DiffResult:
    """Return a bounded list of field-level differences.

    Only fields present in *either* config are compared; values are compared in
    their JSON-safe canonical form.  At most ``limit`` differences are returned,
    with ``truncated=True`` and the real ``total`` when more exist.
    """
    limit = max(1, int(limit))
    total = 0
    diffs: List[DiffEntry] = []

    wanted_sections: List[str]
    if sections is None:
        wanted_sections = list(Section.ALL)
        include_top = True
    else:
        wanted_sections = [s for s in sections if s in Section.ALL]
        include_top = Section.TOP in sections

    if include_top and checkpoint.product_version != current.product_version:
        total += 1
        if len(diffs) < limit:
            diffs.append(
                DiffEntry("product_version", Section.TOP,
                          checkpoint.product_version, current.product_version)
            )

    for section in wanted_sections:
        a = checkpoint._section(section)
        b = current._section(section)
        for name in sorted(set(a) | set(b)):
            va = _json_safe(a.get(name))
            vb = _json_safe(b.get(name))
            if va != vb:
                total += 1
                if len(diffs) < limit:
                    diffs.append(DiffEntry(name, section, va, vb))

    return DiffResult(diffs=diffs[:limit], truncated=total > limit, total=total)


# ---------------------------------------------------------------------------
# Settings / backend mappings (single field-spec source)
# ---------------------------------------------------------------------------


def _read_settings_attr(settings: Any, fd: FieldDef) -> Any:
    """Read a field value from a settings object or plain dict (duck-typed)."""
    if fd.qt_source is None:
        return None
    if isinstance(settings, Mapping):
        return settings.get(fd.qt_source)
    return getattr(settings, fd.qt_source, None)


def collect_from_settings(
    settings: Any,
    *,
    product_version: str = "",
) -> RunConfig:
    """Map a plain settings/state object to a canonical :class:`RunConfig`.

    Only fields with a ``qt_source`` are read; ``presence`` in
    {checkpoint, optional} fields are skipped (they originate at runtime, not
    from settings).  No I/O occurs.  Values are coerced leniently (uncoercible
    values are dropped rather than raising, matching the defensive settings
    coercion elsewhere).
    """
    sci: Dict[str, Any] = {}
    exe: Dict[str, Any] = {}
    prov: Dict[str, Any] = {}
    for fd in FIELD_DEFS:
        if fd.section == Section.TOP or fd.derived_from:
            continue
        if fd.qt_source is None:
            continue
        if fd.presence in (PRESENCE_CHECKPOINT, PRESENCE_OPTIONAL):
            # Optional-but-settings-sourced fields (e.g. match_background_for_final)
            # are read only when they carry a non-None value.
            raw = _read_settings_attr(settings, fd)
            if raw is None:
                continue
            try:
                value = _coerce(fd.kind, raw)
            except ValidationError:
                continue
            if value is None:
                continue
            _put((sci, exe, prov), fd, value)
            continue
        raw = _read_settings_attr(settings, fd)
        if raw is None:
            continue
        try:
            value = _coerce(fd.kind, raw)
        except ValidationError:
            continue
        if value is None:
            continue
        _put((sci, exe, prov), fd, value)

    cfg = RunConfig.from_sections(
        product_version=product_version,
        scientific=sci,
        execution=exe,
        provenance=prov,
    )
    return cfg


def _put(sections: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]],
         fd: FieldDef, value: Any) -> None:
    sci, exe, prov = sections
    target = sci if fd.section == Section.SCIENTIFIC else (
        exe if fd.section == Section.EXECUTION else prov
    )
    target[fd.name] = _json_safe(value)


@dataclass
class ApplyReport:
    """Result of writing a canonical config back onto a settings object."""

    applied: Dict[str, Any] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)


def apply_to_settings(cfg: RunConfig, settings: Any) -> ApplyReport:
    """Restore ``cfg`` onto a settings/state object (map-from).

    Only ``restore_eligible`` fields with a known ``qt_source`` are written.
    Fields whose target attribute is absent from ``settings`` are skipped.
    Unknown fields in the config are reported, never promoted to resume-critical.
    """
    report = ApplyReport()
    for fd in FIELD_DEFS:
        if fd.section == Section.TOP or fd.derived_from:
            continue
        if not fd.restore_eligible or fd.qt_source is None:
            continue
        value = cfg.get(fd.section, fd.name)
        if value is None:
            continue
        if isinstance(settings, Mapping):
            if isinstance(settings, dict):
                settings[fd.qt_source] = value
                report.applied[fd.qt_source] = value
            else:
                report.skipped.append(fd.name)
            continue
        if not hasattr(settings, fd.qt_source):
            report.skipped.append(fd.name)
            continue
        setattr(settings, fd.qt_source, value)
        report.applied[fd.qt_source] = value

    # Surface unknown canonical names in the config (defensive; allowlist).
    known = set(_FIELD_BY_NAME)
    for section in Section.ALL:
        for name in cfg._section(section):
            if name not in known:
                report.unknown.append(name)
    return report


def map_to_backend(cfg: RunConfig) -> Dict[str, Any]:
    """Map a canonical config to backend kwargs (backend/engine names).

    Driven by the same :data:`FIELD_DEFS` (``backend_name``).  Documented value
    transformations: ``winsor_limits`` list -> tuple; ``master_tile_crop_percent``
    percent -> ``master_tile_crop_percent_decimal`` decimal (engine fingerprint).
    """
    out: Dict[str, Any] = {}
    for fd in FIELD_DEFS:
        if fd.section == Section.TOP or not fd.backend_mapped or fd.derived_from:
            continue
        value = cfg.get(fd.section, fd.name)
        if value is None:
            continue
        out[fd.backend_name or fd.name] = _backend_transform(fd, value)
    return out


def _backend_transform(fd: FieldDef, value: Any) -> Any:
    if fd.name == "winsor_limits":
        return tuple(value)
    if fd.name == "master_tile_crop_percent":
        return float(value) / 100.0
    return value


def _backend_to_canonical(fd: FieldDef, value: Any) -> Any:
    """Reverse the documented backend/engine representation into canonical form.

    ``winsor_limits`` arrives as a tuple/list and becomes the canonical list
    (handled by :func:`_coerce`); ``master_tile_crop_percent`` arrives as the
    engine's *decimal* form and is converted to the canonical *percent* unit.
    """
    value = _coerce(fd.kind, value)
    if fd.name == "master_tile_crop_percent":
        # Engine fingerprint attribute ``master_tile_crop_percent_decimal`` is
        # ``percent / 100``; the canonical unit is percent.
        return float(value) * 100.0
    return value


def collect_from_backend(
    backend: Any,
    *,
    product_version: str = "",
) -> RunConfig:
    """Map a configured backend/engine state to a canonical classic RunConfig.

    Reads the classic SUM/W fingerprint fields from the engine instance's
    attributes, keyed by their exact engine attribute name
    (``legacy_fingerprint_key`` where it differs from the canonical name), and
    reverses the two documented engine representations:
    ``master_tile_crop_percent_decimal`` (decimal) -> ``master_tile_crop_percent``
    (percent) and ``winsor_limits`` (tuple) -> canonical list.  Absent/``None``
    fields are omitted from the canonical payload (the authoritative fingerprint
    still hashes them as ``None`` via :func:`classic_fingerprint`).  A present
    but uncoercible runtime value fails closed (:class:`ValidationError`) rather
    than being silently omitted, so the canonical payload can never diverge
    from the engine's authoritative classic fingerprint.  Driven solely by
    :data:`FIELD_DEFS` — there is no parallel hard-coded field list.  No I/O
    occurs.
    """
    sci: Dict[str, Any] = {}
    for fd in FIELD_DEFS:
        if FingerprintDomain.CLASSIC_SUMW not in fd.fingerprint_domains:
            continue
        engine_key = fd.legacy_fingerprint_key or fd.name
        raw = getattr(backend, engine_key, None)
        if raw is None:
            continue
        value = _backend_to_canonical(fd, raw)
        sci[fd.name] = _json_safe(value)
    return RunConfig.from_sections(
        product_version=product_version,
        scientific=sci,
    )


# ---------------------------------------------------------------------------
# Serialisation (atomic write, explicit path)
# ---------------------------------------------------------------------------


def write_cfg(cfg: RunConfig, path: str) -> None:
    """Atomically write the canonical config to ``path`` (caller path only).

    Writes UTF-8 compact JSON + trailing newline via a temp file + ``os.replace``
    in the destination directory.  Rejects non-JSON/NaN/Inf values.  Never
    writes anywhere else and never writes at import/construction time.
    """
    canonical = cfg.to_canonical_dict()
    _scan_unsafe(canonical)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload += "\n"
    data = payload.encode("utf-8")

    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".runcfg-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# V2 reader (fail-closed, read-only)
# ---------------------------------------------------------------------------

_MAX_UNKNOWN_REPORT = 100


@dataclass
class ReadReport:
    """Result of reading a schema-v2 ``.cfg``.

    ``config`` is the validated :class:`RunConfig`.  ``unknown_keys`` reports
    (bounded, name only, never promoted) keys found in the document that do not
    map to a canonical field.  ``diagnostics`` carries bounded non-fatal notes.
    """

    config: RunConfig
    unknown_keys: Tuple[str, ...] = ()
    diagnostics: Tuple[str, ...] = ()


def _reject_json_constant(value: str) -> Any:
    """Reject non-standard JSON constants (``NaN``/``Infinity``/``-Infinity``)."""
    raise ValidationError(f"non-finite JSON number {value!r}")


def _scan_unsafe(value: Any, path: str = "") -> None:
    """Fail closed on any secret/credential-like key, at any nesting depth.

    Only the key *name* (and path) is ever reported; the value is never read,
    compared or included in the error.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            ks = str(k)
            where = f"{path}.{ks}" if path else ks
            if is_unsafe_key(ks):
                raise UnsafeConfigError(f"unsafe key {ks!r} at {where or '<root>'}")
            _scan_unsafe(v, where)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _scan_unsafe(v, f"{path}[{i}]")


def read_cfg(path: str) -> ReadReport:
    """Read and validate a schema-v2 ``.cfg`` (read-only, never writes).

    Fails closed on: unreadable/non-UTF-8 bytes, non-JSON content (including
    ``NaN``/``Infinity``), a non-object top level, ``schema_version`` other than
    2, a non-string ``product_version``, missing/malformed config sections, any
    secret/credential-like key (at any depth, value never reported), and any
    field value that fails coercion.  Unknown keys are reported (bounded) and
    never promoted.  Returns the validated :class:`RunConfig` plus a bounded
    report.
    """
    try:
        with open(os.fspath(path), "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise ValidationError(f"cannot read cfg {path!r}: {exc}") from exc

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"cfg is not valid UTF-8: {exc}") from exc
    try:
        data = json.loads(text, parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise ValidationError(f"cfg is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("cfg top-level is not a JSON object")

    # Secret/credential-like keys anywhere fail closed (name only).
    _scan_unsafe(data)

    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(
            f"unsupported schema_version {data.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION})"
        )

    product_version = data.get("product_version")
    if not isinstance(product_version, str):
        raise ValidationError("product_version must be a string")

    unknown: List[str] = []
    diagnostics: List[str] = []

    # Top-level unknown keys (bounded report).
    _KNOWN_TOP = {"schema_version", "product_version", *Section.ALL}
    for key in data:
        if key not in _KNOWN_TOP:
            unknown.append(str(key))

    sections: Dict[str, Dict[str, Any]] = {}
    for section in Section.ALL:
        raw_section = data.get(section)
        if not isinstance(raw_section, dict):
            raise ValidationError(f"missing or malformed section {section!r}")
        sections[section] = raw_section

    sci: Dict[str, Any] = {}
    exe: Dict[str, Any] = {}
    prov: Dict[str, Any] = {}
    targets = {
        Section.SCIENTIFIC: sci,
        Section.EXECUTION: exe,
        Section.PROVENANCE: prov,
    }
    for section in Section.ALL:
        target = targets[section]
        for key, value in sections[section].items():
            k = str(key)
            fd = _FIELD_BY_NAME.get(k)
            if fd is None or fd.section != section:
                unknown.append(f"{section}.{k}")
                continue
            try:
                coerced = _coerce(fd.kind, value)
            except ValidationError as exc:
                raise ValidationError(f"field {k!r}: {exc}") from exc
            target[k] = _json_safe(coerced)

    # Bound the unknown-key report; never promote any unknown field.
    if len(unknown) > _MAX_UNKNOWN_REPORT:
        diagnostics.append(
            f"unknown-key report truncated to {_MAX_UNKNOWN_REPORT} "
            f"(of {len(unknown)})"
        )
        unknown = unknown[:_MAX_UNKNOWN_REPORT]

    cfg = RunConfig.from_sections(
        product_version=product_version,
        scientific=sci,
        execution=exe,
        provenance=prov,
    )
    return ReadReport(
        config=cfg,
        unknown_keys=tuple(unknown),
        diagnostics=tuple(diagnostics),
    )


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------


def parse_legacy_cfg(path: str) -> Dict[str, Any]:
    """Read a legacy JSON ``.cfg`` (read-only) and return its dict.

    Raises :class:`ValidationError` on unreadable/non-JSON/non-mapping content.
    """
    with open(os.fspath(path), "rb") as fh:
        raw = fh.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValidationError(f"legacy cfg is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("legacy cfg top-level is not a JSON object")
    return data


@dataclass
class LegacyMigrationResult:
    """Configuration restoration/reproducibility object from a legacy CFG.

    Explicitly **not** a scientific checkpoint or resumable claim: ``resumable``
    is always ``False`` for a legacy-CFG-only migration.
    """

    ok: bool
    config: Optional[RunConfig]
    product_version: str
    classifications: Dict[str, str] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    resumable: bool = False


def _mode_values_equivalent(a: Any, b: Any) -> bool:
    """Compare two coerced values of the same field for alias-consistency.

    String tokens are normalised ('_' vs '-' and case) before comparison; other
    types compare by equality.
    """
    if isinstance(a, str) and isinstance(b, str):
        return _normalize_mode_token(a) == _normalize_mode_token(b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) == list(b)
    return a == b


def migrate_legacy(data: Mapping[str, Any]) -> LegacyMigrationResult:
    """Migrate a legacy 5.x JSON ``.cfg`` dict into a canonical config.

    Classification and mapping rules:

    * every input key is classified ``mapped``/``renamed``/``obsolete``/
      ``nonscientific``/``unknown``/``unsafe``;
    * secret keys are classified ``unsafe`` and **never** serialised (only the
      key name may appear in diagnostics);
    * known semantics are mapped through :data:`FIELD_DEFS`; when several legacy
      aliases target one resume-critical field with conflicting values the
      migration **fails closed** (:class:`AmbiguousLegacyError`);
    * a resume-critical field whose value is unsafe/secret fails closed
      (:class:`UnsafeLegacyError`);
    * a required resume-critical value that cannot be coerced fails closed
      (:class:`ValidationError`).

    The result is a restoration/reproducibility object only.
    """
    if not isinstance(data, Mapping):
        raise ValidationError("legacy cfg is not a JSON object")

    classifications: Dict[str, int] = {c: 0 for c in LegacyClass.ALL}
    per_key: Dict[str, str] = {}
    diagnostics: List[str] = []

    # Group present legacy values by their target canonical field.
    gathered: Dict[str, Dict[str, Any]] = {}
    product_version = ""

    for key, value in data.items():
        fd, klass = classify_legacy_key(str(key))
        if klass == LegacyClass.UNSAFE:
            # Name only; never the value.
            classifications[LegacyClass.UNSAFE] += 1
            per_key[str(key)] = LegacyClass.UNSAFE
            diagnostics.append(f"unsafe key excluded: {key!r}")
            continue
        classifications[klass] += 1
        per_key[str(key)] = klass

        if fd is None:
            # obsolete / nonscientific / unknown — no canonical target.
            if klass == LegacyClass.OBSOLETE:
                diagnostics.append(f"obsolete key ignored: {key!r}")
            elif klass == LegacyClass.NONSCIENTIFIC:
                diagnostics.append(f"non-scientific key ignored: {key!r}")
            else:
                diagnostics.append(f"unknown key ignored: {key!r}")
            continue

        if fd.section == Section.TOP:
            product_version = _coerce(fd.kind, value) or ""
            continue

        gathered.setdefault(fd.name, {})[str(key)] = value

    # Build canonical sections, detecting alias conflicts on resume-critical
    # fields and failing closed on unsafe/ambiguous resume-critical values.
    sci: Dict[str, Any] = {}
    exe: Dict[str, Any] = {}
    prov: Dict[str, Any] = {}

    for fd in FIELD_DEFS:
        if fd.section == Section.TOP or fd.derived_from:
            continue
        if fd.name not in gathered:
            continue
        raw_values = gathered[fd.name]

        # Collapse every alias value through coercion and check consistency.
        coerced_values = []
        for alias, raw in raw_values.items():
            try:
                cv = _coerce(fd.kind, raw)
            except ValidationError as exc:
                if _is_classic_fingerprint(fd):
                    raise ValidationError(
                        f"resume-critical field {fd.name!r} has uncoercible "
                        f"value from {alias!r}: {exc}"
                    ) from exc
                diagnostics.append(
                    f"field {fd.name!r} (alias {alias!r}) uncoercible; skipped: {exc}"
                )
                continue
            if cv is None:
                continue
            coerced_values.append((alias, cv))

        if not coerced_values:
            continue

        first_alias, first_value = coerced_values[0]
        for alias, cv in coerced_values[1:]:
            if not _mode_values_equivalent(first_value, cv):
                if _is_classic_fingerprint(fd):
                    raise AmbiguousLegacyError(
                        f"resume-critical field {fd.name!r} has conflicting "
                        f"legacy values ({first_alias!r} vs {alias!r})"
                    )
                diagnostics.append(
                    f"field {fd.name!r} has conflicting legacy values "
                    f"({first_alias!r} vs {alias!r}); keeping {first_alias!r}"
                )

        value = first_value

        if _is_classic_fingerprint(fd) and isinstance(value, str) and is_unsafe_key(str(value)):
            # Defensive: a resume-critical field must never hold secret material.
            raise UnsafeLegacyError(
                f"resume-critical field {fd.name!r} carries an unsafe value"
            )

        _put((sci, exe, prov), fd, value)

    cfg = RunConfig.from_sections(
        product_version=product_version,
        scientific=sci,
        execution=exe,
        provenance=prov,
    )

    return LegacyMigrationResult(
        ok=True,
        config=cfg,
        product_version=product_version,
        classifications=per_key,
        diagnostics=diagnostics,
        counts=classifications,
        resumable=False,
    )
