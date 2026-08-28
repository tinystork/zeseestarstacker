"""Run CFG v2 contract tests: ``seestar.run_contract``.

Covers the canonical per-run configuration model, its single field-definition
source, deterministic serialisation / scientific fingerprint / digest, the
bounded diff, the settings/backend mappings, and the legacy 5.x ``.cfg``
migration — all in complete isolation (no Qt, no Tk, no numpy, no astropy).
"""

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures"

# Real witness path (may be absent in CI / on other machines).
WITNESS = Path("/media/tristan/X10 Pro/M16/out/_stack_20250709_194138.cfg")
WITNESS_SHA = "0985a53e61db8acedc94d554d00a9f967392b0203f3b9e9a94c47b0305a79626"


# ---------------------------------------------------------------------------
# Load seestar.run_contract with heavy deps blocked (import hygiene).
# ---------------------------------------------------------------------------
def _load_contract():
    blocked = {}
    for name in (
        "numpy",
        "astropy",
        "scipy",
        "PySide6",
        "PyQt5",
        "PyQt6",
        "tkinter",
        "cv2",
    ):
        blocked[name] = sys.modules.get(name)
        sys.modules[name] = None
    try:
        spec = importlib.util.spec_from_file_location(
            "seestar_run_contract", ROOT / "seestar" / "run_contract.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["seestar_run_contract"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, orig in blocked.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


rc = _load_contract()


def _config(**sections) -> "rc.RunConfig":
    return rc.RunConfig.from_sections(**sections)


# ---------------------------------------------------------------------------
# 1. Import hygiene and schema shape
# ---------------------------------------------------------------------------
def test_module_is_pure_stdlib():
    # The module source must not import any GUI/engine/scientific dependency.
    src = (ROOT / "seestar" / "run_contract.py").read_text(encoding="utf-8")
    for forbidden in ("import numpy", "import astropy", "import PySide6",
                      "import tkinter", "import scipy", "import cv2",
                      "from numpy", "from astropy"):
        assert forbidden not in src, f"forbidden import {forbidden!r} found"


def test_package_import_is_cheap():
    # ``import seestar.run_contract`` must not pull the engine or GUI.
    sys.path.insert(0, str(ROOT))
    try:
        import seestar.run_contract as pkg_rc  # noqa: F401
    finally:
        sys.path.pop(0)
    assert pkg_rc.SCHEMA_VERSION == 2


def test_schema_v2_shape():
    cfg = _config(
        product_version="8.2.0",
        scientific={"stacking_mode": "kappa-sigma", "kappa": 2.5},
        execution={"input_folder": "/tmp/in"},
    )
    d = cfg.to_canonical_dict()
    assert d["schema_version"] == 2
    assert d["product_version"] == "8.2.0"
    assert set(d) == {"schema_version", "product_version",
                      "scientific_config", "execution_config", "provenance"}
    assert d["scientific_config"]["stacking_mode"] == "kappa-sigma"
    assert d["execution_config"]["input_folder"] == "/tmp/in"
    assert d["provenance"] == {}


# ---------------------------------------------------------------------------
# 2. Single field-definition source
# ---------------------------------------------------------------------------
def test_field_defs_capture_required_attributes():
    assert len(rc.FIELD_DEFS) >= 90
    names = set()
    for fd in rc.FIELD_DEFS:
        assert fd.name
        assert fd.section in (rc.Section.SCIENTIFIC, rc.Section.EXECUTION,
                              rc.Section.PROVENANCE, rc.Section.TOP)
        assert fd.kind
        # presence is one of the documented values
        assert fd.presence in (rc.PRESENCE_ALWAYS, rc.PRESENCE_CHECKPOINT,
                               rc.PRESENCE_OPTIONAL)
        assert fd.name not in names, f"duplicate canonical field {fd.name}"
        names.add(fd.name)


def test_classic_fingerprint_coverage():
    """The canonical fingerprint fields cover the classic
    ``_RESUME_FINGERPRINT_ATTRS`` (with the documented decimal->percent rename).
    """
    classic = {
        "stacking_mode", "kappa", "stack_kappa_low", "stack_kappa_high",
        "winsor_limits", "normalize_method", "weighting_method",
        "use_quality_weighting", "weight_by_snr", "weight_by_stars",
        "snr_exponent", "stars_exponent", "min_weight", "correct_hot_pixels",
        "hot_pixel_threshold", "neighborhood_size", "bayer_pattern",
        "batch_size", "chunk_size", "apply_batch_feathering", "apply_feathering",
        "feather_blur_px", "apply_master_tile_crop", "master_tile_crop_percent",
        "apply_low_wht_mask", "low_wht_percentile", "low_wht_soften_px",
    }
    assert rc.classic_fingerprint_names() == classic


def test_scientific_allowlist_covers_drizzle_contract():
    sci_names = {fd.name for fd in rc.FIELD_DEFS if fd.section == rc.Section.SCIENTIFIC}
    required = {
        "use_drizzle", "drizzle_mode", "drizzle_processing_policy",
        "drizzle_scale_requested", "drizzle_scale_effective",
        "drizzle_kernel_requested", "drizzle_kernel_effective",
        "drizzle_pixfrac_requested", "drizzle_pixfrac_effective",
        "drizzle_wht_threshold_requested", "drizzle_wht_threshold_effective",
        "drizzle_wht_policy", "drizzle_fillval",
        "bayer_pattern", "correct_hot_pixels", "hot_pixel_threshold",
        "neighborhood_size", "use_quality_weighting",
        "background_match_contract", "background_match_contract_version",
        "output_grid_contract", "output_grid_contract_version",
        "registration_contract", "registration_contract_version",
    }
    assert required <= sci_names


def test_runtime_effective_values_are_checkpoint_or_optional():
    """Runtime-effective drizzle values must not be claimed from settings."""
    for fd in rc.FIELD_DEFS:
        if fd.name.endswith("_effective") or fd.name in (
            "drizzle_fillval", "drizzle_lib_version", "drizzle_wht_policy",
        ):
            assert fd.presence in (rc.PRESENCE_CHECKPOINT, rc.PRESENCE_OPTIONAL), fd.name


# ---------------------------------------------------------------------------
# 3. Determinism, fingerprint, digest
# ---------------------------------------------------------------------------
def test_deterministic_bytes_and_digests():
    a = _config(
        product_version="8.2.0",
        scientific={"stacking_mode": "kappa-sigma", "kappa": 2.5,
                    "winsor_limits": (0.05, 0.05), "use_drizzle": True},
        execution={"input_folder": "/tmp/in"},
    )
    b = _config(
        product_version="8.2.0",
        execution={"input_folder": "/tmp/in"},
        scientific={"use_drizzle": True, "winsor_limits": (0.05, 0.05),
                    "kappa": 2.5, "stacking_mode": "kappa-sigma"},
    )
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.full_digest() == b.full_digest()
    assert a.scientific_fingerprint() == b.scientific_fingerprint()
    # Bytes are UTF-8, compact JSON, deterministic key order.
    parsed = json.loads(a.to_canonical_bytes().decode("utf-8"))
    assert parsed["schema_version"] == 2
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_fingerprint_only_tracks_fingerprint_fields():
    base = dict(scientific={"stacking_mode": "kappa-sigma", "kappa": 2.5})
    a = _config(scientific={**base["scientific"], "apply_final_scnr": False})
    b = _config(scientific={**base["scientific"], "apply_final_scnr": True})
    # SCNR is a non-fingerprint scientific field: fingerprint unchanged...
    assert a.scientific_fingerprint() == b.scientific_fingerprint()
    # ...but the whole-config digest differs.
    assert a.full_digest() != b.full_digest()

    c = _config(scientific={**base["scientific"], "kappa": 2.5})
    d = _config(scientific={**base["scientific"], "kappa": 3.0})
    assert c.scientific_fingerprint() != d.scientific_fingerprint()


def test_reject_nan_inf_and_non_json():
    with pytest.raises(rc.ValidationError):
        _config(scientific={"kappa": float("nan")})
    with pytest.raises(rc.ValidationError):
        _config(scientific={"kappa": float("inf")})
    with pytest.raises(rc.ValidationError):
        _config(execution={"mosaic_settings": {"x": object()}})
    with pytest.raises(rc.ValidationError):
        _config(scientific={"winsor_limits": b"bytes"})


def test_no_io_during_construction(tmp_path):
    os.chdir(tmp_path)
    try:
        cfg = _config(scientific={"stacking_mode": "mean"})
        cfg.to_canonical_bytes()
        cfg.scientific_fingerprint()
        cfg.full_digest()
    finally:
        os.chdir(str(ROOT))
    assert list(tmp_path.iterdir()) == []


def test_derived_processing_policy():
    cfg = _config(scientific={"drizzle_mode": "Incremental"})
    assert cfg.scientific["drizzle_processing_policy"] == "incremental"
    cfg2 = _config(scientific={"drizzle_mode": "Final"})
    assert cfg2.scientific["drizzle_processing_policy"] == "standard"


# ---------------------------------------------------------------------------
# 4. Bounded diff
# ---------------------------------------------------------------------------
def test_diff_returns_bounded_list():
    a = _config(scientific={"kappa": 2.5, "stacking_mode": "mean"})
    b = _config(scientific={"kappa": 3.0, "stacking_mode": "median"})
    r = rc.diff_configs(a, b)
    assert r.total == 2
    assert not r.truncated
    assert {d.field for d in r.diffs} == {"kappa", "stacking_mode"}


def test_diff_truncates():
    a = _config(scientific={"kappa": 2.5, "stacking_mode": "mean",
                            "snr_exponent": 1.0, "min_weight": 0.1})
    b = _config(scientific={"kappa": 3.0, "stacking_mode": "median",
                            "snr_exponent": 2.0, "min_weight": 0.2})
    r = rc.diff_configs(a, b, limit=2)
    assert r.truncated is True
    assert r.total == 4
    assert len(r.diffs) == 2


def test_diff_sees_product_version():
    a = _config(product_version="8.1.0")
    b = _config(product_version="8.2.0")
    r = rc.diff_configs(a, b)
    assert r.total == 1
    assert r.diffs[0].field == "product_version"


# ---------------------------------------------------------------------------
# 5. Settings / backend mappings
# ---------------------------------------------------------------------------
def test_collect_from_settings_dict():
    settings = {
        "stacking_mode": "kappa-sigma",
        "kappa": 2.5,
        "stack_winsor_limits": "0.10,0.20",
        "stack_norm_method": "linear_fit",
        "stack_weight_method": "noise_variance",
        "drizzle_mode": "Incremental",
        "drizzle_scale": 2,
        "save_final_as_float32": True,
        "input_folder": "/tmp/in",
        "cleanup_temp": False,
        "mosaic_settings": {"kernel": "square"},
    }
    cfg = rc.collect_from_settings(settings)
    assert cfg.scientific["stacking_mode"] == "kappa-sigma"
    assert cfg.scientific["winsor_limits"] == [0.10, 0.20]
    assert cfg.scientific["normalize_method"] == "linear_fit"
    assert cfg.scientific["weighting_method"] == "noise_variance"
    assert cfg.scientific["drizzle_processing_policy"] == "incremental"
    assert cfg.execution["save_as_float32"] is True
    assert cfg.execution["input_folder"] == "/tmp/in"
    assert cfg.execution["cleanup_temp"] is False


def test_apply_to_settings_restore_and_unknown_report():
    cfg = _config(
        scientific={"stacking_mode": "median", "kappa": 3.0},
        execution={"input_folder": "/tmp/in"},
    )
    state = {}
    report = rc.apply_to_settings(cfg, state)
    assert state["stacking_mode"] == "median"
    assert state["input_folder"] == "/tmp/in"
    assert report.unknown == []


def test_apply_to_settings_skips_missing_attrs():
    cfg = _config(scientific={"kappa": 3.0})
    obj = types.SimpleNamespace()
    report = rc.apply_to_settings(cfg, obj)
    # kappa is restore-eligible and obj has no attr -> skipped, no crash.
    assert report.applied == {}
    assert "kappa" in report.skipped


def test_map_to_backend_names_and_transforms():
    cfg = _config(
        scientific={
            "winsor_limits": [0.05, 0.05],
            "normalize_method": "linear_fit",
            "master_tile_crop_percent": 18.0,
            "snr_exponent": 1.8,
        },
        execution={
            "input_folder": "/tmp/in",
            "output_folder": "/tmp/out",
            "reference_image_path": "/tmp/ref.fit",
            "cleanup_temp": True,
            "save_as_float32": False,
            "mosaic_mode_active": True,
        },
    )
    kw = rc.map_to_backend(cfg)
    assert kw["winsor_limits"] == (0.05, 0.05)
    assert kw["normalize_method"] == "linear_fit"
    assert kw["master_tile_crop_percent"] == pytest.approx(0.18)
    assert kw["snr_exp"] == 1.8
    assert kw["input_dir"] == "/tmp/in"
    assert kw["output_dir"] == "/tmp/out"
    assert kw["reference_path_ui"] == "/tmp/ref.fit"
    assert kw["perform_cleanup"] is True
    assert kw["save_as_float32"] is False
    assert kw["is_mosaic_run"] is True


# ---------------------------------------------------------------------------
# 6. Legacy migration (sanitized fixture)
# ---------------------------------------------------------------------------
def _load_fixture():
    path = FIXTURE_DIR / "legacy_stack_5.6.0_sanitized.cfg"
    return rc.parse_legacy_cfg(str(path)), path


def test_sanitized_fixture_classification_counts():
    data, _ = _load_fixture()
    res = rc.migrate_legacy(data)
    assert res.ok is True
    assert res.resumable is False
    assert res.counts[rc.LegacyClass.UNKNOWN] == 0
    assert res.counts[rc.LegacyClass.UNSAFE] == 1
    assert res.counts[rc.LegacyClass.OBSOLETE] >= 3
    assert res.counts[rc.LegacyClass.NONSCIENTIFIC] >= 8
    assert res.counts[rc.LegacyClass.MAPPED] > 0
    assert res.counts[rc.LegacyClass.RENAMED] > 0
    # every fixture key is classified
    assert set(res.classifications) == set(data)


def test_sanitized_fixture_secret_excluded():
    data, _ = _load_fixture()
    res = rc.migrate_legacy(data)
    assert res.classifications["astrometry_api_key"] == rc.LegacyClass.UNSAFE
    blob = json.dumps(res.config.to_canonical_dict(), sort_keys=True)
    assert "astrometry_api_key" not in blob
    assert "REDACTED_PLACEHOLDER" not in blob
    # name may appear in diagnostics, value never does
    for diag in res.diagnostics:
        assert "REDACTED_PLACEHOLDER" not in diag


def test_sanitized_fixture_maps_known_semantics():
    data, _ = _load_fixture()
    res = rc.migrate_legacy(data)
    cfg = res.config
    # stacking_mode / stack_method / stack_reject_algo collapse consistently
    assert cfg.scientific["stacking_mode"] == "winsorized-sigma-clip"
    assert cfg.scientific["normalize_method"] == "linear_fit"
    assert cfg.scientific["weighting_method"] == "noise_variance"
    assert cfg.scientific["winsor_limits"] == [0.05, 0.05]
    assert cfg.scientific["drizzle_processing_policy"] == "standard"
    assert cfg.scientific["drizzle_scale_requested"] == 2.0
    assert cfg.scientific["drizzle_wht_threshold_requested"] == 0.7
    assert cfg.execution["save_as_float32"] is False
    assert cfg.product_version == "5.6.0"


def test_migrate_rejects_non_mapping():
    with pytest.raises(rc.ValidationError):
        rc.migrate_legacy([1, 2, 3])


def test_ambiguous_resume_critical_alias_fails_closed():
    data = {
        "version": "5.6.0",
        "stacking_mode": "kappa-sigma",
        "stack_method": "mean",  # conflicts with stacking_mode
    }
    with pytest.raises(rc.AmbiguousLegacyError):
        rc.migrate_legacy(data)


def test_uncoercible_resume_critical_fails_closed():
    data = {"version": "5.6.0", "kappa": "not-a-number"}
    with pytest.raises(rc.ValidationError):
        rc.migrate_legacy(data)


def test_legacy_cfg_never_resumable():
    data, _ = _load_fixture()
    res = rc.migrate_legacy(data)
    assert res.resumable is False
    # The result is a restoration/reproducibility object, not a checkpoint.
    assert isinstance(res.config, rc.RunConfig)


# ---------------------------------------------------------------------------
# 7. Atomic write (explicit path only)
# ---------------------------------------------------------------------------
def test_write_cfg_atomic_roundtrip(tmp_path):
    cfg = _config(
        product_version="8.2.0",
        scientific={"stacking_mode": "kappa-sigma", "kappa": 2.5},
        execution={"input_folder": "/tmp/in"},
    )
    target = tmp_path / "run_config.cfg"
    rc.write_cfg(cfg, str(target))
    raw = target.read_bytes()
    assert raw.endswith(b"\n")
    # deterministic JSON, no partial temp files left behind
    assert not list(tmp_path.glob(".runcfg-*.tmp"))
    reloaded = rc.parse_legacy_cfg(str(target))
    assert reloaded["schema_version"] == 2
    assert reloaded["product_version"] == "8.2.0"


def test_write_cfg_rejects_nan(tmp_path):
    # NaN is rejected at construction, before any write.
    with pytest.raises(rc.ValidationError):
        _config(scientific={"kappa": float("nan")})


# ---------------------------------------------------------------------------
# 8. Real 5.6.0 witness (skipped cleanly when absent)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not WITNESS.exists(), reason="real legacy witness not mounted")
def test_real_witness_full_classification_and_secret_exclusion():
    raw = WITNESS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == WITNESS_SHA
    data = rc.parse_legacy_cfg(str(WITNESS))
    assert data.get("version") == "5.6.0"
    assert "astrometry_api_key" in data

    res = rc.migrate_legacy(data)
    assert res.ok is True
    assert res.resumable is False
    # every key classified, none unknown
    assert set(res.classifications) == set(data)
    assert res.counts[rc.LegacyClass.UNKNOWN] == 0
    assert res.counts[rc.LegacyClass.UNSAFE] == 1

    # secret key classified unsafe and absent from every serialised/digestable
    # payload; only its key *name* is ever reported, never its value.
    secret_value = data["astrometry_api_key"]
    assert res.classifications["astrometry_api_key"] == rc.LegacyClass.UNSAFE
    canonical = res.config.to_canonical_dict()
    blob = json.dumps(canonical, sort_keys=True)
    assert secret_value not in blob
    assert "astrometry_api_key" not in blob
    for diag in res.diagnostics:
        assert secret_value not in diag

    # The migrated config is a reproducibility object only, never a resumable
    # claim; the fingerprint fields that are present are consistent.
    assert res.config.product_version == "5.6.0"
    assert res.config.scientific["stacking_mode"] == "winsorized-sigma-clip"
