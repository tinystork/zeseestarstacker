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
    assert a.classic_fingerprint() == b.classic_fingerprint()
    # Bytes are UTF-8, compact JSON, deterministic key order.
    parsed = json.loads(a.to_canonical_bytes().decode("utf-8"))
    assert parsed["schema_version"] == 2
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_fingerprint_only_tracks_fingerprint_fields():
    base = dict(scientific={"stacking_mode": "kappa-sigma", "kappa": 2.5})
    a = _config(scientific={**base["scientific"], "apply_final_scnr": False})
    b = _config(scientific={**base["scientific"], "apply_final_scnr": True})
    # SCNR is a non-fingerprint scientific field: fingerprint unchanged...
    assert a.classic_fingerprint() == b.classic_fingerprint()
    # ...but the whole-config digest differs.
    assert a.full_digest() != b.full_digest()

    c = _config(scientific={**base["scientific"], "kappa": 2.5})
    d = _config(scientific={**base["scientific"], "kappa": 3.0})
    assert c.classic_fingerprint() != d.classic_fingerprint()


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
        cfg.classic_fingerprint()
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


# ---------------------------------------------------------------------------
# 9. V2 reader (fail-closed, read-only)
# ---------------------------------------------------------------------------
def _full_config():
    return _config(
        product_version="8.2.0",
        scientific={
            "stacking_mode": "kappa-sigma", "kappa": 2.5,
            "winsor_limits": [0.05, 0.05], "chunk_size": None,
            "use_drizzle": True,
        },
        execution={"input_folder": "/tmp/in", "save_as_float32": True},
        provenance={"producer": "zeseestarstacker"},
    )


def test_read_cfg_roundtrip_bytes_and_digest(tmp_path):
    cfg = _full_config()
    target = tmp_path / "run_config.cfg"
    rc.write_cfg(cfg, str(target))
    report = rc.read_cfg(str(target))
    reloaded = report.config
    assert reloaded.to_canonical_bytes() == cfg.to_canonical_bytes()
    assert reloaded.full_digest() == cfg.full_digest()
    assert reloaded.classic_fingerprint() == cfg.classic_fingerprint()
    assert report.unknown_keys == ()
    # a null optional field (chunk_size) round-trips as null, not as absent.
    assert "chunk_size" in reloaded.scientific
    assert reloaded.scientific["chunk_size"] is None


def test_read_cfg_wrong_schema_rejected(tmp_path):
    target = tmp_path / "bad.cfg"
    target.write_text(json.dumps({
        "schema_version": 3,
        "product_version": "8.2.0",
        "scientific_config": {}, "execution_config": {}, "provenance": {},
    }), encoding="utf-8")
    with pytest.raises(rc.ValidationError):
        rc.read_cfg(str(target))


def test_read_cfg_missing_schema_rejected(tmp_path):
    target = tmp_path / "bad.cfg"
    target.write_text(json.dumps({
        "product_version": "8.2.0",
        "scientific_config": {}, "execution_config": {}, "provenance": {},
    }), encoding="utf-8")
    with pytest.raises(rc.ValidationError):
        rc.read_cfg(str(target))


def test_read_cfg_missing_section_rejected(tmp_path):
    target = tmp_path / "bad.cfg"
    target.write_text(json.dumps({
        "schema_version": 2, "product_version": "8.2.0",
        "scientific_config": {}, "execution_config": {},
    }), encoding="utf-8")
    with pytest.raises(rc.ValidationError):
        rc.read_cfg(str(target))


def test_read_cfg_product_version_not_string_rejected(tmp_path):
    target = tmp_path / "bad.cfg"
    target.write_text(json.dumps({
        "schema_version": 2, "product_version": 820,
        "scientific_config": {}, "execution_config": {}, "provenance": {},
    }), encoding="utf-8")
    with pytest.raises(rc.ValidationError):
        rc.read_cfg(str(target))


def test_read_cfg_corrupt_json_rejected(tmp_path):
    target = tmp_path / "bad.cfg"
    target.write_text('{"schema_version": 2, "product_version": ', encoding="utf-8")
    with pytest.raises(rc.ValidationError):
        rc.read_cfg(str(target))


def test_read_cfg_non_finite_json_rejected(tmp_path):
    target = tmp_path / "bad.cfg"
    target.write_text(
        '{"schema_version": 2, "product_version": "8.2.0", '
        '"scientific_config": {"kappa": NaN}, '
        '"execution_config": {}, "provenance": {}}',
        encoding="utf-8",
    )
    with pytest.raises(rc.ValidationError):
        rc.read_cfg(str(target))


def test_read_cfg_type_error_rejected(tmp_path):
    target = tmp_path / "bad.cfg"
    target.write_text(json.dumps({
        "schema_version": 2, "product_version": "8.2.0",
        "scientific_config": {"kappa": "not-a-number"},
        "execution_config": {}, "provenance": {},
    }), encoding="utf-8")
    with pytest.raises(rc.ValidationError):
        rc.read_cfg(str(target))


def test_read_cfg_unsafe_key_fails_closed_without_value(tmp_path):
    secret = "REDACTED_PLACEHOLDER_123"
    target = tmp_path / "bad.cfg"
    target.write_text(json.dumps({
        "schema_version": 2, "product_version": "8.2.0",
        "scientific_config": {"kappa": 2.5},
        "execution_config": {"astrometry_api_key": secret},
        "provenance": {},
    }), encoding="utf-8")
    with pytest.raises(rc.UnsafeConfigError) as excinfo:
        rc.read_cfg(str(target))
    # the key name is reported, the secret value never is
    assert "astrometry_api_key" in str(excinfo.value)
    assert secret not in str(excinfo.value)


def test_read_cfg_unsafe_key_nested_fails_closed(tmp_path):
    secret = "NESTED_TOKEN_XYZ"
    target = tmp_path / "bad.cfg"
    target.write_text(json.dumps({
        "schema_version": 2, "product_version": "8.2.0",
        "scientific_config": {},
        "execution_config": {"mosaic_settings": {"api_token": secret}},
        "provenance": {},
    }), encoding="utf-8")
    with pytest.raises(rc.UnsafeConfigError) as excinfo:
        rc.read_cfg(str(target))
    assert secret not in str(excinfo.value)


def test_read_cfg_unknown_keys_reported_never_promoted(tmp_path):
    target = tmp_path / "cfg.cfg"
    target.write_text(json.dumps({
        "schema_version": 2, "product_version": "8.2.0",
        "scientific_config": {"kappa": 2.5, "bogus_science": 1},
        "execution_config": {"input_folder": "/tmp", "extra_exec": 2},
        "provenance": {},
        "extra_top": 3,
    }), encoding="utf-8")
    report = rc.read_cfg(str(target))
    assert "scientific_config.bogus_science" in report.unknown_keys
    assert "execution_config.extra_exec" in report.unknown_keys
    assert "extra_top" in report.unknown_keys
    # never promoted into the canonical model
    assert report.config.scientific.get("bogus_science") is None
    assert report.config.execution.get("extra_exec") is None
    assert report.config.scientific["kappa"] == 2.5


def test_read_cfg_never_writes(tmp_path):
    target = tmp_path / "cfg.cfg"
    target.write_text(json.dumps({
        "schema_version": 2, "product_version": "8.2.0",
        "scientific_config": {}, "execution_config": {}, "provenance": {},
    }), encoding="utf-8")
    before = sorted(p.name for p in tmp_path.iterdir())
    rc.read_cfg(str(target))
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after == ["cfg.cfg"]


def test_write_cfg_atomic_failure_raises(tmp_path):
    cfg = _full_config()
    missing_dir = tmp_path / "does" / "not" / "exist"
    with pytest.raises(OSError):
        rc.write_cfg(cfg, str(missing_dir / "run_config.cfg"))
    assert not missing_dir.exists()


# ---------------------------------------------------------------------------
# 9b. Strict-null semantics (v2 reader fail-closed)
# ---------------------------------------------------------------------------
def _write_v2_doc(tmp_path, *, scientific=None, execution=None, provenance=None,
                  filename="strict_null.cfg"):
    doc = {
        "schema_version": 2,
        "product_version": "8.2.0",
        "scientific_config": scientific or {},
        "execution_config": execution or {},
        "provenance": provenance or {},
    }
    target = tmp_path / filename
    target.write_text(json.dumps(doc), encoding="utf-8")
    return target


@pytest.mark.parametrize(
    "section, field",
    [
        ("scientific_config", "stacking_mode"),    # KIND_STR
        ("scientific_config", "kappa"),            # KIND_FLOAT
        ("scientific_config", "neighborhood_size"),  # KIND_INT
        ("scientific_config", "use_drizzle"),      # KIND_BOOL
        ("scientific_config", "winsor_limits"),    # KIND_WINSOR
        ("execution_config", "mosaic_settings"),   # KIND_DICT
        ("execution_config", "input_folder"),      # KIND_STR (execution)
    ],
)
def test_read_cfg_rejects_null_for_strict_kinds(tmp_path, section, field):
    sections = {
        "scientific_config": {},
        "execution_config": {},
    }
    sections[section][field] = None
    target = _write_v2_doc(
        tmp_path,
        scientific=sections["scientific_config"],
        execution=sections["execution_config"],
    )
    with pytest.raises(rc.ValidationError) as excinfo:
        rc.read_cfg(str(target))
    # the field name is surfaced; no secret value can be involved
    assert field in str(excinfo.value)


@pytest.mark.parametrize(
    "section, field, kind",
    [
        ("scientific_config", "chunk_size", "int_or_none"),
        ("execution_config", "num_processing_workers", "int_or_none"),
        ("scientific_config", "match_background_for_final", "bool_or_none"),
    ],
)
def test_read_cfg_accepts_null_for_nullable_kinds(tmp_path, section, field, kind):
    sections = {
        "scientific_config": {},
        "execution_config": {},
    }
    sections[section][field] = None
    target = _write_v2_doc(
        tmp_path,
        scientific=sections["scientific_config"],
        execution=sections["execution_config"],
    )
    report = rc.read_cfg(str(target))
    section_map = {
        "scientific_config": report.config.scientific,
        "execution_config": report.config.execution,
    }
    # null is preserved (round-trippable), never dropped nor rejected.
    assert field in section_map[section]
    assert section_map[section][field] is None


def test_coerce_strict_kinds_reject_none():
    """Every strictly typed kind rejects explicit ``None`` (including kinds
    with no reader-reachable field, e.g. ``KIND_LIST``)."""
    for kind in (rc.KIND_STR, rc.KIND_INT, rc.KIND_FLOAT, rc.KIND_BOOL,
                 rc.KIND_WINSOR, rc.KIND_LIST, rc.KIND_DICT):
        with pytest.raises(rc.ValidationError):
            rc._coerce(kind, None)


def test_coerce_nullable_kinds_preserve_none():
    """Every nullable kind preserves explicit ``None`` (including kinds with no
    reader-reachable field, e.g. ``KIND_STR_OR_NONE``/``KIND_FLOAT_OR_NONE``)."""
    for kind in (rc.KIND_INT_OR_NONE, rc.KIND_FLOAT_OR_NONE,
                 rc.KIND_BOOL_OR_NONE, rc.KIND_STR_OR_NONE):
        assert rc._coerce(kind, None) is None


# ---------------------------------------------------------------------------
# 10. Fingerprint domains (classic v1 vs drizzle effective contract)
# ---------------------------------------------------------------------------
def _drizzle_complete(**overrides):
    sci = {
        "weighting_method": "noise_variance",
        "use_quality_weighting": False,
        "weight_by_snr": True,
        "weight_by_stars": False,
        "snr_exponent": 1.0,
        "stars_exponent": 0.5,
        "min_weight": 0.1,
        "correct_hot_pixels": True,
        "hot_pixel_threshold": 3.0,
        "neighborhood_size": 5,
        "bayer_pattern": "GRBG",
        "drizzle_scale_effective": 2.0,
        "drizzle_kernel_effective": "square",
        "drizzle_pixfrac_effective": 1.0,
        "drizzle_wht_threshold_effective": 0.7,
        "drizzle_wht_policy": "relative_coverage_v1",
        "drizzle_fillval": "indef",
        "drizzle_double_norm_fix": False,
        "background_match_contract": "bgmatch_v1",
        "background_match_contract_version": 1,
        "output_grid_contract": "grid_v1",
        "output_grid_contract_version": 1,
        "registration_contract": "reg_v1",
        "registration_contract_version": 1,
    }
    sci.update(overrides)
    return _config(scientific=sci)


def test_scientific_fingerprint_requires_domain():
    cfg = _config(scientific={"kappa": 2.5})
    with pytest.raises(TypeError):
        rc.scientific_fingerprint(cfg)  # noqa: domain is required
    with pytest.raises(rc.ValidationError):
        rc.scientific_fingerprint(cfg, domain="bogus")


def test_fingerprint_domain_helpers():
    classic = rc.fingerprint_field_defs(rc.FingerprintDomain.CLASSIC_SUMW)
    drizzle = rc.fingerprint_field_defs(rc.FingerprintDomain.DRIZZLE)
    assert len(classic) == 27
    assert rc.required_fingerprint_names(rc.FingerprintDomain.CLASSIC_SUMW) == \
        rc.classic_fingerprint_names()
    assert len(drizzle) >= 24
    # shared fields live in both domains
    shared = rc.classic_fingerprint_names() & rc.required_fingerprint_names(
        rc.FingerprintDomain.DRIZZLE
    )
    assert "bayer_pattern" in shared
    assert "min_weight" in shared
    with pytest.raises(rc.ValidationError):
        rc.fingerprint_field_defs("nope")


def test_drizzle_fingerprint_refuses_missing_effective_field():
    for missing in (
        "drizzle_scale_effective",
        "drizzle_wht_policy",
        "background_match_contract",
        "registration_contract_version",
        "bayer_pattern",
        "min_weight",
    ):
        sci = _drizzle_complete().scientific.copy()
        del sci[missing]
        cfg = _config(scientific=sci)
        with pytest.raises(rc.ValidationError):
            cfg.drizzle_fingerprint()


def test_drizzle_fingerprint_changes_per_effective_class():
    base = _drizzle_complete()
    base_fp = base.drizzle_fingerprint()
    variants = [
        {"bayer_pattern": "RGGB"},                       # Bayer/hot-pixel
        {"correct_hot_pixels": False},
        {"hot_pixel_threshold": 4.0},
        {"min_weight": 0.5},                            # quality/weight policy
        {"weighting_method": "snr"},
        {"snr_exponent": 1.5},
        {"drizzle_scale_effective": 3.0},               # effective scale/kernel/...
        {"drizzle_kernel_effective": "point"},
        {"drizzle_pixfrac_effective": 0.8},
        {"drizzle_wht_threshold_effective": 0.9},
        {"drizzle_wht_policy": "absolute_v1"},
        {"drizzle_fillval": "nan"},
        {"drizzle_double_norm_fix": True},
        {"background_match_contract": "bgmatch_v2"},    # background-match contract
        {"background_match_contract_version": 2},
        {"output_grid_contract": "grid_v2"},            # output-grid contract
        {"output_grid_contract_version": 2},
        {"registration_contract": "reg_v2"},            # registration contract
        {"registration_contract_version": 2},
    ]
    for overrides in variants:
        assert _drizzle_complete(**overrides).drizzle_fingerprint() != base_fp, overrides


def test_drizzle_fingerprint_ignores_requested_and_policy():
    base = _drizzle_complete()
    base_fp = base.drizzle_fingerprint()
    # requested-only + presentation/resource policy must not change the hash
    requested = _drizzle_complete(
        drizzle_scale_requested=3.0,
        drizzle_kernel_requested="lanczos",
        drizzle_pixfrac_requested=0.6,
        drizzle_wht_threshold_requested=0.5,
        drizzle_mode="Incremental",
        drizzle_group_size=99,
        use_drizzle=False,
    )
    assert requested.drizzle_fingerprint() == base_fp


def test_drizzle_domain_separated_from_classic():
    cfg = _drizzle_complete()
    assert cfg.drizzle_fingerprint() != cfg.classic_fingerprint()
    # classic-only fields do not perturb the drizzle hash, and vice versa
    assert _drizzle_complete(stacking_mode="mean").drizzle_fingerprint() == \
        cfg.drizzle_fingerprint()
    assert _drizzle_complete(drizzle_scale_effective=5.0).classic_fingerprint() == \
        cfg.classic_fingerprint()


# ---------------------------------------------------------------------------
# 11. Backend/engine collection seam (collect_from_backend)
# ---------------------------------------------------------------------------
def _backend_obj(**attrs):
    b = types.SimpleNamespace()
    for k, v in attrs.items():
        setattr(b, k, v)
    return b


def test_collect_from_backend_percent_and_winsor_parity():
    backend = _backend_obj(
        stacking_mode="kappa-sigma",
        kappa=2.5,
        stack_kappa_low=2.5,
        stack_kappa_high=2.5,
        winsor_limits=(0.05, 0.05),
        normalize_method="none",
        weighting_method="none",
        use_quality_weighting=False,
        weight_by_snr=True,
        weight_by_stars=True,
        snr_exponent=1.0,
        stars_exponent=0.5,
        min_weight=0.1,
        correct_hot_pixels=True,
        hot_pixel_threshold=3.0,
        neighborhood_size=5,
        bayer_pattern="GRBG",
        batch_size=10,
        apply_batch_feathering=True,
        apply_feathering=False,
        feather_blur_px=256,
        apply_master_tile_crop=False,
        master_tile_crop_percent_decimal=0.18,
        apply_low_wht_mask=False,
        low_wht_percentile=5,
        low_wht_soften_px=128,
    )
    cfg = rc.collect_from_backend(backend)
    # engine decimal -> canonical percent
    assert cfg.scientific["master_tile_crop_percent"] == 18.0
    # winsor tuple -> canonical list
    assert cfg.scientific["winsor_limits"] == [0.05, 0.05]
    # a None-valued optional classic field is omitted from the payload
    assert "chunk_size" not in cfg.scientific
    assert cfg.product_version == ""


def test_collect_from_backend_absent_fields_omitted():
    backend = _backend_obj(stacking_mode="mean", kappa=2.5)
    cfg = rc.collect_from_backend(backend)
    assert set(cfg.scientific) == {"stacking_mode", "kappa"}


def test_collect_from_backend_ignores_non_classic_fields():
    # A backend that also carries drizzle/execution fields must not leak them
    # into the classic payload (classic domain only).
    backend = _backend_obj(
        stacking_mode="kappa-sigma",
        kappa=2.5,
        drizzle_mode="Final",
        drizzle_scale_effective=2.0,
        output_folder="/tmp/out",
    )
    cfg = rc.collect_from_backend(backend)
    assert set(cfg.scientific) == {"stacking_mode", "kappa"}


def test_collect_from_backend_no_io(tmp_path):
    os.chdir(tmp_path)
    try:
        backend = _backend_obj(stacking_mode="mean", kappa=2.5)
        cfg = rc.collect_from_backend(backend)
        cfg.classic_fingerprint()
        cfg.full_digest()
    finally:
        os.chdir(str(ROOT))
    assert list(tmp_path.iterdir()) == []


def test_collect_from_backend_fails_closed_on_uncoercible():
    """A present but uncoercible classic runtime value refuses collection
    (fail closed), never a silently-omitted field that would diverge from the
    engine's authoritative classic fingerprint."""
    backend = _backend_obj(stacking_mode="kappa-sigma", kappa="not-a-float")
    with pytest.raises(rc.ValidationError):
        rc.collect_from_backend(backend)
