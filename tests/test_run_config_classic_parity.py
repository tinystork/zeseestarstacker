"""Classic v1 fingerprint parity with the real queue-manager engine.

``seestar.run_contract`` is a pure-stdlib contract module; the engine
(``SeestarQueuedStacker._scientific_fingerprint``) is the legacy manifest-v1
authority.  These tests prove byte-for-byte equality between the two for
representative *complete* configurations, plus the migrated-legacy-witness
path, without touching or mutating any manifest.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))

import seestar.run_contract as rc  # noqa: E402
from seestar.queuep.queue_manager import (  # noqa: E402
    SeestarQueuedStacker,
    _RESUME_FINGERPRINT_ATTRS,
)


# Representative *complete* classic configs (canonical names/units).  The
# engine equivalent is derived mechanically (percent -> decimal, list -> tuple).
CONFIG_A = {
    "stacking_mode": "kappa-sigma",
    "kappa": 2.5,
    "stack_kappa_low": 2.5,
    "stack_kappa_high": 2.5,
    "winsor_limits": [0.05, 0.05],
    "normalize_method": "none",
    "weighting_method": "none",
    "use_quality_weighting": False,
    "weight_by_snr": True,
    "weight_by_stars": True,
    "snr_exponent": 1.0,
    "stars_exponent": 0.5,
    "min_weight": 0.1,
    "correct_hot_pixels": True,
    "hot_pixel_threshold": 3.0,
    "neighborhood_size": 5,
    "bayer_pattern": "GRBG",
    "batch_size": 10,
    "chunk_size": None,
    "apply_batch_feathering": True,
    "apply_feathering": False,
    "feather_blur_px": 256,
    "apply_master_tile_crop": False,
    "master_tile_crop_percent": 18.0,
    "apply_low_wht_mask": False,
    "low_wht_percentile": 5,
    "low_wht_soften_px": 128,
}

CONFIG_B = {
    "stacking_mode": "winsorized-sigma-clip",
    "kappa": 3.0,
    "stack_kappa_low": 3.0,
    "stack_kappa_high": 2.0,
    "winsor_limits": [0.10, 0.20],
    "normalize_method": "linear_fit",
    "weighting_method": "snr",
    "use_quality_weighting": True,
    "weight_by_snr": False,
    "weight_by_stars": False,
    "snr_exponent": 1.5,
    "stars_exponent": 0.8,
    "min_weight": 0.01,
    "correct_hot_pixels": False,
    "hot_pixel_threshold": 4.0,
    "neighborhood_size": 7,
    "bayer_pattern": "RGGB",
    "batch_size": 4,
    "chunk_size": 4096,
    "apply_batch_feathering": False,
    "apply_feathering": True,
    "feather_blur_px": 128,
    "apply_master_tile_crop": True,
    "master_tile_crop_percent": 25.0,
    "apply_low_wht_mask": True,
    "low_wht_percentile": 10,
    "low_wht_soften_px": 64,
}


def _to_engine_attrs(sci):
    """Map a canonical classic scientific dict to the exact engine attribute
    names/units used by ``_RESUME_FINGERPRINT_ATTRS``."""
    attrs = {}
    for k, v in sci.items():
        if k == "master_tile_crop_percent":
            attrs["master_tile_crop_percent_decimal"] = float(v) / 100.0
        elif k == "winsor_limits":
            attrs["winsor_limits"] = tuple(v)
        else:
            attrs[k] = v
    return attrs


def _engine_stack(attrs):
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    for k, v in attrs.items():
        setattr(o, k, v)
    return o


def test_classic_domain_key_names_match_engine():
    """Every classic field maps, via its legacy fingerprint key, to exactly the
    engine's ``_RESUME_FINGERPRINT_ATTRS`` (no drift, no parallel list)."""
    mapped = set()
    for fd in rc.fingerprint_field_defs(rc.FingerprintDomain.CLASSIC_SUMW):
        mapped.add(fd.legacy_fingerprint_key or fd.name)
    assert mapped == set(_RESUME_FINGERPRINT_ATTRS)
    # and the canonical names differ from the engine names in exactly the one
    # documented rename (percent -> decimal).
    canonical = rc.classic_fingerprint_names()
    expected = {n for n in _RESUME_FINGERPRINT_ATTRS} - {
        "master_tile_crop_percent_decimal"
    } | {"master_tile_crop_percent"}
    assert canonical == expected


@pytest.mark.parametrize("sci", [CONFIG_A, CONFIG_B])
def test_classic_fingerprint_matches_engine(sci):
    assert set(sci) == rc.classic_fingerprint_names()
    engine_fp = _engine_stack(_to_engine_attrs(sci))._scientific_fingerprint()
    cfg = rc.RunConfig.from_sections(scientific=sci)
    assert rc.classic_fingerprint(cfg) == engine_fp
    # module-level and method-level APIs agree
    assert cfg.classic_fingerprint() == engine_fp
    assert rc.scientific_fingerprint(cfg, domain=rc.FingerprintDomain.CLASSIC_SUMW) == engine_fp


def test_classic_fingerprint_includes_none_for_absent_fields():
    """The v1 payload keeps every attribute, ``None`` where unavailable, exactly
    like ``getattr(self, attr, None)`` in the engine."""
    partial = {"stacking_mode": "mean", "kappa": 2.5}
    cfg = rc.RunConfig.from_sections(scientific=partial)
    attrs = {k: None for k in _RESUME_FINGERPRINT_ATTRS}
    attrs.update(_to_engine_attrs(partial))
    engine_fp = _engine_stack(attrs)._scientific_fingerprint()
    assert rc.classic_fingerprint(cfg) == engine_fp


def test_migrated_legacy_witness_computes_v1_hash():
    """A migrated legacy 5.x witness, completed into a classic config, computes
    the exact v1-compatible hash.  No external manifest is claimed."""
    data = rc.parse_legacy_cfg(str(FIXTURE_DIR / "legacy_stack_5.6.0_sanitized.cfg"))
    res = rc.migrate_legacy(data)
    assert res.ok and res.resumable is False
    cfg = res.config

    # Rebuild the engine-attribute view from the migrated canonical config so
    # the two independent implementations can be compared for the same values.
    attrs = {}
    for fd in rc.fingerprint_field_defs(rc.FingerprintDomain.CLASSIC_SUMW):
        key = fd.legacy_fingerprint_key or fd.name
        value = cfg.get(fd.section, fd.name)
        if value is None:
            attrs[key] = None
        elif fd.name == "master_tile_crop_percent":
            attrs[key] = float(value) / 100.0
        elif fd.name == "winsor_limits":
            attrs[key] = tuple(value)
        else:
            attrs[key] = value

    engine_fp = _engine_stack(attrs)._scientific_fingerprint()
    assert rc.classic_fingerprint(cfg) == engine_fp
    # A hash is a digest, never a reconstruction: it is 64 hex chars.
    assert len(rc.classic_fingerprint(cfg)) == 64


def test_collect_from_backend_roundtrip_matches_engine_fingerprint():
    """The backend collection seam reconstructs the canonical classic payload
    from the *engine's* own attribute names/units (``_RESUME_FINGERPRINT_ATTRS``,
    decimal crop / tuple winsor) and the recomputed classic fingerprint stays
    byte-for-byte equal to ``SeestarQueuedStacker._scientific_fingerprint``."""
    for sci in (CONFIG_A, CONFIG_B):
        # Engine-attribute view (exactly what the queue manager holds after
        # ``start_processing`` binds the backend kwargs).
        engine = _engine_stack(_to_engine_attrs(sci))
        cfg = rc.collect_from_backend(engine)
        # The collection reverses the two documented representations.
        assert cfg.scientific["master_tile_crop_percent"] == sci["master_tile_crop_percent"]
        assert cfg.scientific["winsor_limits"] == list(sci["winsor_limits"])
        assert rc.classic_fingerprint(cfg) == engine._scientific_fingerprint()
