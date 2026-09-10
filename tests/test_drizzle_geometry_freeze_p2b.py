"""P2-B rework-1: one canonical geometry-freeze seam, consumed everywhere.

Covers: idempotent freeze with drift rejection, truthful canonical fields,
no recomputation drift across consumers, legacy/missing-geometry rejection,
and output-grid invariance under the timing refactor.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

dc = importlib.import_module("seestar.core.drizzle_checkpoint")
drizzle_core = importlib.import_module("seestar.core.drizzle_core")
rc = importlib.import_module("seestar.run_contract")
qm_mod = importlib.import_module("seestar.queuep.queue_manager")

N = 32


def _wcs(plate=2.4e-4, shape=(N, N)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = [-plate, plate]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.array_shape = shape
    return w


def _stacker(**attrs):
    obj = object.__new__(qm_mod.SeestarQueuedStacker)
    obj.drizzle_scale = 3.0
    obj.drizzle_kernel = "lanczos2"
    obj.drizzle_pixfrac = 1.0
    obj.drizzle_wht_threshold = 0.0
    obj.drizzle_wht_threshold_effective = 0.0
    obj.weighting_method = "none"
    obj.use_quality_weighting = False
    obj.weight_by_snr = True
    obj.weight_by_stars = True
    obj.snr_exponent = 1.0
    obj.stars_exponent = 0.5
    obj.min_weight = 0.01
    obj.correct_hot_pixels = True
    obj.hot_pixel_threshold = 3.0
    obj.neighborhood_size = 5
    obj.bayer_pattern = "GRBG"
    obj.drizzle_fillval = "0.0"
    obj.drizzle_double_norm_fix = True
    obj.reference_wcs_object = _wcs()
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


# 1 + 2: one idempotent seam, frozen before every consumer, no drift
def test_freeze_is_idempotent_and_drift_raises():
    obj = _stacker()
    first = obj._freeze_drizzle_geometry()
    assert first is not None and first > 0.0
    assert obj.drizzle_pixel_scale_ratio_derived == first
    assert obj.drizzle_pixel_scale_ratio_effective == first
    assert obj.drizzle_pixel_scale_ratio_source == drizzle_core.PIXEL_SCALE_RATIO_SOURCE
    assert obj.drizzle_pixel_scale_ratio_requested is None
    # repeated freeze returns exactly the same fact
    assert obj._freeze_drizzle_geometry() == first
    # a different geometry for the same run is rejected (fail closed)
    obj.drizzle_output_wcs = None
    obj.drizzle_scale = 2.0  # derived ratio would become 1/2 != frozen 1/3
    with pytest.raises(drizzle_core.DrizzleGeometryError):
        obj._freeze_drizzle_geometry()


def test_freeze_without_reference_wcs_is_unresolved():
    obj = _stacker(reference_wcs_object=None)
    assert obj._freeze_drizzle_geometry() is None


# 2 + 5: every canonical config consumer sees ONE stable value
def test_canonical_config_is_stable_and_truthful():
    obj = _stacker()
    cfg1 = dc.build_drizzle_canonical_config(obj, product_version="8.4.0")
    cfg2 = dc.build_drizzle_canonical_config(obj, product_version="8.4.0")
    assert cfg1.full_digest() == cfg2.full_digest()
    sci = cfg1.scientific
    assert sci["pixel_scale_ratio_requested"] is None
    assert sci["pixel_scale_ratio_derived"] == pytest.approx(
        sci["pixel_scale_ratio_effective"]
    )
    assert sci["pixel_scale_ratio_source"] == drizzle_core.PIXEL_SCALE_RATIO_SOURCE
    assert sci["pixel_scale_ratio_effective"] == pytest.approx(1.0 / 3.0, rel=1e-9)


def test_config_built_after_freeze_equals_config_built_by_builder_alone():
    frozen = _stacker()
    frozen._freeze_drizzle_geometry()
    lazy = _stacker()
    assert (
        dc.build_drizzle_canonical_config(frozen, product_version="8.4.0").full_digest()
        == dc.build_drizzle_canonical_config(lazy, product_version="8.4.0").full_digest()
    )


# 5: missing (legacy) geometry state is rejected deterministically
import copy as _copy


def test_legacy_config_without_geometry_is_not_equivalent():
    """Only the four geometry fields differ — the digest must still change."""
    current = dc.build_drizzle_canonical_config(_stacker(), product_version="8.4.0")
    legacy = _copy.deepcopy(current)
    for key in (
        "pixel_scale_ratio_requested",
        "pixel_scale_ratio_derived",
        "pixel_scale_ratio_effective",
        "pixel_scale_ratio_source",
    ):
        legacy.scientific.pop(key, None)
    # baselines are identical: the ONLY semantic difference is geometry
    assert set(current.scientific) - set(legacy.scientific) == {
        "pixel_scale_ratio_requested",
        "pixel_scale_ratio_derived",
        "pixel_scale_ratio_effective",
        "pixel_scale_ratio_source",
    }
    assert current.full_digest() != legacy.full_digest()
    # an explicit geometry VALUE mismatch is rejected too
    other = _copy.deepcopy(current)
    other.scientific["pixel_scale_ratio_effective"] = 0.5
    other.scientific["pixel_scale_ratio_derived"] = 0.5
    assert other.full_digest() != current.full_digest()
    # a SOURCE mismatch is rejected as well
    src = _copy.deepcopy(current)
    src.scientific["pixel_scale_ratio_source"] = "upstream_add_image_default"
    assert src.full_digest() != current.full_digest()


# 7: the timing refactor does not change the canonical grid
def test_freeze_does_not_change_output_grid():
    obj = _stacker()
    ref = obj.reference_wcs_object
    out_before = drizzle_core.build_output_grid(ref, (N, N), 3.0)
    obj._freeze_drizzle_geometry()
    out_after = drizzle_core.build_output_grid(obj.reference_wcs_object, (N, N), 3.0)
    assert out_before[1] == out_after[1]
    assert np.array_equal(out_before[0].wcs.crpix, out_after[0].wcs.crpix)
    assert np.array_equal(out_before[0].wcs.cdelt, out_after[0].wcs.cdelt)
    assert obj.reference_wcs_object is ref


# 4/6: the frozen fact reaches science accumulators and is restored on resume
def test_science_accumulator_receives_and_restores_frozen_ratio():
    obj = _stacker()
    psr = obj._freeze_drizzle_geometry()
    acc = drizzle_core.DrizzleAccumulator((4, 4), kernel="lanczos2", pixel_scale_ratio=psr)
    assert acc.pixel_scale_ratio == psr
    restored = drizzle_core.DrizzleAccumulator.from_native_state(
        (4, 4), np.zeros((4, 4), np.float32), np.ones((4, 4), np.float32),
        kernel="lanczos2", total_exptime=1.0, pixel_scale_ratio=psr,
    )
    assert restored.pixel_scale_ratio == psr
    # support accumulators keep the frozen (upstream-default) semantics
    sup = drizzle_core.DrizzleAccumulator((4, 4), kernel="square", pixfrac=1.0)
    assert sup.pixel_scale_ratio is None
