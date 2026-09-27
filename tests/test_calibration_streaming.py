"""C5 streaming unit tests: the single normalization seam (control-path parity).

These prove the normalization seam replicates the historical ``load_and_validate_fits``
min-max normalization bit-for-bit, operates on the physical domain (no early
normalization), and preserves saturation.  No ZeCalibrator / Qt required.
"""

from __future__ import annotations

import numpy as np

from seestar.core.image_processing import normalize_physical_to_working


def test_normalize_matches_historical_min_max():
    rng = np.random.default_rng(0)
    physical = (rng.random((32, 40)) * 40000 + 200).astype(np.float32)
    physical[0, 0] = 0.0
    physical[0, 1] = 65535.0
    # Replicate the historical loader normalization (float64 stats, float32 math).
    stats = physical.astype(np.float64)
    mn, mx = float(np.nanmin(stats)), float(np.nanmax(stats))
    historical = np.clip(
        (physical.astype(np.float32) - mn) / (mx - mn), 0.0, 1.0
    ).astype(np.float32)
    mine = normalize_physical_to_working(physical)
    assert np.array_equal(historical, mine)
    assert float(np.max(np.abs(historical - mine))) == 0.0


def test_physical_scale_not_prenormalized():
    # Physical input (not 0..1) normalizes to [0,1]; the extremes are exact.
    physical = np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32)
    out = normalize_physical_to_working(physical)
    assert out[0, 0] == 0.0
    assert out[1, 1] == 1.0
    assert out.dtype == np.float32


def test_saturation_survives():
    physical = np.linspace(0, 65535, 100, dtype=np.float32).reshape(10, 10)
    out = normalize_physical_to_working(physical)
    assert out.max() >= 0.999  # saturation survives as the max -> ~1.0


def test_nonfinite_is_deterministic_and_finite():
    # With an Inf present, nanmin/nanmax cannot produce a finite range and the
    # historical "constant" branch applies (all 0.5) — never a crash, never a
    # NaN in the working array.
    physical = np.linspace(0, 100, 100, dtype=np.float32).reshape(10, 10)
    physical[0, 0] = np.nan
    physical[0, 1] = np.inf
    out = normalize_physical_to_working(physical)
    assert np.isfinite(out).all()
    assert out.min() == out.max() == 0.5
