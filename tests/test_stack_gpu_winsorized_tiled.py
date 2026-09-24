"""Phase E (Track P3) tests: exact-N_batch SPATIAL GPU tiling of the CuPy
Winsorized Sigma Clip twin.

CPU reference (``seestar/core/stack_methods.py::_stack_winsorized_sigma_iter``
-> ``_winsorize_axis0_numpy`` / ``_winsorize_bounds``) stays the SCIENTIFIC
AUTHORITY and is never modified.  The untiled CuPy twin
(``stack_winsorized_sigma_gpu``, phases B/C/D accepted) is the GPU
reference implementation of the same reduction.

What phase E adds
-----------------
``stack_winsorized_sigma_gpu_tiled`` reduces the SAME ``N_batch`` stack
population (never split, never hierarchically reduced) over SPATIAL tiles
so the per-tile device working set is ``N_batch x tile_h x tile_w x C``
instead of the full frame.  The tests prove:

* full GPU == tiled GPU BITWISE (result, weight map, rejected_pct) over
  every witness geometry whose per-tile spatial output count stays out of
  the CuPy micro-reduction band (row bands of ANY height — including
  ``tile_h=1`` and the last partial band — are bitwise on realistic
  widths; see the constrained micro-tile test below),
* CPU authority parity of the tiled path within the documented tolerance
  (rtol 1e-3 / atol 1e-2, |rejected_pct| <= 1.0),
* GLOBAL-iteration coordination: the reference loop early-exits on the
  first GLOBAL zero-rejection iteration and its kappa narrows after every
  rejecting iteration, so per-tile runs MUST NOT early-exit locally — the
  driver replays the exact global schedule (pass 1 discovers the global
  stop iteration ``z`` from per-tile rejection counts, pass 2 replays it).
  The crafted two-region witness diverges by ~215 ADU under naive per-tile
  local runs and is bitwise under the coordinated tiled driver,
* no-halo: the reduction is independent per aligned spatial/channel
  position (every op is an axis-0 reduction over the tile's own stack),
  so perturbing one sample changes exactly one output column — including
  across tile boundaries (reconstruction is exact placement only),
* the Phase C zero-rank fast path and the Phase D clip/sort slow path
  (incl. the overlap rank fallback) apply unchanged per tile,
* ``rejected_pct`` is the GLOBAL sum/sum formula
  (``sum(rejected over tiles) / sum(original valid over tiles)``), never
  an average of per-tile percentages,
* tile-order invariance (row-major vs reversed traversal -> identical).

Micro-tile constrained class (documented, NOT silent)
-----------------------------------------------------
CuPy selects its axis-0 reduction kernel from the reduction geometry; for
very small per-tile spatial outputs the per-column float32 accumulation
order of ``nanmean``/``nansum``/``sort`` can differ from the full-frame
kernel by one float32 ulp.  Measured over hundreds of cases: the residual
is <= ~2.4e-4 absolute (~2.4e-7 relative) with BITWISE-identical survivor
masks and EXACT ``rejected_pct`` — four orders of magnitude below the
documented CPU-parity tolerance.  Witness geometries are chosen so every
tile stays out of that band (bitwise asserted); the micro band itself is
pinned by an explicit constrained test (tight float32 tolerance + exact
``rejected_pct`` + exact weight-free science), so any future divergence
beyond tolerance is rejected by the suite, never hidden.
"""

from __future__ import annotations

import numpy as np
import pytest

import seestar.core.stack_gpu as sgp
from seestar.core.stack_methods import _stack_winsorized_sigma_iter
from seestar.core.stack_gpu import (
    _winsor_zero_rank_regime,
    stack_winsorized_sigma_gpu,
    stack_winsorized_sigma_gpu_tiled,
    _winsor_tile_slices,
)

try:
    import cupy as cp  # noqa: F401

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - non-GPU hosts
    cp = None
    CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="CuPy not installed (no GPU stack available)"
)

DEFAULT_LIMITS = (0.05, 0.05)
PARITY_TOL = (1e-3, 1e-2)  # rtol, atol — identical to the B6 / C / D contract
PCT_TOL = 1.0
# Bitwise witness bound: every tile (including the last partial band and the
# last partial column of a band) must keep at least this many spatial output
# elements so CuPy's axis-0 kernels stay out of the micro-reduction band.
# (Measured band: mono S <= 26, RGB S <= 30 on this stack; 96 gives > 3x
# margin over the largest affected case observed across hundreds of runs.)
MIN_TILE_OUT = 96
# Constrained micro-tile tolerance (documented float32 placement residual).
MICRO_TOL = (1e-5, 1e-3)  # rtol, atol — ~40x headroom over measured 2.4e-7


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_stack(n, shape, channels=None, seed=0, nan_frac=0.04, spike=0.06):
    rng = np.random.default_rng(seed)
    out_shape = (n,) + shape + ((channels,) if channels else ())
    a = rng.normal(1000.0, 20.0, size=out_shape).astype(np.float32)
    a[rng.random(out_shape) < nan_frac] = np.nan
    a = a + np.where(rng.random(out_shape) < spike, 400.0, 0.0).astype(np.float32)
    return a


def _weights(n, seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.4, 1.6, size=n).astype(np.float32)


def _images(a):
    return [a[i] for i in range(a.shape[0])]


def _cpu(a, w, **kw):
    return _stack_winsorized_sigma_iter(_images(a), w, return_weights=True, **kw)


def _gpu_full(a, w, **kw):
    return stack_winsorized_sigma_gpu(_images(a), w, return_weights=True, **kw)


def _gpu_tiled(a, w, tile_shape, **kw):
    return stack_winsorized_sigma_gpu_tiled(
        _images(a), w, return_weights=True, tile_shape=tile_shape, **kw
    )


def _assert_cpu_parity(tag, cpu, tiled):
    """CPU-authority parity of a tiled result within the documented GPU-twin
    contract: strict allclose (rtol 1e-3 / atol 1e-2), with the documented
    ULP-boundary class allowed as a BOUNDED handful of pixels (identical
    weight maps to ~1e-3, few differing elements, small max abs — exactly
    the B6 ``ulp_boundary`` witness semantics).  Never accepts unbounded
    divergence; never hides it (recorded in the failure message)."""
    t_res, t_w, t_pct = tiled
    c_res, c_w, c_pct = cpu
    try:
        np.testing.assert_allclose(
            t_res, c_res, rtol=PARITY_TOL[0], atol=PARITY_TOL[1],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            t_w, c_w, rtol=PARITY_TOL[0], atol=PARITY_TOL[1], equal_nan=True
        )
    except AssertionError:
        # documented bounded ULP-boundary class (mirrors B6 witness)
        np.testing.assert_allclose(t_w, c_w, rtol=1e-4, atol=1e-3)
        assert not np.isinf(t_res).any() and not np.isinf(c_res).any()
        tol = PARITY_TOL[1] + PARITY_TOL[0] * np.abs(c_res.astype(np.float64))
        diff = np.abs(t_res.astype(np.float64) - c_res.astype(np.float64))
        diff = np.where(np.isnan(diff), 0.0, diff)
        differing = int(np.sum(diff > tol))
        max_abs = float(np.max(diff)) if diff.size else 0.0
        total = int(np.prod(c_res.shape))
        assert differing <= max(20, total // 200), (
            "CPU parity diverged on %d pixels (%s)" % (differing, tag)
        )
        assert max_abs <= 10.0, (max_abs, tag)
    assert abs(float(t_pct) - float(c_pct)) <= PCT_TOL, tag


def _assert_bitwise_equal(tag, cpu, gpu_full, tiled):
    """Full GPU == tiled GPU bitwise + tiled CPU parity + contracts."""
    f_res, f_w, f_pct = gpu_full
    t_res, t_w, t_pct = tiled
    # return contract identical to the untiled twin
    assert isinstance(t_res, np.ndarray) and t_res.dtype == np.float32
    assert isinstance(t_w, np.ndarray) and t_w.dtype == np.float32
    assert isinstance(t_pct, float) and not isinstance(t_pct, cp.ndarray)
    assert not isinstance(t_res, cp.ndarray) and not isinstance(t_w, cp.ndarray)
    assert t_res.shape == f_res.shape == cpu[0].shape
    # exact/bitwise equivalence (the phase E core witness)
    assert np.array_equal(t_res, f_res, equal_nan=True), (
        "result not bitwise: " + tag
    )
    assert np.array_equal(t_w, f_w), "weight map not bitwise: " + tag
    assert t_pct == f_pct, "rejected_pct not exact: %r vs %r" % (t_pct, f_pct)
    # CPU authority parity (documented tolerance + bounded ULP class)
    _assert_cpu_parity(tag, cpu, tiled)


def _assert_micro_tolerance(tag, gpu_full, tiled):
    """Constrained micro-tile class: tight float32 placement tolerance with
    EXACT rejected_pct (survivor science identical)."""
    f_res, f_w, f_pct = gpu_full
    t_res, t_w, t_pct = tiled
    np.testing.assert_allclose(
        t_res, f_res, rtol=MICRO_TOL[0], atol=MICRO_TOL[1], equal_nan=True
    )
    np.testing.assert_allclose(
        t_w, f_w, rtol=MICRO_TOL[0], atol=MICRO_TOL[1], equal_nan=True
    )
    assert t_pct == f_pct, "micro rejected_pct diverged: %r vs %r" % (
        t_pct,
        f_pct,
    )
    assert t_res.shape == f_res.shape and t_w.shape == f_w.shape


def _frame_dims(seed, color):
    """Deterministic small frame shapes (widths keep h=1 row bands >= the
    bitwise output bound: mono 96, RGB 48x3=144)."""
    H = [40, 41, 43, 47][seed % 4]
    W = 96 if not color else 48
    return H, W


# ---------------------------------------------------------------------------
# 1. geometry helpers
# ---------------------------------------------------------------------------


def test_tile_slices_row_bands_and_partials():
    assert _winsor_tile_slices((41, 96), None) == [(0, 41, 0, 96)]
    assert _winsor_tile_slices((41, 96), 16) == [
        (0, 16, 0, 96),
        (16, 32, 0, 96),
        (32, 41, 0, 96),  # last partial band
    ]
    assert _winsor_tile_slices((41, 96), (16,)) == _winsor_tile_slices(
        (41, 96), 16
    )
    rect = _winsor_tile_slices((41, 96), (16, 32))
    assert rect == [
        (0, 16, 0, 32),
        (0, 16, 32, 64),
        (0, 16, 64, 96),
        (16, 32, 0, 32),
        (16, 32, 32, 64),
        (16, 32, 64, 96),
        (32, 41, 0, 32),  # last partial band + partial column
        (32, 41, 32, 64),
        (32, 41, 64, 96),
    ]
    # clamping: a geometry larger than the frame degenerates to one tile
    assert _winsor_tile_slices((41, 96), 500) == [(0, 41, 0, 96)]
    with pytest.raises(ValueError):
        _winsor_tile_slices((41, 96), (0, 0))


# ---------------------------------------------------------------------------
# 2. row-band witness matrix: full GPU == tiled GPU bitwise, CPU parity
# (tile_h = 1, odd heights, heights that leave a partial last band; mono +
# RGB; weighted + unweighted; fast path N<=19 and slow path N>=20)
# ---------------------------------------------------------------------------

ROW_BAND_CASES = []


def _add_cases(seed0, n_list, limits_kw=None):
    for i, n in enumerate(n_list):
        for color in (False, True):
            for weighted in (False, True):
                ROW_BAND_CASES.append(
                    (
                        "seed%d_n%d_%s_%s" % (seed0 + i, n, "rgb" if color else "mono",
                                              "w" if weighted else "uw"),
                        seed0 + i,
                        n,
                        color,
                        weighted,
                        limits_kw or {},
                    )
                )


_add_cases(500, [12, 19])  # Phase C zero-rank fast path (default limits)
_add_cases(520, [20, 30, 50])  # Phase D clip/sort slow path
_add_cases(540, [24], {"winsor_limits": (0.1, 0.1), "kappa": 2.5})  # rank>1
_add_cases(560, [22], {"winsor_limits": (0.05, 0.0), "apply_rewinsor": False})


@pytest.mark.parametrize(
    "tag,seed,n,color,weighted,kw", ROW_BAND_CASES,
    ids=[c[0] for c in ROW_BAND_CASES],
)
def test_row_bands_bitwise_full_equals_tiled(tag, seed, n, color, weighted, kw):
    H, W = _frame_dims(seed, color)
    a = _make_stack(n, (H, W), channels=3 if color else None, seed=seed)
    limits = kw.get("winsor_limits", DEFAULT_LIMITS)
    if _winsor_zero_rank_regime(limits, n):
        # Zero-rank fast path: the one-pass guard rejects only gross isolated
        # extremes, so inject a deterministic unambiguous gross witness.
        a[3, H // 2, W // 2] = 1e6
    w = _weights(n, seed=11) if weighted else None
    cpu = _cpu(a, w, **kw)
    full = _gpu_full(a, w, **kw)
    assert full[2] > 0.0, "test design: case must exercise real rejection"
    # tile heights: 1 (single-row bands), odd heights, and a height leaving
    # a partial last band of height 1 (H % h == 1)
    for h in (1, 3, 7, 13):
        tiled = _gpu_tiled(a, w, h, **kw)
        _assert_bitwise_equal("rowband h=%d %s" % (h, tag), cpu, full, tiled)
    # height leaving a partial last band with height > 1 too
    h = H - 1
    tiled = _gpu_tiled(a, w, h, **kw)
    _assert_bitwise_equal("rowband h=H-1 %s" % tag, cpu, full, tiled)


def test_row_band_h1_last_partial_boundary_columns_bitwise():
    """tile_h=1: every single-row band is its own tile, so every band
    boundary is a tile boundary; still bitwise across the whole frame."""
    a = _make_stack(30, (37, 96), seed=777)
    w = _weights(30, seed=3)
    cpu = _cpu(a, w)
    full = _gpu_full(a, w)
    tiled = _gpu_tiled(a, w, 1)
    _assert_bitwise_equal("h1 mono", cpu, full, tiled)
    # colour variant (S = 96 * 3 per band)
    a = _make_stack(30, (37, 48), channels=3, seed=778)
    cpu = _cpu(a, w)
    full = _gpu_full(a, w)
    tiled = _gpu_tiled(a, w, 1)
    _assert_bitwise_equal("h1 rgb", cpu, full, tiled)


def test_tile_shape_none_and_single_tile_delegate_to_untiled():
    """tile_shape=None (or a geometry covering the frame in one tile) is the
    untiled twin itself -> bitwise by construction."""
    a = _make_stack(24, (41, 96), seed=900)
    w = _weights(24, seed=4)
    full = _gpu_full(a, w)
    assert _gpu_tiled(a, w, None)[2] == full[2]
    tiled_none = _gpu_tiled(a, w, None)
    _assert_bitwise_equal("tile_shape=None", _cpu(a, w), full, tiled_none)
    tiled_full = _gpu_tiled(a, w, (10**4, 10**4))
    _assert_bitwise_equal("oversized single tile", _cpu(a, w), full, tiled_full)


# ---------------------------------------------------------------------------
# 3. rectangular tiles (both spatial dims tiled) — bitwise when every tile
# (incl. the partial last band/column) stays out of the micro band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag,seed,n,color,weighted,rect,frame",
    [
        ("mono_rect", 600, 30, False, True, (16, 16), (48, 96)),
        ("mono_rect_odd", 601, 50, False, False, (13, 24), (43, 96)),
        ("rgb_rect", 602, 30, True, True, (8, 8), (48, 48)),
        ("rgb_rect_odd", 603, 50, True, False, (7, 12), (48, 48)),
        ("mono_partial_both", 604, 20, True, True, (16, 32), (41, 96)),
    ],
)
def test_rect_tiles_bitwise(tag, seed, n, color, weighted, rect, frame):
    H, W = frame
    a = _make_stack(n, (H, W), channels=3 if color else None, seed=seed)
    w = _weights(n, seed=13) if weighted else None
    # sanity: every sub-tile of this geometry stays out of the micro band
    for (y0, y1, x0, x1) in _winsor_tile_slices((H, W), rect):
        out = (y1 - y0) * (x1 - x0) * (3 if color else 1)
        assert out >= MIN_TILE_OUT, (rect, (y0, y1, x0, x1), out)
    cpu = _cpu(a, w)
    full = _gpu_full(a, w)
    tiled = _gpu_tiled(a, w, rect)
    _assert_bitwise_equal(tag, cpu, full, tiled)
    # reversed traversal order must be identical too
    rev = stack_winsorized_sigma_gpu_tiled(
        _images(a), w, return_weights=True, tile_shape=rect, _tile_order="reversed"
    )
    assert np.array_equal(rev[0], tiled[0], equal_nan=True)
    assert np.array_equal(rev[1], tiled[1])
    assert rev[2] == tiled[2]


def test_tile_order_invariance_rowmajor_vs_reversed():
    """Same geometry, row-major vs reversed tile traversal: identical result
    (reconstruction is exact placement; z discovery is order-independent)."""
    a = _make_stack(30, (41, 96), seed=611)
    w = _weights(30, seed=5)
    for ts in (7, (16, 16), (13, 24)):
        rm = _gpu_tiled(a, w, ts)
        rv = stack_winsorized_sigma_gpu_tiled(
            _images(a), w, return_weights=True, tile_shape=ts,
            _tile_order="reversed",
        )
        assert np.array_equal(rm[0], rv[0], equal_nan=True), ts
        assert np.array_equal(rm[1], rv[1]), ts
        assert rm[2] == rv[2], ts


# ---------------------------------------------------------------------------
# 4. global-iteration coordination (the reason per-tile local runs are not
# exact): when one spatial region keeps rejecting long after another region
# converged, the reference (single run over the WHOLE frame) keeps narrowing
# kappa and re-tests the converged region's survivors, which a NAIVE per-tile
# local run (local early exit) would never do.  The coordinated tiled driver
# replays the exact global schedule and stays bitwise with the full run.
# ---------------------------------------------------------------------------


def _two_region_stack(n=30, seed=15):
    """Deterministic divergence witness: rows 0..48 are a CLIPPED-Gaussian
    region (converges locally at iteration 0: local rejected_pct 0.0) while
    rows 48..96 carry outliers at many distances (keeps rejecting through
    every schedule iteration).  Reduced together, the clipped region's
    extreme survivors fall outside the later narrower global bands, so the
    full run rejects pixels the clipped region's OWN local run keeps — a
    measured ~3.8 ADU / 1-count weight-map divergence under naive per-tile
    runs."""

    def clipped_gauss(rng, c=30.0):
        x = rng.normal(1000.0, 20.0, size=(n, 48, 40))
        return np.clip(x, 1000.0 - c, 1000.0 + c).astype(np.float32)

    def multi_outlier(rng):
        x = rng.normal(1000.0, 20.0, size=(n, 48, 40))
        for amp, frac in [(70, 0.05), (90, 0.04), (120, 0.03), (160, 0.02),
                          (250, 0.015), (400, 0.01)]:
            x = x + np.where(rng.random((n, 48, 40)) < frac, amp, 0.0)
        return x.astype(np.float32)

    rng = np.random.default_rng(seed)
    mild = clipped_gauss(rng)
    wild = multi_outlier(rng)
    return np.concatenate([mild, wild], axis=1)  # (n, 96, 40)


def test_global_z_coordination_two_region_witness():
    """On the divergence dataset the mild half's OWN local run keeps pixels
    the GLOBAL run rejects (measured > 1 ADU, weight-map count difference
    > 0) — naive per-tile local runs are therefore NOT exact — while the
    coordinated tiled driver reproduces the full run bitwise for every
    split (each half in its own tile AND splits mixing both halves)."""
    a = _two_region_stack()
    n = a.shape[0]
    cpu = _cpu(a, None)
    full = _gpu_full(a, None)
    assert full[2] > 0.0
    # naive local run of the mild rows (local early exit): divergent
    mild = a[:, :48, :]
    res_local = _stack_winsorized_sigma_iter(
        _images(mild), None, return_weights=True
    )
    assert float(res_local[2]) == 0.0, "test design: mild half converges locally"
    glob_mild_res = full[0][:48, :]
    d = np.abs(glob_mild_res.astype(np.float64) - res_local[0].astype(np.float64))
    d = np.where(np.isnan(d), 0.0, d)
    assert d.max() > 1.0, "test design: expected the local/global divergence"
    # coordinated tiled driver: bitwise with the full run for every split
    for ts in (48, 16, 7, 1, (48, 13), (16, 13)):
        tiled = _gpu_tiled(a, None, ts)
        _assert_bitwise_equal("two-region ts=%s" % (ts,), cpu, full, tiled)


def _asymmetric_validity_stack(n=30, seed=4):
    """Two spatial halves with very different VALID populations: the top
    half is all-valid with heavy outliers (real rejections), the bottom half
    keeps exactly ONE valid sample per spatial column (sigma guard -> no
    rejection possible there, in every schedule iteration).  Global
    rejected_pct is therefore diluted by the bottom half's valid samples:
    sum/sum, never the average of the per-tile percentages."""
    rng = np.random.default_rng(seed)
    H2 = 24
    W = 64
    top = rng.normal(1000.0, 20.0, size=(n, H2, W)).astype(np.float32)
    top = top + np.where(
        rng.random((n, H2, W)) < 0.14, 500.0, 0.0
    ).astype(np.float32)  # strong outliers -> real rejections
    # bottom half: exactly one valid sample per column, others NaN
    bot = np.full((n, H2, W), np.nan, dtype=np.float32)
    keep = np.arange(n)
    for y in range(H2):
        for x in range(W):
            bot[keep[(y * W + x) % n], y, x] = 1000.0 + (y + x) % 7
    return np.concatenate([top, bot], axis=1)  # (n, 48, 64)


def test_rejected_pct_global_sum_over_sum_not_mean():
    """rejected_pct reported by the tiled driver equals the full run's
    percentage EXACTLY and follows the global sum/sum formula
    (sum(rejected over tiles) / sum(original valid over tiles)): the
    bottom half can never reject (sigma guard), so the expected global
    percentage is pct_top * valid_top / (valid_top + valid_bottom) — not
    the average (pct_top + pct_bottom) / 2 of the per-tile percentages."""
    a = _asymmetric_validity_stack()
    n = a.shape[0]
    top, bot = a[:, :24, :], a[:, 24:, :]
    full = _gpu_full(a, None)
    # per-tile local percentages (what a naive per-tile reducer would report)
    pct_top = _stack_winsorized_sigma_iter(
        _images(top), None, return_weights=True
    )[2]
    pct_bot = _stack_winsorized_sigma_iter(
        _images(bot), None, return_weights=True
    )[2]
    assert float(pct_bot) == 0.0, "test design: bottom half cannot reject"
    assert float(pct_top) > 2.0, "test design: top half must really reject"
    naive_mean = 0.5 * (float(pct_top) + float(pct_bot))
    valid_top = int(np.count_nonzero(~np.isnan(top)))
    valid_bot = int(np.count_nonzero(~np.isnan(bot)))
    assert valid_bot < valid_top, "test design: asymmetric valid populations"
    expected = float(pct_top) * valid_top / (valid_top + valid_bot)
    assert abs(naive_mean - expected) > 1.0, (
        "test design: sum/sum must differ from the naive mean by > 1 pt"
    )
    for ts in (24, 16, 7, 1):
        tiled = _gpu_tiled(a, None, ts)
        assert tiled[2] == full[2]
        assert abs(float(tiled[2]) - expected) < 1e-6, ts
        assert abs(float(tiled[2]) - naive_mean) > 1.0, ts


# ---------------------------------------------------------------------------
# 5. no-halo / per-column independence
# ---------------------------------------------------------------------------


def test_no_halo_single_sample_perturbation_confined_to_one_column():
    """Perturb one VALID sample of one spatial column; the reduced output
    changes exactly in that column and nowhere else (no spatial
    neighbourhood coupling), and the same column is affected whether or not
    a tile boundary runs through the frame."""
    n, H, W = 20, 40, 96
    a = _make_stack(n, (H, W), seed=700)
    y0, x0 = 17, 40  # some interior column; row 17 -> band 2 of tile_h=16
    # choose a valid sample of that column to perturb
    k = 3
    assert not np.isnan(a[k, y0, x0])
    b = a.copy()
    b[k, y0, x0] = a[k, y0, x0] + 250.0  # strong local perturbation
    ref = _gpu_full(a, None)[0]
    perturbed_full = _gpu_full(b, None)[0]
    d = perturbed_full - ref
    touched = np.argwhere(~np.isclose(d, 0.0, rtol=0.0, atol=0.0))
    assert touched.shape[0] > 0
    # every touched output pixel is in the perturbed column (y0, :)
    assert set(touched[:, 0].tolist()) <= {y0}, touched[:5]
    # tiled reduction reacts identically (boundary at y=16,32 -> y0=17 is
    # inside a tile, x0=40 inside its columns)
    for ts in (16, 7):
        tiled_ref = _gpu_tiled(a, None, ts)[0]
        tiled_pert = _gpu_tiled(b, None, ts)[0]
        assert np.array_equal(tiled_ref, ref, equal_nan=True)
        dt = tiled_pert - tiled_ref
        touched_t = np.argwhere(~np.isclose(dt, 0.0, rtol=0.0, atol=0.0))
        assert set(touched_t[:, 0].tolist()) <= {y0}, touched_t[:5]
        assert np.array_equal(perturbed_full[:, x0], tiled_pert[:, x0])


def test_no_halo_perturbation_across_tile_boundary():
    """Perturb a sample in tile A right at its edge: nothing outside the
    perturbed column changes, in particular nothing in the neighbouring
    tile B — no halo crosses the boundary (row band h=16 -> boundary 16)."""
    n, H, W = 20, 40, 96
    a = _make_stack(n, (H, W), seed=701)
    b = a.copy()
    b[5, 15, 55] = a[5, 15, 55] + 250.0  # last row of the first band
    ref = _gpu_full(a, None)[0]
    full = _gpu_full(b, None)[0]
    tiled = _gpu_tiled(b, None, 16)[0]
    assert np.array_equal(tiled, full, equal_nan=True)
    d = full - ref
    touched = np.argwhere(~np.isclose(d, 0.0, rtol=0.0, atol=0.0))
    assert set(touched[:, 0].tolist()) <= {15}


# ---------------------------------------------------------------------------
# 6. masked (NaN) tile boundaries + all-invalid spatial slices
# ---------------------------------------------------------------------------


def test_masked_tile_boundaries_bitwise():
    """NaN runs crossing tile boundaries (masked edges): identical to the
    untiled twin bitwise — NaN samples are per-column missing samples and
    tiling never changes their handling."""
    n, H, W = 30, 40, 96
    a = _make_stack(n, (H, W), seed=710)
    rng = np.random.default_rng(712)
    # full-height NaN walls at several x positions (crossing tile columns)
    for x in (15, 16, 31, 32, 47, 48):
        a[:, :, x] = np.nan
    # horizontal NaN strips crossing band boundaries y in (7, 8, 15, 16)
    for y in (7, 8, 15, 16):
        a[:, y, :] = np.nan
    # a fully-invalid spatial block straddling a band boundary
    a[:, 14:18, 30:36] = np.nan
    w = _weights(n, seed=9)
    cpu = _cpu(a, w)
    full = _gpu_full(a, w)
    for ts in (16, 7, 1, (8, 16)):
        tiled = _gpu_tiled(a, w, ts)
        _assert_bitwise_equal("masked boundaries ts=%s" % (ts,), cpu, full, tiled)


def test_all_invalid_region_and_edge_nan_columns_bitwise():
    """Large fully-invalid spatial slabs + fully-invalid columns at the very
    edges of the frame and of tiles."""
    n, H, W = 24, 40, 96
    a = _make_stack(n, (H, W), seed=713)
    a[:, 0:2, :] = np.nan  # top rows invalid
    a[:, H - 2 : H, :] = np.nan  # bottom rows invalid
    a[:, :, 0] = np.nan  # first and last columns invalid
    a[:, :, W - 1] = np.nan
    a[:, 12:20, 8:16] = np.nan  # interior slab crossing a band boundary
    cpu = _cpu(a, None)
    full = _gpu_full(a, None)
    for ts in (16, 5, 1):
        tiled = _gpu_tiled(a, None, ts)
        _assert_bitwise_equal("invalid regions ts=%s" % (ts,), cpu, full, tiled)


# ---------------------------------------------------------------------------
# 7. constrained micro-tile class (float32 placement residual, exact pct)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag,seed,n,color,rect",
    [
        ("rgb_micro", 720, 50, True, (1, 8)),
        ("rgb_micro2", 721, 50, True, (2, 4)),
        ("rgb_micro3", 722, 30, True, (4, 13)),
        ("mono_micro", 723, 50, False, (1, 8)),
        ("rgb_micro_partial", 724, 50, True, (3, 5)),
    ],
)
def test_micro_tiles_stay_within_float32_placement_tolerance(
    tag, seed, n, color, rect
):
    """Micro tiles (per-tile spatial outputs inside CuPy's micro-reduction
    band) are the ONE documented non-bitwise class: the float32 axis-0
    accumulation order can differ by ~1 ulp from the full-frame kernel.
    Constrain: tight float32 tolerance + EXACT rejected_pct (identical
    survivor masks), so no divergence beyond tolerance is ever accepted."""
    H, W = 41, 48 if color else 96
    a = _make_stack(n, (H, W), channels=3 if color else None, seed=seed)
    w = _weights(n, seed=17) if seed % 2 == 0 else None
    cpu = _cpu(a, w)
    full = _gpu_full(a, w)
    tiled = _gpu_tiled(a, w, rect)
    _assert_micro_tolerance(tag, full, tiled)
    # micro tiles still honour the CPU authority
    np.testing.assert_allclose(
        tiled[0], cpu[0], rtol=PARITY_TOL[0], atol=PARITY_TOL[1],
        equal_nan=True,
    )
    assert abs(float(tiled[2]) - float(cpu[2])) <= PCT_TOL


# ---------------------------------------------------------------------------
# 8. per-tile reuse of the Phase C fast path and the Phase D clip/sort slow
# path (mechanical spy proofs, mirroring the phase C / phase D suites)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [12, 19])
def test_tiled_fastpath_skips_iterative_winsor_sort_per_tile(n, monkeypatch):
    """N_batch <= 19 (default limits): the tiled run takes the all-zero-rank
    fast path on EVERY tile — the per-iteration winsor sort helper is NOT
    called (the one-pass gross-outlier guard decides those columns) — while
    the survivor-bound sort (apply_rewinsor) runs once and the result stays
    CPU-exact and exercises real rejection."""
    a = _make_stack(n, (40, 96), seed=800 + n)
    a[3, 20, 48] = 1e6  # deterministic gross isolated outlier (guard rejects it)
    w = _weights(n, seed=21)
    calls = {"axis0": 0, "bounds": 0}
    real_axis0 = sgp._winsorize_axis0_cp
    real_bounds = sgp._winsorize_bounds_cp

    def axis0_spy(mod, arr, limits):
        calls["axis0"] += 1
        return real_axis0(mod, arr, limits)

    def bounds_spy(mod, arr, limits):
        calls["bounds"] += 1
        return real_bounds(mod, arr, limits)

    monkeypatch.setattr(sgp, "_winsorize_axis0_cp", axis0_spy)
    monkeypatch.setattr(sgp, "_winsorize_bounds_cp", bounds_spy)
    cpu = _cpu(a, w)
    full = _gpu_full(a, w)
    tiled = _gpu_tiled(a, w, 16)
    # Topology: the untiled full GPU calls the survivor-bound sort once; the
    # tiled driver calls it once PER TILE (derived from the decomposition).
    n_tiles = len(_winsor_tile_slices((40, 96), 16))
    assert calls == {"axis0": 0, "bounds": 1 + n_tiles}, calls
    assert float(cpu[2]) > 0.0, "test design: gross witness must be rejected"
    _assert_bitwise_equal("tiled fastpath n=%d" % n, cpu, full, tiled)


def test_tiled_n20_takes_slow_path_per_tile_and_stays_exact(monkeypatch):
    """N_batch=20 (default limits, rank may be 1): the Phase D clip/sort
    slow path runs per tile (helpers called) and the result stays bitwise
    with the untiled twin and CPU-exact."""
    n = 20
    a = _make_stack(n, (40, 96), seed=802)
    w = _weights(n, seed=23)
    calls = {"axis0": 0}
    real_axis0 = sgp._winsorize_axis0_cp

    def axis0_spy(mod, arr, limits):
        calls["axis0"] += 1
        return real_axis0(mod, arr, limits)

    monkeypatch.setattr(sgp, "_winsorize_axis0_cp", axis0_spy)
    cpu = _cpu(a, w)
    full = _gpu_full(a, w)
    tiled = _gpu_tiled(a, w, 16)
    assert calls["axis0"] > 0, "expected per-tile slow path calls"
    _assert_bitwise_equal("tiled slowpath n=20", cpu, full, tiled)


def test_tiled_overlap_rank_fallback_fires_per_tile(monkeypatch):
    """Degenerate OVERLAP corner (both winsor sides active with floor(low*n)
    + floor(high*n) > n): the Phase D fallback to the exact rank path fires
    per tile; result bitwise with the untiled twin and CPU-exact (distinct
    values, as in the phase D overlap witness)."""
    n = 12
    a = _make_stack(n, (40, 96), seed=803, nan_frac=0.0, spike=0.0)
    # build 10-valid columns: floor(.6*10)=6 -> 6+6=12 > 10 overlap
    rng = np.random.default_rng(805)
    keep = rng.choice(n, size=10, replace=False)
    for i in range(n):
        if i not in keep:
            a[i, :, :] = np.nan
    a = np.round(a / 4.0) * 4.0  # ties; distinct within columns after winsor
    kw = {"winsor_limits": (0.6, 0.6), "kappa": 2.0}
    calls = {"rank": 0, "axis0": 0}
    real_rank = sgp._winsorize_axis0_rank_path_cp
    real_axis0 = sgp._winsorize_axis0_cp

    def rank_spy(mod, arr, limits):
        calls["rank"] += 1
        return real_rank(mod, arr, limits)

    def axis0_spy(mod, arr, limits):
        calls["axis0"] += 1
        out = real_axis0(mod, arr, limits)
        return out

    monkeypatch.setattr(sgp, "_winsorize_axis0_rank_path_cp", rank_spy)
    monkeypatch.setattr(sgp, "_winsorize_axis0_cp", axis0_spy)
    cpu = _cpu(a, None, **kw)
    full = _gpu_full(a, None, **kw)
    assert calls["rank"] > 0, "test design: full run must hit the fallback"
    calls["rank"] = 0
    calls["axis0"] = 0
    tiled = _gpu_tiled(a, None, 16, **kw)
    assert calls["rank"] > 0, "expected per-tile rank fallback"
    assert calls["axis0"] > 0
    _assert_bitwise_equal("tiled overlap", cpu, full, tiled)


def test_tiled_extreme_limits_index_error_matches_untiled():
    """Extreme single-sided limit (1.0, 0.0) on an all-valid column raises
    the CPU's IndexError on the tiled path exactly like the untiled twin."""
    n = 10
    a = _make_stack(n, (40, 96), seed=806, nan_frac=0.0)
    kw = {"winsor_limits": (1.0, 0.0)}
    with pytest.raises(IndexError):
        _gpu_full(a, None, **kw)
    with pytest.raises(IndexError):
        _gpu_tiled(a, None, 16, **kw)
    with pytest.raises(IndexError):
        _gpu_tiled(a, None, 1, **kw)


def test_tiled_zero_limits_any_n_fastpath(monkeypatch):
    """(0.0, 0.0) limits: all-zero-rank fast path for ANY N on the tiled path
    (per-iteration winsor sort skipped; survivor-bound sort retained), bitwise
    with the untiled twin."""
    n = 30
    a = _make_stack(n, (40, 96), seed=807)
    a[3, 20, 48] = 1e6  # deterministic gross isolated outlier (guard rejects it)
    kw = {"winsor_limits": (0.0, 0.0)}
    calls = {"axis0": 0, "bounds": 0}
    real_axis0 = sgp._winsorize_axis0_cp
    real_bounds = sgp._winsorize_bounds_cp

    def axis0_spy(mod, arr, limits):
        calls["axis0"] += 1
        return real_axis0(mod, arr, limits)

    def bounds_spy(mod, arr, limits):
        calls["bounds"] += 1
        return real_bounds(mod, arr, limits)

    monkeypatch.setattr(sgp, "_winsorize_axis0_cp", axis0_spy)
    monkeypatch.setattr(sgp, "_winsorize_bounds_cp", bounds_spy)
    cpu = _cpu(a, None, **kw)
    full = _gpu_full(a, None, **kw)
    tiled = _gpu_tiled(a, None, 16, **kw)
    n_tiles = len(_winsor_tile_slices((40, 96), 16))
    assert calls == {"axis0": 0, "bounds": 1 + n_tiles}, calls
    assert float(cpu[2]) > 0.0, "test design: gross witness must be rejected"
    _assert_bitwise_equal("tiled zero limits n=30", cpu, full, tiled)


# ---------------------------------------------------------------------------
# 9. contract: no-return_weights form + regime predicate used per tile
# ---------------------------------------------------------------------------


def test_tiled_return_contract_without_weights():
    """return_weights=False returns (result, rejected_pct) NumPy/float like
    the untiled twin."""
    a = _make_stack(24, (40, 96), seed=900)
    out = stack_winsorized_sigma_gpu_tiled(
        _images(a), None, tile_shape=16
    )
    full = stack_winsorized_sigma_gpu(_images(a), None)
    assert isinstance(out[0], np.ndarray) and out[0].dtype == np.float32
    assert isinstance(out[1], float)
    assert np.array_equal(out[0], full[0], equal_nan=True)
    assert out[1] == full[1]


def test_zero_rank_regime_identical_across_tiles():
    """The Phase C regime predicate is decided on N_batch (identical for
    every tile), never per-tile pixel counts."""
    for n, limits, expected in [
        (19, (0.05, 0.05), True),
        (20, (0.05, 0.05), False),
        (30, (0.05, 0.05), False),
        (50, (0.0, 0.0), True),
        (50, (0.03, 0.03), False),
    ]:
        assert _winsor_zero_rank_regime(limits, n) is expected


# ---------------------------------------------------------------------------
# R3: small-N normalized witness — full vs tiled bitwise + CPU parity
# ---------------------------------------------------------------------------
def test_tiled_small_n_normalized_witness():
    """N=5 normalized witness (isolated .9978 over a ~.04 background in every
    column) reduces identically full vs tiled (bitwise) and matches the CPU,
    with the one-pass guard rejecting exactly the .9978 frame (20%%)."""
    n = 5
    H, W = 4, 96
    a = np.full((n, H, W), 0.04, dtype=np.float32)
    a[0, :, :] = 0.0401
    a[1, :, :] = 0.0399
    a[2, :, :] = 0.0402
    a[4, :, :] = 0.9978
    cpu = _cpu(a, None)
    full = _gpu_full(a, None)
    assert cpu[2] == pytest.approx(20.0)
    for ts in (1, 2, (2, 32)):
        tiled = _gpu_tiled(a, None, ts)
        _assert_bitwise_equal("small_n_normalized ts=%s" % (ts,), cpu, full, tiled)


def test_tiled_small_n_normalized_scaled_parity():
    """The small-N normalized witness, scaled, stays full-vs-tiled bitwise
    and CPU-exact (the guard classification is scale-invariant)."""
    base = [0.0400, 0.0401, 0.0399, 0.0402, 0.9978]
    H, W = 4, 96
    for scale in (1000.0, 65535.0, 1e-3):
        a = np.empty((5, H, W), dtype=np.float32)
        for i, v in enumerate(base):
            a[i, :, :] = v * scale
        cpu = _cpu(a, None)
        full = _gpu_full(a, None)
        assert cpu[2] == pytest.approx(20.0)
        tiled = _gpu_tiled(a, None, 2)
        _assert_bitwise_equal("small_n_scaled_%.4g ts=2" % scale, cpu, full, tiled)
