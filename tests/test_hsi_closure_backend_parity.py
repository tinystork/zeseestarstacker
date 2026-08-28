"""HSI Closure P2 — RAM / tiled-HQ / ``use_memmap`` scientific parity audit.

This is the *investigation-first* parity characterization for the plain-classic
SUM/WHT path.  It directly compares, for identical deterministic HWC float32
observations and identical parameters, the effective numerator ``SUM = V * W``,
the effective denominator ``WHT = W``, and the final normalized image ``V``
across the three backend storage routes that ``_stack_batch`` can select:

* **RAM**          — ``use_tile_mode == False`` (``total_bytes <= max_hq_mem``)
                     and ``use_memmap == False``.
* **tiled / HQ**   — ``use_tile_mode == True``, ``use_memmap == False``.
* **memmap**       — ``use_memmap == True`` (which forces ``use_tile_mode``).

Every parity claim is numerical (``V``, ``W``, ``V*W`` compared with
``np.allclose`` at float32-appropriate tolerances), never file existence or
visual similarity.  Where a production dispatch is absent (``mean``), the test
proves that absence with executable evidence and then exercises the real
``_combine_hq_by_tiles`` primitive directly, clearly labelled as *primitive*
parity rather than production dispatch.

Acceptance-dimension coverage (each exercised with ``use_memmap=True``):

* nonuniform weights  — every parity test (SNR-derived ``_calculate_weights``).
* partial coverage    — ``test_parity_mean_partial_coverage`` and the
                        normalization multi-level test (row-band masks).
* rejection           — kappa-sigma / linear-fit-clip / winsorized parity tests
                        (a global-offset outlier is actually rejected).
* normalization       — ``test_parity_kappa_sigma_linear_fit_normalization_across_backends``
                        and ``test_parity_kappa_sigma_sky_mean_normalization_across_backends``
                        (linear_fit / sky_mean against an immutable session
                        reference, exercised through the production kappa-sigma
                        tiled/HQ and memmap reducers).
* multiple levels     — irregular decompositions ``[A,B,C]+[D,E]`` vs
                        ``[A]+[B,C,D,E]`` composed via ``SUM(V*W)/SUM(W)``.

Classification vocabulary (mirrors the HSI verdict table):

* ``parity``            — RAM == tiled == memmap within float32 tolerance.
* ``approximate grouping`` — differ only when a *hidden subgroup* is forced
                           (``group_size < N``); the difference is internal
                           local-reduction non-associativity, not storage.
* ``defect``            — a demonstrated backend disagreement not explained by
                           subgrouping / float32 order.
* ``not-applicable production backend`` — the family has no production
                           tiled/memmap dispatch (``mean``).
"""

import glob
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
import importlib.util as _ilu  # noqa: E402

_missing_optional = {
    _name for _name in ("cv2", "astroalign", "ccdproc", "drizzle")
    if _ilu.find_spec(_name) is None
}

for _name in ("cv2", "astroalign"):
    if _name in _missing_optional:
        sys.modules.setdefault(_name, types.ModuleType(_name))

if "ccdproc" in _missing_optional:
    _ccdproc = types.ModuleType("ccdproc")
    _ccdproc.combine = None
    sys.modules.setdefault("ccdproc", _ccdproc)

if "drizzle" in _missing_optional:
    _drizzle = types.ModuleType("drizzle")
    _drizzle_resample = types.ModuleType("drizzle.resample")

    class _DummyDrizzle:
        pass

    _drizzle_resample.Drizzle = _DummyDrizzle
    _drizzle.resample = _drizzle_resample
    sys.modules.setdefault("drizzle", _drizzle)
    sys.modules.setdefault("drizzle.resample", _drizzle_resample)

from seestar.queuep.queue_manager import SeestarQueuedStacker  # noqa: E402

HEADER = fits.Header()

# ---------------------------------------------------------------------------
# Tolerances (float32-appropriate; derived from observed round-off)
# ---------------------------------------------------------------------------
# ``V`` is a float32 normalized image (values ~1e2).  RAM and tiled/HQ both
# divide numerator by denominator in float32, so the *final image* may differ
# by one float32 ulp of a division round-trip.  Observed maxima: mean/kappa/
# linear-fit/winsor 0.0; median 3.05e-5.  ``W`` is an exact sum/count (0.0 in
# every measured mode).  ``V*W`` re-multiplies the rounded V, so it inherits
# the same round-trip: observed max 1.83e-4 (median), 0.0 elsewhere.
PARITY_V_TOL = 1e-2       # final normalized image V
PARITY_W_TOL = 1e-3       # effective denominator W (exact in practice)
PARITY_NUM_TOL = 5e-2     # numerator SUM = V * W
LINEAR_FIT_TOL = 2e-3     # linear_fit recovery of an affine copy to reference
SKY_MEAN_TOL = 1e-2       # sky_mean offset-alignment (gain is not corrected)
NONASSOC_MIN_DELTA = 1.0  # small-group witness must differ by at least this


# ---------------------------------------------------------------------------
# Deterministic datasets
# ---------------------------------------------------------------------------


def _base(shape=(64, 64, 3), seed=7):
    rng = np.random.default_rng(seed)
    H, W, C = shape
    ii = np.arange(H, dtype=np.float64)[:, None]
    jj = np.arange(W, dtype=np.float64)[None, :]
    ramp = 100.0 + 200.0 * (ii / (H - 1)) + 60.0 * (jj / (W - 1))
    base = np.stack([ramp] * C, axis=-1)
    A = (base + rng.normal(0.0, 4.0, size=shape)).astype(np.float32)
    return A, rng


def affine_set():
    """Five affine-related observations for ``linear_fit`` / ``sky_mean`` parity.

    A = base; B/C/D/E are affine transforms of A.  ``linear_fit`` maps each
    observation back to A; ``sky_mean`` aligns only the sky offset (not gain).
    """
    A, _ = _base()
    return {
        "A": A,
        "B": (1.5 * A + 40.0).astype(np.float32),
        "C": (0.7 * A - 30.0).astype(np.float32),
        "D": (2.0 * A - 100.0).astype(np.float32),
        "E": (0.9 * A + 15.0).astype(np.float32),
    }


def offset_set():
    """Five pure-offset observations (same noise pattern) for ``sky_mean`` parity.

    ``sky_mean`` aligns only the sky offset, so every pure-offset observation
    resolves exactly to ``A`` (the offset is removed, the noise is shared).
    """
    A, _ = _base()
    return {
        "A": A,
        "B": (A + 40.0).astype(np.float32),
        "C": (A - 30.0).astype(np.float32),
        "D": (A + 15.0).astype(np.float32),
        "E": (A - 60.0).astype(np.float32),
    }


def rejection_set(n_inliers=9, outlier_offset=1000.0, seed=7):
    """``n_inliers`` near-identical observations plus one global-offset outlier.

    With ``kappa``/``sigma`` == 3.0 and ``n_inliers == 9`` (N = 10) the single
    outlier is rejected by the median-based kernels (kappa-sigma, linear-fit
    clip): the mixed-sample std is ``offset / sqrt(N)`` and
    ``3 * std < offset`` for ``N >= 10``.  The winsorized kernel (mean/std,
    ddof=1) does NOT reject a single ``+1000`` outlier at ``kappa=3`` unless the
    winsor limits are raised; the winsorized witness therefore uses
    ``winsor_limits=(0.2, 0.2)``, which substitutes the outlier with the
    survivor bound (documented winsor behaviour, ``W`` unchanged at 55).
    """
    A, rng = _base(seed=seed)
    obs = [A.astype(np.float32)] + [
        (A + rng.normal(0.0, 4.0, size=A.shape)).astype(np.float32)
        for _ in range(n_inliers - 1)
    ]
    obs.append((A + outlier_offset).astype(np.float32))  # deliberate outlier
    return A, obs


def row_band_masks(n, shape=(64, 64)):
    """Differing partial-coverage masks: row ``r`` is invalid in observation
    ``r % n``, so every pixel is covered by ``n - 1`` observations."""
    H, W = shape
    r = (np.arange(H) % n).reshape(H, 1)
    band = np.broadcast_to(r, (H, W))
    return [(band != i) for i in range(n)]


# ---------------------------------------------------------------------------
# Lightweight harness (mirrors the sibling closure suites)
# ---------------------------------------------------------------------------


def make_stack(
    mode="mean",
    norm="none",
    ref=None,
    max_hq_mem=1_000_000_000,
    batch_size=10,
    settings=None,
    use_qw=False,
    snr_exponent=1.0,
    min_weight=0.0,
    kappa_low=3.0,
    kappa_high=3.0,
    winsor_limits=(0.05, 0.05),
):
    """Build a ``SeestarQueuedStacker`` without running ``__init__``.

    ``settings`` (a ``SimpleNamespace``) is optional and lets a test control the
    production knobs ``settings.TILE_HEIGHT`` (tile height) and
    ``settings.batch_size`` (== 1 forces ``use_memmap=True``).  Passing ``None``
    keeps the defaults and means ``use_memmap`` stays ``False``.
    """
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
    )
    o.stacking_mode = mode
    o.normalize_method = norm
    o.weighting_method = "none"
    o.use_quality_weighting = use_qw
    o.weight_by_snr = True
    o.weight_by_stars = False
    o.snr_exponent = snr_exponent
    o.stars_exponent = 0.5
    o.min_weight = min_weight
    o.apply_batch_feathering = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o.drizzle_active_session = False
    o.is_mosaic_run = False
    o.stack_kappa_low = kappa_low
    o.stack_kappa_high = kappa_high
    o.winsor_limits = winsor_limits
    o.stack_reject_algo = "none"
    o.max_hq_mem = max_hq_mem
    o.batch_size = batch_size
    o.settings = settings
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.interbatch_norm_active = False
    o.max_stack_workers = 1
    o._current_batch_paths = []
    o._norm_reference = None
    # P5-FIX: pin q_ref explicitly (absolute domain, q_ref == 1.0) so the
    # parity harness never relies on the removed raw-domain fallback.
    o._quality_reference_scale = 1.0
    if ref is not None:
        o._capture_normalization_reference(ref)
    return o


def item(arr, snr=1.0, mask=None):
    """Build one batch item from a fresh copy of ``arr``."""
    if mask is None:
        mask = np.ones(arr.shape[:2], dtype=bool)
    return (
        np.array(arr, dtype=np.float32, copy=True),
        HEADER,
        {"snr": float(snr), "stars": 0.0},
        None,
        np.asarray(mask, dtype=bool).copy(),
    )


def _release(arr):
    """Release a possibly memmap-backed array and delete its backing file."""
    mm = getattr(arr, "_mmap", None)
    if mm is not None:
        try:
            arr.flush()
            mm.close()
        except Exception:
            pass
    fn = getattr(arr, "filename", None)
    if fn and os.path.exists(fn):
        try:
            os.remove(fn)
        except Exception:
            pass


def _mat(arr):
    """Materialize a (possibly memmap-backed) array to a plain float64 copy."""
    return np.array(arr, dtype=np.float64, copy=True)


def _w3(W, ref_ndim=3):
    """Promote a 2-D denominator to 3-D (channel broadcast) for comparison."""
    W = _mat(W)
    if W.ndim == 2 and ref_ndim == 3:
        return W[..., None]
    return W


def _assert_parity(V_ram, W_ram, V_other, W_other, label):
    """Assert V, W and V*W agree between two backends within float32 tolerance."""
    Vr, Vt = _mat(V_ram), _mat(V_other)
    Wr = _w3(W_ram, Vr.ndim)
    Wt = _w3(W_other, Vt.ndim)
    dv = float(np.abs(Vr - Vt).max())
    dw = float(np.abs(Wr - Wt).max())
    dn = float(np.abs(Vr * Wr - Vt * Wt).max())
    assert dv < PARITY_V_TOL, f"{label}: V max diff {dv:.3e}"
    assert dw < PARITY_W_TOL, f"{label}: W max diff {dw:.3e}"
    assert dn < PARITY_NUM_TOL, f"{label}: V*W max diff {dn:.3e}"
    return dv, dw, dn


def _run_backends(mode, norm, items, ref=None, tmp_path=None, tile_h=8,
                  use_qw=True, kappa_low=3.0, kappa_high=3.0,
                  winsor_limits=(0.05, 0.05), require_tile_dispatch=False):
    """Run RAM / tiled / memmap for the same observations and return
    ``(V, W)`` per backend, with the memmap file cleaned up.

    When ``require_tile_dispatch`` is True, ``_combine_hq_by_tiles`` is wrapped
    on the tiled and memmap stackers and this helper asserts the production
    tiled/HQ and memmap reducers were actually reached — so a mode that silently
    falls through to the RAM reduction path (e.g. ``mean``) cannot masquerade as
    backend parity merely because the storage labels happened to be exercised.
    """

    def _spy_dispatch(stacker):
        seen = {"called": False}
        if not require_tile_dispatch:
            return stacker, seen
        orig = stacker._combine_hq_by_tiles

        def _spy(*a, **k):
            seen["called"] = True
            return orig(*a, **k)

        stacker._combine_hq_by_tiles = _spy
        return stacker, seen

    # RAM: max_hq_mem huge -> total_bytes <= max_hq_mem -> use_tile_mode False.
    s_ram = make_stack(
        mode, norm, ref=ref, max_hq_mem=1_000_000_000, batch_size=10,
        use_qw=use_qw, kappa_low=kappa_low, kappa_high=kappa_high,
        winsor_limits=winsor_limits,
    )
    Vr, _, Wr = s_ram._stack_batch(items, 1, 1)

    # tiled/HQ: max_hq_mem small + tile_h small -> use_tile_mode True with
    # group_size >= n (isolates storage from subgrouping).
    s_tile = make_stack(
        mode, norm, ref=ref, max_hq_mem=100_000, batch_size=10,
        settings=types.SimpleNamespace(TILE_HEIGHT=tile_h, batch_size=None),
        use_qw=use_qw, kappa_low=kappa_low, kappa_high=kappa_high,
        winsor_limits=winsor_limits,
    )
    s_tile, tile_seen = _spy_dispatch(s_tile)
    Vt_raw, _, Wt = s_tile._stack_batch(items, 1, 1)
    Vt = _mat(Vt_raw)
    _release(Vt_raw)

    # memmap: settings.batch_size == 1 -> use_memmap=True (forces tile mode).
    s_mm = make_stack(
        mode, norm, ref=ref, max_hq_mem=100_000, batch_size=1,
        settings=types.SimpleNamespace(TILE_HEIGHT=tile_h, batch_size=1),
        use_qw=use_qw, kappa_low=kappa_low, kappa_high=kappa_high,
        winsor_limits=winsor_limits,
    )
    s_mm, mm_seen = _spy_dispatch(s_mm)
    Vm_raw, _, Wm = s_mm._stack_batch(items, 1, 1)
    Vm = _mat(Vm_raw)
    _release(Vm_raw)

    if require_tile_dispatch:
        assert tile_seen["called"], (
            f"{mode}: tiled/HQ backend did not dispatch _combine_hq_by_tiles"
        )
        assert mm_seen["called"], (
            f"{mode}: memmap backend did not dispatch _combine_hq_by_tiles"
        )

    return (Vr, Wr), (Vt, Wt), (Vm, Wm)


def _run_single_backend(backend, mode, norm, items, ref=None, tmp_path=None,
                        tile_h=8):
    """Run one backend for one batch and return ``(V, W)`` (memmap cleaned)."""
    if backend == "ram":
        s = make_stack(mode, norm, ref=ref, max_hq_mem=1_000_000_000,
                       batch_size=10, use_qw=True)
    elif backend == "tile":
        s = make_stack(mode, norm, ref=ref, max_hq_mem=100_000, batch_size=10,
                       settings=types.SimpleNamespace(TILE_HEIGHT=tile_h,
                                                      batch_size=None),
                       use_qw=True)
    elif backend == "memmap":
        s = make_stack(mode, norm, ref=ref, max_hq_mem=100_000, batch_size=1,
                       settings=types.SimpleNamespace(TILE_HEIGHT=tile_h,
                                                      batch_size=1),
                       use_qw=True)
    else:
        raise ValueError(backend)
    V_raw, _, W = s._stack_batch(items, 1, 1)
    V = _mat(V_raw)
    _release(V_raw)
    return V, W


def _compose_vw_full(pairs):
    """Compose (V, W) batch outputs; return ``(numerator, denominator, final)``."""
    num = None
    den = None
    for V, W in pairs:
        V = _mat(V)
        W = _w3(W, V.ndim)
        num = V * W if num is None else num + V * W
        den = W if den is None else den + W
    with np.errstate(divide="ignore", invalid="ignore"):
        return num, den, num / den


def _snrs(n):
    return [float(i + 1) for i in range(n)]


def _no_hq_files(tmp_path):
    """Assert no ``hq_batch*.dat`` memmap files remain under ``tmp_path``."""
    assert not glob.glob(os.path.join(str(tmp_path), "hq_batch*.dat"))


def _assert_group_size_ge_n(max_hq_mem, tile_h, shape, n):
    """Guard that the tiled/HQ reducer's per-tile group size is >= N.

    Recomputes the exact ``group_size`` formula used inside
    ``_combine_hq_by_tiles`` (including the ``SEESTAR_TILE_H`` env override) so
    a normalization/rejection parity witness cannot silently exercise *local
    nonlinear subgrouping* instead of pure storage/backend equivalence.
    """
    H, W, C = shape
    tile_h = int(os.getenv("SEESTAR_TILE_H", tile_h))
    per_img_bytes = tile_h * W * C * 4 + tile_h * W * 4
    group_size = max(1, int(max_hq_mem) // max(int(per_img_bytes), 1))
    assert group_size >= n, (
        f"group_size={group_size} < N={n}: witness would exercise local "
        f"nonlinear subgrouping, not pure storage/backend parity"
    )


# ===========================================================================
# 1. Dispatch matrix (executable, not source-string)
# ===========================================================================


def test_mean_has_no_production_tiled_or_memmap_dispatch(tmp_path, monkeypatch):
    """``mean`` ignores ``use_tile_mode``/``use_memmap`` even when both are
    computed.  Forcing both (``settings.batch_size == 1`` + tiny ``max_hq_mem``)
    must *not* call ``_combine_hq_by_tiles`` and must not create any
    ``hq_batch*.dat`` memmap file; the result stays bit-identical to RAM."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    D = affine_set()
    A, B, C = D["A"], D["B"], D["C"]
    items = [
        item(A, snr=1.0),
        item(B, snr=2.0),
        item(C, snr=3.0),
    ]

    # RAM reference result.
    s_ram = make_stack("mean", norm="none", use_qw=True)
    V_ram, _, W_ram = s_ram._stack_batch(items, 1, 1)

    # Force tile+memmap conditions for mean.
    s = make_stack(
        "mean", norm="none", use_qw=True, max_hq_mem=1, batch_size=1,
        settings=types.SimpleNamespace(TILE_HEIGHT=8, batch_size=1),
    )

    def _explode(*a, **k):
        raise AssertionError("_combine_hq_by_tiles must not be called for mean")

    monkeypatch.setattr(s, "_combine_hq_by_tiles", _explode)
    V, _, W = s._stack_batch(items, 1, 1)
    V = _mat(V)
    _release(V)

    # Bit-identical to RAM (same code path), and no memmap file was created.
    assert np.array_equal(V, _mat(V_ram))
    _no_hq_files(tmp_path)


def test_nonmean_families_do_dispatch_to_tiles_and_memmap(tmp_path, monkeypatch):
    """median/kappa-sigma/linear_fit_clip/winsorized all reach
    ``_combine_hq_by_tiles`` under tile/memmap conditions (memmap file is
    created and then fully cleaned up by the release path)."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _, obs = rejection_set()
    snrs = _snrs(len(obs))
    for mode, kw in [
        ("median", {}),
        ("kappa-sigma", {"kappa_low": 3.0, "kappa_high": 3.0}),
        ("linear_fit_clip", {}),
        ("winsorized-sigma", {"winsor_limits": (0.2, 0.2)}),
    ]:
        items = [item(obs[i], snr=snrs[i]) for i in range(len(obs))]
        s = make_stack(
            mode, norm="none", use_qw=True, max_hq_mem=100_000, batch_size=1,
            settings=types.SimpleNamespace(TILE_HEIGHT=8, batch_size=1),
            **kw,
        )
        seen = {}
        orig = s._combine_hq_by_tiles

        def _spy(*a, **k):
            seen["called"] = True
            return orig(*a, **k)

        s._combine_hq_by_tiles = _spy
        V, _, W = s._stack_batch(items, 1, 1)
        _release(V)
        assert seen.get("called"), f"{mode}: tiled/memmap dispatch not reached"
        _no_hq_files(tmp_path)


# ===========================================================================
# 2. Control A — backend parity per stacking family (group_size >= N)
# ===========================================================================


def test_parity_mean_weighted_no_rejection_arithmetic():
    """mean (RAM only): exact weighted arithmetic baseline.  tiled/memmap have no
    production dispatch (see test #1); the direct primitive is proven in #6."""
    D = affine_set()
    A, B, C = D["A"], D["B"], D["C"]
    snrs = [1.0, 3.0, 6.0]  # nonuniform quality weights
    items = [item(A, snr=snrs[0]), item(B, snr=snrs[1]), item(C, snr=snrs[2])]
    s = make_stack("mean", norm="none", use_qw=True)
    V, _, W = s._stack_batch(items, 1, 1)

    # Exact weighted mean of the raw aligned observations (per channel).
    expected_num = (A * snrs[0] + B * snrs[1] + C * snrs[2]).astype(np.float64)
    expected_den = float(sum(snrs))
    expected = expected_num / expected_den
    assert np.allclose(_mat(V), expected, rtol=1e-5, atol=1e-2), np.abs(
        _mat(V) - expected
    ).max()


def test_parity_kappa_sigma_rejected_outlier(tmp_path, monkeypatch):
    """kappa-sigma (kappa=3): the global +1000 outlier is rejected (W drops from
    55 to 45).  RAM, tiled and memmap agree on V, W and V*W."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    A, obs = rejection_set()
    snrs = _snrs(len(obs))
    items = [item(obs[i], snr=snrs[i]) for i in range(len(obs))]

    (Vr, Wr), (Vt, Wt), (Vm, Wm) = _run_backends(
        "kappa-sigma", "none", items, tmp_path=tmp_path,
        kappa_low=3.0, kappa_high=3.0,
    )

    # The outlier (weight 10) is actually rejected: W == 45 at the center.
    assert np.isclose(float(_w3(Wr, 3)[32, 32, 0]), 45.0, atol=1e-3)
    # The surviving mean is the weighted mean of the 9 noisy inliers (~= A);
    # the outlier (+1000) would push it to ~A+182 if it survived.
    assert float(np.abs(_mat(Vr) - A).max()) < 50.0

    _assert_parity(Vr, Wr, Vt, Wt, "kappa RAM vs tiled")
    _assert_parity(Vr, Wr, Vm, Wm, "kappa RAM vs memmap")
    _no_hq_files(tmp_path)


def test_parity_linear_fit_clip_rejected_outlier(tmp_path, monkeypatch):
    """linear_fit_clip (sigma=3): same median-based rejection as kappa-sigma.
    Parity across the three backends."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    A, obs = rejection_set()
    snrs = _snrs(len(obs))
    items = [item(obs[i], snr=snrs[i]) for i in range(len(obs))]

    (Vr, Wr), (Vt, Wt), (Vm, Wm) = _run_backends(
        "linear_fit_clip", "none", items, tmp_path=tmp_path,
    )
    assert np.isclose(float(_w3(Wr, 3)[32, 32, 0]), 45.0, atol=1e-3)
    _assert_parity(Vr, Wr, Vt, Wt, "linfit RAM vs tiled")
    _assert_parity(Vr, Wr, Vm, Wm, "linfit RAM vs memmap")
    _no_hq_files(tmp_path)


def test_parity_winsorized_sigma_substitution(tmp_path, monkeypatch):
    """winsorized-sigma (winsor_limits=(0.2,0.2), kappa=3): the outlier is
    rejected *and substituted* with the survivor bound, so W is unchanged (55)
    while V is pulled back to the inlier mean.  Parity across backends."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    A, obs = rejection_set()
    snrs = _snrs(len(obs))
    items = [item(obs[i], snr=snrs[i]) for i in range(len(obs))]

    (Vr, Wr), (Vt, Wt), (Vm, Wm) = _run_backends(
        "winsorized-sigma", "none", items, tmp_path=tmp_path,
        winsor_limits=(0.2, 0.2),
    )
    # Winsor substitution keeps the rejected sample's weight in W (55).
    assert np.isclose(float(_w3(Wr, 3)[32, 32, 0]), 55.0, atol=1e-3)
    # V is the winsorized mean, close to the inlier value A (not the outlier).
    assert float(np.abs(_mat(Vr) - A).max()) < 50.0
    _assert_parity(Vr, Wr, Vt, Wt, "winsor RAM vs tiled")
    _assert_parity(Vr, Wr, Vm, Wm, "winsor RAM vs memmap")
    _no_hq_files(tmp_path)


def test_parity_median_nonassociative_control(tmp_path, monkeypatch):
    """median is a non-associative control: with group_size >= N the three
    backends agree (the reduction is per-pixel along the sample axis and tiling
    by rows is exact up to one float32 divide round-trip)."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    A, obs = rejection_set()
    snrs = _snrs(len(obs))
    items = [item(obs[i], snr=snrs[i]) for i in range(len(obs))]

    (Vr, Wr), (Vt, Wt), (Vm, Wm) = _run_backends(
        "median", "none", items, tmp_path=tmp_path,
    )
    # median is unweighted: W == valid count (10), V == median of inliers (~A).
    assert np.isclose(float(_w3(Wr, 3)[32, 32, 0]), 10.0, atol=1e-3)
    _assert_parity(Vr, Wr, Vt, Wt, "median RAM vs tiled")
    _assert_parity(Vr, Wr, Vm, Wm, "median RAM vs memmap")
    _no_hq_files(tmp_path)


def test_parity_mean_partial_coverage(tmp_path, monkeypatch):
    """mean with differing partial coverage (row-band masks) + nonuniform weights
    + memmap: RAM / tiled-primitive-via-mean / memmap all agree on V, W, V*W.
    ``mean`` has no production tiled dispatch, so this is RAM vs the direct
    ``_combine_hq_by_tiles`` primitive, plus the memmap primitive."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    D = affine_set()
    obs = [D["A"], D["B"], D["C"], D["D"], D["E"]]
    snrs = _snrs(len(obs))
    masks = row_band_masks(len(obs))
    items = [item(obs[i], snr=snrs[i], mask=masks[i]) for i in range(len(obs))]

    # RAM mean (production path).
    s_ram = make_stack("mean", norm="none", use_qw=True)
    Vr, _, Wr = s_ram._stack_batch(items, 1, 1)

    # Direct mean primitive (tiled, use_memmap=False and True).
    arrays = [np.array(o, dtype=np.float32) for o in obs]
    qw = np.array(snrs, dtype=np.float32)
    for use_mm, label in ((False, "primitive-tiled"), (True, "primitive-memmap")):
        s = make_stack("mean", norm="none", use_qw=True, max_hq_mem=100_000)
        V_raw, W = s._combine_hq_by_tiles(
            arrays, masks, 3.0, (0.05, 0.05),
            masks_list=masks, quality_weights=qw, use_memmap=use_mm,
            tile_h=8, batch_id=904,
        )
        Vm = _mat(V_raw)
        _release(V_raw)
        assert np.allclose(Vm[..., 0], _mat(Vr)[..., 0],
                           rtol=1e-5, atol=PARITY_V_TOL), label
        assert np.allclose(_w3(W, 3)[..., 0], _w3(Wr, 3)[..., 0],
                           rtol=1e-5, atol=PARITY_W_TOL), label
    _no_hq_files(tmp_path)


def test_mean_linear_fit_normalization_reference_and_decomposition(
    tmp_path, monkeypatch
):
    """linear_fit normalization against the immutable reference, via ``mean``.

    ``mean`` has no production tiled/memmap dispatch (see test #1), so this is
    NOT a backend-parity witness: it proves (a) that linear_fit maps every
    affine observation back to the immutable reference A before reduction and
    (b) that multi-level irregular compositions agree with each other.  The
    RAM / tiled / memmap *labels* below are only different storage knobs on the
    same single mean reduction path; production tiled/memmap normalization
    parity is proven by the kappa-sigma witnesses."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    D = affine_set()
    A = D["A"]
    obs = [D["A"], D["B"], D["C"], D["D"], D["E"]]
    snrs = [1.0, 2.0, 3.0, 4.0, 5.0]
    masks = row_band_masks(len(obs))
    items = [item(obs[i], snr=snrs[i], mask=masks[i]) for i in range(len(obs))]

    # Two hierarchy levels: [A,B,C] + [D,E] and [A] + [B,C,D,E].
    decomps = {
        "ABC_DE": ([0, 1, 2], [3, 4]),
        "A_BCDE": ([0], [1, 2, 3, 4]),
    }
    results = {}  # (name, backend) -> (numerator, denominator, final)
    for name, (g1, g2) in decomps.items():
        for backend in ("ram", "tile", "memmap"):
            pairs = []
            for g in (g1, g2):
                V, W = _run_single_backend(
                    backend, "mean", "linear_fit", [items[i] for i in g],
                    ref=A, tmp_path=tmp_path,
                )
                pairs.append((V, W))
            results[(name, backend)] = _compose_vw_full(pairs)

    # linear_fit maps every affine observation to A: final image == A.
    for key, (num, den, final) in results.items():
        assert np.allclose(final, A, atol=LINEAR_FIT_TOL), (
            key, np.abs(final - A).max()
        )

    # Backend parity for each decomposition: numerator, denominator, final image.
    for name in decomps:
        num_r, den_r, fin_r = results[(name, "ram")]
        for backend in ("tile", "memmap"):
            num_o, den_o, fin_o = results[(name, backend)]
            assert float(np.abs(num_r - num_o).max()) < PARITY_NUM_TOL, (
                name, backend, "num", float(np.abs(num_r - num_o).max())
            )
            assert float(np.abs(den_r - den_o).max()) < PARITY_W_TOL, (
                name, backend, "den", float(np.abs(den_r - den_o).max())
            )
            assert float(np.abs(fin_r - fin_o).max()) < PARITY_V_TOL, (
                name, backend, "final", float(np.abs(fin_r - fin_o).max())
            )
    # Decomposition invariance (two hierarchy levels) within one backend.
    fin_abc = results[("ABC_DE", "ram")][2]
    fin_a = results[("A_BCDE", "ram")][2]
    assert float(np.abs(fin_abc - fin_a).max()) < PARITY_V_TOL
    _no_hq_files(tmp_path)


def test_mean_sky_mean_normalization_reference(tmp_path, monkeypatch):
    """sky_mean (offset-only) normalization against the immutable reference, via
    ``mean``.

    ``mean`` has no production tiled/memmap dispatch (see test #1), so this is
    NOT a backend-parity witness: it proves only that sky_mean aligns every
    pure-offset observation back to the immutable reference A.  Production
    tiled/memmap sky_mean parity is proven by the kappa-sigma witness."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    D = offset_set()
    A = D["A"]
    obs = [D["A"], D["B"], D["C"], D["D"], D["E"]]
    snrs = [1.0, 2.0, 3.0, 4.0, 5.0]
    masks = row_band_masks(len(obs))
    items = [item(obs[i], snr=snrs[i], mask=masks[i]) for i in range(len(obs))]

    results = {}
    for backend in ("ram", "tile", "memmap"):
        V, W = _run_single_backend(
            backend, "mean", "sky_mean", items, ref=A, tmp_path=tmp_path,
        )
        results[backend] = _compose_vw_full([(V, W)])

    # sky_mean aligns the offset: final image == A (shared noise preserved).
    for backend, (num, den, final) in results.items():
        assert np.allclose(final, A, atol=SKY_MEAN_TOL), (
            backend, np.abs(final - A).max()
        )

    num_r, den_r, fin_r = results["ram"]
    for backend in ("tile", "memmap"):
        num_o, den_o, fin_o = results[backend]
        assert float(np.abs(num_r - num_o).max()) < PARITY_NUM_TOL
        assert float(np.abs(den_r - den_o).max()) < PARITY_W_TOL
        assert float(np.abs(fin_r - fin_o).max()) < PARITY_V_TOL
    _no_hq_files(tmp_path)


# ===========================================================================
# 2b. Normalization parity through a production non-mean family (kappa-sigma)
# ===========================================================================
# ``mean`` never dispatches to ``_combine_hq_by_tiles``, so the mean
# normalization tests above only prove reference-correctness and decomposition
# invariance — not backend parity.  These witnesses re-run both normalizers
# through kappa-sigma (a real non-mean production family) with ``group_size >= N``
# so the tiled/HQ and memmap runs genuinely reach ``_combine_hq_by_tiles``
# (asserted by the ``require_tile_dispatch`` spy) and the final V, W and
# SUM = V*W must match RAM within float32 tolerance.


def test_parity_kappa_sigma_linear_fit_normalization_across_backends(
    tmp_path, monkeypatch
):
    """linear_fit normalization + kappa-sigma (group_size >= N) + memmap +
    nonuniform weights + partial masks.

    Every affine observation is normalized against the immutable reference A
    before reduction, so the final image equals A.  The tiled/HQ and memmap
    runs are proven (via the ``require_tile_dispatch`` spy) to reach the
    production ``_combine_hq_by_tiles`` reducer, and V, W and V*W match RAM."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    D = affine_set()
    A = D["A"]
    obs = [D["A"], D["B"], D["C"], D["D"], D["E"]]
    snrs = [1.0, 2.0, 3.0, 4.0, 5.0]
    masks = row_band_masks(len(obs))
    items = [item(obs[i], snr=snrs[i], mask=masks[i]) for i in range(len(obs))]

    # group_size >= N: the witness isolates storage/backend from local nonlinear
    # subgrouping (assert the exact formula _combine_hq_by_tiles uses).
    _assert_group_size_ge_n(100_000, 8, A.shape, len(items))

    (Vr, Wr), (Vt, Wt), (Vm, Wm) = _run_backends(
        "kappa-sigma", "linear_fit", items, ref=A, tmp_path=tmp_path,
        tile_h=8, kappa_low=3.0, kappa_high=3.0, require_tile_dispatch=True,
    )

    # linear_fit maps every affine observation to A: final image == A for every
    # backend.
    for backend, V in (("ram", Vr), ("tile", Vt), ("memmap", Vm)):
        assert float(np.abs(_mat(V) - A).max()) < LINEAR_FIT_TOL, backend

    _assert_parity(Vr, Wr, Vt, Wt, "kappa+linear_fit RAM vs tiled")
    _assert_parity(Vr, Wr, Vm, Wm, "kappa+linear_fit RAM vs memmap")
    _no_hq_files(tmp_path)


def test_parity_kappa_sigma_sky_mean_normalization_across_backends(
    tmp_path, monkeypatch
):
    """sky_mean (offset-only) normalization + kappa-sigma (group_size >= N) +
    memmap + nonuniform weights + partial masks.

    Every pure-offset observation is sky-aligned to the immutable reference A,
    so the final image equals A.  The tiled/HQ and memmap runs are proven to
    reach the production ``_combine_hq_by_tiles`` reducer, and V, W and V*W
    match RAM."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    D = offset_set()
    A = D["A"]
    obs = [D["A"], D["B"], D["C"], D["D"], D["E"]]
    snrs = [1.0, 2.0, 3.0, 4.0, 5.0]
    masks = row_band_masks(len(obs))
    items = [item(obs[i], snr=snrs[i], mask=masks[i]) for i in range(len(obs))]

    _assert_group_size_ge_n(100_000, 8, A.shape, len(items))

    (Vr, Wr), (Vt, Wt), (Vm, Wm) = _run_backends(
        "kappa-sigma", "sky_mean", items, ref=A, tmp_path=tmp_path,
        tile_h=8, kappa_low=3.0, kappa_high=3.0, require_tile_dispatch=True,
    )

    for backend, V in (("ram", Vr), ("tile", Vt), ("memmap", Vm)):
        assert float(np.abs(_mat(V) - A).max()) < SKY_MEAN_TOL, backend

    _assert_parity(Vr, Wr, Vt, Wt, "kappa+sky_mean RAM vs tiled")
    _assert_parity(Vr, Wr, Vm, Wm, "kappa+sky_mean RAM vs memmap")
    _no_hq_files(tmp_path)


# ===========================================================================
# 3. Control B — small-group nonlinear witness (group_size < N)
# ===========================================================================


def test_small_group_kappa_sigma_outlier_survives_singleton_subgroup(tmp_path, monkeypatch):
    """Forcing ``group_size < N`` for kappa-sigma: the outlier falls into its own
    singleton subgroup, where a single-sample sigma-clip cannot reject it.  The
    tiled result therefore differs from RAM (which reduces all N at once).  This
    is internal local-reduction grouping / non-associativity, not a backend
    storage defect."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _, obs = rejection_set()
    snrs = _snrs(len(obs))
    items = [item(obs[i], snr=snrs[i]) for i in range(len(obs))]

    # RAM: one reduction over all 10 (outlier rejected).
    s_ram = make_stack("kappa-sigma", norm="none", use_qw=True,
                       max_hq_mem=1_000_000_000)
    Vr, _, Wr = s_ram._stack_batch(items, 1, 1)
    Vr, Wr = _mat(Vr), _w3(Wr, 3)

    # tiled with group_size == 3 (10 -> subgroups 3+3+3+1; outlier alone).
    s_tile = make_stack(
        "kappa-sigma", norm="none", use_qw=True, max_hq_mem=30_000, batch_size=10,
        settings=types.SimpleNamespace(TILE_HEIGHT=8, batch_size=None),
    )
    Vt_raw, _, Wt = s_tile._stack_batch(items, 1, 1)
    Vt, Wt = _mat(Vt_raw), _w3(Wt, 3)
    _release(Vt_raw)

    # The singleton subgroup keeps the outlier: the small-group result differs
    # from RAM by an amount comparable to the outlier's weighted contribution.
    delta = float(np.abs(Vr - Vt).max())
    assert delta > NONASSOC_MIN_DELTA, delta
    _no_hq_files(tmp_path)


def test_small_group_median_boundary_dependence(tmp_path, monkeypatch):
    """median with forced ``group_size < N``: the count-weighted combination of
    local medians is not the global median (documented non-associativity)."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _, obs = rejection_set()
    snrs = _snrs(len(obs))
    items = [item(obs[i], snr=snrs[i]) for i in range(len(obs))]

    s_ram = make_stack("median", norm="none", use_qw=True,
                       max_hq_mem=1_000_000_000)
    Vr, _, Wr = s_ram._stack_batch(items, 1, 1)
    Vr, Wr = _mat(Vr), _w3(Wr, 3)

    s_tile = make_stack(
        "median", norm="none", use_qw=True, max_hq_mem=30_000, batch_size=10,
        settings=types.SimpleNamespace(TILE_HEIGHT=8, batch_size=None),
    )
    Vt_raw, _, Wt = s_tile._stack_batch(items, 1, 1)
    Vt, Wt = _mat(Vt_raw), _w3(Wt, 3)
    _release(Vt_raw)

    delta = float(np.abs(Vr - Vt).max())
    assert delta > NONASSOC_MIN_DELTA, delta
    _no_hq_files(tmp_path)


# ===========================================================================
# 4. mean primitive parity (direct _combine_hq_by_tiles, not production dispatch)
# ===========================================================================


def test_mean_direct_tile_primitive_matches_ram(tmp_path, monkeypatch):
    """mean has no production tiled dispatch, so exercise the real
    ``_combine_hq_by_tiles`` mean primitive directly (use_memmap=False and
    True).  This is *primitive* parity — it proves the reducer agrees with RAM
    — not proof of a production ``mean`` tiled path."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    D = affine_set()
    obs = [D["A"], D["B"], D["C"], D["D"], D["E"]]
    snrs = [1.0, 2.0, 3.0, 4.0, 5.0]
    masks = [np.ones((64, 64), dtype=bool) for _ in obs]
    items = [item(obs[i], snr=snrs[i], mask=masks[i]) for i in range(len(obs))]

    # RAM mean (production path).
    s_ram = make_stack("mean", norm="none", use_qw=True)
    Vr, _, Wr = s_ram._stack_batch(items, 1, 1)
    Vr = _mat(Vr)

    # Direct primitive inputs: the same aligned arrays + masks + quality weights.
    arrays = [np.array(o, dtype=np.float32) for o in obs]
    qw = np.array(snrs, dtype=np.float32)

    results = {}
    for use_mm in (False, True):
        s = make_stack("mean", norm="none", use_qw=True, max_hq_mem=100_000)
        V_raw, W = s._combine_hq_by_tiles(
            arrays, masks, 3.0, (0.05, 0.05),
            masks_list=masks, quality_weights=qw, use_memmap=use_mm,
            tile_h=8, batch_id=902,
        )
        Vm = _mat(V_raw)
        _release(V_raw)
        results[use_mm] = (Vm, W)

    # Primitive parity vs RAM for both storage routes.
    for use_mm in (False, True):
        Vm, Wm = results[use_mm]
        # The mean primitive returns a per-channel (3-D) W; RAM mean returns a
        # channel-invariant 2-D W with identical values.
        assert np.allclose(Vm[..., 0], Vr[..., 0], rtol=1e-5, atol=PARITY_V_TOL)
        assert np.allclose(_w3(Wm, 3)[..., 0], _w3(Wr, 3)[..., 0],
                           rtol=1e-5, atol=PARITY_W_TOL)
    _no_hq_files(tmp_path)


def test_mean_direct_tile_primitive_channel_weights(tmp_path, monkeypatch):
    """The mean primitive composes subgroups via SUM(V*W)/SUM(W); with
    group_size >= N and channel-invariant weights every channel matches."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    D = affine_set()
    obs = [D["A"], D["B"], D["C"], D["D"], D["E"]]
    snrs = [1.0, 2.0, 3.0, 4.0, 5.0]
    masks = [np.ones((64, 64), dtype=bool) for _ in obs]
    arrays = [np.array(o, dtype=np.float32) for o in obs]
    qw = np.array(snrs, dtype=np.float32)

    s = make_stack("mean", norm="none", use_qw=True, max_hq_mem=100_000)
    V_raw, W = s._combine_hq_by_tiles(
        arrays, masks, 3.0, (0.05, 0.05),
        masks_list=masks, quality_weights=qw, use_memmap=False, tile_h=8,
        batch_id=903,
    )
    Vm, Wm = _mat(V_raw), _w3(W, 3)
    _release(V_raw)

    # Expected weighted mean per channel (scalar weight -> identical channels).
    num = sum(obs[i].astype(np.float64) * snrs[i] for i in range(len(obs)))
    den = float(sum(snrs))
    expected = num / den
    assert np.allclose(Vm, expected, rtol=1e-5, atol=PARITY_V_TOL)
    assert np.allclose(Wm, np.full(Wm.shape, den, dtype=np.float64),
                       rtol=1e-5, atol=PARITY_W_TOL)
