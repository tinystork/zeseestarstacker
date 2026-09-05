"""HSI Closure P4 — real weighted-intermediate reprojection transport witness.

Purpose
-------
Close *only* HSI Closure section 4 by measuring, with the real interpolation
engine (``reproject.reproject_interp``, never mocked), whether the current
production reprojection paths transport the hierarchical ``SUM/WHT`` numerator
faithfully.

HSI contract under audit
------------------------
The accepted hierarchical representation is ``V = SUM / WHT`` with
``SUM = V * WHT``.  For a single batch the scientific numerator is therefore
``S = V * W`` and the denominator is ``W``.

When a *weighted intermediate* ``(V, W)`` enters spatial resampling, the
faithful transport of that numerator/denominator pair is::

    R(S)   = R(V * W)          (direct numerator transport)
    R(W)                        (denominator transport)

The current production paths instead transport ``V`` and ``W`` **separately**
and multiply *after* resampling::

    R(V) * R(W)                (separate value/weight transport)

For a linear interpolation operator ``R`` these two differ in general::

    R(V) * R(W) - R(V * W) = -sum_{k<l} a_k a_l (V_k - V_l)(W_k - W_l)

which is non-zero whenever both ``V`` and ``W`` vary over the interpolation
stencil (i.e. under a fractional-pixel shift).  This module proves the actual
numerical consequence rather than assuming it, and classifies it.

Production paths traced (all use separate transport)
----------------------------------------------------
1. ``SeestarQueuedStacker._reproject_batch_to_reference`` calls ``reproject_interp``
   separately on ``batch_image`` (V) and ``batch_wht`` (W) and returns both
   (``seestar/queuep/queue_manager.py`` ~7979).
2. ``seestar.core.incremental_reprojection.initialize_master`` /
   ``reproject_and_combine`` reproject ``batch_img`` and ``batch_cov``
   separately, then form ``master_sum += R(V) * R(W)`` and
   ``master_cov += R(W)`` (~58-76 / ~119-138).
3. ``SeestarQueuedStacker._reproject_classic_batches`` feeds the persisted ``V``
   plus effective per-channel ``WHT`` into
   ``enhancement.reproject_utils.reproject_and_coadd`` with ``V`` as input data
   and ``W`` as ``input_weights`` (~13571).
4. ``reproject_utils.reproject_and_coadd`` (local accumulator) computes
   ``proj_img = R(V)``, ``weight_proj = R(W) * footprint``, then
   ``sum_image += proj_img * weight_proj`` and ``cov_image += weight_proj``
   (~the ``[B1-COADD-FIX]`` block).

The astropy reference ``reproject.mosaicking.reproject_and_coadd`` (the default
branch of path 4) has the same conceptual semantics and is tested here as a
supplementary check; the verdict does not depend on that optional branch.

Verdict
-------
The separate transport is a *documented* approximation (it reproduces the
astropy ``reproject_and_coadd`` reference exactly, and the HSI design scope
reserves exact ``SUM/WHT`` composability to the plain non-reproject path), not
a backend or rejection defect.  See the module-level notes and the report.
"""

import numpy as np
import pytest
from astropy.wcs import WCS
from reproject import reproject_interp

from seestar.enhancement.reproject_utils import reproject_and_coadd
from seestar.core.incremental_reprojection import initialize_master

# ---------------------------------------------------------------------------
# Tolerances (chosen to separate a robust interior signal from float32 noise)
# ---------------------------------------------------------------------------
# Controls (identity transform, constant W) must agree to interpolation
# tolerance.  Observed: identity ~2.3e-11, constant W ~0.0.
EXACT_TOL = 1e-5
# The adversarial witness must be robustly non-zero over the *interior* (not a
# boundary / NaN artifact).  Observed with the fixtures below: max ~2.73,
# min ~0.99 over every interior pixel.
WITNESS_MIN_MAX_DELTA = 1.0
WITNESS_MIN_MEAN_ABS_DELTA = 0.5
# Production paths cast to float32; their numerator must equal the separate
# transport R(V)*R(W) to float32 tolerance (observed ~2e-4) while remaining
# clearly different from the direct transport R(V*W) (> 1.0).
PATH_F32_TOL = 2e-3


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SHAPE = (24, 24)  # (H, W)
SHIFT = (0.5, 0.3)  # genuine fractional-pixel shift (output CRPIX offset)


def make_wcs(shape=SHAPE, shift=(0.0, 0.0)):
    """Valid celestial TAN WCS; ``shift`` offsets CRPIX by a fractional pixel."""
    H, W = shape
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.cdelt = np.array([-0.01, 0.01])  # deg/pixel
    w.wcs.crpix = [W / 2.0 + shift[0], H / 2.0 + shift[1]]
    w.pixel_shape = (W, H)
    w.array_shape = (H, W)
    return w


def make_v(shape=SHAPE):
    """Deterministic non-constant science value (gradient + Gaussian bump)."""
    H, W = shape
    i, j = np.mgrid[0:H, 0:W]
    bump = 20.0 * np.exp(-((i - 12.0) ** 2 + (j - 8.0) ** 2) / (2 * 4.0**2))
    return (10.0 + 3.0 * i + 5.0 * j + bump).astype(np.float64)


def make_w(shape=SHAPE):
    """Deterministic positive spatially-varying weight (non-linear)."""
    H, W = shape
    i, j = np.mgrid[0:H, 0:W]
    return (1.0 + 0.25 * i + 0.4 * j + 0.05 * i * j + 0.01 * i * i).astype(
        np.float64
    )


def interior_mask(shape=SHAPE, inset=3):
    """Boolean mask of fully-covered interior pixels (excludes resample edges)."""
    m = np.zeros(shape, dtype=bool)
    m[inset:-inset, inset:-inset] = True
    return m


def reproj(arr, wcs_in, wcs_out, shape=SHAPE):
    """Real ``reproject_interp`` (bilinear) returning array and footprint."""
    return reproject_interp(
        (arr, wcs_in), wcs_out, shape_out=shape, return_footprint=True
    )


def _common_valid_mask(*footprints, shape=SHAPE):
    """Interior pixels where every reprojected footprint is fully covered."""
    m = interior_mask(shape)
    for fp in footprints:
        m &= np.asarray(fp) > 0
    return m


# ---------------------------------------------------------------------------
# Core witness: direct R(V*W) vs separate R(V)*R(W)
# ---------------------------------------------------------------------------


def test_adversarial_weighted_intermediate_transport_mismatch():
    """Direct numerator transport R(V*W) differs from R(V)*R(W) on the interior.

    The difference must be robust, finite and *not* confined to the resampled
    boundary — proving a genuine scientific consequence of separate transport,
    not a NaN / edge artifact.
    """
    V = make_v()
    W = make_w()
    S = V * W
    wcs_in = make_wcs()
    wcs_out = make_wcs(shift=SHIFT)

    RS, fpS = reproj(S, wcs_in, wcs_out)
    RV, fpV = reproj(V, wcs_in, wcs_out)
    RW, fpW = reproj(W, wcs_in, wcs_out)

    mask = _common_valid_mask(fpS, fpV, fpW)
    direct = RS[mask]
    separate = (RV * RW)[mask]
    delta = direct - separate

    assert np.isfinite(delta).all(), "witness must be finite"
    # Footprints in the interior are exactly 1.0, so this is a pure
    # interpolation-product difference, not a footprint artefact.
    assert np.all(np.asarray(fpS)[mask] == 1.0)
    assert np.all(np.asarray(fpV)[mask] == 1.0)
    assert np.all(np.asarray(fpW)[mask] == 1.0)

    max_abs = float(np.max(np.abs(delta)))
    mean_abs = float(np.mean(np.abs(delta)))
    assert max_abs > WITNESS_MIN_MAX_DELTA, f"max|delta|={max_abs}"
    assert (
        mean_abs > WITNESS_MIN_MEAN_ABS_DELTA
    ), f"mean|delta|={mean_abs}"


def test_control_identity_transform_agrees():
    """Direct and separate transport must agree exactly under identity WCS."""
    V = make_v()
    W = make_w()
    S = V * W
    wcs = make_wcs()

    RS, fpS = reproj(S, wcs, wcs)
    RV, fpV = reproj(V, wcs, wcs)
    RW, fpW = reproj(W, wcs, wcs)

    mask = _common_valid_mask(fpS, fpV, fpW)
    delta = RS[mask] - (RV * RW)[mask]
    assert np.max(np.abs(delta)) < EXACT_TOL


def test_control_constant_weight_agrees():
    """Multiplicativity holds (within tolerance) when W is spatially constant."""
    V = make_v()
    W = np.full(SHAPE, 2.0, dtype=np.float64)
    S = V * W
    wcs_in = make_wcs()
    wcs_out = make_wcs(shift=SHIFT)

    RS, fpS = reproj(S, wcs_in, wcs_out)
    RV, fpV = reproj(V, wcs_in, wcs_out)
    RW, fpW = reproj(W, wcs_in, wcs_out)

    mask = _common_valid_mask(fpS, fpV, fpW)
    delta = RS[mask] - (RV * RW)[mask]
    assert np.max(np.abs(delta)) < EXACT_TOL


# ---------------------------------------------------------------------------
# Real production path C: core incremental initialize_master
# ---------------------------------------------------------------------------


def test_incremental_initialize_master_uses_separate_transport():
    """``initialize_master`` returns master_sum = R(V)*R(W), not R(V*W)."""
    V = make_v()
    W = make_w()
    S = V * W
    wcs_in = make_wcs()
    wcs_out = make_wcs(shift=SHIFT)

    # Reference values (float64, real interpolation).
    RS, fpS = reproj(S, wcs_in, wcs_out)
    RV, fpV = reproj(V, wcs_in, wcs_out)
    RW, fpW = reproj(W, wcs_in, wcs_out)
    separate = RV * RW

    # Real production kernel (float32 path, use_gpu=False).
    master_sum, master_cov = initialize_master(
        V.astype(np.float32), W.astype(np.float32), wcs_in, wcs_out
    )

    mask = _common_valid_mask(fpS, fpV, fpW)
    num = master_sum[mask].astype(np.float64)
    cov = master_cov[mask].astype(np.float64)
    direct = RS[mask]
    sep = separate[mask]
    den = RW[mask]

    # The produced numerator equals the *separate* transport ...
    assert np.max(np.abs(num - sep)) < PATH_F32_TOL
    # ... and the produced denominator equals R(W).
    assert np.max(np.abs(cov - den)) < PATH_F32_TOL
    # ... but is *not* the direct numerator transport.
    assert np.max(np.abs(num - direct)) > WITNESS_MIN_MAX_DELTA


# ---------------------------------------------------------------------------
# Real production path D: reproject_and_coadd local fallback
# ---------------------------------------------------------------------------


def test_reproject_and_coadd_local_fallback_uses_separate_transport(monkeypatch):
    """Local accumulator returns output*coverage = R(V)*R(W), not R(V*W)."""
    # monkeypatch.setenv saves the prior value (or its absence) and restores it
    # exactly after the test, so a pre-existing REPROJECT_FORCE_LOCAL survives.
    monkeypatch.setenv("REPROJECT_FORCE_LOCAL", "1")

    V = make_v()
    W = make_w()
    S = V * W
    wcs_in = make_wcs()
    wcs_out = make_wcs(shift=SHIFT)

    RS, fpS = reproj(S, wcs_in, wcs_out)
    RV, fpV = reproj(V, wcs_in, wcs_out)
    RW, fpW = reproj(W, wcs_in, wcs_out)
    separate = RV * RW

    out, coverage = reproject_and_coadd(
        [(V, wcs_in)],
        output_projection=wcs_out,
        shape_out=SHAPE,
        input_weights=[W],
        reproject_function=reproject_interp,
        combine_function="mean",
        match_background=False,
    )

    mask = _common_valid_mask(fpS, fpV, fpW)
    # For a single weighted intermediate the local accumulator's numerator
    # (sum_image) equals output * coverage.
    numerator = (out * coverage)[mask].astype(np.float64)
    direct = RS[mask]
    sep = separate[mask]

    assert np.max(np.abs(numerator - sep)) < PATH_F32_TOL
    assert np.max(np.abs(numerator - direct)) > WITNESS_MIN_MAX_DELTA


# ---------------------------------------------------------------------------
# Supplementary: astropy reference branch has the same conceptual semantics
# (not required for the verdict; deterministic when reproject is installed)
# ---------------------------------------------------------------------------


def test_astropy_reproject_and_coadd_has_same_semantics():
    """The default astropy reference branch also implements R(V)*R(W)."""
    from reproject.mosaicking import reproject_and_coadd as astropy_coadd

    V = make_v()
    W = make_w()
    S = V * W
    wcs_in = make_wcs()
    wcs_out = make_wcs(shift=SHIFT)

    RS, fpS = reproj(S, wcs_in, wcs_out)
    RV, fpV = reproj(V, wcs_in, wcs_out)
    RW, fpW = reproj(W, wcs_in, wcs_out)
    separate = RV * RW

    out, coverage = astropy_coadd(
        [(V, wcs_in)],
        output_projection=wcs_out,
        shape_out=SHAPE,
        input_weights=[W],
        reproject_function=reproject_interp,
        combine_function="mean",
        match_background=False,
    )

    mask = _common_valid_mask(fpS, fpV, fpW)
    numerator = (np.asarray(out) * np.asarray(coverage))[mask].astype(np.float64)
    direct = RS[mask]
    sep = separate[mask]

    assert np.max(np.abs(numerator - sep)) < EXACT_TOL
    assert np.max(np.abs(numerator - direct)) > WITNESS_MIN_MAX_DELTA
