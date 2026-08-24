"""HSI Closure P1-FIX — post-fix normalization / batch-boundary contract.

This suite is the *post-correction* contract test for the plain-classic
SUM/WHT path.  It pins, with deterministic executable evidence, that:

* every aligned source observation is normalized independently against ONE
  immutable session reference, before any reduction (including singleton
  batches), for ``normalize_method`` in {``linear_fit``, ``sky_mean``};
* ``none`` remains a strict no-op (aligned observations pass through unchanged);
* the fixed-reference normalization makes the mean reduction decomposition
  invariant (ABC == AB+C == A+BC == CA+B and ABCD == AB+CD) for both
  ``linear_fit`` and ``sky_mean`` within float32 tolerance;
* RAM, tiled/HQ and ``use_memmap=True`` reducers all consume the *same*
  aligned + source-normalized sample arrays (never reloading raw source paths);
* the captured reference is an immutable float32 copy.

The historical batch/order dependence of the *pre-correction* code is preserved
below as explicit ``baseline-before-correction`` legacy simulations that call
the raw ``_normalize_images_*`` helpers with a batch-local index-0 reference
(the defect that P1-FIX removed), so the counterexamples remain executable
without depending on the corrected production path.
"""

import glob
import inspect
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

if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = []
    settings_mod = types.ModuleType("seestar.gui.settings")

    class DummySettingsManager:
        pass

    settings_mod.SettingsManager = DummySettingsManager
    gui_pkg.settings = settings_mod
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod

from seestar.queuep.queue_manager import SeestarQueuedStacker  # noqa: E402
from seestar.core.normalization import (  # noqa: E402
    _normalize_images_linear_fit,
    _normalize_images_sky_mean,
)
from seestar.core.stack_methods import _stack_median, _stack_mean  # noqa: E402

HEADER = fits.Header()

# ---------------------------------------------------------------------------
# Deterministic adversarial dataset
# ---------------------------------------------------------------------------
# A = base (64x64 float32, smooth ramp + seeded noise -> wide, non-degenerate
#        25/90 percentile spread).
# B = 1.5 * A + 40   (affine, gain != 1)
# C = 0.7 * A - 30   (affine, gain != 1)
# Bs = A + 40        (pure offset, for sky_mean offset-alignment proof)
# Cs = A - 30        (pure offset)
# D = 2.0 * A - 100  (affine, for the irregular non-singleton decomposition)


def _build_dataset(shape=(64, 64)):
    rng = np.random.default_rng(1234)
    H, W = shape
    ii = np.arange(H, dtype=np.float64)[:, None]
    jj = np.arange(W, dtype=np.float64)[None, :]
    ramp = 100.0 + 200.0 * (ii / (H - 1)) + 60.0 * (jj / (W - 1))
    A = (ramp + rng.normal(0.0, 5.0, size=shape)).astype(np.float32)
    return {
        "A": A,
        "B": (1.5 * A + 40.0).astype(np.float32),
        "C": (0.7 * A - 30.0).astype(np.float32),
        "Bs": (A + 40.0).astype(np.float32),
        "Cs": (A - 30.0).astype(np.float32),
        "D": (2.0 * A - 100.0).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Lightweight harness (mirrors tests/test_hierarchical_stacking_integrity.py)
# ---------------------------------------------------------------------------


def make_stack(
    mode="mean",
    norm="none",
    batch_size=10,
    max_hq_mem=1_000_000_000,
    ref=None,
    plain=True,
    non_plain_kind="reproject_between",
):
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.stacking_mode = mode
    o.normalize_method = norm
    o.weighting_method = "none"
    o.use_quality_weighting = False
    o.weight_by_snr = True
    o.weight_by_stars = False
    o.snr_exponent = 1.0
    o.stars_exponent = 0.5
    o.min_weight = 0.0
    o.apply_batch_feathering = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o.drizzle_active_session = False
    o.is_mosaic_run = False
    if not plain:
        if non_plain_kind == "mosaic":
            o.is_mosaic_run = True
        elif non_plain_kind == "drizzle":
            o.drizzle_active_session = True
        elif non_plain_kind == "reproject_between":
            o.reproject_between_batches = True
        elif non_plain_kind == "reproject_coadd":
            o.reproject_coadd_final = True
        else:
            raise ValueError(f"unknown non_plain_kind {non_plain_kind!r}")
    o.stack_kappa_low = 3.0
    o.stack_kappa_high = 3.0
    o.winsor_limits = (0.05, 0.05)
    o.stack_reject_algo = "none"
    o.max_hq_mem = max_hq_mem
    o.batch_size = batch_size
    o.settings = None
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.interbatch_norm_active = False
    o.max_stack_workers = 1
    o._current_batch_paths = []
    # P1-FIX: capture the immutable session reference exactly as the worker
    # does, so the post-fix contract (normalize every observation against it)
    # is exercised end-to-end.
    o._capture_normalization_reference(ref)
    return o


def fresh_item(img):
    """Build one batch item from a *fresh copy* of ``img``.

    The copy is mandatory: ``_normalize_images_sky_mean`` mutates non-reference
    inputs in place, so reusing the same array object across sub-batches would
    leak normalization state and falsify the decomposition experiment.
    """
    m = np.ones(img.shape[:2], dtype=bool)
    return (
        np.array(img, dtype=np.float32, copy=True),
        HEADER,
        {"snr": 1.0, "stars": 0.0},
        None,
        m,
    )


def _compose_vw(pairs):
    """Compose (V, W) batch outputs via the real HSI contract SUM(V*W)/SUM(W)."""
    num = None
    den = None
    for V, W in pairs:
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        if W.ndim == 2 and V.ndim == 3:
            W = W[..., None]
        num = V * W if num is None else num + V * W
        den = W if den is None else den + W
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def reduce_decomposition(stack, groups):
    """Stack each group independently and compose via sum(V*W)/sum(W)."""
    pairs = []
    for g in groups:
        items = [fresh_item(x) for x in g]
        V, _hdr, W = stack._stack_batch(items, 1, 1)
        pairs.append((V, W))
    return _compose_vw(pairs)


def _diff(x, y):
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.abs(d).max()), float(d.mean())


def _close_memmap(arr):
    """Release a numpy.lib.format.open_memmap handle (its underlying mmap)."""
    mm = getattr(arr, "_mmap", None)
    if mm is not None:
        try:
            mm.close()
        except Exception:
            pass


def _release_memmap_file(arr):
    """Fully release a memmap and remove its backing file (as production does)."""
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


def _materialize(V):
    """Copy a (possibly memmap-backed) stack output to a plain float64 array."""
    arr = V[..., 0] if V.ndim == 3 else V
    return np.array(arr, dtype=np.float64, copy=True)


# Tolerances derived from the observed float32 round-off of the current helpers.
NORM_TOL = 2e-3          # linear_fit / sky_mean recovery tolerance
INVARIANT_TOL = 1e-3     # "none" decomposition invariance tolerance
SKY_INVARIANT_TOL = 1e-2  # sky_mean decomposition invariance (offset-only align)


# ===========================================================================
# A. Direct helper proofs (characterize the normalization helpers themselves)
# ===========================================================================


def test_linear_fit_normalizes_affine_transforms_to_reference():
    D = _build_dataset()
    A, B, C = D["A"], D["B"], D["C"]
    out = _normalize_images_linear_fit([A, B, C], 0)
    # Reference is returned unchanged; the affine copies are mapped back to it.
    assert out[0] is not None and np.allclose(out[0], A, atol=NORM_TOL)
    for o in (out[1], out[2]):
        assert np.allclose(o, A, atol=NORM_TOL), np.abs(o - A).max()


def test_sky_mean_aligns_sky_offset_but_not_gain():
    D = _build_dataset()
    A, B, C = D["A"], D["B"], D["C"]
    out = _normalize_images_sky_mean([A, B, C], 0)
    sky_A = float(np.percentile(A, 25.0))
    # The sky (25th-percentile) of every normalized image equals the reference
    # sky: offsets are aligned.  The affine gain is *not* corrected.
    for o in (out[1], out[2]):
        assert np.isclose(float(np.percentile(o, 25.0)), sky_A, atol=NORM_TOL)
        assert not np.allclose(o, A, atol=NORM_TOL)  # gain mismatch remains


def test_sky_mean_recovers_pure_offset_transforms():
    D = _build_dataset()
    A, Bs, Cs = D["A"], D["Bs"], D["Cs"]
    out = _normalize_images_sky_mean([A, Bs, Cs], 0)
    for o in (out[1], out[2]):
        assert np.allclose(o, A, atol=NORM_TOL), np.abs(o - A).max()


# ===========================================================================
# B. Baseline-before-correction legacy simulations (batch-local reference)
# ===========================================================================
# These do NOT call the production ``_stack_batch`` path; they reproduce the
# pre-fix batch-local index-0 normalization directly with the raw helpers so
# the historical counterexamples stay executable as documentation of the
# defect that P1-FIX removed.


def test_baseline_legacy_batch_local_linear_fit_was_decomposition_dependent():
    D = _build_dataset()
    A, B, C = D["A"], D["B"], D["C"]

    def legacy_reduce(groups):
        pairs = []
        for g in groups:
            norm = _normalize_images_linear_fit(
                [np.array(x, dtype=np.float32, copy=True) for x in g], 0
            )
            V = np.mean(np.stack(norm, axis=0), axis=0).astype(np.float32)
            W = np.ones(V.shape, dtype=np.float32) * len(g)
            pairs.append((V, W))
        return _compose_vw(pairs)

    global_ = legacy_reduce([[A, B, C]])
    # Batch-local reference: [A,B] normalizes to A, singleton [C] is unnormalized.
    split = legacy_reduce([[A, B], [C]])
    maxd, _ = _diff(split, global_)
    assert maxd > 1.0  # the pre-fix defect: decomposition dependence


def test_baseline_legacy_batch_local_sky_mean_was_decomposition_dependent():
    D = _build_dataset()
    A, Bs, Cs = D["A"], D["Bs"], D["Cs"]

    def legacy_reduce(groups):
        pairs = []
        for g in groups:
            norm = _normalize_images_sky_mean(
                [np.array(x, dtype=np.float32, copy=True) for x in g], 0
            )
            V = np.mean(np.stack(norm, axis=0), axis=0).astype(np.float32)
            W = np.ones(V.shape, dtype=np.float32) * len(g)
            pairs.append((V, W))
        return _compose_vw(pairs)

    global_ = legacy_reduce([[A, Bs, Cs]])
    split = legacy_reduce([[A, Bs], [Cs]])
    maxd, _ = _diff(split, global_)
    assert maxd > 1.0  # pre-fix defect: singleton bypass skips offset alignment


# ===========================================================================
# C. none + mean: decomposition invariance (unchanged arithmetic contract)
# ===========================================================================


def test_none_mean_is_decomposition_invariant():
    D = _build_dataset()
    A, B, C = D["A"], D["B"], D["C"]
    stack = make_stack("mean", norm="none")

    global_ = reduce_decomposition(stack, [[A, B, C]])
    for groups in (
        [[A, B], [C]],
        [[A], [B, C]],
        [[C, A], [B]],
    ):
        r = reduce_decomposition(stack, groups)
        maxd, meand = _diff(r, global_)
        assert maxd < INVARIANT_TOL, (groups, maxd, meand)


# ===========================================================================
# D. linear_fit: fixed-reference decomposition invariance (post-fix contract)
# ===========================================================================


def _abc_decompositions():
    D = _build_dataset()
    A, B, C = D["A"], D["B"], D["C"]
    return A, {
        "ABC": [[A, B, C]],
        "AB_C": [[A, B], [C]],
        "A_BC": [[A], [B, C]],
        "CA_B": [[C, A], [B]],
    }


def _abcd_decompositions():
    D = _build_dataset()
    A, B, C, Darr = D["A"], D["B"], D["C"], D["D"]
    return A, {
        "ABCD": [[A, B, C, Darr]],
        "AB_CD": [[A, B], [C, Darr]],
    }


def test_linear_fit_mean_is_decomposition_invariant():
    A, decomps = _abc_decompositions()
    stack = make_stack("mean", norm="linear_fit", ref=A)

    global_ = reduce_decomposition(stack, decomps["ABC"])
    # With a fixed session reference A, every affine observation resolves to A.
    assert np.allclose(global_, A, atol=NORM_TOL), np.abs(global_ - A).max()

    for name, groups in decomps.items():
        r = reduce_decomposition(stack, groups)
        maxd, meand = _diff(r, global_)
        # Every legitimate decomposition yields the same SUM/WHT result (A).
        assert maxd < NORM_TOL, (name, maxd, meand)


def test_linear_fit_irregular_decomposition_is_invariant():
    A, decomps = _abcd_decompositions()
    stack = make_stack("mean", norm="linear_fit", ref=A)

    abcd = reduce_decomposition(stack, decomps["ABCD"])
    ab_cd = reduce_decomposition(stack, decomps["AB_CD"])
    assert np.allclose(abcd, A, atol=NORM_TOL), np.abs(abcd - A).max()
    maxd, meand = _diff(ab_cd, abcd)
    assert maxd < NORM_TOL, (maxd, meand)


# ===========================================================================
# E. sky_mean: fixed-reference decomposition invariance (post-fix contract)
# ===========================================================================
# sky_mean aligns only the sky offset (not the affine gain), so the resolved
# result is NOT the reference A itself; but because every observation is now
# normalized against the *same* fixed reference, the SUM/WHT mean is
# decomposition invariant (deterministic per observation).


def test_sky_mean_mean_is_decomposition_invariant():
    A, decomps = _abc_decompositions()
    stack = make_stack("mean", norm="sky_mean", ref=A)

    global_ = reduce_decomposition(stack, decomps["ABC"])
    for name, groups in decomps.items():
        r = reduce_decomposition(stack, groups)
        maxd, meand = _diff(r, global_)
        assert maxd < SKY_INVARIANT_TOL, (name, maxd, meand)


def test_sky_mean_irregular_decomposition_is_invariant():
    A, decomps = _abcd_decompositions()
    stack = make_stack("mean", norm="sky_mean", ref=A)

    abcd = reduce_decomposition(stack, decomps["ABCD"])
    ab_cd = reduce_decomposition(stack, decomps["AB_CD"])
    maxd, meand = _diff(ab_cd, abcd)
    assert maxd < SKY_INVARIANT_TOL, (maxd, meand)


def test_sky_mean_offset_decomposition_is_invariant():
    D = _build_dataset()
    A, Bs, Cs = D["A"], D["Bs"], D["Cs"]
    stack = make_stack("mean", norm="sky_mean", ref=A)

    global_ = reduce_decomposition(stack, [[A, Bs, Cs]])
    assert np.allclose(global_, A, atol=NORM_TOL), np.abs(global_ - A).max()
    for groups in ([[A, Bs], [Cs]], [[A], [Bs, Cs]], [[Cs, A], [Bs]]):
        r = reduce_decomposition(stack, groups)
        maxd, meand = _diff(r, global_)
        assert maxd < NORM_TOL, (groups, maxd, meand)


# ===========================================================================
# F. Singleton source normalization (post-fix contract)
# ===========================================================================


def test_singleton_batch_is_normalized_linear_fit():
    D = _build_dataset()
    A, B = D["A"], D["B"]  # B = 1.5*A + 40 (affine)

    # A single-image batch is now normalized against the fixed reference A
    # *before* the early return, so B is mapped back to A.
    stack = make_stack("mean", norm="linear_fit", ref=A)
    V, hdr, W = stack._stack_batch([fresh_item(B)], 1, 1)
    assert hdr.get("STK_NOTE") == "single image"
    assert np.allclose(V, A, atol=NORM_TOL), np.abs(V - A).max()
    assert not np.allclose(V, B, atol=NORM_TOL)  # no longer returned verbatim


def test_singleton_batch_is_normalized_sky_mean():
    D = _build_dataset()
    A, B = D["A"], D["B"]
    stack = make_stack("mean", norm="sky_mean", ref=A)
    V, hdr, W = stack._stack_batch([fresh_item(B)], 1, 1)
    assert hdr.get("STK_NOTE") == "single image"
    sky_A = float(np.percentile(A, 25.0))
    assert np.isclose(float(np.percentile(V, 25.0)), sky_A, atol=NORM_TOL)


def test_singleton_batch_none_is_unchanged():
    D = _build_dataset()
    B = D["B"]
    stack = make_stack("mean", norm="none")
    V, hdr, W = stack._stack_batch([fresh_item(B)], 1, 1)
    assert hdr.get("STK_NOTE") == "single image"
    assert np.allclose(V, B, atol=1e-6)  # none is a strict no-op


# ===========================================================================
# G. No repeat: continuous SUM/W composition does not re-normalize
# ===========================================================================


def test_continuous_sumw_does_not_renormalize_batch_outputs():
    H = W = 4
    v1 = np.full((H, W, 3), 10.0, dtype=np.float32)
    v2 = np.full((H, W, 3), 30.0, dtype=np.float32)
    w = np.ones((H, W), dtype=np.float32)

    stack = make_stack("mean", norm="linear_fit")
    stack.memmap_shape = (H, W, 3)
    stack.memmap_dtype_sum = np.float32
    stack.memmap_dtype_wht = np.float32
    stack.cumulative_sum_memmap = np.zeros((H, W, 3), dtype=np.float32)
    stack.cumulative_wht_memmap = np.zeros((H, W, 3), dtype=np.float32)
    stack.stacked_batches_count = 0
    stack.images_in_cumulative_stack = 0
    stack.total_exposure_seconds = 0.0
    stack.failed_stack_count = 0
    stack.current_stack_header = None
    stack.correct_hot_pixels = False
    stack.use_quality_weighting = False
    stack._checkpointing_enabled = False
    stack._resume_completed_sources = []
    stack.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None, debug=lambda *a, **k: None
    )

    hdr = fits.Header()
    hdr["NIMAGES"] = 1
    hdr["TOTEXP"] = 1.0

    stack._combine_batch_result(v1, hdr, w)
    stack._combine_batch_result(v2, hdr, w)

    # Both batch outputs are accumulated verbatim: no normalization of the
    # second batch against the first occurs at the SUM/W level.
    assert np.allclose(stack.cumulative_sum_memmap[0, 0], [40.0, 40.0, 40.0], atol=1e-4)
    assert np.allclose(stack.cumulative_wht_memmap[0, 0], [2.0, 2.0, 2.0], atol=1e-4)
    with np.errstate(divide="ignore", invalid="ignore"):
        final = stack.cumulative_sum_memmap / stack.cumulative_wht_memmap
    assert np.allclose(final[0, 0], [20.0, 20.0, 20.0], atol=1e-4)


# ===========================================================================
# H. Backend sample-consumption parity (RAM / tiled-HQ / use_memmap)
# ===========================================================================
# Use median mode so the tiled/HQ path (which only exists for non-mean modes)
# is exercised.  With B = A + 200 (pure offset) linear_fit maps B -> A, so:
#   * normalized median of [A, B]  == A
#   * raw       median of [A, B]  == A + 100
# which makes "normalized vs raw samples consumed" unambiguous.


def _median_probe_pair():
    A = _build_dataset()["A"]
    B = (A + 200.0).astype(np.float32)
    return A, B


def test_backend_ram_consumes_normalized_samples():
    A, B = _median_probe_pair()
    # RAM median path (use_tile_mode False): image_data_list is normalized.
    stack = make_stack("median", norm="linear_fit", max_hq_mem=1_000_000_000, ref=A)
    V, _hdr, W = stack._stack_batch([fresh_item(A), fresh_item(B)], 1, 1)
    assert np.allclose(V, A, atol=NORM_TOL), np.abs(V - A).max()


def test_backend_tiled_in_memory_consumes_normalized_samples(tmp_path, monkeypatch):
    A, B = _median_probe_pair()
    # batch_size == 1 -> tile_inputs = image_data_list (normalized), even though
    # tile/memmap mode is forced.  Result == A (normalized median).
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    stack = make_stack("median", norm="linear_fit", batch_size=1, max_hq_mem=1, ref=A)
    V, _hdr, W = stack._stack_batch([fresh_item(A), fresh_item(B)], 1, 1)
    V2 = _materialize(V)
    _release_memmap_file(V)
    assert np.allclose(V2, A, atol=NORM_TOL), np.abs(V2 - A).max()
    assert not glob.glob(os.path.join(str(tmp_path), "hq_batch*.dat"))


def test_backend_tiled_batch_gt1_consumes_normalized_samples(tmp_path, monkeypatch):
    A, B = _median_probe_pair()
    # batch_size != 1 must STILL consume the in-memory aligned + normalized
    # arrays (image_data_list), never reload raw _current_batch_paths.
    fa = str(tmp_path / "a.fits")
    fb = str(tmp_path / "b.fits")
    fits.PrimaryHDU(data=A.astype(np.float32)).writeto(fa, overwrite=True)
    fits.PrimaryHDU(data=B.astype(np.float32)).writeto(fb, overwrite=True)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    stack = make_stack("median", norm="linear_fit", batch_size=10, max_hq_mem=1, ref=A)
    stack._current_batch_paths = [fa, fb]
    V, _hdr, W = stack._stack_batch([fresh_item(A), fresh_item(B)], 1, 1)

    V2 = _materialize(V)
    # The tiled path now stacks the *normalized* arrays: median == A.
    assert np.allclose(V2, A, atol=NORM_TOL), np.abs(V2 - A).max()
    # It is NOT the raw median (which would be ~A + 100).
    assert not np.allclose(V2, ((A + B) / 2.0), atol=1.0)


def test_backend_memmap_via_stack_batch_consumes_normalized(tmp_path, monkeypatch):
    A, B = _median_probe_pair()
    # batch_size == 1 forces use_memmap=True through the real _stack_batch path:
    # the memmap accumulator must consume the normalized arrays, and no
    # hq_batch*.dat file may remain after the handle is released.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    stack = make_stack("median", norm="linear_fit", batch_size=1, max_hq_mem=1, ref=A)
    V, _hdr, W = stack._stack_batch([fresh_item(A), fresh_item(B)], 1, 1)
    V2 = _materialize(V)
    _release_memmap_file(V)
    assert np.allclose(V2, A, atol=NORM_TOL), np.abs(V2 - A).max()
    assert not glob.glob(os.path.join(str(tmp_path), "hq_batch*.dat"))


def test_backend_memmap_direct_in_memory_inputs(tmp_path, monkeypatch):
    A, B = _median_probe_pair()
    # Explicit use_memmap=True with *in-memory normalized* arrays: the memmap
    # accumulator stores whatever inputs are passed (normalized here).
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    norm = _normalize_images_linear_fit([A, B], 0)
    masks = [np.ones(A.shape, dtype=bool), np.ones(B.shape, dtype=bool)]
    qw = np.ones(2, dtype=np.float32)
    stack = make_stack("median", norm="linear_fit", max_hq_mem=1, ref=A)
    V, W = stack._combine_hq_by_tiles(
        norm, masks, 3.0, (0.05, 0.05),
        masks_list=masks, quality_weights=qw, use_memmap=True, tile_h=8,
        batch_id=901,
    )
    V2 = _materialize(V)
    _release_memmap_file(V)
    assert np.allclose(V2, A, atol=NORM_TOL), np.abs(V2 - A).max()
    assert not glob.glob(os.path.join(str(tmp_path), "hq_batch*.dat"))


# ===========================================================================
# I. Reference immutability + capture helper
# ===========================================================================


def test_normalization_reference_is_immutable():
    D = _build_dataset()
    A, B = D["A"], D["B"]
    stack = make_stack("mean", norm="linear_fit", ref=A)
    ref_before = np.array(stack._norm_reference, dtype=np.float32, copy=True)

    # Normalizing sources must not mutate the stored reference.
    _ = stack._normalize_sources_against_reference(
        [np.array(B, dtype=np.float32, copy=True)]
    )
    assert np.array_equal(stack._norm_reference, ref_before)

    # Even a later in-place mutation of the mutable alignment anchor that the
    # reference was copied from must not leak into the stored reference.
    anchor = np.array(A, dtype=np.float32, copy=True)
    stack._capture_normalization_reference(anchor)
    anchor += 999.0
    assert not np.allclose(stack._norm_reference, anchor, atol=1.0)
    assert np.array_equal(stack._norm_reference, np.array(A, dtype=np.float32))


def test_capture_helper_copies_and_clears():
    A = _build_dataset()["A"]
    stack = make_stack("mean", norm="none")
    assert stack._norm_reference is None

    stack._capture_normalization_reference(A)
    assert stack._norm_reference is not None
    assert stack._norm_reference.dtype == np.float32
    assert stack._norm_reference is not A  # a copy, not an alias

    stack._capture_normalization_reference(None)
    assert stack._norm_reference is None


# ===========================================================================
# J. Corrective iteration C1: plain-classic scoping, non-plain preservation,
#    fail-closed reference, and session-scoped reference release
# ===========================================================================


def test_non_plain_linear_fit_uses_batch_local_index0_not_reference():
    """Non-plain multi-image batches keep the historical batch-local index-0
    source normalization and never consult the session ``_norm_reference``."""
    D = _build_dataset()
    A, B = D["A"], D["B"]  # B = 1.5*A + 40 (affine)

    stack = make_stack(
        "mean", norm="linear_fit", plain=False, non_plain_kind="reproject_between"
    )
    # Plant a sentinel reference: if the non-plain path consulted it, the
    # reduction would collapse toward zeros instead of batch-local A.
    stack._norm_reference = np.zeros(A.shape, dtype=np.float32)

    V, hdr, W = stack._stack_batch([fresh_item(A), fresh_item(B)], 1, 1)

    # Batch-local index-0: B is mapped onto A, so the mean resolves to A.
    assert np.allclose(V, A, atol=NORM_TOL), np.abs(V - A).max()
    # The planted reference was NOT used (otherwise V would be ~zeros).
    assert not np.allclose(V, stack._norm_reference, atol=1.0)


def test_non_plain_singleton_bypasses_source_normalization():
    """A non-plain singleton mean batch returns the raw aligned image verbatim,
    bypassing source normalization exactly as historically."""
    D = _build_dataset()
    B = D["B"]
    for kind in ("mosaic", "drizzle", "reproject_between", "reproject_coadd"):
        stack = make_stack("mean", norm="linear_fit", plain=False, non_plain_kind=kind)
        assert stack._norm_reference is None
        V, hdr, W = stack._stack_batch([fresh_item(B)], 1, 1)
        assert hdr.get("STK_NOTE") == "single image"
        assert np.allclose(V, B, atol=1e-6)  # returned verbatim, no normalization


def test_non_plain_capture_gate_never_retains_reference():
    """The worker capture seam is gated behind ``_should_capture_norm_reference``,
    so non-plain sessions never create or retain a full-frame ``_norm_reference``."""
    src = inspect.getsource(SeestarQueuedStacker._worker)
    assert "self._should_capture_norm_reference()" in src
    assert "self._capture_normalization_reference(" in src

    for kind in ("mosaic", "drizzle", "reproject_between", "reproject_coadd"):
        s = make_stack("mean", norm="linear_fit", plain=False, non_plain_kind=kind)
        assert s._is_plain_classic() is False
        assert s._should_capture_norm_reference() is False

    # Plain classic captures only for linear_fit / sky_mean; never for none.
    assert make_stack("mean", norm="linear_fit")._should_capture_norm_reference() is True
    assert make_stack("mean", norm="sky_mean")._should_capture_norm_reference() is True
    assert make_stack("mean", norm="none")._should_capture_norm_reference() is False


def test_plain_classic_missing_reference_linear_fit_raises():
    D = _build_dataset()
    A, B = D["A"], D["B"]
    stack = make_stack("mean", norm="linear_fit", ref=None)
    assert stack._norm_reference is None
    with pytest.raises(RuntimeError, match="session reference"):
        stack._stack_batch([fresh_item(A), fresh_item(B)], 1, 1)


def test_plain_classic_missing_reference_sky_mean_raises():
    D = _build_dataset()
    B = D["B"]
    stack = make_stack("mean", norm="sky_mean", ref=None)
    assert stack._norm_reference is None
    # Fails before the singleton fast path and before any reduction.
    with pytest.raises(RuntimeError, match="session reference"):
        stack._stack_batch([fresh_item(B)], 1, 1)


def test_plain_classic_missing_reference_none_is_still_noop():
    D = _build_dataset()
    B = D["B"]
    stack = make_stack("mean", norm="none", ref=None)
    assert stack._norm_reference is None
    V, hdr, W = stack._stack_batch([fresh_item(B)], 1, 1)
    assert hdr.get("STK_NOTE") == "single image"
    assert np.allclose(V, B, atol=1e-6)  # none never needs a reference


def test_worker_finally_releases_norm_reference():
    """The worker finally block releases the full-frame reference on every exit,
    and the ``_release_norm_reference`` seam actually clears it."""
    src = inspect.getsource(SeestarQueuedStacker._worker)
    idx_release = src.index("self._release_norm_reference()")
    # The release lives in the worker's final finally block, immediately before
    # the final gc.collect().
    assert src.rindex("finally:") < idx_release
    idx_gc = src.index("gc.collect()", idx_release)
    assert idx_gc > idx_release
    assert src[idx_release + len("self._release_norm_reference()"):idx_gc].strip() == ""

    # Behavioral: the seam clears a captured copy.
    A = _build_dataset()["A"]
    stack = make_stack("mean", norm="linear_fit", ref=A)
    assert stack._norm_reference is not None
    stack._release_norm_reference()
    assert stack._norm_reference is None
