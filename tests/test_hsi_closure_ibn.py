"""HSI Closure P1-FIX — production inter-batch MASTER_REF + BG2D layer gating.

This suite is the *post-correction* contract for the automatic batch-output
inter-batch normalization (IBN) layer.  P1 established that this second layer
is auto-started unconditionally by ``start_processing()`` and applied to every
multi-image mini-stack, which broke the plain-classic ``SUM/WHT`` exact
composition (it made even ``normalize_method=none`` decomposition-dependent).

The correction gates the layer so that:

* a **plain classic** session (no mosaic, no drizzle, no inter-batch/final
  reprojection) never starts or applies IBN — its ``_stack_batch`` output is
  not BG2D-subtracted, gain-normalized, or radially feathered;
* a **non-plain** session (mosaic / drizzle / reproject) still auto-starts and
  applies the existing IBN layer unchanged.

The IBN layer itself is *not* redesigned: the branch selection, first-batch
master identity, 10,000-pixel gain gate, and radial feathering behaviour are
preserved and re-characterized below for the non-plain route only.
"""

import inspect
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

HEADER = fits.Header()


# ---------------------------------------------------------------------------
# Deterministic adversarial dataset (large, nonuniform background & coverage)
# ---------------------------------------------------------------------------
# base = smooth spatial gradient + central Gaussian "source" + seeded noise
# (a 2-D smooth background so BG2D actually removes a non-trivial model, plus
#  a bright compact source so the median-ratio gain operates on a real signal).
# A = base
# B = 1.5 * base + 40   (affine gain != 1)
# C = 0.7 * base - 30   (affine gain != 1)
# D = 2.0 * base - 100  (affine gain != 1)
# Coverage: A/D full frame; B left 2/3; C right 2/3 (non-uniform overlap).


def _build_dataset(shape=(128, 128)):
    rng = np.random.default_rng(20260824)
    H, W = shape
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]
    bg = 80.0 + 120.0 * (yy / (H - 1)) + 40.0 * (xx / (W - 1))
    cy, cx = H / 2.0, W / 2.0
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    src = 400.0 * np.exp(-dist2 / (2.0 * 18.0 ** 2))
    base = (bg + src + rng.normal(0.0, 6.0, size=shape)).astype(np.float32)
    return {
        "A": base,
        "B": (1.5 * base + 40.0).astype(np.float32),
        "C": (0.7 * base - 30.0).astype(np.float32),
        "D": (2.0 * base - 100.0).astype(np.float32),
    }


def _mask_full(shape):
    return np.ones(shape, dtype=bool)


def _mask_left(shape):
    H, W = shape
    m = np.zeros(shape, dtype=bool)
    m[:, : int(W * 2 / 3)] = True
    return m


def _mask_right(shape):
    H, W = shape
    m = np.zeros(shape, dtype=bool)
    m[:, int(W * 1 / 3):] = True
    return m


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

_NON_PLAIN_KINDS = ("mosaic", "drizzle", "reproject_between", "reproject_coadd")


def _apply_ibn_gate(o):
    """Mirror the P1-FIX gating placed in ``start_processing`` after all mode
    flags are resolved (this is the exact production condition)."""
    o._final_combine_ibn_started = False
    o._final_combine_ibn_master_set = False
    o._final_combine_ibn_master_batch_idx = None
    if not o._is_plain_classic():
        o._interbatch_start_session()
    else:
        o.interbatch_norm_active = False


def make_stack(mode="mean", norm="none", batch_size=10, plain=True,
               non_plain_kind="reproject_between"):
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
    o.is_mosaic_run = False
    o.drizzle_active_session = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    if not plain:
        if non_plain_kind == "mosaic":
            o.is_mosaic_run = True
        elif non_plain_kind == "drizzle":
            o.drizzle_active_session = True
        elif non_plain_kind == "reproject_between":
            o.reproject_between_batches = True
        elif non_plain_kind == "reproject_coadd":
            o.reproject_coadd_final = True
    o.stack_kappa_low = 3.0
    o.stack_kappa_high = 3.0
    o.winsor_limits = (0.05, 0.05)
    o.stack_reject_algo = "none"
    o.max_hq_mem = 1_000_000_000
    o.batch_size = batch_size
    o.settings = None
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.max_stack_workers = 1
    o._current_batch_paths = []
    o.stack_final_combine = "mean"
    o.use_classic_batches_for_final_coadd = batch_size == 1
    # Deterministic quality metric (avoids spawning the ProcessPoolExecutor used
    # by _calculate_quality_metrics).  The MASTER_REF selection does not depend on
    # the score when _ibn_master_min == 1 (the production default), so this is a
    # pure performance/determinism patch, not a scientific-operation patch.
    o._calculate_quality_metrics = lambda image_data: {"snr": 1.0, "stars": 0.0}
    _apply_ibn_gate(o)
    return o


def fresh_item(img, mask):
    return (
        np.array(img, dtype=np.float32, copy=True),
        HEADER,
        {"snr": 1.0, "stars": 0.0},
        None,
        np.array(mask, dtype=bool, copy=True),
    )


# ===========================================================================
# A. Plain-classic vs non-plain gating (post-fix contract)
# ===========================================================================


def test_is_plain_classic_classification():
    o = make_stack("mean", plain=True)
    assert o._is_plain_classic() is True
    for kind in _NON_PLAIN_KINDS:
        n = make_stack("mean", plain=False, non_plain_kind=kind)
        assert n._is_plain_classic() is False, kind


def test_plain_classic_gate_disables_ibn():
    o = make_stack("mean", plain=True)
    assert o.interbatch_norm_active is False
    assert getattr(o, "_ibn_master_ready", False) is False


def test_non_plain_gate_enables_ibn():
    for kind in _NON_PLAIN_KINDS:
        o = make_stack("mean", plain=False, non_plain_kind=kind)
        assert o.interbatch_norm_active is True, kind
        assert o._ibn_master_min == 1
        assert o._ibn_min_overlap == 10000


def test_start_processing_gates_ibn_after_flags_resolved():
    src = inspect.getsource(SeestarQueuedStacker.start_processing)
    # The gate uses the reliable _is_plain_classic() predicate.
    assert "if not self._is_plain_classic():" in src
    assert "self._interbatch_start_session()" in src
    # The unconditional early call (immediately after the ``perform_cleanup``
    # configuration) is gone: the gate now sits after that config block.
    idx_perf = src.index("self.perform_cleanup = bool(perform_cleanup)")
    idx_gate = src.index("if not self._is_plain_classic():")
    assert idx_gate > idx_perf


# ===========================================================================
# B. Plain classic applies no IBN and no radial WHT alteration
# ===========================================================================


def test_plain_classic_stack_batch_applies_no_ibn():
    D = _build_dataset()
    A, B = D["A"], D["B"]
    full = _mask_full(A.shape)

    s = make_stack("mean", norm="none", batch_size=10, plain=True)
    V, hdr, W = s._stack_batch([fresh_item(A, full), fresh_item(B, full)], 1, 2)

    # No BG2D, no gain, no IBN bookkeeping.
    assert getattr(s, "interbatch_norm_active", False) is False
    assert getattr(s, "_ibn_applied", 0) == 0
    assert getattr(s, "_ibn_bg_applied", 0) == 0
    assert getattr(s, "_ibn_batches_seen", 0) == 0

    # No radial feathering from IBN: a uniform full-frame mask yields a uniform
    # coverage map (corners == centres == 2.0), untouched by IBN.
    W2 = W[..., 0] if W.ndim == 3 else W
    assert float(W2[0, 0]) == pytest.approx(2.0, abs=1e-3)
    assert float(W2[64, 64]) == pytest.approx(2.0, abs=1e-3)


def test_non_plain_stack_batch_still_feathers_via_ibn():
    D = _build_dataset()
    A, B = D["A"], D["B"]
    full = _mask_full(A.shape)

    s = make_stack("mean", norm="none", batch_size=10, plain=False,
                   non_plain_kind="reproject_between")
    s._ibn_min_overlap = 1
    V, hdr, W = s._stack_batch([fresh_item(A, full), fresh_item(B, full)], 1, 2)

    # IBN is active and its BG2D step ran (first batch: master not ready yet,
    # so it is clipped + returned after feathering).
    assert s.interbatch_norm_active is True
    assert s._ibn_bg_applied >= 1
    # The IBN radial feather altered the coverage map: centre unfeathered,
    # corner downweighted to the radial floor.
    W2 = W[..., 0] if W.ndim == 3 else W
    assert float(W2[64, 64]) == pytest.approx(2.0, abs=1e-3)
    assert float(W2[0, 0]) < 1.0


# ===========================================================================
# C. IBN layer preserved (non-plain characterization, unchanged behaviour)
# ===========================================================================


def test_interbatch_start_session_always_enables_and_sets_production_defaults():
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o._interbatch_start_session()
    assert o.interbatch_norm_active is True
    assert o._ibn_master_min == 1
    assert o._ibn_candidate_limit == 8
    assert o._ibn_min_overlap == 10000
    assert o._ibn_use_percentile_ratio is True


def test_branch_selection_batch_size_gt1():
    s = make_stack("mean", norm="none", batch_size=10, plain=True)
    assert s.use_classic_batches_for_final_coadd is False
    assert s._should_use_final_combine_ibn() is False


def test_branch_selection_batch_size_eq1():
    s = make_stack("mean", norm="none", batch_size=1, plain=True)
    assert s.use_classic_batches_for_final_coadd is True
    assert s._should_use_final_combine_ibn() is True


def test_branch_selection_reproject_disables_final_combine_ibn():
    s = make_stack("mean", norm="none", batch_size=1, plain=True)
    s.reproject_between_batches = True
    assert s._should_use_final_combine_ibn() is False
    s.reproject_between_batches = False
    s.reproject_coadd_final = True
    assert s._should_use_final_combine_ibn() is False


def test_single_image_mean_batch_bypasses_ibn():
    D = _build_dataset()
    B = D["B"]
    full = _mask_full(B.shape)
    s = make_stack("mean", norm="linear_fit", batch_size=10, plain=False)
    V, hdr, W = s._stack_batch([fresh_item(B, full)], 1, 1)
    assert hdr.get("STK_NOTE") == "single image"
    assert np.allclose(W, 1.0, atol=1e-6)  # coverage untouched (no feather)
    assert getattr(s, "_ibn_applied", 0) == 0
    assert getattr(s, "_ibn_bg_applied", 0) == 0


def test_master_reference_is_the_first_batch():
    D = _build_dataset()
    A, B = D["A"], D["B"]
    full = _mask_full(A.shape)
    s = make_stack("mean", norm="none", batch_size=10, plain=False)
    s._ibn_min_overlap = 1
    V1, _h, _w = s._stack_batch([fresh_item(A, full), fresh_item(B, full)], 1, 2)
    assert s._ibn_master_ready is True
    assert float(np.isclose(np.nanmedian(s._ibn_ref_image), np.nanmedian(V1), atol=1e-3))
    assert float(s._ibn_last_scale) == 1.0


def test_gain_layer_skipped_below_min_overlap():
    small = _build_dataset((32, 32))["A"]  # 1024 px < 10000
    B2 = (1.5 * small + 40.0).astype(np.float32)
    f2 = _mask_full(small.shape)

    s = make_stack("mean", norm="none", batch_size=10, plain=False)
    bn = 0
    for g in ([(small, f2), (B2, f2)], [(B2, f2), (small, f2)]):
        bn += 1
        s._stack_batch([fresh_item(i, m) for (i, m) in g], bn, 2)
    assert s._ibn_applied == 0
    assert s._ibn_last_scale is None

    # Lowered threshold -> gain active on the same images.
    s2 = make_stack("mean", norm="none", batch_size=10, plain=False)
    s2._ibn_min_overlap = 1
    bn = 0
    for g in ([(small, f2), (B2, f2)], [(B2, f2), (small, f2)]):
        bn += 1
        s2._stack_batch([fresh_item(i, m) for (i, m) in g], bn, 2)
    assert s2._ibn_applied == 2
    assert np.isfinite(s2._ibn_last_scale)


def test_ibn_feathers_coverage_weights_radially():
    D = _build_dataset()
    A, B = D["A"], D["B"]
    full = _mask_full(A.shape)
    s = make_stack("mean", norm="none", batch_size=10, plain=False)
    s._ibn_min_overlap = 1
    bn = 0
    centers = []
    corners = []
    for g in ([(A, full), (B, full)], [(B, full), (A, full)]):
        bn += 1
        _V, _h, W = s._stack_batch([fresh_item(i, m) for (i, m) in g], bn, 2)
        W2 = W[..., 0] if W.ndim == 3 else W
        centers.append(float(W2[64, 64]))
        corners.append(float(W2[0, 0]))
    assert all(abs(c - 2.0) < 1e-3 for c in centers)
    assert all(abs(c - 0.2) < 1e-3 for c in corners)
    assert len(s._ibn_feather_cache) == 1
