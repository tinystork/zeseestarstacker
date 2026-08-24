"""HSI Closure P5-FIX — ``min_weight`` relative-floor correction acceptance.

This suite documents the P5 corrective contract and asserts the *corrected*
behaviour (it replaces the previous P5-AUDIT ``audit-of-defect`` suite):

* The raw quality metric ``q(scores)`` is factored into one shared formula used
  for both the immutable session reference (``q_ref``) and every source.
* A source weight is ``max(q(source) / q_ref, min_weight)``: ``min_weight`` is a
  *relative* floor expressed as a fraction of the reference quality scale, and
  the reference itself maps to weight ``1.0``.
* ``q_ref`` is pinned exactly once from the actual session reference (never from
  the first item of an arbitrary worker batch, a cumulative stack, or a changing
  reprojection reference), persisted in the resume manifest, and restored
  verbatim (never recomputed) on resume.
* The configuration seams (Qt surface, settings validation, run-config
  transport, backend clamp) agree on ``[0.01, 1.0]``.

The pre-P1 oracle (reconstructed verbatim from the lines removed by the accepted
P1 correction) is retained *only* as historical evidence of what changed: it is
never used as ground truth.
"""

import glob
import importlib.util as _ilu
import inspect
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]

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

from seestar.queuep.queue_manager import (  # noqa: E402
    SeestarQueuedStacker,
    _RESUME_FINGERPRINT_ATTRS,
    _normalize_min_weight,
    _QualityReferenceError,
    DEFAULT_MIN_WEIGHT,
    _QUALITY_METRIC_FLOOR,
    _QUALITY_FACTOR_CAP,
    _QUALITY_METRIC_CAP,
)

HEADER = fits.Header()

# ===========================================================================
# Pre-P1 oracle — reconstructed verbatim from the lines removed in the accepted
# P1 correction (git diff HEAD).  Used ONLY as historical evidence of what
# changed, never as ground truth.
# ===========================================================================


def _pre_p1_calculate_weights(
    batch_scores,
    *,
    snr_exp=1.0,
    stars_exp=0.5,
    min_weight=0.01,
    weight_by_snr=True,
    weight_by_stars=False,
):
    """The historical batch-local mean-normalised weight computation.

    Exact reconstruction of the removed code path:

        raw -> normalize to mean 1 -> floor at min_weight -> renormalize to mean 1
    """
    num_images = len(batch_scores)
    if num_images == 0:
        return np.array([])
    raw_weights = np.ones(num_images, dtype=np.float32)
    for i, scores in enumerate(batch_scores):
        weight = 1.0
        if weight_by_snr:
            weight *= max(scores.get("snr", 0.0), 0.0) ** snr_exp
        if weight_by_stars:
            weight *= max(scores.get("stars", 0.0), 0.0) ** stars_exp
        raw_weights[i] = max(weight, 1e-9)
    sum_weights = np.sum(raw_weights)
    if sum_weights > 1e-9:
        normalized_weights = raw_weights * (num_images / sum_weights)
    else:
        normalized_weights = np.ones(num_images, dtype=np.float32)
    normalized_weights = np.maximum(normalized_weights, min_weight)
    sum_weights_final = np.sum(normalized_weights)
    if sum_weights_final > 1e-9:
        normalized_weights = normalized_weights * (num_images / sum_weights_final)
    else:
        normalized_weights = np.ones(num_images, dtype=np.float32)
    return normalized_weights


# ===========================================================================
# Lightweight harness (mirrors the sibling HSI closure suites)
# ===========================================================================


def make_stack(
    mode="mean",
    norm="none",
    max_hq_mem=1_000_000_000,
    batch_size=10,
    settings=None,
    use_qw=True,
    snr_exp=1.0,
    stars_exp=0.5,
    min_weight=0.01,
    weight_by_stars=False,
    q_ref=None,
):
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
    o.weight_by_stars = weight_by_stars
    o.snr_exponent = snr_exp
    o.stars_exponent = stars_exp
    o.min_weight = min_weight
    o.apply_batch_feathering = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o.drizzle_active_session = False
    o.is_mosaic_run = False
    o.stack_kappa_low = 3.0
    o.stack_kappa_high = 3.0
    o.winsor_limits = (0.05, 0.05)
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
    # P5-FIX: pin the immutable session quality reference scale.
    o._quality_reference_scale = q_ref
    return o


def item(arr, snr=1.0, stars=0.0, mask=None):
    """Build one batch item from a fresh copy of ``arr``."""
    if mask is None:
        mask = np.ones(arr.shape[:2], dtype=bool)
    return (
        np.array(arr, dtype=np.float32, copy=True),
        HEADER,
        {"snr": float(snr), "stars": float(stars)},
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
    return np.array(arr, dtype=np.float64, copy=True)


def _w3(W, ref_ndim):
    W = _mat(W)
    if W.ndim == 2 and ref_ndim == 3:
        return W[..., None]
    return W


def _scores(snrs):
    return [{"snr": float(s), "stars": 0.0} for s in snrs]


def _base_color(shape=(32, 32, 3), seed=7):
    rng = np.random.default_rng(seed)
    H, W, C = shape
    ii = np.arange(H, dtype=np.float64)[:, None]
    jj = np.arange(W, dtype=np.float64)[None, :]
    ramp = 100.0 + 200.0 * (ii / (H - 1)) + 60.0 * (jj / (W - 1))
    base = np.stack([ramp] * C, axis=-1)
    return (base + rng.normal(0.0, 4.0, size=shape)).astype(np.float32), rng


# ===========================================================================
# A1 — raw metric factoring and reference anchoring
# ===========================================================================


def test_raw_quality_metric_is_shared_product():
    """``_raw_quality_metric`` is the single shared formula: the product of the
    enabled factors raised to their exponents, floored at 1e-9."""
    stack = make_stack(snr_exp=1.0, stars_exp=0.5, weight_by_stars=True)
    assert stack._raw_quality_metric({"snr": 4.0, "stars": 9.0}) == pytest.approx(
        4.0 * 3.0, rel=1e-9
    )
    # A failed/zero metric yields the safety floor, never zero/NaN/Inf.
    assert stack._raw_quality_metric({"snr": 0.0, "stars": 0.0}) == 1e-9
    # A negative metric is treated as zero before exponentiation.
    assert stack._raw_quality_metric({"snr": -5.0, "stars": 0.0}) == 1e-9


def test_raw_quality_metric_nan_inf_overflow_sanitized():
    """F1: ``q(scores)`` is always a finite positive float.

    NaN/invalid/negative factors collapse to the safety floor (never NaN/Inf/
    negative); ``+Inf`` / positive-overflow factors saturate to a finite upper
    bound (never the minimum), keeping high-quality saturation monotonic.
    """
    stack = make_stack(snr_exp=1.0, weight_by_stars=False)

    # NaN factor -> finite positive floor, no exception.
    q_nan = stack._raw_quality_metric({"snr": float("nan")})
    assert np.isfinite(q_nan) and q_nan > 0.0
    assert q_nan == _QUALITY_METRIC_FLOOR

    # Non-numeric / negative factor -> floor.
    assert stack._raw_quality_metric({"snr": "garbage"}) == _QUALITY_METRIC_FLOOR
    assert stack._raw_quality_metric({"snr": -3.0}) == _QUALITY_METRIC_FLOOR

    # +Inf factor -> finite upper saturation, not the minimum.
    q_inf = stack._raw_quality_metric({"snr": float("inf")})
    assert np.isfinite(q_inf)
    assert q_inf == _QUALITY_FACTOR_CAP

    # Positive overflow of the whole product -> finite metric cap, never min.
    stack_big = make_stack(snr_exp=100.0, weight_by_stars=False)
    q_overflow = stack_big._raw_quality_metric({"snr": 1e6})
    assert np.isfinite(q_overflow)
    assert q_overflow == _QUALITY_METRIC_CAP

    # Monotonic high-quality saturation: +Inf and a huge finite factor saturate
    # to the same finite value, both strictly above a normal source.
    q_huge = stack._raw_quality_metric({"snr": 1e300})
    q_normal = stack._raw_quality_metric({"snr": 100.0})
    assert q_inf == q_huge
    assert q_inf > q_normal > 0.0


def test_raw_quality_metric_never_negative_nan_inf_for_all_inputs():
    """F1 adversarial sweep: no score (including NaN/+Inf/-Inf/negative/
    nonnumeric) yields an exception, NaN, Inf, or a negative value."""
    stack = make_stack(snr_exp=1.0, stars_exp=0.5, weight_by_stars=True)
    inputs = [
        float("nan"),
        float("inf"),
        float("-inf"),
        -1.0,
        0.0,
        1e300,
        "abc",
        None,
        True,
    ]
    for bad in inputs:
        q = stack._raw_quality_metric({"snr": bad, "stars": bad})
        assert np.isfinite(q), bad
        assert q > 0.0, bad


def test_relative_weights_reference_anchor():
    """q_ref=50, q=[10,50,100]: min_weight=0.01 -> [0.2,1,2]; 0.5 -> [0.5,1,2]."""
    scores = _scores([10.0, 50.0, 100.0])

    w001 = make_stack(min_weight=0.01, q_ref=50.0)._calculate_weights(scores)
    w050 = make_stack(min_weight=0.5, q_ref=50.0)._calculate_weights(scores)

    assert np.allclose(w001, [0.2, 1.0, 2.0], rtol=1e-6)
    assert np.allclose(w050, [0.5, 1.0, 2.0], rtol=1e-6)


def test_reference_source_maps_to_weight_one():
    """A source whose metric equals q_ref receives weight exactly 1.0 (before
    the floor); weaker/stronger sources scale linearly around it."""
    stack = make_stack(min_weight=0.01, q_ref=50.0)
    w = stack._calculate_weights(_scores([5.0, 50.0, 500.0]))
    assert np.allclose(w, [0.1, 1.0, 10.0], rtol=1e-6)


def test_min_weight_is_relative_floor_not_absolute():
    """The floor binds in the *relative* domain: a source at q/q_ref below
    ``min_weight`` is raised to ``min_weight``; a source above it is untouched."""
    stack = make_stack(min_weight=0.5, q_ref=100.0)
    w = stack._calculate_weights(_scores([10.0, 50.0, 100.0]))
    # relative = [0.1, 0.5, 1.0] -> floored to [0.5, 0.5, 1.0].
    assert np.allclose(w, [0.5, 0.5, 1.0], rtol=1e-6)
    assert float(np.min(w)) >= 0.5 - 1e-6  # strict floor


# ===========================================================================
# A2 — batch independence, rescale invariance, companion independence
# ===========================================================================


def test_reorder_split_add_companion_invariant_relative():
    """With a pinned q_ref the weight is a deterministic per-frame function of
    its own metric and the reference: reorder/split/add-companion never change
    an existing frame's weight."""
    scores = _scores([10.0, 50.0, 100.0])
    stack = make_stack(min_weight=0.01, q_ref=50.0)

    full = stack._calculate_weights(scores)
    assert np.allclose(full, [0.2, 1.0, 2.0], rtol=1e-6)

    reordered = stack._calculate_weights([scores[2], scores[0], scores[1]])
    assert np.allclose(reordered, [2.0, 0.2, 1.0], rtol=1e-6)

    part1 = stack._calculate_weights([scores[0]])
    part2 = stack._calculate_weights([scores[1], scores[2]])
    assert np.allclose(part1, [0.2], rtol=1e-6)
    assert np.allclose(part2, [1.0, 2.0], rtol=1e-6)

    with_extra = stack._calculate_weights(scores + [{"snr": 200.0, "stars": 0.0}])
    assert np.allclose(with_extra[:3], full, rtol=1e-6)
    assert np.allclose(with_extra[3], [4.0], rtol=1e-6)  # 200/50

    assert np.array_equal(full, stack._calculate_weights(list(scores)))


def test_common_rescale_leaves_weights_unchanged():
    """A common rescale of the source metrics AND the reference metric leaves
    relative weights unchanged (scale invariance)."""
    base = _scores([10.0, 50.0, 100.0])
    w1 = make_stack(min_weight=0.01, q_ref=50.0)._calculate_weights(base)

    for factor in (1e-4, 1e3):
        w = make_stack(
            min_weight=0.01, q_ref=50.0 * factor
        )._calculate_weights(_scores([10.0 * factor, 50.0 * factor, 100.0 * factor]))
        assert np.allclose(w1, w, rtol=1e-6), factor


def test_relative_floor_is_scale_invariant():
    """The relative floor is invariant to a common rescale too: 1:10:20 ratios
    with a binding floor give the same floored weights at any scale."""
    w1 = make_stack(min_weight=0.5, q_ref=50.0)._calculate_weights(
        _scores([5.0, 50.0, 100.0])
    )
    w2 = make_stack(min_weight=0.5, q_ref=5.0)._calculate_weights(
        _scores([0.5, 5.0, 10.0])
    )
    assert np.allclose(w1, [0.5, 1.0, 2.0], rtol=1e-6)
    assert np.allclose(w1, w2, rtol=1e-6)


def test_companion_never_changes_existing_weight():
    """Changing a batch companion never changes an existing source weight."""
    stack = make_stack(min_weight=0.01, q_ref=50.0)
    alone = stack._calculate_weights(_scores([10.0]))
    with_weak = stack._calculate_weights(_scores([10.0, 0.001]))
    with_strong = stack._calculate_weights(_scores([10.0, 5000.0]))
    assert np.allclose(alone, [0.2], rtol=1e-6)
    assert np.allclose(with_weak[0], 0.2, rtol=1e-6)
    assert np.allclose(with_strong[0], 0.2, rtol=1e-6)


def test_reference_scale_pinned_not_derived_from_batch():
    """The pinned q_ref governs even when the first batch item is not the
    reference: the first item's metric is never promoted to the reference."""
    stack = make_stack(min_weight=0.01, q_ref=50.0)
    # 200/50 == 4.0 (not 200/200 == 1.0, which is what a batch-local index-0
    # reference would produce).
    assert np.allclose(stack._calculate_weights(_scores([200.0])), [4.0], rtol=1e-6)
    stack._quality_reference_scale = 100.0
    assert np.allclose(stack._calculate_weights(_scores([200.0])), [2.0], rtol=1e-6)


# ===========================================================================
# A3 — production seams: singleton + multi-image, SUM = V * W
# ===========================================================================


def test_singleton_and_multiimage_sumw_consistency_relative():
    """mean mode, q_ref=50, min_weight=0.5: snr=5 -> rel 0.1 -> floored 0.5,
    snr=50 -> rel 1.0.  Singleton and multi-image seams both obey SUM = V*W and
    compose identically."""
    A = np.full((16, 16), 100.0, dtype=np.float32)   # snr 5  -> 0.5
    B = np.full((16, 16), 300.0, dtype=np.float32)   # snr 50 -> 1.0

    stack = make_stack("mean", norm="none", use_qw=True, min_weight=0.5, q_ref=50.0)
    V, _hdr, W = stack._stack_batch([item(A, snr=5.0), item(B, snr=50.0)], 1, 1)
    Vm, Wm = _mat(V), _mat(W)
    expected = (0.5 * 100.0 + 1.0 * 300.0) / 1.5
    assert np.allclose(Vm, expected, rtol=1e-5)
    assert np.allclose(Wm, 1.5, rtol=1e-5)
    assert np.allclose(Vm * Wm, 0.5 * 100.0 + 1.0 * 300.0, rtol=1e-4)

    Va, _ha, Wa = stack._stack_batch([item(A, snr=5.0)], 1, 1)
    Vb, _hb, Wb = stack._stack_batch([item(B, snr=50.0)], 1, 1)
    assert np.allclose(_mat(Va), 100.0, rtol=1e-6)
    assert np.allclose(_mat(Wa), 0.5, rtol=1e-5)
    assert np.allclose(_mat(Vb), 300.0, rtol=1e-6)
    assert np.allclose(_mat(Wb), 1.0, rtol=1e-5)

    num = _mat(Va) * _mat(Wa) + _mat(Vb) * _mat(Wb)
    den = _mat(Wa) + _mat(Wb)
    with np.errstate(divide="ignore", invalid="ignore"):
        composed = num / den
    assert np.allclose(composed, expected, rtol=1e-5)


def test_singleton_stack_batch_missing_q_ref_fails_closed():
    """F2: a quality-weighted singleton batch with no pinned q_ref raises
    ``_QualityReferenceError`` (never silently continues unweighted)."""
    A = np.full((16, 16), 100.0, dtype=np.float32)
    stack = make_stack("mean", norm="none", use_qw=True, min_weight=0.5, q_ref=None)
    with pytest.raises(_QualityReferenceError):
        stack._stack_batch([item(A, snr=5.0)], 1, 1)


def test_multiimage_stack_batch_missing_q_ref_fails_closed():
    """F2: a quality-weighted multi-image batch with no pinned q_ref raises
    ``_QualityReferenceError`` (never silently falls back to uniform or
    absolute-weighted weights)."""
    A = np.full((16, 16), 100.0, dtype=np.float32)
    B = np.full((16, 16), 300.0, dtype=np.float32)
    stack = make_stack("mean", norm="none", use_qw=True, min_weight=0.5, q_ref=None)
    with pytest.raises(_QualityReferenceError):
        stack._stack_batch([item(A, snr=5.0), item(B, snr=50.0)], 1, 1)


def test_manifest_writer_missing_q_ref_fails_closed(tmp_path):
    """F2: a quality-weighted manifest write with no pinned q_ref raises
    ``_QualityReferenceError`` (never writes a checkpoint a later resume would
    reject)."""
    out = tmp_path / "out"
    out.mkdir()
    stack = _resume_stack(out, 0.5, use_qw=True, q_ref=None)
    with pytest.raises(_QualityReferenceError):
        stack._write_resume_manifest(state="clean")


def test_manifest_writer_malformed_q_ref_fails_closed(tmp_path):
    """F2: a malformed (non-finite/non-positive) q_ref refuses a manifest
    write too."""
    out = tmp_path / "out"
    out.mkdir()
    for bad in (float("nan"), float("inf"), 0.0, -1.0, "garbage", None):
        stack = _resume_stack(out, 0.5, use_qw=True, q_ref=bad)
        with pytest.raises(_QualityReferenceError):
            stack._write_resume_manifest(state="clean")


def test_unequal_decomposition_invariant_with_binding_floor():
    """[A,B,C] == [A,B]+[C] == [C,A]+[B] with a binding *relative* floor: the
    weight is a deterministic per-frame function of its metric and q_ref, so
    SUM/WHT composition is exact regardless of batch boundaries."""
    A = np.full((16, 16), 100.0, dtype=np.float32)   # snr 5  -> rel 0.1 -> 0.5
    B = np.full((16, 16), 200.0, dtype=np.float32)   # snr 25 -> rel 0.5 -> 0.5
    C = np.full((16, 16), 300.0, dtype=np.float32)   # snr 50 -> rel 1.0 -> 1.0

    stack = make_stack("mean", norm="none", use_qw=True, min_weight=0.5, q_ref=50.0)

    def reduce(groups):
        num = None
        den = None
        for g in groups:
            V, _h, W = stack._stack_batch([item(x, snr=s) for x, s in g], 1, 1)
            V = _mat(V)
            W = _mat(W)
            num = V * W if num is None else num + V * W
            den = W if den is None else den + W
        with np.errstate(divide="ignore", invalid="ignore"):
            return num / den

    expected = (0.5 * 100 + 0.5 * 200 + 1.0 * 300) / (0.5 + 0.5 + 1.0)  # 450/2.0
    global_ = reduce([[(A, 5.0), (B, 25.0), (C, 50.0)]])
    assert np.allclose(global_, expected, rtol=1e-5)

    for groups in (
        [[(A, 5.0), (B, 25.0)], [(C, 50.0)]],
        [[(C, 50.0), (A, 5.0)], [(B, 25.0)]],
        [[(A, 5.0)], [(B, 25.0)], [(C, 50.0)]],
    ):
        r = reduce(groups)
        assert np.allclose(r, expected, rtol=1e-5), groups


def test_backend_parity_with_binding_relative_floor(tmp_path, monkeypatch):
    """kappa-sigma (group_size >= N), q_ref=50, min_weight=0.5, snrs=[5,50,100,
    150,200] -> relative [0.1,1,2,3,4] -> [0.5,1,2,3,4].  RAM / tiled-HQ /
    memmap agree on V, W and SUM=V*W, and the effective per-channel WHT equals
    the floored weight sum (10.5) at every fully-covered pixel."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    A, rng = _base_color()
    snrs = [5.0, 50.0, 100.0, 150.0, 200.0]
    obs = [A] + [(A + rng.normal(0.0, 4.0, size=A.shape)).astype(np.float32)
                 for _ in range(4)]
    items = [item(obs[i], snr=snrs[i]) for i in range(len(obs))]

    def run(backend):
        if backend == "ram":
            s = make_stack("kappa-sigma", norm="none", use_qw=True,
                           min_weight=0.5, q_ref=50.0,
                           max_hq_mem=1_000_000_000, batch_size=10)
        elif backend == "tile":
            s = make_stack("kappa-sigma", norm="none", use_qw=True,
                           min_weight=0.5, q_ref=50.0,
                           max_hq_mem=100_000, batch_size=10,
                           settings=types.SimpleNamespace(TILE_HEIGHT=8, batch_size=None))
        else:  # memmap
            s = make_stack("kappa-sigma", norm="none", use_qw=True,
                           min_weight=0.5, q_ref=50.0,
                           max_hq_mem=100_000, batch_size=1,
                           settings=types.SimpleNamespace(TILE_HEIGHT=8, batch_size=1))
        V_raw, _, W = s._stack_batch(items, 1, 1)
        V = _mat(V_raw)
        _release(V_raw)
        return V, _w3(W, V.ndim)

    (Vr, Wr), (Vt, Wt), (Vm, Wm) = run("ram"), run("tile"), run("memmap")

    # Effective per-channel WHT at a fully covered interior pixel = 10.5.
    assert np.allclose(Wr[16, 16], [10.5] * 3, rtol=1e-5)
    assert Wr.shape == (32, 32, 3)

    for other in (("tile", Vt, Wt), ("memmap", Vm, Wm)):
        assert np.allclose(Vr, other[1], rtol=1e-5, atol=1e-2), other[0]
        assert np.allclose(Wr, other[2], rtol=1e-5, atol=1e-3), other[0]
        assert np.allclose(Vr * Wr, other[1] * other[2], rtol=1e-5, atol=1e-2), other[0]

    # The floored weight (0.5 for the weakest source, not its raw 0.1) entered
    # the denominator.
    assert np.isclose(float(Wr[16, 16, 0]), 0.5 + 1.0 + 2.0 + 3.0 + 4.0, rtol=1e-5)
    assert not glob.glob(os.path.join(str(tmp_path), "hq_batch*.dat"))


# ===========================================================================
# A4 — historical defect evidence (pre-P1 oracle, retained as "what changed")
# ===========================================================================


def test_pre_p1_oracle_was_batch_local_and_relative_not_strict():
    """The removed pre-P1 algorithm was batch-local mean-normalised: its floor
    was relative and not even strict (the final renormalisation could push the
    minimum back below ``min_weight``).  This is retained only as proof of the
    defect the P1/P5 corrections removed."""
    scores = _scores([0.001, 0.005, 1.0])
    pre = _pre_p1_calculate_weights(scores, min_weight=0.01)
    assert float(np.min(pre)) < 0.01       # relative floor, re-normalised away
    assert np.isclose(float(np.sum(pre)), 3.0, atol=1e-5)  # still mean 1

    # The corrected relative floor is strict and batch-independent.
    corrected = make_stack(min_weight=0.01, q_ref=1.0)._calculate_weights(scores)
    assert float(np.min(corrected)) >= 0.01 - 1e-6


def test_corrected_floor_not_inert_for_typical_snr():
    """The pre-P1 defect was that the *absolute* floor was inert for typical
    SNR (10-100) because the raw metric product has no [0,1] scale.  With a
    pinned reference the floor is meaningful again: min_weight is a fraction of
    the reference quality scale.  Without a pinned reference the quality-
    weighted path now fails closed (never silently falls back to the absolute
    domain)."""
    scores = _scores([10.0, 50.0, 100.0])
    # No reference scale: fail closed, no raw-domain fallback.
    with pytest.raises(_QualityReferenceError):
        make_stack(min_weight=0.5, q_ref=None)._calculate_weights(scores)

    # Pinned reference: the same setting now floors relative to q_ref.
    relative = make_stack(min_weight=0.5, q_ref=50.0)._calculate_weights(scores)
    assert np.allclose(relative, [0.5, 1.0, 2.0], rtol=1e-6)


# ===========================================================================
# A5 — configuration transport & clamping (no GUI launch)
# ===========================================================================


def _load_run_config_and_settings():
    run_config_spec = _ilu.spec_from_file_location(
        "seestar_run_config", ROOT / "seestar" / "gui" / "run_config.py"
    )
    run_config = _ilu.module_from_spec(run_config_spec)
    sys.modules["seestar_run_config"] = run_config
    run_config_spec.loader.exec_module(run_config)

    settings_spec = _ilu.spec_from_file_location(
        "seestar_settings_manager", ROOT / "seestar" / "gui" / "settings.py"
    )
    settings_mod = _ilu.module_from_spec(settings_spec)
    sys.modules["seestar_settings_manager"] = settings_mod
    settings_spec.loader.exec_module(settings_mod)
    return run_config, settings_mod


def test_run_config_transports_min_w_to_min_weight():
    run_config, settings_mod = _load_run_config_and_settings()
    sm = settings_mod.SettingsManager(settings_file="unused.json")
    assert sm.min_weight == 0.01  # default
    sm.min_weight = 0.37
    assert run_config.build_backend_kwargs(sm)["min_w"] == 0.37


def test_backend_clamp_min_weight_to_001_10():
    """The backend transport seam normalizes min_weight via
    ``_normalize_min_weight`` (source assertion) and maps edge values as
    documented: NaN/Inf/nonnumeric -> 0.01, finite -> clamp to [0.01, 1.0]."""
    src = inspect.getsource(SeestarQueuedStacker.start_processing)
    assert "self.min_weight = _normalize_min_weight(min_w)" in src

    assert _normalize_min_weight(0.005) == 0.01
    assert _normalize_min_weight(0.0) == 0.01
    assert _normalize_min_weight(-1.0) == 0.01
    assert _normalize_min_weight(0.5) == 0.5
    assert _normalize_min_weight(1.0) == 1.0
    assert _normalize_min_weight(5.0) == 1.0


def test_settings_validation_clips_min_weight_to_001():
    """Settings validation now enforces the full [0.01, 1.0] range itself, so
    ``0.005`` cannot pass one seam only to be clamped later: it is clipped to
    0.01 at the settings seam, agreeing with the backend clamp."""
    _, settings_mod = _load_run_config_and_settings()
    sm = settings_mod.SettingsManager(settings_file="unused.json")

    sm.min_weight = 5.0
    sm.validate_settings()
    assert sm.min_weight == 1.0

    sm.min_weight = 0.0
    sm.validate_settings()
    assert sm.min_weight == 0.01

    sm.min_weight = -0.5
    sm.validate_settings()
    assert sm.min_weight == 0.01

    # The corrected seam: sub-0.01 is clipped at validation, not left for the
    # backend to fix silently later.
    sm.min_weight = 0.005
    sm.validate_settings()
    assert sm.min_weight == 0.01

    sm.min_weight = 0.37
    sm.validate_settings()
    assert sm.min_weight == 0.37


def test_settings_validation_normalizes_nan_inf_nonnumeric():
    """F3: NaN, +/-Inf and nonnumeric min_weight normalize to the documented
    default 0.01 (not silently left as NaN or clamped to a spurious bound)."""
    _, settings_mod = _load_run_config_and_settings()
    sm = settings_mod.SettingsManager(settings_file="unused.json")

    for bad in (float("nan"), float("inf"), float("-inf"), "abc", None, True):
        sm.min_weight = bad
        sm.validate_settings()
        assert sm.min_weight == 0.01, (bad, sm.min_weight)
        assert np.isfinite(sm.min_weight)

    # Boundaries stay put.
    sm.min_weight = 0.01
    sm.validate_settings()
    assert sm.min_weight == 0.01
    sm.min_weight = 1.0
    sm.validate_settings()
    assert sm.min_weight == 1.0


def test_backend_and_settings_min_weight_normalization_agree():
    """F3: the backend transport normalization and the settings seam produce
    identical results for every edge value (no drift between the two)."""
    _, settings_mod = _load_run_config_and_settings()
    for value in (
        0.005, 0.0, -1.0, 0.5, 1.0, 5.0,
        float("nan"), float("inf"), float("-inf"), "abc", None, True, 0.01,
    ):
        backend = _normalize_min_weight(value)
        settings = settings_mod.normalize_min_weight(value)
        assert backend == settings, value
        assert backend == pytest.approx(settings)


def test_tk_default_min_weight_is_001():
    """F3: the Tk surface initializes min_weight_var to the documented default
    0.01 (source-level assertion, no GUI launch), agreeing with settings and
    the backend."""
    src = (ROOT / "seestar" / "gui" / "main_window.py").read_text(
        encoding="utf-8"
    )
    assert "self.min_weight_var = tk.DoubleVar(value=0.01)" in src
    assert "self.min_weight_var = tk.DoubleVar(value=0.1)" not in src


def test_qt_surface_min_weight_min_is_001():
    """The Qt settings surface advertises [0.01, 1.0] for min_weight (no GUI
    launch; a source-level assertion on the field spec)."""
    mw_src = (ROOT / "seestar" / "gui_qt" / "main_window.py").read_text(
        encoding="utf-8"
    )
    assert '_field("min_weight", "Minimum weight", "float", 0.01, 1.0, 0.01, 3)' in mw_src

    # The Qt-side default is also 0.01.
    ss_spec = _ilu.spec_from_file_location(
        "seestar_qt_settings_state", ROOT / "seestar" / "gui_qt" / "settings_state.py"
    )
    ss_mod = _ilu.module_from_spec(ss_spec)
    sys.modules["seestar_qt_settings_state"] = ss_mod
    ss_spec.loader.exec_module(ss_mod)
    assert ss_mod.QtSettingsState().min_weight == 0.01


# ===========================================================================
# A6 — resume fingerprint + q_ref persistence / fail-closed restore
# ===========================================================================


def _resume_stack(out_dir, min_weight, use_qw=True, q_ref=None):
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.output_folder = str(out_dir)
    o.is_mosaic_run = False
    o.drizzle_active_session = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o.min_weight = min_weight
    o.use_quality_weighting = use_qw
    o._quality_reference_scale = q_ref
    return o


def _write_min_manifest(out_dir, fingerprint, q_ref=None):
    memdir = Path(out_dir) / "memmap_accumulators"
    memdir.mkdir(parents=True, exist_ok=True)
    (memdir / "resume_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "clean",
                "mode": "classic_sumw",
                "fingerprint": fingerprint,
                "quality_reference_scale": q_ref,
            }
        )
    )


def test_resume_fingerprint_includes_min_weight():
    assert "min_weight" in _RESUME_FINGERPRINT_ATTRS
    # The scientific fingerprint is sensitive to min_weight alone.
    fp_a = _resume_stack(Path("unused"), 0.1)._scientific_fingerprint()
    fp_b = _resume_stack(Path("unused"), 0.5)._scientific_fingerprint()
    assert fp_a != fp_b


def test_changing_min_weight_refuses_incompatible_checkpoint(tmp_path):
    fp_010 = _resume_stack(tmp_path, 0.1)._scientific_fingerprint()
    _write_min_manifest(tmp_path, fp_010)

    # Matching min_weight -> passes the fingerprint gate (fails later on the
    # minimal manifest's missing shape/dtype, but NOT on configuration).
    ok_same, reason_same = _resume_stack(tmp_path, 0.1)._validate_and_open_resume(
        (2, 2, 3)
    )
    assert ok_same is False
    assert "configuration mismatch" not in reason_same

    # Changed min_weight -> refused at the fingerprint gate.
    ok_diff, reason_diff = _resume_stack(tmp_path, 0.5)._validate_and_open_resume(
        (2, 2, 3)
    )
    assert ok_diff is False
    assert "configuration mismatch" in reason_diff


def test_quality_weighted_missing_q_ref_refused_before_mutation(tmp_path):
    """A quality-weighted checkpoint with a missing q_ref must refuse resume at
    the early read-only validation, before any reference/artifact mutation."""
    stack = _resume_stack(tmp_path, 0.1, use_qw=True, q_ref=50.0)
    fp = stack._scientific_fingerprint()
    _write_min_manifest(tmp_path, fp, q_ref=None)  # missing scale

    ok, reason, _ = stack._validate_resume_headless()
    assert ok is False
    assert "quality reference scale" in reason


def test_quality_weighted_malformed_q_ref_refused(tmp_path):
    """Non-numeric / nonfinite / nonpositive q_ref values are refused fail-closed
    for a quality-weighted checkpoint (never silently treated as 1)."""
    stack = _resume_stack(tmp_path, 0.1, use_qw=True, q_ref=50.0)
    fp = stack._scientific_fingerprint()

    for bad in ("not-a-number", float("nan"), float("inf"), 0.0, -5.0, True, None):
        _write_min_manifest(tmp_path, fp, q_ref=bad)
        ok, reason, _ = stack._validate_resume_headless()
        assert ok is False, (bad, reason)
        assert "quality reference scale" in reason, (bad, reason)
