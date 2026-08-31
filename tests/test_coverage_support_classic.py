"""COV-01B-CLASSIC focused tests: plain-classic support-domain wiring.

Unit-level only (no full pipeline run): the ordered per-exposure support
payload, the atomic accumulate_support_pair seam, boolean-mask validation,
fail-before-mutation, and ARRAY-EXACT decomposition invariance (61 vs 3+17+41
vs singletons).
"""

import numpy as np
import pytest

from seestar.core.coverage_support import accumulate_support_pair
from seestar.queuep.queue_manager import SeestarQueuedStacker, _ResumeCheckpointError


def _fresh(tmp_path, name, shape=(4, 5)):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    qm = object.__new__(SeestarQueuedStacker)
    qm.output_folder = str(d)
    qm.coverage_sup_w1_memmap = None
    qm.coverage_sup_w2_memmap = None
    qm._support_state_available = True
    qm._support_unavailable_reason = None
    qm._create_support_memmaps(shape)
    return qm


def _payload(masks, scalars):
    return [(m, float(s)) for m, s in zip(masks, scalars)]


# ---------------------------------------------------------------------------
# accumulate_support_pair adversarial (dtype / existing accumulator state)
# ---------------------------------------------------------------------------
def test_accumulate_pair_rejects_wrong_accumulator_dtype():
    w1 = np.zeros((2, 2), dtype=np.int64)
    w2 = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(TypeError):
        accumulate_support_pair(w1, w2, np.ones((2, 2)), dtype=np.float64)


def test_accumulate_pair_rejects_existing_nonfinite_accumulator():
    w1 = np.zeros((2, 2), dtype=np.float64)
    w2 = np.full((2, 2), np.inf, dtype=np.float64)
    with pytest.raises(ValueError):
        accumulate_support_pair(w1, w2, np.ones((2, 2)), dtype=np.float64)


def test_accumulate_pair_rejects_existing_negative_accumulator():
    w1 = np.zeros((2, 2), dtype=np.float64)
    w2 = np.full((2, 2), -1.0, dtype=np.float64)
    with pytest.raises(ValueError):
        accumulate_support_pair(w1, w2, np.ones((2, 2)), dtype=np.float64)


def test_accumulate_pair_failure_is_byte_identical():
    w1 = np.full((2, 2), 1.0, dtype=np.float64)
    w2 = np.full((2, 2), 2.0, dtype=np.float64)
    w1_before = w1.copy()
    w2_before = w2.copy()
    with pytest.raises(ValueError):
        accumulate_support_pair(w1, w2, np.full((2, 2), np.nan), dtype=np.float64)
    assert np.array_equal(w1, w1_before)
    assert np.array_equal(w2, w2_before)


# ---------------------------------------------------------------------------
# _build_support_payload validation
# ---------------------------------------------------------------------------
def test_build_payload_cardinality_mismatch_fails_closed(tmp_path):
    qm = _fresh(tmp_path, "card")
    with pytest.raises(_ResumeCheckpointError):
        qm._build_support_payload(
            [np.ones((4, 5), bool), np.ones((4, 5), bool)], [1.0]
        )


def test_build_payload_invalid_scalar_fails_closed(tmp_path):
    qm = _fresh(tmp_path, "scalar")
    with pytest.raises(_ResumeCheckpointError):
        qm._build_support_payload([np.ones((4, 5), bool)], [np.nan])
    with pytest.raises(_ResumeCheckpointError):
        qm._build_support_payload([np.ones((4, 5), bool)], [-1.0])


def test_build_payload_missing_masks_fails_closed(tmp_path):
    qm = _fresh(tmp_path, "missing")
    with pytest.raises(_ResumeCheckpointError):
        qm._build_support_payload(None, [1.0])


# ---------------------------------------------------------------------------
# _apply_support_payload validation + order
# ---------------------------------------------------------------------------
def test_apply_payload_rejects_nonboolean_mask(tmp_path):
    qm = _fresh(tmp_path, "nonbool")
    with pytest.raises(_ResumeCheckpointError):
        qm._apply_support_payload([(np.ones((4, 5), np.float32), 1.0)])


def test_apply_payload_rejects_shape_mismatch_no_transpose(tmp_path):
    qm = _fresh(tmp_path, "shape")
    with pytest.raises(_ResumeCheckpointError):
        qm._apply_support_payload([(np.ones((3, 5), bool), 1.0)])
    # A transposed mask must also fail (no silent transpose re-binding).
    with pytest.raises(_ResumeCheckpointError):
        qm._apply_support_payload([(np.ones((5, 4), bool), 1.0)])


def test_known_masks_scalars_exact_sup(tmp_path):
    qm = _fresh(tmp_path, "known", shape=(2, 2))
    m0 = np.array([[True, True], [False, True]], dtype=bool)
    m1 = np.array([[True, False], [True, True]], dtype=bool)
    qm._apply_support_payload(_payload([m0, m1], [2.0, 3.0]))
    w1 = qm.coverage_sup_w1_memmap
    w2 = qm.coverage_sup_w2_memmap
    expected_w1 = m0.astype(np.float64) * 2.0 + m1.astype(np.float64) * 3.0
    expected_w2 = m0.astype(np.float64) * 4.0 + m1.astype(np.float64) * 9.0
    assert np.array_equal(w1, expected_w1)
    assert np.array_equal(w2, expected_w2)


def test_decomposition_exact_61_vs_partitions(tmp_path):
    rng = np.random.default_rng(0)
    shape = (4, 5)
    masks = [rng.random(shape) > 0.5 for _ in range(61)]
    scalars = [1.0 + 0.01 * i for i in range(61)]

    qm_all = _fresh(tmp_path, "all", shape)
    qm_all._apply_support_payload(_payload(masks, scalars))

    qm_part = _fresh(tmp_path, "part", shape)
    qm_part._apply_support_payload(_payload(masks[:3], scalars[:3]))
    qm_part._apply_support_payload(_payload(masks[3:20], scalars[3:20]))
    qm_part._apply_support_payload(_payload(masks[20:], scalars[20:]))

    qm_single = _fresh(tmp_path, "single", shape)
    for m, s in zip(masks, scalars):
        qm_single._apply_support_payload([(m, float(s))])

    w1_all = qm_all.coverage_sup_w1_memmap
    w2_all = qm_all.coverage_sup_w2_memmap
    assert np.array_equal(w1_all, qm_part.coverage_sup_w1_memmap)
    assert np.array_equal(w2_all, qm_part.coverage_sup_w2_memmap)
    assert np.array_equal(w1_all, qm_single.coverage_sup_w1_memmap)
    assert np.array_equal(w2_all, qm_single.coverage_sup_w2_memmap)


# ---------------------------------------------------------------------------
# REWORK R1: adversarial + production witnesses
# ---------------------------------------------------------------------------

def _bare_mean_stack(weighting_method="none"):
    import types as _types
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.logger = _types.SimpleNamespace(
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    o.stacking_mode = "mean"
    o.normalize_method = "none"
    o.weighting_method = weighting_method
    o.use_quality_weighting = False
    o.weight_by_snr = False
    o.weight_by_stars = False
    o.snr_exponent = 1.0
    o.stars_exponent = 0.5
    o.min_weight = 0.0
    o.apply_batch_feathering = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o.drizzle_active_session = False
    o.is_mosaic_run = False
    o.stack_kappa_low = 3.0
    o.stack_kappa_high = 3.0
    o.winsor_limits = (0.05, 0.05)
    o.stack_reject_algo = "none"
    o.max_hq_mem = 1_000_000_000
    o.batch_size = 10
    o.settings = None
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.interbatch_norm_active = False
    o.max_stack_workers = 1
    o._current_batch_paths = []
    o._norm_reference = None
    o._is_plain_classic = lambda: False
    o._support_state_available = True
    return o


def test_combine_fails_closed_absent_payload(tmp_path):
    from astropy.io import fits as _fits
    qm = _fresh(tmp_path, "fc")
    hdr = _fits.Header()  # no _coverage_support_payload attribute
    with pytest.raises(_ResumeCheckpointError):
        qm._combine_batch_result(
            np.ones((4, 5, 3), np.float32), hdr, np.ones((4, 5), np.float32)
        )


@pytest.mark.parametrize("bad", [None, 42, "x", 3.14])
def test_combine_fails_closed_malformed_payload(tmp_path, bad):
    from astropy.io import fits as _fits
    qm = _fresh(tmp_path, "fc")
    hdr = _fits.Header()
    hdr._coverage_support_payload = bad
    with pytest.raises(_ResumeCheckpointError):
        qm._combine_batch_result(
            np.ones((4, 5, 3), np.float32), hdr, np.ones((4, 5), np.float32)
        )


def test_manifest_metadata_fails_closed_missing_accumulators():
    qm = object.__new__(SeestarQueuedStacker)
    qm._support_state_available = True
    qm.coverage_sup_w1_memmap = None
    qm.coverage_sup_w2_memmap = None
    with pytest.raises(_ResumeCheckpointError):
        qm._support_manifest_metadata()


def test_support_artifacts_detected_as_resume_signal(tmp_path):
    import pathlib
    d = tmp_path / "out"
    memdir = d / "memmap_accumulators"
    memdir.mkdir(parents=True)
    (memdir / "coverage_SUP_W1.npy").write_bytes(b"x")
    qm = object.__new__(SeestarQueuedStacker)
    assert qm._resume_artifacts_present(str(d)) is True


def test_validate_support_readonly_shape_cross_check(tmp_path):
    import pathlib
    qm = _fresh(tmp_path, "xs", shape=(4, 5))
    memdir = pathlib.Path(qm.output_folder) / "memmap_accumulators"
    support_meta = {"schema": "sup_v1", "dtype": "float64", "shape": [3, 3]}
    ok, reason = qm._validate_support_readonly(support_meta, memdir, (4, 5, 3))
    assert ok is False
    assert "shape" in reason


def test_production_singleton_vs_multi_support_consistency():
    from astropy.io import fits as _fits
    o = _bare_mean_stack(weighting_method="variance")
    rng = np.random.default_rng(7)

    def make_img(mu, sig):
        return np.stack([
            rng.normal(mu, sig, (4, 5)),
            rng.normal(mu, sig * 0.2, (4, 5)),
            rng.normal(mu, sig * 0.2, (4, 5)),
        ], axis=-1).astype(np.float32)

    img_a = make_img(100.0, 30.0)
    img_b = make_img(50.0, 10.0)
    mask = np.ones((4, 5), dtype=bool)

    def item(img):
        return (img, _fits.Header(), {"snr": 1.0, "stars": 0.0}, None, mask)

    _, hdr_multi, _ = o._stack_batch([item(img_a), item(img_b)], 1, 1)
    q_a_multi = hdr_multi._coverage_support_payload[0][1]

    _, hdr_single, _ = o._stack_batch([item(img_a)], 1, 1)
    q_a_single = hdr_single._coverage_support_payload[0][1]

    assert q_a_multi == q_a_single


# ---------------------------------------------------------------------------
# REWORK R2: production transaction + resume/cleanup witnesses
# ---------------------------------------------------------------------------

def _prod_stack(tmp_path, name, shape=(4, 5, 3), stacking_mode="mean", weighting_method="none"):
    import types as _types
    import os as _os
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.output_folder = str(tmp_path / name)
    _os.makedirs(o.output_folder, exist_ok=True)
    o.update_progress = lambda *a, **k: None
    o.logger = _types.SimpleNamespace(
        warning=lambda *a, **k: None, debug=lambda *a, **k: None,
        info=lambda *a, **k: None, error=lambda *a, **k: None)
    o.stacking_mode = stacking_mode
    o.normalize_method = "none"
    o.weighting_method = weighting_method
    o.use_quality_weighting = False
    o.weight_by_snr = False
    o.weight_by_stars = False
    o.snr_exponent = 1.0
    o.stars_exponent = 0.5
    o.min_weight = 0.0
    o.apply_batch_feathering = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o.drizzle_active_session = False
    o.is_mosaic_run = False
    o.stack_kappa_low = 3.0
    o.stack_kappa_high = 3.0
    o.winsor_limits = (0.05, 0.05)
    o.stack_reject_algo = "none"
    o.max_hq_mem = 1_000_000_000
    o.batch_size = 10
    o.settings = None
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.interbatch_norm_active = False
    o.max_stack_workers = 1
    o._current_batch_paths = []
    o._norm_reference = None
    o._is_plain_classic = lambda: True
    o._support_state_available = True
    o._checkpointing_enabled = False
    o.coverage_sup_w1_memmap = None
    o.coverage_sup_w2_memmap = None
    o._support_unavailable_reason = None
    o._create_support_memmaps(shape[:2])
    memdir = _os.path.join(o.output_folder, "memmap_accumulators")
    _os.makedirs(memdir, exist_ok=True)
    o.sum_memmap_path = _os.path.join(memdir, "cumulative_SUM.npy")
    o.wht_memmap_path = _os.path.join(memdir, "cumulative_WHT.npy")
    o.cumulative_sum_memmap = np.lib.format.open_memmap(
        o.sum_memmap_path, mode="w+", dtype=np.float32, shape=shape)
    o.cumulative_wht_memmap = np.lib.format.open_memmap(
        o.wht_memmap_path, mode="w+", dtype=np.float32, shape=shape)
    o.cumulative_sum_memmap[:] = 0.0
    o.cumulative_wht_memmap[:] = 0.0
    o.memmap_shape = shape
    o.memmap_dtype_sum = np.float32
    o.memmap_dtype_wht = np.float32
    o.stacked_batches_count = 0
    o.images_in_cumulative_stack = 0
    o.total_exposure_seconds = 0.0
    o._exposure_unknown_count = 0
    o._exposure_min = None
    o._exposure_max = None
    o.failed_stack_count = 0
    o.stop_processing = False
    o.current_stack_header = None
    return o


def _item(img, mask):
    from astropy.io import fits as _fits
    return (img, _fits.Header(), {"snr": 1.0, "stars": 0.0}, None, mask)


def _make_images(rng, n, shape=(4, 5, 3)):
    return [rng.normal(100.0, 20.0, shape).astype(np.float32) for _ in range(n)]


def _run_batch(stack, items):
    stacked, hdr, cov = stack._stack_batch(items, 1, 1)
    assert stacked is not None
    stack._combine_batch_result(stacked, hdr, cov)


@pytest.mark.parametrize("weighting_method", ["none", "variance", "fwhm"])
def test_production_decomposition_singleton_vs_multi(tmp_path, weighting_method):
    rng = np.random.default_rng(11)
    shape = (4, 5, 3)
    mask = np.ones((4, 5), dtype=bool)
    imgs = _make_images(rng, 3, shape)
    items = [_item(im, mask) for im in imgs]

    # 3 exposures in one batch
    one = _prod_stack(tmp_path, "one", shape, "mean", weighting_method)
    _run_batch(one, items)

    # 3 singleton batches
    sgl = _prod_stack(tmp_path, "sgl", shape, "mean", weighting_method)
    for it in items:
        _run_batch(sgl, [it])

    assert np.array_equal(one.coverage_sup_w1_memmap, sgl.coverage_sup_w1_memmap)
    assert np.array_equal(one.coverage_sup_w2_memmap, sgl.coverage_sup_w2_memmap)


def test_production_median_route_commits_support(tmp_path):
    rng = np.random.default_rng(12)
    shape = (4, 5, 3)
    mask = np.ones((4, 5), dtype=bool)
    imgs = _make_images(rng, 4, shape)
    items = [_item(im, mask) for im in imgs]
    stack = _prod_stack(tmp_path, "med", shape, "median", "none")
    _run_batch(stack, items)
    # median: W2 == sum of mask**2 (unit weights) == n per pixel
    assert np.all(stack.coverage_sup_w2_memmap == 4.0)
    assert np.all(stack.coverage_sup_w1_memmap == 4.0)


def test_legacy_manifest_no_support_marks_unavailable(tmp_path):
    import pathlib
    stack = _prod_stack(tmp_path, "legacy", shape=(4, 5, 3))
    memdir = pathlib.Path(stack.output_folder) / "memmap_accumulators"
    manifest = {"support": None}  # legacy: no support metadata
    stack._load_support_on_resume(manifest, memdir)
    assert stack._support_state_available is False
    assert stack._support_unavailable_reason is not None
    assert stack.coverage_sup_w1_memmap is None


def test_failed_fresh_cleanup_removes_support_artifacts(tmp_path):
    import pathlib
    import os as _os
    stack = _prod_stack(tmp_path, "clean", shape=(4, 5, 3))
    memdir = pathlib.Path(stack.output_folder) / "memmap_accumulators"
    w1_path = memdir / "coverage_SUP_W1.npy"
    w2_path = memdir / "coverage_SUP_W2.npy"
    assert w1_path.exists() and w2_path.exists()
    # snapshot pre-existing state (nothing) then remove attempt-created artifacts
    stack._attempt_preexisting_state = set()
    stack._remove_attempt_created_state()
    assert not w1_path.exists()
    assert not w2_path.exists()


# ---------------------------------------------------------------------------
# REWORK R3: preflight / manifest identity / dirty / cleanup / quality / no-SCI-change
# ---------------------------------------------------------------------------

def _combine_harness(tmp_path, name, nimages=1):
    from astropy.io import fits as _fits
    stack = _prod_stack(tmp_path, name, shape=(4, 5, 3))
    hdr = _fits.Header()
    hdr["NIMAGES"] = nimages
    cov = np.ones((4, 5, 3), np.float32)
    v = np.ones((4, 5, 3), np.float32)
    return stack, v, hdr, cov


@pytest.mark.parametrize("bad", [
    [(np.ones((4, 5), bool), 1.0), (np.ones((4, 5), bool), 1.0)],  # cardinality 2 vs 1
    [("not-a-pair",)],
    [(np.ones((4, 5), np.float32), 1.0)],  # non-bool mask
    [(np.ones((3, 5), bool), 1.0)],  # wrong shape mask
    [(np.ones((4, 5), bool), np.nan)],  # NaN scalar
    [(np.ones((4, 5), bool), -1.0)],  # negative scalar
])
def test_combine_preflight_rejects_malformed_payload(tmp_path, bad):
    stack, v, hdr, cov = _combine_harness(tmp_path, "pre")
    hdr._coverage_support_payload = bad
    w1_before = stack.coverage_sup_w1_memmap.copy()
    w2_before = stack.coverage_sup_w2_memmap.copy()
    sum_before = stack.cumulative_sum_memmap.copy()
    wht_before = stack.cumulative_wht_memmap.copy()
    with pytest.raises(_ResumeCheckpointError):
        stack._combine_batch_result(v, hdr, cov)
    assert np.array_equal(stack.coverage_sup_w1_memmap, w1_before)
    assert np.array_equal(stack.coverage_sup_w2_memmap, w2_before)
    assert np.array_equal(stack.cumulative_sum_memmap, sum_before)
    assert np.array_equal(stack.cumulative_wht_memmap, wht_before)


def test_manifest_metadata_rejects_missing_file(tmp_path):
    import pathlib
    import os as _os
    stack = _prod_stack(tmp_path, "missing", shape=(4, 5, 3))
    memdir = pathlib.Path(stack.output_folder) / "memmap_accumulators"
    _os.remove(memdir / "coverage_SUP_W1.npy")
    with pytest.raises(_ResumeCheckpointError):
        stack._support_manifest_metadata()


def test_support_apply_failure_keeps_dirty(tmp_path, monkeypatch):
    from astropy.io import fits as _fits
    stack, v, hdr, cov = _combine_harness(tmp_path, "dirty")
    stack._checkpointing_enabled = True
    # finite scalar that passes preflight but whose square overflows float64
    hdr._coverage_support_payload = [(np.ones((4, 5), bool), 1e200)]
    states = []
    monkeypatch.setattr(
        stack, "_write_resume_manifest",
        lambda *a, **k: states.append(k.get("state")),
    )
    stack._combine_batch_result(v, hdr, cov)
    assert "dirty" in states
    assert "clean" not in states


def test_cleanup_failed_start_removes_support(tmp_path):
    import pathlib
    stack = _prod_stack(tmp_path, "cleanup", shape=(4, 5, 3))
    memdir = pathlib.Path(stack.output_folder) / "memmap_accumulators"
    stack._attempt_preexisting_state = set()
    stack._cleanup_failed_start()
    assert stack.coverage_sup_w1_memmap is None
    assert stack.coverage_sup_w2_memmap is None
    assert not (memdir / "coverage_SUP_W1.npy").exists()
    assert not (memdir / "coverage_SUP_W2.npy").exists()


def test_legacy_resume_preserves_sumwht(tmp_path):
    import pathlib
    stack = _prod_stack(tmp_path, "legacy2", shape=(4, 5, 3))
    memdir = pathlib.Path(stack.output_folder) / "memmap_accumulators"
    # accumulate some SUM/WHT, then simulate a legacy (support-less) resume
    stack.cumulative_sum_memmap[:] = 7.0
    stack.cumulative_wht_memmap[:] = 3.0
    sum_before = stack.cumulative_sum_memmap.copy()
    wht_before = stack.cumulative_wht_memmap.copy()
    stack._load_support_on_resume({"support": None}, memdir)
    assert stack._support_state_available is False
    assert stack._support_unavailable_reason is not None
    # SUM/WHT are preserved (not touched by legacy resume)
    assert np.array_equal(stack.cumulative_sum_memmap, sum_before)
    assert np.array_equal(stack.cumulative_wht_memmap, wht_before)


def test_support_tracking_no_sci_change(tmp_path):
    rng = np.random.default_rng(21)
    shape = (4, 5, 3)
    mask = np.ones((4, 5), dtype=bool)
    imgs = _make_images(rng, 3, shape)
    items = [_item(im, mask) for im in imgs]
    with_sup = _prod_stack(tmp_path, "ws", shape, "mean", "none")
    _run_batch(with_sup, items)
    without = _prod_stack(tmp_path, "wo", shape, "mean", "none")
    without._support_state_available = False
    _run_batch(without, items)
    assert np.array_equal(with_sup.cumulative_sum_memmap, without.cumulative_sum_memmap)
    assert np.array_equal(with_sup.cumulative_wht_memmap, without.cumulative_wht_memmap)
