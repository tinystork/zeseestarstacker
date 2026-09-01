"""COV-01D: per-original-exposure positive support through Reproject."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from seestar.queuep.queue_manager import (
    SeestarQueuedStacker,
    _ResumeCheckpointError,
)


def _wcs(shape=(5, 6), *, dx=0.0, dy=0.0):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2 + dx, shape[0] / 2 + dy]
    w.wcs.cdelt = np.array([-1.0 / 3600.0, 1.0 / 3600.0])
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


def _bare(tmp_path):
    q = object.__new__(SeestarQueuedStacker)
    q.output_folder = str(tmp_path)
    q.coverage_sup_w1_memmap = None
    q.coverage_sup_w2_memmap = None
    q._support_state_available = False
    q._support_unavailable_reason = None
    q._reproject_support_tracking_enabled = True
    q._reproject_support_sidecars = {}
    q.reproject_between_batches = False
    q.reproject_coadd_final = False
    return q


def test_fractional_transform_is_positive_and_footprint_gated(tmp_path):
    q = _bare(tmp_path)
    mask = np.zeros((5, 6), dtype=bool)
    mask[1:4, 1:4] = True
    out = q._reproject_positive_support(
        mask, 2.0, _wcs(), _wcs(dx=0.5), mask.shape
    )
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 2.0))
    assert np.any((out > 0.0) & (out < 2.0))


def test_square_happens_after_each_exposure_transform(tmp_path):
    q = _bare(tmp_path)
    q.coverage_sup_w1_memmap = np.zeros((5, 6), dtype=np.float64)
    q.coverage_sup_w2_memmap = np.zeros((5, 6), dtype=np.float64)
    q._support_state_available = True
    mask = np.zeros((5, 6), dtype=bool)
    mask[1:4, 1:4] = True
    payload = q._build_reproject_support_payload(
        [mask], [2.0], [_wcs()], _wcs(dx=0.5), mask.shape
    )
    entry = payload[0]
    transformed = q._reproject_positive_support(
        entry["mask"],
        entry["scalar"],
        entry["input_wcs"],
        entry["target_wcs"],
        entry["target_shape"],
    )
    q._validate_support_payload(payload, mask.shape, 1)
    q._apply_support_payload(payload)
    assert np.array_equal(q.coverage_sup_w1_memmap, transformed)
    assert np.array_equal(q.coverage_sup_w2_memmap, transformed**2)


def test_reproject_decomposition_is_array_exact(tmp_path):
    rng = np.random.default_rng(4)
    masks = [rng.random((5, 6)) > 0.4 for _ in range(11)]
    scalars = [1.0 + i / 10.0 for i in range(11)]
    source_wcs = [_wcs(dx=i / 13.0) for i in range(11)]
    target = _wcs(dx=0.25)

    payload = _bare(tmp_path)._build_reproject_support_payload(
        masks, scalars, source_wcs, target, (5, 6)
    )

    def accumulate(partitions):
        q = _bare(tmp_path)
        q.coverage_sup_w1_memmap = np.zeros((5, 6), dtype=np.float64)
        q.coverage_sup_w2_memmap = np.zeros((5, 6), dtype=np.float64)
        q._support_state_available = True
        for lo, hi in partitions:
            q._apply_support_payload(payload[lo:hi])
        return q.coverage_sup_w1_memmap.copy(), q.coverage_sup_w2_memmap.copy()

    all_at_once = accumulate([(0, 11)])
    partitioned = accumulate([(0, 3), (3, 7), (7, 11)])
    singletons = accumulate([(i, i + 1) for i in range(11)])
    assert np.array_equal(all_at_once[0], partitioned[0])
    assert np.array_equal(all_at_once[1], partitioned[1])
    assert np.array_equal(all_at_once[0], singletons[0])
    assert np.array_equal(all_at_once[1], singletons[1])


def test_between_batch_stage_requires_per_exposure_wcs(tmp_path):
    q = _bare(tmp_path)
    q.reproject_between_batches = True
    q.reference_wcs_object = _wcs()
    q.memmap_shape = (5, 6, 3)
    with pytest.raises(_ResumeCheckpointError, match="celestial WCS"):
        q._stage_batch_support(
            [np.ones((5, 6), dtype=bool)], [1.0], [None]
        )


def test_between_batch_stage_accepts_numpy_quality_scalars(tmp_path):
    q = _bare(tmp_path)
    q.reproject_between_batches = True
    q.reference_wcs_object = _wcs()
    q.memmap_shape = (5, 6, 3)
    payload, sources = q._stage_batch_support(
        [np.ones((5, 6), dtype=bool)],
        np.asarray([1.25], np.float32),
        [_wcs()],
    )
    assert sources is None
    assert len(payload) == 1
    assert payload[0]["kind"] == "reproject_support_source"
    q._validate_support_payload(payload, (5, 6), 1)


def test_final_coadd_sidecars_reproject_original_exposures(tmp_path):
    q = _bare(tmp_path)
    q.reproject_coadd_final = True
    out_dir = tmp_path / "classic_batch_outputs"
    out_dir.mkdir()
    sci1 = out_dir / "classic_batch_001.fits"
    sci2 = out_dir / "classic_batch_002.fits"
    fits.writeto(sci1, np.zeros((3, 5, 6), np.float32), overwrite=True)
    fits.writeto(sci2, np.zeros((3, 5, 6), np.float32), overwrite=True)

    masks = []
    records = []
    for idx, (sci, dx, scalar) in enumerate(
        ((sci1, 0.0, 1.0), (sci2, 0.4, 2.0))
    ):
        mask = np.zeros((5, 6), dtype=bool)
        mask[1 + idx : 4 + idx, 1:5] = True
        masks.append((mask, scalar, _wcs(dx=dx)))
        payload = q._build_reproject_support_source_payload(
            [mask], [scalar], [_wcs(dx=dx)]
        )
        q._persist_reproject_support_sidecar(str(sci), payload, 1)
        records.append((str(sci), []))

    target = _wcs(dx=0.2)
    # Expected values use the durable source records (the exact inputs the
    # finalizer replays), not the pre-serialization in-memory WCS objects.
    expected_maps = []
    for sci, _wht in records:
        for mask, scalar, source in q._load_reproject_support_sidecar(sci):
            expected_maps.append(
                q._reproject_positive_support(mask, scalar, source, target, (5, 6))
            )
    q._finalize_reproject_support(records, target, (5, 6))
    assert np.array_equal(q.coverage_sup_w1_memmap, sum(expected_maps))
    assert np.array_equal(
        q.coverage_sup_w2_memmap, sum(support**2 for support in expected_maps)
    )


def test_final_coadd_missing_sidecar_fails_before_publication(tmp_path):
    q = _bare(tmp_path)
    q.reproject_coadd_final = True
    sci = tmp_path / "classic_batch_001.fits"
    fits.writeto(sci, np.zeros((3, 5, 6), np.float32), overwrite=True)
    with pytest.raises(_ResumeCheckpointError, match="cannot read"):
        q._finalize_reproject_support([(str(sci), [])], _wcs(), (5, 6))
    memdir = Path(tmp_path) / "memmap_accumulators"
    assert not (memdir / "coverage_SUP_W1.npy").exists()
    assert not (memdir / "coverage_SUP_W2.npy").exists()


def test_final_coadd_half_publish_is_rolled_back(tmp_path, monkeypatch):
    q = _bare(tmp_path)
    q.reproject_coadd_final = True
    sci = tmp_path / "classic_batch_001.fits"
    fits.writeto(sci, np.zeros((3, 5, 6), np.float32), overwrite=True)
    payload = q._build_reproject_support_source_payload(
        [np.ones((5, 6), dtype=bool)], [1.0], [_wcs()]
    )
    q._persist_reproject_support_sidecar(str(sci), payload, 1)

    import os

    real_replace = os.replace
    calls = {"n": 0}

    def fail_second_support_publish(src, dst):
        if Path(dst).name in {"coverage_SUP_W1.npy", "coverage_SUP_W2.npy"}:
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("injected second publish failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_second_support_publish)
    with pytest.raises(OSError, match="injected"):
        q._finalize_reproject_support([(str(sci), [])], _wcs(), (5, 6))
    memdir = Path(tmp_path) / "memmap_accumulators"
    assert not (memdir / "coverage_SUP_W1.npy").exists()
    assert not (memdir / "coverage_SUP_W2.npy").exists()
    assert q.coverage_sup_w1_memmap is None
    assert q.coverage_sup_w2_memmap is None


def test_support_sidecar_rejects_mask_scalar_cardinality(tmp_path):
    q = _bare(tmp_path)
    q.reproject_coadd_final = True
    sci = tmp_path / "classic_batch_001.fits"
    mask = np.ones((5, 6), dtype=bool)
    payload = q._build_reproject_support_source_payload(
        [mask], [1.0], [_wcs()]
    )
    with pytest.raises(_ResumeCheckpointError, match="cardinality"):
        q._persist_reproject_support_sidecar(str(sci), payload, 2)


def test_support_sidecar_rejects_silent_mask_dtype_coercion(tmp_path):
    q = _bare(tmp_path)
    q.reproject_coadd_final = True
    sci = tmp_path / "classic_batch_001.fits"
    sidecar = tmp_path / "classic_batch_001_support.npz"
    header_text = _wcs().to_header(relax=True).tostring(
        sep="\n", endcard=False, padding=False
    )
    np.savez_compressed(
        sidecar,
        schema=np.asarray("reproject_sup_v1"),
        masks=np.ones((1, 5, 6), dtype=np.float32),
        scalars=np.ones(1, dtype=np.float64),
        wcs_headers=np.asarray([header_text], dtype=np.str_),
    )
    with pytest.raises(_ResumeCheckpointError, match="mask dtype"):
        q._load_reproject_support_sidecar(str(sci))
