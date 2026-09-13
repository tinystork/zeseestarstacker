"""Rework-3 (F8/SIP): exact SIP geometry support across the closure mission.

Covers the physical-witness failure mode: a frozen USER reference whose
solved WCS carries ASTAP-style TAN-SIP distortion must build an exactly-scaled
output grid (not be refused), flow through the real production start seam to
initialized accumulators, be guarded by the full-grid identity snapshot, and
round-trip through the native Drizzle checkpoint.
"""

import math

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS, Sip

import seestar.queuep.queue_manager as qm
from seestar.core.drizzle_core import (
    DrizzleAccumulator,
    DrizzleGeometryError,
    build_output_grid,
    pixmap_from_alignment,
)
from seestar.core.drizzle_checkpoint import (
    DrizzleCheckpointWriter,
    build_drizzle_canonical_config,
    read_drizzle_checkpoint,
    reconstruct_input_reference_wcs,
    serialize_input_reference_geometry,
)

REF_SHAPE = (60, 100)
SCALES = [1, 2, 3, 4]


def _sip_arrays():
    a = np.zeros((3, 3), dtype=float)
    b = np.zeros((3, 3), dtype=float)
    a[1, 0] = 2e-5
    a[2, 0] = 1e-8
    a[0, 2] = -1e-7
    b[1, 0] = -1e-5
    b[2, 1] = 2e-8
    b[0, 1] = 3e-7
    return a, b


def _sip_reference(shape=REF_SHAPE, angle=37.0, encoding="cd", inverse=False,
                   crpix=(41.3, 22.7), crval=(275.0, 30.0)):
    h, w = shape
    ref = WCS(naxis=2)
    ref.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    ref.wcs.crpix = list(crpix)
    ref.wcs.crval = list(crval)
    ref.wcs.cunit = ["deg", "deg"]
    theta = math.radians(angle)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    if encoding == "cd":
        ref.wcs.cd = np.diag([-0.001, 0.001]) @ rot
    else:
        ref.wcs.pc = rot
        ref.wcs.cdelt = [-0.001, 0.001]
    a, b = _sip_arrays()
    if inverse:
        ap = np.zeros((2, 2), dtype=float)
        bp = np.zeros((2, 2), dtype=float)
        ap[1, 0] = 1e-5
        bp[1, 0] = -1e-5
    else:
        ap = np.zeros((1, 1), dtype=float)
        bp = np.zeros((1, 1), dtype=float)
    ref.sip = Sip(a, b, ap, bp, np.asarray(ref.wcs.crpix))
    ref.array_shape = shape
    ref.pixel_shape = (w, h)
    return ref


def _expected_out(pts, scale):
    return scale * np.asarray(pts) + (scale - 1.0) / 2.0


def _sample_points(shape=REF_SHAPE, n=400):
    h, w = shape
    rng = np.random.default_rng(7)
    return rng.uniform([-5.0, -5.0], [w + 5.0, h + 5.0], size=(n, 2))


# ---------------------------------------------------------------------------
# gate 2: exact scaled SIP copy (mapping equality, metadata, no mutation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("encoding", ["cd", "pc"])
@pytest.mark.parametrize("inverse", [False, True])
def test_sip_scaled_copy_mapping_and_metadata(scale, encoding, inverse):
    ref = _sip_reference(encoding=encoding, inverse=inverse)
    a_before = ref.sip.a.copy()
    b_before = ref.sip.b.copy()

    out, out_shape = build_output_grid(ref, REF_SHAPE, scale)

    assert out.sip is not None
    assert out.sip.a_order == ref.sip.a_order  # order preserved
    # the distortions are retained, not erased
    assert np.any(out.sip.a != 0.0)
    assert out.sip.a_order == ref.sip.a_order
    assert out.sip.b_order == ref.sip.b_order
    # inverse presence preserved
    assert (out.sip.ap_order > 0) == (ref.sip.ap_order > 0)
    # reference is untouched (independent copy)
    assert np.array_equal(ref.sip.a, a_before) and np.array_equal(ref.sip.b, b_before)

    assert out_shape == (REF_SHAPE[0] * scale, REF_SHAPE[1] * scale)
    assert out.array_shape == out_shape
    assert out.pixel_shape == (out_shape[1], out_shape[0])
    assert np.allclose(out.pixel_scale_matrix, ref.pixel_scale_matrix / scale)
    assert out.wcs.crpix[0] == pytest.approx(scale * (ref.wcs.crpix[0] - 0.5) + 0.5)

    pts = _sample_points()
    sky = ref.all_pix2world(pts, 0)
    remapped = out.all_world2pix(sky, 0)
    residual = np.max(np.abs(remapped - _expected_out(pts, scale)))
    assert residual < 1e-6, residual


def test_sip_scale_one_is_identity():
    ref = _sip_reference()
    out, _ = build_output_grid(ref, REF_SHAPE, 1)
    assert np.allclose(out.sip.a, ref.sip.a)
    assert np.allclose(out.sip.b, ref.sip.b)
    pts = _sample_points()
    residual = np.max(np.abs(out.all_world2pix(ref.all_pix2world(pts, 0), 0) - pts))
    assert residual < 1e-9


@pytest.mark.parametrize("scale", SCALES)
def test_sip_perimeter_and_centre_containment(scale):
    ref = _sip_reference()
    h, w = REF_SHAPE
    out, out_shape = build_output_grid(ref, REF_SHAPE, scale)
    tf = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    _pixmap, mask = pixmap_from_alignment((h, w), tf, ref, out)
    assert mask.all()
    # independent perimeter projection into the output canvas bounds
    xs = np.linspace(-0.5, w - 0.5, 30)
    ys = np.linspace(-0.5, h - 0.5, 30)
    edges = [(x, -0.5) for x in xs] + [(x, h - 0.5) for x in xs]
    edges += [(-0.5, y) for y in ys] + [(w - 0.5, y) for y in ys]
    sky = ref.all_pix2world(np.array(edges), 0)
    px = out.all_world2pix(sky, 0)
    assert px[:, 0].min() >= -0.5 - 1e-6
    assert px[:, 0].max() <= out_shape[1] - 0.5 + 1e-6
    assert px[:, 1].min() >= -0.5 - 1e-6
    assert px[:, 1].max() <= out_shape[0] - 0.5 + 1e-6


@pytest.mark.parametrize("kernel", ["square", "lanczos2"])
def test_sip_tiny_deposition(kernel):
    ref = _sip_reference(shape=(8, 10))
    out, out_shape = build_output_grid(ref, (8, 10), 2)
    tf = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pixmap, mask = pixmap_from_alignment((8, 10), tf, ref, out)
    assert mask.all()
    data = np.full((8, 10), 4.0, np.float32)
    acc = DrizzleAccumulator(out_shape, kernel=kernel, pixfrac=1.0)
    acc.add(data, np.ones((8, 10), np.float32), pixmap, in_grid_mask=mask)
    sci = acc.finalize()
    assert np.isfinite(sci).all() and sci.sum() > 0.0


# ---------------------------------------------------------------------------
# gate 5: remaining unsupported distortions stay clearly refused
# ---------------------------------------------------------------------------


def test_lookup_and_det2im_distortions_refused():
    class _Prm:
        cpdis1 = object()

    class _Lookup:
        is_celestial = True
        pixel_shape = (100, 60)
        sip = None
        wcs = _Prm()

    with pytest.raises(ValueError):
        build_output_grid(_Lookup(), REF_SHAPE, 2)

    class _Det:
        is_celestial = True
        pixel_shape = (100, 60)
        sip = None
        det2im1 = object()
        wcs = None

    with pytest.raises(ValueError):
        build_output_grid(_Det(), REF_SHAPE, 2)


def test_sip_origin_mismatch_refused():
    ref = _sip_reference()
    ref.sip = Sip(
        *_sip_arrays(),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.asarray(ref.wcs.crpix) + 3.0,
    )
    with pytest.raises(ValueError):
        build_output_grid(ref, REF_SHAPE, 2)


# ---------------------------------------------------------------------------
# gate 1: real production start seam with a SIP-bearing frozen USER reference
# ---------------------------------------------------------------------------


class _NoopExecutor:
    def __init__(self, max_workers=1, **kwargs):
        self._max_workers = max_workers

    def shutdown(self, *args, **kwargs):
        pass


def _write_sip_fits(path, shape=(32, 40), ra=276.0):
    w = _sip_reference(shape=shape, crpix=(16.5, 20.5), crval=(ra, 20.0))
    header = w.to_header(relax=True)
    header["EXPTIME"] = 20.0
    fits.PrimaryHDU(np.ones(shape, np.float32), header).writeto(str(path))


def _sip_grid_subject(scale=2.0):
    obj = object.__new__(qm.SeestarQueuedStacker)
    obj.drizzle_scale = scale
    ref = _sip_reference()
    obj.reference_wcs_object = ref
    obj.drizzle_output_wcs = build_output_grid(ref, REF_SHAPE, scale)[0]
    obj.drizzle_output_shape_hw = (REF_SHAPE[0] * scale, REF_SHAPE[1] * scale)
    return obj


def test_sip_identity_drift_rejected_and_idempotent():
    obj = _sip_grid_subject()
    first = obj._freeze_drizzle_geometry()
    assert first is not None
    assert obj._freeze_drizzle_geometry() == first
    # same-PSR input SIP coefficient drift must be rejected
    obj.reference_wcs_object.sip.a[1, 0] *= 1.5
    with pytest.raises(DrizzleGeometryError):
        obj._freeze_drizzle_geometry()


def test_sip_output_coefficient_and_presence_drift_rejected():
    obj = _sip_grid_subject()
    obj._freeze_drizzle_geometry()
    obj.drizzle_output_wcs.sip.b[1, 0] *= 2.0
    with pytest.raises(DrizzleGeometryError):
        obj._freeze_drizzle_geometry()

    obj2 = _sip_grid_subject()
    obj2._freeze_drizzle_geometry()
    # replacing forward-only SIP with an inverse-bearing SIP is a mutation
    ref = obj2.drizzle_output_wcs
    ap = np.zeros((2, 2))
    ap[1, 0] = 1e-5
    bp = np.zeros((2, 2))
    bp[1, 0] = -1e-5
    ref.sip = Sip(ref.sip.a.copy(), ref.sip.b.copy(), ap, bp, np.asarray(ref.wcs.crpix))
    with pytest.raises(DrizzleGeometryError):
        obj2._freeze_drizzle_geometry()


def _fake_qm(kernel="square"):
    class _Qm:
        pass

    m = _Qm()
    m.weighting_method = "none"
    m.use_quality_weighting = False
    m.weight_by_snr = True
    m.weight_by_stars = True
    m.snr_exponent = 1.0
    m.stars_exponent = 0.5
    m.min_weight = 0.01
    m.correct_hot_pixels = True
    m.hot_pixel_threshold = 3.0
    m.neighborhood_size = 5
    m.bayer_pattern = "GRBG"
    m.drizzle_scale = 1.0
    m.drizzle_kernel = kernel
    m.drizzle_pixfrac = 1.0
    m.drizzle_wht_threshold_effective = 0.0
    m.drizzle_fillval = "0.0"
    return m


def _identity(path):
    import os

    st = os.stat(path)
    return {
        "path": os.path.normcase(str(path)),
        "name": os.path.basename(str(path)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _frames(shape, n):
    h, w = shape
    yy, xx = np.indices(shape, dtype=np.float64)
    out = []
    for i in range(n):
        tf = np.array([[1.0, 0.0, 0.3 * i], [0.0, 1.0, -0.2 * i]])
        px = tf[0, 0] * xx + tf[0, 1] * yy + tf[0, 2]
        py = tf[1, 0] * xx + tf[1, 1] * yy + tf[1, 2]
        pixmap = np.dstack((px, py))
        mask = (
            (pixmap[..., 0] >= 0.0) & (pixmap[..., 0] < shape[1])
            & (pixmap[..., 1] >= 0.0) & (pixmap[..., 1] < shape[0])
        )
        data = (np.sin(xx / 3.0 + i) + 10.0).astype(np.float32)
        out.append((data, np.full(shape, 0.8, np.float32), pixmap, mask))
    return out


def _sip_checkpoint(tmp_path, n_sources=4, frame_count=2):
    ref_wcs = _sip_reference(shape=(24, 24), crpix=(12.5, 12.5), crval=(10.0, 20.0))
    out_wcs, out_shape = build_output_grid(
        _sip_reference(shape=(32, 32), crpix=(16.5, 16.5)), (32, 32), 1.0
    )
    cfg = build_drizzle_canonical_config(_fake_qm(), product_version="8.5.0")
    writer = DrizzleCheckpointWriter(str(tmp_path), "8.5.0", cfg, out_wcs, out_shape)

    def _new_accs():
        return [DrizzleAccumulator(out_shape, kernel="square", pixfrac=1.0) for _ in range(3)]

    accs = _new_accs()
    frames = _frames(out_shape, n_sources)
    for data, weight, pixmap, mask in frames[:frame_count]:
        for acc in accs:
            acc.add(data, weight, pixmap, exptime=1.0, in_units="counts", in_grid_mask=mask)

    from pathlib import Path

    ref_path = Path(tmp_path) / "reference.fit"
    ref_path.write_bytes(b"reference-bytes")
    src_idents = []
    for i in range(n_sources):
        p = Path(tmp_path) / f"src_{i}.fit"
        p.write_bytes(b"src-%d" % i)
        src_idents.append(_identity(p))
    binding = {
        "input_roots": [str(tmp_path)],
        "reference": _identity(ref_path),
        "plan": {"sources": src_idents, "decomposition": [n_sources]},
        "reference_geometry": serialize_input_reference_geometry(ref_wcs, (24, 24), _identity(ref_path)),
    }
    counters = {
        "frame_count": frame_count, "stacked_batches_count": frame_count,
        "total_exposure_seconds": float(frame_count), "exposure_unknown_count": 0,
        "exposure_min": 1.0, "exposure_max": 1.0,
    }
    writer.commit(accs, session_binding=binding, counters=counters,
                  completed_sources=src_idents[:frame_count])
    return {
        "ref_wcs": ref_wcs, "out_wcs": out_wcs, "out_shape": out_shape,
        "frames": frames, "src_idents": src_idents, "writer": writer,
        "_new_accs": _new_accs,
    }


def test_sip_checkpoint_roundtrip_and_continuation(tmp_path):
    ctx = _sip_checkpoint(tmp_path)
    result = read_drizzle_checkpoint(str(tmp_path))

    # output SIP persisted and mapping-equivalent
    assert result.wcs.sip is not None
    pts = _sample_points((32, 32), n=200)
    assert np.allclose(
        result.wcs.all_world2pix(ctx["out_wcs"].all_pix2world(pts, 0), 0),
        pts, atol=1e-9,
    )
    # input-reference SIP persisted and mapping-equivalent
    restored_ref = reconstruct_input_reference_wcs(result.reference_geometry)
    assert restored_ref is not None and restored_ref.sip is not None
    pts_in = _sample_points((24, 24), n=200)
    assert np.allclose(
        restored_ref.all_world2pix(ctx["ref_wcs"].all_pix2world(pts_in, 0), 0),
        pts_in, atol=1e-9,
    )

    # write/read/CONTINUE with SIP, uninterrupted vs stop/resume
    from seestar.core.drizzle_checkpoint import DrizzleCheckpointWriter as _W

    cont = _W.from_validated_result(result)
    data, weight, pixmap, mask = ctx["frames"][2]
    for acc in cont.accumulators:
        acc.add(data, weight, pixmap, exptime=1.0, in_units="counts", in_grid_mask=mask)
    cont_binding = {
        "input_roots": result.session["input_roots"],
        "reference": result.session["reference"],
        "plan": result.session["plan"],
    }
    cont.writer.commit(
        cont.accumulators, session_binding=cont_binding,
        counters={
            "frame_count": 3, "stacked_batches_count": 3,
            "total_exposure_seconds": 3.0, "exposure_unknown_count": 0,
            "exposure_min": 1.0, "exposure_max": 1.0,
        },
        completed_sources=list(result.session["plan"]["sources"][:3]),
    )
    again = read_drizzle_checkpoint(str(tmp_path))
    assert again.wcs.sip is not None

    # uninterrupted reference accumulation over all three frames
    ref_accs = ctx["_new_accs"]()
    for data, weight, pixmap, mask in ctx["frames"][:3]:
        for acc in ref_accs:
            acc.add(data, weight, pixmap, exptime=1.0, in_units="counts", in_grid_mask=mask)
    for resumed, fresh in zip(again.accumulators, ref_accs):
        assert np.array_equal(resumed._out_img, fresh._out_img)
        assert np.array_equal(resumed._out_wht, fresh._out_wht)


def test_start_processing_sip_reference_initializes_accumulators(tmp_path):
    root = tmp_path / "input"
    out = tmp_path / "output"
    root.mkdir(parents=True)
    out.mkdir(parents=True)
    _write_sip_fits(root / "A.fit", ra=275.0)
    _write_sip_fits(root / "B.fit", ra=276.0)

    def _sip_from_header(header):
        w = WCS(header)
        return w if w.is_celestial else w.celestial

    selector_calls = []

    def choose(*args, **kwargs):
        selector_calls.append("AUTO")
        from seestar.core.geometry_reference import GeometrySelection, ResolvedReference

        return GeometrySelection(ResolvedReference(str(root / "A.fit"), "AUTO_GEOMETRY"))

    import seestar.queuep.queue_manager as qm

    orig_select, orig_exec = qm.select_geometry_reference, qm.ProcessPoolExecutor
    qm.select_geometry_reference = choose
    qm.ProcessPoolExecutor = _NoopExecutor
    try:
        st = qm.SeestarQueuedStacker(batch_size=1, autotune=False)
        st.reference_origin_hint = "USER"
        st.update_progress = lambda *a, **k: None
        st._solve_astrometry_async = lambda path, header, settings, **kw: _sip_from_header(header)
        snapshot = {}
        import threading

        done = threading.Event()

        def worker():
            frozen = st._consume_frozen_reference_for_worker()
            snapshot.update(
                source=frozen.source_basename,
                origin=frozen.origin,
                accumulators_ready=bool(getattr(st, "drizzle_accumulators", None)),
                grid_sip=bool(
                    st.drizzle_output_wcs is not None
                    and getattr(st.drizzle_output_wcs, "sip", None) is not None
                ),
                shape=tuple(st.drizzle_output_shape_hw),
                crval=list(st.drizzle_output_wcs.wcs.crval),
            )
            st.processing_active = False
            done.set()

        st._worker = worker
        started = st.start_processing(
            str(root), str(out), reference_path_ui=str(root / "B.fit"),
            use_drizzle=True, drizzle_scale=2, drizzle_kernel="lanczos2",
            batch_size=1, correct_hot_pixels=False, perform_cleanup=False,
            move_stacked=False, reproject_between_batches=False,
            reproject_coadd_final=False,
        )
        if st.processing_thread is not None:
            st.processing_thread.join(10)
        st.quality_executor.shutdown()
    finally:
        qm.select_geometry_reference = orig_select
        qm.ProcessPoolExecutor = orig_exec

    assert started and done.is_set()
    assert snapshot["source"] == "B.fit"
    assert snapshot["origin"] == "USER"
    assert selector_calls == []
    assert snapshot["accumulators_ready"] is True
    assert snapshot["grid_sip"] is True
    assert snapshot["shape"] == (64, 80)
    assert snapshot["crval"] == [276.0, 20.0]
