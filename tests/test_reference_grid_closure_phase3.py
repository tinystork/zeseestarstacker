"""GAR-06 Phase 3 acceptance: REFERENCE_GRID_LIFETIME_ACCEPT.

The reference/output grid caches live for exactly one accepted run: a fresh
accepted run resets them, a refused concurrent Start mutates nothing, same-run
retries are idempotent, and a full-grid identity guard (separate from the
scalar PSR guard) fails closed on any same-run geometry drift.
"""

import math
import threading
from types import MethodType

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import seestar.queuep.queue_manager as qm
from seestar.core.drizzle_core import (
    DrizzleGeometryError,
    build_output_grid,
)


# ---------------------------------------------------------------------------
# full-grid identity guard (separate from the scalar PSR guard)
# ---------------------------------------------------------------------------


def _rot_wcs(angle=37.0, shape=(32, 32), plate=4.4e-4):
    w = WCS(naxis=2)
    w.wcs.crpix = [(shape[1] + 1) / 2.0, (shape[0] + 1) / 2.0]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    a = math.radians(angle)
    w.wcs.pc = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    w.wcs.cdelt = [-plate, plate]
    w.array_shape = shape
    return w


@pytest.mark.parametrize("scale", [1.0, 2.0, 3.0])
def test_same_psr_orientation_mutation_is_rejected(scale):
    obj = object.__new__(qm.SeestarQueuedStacker)
    obj.drizzle_scale = scale
    obj.reference_wcs_object = _rot_wcs()
    obj.drizzle_output_wcs = build_output_grid(obj.reference_wcs_object, (32, 32), scale)[0]
    obj.drizzle_output_shape_hw = (int(32 * scale), int(32 * scale))
    first = obj._freeze_drizzle_geometry()
    assert first is not None

    # rotate the output grid to axis-aligned: identical angular pixel scale
    # (so PSR and the scalar guard are unchanged) but a materially different
    # effective matrix and orientation.
    obj.drizzle_output_wcs.wcs.pc = np.eye(2)
    with pytest.raises(DrizzleGeometryError):
        obj._freeze_drizzle_geometry()


def test_same_run_retry_is_idempotent():
    obj = object.__new__(qm.SeestarQueuedStacker)
    obj.drizzle_scale = 2.0
    obj.reference_wcs_object = _rot_wcs()
    obj.drizzle_output_wcs = build_output_grid(obj.reference_wcs_object, (32, 32), 2.0)[0]
    obj.drizzle_output_shape_hw = (64, 64)
    first = obj._freeze_drizzle_geometry()
    second = obj._freeze_drizzle_geometry()
    assert first == second


def _zpn_wcs(shape=(32, 32)):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---ZPN", "DEC--ZPN"]
    w.wcs.crpix = [(shape[1] + 1) / 2.0, (shape[0] + 1) / 2.0]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.cdelt = [-0.001, 0.001]
    w.wcs.set_pv([(2, 0, 0.0), (2, 1, 1.0)])
    w.array_shape = shape
    w.pixel_shape = (shape[1], shape[0])
    return w


def test_pv_projection_drift_is_rejected_and_idempotent():
    """F8: a changed projection parameter (PV) must fail closed even though
    the angular pixel scale (PSR) is unchanged; repeated resolution stays
    idempotent."""
    ref = _zpn_wcs()
    obj = object.__new__(qm.SeestarQueuedStacker)
    obj.drizzle_scale = 2.0
    obj.reference_wcs_object = ref
    obj.drizzle_output_wcs = build_output_grid(ref, (32, 32), 2.0)[0]
    obj.drizzle_output_shape_hw = (64, 64)
    first = obj._freeze_drizzle_geometry()
    assert first is not None
    assert obj._freeze_drizzle_geometry() == first

    ref.wcs.set_pv([(2, 0, 0.0), (2, 1, 1.1)])
    with pytest.raises(DrizzleGeometryError):
        obj._freeze_drizzle_geometry()


def test_canonical_projection_params_full_identity():
    canon = qm._canonical_projection_params
    assert canon("pv", None) == ()
    assert canon("pv", []) == ()
    # Astropy triples are canonicalized with full (axis, index, value) identity,
    # order-independently.
    assert canon("pv", [(2, 1, 1.0), (2, 0, 0.0)]) == ((2, 0, 0.0), (2, 1, 1.0))
    assert canon("pv", [(2, 0, 0.0), (2, 1, 1.0)]) == canon(
        "pv", [(2, 1, 1.0), (2, 0, 0.0)]
    )
    # PS string values are preserved.
    assert canon("ps", [(1, 0, "A"), (2, 0, "B")]) == ((1, 0, "A"), (2, 0, "B"))
    # Malformed entries map to a distinct non-empty sentinel, never to "absent".
    malformed = canon("pv", [object()])
    assert malformed and malformed != ()
    assert canon("pv", [(2, 1, 1.0, 9)]) == (("__malformed__", "pv"),)
    # Non-finite values normalize deterministically.
    assert canon("pv", [(2, 1, float("nan"))]) == canon(
        "pv", [(2, 1, float("nan"))
    ])


def test_frame_facts_include_pv_and_ps():
    class _Prm:
        ctype = ("RA---ZPN", "DEC--ZPN")
        cunit = ("deg", "deg")
        radesys = "ICRS"
        equinox = 2000.0
        lonpole = 180.0
        latpole = 20.0
        crval = (10.0, 20.0)
        crpix = (16.5, 16.5)

        def get_pv(self):
            return [(2, 0, 0.0), (2, 1, 1.0)]

        def get_ps(self):
            return [(1, 0, "A")]

    class _W:
        wcs = _Prm()
        array_shape = (32, 32)
        pixel_shape = (32, 32)

    obj = object.__new__(qm.SeestarQueuedStacker)
    obj.drizzle_scale = 1.0
    obj._frozen_reference = None
    snap = obj._drizzle_grid_identity_snapshot(_W(), _W(), 1.0, 1.0)
    ref_frame = dict(snap["reference_frame"])
    assert ref_frame["pv"] == ((2, 0, 0.0), (2, 1, 1.0))
    assert ref_frame["ps"] == ((1, 0, "A"),)


def _grid_subject(scale=2.0):
    obj = object.__new__(qm.SeestarQueuedStacker)
    obj.drizzle_scale = scale
    obj.reference_wcs_object = _rot_wcs()
    obj.drizzle_output_wcs = build_output_grid(obj.reference_wcs_object, (32, 32), scale)[0]
    obj.drizzle_output_shape_hw = (int(32 * scale), int(32 * scale))
    return obj


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: setattr(o.reference_wcs_object.wcs, "radesys", "FK5"),
        lambda o: setattr(o.reference_wcs_object, "array_shape", (31, 32)),
        lambda o: setattr(o.drizzle_output_wcs.wcs, "radesys", "FK5"),
        lambda o: setattr(o.reference_wcs_object.wcs, "lonpole", 10.0),
        lambda o: setattr(o.drizzle_output_wcs.wcs, "equinox", 2000.0),
    ],
)
def test_frame_and_shape_mutations_are_rejected(mutate):
    """F3: frame metadata and reference *shape* drift must fail closed, not
    only CTYPE/matrix/PSR."""
    obj = _grid_subject()
    obj._freeze_drizzle_geometry()
    mutate(obj)
    with pytest.raises(DrizzleGeometryError):
        obj._freeze_drizzle_geometry()


def test_repeated_equivalent_resolution_is_idempotent():
    obj = _grid_subject()
    first = obj._freeze_drizzle_geometry()
    assert obj._freeze_drizzle_geometry() == first
    # equivalent re-resolution from the same reference/scale stays stable
    obj.drizzle_output_wcs = build_output_grid(obj.reference_wcs_object, (32, 32), 2.0)[0]
    assert obj._freeze_drizzle_geometry() == first


def test_reference_geometry_drift_is_rejected():
    obj = object.__new__(qm.SeestarQueuedStacker)
    obj.drizzle_scale = 2.0
    obj.reference_wcs_object = _rot_wcs()
    obj.drizzle_output_wcs = build_output_grid(obj.reference_wcs_object, (32, 32), 2.0)[0]
    obj.drizzle_output_shape_hw = (64, 64)
    obj._freeze_drizzle_geometry()
    # same scale / same output grid, but the reference CRVAL moved
    obj.reference_wcs_object.wcs.crval = [11.0, 21.0]
    with pytest.raises(DrizzleGeometryError):
        obj._freeze_drizzle_geometry()


# ---------------------------------------------------------------------------
# per-run reset
# ---------------------------------------------------------------------------


class _Dummy:
    pass


def _reset_subject():
    obj = _Dummy()
    obj.fixed_output_wcs = object()
    obj.fixed_output_shape = (1, 1)
    obj.drizzle_output_wcs = object()
    obj.drizzle_output_shape_hw = (1, 1)
    obj.reference_wcs_object = object()
    obj.reference_header_for_wcs = object()
    obj.ref_wcs_header = object()
    obj.reference_pixel_scale_arcsec = 1.0
    obj._drizzle_grid_identity = {"x": 1}
    obj._drizzle_resume_result = object()
    obj._drizzle_resume_continuation = object()
    obj._frozen_reference = object()
    obj._resolved_reference_origin = "AUTO_GEOMETRY"
    obj._automatic_reference_resolution_count = 1
    obj._reference_geometry_stats = {"x": 1}
    obj._resume_requested = False
    obj.aligner = _Dummy()
    obj.aligner.reference_image_path = "/x"
    for name in ("_reset_run_geometry", "_clear_frozen_reference"):
        setattr(obj, name, MethodType(getattr(qm.SeestarQueuedStacker, name), obj))
    return obj


def test_fresh_run_reset_clears_every_geometry_carrier():
    obj = _reset_subject()
    obj._reset_run_geometry()
    assert obj.fixed_output_wcs is None
    assert obj.fixed_output_shape is None
    assert obj.drizzle_output_wcs is None
    assert obj.drizzle_output_shape_hw is None
    assert obj.reference_wcs_object is None
    assert obj._drizzle_grid_identity is None
    assert obj._frozen_reference is None
    assert obj._drizzle_resume_result is None
    assert obj._automatic_reference_resolution_count == 0


def test_resume_reset_preserves_persisted_checkpoint_result():
    obj = _reset_subject()
    obj._resume_requested = True
    persisted = obj._drizzle_resume_result
    obj._reset_run_geometry()
    assert obj.fixed_output_wcs is None
    assert obj.drizzle_output_wcs is None
    assert obj._drizzle_grid_identity is None
    # the persisted checkpoint result must survive so it can be restored
    assert obj._drizzle_resume_result is persisted


def test_resume_reset_clears_prior_reference_owner(tmp_path):
    """F2: a resume accepted on a reused stacker must clear the previous run's
    frozen reference owner, then freeze the checkpoint-resolved RESUME identity
    without a conflict, while preserving the persisted checkpoint result."""
    st = qm.SeestarQueuedStacker(batch_size=1, autotune=False)
    st.update_progress = lambda *a, **k: None
    old = tmp_path / "old.fit"
    new = tmp_path / "new.fit"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    st._freeze_reference(str(old), "USER")
    assert st._frozen_reference.source_basename == "old.fit"

    st._resume_requested = True
    st._drizzle_resume_result = "sentinel-result"
    st._reset_run_geometry()
    assert st._frozen_reference is None
    assert st._drizzle_resume_result == "sentinel-result"
    frozen = st._freeze_reference(str(new), "RESUME")
    assert frozen.origin == "RESUME"
    assert frozen.source_basename == "new.fit"


def test_refused_concurrent_start_mutates_nothing():
    st = qm.SeestarQueuedStacker(batch_size=1, autotune=False)
    st.processing_active = True
    st.fixed_output_wcs = "sentinel-fixed"
    st.drizzle_output_wcs = "sentinel-drizzle"
    st.drizzle_output_shape_hw = (7, 9)
    st._frozen_reference = "sentinel-frozen"
    started = st.start_processing(
        "/nonexistent-input", "/nonexistent-output",
        use_drizzle=True, drizzle_scale=2, drizzle_kernel="lanczos2",
    )
    assert started is False
    assert st.fixed_output_wcs == "sentinel-fixed"
    assert st.drizzle_output_wcs == "sentinel-drizzle"
    assert st.drizzle_output_shape_hw == (7, 9)
    assert st._frozen_reference == "sentinel-frozen"


# ---------------------------------------------------------------------------
# reused-stacker integration: fresh run rebuilds the grid
# ---------------------------------------------------------------------------


class _NoopExecutor:
    def __init__(self, max_workers=1, **kwargs):
        self._max_workers = max_workers

    def shutdown(self, *args, **kwargs):
        pass


def _write_fits(path, ra, shape=(32, 40)):
    w = WCS(naxis=2)
    w.wcs.crpix = [(shape[1] + 1) / 2.0, (shape[0] + 1) / 2.0]
    w.wcs.crval = [ra, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-0.001, 0.001]
    header = w.to_header()
    header["EXPTIME"] = 20.0
    fits.PrimaryHDU(np.ones(shape, np.float32), header).writeto(str(path))


def _run_once(st, root, out, request, scale, selector_calls):
    def choose(*args, **kwargs):
        selector_calls.append("AUTO")
        from seestar.core.geometry_reference import GeometrySelection, ResolvedReference

        return GeometrySelection(ResolvedReference(str(root / "A.fit"), "AUTO_GEOMETRY"))

    import seestar.queuep.queue_manager as _qm

    orig = _qm.select_geometry_reference
    _qm.select_geometry_reference = choose
    try:
        snapshot = {}
        done = threading.Event()

        def worker():
            frozen = st._consume_frozen_reference_for_worker()
            snapshot.update(
                source=frozen.source_basename,
                origin=frozen.origin,
                grid_crval=list(st.drizzle_output_wcs.wcs.crval),
                shape=tuple(st.drizzle_output_shape_hw),
                fixed_is_drizzle=st.fixed_output_wcs is st.drizzle_output_wcs,
                fixed_shape=tuple(st.fixed_output_shape),
            )
            st.processing_active = False
            done.set()

        st._worker = worker
        started = st.start_processing(
            str(root), str(out), reference_path_ui=request,
            use_drizzle=True, drizzle_scale=scale, drizzle_kernel="lanczos2",
            batch_size=1, correct_hot_pixels=False, perform_cleanup=False,
            move_stacked=False, reproject_between_batches=False,
            reproject_coadd_final=False,
        )
        if st.processing_thread is not None:
            st.processing_thread.join(10)
        return started, done, snapshot
    finally:
        _qm.select_geometry_reference = orig


def test_drizzle_grid_provenance_record_once(caplog):
    """F6: one compact, computed DRIZZLE_GRID provenance record per resolution."""
    import logging

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    qm.logger.addHandler(handler)
    old_level = qm.logger.level
    qm.logger.setLevel(logging.INFO)
    try:
        obj = _grid_subject()
        obj._reference_requested_raw = "/in/B.fit"
        obj._freeze_drizzle_geometry()
        obj._freeze_drizzle_geometry()  # idempotent: no second record
    finally:
        qm.logger.removeHandler(handler)
        qm.logger.setLevel(old_level)

    grid = [m for m in records if m.startswith("DRIZZLE_GRID")]
    assert len(grid) == 1
    line = grid[0]
    assert "builder=seestar.core.drizzle_core.build_output_grid" in line
    assert "contract=m3_output_grid_v2 v2" in line
    assert "crpix_convention=fits_edge_centre_v1" in line
    assert "orientation_preserved=True" in line
    assert "requested_manual='/in/B.fit'" in line


def test_drizzle_grid_resume_provenance_record(caplog):
    import logging

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    qm.logger.addHandler(handler)
    old_level = qm.logger.level
    qm.logger.setLevel(logging.INFO)
    try:
        obj = _grid_subject()
        obj._drizzle_grid_provenance_emitted = True
        obj._emit_drizzle_grid_provenance("resume")
    finally:
        qm.logger.removeHandler(handler)
        qm.logger.setLevel(old_level)
    resume = [m for m in records if m.startswith("DRIZZLE_GRID kind=resume")]
    assert len(resume) == 1


def test_reused_stacker_rebuilds_grid_for_new_reference_and_scale(tmp_path):
    root = tmp_path / "input"
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    root.mkdir(parents=True)
    out1.mkdir()
    out2.mkdir()
    _write_fits(root / "A.fit", 275.0)
    _write_fits(root / "B.fit", 276.0)

    import seestar.queuep.queue_manager as _qm

    orig_exec = _qm.ProcessPoolExecutor
    _qm.ProcessPoolExecutor = _NoopExecutor
    try:
        st = _qm.SeestarQueuedStacker(batch_size=1, autotune=False)
        st.update_progress = lambda *a, **k: None
        st._solve_astrometry_async = lambda path, header, settings, **kwargs: WCS(header).celestial
        calls = {}

        started, done, snap = _run_once(st, root, out1, str(root / "B.fit"), 2, [])
        assert started and done.is_set()
        assert snap["source"] == "B.fit"
        assert snap["grid_crval"] == [276.0, 20.0]
        assert snap["shape"] == (64, 80)
        assert snap["fixed_is_drizzle"] is True

        started2, done2, snap2 = _run_once(st, root, out2, str(root / "A.fit"), 3, [])
        refusal = getattr(st, "startup_refusal", None)
        assert started2 and done2.is_set(), (
            getattr(refusal, "code", None),
            getattr(refusal, "technical_detail", None),
            getattr(st, "processing_error", None),
            getattr(st, "processing_active", None),
        )
        assert snap2["source"] == "A.fit"
        assert snap2["grid_crval"] == [275.0, 20.0]
        assert snap2["shape"] == (96, 120)
        assert snap2["fixed_is_drizzle"] is True
    finally:
        _qm.ProcessPoolExecutor = orig_exec
