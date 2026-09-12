"""GAR-06 Phase 1 acceptance: MANUAL_REFERENCE_AUTHORITY_ACCEPT.

Explicit manual-reference intent is the *presence* of a non-blank request,
resolved deterministically against the declared input context, or failed
closed.  It never silently falls through to AUTO, never searches arbitrary
filesystem locations, and never substitutes a generated ``stacked/`` artifact
for a fresh explicit request.
"""

import os
import threading

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import seestar.queuep.queue_manager as qm
from seestar.core.geometry_reference import (
    ReferenceResolutionError,
    resolve_explicit_reference,
)


def _write_fits(path, ra=10.0, shape=(32, 40)):
    w = WCS(naxis=2)
    w.wcs.crpix = [(shape[1] + 1) / 2.0, (shape[0] + 1) / 2.0]
    w.wcs.crval = [ra, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-0.001, 0.001]
    header = w.to_header()
    header["EXPTIME"] = 20.0
    fits.PrimaryHDU(np.ones(shape, np.float32), header).writeto(str(path))


# ---------------------------------------------------------------------------
# resolver unit coverage
# ---------------------------------------------------------------------------


def test_absolute_request_resolves_canonically(tmp_path):
    target = tmp_path / "B.fit"
    _write_fits(target)
    resolved = resolve_explicit_reference(str(target), search_roots=[str(tmp_path)])
    assert resolved == os.path.realpath(str(target))


def test_relative_request_resolves_against_input_context_not_cwd(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    target = root / "B.fit"
    _write_fits(target)
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    # Same basename does not exist relative to process CWD, only in the declared
    # input context -> resolution must still succeed.
    assert resolve_explicit_reference("B.fit", search_roots=[str(root)]) == os.path.realpath(
        str(target)
    )
    # A sub-path relative to the declared root also resolves.
    subdir = root / "deep"
    subdir.mkdir()
    _write_fits(subdir / "C.fit")
    assert resolve_explicit_reference(
        os.path.join("deep", "C.fit"), search_roots=[str(root)]
    ) == os.path.realpath(str(subdir / "C.fit"))


def test_duplicate_basename_is_ambiguous(tmp_path):
    r1 = tmp_path / "r1"
    r2 = tmp_path / "r2"
    r1.mkdir()
    r2.mkdir()
    _write_fits(r1 / "dup.fit", ra=10.0)
    _write_fits(r2 / "dup.fit", ra=11.0)
    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_explicit_reference("dup.fit", search_roots=[str(r1), str(r2)])
    assert exc.value.code == ReferenceResolutionError.CODE_AMBIGUOUS


def test_missing_request_is_unresolved(tmp_path):
    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_explicit_reference("missing.fit", search_roots=[str(tmp_path)])
    assert exc.value.code == ReferenceResolutionError.CODE_UNRESOLVED
    with pytest.raises(ReferenceResolutionError) as exc2:
        resolve_explicit_reference(str(tmp_path / "nope.fit"), search_roots=[str(tmp_path)])
    assert exc2.value.code == ReferenceResolutionError.CODE_UNRESOLVED


def test_directory_request_is_refused(tmp_path):
    (tmp_path / "adir").mkdir()
    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_explicit_reference("adir", search_roots=[str(tmp_path)])
    assert exc.value.code == ReferenceResolutionError.CODE_DIRECTORY


def test_invalid_fits_request_is_refused(tmp_path):
    bad = tmp_path / "notfits.fit"
    bad.write_bytes(b"this is not a FITS file")
    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_explicit_reference(str(bad), search_roots=[str(tmp_path)])
    assert exc.value.code == ReferenceResolutionError.CODE_INVALID


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
def test_inaccessible_request_is_refused(tmp_path):
    target = tmp_path / "locked.fit"
    _write_fits(target)
    os.chmod(target, 0)
    try:
        with pytest.raises(ReferenceResolutionError) as exc:
            resolve_explicit_reference(str(target), search_roots=[str(tmp_path)])
        assert exc.value.code == ReferenceResolutionError.CODE_INACCESSIBLE
    finally:
        os.chmod(target, 0o644)


def test_blank_request_is_refused(tmp_path):
    for blank in (None, "", "   "):
        with pytest.raises(ReferenceResolutionError) as exc:
            resolve_explicit_reference(blank, search_roots=[str(tmp_path)])
        assert exc.value.code == ReferenceResolutionError.CODE_EMPTY


# ---------------------------------------------------------------------------
# end-to-end start_processing authority
# ---------------------------------------------------------------------------


class _NoopExecutor:
    def __init__(self, max_workers=1, **kwargs):
        self._max_workers = max_workers

    def shutdown(self, *args, **kwargs):
        pass


def _drive_start(tmp_path, request, additional=None, origin_hint="USER"):
    """Run a real (stubbed-executor) Drizzle Standard start; return facts."""
    root = tmp_path / "input"
    out = tmp_path / ("output_" + str(abs(hash((request, str(additional))))))
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, ra in (("A.fit", 275.0), ("B.fit", 276.0)):
        _write_fits(root / name, ra=ra)
    selector_calls = []

    def choose(*args, **kwargs):
        selector_calls.append("AUTO")
        from seestar.core.geometry_reference import GeometrySelection, ResolvedReference

        return GeometrySelection(ResolvedReference(str(root / "A.fit"), "AUTO_GEOMETRY"))

    import seestar.queuep.queue_manager as _qm

    orig_select = _qm.select_geometry_reference
    orig_exec = _qm.ProcessPoolExecutor
    _qm.select_geometry_reference = choose
    _qm.ProcessPoolExecutor = _NoopExecutor
    try:
        st = _qm.SeestarQueuedStacker(batch_size=1, autotune=False)
        st.reference_origin_hint = origin_hint
        events = []
        st.update_progress = lambda message, progress=None, level=None: events.append(str(message))
        st._solve_astrometry_async = lambda path, header, settings, **kwargs: WCS(header).celestial
        snapshot = {}
        done = threading.Event()

        def worker():
            frozen = st._consume_frozen_reference_for_worker()
            snapshot.update(
                source=frozen.source_basename,
                origin=frozen.origin,
                aligner=st.aligner.reference_image_path,
                wcs_crval=list(st.reference_wcs_object.wcs.crval),
                grid_crval=list(st.drizzle_output_wcs.wcs.crval),
                shape=tuple(st.drizzle_output_shape_hw),
            )
            st.processing_active = False
            done.set()

        st._worker = worker
        kwargs = dict(
            use_drizzle=True,
            drizzle_scale=2,
            drizzle_kernel="lanczos2",
            batch_size=1,
            correct_hot_pixels=False,
            perform_cleanup=False,
            move_stacked=False,
            reproject_between_batches=False,
            reproject_coadd_final=False,
        )
        if additional:
            kwargs["initial_additional_folders"] = additional
        started = st.start_processing(str(root), str(out), reference_path_ui=request, **kwargs)
        if st.processing_thread is not None:
            st.processing_thread.join(10)
        st.quality_executor.shutdown()
        return st, started, done, snapshot, selector_calls, events
    finally:
        _qm.select_geometry_reference = orig_select
        _qm.ProcessPoolExecutor = orig_exec


def test_absolute_reference_is_user_and_never_auto(tmp_path):
    _, started, done, snapshot, calls, events = _drive_start(tmp_path, None, origin_hint=None)
    # sanity baseline: no manual -> AUTO
    assert started and done.is_set(), events[-15:]
    assert snapshot["source"] == "A.fit" and calls == ["AUTO"]

    _, started, done, snapshot, calls, events = _drive_start(
        tmp_path / "abs", str((tmp_path / "abs" / "input" / "B.fit"))
    )
    assert started and done.is_set(), events[-15:]
    assert snapshot["source"] == "B.fit"
    assert snapshot["origin"] == "USER"
    assert snapshot["wcs_crval"] == [276.0, 20.0]
    assert snapshot["grid_crval"] == [276.0, 20.0]
    assert calls == []


def test_basename_reference_resolves_in_input_context_not_cwd(tmp_path, monkeypatch):
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    _, started, done, snapshot, calls, events = _drive_start(tmp_path / "base", "B.fit")
    assert started and done.is_set(), events[-15:]
    assert snapshot["source"] == "B.fit"
    assert snapshot["origin"] == "USER"
    assert calls == []


def test_missing_explicit_fails_closed_without_auto(tmp_path):
    st, started, done, snapshot, calls, events = _drive_start(
        tmp_path / "miss", str(tmp_path / "miss" / "input" / "missing.fit")
    )
    assert started is False
    assert calls == []
    assert done.is_set() is False
    assert st.startup_refusal is not None
    assert st.startup_refusal.code == qm.StartupRefusal.CODE_MANUAL_REFERENCE_UNRESOLVED
    assert st.processing_active is False


def test_directory_explicit_fails_closed(tmp_path):
    base = tmp_path / "dir"
    base.mkdir()
    (base / "input" / "adir").mkdir(parents=True)
    st, started, done, snapshot, calls, events = _drive_start(base, str(base / "input" / "adir"))
    assert started is False
    assert calls == []
    assert st.startup_refusal is not None
    assert st.startup_refusal.code == qm.StartupRefusal.CODE_MANUAL_REFERENCE_UNRESOLVED


def test_conflicting_manual_freeze_is_refused(tmp_path):
    from types import MethodType

    class _Dummy:
        pass

    st = _Dummy()
    st.aligner = _Dummy()
    st.aligner.reference_image_path = None
    st.update_progress = lambda message, level=None: None
    for name in (
        "_freeze_reference",
        "_clear_frozen_reference",
        "_sync_frozen_reference_to_aligner",
    ):
        setattr(st, name, MethodType(getattr(qm.SeestarQueuedStacker, name), st))
    a = tmp_path / "A.fit"
    b = tmp_path / "B.fit"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    d1 = st._freeze_reference(str(a), "USER")
    assert st._freeze_reference(str(a), "USER") is d1  # idempotent
    with pytest.raises(RuntimeError):
        st._freeze_reference(str(b), "USER")
    with pytest.raises(RuntimeError):
        st._freeze_reference(str(a), "RESUME")


def test_requested_policy_reports_explicit_intent_by_presence():
    class _Dummy:
        pass

    st = _Dummy()
    st._resume_requested = False
    st.reference_origin_hint = None
    # presence of a non-blank request is explicit intent, even if unresolvable
    assert (
        qm.SeestarQueuedStacker._reference_requested_policy(st, "B.fit") == "user"
    )
    assert qm.SeestarQueuedStacker._reference_requested_policy(st, None) == "auto"
    st.reference_origin_hint = "ZEANALYSER_V1"
    assert (
        qm.SeestarQueuedStacker._reference_requested_policy(st, "B.fit")
        == "zeanalyser"
    )
    st._resume_requested = True
    assert (
        qm.SeestarQueuedStacker._reference_requested_policy(st, "B.fit") == "resume"
    )
