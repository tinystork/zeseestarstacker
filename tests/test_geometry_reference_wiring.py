"""Wiring tests for the GAR-04 geometry-aware reference (M3)."""

from seestar.queuep.queue_manager import SeestarQueuedStacker
from seestar.core.geometry_reference import (
    GeometrySelection,
    LegacySelection,
    ORIGIN_AUTO_GEOMETRY,
    ORIGIN_AUTO_LEGACY,
    ResolvedReference,
)


class _Dummy:
    pass


def _make_stacker():
    st = _Dummy()
    st.aligner = _Dummy()
    st.aligner.reference_image_path = None
    st.bayer_pattern = "GRBG"
    st.correct_hot_pixels = False
    st.hot_pixel_threshold = 3.0
    st.neighborhood_size = 5
    st.stop_processing = False
    st._resolved_reference_origin = None
    st.update_progress = lambda msg, level=None: None
    return st


def test_resolve_automatic_reference_pins_geometry(tmp_path, monkeypatch):
    import seestar.queuep.queue_manager as qm
    st = _make_stacker()
    target = str(tmp_path / "best.fit")
    monkeypatch.setattr(qm, "canonical_session_sources", lambda *a, **k: ["/a.fit", "/b.fit", target])
    monkeypatch.setattr(qm, "select_geometry_reference", lambda *a, **k: GeometrySelection(ResolvedReference(target, ORIGIN_AUTO_GEOMETRY)))
    SeestarQueuedStacker._resolve_automatic_reference(st, "/input", None, None, -1)
    assert st._resolved_reference_origin == ORIGIN_AUTO_GEOMETRY
    assert st.aligner.reference_image_path == target


def test_resolve_automatic_reference_falls_back_legacy(tmp_path, monkeypatch):
    import seestar.queuep.queue_manager as qm
    st = _make_stacker()
    monkeypatch.setattr(qm, "canonical_session_sources", lambda *a, **k: ["/a.fit", "/b.fit"])
    monkeypatch.setattr(qm, "select_geometry_reference", lambda *a, **k: GeometrySelection(ResolvedReference(None, ORIGIN_AUTO_LEGACY)))
    SeestarQueuedStacker._resolve_automatic_reference(st, "/input", None, None, -1)
    assert st._resolved_reference_origin == ORIGIN_AUTO_LEGACY
    assert st.aligner.reference_image_path is None


def test_resolve_automatic_reference_no_sources(tmp_path, monkeypatch):
    import seestar.queuep.queue_manager as qm
    st = _make_stacker()
    monkeypatch.setattr(qm, "canonical_session_sources", lambda *a, **k: [])
    SeestarQueuedStacker._resolve_automatic_reference(st, "/input", None, None, -1)
    assert st._resolved_reference_origin == ORIGIN_AUTO_LEGACY
    assert st.aligner.reference_image_path is None


def test_exactly_one_automatic_resolution_is_enforced(tmp_path, monkeypatch):
    import seestar.queuep.queue_manager as qm

    st = _make_stacker()
    target = str(tmp_path / "central.fit")
    calls = {"geometry": 0}

    monkeypatch.setattr(qm, "canonical_session_sources", lambda *a, **k: [target])

    def select_once(*args, **kwargs):
        calls["geometry"] += 1
        return GeometrySelection(ResolvedReference(target, ORIGIN_AUTO_GEOMETRY))

    monkeypatch.setattr(qm, "select_geometry_reference", select_once)
    SeestarQueuedStacker._resolve_automatic_reference(st, "/input", None, None, -1)
    SeestarQueuedStacker._resolve_automatic_reference(st, "/input", None, None, -1)
    assert calls == {"geometry": 1}
    assert st._automatic_reference_resolution_count == 1
    assert st.aligner.reference_image_path == target
    assert st.aligner._reference_resolution_frozen is True


def test_legacy_fallback_is_resolved_and_frozen_once(tmp_path, monkeypatch):
    import seestar.queuep.queue_manager as qm

    st = _make_stacker()
    target = str(tmp_path / "legacy.fit")
    calls = {"legacy": 0}
    monkeypatch.setattr(qm, "canonical_session_sources", lambda *a, **k: ["/a.fit"])
    monkeypatch.setattr(
        qm,
        "select_geometry_reference",
        lambda *a, **k: GeometrySelection(
            ResolvedReference(None, ORIGIN_AUTO_LEGACY), fallback_reason="no geometry"
        ),
    )
    monkeypatch.setattr(qm, "legacy_session_sources", lambda *a, **k: [target])

    def legacy_once(*args, **kwargs):
        calls["legacy"] += 1
        return LegacySelection(ResolvedReference(target, ORIGIN_AUTO_LEGACY), 1, 3.0)

    monkeypatch.setattr(qm, "select_legacy_reference", legacy_once)
    SeestarQueuedStacker._resolve_automatic_reference(st, "/input", None, None, -1)
    SeestarQueuedStacker._resolve_automatic_reference(st, "/input", None, None, -1)
    assert calls == {"legacy": 1}
    assert st._resolved_reference_origin == ORIGIN_AUTO_LEGACY
    assert st.aligner.reference_image_path == target
    assert st.aligner._reference_resolution_frozen is True


def test_frozen_reference_materialization_never_reselects(tmp_path, monkeypatch):
    import numpy as np
    from astropy.io import fits
    import seestar.core.alignment as alignment

    reference = tmp_path / "chosen.fit"
    reference.write_bytes(b"exists")
    calls = []

    def load_once(path):
        calls.append(path)
        return np.ones((4, 4, 3), dtype=np.float32), fits.Header()

    monkeypatch.setattr(alignment, "load_and_validate_fits", load_once)
    aligner = alignment.SeestarAligner()
    aligner.correct_hot_pixels = False
    aligner.reference_image_path = str(reference)
    aligner._reference_resolution_frozen = True
    monkeypatch.setattr(aligner, "_save_reference_image", lambda *a, **k: None)

    for _ in range(2):
        image, header = aligner._get_reference_image(
            str(tmp_path), [f"edge_{index}.fit" for index in range(20)], str(tmp_path)
        )
        assert image is not None
        assert header["HIERARCH SEESTAR REF SRCFILE"] == reference.name
    assert calls == [str(reference), str(reference)]


def test_qt_startup_progress_is_delivered_directly_before_queue_drain(monkeypatch):
    from queue import Queue
    import seestar.queuep.queue_manager as qm

    st = _Dummy()
    st.progress_callback = lambda message, progress, level=None: events.append(
        (message, progress, level)
    )
    st.gui_event_queue = Queue()
    st._direct_startup_progress = True
    st.logger = None
    events = []
    monkeypatch.setattr(qm, "_QM_LAST_GUI_PUSH", 0.0)

    SeestarQueuedStacker.update_progress(
        st, "Scanning dataset pointing... 10%", 10, level="INFO"
    )
    assert events == [("Scanning dataset pointing... 10%", 10, "INFO")]
    assert st.gui_event_queue.empty()
