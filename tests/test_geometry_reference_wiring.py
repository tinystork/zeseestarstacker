"""Wiring tests for the GAR-04 geometry-aware reference (M3)."""

from seestar.queuep.queue_manager import SeestarQueuedStacker
from seestar.core.geometry_reference import (
    GeometrySelection,
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

