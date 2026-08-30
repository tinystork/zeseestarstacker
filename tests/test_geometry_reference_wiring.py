"""Wiring tests for the GAR geometry-aware reference lifecycle."""

from types import MethodType

import pytest

from seestar.queuep.queue_manager import SeestarQueuedStacker
from seestar.core.reference_state import FrozenReference
from seestar.core.geometry_reference import (
    GeometrySelection,
    LegacySelection,
    ORIGIN_AUTO_GEOMETRY,
    ORIGIN_AUTO_LEGACY,
    ORIGIN_USER,
    ORIGIN_ZEANALYSER,
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
    for helper_name in (
        "_resolve_automatic_reference",
        "_clear_frozen_reference",
        "_sync_frozen_reference_to_aligner",
        "_freeze_reference",
        "_record_materialized_reference",
        "_require_frozen_reference",
        "_consume_frozen_reference_for_worker",
        "_run_started_input_fields",
    ):
        setattr(
            st,
            helper_name,
            MethodType(getattr(SeestarQueuedStacker, helper_name), st),
        )
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
    materialized = tmp_path / "temp_processing" / "reference_image.fit"
    materialized.parent.mkdir()
    materialized.write_bytes(b"prepared")
    aligner.set_frozen_reference(
        FrozenReference(
            source_path=str(reference),
            origin=ORIGIN_AUTO_GEOMETRY,
            materialized_path=str(materialized),
        )
    )
    monkeypatch.setattr(aligner, "_save_reference_image", lambda *a, **k: None)

    image, header = aligner._get_reference_image(
        str(tmp_path), [f"edge_{index}.fit" for index in range(20)], str(tmp_path)
    )
    assert image is not None
    assert header["HIERARCH SEESTAR REF SRCFILE"] == reference.name
    assert calls == [str(reference)]

    # A moved source may fall back to the run-local prepared FITS for loading,
    # but the canonical identity must remain the original source filename.
    reference.unlink()
    image, header = aligner._get_reference_image(
        str(tmp_path), [f"edge_{index}.fit" for index in range(20)], str(tmp_path)
    )
    assert image is not None
    assert header["HIERARCH SEESTAR REF SRCFILE"] == reference.name
    assert calls == [str(reference), str(materialized)]


def test_auto_geometry_handoff_keeps_one_source_and_one_decision(
    tmp_path, monkeypatch
):
    import seestar.queuep.queue_manager as qm

    st = _make_stacker()
    events = []
    st.update_progress = lambda message, progress=None, level=None: events.append(message)
    source = tmp_path / "central.fit"
    source.write_bytes(b"source")
    materialized = tmp_path / "temp_processing" / "reference_image.fit"
    materialized.parent.mkdir()
    materialized.write_bytes(b"prepared")
    calls = {"geometry": 0}

    monkeypatch.setattr(qm, "canonical_session_sources", lambda *a, **k: [str(source)])

    def select_once(*args, **kwargs):
        calls["geometry"] += 1
        return GeometrySelection(
            ResolvedReference(str(source), ORIGIN_AUTO_GEOMETRY)
        )

    monkeypatch.setattr(qm, "select_geometry_reference", select_once)
    st._resolve_automatic_reference("/input", None, None, -1)
    st._record_materialized_reference(str(materialized))
    consumed = st._consume_frozen_reference_for_worker()
    st._resolve_automatic_reference("/input", None, None, -1)

    assert calls == {"geometry": 1}
    assert st._automatic_reference_resolution_count == 1
    assert consumed.source_path == str(source.resolve())
    assert consumed.materialized_path == str(materialized.resolve())
    assert st.aligner.reference_image_path == consumed.source_path
    assert st.aligner.frozen_reference == consumed
    assert "Frozen reference: central.fit" in events
    assert "Worker consumes: central.fit" in events


@pytest.mark.parametrize("origin", [ORIGIN_USER, ORIGIN_ZEANALYSER])
def test_explicit_reference_origins_share_frozen_handoff(tmp_path, origin):
    st = _make_stacker()
    st.update_progress = lambda message, progress=None, level=None: None
    source = tmp_path / (origin.lower() + ".fit")
    source.write_bytes(b"source")
    materialized = tmp_path / "temp_processing" / "reference_image.fit"
    materialized.parent.mkdir(exist_ok=True)
    materialized.write_bytes(b"prepared")

    frozen = st._freeze_reference(str(source), origin)
    st._record_materialized_reference(str(materialized))
    consumed = st._consume_frozen_reference_for_worker()

    assert consumed.source_path == frozen.source_path
    assert consumed.origin == origin
    assert consumed.materialized_path == str(materialized.resolve())
    assert st.aligner.reference_image_path == frozen.source_path


def test_genuinely_missing_frozen_reference_fails_closed():
    st = _make_stacker()
    with pytest.raises(RuntimeError) as excinfo:
        st._require_frozen_reference()
    assert str(excinfo.value) == (
        "Frozen registration reference is unavailable; "
        "automatic reselection is forbidden."
    )


def test_run_started_input_count_is_not_false_zero():
    st = _make_stacker()
    st.files_in_queue = 0
    assert st._run_started_input_fields() == {
        "input_count": None,
        "input_count_state": "not_yet_enumerated",
    }
    st.files_in_queue = 1602
    assert st._run_started_input_fields() == {
        "input_count": 1602,
        "input_count_state": "known",
    }


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


def test_reference_handoff_diagnostics_bypass_gui_debounce(monkeypatch):
    import seestar.queuep.queue_manager as qm

    st = _Dummy()
    events = []
    st.progress_callback = lambda message, progress, level=None: events.append(message)
    st.gui_event_queue = None
    st.logger = None
    monkeypatch.setattr(qm, "_QM_LAST_GUI_PUSH", qm._mono())

    for message in (
        "Reference selected: central.fit",
        "Frozen reference: central.fit",
        "Worker consumes: central.fit",
    ):
        SeestarQueuedStacker.update_progress(st, message, level="INFO")

    assert events == [
        "Reference selected: central.fit",
        "Frozen reference: central.fit",
        "Worker consumes: central.fit",
    ]
