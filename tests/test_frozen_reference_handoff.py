"""GAR-05B end-to-end lifecycle closure for frozen references."""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest
from astropy.io import fits

import seestar.queuep.queue_manager as queue_manager_module
from seestar.core.geometry_reference import (
    GeometrySelection,
    ORIGIN_AUTO_GEOMETRY,
    ORIGIN_USER,
    ORIGIN_ZEANALYSER,
    ResolvedReference,
)
from seestar.queuep.queue_manager import SeestarQueuedStacker


class _NoopExecutor:
    def __init__(self, max_workers=1, **_kwargs):
        self._max_workers = max_workers

    def shutdown(self, *_args, **_kwargs):
        return None


def _write_source(path):
    yy, xx = np.indices((32, 32))
    data = ((xx + yy) % 17).astype(np.uint16) * 100
    header = fits.Header()
    header["RA"] = 275.0
    header["DEC"] = -13.7
    header["BAYERPAT"] = "GRBG"
    header["EXPTIME"] = 10.0
    fits.PrimaryHDU(data=data, header=header).writeto(path)


def _run_lifecycle(monkeypatch, tmp_path, *, explicit_origin=None):
    monkeypatch.setattr(queue_manager_module, "ProcessPoolExecutor", _NoopExecutor)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    source = input_dir / "central.fit"
    _write_source(source)

    selection_calls = {"geometry": 0}

    def select_geometry(*_args, **_kwargs):
        selection_calls["geometry"] += 1
        return GeometrySelection(
            ResolvedReference(str(source), ORIGIN_AUTO_GEOMETRY),
            geometry_source="native_pointing",
        )

    monkeypatch.setattr(
        queue_manager_module, "select_geometry_reference", select_geometry
    )
    stacker = SeestarQueuedStacker(batch_size=1, autotune=False)
    stacker.reference_origin_hint = explicit_origin
    events = []
    stacker.update_progress = (
        lambda message, progress=None, level=None: events.append(str(message))
    )
    worker_snapshot = {}
    worker_started = threading.Event()

    def worker():
        descriptor = stacker._consume_frozen_reference_for_worker()
        worker_snapshot.update(
            source_path=descriptor.source_path,
            materialized_path=descriptor.materialized_path,
            origin=descriptor.origin,
            aligner_path=stacker.aligner.reference_image_path,
        )
        worker_started.set()
        stacker.processing_active = False

    stacker._worker = worker
    started = stacker.start_processing(
        str(input_dir),
        str(output_dir),
        reference_path_ui=(str(source) if explicit_origin else None),
        batch_size=1,
        correct_hot_pixels=False,
        perform_cleanup=False,
        move_stacked=False,
        reproject_between_batches=False,
        reproject_coadd_final=False,
    )
    if stacker.processing_thread is not None:
        stacker.processing_thread.join(timeout=10)
    stacker.quality_executor.shutdown()
    assert started is True
    assert worker_started.is_set()
    return stacker, source, selection_calls, events, worker_snapshot


def test_auto_geometry_resolve_materialize_worker_same_source(monkeypatch, tmp_path):
    stacker, source, calls, events, snapshot = _run_lifecycle(
        monkeypatch, tmp_path
    )

    assert calls == {"geometry": 1}
    assert stacker._automatic_reference_resolution_count == 1
    assert snapshot["origin"] == ORIGIN_AUTO_GEOMETRY
    assert snapshot["source_path"] == os.path.realpath(source)
    assert snapshot["aligner_path"] == snapshot["source_path"]
    assert snapshot["materialized_path"] != snapshot["source_path"]
    assert snapshot["materialized_path"].endswith(
        os.path.join("temp_processing", "reference_image.fit")
    )
    assert "Frozen reference: central.fit" in events
    assert "Worker consumes: central.fit" in events
    assert not any("AUTO_LEGACY" in event for event in events)


@pytest.mark.parametrize("origin", [ORIGIN_USER, ORIGIN_ZEANALYSER])
def test_explicit_origins_use_same_frozen_worker_contract(
    monkeypatch, tmp_path, origin
):
    stacker, source, calls, events, snapshot = _run_lifecycle(
        monkeypatch, tmp_path, explicit_origin=origin
    )

    assert calls == {"geometry": 0}
    assert stacker._automatic_reference_resolution_count == 0
    assert snapshot["origin"] == origin
    assert snapshot["source_path"] == os.path.realpath(source)
    assert snapshot["aligner_path"] == snapshot["source_path"]
    assert snapshot["materialized_path"] != snapshot["source_path"]
    assert "Frozen reference: central.fit" in events
    assert "Worker consumes: central.fit" in events
    assert not any("Selecting registration reference" in event for event in events)
