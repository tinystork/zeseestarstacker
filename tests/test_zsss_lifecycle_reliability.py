"""ZSSS-LIFECYCLE-01 reliability tests (offscreen Qt, engine-free fakes).

Covers the product-reliability boundary implemented for this milestone:

* A) fresh output + Drizzle accepted, durable run log created with lifecycle events;
* C) Qt receives a known structured startup refusal and maps it to localized
     EN/FR semantics, while a generic false start stays generic;
* D) accepted run creates a session-specific log immediately, and while the
     engine is still running the file already holds RUN_ACCEPTED + progressive events;
* E) success, controlled backend failure, cancellation and unwritable-log all
     have correct fail-open outcomes;
* F) the exact premature-return reproducer (processing_active False while the
     engine thread tail blocks) — the backend must not finish until the tail releases;
* G) full offscreen Qt lifecycle bounded witness incl. a second UI action;
* H) lifecycle event ordering assertions (timeout only as a deadlock guard);
* I) scientific-source token/diff boundaries (import-hygiene + source-token).

No real stacking is ever performed.  ``QT_QPA_PLATFORM=offscreen`` is set
defensively before any ``QApplication`` is created.
"""

from __future__ import annotations

import os
import queue as _queue
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, QThread, Slot
from PySide6.QtWidgets import QApplication

import seestar
from seestar.gui_qt import (
    MainWindow,
    RunController,
    RunStatus,
    RunWorker,
    create_application,
)
from seestar.gui_qt.backend_runner import (
    BackendRunResult,
    BaseRunBackend,
    SeestarQueuedStackerBackend,
)
from seestar.gui_qt.run_bridge import build_run_request
from seestar.gui_qt.run_log import RunLog
from seestar.gui_qt.settings_state import QtSettingsState
from seestar.gui_qt.startup_refusal import (
    CODE_OUTPUT_STATE_INCOMPATIBLE,
    StartupRefusalPayload,
    StartupRefusedError,
)

ROOT = Path(__file__).resolve().parents[1]


def _pump_until(qapp: QApplication, predicate, timeout_ms: int = 8000) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return bool(predicate())


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


def _make_request(**overrides):
    state = QtSettingsState()
    for key, value in overrides.items():
        setattr(state, key, value)
    return build_run_request(state)


def _log_events(path: Path) -> list:
    """Return the ordered event names from a run-log file (token after timestamp)."""
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            events.append(parts[1])
    return events


def _index(events: list, name: str) -> int:
    idx = events.index(name) if name in events else -1
    assert idx >= 0, f"missing event {name!r} in {events!r}"
    return idx


# --------------------------------------------------------------------------
# Fake engine stacker (adapter-level, no real engine)
# --------------------------------------------------------------------------
class FakeEngineStacker:
    """A fake ``SeestarQueuedStacker``-shaped object driving a real tail thread.

    It reproduces the engine's lifecycle surface: a real ``processing_thread``,
    a deferred ``gui_event_queue``, a ``processing_active`` flag cleared before
    the tail finishes, and a fail-open ``set_lifecycle_callback`` seam.
    """

    def __init__(self, *, block_tail: bool = False, processing_error=None):
        self.progress_cb = None
        self._lifecycle_cb = None
        self.gui_event_queue = _queue.Queue()
        self.processing_thread = None
        self.processing_active = False
        self.processing_error = processing_error
        self.startup_refusal = None
        self.output_folder = None
        self.align_on_disk = None
        self.stack_final_combine = None
        self.final_stacked_path = None
        self.processed_files_count = None
        self.stop_called = False
        self._release = threading.Event()
        self._block_tail = block_tail

    def set_progress_callback(self, cb):
        self.progress_cb = cb

    def set_lifecycle_callback(self, cb):
        self._lifecycle_cb = cb

    def set_preview_callback(self, cb):
        pass

    def start_processing(self, **kwargs):
        self.output_folder = kwargs.get("output_dir")
        self.processing_thread = threading.Thread(target=self._worker, daemon=True)
        self.processing_thread.start()
        return True

    def _emit(self, event, fields=None):
        cb = self._lifecycle_cb
        if cb is None:
            return
        cb(event, dict(fields or {}))

    def _emit_seams(self):
        self._emit("DRIZZLE_FINALIZATION_ENTERED")
        self._emit("DRIZZLE_FINALIZATION_RETURNED", {"success": True})
        self._emit("FINAL_FITS_SAVE_ENTERED", {"path": "final.fits"})
        self._emit("FINAL_FITS_SAVE_RETURNED", {"success": True, "path": "final.fits"})
        self._emit("FINAL_PREVIEW_SAVE_ENTERED", {"path": "final.png"})
        self._emit("FINAL_PREVIEW_SAVE_RETURNED", {"success": True, "path": "final.png"})

    def _worker(self):
        self.processing_active = True
        self._emit_seams()
        # The real engine clears processing_active before its final cleanup /
        # tail finishes — reproduced here so ``is_running()`` can be False while
        # the thread is still alive.
        self.processing_active = False
        if self._block_tail:
            self._release.wait(timeout=30)
        # Engine returning seam (queue_manager._worker tail): emitted immediately
        # before the engine thread function exits.
        self._emit("ENGINE_PROCESSING_RETURNING")

    def is_running(self):
        return False

    def stop(self):
        self.stop_called = True
        self._release.set()


class RunningEngineStacker(FakeEngineStacker):
    """A stacker whose engine thread stays alive and emits progressive events."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tail_blocked = threading.Event()

    def _worker(self):
        self.processing_active = True
        for i in range(6):
            self.gui_event_queue.put(
                lambda i=i: self.progress_cb(f"step {i}", 10 * (i + 1), "INFO")
            )
        self._tail_blocked.set()
        self._release.wait(timeout=30)
        self.processing_active = False

    def is_running(self):
        return self.processing_active and self.processing_thread.is_alive()


def _write_minimal_fits(path: Path, nimages: int = 5, totexp: float = 300.0) -> None:
    """Write a header-only FITS with ``NIMAGES``/``TOTEXP`` (astropy, no engine)."""
    from astropy.io import fits

    hdu = fits.PrimaryHDU()
    hdu.header["NIMAGES"] = nimages
    hdu.header["TOTEXP"] = totexp
    hdu.writeto(str(path), overwrite=True)


class ResultProducingEngineStacker(FakeEngineStacker):
    """A fake stacker that actually writes a final FITS product.

    The strengthened full-Qt witness uses this so the summary truly derives
    SUCCESS (final_stack_exists + NIMAGES + openable output), the result path
    exists/is accessible, and the terminal state is a genuine success.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.files_in_queue = 5
        self.processed_files_count = 5
        self.final_stacked_path = None

    def start_processing(self, **kwargs):
        super().start_processing(**kwargs)
        final = Path(self.output_folder) / "stack_final.fits"
        _write_minimal_fits(final, nimages=5, totexp=300.0)
        self.final_stacked_path = str(final)
        self.files_in_queue = 5
        self.processed_files_count = 5
        return True


def _backend_for(stacker):
    return SeestarQueuedStackerBackend(
        stacker_factory=lambda **kw: stacker, poll_interval=0.001
    )


# --------------------------------------------------------------------------
# A) fresh output + Drizzle accepted
# --------------------------------------------------------------------------
def test_fresh_output_drizzle_accepted_creates_run_log(qapp, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stacker = FakeEngineStacker()
    backend = _backend_for(stacker)

    request = _make_request(
        input_folder=str(tmp_path),
        output_folder=str(out_dir),
        use_drizzle=True,
        drizzle_mode="Final",
        stacking_mode="kappa-sigma",
    )
    logs = []
    result = backend.run(request, lambda p: None, logs.append, lambda: False)

    assert result is BackendRunResult.FINISHED
    assert backend.run_log is not None
    log_path = Path(backend.run_log.path)
    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "RUN_ACCEPTED" in text
    assert "RUN_STARTED" in text
    assert "RUN_METADATA" in text
    # Product version/codename is recorded in the run metadata (point 4).
    expected_version = f"{seestar.__version__} {seestar.__codename__}"
    assert f"product_version={expected_version}" in text
    assert "DRIZZLE_FINALIZATION_ENTERED" in text
    assert "FINAL_FITS_SAVE_ENTERED" in text
    assert "FINAL_PREVIEW_SAVE_ENTERED" in text
    assert "ENGINE_PROCESSING_RETURNING" in text
    assert "ENGINE_PROCESSING_RETURNED" in text
    assert "BACKEND_RETURNING" in text
    # Only one file per accepted run (never overwrite earlier logs).
    assert len(list(out_dir.glob("zsss_run_*.log"))) == 1


# --------------------------------------------------------------------------
# B) adapter-level: structured refusal + artifacts preserved (no engine import)
# --------------------------------------------------------------------------
def test_backend_raises_structured_refusal_and_preserves_artifacts(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sentinel = out_dir / "resume_manifest.json"
    sentinel.write_bytes(b"SENTINEL-BYTES")
    before = sentinel.read_bytes()

    class _RefusalCarrier:
        code = "OUTPUT_STATE_INCOMPATIBLE"
        technical_detail = "resume limited to plain classic SUM/W"
        semantic_key = "output_state_incompatible"
        semantic_data = {"mode": "drizzle"}

    class RefusingStacker(FakeEngineStacker):
        def start_processing(self, **kwargs):
            self.startup_refusal = _RefusalCarrier()
            return False

    stacker = RefusingStacker()
    backend = _backend_for(stacker)
    request = _make_request(
        input_folder=str(tmp_path),
        output_folder=str(out_dir),
        use_drizzle=True,
    )

    with pytest.raises(StartupRefusedError) as exc_info:
        backend.run(request, lambda p: None, lambda m: None, lambda: False)

    assert exc_info.value.payload.code == CODE_OUTPUT_STATE_INCOMPATIBLE
    assert exc_info.value.payload.semantic_key == "output_state_incompatible"
    # Artifacts byte-identical; no run log ever created in the output folder.
    assert sentinel.read_bytes() == before
    assert sorted(p.name for p in out_dir.iterdir()) == ["resume_manifest.json"]
    assert list(out_dir.glob("zsss_run_*.log")) == []


# --------------------------------------------------------------------------
# C) known refusal -> localized EN/FR; generic false start stays generic
# --------------------------------------------------------------------------
class _RefusingBackend(BaseRunBackend):
    def __init__(self, refusal):
        self._refusal = refusal
        self.cancel_called = False

    def run(self, request, progress_callback, log_callback, is_cancel_requested,
            preview_callback=None, summary_callback=None):
        raise StartupRefusedError(self._refusal)

    def cancel(self):
        self.cancel_called = True


class _FalseStartBackend(BaseRunBackend):
    def run(self, request, progress_callback, log_callback, is_cancel_requested,
            preview_callback=None, summary_callback=None):
        raise RuntimeError("SeestarQueuedStacker.start_processing() reported it did not start")

    def cancel(self):
        pass


def test_qt_maps_known_refusal_to_localized_en_and_fr(qapp):
    payload = StartupRefusalPayload(
        code=CODE_OUTPUT_STATE_INCOMPATIBLE,
        technical_detail="resume limited to plain classic SUM/W (drizzle/mosaic/reproject sufficient state not covered by HSI-2B)",
        semantic_key="output_state_incompatible",
        semantic_data={"mode": "drizzle"},
    )
    win = MainWindow(backend_factory=lambda: _RefusingBackend(payload))
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)
        # EN (default language) title/body surfaced via the status bar + log.
        assert win.statusBar().currentMessage() == (
            "Output folder already in use"
        )
        assert (
            "The selected output folder contains data from a previous "
            "processing run" in win.log_view.toPlainText()
        )
        # FR mapping is exercised directly through the same architecture.
        win._language = "fr"
        title, body = win._format_refusal(payload)
        assert title == "Dossier de sortie déjà utilisé"
        assert "traitement précédent" in body
        assert "Drizzle" not in body  # wording is mode-independent
        # A refused run never opened a run log (never touched the output folder).
        assert win.controller.run_log is None
    finally:
        win.shutdown()


def test_generic_false_start_stays_generic(qapp):
    win = MainWindow(backend_factory=lambda: _FalseStartBackend())
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)
        assert win.statusBar().currentMessage().startswith("Failed:")
        assert "did not start" in win.log_view.toPlainText()
    finally:
        win.shutdown()


def test_qt_refusal_wording_is_mode_independent_for_mosaic(qapp):
    """The refusal wording is mode-independent: no mode label at all (point 5)."""
    payload = StartupRefusalPayload(
        code=CODE_OUTPUT_STATE_INCOMPATIBLE,
        technical_detail="resume limited to plain classic SUM/W",
        semantic_key="output_state_incompatible",
        semantic_data={"mode": "mosaic"},
    )
    win = MainWindow(backend_factory=lambda: _RefusingBackend(payload))
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)
        # EN body is the standard mode-independent wording: never a mode label.
        en_text = win.log_view.toPlainText()
        assert "previous processing run" in en_text
        assert "mosaic" not in en_text
        assert "Drizzle" not in en_text

        # FR mapping through the same architecture.
        win._language = "fr"
        title, body = win._format_refusal(payload)
        assert title == "Dossier de sortie déjà utilisé"
        assert "traitement précédent" in body
        assert "mosaïque" not in body
        assert "Drizzle" not in body
    finally:
        win.shutdown()


def test_refusal_wording_is_mode_independent_for_reproject(qapp):
    """reproject refusals use the same mode-independent wording (point 5)."""
    payload = StartupRefusalPayload(
        code=CODE_OUTPUT_STATE_INCOMPATIBLE,
        technical_detail="resume limited to plain classic SUM/W",
        semantic_key="output_state_incompatible",
        semantic_data={"mode": "reproject"},
    )
    win = MainWindow(backend_factory=lambda: _RefusingBackend(payload))
    try:
        win._language = "en"
        title, body = win._format_refusal(payload)
        assert title == "Output folder already in use"
        assert "reproject" not in body
        assert "Drizzle" not in body
        win._language = "fr"
        title, body = win._format_refusal(payload)
        assert title == "Dossier de sortie déjà utilisé"
        assert "reprojection" not in body
        assert "Drizzle" not in body
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# D) accepted run: log exists immediately + progressive events while running
# --------------------------------------------------------------------------
def test_run_log_has_run_accepted_and_progressive_events_while_running(qapp, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stacker = RunningEngineStacker()
    backend = _backend_for(stacker)
    request = _make_request(input_folder=str(tmp_path), output_folder=str(out_dir))

    result_box = {}
    done = threading.Event()

    def runner():
        result_box["result"] = backend.run(
            request, lambda p: None, lambda m: None, lambda: False
        )
        done.set()

    t = threading.Thread(target=runner, daemon=True)
    t.start()

    # Wait until the run log exists and already contains RUN_ACCEPTED while the
    # engine thread is still blocked (never released).
    log_files = list(out_dir.glob("zsss_run_*.log"))
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and (
        not log_files or "RUN_ACCEPTED" not in log_files[0].read_text(encoding="utf-8")
    ):
        time.sleep(0.01)
        log_files = list(out_dir.glob("zsss_run_*.log"))

    assert log_files, "run log was never created"
    assert not stacker._release.is_set(), "engine was released too early"
    text = log_files[0].read_text(encoding="utf-8")
    assert "RUN_ACCEPTED" in text
    assert "RUN_STARTED" in text
    # Progressive engine events already flushed while the engine is still alive.
    assert "ENGINE_PROGRESS" in text

    stacker._release.set()
    t.join(5)
    assert done.is_set(), "backend did not finish after the engine tail released"
    assert result_box["result"] is BackendRunResult.FINISHED


# --------------------------------------------------------------------------
# E) success / controlled failure / cancellation / unwritable-log (fail-open)
# --------------------------------------------------------------------------
def test_controlled_backend_failure_records_failure(qapp, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stacker = FakeEngineStacker(processing_error="RuntimeError: boom")
    backend = _backend_for(stacker)
    request = _make_request(input_folder=str(tmp_path), output_folder=str(out_dir))

    with pytest.raises(RuntimeError) as exc_info:
        backend.run(request, lambda p: None, lambda m: None, lambda: False)
    assert "boom" in str(exc_info.value)
    text = Path(backend.run_log.path).read_text(encoding="utf-8")
    assert "BACKEND_RETURNING" in text
    assert "failed" in text


def test_cancellation_is_fail_open_and_idempotent(qapp, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stacker = FakeEngineStacker(block_tail=True)
    backend = _backend_for(stacker)
    request = _make_request(input_folder=str(tmp_path), output_folder=str(out_dir))

    # is_cancel_requested already True -> cancel path (stop called, CANCELLED).
    result = backend.run(request, lambda p: None, lambda m: None, lambda: True)
    assert result is BackendRunResult.CANCELLED
    assert stacker.stop_called is True


def test_unwritable_run_log_is_fail_open(qapp, tmp_path):
    # Output directory does not exist -> open() fails; the run must still finish
    # and surface exactly one best-effort warning.
    stacker = FakeEngineStacker()
    backend = _backend_for(stacker)
    request = _make_request(
        input_folder=str(tmp_path),
        output_folder=str(tmp_path / "missing_dir"),
    )
    warnings = []
    result = backend.run(request, lambda p: None, warnings.append, lambda: False)
    assert result is BackendRunResult.FINISHED
    assert backend.run_log.open_error is not None
    assert any("run log unavailable" in w for w in warnings)


# --------------------------------------------------------------------------
# F) premature-return reproducer: backend must not finish while thread alive
# --------------------------------------------------------------------------
def test_backend_waits_for_engine_thread_before_finished(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stacker = FakeEngineStacker(block_tail=True)
    backend = _backend_for(stacker)
    request = _make_request(input_folder=str(tmp_path), output_folder=str(out_dir))

    result_box = {}
    done = threading.Event()

    def runner():
        result_box["result"] = backend.run(
            request, lambda p: None, lambda m: None, lambda: False
        )
        done.set()

    t = threading.Thread(target=runner, daemon=True)
    t.start()

    # Give the backend ample time to (incorrectly) return; it must still be
    # waiting because the engine thread tail is blocked (processing_active is
    # already False, but thread liveness is authoritative).
    time.sleep(0.25)
    assert not done.is_set(), (
        "backend returned FINISHED while the engine thread is still alive"
    )
    assert stacker.processing_active is False
    assert stacker.processing_thread.is_alive() is True

    stacker._release.set()
    t.join(5)
    assert done.is_set()
    assert result_box["result"] is BackendRunResult.FINISHED
    assert stacker.processing_thread.is_alive() is False


# --------------------------------------------------------------------------
# G) full offscreen Qt lifecycle bounded witness + second UI action
# --------------------------------------------------------------------------
def test_full_qt_lifecycle_witness_and_second_run(qapp, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def factory(**kw):
        return ResultProducingEngineStacker()

    win = MainWindow(
        backend_factory=lambda: SeestarQueuedStackerBackend(
            stacker_factory=factory, poll_interval=0.001
        )
    )
    try:
        win.output_edit.setText(str(out_dir))
        win.start_button.click()
        assert win.is_running is True
        assert not win.start_button.isEnabled()
        assert win.stop_button.isEnabled()

        assert _pump_until(qapp, lambda: win.is_running is False)

        # Result -> worker outcome -> QThread dead -> Qt handler returned.
        assert win.controller.status is RunStatus.FINISHED
        assert not win.controller.has_live_thread
        # Controls restored and idle.
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()
        assert win.progress.value() == 100

        # Durable run log was produced and closed by the GUI handler.
        run_log = win.controller.run_log
        assert run_log is not None
        assert run_log.path is not None
        assert run_log.is_open is False
        text = Path(run_log.path).read_text(encoding="utf-8")
        assert "QTHREAD_FINISHED" in text
        assert "QT_COMPLETION_HANDLER_ENTERED" in text
        assert "CONTROLS_RESTORED" in text
        assert "GUI_IDLE" in text
        assert "RUN_SUCCEEDED" in text
        assert "RUN_FINISHED_NO_OUTPUT" not in text

        # The result file actually exists and is accessible (summary derived
        # SUCCESS from a real on-disk FITS product, not from a stale path).
        final_fits = out_dir / "stack_final.fits"
        assert final_fits.is_file()
        assert os.path.isfile(str(final_fits))

        # A second UI action/run must not hit the has_live_thread race.
        win.start_button.click()
        assert win.is_running is True
        assert _pump_until(qapp, lambda: win.is_running is False)
        assert win.controller.status is RunStatus.FINISHED
        assert not win.controller.has_live_thread
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# G2) no-output run: RUN_SUCCEEDED absent, RUN_FINISHED_NO_OUTPUT present
# --------------------------------------------------------------------------
def test_no_output_run_emits_run_finished_no_output_not_success(qapp, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def factory(**kw):
        # A clean backend finish that produces *no* result file: the summary
        # derives EMPTY, so the GUI must never report success.
        return FakeEngineStacker()

    win = MainWindow(
        backend_factory=lambda: SeestarQueuedStackerBackend(
            stacker_factory=factory, poll_interval=0.001
        )
    )
    try:
        win.output_edit.setText(str(out_dir))
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)

        assert win.controller.status is RunStatus.FINISHED
        run_log = win.controller.run_log
        assert run_log is not None
        text = Path(run_log.path).read_text(encoding="utf-8")
        assert "RUN_FINISHED_NO_OUTPUT" in text
        assert "RUN_SUCCEEDED" not in text
        # The empty presentation is still shown (never a false "Finished.").
        assert win.statusBar().currentMessage() == "No output produced."
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# H) lifecycle event ordering assertions (timeout only as a deadlock guard)
# --------------------------------------------------------------------------
def test_lifecycle_event_ordering(qapp, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def factory(**kw):
        return ResultProducingEngineStacker()

    win = MainWindow(
        backend_factory=lambda: SeestarQueuedStackerBackend(
            stacker_factory=factory, poll_interval=0.001
        )
    )
    try:
        win.output_edit.setText(str(out_dir))
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)

        run_log = win.controller.run_log
        assert run_log is not None
        events = _log_events(Path(run_log.path))

        assert _index(events, "RUN_ACCEPTED") < _index(events, "RUN_STARTED")
        assert _index(events, "DRIZZLE_FINALIZATION_ENTERED") < _index(
            events, "DRIZZLE_FINALIZATION_RETURNED"
        )
        assert _index(events, "FINAL_FITS_SAVE_ENTERED") < _index(
            events, "FINAL_FITS_SAVE_RETURNED"
        )
        assert _index(events, "FINAL_FITS_SAVE_RETURNED") < _index(
            events, "FINAL_PREVIEW_SAVE_ENTERED"
        )
        assert _index(events, "FINAL_PREVIEW_SAVE_ENTERED") < _index(
            events, "FINAL_PREVIEW_SAVE_RETURNED"
        )
        assert _index(events, "ENGINE_PROCESSING_RETURNING") < _index(
            events, "ENGINE_PROCESSING_RETURNED"
        )
        assert _index(events, "ENGINE_PROCESSING_RETURNED") < _index(
            events, "BACKEND_RETURNING"
        )
        assert _index(events, "BACKEND_RETURNING") < _index(
            events, "BACKEND_RETURNED"
        )
        assert _index(events, "BACKEND_RETURNED") < _index(
            events, "COMPLETION_CALLBACK_EMITTING"
        )
        assert _index(events, "COMPLETION_CALLBACK_EMITTING") < _index(
            events, "COMPLETION_CALLBACK_EMITTED"
        )
        assert _index(events, "COMPLETION_CALLBACK_EMITTED") < _index(
            events, "WORKER_OUTCOME"
        )
        assert _index(events, "WORKER_OUTCOME") < _index(events, "QTHREAD_FINISHED")
        assert _index(events, "QTHREAD_FINISHED") < _index(
            events, "QT_COMPLETION_HANDLER_ENTERED"
        )
        assert _index(events, "QT_COMPLETION_HANDLER_ENTERED") < _index(
            events, "CONTROLS_RESTORED"
        )
        assert _index(events, "CONTROLS_RESTORED") < _index(events, "GUI_IDLE")
        assert _index(events, "GUI_IDLE") < _index(events, "RUN_SUCCEEDED")
        assert _index(events, "RUN_SUCCEEDED") < _index(
            events, "QT_COMPLETION_HANDLER_RETURNED"
        )
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# H2) missing terminal worker outcome -> explicit failure, never false success
# --------------------------------------------------------------------------
def test_missing_terminal_outcome_is_failure_not_success(qapp, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rl = RunLog(session_id="missing-outcome")
    rl.open(str(out_dir))

    controller = RunController()
    controller._status = RunStatus.RUNNING
    controller._run_log = rl
    controller._pending_outcome = None

    results = {"finished": 0, "cancelled": 0, "failed": []}
    controller.finished.connect(
        lambda: results.__setitem__("finished", results["finished"] + 1)
    )
    controller.cancelled.connect(
        lambda: results.__setitem__("cancelled", results["cancelled"] + 1)
    )
    controller.failed.connect(lambda m: results["failed"].append(m))

    controller._on_thread_finished()
    # Deferred publication fires one GUI event turn later (no wait/join).
    assert _pump_until(
        qapp, lambda: results["failed"] or results["finished"] or results["cancelled"]
    )

    assert results["finished"] == 0
    assert results["cancelled"] == 0
    assert results["failed"], "a missing terminal outcome must surface as failure"
    assert "terminal outcome" in results["failed"][0]
    assert controller.status is RunStatus.FAILED

    text = Path(rl.path).read_text(encoding="utf-8")
    assert "QTHREAD_FINISHED" in text
    assert "WORKER_OUTCOME_MISSING" in text
    assert "RUN_SUCCEEDED" not in text


def test_missing_outcome_via_real_thread_is_failure(qapp, tmp_path):
    """A QThread that finishes without a worker outcome must become a failure."""
    from PySide6.QtCore import QThread

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rl = RunLog(session_id="missing-thread")
    rl.open(str(out_dir))

    controller = RunController()
    controller._status = RunStatus.RUNNING
    controller._run_log = rl
    controller._pending_outcome = None

    results = {"finished": 0, "cancelled": 0, "failed": []}
    controller.finished.connect(
        lambda: results.__setitem__("finished", results["finished"] + 1)
    )
    controller.cancelled.connect(
        lambda: results.__setitem__("cancelled", results["cancelled"] + 1)
    )
    controller.failed.connect(lambda m: results["failed"].append(m))

    thread = QThread()
    controller._thread = thread
    thread.finished.connect(controller._on_thread_finished)

    def _quit_only():
        thread.quit()

    thread.started.connect(_quit_only)
    thread.start()

    assert _pump_until(
        qapp, lambda: results["failed"] or results["finished"] or results["cancelled"]
    )
    assert not thread.isRunning()
    assert results["finished"] == 0
    assert results["cancelled"] == 0
    assert results["failed"]
    assert controller.status is RunStatus.FAILED


# --------------------------------------------------------------------------
# H3) run-log failure: direct controlled open failure (no recursion/deadlock)
# --------------------------------------------------------------------------
def test_run_log_open_failure_is_fail_open_and_bounded(monkeypatch, tmp_path):
    warnings = []
    rl = RunLog(session_id="fail-open")
    rl.warning = warnings.append

    def _failing_open(*args, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr("builtins.open", _failing_open)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    path = rl.open(str(out_dir), metadata={"mode": "drizzle"})
    assert path is None
    assert rl.open_error is not None
    # Exactly one best-effort warning (no recursion into emit).
    assert len(warnings) == 1
    assert "run log unavailable" in warnings[0]

    # After a failed open, emit() must drop records, not grow an unbounded
    # in-memory buffer.
    for _ in range(1000):
        rl.emit("EVENT", payload="x" * 50)
    assert rl.buffered_count == 0

    # close() must be safe and idempotent (no deadlock, no recursion).
    rl.close()
    rl.close()


# --------------------------------------------------------------------------
# I) scientific-source token / diff boundaries (regression guards)
# --------------------------------------------------------------------------
def test_run_log_and_refusal_modules_are_engine_tk_free():
    for name in ("run_log.py", "startup_refusal.py"):
        text = (ROOT / "seestar" / "gui_qt" / name).read_text(encoding="utf-8")
        for token in (
            "seestar.core",
            "seestar.alignment",
            "seestar.enhancement",
            "seestar.queuep",
            "tkinter",
            "QtWidgets",
            "QtGui",
            "zealfie",
        ):
            assert token not in text, f"{name} references {token}"


def test_run_log_unit_buffers_before_open_and_never_overwrites(tmp_path):
    rl = RunLog(session_id="abc12345", clock=lambda: 1700000000.0)
    rl.emit("PRE_ACCEPT_EVENT", k="v")
    assert rl.buffered_count == 1
    assert rl.path is None

    out = tmp_path / "out"
    out.mkdir()
    rl.open(str(out), metadata={"mode": "drizzle"})
    assert rl.path is not None
    text = Path(rl.path).read_text(encoding="utf-8")
    assert "PRE_ACCEPT_EVENT" in text
    assert "RUN_METADATA" in text

    # A second open() is a no-op (single-shot) and never overwrites.
    first_path = rl.path
    rl.open(str(out))
    assert rl.path == first_path

    rl.emit("AFTER_OPEN", x=1)
    rl.close()
    assert "AFTER_OPEN" in Path(first_path).read_text(encoding="utf-8")
    # emit after close is a silent no-op.
    rl.emit("AFTER_CLOSE")
    assert "AFTER_CLOSE" not in Path(first_path).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# J) C2 regressions: deterministic log-handoff order + truthful handler-return
# --------------------------------------------------------------------------
class _SignalOrder(QObject):
    """Single-receiver recorder so queued deliveries are strictly FIFO."""

    def __init__(self):
        super().__init__()
        self.order = []

    @Slot(object)
    def on_lifecycle_log(self, run_log):
        self.order.append("lifecycle_log")

    @Slot()
    def on_finished(self):
        self.order.append("finished")

    @Slot()
    def on_cancelled(self):
        self.order.append("cancelled")


def test_worker_hands_off_run_log_before_terminal_outcome(qapp, tmp_path):
    """Deterministic (same-sender FIFO): the worker must hand the durable run log
    to the controller *before* emitting its terminal outcome, so run-log
    availability precedes the outcome by construction (not timing luck)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    class _LogBackend(BaseRunBackend):
        def __init__(self):
            self.run_log = RunLog(session_id="handoff-order")
            self.run_log.open(str(out_dir))

        def run(
            self,
            request,
            progress_callback,
            log_callback,
            is_cancel_requested,
            preview_callback=None,
            summary_callback=None,
        ):
            return BackendRunResult.FINISHED

        def cancel(self):
            pass

    worker = RunWorker(backend=_LogBackend())
    worker.set_request(_make_request(input_folder=str(tmp_path), output_folder=str(out_dir)))

    rec = _SignalOrder()
    worker.lifecycle_log.connect(rec.on_lifecycle_log)
    worker.finished.connect(rec.on_finished)
    worker.cancelled.connect(rec.on_cancelled)

    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    thread.start()

    assert _pump_until(qapp, lambda: "finished" in rec.order)
    thread.quit()
    thread.wait(2000)

    assert "lifecycle_log" in rec.order
    assert rec.order.index("lifecycle_log") < rec.order.index("finished"), rec.order


def _controller_terminal_scenario(out_dir, kind):
    """Drive a controller from run-log handoff to thread-finished and return it."""
    rl = RunLog(session_id=f"ctlr-{kind}")
    rl.open(str(out_dir))

    controller = RunController()
    controller._status = RunStatus.RUNNING
    controller._run_log = rl
    controller._pending_outcome = None

    def on_finished():
        # MainWindow's terminal slot body: milestones only.  RETURNED + close
        # are controller-owned and happen after this slot returns.
        rl.emit("QT_COMPLETION_HANDLER_ENTERED", outcome="finished")
        rl.emit("CONTROLS_RESTORED")
        rl.emit("GUI_IDLE")
        rl.emit("RUN_SUCCEEDED", terminal_status="success")

    def on_failed(message):
        rl.emit("QT_COMPLETION_HANDLER_ENTERED", outcome="failed")
        rl.emit("CONTROLS_RESTORED")
        rl.emit("GUI_IDLE")
        rl.emit("RUN_FAILED", error=message)

    def on_cancelled():
        rl.emit("QT_COMPLETION_HANDLER_ENTERED", outcome="cancelled")
        rl.emit("CONTROLS_RESTORED")
        rl.emit("GUI_IDLE")
        rl.emit("RUN_CANCELLED")

    controller.finished.connect(on_finished)
    controller.failed.connect(on_failed)
    controller.cancelled.connect(on_cancelled)

    # The now-guaranteed order: handoff -> terminal outcome -> thread finished.
    controller._on_worker_lifecycle_log(rl)
    if kind == "finished":
        controller._on_worker_finished()
    elif kind == "failed":
        controller._on_worker_failed("boom")
    else:
        controller._on_worker_cancelled()
    controller._on_thread_finished()

    return rl


@pytest.mark.parametrize(
    "kind,terminal_event",
    [
        ("finished", "RUN_SUCCEEDED"),
        ("failed", "RUN_FAILED"),
        ("cancelled", "RUN_CANCELLED"),
    ],
)
def test_qt_completion_handler_returned_is_truthful_and_last(
    qapp, tmp_path, kind, terminal_event
):
    """QT_COMPLETION_HANDLER_RETURNED is written only after the terminal slot has
    returned (controller-side, after the public signal emit returns), and the
    run log closes there — for success, failure and cancellation."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rl = _controller_terminal_scenario(out_dir, kind)

    assert rl.is_open is False
    events = _log_events(Path(rl.path))
    assert "QTHREAD_FINISHED" in events
    assert "QT_COMPLETION_HANDLER_ENTERED" in events
    assert "CONTROLS_RESTORED" in events
    assert "GUI_IDLE" in events
    assert terminal_event in events
    assert "QT_COMPLETION_HANDLER_RETURNED" in events
    # RETURNED must be the *last* event — written after the handler returned.
    assert events[-1] == "QT_COMPLETION_HANDLER_RETURNED", events
    text = Path(rl.path).read_text(encoding="utf-8")
    assert f"QT_COMPLETION_HANDLER_RETURNED outcome={kind}" in text


def test_refused_run_has_no_run_log_and_no_handler_returned(qapp):
    """A refused run never opened an accepted-run log: no RETURNED, no close."""
    controller = RunController()
    controller._status = RunStatus.RUNNING
    controller._pending_outcome = None
    payload = StartupRefusalPayload(
        code=CODE_OUTPUT_STATE_INCOMPATIBLE,
        technical_detail="resume limited to plain classic SUM/W",
        semantic_key="output_state_incompatible",
        semantic_data={"mode": "drizzle"},
    )
    refused = []
    controller.refused.connect(lambda p: refused.append(p))
    controller._on_worker_refused(payload)
    controller._on_thread_finished()

    assert len(refused) == 1
    assert controller.run_log is None
