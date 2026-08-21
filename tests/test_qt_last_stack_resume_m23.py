"""M23 tests: last-stack output pre-fill + run summary + auto-resume seam.

Closes checklist 10.1 (last-stack display / resume parity) across three
sub-items, all engine-free (fake backends/stackers/runners) so no real
stacking, no FITS/PNG side effects beyond a small test fixture, and no Tk:

(a) last-stack -> output pre-fill parity:
    * manual edit / browse / persisted-load pre-fill the output folder from the
      last-stack path *only when the output folder is empty* (Tk
      ``_on_last_stack_changed`` guard),
    * a non-empty output folder is never clobbered,
    * ``_sync_state_from_controls`` still mirrors both fields into the model.

(b) processing-summary payload + dialog:
    * the backend adapter emits a :class:`SummaryPayload` (computed lazily from
      ``final.fits`` NIMAGES/TOTEXP) via the worker->controller->MainWindow
      ``summary`` signal,
    * the Qt dialog formats/display the payload (Status / Total Processing
      Time / Files Attempted / header fields),
    * shown at the end of both the regular and the boring run paths,
    * ``summary_payload`` never imports astropy at module level (hygiene).

(c) engine auto-resume seam:
    * the Qt run path forwards ``output_folder`` as ``start_processing``
      ``output_dir`` (the exact folder ``_can_resume`` checks), and the resume
      condition (memmap_accumulators/cumulative_SUM.npy +
      cumulative_WHT.npy + batches_count.txt) evaluates True against that
      forwarded folder.

No real stacking, no engine, no Tk. ``QT_QPA_PLATFORM=offscreen`` is set
defensively before any ``QApplication``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt.backend_runner import (
    BackendRunResult,
    BaseRunBackend,
    SeestarQueuedStackerBackend,
)
from seestar.gui_qt.boring_runner import BoringRunnerBase
from seestar.gui_qt.summary_payload import (
    SummaryPayload,
    build_summary_payload,
    derive_terminal_status,
    read_final_fits_header,
)

ROOT = Path(__file__).resolve().parents[1]


def _pump_until(qapp: QApplication, predicate, timeout_ms: int = 5000) -> bool:
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


@pytest.fixture()
def window(qapp):
    win = MainWindow()
    yield win
    win.shutdown()


# --------------------------------------------------------------------------
# (a) last-stack -> output pre-fill parity
# --------------------------------------------------------------------------
def test_manual_edit_prefills_output_when_empty(window):
    assert window.output_edit.text() == ""
    window.last_stack_edit.setText("/data/runs/last.fit")
    assert window.output_edit.text() == "/data/runs"
    # The model stays in sync with the pre-fill.
    state = window.collect_settings_state()
    assert state.last_stack_path == "/data/runs/last.fit"
    assert state.output_folder == "/data/runs"


def test_manual_edit_does_not_prefill_when_output_set(window):
    window.output_edit.setText("/keep/me")
    window.last_stack_edit.setText("/data/runs/last.fit")
    assert window.output_edit.text() == "/keep/me"
    state = window.collect_settings_state()
    assert state.output_folder == "/keep/me"
    assert state.last_stack_path == "/data/runs/last.fit"


def test_manual_edit_clears_to_empty_prefills_on_next_edit(window):
    window.last_stack_edit.setText("/data/runs/last.fit")
    assert window.output_edit.text() == "/data/runs"
    # Clearing the output re-arms the pre-fill on the next last-stack change.
    window.output_edit.setText("")
    window.last_stack_edit.setText("/data/runs/other.fit")
    assert window.output_edit.text() == "/data/runs"


def test_browse_prefills_output_when_empty(window, monkeypatch):
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: ("/picked/folder/stack.fits", "")),
    )
    window._browse_last_stack()
    assert window.last_stack_edit.text() == "/picked/folder/stack.fits"
    assert window.output_edit.text() == "/picked/folder"


def test_persisted_load_prefills_output_when_empty(tmp_path):
    p = str(tmp_path / "seestar_settings.json")
    import json

    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"last_stack_path": "/persist/folder/last.fit", "output_folder": ""}, fh)
    win = MainWindow(settings_path=p)
    try:
        assert win.last_stack_edit.text() == "/persist/folder/last.fit"
        assert win.output_edit.text() == "/persist/folder"
        state = win.collect_settings_state()
        assert state.output_folder == "/persist/folder"
    finally:
        win.shutdown()


def test_persisted_load_does_not_clobber_nonempty_output(tmp_path):
    p = str(tmp_path / "seestar_settings.json")
    import json

    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"last_stack_path": "/persist/folder/last.fit", "output_folder": "/already/set"}, fh)
    win = MainWindow(settings_path=p)
    try:
        assert win.output_edit.text() == "/already/set"
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (b) summary payload + dialog
# --------------------------------------------------------------------------
REAL_OUTPUT_NAMES = [
    "final.fits",
    "stack_final_classic_sumw.fit",
    "stack_final_drizzle_final.fit",
    "stack_final_classic_reproject.fit",
]


@pytest.mark.parametrize("filename", REAL_OUTPUT_NAMES)
def test_build_summary_payload_reads_final_fits_header(tmp_path, filename):
    from astropy.io import fits

    out = tmp_path / "out"
    out.mkdir()
    hdu = fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32))
    hdu.header["NIMAGES"] = 12
    hdu.header["TOTEXP"] = 30.0
    real = out / filename
    hdu.writeto(real, overwrite=True)

    # The real product path is the source of truth (not <output>/final.fits).
    payload = build_summary_payload(
        status="finished",
        duration_seconds=65.0,
        files_attempted=12,
        output_dir=str(out),
        final_stack_path=str(real),
    )
    assert payload.status == "finished"
    assert payload.final_stack_file == str(real)
    assert payload.final_stack_exists is True
    assert payload.images_in_final_stack == 12
    assert payload.total_exposure_seconds == pytest.approx(30.0)
    assert payload.can_open_output is True


def test_build_summary_payload_legacy_final_fits_fallback(tmp_path):
    from astropy.io import fits

    out = tmp_path / "out"
    out.mkdir()
    hdu = fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32))
    hdu.header["NIMAGES"] = 12
    hdu.header["TOTEXP"] = 30.0
    hdu.writeto(out / "final.fits", overwrite=True)

    # No final_stack_path -> legacy <output>/final.fits convention is kept.
    payload = build_summary_payload(
        status="finished",
        duration_seconds=65.0,
        files_attempted=12,
        output_dir=str(out),
    )
    assert payload.status == "finished"
    assert payload.final_stack_file == str(out / "final.fits")
    assert payload.final_stack_exists is True
    assert payload.images_in_final_stack == 12
    assert payload.total_exposure_seconds == pytest.approx(30.0)
    assert payload.can_open_output is True


def test_build_summary_payload_missing_final_stack_path(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    missing = out / "stack_final_drizzle_final.fit"

    payload = build_summary_payload(
        status="finished",
        duration_seconds=65.0,
        files_attempted=12,
        output_dir=str(out),
        final_stack_path=str(missing),
    )
    assert payload.final_stack_file == str(missing)
    assert payload.final_stack_exists is False
    assert payload.images_in_final_stack is None
    assert payload.total_exposure_seconds is None
    # The existing presentation helper maps the missing product to empty/no-output.
    assert derive_terminal_status(payload) == "empty"


def test_read_final_fits_header_missing_path_returns_empty():
    assert read_final_fits_header("") == {}
    assert read_final_fits_header("/no/such/dir/final.fits") == {}


def test_summary_payload_module_is_astropy_free_at_module_level():
    text = (ROOT / "seestar" / "gui_qt" / "summary_payload.py").read_text(
        encoding="utf-8"
    )
    stripped = [ln.strip() for ln in text.splitlines()]
    for line in stripped:
        assert not line.startswith(("import astropy", "from astropy")), (
            f"summary_payload.py imports astropy at top level: {line!r}"
        )
    # The astropy reach is a lazy importlib call with a split string literal.
    assert "importlib.import_module" in text
    assert '"astropy"' in text


def test_summary_payload_import_does_not_pull_astropy():
    code = (
        "import sys\n"
        "import seestar.gui_qt.summary_payload  # noqa: F401\n"
        "_bad = [m for m in sys.modules if m.split('.')[0] == 'astropy']\n"
        "if _bad:\n"
        "    print('BAD_MODULES:', _bad)\n"
        "    sys.exit(1)\n"
        "print('SUMMARY_HYGIENE_OK')\n"
    )
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert proc.returncode == 0, (
        f"summary_payload import hygiene violated: stdout={proc.stdout!r} "
        f"stderr={proc.stderr!r}"
    )


def _summary_label(dialog):
    labels = dialog.findChildren(QLabel)
    assert labels, "summary dialog has no QLabel"
    return labels[0].text()


class FakeSummaryBackend(BaseRunBackend):
    """Finishes immediately and emits a fixed summary payload."""

    def __init__(self, payload: SummaryPayload) -> None:
        self.payload = payload
        self.summary_calls = []

    def run(
        self,
        request,
        progress_callback,
        log_callback,
        is_cancel_requested,
        preview_callback=None,
        summary_callback=None,
    ):
        progress_callback(100)
        if summary_callback is not None:
            self.summary_calls.append(self.payload)
            summary_callback(self.payload)
        return BackendRunResult.FINISHED

    def cancel(self):
        pass


def test_regular_run_end_shows_summary_dialog(qapp):
    payload = SummaryPayload(
        status="finished",
        duration_seconds=65.0,
        files_attempted=12,
        final_stack_file="/out/final.fits",
        final_stack_exists=True,
        images_in_final_stack=12,
        total_exposure_seconds=30.0,
        can_open_output=True,
    )
    backend = FakeSummaryBackend(payload)
    win = MainWindow(backend_factory=lambda: backend)
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)
        assert backend.summary_calls == [payload]
        assert win._summary_dialog is not None
        text = _summary_label(win._summary_dialog)
        assert "Status: SUCCESS" in text
        assert "Total Processing Time: 1:05" in text
        assert "Files Attempted: 12" in text
        assert "Images in Final Stack: 12" in text
        assert "Total Exposure (Final Stack): 0:30" in text
        # The dialog offers Open Output only when can_open_output is set.
        from PySide6.QtWidgets import QDialogButtonBox

        buttons = win._summary_dialog.findChildren(QDialogButtonBox)
        assert buttons
    finally:
        win.shutdown()


def test_regular_run_no_summary_when_backend_emits_none(qapp):
    """A backend that emits no summary (narrow signature) must not crash the UI."""

    class NoSummaryBackend(BaseRunBackend):
        def run(
            self,
            request,
            progress_callback,
            log_callback,
            is_cancel_requested,
            preview_callback=None,
        ):
            progress_callback(100)
            return BackendRunResult.FINISHED

        def cancel(self):
            pass

    win = MainWindow(backend_factory=NoSummaryBackend)
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)
        assert win._summary_dialog is None
        assert win.is_running is False
    finally:
        win.shutdown()


class FakeSummaryBoringRunner(BoringRunnerBase):
    """Records the request and emits a summary then finished."""

    def __init__(self) -> None:
        super().__init__()
        self.start_calls = []
        self.cancel_called = False
        self._active = False

    def start(self, request) -> None:
        self.start_calls.append(request)
        self._active = True
        self.started.emit()

    def cancel(self) -> None:
        self.cancel_called = True
        self._active = False

    def is_running(self) -> bool:
        return self._active

    def signal_finished(self, payload: SummaryPayload) -> None:
        self._active = False
        self.summary.emit(payload)
        self.finished.emit(0)


def test_boring_run_end_shows_summary_dialog(qapp, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "a.fits").write_bytes(b"")
    csv_path = input_dir / "stack_plan.csv"
    csv_path.write_text("a.fits\n", encoding="utf-8")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    fakes = []

    def factory():
        fake = FakeSummaryBoringRunner()
        fakes.append(fake)
        return fake

    win = MainWindow(boring_runner_factory=factory)
    try:
        win.input_edit.setText(str(input_dir))
        win.output_edit.setText(str(output_dir))
        win.batch_spin.setValue(1)
        win.start_button.click()
        assert len(fakes) == 1
        payload = SummaryPayload(
            status="finished",
            duration_seconds=7.0,
            files_attempted=1,
            final_stack_file=str(output_dir / "final.fits"),
            final_stack_exists=True,
            images_in_final_stack=1,
            total_exposure_seconds=10.0,
            can_open_output=True,
        )
        fakes[0].signal_finished(payload)
        assert win.is_running is False
        assert win._summary_dialog is not None
        text = _summary_label(win._summary_dialog)
        assert "Status: SUCCESS" in text
        assert "Files Attempted: 1" in text
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (c) auto-resume seam: output_folder reaches start_processing output_dir
# --------------------------------------------------------------------------
class RecordingStacker:
    """Fake stacker recording start_processing kwargs and resume state."""

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = dict(kwargs)
        self.align_on_disk = None
        self.progress_cb = None
        self.start_kwargs = None
        self.output_folder = None
        self.processed_files_count = 7
        self.stop_called = False

    def set_progress_callback(self, cb) -> None:
        self.progress_cb = cb

    def start_processing(self, **kwargs):
        self.start_kwargs = dict(kwargs)
        self.output_folder = kwargs.get("output_dir")
        return True

    def is_running(self) -> bool:
        return False

    def stop(self) -> None:
        self.stop_called = True


def _backend_for_stackers(instances):
    def factory(**kwargs):
        stacker = RecordingStacker(**kwargs)
        instances.append(stacker)
        return stacker

    return SeestarQueuedStackerBackend(stacker_factory=factory, poll_interval=0.001)


def _request(output_folder="/out"):
    from seestar.gui_qt.settings_state import QtSettingsState
    from seestar.gui_qt.run_bridge import build_run_request

    state = QtSettingsState()
    state.input_folder = "/in"
    state.output_folder = output_folder
    state.batch_size = 4
    return build_run_request(state)


def test_qt_run_path_forwards_output_dir_to_engine():
    instances = []
    backend = _backend_for_stackers(instances)
    summaries = []
    result = backend.run(
        _request(output_folder="/out"),
        lambda p: None,
        lambda m: None,
        lambda: False,
        summary_callback=summaries.append,
    )
    assert result is BackendRunResult.FINISHED
    assert instances[0].start_kwargs["output_dir"] == "/out"
    # The summary payload also carried the same output folder.
    assert len(summaries) == 1
    assert summaries[0].final_stack_file == "/out/final.fits"
    assert summaries[0].files_attempted == 7


def test_backend_emit_summary_uses_final_stacked_path(tmp_path):
    """The real engine product path (final_stacked_path) becomes the summary's
    source of truth, replacing the hardcoded <output_dir>/final.fits."""
    from astropy.io import fits

    out = tmp_path / "out"
    out.mkdir()
    real = out / "stack_final_drizzle_final.fit"
    hdu = fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32))
    hdu.header["NIMAGES"] = 5
    hdu.header["TOTEXP"] = 45.0
    hdu.writeto(real, overwrite=True)

    class FinalPathStacker:
        def __init__(self, **kwargs):
            self.align_on_disk = None
            self.output_folder = str(out)
            self.final_stacked_path = str(real)
            self.processed_files_count = 5
            self.stop_called = False

        def set_progress_callback(self, cb):
            pass

        def start_processing(self, **kwargs):
            return True

        def is_running(self):
            return False

        def stop(self):
            self.stop_called = True

    backend = SeestarQueuedStackerBackend(
        stacker_factory=lambda **kw: FinalPathStacker(**kw), poll_interval=0.001
    )
    summaries = []
    result = backend.run(
        _request(output_folder=str(out)),
        lambda p: None,
        lambda m: None,
        lambda: False,
        summary_callback=summaries.append,
    )
    assert result is BackendRunResult.FINISHED
    assert len(summaries) == 1
    payload = summaries[0]
    assert payload.final_stack_file == str(real)
    assert payload.final_stack_exists is True
    assert payload.images_in_final_stack == 5
    assert payload.total_exposure_seconds == pytest.approx(45.0)
    assert payload.can_open_output is True
    # derive_terminal_status works with the new payload: a real product -> success.
    assert derive_terminal_status(payload) == "success"


def test_auto_resume_condition_matches_forwarded_output_dir(tmp_path):
    # Build the exact engine ``_can_resume`` artifact set in the output folder.
    memdir = tmp_path / "memmap_accumulators"
    memdir.mkdir()
    np.lib.format.open_memmap(
        memdir / "cumulative_SUM.npy", mode="w+", dtype=np.float32, shape=(2, 2, 3)
    )[:]
    np.lib.format.open_memmap(
        memdir / "cumulative_WHT.npy", mode="w+", dtype=np.float32, shape=(2, 2)
    )[:]
    (tmp_path / "batches_count.txt").write_text("2", encoding="utf-8")

    instances = []
    backend = _backend_for_stackers(instances)
    backend.run(
        _request(output_folder=str(tmp_path)),
        lambda p: None,
        lambda m: None,
        lambda: False,
    )

    output_dir = instances[0].start_kwargs["output_dir"]
    assert output_dir == str(tmp_path)

    # The exact files SeestarQueuedStacker._can_resume(Path(self.output_folder))
    # requires all exist under the folder the Qt run path forwarded.
    required = [
        "memmap_accumulators/cumulative_SUM.npy",
        "memmap_accumulators/cumulative_WHT.npy",
        "batches_count.txt",
    ]
    assert all((Path(output_dir) / r).exists() for r in required)
