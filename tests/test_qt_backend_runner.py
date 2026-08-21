"""M4 seam tests: the Qt backend runner adapter.

These tests exercise the new runner abstraction
(:mod:`seestar.gui_qt.backend_runner`) under the ``offscreen`` Qt platform:

* the default simulated backend still drives the M3 lifecycle unchanged,
* ``RunWorker``/``RunController`` accept an injected fake backend that receives
  the canonical :class:`RunRequest`, emits progress/log, and completes,
* injected-backend cancellation calls ``cancel()`` and yields cancelled
  semantics,
* ``SeestarQueuedStackerBackend`` does **not** import the engine when merely
  imported, and (with a fake stacker factory) maps ``align_on_disk``,
  ``set_progress_callback`` and ``start_processing(**backend_kwargs)`` correctly
  without running the real backend,
* the source surface stays free of forbidden engine/Tk tokens.

No real stacking is ever performed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import RunController, RunStatus
from seestar.gui_qt import create_application
from seestar.gui_qt.backend_runner import (
    BackendRunResult,
    BaseRunBackend,
    SeestarQueuedStackerBackend,
)
from seestar.gui_qt.run_bridge import build_run_request
from seestar.gui_qt.settings_state import QtSettingsState

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


def _make_request(**overrides):
    state = QtSettingsState()
    for key, value in overrides.items():
        setattr(state, key, value)
    return build_run_request(state)


# --------------------------------------------------------------------------
# Fakes (test-controlled, no real engine)
# --------------------------------------------------------------------------
class FakeBackend(BaseRunBackend):
    """A fake run backend: records the request and drives callbacks.

    ``complete_immediately=True`` finishes normally; ``False`` polls
    ``is_cancel_requested()`` until cancellation and returns CANCELLED.
    """

    def __init__(self, complete_immediately: bool = True) -> None:
        self.complete_immediately = complete_immediately
        self.run_calls = []
        self.cancel_called = False

    def run(
        self,
        request,
        progress_callback,
        log_callback,
        is_cancel_requested,
        preview_callback=None,
    ):
        self.run_calls.append(request)
        log_callback("fake backend started")
        progress_callback(10)
        progress_callback(50)
        if self.complete_immediately:
            progress_callback(100)
            log_callback("fake backend finished")
            return BackendRunResult.FINISHED
        while not is_cancel_requested():
            time.sleep(0.001)
        return BackendRunResult.CANCELLED

    def cancel(self):
        self.cancel_called = True


class FakeStacker:
    """A fake ``SeestarQueuedStacker``-shaped object recording its API usage."""

    def __init__(self, *, stay_running: bool = False, **kwargs) -> None:
        self.init_kwargs = dict(kwargs)
        self.stay_running = stay_running
        self.align_on_disk = None
        self.progress_cb = None
        self.start_kwargs = None
        self.start_count = 0
        self.stop_called = False
        self._running = False

    def set_progress_callback(self, cb) -> None:
        self.progress_cb = cb

    def start_processing(self, **kwargs):
        self.start_kwargs = dict(kwargs)
        self.start_count += 1
        self._running = True
        return True

    def is_running(self) -> bool:
        if self.stay_running:
            return self._running
        self._running = False
        return False

    def stop(self) -> None:
        self.stop_called = True
        self._running = False


def _make_stackers_factory(instances, *, stay_running: bool = False):
    def factory(**kwargs):
        stacker = FakeStacker(stay_running=stay_running, **kwargs)
        instances.append(stacker)
        return stacker

    return factory


# --------------------------------------------------------------------------
# Source-token / import-hygiene invariants
# --------------------------------------------------------------------------
def test_backend_runner_source_is_engine_and_tk_free():
    text = (ROOT / "seestar" / "gui_qt" / "backend_runner.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "seestar.core",
        "seestar.alignment",
        "seestar.enhancement",
        "seestar.queuep",
        "tkinter",
        "seestar.gui.settings",
        "seestar.gui.main_window",
        "zesolver_adapter",
        "zesolver.api",
        "zealfie",
        "QtWidgets",
        "QtGui",
    )
    for token in forbidden:
        assert token not in text, f"backend_runner.py references {token}"
    # The engine reachability must be lazy and split (no dotted engine token).
    assert "importlib.import_module" in text
    assert "SeestarQueuedStacker" in text
    assert "_load_stackers_class" in text


def test_backend_runner_import_does_not_pull_engine():
    """Importing the runner (or the whole package) must not import the engine."""
    code = (
        "import sys\n"
        "import seestar.gui_qt.backend_runner  # noqa: F401\n"
        "import seestar.gui_qt  # noqa: F401\n"
        "_bad = [m for m in sys.modules\n"
        "        if m.startswith('seestar.queuep')\n"
        "        or m.startswith('seestar.core')\n"
        "        or m.startswith('seestar.alignment')\n"
        "        or m.startswith('seestar.enhancement')\n"
        "        or m.startswith('tkinter')]\n"
        "if _bad:\n"
        "    print('BAD_MODULES:', _bad)\n"
        "    sys.exit(1)\n"
        "print('IMPORT_HYGIENE_OK')\n"
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
        f"import hygiene violated: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )


# --------------------------------------------------------------------------
# Default simulated backend still drives the lifecycle
# --------------------------------------------------------------------------
def test_default_backend_is_simulated(qapp):
    controller = RunController()
    logs = []
    finished = []
    controller.log_message.connect(logs.append)
    controller.finished.connect(lambda: finished.append(True))
    try:
        controller.start(_make_request(batch_size=4), steps=5, step_delay_ms=1)
        assert _pump_until(qapp, lambda: controller.status is RunStatus.FINISHED)
        assert finished == [True]
        assert any("Simulated run started" in line for line in logs)
        assert any("Simulated run finished" in line for line in logs)
    finally:
        controller.shutdown()


# --------------------------------------------------------------------------
# Injected fake backend (worker/controller seam)
# --------------------------------------------------------------------------
def test_worker_runs_injected_fake_backend(qapp):
    fake = FakeBackend(complete_immediately=True)
    controller = RunController()
    progress = []
    logs = []
    finished = []
    failed = []
    controller.progress_changed.connect(progress.append)
    controller.log_message.connect(logs.append)
    controller.finished.connect(lambda: finished.append(True))
    controller.failed.connect(failed.append)

    request = _make_request(batch_size=2)
    try:
        controller.start(request, backend=fake)
        assert controller.is_running is True
        assert _pump_until(qapp, lambda: controller.status is RunStatus.FINISHED)

        assert finished == [True]
        assert failed == []
        # The backend received the *canonical* RunRequest (same object handed in).
        assert len(fake.run_calls) == 1
        from seestar.gui.run_config import RunRequest as Canonical

        assert isinstance(fake.run_calls[0], Canonical)
        assert fake.run_calls[0] is request
        assert fake.run_calls[0].backend_kwargs["batch_size"] == 2
        # Progress/log flowed through the signals in order.
        assert progress == [10, 50, 100]
        assert "fake backend started" in logs
        assert "fake backend finished" in logs
        assert not controller.has_live_thread
    finally:
        controller.shutdown()


def test_injected_fake_backend_cancellation_calls_cancel(qapp):
    fake = FakeBackend(complete_immediately=False)
    controller = RunController()
    cancelled = []
    finished = []
    controller.cancelled.connect(lambda: cancelled.append(True))
    controller.finished.connect(lambda: finished.append(True))
    try:
        controller.start(_make_request(), backend=fake)
        controller.cancel()
        assert _pump_until(qapp, lambda: controller.status is RunStatus.CANCELLED)

        assert cancelled == [True]
        assert finished == []
        assert fake.cancel_called is True
        assert controller.is_running is False
        assert not controller.has_live_thread
    finally:
        controller.shutdown()


def test_controller_rejects_non_backend_type(qapp):
    controller = RunController()
    with pytest.raises(TypeError):
        controller.start(_make_request(), backend=object())
    assert controller.is_running is False


# --------------------------------------------------------------------------
# SeestarQueuedStackerBackend (fake stacker factory — no real engine)
# --------------------------------------------------------------------------
def test_seestar_backend_maps_request_to_stackers():
    instances = []
    backend = SeestarQueuedStackerBackend(
        stacker_factory=_make_stackers_factory(instances),
        poll_interval=0.001,
        settings="FAKE_SETTINGS",
    )
    request = _make_request(batch_size=4, input_folder="/in", output_folder="/out")

    progress = []
    logs = []
    result = backend.run(request, progress.append, logs.append, lambda: False)

    assert result is BackendRunResult.FINISHED
    assert len(instances) == 1
    stacker = instances[0]

    # align_on_disk came from the canonical RunRequest (batch_size 4 -> True).
    assert stacker.align_on_disk is True
    # start_processing must NOT receive the unsupported seam-only field; it is
    # applied to the stacker instance instead.
    assert "stack_final_combine" not in stacker.start_kwargs
    expected_start_kwargs = dict(request.backend_kwargs)
    expected_start_kwargs.pop("stack_final_combine")
    assert stacker.start_kwargs == expected_start_kwargs
    assert stacker.stack_final_combine == request.backend_kwargs["stack_final_combine"]
    assert stacker.start_count == 1
    # progress callback was installed and is callable.
    assert callable(stacker.progress_cb)
    # init kwargs forwarded (settings).
    assert stacker.init_kwargs.get("settings") == "FAKE_SETTINGS"


def test_seestar_backend_applies_stack_final_combine_to_settings():
    """The selected final-combine value reaches the stacker and its settings."""

    class Settings:
        def __init__(self) -> None:
            self.stack_final_combine = "mean"

    instances = []

    def factory(**kwargs):
        stacker = FakeStacker(**kwargs)
        stacker.settings = Settings()
        instances.append(stacker)
        return stacker

    backend = SeestarQueuedStackerBackend(
        stacker_factory=factory,
        poll_interval=0.001,
    )
    request = _make_request(stack_final_combine="winsorized_sigma_clip")

    result = backend.run(request, lambda p: None, lambda m: None, lambda: False)

    assert result is BackendRunResult.FINISHED
    stacker = instances[0]
    assert stacker.stack_final_combine == "winsorized_sigma_clip"
    assert stacker.settings.stack_final_combine == "winsorized_sigma_clip"
    assert "stack_final_combine" not in stacker.start_kwargs


def test_seestar_backend_align_on_disk_false_and_progress_cb_mapping():
    instances = []
    backend = SeestarQueuedStackerBackend(
        stacker_factory=_make_stackers_factory(instances),
        poll_interval=0.001,
    )
    request = _make_request(batch_size=0)  # in-memory -> align_on_disk False

    progress = []
    logs = []
    result = backend.run(request, progress.append, logs.append, lambda: False)

    assert result is BackendRunResult.FINISHED
    stacker = instances[0]
    assert stacker.align_on_disk is False

    # The stacker's progress callback signature is (message, progress, level).
    stacker.progress_cb("50% done", 50.0, "INFO")
    assert logs[-1] == "50% done"
    assert progress[-1] == 50

    # A message-only call (progress=None) logs but does not emit percent.
    before = len(progress)
    stacker.progress_cb("banner only", None, None)
    assert logs[-1] == "banner only"
    assert len(progress) == before


def test_seestar_backend_cancel_calls_stop():
    instances = []
    backend = SeestarQueuedStackerBackend(
        stacker_factory=_make_stackers_factory(instances),
        poll_interval=0.001,
    )
    request = _make_request(batch_size=4)

    # is_cancel_requested is already True -> the run must stop the backend.
    result = backend.run(request, lambda p: None, lambda m: None, lambda: True)

    assert result is BackendRunResult.CANCELLED
    assert instances[0].stop_called is True


def test_seestar_backend_false_start_raises():
    instances = []

    class NonStartingStacker(FakeStacker):
        def start_processing(self, **kwargs):
            super().start_processing(**kwargs)
            return False

    def factory(**kwargs):
        stacker = NonStartingStacker(**kwargs)
        instances.append(stacker)
        return stacker

    backend = SeestarQueuedStackerBackend(stacker_factory=factory)
    with pytest.raises(RuntimeError):
        backend.run(
            _make_request(batch_size=4),
            lambda p: None,
            lambda m: None,
            lambda: False,
        )


def test_controller_cancel_drives_real_backend_stop(qapp):
    """End-to-end: the lazy real-backend adapter through the full lifecycle."""
    instances = []
    backend = SeestarQueuedStackerBackend(
        stacker_factory=_make_stackers_factory(instances, stay_running=True),
        poll_interval=0.001,
    )
    controller = RunController()
    cancelled = []
    controller.cancelled.connect(lambda: cancelled.append(True))
    try:
        controller.start(_make_request(batch_size=4), backend=backend)
        # Wait until the worker actually reached the backend.
        assert _pump_until(
            qapp, lambda: len(instances) == 1 and instances[0].start_count == 1
        )
        controller.cancel()
        assert _pump_until(qapp, lambda: controller.status is RunStatus.CANCELLED)

        assert cancelled == [True]
        assert instances[0].stop_called is True
        assert not controller.has_live_thread
    finally:
        controller.shutdown()
