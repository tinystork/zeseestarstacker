"""M3 seam tests: Qt worker/controller lifecycle (progress/log/cancel/finish).

These tests exercise the PySide6 run lifecycle — :class:`RunController` +
:class:`RunWorker` — under the ``offscreen`` Qt platform plugin, with **no**
scientific backend and **no** Tk.  They verify:

* a stub run completes and emits progress/log/finished without backend imports,
* cancellation leaves the controller idle and the QThread reaped,
* the MainWindow start/stop buttons drive the lifecycle through a canonical
  :class:`~seestar.gui_qt.run_bridge.RunRequest`,
* shutdown/close tears down a live run idempotently,
* the worker/controller modules only use QtCore (no widgets, no Tk, no engine),
* fresh-process import hygiene still holds.
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

from seestar.gui_qt import MainWindow, RunController, RunStatus, RunWorker
from seestar.gui_qt import create_application
from seestar.gui_qt.run_bridge import RunRequest, build_run_request
from seestar.gui_qt.settings_state import QtSettingsState

ROOT = Path(__file__).resolve().parents[1]


def _pump_until(qapp: QApplication, predicate, timeout_ms: int = 5000) -> bool:
    """Pump the GUI event loop until ``predicate`` is true (or timeout).

    The worker runs on a separate QThread; pumping delivers its queued signals
    to the GUI thread without blocking it.
    """
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


def _make_request(**overrides) -> RunRequest:
    state = QtSettingsState()
    for key, value in overrides.items():
        setattr(state, key, value)
    return build_run_request(state)


# --------------------------------------------------------------------------
# Public surface
# --------------------------------------------------------------------------
def test_gui_qt_exports_lifecycle_components():
    import seestar.gui_qt as gui_qt

    assert gui_qt.RunController is RunController
    assert gui_qt.RunWorker is RunWorker
    assert gui_qt.RunStatus is RunStatus


# --------------------------------------------------------------------------
# Controller completes a stub run (no backend imports)
# --------------------------------------------------------------------------
def test_controller_completes_stub_run(qapp):
    controller = RunController()
    progress = []
    logs = []
    finished = []
    failed = []
    cancelled = []

    controller.progress_changed.connect(progress.append)
    controller.log_message.connect(logs.append)
    controller.finished.connect(lambda: finished.append(True))
    controller.failed.connect(failed.append)
    controller.cancelled.connect(lambda: cancelled.append(True))

    try:
        controller.start(_make_request(batch_size=4), steps=5, step_delay_ms=1)
        assert controller.is_running is True
        assert _pump_until(qapp, lambda: controller.status is RunStatus.FINISHED)

        assert finished == [True]
        assert failed == []
        assert cancelled == []
        assert progress[-1] == 100
        assert progress == sorted(progress)
        assert any("Simulated run started" in line for line in logs)
        assert any("Simulated run finished" in line for line in logs)
        assert not controller.has_live_thread
    finally:
        controller.shutdown()


def test_worker_controller_use_qtcore_only():
    """Worker/controller must not import QtWidgets (they must not touch widgets)."""
    pkg_dir = ROOT / "seestar" / "gui_qt"
    for name in ("run_worker.py", "run_controller.py"):
        text = (pkg_dir / name).read_text(encoding="utf-8")
        assert "QtWidgets" not in text, f"{name} imports QtWidgets"
        assert "QtGui" not in text, f"{name} imports QtGui"


def test_worker_controller_source_is_tk_engine_free():
    """No Tk/engine/private-ZeSolver tokens anywhere in the lifecycle modules."""
    pkg_dir = ROOT / "seestar" / "gui_qt"
    forbidden = (
        "seestar.core",
        "seestar.alignment",
        "seestar.enhancement",
        "seestar.queuep",
        "tkinter",
        "seestar.gui.settings",
        "seestar.gui.main_window",
        "seestar.gui.boring_stack",
        "zesolver_adapter",
        "zesolver.api",
        "zealfie",
    )
    for name in ("run_worker.py", "run_controller.py"):
        text = (pkg_dir / name).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{name} references {token}"


# --------------------------------------------------------------------------
# Cancellation / shutdown semantics
# --------------------------------------------------------------------------
def test_cancellation_leaves_controller_idle_and_thread_reaped(qapp):
    controller = RunController()
    cancelled = []
    finished = []
    controller.cancelled.connect(lambda: cancelled.append(True))
    controller.finished.connect(lambda: finished.append(True))

    try:
        # A deliberately slow run so cancellation (not natural finish) wins.
        controller.start(
            _make_request(batch_size=2), steps=100_000, step_delay_ms=5
        )
        assert controller.has_live_thread is True

        controller.cancel()
        assert _pump_until(qapp, lambda: controller.status is RunStatus.CANCELLED)

        assert cancelled == [True]
        assert finished == []
        assert controller.is_running is False
        assert not controller.has_live_thread
        # Thread reference was cleared after the thread finished.
        assert controller._thread is None
    finally:
        controller.shutdown()


def test_shutdown_cleans_live_run_idempotently(qapp):
    controller = RunController()
    try:
        controller.start(_make_request(), steps=100_000, step_delay_ms=5)
        assert controller.is_running is True

        controller.shutdown()
        assert controller.is_running is False
        assert controller.status is RunStatus.IDLE
        assert not controller.has_live_thread

        # Idempotent: a second call must not raise or change state.
        controller.shutdown()
        assert controller.status is RunStatus.IDLE
    finally:
        controller.shutdown()


def test_controller_rejects_non_runrequest(qapp):
    controller = RunController()
    with pytest.raises(TypeError):
        controller.start({"not": "a RunRequest"})
    assert controller.is_running is False


def test_controller_rejects_double_start(qapp):
    controller = RunController()
    try:
        controller.start(_make_request(), steps=100_000, step_delay_ms=5)
        with pytest.raises(RuntimeError):
            controller.start(_make_request())
    finally:
        controller.shutdown()


# --------------------------------------------------------------------------
# MainWindow wiring
# --------------------------------------------------------------------------
def test_main_window_start_button_starts_lifecycle(qapp):
    win = MainWindow()
    try:
        win.batch_spin.setValue(4)
        win.input_edit.setText("/inputs")

        win.start_button.click()
        assert win.is_running is True
        assert not win.start_button.isEnabled()
        assert win.stop_button.isEnabled()

        assert _pump_until(qapp, lambda: win.is_running is False)

        assert win.is_running is False
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()
        assert win.progress.value() == 100
        assert "Simulated run" in win.log_view.toPlainText()
        assert win.controller.status is RunStatus.FINISHED
    finally:
        win.shutdown()


def test_main_window_stop_button_requests_cancellation(qapp):
    win = MainWindow()
    cancelled = []
    finished = []
    win.controller.cancelled.connect(lambda: cancelled.append(True))
    win.controller.finished.connect(lambda: finished.append(True))
    try:
        # Swap in a slow worker so the Stop button beats natural completion.
        win.start_button.click()
        assert win.is_running is True

        win.stop_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)

        assert cancelled == [True]
        assert finished == []
        assert win.controller.status is RunStatus.CANCELLED
        assert not win.controller.has_live_thread
    finally:
        win.shutdown()


def test_main_window_close_cleans_up_live_run_idempotently(qapp):
    win = MainWindow()
    try:
        win.start_button.click()
        assert win.is_running is True

        win.close()
        assert win.is_running is False
        assert win.shutdown_called is True
        assert not win.controller.has_live_thread

        # Idempotent shutdown after close.
        win.shutdown()
        assert win.shutdown_called is True
    finally:
        win.shutdown()


def test_main_window_run_uses_canonical_run_request(qapp):
    """Start builds a canonical RunRequest and hands it to the controller."""
    win = MainWindow()
    seen = []

    original_start = win.controller.start

    def spy_start(request, **kwargs):
        assert isinstance(request, RunRequest)
        from seestar.gui.run_config import RunRequest as Canonical

        assert request is not None and isinstance(request, Canonical)
        seen.append(request)
        original_start(request, **kwargs)

    win.controller.start = spy_start
    try:
        win.batch_spin.setValue(4)
        win.start_button.click()
        assert len(seen) == 1
        assert seen[0].backend_kwargs["batch_size"] == 4
        assert _pump_until(qapp, lambda: win.is_running is False)
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Fresh-process import hygiene
# --------------------------------------------------------------------------
def test_import_hygiene_fresh_process():
    """``import seestar.gui_qt`` must not pull Tk/engine into a fresh process."""
    code = (
        "import sys\n"
        "import seestar.gui_qt  # noqa: F401\n"
        "_bad = [m for m in sys.modules\n"
        "        if m.startswith('tkinter')\n"
        "        or m.startswith('seestar.core')\n"
        "        or m.startswith('seestar.alignment')\n"
        "        or m.startswith('seestar.enhancement')\n"
        "        or m.startswith('seestar.queuep')\n"
        "        or m in ('seestar.gui.main_window', 'seestar.gui.settings',"
        " 'seestar.gui.boring_stack')]\n"
        "if _bad:\n"
        "    print('BAD_MODULES:', _bad)\n"
        "    sys.exit(1)\n"
        "from seestar.gui_qt.run_bridge import RunRequest as Q\n"
        "from seestar.gui.run_config import RunRequest as C\n"
        "assert Q is C\n"
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
