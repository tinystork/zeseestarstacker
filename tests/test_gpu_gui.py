"""M5 tests: capability-aware GPU status and enablement in the Qt shell.

Offscreen tests (``QT_QPA_PLATFORM=offscreen``) covering:

a. ``gpu_bridge.probe_gpu()`` returns a capability object with a ``.state``
   attribute (or ``None``) without raising; on this GPU host the state is
   ``"ready"``.
b. ``describe_capability`` / ``describe_policy`` return non-empty lines for
   ready / not-ready / unavailable capability objects.
c. With a "ready" capability (probe monkeypatched) the Use GPU checkbox is
   ENABLED and the status label carries the device name.
d. With a "not ready" capability the checkbox is DISABLED and the label does
   not claim "ready".
e. The toggle re-renders the resolved-state line (enabled+checked ->
   "GPU acceleration enabled — CuPy / <device>"; enabled+unchecked ->
   "<device> — GPU acceleration disabled").
f. The probe runs OFF the Qt main thread (F3): the window defers the cold
   probe to a ``GpuProbeWorker`` QThread (never synchronously during
   construction), and the worker executes its probe on a different thread.
g. The bridge source stays import-hygiene-clean (no engine dotted token).

No real stacking, no engine at import time (only through the bridge's lazy
split-string import), no FITS/PNG writes, no subprocess.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt import gpu_bridge

# Tests may reference the engine module freely (the hygiene gate scans only
# ``seestar/gui_qt/*.py`` sources, never tests).
_GPU_MOD = importlib.import_module(".".join(("seestar", "core", "gpu")))

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


@pytest.fixture()
def window(qapp):
    win = MainWindow()
    yield win
    win.shutdown()


def _fake_caps(**overrides):
    """Ready-by-default fake ``GpuCapabilities``."""
    base = dict(
        gpu_detected=True,
        cuda_runtime_ready=True,
        cupy_ready=True,
        opencv_cuda_ready=False,
        backend_ready=True,
        device_name="Fake NVIDIA RTX 9999",
        device_vram_mb=8192,
        compute_capability="9.9",
        failure_reason=None,
        state="ready",
    )
    base.update(overrides)
    return _GPU_MOD.GpuCapabilities(**base)


# ---------------------------------------------------------------------------
# a. bridge probe
# ---------------------------------------------------------------------------


def test_bridge_probe_returns_caps_or_none_without_raising():
    caps = gpu_bridge.probe_gpu()
    if caps is None:
        pytest.skip("engine probe unavailable on this host")
    assert hasattr(caps, "state")
    assert caps.state in {
        "no_gpu",
        "gpu_no_runtime",
        "cuda_no_backend",
        "backend_error",
        "ready",
    }
    # On this host (MX150 + cupy-cuda12x) the probe reports a ready backend.
    if getattr(caps, "backend_ready", False):
        assert caps.state == "ready"


# ---------------------------------------------------------------------------
# b. describe helpers
# ---------------------------------------------------------------------------


def test_describe_helpers_non_empty():
    ready = _fake_caps()
    none_line = gpu_bridge.describe_capability(None)
    assert none_line == "GPU status unavailable"
    assert gpu_bridge.describe_capability(ready)
    assert gpu_bridge.describe_policy(ready, request_gpu=False)
    assert gpu_bridge.describe_policy(ready, request_gpu=True)
    assert gpu_bridge.describe_policy(None, request_gpu=True) == (
        "GPU status unavailable"
    )


# ---------------------------------------------------------------------------
# c/d. capability-driven enablement + label
# ---------------------------------------------------------------------------


def test_ready_capability_enables_checkbox_and_shows_device(monkeypatch, window):
    caps = _fake_caps()
    monkeypatch.setattr(gpu_bridge, "probe_gpu", lambda: caps)
    window._refresh_gpu_status()

    assert window.use_gpu_check.isEnabled() is True
    assert caps.device_name in window.gpu_status_label.text()


def test_not_ready_capability_disables_checkbox_and_no_ready_claim(
    monkeypatch, window
):
    caps = _fake_caps(
        gpu_detected=False,
        cuda_runtime_ready=False,
        cupy_ready=False,
        backend_ready=False,
        device_name=None,
        device_vram_mb=None,
        compute_capability=None,
        state="no_gpu",
    )
    monkeypatch.setattr(gpu_bridge, "probe_gpu", lambda: caps)
    window._refresh_gpu_status()

    assert window.use_gpu_check.isEnabled() is False
    text = window.gpu_status_label.text()
    assert text == "No compatible GPU detected"
    assert "ready" not in text.lower()


def test_backend_error_capability_disables_checkbox(monkeypatch, window):
    caps = _fake_caps(
        cupy_ready=False,
        backend_ready=False,
        state="backend_error",
        failure_reason="cupy: real-kernel init failed (RuntimeError: CUDA headers missing)",
    )
    monkeypatch.setattr(gpu_bridge, "probe_gpu", lambda: caps)
    window._refresh_gpu_status()

    assert window.use_gpu_check.isEnabled() is False
    text = window.gpu_status_label.text()
    assert "backend unavailable" in text
    assert "ready" not in text.lower()


# ---------------------------------------------------------------------------
# e. toggle re-renders the resolved state
# ---------------------------------------------------------------------------


def test_toggle_rerenders_resolved_state(monkeypatch, window):
    caps = _fake_caps()
    monkeypatch.setattr(gpu_bridge, "probe_gpu", lambda: caps)
    window._refresh_gpu_status()

    # Enabled + unchecked -> "<device> — GPU acceleration disabled".
    window.use_gpu_check.setChecked(False)
    label_off = window.gpu_status_label.text()
    assert caps.device_name in label_off
    assert "GPU acceleration disabled" in label_off

    # Checked -> the truthful resolved policy line.
    window.use_gpu_check.setChecked(True)
    label_on = window.gpu_status_label.text()
    assert "GPU acceleration enabled" in label_on
    assert caps.device_name in label_on

    # Unchecking again restores the disabled line.
    window.use_gpu_check.setChecked(False)
    assert "GPU acceleration disabled" in window.gpu_status_label.text()


# ---------------------------------------------------------------------------
# f. import hygiene of the new bridge
# ---------------------------------------------------------------------------


def test_gpu_bridge_source_is_hygiene_clean():
    text = (ROOT / "seestar" / "gui_qt" / "gpu_bridge.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "seestar.core",
        "seestar.queuep",
        "tkinter",
        "zealfie",
        "zesolver",
    ):
        assert token not in text, f"gpu_bridge.py references {token}"


# ---------------------------------------------------------------------------
# f. off-main-thread probing (F3)
# ---------------------------------------------------------------------------


def test_probe_worker_executes_on_background_thread(qapp):
    """The probe worker runs its probe off the Qt main thread and delivers
    the result through the signal."""
    import threading

    from PySide6.QtCore import QEventLoop, QObject, QTimer, Qt
    from PySide6.QtCore import Signal as QtSignal

    main_thread = threading.current_thread()
    observed = {}

    def probe_fn():
        observed["thread"] = threading.current_thread()
        observed["caps"] = _fake_caps()
        return observed["caps"]

    worker = gpu_bridge.GpuProbeWorker(probe_fn=probe_fn)

    class _Sink(QObject):
        got = QtSignal(object)

    sink = _Sink()
    received = []
    loop = QEventLoop()
    worker.resultReady.connect(sink.got, Qt.QueuedConnection)
    sink.got.connect(lambda caps: (received.append(caps), loop.quit()))
    QTimer.singleShot(8000, loop.quit)  # safety timeout

    worker.start()
    loop.exec()
    assert worker.wait(2000)

    assert observed["thread"] is not main_thread
    assert len(received) == 1
    assert received[0] is observed["caps"]


def test_window_probe_is_deferred_to_worker_not_constructor(qapp, monkeypatch):
    """Window construction never probes synchronously: the probe starts only
    after the event loop turns (via the worker), and the result lands on the
    status label / checkbox enablement."""
    import threading

    from PySide6.QtCore import QEventLoop, QTimer

    calls = []
    caps = _fake_caps()

    def fake_probe():
        calls.append(threading.current_thread())
        return caps

    monkeypatch.setattr(gpu_bridge, "probe_gpu", fake_probe)
    win = MainWindow()
    try:
        # Construction must not probe (the cold probe would block the GUI
        # thread); the label still shows the probing placeholder.
        assert calls == []
        assert win._gpu_probe_worker is None

        # Let the deferred singleShot fire: the worker probes off-thread and
        # the queued result re-enables the checkbox and fills the label.
        loop = QEventLoop()
        timer = QTimer()

        def check():
            if caps.device_name in win.gpu_status_label.text():
                loop.quit()

        timer.timeout.connect(check)
        timer.start(20)
        QTimer.singleShot(8000, loop.quit)
        loop.exec()
        timer.stop()

        assert calls, "the deferred probe never ran"
        assert calls[0] is not threading.current_thread()
        # Unchecked default: truthful "disabled" line, checkbox now enabled.
        assert caps.device_name in win.gpu_status_label.text()
        assert "GPU acceleration disabled" in win.gpu_status_label.text()
        assert win.use_gpu_check.isEnabled() is True
        # Checking the (now enabled) box renders the truthful enabled line.
        win.use_gpu_check.setChecked(True)
        assert "GPU acceleration enabled" in win.gpu_status_label.text()
    finally:
        win.shutdown()
