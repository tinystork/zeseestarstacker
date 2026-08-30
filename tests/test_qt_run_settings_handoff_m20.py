"""M20 seam tests: Qt run-settings handoff (``use_gpu`` / ``max_hq_mem_gb``).

Backend E2E part 1: the Qt run flow must consume the Qt-collected settings —
at minimum ``use_gpu`` and ``max_hq_mem_gb`` (checklist 15.15 / 15.30 backend
halves).  These tests pin, without touching the engine:

* the Qt handoff attaches the collected seam fields to the canonical
  ``RunRequest`` (inspectable in ``backend_kwargs``),
* the canonical shared builder (``build_backend_kwargs`` / ``build_run_request``)
  stays byte-identical for Tk-style settings (golden case),
* a bare surface (no persisted settings / untouched controls) degrades to the
  Qt/Tk defaults,
* the Qt backend adapter applies the seam fields to the stacker *instance*
  (never forwards them to ``start_processing``),
* the new handoff module preserves the gui_qt import-hygiene invariant.

No real stacking, no engine, no Tk, no FITS/PNG writes.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt.backend_runner import (
    BackendRunResult,
    BaseRunBackend,
    SeestarQueuedStackerBackend,
)
from seestar.gui_qt.run_bridge import (
    RunRequest,
    build_backend_kwargs,
    build_run_request,
    split_backend_kwargs,
)
from seestar.gui_qt.run_handoff import QT_SEAM_FIELDS, attach_run_settings
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


@pytest.fixture()
def window(qapp):
    win = MainWindow()
    yield win
    win.shutdown()


# --------------------------------------------------------------------------
# (1) mapping assertion: collected QtSettingsState -> RunRequest.backend_kwargs
# --------------------------------------------------------------------------
def test_attach_run_settings_injects_seam_fields():
    state = QtSettingsState(use_gpu=True, max_hq_mem_gb=16.0)
    request = build_run_request(state)
    # The canonical builder must NOT already carry the fields (Tk parity)...
    assert "use_gpu" not in request.backend_kwargs
    assert "max_hq_mem_gb" not in request.backend_kwargs

    attached = attach_run_settings(
        request,
        use_gpu=state.use_gpu,
        max_hq_mem_gb=state.max_hq_mem_gb,
        reference_origin_hint="ZEANALYSER_V1",
    )
    assert attached.backend_kwargs["use_gpu"] is True
    assert attached.backend_kwargs["max_hq_mem_gb"] == 16.0
    assert attached.backend_kwargs["reference_origin_hint"] == "ZEANALYSER_V1"
    # The seam fields are the Qt-collected values, verbatim.
    assert attached.backend_kwargs["use_gpu"] is state.use_gpu


def test_attach_run_settings_does_not_mutate_original_request():
    state = QtSettingsState(use_gpu=True, max_hq_mem_gb=16.0)
    request = build_run_request(state)
    before = dict(request.backend_kwargs)
    attach_run_settings(request, use_gpu=True, max_hq_mem_gb=16.0)
    # The original snapshot is untouched (still immutable, still no seam fields).
    assert dict(request.backend_kwargs) == before
    assert "use_gpu" not in request.backend_kwargs


def test_main_window_build_run_request_carries_seam_settings(window):
    window.use_gpu_check.setChecked(True)
    window.max_hq_mem_spin.setValue(32)

    request = window.build_run_request()
    assert isinstance(request, RunRequest)
    assert request.backend_kwargs["use_gpu"] is True
    assert request.backend_kwargs["max_hq_mem_gb"] == 32.0
    # The bytes conversion is NOT done in the snapshot (backend adapter's job).
    assert "max_hq_mem" not in request.backend_kwargs


def test_main_window_build_run_request_carries_reference_origin_hint(window):
    window._reference_origin_hint = "ZEANALYSER_V1"
    request = window.build_run_request()
    assert request.backend_kwargs["reference_origin_hint"] == "ZEANALYSER_V1"


# --------------------------------------------------------------------------
# (2) Tk parity: canonical builder output unchanged (golden case)
# --------------------------------------------------------------------------
def test_canonical_builder_golden_unchanged_with_tk_settings_manager():
    """``build_backend_kwargs`` ignores ``use_gpu``/``max_hq_mem_gb``.

    Loads the *real* Tk ``SettingsManager`` standalone (file path, no Tk root)
    and pins that the canonical builder produces exactly the historical surface:
    the two Qt seam fields are absent, and ``split_backend_kwargs`` still only
    partitions ``stack_final_combine``.
    """
    settings_spec = importlib.util.spec_from_file_location(
        "m20_settings_manager", ROOT / "seestar" / "gui" / "settings.py"
    )
    settings_mod = importlib.util.module_from_spec(settings_spec)
    sys.modules["m20_settings_manager"] = settings_mod
    settings_spec.loader.exec_module(settings_mod)
    SettingsManager = settings_mod.SettingsManager

    sm = SettingsManager(settings_file="unused.json")
    sm.validate_settings()

    kwargs = build_backend_kwargs(sm)
    assert "use_gpu" not in kwargs
    assert "max_hq_mem_gb" not in kwargs
    assert "max_hq_mem" not in kwargs
    # The historical golden: stack_final_combine is the only seam-only field.
    start_kwargs, seam_kwargs = split_backend_kwargs(kwargs)
    assert seam_kwargs == {"stack_final_combine": kwargs["stack_final_combine"]}
    assert "use_gpu" not in start_kwargs
    assert "max_hq_mem_gb" not in start_kwargs


def test_canonical_builder_ignores_qt_seam_attributes():
    """Even a settings object that HAS the attrs is left untouched (Tk parity)."""
    state = QtSettingsState(use_gpu=True, max_hq_mem_gb=16.0)
    kwargs = build_backend_kwargs(state)
    assert "use_gpu" not in kwargs
    assert "max_hq_mem_gb" not in kwargs


# --------------------------------------------------------------------------
# (3) fallback: no Qt settings -> defaults unchanged
# --------------------------------------------------------------------------
def test_fallback_defaults_unchanged():
    request = build_run_request(QtSettingsState())
    attached = attach_run_settings(request)
    # A bare surface degrades to the Qt/Tk defaults.
    assert attached.backend_kwargs["use_gpu"] is False
    assert attached.backend_kwargs["max_hq_mem_gb"] == 8.0


def test_main_window_defaults_carry_default_seam_settings(window):
    request = window.build_run_request()
    assert request.backend_kwargs["use_gpu"] is False
    assert request.backend_kwargs["max_hq_mem_gb"] == 8.0


# --------------------------------------------------------------------------
# (4) import hygiene preserved for the new handoff module
# --------------------------------------------------------------------------
def test_run_handoff_source_is_engine_and_tk_free():
    text = (ROOT / "seestar" / "gui_qt" / "run_handoff.py").read_text(
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
        "numpy",
    )
    for token in forbidden:
        assert token not in text, f"run_handoff.py references {token}"


def test_run_bridge_and_handoff_import_hygiene_fresh_process():
    code = (
        "import sys\n"
        "import seestar.gui_qt.run_handoff  # noqa: F401\n"
        "import seestar.gui_qt.run_bridge  # noqa: F401\n"
        "_bad = [m for m in sys.modules\n"
        "        if m.startswith('seestar.queuep')\n"
        "        or m.startswith('seestar.core')\n"
        "        or m.startswith('seestar.alignment')\n"
        "        or m.startswith('seestar.enhancement')\n"
        "        or m.startswith('tkinter')\n"
        "        or m in ('seestar.gui.settings', 'seestar.gui.main_window')]\n"
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


def test_run_request_class_is_still_canonical():
    """``attach_run_settings`` returns the canonical ``RunRequest`` class."""
    from seestar.gui.run_config import RunRequest as CanonicalRunRequest

    request = build_run_request(QtSettingsState())
    attached = attach_run_settings(request)
    assert type(attached) is CanonicalRunRequest
    assert attached is not request


# --------------------------------------------------------------------------
# (5) backend E2E: seam fields applied to the stacker instance, not kwargs
# --------------------------------------------------------------------------
class _FakeStacker:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = dict(kwargs)
        self.align_on_disk = None
        self.progress_cb = None
        self.start_kwargs = None
        self.stop_called = False
        self._running = False

    def set_progress_callback(self, cb) -> None:
        self.progress_cb = cb

    def start_processing(self, **kwargs):
        self.start_kwargs = dict(kwargs)
        self._running = True
        return True

    def is_running(self) -> bool:
        self._running = False
        return False

    def stop(self) -> None:
        self.stop_called = True
        self._running = False


def test_backend_applies_seam_gpu_and_mem_to_stackers_instance():
    instances = []

    def factory(**kwargs):
        stacker = _FakeStacker(**kwargs)
        instances.append(stacker)
        return stacker

    backend = SeestarQueuedStackerBackend(stacker_factory=factory, poll_interval=0.001)
    state = QtSettingsState(use_gpu=True, max_hq_mem_gb=16.0)
    request = attach_run_settings(
        build_run_request(state),
        use_gpu=state.use_gpu,
        max_hq_mem_gb=state.max_hq_mem_gb,
        reference_origin_hint="ZEANALYSER_V1",
    )

    result = backend.run(request, lambda p: None, lambda m: None, lambda: False)

    assert result is BackendRunResult.FINISHED
    stacker = instances[0]
    # Seam fields reached the stacker instance...
    assert stacker.use_gpu is True
    assert stacker.max_hq_mem == 16 * 1024 ** 3
    assert stacker.reference_origin_hint == "ZEANALYSER_V1"
    # ...and were filtered out of the start_processing surface.
    assert "use_gpu" not in stacker.start_kwargs
    assert "max_hq_mem_gb" not in stacker.start_kwargs


def test_backend_seam_defaults_leave_gpu_off_and_default_mem():
    instances = []

    def factory(**kwargs):
        stacker = _FakeStacker(**kwargs)
        instances.append(stacker)
        return stacker

    backend = SeestarQueuedStackerBackend(stacker_factory=factory, poll_interval=0.001)
    request = attach_run_settings(build_run_request(QtSettingsState()))

    result = backend.run(request, lambda p: None, lambda m: None, lambda: False)

    assert result is BackendRunResult.FINISHED
    stacker = instances[0]
    assert stacker.use_gpu is False
    assert stacker.max_hq_mem == 8 * 1024 ** 3


# --------------------------------------------------------------------------
# (6) offscreen smoke: launch the Qt run flow, capture the request on the backend
# --------------------------------------------------------------------------
class _RecordingBackend(BaseRunBackend):
    def __init__(self) -> None:
        self.request = None

    def run(self, request, progress_callback, log_callback, is_cancel_requested,
            preview_callback=None):
        self.request = request
        log_callback("recorded")
        progress_callback(100)
        return BackendRunResult.FINISHED

    def cancel(self) -> None:
        pass


def test_offscreen_smoke_gpu_and_mem_reach_backend(qapp):
    backend = _RecordingBackend()
    win = MainWindow(backend_factory=lambda: backend)
    try:
        win.use_gpu_check.setChecked(True)
        win.max_hq_mem_spin.setValue(24)
        win.start_button.click()
        assert _pump_until(qapp, lambda: backend.request is not None)
        assert _pump_until(qapp, lambda: win.is_running is False)

        request = backend.request
        assert isinstance(request, RunRequest)
        assert request.backend_kwargs["use_gpu"] is True
        assert request.backend_kwargs["max_hq_mem_gb"] == 24.0
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (7) no new localization surface
# --------------------------------------------------------------------------
def test_handoff_adds_no_localization_keys():
    """The handoff layer adds no user-facing strings (nothing to localize)."""
    assert QT_SEAM_FIELDS == (
        "use_gpu",
        "max_hq_mem_gb",
        "reference_origin_hint",
    )
