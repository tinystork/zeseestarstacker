"""M5 seam tests: explicit Qt backend activation path.

These tests exercise the new *backend selection/injection* path wired into the
PySide6 shell:

* :class:`MainWindow` accepts ``backend_factory`` and ``backend_mode`` without
  disturbing the default simulated behaviour,
* the Start button passes the canonical :class:`RunRequest` to an injected fake
  backend and updates progress/log/finish through queued GUI-thread slots,
* the Stop button cancels an injected backend and calls ``cancel()``,
* ``backend_mode="seestar"`` resolves a lazy
  :class:`SeestarQueuedStackerBackend` without importing the engine,
* ``seestar.qt_main`` exposes ``--backend simulated|seestar`` (default
  ``simulated``) with no engine import in a fresh interpreter.

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

import seestar
from seestar.gui_qt import MainWindow, RunStatus, create_application
from seestar.gui_qt.backend_runner import (
    BackendRunResult,
    BaseRunBackend,
    SeestarQueuedStackerBackend,
)
from seestar.gui.run_config import RunRequest as CanonicalRunRequest

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


class FakeBackend(BaseRunBackend):
    """A fake backend recording the request and driving callbacks."""

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

    def cancel(self) -> None:
        self.cancel_called = True


# --------------------------------------------------------------------------
# Source-token / import-hygiene invariants for the new seams
# --------------------------------------------------------------------------
def test_activation_source_is_tk_engine_free():
    files = [
        ROOT / "seestar" / "qt_main.py",
        ROOT / "seestar" / "gui_qt" / "app.py",
        ROOT / "seestar" / "gui_qt" / "main_window.py",
    ]
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
    for path in files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} references {token}"


def test_activation_fresh_process_import_hygiene_and_seestar_mode():
    """Fresh interpreter: entrypoint import, CLI parse, seestar mode resolution.

    ``import seestar.qt_main`` and ``import seestar.gui_qt`` must leave no
    Tk/engine modules in ``sys.modules``; the canonical ``RunRequest`` must be
    identical; ``--backend seestar`` must parse to the seestar mode; and
    constructing a seestar-mode window + resolving its backend must still not
    import the engine (no Start clicked).
    """
    code = (
        "import os\n"
        "os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
        "import sys\n"
        "import seestar.qt_main  # noqa: F401\n"
        "import seestar.gui_qt  # noqa: F401\n"
        "from seestar.gui_qt.run_bridge import RunRequest as Q\n"
        "from seestar.gui.run_config import RunRequest as C\n"
        "assert Q is C, 'RunRequest is not canonical'\n"
        "mode, remaining = seestar.qt_main.parse_qt_args(['--backend', 'seestar'])\n"
        "assert mode == 'seestar', mode\n"
        "assert remaining == [], remaining\n"
        "from seestar.gui_qt import MainWindow, create_application\n"
        "from seestar.gui_qt.backend_runner import SeestarQueuedStackerBackend\n"
        "app = create_application([])\n"
        "win = MainWindow(backend_mode='seestar')\n"
        "backend = win.resolve_backend()\n"
        "assert isinstance(backend, SeestarQueuedStackerBackend)\n"
        "win.shutdown()\n"
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
        f"activation import hygiene violated: stdout={proc.stdout!r} "
        f"stderr={proc.stderr!r}"
    )


# --------------------------------------------------------------------------
# Default MainWindow stays simulated
# --------------------------------------------------------------------------
def test_default_main_window_starts_simulated_backend(qapp):
    win = MainWindow()
    seen_kwargs = []
    original_start = win.controller.start

    def spy_start(request, **kwargs):
        seen_kwargs.append(kwargs)
        original_start(request, **kwargs)

    win.controller.start = spy_start
    try:
        win.start_button.click()
        assert len(seen_kwargs) == 1
        # Default path passes no explicit backend -> simulated default.
        assert "backend" not in seen_kwargs[0]
        assert _pump_until(qapp, lambda: win.is_running is False)
        assert win.controller.status is RunStatus.FINISHED
        assert win.progress.value() == 100
        assert "Simulated run" in win.log_view.toPlainText()
    finally:
        win.shutdown()


def test_default_backend_mode_resolves_to_none(qapp):
    win = MainWindow()
    try:
        assert win.backend_mode == "simulated"
        assert win.backend_factory is None
        assert win.resolve_backend() is None
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# backend_factory injection
# --------------------------------------------------------------------------
def test_backend_factory_injection_drives_canonical_run_request(qapp):
    fakes = []

    def factory():
        fake = FakeBackend(complete_immediately=True)
        fakes.append(fake)
        return fake

    win = MainWindow(backend_factory=factory)
    try:
        win.batch_spin.setValue(4)
        win.input_edit.setText("/inputs")
        win.start_button.click()

        assert _pump_until(qapp, lambda: win.is_running is False)
        assert len(fakes) == 1
        fake = fakes[0]
        assert len(fake.run_calls) == 1
        assert isinstance(fake.run_calls[0], CanonicalRunRequest)
        assert fake.run_calls[0].backend_kwargs["batch_size"] == 4
        assert fake.run_calls[0].backend_kwargs["input_dir"] == "/inputs"

        # UI updated via queued GUI-thread slots.
        assert win.progress.value() == 100
        assert "fake backend started" in win.log_view.toPlainText()
        assert "fake backend finished" in win.log_view.toPlainText()
        assert win.controller.status is RunStatus.FINISHED
    finally:
        win.shutdown()


def test_stop_button_cancels_injected_backend(qapp):
    fakes = []

    def factory():
        fake = FakeBackend(complete_immediately=False)
        fakes.append(fake)
        return fake

    win = MainWindow(backend_factory=factory)
    cancelled = []
    win.controller.cancelled.connect(lambda: cancelled.append(True))
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: len(fakes) == 1)
        win.stop_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)

        assert fakes[0].cancel_called is True
        assert cancelled == [True]
        assert win.controller.status is RunStatus.CANCELLED
        assert not win.controller.has_live_thread
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# backend_mode="seestar" (lazy real backend selection, no Start in tests)
# --------------------------------------------------------------------------
def test_backend_mode_seestar_resolves_lazy_backend(qapp):
    win = MainWindow(backend_mode="seestar")
    try:
        assert win.backend_mode == "seestar"
        backend = win.resolve_backend()
        assert isinstance(backend, SeestarQueuedStackerBackend)
    finally:
        win.shutdown()


def test_backend_mode_validation(qapp):
    with pytest.raises(ValueError):
        MainWindow(backend_mode="bogus")
    with pytest.raises(TypeError):
        MainWindow(backend_factory="not-callable")


# --------------------------------------------------------------------------
# M6: real-backend start preflight (settings validation seam)
# --------------------------------------------------------------------------
def test_seestar_mode_empty_folders_blocks_start(qapp):
    """Empty folders under seestar mode must not start the controller."""
    win = MainWindow(backend_mode="seestar")
    calls = []
    original_start = win.controller.start

    def spy_start(request, **kwargs):
        calls.append((request, kwargs))
        original_start(request, **kwargs)

    win.controller.start = spy_start
    try:
        win.start_button.click()

        # The preflight must short-circuit: controller.start never called.
        assert calls == []
        assert win.is_running is False
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()
        assert win.controller.status is RunStatus.IDLE
        assert not win.controller.has_live_thread

        text = win.log_view.toPlainText()
        assert "Cannot start real backend" in text
        assert "Input folder is empty." in text
        assert "Output folder is empty." in text
        assert "Cannot start real backend" in win.statusBar().currentMessage()
    finally:
        win.shutdown()


def test_seestar_mode_valid_settings_reaches_controller_with_lazy_backend(qapp):
    """Valid input/output reaches controller.start(backend=SeestarQueuedStackerBackend).

    The spy intercepts ``controller.start`` so the lazy real backend (and the
    engine) is never actually executed — only its construction is verified.
    """
    win = MainWindow(backend_mode="seestar")
    seen = []
    original_start = win.controller.start

    def spy_start(request, **kwargs):
        seen.append((request, kwargs))
        # Deliberately do NOT call original_start: that would run the engine.

    win.controller.start = spy_start
    try:
        win.input_edit.setText("/inputs")
        win.output_edit.setText("/outputs")
        win.start_button.click()

        assert len(seen) == 1
        request, kwargs = seen[0]
        assert isinstance(request, CanonicalRunRequest)
        backend = kwargs.get("backend")
        assert isinstance(backend, SeestarQueuedStackerBackend)
        assert request.backend_kwargs["input_dir"] == "/inputs"
        assert request.backend_kwargs["output_dir"] == "/outputs"
        assert "Cannot start real backend" not in win.log_view.toPlainText()
        # We never actually started (spy suppressed the real start).
        assert win.is_running is False
    finally:
        win.shutdown()


def test_backend_factory_not_blocked_by_preflight_with_empty_folders(qapp):
    """backend_factory stays usable with empty folders (not preflight-blocked)."""
    fakes = []

    def factory():
        fake = FakeBackend(complete_immediately=True)
        fakes.append(fake)
        return fake

    win = MainWindow(backend_factory=factory)
    try:
        win.start_button.click()  # empty folders on the factory/injection path
        assert _pump_until(qapp, lambda: win.is_running is False)
        assert len(fakes) == 1
        assert len(fakes[0].run_calls) == 1
        assert "Cannot start real backend" not in win.log_view.toPlainText()
        assert win.controller.status is RunStatus.FINISHED
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# M12: real-backend preflight hardening (batch size 1 + reproject solver gate)
# --------------------------------------------------------------------------
def test_seestar_mode_batch_size_one_routes_to_boring_not_controller(qapp):
    """batch_size == 1 now routes to the boring CSV path, not RunController.start.

    With no ``stack_plan.csv`` present (and a non-existent input folder), the
    boring preflight blocks the start — ``controller.start`` is never reached —
    proving the Qt shell no longer silently launches the normal queue-manager
    backend for ``batch_size == 1``.
    """
    win = MainWindow(backend_mode="seestar")
    seen = []
    original_start = win.controller.start

    def spy_start(request, **kwargs):
        seen.append((request, kwargs))
        # Deliberately do NOT call original_start: that would run the engine.

    win.controller.start = spy_start
    try:
        win.input_edit.setText("/inputs")
        win.output_edit.setText("/outputs")
        win.batch_spin.setValue(1)
        win.start_button.click()

        assert seen == []  # RunController.start must NOT be called
        assert win.is_running is False
        assert "Cannot start boring stack" in win.log_view.toPlainText()
    finally:
        win.shutdown()

def test_seestar_mode_reproject_without_solver_blocks_start(qapp):
    """Reproject enabled + solver 'none' must not start the real backend."""
    win = MainWindow(backend_mode="seestar")
    calls = []
    original_start = win.controller.start

    def spy_start(request, **kwargs):
        calls.append((request, kwargs))
        original_start(request, **kwargs)

    win.controller.start = spy_start
    try:
        win.input_edit.setText("/inputs")
        win.output_edit.setText("/outputs")
        win.final_combine_combo.setCurrentText("Reproject")
        win.solver_combo.setCurrentText("none")
        win.start_button.click()

        assert calls == []
        assert win.is_running is False
        assert win.controller.status is RunStatus.IDLE
        assert not win.controller.has_live_thread

        text = win.log_view.toPlainText()
        assert "Cannot start real backend" in text
        assert "requires a local astrometric solver" in text
    finally:
        win.shutdown()


def test_seestar_mode_reproject_with_astap_path_reaches_controller(qapp):
    """Reproject enabled + ASTAP path configured passes preflight.

    The spy intercepts ``controller.start`` so the real engine is never run;
    reaching the controller proves the solver gate accepted the settings.
    """
    win = MainWindow(backend_mode="seestar")
    seen = []
    original_start = win.controller.start

    def spy_start(request, **kwargs):
        seen.append((request, kwargs))
        # Deliberately do NOT call original_start: that would run the engine.

    win.controller.start = spy_start
    try:
        win.input_edit.setText("/inputs")
        win.output_edit.setText("/outputs")
        win.final_combine_combo.setCurrentText("Reproject")
        win.solver_combo.setCurrentText("astap")
        win._settings_widgets["astap_path"].setText("/usr/bin/astap")
        win.start_button.click()

        assert len(seen) == 1
        assert "Cannot start real backend" not in win.log_view.toPlainText()
        assert win.is_running is False
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# M13: batch-size contract + ZeSolver-only solver gate at the shell level
# --------------------------------------------------------------------------
def test_batch_size_zero_normalizes_to_auto_sentinel(qapp):
    """UI 0 on a normal stack becomes the -1 Auto sentinel before the request."""
    win = MainWindow()
    seen = []
    original_start = win.controller.start

    def spy_start(request, **kwargs):
        seen.append(request)
        original_start(request, **kwargs)

    win.controller.start = spy_start
    try:
        win.batch_spin.setValue(0)
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)
        assert len(seen) == 1
        assert seen[0].backend_kwargs["batch_size"] == -1
        assert seen[0].align_on_disk is False
    finally:
        win.shutdown()


def test_batch_size_zero_with_reproject_coadd_stays_zero(qapp):
    """UI 0 + 'Reproject and coadd' keeps the special batch-zero mode (0)."""
    win = MainWindow()
    seen = []
    original_start = win.controller.start

    def spy_start(request, **kwargs):
        seen.append(request)
        original_start(request, **kwargs)

    win.controller.start = spy_start
    try:
        win.batch_spin.setValue(0)
        win.final_combine_combo.setCurrentText("Reproject & Coadd")
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)
        assert len(seen) == 1
        assert seen[0].backend_kwargs["batch_size"] == 0
        assert seen[0].align_on_disk is False
    finally:
        win.shutdown()


def test_seestar_mode_zesolver_operational_without_astap_reaches_controller(qapp):
    """ZeSolver operational authorises Reproject with no ASTAP fallback.

    The readiness probe is injected (``solver_probe=lambda: True``) so the gate
    is exercised without importing the engine; reaching ``controller.start``
    proves the gate accepted a ZeSolver-only configuration.
    """
    win = MainWindow(backend_mode="seestar", solver_probe=lambda: True)
    seen = []
    original_start = win.controller.start

    def spy_start(request, **kwargs):
        seen.append((request, kwargs))
        # Suppress the real start (no engine run in this test).

    win.controller.start = spy_start
    try:
        win.input_edit.setText("/inputs")
        win.output_edit.setText("/outputs")
        win.final_combine_combo.setCurrentText("Reproject")
        win.solver_combo.setCurrentText("zesolver")
        # astap_path left empty on purpose.
        win.start_button.click()

        assert len(seen) == 1
        assert "Cannot start real backend" not in win.log_view.toPlainText()
        assert seen[0][0].backend_kwargs["local_solver_preference"] == "zesolver"
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# CLI parsing (seestar.qt_main)
# --------------------------------------------------------------------------
def test_cli_parser_default_seestar():
    import seestar.qt_main as qt_main

    assert qt_main.parse_qt_args([]) == ("seestar", [])
    assert qt_main.parse_qt_args(["--backend", "seestar"]) == ("seestar", [])
    assert qt_main.parse_qt_args(["--backend=seestar"]) == ("seestar", [])


def test_cli_parser_explicit_simulated_accepted():
    import seestar.qt_main as qt_main

    assert qt_main.parse_qt_args(["--backend", "simulated"]) == ("simulated", [])
    assert qt_main.parse_qt_args(["--backend=simulated"]) == ("simulated", [])
    assert qt_main.parse_qt_args(["--backend", "simulated", "--verbose"]) == (
        "simulated",
        ["--verbose"],
    )


def test_cli_parser_explicit_seestar_and_remaining_args():
    import seestar.qt_main as qt_main

    assert qt_main.parse_qt_args(["--backend", "seestar"]) == ("seestar", [])
    assert qt_main.parse_qt_args(["--backend=seestar"]) == ("seestar", [])
    assert qt_main.parse_qt_args(["--backend", "seestar", "--verbose"]) == (
        "seestar",
        ["--verbose"],
    )
    assert qt_main.parse_qt_args(["--foo", "--backend=seestar", "--bar"]) == (
        "seestar",
        ["--foo", "--bar"],
    )


def test_cli_parser_rejects_invalid():
    import seestar.qt_main as qt_main

    with pytest.raises(SystemExit):
        qt_main.parse_qt_args(["--backend", "bogus"])
    with pytest.raises(SystemExit):
        qt_main.parse_qt_args(["--backend"])


def test_startup_witness_resolves_real_backend_offscreen():
    """``ZSSS_QT_STARTUP_WITNESS=1`` smoke test (no event loop, returns 0)."""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["ZSSS_QT_STARTUP_WITNESS"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "seestar.qt_main"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert proc.returncode == 0, (
        f"startup witness failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    out = proc.stdout
    assert "ZSSS_QT_STARTUP_WITNESS_TITLE=" in out
    # Exact source-derived product version (subprocess reads the same
    # seestar/__init__.py via cwd=ROOT, so this matches the in-process value).
    expected_version = f"{seestar.__version__} {seestar.__codename__}"
    assert f"ZSSS_QT_STARTUP_WITNESS_VERSION={expected_version!r}" in out
    assert "ZSSS_QT_STARTUP_WITNESS_BACKEND_MODE='seestar'" in out
    assert "ZSSS_QT_STARTUP_WITNESS_BACKEND_CLASS=SeestarQueuedStackerBackend" in out
    assert "ZSSS_QT_STARTUP_WITNESS_ICON=present" in out
    assert "ZSSS_QT_STARTUP_WITNESS_PREVIEW=present" in out
