"""M4 boring-route tests: ``batch_size == 1`` → ``boring_stack.py`` routing.

These tests exercise the Qt shell's boring (single-batch CSV) route seam:

* the pure-stdlib CSV parser (:func:`boring_route.parse_stack_plan_csv`) and
  command builder (:func:`boring_route.build_boring_request`);
* the boring checkbox ↔ batch-size synchronisation (Tk parity);
* missing/empty/invalid ``stack_plan.csv`` blocks start and stays idle;
* a valid ``stack_plan.csv`` routes to the injected boring runner, **not**
  :meth:`RunController.start`;
* button/run-state lifecycle while the boring route is active;
* source/import hygiene for the two new modules.

No real subprocess is ever spawned: every window under test injects a
``FakeBoringRunner`` factory, so the command is asserted without launching
``boring_stack.py``.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt.boring_route import (
    BoringCsvError,
    BoringRunRequest,
    build_boring_request,
    csv_path_for,
    parse_stack_plan_csv,
    resolve_boring_script_path,
)
from seestar.gui_qt.boring_runner import BoringRunnerBase

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


class FakeBoringRunner(BoringRunnerBase):
    """Records the request and drives terminal signals without a real process."""

    def __init__(self) -> None:
        super().__init__()
        self.start_calls = []
        self.cancel_called = False
        self._active = False

    def start(self, request: BoringRunRequest) -> None:
        self.start_calls.append(request)
        self._active = True
        self.started.emit()

    def cancel(self) -> None:
        self.cancel_called = True
        self._active = False

    def is_running(self) -> bool:
        return self._active

    # --- test drivers ---
    def signal_finished(self, code: int = 0) -> None:
        self._active = False
        self.finished.emit(code)

    def signal_failed(self, message: str) -> None:
        self._active = False
        self.failed.emit(message)

    def signal_cancelled(self) -> None:
        self._active = False
        self.cancelled.emit()


def _write_csv(path: Path, rows) -> Path:
    """Write ``stack_plan.csv`` with the given rows (lists of cells)."""
    csv_path = path / "stack_plan.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)
    return csv_path


def _make_inputs(tmp_path: Path, csv_rows=None, fits_names=("a.fits",)):
    """Create an input folder with the given FITS files and optional CSV."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(exist_ok=True)
    for name in fits_names:
        (input_dir / name).write_bytes(b"")
    if csv_rows is not None:
        _write_csv(input_dir, csv_rows)
    return input_dir


def _window_with_fake(tmp_path: Path, csv_rows=None, fits_names=("a.fits",)):
    """Build a window + fake-runner factory with a prepared input folder."""
    input_dir = _make_inputs(tmp_path, csv_rows, fits_names)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)

    fakes = []

    def factory():
        fake = FakeBoringRunner()
        fakes.append(fake)
        return fake

    win = MainWindow(boring_runner_factory=factory)
    win.input_edit.setText(str(input_dir))
    win.output_edit.setText(str(output_dir))
    win.batch_spin.setValue(1)
    return win, fakes, input_dir, output_dir


# --------------------------------------------------------------------------
# Pure CSV parser
# --------------------------------------------------------------------------
def test_parse_missing_csv_raises(tmp_path):
    with pytest.raises(BoringCsvError) as exc:
        parse_stack_plan_csv(str(tmp_path / "nope" / "stack_plan.csv"))
    assert "stack_plan.csv not found" in str(exc.value)


def test_parse_empty_csv_raises(tmp_path):
    csv_path = _write_csv(tmp_path, [])
    with pytest.raises(BoringCsvError) as exc:
        parse_stack_plan_csv(str(csv_path))
    assert "empty" in str(exc.value).lower()


def test_parse_header_file_path_column(tmp_path):
    input_dir = _make_inputs(tmp_path, fits_names=("a.fits", "b.fits"))
    csv_path = _write_csv(
        input_dir, [["file_path", "weight"], ["a.fits", "1.0"], ["b.fits", "0.5"]]
    )
    parsed = parse_stack_plan_csv(str(csv_path))
    assert parsed.ordered_files == [
        str(input_dir / "a.fits"),
        str(input_dir / "b.fits"),
    ]
    assert parsed.weights == ["1.0", "0.5"]


def test_parse_order_file_with_index_column(tmp_path):
    input_dir = _make_inputs(tmp_path, fits_names=("a.fits", "b.fits"))
    csv_path = _write_csv(input_dir, [["1", "a.fits"], ["2", "b.fits"]])
    parsed = parse_stack_plan_csv(str(csv_path))
    assert parsed.ordered_files == [
        str(input_dir / "a.fits"),
        str(input_dir / "b.fits"),
    ]


def test_parse_relative_paths_resolved_against_csv_dir(tmp_path):
    input_dir = _make_inputs(tmp_path, fits_names=())
    sub = input_dir / "sub"
    sub.mkdir()
    (sub / "c.fits").write_bytes(b"")
    csv_path = _write_csv(input_dir, [["sub/c.fits"]])
    parsed = parse_stack_plan_csv(str(csv_path))
    assert parsed.ordered_files == [str(sub / "c.fits")]


def test_parse_missing_listed_file_raises(tmp_path):
    input_dir = _make_inputs(tmp_path, fits_names=("a.fits",))
    csv_path = _write_csv(input_dir, [["a.fits"], ["missing.fits"]])
    with pytest.raises(BoringCsvError) as exc:
        parse_stack_plan_csv(str(csv_path))
    assert "File listed in stack_plan.csv not found" in str(exc.value)
    assert "missing.fits" in str(exc.value)


def test_parse_header_only_csv_raises(tmp_path):
    input_dir = _make_inputs(tmp_path, fits_names=("a.fits",))
    csv_path = _write_csv(input_dir, [["file_path"]])
    with pytest.raises(BoringCsvError) as exc:
        parse_stack_plan_csv(str(csv_path))
    assert "no valid file paths" in str(exc.value)


def test_csv_path_for_joins_stack_plan():
    assert csv_path_for("/in") == os.path.join("/in", "stack_plan.csv")


# --------------------------------------------------------------------------
# Pure command builder
# --------------------------------------------------------------------------
def test_build_boring_request_command_shape():
    req = build_boring_request(
        csv_path="/in/stack_plan.csv",
        output_dir="/out",
        batch_size=1,
        chunk_size=77,
        normalize_method="sky_mean",
        save_final_as_float32=True,
        final_combine="reproject_coadd",
        max_mem_gb=16.0,
        python_executable="/usr/bin/python3",
    )
    cmd = req.command
    assert cmd[0] == "/usr/bin/python3"
    assert cmd[1] == resolve_boring_script_path()
    assert cmd[1].endswith("boring_stack.py")

    def arg(name):
        return cmd[cmd.index(name) + 1]

    assert "--csv" in cmd and arg("--csv") == "/in/stack_plan.csv"
    assert "--out" in cmd and arg("--out") == "/out"
    assert "--batch-size" in cmd and arg("--batch-size") == "1"
    assert "--max-mem" in cmd and arg("--max-mem") == "16.0"
    assert "--chunk-size" in cmd and arg("--chunk-size") == "77"
    assert "--log-dir" in cmd and arg("--log-dir") == os.path.join("/out", "logs")
    assert "--norm" in cmd and arg("--norm") == "sky_mean"
    assert "--save-as-float32" in cmd
    assert "--final-combine" in cmd and arg("--final-combine") == "reproject_coadd"


def test_build_boring_request_no_save_as_float32_flag():
    req = build_boring_request(
        csv_path="/in/stack_plan.csv",
        output_dir="/out",
        save_final_as_float32=False,
        python_executable="/py",
    )
    assert "--no-save-as-float32" in req.command
    assert "--save-as-float32" not in req.command


# --------------------------------------------------------------------------
# Checkbox <-> batch size synchronisation (Tk parity)
# --------------------------------------------------------------------------
def test_checkbox_sets_batch_to_one_and_locks_spinbox(qapp):
    win = MainWindow()
    try:
        assert win.batch_spin.value() == 0
        assert win.batch_spin.isEnabled()
        win.boring_check.setChecked(True)
        assert win.batch_spin.value() == 1
        assert not win.batch_spin.isEnabled()
    finally:
        win.shutdown()


def test_batch_one_checks_checkbox(qapp):
    win = MainWindow()
    try:
        win.batch_spin.setValue(1)
        assert win.boring_check.isChecked()
    finally:
        win.shutdown()


def test_batch_away_from_one_unchecks_checkbox(qapp):
    win = MainWindow()
    try:
        win.batch_spin.setValue(1)
        assert win.boring_check.isChecked()
        win.batch_spin.setValue(5)
        assert not win.boring_check.isChecked()
        assert win.batch_spin.isEnabled()
        assert win.batch_spin.value() == 5
    finally:
        win.shutdown()


def test_uncheck_resets_batch_to_zero_and_unlocks(qapp):
    win = MainWindow()
    try:
        win.boring_check.setChecked(True)
        assert win.batch_spin.value() == 1
        win.boring_check.setChecked(False)
        assert win.batch_spin.value() == 0
        assert win.batch_spin.isEnabled()
    finally:
        win.shutdown()


def test_boring_mode_gates_drizzle_controls(qapp):
    win = MainWindow()
    try:
        win.drizzle_check.setChecked(True)
        assert win.drizzle_check.isEnabled()
        win.boring_check.setChecked(True)
        # Drizzle is incompatible with boring mode: disabled + unchecked.
        assert not win.drizzle_check.isEnabled()
        assert not win.drizzle_check.isChecked()
        assert not win.drizzle_mode_combo.isEnabled()
        assert not win.drizzle_group_spin.isEnabled()
        # Un-checking boring re-enables the drizzle *checkbox* only.  The
        # sub-options stay gated by the (now force-unchecked) drizzle flag —
        # M16 added Tk-parity drizzle gating (`_update_drizzle_options_state`),
        # so ``drizzle_mode_combo`` / ``drizzle_group_spin`` are enabled by the
        # Enable-drizzle flag, not merely by "not boring".
        win.boring_check.setChecked(False)
        assert win.drizzle_check.isEnabled()
        assert not win.drizzle_mode_combo.isEnabled()
        assert not win.drizzle_group_spin.isEnabled()
        # Re-checking drizzle re-enables the mode combo; the group-size spin is
        # enabled only in the Large-dataset (Incremental) mode.
        win.drizzle_check.setChecked(True)
        assert win.drizzle_mode_combo.isEnabled()
        assert not win.drizzle_group_spin.isEnabled()
        win.drizzle_mode_combo.setCurrentText("Incremental")
        assert win.drizzle_group_spin.isEnabled()
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Start routing / preflight
# --------------------------------------------------------------------------
def test_missing_csv_blocks_start_and_logs(qapp, tmp_path):
    input_dir = _make_inputs(tmp_path, fits_names=("a.fits",))
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    fakes = []
    controller_calls = []

    def factory():
        fake = FakeBoringRunner()
        fakes.append(fake)
        return fake

    win = MainWindow(boring_runner_factory=factory)

    def spy_start(request, **kwargs):
        controller_calls.append(request)

    win.controller.start = spy_start
    try:
        win.input_edit.setText(str(input_dir))
        win.output_edit.setText(str(output_dir))
        win.batch_spin.setValue(1)
        win.start_button.click()

        assert controller_calls == []
        assert fakes == []  # runner never even created
        assert win.is_running is False
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()

        text = win.log_view.toPlainText()
        assert "Cannot start boring stack" in text
        assert "stack_plan.csv not found" in text
        assert "Cannot start boring stack" in win.statusBar().currentMessage()
    finally:
        win.shutdown()


def test_missing_listed_file_blocks_start(qapp, tmp_path):
    input_dir = _make_inputs(tmp_path, fits_names=("a.fits",))
    _write_csv(input_dir, [["a.fits"], ["missing.fits"]])
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    fakes = []
    controller_calls = []

    def factory():
        fake = FakeBoringRunner()
        fakes.append(fake)
        return fake

    win = MainWindow(boring_runner_factory=factory)

    def spy_start(request, **kwargs):
        controller_calls.append(request)

    win.controller.start = spy_start
    try:
        win.input_edit.setText(str(input_dir))
        win.output_edit.setText(str(output_dir))
        win.batch_spin.setValue(1)
        win.start_button.click()

        assert controller_calls == []
        assert fakes == []
        assert win.is_running is False
        text = win.log_view.toPlainText()
        assert "File listed in stack_plan.csv not found" in text
    finally:
        win.shutdown()


def test_valid_csv_routes_to_boring_runner_not_controller(qapp, tmp_path):
    input_dir = _make_inputs(tmp_path, fits_names=("a.fits", "b.fits"))
    csv_path = _write_csv(input_dir, [["a.fits"], ["b.fits"]])
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    fakes = []
    controller_calls = []

    def factory():
        fake = FakeBoringRunner()
        fakes.append(fake)
        return fake

    win = MainWindow(boring_runner_factory=factory)

    def spy_start(request, **kwargs):
        controller_calls.append(request)

    win.controller.start = spy_start
    try:
        win.input_edit.setText(str(input_dir))
        win.output_edit.setText(str(output_dir))
        win.batch_spin.setValue(1)
        win.start_button.click()

        assert controller_calls == []  # RunController.start NOT called
        assert len(fakes) == 1
        assert len(fakes[0].start_calls) == 1
        req = fakes[0].start_calls[0]
        assert isinstance(req, BoringRunRequest)
        assert req.csv_path == str(csv_path)
        assert req.output_dir == str(output_dir)
        assert req.batch_size == 1

        # While boring is active: Start disabled, Stop enabled.
        assert win.is_running is True
        assert not win.start_button.isEnabled()
        assert win.stop_button.isEnabled()
    finally:
        win.shutdown()


def test_valid_csv_command_arguments(qapp, tmp_path):
    input_dir = _make_inputs(tmp_path, fits_names=("a.fits",))
    _write_csv(input_dir, [["a.fits"]])
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    fakes = []

    def factory():
        fake = FakeBoringRunner()
        fakes.append(fake)
        return fake

    win = MainWindow(boring_runner_factory=factory)
    try:
        win.input_edit.setText(str(input_dir))
        win.output_edit.setText(str(output_dir))
        win.batch_spin.setValue(1)
        win._settings_widgets["save_final_as_float32"].setChecked(True)
        win._settings_widgets["stack_norm_method"].setCurrentText("linear_fit")
        win.start_button.click()

        req = fakes[0].start_calls[0]
        cmd = req.command

        def arg(name):
            return cmd[cmd.index(name) + 1]

        assert cmd[0] == sys.executable
        assert cmd[1] == resolve_boring_script_path()
        assert "--csv" in cmd and arg("--csv") == str(input_dir / "stack_plan.csv")
        assert "--out" in cmd and arg("--out") == str(output_dir)
        assert "--batch-size" in cmd and arg("--batch-size") == "1"
        assert "--chunk-size" in cmd and int(arg("--chunk-size")) >= 0
        assert "--log-dir" in cmd and arg("--log-dir") == str(output_dir / "logs")
        assert "--norm" in cmd and arg("--norm") == "linear_fit"
        assert "--save-as-float32" in cmd
        assert "--final-combine" in cmd and arg("--final-combine") == "mean"
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Boring lifecycle (buttons / run state)
# --------------------------------------------------------------------------
def test_boring_finished_resets_state(qapp, tmp_path):
    win, fakes, _in, _out = _window_with_fake(tmp_path, csv_rows=[["a.fits"]])
    try:
        win.start_button.click()
        assert win.is_running is True
        fake = fakes[0]
        fake.signal_finished(0)

        assert win.is_running is False
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()
        assert win.progress.value() == 100
        assert "Boring stack finished" in win.log_view.toPlainText()
    finally:
        win.shutdown()


def test_boring_failure_resets_state_and_logs(qapp, tmp_path):
    win, fakes, _in, _out = _window_with_fake(tmp_path, csv_rows=[["a.fits"]])
    try:
        win.start_button.click()
        fake = fakes[0]
        fake.signal_failed("boom")

        assert win.is_running is False
        assert "Boring stack failed: boom" in win.log_view.toPlainText()
        assert "Boring stack failed" in win.statusBar().currentMessage()
    finally:
        win.shutdown()


def test_stop_cancels_boring_runner(qapp, tmp_path):
    win, fakes, _in, _out = _window_with_fake(tmp_path, csv_rows=[["a.fits"]])
    try:
        win.start_button.click()
        fake = fakes[0]
        assert win.stop_button.isEnabled()
        win.stop_button.click()
        assert fake.cancel_called is True
        fake.signal_cancelled()

        assert win.is_running is False
        assert "Boring stack cancelled" in win.log_view.toPlainText()
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Source / import hygiene for the new modules
# --------------------------------------------------------------------------
def test_boring_modules_source_is_tk_engine_free():
    files = [
        ROOT / "seestar" / "gui_qt" / "boring_route.py",
        ROOT / "seestar" / "gui_qt" / "boring_runner.py",
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


def test_boring_route_fresh_process_import_hygiene():
    """Fresh interpreter: importing the boring route/runner must stay clean."""
    import subprocess

    code = (
        "import sys\n"
        "import seestar.gui_qt.boring_route  # noqa: F401\n"
        "import seestar.gui_qt.boring_runner  # noqa: F401\n"
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
        f"boring route import hygiene violated: stdout={proc.stdout!r} "
        f"stderr={proc.stderr!r}"
    )
