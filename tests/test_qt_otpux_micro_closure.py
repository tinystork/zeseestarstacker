"""ZSSS-MICRO-CLOSURE tests: real startup-refusal propagation & user guidance.

The human manual test (an output directory containing previous processing
state + an incompatible run configuration) previously ended on the generic
technical failure path — a Critical "Run failed" box carrying the flattened
``"SeestarQueuedStacker.start_processing() reported it did not start"`` string
— instead of the structured ``OUTPUT_STATE_INCOMPATIBLE`` handling
(``_on_run_refused`` / ``_format_refusal`` / one owned localized Warning
``QMessageBox``).

Contract under test:

* The engine classifies *every* resume-requested early refusal as the known
  ``OUTPUT_STATE_INCOMPATIBLE`` condition (covered engine-side by
  ``tests/test_zsss_startup_refusal_qm.py``), so the Qt adapter raises the
  structured ``StartupRefusedError`` instead of the generic false-start string.
* End-to-end (real ``SeestarQueuedStackerBackend`` + RunWorker + RunController
  + MainWindow, fake engine refusing with the structured carrier): exactly one
  **Warning** box with the localized mode-independent actionable body, NO
  Critical "Run failed" box, GUI idle, old output contents byte-identical, no
  run log written, and the user can select another folder and retry.
* A genuinely unknown false start (engine returns ``False`` with no structured
  carrier) stays the generic **Critical** technical failure.
* FR and EN both carry the exact new wording (proper FR typographic
  apostrophe, ``\\n\\n`` paragraph separation); the old mode-specific keys are
  gone.
* State-preservation matrix: refused runs never touch the output folder, never
  open a run log, and restore a consistent idle GUI.

Engine-free (fake engines only, real adapter).  ``QT_QPA_PLATFORM=offscreen``
is set defensively before any ``QApplication`` is created.
"""

from __future__ import annotations

import os
import threading
import time
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from seestar.gui_qt import MainWindow, RunStatus, create_application
from seestar.gui_qt import localization
from seestar.gui_qt.backend_runner import SeestarQueuedStackerBackend
from seestar.gui_qt.run_bridge import build_run_request
from seestar.gui_qt.settings_state import QtSettingsState
from seestar.gui_qt.startup_refusal import (
    CODE_OUTPUT_STATE_INCOMPATIBLE,
    StartupRefusalPayload,
    StartupRefusedError,
)

EN_TITLE = "Output folder already in use"
EN_BODY = (
    "The selected output folder contains data from a previous processing run.\n\n"
    "If you want to resume that run, make sure the selected processing mode "
    "supports resume.\n\n"
    "If you want to start a new stack, select a new or empty output folder."
)
FR_TITLE = "Dossier de sortie déjà utilisé"
FR_BODY = (
    "Le dossier de sortie sélectionné contient les données d\u2019un "
    "traitement précédent.\n\n"
    "Si vous souhaitez reprendre ce traitement, vérifiez que le mode "
    "sélectionné est compatible avec la reprise.\n\n"
    "Si vous souhaitez démarrer un nouveau stack, sélectionnez un nouveau "
    "dossier de sortie ou un dossier vide."
)
GENERIC_FALSE_START = (
    "SeestarQueuedStacker.start_processing() reported it did not start"
)


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


# --------------------------------------------------------------------------
# Fake engines (no real engine import; the real adapter drives them)
# --------------------------------------------------------------------------
class _RefusingEngine:
    """Fake engine reproducing the *real* refusal seam: set the structured
    ``startup_refusal`` carrier on self, then return ``False``.

    Mirrors ``SeestarQueuedStacker.start_processing`` (the early
    resume-preflight site): ``startup_refusal`` is reset at each start attempt
    and set to the structured carrier before the refusal return.
    """

    def __init__(self, detail, semantic_data=None):
        self.startup_refusal = None
        self.detail = detail
        self.semantic_data = dict(semantic_data or {})
        self.output_folder = None
        self.align_on_disk = None
        self.stop_called = False

    def set_progress_callback(self, cb):
        pass

    def set_lifecycle_callback(self, cb):
        pass

    def set_preview_callback(self, cb):
        pass

    def start_processing(self, **kwargs):
        self.startup_refusal = None
        self.output_folder = kwargs.get("output_dir")
        self.startup_refusal = types.SimpleNamespace(
            code=CODE_OUTPUT_STATE_INCOMPATIBLE,
            technical_detail=self.detail,
            semantic_key="output_state_incompatible",
            semantic_data=dict(self.semantic_data),
        )
        return False

    def is_running(self):
        return False

    def stop(self):
        self.stop_called = True


class _UnknownFalseStartEngine(_RefusingEngine):
    """Fake engine whose ``start_processing`` returns False with NO structured
    carrier — the genuinely unknown false start (stays generic)."""

    def start_processing(self, **kwargs):
        self.startup_refusal = None
        self.output_folder = kwargs.get("output_dir")
        return False


class _AcceptingEngine:
    """Fake engine that accepts and runs a short worker thread to completion."""

    def __init__(self):
        self.startup_refusal = None
        self.output_folder = None
        self.align_on_disk = None
        self.processing_active = False
        self.processing_thread = None
        self.progress_cb = None
        self.stop_called = False

    def set_progress_callback(self, cb):
        self.progress_cb = cb

    def set_lifecycle_callback(self, cb):
        pass

    def set_preview_callback(self, cb):
        pass

    def start_processing(self, **kwargs):
        self.startup_refusal = None
        self.output_folder = kwargs.get("output_dir")
        self.processing_thread = threading.Thread(target=self._worker, daemon=True)
        self.processing_thread.start()
        return True

    def _worker(self):
        self.processing_active = True
        if self.progress_cb is not None:
            self.progress_cb("stacking", 10, "INFO")
        time.sleep(0.05)
        self.processing_active = False

    def is_running(self):
        return bool(self.processing_active) and (
            self.processing_thread is not None
            and self.processing_thread.is_alive()
        )

    def stop(self):
        self.stop_called = True
        self.processing_active = False


def _backend_for(stacker):
    return SeestarQueuedStackerBackend(
        stacker_factory=lambda **kw: stacker, poll_interval=0.001
    )


class _RefuseThenAcceptFactory:
    """Backend factory: first Start refuses (known output state), second
    accepts — the exact manual scenario: refuse, then pick a new folder and
    retry successfully.  Each Start builds a fresh real backend adapter."""

    def __init__(self, detail, semantic_data=None):
        self.detail = detail
        self.semantic_data = semantic_data
        self.phase = 0
        self.backends = []

    def __call__(self):
        if self.phase == 0:
            stacker = _RefusingEngine(self.detail, self.semantic_data)
        else:
            stacker = _AcceptingEngine()
        self.phase += 1
        backend = _backend_for(stacker)
        self.backends.append(backend)
        return backend


def _populated_output_dir(base: Path) -> Path:
    """An output dir holding previous processing state (manual scenario)."""
    out_dir = base / "out"
    out_dir.mkdir()
    (out_dir / "batches_count.txt").write_text("3", encoding="utf-8")
    sentinel = out_dir / "final_stack.fits"
    sentinel.write_bytes(b"PREVIOUS-PROCESSING-STATE-BYTES")
    return out_dir


# --------------------------------------------------------------------------
# A + F. known output state -> structured Warning, no Critical, retry works
# --------------------------------------------------------------------------
def test_known_output_state_refusal_full_witness(qapp, tmp_path):
    """End-to-end witness on the real start path (real adapter, RunWorker,
    RunController, MainWindow): the engine's known OUTPUT_STATE_INCOMPATIBLE
    refusal surfaces as exactly one localized Warning box — never the generic
    Critical "Run failed" — old output state is untouched, and the user can
    select another folder and retry successfully."""
    out_dir = _populated_output_dir(tmp_path)
    sentinel_before = (out_dir / "final_stack.fits").read_bytes()

    factory = _RefuseThenAcceptFactory(
        "scientific configuration mismatch", {"mode": "plain_classic"}
    )
    win = MainWindow(backend_factory=factory)
    try:
        win.input_edit.setText(str(tmp_path / "input"))
        win.output_edit.setText(str(out_dir))

        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)

        # Exactly one owned box, Warning severity, localized EN title/body.
        assert win.error_box_count == 1
        box = win.error_message_box
        assert box is not None and box.isVisible()
        assert box.icon() is QMessageBox.Icon.Warning, "must be Warning, not Critical"
        assert box.windowTitle() == EN_TITLE
        assert box.text() == EN_BODY
        assert box.textFormat() == Qt.TextFormat.PlainText
        # No generic Critical "Run failed" box: the refusal path never called
        # _on_run_failed, so the generic string is absent from the dialog.
        assert "did not start" not in box.text()
        assert "did not start" not in box.windowTitle()

        # Status bar carries the localized title; the run log keeps the
        # precise technical detail (never the dialog's primary text).
        assert win.statusBar().currentMessage() == EN_TITLE
        log_text = win.log_view.toPlainText()
        assert EN_BODY in log_text
        assert "scientific configuration mismatch" in log_text

        # Run not started; GUI idle; controller terminal + thread reaped.
        assert win.controller.status is RunStatus.FAILED
        assert not win.controller.has_live_thread
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()
        # A refused run never opened a run log and never touched the folder.
        assert win.controller.run_log is None
        assert (out_dir / "final_stack.fits").read_bytes() == sentinel_before
        assert sorted(p.name for p in out_dir.iterdir()) == [
            "batches_count.txt",
            "final_stack.fits",
        ]

        # -------------------------------------------------- retry in a new folder
        new_out = tmp_path / "fresh_out"
        new_out.mkdir()
        win.output_edit.setText(str(new_out))
        win.start_button.click()
        assert _pump_until(
            qapp,
            lambda: win.is_running is False
            and win.controller.status is RunStatus.FINISHED,
        )
        # The accepted run wrote its durable run log into the NEW folder only.
        assert list(new_out.glob("zsss_run_*.log")), "accepted run must open a run log"
        # Old output contents still untouched after the successful retry.
        assert (out_dir / "final_stack.fits").read_bytes() == sentinel_before
        assert sorted(p.name for p in out_dir.iterdir()) == [
            "batches_count.txt",
            "final_stack.fits",
        ]
        # Success never adds an error box (still exactly the one Warning).
        assert win.error_box_count == 1
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# B. genuinely unknown false start stays generic Critical
# --------------------------------------------------------------------------
def test_unknown_false_start_stays_generic_critical(qapp, tmp_path):
    """An engine that returns False with NO structured carrier must keep the
    generic technical failure path: Critical "Run failed" with the flattened
    string, never the localized Warning guidance."""
    stacker = _UnknownFalseStartEngine("unused")
    win = MainWindow(backend_factory=lambda: _backend_for(stacker))
    try:
        win.input_edit.setText(str(tmp_path))
        win.output_edit.setText(str(tmp_path / "out"))
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)

        assert win.error_box_count == 1
        box = win.error_message_box
        assert box is not None and box.isVisible()
        assert box.icon() is QMessageBox.Icon.Critical
        assert box.windowTitle() == "Run failed"
        assert GENERIC_FALSE_START in box.text()
        # The structured guidance must not leak into the generic path.
        assert box.text() != EN_BODY
        assert "Output folder already in use" not in box.text()

        assert win.statusBar().currentMessage().startswith("Failed:")
        assert GENERIC_FALSE_START in win.log_view.toPlainText()
        assert win.controller.status is RunStatus.FAILED
        assert not win.controller.has_live_thread
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# C. FR/EN localization: parity + exact wording (proper FR apostrophes)
# --------------------------------------------------------------------------
def test_refusal_wording_keys_have_en_fr_parity():
    for key in (
        "startup_refusal_output_state_incompatible_title",
        "startup_refusal_output_state_incompatible_body",
    ):
        entry = localization.TRANSLATIONS[key]
        assert set(entry) == {"en", "fr"}, f"key {key!r} must have en+fr"
        assert entry["en"], f"key {key!r} has empty en"
        assert entry["fr"], f"key {key!r} has empty fr"
    # The mode-specific keys are replaced by the single mode-independent body.
    assert "startup_refusal_output_state_incompatible_body_generic" not in (
        localization.TRANSLATIONS
    )
    assert "startup_refusal_mode_mosaic" not in localization.TRANSLATIONS
    assert "startup_refusal_mode_reproject" not in localization.TRANSLATIONS


def test_refusal_wording_exact_en_fr():
    assert (
        localization.translate(
            "startup_refusal_output_state_incompatible_title", "en"
        )
        == EN_TITLE
    )
    assert (
        localization.translate(
            "startup_refusal_output_state_incompatible_body", "en"
        )
        == EN_BODY
    )
    assert (
        localization.translate(
            "startup_refusal_output_state_incompatible_title", "fr"
        )
        == FR_TITLE
    )
    assert (
        localization.translate(
            "startup_refusal_output_state_incompatible_body", "fr"
        )
        == FR_BODY
    )
    # Proper FR typographic apostrophe (U+2019) in "d'un".
    assert "\u2019" in FR_BODY
    assert "d'un" not in FR_BODY
    # The wording never says the folder cannot be reused and never names a mode.
    assert "cannot be reused" not in EN_BODY
    assert "réutilis" not in FR_BODY
    assert "Drizzle" not in EN_BODY and "Drizzle" not in FR_BODY


def test_fr_refusal_box_end_to_end(qapp, tmp_path):
    """Full French path through the real start pipeline: the owned box shows
    the exact FR replacement title/body."""
    stacker = _RefusingEngine(
        "resume limited to plain classic SUM/W", {"mode": "drizzle"}
    )
    win = MainWindow(backend_factory=lambda: _backend_for(stacker))
    try:
        win.language_combo.setCurrentText("Français")
        win.input_edit.setText(str(tmp_path))
        win.output_edit.setText(str(tmp_path / "out"))
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)

        assert win.error_box_count == 1
        box = win.error_message_box
        assert box is not None and box.isVisible()
        assert box.icon() is QMessageBox.Icon.Warning
        assert box.windowTitle() == FR_TITLE
        assert box.text() == FR_BODY
        assert win.statusBar().currentMessage() == FR_TITLE
        assert FR_BODY in win.log_view.toPlainText()
        assert win.controller.status is RunStatus.FAILED
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# D. state-preservation matrix (handler + payload contract)
# --------------------------------------------------------------------------
def test_on_run_refused_handler_state_preservation(qapp):
    """Direct handler matrix: one Warning box, localized, idle GUI, terminal
    time state, and the technical detail logged after the guidance body."""
    payload = StartupRefusalPayload(
        code=CODE_OUTPUT_STATE_INCOMPATIBLE,
        technical_detail="scientific configuration mismatch",
        semantic_key="output_state_incompatible",
        semantic_data={"mode": "plain_classic"},
    )
    win = MainWindow()
    try:
        win._on_run_refused(payload)

        assert not win.is_running
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()
        assert win.statusBar().currentMessage() == EN_TITLE
        log_text = win.log_view.toPlainText()
        assert EN_BODY in log_text
        assert "scientific configuration mismatch" in log_text
        assert win.error_box_count == 1
        box = win.error_message_box
        assert box.icon() is QMessageBox.Icon.Warning
        assert box.windowTitle() == EN_TITLE
        assert box.text() == EN_BODY
        assert "did not start" not in box.text()
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# adapter-level witness: real backend raises the structured refusal
# --------------------------------------------------------------------------
def test_adapter_raises_structured_refusal_for_known_engine_carrier(tmp_path):
    """The real ``SeestarQueuedStackerBackend`` maps an engine carrier set with
    a False return to ``StartupRefusedError`` (never the generic string), and
    never touches the output folder."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sentinel = out_dir / "previous.txt"
    sentinel.write_bytes(b"SENTINEL")
    before = sentinel.read_bytes()

    stacker = _RefusingEngine(
        "scientific configuration mismatch", {"mode": "plain_classic"}
    )
    backend = _backend_for(stacker)
    request = _make_request(
        input_folder=str(tmp_path), output_folder=str(out_dir)
    )

    with pytest.raises(StartupRefusedError) as exc_info:
        backend.run(request, lambda p: None, lambda m: None, lambda: False)

    assert exc_info.value.payload.code == CODE_OUTPUT_STATE_INCOMPATIBLE
    assert exc_info.value.payload.semantic_key == "output_state_incompatible"
    assert exc_info.value.payload.technical_detail == (
        "scientific configuration mismatch"
    )
    # No run log opened; output contents byte-identical.
    assert sentinel.read_bytes() == before
    assert sorted(p.name for p in out_dir.iterdir()) == ["previous.txt"]
    assert list(out_dir.glob("zsss_run_*.log")) == []


def test_adapter_generic_false_start_without_carrier(tmp_path):
    """The real adapter keeps the plain generic RuntimeError when the engine
    returns False with no structured carrier."""
    stacker = _UnknownFalseStartEngine("unused")
    backend = _backend_for(stacker)
    request = _make_request(
        input_folder=str(tmp_path), output_folder=str(tmp_path / "out")
    )

    with pytest.raises(RuntimeError) as exc_info:
        backend.run(request, lambda p: None, lambda m: None, lambda: False)

    assert str(exc_info.value) == GENERIC_FALSE_START
    assert not isinstance(exc_info.value, StartupRefusedError)
