"""Resume Contract v2 (RSM2-01) explicit-intent contract tests.

These tests pin the first bounded resume-contract slice: a stable, explicit
Fresh/Resume run intent carried end-to-end, differentiated startup refusals,
early-refusal cleanup, and the removal of the obsolete constructor-created
package-local ``cumulative_wht.dat``.

Covered matrix rows:

* empty output + FRESH -> no early intent refusal (fresh proceeds),
* empty output + RESUME -> ``RESUME_STATE_MISSING``,
* existing state + FRESH -> ``FRESH_OUTPUT_HAS_STATE`` (read-only, no mutation),
* existing classic state + RESUME -> proceeds (full HSI validation is exercised
  in ``test_resume.py``), and unsupported non-classic + RESUME ->
  ``RESUME_MODE_UNSUPPORTED`` (exercised in ``test_zsss_startup_refusal_qm.py``),
* persisted ``last_stack_path`` alone never sets RESUME,
* repeated Start after a refusal reaches fresh validation normally,
* the constructor has no package filesystem side effect.

No real stacking is performed.  The engine is imported lazily so the Qt-side
rows stay fast and import-hygiene-clean.
"""

from __future__ import annotations

import os
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Qt-side imports (pure; no engine import at module level).
from seestar.gui_qt.backend_runner import (  # noqa: E402
    BackendRunResult,
    SeestarQueuedStackerBackend,
)
from seestar.gui_qt.run_bridge import (  # noqa: E402
    RUN_INTENT_FRESH,
    RUN_INTENT_RESUME,
    RunRequest,
    build_run_request,
    normalize_run_intent,
)
from seestar.gui_qt.run_handoff import attach_run_settings  # noqa: E402
from seestar.gui_qt.settings_state import QtSettingsState  # noqa: E402
from seestar.gui_qt.startup_refusal import (  # noqa: E402
    CODE_FRESH_OUTPUT_HAS_STATE,
    CODE_RESUME_MODE_UNSUPPORTED,
    CODE_RESUME_STATE_MISSING,
)


# --------------------------------------------------------------------------
# run_config vocabulary
# --------------------------------------------------------------------------
def test_normalize_run_intent_closed_vocabulary():
    assert normalize_run_intent(None) == RUN_INTENT_FRESH
    assert normalize_run_intent("fresh") == RUN_INTENT_FRESH
    assert normalize_run_intent("resume") == RUN_INTENT_RESUME
    assert normalize_run_intent("RESUME") == RUN_INTENT_FRESH
    assert normalize_run_intent("") == RUN_INTENT_FRESH
    assert normalize_run_intent(123) == RUN_INTENT_FRESH
    assert normalize_run_intent(True) == RUN_INTENT_FRESH


# --------------------------------------------------------------------------
# Qt: default is FRESH; last_stack_path alone never sets RESUME
# --------------------------------------------------------------------------
def test_default_state_is_fresh():
    state = QtSettingsState()
    assert state.resume_intent == "fresh"
    assert state.resume_source == ""
    request = build_run_request(state)
    assert request.resume_intent == RUN_INTENT_FRESH
    assert request.resume_source is None


def test_last_stack_path_alone_stays_fresh():
    state = QtSettingsState(last_stack_path="/data/runs/last.fit")
    request = build_run_request(state)
    assert request.resume_intent == RUN_INTENT_FRESH
    assert request.resume_source is None


def test_explicit_resume_flows_through_run_request():
    state = QtSettingsState(
        resume_intent="resume",
        resume_source="/data/runs",
        last_stack_path="/data/runs/last.fit",
    )
    request = build_run_request(state)
    assert request.resume_intent == RUN_INTENT_RESUME
    assert request.resume_source == "/data/runs"


def test_attach_run_settings_preserves_intent():
    state = QtSettingsState(resume_intent="resume", resume_source="/data/runs")
    request = build_run_request(state)
    attached = attach_run_settings(request, use_gpu=False, max_hq_mem_gb=8.0)
    assert attached.resume_intent == RUN_INTENT_RESUME
    assert attached.resume_source == "/data/runs"
    # The original request is untouched.
    assert request.resume_intent == RUN_INTENT_RESUME


# --------------------------------------------------------------------------
# Backend adapter: resume_intent/resume_source reach start_processing
# --------------------------------------------------------------------------
class _RecordingStacker:
    def __init__(self, **kwargs) -> None:
        self.align_on_disk = None
        self.start_kwargs = None
        self.stop_called = False

    def set_progress_callback(self, cb) -> None:
        pass

    def start_processing(self, **kwargs):
        self.start_kwargs = dict(kwargs)
        return True

    def is_running(self) -> bool:
        return False

    def stop(self) -> None:
        self.stop_called = True


def test_backend_forwards_resume_intent_to_start_processing():
    instances = []

    def factory(**kwargs):
        stacker = _RecordingStacker(**kwargs)
        instances.append(stacker)
        return stacker

    backend = SeestarQueuedStackerBackend(stacker_factory=factory, poll_interval=0.001)
    state = QtSettingsState(resume_intent="resume", resume_source="/data/runs")
    request = attach_run_settings(
        build_run_request(state), use_gpu=state.use_gpu, max_hq_mem_gb=state.max_hq_mem_gb
    )

    result = backend.run(request, lambda p: None, lambda m: None, lambda: False)

    assert result is BackendRunResult.FINISHED
    stacker = instances[0]
    assert stacker.start_kwargs["resume_intent"] == RUN_INTENT_RESUME
    assert stacker.start_kwargs["resume_source"] == "/data/runs"


def test_backend_defaults_to_fresh_intent():
    instances = []

    def factory(**kwargs):
        stacker = _RecordingStacker(**kwargs)
        instances.append(stacker)
        return stacker

    backend = SeestarQueuedStackerBackend(stacker_factory=factory, poll_interval=0.001)
    request = attach_run_settings(build_run_request(QtSettingsState()))

    backend.run(request, lambda p: None, lambda m: None, lambda: False)

    assert instances[0].start_kwargs["resume_intent"] == RUN_INTENT_FRESH
    assert instances[0].start_kwargs["resume_source"] is None


# --------------------------------------------------------------------------
# Engine-level explicit-intent gates (lazy engine import)
# --------------------------------------------------------------------------
def _engine_stackers():
    from seestar.queuep.queue_manager import SeestarQueuedStacker, StartupRefusal

    return SeestarQueuedStacker, StartupRefusal


def _gate_stacker(out_dir, input_dir, *, drizzle=False):
    """Bare stacker with just enough state to reach the intent gate."""
    SeestarQueuedStacker, _ = _engine_stackers()
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    o.processing_active = False
    o.stop_processing = False
    o.user_requested_stop = False
    o.startup_refusal = None
    o.aligner = types.SimpleNamespace(stop_processing=False)
    o.autotuner = None
    o.current_folder = str(input_dir)
    o.output_folder = str(out_dir)
    o.is_mosaic_run = False
    o.drizzle_active_session = drizzle
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    return o


def _mark_state(out_dir) -> bytes:
    marker = Path(out_dir) / "batches_count.txt"
    marker.write_text("2", encoding="utf-8")
    return marker.read_bytes()


def test_fresh_over_existing_state_refuses_readonly(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    before = _mark_state(out)

    _, StartupRefusal = _engine_stackers()
    s = _gate_stacker(out, tmp_path / "in")
    result = s.start_processing(
        input_dir=str(tmp_path / "in"),
        output_dir=str(out),
        resume_intent="fresh",
    )

    assert result is False
    assert s.startup_refusal is not None
    assert s.startup_refusal.code == CODE_FRESH_OUTPUT_HAS_STATE
    assert s.startup_refusal.code == StartupRefusal.CODE_FRESH_OUTPUT_HAS_STATE
    # Read-only: no new files, existing state byte-identical.
    assert (out / "batches_count.txt").read_bytes() == before
    assert sorted(p.name for p in out.iterdir()) == ["batches_count.txt"]
    assert s.processing_active is False


def test_resume_without_state_refuses_missing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()  # empty: no recognized run state

    s = _gate_stacker(out, tmp_path / "in")
    result = s.start_processing(
        input_dir=str(tmp_path / "in"),
        output_dir=str(out),
        resume_intent="resume",
    )

    assert result is False
    assert s.startup_refusal is not None
    assert s.startup_refusal.code == CODE_RESUME_STATE_MISSING
    assert s.processing_active is False


def test_fresh_over_empty_does_not_refuse_at_intent_gate(tmp_path):
    out = tmp_path / "out"
    out.mkdir()  # empty

    s = _gate_stacker(out, tmp_path / "in")
    result = s.start_processing(
        input_dir=str(tmp_path / "in"),
        output_dir=str(out),
        resume_intent="fresh",
    )

    # Not an intent refusal: the run got past the intent gate and failed later
    # for an unrelated reason (the input directory does not exist).
    assert result is False
    assert s.startup_refusal is None
    assert s._resume_requested is False


def test_repeated_start_after_refusal_reaches_fresh_validation(tmp_path):
    out_state = tmp_path / "out_state"
    out_state.mkdir()
    _mark_state(out_state)
    out_empty = tmp_path / "out_empty"
    out_empty.mkdir()

    s = _gate_stacker(out_state, tmp_path / "in")
    r1 = s.start_processing(
        input_dir=str(tmp_path / "in"),
        output_dir=str(out_state),
        resume_intent="fresh",
    )
    assert r1 is False
    assert s.startup_refusal is not None
    assert s.startup_refusal.code == CODE_FRESH_OUTPUT_HAS_STATE
    assert s.processing_active is False

    # Second Start on the same instance: the stale refusal is reset and the
    # fresh run reaches normal validation (past the intent gate).
    r2 = s.start_processing(
        input_dir=str(tmp_path / "in"),
        output_dir=str(out_empty),
        resume_intent="fresh",
    )
    assert s.startup_refusal is None
    assert s._resume_requested is False
    # Fails later for an unrelated reason (missing input dir), never an intent
    # refusal and never a residual stale flag.
    assert r2 is False
    assert s.processing_active is False


# --------------------------------------------------------------------------
# Constructor: no package filesystem side effect (obsolete cumulative_wht.dat)
# --------------------------------------------------------------------------
def test_constructor_has_no_package_filesystem_side_effect():
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    pkg_dir = Path(__file__).resolve().parents[1] / "seestar"
    marker = pkg_dir / "cumulative_wht.dat"
    if marker.exists():
        marker.unlink()

    stacker = SeestarQueuedStacker()
    try:
        # Inert until a real output-bound allocation opens them.
        assert stacker.cumulative_wht_path is None
        assert stacker.cumulative_wht_memmap is None
        assert not marker.exists()
    finally:
        for name in ("quality_executor",):
            exe = getattr(stacker, name, None)
            if exe is not None:
                try:
                    exe.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass


def test_startup_refusal_codes_present():
    # The three new stable codes are exported from the Qt carrier and the
    # engine carrier (duck-typed, no cross-import).
    assert CODE_FRESH_OUTPUT_HAS_STATE == "FRESH_OUTPUT_HAS_STATE"
    assert CODE_RESUME_STATE_MISSING == "RESUME_STATE_MISSING"
    assert CODE_RESUME_MODE_UNSUPPORTED == "RESUME_MODE_UNSUPPORTED"

    from seestar.queuep.queue_manager import StartupRefusal

    assert StartupRefusal.CODE_FRESH_OUTPUT_HAS_STATE == "FRESH_OUTPUT_HAS_STATE"
    assert StartupRefusal.CODE_RESUME_STATE_MISSING == "RESUME_STATE_MISSING"
    assert StartupRefusal.CODE_RESUME_MODE_UNSUPPORTED == "RESUME_MODE_UNSUPPORTED"
