"""Stage E2 — GUI/UX migration: HQ RAM Limit removed, AUTO is the product
CPU memory policy.

Contract (see ``docs/8.4.0_state.md`` stage E2): the "HQ RAM Limit (GB)"
editable field disappears as a normal user-facing scientific control across
the product GUI; the normal UX is a read-only "CPU memory policy: Automatic"
status; the legacy persisted HQ RAM value never silently overrides AUTO; the
expert override remains available ONLY through the explicit env/CLI seam
(``ZSSS_CPU_MEMORY_OVERRIDE_BYTES``), provenance-visible, never a normal GUI
knob.

These tests exercise the REAL Qt main window / run handoff / backend runner /
E1 policy-capture seams (offscreen):
* persisted legacy "HQ RAM Limit = 16 GB" in the settings model does NOT
  become the runtime budget of the normal 8.4.0 product policy (no
  multi-GiB allocation required — policy record only);
* the normal GUI/state path no longer exposes an editable HQ RAM field; the
  AUTO status row is present; no new tuning knob is introduced;
* the expert override (env seam) still works and is provenance-visible
  (mode=override, requested_budget_bytes).

No real stacking, no multi-GiB allocation, no Tk.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QLabel

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt.backend_runner import (
    BackendRunResult,
    SeestarQueuedStackerBackend,
)
from seestar.gui_qt.settings_state import QtSettingsState
from seestar.queuep.queue_manager import SeestarQueuedStacker

GIB = 1024 ** 3


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


def _policy_stack(available: int = 64 * GIB, total: int = 64 * GIB):
    """Bare E1-capable stacker with deterministic RAM overrides and a
    poisoned legacy ``max_hq_mem`` attribute."""
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.logger = logging.getLogger("zsss.e2.migration")
    o.request_gpu = False
    o.max_stack_workers = 1
    # Legacy HQ RAM attribute (historical persisted value, GB -> bytes): must
    # NEVER become the AUTO runtime budget.
    o.max_hq_mem = 16 * GIB
    o._cpu_total_ram_bytes_override = int(total)
    o._cpu_available_ram_bytes_override = int(available)
    o._cpu_process_rss_bytes_override = int(400 * 1024 ** 2)
    o._cpu_mem_policy_preflight = None
    return o


# ---------------------------------------------------------------------------
# 1. Persisted legacy "HQ RAM Limit = 16 GB" never becomes the runtime budget
# ---------------------------------------------------------------------------
def test_persisted_legacy_hq_ram_never_becomes_runtime_budget():
    """Normal 8.4.0 product policy with a persisted historical 16 GB value:
    the effective policy is AUTO and the legacy value does NOT become the
    runtime budget (policy record only — no multi-GiB allocation)."""
    o = _policy_stack(available=64 * GIB)
    # Sanity: the legacy attribute is present (poison would be consumed if the
    # E1 policy still consulted it).
    assert o.max_hq_mem == 16 * GIB
    pre = o._capture_cpu_memory_policy_preflight()
    assert pre is not None
    assert pre.mode == "AUTO"
    # AUTO ceiling comes from available RAM minus the named reserve — NOT from
    # the legacy 16 GiB persisted value.
    assert pre.policy_ceiling_bytes != 16 * GIB
    assert pre.policy_ceiling_bytes < 64 * GIB
    assert pre.requested_budget_bytes is None


def test_persisted_legacy_hq_ram_in_settings_model_not_forwarded(qapp, window):
    """A persisted legacy ``max_hq_mem_gb = 16.0`` in the settings model is
    retained for migration/diagnostics but never forwarded to the run
    request (the GUI no longer exposes or consumes it as a control)."""
    window.settings_state.max_hq_mem_gb = 16.0  # historical persisted value
    state = window.collect_settings_state()
    assert state.max_hq_mem_gb == 16.0  # migration/diagnostics carrier only
    request = window.build_run_request()
    assert "max_hq_mem_gb" not in request.backend_kwargs
    assert "max_hq_mem" not in request.backend_kwargs
    assert "use_gpu" in request.backend_kwargs  # seam still works


def test_persisted_legacy_hq_ram_not_applied_by_backend(qapp, window):
    """End-to-end seam: even with a persisted 16 GB legacy value, the Qt
    backend adapter does NOT set any max_hq_mem on the stacker instance."""
    instances = []

    class _FakeStacker:
        def __init__(self, **kwargs):
            self.init_kwargs = dict(kwargs)
            self.align_on_disk = None
            self.progress_cb = None
            self.start_kwargs = None
            self._running = False

        def set_progress_callback(self, cb):
            self.progress_cb = cb

        def start_processing(self, **kwargs):
            self.start_kwargs = dict(kwargs)
            self._running = True
            return True

        def is_running(self):
            self._running = False
            return False

        def stop(self):
            self._running = False

    def factory(**kwargs):
        stacker = _FakeStacker(**kwargs)
        instances.append(stacker)
        return stacker

    window.settings_state.max_hq_mem_gb = 16.0
    backend = SeestarQueuedStackerBackend(
        stacker_factory=factory, poll_interval=0.001
    )
    request = window.build_run_request()
    result = backend.run(request, lambda p: None, lambda m: None, lambda: False)
    assert result is BackendRunResult.FINISHED
    stacker = instances[0]
    # Stage E2: no legacy budget is forced on the engine instance (AUTO
    # policy resolves at execution); GPU intent seam still applies.
    assert not hasattr(stacker, "max_hq_mem") or stacker.max_hq_mem is None
    assert "max_hq_mem_gb" not in stacker.start_kwargs


# ---------------------------------------------------------------------------
# 2. GUI no longer exposes an editable HQ RAM field; AUTO status present; no
#    new tuning knob.
# ---------------------------------------------------------------------------
def test_no_editable_hq_ram_field_and_auto_status_present(qapp, window):
    assert not hasattr(window, "max_hq_mem_spin"), (
        "the HQ RAM QSpinBox must be removed, not kept disabled"
    )
    assert isinstance(window.cpu_memory_status_label, QLabel)
    status = window.cpu_memory_status_label.text().lower()
    assert "automatic" in status
    # No new tuning knob appears next to the AUTO status.
    assert not hasattr(window, "cpu_memory_combo")
    assert not hasattr(window, "winsor_ram_factor_spin")
    assert not hasattr(window, "tile_memory_spin")


def test_run_request_has_no_hq_ram_fields(qapp, window):
    request = window.build_run_request()
    assert "max_hq_mem_gb" not in request.backend_kwargs
    assert "max_hq_mem" not in request.backend_kwargs
    # GPU intent is the only memory-adjacent seam and stays present.
    assert "use_gpu" in request.backend_kwargs


# ---------------------------------------------------------------------------
# 3. Expert override env seam still works and is provenance-visible
# ---------------------------------------------------------------------------
def test_override_env_is_provenance_visible(monkeypatch):
    o = _policy_stack(available=64 * GIB)
    monkeypatch.setenv("ZSSS_CPU_MEMORY_OVERRIDE_BYTES", str(2 * GIB))
    pre = o._capture_cpu_memory_policy_preflight()
    assert pre is not None
    assert pre.mode == "OVERRIDE"
    assert pre.policy_ceiling_bytes == 2 * GIB
    assert pre.requested_budget_bytes == 2 * GIB


def test_override_env_does_not_add_gui_knob(qapp, window, monkeypatch):
    """The override is an env seam: it is provenance-visible to the engine but
    never surfaces as a GUI control or a run-request field."""
    monkeypatch.setenv("ZSSS_CPU_MEMORY_OVERRIDE_BYTES", str(2 * GIB))
    assert not hasattr(window, "max_hq_mem_spin")
    request = window.build_run_request()
    assert "max_hq_mem_gb" not in request.backend_kwargs
    assert "max_hq_mem" not in request.backend_kwargs
