"""C21 — ``calibration_orientation`` seam propagation + generic guard.

Two things are pinned here:

1. A **generic guard**: no seam-only key (``use_gpu``, ``reference_origin_hint``,
   ``calibration_enabled``, ``calibration_master_folder``, ``calibration_orientation``,
   …) may ever reach ``SeestarQueuedStacker.start_processing``.  We split the full
   run surface and require ``set(start_kwargs) ⊆`` the *real* signature (via
   ``inspect.signature``).  A future forgotten seam key fails here, not in prod.

2. An **end-to-end GUI handoff**: ``QtSettingsState`` (orientation checkbox) →
   ``build_run_request`` → ``attach_run_settings`` → ``split_backend_kwargs`` +
   ``_apply_seam_kwargs`` ⇒ the three assertions (accepted start_kwargs,
   ``_calibration_orientation`` == "identity"/"", and
   ``_calibration_enabled``/``_calibration_master_folder``).

Qt-free engine import is done only inside the guard test (it needs the real
signature); the propagation test uses the fake-stacker backend path like M20.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from seestar.gui_qt.backend_runner import (
    BackendRunResult,
    SeestarQueuedStackerBackend,
)
from seestar.gui_qt.run_bridge import build_run_request, split_backend_kwargs
from seestar.gui_qt.run_handoff import attach_run_settings
from seestar.gui_qt.settings_state import QtSettingsState


def _full_request(orientation: str = "identity") -> object:
    """The full run surface a real checked Calibration tab would produce."""
    state = QtSettingsState(
        calibration_enabled=True,
        calibration_master_folder="/tmp/synthetic_masters",
        calibration_orientation=orientation,
    )
    return attach_run_settings(
        build_run_request(state),
        use_gpu=True,
        reference_origin_hint="ZEANALYSER_V1",
        calibration_enabled=True,
        calibration_master_folder="/tmp/synthetic_masters",
        calibration_orientation=orientation,
    )


def test_generic_guard_no_seam_key_reaches_start_processing():
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    accepted = set(inspect.signature(SeestarQueuedStacker.start_processing).parameters)
    start_kwargs, seam_kwargs = split_backend_kwargs(_full_request().backend_kwargs)

    unexpected = sorted(set(start_kwargs) - accepted)
    assert not unexpected, f"seam keys leaked into start_processing: {unexpected}"

    # And the seam keys actually landed on the seam side (not start).
    for key in (
        "use_gpu",
        "reference_origin_hint",
        "calibration_enabled",
        "calibration_master_folder",
        "calibration_orientation",
    ):
        assert key in seam_kwargs, f"{key} missing from seam_kwargs"
        assert key not in start_kwargs, f"{key} leaked into start_kwargs"


class _FakeStacker:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = dict(kwargs)
        self.start_kwargs = None
        self._running = False

    def set_progress_callback(self, cb) -> None:
        pass

    def start_processing(self, **kwargs):
        # Mirror the real TypeError surface: any kwarg not in the real signature
        # is rejected here via a strict signature check.
        from seestar.queuep.queue_manager import SeestarQueuedStacker

        accepted = set(inspect.signature(SeestarQueuedStacker.start_processing).parameters)
        extra = set(kwargs) - accepted
        if extra:
            raise TypeError(f"unexpected keyword arguments: {sorted(extra)}")
        self.start_kwargs = dict(kwargs)
        self._running = True
        return True

    def is_running(self) -> bool:
        self._running = False
        return False

    def stop(self) -> None:
        self._running = False


def _run(request) -> _FakeStacker:
    instances = []

    def factory(**kwargs):
        stacker = _FakeStacker(**kwargs)
        instances.append(stacker)
        return stacker

    backend = SeestarQueuedStackerBackend(stacker_factory=factory, poll_interval=0.001)
    result = backend.run(request, lambda p: None, lambda m: None, lambda: False)
    assert result is BackendRunResult.FINISHED
    return instances[0]


def test_propagation_checked_orientation_applies_seam_fields():
    stacker = _run(_full_request(orientation="identity"))

    # (a) start_processing accepted the split kwargs (no TypeError raised).
    assert stacker.start_kwargs is not None
    # (b) orientation declaration reached the engine instance.
    assert stacker._calibration_orientation == "identity"
    # (c) the sibling calibration seam fields reached the instance too.
    assert stacker._calibration_enabled is True
    assert stacker._calibration_master_folder == "/tmp/synthetic_masters"
    # And none of them leaked into the start_processing kwargs.
    for key in ("calibration_orientation", "calibration_enabled",
                "calibration_master_folder"):
        assert key not in stacker.start_kwargs, key


def test_propagation_unchecked_orientation_is_empty_string():
    stacker = _run(_full_request(orientation=""))

    assert stacker._calibration_orientation == ""
    assert stacker._calibration_enabled is True
    assert stacker._calibration_master_folder == "/tmp/synthetic_masters"
    assert "calibration_orientation" not in stacker.start_kwargs


def test_split_backend_kwargs_partitions_calibration_orientation():
    start_kwargs, seam_kwargs = split_backend_kwargs(_full_request().backend_kwargs)
    assert seam_kwargs["calibration_orientation"] == "identity"
    assert "calibration_orientation" not in start_kwargs
