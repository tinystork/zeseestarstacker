"""M3-D propagation contract: ``drizzle_group_size`` GUI/settings -> backend.

``drizzle_group_size`` is a RESOURCE/PREVIEW policy knob (size of the group
used for the incremental DISPLAY-ONLY preview cadence), never a science
setting.  This file pins the full wiring contract, end to end, without a real
Tk display and without spawning worker threads / process pools:

* the GUI ``backend_kwargs`` forward ``self.settings.drizzle_group_size`` into
  ``start_processing`` (static source check, since a headless Tk run is not
  available here);
* ``start_processing`` accepts a ``drizzle_group_size`` keyword with default 50
  and coerces it defensively at the backend boundary (>= 1, fallback 50);
* the backend session actually uses the coerced value: an ``incremental``
  session with ``drizzle_group_size == 3`` emits DISPLAY-ONLY previews after
  frames 3, 6, 9 and flushes the trailing partial group.

Uses the same GUI-stub import pattern as ``test_m3d_policy.py`` and a
lightweight ``__new__`` harness so no worker threads or process pools are
spawned.
"""

import importlib
import inspect
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub GUI modules to avoid Tk / real settings during import (same pattern as
# test_m3d_policy.py and test_boring_drizzle_boundary.py).
if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = []
    settings_mod = types.ModuleType("seestar.gui.settings")
    settings_mod.SettingsManager = object
    settings_mod.TILE_HEIGHT = 512
    hist_mod = types.ModuleType("seestar.gui.histogram_widget")
    hist_mod.HistogramWidget = object
    gui_pkg.settings = settings_mod
    gui_pkg.histogram_widget = hist_mod
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod
    sys.modules["seestar.gui.histogram_widget"] = hist_mod

qm = importlib.import_module("seestar.queuep.queue_manager")

MAIN_WINDOW_SRC = (ROOT / "seestar" / "gui" / "main_window.py").read_text(
    encoding="utf-8"
)
QM_SRC = (ROOT / "seestar" / "queuep" / "queue_manager.py").read_text(
    encoding="utf-8"
)


# --------------------------------------------------------------------------
# A) GUI/settings -> backend_kwargs -> start_processing (wiring)
# --------------------------------------------------------------------------


def test_gui_backend_kwargs_forwards_settings_group_size():
    # The GUI must forward the settings value into backend_kwargs so it reaches
    # start_processing (the previously missing seam).
    assert '"drizzle_group_size": self.settings.drizzle_group_size' in MAIN_WINDOW_SRC


def test_start_processing_signature_accepts_drizzle_group_size():
    sig = inspect.signature(qm.SeestarQueuedStacker.start_processing)
    assert "drizzle_group_size" in sig.parameters
    assert sig.parameters["drizzle_group_size"].default == 50


def test_start_processing_coerces_group_size_onto_session():
    # The body of start_processing must assign the (coerced) argument to the
    # session attribute before deriving the processing policy, so the value
    # received from the GUI is the value the session actually uses.
    assert "self.drizzle_group_size = _coerce_drizzle_group_size(drizzle_group_size)" in QM_SRC


# --------------------------------------------------------------------------
# B) Backend boundary coercion (>= 1, fallback 50)
# --------------------------------------------------------------------------


def test_coerce_group_size_keeps_valid_values():
    assert qm._coerce_drizzle_group_size(3) == 3
    assert qm._coerce_drizzle_group_size("3") == 3
    assert qm._coerce_drizzle_group_size(120) == 120


def test_coerce_group_size_clamps_below_one():
    # < 1 clamps to 1 (never 0 / negative), matching the existing max(1, ...)
    # cadence guard in _drizzle_group_tick / _drizzle_flush_partial_group.
    assert qm._coerce_drizzle_group_size(0) == 1
    assert qm._coerce_drizzle_group_size(-7) == 1


def test_coerce_group_size_invalid_falls_back_to_default():
    # Non-numeric / missing values fall back to the 50 default.
    assert qm._coerce_drizzle_group_size("abc") == 50
    assert qm._coerce_drizzle_group_size(None) == 50


# --------------------------------------------------------------------------
# C) Backend session uses the propagated value (incremental cadence)
# --------------------------------------------------------------------------


def _make_incremental_obj(group_size):
    obj = qm.SeestarQueuedStacker.__new__(qm.SeestarQueuedStacker)
    obj.drizzle_processing_policy = "incremental"
    obj.drizzle_group_size = group_size
    obj._drizzle_frame_count = 0
    obj._drizzle_group_index = 0
    obj.drizzle_accumulators = None
    return obj


def test_incremental_cadence_group_size_3_previews_at_3_6_9():
    obj = _make_incremental_obj(3)
    previews = []
    # Observe the DISPLAY-ONLY preview trigger without running the actual
    # accumulator preview (no accumulator / no heavy processing).
    obj._update_preview_drizzle_accumulator = lambda: previews.append(
        obj._drizzle_frame_count
    )
    for _ in range(9):
        obj._drizzle_group_tick()
    assert previews == [3, 6, 9]
    assert obj._drizzle_group_index == 3


def test_incremental_flush_partial_group_size_3():
    obj = _make_incremental_obj(3)
    previews = []
    obj._update_preview_drizzle_accumulator = lambda: previews.append(
        obj._drizzle_frame_count
    )
    for _ in range(10):
        obj._drizzle_group_tick()
    # 10 % 3 == 1 -> trailing partial group -> one extra preview at frame 10.
    obj._drizzle_flush_partial_group()
    assert previews == [3, 6, 9, 10]
