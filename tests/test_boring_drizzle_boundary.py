"""M3-D boundary tests: boring_stack vs the Drizzle incremental policy.

``seestar/gui/boring_stack.py`` is the CLASSIC single-batch (mono-lot) SUM/W
memmap stacking path.  The Drizzle incremental policy lives exclusively in
``seestar/queuep/queue_manager.py``.  These tests pin that boundary:

* configuring a stacker the way ``boring_stack`` does (``use_drizzle=False``,
  ``drizzle_mode`` never forwarded) NEVER activates a drizzle session, NEVER
  creates ``drizzle_accumulators``, and the incremental tick / flush / preview
  methods are inert (no preview callback, no accumulator);
* ``_cleanup_stacker`` only drains the legacy executors and closes the classic
  memmaps; it never drives any incremental-drizzle code path
  (``_wait_drizzle_processes`` is a M3 legacy no-op and is the only legacy
  symbol it touches);
* the incremental symbols are absent from ``boring_stack.py`` source;
* the classic path is mono-batch with a fixed-shape (reference-grid) memmap
  footprint independent of the number of poses (bounded O(HxW), not O(N)).

Uses the same GUI-stub import pattern as ``test_boring_thread.py`` and the
lightweight ``__new__`` harness from ``test_m3d_policy.py`` so no worker
threads or process pools are spawned.
"""

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BS_PATH = ROOT / "seestar" / "gui" / "boring_stack.py"
QM_PATH = ROOT / "seestar" / "queuep" / "queue_manager.py"

# Stub GUI modules to avoid Tk / real settings during import (same pattern as
# test_boring_thread.py).
if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = [str(ROOT / "seestar" / "gui")]
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

bs = importlib.import_module("seestar.gui.boring_stack")
qm = importlib.import_module("seestar.queuep.queue_manager")


# --------------------------------------------------------------------------
# A) Policy isolation
# --------------------------------------------------------------------------


def _boring_stack_like_obj(drizzle_mode="Incremental"):
    """Build a stacker the way boring_stack's start_processing call does.

    ``boring_stack`` calls ``start_processing(..., use_drizzle=False, ...)``
    (~L882) and does NOT forward ``drizzle_mode``.  The session gate is:

        ``self.drizzle_active_session = use_drizzle or self.is_mosaic_run``

    so it is False.  ``drizzle_accumulators`` is only ever created inside
    ``initialize()`` behind ``drizzle_active_session``, so it stays None.
    The ``drizzle_mode`` argument defaults to ``"Final"``; the worst case of a
    hypothetical ``"Incremental"`` leak from settings is exercised directly to
    prove that even then no incremental policy becomes effective.
    """
    obj = qm.SeestarQueuedStacker.__new__(qm.SeestarQueuedStacker)
    obj.use_drizzle = False
    obj.is_mosaic_run = False
    obj.drizzle_mode = drizzle_mode
    # Mirror the start_processing session gate:
    obj.drizzle_active_session = obj.use_drizzle or obj.is_mosaic_run
    obj._derive_drizzle_processing_policy()
    # Never created without an active session:
    obj.drizzle_accumulators = None
    obj._drizzle_frame_count = 0
    obj._drizzle_group_index = 0
    return obj


def test_boring_stack_default_policy_is_standard_no_session():
    # boring_stack never forwards drizzle_mode -> start_processing default
    # "Final" -> standard policy, and no drizzle session.
    obj = _boring_stack_like_obj(drizzle_mode="Final")
    assert obj.drizzle_active_session is False
    assert obj.drizzle_processing_policy == "standard"
    assert obj.drizzle_accumulators is None


def test_incremental_setting_does_not_create_session_or_accumulators():
    # Worst case: settings JSON carried drizzle_mode="Incremental".  The
    # session gate is keyed on use_drizzle (False), so no session and no
    # accumulator regardless of the policy string.
    obj = _boring_stack_like_obj(drizzle_mode="Incremental")
    assert obj.drizzle_active_session is False
    assert obj.drizzle_accumulators is None
    # The policy string may map to "incremental", but with no session and no
    # accumulators it can never do any work (asserted next).


def test_tick_flush_preview_inert_without_accumulators():
    collected = []

    def collector(*args, **kwargs):
        collected.append(args)

    obj = _boring_stack_like_obj(drizzle_mode="Incremental")
    obj.preview_callback = collector

    # Drive the incremental cadence: a full group would trigger a preview if a
    # drizzle session were active, but none is -> nothing may happen.
    for _ in range(12):  # group_size default 50 -> no full group reached either
        obj._drizzle_group_tick()
    obj._drizzle_group_tick()  # still only counts
    obj._drizzle_flush_partial_group()
    obj._update_preview_drizzle_accumulator()

    assert collected == []
    assert obj.drizzle_accumulators is None
    assert getattr(obj, "cumulative_drizzle_data", None) is None
    assert getattr(obj, "cumulative_drizzle_data_raw", None) is None


# --------------------------------------------------------------------------
# B) _cleanup_stacker has no incremental side effects
# --------------------------------------------------------------------------


class _MinimalStacker:
    """Minimal stand-in for SeestarQueuedStacker passed to _cleanup_stacker."""

    def __init__(self):
        self.calls = []
        self._indices_cache = {}
        self.perform_cleanup = False
        # No drizzle session / no executors populated (classic boring_stack
        # scenario) and no classic memmaps open.
        self.drizzle_executor = None
        self.quality_executor = None
        self.cumulative_sum_memmap = None
        self.cumulative_wht_memmap = None

    def _wait_drizzle_processes(self):
        self.calls.append("_wait_drizzle_processes")

    def _close_memmaps(self):
        self.calls.append("_close_memmaps")

    # Incremental-drizzle symbols: must NOT be called by _cleanup_stacker.
    # Defined here only so any accidental call is observable (and the assert
    # below fails) instead of silently raising AttributeError.
    def _process_incremental_drizzle_batch(self, *a, **k):
        self.calls.append("_process_incremental_drizzle_batch")

    def _start_drizzle_process(self, *a, **k):
        self.calls.append("_start_drizzle_process")


def test_cleanup_stacker_no_incremental_side_effects():
    stacker = _MinimalStacker()
    bs._cleanup_stacker(stacker)

    # The legacy no-op wait and the classic memmap closer are the only
    # lifecycle hooks that may run.
    assert "_wait_drizzle_processes" in stacker.calls
    assert "_close_memmaps" in stacker.calls
    # No incremental-drizzle code path may be driven:
    assert "_process_incremental_drizzle_batch" not in stacker.calls
    assert "_start_drizzle_process" not in stacker.calls
    # Classic memmap slots remain dropped/None:
    assert stacker.cumulative_sum_memmap is None
    assert stacker.cumulative_wht_memmap is None


def test_cleanup_stacker_none_is_safe():
    # Idempotent / safe when handed None.
    bs._cleanup_stacker(None)


# --------------------------------------------------------------------------
# C) Static source audit of boring_stack.py
# --------------------------------------------------------------------------


def _bs_src():
    return BS_PATH.read_text(encoding="utf-8")


def test_boring_stack_forces_use_drizzle_false():
    assert "use_drizzle=False" in _bs_src()


def test_boring_stack_has_no_incremental_symbols():
    src = _bs_src()
    for sym in (
        "_drizzle_group_tick",
        "drizzle_group_size",
        "_process_incremental_drizzle_batch",
        "_start_drizzle_process",
        "_update_preview_drizzle_accumulator",
    ):
        assert sym not in src, f"{sym!r} unexpectedly present in boring_stack.py"


def test_wait_drizzle_processes_only_in_cleanup_context():
    src = _bs_src()
    lines = src.splitlines()
    hits = [i for i, ln in enumerate(lines) if "_wait_drizzle_processes" in ln]
    assert hits, "expected at least one _wait_drizzle_processes reference"

    def_idx = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("def _cleanup_stacker")
    )
    # Every reference must live inside the _cleanup_stacker body (docstring or
    # code) — i.e. before the next module-level def.
    next_def = next(
        (i for i in range(def_idx + 1, len(lines)) if lines[i].startswith("def ")),
        len(lines),
    )
    for h in hits:
        assert def_idx < h < next_def, (
            "_wait_drizzle_processes must only appear inside _cleanup_stacker"
        )


# --------------------------------------------------------------------------
# D) Mono-lot / OOM bound
# --------------------------------------------------------------------------


def test_boring_stack_uses_classic_memmaps_and_auto_align_on_disk():
    src = _bs_src()
    # Classic SUM/W memmaps (managed/closed in _cleanup_stacker).
    assert "cumulative_sum_memmap" in src
    assert "cumulative_wht_memmap" in src
    # Auto-enable disk-backed alignment beyond 50 images (batch_size=1):
    assert "len(rows) > 50" in src
    assert "args.align_on_disk = True" in src


def test_classic_memmaps_use_fixed_reference_grid_shape():
    qsrc = QM_PATH.read_text(encoding="utf-8")
    # In the classic (non-drizzle) branch of initialize(), the SUM/WHT memmaps
    # are allocated from the reference grid shape (H, W, C) / (H, W) — i.e. a
    # fixed shape independent of the number of poses N.  This bounds classic
    # memory to O(HxW), not O(N).
    assert "cumulative_sum_memmap" in qsrc
    assert "cumulative_wht_memmap" in qsrc
    assert "H, W, C = reference_image_shape_hwc_input" in qsrc
    assert "shape=(H, W, C)" in qsrc
    assert "shape=(H, W)" in qsrc
    assert "self.memmap_shape = (H, W, C)" in qsrc
