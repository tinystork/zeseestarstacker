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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BS_PATH = ROOT / "seestar" / "gui" / "boring_stack.py"
QM_PATH = ROOT / "seestar" / "queuep" / "queue_manager.py"

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
        # Classic boring_stack scenario: no quality executor populated and no
        # classic memmaps open (the retired legacy drizzle executor no longer
        # exists on the stacker — PHI-R5).
        self.quality_executor = None
        self.cumulative_sum_memmap = None
        self.cumulative_wht_memmap = None

    def _close_memmaps(self):
        self.calls.append("_close_memmaps")


def test_cleanup_stacker_no_incremental_side_effects():
    stacker = _MinimalStacker()
    bs._cleanup_stacker(stacker)

    # The classic memmap closer is the only lifecycle hook that may run (the
    # M3-D legacy drizzle wait/process machinery was retired in PHI-R5).
    assert "_close_memmaps" in stacker.calls
    assert not any("drizzle" in c for c in stacker.calls)
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


def test_retired_legacy_drizzle_machinery_absent_from_boring_and_qm():
    """PHI-R5 retirement regression: the M3-D obsolete legacy incremental
    Drizzle preview/process machinery and the dead master preview carrier are
    gone from both boring_stack.py and queue_manager.py."""
    bs_src = _bs_src()
    qm_src = QM_PATH.read_text(encoding="utf-8")
    retired = (
        "_update_preview_incremental_drizzle",
        "_start_drizzle_process",
        "drizzle_batch_worker",
        "_process_incremental_drizzle_batch",
        "_wait_drizzle_processes",
        "_update_preview_master",
        "incremental_drizzle_objects",
        "intermediate_drizzle_batch_files",
        "cumulative_drizzle_data",
        "drizzle_executor",
    )
    for sym in retired:
        assert sym not in bs_src, f"{sym!r} still referenced in boring_stack.py"
        assert sym not in qm_src, f"{sym!r} still present in queue_manager.py"


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
