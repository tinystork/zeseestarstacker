"""W-1 mode consistency: one mode == one coherent accumulation/finalization.

Regression tests for the GUI witness bug where a run ended with
``Accumulateurs memmap SUM/WHT non disponibles pour stacking classique.``
because a drizzle accumulation was finalized as a classic SUM/W stack (flag
combinations inherited from settings initialized one accumulation strategy
and finalized another).

These tests exercise, with a lightweight harness (no M16 run):

1. Drizzle-only finalization (accumulators filled, memmaps absent) -> OK,
   no memmap error.
2. Reproject&Coadd finalization (SCI/WHT provided, memmaps absent) -> OK.
3. Classic SUM/W finalization (memmaps present) -> OK.
4. No accumulation (0 images) -> clean UPSTREAM failure, clear message, never
   an arbitrary selection of another mode.
5. Inherited flag combinations -> a single explicit coherent mode, never a
   fallback.
6. The witness scenario can no longer produce the memmap error message.
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

qm = importlib.import_module("seestar.queuep.queue_manager")
from seestar.core.drizzle_core import DrizzleAccumulator  # noqa: E402


MEM_MSG = "Accumulateurs memmap SUM/WHT non disponibles pour stacking classique."


def _make_obj(tmp_path, **overrides):
    """Minimal stacker double with the attributes ``_save_final_stack`` needs."""
    obj = types.SimpleNamespace()
    obj.update_progress = lambda *a, **k: None
    obj._close_memmaps = lambda: None
    obj._wait_drizzle_processes = lambda: None
    obj._validate_drizzle_science = lambda *a, **k: None
    obj.logger = qm.logger
    obj.save_final_as_float32 = True
    obj.preserve_linear_output = True
    obj.drizzle_wht_threshold = 0.0
    obj.images_in_cumulative_stack = 0
    obj.total_exposure_seconds = 1.0
    obj.output_folder = str(tmp_path)
    obj.output_filename = "out.fit"
    obj.current_stack_header = fits.Header()
    obj.drizzle_active_session = False
    obj.is_mosaic_run = False
    obj.drizzle_mode = "Final"
    obj.drizzle_output_wcs = None
    obj.drizzle_fillval = "0.0"
    obj.reproject_between_batches = False
    obj.reproject_coadd_final = False
    obj.cumulative_sum_memmap = None
    obj.cumulative_wht_memmap = None
    obj.drizzle_accumulators = None
    obj.finalization_mode = None
    obj.batch_size = 0
    obj.reference_header_for_wcs = None
    obj.aligned_files_count = 0
    obj.processing_error = None
    for k, v in overrides.items():
        setattr(obj, k, v)
    return obj


def _filled_accumulators(shape=(4, 4)):
    accs = [DrizzleAccumulator(shape) for _ in range(3)]
    for acc in accs:
        acc._out_img[:] = 1.0
        acc._out_wht[:] = 1.0
    return accs


def _empty_accumulators(shape=(4, 4)):
    return [DrizzleAccumulator(shape) for _ in range(3)]


# ---------------------------------------------------------------------------
# mode decision (single source of truth)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags,expected",
    [
        ({"is_mosaic_run": True}, qm.FINALIZATION_MODE_MOSAIC),
        ({"is_mosaic_run": True, "drizzle_active_session": True}, qm.FINALIZATION_MODE_MOSAIC),
        ({"drizzle_active_session": True}, qm.FINALIZATION_MODE_DRIZZLE),
        # Drizzle takes precedence over stray reproject flags (the witness bug).
        (
            {"drizzle_active_session": True, "reproject_coadd_final": True},
            qm.FINALIZATION_MODE_DRIZZLE,
        ),
        (
            {"drizzle_active_session": True, "reproject_between_batches": True},
            qm.FINALIZATION_MODE_DRIZZLE,
        ),
        (
            {
                "drizzle_active_session": True,
                "reproject_coadd_final": True,
                "reproject_between_batches": True,
            },
            qm.FINALIZATION_MODE_DRIZZLE,
        ),
        ({"reproject_between_batches": True}, qm.FINALIZATION_MODE_CLASSIC_SUMW),
        ({"reproject_coadd_final": True}, qm.FINALIZATION_MODE_REPROJECT_COADD),
        (
            {"reproject_between_batches": True, "reproject_coadd_final": True},
            qm.FINALIZATION_MODE_CLASSIC_SUMW,
        ),
        ({}, qm.FINALIZATION_MODE_CLASSIC_SUMW),
    ],
)
def test_decide_finalization_mode_matrix(flags, expected):
    obj = _make_obj("/tmp", **flags)
    assert qm._decide_finalization_mode(obj) == expected


# ---------------------------------------------------------------------------
# 1. Drizzle-only finalization
# ---------------------------------------------------------------------------


def test_drizzle_only_no_memmap_error(tmp_path):
    obj = _make_obj(
        tmp_path,
        drizzle_active_session=True,
        drizzle_accumulators=_filled_accumulators(),
        # memmaps deliberately absent (None) -> must NOT be required
    )
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_drizzle_final"
    )
    assert obj.final_stacked_path is not None
    assert Path(obj.final_stacked_path).exists()
    assert obj.processing_error is None
    assert MEM_MSG not in str(obj.processing_error)


# ---------------------------------------------------------------------------
# 2. Reproject&Coadd finalization (SCI/WHT provided, memmaps absent)
# ---------------------------------------------------------------------------


def test_reproject_coadd_no_memmap_error(tmp_path):
    data = np.ones((4, 4, 3), dtype=np.float32)
    wht = np.ones((4, 4), dtype=np.float32)
    obj = _make_obj(
        tmp_path,
        reproject_coadd_final=True,
        # memmaps deliberately absent
    )
    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix="_classic_reproject",
        drizzle_final_sci_data=data,
        drizzle_final_wht_data=wht,
    )
    assert obj.final_stacked_path is not None
    assert Path(obj.final_stacked_path).exists()
    assert MEM_MSG not in str(obj.processing_error)


# ---------------------------------------------------------------------------
# 3. Classic SUM/W finalization (memmaps present)
# ---------------------------------------------------------------------------


def test_classic_sumw_with_memmaps(tmp_path):
    obj = _make_obj(
        tmp_path,
        images_in_cumulative_stack=1,
        cumulative_sum_memmap=np.ones((4, 4, 3), dtype=np.float32),
        cumulative_wht_memmap=np.ones((4, 4), dtype=np.float32),
    )
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_classic_sumw"
    )
    assert obj.final_stacked_path is not None
    assert Path(obj.final_stacked_path).exists()


# ---------------------------------------------------------------------------
# 4. No accumulation -> clean UPSTREAM failure
# ---------------------------------------------------------------------------


def test_no_accumulation_classic_fails_upstream(tmp_path):
    obj = _make_obj(
        tmp_path,
        images_in_cumulative_stack=0,
        cumulative_sum_memmap=None,
        cumulative_wht_memmap=None,
    )
    ok, msg = qm.SeestarQueuedStacker._check_finalization_ready(obj)
    assert ok is False
    assert "memmap" in msg or "aucune image" in msg


def test_no_accumulation_drizzle_fails_upstream(tmp_path):
    obj = _make_obj(
        tmp_path,
        drizzle_active_session=True,
        drizzle_accumulators=_empty_accumulators(),
    )
    ok, msg = qm.SeestarQueuedStacker._check_finalization_ready(obj)
    assert ok is False
    assert "aucune image" in msg
    assert "memmap" not in msg


def test_no_accumulation_classic_empty_memmaps_fails_upstream(tmp_path):
    obj = _make_obj(
        tmp_path,
        images_in_cumulative_stack=0,
        cumulative_sum_memmap=np.ones((4, 4, 3), dtype=np.float32),
        cumulative_wht_memmap=np.ones((4, 4), dtype=np.float32),
    )
    ok, msg = qm.SeestarQueuedStacker._check_finalization_ready(obj)
    assert ok is False
    assert "aucune image" in msg


# ---------------------------------------------------------------------------
# 5 + 6. Inherited flag combination / witness scenario
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_flags",
    [
        {"reproject_coadd_final": True},
        {"reproject_between_batches": True},
        {"reproject_coadd_final": True, "reproject_between_batches": True},
    ],
)
def test_witness_scenario_finalizes_as_drizzle(tmp_path, extra_flags):
    # The witness: drizzle accumulation initialized (accumulators filled) but a
    # stray inherited reproject flag previously routed finalization into the
    # SUM/W branch -> "memmap non disponibles".  Now it must finalize as drizzle.
    obj = _make_obj(
        tmp_path,
        drizzle_active_session=True,
        drizzle_accumulators=_filled_accumulators(),
        **extra_flags,
    )
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_drizzle_final"
    )
    assert obj.final_stacked_path is not None
    assert Path(obj.final_stacked_path).exists()
    assert obj.processing_error is None
    assert MEM_MSG not in str(obj.processing_error)


def test_witness_scenario_empty_drizzle_fails_cleanly_not_memmap(tmp_path):
    # Same inherited flags but NO accumulated image: the failure must be a clear
    # drizzle "no accumulation" message, never the classic memmap error.
    obj = _make_obj(
        tmp_path,
        drizzle_active_session=True,
        reproject_coadd_final=True,
        drizzle_accumulators=_empty_accumulators(),
    )
    ok, msg = qm.SeestarQueuedStacker._check_finalization_ready(obj)
    assert ok is False
    assert MEM_MSG not in msg
    assert "Drizzle" in msg or "drizzle" in msg.lower()


def test_reproject_coadd_missing_sci_wht_raises_clear_error(tmp_path):
    # Reproject&Coadd declared but SCI/WHT not provided: explicit contract
    # violation, not a fallback into SUM/W.
    obj = _make_obj(tmp_path, reproject_coadd_final=True)
    with pytest.raises(ValueError) as excinfo:
        qm.SeestarQueuedStacker._save_final_stack(
            obj, output_filename_suffix="_classic_reproject"
        )
    assert "SCI/WHT" in str(excinfo.value)


def test_explicit_mode_consumed_by_save(tmp_path):
    # The pipeline transmits the mode explicitly at initialization; _save_final_stack
    # must consume it even when the flags alone would be ambiguous.
    obj = _make_obj(
        tmp_path,
        drizzle_active_session=True,
        drizzle_accumulators=_filled_accumulators(),
        finalization_mode=qm.FINALIZATION_MODE_DRIZZLE,
    )
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_drizzle_final"
    )
    assert obj.final_stacked_path is not None
    assert Path(obj.final_stacked_path).exists()
