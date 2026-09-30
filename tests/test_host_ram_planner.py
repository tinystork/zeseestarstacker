"""Lot D — pure host-RAM planner + scratch-store tests (no device).

The conjoint host planner (``seestar.core.host_ram_planner``) decides
``in_memory | spill_inputs | memmap_outputs | spill_and_memmap | refuse`` and
returns the INCLUSIVE host tile cap that a tiled geometry must respect so its
host working set fits alongside the resident inputs and the full-frame
SCI/WHT outputs.  The scratch store (``seestar.core.scratch_store``) owns a
run-scoped directory under the output folder (never /tmp), refuses an
insufficient volume BEFORE writing, spills aligned frames to read-only memmap,
and cleans up only its own artifacts.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from seestar.core.host_ram_planner import (
    HOST_IN_MEMORY,
    HOST_MEMMAP_OUTPUTS,
    HOST_REFUSE,
    HOST_SPILL_AND_MEMMAP,
    HOST_SPILL_INPUTS,
    REASON_HOST_BUDGET_NEGATIVE,
    REASON_HOST_NO_VALID_TILE,
    HostRamDecision,
    plan_host_ram_execution,
)
from seestar.core.scratch_store import (
    SCRATCH_SUBDIR,
    ScratchSpaceRefused,
    ScratchStore,
)

MiB = 1024 ** 2

# The exact failing run geometry: N=3, RGB 2822x4144, ~333 MiB available.
_H, _W, _C, _N = 2822, 4144, 3, 3
_S_FULL = _H * _W
_RESIDENT = _N * _S_FULL * _C * 4 + _N * _S_FULL  # frames + masks
_OUTPUT = 2 * _S_FULL * _C * 4


# ---------------------------------------------------------------------------
# A. planner vocabulary + structural exact-N
# ---------------------------------------------------------------------------

def test_decision_vocabulary_and_no_n_split_field():
    d = plan_host_ram_execution(
        n=_N, frame_shape=(_H, _W), channels=_C,
        available_ram_bytes=8 * 1024 ** 3, reserve_bytes=512 * MiB,
        resident_input_bytes=_RESIDENT, output_bytes=_OUTPUT,
    )
    assert d.strategy in (
        HOST_IN_MEMORY, HOST_SPILL_INPUTS, HOST_MEMMAP_OUTPUTS,
        HOST_SPILL_AND_MEMMAP, HOST_REFUSE,
    )
    assert d.n == _N  # frozen N
    for field in ("n_tile", "n_per_tile", "reduced_n", "split_n"):
        assert not hasattr(d, field), field


def test_in_memory_when_plenty_of_ram():
    d = plan_host_ram_execution(
        n=_N, frame_shape=(_H, _W), channels=_C,
        available_ram_bytes=8 * 1024 ** 3, reserve_bytes=512 * MiB,
        resident_input_bytes=_RESIDENT, output_bytes=_OUTPUT,
    )
    assert d.strategy == HOST_IN_MEMORY
    # Plenty of RAM: the whole frame fits as a single tile -> cap == full frame.
    assert d.host_tile_outputs_cap == _S_FULL
    assert d.spill_input_bytes == 0 and d.output_memmap_bytes == 0


def test_333mib_real_base_reserve_memmapped_outputs():
    """F1: the EXACT 333 MiB scenario with the REAL named base reserve
    (``recommended_reserve_bytes(available, 0, 1)`` — output serialization is
    NOT double-counted because the host planner models SCI/WHT explicitly as
    float32).  Strategy = memmap_outputs, small admissible tile, N unchanged."""
    from seestar.core.cpu_memory_planner import recommended_reserve_bytes

    available = 333 * MiB
    reserve = recommended_reserve_bytes(available, 0, 1)
    # Base reserve must NOT include the 2-frame output serialization.
    assert reserve == 256 * MiB
    d = plan_host_ram_execution(
        n=_N, frame_shape=(_H, _W), channels=_C,
        available_ram_bytes=available, reserve_bytes=reserve,
        resident_input_bytes=_RESIDENT, output_bytes=_OUTPUT,
        freed_input_bytes=0,
    )
    # With the REAL base reserve, outputs cannot fit in RAM -> memmap_outputs
    # (NOT spill_and_memmap: freed=0, spill gains nothing).
    assert d.strategy == HOST_MEMMAP_OUTPUTS
    assert d.memmaps_outputs is True
    assert d.output_memmap_bytes == _OUTPUT
    cap = d.host_tile_outputs_cap
    assert cap is not None and cap >= 96 and cap < _S_FULL
    assert d.n == _N  # N unchanged


def test_333mib_full_reserve_would_double_count():
    """F1/F5: ``recommended_reserve_bytes(available, frame, 1)`` embeds
    ``2 x frame`` output serialization; using it AS the reserve AND separately
    counting output would double-count.  The host planner therefore uses the
    BASE reserve (frame_bytes=0)."""
    from seestar.core.cpu_memory_planner import recommended_reserve_bytes

    available = 333 * MiB
    frame = _S_FULL * _C * 4
    full = recommended_reserve_bytes(available, frame, 1)
    base = recommended_reserve_bytes(available, 0, 1)
    assert full == base + 2 * frame  # the output-serialization component
    assert full > available  # exceeds available -> would force spill_and_memmap


def test_output_cost_always_float32():
    """F5: SCI/WHT are float32 regardless of INPUT dtype (uint16/float64 inputs
    must not under/over-estimate the output cost)."""
    for isz in (2, 8):
        d = plan_host_ram_execution(
            n=_N, frame_shape=(_H, _W), channels=_C,
            dtype_itemsize=isz,
            available_ram_bytes=8 * 1024 ** 3, reserve_bytes=512 * MiB,
            resident_input_bytes=_N * _S_FULL * _C * isz + _N * _S_FULL,
            output_bytes=None,  # default must be float32
        )
        # Output default = 2 * H * W * C * 4 (float32), independent of isz.
        assert d.output_bytes == _OUTPUT


def test_refuse_when_even_minimum_tile_cannot_fit():
    d = plan_host_ram_execution(
        n=3, frame_shape=(10, 10), channels=1,
        available_ram_bytes=1000, reserve_bytes=0,
        resident_input_bytes=0, output_bytes=800,
        min_tile_out=96,
    )
    assert d.strategy == HOST_REFUSE
    assert d.reason in (REASON_HOST_BUDGET_NEGATIVE, REASON_HOST_NO_VALID_TILE)
    assert d.n == 3  # still never reduces N


def test_negative_available_is_budget_negative():
    # reserve exceeds available -> in_memory budget <= 0; spill budget may still
    # be positive (resident freed), so the refusal reason is host_no_valid_tile
    # when resident makes room, else host_budget_negative.
    d = plan_host_ram_execution(
        n=_N, frame_shape=(_H, _W), channels=_C,
        available_ram_bytes=100 * MiB, reserve_bytes=256 * MiB,
        resident_input_bytes=0, output_bytes=_OUTPUT,
    )
    assert d.strategy == HOST_REFUSE
    assert d.reason == REASON_HOST_BUDGET_NEGATIVE


def test_spill_requires_proven_freed_bytes():
    # F2 honest model: with freed_input_bytes=0 (the wiring's default — it
    # cannot prove the batch released its ndarray references), spill provides
    # NO budget benefit, so it is NEVER selected even when resident dominates.
    d = plan_host_ram_execution(
        n=3, frame_shape=(2822, 4144), channels=3,
        available_ram_bytes=256 * MiB, reserve_bytes=256 * MiB,
        resident_input_bytes=_RESIDENT, output_bytes=_OUTPUT,
        freed_input_bytes=0,
    )
    assert d.strategy not in (HOST_SPILL_INPUTS, HOST_SPILL_AND_MEMMAP)
    assert d.spills_inputs is False
    assert d.spill_input_bytes == 0


def test_spill_selected_only_when_freed_proven():
    # When the caller PROVES the resident inputs were freed (freed_input_bytes
    # > 0), spill_inputs becomes viable: budget_spilled grows by exactly the
    # proven amount, never by the full unproven resident.
    freed = _RESIDENT  # caller proved the full resident is collectable
    d = plan_host_ram_execution(
        n=3, frame_shape=(2822, 4144), channels=3,
        available_ram_bytes=256 * MiB, reserve_bytes=256 * MiB,
        resident_input_bytes=_RESIDENT, output_bytes=_OUTPUT,
        freed_input_bytes=freed,
    )
    assert d.strategy in (HOST_SPILL_INPUTS, HOST_SPILL_AND_MEMMAP)
    assert d.spills_inputs is True
    assert d.spill_input_bytes == freed


def test_spill_budget_uses_freed_not_resident():
    # The spilled budget must reflect the PROVEN freed bytes, not the resident
    # total: budget_spilled == available - reserve + freed_input_bytes.
    d = plan_host_ram_execution(
        n=3, frame_shape=(2822, 4144), channels=3,
        available_ram_bytes=256 * MiB, reserve_bytes=256 * MiB,
        resident_input_bytes=_RESIDENT, output_bytes=_OUTPUT,
        freed_input_bytes=100 * MiB,
    )
    assert d.details["budget_spilled"] == 100 * MiB
    assert d.details["budget_in_memory"] == 0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        plan_host_ram_execution(
            n=0, frame_shape=(10, 10), channels=1,
            available_ram_bytes=100, reserve_bytes=0,
        )
    with pytest.raises(ValueError):
        plan_host_ram_execution(
            n=1, frame_shape=(0, 10), channels=1,
            available_ram_bytes=100, reserve_bytes=0,
        )


# ---------------------------------------------------------------------------
# B. scratch store
# ---------------------------------------------------------------------------

def test_scratch_under_output_folder_never_tmp():
    d = tempfile.mkdtemp()
    out = os.path.join(d, "run")
    os.makedirs(out)
    store = ScratchStore(out)
    store.ensure_dir()
    assert store.dir == os.path.join(os.path.abspath(out), SCRATCH_SUBDIR)
    assert not store.dir.startswith("/tmp")
    assert os.path.isdir(store.dir)


def test_spill_preserves_values_dtype_geometry_readonly():
    d = tempfile.mkdtemp()
    store = ScratchStore(os.path.join(d, "run"))
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(8, 12, 3)).astype(np.float32)
    mm = store.spill_image(arr, "img_0")
    assert mm.dtype == arr.dtype and mm.shape == arr.shape
    assert not mm.flags.writeable
    assert np.array_equal(np.asarray(mm), arr)
    store.cleanup()


def test_check_space_refuses_before_write():
    d = tempfile.mkdtemp()
    store = ScratchStore(os.path.join(d, "run"))
    store.ensure_dir()
    with pytest.raises(ScratchSpaceRefused):
        store.check_space(10 ** 18)
    # Nothing was written.
    assert store._created == []


def test_cleanup_removes_only_owned_files_and_dir():
    d = tempfile.mkdtemp()
    store = ScratchStore(os.path.join(d, "run"))
    arr = np.zeros((4, 4), dtype=np.float32)
    mm1 = store.spill_image(arr, "a")
    mm2 = store.spill_image(arr, "b")
    owned = list(store._created)
    # A foreign file in the dir must survive cleanup.
    foreign = os.path.join(store.dir, "foreign.txt")
    with open(foreign, "w") as f:
        f.write("keep me")
    store.cleanup()
    for p in owned:
        assert not os.path.exists(p), p
    assert os.path.exists(foreign)
    assert os.path.isdir(store.dir)  # not removed while foreign file present


def test_cleanup_idempotent():
    d = tempfile.mkdtemp()
    store = ScratchStore(os.path.join(d, "run"))
    store.spill_image(np.zeros((3, 3), np.float32), "x")
    store.cleanup()
    store.cleanup()  # second cleanup must not raise


def test_cleanup_on_exception_no_orphan_scratch():
    """An exception mid-reduction must leave no orphan scratch file."""
    d = tempfile.mkdtemp()
    out = os.path.join(d, "run")
    store = ScratchStore(out)
    store.spill_image(np.zeros((3, 3), np.float32), "a")
    store.new_memmap("sci", (3, 3), np.float32)
    owned = list(store._created)
    scratch_dir = store.dir
    try:
        raise RuntimeError("simulated reduction failure")
    except RuntimeError:
        store.cleanup()  # exception path
    for p in owned:
        assert not os.path.exists(p), p
    assert not os.path.isdir(scratch_dir)  # empty -> removed


def test_cleanup_on_cancel_no_orphan_scratch():
    """A cancelled run (KeyboardInterrupt-equivalent) leaves no scratch."""
    d = tempfile.mkdtemp()
    store = ScratchStore(os.path.join(d, "run"))
    store.spill_image(np.zeros((4, 4), np.float32), "img")
    owned = list(store._created)
    store.cleanup()  # cancel/abort path
    for p in owned:
        assert not os.path.exists(p), p


# ---------------------------------------------------------------------------
# C. exact-N CPU parity: memmap inputs/outputs vs in-RAM arrays
# ---------------------------------------------------------------------------

def _small_batch(shape=(64, 80), n=5, seed=42):
    rng = np.random.default_rng(seed)
    imgs = [rng.normal(size=shape).astype(np.float32) for _ in range(n)]
    imgs[0][0, 0] = np.nan
    return imgs, np.ones(n, dtype=np.float32)


def test_cpu_memmap_inputs_bitwise_equal_to_arrays():
    from seestar.core.cpu_winsor_exact_n import (
        stack_winsorized_sigma_cpu_tiled,
    )

    imgs, w = _small_batch()
    ref = stack_winsorized_sigma_cpu_tiled(
        imgs, w, return_weights=True, tile_shape=(8,),
        kappa=3.0, winsor_limits=(0.2, 0.2),
    )
    d = tempfile.mkdtemp()
    store = ScratchStore(os.path.join(d, "run"))
    mm_imgs = [store.spill_image(im, "img_%d" % i) for i, im in enumerate(imgs)]
    try:
        got = stack_winsorized_sigma_cpu_tiled(
            mm_imgs, w, return_weights=True, tile_shape=(8,),
            kappa=3.0, winsor_limits=(0.2, 0.2),
        )
    finally:
        store.cleanup()
    assert np.array_equal(ref[0], got[0])  # result
    assert np.array_equal(ref[1], got[1])  # sum_w
    assert abs(ref[2] - got[2]) < 1e-6  # rejected_pct


def test_cpu_memmap_outputs_bitwise_equal_to_arrays():
    from seestar.core.cpu_winsor_exact_n import (
        stack_winsorized_sigma_cpu_tiled,
    )

    imgs, w = _small_batch()
    ref = stack_winsorized_sigma_cpu_tiled(
        imgs, w, return_weights=True, tile_shape=(8,),
        kappa=3.0, winsor_limits=(0.2, 0.2),
    )
    d = tempfile.mkdtemp()
    store = ScratchStore(os.path.join(d, "run"))
    store.ensure_dir()
    sci = store.new_memmap("sci", (64, 80), np.float32)
    wht = store.new_memmap("wht", (64, 80), np.float32)
    try:
        got = stack_winsorized_sigma_cpu_tiled(
            imgs, w, return_weights=True, tile_shape=(8,),
            kappa=3.0, winsor_limits=(0.2, 0.2),
            out_result=sci, out_sum_w=wht,
        )
        assert hasattr(got[0], "_mmap") and hasattr(got[1], "_mmap")
        assert np.array_equal(ref[0], np.asarray(got[0]))
        assert np.array_equal(ref[1], np.asarray(got[1]))
        assert abs(ref[2] - got[2]) < 1e-6
    finally:
        store.cleanup()
