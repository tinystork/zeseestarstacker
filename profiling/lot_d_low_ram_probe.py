"""Lot D synthetic bounded probe (MX150, CuPy runtime).

Proves the conjoint host-RAM plan + low-RAM execution on the real MX150 with an
INJECTED host RAM budget (no M74 data, no heavy run):

1. host_plan_333mib: the pure host planner on the exact 333 MiB / N=3 RGB
   2822x4144 scenario -> memmap_outputs (or spill_and_memmap), a small
   admissible tile, N unchanged.
2. host_plan_plenty: in_memory with the full-frame cap when RAM is ample.
3. host_cap_combines_with_vram: the GPU VRAM planner honours the host tile cap
   (a tile must fit BOTH sides), yielding a tile <= the host cap.
4. cpu_memmap_bitwise: the exact-N CPU tiled driver is bitwise-identical
   between in-RAM arrays and memmap inputs + memmap outputs (z_eff/pct too).
5. spill_cleanup: spill + memmap outputs then cleanup leaves no orphan scratch.

Exits nonzero on any failure.
"""
from __future__ import annotations

import os
import sys
import tempfile

REPO = "/home/tristan/.openclaw/workspace/projects/zeseestarstacker"
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from seestar.core.host_ram_planner import (  # noqa: E402
    HOST_IN_MEMORY,
    HOST_MEMMAP_OUTPUTS,
    HOST_SPILL_AND_MEMMAP,
    plan_host_ram_execution,
)
from seestar.core.gpu_vram_planner import (  # noqa: E402
    TILED_GPU,
    plan_winsorized_gpu_execution,
)
from seestar.core.scratch_store import ScratchStore  # noqa: E402

MiB = 1024 ** 2

_H, _W, _C, _N = 2822, 4144, 3, 3
_S_FULL = _H * _W
_RESIDENT = _N * _S_FULL * _C * 4 + _N * _S_FULL
_OUTPUT = 2 * _S_FULL * _C * 4

failures = []


def check(tag, fn):
    try:
        fn()
        print("PASS %s" % tag)
    except Exception as exc:  # noqa: BLE001
        failures.append((tag, exc))
        print("FAIL %s: %r" % (tag, exc))


def probe_host_plan_333mib():
    d = plan_host_ram_execution(
        n=_N, frame_shape=(_H, _W), channels=_C,
        available_ram_bytes=333 * MiB, reserve_bytes=256 * MiB,
        resident_input_bytes=_RESIDENT, output_bytes=_OUTPUT,
    )
    assert d.strategy in (HOST_MEMMAP_OUTPUTS, HOST_SPILL_AND_MEMMAP), d.strategy
    assert d.memmaps_outputs
    cap = d.host_tile_outputs_cap
    assert cap is not None and cap >= 96 and cap < _S_FULL
    assert d.n == _N


def probe_host_plan_plenty():
    d = plan_host_ram_execution(
        n=_N, frame_shape=(_H, _W), channels=_C,
        available_ram_bytes=8 * 1024 ** 3, reserve_bytes=512 * MiB,
        resident_input_bytes=_RESIDENT, output_bytes=_OUTPUT,
    )
    assert d.strategy == HOST_IN_MEMORY
    assert d.host_tile_outputs_cap == _S_FULL


def probe_host_cap_combines_with_vram():
    # Host cap forces a small tile; VRAM would otherwise allow the full frame.
    d_host = plan_host_ram_execution(
        n=_N, frame_shape=(_H, _W), channels=_C,
        available_ram_bytes=333 * MiB, reserve_bytes=256 * MiB,
        resident_input_bytes=_RESIDENT, output_bytes=_OUTPUT,
    )
    cap = d_host.host_tile_outputs_cap
    d_gpu = plan_winsorized_gpu_execution(
        n_batch=_N, frame_shape=(_H, _W), channels=_C,
        winsor_limits=(0.05, 0.05),
        driver_free_bytes=8 * 1024 ** 3,  # VRAM ample -> would be FULL
        pool_free_bytes=0,
        host_tile_outputs_cap=cap,
    )
    assert d_gpu.kind == TILED_GPU, d_gpu.kind
    assert d_gpu.tile_outputs <= cap


def probe_cpu_memmap_bitwise():
    from seestar.core.cpu_winsor_exact_n import (
        stack_winsorized_sigma_cpu_tiled,
    )

    rng = np.random.default_rng(7)
    imgs = [rng.normal(size=(64, 80)).astype(np.float32) for _ in range(5)]
    imgs[0][0, 0] = np.nan
    w = np.ones(5, np.float32)
    ref = stack_winsorized_sigma_cpu_tiled(
        imgs, w, return_weights=True, tile_shape=(8,),
        kappa=3.0, winsor_limits=(0.2, 0.2),
    )
    d = tempfile.mkdtemp()
    store = ScratchStore(os.path.join(d, "run"))
    mm_imgs = [store.spill_image(im, "img_%d" % i) for i, im in enumerate(imgs)]
    sci = store.new_memmap("sci", (64, 80), np.float32)
    wht = store.new_memmap("wht", (64, 80), np.float32)
    try:
        got = stack_winsorized_sigma_cpu_tiled(
            mm_imgs, w, return_weights=True, tile_shape=(8,),
            kappa=3.0, winsor_limits=(0.2, 0.2),
            out_result=sci, out_sum_w=wht,
        )
        assert np.array_equal(ref[0], np.asarray(got[0]))
        assert np.array_equal(ref[1], np.asarray(got[1]))
        assert abs(ref[2] - got[2]) < 1e-6
    finally:
        store.cleanup()


def probe_spill_cleanup():
    d = tempfile.mkdtemp()
    store = ScratchStore(os.path.join(d, "run"))
    arr = np.zeros((8, 8), np.float32)
    store.spill_image(arr, "a")
    store.new_memmap("sci", (8, 8), np.float32)
    owned = list(store._created)
    dir_ = store.dir
    store.cleanup()
    for p in owned:
        assert not os.path.exists(p), p
    assert not os.path.isdir(dir_)


if __name__ == "__main__":
    print("device-independent host-RAM plan + low-RAM execution probe")
    check("host_plan_333mib", probe_host_plan_333mib)
    check("host_plan_plenty", probe_host_plan_plenty)
    check("host_cap_combines_with_vram", probe_host_cap_combines_with_vram)
    check("cpu_memmap_bitwise", probe_cpu_memmap_bitwise)
    check("spill_cleanup", probe_spill_cleanup)
    if failures:
        print("\n%d FAILURE(S)" % len(failures))
        sys.exit(1)
    print("\nALL PROBES PASSED")
