"""Lot D rework-1 — queue-manager wiring tests for the conjoint host-RAM plan.

No device, no heavy run: constructs a bare ``SeestarQueuedStacker`` via
``__new__`` and drives the host-plan seams directly, proving:

* F3: a HOST_REFUSE or scratch/quota failure is a CONTROLLED refusal
  (``CpuWinsorMemoryRefused``) — the legacy (greedier) CPU fallback is NEVER
  invoked, and no partial scratch file survives;
* F4: the CPU fallback REUSES the already-resolved host plan + memmap outputs
  via ``_winsorized_host_ctx`` (no double plan, no orphan store);
* F5/F1: ``_winsorized_host_ram_plan`` sizes SCI/WHT as float32 and uses the
  BASE reserve (no output double-count).
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import pytest

from seestar.queuep.queue_manager import SeestarQueuedStacker
from seestar.core.cpu_winsor_exact_n import CpuWinsorMemoryRefused
from seestar.core.host_ram_planner import (
    HOST_MEMMAP_OUTPUTS,
    HOST_REFUSE,
)
from seestar.core.scratch_store import ScratchSpaceRefused

MiB = 1024 ** 2


def _bare_stacker(available=None, output_folder=None):
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.logger = logging.getLogger("zsss.test.lowram")
    o._cpu_available_ram_bytes_override = available
    o._cpu_process_rss_bytes_override = None
    o.output_folder = output_folder
    o._winsorized_scratch_stores = []
    o._winsorized_host_ctx = None
    o._cpu_mem_policy_preflight = None
    o.winsor_limits = (0.2, 0.2)
    o.request_gpu = False
    o.stacking_mode = "winsorized-sigma-clip"
    o.stack_reject_algo = "winsorized_sigma_clip"
    # ``effective_backend`` is a read-only property backed by the cached
    # acceleration policy; inject a CPU policy directly (bare __new__ object).
    from types import SimpleNamespace

    o._acceleration_policy = SimpleNamespace(backend="cpu")
    return o


def _small_images(n=3, shape=(64, 80)):
    rng = np.random.default_rng(0)
    return [rng.normal(size=shape).astype(np.float32) for _ in range(n)]


def test_winsorized_host_ram_plan_float32_output_and_base_reserve():
    o = _bare_stacker(available=8 * 1024 ** 3, output_folder=None)
    imgs = _small_images()
    decision, available, rss, reserve, resident, output = (
        o._winsorized_host_ram_plan(imgs, None, len(imgs), (64, 80), 1)
    )
    # F5: output = 2 * H * W * C * 4 (float32), not input isz.
    assert output == 2 * 64 * 80 * 1 * 4
    # F1: base reserve (no output double-count) = fixed min 256 MiB (2% of
    # 8 GiB = ~164 MiB < 256 MiB, so fixed min wins).
    assert reserve == 256 * MiB
    assert decision.n == 3


def test_host_refusal_raises_controlled_no_legacy_fallback():
    # Tiny available RAM + base reserve -> HOST_REFUSE -> controlled refusal.
    o = _bare_stacker(available=10 * MiB, output_folder=None)
    imgs = _small_images(shape=(2822, 4144))  # large resident/output
    imgs = [np.zeros((64, 80), np.float32) for _ in range(3)]
    called = {"cpu": False}

    def fn_cpu(*a, **k):
        called["cpu"] = True
        return (np.zeros((64, 80), np.float32), 0.0)

    # Force refusal: available < reserve -> budget negative -> refuse.
    o._cpu_available_ram_bytes_override = 10 * MiB
    decision, _a, _r, _res, _resid, _out = o._winsorized_host_ram_plan(
        imgs, None, len(imgs), (64, 80), 1
    )
    assert decision.strategy == HOST_REFUSE
    # The dispatch seam must raise, never call fn_cpu (no legacy fallback).
    with pytest.raises(CpuWinsorMemoryRefused):
        o._gpu_reduce_winsorized(fn_cpu, imgs, np.ones(3, np.float32))
    assert called["cpu"] is False


def test_scratch_failure_raises_controlled_and_cleans_up():
    # memmap_outputs selected, but scratch creation fails -> controlled refusal
    # and NO orphan scratch file.
    o = _bare_stacker(available=333 * MiB, output_folder=tempfile.mkdtemp())
    imgs = _small_images(shape=(2822, 4144))
    # Force a huge resident+output so memmap_outputs is selected.
    imgs = [np.zeros((2822, 4144), np.float32) for _ in range(3)]
    decision, _a, _r, _res, _resid, _out = o._winsorized_host_ram_plan(
        imgs, None, len(imgs), (2822, 4144), 1
    )
    assert decision.strategy == HOST_MEMMAP_OUTPUTS

    def boom(*a, **k):
        raise ScratchSpaceRefused("no disk")

    o._make_winsorized_scratch = boom
    with pytest.raises(CpuWinsorMemoryRefused):
        o._gpu_reduce_winsorized(
            lambda *a, **k: (np.zeros((2822, 4144), np.float32), 0.0),
            imgs, np.ones(3, np.float32),
        )
    # No orphan scratch files (cleanup ran even though creation failed).
    for store in o._winsorized_scratch_stores:
        assert store._created == []


def test_cpu_fallback_reuses_host_context_no_double_plan():
    o = _bare_stacker(available=333 * MiB, output_folder=tempfile.mkdtemp())
    imgs = [np.zeros((2822, 4144), np.float32) for _ in range(3)]
    decision, _a, _r, _res, _resid, _out = o._winsorized_host_ram_plan(
        imgs, None, len(imgs), (2822, 4144), 1
    )
    assert decision.strategy == HOST_MEMMAP_OUTPUTS

    ctx_seen = {}

    def fn_cpu(sp_images, w=None, **kw):
        ctx_seen["ctx"] = getattr(o, "_winsorized_host_ctx", None)
        # The CPU path must see the transported plan (memmap_outputs).
        assert ctx_seen["ctx"] is not None
        return (np.zeros((2822, 4144), np.float32),
                np.ones((2822, 4144), np.float32), 0.0)

    o._gpu_reduce_winsorized(fn_cpu, imgs, np.ones(3, np.float32))
    # Context cleared after the CPU fallback.
    assert getattr(o, "_winsorized_host_ctx", None) is None
    # The transported plan was the memmap_outputs plan (no re-resolution).
    assert ctx_seen["ctx"][0].strategy == HOST_MEMMAP_OUTPUTS
