"""8.4.0 pre-W80 stage C — explicit byte-budget contract (queue -> worker -> core).

Mission ``zsss-840-prew80-20260907``.  Regression witnesses for the confirmed
hidden-budget propagation defect:

* PRE-FIX (historical archaeology evidence, ``archaeology-witness.json``):
  outer resolved budget ``25769803949`` (24 GiB + 173) reached the queue
  wrapper, but the deep core primitive received ``2000000000`` (the env /
  2 GiB default) because the worker tuple carried no budget field.
* POST-FIX (this suite): the worker tuple carries the resolved budget as an
  explicit 9th field; ``_stack_worker`` forwards it VERBATIM into
  ``_stack_winsorized_sigma``; the direct seam and a REAL ``ProcessPoolExecutor``
  seam (spawn context, not a mocked executor) observe an identical memory
  contract.  A sentinel spy at the core boundary proves the exact value
  arrives (no silent 1/2 GiB / env fallback on the production path).
"""

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pytest

from seestar.queuep import queue_manager as qm
from seestar.queuep.queue_manager import _stack_worker
from seestar.core import stack_methods as sm

# Historical archaeology sentinel (24 GiB + 173 bytes) -> pre-fix deep 2e9.
SENTINEL = 25769803949
PRE_FIX_DEEP = 2_000_000_000
MiB = 1024 * 1024


def _small_images(n=8, h=16, w=16, c=1, seed=0):
    rng = np.random.default_rng(seed)
    return [
        rng.normal(100.0, 5.0, size=(h, w) if c == 1 else (h, w, c)).astype(np.float32)
        for _ in range(n)
    ]


def _worker_args(images, budget, weights=None, return_weights=False):
    """9-field worker tuple exactly as the production queue wrapper builds it."""
    return (
        "winsorized-sigma",
        images,
        weights,
        3.0,
        3.0,
        (0.05, 0.05),
        True,
        return_weights,
        budget,
    )


def test_worker_forwards_exact_sentinel_to_core_boundary(monkeypatch):
    """§45 sentinel regression: the deep core boundary receives the EXACT
    outer value (post-fix), not the historical 2e9 default."""
    seen = {}

    def spy_core(images, weights, **kw):
        seen["max_mem_bytes"] = kw.get("max_mem_bytes")
        return np.zeros(np.asarray(images[0]).shape, dtype=np.float32), 0.0

    monkeypatch.setattr(sm, "_stack_winsorized_sigma", spy_core)
    imgs = _small_images(n=4, h=8, w=8)
    res = _stack_worker(_worker_args(imgs, SENTINEL))
    assert seen["max_mem_bytes"] == SENTINEL
    assert seen["max_mem_bytes"] != PRE_FIX_DEEP  # regression vs archaeology
    assert res is not None


def test_queue_wrapper_threads_resolved_budget_through_tuple(monkeypatch):
    """The queue Winsorized wrapper resolves the budget ONCE and the tuple
    carries it explicitly (direct seam, max_stack_workers == 1)."""
    seen = {}

    def fake_worker(args):
        seen["tuple"] = args
        return np.zeros((8, 8), dtype=np.float32), 0.0

    monkeypatch.setattr(qm, "_stack_worker", fake_worker)
    o = qm.SeestarQueuedStacker.__new__(qm.SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.max_stack_workers = 1
    imgs = _small_images(n=4, h=8, w=8)
    o._stack_winsorized_sigma(
        imgs, None, kappa=3.0, max_mem_bytes=SENTINEL, return_weights=False
    )
    args = seen["tuple"]
    assert len(args) == 9
    assert args[8] == SENTINEL


def test_budget_reaches_core_in_direct_seam():
    """Discriminating direct test: images bigger than the resolved budget must
    make the CORE guard raise (pre-fix, the 2e9 default would have silently
    succeeded)."""
    imgs = _small_images(n=64, h=64, w=64)  # 64*64*64*4 = 1 MiB cube
    budget = 512 * 1024  # below cube -> core guard must fire
    with pytest.raises(MemoryError, match="Stack exceeds max_mem_bytes"):
        _stack_worker(_worker_args(imgs, budget))


def test_budget_reaches_core_in_real_pool_seam():
    """§16 multiprocessing parity with a REAL ProcessPoolExecutor (spawn, as
    the queue wrapper uses): the identical tuple -> identical core behaviour
    (MemoryError raised deep in the child, propagated through .result())."""
    imgs = _small_images(n=64, h=64, w=64)
    budget = 512 * 1024
    args = _worker_args(imgs, budget)
    with ProcessPoolExecutor(
        max_workers=1, mp_context=get_context("spawn")
    ) as exe:
        fut = exe.submit(_stack_worker, args)
        with pytest.raises(MemoryError, match="Stack exceeds max_mem_bytes"):
            fut.result()


def test_direct_and_pool_contract_identical_on_success():
    """Both seams accept the same resolved budget and produce the SAME stacked
    result when the budget is sufficient (identical contract)."""
    imgs = _small_images(n=8, h=16, w=16, seed=3)
    weights = np.ones(8, dtype=np.float32)
    args = _worker_args(imgs, SENTINEL, weights=weights, return_weights=True)
    direct = _stack_worker(args)
    with ProcessPoolExecutor(
        max_workers=1, mp_context=get_context("spawn")
    ) as exe:
        pooled = exe.submit(_stack_worker, args).result()
    assert np.array_equal(direct[0], pooled[0])
    assert np.array_equal(direct[1], pooled[1])
    assert direct[2] == pooled[2]


def test_legacy_8field_tuple_still_accepted_for_non_production_callers():
    """Backward-compatible worker tolerance: an 8-field tuple (legacy direct
    callers / tests) maps to the documented standalone default (env / 2 GiB),
    never to the sentinel, and never breaks the dispatcher."""
    seen = {}

    def spy_core(images, weights, **kw):
        seen["max_mem_bytes"] = kw.get("max_mem_bytes")
        return np.zeros(np.asarray(images[0]).shape, dtype=np.float32), 0.0

    old = sm._stack_winsorized_sigma
    sm._stack_winsorized_sigma = spy_core
    try:
        imgs = _small_images(n=4, h=8, w=8)
        args8 = _worker_args(imgs, None)[:8]
        _stack_worker(args8)
    finally:
        sm._stack_winsorized_sigma = old
    assert seen["max_mem_bytes"] is None  # core resolves the standalone default


def test_queue_wrapper_guard_uses_resolved_budget(monkeypatch):
    """The queue wrapper's own pre-guard uses the resolved budget (not a
    frozen 1e9 default): a small explicit budget refuses a bigger cube."""
    o = qm.SeestarQueuedStacker.__new__(qm.SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.max_stack_workers = 1
    imgs = _small_images(n=64, h=64, w=64)  # 1 MiB cube
    with pytest.raises(RuntimeError, match="Stack exceeds max_mem_bytes"):
        o._stack_winsorized_sigma(imgs, None, max_mem_bytes=512 * 1024)
