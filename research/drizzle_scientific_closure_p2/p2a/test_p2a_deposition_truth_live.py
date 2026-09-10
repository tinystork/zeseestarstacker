"""Hermetic tests for the P2-A live deposition-truth witness (rework-2).

Covers the required rework-2 evidence:

1. hermetic synthetic cancellation (substantial +/- deltas, tiny positive
   residual, exact per-target closure, reconstructed SCI);
2. stable control with materially better cancellation quality;
3. default-off science neutrality (accumulator SCI/WHT equality off vs on);
4. two sequential runs in one process (no contamination);
5. L2 vs L3 target filtering (never relabel sampled data);
6. per-target closure + resume labelling;
7. repeated persist must not reset the run;
8. malformed / out-of-range target truthfulness;
9. bounded state with an honest *measured* size witness;
10. the canonical run-start seam binds a fresh recorder.

No physical artifacts and no network are required.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

dt = importlib.import_module("seestar.core.drizzle_deposition_truth")
drizzle_core = importlib.import_module("seestar.core.drizzle_core")
qm = importlib.import_module("seestar.queuep.queue_manager")


class FakeAcc:
    """Minimal passive accumulator stand-in (scalar state only)."""

    def __init__(self, out_img, out_wht, kernel="lanczos2"):
        self._out_img = out_img
        self._out_wht = out_wht
        self.kernel = kernel


def _write_targets(tmp_path, specs, name="targets.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"targets": specs}), encoding="utf-8")
    return str(p)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(dt.TARGETS_ENV, raising=False)
    monkeypatch.delenv(dt.OUT_ENV, raising=False)
    dt.reset()
    yield
    dt.reset()


def _enable(monkeypatch, tmp_path, specs, name="targets.json"):
    monkeypatch.setenv(dt.TARGETS_ENV, _write_targets(tmp_path, specs, name))
    dt.reset()
    assert dt.is_enabled()


# ---------------------------------------------------------------------------
# 1 + 2 + 6  cancellation, closure, stable control
# ---------------------------------------------------------------------------


def test_synthetic_cancellation_closure_and_reconstructed_sci(tmp_path, monkeypatch):
    _enable(
        monkeypatch,
        tmp_path,
        [
            {"kernel": "lanczos2", "channel": 0, "row": 0, "col": 0, "category": "catastrophic"},
            {"kernel": "lanczos2", "channel": 0, "row": 0, "col": 1, "category": "stable"},
        ],
    )
    dt.start_run(kernel="lanczos2", scale=3.0, resume=False, out_dir=str(tmp_path))

    img = np.zeros((1, 2), dtype=np.float32)
    wht = np.zeros((1, 2), dtype=np.float32)
    acc = FakeAcc(img, wht)

    value = 100.0
    d_seq = [1.0, -0.98, 1.0, -0.98, 1.0, -0.98, 1.0, -0.98, 1.0, -0.96]  # +0.12
    for k, d_cat in enumerate(d_seq):
        before = dt.snapshot(0, acc)
        img[0, 1] = value
        wht[0, 1] += 0.5
        wht[0, 0] += d_cat
        img[0, 0] = value
        dt.record(0, "lanczos2", f"f{k}", before, dt.snapshot(0, acc))

    art = dt.build_artifact()
    cat, stable = art["targets"][0], art["targets"][1]

    # per-target closure against final-minus-initial native state
    assert cat["sum_D"] == pytest.approx(float(wht[0, 0]), rel=1e-12, abs=1e-12)
    assert cat["closure_sum_D_err"] == pytest.approx(0.0, abs=1e-12)
    assert cat["closure_sum_N_err"] == pytest.approx(0.0, abs=1e-4)
    assert cat["initial_state_zero"] is True
    assert cat["resume_nonzero_initial"] is False

    assert cat["pos_D"] > 4.0
    assert cat["neg_D"] < -4.0
    assert 0.0 < cat["sum_D"] < 0.25
    assert cat["sum_abs_D"] > 9.0
    assert cat["cancellation_quality"] == pytest.approx(0.12 / 9.88, rel=1e-5)
    assert cat["cancellation_quality"] < 0.05
    assert cat["reconstructed_sci"] == pytest.approx(100.0, rel=1e-5)

    assert stable["neg_D"] == 0.0
    assert stable["cancellation_quality"] == pytest.approx(1.0)
    assert stable["cancellation_quality"] > 50.0 * cat["cancellation_quality"]


def test_resume_nonzero_initial_is_labelled(tmp_path, monkeypatch):
    _enable(monkeypatch, tmp_path, [{"kernel": "lanczos2", "channel": 0, "row": 0, "col": 0}])
    dt.start_run(kernel="lanczos2", resume=True, out_dir=str(tmp_path))
    img = np.array([[5.0]], dtype=np.float32)
    wht = np.array([[2.0]], dtype=np.float32)  # non-zero initial state
    acc = FakeAcc(img, wht)
    before = dt.snapshot(0, acc)
    wht[0, 0] += 1.0
    dt.record(0, "lanczos2", "f", before, dt.snapshot(0, acc))
    rec = dt.build_artifact()["targets"][0]
    assert rec["initial_state_zero"] is False
    assert rec["resume_nonzero_initial"] is True
    # closure is against final-minus-initial, never total final state
    assert rec["closure_sum_D_err"] == pytest.approx(0.0, abs=1e-12)
    assert rec["initial_native_wht"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 3  default-off science neutrality
# ---------------------------------------------------------------------------


def test_default_off_science_neutrality(tmp_path, monkeypatch):
    def drive():
        acc = drizzle_core.DrizzleAccumulator((48, 48), kernel="lanczos2", pixfrac=1.0)
        rng = np.random.default_rng(3)
        yy, xx = np.indices((16, 16), dtype=np.float64)
        pixmap = np.dstack((3.0 * (xx + 0.5), 3.0 * (yy + 0.5)))
        mask = np.ones((16, 16), dtype=np.float32)
        for _ in range(3):
            acc.add(rng.normal(50.0, 5.0, (16, 16)).astype(np.float32), mask, pixmap,
                    exptime=20.0, in_units="counts")
        return acc

    off = drive()
    _enable(monkeypatch, tmp_path, [{"kernel": "lanczos2", "channel": 0, "row": 24, "col": 24}])
    dt.start_run(kernel="lanczos2", out_dir=str(tmp_path))
    on = drizzle_core.DrizzleAccumulator((48, 48), kernel="lanczos2", pixfrac=1.0)
    rng = np.random.default_rng(3)
    yy, xx = np.indices((16, 16), dtype=np.float64)
    pixmap = np.dstack((3.0 * (xx + 0.5), 3.0 * (yy + 0.5)))
    mask = np.ones((16, 16), dtype=np.float32)
    for _ in range(3):
        before = dt.snapshot(0, on)
        on.add(rng.normal(50.0, 5.0, (16, 16)).astype(np.float32), mask, pixmap,
               exptime=20.0, in_units="counts")
        dt.record(0, "lanczos2", "f", before, dt.snapshot(0, on))

    assert np.array_equal(off._out_img, on._out_img)
    assert np.array_equal(off._out_wht, on._out_wht)
    art = dt.build_artifact()
    assert art["enabled"] is True
    assert art["targets"][0]["n_adds"] == 3
    assert art["targets"][0]["closure_sum_D_err"] == pytest.approx(0.0, abs=1e-9)


def test_disabled_module_is_a_noop(monkeypatch):
    assert dt.is_enabled() is False
    assert dt.snapshot(0, FakeAcc(np.zeros((1, 1)), np.zeros((1, 1)))) is None
    dt.record(0, "lanczos2", "f", None, None)
    assert dt.start_run(kernel="lanczos2") is None
    assert dt.build_artifact()["enabled"] is False


# ---------------------------------------------------------------------------
# 4  sequential runs in one process (no contamination)
# ---------------------------------------------------------------------------


def test_two_sequential_runs_no_contamination(tmp_path, monkeypatch):
    _enable(
        monkeypatch,
        tmp_path,
        [
            {"kernel": "lanczos2", "channel": 0, "row": 0, "col": 0, "category": "l2"},
            {"kernel": "lanczos3", "channel": 0, "row": 0, "col": 0, "category": "l3"},
        ],
    )

    def one_run(kernel):
        dt.start_run(kernel=kernel, out_dir=str(tmp_path))
        img = np.zeros((1, 1), dtype=np.float32)
        wht = np.zeros((1, 1), dtype=np.float32)
        acc = FakeAcc(img, wht, kernel=kernel)
        for _ in range(3):
            before = dt.snapshot(0, acc)
            wht[0, 0] += 1.0
            img[0, 0] = 7.0
            dt.record(0, kernel, "f", before, dt.snapshot(0, acc))
        return dt.build_artifact()

    a = one_run("lanczos2")
    b = one_run("lanczos3")

    assert a["provenance"]["effective_kernel"] == "lanczos2"
    assert b["provenance"]["effective_kernel"] == "lanczos3"
    assert [t["category"] for t in a["targets"]] == ["l2"]
    assert [t["category"] for t in b["targets"]] == ["l3"]
    # fresh recorder: no aggregate carried over from run 1
    assert b["targets"][0]["n_adds"] == 3
    assert b["targets"][0]["sum_D"] == pytest.approx(3.0)
    assert a["targets"][0]["sum_D"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 5  kernel filtering, never relabel
# ---------------------------------------------------------------------------


def test_kernel_filtering_and_wildcard(tmp_path, monkeypatch):
    _enable(
        monkeypatch,
        tmp_path,
        [
            {"kernel": "lanczos2", "channel": 0, "row": 0, "col": 0, "category": "l2"},
            {"kernel": "lanczos3", "channel": 0, "row": 0, "col": 1, "category": "l3"},
            {"channel": 0, "row": 0, "col": 2, "category": "wildcard"},
        ],
    )
    meta = dt.start_run(kernel="lanczos2", out_dir=str(tmp_path))
    assert meta["selected_targets"] == 2
    assert meta["skipped_by_kernel"] == 1

    img = np.zeros((1, 3), dtype=np.float32)
    wht = np.zeros((1, 3), dtype=np.float32)
    acc = FakeAcc(img, wht)
    for _ in range(2):
        before = dt.snapshot(0, acc)
        wht[0, :] += 1.0
        img[0, :] = 3.0
        dt.record(0, "lanczos2", "f", before, dt.snapshot(0, acc))
    # a foreign-kernel call must never be folded / relabelled
    before = dt.snapshot(0, acc)
    wht[0, :] += 1.0
    dt.record(0, "lanczos3", "f", before, dt.snapshot(0, acc))

    art = dt.build_artifact()
    assert art["provenance"]["effective_kernel"] == "lanczos2"
    assert art["kernel_mismatch_events"] == 1
    assert {t["category"] for t in art["targets"]} == {"l2", "wildcard"}
    for t in art["targets"]:
        assert t["n_adds"] == 2
        assert t["sum_D"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 7  repeated persist must not reset
# ---------------------------------------------------------------------------


def test_repeated_persist_does_not_reset(tmp_path, monkeypatch):
    _enable(monkeypatch, tmp_path, [{"kernel": "lanczos2", "channel": 0, "row": 0, "col": 0}])
    dt.start_run(kernel="lanczos2", out_dir=str(tmp_path))
    img = np.zeros((1, 1), dtype=np.float32)
    wht = np.zeros((1, 1), dtype=np.float32)
    acc = FakeAcc(img, wht)

    def add(n):
        for _ in range(n):
            before = dt.snapshot(0, acc)
            wht[0, 0] += 1.0
            img[0, 0] = 1.0
            dt.record(0, "lanczos2", "f", before, dt.snapshot(0, acc))

    add(2)
    p1 = dt.persist(str(tmp_path))
    p2 = dt.persist(str(tmp_path))
    assert p1 and p2 and Path(p1).is_file()
    add(3)
    art = dt.build_artifact()
    assert art["targets"][0]["n_adds"] == 5
    assert art["targets"][0]["sum_D"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 8  malformed / out-of-range truthfulness
# ---------------------------------------------------------------------------


def test_malformed_and_out_of_range_targets(tmp_path, monkeypatch):
    specs = [
        {"kernel": "lanczos2", "channel": 0, "row": 0, "col": 0},
        {"kernel": "lanczos2", "col": 1},  # malformed (missing channel/row)
        {"kernel": "lanczos2", "channel": 0, "row": 999, "col": 999},  # out of range
    ]
    _enable(monkeypatch, tmp_path, specs)
    assert dt.dropped_targets() == 1
    dt.start_run(kernel="lanczos2", out_dir=str(tmp_path))
    img = np.zeros((1, 2), dtype=np.float32)
    wht = np.zeros((1, 2), dtype=np.float32)
    acc = FakeAcc(img, wht)
    before = dt.snapshot(0, acc)
    wht[0, 0] += 1.0
    dt.record(0, "lanczos2", "f", before, dt.snapshot(0, acc))
    art = dt.build_artifact()
    assert art["provenance"]["dropped_targets"] == 1
    assert art["missing_state_events"] >= 1
    oob = [t for t in art["targets"] if t["row"] == 999][0]
    assert oob["n_adds"] == 0
    assert oob["initial_native_wht"] is None


# ---------------------------------------------------------------------------
# 9  bounded, measured state + determinism
# ---------------------------------------------------------------------------


def test_bounded_state_measured_and_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "MAX_TARGETS", 64)
    monkeypatch.setattr(dt, "MAX_ROWS", 32)
    specs = [{"kernel": "lanczos2", "channel": 0, "row": 0, "col": i} for i in range(64)]
    _enable(monkeypatch, tmp_path, specs)
    dt.start_run(kernel="lanczos2", out_dir=str(tmp_path))
    img = np.zeros((1, 64), dtype=np.float32)
    wht = np.zeros((1, 64), dtype=np.float32)
    acc = FakeAcc(img, wht)
    for k in range(40):
        before = dt.snapshot(0, acc)
        wht[0, :] += 1.0
        img[0, :] = 2.0
        dt.record(0, "lanczos2", f"f{k}", before, dt.snapshot(0, acc))
    a1 = dt.build_artifact()
    a2 = dt.build_artifact()
    assert a1 == a2
    assert a1["rows_truncated"] is True
    assert a1["rows_recorded"] == dt.MAX_ROWS
    assert a1["targets"][0]["sum_D"] == pytest.approx(40.0)
    # honest measured in-process state: the compact numpy row witness only
    assert a1["measured_state_bytes"] > 0
    assert a1["measured_state_bytes"] <= 8 * 1024 * 1024


def test_full_cap_state_witness(tmp_path, monkeypatch):
    """Measure the honest worst case at the documented hard caps."""
    monkeypatch.setattr(dt, "MAX_TARGETS", 64)
    monkeypatch.setattr(dt, "MAX_ROWS", 2048)
    specs = [{"kernel": "lanczos2", "channel": 0, "row": 0, "col": i} for i in range(64)]
    _enable(monkeypatch, tmp_path, specs)
    dt.start_run(kernel="lanczos2", out_dir=str(tmp_path))
    img = np.zeros((1, 64), dtype=np.float32)
    wht = np.zeros((1, 64), dtype=np.float32)
    acc = FakeAcc(img, wht)
    for k in range(2048):
        wht[0, :] += 1.0
        img[0, :] = 2.0
        dt.record(0, "lanczos2", f"f{k}", dt.snapshot(0, acc), dt.snapshot(0, acc))
    art = dt.build_artifact()
    assert art["rows_recorded"] == 2048
    measured = art["measured_state_bytes"]
    print(f"FULL_CAP_MEASURED_STATE_BYTES={measured}")
    # 2048 rows x 64 targets x 3 float64 = 3.1 MiB payload; a conservative
    # ceiling of 16 MiB covers python object overhead honestly.
    assert 3_000_000 < measured < 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# 10  canonical run-start seam
# ---------------------------------------------------------------------------


def test_canonical_run_start_seam_binds_fresh_recorder(tmp_path, monkeypatch):
    _enable(monkeypatch, tmp_path, [{"kernel": "lanczos2", "channel": 0, "row": 0, "col": 0}])
    calls = []
    monkeypatch.setattr(dt, "start_run", lambda **kw: calls.append(kw))

    class Stub:
        output_folder = str(tmp_path)
        reference_wcs_object = None
        drizzle_output_wcs = None
        update_progress = None

    qm.SeestarQueuedStacker._init_drizzle_science_diagnostics(
        Stub(), "lanczos2", 1.0, 1.0, 3.0, None
    )
    assert calls and calls[0]["kernel"] == "lanczos2"
    assert calls[0]["resume"] is False
    assert calls[0]["out_dir"] == str(tmp_path)

    calls.clear()
    qm.SeestarQueuedStacker._init_drizzle_science_diagnostics(
        Stub(), "lanczos3", 1.0, 1.0, 3.0, object()
    )
    assert calls and calls[0]["kernel"] == "lanczos3"
    assert calls[0]["resume"] is True
