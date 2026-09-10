"""Hermetic tests for the P2-A live deposition-truth witness.

Covers the required P2-A rework-1 evidence:

1. hermetic synthetic cancellation (substantial +/- deltas, tiny positive
   residual, exact aggregate closure, reconstructed SCI);
2. stable control with materially better cancellation quality;
3. default-off science neutrality (accumulator SCI/WHT equality off vs on);
4. bounded memory/state and deterministic artifact.

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


class FakeAcc:
    """Minimal passive accumulator stand-in (scalar state only)."""

    def __init__(self, out_img, out_wht, kernel="lanczos2"):
        self._out_img = out_img
        self._out_wht = out_wht
        self.kernel = kernel


def _targets(tmp_path, specs):
    p = tmp_path / "targets.json"
    p.write_text(json.dumps({"targets": specs}), encoding="utf-8")
    return str(p)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(dt.TARGETS_ENV, raising=False)
    monkeypatch.delenv(dt.OUT_ENV, raising=False)
    dt.reset()
    yield
    dt.reset()


def test_synthetic_cancellation_closure_and_reconstructed_sci(tmp_path, monkeypatch):
    monkeypatch.setenv(
        dt.TARGETS_ENV,
        _targets(
            tmp_path,
            [
                {"run": "s", "kernel": "lanczos2", "channel": 0, "row": 0, "col": 0,
                 "category": "catastrophic"},
                {"run": "s", "kernel": "lanczos2", "channel": 0, "row": 0, "col": 1,
                 "category": "stable"},
            ],
        ),
    )
    assert dt.is_enabled()

    img = np.zeros((1, 2), dtype=np.float32)
    wht = np.zeros((1, 2), dtype=np.float32)
    acc = FakeAcc(img, wht)

    # catastrophic target: alternating +/- deltas, tiny positive residual
    # stable target: uniformly positive deltas
    value = 100.0
    d_seq = [1.0, -0.98, 1.0, -0.98, 1.0, -0.98, 1.0, -0.98, 1.0, -0.96]  # sum=+0.2
    for k, d_cat in enumerate(d_seq):
        before = dt.snapshot(0, acc)
        # exact cumulative numerator/denominator bookkeeping; the recorder only
        # reads the scalar state, so set the state directly
        img[0, 1] = value
        wht[0, 1] += 0.5
        wht[0, 0] += d_cat
        img[0, 0] = value
        after = dt.snapshot(0, acc)
        dt.record(0, "lanczos2", f"f{k}", before, after)

    art = dt.build_artifact(accs=[acc])
    cat, stable = art["targets"][0], art["targets"][1]

    # closure: aggregate deltas reproduce the final native state exactly
    assert cat["sum_D"] == pytest.approx(float(wht[0, 0]), rel=1e-12, abs=1e-12)
    # float32 state accumulates the numerator; allow float32 round-off
    assert cat["sum_N"] == pytest.approx(float(img[0, 0] * wht[0, 0]), rel=1e-5, abs=1e-4)

    # substantial positive and negative deltas, tiny positive residual
    assert cat["pos_D"] > 4.0
    assert cat["neg_D"] < -4.0
    assert 0.0 < cat["sum_D"] < 0.25
    assert cat["sum_abs_D"] > 9.0
    # positives 5*1.0 = 5.0 ; negatives 4*0.98 + 0.96 = 4.88 ; residual +0.12
    assert cat["cancellation_quality"] == pytest.approx(0.12 / 9.88, rel=1e-5)
    assert cat["cancellation_quality"] < 0.05

    # reconstructed SCI is the independently summed N/D
    assert cat["reconstructed_sci"] == pytest.approx(100.0, rel=1e-5)
    assert cat["reconstructed_vs_native_rel_err"] < 1e-5

    # stable control: materially better conditioning
    assert stable["neg_D"] == 0.0
    assert stable["cancellation_quality"] == pytest.approx(1.0)
    assert stable["cancellation_quality"] > 50.0 * cat["cancellation_quality"]


def test_default_off_science_neutrality(tmp_path, monkeypatch):
    """Instrumentation off vs on must leave the native buffers bit-identical."""

    def run_once():
        acc = drizzle_core.DrizzleAccumulator((48, 48), kernel="lanczos2", pixfrac=1.0)
        rng = np.random.default_rng(3)
        yy, xx = np.indices((16, 16), dtype=np.float64)
        px = 3.0 * (xx + 0.5)
        py = 3.0 * (yy + 0.5)
        pixmap = np.dstack((px, py))
        mask = np.ones((16, 16), dtype=np.float32)
        for _ in range(3):
            data = rng.normal(50.0, 5.0, (16, 16)).astype(np.float32)
            acc.add(data, mask, pixmap, exptime=20.0, in_units="counts")
        return np.array(acc._out_img, copy=True), np.array(acc._out_wht, copy=True)

    off_img, off_wht = run_once()

    monkeypatch.setenv(
        dt.TARGETS_ENV,
        _targets(tmp_path, [{"channel": 0, "row": 24, "col": 24, "category": "x"}]),
    )
    dt.reset()
    assert dt.is_enabled()
    # drive the same adds with the passive hook active
    acc = drizzle_core.DrizzleAccumulator((48, 48), kernel="lanczos2", pixfrac=1.0)
    rng = np.random.default_rng(3)
    yy, xx = np.indices((16, 16), dtype=np.float64)
    pixmap = np.dstack((3.0 * (xx + 0.5), 3.0 * (yy + 0.5)))
    mask = np.ones((16, 16), dtype=np.float32)
    for _ in range(3):
        data = rng.normal(50.0, 5.0, (16, 16)).astype(np.float32)
        before = dt.snapshot(0, acc)
        acc.add(data, mask, pixmap, exptime=20.0, in_units="counts")
        dt.record(0, "lanczos2", "f", before, dt.snapshot(0, acc))

    assert np.array_equal(off_img, acc._out_img)
    assert np.array_equal(off_wht, acc._out_wht)
    art = dt.build_artifact(accs=[acc])
    assert art["enabled"] is True
    assert art["targets"][0]["n_adds"] == 3
    assert art["targets"][0]["sum_D"] == pytest.approx(
        float(art["targets"][0]["final_native_wht"]), rel=1e-6, abs=1e-12
    )


def test_disabled_module_is_a_noop(monkeypatch):
    assert dt.is_enabled() is False
    assert dt.snapshot(0, FakeAcc(np.zeros((1, 1)), np.zeros((1, 1)))) is None
    dt.record(0, "lanczos2", "f", None, None)
    art = dt.build_artifact()
    assert art["enabled"] is False


def test_bounded_targets_rows_and_determinism(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "MAX_TARGETS", 3)
    monkeypatch.setattr(dt, "MAX_ROWS", 2)
    specs = [
        {"channel": 0, "row": 0, "col": i, "category": "c"} for i in range(6)
    ]
    monkeypatch.setenv(dt.TARGETS_ENV, _targets(tmp_path, specs))
    dt.reset()
    assert dt.is_enabled()
    assert dt.dropped_targets() == 3

    img = np.zeros((1, 6), dtype=np.float32)
    wht = np.zeros((1, 6), dtype=np.float32)
    acc = FakeAcc(img, wht)
    for k in range(5):
        before = dt.snapshot(0, acc)
        wht[0, :] += 1.0
        img[0, :] = 2.0
        after = dt.snapshot(0, acc)
        dt.record(0, "lanczos2", f"f{k}", before, after)

    a1 = dt.build_artifact(accs=[acc])
    a2 = dt.build_artifact(accs=[acc])
    assert a1 == a2
    assert a1["n_targets"] == 3
    assert a1["rows_truncated"] is True
    assert a1["rows_recorded"] == 2
    assert len(a1["rows"]) == dt.MAX_ROWS
    assert a1["targets"][0]["sum_D"] == pytest.approx(5.0)
    assert a1["targets"][0]["cancellation_quality"] == pytest.approx(1.0)
