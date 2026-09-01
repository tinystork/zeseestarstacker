"""PHI-R2 — debug-gated preview pipeline telemetry + deterministic witnesses.

Covers the PHI-R2 instrumentation contract (``docs/phi_viewer_archaeology.md``
§7) and the required reproduction witnesses:

1. **HDR / strong-WB plateau witness** — a deterministic synthetic Option-A
   payload with over-range raw-linear headroom and a strong WB gain proves
   *where* the first ``==1`` plateau appears in the current baseline pipeline
   (anchor map, then WB) and that telemetry is a **no-op when disabled**;
2. **reordered payload / resolution witness** — two valid Option-A payloads
   (same and changed metadata) delivered in adversarial order prove the
   current last-wins acceptance contract (no monotonic generation gate),
   deterministically, without timing/sleeps;
3. **producer-isolation proof** — Classic SUM/W and standard Drizzle producer
   runs (with telemetry enabled) leave scientific accumulators/inputs
   bit-identical;
4. **focused Qt + producer runs** under ``QT_QPA_PLATFORM=offscreen`` validate
   the stage records (source route, factor, dtype, shape, min/p01/median/p99/
   max, source-buffer identity) and their compact/bounded shape.

The trace gate is ``ZSSS_PHI_TRACE`` (any value other than ``0``/``false``/
``no``/``off`` enables; default disabled).  All records are compact
single-line ``PREVIEW_STAGE`` debug records — never per-pixel dumps.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seestar.gui_qt import BackendPreviewPayload, MainWindow, create_application  # noqa: E402
from seestar.gui_qt.main_window import _DRIZZLE_PREVIEW_SRC, _SUMW_PREVIEW_SRC  # noqa: E402
from seestar.gui_qt.preview_analysis import compute_anchors  # noqa: E402
from seestar.queuep.queue_manager import SeestarQueuedStacker  # noqa: E402
from seestar.core.drizzle_core import DrizzleAccumulator  # noqa: E402
from seestar.utils.phi_trace import phi_trace_enabled  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _classic_stack():
    obj = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    obj.update_progress = lambda *a, **k: None
    obj.preview_callback = None
    obj.current_stack_header = fits.Header()
    obj.images_in_cumulative_stack = 3
    obj.files_in_queue = 10
    obj.stacked_batches_count = 1
    obj.total_batches_estimated = 2
    return obj


def _drizzle_stack():
    obj = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    obj.update_progress = lambda *a, **k: None
    obj.preview_callback = None
    obj.current_stack_header = fits.Header()
    obj._drizzle_frame_count = 2
    obj.files_in_queue = 5
    obj.stacked_batches_count = 1
    obj.total_batches_estimated = 2
    return obj


def _fill_drizzle(obj, shape):
    obj.drizzle_accumulators = [DrizzleAccumulator(shape) for _ in range(3)]
    for i, acc in enumerate(obj.drizzle_accumulators):
        acc._out_img[:] = float(i + 1) * 10.0
        acc._out_wht[:] = 2.0


def _legacy_normalize(arr):
    """Min/max -> [0, 1] legacy display carrier (display-only)."""
    arr64 = arr.astype(np.float64)
    mn = float(np.nanmin(arr64))
    mx = float(np.nanmax(arr64))
    return np.clip((arr64 - mn) / (mx - mn), 0.0, 1.0).astype(np.float32)


def _hdr_raw(size: int = 64, seed: int = 7) -> np.ndarray:
    """Deterministic raw-linear HDR RGB: red channel carries headroom > 1."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.0, 2.0, size=(size, size))
    r = g * 1.4  # red-dominant strong headroom
    b = g.copy()
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _pump_until(predicate, timeout_ms: int = 5000) -> bool:
    """Pump the Qt event loop until ``predicate`` is true (or time out)."""
    from PySide6.QtWidgets import QApplication
    import time

    app = QApplication.instance()
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


def _wait_histogram(win, timeout_ms: int = 5000) -> bool:
    """Wait until the applied histogram model matches the current WB revision."""
    return _pump_until(
        lambda: win._histogram_model is not None
        and win._histogram_model_revision == win._wb_only_revision,
        timeout_ms,
    )


def _parse_stage_records(caplog):
    """Return ``PREVIEW_STAGE`` record messages in emission order."""
    out = []
    for record in caplog.records:
        msg = getattr(record, "message", "") or record.getMessage()
        if msg.startswith("PREVIEW_STAGE"):
            out.append(msg)
    return out


def _fields(msg: str) -> dict:
    """Parse a ``PREVIEW_STAGE`` line into a key/value dict."""
    d = {}
    for tok in msg.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    from PySide6.QtWidgets import QApplication

    assert app is QApplication.instance()
    return app


# ---------------------------------------------------------------------------
# Witness 1a — trace gate disabled by default (no-op, no records)
# ---------------------------------------------------------------------------

def test_phi_trace_gate_disabled_produces_no_records(caplog, monkeypatch):
    monkeypatch.delenv("ZSSS_PHI_TRACE", raising=False)
    caplog.set_level(logging.DEBUG)
    assert not phi_trace_enabled()

    obj = _classic_stack()
    obj.preview_downsample_factor = 1
    H, W = 16, 24
    avg = np.linspace(0.0, 1.0, H * W * 3, dtype=np.float32).reshape(H, W, 3)
    obj.cumulative_sum_memmap = avg.astype(np.float32)
    obj.cumulative_wht_memmap = np.ones((H, W), dtype=np.float32)
    collected = []
    obj.preview_callback = lambda *a: collected.append(a)
    obj._update_preview_sum_w(downsample_factor=2)

    assert len(collected) == 1  # pipeline still delivers normally
    assert _parse_stage_records(caplog) == []


# ---------------------------------------------------------------------------
# Witness 1b — HDR/strong-WB plateau: where does ==1 first appear?
# ---------------------------------------------------------------------------

def test_phi_witness_hdr_strong_wb_first_plateau(qapp, caplog, monkeypatch):
    """Prove where the ``>=1`` headroom first becomes an ``==1`` plateau.

    Diagnostic facts asserted (not a repaired behaviour):
    * raw_source carries max > 1 (headroom exists at the raw-linear boundary);
    * anchor_mapped is the FIRST stage whose max == 1.0 exactly (the anchor map
      clips over-range signal in the current baseline);
    * wb_only (with a strong 3x red gain) is also == 1.0 — the plateau already
      exists before WB, so WB is not the first clip;
    * histogram_output reports the in-domain max == 1.0 (the histogram only
      ever sees the clipped WB-only domain).
    """
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)
    win = MainWindow()
    try:
        raw = _hdr_raw()
        assert float(np.max(raw)) > 1.0  # headroom precondition
        win._wb = (3.0, 1.0, 1.0)  # strong red gain
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(raw), raw),
                stack_name="hdr",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
                image_count=1,
                current_batch=1,
            )
        )
        assert _wait_histogram(win)

        records = _parse_stage_records(caplog)
        by_stage = {}
        for msg in records:
            f = _fields(msg)
            by_stage.setdefault(f["stage"], []).append(f)

        # raw_source: headroom > 1 present.
        raw_recs = by_stage.get("raw_source", [])
        assert raw_recs, "missing raw_source trace"
        assert float(raw_recs[0]["max"]) > 1.0

        # The first stage whose max becomes exactly 1.0 is anchor_mapped.
        first_plateau = None
        for msg in records:
            f = _fields(msg)
            if "max" in f and float(f["max"]) == 1.0:
                first_plateau = f["stage"]
                break
        assert first_plateau == "anchor_mapped", (
            f"first ==1 plateau should be anchor_mapped, got {first_plateau}"
        )

        # WB-only already plateaued (WB is not the first clip site).
        wb_recs = by_stage.get("wb_only", [])
        assert wb_recs, "missing wb_only trace"
        assert float(wb_recs[0]["max"]) == 1.0

        # Histogram output max is in-domain == 1.0 (no headroom visible).
        hist_recs = by_stage.get("histogram_output", [])
        assert hist_recs, "missing histogram_output trace"
        assert float(hist_recs[0]["R_max"]) == 1.0
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# Witness 2 — reordered payload/resolution: current last-wins contract
# ---------------------------------------------------------------------------

def _option_a_payload(raw, batch, name, prev_src=_SUMW_PREVIEW_SRC):
    return BackendPreviewPayload(
        data=(_legacy_normalize(raw), raw),
        stack_name=name,
        header={"PREV_SRC": prev_src},
        image_count=batch,
        current_batch=batch,
    )


def test_phi_witness_reordered_payloads_last_wins(qapp, caplog, monkeypatch):
    """Current acceptance contract: last delivered valid payload replaces the
    source unconditionally (no monotonic generation gate), deterministically.

    Scenario A (same metadata): full-res 1/1 delivered first, stale half-res
    delivered second -> the stale half-res *replaces* the full-res display.
    Scenario B (changed metadata): payload with batch 3 (newer) delivered
    first, batch 2 (older) delivered second -> the older one wins.
    Both prove the documented no-generation-gate limitation; no sleeps.
    """
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)
    win = MainWindow()
    try:
        full = _hdr_raw(size=64, seed=1)
        half = _hdr_raw(size=32, seed=1)

        # Scenario A: same identity (batch 2), adversarial order.
        win._on_preview(_option_a_payload(full, 2, "full-1/1"))
        win._on_preview(_option_a_payload(half, 2, "stale-1/2"))
        assert win._preview_source is not None
        assert win._preview_source.width() == 32  # stale half-res replaced full
        assert win._preview_identity[1:] == ("batch", 2)

        # Scenario B: changed metadata — newer batch first, older batch last.
        win._on_preview(_option_a_payload(full, 3, "newer-batch3-1/1"))
        win._on_preview(_option_a_payload(half, 2, "older-batch2-1/2"))
        assert win._preview_source.width() == 32  # older batch wins (last)
        assert win._preview_identity[1:] == ("batch", 2)

        # The trace documents the arrival order (identity + delivered shape),
        # making the ordering limitation observable without timing.
        records = _parse_stage_records(caplog)
        arrivals = [
            _fields(m) for m in records if _fields(m)["stage"] == "payload_arrive"
        ]
        assert len(arrivals) >= 4
        assert arrivals[-1]["identity"] == "batch:2"
        assert arrivals[-1]["shape"] == "32x32x3"
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# Witness 3 — producer isolation with telemetry ENABLED
# ---------------------------------------------------------------------------

def test_phi_witness_producer_isolation_with_trace_on(caplog, monkeypatch):
    """Classic + Drizzle producers leave science state bit-identical even when
    the trace gate is enabled, and the records stay compact/bounded."""
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)

    # Classic: SUM/W memmaps unchanged.
    obj = _classic_stack()
    obj.preview_downsample_factor = 1
    H, W = 16, 24
    rng = np.random.default_rng(30)
    sum_mm = rng.uniform(0.0, 10.0, size=(H, W, 3)).astype(np.float32)
    sum_before = sum_mm.copy()
    wht_mm = np.full((H, W), 2.0, dtype=np.float32)
    wht_before = wht_mm.copy()
    obj.cumulative_sum_memmap = sum_mm
    obj.cumulative_wht_memmap = wht_mm
    collected = []
    obj.preview_callback = lambda *a: collected.append(a)
    obj._update_preview_sum_w(downsample_factor=2)
    assert np.array_equal(sum_mm, sum_before)
    assert np.array_equal(wht_mm, wht_before)

    # Drizzle: accumulator state unchanged.
    obj2 = _drizzle_stack()
    obj2.preview_downsample_factor = 1
    _fill_drizzle(obj2, (H, W))
    img_before = [a._out_img.copy() for a in obj2.drizzle_accumulators]
    wht_before = [a._out_wht.copy() for a in obj2.drizzle_accumulators]
    collected2 = []
    obj2.preview_callback = lambda *a: collected2.append(a)
    obj2._update_preview_drizzle_accumulator()
    for acc, bi, bw in zip(obj2.drizzle_accumulators, img_before, wht_before):
        assert np.array_equal(acc._out_img, bi)
        assert np.array_equal(acc._out_wht, bw)

    # Records are compact single lines with the required fields, never
    # whole-array dumps.
    records = _parse_stage_records(caplog)
    assert records, "expected PREVIEW_STAGE records with gate on"
    for msg in records:
        assert "\n" not in msg
        assert len(msg) < 400, f"record too long ({len(msg)}): {msg}"
        assert "array(" not in msg
        assert "[" not in msg
        f = _fields(msg)
        assert f["route"] in ("classic", "drizzle", "qt")
        for field in ("stage", "dtype", "shape", "min", "p01", "median", "p99", "max"):
            assert field in f, f"record missing {field}: {msg}"
        if f["stage"] in ("source", "pre_resize", "post_resize"):
            assert "factor" in f
            assert "src" in f
            assert "src_id" in f


# ---------------------------------------------------------------------------
# Producer stage sequence: direct-source resolution sequence 1 -> 2 -> 3 -> 4 -> 1
# ---------------------------------------------------------------------------

def test_phi_witness_classic_direct_source_resolution_sequence(caplog, monkeypatch):
    """Classic SUM/W: every factor output is a direct INTER_AREA resize of the
    full-res SUM/W divide (no cumulative-resize chain); the final 1/1 equals a
    fresh direct 1/1 render; science arrays stay unchanged."""
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)
    obj = _classic_stack()
    H, W = 64, 96
    rng = np.random.default_rng(11)
    avg = rng.uniform(0.0, 50.0, size=(H, W, 3)).astype(np.float32)
    sum_before = avg.copy()
    obj.cumulative_sum_memmap = avg.astype(np.float32)
    obj.cumulative_wht_memmap = np.ones((H, W), dtype=np.float32)
    wht_before = obj.cumulative_wht_memmap.copy()

    def render(factor):
        collected = []
        obj.preview_callback = lambda *a: collected.append(a)
        obj._update_preview_sum_w(downsample_factor=factor)
        assert len(collected) == 1
        return collected[0][0]

    import cv2

    raws = {}
    for f in (1, 2, 3, 4, 1):
        legacy, raw = render(f)
        raws[f] = raw
        assert legacy.shape == raw.shape
        if f == 1:
            assert raw.shape == (H, W, 3)
        else:
            assert raw.shape == (H // f, W // f, 3)
            expected = cv2.resize(
                raws[1], (W // f, H // f), interpolation=cv2.INTER_AREA
            )
            assert np.array_equal(raw, expected), f"factor {f} not a direct resize"

    # Final 1/1 == a fresh direct 1/1 render from the authoritative source.
    fresh_legacy, fresh_raw = render(1)
    assert np.array_equal(raws[1], fresh_raw)
    assert np.array_equal(obj.cumulative_sum_memmap, sum_before)
    assert np.array_equal(obj.cumulative_wht_memmap, wht_before)


def test_phi_witness_drizzle_direct_source_resolution_sequence(caplog, monkeypatch):
    """Standard Drizzle: finalize('divide') HWC stack resized directly for each
    factor; final 1/1 equals a fresh direct 1/1; accumulators unchanged."""
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)
    obj = _drizzle_stack()
    H, W = 64, 96
    _fill_drizzle(obj, (H, W))
    img_before = [a._out_img.copy() for a in obj.drizzle_accumulators]
    wht_before = [a._out_wht.copy() for a in obj.drizzle_accumulators]

    def render(factor):
        obj.preview_downsample_factor = factor
        collected = []
        obj.preview_callback = lambda *a: collected.append(a)
        obj._update_preview_drizzle_accumulator()
        assert len(collected) == 1
        return collected[0][0]

    import cv2

    raws = {}
    for f in (1, 2, 3, 4, 1):
        legacy, raw = render(f)
        raws[f] = raw
        assert legacy.shape == raw.shape
        if f == 1:
            assert raw.shape == (H, W, 3)
        else:
            assert raw.shape == (H // f, W // f, 3)
            expected = cv2.resize(
                raws[1], (W // f, H // f), interpolation=cv2.INTER_AREA
            )
            assert np.array_equal(raw, expected), f"factor {f} not a direct resize"

    fresh_legacy, fresh_raw = render(1)
    assert np.array_equal(raws[1], fresh_raw)
    for acc, bi, bw in zip(obj.drizzle_accumulators, img_before, wht_before):
        assert np.array_equal(acc._out_img, bi)
        assert np.array_equal(acc._out_wht, bw)
