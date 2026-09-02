"""PHI-R2 — debug-gated preview pipeline telemetry + deterministic witnesses.

Covers the PHI-R2 instrumentation contract (``docs/phi_viewer_archaeology.md``
§7) and the required reproduction witnesses:

1. **HDR / strong-WB headroom witness** — a deterministic synthetic Option-A
   payload with over-range raw-linear headroom and a strong WB gain.  R2 used
   it to prove *where* the first ``==1`` plateau appeared in the then-baseline
   pipeline (anchor map); PHI-R3 rewrote it as the repaired-semantics witness:
   the headroom is preserved through ``anchor_mapped``/``wb_only`` into the
   histogram analysis while the display stays bounded at the uint8 boundary,
   and telemetry stays a **no-op when disabled**;
2. **reordered payload / resolution witness** — two valid Option-A payloads
   (same and changed metadata) delivered in adversarial order prove the
   historical last-wins acceptance contract for **unsequenced/legacy**
   payloads (PHI-R3 keeps this as the legacy fallback; sequenced payloads
   obey the monotonic gate — items 11), deterministically, without
   timing/sleeps;
3. **producer-isolation proof** — Classic SUM/W and standard Drizzle producer
   runs (with telemetry enabled) leave scientific accumulators/inputs
   bit-identical;
4. **focused Qt + producer runs** under ``QT_QPA_PLATFORM=offscreen`` validate
   the stage records (source route, factor, dtype, shape, min/p01/median/p99/
   max, source-buffer identity) and their compact/bounded shape.

PHI-REWORK-1 additions:

5. deterministic-sample counter semantics (n / under_n / over_n / zero_n /
   one_n, dtype-aware) for float analysis and uint8 display buffers;
6. producer-to-Qt propagation of the true monotonic preview sequence
   (PREV_SEQ) and effective resolution (PREV_RES) for both active producers,
   with legacy/test fallback; producer traces and payload agree;
7. async histogram_output attribution via a scheduled-time ctx snapshot, and
   renamed display-stage labels (display_input / display_output).

PHI-REWORK-1.3 additions:

8. the debug gate is a genuine no-op: with ZSSS_PHI_TRACE disabled no producer
   creates/mutates ``_phi_preview_seq`` and no PREV_* header card is added;
9. truthful requested-vs-effective resolution metadata: PREV_REQ (requested
   factor) vs PREV_RES (factor actually applied — 1 when the Classic small-
   image guard or a resize failure skips the downsample) plus PREV_CAP for the
   Drizzle max-side display guard; delivered geometry is asserted in tests.

PHI-R3 (behavioural integrity correction) additions:

10. **Headroom preservation witness** — the anchor mapping and the WB
    derivation no longer hard-clip finite out-of-range float signal: HDR /
    strong-WB headroom survives ``raw_source -> anchor_mapped -> wb_only``
    into the histogram analysis (per-channel stats max and the model's
    explicit analysis range ``upper > 1.0``), while the display stays bounded
    at the final rendering boundary (uint8, saturated at 255) — analysis
    headroom is never conflated with display saturation;
11. **Monotonic acceptance gate** — when a payload carries a producer
    ``PREV_SEQ``, Qt drops stale (older) and duplicate (equal) emissions of
    the current run before they can replace analysis/display state or schedule
    work; the gate resets at run start so a new run's first sequenced payload
    is accepted; legacy/unsequenced payloads keep the historical
    unconditional last-wins acceptance (W2 remains the legacy-contract
    witness);
12. **Explicit histogram analysis-domain semantics** — the float model
    declares its preserved analysis range ``(0, upper)``,
    ``upper = max(1.0, finite max)``, with counts/stats on the exact same
    finite non-negative sample; reset view reveals the full analysis range;
    the model/stats/trace are labelled analysis data, never the bounded
    uint8 display histogram.

PHI-R3.1 (analysis-unit BP/WP + float display rendering) additions:

13. **Analysis-unit display controls** — for Option-A float previews the
    black/white points (sliders, spins, histogram markers and drags) operate
    in the preserved analysis units ``[0, upper]`` (grid-ceiling control
    domain), so a white point above ``1`` is a first-class value;
    deterministic reconcile (no inversion) when a new preview's analysis
    range no longer fits; legacy single-array payloads keep the historical
    ``[0, 1]`` domain;
14. **Float display rendering** — the visible Option-A display is rendered
    from the preserved float analysis/WB source with the user BP/WP applied
    in float *before* the final uint8/QImage conversion (the only clipping
    boundary), so a white point above 1 visibly recovers preserved headroom
    instead of remaining white-clipped; the Option-A ``display_input`` trace
    is analysis data (headroom possible) while legacy payloads keep the
    uint8-derived ``[0, 1]`` chain.

PHI-R3.2 (REWORK-R3.2, F1/F2) additions:

15. **Trace-independent producer identity gate** — the active producers stamp
    ``PREV_SEQ``/``PREV_RUN`` (+ resolution cards) on every payload whether or
    not ``ZSSS_PHI_TRACE`` is set (only PREVIEW_STAGE records are gated); the
    Qt acceptance rule additionally rejects *foreign* payloads whose producer
    run/session id differs from the id bound to the current run at start, so a
    late old-run payload cannot poison the new run's sequence high-water mark
    in either arrival order (the R2 gate-off producer tests were rewritten to
    the new no-trace contract);
16. **Frozen-view reconciliation** — ``HistogramView.set_model`` revalidates a
    frozen/manual view range against the new model analysis domain (valid
    manual range preserved; out-of-domain window clamped; degenerate fallback
    to the full range), keeping inline and detached surfaces synchronized.

PHI-R3.3 (REWORK-R3.3, final corrective iteration, F2/F3) additions:

17. **Detached view-policy mirroring** — the detached histogram surface mirrors
    the inline view *policy* (window + frozen-vs-unfrozen + auto-zoom) via
    ``mirror_state_from`` whenever it is synchronized, after every model
    application and after legacy data installation: an unfrozen inline view
    never creates an artificial detached frozen range (Nono open-after-reset
    divergence closed) and a genuine manual zoom is reconciled identically on
    both surfaces;
18. **Model→legacy view reset + real-backend stamping** — ``set_data`` /
    ``set_legacy_data`` clear frozen float-model view state on a model→legacy
    transition and restore a valid legacy ``[0, 1]`` window automatically
    (legacy→legacy refreshes keep the manual-zoom contract); the real
    ``SeestarQueuedStackerBackend`` construction/stamping path is covered end
    to end under trace off (assigned session reaches the backend-created
    stacker; the active producer's callback header carries ``PREV_RUN`` +
    ``PREV_SEQ`` with zero PREVIEW_STAGE records; reused-backend re-stamping
    covered).

PHI-R4 (route reachability & integrity parity) additions:

19. **Live route/dispatch inventory** — Classic (SUM/W, non-drizzle per-batch +
    refresh), standard Drizzle (single-accumulator per-pose/per-group +
    refresh) and the legacy incremental Drizzle preview are mapped to their
    exact live dispatch predicates (``refresh_preview`` session predicate,
    ``_drizzle_group_tick``/``_drizzle_flush_partial_group`` policy cadence);
    the reproject/master coadd carrier is a guarded no-op;
20. **Dead/guarded legacy dispatch proof (R4)** — the legacy incremental
    Drizzle producer (cumulative-display-data based) was unreachable: an
    AST-based static guard proved every call to its dead subgraph lived inside
    the subgraph itself, and runtime spies across supported dispatch never
    invoked it (recorded and tested — no speculative removal before the human
    gate).

PHI-R5 (approved retirement) additions:

21. **Legacy machinery retired** — after Tristan's explicit approval of the R5
    human gate, the M3-D OBSOLETE LEGACY incremental-Drizzle
    preview/process/carrier chain and the dead reproject/master preview
    carrier were removed from ``queue_manager.py`` (methods, module helpers,
    sole-purpose state incl. the retired ``drizzle_executor``, dead imports);
    ``boring_stack._cleanup_stacker`` drains only the quality executor;
    ``test_phi_r5_legacy_machinery_retired_no_supported_dispatch_invokes_it``
    proves the retired symbols are absent and that Classic / standard Drizzle
    dispatch still refreshes safely with no speculative fallback to the
    deleted routes; the forensic ``_process_incremental_drizzle_batch``
    coverage in ``test_save_final_stack.py`` was migrated to a positive
    supported-path invariant (active standard-Drizzle preview identity + final
    FITS science preservation, trace off).

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
from PySide6.QtGui import QImage

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

def test_phi_r3_witness_hdr_strong_wb_headroom_preserved_to_histogram(qapp, caplog, monkeypatch):
    """PHI-R3 headroom witness: HDR/strong-WB float headroom is preserved
    through the analysis path and reaches the histogram; the display stays
    bounded at the final rendering boundary.

    Post-R3 repaired semantics asserted (the R2 diagnostic that the first
    ``==1.0`` plateau appears at ``anchor_mapped`` no longer holds):
    * raw_source carries max > 1 (headroom exists at the raw-linear boundary);
    * anchor_mapped keeps over_n > 0 and max > 1 — the anchor mapping does NOT
      clip the bright tail any more;
    * wb_only (with a strong 3x red gain) keeps max > 1 and over_n > 0 — WB
      does not clip either; no ``==1.0`` plateau is fabricated;
    * histogram_output reports an analysis-domain per-channel max > 1
      (``R_max``) — the headroom reached the histogram analysis and is no
      longer indistinguishable from display saturation;
    * display_output is the bounded uint8 screen domain: max == 255 and the
      saturated-pixel counter (one_n, ``== 255``) is > 0 — saturation exists
      only at the display boundary.
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
        assert int(raw_recs[0]["over_n"]) > 0

        # anchor_mapped: headroom PRESERVED (no clip) — over_n stays > 0 and
        # the max is still > 1.0 (the R2 first-plateau site is gone).
        mapped_recs = by_stage.get("anchor_mapped", [])
        assert mapped_recs, "missing anchor_mapped trace"
        assert float(mapped_recs[0]["max"]) > 1.0, (
            "anchor map must preserve headroom (no hard clip)"
        )
        assert int(mapped_recs[0]["over_n"]) > 0
        # No fabricated exact-one plateau: the preserved tail is > 1, not == 1.
        assert int(mapped_recs[0]["one_n"]) == 0 or int(mapped_recs[0]["one_n"]) < int(
            mapped_recs[0]["over_n"]
        )

        # wb_only: 3x red gain keeps/extends headroom (no WB clip).
        wb_recs = by_stage.get("wb_only", [])
        assert wb_recs, "missing wb_only trace"
        assert float(wb_recs[0]["max"]) > 1.0, "WB must not clip headroom"
        assert int(wb_recs[0]["over_n"]) > 0

        # The analysis float stages never fabricate a ==1.0 plateau between the
        # raw source and the WB-only buffer.
        for stage in ("raw_source", "anchor_mapped", "wb_only"):
            f = by_stage[stage][0]
            assert float(f["max"]) > 1.0, f"{stage} max should stay > 1.0"

        # Histogram output: analysis-domain per-channel max > 1 — the headroom
        # reached the histogram analysis (distinguishable from display
        # saturation, which is bounded at uint8).
        hist_recs = by_stage.get("histogram_output", [])
        assert hist_recs, "missing histogram_output trace"
        r_max = float(hist_recs[0]["R_max"])
        assert r_max > 1.0, f"histogram R_max should show headroom, got {r_max}"
        assert float(win._histogram_model["stats"]["R"]["max"]) == pytest.approx(
            r_max, rel=1e-5
        )

        # Display output is the bounded uint8 screen domain: saturated at 255.
        disp_recs = by_stage.get("display_output", [])
        assert disp_recs, "missing display_output trace"
        assert disp_recs[0]["dtype"] == "uint8"
        assert int(disp_recs[0]["max"]) == 255
        assert int(disp_recs[0]["one_n"]) > 0  # 255-saturated display pixels
        # The stored display source is an 8-bit QImage by construction.
        assert win._preview_source is not None
        assert win._preview_source.format() in (
            QImage.Format.Format_Grayscale8,
            QImage.Format.Format_RGB888,
            QImage.Format.Format_RGBA8888,
        )
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
    """Legacy-contract witness (PHI-R3): unsequenced payloads keep last-wins.

    Payloads without a producer ``PREV_SEQ`` (legacy producers / test
    payloads) bypass the PHI-R3 monotonic gate and retain the historical
    unconditional acceptance: the last delivered valid payload replaces the
    source.  The sequenced-emission contract (stale/duplicate rejection) is
    covered by the dedicated ``test_phi_r3_seq_gate_*`` tests.

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


# ---------------------------------------------------------------------------
# REWORK-1 — deterministic-sample counters (item 1)
# ---------------------------------------------------------------------------

def test_phi_trace_counters_float_domain(caplog, monkeypatch):
    """Float [0,1] domain counters: n / under_n / over_n / zero_n / one_n are
    exact on a small hand-built array (no subsampling, stride == 1)."""
    import logging as _logging

    from seestar.utils.phi_trace import phi_trace_stage

    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(_logging.DEBUG)
    arr = np.array(
        [[-2.0, -0.5, 0.0, 0.0, 0.25, 1.0, 1.0, 1.0, 1.5, 3.0]],
        dtype=np.float64,
    )
    phi_trace_stage(
        _logging.getLogger("phi.test"), route="classic", stage="source", arr=arr
    )
    records = _parse_stage_records(caplog)
    assert len(records) == 1
    f = _fields(records[0])
    assert f["n"] == "10"
    assert f["under_n"] == "2"  # -2.0, -0.5
    assert f["over_n"] == "2"  # 1.5, 3.0
    assert f["zero_n"] == "2"  # two exact zeros
    assert f["one_n"] == "3"  # three exact ones


def test_phi_trace_counters_uint8_display(caplog, monkeypatch):
    """uint8 display buffers: one_n counts saturated == 255 pixels; under/over
    are always 0 (dtype range bounds), n is the finite sample size."""
    import logging as _logging

    from seestar.utils.phi_trace import phi_trace_stage

    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(_logging.DEBUG)
    arr = np.array([[0, 0, 1, 127, 255, 255]], dtype=np.uint8)
    phi_trace_stage(
        _logging.getLogger("phi.test"), route="qt", stage="display_output", arr=arr
    )
    records = _parse_stage_records(caplog)
    assert len(records) == 1
    f = _fields(records[0])
    assert f["dtype"] == "uint8"
    assert f["n"] == "6"
    assert f["under_n"] == "0"
    assert f["over_n"] == "0"
    assert f["zero_n"] == "2"  # two exact zeros
    assert f["one_n"] == "2"  # two saturated == 255 pixels


def test_phi_trace_counters_absent_when_disabled(caplog, monkeypatch):
    """Gate off: no PREVIEW_STAGE record at all (counters or otherwise)."""
    import logging as _logging

    from seestar.utils.phi_trace import phi_trace_stage

    monkeypatch.delenv("ZSSS_PHI_TRACE", raising=False)
    caplog.set_level(_logging.DEBUG)
    arr = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    phi_trace_stage(
        _logging.getLogger("phi.test"), route="classic", stage="source", arr=arr
    )
    assert _parse_stage_records(caplog) == []


# ---------------------------------------------------------------------------
# REWORK-1 — producer PREV_SEQ / PREV_RES propagation (item 2)
# ---------------------------------------------------------------------------

def _classic_with_sumw(H=16, W=24):
    obj = _classic_stack()
    avg = np.linspace(0.0, 1.0, H * W * 3, dtype=np.float32).reshape(H, W, 3)
    obj.cumulative_sum_memmap = avg.astype(np.float32)
    obj.cumulative_wht_memmap = np.ones((H, W), dtype=np.float32)
    return obj


def test_phi_producer_seq_header_and_trace_agree_classic(caplog, monkeypatch):
    """Classic producer (normal-size image): header PREV_SEQ is monotonic across
    emissions, equals the trace seq, PREV_REQ is the requested factor, and
    PREV_RES matches the actually-applied factor and the delivered geometry."""
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)
    obj = _classic_with_sumw(H=64, W=96)  # big enough: all resizes run
    collected = []
    obj.preview_callback = lambda *a: collected.append(a)

    obj._update_preview_sum_w(downsample_factor=2)
    obj._update_preview_sum_w(downsample_factor=1)
    obj._update_preview_sum_w(downsample_factor=3)

    assert len(collected) == 3
    seqs = []
    for args in collected:
        header = args[1]
        seqs.append(int(header["PREV_SEQ"]))
        assert "PREV_REQ" in header
        assert "PREV_RES" in header
    assert seqs == [1, 2, 3]  # strictly monotonic per emission

    # Requested/effective factors are truthful and delivered geometry agrees.
    for args, req in zip(collected, (2, 1, 3)):
        header = args[1]
        assert int(header["PREV_REQ"]) == req
        if req > 1:
            assert int(header["PREV_RES"]) == req
            assert args[0][0].shape == (64 // req, 96 // req, 3)
            assert args[0][1].shape == (64 // req, 96 // req, 3)
        else:
            assert int(header["PREV_RES"]) == 1
            assert args[0][0].shape == (64, 96, 3)

    # Trace records for the same emissions carry the same seq values.
    records = _parse_stage_records(caplog)
    trace_seqs = sorted(int(_fields(m)["seq"]) for m in records)
    assert trace_seqs == [1, 1, 1, 2, 2, 2, 3, 3, 3]  # 3 stages per emission

    # Effective factor in the post_resize trace matches PREV_RES on the payload.
    post = [_fields(m) for m in records if _fields(m)["stage"] == "post_resize"]
    assert len(post) == 3
    for f, args, req in zip(post, collected, (2, 1, 3)):
        assert int(f["factor"]) == int(args[1]["PREV_RES"])
        assert int(f["req"]) == req


def test_phi_producer_seq_header_and_trace_agree_drizzle(caplog, monkeypatch):
    """Drizzle producer: same monotonic PREV_SEQ / PREV_REQ / PREV_RES contract
    with truthful delivered geometry (no cap for a small fixture)."""
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)
    obj = _drizzle_stack()
    _fill_drizzle(obj, (64, 96))
    collected = []
    obj.preview_callback = lambda *a: collected.append(a)

    obj.preview_downsample_factor = 1
    obj._update_preview_drizzle_accumulator()
    obj.preview_downsample_factor = 2
    obj._update_preview_drizzle_accumulator()

    assert len(collected) == 2
    seqs = [int(a[1]["PREV_SEQ"]) for a in collected]
    assert seqs == [1, 2]
    assert [int(a[1]["PREV_REQ"]) for a in collected] == [1, 2]
    assert [int(a[1]["PREV_RES"]) for a in collected] == [1, 2]
    assert [int(a[1]["PREV_CAP"]) for a in collected] == [0, 0]
    # Delivered geometry agrees with the effective factor.
    assert collected[0][0][0].shape == (64, 96, 3)
    assert collected[1][0][0].shape == (32, 48, 3)

    records = _parse_stage_records(caplog)
    trace_seqs = sorted(int(_fields(m)["seq"]) for m in records)
    assert trace_seqs == [1, 1, 1, 2, 2, 2]
    post = [_fields(m) for m in records if _fields(m)["stage"] == "post_resize"]
    for f, args in zip(post, collected):
        assert int(f["factor"]) == int(args[1]["PREV_RES"])
        assert int(f["req"]) == int(args[1]["PREV_REQ"])


# ---------------------------------------------------------------------------
# REWORK-1.3 — gate is a genuine no-op (Defect A)
# ---------------------------------------------------------------------------

def test_phi_r32_producer_identity_metadata_present_trace_off_classic(caplog, monkeypatch):
    """PHI-R3.2 (F1): with ZSSS_PHI_TRACE disabled the Classic producer still
    stamps the required payload ordering/run identity metadata (PREV_SEQ,
    PREV_RUN, PREV_REQ, PREV_RES) on every payload — the Qt acceptance gate is
    a production correction, not debug telemetry.  Only the PREVIEW_STAGE
    debug records are suppressed (telemetry stays a no-op)."""
    monkeypatch.delenv("ZSSS_PHI_TRACE", raising=False)
    caplog.set_level(logging.DEBUG)
    assert not phi_trace_enabled()
    obj = _classic_with_sumw(H=64, W=96)
    collected = []
    obj.preview_callback = lambda *a: collected.append(a)
    obj._update_preview_sum_w(downsample_factor=2)
    obj._update_preview_sum_w(downsample_factor=1)

    assert len(collected) == 2
    # Required display metadata is present and consistent across emissions.
    seqs = [int(a[1]["PREV_SEQ"]) for a in collected]
    assert seqs == [1, 2]  # per-emission monotonic sequence
    runs = {int(a[1]["PREV_RUN"]) for a in collected}
    assert len(runs) == 1  # one durable producer run/session per stacker
    for args in collected:
        header = args[1]
        assert "PREV_REQ" in header and "PREV_RES" in header
        assert "PREV_CAP" not in header  # Classic route has no cap card
    # Debug telemetry is still a genuine no-op with the gate off.
    assert _parse_stage_records(caplog) == []


def test_phi_r32_producer_identity_metadata_present_trace_off_drizzle(caplog, monkeypatch):
    """Same PHI-R3.2 contract for the standard Drizzle producer (PREV_CAP
    included); two stacker instances get distinct durable run/session ids."""
    monkeypatch.delenv("ZSSS_PHI_TRACE", raising=False)
    caplog.set_level(logging.DEBUG)
    assert not phi_trace_enabled()
    obj = _drizzle_stack()
    _fill_drizzle(obj, (64, 96))
    collected = []
    obj.preview_callback = lambda *a: collected.append(a)
    obj.preview_downsample_factor = 2
    obj._update_preview_drizzle_accumulator()
    obj._update_preview_drizzle_accumulator()

    assert len(collected) == 2
    seqs = [int(a[1]["PREV_SEQ"]) for a in collected]
    assert seqs == [1, 2]
    runs = {int(a[1]["PREV_RUN"]) for a in collected}
    assert len(runs) == 1
    for args in collected:
        header = args[1]
        for key in ("PREV_SEQ", "PREV_RUN", "PREV_REQ", "PREV_RES", "PREV_CAP"):
            assert key in header
    assert _parse_stage_records(caplog) == []

    # A second stacker instance (a new run/session) gets a distinct run id.
    obj2 = _drizzle_stack()
    _fill_drizzle(obj2, (64, 96))
    collected2 = []
    obj2.preview_callback = lambda *a: collected2.append(a)
    obj2.preview_downsample_factor = 1
    obj2._update_preview_drizzle_accumulator()
    assert int(collected2[0][1]["PREV_RUN"]) not in runs


# ---------------------------------------------------------------------------
# REWORK-1.3 — truthful requested vs effective resolution (Defect B)
# ---------------------------------------------------------------------------

def test_phi_classic_small_image_no_resize_truthful(caplog, monkeypatch):
    """Classic with a too-small image: the requested factor 2 is skipped by the
    new_h/new_w > 10 guard, so PREV_REQ=2 but PREV_RES=1 and the delivered
    payload is full-resolution — the metadata must not lie about it."""
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)
    obj = _classic_with_sumw(H=16, W=24)  # 16//2=8 <= 10 -> guard skips resize
    collected = []
    obj.preview_callback = lambda *a: collected.append(a)
    obj._update_preview_sum_w(downsample_factor=2)

    assert len(collected) == 1
    header = collected[0][1]
    assert int(header["PREV_REQ"]) == 2
    assert int(header["PREV_RES"]) == 1  # resize skipped
    assert collected[0][0][0].shape == (16, 24, 3)  # full-res delivered
    assert collected[0][0][1].shape == (16, 24, 3)

    post = [
        _fields(m)
        for m in _parse_stage_records(caplog)
        if _fields(m)["stage"] == "post_resize"
    ]
    assert len(post) == 1
    assert int(post[0]["factor"]) == 1
    assert int(post[0]["req"]) == 2
    assert post[0]["shape"] == "16x24x3"


def test_phi_drizzle_precap_metadata_truthful(caplog, monkeypatch):
    """Drizzle with the max-side cap firing (bounded fixture): PREV_CAP=1, the
    delivered geometry reflects the cap-then-factor chain, and PREV_RES matches
    the final delivered geometry."""
    import seestar.queuep.queue_manager as qm

    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    monkeypatch.setattr(qm, "_MAX_PREVIEW_SIDE_PX", 40)  # force the cap
    caplog.set_level(logging.DEBUG)
    obj = _drizzle_stack()
    _fill_drizzle(obj, (64, 96))  # max side 96 > 40 -> cap fires
    collected = []
    obj.preview_callback = lambda *a: collected.append(a)

    obj.preview_downsample_factor = 1
    obj._update_preview_drizzle_accumulator()
    # cap: scale=40/96 -> (int(96*40/96), int(64*40/96)) = (40, 26)
    assert collected[-1][1]["PREV_CAP"] == 1
    assert collected[-1][0][0].shape == (26, 40, 3)
    assert int(collected[-1][1]["PREV_REQ"]) == 1
    assert int(collected[-1][1]["PREV_RES"]) == 1

    obj.preview_downsample_factor = 2
    obj._update_preview_drizzle_accumulator()
    # cap then factor 2: (26//2, 40//2) = (13, 20)
    assert collected[-1][1]["PREV_CAP"] == 1
    assert collected[-1][0][0].shape == (13, 20, 3)
    assert int(collected[-1][1]["PREV_REQ"]) == 2
    assert int(collected[-1][1]["PREV_RES"]) == 2

    post = [
        _fields(m)
        for m in _parse_stage_records(caplog)
        if _fields(m)["stage"] == "post_resize"
    ]
    assert len(post) == 2
    assert post[0]["shape"] == "26x40x3"
    assert post[1]["shape"] == "13x20x3"
    assert post[1]["cap"] == "1"


def test_phi_qt_payload_seq_res_propagation_and_fallback(qapp, caplog, monkeypatch):
    """Qt payload_arrive: pseq/preq/pres/pcap come from the header when present;
    the fallback (image_count/current_batch) is used when the header lacks them."""
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)
    win = MainWindow()
    try:
        raw = _hdr_raw(size=32, seed=3)
        # Header-carrying payload: producer metadata wins over image_count.
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(raw), raw),
                stack_name="meta",
                header={
                    "PREV_SRC": _SUMW_PREVIEW_SRC,
                    "PREV_SEQ": 17,
                    "PREV_REQ": 2,
                    "PREV_RES": 1,
                    "PREV_CAP": 0,
                },
                image_count=99,
                current_batch=5,
            )
        )
        # Legacy-style payload: no PREV_* -> fallback to counters.
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(raw), raw),
                stack_name="legacy",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
                image_count=7,
                current_batch=3,
            )
        )

        arrivals = [
            _fields(m)
            for m in _parse_stage_records(caplog)
            if _fields(m)["stage"] == "payload_arrive"
        ]
        assert len(arrivals) == 2
        assert arrivals[0]["pseq"] == "17"
        assert arrivals[0]["preq"] == "2"
        assert arrivals[0]["pres"] == "1"
        assert arrivals[0]["pcap"] == "0"
        assert arrivals[0]["seq"] == "17"  # producer seq, not image_count
        assert arrivals[1]["pseq"] == "-"
        assert arrivals[1]["preq"] == "-"
        assert arrivals[1]["pres"] == "-"
        assert arrivals[1]["pcap"] == "-"
        assert arrivals[1]["seq"] == "7"  # fallback to image_count
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# REWORK-1 — stale histogram context snapshot (item 3)
# ---------------------------------------------------------------------------

def test_phi_histogram_output_uses_scheduled_ctx_snapshot(qapp, caplog, monkeypatch):
    """histogram_output attribution: the record uses the ctx snapshot taken at
    schedule time — a later change of ``_phi_trace_ctx`` without a re-schedule
    (e.g. a payload that updates the context but fails to ingest / does not
    re-derive the WB buffer) must NOT leak into an in-flight result.

    Deterministic: deliver payload A (schedules with A's ctx), then overwrite
    the live ctx with B-like values without touching the scheduled revision,
    then apply a result for A's revision — the record must carry A's pseq."""
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)
    win = MainWindow()
    try:
        raw_a = _hdr_raw(size=32, seed=4)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(raw_a), raw_a),
                stack_name="A",
                header={
                    "PREV_SRC": _SUMW_PREVIEW_SRC,
                    "PREV_SEQ": 1,
                    "PREV_RES": 1,
                },
            )
        )
        # A's schedule exists with A's snapshot.
        assert win._histogram_trace_ctx is not None
        assert win._histogram_trace_ctx.get("pseq") == "1"
        scheduled_revision = win._wb_only_revision

        # Simulate a payload arrival that changes the live context but does NOT
        # re-schedule (no WB re-derivation — e.g. an unusable/legacy payload
        # that updates ``_phi_trace_ctx`` in ``_on_preview`` before ingest).
        win._phi_trace_ctx = {
            "src": _SUMW_PREVIEW_SRC,
            "identity": "batch:9",
            "res": "x1/1",
            "shape": "16x16x3",
            "pseq": "42",
            "pres": "4",
        }
        assert win._histogram_trace_ctx.get("pseq") == "1"  # snapshot intact

        # Apply a synthetic result for the revision that was scheduled for A.
        win._histogram_model = None
        win._histogram_model_revision = None
        fake_result = {
            "bins": 512,
            "channels": ["L"],
            "stats": {
                "L": {"min": 0.0, "max": 1.0, "median": 0.5, "mean": 0.5, "std": 0.3}
            },
        }
        win._on_histogram_result(
            0, fake_result, (win._analysis_generation, scheduled_revision)
        )
        hist_recs = [
            _fields(m)
            for m in _parse_stage_records(caplog)
            if _fields(m)["stage"] == "histogram_output"
        ]
        assert hist_recs, "missing histogram_output record"
        assert hist_recs[-1]["pseq"] == "1"  # A's snapshot, not the live 42
        assert hist_recs[-1]["identity"] == "none"  # A had no identity
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# REWORK-1 — display-stage attribution (item 3)
# ---------------------------------------------------------------------------

def test_phi_display_stage_labels_are_display_domain(qapp, caplog, monkeypatch):
    """display_input/display_output attribution (PHI-R3.1): the Option-A
    float display chain feeds the preserved float analysis/WB buffer (with
    headroom > 1 possible) into display_input and emits the bounded uint8
    display_output (one_n == 255 saturated count); the legacy QImage chain
    (single-array payloads) keeps a uint8-derived [0,1] display_input."""
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)
    win = MainWindow()
    try:
        # --- Option-A payload: float display chain -------------------------
        raw = _hdr_raw(size=32, seed=6)
        win._wb = (1.0, 1.0, 1.0)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(raw), raw),
                stack_name="disp",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        # Force a non-identity display adjustment so the tone chain runs.
        win._black_point = 0.1
        win._refresh_preview_view()

        by_stage = {}
        for m in _parse_stage_records(caplog):
            f = _fields(m)
            by_stage.setdefault(f["stage"], []).append(f)

        # Option-A display_input is the preserved float analysis/WB buffer
        # entering the tone chain — it may carry headroom > 1 (it is NOT a
        # pre-quantized uint8-derived [0,1] buffer).
        disp_in = by_stage.get("display_input", [])
        assert disp_in, "missing display_input record"
        assert float(disp_in[0]["max"]) > 1.0  # analysis headroom preserved
        assert int(disp_in[0]["over_n"]) > 0
        assert "stretch" in disp_in[0]

        disp_out = by_stage.get("display_output", [])
        assert disp_out, "missing display_output record"
        assert disp_out[0]["dtype"] == "uint8"
        assert int(disp_out[0]["over_n"]) == 0  # uint8: no >255 values
        assert int(disp_out[0]["max"]) == 255
        # one_n counts the saturated == 255 display pixels (uint8 domain); with
        # the white point still at 0.99 the HDR red tail saturates there.
        assert int(disp_out[0]["one_n"]) > 0

        # --- Legacy single-array payload: QImage-derived display chain -----
        caplog.clear()
        legacy = _legacy_normalize(raw)
        win._on_preview(
            BackendPreviewPayload(
                data=legacy, stack_name="legacy-disp"  # single array, not Option-A
            )
        )
        by_stage2 = {}
        for m in _parse_stage_records(caplog):
            f = _fields(m)
            by_stage2.setdefault(f["stage"], []).append(f)
        disp_in2 = by_stage2.get("display_input", [])
        assert disp_in2, "missing legacy display_input record"
        assert float(disp_in2[0]["max"]) <= 1.0  # uint8-derived display domain
        disp_out2 = by_stage2.get("display_output", [])
        assert disp_out2, "missing legacy display_output record"
        assert disp_out2[0]["dtype"] == "uint8"
        assert int(disp_out2[0]["max"]) == 255
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# PHI-R3 — monotonic producer-sequence acceptance gate (Qt arrival)
# ---------------------------------------------------------------------------

def _seq_payload(raw, seq, batch=1, name="seq", image_count=None, prev_src=_SUMW_PREVIEW_SRC, run=None):
    """Option-A payload carrying producer identity metadata (PREV_SEQ and,
    when ``run`` is given, PREV_RUN) exactly as the active producers stamp
    them on every emission (PHI-R3.2 — independent of the trace gate)."""
    header = {
        "PREV_SRC": prev_src,
        "PREV_SEQ": seq,
        "PREV_REQ": 1,
        "PREV_RES": 1,
    }
    if run is not None:
        header["PREV_RUN"] = run
    return BackendPreviewPayload(
        data=(_legacy_normalize(raw), raw),
        stack_name=name,
        header=header,
        image_count=image_count if image_count is not None else batch,
        current_batch=batch,
    )


def test_phi_r3_seq_gate_drops_stale_and_duplicate_within_run(qapp):
    """A stale (older) or duplicate (equal) producer-sequenced emission is
    dropped at Qt arrival: it must not replace the displayed identity, must not
    re-ingest analysis state, and must not schedule histogram work."""
    win = MainWindow()
    try:
        # First sequenced payload of the run: accepted (gate starts open).
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=1), seq=5, batch=5))
        assert win._displayed_identity[1:] == ("batch", 5)
        assert win._last_accepted_preview_seq == 5
        revision_after_first = win._wb_only_revision

        # Newer emission: accepted.
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=2), seq=7, batch=7))
        assert win._displayed_identity[1:] == ("batch", 7)
        assert win._last_accepted_preview_seq == 7

        # Stale emission (seq 6 < 7): dropped — display keeps batch 7.
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=3), seq=6, batch=6))
        assert win._displayed_identity[1:] == ("batch", 7), "stale payload applied!"
        assert win._last_accepted_preview_seq == 7

        # Duplicate emission (seq 7 == 7): dropped even with different content.
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=4), seq=7, batch=99))
        assert win._displayed_identity[1:] == ("batch", 7), "duplicate payload applied!"
        assert win._last_accepted_preview_seq == 7

        # No analysis re-ingestion / no histogram scheduling happened for the
        # dropped payloads (the WB-only revision and the scheduled revision
        # stayed at the last accepted payload's values).
        assert win._wb_only_revision == revision_after_first + 1
    finally:
        win.shutdown()


def test_phi_r3_seq_gate_reordered_newer_older_obeys_monotonic_rule(qapp, caplog, monkeypatch):
    """Deterministic reordered sequenced arrivals: newer wins, older is
    refused, and the acceptance record documents the monotonic rule with an
    explicit drop reason (stale vs duplicate)."""
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)
    win = MainWindow()
    try:
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=1), seq=9, batch=9))
        assert win._displayed_identity[1:] == ("batch", 9)

        # Older emission delivered after a newer one: refused (no reorder win).
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=2), seq=3, batch=3))
        assert win._displayed_identity[1:] == ("batch", 9)

        # Then a genuinely newer emission: accepted.
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=3), seq=10, batch=10))
        assert win._displayed_identity[1:] == ("batch", 10)

        # The arrival trace marks each gated drop with its reason.
        drops = [
            _fields(m)
            for m in _parse_stage_records(caplog)
            if _fields(m).get("drop") is not None
        ]
        assert len(drops) == 1
        assert drops[0]["stage"] == "payload_arrive"
        assert drops[0]["drop"] == "stale"
        assert drops[0]["pseq"] == "3"
        assert drops[0]["seq"] == "3"
    finally:
        win.shutdown()


def test_phi_r3_seq_gate_run_reset_accepts_first_payload_of_new_run(qapp):
    """The sequence gate resets at the run boundary: the first sequenced
    payload of a new run (the fresh producer restarts its counter at 1) is
    accepted even though its sequence is not newer than the previous run's
    high-water mark."""
    win = MainWindow()
    try:
        # Run 1: two sequenced payloads accepted (high-water mark 2).
        win._on_run_started()
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=1), seq=1, batch=1))
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=2), seq=2, batch=2))
        assert win._last_accepted_preview_seq == 2
        assert win._displayed_identity[1:] == ("batch", 2)
        run_before = win._run_context_id

        # Run 2 starts: _on_run_started resets the gate.
        win._on_run_started()
        assert win._run_context_id == run_before + 1
        assert win._last_accepted_preview_seq is None

        # Without the reset this would be rejected (1 <= 2); with the reset the
        # new run's first payload is accepted.
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=3), seq=1, batch=1))
        assert win._displayed_identity[1:] == ("batch", 1), (
            "new run's first sequenced payload must not be rejected"
        )
        assert win._last_accepted_preview_seq == 1
    finally:
        win.shutdown()


def test_phi_r3_seq_gate_legacy_unsequenced_payloads_bypass(qapp):
    """Legacy/unsequenced payloads (no PREV_SEQ) bypass the gate entirely and
    keep the historical unconditional last-wins acceptance, mixed with
    sequenced payloads."""
    win = MainWindow()
    try:
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=1), seq=5, batch=5))
        assert win._displayed_identity[1:] == ("batch", 5)
        assert win._last_accepted_preview_seq == 5

        # Unsequenced legacy payload: accepted unconditionally (even though its
        # Qt counters are 'older' than the gate's high-water mark).
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(_hdr_raw(size=32, seed=6)), _hdr_raw(size=32, seed=6)),
                stack_name="legacy",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
                image_count=3,
                current_batch=3,
            )
        )
        assert win._displayed_identity[1:] == ("batch", 3), (
            "legacy payload must keep unconditional acceptance"
        )
        # The gate high-water mark is untouched by unsequenced payloads: a
        # stale sequenced emission is still refused.
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=7), seq=4, batch=4))
        assert win._displayed_identity[1:] == ("batch", 3), (
            "stale sequenced payload must still be dropped after a legacy one"
        )
        assert win._last_accepted_preview_seq == 5

        # A newer sequenced emission is accepted again.
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=8), seq=6, batch=6))
        assert win._displayed_identity[1:] == ("batch", 6)
    finally:
        win.shutdown()


def test_phi_r3_seq_gate_is_metadata_driven_not_trace_gated(qapp, caplog, monkeypatch):
    """The acceptance gate is driven by the presence of the PREV_SEQ header
    card, not by the ZSSS_PHI_TRACE debug gate: with tracing disabled a
    sequenced payload is still dropped (silently, no record) and an
    unsequenced one is still accepted."""
    monkeypatch.delenv("ZSSS_PHI_TRACE", raising=False)
    caplog.set_level(logging.DEBUG)
    assert not phi_trace_enabled()
    win = MainWindow()
    try:
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=1), seq=2, batch=2))
        assert win._displayed_identity[1:] == ("batch", 2)

        # Stale duplicate of the same emission with trace disabled: dropped and
        # no PREVIEW_STAGE record is produced (gate-off no-op guarantee).
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=2), seq=2, batch=99))
        assert win._displayed_identity[1:] == ("batch", 2)
        assert _parse_stage_records(caplog) == []
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# PHI-R3 — headroom reaches the histogram analysis; display stays bounded;
#          view range reveals the preserved analysis domain
# ---------------------------------------------------------------------------

def test_phi_r3_hdr_headroom_reaches_histogram_analysis_and_view(qapp):
    """Pipeline regression (criterion F): an HDR raw with a strong WB gain ends
    with analysis headroom above 1.0 in the pristine/WB-only buffers AND in the
    applied histogram model (stats max, explicit range upper); the float model
    is analysis data (range upper > 1.0) while the rendered display QImage is
    the bounded uint8 screen domain."""
    win = MainWindow()
    try:
        raw = _hdr_raw(size=48, seed=21)
        win._wb = (3.0, 1.0, 1.0)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(raw), raw),
                stack_name="r3-hdr",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)

        # Analysis buffers keep finite headroom above the display window.
        pristine = win._pristine_float
        wb_only = win._wb_only_float
        assert pristine is not None and wb_only is not None
        assert float(np.max(pristine)) > 1.0
        assert float(np.max(wb_only)) > 1.0

        # The applied histogram model is analysis data over the preserved
        # range: upper = max(1.0, finite max) > 1.0 and the R stats see the
        # headroom (this is what distinguishes analysis from display
        # saturation — the uint8 display histogram can never exceed 255).
        model = win._histogram_model
        assert model is not None
        assert model["range"][0] == 0.0
        assert float(model["range"][1]) > 1.0
        assert float(model["full_range"][1]) == float(model["range"][1])
        assert float(model["stats"]["R"]["max"]) > 1.0
        assert float(model["stats"]["R"]["max"]) == pytest.approx(
            float(np.max(wb_only[..., 0]))
        )
        # Headroom pixels are counted in bins above the display window.
        counts_r = model["counts"]["R"]
        assert int(counts_r.sum()) == int(wb_only[..., 0].size)
        upper = float(model["range"][1])
        display_top_bin = int(1.0 / upper * 512)
        assert int(counts_r[display_top_bin + 1 :].sum()) > 0

        # View semantics: initial window is the display level [0, 1]; the reset
        # action reveals the full preserved analysis range (0, upper).
        view = win.right_histogram_view
        assert view._model_range == (0.0, upper)
        assert view.view_range == (0.0, 1.0)
        view.reset_histogram_view()
        assert view.view_range[0] == 0.0
        assert view.view_range[1] == pytest.approx(upper)

        # Display remains bounded: the stored display source is an 8-bit QImage
        # (the render boundary clips the mapped float to uint8).
        assert win._preview_source is not None
        assert win._preview_source.format() in (
            QImage.Format.Format_Grayscale8,
            QImage.Format.Format_RGB888,
            QImage.Format.Format_RGBA8888,
        )
        # And the full display-adjustment chain output stays uint8-bounded.
        from seestar.gui_qt.preview_adjust import apply_preview_adjustments

        adjusted = apply_preview_adjustments(
            win._preview_source, wb=win._wb, stretch="asinh"
        )
        assert adjusted is not None
        assert adjusted.format() in (
            QImage.Format.Format_Grayscale8,
            QImage.Format.Format_RGB888,
            QImage.Format.Format_RGBA8888,
        )
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# PHI-R3.1 — analysis-unit BP/WP display controls + float display rendering
# ---------------------------------------------------------------------------

def _hdr_raw_with_stars(size: int = 48, seed: int = 21, star_frac: float = 0.02, star_boost: float = 4.0) -> np.ndarray:
    """Deterministic raw-linear HDR RGB with a few bright 'star' pixels.

    The bulk is ``g ~ U(0, 2)`` (red 1.4x, blue = g) plus ``star_frac`` of
    pixels boosted by ``star_boost``, so the raw max (and the preserved mapped
    headroom) sits far above the p95 anchor — a deterministic stand-in for the
    astro case where the top few percent of pixels carry display headroom.
    """
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.0, 2.0, size=(size, size))
    stars = rng.random((size, size)) < star_frac
    g = np.where(stars, g * star_boost, g)
    r = g * 1.4
    b = g.copy()
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _qimage_to_array(img) -> np.ndarray:
    """Read a QImage back into a uint8 (H, W, C) array for pixel assertions."""
    import seestar.gui_qt.preview_adjust as _pa

    np_mod = _pa._load_numpy()
    arr = _pa._image_to_array(np_mod, img)
    assert arr is not None
    return arr



def _grid_ceil(value: float) -> float:
    """Grid ceiling of an analysis-upper value (0.001 control resolution)."""
    import math

    return math.ceil(value * 1000.0) / 1000.0

def test_phi_r31_histogram_range_exceeds_one_and_controls_scope(qapp):
    """R3.1-E(i): the applied histogram range exceeds 1 for an HDR Option-A
    payload and the BP/WP controls are re-scoped to the same analysis upper."""
    win = MainWindow()
    try:
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(_hdr_raw_with_stars()), _hdr_raw_with_stars()),
                stack_name="r31",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        upper = win._analysis_upper
        assert upper > 1.0, f"analysis upper should exceed 1, got {upper}"
        model = win._histogram_model
        # Model range upper == the raw analysis upper (identical computation).
        assert model["range"] == (0.0, pytest.approx(upper))
        # Controls operate in the analysis units: the spin/slider maxima are
        # the grid ceiling of the analysis upper (the 0.001 control grid).
        grid_upper = _grid_ceil(upper)
        assert win.stretch_wp_spin.maximum() == pytest.approx(grid_upper)
        assert win.stretch_bp_spin.maximum() == pytest.approx(grid_upper)
        assert win.stretch_wp_slider.maximum() == int(round(grid_upper / 0.001))
        # Defaults are kept when they still fit (no spurious reconcile).
        assert win._black_point == 0.01
        assert win._white_point == 0.99
    finally:
        win.shutdown()


def test_phi_r31_wp_above_one_set_and_dragged_synced_everywhere(qapp):
    """R3.1-E(ii): a white point above 1 is accepted by the controls AND by a
    histogram drag, and stays synchronized across the inline + detached
    histogram surfaces and the spin/slider controls."""
    win = MainWindow()
    try:
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(_hdr_raw_with_stars()), _hdr_raw_with_stars()),
                stack_name="r31",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        upper = win._analysis_upper
        assert upper > 2.0  # fixture guarantee: room for wp > 1 drags

        detached = win._open_detached_histogram()
        qapp = win._histogram_views()
        views = [win.right_histogram_view, detached.histogram_view]

        # (a) Spin edit: WP = 2.0 (analysis units).
        win.stretch_wp_spin.setValue(2.0)
        assert win._white_point == 2.0
        assert win.stretch_wp_spin.value() == 2.0
        assert win.stretch_wp_slider.value() == 2000
        assert win._black_point == 0.01  # untouched
        for view in views:
            assert view.white_point == 2.0
            assert view.black_point == 0.01

        # (b) Histogram drag: pull the white handle up to 2.5.
        view = win.right_histogram_view
        x_wp = view._level_to_x(view.white_point)
        assert view._start_drag_at(x_wp) == "max"
        view._drag_at(view._level_to_x(2.5))
        view._end_drag()
        assert win._white_point == 2.5
        assert win.stretch_wp_spin.value() == 2.5
        assert win.stretch_wp_slider.value() == 2500
        for v in views:
            assert v.white_point == 2.5
        assert win._black_point == 0.01

        # (c) The markers live within the extended marker domain of the views
        # (the grid-ceiling control domain, never below the raw analysis top).
        for v in views:
            assert v._marker_upper == pytest.approx(_grid_ceil(upper))
            assert v._marker_upper >= upper
    finally:
        win.shutdown()


def test_phi_r31_float_display_recovers_headroom_via_white_point(qapp):
    """R3.1-E(iii): the visible Option-A display is rendered from the preserved
    float analysis source — with the white point at 1 the star headroom is
    white-clipped, and raising the white point above 1 visibly recovers the
    highlight structure (fewer saturated pixels, midtones matching the linear
    analysis-unit mapping)."""
    win = MainWindow()
    try:
        raw = _hdr_raw_with_stars(size=48, seed=21)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(raw), raw),
                stack_name="r31",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        upper = win._analysis_upper
        assert upper > 2.0

        # Linear analysis-unit stretch, black 0, white 1: headroom clips white.
        win._stretch = "linear"
        win._black_point = 0.0
        win._white_point = 1.0
        img_clipped = win._render_option_a_display()
        arr_clipped = _qimage_to_array(img_clipped)

        # White point above 1: preserved headroom is mapped back into range.
        win._white_point = 3.0
        assert win._white_point == 3.0
        img_recovered = win._render_option_a_display()
        arr_rec = _qimage_to_array(img_recovered)

        red_clipped = arr_clipped[..., 0]
        red_rec = arr_rec[..., 0]
        n_white_a = int((red_clipped == 255).sum())
        n_white_b = int((red_rec == 255).sum())
        assert n_white_a > n_white_b > 0, (n_white_a, n_white_b)
        assert not np.array_equal(arr_clipped, arr_rec)

        # Highlight structure retained: for a preserved analysis value x in the
        # headroom band (1, 3), the recovered pixel equals round(x/3*255) — the
        # exact linear analysis-unit mapping — while the clipped render had 255.
        wb = win._wb_only_float
        band = (wb[..., 0] > 1.05) & (wb[..., 0] < 2.9)
        assert band.any(), "fixture must contain headroom-band pixels"
        idx = np.argwhere(band)[0]
        y, x = int(idx[0]), int(idx[1])
        x_val = float(wb[y, x, 0])
        expected = int(round(x_val / 3.0 * 255.0))
        assert red_clipped[y, x] == 255  # white-clipped before the fix
        assert red_rec[y, x] == expected, (red_rec[y, x], expected, x_val)
    finally:
        win.shutdown()


def test_phi_r31_bp_wp_validate_and_order_in_extended_range(qapp):
    """R3.1-E(iv): black/white points validate and keep their order in the
    extended analysis range (no inversion, deterministic clamping, values
    above 1 accepted, nothing above the analysis upper)."""
    win = MainWindow()
    try:
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(_hdr_raw_with_stars()), _hdr_raw_with_stars()),
                stack_name="r31",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        upper = win._analysis_upper
        assert upper > 2.0

        # Both points above 1, ordered.
        win.stretch_bp_spin.setValue(0.5)
        win.stretch_wp_spin.setValue(2.0)
        assert win._black_point == 0.5 < win._white_point == 2.0

        # Raising the black point above the white point clamps it (never the
        # white point) preserving the one-step separation.
        win.stretch_bp_spin.setValue(2.5)
        assert win._black_point == pytest.approx(1.999)
        assert win._black_point < win._white_point == 2.0

        # Lowering the white point below the black point pushes it up instead.
        win.stretch_wp_spin.setValue(0.1)
        assert win._white_point == pytest.approx(2.0)
        assert win._black_point < win._white_point

        # Nothing can exceed the control domain (spin range clamps at the
        # grid-ceiling analysis upper).
        win.stretch_wp_spin.setValue(upper * 4.0)
        assert win._white_point == pytest.approx(_grid_ceil(upper))
        assert win._black_point < win._white_point
    finally:
        win.shutdown()


def test_phi_r31_legacy_path_keeps_01_domain(qapp):
    """R3.1-E(v): legacy single-array payloads keep the historical [0, 1]
    display/control semantics even after an Option-A preview had extended the
    analysis domain."""
    win = MainWindow()
    try:
        # Option-A first: controls are re-scoped above 1.
        raw = _hdr_raw_with_stars()
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(raw), raw),
                stack_name="r31-oa",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        assert win._analysis_upper > 1.0
        assert win.stretch_wp_spin.maximum() > 1.0

        # Legacy single-array payload: analysis state cleared; [0, 1] domain.
        win._on_preview(
            BackendPreviewPayload(
                data=_legacy_normalize(raw), stack_name="r31-legacy"
            )
        )
        assert win._pristine_float is None
        assert win._analysis_upper == 1.0
        assert win.stretch_bp_spin.maximum() == 1.0
        assert win.stretch_wp_spin.maximum() == 1.0
        assert win.stretch_wp_slider.maximum() == 1000
        for view in win._histogram_views():
            assert view._marker_upper == 1.0
        # A white point above 1 is clamped to the historical 1.0 domain.
        win.stretch_wp_spin.setValue(1.5)
        assert win._white_point == 1.0
    finally:
        win.shutdown()


def test_phi_r31_analysis_domain_shrink_reconciles_without_inversion(qapp):
    """R3.1-D: when a new preview's analysis range no longer fits the current
    white point, the pair is reconciled deterministically in analysis units
    (white point pulled down to the new upper, black point stays below) and a
    later larger range never silently restores the old white point."""
    win = MainWindow()
    try:
        bright = _hdr_raw_with_stars(size=48, seed=21)
        dim = _hdr_raw_with_stars(size=48, seed=21, star_frac=0.0)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(bright), bright),
                stack_name="bright",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        upper_bright = win._analysis_upper
        assert upper_bright > 2.0
        win.stretch_wp_spin.setValue(3.0)
        assert win._white_point == 3.0

        # Dimmer preview: the analysis upper shrinks below the white point.
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(dim), dim),
                stack_name="dim",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        upper_dim = win._analysis_upper
        assert upper_bright > upper_dim > 1.0
        # Pulled down to the grid-ceiling control domain of the new range.
        assert win._white_point == pytest.approx(_grid_ceil(upper_dim))
        assert win._black_point < win._white_point  # no inversion
        assert win._black_point == 0.01

        # A new bright preview grows the range again but does NOT silently
        # restore the old white point (deterministic, user intent preserved).
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(bright), bright),
                stack_name="bright2",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        assert win._analysis_upper == pytest.approx(upper_bright)
        assert win._white_point == pytest.approx(_grid_ceil(upper_dim))
        assert win._black_point < win._white_point
    finally:
        win.shutdown()


# ---------------------------------------------------------------------------
# PHI-R3.2 (REWORK-R3.2) — trace-independent producer identity gate (F1) and
# frozen-view reconciliation on model-domain change (F2)
# ---------------------------------------------------------------------------

def test_phi_r32_producer_session_preset_by_run_lifecycle_is_stamped(caplog, monkeypatch):
    """The producer honours a run-lifecycle-assigned session id: when the Qt
    run lifecycle pre-binds ``_phi_producer_session`` (via the backend seam),
    every emission of that stacker carries PREV_RUN == the assigned id."""
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)
    obj = _classic_with_sumw(H=64, W=96)
    obj._phi_producer_session = 42  # assigned by the Qt run lifecycle
    collected = []
    obj.preview_callback = lambda *a: collected.append(a)
    obj._update_preview_sum_w(downsample_factor=2)
    obj._update_preview_sum_w(downsample_factor=2)
    assert len(collected) == 2
    assert [int(a[1]["PREV_SEQ"]) for a in collected] == [1, 2]
    assert {int(a[1]["PREV_RUN"]) for a in collected} == {42}


def test_phi_r32_run_bound_gate_same_run_stale_and_duplicate(qapp):
    """With the producer run/session identity bound at run start, same-run
    stale/duplicate sequenced emissions are dropped deterministically."""
    win = MainWindow()
    try:
        win._pending_preview_run_session = 11
        win._on_run_started()
        assert win._preview_run_session == 11
        assert win._pending_preview_run_session is None

        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=1), seq=1, batch=1, run=11))
        assert win._displayed_identity[1:] == ("batch", 1)
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=2), seq=2, batch=2, run=11))
        assert win._displayed_identity[1:] == ("batch", 2)
        assert win._last_accepted_preview_seq == 2

        # Stale (older) and duplicate (equal) of the SAME bound run.
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=3), seq=1, batch=9, run=11))
        assert win._displayed_identity[1:] == ("batch", 2)
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=4), seq=2, batch=9, run=11))
        assert win._displayed_identity[1:] == ("batch", 2)
        assert win._last_accepted_preview_seq == 2
    finally:
        win.shutdown()


def test_phi_r32_foreign_old_run_payload_cannot_poison_new_run(qapp, caplog, monkeypatch):
    """PHI-R3.2 cross-run safety in BOTH arrival orders: a late payload of a
    previous producer session (queued across the run boundary) is rejected as
    foreign and never poisons the new run's sequence high-water mark, so later
    valid current-run payloads are never dropped."""
    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)

    # Order A — old-run payload arrives BEFORE the new run's first payload.
    win = MainWindow()
    try:
        win._pending_preview_run_session = 22
        win._on_run_started()
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=1), seq=5, batch=5, run=1))
        # Foreign old-run emission: dropped (no display state at all).
        assert win._preview_source is None
        assert win._last_accepted_preview_seq is None

        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=2), seq=1, batch=1, run=22))
        assert win._displayed_identity[1:] == ("batch", 1)
        assert win._last_accepted_preview_seq == 1
        win._on_preview(_seq_payload(_hdr_raw(size=32, seed=3), seq=2, batch=2, run=22))
        assert win._displayed_identity[1:] == ("batch", 2)
    finally:
        win.shutdown()

    caplog.clear()

    # Order B — the exact Nono poison scenario: new run's payload accepted
    # first, then an OLD-run payload with a HIGH sequence arrives late.  It
    # must be dropped without advancing the high-water mark, and the next
    # valid current-run payload must still be accepted.
    win2 = MainWindow()
    try:
        win2._pending_preview_run_session = 33
        win2._on_run_started()
        win2._on_preview(_seq_payload(_hdr_raw(size=32, seed=4), seq=1, batch=1, run=33))
        assert win2._displayed_identity[1:] == ("batch", 1)
        assert win2._last_accepted_preview_seq == 1

        # Late old-run payload with a huge sequence: foreign, must NOT move hw.
        win2._on_preview(_seq_payload(_hdr_raw(size=32, seed=5), seq=99, batch=99, run=7))
        assert win2._displayed_identity[1:] == ("batch", 1)
        assert win2._last_accepted_preview_seq == 1, "foreign payload poisoned hw!"

        # The next valid current-run payload is accepted (never dropped).
        win2._on_preview(_seq_payload(_hdr_raw(size=32, seed=6), seq=2, batch=2, run=33))
        assert win2._displayed_identity[1:] == ("batch", 2)
        assert win2._last_accepted_preview_seq == 2

        # The arrival trace documents the foreign drop with its run id.
        drops = [
            _fields(m)
            for m in _parse_stage_records(caplog)
            if _fields(m).get("drop") is not None
        ]
        assert any(d["drop"] == "foreign" and d["prun"] == "7" for d in drops)
    finally:
        win2.shutdown()


def test_phi_r32_legacy_unsequenced_still_bypasses_when_run_bound(qapp):
    """Unsequenced legacy payloads bypass the gate even while a producer run
    session is bound (backward compatibility unchanged)."""
    win = MainWindow()
    try:
        win._pending_preview_run_session = 44
        win._on_run_started()
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(_hdr_raw(size=32, seed=1)), _hdr_raw(size=32, seed=1)),
                stack_name="legacy",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
                current_batch=3,
                image_count=3,
            )
        )
        assert win._displayed_identity[1:] == ("batch", 3)
        assert win._last_accepted_preview_seq is None  # gate untouched
    finally:
        win.shutdown()


def test_phi_r32_start_assigns_session_and_binds_at_run_start(qapp):
    """MainWindow._on_start allocates a durable session id, hands it to a
    PHI-capable backend (``set_preview_session``) and binds the same id at run
    start (``_on_run_started`` consumes the pending id)."""
    from seestar.gui_qt.backend_runner import BackendRunResult, BaseRunBackend

    holders = []

    class SessionCaptureBackend(BaseRunBackend):
        def __init__(self, data):
            self._data = data
            self.preview_session = None

        def set_preview_session(self, session):
            self.preview_session = int(session)

        def run(
            self,
            request,
            progress_callback,
            log_callback,
            is_cancel_requested,
            preview_callback=None,
        ):
            progress_callback(50)
            if preview_callback is not None:
                preview_callback(
                    BackendPreviewPayload(
                        data=self._data,
                        stack_name="session-preview",
                        image_count=1,
                        total_images=5,
                    )
                )
            progress_callback(100)
            return BackendRunResult.FINISHED

        def cancel(self) -> None:
            pass

    raw = np.zeros((8, 12, 3), dtype=np.float32)
    raw[:, :, 0] = 1.0
    win = MainWindow(
        backend_factory=lambda: (
            holders.append(SessionCaptureBackend(raw)) or holders[-1]
        )
    )
    try:
        win.start_button.click()
        # ``_on_start`` runs synchronously: the session id was allocated and
        # handed to the backend BEFORE the worker started, and ``_on_run_started``
        # bound it synchronously.  Wait only for the fake run to finish.
        assert _pump_until(lambda: win._running is False)
        backend = holders[0]
        assert win._pending_preview_run_session is None  # consumed at start
        assert backend.preview_session is not None
        # The GUI bound the exact id it handed to the backend.
        assert win._preview_run_session == backend.preview_session
        # The unsequenced fake payload kept the legacy acceptance path.
        assert win._last_accepted_preview_seq is None
    finally:
        win.shutdown()


def test_phi_r32_frozen_view_reconciled_on_model_shrink(qapp):
    """F2 (Nono repro): a frozen/manual zoom window wider than the new model
    domain is clamped into it — the painted axis never extends beyond the data
    domain after a shrink."""
    from seestar.gui_qt.histogram_view import HistogramView

    view = HistogramView()
    try:
        view.resize(256, 80)
        # Wide model: range (0, 4), robust zoom window near the top.
        view.set_model(_synthetic_model_wide(x_range=(0.1195, 3.9805), upper=4.0))
        view.zoom_histogram()  # freezes (0.1195, 3.9805)
        assert view.view_range == (pytest.approx(0.1195), pytest.approx(3.9805))
        assert view._frozen_range is not None

        # Replacement model with a smaller domain (0, 1.2).
        view.set_model(_synthetic_model_wide(x_range=(0.05, 1.15), upper=1.2))
        lo, hi = view.view_range
        assert hi <= 1.2 + 1e-9, f"stale window survived shrink: {(lo, hi)}"
        assert lo == pytest.approx(0.1195)  # valid part preserved
        assert hi == pytest.approx(1.2)  # clamped to the new domain top
        assert view._model_range == (0.0, 1.2)
        assert view._frozen_range == view.view_range
    finally:
        view.deleteLater()


def test_phi_r32_frozen_view_preserved_on_grow_and_valid_shrink(qapp):
    """F2: a still-valid manual zoom is preserved verbatim across model-domain
    changes (grow and benign shrink); only out-of-domain windows are clamped."""
    from seestar.gui_qt.histogram_view import HistogramView

    view = HistogramView()
    try:
        view.resize(256, 80)
        view.set_model(_synthetic_model_wide(x_range=(0.1, 0.9), upper=1.2))
        view.zoom_histogram()
        assert view.view_range == (pytest.approx(0.1), pytest.approx(0.9))

        # Grow: model domain (0, 4) — the valid manual range is preserved.
        view.set_model(_synthetic_model_wide(x_range=(0.05, 3.9), upper=4.0))
        assert view.view_range == (pytest.approx(0.1), pytest.approx(0.9))
        # Benign shrink to (0, 2.5): still valid, preserved.
        view.set_model(_synthetic_model_wide(x_range=(0.05, 2.4), upper=2.5))
        assert view.view_range == (pytest.approx(0.1), pytest.approx(0.9))

        # Degenerate after clamping (window fully beyond the new domain):
        # falls back to the full analysis range, never inverted.
        view.set_model(_synthetic_model_wide(x_range=(2.3, 2.45), upper=2.5))
        view.zoom_histogram()
        assert view.view_range == (pytest.approx(2.3), pytest.approx(2.45))
        view.set_model(_synthetic_model_wide(x_range=(0.05, 1.15), upper=1.2))
        lo, hi = view.view_range
        assert lo == pytest.approx(0.0) and hi == pytest.approx(1.2)
    finally:
        view.deleteLater()


def test_phi_r32_inline_and_detached_ranges_stay_synced_after_shrink(qapp):
    """F2 integration: after a real preview whose analysis upper shrinks, the
    inline and detached histogram surfaces keep the same reconciled view range
    (bounded by the new model domain), and markers stay within their extended
    domain."""
    win = MainWindow()
    try:
        # Bright frame first (large analysis upper).
        bright = _hdr_raw_with_stars(size=48, seed=21)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(bright), bright),
                stack_name="bright",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        upper_bright = win._analysis_upper
        assert upper_bright > 2.0
        detached = win._open_detached_histogram()
        # Manual zoom on the inline surface (freezes a window).
        win.right_histogram_view.zoom_histogram()
        detached.histogram_view.set_view_range(*win.right_histogram_view.view_range)

        # Dim frame: analysis upper shrinks towards ~1.
        dim = _hdr_raw_with_stars(size=48, seed=21, star_frac=0.0)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(dim), dim),
                stack_name="dim",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        upper_dim = win._analysis_upper
        assert 1.0 < upper_dim < upper_bright

        inline = win.right_histogram_view
        assert inline._model_range[1] == pytest.approx(upper_dim)
        # Inline range is reconciled (bounded by the new domain).
        lo, hi = inline.view_range
        assert hi <= upper_dim + 1e-9, f"inline view beyond domain: {(lo, hi)}"
        # Detached mirrors the model + reconciled range (no desync).
        assert detached.histogram_view._model_range == inline._model_range
        assert detached.histogram_view.view_range == inline.view_range
        # Markers/controls remain in their valid extended domain.
        assert win._white_point <= win._bp_wp_control_upper + 1e-9
        assert win._black_point < win._white_point
        assert inline.black_point == win._black_point
        assert detached.histogram_view.white_point == win._white_point
    finally:
        win.shutdown()


def _synthetic_model_wide(x_range, upper=4.0, channels=("R", "G", "B")):
    """Minimal float model with an explicit analysis range upper (F2 helper)."""
    bins = 512
    counts, log_counts, stats = {}, {}, {}
    for c in channels:
        arr = np.zeros(bins, dtype=np.int64)
        arr[10] = 1
        counts[c] = arr
        log_counts[c] = np.log1p(arr.astype(np.float64))
        stats[c] = {"min": 0.0, "max": upper, "median": 0.5, "mean": 0.5, "std": 0.1}
    return {
        "bins": bins,
        "range": (0.0, upper),
        "channels": list(channels),
        "counts": counts,
        "log_counts": log_counts,
        "stats": stats,
        "x_range": x_range,
        "full_range": (0.0, upper),
    }


# ---------------------------------------------------------------------------
# PHI-R3.3 (REWORK-R3.3, final corrective iteration) — F2 detached policy
# mirroring, F3 model→legacy view-state reset, real-backend stamping test
# ---------------------------------------------------------------------------

def test_phi_r33_detached_open_after_unfrozen_reset_stays_in_sync(qapp):
    """F2 (Nono repro): opening the detached view from an inline view with NO
    frozen/manual range must not manufacture a detached frozen range — after a
    model analysis-upper shrink both surfaces deterministically end with the
    same unfrozen, in-domain view (inline default display window)."""
    win = MainWindow()
    try:
        bright = _hdr_raw_with_stars(size=48, seed=21)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(bright), bright),
                stack_name="bright",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        upper_bright = win._analysis_upper
        assert upper_bright > 2.0

        # Unfrozen inline state: reset to the full analysis range.
        inline = win.right_histogram_view
        inline.reset_histogram_view()
        assert inline._frozen_range is None
        assert inline.view_range[1] == pytest.approx(upper_bright)

        # Opening the detached view mirrors the *policy*: still unfrozen.
        detached = win._open_detached_histogram()
        detached_view = detached.histogram_view
        assert detached_view._frozen_range is None, (
            "detached must not get an artificial frozen range"
        )
        assert detached_view.view_range == inline.view_range
        assert detached_view.auto_zoom_enabled == inline.auto_zoom_enabled

        # Model shrink: both surfaces stay synchronized, unfrozen, in-domain.
        dim = _hdr_raw_with_stars(size=48, seed=21, star_frac=0.0)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(dim), dim),
                stack_name="dim",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        upper_dim = win._analysis_upper
        assert 1.0 < upper_dim < upper_bright

        assert inline._model_range[1] == pytest.approx(upper_dim)
        assert detached_view._model_range == inline._model_range
        assert inline._frozen_range is None
        assert detached_view._frozen_range is None
        assert detached_view.view_range == inline.view_range
        lo, hi = inline.view_range
        assert lo == 0.0 and hi <= 1.0 + 1e-9  # unfrozen display window
        # Marker domains and markers stay equal and valid.
        assert detached_view._marker_upper == inline._marker_upper
        assert detached_view.white_point == inline.white_point == win._white_point
    finally:
        win.shutdown()


def test_phi_r33_detached_genuine_manual_frozen_reconciled_on_shrink(qapp):
    """F2: a genuine manual frozen view (zoom) is preserved on the detached
    surface and reconciled identically with the inline surface on shrink."""
    win = MainWindow()
    try:
        bright = _hdr_raw_with_stars(size=48, seed=21)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(bright), bright),
                stack_name="bright",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        upper_bright = win._analysis_upper
        inline = win.right_histogram_view
        inline.zoom_histogram()  # genuine manual/robust freeze
        assert inline._frozen_range is not None

        detached = win._open_detached_histogram()
        detached_view = detached.histogram_view
        assert detached_view._frozen_range == inline._frozen_range
        assert detached_view.view_range == inline.view_range

        dim = _hdr_raw_with_stars(size=48, seed=21, star_frac=0.0)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(dim), dim),
                stack_name="dim",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        upper_dim = win._analysis_upper
        # Both surfaces reconcile the frozen window into the new domain and
        # remain identical (frozen state preserved where valid).
        assert inline._frozen_range is not None
        assert detached_view._frozen_range == inline._frozen_range
        assert detached_view.view_range == inline.view_range
        lo, hi = inline.view_range
        assert 0.0 <= lo < hi <= upper_dim + 1e-9
    finally:
        win.shutdown()


def test_phi_r33_option_a_to_legacy_transition_resets_views(qapp):
    """F3 (Nono repro): an Option-A float model with a frozen zoom above 1
    followed by a legacy single-array payload leaves every surface on a valid
    legacy [0, 1] window automatically (no frozen float-model state, marker
    domain and BP/WP controls back at [0, 1], inline/detached equal)."""
    win = MainWindow()
    try:
        bright = _hdr_raw_with_stars(size=48, seed=21)
        win._on_preview(
            BackendPreviewPayload(
                data=(_legacy_normalize(bright), bright),
                stack_name="bright",
                header={"PREV_SRC": _SUMW_PREVIEW_SRC},
            )
        )
        assert _wait_histogram(win)
        assert win._analysis_upper > 2.0
        inline = win.right_histogram_view
        inline.zoom_histogram()
        assert inline._frozen_range is not None
        assert inline._frozen_range[1] > 1.0  # zoomed above the display top
        detached = win._open_detached_histogram()
        detached_view = detached.histogram_view
        assert detached_view._frozen_range == inline._frozen_range

        # Legacy single-array payload (analysis state cleared synchronously).
        win._on_preview(
            BackendPreviewPayload(
                data=_legacy_normalize(bright), stack_name="legacy"
            )
        )
        assert win._pristine_float is None
        # Both surfaces: no float model, no frozen range, valid [0, 1] window.
        for view in (inline, detached_view):
            assert view._model is None
            assert view._frozen_range is None
            assert view.view_range == (0.0, 1.0)
            assert view._marker_upper == 1.0
        # BP/WP controls are back at the legacy [0, 1] domain.
        assert win._analysis_upper == 1.0
        assert win.stretch_bp_spin.maximum() == 1.0
        assert win.stretch_wp_spin.maximum() == 1.0
        assert win._black_point == 0.01 and win._white_point == 0.99
        assert inline.black_point == detached_view.black_point == win._black_point
        assert inline.white_point == detached_view.white_point == win._white_point
    finally:
        win.shutdown()


def test_phi_r33_real_backend_stamps_session_and_producer_emits_trace_off(caplog, monkeypatch):
    """Coverage gap: the REAL SeestarQueuedStackerBackend construction path
    stamps the GUI-assigned preview session onto the stacker it creates
    (``_ensure_stackers``), and an active producer delivered through that
    stacker emits PREV_RUN + PREV_SEQ with ZSSS_PHI_TRACE unset (fake
    stacker/input fixtures only, no hardware/user data)."""
    import types

    from types import SimpleNamespace

    from seestar.gui_qt.backend_runner import SeestarQueuedStackerBackend
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    monkeypatch.delenv("ZSSS_PHI_TRACE", raising=False)
    caplog.set_level(logging.DEBUG)
    assert not phi_trace_enabled()

    collected = []

    class _FixtureStacker:
        """Cheap stand-in the real backend constructs via ``stacker_factory``
        (real __init__ would spawn process pools — kept out of tests)."""

        def __init__(self, **kwargs):
            self.current_stack_header = fits.Header()
            self.preview_callback = None
            self.preview_downsample_factor = 1
            self.images_in_cumulative_stack = 3
            self.files_in_queue = 10
            self.stacked_batches_count = 1
            self.total_batches_estimated = 2
            self.cumulative_sum_memmap = None
            self.cumulative_wht_memmap = None
            # Run the REAL Classic active producer on this fixture instance so
            # the delivered callback payload is a genuine producer emission.
            self._update_preview_sum_w = types.MethodType(
                SeestarQueuedStacker._update_preview_sum_w, self
            )

    backend = SeestarQueuedStackerBackend(
        stacker_factory=_FixtureStacker, poll_interval=0.01
    )
    backend.set_preview_session(77)  # what MainWindow._on_start does
    stacker = backend._ensure_stackers(SimpleNamespace(align_on_disk=False))
    # Real backend stamping: the stacker it created carries the assigned id.
    assert stacker._phi_producer_session == 77

    # Drive one genuine active-producer emission (trace OFF).
    H, W = 16, 24
    avg = np.linspace(0.0, 1.0, H * W * 3, dtype=np.float32).reshape(H, W, 3)
    stacker.cumulative_sum_memmap = avg.astype(np.float32)
    stacker.cumulative_wht_memmap = np.ones((H, W), dtype=np.float32)
    stacker.preview_callback = lambda *a: collected.append(a)
    stacker._update_preview_sum_w(downsample_factor=1)
    assert len(collected) == 1
    header = collected[0][1]
    assert int(header["PREV_RUN"]) == 77  # assigned session stamped
    assert int(header["PREV_SEQ"]) == 1  # per-emission sequence
    assert "PREV_REQ" in header and "PREV_RES" in header
    # Telemetry stays a genuine no-op with the gate off.
    assert _parse_stage_records(caplog) == []

    # Second emission advances the sequence under the same session.
    stacker._update_preview_sum_w(downsample_factor=1)
    assert int(collected[1][1]["PREV_RUN"]) == 77
    assert int(collected[1][1]["PREV_SEQ"]) == 2

    # Reused backend (new run): set_preview_session re-stamps the existing
    # stacker so payloads of the new run carry the new session id.
    backend.set_preview_session(88)
    assert stacker._phi_producer_session == 88
    stacker._update_preview_sum_w(downsample_factor=1)
    assert int(collected[2][1]["PREV_RUN"]) == 88
    assert _parse_stage_records(caplog) == []


# ---------------------------------------------------------------------------
# PHI-R4 — preview route reachability and integrity parity (dispatch predicates)
# ---------------------------------------------------------------------------

def _dispatch_stub():
    """Minimal live-dispatch stand-in (real methods, fixture attributes)."""
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    obj = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    obj.update_progress = lambda *a, **k: None
    obj.preview_callback = None
    obj.current_stack_header = fits.Header()
    obj.preview_downsample_factor = 2
    obj.drizzle_active_session = False
    obj.drizzle_accumulators = None
    obj.drizzle_processing_policy = "standard"
    obj.drizzle_group_size = 2
    obj._drizzle_frame_count = 0
    obj._drizzle_group_index = 0
    # Spy seams (instance attributes shadow the real bound methods; the live
    # dispatch call sites invoke them with no arguments, exactly as here).
    obj.calls = {"drizzle": 0, "sumw": 0}

    def _spy_drizzle():
        obj.calls["drizzle"] += 1
        return None

    def _spy_sumw():
        obj.calls["sumw"] += 1
        return None

    obj._update_preview_drizzle_accumulator = _spy_drizzle
    obj._update_preview_sum_w = _spy_sumw
    return obj


def test_phi_r4_refresh_preview_dispatches_on_session_predicate():
    """refresh_preview (live resolution/control refresh) dispatches on the
    real predicate: active-drizzle session + accumulators -> standard Drizzle
    accumulator route; otherwise -> Classic SUM/W route."""
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    # Drizzle session with accumulators: the accumulator route is selected.
    obj = _dispatch_stub()
    obj.drizzle_active_session = True
    obj.drizzle_accumulators = ["acc"]  # predicate truthiness only
    SeestarQueuedStacker.refresh_preview(obj)
    assert obj.calls["drizzle"] == 1
    assert obj.calls["sumw"] == 0

    # Non-drizzle session: the Classic SUM/W route is selected.
    obj2 = _dispatch_stub()
    obj2.drizzle_active_session = False
    SeestarQueuedStacker.refresh_preview(obj2)
    assert obj2.calls["sumw"] == 1
    assert obj2.calls["drizzle"] == 0

    # Predicate edge (documented): drizzle session but no accumulators yet
    # falls back to the SUM/W route in refresh_preview.
    obj3 = _dispatch_stub()
    obj3.drizzle_active_session = True
    obj3.drizzle_accumulators = None
    SeestarQueuedStacker.refresh_preview(obj3)
    assert obj3.calls["sumw"] == 1
    assert obj3.calls["drizzle"] == 0


def test_phi_r4_drizzle_preview_cadence_both_policies_use_accumulator_route():
    """Both drizzle processing policies emit DISPLAY previews through the
    active accumulator route (_update_preview_drizzle_accumulator) — the
    policy predicate (standard vs incremental) only changes the cadence.  The
    legacy incremental producer is never selected by live dispatch."""
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    # Standard policy: a preview after EVERY accumulated pose.
    obj = _dispatch_stub()
    obj.drizzle_processing_policy = "standard"
    for _ in range(3):
        SeestarQueuedStacker._drizzle_group_tick(obj)
    assert obj.calls["drizzle"] == 3

    # Incremental policy (legacy drizzle_mode, M3 single-accumulator science):
    # preview at group boundaries only, still via the accumulator route.
    obj2 = _dispatch_stub()
    obj2.drizzle_processing_policy = "incremental"
    obj2.drizzle_group_size = 2
    SeestarQueuedStacker._drizzle_group_tick(obj2)  # frame 1: no group yet
    assert obj2.calls["drizzle"] == 0
    SeestarQueuedStacker._drizzle_group_tick(obj2)  # frame 2: group 1 -> preview
    assert obj2.calls["drizzle"] == 1
    SeestarQueuedStacker._drizzle_group_tick(obj2)  # frame 3
    assert obj2.calls["drizzle"] == 1
    SeestarQueuedStacker._drizzle_group_tick(obj2)  # frame 4: group 2 -> preview
    assert obj2.calls["drizzle"] == 2

    # Trailing partial group flush (incremental policy) -> accumulator route.
    obj3 = _dispatch_stub()
    obj3.drizzle_processing_policy = "incremental"
    obj3.drizzle_group_size = 2
    obj3._drizzle_frame_count = 5  # 5 % 2 != 0 -> one partial-group preview
    SeestarQueuedStacker._drizzle_flush_partial_group(obj3)
    assert obj3.calls["drizzle"] == 1
    # Even frame count -> no partial preview.
    obj4 = _dispatch_stub()
    obj4.drizzle_processing_policy = "incremental"
    obj4.drizzle_group_size = 2
    obj4._drizzle_frame_count = 4
    SeestarQueuedStacker._drizzle_flush_partial_group(obj4)
    assert obj4.calls["drizzle"] == 0


def test_phi_r5_legacy_machinery_retired_no_supported_dispatch_invokes_it():
    """PHI-R5 retirement regression (criterion 4): the M3-D OBSOLETE LEGACY
    incremental-Drizzle preview/process chain, its display carrier state and
    the dead reproject/master preview carrier are REMOVED from queue_manager.
    A source-level audit proves the retired symbols no longer exist, and a
    representative supported dispatch (refresh_preview under both session
    states + per-pose/group ticks) still refreshes Classic / standard Drizzle
    safely with no legacy invocation possible (no speculative fallback)."""
    import ast

    qm_path = Path(__file__).resolve().parents[1] / "seestar/queuep/queue_manager.py"
    qm_src = qm_path.read_text(encoding="utf-8")
    retired = (
        "_update_preview_incremental_drizzle",
        "_start_drizzle_process",
        "drizzle_batch_worker",
        "_process_incremental_drizzle_batch",
        "_wait_drizzle_processes",
        "_update_preview_master",
        "_incremental_reproject_coadd",
        "incremental_drizzle_objects",
        "incremental_drizzle_sci_arrays",
        "incremental_drizzle_wht_arrays",
        "intermediate_drizzle_batch_files",
        "cumulative_drizzle_data",
        "current_stack_data_raw",
        "master_sum",
        "master_coverage",
        "drizzle_processes",
        "drizzle_executor",
        "reproject_output_wcs",
    )
    for sym in retired:
        assert sym not in qm_src, f"{sym!r} still present after PHI-R5"
    tree = ast.parse(qm_src)
    fns = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for gone in ("_update_preview", "_update_preview_master", "_save_intermediate_stack"):
        assert gone not in fns, f"{gone!r} still defined after PHI-R5"

    # Supported dispatch still refreshes safely (both session states), and the
    # group-tick cadence still previews through the accumulator route only.
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    obj = _dispatch_stub()
    obj.drizzle_active_session = True
    obj.drizzle_accumulators = ["acc"]
    SeestarQueuedStacker.refresh_preview(obj)
    SeestarQueuedStacker._drizzle_group_tick(obj)
    SeestarQueuedStacker._drizzle_group_tick(obj)
    assert obj.calls["drizzle"] == 3
    assert obj.calls["sumw"] == 0

    obj2 = _dispatch_stub()
    obj2.drizzle_active_session = False
    SeestarQueuedStacker.refresh_preview(obj2)
    assert obj2.calls["sumw"] == 1
    assert obj2.calls["drizzle"] == 0


def test_phi_r4_route_labels_distinguish_active_producers(qapp, caplog, monkeypatch):
    """Trace/metadata source labels: producer stage records and the Qt
    payload_arrive route distinguish Classic (SUM/W), standard Drizzle
    (accumulator) and legacy-incremental (dead) routes truthfully."""
    from seestar.utils.phi_trace import phi_trace_stage

    monkeypatch.setenv("ZSSS_PHI_TRACE", "1")
    caplog.set_level(logging.DEBUG)

    # Producer stage records carry route + src labels for the supported
    # active producers (the legacy incremental route was retired in PHI-R5).
    phi_trace_stage(
        logging.getLogger("phi.test"), route="classic", stage="source", arr=_hdr_raw(size=8), src="SUM/W"
    )
    phi_trace_stage(
        logging.getLogger("phi.test"), route="drizzle", stage="source", arr=_hdr_raw(size=8), src="Drizzle"
    )
    routes = {_fields(m)["route"] for m in _parse_stage_records(caplog)}
    assert {"classic", "drizzle"} <= routes

    # Qt mode derivation from real producer headers (dispatch-level payloads):
    # Classic SUM/W header -> "classic"; Drizzle Accumulator -> "drizzle";
    # no PREV_SRC -> "reproject" fallback label (legacy carriers).
    win = MainWindow()
    try:
        raw = _hdr_raw(size=16, seed=3)
        classic_payload = BackendPreviewPayload(
            data=(_legacy_normalize(raw), raw), stack_name="c",
            header={"PREV_SRC": _SUMW_PREVIEW_SRC, "PREV_SEQ": 1, "PREV_RUN": 5},
        )
        drizzle_payload = BackendPreviewPayload(
            data=(_legacy_normalize(raw), raw), stack_name="d",
            header={"PREV_SRC": _DRIZZLE_PREVIEW_SRC, "PREV_SEQ": 1, "PREV_RUN": 5},
        )
        assert win._derive_preview_mode(classic_payload) == "classic"
        assert win._derive_preview_mode(drizzle_payload) == "drizzle"
        legacy_payload = BackendPreviewPayload(data=raw, stack_name="l")
        assert win._derive_preview_mode(legacy_payload) == "reproject"
        # The classic/drizzle sources are Option-A raw-linear carriers; the
        # single-array legacy payload keeps the legacy QImage path.
        from seestar.gui_qt.main_window import _is_option_a_preview_payload

        assert _is_option_a_preview_payload(classic_payload.data)
        assert _is_option_a_preview_payload(drizzle_payload.data)
        assert not _is_option_a_preview_payload(legacy_payload.data)
    finally:
        win.shutdown()
