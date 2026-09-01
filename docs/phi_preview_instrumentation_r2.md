# PHI-R2 — Preview pipeline instrumentation and reproduction witnesses

**Mission:** PHI — Live Preview / Histogram Integrity V2  
**Baseline:** `9cd8e8529fb9c6888fe7bf49100b2830493062d2` (PHI-R1, clean)  
**Working branch:** `fix/live-preview-histogram-v2`  
**Scope:** evidence-only, debug-gated compact live-preview/histogram telemetry and deterministic witnesses that prove or rule out the R1 hypotheses. **No viewer behavioural, tone-curve, histogram-semantic, clipping, cache, or async correction is included in this commit.**

## 1. Telemetry contract

### 1.1 Debug gate

One documented environment-variable gate: `ZSSS_PHI_TRACE`. Any value other
than `0` / `false` / `no` / `off` (case-insensitive) enables tracing; default
(unset) is **disabled**. When disabled, every trace call is a no-op that
returns before importing numpy or touching any array — production behaviour is
bit-identical (verified by `test_phi_trace_gate_disabled_produces_no_records`
and the untouched existing suite).

### 1.2 Record format and boundedness

Every enabled record is a single compact line at `logger.debug` prefixed
`PREVIEW_STAGE`, carrying at least: `route`, `stage`, `dtype`, `shape`, `min`,
`p01`, `median`, `p99`, `max`, plus stage-specific fields (`factor`, `src`,
`src_id`, `seq`, `identity`, `res`, `wb`, `lo`/`hi`, `bins`, per-channel max/
median). Statistics are computed on a deterministic fixed-stride subsample
(capped at `1_000_000` elements), so tracing never copies a whole array and
never logs per-pixel data. Tests assert records stay `< 400` chars, contain no
newline, and never contain an array dump (`array(` / `[`).

### 1.3 Instrumented stages

Producer side (`seestar/queuep/queue_manager.py`):

| Route | Stages | Array traced | Extras |
|---|---|---|---|
| `classic` (`_update_preview_sum_w`) | `source`, `pre_resize`, `post_resize` | SUM/W divide (full-res) → raw-linear full-res → raw-linear sent | `factor`, `src=SUM/W`, `src_id=id(memmap)`, `seq` |
| `drizzle` (`_update_preview_drizzle_accumulator`) | `source`, `pre_resize`, `post_resize` | `finalize("divide")` HWC stack → pre-factor → post-factor | `factor`, `src=Drizzle`, `src_id=id(accumulators)`, `seq` |
| `legacy_drizzle` (`_update_preview_incremental_drizzle`) | `source` | cached `cumulative_drizzle_data` | `factor`, `src=LegacyDrizzle`, `src_id`, `seq` (reachability probe, R1 hypothesis 5) |

Qt side (`seestar/gui_qt/main_window.py`, `preview_adjust.py`):

| Stage | Where | Array traced | Extras |
|---|---|---|---|
| `payload_arrive` | `_on_preview` | — (delivered shape via attribute) | `src`, `identity`, `res`, `shape`, `seq` |
| `raw_source` | `_ingest_option_a_preview` | `extract_raw_linear` copy | ctx (src/identity/res/shape) |
| `anchor_mapped` | `_ingest_option_a_preview` | `map_raw_linear` output | `lo`/`hi` anchors + ctx |
| `wb_only` | `_ensure_wb_only_float` | `apply_wb_float` output | `wb` gains + ctx |
| `stretch_input` | `apply_preview_adjustments` | post-WB float `[0,1]` display domain | `stretch`, `bp`, `wp` |
| `stretch_output` | `apply_preview_adjustments` | final uint8 `[0,255]` display | `stretch` |
| `histogram_output` | `_on_histogram_result` | — (model stats) | `bins`, per-channel `*_max`/`*_med` + ctx |

The helper lives in `seestar/utils/phi_trace.py` (display-only: no science or
Qt imports, lazy numpy, never mutates arrays). The Qt context (`src`,
`identity`, `res`, `shape`) is stashed in `_phi_trace_ctx` per payload so stage
records are attributable to the arriving preview and its requested/effective
resolution identity.

## 2. Witness outcomes

### W1 — HDR / strong-WB: where does the first `==1` plateau appear?

`tests/test_phi_preview_pipeline.py::test_phi_witness_hdr_strong_wb_first_plateau`

Deterministic synthetic Option-A payload (seeded, red-dominant, raw max `> 1`)
with a strong 3x red WB gain, `ZSSS_PHI_TRACE=1`, offscreen. Asserts, from the
emitted stage records in order:

1. `raw_source.max > 1.0` — over-range headroom exists at the raw-linear boundary;
2. the **first** stage whose `max == 1.0` is `anchor_mapped` — the anchor map
   (`map_raw_linear`) is the first clip site in the current baseline;
3. `wb_only.max == 1.0` — the plateau already exists before WB; the WB clip is
   **not** the first clip;
4. `histogram_output` in-domain per-channel max `== 1.0` — the histogram only
   ever sees the clipped WB-only domain (R1 hypothesis 2 confirmed: it cannot
   distinguish preserved headroom from display-domain saturation).

This is a *diagnostic* witness: it asserts where clipping happens today, not a
repaired behaviour.

### W2 — Reordered payloads: current last-wins acceptance contract

`tests/test_phi_preview_pipeline.py::test_phi_witness_reordered_payloads_last_wins`

Two valid Option-A payloads delivered deterministically (no sleeps):

- same identity (batch 2): full-res 1/1 first, stale half-res second → the
  half-res **replaces** the full-res display (`_preview_source.width() == 32`);
- changed metadata (batch 3 then batch 2): the older batch wins (last-wins).

Both prove R1 hypothesis 3's acceptance limitation: `_on_preview` has no
monotonic generation/resolution gate and every valid payload replaces
`_preview_source` unconditionally. The trace records the arrival order with
`identity` + delivered `shape`, making the ordering observable. **No corrective
gate is added in R2** (explicitly out of scope).

### W3 — Producer isolation with telemetry enabled

`tests/test_phi_preview_pipeline.py::test_phi_witness_producer_isolation_with_trace_on`

Classic SUM/W memmaps and standard-Drizzle accumulator `_out_img`/`_out_wht`
are asserted bit-identical after producer runs **with the trace gate on** —
telemetry itself cannot mutate science state. Existing isolation proof
(`tests/test_preview_raw_linear_producer.py`) also still passes.

### W3b — Direct-source resolution sequence 1 → 2 → 3 → 4 → 1 (both producers)

`test_phi_witness_classic_direct_source_resolution_sequence` and
`test_phi_witness_drizzle_direct_source_resolution_sequence`:

- every factor `f` output equals `cv2.resize(fullres_raw, (W//f, H//f),
  INTER_AREA)` computed independently in the test → each payload is a direct
  resize of the authoritative source, **no cumulative-resize chain**
  (R1 hypothesis 4 ruled out for the active producers);
- the final 1/1 equals a fresh direct 1/1 render;
- science arrays/accumulators unchanged across the whole sequence.

### W4 — Gate contract

`test_phi_trace_gate_disabled_produces_no_records`: with the gate unset, the
producer delivers normally and **zero** `PREVIEW_STAGE` records are emitted.
`test_phi_witness_producer_isolation_with_trace_on` additionally asserts every
enabled record is compact, single-line, and carries the required fields.

## 3. Invariant confirmation

- No mutation of SUM/WHT, Drizzle accumulators, WCS, registration/
  reconstruction, final FITS, or saved PNG paths (producer-isolation tests +
  diff review: trace calls are read-only `logger.debug` emissions).
- No rendering/tonemapping/histogram semantic correction; no shoulder renderer;
  no cache rewrite; no async/race correction; no UI/settings change.
- Default production behaviour silent and bit-identical (gate off by default;
  all existing tests green).

## 4. Validation

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_phi_preview_pipeline.py tests/test_preview_raw_linear_producer.py \
  tests/test_preview_analysis.py tests/test_qt_histogram_otpux_h1.py \
  tests/test_qt_histogram_otpux_h2.py tests/test_qt_histogram_otpux_drift.py \
  tests/test_qt_preview_reconcile_m24.py tests/test_qt_res_live_m22.py \
  tests/test_qt_otpux_stable_a.py tests/test_qt_preview.py
# 156 passed

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_qt_audit_m255f.py tests/test_qt_display_state_otpux.py \
  tests/test_qt_expert_m15.py tests/test_qt_histogram_m14.py \
  tests/test_qt_preview_controls.py tests/test_qt_preview_ergonomics_m17.py \
  tests/test_qt_settings_surface.py tests/test_qt_shell.py \
  tests/test_qt_stacking_m16.py
# 119 passed

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_phi_preview_pipeline.py
# 6 passed

python -m py_compile <all modified files>      # OK
git diff --check                               # OK
```

## 5. Remaining uncertainty / PHI-R3 recommendation

- The **legacy incremental Drizzle preview** route (`legacy_drizzle`) is now
  instrumented for reachability but no live-workload evidence yet proves it
  fires in production (the standard route is the active one). PHI-R3 can use a
  real-run trace to decide whether to retire it.
- The **ordering limitation is confirmed but uncorrected by design**; the
  decision on a monotonic generation gate (payload sequence vs requested
  resolution) belongs to a behavioural PHI-R3 step with its own tests.
- The plateau location (`anchor_mapped`) and histogram headroom loss are now
  reproducible diagnostics; the *intended* repaired mapping/WB/histogram
  semantics are explicitly **not** decided here — that is PHI-R3's decision
  gate.
