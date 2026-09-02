# PHI-R2/R3 — Preview pipeline instrumentation and reproduction witnesses

> **Closure:** the PHI-R1 → R5 programme is closed.  See
> `docs/phi_live_preview_integrity_closure.md` for the user-visible resolution,
> the acceptance matrix and the remaining non-blockers.  Sections marked
> historical/superseded below are kept as records of their phase.

**Status:** R2 (instrumentation, evidence-only) committed at `63e9b87` with
accepted corrective rework in the working tree; **PHI-R3 (behavioural
integrity correction) implemented on top in the same uncommitted working
tree diff** — see §7; **PHI-R3.1 (analysis-unit BP/WP + float display
rendering rework) implemented on top of R3** — see §8; **PHI-R3.2
(REWORK-R3.2: trace-independent producer identity gate + frozen-view
reconciliation) implemented on top of R3.1** — see §9; **PHI-R3.3
(REWORK-R3.3, final corrective iteration: detached view-policy mirroring,
model→legacy view-state reset, real-backend stamping test) implemented on top
of R3.2** — see §10; **PHI-R4 (preview route reachability & integrity
parity: live route/dispatch inventory + legacy incremental-Drizzle
dead-dispatch proof) implemented on top of R3.3** — see §11; **PHI-R5
(approved retirement of the dead legacy preview machinery) implemented on top
of R4** — see §12.  R2's W1 diagnostic is superseded by the R3 headroom
witness; W2 remains valid as the *legacy unsequenced* acceptance witness.

**Mission:** PHI — Live Preview / Histogram Integrity V2
**Baseline:** `9cd8e8529fb9c6888fe7bf49100b2830493062d2` (PHI-R1, clean)
**Working branch:** `fix/live-preview-histogram-v2`
**R2 scope:** evidence-only, debug-gated compact live-preview/histogram
telemetry and deterministic witnesses that prove or rule out the R1
hypotheses. **No viewer behavioural, tone-curve, histogram-semantic, clipping,
cache, or async correction is included in the R2 commit.**

**R3 scope (§7):** the smallest production-safe behavioural correction the R2
witnesses gated — preserve finite float headroom through the analysis/WB/
histogram path, explicit analysis-range histogram semantics, and a Qt
monotonic producer-sequence acceptance gate.  No scientific-engine, saved-
product, WCS/registration/reconstruction, callback-signature, or dependency
change.

**R3.1 scope (§8):** make the black and white points **genuinely adjustable**
in the analysis range (including values above 1) and render the Option-A
visible display from the preserved float analysis/WB source with the
user-selected BP/WP applied **before** the final uint8 conversion — a white
point above 1 visibly recovers headroom instead of remaining white-clipped.
Legacy QImage-only payloads keep the historical `[0, 1]` semantics.

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
`p01`, `median`, `p99`, `max`, plus **deterministic-sample counters** `n`,
`under_n`, `over_n`, `zero_n`, `one_n`, and stage-specific fields (`factor`,
`src`, `src_id`, `seq`, `identity`, `res`, `pseq`, `pres`, `wb`, `lo`/`hi`,
`bins`, per-channel max/median). Statistics are computed on a deterministic
fixed-stride subsample (capped at `1_000_000` elements), so tracing never
copies a whole array and never logs per-pixel data. Tests assert records stay
`< 400` chars, contain no newline, and never contain an array dump
(`array(` / `[`).

#### Counter semantics (dtype-aware, unambiguous, bounded)

All counters are computed on the same bounded finite sample (denominator
`n` = number of finite sampled values), so they are comparable across stages:

| Counter | Float analysis buffers | Integer display buffers (uint8) |
|---|---|---|
| `n` | finite sample size | finite sample size |
| `under_n` | count `< 0` (sub-black floor; analysis buffers are floored at 0) | always `0` (unsigned) |
| `over_n` | count `> 1` (**preserved analysis headroom** — PHI-R3) | `> dtype max` → always `0` for uint8 |
| `zero_n` | count `== 0` | count `== 0` |
| `one_n` | count `== 1` exactly (no artificial clip plateau — PHI-R3) | count `== 255` (saturated display) |

For the float analysis stages the plateau/headroom question is answered
directly: headroom exists when `over_n > 0` at `raw_source`.  **R2 (baseline):**
the first stage with `one_n > 0` (and `over_n == 0`) was the first clip site
(`anchor_mapped`).  **PHI-R3 (repaired):** no analysis stage clips —
`over_n > 0` legitimately persists through `anchor_mapped` and `wb_only`
(preserved finite headroom), `one_n` counts only *exact* `== 1.0` values
(no fabricated plateau), and the first place saturation is observable is the
uint8 display boundary (`display_output`, where `one_n` counts `== 255`).  An
analysis stage is therefore never conflated with display saturation.

The arrival records additionally carry the PHI-R3 monotonic-acceptance
outcome: an accepted `payload_arrive` record has no `drop` field; a payload
refused by the run-scoped sequence gate carries `drop=stale` (older emission)
or `drop=duplicate` (repeated emission).

### 1.3 Instrumented stages

Producer side (`seestar/queuep/queue_manager.py`):

| Route | Stages | Array traced | Extras |
|---|---|---|---|
| `classic` (`_update_preview_sum_w`) | `source`, `pre_resize`, `post_resize` | SUM/W divide (full-res) → raw-linear full-res → raw-linear sent | `factor` (effective), `req` (requested), `src=SUM/W`, `src_id=id(memmap)`, `seq` (true producer sequence, == header `PREV_SEQ`) |
| `drizzle` (`_update_preview_drizzle_accumulator`) | `source`, `pre_resize`, `post_resize` | `finalize("divide")` HWC stack → pre-factor → post-factor | `factor` (effective), `req` (requested), `cap` (0/1 max-side guard), `src=Drizzle`, `src_id=id(accumulators)`, `seq` (true producer sequence, == header `PREV_SEQ`) |
| `legacy_drizzle` (`_update_preview_incremental_drizzle`) | `source` | cached `cumulative_drizzle_data` | `factor`, `src=LegacyDrizzle`, `src_id`, `seq` (reachability probe, R1 hypothesis 5; no header metadata) |

Both active producers write the **required PHI production display metadata**
onto the delivered payload on **every emission — independently of the
`ZSSS_PHI_TRACE` debug gate** (PHI-R3.2, F1): only the `PREVIEW_STAGE`
records / stage statistics are debug-gated, never the payload identity:

* `PREV_RUN` — the durable producer run/session id (bound once per stacker
  instance: the Qt run lifecycle assigns it at Start and the backend stamps
  it, or the process-monotonic counter for engine-only use);
* `PREV_SEQ` — a per-stacker monotonic counter bumped once per preview
  emission, shared by the producer trace records;
* `PREV_REQ` — the **requested** GUI resolution factor (clamped 1..4);
* `PREV_RES` — the **effective** factor actually applied to the delivered
  payload (1 when no resize ran: Classic factor 1, the too-small-image guard
  `new_h/new_w <= 10`, or a resize failure — in all those cases the payload is
  delivered at full resolution);
* `PREV_CAP` (Drizzle only) — 1 when the max-side display guard
  (`_MAX_PREVIEW_SIDE_PX`, 1000) resized the array before the GUI factor, 0
  otherwise.

The Qt side reads these from the payload header so the trace sequence is the
true production order, not the Qt-carried `image_count` / `current_batch`
counters (which need not be monotonic in production order).

Qt side (`seestar/gui_qt/main_window.py`, `preview_adjust.py`):

| Stage | Where | Array traced | Extras |
|---|---|---|---|
| `payload_arrive` | `_on_preview` | — (delivered shape via attribute) | `src`, `identity`, `res`, `shape`, `seq`, `pseq`, `prun`, `preq`, `pres`, `pcap`, `drop` (stale/duplicate/foreign on gated drops) |
| `raw_source` | `_ingest_option_a_preview` | `extract_raw_linear` copy | ctx (src/identity/res/shape/pseq/prun/preq/pres/pcap) |
| `anchor_mapped` | `_ingest_option_a_preview` | `map_raw_linear` output | `lo`/`hi` anchors + ctx |
| `wb_only` | `_ensure_wb_only_float` | `apply_wb_float` output (analysis buffer) | `wb` gains + ctx |
| `display_input` | `apply_preview_adjustments` | QImage-derived uint8→float `[0,1]` display buffer entering the tone chain | `stretch`, `bp`, `wp` |
| `display_output` | `apply_preview_adjustments` | final uint8 `[0,255]` display buffer | `stretch` |
| `histogram_output` | `_on_histogram_result` | — (model stats) | `bins`, per-channel `*_max`/`*_med` + **scheduled-time ctx snapshot** |

The helper lives in `seestar/utils/phi_trace.py` (display-only: no science or
Qt imports, lazy numpy, never mutates arrays). The Qt context (`src`,
`identity`, `res`, `shape`, `pseq`, `preq`, `pres`, `pcap`) is stashed in
`_phi_trace_ctx` per payload so stage records are attributable to the arriving
preview and its producer sequence/resolution identity. The `histogram_output`
record uses a **snapshot** of the context taken at schedule time
(`_histogram_trace_ctx`), so an asynchronous worker result is attributed to
the payload that was current when the request was made — never to an obsolete
or newer `_phi_trace_ctx`. `display_input`/`display_output` are explicitly
labelled as the QImage-derived display chain (they are **not** the
`_wb_only_float` analysis buffer, which is traced separately as `wb_only`).

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
# 174 passed (156 pre-existing + 18 PHI tests: 6 original + 8 REWORK-1 + 4 REWORK-1.3)

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_qt_audit_m255f.py tests/test_qt_display_state_otpux.py \
  tests/test_qt_expert_m15.py tests/test_qt_histogram_m14.py \
  tests/test_qt_preview_controls.py tests/test_qt_preview_ergonomics_m17.py \
  tests/test_qt_settings_surface.py tests/test_qt_shell.py \
  tests/test_qt_stacking_m16.py
# 119 passed

python -m py_compile <all modified files>      # OK
git diff --check                               # OK (working tree clean)
git diff --check HEAD                          # OK (corrective diff clean)
git diff --check 9cd8e85..HEAD                 # still fails on the committed doc
                                                # (trailing whitespace at lines 3-5 inside the
                                                # prior commit 63e9b87); the working tree fixes
                                                # it, but the committed range cannot be clean
                                                # without a history rewrite, which is out of
                                                # scope here.
```

## 5. REWORK-1 record

Independent review raised five acceptance failures; all were corrected in the
uncommitted working-tree diff on top of `63e9b87`:

1. **Counters** — `_stage_stats` now emits deterministic-sample counters `n`,
   `under_n`, `over_n`, `zero_n`, `one_n` (dtype-aware: float `[0,1]` domain,
   uint8 display `== 255` exact-one equivalent), documented in §1.2 and tested
   by `test_phi_trace_counters_*`.
2. **True producer sequence** — both active producers bump one per-stacker
   monotonic counter per emission, share it across their trace records, and
   write it as `PREV_SEQ` (plus `PREV_REQ`/`PREV_RES`/`PREV_CAP` — see §5b)
   into the payload header, all strictly gate-dependent (R2 contract; **superseded by
   PHI-R3.2 §9** — the payload identity cards are now unconditional); the Qt trace context
   reads `pseq`/`preq`/`pres`/`pcap` from the header (fallback to
   `image_count`/`current_batch` only when absent). Producer traces and
   payload data now agree by construction.
3. **Attribution** — `histogram_output` uses a scheduled-time ctx snapshot
   (`_histogram_trace_ctx`) so async results never borrow a newer/obsolete
   payload context; the QImage-derived display stages are relabelled
   `display_input`/`display_output` (they are not the `_wb_only_float`
   analysis buffer).
4. **Docs** — trailing whitespace removed from the header lines in the
   working tree (`git diff --check` clean); the committed copy inside
   `63e9b87` still carries it until Junior/Tristan handle the commit, so
   `git diff --check 9cd8e85..HEAD` intentionally still fails (see §4).
   Contract tables updated to the real stages/counters/fields.
5. **Arithmetic** — the exact focused command contains **174** tests (156
   pre-existing + 18 PHI tests: 6 original witnesses + 8 REWORK-1 + 4
   REWORK-1.3 counter/metadata/attribution tests), reported accurately here
   and re-run.

## 5b. REWORK-1.3 record

Junior's independent source review found two material telemetry-metadata
defects; both were corrected in the same uncommitted working-tree diff:

1. **Gate is now a genuine no-op (Defect A).** `_phi_preview_seq` is created /
   mutated and the `PREV_SEQ`/`PREV_REQ`/`PREV_RES`/`PREV_CAP` header cards
   are added **only** when `ZSSS_PHI_TRACE` is enabled. With tracing disabled
   neither active producer touches the attribute or the payload header —
   pre-PHI behaviour is bit-identical (tests
   `test_phi_gate_off_no_producer_metadata_classic` and
   `test_phi_gate_off_no_producer_metadata_drizzle`).
   *Historical R2 record — superseded by PHI-R3.2 (§9.1): the ordering/run
   identity cards are now required production display metadata, emitted
   unconditionally; only the PREVIEW_STAGE records remain debug-gated (the
   old no-metadata tests were rewritten to the new no-trace contract).*
2. **Truthful requested-vs-effective resolution (Defect B).** `PREV_REQ` is
   the requested factor; `PREV_RES` is the factor actually applied (1 when
   Classic's too-small-image guard `new_h/new_w <= 10` skips the resize, when
   the requested factor is 1, or when the resize fails — the payload is then
   full-resolution). Drizzle's max-side display guard is recorded separately
   as `PREV_CAP` (0/1) so the cap-then-factor chain is unambiguous. The Qt
   trace context carries `preq`/`pres`/`pcap` alongside `pseq` with the safe
   legacy fallback (tests
   `test_phi_classic_small_image_no_resize_truthful`,
   `test_phi_drizzle_precap_metadata_truthful`, and the extended
   `test_phi_producer_seq_header_and_trace_agree_*` which now assert delivered
   geometry, not just the requested/clamped factor).

## 6. Remaining uncertainty after R2 (historical) / PHI-R3 recommendation

- The **legacy incremental Drizzle preview** route (`legacy_drizzle`) is now
  instrumented for reachability but no live-workload evidence yet proves it
  fires in production (the standard route is the active one). Still open after
  R3; a real-run trace can decide whether to retire it.
- The **ordering limitation** was confirmed in R2 and is **corrected in PHI-R3**
  (§7.3): a Qt monotonic producer-sequence acceptance gate drops stale/
  duplicate `PREV_SEQ`-carrying emissions within a run; unsequenced legacy
  payloads keep last-wins (W2 remains their witness).
- The plateau location (`anchor_mapped`) and histogram headroom loss were R2
  reproducible diagnostics; the repaired mapping/WB/histogram semantics are
  **implemented in PHI-R3** (§7.1/§7.2) — the R2 W1 diagnostic is superseded
  by the R3 headroom-preservation witness.

## 7. PHI-R3 record — behavioural integrity correction (uncommitted working tree)

R3 implements the product contract approved by Tristan (three items) as the
smallest production-safe change on top of the R2 working tree.  **Producer
code is untouched**: the R2 gate-off no-op guarantee (no `PREV_*` card, no
`_phi_preview_seq` attribute when `ZSSS_PHI_TRACE` is off) and all R2
producer/trace contracts remain byte-identical.

### 7.1 Headroom preservation (contract items 1-2)

`seestar/gui_qt/preview_analysis.py`:

- `map_raw_linear` no longer clips to `[0, 1]`: ``mapped = (raw - lo)/(hi - lo)``
  keeps finite out-of-range values verbatim.  Sub-black mapped values (a pure
  anchor-mapping artifact below the display floor) floor at exactly `0.0`
  (identical to the pre-R3 low-side clip) and non-finite results (NaN/Inf
  input, float64 overflow) sanitize to `0.0` — **no NaN/Inf propagation**,
  buffers are finite by construction.  Non-mutating.
- `apply_wb_float` no longer clips the per-channel gains to `[0, 1]`: strong
  gains legitimately produce analysis values `> 1`; the same finite/floor/
  sanitize guarantees apply.
- **R3 boundary note:** the bounded conversion was, at R3 time, the
  ``QImage``/uint8 boundary.  **PHI-R3.1 (§8) replaced the Option-A display
  source**: the visible preview is now rendered from the preserved float
  analysis/WB buffer with the user BP/WP applied in float *before* the final
  uint8 conversion (the stored neutral ``QImage`` from
  ``preview_render._to_uint8`` remains only as a geometry/fallback carrier).
  Saved FITS/PNG and scientific paths are untouched (no producer/science-file
  change).

### 7.2 Explicit histogram analysis-range semantics (contract item 1, criterion B)

`compute_histogram_float` counts/stats now describe the **preserved analysis
population**: 512 bins over `(0, upper)` with
`upper = max(1.0, finite max)` over the buffer.  With no headroom `upper == 1.0`
and the model is bit-identical to the pre-R3 `[0, 1]` contract (bin 0/256/511
placements unchanged); with HDR/WB headroom `upper` extends so headroom bins,
per-channel stats (`max > 1`) and `full_range` are truthful.  Counts and all
five stats share the exact same finite non-negative sample (sub-black and
non-finite excluded from both; fail-closed `None` for an unusable required
channel, unchanged).  The model/stats are explicitly **analysis data** (never
claimed as the bounded uint8 display histogram): `format_histogram_stats` and
the `histogram_output` trace show the analysis max, distinguishing preserved
headroom from display saturation by construction.  Auto Stretch / Auto WB
outputs are unchanged (their samples already exclude `>= 1` / `>= 0.98`).

`seestar/gui_qt/histogram_view.py` becomes model-range aware (no broad
renderer rewrite): float-model bars are placed in the model's declared range;
the initial view stays the display-level window `[0, 1]` (BP/WP live there);
`reset_histogram_view`/`reset_zoom` reveal the full analysis range
(identical to `[0, 1]` when no headroom exists); auto/manual zoom validates
the robust X range against the model upper bound.  Legacy 256-bin `QImage`
histograms keep the historical `[0, 1]` display-level window unchanged.

### 7.3 Monotonic acceptance gate (contract item 3, criterion C)

`seestar/gui_qt/main_window.py` `_on_preview`: when the payload header carries
a producer `PREV_SEQ`, acceptance is monotonic **within a run**: the first
sequenced payload opens the gate; a strictly newer sequence advances it;
an equal sequence is a **duplicate** and an older one is **stale** — both are
dropped at arrival, before the label/analysis/display state changes or any
histogram work is scheduled (a gated drop emits `payload_arrive` with
`drop=stale|duplicate` when tracing is on; `_phi_trace_ctx` keeps describing
the last *accepted* payload).  Payloads without `PREV_SEQ` (legacy
producers/tests, initial-preview loads) bypass the gate and keep the
historical unconditional last-wins acceptance.  The gate
(`_last_accepted_preview_seq`) resets in `_on_run_started` (the run/lifecycle
boundary; a fresh per-run stacker restarts its counter at 1), so a new run's
first sequenced payload is never rejected.  The gate is driven by the
metadata card's presence, not by the `ZSSS_PHI_TRACE` debug gate.

Known residual — **RESOLVED in PHI-R3.2 (§9.1)**: the R3 gate was inert in
trace-off production (producers stamped the cards only under `ZSSS_PHI_TRACE`)
and the per-run high-water reset alone could not distinguish a late old-run
payload.  R3.2 makes `PREV_SEQ`/`PREV_RUN` required payload metadata
(trace-independent) and binds a durable producer run/session id at the run
lifecycle boundary, so a foreign (other-session) payload is dropped in either
arrival order and can never poison the current run's high-water mark.

### 7.4 Tests (all deterministic, offscreen)

- `tests/test_phi_preview_pipeline.py` — R2 W1 rewritten as the R3 headroom
  witness; W2 kept as the legacy last-wins witness; new gate tests (stale /
  duplicate / reordered / run reset / legacy bypass / metadata-driven not
  trace-gated) and the pipeline headroom-to-histogram regression with display
  boundedness; 24 tests total.
- `tests/test_preview_analysis.py` — updated to the R3 analysis-domain
  semantics (headroom-preserving map/WB, explicit analysis range, sanitization)
  and new HDR-range tests; 40 tests total.
- `tests/test_qt_otpux_final_ux.py` — reset-view assertion updated to the full
  analysis range (surface sync semantics unchanged).

Validation (exact commands and results in the PHI-R3 report): focused PHI +
preview/histogram/Qt suites green (182 + 192 + 64 …); `py_compile` clean;
`git diff --check` clean.  Pre-existing `tests/test_resume.py` failures (4,
reference-materialization fixtures) reproduce identically without the R3 diff
and are unrelated to PHI.

## 8. PHI-R3.1 record — analysis-unit BP/WP + float display rendering (uncommitted working tree)

Junior's independent review rejected R3 because the extended histogram alone
could not repair the preview: (a) BP/WP stayed hard-limited to `[0, 1]` by the
normalisation seams, and (b) Option-A ingestion still converted the pristine
float to `QImage`/uint8 **before** the stretch chain, so analysis headroom was
clipped before any user white point could map it back into visible range.
R3.1 removes both limitations.

### 8.1 Unit contract (the actual units of the analysis histogram and BP/WP)

- **Analysis units.**  The Option-A float analysis domain is `[0, upper]`,
  `upper = max(1.0, finite max)` of the WB-only analysis buffer.  The float
  histogram (model `range`/`full_range`), the BP/WP display controls
  (sliders/spins/markers/drags) and the display stretch parameters all speak
  this one unit system.  The *control domain* is the grid ceiling of the raw
  upper (`ceil(upper / 0.001) * 0.001`) because the 0.001-grid widgets cannot
  represent a finer bound — never rounded down, so the true data top is always
  reachable.
- **Legacy units.**  Single-array (non-Option-A) payloads keep the historical
  Tk-parity `[0, 1]` display-level domain everywhere (controls, markers,
  stretch chain) — bit-identical behaviour.
- **BP/WP semantics (Option-A).**  The black point is the analysis value
  mapped to display black, the white point the analysis value mapped to
  display white.  Both are first-class values above `1`; the marker positions,
  the spin/slider values, the drag coordinate conversion and the stretch curve
  reference always agree (no silent remapping).

### 8.2 Float display rendering (defect b)

- `preview_adjust.render_analysis_display(analysis, stretch, bp, wp, gamma,
  bcs)` renders the visible Option-A display from the **WB-only float analysis
  buffer** with **fixed-reference analysis-unit stretch curves** (linear:
  `clip(shift/win, 0, 1)`; asinh/log: the same `bp..wp` window mapped through
  the fixed-gain curve — `wp` is the white level for every mode, unlike the
  legacy adaptive QImage chain which normalises by the data maximum; `auto`:
  fill-range).  Gamma and brightness/contrast/saturation apply in float after
  the stretch; `np.rint(clip(y,0,1)*255).astype(uint8)` → `QImage` is the
  **only clipping boundary** of the display path.
- `MainWindow._refresh_preview_view` uses this float path whenever Option-A
  float state is present (the WB-only buffer is shared with the histogram —
  no double WB).  A white point above `1` visibly recovers preserved headroom
  instead of leaving it white-clipped.  Legacy payloads keep
  `apply_preview_adjustments`.
- Trace: the Option-A `display_input` record is the preserved float
  analysis/WB buffer entering the tone chain (headroom `> 1` possible —
  analysis data, not a pre-quantized `[0,1]` buffer); `display_output` is the
  bounded uint8 screen domain.

### 8.3 Analysis-unit BP/WP controls (defect a)

- `preview_adjust.normalize_bp_wp` / `quantize_bp_wp` / `clamp_bp_wp_edit`
  take an explicit `max_value` domain (`1.0` legacy default; analysis grid
  upper for Option-A) — same deterministic seams, both paths.
- `MainWindow` tracks `_analysis_upper` (raw, `analysis_upper_bound`, the
  exact same deterministic value the histogram model declares) and the
  grid-ceiling `_bp_wp_control_upper`; `_sync_analysis_domain()` (run on every
  WB-only derivation, i.e. new source or WB change) retools both BP/WP spin
  ranges and slider ranges, reconciles the current pair deterministically in
  analysis units without inversion (a white point above the new upper is
  pulled down to it; a later larger range never silently restores it), and
  re-scopes every live histogram view's marker domain
  (`HistogramView.set_analysis_domain`).  Runs under the BP/WP re-entrancy
  guard; no-op when the domain is unchanged.
- `HistogramView`: markers and drags operate over `[0, marker_upper]` (the
  MainWindow-pushed grid-ceiling domain; `set_model` deliberately does not
  shrink it, so an async model application cannot race the synchronous
  controls).  `set_range` / drag clamp / grid quantization all use the marker
  domain; legacy `set_data`/`set_legacy_data` reset it to `1.0`.
- Detached histogram surface is re-scoped identically (open + refresh paths).
- Auto Stretch keeps its existing normalised default policy (BP/WP from the
  `(0,1)`-sample percentiles, `<= 1`, then written through the analysis-unit
  seams); manual histogram/controls can use the full analysis range.

### 8.4 New R3.1 tests (tests/test_phi_preview_pipeline.py, +6)

`test_phi_r31_histogram_range_exceeds_one_and_controls_scope` (E-i),
`test_phi_r31_wp_above_one_set_and_dragged_synced_everywhere` (E-ii: spin edit
+ histogram drag to WP 2.0/2.5 synchronized across inline + detached views and
the spin/slider controls), `test_phi_r31_float_display_recovers_headroom_via_white_point`
(E-iii: linear WP=1 clips the star headroom white; WP=3 recovers it — pixel-
exact midtones matching `round(x/3*255)` and fewer saturated pixels),
`test_phi_r31_bp_wp_validate_and_order_in_extended_range` (E-iv),
`test_phi_r31_legacy_path_keeps_01_domain` (E-v), and
`test_phi_r31_analysis_domain_shrink_reconciles_without_inversion` (D).
The R2 display-stage test now distinguishes the Option-A float chain (analysis
`display_input`) from the legacy QImage chain (`[0,1]` input).

### 8.5 Validation (see the PHI-R3.1 report)

Focused PHI/analysis/histogram/Qt suites green (825 passed across the full Qt
+ preview + PHI sweep — 819 pre-R3.1 + 6 new tests); `py_compile` clean;
`git diff --check` clean.  `tests/test_resume.py` pre-existing failures (4,
reference-materialization fixtures) reproduce identically without any PHI diff.

## 9. PHI-R3.2 record — REWORK-R3.2 (F1 trace-independent producer gate; F2 frozen-view reconciliation)

Nono review-2 (report `PHI-LIVE-PREVIEW-INTEGRITY-V2.nono.r2.md`, both
findings independently confirmed by Junior) found: (F1, HIGH) the monotonic
producer gate was dead in normal production because the active producers
stamped `PREV_SEQ`/`PREV_*` only under `ZSSS_PHI_TRACE`, so trace-off payloads
were unsequenced and kept last-wins — and the per-run high-water reset alone
could not stop a late old-run payload poisoning a new run; (F2, MED/HIGH)
`HistogramView.set_model` blindly restored `_frozen_range` after a model
domain shrink, leaving a stale window beyond the data domain and desyncing
inline/detached surfaces.  R3.2 corrects both.

### 9.1 F1 — trace-independent producer ordering/run identity (production gate)

**Producer side (`seestar/queuep/queue_manager.py`).**  Both active producers
(Classic `_update_preview_sum_w`, standard Drizzle
`_update_preview_drizzle_accumulator`) now stamp the **required payload
ordering/run identity metadata on every emission**, independent of the debug
gate:

* `PREV_RUN` — durable producer run/session id (new).  Bound once per stacker
  instance via `_phi_producer_session`: the Qt run lifecycle assigns a
  per-window monotonic id at Start and the backend stamps it onto the stacker
  at construction; engine-only usage falls back to a process-wide monotonic
  counter (`itertools.count`, never reused in the process);
* `PREV_SEQ` — per-emission monotonic sequence (bumped unconditionally);
* `PREV_REQ`/`PREV_RES`/`PREV_CAP` — resolution identity cards, also
  unconditional now (header shape is identical with or without
  `ZSSS_PHI_TRACE`).

Only the `PREVIEW_STAGE` debug records / stage statistics remain
debug-gated (genuine no-op with the gate off: zero records, zero telemetry
state).  Callback signature, payload tuple shape, science accumulation, FITS,
saved PNG, WCS and registration/reconstruction are untouched.

**Run-lifecycle binding.**  `MainWindow._on_start` allocates
`_next_preview_run_session` and hands it to the real backend
(`SeestarQueuedStackerBackend.set_preview_session`); the backend stamps it on
the stacker at construction (`_ensure_stackers`); `_on_run_started` binds the
expected id (`_preview_run_session`) from the pending id and resets the
sequence high-water mark.  Simulated/legacy backends (no seam) stay
unsequenced and keep the legacy fallback.

**Qt acceptance rule (`_accept_preview_payload`).**  For every payload
carrying `PREV_SEQ`: (1) when `PREV_RUN` is present and the run has a bound
session, a payload of any other session is **foreign** — dropped without
touching the high-water mark (a late old-run payload can never poison the new
run, in either arrival order); when no session is bound yet the first
sequenced payload's run is bound lazily (idle/direct-call/test usage); (2)
the monotonic sequence rule then applies within the bound run (first accepts;
strictly newer advances; equal = duplicate, older = stale — both dropped).
Sequenced payloads without `PREV_RUN` (third-party/synthetic) fall back to
the run-scoped monotonic gate only.  Unsequenced payloads bypass the gate
unchanged.  Trace `payload_arrive` records carry `prun` and
`drop=stale|duplicate|foreign`.

### 9.2 F2 — frozen/manual view-range reconciliation on model-domain change

`HistogramView.set_model` now revalidates a frozen/manual view range against
the new model analysis upper *before* `_apply_view_after_data` restores it
(`_reconcile_range_to_upper`): a still-valid manual range is preserved
verbatim (shrink or grow), an out-of-domain window is clamped into the new
domain, and only a window that becomes degenerate after clamping falls back
to the full analysis range — the painted axis never extends beyond the data
domain, never inverts.  Because both inline and detached surfaces receive the
same model and reconcile identically (and `_sync_detached_histogram` copies
the inline result through the same validation), the two surfaces cannot
desync after a domain change.  BP/WP markers/controls keep their R3.1
analysis-domain reconciliation (unchanged).

### 9.3 Tests (all deterministic, offscreen)

- `test_phi_r32_producer_identity_metadata_present_trace_off_classic/drizzle`
  — trace-off producer delivery still stamps PREV_SEQ/PREV_RUN (+resolution
  cards), sequences monotonic, one run id per stacker, distinct ids across
  stackers, zero PREVIEW_STAGE records (telemetry no-op preserved).
- `test_phi_r32_producer_session_preset_by_run_lifecycle_is_stamped` — a
  run-lifecycle-assigned session id is honoured by the producer.
- `test_phi_r32_run_bound_gate_same_run_stale_and_duplicate` — bound-run
  monotonic gating.
- `test_phi_r32_foreign_old_run_payload_cannot_poison_new_run` — cross-run
  safety in both arrival orders (old-before-new and the late-old-after-new
  poison scenario), foreign drop trace with `prun`.
- `test_phi_r32_legacy_unsequenced_still_bypasses_when_run_bound`,
  `test_phi_r32_start_assigns_session_and_binds_at_run_start` — lifecycle
  binding through `_on_start`/`_on_run_started` with a PHI-capable backend.
- F2: `test_phi_r32_frozen_view_reconciled_on_model_shrink` (Nono repro),
  `test_phi_r32_frozen_view_preserved_on_grow_and_valid_shrink`,
  `test_phi_r32_inline_and_detached_ranges_stay_synced_after_shrink`.

Test file totals: `tests/test_phi_preview_pipeline.py` now 38 tests
(18 R2/REWORK + 6 R3 + 6 R3.1 + 8 R3.2).  The R2 "gate-off no-op" producer
tests were rewritten to the new no-trace contract (metadata present, records
absent) — the old assertions validated the F1 residual.

### 9.4 Validation

Focused PHI + full Qt/preview sweep: 833 passed (825 pre-R3.2 + 8 new);
`py_compile` clean; `git diff --check` clean.  Pre-existing (proven by stash
probe, unrelated to PHI): `tests/test_resume.py` (4) and
`tests/test_reproject_zm_wcs_fix.py::test_mode0_final_keeps_science_with_artifact_reference_header`
+ `tests/test_boring_drizzle_boundary.py::test_classic_memmaps_use_fixed_reference_grid_shape`
fail identically without the R3.2 diff.

## 10. PHI-R3.3 record — REWORK-R3.3 (final corrective iteration: F2 detached policy mirroring, F3 model→legacy view reset, backend stamping test)

Nono review-3 (report `PHI-LIVE-PREVIEW-INTEGRITY-V2.nono.r3.md`) accepted the
F1 producer identity for the standard lifecycle and found two integration/
compatibility view-state defects plus one coverage gap.

### 10.1 F2 — explicit frozen-state policy propagation to the detached surface

`HistogramView.set_view_range` no longer manufactures a frozen range from a
coordinate snapshot (snapshot semantics).  New
`HistogramView.mirror_state_from(other)` copies the **policy** — view window,
frozen-vs-unfrozen state (a genuine manual/robust zoom verbatim; unfrozen
stays unfrozen — never an artificial freeze) and the auto-zoom flag — from
the authoritative inline view onto the detached surface.  MainWindow calls it
(`_mirror_detached_view_policy`) whenever the detached view is (re)synchronized
(`_sync_detached_histogram`), after every float-model application
(`_on_histogram_result`) and after legacy data installation
(`_refresh_histogram_legacy`).  Both surfaces therefore deterministically end
with the same valid analysis view range and frozen policy after any model
shrink/grow — the Nono open-after-unfrozen-reset divergence is gone.

### 10.2 F3 — model→legacy transitions reset the view to a valid legacy `[0,1]`

`HistogramView.set_data`/`set_legacy_data` now clear any frozen/manual view
state **when the surface previously held a float model** (model→legacy
transition) and restore a valid legacy `[0, 1]` window automatically.
Legacy→legacy data refreshes keep the historical manual-zoom-preservation
semantics (M15/expert contract).  Applied per surface, so inline and detached
both reset; MainWindow additionally mirrors the inline policy afterwards.

### 10.3 Coverage gap — real backend stamping under trace off

`test_phi_r33_real_backend_stamps_session_and_producer_emits_trace_off`
constructs a real `SeestarQueuedStackerBackend` with a fixture `stacker_factory`
(no real engine/hardware/user data), assigns the GUI session via
`set_preview_session`, lets the real `_ensure_stackers` create and stamp the
stacker, then drives a genuine Classic producer emission (`_update_preview_sum_w`
bound onto the fixture) with `ZSSS_PHI_TRACE` unset: the delivered callback
header carries the assigned `PREV_RUN` + `PREV_SEQ` (and resolution cards)
while zero `PREVIEW_STAGE` records are emitted.  Reused-backend re-stamping
(`set_preview_session` on an existing stacker) is also covered — the seam now
re-stamps an already-created stacker so a reused backend can never keep a
previous run's session id.

### 10.4 Tests / validation

+4 tests in `tests/test_phi_preview_pipeline.py` (42 total):
`test_phi_r33_detached_open_after_unfrozen_reset_stays_in_sync` (Nono F2
repro), `test_phi_r33_detached_genuine_manual_frozen_reconciled_on_shrink`,
`test_phi_r33_option_a_to_legacy_transition_resets_views` (Nono F3 repro),
`test_phi_r33_real_backend_stamps_session_and_producer_emits_trace_off`.
Full Qt + preview + PHI sweep: 837 passed; `py_compile` clean;
`git diff --check` clean.  Pre-existing (proven by stash probes, unrelated to
PHI): `tests/test_resume.py` (4), `tests/test_reproject_zm_wcs_fix.py`
(1) and `tests/test_boring_drizzle_boundary.py` (1) fail identically without
the PHI diff.

## 11. PHI-R4 record — preview route reachability and integrity parity

### 11.1 Route/dispatch inventory (live preview producer paths → Qt ingest)

All live preview emissions happen on the stacker worker thread during
`start_processing`; the Qt GUI receives them through the backend preview
callback adapter → `BackendPreviewPayload` → `MainWindow._on_preview`.

| # | Route (Qt mode) | Producer fn | Live dispatch predicate (exact site) | Payload / identity |
|---|---|---|---|---|
| 1 | `classic` | `_update_preview_sum_w` | non-drizzle session (`not drizzle_active_session`) after every completed classic batch: `_worker` (7542), `_process_completed_batch` (10413), `_flush_current_batch` (10527), tail-batch `_worker` (8073); and `refresh_preview` else-branch (5004) for live resolution/control refreshes | Option-A tuple `(legacy_normalized, raw_linear)`; `PREV_SRC="SUM/W Accumulators"`; `PREV_RUN`/`PREV_SEQ`/`PREV_REQ`/`PREV_RES` always (R3.2) |
| 2 | `drizzle` | `_update_preview_drizzle_accumulator` | `drizzle_active_session` + `drizzle_accumulators`: per accepted pose in standard policy and per completed group in incremental policy (`_drizzle_group_tick` 21755/21763), trailing partial group (`_drizzle_flush_partial_group` 21780), and `refresh_preview` if-branch (5001).  `drizzle_processing_policy` (derived from legacy `drizzle_mode`, which is inert for science) changes **cadence only** — both policies use the same single-accumulator display route | Option-A tuple; `PREV_SRC="Drizzle Accumulator"`; `PREV_RUN`/`PREV_SEQ`/`PREV_REQ`/`PREV_RES`/`PREV_CAP` always |
| 3 | legacy incremental (forensic) | `_update_preview_incremental_drizzle` (cumulative-display-data based) | **UNREACHABLE (dead/guarded dispatch).**  Its only call site (11299) sits inside `_process_incremental_drizzle_batch`, which is reached only via `drizzle_batch_worker` ← `_start_drizzle_process` — all three are M3-D OBSOLETE LEGACY with zero live callers; `incremental_drizzle_objects` is always `[]` in M3 (init 3938, cleared at run start 4573) so even the direct guard (`preview_callback and incremental_drizzle_objects`) is false | would be an Option-A-shaped tuple over *display-domain cached* `cumulative_drizzle_data` with **no** PHI identity cards and no `PREV_SRC` — a resurrected legacy route would need an explicit non-Option-A compatibility contract; recorded, not modified (no speculative change to dead code) |
| 4 | reproject/master carrier | `_update_preview` / `_update_preview_master` | **Dead/guarded no-op**: `_update_preview_master` is only called by `_incremental_reproject_coadd`, which has no live callers; its data writers (`current_stack_data`/`_raw`) are only ever set by the two dead paths, so the live `_update_preview` call (add-folder prequeue 19527) early-returns on `current_stack_data is None` | n/a (no live emission); legacy tuple carrier if ever resurrected |
| 5 | legacy single-array (Qt side) | — | payloads whose `data` is a lone array / non-Option-A tuple (all non-PHI producers/tests, initial-preview loads) | legacy QImage render path, `[0,1]` semantics, unsequenced acceptance |

Qt ingest predicates: `_is_option_a_preview_payload` (structural second-element
image array) selects the Option-A float pipeline; `_derive_preview_mode`
labels `classic`/`drizzle`/`reproject` from `PREV_SRC`; the R3.2 acceptance
gate applies to `PREV_SEQ`/`PREV_RUN`-carrying payloads (routes 1-2), while
legacy/unsequenced payloads keep the documented unconditional fallback.

### 11.2 Supported contract per route + residuals

- Route 1 (Classic) and route 2 (Drizzle): full Option-A float pipeline with
  trace-independent producer identity — ordering/attribution contract applies.
- Routes 3-4 were dead/guarded legacy: recorded and tested as such in R4, and
  **retired in PHI-R5** (§12) after Tristan's explicit approval of the human
  gate (no speculative behavioural removal occurred before that approval).
- Residuals: (a) if route 3 or 4 were ever resurrected, its tuple payload
  would be Option-A-shaped without identity metadata and without truthful
  raw-linear provenance — it must then receive an explicit non-Option-A
  compatibility contract before use; (b) `refresh_preview` falls back to the
  SUM/W route when a drizzle session has no accumulators yet (bootstrap edge,
  same contract as route 1); (c) the legacy `_update_preview` (route 4)
  carries no `PREV_SRC` and would be labelled `reproject` — currently a no-op.
- No source-buffer mutation, no cumulative-resize reintroduction: both active
  producers re-derive directly from authoritative sources per emission (R2
  direct-source resolution witnesses still green).

### 11.3 Tests (deterministic, dispatch predicates)

+4 tests in `tests/test_phi_preview_pipeline.py` (46 total):
`test_phi_r4_refresh_preview_dispatches_on_session_predicate`,
`test_phi_r4_drizzle_preview_cadence_both_policies_use_accumulator_route`,
`test_phi_r4_legacy_incremental_drizzle_is_dead_guarded_dispatch` (AST-based
static dead-subgraph guard + runtime spy proof across supported dispatch),
`test_phi_r4_route_labels_distinguish_active_producers`.  Validation: focused
PHI/producer/Drizzle/backend/Qt preview suites + full sweep — see the PHI-R4
report (`PHI-LIVE-PREVIEW-INTEGRITY-V2.coco.r4.md`).

### 11.4 R5 target (measured)

Reachability was proven in R4: the legacy incremental Drizzle preview and the
reproject/master carrier are dead/guarded in supported flows.  The R5 human
gate (retire the M3-D OBSOLETE LEGACY double-pass drizzle machinery and the
dead reproject/master preview carrier) was **approved by Tristan and executed
in PHI-R5** — see §12.  Remaining measured item: (c) confirm real-run
(non-offscreen) trace of route 1/2 cadence under load (R6 candidate).

## 12. PHI-R5 record — approved retirement of the dead legacy preview machinery

Tristan explicitly approved the R5 human gate (remove the M3-D obsolete legacy
incremental Drizzle preview/process chain and the dead reproject/master
preview carrier; migrate forensic tests/coverage; validate).  All R1-R4 and
checkpoint-startup working-tree work preserved; no commit.

### 12.1 Removed (queue_manager.py, after repository-wide call-graph proof)

Dead legacy incremental Drizzle preview/process chain and its sole-purpose
state:

- `drizzle_batch_worker` (module fn), `_start_drizzle_process`,
  `_process_incremental_drizzle_batch` (incl. its nested channel closures and
  the embedded preview block), `_update_preview_incremental_drizzle`,
  `_wait_drizzle_processes` — the closed M3-D OBSOLETE LEGACY subgraph with
  zero live callers (R4 AST proof), plus the
  `_DRZ_PREV_MIN_DT`/`_last_drz_prev` throttle globals that existed solely for
  that chain.  (The `_drz_batch_version_string` helper and
  `GLOBAL_DRZ_BATCH_VERSION_STRING_ULTRA_DEBUG` were initially removed with
  it, then **restored**: the constant is supported cross-module display API —
  the Tk GUI imports it as `APP_VERSION` and `test_version_consistency.py`
  pins its derivation — only its legacy debug-log consumers were retired.)
- Dead reproject/master preview carrier: `_update_preview`,
  `_update_preview_master`, `_incremental_reproject_coadd` (its only live
  call site was the folder-add prequeue, which is now a plain queue/ETA
  refresh), and `_save_intermediate_stack` (existed solely to write the
  `current_stack_data` carrier);
- Sole-purpose state/attributes: `incremental_drizzle_objects`,
  `incremental_drizzle_sci_arrays`, `incremental_drizzle_wht_arrays`,
  `intermediate_drizzle_batch_files`, `cumulative_drizzle_data*`
  (incl. the two store lines in the ACTIVE drizzle producer — nothing read
  them once the carrier was gone), `current_stack_data_raw`,
  `master_sum`/`master_coverage`, `reproject_output_wcs`, `drizzle_processes`,
  `drizzle_executor` (incl. its `__init__` pool creation, `__getstate__`
  entries and the active producer's display-artifact stores);
- Dead imports: `drizzle_finalize`, the `incremental_reprojection` import
  tuple (`reproject_and_coadd_batch`, `reproject_and_combine`,
  `initialize_master`).

**Preserved untouched (supported):** Classic SUM/W producer + worker batch
paths, standard Drizzle single-accumulator producer and cadence
(`_drizzle_group_tick`/`_drizzle_flush_partial_group`, `refresh_preview`
session predicate), reproject-between-batches / reproject-coadd-final
finalization (`_reproject_classic_batches_zm`, `_finalize_single_classic_batch`,
`_combine_batch_result`), `_save_final_stack`, FITS/PNG saving, WCS,
registration, the quality executor, and `seestar/core/incremental_reprojection.py`
(real science core).  No reachable science code was deleted because its
preview helper was obsolete.

### 12.2 Companion edits (supported code referencing the removed symbols)

- `seestar/gui/boring_stack.py`: `_cleanup_stacker` no longer calls
  `_wait_drizzle_processes()` and drains only the quality executor (the
  classic SUM/W boring route never created a drizzle session).
- Tests: `test_boring_drizzle_boundary.py` fixture/cleanup/source-audit
  rewritten to the retirement contract; `test_frozen_reference_handoff.py`,
  `test_resume_intent_contract_rsm2.py`, `test_reliability_source_immutability_r1.py`
  drop the retired executor from teardown; `test_worker_incremental_drizzle.py`
  drops the retired negative spies; `test_save_final_stack.py` — the two
  forensic `_process_incremental_drizzle_batch` tests were **migrated** to a
  positive supported-path invariant test (active standard-Drizzle preview
  producer emits the Option-A identity payload with trace off and leaves the
  per-channel accumulators bit-identical; the supported final FITS save from
  those accumulators preserves the per-channel science);
  `test_phi_preview_pipeline.py` — the R4 dead-dispatch test was replaced by
  the R5 retirement regression (`test_phi_r5_legacy_machinery_retired_no_supported_dispatch_invokes_it`:
  retired symbols absent from source, supported refresh/tick dispatch safe,
  no speculative fallback).

### 12.3 Supported-route regression coverage (kept + added)

Classic / standard Drizzle refresh safety is covered by
`test_phi_r4_refresh_preview_dispatches_on_session_predicate`,
`test_phi_r4_drizzle_preview_cadence_both_policies_use_accumulator_route`,
`test_phi_r5_legacy_machinery_retired_no_supported_dispatch_invokes_it`,
the producer isolation/identity witnesses, and the migrated
`test_supported_drizzle_preview_and_final_save_invariants`.

### 12.4 Validation

Focused PHI (46), producer/raw-linear, analysis, save-final-stack, boring
boundary, worker-incremental, drizzle-native and the full Qt + preview + PHI
sweep green (see the PHI-R5 report `PHI-LIVE-PREVIEW-INTEGRITY-V2.coco.r5.md`);
`py_compile` clean; `git diff --check` clean.  Pre-existing (proven by stash
probe, identical without the R5 diff): `tests/test_resume.py` (4),
`tests/test_reproject_zm_wcs_fix.py` (1),
`tests/test_boring_drizzle_boundary.py::test_classic_memmaps_use_fixed_reference_grid_shape`,
`tests/test_save_final_stack.py::test_save_final_stack_radec_from_reference_header`,
`tests/test_worker_incremental_drizzle.py::test_worker_calls_add_frame_to_drizzle_accumulators`.

### 12.5 Remaining limitations (accurate)

No real-hardware/GPU run was performed (offscreen + fixture evidence only);
real-run cadence tracing of routes 1-2 under load remains an R6 candidate.

## 13. PHI-AUTO-HISTOGRAM-UX-V1 record — robust plot/bin domain + analysis-unit Auto Stretch

**Status:** implemented on the same uncommitted working tree
(`fix/live-preview-histogram-v2`, HEAD `63e9b875e4db140333f2cb8691235f54bc629f4f`).
Corrects the R3-era histogram/auto-stretch semantics that the user production
evidence exposed (screenshots, not instructions): the float histogram in the
extended analysis domain degraded to a few widely spaced vertical spikes on a
sparse extreme finite tail (maxima ~R282/G222/B151 while the robust window is
~0.25–2.36), and Live Auto Stretch left the bright nebula/core saturated even
though the X axis can exceed 1.  No regression of the accepted PHI-R3.1
manual BP/WP (>1) pathway; the manual above-1 controls, the float pre-uint8
render path, the legacy-transition handling, the trace-independent producer
identity gate and the active Classic/standard-Drizzle contracts are intact.

### 13.1 Exact semantics — Auto Stretch (analysis units, no hidden [0,1] cap)

`compute_auto_stretch_float` now operates on the WB-only float analysis buffer
**in its preserved analysis units**:

1. Input sample S = finite mapped values with `mask > 0` (when given),
   excluding only the exact legacy clip boundaries `0.0` and `1.0` — every
   finite value **above 1** (preserved HDR headroom) is kept.
2. Percentiles/background-MAD steps are unchanged (`p005`/`p60`/`p995`,
   `B = {s <= p60}`, `bg = median(B)`, `sigma = 1.4826 * MAD(B)`,
   `bp = max(p005, bg - 2.8σ)`, `wp_raw = max(p995, bg + 8σ, bp + sep)`; no
   min/max renormalization anywhere; deterministic capped sample).
3. Final separation clip replaces the hard `[0, 1]` display-window ceiling by
   `D = max(1.0, p99.5(S))`:
   `bp = clip(bp_raw, 0, D - sep)`, `wp = clip(wp_raw, bp + sep, D)`.
   - In-window `[0,1]` buffers: `p99.5(S) <= 1` ⇒ `D == 1.0` ⇒ bit-identical
     to the ratified §5.5 output (legacy QImage `compute_auto_stretch`
     unchanged).
   - Meaningful dense bright tail above 1 ⇒ `D > 1` ⇒ the estimator selects
     `wp > 1` and the fixed-reference analysis-unit stretch visibly recovers
     non-saturated highlight structure.
   - Isolated extreme outlier ⇒ p99.5 is outlier-robust ⇒ WP is never
     expanded to the outlier/max.
4. BP/WP remain ordered/quantized within the current analysis domain by the
   shared `normalize_bp_wp`/`quantize_bp_wp` seams; live and one-shot Auto
   Stretch call the same deterministic function on the same WB-only buffer.

### 13.2 Exact semantics — histogram plot/bin range vs analysis domain

The 512-bin histogram model now declares **distinct roles** (no mislabelling):

* `range`/`full_range` = the full preserved **analysis/control domain**
  `(0, upper)`, `upper = max(1.0, finite max)` — stats (incl. the tail `max`),
  the BP/WP marker domain and the F2 frozen-range reconcile upper.  Unchanged.
* `bin_range = (0, bin_hi)` — the **plotting/bin range** the 512
  `counts`/`log_counts` bins live in.  `bin_hi` is:
  - the full analysis upper when the top is *dense* (finite max within
    `1.25x` of the robust 99.5 % top, i.e. a continuous population that
    really reaches its max) or the in-domain sample is small (`<= 512`
    values — fewer than one value per bin, so a single top value is not a
    distinguishable sparse tail): those cases are bit-identical to the
    legacy/R3 binning (zero overflow); or
  - the robust top percentile (p99.5 of the same deterministic concatenated
    in-domain sample used for the robust viewport) when a *sparse extreme far
    tail* exists — so the auto/default visible window holds most of the 512
    bins instead of ~4 widely spaced spikes.
* `x_range` = robust viewport (p0.5..p99.5 of the same sample): always at or
  below the bin high, so auto zoom stays inside the binned domain.
* `overflow` (per channel) and `overflow_total` count the in-domain values
  strictly above `bin_hi` — the tail is never silently dropped; its presence
  and extent stay truthful via `full_range`, the per-channel stats `max` (the
  UI status line) and a plot-top marker drawn by the histogram widget at
  `bin_hi` when the view shows it.
* The widget paints bars over `bin_range`, keeps marker/reset/zoom domains on
  the analysis range, and inline/detached surfaces receive the same model
  object (mirror policy unchanged).

### 13.3 Tests / validation

Focused deterministic suite `tests/test_phi_auto_histogram_ux_v1.py`
(capped-auto regression, robust auto >1 / anti-outlier, legacy in-window
parity, dense plotted-bin occupancy with sparse extreme outliers + truthful
tail metadata, explicit model roles, small-sample full-range parity, widget
bar-domain/overflow-marker + paint assertions, zoom/reset coherence);
existing PHI/analysis/Qt histogram suites kept green
(`test_preview_analysis.py`, `test_phi_preview_pipeline.py`,
`test_qt_histogram_otpux_h1.py`, `test_qt_histogram_otpux_drift.py`,
`test_qt_display_state_otpux.py`, `test_qt_otpux_final_ux.py`,
`test_qt_histogram_m14.py`, `test_qt_expert_m15.py`); `py_compile` clean;
`git diff --check` clean.

**Recorded limitation (accurate):** no real hardware/GPU run was performed —
all evidence is offscreen-Qt + deterministic fixtures, exactly as recorded in
the PHI closure (§12.5 and `docs/phi_live_preview_integrity_closure.md`);
this correction inherits that limitation.
