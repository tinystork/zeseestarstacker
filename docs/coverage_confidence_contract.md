# Coverage Confidence Contract — COV-01A (positive support core)

<!-- project: path:/home/tristan/.openclaw/workspace/projects/zeseestarstacker -->

## Scope / status

COV-01A is the first bounded subgate of COV-01.  It introduces and proves the
isolated, backend-neutral **positive support core** state/math contract only.
COV-00 (architecture archaeology) is ACCEPTED at f2d54b3.  This gate is NOT
wired into QueueManager, the classic/drizzle/reproject reducers, checkpoints,
GUI, render, or any existing scientific reducer — those are later COV-01
subgates after Junior ACCEPT.

## Contract

For each original exposure `i`, define a per-pixel positive support

```
s_i = valid_geometric_support
      * optional_quality_significance
      * optional_spatial_support_taper
```

and accumulate

```
SUP_W1 += s_i
SUP_W2 += s_i**2
```

with the derived effective support

```
N_eff_support = SUP_W1**2 / SUP_W2     where SUP_W2 > 0
                0.0                     otherwise (documented neutral value)
```

## Semantics (non-negotiable)

1. This is a **support/confidence** domain, NOT the scientific estimator
   denominator and NOT a rejection-survivor count.
2. No dependency on scientific WHT; no low-WHT gain; no science-pixel mutation.
3. No rejection-mask semantics: for rejection reducers the support confidence
   describes the *original geometric/quality* support and may deliberately
   differ from the surviving estimator WHT.  Rejection masks are never
   consumed here.
4. `SUP_W1` is a raw exposure count ONLY in the unit-weight case
   (`s_i ∈ {0,1}`).  The API never reports `SUP_W1` as a count.
5. Median science is unchanged and its support remains independent.

## Module / API

`seestar/core/coverage_support.py` — `PositiveSupportAccumulator`

* `PositiveSupportAccumulator(shape, *, dtype=float64)` — `shape` is exactly
  2-D `(H, W)`
* `.add(support)` — atomic per-original-exposure accumulation
* `.support_w1` / `.support_w2` — owned read copies of SUP_W1 / SUP_W2
* `.n_eff_support` — pure derived view (never mutates state; overflow-resistant)
* `.to_state()` / `.from_state(state)` — canonical persistence-ready snapshot /
  restore (arrays kept as ndarrays, not JSON-serialized)
* `SUPPORT_STATE_VERSION = 1`, `SUPPORT_DTYPES = (float32, float64)`

## Guarantees

* **Fail-before-mutation**: negative / NaN / Inf / shape-mismatch /
  non-finite-squared support is rejected before SUP_W1 or SUP_W2 changes.
* **Cumulative-overflow preflight**: if either candidate SUP_W1 or SUP_W2
  would become non-finite after accumulation, the `add` is rejected before
  either array is mutated (both stay byte/array identical).
* **Atomic pair mutation**: a failed `add()` or `from_state()` leaves both
  SUP_W1 and SUP_W2 byte/array unchanged.
* **Restore after full validation**: `from_state` validates type, version,
  shape, dtype, and both arrays (shape/dtype/finiteness/non-negativity) before
  constructing any visible object; restored arrays are owned copies (no
  aliasing to the caller's state).
* **No batch/merge API**: decomposition invariance across partition markers
  (61 vs 3+17+41 vs 1+…+1) is exact by construction for an identical ordered
  sequence (same float64 operation order).

## Dtype / memory decision

Accumulators default to `numpy.float64`.  `N_eff_support` is evaluated
exact-first (naive `SUP_W1**2 / SUP_W2` where the square stays finite) with an
overflow-resistant fallback `(SUP_W1 / sqrt(SUP_W2))**2` for pixels whose
square would overflow, so a finite SUP_W1 and SUP_W2 never silently degrade to
an undefined 0.0 via an intermediate square overflow.  float64 provides ample
precision headroom for the up-to-~100k-exposure target (a 100k unit-weight
witness is exact on this implementation).  float32 remains an opt-in,
memory-constrained option whose integer-exactness ceiling is 2**24 and which
must be used only under a documented tolerance.

Bytes-per-pixel for the SUP_W1+SUP_W2 pair: **16** (float64) / **8** (float32).
Support is channel-invariant and exactly 2-D `(H, W)` — one per-pixel map per
exposure regardless of channel count — so there is no hidden HWC duplication.

Microbenchmark (real, venv python 3.13.5 / numpy 2.5.2, this machine):

* 1080×1920 (2.1 MP): 21.64 ms/add, 16.0 bytes/px pair, 33.18 MB.
* 4096×4096 (16.8 MP): 165.53 ms/add, 16.0 bytes/px pair, 268.44 MB.
* 100k unit adds on 64×64: 4.317 s; SUP_W1 == 100000.0 exact;
  N_eff_support == 100000.0 exact.

The cumulative-overflow preflight allocates two full-frame temporaries per `add`
(the candidate SUP_W1/SUP_W2 sums), roughly doubling the add-path cost versus a
naive `+=` (measured ~9 → ~21.6 ms/add at 2.1 MP and ~92 → ~165 ms/add at
16.8 MP).  This is the bounded, documented cost of fail-before-mutation
atomicity and is confined to this isolated core.

## Tests

`tests/test_coverage_support.py` — 23 tests, all passing.  Covers: exact
W1/W2 known-weight witness; derived N_eff witness; zero/undefined support;
invalid negative/NaN/Inf/shape fail-before-mutation; overflow-squared
fail-before-mutation; shape (incl. 3-D rejection) / dtype validation;
restore-invalid-state fail-closed; snapshot/restore exactness + no aliasing;
decomposition partitions (61 vs 3+17+41 vs singletons); unit-weight reduces to
count; 100k float64 exact; float32 supported; cumulative-overflow regression
(float64 and float32); huge-but-valid N_eff not overflow; N_eff does not mutate
state.

## Limitations (COV-01A only)

* In-memory ndarray only — no memmap / disk backing yet.
* Not integrated with QueueManager, classic/drizzle/reproject, checkpoints,
  GUI, render, or any reducer.
* No scalar constant-support shortcut (support must be a full-shape 2-D array).
* The cumulative-overflow preflight allocates two bounded full-frame temporaries
  per `add` (documented in the benchmark below).
* No alternate merge API (and therefore no alternate-rounding tolerance to
  document); decomposition invariance is exact only for identical operation
  order.

## Deferred to COV-01B (explicitly not done here)

* Backend wiring into the per-batch reducer loop, before irreversible
  mini-stack reduction (COV-01 seam from the archaeology).
* Persistence / transaction ownership: classic memmap manifest + drizzle
  checkpoint fields, and registration into the failed-start cleanup allowlist
  (`_ATTEMPT_CREATED_CHECKPOINT_ARTIFACTS`).
* Final dtype policy for large outputs (float32 vs float64 under a documented
  tolerance) if memory pressure demands it.
* Reproject per-exposure support transform accumulation (`R(s_i)`, `R(s_i)**2`).
* Unit-weight fast path optimization (skip the square when `s_i ∈ {0,1}`).


## COV-01C — Drizzle positive-support domain

Drizzle accumulates a **distinct positive per-original-exposure support**,
independent of the native (possibly signed Lanczos) WHT:

* Native `out_img` / `out_wht` are **byte-identical** with and without support
  tracking (support is purely additive after the native deposition).
* Support uses a **square-kernel** `DrizzleAccumulator` pair
  (`drizzle_sup_w1` / `drizzle_sup_w2`), channel-invariant 2-D, depositing the
  validity footprint per frame: `s_i = weight` (q = 1.0, uniform),
  `SUP_W1 += s_i`, `SUP_W2 += s_i²`.
* `N_eff_support = SUP_W1² / SUP_W2` (overflow-resistant), never the native
  estimator N_eff and never the signed WHT.

Persistence belongs to the **native Drizzle checkpoint transaction**.  The
optional additive schema-v1 `support` field in `.m3d_checkpoint/checkpoint.json`
references generation-unique `support_w1` / `support_w2` float32 artifacts with
the same shape, size and SHA-256 descriptors as native channel artifacts.  The
writer snapshots and validates both positive arrays before writing anything,
writes native SCI/WHT and support artifacts under the same generation, then
publishes all of them through the existing `checkpoint.json` atomic replace —
the sole commit point.  A support write/validation failure removes only that
attempt's generation files and leaves the previous manifest and generation
authoritative and resumable.

The reader validates both support descriptors and artifacts before exposing any
reconstructed state, then rebuilds the two square-kernel accumulators with
`DrizzleAccumulator.from_native_state`.  Missing top-level `support` is the only
legacy signal: native SCI/WHT resumes unchanged with confidence unavailable.
Present-but-partial/corrupt/mixed-generation support fails closed.  Continuation
cannot drop committed support or fabricate support for a legacy run, and
SUP_W1/SUP_W2 cannot roll back.  No separate `drizzle_support` directory or
second commit point exists.

Tests: `tests/test_coverage_drizzle.py` (kernel parity incl. signed Lanczos,
decomposition determinism, N_eff, native transaction roundtrip, exact
Stop→Resume, legacy reopen, support preflight/write failure and orphan cleanup).

*Prepared for COV-01A/COV-01C and closed transactionally in COV-01C REWORK R2.
No production code changes outside the isolated core module, the
classic/drizzle backend wiring, and their focused tests + this contract doc.*


## COV-01D — Reproject positive-support domain

Both non-resumable Reproject routes now preserve the original-exposure
decomposition of positive support:

* **Reproject between batches:** each accepted exposure's boolean geometric
  mask is multiplied by its exact effective scalar quality weight, transformed
  independently onto the frozen reference grid, then committed in original
  exposure order as `SUP_W1 += R(s_i)` and `SUP_W2 += R(s_i)**2`.
* **Reproject + Coadd final:** because the final output grid may not exist while
  mini-stacks are produced, each batch writes a versioned, non-pickle NPZ
  sidecar containing the original ordered masks, scalar weights and celestial
  WCS headers.  After the scientific final grid and crop are known, these
  records are replayed per exposure directly onto that grid.  Temporary
  float64 memmaps are published as the canonical support pair only after the
  complete replay succeeds; a half-publication is rolled back.

In both routes the reprojection footprint gates support outside the real input
domain, output support is finite and non-negative, and squaring happens only
after each individual transform.  No batch aggregate, batch count, scientific
WHT, rejection-survivor mask, radial feather, or cosmetic gain enters the
support domain.  SCI/WHT reducers and their documented `R(V) * R(W)`
approximation remain unchanged.  Reproject Resume remains unsupported.

Focused proof: `tests/test_coverage_support_reproject.py` covers fractional-WCS
transforms, post-transform squaring, exact partition invariance, quality
scalars, fail-closed WCS/cardinality/sidecar handling, final-grid replay, and
rollback of an injected half-publication failure.

## COV-02 — Real footprint-aware support taper

The historical radial batch feathering (a radial falloff applied to the batch
denominator AFTER the numerator was divided) is replaced, for the
coverage-aware domain, by a real footprint-following support taper.

* `make_footprint_taper(mask, feather_px, floor)` computes a per-exposure
  `a_i(x, y) ∈ [0, 1]` from the Euclidean distance transform of the real
  transformed valid mask: `1.0` in the interior, ramping toward `floor` over
  `feather_px` pixels near the actual support boundary, and `0.0` outside.
  It is translation/rotation invariant and never a radial distance from the
  image centre.
* **Support-confidence domain** (Classic, and by construction the reproject /
  drizzle support paths already footprint-gated): the per-exposure support is
  `s_i = valid_mask_i * quality_i * a_i`, so `SUP_W1 += s_i` and
  `SUP_W2 += s_i²` now describe footprint-aware positive support.
* **Scientific use** (mean, the fractional-weighting estimator): the same taper
  acts consistently on numerator and denominator —
  `w_i^eff = q_i * m_i * a_i`, `SUM += x_i w_i^eff`, `WHT += w_i^eff` — so a
  constant field stays constant (no centre/edge photometric gain) and low
  coverage never brightens a pixel.  No `SCI *= taper` or `SCI *= 1/coverage`
  is performed after normalisation.
* Median and the rejection reducers keep their scientific estimators unchanged;
  their coverage maps are no longer radially distorted in the mean path.  The
  historical radial falloff remains gated by `apply_batch_feathering`
  (default on in production) and is the explicit COV-05 cleanup target for the
  remaining reducer paths.

Taper settings are `support_taper_px` (default `8.0` px) and
`support_taper_floor` (default `0.0`).  The taper is applied only when
`apply_batch_feathering` is enabled, so the exact `mask * quality` accumulation
contract and the existing HSI closure suites (which disable feathering) are
unchanged.

Focused proof: `tests/test_coverage_taper.py` covers interior/boundary/outside
behaviour, translation invariance, the radial-mask failure witness, validation
of `feather_px`/`floor`, the taper folded into the support domain, and mean
flat-field invariance.
