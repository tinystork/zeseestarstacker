# Coverage Support — Classic Backend Wiring (COV-01B-CLASSIC)

<!-- project: path:/home/tristan/.openclaw/workspace/projects/zeseestarstacker -->

## Scope

Wires the accepted COV-01A positive support domain through the PLAIN CLASSIC /
classic_sumw backend only.  No Drizzle/reproject/mosaic/COV-02..05 changes.
Scientific SCI/WHT and every reducer are unchanged.

## Contract

Per accepted ORIGINAL exposure i, before irreversible mini-stack reduction:

```
s_i = valid_geometric_support_i * effective positive per-exposure quality significance_i
SUP_W1 += s_i
SUP_W2 += s_i**2
```

Spatial taper is 1.0 in this gate (COV-02 owns taper).  Uniform weighting gives
mask support; quality/variance/FWHM weighting uses the effective finite
non-negative scalar actually consumed by the reducer for that original source
(the per-image `quality_weights[i]`, already folded with the variance/FWHM
`extra_w` factor).  Rejection-survivor masks and final/batch WHT are never
consumed.

## State model

* Two channel-invariant 2-D float64 memmaps under `memmap_accumulators`:
  `coverage_SUP_W1.npy` and `coverage_SUP_W2.npy` (16 bytes/pixel pair).
* Manifest carries a `support` metadata object (schema `sup_v1`, dtype, shape).
* `_support_state_available` is monotonic per run: True for fresh classic and
  valid-support resume; False for a legacy support-less checkpoint (science
  resumes unchanged; cosmetic confidence render is disabled, never fabricated).
* Support artifacts join `_ATTEMPT_CREATED_CHECKPOINT_ARTIFACTS` and are
  removed only by ownership-safe failed-start cleanup.

## Transactional handoff (decomposition-exact)

`_stack_batch` returns an ORDERED payload of (boolean mask reference, validated
float scalar) tuples for the accepted original exposures — never a pre-summed
batch delta, never an instance-global staged value.  `_combine_batch_result`
applies it after dirty marking, ONE exposure at a time in original order via
the shared atomic seam `accumulate_support_pair` (COV-01A math, fail-before-
mutation on shape/dtype/finiteness/square/cumulative overflow).  This makes
61 vs 3+17+41 vs singletons ARRAY-EXACT.

A rejected/failed batch never reaches the commit, so support is unchanged.
A support mutation failure after dirty raises `_ResumeCheckpointError`, which
sets `processing_error` + `stop_processing` and leaves the checkpoint dirty.

## Resume / legacy / cleanup

* Fresh: `_create_support_memmaps` (refuses to recreate/zero existing support).
* Resume with support: read-only validation (schema/dtype/shape/finite/non-neg)
  before opening `r+`; restore exact support state and continue once.
* Legacy without support: `_support_state_available = False`, logged.
* Malformed/present support metadata or missing artifact: fail closed before
  any `r+` open.
* `_close_memmaps` closes both support handles.

## Tests

`tests/test_coverage_support.py` (COV-01A core) +
`tests/test_coverage_support_classic.py` (dtype/state adversarial, payload
validation, boolean-mask/shape validation, exact SUP_W1/SUP_W2 witness, and
ARRAY-EXACT 61 vs 3+17+41 vs singletons decomposition).

## Deferred (not this gate)

* Drizzle / reproject / mosaic support accumulation.
* COV-02 spatial taper, COV-03 normalization, COV-04 final render, COV-05
  cleanup-after-proof.
* Per-mode support metadata for non-classic finalization.

