# Resume Contract v2 — product contract (authoritative)

Status: authoritative description, not a development diary.

## Fresh vs Resume intent

Resume occurs **only** on an explicit current user action. A persisted Last Stack
path is a run *locator*, never implicit Resume intent. Fresh and Resume are
explicit, distinct operations.

- `resume_intent=fresh` → refuses if the output folder already holds recognized
  run/resume state (never overwrites).
- `resume_intent=resume` → requires a recognized checkpoint; the backend is
  authoritative and fails closed on any invalid state.

## Checkpoint kinds

- **Classic SUM/W** — `memmap_accumulators/` + `resume_manifest.json`
  (cumulative_SUM.npy / cumulative_WHT.npy). Resume restores the SUM/WHT memmaps.
- **Drizzle native** — `.m3d_checkpoint/` (checkpoint.json + generation
  `gen-NNNNNNNN-chC-out_{img,wht}.npy`). Standard and Incremental, non-mosaic, non-reproject.
  Resume re-reads disk and restores the three native SCI/WHT accumulators, then
  re-arms the writer at generation N+1.

## run_config.cfg

Schema-v2, UTF-8, atomically written (temp + `os.replace`). A versioned recipe
describing the run: `scientific_config` (fingerprint inputs), `execution_config`
(run context: input/output folders, output filename, …), `provenance`. It is
configuration evidence only — never a scientific checkpoint.

Secrets never serialize: the writer recursively rejects secret-like keys
(`api_key`, `api_token`, `password`, `secret`, `credential`, …) before any
write; a failed validation leaves no partial file.

### Canonical config freshness (fresh-start lifecycle)

A fresh Classic run writes the manifest twice before the worker launches: a
bootstrap write inside `_initialize_classic_sumw_accumulators()` (as soon as
SUM/WHT are allocated) and the authoritative plan-binding write inside
`_checkpoint_preflight()` (after queue planning).  Queue planning can
legitimately re-bind the runtime-effective `batch_size` fingerprint field
(stack-plan `use_plan` → `0`, single-batch CSV → `999999999`, prepared-queue
single-batch adaptation → `1`) between those two writes.  The canonical
`run_config.cfg`/`scientific_config` persisted by *every* manifest write is
collected from the engine state at that write: the instance cache is keyed by
the engine's authoritative classic scientific fingerprint, so an unchanged
effective contract reuses one identical model (identical digest on every
write) while a legitimate runtime-effective transition recollects the model
before persisting.  Genuine canonical-vs-engine divergences are still refused
by the fail-closed fingerprint equality check before any write.

## Scientific fingerprint

Derived from the canonical *scientific* subset only. GUI-only/presentation
preferences are never fingerprinted. Execution context and Drizzle grouping policy
(drizzle_mode, drizzle_group_size) are **not** fingerprinted — they do not alter
deposited science under the M3 single-accumulator architecture.

## Manifest v1 compatibility (legacy)

A Classic `resume_manifest.json` schema-v1 stores only the scientific fingerprint,
never a `run_config.cfg`. On explicit Resume:

- v1 + no CFG + current fingerprint matches → Resume (current effective settings).
- v1 + no CFG + fingerprint mismatch → clean refusal.
- The fingerprint is never bypassed, reversed, or weakened.

## Manifest/config v2

`run_config.cfg` present → restored into Qt state (user sees the configuration
being resumed); backend re-validates independently (UI restore is not the safety
mechanism). Field-level mismatch diagnostics are bounded; a v1 without config
payload may only report a fingerprint mismatch.

## Standard vs Incremental (Drizzle policy)

Standard and Incremental/Large-dataset share one native Drizzle accumulator and one
checkpoint implementation. `drizzle_mode` and `drizzle_group_size` are persisted
and restored as run policy: Standard → Standard, Incremental → Incremental, with the
original group size. They are grouping/preview/resource policy, not science.

## clean/dirty and lifecycle

- Classic manifest: `dirty` before accumulator mutation, `clean` after a batch
  commit. Resume refuses a `dirty` checkpoint (fail-closed, no silent skip).
- A failed **first** Drizzle checkpoint attempt is cleaned up: an empty/stale
  `.m3d_checkpoint` created by the failed attempt is removed; a previously
  committed valid checkpoint is never deleted.
- Resume source and output folder must resolve to the same run (normalized,
  platform-safe comparison); a mismatch is refused.
- Repeated Start after an early refusal is safe (no package-local mmap/file
  leak, no stale processing_active/resume/stop flag).

## Secret exclusion

Writer-side recursive scan before serialization; reader-side recursive scan on
load. Secret values are never echoed in exceptions or logs.

## Unsupported (fail-closed)

- Mosaic resume — unsupported.
- Reprojection resume — unsupported.
- Arbitrary FIT-as-checkpoint recovery — unsupported.
