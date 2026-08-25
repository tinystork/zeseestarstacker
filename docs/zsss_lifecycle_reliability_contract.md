# ZSSS-LIFECYCLE-01 — Lifecycle Reliability Contract

Status: implementation + Stage-1 + 80-frame Stage-2 witnesses validated
Baseline: `8e51b95c32bbd0a51044c7edae9f30357b335061`
Branch: `feature/registration-field-rotation`

This document is the *contract* for the product-reliability boundary added in
ZSSS-LIFECYCLE-01.  It records the intended behaviour, not the implementation
report; acceptance is by independent review.

---

## 1. Structured startup refusal

**Stable code:** `OUTPUT_STATE_INCOMPATIBLE` (defined once on the engine carrier
`seestar.queuep.queue_manager.StartupRefusal` and mirrored on the GUI payload
`seestar.gui_qt.startup_refusal.CODE_OUTPUT_STATE_INCOMPATIBLE`).

**When it fires:** the output folder already holds resume/processing artifacts
(`_resume_requested` true) **and** the requested mode is non-plain (Drizzle,
mosaic, inter-batch or final reproject).  This is the existing, scientifically
correct read-only refusal, now carried structurally instead of as a progress
string.

**Carrier fields:** `code` (stable), `technical_detail` (free-form, never
parsed downstream), `semantic_key` (`output_state_incompatible`),
`semantic_data` (`{"mode": "drizzle"|"mosaic"|"reproject"}`).

**Contract:**

1. `start_processing` resets `self.startup_refusal = None` at the start of every
   attempt and sets it only on the known refusal above (via
   `_build_startup_refusal`).  All other early refusals stay generic.
2. The backend adapter raises `StartupRefusedError(payload)` (distinct from the
   generic `"start_processing() reported it did not start"` false start).
3. The Qt shell maps the stable code through the existing localization
   architecture (`localization.translate`) to EN/FR.  The wording is
   **mode-correct**, selected from `semantic_data.mode`:

   * Title (all modes) — EN: *"Output folder already contains processing data"*;
     FR: *"Le dossier de sortie contient déjà un traitement"*.
   * Drizzle body — EN: *"The selected output folder contains data from a
     previous processing session and cannot be reused for this Drizzle run.
     Please select a new output folder."*; FR: *"Le dossier de sortie
     sélectionné contient les données d'un traitement précédent et ne peut pas
     être réutilisé pour ce traitement Drizzle. Veuillez sélectionner un nouveau
     dossier de sortie."*
   * mosaic body — EN: *"...cannot be reused for this mosaic run..."*;
     FR: *"...ne peut pas être réutilisé pour ce traitement mosaïque..."*.
   * reproject body — EN: *"...cannot be reused for this reproject run..."*;
     FR: *"...ne peut pas être réutilisé pour ce traitement reprojection..."*.
   * Non-Drizzle modes never claim "Drizzle".

4. Refusal never deletes/overwrites/continues; artifacts are preserved.
5. Startup-side unwind: the engine stops the autotuner it started before the
   early refusal (the smallest safe correction — `stacker.stop()` returns early
   while `processing_active` is False), so a refused start leaks no service.

## 2. Truthful engine/backend completion

**Authoritative fact:** engine-thread liveness
(`stacker.processing_thread.is_alive()`), not the `processing_active` flag.

**Contract:**

1. `SeestarQueuedStackerBackend.run` never returns `FINISHED` while the engine
   thread is alive.  After the poll loop observes `is_running() == False`, the
   backend waits (bounded-slice join on the worker thread, never the GUI thread)
   for `processing_thread` to terminate, draining the deferred GUI event queue
   so terminal progress still flows.
2. Real engine/backend seams (exact, not inferred aliases):

   * `ENGINE_PROCESSING_RETURNING` — emitted by the engine at the final tail of
     `queue_manager._worker`, *after* required cleanup (autotuner stop, executor
     shutdown, memmap close, norm-reference release, gc) and immediately before
     the worker function exits.
   * `ENGINE_PROCESSING_RETURNED` — emitted by the backend *after* the engine
     thread is actually dead/joined (thread liveness authoritative).
   * `ENGINE_PROCESSING_FAILED` — emitted by the engine at its worker except
     blocks, carrying a bounded traceback, before the finally tail (existing
     stderr print and `processing_error` behaviour are preserved).
   * `BACKEND_RETURNING(status=…)` — emitted by the backend at its actual
     pre-return/pre-raise seam.  For the success and cancelled paths it is
     written immediately before `run` returns; for the failure path it is
     written *after* the `_stop_stackers()` cleanup call and immediately before
     the raise (never before potentially blocking cleanup work).
   * `BACKEND_RETURNED(status=…)` — emitted by `RunWorker` immediately after
     `backend.run` returns.
   * `BACKEND_RAISED(error=…, traceback=…)` — emitted by `RunWorker` when the
     backend raises, with a bounded `traceback.format_exc()`.
   * `COMPLETION_CALLBACK_EMITTING/EMITTED(kind=…)` — emitted by `RunWorker`
     around its own finished/cancelled/failed terminal signal (the summary
     callback is never called a completion callback).

3. After engine termination, a populated `processing_error` makes the backend
   report failure (raise), never success.
4. Cancellation stays non-blocking from the GUI and idempotent (`stop()`).

**Root-cause evidence (precise):** the defect this seam corrects is a *proven*
premature backend/Qt completion seam — the engine clears `processing_active`
before its tail cleanup and thread exit, so `is_running() == False` used to be
read as "done" while the engine thread was still alive.  Thread liveness is now
the authoritative fact.  A five-frame real M16 Stage-1 witness and a subsequent
80-frame M16 Stage-2 witness confirm end-to-end completion and the final log
sequence on this machine.

## 3. Truthful Qt completion

**Contract:**

1. The controller's public terminal signal (which makes MainWindow idle and
   re-enables Start) fires only after the owned `QThread` has actually finished.
2. Worker outcome is stored (`_pending_outcome`) when the worker signals arrive,
   and published from the GUI thread in `_on_thread_finished` (after reaping the
   QThread and recording `QTHREAD_FINISHED`).  Summary-before-success ordering
   is preserved.
   The worker hands the durable run log to the controller (`lifecycle_log`)
   **before** emitting its terminal outcome, so same-sender FIFO makes run-log
   availability precede the outcome *by construction* — the controller can never
   publish with a still-unset run log.
3. A missing terminal worker notification is an **explicit failure**, never
   false success: `_on_thread_finished` records `WORKER_OUTCOME_MISSING` and
   emits `failed(...)` with a technical detail.  A one-GUI-event-turn deferral
   (no wait/join) closes the cross-sender queued-signal race before treating a
   late outcome as missing.
4. At GUI terminal-handler return: `running` false, Stop disabled, Start enabled
   as appropriate, progress complete on success, result accessible, event loop
   responsive.
5. A second UI action/run cannot hit the `has_live_thread` race because the
   thread reference is cleared before the public signal fires.
6. `QT_COMPLETION_HANDLER_RETURNED` is **truthful and controller-owned**: it is
   written by the controller *immediately after* the public terminal signal emit
   returns (i.e. only once the MainWindow terminal slot has actually finished),
   and the run log is closed there.  MainWindow never claims RETURNED before its
   slot returns; the controller also finalizes the failure and cancellation
   paths.  A refused run never opened an accepted-run log, so it has no
   RETURNED and no close.

## 4. Persistent run log

**Naming:** `output_dir/zsss_run_<UTC-sortable-timestamp>_<short-session-id>.log`
— one file per accepted run, opened with exclusive create (`"x"`), never
overwriting an earlier log.  Timestamp is `YYYYMMDD-HHMMSS-ffffff` (UTC, sortable,
microsecond suffix for uniqueness) plus an 8-hex session id.

**Open policy:** the `RunLog` object may be created before `start_processing`
and may buffer a tiny number of pre-accept lifecycle events, but the *file* is
opened only immediately after `start_processing` returns `True`.  An
incompatible output folder is never touched before acceptance; a refused run
never creates/modifies a log.

**Required content:** each accepted run's `RUN_METADATA` records the product
version/codename (read from `seestar.__version__` / `seestar.__codename__` via a
cheap, import-hygienic source); `RUN_STARTED` records the input count when known
after acceptance (`stacker.files_in_queue`); `BACKEND_RETURNING` records the
terminal `processed_files_count`.  Allowlisted metadata only (mode / drizzle /
mosaic / batch size / input / output / reference / product version).  No image
arrays, no secrets, no unbounded config dumps.  A fatal engine error carries a
**bounded** `traceback.format_exc()` (truncated, single-line) in
`ENGINE_PROCESSING_FAILED` / `BACKEND_RAISED` / `WORKER_OUTCOME`.  A hard process
crash has no Python traceback to capture (documented limitation).

**Records:** plain-text, one `<ISO-8601> <EVENT> [k=v …]` line per event, with
stable event names and bounded fields.  The minimum event surface, in order, for
a successful run is:

`RUN_METADATA`, `RUN_ACCEPTED`, `RUN_STARTED`,
`DRIZZLE_FINALIZATION_ENTERED/RETURNED` (when applicable),
`FINAL_FITS_SAVE_ENTERED/RETURNED`,
`FINAL_PREVIEW_SAVE_ENTERED/RETURNED`,
`ENGINE_PROCESSING_RETURNING` (engine tail),
`ENGINE_PROCESSING_RETURNED` (after thread death),
`BACKEND_RETURNING` (backend, before return),
`BACKEND_RETURNED` (worker, after return),
`COMPLETION_CALLBACK_EMITTING/EMITTED` (around terminal signal),
`WORKER_OUTCOME`,
`QTHREAD_FINISHED`,
`QT_COMPLETION_HANDLER_ENTERED`,
`CONTROLS_RESTORED`, `GUI_IDLE`,
`RUN_SUCCEEDED` (only when `derive_terminal_status(...) == "success"`) or
`RUN_FINISHED_NO_OUTPUT` (clean finish with no output) or
`RUN_FAILED` / `RUN_CANCELLED`,
`QT_COMPLETION_HANDLER_RETURNED` (controller-side, after the terminal slot
returns; the log is closed immediately after).

**Truthful terminal status:** the GUI emits `RUN_SUCCEEDED` only when the
summary derives `success`; a clean backend finish with no output emits
`RUN_FINISHED_NO_OUTPUT`, never success.

**Flush/close policy:** every record is flushed per line (no `fsync`); a freeze
leaves the last completed event on disk.  The log is closed after the final
GUI/QThread lifecycle is recorded — concretely, by the controller immediately
after the public terminal signal emit returns (same seam as
`QT_COMPLETION_HANDLER_RETURNED`).  Open/write/close failures never fail science
and surface exactly one best-effort warning (no recursion); after a failed open,
`emit` drops records so the in-memory buffer never grows without bound.

**Progress mirroring (scope of interception):** `ENGINE_PROGRESS` mirrors the
authoritative progress-callback lines into both the GUI log and the persistent
run log (bounded message/percent/level).  Python `logging` output and legacy
`print` output are **not** globally intercepted — only the explicit entry/return
seams listed in §5 and the progress-callback adapter are recorded; a message
that reaches neither seam nor the progress callback is absent from the run log.

`registration_diagnostics.jsonl` remains scientifically separate; the run log
uses its own session id, never shares the registration-diagnostics session id,
and does not alter that schema.

## 5. Durable instrumentation

Only entry/return seams are instrumented (no "last log" aliases, no debug-print
spray): Drizzle finalization, final FITS save, final preview save (all in
`_save_final_stack`), engine worker returning, backend returning/returned,
worker completion callback, QThread finish, and the GUI terminal handler.

## 6. Frozen invariants (unchanged)

Homography/TPS/Similarity (Euclidean kept); reference-selection/immutable-target;
HSI/SUM-WHT/weighting/rejection/normalization; Drizzle SCI/WHT/pixmap/pre-warp;
resume expansion (plain classic only, Drizzle/reproject/mosaic fail closed);
no general logging rewrite, no Qt redesign, no Tk work, no version bump, no
dependency.

Scientific impact: none on HSI/registration/Drizzle/resume science and none on
ZeAlfie — all changes are lifecycle/observability boundaries around existing
science.

## 7. Stage-1 real M16 witness (2026-08-25)

Five real M16 FITS frames from `/home/tristan/M16/quick/` were exposed through
an isolated input directory and processed by the real PySide6 backend in
Drizzle Final, CPU mode.  The run completed in 105.592 seconds and produced
`stack_final_drizzle_final.fit` plus `stack_final_drizzle_final.png`.

The persistent log existed and already contained `RUN_STARTED` and
`ENGINE_PROGRESS` while the engine was active.  Its terminal sequence reached
`FINAL_FITS_SAVE_RETURNED success=True`,
`FINAL_PREVIEW_SAVE_RETURNED success=True`, `ENGINE_PROCESSING_RETURNED`,
`BACKEND_RETURNED`, `WORKER_OUTCOME`, `QTHREAD_FINISHED`,
`QT_COMPLETION_HANDLER_ENTERED`, `GUI_IDLE`, `RUN_SUCCEEDED`, and finally
`QT_COMPLETION_HANDLER_RETURNED` before close.

Final UI state: controller/thread idle, Start enabled, Stop disabled, progress
100%, result accessible.  A second tab-selection action was processed without
restarting ZSSS.  A 100 ms GUI-thread timer continued firing throughout the run;
the largest observed interval under CPU load was 1.645 seconds, with no
permanent event-loop stall.

Witness artifacts and the exact run log are retained under
`review/zsss_m16_stage1_20260825_UkLCqH/`.  This is the Stage-1 small witness;
the larger Stage-2 witness is recorded below.

## 8. Stage-2 real M16 witness (2026-08-25)

Tristan ran the real PySide6 application against 80 M16 FITS frames from
`/home/tristan/M16/long/`, using Drizzle Incremental with
`winsorized-sigma-clip`, and reported the run successful.  The accepted-run
lifecycle lasted 630.327 seconds and wrote its results under
`/home/tristan/M16/out/outdrizzle2/`.

Independent artifact inspection established:

* 80 `Aligned` and 80 `Moved to stacked` progress records, with no unaligned
  files;
* 80 valid RF2 registration records, all successful, all using the same
  immutable reference provenance and the Euclidean model;
* a valid `(3, 3840, 2160)` uint16 FITS with `NIMAGES=80`, 100% finite samples,
  and valid FITS `CHECKSUM` / `DATASUM`;
* a valid RGB `(2160, 3840)` PNG preview;
* no traceback, failure, cancellation, or genuine error/warning lifecycle
  record;
* the complete terminal chain through Drizzle/FITS/preview success, engine
  thread death, backend return, worker outcome, QThread finish, controls
  restoration, `GUI_IDLE`, `RUN_SUCCEEDED`, and finally
  `QT_COMPLETION_HANDLER_RETURNED`.

The user-observed success plus the durable terminal chain closes the larger-run
lifecycle gate.  The Stage-1 witness remains the explicit proof that a second
UI action is accepted after completion.  The unrelated pre-existing metadata
finding `TOTEXP=0.0` remains outside this lifecycle mission and is not treated
as lifecycle or scientific-output failure.
