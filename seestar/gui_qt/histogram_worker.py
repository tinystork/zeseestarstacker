"""Bounded latest-wins histogram worker/coordinator (ZSSS-OTPUX-HIST-H2).

Moves the authoritative Option-A ``compute_histogram_float`` off the Qt GUI
thread behind a *bounded latest-wins* lifecycle:

* at most **one** histogram computation runs at any time;
* at most **one** pending request is retained;
* a newer request **replaces** the older pending request (coalescing) — there is
  no unbounded QThread/QThreadPool queue under rapid preview/WB updates;
* every scheduled source/WB revision carries an **explicit monotonic generation
  token**;
* results are marshalled back to the GUI thread through a queued signal, where
  the generation token (and the caller's source check) decides apply-vs-discard;
* widget/model/status updates happen **only on the GUI thread**: the coordinator
  lives on the GUI thread and its slots run there.

Lifecycle (conceptual, matches ``docs/output_truthfulness_preview_audit.md``):

    source generation N
        -> request N (schedule, monotonic token N)
        -> worker computes N off the GUI thread
        -> worker emits result N (queued to the GUI thread)
        -> coordinator GUI-thread slot
        -> generation check (N == latest?) -> apply or discard.

If N+1 is requested before N completes, N's result is discarded and the pending
N+1 runs; N can never overwrite N+1.  ``invalidate()`` (reset/new context/
shutdown) bumps the generation so any in-flight or pending result is discarded
and cannot repopulate a cleared UI.

Input ownership: the worker only *reads* the analysis buffer handed to it.  The
owner (``MainWindow``) treats the buffer as immutable-by-convention and
*replaces* (never mutates) it on re-derivation, so no GUI-thread copy is made
solely for the worker; the float64 ndarray is shared by reference and read
without mutation (``compute_histogram_float`` is pure and read-only).

Instrumentation (H2/H4 seams, all GUI-thread readable):

* ``requests_scheduled`` — generations requested (``schedule`` calls),
* ``jobs_started`` — actual worker compute invocations dispatched,
* ``pending_replaced`` — pending requests coalesced/replaced,
* ``stale_discarded`` — results discarded because a newer generation exists,
* ``latest_applied`` — latest results applied,
* ``last_latency_ms`` / ``max_latency_ms`` / ``recent_latencies`` — latency.

No noisy production logging and no external dependencies.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

from .preview_analysis import compute_histogram_float

# Bounded recent-latency ring (used by the H4 latency witness).
_RECENT_LATENCY_LIMIT = 64


class HistogramWorker(QObject):
    """Runs one histogram computation off the GUI thread.

    The worker holds a single injected compute callable and exposes a single
    queued ``compute`` slot.  It is moved onto a dedicated :class:`QThread` and
    never touches widgets or mutates the analysis buffer.  Its event loop stays
    alive across requests (long-lived worker); shutdown is driven by the owner
    via ``QThread.quit()``.
    """

    # generation, result, source_token, latency_ms
    result_ready = Signal(int, object, object, float)

    def __init__(self, compute_fn: Callable[[Any], Any], parent: Optional[QObject] = None):
        super().__init__(parent)
        self._compute_fn = compute_fn

    @Slot(int, object, object)
    def compute(self, generation: int, buffer: Any, source_token: object) -> None:
        """Run the injected compute function (worker thread).

        A failing analysis (or a ``None``-returning one) is reported back as a
        ``None`` result; the coordinator's apply path treats ``None`` as
        fail-closed "no data" and never fabricates a histogram.
        """
        start = time.perf_counter()
        try:
            result = self._compute_fn(buffer)
        except Exception:
            result = None
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.result_ready.emit(generation, result, source_token, latency_ms)


class HistogramCoordinator(QObject):
    """GUI-thread owner of the bounded latest-wins histogram pipeline.

    Parameters
    ----------
    compute_fn:
        The analysis callable executed on the worker thread.  Defaults to
        :func:`~seestar.gui_qt.preview_analysis.compute_histogram_float`.
        Tests inject a slow/controlled callable here — no monkeypatching of
        global scientific state is required.
    """

    # generation, result, source_token — emitted only for the latest result.
    result_ready = Signal(int, object, object)
    # generation, buffer, source_token — queued dispatch to the worker slot.
    _request = Signal(int, object, object)

    def __init__(
        self,
        compute_fn: Optional[Callable[[Any], Any]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._compute_fn = compute_fn if compute_fn is not None else compute_histogram_float
        self._thread: Optional[QThread] = None
        self._worker: Optional[HistogramWorker] = None
        # Monotonic generation token.  Bumped on every ``schedule`` and on every
        # ``invalidate``, so it always equals the identity of the newest event.
        self._generation: int = 0
        # ``(generation, buffer, source_token)`` currently dispatched, or None.
        self._running: Optional[tuple] = None
        # ``(generation, buffer, source_token)`` latest pending request, or None.
        self._pending: Optional[tuple] = None
        self._closed: bool = False
        self._result_disconnected: bool = False

        # Instrumentation counters (GUI-thread readable).
        self.requests_scheduled: int = 0
        self.jobs_started: int = 0
        self.pending_replaced: int = 0
        self.stale_discarded: int = 0
        self.latest_applied: int = 0
        self.last_latency_ms: float = 0.0
        self.max_latency_ms: float = 0.0
        self.recent_latencies: deque = deque(maxlen=_RECENT_LATENCY_LIMIT)

    # ------------------------------------------------------------ state (tests)
    @property
    def generation(self) -> int:
        """The current monotonic generation token (newest event identity)."""
        return self._generation

    @property
    def running_generation(self) -> Optional[int]:
        """Generation currently being computed, or ``None``."""
        return self._running[0] if self._running is not None else None

    @property
    def pending_generation(self) -> Optional[int]:
        """Generation of the latest pending request, or ``None``."""
        return self._pending[0] if self._pending is not None else None

    @property
    def is_running(self) -> bool:
        """True while a computation is in flight (dispatched)."""
        return self._running is not None

    @property
    def is_pending(self) -> bool:
        """True while a pending request is retained."""
        return self._pending is not None

    @property
    def has_live_thread(self) -> bool:
        """True while the owned QThread is still running."""
        thread = self._thread
        return thread is not None and thread.isRunning()

    # ------------------------------------------------------------- scheduling
    def schedule(self, buffer: Any, source_token: object = None) -> int:
        """Schedule a histogram computation for ``buffer`` (GUI thread).

        Assigns the next monotonic generation token and returns it.  When the
        worker is idle the request is dispatched immediately; otherwise it
        becomes the (only) pending request, replacing any older pending request.
        """
        if self._closed:
            return self._generation
        self._generation += 1
        generation = self._generation
        self.requests_scheduled += 1
        self._ensure_started()
        if self._running is None:
            self._running = (generation, buffer, source_token)
            self._dispatch(generation, buffer, source_token)
        else:
            if self._pending is not None:
                self.pending_replaced += 1
            self._pending = (generation, buffer, source_token)
        return generation

    def invalidate(self) -> None:
        """Discard any in-flight/pending result (reset / new context / shutdown).

        Bumps the generation token so every previously scheduled result becomes
        stale, and drops the retained pending request.  The running computation
        (if any) finishes in place; its result is discarded on arrival.
        """
        self._generation += 1
        self._pending = None

    # ------------------------------------------------------------ result slot
    @Slot(int, object, object, float)
    def _on_result(self, generation: int, result: object, source_token: object, latency_ms: float) -> None:
        """GUI-thread slot: apply the latest result, discard a stale one.

        A result is applied only when its generation is still the newest event
        (``generation == self._generation``).  Otherwise (a newer request or an
        ``invalidate`` intervened) it is discarded, and the latest pending
        request — if any — is dispatched next.  A result is never allowed to
        overwrite a newer one.
        """
        self.last_latency_ms = float(latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, float(latency_ms))
        self.recent_latencies.append(float(latency_ms))

        # Ignore a spurious/duplicate result for a generation we no longer track.
        if self._running is None or self._running[0] != generation:
            return
        self._running = None

        if generation == self._generation:
            self.latest_applied += 1
            self.result_ready.emit(generation, result, source_token)
        else:
            self.stale_discarded += 1

        # Start the latest pending request (it is the newest scheduled one).
        if self._pending is not None:
            pending_generation, pending_buffer, pending_token = self._pending
            self._pending = None
            self._running = (pending_generation, pending_buffer, pending_token)
            self._dispatch(pending_generation, pending_buffer, pending_token)

    # ---------------------------------------------------------------- worker
    def _ensure_started(self) -> None:
        """Lazily create + start the dedicated worker QThread (idempotent)."""
        if self._thread is not None:
            return
        thread = QThread(self)
        worker = HistogramWorker(self._compute_fn)
        worker.moveToThread(thread)
        # Explicit queued connections: dispatch runs on the worker thread, the
        # result slot on the coordinator's (GUI) thread.
        self._request.connect(worker.compute, Qt.ConnectionType.QueuedConnection)
        worker.result_ready.connect(self._on_result, Qt.ConnectionType.QueuedConnection)
        # Delete the worker object when the thread's event loop finishes (safe:
        # the DeferredDelete is processed inside QThread::run() before ``wait()``
        # returns, so the worker is never destroyed while its thread runs).  The
        # QThread itself is *not* auto-deleted here: ``shutdown()`` retains
        # ``self._thread`` across a timed-out wait for a later retry, and an
        # auto-deleted QThread would leave a dangling wrapper whose
        # ``isRunning()``/``wait()`` raise RuntimeError.  Explicit QThread
        # cleanup happens in ``shutdown()`` after a successful join.
        thread.finished.connect(worker.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _dispatch(self, generation: int, buffer: Any, source_token: object) -> None:
        """Dispatch one computation to the worker thread (queued)."""
        self.jobs_started += 1
        self._request.emit(generation, buffer, source_token)

    # --------------------------------------------------------------- shutdown
    def shutdown(self, wait_ms: int = 5000) -> bool:
        """Idempotent teardown: invalidate + stop and join the worker thread.

        Returns ``True`` once no owned QThread is still running (fully stopped).
        Returns ``False`` when the worker did not finish within ``wait_ms``
        (e.g. a controlled slow compute still in flight); the thread/worker
        references are retained so a later ``shutdown`` call can retry, and no
        running thread is ever destroyed.
        """
        self._closed = True
        self.invalidate()
        thread = self._thread
        worker = self._worker
        if thread is None:
            return True
        # Disconnect the result channel so no post-shutdown delivery can run
        # (idempotent: shutdown may be retried after an incomplete wait).
        if not self._result_disconnected:
            try:
                worker.result_ready.disconnect(self._on_result)
            except (RuntimeError, TypeError):
                pass
            self._result_disconnected = True
        thread.quit()
        if not thread.wait(wait_ms):
            # Do NOT drop references: dropping them would destroy a running
            # thread.  The thread/worker remain for a later retry.
            return False
        # Explicit cleanup after a successful join.  The worker has already been
        # deleted by ``thread.finished -> worker.deleteLater`` (processed inside
        # QThread::run() before ``wait()`` returned), so drop both references and
        # clear the running/pending state.  No post-shutdown result delivery can
        # occur: the result channel was disconnected above (possibly on an
        # earlier timed-out attempt), and ``_running`` is cleared here because
        # ``_on_result`` can no longer run to clear it.
        self._worker = None
        self._thread = None
        self._running = None
        self._pending = None
        # Delete the QThread C++ object on the GUI thread now that nothing holds
        # a reference to it (explicit, leak-free ownership teardown).
        thread.deleteLater()
        return True
