"""Persistent per-run lifecycle log (pure stdlib, thread-safe, fail-open).

ZSSS-LIFECYCLE-01 durable-observability carrier.  One ``RunLog`` instance is
shared (by reference) across the backend, the worker, the controller and the
main window so every lifecycle event of one accepted run lands in a single
durable file:

    <output_dir>/zsss_run_<UTC-sortable-timestamp>_<short-session-id>.log

Design constraints honoured here:

* **Pure stdlib.**  No Qt, no Tk, no engine, no astropy.  Importable and
  unit-testable in complete isolation.
* **Thread-safe.**  A single ``threading.Lock`` guards the buffer/file so the
  engine thread, worker thread and GUI thread can all emit without races.
* **Fail-open.**  ``open`` / ``emit`` / ``close`` never raise: a failure is
  captured as a one-shot warning (surfaced through an optional ``warning``
  handler, never by recursing into ``emit``) and science is never affected.
* **Buffered before acceptance.**  The object may be created and receive
  ``emit`` calls before ``open`` (a tiny number of pre-accept engine lifecycle
  events); those events are buffered and flushed into the file by ``open``.
  No file is created before ``open`` is called, so an incompatible output
  folder is never touched before the run is accepted.
* **Never overwrites.**  Each ``open`` writes to a fresh, timestamped,
  session-suffixed filename.  ``open`` is single-shot per instance.

The file is plain text, one ``<ISO-8601> <EVENT> [k=v ...]`` record per line.
Every record is flushed per line (explicit flush, no ``fsync``) so a freeze
leaves the last completed event on disk.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

# Sortable UTC filename component (no colons, filesystem-safe).
_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"

# Bound the size of a single emitted record so a runaway engine message can
# never bloat the log or leak a huge array/secret dump.
_MAX_FIELD_LEN = 2000
_MAX_LINE_LEN = 8000


def _iso_timestamp(seconds: float) -> str:
    """Return a sortable UTC ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds)) + (
        f".{int((seconds % 1.0) * 1_000_000):06d}Z"
    )


def _safe_value(value: Any) -> str:
    """Render one field value as a bounded, single-line token.

    Strings are kept verbatim (bounded); everything else uses ``repr``.  The
    result is bounded and newline-normalised so a record always stays a single
    line and can never grow without bound.
    """
    if isinstance(value, str):
        text = value
    else:
        try:
            text = repr(value)
        except Exception:
            text = "<unrepresentable>"
    if len(text) > _MAX_FIELD_LEN:
        text = text[:_MAX_FIELD_LEN] + "…"
    return text.replace("\r", " ").replace("\n", " ")


def _default_session_id() -> str:
    """Return an 8-hex short session id (collision-resistant enough per run)."""
    import hashlib

    seed = f"{time.time_ns()}-{os.getpid()}-{id(object())}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]


class RunLog:
    """Thread-safe, fail-open, buffered per-run lifecycle log.

    Parameters
    ----------
    session_id:
        Optional short session id used in the filename.  A fresh one is
        generated when omitted.
    clock:
        Optional monotonic/absolute clock (tests inject a fake); defaults to
        :func:`time.time` (UTC wall clock used for both the filename and the
        per-record timestamp).
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.session_id = session_id or _default_session_id()
        self._clock = clock if clock is not None else time.time
        self._lock = threading.Lock()
        self._path: Optional[str] = None
        self._file = None
        self._buffer: List[str] = []
        self._opened = False
        self._closed = False
        self._open_error: Optional[str] = None
        self._warning_emitted = False
        # Best-effort warning sink (never recurses into ``emit``).
        self.warning: Optional[Callable[[str], None]] = None

    # ------------------------------------------------------------------ state
    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def is_open(self) -> bool:
        return self._file is not None and not self._closed

    @property
    def open_error(self) -> Optional[str]:
        return self._open_error

    @property
    def buffered_count(self) -> int:
        """Number of buffered (pre-open) records currently held."""
        with self._lock:
            return len(self._buffer)

    # ------------------------------------------------------------- internals
    def _warn(self, message: str) -> None:
        """Surface a one-shot best-effort warning (never raises, never recurses)."""
        if self._warning_emitted:
            return
        self._warning_emitted = True
        handler = self.warning
        if handler is None:
            return
        try:
            handler(message)
        except Exception:
            pass

    def _format_record(self, event: str, fields: Dict[str, Any]) -> str:
        parts = [_iso_timestamp(self._clock()), event]
        for key, value in fields.items():
            if not isinstance(key, str) or not key:
                continue
            parts.append(f"{key}={_safe_value(value)}")
        line = " ".join(parts)
        if len(line) > _MAX_LINE_LEN:
            line = line[:_MAX_LINE_LEN] + "…"
        return line

    # ---------------------------------------------------------------- public
    def open(
        self,
        output_dir: Optional[str],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create the log file in ``output_dir`` and flush any buffered records.

        Single-shot: the first successful ``open`` wins; later calls are a
        no-op.  On failure the instance stays buffered-but-never-written and
        records a one-shot warning; subsequent ``emit`` calls are dropped so a
        failure to open cannot grow an unbounded in-memory buffer.
        """
        with self._lock:
            if self._opened or self._closed:
                return self._path
            self._opened = True
            if not output_dir:
                self._open_error = "no output directory"
                self._warn(f"run log unavailable: {self._open_error}")
                return None
            try:
                # The engine creates the output folder on acceptance; we never
                # create it here (never touch an incompatible/uncreated folder).
                if not os.path.isdir(output_dir):
                    self._open_error = "output directory does not exist"
                    self._warn(f"run log unavailable: {self._open_error}")
                    return None
            except OSError:
                self._open_error = "output directory is not accessible"
                self._warn(f"run log unavailable: {self._open_error}")
                return None
            timestamp = time.strftime(_TIMESTAMP_FORMAT, time.gmtime(self._clock()))
            micro = int((self._clock() % 1.0) * 1_000_000)
            filename = (
                f"zsss_run_{timestamp}-{micro:06d}_{self.session_id}.log"
            )
            path = os.path.join(output_dir, filename)
            try:
                # "x" guarantees we never overwrite an earlier run log.
                self._file = open(path, "x", encoding="utf-8")
            except OSError as exc:
                self._open_error = f"cannot create run log: {exc}"
                self._warn(f"run log unavailable: {self._open_error}")
                return None
            self._path = path
            # Flush any pre-accept lifecycle events buffered before open().
            buffered = self._buffer
            self._buffer = []
            for record in buffered:
                self._write_locked(record)
            if metadata:
                self._write_locked(
                    self._format_record("RUN_METADATA", dict(metadata))
                )
        return self._path

    def _write_locked(self, record: str) -> None:
        if self._file is None:
            return
        try:
            self._file.write(record + "\n")
            self._file.flush()
        except Exception as exc:
            self._warn(f"run log write failed: {exc}")

    def emit(self, event: str, **fields: Any) -> None:
        """Record one lifecycle event (thread-safe, never raises).

        Before ``open`` the record is buffered; after a failed ``open`` it is
        dropped; after ``close`` it is a silent no-op.
        """
        with self._lock:
            if self._closed:
                return
            record = self._format_record(str(event), fields)
            if not self._opened:
                self._buffer.append(record)
                return
            if self._file is None:
                return
            self._write_locked(record)

    def close(self) -> None:
        """Flush and close the log file (idempotent, never raises)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._file is not None:
                try:
                    self._file.flush()
                except Exception as exc:
                    self._warn(f"run log flush failed: {exc}")
                try:
                    self._file.close()
                except Exception as exc:
                    self._warn(f"run log close failed: {exc}")
                self._file = None
            self._buffer = []
