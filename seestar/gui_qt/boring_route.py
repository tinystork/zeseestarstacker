"""Qt-side boring (single-batch CSV) route helpers — pure stdlib.

This module restores the Qt shell's *essential parity* with the historical Tk
``batch_size == 1`` boring-stack path without importing the Tk GUI or the
scientific engine at ``import seestar.gui_qt`` time.

It owns two pure, testable responsibilities:

* :func:`parse_stack_plan_csv` — a conservative replication of the Tk
  ``boring_stack.read_rows`` / ``_prepare_single_batch_if_needed`` parsing
  rules (``file_path`` header, ``order,file``/``index,file`` fallback, optional
  leading numeric index column, relative paths resolved against the CSV's
  directory, missing-file detection).
* :func:`build_boring_request` — the ``boring_stack.py`` subprocess command
  builder, resolved by *filesystem path* (never imported), so a fresh
  ``import seestar.gui_qt`` leaves ``sys.modules`` free of the Tk/engine
  surface.

Import-hygiene: this module is stdlib-only (``csv``/``os``/``sys``/``dataclasses``/
``typing``).  The one heavy dependency (``psutil``) is imported lazily inside
:func:`get_auto_chunk_size` and is not part of the boring/Tk/engine surface.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# Column-name tokens recognised by the historical CSV reader (``read_rows``).
# A header row whose cells include any of these is treated as a header (and
# skipped), and a *data* cell equal to one of these is treated as a stray
# repeated header token and ignored.
_HEADER_TOKENS = {
    "order",
    "file",
    "filename",
    "path",
    "index",
    "weight",
    "file_path",
}

# CSV filename expected inside the input folder (Tk parity).
STACK_PLAN_FILENAME = "stack_plan.csv"


class BoringCsvError(Exception):
    """Raised when ``stack_plan.csv`` is missing, empty or lists missing files.

    The message is human-readable and safe to surface directly in the UI log /
    status bar.  ``path`` carries the CSV path (when known) for callers that
    want to echo it back.
    """

    def __init__(self, message: str, path: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.path = path


@dataclass(frozen=True)
class BoringCsvParse:
    """Parsed ``stack_plan.csv`` contents (ordered, existing file paths).

    ``weights`` is aligned 1:1 with ``ordered_files`` (``""`` when the CSV has
    no weight column); it is retained for parity but is *not* consumed by the
    Qt command builder — ``boring_stack.py`` re-reads the CSV itself.
    """

    ordered_files: List[str]
    weights: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class BoringRunRequest:
    """Immutable snapshot of a boring-stack subprocess launch.

    ``command`` is the full argv handed to the subprocess runner.  The scalar
    fields below are the structured inputs used to build it, kept as named
    fields so tests can assert individual arguments without string-matching the
    whole list.
    """

    command: List[str]
    script_path: str
    csv_path: str
    output_dir: str
    batch_size: int = 1
    chunk_size: int = 50
    normalize_method: str = "none"
    save_final_as_float32: bool = False
    final_combine: str = "mean"
    max_mem_gb: float = 8.0
    request_gpu: bool = False


def csv_path_for(input_folder: str) -> str:
    """Return the ``stack_plan.csv`` path for an input folder (Tk parity)."""
    return os.path.join(input_folder, STACK_PLAN_FILENAME)


def parse_stack_plan_csv(csv_path: str) -> BoringCsvParse:
    """Parse ``stack_plan.csv`` following the historical Tk reading rules.

    Returns a :class:`BoringCsvParse` with the ordered, *existing* absolute
    file paths.  Raises :class:`BoringCsvError` when the CSV is missing, empty
    (no valid data rows), unreadable, or lists at least one file that does not
    exist — mirroring the Tk ``_prepare_single_batch_if_needed`` contract that
    aborts the single-batch route instead of silently proceeding.

    Rules replicated from ``boring_stack.read_rows``:

    * a ``file_path`` column (case-insensitive) is authoritative, with an
      optional ``weight`` column;
    * otherwise, if the first row contains any header token (``order``/
      ``file``/``filename``/``path``/``index``/``weight``) it is a header row;
    * a leading numeric cell is treated as an ignored index column
      (``1,file.fits`` → ``file.fits``);
    * blank rows and stray repeated header-token cells are skipped;
    * relative paths are resolved against the CSV's own directory;
    * every resolved path must exist as a file, else the parse fails.
    """
    if not os.path.isfile(csv_path):
        raise BoringCsvError(
            f"stack_plan.csv not found: {csv_path}", path=csv_path
        )

    try:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        raise BoringCsvError(
            f"cannot read stack_plan.csv: {exc}", path=csv_path
        ) from exc

    if not rows:
        raise BoringCsvError("stack_plan.csv is empty", path=csv_path)

    header = [cell.strip().lower() for cell in rows[0]]
    file_idx: Optional[int] = None
    weight_idx: Optional[int] = None
    data_rows = rows

    if "file_path" in header:
        file_idx = header.index("file_path")
        weight_idx = header.index("weight") if "weight" in header else None
        data_rows = rows[1:]
    else:
        has_header = any(token in _HEADER_TOKENS for token in header)
        if has_header:
            data_rows = rows[1:]
            if "weight" in header:
                weight_idx = header.index("weight")

    base_dir = os.path.dirname(os.path.abspath(csv_path))
    ordered_files: List[str] = []
    weights: List[str] = []

    for row in data_rows:
        if not row:
            continue
        if file_idx is not None:
            if len(row) <= file_idx:
                continue
            cell = row[file_idx].strip()
        else:
            cell = row[0].strip()
            # Optional leading index column: ``1,file.fits`` -> ``file.fits``.
            if cell.isdigit() and len(row) > 1:
                cell = row[1].strip()

        if not cell or cell.lower() in _HEADER_TOKENS:
            continue

        if not os.path.isabs(cell):
            cell = os.path.join(base_dir, cell)
        abs_cell = os.path.abspath(cell)

        weight = ""
        if weight_idx is not None and len(row) > weight_idx:
            weight = row[weight_idx].strip()

        ordered_files.append(abs_cell)
        weights.append(weight)

    if not ordered_files:
        raise BoringCsvError(
            "stack_plan.csv contains no valid file paths", path=csv_path
        )

    missing = [path for path in ordered_files if not os.path.isfile(path)]
    if missing:
        raise BoringCsvError(
            f"File listed in stack_plan.csv not found: {missing[0]}",
            path=csv_path,
        )

    return BoringCsvParse(ordered_files=ordered_files, weights=weights)


def resolve_boring_script_path() -> str:
    """Return the filesystem path to ``seestar/gui/boring_stack.py``.

    Computed from ``__file__`` (this module lives in ``seestar/gui_qt/``, a
    sibling of ``seestar/gui/``) so the script is reached *without* importing
    it — preserving the import-hygiene guarantee that ``import seestar.gui_qt``
    never pulls in the Tk GUI or the engine.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, os.pardir, "gui", "boring_stack.py")
    )


def get_auto_chunk_size() -> int:
    """Return an automatic chunk size based on system RAM (Tk parity).

    Lazily imports ``psutil`` (never at module import time) and reproduces the
    Tk ``_get_auto_chunk_size`` thresholds exactly.  Falls back to ``50`` when
    ``psutil`` is unavailable or fails.
    """
    try:
        import psutil  # noqa: PLC0415 - lazy import keeps this module engine-free
    except Exception:
        return 50
    try:
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 50
    if total_gb >= 64:
        return 100
    if total_gb >= 32:
        return 50
    if total_gb >= 16:
        return 25
    return 10


def build_boring_request(
    *,
    csv_path: str,
    output_dir: str,
    batch_size: int = 1,
    chunk_size: int = 50,
    normalize_method: str = "none",
    save_final_as_float32: bool = False,
    final_combine: str = "mean",
    max_mem_gb: float = 8.0,
    request_gpu: bool = False,
    python_executable: Optional[str] = None,
) -> BoringRunRequest:
    """Build the immutable boring-stack subprocess command.

    The command mirrors the historical Tk ``start_processing`` boring branch:

    * ``--csv`` / ``--out`` / ``--batch-size 1`` / ``--chunk-size`` (auto
      chunk logic) / ``--log-dir`` (``<output>/logs``) / ``--norm``
      (normalization) / ``--save-as-float32`` or ``--no-save-as-float32`` /
      ``--final-combine`` (final-combine key) / ``--max-mem``.
    * GPU intent (F7): only the BOOLEAN user intent crosses the subprocess
      boundary as ``--gpu`` / ``--no-gpu``; the subprocess resolves its OWN
      probe/policy (``SeestarQueuedStacker(gpu=...)`` →
      ``acceleration_policy``), so no hardware assumption is passed from Qt.

    ``python_executable`` defaults to :data:`sys.executable` (overridable in
    tests so the assertion never depends on the interpreter path).

    ``max_mem_gb`` (default ``8.0``) is the HQ RAM limit forwarded verbatim as
    ``--max-mem``.  The Tk boring branch always passes this value
    (``str(getattr(self.settings, "max_hq_mem_gb", 8))``), so the Qt shell
    passes ``float(state.max_hq_mem_gb)`` (default ``8.0``) for byte-identical
    parity; the ``8.0`` default here only applies when a caller passes nothing.
    """
    script_path = resolve_boring_script_path()
    command = [
        python_executable or sys.executable,
        script_path,
        "--csv",
        csv_path,
        "--out",
        output_dir,
        "--batch-size",
        str(batch_size),
        "--max-mem",
        str(max_mem_gb),
        "--chunk-size",
        str(chunk_size),
        "--log-dir",
        os.path.join(output_dir, "logs"),
        "--norm",
        str(normalize_method),
        "--save-as-float32" if save_final_as_float32 else "--no-save-as-float32",
        "--final-combine",
        str(final_combine),
        "--gpu" if request_gpu else "--no-gpu",
    ]
    return BoringRunRequest(
        command=command,
        script_path=script_path,
        csv_path=csv_path,
        output_dir=output_dir,
        batch_size=batch_size,
        chunk_size=chunk_size,
        normalize_method=normalize_method,
        save_final_as_float32=save_final_as_float32,
        final_combine=final_combine,
        max_mem_gb=max_mem_gb,
        request_gpu=request_gpu,
    )
