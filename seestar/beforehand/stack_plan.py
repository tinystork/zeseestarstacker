"""Utility to build stacking plans for ZeAnalyser.

This module exposes functions to generate CSV stacking plans from
analysis results.  Each result dict should at least contain the keys
``'mount'``, ``'bortle'``, ``'telescope'``, ``'date_obs'``, ``'filter'``,
``'exposure'`` and ``'path'``.  Results with ``status!='ok'`` are ignored.

The main function :func:`generate_stacking_plan` filters results
according to provided criteria, sorts them, groups them into batches and
returns a list of rows ready to be written to CSV.

The helper :func:`write_stacking_plan_csv` writes the CSV file.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import os
from typing import Dict, Iterable, List, Sequence, Tuple


FieldList = Dict[str, Sequence[str]]
SortSpec = Sequence[Tuple[str, bool]]  # (field, reverse)


def _extract_session_date(date_obs: str | None) -> str:
    """Return ``YYYY-MM-DD`` from DATE-OBS value or ``''`` if unavailable."""
    if not date_obs:
        return ""
    return str(date_obs).split("T")[0]


def generate_stacking_plan(
    results: Iterable[Dict],
    *,
    include_exposure_in_batch: bool = False,
    criteria: FieldList | None = None,
    sort_spec: SortSpec | None = None,
) -> List[Dict[str, str]]:
    """Build stacking plan rows from analysis results.

    Parameters
    ----------
    results : iterable of dict
        Analysis results.
    include_exposure_in_batch : bool, optional
        If ``True`` the exposure value is part of the ``batch_id``.
    criteria : dict, optional
        Mapping ``field -> allowed values``.  Values are sequences of
        strings.  ``None`` means no filtering on this field.
    sort_spec : sequence, optional
        List of ``(field, reverse)`` tuples used for sorting.  Sorting is
        stable so later items take precedence.

    Returns
    -------
    list of dict
        Each dict represents a row of the CSV with keys ``order``,
        ``batch_id`` and the fields ``mount``, ``bortle``, ``telescope``,
        ``session_date``, ``filter``, ``exposure`` and ``file_path``.
    """
    criteria = criteria or {}
    sort_spec = sort_spec or []

    # Filter step
    rows = []
    for r in results:
        if r.get("status") != "ok":
            continue
        mount = r.get("mount", "")
        bortle = str(r.get("bortle", ""))
        tele = r.get("telescope") or "Unknown"
        session_date = _extract_session_date(r.get("date_obs"))
        filt = r.get("filter", "")
        expo = r.get("exposure", "")
        path = r.get("path", "")

        values = {
            "mount": mount,
            "bortle": bortle,
            "telescope": tele,
            "session_date": session_date,
            "filter": filt,
            "exposure": str(expo),
        }

        skip = False
        for field, allowed in criteria.items():
            if allowed is None:
                continue
            val = values.get(field)
            if val not in allowed:
                skip = True
                break
        if skip:
            continue

        row = {
            "mount": mount,
            "bortle": bortle,
            "telescope": tele,
            "session_date": session_date,
            "filter": filt,
            "exposure": str(expo),
            "file_path": path,
        }
        rows.append(row)

    # Sorting using stable sort
    for field, reverse in reversed(sort_spec):
        rows.sort(key=lambda r, f=field: r.get(f, ""), reverse=reverse)

    # Batch id creation
    plan_rows = []
    batch_counts = defaultdict(int)
    for row in rows:
        batch_parts = [row["telescope"], row["session_date"], row["filter"]]
        if include_exposure_in_batch:
            batch_parts.append(row["exposure"])
        batch_id = "_".join(part for part in batch_parts if part)
        batch_counts[batch_id] += 1
        plan_rows.append({"batch_id": batch_id, **row})

    # Assign order
    for i, row in enumerate(plan_rows, start=1):
        row["order"] = i

    return plan_rows


def write_stacking_plan_csv(csv_path: str, rows: Iterable[Dict[str, str]]) -> None:
    """Write stacking plan rows to ``csv_path`` in UTF-8."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "order",
                    "batch_id",
                    "mount",
                    "bortle",
                    "telescope",
                    "session_date",
                    "filter",
                    "exposure",
                    "file_path",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        row.get("order", ""),
                        row.get("batch_id", ""),
                        row.get("mount", ""),
                        row.get("bortle", ""),
                        row.get("telescope", ""),
                        row.get("session_date", ""),
                        row.get("filter", ""),
                        row.get("exposure", ""),
                        row.get("file_path", ""),
                    ]
                )
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write to '{csv_path}'. File is in use or the directory is not writable."
        ) from exc

    return None
