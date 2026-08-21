"""Qt-only helpers for progress/log time ergonomics (M6).

Pure, dependency-free functions for formatting elapsed/remaining durations and
deriving a simple, deterministic remaining-time estimate from an elapsed time
and a percent-complete value.  No Qt, Tk or engine imports, so the whole
``gui_qt`` source surface stays clean under the import-hygiene scan and these
helpers are unit-testable without a live ``QApplication`` or any real sleeping.

The estimate is intentionally naive and honest:

* ``elapsed = now - start`` (seconds),
* ``remaining = elapsed * (100 - percent) / percent`` when ``0 < percent < 100``,
* ``None`` (rendered as ``"—"``) whenever the estimate is unknowable.
"""

from __future__ import annotations

from typing import Optional

UNKNOWN = "—"


def format_duration(seconds: Optional[float]) -> str:
    """Format a non-negative duration as ``H:MM:SS`` (or ``MM:SS`` under an hour).

    Returns :data:`UNKNOWN` (``"—"``) for ``None`` or negative input, matching
    the "unknown" state the progress surface shows before a meaningful
    estimate exists.
    """
    if seconds is None or seconds < 0:
        return UNKNOWN
    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def estimate_remaining_seconds(
    elapsed_seconds: Optional[float], percent: Optional[float]
) -> Optional[float]:
    """Estimate remaining seconds from elapsed time and a percent in ``(0, 100)``.

    Returns ``None`` (unknown) when the estimate cannot be formed honestly:

    * ``percent`` is ``None`` / non-numeric,
    * ``percent <= 0`` (no progress yet — division by zero avoided),
    * ``percent >= 100`` (the caller treats "done" separately),
    * ``elapsed_seconds`` is ``None`` or negative.

    Otherwise returns ``elapsed * (100 - percent) / percent``.
    """
    if elapsed_seconds is None or elapsed_seconds < 0:
        return None
    try:
        p = float(percent)
    except (TypeError, ValueError):
        return None
    if p <= 0.0 or p >= 100.0:
        return None
    return float(elapsed_seconds) * (100.0 - p) / p
