"""Final-combination business mapping (Tk parity).

Canonical mapping between the user-facing "Final Combine" choice and the
backend-relevant fields ``stack_final_combine``, ``reproject_between_batches``
and ``reproject_coadd_final``.

The values here are byte-for-byte the same as the historical Tk GUI:

* the Tk ``final_keys`` list (the five display-order keys), and
* the Tk settings manager's ``update_from_ui`` derivation
  ``reproject_between_batches = (stack_final_combine == "reproject")`` /
  ``reproject_coadd_final = (stack_final_combine == "reproject_coadd")``.

Labels match the Tk ``localization`` ``combine_method_*`` entries.  This module
is pure stdlib (no Qt, no Tk, no engine) so it can be unit-tested in complete
isolation.
"""

from __future__ import annotations

from typing import Dict, Tuple

# Display-order list of backend keys (identical to the Tk ``final_keys`` list).
FINAL_COMBINE_KEYS = (
    "mean",
    "median",
    "winsorized_sigma_clip",
    "reproject",
    "reproject_coadd",
)

# Key -> user-facing label (identical to the Tk ``localization`` entries).
FINAL_COMBINE_LABELS: Dict[str, str] = {
    "mean": "Mean",
    "median": "Median",
    "winsorized_sigma_clip": "Winsorized Sigma Clip",
    "reproject": "Reproject",
    "reproject_coadd": "Reproject & Coadd",
}

# Label -> key reverse lookup.
FINAL_COMBINE_LABEL_TO_KEY: Dict[str, str] = {
    label: key for key, label in FINAL_COMBINE_LABELS.items()
}


def final_combine_flags(key: str) -> Tuple[bool, bool]:
    """Return ``(reproject_between_batches, reproject_coadd_final)`` for a key.

    Mirrors the Tk settings manager's ``update_from_ui``:
    ``reproject_between_batches = (combine == "reproject")`` and
    ``reproject_coadd_final = (combine == "reproject_coadd")``.  The plain
    combine keys (``mean``/``median``/``winsorized_sigma_clip``) map to
    ``(False, False)``.
    """
    return (key == "reproject", key == "reproject_coadd")


def final_combine_key_from_flags(
    reproject_between_batches: bool,
    reproject_coadd_final: bool,
    fallback: str = "mean",
) -> str:
    """Reconstruct a final-combine key from the two derived flags.

    Used for the low-level (internal/advanced) checkbox path when the flags are
    set directly rather than via the user-facing combo.  ``fallback`` is the
    last plain combine value (``mean``/``median``/``winsorized_sigma_clip``).
    """
    if reproject_between_batches:
        return "reproject"
    if reproject_coadd_final:
        return "reproject_coadd"
    if fallback in FINAL_COMBINE_KEYS:
        return fallback
    return "mean"
