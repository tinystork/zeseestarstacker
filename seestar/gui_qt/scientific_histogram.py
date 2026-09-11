"""Bounded, read-only FINAL SCIENTIFIC histogram helper (R1).

Mission ``zsss-signed-float32-display-histogram-20260911`` (phase
``r1-final-scientific-histogram``).

Semantic separation (Qt UI)
---------------------------
The live Qt histogram is the **display / analysis-domain** histogram: it is
computed by :func:`seestar.gui_qt.preview_analysis.compute_histogram_float` on
the *mapped WB* buffer, whose domain is the non-negative ``[0, upper]``
analysis window.  That surface stays exactly as-is.

This module derives a **scientific** histogram directly from the durable final
scientific carrier (the signed float32 FITS written by the engine), preserving
the true signed ``[min, max]`` (e.g. approx. ``-4.73 .. +82.67``).  It is:

* **read-only** — the carrier is never mutated, offset or clipped;
* **bounded** — binning / order statistics run over the deterministic
  fixed-stride ``_cap_sample`` (``MAX_SAMPLE_PIXELS``) reused from
  ``preview_analysis``; exact ``min``/``max`` are single-pass finite
  reductions (off the GUI thread — the whole computation runs on the
  histogram worker thread via a :class:`~seestar.gui_qt.histogram_worker.HistogramCoordinator`);
* **sample-retaining** — the capped per-channel SAMPLE is returned inside the
  model so a later task (R2, adaptive re-binning on a narrow zoom) can re-bin a
  visible sub-range without a full recompute / a second carrier read.

The model is explicitly tagged ``domain == "scientific"`` (vs. the display
model, which has no such tag), so the UI can distinguish the two spaces.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Optional, Tuple

from .preview_analysis import MAX_SAMPLE_PIXELS, _cap_sample

# Bin count parity with the display histogram.
SCIENTIFIC_HISTOGRAM_BINS = 512


def _load_numpy():
    """Lazily import numpy (module object, or ``None`` when unavailable)."""
    try:
        return importlib.import_module("numpy")
    except Exception:
        return None


def _ensure_hwc(np: Any, arr: Any) -> Any:
    """Normalize a 3-D carrier to HWC channel-last order (2-D is returned as-is).

    The scientific FITS carrier is stored channel-first ``(C, H, W)`` (e.g.
    ``(3, 5760, 3240)``); the pure helper contract is channel-last ``(H, W, C)``.
    A 3-D array whose first axis is a small channel count (1/3/4) and whose last
    axis is not is transposed.  This is a *view*, never a data mutation.
    """
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[2]:
            return np.moveaxis(arr, 0, -1)
    return arr


def _scientific_channels(np: Any, arr: Any) -> List[Tuple[str, Any]]:
    """Return ``[(name, 1D flattened values)]`` for a signed carrier (HWC/2D)."""
    if arr.ndim == 2:
        return [("L", arr.ravel())]
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            return [
                ("R", arr[..., 0].ravel()),
                ("G", arr[..., 1].ravel()),
                ("B", arr[..., 2].ravel()),
            ]
        if arr.shape[2] == 1:
            return [("L", arr[..., 0].ravel())]
    return []


def compute_scientific_histogram(
    raw_signed: Any,
    bins: int = SCIENTIFIC_HISTOGRAM_BINS,
    source: str = "in_memory",
) -> Optional[Dict[str, Any]]:
    """Bounded truthful histogram/stats of a signed scientific carrier.

    Parameters
    ----------
    raw_signed:
        2-D ``(H, W)`` or 3-D ``(H, W, C)`` / ``(C, H, W)`` signed array (the
        final scientific float32 carrier).  Values are read only, never
        modified.
    bins:
        Bin count (``512`` by default, display parity).
    source:
        Provenance tag recorded in the model (e.g. ``"final_fits"``).

    Returns
    -------
    ``None`` when unavailable/unusable (no numpy, empty, unsupported shape, or a
    channel with no finite value).  Otherwise a dict with:

    * ``domain`` — ``"scientific"`` (the display/scientific distinction tag);
    * ``source`` — provenance tag;
    * ``channels`` — ``["L"]`` (mono) or ``["R", "G", "B"]``;
    * ``range`` — truthful global finite ``(min, max)`` of the carrier
      (``min`` may be negative; never clipped to 0);
    * ``bins`` — bin count;
    * ``counts`` — per-channel ``int64`` bin counts over the channel's signed
      range;
    * ``edges`` — per-channel bin edges (signed domain);
    * ``stats`` — per-channel ``{min, max, median, mean, std}`` (signed);
    * ``samples`` — per-channel **retained bounded deterministic SAMPLE**
      (1-D finite array, ``<= MAX_SAMPLE_PIXELS``) for R2 re-binning;
    * ``sample_pixels`` / ``total_finite`` — sampling diagnostics.

    ``min``/``max`` are exact finite reductions over the whole channel (a
    single pass); median/mean/std and the bins use the capped sample.
    """
    np = _load_numpy()
    if np is None:
        return None
    arr = np.asarray(raw_signed)
    if arr.size == 0 or arr.ndim not in (2, 3):
        return None
    arr = _ensure_hwc(np, arr)
    channels = _scientific_channels(np, arr)
    if not channels:
        return None

    counts: Dict[str, Any] = {}
    edges: Dict[str, Any] = {}
    stats: Dict[str, Dict[str, float]] = {}
    samples: Dict[str, Any] = {}
    global_min = None
    global_max = None
    total_finite = 0
    sample_pixels = 0

    for name, values in channels:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None  # fail closed: never fabricate a synthetic pixel
        cmin = float(np.min(finite))
        cmax = float(np.max(finite))
        total_finite += int(finite.size)
        # Bounded, deterministic histogram/stats sample (R2 re-binning basis).
        sample = _cap_sample(np, finite)
        samples[name] = sample
        sample_pixels += int(sample.size)
        if cmax <= cmin:
            # Constant channel: widen symmetrically so the range is valid.
            lo, hi = cmin - 0.5, cmax + 0.5
        else:
            lo, hi = cmin, cmax
        hist, bin_edges = np.histogram(sample, bins=bins, range=(lo, hi))
        counts[name] = hist.astype(np.int64)
        edges[name] = bin_edges
        stats[name] = {
            "min": cmin,
            "max": cmax,
            "median": float(np.median(sample)),
            "mean": float(np.mean(sample, dtype=np.float64)),
            "std": float(np.std(sample, dtype=np.float64)),
        }
        global_min = cmin if global_min is None else min(global_min, cmin)
        global_max = cmax if global_max is None else max(global_max, cmax)

    return {
        "domain": "scientific",
        "source": source,
        "channels": [name for name, _ in channels],
        "range": (float(global_min), float(global_max)),
        "bins": int(bins),
        "counts": counts,
        "edges": edges,
        "stats": stats,
        "samples": samples,
        "sample_pixels": int(sample_pixels),
        "total_finite": int(total_finite),
    }


def _header_truthy(value) -> bool:
    """Coerce a FITS logical/string value to bool (``T``/``F``/``true``/``1``)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().upper() in ("T", "TRUE", "1", "YES")
    if value is None:
        return False
    return bool(value)


def read_raw_signed_fits(path: str) -> Optional[Any]:
    """Read the RAW signed pixel data of a FITS carrier (never rescaled).

    ``astropy.io.fits`` is imported lazily (split string literal) so this module
    is importable without astropy.  ``do_not_scale_image_data=True`` guarantees
    the physical signed values are preserved exactly as written (no BZERO/
    BSCALE application), so a BITPIX=-32 carrier keeps its negative floor.

    R3: the exported FITS may carry a constant additive viewer-compatibility
    offset (``ZSCOMPAT``/``ZSOFFSET``).  When ``ZSCOMPAT`` is true and
    ``ZSOFFSET != 0``, this returns the ORIGINAL signed carrier
    (``stored - ZSOFFSET``) so the final scientific histogram keeps describing
    the true science (min approx. -4.73), not the offset export (min 0).  The
    subtraction is float32-safe and never clips.

    Returns ``None`` for a missing/unreadable path.
    """
    np = _load_numpy()
    if np is None or not path or not os.path.isfile(path):
        return None
    try:
        fits = importlib.import_module(".".join(("astropy", "io", "fits")))
        with fits.open(path, do_not_scale_image_data=True, memmap=False) as hdul:
            data = None
            header = None
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None:
                    data = hdu.data
                    header = getattr(hdu, "header", None)
                    break
            if data is None:
                return None
            arr = np.array(data, copy=True)
            zsoffset = None if header is None else header.get("ZSOFFSET")
            zscompat = None if header is None else header.get("ZSCOMPAT")
        if zsoffset is not None:
            try:
                offset = float(zsoffset)
            except (TypeError, ValueError):
                offset = 0.0
            if offset != 0.0 and _header_truthy(zscompat):
                arr = arr - offset
        return arr
    except Exception:
        return None


def scientific_histogram_from_fits(
    path: str, bins: int = SCIENTIFIC_HISTOGRAM_BINS
) -> Optional[Dict[str, Any]]:
    """Read the durable final scientific FITS and compute its signed histogram.

    Runs on the histogram worker thread (never the GUI thread).  Returns ``None``
    when the file is missing/unreadable (fail closed).
    """
    arr = read_raw_signed_fits(path)
    if arr is None:
        return None
    return compute_scientific_histogram(arr, bins=bins, source="final_fits")


def format_scientific_histogram_status(model: Optional[Dict[str, Any]]) -> str:
    """Deterministic, clearly-labelled status line for the scientific model.

    Returns ``""`` for an absent model (so the label stays empty until a final
    scientific carrier is available).  The text explicitly names the scientific
    space and the provenance so it can never be confused with the live
    display-domain histogram status.
    """
    if not model:
        return ""
    try:
        lo, hi = model.get("range") or (0.0, 0.0)
        parts = []
        for name in model.get("channels") or []:
            st = (model.get("stats") or {}).get(name) or {}
            if not st:
                continue
            parts.append(
                f"{name} {float(st.get('min', float('nan'))):.4g}"
                f"..{float(st.get('max', float('nan'))):.4g}"
            )
        provenance = "final FITS" if model.get("source") == "final_fits" else str(
            model.get("source") or "signed carrier"
        )
        body = " ".join(parts)
        return f"Scientific ({provenance}) [{float(lo):.4g}..{float(hi):.4g}] {body}".strip()
    except Exception:
        return "Scientific (final FITS)"
