"""Passive, fail-open Drizzle science diagnostics (ZSSS-DRIZZLE-CLOSURE-P1).

This module is **instrumentation only**.  It observes the *current* production
Drizzle (Standard M3) science without ever mutating it, influencing it, or
changing control flow.  Its sole output is one bounded, versioned JSON artifact
per run written atomically/best-effort into the run output area.

Hard contract (Phase 1 scientific freeze)
-----------------------------------------
* **Passive** — every helper reads its inputs; none of them mutate the arrays
  they are given (all reductions allocate their own temporaries).  The
  collector never returns anything consumed by the stacking path.
* **Fail-open** — every public entry point catches ``Exception`` internally.
  A calculation failure or an I/O failure only warns (debug log) and returns a
  falsy/neutral value; it can never abort a run or change a scientific array.
* **Bounded** — no full-frame array is ever serialized.  Only scalars, short
  bounded lists (percentiles, per-channel/per-bin summaries) and the bounded
  lifecycle ring are persisted.  Distance maps are computed transiently and
  dropped; they are never persisted.
* **No science rule** — the module *reports* threshold/N_eff/boundary/
  conditioning candidate data for human/architect comparison.  It never
  applies a floor, mask, clip or rejection, and never declares negative
  science invalid.

Documented methods
------------------
* Angular pixel scale: the mean of ``abs(proj_plane_pixel_scales(wcs))`` in
  degrees/pixel at the WCS reference point (a robust scalar for a not-too-
  skewed grid), falling back to ``sqrt(|det(CD)|)``.  The geometry candidate
  is ``output_scale / input_scale`` and is **diagnostic-only** — it is never
  passed to ``Drizzle.add_image`` in Phase 1.
* Robust extrema: deterministic order-statistic selection over the *valid*
  physical-support population (stable argsort, flat-index tie-break).  Sample
  positions/sizes are fixed constants so JSON output is reproducible.
* Distance bins: the Euclidean distance transform (EDT) of the physical
  support mask gives each support pixel its distance to the nearest unsupported
  pixel; bins are the documented half-open intervals ``[0,1]``, ``(1,4]``,
  ``(4,8]``, ``(8,16]``, ``(16,inf)``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
ARTIFACT_FILENAME = "drizzle_science_diagnostics.json"

# Neutral support floor used by the current production contract.  Repeated here
# as a *diagnostic constant* so the module can label the current epsilon without
# importing the science module (keeps the dependency direction one-way).
WEIGHT_EPSILON = 1e-9

# Diagnostic "small positive WHT" cut for the fraction reported in the WHT and
# boundary sections.  Purely descriptive; never applied as a mask.
SMALL_POSITIVE_WHT = 1e-4

# Deterministic threshold-sweep candidates (diagnostic only).
DEFAULT_ABS_THRESHOLDS = (1e-4, 1e-3, 1e-2, 1e-1)
DEFAULT_REL_THRESHOLDS = (1e-3, 1e-2, 0.05, 0.1)

# Bounded extrema sample size (per channel / per selection side).
EXTREMA_COUNT = 8

# Documented boundary distance bins (distance in output pixels).
DISTANCE_BIN_LABELS = ("0-1", "2-4", "5-8", "9-16", ">16")
DISTANCE_BIN_EDGES = (
    (0.0, 1.0),
    (1.0, 4.0),
    (4.0, 8.0),
    (8.0, 16.0),
    (16.0, math.inf),
)

# Local positive-WHT reference: tile edge (output pixels) for the bounded
# per-tile robust reference used by the conditioning-candidate section.
LOCAL_REFERENCE_TILE = 64

# Native WHT magnitude bins reported in the WHT diagnostics section.
WHT_MAGNITUDE_BINS = (
    ("neg", lambda w: w < 0.0),
    ("le_1e-4", lambda w: (w > 0.0) & (w <= 1e-4)),
    ("le_1e-3", lambda w: (w > 1e-4) & (w <= 1e-3)),
    ("le_1e-2", lambda w: (w > 1e-3) & (w <= 1e-2)),
    ("le_1e-1", lambda w: (w > 1e-2) & (w <= 1e-1)),
    ("gt_1e-1", lambda w: w > 1e-1),
)


# ---------------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------------


def _f(value):
    """Return a finite JSON-safe ``float`` or ``None`` (never raises)."""
    try:
        if value is None or isinstance(value, bool):
            return None
        f = float(value)
        return f if math.isfinite(f) else None
    except Exception:  # noqa: BLE001 — diagnostics are fail-open
        return None


def _i(value):
    """Return a JSON-safe ``int`` or ``None`` (never raises)."""
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def robust_pixel_scale_deg(wcs):
    """Return ``(scale_deg_per_pixel, method)`` or ``(None, reason)``.

    Method (documented, deterministic):

    1. ``astropy.wcs.utils.proj_plane_pixel_scales`` — the projected linear
       pixel scale (degrees/pixel) along each WCS axis at the reference point;
       the reported scalar is the mean of the absolute values.
    2. Fallback: ``sqrt(|det(CD)|)`` of the pixel scale matrix (for a
       skewed/anisotropic grid this is the geometric mean of the axis scales).

    Never raises; an unusable/absent WCS yields ``None`` with a short reason.
    """
    if wcs is None:
        return None, "wcs_absent"
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=np.float64)
        scales = scales[np.isfinite(scales)]
        if scales.size == 0:
            return None, "wcs_no_finite_scale"
        val = float(np.mean(np.abs(scales)))
        if not math.isfinite(val) or val <= 0.0:
            return None, "wcs_nonpositive_scale"
        return val, "proj_plane_pixel_scales_mean_abs"
    except Exception:  # noqa: BLE001
        try:
            cd = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
            det = float(np.linalg.det(cd))
            val = math.sqrt(abs(det))
            if not math.isfinite(val) or val <= 0.0:
                return None, "wcs_det_nonpositive"
            return val, "sqrt_abs_det_cd"
        except Exception:  # noqa: BLE001
            return None, "wcs_unavailable"


def geometry_diagnostic(reference_wcs, output_wcs, kernel=None, scale=None):
    """Build the resolved-geometry diagnostic record (fail-open).

    Returns a JSON-safe dict.  ``pixel_scale_ratio_candidate`` is
    ``output_pixel_scale / input_pixel_scale`` (diagnostic only; never passed
    upstream in Phase 1).  When either WCS is unavailable/invalid the record
    carries ``available=False`` and an explicit ``reason`` instead of a
    fabricated ratio.
    """
    in_scale, in_method = robust_pixel_scale_deg(reference_wcs)
    out_scale, out_method = robust_pixel_scale_deg(output_wcs)
    rec = {
        "available": False,
        "reason": None,
        "method": None,
        "input_pixel_scale_deg": _f(in_scale),
        "output_pixel_scale_deg": _f(out_scale),
        "pixel_scale_ratio_current": 1.0,
        "pixel_scale_ratio_current_source": "upstream_default",
        "pixel_scale_ratio_candidate": None,
        "candidate_source": "wcs_ratio",
        "kernel": None if kernel is None else str(kernel),
        "scale": _f(scale),
    }
    if in_scale is None or out_scale is None:
        rec["reason"] = "input_" + in_method if in_scale is None else (
            "output_" + out_method
        )
        return rec
    ratio = out_scale / in_scale
    if not math.isfinite(ratio) or ratio <= 0.0:
        rec["reason"] = "nonpositive_wcs_ratio"
        return rec
    rec["available"] = True
    rec["method"] = f"{in_method}|{out_method}"
    rec["pixel_scale_ratio_candidate"] = _f(ratio)
    return rec


def contract_diagnostic(kernel, pixfrac, exptime=1.0, in_units="counts",
                        fillval="0.0", iscale=None, pixel_scale_ratio=None):
    """Describe the *effective* ``Drizzle.add_image`` call semantics.

    Explicit-vs-default provenance is recorded per argument so the current
    omissions are unambiguous: ``iscale`` and ``pixel_scale_ratio`` are
    ``upstream_default`` (the wrapper does not pass them), ``fillval`` is the
    accumulator's construction default.  This mirrors the wrapper's literal
    call and is cross-checked by a spy test against the real engine call.
    """
    try:
        in_units = str(in_units)
    except Exception:  # noqa: BLE001
        in_units = "counts"
    expscale = exptime if in_units == "counts" else 1.0
    return {
        "kernel_effective": None if kernel is None else str(kernel),
        "kernel_source": "explicit",
        "pixfrac_effective": _f(pixfrac),
        "pixfrac_source": "explicit",
        "iscale_effective": 1.0 if iscale is None else _f(iscale),
        "iscale_source": "upstream_default" if iscale is None else "explicit",
        "pixel_scale_ratio_effective": (
            1.0 if pixel_scale_ratio is None else _f(pixel_scale_ratio)
        ),
        "pixel_scale_ratio_source": (
            "upstream_default" if pixel_scale_ratio is None else "explicit"
        ),
        "in_units_effective": in_units,
        "in_units_source": "explicit",
        "exptime_effective": _f(exptime),
        "exptime_source": "explicit",
        "wht_scale_effective": _f(expscale),
        "wht_scale_source": "explicit",
        "fillval_effective": None if fillval is None else str(fillval),
        "fillval_source": "accumulator_default",
        "weight_map_present": True,
        "weight_map_source": "explicit",
    }


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------


def sci_channel_stats(arr):
    """Pre-stretch SCI statistics for one channel (read-only, fail-open).

    Reports ``min``, ``P0.001``, ``P0.01``, ``median``, ``P99.99``,
    ``P99.999``, ``max``, the fraction ``< 0``, the finite fraction and the
    non-finite fraction.  Negative values are *reported*, never declared
    invalid.  Never raises.
    """
    out = {
        "count": 0,
        "finite_count": 0,
        "min": None,
        "p0_001": None,
        "p0_01": None,
        "median": None,
        "p99_99": None,
        "p99_999": None,
        "max": None,
        "fraction_lt_zero": None,
        "finite_fraction": None,
        "nonfinite_fraction": None,
    }
    try:
        a = np.asarray(arr)
        flat = a.ravel()
        total = int(flat.size)
        out["count"] = total
        if total == 0:
            return out
        finite_mask = np.isfinite(flat)
        n_finite = int(np.count_nonzero(finite_mask))
        out["finite_count"] = n_finite
        out["finite_fraction"] = _f(n_finite / total)
        out["nonfinite_fraction"] = _f((total - n_finite) / total)
        if n_finite == 0:
            return out
        finite = flat[finite_mask].astype(np.float64, copy=False)
        out["min"] = _f(np.min(finite))
        out["max"] = _f(np.max(finite))
        out["median"] = _f(np.median(finite))
        out["p0_001"] = _f(np.percentile(finite, 0.001))
        out["p0_01"] = _f(np.percentile(finite, 0.01))
        out["p99_99"] = _f(np.percentile(finite, 99.99))
        out["p99_999"] = _f(np.percentile(finite, 99.999))
        n_neg = int(np.count_nonzero(finite < 0.0))
        out["fraction_lt_zero"] = _f(n_neg / n_finite)
    except Exception as exc:  # noqa: BLE001
        logger.debug("sci_channel_stats failed (non-fatal): %s", exc)
    return out


def sci_stats_hwc(arr):
    """Per-channel :func:`sci_channel_stats` for ``(H, W)`` or ``(H, W, C)``."""
    try:
        a = np.asarray(arr)
        if a.ndim == 2:
            return [sci_channel_stats(a)]
        if a.ndim == 3:
            return [sci_channel_stats(a[..., c]) for c in range(a.shape[-1])]
        return []
    except Exception as exc:  # noqa: BLE001
        logger.debug("sci_stats_hwc failed (non-fatal): %s", exc)
        return []


def _percentile(finite_sorted_or_flat, p):
    try:
        return _f(np.percentile(finite_sorted_or_flat, p))
    except Exception:  # noqa: BLE001
        return None


def wht_channel_diagnostics(wht, sci=None, n_extrema=EXTREMA_COUNT):
    """Native *signed* WHT diagnostics for one channel (read-only, fail-open).

    Reports min/max, robust percentiles of strictly positive WHT, the
    documented magnitude-bin fractions, and the native WHT at the SCI min/max
    plus the deterministic robust SCI extrema when ``sci`` is provided.  No
    clipping/abs()/masking is applied.
    """
    out = {
        "min": None,
        "max": None,
        "positive_percentiles": {},
        "bin_fractions": {},
        "n_positive": None,
        "sci_min_wht": None,
        "sci_max_wht": None,
        "extrema": [],
    }
    try:
        w = np.asarray(wht, dtype=np.float64)
        flat = w.ravel()
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return out
        out["min"] = _f(np.min(finite))
        out["max"] = _f(np.max(finite))
        total = int(flat.size)
        positive = flat[np.isfinite(flat) & (flat > 0.0)]
        out["n_positive"] = int(positive.size)
        if positive.size:
            out["positive_percentiles"] = {
                "p1": _percentile(positive, 1.0),
                "p50": _percentile(positive, 50.0),
                "p99": _percentile(positive, 99.0),
                "p99_9": _percentile(positive, 99.9),
                "max": _f(np.max(positive)),
            }
        for label, pred in WHT_MAGNITUDE_BINS:
            try:
                n = int(np.count_nonzero(pred(flat)))
                out["bin_fractions"][label] = _f(n / total) if total else None
            except Exception:  # noqa: BLE001
                out["bin_fractions"][label] = None
        if sci is not None:
            s = np.asarray(sci, dtype=np.float64)
            if s.shape == w.shape:
                sflat = s.ravel()
                finite_s = np.isfinite(sflat)
                if np.any(finite_s):
                    i_min = int(np.argmin(np.where(finite_s, sflat, np.inf)))
                    i_max = int(np.argmax(np.where(finite_s, sflat, -np.inf)))
                    out["sci_min_wht"] = _f(flat[i_min])
                    out["sci_max_wht"] = _f(flat[i_max])
                out["extrema"] = sci_extrema_records(s, w, n=n_extrema)
    except Exception as exc:  # noqa: BLE001
        logger.debug("wht_channel_diagnostics failed (non-fatal): %s", exc)
    return out


def wht_diagnostics_hwc(wht, sci=None, n_extrema=EXTREMA_COUNT):
    """Per-channel :func:`wht_channel_diagnostics` for ``(H, W)``/``(H, W, C)``."""
    try:
        w = np.asarray(wht)
        s = None if sci is None else np.asarray(sci)
        if w.ndim == 2:
            return [wht_channel_diagnostics(w, s, n_extrema)]
        if w.ndim == 3:
            res = []
            for c in range(w.shape[-1]):
                sc = None if s is None else s[..., c]
                res.append(wht_channel_diagnostics(w[..., c], sc, n_extrema))
            return res
        return []
    except Exception as exc:  # noqa: BLE001
        logger.debug("wht_diagnostics_hwc failed (non-fatal): %s", exc)
        return []


def _valid_support_mask(sci, wht, wht_threshold):
    """Documented physical-support mask: finite SCI, finite WHT, WHT > threshold."""
    s = np.asarray(sci, dtype=np.float64)
    w = np.asarray(wht, dtype=np.float64)
    return np.isfinite(s) & np.isfinite(w) & (w > float(wht_threshold))


def select_extrema_indices(values_2d, valid_mask, n):
    """Deterministic bounded high/low index selection over a valid population.

    Returns ``(low_indices, high_indices)`` as *flat* indices into
    ``values_2d``.  Selection is by stable argsort of the valid values (ties
    broken by original flat order); at most ``n`` indices per side.  Never
    raises (returns two empty arrays on failure).
    """
    try:
        v = np.asarray(values_2d, dtype=np.float64).ravel()
        m = np.asarray(valid_mask, dtype=bool).ravel()
        idx = np.flatnonzero(m & np.isfinite(v))
        if idx.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        k = max(0, min(int(n), idx.size))
        if k == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        order = np.argsort(v[idx], kind="stable")
        lo = idx[order[:k]]
        hi = idx[order[-k:][::-1]]
        return lo.astype(np.int64), hi.astype(np.int64)
    except Exception as exc:  # noqa: BLE001
        logger.debug("select_extrema_indices failed (non-fatal): %s", exc)
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)


def _flat_index_records(values_2d, flat_indices, width, extra_arrays=None):
    """Build bounded JSON-safe records for flat indices (row/col + values)."""
    recs = []
    try:
        v = np.asarray(values_2d, dtype=np.float64).ravel()
        extras = {}
        if extra_arrays:
            for name, arr in extra_arrays.items():
                try:
                    extras[name] = np.asarray(arr, dtype=np.float64).ravel()
                except Exception:  # noqa: BLE001
                    extras[name] = None
        for fi in np.asarray(flat_indices).tolist():
            rec = {
                "index": int(fi),
                "row": int(fi) // int(width) if width else None,
                "col": int(fi) % int(width) if width else None,
                "sci": _f(v[fi]) if fi < v.size else None,
            }
            for name, arr in extras.items():
                rec[name] = _f(arr[fi]) if arr is not None and fi < arr.size else None
            recs.append(rec)
    except Exception as exc:  # noqa: BLE001
        logger.debug("_flat_index_records failed (non-fatal): %s", exc)
    return recs


def sci_extrema_records(sci, wht, n=EXTREMA_COUNT, w1=None, w2=None,
                        neff=None, distance=None):
    """SCI min/max + robust SCI extrema with optional per-pixel co-diagnostics."""
    try:
        s = np.asarray(sci, dtype=np.float64)
        w = np.asarray(wht, dtype=np.float64)
        h, wid = s.shape
        mask = _valid_support_mask(s, w, WEIGHT_EPSILON)
        extras = {"wht": w}
        if w1 is not None:
            extras["sup_w1"] = w1
        if w2 is not None:
            extras["sup_w2"] = w2
        if neff is not None:
            extras["n_eff"] = neff
        if distance is not None:
            extras["distance"] = distance
        lo, hi = select_extrema_indices(s, mask, n)
        flat = s.ravel()
        finite = np.isfinite(flat) & mask.ravel()
        recs = []
        if np.any(finite):
            idx = np.flatnonzero(finite)
            vals = flat[idx]
            i_min = idx[int(np.argmin(vals))]
            i_max = idx[int(np.argmax(vals))]
            recs = _flat_index_records(s, [i_min, i_max], wid, extras)
            # rename the two head records for clarity
            if recs:
                recs[0]["kind"] = "sci_abs_min"
                recs[1]["kind"] = "sci_abs_max"
        low_recs = _flat_index_records(s, lo, wid, extras)
        high_recs = _flat_index_records(s, hi, wid, extras)
        for r in low_recs:
            r["kind"] = "sci_robust_low"
        for r in high_recs:
            r["kind"] = "sci_robust_high"
        recs = recs + low_recs + high_recs
        # de-duplicate indices while preserving order
        seen = set()
        uniq = []
        for r in recs:
            if r["index"] in seen:
                continue
            seen.add(r["index"])
            uniq.append(r)
        return uniq
    except Exception as exc:  # noqa: BLE001
        logger.debug("sci_extrema_records failed (non-fatal): %s", exc)
        return []


# ---------------------------------------------------------------------------
# threshold sweep (diagnostic only)
# ---------------------------------------------------------------------------


def threshold_sweep(wht, sci, positive_reference=None,
                    abs_candidates=DEFAULT_ABS_THRESHOLDS,
                    rel_candidates=DEFAULT_REL_THRESHOLDS):
    """Report what an (unapplied) WHT threshold *would* remove / keep.

    Current epsilon plus absolute candidates plus relative candidates derived
    from a robust positive-WHT reference.  For each threshold it reports the
    number and fraction of total physical-support pixels that would be removed
    and the SCI extrema that would remain.  Nothing is ever applied.
    """
    out = {"current_epsilon": _f(WEIGHT_EPSILON), "candidates": [],
           "positive_reference": _f(positive_reference),
           "positive_reference_source": None}
    try:
        w = np.asarray(wht, dtype=np.float64)
        s = np.asarray(sci, dtype=np.float64)
        if w.shape != s.shape:
            return out
        total = int(w.size)
        support = _valid_support_mask(s, w, WEIGHT_EPSILON)
        n_support = int(np.count_nonzero(support))
        out["total_pixels"] = total
        out["support_pixels"] = n_support

        ref = positive_reference
        ref_source = "provided"
        if ref is None:
            positive = w[np.isfinite(w) & (w > 0.0)]
            if positive.size:
                ref = float(np.percentile(positive, 99.0))
                ref_source = "positive_wht_p99"
        out["positive_reference"] = _f(ref)
        out["positive_reference_source"] = ref_source

        candidates = [("current_epsilon", float(WEIGHT_EPSILON))]
        for a in abs_candidates:
            candidates.append((f"abs_{a:g}", float(a)))
        if ref is not None and ref > 0.0:
            for r in rel_candidates:
                candidates.append((f"rel_{r:g}", float(r) * float(ref)))

        for name, thr in candidates:
            remove = support & ~(w > thr) & np.isfinite(w)
            n_remove = int(np.count_nonzero(remove))
            keep = support & (w > thr)
            sci_keep = s[keep]
            sci_keep = sci_keep[np.isfinite(sci_keep)]
            rec = {
                "name": name,
                "threshold": _f(thr),
                "removed_pixels": n_remove,
                "removed_fraction_of_support": (
                    _f(n_remove / n_support) if n_support else None
                ),
                "removed_fraction_of_total": (
                    _f(n_remove / total) if total else None
                ),
                "remaining_pixels": int(np.count_nonzero(keep)),
                "remaining_sci_min": _f(np.min(sci_keep)) if sci_keep.size else None,
                "remaining_sci_max": _f(np.max(sci_keep)) if sci_keep.size else None,
            }
            out["candidates"].append(rec)
    except Exception as exc:  # noqa: BLE001
        logger.debug("threshold_sweep failed (non-fatal): %s", exc)
    return out


# ---------------------------------------------------------------------------
# support conditioning / N_eff
# ---------------------------------------------------------------------------


def _neff_from_sup(w1, w2):
    """N_eff = SUP_W1**2 / SUP_W2 (overflow-resistant); 0 where W2 <= 0."""
    try:
        a = np.asarray(w1, dtype=np.float64)
        b = np.asarray(w2, dtype=np.float64)
        valid = b > 0.0
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            w1_sq = a * a
            out = w1_sq / b
            ratio = a / np.sqrt(b)
            safe = ratio * ratio
        overflowed = ~np.isfinite(w1_sq)
        out = np.where(overflowed & valid, safe, out)
        out = np.where(valid & np.isfinite(out) & (out >= 0.0), out, 0.0)
        return out
    except Exception:  # noqa: BLE001
        return None


def support_conditioning(w1, w2):
    """Bounded N_eff summaries for a genuinely present SUP_W1/SUP_W2 pair."""
    out = {
        "available": False,
        "reason": None,
        "n_eff_min": None,
        "n_eff_median": None,
        "n_eff_mean": None,
        "n_eff_max": None,
        "positive_fraction": None,
        "sup_w1_max": None,
        "sup_w2_max": None,
    }
    try:
        if w1 is None or w2 is None:
            out["reason"] = "support_accumulator_absent"
            return out
        a = np.asarray(w1, dtype=np.float64)
        b = np.asarray(w2, dtype=np.float64)
        if a.shape != b.shape:
            out["reason"] = "support_shape_mismatch"
            return out
        neff = _neff_from_sup(a, b)
        if neff is None:
            out["reason"] = "n_eff_unavailable"
            return out
        finite = neff[np.isfinite(neff)]
        out["available"] = True
        if finite.size:
            out["n_eff_min"] = _f(np.min(finite))
            out["n_eff_median"] = _f(np.median(finite))
            out["n_eff_mean"] = _f(np.mean(finite))
            out["n_eff_max"] = _f(np.max(finite))
            out["positive_fraction"] = _f(
                np.count_nonzero(finite > 0.0) / finite.size
            )
        fa = a[np.isfinite(a)]
        fb = b[np.isfinite(b)]
        out["sup_w1_max"] = _f(np.max(fa)) if fa.size else None
        out["sup_w2_max"] = _f(np.max(fb)) if fb.size else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("support_conditioning failed (non-fatal): %s", exc)
        out["reason"] = "exception"
    return out


def _local_positive_reference(wht, tile=LOCAL_REFERENCE_TILE):
    """Bounded per-tile robust positive-WHT reference (documented).

    Non-overlapping ``tile`` squares: each tile's reference is its
    ``P90`` of strictly-positive finite WHT; empty tiles get the global
    positive ``P90``.  Returns ``(tile_ref_map, global_ref, meta)``; the map is
    one scalar per tile (bounded by ``ceil(H/tile)*ceil(W/tile)``).
    """
    w = np.asarray(wht, dtype=np.float64)
    h, wid = w.shape
    positive = w[np.isfinite(w) & (w > 0.0)]
    global_ref = float(np.percentile(positive, 90.0)) if positive.size else 0.0
    tried = int(np.ceil(h / tile))
    twid = int(np.ceil(wid / tile))
    ref = np.full((tried, twid), global_ref, dtype=np.float64)
    for i in range(tried):
        r0 = i * tile
        r1 = min(h, r0 + tile)
        for j in range(twid):
            c0 = j * tile
            c1 = min(wid, c0 + tile)
            block = w[r0:r1, c0:c1]
            bp = block[np.isfinite(block) & (block > 0.0)]
            if bp.size:
                ref[i, j] = float(np.percentile(bp, 90.0))
    return ref, global_ref, {"tile": int(tile), "tiles_hw": (tried, twid),
                             "statistic": "positive_p90"}


def _tile_index(row, col, tile):
    return int(row) // int(tile), int(col) // int(tile)


def conditioning_candidates(sci, wht, w1=None, w2=None, n_extrema=EXTREMA_COUNT,
                            tile=LOCAL_REFERENCE_TILE, distance=None):
    """Bounded candidate conditioning evidence at the SCI extrema.

    For each selected extreme pixel reports: native signed WHT, WHT divided by
    a bounded *local* robust positive-WHT reference, WHT divided by a
    SUP_W1-derived reference, N_eff, and the distance to the physical support
    boundary.  Purely observational — no rule is chosen or applied.
    """
    out = {
        "tile": int(tile),
        "local_reference_statistic": "positive_p90",
        "sup_w1_reference": None,
        "extrema": [],
        "reason": None,
    }
    try:
        s = np.asarray(sci, dtype=np.float64)
        w = np.asarray(wht, dtype=np.float64)
        if s.shape != w.shape:
            out["reason"] = "shape_mismatch"
            return out
        h, wid = s.shape
        mask = _valid_support_mask(s, w, WEIGHT_EPSILON)
        local_ref, global_ref, meta = _local_positive_reference(w, tile)
        out["global_positive_reference"] = _f(global_ref)
        out["local_reference_tiles_hw"] = meta["tiles_hw"]

        neff = None
        w1_ref = None
        if w1 is not None and w2 is not None:
            neff = _neff_from_sup(w1, w2)
            a = np.asarray(w1, dtype=np.float64)
            pos = a[np.isfinite(a) & (a > 0.0)]
            if pos.size:
                w1_ref = float(np.median(pos))
        out["sup_w1_reference"] = _f(w1_ref)

        lo, hi = select_extrema_indices(s, mask, n_extrema)
        flat_s = s.ravel()
        flat_w = w.ravel()
        flat_neff = None if neff is None else neff.ravel()
        flat_w1 = None if w1 is None else np.asarray(w1, dtype=np.float64).ravel()
        flat_dist = None if distance is None else np.asarray(distance, dtype=np.float64).ravel()
        for fi in list(lo) + list(hi):
            fi = int(fi)
            row, col = divmod(fi, wid)
            ti, tj = _tile_index(row, col, tile)
            local = float(local_ref[ti, tj]) if local_ref.size else global_ref
            rec = {
                "index": fi,
                "row": int(row),
                "col": int(col),
                "sci": _f(flat_s[fi]),
                "wht": _f(flat_w[fi]),
                "wht_over_local_ref": (
                    _f(flat_w[fi] / local) if local not in (0.0, None) else None
                ),
                "wht_over_global_ref": (
                    _f(flat_w[fi] / global_ref) if global_ref else None
                ),
                "wht_over_sup_w1_ref": (
                    _f(flat_w[fi] / w1_ref)
                    if w1_ref not in (None, 0.0)
                    else None
                ),
                "n_eff": _f(flat_neff[fi]) if flat_neff is not None else None,
                "sup_w1": _f(flat_w1[fi]) if flat_w1 is not None else None,
                "distance": _f(flat_dist[fi]) if flat_dist is not None else None,
            }
            out["extrema"].append(rec)
    except Exception as exc:  # noqa: BLE001
        logger.debug("conditioning_candidates failed (non-fatal): %s", exc)
        out["reason"] = "exception"
    return out


# ---------------------------------------------------------------------------
# spatial boundary diagnostics
# ---------------------------------------------------------------------------


def _support_distance_map(support_mask):
    """Transient EDT distance-to-boundary of a support mask (or ``None``)."""
    try:
        from scipy.ndimage import distance_transform_edt

        m = np.asarray(support_mask, dtype=bool)
        if not np.any(m):
            return None
        return distance_transform_edt(m).astype(np.float64)
    except Exception as exc:  # noqa: BLE001
        logger.debug("support distance map unavailable (non-fatal): %s", exc)
        return None


def spatial_boundary_diagnostics(sci, wht, support_mask, neff=None,
                                 distance=None):
    """Per-distance-bin summaries over the actual physical support map.

    The distance map is computed transiently (never persisted) unless one is
    supplied.  For each documented bin it reports SCI extrema, the fraction of
    small-positive WHT, the number/fraction of extreme pixels and N_eff stats.
    """
    out = {"available": False, "reason": None, "bins": [],
           "small_positive_wht_cut": _f(SMALL_POSITIVE_WHT)}
    try:
        s = np.asarray(sci, dtype=np.float64)
        w = np.asarray(wht, dtype=np.float64)
        m = np.asarray(support_mask, dtype=bool)
        if s.shape != w.shape or s.shape != m.shape:
            out["reason"] = "shape_mismatch"
            return out
        if distance is None:
            distance = _support_distance_map(m)
        if distance is None:
            out["reason"] = "distance_map_unavailable"
            return out
        d = np.asarray(distance, dtype=np.float64)
        if d.shape != s.shape:
            out["reason"] = "distance_shape_mismatch"
            return out
        total_support = int(np.count_nonzero(m))
        out["total_support_pixels"] = total_support
        lo, hi = select_extrema_indices(s, m, EXTREMA_COUNT)
        extreme_idx = set(int(x) for x in list(lo) + list(hi))
        small_positive = np.isfinite(w) & (w > 0.0) & (w <= SMALL_POSITIVE_WHT)

        for label, (lo_e, hi_e) in zip(DISTANCE_BIN_LABELS, DISTANCE_BIN_EDGES):
            if math.isinf(hi_e):
                bin_mask = m & (d > lo_e)
            elif lo_e <= 0.0:
                bin_mask = m & (d >= 0.0) & (d <= hi_e)
            else:
                bin_mask = m & (d > lo_e) & (d <= hi_e)
            n_bin = int(np.count_nonzero(bin_mask))
            rec = {
                "label": label,
                "pixels": n_bin,
                "fraction_of_support": (
                    _f(n_bin / total_support) if total_support else None
                ),
                "sci_min": None,
                "sci_max": None,
                "small_positive_wht_fraction": None,
                "extreme_pixels": 0,
                "extreme_fraction": None,
                "n_eff_min": None,
                "n_eff_median": None,
                "n_eff_max": None,
            }
            if n_bin:
                sv = s[bin_mask]
                sv = sv[np.isfinite(sv)]
                if sv.size:
                    rec["sci_min"] = _f(np.min(sv))
                    rec["sci_max"] = _f(np.max(sv))
                n_small = int(np.count_nonzero(small_positive & bin_mask))
                rec["small_positive_wht_fraction"] = _f(n_small / n_bin)
                flat_idx = np.flatnonzero(bin_mask.ravel())
                n_ext = int(sum(1 for i in flat_idx.tolist() if i in extreme_idx))
                rec["extreme_pixels"] = n_ext
                rec["extreme_fraction"] = _f(n_ext / n_bin)
                if neff is not None:
                    nv = np.asarray(neff, dtype=np.float64)[bin_mask]
                    nv = nv[np.isfinite(nv)]
                    if nv.size:
                        rec["n_eff_min"] = _f(np.min(nv))
                        rec["n_eff_median"] = _f(np.median(nv))
                        rec["n_eff_max"] = _f(np.max(nv))
            out["bins"].append(rec)
        out["available"] = True
    except Exception as exc:  # noqa: BLE001
        logger.debug("spatial_boundary_diagnostics failed (non-fatal): %s", exc)
        out["reason"] = "exception"
    return out


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def lifecycle_record(stage, obj, **extra):
    """Build one bounded SUPPORT_LIFECYCLE record from a queue-manager object.

    Captures the support presence/availability *as observed at the seam*, the
    stopped/finalization state, and any caller-supplied scalar context.  The
    classic memmap flags and the Drizzle accumulator flags are reported
    separately so an early release/reset is distinguishable from stale logging.
    """
    rec = {
        "stage": str(stage),
        "ts": time.time(),
        "support_available": bool(getattr(obj, "_drizzle_support_available", False)),
        "support_reason": getattr(
            obj, "_drizzle_support_unavailable_reason", None
        ),
        "classic_sup_w1_present": getattr(obj, "coverage_sup_w1_memmap", None)
        is not None,
        "classic_sup_w2_present": getattr(obj, "coverage_sup_w2_memmap", None)
        is not None,
        "classic_support_state_available": bool(
            getattr(obj, "_support_state_available", False)
        ),
        "drizzle_sup_w1_present": getattr(obj, "drizzle_sup_w1", None) is not None,
        "drizzle_sup_w2_present": getattr(obj, "drizzle_sup_w2", None) is not None,
        "stopped": bool(
            getattr(obj, "user_requested_stop", False)
            or getattr(obj, "stop_processing", False) is True
        ),
        "finalization_mode": getattr(obj, "finalization_mode", None),
        "coverage_render_status": getattr(obj, "coverage_render_status", None),
        "frame_count": _i(getattr(obj, "_drizzle_frame_count", None)),
    }
    for key, value in extra.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            rec[key] = _f(value) if isinstance(value, float) else value
        else:
            rec[key] = str(value)
    return rec


# ---------------------------------------------------------------------------
# high-level finalization summary
# ---------------------------------------------------------------------------


def _channel_mean(arr):
    """Bounded derived 2-D channel-mean view of an HWC/2-D array."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        with np.errstate(invalid="ignore"):
            return np.mean(a, axis=-1)
    raise ValueError("unsupported ndim for channel mean")


def summarize_run(sci_hwc, wht_hwc, sup_w1=None, sup_w2=None, distance=None):
    """Compute every bounded per-run diagnostic section (read-only, fail-open).

    The HWC science and native WHT are summarised per channel; the derived
    2-D boundary/conditioning sections use the documented channel-mean view of
    SCI and WHT and the channel-invariant physical support map (SUP_W1 > 0
    when support is present, otherwise any-channel valid native WHT).  No input
    array is mutated; the distance map is computed transiently unless supplied.
    """
    sections = {
        "sci_stats": [],
        "wht_diagnostics": [],
        "threshold_sweep": [],
        "support": {"available": False, "reason": "unavailable"},
        "support_extrema": [],
        "boundary_bins": {"available": False, "reason": "unavailable"},
        "conditioning_candidates": {"extrema": [], "reason": "unavailable"},
        "derived_view": "channel_mean_2d",
    }
    try:
        s = np.asarray(sci_hwc, dtype=np.float64)
        w = np.asarray(wht_hwc, dtype=np.float64)
        if s.ndim == 2:
            s = s[..., None]
        if w.ndim == 2:
            w = w[..., None]
        if s.shape != w.shape:
            sections["notes"] = "sci/wht shape mismatch"
            return sections

        sections["sci_stats"] = [
            sci_channel_stats(s[..., c]) for c in range(s.shape[-1])
        ]
        sections["wht_diagnostics"] = [
            wht_channel_diagnostics(w[..., c], s[..., c])
            for c in range(s.shape[-1])
        ]
        sections["threshold_sweep"] = [
            threshold_sweep(w[..., c], s[..., c]) for c in range(s.shape[-1])
        ]

        neff = None
        support_mask = None
        if sup_w1 is not None and sup_w2 is not None:
            neff = _neff_from_sup(sup_w1, sup_w2)
            try:
                a1 = np.asarray(sup_w1, dtype=np.float64)
                support_mask = np.isfinite(a1) & (a1 > 0.0)
            except Exception:  # noqa: BLE001
                support_mask = None
        if support_mask is None or not np.any(support_mask):
            wm = np.asarray(w, dtype=np.float64)
            support_mask = np.any(np.isfinite(wm) & (wm > WEIGHT_EPSILON), axis=-1)

        sci2d = _channel_mean(s)
        wht2d = _channel_mean(w)

        if neff is not None:
            sections["support"] = support_conditioning(sup_w1, sup_w2)
        else:
            sections["support"] = {
                "available": False,
                "reason": "support_accumulator_absent",
            }

        dist = distance
        if dist is None:
            dist = _support_distance_map(support_mask)

        sections["support_extrema"] = sci_extrema_records(
            sci2d, wht2d, w1=sup_w1, w2=sup_w2, neff=neff, distance=dist
        )
        sections["boundary_bins"] = spatial_boundary_diagnostics(
            sci2d, wht2d, support_mask, neff=neff, distance=dist
        )
        cand = conditioning_candidates(
            sci2d, wht2d, w1=sup_w1, w2=sup_w2, distance=dist
        )
        # attach the per-extreme N_eff/WHT/SUP context already captured
        cand["support_extrema"] = sections["support_extrema"]
        sections["conditioning_candidates"] = cand
    except Exception as exc:  # noqa: BLE001
        logger.debug("summarize_run failed (non-fatal): %s", exc)
    return sections


# ---------------------------------------------------------------------------
# collector / artifact writer
# ---------------------------------------------------------------------------


class DrizzleScienceDiagnostics:
    """Bounded, passive per-run collector + atomic artifact writer.

    Every setter and the writer are fail-open: an internal error is logged at
    debug level and leaves the collector usable and the science untouched.
    Only one artifact is produced (``drizzle_science_diagnostics.json``) and it
    is written atomically (temp file + ``os.replace``) so a reader never sees a
    partial file.
    """

    def __init__(self, run_token=None, output_folder=None):
        self.run_token = None if run_token is None else str(run_token)
        self.output_folder = None if output_folder is None else str(output_folder)
        self.created_ts = time.time()
        self.kernel = None
        self.scale = None
        self.pixfrac_requested = None
        self.pixfrac_effective = None
        self.iscale_current = 1.0
        self.pixel_scale_ratio_current = 1.0
        self.pixel_scale_ratio_candidate = None
        self.geometry = None
        self.contract = None
        self.sci_stats = None
        self.wht_diagnostics = None
        self.threshold_sweep = None
        self.support = None
        self.boundary_bins = None
        self.conditioning = None
        self.lifecycle = []
        self.notes = []
        self.stop_state = {}
        self.finalization_state = {}
        self.artifact_path = None
        self.write_count = 0

    # -- setters (all fail-open) -------------------------------------------
    def _safe(self, fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — fail-open is the contract
            logger.debug("drizzle diagnostics setter failed (non-fatal): %s", exc)

    def set_run_config(self, kernel=None, scale=None, pixfrac_requested=None,
                       pixfrac_effective=None):
        def _do():
            if kernel is not None:
                self.kernel = str(kernel)
            self.scale = _f(scale)
            self.pixfrac_requested = _f(pixfrac_requested)
            self.pixfrac_effective = _f(pixfrac_effective)
        self._safe(_do)

    def set_geometry(self, record):
        self._safe(lambda: setattr(self, "geometry", dict(record)))

    def set_contract(self, record):
        self._safe(lambda: setattr(self, "contract", dict(record)))

    def set_sci_stats(self, per_channel):
        self._safe(lambda: setattr(self, "sci_stats", list(per_channel)))

    def set_wht_diagnostics(self, per_channel):
        self._safe(lambda: setattr(self, "wht_diagnostics", list(per_channel)))

    def set_threshold_sweep(self, sweep):
        self._safe(lambda: setattr(self, "threshold_sweep", list(sweep)))

    def set_support(self, conditioning, extrema=None):
        def _do():
            payload = dict(conditioning)
            if extrema is not None:
                payload["extrema"] = list(extrema)
            self.support = payload
        self._safe(_do)

    def set_boundary_bins(self, record):
        self._safe(lambda: setattr(self, "boundary_bins", dict(record)))

    def set_conditioning(self, record):
        self._safe(lambda: setattr(self, "conditioning", dict(record)))

    def add_lifecycle(self, record):
        def _do():
            if len(self.lifecycle) < 4096:
                self.lifecycle.append(dict(record))
        self._safe(_do)

    def note(self, message):
        def _do():
            if len(self.notes) < 64:
                self.notes.append(str(message))
        self._safe(_do)

    def set_stop_state(self, **fields):
        def _do():
            self.stop_state.update({k: _f(v) if isinstance(v, float) else v
                                    for k, v in fields.items()})
        self._safe(_do)

    def set_finalization_state(self, **fields):
        def _do():
            self.finalization_state.update(
                {k: _f(v) if isinstance(v, float) else v for k, v in fields.items()}
            )
        self._safe(_do)

    # -- serialization -----------------------------------------------------
    def lifecycle_summary(self):
        """Bounded summary of the lifecycle ring (stage counts + last stage)."""
        summary = {"count": len(self.lifecycle), "stage_counts": {},
                   "last_stage": None, "first_support_available": None,
                   "last_support_available": None}
        try:
            for rec in self.lifecycle:
                stage = rec.get("stage")
                summary["stage_counts"][stage] = (
                    summary["stage_counts"].get(stage, 0) + 1
                )
                if summary["first_support_available"] is None:
                    summary["first_support_available"] = rec.get(
                        "support_available"
                    )
                summary["last_support_available"] = rec.get("support_available")
            if self.lifecycle:
                summary["last_stage"] = self.lifecycle[-1].get("stage")
        except Exception as exc:  # noqa: BLE001
            logger.debug("lifecycle_summary failed (non-fatal): %s", exc)
        return summary

    def to_dict(self):
        """Return the bounded, JSON-safe artifact payload (never raises)."""
        try:
            geometry = self.geometry or {}
            contract = self.contract or {}
            return {
                "schema_version": SCHEMA_VERSION,
                "run_token": self.run_token,
                "created_ts": _f(self.created_ts),
                "generated_ts": _f(time.time()),
                "kernel": self.kernel,
                "scale": _f(self.scale),
                "pixfrac_requested": _f(self.pixfrac_requested),
                "pixfrac_effective": _f(self.pixfrac_effective),
                "iscale_current": _f(self.iscale_current),
                "pixel_scale_ratio_current": _f(self.pixel_scale_ratio_current),
                "pixel_scale_ratio_candidate": _f(
                    self.pixel_scale_ratio_candidate
                    if self.pixel_scale_ratio_candidate is not None
                    else geometry.get("pixel_scale_ratio_candidate")
                ),
                "pixel_scale_ratio_source": "upstream_default",
                "geometry": geometry,
                "add_image_contract": contract,
                "sci_stats": self.sci_stats,
                "wht_diagnostics": self.wht_diagnostics,
                "threshold_sweep": self.threshold_sweep,
                "support": self.support,
                "boundary_bins": self.boundary_bins,
                "conditioning_candidates": self.conditioning,
                "lifecycle": list(self.lifecycle),
                "lifecycle_summary": self.lifecycle_summary(),
                "stop_state": dict(self.stop_state),
                "finalization_state": dict(self.finalization_state),
                "notes": list(self.notes),
                "diagnostic_only": True,
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("to_dict failed (non-fatal): %s", exc)
            return {"schema_version": SCHEMA_VERSION, "diagnostic_only": True,
                    "error": "to_dict_failed"}

    def _bounded_payload(self):
        try:
            return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)
        except Exception as exc:  # noqa: BLE001
            logger.debug("diagnostics serialization failed (non-fatal): %s", exc)
            return None

    def write(self, path=None):
        """Atomically write the artifact.  Fail-open: returns ``False``.

        A temporary file in the destination directory is written then
        ``os.replace``d over the target, so a reader never observes a partial
        artifact.  Any failure (serialization, permissions, missing directory)
        is swallowed.
        """
        try:
            target = path or self.artifact_path
            if target is None and self.output_folder:
                target = os.path.join(self.output_folder, ARTIFACT_FILENAME)
            if not target:
                return False
            payload = self._bounded_payload()
            if payload is None:
                return False
            directory = os.path.dirname(os.fspath(target))
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp = f"{target}.tmp.{os.getpid()}.{int(time.time_ns())}"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.write("\n")
            os.replace(tmp, target)
            self.artifact_path = target
            self.write_count += 1
            return True
        except Exception as exc:  # noqa: BLE001 — fail-open is the contract
            logger.debug("drizzle diagnostics write failed (non-fatal): %s", exc)
            return False


__all__ = [
    "SCHEMA_VERSION",
    "ARTIFACT_FILENAME",
    "WEIGHT_EPSILON",
    "SMALL_POSITIVE_WHT",
    "DEFAULT_ABS_THRESHOLDS",
    "DEFAULT_REL_THRESHOLDS",
    "DISTANCE_BIN_LABELS",
    "DISTANCE_BIN_EDGES",
    "DrizzleScienceDiagnostics",
    "robust_pixel_scale_deg",
    "geometry_diagnostic",
    "contract_diagnostic",
    "sci_channel_stats",
    "sci_stats_hwc",
    "wht_channel_diagnostics",
    "wht_diagnostics_hwc",
    "select_extrema_indices",
    "sci_extrema_records",
    "threshold_sweep",
    "support_conditioning",
    "conditioning_candidates",
    "spatial_boundary_diagnostics",
    "lifecycle_record",
    "summarize_run",
]
