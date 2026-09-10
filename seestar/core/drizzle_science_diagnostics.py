"""Passive, fail-open, memory-bounded Drizzle science diagnostics (ZSSS-DRIZZLE-CLOSURE-P1).

This module is **instrumentation only**.  It observes the *current* production
Standard-M3 Drizzle science without ever mutating it, influencing it, or
changing control flow.  Its sole output is one bounded, versioned JSON artifact
per run written atomically/best-effort into the run output area.

Hard contract (Phase 1 scientific freeze)
-----------------------------------------
* **Passive** — every helper reads its inputs; none of them mutate the arrays
  they are given (all reductions allocate their own temporaries).
* **Fail-open** — every public entry point catches ``Exception`` internally.  A
  calculation failure or an I/O failure only warns (debug log) and returns a
  falsy/neutral value; it can never abort a run or change a scientific array.
* **Memory-bounded** — the summary never converts a whole HWC SCI/WHT cube to
  float64.  It streams **row chunks of float32 views**, keeps only bounded
  sample/candidate buffers, and holds **at most one** justified O(HW) boundary
  work buffer (the EDT distance map) with an explicit byte budget.  Percentiles
  use a deterministic, documented bounded sample; extrema use bounded per-chunk
  candidate merges (no full-array argsort).  See the constants below.
* **No science rule** — the module *reports* threshold/N_eff/boundary/
  conditioning candidate data for human/architect comparison.  It never applies
  a floor, mask, clip or rejection, and never declares negative science invalid.

Documented methods
------------------
* Angular pixel scale: mean of ``abs(proj_plane_pixel_scales(wcs))`` in
  degrees/pixel (fallback ``sqrt(|det(CD)|)``).  The geometry candidate is
  ``output_scale / input_scale`` and is diagnostic-only.
* Percentiles: deterministic uniform-stride sample over the finite population
  (stride ``max(1, total // MAX_SAMPLE_COUNT)``), capped at
  :data:`MAX_SAMPLE_COUNT` values; method/stride/count are recorded in the
  artifact.  Min/max and all counts/fractions are exact (chunked reductions).
* Extrema: exact absolute min/max plus the ``k`` smallest/largest SCI values over
  the physical support, selected with bounded per-chunk ``argpartition``
  candidate merges (tie-break by flat index).  Coordinates address the final
  (post-crop) SCI grid.
* Distance bins: EDT of the physical support mask with the array exterior
  treated as a support boundary (false-padded then cropped); bins are the
  half-open intervals ``[0,1]``, ``(1,4]``, ``(4,8]``, ``(8,16]``, ``(16,inf)``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.1"
ARTIFACT_FILENAME = "drizzle_science_diagnostics.json"

# Neutral support floor used by the current production contract (diagnostic
# constant; the science module stays the single source of truth).
WEIGHT_EPSILON = 1e-9

# Diagnostic "small positive WHT" cut (descriptive only; never applied).
SMALL_POSITIVE_WHT = 1e-4

# Deterministic threshold-sweep candidates (diagnostic only).
DEFAULT_ABS_THRESHOLDS = (1e-4, 1e-3, 1e-2, 1e-1)
DEFAULT_REL_THRESHOLDS = (1e-3, 1e-2, 0.05, 0.1)

# Bounded extrema sample size (per channel / per selection side).
EXTREMA_COUNT = 8

# Streaming chunk height (rows) for every full-array reduction.
ROW_CHUNK = 64

# Hard cap on the deterministic percentile sample (values per channel).
MAX_SAMPLE_COUNT = 200_000

# Hard cap on the single justified O(HW) boundary work buffer (EDT distance
# map, float64).  Larger grids skip the boundary section fail-open.
MAX_BOUNDARY_WORK_BYTES = 512 * 1024 * 1024

# Local positive-WHT reference tile edge (output pixels) — bounded tile map.
LOCAL_REFERENCE_TILE = 64

# Documented boundary distance bins (distance in output pixels).
DISTANCE_BIN_LABELS = ("0-1", "2-4", "5-8", "9-16", ">16")
DISTANCE_BIN_EDGES = (
    (0.0, 1.0),
    (1.0, 4.0),
    (4.0, 8.0),
    (8.0, 16.0),
    (16.0, math.inf),
)

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

    Method (documented, deterministic): mean of the absolute projected linear
    pixel scales at the WCS reference point
    (``astropy.wcs.utils.proj_plane_pixel_scales``), fallback
    ``sqrt(|det(CD)|)``.  Never raises.
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
            val = math.sqrt(abs(float(np.linalg.det(cd))))
            if not math.isfinite(val) or val <= 0.0:
                return None, "wcs_det_nonpositive"
            return val, "sqrt_abs_det_cd"
        except Exception:  # noqa: BLE001
            return None, "wcs_unavailable"


def geometry_diagnostic(reference_wcs, output_wcs, kernel=None, scale=None):
    """Resolved-geometry record (fail-open).  Candidate = output/input scale."""
    in_scale, in_method = robust_pixel_scale_deg(reference_wcs)
    out_scale, out_method = robust_pixel_scale_deg(output_wcs)
    rec = {
        "available": False,
        "reason": None,
        "method": None,
        "input_pixel_scale_deg": _f(in_scale),
        "output_pixel_scale_deg": _f(out_scale),
        "pixel_scale_ratio_current": 1.0,
        "pixel_scale_ratio_current_source": "upstream_add_image_default",
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
    """Effective ``Drizzle.add_image`` call semantics with precise provenance.

    Provenance semantics (exact distinction):

    * ``kernel`` / ``fillval`` are resolved at **Drizzle construction** (the
      accumulator's constructor / wrapper default) -> ``drizzle_construction_*``;
    * ``iscale`` / ``pixel_scale_ratio`` are **upstream ``add_image`` defaults**
      (the wrapper does not pass them) -> ``upstream_add_image_default``.
    """
    try:
        in_units = str(in_units)
    except Exception:  # noqa: BLE001
        in_units = "counts"
    expscale = exptime if in_units == "counts" else 1.0
    return {
        "kernel_effective": None if kernel is None else str(kernel),
        "kernel_source": "drizzle_construction_explicit",
        "pixfrac_effective": _f(pixfrac),
        "pixfrac_source": "drizzle_construction_explicit",
        "iscale_effective": 1.0 if iscale is None else _f(iscale),
        "iscale_source": (
            "upstream_add_image_default" if iscale is None else "explicit"
        ),
        "pixel_scale_ratio_effective": (
            1.0 if pixel_scale_ratio is None else _f(pixel_scale_ratio)
        ),
        "pixel_scale_ratio_source": (
            "upstream_add_image_default"
            if pixel_scale_ratio is None
            else "explicit"
        ),
        "in_units_effective": in_units,
        "in_units_source": "explicit",
        "exptime_effective": _f(exptime),
        "exptime_source": "explicit",
        "wht_scale_effective": _f(expscale),
        "wht_scale_source": "explicit",
        "fillval_effective": None if fillval is None else str(fillval),
        "fillval_source": "drizzle_construction_default",
        "weight_map_present": True,
        "weight_map_source": "explicit",
    }


# ---------------------------------------------------------------------------
# streaming helpers (bounded memory)
# ---------------------------------------------------------------------------


def _iter_row_chunks(height, width, chunk_rows=ROW_CHUNK):
    """Yield ``(row0, row1)`` half-open row ranges."""
    step = max(1, int(chunk_rows))
    for r0 in range(0, int(height), step):
        yield r0, min(int(height), r0 + step)


def _sample_stride(total, cap=MAX_SAMPLE_COUNT):
    return max(1, int(total) // max(1, int(cap)))


def _stride_select(flat_chunk, base_index, stride):
    """Deterministic uniform-stride sample of a chunk (bounded, finite only)."""
    off = (-int(base_index)) % int(stride)
    sel = flat_chunk[off::stride]
    if sel.size:
        sel = sel[np.isfinite(sel)]
    return sel


def _pct(sample, p):
    if sample is None or sample.size == 0:
        return None
    try:
        return _f(np.percentile(sample, p))
    except Exception:  # noqa: BLE001
        return None


def _merge_low(cur_v, cur_i, new_v, new_i, k):
    """Return the ``k`` smallest ``(value, index)`` from bounded candidates."""
    if new_v.size == 0:
        return cur_v, cur_i
    v = np.concatenate([cur_v, new_v]) if cur_v.size else new_v
    i = np.concatenate([cur_i, new_i]) if cur_i.size else new_i
    if v.size <= k:
        order = sorted(range(v.size), key=lambda x: (v[x], int(i[x])))
        order = np.asarray(order, dtype=np.int64)
        return v[order], i[order]
    part = np.argpartition(v, k - 1)[:k]
    cand = sorted(part.tolist(), key=lambda x: (v[x], int(i[x])))
    cand = np.asarray(cand, dtype=np.int64)
    return v[cand], i[cand]


def _merge_high(cur_v, cur_i, new_v, new_i, k):
    """Return the ``k`` largest ``(value, index)`` from bounded candidates."""
    if new_v.size == 0:
        return cur_v, cur_i
    v = np.concatenate([cur_v, new_v]) if cur_v.size else new_v
    i = np.concatenate([cur_i, new_i]) if cur_i.size else new_i
    if v.size <= k:
        order = sorted(range(v.size), key=lambda x: (-v[x], int(i[x])))
        order = np.asarray(order, dtype=np.int64)
        return v[order], i[order]
    part = np.argpartition(-v, k - 1)[:k]
    cand = sorted(part.tolist(), key=lambda x: (-v[x], int(i[x])))
    cand = np.asarray(cand, dtype=np.int64)
    return v[cand], i[cand]


def bounded_extrema_indices(values_2d, valid_mask, n, chunk_rows=ROW_CHUNK):
    """Bounded low/high flat-index selection via per-chunk argpartition.

    Streams row chunks (no full-array float64 copy, no full argsort) and merges
    a bounded candidate buffer; ties break by flat index.  Returns ``(lo, hi)``
    int64 arrays.  Fail-open: empty arrays on error.
    """
    try:
        v = np.asarray(values_2d, dtype=np.float32)
        m = np.asarray(valid_mask, dtype=bool)
        if v.shape != m.shape:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        h, wid = v.shape
        k = max(1, int(n))
        lo_v = np.empty(0); lo_i = np.empty(0, dtype=np.int64)
        hi_v = np.empty(0); hi_i = np.empty(0, dtype=np.int64)
        for r0, r1 in _iter_row_chunks(h, wid, chunk_rows):
            vc = v[r0:r1]
            mc = m[r0:r1]
            valid = mc & np.isfinite(vc)
            if not np.any(valid):
                continue
            rr, cc = np.nonzero(valid)
            gi = (r0 + rr).astype(np.int64) * wid + cc.astype(np.int64)
            vv = vc[valid].astype(np.float64)
            if vv.size > k:
                lp = np.argpartition(vv, k - 1)[:k]
                hp = np.argpartition(-vv, k - 1)[:k]
            else:
                lp = np.arange(vv.size)
                hp = np.arange(vv.size)
            lo_v, lo_i = _merge_low(lo_v, lo_i, vv[lp], gi[lp], k)
            hi_v, hi_i = _merge_high(hi_v, hi_i, vv[hp], gi[hp], k)
        return lo_i.astype(np.int64), hi_i.astype(np.int64)
    except Exception as exc:  # noqa: BLE001
        logger.debug("bounded_extrema_indices failed (non-fatal): %s", exc)
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)


def select_extrema_indices(values_2d, valid_mask, n):
    """Deterministic bounded high/low flat-index selection (returns ``(lo, hi)``).

    Uses ``argpartition`` candidate selection (never a full argsort); ties are
    broken by original flat order.  Bounded by ``n`` per side.  Never raises.
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
        vals = v[idx]
        lo_part = np.argpartition(vals, k - 1)[:k]
        hi_part = np.argpartition(-vals, k - 1)[:k]
        lo = sorted(lo_part.tolist(), key=lambda x: (vals[x], int(idx[x])))
        hi = sorted(hi_part.tolist(), key=lambda x: (-vals[x], int(idx[x])))
        return idx[np.asarray(lo, dtype=np.int64)], idx[np.asarray(hi, dtype=np.int64)]
    except Exception as exc:  # noqa: BLE001
        logger.debug("select_extrema_indices failed (non-fatal): %s", exc)
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)


# ---------------------------------------------------------------------------
# physical support
# ---------------------------------------------------------------------------


def physical_support_mask(sup_w1, shape_hw):
    """Return ``(mask, source)`` for the physical support.

    ``(SUP_W1 > 0) & finite`` is the physical support whenever the pair exists
    (``source="sup_w1_positive"``).  Absence of the pair is labelled explicitly
    (never silently called physical support).
    """
    if sup_w1 is not None:
        try:
            a = np.asarray(sup_w1, dtype=np.float32)
            if a.shape == tuple(shape_hw):
                return (np.isfinite(a) & (a > 0.0)), "sup_w1_positive"
        except Exception:  # noqa: BLE001
            pass
    return None, "support_pair_absent"


def native_wht_fallback_mask(wht_hwc):
    """Explicitly-labelled fallback support: any-channel finite native WHT > eps."""
    a = np.asarray(wht_hwc, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    mask = np.zeros(a.shape[:2], dtype=bool)
    for c in range(a.shape[-1]):
        w = a[..., c]
        mask |= np.isfinite(w) & (w > WEIGHT_EPSILON)
    return mask


def support_distance_map(support_mask, max_bytes=MAX_BOUNDARY_WORK_BYTES):
    """Transient EDT distance-to-boundary with array edges as physical boundaries.

    Returns ``(distance_float64, meta)``.  The mask is false-padded by one pixel
    before the transform and cropped afterwards, so the output exterior and
    fully-covered grids yield a symmetric, meaningful field.  ``meta`` records
    the dtype/bytes budget and the reason when skipped (never raises).
    """
    meta = {
        "available": False,
        "reason": None,
        "dtype": "float32",
        "bytes": 0,
        "transient_float64_bytes": 0,
        "edge_padding": True,
        "max_bytes": int(max_bytes),
    }
    try:
        from scipy.ndimage import distance_transform_edt

        m = np.asarray(support_mask, dtype=bool)
        if m.ndim != 2:
            meta["reason"] = "mask_not_2d"
            return None, meta
        if not np.any(m):
            meta["reason"] = "empty_support"
            return None, meta
        h, w = m.shape
        transient = int(h + 2) * int(w + 2) * 8
        meta["transient_float64_bytes"] = transient
        if transient > int(max_bytes):
            meta["bytes"] = transient
            meta["reason"] = "boundary_work_buffer_over_budget"
            return None, meta
        padded = np.zeros((h + 2, w + 2), dtype=bool)
        padded[1:-1, 1:-1] = m
        dist = distance_transform_edt(padded)
        # Immediately materialise the resident buffer as float32 (half the
        # bytes) and release the float64 transform + padded mask so at most one
        # O(HW) boundary work buffer is ever resident.
        out = np.ascontiguousarray(dist[1:-1, 1:-1], dtype=np.float32)
        meta["bytes"] = int(out.nbytes)
        del dist, padded
        meta["available"] = True
        return out, meta
    except Exception as exc:  # noqa: BLE001
        logger.debug("support distance map unavailable (non-fatal): %s", exc)
        meta["reason"] = "exception"
        return None, meta


# ---------------------------------------------------------------------------
# chunked per-channel statistics
# ---------------------------------------------------------------------------


def _chunked_sci_stats(sci_2d, chunk_rows=ROW_CHUNK, sample_cap=MAX_SAMPLE_COUNT):
    """Exact min/max/counts/fractions + bounded-sample percentiles (float32 stream)."""
    out = {
        "count": 0, "finite_count": 0, "min": None, "p0_001": None,
        "p0_01": None, "median": None, "p99_99": None, "p99_999": None,
        "max": None, "fraction_lt_zero": None, "finite_fraction": None,
        "nonfinite_fraction": None, "sample_method": "deterministic_uniform_stride",
        "sample_stride": None, "sample_count": 0, "sample_cap": int(sample_cap),
    }
    try:
        a = np.asarray(sci_2d, dtype=np.float32)
        h, w = a.shape
        total = int(h * w)
        out["count"] = total
        if total == 0:
            return out
        stride = _sample_stride(total, sample_cap)
        out["sample_stride"] = int(stride)
        n_fin = 0
        n_neg = 0
        mn = math.inf
        mx = -math.inf
        samples = []
        for r0, r1 in _iter_row_chunks(h, w, chunk_rows):
            ch = a[r0:r1]
            fin = np.isfinite(ch)
            nf = int(np.count_nonzero(fin))
            n_fin += nf
            if nf == 0:
                continue
            vals = ch[fin]
            if vals.size:
                mn = min(mn, float(np.min(vals)))
                mx = max(mx, float(np.max(vals)))
                n_neg += int(np.count_nonzero(vals < 0.0))
            sel = _stride_select(ch.reshape(-1), r0 * w, stride)
            if sel.size:
                samples.append(sel.astype(np.float64, copy=True))
        out["finite_count"] = n_fin
        out["finite_fraction"] = _f(n_fin / total)
        out["nonfinite_fraction"] = _f((total - n_fin) / total)
        if n_fin:
            out["min"] = _f(mn)
            out["max"] = _f(mx)
            out["fraction_lt_zero"] = _f(n_neg / n_fin)
        sample = np.concatenate(samples) if samples else np.empty(0)
        if sample.size > sample_cap:
            sample = sample[:: max(1, sample.size // sample_cap)][:sample_cap]
        out["sample_count"] = int(sample.size)
        if sample.size:
            out["median"] = _pct(sample, 50.0)
            out["p0_001"] = _pct(sample, 0.001)
            out["p0_01"] = _pct(sample, 0.01)
            out["p99_99"] = _pct(sample, 99.99)
            out["p99_999"] = _pct(sample, 99.999)
    except Exception as exc:  # noqa: BLE001
        logger.debug("_chunked_sci_stats failed (non-fatal): %s", exc)
    return out


def sci_channel_stats(arr, chunk_rows=ROW_CHUNK, sample_cap=MAX_SAMPLE_COUNT):
    """Pre-stretch SCI statistics for one channel (chunked, read-only, fail-open)."""
    return _chunked_sci_stats(arr, chunk_rows, sample_cap)


def sci_stats_hwc(arr, chunk_rows=ROW_CHUNK, sample_cap=MAX_SAMPLE_COUNT):
    """Per-channel :func:`sci_channel_stats` (never converts the cube wholesale)."""
    try:
        a = np.asarray(arr)
        if a.ndim == 2:
            return [sci_channel_stats(a, chunk_rows, sample_cap)]
        if a.ndim == 3:
            return [
                sci_channel_stats(a[..., c], chunk_rows, sample_cap)
                for c in range(a.shape[-1])
            ]
        return []
    except Exception as exc:  # noqa: BLE001
        logger.debug("sci_stats_hwc failed (non-fatal): %s", exc)
        return []


def _chunked_wht_stats(wht_2d, chunk_rows=ROW_CHUNK, sample_cap=MAX_SAMPLE_COUNT):
    """Native signed-WHT stats (exact counts/min/max + sampled positive pcts)."""
    out = {
        "min": None, "max": None, "n_positive": None, "positive_percentiles": {},
        "bin_fractions": {},
        "sample_method": "deterministic_uniform_stride",
        "sample_count": 0, "sample_cap": int(sample_cap),
    }
    try:
        w = np.asarray(wht_2d, dtype=np.float32)
        h, wid = w.shape
        total = int(h * wid)
        if total == 0:
            return out
        stride = _sample_stride(total, sample_cap)
        mn = math.inf
        mx = -math.inf
        n_pos = 0
        bin_counts = {label: 0 for label, _ in WHT_MAGNITUDE_BINS}
        pos_samples = []
        for r0, r1 in _iter_row_chunks(h, wid, chunk_rows):
            ch = w[r0:r1]
            fin = np.isfinite(ch)
            if not np.any(fin):
                continue
            vals = ch[fin]
            mn = min(mn, float(np.min(vals)))
            mx = max(mx, float(np.max(vals)))
            n_pos += int(np.count_nonzero(vals > 0.0))
            for label, pred in WHT_MAGNITUDE_BINS:
                bin_counts[label] += int(np.count_nonzero(pred(vals)))
            flat = ch.reshape(-1)
            sel = _stride_select(flat, r0 * wid, stride)
            if sel.size:
                p = sel[sel > 0.0]
                if p.size:
                    pos_samples.append(p.astype(np.float64, copy=True))
        if math.isfinite(mn):
            out["min"] = _f(mn)
            out["max"] = _f(mx)
        out["n_positive"] = int(n_pos)
        for label in bin_counts:
            out["bin_fractions"][label] = _f(bin_counts[label] / total) if total else None
        pos_sample = np.concatenate(pos_samples) if pos_samples else np.empty(0)
        if pos_sample.size > sample_cap:
            pos_sample = pos_sample[:: max(1, pos_sample.size // sample_cap)][:sample_cap]
        out["sample_count"] = int(pos_sample.size)
        if pos_sample.size:
            out["positive_percentiles"] = {
                "p1": _pct(pos_sample, 1.0),
                "p50": _pct(pos_sample, 50.0),
                "p99": _pct(pos_sample, 99.0),
                "p99_9": _pct(pos_sample, 99.9),
                "max": _f(float(np.max(pos_sample))),
            }
    except Exception as exc:  # noqa: BLE001
        logger.debug("_chunked_wht_stats failed (non-fatal): %s", exc)
    return out


def wht_channel_diagnostics(wht, sci=None, n_extrema=EXTREMA_COUNT,
                            chunk_rows=ROW_CHUNK, sample_cap=MAX_SAMPLE_COUNT):
    """Native signed-WHT diagnostics for one channel (chunked, read-only)."""
    out = _chunked_wht_stats(wht, chunk_rows, sample_cap)
    out.setdefault("sci_min_wht", None)
    out.setdefault("sci_max_wht", None)
    out.setdefault("extrema", [])
    try:
        if sci is not None:
            s = np.asarray(sci)
            if s.shape == np.asarray(wht).shape:
                # bounded extrema over native-WHT-derived validity (diagnostics only)
                mask = np.isfinite(np.asarray(wht, dtype=np.float32))
                recs = bounded_extrema_records(
                    s, wht, mask, None, None, None, None, n=n_extrema,
                    chunk_rows=chunk_rows,
                )
                out["extrema"] = recs
                for r in recs:
                    if r.get("kind") == "sci_abs_min":
                        out["sci_min_wht"] = r.get("wht")
                    if r.get("kind") == "sci_abs_max":
                        out["sci_max_wht"] = r.get("wht")
    except Exception as exc:  # noqa: BLE001
        logger.debug("wht_channel_diagnostics extrema failed (non-fatal): %s", exc)
    return out


def wht_diagnostics_hwc(wht, sci=None, n_extrema=EXTREMA_COUNT,
                        chunk_rows=ROW_CHUNK, sample_cap=MAX_SAMPLE_COUNT):
    """Per-channel :func:`wht_channel_diagnostics` (never converts the cube)."""
    try:
        w = np.asarray(wht)
        s = None if sci is None else np.asarray(sci)
        if w.ndim == 2:
            return [wht_channel_diagnostics(w, s, n_extrema, chunk_rows, sample_cap)]
        if w.ndim == 3:
            res = []
            for c in range(w.shape[-1]):
                sc = None if s is None else s[..., c]
                res.append(
                    wht_channel_diagnostics(
                        w[..., c], sc, n_extrema, chunk_rows, sample_cap
                    )
                )
            return res
        return []
    except Exception as exc:  # noqa: BLE001
        logger.debug("wht_diagnostics_hwc failed (non-fatal): %s", exc)
        return []


# ---------------------------------------------------------------------------
# bounded extrema
# ---------------------------------------------------------------------------


def _flat_record(sci_2d, wht_2d, width, fi, w1=None, w2=None, neff=None,
                 distance=None, kind=None):
    def _at(arr, i):
        if arr is None:
            return None
        try:
            a = np.asarray(arr)
            if a.ndim == 2:
                return _f(a.ravel()[i])
            return _f(a[i])
        except Exception:  # noqa: BLE001
            return None

    return {
        "kind": kind,
        "index": int(fi),
        "row": int(fi) // int(width) if width else None,
        "col": int(fi) % int(width) if width else None,
        "sci": _f(np.asarray(sci_2d, dtype=np.float32).ravel()[fi]),
        "wht": _f(np.asarray(wht_2d, dtype=np.float32).ravel()[fi]),
        "sup_w1": _at(w1, fi),
        "sup_w2": _at(w2, fi),
        "n_eff": _at(neff, fi),
        "distance": _at(distance, fi),
    }


def bounded_extrema_records(sci_2d, wht_2d, support_mask, w1=None, w2=None,
                            neff=None, distance=None, n=EXTREMA_COUNT,
                            chunk_rows=ROW_CHUNK):
    """Exact abs min/max + bounded k lowest/highest SCI over physical support.

    Single chunked pass; per-chunk ``argpartition`` candidates merged with a
    bounded running buffer (no full argsort).  Coordinates are flat row-major
    on the (post-crop) supplied grid.  Fail-open: returns ``[]`` on error.
    """
    try:
        s = np.asarray(sci_2d, dtype=np.float32)
        w = np.asarray(wht_2d, dtype=np.float32)
        m = np.asarray(support_mask, dtype=bool)
        if s.shape != w.shape or s.shape != m.shape:
            return []
        h, wid = s.shape
        k = max(1, int(n))
        lo_v = np.empty(0, dtype=np.float64)
        lo_i = np.empty(0, dtype=np.int64)
        hi_v = np.empty(0, dtype=np.float64)
        hi_i = np.empty(0, dtype=np.int64)
        abs_min_i = None
        abs_max_i = None
        abs_min_v = math.inf
        abs_max_v = -math.inf
        for r0, r1 in _iter_row_chunks(h, wid, chunk_rows):
            sc = s[r0:r1]
            mc = m[r0:r1]
            valid = mc & np.isfinite(sc)
            if not np.any(valid):
                continue
            rr, cc = np.nonzero(valid)
            gi = (r0 + rr).astype(np.int64) * wid + cc.astype(np.int64)
            v = sc[valid].astype(np.float64)
            # absolute min/max (first occurrence by flat order)
            amin = int(np.argmin(v))
            amax = int(np.argmax(v))
            if v[amin] < abs_min_v:
                abs_min_v = float(v[amin])
                abs_min_i = int(gi[amin])
            if v[amax] > abs_max_v:
                abs_max_v = float(v[amax])
                abs_max_i = int(gi[amax])
            # bounded low/high candidates
            if v.size > k:
                lo_part = np.argpartition(v, k - 1)[:k]
                hi_part = np.argpartition(-v, k - 1)[:k]
            else:
                lo_part = np.arange(v.size)
                hi_part = np.arange(v.size)
            lo_v, lo_i = _merge_low(
                lo_v, lo_i, v[lo_part], gi[lo_part], k
            )
            hi_v, hi_i = _merge_high(
                hi_v, hi_i, v[hi_part], gi[hi_part], k
            )
        recs = []
        if abs_min_i is not None:
            recs.append(_flat_record(s, w, wid, abs_min_i, w1, w2, neff,
                                     distance, "sci_abs_min"))
        if abs_max_i is not None:
            recs.append(_flat_record(s, w, wid, abs_max_i, w1, w2, neff,
                                     distance, "sci_abs_max"))
        for fv, fi in zip(lo_v, lo_i):
            recs.append(_flat_record(s, w, wid, int(fi), w1, w2, neff,
                                     distance, "sci_robust_low"))
        for fv, fi in zip(hi_v, hi_i):
            recs.append(_flat_record(s, w, wid, int(fi), w1, w2, neff,
                                     distance, "sci_robust_high"))
        seen = set()
        uniq = []
        for r in recs:
            if r["index"] in seen:
                continue
            seen.add(r["index"])
            uniq.append(r)
        return uniq
    except Exception as exc:  # noqa: BLE001
        logger.debug("bounded_extrema_records failed (non-fatal): %s", exc)
        return []


def sci_extrema_records(sci, wht, n=EXTREMA_COUNT, w1=None, w2=None,
                        neff=None, distance=None, chunk_rows=ROW_CHUNK):
    """SCI extrema over native-WHT validity (compat helper; bounded)."""
    try:
        s = np.asarray(sci, dtype=np.float32)
        w = np.asarray(wht, dtype=np.float32)
        mask = np.isfinite(s) & np.isfinite(w) & (w > WEIGHT_EPSILON)
        return bounded_extrema_records(s, w, mask, w1, w2, neff, distance, n,
                                       chunk_rows)
    except Exception as exc:  # noqa: BLE001
        logger.debug("sci_extrema_records failed (non-fatal): %s", exc)
        return []


# ---------------------------------------------------------------------------
# support conditioning (N_eff over support only)
# ---------------------------------------------------------------------------


def support_conditioning(w1, w2, chunk_rows=ROW_CHUNK, sample_cap=MAX_SAMPLE_COUNT):
    """N_eff summaries computed **only** over valid physical support.

    Valid support = ``SUP_W1 > 0`` AND ``SUP_W2 > 0`` (finite).  Off-support and
    invalid fractions are reported separately; min/median/mean/max/percentiles
    are never diluted by forced-zero off-support pixels.
    """
    out = {
        "available": False, "reason": None, "support_pixels": None, "total_pixels": None,
        "n_eff_min": None, "n_eff_median": None, "n_eff_mean": None, "n_eff_max": None,
        "n_eff_p01": None, "n_eff_p99": None,
        "off_support_fraction": None, "invalid_support_fraction": None,
        "sample_method": "deterministic_uniform_stride", "sample_count": 0,
        "sample_cap": int(sample_cap),
    }
    try:
        if w1 is None or w2 is None:
            out["reason"] = "support_accumulator_absent"
            return out
        a = np.asarray(w1, dtype=np.float32)
        b = np.asarray(w2, dtype=np.float32)
        if a.shape != b.shape:
            out["reason"] = "support_shape_mismatch"
            return out
        h, wid = a.shape
        total = int(h * wid)
        out["total_pixels"] = total
        stride = _sample_stride(total, sample_cap)
        n_sup = 0
        n_off = 0
        n_invalid = 0
        mn = math.inf
        mx = -math.inf
        ssum = 0.0
        samples = []
        for r0, r1 in _iter_row_chunks(h, wid, chunk_rows):
            ac = a[r0:r1]
            bc = b[r0:r1]
            pos = np.isfinite(ac) & np.isfinite(bc) & (ac > 0.0)
            valid = pos & (bc > 0.0)
            n_sup += int(np.count_nonzero(valid))
            n_off += int(np.count_nonzero(~pos))
            n_invalid += int(np.count_nonzero(pos & ~valid))
            if np.any(valid):
                av = ac[valid].astype(np.float64)
                bv = bc[valid].astype(np.float64)
                r = av / np.sqrt(bv)
                ne = r * r
                mn = min(mn, float(np.min(ne)))
                mx = max(mx, float(np.max(ne)))
                ssum += float(np.sum(ne))
                # stride sample of support N_eff
                flat = ac.reshape(-1)
                base = r0 * wid
                off = (-base) % stride
                sel = flat[off::stride]
                if sel.size:
                    rb = bc.reshape(-1)[off::stride]
                    ok = np.isfinite(sel) & np.isfinite(rb) & (sel > 0.0) & (rb > 0.0)
                    if np.any(ok):
                        rr = sel[ok].astype(np.float64) / np.sqrt(
                            rb[ok].astype(np.float64)
                        )
                        samples.append(rr * rr)
        out["support_pixels"] = int(n_sup)
        if total:
            out["off_support_fraction"] = _f(n_off / total)
            out["invalid_support_fraction"] = _f(n_invalid / total)
        if n_sup:
            out["available"] = True
            out["n_eff_min"] = _f(mn)
            out["n_eff_max"] = _f(mx)
            out["n_eff_mean"] = _f(ssum / n_sup)
        sample = np.concatenate(samples) if samples else np.empty(0)
        if sample.size > sample_cap:
            sample = sample[:: max(1, sample.size // sample_cap)][:sample_cap]
        out["sample_count"] = int(sample.size)
        if sample.size:
            out["n_eff_median"] = _pct(sample, 50.0)
            out["n_eff_p01"] = _pct(sample, 1.0)
            out["n_eff_p99"] = _pct(sample, 99.0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("support_conditioning failed (non-fatal): %s", exc)
        out["reason"] = "exception"
    return out


# ---------------------------------------------------------------------------
# threshold sweep (diagnostic only; separations explicit)
# ---------------------------------------------------------------------------


def threshold_sweep(wht, sci, support_mask=None, positive_reference=None,
                    chunk_rows=ROW_CHUNK,
                    abs_candidates=DEFAULT_ABS_THRESHOLDS,
                    rel_candidates=DEFAULT_REL_THRESHOLDS,
                    support_source="physical"):
    """Report what an (unapplied) threshold *would* do, with explicit denominators.

    Populated regardless of feasibility; never applied to science.  The support
    denominator is labelled truthfully: ``support_source == "physical"`` reports
    ``physical_support_pixels``; a native-WHT-derived fallback sets
    ``physical_support_pixels=None`` and reports ``fallback_support_pixels`` /
    ``support_population_pixels`` instead, so positive native signed WHT is never
    silently presented as physical support.
    """
    is_physical = str(support_source) == "physical"
    out = {
        "current_epsilon": _f(WEIGHT_EPSILON),
        "support_source": str(support_source),
        "support_denominator_label": (
            "physical_support_pixels" if is_physical else "fallback_support_pixels"
        ),
        "physical_support_pixels": None,
        "fallback_support_pixels": None,
        "support_population_pixels": None,
        "currently_valid_positive_native_wht_pixels": None,
        "positive_reference": _f(positive_reference),
        "positive_reference_source": None,
        "candidates": [],
    }
    try:
        w = np.asarray(wht, dtype=np.float32)
        s = np.asarray(sci, dtype=np.float32)
        if w.shape != s.shape:
            return out
        if support_mask is None:
            out["note"] = "support population unavailable"
            return out
        m = np.asarray(support_mask, dtype=bool)
        h, wid = w.shape
        ref = positive_reference
        ref_source = "provided"
        if ref is None:
            sample, _stride = _chunked_positive_sample(w, chunk_rows)
            if sample.size:
                ref = float(np.percentile(sample, 99.0))
                ref_source = "positive_wht_p99_sampled"
        out["positive_reference"] = _f(ref)
        out["positive_reference_source"] = ref_source

        candidates = [("current_epsilon", float(WEIGHT_EPSILON))]
        for a in abs_candidates:
            candidates.append((f"abs_{a:g}", float(a)))
        if ref is not None and ref > 0.0:
            for r in rel_candidates:
                candidates.append((f"rel_{r:g}", float(r) * float(ref)))

        n_support = 0
        n_current = 0
        removed = {name: 0 for name, _ in candidates}
        rem_min = {name: math.inf for name, _ in candidates}
        rem_max = {name: -math.inf for name, _ in candidates}
        for r0, r1 in _iter_row_chunks(h, wid, chunk_rows):
            mc = m[r0:r1]
            wc = w[r0:r1]
            sc = s[r0:r1]
            n_support += int(np.count_nonzero(mc))
            cur = mc & np.isfinite(wc) & (wc > WEIGHT_EPSILON)
            n_current += int(np.count_nonzero(cur))
            for name, thr in candidates:
                keep = cur & (wc > thr)
                removed[name] += int(np.count_nonzero(cur)) - int(np.count_nonzero(keep))
                kv = sc[keep]
                if kv.size:
                    rem_min[name] = min(rem_min[name], float(np.min(kv)))
                    rem_max[name] = max(rem_max[name], float(np.max(kv)))
        out["currently_valid_positive_native_wht_pixels"] = int(n_current)
        if is_physical:
            out["physical_support_pixels"] = int(n_support)
        else:
            out["fallback_support_pixels"] = int(n_support)
            out["support_population_pixels"] = int(n_support)
        for name, thr in candidates:
            out["candidates"].append({
                "name": name,
                "threshold": _f(thr),
                "newly_removed_from_current_valid": int(removed[name]),
                "removed_fraction_of_physical_support": (
                    _f(removed[name] / n_support) if n_support else None
                ),
                "remaining_pixels": int(n_current - removed[name]),
                "remaining_sci_min": _f(rem_min[name])
                if math.isfinite(rem_min[name]) else None,
                "remaining_sci_max": _f(rem_max[name])
                if math.isfinite(rem_max[name]) else None,
            })
    except Exception as exc:  # noqa: BLE001
        logger.debug("threshold_sweep failed (non-fatal): %s", exc)
    return out


# ---------------------------------------------------------------------------
# boundary bins (per channel)
# ---------------------------------------------------------------------------


def _support_bin_index(dist_value):
    for i, (lo, hi) in enumerate(DISTANCE_BIN_EDGES):
        if math.isinf(hi):
            if dist_value > lo:
                return i
        elif lo <= 0.0:
            if lo <= dist_value <= hi:
                return i
        elif lo < dist_value <= hi:
            return i
    return len(DISTANCE_BIN_EDGES) - 1


# Halo (rows) required for exact bin discrimination through <= 16 pixels.
BOUNDARY_HALO = 17
# Row-tile height for the streaming boundary EDT work buffer.
BOUNDARY_TILE_ROWS = ROW_CHUNK


def _tile_distances(support_mask, r0, r1, halo=BOUNDARY_HALO):
    """Bounded row-tile distance-to-boundary (float64, transient).

    Only the distance values for rows ``[r0, r1)`` are returned (exact through
    ``halo - 1 >= 16`` output pixels).  The four global array edges are treated
    as support boundaries (false padding); interior continuations are padded
    True so no artificial window boundary is introduced.  At most one such tile
    work buffer is resident at a time; the caller must consume and release it.
    """
    from scipy.ndimage import distance_transform_edt

    m = np.asarray(support_mask, dtype=bool)
    h, w = m.shape
    tr0 = max(0, r0 - halo)
    tr1 = min(h, r1 + halo)
    sub = m[tr0:tr1]
    padded = np.empty((sub.shape[0] + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = sub
    # top/bottom: global exterior when the window reaches the array edge, else
    # an interior continuation (True) so no false boundary appears.
    if tr0 == 0:
        padded[0, :] = False
    else:
        padded[0, :] = True
    if tr1 == h:
        padded[-1, :] = False
    else:
        padded[-1, :] = True
    # left/right array edges are always global exterior
    padded[:, 0] = False
    padded[:, -1] = False
    dist = distance_transform_edt(padded)
    inner = dist[1:-1, 1:-1]
    return inner[r0 - tr0 : r1 - tr0]


def _local_support_distance(support_mask, row, col, max_radius=BOUNDARY_HALO):
    """Exact local distance-to-boundary for one pixel, or ``None`` if > radius.

    Uses a bounded ``(2*max_radius+1)`` window with the same edge semantics as
    the tiled map (global edges are background; interior continuations are
    padded True).  ``None`` means the true distance exceeds ``max_radius`` (a
    truthful "unresolved />16" representation).
    """
    try:
        from scipy.ndimage import distance_transform_edt

        m = np.asarray(support_mask, dtype=bool)
        h, w = m.shape
        r = int(row); c = int(col)
        if not (0 <= r < h and 0 <= c < w):
            return None
        R = int(max_radius)
        r0 = max(0, r - R); r1 = min(h, r + R + 1)
        c0 = max(0, c - R); c1 = min(w, c + R + 1)
        sub = m[r0:r1, c0:c1]
        padded = np.empty((sub.shape[0] + 2, sub.shape[1] + 2), dtype=bool)
        padded[1:-1, 1:-1] = sub
        padded[0, :] = False if r0 == 0 else True
        padded[-1, :] = False if r1 == h else True
        padded[:, 0] = False if c0 == 0 else True
        padded[:, -1] = False if c1 == w else True
        dist = distance_transform_edt(padded)
        val = float(dist[1 + (r - r0), 1 + (c - c0)])
        if not math.isfinite(val) or val > R:
            return None
        return val
    except Exception as exc:  # noqa: BLE001
        logger.debug("_local_support_distance failed (non-fatal): %s", exc)
        return None


def spatial_boundary_diagnostics(sci, wht, support_mask, neff=None,
                                 sup_w1=None, sup_w2=None,
                                 chunk_rows=BOUNDARY_TILE_ROWS,
                                 n_extrema=EXTREMA_COUNT, halo=BOUNDARY_HALO):
    """Per-distance-bin summaries over the physical support (per channel).

    Streams **row tiles** with a bounded halo and computes a transient tile EDT
    (never a full-frame distance map, never a full float64 upcast of a resident
    float32 map).  ``sci``/``wht`` may be ``(H, W)`` or ``(H, W, C)``; one bin
    list per channel.  N_eff is computed per tile from the support pair when
    supplied (or from a caller-provided ``neff`` array), never materialised.
    """
    out = {
        "available": False, "reason": None,
        "small_positive_wht_cut": _f(SMALL_POSITIVE_WHT),
        "support_source": None, "per_channel": [],
        "distance_algorithm": "row_tile_edt_halo",
    }
    try:
        s = np.asarray(sci, dtype=np.float32)
        w = np.asarray(wht, dtype=np.float32)
        m = np.asarray(support_mask, dtype=bool)
        if s.ndim == 2:
            s = s[..., None]
            w = w[..., None]
        if s.shape[:2] != m.shape or w.shape[:2] != m.shape:
            out["reason"] = "shape_mismatch"
            return out
        h, wid, nch = s.shape
        tile_rows = max(1, int(chunk_rows))
        halo = max(17, int(halo))
        out["halo"] = int(halo)
        out["tile_rows"] = int(tile_rows)
        out["max_temporary_bytes"] = int(
            (tile_rows + 2 * halo + 4) * (wid + 2) * 8
        )
        total_support = 0
        nbins = len(DISTANCE_BIN_EDGES)
        edge_arr = np.asarray([hi for _lo, hi in DISTANCE_BIN_EDGES[:-1]],
                              dtype=np.float64)

        ext_arrs = []
        acc = []
        for c in range(nch):
            lo_i, hi_i = bounded_extrema_indices(
                s[..., c], m, int(n_extrema), chunk_rows=tile_rows
            )
            ext_arrs.append(np.sort(np.asarray(list(lo_i) + list(hi_i), dtype=np.int64)))
            acc.append({
                "counts": np.zeros(nbins, dtype=np.int64),
                "sci_mn": np.full(nbins, math.inf),
                "sci_mx": np.full(nbins, -math.inf),
                "small": np.zeros(nbins, dtype=np.int64),
                "extreme": np.zeros(nbins, dtype=np.int64),
                "ns": np.zeros(nbins),
                "nc": np.zeros(nbins, dtype=np.int64),
                "nmn": np.full(nbins, math.inf),
                "nmx": np.full(nbins, -math.inf),
            })

        sup1 = None if sup_w1 is None else np.asarray(sup_w1, dtype=np.float32)
        sup2 = None if sup_w2 is None else np.asarray(sup_w2, dtype=np.float32)
        neff_arr = None if neff is None else np.asarray(neff, dtype=np.float32)

        for r0 in range(0, h, tile_rows):
            r1 = min(h, r0 + tile_rows)
            mc = m[r0:r1]
            if not np.any(mc):
                continue
            total_support += int(np.count_nonzero(mc))
            dist_rows = _tile_distances(m, r0, r1, halo)
            row_idx, col_idx = np.nonzero(mc)
            gi = (r0 + row_idx).astype(np.int64) * wid + col_idx.astype(np.int64)
            dvals = dist_rows[row_idx, col_idx]
            bi = np.searchsorted(edge_arr, dvals, side="left")
            if sup1 is not None and sup2 is not None:
                a1 = sup1[r0:r1][row_idx, col_idx]
                a2 = sup2[r0:r1][row_idx, col_idx]
                ok = np.isfinite(a1) & np.isfinite(a2) & (a1 > 0.0) & (a2 > 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rr = a1 / np.sqrt(a2)
                    ne = np.where(ok, rr * rr, np.nan)
            elif neff_arr is not None:
                ne = neff_arr[r0:r1][row_idx, col_idx]
            else:
                ne = None
            for c in range(nch):
                a = acc[c]
                sv = s[r0:r1, :, c][row_idx, col_idx]
                wv = w[r0:r1, :, c][row_idx, col_idx]
                small_v = np.isfinite(wv) & (wv > 0.0) & (wv <= SMALL_POSITIVE_WHT)
                ea = ext_arrs[c]
                if ea.size:
                    pos = np.searchsorted(ea, gi)
                    pos = np.clip(pos, 0, ea.size - 1)
                    ext_member = ea[pos] == gi
                else:
                    ext_member = np.zeros(gi.size, dtype=bool)
                for k in range(nbins):
                    sel = bi == k
                    nsel = int(np.count_nonzero(sel))
                    if nsel == 0:
                        continue
                    a["counts"][k] += nsel
                    vv = sv[sel]
                    fin = np.isfinite(vv)
                    if np.any(fin):
                        a["sci_mn"][k] = min(a["sci_mn"][k], float(np.min(vv[fin])))
                        a["sci_mx"][k] = max(a["sci_mx"][k], float(np.max(vv[fin])))
                    a["small"][k] += int(np.count_nonzero(small_v[sel]))
                    a["extreme"][k] += int(np.count_nonzero(ext_member[sel]))
                    if ne is not None:
                        nv = ne[sel]
                        nv = nv[np.isfinite(nv)]
                        if nv.size:
                            a["ns"][k] += float(np.sum(nv))
                            a["nc"][k] += int(nv.size)
                            a["nmn"][k] = min(a["nmn"][k], float(np.min(nv)))
                            a["nmx"][k] = max(a["nmx"][k], float(np.max(nv)))
            del dist_rows

        out["total_support_pixels"] = int(total_support)
        per_channel = []
        for c in range(nch):
            a = acc[c]
            bins = []
            for k, label in enumerate(DISTANCE_BIN_LABELS):
                n = int(a["counts"][k])
                bins.append({
                    "label": label,
                    "pixels": n,
                    "fraction_of_support": (
                        _f(n / total_support) if total_support else None
                    ),
                    "sci_min": _f(a["sci_mn"][k]) if math.isfinite(a["sci_mn"][k]) else None,
                    "sci_max": _f(a["sci_mx"][k]) if math.isfinite(a["sci_mx"][k]) else None,
                    "small_positive_wht_fraction": (
                        _f(a["small"][k] / n) if n else None
                    ),
                    "extreme_pixels": int(a["extreme"][k]),
                    "extreme_fraction": (
                        _f(a["extreme"][k] / n) if n else None
                    ),
                    "n_eff_min": _f(a["nmn"][k]) if math.isfinite(a["nmn"][k]) else None,
                    "n_eff_mean": (
                        _f(a["ns"][k] / a["nc"][k]) if a["nc"][k] else None
                    ),
                    "n_eff_max": _f(a["nmx"][k]) if math.isfinite(a["nmx"][k]) else None,
                })
            per_channel.append({"channel": c, "bins": bins})
        out["per_channel"] = per_channel
        out["available"] = True
    except Exception as exc:  # noqa: BLE001
        logger.debug("spatial_boundary_diagnostics failed (non-fatal): %s", exc)
        out["reason"] = "exception"
    return out


# ---------------------------------------------------------------------------
# local references + conditioning candidates (per channel)
# ---------------------------------------------------------------------------


def _chunked_positive_sample(arr_2d, chunk_rows=ROW_CHUNK, cap=MAX_SAMPLE_COUNT):
    """Deterministic bounded sample of strictly-positive finite values (streamed).

    Never materialises a full-channel positive subset; the returned sample is
    capped at ``cap``.  Returns ``(sample, stride)``.
    """
    a = np.asarray(arr_2d, dtype=np.float32)
    h, wid = a.shape
    stride = _sample_stride(int(h) * int(wid), cap)
    out = []
    for r0, r1 in _iter_row_chunks(h, wid, chunk_rows):
        sel = _stride_select(a[r0:r1].reshape(-1), r0 * wid, stride)
        if sel.size:
            p = sel[sel > 0.0]
            if p.size:
                out.append(p.astype(np.float64, copy=True))
    s = np.concatenate(out) if out else np.empty(0)
    if s.size > cap:
        s = s[:: max(1, s.size // cap)][:cap]
    return s, stride


def _local_positive_reference(wht_2d, tile=LOCAL_REFERENCE_TILE,
                              chunk_rows=ROW_CHUNK, sample_cap=MAX_SAMPLE_COUNT):
    """Bounded per-tile positive-WHT reference (one scalar per tile).

    The global fallback is a deterministic bounded sample; each tile's positive
    subset is bounded by ``tile**2`` — no full-channel positive index copy.
    """
    w = np.asarray(wht_2d, dtype=np.float32)
    h, wid = w.shape
    gsample, _gstride = _chunked_positive_sample(w, chunk_rows, sample_cap)
    global_ref = float(np.percentile(gsample, 90.0)) if gsample.size else 0.0
    th = int(math.ceil(h / tile))
    tw = int(math.ceil(wid / tile))
    ref = np.full((th, tw), global_ref, dtype=np.float64)
    for i in range(th):
        r0 = i * tile
        r1 = min(h, r0 + tile)
        for j in range(tw):
            c0 = j * tile
            c1 = min(wid, c0 + tile)
            block = w[r0:r1, c0:c1]
            bp = block[np.isfinite(block) & (block > 0.0)]
            if bp.size:
                ref[i, j] = float(np.percentile(bp, 90.0))
    return ref, global_ref, int(tile), (th, tw)


def conditioning_candidates(sci, wht, support_mask, w1=None, w2=None,
                            neff=None, distance=None, n_extrema=EXTREMA_COUNT,
                            tile=LOCAL_REFERENCE_TILE, chunk_rows=ROW_CHUNK):
    """Per-channel, per-extreme candidate conditioning evidence (bounded).

    For each channel's selected SCI extrema reports native signed WHT,
    WHT / bounded local positive-WHT reference, WHT / SUP_W1-derived reference,
    N_eff and distance to the physical support boundary.  Local references use
    bounded tile samples; the boundary distance is queried locally (bounded
    window) per extreme.  No rule is chosen or applied.
    """
    out = {
        "tile": int(tile), "local_reference_statistic": "positive_p90",
        "local_reference_method": "bounded_tile_sample",
        "local_reference_sample_cap": int(MAX_SAMPLE_COUNT),
        "distance_method": "local_window_edt", "per_channel": [], "reason": None,
    }
    try:
        s = np.asarray(sci, dtype=np.float32)
        w = np.asarray(wht, dtype=np.float32)
        m = np.asarray(support_mask, dtype=bool)
        if s.ndim == 2:
            s = s[..., None]
            w = w[..., None]
        if s.shape[:2] != m.shape:
            out["reason"] = "shape_mismatch"
            return out
        w1_ref = None
        if w1 is not None:
            samp, _st = _chunked_positive_sample(np.asarray(w1, dtype=np.float32),
                                                 chunk_rows)
            if samp.size:
                w1_ref = float(np.median(samp))
        out["sup_w1_reference"] = _f(w1_ref)
        for c in range(s.shape[-1]):
            sc = s[..., c]
            wc = w[..., c]
            local_ref, global_ref, _tile, thw = _local_positive_reference(
                wc, tile, chunk_rows
            )
            recs = bounded_extrema_records(
                sc, wc, m, w1, w2, None, None, n=n_extrema,
                chunk_rows=chunk_rows,
            )
            rows = []
            for r in recs:
                row = r["row"]
                col = r["col"]
                ti = int(row) // int(tile)
                tj = int(col) // int(tile)
                local = float(local_ref[ti, tj]) if local_ref.size else global_ref
                wv = r.get("wht")
                dist = _local_support_distance(m, row, col)
                rows.append({
                    "kind": r["kind"], "index": r["index"], "row": row,
                    "col": col, "sci": r.get("sci"), "wht": wv,
                    "wht_over_local_ref": (
                        _f(wv / local) if wv is not None and local not in (0.0,) else None
                    ),
                    "wht_over_global_ref": (
                        _f(wv / global_ref) if wv is not None and global_ref else None
                    ),
                    "wht_over_sup_w1_ref": (
                        _f(wv / w1_ref)
                        if wv is not None and w1_ref not in (None, 0.0) else None
                    ),
                    "n_eff": r.get("n_eff"),
                    "sup_w1": r.get("sup_w1"),
                    "sup_w2": r.get("sup_w2"),
                    "distance": _f(dist),
                    "distance_resolved": dist is not None,
                    "distance_bound": float(BOUNDARY_HALO),
                })
            out["per_channel"].append({
                "channel": c, "global_positive_reference": _f(global_ref),
                "local_reference_tiles_hw": list(thw), "extrema": rows,
            })
    except Exception as exc:  # noqa: BLE001
        logger.debug("conditioning_candidates failed (non-fatal): %s", exc)
        out["reason"] = "exception"
    return out


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def lifecycle_record(stage, obj, **extra):
    """Build one bounded SUPPORT_LIFECYCLE record from a queue-manager object."""
    rec = {
        "stage": str(stage),
        "ts": time.time(),
        "support_available": bool(getattr(obj, "_drizzle_support_available", False)),
        "support_reason": getattr(obj, "_drizzle_support_unavailable_reason", None),
        "classic_sup_w1_present": getattr(obj, "coverage_sup_w1_memmap", None) is not None,
        "classic_sup_w2_present": getattr(obj, "coverage_sup_w2_memmap", None) is not None,
        "classic_support_state_available": bool(
            getattr(obj, "_support_state_available", False)
        ),
        "drizzle_sup_w1_present": getattr(obj, "drizzle_sup_w1", None) is not None,
        "drizzle_sup_w2_present": getattr(obj, "drizzle_sup_w2", None) is not None,
        "drizzle_support_released": False,
        "stopped": bool(
            getattr(obj, "user_requested_stop", False)
            or getattr(obj, "stop_processing", False) is True
        ),
        "finalization_mode": getattr(obj, "finalization_mode", None),
        "coverage_render_status": getattr(obj, "coverage_render_status", None),
        "frame_count": _i(getattr(obj, "_drizzle_frame_count", None)),
    }
    for key, value in extra.items():
        if isinstance(value, bool) or value is None:
            rec[key] = value
        elif isinstance(value, (int,)):
            rec[key] = value
        elif isinstance(value, float):
            rec[key] = _f(value)
        else:
            rec[key] = str(value)
    return rec


# ---------------------------------------------------------------------------
# collector / artifact writer
# ---------------------------------------------------------------------------

# Stages that must survive retention (always retained).
TERMINAL_STAGES = frozenset({
    "accumulators_ready", "fresh_m3_init", "first_science_deposit",
    "first_support_deposit", "stop_requested", "drizzle_finalization_entered",
    "drizzle_finalization_pre_save", "drizzle_finalization_returned",
    "coverage_render_entered", "coverage_render_exited", "fits_save",
    "memmap_cleanup_entered", "memmap_cleanup_returned",
    "drizzle_support_unavailable", "artifact_final",
})
# Stages coalesced in place (repetitive high-frequency events).
COALESCE_STAGES = frozenset({"checkpoint_save"})
MAX_LIFECYCLE_EVENTS = 256


class DrizzleScienceDiagnostics:
    """Bounded, passive per-run collector + atomic artifact writer.

    Every setter and the writer are fail-open.  Lifecycle retention is bounded:
    terminal/stop/finalization/first-deposit facts are always retained, while
    repetitive ``checkpoint_save`` events are coalesced (count + first/last
    generation/frame) so they can never crowd out terminal evidence.
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
        self.crop = None
        self.wht_policy = None
        self.sections_meta = None
        self.lifecycle = []
        self.coalesced = {}
        self.notes = []
        self.stop_state = {}
        self.finalization_state = {}
        self.artifact_path = None
        self.write_count = 0

    # -- setters (all fail-open) -------------------------------------------
    def _safe(self, fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
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

    def set_sci_stats(self, value):
        self._safe(lambda: setattr(self, "sci_stats", list(value)))

    def set_wht_diagnostics(self, value):
        self._safe(lambda: setattr(self, "wht_diagnostics", list(value)))

    def set_threshold_sweep(self, value):
        self._safe(lambda: setattr(self, "threshold_sweep", list(value)))

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

    def set_crop(self, record):
        self._safe(lambda: setattr(self, "crop", dict(record)))

    def set_wht_policy(self, record):
        self._safe(lambda: setattr(self, "wht_policy", dict(record)))

    def set_sections_meta(self, record):
        self._safe(lambda: setattr(self, "sections_meta", dict(record)))

    def add_lifecycle(self, record):
        def _do():
            stage = record.get("stage")
            if stage in COALESCE_STAGES:
                prev = self.coalesced.get(stage)
                if prev is None:
                    rec = dict(record)
                    rec["count"] = 1
                    self.coalesced[stage] = rec
                else:
                    prev["count"] = int(prev.get("count", 1)) + 1
                    prev["last_ts"] = record.get("ts")
                    for k in ("generation", "frame_count"):
                        if k in record:
                            prev.setdefault("first_" + k, prev.get(k))
                            prev[k] = record[k]
                return
            self.lifecycle.append(dict(record))
            if len(self.lifecycle) > MAX_LIFECYCLE_EVENTS:
                # evict the oldest non-terminal event; never drop terminal ones
                for idx, rec in enumerate(self.lifecycle):
                    if rec.get("stage") not in TERMINAL_STAGES:
                        del self.lifecycle[idx]
                        break
                else:
                    # all terminal: drop the oldest (bounded hard cap)
                    del self.lifecycle[0]
        self._safe(_do)

    def note(self, message):
        self._safe(lambda: (len(self.notes) < 64) and self.notes.append(str(message)))

    def set_stop_state(self, **fields):
        def _do():
            self.stop_state.update(
                {k: (_f(v) if isinstance(v, float) else v) for k, v in fields.items()}
            )
        self._safe(_do)

    def set_finalization_state(self, **fields):
        def _do():
            self.finalization_state.update(
                {k: (_f(v) if isinstance(v, float) else v) for k, v in fields.items()}
            )
        self._safe(_do)

    # -- serialization -----------------------------------------------------
    def lifecycle_summary(self):
        summary = {
            "count": len(self.lifecycle), "stage_counts": {}, "last_stage": None,
            "first_support_available": None, "last_support_available": None,
            "coalesced": {k: dict(v) for k, v in self.coalesced.items()},
            "max_events": MAX_LIFECYCLE_EVENTS,
        }
        try:
            for rec in self.lifecycle:
                stage = rec.get("stage")
                summary["stage_counts"][stage] = summary["stage_counts"].get(stage, 0) + 1
                if summary["first_support_available"] is None:
                    summary["first_support_available"] = rec.get("support_available")
                summary["last_support_available"] = rec.get("support_available")
            if self.lifecycle:
                summary["last_stage"] = self.lifecycle[-1].get("stage")
        except Exception as exc:  # noqa: BLE001
            logger.debug("lifecycle_summary failed (non-fatal): %s", exc)
        return summary

    def lifecycle_stages(self):
        """Ordered list of persisted stage names (excluding coalesced)."""
        return [r.get("stage") for r in self.lifecycle]

    def to_dict(self):
        try:
            geometry = self.geometry or {}
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
                "pixel_scale_ratio_source": "upstream_add_image_default",
                "geometry": geometry,
                "add_image_contract": self.contract or {},
                "crop": self.crop or {},
                "effective_wht_policy": self.wht_policy or {},
                "sections_meta": self.sections_meta or {},
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
        """Atomically write the artifact (temp + ``os.replace``).  Fail-open.

        The temporary file is always removed (best-effort ``finally``) when the
        write/replace path fails, so a failed write cannot leak a stale
        ``*.tmp.*`` sibling.
        """
        tmp = None
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
            tmp = None
            self.artifact_path = target
            self.write_count += 1
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("drizzle diagnostics write failed (non-fatal): %s", exc)
            return False
        finally:
            if tmp is not None:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:  # noqa: BLE001 - fail-open cleanup
                    logger.debug("diagnostics temp cleanup failed (non-fatal)")


# ---------------------------------------------------------------------------
# high-level finalization summary (bounded)
# ---------------------------------------------------------------------------


def summarize_run(sci_hwc, wht_hwc, sup_w1=None, sup_w2=None,
                  support_mask=None, crop=None, chunk_rows=ROW_CHUNK,
                  sample_cap=MAX_SAMPLE_COUNT, n_extrema=EXTREMA_COUNT):
    """Compute every bounded per-run diagnostic section (read-only, fail-open).

    Streams row chunks of float32 views; never converts a whole HWC cube to
    float64.  All sections are **per channel**.  ``support_mask`` is the physical
    support (``SUP_W1 > 0``) when available; otherwise the explicitly-labelled
    native-WHT-derived fallback is used.  ``crop`` records the bbox origin.
    """
    sections = {
        "sci_stats": [], "wht_diagnostics": [], "threshold_sweep": [],
        "support": {"available": False, "reason": "unavailable"},
        "support_extrema": [], "boundary_bins": {"available": False, "reason": "unavailable"},
        "conditioning_candidates": {"per_channel": [], "reason": "unavailable"},
        "support_source": None,
        "meta": {"chunk_rows": int(chunk_rows), "sample_cap": int(sample_cap),
                 "extrema_count": int(n_extrema)},
    }
    try:
        s = np.asarray(sci_hwc)
        w = np.asarray(wht_hwc)
        if s.ndim == 2:
            s = s[..., None]
        if w.ndim == 2:
            w = w[..., None]
        if s.shape != w.shape:
            sections["meta"]["error"] = "sci/wht shape mismatch"
            return sections
        h, wid, nch = s.shape
        sections["meta"]["shape_hwc"] = [int(h), int(wid), int(nch)]
        sections["meta"]["crop"] = dict(crop or {})

        mask2d = None
        support_source = "support_pair_absent"
        if support_mask is not None:
            try:
                m = np.asarray(support_mask, dtype=bool)
                if m.shape == (h, wid):
                    mask2d = m
                    support_source = "sup_w1_positive"
            except Exception:  # noqa: BLE001
                mask2d = None
        if mask2d is None:
            mask2d = native_wht_fallback_mask(w)
            support_source = "native_wht_derived_fallback"
        sections["support_source"] = support_source

        # Bounded streaming boundary analysis: NO full-frame distance map and no
        # full float64 upcast.  One row-tile EDT work buffer at a time.
        degraded = []

        for c in range(nch):
            sc = s[..., c]
            wc = w[..., c]
            sections["sci_stats"].append(
                sci_channel_stats(sc, chunk_rows, sample_cap)
            )
            sections["wht_diagnostics"].append(
                wht_channel_diagnostics(wc, sc, n_extrema, chunk_rows, sample_cap)
            )
            sections["threshold_sweep"].append(
                threshold_sweep(wc, sc, mask2d, chunk_rows=chunk_rows,
                                support_source=("physical" if support_source == "sup_w1_positive"
                                                else "native_positive_wht_fallback"))
            )

        if sup_w1 is not None and sup_w2 is not None:
            sections["support"] = support_conditioning(sup_w1, sup_w2, chunk_rows, sample_cap)
        else:
            sections["support"] = {"available": False,
                                   "reason": "support_accumulator_absent"}

        extrema_by_channel = []
        for c in range(nch):
            recs = bounded_extrema_records(
                s[..., c], w[..., c], mask2d, sup_w1, sup_w2, None, None,
                n=n_extrema, chunk_rows=chunk_rows,
            )
            extrema_by_channel.append(recs)
        sections["support_extrema"] = extrema_by_channel

        sections["boundary_bins"] = spatial_boundary_diagnostics(
            s, w, mask2d, sup_w1=sup_w1, sup_w2=sup_w2,
            chunk_rows=chunk_rows, n_extrema=n_extrema,
        )
        sections["boundary_bins"]["support_source"] = support_source
        sections["meta"]["boundary_algorithm"] = sections["boundary_bins"].get(
            "distance_algorithm"
        )
        sections["meta"]["boundary_halo"] = sections["boundary_bins"].get("halo")
        sections["meta"]["max_temporary_bytes"] = sections["boundary_bins"].get(
            "max_temporary_bytes"
        )
        if not sections["boundary_bins"].get("available"):
            degraded.append("boundary_bins")
        sections["conditioning_candidates"] = conditioning_candidates(
            s, w, mask2d, w1=sup_w1, w2=sup_w2,
            n_extrema=n_extrema, chunk_rows=chunk_rows,
        )
        if sections["conditioning_candidates"].get("reason"):
            degraded.append("conditioning_candidates")
        sections["meta"]["degraded_sections"] = degraded
    except Exception as exc:  # noqa: BLE001
        logger.debug("summarize_run failed (non-fatal): %s", exc)
    return sections


__all__ = [
    "SCHEMA_VERSION", "ARTIFACT_FILENAME", "WEIGHT_EPSILON",
    "SMALL_POSITIVE_WHT", "DEFAULT_ABS_THRESHOLDS", "DEFAULT_REL_THRESHOLDS",
    "DISTANCE_BIN_LABELS", "DISTANCE_BIN_EDGES", "ROW_CHUNK",
    "MAX_SAMPLE_COUNT", "MAX_BOUNDARY_WORK_BYTES", "MAX_LIFECYCLE_EVENTS",
    "TERMINAL_STAGES", "COALESCE_STAGES",
    "DrizzleScienceDiagnostics", "robust_pixel_scale_deg", "geometry_diagnostic",
    "contract_diagnostic", "sci_channel_stats", "sci_stats_hwc",
    "wht_channel_diagnostics", "wht_diagnostics_hwc", "select_extrema_indices",
    "bounded_extrema_indices",
    "sci_extrema_records", "bounded_extrema_records", "threshold_sweep",
    "support_conditioning", "conditioning_candidates",
    "spatial_boundary_diagnostics", "support_distance_map",
    "physical_support_mask", "native_wht_fallback_mask", "lifecycle_record",
    "summarize_run",
]
