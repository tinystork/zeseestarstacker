"""Deterministic real-engine Drizzle witness probes (research only).

ZSSS drizzle-contract archaeology (mission zsss-drizzle-contract-20260909).

This module runs the REAL installed ``drizzle`` 2.2.0 engine through the SAME
parameter convention ZSSS uses for its Standard M3 direct accumulation
(see ``seestar/core/drizzle_core.py::DrizzleAccumulator`` and
``queue_manager._add_frame_to_drizzle_accumulators``):

* output grid: ``round(H*scale) x round(W*scale)`` (ZSSS ``build_output_grid``
  convention — CDELT/scale & CRPIX*scale make the output pixel index equal to
  ``scale * input_pixel_index`` for an identity alignment);
* pixmap: output coordinate ``scale*(i+0.5)`` for every input pixel centre
  (deterministic stand-in for the WCS world round-trip on an aligned grid);
* ``Drizzle(out_img=..., out_wht=..., kernel=..., fillval=...)`` then
  ``add_image(data=..., exptime=..., pixmap=..., weight_map=..., in_units=
  "counts", pixfrac=..., wht_scale=exptime)`` — mirroring
  ``DrizzleAccumulator.add``.  Like ZSSS, NO ``iscale`` and NO
  ``pixel_scale_ratio`` are passed by default (upstream defaults 1.0 / 1.0);
  the A/B cells pass an explicit ``pixel_scale_ratio=scale``.
* data are float32 ADU counts; ``weight_map`` = ones (weighting OFF) or a
  deterministic finite validity mask (weighting ON).

Measurements per cell: SCI min/max/mean/median/robust percentiles / negative
fraction / non-finite fraction / extreme fraction, native WHT min/max/sign
distribution, positive-support, constant/background behaviour.  Lanczos cells
add WHT-bin statistics (WHT<=0, (0,1e-9], (1e-9,1e-7], (1e-7,1e-5], ...).

Artifacts: ``artifacts/<cell>.npz`` raw arrays + ``artifacts/metrics.csv`` +
``artifacts/metrics.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from drizzle.resample import Drizzle

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# ---------------------------------------------------------------------------
# Deterministic inputs (A-E)
# ---------------------------------------------------------------------------


def input_constant(shape=(32, 32), value=100.0, seed=None):
    rng = np.random.default_rng(seed or 0)
    base = np.full(shape, value, dtype=np.float32)
    # deterministic tiny noise so percentiles are non-degenerate
    return (base + rng.normal(0.0, 0.05, shape).astype(np.float32)).astype(
        np.float32
    )


def input_star(shape=(48, 48), peak=1000.0, sigma=1.2, seed=1):
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, dtype=np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    img = (
        peak
        * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2)))
    ).astype(np.float32)
    return (img + rng.normal(0.0, 0.01, shape).astype(np.float32)).astype(
        np.float32
    )


def input_edge(shape=(32, 32), level=200.0):
    img = np.zeros(shape, dtype=np.float32)
    img[:, : shape[1] // 2] = level
    return img


def input_partial(shape=(32, 32), shift=8.0, seed=3):
    """A sharp field translated partly outside the output grid (footprint edge
    inside the output window -> partial coverage / uncovered border)."""
    img = input_constant(shape, value=150.0, seed=seed)
    # translate by 'shift' pixels in x/y -> left/top columns map outside
    return img, {"shift": shift}


def pixmap_identity(shape, scale, shift=(0.0, 0.0)):
    """Output coordinates of input pixel centres under ZSSS grid convention.

    Output pixel index == scale * input pixel index (CRPIX/CDELT derivation);
    pixel centre (i+0.5) maps to scale*(i+0.5).  Optional shift in INPUT
    pixels (for the partial-footprint cell; shifted centres that land outside
    the output grid are masked by the caller as ZSSS does).
    """
    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    # input centre coordinates
    px = xx + 0.5 + shift[0]
    py = yy + 0.5 + shift[1]
    out = np.empty((h, w, 2), dtype=np.float64)
    out[..., 0] = scale * px
    out[..., 1] = scale * py
    return out


def in_grid_mask(pixmap, out_shape):
    """ZSSS ``in_grid_mask``: pixel centre inside the output grid."""
    h, w = out_shape
    mx = (pixmap[..., 0] >= 0) & (pixmap[..., 0] < w)
    my = (pixmap[..., 1] >= 0) & (pixmap[..., 1] < h)
    return mx & my


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    key: str
    kernel: str
    scale: float
    input_name: str
    weighting: str = "off"
    n_frames: int = 1
    pixel_scale_ratio: float | None = None
    iscale: float = 1.0
    exptime: float = 1.0
    pixfrac: float = 1.0
    sci: np.ndarray = field(default=None, repr=False)
    wht: np.ndarray = field(default=None, repr=False)
    mask_in: np.ndarray = field(default=None, repr=False)
    extras: dict = field(default_factory=dict)


def _safe_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return {}
    finite = a[np.isfinite(a)]
    n_fin = finite.size
    out = {}
    if n_fin:
        out["min"] = float(np.min(finite))
        out["max"] = float(np.max(finite))
        out["mean"] = float(np.mean(finite))
        out["median"] = float(np.median(finite))
        for q in (1.0, 5.0, 95.0, 99.0):
            out[f"p{q:g}"] = float(np.percentile(finite, q))
    out["negative_fraction"] = float(np.mean(a < 0)) if a.size else None
    out["positive_fraction"] = float(np.mean(a > 0)) if a.size else None
    out["zero_fraction"] = float(np.mean(a == 0)) if a.size else None
    out["nonfinite_fraction"] = float(
        1.0 - np.mean(np.isfinite(a))
    ) if a.size else None
    return out


def measure(cell: Cell) -> dict:
    sci = np.asarray(cell.sci, dtype=np.float32)
    wht = np.asarray(cell.wht, dtype=np.float32)
    m = {"key": cell.key, "kernel": cell.kernel, "scale": cell.scale,
         "input": cell.input_name, "weighting": cell.weighting,
         "n_frames": cell.n_frames,
         "pixel_scale_ratio": cell.pixel_scale_ratio,
         "iscale": cell.iscale, "exptime": cell.exptime,
         "pixfrac": cell.pixfrac,
         "sci": _safe_stats(sci), "wht": _safe_stats(wht)}
    # support: strict positive native WHT (ZSSS WEIGHT_EPSILON view)
    pos = wht > 1e-9
    if np.any(pos):
        m["positive_support_fraction"] = float(np.mean(pos))
        m["sci_on_support_mean"] = float(np.mean(sci[pos]))
        m["sci_on_support_neg_fraction"] = float(np.mean(sci[pos] < 0))
        m["sci_on_support_extreme_p99_abs"] = float(
            np.percentile(np.abs(sci[pos]), 99)
        )
    else:
        m["positive_support_fraction"] = 0.0
    if cell.extras.get("star", False):
        m["star"] = _star_metrics(sci, wht)
    return m


def _star_metrics(sci, wht):
    pos = wht > 1e-9
    if not np.any(pos):
        return {}
    s = np.asarray(sci, dtype=np.float64)
    support = np.where(pos, s, 0.0)
    total = float(np.sum(support))
    idx = np.unravel_index(int(np.argmax(np.where(pos, s, -np.inf))), s.shape)
    peak = float(s[idx])
    # centroid on positive support
    yy, xx = np.mgrid[0:s.shape[0], 0:s.shape[1]]
    wsum = np.sum(np.where(pos, s - np.min(s[pos]), 0.0))
    if wsum > 0:
        cy = float(np.sum(yy * np.where(pos, s - np.min(s[pos]), 0.0)) / wsum)
        cx = float(np.sum(xx * np.where(pos, s - np.min(s[pos]), 0.0)) / wsum)
    else:
        cy = cx = None
    neg = s[s < 0]
    return {
        "integrated_flux": total,
        "peak": peak,
        "centroid_y": cy,
        "centroid_x": cx,
        "negative_ringing_min": float(np.min(neg)) if neg.size else 0.0,
        "negative_ringing_sum_abs": float(np.sum(np.abs(neg))),
    }


def wht_bins(sci, wht):
    """Bin SCI statistics against native WHT (Lanczos signed analysis)."""
    wht = np.asarray(wht, dtype=np.float64).ravel()
    sci = np.asarray(sci, dtype=np.float64).ravel()
    edges = [-np.inf, 0.0, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0, np.inf]
    labels = ["<=0", "(0,1e-9]", "(1e-9,1e-7]", "(1e-7,1e-5]",
              "(1e-5,1e-3]", "(1e-3,1e-1]", "(1e-1,1]", ">1"]
    rows = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        sel = np.where(np.isfinite(wht) & (wht > lo) & (wht <= hi))[0]
        n = sel.size
        row = {"bin": lab, "count": int(n)}
        if n:
            s = np.abs(sci[sel])
            row["sci_abs_max"] = float(np.max(s))
            row["sci_abs_p99"] = float(np.percentile(s, 99))
            row["sci_abs_p95"] = float(np.percentile(s, 95))
            row["sci_neg_fraction"] = float(np.mean(sci[sel] < 0))
            row["sci_median"] = float(np.median(sci[sel]))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Deposition driver (mirrors DrizzleAccumulator.add)
# ---------------------------------------------------------------------------


def deposit(frames, kernel, scale, exptime=1.0, weight=None, pixfrac=1.0,
            pixel_scale_ratio=None, iscale=1.0, wht_scale=None,
            out_shape=None, disable_ctx=False, resume_state=None,
            fillval="0.0", shape_hw=None):
    """Deposit frames through the REAL drizzle engine with ZSSS conventions.

    ``resume_state=(out_img, out_wht, total_exptime)`` continues an earlier
    accumulation exactly like ``DrizzleAccumulator.from_native_state``.
    Returns ``(sci, wht)`` where ``sci`` = native weighted mean ``out_img`` and
    ``wht`` = native total weight (signed for Lanczos), i.e. the raw engine
    buffers — finalization is left to the caller (no WEIGHT_EPSILON masking
    here so pathology is fully observable).
    """
    h, w = shape_hw if shape_hw is not None else frames[0].shape
    if out_shape is None:
        out_shape = (int(round(h * scale)), int(round(w * scale)))
    if resume_state is not None:
        out_img, out_wht, total_exptime = resume_state
        d = Drizzle(out_img=out_img, out_wht=out_wht, kernel=kernel,
                    fillval=fillval, exptime=total_exptime,
                    disable_ctx=disable_ctx)
    else:
        d = Drizzle(
            out_img=np.zeros(out_shape, dtype=np.float32),
            out_wht=np.zeros(out_shape, dtype=np.float32),
            kernel=kernel,
            fillval=fillval,
            disable_ctx=disable_ctx,
        )
    for f in frames:
        f = np.asarray(f, dtype=np.float32)
        pix = pixmap_identity(f.shape[:2], scale)
        mask = in_grid_mask(pix, out_shape)
        wm = np.ones(f.shape, dtype=np.float32)
        if weight is not None:
            wm = np.asarray(weight, dtype=np.float32) * mask.astype(np.float32)
        else:
            wm = wm * mask.astype(np.float32)
        expscale = exptime  # in_units="counts" -> wht_scale = exptime
        kwargs = dict(
            data=f,
            exptime=exptime,
            pixmap=pix,
            weight_map=wm,
            in_units="counts",
            pixfrac=pixfrac,
            wht_scale=expscale if wht_scale is None else wht_scale,
        )
        if pixel_scale_ratio is not None:
            kwargs["pixel_scale_ratio"] = pixel_scale_ratio
        if iscale != 1.0:
            kwargs["iscale"] = iscale
        d.add_image(**kwargs)
    return np.array(d.out_img, dtype=np.float32, copy=True), np.array(
        d.out_wht, dtype=np.float32, copy=True
    )


def save_cell(cell: Cell, metrics: dict, bins=None, extra_arrays=None) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    base = ARTIFACTS / cell.key
    np.savez_compressed(
        str(base) + ".npz",
        sci=cell.sci,
        wht=cell.wht,
        mask_in=(cell.mask_in if cell.mask_in is not None else np.array(True)),
    )
    row = {}
    row.update(metrics)
    # prefix nested statistic blocks to avoid key collisions (sci vs wht)
    for prefix, block in (("sci_", metrics["sci"]), ("wht_", metrics["wht"])):
        for k, v in block.items():
            row[prefix + k] = v
    row.pop("sci", None)
    row.pop("wht", None)
    for k, v in list(row.items()):
        if isinstance(v, (dict, list)):
            row[k] = json.dumps(v)
    return row
