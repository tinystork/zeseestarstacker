"""Neutral shared paired-overlap primitive for Classic support-aware normalization.

Phase 1 of the support-aware overlap photometry effort (mission
``zsss-support-overlap-p1-20260907``).  This module is deliberately **pure /
backend-neutral and testable**: it contains no QueueManager state, no I/O and
no Drizzle deposit logic.  It is the single estimator shared by the Classic
``linear_fit`` / ``sky_mean`` normalization paths.

Reuse of Drizzle math (verbatim)
--------------------------------
The Drizzle background module (:mod:`seestar.core.drizzle_background`) is the
authoritative home of the robust location estimator and of the deterministic
bounded sampling conventions.  This module **imports** those functions and
constants rather than re-implementing them, so there is exactly one
implementation and old/new results are identical by construction:

* ``robust_location`` (median / MAD sigma-clip location),
* ``DEFAULT_MIN_OVERLAP_SAMPLES`` (200, the Drizzle minimum-overlap
  precedent),
* ``DEFAULT_MAX_SAMPLES`` (250000, the deterministic sample budget),
* the stable reason vocabulary (``REASON_ACCEPTED``,
  ``REASON_INSUFFICIENT_OVERLAP``, ``REASON_NO_VALID_SAMPLES``,
  ``REASON_DEGENERATE_GEOMETRY``).

Only truly generic Classic-geometry helpers (synthetic-detector warp,
content-validity warp) live here; they never touch deposit/kernel/pixfrac/WHT
or signed Drizzle outputs.

Geometry decides, never brightness
----------------------------------
The geometric support of an aligned Classic source on the reference canvas is
obtained by warping a synthetic all-ones detector with the **same 2x3 affine
M** and the **same warp semantics** the Classic alignment used
(``INTER_LINEAR`` + ``BORDER_CONSTANT NaN``), so the support mask reproduces
exactly the pixels where the real warp produced content — including the
NaN-border ring that even zero-weight out-of-bounds taps create.  A source
pixel that legitimately carries value 0 (astronomical zero) stays supported;
a repaired/NaN border pixel does not.  No luminance threshold is ever used.

Conservative boundary policy (documented)
-----------------------------------------
* ``GEOM_SUPPORT_TOL = 1e-6``: an output pixel is geometrically supported when
  its synthetic ones-warp is finite and ``>= 1 - 1e-6`` (i.e. all four
  bilinear taps lie inside the source detector; the NaN ring is excluded).
* After that threshold the support is eroded by one 3x3 pass
  (``GEOM_ERODE_PX = 1``) as a documented reliability margin against
  sub-pixel boundary effects.  This is a boundary *policy*, not aggressive
  output cropping: the estimator only *samples* inside the eroded support and
  never crops or alters science images.

Content validity
----------------
``content_validity_after_loader`` converts an opt-in "originally non-finite
before loader repair" mask into the content-valid mask consumed here.  For
Bayer (CFA) sources the loader repair happens before debayer, so a repaired
sample can influence a small debayered neighbourhood; the conservative
documented handling is a one-pixel dilation (``DEBAYER_INFLUENCE_PX = 1``)
applied in CFA space, without changing any debayer science.  A canvas pixel
is then content-valid iff every bilinear tap of its inverse-mapped position
is content-valid (equivalently: the content mask warped with the same M is
``1`` within ``CONTENT_WARP_TOL = 1e-6`` and finite).

Estimators
----------
``estimate_sky_mean_offset`` and ``estimate_linear_fit`` operate on the
**aligned source canvas** and the **immutable reference canvas** (both on the
same pixel grid, so the same canvas coordinates are the same sky
coordinates).  They sample only positions that are simultaneously:

* geometrically supported by the source (M footprint, eroded),
* content-valid on the source side,
* content-valid on the reference side,
* finite in both arrays.

* ``sky_mean``: scalar luminance (Rec.709 ``0.299 R + 0.587 G + 0.114 B``)
  paired difference ``I - R`` at the common positions, then
  ``robust_location`` -> a single scalar offset to **subtract from every
  channel** (same scalar across RGB).  No gain, no per-channel rescale.
* ``linear_fit``: the existing per-channel ``P25``/``P90`` model
  ``a = where(delta_src > 1e-5, delta_ref / max(delta_src, 1e-9), 1)``,
  ``b = ref_low - a * src_low`` — with **all percentiles computed on the same
  common positions** (the only change vs. the legacy full-frame helpers is the
  sample support).  Degenerate-denominator handling is preserved verbatim.

Neutral fallbacks
-----------------
* fewer than ``DEFAULT_MIN_OVERLAP_SAMPLES`` common valid positions, or a
  missing/degenerate geometry, or no valid samples -> deterministic neutral
  correction (``sky_mean`` offset ``0``; ``linear_fit`` identity ``a=1,b=0``)
  with a stable structured reason.  There is **no full-canvas legacy
  fallback**: an unavailable/insufficient support never silently reverts to
  whole-frame statistics.

All functions are deterministic pure functions of their inputs (no RNG, no
global state, no file I/O).  ``cv2`` is only used for the geometry/content
warp helpers which must reproduce the production warp bit-for-bit; the
estimators themselves are plain NumPy.
"""

from __future__ import annotations

import numpy as np

from .drizzle_background import (
    DEFAULT_CLIP_ITERATIONS,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_OVERLAP_SAMPLES,
    DEFAULT_SIGMA_CLIP,
    REASON_ACCEPTED,
    REASON_DEGENERATE_GEOMETRY,
    REASON_INSUFFICIENT_OVERLAP,
    REASON_NO_VALID_SAMPLES,
    robust_location,
)

__all__ = [
    # verbatim Drizzle reuse (single implementation, no competing estimator)
    "robust_location",
    "DEFAULT_MIN_OVERLAP_SAMPLES",
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_SIGMA_CLIP",
    "DEFAULT_CLIP_ITERATIONS",
    # stable reason vocabulary
    "REASON_ACCEPTED",
    "REASON_INSUFFICIENT_OVERLAP",
    "REASON_NO_VALID_SAMPLES",
    "REASON_DEGENERATE_GEOMETRY",
    "REASON_NO_GEOMETRY",
    "REASON_NO_SOURCE_CONTENT",
    "REASON_NO_REFERENCE_CONTENT",
    # documented boundary/validity policy
    "GEOM_SUPPORT_TOL",
    "GEOM_ERODE_PX",
    "CONTENT_WARP_TOL",
    "DEBAYER_INFLUENCE_PX",
    # helpers / estimators
    "luminance",
    "geometry_support_mask_or_none",
    "estimate_sky_mean_from_geometry",
    "estimate_linear_fit_from_geometry",
    "warp_synthetic_detector",
    "geometry_support_mask",
    "warp_content_mask",
    "content_valid_canvas",
    "content_validity_after_loader",
    "effective_common_support",
    "paired_common_positions",
    "estimate_sky_mean_offset",
    "estimate_linear_fit",
]

# Extra reason: geometry (M) completely absent.  Kept next to the verbatim
# Drizzle vocabulary so diagnostics share one style and stay parseable.
REASON_NO_GEOMETRY = "no_geometry"
REASON_NO_SOURCE_CONTENT = "no_source_content_validity"
REASON_NO_REFERENCE_CONTENT = "no_reference_content_validity"

# ---------------------------------------------------------------------------
# Documented boundary / validity policy constants
# ---------------------------------------------------------------------------
GEOM_SUPPORT_TOL = 1e-6   # ones-warp >= 1 - tol  => all four taps in detector
GEOM_ERODE_PX = 1         # one 3x3 erosion pass after the threshold
CONTENT_WARP_TOL = 1e-6   # content mask warp == 1 within tol => all taps valid
DEBAYER_INFLUENCE_PX = 1  # conservative CFA->RGB influence radius


# ---------------------------------------------------------------------------
# Pure colour helpers
# ---------------------------------------------------------------------------
def luminance(img):
    """Rec.709 luminance of an ``(H, W)`` or ``(H, W, C)`` float array.

    For monochrome input the array itself is returned (no copy).  For RGB the
    classic ``0.299 R + 0.587 G + 0.114 B`` coefficients used by the legacy
    Classic sky helpers are applied so the new estimator speaks the same
    photometric language as the code it replaces.
    """
    a = np.asarray(img)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] == 3:
        return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    raise ValueError("luminance expects (H,W) or (H,W,3)")


# ---------------------------------------------------------------------------
# Synthetic-detector geometry helpers (same warp as Classic alignment)
# ---------------------------------------------------------------------------
def _warp_2d(data2d, M, dsize):
    """cv2 affine warp with the EXACT Classic production semantics.

    ``M`` maps ORIGINAL pixels -> reference-grid pixels (2x3 float64).
    ``dsize = (canvas_w, canvas_h)``.  Border is ``BORDER_CONSTANT`` with
    value ``NaN`` and interpolation is ``INTER_LINEAR``, mirroring
    ``SeestarAligner._align_cpu``/``_align_image`` so NaN-ring contamination
    (including zero-weight out-of-bounds taps) is reproduced bit-for-bit.
    """
    import cv2

    return cv2.warpAffine(
        data2d,
        np.asarray(M, dtype=np.float64),
        dsize,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )


def _validate_geometry(M):
    """Validate a 2x3 affine geometry ``M`` for support computation.

    Returns ``None`` when ``M`` is a valid, finite, non-degenerate 2x3 affine;
    otherwise returns a stable machine-readable reason string
    (``REASON_NO_GEOMETRY`` / ``REASON_DEGENERATE_GEOMETRY``).  Missing
    geometry must never be silently replaced by an identity/full-support guess.
    """
    if M is None:
        return REASON_NO_GEOMETRY
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (2, 3):
        return REASON_NO_GEOMETRY
    if not np.all(np.isfinite(M)):
        return REASON_NO_GEOMETRY
    a = M[:2, :2]
    with np.errstate(over="ignore", invalid="ignore"):
        det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    if not np.isfinite(det) or abs(det) <= 1e-12:
        return REASON_DEGENERATE_GEOMETRY
    return None


def warp_synthetic_detector(src_shape_hw, M, canvas_shape_hw):
    """Warp an all-ones detector of ``src_shape_hw`` with ``M`` onto the canvas.

    Returns the raw warp (float32 ``(H, W)`` canvas): finite pixels are
    exactly ``1.0`` in the interior, ``NaN`` where the warp had no source
    contribution or hit the NaN-border contamination ring.

    Raises
    ------
    ValueError
        When ``M`` is missing, malformed, non-finite or singular (geometry
        must be validated before use; callers that need a neutral path should
        call :func:`geometry_support_mask_or_none` instead of guessing).
    """
    reason = _validate_geometry(M)
    if reason is not None:
        raise ValueError(f"invalid geometry for support warp: {reason}")
    h, w = int(src_shape_hw[0]), int(src_shape_hw[1])
    ones = np.ones((h, w), dtype=np.float32)
    canvas_w, canvas_h = int(canvas_shape_hw[1]), int(canvas_shape_hw[0])
    return _warp_2d(ones, M, (canvas_w, canvas_h))


def geometry_support_mask(src_shape_hw, M, canvas_shape_hw):
    """Deterministic conservative geometric support of a source on the canvas.

    ``True`` where the synthetic detector warp is finite and ``>= 1 - tol``
    (all bilinear taps inside the source detector, NaN ring excluded) and the
    mask survives one 3x3 erosion pass (documented reliability margin).

    Raises ``ValueError`` when ``M`` is missing/malformed/non-finite/singular
    (see :func:`warp_synthetic_detector`); the neutral caller variant is
    :func:`geometry_support_mask_or_none`.

    Pure function: ``(bool HxW canvas mask)``.
    """
    w = warp_synthetic_detector(src_shape_hw, M, canvas_shape_hw)
    mask = np.isfinite(w) & (w >= 1.0 - GEOM_SUPPORT_TOL)
    if GEOM_ERODE_PX > 0 and mask.any():
        import cv2

        mask = cv2.erode(
            mask.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=GEOM_ERODE_PX,
        ).astype(bool)
    return mask


def geometry_support_mask_or_none(src_shape_hw, M, canvas_shape_hw):
    """Neutral geometry-support variant returning ``(mask, reason)``.

    ``mask`` is ``None`` (with a stable reason) when the geometry is missing,
    malformed, non-finite or singular — never an identity/full-support guess.
    """
    reason = _validate_geometry(M)
    if reason is not None:
        return None, reason
    try:
        return geometry_support_mask(src_shape_hw, M, canvas_shape_hw), REASON_ACCEPTED
    except ValueError as exc:
        return None, str(exc)


def content_validity_after_loader(invalid_before_repair, bayer=False):
    """Turn a loader opt-in invalidity report into a content-valid bool mask.

    ``invalid_before_repair`` is the boolean mask of samples that were
    non-finite *before* the loader repaired them (same spatial layout as the
    raw data, i.e. CFA 2D or RGB).  Returns a ``bool (H, W)`` content-valid
    mask with conservative Bayer influence already applied:

    * RGB input (``bayer=False``): a pixel is content-valid iff **no** channel
      sample at that position was originally non-finite.  No spatial dilation
      is applied (the loader repair is per-sample in an already-debayered
      RGB layout).
    * CFA input (``bayer=True``): the invalid mask is first dilated by
      ``DEBAYER_INFLUENCE_PX`` (one 3x3 pass, documented conservative
      CFA->RGB influence radius of a repaired sample), then inverted.
      Debayer science itself is never modified.

    Pure function (``cv2`` used only for the dilation).
    """
    inv = np.asarray(invalid_before_repair, dtype=bool)
    if inv.ndim == 3:
        inv = inv.any(axis=-1)  # any-channel invalidity invalidates the pixel
    if inv.ndim != 2:
        raise ValueError("invalidity mask must be 2D or 3D (H,W,C)")
    if bayer and DEBAYER_INFLUENCE_PX > 0 and inv.any():
        import cv2

        k = 2 * DEBAYER_INFLUENCE_PX + 1
        inv = cv2.dilate(
            inv.astype(np.uint8), np.ones((k, k), np.uint8)
        ).astype(bool)
    return ~inv


def warp_content_mask(content_mask_01, M, canvas_shape_hw):
    """Warp a source-frame 0/1 content mask with ``M`` (bilinear, NaN border).

    ``content_mask_01`` is ``(H_src, W_src)`` float32 with ``1`` == valid.
    Returns the canvas warp: value ``1`` (within ``CONTENT_WARP_TOL``) and
    finite means *every* contributing source tap was content-valid.
    """
    m = np.asarray(content_mask_01, dtype=np.float32)
    canvas_w, canvas_h = int(canvas_shape_hw[1]), int(canvas_shape_hw[0])
    return _warp_2d(m, M, (canvas_w, canvas_h))


def content_valid_canvas(content_mask_01, M, canvas_shape_hw):
    """Canvas bool mask: pixel content-valid iff all bilinear taps are valid.

    Equivalently the content-mask warp is finite and ``>= 1 - tol``; NaN
    (out-of-canvas / NaN-ring) positions are automatically excluded.  The same
    erosion policy as :func:`geometry_support_mask` is applied so both masks
    share one conservative boundary convention.
    """
    w = warp_content_mask(content_mask_01, M, canvas_shape_hw)
    mask = np.isfinite(w) & (w >= 1.0 - CONTENT_WARP_TOL)
    if GEOM_ERODE_PX > 0 and mask.any():
        import cv2

        mask = cv2.erode(
            mask.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=GEOM_ERODE_PX,
        ).astype(bool)
    return mask


# ---------------------------------------------------------------------------
# Common-position selection
# ---------------------------------------------------------------------------
def effective_common_support(
    src_geom, src_content, ref_content, src_finite=None, ref_finite=None
):
    """Effective support mask: geometric AND source-content AND ref-content.

    ``src_finite``/``ref_finite`` (optional bool canvas masks) additionally
    require the actual pixel values to be finite — content validity already
    covers loader-repaired samples, but callers may want to exclude any
    remaining non-finite value explicitly.
    """
    eff = src_geom & src_content & ref_content
    if src_finite is not None:
        eff = eff & np.asarray(src_finite, dtype=bool)
    if ref_finite is not None:
        eff = eff & np.asarray(ref_finite, dtype=bool)
    return eff


def _sample_positions(effective, max_samples):
    """Deterministic bounded positions inside ``effective``.

    Counts valid pixels first (cheap scalar), then materialises coordinates
    only when the count already fits the budget; large overlaps use a
    deterministic regular stride without ever allocating full-frame index
    pairs.  ``max_samples`` must be a finite positive int (validated) so a
    zero/negative/NaN budget can never loop forever.

    Returns ``(yy, xx)`` int64 arrays of canvas pixel positions.
    """
    if not np.isfinite(max_samples) or int(max_samples) <= 0:
        raise ValueError("max_samples must be a finite positive integer")
    max_samples = int(max_samples)
    n_eff = int(np.count_nonzero(effective))
    if n_eff == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    if n_eff <= max_samples:
        return np.nonzero(effective)
    # Deterministic stride grid: iterate strides until the sampled count is
    # within the budget (same rule as the Drizzle estimator) and only then
    # build bounded coordinate arrays.
    h, w = effective.shape
    stride = 1
    while True:
        ny = (h + stride - 1) // stride
        nx = (w + stride - 1) // stride
        if ny * nx <= max_samples:
            break
        stride += 1
    ys = np.arange(0, h, stride, dtype=np.int64)
    xs = np.arange(0, w, stride, dtype=np.int64)
    sel = effective[np.ix_(ys, xs)]
    yy_s, xx_s = np.nonzero(sel)
    return ys[yy_s], xs[xx_s]


def paired_common_positions(
    src_geom,
    src_content,
    ref_content,
    src_finite=None,
    ref_finite=None,
    max_samples=DEFAULT_MAX_SAMPLES,
):
    """Return ``(yy, xx, n_overlap, total_effective)`` for the common support.

    Only positions where the source is geometrically supported AND
    content-valid on both sides are kept; sampling is deterministic and
    bounded by ``max_samples`` (Drizzle budget).  ``total_effective`` is the
    real (unsampled) overlap count, ``n_overlap`` the count actually used for
    the estimate (capped by the sampling budget, mirroring Drizzle).
    """
    if not np.isfinite(max_samples) or int(max_samples) <= 0:
        raise ValueError("max_samples must be a finite positive integer")
    eff = effective_common_support(
        src_geom, src_content, ref_content, src_finite, ref_finite
    )
    n_eff = int(np.count_nonzero(eff))
    yy, xx = _sample_positions(eff, int(max_samples))
    return yy, xx, int(yy.size), n_eff


# ---------------------------------------------------------------------------
# sky_mean estimator (scalar luminance paired difference, robust location)
# ---------------------------------------------------------------------------
def estimate_sky_mean_offset(
    src_canvas,
    ref_canvas,
    src_geom,
    src_content,
    ref_content,
    min_overlap_samples=DEFAULT_MIN_OVERLAP_SAMPLES,
    max_samples=DEFAULT_MAX_SAMPLES,
    sigma_clip=DEFAULT_SIGMA_CLIP,
    clip_iterations=DEFAULT_CLIP_ITERATIONS,
):
    """Scalar sky offset from paired luminance ``I - R`` at common positions.

    Parameters
    ----------
    src_canvas : ndarray
        Aligned source on the reference canvas, ``(H, W)`` or ``(H, W, C)``.
    ref_canvas : ndarray
        Immutable reference on the same canvas (same shape as ``src_canvas``).
    src_geom, src_content, ref_content : bool canvas masks (see module docs).

    Returns
    -------
    offset : float
        Scalar to SUBTRACT from every channel of the source (``0.0`` when the
        overlap is insufficient / geometry absent -> neutral).
    diagnostics : dict
        Bounded structured diagnostics (reason, counts, fractions, used).
    """
    src = np.asarray(src_canvas, dtype=np.float32)
    ref = np.asarray(ref_canvas, dtype=np.float32)
    src_lum = luminance(src)
    ref_lum = luminance(ref)
    if src_lum.shape != ref_lum.shape:
        raise ValueError("src/ref canvas shapes differ")
    if src_lum.ndim != 2:
        raise ValueError("src/ref canvas must be 2D or 3D(H,W,C)")

    src_finite = np.isfinite(src_lum)
    ref_finite = np.isfinite(ref_lum)
    yy, xx, n_overlap, n_eff = paired_common_positions(
        np.asarray(src_geom, dtype=bool),
        np.asarray(src_content, dtype=bool),
        np.asarray(ref_content, dtype=bool),
        src_finite=src_finite,
        ref_finite=ref_finite,
        max_samples=max_samples,
    )
    # Diagnostic-only scalar evidence (never a science input): the number of
    # canvas pixels inside the eroded geometry-support mask, attached while
    # the mask is alive.  It is NOT retained past this function and never
    # changes sample selection or the estimator math.
    n_geometric = int(np.count_nonzero(np.asarray(src_geom, dtype=bool)))
    diag = {
        "reason": REASON_NO_VALID_SAMPLES,
        "n_overlap": n_overlap,
        "n_effective": n_eff,
        "n_geometric": n_geometric,
        "n_used": 0,
        "offset": 0.0,
        "estimator": "drizzle_robust_location",
        "min_overlap_samples": int(min_overlap_samples),
        "method": "sky_mean",
    }
    if n_overlap < max(1, int(min_overlap_samples)):
        diag["reason"] = REASON_INSUFFICIENT_OVERLAP
        return 0.0, diag
    delta = (src_lum[yy, xx] - ref_lum[yy, xx]).astype(np.float64)
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        diag["reason"] = REASON_NO_VALID_SAMPLES
        return 0.0, diag
    offset, used = robust_location(
        delta, sigma=sigma_clip, iterations=clip_iterations
    )
    diag["offset"] = float(offset)
    diag["n_used"] = int(used)
    diag["reason"] = REASON_ACCEPTED
    return float(offset), diag


def apply_sky_mean_offset(src_canvas, offset):
    """Subtract a scalar offset from every channel (float32, non-mutating).

    The offset is the same scalar across RGB (single-luminance sky model).
    Returns a new float32 array; the input is never modified (read-only and
    memmap-safe callers included).
    """
    src = np.asarray(src_canvas)
    out = np.array(src, dtype=np.float32, copy=True)
    if offset == 0.0:
        return out
    if out.ndim == 2:
        out -= np.float32(offset)
    else:
        out -= np.float32(offset)
    return out


# ---------------------------------------------------------------------------
# linear_fit estimator (legacy per-channel P25/P90 model, common positions)
# ---------------------------------------------------------------------------
def estimate_linear_fit(
    src_canvas,
    ref_canvas,
    src_geom,
    src_content,
    ref_content,
    min_overlap_samples=DEFAULT_MIN_OVERLAP_SAMPLES,
    max_samples=DEFAULT_MAX_SAMPLES,
):
    """Per-channel P25/P90 linear fit on the SAME common positions.

    The model is verbatim the legacy helper formula
    (``core/normalization.py``):

    * ``delta_src = src_high - src_low``, ``delta_ref = ref_high - ref_low``
    * ``a = where(delta_src > 1e-5, delta_ref / max(delta_src, 1e-9), 1.0)``
    * ``b = ref_low - a * src_low``
    * correction ``out = a * src + b`` (per channel).

    The only change is the sample support: every percentile is computed over
    the common reliable positions instead of the full visible frame.  If the
    common support is insufficient the correction is explicitly neutral
    (``a=1, b=0``) with ``REASON_INSUFFICIENT_OVERLAP`` — never a full-frame
    fallback.

    Returns ``((a, b), diagnostics)`` where ``a``/``b`` are per-channel
    float64 arrays (``(C,)``; a scalar pair for mono is still returned as
    1-element arrays for a uniform caller contract).
    """
    src = np.asarray(src_canvas, dtype=np.float32)
    ref = np.asarray(ref_canvas, dtype=np.float32)
    if src.shape != ref.shape:
        raise ValueError("src/ref canvas shapes differ")
    is_color = src.ndim == 3 and src.shape[2] == 3
    if src.ndim == 2:
        src_3 = src[..., None]
        ref_3 = ref[..., None]
    elif is_color:
        src_3 = src
        ref_3 = ref
    else:
        raise ValueError("src/ref canvas must be 2D or 3D(H,W,C)")
    n_ch = src_3.shape[2]

    src_finite = np.all(np.isfinite(src_3), axis=-1)
    ref_finite = np.all(np.isfinite(ref_3), axis=-1)
    yy, xx, n_overlap, n_eff = paired_common_positions(
        np.asarray(src_geom, dtype=bool),
        np.asarray(src_content, dtype=bool),
        np.asarray(ref_content, dtype=bool),
        src_finite=src_finite,
        ref_finite=ref_finite,
        max_samples=max_samples,
    )
    # Diagnostic-only scalar evidence (never a science input): geometric
    # support pixel count attached while the mask is alive; never retained.
    n_geometric = int(np.count_nonzero(np.asarray(src_geom, dtype=bool)))
    diag = {
        "reason": REASON_NO_VALID_SAMPLES,
        "n_overlap": n_overlap,
        "n_effective": n_eff,
        "n_geometric": n_geometric,
        "estimator": "percentile_p25_p90",
        "min_overlap_samples": int(min_overlap_samples),
        "method": "linear_fit",
    }
    if n_overlap < max(1, int(min_overlap_samples)):
        diag["reason"] = REASON_INSUFFICIENT_OVERLAP
        a = np.ones(n_ch, dtype=np.float64)
        b = np.zeros(n_ch, dtype=np.float64)
        diag["a"] = [float(v) for v in a]
        diag["b"] = [float(v) for v in b]
        return (a, b), diag

    a = np.ones(n_ch, dtype=np.float64)
    b = np.zeros(n_ch, dtype=np.float64)
    for ch in range(n_ch):
        src_ch = src_3[yy, xx, ch].astype(np.float64)
        ref_ch = ref_3[yy, xx, ch].astype(np.float64)
        src_ch = src_ch[np.isfinite(src_ch)]
        ref_ch = ref_ch[np.isfinite(ref_ch)]
        if src_ch.size == 0 or ref_ch.size == 0:
            # Cannot be reached when src/ref finite masks above are used, but
            # keep the estimator total on pathological NaN-only inputs.
            diag["reason"] = REASON_NO_VALID_SAMPLES
            a[ch], b[ch] = 1.0, 0.0
            continue
        src_low = float(np.percentile(src_ch, 25.0))
        src_high = float(np.percentile(src_ch, 90.0))
        ref_low = float(np.percentile(ref_ch, 25.0))
        ref_high = float(np.percentile(ref_ch, 90.0))
        delta_src = src_high - src_low
        delta_ref = ref_high - ref_low
        # Verbatim legacy model (np.where form preserved, degenerate handled).
        a_ch = float(
            np.where(delta_src > 1e-5, delta_ref / max(delta_src, 1e-9), 1.0)
        )
        b_ch = ref_low - a_ch * src_low
        a[ch] = a_ch
        b[ch] = b_ch
    diag["a"] = [float(v) for v in a]
    diag["b"] = [float(v) for v in b]
    diag["reason"] = REASON_ACCEPTED
    return (a, b), diag


def apply_linear_fit(src_canvas, a, b):
    """Apply the per-channel linear fit (float32, non-mutating).

    ``a``/``b`` are per-channel float64 arrays broadcast over the last axis.
    Returns a new float32 array; the input is never modified.
    """
    src = np.asarray(src_canvas)
    out = np.array(src, dtype=np.float32, copy=True)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if out.ndim == 2:
        if a.size:
            out = out * np.float32(a[0]) + np.float32(b[0])
    else:
        out = out * a.reshape((1, 1, -1)) + b.reshape((1, 1, -1))
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Geometry-level seam estimators (mask derivation + neutral on bad geometry)
# ---------------------------------------------------------------------------
def _neutral_sky_diag(reason, method="sky_mean"):
    # Geometric support is explicitly UNKNOWN (never fabricated as 0 or as
    # all-valid) whenever the neutral path never derived a geometry mask.
    return {
        "reason": reason,
        "n_overlap": 0,
        "n_effective": 0,
        "n_geometric": None,
        "n_used": 0,
        "offset": 0.0,
        "estimator": "drizzle_robust_location",
        "min_overlap_samples": int(DEFAULT_MIN_OVERLAP_SAMPLES),
        "method": method,
    }


def _neutral_linear_diag(reason, n_ch, method="linear_fit"):
    a = np.ones(max(1, int(n_ch)), dtype=np.float64)
    b = np.zeros(max(1, int(n_ch)), dtype=np.float64)
    return (a, b), {
        "reason": reason,
        "n_overlap": 0,
        "n_effective": 0,
        "n_geometric": None,  # geometry never derived -> explicitly unknown
        "estimator": "percentile_p25_p90",
        "min_overlap_samples": int(DEFAULT_MIN_OVERLAP_SAMPLES),
        "method": method,
        "a": [float(v) for v in a],
        "b": [float(v) for v in b],
    }


def estimate_sky_mean_from_geometry(
    src_canvas,
    ref_canvas,
    src_shape_hw,
    M,
    src_content_mask_01=None,
    ref_content_mask=None,
    min_overlap_samples=DEFAULT_MIN_OVERLAP_SAMPLES,
    max_samples=DEFAULT_MAX_SAMPLES,
    sigma_clip=DEFAULT_SIGMA_CLIP,
    clip_iterations=DEFAULT_CLIP_ITERATIONS,
):
    """Seam-level sky_mean: derive geometry/content masks from ``M`` then estimate.

    The caller supplies the aligned source canvas, the immutable reference
    canvas, the ORIGINAL (pre-warp) source shape, the 2x3 affine ``M``, the
    source-frame content-validity 0/1 mask (loader report, conservative Bayer
    influence) and the reference content-validity mask.  Missing content
    evidence is NEVER replaced by an all-valid guess: when either content mask
    is omitted the result is explicitly neutral (``offset 0.0``) with a stable
    reason, and the same applies to missing/malformed/non-finite/singular
    geometry.
    """
    if src_content_mask_01 is None:
        return 0.0, _neutral_sky_diag(REASON_NO_SOURCE_CONTENT)
    if ref_content_mask is None:
        return 0.0, _neutral_sky_diag(REASON_NO_REFERENCE_CONTENT)
    geom, reason = geometry_support_mask_or_none(src_shape_hw, M, src_canvas.shape[:2])
    if geom is None:
        return 0.0, _neutral_sky_diag(reason)
    content = content_valid_canvas(src_content_mask_01, M, src_canvas.shape[:2])
    return estimate_sky_mean_offset(
        src_canvas,
        ref_canvas,
        geom,
        content,
        np.asarray(ref_content_mask, dtype=bool),
        min_overlap_samples=min_overlap_samples,
        max_samples=max_samples,
        sigma_clip=sigma_clip,
        clip_iterations=clip_iterations,
    )


def estimate_linear_fit_from_geometry(
    src_canvas,
    ref_canvas,
    src_shape_hw,
    M,
    src_content_mask_01=None,
    ref_content_mask=None,
    min_overlap_samples=DEFAULT_MIN_OVERLAP_SAMPLES,
    max_samples=DEFAULT_MAX_SAMPLES,
):
    """Seam-level linear_fit: derive geometry/content masks from ``M`` then estimate.

    Same neutral-on-bad-geometry contract as
    :func:`estimate_sky_mean_from_geometry` (identity correction ``a=1, b=0``
    with a stable reason when geometry or content evidence is unavailable).
    Missing source/reference content masks are never replaced by all-valid
    guesses.
    """
    if src_content_mask_01 is None:
        n_ch = 3 if (np.asarray(src_canvas).ndim == 3) else 1
        return _neutral_linear_diag(REASON_NO_SOURCE_CONTENT, n_ch)
    if ref_content_mask is None:
        n_ch = 3 if (np.asarray(src_canvas).ndim == 3) else 1
        return _neutral_linear_diag(REASON_NO_REFERENCE_CONTENT, n_ch)
    geom, reason = geometry_support_mask_or_none(src_shape_hw, M, src_canvas.shape[:2])
    if geom is None:
        n_ch = 3 if (np.asarray(src_canvas).ndim == 3) else 1
        return _neutral_linear_diag(reason, n_ch)
    content = content_valid_canvas(src_content_mask_01, M, src_canvas.shape[:2])
    return estimate_linear_fit(
        src_canvas,
        ref_canvas,
        geom,
        content,
        np.asarray(ref_content_mask, dtype=bool),
        min_overlap_samples=min_overlap_samples,
        max_samples=max_samples,
    )
