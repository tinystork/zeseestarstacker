"""Scientific drizzle core.

Rebuild of the Seestar drizzle stacking on a sound scientific basis:

* original (non-resampled) input pixels are kept untouched,
* the geometric alignment is expressed as a 2x3 affine ``tf`` (rotation +
  translation, scale 1.0) composed into the reference WCS,
* a single drizzle accumulator is used per channel,
* a final normalization step produces the stacked image.

The memory policy (frame grouping / batching) is kept *outside* the
accumulation path and MUST NOT change the result (batch invariance).

Only ``numpy``, ``astropy.wcs`` and ``drizzle.resample`` are used (no
``cv2``, no ``scipy``, no ``cupy``).

.. note::
    ``drizzle`` 2.2.0 stores the *weighted mean* in ``out_img`` (the native
    intermediate image users consume) and the accumulated kernel *weight /
    count* map in ``out_wht`` (which may be **signed** for the Lanczos
    kernels).  :class:`DrizzleAccumulator` therefore reports the native
    ``out_img`` directly as the final science (``finalize("divide")``) and
    keeps the weighted flux ``out_img * out_wht`` only as *derived
    bookkeeping* in the ``sci`` property (exposure-scaled weight is folded in
    via ``wht_scale = expscale``).  A sample is valid only where its native
    science and native WHT are finite AND the WHT is strictly above
    :data:`WEIGHT_EPSILON`; invalid samples become ``0.0`` — never
    ``abs(wht)``, never a huge-value clip, never percentile hiding.
"""

from __future__ import annotations

import itertools

import numpy as np
from astropy.wcs import WCS
from drizzle.resample import Drizzle

__all__ = [
    "build_output_grid",
    "pixmap_from_alignment",
    "DrizzleAccumulator",
    "drizzle_stream",
    "support_integrity_violations",
    "VALID_DRIZZLE_KERNELS",
    "validate_drizzle_kernel",
    "validate_drizzle_pixfrac",
    "WEIGHT_EPSILON",
    "LANCZOS_KERNELS",
    "WhtThresholdResult",
    "wht_relative_threshold",
]

# Small positive weight floor below which a Drizzle output sample is considered
# *unsupported*.  Continuity-friendly (matches the historical ``drizzle_finalize``
# clip) and strictly greater than any signed-Lanczos near-zero/negative weight
# that must never be treated as physical coverage.  A sample is valid only when
# its native WHT is finite AND strictly above this epsilon.
WEIGHT_EPSILON = 1e-9

# Kernels accepted by the underlying ``drizzle.resample.Drizzle`` engine
# (drizzle 2.2.0).  ``square`` is the flux-conserving default.
VALID_DRIZZLE_KERNELS = frozenset(
    {"square", "gaussian", "point", "turbo", "lanczos2", "lanczos3"}
)

# Lanczos kernels: upstream drizzle 2.2.0 ignores ``pixfrac`` (assumed 1.0)
# and produces a *signed* native WHT (negative lobes near coverage edges).
LANCZOS_KERNELS = frozenset({"lanczos2", "lanczos3"})


def validate_drizzle_kernel(kernel):
    """Normalize a drizzle kernel name to a safe value (deterministic).

    Returns ``(kernel, reason)`` where ``kernel`` is a member of
    :data:`VALID_DRIZZLE_KERNELS` and ``reason`` is ``None`` on success or a
    short human-readable explanation of the fallback.  Unknown / non-string
    kernels fall back to ``"square"`` (the flux-conserving default) rather than
    failing the run, consistent with the existing settings-coercion conventions
    (a *well-logged* deterministic fallback, never a silent scientific change).
    """
    k = str(kernel or "square").strip().lower()
    if k in VALID_DRIZZLE_KERNELS:
        return k, None
    return "square", f"unknown drizzle kernel {kernel!r} -> 'square'"


def validate_drizzle_pixfrac(pixfrac):
    """Normalize a drizzle ``pixfrac`` to a safe value in ``(0, 1]``.

    Returns ``(pixfrac, reason)``.  NaN/Inf, non-numeric, non-positive or
    ``> 1`` values fall back to ``1.0`` (the flux-conserving point-pixel
    default) with a short explanation, matching the existing settings
    conventions (deterministic well-logged fallback).
    """
    try:
        p = float(pixfrac)
    except (TypeError, ValueError):
        return 1.0, f"non-numeric drizzle pixfrac {pixfrac!r} -> 1.0"
    if not np.isfinite(p) or p <= 0.0 or p > 1.0:
        return 1.0, f"drizzle pixfrac {pixfrac!r} outside (0, 1] -> 1.0"
    return p, None


def build_output_grid(reference_wcs, reference_shape_hw, scale):
    """Build the output WCS and grid shape for a drizzle scale factor.

    The output WCS keeps the same projection / CRVAL / CTYPE as the reference;
    ``CDELT`` is divided by ``scale`` and ``CRPIX`` is multiplied by ``scale``.

    Parameters
    ----------
    reference_wcs : `astropy.wcs.WCS`
        Reference WCS.
    reference_shape_hw : tuple of int
        Reference grid shape ``(height, width)``.
    scale : float
        Output scale factor (``>= 1.0``).

    Returns
    -------
    out_wcs : `astropy.wcs.WCS`
        Output WCS (``CDELT / scale``, ``CRPIX * scale``).
    out_shape_hw : tuple of int
        Output grid shape ``(round(H * scale), round(W * scale))``.
    """
    scale = float(scale)
    if scale < 1.0:
        raise ValueError("scale must be >= 1.0")

    out_wcs = reference_wcs.deepcopy()
    out_wcs.wcs.crpix = np.asarray(reference_wcs.wcs.crpix, dtype=float) * scale
    out_wcs.wcs.cdelt = np.asarray(reference_wcs.wcs.cdelt, dtype=float) / scale

    out_h = int(round(reference_shape_hw[0] * scale))
    out_w = int(round(reference_shape_hw[1] * scale))
    return out_wcs, (out_h, out_w)


def pixmap_from_alignment(data_shape_hw, tf, reference_wcs, output_wcs):
    """Map original input pixels to the output grid through the reference WCS.

    ``tf`` is a 2x3 affine mapping ORIGINAL pixel ``(x, y)`` to reference-grid
    pixel coordinates (rotation + translation, scale 1.0).  No ``warpAffine``
    is applied to the data: only the pixel centres are mapped.

    Parameters
    ----------
    data_shape_hw : tuple of int
        Input image shape ``(height, width)``.
    tf : array_like
        2x3 affine array.
    reference_wcs : `astropy.wcs.WCS`
    output_wcs : `astropy.wcs.WCS`

    Returns
    -------
    pixmap : ndarray (float64, shape ``(Ny, Nx, 2)``)
        Output ``(x, y)`` coordinate of every input pixel centre.
    in_grid_mask : ndarray (bool, shape ``(Ny, Nx)``)
        ``True`` where the mapped centre lies inside the output grid.  Pixels
        whose centre falls outside (or maps to NaN/Inf) are masked out so they
        can never contribute partial flux folded back onto the borders.
    """
    tf = np.asarray(tf, dtype=np.float64)
    if tf.shape != (2, 3):
        raise ValueError("tf must be a 2x3 affine array")

    height, width = data_shape_hw
    yy, xx = np.indices((height, width), dtype=np.float64)

    # p_ref = tf @ [x, y, 1]
    px = tf[0, 0] * xx + tf[0, 1] * yy + tf[0, 2]
    py = tf[1, 0] * xx + tf[1, 1] * yy + tf[1, 2]

    sky_ra, sky_dec = reference_wcs.all_pix2world(px, py, 0)
    out_x, out_y = output_wcs.all_world2pix(sky_ra, sky_dec, 0)

    if output_wcs.array_shape is not None:
        out_h, out_w = output_wcs.array_shape
    else:
        out_w, out_h = output_wcs.pixel_shape

    # A tiny tolerance absorbs the ~1e-11 round-trip error of the WCS
    # transform so that pixels mapped exactly to the grid origin are not
    # spuriously masked.  The upper bound stays strict: a centre mapped
    # exactly to ``out_w`` (or ``out_h``) is outside the grid.
    eps = 1e-9
    finite = np.isfinite(out_x) & np.isfinite(out_y)
    in_grid_mask = (
        finite
        & (out_x >= -eps)
        & (out_x < out_w)
        & (out_y >= -eps)
        & (out_y < out_h)
    )

    # Replace only non-finite coordinates with 0.0 (they are masked out via
    # ``in_grid_mask`` anyway).  Out-of-grid *finite* coordinates are kept
    # as-is: setting them to an in-grid value (e.g. 0.0) makes the drizzle
    # C kernel drop the whole frame's flux, and huge offsets are likewise
    # unsafe, so the original out-of-grid values are the safest choice.
    out_x = np.where(finite, out_x, 0.0)
    out_y = np.where(finite, out_y, 0.0)

    pixmap = np.dstack((out_x, out_y)).astype(np.float64)
    return pixmap, in_grid_mask


class DrizzleAccumulator:
    """A single-channel drizzle accumulator wrapping
    :class:`drizzle.resample.Drizzle`.

    Parameters
    ----------
    out_shape_hw : tuple of int
        Output grid shape ``(height, width)``.
    kernel : str, optional
        Drizzle kernel (default ``"square"``).
    pixfrac : float, optional
        Pixel fraction, passed to ``add_image`` (not to the constructor).
    fillval : str or float, optional
        Fill value for uncovered output pixels (default ``"0.0"``).
    """

    def __init__(self, out_shape_hw, kernel="square", pixfrac=1.0, fillval="0.0"):
        self.out_shape_hw = tuple(int(v) for v in out_shape_hw)
        self.kernel = kernel
        self.pixfrac = float(pixfrac)

        self._out_img = np.zeros(self.out_shape_hw, dtype=np.float32)
        self._out_wht = np.zeros(self.out_shape_hw, dtype=np.float32)
        self._drizzle = Drizzle(
            out_img=self._out_img,
            out_wht=self._out_wht,
            kernel=kernel,
            fillval=fillval,
        )

    @property
    def sci(self):
        """Accumulated *weighted flux* ``out_img * out_wht`` (derived bookkeeping).

        This is **not** the native final science.  ``drizzle`` 2.2.0 keeps the
        weighted *mean* in ``out_img`` and the total weight in ``out_wht``; their
        product is the weighted flux.  ``finalize("divide")`` returns the native
        ``out_img`` directly (see :meth:`finalize`), not this derived quantity.
        Retained for compatibility / bookkeeping only.
        """
        return (self._out_img * self._out_wht).astype(np.float32)

    @property
    def wht(self):
        """Native accumulated weight/count map (exposure-scaled), as a copy.

        This is the **native signed** mathematical Drizzle WHT.  For the Lanczos
        kernels it may contain negative values near coverage edges; callers must
        not treat it as a positive physical coverage map.  No clipping is applied.
        """
        return self._out_wht.copy()

    def add(self, data, weight_map, pixmap, exptime=1.0, in_units="counts",
            in_grid_mask=None):
        """Add one 2D channel image to the accumulator.

        Pixels whose centre is outside the output grid are given zero weight
        (``weight_map * in_grid_mask``) so they cannot contribute.

        The exposure scale (``exptime`` for ``"counts"``, ``1.0`` for
        ``"cps"``) is folded into the weight via ``wht_scale`` so that the
        final ``sci / wht`` ratio is the exposure-weighted mean of the rate.
        """
        data = np.asarray(data, dtype=np.float32)
        weight_map = np.asarray(weight_map, dtype=np.float32)

        if in_grid_mask is not None:
            weight_map = weight_map * np.asarray(in_grid_mask, dtype=np.float32)

        expscale = exptime if in_units == "counts" else 1.0

        self._drizzle.add_image(
            data=data,
            exptime=exptime,
            pixmap=pixmap,
            weight_map=weight_map,
            in_units=in_units,
            pixfrac=self.pixfrac,
            wht_scale=expscale,
        )

    def finalize(self, mode="divide"):
        """Normalise the accumulated image.

        Mirrors ``seestar.core.drizzle_utils.drizzle_finalize`` (no GPU):

        * ``"divide"``   -> native ``out_img`` on valid support, ``0`` elsewhere
        * ``"none"``     -> weighted flux ``out_img * out_wht`` on valid support
        * ``"max"``      -> native ``out_img * max(valid wht)``
        * ``"n_images"`` -> native ``out_img * mean(valid wht)``

        A sample is *valid* only when its native science AND native WHT are
        finite AND the WHT is strictly above :data:`WEIGHT_EPSILON`.  Invalid
        samples become ``0.0`` (the established final-FITS contract) — never
        ``abs(wht)``, never a huge-value clip, never percentile hiding.  The
        result is a *private* finite ``float32`` copy: the engine accumulation
        buffers are never mutated.
        """
        if mode not in {"divide", "none", "max", "n_images"}:
            mode = "divide"

        # Private float32 copies — never mutate the engine's accumulation buffers.
        native = np.array(self._out_img, dtype=np.float32, copy=True)
        wht = np.array(self._out_wht, dtype=np.float32, copy=False)

        finite_native = np.isfinite(native)
        finite_wht = np.isfinite(wht)
        valid = finite_native & finite_wht & (wht > WEIGHT_EPSILON)

        if mode == "none":
            # Derived bookkeeping (weighted flux), gated by the same support.
            result = np.where(valid, native * wht, 0.0).astype(np.float32)
        else:
            # Native weighted-mean science preserved directly on valid support.
            result = np.where(valid, native, 0.0).astype(np.float32)
            if mode == "max":
                result *= float(np.max(wht[valid])) if np.any(valid) else 0.0
            elif mode == "n_images":
                result *= float(np.mean(wht[valid])) if np.any(valid) else 0.0

        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _resolve_pixmap(pixmap_or_tf, data_shape_hw, reference_wcs, output_wcs):
    """Return a ``(pixmap, in_grid_mask)`` pair from a pixmap or a 2x3 tf.

    When a pixmap is supplied the mask is ``None`` (the caller may provide it
    separately); when a ``tf`` is supplied both the pixmap and the mask are
    computed.
    """
    arr = np.asarray(pixmap_or_tf)
    if arr.ndim == 2 and arr.shape == (2, 3):
        if reference_wcs is None or output_wcs is None:
            raise ValueError(
                "reference_wcs and output_wcs are required when a tf is provided"
            )
        return pixmap_from_alignment(data_shape_hw, arr, reference_wcs, output_wcs)
    return arr, None


def support_integrity_violations(sci_hwc, wht_hwc, epsilon=WEIGHT_EPSILON):
    """Report samples whose science is nonzero on *invalid* native WHT support.

    M3 contract: a sample invalid by native WHT (not finite, or ``<= epsilon``)
    MUST map to a ``0.0`` science value — never a nonzero finite value, never
    NaN/Inf.  This is the automatic support-integrity gate (finite/support
    logic, *not* an arbitrary ADU limit).  It returns a bounded list of
    ``(channel, n_violations, max_abs_violation)`` tuples for channels that
    violate the invariant; an empty list means the invariant holds.

    Parameters
    ----------
    sci_hwc : ndarray
        Final science ``(H, W)`` or ``(H, W, C)`` (``finalize("divide")`` output).
    wht_hwc : ndarray
        Native signed WHT ``(H, W)`` or ``(H, W, C)`` (same shape as ``sci_hwc``).
    epsilon : float
        Support floor (:data:`WEIGHT_EPSILON`).
    """
    sci = np.asarray(sci_hwc, dtype=np.float32)
    wht = np.asarray(wht_hwc, dtype=np.float32)
    if sci.ndim == 2:
        sci = sci[..., None]
    if wht.ndim == 2:
        wht = wht[..., None]
    if sci.shape != wht.shape:
        raise ValueError("science and WHT shapes must match")

    violations = []
    n_channels = sci.shape[-1]
    for c in range(n_channels):
        w = wht[..., c]
        s = sci[..., c]
        invalid = ~(np.isfinite(w) & (w > epsilon))
        bad = invalid & (~np.isfinite(s) | (s != 0.0))
        n_bad = int(np.count_nonzero(bad))
        if n_bad:
            max_abs = float(np.max(np.abs(s[bad]))) if np.any(bad) else 0.0
            violations.append((c, n_bad, max_abs))
    return violations


def drizzle_stream(accumulators, frame_iter, group_size=1,
                   reference_wcs=None, output_wcs=None):
    """Stream frames through the per-channel accumulators.

    ``frame_iter`` yields tuples
    ``(data_cxhxw, weight_cxhxw, pixmap_or_tf, exptime, in_units)`` where the
    third element is either a precomputed pixmap ``(Ny, Nx, 2)`` or a 2x3
    affine ``tf`` (the latter requires ``reference_wcs`` and ``output_wcs``).

    An optional 6th element ``in_grid_mask`` (bool, ``(Ny, Nx)``) may be
    supplied to exclude pixels whose transformed centre falls outside the
    output grid.  When a ``tf`` is supplied instead of a pixmap, the mask is
    computed automatically.

    ``group_size`` only controls how many frames are preloaded from the
    iterator at once; the accumulation order (and therefore the result) is
    identical for every ``group_size``.

    Returns
    -------
    final_sci_hwc : ndarray (float32, ``(H, W, C)``)
        Normalised science image per channel.
    final_wht_hwc : ndarray (float32, ``(H, W, C)``)
        Accumulated weight image per channel.
    """
    accumulators = list(accumulators)
    n_channels = len(accumulators)
    if n_channels == 0:
        raise ValueError("at least one accumulator is required")

    group_size = max(1, int(group_size))
    iterator = iter(frame_iter)

    while True:
        group = list(itertools.islice(iterator, group_size))
        if not group:
            break
        for frame in group:
            data_cxhxw, weight_cxhxw, pixmap_or_tf, exptime, in_units = frame[:5]
            in_grid_mask = frame[5] if len(frame) > 5 else None
            data_cxhxw = np.asarray(data_cxhxw, dtype=np.float32)
            weight_cxhxw = np.asarray(weight_cxhxw, dtype=np.float32)

            if data_cxhxw.shape[0] != n_channels:
                raise ValueError(
                    f"frame has {data_cxhxw.shape[0]} channels but "
                    f"{n_channels} accumulators were provided"
                )

            pixmap, _mask = _resolve_pixmap(
                pixmap_or_tf, data_cxhxw.shape[1:], reference_wcs, output_wcs
            )
            if _mask is not None:
                in_grid_mask = _mask

            for ch in range(n_channels):
                accumulators[ch].add(
                    data_cxhxw[ch],
                    weight_cxhxw[ch],
                    pixmap,
                    exptime=exptime,
                    in_units=in_units,
                    in_grid_mask=in_grid_mask,
                )

    final_sci = np.stack([acc.finalize() for acc in accumulators], axis=-1)
    final_wht = np.stack([acc.wht for acc in accumulators], axis=-1)
    return final_sci.astype(np.float32), final_wht.astype(np.float32)


class WhtThresholdResult:
    """Result of applying the *relative* WHT threshold policy.

    The public ``WHT Threshold %`` (``0..1``) is a **coverage/support policy**,
    not a raw absolute weight: it is interpreted as a fraction of a robust
    high-support reference.  This small value object carries the reference
    support, the absolute cutoff, the resulting validity mask and bounded
    diagnostics for logging/observability.  It is deliberately *not* the
    photometric fix (background matching is).

    Attributes
    ----------
    fraction : float
        The requested relative threshold (``0..1``), clamped to ``[0, 1]``.
    tile_size : int
        Edge length (pixels) of the fixed-size square tile used to establish
        *spatial* support for the reference level.
    tile_support_min : int
        Minimum number of positive pixels a tile must contain before it can
        define a supported level (the ``tile_support_min``-th largest positive
        value within that tile).
    n_phase_offsets : int
        Number of deterministic half-tile phase positions *per axis* used to
        avoid tile-boundary sensitivity (``n_phase_offsets ** 2`` tile grids).
    reference_support : float
        Spatially supported robust maximum coverage level (block-supported
        upper reference), in the same units as the WHT map.
    cutoff : float
        Absolute cutoff ``= fraction * reference_support``.
    mask : ndarray (bool)
        ``True`` where the policy declares a pixel *valid* (positive weight
        AND ``>= cutoff``).  Zero-weight pixels are always invalid.
    masked_fraction : float
        Fraction of positive-weight pixels that are masked out (``0..1``).
    n_positive : int
        Number of positive finite WHT pixels.
    n_valid : int
        Number of pixels kept by the policy.
    reason : str
        ``"applied"``, ``"no_supported_tile"`` (degenerate) or
        ``"no_positive_weight"``.
    """

    __slots__ = (
        "fraction",
        "tile_size",
        "tile_support_min",
        "n_phase_offsets",
        "reference_support",
        "cutoff",
        "mask",
        "masked_fraction",
        "n_positive",
        "n_valid",
        "reason",
    )

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k))

    def to_dict(self):
        """JSON-safe, bounded summary (mask excluded)."""
        return {
            "fraction": float(self.fraction),
            "tile_size": int(self.tile_size),
            "tile_support_min": int(self.tile_support_min),
            "n_phase_offsets": int(self.n_phase_offsets),
            "reference_support": float(self.reference_support),
            "cutoff": float(self.cutoff),
            "masked_fraction": float(self.masked_fraction),
            "n_positive": int(self.n_positive),
            "n_valid": int(self.n_valid),
            "reason": self.reason,
        }


def _block_supported_reference(arr, positive, tile_size, tile_support_min,
                               n_phase_offsets):
    """Spatially supported (block) robust maximum of a positive WHT footprint.

    Partition the 2-D positive footprint into fixed-size ``tile_size`` squares
    under ``n_phase_offsets`` deterministic half-tile phase positions per axis
    (``n_phase_offsets ** 2`` grids).  A tile *supports* a level only when it
    contains at least ``tile_support_min`` positive pixels; its supported level
    is the ``tile_support_min``-th largest positive value within the tile.  The
    reference is the maximum supported level across all tiles and all phase
    grids.

    This is a genuine *spatial* support requirement — a level can only define
    the reference if ``tile_support_min`` positive pixels co-occur inside a
    compact ``tile_size`` square — unlike a global order statistic, which a
    spatially scattered population of outliers could otherwise satisfy.
    Sparse-positive kernels/pixfrac are handled by counting positive pixels
    within a tile (geometric neighbours are never required to be positive).

    Returns ``None`` when no tile reaches the minimum supported population.
    """
    h, w = arr.shape
    best = None
    phase_steps = [
        round(tile_size * i / n_phase_offsets) for i in range(n_phase_offsets)
    ]
    seen = set()
    for oy in phase_steps:
        for ox in phase_steps:
            if (oy, ox) in seen:
                continue
            seen.add((oy, ox))
            for ty in range(oy, h, tile_size):
                y1 = min(ty + tile_size, h)
                for tx in range(ox, w, tile_size):
                    x1 = min(tx + tile_size, w)
                    block_pos = positive[ty:y1, tx:x1]
                    n = int(np.count_nonzero(block_pos))
                    if n < tile_support_min:
                        continue
                    vals = arr[ty:y1, tx:x1][block_pos]
                    k = n - tile_support_min
                    level = float(np.partition(vals, k)[k])
                    if best is None or level > best:
                        best = level
    return best


def wht_relative_threshold(
    wht,
    fraction,
    *,
    tile_size=8,
    tile_support_min=4,
    n_phase_offsets=2,
):
    """Apply the relative WHT threshold policy to a final coverage map.

    The public ``WHT Threshold %`` (float ``0..1``) is interpreted as a
    **fraction of a documented per-finalization reference support**, not as a
    raw absolute weight.

    Reference support — *spatially supported robust maximum*
    ---------------------------------------------------------
    The reference support is the highest coverage level that is *spatially
    supported* by a compact block of the positive footprint (see
    :func:`_block_supported_reference`): the positive footprint is partitioned
    into fixed-size ``tile_size`` squares under ``n_phase_offsets``
    deterministic half-tile phase positions per axis, a tile contributes its
    ``tile_support_min``-th largest positive value when it holds at least
    ``tile_support_min`` positive pixels, and the reference is the maximum
    contributed level.

    This is a **robust spatial maximum**: an isolated pathological maximum
    (e.g. one hot/over-weighted pixel) can never define the reference, because
    no tile contains ``tile_support_min`` such pixels; a compact full-support
    plateau (e.g. ``2%`` of the footprint) *is* recovered because it populates
    at least one whole tile.  A spatially *scattered* population of outliers
    (``> 0.5%`` globally but fewer than ``tile_support_min`` in every tile) does
    **not** define the reference — the reference stays at the surrounding
    supported background level.  The phase offsets remove base-tile boundary
    sensitivity (a compact cluster straddling a base boundary is still found).
    The algorithm is **scale-invariant** under exposure scaling
    (``wht -> k * wht`` leaves the mask unchanged and scales the reference by
    ``k``), and it operates only on *positive* values within tiles, so
    sparse-positive kernels/pixfrac need no positive geometric neighbours.

    Degenerate edge behaviour: when no tile anywhere reaches
    ``tile_support_min`` positive pixels (a footprint smaller than the minimum
    supported population, or a purely isolated-pixel layout), the reference
    collapses to the *minimum* positive value (a deterministic
    keep-everything choice for a tiny footprint) with reason
    ``"no_supported_tile"``.

    Parameters
    ----------
    wht : ndarray
        Final coverage map, ``(H, W)`` float.  A 3-D ``(H, W, C)`` array is
        reduced to ``(H, W)`` with the per-pixel channel **mean** (the same
        reduction used by the finalizer for display/post-processing).
    fraction : float
        Relative threshold in ``[0, 1]``.  ``0`` keeps every positive-weight
        pixel; ``1`` keeps only pixels at/above the reference support.
    tile_size : int
        Fixed-size square tile edge in pixels (default ``8``).
    tile_support_min : int
        Minimum positive pixels per tile (default ``4``).
    n_phase_offsets : int
        Phase positions per axis (default ``2`` -> ``{0, tile_size // 2}``,
        i.e. ``4`` tile grids).

    Returns
    -------
    WhtThresholdResult
    """
    arr = np.asarray(wht, dtype=np.float32)
    if arr.ndim == 3:
        # Channel reduction semantics: per-pixel mean over the channel axis.
        arr = np.mean(arr, axis=-1, dtype=np.float32)
    elif arr.ndim != 2:
        raise ValueError("wht must be 2-D or 3-D")

    fraction = float(fraction)
    if not np.isfinite(fraction):
        fraction = 0.0
    fraction = min(1.0, max(0.0, fraction))

    tile_size = max(2, int(tile_size))
    tile_support_min = max(1, int(tile_support_min))
    n_phase_offsets = max(1, int(n_phase_offsets))

    finite = np.isfinite(arr)
    positive = finite & (arr > 0.0)
    n_positive = int(np.count_nonzero(positive))

    result = WhtThresholdResult(
        fraction=fraction,
        tile_size=tile_size,
        tile_support_min=tile_support_min,
        n_phase_offsets=n_phase_offsets,
        reference_support=0.0,
        cutoff=0.0,
        mask=np.zeros(arr.shape, dtype=bool),
        masked_fraction=0.0,
        n_positive=n_positive,
        n_valid=0,
        reason="applied",
    )

    if n_positive == 0:
        result.reason = "no_positive_weight"
        return result

    reference_support = _block_supported_reference(
        arr, positive, tile_size, tile_support_min, n_phase_offsets
    )

    if (
        reference_support is None
        or not np.isfinite(reference_support)
        or reference_support <= 0.0
    ):
        # Degenerate: no tile reaches the minimum supported population.
        # Reference collapses to the minimum positive value (keep-everything),
        # a deterministic choice for a tiny / isolated-pixel footprint.
        reference_support = float(np.min(arr[positive]))
        result.reason = "no_supported_tile"

    cutoff = fraction * reference_support
    mask = positive & (arr >= cutoff)
    n_valid = int(np.count_nonzero(mask))

    result.reference_support = float(reference_support)
    result.cutoff = cutoff
    result.mask = mask
    result.n_valid = n_valid
    result.masked_fraction = 1.0 - (n_valid / n_positive)
    return result
