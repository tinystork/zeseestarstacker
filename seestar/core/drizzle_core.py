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
    ``drizzle`` 2.2.0 stores the *weighted mean* in ``out_img`` and the total
    weight in ``out_wht`` (the weighted flux is ``out_img * out_wht``).  To
    expose the standard drizzle semantics (weighted flux in ``sci``, exposure
    scaled weight in ``wht``), :class:`DrizzleAccumulator` reports
    ``sci = out_img * out_wht`` and passes ``wht_scale = expscale`` so that
    ``sci / wht`` yields the exposure-weighted mean of the count rate.
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
]


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
        """Accumulated science image (weighted flux ``sum(w * f)``).

        ``drizzle`` 2.2.0 keeps the weighted *mean* in ``out_img`` and the
        total weight in ``out_wht``; their product is the weighted flux.
        """
        return (self._out_img * self._out_wht).astype(np.float32)

    @property
    def wht(self):
        """Accumulated weight image (exposure-scaled weight), as a copy."""
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

        * ``"divide"``   -> ``sci / max(wht, 1e-9)``
        * ``"none"``     -> ``sci``
        * ``"max"``      -> ``sci / wht_safe * max(wht_safe)``
        * ``"n_images"`` -> ``sci / wht_safe * mean(wht_safe)``

        Invalid values are replaced by ``0`` and the result is ``float32``.
        """
        sci = self.sci
        wht = self._out_wht.astype(np.float32)

        if mode not in {"divide", "none", "max", "n_images"}:
            mode = "divide"

        if mode == "none":
            result = sci
        else:
            wht_safe = np.maximum(wht, 1e-9)
            result = sci / wht_safe
            if mode == "max":
                result *= float(np.max(wht_safe))
            elif mode == "n_images":
                result *= float(np.mean(wht_safe))

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
