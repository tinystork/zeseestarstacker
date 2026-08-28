"""Immutable run-level additive background matching for M3 Drizzle.

Purpose
-------
M3 Drizzle deposits each accepted *original* (non-resampled) frame exactly
once into per-channel :class:`~seestar.core.drizzle_core.DrizzleAccumulator`
objects.  When the accepted frames carry *different* additive sky backgrounds
(e.g. changing moon glow, airglow, or detector offset), the weighted mean that
Drizzle produces develops a spatial "step" wherever the set of contributing
frames changes (partial overlap, field rotation, dithering).  This module
removes that step by matching, per channel, each frame's constant background
offset to a single **immutable run-level anchor** *before* the one and only
Drizzle deposition.

The anchor is the **immutable registration reference** returned by the aligner
(``_get_reference_image``), not the first accepted frame.  It is rescaled into
the same ADU photometric domain as the deposited frames (see
:func:`rescale_01_to_adu`) and carries identity geometry (the reference *is*
the reference grid).

This is *not* generic background subtraction and it is *not* a reprojection.
It is a constant per-channel additive correction ``frame - offset`` where
``offset`` is estimated from geometrically corresponding sky/overlap samples
only.  Estimation may resample for *measurement* (bilinear sampling of the
anchor at mapped positions) but the deposited science is never resampled.

Geometry / measurement mapping
------------------------------
``estimate_background_offsets`` accepts *either* an explicit 2x3 affine
``tf`` (ORIGINAL pixel -> reference grid) *or* a ``native_wcs`` +
``reference_wcs`` pair (for the astrometry-single path where ``tf is None``).
In the WCS path, each sampled frame pixel is mapped through
``native_wcs.all_pix2world`` then ``reference_wcs.all_world2pix`` so the
comparison always happens at the *same sky position* — never at a false
identity position.  If no geometry is available, or the geometry is singular /
pathological, the estimator returns a deterministic neutral correction with a
structured ``degenerate_geometry`` / ``invalid_wcs`` reason.

Immutability
------------
The anchor is captured once and never mutated.  The per-frame correction is a
pure function of:

* the immutable anchor state,
* the current frame's data + validity mask,
* the frame's geometry (``tf`` or ``native_wcs``) relative to the reference
  grid.

Later frames cannot change an earlier frame's correction: once a corrected
frame is deposited into SCI/WHT its contribution is immutable.

Memory bound
------------
Estimation uses deterministic bounded sampling (a regular grid with a
documented maximum sample budget, default ``<= 250000`` pixels), never
full-frame float64 temporaries.  The anchor pixels are stored as ``float32``.
Diagnostics report the candidate/sample/used counts and the sampling
stride/budget.

Persistence boundary
--------------------
:class:`BackgroundAnchor` exposes ``to_metadata`` / ``from_metadata`` helpers
that reconstruct the *scalar anchor contract* (identity, provenance, geometry,
per-channel background) — sufficient to verify that a re-derived anchor
matches the run-level anchor.  This is deliberately **not** a full Drizzle
SCI/WHT disk-checkpoint contract: the full per-pixel anchor and accumulators
are not serialized to disk in the 8.1.0 baseline (see the Drizzle architecture
document).

Only ``numpy`` is used (no ``scipy``, no ``cv2``, no ``cupy``).  WCS objects
are duck-typed (``all_pix2world`` / ``all_world2pix``), so this module stays
independent of ``astropy.wcs``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "BackgroundAnchor",
    "estimate_background_offsets",
    "apply_background_offsets",
    "invert_affine_2x3",
    "sample_bilinear",
    "robust_location",
    "rescale_01_to_adu",
    "apply_wb_basique",
    "native_wcs_to_reference_coords",
    "ANCHOR_VERSION",
    "DEFAULT_MIN_OVERLAP_SAMPLES",
    "DEFAULT_SIGMA_CLIP",
    "DEFAULT_CLIP_ITERATIONS",
    "DEFAULT_MAX_SAMPLES",
    "WB_GAIN_CLIP",
    "WB_GATE",
]

# Version of the anchor *contract* (schema of ``to_metadata``).  Bump only
# when the metadata schema or the estimator semantics change in a way that
# would make a reconstructed anchor incomparable.
ANCHOR_VERSION = 1

# Exact parameters of the shared Classic/M3 "WB basique" per-frame RGB gain
# (R/B channel medians pulled toward G).  These are the verbatim constants of
# the existing ``_process_file`` inline block and must never diverge from it.
WB_GATE = 1e-6          # med_g below this -> no gain applied (identity)
WB_GAIN_CLIP = (0.5, 2.0)  # per-channel gain clamp bounds

# Conservative floors/limits for the estimator.  ``MIN_OVERLAP_SAMPLES`` is the
# minimum number of valid, geometrically corresponding samples required before
# a non-neutral correction is accepted; below it the correction is
# deterministically neutral (zero) with an explicit structured reason.
DEFAULT_MIN_OVERLAP_SAMPLES = 200
DEFAULT_SIGMA_CLIP = 3.0
DEFAULT_CLIP_ITERATIONS = 3

# Maximum measurement sample budget (pixels).  Estimation never materialises a
# full-frame float64 grid; it samples at most this many pixel positions on a
# deterministic regular grid.
DEFAULT_MAX_SAMPLES = 250_000

# Reason vocabulary (stable strings, consumed by diagnostics / logs / tests).
REASON_ACCEPTED = "accepted"
REASON_INSUFFICIENT_OVERLAP = "insufficient_overlap"
REASON_NO_VALID_SAMPLES = "no_valid_samples"
REASON_DEGENERATE_GEOMETRY = "degenerate_geometry"
REASON_INVALID_WCS = "invalid_wcs"
REASON_NO_ANCHOR_DATA = "no_anchor_data"


def rescale_01_to_adu(data, adu_scale=65535.0, tol=1e-5):
    """Rescale a ``[0, 1]``-range image into the ADU domain used for Drizzle.

    This is the exact photometric preparation applied by the Drizzle/Mosaic
    source path (``load -> debayer -> hot pixels -> [0,1] -> ADU``): if the
    image maximum lies in ``[0, 1]`` (within ``tol``), it is multiplied by
    ``adu_scale`` (``65535``); the result is always clipped to ``>= 0`` and
    returned as ``float32``.  It is a *pure* helper shared by the normal source
    preparation and the reference-anchor preparation so the two cannot drift.
    """
    arr = np.asarray(data)
    current_max_val = np.nanmax(arr)
    if current_max_val <= 1.0 + tol and current_max_val > -tol:
        arr = arr * adu_scale
    return np.clip(arr, 0.0, None).astype(np.float32)


def apply_wb_basique(img):
    """Apply the exact Classic/M3 per-frame "WB basique" RGB gains (R/B toward G).

    This is a verbatim extraction of the ``_process_file`` inline block so the
    Drizzle background anchor and the deposited frames share the **same RGB
    photometric domain**.  ``_process_file`` applies it after debayering and
    before hot-pixel correction / ADU rescale; ``_get_reference_image`` does
    **not** apply it, so the anchor capture must apply it before the ADU
    rescale (otherwise the anchor and frames differ multiplicatively in R/B).

    The math, exactly as the existing inline block:

    * ``med_r/med_g/med_b = np.median`` of each float32 channel;
    * if ``med_g > WB_GATE`` (``1e-6``):
      ``gain_r = clip(med_g / max(med_r, 1e-6), 0.5, 2.0)``,
      ``gain_b = clip(med_g / max(med_b, 1e-6), 0.5, 2.0)`` and the R/B
      channels are multiplied in-place (G is untouched);
    * otherwise no gain is applied (identity, ``gain_r = gain_b = 1.0``).

    The helper operates in float32 and returns a **new** float32 array (the
    input is never mutated); the float32 medians / clamped float32 gains / the
    in-place float32 multiply reproduce the inline block bit-for-bit, so the
    Classic output is numerically unchanged.

    Returns ``(out_float32, info)`` where ``info`` is a bounded diagnostics
    dict of plain Python floats: ``applied``, ``med_r``, ``med_g``, ``med_b``,
    ``gain_r``, ``gain_b``.
    """
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("apply_wb_basique expects an (H, W, 3) RGB array")

    r_ch = img[..., 0]
    g_ch = img[..., 1]
    b_ch = img[..., 2]
    med_r = np.median(r_ch)
    med_g = np.median(g_ch)
    med_b = np.median(b_ch)

    out = np.array(img, dtype=np.float32, copy=True)
    gain_r = 1.0
    gain_b = 1.0
    applied = False
    if med_g > WB_GATE:
        gain_r = np.clip(med_g / max(med_r, WB_GATE), WB_GAIN_CLIP[0], WB_GAIN_CLIP[1])
        gain_b = np.clip(med_g / max(med_b, WB_GATE), WB_GAIN_CLIP[0], WB_GAIN_CLIP[1])
        out[..., 0] *= gain_r
        out[..., 2] *= gain_b
        applied = True

    info = {
        "applied": bool(applied),
        "med_r": float(med_r),
        "med_g": float(med_g),
        "med_b": float(med_b),
        "gain_r": float(gain_r),
        "gain_b": float(gain_b),
    }
    return out, info


def native_wcs_to_reference_coords(native_wcs, reference_wcs, x, y):
    """Map native-WCS pixel coords to reference-grid coords (measurement-only).

    ``x``/``y`` are arrays (broadcast to a common shape) of 0-indexed pixel
    coordinates on the *native* frame grid.  They are mapped to the *reference*
    grid via ``native_wcs.all_pix2world`` then ``reference_wcs.all_world2pix``.
    This is the measurement analogue of the deposition mapping (which composes
    an identity ``tf`` with ``native_wcs``); it never alters the deposited
    science.

    Returns ``(ref_x, ref_y)`` arrays.  A mapping that fails (singular /
    pathological WCS) raises, and callers must convert that into a neutral
    ``invalid_wcs`` fallback.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ra, dec = native_wcs.all_pix2world(x, y, 0)
    ref_x, ref_y = reference_wcs.all_world2pix(ra, dec, 0)
    return np.asarray(ref_x, dtype=np.float64), np.asarray(ref_y, dtype=np.float64)


def _measurement_sample_indices(h, w, max_samples):
    """Deterministic bounded sample of pixel indices for background measurement.

    Returns ``(yy, xx, stride)`` as flat ``int64`` arrays of 0-indexed pixel
    centres.  For ``h * w <= max_samples`` every pixel is sampled (exact
    small-image behaviour); otherwise a regular grid with stride ``stride``
    keeps the sample count ``<= max_samples``.
    """
    n = int(h) * int(w)
    if n <= int(max_samples):
        yy, xx = np.indices((h, w), dtype=np.int64)
        return yy.ravel(), xx.ravel(), 1
    stride = 1
    while True:
        ny = (h + stride - 1) // stride
        nx = (w + stride - 1) // stride
        if ny * nx <= int(max_samples):
            break
        stride += 1
    ys = np.arange(0, h, stride, dtype=np.int64)
    xs = np.arange(0, w, stride, dtype=np.int64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return yy.ravel(), xx.ravel(), stride


def invert_affine_2x3(tf):
    """Invert a 2x3 affine ``tf`` (maps ``p = A @ q + t``).

    Returns the 2x3 affine mapping ``p -> q`` (linear ``A^-1``, translation
    ``-A^-1 @ t``).  Raises :class:`numpy.linalg.LinAlgError` if the linear
    part is singular (callers must handle it as a degenerate geometry).
    """
    tf = np.asarray(tf, dtype=np.float64)
    if tf.shape != (2, 3):
        raise ValueError("tf must be a 2x3 affine array")
    a = tf[:2, :2]
    t = tf[:2, 2]
    a_inv = np.linalg.inv(a)
    return np.hstack([a_inv, (-a_inv @ t).reshape(2, 1)])


def sample_bilinear(data, x, y):
    """Bilinearly sample a ``(H, W, C)`` array at fractional ``(x, y)`` coords.

    ``x``/``y`` are arrays broadcast to a common shape ``S`` (the output has
    shape ``S + (C,)``).  Sample locations whose interpolation footprint falls
    outside the array (``x < 0``, ``y < 0``, ``x > W-2``, ``y > H-2``) yield
    ``NaN`` in every channel, so they are excluded downstream as invalid.

    The input is *not* upcast to a full float64 copy: only the gathered
    neighbourhood is promoted, keeping the memory transient proportional to the
    number of sample positions ``S``.
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError("sample_bilinear expects a (H, W, C) array")
    h, w, c = data.shape
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0

    valid = (x0 >= 0) & (x0 + 1 < w) & (y0 >= 0) & (y0 + 1 < h)
    # Clamp to safe indices for the gather; invalid locations are zeroed after.
    x0s = np.clip(x0, 0, w - 2)
    y0s = np.clip(y0, 0, h - 2)
    x1s = x0s + 1
    y1s = y0s + 1

    v00 = data[y0s, x0s, :].astype(np.float64)
    v10 = data[y0s, x1s, :].astype(np.float64)
    v01 = data[y1s, x0s, :].astype(np.float64)
    v11 = data[y1s, x1s, :].astype(np.float64)

    w00 = ((1.0 - fx) * (1.0 - fy))[..., None]
    w10 = (fx * (1.0 - fy))[..., None]
    w01 = ((1.0 - fx) * fy)[..., None]
    w11 = (fx * fy)[..., None]
    out = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11
    out = np.where(valid[..., None], out, np.nan)
    return out


def robust_location(samples, sigma=DEFAULT_SIGMA_CLIP, iterations=DEFAULT_CLIP_ITERATIONS):
    """Robust location of a 1-D sample set via iterative sigma-clipped median.

    The estimator is deliberately conservative for sky background matching:

    * it starts from the median (insensitive to bright sources and to moderate
      diffuse structure, both of which are minority outliers around the sky
      level);
    * it sigma-clips around that median using a robust MAD-based scale, which
      removes stars, hot pixels and gradient tails from the estimate;
    * it returns the *clipped median* (a finite, outlier-robust central value).

    ``NaN``/``Inf`` values are ignored.  Returns ``(location, used_count)``
    where ``used_count`` is the number of samples retained after clipping.
    """
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return 0.0, 0

    keep = np.ones(samples.size, dtype=bool)
    used = samples.size
    for _ in range(max(1, int(iterations))):
        cur = samples[keep]
        if cur.size == 0:
            break
        med = np.median(cur)
        mad = np.median(np.abs(cur - med))
        scale = mad * 1.4826
        if not np.isfinite(scale) or scale <= 1e-12:
            # Degenerate (constant) sample set: the median is already exact.
            break
        lo = med - sigma * scale
        hi = med + sigma * scale
        new_keep = keep.copy()
        new_keep[keep] = (cur >= lo) & (cur <= hi)
        if int(new_keep.sum()) == int(keep.sum()):
            break
        keep = new_keep

    cur = samples[keep]
    if cur.size == 0:
        return float(np.median(samples)), 0
    return float(np.median(cur)), int(cur.size)


class BackgroundAnchor:
    """Immutable run-level per-channel background anchor for M3 Drizzle.

    Holds the anchor frame's data (float32, ``(H, W, C)``) together with the
    geometry that places it on the reference grid (``tf`` maps anchor pixels
    to reference-grid pixels; the registration reference itself has identity
    ``tf``).  ``reference_shape_hw`` is the reference-grid ``(H, W)``.

    The object is created once per run and never mutated afterwards.  The data
    array is stored as a private read-only-copy attribute and only exposed via
    the ``sample`` measurement path.
    """

    def __init__(self, anchor_data, tf=None, reference_shape_hw=None,
                 provenance=None, version=ANCHOR_VERSION):
        # Own a *private* float32 copy.  ``np.asarray`` would alias a float32
        # caller array, so ``setflags(write=False)`` below would then make the
        # caller's array read-only.  ``copy=True`` guarantees the anchor never
        # shares storage with — or mutates the flags of — the caller's array.
        data = np.array(anchor_data, dtype=np.float32, copy=True)
        if data.ndim == 2:
            data = np.repeat(data[..., None], 3, axis=2)
        if data.ndim != 3 or data.shape[2] < 1:
            raise ValueError("anchor data must be (H, W) or (H, W, C)")

        self._data = data
        self._data.setflags(write=False)
        self._shape_hwc = tuple(int(v) for v in data.shape)
        self._n_channels = int(data.shape[2])

        if tf is None:
            tf = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        tf = np.asarray(tf, dtype=np.float64)
        if tf.shape != (2, 3):
            raise ValueError("anchor tf must be a 2x3 affine array")
        self._tf = tf.copy()

        if reference_shape_hw is None:
            reference_shape_hw = tuple(data.shape[:2])
        self._reference_shape_hw = (int(reference_shape_hw[0]), int(reference_shape_hw[1]))
        self._provenance = str(provenance) if provenance is not None else "unknown"
        self._version = int(version)

        # Precompute the anchor's robust per-channel background (used for
        # metadata/diagnostics and for the neutral-fallback scalar contract).
        self._background = self._compute_background()

    # ------------------------------------------------------------------ #
    # read-only accessors
    # ------------------------------------------------------------------ #
    @property
    def shape(self):
        """Anchor data shape ``(H, W, C)`` (safe even for metadata-only anchors)."""
        return self._shape_hwc

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def reference_shape_hw(self):
        return self._reference_shape_hw

    @property
    def provenance(self):
        return self._provenance

    @property
    def version(self):
        return self._version

    @property
    def tf(self):
        """Copy of the anchor's 2x3 geometry (anchor pixels -> reference grid)."""
        return self._tf.copy()

    @property
    def background(self):
        """Robust per-channel background scalars of the anchor (copy)."""
        return self._background.copy()

    def _compute_background(self):
        bg = []
        for ch in range(self._n_channels):
            vals = self._data[..., ch].ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                bg.append(0.0)
            else:
                bg.append(float(np.median(vals)))
        return np.asarray(bg, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # measurement-only sampling
    # ------------------------------------------------------------------ #
    def sample(self, ref_x, ref_y):
        """Sample the anchor at reference-grid coordinates ``(ref_x, ref_y)``.

        ``ref_x``/``ref_y`` are arrays broadcast to a common shape ``S``; the
        result has shape ``S + (C,)`` (float64).  Coordinates are mapped back
        through the inverse of ``tf`` before sampling the anchor data, so this
        is a *measurement-only* interpolation that never alters the anchor.
        """
        self._require_data()
        inv = invert_affine_2x3(self._tf)
        x = np.asarray(ref_x, dtype=np.float64)
        y = np.asarray(ref_y, dtype=np.float64)
        ax = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]
        ay = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]
        return sample_bilinear(self._data, ax, ay)

    # ------------------------------------------------------------------ #
    # scalar-contract serialization (not a full checkpoint)
    # ------------------------------------------------------------------ #
    def to_metadata(self):
        """Serialize the *scalar* anchor contract (no pixel data).

        The returned dict is JSON-safe and sufficient to verify that a
        re-derived anchor matches this run's anchor.  It does **not** serialize
        the per-pixel anchor or the Drizzle accumulators (full disk checkpoint
        support is out of scope for the 8.1.0 baseline).
        """
        return {
            "version": self._version,
            "provenance": self._provenance,
            "reference_shape_hw": list(self._reference_shape_hw),
            "anchor_shape_hwc": [int(v) for v in self._shape_hwc],
            "tf": [[float(v) for v in row] for row in self._tf],
            "background_per_channel": [float(v) for v in self._background],
            "n_channels": self._n_channels,
        }

    @classmethod
    def from_metadata(cls, metadata):
        """Reconstruct the *scalar* anchor contract from ``to_metadata`` output.

        The reconstructed anchor carries no pixel data (``sample`` raises
        :class:`RuntimeError`).  It exists to verify identity/geometry and to
        reconstruct the neutral-fallback scalar contract, never to re-run
        per-pixel matching.  The original anchor shape is preserved separately
        (``shape`` remains inspectable even though ``_data`` is ``None``).
        """
        if not isinstance(metadata, dict):
            raise ValueError("anchor metadata must be a dict")
        version = int(metadata.get("version", ANCHOR_VERSION))
        provenance = str(metadata.get("provenance", "unknown"))
        ref_shape = metadata.get("reference_shape_hw")
        anchor_shape = metadata.get("anchor_shape_hwc")
        tf = metadata.get("tf")
        bg = metadata.get("background_per_channel")

        if not isinstance(ref_shape, (list, tuple)) or len(ref_shape) != 2:
            raise ValueError("anchor metadata has invalid reference_shape_hw")
        if not isinstance(tf, (list, tuple)) or len(tf) != 2:
            raise ValueError("anchor metadata has invalid tf")

        n_channels = 3
        if isinstance(bg, (list, tuple)) and len(bg) > 0:
            n_channels = len(bg)
        # A tiny placeholder data array is used: the reconstructed anchor is
        # pixel-less (see below), so the full anchor shape is never allocated.
        placeholder = np.zeros((1, 1, max(1, n_channels)), dtype=np.float32)

        anchor = cls(
            placeholder,
            tf=np.asarray(tf, dtype=np.float64),
            reference_shape_hw=tuple(ref_shape),
            provenance=provenance,
            version=version,
        )
        # Replace the placeholder-derived background with the documented values
        # when present (reconstruct the scalar contract verbatim).
        if bg is not None and len(bg) == anchor.n_channels:
            anchor._background = np.asarray([float(v) for v in bg], dtype=np.float64)
        # Mark the reconstructed anchor as pixel-less, but keep the documented
        # anchor shape (the placeholder shape is never exposed as the truth).
        anchor._data = None
        if isinstance(anchor_shape, (list, tuple)) and len(anchor_shape) == 3:
            anchor._shape_hwc = tuple(int(v) for v in anchor_shape)
        return anchor

    def _require_data(self):
        if self._data is None:
            raise RuntimeError(
                "this BackgroundAnchor was reconstructed from metadata and "
                "carries no pixel data (sample() is unavailable)"
            )


def estimate_background_offsets(
    frame,
    weight,
    tf,
    anchor,
    *,
    native_wcs=None,
    reference_wcs=None,
    min_overlap_samples=DEFAULT_MIN_OVERLAP_SAMPLES,
    sigma_clip=DEFAULT_SIGMA_CLIP,
    clip_iterations=DEFAULT_CLIP_ITERATIONS,
    max_samples=DEFAULT_MAX_SAMPLES,
):
    """Estimate the per-channel additive background offset of ``frame`` vs ``anchor``.

    Pure function of the immutable anchor + current frame + geometry:

    1. deterministically sample frame pixel centres (bounded budget),
    2. map each sampled centre to the reference grid via ``tf`` (affine) or via
       ``native_wcs`` -> ``reference_wcs`` (celestial, when ``tf is None``),
    3. sample the anchor at those positions (bilinear, measurement-only),
    4. compute ``delta = frame - anchor_sample`` per channel,
    5. restrict to valid samples (weight > 0, in reference grid, finite),
    6. robustly estimate each channel's constant offset (sigma-clipped median),
    7. fall back to a neutral (zero) correction with a structured reason when
       overlap/confidence is insufficient or geometry is degenerate.

    Parameters
    ----------
    frame : ndarray
        Original frame data, ``(H, W, C)`` float32/float64.
    weight : ndarray
        Validity map ``(H, W)`` (``> 0`` == valid).  May broadcast from ``(H, W)``.
    tf : array_like or None
        2x3 affine mapping ORIGINAL pixels -> reference-grid pixels.  When
        ``None``, ``native_wcs`` + ``reference_wcs`` must be supplied.
    anchor : BackgroundAnchor
        Immutable run-level anchor.
    native_wcs : WCS-like, optional
        The frame's own resolved WCS (used only when ``tf is None``).
    reference_wcs : WCS-like, optional
        The immutable reference-grid WCS (used only when ``tf is None``).
    min_overlap_samples : int
        Minimum valid geometric samples for a non-neutral correction.
    sigma_clip : float
        Sigma threshold for the robust clip.
    clip_iterations : int
        Number of clipping iterations.
    max_samples : int
        Maximum measurement sample budget (pixels).

    Returns
    -------
    offsets : ndarray (float64, ``(C,)``)
        Constant per-channel correction to *subtract* from the frame so its
        background matches the anchor.  Neutral (all zero) on fallback.
    diagnostics : dict
        Structured, bounded diagnostics (see module docstring).
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.ndim != 3:
        raise ValueError("frame must be (H, W) or (H, W, C)")
    h, w, n_ch = frame.shape

    weight = np.asarray(weight, dtype=np.float32)
    if weight.shape != (h, w):
        try:
            weight = np.broadcast_to(weight, (h, w)).astype(np.float32, copy=False)
        except ValueError:
            weight = np.zeros((h, w), dtype=np.float32)

    diag = {
        "version": ANCHOR_VERSION,
        "provenance": anchor.provenance,
        "reason": REASON_NO_VALID_SAMPLES,
        "offsets": [0.0] * n_ch,
        "n_candidate": 0,
        "stride": 1,
        "max_samples": int(max_samples),
        "n_overlap": 0,
        "n_used": 0,
        "confidence": 0.0,
        "anchor_background": [float(v) for v in anchor.background],
        "frame_background": [0.0] * n_ch,
        "robust_scale": [0.0] * n_ch,
    }

    if n_ch == 0 or frame.size == 0:
        return np.zeros(n_ch, dtype=np.float64), diag

    # 1. deterministic bounded sample of frame pixel centres
    yy, xx, stride = _measurement_sample_indices(h, w, max_samples)
    diag["n_candidate"] = int(yy.size)
    diag["stride"] = int(stride)
    if yy.size == 0:
        diag["reason"] = REASON_NO_VALID_SAMPLES
        return np.zeros(n_ch, dtype=np.float64), diag

    xx_f = xx.astype(np.float64)
    yy_f = yy.astype(np.float64)

    # 2. map sampled centres to the reference grid
    if tf is not None:
        tf = np.asarray(tf, dtype=np.float64)
        if tf.shape != (2, 3):
            raise ValueError("tf must be a 2x3 affine array")
        a = tf[:2, :2]
        det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
        if not np.isfinite(det) or abs(det) < 1e-12:
            diag["reason"] = REASON_DEGENERATE_GEOMETRY
            return np.zeros(n_ch, dtype=np.float64), diag
        ref_x = tf[0, 0] * xx_f + tf[0, 1] * yy_f + tf[0, 2]
        ref_y = tf[1, 0] * xx_f + tf[1, 1] * yy_f + tf[1, 2]
    elif native_wcs is not None and reference_wcs is not None:
        try:
            ref_x, ref_y = native_wcs_to_reference_coords(
                native_wcs, reference_wcs, xx_f, yy_f
            )
        except Exception:
            diag["reason"] = REASON_INVALID_WCS
            return np.zeros(n_ch, dtype=np.float64), diag
    else:
        # No geometry at all: nothing truthful to compare against.
        diag["reason"] = REASON_DEGENERATE_GEOMETRY
        return np.zeros(n_ch, dtype=np.float64), diag

    # 2b. sample the frame at the sampled positions
    frame_sample = frame[yy, xx, :].astype(np.float64)  # (S, C)
    weight_sample = weight[yy, xx]

    # 3. sample the anchor at mapped positions (measurement-only)
    try:
        anchor_sample = anchor.sample(ref_x, ref_y)  # (S, C)
    except RuntimeError:
        diag["reason"] = REASON_NO_ANCHOR_DATA
        return np.zeros(n_ch, dtype=np.float64), diag
    except np.linalg.LinAlgError:
        diag["reason"] = REASON_DEGENERATE_GEOMETRY
        return np.zeros(n_ch, dtype=np.float64), diag

    # 4/5. validity: in reference grid, weight > 0, finite everywhere
    ref_h, ref_w = anchor.reference_shape_hw
    eps = 1e-9
    in_grid = (
        (ref_x >= -eps) & (ref_x < ref_w) & (ref_y >= -eps) & (ref_y < ref_h)
    )
    finite_frame = np.all(np.isfinite(frame_sample), axis=-1)
    finite_anchor = np.all(np.isfinite(anchor_sample), axis=-1)
    valid = in_grid & (weight_sample > 0) & finite_frame & finite_anchor

    n_overlap = int(np.count_nonzero(valid))
    diag["n_overlap"] = n_overlap
    if n_overlap < max(1, int(min_overlap_samples)):
        diag["reason"] = REASON_INSUFFICIENT_OVERLAP
        # still record frame backgrounds (median of valid frame samples) for
        # bounded observability even on the neutral path
        for ch in range(n_ch):
            chv = frame_sample[valid, ch]
            if chv.size:
                diag["frame_background"][ch] = float(np.median(chv))
        return np.zeros(n_ch, dtype=np.float64), diag

    offsets = np.zeros(n_ch, dtype=np.float64)
    n_used_total = 0
    for ch in range(n_ch):
        delta = (frame_sample[valid, ch] - anchor_sample[valid, ch])
        delta = delta[np.isfinite(delta)]
        diag["frame_background"][ch] = float(np.median(frame_sample[valid, ch]))
        if delta.size == 0:
            offsets[ch] = 0.0
            continue
        loc, used = robust_location(delta, sigma=sigma_clip, iterations=clip_iterations)
        offsets[ch] = loc
        n_used_total = max(n_used_total, used)
        # robust scale of the delta distribution (MAD) for observability
        med = np.median(delta)
        mad = np.median(np.abs(delta - med))
        diag["robust_scale"][ch] = float(mad * 1.4826)

    diag["offsets"] = [float(v) for v in offsets]
    diag["n_used"] = int(n_used_total)
    diag["confidence"] = float(min(1.0, n_overlap / max(1, yy.size)))
    diag["reason"] = REASON_ACCEPTED

    return offsets, diag


def apply_background_offsets(frame, offsets):
    """Subtract per-channel constant offsets from a frame (pure, non-mutating).

    ``frame`` may be ``(H, W)`` or ``(H, W, C)``; ``offsets`` is broadcast over
    the channel axis.  Returns a new float32 array.

    The correction is performed in a **private float32 copy**: the caller's
    input array is never mutated and no full-frame float64 temporary (result or
    intermediate) is ever materialised.  The offsets are cast to float32 and
    subtracted in-place on that copy, so the transient memory is one float32
    frame copy (the returned corrected array), not a float64 upcast plus a
    second corrected copy.
    """
    frame = np.asarray(frame)
    corrected = np.array(frame, dtype=np.float32, copy=True)
    offsets = np.asarray(offsets, dtype=np.float32)
    if corrected.ndim == 2:
        if offsets.size:
            corrected -= offsets[0]
    else:
        if offsets.size:
            corrected -= offsets.reshape((1, 1, -1))
    return corrected
