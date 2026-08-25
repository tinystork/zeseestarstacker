"""M3-C integration tests: Drizzle core wired into ``queue_manager``.

Small, fast, synthetic tests (32x32 / 64x64 images, no real FITS runs).  They
demonstrate the 6 science/engineering requirements of mission M3-C:

1. no pre-resampling: the ORIGINAL (non-warped) pose enters the core,
2. ``tf`` convention/direction documented + tested (translation, rotation,
   integrated sub-pixel centroid),
3. weight provenance (luminance validity mask on the ORIGINAL image, NaN/≤thr
   region excluded),
4. bounded memory (RSS < 150 MB over 30 frames, buffers released),
5. batch invariance at QM level (group_size 1 vs 8),
6. science validated BEFORE uint16 conversion in ``_save_final_stack``.

The tf convention is: **``tf`` maps ORIGINAL pixels to the REFERENCE grid**
(same direction as ``cv2.warpAffine`` in the aligner).  No ``warpAffine`` is
applied to the data: only the pixel *centres* are mapped via ``tf`` into the
reference WCS then into the output grid (``pixmap_from_alignment``).
"""

import gc
import math
import resource

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from seestar.core.drizzle_core import (
    DrizzleAccumulator,
    build_output_grid,
    drizzle_stream,
    pixmap_from_alignment,
)
from seestar.queuep.queue_manager import SeestarQueuedStacker


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_wcs(shape_hw, crval=(10.0, 20.0), cdelt=(-0.001, 0.001)):
    """A simple TAN WCS whose reference pixel is the grid centre."""
    h, w = shape_hw
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    wcs.wcs.crval = list(crval)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = np.array(list(cdelt))
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
    wcs.array_shape = (h, w)
    return wcs


def gauss2d(shape_hw, amp, sig, pos):
    """A centred 2D Gaussian (``(x, y)`` position, ``(H, W)`` shape)."""
    h, w = shape_hw
    yy, xx = np.indices((h, w))
    return (
        amp * np.exp(-((xx - pos[0]) ** 2 + (yy - pos[1]) ** 2) / (2.0 * sig ** 2))
    ).astype(np.float32)


def identity_tf():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def translation_tf(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def rotation_tf(angle_deg, centre):
    """2x3 affine rotating by ``angle_deg`` around ``centre=(cx, cy)``."""
    a = math.radians(angle_deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.array(centre, dtype=np.float64)
    t = c - r @ c
    return np.array(
        [[r[0, 0], r[0, 1], t[0]], [r[1, 0], r[1, 1], t[1]]], dtype=np.float64
    )


def apply_tf(tf, xy):
    """Apply a 2x3 affine to an ``(N, 2)`` array of ``(x, y)`` points."""
    xy = np.asarray(xy, dtype=np.float64)
    ones = np.ones((xy.shape[0], 1))
    return (tf @ np.hstack([xy, ones]).T).T


def centroid(img):
    h, w = img.shape
    yy, xx = np.indices((h, w))
    total = img.sum()
    return (img * xx).sum() / total, (img * yy).sum() / total


def make_fake_qm(shape_hw, scale=1.0):
    """A minimal ``SeestarQueuedStacker`` (``object.__new__`` + targeted attrs)."""
    qm = object.__new__(SeestarQueuedStacker)
    qm.drizzle_active_session = True
    qm.is_mosaic_run = False
    qm.reference_wcs_object = make_wcs(shape_hw)
    qm.drizzle_output_wcs, out_shape = build_output_grid(
        qm.reference_wcs_object, shape_hw, scale
    )
    qm.drizzle_accumulators = [DrizzleAccumulator(out_shape) for _ in range(3)]
    qm.update_progress = lambda *a, **k: None
    return qm


# ---------------------------------------------------------------------------
# 1. no pre-resampling enters the core
# ---------------------------------------------------------------------------


def test_no_resampling_enters_core(monkeypatch):
    shape = (32, 32)
    qm = make_fake_qm(shape)

    captured = {"data": [], "pixmap": []}
    orig_add = DrizzleAccumulator.add

    def spy_add(self, data, weight_map, pixmap, exptime=1.0, in_units="counts",
                in_grid_mask=None):
        captured["data"].append(np.array(data, copy=True))
        captured["pixmap"].append(np.array(pixmap, copy=True))
        return orig_add(self, data, weight_map, pixmap, exptime=exptime,
                        in_units=in_units, in_grid_mask=in_grid_mask)

    monkeypatch.setattr(DrizzleAccumulator, "add", spy_add)

    rng = np.random.default_rng(0)
    original = (rng.random((shape[0], shape[1], 3)) * 1000.0).astype(np.float32)
    weight = np.ones(shape, dtype=np.float32)
    header = fits.Header()
    header["EXPTIME"] = 2.5
    tf = translation_tf(0.5, -0.25)

    ok = qm._add_frame_to_drizzle_accumulators(original, header, tf, weight)
    assert ok is True

    assert len(captured["data"]) == 3
    for ch in range(3):
        # The ORIGINAL (non-resampled) channel is fed to the core verbatim; the
        # tf is only applied to the pixel *centres* (pixmap), never to the data.
        assert np.array_equal(captured["data"][ch], original[..., ch])
    assert captured["pixmap"][0].shape == (shape[0], shape[1], 2)


# ---------------------------------------------------------------------------
# 2. tf convention / direction
# ---------------------------------------------------------------------------


def test_tf_translation_direction():
    shape = (32, 32)
    ref = make_wcs(shape)
    out_wcs, _ = build_output_grid(ref, shape, 1.0)

    tf = translation_tf(2.0, -3.0)
    pixmap, _mask = pixmap_from_alignment(shape, tf, ref, out_wcs)

    # centre of the source maps to centre + tf in the reference grid
    cx, cy = 16.0, 16.0
    px = float(pixmap[int(cy), int(cx), 0])
    py = float(pixmap[int(cy), int(cx), 1])
    assert abs(px - (cx + 2.0)) < 1e-6
    assert abs(py - (cy - 3.0)) < 1e-6


def test_tf_rotation_direction():
    shape = (40, 40)
    ref = make_wcs(shape)
    out_wcs, _ = build_output_grid(ref, shape, 1.0)

    centre = (20.0, 20.0)
    tf = rotation_tf(25.0, centre)
    pixmap, _mask = pixmap_from_alignment(shape, tf, ref, out_wcs)

    # point (x, y) maps to R . (x, y) + t (analytic formula)
    x, y = 14.0, 26.0
    expected = apply_tf(tf, np.array([[x, y]]))[0]
    px = float(pixmap[int(round(y)), int(round(x)), 0])
    py = float(pixmap[int(round(y)), int(round(x)), 1])
    assert abs(px - expected[0]) < 1e-5
    assert abs(py - expected[1]) < 1e-5


def test_integrated_subpixel_centroid_via_qm():
    shape = (32, 32)
    qm = make_fake_qm(shape)

    tf = translation_tf(0.3, -0.7)
    frame = gauss2d(shape, 100.0, 1.5, (16.0, 16.0))
    rgb = np.stack([frame, frame, frame], axis=-1)
    header = fits.Header()
    header["EXPTIME"] = 1.0

    ok = qm._add_frame_to_drizzle_accumulators(rgb, header, tf, np.ones(shape, np.float32))
    assert ok is True

    sci = qm.drizzle_accumulators[0].finalize()
    cx, cy = centroid(sci)
    assert abs(cx - 16.3) < 0.1
    assert abs(cy - 15.3) < 0.1


# ---------------------------------------------------------------------------
# 3. weight provenance (luminance validity mask on the ORIGINAL image)
# ---------------------------------------------------------------------------


def test_weight_provenance_and_original_return(tmp_path):
    shape = (32, 32)
    qm = object.__new__(SeestarQueuedStacker)
    qm.align_on_disk = False
    qm.bayer_pattern = "GRBG"
    qm.correct_hot_pixels = False
    qm.drizzle_active_session = True
    qm.is_mosaic_run = False
    qm.use_quality_weighting = False
    qm.batch_size = 0
    qm.reproject_between_batches = False
    qm.reproject_coadd_final = False
    qm.reference_wcs_object = make_wcs(shape)
    qm.update_progress = lambda *a, **k: None

    class FakeAligner:
        def _align_image(self, img, ref, file_name, force_same_shape_as_ref=True,
                         use_disk=False, return_M=False, transform_only=False,
                         return_diagnostics=False):
            # A warpAffine-like transformation, clearly different from the
            # original: shift every column by 3.  This must NOT be what the
            # drizzle path returns.
            out = np.roll(img, 3, axis=1).astype(np.float32)
            M = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            diag = {"model": "euclidean", "match_count": 0} if return_diagnostics else None
            if transform_only:
                # RF2 transform-only contract: no warp, return original + tf.
                if return_diagnostics:
                    return img, True, M, diag
                if return_M:
                    return img, True, M
                return img, True
            if return_M:
                return out, True, M
            return out, True

    qm.aligner = FakeAligner()

    # bright background + dark (luminance <= threshold) region + star.
    img = np.full((shape[0], shape[1], 3), 100.0, dtype=np.float32)
    img[10:15, 10:15, :] = 0.0
    g = gauss2d(shape, 400.0, 2.0, (22.0, 22.0))
    for c in range(3):
        img[..., c] += g

    hdr = fits.Header()
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = shape[1]
    hdr["NAXIS2"] = shape[0]
    hdr["NAXIS3"] = 3
    path = tmp_path / "in.fits"
    fits.PrimaryHDU(data=img, header=hdr).writeto(path, overwrite=True)

    ref_data = np.zeros((shape[0], shape[1], 3), dtype=np.float32)

    data, header, scores, wcs, matrix_m, mask = qm._process_file(
        str(path), ref_data, solve_astrometry_for_this_file=False
    )

    assert mask is not None
    # dark region -> mask 0 (validity mask, not a noise map)
    assert mask[10:15, 10:15].sum() == 0
    # bright region -> mask 1
    assert mask[22, 22] == 1

    # the returned data is the ORIGINAL image (dark region still dark) — not
    # the warpAffine-aligned image (which would shift the dark block).
    assert data is not None
    assert data[10, 10].sum() < 1e-3
    assert data[22, 22].sum() > 0


# ---------------------------------------------------------------------------
# 4. bounded memory over 30 frames
# ---------------------------------------------------------------------------


def _rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def test_memory_bounded_over_30_frames():
    shape = (32, 32)
    qm = make_fake_qm(shape)
    header = fits.Header()
    header["EXPTIME"] = 1.0

    def make_frame(i):
        tf = translation_tf((i % 5) * 0.1, ((i * 7) % 5) * 0.1)
        frame = gauss2d(shape, 50.0 + i, 1.5, (16.0, 16.0))
        return np.stack([frame, frame, frame], axis=-1), tf

    gc.collect()
    baseline = _rss_kb()
    paliers = []
    for i in range(30):
        rgb, tf = make_frame(i)
        qm._add_frame_to_drizzle_accumulators(rgb, header, tf, np.ones(shape, np.float32))
        if (i + 1) % 5 == 0:
            gc.collect()
            paliers.append(_rss_kb() - baseline)

    gc.collect()
    total_growth = _rss_kb() - baseline
    total_growth_mb = total_growth / 1024.0  # ru_maxrss is KiB on Linux
    assert total_growth_mb < 150.0, f"total RSS growth {total_growth_mb:.1f} MB"

    per_palier_mb = [
        (paliers[i] - (paliers[i - 1] if i else 0)) / 1024.0
        for i in range(len(paliers))
    ]
    assert max(per_palier_mb, default=0.0) < 50.0, (
        f"per-palier RSS growth {max(per_palier_mb, default=0.0):.1f} MB"
    )


# ---------------------------------------------------------------------------
# 5. batch invariance at QM level (group_size 1 vs 8)
# ---------------------------------------------------------------------------


def test_batch_invariance_at_qm_level():
    shape = (32, 32)
    shifts = [(0.0, 0.0), (0.3, -0.7), (-0.5, 0.2), (0.8, 0.4),
              (-0.2, -0.4), (0.6, -0.1), (-0.7, 0.9), (0.1, 0.5)]
    amps = [100.0, 80.0, 120.0, 60.0, 90.0, 110.0, 70.0, 130.0]

    def build_frames():
        frames = []
        for i, (sx, sy) in enumerate(shifts):
            tf = rotation_tf(8.0, (16.0, 16.0)) if i == 7 else translation_tf(sx, sy)
            frames.append((gauss2d(shape, amps[i], 1.5, (16.0 + sx, 16.0 + sy)), tf))
        return frames

    ref = make_wcs(shape)
    out_wcs, out_shape = build_output_grid(ref, shape, 1.0)

    # group_size=1 path: QM per-frame add (the worker branch).
    qm1 = make_fake_qm(shape)
    header = fits.Header()
    header["EXPTIME"] = 1.0
    for frame, tf in build_frames():
        rgb = np.stack([frame, frame, frame], axis=-1)
        qm1._add_frame_to_drizzle_accumulators(rgb, header, tf, np.ones(shape, np.float32))
    sci1 = np.stack([a.finalize() for a in qm1.drizzle_accumulators], axis=-1)
    wht1 = np.stack([a.wht for a in qm1.drizzle_accumulators], axis=-1)

    # group_size=8 path: same frames through drizzle_stream.
    accs8 = [DrizzleAccumulator(out_shape) for _ in range(3)]

    def gen():
        for frame, tf in build_frames():
            data = np.stack([frame, frame, frame], axis=0)
            yield (data, np.ones((3, *shape), np.float32), tf, 1.0, "counts")

    sci8, wht8 = drizzle_stream(accs8, gen(), group_size=8,
                                reference_wcs=ref, output_wcs=out_wcs)

    assert np.allclose(sci1, sci8, atol=1e-6, rtol=1e-5)
    assert np.allclose(wht1, wht8, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# 6. science validated BEFORE uint16 conversion
# ---------------------------------------------------------------------------


def test_science_validated_before_uint16(tmp_path):
    shape = (64, 64)
    qm = make_fake_qm(shape)

    # synthetic final science: background 0.01 + star 0.4 (normalised units).
    bg = 0.01
    star = 0.4
    frame = np.full(shape, bg, dtype=np.float32)
    frame = frame + gauss2d(shape, star - bg, 2.0, (32.0, 32.0))
    rgb = np.stack([frame, frame, frame], axis=-1)
    header = fits.Header()
    header["EXPTIME"] = 1.0
    qm._add_frame_to_drizzle_accumulators(rgb, header, identity_tf(), np.ones(shape, np.float32))

    qm.output_folder = str(tmp_path)
    qm.output_filename = "out.fit"
    qm.save_final_as_float32 = False
    qm.preserve_linear_output = False
    qm.drizzle_wht_threshold = 0.0
    qm.current_stack_header = fits.Header()
    qm.images_in_cumulative_stack = 1
    qm.total_exposure_seconds = 1.0
    qm.reproject_between_batches = False
    qm.reproject_coadd_final = False
    qm.cumulative_sum_memmap = None
    qm.cumulative_wht_memmap = None
    qm.reference_header_for_wcs = None
    qm.batch_size = 0
    qm._close_memmaps = lambda: None
    qm._wait_drizzle_processes = lambda: None

    qm._save_final_stack(output_filename_suffix="_drizzle_final")

    # validation stats captured on the *float* science, before uint16.
    stats = qm._m3_drizzle_validation_stats
    assert stats is not None
    assert abs(stats["median_per_channel"][0] - bg) < 0.01
    assert abs(stats["max_per_channel"][0] - star) < 0.05
    assert stats["wcs_scale_arcsec"] is not None
    assert stats["structure_ok"] is True

    sci_max = stats["max_per_channel"][0]
    sci_med = stats["median_per_channel"][0]

    saved = fits.getdata(qm.final_stacked_path)
    ch0 = saved[0] if saved.ndim == 3 else saved
    s = 65535.0 / max(sci_max, 1e-9)
    med = float(np.median(ch0))
    mx = float(np.max(ch0))
    # med ~ sci_med * s, max ~ sci_max * s == 65535 (full scale)
    assert abs(med - sci_med * s) < 0.02 * (sci_med * s + 1.0)
    assert abs(mx - 65535.0) <= 2.0


# ---------------------------------------------------------------------------
# M3-C2: tf câblé dans le chemin drizzle standard (blocker)
# ---------------------------------------------------------------------------


def _build_star_fields(bg=100.0, noise=1.0):
    """Synthetic 64x64 field: bright star at (24,26) in the reference, shifted
    by (-4,-6) in the source so it appears at (20,20).  Faint companions keep
    astroalign well-conditioned.  Both are normalised to [0, 1]."""
    from scipy.ndimage import shift as _ndshift

    shape = (64, 64)
    rng = np.random.default_rng(0)
    ref = np.full(shape, bg, dtype=np.float32)
    ref += rng.normal(0.0, noise, shape).astype(np.float32)
    ref += gauss2d(shape, 5000.0, 2.0, (24.0, 26.0))
    for _ in range(9):
        ref += gauss2d(shape, 800.0, 1.6, (rng.uniform(12, 52), rng.uniform(12, 52)))
    src = _ndshift(ref, shift=(-6.0, -4.0), order=1, mode="constant")
    src = np.clip(src, 0.0, None)
    return src / src.max(), ref / ref.max()


def _standard_qm_with_real_aligner(shape):
    """Minimal QM wired for the drizzle standard path with a REAL aligner."""
    from seestar.core.alignment import SeestarAligner

    qm = object.__new__(SeestarQueuedStacker)
    qm.align_on_disk = False
    qm.bayer_pattern = "GRBG"
    qm.correct_hot_pixels = False
    qm.drizzle_active_session = True
    qm.is_mosaic_run = False
    qm.use_quality_weighting = False
    qm.batch_size = 0
    qm.reproject_between_batches = False
    qm.reproject_coadd_final = False
    qm.reference_wcs_object = make_wcs(shape)
    qm.update_progress = lambda *a, **k: None
    qm.warned_unaligned_source_folders = set()
    qm.aligner = SeestarAligner()
    qm.aligner.update_progress = lambda *a, **k: None
    qm.drizzle_output_wcs, out_shape = build_output_grid(
        qm.reference_wcs_object, shape, 1.0
    )
    qm.drizzle_accumulators = [DrizzleAccumulator(out_shape) for _ in range(3)]
    return qm


def _centroid_above(img, frac=0.5):
    """Intensity-weighted centroid of the pixels above ``frac * max`` (isolates
    the bright star from faint companions and background)."""
    thr = float(np.max(img)) * frac
    m = img >= thr
    if not np.any(m):
        return None
    h, w = img.shape
    yy, xx = np.indices((h, w))
    tot = img[m].sum()
    return (img[m] * xx[m]).sum() / tot, (img[m] * yy[m]).sum() / tot


def test_standard_path_produces_tf(tmp_path):
    """The standard (non-mosaic) drizzle path must produce a tf: a real
    SeestarAligner aligns the synthetic field (star (20,20) -> reference
    (24,26)) and ``_process_file`` returns a non-None ``matrice_M_calculee``
    which, fed to the accumulator, lands the star at (24,26)."""
    shape = (64, 64)
    src, ref = _build_star_fields()
    qm = _standard_qm_with_real_aligner(shape)

    rgb = np.stack([src, src, src], axis=-1)
    hdr = fits.Header()
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = shape[1]
    hdr["NAXIS2"] = shape[0]
    hdr["NAXIS3"] = 3
    path = tmp_path / "in.fits"
    fits.PrimaryHDU(data=rgb, header=hdr).writeto(path, overwrite=True)
    ref_rgb = np.stack([ref, ref, ref], axis=-1).astype(np.float32)

    data, header, _scores, _wcs, matrix_m, mask = qm._process_file(
        str(path), ref_rgb, solve_astrometry_for_this_file=False
    )

    # BLOCKER: the standard drizzle path must now produce the tf.
    assert matrix_m is not None
    assert matrix_m.shape == (2, 3)
    # direction sanity: M maps the source star (20,20) towards the reference (24,26).
    p = matrix_m @ np.array([20.0, 20.0, 1.0])
    assert abs(p[0] - 24.0) < 0.5
    assert abs(p[1] - 26.0) < 0.5

    ok = qm._add_frame_to_drizzle_accumulators(data, header, matrix_m, mask)
    assert ok is True

    sci = qm.drizzle_accumulators[0].finalize()
    cx, cy = _centroid_above(sci, 0.5)
    assert cx is not None and cy is not None
    assert abs(cx - 24.0) < 0.1
    assert abs(cy - 26.0) < 0.1


def test_tf_direction_matches_aligner(monkeypatch):
    """Lock the tf convention: M maps ORIGINAL pixels to the reference grid,
    same direction as ``cv2.warpAffine`` (no inversion).  A deterministic exact
    translation (+4,+6) must map (20,20) -> (24,26) to 1e-4."""
    import astroalign as aa
    from skimage.transform import SimilarityTransform

    shape = (64, 64)
    src, ref = _build_star_fields()

    exact = SimilarityTransform(rotation=0.0, translation=(4.0, 6.0))
    monkeypatch.setattr(aa, "find_transform", lambda source, target: (exact, ([], [])))

    from seestar.core.alignment import SeestarAligner

    aligner = SeestarAligner()
    src_rgb = np.stack([src, src, src], axis=-1)
    ref_rgb = np.stack([ref, ref, ref], axis=-1).astype(np.float32)
    aligned, success, M = aligner._align_image(
        src_rgb, ref_rgb, "x", force_same_shape_as_ref=True, return_M=True
    )
    assert success
    assert M.shape == (2, 3)

    p = M @ np.array([20.0, 20.0, 1.0])
    assert abs(p[0] - 24.0) < 1e-4
    assert abs(p[1] - 26.0) < 1e-4

    # The aligned image must land the star at (24,26): the returned M is the
    # very matrix cv2.warpAffine used (no hidden inversion).
    ch0 = aligned[..., 0]
    assert np.unravel_index(np.argmax(ch0), ch0.shape) == (26, 24)


def test_native_wcs_fallback():
    """When tf is None but a native WCS is supplied (astrometry-single), the
    original pose is reprojected through its OWN WCS (identity tf) into the
    output grid and the flux lands at the right place."""
    shape = (32, 32)
    qm = make_fake_qm(shape)
    native_wcs = make_wcs(shape)

    frame = gauss2d(shape, 100.0, 1.5, (16.0, 16.0))
    rgb = np.stack([frame, frame, frame], axis=-1)
    header = fits.Header()
    header["EXPTIME"] = 1.0

    ok = qm._add_frame_to_drizzle_accumulators(
        rgb, header, None, np.ones(shape, np.float32), native_wcs=native_wcs
    )
    assert ok is True

    sci = qm.drizzle_accumulators[0].finalize()
    cx, cy = centroid(sci)
    assert abs(cx - 16.0) < 0.1
    assert abs(cy - 16.0) < 0.1
