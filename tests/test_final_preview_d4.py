"""D4 tests: display-only signed preview stretch + zero-wall measurements.

Root-cause tracing for the interrupted Drizzle near-black preview / giant
zero wall, bounded to the BACKEND display path (``_save_final_stack`` display
data -> ``save_preview_image``/``stretch_display_data`` -> PNG/histogram):

1. scientific computation (signed Lanczos ringing) - legitimate, kept signed
   since D3.5;
2. output-domain clip - the OLD ``np.clip(...,0,None)`` created an artificial
   exact-zero population over signed science (fixed by D3.5: clip now lives
   only on the display/uint16 paths);
3. preview percentile stretch - the legacy stretch sampled ONLY strictly
   positive pixels (>0.001), so the black point landed inside the faint
   signal of a mostly-zero / signed frame and everything at or below it
   collapsed to an artificial pure-black wall (fixed by D4:
   ``stretch_display_data`` samples the FULL finite SIGNED distribution when
   negatives or a significant zero population are present);
4. interrupted/incomplete processing - partial accumulation lowers SNR and
   widens no-data support; the FITS header now records
   ``PROCSTAT=STOPPED_PARTIAL`` / ``ERROR_PARTIAL`` so interrupted output is
   identifiable as incomplete (no invented data).

Witness: kernel-aware (square / lanczos2 / lanczos3) measurements of zero
fraction, negative fraction, black point, low-percentile compression,
faint-signal visibility, point-source response and histogram shape.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import seestar.queuep.queue_manager as qm
from seestar.core.drizzle_core import DrizzleAccumulator
from seestar.core.image_processing import save_preview_image, stretch_display_data

import tests.test_drizzle_semantics_d2 as d2


def _gauss(shape, amp, sig, cx, cy):
    h, w = shape
    yy, xx = np.indices((h, w))
    return (amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig**2))).astype(
        np.float32
    )


def _kernel_display_frame(kernel, shape=(48, 48)):
    """Deterministic display frame per kernel: bright point source at
    fractional phases (Lanczos ringing -> signed for lanczos2/3), a faint
    diffuse halo near zero, and a large no-data zero region (simulating an
    interrupted/partial accumulation with big empty support)."""
    h, w = shape
    acc = DrizzleAccumulator((h, w), kernel=kernel, pixfrac=1.0)
    star = _gauss((h, w), 5000.0, 0.9, 24.35, 24.55)
    for dx, dy in ((0.0, 0.0), (0.5, 0.5), (-0.5, 0.25)):
        yy, xx = np.indices((h, w), dtype=np.float64)
        acc.add(star, np.ones((h, w), dtype=np.float32),
                np.dstack((xx + dx, yy + dy)).astype(np.float64))
    sci = acc.finalize("divide")
    # faint diffuse halo (0.05% of the point-source peak, broad) + large empty
    # region (right half = no-data zeros, as in an interrupted run).  The halo
    # is deliberately far below the point-source core so the LEGACY
    # positive-only black-point sampling lands inside it and crushes it.
    faint = _gauss((h, w), 0.0005 * float(np.max(sci)), 6.0, 14.0, 14.0)
    frame = sci + faint
    frame[:, w // 2 :] = 0.0
    return frame.astype(np.float32)


def _region_masks(shape):
    h, w = shape
    yy, xx = np.indices((h, w))
    faint_region = (xx - 14) ** 2 + (yy - 14) ** 2 <= 5 ** 2
    empty_region = xx >= w // 2
    return faint_region, empty_region


def _legacy_stretch_map(arr):
    """Reproduce the OLD positive-only stretch mapping for comparison."""
    flat = arr[np.isfinite(arr)]
    pos = flat[flat > 0.001]
    bp = float(np.percentile(pos, 1.0)) if pos.size >= 20 else float(np.min(flat))
    wp = float(np.percentile(pos, 99.0)) if pos.size >= 20 else float(np.max(flat))
    if wp <= bp + 1e-7:
        wp = bp + 1e-7
    out = np.clip((arr - bp) / (wp - bp + 1e-9), 0.0, 1.0)
    return out


@pytest.mark.parametrize("kernel", ["square", "lanczos2", "lanczos3"])
def test_d4_kernel_preview_witness(kernel):
    frame = _kernel_display_frame(kernel)
    faint_mask, empty_mask = _region_masks(frame.shape)
    finite = frame[np.isfinite(frame)]
    neg_frac = float(np.mean(finite < 0))
    zero_frac = float(np.mean(finite == 0))

    out, params = stretch_display_data(frame, enhanced_stretch=False, primary=True)
    assert out.shape == frame.shape
    assert np.all(np.isfinite(out))
    assert float(np.min(out)) >= 0.0 and float(np.max(out)) <= 1.0
    assert params["black_point"] <= params["white_point"]
    assert params["negative_fraction"] == pytest.approx(neg_frac)

    out_f = out[np.isfinite(out)]
    hist, _ = np.histogram(out_f, bins=20, range=(0.0, 1.0))
    out_zero_frac = float(np.mean(out_f == 0))
    # point-source response: the peak survives to the top of the display range
    assert float(np.max(out)) > 0.8
    # low-percentile compression: median display value stays low (image is
    # mostly empty sky/no-data), but is not crushed to an exact 0 wall.
    assert float(np.median(out_f)) < 0.2
    # faint-signal visibility: the faint diffuse halo renders clearly above
    # the empty/no-data floor (mean ratio > 3), i.e. no artificial zero wall
    # swallows the faint signal.
    mean_faint = float(np.mean(out[faint_mask]))
    mean_empty = float(np.mean(out[empty_mask]))
    assert mean_faint > 3.0 * max(mean_empty, 1e-6), (mean_faint, mean_empty)
    # histogram shape: no single bin holds the whole population (no fully
    # monolithic wall); the real no-data zeros may legitimately dominate but
    # the faint/star pixels must still occupy their own bins.
    assert float(np.max(hist)) / float(out_f.size) < 0.97

    # legacy mapping of the SAME signed frame crushes the faint signal into
    # the zero wall (bp sampled from the positive-only population):
    legacy = _legacy_stretch_map(frame)
    mean_faint_legacy = float(np.mean(legacy[faint_mask]))
    print(
        f"D4[{kernel}]: in neg_frac={neg_frac:.4f} zero_frac={zero_frac:.4f} "
        f"| bp={params['black_point']:.4g} wp={params['white_point']:.4g} "
        f"basis={params['basis']} | out zero_frac={out_zero_frac:.4f} "
        f"median={np.median(out_f):.4f} p1={np.percentile(out_f,1):.4f} "
        f"p10={np.percentile(out_f,10):.4f} p50={np.percentile(out_f,50):.4f} "
        f"| faint_mean={mean_faint:.4f} empty_mean={mean_empty:.4f} "
        f"legacy_faint_mean={mean_faint_legacy:.4f} | "
        f"hist_top_bin_frac={np.max(hist)/out_f.size:.4f}"
    )
    # On a signed frame the D4 stretch must NOT do worse on faint visibility
    # than the legacy positive-only mapping it replaces.
    assert mean_faint >= mean_faint_legacy - 1e-6


def test_d4_preview_png_end_to_end_signed(tmp_path):
    """The backend PNG writer handles the signed Lanczos frame end-to-end."""
    frame = _kernel_display_frame("lanczos3")
    png_path = str(tmp_path / "preview.png")
    ok = save_preview_image(frame, png_path, apply_stretch=True)
    assert ok
    from PIL import Image

    img = np.asarray(Image.open(png_path).convert("L"))
    assert img.shape == frame.shape
    assert img.min() >= 0 and img.max() <= 255
    assert img.max() > 200  # point source visible
    print(f"D4 PNG e2e: shape={img.shape} min={img.min()} max={img.max()} "
          f"black_px={(img == 0).sum()}/{img.size}")


def test_d4_interrupted_run_identifiable_in_header(tmp_path):
    """Interrupted/incomplete output is identifiable: PROCSTAT records
    STOPPED_PARTIAL / ERROR_PARTIAL in the FITS header (no invented data)."""
    obj = d2._stacker_dummy(tmp_path, save_as_float32=True)
    obj.preserve_linear_output = True
    sci = np.arange(16, dtype=np.float32).reshape(4, 4)
    obj.drizzle_accumulators = []
    for _ in range(3):
        acc = DrizzleAccumulator((4, 4), kernel="square", pixfrac=1.0)
        acc._out_img[:] = sci
        acc._out_wht[:] = 1.0
        obj.drizzle_accumulators.append(acc)
    # stopped-early run
    obj.processing_error = None
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_drizzle_final", stopped_early=True
    )
    hdr = fits.getheader(obj.final_stacked_path)
    assert hdr.get("PROCSTAT") == "STOPPED_PARTIAL"
    # error run
    obj2 = d2._stacker_dummy(tmp_path, save_as_float32=True)
    obj2.preserve_linear_output = True
    obj2.drizzle_accumulators = []
    for _ in range(3):
        acc = DrizzleAccumulator((4, 4), kernel="square", pixfrac=1.0)
        acc._out_img[:] = sci
        acc._out_wht[:] = 1.0
        obj2.drizzle_accumulators.append(acc)
    obj2.processing_error = "boom"
    qm.SeestarQueuedStacker._save_final_stack(
        obj2, output_filename_suffix="_drizzle_final", stopped_early=False
    )
    hdr2 = fits.getheader(obj2.final_stacked_path)
    assert hdr2.get("PROCSTAT") == "ERROR_PARTIAL"
    # clean run: no PROCSTAT
    obj3 = d2._stacker_dummy(tmp_path, save_as_float32=True)
    obj3.preserve_linear_output = True
    obj3.drizzle_accumulators = []
    for _ in range(3):
        acc = DrizzleAccumulator((4, 4), kernel="square", pixfrac=1.0)
        acc._out_img[:] = sci
        acc._out_wht[:] = 1.0
        obj3.drizzle_accumulators.append(acc)
    obj3.processing_error = None
    qm.SeestarQueuedStacker._save_final_stack(
        obj3, output_filename_suffix="_drizzle_final"
    )
    assert "PROCSTAT" not in fits.getheader(obj3.final_stacked_path)
