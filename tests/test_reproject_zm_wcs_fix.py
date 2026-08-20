"""Regression tests for the mode-0 (batch_size=0) Reproject&Coadd WCS fix.

Root cause fixed here:
- ``_canonicalize_wcs_scale`` (astrometry_solver): ASTAP writes ``.wcs`` files
  with the pixel scale encoded in CD, CDELT *and* PC simultaneously; astropy
  keeps all three and wcslib then uses ``PC x CDELT`` so the effective scale
  becomes ``scale^2`` (~0.0016 arcsec/pix for a genuine 2.37 arcsec/pix Seestar
  solution).  Every transform is corrupted and the "Reference WCS pixel scale
  ... outside [0.1, 30.0]; clipping" warning fires.  The fix rebuilds
  PC (dimensionless rotation) + CDELT (scale) from the CD matrix.
- ``_REFERENCE_WCS_KEYS`` / ``_header_has_wcs_keywords`` (queue_manager): the
  mode-0 batch WCS inheritance copied CDELT but not PC, so batches received a
  scale-1 WCS; the final reproject then squeezed every batch into a sub-pixel
  blob producing the "uniform white" final.  PC1_1..PC2_2 are now copied too.
"""

import numpy as np
import pytest

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from seestar.alignment.astrometry_solver import (
    AstrometrySolver,
    _canonicalize_wcs_scale,
)
from seestar.queuep.queue_manager import (
    SeestarQueuedStacker,
    _REFERENCE_WCS_KEYS,
    _header_has_wcs_keywords,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _triple_encoded_astap_header(crpix=(540.5, 960.5), crval=(274.6802672266, -13.74280447684)):
    """Build a header in the exact form ASTAP v2026.03.20 writes: PC, CDELT
    and CD all carrying the 2.37 arcsec/pix scale (PC should be dimensionless)."""
    scale = 6.588577763717e-4  # deg/pix (~2.372 arcsec/pix)
    rot = np.array(
        [[-0.99952343976455, -0.031080508534006], [0.030868970848453, -0.99951688429424]]
    )
    cd = rot * scale
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["CRPIX1"] = crpix[0]
    hdr["CRPIX2"] = crpix[1]
    hdr["CRVAL1"] = crval[0]
    hdr["CRVAL2"] = crval[1]
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    # triple encoding, as ASTAP writes it:
    hdr["PC1_1"] = cd[0, 0]
    hdr["PC1_2"] = cd[0, 1]
    hdr["PC2_1"] = cd[1, 0]
    hdr["PC2_2"] = cd[1, 1]
    hdr["CDELT1"] = scale
    hdr["CDELT2"] = scale
    hdr["CD1_1"] = cd[0, 0]
    hdr["CD1_2"] = cd[0, 1]
    hdr["CD2_1"] = cd[1, 0]
    hdr["CD2_2"] = cd[1, 1]
    hdr["RADESYS"] = "FK5"
    hdr["EQUINOX"] = 2000.0
    hdr["LONPOLE"] = 180.0
    hdr["LATPOLE"] = crval[1]
    return hdr


def _wcs_scale_arcsec(wcs_obj) -> float:
    cd = wcs_obj.wcs.pc @ np.diag(wcs_obj.wcs.cdelt)
    return float(np.sqrt(abs(np.linalg.det(cd))) * 3600.0)


def _scales_arcsec(wcs_obj):
    scales = proj_plane_pixel_scales(wcs_obj)
    out = []
    for s in scales:
        if hasattr(s, "to_value"):
            s = s.to_value("deg")
        out.append(float(s) * 3600.0)
    return out


# ---------------------------------------------------------------------------
# 1. canonicalization of the triple-encoded ASTAP WCS
# ---------------------------------------------------------------------------

def test_canonicalize_triple_encoded_wcs_scale():
    hdr = _triple_encoded_astap_header()
    w = WCS(hdr, naxis=2, relax=True, fix=True)

    # Before the fix: astropy keeps PC=scale AND CDELT=scale, wcslib applies
    # PC x CDELT -> bogus scale^2 (~0.0016 arcsec/pix) and broken transforms.
    assert _wcs_scale_arcsec(w) < 0.01  # scale^2 regime (broken)

    _canonicalize_wcs_scale(w)

    assert _wcs_scale_arcsec(w) == pytest.approx(2.3727, abs=1e-3)
    arcsec = _scales_arcsec(w)
    assert arcsec[0] == pytest.approx(2.3727, abs=1e-3)
    assert arcsec[1] == pytest.approx(2.3727, abs=1e-3)

    # to_header must be single-encoded: CDELT carries the scale, PC is
    # dimensionless, no CD keywords.
    h2 = w.to_header(relax=True)
    assert h2["CDELT1"] == pytest.approx(6.5886e-4, rel=1e-3)
    assert h2["CDELT2"] == pytest.approx(6.5886e-4, rel=1e-3)
    assert abs(h2["PC1_1"]) == pytest.approx(0.9995, abs=1e-3)
    assert abs(h2["PC2_2"]) == pytest.approx(0.9995, abs=1e-3)
    assert "CD1_1" not in h2
    assert "CD2_2" not in h2

    # round-trip: re-parsing the canonical header keeps the correct scale
    w_re = WCS(h2, naxis=2)
    assert _wcs_scale_arcsec(w_re) == pytest.approx(2.3727, abs=1e-3)


def test_canonicalize_transform_preserved_against_ground_truth():
    """After canonicalization, pixel->world is identical to the pure-CD
    representation of the same solution (transform invariance)."""
    hdr = _triple_encoded_astap_header()
    w = WCS(hdr, naxis=2, relax=True, fix=True)

    # pure-CD WCS: the canonical ground truth built from the same CD matrix
    cd = np.asarray(w.wcs.cd, dtype=float)
    hdr_cd = fits.Header()
    hdr_cd["CRPIX1"] = hdr["CRPIX1"]
    hdr_cd["CRPIX2"] = hdr["CRPIX2"]
    hdr_cd["CRVAL1"] = hdr["CRVAL1"]
    hdr_cd["CRVAL2"] = hdr["CRVAL2"]
    hdr_cd["CTYPE1"] = "RA---TAN"
    hdr_cd["CTYPE2"] = "DEC--TAN"
    hdr_cd["CUNIT1"] = "deg"
    hdr_cd["CUNIT2"] = "deg"
    hdr_cd["CD1_1"] = cd[0, 0]
    hdr_cd["CD1_2"] = cd[0, 1]
    hdr_cd["CD2_1"] = cd[1, 0]
    hdr_cd["CD2_2"] = cd[1, 1]
    w_truth = WCS(hdr_cd, naxis=2, relax=True, fix=True)

    _canonicalize_wcs_scale(w)
    pix = np.array(
        [[0, 0], [1079, 0], [1079, 1919], [0, 1919], [540.5, 960.5]], dtype=float
    )
    sky_canon = w.pixel_to_world(pix[:, 0], pix[:, 1])
    sky_truth = w_truth.pixel_to_world(pix[:, 0], pix[:, 1])
    assert np.abs(sky_canon.ra.deg - sky_truth.ra.deg).max() < 1e-9
    assert np.abs(sky_canon.dec.deg - sky_truth.dec.deg).max() < 1e-9


def test_parse_astap_wcs_file_now_canonical(tmp_path):
    """The real parse path (used for every ASTAP solution) returns a
    single-encoded WCS with the physically correct pixel scale."""
    solver = AstrometrySolver()
    wcs_path = tmp_path / "astap_solution.wcs"
    wcs_path.write_text(_triple_encoded_astap_header().tostring(sep="\n") + "\n")

    parsed = solver._parse_wcs_file_content(str(wcs_path), (1920, 1080))

    assert parsed is not None
    assert parsed.is_celestial
    # image_shape_hw=(1920,1080) -> pixel_shape=(naxis1, naxis2)=(1080,1920)
    assert parsed.pixel_shape == (1080, 1920)
    assert _wcs_scale_arcsec(parsed) == pytest.approx(2.3727, abs=1e-3)
    # no double encoding in the round-tripped header
    h = parsed.to_header(relax=True)
    assert "CD1_1" not in h
    assert abs(h["PC1_1"]) == pytest.approx(0.9995, abs=1e-3)


def test_canonicalize_leaves_single_encoded_wcs_untouched():
    """WCSes without a CD matrix (already single-encoded) are unchanged."""
    w = WCS(naxis=2)
    w.wcs.crpix = [540.5, 960.5]
    w.wcs.cdelt = [6.5886e-4, 6.5933e-4]
    w.wcs.crval = [274.68, -13.74]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    pc_before = w.wcs.pc.copy()
    cdelt_before = w.wcs.cdelt.copy()
    _canonicalize_wcs_scale(w)
    assert np.allclose(w.wcs.pc, pc_before)
    assert np.allclose(w.wcs.cdelt, cdelt_before)


# ---------------------------------------------------------------------------
# 2. header WCS detection
# ---------------------------------------------------------------------------

def test_header_has_wcs_keywords():
    h_cd = fits.Header()
    for k in ("CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        h_cd[k] = 1.0 if k.startswith(("CRV", "CTY")) else -1.0e-4
    assert _header_has_wcs_keywords(h_cd)

    h_pc = fits.Header()
    for k in ("CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2", "CDELT1", "CDELT2",
              "PC1_1", "PC1_2", "PC2_1", "PC2_2"):
        h_pc[k] = 1.0 if k.startswith(("CRV", "CTY")) else 6.6e-4
    assert _header_has_wcs_keywords(h_pc)

    h_partial = fits.Header()
    h_partial["CRVAL1"] = 1.0
    h_partial["CRVAL2"] = 2.0
    h_partial["CDELT1"] = 6.6e-4  # CDELT alone is NOT a usable WCS
    assert not _header_has_wcs_keywords(h_partial)

    assert not _header_has_wcs_keywords(fits.Header())
    assert not _header_has_wcs_keywords(None)


def test_reference_wcs_keys_contains_pc_and_cdelt():
    keys = set(_REFERENCE_WCS_KEYS)
    assert {"CDELT1", "CDELT2", "PC1_1", "PC1_2", "PC2_1", "PC2_2"} <= keys


# ---------------------------------------------------------------------------
# 3. end-to-end: mode-0 final pass keeps the science with the artifact-form
#    reference header (PC carries the scale, CDELT=1.0 -- the real M16 state)
# ---------------------------------------------------------------------------

def _make_fake_qm(out_dir, ref_wcs, ref_shape, ref_header):
    qm = object.__new__(SeestarQueuedStacker)
    qm.output_folder = out_dir
    qm.output_filename = ""  # real runs leave it empty -> stack_final{suffix}.fit
    qm.batch_size = 0
    qm.reproject_coadd_final = True
    qm.reproject_between_batches = False
    qm.drizzle_active_session = False
    qm.is_mosaic_run = False
    qm.drizzle_mode = "Final"
    qm.freeze_reference_wcs = True
    qm.reference_wcs_object = ref_wcs
    qm.reference_shape = ref_shape
    qm.reference_header_for_wcs = ref_header
    qm.ref_wcs_header = ref_header
    qm.unsolved_classic_batch_files = set()
    qm.apply_master_tile_crop = False
    qm.master_tile_crop_percent_decimal = 0.0
    qm.solve_batches = False
    qm.images_in_cumulative_stack = 2
    qm.cumulative_sum_memmap = None
    qm.cumulative_wht_memmap = None
    qm.save_final_as_float32 = False
    qm.preserve_linear_output = False
    qm.current_stack_header = fits.Header()
    qm.drizzle_wht_threshold = 0.0
    qm.drizzle_output_wcs = None
    qm.raw_adu_data_for_ui_histogram = None
    qm.last_saved_data_for_preview = None
    qm.feathering_applied_in_session = False
    qm.scnr_applied = False
    qm.neutralize_background_applied = False
    qm.background_subtracted_applied = False
    qm.edge_crop_applied = False
    qm.apply_scnr = False
    qm.neutralize_background_automatic = False
    qm.apply_edge_crop = False
    qm.background_subtraction_photutils = False
    qm.save_fits_png = False
    qm.postprocess_background_subtraction_photutils = False
    qm.low_wht_mask_value = 0.0
    qm.feather_blur_px = 256
    qm.total_exposure_seconds = 20.0
    qm.output_folder_for_final = None
    qm.suppress_save = False
    qm.progress_callback = None
    qm.preview_callback = None
    qm.gui_event_queue = None
    qm.stop_processing = False
    qm.processing_error = None
    qm.final_stacked_path = None
    qm.current_stack_data = None
    qm.current_stack_data_raw = None
    qm.settings = None
    qm._match_background_for_final_set = False
    qm.match_background_for_final = None
    qm._wait_drizzle_processes = lambda: None
    qm._close_memmaps = lambda: None
    qm._move_to_stacked = lambda *a, **k: None
    qm._update_batch_count_file = lambda *a, **k: None
    qm._update_batches_meta = lambda *a, **k: None
    qm._save_partial_stack = lambda *a, **k: None
    qm._increment_aligned_counter = lambda *a, **k: None
    qm._send_eta_update = lambda *a, **k: None
    qm._move_to_unaligned = lambda *a, **k: None
    qm._update_preview_sum_w = lambda *a, **k: None
    qm.update_progress = lambda *a, **k: None
    return qm


def _synthetic_batch(h, w, star_pos, star_amp, bg):
    img = np.full((h, w, 3), bg, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-(((xx - star_pos[0]) ** 2 + (yy - star_pos[1]) ** 2) / (2 * 2.5 ** 2)))
    for c in range(3):
        img[..., c] += star_amp[c] * g
    return img


def test_mode0_final_keeps_science_with_artifact_reference_header(tmp_path):
    h, w = 64, 64
    scale = 6.5886e-4  # deg/pix (2.372 arcsec/pix)
    # canonical reference WCS as returned by the fixed ASTAP parse
    ref_wcs = WCS(naxis=2)
    ref_wcs.wcs.crpix = [32.5, 32.5]
    ref_wcs.wcs.cdelt = [scale, scale]
    ref_wcs.wcs.crval = [274.68, -13.74]
    ref_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    ref_wcs.wcs.pc = np.array([[-0.9995, -0.0311], [0.0309, -0.9995]])
    ref_wcs.pixel_shape = (w, h)

    # reference header in the REAL M16 artifact form: PC carries the scale,
    # CDELT=1.0 (the form the worker stores from the dataset frames).
    ref_header = fits.Header()
    ref_header["CRPIX1"] = 32.5
    ref_header["CRPIX2"] = 32.5
    ref_header["CRVAL1"] = 274.68
    ref_header["CRVAL2"] = -13.74
    ref_header["CTYPE1"] = "RA---TAN"
    ref_header["CTYPE2"] = "DEC--TAN"
    ref_header["CUNIT1"] = "deg"
    ref_header["CUNIT2"] = "deg"
    ref_header["CDELT1"] = 1.0
    ref_header["CDELT2"] = 1.0
    ref_header["PC1_1"] = -0.00065849
    ref_header["PC1_2"] = -0.00002041
    ref_header["PC2_1"] = 0.00002029
    ref_header["PC2_2"] = -0.00065907

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    batch_dir = out_dir / "classic_batch_outputs"
    batch_dir.mkdir()

    star_amp = (0.40, 0.32, 0.22)
    bg = 0.0124
    batch_files = []
    for i in range(1, 3):
        img = _synthetic_batch(h, w, (30.0, 32.0), star_amp, bg)
        hdr = fits.Header()
        hdr["NAXIS"] = 3
        hdr["NAXIS1"] = w
        hdr["NAXIS2"] = h
        hdr["NAXIS3"] = 3
        sci = batch_dir / f"classic_batch_{i:03d}.fits"
        fits.PrimaryHDU(data=np.moveaxis(img, -1, 0), header=hdr).writeto(sci, overwrite=True)
        whts = []
        for c in range(3):
            wp = batch_dir / f"classic_batch_{i:03d}_wht_{c}.fits"
            fits.PrimaryHDU(data=np.ones((h, w), dtype=np.float32)).writeto(wp, overwrite=True)
            whts.append(str(wp))
        batch_files.append((str(sci), whts))

    qm = _make_fake_qm(str(out_dir), ref_wcs, (h, w), ref_header)
    ok = qm._reproject_classic_batches_zm(batch_files)
    assert ok

    final = out_dir / "stack_final_classic_reproject_zm.fit"
    assert final.exists(), "final FITS not produced"
    data = fits.getdata(final)
    hdr_final = fits.getheader(final)

    assert data.dtype == np.uint16
    med = [float(np.median(data[c])) for c in range(3)]
    mx = [float(np.max(data[c])) for c in range(3)]
    # background survives (median in the batch-background regime, not "white")
    assert all(0.0 < m < 30000 for m in med), f"median exploded: {med}"
    # the star survives: max well above median in every channel
    for c in range(3):
        assert mx[c] > 3.0 * med[c], f"channel {c}: max {mx[c]} not above median {med[c]}"
    # not a uniform image
    assert mx[0] - med[0] > 1000

    # final WCS is single-encoded with the physically correct pixel scale
    w_final = WCS(hdr_final, naxis=2)
    assert _wcs_scale_arcsec(w_final) == pytest.approx(2.3727, abs=1e-2)
    h2 = w_final.to_header(relax=True)
    assert "CD1_1" not in h2
    assert abs(h2["PC1_1"]) == pytest.approx(0.9995, abs=1e-3)
