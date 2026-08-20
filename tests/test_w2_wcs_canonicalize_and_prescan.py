"""Mission ZSSS W-2 — objective-level regression tests.

Two concerns are covered here:

1. ASTAP ``.wcs`` canonicalization: reading a triple-encoded ASTAP sidecar
   (CD + CDELT + PC all carrying the scale) must produce a single-encoded WCS
   (PC dimensionless + CDELT scale) with **zero** ``RuntimeWarning: cdelt will
   be ignored since cd is present`` and an unchanged celestial transform.

2. Conditional per-file pre-scan solve in ``_prepare_global_reprojection_grid``:
   when a fixed reference WCS already exists (``freeze_reference_wcs``), the
   per-file plate-solving is skipped (a fixed reference suffices for Reproject
   and Drizzle poses originales).  A mosaic (multiple WCS needed) keeps its
   pre-scan.
"""

import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from seestar.alignment.astrometry_solver import AstrometrySolver
from seestar.queuep.queue_manager import SeestarQueuedStacker


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _triple_encoded_astap_header(
    crpix=(540.5, 960.5), crval=(274.6802672266, -13.74280447684)
):
    """Build a header in the exact form ASTAP writes: PC, CDELT and CD all
    carrying the 2.37 arcsec/pix scale (PC should be dimensionless)."""
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


def _pure_cd_ground_truth(hdr):
    """Ground-truth WCS built from the CD matrix only."""
    hdr_cd = fits.Header()
    for k in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2",
              "CUNIT1", "CUNIT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        hdr_cd[k] = hdr[k]
    return WCS(hdr_cd, naxis=2, relax=True, fix=True)


def _celestial_wcs(shape_hw=(200, 200), crval=(274.68, -13.74)):
    """A minimal celestial WCS carrying a pixel_shape."""
    scale = 6.588577763717e-4
    w = WCS(naxis=2)
    w.wcs.crpix = [shape_hw[1] / 2.0, shape_hw[0] / 2.0]
    w.wcs.cdelt = [scale, scale]
    w.wcs.crval = list(crval)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.pc = np.array([[-0.9995, -0.0311], [0.0309, -0.9995]])
    w.pixel_shape = (shape_hw[1], shape_hw[0])
    return w


def _make_qm():
    """A bare SeestarQueuedStacker instance (no __init__) for unit tests."""
    qm = object.__new__(SeestarQueuedStacker)
    qm.update_progress = lambda *a, **k: None
    qm.stop_processing = False
    qm.all_input_filepaths = []
    qm.queue = None
    qm.freeze_reference_wcs = False
    qm.is_mosaic_run = False
    qm.reproject_between_batches = False
    qm.reproject_coadd_final = False
    qm.drizzle_active_session = False
    qm.batch_size = 0
    qm.local_solver_preference = "none"
    qm.astap_path = ""
    qm.astap_data_dir = None
    qm.astap_search_radius = 3.0
    qm.astap_downsample = 2
    qm.astap_sensitivity = 100
    qm.reference_pixel_scale_arcsec = None
    qm.stack_final_combine = None
    qm.reference_wcs_object = None
    qm.reference_shape = None
    qm.reference_header_for_wcs = None
    qm.ref_wcs_header = None
    qm.drizzle_scale = 1.0
    qm._ensure_memmaps_match_reference = lambda: None
    return qm


class _SpySolver:
    """Records every ``solve`` call and returns a fixed celestial WCS."""

    def __init__(self, wcs=None):
        self.calls = []
        self._wcs = wcs if wcs is not None else _celestial_wcs()

    def solve(self, path, hdr, settings, update_header_with_solution=False,
              batch_size=None, final_combine=None):
        self.calls.append(path)
        return self._wcs


def _write_fits(path, shape=(16, 16)):
    """Write a minimal valid FITS file so ``fits.getheader`` succeeds."""
    data = np.zeros(shape, dtype=np.float32)
    fits.writeto(path, data, overwrite=True)
    return str(path)


# ---------------------------------------------------------------------------
# 1. ASTAP triple-encoded WCS canonicalization
# ---------------------------------------------------------------------------

def test_astap_triple_encoded_wcs_no_cdelt_warning_mono_encoded(tmp_path):
    solver = AstrometrySolver()
    wcs_path = tmp_path / "astap_triple.wcs"
    wcs_path.write_text(_triple_encoded_astap_header().tostring(sep="\n") + "\n")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        parsed = solver._parse_wcs_file_content(str(wcs_path), (1920, 1080))

    cdelt_warnings = [
        w for w in captured
        if issubclass(w.category, RuntimeWarning) and "cdelt" in str(w.message).lower()
    ]
    assert cdelt_warnings == [], (
        f"unexpected cdelt RuntimeWarning: {[str(w.message) for w in cdelt_warnings]}"
    )

    assert parsed is not None and parsed.is_celestial
    # truly single-encoded: CD matrix deleted, PC (dimensionless) + CDELT (scale)
    assert parsed.wcs.has_cd() is False
    h = parsed.to_header(relax=True)
    assert "CD1_1" not in h and "CD2_2" not in h
    assert abs(h["PC1_1"]) == pytest.approx(0.9995, abs=1e-3)
    assert h["CDELT1"] == pytest.approx(6.5886e-4, rel=1e-3)


def test_astap_triple_encoded_wcs_transform_unchanged(tmp_path):
    """pixel->world must be identical to the pure-CD representation."""
    solver = AstrometrySolver()
    hdr = _triple_encoded_astap_header()
    wcs_path = tmp_path / "astap_triple2.wcs"
    wcs_path.write_text(hdr.tostring(sep="\n") + "\n")

    parsed = solver._parse_wcs_file_content(str(wcs_path), (1920, 1080))
    truth = _pure_cd_ground_truth(hdr)

    pix = np.array(
        [[0, 0], [1079, 0], [1079, 1919], [0, 1919], [540.5, 960.5], [10.3, 77.7]],
        dtype=float,
    )
    sky_parsed = parsed.pixel_to_world(pix[:, 0], pix[:, 1])
    sky_truth = truth.pixel_to_world(pix[:, 0], pix[:, 1])
    assert np.abs(sky_parsed.ra.deg - sky_truth.ra.deg).max() < 1e-9
    assert np.abs(sky_parsed.dec.deg - sky_truth.dec.deg).max() < 1e-9


# ---------------------------------------------------------------------------
# 2. Conditional pre-scan solve in _prepare_global_reprojection_grid
# ---------------------------------------------------------------------------

def test_prescan_skipped_when_fixed_reference_wcs(tmp_path):
    """Reproject / Drizzle with a fixed reference WCS must not re-solve files."""
    ref_wcs = _celestial_wcs(shape_hw=(1080, 1920))
    ref_hdr = ref_wcs.to_header(relax=True)
    ref_hdr["NAXIS1"] = 1920
    ref_hdr["NAXIS2"] = 1080

    qm = _make_qm()
    qm.freeze_reference_wcs = True
    qm.reference_wcs_object = ref_wcs
    qm.reference_header_for_wcs = ref_hdr
    qm.ref_wcs_header = ref_hdr
    qm.all_input_filepaths = [
        _write_fits(tmp_path / "a.fits"),
        _write_fits(tmp_path / "b.fits"),
        _write_fits(tmp_path / "c.fits"),
    ]
    qm.astrometry_solver = _SpySolver()

    ok = qm._prepare_global_reprojection_grid()

    assert ok is True
    assert qm.astrometry_solver.calls == [], (
        "per-file solve must be skipped when a fixed reference WCS exists"
    )
    # the fixed reference is inherited unchanged
    assert qm.reference_wcs_object is ref_wcs
    assert qm.reference_shape == (1080, 1920)


def test_prescan_active_for_mosaic(tmp_path):
    """A mosaic still solves every input file (multiple WCS required)."""
    spy = _SpySolver()
    qm = _make_qm()
    qm.is_mosaic_run = True
    qm.freeze_reference_wcs = False
    qm.reference_wcs_object = None
    m1 = _write_fits(tmp_path / "m1.fits")
    m2 = _write_fits(tmp_path / "m2.fits")
    qm.all_input_filepaths = [m1, m2]
    qm.astrometry_solver = spy

    ok = qm._prepare_global_reprojection_grid()

    assert ok is True
    assert sorted(spy.calls) == sorted([m1, m2])
    assert qm.reference_wcs_object is not None


def test_prescan_solves_when_no_reference_yet(tmp_path):
    """Without a fixed reference the pre-scan still solves to establish it."""
    spy = _SpySolver()
    qm = _make_qm()
    qm.freeze_reference_wcs = True
    qm.reference_wcs_object = None
    single = _write_fits(tmp_path / "single.fits")
    qm.all_input_filepaths = [single]
    qm.astrometry_solver = spy

    ok = qm._prepare_global_reprojection_grid()

    assert ok is True
    assert spy.calls == [single]
    assert qm.reference_wcs_object is not None
