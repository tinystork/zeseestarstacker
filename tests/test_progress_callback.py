import numpy as np
from types import SimpleNamespace, MethodType
from astropy.io import fits
from astropy.wcs import WCS
from seestar.queuep.queue_manager import SeestarQueuedStacker


def test_update_progress_tolerates_level_and_missing_params():
    dummy = SimpleNamespace(progress_callback=lambda msg: None, logger=None)
    # Should not raise even though callback only accepts one argument
    SeestarQueuedStacker.update_progress(dummy, "msg", progress=10.0, level="warning")


def test_finalize_streaming_with_simple_callback(tmp_path):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.cdelt = [0.1, 0.1]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = wcs.to_header()
    data = np.ones((5, 5), dtype=np.float32)
    fits.writeto(tmp_path / "aligned_00000.fits", data, hdr, overwrite=True)
    fits.writeto(tmp_path / "aligned_00001.fits", data * 2, hdr, overwrite=True)

    dummy = SimpleNamespace(
        progress_callback=lambda msg: None,
        logger=None,
        reference_wcs_object=None,
        reference_shape=None,
    )
    dummy.update_progress = MethodType(SeestarQueuedStacker.update_progress, dummy)

    out_fp = tmp_path / "final.fits"
    ok = SeestarQueuedStacker._finalize_reproject_and_coadd_streaming(
        dummy, str(tmp_path), str(out_fp)
    )
    assert ok
    assert out_fp.exists() and out_fp.stat().st_size > 0
