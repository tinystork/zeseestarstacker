"""ZESOLVER-GATE-01: the frozen reference WCS survives the _worker re-selection.

When a global reference is plate-solved once in ``start_processing``, its
celestial WCS is frozen in ``self.reference_wcs_object``.  ``_worker`` then runs
``_get_reference_image`` again and overwrites ``reference_header_for_wcs`` with
the ORIGINAL header (no WCS).  ``_reinject_frozen_reference_wcs`` must restore the
frozen WCS so every classic batch inherits the same immutable grid and is never
re-solved downstream.  Plain-classic modes (no solved WCS) are a no-op.
"""

from __future__ import annotations

from astropy.io import fits
from astropy.wcs import WCS

from seestar.queuep.queue_manager import SeestarQueuedStacker


def _make_qm() -> SeestarQueuedStacker:
    qm = object.__new__(SeestarQueuedStacker)
    qm.reference_wcs_object = None
    qm.reference_header_for_wcs = None
    qm.ref_wcs_header = None
    return qm


def _celestial_wcs() -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [960.0, 540.0]
    w.wcs.cdelt = [-0.00066, 0.00066]
    w.wcs.crval = [274.6, -13.7]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_reinject_preserves_celestial_wcs() -> None:
    qm = _make_qm()
    qm.reference_wcs_object = _celestial_wcs()
    qm.reference_header_for_wcs = fits.Header()
    qm.ref_wcs_header = qm.reference_header_for_wcs

    qm._reinject_frozen_reference_wcs()

    assert "CTYPE1" in qm.reference_header_for_wcs
    assert "CRVAL1" in qm.reference_header_for_wcs
    assert WCS(qm.reference_header_for_wcs, naxis=2).has_celestial


def test_reinject_is_noop_when_no_solved_wcs() -> None:
    qm = _make_qm()
    qm.reference_wcs_object = None
    qm.reference_header_for_wcs = fits.Header()
    qm.ref_wcs_header = qm.reference_header_for_wcs

    qm._reinject_frozen_reference_wcs()

    assert "CTYPE1" not in qm.reference_header_for_wcs
    assert "CRVAL1" not in qm.reference_header_for_wcs
