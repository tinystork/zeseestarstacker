"""Regression: QApplication must not leave a non-C libc numeric locale.

On a French-locale host, PySide6's QApplication calls setlocale(LC_NUMERIC, "")
and switches the decimal separator to comma, which historically made
cdrizzle.tdriz reject the fill-value string "0.0" ("Illegal fill value").
ZSSS bootstrap pins LC_NUMERIC to "C" so scientific processing is locale-safe,
while Qt UI localization (QLocale) stays independent of the libc numeric locale.
"""

import locale
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from seestar.core.drizzle_core import DrizzleAccumulator


def test_qt_bootstrap_pins_c_numeric_locale_and_drizzle_succeeds():
    from seestar.gui_qt import create_application
    create_application([])

    # Scientific invariant: the C numeric locale must be restored/pinned.
    assert locale.setlocale(locale.LC_NUMERIC, None) == "C"

    # Qt UI localization is independent of the libc numeric locale: it stays
    # the host locale (fr_FR here) rather than being forced to "C".
    from PySide6.QtCore import QLocale
    assert QLocale().name() != "C"

    h = w = 8
    acc = DrizzleAccumulator((h, w), kernel="square", pixfrac=1.0, fillval="0.0")
    yy, xx = np.indices((h, w), dtype=np.float64)
    data = np.ones((h, w), dtype=np.float32)
    weight = np.ones((h, w), dtype=np.float32)
    pixmap = np.dstack((xx, yy))
    in_grid = np.ones((h, w), dtype=bool)

    # Must not raise "Illegal fill value".
    acc.add(data, weight, pixmap, exptime=1.0, in_units="counts", in_grid_mask=in_grid)
    out = acc.finalize("divide")
    assert out.shape == (h, w)
    assert np.isfinite(out).all()
