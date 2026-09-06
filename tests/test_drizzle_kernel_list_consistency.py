"""Drizzle-kernel list consistency (A6).

The drizzle 2.2.0 engine supports exactly six kernels — ``square``,
``gaussian``, ``point``, ``turbo``, ``lanczos2``, ``lanczos3`` — and
**rejects** ``tophat`` (``resample.py`` raises ``ValueError``).  Every
user-facing kernel surface must therefore contain exactly the engine set and
must never offer ``tophat``:

* the Qt stacking-tab combo vocabulary ``DRIZZLE_KERNELS`` (and the Mosaic
  kernel combo that reuses it),
* the Qt resume-locator "representable by this UI" allowlist,
* the Tk settings validation list and the Tk kernel combo.

These tests pin the Qt list against the engine constant in **both**
directions (every Qt kernel is engine-supported AND every engine kernel is
offered), so a future divergence is caught at test time instead of producing
a "user selected X, engine did Y" discrepancy.

No display, no stacking, no FITS/PNG writes.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from seestar.core.drizzle_core import VALID_DRIZZLE_KERNELS
from seestar.gui_qt.main_window import DRIZZLE_KERNELS

EXPECTED_SIX = {"square", "gaussian", "point", "turbo", "lanczos2", "lanczos3"}


def test_engine_valid_set_is_the_six_supported_kernels():
    assert VALID_DRIZZLE_KERNELS == EXPECTED_SIX
    assert "tophat" not in VALID_DRIZZLE_KERNELS


def test_qt_kernel_list_is_exactly_the_engine_set():
    # Qt combo vocabulary == engine VALID set (both directions).
    assert set(DRIZZLE_KERNELS) == VALID_DRIZZLE_KERNELS
    assert set(DRIZZLE_KERNELS) == EXPECTED_SIX


def test_qt_kernel_list_has_no_duplicates_and_no_tophat():
    assert len(DRIZZLE_KERNELS) == len(set(DRIZZLE_KERNELS)) == 6
    assert "tophat" not in DRIZZLE_KERNELS
    # The combo is fed directly from this list; an unsupported kernel can
    # therefore never be selected (no silent remap needed).
    for kernel in DRIZZLE_KERNELS:
        assert kernel in VALID_DRIZZLE_KERNELS


def test_qt_mosaic_kernel_combo_reuses_the_same_list():
    # The Mosaic "kernel" field spec is the same DRIZZLE_KERNELS vocabulary.
    from seestar.gui_qt.main_window import MOSAIC_FIELDS

    mosaic_kernel = next(field for field in MOSAIC_FIELDS if field[0] == "kernel")
    assert tuple(mosaic_kernel[3]) == tuple(DRIZZLE_KERNELS)
