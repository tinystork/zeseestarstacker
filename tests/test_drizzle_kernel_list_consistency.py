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

D5 extensions: the tests also drive a REAL offscreen ``MainWindow`` so the
live combo items (not just the vocabulary constant) are checked against the
engine set in both directions, the backend seam (``split_backend_kwargs``)
is verified to carry exactly the engine-qualified strings, and the kernel
normalization boundaries are pinned as never silently remapping an offered
kernel: ``validate_drizzle_kernel`` is the identity for every offered kernel
(unknown names fall back to ``square`` WITH an explicit reason), the engine
``_normalize_effective_drizzle_config`` preserves the requested kernel, and
the boring final-combine resolver never consumes or remaps kernel names.

These tests pin the Qt list against the engine constant in **both**
directions (every Qt kernel is engine-supported AND every engine kernel is
offered), so a future divergence is caught at test time instead of producing
a "user selected X, engine did Y" discrepancy.

No display, no stacking, no FITS/PNG writes.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from seestar.core.drizzle_core import VALID_DRIZZLE_KERNELS, validate_drizzle_kernel
from seestar.gui_qt.main_window import DRIZZLE_KERNELS

EXPECTED_SIX = {"square", "gaussian", "point", "turbo", "lanczos2", "lanczos3"}


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    from seestar.gui_qt import create_application

    app = create_application([])
    assert app is QApplication.instance()
    return app


@pytest.fixture()
def window(qapp):
    from seestar.gui_qt import MainWindow

    win = MainWindow()
    yield win
    win.shutdown()


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


# ---------------------------------------------------------------------------
# D5: live-GUI combo == backend-accepted set == qualified engine set, and the
# kernel validation/resolution boundaries never silently remap a user kernel.
# ---------------------------------------------------------------------------


def _combo_items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_live_qt_kernel_combo_matches_engine_set_both_directions(window):
    """The REAL stacking-tab combo offers exactly the qualified engine set:
    every offered kernel is engine-qualified AND every engine kernel is
    offered (no unsupported kernel can be selected, none is missing)."""
    # Drizzle off by default -> combo disabled but still fully populated.
    assert window.drizzle_kernel_combo.isEnabled() is False
    offered = set(_combo_items(window.drizzle_kernel_combo))
    assert offered == VALID_DRIZZLE_KERNELS == EXPECTED_SIX
    assert len(offered) == window.drizzle_kernel_combo.count() == 6

    window.drizzle_check.setChecked(True)
    assert window.drizzle_kernel_combo.isEnabled() is True
    # Selection stays exact text; every offered kernel is selectable and
    # engine-qualified, so the request can never carry an unsupported kernel.
    for kernel in DRIZZLE_KERNELS:
        window.drizzle_kernel_combo.setCurrentText(kernel)
        assert window.drizzle_kernel_combo.currentText() == kernel
        state = window.collect_settings_state()
        assert state.drizzle_kernel == kernel
        request = window.build_run_request()
        assert request.backend_kwargs["drizzle_kernel"] == kernel


def test_backend_accepted_set_equals_engine_set(window):
    """The backend seam (split_backend_kwargs -> start_processing surface)
    accepts exactly the engine-qualified kernel strings."""
    from seestar.gui_qt.run_bridge import split_backend_kwargs

    window.drizzle_check.setChecked(True)
    for kernel in sorted(VALID_DRIZZLE_KERNELS):
        window.drizzle_kernel_combo.setCurrentText(kernel)
        request = window.build_run_request()
        start_kwargs, _seam = split_backend_kwargs(request.backend_kwargs)
        assert start_kwargs["drizzle_kernel"] == kernel
        assert "drizzle_kernel" not in _seam


def test_kernel_validation_never_remaps_an_offered_kernel(window):
    """validate_drizzle_kernel is the ONLY kernel-normalization boundary and
    it is the identity for every offered kernel (never a silent remap).
    Unsupported names fall back to 'square' WITH an explicit reason."""
    for kernel in DRIZZLE_KERNELS:
        eff, reason = validate_drizzle_kernel(kernel)
        assert eff == kernel
        assert reason is None

    # Engine effective-config normalization preserves the requested kernel.
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    qm = object.__new__(SeestarQueuedStacker)
    for kernel in DRIZZLE_KERNELS:
        qm.drizzle_kernel = kernel
        qm._normalize_effective_drizzle_config()
        assert qm.drizzle_kernel == kernel

    # Unknown kernels: a WELL-LOGGED deterministic fallback, never a silent
    # remap of a user kernel to a different one.
    eff, reason = validate_drizzle_kernel("tophat")
    assert eff == "square"
    assert reason and "tophat" in reason


def test_resolve_final_combine_never_consumes_or_remaps_kernels():
    """The final-combine alias resolver (boring_stack) is disjoint from the
    drizzle-kernel vocabulary: a kernel string handed to it passes through
    unchanged (no silent remap to another kernel), and canonical combine
    tokens resolve to themselves."""
    from seestar.gui.boring_stack import _resolve_final_combine

    class _Empty:
        stack_final_combine = None

    for kernel in DRIZZLE_KERNELS:
        resolved = _resolve_final_combine(kernel, _Empty())
        assert resolved == kernel  # pass-through, never remapped

    for canonical in ("mean", "median", "winsorized_sigma_clip", "reject"):
        assert _resolve_final_combine(canonical, _Empty()) == canonical
    assert _resolve_final_combine(None, _Empty()) == "mean"  # default


def test_offered_kernels_never_reach_a_remap_path_in_requests(window):
    """End-to-end contract: for every live-offered kernel the kernel carried
    by the request equals the combo text (text-based, no index remap)."""
    window.drizzle_check.setChecked(True)
    for kernel in _combo_items(window.drizzle_kernel_combo):
        window.drizzle_kernel_combo.setCurrentText(kernel)
        idx = window.drizzle_kernel_combo.currentIndex()
        assert window.drizzle_kernel_combo.itemText(idx) == kernel
        state = window.collect_settings_state()
        assert state.drizzle_kernel == kernel
        request = window.build_run_request()
        assert request.backend_kwargs["drizzle_kernel"] == kernel
        assert request.backend_kwargs["drizzle_kernel"] in VALID_DRIZZLE_KERNELS
