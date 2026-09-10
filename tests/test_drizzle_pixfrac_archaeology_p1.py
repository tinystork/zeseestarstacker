"""Phase-1 pixfrac archaeology probes (ZSSS-DRIZZLE-CLOSURE-P1, item 11).

Executable evidence, produced WITHOUT any product change:

* real ``drizzle`` 2.2.0 probes for every offered kernel at 0.5 / 0.8 / 1.0 and
  the legacy ``>1`` value (1.5);
* Lanczos2/3 document the ignored/forced-1.0 semantics (bit-identical output
  for every pixfrac);
* the backend coercion ``validate_drizzle_pixfrac`` normalizes outside
  ``(0, 1]`` to ``1.0`` (so a raw engine call is never handed a legacy ``>1``);
* the persisted resume locator still accepts an effective value up to ``2.0``
  (documented as archaeology only — no product endorsement of ``>1``).
"""

import inspect

import numpy as np
import pytest

from seestar.core import drizzle_science_diagnostics as dsd
from seestar.core.drizzle_core import (
    VALID_DRIZZLE_KERNELS,
    validate_drizzle_pixfrac,
)
from seestar import resume_locator


def _probe(kernel, pixfrac, shift=0.5):
    from drizzle.resample import Drizzle

    out = np.zeros((16, 16), dtype=np.float32)
    wht = np.zeros((16, 16), dtype=np.float32)
    engine = Drizzle(out_img=out, out_wht=wht, kernel=kernel, fillval="0.0")
    data = np.zeros((8, 8), dtype=np.float32)
    data[3:5, 3:5] = 1.0
    yy, xx = np.indices((8, 8), dtype=np.float64)
    pixmap = np.dstack((xx + 4.0 + shift, yy + 4.0))
    w = np.ones((8, 8), dtype=np.float32)
    engine.add_image(data=data, exptime=1.0, pixmap=pixmap, weight_map=w,
                     in_units="counts", pixfrac=pixfrac, wht_scale=1.0)
    return out.copy(), wht.copy()


PIXFRACS = (0.5, 0.8, 1.0, 1.5)


@pytest.mark.parametrize("kernel", ["lanczos2", "lanczos3"])
def test_lanczos_ignores_pixfrac_for_every_offered_value(kernel):
    base_out, base_wht = _probe(kernel, 1.0)
    for pf in PIXFRACS:
        out, wht = _probe(kernel, pf)
        assert np.array_equal(out, base_out), (kernel, pf)
        assert np.array_equal(wht, base_wht), (kernel, pf)


def test_point_kernel_ignores_pixfrac():
    base_out, base_wht = _probe("point", 1.0)
    for pf in (0.5, 1.5):
        out, wht = _probe("point", pf)
        assert np.array_equal(out, base_out)
        assert np.array_equal(wht, base_wht)


def test_square_and_turbo_pixfrac_is_active_and_gt1_spreads():
    for kernel in ("square", "turbo"):
        counts = {}
        for pf in (0.5, 1.0, 1.5):
            out, _ = _probe(kernel, pf)
            counts[pf] = int(np.count_nonzero(out))
        assert counts[0.5] <= counts[1.0] <= counts[1.5], (kernel, counts)


def test_gaussian_pixfrac_changes_weight_distribution():
    _, w05 = _probe("gaussian", 0.5)
    _, w10 = _probe("gaussian", 1.0)
    assert not np.array_equal(w05, w10)


def test_validate_drizzle_pixfrac_normalizes_outside_zero_one():
    assert validate_drizzle_pixfrac(0.5) == (0.5, None)
    assert validate_drizzle_pixfrac(1.0) == (1.0, None)
    for bad in (1.5, 2.0, 0.0, -0.1, float("nan"), float("inf"), "x", None):
        value, reason = validate_drizzle_pixfrac(bad)
        assert value == 1.0
        assert reason


def test_offered_kernel_set_is_the_engine_set():
    assert VALID_DRIZZLE_KERNELS == frozenset(
        {"square", "gaussian", "point", "turbo", "lanczos2", "lanczos3"}
    )


def test_resume_locator_still_accepts_effective_pixfrac_up_to_two():
    """Archaeology: the resume locator bound is <= 2.0 (not a >1 endorsement)."""
    src = inspect.getsource(resume_locator)
    assert "0.01 <= pixfrac <= 2.0" in src


def test_contract_diagnostic_never_reports_a_candidate_as_effective():
    contract = dsd.contract_diagnostic("lanczos3", 1.0, exptime=1.0)
    assert contract["pixel_scale_ratio_effective"] == 1.0
    assert contract["pixel_scale_ratio_source"] == "upstream_default"
    assert contract["iscale_effective"] == 1.0
