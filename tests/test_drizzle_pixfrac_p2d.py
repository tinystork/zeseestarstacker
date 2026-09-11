"""P2-D focused tests: canonical pixfrac policy, provenance, legacy handling.

Covers the bounded P2-D contract without instantiating GUIs: the canonical
policy helper, backend raw/effective/reason capture, run-contract provenance,
checkpoint fail-closed, resume-locator representability, UI ranges/kernel states
(source-level), and neutrality / PSR / iscale / no-conditioning assertions.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

dx = importlib.import_module("seestar.core.drizzle_core")
qm = importlib.import_module("seestar.queuep.queue_manager")
ck = importlib.import_module("seestar.core.drizzle_checkpoint")
rc = importlib.import_module("seestar.run_contract")
loc = importlib.import_module("seestar.resume_locator")


# --- 1) canonical policy helper ---------------------------------------------
@pytest.mark.parametrize(
    "kernel,editable,applicable,reason",
    [
        ("square", True, True, None),
        ("turbo", True, True, None),
        ("gaussian", True, True, None),
        ("lanczos2", False, True, dx.PIXFRAC_REASON_LANCZOS_FIXED),
        ("lanczos3", False, True, dx.PIXFRAC_REASON_LANCZOS_FIXED),
        ("point", False, False, dx.PIXFRAC_REASON_POINT_IGNORED),
    ],
)
def test_pixfrac_ui_policy(kernel, editable, applicable, reason):
    ed, val, rsn, app = dx.pixfrac_ui_policy(kernel)
    assert (ed, app, rsn) == (editable, applicable, reason)
    if not editable:
        assert val == dx.PIXFRAC_MAX == 1.0


def test_canonical_envelope_and_classification():
    assert dx.PIXFRAC_MIN == 0.01 and dx.PIXFRAC_MAX == 1.0
    eff, raw, rsn = dx.classify_drizzle_pixfrac(2.0)
    assert (eff, raw, rsn) == (1.0, 2.0, dx.PIXFRAC_REASON_GT_ONE)
    assert dx.classify_drizzle_pixfrac(1.0) == (1.0, 1.0, None)
    assert dx.classify_drizzle_pixfrac(0.5) == (0.5, 0.5, None)
    assert dx.classify_drizzle_pixfrac(0.005)[0] == dx.PIXFRAC_MIN
    assert dx.classify_drizzle_pixfrac("abc")[1] is None
    assert dx.classify_drizzle_pixfrac(float("nan"))[0] == 1.0


# --- 2) backend raw/effective/reason capture --------------------------------
def _stacker(**attrs):
    obj = object.__new__(qm.SeestarQueuedStacker)
    obj.drizzle_kernel = attrs.pop("kernel", "square")
    obj.drizzle_pixfrac = attrs.pop("pixfrac", 1.0)
    obj.drizzle_wht_threshold = 0.0
    obj.drizzle_scale = 1.0
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def test_backend_coerces_gt_one_and_preserves_raw_requested():
    obj = _stacker(pixfrac=2.0)
    obj._normalize_effective_drizzle_config()
    assert obj.drizzle_pixfrac == 1.0
    assert obj.drizzle_pixfrac_requested == 2.0
    assert obj.drizzle_pixfrac_reason == dx.PIXFRAC_REASON_GT_ONE


def test_backend_lanczos_effective_one():
    obj = _stacker(kernel="lanczos2", pixfrac=0.7)
    obj._normalize_effective_drizzle_config()
    assert obj.drizzle_pixfrac == 1.0
    assert obj.drizzle_pixfrac_requested == 0.7
    assert obj.drizzle_pixfrac_reason is None


# --- 3) run contract provenance --------------------------------------------
def test_run_contract_pixfrac_reason_round_trip():
    obj = _stacker(pixfrac=1.5)
    obj._normalize_effective_drizzle_config()
    cfg = ck.build_drizzle_canonical_config(obj, product_version="8.4.0")
    sci = cfg.scientific
    assert sci["drizzle_pixfrac_effective"] == 1.0
    assert sci["drizzle_pixfrac_requested"] == 1.5
    assert sci["drizzle_pixfrac_reason"] == dx.PIXFRAC_REASON_GT_ONE
    # provenance-only: not independently fingerprinted
    assert "drizzle_pixfrac_reason" not in rc._drizzle_fingerprint_payload(cfg)
    # legacy config without a reason is still accepted
    legacy = rc.RunConfig.from_sections(
        product_version="8.4.0",
        scientific={k: v for k, v in sci.items() if k != "drizzle_pixfrac_reason"},
    )
    assert "drizzle_pixfrac_reason" not in legacy.scientific


# --- 4) checkpoint fail-closed ---------------------------------------------
def test_checkpoint_canonical_pixfrac_gt_one_fails_closed():
    with pytest.raises(ck.DrizzleCheckpointError) as exc:
        ck._check_deposition_matches_canonical(
            "lanczos2", 1.0, "0.0",
            {"drizzle_kernel_effective": "lanczos2",
             "drizzle_pixfrac_effective": 1.5,
             "drizzle_fillval": "0.0"},
            "test",
        )
    assert dx.PIXFRAC_REASON_CHECKPOINT_GT_ONE in str(exc.value)


def test_checkpoint_canonical_pixfrac_at_boundary_ok():
    ck._check_deposition_matches_canonical(
        "square", 1.0, "0.0",
        {"drizzle_kernel_effective": "square",
         "drizzle_pixfrac_effective": 1.0,
         "drizzle_fillval": "0.0"},
        "test",
    )


# --- 5) resume locator representability ------------------------------------
def test_resume_locator_bound_is_canonical():
    src = inspect.getsource(loc)
    assert "0.01 <= pixfrac <= 1.0" in src
    assert "pixfrac_not_representable_by_ui" in src


# --- 6) UI surfaces (source-level; no GUI instantiation) -------------------
def test_ui_ranges_and_kernel_states():
    qt = (ROOT / "seestar/gui_qt/main_window.py").read_text(encoding="utf-8")
    tk = (ROOT / "seestar/gui/main_window.py").read_text(encoding="utf-8")
    mos = (ROOT / "seestar/gui/mosaic_gui.py").read_text(encoding="utf-8")
    assert "(0.01, 1.0, 0.05, 2)" in qt
    assert "setRange(0.01, 1.0)" in qt
    assert "to=1.00" in tk
    assert "to=1.00" in mos
    assert "to=2.00" not in tk and "to=2.00" not in mos
    assert "lanczos2" in qt and "point" in qt  # kernel-aware branch present
    assert "Fixed at 1.0" in qt and "Not applicable" in qt


# --- 7) neutrality / PSR / iscale / no-conditioning ------------------------
def test_no_conditioning_and_psr_iscale_untouched():
    assert not hasattr(dx, "CONDITIONING_THRESHOLD")
    src = inspect.getsource(dx.DrizzleAccumulator)
    assert "pixel_scale_ratio" in src
    assert "abs(" not in src.split("def add")[1].split("def finalize")[0]
    # iscale is never passed by the wrapper
    assert "iscale=" not in inspect.getsource(dx.DrizzleAccumulator.add)
    # P2-B geometry reason tokens unchanged
    assert dx.PIXEL_SCALE_RATIO_SOURCE == "wcs_output_input_ratio"


def test_pixfrac_policy_does_not_touch_accumulator_science():
    acc = dx.DrizzleAccumulator((4, 4), kernel="lanczos2", pixfrac=1.0)
    before_img = acc._out_img.copy()
    before_wht = acc._out_wht.copy()
    dx.pixfrac_ui_policy("lanczos2")
    dx.classify_drizzle_pixfrac(2.0)
    assert np.array_equal(before_img, acc._out_img)
    assert np.array_equal(before_wht, acc._out_wht)
