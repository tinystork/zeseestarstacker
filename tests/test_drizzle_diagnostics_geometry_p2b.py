"""P2-B rework-2: diagnostics truthfulness for the frozen geometry fact."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

dsd = importlib.import_module("seestar.core.drizzle_science_diagnostics")
drizzle_core = importlib.import_module("seestar.core.drizzle_core")

SRC = drizzle_core.PIXEL_SCALE_RATIO_SOURCE
N = 32


def _wcs(plate=2.4e-4):
    w = WCS(naxis=2)
    w.wcs.crpix = [N / 2, N / 2]
    w.wcs.cdelt = [-plate, plate]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.array_shape = (N, N)
    return w


def _diag(tmp_path, effective, source, requested=None):
    d = dsd.DrizzleScienceDiagnostics(run_token="t", output_folder=str(tmp_path))
    d.set_run_config(kernel="lanczos2", scale=3.0, pixfrac_requested=1.0,
                     pixfrac_effective=1.0)
    d.set_geometry(
        dsd.geometry_diagnostic(_wcs(), _wcs(2.4e-4 / 3.0), kernel="lanczos2", scale=3.0,
                                pixel_scale_ratio_effective=effective,
                                pixel_scale_ratio_source=source)
    )
    d.set_pixel_scale_ratio(effective=effective, source=source, requested=requested)
    d.set_contract(
        dsd.contract_diagnostic("lanczos2", 1.0, pixel_scale_ratio=effective)
    )
    return d


def test_corrected_geometry_reports_frozen_value_and_source(tmp_path):
    eff = 1.0 / 3.0
    d = _diag(tmp_path, eff, SRC, requested=None)
    out = d.to_dict()
    assert out["pixel_scale_ratio_requested"] is None
    assert out["pixel_scale_ratio_current"] == pytest.approx(eff)
    assert out["pixel_scale_ratio_source"] == SRC
    assert out["geometry"]["pixel_scale_ratio_current"] == pytest.approx(eff)
    assert out["geometry"]["pixel_scale_ratio_current_source"] == SRC
    # agreement with the real add_image contract
    assert out["add_image_contract"]["pixel_scale_ratio_effective"] == pytest.approx(eff)
    assert out["add_image_contract"]["pixel_scale_ratio_source"] == "explicit"
    # legacy candidate remains a candidate, never the effective value
    assert out["geometry"]["pixel_scale_ratio_candidate"] == pytest.approx(eff)
    assert out["geometry"]["pixel_scale_ratio_candidate"] is not out["pixel_scale_ratio_current"]


def test_legacy_geometry_stays_truthful(tmp_path):
    d = _diag(tmp_path, None, None)
    out = d.to_dict()
    assert out["pixel_scale_ratio_current"] == 1.0
    assert out["pixel_scale_ratio_source"] == "upstream_add_image_default"
    assert out["geometry"]["pixel_scale_ratio_current"] == 1.0
    assert out["geometry"]["pixel_scale_ratio_current_source"] == "upstream_add_image_default"


def test_geometry_diagnostic_does_not_fabricate_when_unavailable():
    rec = dsd.geometry_diagnostic(None, None, kernel="lanczos2", scale=3.0)
    assert rec["available"] is False
    assert rec["pixel_scale_ratio_current"] == 1.0
    assert rec["pixel_scale_ratio_current_source"] == "upstream_add_image_default"
    assert rec["pixel_scale_ratio_candidate"] is None


def test_persisted_artifact_carries_the_frozen_fact(tmp_path):
    eff = 1.0 / 3.0
    d = _diag(tmp_path, eff, SRC)
    d.write()
    path = tmp_path / "drizzle_science_diagnostics.json"
    assert path.is_file()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["pixel_scale_ratio_current"] == pytest.approx(eff)
    assert on_disk["pixel_scale_ratio_source"] == SRC
    assert on_disk["add_image_contract"]["pixel_scale_ratio_effective"] == pytest.approx(eff)
