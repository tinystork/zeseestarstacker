"""P2-D1 rework-3: settings migration + neutrality evidence."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

dx = importlib.import_module("seestar.core.drizzle_core")
ss = importlib.import_module("seestar.gui_qt.settings_state")
gsettings = importlib.import_module("seestar.gui.settings")
smigration = importlib.import_module("seestar.settings_migration")


# --- settings migration (behavioral) ---------------------------------------
def test_qt_settings_legacy_two_migrates_with_carrier():
    st = ss.QtSettingsState.from_dict({"drizzle_pixfrac": 2.0})
    assert st.drizzle_pixfrac == 1.0
    assert st.drizzle_pixfrac_requested_raw == 2.0
    assert st.drizzle_pixfrac_reason == "pixfrac_gt_one_coerced_to_one"
    rt = ss.QtSettingsState.from_dict(st.to_dict())
    assert rt.drizzle_pixfrac == 1.0 and rt.drizzle_pixfrac_requested_raw == 2.0


def test_qt_settings_valid_value_unchanged():
    st = ss.QtSettingsState.from_dict({"drizzle_pixfrac": 0.8})
    assert st.drizzle_pixfrac == 0.8


@pytest.mark.parametrize(
    "value,eff,raw,reason",
    [
        (2.0, 1.0, 2.0, "pixfrac_gt_one_coerced_to_one"),
        (0.005, 0.01, 0.005, "pixfrac_below_minimum_clamped"),
        (0.8, 0.8, 0.8, None),
        (1.0, 1.0, 1.0, None),
    ],
)
def test_nested_mosaic_pixfrac_migration(value, eff, raw, reason):
    out, r, rsn = gsettings.migrate_mosaic_pixfrac({"pixfrac": value})
    assert out["pixfrac"] == pytest.approx(eff)
    assert r == pytest.approx(raw)
    assert rsn == reason
    if reason:
        assert out["pixfrac_reason"] == reason
        assert out["pixfrac_requested_raw"] == pytest.approx(raw)


def test_nested_mosaic_invalid_and_nonfinite():
    for bad in ("abc", float("nan"), float("inf"), 0.0, -1.0):
        out, r, rsn = gsettings.migrate_mosaic_pixfrac({"pixfrac": bad})
        assert out["pixfrac"] == 0.8
        assert rsn == "pixfrac_invalid_defaulted"


@pytest.mark.parametrize(
    "value,expected,raw,reason",
    [
        (2.0, 1.0, 2.0, "pixfrac_gt_one_coerced_to_one"),
        (0.005, 0.01, 0.005, "pixfrac_below_minimum_clamped"),
        (0.8, 0.8, None, None),
    ],
)
def test_shared_loader_migrates_current_schema_pixfrac(
    value, expected, raw, reason
):
    data, changed = smigration.migrate_settings_data(
        {
            smigration.SETTINGS_SCHEMA_VERSION_KEY:
                smigration.CURRENT_SETTINGS_SCHEMA_VERSION,
            "drizzle_pixfrac": value,
        }
    )
    assert data["drizzle_pixfrac"] == pytest.approx(expected)
    assert changed is (reason is not None)
    assert data.get("drizzle_pixfrac_requested_raw") == raw
    assert data.get("drizzle_pixfrac_reason") == reason


@pytest.mark.parametrize(
    "value,expected,raw,reason",
    [
        (2.0, 1.0, 2.0, "pixfrac_gt_one_coerced_to_one"),
        (0.005, 0.01, 0.005, "pixfrac_below_minimum_clamped"),
        (0.8, 0.8, None, None),
    ],
)
def test_tk_settings_file_round_trip_preserves_pixfrac_migration(
    tmp_path, value, expected, raw, reason
):
    path = tmp_path / "settings.json"
    seed = gsettings.SettingsManager(str(path))
    seed.save_settings()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["drizzle_pixfrac"] = value
    payload.pop("drizzle_pixfrac_requested_raw", None)
    payload.pop("drizzle_pixfrac_reason", None)
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = gsettings.SettingsManager(str(path))
    assert loaded.load_settings() is True
    assert loaded.drizzle_pixfrac == pytest.approx(expected)
    assert getattr(loaded, "drizzle_pixfrac_requested_raw", None) == raw
    observed_reason = getattr(loaded, "drizzle_pixfrac_reason", "") or None
    assert observed_reason == reason

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["drizzle_pixfrac"] == pytest.approx(expected)
    assert persisted.get("drizzle_pixfrac_requested_raw") == raw
    assert persisted.get("drizzle_pixfrac_reason") == reason


# --- neutrality: explicit 1 vs requested2->effective1 ----------------------
def _deposit(pixfrac, kernel="square"):
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    yy, xx = np.indices((8, 8), dtype=np.float64)
    pix = np.dstack((3.0 * (xx + 0.5), 3.0 * (yy + 0.5)))
    acc = dx.DrizzleAccumulator((24, 24), kernel=kernel, pixfrac=pixfrac)
    sup = dx.DrizzleAccumulator((24, 24), kernel="square", pixfrac=1.0)
    for _ in range(2):
        mask = np.ones((8, 8), np.float32)
        acc.add(data, mask, pix, exptime=1.0, in_units="counts")
        sup.add(mask, mask * mask, pix, exptime=1.0, in_units="cps")
    return acc, sup


def test_identical_effective_parameters_are_bit_identical():
    a, a_sup = _deposit(1.0)
    b, b_sup = _deposit(1.0)
    assert np.array_equal(a._out_img, b._out_img)
    assert np.array_equal(a._out_wht, b._out_wht)
    assert np.array_equal(a_sup._out_wht, b_sup._out_wht)
    # the resolved effective for a legacy requested==2 is exactly 1.0
    eff, raw, rsn = dx.resolve_drizzle_pixfrac("square", 2.0)
    assert (eff, raw) == (1.0, 2.0)
    assert rsn == "pixfrac_gt_one_coerced_to_one"
