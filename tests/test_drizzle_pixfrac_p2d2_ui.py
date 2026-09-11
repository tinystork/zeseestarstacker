"""P2-D2 behavioral tests for the two Tk kernel-aware pixfrac surfaces.

The real state handlers are invoked against faithful variable/widget doubles.  In
particular, the Mosaic drizzle frame contains the same pixfrac spinbox object so
the recursive parent-gating pass is represented and ordering regressions fail.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

tk_std = importlib.import_module("seestar.gui.main_window")
tk_mos = importlib.import_module("seestar.gui.mosaic_gui")

NORMAL, DISABLED = "normal", "disabled"
EDITABLE = ("square", "turbo", "gaussian")
FIXED = ("lanczos2", "lanczos3")
ALL_KERNELS = EDITABLE + FIXED + ("point",)


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Widget:
    def __init__(self, widget_class="TFrame", children=None):
        self.widget_class = widget_class
        self.children = list(children or [])
        self.state = None
        self.text = ""
        self.bindings = {}

    def winfo_exists(self):
        return True

    def winfo_children(self):
        return self.children

    def winfo_class(self):
        return self.widget_class

    def winfo_name(self):
        return "fake"

    def winfo_ismapped(self):
        return False

    def config(self, **kwargs):
        if "state" in kwargs:
            self.state = kwargs["state"]
        if "text" in kwargs:
            self.text = kwargs["text"]

    configure = config

    def bind(self, sequence, callback):
        self.bindings[sequence] = callback

    def pack(self, *args, **kwargs):
        return None

    def pack_forget(self):
        return None


class _DynamicFake:
    """Supply inert widgets for unrelated controls touched by a real handler."""

    def __getattr__(self, name):
        widget = _Widget()
        setattr(self, name, widget)
        return widget


class _Parent:
    @staticmethod
    def tr(_key, default=""):
        return default


def _standard(kernel, *, enabled=True, value=0.8):
    fake = _DynamicFake()
    fake.tr = lambda _key, default="": default
    fake.use_drizzle_var = _Var(enabled)
    fake.drizzle_mode_var = _Var("Final")
    fake.drizzle_kernel_var = _Var(kernel)
    fake.drizzle_pixfrac_var = _Var(value)
    fake.drizzle_pixfrac_spinbox = _Widget("TSpinbox")
    fake.drizzle_pixfrac_label = _Widget("TLabel")
    return fake


def _mosaic(kernel, *, enabled=True, value=0.8):
    fake = _DynamicFake()
    fake.parent_gui = _Parent()
    fake.local_mosaic_active_var = _Var(enabled)
    fake.local_mosaic_align_mode_var = _Var("local_fast_fallback")
    fake.local_drizzle_kernel_var = _Var(kernel)
    fake.local_drizzle_pixfrac_var = _Var(value)
    fake.pixfrac_spinbox = _Widget("TSpinbox")
    fake.pixfrac_label = _Widget("TLabel")
    fake.alignment_mode_frame = _Widget()
    fake.astrometry_config_frame = _Widget()
    fake.drizzle_options_frame = _Widget(children=[fake.pixfrac_spinbox])
    fake.fastaligner_options_frame = _Widget()
    return fake


@pytest.mark.parametrize("kernel", EDITABLE)
def test_tk_standard_editable_kernels_preserve_value(kernel):
    fake = _standard(kernel)
    tk_std.SeestarStackerGUI._update_drizzle_options_state(fake)
    assert fake.drizzle_pixfrac_spinbox.state == NORMAL
    assert fake.drizzle_pixfrac_var.get() == pytest.approx(0.8)
    assert fake.drizzle_pixfrac_label.text == "Pixfrac:"


@pytest.mark.parametrize("kernel", FIXED)
def test_tk_standard_lanczos_is_fixed_one(kernel):
    fake = _standard(kernel)
    tk_std.SeestarStackerGUI._update_drizzle_options_state(fake)
    assert fake.drizzle_pixfrac_spinbox.state == DISABLED
    assert fake.drizzle_pixfrac_var.get() == pytest.approx(1.0)
    assert "1.0" in fake.drizzle_pixfrac_label.text


def test_tk_standard_point_is_na_and_round_trip_is_editable():
    fake = _standard("point")
    tk_std.SeestarStackerGUI._update_drizzle_options_state(fake)
    assert fake.drizzle_pixfrac_spinbox.state == DISABLED
    assert fake.drizzle_pixfrac_var.get() == pytest.approx(1.0)
    assert "N/A" in fake.drizzle_pixfrac_label.text

    fake.drizzle_kernel_var.set("gaussian")
    tk_std.SeestarStackerGUI._update_drizzle_options_state(fake)
    assert fake.drizzle_pixfrac_spinbox.state == NORMAL
    assert fake.drizzle_pixfrac_var.get() <= 1.0
    assert fake.drizzle_pixfrac_label.text == "Pixfrac:"


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_tk_standard_parent_gate_is_authoritative(kernel):
    fake = _standard(kernel, enabled=False)
    tk_std.SeestarStackerGUI._update_drizzle_options_state(fake)
    assert fake.drizzle_pixfrac_spinbox.state == DISABLED


@pytest.mark.parametrize("kernel", EDITABLE)
def test_tk_mosaic_editable_kernels_preserve_value(kernel):
    fake = _mosaic(kernel)
    tk_mos.MosaicSettingsWindow._update_options_state(fake)
    assert fake.pixfrac_spinbox.state == NORMAL
    assert fake.local_drizzle_pixfrac_var.get() == pytest.approx(0.8)
    assert fake.pixfrac_label.text == "Pixfrac:"


@pytest.mark.parametrize("kernel", FIXED)
def test_tk_mosaic_lanczos_stays_disabled_after_parent_frame_pass(kernel):
    fake = _mosaic(kernel)
    tk_mos.MosaicSettingsWindow._update_options_state(fake)
    assert fake.pixfrac_spinbox.state == DISABLED
    assert fake.local_drizzle_pixfrac_var.get() == pytest.approx(1.0)
    assert "1.0" in fake.pixfrac_label.text


def test_tk_mosaic_point_is_na_and_round_trip_is_editable():
    fake = _mosaic("point")
    tk_mos.MosaicSettingsWindow._update_options_state(fake)
    assert fake.pixfrac_spinbox.state == DISABLED
    assert fake.local_drizzle_pixfrac_var.get() == pytest.approx(1.0)
    assert "N/A" in fake.pixfrac_label.text

    fake.local_drizzle_kernel_var.set("turbo")
    tk_mos.MosaicSettingsWindow._update_options_state(fake)
    assert fake.pixfrac_spinbox.state == NORMAL
    assert fake.local_drizzle_pixfrac_var.get() <= 1.0
    assert fake.pixfrac_label.text == "Pixfrac:"


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_tk_mosaic_parent_gate_is_authoritative(kernel):
    fake = _mosaic(kernel, enabled=False)
    tk_mos.MosaicSettingsWindow._update_options_state(fake)
    assert fake.pixfrac_spinbox.state == DISABLED
