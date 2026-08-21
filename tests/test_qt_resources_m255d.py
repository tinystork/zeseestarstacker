"""Visual-resources tests for the Qt shell (M25.5-D).

Covers the three "real product" chrome items added in M25.5-D:

* window icon from the packaged ``seestar/icon/icon.png`` (best-effort),
* the empty-preview placeholder rendered from the packaged
  ``seestar/icon/back.png`` (scaled, centred, aspect-preserving, Tk parity),
* the real window title ``"<name>  –  <version>"`` read from the package's own
  ``__version__`` / ``__codename__`` (lazy, engine/Tk/astropy-free).

All resource loading is best-effort: a missing/undecodable resource must never
raise and must fall back to the pre-M25.5-D behaviour (no icon / cleared empty
preview).  No FITS/PNG writes, no science calls and no engine subprocess.
"""

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

import seestar.gui_qt.main_window as main_window
from seestar.gui_qt import (
    PRODUCT_TITLE,
    MainWindow,
    create_application,
    default_window_title,
    load_window_icon,
    product_version,
)


@pytest.fixture(scope="session", autouse=True)
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


# --------------------------------------------------------------------------
# Window icon
# --------------------------------------------------------------------------
def test_window_icon_is_set_on_main_window(qapp):
    win = MainWindow()
    try:
        icon = win.windowIcon()
        assert not icon.isNull()
        # The icon carries a real (non-empty) pixmap.
        assert not icon.pixmap(64, 64).isNull()
    finally:
        win.shutdown()


def test_window_icon_loader_returns_packaged_icon(qapp):
    icon = load_window_icon()
    assert icon is not None
    assert not icon.isNull()
    assert not icon.pixmap(32, 32).isNull()


def test_window_icon_missing_resource_no_raise(qapp, monkeypatch):
    monkeypatch.setattr(main_window, "load_window_icon", lambda: None)
    win = MainWindow()
    try:
        # No icon is applied; the window still opens with the default icon.
        assert win.windowIcon().isNull()
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# back.png empty-preview placeholder
# --------------------------------------------------------------------------
def test_empty_preview_renders_back_pixmap(qapp):
    win = MainWindow()
    try:
        pm = win.preview_image_label.pixmap()
        assert pm is not None and not pm.isNull()
        # back.png is square (750x750); aspect-preserving scaling keeps 1:1.
        assert pm.width() == pm.height()
    finally:
        win.shutdown()


def test_empty_preview_scales_to_fit_label(qapp):
    win = MainWindow()
    try:
        win.preview_image_label.resize(400, 300)
        win._show_empty_preview()
        pm = win.preview_image_label.pixmap()
        assert pm is not None and not pm.isNull()
        # Aspect ratio preserved (back.png is 750x750 -> square), and the
        # placeholder was actually scaled down from the native 750px.
        assert pm.width() == pm.height()
        assert pm.width() < 750
    finally:
        win.shutdown()


def test_empty_preview_missing_resource_keeps_old_behavior(qapp, monkeypatch):
    monkeypatch.setattr(main_window, "load_empty_preview_pixmap", lambda: None)
    win = MainWindow()
    try:
        # Missing resource -> the pre-M25.5-D cleared (null) pixmap.
        pm = win.preview_image_label.pixmap()
        assert pm is None or pm.isNull()
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Real product title (Tk byte-identical)
# --------------------------------------------------------------------------
def test_product_version_from_source():
    assert product_version() == "7.0.2 Boring ostentus"


def test_default_title_matches_tk_exactly():
    # Tk: f"{self.tr('title')}  –  {self.app_version}" where tr('title') is
    # "Seestar Stacker" and app_version is "7.0.2 Boring ostentus"; separator
    # is two spaces, EN DASH (U+2013), two spaces.
    expected = "Seestar Stacker  \u2013  7.0.2 Boring ostentus"
    assert default_window_title() == expected
    assert PRODUCT_TITLE == "Seestar Stacker"


def test_bare_main_window_title_includes_version(qapp):
    win = MainWindow()
    try:
        assert win.windowTitle() == default_window_title()
        assert win.windowTitle() == "Seestar Stacker  \u2013  7.0.2 Boring ostentus"
        assert "7.0.2" in win.windowTitle()
    finally:
        win.shutdown()


def test_explicit_title_still_overrides(qapp):
    win = MainWindow(title="Custom Qt Shell")
    try:
        assert win.windowTitle() == "Custom Qt Shell"
    finally:
        win.shutdown()


def test_product_version_fallback_no_version_no_raise(monkeypatch):
    import seestar

    monkeypatch.delattr(seestar, "__version__", raising=False)
    monkeypatch.delattr(seestar, "__codename__", raising=False)
    # Version source missing -> empty version, no raise, no version suffix.
    assert product_version() == ""
    assert default_window_title() == "Seestar Stacker"


# --------------------------------------------------------------------------
# Import hygiene: version reading is lazy (no engine/Tk/astropy)
# --------------------------------------------------------------------------
def test_version_reading_is_lazy_fresh_process():
    """A fresh interpreter must not read ``seestar.__version__`` at import time.

    We seed a sentinel version *before* importing ``seestar.gui_qt``: if the
    module (or any gui_qt submodule) read the version eagerly, the sentinel
    would leak into the module-level ``PRODUCT_TITLE`` / ``DEFAULT_TITLE``
    constants.  The lazy ``product_version()`` reads it only when called.
    """
    root = Path(__file__).resolve().parents[1]
    code = (
        "import seestar\n"
        "seestar.__version__ = 'SENTINEL-9.9.9'\n"
        "seestar.__codename__ = 'CODEX'\n"
        "import seestar.gui_qt.main_window as mw\n"
        "assert 'SENTINEL-9.9.9' not in mw.PRODUCT_TITLE\n"
        "assert 'SENTINEL-9.9.9' not in mw.DEFAULT_TITLE\n"
        "assert mw.product_version() == 'SENTINEL-9.9.9 CODEX'\n"
        "print('LAZY_OK')\n"
    )
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
    )
    assert proc.returncode == 0, (
        f"version reading is not lazy: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
