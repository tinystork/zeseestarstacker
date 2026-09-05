"""PySide6 main-window shell for ZeSeestarStacker (Tk-like topology).

This module builds the Qt window used by the official Qt entry point
``seestar.qt_main:main`` (console script ``zeseestarstacker``).  A bare
``MainWindow()`` still defaults to the simulated backend; the console entry
point opts into the real ``seestar`` backend by default.  This module
deliberately does NOT:

* launch any real stacking,
* import the Tk GUI,
* import the scientific engine.

Threading rule (enforced by design, not just convention): the shell starts a
run on a dedicated ``QThread`` via a
:class:`~seestar.gui_qt.run_controller.RunController`, but nothing mutates Qt
widgets off the GUI thread.  The worker communicates with this window
exclusively through Qt queued signals — never by touching widgets directly, and
never by receiving or storing widget references.

Topology mirrors the historical Tk window (``0d9af8b``): a horizontal
``QSplitter`` with a scrollable left control panel (language placeholder +
``QTabWidget`` with ``Stacking`` / ``Expert`` / ``Preview controls`` tabs +
progress/status/log area) and a persistent right preview/action panel
(preview image + metadata, zoom/resolution/rotation controls, a persistent
display histogram and the action buttons Start / Stop / Analyse / Solver /
View Inputs / Add Folder / Open Output).  Start/Stop are functional, as are the
display-only preview zoom / rotation / resolution controls (M5) and the
display-only preview WB / stretch / histogram controls (M10/M14, with the
single interactive right-panel histogram).  The Analyse
button launches the standalone ZeAnalyser
product on the current input folder (M7) via a stdlib-only launch seam; it
never touches the stacking backend.

The Stacking tab exposes input/output/temp/filename, batch size, stacking mode,
the final-combination business selector, drizzle and local-solver controls; the
Expert tab (M10) exposes the rest of the backend-relevant
:class:`~seestar.gui_qt.settings_state.QtSettingsState` surface in a scrollable,
grouped form.  Both feed the same model, which
:meth:`MainWindow.build_run_request` turns into a validated, immutable
:class:`~seestar.gui_qt.run_bridge.RunRequest`, which the Start button hands to
the lifecycle controller.  A bare ``MainWindow()`` defaults to the simulated
backend (labelled as such next to Start); the console entry point
``seestar.qt_main`` opts into the real ``SeestarQueuedStacker`` by default, and
real activation stays explicit via ``backend_mode``/``backend_factory``.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QByteArray, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import (
    analyzer_launch,
    boring_route,
    gpu_bridge,
    initial_preview,
    localization,
    settings_persistence,
)
from .backend_runner import (
    BackendPreviewPayload,
    BaseRunBackend,
    SeestarQueuedStackerBackend,
)
from .boring_runner import BoringRunnerBase, QProcessBoringRunner
from .final_combine import (
    FINAL_COMBINE_LABELS,
    FINAL_COMBINE_LABEL_TO_KEY,
    FINAL_COMBINE_KEYS,
    final_combine_flags,
)
from .preview_render import render_preview_image
from .preview_adjust import (
    BP_WP_MIN_SEPARATION,
    BLACK_POINT_MAX,
    BLACK_POINT_MIN,
    BLACK_POINT_STEP,
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    BRIGHTNESS_STEP,
    CONTRAST_MAX,
    CONTRAST_MIN,
    CONTRAST_STEP,
    DEFAULT_BLACK_POINT,
    DEFAULT_BRIGHTNESS,
    DEFAULT_CONTRAST,
    DEFAULT_GAMMA,
    DEFAULT_SATURATION,
    DEFAULT_STRETCH,
    DEFAULT_WB,
    DEFAULT_WHITE_POINT,
    GAMMA_MAX,
    GAMMA_MIN,
    GAMMA_STEP,
    SATURATION_MAX,
    SATURATION_MIN,
    SATURATION_STEP,
    STRETCH_MODES,
    WB_MAX,
    WB_MIN,
    WB_STEP,
    WHITE_POINT_MAX,
    WHITE_POINT_MIN,
    WHITE_POINT_STEP,
    apply_preview_adjustments,
    apply_preview_wb,
    clamp_bp_wp_edit,
    compute_auto_wb,
    compute_auto_stretch,
    compute_histogram,
    compute_histogram_percentile,
    compute_histogram_stats,
    normalize_bp_wp,
    render_analysis_display,
)
from .preview_analysis import (
    adapt_anchors_for_drift,
    analysis_upper_bound,
    apply_wb_float,
    compute_anchors,
    compute_auto_stretch_float,
    compute_auto_wb_float,
    extract_raw_linear,
    map_raw_linear,
)
from .histogram_view import (
    HistogramView,
    format_histogram_overflow,
    format_histogram_stats,
)
from .histogram_window import DetachedHistogramWindow
from .histogram_worker import HistogramCoordinator
from .preview_image_view import PreviewImageView
from .preview_view import (
    ZOOM_FACTORS,
    ZOOM_LABELS,
    ZOOM_STEP,
    clamp_zoom_factor,
    fit_scale,
    preset_label_for_factor,
    render_view,
    zoomed_image_size,
)
from .progress_time import UNKNOWN, estimate_remaining_seconds, format_duration
from .run_bridge import (
    RUN_INTENT_FRESH,
    RUN_INTENT_RESUME,
    RunRequest,
    build_run_request as _build_run_request,
)
from .run_controller import RunController
from .run_handoff import attach_run_settings
from .settings_validation import normalize_batch_size, validate_settings_for_backend
from .settings_state import QtSettingsState
from .solver_config_bridge import write_solver_config
from .solver_dialog import SolverSettingsDialog
from .solver_probe import probe_zesolver_operational
from .summary_payload import SummaryPayload, derive_terminal_status

from .resources import load_empty_preview_pixmap, load_window_icon

from seestar import resume_locator
from seestar.utils.phi_trace import phi_trace_enabled, phi_trace_stage

logger = logging.getLogger(__name__)

# Real product window-title *name* (the Tk ``localization`` "title" key, en/fr
# identical).  The full default window title appends the lazily-read package
# version via :func:`default_window_title`, so importing ``seestar.gui_qt``
# never reads the version eagerly and stays engine/Tk/astropy-free.
PRODUCT_TITLE = "Seestar Stacker"

# Backward-compatible base title (no version).  ``MainWindow`` composes the
# real title (name + version) at construction time via ``default_window_title``
# so the version stays a lazy, hygiene-safe read.
DEFAULT_TITLE = PRODUCT_TITLE


def product_version() -> str:
    """Return the product version string (``"8.0.0 Phoenix consedit"``), or ``""``.

    Lazy and hygiene-safe: it imports the already-imported ``seestar`` parent
    package (whose ``__init__`` only binds ``__version__`` / ``__codename__``
    and lazy re-exports — no engine, Tk or astropy) and reads the two
    attributes.  Works identically from a source checkout and an installed
    wheel.  Never raises: any failure degrades to ``""``.
    """
    try:
        import seestar
    except Exception:
        return ""
    version = getattr(seestar, "__version__", "") or ""
    codename = getattr(seestar, "__codename__", "") or ""
    if version and codename:
        return f"{version} {codename}"
    return version


def default_window_title() -> str:
    """Return the real product window title (byte-identical to the Tk title).

    Tk sets ``f"{self.tr('title')}  –  {self.app_version}"`` where
    ``tr('title')`` is ``"Seestar Stacker"`` and ``app_version`` is
    ``"8.0.0 Phoenix consedit"`` (``__version__ + " " + __codename__``).  The
    separator is two spaces, an EN DASH (U+2013), two spaces.
    """
    version = product_version()
    if version:
        return f"{PRODUCT_TITLE}  \u2013  {version}"
    return PRODUCT_TITLE

# Backend selection modes understood by the shell's Start button.
BACKEND_MODES = ("simulated", "seestar")
DEFAULT_BACKEND_MODE = "simulated"

# ZeAnalyser reference-return watcher poll interval (ms).  Matches the Tk
# ``after(1000, ...)`` surveillance cadence exactly: one command-file check per
# tick on the GUI thread, never a busy loop and never a worker thread.
ANALYZER_WATCH_INTERVAL_MS = 1000

# Left-panel tab labels (Tk ``control_notebook`` parity: Stacking / Expert /
# Preview controls, plus the Qt-only System tab — M25.5-C).
TAB_STACKING = "Stacking"
TAB_EXPERT = "Expert"
TAB_SYSTEM = "System"
TAB_PREVIEW_CONTROLS = "Preview controls"

# Presentation theme modes (M25.5-C).  ``system`` follows the platform/style
# palette (no custom palette is imposed), ``dark`` / ``light`` apply a small
# Qt ``QPalette``.  Purely presentation — never read by the engine or Tk.
THEME_MODES = ("system", "dark", "light")
DEFAULT_THEME = "system"
THEME_CHOICE_KEYS = {
    "system": "theme_system",
    "dark": "theme_dark",
    "light": "theme_light",
}


def _complete_disabled_group(p: QPalette, disabled_text: QColor) -> None:
    """Complete a theme palette's Disabled color group (FRP-L1).

    A palette whose Disabled group is incomplete renders disabled widgets with
    the Active group's full-contrast colors (or with style/platform defaults on
    native themes) — a disabled QLabel / group-box title / tab label / input is
    then indistinguishable from an enabled one, and unset roles can leak dark
    colors into the light theme.  This helper explicitly completes the Disabled
    group for the roles the theme defines:

    * **Text roles** (``WindowText``, ``Text``, ``ButtonText``,
      ``PlaceholderText``) use the theme's dimmed ``disabled_text`` gray — the
      shared dimming fix (readable, visibly disabled);
    * **Structural roles** (``Window``, ``Base``, ``AlternateBase``,
      ``Button``, ``Highlight``, ``HighlightedText``, ``ToolTipBase``,
      ``ToolTipText``) mirror their Active-group values so a disabled surface
      never falls back to a color that clashes with the selected theme.

    Only the Disabled group is touched — every Active (and Inactive) color
    stays byte-identical, so the normal appearance is unchanged.
    """
    for role in (
        QPalette.ColorRole.Window,
        QPalette.ColorRole.Base,
        QPalette.ColorRole.AlternateBase,
        QPalette.ColorRole.Button,
        QPalette.ColorRole.ToolTipBase,
        QPalette.ColorRole.ToolTipText,
        QPalette.ColorRole.Highlight,
        QPalette.ColorRole.HighlightedText,
    ):
        p.setColor(
            QPalette.ColorGroup.Disabled,
            role,
            p.color(QPalette.ColorGroup.Active, role),
        )
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.PlaceholderText,
    ):
        p.setColor(QPalette.ColorGroup.Disabled, role, disabled_text)


def _dark_palette() -> QPalette:
    """Return a compact dark ``QPalette`` (no stylesheet, presentation-only)."""
    p = QPalette()
    window = QColor(53, 53, 53)
    base = QColor(35, 35, 35)
    text = QColor(255, 255, 255)
    disabled_text = QColor(127, 127, 127)
    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, window)
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.PlaceholderText, disabled_text)
    p.setColor(QPalette.ColorRole.Button, window)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    p.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    _complete_disabled_group(p, disabled_text)
    return p


def _light_palette() -> QPalette:
    """Return a compact light ``QPalette`` (no stylesheet, presentation-only)."""
    p = QPalette()
    window = QColor(240, 240, 240)
    base = QColor(255, 255, 255)
    text = QColor(0, 0, 0)
    disabled_text = QColor(127, 127, 127)
    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(233, 231, 227))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.PlaceholderText, disabled_text)
    p.setColor(QPalette.ColorRole.Button, window)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    p.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    _complete_disabled_group(p, disabled_text)
    return p

# Minimal visible subset of stacking modes, drizzle modes and local-solver
# preferences.  Values are the *backend* keys, not display labels.
STACKING_MODES = [
    "kappa-sigma",
    "classic",
    "mean",
    "median",
    "winsorized-sigma-clip",
    "linear-fit-clip",
]
# Drizzle processing mode backend values (persistence + engine keys, unchanged).
DRIZZLE_MODES = ["Final", "Incremental"]
# User-facing drizzle mode combo labels, one per ``DRIZZLE_MODES`` entry (order
# matched).  The combo displays these localized labels while ``itemData`` keeps
# the backend value (``Final`` / ``Incremental``) for state/persistence/engine,
# so "Final"/"Incremental" are never shown to the user (R3b contract).  The Tk
# oracle labels the second entry "Large dataset / incremental"; the contract
# target keeps the shorter "Large dataset" spelling used across the Qt shell.
DRIZZLE_MODE_LABEL_KEYS = ["drizzle_mode_standard", "drizzle_mode_large_dataset"]
SOLVER_PREFERENCES = ["none", "astap", "zesolver"]

# Preview resolution-cycle factors (Tk ``preview_res_button`` parity, M17).
# The Qt shell cycles the same 1/1..1/4 label plus a local display downsample;
# since M22 it ALSO drives the engine ``preview_downsample_factor`` +
# ``refresh_preview`` during an active run via a thread-safe control channel
# (idle clicks stay display-only).  The Qt default is 1 (native) — the shell's
# display-only preview is never engine-downsampled, so "1/1" is its native
# state; the Tk initial factor is 2 (engine ``preview_downsample_factor``
# default), a documented deviation.
PREVIEW_RES_FACTORS = (1, 2, 3, 4)
DEFAULT_PREVIEW_RES_FACTOR = 1

# Language combo label -> ZeAnalyser ``--lang`` code (Tk settings default is
# ``"en"``).  Sourced from the Qt-local ``localization`` module (M9).
LANGUAGE_CODE_BY_TEXT = localization.LANGUAGE_CODE_BY_TEXT

# File-dialog filter for reference / last-stack images (Tk parity).
FITS_FILE_FILTER = "FITS files (*.fit *.fits)"

# Choice vocabularies shared with the Tk GUI / backend (backend keys, not
# display labels).  These are the "known choices" for combo-box fields in the
# Settings surface; everything else is a plain scalar/string widget.
NORM_METHODS = ["none", "linear_fit", "sky_mean"]
WEIGHT_METHODS = ["none", "noise_variance", "noise_fwhm", "snr", "stars"]
DRIZZLE_KERNELS = [
    "square",
    "gaussian",
    "point",
    "tophat",
    "turbo",
    "lanczos2",
    "lanczos3",
]
BAYER_PATTERNS = ["GRBG", "RGGB", "GBRG", "BGGR"]
SCNR_TARGET_CHANNELS = ["green", "red", "blue"]
MOSAIC_ALIGNMENT_MODES = ["local_fast_fallback", "local_fast_only", "astrometry_per_panel"]
# BN grid-size choices (Tk ``bn_grid_size_combo`` values, ``24x24`` default).
BN_GRID_SIZES = ["8x8", "16x16", "24x24", "32x32", "64x64"]

# Tri-state for ``match_background_for_final``.  ``default`` (and the older
# ``none`` spelling) both mean "unset" -> ``None`` at the backend, which is the
# exact existing semantic in ``run_config.build_backend_kwargs``.
MATCH_BG_CHOICES = ("default", "true", "false")
MATCH_BG_FROM_TEXT = {"default": None, "true": True, "false": False}
MATCH_BG_TO_TEXT = {None: "default", True: "true", False: "false"}

# Honest backend-mode notice (M9 fix): makes the simulated default explicit so
# a human witness is never misled into thinking a Start click ran real science.
SIMULATED_BACKEND_NOTICE = (
    "Backend: simulated (dev/test) — Start runs no real stacking. "
    "Launch with '--backend seestar' to use the real engine."
)
SEESTAR_BACKEND_NOTICE = "Backend: seestar — real engine."


def _field(attr, label, kind, *params):
    """Build one Settings-surface field spec tuple.

    ``kind`` is one of ``bool``/``int``/``float``/``str``/``combo``/``list``/
    ``match_bg``.  ``params`` are kind-specific:

    * ``int``   -> ``(minimum, maximum, single_step)``
    * ``float`` -> ``(minimum, maximum, single_step, decimals)``
    * ``combo`` -> ``(choices, ...)`` (a single tuple of backend keys)
    * others   -> none
    """
    return (attr, label, kind) + tuple(params)


# Settings surface, grouped into readable sections.  Only backend-relevant
# ``QtSettingsState`` fields NOT already shown on the Stack tab appear here.
# ``Mosaic`` (``None``) is built by :meth:`MainWindow._build_mosaic_section`
# because its ``mosaic_settings`` is a nested dict.
SETTINGS_SECTIONS = [
    (
        "Stacking / Paths",
        [
            _field("kappa", "Kappa", "float", 1.0, 5.0, 0.1, 2),
            _field("stack_kappa_low", "Kappa low", "float", 0.0, 10.0, 0.1, 2),
            _field("stack_kappa_high", "Kappa high", "float", 0.0, 10.0, 0.1, 2),
            _field("stack_winsor_limits", "Winsor limits (low,high)", "str"),
            _field("stack_norm_method", "Normalize method", "combo", NORM_METHODS),
            _field("stack_weight_method", "Weighting method", "combo", WEIGHT_METHODS),
            _field("order_file_list", "Order file list (comma-separated)", "list"),
        ],
    ),
    (
        "Calibration / Hot Pixels",
        [
            _field("correct_hot_pixels", "Correct hot pixels", "bool"),
            _field("hot_pixel_threshold", "Hot pixel threshold", "float", 0.5, 10.0, 0.1, 2),
            _field("neighborhood_size", "Neighborhood size", "int", 1, 20, 1),
            _field("bayer_pattern", "Bayer pattern", "combo", BAYER_PATTERNS),
            _field("cleanup_temp", "Clean up temp files", "bool"),
        ],
    ),
    (
        "Quality Weighting",
        [
            _field("weight_by_snr", "Weight by SNR", "bool"),
            _field("weight_by_stars", "Weight by stars", "bool"),
            _field("snr_exponent", "SNR exponent", "float", 0.0, 10.0, 0.1, 2),
            _field("stars_exponent", "Stars exponent", "float", 0.0, 10.0, 0.1, 2),
            _field("min_weight", "Minimum weight", "float", 0.01, 1.0, 0.01, 3),
        ],
    ),
    (
        "Colour / Post-processing",
        [
            _field("apply_chroma_correction", "Chroma correction", "bool"),
            _field("apply_final_scnr", "Final SCNR", "bool"),
            _field("final_scnr_target_channel", "SCNR target channel", "combo", SCNR_TARGET_CHANNELS),
            _field("final_scnr_amount", "SCNR amount", "float", 0.0, 1.0, 0.05, 2),
            _field("final_scnr_preserve_luminosity", "SCNR preserve luminosity", "bool"),
            _field("apply_bn", "Enable BN", "bool"),
            _field("bn_grid_size_str", "BN grid size", "combo", BN_GRID_SIZES),
            _field("bn_perc_low", "BN percentile low", "int", 0, 40, 1),
            _field("bn_perc_high", "BN percentile high", "int", 10, 95, 1),
            _field("bn_std_factor", "BN std factor", "float", 0.5, 5.0, 0.1, 1),
            _field("bn_min_gain", "BN min gain", "float", 0.1, 2.0, 0.1, 1),
            _field("bn_max_gain", "BN max gain", "float", 1.0, 10.0, 0.1, 1),
            _field("apply_cb", "Enable Edge/Chroma Correction", "bool"),
            _field("cb_border_size", "CB border size", "int", 5, 150, 5),
            _field("cb_blur_radius", "CB blur radius", "int", 0, 50, 1),
            _field("cb_min_b_factor", "CB min B factor", "float", 0.1, 1.0, 0.05, 2),
            _field("cb_max_b_factor", "CB max B factor", "float", 1.0, 3.0, 0.05, 2),
        ],
    ),
    (
        "Cropping",
        [
            _field("apply_master_tile_crop", "Master tile crop", "bool"),
            _field("master_tile_crop_percent", "Master tile crop %", "float", 0.0, 25.0, 0.5, 1),
            _field("apply_final_crop", "Enable Final Cropping", "bool"),
            _field("final_edge_crop_percent", "Final edge crop %", "float", 0.0, 25.0, 0.5, 1),
        ],
    ),
    (
        "Photutils BN",
        [
            _field("apply_photutils_bn", "Photutils background normalization", "bool"),
            _field("photutils_bn_box_size", "Box size", "int", 16, 1024, 16),
            _field("photutils_bn_filter_size", "Filter size", "int", 1, 15, 2),
            _field("photutils_bn_sigma_clip", "Sigma clip", "float", 1.0, 5.0, 0.1, 1),
            _field("photutils_bn_exclude_percentile", "Exclude percentile", "float", 0.0, 100.0, 1.0, 1),
        ],
    ),
    (
        "Coverage / Edge Reconstruction",
        [
            _field("apply_batch_feathering", "Coverage support taper", "bool"),
            _field("apply_coverage_render", "Coverage-aware final reconstruction", "bool"),
            _field("apply_low_wht_mask", "Low-weight mask", "bool"),
            _field("low_wht_percentile", "Low-weight percentile", "int", 1, 100, 1),
            _field("low_wht_soften_px", "Low-weight soften (px)", "int", 32, 512, 16),
        ],
    ),
    ("Mosaic", None),
    (
        "Solver",
        [
            _field("astap_path", "ASTAP path", "str"),
            _field("astap_data_dir", "ASTAP data dir", "str"),
            _field("astap_search_radius", "Search radius", "float", 0.0, 360.0, 0.5, 2),
            _field("astap_downsample", "Downsample", "int", 1, 16, 1),
            _field("astap_sensitivity", "Sensitivity", "int", 1, 10000, 1),
        ],
    ),
    (
        "Output / Reprojection",
        [
            _field("save_final_as_float32", "Save final as float32", "bool"),
            _field("preserve_linear_output", "Preserve linear output", "bool"),
        ],
    ),
    (
        "Final Background Matching",
        [
            _field("match_background_for_final", "Match background for final", "match_bg"),
        ],
    ),
]

# Nested ``mosaic_settings`` sub-fields exposed as real widgets (key, label,
# kind, params).  ``fillval`` stays a string so its default ``"0.0"`` round-
# trips exactly with the Tk defaults.
MOSAIC_FIELDS = [
    ("kernel", "Kernel", "combo", DRIZZLE_KERNELS),
    ("pixfrac", "Pixfrac", "float", (0.01, 2.0, 0.05, 2)),
    ("use_gpu", "Use GPU", "bool", ()),
    ("fillval", "Fill value", "str", ()),
    ("wht_threshold", "WHT threshold", "float", (0.0, 1.0, 0.01, 3)),
    ("alignment_mode", "Alignment mode", "combo", MOSAIC_ALIGNMENT_MODES),
    ("fastalign_orb_features", "ORB features", "int", (1, 100000, 100)),
    ("fastalign_min_abs_matches", "Min abs matches", "int", (1, 10000, 1)),
    ("fastalign_min_ransac", "Min RANSAC", "int", (1, 10000, 1)),
    ("fastalign_ransac_thresh", "RANSAC threshold", "float", (0.0, 100.0, 0.5, 2)),
    ("fastalign_dao_fwhm", "DAO FWHM", "float", (0.1, 100.0, 0.5, 2)),
    ("fastalign_dao_thr_sig", "DAO threshold sigma", "float", (0.0, 100.0, 0.5, 2)),
    ("fastalign_dao_max_stars", "DAO max stars", "int", (1, 100000, 10)),
    ("mosaic_scale_factor", "Scale factor", "int", (1, 100, 1)),
]


# English section title -> Qt-local translation key (M9).  The ``Mosaic``
# section (fields ``None``) is handled separately in ``_build_mosaic_section``.
SECTION_TITLE_KEYS = {
    "Stacking / Paths": "section_stacking_paths",
    "Calibration / Hot Pixels": "section_calibration",
    "Quality Weighting": "section_quality_weighting",
    "Colour / Post-processing": "section_colour_post",
    "Cropping": "section_cropping",
    "Photutils BN": "section_photutils_bn",
    "Coverage / Edge Reconstruction": "section_coverage_reconstruction",
    "Solver": "section_solver",
    "Output / Reprojection": "section_output_reprojection",
    "Final Background Matching": "section_final_bg_matching",
}

# attr -> translation key for Settings field labels (M9).  The Expert-tab
# (BN / CB / cropping / Photutils / coverage / low-weight) labels were added
# in M15 so every Expert-tab control localizes to FR/EN.
LOCALIZED_SETTINGS_FIELD_KEYS = {
    "kappa": "field_kappa",
    "stack_norm_method": "field_normalize_method",
    "stack_weight_method": "field_weighting_method",
    "correct_hot_pixels": "field_correct_hot_pixels",
    "bayer_pattern": "field_bayer_pattern",
    "cleanup_temp": "field_cleanup_temp",
    "weight_by_snr": "field_weight_by_snr",
    "weight_by_stars": "field_weight_by_stars",
    "drizzle_kernel": "field_drizzle_kernel",
    "apply_bn": "field_apply_bn",
    "bn_grid_size_str": "field_bn_grid_size",
    "bn_perc_low": "field_bn_perc_low",
    "bn_perc_high": "field_bn_perc_high",
    "bn_std_factor": "field_bn_std_factor",
    "bn_min_gain": "field_bn_min_gain",
    "bn_max_gain": "field_bn_max_gain",
    "apply_cb": "field_apply_cb",
    "cb_border_size": "field_cb_border_size",
    "cb_blur_radius": "field_cb_blur_radius",
    "cb_min_b_factor": "field_cb_min_b_factor",
    "cb_max_b_factor": "field_cb_max_b_factor",
    "apply_master_tile_crop": "field_master_tile_crop",
    "master_tile_crop_percent": "field_master_tile_crop_percent",
    "apply_final_crop": "field_apply_final_crop",
    "final_edge_crop_percent": "field_final_edge_crop_percent",
    "apply_photutils_bn": "field_apply_photutils_bn",
    "photutils_bn_box_size": "field_photutils_bn_box_size",
    "photutils_bn_filter_size": "field_photutils_bn_filter_size",
    "photutils_bn_sigma_clip": "field_photutils_bn_sigma_clip",
    "photutils_bn_exclude_percentile": "field_photutils_bn_exclude_percentile",
    "apply_batch_feathering": "field_apply_batch_feathering",
    "apply_coverage_render": "field_apply_coverage_render",
    "apply_low_wht_mask": "field_apply_low_wht_mask",
    "low_wht_percentile": "field_low_wht_percentile",
    "low_wht_soften_px": "field_low_wht_soften_px",
    "save_final_as_float32": "field_save_as_float32",
    "preserve_linear_output": "field_preserve_linear_output",
    "match_background_for_final": "field_match_bg",
}

# Mosaic sub-field key -> translation key (M9).  Explicit (rather than a plain
# ``mosaic_`` prefix) because ``MOSAIC_FIELDS`` already prefixes one key.
MOSAIC_FIELD_KEYS = {
    "kernel": "mosaic_kernel",
    "pixfrac": "mosaic_pixfrac",
    "use_gpu": "mosaic_use_gpu",
    "fillval": "mosaic_fillval",
    "wht_threshold": "mosaic_wht_threshold",
    "alignment_mode": "mosaic_alignment_mode",
    "fastalign_orb_features": "mosaic_fastalign_orb_features",
    "fastalign_min_abs_matches": "mosaic_fastalign_min_abs_matches",
    "fastalign_min_ransac": "mosaic_fastalign_min_ransac",
    "fastalign_ransac_thresh": "mosaic_fastalign_ransac_thresh",
    "fastalign_dao_fwhm": "mosaic_fastalign_dao_fwhm",
    "fastalign_dao_thr_sig": "mosaic_fastalign_dao_thr_sig",
    "fastalign_dao_max_stars": "mosaic_fastalign_dao_max_stars",
    "mosaic_scale_factor": "mosaic_scale_factor",
}

# Expert-tab enable flags -> the sub-option widget attrs each flag gates
# (mirrors the Tk ``_update_*_options_state`` / ``_update_master_tile_crop_state``
# enabler logic: unchecked disables the gated widgets, checked re-enables them).
EXPERT_ENABLER_GATES = {
    "apply_bn": [
        "bn_grid_size_str",
        "bn_perc_low",
        "bn_perc_high",
        "bn_std_factor",
        "bn_min_gain",
        "bn_max_gain",
    ],
    "apply_cb": [
        "cb_border_size",
        "cb_blur_radius",
        "cb_min_b_factor",
        "cb_max_b_factor",
    ],
    "apply_final_crop": ["final_edge_crop_percent"],
    "apply_master_tile_crop": ["master_tile_crop_percent"],
    "apply_photutils_bn": [
        "photutils_bn_box_size",
        "photutils_bn_filter_size",
        "photutils_bn_sigma_clip",
        "photutils_bn_exclude_percentile",
    ],
    "apply_low_wht_mask": ["low_wht_percentile", "low_wht_soften_px"],
    # Final SCNR (Tk Stacking-tab ``_update_final_scnr_options_state``): the
    # "Apply Final SCNR" checkbox gates the target-channel / amount /
    # preserve-luminosity sub-options.  In Qt these live on the Expert tab
    # "Colour / Post-processing" section but the gating semantics are identical.
    "apply_final_scnr": [
        "final_scnr_target_channel",
        "final_scnr_amount",
        "final_scnr_preserve_luminosity",
    ],
}

# The set of Expert-tab attributes the "Reset Expert Settings" button restores
# to ``QtSettingsState`` defaults.  This mirrors the Tk ``_reset_expert_settings``
# reset set (BN / CB / master-tile crop / final crop / coverage support /
# Photutils BN) and additionally resets the Low WHT Mask group
# (``apply_low_wht_mask`` / ``low_wht_percentile`` / ``low_wht_soften_px``),
# which the Tk button omits (a Tk oversight) — see the M15 checklist note.
# ``apply_batch_feathering`` and ``apply_coverage_render`` have no gated
# widgets but are still reset targets. Output-format fields (``save_final_as_float32`` /
# ``preserve_linear_output``) are deliberately NOT reset, matching the Tk
# button which leaves them untouched.
EXPERT_RESET_ATTRS = [
    "apply_bn",
    "bn_grid_size_str",
    "bn_perc_low",
    "bn_perc_high",
    "bn_std_factor",
    "bn_min_gain",
    "bn_max_gain",
    "apply_cb",
    "cb_border_size",
    "cb_blur_radius",
    "cb_min_b_factor",
    "cb_max_b_factor",
    "apply_master_tile_crop",
    "master_tile_crop_percent",
    "apply_final_crop",
    "final_edge_crop_percent",
    "apply_batch_feathering",
    "apply_coverage_render",
    "apply_low_wht_mask",
    "low_wht_percentile",
    "low_wht_soften_px",
    "apply_photutils_bn",
    "photutils_bn_box_size",
    "photutils_bn_filter_size",
    "photutils_bn_sigma_clip",
    "photutils_bn_exclude_percentile",
]


def _is_option_a_preview_payload(data) -> bool:
    """Return ``True`` when ``data`` is an Option-A ``(legacy, raw_linear)`` pair.

    Option-A payloads are 2+ element tuples/lists whose *second* element is a
    valid 2D/3D numeric array (the raw-linear source) whose dtype kind is
    ``f``/``i``/``u`` — exactly the arrays that
    :func:`preview_analysis._as_float_array` will ingest.  Legacy producers
    send a lone array (or a tuple whose second element is not such an image
    array — e.g. bool, text, bytes, object or structured data), which keeps the
    existing ``render_preview_image`` QImage path.  Detection is structural
    (``ndim``/``shape``/``dtype.kind`` attributes, no numpy import) so the
    fresh ``import seestar.gui_qt`` hygiene invariant holds.
    """
    if not isinstance(data, (tuple, list)) or len(data) < 2:
        return False
    second = data[1]
    ndim = getattr(second, "ndim", None)
    shape = getattr(second, "shape", None)
    if ndim not in (2, 3) or shape is None:
        return False
    try:
        if not all(int(d) > 0 for d in shape):
            return False
    except (TypeError, ValueError):
        return False
    # The float core (preview_analysis._as_float_array) accepts only float,
    # signed-integer and unsigned-integer arrays.  Anything else (bool, string,
    # bytes, object, structured/void, or a missing/unsupported dtype) must not
    # be classified Option-A and instead follows the legacy QImage path.
    kind = getattr(getattr(second, "dtype", None), "kind", None)
    return kind in ("f", "i", "u")


# ZSSS-OTPUX-STABLE-A: authoritative engine preview-source labels.  The engine
# tags every live preview payload header with ``PREV_SRC``; these are the two
# labels the backend actually emits (see ``queue_manager._update_preview_sum_w``
# and ``queue_manager._update_preview_drizzle_accumulator``).  A payload whose
# header carries no ``PREV_SRC`` is the legacy incremental-reprojection/coadd
# path (``queue_manager._update_preview``).
_DRIZZLE_PREVIEW_SRC = "Drizzle Accumulator"
_SUMW_PREVIEW_SRC = "SUM/W Accumulators"


def _positive_int(value) -> Optional[int]:
    """Return ``value`` as a strictly-positive ``int``, else ``None``.

    Used to read engine-provided counters (``image_count`` / ``current_batch``)
    defensively: a missing, non-numeric or non-positive counter never produces
    a fabricated identity.
    """
    try:
        ival = int(value)
    except (TypeError, ValueError):
        return None
    if ival <= 0:
        return None
    return ival


class MainWindow(QMainWindow):
    """Minimal side-by-side Qt main window used for offscreen smoke tests.

    The window owns presentation state (title, tabs, progress bar, status bar)
    plus a small settings form whose values are mirrored into a
    :class:`QtSettingsState` model.  It never touches the scientific engine at
    import time.  The default Start path begins a *simulated* run through its
    :class:`RunController`; an explicit backend factory or ``backend_mode`` can
    opt into another backend.  All widget updates happen in GUI-thread slots
    connected to the controller's queued signals.
    """

    # Worker-thread -> GUI-thread delivery for the initial-preview auto-load
    # (M12).  Emitted from a daemon ``threading.Thread``; the connection is
    # explicitly queued so no widget is ever touched off the GUI thread.
    _initial_preview_result = Signal(object)

    def __init__(
        self,
        title: Optional[str] = None,
        parent: Optional[QWidget] = None,
        *,
        backend_factory: Optional[Callable[[], BaseRunBackend]] = None,
        backend_mode: str = DEFAULT_BACKEND_MODE,
        solver_probe: Optional[Callable[[], bool]] = None,
        boring_runner_factory: Optional[Callable[[], BoringRunnerBase]] = None,
        clock: Optional[Callable[[], float]] = None,
        analyzer_launcher: Optional[Callable[[str, str, str], bool]] = None,
        analyzer_command_file_maker: Optional[Callable[[], str]] = None,
        shutdown_wait_ms: int = 5000,
        settings_path: Optional[str] = None,
        histogram_compute_fn: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(parent)
        if backend_mode not in BACKEND_MODES:
            raise ValueError(
                f"backend_mode must be one of {BACKEND_MODES!r}, got {backend_mode!r}"
            )
        if backend_factory is not None and not callable(backend_factory):
            raise TypeError("backend_factory must be callable or None")
        if solver_probe is not None and not callable(solver_probe):
            raise TypeError("solver_probe must be callable or None")
        if boring_runner_factory is not None and not callable(boring_runner_factory):
            raise TypeError("boring_runner_factory must be callable or None")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable or None")
        if analyzer_launcher is not None and not callable(analyzer_launcher):
            raise TypeError("analyzer_launcher must be callable or None")
        if analyzer_command_file_maker is not None and not callable(
            analyzer_command_file_maker
        ):
            raise TypeError("analyzer_command_file_maker must be callable or None")
        if not isinstance(shutdown_wait_ms, int) or shutdown_wait_ms < 0:
            raise ValueError("shutdown_wait_ms must be a non-negative int")
        self.backend_factory = backend_factory
        self.backend_mode = backend_mode
        self.boring_runner_factory = boring_runner_factory
        # Injectable monotonic clock (tests inject a controllable fake so the
        # elapsed/remaining surface is testable without any real sleeping).
        self._clock = clock if clock is not None else time.monotonic
        # Start timestamp of the current run (None when idle); drives the
        # elapsed/remaining time labels.
        self._run_started_at: Optional[float] = None
        # Terminal run-summary payload delivered by the backend/boring runner
        # just before the ``finished`` signal (None when none arrived).  It is
        # consumed once by ``_show_pending_summary`` at run end.
        self._last_summary_payload: Optional[SummaryPayload] = None
        # Currently shown (non-modal) summary dialog, retained so it stays
        # alive and is inspectable by tests.
        self._summary_dialog: Optional[QDialog] = None
        # Currently shown (window-modal, non-blocking) terminal-failure box.
        # Reused across failures so repeated signals never pile boxes up; the
        # single owned reference keeps it alive and inspectable by tests.
        self._error_message_box: Optional[QMessageBox] = None
        # Presentation count for the owned error box (read-only test seam).
        self._error_box_count: int = 0
        self.solver_probe = (
            solver_probe if solver_probe is not None else probe_zesolver_operational
        )
        # ZeAnalyser launch seam (M7).  Injectable so tests never spawn a real
        # ZeAnalyser process.  The launcher has signature
        # ``(input_folder, lang, command_file_path) -> bool`` (True = spawned);
        # the maker returns the command-file path to pass via
        # ``ZEANALYSER_COMMAND_FILE``.
        self._analyzer_launcher = (
            analyzer_launcher
            if analyzer_launcher is not None
            else analyzer_launch.launch_analyzer
        )
        self._analyzer_command_file_maker = (
            analyzer_command_file_maker
            if analyzer_command_file_maker is not None
            else analyzer_launch.make_command_file_path
        )
        # Command-file path used by the ZeAnalyser reference-return protocol;
        # set on launch and consumed (single-shot) by
        # ``_check_analyzer_command_file``.
        self._analyzer_command_file_path: Optional[str] = None
        # Ephemeral provenance for the visible reference field.  It is not a
        # persisted scientific setting: ZeAnalyser sets it for the immediate
        # run; direct typing/browsing resets it to USER semantics.
        self._reference_origin_hint: Optional[str] = None
        # ZeAnalyser reference-return watcher (M25.5-A).  A GUI-thread QTimer
        # polls the command file written by the launched ZeAnalyser and applies
        # the historical Tk consequences when ``REFERENCE=`` arrives.  It is
        # parented to ``self`` so Qt destroys it with the window (no callback
        # into a destroyed ``MainWindow``) and is explicitly stopped by
        # :meth:`shutdown`.
        self._analyzer_watch_timer = QTimer(self)
        self._analyzer_watch_timer.setInterval(ANALYZER_WATCH_INTERVAL_MS)
        self._analyzer_watch_timer.timeout.connect(self._analyzer_watch_tick)
        self._shutdown_wait_ms = shutdown_wait_ms
        # Settings/geometry persistence (M8).  ``None`` disables persistence so
        # bare ``MainWindow()`` constructions (tests) never touch a real file;
        # the Qt entry point passes the platform-aware user-config default
        # (``settings_persistence.resolve_settings_path``, M25.5-B).
        self._settings_path = os.path.abspath(settings_path) if settings_path else None
        self.setWindowTitle(title if title is not None else default_window_title())
        # Window icon (Tk ``root.iconphoto(True, ...)`` parity, M25.5-D).
        # Best-effort: a missing/undecodable packaged icon leaves the default
        # (empty) icon and the window still opens.
        _icon = load_window_icon()
        if _icon is not None:
            self.setWindowIcon(_icon)
        # Qt-local localization state (M9).  English by default; a persisted
        # ``language`` field (when present) overrides this after controls are
        # built.  ``_last_*`` / ``_preview_detail`` keep the raw value behind
        # the dynamic labels so a language switch can re-render them.
        self._language: str = localization.normalize_language("en")
        # Presentation theme mode (M25.5-C).  ``system`` (default) follows the
        # platform/style palette; a persisted ``theme`` field overrides it after
        # controls are built (mirrors the language flow).
        self._theme: str = DEFAULT_THEME
        self._text_bindings: List[tuple] = []
        self._last_elapsed_seconds: Optional[float] = 0.0
        self._last_remaining_text: str = UNKNOWN
        self._preview_detail: str = "—"
        self._running: bool = False
        # True while the boring (single-batch CSV) subprocess route is active.
        self._boring_active: bool = False
        # Lazily-created boring subprocess runner (default real runner only
        # used outside tests; tests inject a factory).
        self._boring_runner: Optional[BoringRunnerBase] = None
        self._shutdown_called: bool = False
        # Additional folders staged by the user before a run (Tk
        # ``additional_folders_to_process`` parity).  Passed as a copied list
        # into the RunRequest on Start.
        self._additional_folders: List[str] = []
        # Preview view state (M5): the copied original display image used as
        # the source for all display-only view transformations, plus the
        # accumulated clockwise rotation in degrees (0/90/180/270).
        self._preview_source: Optional[QImage] = None
        self._preview_rotation: int = 0
        # Option-A display-analysis state (ZSSS-OTPUX-QT-DISPLAY-STATE-01).
        # These are *display-only* owned copies and never feed the backend or
        # any scientific output:
        #   _raw_linear     - independent float64 copy of the raw-linear source
        #   _pristine_float - mapped pre-WB float buffer (analysis domain:
        #                     finite, >= 0; preserved headroom may exceed 1.0)
        #   _wb_only_float  - derived WB-only float buffer (same analysis
        #                     domain; WB gains are not clipped)
        #   _anchor_lo/hi   - stable p0.5/p99.5 anchors (hysteretically widened
        #                     for large drift; None until the first preview)
        #   _analysis_generation - explicit context/generation counter,
        #                     incremented on every analysis reset
        #   _wb_only_wb     - the WB gains used to derive _wb_only_float
        #   _wb_only_revision - monotonic counter bumped each time the WB-only
        #                     buffer is actually re-derived (WB/source change)
        self._raw_linear = None
        self._pristine_float = None
        self._wb_only_float = None
        self._wb_only_wb = None
        self._anchor_lo = None
        self._anchor_hi = None
        self._analysis_generation: int = 0
        self._wb_only_revision: int = 0
        self._phi_trace_ctx = None
        # Preview-resolution cycle factor (Tk ``preview_res_button`` parity,
        # M17).  Display-only GUI state; never touches the engine or
        # ``_preview_source``.  Default 1 (native) — see the module comment.
        self._preview_res_factor: int = DEFAULT_PREVIEW_RES_FACTOR
        # Continuous zoom factor (Tk ``PreviewManager.zoom_level`` parity, M18).
        # The ``zoom_combo`` percent presets and the mouse-wheel zoom both set
        # this single factor; "Fit" is a combo *mode* (not a numeric factor)
        # handled separately.  Range ``[MIN_ZOOM, MAX_ZOOM]``.
        self._preview_zoom_factor: float = 1.0
        # Pan offset in viewport pixels, relative to a centred image (Tk
        # ``_view_offset_x`` / ``_view_offset_y`` parity).  Unbounded — Tk
        # applies no clamping, so neither does the Qt shell.
        self._view_offset_x: float = 0.0
        self._view_offset_y: float = 0.0
        # Re-entrancy guard for the combo <-> continuous-factor sync loop.
        self._zoom_sync_guard: bool = False
        # Input folder whose first FITS was last successfully auto-loaded
        # (M12).  Avoids redundant reloads on repeated settings restore; it is
        # cleared when the folder changes or the load fails.
        self._last_preview_folder: Optional[str] = None
        # Preview display-adjustment state (M10/M13): white-balance R/G/B
        # gains, the display-stretch mode, black/white points, gamma and
        # brightness/contrast/saturation.  These act only on the derived
        # display image, never on ``_preview_source`` or any scientific output.
        # Defaults match the Tk GUI.  Raw histogram stats string (or ``None``)
        # drives the localized status label.
        self._wb: tuple = DEFAULT_WB
        self._stretch: str = DEFAULT_STRETCH
        self._black_point: float = DEFAULT_BLACK_POINT
        self._white_point: float = DEFAULT_WHITE_POINT
        self._gamma: float = DEFAULT_GAMMA
        self._brightness: float = DEFAULT_BRIGHTNESS
        self._contrast: float = DEFAULT_CONTRAST
        self._saturation: float = DEFAULT_SATURATION
        self._histogram_stats: Optional[str] = None
        # Cached authoritative histogram model (H1/H2).  The model is computed
        # *off the GUI thread* only when the WB-only analysis buffer is
        # re-derived (a new raw source or a WB change) — never on BP/WP/stretch/
        # gamma/BCS/zoom/pan/rotation.  ``_histogram_model_revision`` is the
        # ``_wb_only_revision`` the applied model was computed from;
        # ``_histogram_scheduled_revision`` is the revision a request was last
        # scheduled for (prevents duplicate scheduling on unrelated refreshes);
        # ``_histogram_compute_count`` is a testable seam counting real
        # recompute *decisions* (one per WB/source revision; see the property);
        # ``_histogram_source_stale`` counts results the GUI-thread source check
        # rejected (defence-in-depth on top of the coordinator generation check).
        self._histogram_model = None
        self._histogram_model_revision = None
        self._histogram_scheduled_revision = None
        self._histogram_trace_ctx = None
        self._histogram_source_stale: int = 0
        # Final OTPUX UX addendum: the detached histogram is a lazy, non-modal
        # second presentation surface.  It never owns a model or worker; the
        # reference stays alive across close/reopen so its geometry/view state
        # can be restored without altering processing or display state.
        self._detached_histogram_window: Optional[DetachedHistogramWindow] = None
        # Independent live-auto intent.  Enabled by default to restore batch-
        # boundary adaptation; direct manual BP/WP or WB-gain edits disable
        # only their corresponding flag.  One-shot Auto buttons do not.
        self._live_auto_stretch_enabled: bool = True
        self._live_auto_wb_enabled: bool = True
        self._last_live_auto_batch_token = None
        self._live_auto_stretch_count: int = 0
        self._live_auto_wb_count: int = 0
        # ZSSS-OTPUX-STABLE-A — stable scientific-preview identity.  The live
        # auto dedupe token is now the full identity ``(run_context_id, family,
        # counter)`` derived from engine metadata (positive ``image_count`` /
        # ``current_batch``), never the GUI ``drizzle_group_spin`` cadence.
        # ``family`` is ``"drizzle"`` (accepted-frame counter) or ``"batch"``
        # (Classic/Reproject share ``current_batch``); the ``PREV_SRC`` header
        # is *not* an identity dimension within the batch family because
        # ``refresh_preview`` re-renders every non-Drizzle session through the
        # SUM/W route.  ``_run_context_id`` is bumped once per run so identities
        # never collide across runs.  ``_raw_revision`` advances once per *new*
        # displayed scientific preview (a changed identity) and never on a
        # duplicate callback / repaint / same-batch route change.  ``_live_bp`` /
        # ``_live_wp`` record the BP/WP last written by live auto stretch so the
        # witness can compare live vs one-shot Auto Stretch on the same buffer.
        self._run_context_id: int = 0
        # PHI-R3.2 durable producer run/session identity (closes the cross-run
        # gate residual).  Each run is stamped with a durable producer session
        # id that both the GUI and the payloads share:
        #   _next_preview_run_session  — per-window monotonic allocator;
        #   _pending_preview_run_session — the id allocated for the run about
        #       to start (consumed/bound by ``_on_run_started``);
        #   _preview_run_session       — the producer session id bound to the
        #       CURRENT run (``None`` = no PHI-bound producer, e.g. simulated/
        #       legacy backends or idle state).
        # The GUI assigns the id at Start and hands it to the real backend
        # (``set_preview_session``), which stamps it onto the stacker so every
        # active-producer payload of the run carries ``PREV_RUN`` == the bound
        # id.  A payload carrying any other session id (a late old-run payload
        # queued across the boundary) is rejected as foreign and can never
        # poison the current run's sequence high-water mark.
        self._next_preview_run_session: int = 0
        self._pending_preview_run_session: Optional[int] = None
        self._preview_run_session: Optional[int] = None
        # PHI-R3 monotonic producer-sequence gate.  ``_last_accepted_preview_seq``
        # is the highest ``PREV_SEQ`` accepted from the current run's producer
        # (``None`` = no sequenced payload accepted yet in this run).  A payload
        # carrying a ``PREV_SEQ`` at or below it is a stale or duplicate
        # producer emission and is dropped before it can replace analysis/
        # display state or schedule work; payloads without ``PREV_SEQ``
        # (legacy/unsequenced producers and test payloads) bypass the gate and
        # keep the historical last-wins acceptance.  The gate resets at run
        # start (``_on_run_started``) so the first sequenced payload of a new
        # run is never rejected.
        self._last_accepted_preview_seq: Optional[int] = None
        # PHI-R3.1 analysis-domain state for the display controls.  For
        # Option-A float previews the black/white points (and their sliders /
        # spins / histogram markers) operate in the preserved analysis units
        # ``[0, _analysis_upper]`` with
        # ``_analysis_upper = max(1.0, finite max)`` of the WB-only analysis
        # buffer (exactly the float-histogram model range upper — the same
        # deterministic computation).  ``_bp_wp_control_upper`` remembers the
        # upper the BP/WP controls were last retooled for, so the retool only
        # runs when the analysis range actually changed.  Legacy QImage-only
        # previews keep the historical ``[0, 1]`` domain (upper stays ``1.0``).
        self._analysis_upper: float = 1.0
        self._bp_wp_control_upper: float = 1.0
        self._preview_mode = None
        self._preview_identity = None
        self._displayed_identity = None
        self._raw_revision: int = 0
        self._live_bp = None
        self._live_wp = None
        # Bounded latest-wins histogram coordinator (owns the worker QThread).
        # Created here (before widgets) so the GUI-thread result channel can be
        # wired in ``_wire_controls``; the thread itself is lazy (first
        # ``schedule``), so a bare ``MainWindow()`` spawns no thread.
        self._histogram_coordinator = HistogramCoordinator(
            compute_fn=histogram_compute_fn, parent=self
        )
        # True once teardown has begun (set at the top of ``shutdown``) so an
        # in-flight histogram result can never touch the UI during/after close.
        self._shutting_down: bool = False
        # Legacy single-array histogram/stats cache (H1 corrective).  Keyed by
        # ``(id(_preview_source), _wb)`` so a new source or a WB change
        # recomputes exactly once, while BP/WP/stretch/gamma/BCS/zoom/pan/
        # rotation refreshes only re-sync markers with zero recompute.
        self._legacy_hist_key: Optional[tuple] = None
        self._legacy_hist: Optional[Dict[str, Any]] = None
        self._legacy_hist_percentile: float = 1.0
        self._legacy_hist_stats: Optional[str] = None
        # Re-entrancy guards for atomic BP/WP and WB control application.
        self._bp_wp_sync_guard: bool = False
        self._wb_sync_guard: bool = False
        # Re-entrancy guard for the Resume apply flow: while a discovered
        # config is being pushed into the controls, the programmatic widget
        # writes must not be mistaken for user-originated path changes that
        # would invalidate the just-prepared Resume (RSM2-02C R1).
        self._applying_resume_result: bool = False
        self.settings_state: QtSettingsState = QtSettingsState()
        self.controller = RunController(self)

        self._build_central()
        self._build_status_bar()
        self._wire_controls()
        self._wire_settings_controls()
        self._wire_controller()
        self._sync_state_from_controls()
        # Apply the default presentation theme palette at construction so a
        # bare window always starts from the platform/style palette (M25.5-C).
        # A persisted ``theme`` is re-applied by ``_load_persisted_settings``.
        self._apply_theme_palette(self._theme)
        if self._settings_path:
            self._load_persisted_settings()

    # ------------------------------------------------------------------ UI
    def _build_central(self) -> None:
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_panel = self._build_left_panel()
        self.right_panel = self._build_right_panel()
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        self.setCentralWidget(self.splitter)

    def _build_left_panel(self) -> QScrollArea:
        """Build the scrollable left control panel (Tk ``control_notebook``)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        outer = QVBoxLayout(container)

        self.tabs = QTabWidget()
        self._stacking_tab = self._build_stacking_tab()
        self._settings_tab = self._build_settings_tab()
        self._system_tab = self._build_system_tab()
        self._preview_controls_tab = self._build_preview_controls_tab()
        self.tabs.addTab(self._stacking_tab, self._tr("tab_stacking"))
        self.tabs.addTab(self._settings_tab, self._tr("tab_expert"))
        self.tabs.addTab(self._system_tab, self._tr("tab_system"))
        self.tabs.addTab(self._preview_controls_tab, self._tr("tab_preview_controls"))
        outer.addWidget(self.tabs)

        # Progress / status area (Tk ``progress_frame``).
        self.progress_label = QLabel(self._tr("progress_label"))
        self._bind_text(self.progress_label, "progress_label")
        outer.addWidget(self.progress_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        outer.addWidget(self.progress)

        # Elapsed / remaining time surface (M6).  Driven only by the existing
        # progress lifecycle signals; unknown values render as "—".
        time_row = QHBoxLayout()
        self.elapsed_label = QLabel()
        self.remaining_label = QLabel()
        self._render_elapsed_label()
        self._render_remaining_label()
        time_row.addWidget(self.elapsed_label)
        time_row.addStretch(1)
        time_row.addWidget(self.remaining_label)
        outer.addLayout(time_row)

        # Log area (Tk ``status_text``) + Copy Log action (M6).
        log_header = QHBoxLayout()
        self.log_label = QLabel(self._tr("log_label"))
        self._bind_text(self.log_label, "log_label")
        log_header.addWidget(self.log_label)
        log_header.addStretch(1)
        self.copy_log_button = QPushButton(self._tr("copy_log"))
        self._bind_text(self.copy_log_button, "copy_log")
        self.copy_log_button.setEnabled(False)
        log_header.addWidget(self.copy_log_button)
        outer.addLayout(log_header)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        outer.addWidget(self.log_view, 1)

        scroll.setWidget(container)
        return scroll

    def _build_system_tab(self) -> QWidget:
        """Build the "System" tab (M25.5-C): application/runtime settings.

        Groups the Language switch (moved from the left-panel top), the
        "Use GPU" toggle (moved from the Stacking tab — same ``use_gpu`` state
        key and same M20 RunRequest/seam plumbing, only the widget location
        changed) and the presentation-only Appearance theme selector
        (System/Dark/Light).
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        form = QFormLayout()

        # Language control (M9): user-triggered FR/EN switch.  Same widget and
        # same ``_on_language_changed`` live re-translation wiring as before.
        self.language_label = QLabel(self._tr("language_label"))
        self._bind_text(self.language_label, "language_label")
        self.language_combo = QComboBox()
        self.language_combo.addItems(
            [localization.LANGUAGE_LABELS[code] for code in localization.SUPPORTED_LANGUAGES]
        )
        self.language_combo.setEnabled(True)
        form.addRow(self.language_label, self.language_combo)

        # Use GPU toggle (moved from the Stacking tab).  Same QCheckBox, same
        # ``drizzle_use_gpu`` label key, same ``use_gpu`` state key and same
        # M20 seam plumbing.  Enablement is capability-driven (M5): the probe
        # runs deferred (never during startup) and the status label reports the
        # REAL probed GPU capability instead of an arbitrary boolean.
        self.use_gpu_check = QCheckBox(self._tr("drizzle_use_gpu"))
        self._bind_text(self.use_gpu_check, "drizzle_use_gpu")
        self.use_gpu_check.setChecked(bool(self.settings_state.use_gpu))
        form.addRow("", self.use_gpu_check)

        # GPU status label (M5): live capability line, refreshed when the
        # checkbox toggles.  Initially the probing placeholder; the deferred
        # refresh replaces it as soon as the event loop turns.  Deliberately
        # NOT ``_bind_text``-bound: the text is dynamic (probe result), so a
        # language re-translation must not clobber it.
        self.gpu_status_header = QLabel(self._tr("gpu_status"))
        self._bind_text(self.gpu_status_header, "gpu_status")
        self.gpu_status_label = QLabel()
        self.gpu_status_label.setText(self._tr("gpu_status_probing"))
        form.addRow(self.gpu_status_header, self.gpu_status_label)
        self._gpu_capabilities = None
        self._gpu_probe_worker = None  # owned worker; see _start_gpu_probe
        self._refresh_gpu_check_enabled()
        # Defer the real probe to a background QThread (F3): the cold probe
        # (CuPy import/JIT, nvidia-smi) must never run on the Qt main thread.
        QTimer.singleShot(0, self._start_gpu_probe)

        # Appearance / theme (presentation-only).  Default System.
        self.appearance_label = QLabel(self._tr("appearance_label"))
        self._bind_text(self.appearance_label, "appearance_label")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems([self._tr(THEME_CHOICE_KEYS[m]) for m in THEME_MODES])
        self.theme_combo.setCurrentIndex(THEME_MODES.index(DEFAULT_THEME))
        form.addRow(self.appearance_label, self.theme_combo)

        layout.addLayout(form)
        layout.addStretch(1)
        return panel

    def _build_stacking_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # --- Minimal settings form (visible subset only) ---
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.temp_edit = QLineEdit()
        self.output_filename_edit = QLineEdit()
        self.reference_edit = QLineEdit()
        self.last_stack_edit = QLineEdit()

        self.browse_input_button = QPushButton(self._tr("browse"))
        self._bind_text(self.browse_input_button, "browse")
        self.browse_output_button = QPushButton(self._tr("browse"))
        self._bind_text(self.browse_output_button, "browse")
        self.browse_temp_button = QPushButton(self._tr("browse"))
        self._bind_text(self.browse_temp_button, "browse")
        self.browse_reference_button = QPushButton(self._tr("browse"))
        self._bind_text(self.browse_reference_button, "browse")
        self.browse_last_stack_button = QPushButton("…")

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(-1, 1_000_000)
        self.batch_spin.setValue(0)
        self.batch_spin.setToolTip("Batch size (-1 = auto).")

        # Boring (single-batch CSV) mode toggle — Tk ``boring_thread_check``
        # parity.  Checked <=> batch_size == 1.
        self.boring_check = QCheckBox(self._tr("boring_check"))
        self._bind_text(self.boring_check, "boring_check")
        self.boring_check.setChecked(False)

        self.stacking_mode_combo = QComboBox()
        self.stacking_mode_combo.addItems(STACKING_MODES)
        self.stacking_mode_combo.setCurrentText("kappa-sigma")

        # Final-combination business control (Tk ``stack_final_combo``).
        self.final_combine_combo = QComboBox()
        self.final_combine_combo.addItems(
            [FINAL_COMBINE_LABELS[k] for k in FINAL_COMBINE_KEYS]
        )
        default_label = FINAL_COMBINE_LABELS.get(
            self.settings_state.stack_final_combine, "Mean"
        )
        self.final_combine_combo.setCurrentText(default_label)

        # HQ RAM limit (GB) — Tk ``max_hq_mem_var`` (Stacking tab).  Forwarded
        # to the boring single-batch subprocess as ``--max-mem`` (M25) and to
        # the regular run path as the M20 seam field ``max_hq_mem_gb``.
        self.max_hq_mem_spin = QSpinBox()
        self.max_hq_mem_spin.setRange(1, 64)
        self.max_hq_mem_spin.setSingleStep(1)
        self.max_hq_mem_spin.setValue(int(self.settings_state.max_hq_mem_gb))

        self.drizzle_check = QCheckBox(self._tr("drizzle_check"))
        self._bind_text(self.drizzle_check, "drizzle_check")
        self.drizzle_check.setChecked(False)

        self.drizzle_mode_combo = QComboBox()
        # Label displayed / backend value stored (R3b): the combo shows the
        # localized "Standard" / "Large dataset" labels while ``itemData``
        # carries "Final" / "Incremental" for state, persistence and engine.
        for mode, label_key in zip(DRIZZLE_MODES, DRIZZLE_MODE_LABEL_KEYS):
            self.drizzle_mode_combo.addItem(self._tr(label_key), mode)
        self.drizzle_mode_combo.setCurrentIndex(DRIZZLE_MODES.index("Final"))

        self.drizzle_group_spin = QSpinBox()
        self.drizzle_group_spin.setRange(1, 100_000)
        self.drizzle_group_spin.setSingleStep(10)
        self.drizzle_group_spin.setValue(50)

        # Drizzle advanced sub-options (D3 contract): moved from the Expert-tab
        # "Drizzle Advanced" section into the Drizzle block of the Stacking tab,
        # in the exact Tk "Options Drizzle" order, before local solver.
        # Same widget specs as the former Expert fields.
        self.drizzle_scale_spin = QSpinBox()
        # x1 is a valid runtime-effective value and must remain representable
        # when restoring a headless standard-Drizzle checkpoint.  The normal
        # new-run default remains x2.
        self.drizzle_scale_spin.setRange(1, 4)
        self.drizzle_scale_spin.setSingleStep(1)
        self.drizzle_scale_spin.setValue(int(self.settings_state.drizzle_scale))

        self.drizzle_wht_spin = QDoubleSpinBox()
        self.drizzle_wht_spin.setRange(0.0, 1.0)
        self.drizzle_wht_spin.setSingleStep(0.01)
        self.drizzle_wht_spin.setDecimals(3)
        self.drizzle_wht_spin.setValue(float(self.settings_state.drizzle_wht_threshold))

        self.drizzle_kernel_combo = QComboBox()
        self.drizzle_kernel_combo.addItems(list(DRIZZLE_KERNELS))
        kernel_text = str(self.settings_state.drizzle_kernel)
        if kernel_text in DRIZZLE_KERNELS:
            self.drizzle_kernel_combo.setCurrentText(kernel_text)

        self.drizzle_pixfrac_spin = QDoubleSpinBox()
        self.drizzle_pixfrac_spin.setRange(0.01, 2.0)
        self.drizzle_pixfrac_spin.setSingleStep(0.05)
        self.drizzle_pixfrac_spin.setDecimals(2)
        self.drizzle_pixfrac_spin.setValue(float(self.settings_state.drizzle_pixfrac))

        self.solver_combo = QComboBox()
        self.solver_combo.addItems(SOLVER_PREFERENCES)
        self.solver_combo.setCurrentText("none")

        # Explicit New/Resume selector (RSM2-02C).  Fresh/New by default on
        # every construction; a persisted/browsed/edited last-stack path alone
        # never selects Resume — only this explicit user choice does.
        self.resume_mode_combo = QComboBox()
        self.resume_mode_combo.addItem(self._tr("resume_mode_new"), "fresh")
        self.resume_mode_combo.addItem(self._tr("resume_mode_resume"), "resume")
        self.resume_mode_combo.setCurrentIndex(0)

        form = QFormLayout()
        self._add_form_row(
            form,
            "input_folder",
            self._path_row(self.input_edit, self.browse_input_button),
        )
        self._add_form_row(
            form,
            "output_folder",
            self._path_row(self.output_edit, self.browse_output_button),
        )
        self._add_form_row(
            form,
            "temp_folder",
            self._path_row(self.temp_edit, self.browse_temp_button),
        )
        self._add_form_row(form, "output_filename", self.output_filename_edit)
        self._add_form_row(
            form,
            "reference_image",
            self._path_row(self.reference_edit, self.browse_reference_button),
        )
        self._add_form_row(
            form,
            "last_stack",
            self._path_row(self.last_stack_edit, self.browse_last_stack_button),
        )
        self._add_form_row(form, "resume_mode_label", self.resume_mode_combo)
        self._add_form_row(form, "batch_size", self.batch_spin)
        form.addRow("", self.boring_check)
        self._add_form_row(form, "stacking_mode", self.stacking_mode_combo)
        self._add_form_row(form, "final_combine", self.final_combine_combo)
        self._add_form_row(form, "hq_ram_limit", self.max_hq_mem_spin)
        form.addRow("", self.drizzle_check)
        self._add_form_row(form, "drizzle_mode", self.drizzle_mode_combo)
        self._add_form_row(form, "drizzle_group_size", self.drizzle_group_spin)
        self._add_form_row(form, "field_drizzle_scale", self.drizzle_scale_spin)
        self._add_form_row(form, "field_drizzle_wht_threshold", self.drizzle_wht_spin)
        self._add_form_row(form, "field_drizzle_kernel", self.drizzle_kernel_combo)
        self._add_form_row(form, "field_drizzle_pixfrac", self.drizzle_pixfrac_spin)
        self._add_form_row(form, "local_solver", self.solver_combo)
        layout.addLayout(form)
        layout.addStretch(1)

        return panel

    def _path_row(self, edit: QLineEdit, button: QPushButton) -> QWidget:
        """Build a horizontal row with a line edit and a Browse button."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        h.addWidget(button)
        return row

    def _make_slider_spin_pair(
        self,
        min_val: float,
        max_val: float,
        step: float,
        decimals: int,
        default: float,
        on_change,
    ):
        """Build a Tk-parity slider + spinbox pair.

        Returns ``(row_widget, slider, spin)`` where the slider drives a coarse
        sweep and the spinbox shows the exact numeric value next to it (Tk's
        ``_create_slider_spinbox_group`` shows the value in a spinbox beside
        the slider).  A guard flag keeps the two widgets in sync without
        ping-ponging; ``on_change`` fires once per user edit.
        """
        slider = QSlider(Qt.Orientation.Horizontal)
        n = max(1, int(round((max_val - min_val) / step)))
        slider.setRange(0, n)

        spin = QDoubleSpinBox()
        spin.setRange(float(min_val), float(max_val))
        spin.setSingleStep(float(step))
        spin.setDecimals(int(decimals))

        guard = {"busy": False}

        def _slider_changed(value: int) -> None:
            if guard["busy"]:
                return
            guard["busy"] = True
            try:
                spin.setValue(min_val + value * step)
            finally:
                guard["busy"] = False
            on_change()

        def _spin_changed(value: float) -> None:
            if guard["busy"]:
                return
            guard["busy"] = True
            try:
                slider.setValue(int(round((value - min_val) / step)))
            finally:
                guard["busy"] = False
            on_change()

        # Set the initial values *before* wiring the signals so the setup
        # never fires ``on_change`` while the caller is still assigning the
        # resulting attributes.
        slider.setValue(int(round((default - min_val) / step)))
        spin.setValue(float(default))

        slider.valueChanged.connect(_slider_changed)
        spin.valueChanged.connect(_spin_changed)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        h.addWidget(spin)
        return row, slider, spin

    def _build_preview_controls_tab(self) -> QWidget:
        """Left-panel "Preview controls" tab (M10/M13): display-only WB /
        stretch / black-white-gamma / brightness-contrast-saturation /
        histogram.  These controls act only on the *displayed* preview pixmap
        (a derived image), never on the stored source image or any scientific
        output.  All controls are inert until a renderable preview arrives.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # White balance: Auto WB + Reset buttons, then per-channel R/G/B gains
        # (neutral = 1.0) shown as slider + numeric spinbox pairs.
        self.wb_group = QGroupBox(self._tr("wb_group"))
        self._bind_text(self.wb_group, "wb_group")
        wb_form = QFormLayout(self.wb_group)
        wb_buttons = QWidget()
        wb_btn_row = QHBoxLayout(wb_buttons)
        wb_btn_row.setContentsMargins(0, 0, 0, 0)
        self.auto_wb_button = QPushButton(self._tr("auto_wb"))
        self._bind_text(self.auto_wb_button, "auto_wb")
        self.live_auto_wb_check = QCheckBox(self._tr("live_auto_wb"))
        self._bind_text(self.live_auto_wb_check, "live_auto_wb")
        self.live_auto_wb_check.setChecked(self._live_auto_wb_enabled)
        self.wb_reset_button = QPushButton(self._tr("wb_reset"))
        self._bind_text(self.wb_reset_button, "wb_reset")
        wb_btn_row.addWidget(self.auto_wb_button)
        wb_btn_row.addWidget(self.wb_reset_button)
        wb_btn_row.addWidget(self.live_auto_wb_check)
        wb_btn_row.addStretch(1)
        wb_form.addRow("", wb_buttons)
        self.wb_r_row, self.wb_r_slider, self.wb_r_spin = self._make_slider_spin_pair(
            WB_MIN, WB_MAX, WB_STEP, 2, DEFAULT_WB[0], self._on_wb_changed
        )
        self.wb_g_row, self.wb_g_slider, self.wb_g_spin = self._make_slider_spin_pair(
            WB_MIN, WB_MAX, WB_STEP, 2, DEFAULT_WB[1], self._on_wb_changed
        )
        self.wb_b_row, self.wb_b_slider, self.wb_b_spin = self._make_slider_spin_pair(
            WB_MIN, WB_MAX, WB_STEP, 2, DEFAULT_WB[2], self._on_wb_changed
        )
        self._add_form_row(wb_form, "wb_red", self.wb_r_row)
        self._add_form_row(wb_form, "wb_green", self.wb_g_row)
        self._add_form_row(wb_form, "wb_blue", self.wb_b_row)
        layout.addWidget(self.wb_group)

        # Display stretch (Asinh default, Tk parity) + black/white/gamma.
        self.stretch_group = QGroupBox(self._tr("stretch_group"))
        self._bind_text(self.stretch_group, "stretch_group")
        stretch_form = QFormLayout(self.stretch_group)
        self.stretch_combo = QComboBox()
        self.stretch_combo.addItems(list(STRETCH_MODES))
        self.stretch_combo.setCurrentText(DEFAULT_STRETCH)
        self._add_form_row(stretch_form, "stretch_label", self.stretch_combo)
        (
            self.stretch_bp_row,
            self.stretch_bp_slider,
            self.stretch_bp_spin,
        ) = self._make_slider_spin_pair(
            BLACK_POINT_MIN,
            BLACK_POINT_MAX,
            BLACK_POINT_STEP,
            3,
            DEFAULT_BLACK_POINT,
            self._on_stretch_bp_changed,
        )
        (
            self.stretch_wp_row,
            self.stretch_wp_slider,
            self.stretch_wp_spin,
        ) = self._make_slider_spin_pair(
            WHITE_POINT_MIN,
            WHITE_POINT_MAX,
            WHITE_POINT_STEP,
            3,
            DEFAULT_WHITE_POINT,
            self._on_stretch_wp_changed,
        )
        (
            self.stretch_gamma_row,
            self.stretch_gamma_slider,
            self.stretch_gamma_spin,
        ) = self._make_slider_spin_pair(
            GAMMA_MIN,
            GAMMA_MAX,
            GAMMA_STEP,
            2,
            DEFAULT_GAMMA,
            self._on_stretch_gamma_changed,
        )
        self._add_form_row(stretch_form, "stretch_black", self.stretch_bp_row)
        self._add_form_row(stretch_form, "stretch_white", self.stretch_wp_row)
        self._add_form_row(stretch_form, "stretch_gamma", self.stretch_gamma_row)
        self.auto_stretch_button = QPushButton(self._tr("auto_stretch"))
        self._bind_text(self.auto_stretch_button, "auto_stretch")
        self.live_auto_stretch_check = QCheckBox(self._tr("live_auto_stretch"))
        self._bind_text(self.live_auto_stretch_check, "live_auto_stretch")
        self.live_auto_stretch_check.setChecked(self._live_auto_stretch_enabled)
        self.stretch_reset_button = QPushButton(self._tr("stretch_reset"))
        self._bind_text(self.stretch_reset_button, "stretch_reset")
        stretch_buttons = QWidget()
        stretch_btn_row = QHBoxLayout(stretch_buttons)
        stretch_btn_row.setContentsMargins(0, 0, 0, 0)
        stretch_btn_row.addWidget(self.auto_stretch_button)
        stretch_btn_row.addWidget(self.stretch_reset_button)
        stretch_btn_row.addWidget(self.live_auto_stretch_check)
        stretch_btn_row.addStretch(1)
        stretch_form.addRow("", stretch_buttons)
        layout.addWidget(self.stretch_group)

        # Image adjustments: brightness / contrast / saturation (Tk parity).
        self.bcs_group = QGroupBox(self._tr("bcs_group"))
        self._bind_text(self.bcs_group, "bcs_group")
        bcs_form = QFormLayout(self.bcs_group)
        (
            self.brightness_row,
            self.brightness_slider,
            self.brightness_spin,
        ) = self._make_slider_spin_pair(
            BRIGHTNESS_MIN,
            BRIGHTNESS_MAX,
            BRIGHTNESS_STEP,
            2,
            DEFAULT_BRIGHTNESS,
            self._on_bcs_changed,
        )
        (
            self.contrast_row,
            self.contrast_slider,
            self.contrast_spin,
        ) = self._make_slider_spin_pair(
            CONTRAST_MIN,
            CONTRAST_MAX,
            CONTRAST_STEP,
            2,
            DEFAULT_CONTRAST,
            self._on_bcs_changed,
        )
        (
            self.saturation_row,
            self.saturation_slider,
            self.saturation_spin,
        ) = self._make_slider_spin_pair(
            SATURATION_MIN,
            SATURATION_MAX,
            SATURATION_STEP,
            2,
            DEFAULT_SATURATION,
            self._on_bcs_changed,
        )
        self._add_form_row(bcs_form, "brightness", self.brightness_row)
        self._add_form_row(bcs_form, "contrast", self.contrast_row)
        self._add_form_row(bcs_form, "saturation", self.saturation_row)
        self.bcs_reset_button = QPushButton(self._tr("bcs_reset"))
        self._bind_text(self.bcs_reset_button, "bcs_reset")
        bcs_form.addRow("", self.bcs_reset_button)
        layout.addWidget(self.bcs_group)

        self._set_preview_controls_enabled(False)
        layout.addStretch(1)
        return panel

    def _build_right_panel(self) -> QWidget:
        """Build the persistent right preview/action panel (Tk ``preview_frame``
        + view + histogram + action buttons).

        The histogram group here is the persistent right-panel surface required
        by checklist item 13.4.  It is the *single* live histogram surface (the
        duplicated Preview-controls-tab histogram was removed in M14) and gains
        the Tk histogram interactions via :class:`HistogramView`.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Preview (persistent across left-tab switches).
        preview_group = QGroupBox(self._tr("preview_group"))
        self._bind_text(preview_group, "preview_group")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel()
        self._render_preview_label()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_label)
        self.preview_image_label = PreviewImageView()
        preview_layout.addWidget(self.preview_image_label, 1)
        # Initial empty-preview placeholder: the packaged ``back.png`` (Tk
        # ``PreviewManager`` background parity, M25.5-D).  Best-effort — a
        # missing resource keeps the cleared (null) pixmap.
        self._show_empty_preview()
        layout.addWidget(preview_group, 1)

        # Zoom / resolution / rotation controls (basic/disabled placeholders).
        view_group = QGroupBox(self._tr("view_group"))
        self._bind_text(view_group, "view_group")
        view_layout = QGridLayout(view_group)
        self.zoom_caption = QLabel(self._tr("zoom_label"))
        self._bind_text(self.zoom_caption, "zoom_label")
        view_layout.addWidget(self.zoom_caption, 0, 0)
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(list(ZOOM_LABELS))
        # Default view is 100% (native size) — Tk parity: the preview starts at
        # full resolution, not fitted into the (not-yet-laid-out) label.
        self.zoom_combo.setCurrentText("100%")
        self.zoom_combo.setEnabled(False)
        view_layout.addWidget(self.zoom_combo, 0, 1)
        self.resolution_caption = QLabel(self._tr("resolution_label"))
        self._bind_text(self.resolution_caption, "resolution_label")
        view_layout.addWidget(self.resolution_caption, 0, 2)
        self.resolution_label = QLabel("—")
        view_layout.addWidget(self.resolution_label, 0, 3)
        self.rotate_left_button = QPushButton("⟲")
        self.rotate_left_button.setEnabled(False)
        self.rotate_right_button = QPushButton("⟳")
        self.rotate_right_button.setEnabled(False)
        view_layout.addWidget(self.rotate_left_button, 1, 0)
        view_layout.addWidget(self.rotate_right_button, 1, 1)
        # Preview resolution-cycle button (Tk ``preview_res_button`` parity,
        # M17).  Cycles 1/1..1/4; display-only (never the engine).  Always
        # enabled like the Tk button (the cycle only changes GUI state + the
        # local display, and no-ops while there is no preview).
        self.preview_res_button = QPushButton(self._preview_res_text())
        self.preview_res_button.setToolTip("Cycle preview resolution (1/1..1/4).")
        view_layout.addWidget(self.preview_res_button, 1, 2)
        layout.addWidget(view_group)

        # Persistent display histogram (Tk parity, checklist item 13.4).  This
        # is now the *single* live histogram surface: the duplicated tab
        # histogram was removed in M14, and this right-panel surface gained the
        # Tk histogram interactions (auto-zoom / reset view / zoom / reset zoom
        # + BP/WP line dragging via ``HistogramView``).
        self.right_histogram_group = QGroupBox(self._tr("histogram_group"))
        self._bind_text(self.right_histogram_group, "histogram_group")
        right_histo_layout = QVBoxLayout(self.right_histogram_group)
        self.right_histogram_status = QLabel(self._tr("histogram_empty"))
        self.right_histogram_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_histo_layout.addWidget(self.right_histogram_status)
        self.right_histogram_view = HistogramView()
        self.right_histogram_view.setMinimumSize(256, 64)
        right_histo_layout.addWidget(self.right_histogram_view, 1)
        # Histogram toolbar (Tk ``histo_toolbar`` parity).
        histo_toolbar = QWidget()
        histo_toolbar_row = QHBoxLayout(histo_toolbar)
        histo_toolbar_row.setContentsMargins(0, 0, 0, 0)
        self.auto_zoom_histo_check = QCheckBox(self._tr("histo_auto_zoom"))
        self._bind_text(self.auto_zoom_histo_check, "histo_auto_zoom")
        self.hist_reset_view_button = QPushButton(self._tr("histo_reset"))
        self._bind_text(self.hist_reset_view_button, "histo_reset")
        self.hist_zoom_button = QPushButton(self._tr("histo_zoom"))
        self._bind_text(self.hist_zoom_button, "histo_zoom")
        self.hist_reset_button = QPushButton("R")
        self.hist_reset_button.setToolTip("Reset zoom")
        self.hist_expand_button = QPushButton(self._tr("histo_expand"))
        self._bind_text(self.hist_expand_button, "histo_expand")
        self.hist_expand_button.setToolTip(
            "Open the live histogram in a large window"
        )
        # Inert until a renderable preview arrives (matches the WB/stretch
        # controls); ``_set_preview_controls_enabled`` re-arms them.
        self.auto_zoom_histo_check.setEnabled(False)
        self.hist_reset_view_button.setEnabled(False)
        self.hist_zoom_button.setEnabled(False)
        self.hist_reset_button.setEnabled(False)
        self.hist_expand_button.setEnabled(False)
        histo_toolbar_row.addWidget(self.auto_zoom_histo_check)
        histo_toolbar_row.addWidget(self.hist_reset_view_button)
        histo_toolbar_row.addWidget(self.hist_zoom_button)
        histo_toolbar_row.addWidget(self.hist_reset_button)
        histo_toolbar_row.addWidget(self.hist_expand_button)
        histo_toolbar_row.addStretch(1)
        right_histo_layout.addWidget(histo_toolbar)
        layout.addWidget(self.right_histogram_group)

        # Action buttons (Start/Stop/Analyse/Solver/path actions functional).
        # M25.5-E: a compact single-band QHBoxLayout mirrors the Tk right-panel
        # ``control_frame`` side-by-side packing instead of the former dense
        # 2-column grid — run controls (Start / Stop / Analyse / Solver) packed
        # on the left, folder actions (View Inputs / Add Folder / Open Output)
        # on the right, with a stretch between them so no button is ever forced
        # to the full group width (Open Output loses its former 2-column span).
        self.actions_group = QGroupBox(self._tr("actions_group"))
        self._bind_text(self.actions_group, "actions_group")
        actions_layout = QHBoxLayout(self.actions_group)
        actions_layout.setSpacing(6)
        actions_layout.setContentsMargins(4, 4, 4, 4)
        self.start_button = QPushButton(self._tr("start"))
        self._bind_text(self.start_button, "start")
        self.stop_button = QPushButton(self._tr("stop"))
        self._bind_text(self.stop_button, "stop")
        self.analyse_button = QPushButton(self._tr("analyse"))
        self._bind_text(self.analyse_button, "analyse")
        self.solver_button = QPushButton(self._tr("solver"))
        self._bind_text(self.solver_button, "solver")
        self.view_inputs_button = QPushButton(self._tr("view_inputs"))
        self._bind_text(self.view_inputs_button, "view_inputs")
        self.add_folder_button = QPushButton(self._tr("add_folder"))
        self._bind_text(self.add_folder_button, "add_folder")
        self.open_output_button = QPushButton(self._tr("open_output"))
        self._bind_text(self.open_output_button, "open_output")
        # Analyse (ZeAnalyser launch, M7), View Inputs / Add Folder / Open
        # Output are all wired to user-triggered actions; their enablement is
        # driven by ``_update_path_action_state``.
        self.analyse_button.setEnabled(False)
        # Start is the primary action: mark it the default button (Tk
        # ``Accent.TButton`` parity — the emphasized action).  This is a
        # property only; no stylesheet, no wiring change.
        self.start_button.setDefault(True)
        actions_layout.addWidget(self.start_button)
        actions_layout.addWidget(self.stop_button)
        actions_layout.addWidget(self.analyse_button)
        actions_layout.addWidget(self.solver_button)
        actions_layout.addStretch(1)
        actions_layout.addWidget(self.view_inputs_button)
        actions_layout.addWidget(self.add_folder_button)
        actions_layout.addWidget(self.open_output_button)
        layout.addWidget(self.actions_group)

        # Honest backend-mode notice (M9 fix): always visible next to Start so
        # a witness knows the default Start click is simulated.
        self.backend_notice_label = QLabel(self._backend_notice_text())
        self.backend_notice_label.setWordWrap(True)
        layout.addWidget(self.backend_notice_label)

        return panel

    def _build_settings_tab(self) -> QWidget:
        """Build the scrollable, grouped Settings surface (M10).

        Every backend-relevant :class:`QtSettingsState` field that the Stack
        tab does not already expose gets a real widget here.  Widget references
        stay owned by ``MainWindow`` (in ``self._settings_widgets`` /
        ``self._mosaic_widgets``); they are mirrored into ``settings_state`` by
        :meth:`_sync_state_from_controls` on every change.
        """
        self._settings_widgets = {}
        self._settings_kinds = {}
        self._mosaic_widgets = {}
        # attr -> the QFormLayout hosting its row, so the kappa/winsor
        # visibility toggle (M17) can hide/show the whole label+widget row via
        # ``QFormLayout.setRowVisible``.
        self._settings_forms = {}

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        outer = QVBoxLayout(container)

        # Expert-tab warning banner (Tk ``warning_label`` parity): a red,
        # italicised "Expert Settings!" label above the grouped sections.
        self.expert_warning_label = QLabel(self._tr("expert_warning_text"))
        self._bind_text(self.expert_warning_label, "expert_warning_text")
        self.expert_warning_label.setStyleSheet("color: red; font-style: italic;")
        outer.addWidget(self.expert_warning_label)

        for section_title, fields in SETTINGS_SECTIONS:
            if fields is None:
                group = self._build_mosaic_section()
            else:
                group = self._build_generic_section(SECTION_TITLE_KEYS[section_title], fields)
            outer.addWidget(group)

        # Reset Expert Settings button (Tk ``reset_expert_button`` parity).
        self.reset_expert_button = QPushButton(self._tr("reset_expert_button"))
        self._bind_text(self.reset_expert_button, "reset_expert_button")
        self.reset_expert_button.clicked.connect(self._reset_expert_settings)
        outer.addWidget(self.reset_expert_button)

        outer.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _build_generic_section(self, title_key: str, fields) -> QGroupBox:
        """Build one QGroupBox section from a list of field specs."""
        group = QGroupBox(self._tr(title_key))
        self._bind_text(group, title_key)
        form = QFormLayout(group)
        for field in fields:
            attr, label, kind = field[0], field[1], field[2]
            params = field[3:]
            widget = self._make_settings_widget(attr, kind, *params)
            key = LOCALIZED_SETTINGS_FIELD_KEYS.get(attr)
            if key is not None:
                self._add_form_row(form, key, widget)
            else:
                form.addRow(label, widget)
            self._settings_widgets[attr] = widget
            self._settings_kinds[attr] = kind
            self._settings_forms[attr] = form
        return group

    def _build_mosaic_section(self) -> QGroupBox:
        """Build the Mosaic section (top-level flag + nested dict sub-fields)."""
        state = self.settings_state
        ms = state.mosaic_settings if isinstance(state.mosaic_settings, dict) else {}
        group = QGroupBox(self._tr("section_mosaic"))
        self._bind_text(group, "section_mosaic")
        form = QFormLayout(group)

        self.mosaic_active_check = QCheckBox(self._tr("mosaic_mode_active"))
        self._bind_text(self.mosaic_active_check, "mosaic_mode_active")
        self.mosaic_active_check.setChecked(bool(state.mosaic_mode_active))
        form.addRow("", self.mosaic_active_check)
        self._settings_widgets["mosaic_mode_active"] = self.mosaic_active_check
        self._settings_kinds["mosaic_mode_active"] = "bool"

        for key, label, kind, params in MOSAIC_FIELDS:
            widget = self._make_mosaic_widget(kind, ms.get(key), params)
            self._add_form_row(form, MOSAIC_FIELD_KEYS[key], widget)
            self._mosaic_widgets[key] = (kind, widget)

        return group

    def _make_settings_widget(self, attr: str, kind: str, *params):
        """Create a single settings widget seeded from the model default."""
        default = getattr(self.settings_state, attr)
        if kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(default))
            return widget
        if kind == "int":
            lo, hi, step = params
            widget = QSpinBox()
            widget.setRange(int(lo), int(hi))
            widget.setSingleStep(int(step))
            widget.setValue(int(default))
            return widget
        if kind == "float":
            lo, hi, step, decimals = params
            widget = QDoubleSpinBox()
            widget.setRange(float(lo), float(hi))
            widget.setSingleStep(float(step))
            widget.setDecimals(int(decimals))
            widget.setValue(float(default))
            return widget
        if kind == "str":
            widget = QLineEdit()
            widget.setText(str(default) if default is not None else "")
            return widget
        if kind == "combo":
            choices = params[0]
            widget = QComboBox()
            widget.addItems(list(choices))
            text = str(default)
            if text in choices:
                widget.setCurrentText(text)
            return widget
        if kind == "list":
            widget = QLineEdit()
            text = ", ".join(str(x) for x in default) if default else ""
            widget.setText(text)
            return widget
        if kind == "match_bg":
            widget = QComboBox()
            widget.addItems(list(MATCH_BG_CHOICES))
            widget.setCurrentText(MATCH_BG_TO_TEXT.get(default, "default"))
            return widget
        raise ValueError(f"unknown settings field kind {kind!r}")

    def _make_mosaic_widget(self, kind: str, default, params):
        """Create a single mosaic sub-field widget seeded from the dict default."""
        if kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(default))
            return widget
        if kind == "int":
            lo, hi, step = params
            widget = QSpinBox()
            widget.setRange(int(lo), int(hi))
            widget.setSingleStep(int(step))
            widget.setValue(int(default))
            return widget
        if kind == "float":
            lo, hi, step, decimals = params
            widget = QDoubleSpinBox()
            widget.setRange(float(lo), float(hi))
            widget.setSingleStep(float(step))
            widget.setDecimals(int(decimals))
            widget.setValue(float(default))
            return widget
        if kind == "str":
            widget = QLineEdit()
            widget.setText(str(default) if default is not None else "")
            return widget
        if kind == "combo":
            choices = params
            widget = QComboBox()
            widget.addItems(list(choices))
            text = str(default)
            if text in choices:
                widget.setCurrentText(text)
            return widget
        raise ValueError(f"unknown mosaic field kind {kind!r}")

    def _connect_settings_widget(self, widget) -> None:
        """Connect a settings widget's change signal to the state mirror."""
        if isinstance(widget, QCheckBox):
            widget.stateChanged.connect(self._sync_state_from_controls)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(self._sync_state_from_controls)
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(self._sync_state_from_controls)
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._sync_state_from_controls)
        else:
            raise TypeError(f"unsupported settings widget {widget!r}")

    def _widget_value(self, kind: str, widget):
        """Read a plain Python value from a settings widget by kind."""
        if kind == "bool":
            return widget.isChecked()
        if kind == "int":
            return widget.value()
        if kind == "float":
            return widget.value()
        if kind == "str":
            return widget.text()
        if kind == "combo":
            return widget.currentText()
        if kind == "list":
            return [p.strip() for p in widget.text().split(",") if p.strip()]
        if kind == "match_bg":
            return MATCH_BG_FROM_TEXT[widget.currentText()]
        raise ValueError(f"unknown settings field kind {kind!r}")

    def _read_settings_widget_value(self, attr: str):
        """Read the plain value for one settings field from its widget."""
        kind = self._settings_kinds[attr]
        return self._widget_value(kind, self._settings_widgets[attr])

    def _sync_mosaic_settings(self, state: QtSettingsState) -> None:
        """Mirror the Mosaic sub-widgets into ``state.mosaic_settings``."""
        ms = state.mosaic_settings
        if not isinstance(ms, dict):
            ms = {}
            state.mosaic_settings = ms
        for key, (kind, widget) in self._mosaic_widgets.items():
            ms[key] = self._widget_value(kind, widget)

    def _update_expert_enabler_states(self, *_ignored) -> None:
        """Enable/disable Expert-tab sub-options from their enabler flags (Tk parity).

        Mirrors the Tk ``_update_*_options_state`` / ``_update_master_tile_crop_state``
        methods: an unchecked enabler checkbox disables (greys out) its gated
        sub-option widgets, and checking it re-enables them.  The enabler
        widgets themselves are never disabled by this routine.
        """
        for enabler, gated in EXPERT_ENABLER_GATES.items():
            enabler_widget = self._settings_widgets.get(enabler)
            if enabler_widget is None:
                continue
            checked = bool(enabler_widget.isChecked())
            for attr in gated:
                widget = self._settings_widgets.get(attr)
                if widget is not None:
                    widget.setEnabled(checked)

    def _reset_expert_settings(self) -> None:
        """Reset every Expert-tab setting to its ``QtSettingsState`` default.

        Mirrors the Tk ``reset_expert_settings`` button: it restores the BN /
        CB / master-tile-crop / final-crop / coverage-support /
        coverage-render / low-weight-mask / Photutils-BN widgets to defaults and
        re-applies the enabler gating.  This is display/settings-only: it
        mutates GUI state (widgets + the shared model), never writes FITS/PNG,
        never touches the engine or the settings file, and never touches
        ``_preview_source``.
        """
        defaults = QtSettingsState.defaults()
        for attr in EXPERT_RESET_ATTRS:
            widget = self._settings_widgets.get(attr)
            if widget is None:
                continue
            self._set_settings_widget_value(
                self._settings_kinds[attr], widget, defaults[attr]
            )
        self._update_expert_enabler_states()
        self._sync_state_from_controls()

    def _backend_notice_text(self) -> str:
        """Return the honest backend-mode notice for the current mode."""
        if self.backend_mode == "seestar":
            return self._tr("backend_notice_seestar", SEESTAR_BACKEND_NOTICE)
        return self._tr("backend_notice_simulated", SIMULATED_BACKEND_NOTICE)

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        self.statusBar().showMessage("Idle")

    def _wire_controls(self) -> None:
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self.start_button.clicked.connect(self._on_start)
        self.stop_button.clicked.connect(self._on_stop)
        self.analyse_button.clicked.connect(self._on_analyse)
        self.solver_button.clicked.connect(self._on_solver)
        self.view_inputs_button.clicked.connect(self._show_input_folder_list)
        self.add_folder_button.clicked.connect(self._add_folder)
        self.open_output_button.clicked.connect(self._open_output_folder)
        self.copy_log_button.clicked.connect(self._copy_log_to_clipboard)
        self.zoom_combo.currentIndexChanged.connect(self._on_zoom_changed)
        self.rotate_left_button.clicked.connect(self._on_rotate_left)
        self.rotate_right_button.clicked.connect(self._on_rotate_right)
        self.preview_res_button.clicked.connect(self._on_preview_res_cycle)
        # Mouse-wheel zoom + left-drag pan over the preview surface (M18).
        self.preview_image_label.wheelZoom.connect(self._on_wheel_zoom)
        self.preview_image_label.panDelta.connect(self._on_pan_delta)
        # WB gains / stretch black-white-gamma / B-C-S are wired through
        # ``_make_slider_spin_pair`` (slider <-> spinbox sync + a single
        # ``on_change`` callback), so only the discrete buttons are connected
        # here.
        self.auto_wb_button.clicked.connect(self._on_auto_wb)
        self.live_auto_wb_check.toggled.connect(self._set_live_auto_wb_enabled)
        self.wb_reset_button.clicked.connect(self._on_wb_reset)
        self.stretch_combo.currentIndexChanged.connect(self._on_stretch_changed)
        self.auto_stretch_button.clicked.connect(self._on_auto_stretch)
        self.live_auto_stretch_check.toggled.connect(
            self._set_live_auto_stretch_enabled
        )
        self.stretch_reset_button.clicked.connect(self._on_stretch_reset)
        self.bcs_reset_button.clicked.connect(self._on_bcs_reset)
        # Histogram interactions (M14): the persistent right-panel histogram
        # reproduces the Tk auto-zoom / reset / zoom / reset-zoom behaviours and
        # mirrors BP/WP line drags back into the stretch sliders.
        self.auto_zoom_histo_check.toggled.connect(self._on_hist_auto_zoom_toggled)
        self.hist_reset_view_button.clicked.connect(
            self._reset_histogram_view
        )
        self.hist_zoom_button.clicked.connect(
            self._zoom_histogram
        )
        self.hist_reset_button.clicked.connect(
            self._reset_histogram_zoom
        )
        self.hist_expand_button.clicked.connect(self._open_detached_histogram)
        self.right_histogram_view.rangeChanged.connect(self._on_hist_range_changed)
        self.right_histogram_view.expandRequested.connect(
            self._open_detached_histogram
        )
        # Histogram worker -> GUI thread: the coordinator lives on the GUI
        # thread and emits ``result_ready`` only for the latest (non-stale)
        # result; the queued connection guarantees ``_on_histogram_result`` runs
        # on the GUI thread (the only place widgets may be updated).
        self._histogram_coordinator.result_ready.connect(self._on_histogram_result)
        # Initial-preview auto-load delivery: the daemon worker thread emits
        # this signal; the explicit queued connection guarantees the slot runs
        # on the GUI thread even though the emitter is not a QThread.
        self._initial_preview_result.connect(
            self._on_initial_preview_result, Qt.ConnectionType.QueuedConnection
        )
        self._update_run_state()

    def _wire_controller(self) -> None:
        """Connect the lifecycle controller to this window's GUI-thread slots.

        All of these are queued signal deliveries from the worker thread
        (relayed by the controller), so every slot below runs on the GUI thread
        and is the only place widgets may be updated.
        """
        self.controller.started.connect(self._on_run_started)
        self.controller.progress_changed.connect(self._on_progress)
        self.controller.log_message.connect(self.log)
        self.controller.preview_updated.connect(self._on_preview)
        self.controller.summary_updated.connect(self._on_summary_payload)
        self.controller.finished.connect(self._on_run_finished)
        self.controller.failed.connect(self._on_run_failed)
        self.controller.cancelled.connect(self._on_run_cancelled)
        self.controller.refused.connect(self._on_run_refused)

    def _wire_settings_controls(self) -> None:
        """Mirror every settings widget into ``self.settings_state`` on change."""
        self.input_edit.textChanged.connect(self._sync_state_from_controls)
        self.output_edit.textChanged.connect(self._sync_state_from_controls)
        self.output_edit.textEdited.connect(self._invalidate_resume_on_user_path_change)
        self.temp_edit.textChanged.connect(self._sync_state_from_controls)
        self.output_filename_edit.textChanged.connect(self._sync_state_from_controls)
        self.reference_edit.textChanged.connect(self._sync_state_from_controls)
        self.reference_edit.textEdited.connect(self._on_reference_text_edited)
        self.last_stack_edit.textChanged.connect(self._sync_state_from_controls)
        self.last_stack_edit.textChanged.connect(self._on_last_stack_changed)
        self.last_stack_edit.textEdited.connect(self._invalidate_resume_on_user_path_change)
        self.resume_mode_combo.currentIndexChanged.connect(self._on_resume_mode_changed)
        self.browse_input_button.clicked.connect(self._browse_input)
        self.browse_output_button.clicked.connect(self._browse_output)
        self.browse_temp_button.clicked.connect(self._browse_temp)
        self.browse_reference_button.clicked.connect(self._browse_reference)
        self.browse_last_stack_button.clicked.connect(self._browse_last_stack)
        self.batch_spin.valueChanged.connect(self._sync_state_from_controls)
        self.batch_spin.valueChanged.connect(self._on_batch_size_changed)
        self.boring_check.stateChanged.connect(self._on_boring_check_changed)
        self.max_hq_mem_spin.valueChanged.connect(self._sync_state_from_controls)
        self.stacking_mode_combo.currentIndexChanged.connect(
            self._sync_state_from_controls
        )
        self.stacking_mode_combo.currentIndexChanged.connect(
            self._toggle_kappa_visibility
        )
        self.final_combine_combo.currentIndexChanged.connect(
            self._sync_state_from_controls
        )
        self.final_combine_combo.currentIndexChanged.connect(
            self._toggle_kappa_visibility
        )
        self.drizzle_check.stateChanged.connect(self._sync_state_from_controls)
        self.drizzle_check.stateChanged.connect(self._update_drizzle_gating)
        self.drizzle_mode_combo.currentIndexChanged.connect(
            self._sync_state_from_controls
        )
        self.drizzle_mode_combo.currentIndexChanged.connect(self._update_drizzle_gating)
        self.drizzle_group_spin.valueChanged.connect(self._sync_state_from_controls)
        self.drizzle_scale_spin.valueChanged.connect(self._sync_state_from_controls)
        self.drizzle_wht_spin.valueChanged.connect(self._sync_state_from_controls)
        self.drizzle_kernel_combo.currentIndexChanged.connect(
            self._sync_state_from_controls
        )
        self.drizzle_kernel_combo.currentIndexChanged.connect(
            self._update_drizzle_gating
        )
        self.drizzle_pixfrac_spin.valueChanged.connect(self._sync_state_from_controls)
        self.use_gpu_check.stateChanged.connect(self._sync_state_from_controls)
        self.use_gpu_check.toggled.connect(self._on_gpu_toggle)
        self.solver_combo.currentIndexChanged.connect(self._sync_state_from_controls)

        for widget in self._settings_widgets.values():
            self._connect_settings_widget(widget)
        for _kind, widget in self._mosaic_widgets.values():
            self._connect_settings_widget(widget)
        # Expert-tab enabler flags gate their sub-option widgets (Tk parity).
        for enabler in EXPERT_ENABLER_GATES:
            enabler_widget = self._settings_widgets.get(enabler)
            if enabler_widget is not None:
                enabler_widget.stateChanged.connect(
                    self._update_expert_enabler_states
                )
        self._update_expert_enabler_states()
        self._update_drizzle_gating()
        self._toggle_kappa_visibility()

    # ------------------------------------------------------------ controls
    def _on_start(self) -> None:
        if self._running:
            return
        state = self._effective_settings_state()
        zesolver_operational = False
        if (
            self.backend_mode == "seestar"
            and (state.reproject_between_batches or state.reproject_coadd_final)
            and str(state.local_solver_preference or "").strip().lower()
            == "zesolver"
        ):
            zesolver_operational = self.solver_probe()
        errors = validate_settings_for_backend(
            state,
            self.backend_mode,
            zesolver_operational=zesolver_operational,
        )
        if errors:
            self._on_preflight_failed(errors)
            return
        # Boring single-batch route (batch_size == 1): run boring_stack.py via
        # the injectable subprocess runner instead of the normal backend, so the
        # Qt shell never pretends to support batch_size == 1 while launching the
        # wrong (queue-manager) path.
        if state.batch_size == 1:
            self._start_boring_route(state)
            return
        request = self.build_run_request(
            initial_additional_folders=list(self._additional_folders)
        )
        backend = self.resolve_backend()
        # PHI-R3.2: allocate the durable producer run/session id for this run
        # and hand it to the real backend, which stamps it onto the stacker so
        # every active-producer payload carries PREV_RUN == the id the GUI
        # binds at run start.  Simulated/legacy backends (``backend is None``
        # or without the seam) stay unsequenced and keep the legacy fallback.
        if backend is not None and hasattr(backend, "set_preview_session"):
            self._next_preview_run_session += 1
            session = self._next_preview_run_session
            self._pending_preview_run_session = session
            backend.set_preview_session(session)
        else:
            self._pending_preview_run_session = None
        if backend is None:
            self.controller.start(request)
        else:
            self.controller.start(request, backend=backend)

    def _on_preflight_failed(self, errors: List[str]) -> None:
        """Report a preflight validation failure without starting a run.

        Runs on the GUI thread (invoked synchronously from ``_on_start``).  It
        leaves the window idle — ``is_running`` stays ``False`` and the
        start/stop buttons remain consistent — and surfaces the human-readable
        errors through the status bar and the log tab.  No ``RunController``
        call happens, so no ``failed`` signal is emitted.
        """
        self._report_preflight_failure("Cannot start real backend", errors)

    def _report_preflight_failure(self, prefix: str, errors: List[str]) -> None:
        """Surface a preflight failure (prefix + errors) and stay idle.

        The status bar and log keep the exact ``prefix + errors`` message; the
        owned failure box presents the same plain-text message as a *warning*
        (user-correctable), so a validation attempt yields exactly one box, not
        one per error string.  No ``RunController`` call happens.
        """
        message = prefix + ": " + "; ".join(errors)
        self.log(message)
        self.statusBar().showMessage(message)
        self._show_error_box(
            self._tr("error_box_preflight_title", default="Cannot start run"),
            message,
            severity="warning",
        )
        self._running = False
        self._update_run_state()

    # ------------------------------------------------------- boring route
    def _boring_preflight_errors(self, state: QtSettingsState) -> List[str]:
        """Return preflight errors for the boring (batch_size == 1) route."""
        errors: List[str] = []
        input_folder = str(state.input_folder or "").strip()
        if not input_folder:
            errors.append("Input folder is empty.")
        elif not os.path.isdir(input_folder):
            errors.append(f"Input folder does not exist: {input_folder}")
        if not str(state.output_folder or "").strip():
            errors.append("Output folder is empty.")
        return errors

    def _start_boring_route(self, state: QtSettingsState) -> None:
        """Preflight and launch the boring (single-batch CSV) subprocess route.

        Requires ``<input_folder>/stack_plan.csv`` to exist and parse to a
        non-empty list of existing FITS files.  On any failure it reports a
        clear preflight error and leaves the UI idle — it never calls
        :meth:`RunController.start` and never launches a runner.
        """
        errors = self._boring_preflight_errors(state)
        if errors:
            self._report_preflight_failure("Cannot start boring stack", errors)
            return

        csv_path = boring_route.csv_path_for(str(state.input_folder).strip())
        try:
            parsed = boring_route.parse_stack_plan_csv(csv_path)
        except boring_route.BoringCsvError as exc:
            self._report_preflight_failure("Cannot start boring stack", [exc.message])
            return

        request = boring_route.build_boring_request(
            csv_path=csv_path,
            output_dir=str(state.output_folder).strip(),
            batch_size=1,
            chunk_size=boring_route.get_auto_chunk_size(),
            normalize_method=str(state.stack_norm_method or "none"),
            save_final_as_float32=bool(state.save_final_as_float32),
            final_combine=str(state.stack_final_combine or "mean"),
            max_mem_gb=float(state.max_hq_mem_gb),
        )
        self.log(
            "Launching boring stack "
            f"({len(parsed.ordered_files)} files, "
            f"final_combine={request.final_combine}, batch_size=1)"
        )
        self._boring_active = True
        runner = self._resolve_boring_runner()
        runner.start(request)

    def _resolve_boring_runner(self) -> BoringRunnerBase:
        """Return the (lazily created, signal-wired) boring subprocess runner."""
        if self._boring_runner is None:
            if self.boring_runner_factory is not None:
                self._boring_runner = self.boring_runner_factory()
            else:
                self._boring_runner = QProcessBoringRunner(self)
            self._boring_runner.started.connect(self._on_boring_started)
            self._boring_runner.finished.connect(self._on_boring_finished)
            self._boring_runner.failed.connect(self._on_boring_failed)
            self._boring_runner.cancelled.connect(self._on_boring_cancelled)
            self._boring_runner.log_message.connect(self.log)
            self._boring_runner.summary.connect(self._on_summary_payload)
        return self._boring_runner

    # ------------------------------------------------ boring lifecycle slots
    def _on_boring_started(self) -> None:
        self._running = True
        self._run_started_at = self._now()
        self._update_run_state()
        self.statusBar().showMessage("Boring stack running…")
        # Boring has no percent progress, so elapsed starts at 0 and remaining
        # stays honestly unknown for the whole run.
        self._refresh_time_surface(None)

    def _on_boring_finished(self, exit_code: int) -> None:
        self._boring_active = False
        self._running = False
        self._update_run_state()
        self.progress.setValue(100)
        if derive_terminal_status(self._last_summary_payload) == "success":
            self.statusBar().showMessage("Boring stack finished.")
            self.log("Boring stack finished.")
        else:
            message = self._tr(
                "boring_finished_empty",
                default="Boring stack finished with no output.",
            )
            self.statusBar().showMessage(message)
            self.log(message)
        self._mark_time_terminal("0:00")
        self._show_pending_summary()

    def _on_boring_failed(self, message: str) -> None:
        self._boring_active = False
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage(f"Boring stack failed: {message}")
        self.log(f"Boring stack failed: {message}")
        self._show_error_box(
            self._tr("error_box_boring_failed_title", default="Boring stack failed"),
            message,
            severity="critical",
        )
        self._mark_time_terminal("failed")

    def _on_boring_cancelled(self) -> None:
        self._boring_active = False
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage("Boring stack cancelled.")
        self.log("Boring stack cancelled.")
        self._mark_time_terminal("cancelled")

    def _on_stop(self) -> None:
        if not self._running:
            return
        if self._boring_active:
            runner = self._boring_runner
            if runner is not None:
                runner.cancel()
            self.statusBar().showMessage("Cancelling boring stack…")
            self.log("Stop requested (boring stack).")
            return
        self.controller.cancel()
        self.statusBar().showMessage("Cancelling…")
        self.log("Stop requested.")

    # ------------------------------------------------ boring mode sync / gating
    def _on_boring_check_changed(self, checked: int) -> None:
        """Synchronise the batch spinbox with the boring-thread checkbox.

        Tk ``_toggle_boring_thread`` parity: checking the box forces
        ``batch_size == 1`` and locks the spinbox; unchecking unlocks it and, if
        it was pinned at 1, resets it to 0 (Auto).
        """
        if bool(checked):
            if self.batch_spin.value() != 1:
                self.batch_spin.setValue(1)
            self.batch_spin.setEnabled(False)
        else:
            self.batch_spin.setEnabled(True)
            if self.batch_spin.value() == 1:
                self.batch_spin.setValue(0)
        self._update_boring_gating()

    def _on_batch_size_changed(self, value: int) -> None:
        """Synchronise the boring checkbox with the batch size (Tk parity).

        ``batch_size == 1`` checks the box; anything else unchecks it.  The
        guard prevents a ping-pong with :meth:`_on_boring_check_changed`.
        """
        if value == 1 and not self.boring_check.isChecked():
            self.boring_check.setChecked(True)
        elif value != 1 and self.boring_check.isChecked():
            self.boring_check.setChecked(False)
        self._update_boring_gating()

    def _update_boring_gating(self) -> None:
        """Gate controls incompatible with boring mode (honest interdependency).

        Boring mode forces ``use_drizzle=False`` (the boring CLI hardcodes it),
        so the drizzle controls are disabled and unchecked while boring mode is
        active.  Stacking mode / final-combine remain available: the boring CLI
        forces winsorized-sigma stacking but genuinely supports the
        ``--final-combine`` override, matching the Tk subprocess path.
        """
        boring = self.boring_check.isChecked()
        self.drizzle_check.setEnabled(not boring)
        if boring and self.drizzle_check.isChecked():
            self.drizzle_check.setChecked(False)
        # Re-apply the drizzle sub-option gating (Tk ``_update_drizzle_options_state``)
        # so the mode/group/scale/WHT/kernel/pixfrac/GPU controls reflect the
        # combined boring + enable-drizzle state.
        self._update_drizzle_gating()

    # ----------------------------------------------------------- GPU (M5)
    def _refresh_gpu_check_enabled(self) -> None:
        """Enable the Use GPU checkbox only when a usable backend was probed.

        Capability-driven (M5): the checkbox no longer follows the drizzle
        flag; it reflects ``backend_ready`` from the probed capabilities.
        Never probes by itself — the deferred ``_refresh_gpu_status`` owns
        probing so startup stays unblocked.
        """
        caps = getattr(self, "_gpu_capabilities", None)
        ready = bool(caps is not None and getattr(caps, "backend_ready", False))
        self.use_gpu_check.setEnabled(ready)

    def _refresh_gpu_status(self, *_ignored) -> None:
        """Render capability + enablement, probing if nothing is cached yet.

        Production startup uses :meth:`_start_gpu_probe` so the (cold) probe
        runs on a worker thread; this synchronous variant exists for
        deterministic tests and for re-renders once capabilities are cached.
        """
        caps = getattr(self, "_gpu_capabilities", None)
        if caps is None:
            caps = gpu_bridge.probe_gpu()
            self._gpu_capabilities = caps
        self._render_gpu_state(caps)

    def _start_gpu_probe(self) -> None:
        """Start the off-main-thread GPU probe (F3).

        The cold probe (engine import + CuPy JIT + nvidia-smi) runs inside a
        :class:`~seestar.gui_qt.gpu_bridge.GpuProbeWorker`; the result is
        delivered back to this (GUI) thread via the queued ``resultReady``
        signal.  The worker reference is kept on the window so it can never be
        garbage-collected mid-probe, and the checkbox stays disabled with the
        "probing…" placeholder until the result arrives.
        """
        worker = getattr(self, "_gpu_probe_worker", None)
        if worker is not None and worker.isRunning():
            return  # a probe is already in flight
        if self._gpu_capabilities is not None:
            # Probe already completed (cache warm): nothing to do.
            return
        worker = gpu_bridge.GpuProbeWorker(parent=self)
        worker.resultReady.connect(self._on_gpu_probe_result)
        self._gpu_probe_worker = worker
        worker.start()

    def _on_gpu_probe_result(self, caps) -> None:
        """Main-thread slot: apply the worker's probe result to the UI."""
        self._gpu_capabilities = caps
        self._render_gpu_state(caps)

    def _render_gpu_state(self, caps) -> None:
        """Render the capability line + checkbox enablement for ``caps``."""
        self.gpu_status_label.setText(
            gpu_bridge.describe_policy(
                caps, request_gpu=self.use_gpu_check.isChecked()
            )
        )
        self._refresh_gpu_check_enabled()

    def _stop_gpu_probe(self, wait_ms: Optional[int]) -> bool:
        """Wait for the probe worker to finish; True when fully stopped."""
        worker = getattr(self, "_gpu_probe_worker", None)
        if worker is None:
            return True
        if not worker.isRunning():
            self._gpu_probe_worker = None
            return True
        worker.requestInterruption()
        if not worker.wait(wait_ms):
            return False
        self._gpu_probe_worker = None
        return True

    def _on_gpu_toggle(self, *_ignored) -> None:
        """Re-render the resolved-state line when the checkbox toggles.

        Rendering only (never probes): the checkbox can only be toggled once
        enabled, which implies the probe already completed.
        """
        caps = getattr(self, "_gpu_capabilities", None)
        if caps is not None:
            self.gpu_status_label.setText(
                gpu_bridge.describe_policy(
                    caps, request_gpu=self.use_gpu_check.isChecked()
                )
            )

    def _update_drizzle_gating(self) -> None:
        """Gate drizzle sub-options from the Enable-drizzle flag (Tk parity).

        Mirrors the Tk ``_update_drizzle_options_state`` method: when drizzle is
        disabled (or boring mode forces it off) the mode combo and the scale/WHT/
        kernel/pixfrac sub-options are disabled; the group-size spinbox is
        additionally enabled only in the Large-dataset (``Incremental``) mode,
        exactly like the Tk M3-D policy (``drizzle_group_size`` depends on
        the Large-dataset policy, not the global drizzle flag alone).  The GPU
        toggle is NOT drizzle-gated anymore (M5): its enablement is driven by
        the probed GPU capability (see ``_refresh_gpu_check_enabled``).
        """
        boring = self.boring_check.isChecked()
        drizzle = self.drizzle_check.isChecked() and not boring

        self.drizzle_mode_combo.setEnabled(drizzle)

        # M3-D: group size is only relevant in the Large-dataset (Incremental)
        # policy; Standard keeps the same science with no grouped preview.
        group = drizzle and self.drizzle_mode_combo.currentData() == "Incremental"
        self.drizzle_group_spin.setEnabled(group)

        # M5: the GPU toggle is capability-driven, not drizzle-gated.
        self._refresh_gpu_check_enabled()

        # Drizzle advanced sub-options (Stacking-tab Drizzle block, D3) share
        # the same global Enable-drizzle gate (Tk parity).
        self.drizzle_scale_spin.setEnabled(drizzle)
        signed_wht = self.drizzle_kernel_combo.currentText() in (
            "lanczos2",
            "lanczos3",
        )
        self.drizzle_wht_spin.setEnabled(drizzle and not signed_wht)
        self.drizzle_wht_spin.setToolTip(
            self._tr("drizzle_wht_threshold_signed_tooltip") if signed_wht else ""
        )
        self.drizzle_kernel_combo.setEnabled(drizzle)
        self.drizzle_pixfrac_spin.setEnabled(drizzle)

    def _toggle_kappa_visibility(self, *_ignored) -> None:
        """Show/hide the Kappa Low/High + Winsor-Limits widgets (Tk parity, M17).

        Mirrors the Tk ``_toggle_kappa_visibility``: the Kappa Low/High controls
        are shown when the stacking method is ``kappa-sigma`` /
        ``winsorized-sigma-clip`` or the final-combine is
        ``winsorized_sigma_clip``; the Winsor-Limits control is shown when the
        stacking method or the final-combine is winsorized-sigma.  Purely
        cosmetic: the widgets' values stay in the shared settings model and
        ``build_backend_kwargs`` always passes them (they are never removed).
        The standalone "Kappa" field (backend ``kappa``) is *not* part of the
        Tk kappa frame and stays always visible.
        """
        method = self.stacking_mode_combo.currentText()
        final_key = self._final_combine_key()
        show_kappa = (
            method in ("kappa-sigma", "winsorized-sigma-clip")
            or final_key == "winsorized_sigma_clip"
        )
        show_winsor = (
            method == "winsorized-sigma-clip"
            or final_key == "winsorized_sigma_clip"
        )
        for attr, visible in (
            ("stack_kappa_low", show_kappa),
            ("stack_kappa_high", show_kappa),
            ("stack_winsor_limits", show_winsor),
        ):
            widget = self._settings_widgets.get(attr)
            form = self._settings_forms.get(attr)
            if widget is not None and form is not None:
                form.setRowVisible(widget, visible)

    def _on_solver(self) -> None:
        """Open the solver configuration dialog and apply accepted values.

        This never starts the backend.  The dialog is a pure view/controller
        over the existing solver boundaries (reached lazily), and accepted
        values are written back into the live Qt controls so a subsequent
        ``collect_settings_state()`` / ``build_run_request()`` sees them.

        On accept only, the values are also bridged into the engine solver
        config (the module defining ``load_config`` / ``save_config``) via a
        lazy, function-scoped import — mirroring the Tk ``_on_ok`` timing,
        which writes the engine config only when OK is pressed.  Cancel/ESC
        never reaches this branch.
        """
        dialog = SolverSettingsDialog(self, self.collect_settings_state())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            values = dialog.values()
            self._apply_solver_dialog_values(values)
            # M21: persist the accepted solver fields into the engine solver
            # config (the same file the Tk GUI writes) with Tk-identical
            # load/overlay/save semantics.  Defensive: a write failure never
            # blocks the dialog's OK flow (Tk parity).
            try:
                write_solver_config(values)
            except Exception:
                pass

    def _apply_solver_dialog_values(self, values: dict) -> None:
        """Apply accepted solver-dialog values back to the live Qt controls.

        Setting the widgets triggers the already-wired change signals, which
        keep ``settings_state`` in sync (the single source of truth), so the
        values survive the next ``collect_settings_state()``.
        """
        pref = str(values.get("local_solver_preference", "none"))
        if pref in SOLVER_PREFERENCES:
            self.solver_combo.setCurrentText(pref)
        for attr in (
            "astap_path",
            "astap_data_dir",
            "astap_search_radius",
            "astap_downsample",
            "astap_sensitivity",
        ):
            if attr not in values or attr not in self._settings_widgets:
                continue
            widget = self._settings_widgets[attr]
            value = values[attr]
            if attr in ("astap_path", "astap_data_dir"):
                widget.setText(str(value))
            elif attr == "astap_search_radius":
                widget.setValue(float(value))
            else:
                widget.setValue(int(value))

    # ------------------------------------------------ ZeAnalyser launch seam
    def _current_language_code(self) -> str:
        """Return the ZeAnalyser ``--lang`` code for the current combo label."""
        return LANGUAGE_CODE_BY_TEXT.get(self.language_combo.currentText(), "en")

    def _on_analyse(self) -> None:
        """Launch standalone ZeAnalyser on the current input folder (M7).

        This is a user-triggered, non-blocking launch of an external product.
        It never marks a run active and never touches the stacking backend.  It
        validates the input folder, detects ZeAnalyser, creates the command-file
        path, and spawns the process with ``ZEANALYSER_COMMAND_FILE`` set.  Any
        failure is reported through the log + status bar without raising.
        """
        input_folder = self.input_edit.text().strip()
        if not input_folder or not os.path.isdir(input_folder):
            message = "Analyze: select a valid input folder first."
            self.log(message)
            self.statusBar().showMessage(message)
            return

        lang = self._current_language_code()
        try:
            command_file_path = self._analyzer_command_file_maker()
        except Exception as exc:  # noqa: BLE001 - surface any path-creation failure
            message = f"Analyze: cannot create command file: {exc}"
            self.log(message)
            self.statusBar().showMessage(message)
            return

        self._analyzer_command_file_path = command_file_path
        try:
            launched = self._analyzer_launcher(input_folder, lang, command_file_path)
        except Exception as exc:  # noqa: BLE001 - surface any launch failure
            message = f"Analyze: launch failed: {exc}"
            self.log(message)
            self.statusBar().showMessage(message)
            return

        if not launched:
            message = (
                "Analyze: ZeAnalyser not found (install the 'zeanalyser' command "
                "or the 'zeanalyser' Python module)."
            )
            self.log(message)
            self.statusBar().showMessage(message)
            return

        self.log(f"Analyzer launched on {input_folder}.")
        self.statusBar().showMessage("Analyzer launched.")
        # Arm the ZeAnalyser reference-return watcher (Tk parity): the Tk shell
        # starts its ``after(1000, ...)`` surveillance loop right after the
        # non-blocking launch, so a late ``REFERENCE=`` return is consumed and
        # acted upon without manual action.  The timer is parented to ``self``
        # (Qt destroys it with the window) and stopped by :meth:`shutdown`.
        self._analyzer_watch_timer.start()

    def _check_analyzer_command_file(self) -> Optional[str]:
        """Consume the ZeAnalyser command file once, if present (Qt-safe).

        Reads ``REFERENCE=<path>``, updates the reference field only for a
        non-empty reference, deletes the file best-effort, and returns the
        reference (or ``None``).  The periodic watcher (``_analyzer_watch_tick``
        on a GUI-thread ``QTimer``) reuses this single-shot consumption seam, so
        it is also safe to call directly from the GUI thread or from tests.
        """
        path = self._analyzer_command_file_path
        if not path or not os.path.exists(path):
            return None
        try:
            ref = analyzer_launch.consume_command_file(path)
        except OSError:
            return None
        if ref:
            self._reference_origin_hint = "ZEANALYSER_V1"
            self.reference_edit.setText(ref)
            self.log(f"Analyzer reference received: {ref}")
            self.statusBar().showMessage(f"Analyzer reference: {ref}")
        return ref

    def _analyzer_watch_tick(self) -> None:
        """One tick of the ZeAnalyser reference-return watcher (M25.5-A).

        Mirrors the Tk ``_check_analyzer_command_file`` surveillance loop on a
        GUI-thread ``QTimer``: one command-file check per tick, never a busy
        loop and never a worker thread.  It stops the watcher when the Tk loop
        would stop (no command-file path, or a run already active), applies the
        historical consequences when a reference arrives, and keeps polling
        otherwise.
        """
        # Tk stop conditions, checked first: no command-file path to watch, or
        # a run is already active (``self.processing`` parity).
        if not self._analyzer_command_file_path or self._running:
            self._analyzer_watch_timer.stop()
            return
        ref = self._check_analyzer_command_file()
        if ref is None:
            # No reference yet (file absent or consumed without REFERENCE=):
            # keep watching, exactly like Tk re-arming ``after(1000, ...)``.
            return
        # Reference arrived: apply the Tk-side consequences.  The reference
        # field was already updated by ``_check_analyzer_command_file``.
        if self._apply_analyzer_reference(ref):
            # Settled (consequences applied / processing started): stop.
            self._analyzer_watch_timer.stop()

    def _apply_analyzer_reference(self, ref: str) -> bool:
        """Apply the Tk-side consequences of a returned ZeAnalyser reference.

        Matches Tk ``_check_analyzer_command_file`` exactly: re-sync the input
        folder (reloading the first image when it differs), prepare a default
        output folder when none is set, and start processing.  Returns ``True``
        when the consequences were applied (the watcher should stop) and
        ``False`` when the input folder became invalid (the watcher keeps
        polling, matching Tk's re-arm branch).
        """
        current_input = self.input_edit.text().strip()
        analyzed_folder = os.path.abspath(current_input) if current_input else None

        if not (analyzed_folder and os.path.isdir(analyzed_folder)):
            self.log(
                f"Analyzer reference received but input folder invalid: "
                f"{analyzed_folder!r}; keeping watcher active."
            )
            return False

        # Tk: re-sync the input field if it differs and reload the first image.
        if os.path.normpath(self.input_edit.text()) != os.path.normpath(
            analyzed_folder
        ):
            self.input_edit.setText(analyzed_folder)
            self._try_show_first_input_image()

        # Tk: prepare a default output folder when none is set.
        if not self.output_edit.text().strip():
            default_output = os.path.join(analyzed_folder, "stack_output_analyzer")
            self.output_edit.setText(default_output)

        self.log(f"Analyzer reference received, starting processing: {ref}")
        # Tk calls ``start_processing()`` here; the Qt equivalent is the Start
        # handler (validation/preflight included, never raises).
        self._on_start()
        return True

    # ------------------------------------------------------ path / file actions
    def _browse_input(self) -> None:
        """Select the input folder via a directory dialog (Tk parity)."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", self.input_edit.text().strip()
        )
        if folder:
            self.input_edit.setText(os.path.abspath(folder))
            # Tk parity: picking an input folder auto-loads its first FITS for
            # an immediate preview.
            self._try_show_first_input_image()

    def _browse_output(self) -> None:
        """Select the output folder via a directory dialog (Tk parity)."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.output_edit.text().strip()
        )
        if folder:
            self.output_edit.setText(os.path.abspath(folder))
            # ``setText`` never emits ``textEdited``, so invalidate an armed
            # Resume explicitly (a freshly browsed output is a new target).
            self._invalidate_resume_on_user_path_change()

    def _browse_temp(self) -> None:
        """Select the temporary folder via a directory dialog (Tk parity)."""
        start_dir = self.temp_edit.text().strip() or self.output_edit.text().strip()
        folder = QFileDialog.getExistingDirectory(
            self, "Select Temporary Folder", start_dir
        )
        if folder:
            self.temp_edit.setText(os.path.abspath(folder))

    def _reference_start_dir(self) -> str:
        """Return a sensible start directory for the reference file dialog."""
        current = self.reference_edit.text().strip()
        if current and os.path.isfile(current):
            return os.path.dirname(current)
        input_folder = self.input_edit.text().strip()
        if input_folder and os.path.isdir(input_folder):
            return input_folder
        return "."

    def _on_reference_text_edited(self, text: str) -> None:
        """Mark a directly edited explicit reference as user-provided."""
        self._reference_origin_hint = "USER" if str(text).strip() else None

    def _browse_reference(self) -> None:
        """Select the reference image via a FITS file dialog (Tk parity)."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Image (Optional)",
            self._reference_start_dir(),
            FITS_FILE_FILTER,
        )
        if filepath:
            self._reference_origin_hint = "USER"
            self.reference_edit.setText(os.path.abspath(filepath))

    def _browse_last_stack(self) -> None:
        """Select the previous stack via a FITS file dialog (Tk parity)."""
        start_dir = (
            self.output_edit.text().strip()
            or self.last_stack_edit.text().strip()
            or "."
        )
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select previous stack", start_dir, FITS_FILE_FILTER
        )
        if filepath:
            abs_path = os.path.abspath(filepath)
            self.last_stack_edit.setText(abs_path)
            # ``setText`` never emits ``textEdited``, so invalidate an armed
            # Resume explicitly (a freshly browsed stack is a new target).
            self._invalidate_resume_on_user_path_change()

    def _on_last_stack_changed(self, *_ignored) -> None:
        """Pre-fill the output folder from the last-stack path when empty.

        Tk ``_on_last_stack_changed`` parity: any last-stack change (browse,
        manual edit, persisted-load) pre-fills the output folder with the
        last-stack file's parent directory — but only when the output folder is
        currently empty.  The guard mirrors Tk ``if not output_path.get()``
        exactly, so an already-set output folder is never clobbered.
        """
        if self.output_edit.text().strip():
            return
        p = self.last_stack_edit.text().strip()
        if p:
            self.output_edit.setText(os.path.dirname(p))

    # ------------------------------------------------------ resume selector
    def _on_resume_mode_changed(self, _index: Optional[int] = None) -> None:
        """Handle an explicit New/Resume selector change (RSM2-02C).

        ``Resume`` runs the bounded locator/restore flow; ``New`` (or any
        non-resume value) clears the transient run intent only.  This is the
        *only* path that sets ``resume_intent`` — editing/browsing the Last
        Stack path never does.
        """
        mode = self.resume_mode_combo.currentData()
        if mode == "resume":
            self._activate_resume()
        else:
            self._clear_resume_intent()

    def _clear_resume_intent(self) -> None:
        """Clear the transient resume intent/source (keep last-stack history)."""
        self.settings_state.resume_intent = RUN_INTENT_FRESH
        self.settings_state.resume_source = ""

    def _invalidate_resume_on_user_path_change(self, *_ignored) -> None:
        """Invalidate an armed Resume on a user-originated path change.

        Editing or browsing Last Stack / Output after a Resume has been
        prepared would otherwise leave ``resume_intent`` / ``resume_source``
        pointing at the old run while the request carries the new path — an
        incoherent source/target pairing.  Any such user-originated change
        reverts the selector to New and clears the transient intent/source,
        while keeping the newly entered/browsed path and history; the user may
        explicitly select Resume again to re-run discovery.  Programmatic
        updates during ``_apply_resume_result`` are guarded, and a fresh
        (never-armed) window is a no-op.
        """
        if self._applying_resume_result:
            return
        if self.settings_state.resume_intent != RUN_INTENT_RESUME:
            return
        self._set_resume_mode_combo("fresh")
        self._clear_resume_intent()

    def _set_resume_mode_combo(self, mode: str) -> None:
        """Set the selector to ``mode`` without re-firing the handler."""
        index = 1 if mode == "resume" else 0
        self.resume_mode_combo.blockSignals(True)
        try:
            self.resume_mode_combo.setCurrentIndex(index)
        finally:
            self.resume_mode_combo.blockSignals(False)

    def _activate_resume(self) -> None:
        """Run the explicit Resume flow from the current locator.

        The locator is the selected previous-stack FITS (last-stack path) or,
        when empty, the output folder.  A missing locator prompts a browse.  On
        success the owning run directory's config is restored and the transient
        resume intent/source are set; on failure the selector reverts to New and
        a bounded warning is shown (never a hidden Resume intent).
        """
        locator = self.last_stack_edit.text().strip() or self.output_edit.text().strip()
        if not locator:
            self._browse_last_stack()
            locator = self.last_stack_edit.text().strip()
        result = resume_locator.discover_resume(locator)
        if result.status == resume_locator.STATUS_READY:
            self._apply_resume_result(result)
        else:
            self._refuse_resume(result)

    def _apply_resume_result(self, result) -> None:
        """Restore the discovered config and arm an explicit Resume run intent."""
        state = self.settings_state
        resume_locator.restore_to_settings(
            result.config, state, checkpoint_kind=result.checkpoint_kind
        )
        # Coherent output folder: the resolved owning run directory.
        state.output_folder = result.run_dir
        state.resume_intent = RUN_INTENT_RESUME
        state.resume_source = result.run_dir
        self._applying_resume_result = True
        try:
            self._apply_state_to_controls(state)
        finally:
            self._applying_resume_result = False
        # The trailing control sync never touches resume_intent/resume_source,
        # but re-assert them defensively so the model stays coherent.
        self.settings_state.resume_intent = RUN_INTENT_RESUME
        self.settings_state.resume_source = result.run_dir
        self.log(
            f"Resume prepared from {result.run_dir} "
            f"(config: {result.config_source or 'none'})"
        )
        self.statusBar().showMessage(f"Resume: {result.run_dir}")

    def _refuse_resume(self, result) -> None:
        """Refuse an invalid Resume and leave the window Fresh (no mutation)."""
        self._set_resume_mode_combo("fresh")
        self._clear_resume_intent()
        reason_key = resume_locator.STATUS_REASON_KEYS.get(
            result.status, "resume_refuse_no_run"
        )
        body = self._tr(reason_key, default=reason_key)
        if result.detail:
            body = f"{body}\n\n{result.detail}"
        self.log(
            f"Resume refused: {reason_key}"
            + (f" — {result.detail}" if result.detail else "")
        )
        self.statusBar().showMessage(self._tr("resume_refuse_title"))
        self._show_error_box(
            self._tr("resume_refuse_title", default="Cannot resume"),
            body,
            severity="warning",
        )

    def _input_folder_summary_text(self) -> str:
        """Return the human-readable input-folder summary (main + staged)."""
        main_folder = self.input_edit.text().strip()
        if not main_folder or not os.path.isdir(main_folder):
            return ""
        lines = [os.path.abspath(main_folder)]
        for folder in self._additional_folders:
            abs_path = os.path.abspath(folder)
            if abs_path not in lines:
                lines.append(abs_path)
        return "\n".join(lines)

    def _build_input_folder_dialog(self) -> QDialog:
        """Build the non-backend View Inputs dialog (read-only folder list)."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Input Folders")
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setText(self._input_folder_summary_text())
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        return dialog

    def _show_input_folder_list(self) -> None:
        """Show a non-backend dialog listing main + staged input folders."""
        if not self._input_folder_summary_text():
            message = "No valid input folder set."
            self.log(message)
            self.statusBar().showMessage(message)
            return
        self._build_input_folder_dialog().exec()

    def _open_output_folder(self) -> None:
        """Open the output folder via the desktop service (safe, user-triggered).

        Never raises and never crashes on a missing/invalid path: it logs a
        clear message and leaves the UI idle.
        """
        output_folder = self.output_edit.text().strip()
        if not output_folder:
            message = "Open Output: no output folder set."
            self.log(message)
            self.statusBar().showMessage(message)
            return
        if not os.path.isdir(output_folder):
            message = f"Open Output: output folder does not exist: {output_folder}"
            self.log(message)
            self.statusBar().showMessage(message)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(output_folder))
        self.log(f"Opening output folder: {output_folder}")

    def _validate_additional_folder(self, abs_folder: str) -> Optional[str]:
        """Return an error message, or ``None`` when the folder is acceptable.

        Rejects the main input folder, the output folder, and any subfolder of
        the output folder (Tk parity).  Missing paths are rejected upstream.
        """
        if not os.path.isdir(abs_folder):
            return "Folder not found."
        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()
        abs_input = os.path.abspath(input_text) if input_text else None
        abs_output = os.path.abspath(output_text) if output_text else None
        if abs_input and os.path.normcase(abs_folder) == os.path.normcase(abs_input):
            return "Input folder cannot be added."
        if abs_output:
            if os.path.normcase(abs_folder) == os.path.normcase(abs_output):
                return "Output folder cannot be added."
            if os.path.normcase(abs_folder).startswith(
                os.path.normcase(abs_output) + os.sep
            ):
                return "Cannot add subfolder of output folder."
        return None

    def _add_folder(self) -> None:
        """Stage an additional folder for the next run (Tk ``add_folder`` parity).

        Live-add during an active run is not exposed by ``RunController``, so
        the action is disabled while running; this guard is defensive.
        """
        if self._running:
            message = "Add Folder: live add is not implemented; stop the run first."
            self.log(message)
            self.statusBar().showMessage(message)
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Select Additional Images Folder", self.input_edit.text().strip()
        )
        if not folder:
            return
        abs_folder = os.path.abspath(folder)
        error = self._validate_additional_folder(abs_folder)
        if error:
            self.log(f"Add Folder rejected: {error}")
            self.statusBar().showMessage(error)
            return
        if abs_folder not in self._additional_folders:
            self._additional_folders.append(abs_folder)
            self.log(f"Folder added for next run: {os.path.basename(abs_folder)}")
            self.statusBar().showMessage(f"Added folder: {abs_folder}")
        else:
            message = "Folder already added to the list."
            self.log(message)
            self.statusBar().showMessage(message)

    # ------------------------------------------------- lifecycle callbacks
    def _on_run_started(self) -> None:
        self._running = True
        self._run_started_at = self._now()
        # New run, new scientific-preview identity domain: bump the run/context
        # identity and reset all per-run live-auto dedupe/instrumentation state.
        # Live-auto enablement itself is user intent and deliberately survives
        # between runs; only dedupe state and per-run instrumentation reset here.
        self._run_context_id += 1
        # A new run is a fresh producer sequence domain (each run constructs a
        # fresh stacker whose PREV_SEQ counter restarts at 1): reset the PHI-R3
        # monotonic acceptance gate so the first sequenced payload of the new
        # run is accepted, never rejected as stale/duplicate.  PHI-R3.2: bind
        # the durable producer run/session identity allocated at Start (the id
        # handed to the real backend, which stamps it onto the stacker); a
        # payload carrying any other session id is foreign and dropped.
        self._preview_run_session = self._pending_preview_run_session
        self._pending_preview_run_session = None
        self._last_accepted_preview_seq = None
        self._last_live_auto_batch_token = None
        self._live_auto_stretch_count = 0
        self._live_auto_wb_count = 0
        self._live_bp = None
        self._live_wp = None
        # A new run is a fresh display-anchor context: drop any anchors / float
        # analysis buffers from a previous run (never reused across runs).
        self._reset_preview_analysis()
        # STABLE-B: a new run is also a fresh view-state context.  Reset the
        # accumulated rotation, continuous zoom and pan offsets exactly once so
        # the next run's first preview starts at the defaults (0°, 100%,
        # centred) instead of inheriting the previous run's view state.
        self._preview_rotation = 0
        self._reset_view_transform()
        # STABLE-B-R1: make the reset atomic from the user's perspective.  A
        # retained valid preview must agree with the reset view state (0°,
        # 100%, centred) on screen immediately, before the next backend
        # preview arrives (which can take significant time).  Reconcile the
        # displayed pixmap + resolution label now.  This is a pure view
        # reconciliation (``histogram=False``): it never touches the
        # scientific identity / ``raw_revision``, never mutates the payload /
        # science, and schedules zero histogram compute for the unchanged
        # source.  With no retained preview there is nothing to reconcile and
        # the controls stay correctly disabled.
        if self.has_preview_image:
            self._refresh_preview_view(histogram=False)
        # A second run must not reuse the previous run's terminal progress:
        # reset the bar (and the elapsed/remaining surface) before the first
        # progress signal of the new run arrives.
        self.progress.setValue(0)
        self._update_run_state()
        self.statusBar().showMessage("Running…")
        # Elapsed starts at 0 and remaining is unknown until the first
        # progress signal carries a usable percent (0 < percent < 100).
        self._refresh_time_surface(None)

    def _on_progress(self, percent: int) -> None:
        self.progress.setValue(int(percent))
        self._refresh_time_surface(int(percent))

    def _on_preview(self, payload: BackendPreviewPayload) -> None:
        """Update the Preview tab label from a preview payload (GUI thread only).

        Monotonic acceptance gate (PHI-R3.2): every payload that carries a
        producer ``PREV_SEQ`` (the active Classic/Drizzle producers emit
        ``PREV_SEQ`` + ``PREV_RUN`` on **every** payload — required display
        metadata, independent of the ``ZSSS_PHI_TRACE`` debug gate) is
        accepted only when (a) its producer run/session id matches the id bound
        to the current run at start (:meth:`_on_run_started` — a late payload
        of a previous producer session is *foreign* and dropped without
        touching the sequence high-water mark) and (b) its sequence strictly
        advances the run's monotonic gate — a stale (older) or duplicate
        (equal) emission is dropped *before* it can replace analysis/display
        state or schedule any work.  Payloads without ``PREV_SEQ``
        (legacy/unsequenced producers, initial-preview loads, test payloads)
        bypass the gate and keep the historical unconditional last-wins
        acceptance, so old producers and callers behave exactly as before.
        Sequenced payloads without ``PREV_RUN`` (third-party/synthetic) fall
        back to the run-scoped monotonic gate only.  The gate resets at run
        start, so the first sequenced payload of a new run is never rejected.

        For accepted payloads, the metadata label is updated (stack name and
        counts).  Additionally, when ``payload.data`` is image-like, it is
        converted (strictly display-only, via :func:`preview_render.render_preview_image`)
        and kept as the copied source image for the view transforms (zoom /
        rotation / resolution).  A *valid* preview replaces only the content:
        the user's accumulated rotation, continuous zoom and pan offsets are
        preserved across successive scientific previews (content freshness and
        view state are independent).  Invalid/missing data never raises and
        clears the stored source, the image area, the rotation state and the
        view controls, so no stale preview survives a failed render.
        """
        # PHI-R3.2 monotonic acceptance gate: reject stale/duplicate/foreign
        # producer emissions before any label/analysis/display/work state
        # changes.  The gate applies to every payload that carries a producer
        # ``PREV_SEQ`` (the active producers emit ``PREV_SEQ`` + ``PREV_RUN``
        # on every payload, trace gate irrelevant).  A payload whose producer
        # run/session id does not match the id bound to the current run at
        # start is *foreign* (a late payload of a previous producer session)
        # and is dropped without touching the sequence high-water mark.
        # Unsequenced (legacy) payloads are always accepted.
        producer_seq = self._payload_preview_seq(payload)
        if producer_seq is not None and not self._accept_preview_payload(
            self._payload_preview_run(payload), producer_seq
        ):
            if phi_trace_enabled():
                run = self._payload_preview_run(payload)
                if run is None or run == self._preview_run_session:
                    # Same (or unverifiable) producer session: the refusal is
                    # monotonic — stale (older) or duplicate (equal).  A
                    # refusal implies the high-water mark is already set.
                    reason = (
                        "stale"
                        if producer_seq < self._last_accepted_preview_seq
                        else "duplicate"
                    )
                else:
                    reason = "foreign"
                self._phi_payload_drop_record(payload, producer_seq, reason)
            return
        name = payload.stack_name or "(no stack)"
        detail = name
        if payload.image_count is not None:
            detail += f" — {payload.image_count} img"
            if payload.total_images is not None:
                detail += f" / {payload.total_images}"
        if payload.current_batch is not None:
            detail += f" — batch {payload.current_batch}"
            if payload.total_batches is not None:
                detail += f" / {payload.total_batches}"
        self._preview_detail = detail
        self._render_preview_label()

        # Derive the engine-authoritative scientific-preview identity up front
        # so the displayed preview and the live-auto target always agree on the
        # *same* identity (Classic/Reproject share the batch counter; Drizzle:
        # accepted frame).  The route label (``PREV_SRC``) is recorded
        # separately for truthful instrumentation and is deliberately *not* an
        # identity dimension (``refresh_preview`` re-renders via SUM/W
        # regardless of the run's stacking mode).
        identity = self._derive_preview_identity(payload)
        preview_mode = self._derive_preview_mode(payload)

        if phi_trace_enabled():
            src_label = self._payload_preview_source(payload) or ""
            identity_label = (
                f"{identity[0]}:{identity[1]}" if identity else "none"
            )
            res_label = f"x1/{self._effective_preview_downsample_factor()}"
            # True producer monotonic preview sequence + requested/effective
            # producer resolution (written into the payload header by the
            # producers).  Fall back to the Qt-carried counters only when the
            # producer metadata is absent (legacy/test payloads).  The
            # acceptance gate above already read ``producer_seq``; the payload
            # header is immutable, so this re-read is only for the record.
            producer_req = self._payload_preview_factor(payload, "PREV_REQ")
            producer_res = self._payload_preview_factor(payload, "PREV_RES")
            producer_cap = self._payload_preview_factor(payload, "PREV_CAP")
            seq = producer_seq
            if seq is None:
                seq = getattr(payload, "image_count", None)
            if seq is None:
                seq = getattr(payload, "current_batch", None)
            arr0 = payload.data
            if isinstance(arr0, (tuple, list)) and len(arr0) >= 1:
                arr0 = arr0[0]
            shape = getattr(arr0, "shape", None)
            shape_label = "x".join(str(s) for s in shape) if shape else "-"
            producer_run = self._payload_preview_run(payload)
            self._phi_trace_ctx = {
                "src": src_label,
                "identity": identity_label,
                "res": res_label,
                "shape": shape_label,
                "pseq": str(producer_seq) if producer_seq is not None else "-",
                "prun": str(producer_run) if producer_run is not None else "-",
                "preq": str(producer_req) if producer_req is not None else "-",
                "pres": str(producer_res) if producer_res is not None else "-",
                "pcap": str(producer_cap) if producer_cap is not None else "-",
            }
            phi_trace_stage(
                logger,
                route=preview_mode,
                stage="payload_arrive",
                arr=None,
                **self._phi_trace_ctx,
                seq=int(seq) if seq is not None else -1,
            )
        else:
            self._phi_trace_ctx = None

        option_a = _is_option_a_preview_payload(payload.data)
        if option_a:
            # Option-A: derive the display source from the adaptive-anchor mapped
            # pristine pre-WB float buffer (never from legacy_normalized once
            # raw extraction succeeds).
            image = self._ingest_option_a_preview(payload.data)
        else:
            # Legacy single-array payload: keep the existing QImage render path.
            image = render_preview_image(payload.data)
        if image is not None and not image.isNull():
            # ``render_preview_image`` already returns a deep copy, so storing
            # it here is safe and independent of the payload's buffers.
            self._preview_source = image
            # STABLE-B: preserve the user's accumulated rotation across a valid
            # successive preview.  Rotation (and zoom/pan, below) are view
            # state, independent of the scientific content; they only reset at
            # lifecycle boundaries (_on_run_started, new initial folder, clear,
            # invalid payload).
            if not option_a:
                # A legacy payload carries no raw-linear analysis: drop any
                # stale Option-A buffers so the float state always describes
                # the current (legacy) source.
                self._reset_preview_analysis()
            # Record the displayed scientific-preview identity (and advance the
            # raw revision exactly once per new preview) before live auto runs,
            # so live auto targets the identity currently being displayed.
            self._record_preview_identity(identity, preview_mode)
            # Apply deterministic live display adaptation only when this
            # payload represents a *new completed batch/group*.  The method is
            # atomic (no intermediate refresh), so the unconditional refresh
            # below remains the single render/histogram scheduling point.
            self._apply_live_auto_for_batch(payload)
        else:
            self._preview_source = None
            self._preview_rotation = 0
            # Invalid/missing data clears any stale display-anchor analysis so
            # the next valid preview re-establishes anchors from scratch (and
            # drops any identity instrumentation for the vanished preview).
            self._reset_preview_analysis()
            # An invalid/unrenderable payload is a lifecycle boundary: reset
            # zoom + pan to the Tk defaults (rotation was reset above).
            self._reset_view_transform()
        self._refresh_preview_view()

    def _live_auto_batch_token(self, payload: BackendPreviewPayload):
        """Return a deduplicatable scientific-preview identity, or ``None``.

        A ``(run_context_id, family, counter)`` tuple that changes exactly when
        a *new displayed scientific preview* arrives and stays stable across
        duplicate callbacks (repaints / resolution refreshes) for the same
        preview.  Live auto is gated to an active run and derives the identity
        from engine metadata only — the positive ``image_count`` (Drizzle
        accepted-frame counter) or ``current_batch`` (Classic/Reproject batch
        counter).  The ``PREV_SRC`` header selects only the *family* (Drizzle vs
        batch); within the batch family it is deliberately *not* an identity
        dimension, because ``refresh_preview`` re-renders every non-Drizzle
        session through the SUM/W route (``PREV_SRC="SUM/W Accumulators"``)
        even for a Reproject run, so a same-batch resolution refresh must stay
        inert.  The GUI ``drizzle_group_spin`` widget is deliberately *not* a
        freshness authority: every delivered scientific preview gets its own
        identity, so a displayed preview N can never carry live-auto parameters
        computed for a different identity.
        """
        if not self._running:
            return None
        identity = self._derive_preview_identity(payload)
        if identity is None:
            return None
        family, counter = identity
        return (self._run_context_id, family, counter)

    def _payload_preview_source(self, payload: BackendPreviewPayload) -> str:
        """Return the engine ``PREV_SRC`` label for a payload (``""`` if absent)."""
        header = getattr(payload, "header", None)
        try:
            return str(header.get("PREV_SRC", "")) if header is not None else ""
        except Exception:
            return ""

    def _payload_preview_seq(self, payload: BackendPreviewPayload) -> Optional[int]:
        """Return the producer monotonic preview sequence (``PREV_SEQ``), or None.

        This is the true per-emission producer sequence written by
        ``queue_manager`` (Classic SUM/W and standard Drizzle routes) — not the
        Qt-derived ``image_count``/``current_batch`` counters, which need not be
        monotonic across production order.  Display-only.  When present, the
        value feeds the PHI-R3 monotonic acceptance gate in :meth:`_on_preview`
        (stale/duplicate emissions are dropped); when absent (legacy/test
        payloads) the payload is accepted unconditionally.
        """
        header = getattr(payload, "header", None)
        if header is None:
            return None
        try:
            raw = header.get("PREV_SEQ", None)
        except Exception:
            return None
        if raw is None:
            return None
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        try:
            value = int(raw)
        except Exception:
            return None
        return value if value >= 0 else None

    def _payload_preview_run(self, payload: BackendPreviewPayload) -> Optional[int]:
        """Return the durable producer run/session id (``PREV_RUN``), or None.

        Written by the active Classic/Drizzle producers on every emission
        (required display metadata, never debug-gated).  The id is bound per
        stacker instance — the Qt run lifecycle assigns it at Start, the
        backend stamps it onto the stacker — so all payloads of one run share
        it and a payload of any other producer session is distinguishable at
        Qt arrival.  ``None`` for legacy/unsequenced payloads.
        """
        header = getattr(payload, "header", None)
        if header is None:
            return None
        try:
            raw = header.get("PREV_RUN", None)
        except Exception:
            return None
        if raw is None:
            return None
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        try:
            value = int(raw)
        except Exception:
            return None
        return value

    def _accept_preview_payload(self, producer_run: Optional[int], producer_seq: int) -> bool:
        """Producer acceptance rule (PHI-R3.2 gate, run-scoped).

        Two checks, in order:

        1. *Run/session identity* — when the payload carries a ``PREV_RUN`` and
           the current run has a bound producer session (set at run start from
           the id allocated at Start), a payload from any *other* session (a
           late old-run payload queued across the run boundary) is refused as
           foreign without touching the sequence high-water mark.  When no
           session is bound yet (idle state / direct test calls / simulated
           runs) the first sequenced payload's run is bound lazily.
        2. *Monotonic sequence* — ``producer_seq`` is accepted (recorded as the
           run's high-water mark) when it is the first sequenced payload of the
           run or strictly newer than the last accepted one; an equal sequence
           is a duplicate and an older one is stale — both refused without
           touching any state.

        The gate is reset and re-bound at run start (:meth:`_on_run_started`),
        so a new run's first sequenced payload (the fresh producer restarts
        its counter at 1 under the new session id) is always accepted.
        """
        if producer_run is not None:
            expected = self._preview_run_session
            if expected is None:
                # No run-bound expectation yet: bind lazily to this producer
                # session (real GUI runs always pre-bind at run start, so this
                # only covers idle/direct-call/test sequencing).
                self._preview_run_session = producer_run
            elif producer_run != expected:
                return False  # foreign producer session (late old-run payload)
        last = self._last_accepted_preview_seq
        if last is None or producer_seq > last:
            self._last_accepted_preview_seq = producer_seq
            return True
        return False

    def _phi_payload_drop_record(self, payload: BackendPreviewPayload, seq: int, reason: str) -> None:
        """Emit a ``payload_arrive`` trace record for a gated drop (best-effort).

        The record marks the arrival order and the acceptance reason
        (``drop=stale`` for an older emission, ``drop=duplicate`` for a
        repeated one, ``drop=foreign`` for a payload of another producer
        run/session — PHI-R3.2) so the gate's decision is observable and
        attributable without touching any live state (``_phi_trace_ctx`` keeps
        describing the last *accepted* payload).  Never raises.
        """
        try:
            src_label = self._payload_preview_source(payload) or ""
            identity = self._derive_preview_identity(payload)
            identity_label = f"{identity[0]}:{identity[1]}" if identity else "none"
            arr0 = payload.data
            if isinstance(arr0, (tuple, list)) and len(arr0) >= 1:
                arr0 = arr0[0]
            shape = getattr(arr0, "shape", None)
            shape_label = "x".join(str(s) for s in shape) if shape else "-"
            run = self._payload_preview_run(payload)
            phi_trace_stage(
                logger,
                route=self._derive_preview_mode(payload),
                stage="payload_arrive",
                arr=None,
                src=src_label,
                identity=identity_label,
                shape=shape_label,
                pseq=str(seq),
                prun=str(run) if run is not None else "-",
                seq=seq,
                drop=reason,
            )
        except Exception:
            pass  # tracing is best-effort

    def _payload_preview_factor(self, payload: BackendPreviewPayload, key: str) -> Optional[int]:
        """Return a producer resolution metadata card (``PREV_REQ``/``PREV_RES``/
        ``PREV_CAP``) as an int, or ``None`` when absent.

        ``PREV_REQ`` is the requested GUI resolution factor; ``PREV_RES`` is
        the factor actually applied to the delivered payload (1 when no resize
        ran — e.g. Classic's small-image guard / failure path); ``PREV_CAP``
        is 1 when the Drizzle max-side display guard fired before the factor.
        Tuple-valued FITS cards are unwrapped.  ``None`` for legacy/test
        payloads without the card — the caller falls back safely.
        """
        header = getattr(payload, "header", None)
        if header is None:
            return None
        try:
            raw = header.get(key, None)
        except Exception:
            return None
        if raw is None:
            return None
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        try:
            value = int(raw)
        except Exception:
            return None
        return value

    def _payload_preview_res(self, payload: BackendPreviewPayload) -> Optional[int]:
        """Backward-compatible alias: effective producer resolution (``PREV_RES``).

        Deprecated in favour of :meth:`_payload_preview_factor` with an explicit
        card key; kept so existing callers/tests keep working.  ``None`` when
        the payload does not carry it (legacy routes / test payloads).
        """
        return self._payload_preview_factor(payload, "PREV_RES")

    def _derive_preview_identity(self, payload: BackendPreviewPayload):
        """Derive the engine-authoritative scientific-preview identity.

        Returns ``(family, counter)`` or ``None`` when the payload carries no
        usable identity.  ``family`` is one of:

        * ``"drizzle"`` — ``image_count`` (accepted-frame counter).  The
          standard policy emits one preview per accepted frame and the
          incremental policy one per group, so ``image_count`` is distinct for
          every delivered preview.  The ``drizzle_group_spin`` widget cadence is
          *not* consulted.
        * ``"batch"``    — ``current_batch`` (``stacked_batches_count``), shared
          by the Classic SUM/W and the legacy Reproject/coadd paths.

        The ``PREV_SRC`` header selects only the *family* (Drizzle vs batch); it
        is deliberately **not** an identity dimension within the batch family.
        ``queue_manager._update_preview`` (legacy incremental reproject/coadd)
        emits no ``PREV_SRC``, while ``queue_manager.refresh_preview`` routes
        every non-Drizzle session through ``_update_preview_sum_w`` with
        ``PREV_SRC="SUM/W Accumulators"`` — both key on the same
        ``stacked_batches_count``.  Treating them as the same ``("batch", N)``
        keeps a same-batch resolution refresh inert for dedupe and
        ``raw_revision``.
        """
        preview_source = self._payload_preview_source(payload)
        if _DRIZZLE_PREVIEW_SRC in preview_source:
            family = "drizzle"
            counter = _positive_int(getattr(payload, "image_count", None))
        else:
            family = "batch"
            counter = _positive_int(getattr(payload, "current_batch", None))
        if counter is None:
            return None
        return (family, counter)

    def _derive_preview_mode(self, payload: BackendPreviewPayload) -> str:
        """Return the truthful engine renderer route label for a payload.

        ``"drizzle"`` (Drizzle Accumulator), ``"classic"`` (SUM/W Accumulators)
        or ``"reproject"`` (legacy incremental-reprojection/coadd with no
        ``PREV_SRC``).  This is a *route* observation for the witness only — it
        is deliberately **not** part of the scientific-preview identity, because
        ``refresh_preview`` re-renders every non-Drizzle session through the
        SUM/W route regardless of the run's stacking mode (so a Reproject run's
        resolution refresh truthfully carries ``"classic"`` while remaining the
        same scientific batch).
        """
        preview_source = self._payload_preview_source(payload)
        if _DRIZZLE_PREVIEW_SRC in preview_source:
            return "drizzle"
        if _SUMW_PREVIEW_SRC in preview_source:
            return "classic"
        return "reproject"

    def _record_preview_identity(self, identity, mode) -> None:
        """Record the scientific-preview identity of the displayed source.

        ``identity`` is ``(family, counter)`` from
        :meth:`_derive_preview_identity`, or ``None`` (no usable identity);
        ``mode`` is the truthful route label from :meth:`_derive_preview_mode`.
        The full identity ``(run_context_id, family, counter)`` is stored for
        the witness, and ``_raw_revision`` advances exactly once per *new*
        displayed scientific preview (a changed identity) — never on a duplicate
        callback, an ordinary repaint, or a same-batch resolution refresh whose
        carrier route (``PREV_SRC``) changed.  Display-only: no scientific state
        is touched.
        """
        if identity is None:
            self._preview_mode = None
            self._preview_identity = None
            self._displayed_identity = None
            return
        family, counter = identity
        full = (self._run_context_id, family, counter)
        self._preview_mode = mode
        if full != self._preview_identity:
            self._raw_revision += 1
        self._preview_identity = full
        self._displayed_identity = full

    def _apply_live_auto_for_batch(self, payload: BackendPreviewPayload) -> bool:
        """Apply each enabled live-auto operation once for a new batch.

        Returns whether any operation ran.  The payload/scientific arrays are
        read-only inputs; only Qt-owned controls and display buffers change.
        """
        token = self._live_auto_batch_token(payload)
        if token is None or token == self._last_live_auto_batch_token:
            return False
        # Record the boundary even when both features are disabled.  Enabling a
        # checkbox later must not retroactively process a duplicate callback.
        self._last_live_auto_batch_token = token
        ran = False
        if self._live_auto_wb_enabled:
            gains = self._compute_auto_wb_gains()
            if gains is not None:
                self._apply_wb_gains(gains, refresh=False)
                self._live_auto_wb_count += 1
                ran = True
        if self._live_auto_stretch_enabled:
            points = self._compute_auto_stretch_points()
            if points is not None:
                bp, wp = points
                self.stretch_combo.blockSignals(True)
                try:
                    self.stretch_combo.setCurrentText("asinh")
                finally:
                    self.stretch_combo.blockSignals(False)
                self._stretch = "asinh"
                bp, wp = normalize_bp_wp(
                    round(float(bp), 4),
                    round(float(wp), 4),
                    max_value=self._bp_wp_control_upper,
                )
                self._write_bp_wp_state(bp, wp)
                # Witness: record the BP/WP live auto just wrote so tests can
                # compare live vs one-shot Auto Stretch on the same WB-only
                # buffer within control quantization.
                self._live_bp = self._black_point
                self._live_wp = self._white_point
                self._live_auto_stretch_count += 1
                ran = True
        return ran

    def _reset_preview_analysis(self) -> None:
        """Clear display anchors and the display-only float analysis buffers.

        Called at run start, on a genuinely new initial-preview folder, on an
        explicit preview clear, and on an invalid payload — never on ordinary
        successive backend preview updates (anchors remain stable for small
        changes and widen only for significant drift).  Display-only: it never
        touches the backend, science
        accumulators, or the stored ``_preview_source`` QImage.
        """
        self._raw_linear = None
        self._pristine_float = None
        self._wb_only_float = None
        self._wb_only_wb = None
        self._anchor_lo = None
        self._anchor_hi = None
        # PHI-R3.1: the Option-A analysis domain is gone — the BP/WP display
        # controls and histogram markers return to the legacy [0, 1] domain
        # (spins/sliders are retooled back; markers re-scoped).
        self._analysis_upper = 1.0
        self._sync_analysis_domain()
        # A fresh analysis context also drops the scientific-preview identity
        # instrumentation (the displayed preview no longer exists).  The
        # monotonic ``_raw_revision`` and the run-scoped live-auto target state
        # are deliberately preserved here (reset only at run start).
        self._preview_mode = None
        self._preview_identity = None
        self._displayed_identity = None
        self._analysis_generation += 1
        # A new analysis context invalidates any cached histogram model (the
        # WB-only source is gone; the next refresh recomputes from scratch) and
        # any in-flight/pending worker computation so a stale result cannot
        # repopulate the cleared UI.
        self._histogram_coordinator.invalidate()
        self._histogram_model = None
        self._histogram_model_revision = None
        self._histogram_scheduled_revision = None
        self._histogram_trace_ctx = None
        # Same for the legacy single-array histogram/stats cache.
        self._legacy_hist_key = None
        self._legacy_hist = None
        self._legacy_hist_percentile = 1.0
        self._legacy_hist_stats = None

    def _sync_analysis_domain(self) -> None:
        """Re-scope the BP/WP controls + histogram markers to the analysis domain.

        PHI-R3.1 unit contract: for Option-A float previews the black/white
        points operate in the preserved analysis units ``[0, _analysis_upper]``
        (``upper = max(1.0, finite max)`` of the WB-only buffer — the exact
        upper the float-histogram model declares), so a white point above ``1``
        is a first-class control/marker value.  When the upper changes (a new
        WB-only derivation — new source or WB change) this method:

        1. retools both BP/WP spin ranges (``[0, upper]``) and slider ranges
           (one 0.001 step per tick) — the legacy QImage path keeps upper ==
           ``1.0`` and the historical ranges;
        2. reconciles the current pair deterministically in analysis units
           without inversion (:func:`preview_adjust.normalize_bp_wp` with
           ``max_value=upper``): a white point above the new upper is pulled
           down to it, the black point stays below with the shared minimum
           separation;
        3. re-scopes every live histogram view's marker domain
           (:meth:`HistogramView.set_analysis_domain`) so drags and markers
           agree with the controls.

        Runs under the BP/WP re-entrancy guard (programmatic writes never
        trigger a recursive refresh or disable live-auto) and is a no-op when
        the upper is unchanged.  The *control* domain is the grid ceiling of
        the raw analysis upper (``ceil(upper / step) * step``): the spin/slider
        widgets and the quantized markers can only represent 0.001-grid values,
        so the domain is rounded up to the next grid point (never down — a
        white point must be able to reach the true data top).
        """
        raw_upper = self._analysis_upper
        factor = int(round(1.0 / BLACK_POINT_STEP))
        upper = math.ceil(raw_upper * factor) / factor
        if abs(upper - self._bp_wp_control_upper) < 1e-12:
            return
        self._bp_wp_sync_guard = True
        try:
            n = max(1, int(round(upper / BLACK_POINT_STEP)))
            self.stretch_bp_slider.setRange(0, n)
            self.stretch_wp_slider.setRange(0, n)
            self.stretch_bp_spin.setRange(0.0, float(upper))
            self.stretch_wp_spin.setRange(0.0, float(upper))
            bp, wp = normalize_bp_wp(
                self._black_point, self._white_point, max_value=upper
            )
            self.stretch_bp_spin.setValue(bp)
            self.stretch_wp_spin.setValue(wp)
        finally:
            self._bp_wp_sync_guard = False
        self._black_point = self.stretch_bp_spin.value()
        self._white_point = self.stretch_wp_spin.value()
        self._bp_wp_control_upper = upper
        for view in self._histogram_views():
            view.set_analysis_domain(upper)

    def _ensure_wb_only_float(self):
        """Return the cached WB-only float buffer, re-deriving it only when stale.

        Re-derives from ``_pristine_float`` when there is no cached buffer yet
        or the stored WB gains no longer match the current ``self._wb`` (a WB
        change).  Each actual re-derivation bumps ``_wb_only_revision`` (the
        histogram-model cache key).  Never mutates ``_pristine_float``.
        """
        if self._pristine_float is None:
            self._wb_only_float = None
            self._wb_only_wb = None
            return None
        if self._wb_only_float is not None and self._wb_only_wb == self._wb:
            return self._wb_only_float
        self._wb_only_float = apply_wb_float(self._pristine_float, self._wb)
        self._wb_only_wb = self._wb
        self._wb_only_revision += 1
        # PHI-R3.1: the display BP/WP controls operate in the analysis units of
        # this buffer — recompute the deterministic analysis upper (the exact
        # same value the float-histogram model will declare) and re-scope the
        # controls/markers, reconciling a white point that no longer fits.
        self._analysis_upper = analysis_upper_bound(self._wb_only_float)
        self._sync_analysis_domain()
        if phi_trace_enabled():
            ctx = self._phi_trace_ctx or {}
            phi_trace_stage(
                logger,
                route="qt",
                stage="wb_only",
                arr=self._wb_only_float,
                wb=f"{self._wb[0]:g},{self._wb[1]:g},{self._wb[2]:g}",
                **ctx,
            )
        return self._wb_only_float

    def _ingest_option_a_preview(self, data):
        """Ingest an Option-A payload into frozen-anchor display state.

        Extracts an independent raw-linear copy via the accepted core, computes
        p0.5/p99.5 anchors on the first valid preview of the current context,
        and on each successive preview accommodates legitimate photometric
        drift via a hysteretic monotonic widening of those anchors
        (:func:`preview_analysis.adapt_anchors_for_drift`), then maps through
        the effective anchors into the pristine pre-WB float buffer and returns
        a neutral ``QImage`` carrier (or ``None`` when the payload is
        unusable).  PHI-R3: the mapping preserves finite out-of-range float
        headroom (no premature clip).  PHI-R3.1: the returned ``QImage`` is
        stored as ``_preview_source`` for geometry / fallback only — the
        *visible* display is re-rendered from the preserved float analysis
        source by :meth:`_refresh_preview_view` (float stretch with analysis-
        unit black/white points, final uint8 conversion last).  The input
        payload arrays are never mutated.
        """
        raw = extract_raw_linear(data)
        if raw is None:
            return None
        if phi_trace_enabled():
            ctx = self._phi_trace_ctx or {}
            phi_trace_stage(
                logger,
                route="qt",
                stage="raw_source",
                arr=raw,
                **ctx,
            )
        if self._anchor_lo is None or self._anchor_hi is None:
            self._anchor_lo, self._anchor_hi = compute_anchors(raw)
        else:
            self._anchor_lo, self._anchor_hi = adapt_anchors_for_drift(
                self._anchor_lo, self._anchor_hi, raw
            )
        mapped = map_raw_linear(raw, self._anchor_lo, self._anchor_hi)
        if mapped is None:
            return None
        if phi_trace_enabled():
            ctx = self._phi_trace_ctx or {}
            phi_trace_stage(
                logger,
                route="qt",
                stage="anchor_mapped",
                arr=mapped,
                lo=f"{self._anchor_lo:.6g}",
                hi=f"{self._anchor_hi:.6g}",
                **ctx,
            )
        self._raw_linear = raw
        self._pristine_float = mapped
        # New mapped source: the WB-only buffer is stale until re-derived.
        self._wb_only_float = None
        self._wb_only_wb = None
        return render_preview_image(mapped)

    # ------------------------------------------------ initial preview (M12)
    def _try_show_first_input_image(self) -> None:
        """Auto-load the first FITS of the input folder (Tk parity).

        The folder is validated and scanned synchronously (both are cheap); only
        the actual FITS load + debayer runs on a daemon worker thread, which
        delivers the result back through the queued
        :attr:`_initial_preview_result` signal.  A folder unchanged from the
        last successful load is skipped so repeated settings restores never
        reload the same image.
        """
        input_folder = self.input_edit.text().strip()
        if input_folder:
            input_folder = os.path.abspath(input_folder)

        # Guard against redundant reloads: same folder already loaded.
        if input_folder and input_folder == self._last_preview_folder:
            return

        if not input_folder or not os.path.isdir(input_folder):
            self._last_preview_folder = None
            self._clear_preview(self._tr("preview_no_input_folder"))
            return

        first = initial_preview.find_first_fits_file(input_folder)
        if first is None:
            self._last_preview_folder = None
            self._clear_preview(self._tr("preview_no_fits"))
            return

        # Loading state: label only; the previous image (if any) stays until
        # the new one arrives (Tk parity).
        self._preview_detail = f"{self._tr('preview_loading')} {first}"
        self._render_preview_label()

        # A genuinely new folder is a new display-anchor context: drop any
        # anchors / float analysis buffers from a previous folder or run before
        # the (async) load lands.  Redundant reloads were already skipped above.
        self._reset_preview_analysis()

        bayer_pattern = self.settings_state.bayer_pattern or "GRBG"
        threading.Thread(
            target=self._initial_preview_worker,
            args=(input_folder, first, bayer_pattern),
            daemon=True,
            name="InitialPreviewLoader",
        ).start()

    def _initial_preview_worker(self, folder: str, filename: str, bayer_pattern) -> None:
        """Load + debayer the first FITS on a daemon thread; emit the result.

        Never touches a widget: it only imports the engine lazily, reads the
        file into an in-memory ndarray and emits the queued result signal.
        """
        try:
            data, header = initial_preview.load_initial_preview(
                folder, filename, bayer_pattern
            )
            result = initial_preview.InitialPreviewResult(
                folder=folder, filename=filename, data=data, header=header
            )
        except Exception:
            result = initial_preview.InitialPreviewResult(
                folder=folder, filename=filename, error="load_failed"
            )
        self._initial_preview_result.emit(result)

    def _on_initial_preview_result(self, result) -> None:
        """GUI-thread slot: render a loaded initial preview (or clear on error).

        A fast input-folder switch while a load is in flight can deliver the
        *old* folder's image after the new folder's image.  To keep the preview
        honest we drop any result whose ``folder`` (absolute path) no longer
        matches the currently selected input folder.
        """
        current_folder = self.input_edit.text().strip()
        current_folder = os.path.abspath(current_folder) if current_folder else ""
        if result.folder:
            result_folder = os.path.abspath(result.folder)
            if result_folder != current_folder:
                # Stale result for a folder that is no longer selected.
                return
        if result.error is not None:
            self._last_preview_folder = None
            self._clear_preview(self._tr("preview_error"))
            return
        image = render_preview_image(result.data)
        if image is None or image.isNull():
            self._last_preview_folder = None
            self._clear_preview(self._tr("preview_error"))
            return
        self._preview_source = image
        self._preview_rotation = 0
        self._last_preview_folder = result.folder
        self._preview_detail = f"{self._tr('preview_loaded')} {result.filename}"
        self._render_preview_label()
        self._reset_view_transform()
        self._refresh_preview_view()

    def _clear_preview(self, message: str) -> None:
        """Clear the preview surface and show a localized message."""
        self._preview_detail = message
        self._render_preview_label()
        self._preview_source = None
        self._preview_rotation = 0
        self._reset_preview_analysis()
        self._reset_view_transform()
        self._refresh_preview_view()

    # ------------------------------------------------- preview view controls
    def _on_zoom_changed(self, _index: int) -> None:
        """Recompute the view when the zoom combo changes (user preset pick).

        Any user preset pick recentres the view (resets pan to 0): the combo
        returns the view to a discrete preset (Tk ``zoom_fit`` /
        ``zoom_full_size`` parity).  "Fit" is a mode; the percent presets set
        the single continuous factor to their literal value.  A blank combo
        (custom wheel zoom, only ever set programmatically) is a no-op.
        """
        if self._zoom_sync_guard:
            return
        self._view_offset_x = 0.0
        self._view_offset_y = 0.0
        label = self.zoom_combo.currentText()
        factor = ZOOM_FACTORS.get(label)
        if factor is not None:
            self._preview_zoom_factor = factor
        self._refresh_preview_view()

    def _on_rotate_left(self) -> None:
        """Rotate the preview 90° counter-clockwise (cumulative, modulo 360)."""
        if self._preview_source is None:
            return
        self._preview_rotation = (self._preview_rotation - 90) % 360
        # Tk ``rotate_left`` resets pan to avoid disorientation on aspect flip.
        self._view_offset_x = 0.0
        self._view_offset_y = 0.0
        self._refresh_preview_view()

    def _on_rotate_right(self) -> None:
        """Rotate the preview 90° clockwise (cumulative, modulo 360)."""
        if self._preview_source is None:
            return
        self._preview_rotation = (self._preview_rotation + 90) % 360
        self._view_offset_x = 0.0
        self._view_offset_y = 0.0
        self._refresh_preview_view()

    def _reset_view_transform(self) -> None:
        """Reset zoom + pan to the Tk defaults (100%, centred) without a redraw.

        Mirrors ``PreviewManager.reset_zoom_and_pan``: continuous zoom → 1.0 and
        pan offset → (0, 0), then re-syncs the combo to the ``100%`` preset.
        """
        self._preview_zoom_factor = 1.0
        self._view_offset_x = 0.0
        self._view_offset_y = 0.0
        self._sync_zoom_combo_to_factor()

    def _sync_zoom_combo_to_factor(self) -> None:
        """Re-sync the combo to the current continuous factor (blank = custom)."""
        label = preset_label_for_factor(self._preview_zoom_factor)
        self._zoom_sync_guard = True
        try:
            if label is None:
                self.zoom_combo.setCurrentIndex(-1)
            else:
                self.zoom_combo.setCurrentText(label)
        finally:
            self._zoom_sync_guard = False

    def _on_wheel_zoom(self, direction: int, x: float, y: float) -> None:
        """Mouse-wheel zoom over the preview surface (Tk ``_zoom_on_scroll``).

        Multiplies/divides the continuous factor by ``ZOOM_STEP`` (1.15), clamps
        to ``[MIN_ZOOM, MAX_ZOOM]`` and anchors the zoom at the cursor (the pan
        offset shifts so the pixel under the cursor stays put).  Wheeling from
        "Fit" exits Fit and continues from the current fit scale (Tk ``zoom_fit``
        sets ``zoom_level`` to the fit scale).  Display-only: ``_preview_source``
        is never touched.
        """
        if self._preview_source is None or self._preview_source.isNull():
            return
        if self.zoom_combo.currentText() == "Fit":
            self._preview_zoom_factor = fit_scale(
                self._preview_source,
                self._preview_rotation,
                self._effective_preview_downsample_factor(),
                self.preview_image_label.size(),
            )
        old_zoom = self._preview_zoom_factor
        new_zoom = old_zoom * ZOOM_STEP if direction > 0 else old_zoom / ZOOM_STEP
        new_zoom = clamp_zoom_factor(new_zoom)
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        zoom_ratio = new_zoom / old_zoom
        # Cursor-anchored zoom (Tk parity): keep the pixel under the cursor fixed.
        cx = self.preview_image_label.width() / 2.0
        cy = self.preview_image_label.height() / 2.0
        rel_x = x - (cx + self._view_offset_x)
        rel_y = y - (cy + self._view_offset_y)
        self._view_offset_x += rel_x * (1.0 - zoom_ratio)
        self._view_offset_y += rel_y * (1.0 - zoom_ratio)
        self._preview_zoom_factor = new_zoom
        self._sync_zoom_combo_to_factor()
        self._refresh_preview_view()

    def _on_pan_delta(self, dx: float, dy: float) -> None:
        """Left-drag pan: shift the viewport offset (Tk ``_pan_image`` parity).

        The offset is unbounded (Tk applies no clamping); it is applied by
        ``render_view`` via ``compose_panned_pixmap``.
        """
        if self._preview_source is None or self._preview_source.isNull():
            return
        self._view_offset_x += float(dx)
        self._view_offset_y += float(dy)
        self._refresh_preview_view()

    def _preview_res_text(self) -> str:
        """Return the localized label for the current preview-resolution factor."""
        factor = self._preview_res_factor
        prefix = self._tr("preview_res_prefix")
        return f"{prefix} 1/{factor}" if factor > 1 else f"{prefix} 1/1"

    def _render_preview_res_button(self) -> None:
        """Re-render the Res-cycle button label from the factor + language."""
        self.preview_res_button.setText(self._preview_res_text())

    def _on_preview_res_cycle(self) -> None:
        """Cycle the preview-resolution factor 1→2→3→4→1.

        Advances the factor among 1/2/3/4 (default 1), updates the button label
        and re-renders the local preview at the new factor (display-only, as
        before).  During an active run it additionally forwards the factor to
        the live engine via the thread-safe control channel
        (:meth:`RunController.set_preview_downsample_factor`), which applies
        ``set_preview_downsample_factor`` + ``refresh_preview`` to the running
        stacker on the worker thread (Tk ``_cycle_preview_resolution``
        engine-coupling parity).  When no run is active the button is
        display-only, exactly as before.
        """
        factors = PREVIEW_RES_FACTORS
        if self._preview_res_factor in factors:
            idx = factors.index(self._preview_res_factor)
        else:
            idx = factors.index(DEFAULT_PREVIEW_RES_FACTOR)
        idx = (idx + 1) % len(factors)
        self._preview_res_factor = factors[idx]
        self._render_preview_res_button()
        self._refresh_preview_view()
        # Live engine coupling (M22): only forward while a run is active.  The
        # controller/worker/backend chain is a no-op for backends without a
        # live engine (simulated), and the worker thread applies it to the
        # stacker, so a Res click never races the engine's own processing.
        if self._running:
            self.controller.set_preview_downsample_factor(self._preview_res_factor)

    def _on_wb_changed(self, *_ignored) -> None:
        """Update the white-balance gains and re-render the preview (display-only).

        A single-user WB control edit updates the authoritative ``_wb`` and
        re-renders once.  When the edit arrives while :meth:`_apply_wb_gains`
        is atomically writing several controls (Auto WB / WB Reset) the guard
        suppresses the intermediate callback so only one refresh/job occurs.
        """
        if self._wb_sync_guard:
            return
        # Direct gain edit = explicit user intent.  It disables only live WB;
        # live stretch remains independent.
        self._set_live_auto_wb_enabled(False)
        self._wb = (
            self.wb_r_spin.value(),
            self.wb_g_spin.value(),
            self.wb_b_spin.value(),
        )
        self._refresh_preview_view()

    def _apply_wb_gains(self, gains: tuple, *, refresh: bool = True) -> None:
        """Apply the three WB gains atomically (Auto WB / WB Reset).

        Writes all three slider/spin pairs and the authoritative ``_wb`` under
        a re-entrancy guard so no intermediate callback/job fires, then triggers
        exactly one preview refresh/histogram recompute.  When the effective
        control values are unchanged (idempotent repeat / neutral reset) it
        triggers zero recomputes.
        """
        old_wb = self._wb
        self._wb_sync_guard = True
        try:
            self.wb_r_spin.setValue(gains[0])
            self.wb_g_spin.setValue(gains[1])
            self.wb_b_spin.setValue(gains[2])
        finally:
            self._wb_sync_guard = False
        self._wb = (
            self.wb_r_spin.value(),
            self.wb_g_spin.value(),
            self.wb_b_spin.value(),
        )
        if self._wb == old_wb or not refresh:
            return
        self._refresh_preview_view()

    def _set_live_auto_wb_enabled(self, enabled: bool) -> None:
        """Set/synchronize the independent Live Auto WB intent."""
        enabled = bool(enabled)
        self._live_auto_wb_enabled = enabled
        for checkbox in (
            getattr(self, "live_auto_wb_check", None),
            getattr(
                getattr(self, "_detached_histogram_window", None),
                "live_auto_wb_check",
                None,
            ),
        ):
            if checkbox is not None and checkbox.isChecked() != enabled:
                checkbox.blockSignals(True)
                try:
                    checkbox.setChecked(enabled)
                finally:
                    checkbox.blockSignals(False)

    def _set_live_auto_stretch_enabled(self, enabled: bool) -> None:
        """Set/synchronize the independent Live Auto Stretch intent."""
        enabled = bool(enabled)
        self._live_auto_stretch_enabled = enabled
        for checkbox in (
            getattr(self, "live_auto_stretch_check", None),
            getattr(
                getattr(self, "_detached_histogram_window", None),
                "live_auto_stretch_check",
                None,
            ),
        ):
            if checkbox is not None and checkbox.isChecked() != enabled:
                checkbox.blockSignals(True)
                try:
                    checkbox.setChecked(enabled)
                finally:
                    checkbox.blockSignals(False)

    def _on_wb_reset(self) -> None:
        """Reset the three white-balance gains to their neutral values (atomic)."""
        self._apply_wb_gains(DEFAULT_WB)

    def _on_stretch_changed(self, _index: int) -> None:
        """Update the display-stretch mode and re-render the preview (display-only)."""
        self._stretch = self.stretch_combo.currentText()
        self._refresh_preview_view()

    def _on_auto_wb(self) -> None:
        """Compute auto white-balance gains (display-only).

        For an Option-A preview the gains come from the pristine *pre-WB*
        mapped float buffer via ``compute_auto_wb_float`` (never from the
        already-WB buffer, so repeated explicit calls are deterministic and
        never move the display anchors).  Legacy single-array previews keep the
        existing ``compute_auto_wb`` QImage path.  Gains are written back to
        the WB controls atomically: when the gains actually change, all three
        slider/spin pairs and ``_wb`` update with a single preview refresh /
        histogram recompute; when they are unchanged (repeat AutoWB) zero
        recomputes occur.
        """
        gains = self._compute_auto_wb_gains()
        if gains is not None:
            self._apply_wb_gains(gains)

    def _compute_auto_wb_gains(self):
        """Return deterministic display-only AutoWB gains, or ``None``."""
        if self._preview_source is None or self._preview_source.isNull():
            return None
        if self._pristine_float is not None:
            r_gain, g_gain, b_gain = compute_auto_wb_float(self._pristine_float)
        else:
            r_gain, g_gain, b_gain = compute_auto_wb(self._preview_source)
        return (
            round(float(r_gain), 3),
            round(float(g_gain), 3),
            round(float(b_gain), 3),
        )

    def _on_stretch_bp_changed(self, *_ignored) -> None:
        """Black-point control edited: normalize + re-render (display-only)."""
        if not self._bp_wp_sync_guard:
            self._set_live_auto_stretch_enabled(False)
        self._apply_bp_wp_edit("bp")

    def _on_stretch_wp_changed(self, *_ignored) -> None:
        """White-point control edited: normalize + re-render (display-only)."""
        if not self._bp_wp_sync_guard:
            self._set_live_auto_stretch_enabled(False)
        self._apply_bp_wp_edit("wp")

    def _on_stretch_gamma_changed(self, *_ignored) -> None:
        """Gamma control edited: re-render the preview (display-only)."""
        self._gamma = self.stretch_gamma_spin.value()
        self._refresh_preview_view()

    def _apply_bp_wp_edit(self, driver: str) -> None:
        """Normalize a single-endpoint BP/WP edit and sync every surface.

        Reads the edited spin value, clamps it against the other (un-edited)
        endpoint via :func:`preview_adjust.clamp_bp_wp_edit` so a crossing edit
        deterministically clamps the edited endpoint (never the other), then
        writes the authoritative pair into both controls, ``_black_point`` /
        ``_white_point`` and the histogram handles in one pass.
        """
        if self._bp_wp_sync_guard:
            return
        value = (
            self.stretch_bp_spin.value()
            if driver == "bp"
            else self.stretch_wp_spin.value()
        )
        other = self._white_point if driver == "bp" else self._black_point
        bp, wp = clamp_bp_wp_edit(
            driver, value, other, max_value=self._bp_wp_control_upper
        )
        self._write_bp_wp_state(bp, wp)
        self._refresh_preview_view()

    def _set_bp_wp_pair(self, bp: float, wp: float) -> None:
        """Set both BP/WP points atomically from a validated pair (histogram
        drag mirror / auto stretch).  Normalizes + quantizes via the shared
        seam in the current analysis units (``max_value`` = analysis upper for
        Option-A float previews, legacy ``[0, 1]`` otherwise), writes
        controls/state once and refreshes once."""
        bp, wp = normalize_bp_wp(bp, wp, max_value=self._bp_wp_control_upper)
        self._write_bp_wp_state(bp, wp)
        self._refresh_preview_view()

    def _write_bp_wp_state(self, bp: float, wp: float) -> None:
        """Write the authoritative BP/WP into both controls + state atomically.

        Sets both spins under a re-entrancy guard (so their ``on_change``
        handlers do not recurse), then reads back the spin values (already
        quantized to the 3-decimal control resolution) as the single
        authoritative ``_black_point`` / ``_white_point`` state.
        """
        self._bp_wp_sync_guard = True
        try:
            self.stretch_bp_spin.setValue(bp)
            self.stretch_wp_spin.setValue(wp)
        finally:
            self._bp_wp_sync_guard = False
        self._black_point = self.stretch_bp_spin.value()
        self._white_point = self.stretch_wp_spin.value()

    def _on_stretch_reset(self) -> None:
        """Reset the stretch controls to the Tk defaults (Asinh, 0.01/0.99/1.0)."""
        self.stretch_combo.setCurrentText(DEFAULT_STRETCH)
        self.stretch_bp_spin.setValue(DEFAULT_BLACK_POINT)
        self.stretch_wp_spin.setValue(DEFAULT_WHITE_POINT)
        self.stretch_gamma_spin.setValue(DEFAULT_GAMMA)

    def _on_auto_stretch(self) -> None:
        """Auto Stretch (display-only).

        For an Option-A preview the black/white points come from
        ``compute_auto_stretch_float`` on the cached WB-only float buffer (no
        QImage min/max renormalization).  Legacy single-array previews keep the
        existing ``compute_auto_stretch`` QImage path.  Both write the points
        into the black/white slider+spin controls (atomically, so a stale
        in-between white point never clamps the new black point) and switch the
        stretch method to ``asinh``.  The display anchors / raw / pristine
        buffers are never mutated.
        """
        points = self._compute_auto_stretch_points()
        if points is None:
            return
        bp, wp = points
        self.stretch_combo.setCurrentText("asinh")
        self._set_bp_wp_pair(round(float(bp), 4), round(float(wp), 4))

    def _compute_auto_stretch_points(self):
        """Return deterministic display-only Auto Stretch BP/WP, or ``None``."""
        if self._preview_source is None or self._preview_source.isNull():
            return None
        if self._pristine_float is not None:
            wb_only = self._ensure_wb_only_float()
            if wb_only is None:
                return None
            return compute_auto_stretch_float(wb_only)
        wb_only = apply_preview_wb(self._preview_source, wb=self._wb)
        if wb_only is None or wb_only.isNull():
            return None
        return compute_auto_stretch(wb_only)

    def _on_hist_range_changed(self, bp: float, wp: float) -> None:
        """Mirror a histogram BP/WP drag into the stretch sliders (Tk
        ``update_stretch_from_histogram``).  Routed through the shared BP/WP
        normalization seam so the sliders/spins, authoritative state and
        histogram handles always agree (quantized to one control resolution)."""
        self._set_live_auto_stretch_enabled(False)
        self._set_bp_wp_pair(bp, wp)

    def _on_hist_auto_zoom_toggled(self, checked: bool) -> None:
        """Toggle auto-zoom on the histogram (Tk ``auto_zoom_histogram_var``)."""
        checked = bool(checked)
        for checkbox in (
            getattr(self, "auto_zoom_histo_check", None),
            getattr(
                getattr(self, "_detached_histogram_window", None),
                "auto_zoom_check",
                None,
            ),
        ):
            if checkbox is not None and checkbox.isChecked() != checked:
                checkbox.blockSignals(True)
                try:
                    checkbox.setChecked(checked)
                finally:
                    checkbox.blockSignals(False)
        for view in self._histogram_views():
            view.auto_zoom_enabled = checked
            if checked:
                view.zoom_histogram()
            else:
                view.reset_zoom()

    def _histogram_views(self):
        """Return all live presentation views (one worker/model owner)."""
        views = [self.right_histogram_view]
        detached = self._detached_histogram_window
        if detached is not None:
            views.append(detached.histogram_view)
        return views

    def _mirror_detached_view_policy(self) -> None:
        """Mirror the inline view policy onto the detached surface (PHI-R3.3).

        The inline ``HistogramView`` is the single owner of the view policy
        (window + frozen-vs-unfrozen + auto-zoom).  Whenever the detached
        surface is (re)synchronized or a model application changes the view
        state, this copies the inline policy onto the detached view so the two
        surfaces deterministically end with the same valid view range — in
        particular an unfrozen inline view never creates an artificial frozen
        range on the detached side, and a genuine manual zoom is preserved/
        reconciled identically on both (see
        :meth:`HistogramView.mirror_state_from`).
        """
        detached = self._detached_histogram_window
        if detached is not None:
            detached.histogram_view.mirror_state_from(self.right_histogram_view)

    def _zoom_histogram(self, *_ignored) -> None:
        for view in self._histogram_views():
            view.zoom_histogram()

    def _reset_histogram_view(self, *_ignored) -> None:
        for view in self._histogram_views():
            view.reset_histogram_view()

    def _reset_histogram_zoom(self, *_ignored) -> None:
        for view in self._histogram_views():
            view.reset_zoom()

    def _open_detached_histogram(self, *_ignored):
        """Show the large histogram mirror without scheduling analysis."""
        if self._detached_histogram_window is None:
            dialog = DetachedHistogramWindow(self)
            dialog.rangeChanged.connect(self._on_hist_range_changed)
            dialog.autoZoomToggled.connect(self._on_hist_auto_zoom_toggled)
            dialog.resetViewRequested.connect(self._reset_histogram_view)
            dialog.zoomRequested.connect(self._zoom_histogram)
            dialog.resetZoomRequested.connect(self._reset_histogram_zoom)
            dialog.autoStretchRequested.connect(self._on_auto_stretch)
            dialog.autoWbRequested.connect(self._on_auto_wb)
            dialog.liveAutoStretchToggled.connect(
                self._set_live_auto_stretch_enabled
            )
            dialog.liveAutoWbToggled.connect(self._set_live_auto_wb_enabled)
            dialog.set_texts(self._tr)
            self._detached_histogram_window = dialog
        dialog = self._detached_histogram_window
        self._sync_detached_histogram()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def _sync_detached_histogram(self) -> None:
        """Mirror cached model/state into the detached view with zero compute."""
        dialog = self._detached_histogram_window
        if dialog is None:
            return
        if self._pristine_float is not None:
            dialog.set_model(self._histogram_model)
            dialog.histogram_view.set_analysis_domain(
                self._bp_wp_control_upper
            )
        else:
            dialog.set_legacy_data(
                self._legacy_hist, self._legacy_hist_percentile
            )
        dialog.histogram_view.set_range(self._black_point, self._white_point)
        # PHI-R3.3 (F2): the detached surface mirrors the inline view *policy*
        # (window + frozen-vs-unfrozen + auto-zoom) — never a manufactured
        # freeze from a bare coordinate snapshot.
        self._mirror_detached_view_policy()
        dialog.auto_zoom_check.blockSignals(True)
        dialog.live_auto_stretch_check.blockSignals(True)
        dialog.live_auto_wb_check.blockSignals(True)
        try:
            dialog.auto_zoom_check.setChecked(
                self.right_histogram_view.auto_zoom_enabled
            )
            dialog.live_auto_stretch_check.setChecked(
                self._live_auto_stretch_enabled
            )
            dialog.live_auto_wb_check.setChecked(self._live_auto_wb_enabled)
        finally:
            dialog.auto_zoom_check.blockSignals(False)
            dialog.live_auto_stretch_check.blockSignals(False)
            dialog.live_auto_wb_check.blockSignals(False)
        dialog.stats_label.setText(self.right_histogram_status.text())
        dialog.set_histogram_actions_enabled(self.has_preview_image)

    def _on_bcs_changed(self, *_ignored) -> None:
        """Update brightness/contrast/saturation and re-render (display-only)."""
        self._brightness = self.brightness_spin.value()
        self._contrast = self.contrast_spin.value()
        self._saturation = self.saturation_spin.value()
        self._refresh_preview_view()

    def _on_bcs_reset(self) -> None:
        """Reset the brightness/contrast/saturation controls to neutral 1.0."""
        self.brightness_spin.setValue(DEFAULT_BRIGHTNESS)
        self.contrast_spin.setValue(DEFAULT_CONTRAST)
        self.saturation_spin.setValue(DEFAULT_SATURATION)

    def _effective_preview_downsample_factor(self) -> int:
        """Return the downsample factor to apply at preview render time.

        Single source of truth for the rendered preview resolution (M24): the
        engine pushes preview frames already downsampled by its own
        ``preview_downsample_factor`` (default 2, changed live via the M22
        control channel), so during an active run the local display-only
        downsample must NOT be applied on top — the engine factor *is* the
        displayed resolution (Tk parity: Tk renders the engine-pushed frames
        directly, no second downsample).  When idle (no run), the local
        display-only factor (M17/M18) applies unchanged.

        The factor is derived from the run state (``_running``) alone; no
        separate "engine factor" is stored, so run end restores idle behaviour
        automatically with no stale state to clear.
        """
        return 1 if self._running else self._preview_res_factor

    def _show_empty_preview(self) -> None:
        """Render the packaged ``back.png`` as the empty-preview placeholder.

        Tk ``PreviewManager._redraw_canvas`` draws ``back.png`` centred on the
        preview canvas whenever there is no astro image (and as the background
        behind one).  The Qt shell mirrors that for the empty state only: it
        scales ``back.png`` to *fit* the preview label (aspect ratio preserved,
        centred — the ``QLabel`` alignment is already ``AlignCenter``) and
        leaves the label cleared (null pixmap) when the resource is missing or
        undecodable.  Purely display — no engine / FITS / PNG write.
        """
        pixmap = load_empty_preview_pixmap()
        if pixmap is None or pixmap.isNull():
            self.preview_image_label.clear()
            return
        size = self.preview_image_label.size()
        if size.width() > 1 and size.height() > 1:
            pixmap = pixmap.scaled(
                size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        self.preview_image_label.setPixmap(pixmap)

    def _refresh_preview_view(self, *, histogram: bool = True) -> None:
        """Repaint the preview image + resolution label + histogram.

        PHI-R3.1 unit contract: an Option-A preview (float analysis state
        present) renders the visible display **from the preserved float
        analysis/WB source** — the user-selected stretch (black/white points in
        analysis units, possibly above ``1``), gamma and brightness/contrast/
        saturation are applied in float by
        :func:`preview_adjust.render_analysis_display` **before** the final
        uint8/``QImage`` conversion, which is the only clipping boundary.  A
        white point above ``1`` visibly recovers preserved headroom instead of
        leaving it white-clipped (no premature clamp at anchor mapping / WB /
        QImage ingest).  Legacy single-array payloads keep the historical
        uint8 QImage chain (:func:`preview_adjust.apply_preview_adjustments`)
        with its ``[0, 1]`` Tk-parity semantics.

        The stored source :class:`QImage` is never mutated: a fully adjusted
        *derived* image is produced, then ``render_view`` applies the current
        rotation and zoom to that derived image, so zoom reapplies cleanly
        after rotation and the original display image stays pristine.  The
        display histogram is computed from the *WB-only* analysis buffer (the
        Tk ``image_data_wb`` source), so it tracks white balance but not the
        stretch / gamma / brightness-contrast-saturation (M14 histogram-
        source alignment).  For an Option-A preview the histogram model is
        derived directly from the cached WB-only float buffer (no QImage
        round-trip) and reused unchanged on every refresh that does not change
        the WB-only source (BP/WP/stretch/gamma/BCS/zoom/pan/rotation).

        ``histogram=False`` re-renders only the pixmap + resolution label and
        skips the histogram refresh.  It is used for pure view-state
        reconciliation (e.g. ``_on_run_started`` resetting to 0°/100%/centred)
        where the unchanged source must not schedule any histogram recompute
        decision/job.
        """
        source = self._preview_source
        if source is None or source.isNull():
            self._show_empty_preview()
            self._set_view_controls_enabled(False)
            self._set_preview_controls_enabled(False)
            self.resolution_label.setText("—")
            if histogram:
                self._refresh_histogram()
            return
        if self._pristine_float is not None:
            # Option-A float display path (PHI-R3.1): derive the WB-only
            # analysis buffer (re-derives only when the source or WB changed,
            # and re-scopes the BP/WP analysis-domain controls), then render
            # the display from float with the user-selected BP/WP (analysis
            # units, possibly > 1) applied before the final uint8 conversion.
            adjusted = self._render_option_a_display()
            if adjusted is None or adjusted.isNull():
                # Defensive fallback (float state present but unusable): the
                # stored neutral QImage render, untouched by the stretch chain.
                adjusted = source
        else:
            adjusted = apply_preview_adjustments(
                source,
                wb=self._wb,
                stretch=self._stretch,
                black_point=self._black_point,
                white_point=self._white_point,
                gamma=self._gamma,
                brightness=self._brightness,
                contrast=self._contrast,
                saturation=self._saturation,
            )
            if adjusted is None or adjusted.isNull():
                adjusted = source
        zoom_text = self.zoom_combo.currentText()
        if zoom_text == "Fit":
            zoom_factor = None
        else:
            zoom_factor = self._preview_zoom_factor
        pan_offset = (self._view_offset_x, self._view_offset_y)
        display_downsample = self._effective_preview_downsample_factor()
        pixmap = render_view(
            adjusted,
            self._preview_rotation,
            zoom_text,
            self.preview_image_label.size(),
            downsample_factor=display_downsample,
            zoom_factor=zoom_factor,
            pan_offset=pan_offset,
        )
        if pixmap is None or pixmap.isNull():
            self._show_empty_preview()
            self._set_view_controls_enabled(False)
            self._set_preview_controls_enabled(False)
            self.resolution_label.setText("—")
            if histogram:
                self._refresh_histogram()
            return
        self.preview_image_label.setPixmap(pixmap)
        self._set_view_controls_enabled(True)
        self._set_preview_controls_enabled(True)
        if zoom_text == "Fit":
            disp_w, disp_h = pixmap.width(), pixmap.height()
        else:
            disp_w, disp_h = zoomed_image_size(
                adjusted,
                self._preview_rotation,
                display_downsample,
                self._preview_zoom_factor,
            )
        self.resolution_label.setText(self._resolution_text(source, disp_w, disp_h))
        if histogram:
            self._refresh_histogram()

    def _render_option_a_display(self):
        """Render the visible Option-A display from the float analysis source.

        PHI-R3.1: derives the current WB-only analysis buffer (re-derives only
        when the source or the WB gains changed, and re-scopes the BP/WP
        controls to the analysis domain), then applies the user-selected
        stretch (black/white points in analysis units, possibly above ``1``),
        gamma and brightness/contrast/saturation **in float**, with the final
        uint8/``QImage`` conversion as the only clipping boundary — a white
        point above ``1`` visibly recovers preserved headroom.  Returns the
        deep-copied ``QImage`` or ``None`` (unusable float state).  Display-
        only; never mutates the analysis buffers or any scientific state.
        """
        wb_only = self._ensure_wb_only_float()
        if wb_only is None:
            return None
        return render_analysis_display(
            wb_only,
            stretch=self._stretch,
            black_point=self._black_point,
            white_point=self._white_point,
            gamma=self._gamma,
            brightness=self._brightness,
            contrast=self._contrast,
            saturation=self._saturation,
        )

    def _resolution_text(self, source: QImage, disp_w: int, disp_h: int) -> str:
        """Build the resolution label: original → displayed size + zoom + rotation."""
        zoom = self._zoom_label()
        text = (
            f"{source.width()}x{source.height()} → "
            f"{disp_w}x{disp_h} · {zoom}"
        )
        rotation = self._preview_rotation % 360
        if rotation:
            text += f" · {rotation}°"
        return text

    def _zoom_label(self) -> str:
        """Return the language-neutral zoom label for the current view state."""
        if self.zoom_combo.currentText() == "Fit":
            return "Fit"
        return f"{int(round(self._preview_zoom_factor * 100))}%"

    def _set_view_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable the zoom + rotation view controls together."""
        self.zoom_combo.setEnabled(enabled)
        self.rotate_left_button.setEnabled(enabled)
        self.rotate_right_button.setEnabled(enabled)

    def _set_preview_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable the WB + stretch preview controls together."""
        for spin in (self.wb_r_spin, self.wb_g_spin, self.wb_b_spin):
            spin.setEnabled(enabled)
        for slider in (self.wb_r_slider, self.wb_g_slider, self.wb_b_slider):
            slider.setEnabled(enabled)
        self.auto_wb_button.setEnabled(enabled)
        self.wb_reset_button.setEnabled(enabled)
        self.stretch_combo.setEnabled(enabled)
        self.auto_stretch_button.setEnabled(enabled)
        for spin in (
            self.stretch_bp_spin,
            self.stretch_wp_spin,
            self.stretch_gamma_spin,
        ):
            spin.setEnabled(enabled)
        for slider in (
            self.stretch_bp_slider,
            self.stretch_wp_slider,
            self.stretch_gamma_slider,
        ):
            slider.setEnabled(enabled)
        self.stretch_reset_button.setEnabled(enabled)
        for spin in (self.brightness_spin, self.contrast_spin, self.saturation_spin):
            spin.setEnabled(enabled)
        for slider in (
            self.brightness_slider,
            self.contrast_slider,
            self.saturation_slider,
        ):
            slider.setEnabled(enabled)
        self.bcs_reset_button.setEnabled(enabled)
        # Histogram interaction toolbar (M14).
        # Guarded with ``hasattr``: the Preview controls tab is built before
        # the right panel that owns the histogram toolbar.
        for attr in (
            "auto_zoom_histo_check",
            "hist_reset_view_button",
            "hist_zoom_button",
            "hist_reset_button",
            "hist_expand_button",
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setEnabled(enabled)
        if self._detached_histogram_window is not None:
            self._detached_histogram_window.set_histogram_actions_enabled(enabled)

    def _refresh_histogram(self) -> None:
        """Sync the single persistent histogram surface + stats label.

        The right-panel ``HistogramView`` is the only live histogram surface
        (the M10 duplicated tab histogram was removed in M14).  For an
        Option-A preview the authoritative float model is computed from the
        cached WB-only float buffer (no QImage round-trip) and cached so that
        BP/WP/stretch/gamma/BCS/zoom/pan/rotation refreshes only re-sync the
        BP/WP markers without recomputing the histogram.  Legacy single-array
        previews keep the historical QImage ``set_data`` path.  BP/WP lines are
        re-synced to the current stretch slider values on every call.
        """
        if self._pristine_float is not None:
            self._refresh_histogram_float()
        else:
            self._refresh_histogram_legacy()

    def _refresh_histogram_float(self) -> None:
        """Option-A path: authoritative float model, computed off the GUI thread.

        A request is scheduled through the bounded latest-wins coordinator only
        when the WB-only analysis buffer was actually re-derived (a new raw
        source or a WB change).  BP/WP/stretch/gamma/BCS/zoom/pan/rotation
        refreshes only re-sync the BP/WP markers and re-render the status label
        with zero compute.  The model is applied by :meth:`_on_histogram_result`
        on the GUI thread once the worker finishes.
        """
        wb_only = self._ensure_wb_only_float()
        if wb_only is None:
            # No usable source: invalidate any in-flight/pending computation so
            # a stale result can never repopulate the cleared surface.
            self._histogram_coordinator.invalidate()
            self._histogram_model = None
            self._histogram_model_revision = None
            self._histogram_scheduled_revision = None
            self._histogram_trace_ctx = None
            for view in self._histogram_views():
                view.clear()
            self._histogram_stats = None
            self._render_histogram_status()
            return
        revision = self._wb_only_revision
        if self._histogram_scheduled_revision != revision:
            self._histogram_scheduled_revision = revision
            # Snapshot the trace context at schedule time so an async result
            # is attributed to the payload that was current when the request
            # was made — never to an obsolete/newer ``_phi_trace_ctx``.
            self._histogram_trace_ctx = dict(self._phi_trace_ctx or {})
            # The WB-only buffer is replaced (never mutated) on re-derivation,
            # so it is safe to hand to the worker by reference (read-only).
            self._histogram_coordinator.schedule(
                wb_only, source_token=(self._analysis_generation, revision)
            )
        # Re-sync the BP/WP markers immediately: they do not depend on the model
        # and must stay in lockstep with the sliders on every refresh.
        for view in self._histogram_views():
            view.set_range(self._black_point, self._white_point)
        self._render_histogram_status()

    def _on_histogram_result(self, generation: int, result, source_token) -> None:
        """GUI-thread slot: apply or discard a worker histogram result.

        The coordinator already applied the generation check (latest-wins), so
        this slot performs the *source* check as defence-in-depth: the result is
        applied only when the analysis context and the WB-only revision it was
        computed from still match the current state, and the window has not begun
        shutting down.  Otherwise it is discarded so a stale result can never
        repopulate a cleared/updated UI.  All widget/status updates happen here,
        on the GUI thread.
        """
        if self._shutting_down or self._shutdown_called:
            self._histogram_source_stale += 1
            return
        if not isinstance(source_token, tuple) or len(source_token) != 2:
            self._histogram_source_stale += 1
            return
        context, revision = source_token
        if context != self._analysis_generation or revision != self._wb_only_revision:
            self._histogram_source_stale += 1
            return
        if result is None:
            # Fail-closed: never fabricate a histogram for an unusable buffer.
            self._histogram_model = None
            self._histogram_model_revision = None
            for view in self._histogram_views():
                view.clear()
            self._histogram_stats = None
            self._render_histogram_status()
            return
        self._histogram_model = result
        self._histogram_model_revision = revision
        if phi_trace_enabled():
            # Use the ctx snapshot taken when the request was scheduled, so
            # the histogram output is attributed to the correct payload even
            # when a newer preview arrived while the worker was computing.
            ctx = getattr(self, "_histogram_trace_ctx", None) or {}
            stats = result.get("stats") or {}
            extra = {}
            for ch in result.get("channels") or []:
                ch_stats = stats.get(ch) or {}
                if ch_stats:
                    extra[f"{ch}_max"] = f"{float(ch_stats.get('max', float('nan'))):.6g}"
                    extra[f"{ch}_med"] = f"{float(ch_stats.get('median', float('nan'))):.6g}"
            phi_trace_stage(
                logger,
                route="qt",
                stage="histogram_output",
                arr=None,
                bins=int(result.get("bins", 0) or 0),
                **extra,
                **ctx,
            )
        for view in self._histogram_views():
            # Both surfaces receive the exact same authoritative model object.
            view.set_model(result)
            view.set_range(self._black_point, self._white_point)
        # PHI-R3.3 (F2): after every model application the inline view policy
        # is mirrored onto the detached surface, so a stale/artificial frozen
        # range can never survive a model-domain change on only one surface.
        self._mirror_detached_view_policy()
        stats_text = format_histogram_stats(result.get("stats"))
        # PHI-AUTO-HISTOGRAM-UX-V1: expose in-domain values above the plotted
        # bin high (sparse extreme tail) on the always-visible status line, so
        # the tail is never silently dropped in any zoom state.
        overflow_note = format_histogram_overflow(result)
        self._histogram_stats = (
            f"{stats_text}{overflow_note}" if stats_text else None
        )
        self._render_histogram_status()

    def _refresh_histogram_legacy(self) -> None:
        """Legacy single-array path: keep the historical QImage histogram.

        The histogram + stats are cached by ``(id(source), _wb)`` so a new
        source or a WB change recomputes exactly once, while BP/WP/stretch/
        gamma/BCS/zoom/pan/rotation refreshes only re-sync the BP/WP markers
        and re-render the status label with zero ``compute_histogram`` /
        ``compute_histogram_stats`` calls.
        """
        source = self._preview_source
        if source is None or source.isNull():
            self._legacy_hist_key = None
            self._legacy_hist = None
            self._legacy_hist_percentile = 1.0
            self._legacy_hist_stats = None
            for view in self._histogram_views():
                view.clear()
            self._histogram_stats = None
            self._render_histogram_status()
            return
        key = (id(source), self._wb)
        if self._legacy_hist is None or self._legacy_hist_key != key:
            wb_only = apply_preview_wb(source, wb=self._wb)
            if wb_only is None or wb_only.isNull():
                for view in self._histogram_views():
                    view.clear()
                self._histogram_stats = None
                self._render_histogram_status()
                return
            histogram = compute_histogram(wb_only, bins=256)
            percentile = compute_histogram_percentile(wb_only, 99.5)
            self._legacy_hist = histogram
            self._legacy_hist_percentile = 1.0 if percentile is None else percentile
            self._legacy_hist_stats = compute_histogram_stats(wb_only)
            self._legacy_hist_key = key
            for view in self._histogram_views():
                view.set_legacy_data(
                    self._legacy_hist, self._legacy_hist_percentile
                )
            # PHI-R3.3 (F3/F2): model→legacy transitions clear any frozen
            # float-model view state on every surface (set_legacy_data) and the
            # inline policy is mirrored to the detached view — both end on a
            # valid legacy [0, 1] window without a manual reset.
            self._mirror_detached_view_policy()
        for view in self._histogram_views():
            view.set_range(self._black_point, self._white_point)
        self._histogram_stats = self._legacy_hist_stats
        self._render_histogram_status()

    def _render_histogram_status(self) -> None:
        """Render the single histogram status label from stored stats + language."""
        if self._histogram_stats is None:
            text = self._tr("histogram_empty")
        else:
            text = f"{self._tr('histogram_stats')} {self._histogram_stats}"
        self.right_histogram_status.setText(text)
        if self._detached_histogram_window is not None:
            self._detached_histogram_window.stats_label.setText(text)

    # ------------------------------------------------------ run-log helpers
    def _run_log_emit(self, event: str, **fields) -> None:
        """Emit one durable lifecycle event to the shared run log (if any)."""
        run_log = self.controller.run_log
        if run_log is not None:
            run_log.emit(event, **fields)

    def _format_refusal(self, payload) -> tuple:
        """Map a structured refusal payload to a localized (title, body) pair.

        The known startup-refusal codes are mapped through the existing
        localization architecture; any unknown code falls back to a generic
        (English) refusal so generic false starts stay generic.  The technical
        detail is never used as the primary presentation text.
        """
        code = getattr(payload, "code", None)
        key_by_code = {
            "OUTPUT_STATE_INCOMPATIBLE": "startup_refusal_output_state_incompatible",
            "FRESH_OUTPUT_HAS_STATE": "startup_refusal_fresh_output_has_state",
            "RESUME_STATE_MISSING": "startup_refusal_resume_state_missing",
            "RESUME_MODE_UNSUPPORTED": "startup_refusal_resume_mode_unsupported",
        }
        key = key_by_code.get(code)
        if key:
            title = self._tr(f"{key}_title")
            body = self._tr(f"{key}_body")
            return title, body
        detail = getattr(payload, "technical_detail", "") or str(code)
        return "Cannot start run", f"Cannot start run: {detail}"

    def _on_run_refused(self, payload) -> None:
        """Handle a structured startup refusal (non-blocking, localized)."""
        title, body = self._format_refusal(payload)
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage(title)
        self.log(body)
        # The status bar and run log keep the engine's precise technical
        # reason (never parsed, never the dialog's primary text); the owned
        # Warning box presents only the localized actionable guidance.
        detail = getattr(payload, "technical_detail", "") or ""
        if detail:
            self.log(detail)
        self._show_error_box(title, body, severity="warning")
        self._mark_time_terminal("failed")

    def _on_run_finished(self) -> None:
        """Run-end GUI completion slot (summary dialog only — histogram lifecycle).

        FRP-H1 end-of-run lifecycle: this handler only shows the textual
        processing summary via :meth:`_show_pending_summary` →
        :meth:`_show_summary_dialog` (a pure text dialog that never reads the
        saved FITS or the engine).  It does NOT load a final reconstructed
        image into ``_preview_source``/``_pristine_float`` and no histogram is
        re-derived here — so the final visible histogram represents the **last
        accepted live preview**, not the saved FITS on disk (no FITS readback,
        by design).
        """
        self._run_log_emit("QT_COMPLETION_HANDLER_ENTERED", outcome="finished")
        terminal_status = derive_terminal_status(self._last_summary_payload)
        self._running = False
        self._update_run_state()
        self.progress.setValue(100)
        if terminal_status == "success":
            self.statusBar().showMessage("Finished.")
            self.log("Run finished.")
        else:
            self.statusBar().showMessage(
                self._tr("run_finished_empty", default="No output produced.")
            )
            self.log(
                self._tr(
                    "run_finished_empty_log",
                    default="Run finished with no output.",
                )
            )
        self._mark_time_terminal("0:00")
        self._show_pending_summary()
        self._run_log_emit("CONTROLS_RESTORED")
        self._run_log_emit("GUI_IDLE")
        if terminal_status == "success":
            self._run_log_emit("RUN_SUCCEEDED", terminal_status="success")
        else:
            self._run_log_emit("RUN_FINISHED_NO_OUTPUT", terminal_status="empty")
        # QT_COMPLETION_HANDLER_RETURNED and the log close are owned by the
        # controller, immediately after this slot (the public terminal signal
        # emit) returns — so RETURNED is only written once this handler has
        # actually returned, never before a blocking close/tail.

    def _on_run_failed(self, message: str) -> None:
        self._run_log_emit("QT_COMPLETION_HANDLER_ENTERED", outcome="failed")
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage(f"Failed: {message}")
        self.log(f"Run failed: {message}")
        self._show_error_box(
            self._tr("error_box_run_failed_title", default="Run failed"),
            message,
            severity="critical",
        )
        self._mark_time_terminal("failed")
        self._run_log_emit("CONTROLS_RESTORED")
        self._run_log_emit("GUI_IDLE")
        self._run_log_emit("RUN_FAILED", error=message)
        # Controller-owned RETURNED + close (after this slot returns).

    def _on_run_cancelled(self) -> None:
        self._run_log_emit("QT_COMPLETION_HANDLER_ENTERED", outcome="cancelled")
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage("Cancelled.")
        self.log("Run cancelled.")
        self._mark_time_terminal("cancelled")
        self._run_log_emit("CONTROLS_RESTORED")
        self._run_log_emit("GUI_IDLE")
        self._run_log_emit("RUN_CANCELLED")
        # Controller-owned RETURNED + close (after this slot returns).

    # --------------------------------------------------- run summary (M23)
    def _on_summary_payload(self, payload: SummaryPayload) -> None:
        """Store the terminal summary payload until the run-end slot consumes it.

        The backend/boring runner emits the payload just before its terminal
        ``finished`` signal, so the payload always arrives first and the
        corresponding run-end slot can show it with the run already marked
        finished.
        """
        self._last_summary_payload = payload

    def _show_pending_summary(self) -> None:
        """Show the stored run summary (if any) and clear it for the next run."""
        payload = self._last_summary_payload
        if payload is None:
            return
        self._last_summary_payload = None
        self._show_summary_dialog(payload)

    def _derived_status_label(self, payload: SummaryPayload) -> str:
        """Return the localized terminal-status label shown by the summary dialog.

        Derives ``"SUCCESS"`` vs ``"EMPTY/NO OUTPUT"`` from the payload via
        :func:`~seestar.gui_qt.summary_payload.derive_terminal_status`, leaving
        ``payload.status`` free to keep the backend's raw ``"finished"`` label.
        """
        if derive_terminal_status(payload) == "success":
            return self._tr("summary_status_success", default="SUCCESS")
        return self._tr("summary_status_empty", default="EMPTY/NO OUTPUT")

    def _format_summary_text(self, payload: SummaryPayload) -> str:
        """Render the summary payload into the plain text shown by the dialog."""
        lines = [
            f"{self._tr('summary_status', default='Status')}: "
            f"{self._derived_status_label(payload)}",
            (
                f"{self._tr('summary_total_time', default='Total Processing Time')}: "
                f"{format_duration(payload.duration_seconds)}"
            ),
        ]
        files = payload.files_attempted
        lines.append(
            f"{self._tr('summary_files_attempted', default='Files Attempted')}: "
            f"{files if files is not None else '?'}"
        )
        if payload.final_stack_file:
            not_found = (
                ""
                if payload.final_stack_exists
                else f" ({self._tr('summary_not_found', default='Not Found!')})"
            )
            lines.append(
                f"{self._tr('summary_final_stack_file', default='Final Stack File')}: "
                f"{payload.final_stack_file}{not_found}"
            )
        if payload.images_in_final_stack is not None:
            lines.append(
                f"{self._tr('summary_images_in_final_stack', default='Images in Final Stack')}: "
                f"{payload.images_in_final_stack}"
            )
        if payload.total_exposure_seconds is not None:
            lines.append(
                f"{self._tr('summary_total_exposure', default='Total Exposure (Final Stack)')}: "
                f"{format_duration(payload.total_exposure_seconds)}"
            )
        return "\n".join(lines)

    def _show_summary_dialog(self, payload: SummaryPayload) -> None:
        """Show a non-modal processing-summary dialog (Tk parity, Qt-styled).

        The dialog only *formats* the payload computed by the backend/boring
        runner; it never reads ``final.fits`` or the engine.  "Open Output" is
        offered only when the output folder holds a final stack.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle(
            self._tr("processing_report_title", default="Processing Summary")
        )
        layout = QVBoxLayout(dialog)
        label = QLabel(self._format_summary_text(payload))
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        if payload.can_open_output:
            open_button = buttons.addButton(
                self._tr("open_output", default="Open Output"),
                QDialogButtonBox.ButtonRole.ActionRole,
            )
            open_button.clicked.connect(self._open_output_folder)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        self._summary_dialog = dialog
        dialog.show()

    # --------------------------------------------- terminal-failure surface
    def _show_error_box(
        self, title: str, body: str, severity: str = "critical"
    ) -> QMessageBox:
        """Present ``title``/``body`` through one owned, non-blocking QMessageBox.

        The box is a genuine :class:`QMessageBox` parented to the window and
        retained on ``self._error_message_box`` so it can never be
        garbage-collected.  A new failure **reuses the same box** (replacing the
        previous title/body) so repeated signals never pile boxes up; the
        previous content is deterministically replaced, not appended.

        Presentation is non-blocking by construction: the box is shown via
        :meth:`QWidget.show` with window modality (never ``exec()`` and never a
        static ``QMessageBox.critical/warning/information``), so this method
        returns immediately and never blocks the controller-owned terminal
        cleanup that runs immediately after the failure/refusal signal handler
        returns.

        ``severity`` is ``"warning"`` or ``"critical"`` (default; anything else
        is treated as critical).  The body is always rendered as **plain text**
        so error detail is never interpreted as (possibly malformed) rich text.
        """
        box = self._error_message_box
        if box is None:
            box = QMessageBox(self)
            box.setWindowModality(Qt.WindowModality.WindowModal)
            self._error_message_box = box
        box.setIcon(
            QMessageBox.Icon.Warning
            if severity == "warning"
            else QMessageBox.Icon.Critical
        )
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setWindowTitle(title)
        box.setText(body)
        self._error_box_count += 1
        box.show()
        box.raise_()
        return box

    @property
    def error_message_box(self) -> Optional[QMessageBox]:
        """The currently owned terminal-failure box (None when none shown)."""
        return self._error_message_box

    @property
    def error_box_count(self) -> int:
        """Number of times the owned failure box was presented (test seam)."""
        return self._error_box_count

    def _update_run_state(self) -> None:
        self.start_button.setEnabled(not self._running)
        self.stop_button.setEnabled(self._running)
        self._update_path_action_state()

    def _update_path_action_state(self) -> None:
        """Enable/disable path actions based on current paths and run state.

        * View Inputs  — enabled when the input folder is an existing directory,
        * Open Output  — enabled when the output folder is an existing directory,
        * Add Folder   — enabled pre-run when the input folder is valid; disabled
          while a run is active (live-add is not exposed by ``RunController``).
        * Analyse      — enabled when the input folder is an existing directory
          (user-triggered ZeAnalyser launch; independent of run state).
        """
        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()
        input_valid = bool(input_text) and os.path.isdir(input_text)
        output_valid = bool(output_text) and os.path.isdir(output_text)
        self.view_inputs_button.setEnabled(input_valid)
        self.open_output_button.setEnabled(output_valid)
        self.add_folder_button.setEnabled(input_valid and not self._running)
        self.analyse_button.setEnabled(input_valid)

    # ------------------------------------------------- settings collection
    def _sync_state_from_controls(self, *_ignored) -> None:
        """Copy the current widget values into the settings model.

        Accepts (and ignores) any signal payload so it can be connected to
        ``textChanged``/``valueChanged``/``currentIndexChanged``/``stateChanged``
        directly.
        """
        state = self.settings_state
        state.language = self._language
        state.theme = self._theme
        state.input_folder = self.input_edit.text()
        state.output_folder = self.output_edit.text()
        state.temp_folder = self.temp_edit.text()
        state.output_filename = self.output_filename_edit.text()
        state.reference_image_path = self.reference_edit.text()
        state.last_stack_path = self.last_stack_edit.text()
        state.batch_size = self.batch_spin.value()
        state.stacking_mode = self.stacking_mode_combo.currentText()
        state.use_drizzle = self.drizzle_check.isChecked()
        state.drizzle_mode = self.drizzle_mode_combo.currentData()
        state.drizzle_group_size = self.drizzle_group_spin.value()
        state.drizzle_scale = self.drizzle_scale_spin.value()
        state.drizzle_wht_threshold = self.drizzle_wht_spin.value()
        state.drizzle_kernel = self.drizzle_kernel_combo.currentText()
        state.drizzle_pixfrac = self.drizzle_pixfrac_spin.value()
        state.use_gpu = self.use_gpu_check.isChecked()
        state.max_hq_mem_gb = float(self.max_hq_mem_spin.value())
        state.local_solver_preference = self.solver_combo.currentText()

        # Final-combination business control drives the derived reproject flags
        # (exactly like Tk SettingsManager.update_from_ui).  There is no
        # separate user-facing checkbox for these flags, so the two cannot fall
        # out of sync.
        key = self._final_combine_key()
        state.stack_final_combine = key
        state.reproject_between_batches, state.reproject_coadd_final = (
            final_combine_flags(key)
        )

        # Settings tab controls (scalar fields).
        for attr in self._settings_widgets:
            setattr(state, attr, self._read_settings_widget_value(attr))

        # Mosaic nested dict.
        self._sync_mosaic_settings(state)

        # Path action buttons react to path edits immediately (Tk parity).
        self._update_path_action_state()

    def _final_combine_key(self) -> str:
        """Return the current final-combine backend key from the combo label."""
        label = self.final_combine_combo.currentText()
        return FINAL_COMBINE_LABEL_TO_KEY.get(label, "mean")

    def resolve_backend(self) -> Optional[BaseRunBackend]:
        """Return the run backend for a Start, or ``None`` for the default.

        The default (``backend_factory is None`` and
        ``backend_mode == "simulated"``) returns ``None`` so
        :meth:`RunController.start` falls back to its built-in
        :class:`SimulatedRunBackend`.  A non-default selection produces an
        explicit backend:

        * ``backend_factory`` (when set) is called to produce a fresh backend,
        * ``backend_mode == "seestar"`` lazily constructs a
          :class:`SeestarQueuedStackerBackend` — the real engine is imported
          only when that backend's ``run()`` is invoked, never here.
        """
        if self.backend_factory is not None:
            return self.backend_factory()
        if self.backend_mode == "seestar":
            return SeestarQueuedStackerBackend()
        return None

    def collect_settings_state(self) -> QtSettingsState:
        """Return the settings model, freshly synced from the current widgets.

        This is the single choke point between Qt widgets and the run-request
        builder: it reads *plain* values only, never widgets, so downstream code
        (and tests) never need a live Qt object.
        """
        self._sync_state_from_controls()
        return self.settings_state

    # -------------------------------------------------- settings persistence
    def _set_settings_widget_value(self, kind: str, widget, value) -> None:
        """Write a plain Python value into a settings widget by kind.

        The inverse of :meth:`_widget_value`; used to apply a persisted settings
        model to the Settings-tab / Mosaic widgets.
        """
        if kind == "bool":
            widget.setChecked(bool(value))
        elif kind == "int":
            widget.setValue(int(float(value)))
        elif kind == "float":
            widget.setValue(float(value))
        elif kind == "str":
            widget.setText("" if value is None else str(value))
        elif kind == "combo":
            text = str(value)
            if text in [widget.itemText(i) for i in range(widget.count())]:
                widget.setCurrentText(text)
        elif kind == "list":
            widget.setText(", ".join(str(x) for x in value) if value else "")
        elif kind == "match_bg":
            widget.setCurrentText(MATCH_BG_TO_TEXT.get(value, "default"))
        else:
            raise ValueError(f"unknown settings field kind {kind!r}")

    def _apply_state_to_controls(self, state: QtSettingsState) -> None:
        """Apply a settings model to the visible Qt controls (M8).

        Widgets that cannot represent a persisted value (e.g. a combo choice no
        longer in the vocabulary, or a numeric value clamped by a spinbox) keep
        their representable value, and the trailing sync folds the constrained
        widget values back into the model — so a legacy/corrupt value can never
        leave the UI/model inconsistent.  ``stack_final_combine`` remains the
        single source of truth: setting the final-combine combo re-derives the
        two reproject flags exactly like a user edit.
        """
        widgets = [
            self.input_edit,
            self.output_edit,
            self.temp_edit,
            self.output_filename_edit,
            self.reference_edit,
            self.last_stack_edit,
            self.batch_spin,
            self.stacking_mode_combo,
            self.final_combine_combo,
            self.max_hq_mem_spin,
            self.drizzle_check,
            self.drizzle_mode_combo,
            self.drizzle_group_spin,
            self.drizzle_scale_spin,
            self.drizzle_wht_spin,
            self.drizzle_kernel_combo,
            self.drizzle_pixfrac_spin,
            self.use_gpu_check,
            self.solver_combo,
            self.mosaic_active_check,
        ]
        widgets.extend(self._settings_widgets.values())
        widgets.extend(w for _kind, w in self._mosaic_widgets.values())
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.input_edit.setText(state.input_folder or "")
            self.output_edit.setText(state.output_folder or "")
            self.temp_edit.setText(state.temp_folder or "")
            self.output_filename_edit.setText(state.output_filename or "")
            self.reference_edit.setText(state.reference_image_path or "")
            self.last_stack_edit.setText(state.last_stack_path or "")
            self.batch_spin.setValue(int(state.batch_size))
            if state.stacking_mode in STACKING_MODES:
                self.stacking_mode_combo.setCurrentText(state.stacking_mode)
            label = FINAL_COMBINE_LABELS.get(state.stack_final_combine)
            if label is not None:
                self.final_combine_combo.setCurrentText(label)
            self.max_hq_mem_spin.setValue(int(state.max_hq_mem_gb))
            self.drizzle_check.setChecked(bool(state.use_drizzle))
            if state.drizzle_mode in DRIZZLE_MODES:
                self.drizzle_mode_combo.setCurrentIndex(
                    DRIZZLE_MODES.index(state.drizzle_mode)
                )
            self.drizzle_group_spin.setValue(int(state.drizzle_group_size))
            self.drizzle_scale_spin.setValue(int(state.drizzle_scale))
            self.drizzle_wht_spin.setValue(float(state.drizzle_wht_threshold))
            if state.drizzle_kernel in DRIZZLE_KERNELS:
                self.drizzle_kernel_combo.setCurrentText(state.drizzle_kernel)
            self.drizzle_pixfrac_spin.setValue(float(state.drizzle_pixfrac))
            self.use_gpu_check.setChecked(bool(state.use_gpu))
            if state.local_solver_preference in SOLVER_PREFERENCES:
                self.solver_combo.setCurrentText(state.local_solver_preference)

            for attr, widget in self._settings_widgets.items():
                self._set_settings_widget_value(
                    self._settings_kinds[attr], widget, getattr(state, attr)
                )
            self.mosaic_active_check.setChecked(bool(state.mosaic_mode_active))
            ms = state.mosaic_settings if isinstance(state.mosaic_settings, dict) else {}
            for key, (kind, widget) in self._mosaic_widgets.items():
                self._set_settings_widget_value(kind, widget, ms.get(key))
        finally:
            for widget in widgets:
                widget.blockSignals(False)

        # Reconcile the boring-thread toggle + gating with the loaded batch
        # size, then fold the (possibly constrained) widget values back into the
        # model and refresh path-action enablement.
        self.boring_check.setChecked(self.batch_spin.value() == 1)
        self._update_boring_gating()
        self.settings_state = state
        # Apply the persisted UI language (M9).  ``state.language`` is already
        # normalised by ``QtSettingsState.from_dict``; ``_set_language`` is
        # idempotent for the default English path.
        self._set_language(getattr(state, "language", "en"))
        # Apply the persisted UI theme (M25.5-C).  ``state.theme`` is already
        # normalised by ``QtSettingsState.from_dict``; ``_set_theme`` applies
        # the palette immediately (and is idempotent for the default System).
        self._set_theme(getattr(state, "theme", DEFAULT_THEME))
        self._update_expert_enabler_states()
        self._toggle_kappa_visibility()
        self._sync_state_from_controls()
        # Tk parity: a persisted last-stack path pre-fills the output folder
        # when the output folder is empty (Tk ``_on_last_stack_changed`` fires
        # on the persisted write through its StringVar trace).
        self._on_last_stack_changed()
        self._update_path_action_state()

    def _load_persisted_settings(self) -> None:
        """Load persisted settings + geometry into the window (best-effort).

        A missing or corrupt file yields the code defaults; an unknown value for
        a known field is coerced by ``QtSettingsState.from_dict`` and an unknown
        combo choice degrades to the widget's current value.  Never raises.
        """
        data = settings_persistence.load_settings_json(self._settings_path)
        data, migrated = settings_persistence.migrate_settings_data(data)
        if migrated and self._settings_path:
            if not settings_persistence.save_settings_json(self._settings_path, data):
                self.log(
                    "Could not persist the legacy Feathering settings migration "
                    f"to {self._settings_path}"
                )
        state = QtSettingsState.from_dict(data)
        self._apply_state_to_controls(state)
        # Tk parity: a folder restored from settings auto-loads its first FITS
        # for an immediate preview (guarded against redundant reloads).
        self._try_show_first_input_image()
        geometry = data.get("window_geometry")
        if isinstance(geometry, str) and geometry:
            self._restore_geometry_from_key(geometry)

    def _save_persisted_settings(self) -> None:
        """Save the current settings + geometry to the injected JSON path.

        Best-effort: a write failure is swallowed by the helper and surfaced via
        the log.  Geometry is stored as base64 of ``saveGeometry()`` so it is
        both Qt-safe and JSON-safe.
        """
        if not self._settings_path:
            return
        self._sync_state_from_controls()
        data = self.settings_state.to_dict()
        data[settings_persistence.SETTINGS_SCHEMA_VERSION_KEY] = (
            settings_persistence.CURRENT_SETTINGS_SCHEMA_VERSION
        )
        data["window_geometry"] = self._geometry_to_key()
        if not settings_persistence.save_settings_json(self._settings_path, data):
            self.log(f"Could not save settings to {self._settings_path}")

    def _geometry_to_key(self) -> str:
        """Return the current window geometry as a base64 JSON-safe string."""
        return base64.b64encode(bytes(self.saveGeometry())).decode("ascii")

    def _restore_geometry_from_key(self, value: str) -> bool:
        """Restore window geometry from a base64 string; never raises."""
        try:
            return bool(
                self.restoreGeometry(QByteArray.fromBase64(value.encode("ascii")))
            )
        except (ValueError, TypeError):
            return False

    def _effective_settings_state(self) -> QtSettingsState:
        """Return a settings snapshot with batch-size normalization applied.

        Normalisation (UI ``0`` -> Auto sentinel ``-1``, or the special
        batch-zero mode when ``reproject_coadd_final`` is set) is applied to a
        *copy* of the model whenever it changes the value, so the shared
        ``self.settings_state`` keeps the raw UI value and the special mode is
        never lost across widget edits.
        """
        state = self.collect_settings_state()
        normalized = normalize_batch_size(
            state.batch_size, bool(state.reproject_coadd_final)
        )
        if normalized == state.batch_size:
            return state
        return replace(state, batch_size=normalized)

    def build_run_request(
        self,
        *,
        initial_additional_folders: Optional[List[str]] = None,
        auto_chunk_size: Optional[int] = None,
        special_single: bool = False,
    ) -> RunRequest:
        """Build an immutable :class:`RunRequest` from the current settings.

        This does **not** start the backend: it only collects the visible
        controls into a :class:`QtSettingsState` and forwards it to the
        Qt/Tk-independent ``run_config.build_run_request``, then attaches the
        Qt-collected seam settings (``use_gpu`` / ``max_hq_mem_gb``) that the
        canonical builder intentionally does not emit (M20).  The canonical
        builder's output is unchanged, so the Tk flow stays byte-identical.
        """
        state = self._effective_settings_state()
        request = _build_run_request(
            state,
            initial_additional_folders=initial_additional_folders,
            auto_chunk_size=auto_chunk_size,
            special_single=special_single,
        )
        return attach_run_settings(
            request,
            use_gpu=bool(state.use_gpu),
            max_hq_mem_gb=float(state.max_hq_mem_gb),
            reference_origin_hint=self._reference_origin_hint,
        )

    # ------------------------------------------- progress/log time + copy
    def _now(self) -> float:
        """Return the current time from the injected (or monotonic) clock."""
        return self._clock()

    def _elapsed_seconds(self) -> Optional[float]:
        """Elapsed seconds since the current run started (None when idle)."""
        if self._run_started_at is None:
            return None
        return max(0.0, self._now() - self._run_started_at)

    def _set_elapsed_label(self, seconds: Optional[float]) -> None:
        self._last_elapsed_seconds = seconds
        self._render_elapsed_label()

    def _set_remaining_label(self, text: str) -> None:
        self._last_remaining_text = text
        self._render_remaining_label()

    def _render_elapsed_label(self) -> None:
        """Render the elapsed label from its stored raw value + active language."""
        self.elapsed_label.setText(
            f"{self._tr('elapsed')} {format_duration(self._last_elapsed_seconds)}"
        )

    def _render_remaining_label(self) -> None:
        """Render the remaining label from its stored raw value + active language."""
        self.remaining_label.setText(
            f"{self._tr('remaining')} {self._last_remaining_text}"
        )

    def _refresh_time_surface(self, percent: Optional[int]) -> None:
        """Update the elapsed/remaining labels from elapsed time + percent.

        ``percent`` is ``None`` when unknown (run start, or a boring run with
        no percent progress); ``0`` renders remaining as unknown (no division
        by zero); ``>= 100`` renders remaining as ``0:00`` (done).
        """
        elapsed = self._elapsed_seconds()
        self._set_elapsed_label(elapsed)
        if percent is None:
            self._set_remaining_label(UNKNOWN)
            return
        p = int(percent)
        if p >= 100:
            self._set_remaining_label("0:00")
            return
        if p <= 0:
            self._set_remaining_label(UNKNOWN)
            return
        remaining = estimate_remaining_seconds(
            elapsed if elapsed is not None else 0.0, p
        )
        self._set_remaining_label(format_duration(remaining))

    def _mark_time_terminal(self, state: str) -> None:
        """Show final elapsed time and a terminal remaining state.

        Used on finish (``"0:00"``), failure (``"failed"``) and cancellation
        (``"cancelled"``): elapsed stays visible and the remaining label never
        shows a misleading estimate after a terminal outcome.
        """
        self._set_elapsed_label(self._elapsed_seconds())
        self._set_remaining_label(state)

    def _copy_log_to_clipboard(self) -> None:
        """Copy the full plain-text log to the system clipboard (M6).

        Uses only Qt clipboard APIs, copies the *entire* log verbatim, and
        never mutates the log view or the run state.
        """
        QApplication.clipboard().setText(self.log_view.toPlainText())

    # -------------------------------------------------------- localization
    def _tr(self, key: str, default: Optional[str] = None) -> str:
        """Return the visible string for ``key`` in the active language."""
        return localization.translate(key, self._language, default=default)

    def _bind_text(self, widget, key: str) -> None:
        """Track ``widget`` so ``_refresh_language`` can re-apply its label."""
        self._text_bindings.append((widget, key))

    def _add_form_row(self, form: QFormLayout, key: str, widget: QWidget) -> None:
        """Add a localized ``label + widget`` row to a form and track the label."""
        label = QLabel(self._tr(key))
        self._bind_text(label, key)
        form.addRow(label, widget)

    def _on_language_changed(self, _index: Optional[int] = None) -> None:
        """Handle a user language-combo change (M9)."""
        code = self._current_language_code()
        if code != self._language:
            self._set_language(code)

    def _set_language(self, code: Optional[str]) -> None:
        """Activate ``code`` (normalized), sync the combo/model and refresh labels."""
        self._language = localization.normalize_language(code)
        self.settings_state.language = self._language
        label = localization.language_label_for(self._language)
        if self.language_combo.currentText() != label:
            self.language_combo.blockSignals(True)
            try:
                self.language_combo.setCurrentText(label)
            finally:
                self.language_combo.blockSignals(False)
        self._refresh_language()

    def _on_theme_changed(self, index: Optional[int] = None) -> None:
        """Handle a user theme-combo change (M25.5-C)."""
        if index is None or index < 0 or index >= len(THEME_MODES):
            return
        mode = THEME_MODES[index]
        if mode != self._theme:
            self._set_theme(mode)

    def _set_theme(self, mode: Optional[str]) -> None:
        """Activate ``mode`` (normalized), sync the combo/model and apply palette.

        Presentation-only: never touches widget behaviour, layout logic, the
        engine or the Tk GUI.  ``system`` restores the platform/style default
        palette; ``dark`` / ``light`` apply a compact Qt ``QPalette``.
        """
        if mode not in THEME_MODES:
            mode = DEFAULT_THEME
        self._theme = mode
        self.settings_state.theme = mode
        idx = THEME_MODES.index(mode)
        if self.theme_combo.currentIndex() != idx:
            self.theme_combo.blockSignals(True)
            try:
                self.theme_combo.setCurrentIndex(idx)
            finally:
                self.theme_combo.blockSignals(False)
        self._apply_theme_palette(mode)

    def _apply_theme_palette(self, mode: str) -> None:
        """Apply the palette for ``mode`` to the running application.

        ``system`` re-reads the style's standard palette (i.e. follows the
        platform), so toggling back from dark/light restores the original
        platform look.  Never raises; a missing ``QApplication`` is a no-op.
        """
        app = QApplication.instance()
        if app is None:
            return
        if mode == "system":
            app.setPalette(app.style().standardPalette())
        elif mode == "dark":
            app.setPalette(_dark_palette())
        elif mode == "light":
            app.setPalette(_light_palette())

    @property
    def theme_mode(self) -> str:
        """The active presentation theme mode (``system``/``dark``/``light``)."""
        return self._theme

    def _refresh_theme_combo(self) -> None:
        """Re-localize the theme combo items without changing the selection.

        The combo order is fixed to :data:`THEME_MODES`, so each item is
        re-labelled in place; ``setItemText`` never changes the current index
        (and therefore never fires ``_on_theme_changed``).
        """
        for i, mode in enumerate(THEME_MODES):
            self.theme_combo.setItemText(i, self._tr(THEME_CHOICE_KEYS[mode]))

    def _refresh_drizzle_mode_combo(self) -> None:
        """Re-localize the drizzle mode combo items without touching item data.

        The combo order is fixed to :data:`DRIZZLE_MODES`, so each item label is
        re-set in place; ``setItemText`` never changes the item's data (the
        backend value ``Final`` / ``Incremental``) nor the current index, so it
        can never fire ``_sync_state_from_controls`` or ``_update_drizzle_gating``.
        """
        for i, label_key in enumerate(DRIZZLE_MODE_LABEL_KEYS):
            self.drizzle_mode_combo.setItemText(i, self._tr(label_key))

    def _refresh_language(self) -> None:
        """Re-apply the active language to every localized visible string.

        Runs without rebuilding the window: static labels/buttons/group titles
        are re-set from the stored ``(widget, key)`` bindings, the tab labels
        are re-labelled by index, and the dynamic elapsed/remaining/preview
        labels are re-rendered from their stored raw values.
        """
        for widget, key in self._text_bindings:
            if isinstance(widget, QGroupBox):
                widget.setTitle(self._tr(key))
            else:
                widget.setText(self._tr(key))
        for tab_widget, key in (
            (self._stacking_tab, "tab_stacking"),
            (self._settings_tab, "tab_expert"),
            (self._system_tab, "tab_system"),
            (self._preview_controls_tab, "tab_preview_controls"),
        ):
            idx = self.tabs.indexOf(tab_widget)
            if idx >= 0:
                self.tabs.setTabText(idx, self._tr(key))
        self._refresh_theme_combo()
        self._refresh_drizzle_mode_combo()
        self._render_elapsed_label()
        self._render_remaining_label()
        self._render_preview_label()
        self._render_histogram_status()
        self._render_preview_res_button()
        if self._detached_histogram_window is not None:
            self._detached_histogram_window.set_texts(self._tr)
        if hasattr(self, "backend_notice_label"):
            self.backend_notice_label.setText(self._backend_notice_text())
        self._update_drizzle_gating()

    def _render_preview_label(self) -> None:
        """Render the preview metadata label from its stored detail + language."""
        self.preview_label.setText(
            f"{self._tr('preview_prefix')} {self._preview_detail}"
        )

    # ------------------------------------------------------------- helpers
    def log(self, message: str) -> None:
        """Append a line to the read-only log tab and arm Copy Log."""
        self.log_view.append(message)
        self.copy_log_button.setEnabled(True)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def has_preview_image(self) -> bool:
        """True while a renderable preview source image is stored."""
        return self._preview_source is not None and not self._preview_source.isNull()

    @property
    def preview_rotation(self) -> int:
        """Accumulated preview rotation in clockwise degrees (0/90/180/270)."""
        return self._preview_rotation

    @property
    def shutdown_called(self) -> bool:
        return self._shutdown_called

    @property
    def _histogram_compute_count(self) -> int:
        """Option-A histogram recompute *decisions* (generations requested).

        Each real WB/source revision schedules exactly one request; this mirrors
        the bounded coordinator's ``requests_scheduled`` counter so the H1
        recompute-trigger tests keep their single "compute count" seam while the
        actual computation now runs off the GUI thread (and rapid revisions are
        coalesced into ``jobs_started`` worker invocations).
        """
        return self._histogram_coordinator.requests_scheduled

    @property
    def live_auto_stretch_count(self) -> int:
        """Per-run completed-batch Auto Stretch invocation count."""
        return self._live_auto_stretch_count

    @property
    def live_auto_wb_count(self) -> int:
        """Per-run completed-batch AutoWB invocation count."""
        return self._live_auto_wb_count

    # ---- ZSSS-OTPUX-STABLE-A instrumentation (read-only witness seams) ----
    @property
    def preview_mode(self):
        """Truthful engine renderer route label (``classic``/``reproject``/``drizzle``).

        A route observation of the last displayed preview's ``PREV_SRC`` header,
        *not* a scientific-preview identity dimension (Classic and Reproject
        share the ``"batch"`` identity family keyed by ``current_batch``).
        """
        return self._preview_mode

    @property
    def preview_identity(self):
        """Full identity ``(run_context_id, family, counter)`` of the last
        ingested scientific preview, or ``None``."""
        return self._preview_identity

    @property
    def displayed_identity(self):
        """Full identity ``(run_context_id, family, counter)`` of the currently
        displayed scientific preview, or ``None``."""
        return self._displayed_identity

    @property
    def raw_revision(self) -> int:
        """Monotonic counter of new displayed scientific previews."""
        return self._raw_revision

    @property
    def wb_revision(self) -> int:
        """Monotonic counter of WB-only analysis-buffer re-derivations."""
        return self._wb_only_revision

    @property
    def live_target_identity(self):
        """Full identity last targeted by live auto, or ``None``."""
        return self._last_live_auto_batch_token

    @property
    def live_bp_wp(self):
        """BP/WP pair last written by live auto stretch, or ``(None, None)``."""
        return (self._live_bp, self._live_wp)

    def shutdown(self, wait_ms: Optional[int] = None) -> bool:
        """Idempotent teardown hook.

        Called automatically by :meth:`closeEvent` and safe to call directly
        (e.g. from an application-level ``aboutToQuit`` handler).  It requests
        stop of any live run, waits for the worker QThread to finish, and
        resets UI state.  Safe to call multiple times.

        Returns ``True`` when the window is fully shut down (no live worker
        thread remains) and ``False`` when the controller is still stopping:
        in that case the controller/thread references are intentionally
        retained, completion is **not** recorded, and a later call (or
        :meth:`closeEvent`) retries the teardown.  ``wait_ms`` overrides the
        window's default shutdown timeout (``shutdown_wait_ms``).
        """
        if wait_ms is None:
            wait_ms = self._shutdown_wait_ms
        # Mark teardown as begun so an in-flight histogram result can never
        # touch the UI during/after close (belt-and-suspenders on top of the
        # coordinator's own invalidation + disconnect).
        self._shutting_down = True
        if self._detached_histogram_window is not None:
            self._detached_histogram_window.hide()
        # Close/hide any outstanding terminal-failure box so a normal window
        # close never leaves a stale window-modal dialog behind.
        if self._error_message_box is not None:
            try:
                self._error_message_box.close()
            except Exception:
                pass
        # Stop the ZeAnalyser reference-return watcher so a closing window never
        # leaves a live QTimer callback behind (no zombie timer; M25.5-A).
        self._analyzer_watch_timer.stop()
        # Cancel any active boring subprocess before tearing down the window.
        # The QProcess/runner is parented to the window, so a graceful SIGTERM
        # here (and Qt's own child cleanup on destroy) is the best-effort path;
        # the real runner is only ever active outside tests.
        if self._boring_active and self._boring_runner is not None:
            try:
                self._boring_runner.cancel()
            except Exception:
                pass
        # Stop + join the bounded histogram worker (invalidates in-flight work).
        histogram_stopped = self._histogram_coordinator.shutdown(wait_ms=wait_ms)
        # F3: stop + join the off-thread GPU probe worker (best-effort; the
        # probe is short and interruption only prevents it from starting).
        probe_stopped = self._stop_gpu_probe(wait_ms=wait_ms)
        shutdown_complete = self.controller.shutdown(wait_ms=wait_ms)
        if shutdown_complete and histogram_stopped and probe_stopped:
            self._shutdown_called = True
            self._running = False
            self._update_run_state()
            # Persist settings + geometry on a completed teardown (M8).  A no-op
            # when persistence is disabled (settings_path is None).
            self._save_persisted_settings()
            return True
        # Still stopping: keep the controller/thread references alive and do
        # NOT record completion, so Start stays disabled and a later
        # shutdown/close retries the teardown once the thread stops.
        self._shutdown_called = False
        self._running = True
        self._update_run_state()
        self.statusBar().showMessage("Stopping…")
        self.log("Shutdown incomplete — worker thread still running.")
        return False

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override name
        if self.shutdown():
            event.accept()
        else:
            # Shutdown is incomplete: keep the window (and thus the controller
            # and its live QThread) alive instead of accepting the close and
            # destroying a still-running thread.
            event.ignore()
