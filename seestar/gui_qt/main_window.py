"""PySide6 main-window shell for ZeSeestarStacker (Tk-like topology).

This is a *non-default* GUI shell.  The default entry point remains the Tk
``seestar.main:main`` application; this module is an architectural foothold for
the later parity migration and deliberately does NOT:

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
display-only preview WB / stretch / histogram controls (M10).  The Analyse
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
the lifecycle controller.  The default backend remains simulated (and is
labelled as such next to Start); real ``SeestarQueuedStacker`` activation is
explicit opt-in only.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from dataclasses import replace
from typing import Callable, List, Optional

from PySide6.QtCore import QByteArray, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
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
    QProgressBar,
    QPushButton,
    QScrollArea,
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
    DEFAULT_STRETCH,
    DEFAULT_WB,
    STRETCH_MODES,
    apply_wb_stretch,
    compute_histogram_stats,
    render_histogram_pixmap,
)
from .preview_view import ZOOM_LABELS, render_view
from .progress_time import UNKNOWN, estimate_remaining_seconds, format_duration
from .run_bridge import RunRequest, build_run_request as _build_run_request
from .run_controller import RunController
from .settings_validation import normalize_batch_size, validate_settings_for_backend
from .settings_state import QtSettingsState
from .solver_dialog import SolverSettingsDialog
from .solver_probe import probe_zesolver_operational

DEFAULT_TITLE = "ZeSeestarStacker — PySide6 shell"

# Backend selection modes understood by the shell's Start button.
BACKEND_MODES = ("simulated", "seestar")
DEFAULT_BACKEND_MODE = "simulated"

# Left-panel tab labels (Tk ``control_notebook`` parity: Stacking / Expert /
# Preview controls).
TAB_STACKING = "Stacking"
TAB_EXPERT = "Expert"
TAB_PREVIEW_CONTROLS = "Preview controls"

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
DRIZZLE_MODES = ["Final", "Incremental"]
SOLVER_PREFERENCES = ["none", "astap", "zesolver"]

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

# Tri-state for ``match_background_for_final``.  ``default`` (and the older
# ``none`` spelling) both mean "unset" -> ``None`` at the backend, which is the
# exact existing semantic in ``run_config.build_backend_kwargs``.
MATCH_BG_CHOICES = ("default", "true", "false")
MATCH_BG_FROM_TEXT = {"default": None, "true": True, "false": False}
MATCH_BG_TO_TEXT = {None: "default", True: "true", False: "false"}

# Honest backend-mode notice (M9 fix): makes the simulated default explicit so
# a human witness is never misled into thinking a Start click ran real science.
SIMULATED_BACKEND_NOTICE = (
    "Backend: simulated — Start runs no real stacking. "
    "Launch with '--backend seestar' to use the real engine."
)
SEESTAR_BACKEND_NOTICE = "Backend: seestar — real engine (explicit opt-in)."


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
            _field("min_weight", "Minimum weight", "float", 0.0, 1.0, 0.01, 3),
        ],
    ),
    (
        "Drizzle Advanced",
        [
            _field("drizzle_scale", "Drizzle scale", "int", 1, 10, 1),
            _field("drizzle_wht_threshold", "WHT threshold", "float", 0.0, 1.0, 0.01, 3),
            _field("drizzle_kernel", "Kernel", "combo", DRIZZLE_KERNELS),
            _field("drizzle_pixfrac", "Pixfrac", "float", 0.01, 2.0, 0.05, 2),
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
            _field("bn_grid_size_str", "BN grid size", "str"),
            _field("bn_perc_low", "BN percentile low", "int", 0, 100, 1),
            _field("bn_perc_high", "BN percentile high", "int", 0, 100, 1),
            _field("bn_std_factor", "BN std factor", "float", 0.0, 10.0, 0.1, 2),
            _field("bn_min_gain", "BN min gain", "float", 0.0, 10.0, 0.1, 2),
            _field("bn_max_gain", "BN max gain", "float", 0.0, 100.0, 0.1, 2),
            _field("cb_border_size", "CB border size", "int", 0, 1000, 1),
            _field("cb_blur_radius", "CB blur radius", "int", 0, 100, 1),
            _field("cb_min_b_factor", "CB min B factor", "float", 0.0, 10.0, 0.1, 2),
            _field("cb_max_b_factor", "CB max B factor", "float", 0.0, 100.0, 0.1, 2),
        ],
    ),
    (
        "Cropping",
        [
            _field("apply_master_tile_crop", "Master tile crop", "bool"),
            _field("master_tile_crop_percent", "Master tile crop %", "float", 0.0, 100.0, 0.5, 2),
            _field("final_edge_crop_percent", "Final edge crop %", "float", 0.0, 100.0, 0.5, 2),
        ],
    ),
    (
        "Photutils BN",
        [
            _field("apply_photutils_bn", "Photutils background normalization", "bool"),
            _field("photutils_bn_box_size", "Box size", "int", 1, 4096, 1),
            _field("photutils_bn_filter_size", "Filter size", "int", 1, 100, 1),
            _field("photutils_bn_sigma_clip", "Sigma clip", "float", 0.0, 10.0, 0.1, 2),
            _field("photutils_bn_exclude_percentile", "Exclude percentile", "float", 0.0, 100.0, 0.5, 2),
        ],
    ),
    (
        "Feathering / Low-weight Mask",
        [
            _field("apply_feathering", "Feathering", "bool"),
            _field("feather_blur_px", "Feather blur (px)", "int", 0, 4096, 1),
            _field("apply_batch_feathering", "Batch feathering", "bool"),
            _field("apply_low_wht_mask", "Low-weight mask", "bool"),
            _field("low_wht_percentile", "Low-weight percentile", "int", 0, 100, 1),
            _field("low_wht_soften_px", "Low-weight soften (px)", "int", 0, 4096, 1),
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
    "Drizzle Advanced": "section_drizzle_advanced",
    "Colour / Post-processing": "section_colour_post",
    "Cropping": "section_cropping",
    "Photutils BN": "section_photutils_bn",
    "Feathering / Low-weight Mask": "section_feathering",
    "Solver": "section_solver",
    "Output / Reprojection": "section_output_reprojection",
    "Final Background Matching": "section_final_bg_matching",
}

# attr -> translation key for the *representative* Settings field labels we
# localise (M9).  Remaining field labels stay English until a fuller mapping
# lands; the key-parity test only guards the keys actually registered here.
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
    "apply_master_tile_crop": "field_master_tile_crop",
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
        self._shutdown_wait_ms = shutdown_wait_ms
        # Settings/geometry persistence (M8).  ``None`` disables persistence so
        # bare ``MainWindow()`` constructions (tests) never touch a real file;
        # the Qt entry point passes the CWD ``seestar_settings.json`` default.
        self._settings_path = os.path.abspath(settings_path) if settings_path else None
        self.setWindowTitle(title if title is not None else DEFAULT_TITLE)
        # Qt-local localization state (M9).  English by default; a persisted
        # ``language`` field (when present) overrides this after controls are
        # built.  ``_last_*`` / ``_preview_detail`` keep the raw value behind
        # the dynamic labels so a language switch can re-render them.
        self._language: str = localization.normalize_language("en")
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
        # Input folder whose first FITS was last successfully auto-loaded
        # (M12).  Avoids redundant reloads on repeated settings restore; it is
        # cleared when the folder changes or the load fails.
        self._last_preview_folder: Optional[str] = None
        # Preview display-adjustment state (M10): white-balance R/G/B gains and
        # the display-stretch mode.  These act only on the derived display
        # image, never on ``_preview_source`` or any scientific output.  Raw
        # histogram stats string (or ``None``) drives the localized status label.
        self._wb: tuple = DEFAULT_WB
        self._stretch: str = DEFAULT_STRETCH
        self._histogram_stats: Optional[str] = None
        self.settings_state: QtSettingsState = QtSettingsState()
        self.controller = RunController(self)

        self._build_central()
        self._build_status_bar()
        self._wire_controls()
        self._wire_settings_controls()
        self._wire_controller()
        self._sync_state_from_controls()
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

        # Language control (M9): user-triggered FR/EN switch.
        lang_row = QHBoxLayout()
        self.language_label = QLabel(self._tr("language_label"))
        self._bind_text(self.language_label, "language_label")
        lang_row.addWidget(self.language_label)
        self.language_combo = QComboBox()
        self.language_combo.addItems(
            [localization.LANGUAGE_LABELS[code] for code in localization.SUPPORTED_LANGUAGES]
        )
        self.language_combo.setEnabled(True)
        lang_row.addWidget(self.language_combo)
        lang_row.addStretch(1)
        outer.addLayout(lang_row)

        self.tabs = QTabWidget()
        self._stacking_tab = self._build_stacking_tab()
        self._settings_tab = self._build_settings_tab()
        self._preview_controls_tab = self._build_preview_controls_tab()
        self.tabs.addTab(self._stacking_tab, self._tr("tab_stacking"))
        self.tabs.addTab(self._settings_tab, self._tr("tab_expert"))
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

        self.drizzle_check = QCheckBox(self._tr("drizzle_check"))
        self._bind_text(self.drizzle_check, "drizzle_check")
        self.drizzle_check.setChecked(False)

        self.drizzle_mode_combo = QComboBox()
        self.drizzle_mode_combo.addItems(DRIZZLE_MODES)
        self.drizzle_mode_combo.setCurrentText("Final")

        self.drizzle_group_spin = QSpinBox()
        self.drizzle_group_spin.setRange(1, 1_000_000)
        self.drizzle_group_spin.setValue(50)

        self.solver_combo = QComboBox()
        self.solver_combo.addItems(SOLVER_PREFERENCES)
        self.solver_combo.setCurrentText("none")

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
        self._add_form_row(form, "batch_size", self.batch_spin)
        form.addRow("", self.boring_check)
        self._add_form_row(form, "stacking_mode", self.stacking_mode_combo)
        self._add_form_row(form, "final_combine", self.final_combine_combo)
        form.addRow("", self.drizzle_check)
        self._add_form_row(form, "drizzle_mode", self.drizzle_mode_combo)
        self._add_form_row(form, "drizzle_group_size", self.drizzle_group_spin)
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

    def _build_preview_controls_tab(self) -> QWidget:
        """Left-panel "Preview controls" tab (M10): display-only WB / stretch /
        histogram.  These controls act only on the *displayed* preview pixmap
        (a derived image), never on the stored source image or any scientific
        output.  All controls are inert until a renderable preview arrives.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # White balance: per-channel R/G/B gains (neutral = 1.0).
        self.wb_group = QGroupBox(self._tr("wb_group"))
        self._bind_text(self.wb_group, "wb_group")
        wb_form = QFormLayout(self.wb_group)
        self.wb_r_spin = QDoubleSpinBox()
        self.wb_r_spin.setRange(0.0, 4.0)
        self.wb_r_spin.setSingleStep(0.05)
        self.wb_r_spin.setDecimals(2)
        self.wb_r_spin.setValue(1.0)
        self.wb_g_spin = QDoubleSpinBox()
        self.wb_g_spin.setRange(0.0, 4.0)
        self.wb_g_spin.setSingleStep(0.05)
        self.wb_g_spin.setDecimals(2)
        self.wb_g_spin.setValue(1.0)
        self.wb_b_spin = QDoubleSpinBox()
        self.wb_b_spin.setRange(0.0, 4.0)
        self.wb_b_spin.setSingleStep(0.05)
        self.wb_b_spin.setDecimals(2)
        self.wb_b_spin.setValue(1.0)
        self._add_form_row(wb_form, "wb_red", self.wb_r_spin)
        self._add_form_row(wb_form, "wb_green", self.wb_g_spin)
        self._add_form_row(wb_form, "wb_blue", self.wb_b_spin)
        self.wb_reset_button = QPushButton(self._tr("wb_reset"))
        self._bind_text(self.wb_reset_button, "wb_reset")
        wb_form.addRow("", self.wb_reset_button)
        layout.addWidget(self.wb_group)

        # Display stretch (linear default).
        self.stretch_group = QGroupBox(self._tr("stretch_group"))
        self._bind_text(self.stretch_group, "stretch_group")
        stretch_form = QFormLayout(self.stretch_group)
        self.stretch_combo = QComboBox()
        self.stretch_combo.addItems(list(STRETCH_MODES))
        self.stretch_combo.setCurrentText(DEFAULT_STRETCH)
        self._add_form_row(stretch_form, "stretch_label", self.stretch_combo)
        layout.addWidget(self.stretch_group)

        # Display histogram surface (real signal, not a placeholder).
        self.histogram_group = QGroupBox(self._tr("histogram_group"))
        self._bind_text(self.histogram_group, "histogram_group")
        histo_layout = QVBoxLayout(self.histogram_group)
        self.histogram_status = QLabel(self._tr("histogram_empty"))
        self.histogram_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        histo_layout.addWidget(self.histogram_status)
        self.histogram_view = QLabel()
        self.histogram_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.histogram_view.setMinimumSize(256, 64)
        histo_layout.addWidget(self.histogram_view)
        layout.addWidget(self.histogram_group)

        self._set_preview_controls_enabled(False)
        layout.addStretch(1)
        return panel

    def _build_right_panel(self) -> QWidget:
        """Build the persistent right preview/action panel (Tk ``preview_frame``
        + view + histogram + action buttons).

        The histogram group here is the persistent right-panel surface required
        by checklist item 13.4.  It mirrors the Preview controls tab histogram:
        both are fed by the same display-only pixmap/stats in
        :meth:`_refresh_histogram`, so they update and clear together.
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
        self.preview_image_label = QLabel()
        self.preview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image_label.setMinimumSize(256, 256)
        preview_layout.addWidget(self.preview_image_label, 1)
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
        layout.addWidget(view_group)

        # Persistent display histogram (Tk parity, checklist item 13.4).  This
        # is the *right-panel* surface; the Preview controls tab keeps its own
        # ``histogram_group`` / ``histogram_status`` / ``histogram_view``.  The
        # two surfaces share the same derived preview data via
        # ``_refresh_histogram`` / ``_render_histogram_status``.
        self.right_histogram_group = QGroupBox(self._tr("histogram_group"))
        self._bind_text(self.right_histogram_group, "histogram_group")
        right_histo_layout = QVBoxLayout(self.right_histogram_group)
        self.right_histogram_status = QLabel(self._tr("histogram_empty"))
        self.right_histogram_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_histo_layout.addWidget(self.right_histogram_status)
        self.right_histogram_view = QLabel()
        self.right_histogram_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_histogram_view.setMinimumSize(256, 64)
        right_histo_layout.addWidget(self.right_histogram_view)
        layout.addWidget(self.right_histogram_group)

        # Action buttons (Start/Stop/Analyse/Solver/path actions functional).
        actions_group = QGroupBox(self._tr("actions_group"))
        self._bind_text(actions_group, "actions_group")
        actions_layout = QGridLayout(actions_group)
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
        actions_layout.addWidget(self.start_button, 0, 0)
        actions_layout.addWidget(self.stop_button, 0, 1)
        actions_layout.addWidget(self.analyse_button, 1, 0)
        actions_layout.addWidget(self.solver_button, 1, 1)
        actions_layout.addWidget(self.view_inputs_button, 2, 0)
        actions_layout.addWidget(self.add_folder_button, 2, 1)
        actions_layout.addWidget(self.open_output_button, 3, 0, 1, 2)
        layout.addWidget(actions_group)

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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        outer = QVBoxLayout(container)

        for section_title, fields in SETTINGS_SECTIONS:
            if fields is None:
                group = self._build_mosaic_section()
            else:
                group = self._build_generic_section(SECTION_TITLE_KEYS[section_title], fields)
            outer.addWidget(group)

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

    def _backend_notice_text(self) -> str:
        """Return the honest backend-mode notice for the current mode."""
        if self.backend_mode == "seestar":
            return SEESTAR_BACKEND_NOTICE
        return SIMULATED_BACKEND_NOTICE

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        self.statusBar().showMessage("Idle")

    def _wire_controls(self) -> None:
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
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
        self.wb_r_spin.valueChanged.connect(self._on_wb_changed)
        self.wb_g_spin.valueChanged.connect(self._on_wb_changed)
        self.wb_b_spin.valueChanged.connect(self._on_wb_changed)
        self.wb_reset_button.clicked.connect(self._on_wb_reset)
        self.stretch_combo.currentIndexChanged.connect(self._on_stretch_changed)
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
        self.controller.finished.connect(self._on_run_finished)
        self.controller.failed.connect(self._on_run_failed)
        self.controller.cancelled.connect(self._on_run_cancelled)

    def _wire_settings_controls(self) -> None:
        """Mirror every settings widget into ``self.settings_state`` on change."""
        self.input_edit.textChanged.connect(self._sync_state_from_controls)
        self.output_edit.textChanged.connect(self._sync_state_from_controls)
        self.temp_edit.textChanged.connect(self._sync_state_from_controls)
        self.output_filename_edit.textChanged.connect(self._sync_state_from_controls)
        self.reference_edit.textChanged.connect(self._sync_state_from_controls)
        self.last_stack_edit.textChanged.connect(self._sync_state_from_controls)
        self.browse_input_button.clicked.connect(self._browse_input)
        self.browse_output_button.clicked.connect(self._browse_output)
        self.browse_temp_button.clicked.connect(self._browse_temp)
        self.browse_reference_button.clicked.connect(self._browse_reference)
        self.browse_last_stack_button.clicked.connect(self._browse_last_stack)
        self.batch_spin.valueChanged.connect(self._sync_state_from_controls)
        self.batch_spin.valueChanged.connect(self._on_batch_size_changed)
        self.boring_check.stateChanged.connect(self._on_boring_check_changed)
        self.stacking_mode_combo.currentIndexChanged.connect(
            self._sync_state_from_controls
        )
        self.final_combine_combo.currentIndexChanged.connect(
            self._sync_state_from_controls
        )
        self.drizzle_check.stateChanged.connect(self._sync_state_from_controls)
        self.drizzle_mode_combo.currentIndexChanged.connect(
            self._sync_state_from_controls
        )
        self.drizzle_group_spin.valueChanged.connect(self._sync_state_from_controls)
        self.solver_combo.currentIndexChanged.connect(self._sync_state_from_controls)

        for widget in self._settings_widgets.values():
            self._connect_settings_widget(widget)
        for _kind, widget in self._mosaic_widgets.values():
            self._connect_settings_widget(widget)

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
        """Surface a preflight failure (prefix + errors) and stay idle."""
        message = prefix + ": " + "; ".join(errors)
        self.log(message)
        self.statusBar().showMessage(message)
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
        self.statusBar().showMessage("Boring stack finished.")
        self.log("Boring stack finished.")
        self._mark_time_terminal("0:00")

    def _on_boring_failed(self, message: str) -> None:
        self._boring_active = False
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage(f"Boring stack failed: {message}")
        self.log(f"Boring stack failed: {message}")
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
        self.drizzle_mode_combo.setEnabled(not boring)
        self.drizzle_group_spin.setEnabled(not boring)
        if boring and self.drizzle_check.isChecked():
            self.drizzle_check.setChecked(False)

    def _on_solver(self) -> None:
        """Open the solver configuration dialog and apply accepted values.

        This never starts the backend and never imports the engine: the dialog
        is a pure view/controller over the existing solver boundaries (reached
        lazily), and accepted values are written back into the live Qt controls
        so a subsequent ``collect_settings_state()`` / ``build_run_request()``
        sees them.
        """
        dialog = SolverSettingsDialog(self, self.collect_settings_state())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_solver_dialog_values(dialog.values())

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

    def _check_analyzer_command_file(self) -> Optional[str]:
        """Consume the ZeAnalyser command file once, if present (Qt-safe).

        Reads ``REFERENCE=<path>``, updates the reference field only for a
        non-empty reference, deletes the file best-effort, and returns the
        reference (or ``None``).  The periodic re-arming watcher is a deferred
        delta; this single-shot consumption seam is safe to call from the GUI
        thread or from tests.
        """
        path = self._analyzer_command_file_path
        if not path or not os.path.exists(path):
            return None
        try:
            ref = analyzer_launch.consume_command_file(path)
        except OSError:
            return None
        if ref:
            self.reference_edit.setText(ref)
            self.log(f"Analyzer reference received: {ref}")
            self.statusBar().showMessage(f"Analyzer reference: {ref}")
        return ref

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

    def _browse_reference(self) -> None:
        """Select the reference image via a FITS file dialog (Tk parity)."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Image (Optional)",
            self._reference_start_dir(),
            FITS_FILE_FILTER,
        )
        if filepath:
            self.reference_edit.setText(os.path.abspath(filepath))

    def _browse_last_stack(self) -> None:
        """Select the previous stack via a FITS file dialog (Tk parity).

        When the output folder is empty, it is pre-filled from the selected
        file's parent directory — mirroring Tk ``_on_last_stack_changed``.
        """
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
            if not self.output_edit.text().strip():
                self.output_edit.setText(os.path.dirname(abs_path))

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

        The metadata label is updated unconditionally (stack name and counts).
        Additionally, when ``payload.data`` is image-like, it is converted
        (strictly display-only, via :func:`preview_render.render_preview_image`)
        and kept as the copied source image for the view transforms (zoom /
        rotation / resolution).  Invalid/missing data never raises and clears
        the stored source, the image area, the rotation state and the view
        controls, so no stale preview survives a failed render.
        """
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

        image = render_preview_image(payload.data)
        if image is not None and not image.isNull():
            # ``render_preview_image`` already returns a deep copy, so storing
            # it here is safe and independent of the payload's buffers.
            self._preview_source = image
            self._preview_rotation = 0
        else:
            self._preview_source = None
            self._preview_rotation = 0
        self._refresh_preview_view()

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
        """GUI-thread slot: render a loaded initial preview (or clear on error)."""
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
        self._refresh_preview_view()

    def _clear_preview(self, message: str) -> None:
        """Clear the preview surface and show a localized message."""
        self._preview_detail = message
        self._render_preview_label()
        self._preview_source = None
        self._preview_rotation = 0
        self._refresh_preview_view()

    # ------------------------------------------------- preview view controls
    def _on_zoom_changed(self, _index: int) -> None:
        """Recompute the displayed pixmap/resolution when the zoom label changes."""
        self._refresh_preview_view()

    def _on_rotate_left(self) -> None:
        """Rotate the preview 90° counter-clockwise (cumulative, modulo 360)."""
        if self._preview_source is None:
            return
        self._preview_rotation = (self._preview_rotation - 90) % 360
        self._refresh_preview_view()

    def _on_rotate_right(self) -> None:
        """Rotate the preview 90° clockwise (cumulative, modulo 360)."""
        if self._preview_source is None:
            return
        self._preview_rotation = (self._preview_rotation + 90) % 360
        self._refresh_preview_view()

    def _on_wb_changed(self, *_ignored) -> None:
        """Update the white-balance gains and re-render the preview (display-only)."""
        self._wb = (
            self.wb_r_spin.value(),
            self.wb_g_spin.value(),
            self.wb_b_spin.value(),
        )
        self._refresh_preview_view()

    def _on_wb_reset(self) -> None:
        """Reset the three white-balance gains to their neutral values."""
        self.wb_r_spin.setValue(1.0)
        self.wb_g_spin.setValue(1.0)
        self.wb_b_spin.setValue(1.0)

    def _on_stretch_changed(self, _index: int) -> None:
        """Update the display-stretch mode and re-render the preview (display-only)."""
        self._stretch = self.stretch_combo.currentText()
        self._refresh_preview_view()

    def _refresh_preview_view(self) -> None:
        """Repaint the preview image + resolution label + histogram from the
        stored source.

        The source :class:`QImage` is never mutated: a WB/stretch adjusted
        *derived* image is produced from it, then ``render_view`` applies the
        current rotation and zoom to that derived image, so zoom reapplies
        cleanly after rotation and the original display image stays pristine.
        The display histogram is computed from the same derived (displayed)
        image, so it tracks WB/stretch changes too.
        """
        source = self._preview_source
        if source is None or source.isNull():
            self.preview_image_label.clear()
            self._set_view_controls_enabled(False)
            self._set_preview_controls_enabled(False)
            self.resolution_label.setText("—")
            self._refresh_histogram(None)
            return
        adjusted = apply_wb_stretch(source, self._wb, self._stretch)
        if adjusted is None or adjusted.isNull():
            adjusted = source
        pixmap = render_view(
            adjusted,
            self._preview_rotation,
            self.zoom_combo.currentText(),
            self.preview_image_label.size(),
        )
        if pixmap is None or pixmap.isNull():
            self.preview_image_label.clear()
            self._set_view_controls_enabled(False)
            self._set_preview_controls_enabled(False)
            self.resolution_label.setText("—")
            self._refresh_histogram(None)
            return
        self.preview_image_label.setPixmap(pixmap)
        self._set_view_controls_enabled(True)
        self._set_preview_controls_enabled(True)
        self.resolution_label.setText(self._resolution_text(source, pixmap))
        self._refresh_histogram(adjusted)

    def _resolution_text(self, source: QImage, pixmap: QPixmap) -> str:
        """Build the resolution label: original → displayed size + zoom + rotation."""
        zoom = self.zoom_combo.currentText()
        text = (
            f"{source.width()}x{source.height()} → "
            f"{pixmap.width()}x{pixmap.height()} · {zoom}"
        )
        rotation = self._preview_rotation % 360
        if rotation:
            text += f" · {rotation}°"
        return text

    def _set_view_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable the zoom + rotation view controls together."""
        self.zoom_combo.setEnabled(enabled)
        self.rotate_left_button.setEnabled(enabled)
        self.rotate_right_button.setEnabled(enabled)

    def _set_preview_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable the WB + stretch preview controls together."""
        self.wb_r_spin.setEnabled(enabled)
        self.wb_g_spin.setEnabled(enabled)
        self.wb_b_spin.setEnabled(enabled)
        self.wb_reset_button.setEnabled(enabled)
        self.stretch_combo.setEnabled(enabled)

    def _refresh_histogram(self, image: Optional[QImage]) -> None:
        """Update both display histogram surfaces from the (adjusted) preview.

        The Preview controls tab surface (``histogram_view``) and the persistent
        right-panel surface (``right_histogram_view``) are fed the same pixmap;
        the status labels are re-rendered together by
        :meth:`_render_histogram_status`.
        """
        if image is None or image.isNull():
            self.histogram_view.clear()
            self.right_histogram_view.clear()
            self._histogram_stats = None
            self._render_histogram_status()
            return
        pixmap = render_histogram_pixmap(image)
        if pixmap is not None and not pixmap.isNull():
            self.histogram_view.setPixmap(pixmap)
            self.right_histogram_view.setPixmap(pixmap)
        else:
            self.histogram_view.clear()
            self.right_histogram_view.clear()
        self._histogram_stats = compute_histogram_stats(image)
        self._render_histogram_status()

    def _render_histogram_status(self) -> None:
        """Render both histogram status labels from stored raw stats + language."""
        if self._histogram_stats is None:
            text = self._tr("histogram_empty")
        else:
            text = f"{self._tr('histogram_stats')} {self._histogram_stats}"
        self.histogram_status.setText(text)
        self.right_histogram_status.setText(text)

    def _on_run_finished(self) -> None:
        self._running = False
        self._update_run_state()
        self.progress.setValue(100)
        self.statusBar().showMessage("Finished.")
        self.log("Run finished.")
        self._mark_time_terminal("0:00")

    def _on_run_failed(self, message: str) -> None:
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage(f"Failed: {message}")
        self.log(f"Run failed: {message}")
        self._mark_time_terminal("failed")

    def _on_run_cancelled(self) -> None:
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage("Cancelled.")
        self.log("Run cancelled.")
        self._mark_time_terminal("cancelled")

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
        state.input_folder = self.input_edit.text()
        state.output_folder = self.output_edit.text()
        state.temp_folder = self.temp_edit.text()
        state.output_filename = self.output_filename_edit.text()
        state.reference_image_path = self.reference_edit.text()
        state.last_stack_path = self.last_stack_edit.text()
        state.batch_size = self.batch_spin.value()
        state.stacking_mode = self.stacking_mode_combo.currentText()
        state.use_drizzle = self.drizzle_check.isChecked()
        state.drizzle_mode = self.drizzle_mode_combo.currentText()
        state.drizzle_group_size = self.drizzle_group_spin.value()
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
            self.drizzle_check,
            self.drizzle_mode_combo,
            self.drizzle_group_spin,
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
            self.drizzle_check.setChecked(bool(state.use_drizzle))
            if state.drizzle_mode in DRIZZLE_MODES:
                self.drizzle_mode_combo.setCurrentText(state.drizzle_mode)
            self.drizzle_group_spin.setValue(int(state.drizzle_group_size))
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
        self._sync_state_from_controls()
        self._update_path_action_state()

    def _load_persisted_settings(self) -> None:
        """Load persisted settings + geometry into the window (best-effort).

        A missing or corrupt file yields the code defaults; an unknown value for
        a known field is coerced by ``QtSettingsState.from_dict`` and an unknown
        combo choice degrades to the widget's current value.  Never raises.
        """
        data = settings_persistence.load_settings_json(self._settings_path)
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
        Qt/Tk-independent ``run_config.build_run_request``.
        """
        state = self._effective_settings_state()
        return _build_run_request(
            state,
            initial_additional_folders=initial_additional_folders,
            auto_chunk_size=auto_chunk_size,
            special_single=special_single,
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
            (self._preview_controls_tab, "tab_preview_controls"),
        ):
            idx = self.tabs.indexOf(tab_widget)
            if idx >= 0:
                self.tabs.setTabText(idx, self._tr(key))
        self._render_elapsed_label()
        self._render_remaining_label()
        self._render_preview_label()
        self._render_histogram_status()

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
        # Cancel any active boring subprocess before tearing down the window.
        # The QProcess/runner is parented to the window, so a graceful SIGTERM
        # here (and Qt's own child cleanup on destroy) is the best-effort path;
        # the real runner is only ever active outside tests.
        if self._boring_active and self._boring_runner is not None:
            try:
                self._boring_runner.cancel()
            except Exception:
                pass
        shutdown_complete = self.controller.shutdown(wait_ms=wait_ms)
        if shutdown_complete:
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
