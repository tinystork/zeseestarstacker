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
(preview image + metadata, zoom/resolution/rotation placeholders, histogram
placeholder and the action buttons Start / Stop / Analyse / Solver /
View Inputs / Add Folder / Open Output).  Start/Stop are functional; the other
action buttons and the preview view/histogram controls are topology
placeholders for later milestones.

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

import os
from dataclasses import replace
from typing import Callable, List, Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
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

from . import boring_route
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

    def __init__(
        self,
        title: Optional[str] = None,
        parent: Optional[QWidget] = None,
        *,
        backend_factory: Optional[Callable[[], BaseRunBackend]] = None,
        backend_mode: str = DEFAULT_BACKEND_MODE,
        solver_probe: Optional[Callable[[], bool]] = None,
        boring_runner_factory: Optional[Callable[[], BoringRunnerBase]] = None,
        shutdown_wait_ms: int = 5000,
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
        if not isinstance(shutdown_wait_ms, int) or shutdown_wait_ms < 0:
            raise ValueError("shutdown_wait_ms must be a non-negative int")
        self.backend_factory = backend_factory
        self.backend_mode = backend_mode
        self.boring_runner_factory = boring_runner_factory
        self.solver_probe = (
            solver_probe if solver_probe is not None else probe_zesolver_operational
        )
        self._shutdown_wait_ms = shutdown_wait_ms
        self.setWindowTitle(title if title is not None else DEFAULT_TITLE)
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
        self.settings_state: QtSettingsState = QtSettingsState()
        self.controller = RunController(self)

        self._build_central()
        self._build_status_bar()
        self._wire_controls()
        self._wire_settings_controls()
        self._wire_controller()
        self._sync_state_from_controls()

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

        # Language control placeholder (real Localization switch is a later
        # milestone).
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Français"])
        self.language_combo.setEnabled(False)
        lang_row.addWidget(self.language_combo)
        lang_row.addStretch(1)
        outer.addLayout(lang_row)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_stacking_tab(), TAB_STACKING)
        self.tabs.addTab(self._build_settings_tab(), TAB_EXPERT)
        self.tabs.addTab(self._build_preview_controls_tab(), TAB_PREVIEW_CONTROLS)
        outer.addWidget(self.tabs)

        # Progress / status area (Tk ``progress_frame``).
        outer.addWidget(QLabel("Progression:"))
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        outer.addWidget(self.progress)

        # Log area (Tk ``status_text``).
        outer.addWidget(QLabel("Log:"))
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

        self.browse_input_button = QPushButton("Browse...")
        self.browse_output_button = QPushButton("Browse...")
        self.browse_temp_button = QPushButton("Browse...")
        self.browse_reference_button = QPushButton("Browse...")
        self.browse_last_stack_button = QPushButton("…")

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(-1, 1_000_000)
        self.batch_spin.setValue(0)
        self.batch_spin.setToolTip("Batch size (-1 = auto).")

        # Boring (single-batch CSV) mode toggle — Tk ``boring_thread_check``
        # parity.  Checked <=> batch_size == 1.
        self.boring_check = QCheckBox("Threaded Boring Stack")
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

        self.drizzle_check = QCheckBox("Enable drizzle")
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
        form.addRow(
            "Input folder",
            self._path_row(self.input_edit, self.browse_input_button),
        )
        form.addRow(
            "Output folder",
            self._path_row(self.output_edit, self.browse_output_button),
        )
        form.addRow(
            "Temp folder",
            self._path_row(self.temp_edit, self.browse_temp_button),
        )
        form.addRow("Output filename", self.output_filename_edit)
        form.addRow(
            "Reference image",
            self._path_row(self.reference_edit, self.browse_reference_button),
        )
        form.addRow(
            "Last stack",
            self._path_row(self.last_stack_edit, self.browse_last_stack_button),
        )
        form.addRow("Batch size", self.batch_spin)
        form.addRow("", self.boring_check)
        form.addRow("Stacking mode", self.stacking_mode_combo)
        form.addRow("Final combine", self.final_combine_combo)
        form.addRow("", self.drizzle_check)
        form.addRow("Drizzle mode", self.drizzle_mode_combo)
        form.addRow("Drizzle group size", self.drizzle_group_spin)
        form.addRow("Local solver", self.solver_combo)
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
        """Left-panel "Preview controls" tab (topology placeholder).

        Full WB / stretch / histogram interactivity is a later milestone; the
        tab exists only to hold its place in the Tk-like left panel.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        label = QLabel("Preview controls placeholder — no interactivity yet.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addStretch(1)
        return panel

    def _build_right_panel(self) -> QWidget:
        """Build the persistent right preview/action panel (Tk ``preview_frame``
        + histogram + action buttons)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Preview (persistent across left-tab switches).
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("Preview: —")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_label)
        self.preview_image_label = QLabel()
        self.preview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image_label.setMinimumSize(256, 256)
        preview_layout.addWidget(self.preview_image_label, 1)
        layout.addWidget(preview_group, 1)

        # Zoom / resolution / rotation controls (basic/disabled placeholders).
        view_group = QGroupBox("View")
        view_layout = QGridLayout(view_group)
        view_layout.addWidget(QLabel("Zoom:"), 0, 0)
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["Fit", "100%", "200%", "50%"])
        self.zoom_combo.setEnabled(False)
        view_layout.addWidget(self.zoom_combo, 0, 1)
        view_layout.addWidget(QLabel("Resolution:"), 0, 2)
        self.resolution_label = QLabel("—")
        view_layout.addWidget(self.resolution_label, 0, 3)
        self.rotate_left_button = QPushButton("⟲")
        self.rotate_left_button.setEnabled(False)
        self.rotate_right_button = QPushButton("⟳")
        self.rotate_right_button.setEnabled(False)
        view_layout.addWidget(self.rotate_left_button, 1, 0)
        view_layout.addWidget(self.rotate_right_button, 1, 1)
        layout.addWidget(view_group)

        # Histogram placeholder.
        histo_group = QGroupBox("Histogram")
        histo_layout = QVBoxLayout(histo_group)
        self.histogram_placeholder = QLabel("[ ] Histogram placeholder")
        self.histogram_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        histo_layout.addWidget(self.histogram_placeholder)
        layout.addWidget(histo_group)

        # Action buttons (Start/Stop functional; the rest are topology stubs).
        actions_group = QGroupBox("Actions")
        actions_layout = QGridLayout(actions_group)
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.analyse_button = QPushButton("Analyse")
        self.solver_button = QPushButton("Solver")
        self.view_inputs_button = QPushButton("View Inputs")
        self.add_folder_button = QPushButton("Add Folder")
        self.open_output_button = QPushButton("Open Output")
        # Analyse is the only remaining topology stub (ZeAnalyser launch is a
        # later milestone); View Inputs / Add Folder / Open Output are wired to
        # path actions and their enablement is driven by
        # ``_update_path_action_state``.
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
                group = self._build_generic_section(section_title, fields)
            outer.addWidget(group)

        outer.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _build_generic_section(self, title: str, fields) -> QGroupBox:
        """Build one QGroupBox section from a list of field specs."""
        group = QGroupBox(title)
        form = QFormLayout(group)
        for field in fields:
            attr, label, kind = field[0], field[1], field[2]
            params = field[3:]
            widget = self._make_settings_widget(attr, kind, *params)
            form.addRow(label, widget)
            self._settings_widgets[attr] = widget
            self._settings_kinds[attr] = kind
        return group

    def _build_mosaic_section(self) -> QGroupBox:
        """Build the Mosaic section (top-level flag + nested dict sub-fields)."""
        state = self.settings_state
        ms = state.mosaic_settings if isinstance(state.mosaic_settings, dict) else {}
        group = QGroupBox("Mosaic")
        form = QFormLayout(group)

        self.mosaic_active_check = QCheckBox("Mosaic mode active")
        self.mosaic_active_check.setChecked(bool(state.mosaic_mode_active))
        form.addRow("", self.mosaic_active_check)
        self._settings_widgets["mosaic_mode_active"] = self.mosaic_active_check
        self._settings_kinds["mosaic_mode_active"] = "bool"

        for key, label, kind, params in MOSAIC_FIELDS:
            widget = self._make_mosaic_widget(kind, ms.get(key), params)
            form.addRow(label, widget)
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
        self.start_button.clicked.connect(self._on_start)
        self.stop_button.clicked.connect(self._on_stop)
        self.solver_button.clicked.connect(self._on_solver)
        self.view_inputs_button.clicked.connect(self._show_input_folder_list)
        self.add_folder_button.clicked.connect(self._add_folder)
        self.open_output_button.clicked.connect(self._open_output_folder)
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
        self._update_run_state()
        self.statusBar().showMessage("Boring stack running…")

    def _on_boring_finished(self, exit_code: int) -> None:
        self._boring_active = False
        self._running = False
        self._update_run_state()
        self.progress.setValue(100)
        self.statusBar().showMessage("Boring stack finished.")
        self.log("Boring stack finished.")

    def _on_boring_failed(self, message: str) -> None:
        self._boring_active = False
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage(f"Boring stack failed: {message}")
        self.log(f"Boring stack failed: {message}")

    def _on_boring_cancelled(self) -> None:
        self._boring_active = False
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage("Boring stack cancelled.")
        self.log("Boring stack cancelled.")

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

    # ------------------------------------------------------ path / file actions
    def _browse_input(self) -> None:
        """Select the input folder via a directory dialog (Tk parity)."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", self.input_edit.text().strip()
        )
        if folder:
            self.input_edit.setText(os.path.abspath(folder))

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
        self._update_run_state()
        self.statusBar().showMessage("Running…")

    def _on_progress(self, percent: int) -> None:
        self.progress.setValue(int(percent))

    def _on_preview(self, payload: BackendPreviewPayload) -> None:
        """Update the Preview tab label from a preview payload (GUI thread only).

        The metadata label is updated unconditionally (stack name and counts).
        Additionally, when ``payload.data`` is image-like, it is converted
        (strictly display-only, via :func:`preview_render.render_preview_image`)
        and shown as a pixmap.  Invalid/missing data never raises and clears
        the image area, so no stale preview survives a failed render.
        """
        name = payload.stack_name or "(no stack)"
        text = f"Preview: {name}"
        if payload.image_count is not None:
            text += f" — {payload.image_count} img"
            if payload.total_images is not None:
                text += f" / {payload.total_images}"
        if payload.current_batch is not None:
            text += f" — batch {payload.current_batch}"
            if payload.total_batches is not None:
                text += f" / {payload.total_batches}"
        self.preview_label.setText(text)

        image = render_preview_image(payload.data)
        if image is not None and not image.isNull():
            self.preview_image_label.setPixmap(QPixmap.fromImage(image))
        else:
            self.preview_image_label.clear()

    def _on_run_finished(self) -> None:
        self._running = False
        self._update_run_state()
        self.progress.setValue(100)
        self.statusBar().showMessage("Finished.")
        self.log("Run finished.")

    def _on_run_failed(self, message: str) -> None:
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage(f"Failed: {message}")
        self.log(f"Run failed: {message}")

    def _on_run_cancelled(self) -> None:
        self._running = False
        self._update_run_state()
        self.statusBar().showMessage("Cancelled.")
        self.log("Run cancelled.")

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
        """
        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()
        input_valid = bool(input_text) and os.path.isdir(input_text)
        output_valid = bool(output_text) and os.path.isdir(output_text)
        self.view_inputs_button.setEnabled(input_valid)
        self.open_output_button.setEnabled(output_valid)
        self.add_folder_button.setEnabled(input_valid and not self._running)

    # ------------------------------------------------- settings collection
    def _sync_state_from_controls(self, *_ignored) -> None:
        """Copy the current widget values into the settings model.

        Accepts (and ignores) any signal payload so it can be connected to
        ``textChanged``/``valueChanged``/``currentIndexChanged``/``stateChanged``
        directly.
        """
        state = self.settings_state
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

    # ------------------------------------------------------------- helpers
    def log(self, message: str) -> None:
        """Append a line to the read-only log tab."""
        self.log_view.append(message)

    @property
    def is_running(self) -> bool:
        return self._running

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
