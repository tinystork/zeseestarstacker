"""Minimal PySide6 main-window shell for ZeSeestarStacker.

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

The Stack tab exposes input/output/temp/filename, batch size, stacking mode,
drizzle and local-solver controls; the Settings tab (M10) exposes the rest of
the backend-relevant :class:`~seestar.gui_qt.settings_state.QtSettingsState`
surface in a scrollable, grouped form.  Both feed the same model, which
:meth:`MainWindow.build_run_request` turns into a validated, immutable
:class:`~seestar.gui_qt.run_bridge.RunRequest`, which the Start button hands to
the lifecycle controller.  The default backend remains simulated (and is
labelled as such next to Start); real ``SeestarQueuedStacker`` activation is
explicit opt-in only.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .backend_runner import (
    BackendPreviewPayload,
    BaseRunBackend,
    SeestarQueuedStackerBackend,
)
from .preview_render import render_preview_image
from .run_bridge import RunRequest, build_run_request as _build_run_request
from .run_controller import RunController
from .settings_validation import validate_settings_for_backend
from .settings_state import QtSettingsState

DEFAULT_TITLE = "ZeSeestarStacker — PySide6 shell"

# Backend selection modes understood by the shell's Start button.
BACKEND_MODES = ("simulated", "seestar")
DEFAULT_BACKEND_MODE = "simulated"

# Placeholder tab labels used by the smoke test and future parity migration.
TAB_STACK = "Stack"
TAB_SETTINGS = "Settings"
TAB_PREVIEW = "Preview"
TAB_LOG = "Log"

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
            _field("reference_image_path", "Reference image path", "str"),
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
            _field("reproject_between_batches", "Reproject between batches", "bool"),
            _field("reproject_coadd_final", "Reproject + coadd final", "bool"),
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
    ):
        super().__init__(parent)
        if backend_mode not in BACKEND_MODES:
            raise ValueError(
                f"backend_mode must be one of {BACKEND_MODES!r}, got {backend_mode!r}"
            )
        if backend_factory is not None and not callable(backend_factory):
            raise TypeError("backend_factory must be callable or None")
        self.backend_factory = backend_factory
        self.backend_mode = backend_mode
        self.setWindowTitle(title if title is not None else DEFAULT_TITLE)
        self._running: bool = False
        self._shutdown_called: bool = False
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
        self.tabs = QTabWidget()

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        self.tabs.addTab(self._build_stack_tab(), TAB_STACK)
        self.tabs.addTab(self._build_settings_tab(), TAB_SETTINGS)
        self.tabs.addTab(self._build_preview_tab(), TAB_PREVIEW)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.tabs.addTab(self.log_view, TAB_LOG)

        self.setCentralWidget(self.tabs)

    def _build_stack_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        label = QLabel("Stacking pipeline placeholder — no real work is performed.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        # --- Minimal settings form (visible subset only) ---
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.temp_edit = QLineEdit()
        self.output_filename_edit = QLineEdit()

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(-1, 1_000_000)
        self.batch_spin.setValue(0)
        self.batch_spin.setToolTip("Batch size (-1 = auto).")

        self.stacking_mode_combo = QComboBox()
        self.stacking_mode_combo.addItems(STACKING_MODES)
        self.stacking_mode_combo.setCurrentText("kappa-sigma")

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
        form.addRow("Input folder", self.input_edit)
        form.addRow("Output folder", self.output_edit)
        form.addRow("Temp folder", self.temp_edit)
        form.addRow("Output filename", self.output_filename_edit)
        form.addRow("Batch size", self.batch_spin)
        form.addRow("Stacking mode", self.stacking_mode_combo)
        form.addRow("", self.drizzle_check)
        form.addRow("Drizzle mode", self.drizzle_mode_combo)
        form.addRow("Drizzle group size", self.drizzle_group_spin)
        form.addRow("Local solver", self.solver_combo)
        layout.addLayout(form)

        # Honest backend-mode notice (M9 fix): always visible next to Start so
        # a witness knows the default Start click is simulated.
        self.backend_notice_label = QLabel(self._backend_notice_text())
        self.backend_notice_label.setWordWrap(True)
        layout.addWidget(self.backend_notice_label)

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        layout.addStretch(1)

        return panel

    def _build_preview_tab(self) -> QWidget:
        """Build the Preview tab (metadata label + display-only image area)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.preview_label = QLabel("Preview: —")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview_label)
        self.preview_image_label = QLabel()
        self.preview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image_label.setMinimumSize(64, 64)
        layout.addWidget(self.preview_image_label)
        layout.addStretch(1)
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
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        status.addPermanentWidget(self.progress)
        self.setStatusBar(status)
        self.statusBar().showMessage("Idle")

    def _wire_controls(self) -> None:
        self.start_button.clicked.connect(self._on_start)
        self.stop_button.clicked.connect(self._on_stop)
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
        self.batch_spin.valueChanged.connect(self._sync_state_from_controls)
        self.stacking_mode_combo.currentIndexChanged.connect(
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
        state = self.collect_settings_state()
        errors = validate_settings_for_backend(state, self.backend_mode)
        if errors:
            self._on_preflight_failed(errors)
            return
        request = self.build_run_request()
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
        message = "Cannot start real backend: " + "; ".join(errors)
        self.log(message)
        self.statusBar().showMessage(message)
        self._running = False
        self._update_run_state()

    def _on_stop(self) -> None:
        if not self._running:
            return
        self.controller.cancel()
        self.statusBar().showMessage("Cancelling…")
        self.log("Stop requested.")

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
        state.batch_size = self.batch_spin.value()
        state.stacking_mode = self.stacking_mode_combo.currentText()
        state.use_drizzle = self.drizzle_check.isChecked()
        state.drizzle_mode = self.drizzle_mode_combo.currentText()
        state.drizzle_group_size = self.drizzle_group_spin.value()
        state.local_solver_preference = self.solver_combo.currentText()

        # Settings tab controls (scalar fields).
        for attr in self._settings_widgets:
            setattr(state, attr, self._read_settings_widget_value(attr))

        # Mosaic nested dict.
        self._sync_mosaic_settings(state)

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
        state = self.collect_settings_state()
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

    def shutdown(self) -> None:
        """Idempotent teardown hook.

        Called automatically by :meth:`closeEvent` and safe to call directly
        (e.g. from an application-level ``aboutToQuit`` handler).  It requests
        stop of any live run, waits for the worker QThread to finish, and
        resets UI state.  Safe to call multiple times.
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self.controller.shutdown()
        self._running = False
        self._update_run_state()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override name
        self.shutdown()
        event.accept()
