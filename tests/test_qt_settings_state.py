"""M2 seam tests: ``seestar.gui_qt.settings_state`` Qt settings model.

``QtSettingsState`` is the plain, toolkit-agnostic settings model that the
PySide6 shell mirrors its widgets into, and that
``seestar.gui.run_config.build_run_request`` consumes.  These tests verify, with
no QApplication and no engine:

* the full attribute surface required by ``build_backend_kwargs`` is present,
* defaults are aligned with the Tk ``SettingsManager`` defaults,
* a ``RunRequest`` builds cleanly from a default state,
* batch-size / drizzle / solver semantics propagate exactly.
"""

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]

# --- Load seestar.gui_qt.settings_state (pure stdlib) ---
import seestar.gui_qt.settings_state as settings_state  # noqa: E402

QtSettingsState = settings_state.QtSettingsState

# --- Load seestar.gui_qt.run_bridge (isolated run_config loader) ---
from seestar.gui_qt.run_bridge import (  # noqa: E402
    RunRequest,
    build_backend_kwargs,
    build_run_request,
    compute_align_on_disk,
)

# --- Load seestar.gui.settings standalone (for a real SettingsManager) ---
settings_spec = importlib.util.spec_from_file_location(
    "seestar_settings_manager_qt", ROOT / "seestar" / "gui" / "settings.py"
)
settings_mod = importlib.util.module_from_spec(settings_spec)
sys.modules["seestar_settings_manager_qt"] = settings_mod
settings_spec.loader.exec_module(settings_mod)

SettingsManager = settings_mod.SettingsManager


def _make_state() -> QtSettingsState:
    return QtSettingsState()


# --------------------------------------------------------------------------
# Attribute surface
# --------------------------------------------------------------------------
def test_state_is_a_plain_dataclass():
    assert QtSettingsState.__dataclass_fields__
    state = _make_state()
    # The full surface required by build_backend_kwargs is present.
    kwargs = build_backend_kwargs(state)
    assert isinstance(kwargs, dict)
    assert len(kwargs) >= 70


def test_defaults_cover_full_required_surface():
    defaults = QtSettingsState.defaults()
    # Every attribute read by build_backend_kwargs must exist on the model.
    state = _make_state()
    build_backend_kwargs(state)  # would raise AttributeError if anything missing
    assert "batch_size" in defaults
    assert "drizzle_group_size" in defaults
    assert "local_solver_preference" in defaults
    assert "mosaic_settings" in defaults
    assert "order_file_list" in defaults


def test_defaults_aligned_with_settings_manager():
    sm = SettingsManager(settings_file="unused.json")
    sm_defaults = sm.get_default_values()
    qt_defaults = QtSettingsState.defaults()
    assert qt_defaults, "QtSettingsState must expose defaults"
    for key, value in qt_defaults.items():
        # ``theme`` is a Qt-shell presentation-only field (M25.5-C) with no Tk
        # equivalent — the Tk ``SettingsManager`` has no theme, so there is no
        # default to align against.  Exclude it from the backend-default
        # alignment invariant (it is never read by the engine or Tk).
        if key == "theme":
            continue
        assert key in sm_defaults, f"QtSettingsState.{key} missing from SettingsManager"
        assert value == sm_defaults[key], (
            f"default mismatch for {key}: {value!r} != {sm_defaults[key]!r}"
        )


def test_mutable_defaults_are_independent_instances():
    a = _make_state()
    b = _make_state()
    a.order_file_list.append("x.fit")
    a.mosaic_settings["alignment_mode"] = "changed"
    assert b.order_file_list == []
    assert b.mosaic_settings["alignment_mode"] == "local_fast_fallback"


# --------------------------------------------------------------------------
# RunRequest integration
# --------------------------------------------------------------------------
def test_run_request_builds_from_default_state():
    state = _make_state()
    req = build_run_request(state)
    assert isinstance(req, RunRequest)
    assert req.align_on_disk is False  # batch_size 0
    assert "chunk_size" not in req.backend_kwargs
    assert req.backend_kwargs["batch_size"] == 0
    assert req.backend_kwargs["stacking_mode"] == "kappa-sigma"
    assert req.backend_kwargs["local_solver_preference"] == "none"
    assert req.backend_kwargs["drizzle_group_size"] == 50


def test_batch_size_chunk_size_semantics():
    state = _make_state()

    # batch_size == 1 non-special -> chunk_size added, align_on_disk True
    state.batch_size = 1
    req = build_run_request(state, auto_chunk_size=42, special_single=False)
    assert req.align_on_disk is True
    assert req.backend_kwargs["chunk_size"] == 42

    # special_single suppresses chunk_size
    req = build_run_request(state, auto_chunk_size=42, special_single=True)
    assert req.align_on_disk is True
    assert "chunk_size" not in req.backend_kwargs

    # batch_size >= 2 -> no chunk_size
    state.batch_size = 5
    req = build_run_request(state, auto_chunk_size=42)
    assert "chunk_size" not in req.backend_kwargs
    assert req.align_on_disk is True


def test_drizzle_and_solver_propagation():
    state = _make_state()
    state.use_drizzle = True
    state.drizzle_mode = "Incremental"
    state.drizzle_group_size = 77
    state.local_solver_preference = "zesolver"
    state.astap_search_radius = 12.5

    kwargs = build_backend_kwargs(state)
    assert kwargs["use_drizzle"] is True
    assert kwargs["drizzle_mode"] == "Incremental"
    assert kwargs["drizzle_group_size"] == 77
    assert kwargs["local_solver_preference"] == "zesolver"
    assert kwargs["astap_search_radius"] == 12.5


def test_run_request_is_immutable_and_copies_containers():
    state = _make_state()
    state.order_file_list = ["a.fit", "b.fit"]
    state.mosaic_settings = {"alignment_mode": "local"}
    folders = ["/extra-a", "/extra-b"]

    req = build_run_request(state, initial_additional_folders=folders)

    assert req.backend_kwargs["ordered_files"] == ["a.fit", "b.fit"]
    assert req.backend_kwargs["ordered_files"] is not state.order_file_list
    assert req.backend_kwargs["mosaic_settings"] == {"alignment_mode": "local"}
    assert req.backend_kwargs["mosaic_settings"] is not state.mosaic_settings
    assert req.backend_kwargs["initial_additional_folders"] == folders
    assert req.backend_kwargs["initial_additional_folders"] is not folders

    # MappingProxyType rejects writes
    try:
        req.backend_kwargs["batch_size"] = 999
        raised = False
    except TypeError:
        raised = True
    assert raised


def test_compute_align_on_disk_semantics():
    assert compute_align_on_disk(-1) is False
    assert compute_align_on_disk(0) is False
    assert compute_align_on_disk(1) is True
    assert compute_align_on_disk(2) is True
    assert compute_align_on_disk("garbage") is False


# --------------------------------------------------------------------------
# M8: to_dict / from_dict persistence round-trip (pure, no Qt)
# --------------------------------------------------------------------------
def test_to_dict_contains_persistence_paths():
    state = _make_state()
    state.last_stack_path = "/outputs/last.fit"
    state.input_folder = "/inputs"
    state.output_folder = "/outputs"
    state.temp_folder = "/tmp"
    state.reference_image_path = "/inputs/ref.fit"
    d = state.to_dict()
    assert d["last_stack_path"] == "/outputs/last.fit"
    assert d["input_folder"] == "/inputs"
    assert d["output_folder"] == "/outputs"
    assert d["temp_folder"] == "/tmp"
    assert d["reference_image_path"] == "/inputs/ref.fit"


def test_from_dict_round_trip_representative_fields():
    state = _make_state()
    state.last_stack_path = "/outputs/last.fit"
    state.input_folder = "/inputs"
    state.output_folder = "/outputs"
    state.temp_folder = "/tmp/stack"
    state.reference_image_path = "/inputs/ref.fit"
    state.output_filename = "stack.fits"
    state.batch_size = 4
    state.stacking_mode = "median"
    state.stack_final_combine = "reproject"
    state.reproject_between_batches = True
    state.reproject_coadd_final = False
    state.use_drizzle = True
    state.drizzle_mode = "Incremental"
    state.drizzle_group_size = 77
    state.local_solver_preference = "zesolver"
    state.astap_search_radius = 12.5
    state.order_file_list = ["a.fit", "b.fit"]
    mosaic = dict(QtSettingsState.defaults()["mosaic_settings"])
    mosaic["kernel"] = "gaussian"
    mosaic["mosaic_scale_factor"] = 3
    state.mosaic_settings = mosaic
    state.match_background_for_final = True

    restored = QtSettingsState.from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()


def test_from_dict_ignores_unknown_keys():
    state = QtSettingsState.from_dict(
        {
            "batch_size": 3,
            "totally_unknown_key": "ignored",
            "another_unknown": 42,
        }
    )
    assert state.batch_size == 3
    # Unknown keys never become attributes.
    assert not hasattr(state, "totally_unknown_key")
    assert not hasattr(state, "another_unknown")
    # Missing known keys fall back to defaults.
    assert state.kappa == 2.5
    assert state.input_folder == ""


def test_from_dict_fills_missing_keys_from_defaults():
    state = QtSettingsState.from_dict({"batch_size": 7})
    defaults = QtSettingsState.defaults()
    assert state.batch_size == 7
    for key, value in defaults.items():
        if key != "batch_size":
            assert getattr(state, key) == value, key


def test_from_dict_coerces_or_falls_back():
    # Wrong type for an int field -> falls back to default.
    s1 = QtSettingsState.from_dict({"batch_size": "not-a-number"})
    assert s1.batch_size == 0

    # Numeric string coerced for int / float.
    s2 = QtSettingsState.from_dict({"batch_size": "3", "kappa": "4.25"})
    assert s2.batch_size == 3
    assert s2.kappa == 4.25

    # bool stays bool; 0/1 ints coerce to bool.
    s3 = QtSettingsState.from_dict({"correct_hot_pixels": False})
    assert s3.correct_hot_pixels is False
    s4 = QtSettingsState.from_dict({"correct_hot_pixels": 0})
    assert s4.correct_hot_pixels is False

    # match_background_for_final None-default accepts canonical strings.
    s5 = QtSettingsState.from_dict({"match_background_for_final": "true"})
    assert s5.match_background_for_final is True
    s6 = QtSettingsState.from_dict({"match_background_for_final": None})
    assert s6.match_background_for_final is None

    # A non-list for a list field falls back to a fresh default list.
    s7 = QtSettingsState.from_dict({"order_file_list": "not-a-list"})
    assert s7.order_file_list == []

    # A dict field merges onto the default so missing sub-keys are filled, and
    # known nested values are coerced/fallbacked too.
    s8 = QtSettingsState.from_dict(
        {"mosaic_settings": {"kernel": "gaussian", "mosaic_scale_factor": "bad"}}
    )
    assert s8.mosaic_settings["kernel"] == "gaussian"
    assert s8.mosaic_settings["alignment_mode"] == "local_fast_fallback"
    assert s8.mosaic_settings["mosaic_scale_factor"] == 1


def test_from_dict_non_dict_returns_defaults():
    for bad in (None, [], "hello", 42):
        state = QtSettingsState.from_dict(bad)
        assert state.to_dict() == QtSettingsState.defaults()


def test_from_dict_mutable_defaults_are_independent():
    a = QtSettingsState.from_dict({"order_file_list": ["x"]})
    b = QtSettingsState.from_dict({})
    a.order_file_list.append("y")
    assert b.order_file_list == []
