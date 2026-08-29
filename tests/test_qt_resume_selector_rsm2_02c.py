"""RSM2-02C — explicit Qt Resume selection, Last Stack locator, CFG restore.

These tests pin the bounded GUI-side resume contract:

* a persisted/browsed/edited last-stack path alone is **never** Resume (the
  selector stays New/Fresh and the RunRequest carries fresh + empty source),
* only the explicit New/Resume selector activates Resume,
* explicit Resume + a schema-v2 ``run_config.cfg`` + a recognized checkpoint
  manifest resolves the owning run directory, restores allowlisted
  resume-critical settings into the state/controls, sets the output folder to
  the owning run dir, and builds a RunRequest with resume intent/source,
* a nested/located FIT resolves to its owning run directory (bounded),
* corrupt / unsafe / ambiguous config, a checkpoint without config, a CFG
  without checkpoint, and legacy-CFG-with/without-checkpoint each fail closed
  with the expected restoration distinction,
* switching Resume -> New clears only the transient intent (never artifacts),
* discovery/restore/refusal never mutates run artifacts.

The locator helper (:mod:`seestar.resume_locator`) is pure stdlib; the
window-level rows use the offscreen QApplication and the simulated backend.
No real stacking, no engine, no Tk.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from seestar import run_contract, resume_locator
from seestar.core.drizzle_checkpoint import build_drizzle_canonical_config
from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt.settings_state import QtSettingsState

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _manifest(schema_version=2, **extra) -> dict:
    base = {
        "schema_version": schema_version,
        "state": "clean",
        "mode": "classic_sumw",
        "fingerprint": "0" * 64,
    }
    base.update(extra)
    return base


def _write_manifest(run_dir: Path, schema_version=2, **extra) -> Path:
    memdir = run_dir / "memmap_accumulators"
    memdir.mkdir(parents=True, exist_ok=True)
    p = memdir / "resume_manifest.json"
    p.write_text(json.dumps(_manifest(schema_version, **extra)), encoding="utf-8")
    return p


def _write_v2_cfg(run_dir: Path, *, scientific=None, execution=None) -> Path:
    cfg = run_contract.RunConfig.from_sections(
        product_version="8.2.0",
        scientific=scientific or {},
        execution=execution or {},
    )
    p = run_dir / "run_config.cfg"
    run_contract.write_cfg(cfg, str(p))
    return p


def _write_raw_cfg(run_dir: Path, obj: dict) -> Path:
    p = run_dir / "run_config.cfg"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _write_legacy_cfg(run_dir: Path, data: dict, name="_stack_legacy.cfg") -> Path:
    p = run_dir / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _drizzle_config(*, scale=2.0, kernel="square", pixfrac=0.8, wht=0.2, mode="Final", group_size=50):
    class Qm:
        weighting_method = "none"
        use_quality_weighting = False
        weight_by_snr = True
        weight_by_stars = True
        snr_exponent = 1.0
        stars_exponent = 0.5
        min_weight = 0.01
        correct_hot_pixels = True
        hot_pixel_threshold = 3.0
        neighborhood_size = 5
        bayer_pattern = "GRBG"
        drizzle_scale = scale
        drizzle_kernel = kernel
        drizzle_pixfrac = pixfrac
        drizzle_wht_threshold_effective = wht
        drizzle_fillval = "0.0"
        drizzle_mode = mode
        drizzle_group_size = group_size

    return build_drizzle_canonical_config(Qm(), product_version="8.2.0")


def _write_drizzle_checkpoint(run_dir: Path, cfg) -> Path:
    ckpt = run_dir / ".m3d_checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    channels = []
    for channel in range(3):
        artifacts = {}
        for kind in ("img", "wht"):
            filename = f"gen-00000001-ch{channel}-out_{kind}.npy"
            (ckpt / filename).write_bytes(b"test-array")
            artifacts[f"out_{kind}"] = {"file": filename}
        channels.append({"channel": channel, **artifacts})
    manifest = {
        "schema_version": 1,
        "mode": "drizzle_native_v1",
        "state": "clean",
        "generation": 1,
        "product_version": cfg.product_version,
        "producer": "zeseestarstacker",
        "drizzle_lib_version": "test",
        "numpy_version": "test",
        "output_shape_hw": [8, 8],
        "wcs": {"CTYPE1": "RA---TAN"},
        "scientific_fingerprint": cfg.drizzle_fingerprint(),
        "scientific_config": cfg.scientific,
        "run_config_digest": cfg.full_digest(),
        "frame_count": 1,
        "stacked_batches_count": 1,
        "total_exposure_seconds": 10.0,
        "exposure_unknown_count": 0,
        "exposure_min": 10.0,
        "exposure_max": 10.0,
        "session": {},
        "completed_sources": [],
        "channels": channels,
    }
    path = ckpt / "checkpoint.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _snapshot(path: Path) -> dict:
    """Return a deterministic {relpath: bytes} snapshot of a directory tree."""
    out = {}
    for p in sorted(path.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(path))] = p.read_bytes()
    return out


def _distinctive_config():
    """A config whose values are visibly different from QtSettingsState defaults."""
    return run_contract.RunConfig.from_sections(
        product_version="8.2.0",
        scientific={
            "stacking_mode": "median",
            "kappa": 3.5,
            "winsor_limits": [0.10, 0.20],
            "weight_by_snr": False,
            "batch_size": 7,
        },
        execution={"input_folder": "/orig/in", "output_filename": "resumed.fit"},
    )


# --------------------------------------------------------------------------
# Pure locator / discovery (no Qt)
# --------------------------------------------------------------------------
def test_resolve_from_fit_directly_in_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_v2_cfg(run_dir, scientific={"stacking_mode": "kappa-sigma"})
    assert resume_locator.resolve_run_directory(str(run_dir / "final.fits")) == str(run_dir)


def test_resolve_from_nested_fit_walks_up(tmp_path):
    run_dir = tmp_path / "run"
    nested = run_dir / "sub" / "deep"
    nested.mkdir(parents=True)
    _write_manifest(run_dir)
    assert resume_locator.resolve_run_directory(str(nested / "stack.fits")) == str(run_dir)


def test_resolve_from_directory_locator(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_v2_cfg(run_dir, scientific={"stacking_mode": "kappa-sigma"})
    assert resume_locator.resolve_run_directory(str(run_dir)) == str(run_dir)


def test_resolve_no_recognized_state_returns_none(tmp_path):
    d = tmp_path / "no_state"
    d.mkdir()
    assert resume_locator.resolve_run_directory(str(d / "x.fit")) is None


def test_resolve_is_bounded(tmp_path):
    # A run dir deeper than MAX_SEARCH_DEPTH is not found from the leaf.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_v2_cfg(run_dir, scientific={"stacking_mode": "kappa-sigma"})
    leaf = run_dir
    for _ in range(resume_locator.MAX_SEARCH_DEPTH + 2):
        leaf = leaf / "sub"
        leaf.mkdir()
    assert resume_locator.resolve_run_directory(str(leaf)) is None


def test_discover_v2_cfg_with_manifest_ready(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_v2_cfg(run_dir, scientific={"stacking_mode": "median", "kappa": 3.5})
    _write_manifest(run_dir)
    result = resume_locator.discover_resume(str(run_dir / "final.fits"))
    assert result.status == resume_locator.STATUS_READY
    assert result.run_dir == str(run_dir)
    assert result.config_source == "v2"
    assert result.config.scientific["stacking_mode"] == "median"
    assert result.checkpoint_kind == resume_locator.CHECKPOINT_KIND_CLASSIC


def test_discover_drizzle_v2_cfg_ready_from_nested_locator(tmp_path):
    run_dir = tmp_path / "run"
    nested = run_dir / "preview" / "deep"
    nested.mkdir(parents=True)
    cfg = _drizzle_config(scale=3.0, kernel="gaussian", pixfrac=0.7, wht=0.25)
    run_contract.write_cfg(cfg, str(run_dir / "run_config.cfg"))
    _write_drizzle_checkpoint(run_dir, cfg)

    result = resume_locator.discover_resume(str(nested / "final.fits"))

    assert result.status == resume_locator.STATUS_READY
    assert result.run_dir == str(run_dir)
    assert result.config_source == "v2"
    assert result.checkpoint_kind == resume_locator.CHECKPOINT_KIND_DRIZZLE


@pytest.mark.parametrize("payload", ["{", "{}"])
def test_discover_drizzle_corrupt_or_incomplete_manifest_refuses(tmp_path, payload):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = _drizzle_config()
    run_contract.write_cfg(cfg, str(run_dir / "run_config.cfg"))
    path = run_dir / ".m3d_checkpoint" / "checkpoint.json"
    path.parent.mkdir()
    path.write_text(payload, encoding="utf-8")

    result = resume_locator.discover_resume(str(run_dir))

    assert result.status == resume_locator.STATUS_CORRUPT_CHECKPOINT


def test_discover_drizzle_requires_v2_and_never_migrates_legacy(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = _drizzle_config()
    _write_drizzle_checkpoint(run_dir, cfg)
    _write_legacy_cfg(run_dir, {"version": "5.6.0", "use_drizzle": True})

    result = resume_locator.discover_resume(str(run_dir))

    assert result.status == resume_locator.STATUS_CONFIG_UNAVAILABLE
    assert result.config is None
    assert result.checkpoint_kind == resume_locator.CHECKPOINT_KIND_DRIZZLE


def test_discover_both_checkpoint_kinds_refuses_ambiguity(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = _drizzle_config()
    run_contract.write_cfg(cfg, str(run_dir / "run_config.cfg"))
    _write_manifest(run_dir)
    _write_drizzle_checkpoint(run_dir, cfg)

    result = resume_locator.discover_resume(str(run_dir))

    assert result.status == resume_locator.STATUS_AMBIGUOUS_CHECKPOINT
    assert result.config is None


def test_discover_drizzle_missing_artifact_refuses_incomplete_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = _drizzle_config()
    run_contract.write_cfg(cfg, str(run_dir / "run_config.cfg"))
    _write_drizzle_checkpoint(run_dir, cfg)
    (run_dir / ".m3d_checkpoint" / "gen-00000001-ch2-out_wht.npy").unlink()

    result = resume_locator.discover_resume(str(run_dir))

    assert result.status == resume_locator.STATUS_CORRUPT_CHECKPOINT


def test_discover_drizzle_cfg_manifest_mismatch_refuses(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original = _drizzle_config(kernel="square", pixfrac=0.8)
    mismatched = _drizzle_config(kernel="lanczos2", pixfrac=1.0)
    run_contract.write_cfg(mismatched, str(run_dir / "run_config.cfg"))
    _write_drizzle_checkpoint(run_dir, original)

    result = resume_locator.discover_resume(str(run_dir))

    assert result.status == resume_locator.STATUS_CORRUPT_CHECKPOINT


def test_discover_cfg_only_without_checkpoint_refuses(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_v2_cfg(run_dir, scientific={"stacking_mode": "median"})
    result = resume_locator.discover_resume(str(run_dir / "final.fits"))
    assert result.status == resume_locator.STATUS_NO_CHECKPOINT
    assert result.config is not None  # config discovered, but not checkpoint evidence
    assert result.reason_key == "resume_refuse_no_checkpoint"


def test_discover_no_cfg_with_v1_checkpoint_ready_current_settings(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir, schema_version=1)  # hash-only v1: no run_config.cfg
    result = resume_locator.discover_resume(str(run_dir / "final.fits"))
    assert result.status == resume_locator.STATUS_READY
    assert result.config is None
    assert result.config_source is None
    assert result.checkpoint_kind == resume_locator.CHECKPOINT_KIND_CLASSIC


def test_discover_no_cfg_with_v2_checkpoint_still_config_unavailable(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir, schema_version=2)  # v2 without run_config.cfg
    result = resume_locator.discover_resume(str(run_dir / "final.fits"))
    assert result.status == resume_locator.STATUS_CONFIG_UNAVAILABLE
    assert result.config is None


def test_discover_corrupt_v2_cfg_fails_closed(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir)
    _write_raw_cfg(
        run_dir,
        {
            "schema_version": 2,
            "product_version": "8.2.0",
            "scientific_config": {"kappa": "not-a-number"},
            "execution_config": {},
            "provenance": {},
        },
    )
    result = resume_locator.discover_resume(str(run_dir / "final.fits"))
    assert result.status == resume_locator.STATUS_CORRUPT_CONFIG
    assert result.config is None


def test_discover_unsafe_v2_cfg_fails_closed(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir)
    _write_raw_cfg(
        run_dir,
        {
            "schema_version": 2,
            "product_version": "8.2.0",
            "scientific_config": {},
            "execution_config": {"astrometry_api_key": "REDACTED"},
            "provenance": {},
        },
    )
    result = resume_locator.discover_resume(str(run_dir / "final.fits"))
    assert result.status == resume_locator.STATUS_UNSAFE_CONFIG
    assert "REDACTED" not in (result.detail or "")


def test_discover_legacy_cfg_with_checkpoint_ready(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir, schema_version=1)
    _write_legacy_cfg(
        run_dir,
        {
            "version": "5.6.0",
            "stacking_mode": "median",
            "kappa": 3.5,
            "stack_winsor_limits": "0.10,0.20",
            "batch_size": 7,
            "input_folder": "/legacy/in",
        },
    )
    result = resume_locator.discover_resume(str(run_dir / "final.fits"))
    assert result.status == resume_locator.STATUS_READY
    assert result.config_source == "legacy"
    assert result.config.scientific["stacking_mode"] == "median"
    assert result.config.scientific["winsor_limits"] == [0.10, 0.20]


def test_discover_legacy_cfg_alone_refuses(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_legacy_cfg(run_dir, {"version": "5.6.0", "stacking_mode": "median"})
    result = resume_locator.discover_resume(str(run_dir / "final.fits"))
    # A legacy CFG is not recognized per-run state for *locating* the run
    # directory (only run_config.cfg and/or the manifest are), so it is never
    # checkpoint evidence and fails closed without a located run dir.
    assert result.status == resume_locator.STATUS_NO_RUN_DIR
    assert result.config is None


def test_discover_ambiguous_legacy_fails_closed(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir)
    _write_legacy_cfg(
        run_dir,
        {"version": "5.6.0", "stacking_mode": "kappa-sigma", "stack_method": "mean"},
    )
    result = resume_locator.discover_resume(str(run_dir / "final.fits"))
    assert result.status == resume_locator.STATUS_AMBIGUOUS_LEGACY


def test_restore_to_settings_normalizes_winsor_and_restores(tmp_path):
    state = QtSettingsState()
    resume_locator.restore_to_settings(_distinctive_config(), state)
    assert state.stacking_mode == "median"
    assert state.kappa == 3.5
    assert state.stack_winsor_limits == "0.1,0.2"  # list -> comma string
    assert state.weight_by_snr is False
    assert state.batch_size == 7
    assert state.input_folder == "/orig/in"
    assert state.output_filename == "resumed.fit"


def test_restore_to_settings_none_keeps_current_settings(tmp_path):
    state = QtSettingsState()
    state.stacking_mode = "median"
    state.kappa = 9.9
    report = resume_locator.restore_to_settings(None, state)
    assert report.applied == {}
    assert report.skipped == []
    # v1 fingerprint-only resume never invents a config: current settings stay.
    assert state.stacking_mode == "median"
    assert state.kappa == 9.9


# --------------------------------------------------------------------------
# Window-level integration (offscreen Qt)
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is not None
    return app


@pytest.fixture()
def window(qapp):
    win = MainWindow()
    yield win
    win.shutdown()


def _build_request(win):
    return win.build_run_request()


def test_persisted_last_stack_startup_selects_new_and_fresh(qapp, tmp_path):
    p = tmp_path / "seestar_settings.json"
    p.write_text(
        json.dumps({"last_stack_path": "/persist/last.fit", "output_folder": ""}),
        encoding="utf-8",
    )
    win = MainWindow(settings_path=str(p))
    try:
        assert win.resume_mode_combo.currentData() == "fresh"
        request = _build_request(win)
        assert request.resume_intent == "fresh"
        assert request.resume_source is None
        state = win.collect_settings_state()
        assert state.resume_intent == "fresh"
        assert state.resume_source == ""
    finally:
        win.shutdown()


def test_manual_last_stack_alone_stays_fresh(window):
    window.last_stack_edit.setText("/data/runs/last.fit")
    assert window.resume_mode_combo.currentData() == "fresh"
    state = window.collect_settings_state()
    assert state.resume_intent == "fresh"
    assert state.resume_source == ""
    request = _build_request(window)
    assert request.resume_intent == "fresh"
    assert request.resume_source is None


def test_browse_last_stack_alone_stays_fresh(window, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: ("/picked/stack.fits", "")),
    )
    window._browse_last_stack()
    assert window.last_stack_edit.text() == "/picked/stack.fits"
    assert window.resume_mode_combo.currentData() == "fresh"
    assert window.collect_settings_state().resume_intent == "fresh"


def test_explicit_resume_v2_restores_and_builds_request(tmp_path, window):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = _distinctive_config()
    run_contract.write_cfg(cfg, str(run_dir / "run_config.cfg"))
    _write_manifest(run_dir)

    window.last_stack_edit.setText(str(run_dir / "final.fits"))
    window.resume_mode_combo.setCurrentIndex(1)  # Resume

    assert window.resume_mode_combo.currentData() == "resume"
    state = window.collect_settings_state()
    assert state.resume_intent == "resume"
    assert state.resume_source == str(run_dir)
    assert state.output_folder == str(run_dir)
    assert window.output_edit.text() == str(run_dir)
    # resume-critical settings restored to model and visible controls
    assert state.stacking_mode == "median"
    assert state.kappa == 3.5
    assert window.stacking_mode_combo.currentText() == "median"

    request = _build_request(window)
    assert request.resume_intent == "resume"
    assert request.resume_source == str(run_dir)


def test_explicit_resume_v1_no_cfg_arms_current_settings(tmp_path, window):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir, schema_version=1)  # fingerprint-only v1, no CFG

    window.last_stack_edit.setText(str(run_dir / "final.fits"))
    window.resume_mode_combo.setCurrentIndex(1)  # Resume

    assert window.resume_mode_combo.currentData() == "resume"
    state = window.collect_settings_state()
    assert state.resume_intent == "resume"
    assert state.resume_source == str(run_dir)
    assert state.output_folder == str(run_dir)
    request = _build_request(window)
    assert request.resume_intent == "resume"
    assert request.resume_source == str(run_dir)


def test_explicit_resume_from_nested_fit(tmp_path, window):
    run_dir = tmp_path / "run"
    nested = run_dir / "sub"
    nested.mkdir(parents=True)
    cfg = _distinctive_config()
    run_contract.write_cfg(cfg, str(run_dir / "run_config.cfg"))
    _write_manifest(run_dir)

    window.last_stack_edit.setText(str(nested / "stack.fits"))
    window.resume_mode_combo.setCurrentIndex(1)

    assert window.resume_mode_combo.currentData() == "resume"
    state = window.collect_settings_state()
    assert state.resume_source == str(run_dir)
    assert state.output_folder == str(run_dir)


@pytest.mark.parametrize(
    "scale,kernel,pixfrac,wht",
    [
        (3.0, "gaussian", 0.65, 0.3),
        # Requested pixfrac is intentionally unavailable in a canonical D1
        # checkpoint; Lanczos restores the persisted effective value honestly.
        (1.0, "lanczos3", 1.0, 0.0),
    ],
)
def test_explicit_resume_drizzle_restores_effective_request(
    tmp_path, window, scale, kernel, pixfrac, wht
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = _drizzle_config(scale=scale, kernel=kernel, pixfrac=pixfrac, wht=wht)
    run_contract.write_cfg(cfg, str(run_dir / "run_config.cfg"))
    _write_drizzle_checkpoint(run_dir, cfg)
    # Start from deliberately incompatible visible state; restore must not
    # accidentally leave Classic, mosaic, or reproject armed.
    window.drizzle_check.setChecked(False)
    window.mosaic_active_check.setChecked(True)
    window.final_combine_combo.setCurrentText("Reproject + coadd")
    window.last_stack_edit.setText(str(run_dir / "final.fits"))

    window.resume_mode_combo.setCurrentIndex(1)

    state = window.collect_settings_state()
    request = window.build_run_request()
    kw = request.backend_kwargs
    assert state.resume_intent == "resume"
    assert state.resume_source == str(run_dir)
    assert state.use_drizzle is True
    assert state.drizzle_mode == "Final"
    assert state.drizzle_scale == int(scale)
    assert state.drizzle_kernel == kernel
    assert state.drizzle_pixfrac == pytest.approx(pixfrac)
    assert state.drizzle_wht_threshold == pytest.approx(wht)
    assert state.mosaic_mode_active is False
    assert state.reproject_between_batches is False
    assert state.reproject_coadd_final is False
    assert request.resume_intent == "resume"
    assert request.resume_source == str(run_dir)
    assert kw["output_dir"] == str(run_dir)
    assert kw["use_drizzle"] is True
    assert kw["drizzle_mode"] == "Final"
    assert kw["drizzle_scale"] == pytest.approx(scale)
    assert kw["drizzle_kernel"] == kernel
    assert kw["drizzle_pixfrac"] == pytest.approx(pixfrac)
    assert kw["drizzle_wht_threshold"] == pytest.approx(wht)
    assert kw["is_mosaic_run"] is False
    assert kw["reproject_between_batches"] is False
    assert kw["reproject_coadd_final"] is False


def test_explicit_resume_drizzle_refusal_never_arms(tmp_path, window):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = _drizzle_config()
    run_contract.write_cfg(cfg, str(run_dir / "run_config.cfg"))
    path = _write_drizzle_checkpoint(run_dir, cfg)
    path.write_text("{}", encoding="utf-8")
    window.last_stack_edit.setText(str(run_dir / "final.fits"))

    window.resume_mode_combo.setCurrentIndex(1)

    state = window.collect_settings_state()
    assert window.resume_mode_combo.currentData() == "fresh"
    assert state.resume_intent == "fresh"
    assert state.resume_source == ""
    assert window.build_run_request().resume_intent == "fresh"


def test_drizzle_resume_then_new_clears_transient_intent(tmp_path, window):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = _drizzle_config()
    run_contract.write_cfg(cfg, str(run_dir / "run_config.cfg"))
    _write_drizzle_checkpoint(run_dir, cfg)
    window.last_stack_edit.setText(str(run_dir / "final.fits"))
    window.resume_mode_combo.setCurrentIndex(1)
    assert window.collect_settings_state().resume_intent == "resume"

    window.resume_mode_combo.setCurrentIndex(0)

    state = window.collect_settings_state()
    assert state.resume_intent == "fresh"
    assert state.resume_source == ""
    assert window.build_run_request().resume_intent == "fresh"


def test_resume_then_new_clears_only_transient_intent(tmp_path, window):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_contract.write_cfg(_distinctive_config(), str(run_dir / "run_config.cfg"))
    _write_manifest(run_dir)

    window.last_stack_edit.setText(str(run_dir / "final.fits"))
    window.resume_mode_combo.setCurrentIndex(1)
    assert window.collect_settings_state().resume_intent == "resume"

    window.resume_mode_combo.setCurrentIndex(0)  # New
    state = window.collect_settings_state()
    assert state.resume_intent == "fresh"
    assert state.resume_source == ""
    # history/output text are kept
    assert window.last_stack_edit.text() == str(run_dir / "final.fits")
    assert state.last_stack_path == str(run_dir / "final.fits")


def test_resume_cfg_only_without_checkpoint_refuses_and_stays_fresh(tmp_path, window):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_contract.write_cfg(_distinctive_config(), str(run_dir / "run_config.cfg"))
    before = _snapshot(run_dir)

    window.last_stack_edit.setText(str(run_dir / "final.fits"))
    window.resume_mode_combo.setCurrentIndex(1)

    # selector reverted to New, no hidden intent, warning shown
    assert window.resume_mode_combo.currentData() == "fresh"
    state = window.collect_settings_state()
    assert state.resume_intent == "fresh"
    assert state.resume_source == ""
    assert window.error_box_count >= 1
    assert _snapshot(run_dir) == before  # no artifact mutation


def test_no_file_mutation_during_discovery_and_refusal(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_contract.write_cfg(_distinctive_config(), str(run_dir / "run_config.cfg"))
    _write_manifest(run_dir)
    before = _snapshot(run_dir)

    # A full ready discovery + restore path must not touch run artifacts.
    result = resume_locator.discover_resume(str(run_dir / "final.fits"))
    assert result.status == resume_locator.STATUS_READY
    state = QtSettingsState()
    resume_locator.restore_to_settings(result.config, state)
    assert _snapshot(run_dir) == before


# --------------------------------------------------------------------------
# RSM2-02C R1 — stale Resume invalidation on user-originated path changes
# --------------------------------------------------------------------------
def _arm_resume(window, tmp_path) -> Path:
    """Arm an explicit Resume against a ready v2 run dir and return the run dir."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_contract.write_cfg(_distinctive_config(), str(run_dir / "run_config.cfg"))
    _write_manifest(run_dir)
    window.last_stack_edit.setText(str(run_dir / "final.fits"))
    window.resume_mode_combo.setCurrentIndex(1)  # Resume
    assert window.resume_mode_combo.currentData() == "resume"
    assert window.collect_settings_state().resume_intent == "resume"
    return run_dir


def _assert_fresh_invalidated(window, *, last_stack=None, output=None):
    """Assert the selector/intent/source reverted to Fresh and paths kept."""
    assert window.resume_mode_combo.currentData() == "fresh"
    state = window.collect_settings_state()
    assert state.resume_intent == "fresh"
    assert state.resume_source == ""
    if last_stack is not None:
        assert window.last_stack_edit.text() == last_stack
        assert state.last_stack_path == last_stack
    if output is not None:
        assert window.output_edit.text() == output
        assert state.output_folder == output
    request = _build_request(window)
    assert request.resume_intent == "fresh"
    assert request.resume_source is None


def test_manual_edit_last_stack_while_armed_invalidates(tmp_path, window):
    from PySide6.QtTest import QTest

    run_dir = _arm_resume(window, tmp_path)
    # Simulate a real user edit (QTest key input emits ``textEdited``, which a
    # programmatic ``setText`` never does) so the user-only wiring is proven.
    window.last_stack_edit.clear()
    QTest.keyClicks(window.last_stack_edit, "/user/edited/stack.fit")
    assert window.last_stack_edit.text() == "/user/edited/stack.fit"
    _assert_fresh_invalidated(window, last_stack="/user/edited/stack.fit")
    # output (owning run dir) is untouched by the last-stack invalidation
    assert window.output_edit.text() == str(run_dir)


def test_manual_edit_output_while_armed_invalidates(tmp_path, window):
    from PySide6.QtTest import QTest

    _arm_resume(window, tmp_path)
    window.output_edit.clear()
    QTest.keyClicks(window.output_edit, "/user/edited/output")
    assert window.output_edit.text() == "/user/edited/output"
    _assert_fresh_invalidated(window, output="/user/edited/output")


def test_browse_last_stack_while_armed_invalidates(tmp_path, window, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    _arm_resume(window, tmp_path)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: ("/picked/new.fits", "")),
    )
    window._browse_last_stack()
    assert window.last_stack_edit.text() == "/picked/new.fits"
    _assert_fresh_invalidated(window, last_stack="/picked/new.fits")


def test_browse_output_while_armed_invalidates(tmp_path, window, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    _arm_resume(window, tmp_path)
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *a, **k: "/picked/output"),
    )
    window._browse_output()
    assert window.output_edit.text() == "/picked/output"
    _assert_fresh_invalidated(window, output="/picked/output")


def test_apply_resume_result_programmatic_output_update_does_not_invalidate(
    tmp_path, window
):
    """Regression: the programmatic output write inside _apply_resume_result
    must not invalidate the just-prepared Resume."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_contract.write_cfg(_distinctive_config(), str(run_dir / "run_config.cfg"))
    _write_manifest(run_dir)
    result = resume_locator.discover_resume(str(run_dir / "final.fits"))
    assert result.status == resume_locator.STATUS_READY

    # In the real flow the selector is already on Resume when
    # ``_apply_resume_result`` runs (combo change -> _activate_resume ->
    # _apply_resume_result); set it without re-firing the handler.
    window._set_resume_mode_combo("resume")
    window._apply_resume_result(result)

    assert window.resume_mode_combo.currentData() == "resume"
    state = window.collect_settings_state()
    assert state.resume_intent == "resume"
    assert state.resume_source == str(run_dir)
    assert state.output_folder == str(run_dir)
    assert window.output_edit.text() == str(run_dir)
    request = _build_request(window)
    assert request.resume_intent == "resume"
    assert request.resume_source == str(run_dir)


def test_programmatic_settext_while_armed_does_not_invalidate(tmp_path, window):
    """Pin the user-only wiring: ``setText`` emits ``textChanged`` (not
    ``textEdited``), so a programmatic last-stack write must never be mistaken
    for a user edit and must not invalidate an armed Resume."""
    run_dir = _arm_resume(window, tmp_path)
    window.last_stack_edit.setText("/programmatic/stack.fit")
    # selector + intent unchanged (only the guarded _apply_resume_result path
    # performs programmatic writes while armed, and it re-asserts Resume).
    assert window.resume_mode_combo.currentData() == "resume"
    state = window.collect_settings_state()
    assert state.resume_intent == "resume"
    assert state.resume_source == str(run_dir)
    assert window.last_stack_edit.text() == "/programmatic/stack.fit"


def test_production_drizzle_config_roundtrips_incremental_policy(tmp_path):
    # Production path: the real builder persists the policy; no manual injection.
    cfg = _drizzle_config(mode="Incremental", group_size=7)
    assert cfg.execution.get("drizzle_mode") == "Incremental"
    assert cfg.execution.get("drizzle_group_size") == 7
    state = QtSettingsState()
    resume_locator.restore_to_settings(
        cfg, state, checkpoint_kind=resume_locator.CHECKPOINT_KIND_DRIZZLE
    )
    assert state.drizzle_mode == "Incremental"
    assert state.drizzle_group_size == 7


def test_drizzle_policy_does_not_alter_scientific_fingerprint():
    standard = _drizzle_config(mode="Final", group_size=50)
    incremental = _drizzle_config(mode="Incremental", group_size=7)
    assert standard.drizzle_fingerprint() == incremental.drizzle_fingerprint()
