"""ZSSS-LIFECYCLE-01 boundary B: startup refusal carrier at the engine seam.

Verifies, against the *real* :class:`SeestarQueuedStacker` class (built bare via
``__new__``, no heavy ``__init__``), that:

* the read-only early resume preflight refuses a non-plain (Drizzle) mode over
  existing resume artifacts *without* touching any sentinel/scientific artifact;
* the structured ``StartupRefusal`` carrier is produced with the stable
  ``OUTPUT_STATE_INCOMPATIBLE`` code, semantic key and mode label;
* the structured reason is available (never parsed from progress strings).

The heavy engine module is imported lazily inside the test so the rest of the
suite stays fast and import-hygiene-clean.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest


def _bare_stacker(out_dir, *, drizzle: bool):
    import sys as _sys

    root = Path(__file__).resolve().parents[1]
    _sys.path.insert(0, str(root))

    # The queue_manager does not import ``seestar.gui.settings``; this stub is
    # kept minimal and only defensive in case a transitive import changes.
    if "seestar.gui" not in _sys.modules:
        seestar_pkg = types.ModuleType("seestar")
        seestar_pkg.__path__ = [str(root / "seestar")]
        gui_pkg = types.ModuleType("seestar.gui")
        gui_pkg.__path__ = []
        settings_mod = types.ModuleType("seestar.gui.settings")
        settings_mod.SettingsManager = type("SettingsManager", (), {})
        gui_pkg.settings = settings_mod
        _sys.modules["seestar.gui.settings"] = settings_mod
        _sys.modules["seestar"] = seestar_pkg
        _sys.modules["seestar.gui"] = gui_pkg

    from seestar.queuep.queue_manager import SeestarQueuedStacker

    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    o.output_folder = str(out_dir)
    o.is_mosaic_run = False
    o.drizzle_active_session = drizzle
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o._resume_requested = True
    return SeestarQueuedStacker, o


def test_resume_artifacts_with_drizzle_refused_readonly(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Sentinel / scientific artifacts that must remain byte-identical.
    manifest = out_dir / "resume_manifest.json"
    manifest.write_bytes(b'{"state":"clean"}')
    final_fits = out_dir / "final.fits"
    final_fits.write_bytes(b"FITS-SENTINEL-BYTES")
    before = {
        "resume_manifest.json": manifest.read_bytes(),
        "final.fits": final_fits.read_bytes(),
    }

    SeestarQueuedStacker, o = _bare_stacker(out_dir, drizzle=True)

    # Structured refusal carrier is available (stable code + semantic key/data).
    from seestar.queuep.queue_manager import StartupRefusal

    assert o._is_plain_classic() is False
    assert o._session_mode_label() == "drizzle"
    refusal = o._build_startup_refusal("resume limited to plain classic SUM/W …")
    assert refusal is not None
    assert refusal.code == StartupRefusal.CODE_OUTPUT_STATE_INCOMPATIBLE
    assert refusal.semantic_key == "output_state_incompatible"
    assert refusal.semantic_data["mode"] == "drizzle"

    # The early preflight itself refuses (read-only) with the same reason.
    ok, reason = o._early_resume_preflight()
    assert ok is False
    assert "resume limited to plain classic SUM/W" in reason

    # Nothing was written or modified: sentinel/scientific artifacts are intact.
    for name, data in before.items():
        assert (out_dir / name).read_bytes() == data
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "final.fits",
        "resume_manifest.json",
    ]


def test_generic_early_refusal_is_not_output_state_incompatible(tmp_path):
    """A non-resume or plain-classic refusal never carries the known code."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    SeestarQueuedStacker, o = _bare_stacker(out_dir, drizzle=False)
    # Plain classic is resumable-eligible; without resume artifacts there is no
    # known incompatible-output refusal, so _build_startup_refusal returns None.
    o._resume_requested = False
    o.drizzle_active_session = True  # drizzle active but no resume artifacts
    assert o._build_startup_refusal("some other reason") is None

    o._resume_requested = True
    o.drizzle_active_session = False  # resume artifacts but plain classic
    assert o._build_startup_refusal("some other reason") is None
