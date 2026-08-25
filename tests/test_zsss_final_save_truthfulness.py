"""ZSSS-LIFECYCLE-01: truthful final-save success flags (engine-side seam).

Verifies, against the *real* :class:`SeestarQueuedStacker` (built bare via
``__new__``, no heavy ``__init__``), that:

* the final preview PNG save emits ``FINAL_PREVIEW_SAVE_RETURNED`` with
  ``success`` reflecting the *actual* ``save_preview_image`` result — ``False``
  on a raising save (with a bounded error), ``True`` only on success, ``False``
  for ``None`` data — never a hardcoded ``True``;
* the final FITS save uses a local ``fits_write_success`` flag (not the earlier
  ``self.final_stacked_path`` assignment) as its success source of truth.

The heavy engine module is imported lazily inside the tests so the rest of the
suite stays fast and import-hygiene-clean.
"""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import pytest


def _bare_preview_stack():
    """Import the real queue_manager and build a bare stacker for the preview seam."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    # Keep the engine import free of the Tk settings module (defensive stub,
    # mirrors test_zsss_startup_refusal_qm.py).
    if "seestar.gui.settings" not in sys.modules:
        seestar_pkg = types.ModuleType("seestar")
        seestar_pkg.__path__ = [str(root / "seestar")]
        gui_pkg = types.ModuleType("seestar.gui")
        gui_pkg.__path__ = []
        settings_mod = types.ModuleType("seestar.gui.settings")
        settings_mod.SettingsManager = type("SettingsManager", (), {})
        gui_pkg.settings = settings_mod
        sys.modules["seestar.gui.settings"] = settings_mod
        sys.modules["seestar"] = seestar_pkg
        sys.modules["seestar.gui"] = gui_pkg

    import seestar.queuep.queue_manager as qm

    o = qm.SeestarQueuedStacker.__new__(qm.SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    events = []
    o._lifecycle_callback = lambda ev, fields: events.append((ev, dict(fields or {})))
    return qm, o, events


def test_preview_save_failure_emits_success_false(tmp_path, monkeypatch):
    qm, o, events = _bare_preview_stack()

    def _boom(*args, **kwargs):
        raise OSError("preview write failed")

    monkeypatch.setattr(qm, "save_preview_image", _boom)

    import numpy as np

    data = np.zeros((3, 4, 4), dtype=np.float32)
    qm._save_final_preview_png(o, data, str(tmp_path / "final.png"))

    assert events[0][0] == "FINAL_PREVIEW_SAVE_ENTERED"
    assert events[1][0] == "FINAL_PREVIEW_SAVE_RETURNED"
    assert events[1][1]["success"] is False
    assert "preview write failed" in events[1][1]["error"]


def test_preview_save_success_emits_success_true(tmp_path, monkeypatch):
    qm, o, events = _bare_preview_stack()
    monkeypatch.setattr(qm, "save_preview_image", lambda *a, **k: None)

    import numpy as np

    data = np.zeros((3, 4, 4), dtype=np.float32)
    qm._save_final_preview_png(o, data, str(tmp_path / "final.png"))

    assert events[0][0] == "FINAL_PREVIEW_SAVE_ENTERED"
    assert events[1][0] == "FINAL_PREVIEW_SAVE_RETURNED"
    assert events[1][1]["success"] is True
    assert events[1][1].get("error") is None


def test_preview_save_none_data_emits_success_false(tmp_path):
    qm, o, events = _bare_preview_stack()
    qm._save_final_preview_png(o, None, str(tmp_path / "final.png"))

    assert events[0][0] == "FINAL_PREVIEW_SAVE_RETURNED"
    assert events[0][1]["success"] is False


def test_preview_save_source_is_not_hardcoded_success():
    qm, _, _ = _bare_preview_stack()
    src = inspect.getsource(qm._save_final_preview_png)
    assert "success = False" in src
    assert "success = True" in src
    assert "success=success" in src
    assert "success=True" not in src  # never a hardcoded True


def test_fits_save_success_uses_local_write_flag():
    qm, _, _ = _bare_preview_stack()
    src = inspect.getsource(qm.SeestarQueuedStacker._save_final_stack)
    assert "fits_write_success = False" in src
    assert "fits_write_success = True" in src
    assert "success=fits_write_success" in src
    # The success flag must not be derived from the earlier
    # ``self.final_stacked_path = fits_path`` assignment.
    assert "success=bool(self.final_stacked_path)" not in src
