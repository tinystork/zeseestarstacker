"""Reliability closure-B — restore the historical ``stacked/`` default.

Mission ZSSS-QT-PREFLIGHT-CLOSURE-B (product decision, human witness Tristan):

    The historical filesystem checkpoint is: a successfully consumed image is
    moved into a ``stacked/`` sub-folder of its source directory, while images
    still to process stay in ``input/``.  After a stop/crash, ``input/``
    approximates the remaining work — a rudimentary but robust checkpoint.

Commit ``f1761d5`` (R1) introduced ``if not self.move_stacked: return`` at the
``_move_to_stacked`` choke point with ``move_stacked=False`` by default, which
neutralised that historical behaviour as an *audit safety net* (not the wanted
product behaviour).

Closure-B restores the historical default: ``move_stacked`` is now True by
default (constructor + ``start_processing`` signature).  ``move_stacked=False``
remains an explicit zero-mutation safety mode for tests/harnesses.

These tests lock the restored semantics at the choke-point level:
  * default (True) -> source moved into ``stacked/``;
  * explicit ``False`` -> zero mutation (R1 guard preserved);
  * witness-style acceptance: N of N accepted -> N in ``stacked/``, 0 in input;
  * partial stop: consumed batch moved, remaining batch stays in input.
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

# Stub GUI modules to avoid Tk dependence during import (same pattern as
# test_reliability_source_immutability_r1.py).  The stubs are restored right
# after the queue-manager import so a full ``pytest tests/`` collection never
# leaks a fake ``seestar.gui`` (empty ``__path__``) into sibling engine tests
# that import ``seestar.gui.run_config`` etc.
_saved_sys_modules = {
    name: sys.modules.get(name)
    for name in ("seestar", "seestar.gui", "seestar.gui.settings", "seestar.gui.histogram_widget")
}
if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = []
    settings_mod = types.ModuleType("seestar.gui.settings")

    class DummySettingsManager:
        pass

    settings_mod.SettingsManager = DummySettingsManager
    hist_mod = types.ModuleType("seestar.gui.histogram_widget")
    hist_mod.HistogramWidget = object
    gui_pkg.settings = settings_mod
    gui_pkg.histogram_widget = hist_mod
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod
    sys.modules["seestar.gui.histogram_widget"] = hist_mod

qm = importlib.import_module("seestar.queuep.queue_manager")

# Restore the real packages so this module's import never pollutes the rest
# of the test session (see comment above).
for _name, _mod in _saved_sys_modules.items():
    if _mod is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _mod


def _lightweight_obj(move_stacked=True):
    """Minimal __new__-based instance for ``_move_to_stacked`` unit tests.

    ``move_stacked=True`` mirrors the restored historical default (closure-B);
    pass ``False`` for the explicit zero-mutation safety mode.
    """
    obj = qm.SeestarQueuedStacker.__new__(qm.SeestarQueuedStacker)
    obj.move_stacked = bool(move_stacked)
    obj.stacked_subdir_name = "stacked"
    obj.update_progress = lambda *a, **k: None
    return obj


def _write_fits(tmp_path, name, shape=(8, 8)):
    p = Path(tmp_path) / name
    fits.writeto(p, np.zeros(shape, dtype=np.float32), overwrite=True)
    return p


# --------------------------------------------------------------------------
# (a) default -> moves to stacked/ (historical behaviour)
# --------------------------------------------------------------------------

def test_start_processing_signature_default_is_true():
    import inspect

    sig = inspect.signature(qm.SeestarQueuedStacker.start_processing)
    assert sig.parameters["move_stacked"].default is True, (
        "closure-B restore: start_processing must default move_stacked=True"
    )


def test_move_to_stacked_default_moves_to_stacked(tmp_path):
    p = _write_fits(tmp_path, "Light_001.fit")
    obj = _lightweight_obj()  # default True
    obj._move_to_stacked([str(p)])

    assert not p.exists(), "default must move the consumed source"
    assert (Path(tmp_path) / "stacked" / p.name).exists(), (
        "moved file must land in <src>/stacked/<base>"
    )


# --------------------------------------------------------------------------
# (b) explicit False -> zero mutation (R1 guard preserved)
# --------------------------------------------------------------------------

def test_move_stacked_false_explicit_zero_mutation(tmp_path):
    p = _write_fits(tmp_path, "Light_001.fit")
    obj = _lightweight_obj(move_stacked=False)
    obj._move_to_stacked([str(p)])

    assert p.exists(), "source must stay in place when move_stacked=False"
    assert not (Path(tmp_path) / "stacked").exists(), (
        "no 'stacked' subdir may be created when move_stacked=False"
    )


# --------------------------------------------------------------------------
# (c) witness acceptance: 13 of 13 accepted -> 13 stacked/, 0 input/
# --------------------------------------------------------------------------

def test_witness_acceptance_13_of_13(tmp_path):
    files = [_write_fits(tmp_path, f"Light_{i:03d}.fit") for i in range(1, 14)]
    obj = _lightweight_obj()  # default True

    # Simulate 13 accepted images (all consumed successfully).
    obj._move_to_stacked([str(f) for f in files])

    stacked_dir = Path(tmp_path) / "stacked"
    assert stacked_dir.is_dir(), "stacked/ subdir must be created"
    moved = sorted(p.name for p in stacked_dir.glob("*.fit"))
    assert len(moved) == 13, f"expected 13 moved files, got {len(moved)}"
    remaining = sorted(p.name for p in Path(tmp_path).glob("*.fit"))
    assert remaining == [], f"no source FITS may remain in input/, got {remaining}"


# --------------------------------------------------------------------------
# (d) partial stop: consumed batch moved, remaining batch stays in input/
# --------------------------------------------------------------------------

def test_partial_stop_consumed_moved_remaining_stay(tmp_path):
    files = [_write_fits(tmp_path, f"Light_{i:03d}.fit") for i in range(1, 6)]
    obj = _lightweight_obj()  # default True

    # One batch of 2 consumed successfully; 3 remain to process.
    consumed = files[:2]
    remaining = files[2:]

    obj._move_to_stacked([str(f) for f in consumed])

    stacked_dir = Path(tmp_path) / "stacked"
    assert stacked_dir.is_dir()
    for f in consumed:
        assert not f.exists(), f"consumed source {f.name} must be moved"
        assert (stacked_dir / f.name).exists(), (
            f"consumed {f.name} must land in stacked/"
        )
    for f in remaining:
        assert f.exists(), f"remaining source {f.name} must stay in input/"
    assert len(list(stacked_dir.glob("*.fit"))) == 2, (
        "only the consumed batch may be in stacked/"
    )
