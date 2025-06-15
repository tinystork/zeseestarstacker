import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]

# Create minimal package structure for seestar with required submodules
seestar_pkg = sys.modules.get("seestar", types.ModuleType("seestar"))
seestar_pkg.__path__ = []
sys.modules["seestar"] = seestar_pkg

core_spec = importlib.util.spec_from_file_location(
    "seestar.core", ROOT / "seestar" / "core" / "__init__.py"
)
core_mod = importlib.util.module_from_spec(core_spec)
sys.modules["seestar.core"] = core_mod
core_spec.loader.exec_module(core_mod)

gui_pkg = sys.modules.get("seestar.gui", types.ModuleType("seestar.gui"))
gui_pkg.__path__ = []
sys.modules["seestar.gui"] = gui_pkg
settings_spec = importlib.util.spec_from_file_location(
    "seestar.gui.settings", ROOT / "seestar" / "gui" / "settings.py"
)
settings_mod = importlib.util.module_from_spec(settings_spec)
sys.modules["seestar.gui.settings"] = settings_mod
settings_spec.loader.exec_module(settings_mod)

queuep_pkg = sys.modules.get("seestar.queuep", types.ModuleType("seestar.queuep"))
queuep_pkg.__path__ = []
sys.modules["seestar.queuep"] = queuep_pkg

align_pkg = types.ModuleType("seestar.alignment")
align_pkg.__path__ = []
solver_mod = types.ModuleType("seestar.alignment.astrometry_solver")
class DummySolver:
    def __init__(self, *a, **k):
        pass
    def solve(self, *a, **k):
        return None

def solve_image_wcs(*a, **k):
    return None

solver_mod.AstrometrySolver = DummySolver
solver_mod.solve_image_wcs = solve_image_wcs
align_pkg.astrometry_solver = solver_mod
sys.modules["seestar.alignment"] = align_pkg
sys.modules["seestar.alignment.astrometry_solver"] = solver_mod

enhancement_pkg = types.ModuleType("seestar.enhancement")
enhancement_pkg.__path__ = []
stack_enh_spec = importlib.util.spec_from_file_location(
    "seestar.enhancement.stack_enhancement",
    ROOT / "seestar" / "enhancement" / "stack_enhancement.py",
)
stack_enh_mod = importlib.util.module_from_spec(stack_enh_spec)
sys.modules["seestar.enhancement.stack_enhancement"] = stack_enh_mod
stack_enh_spec.loader.exec_module(stack_enh_mod)

cc_mod = types.ModuleType("seestar.enhancement.color_correction")
class DummyCB:
    def __init__(self, *a, **k):
        pass
def apply_scnr(image_rgb, target_channel='green', amount=1.0, preserve_luminosity=True):
    return image_rgb
cc_mod.ChromaticBalancer = DummyCB
cc_mod.apply_scnr = apply_scnr
sys.modules["seestar.enhancement.color_correction"] = cc_mod

reproj_mod = types.ModuleType("seestar.enhancement.reproject_utils")
def _missing(*_a, **_k):
    raise ImportError("reproject not available")
reproj_mod.reproject_and_coadd = _missing
reproj_mod.reproject_interp = _missing
sys.modules["seestar.enhancement.reproject_utils"] = reproj_mod

sys.modules["seestar.enhancement"] = enhancement_pkg

qm_spec = importlib.util.spec_from_file_location(
    "seestar.queuep.queue_manager", ROOT / "seestar" / "queuep" / "queue_manager.py"
)
queue_manager = importlib.util.module_from_spec(qm_spec)
sys.modules["seestar.queuep.queue_manager"] = queue_manager
qm_spec.loader.exec_module(queue_manager)


class Dummy:
    pass

def test_preserve_linear_output(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_manager, "save_preview_image", lambda *a, **k: None)
    monkeypatch.setattr(
        queue_manager.fits.HDUList,
        "writeto",
        lambda self, filename, **k: None,
    )
    d = Dummy()
    d.reproject_between_batches = False
    d.cumulative_sum_memmap = None
    d.cumulative_wht_memmap = None
    d.output_folder = str(tmp_path)
    d.output_filename = "result.fit"
    d.images_in_cumulative_stack = 1
    d.total_exposure_seconds = 1.0
    d.drizzle_wht_threshold = 0
    d.save_final_as_float32 = False
    d.current_stack_header = fits.Header()
    d.drizzle_active_session = False
    d.drizzle_mode = "Final"
    d.is_mosaic_run = False
    d.processing_error = None
    d.aligned_files_count = 1
    d.preserve_linear_output = True
    d.drizzle_output_wcs = None

    def update_progress(*args, **kwargs):
        pass

    d.update_progress = update_progress
    d._close_memmaps = lambda: None

    img = np.array([[1.2, 2.3], [0.7, 1.5]], dtype=np.float32)
    img3 = np.stack([img] * 3, axis=2)
    wht = np.ones_like(img, dtype=np.float32)

    queue_manager.SeestarQueuedStacker._save_final_stack(
        d,
        output_filename_suffix="_mosaic_reproject",
        drizzle_final_sci_data=img3,
        drizzle_final_wht_data=wht,
        preserve_linear_output=True,
    )

    assert np.max(d.last_saved_data_for_preview) > 2.0
