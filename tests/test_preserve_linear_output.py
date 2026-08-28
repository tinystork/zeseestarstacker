"""Regression test for ``preserve_linear_output`` in ``_save_final_stack``.

Ensures that when ``preserve_linear_output=True`` the final stack keeps its
linear dynamics (no percentile normalization to ``[0,1]``), so the data handed
back for preview still spans the original linear range.

The previous version of this test built a hand-rolled fake ``seestar`` package
tree (``sys.modules`` stubs for ``seestar.alignment.astrometry_solver``,
``seestar.enhancement.reproject_utils``, etc.) and then executed the *real*
``queue_manager.py`` under the canonical name.  As production modules evolved
(``reproject_utils`` gained ``ensure_wcs_pixel_shape`` and friends), the fake
tree became incomplete, ``exec_module`` raised, and a partially-initialized
``seestar.queuep.queue_manager`` was left in ``sys.modules`` — poisoning every
later test that imports ``SeestarQueuedStacker``.  We now import the real
internal modules directly and only mock the two file-writing operations that
the test must not perform.
"""

import numpy as np
from astropy.io import fits

from seestar.queuep import queue_manager


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
    # The production refactor replaced the old flag heuristics with a single
    # explicit finalization mode.  Selecting MOSAIC is what makes the SCI/WHT
    # data below the finalization source (mirrors test_save_final_stack.py).
    d.finalization_mode = queue_manager.FINALIZATION_MODE_MOSAIC

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
