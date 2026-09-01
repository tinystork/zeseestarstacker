"""COV-02 focused tests: real footprint-aware support taper.

The footprint taper follows the actual transformed valid footprint (translation
and rotation invariant) instead of a radial distance from the image centre.
"""

import numpy as np
import pytest

from seestar.enhancement.weight_utils import (
    make_footprint_taper,
    make_radial_weight_map,
)


def test_interior_unity_boundary_ramp_outside_zero():
    m = np.zeros((20, 30), bool)
    m[5:15, 8:22] = True
    t = make_footprint_taper(m, feather_px=4.0, floor=0.0)
    assert t.shape == (20, 30)
    assert t.dtype == np.float32
    assert t[10, 15] == 1.0          # interior
    assert t[0, 0] == 0.0            # outside
    assert np.all((t >= 0.0) & (t <= 1.0))
    assert 0.0 < t[5, 8] <= 0.26     # boundary pixel is low


def test_translation_invariance_footprint_following():
    # two fully-interior identical footprints at different positions
    a = np.zeros((40, 40), bool)
    a[15:25, 15:25] = True
    b = np.zeros((40, 40), bool)
    b[5:15, 5:15] = True
    ta = make_footprint_taper(a, feather_px=4.0, floor=0.0)
    tb = make_footprint_taper(b, feather_px=4.0, floor=0.0)
    # the taper around the footprint is identical after translation
    assert np.array_equal(ta[15:25, 15:25], tb[5:15, 5:15])


def test_radial_mask_fails_translation_invariance():
    # witness E: a global radial mask depends on absolute position
    r = make_radial_weight_map(40, 40, feather_fraction=0.5, floor=0.0)
    # centre region vs near-corner region differ -> NOT footprint-following
    assert not np.array_equal(r[17:23, 17:23], r[2:8, 33:39])


def test_empty_mask_returns_zero():
    t = make_footprint_taper(np.zeros((8, 8), bool), feather_px=4.0, floor=0.0)
    assert np.all(t == 0.0)


def test_full_mask_ramps_only_at_border():
    m = np.ones((32, 32), bool)
    t = make_footprint_taper(m, feather_px=5.0, floor=0.0)
    assert t[16, 16] == 1.0
    assert t[0, 0] <= 1.0 / 5.0 + 1e-6


def test_positive_floor_keeps_soft_edge():
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    t = make_footprint_taper(m, feather_px=4.0, floor=0.02)
    assert t[5, 5] >= 0.02
    assert t[10, 10] == 1.0
    assert np.all(t[m] >= 0.02)


@pytest.mark.parametrize("bad_floor", [-0.1, 1.0, np.nan, np.inf])
def test_rejects_bad_floor(bad_floor):
    with pytest.raises(ValueError):
        make_footprint_taper(np.ones((8, 8), bool), feather_px=4.0, floor=bad_floor)


@pytest.mark.parametrize("bad_px", [0.0, -1.0, np.nan, np.inf])
def test_rejects_bad_feather_px(bad_px):
    with pytest.raises(ValueError):
        make_footprint_taper(np.ones((8, 8), bool), feather_px=bad_px, floor=0.0)


def test_rejects_nonboolean_mask():
    with pytest.raises(ValueError):
        make_footprint_taper(np.ones((8, 8), np.float32), feather_px=4.0, floor=0.0)


def test_rejects_1d_mask():
    with pytest.raises(ValueError):
        make_footprint_taper(np.ones(8, bool), feather_px=4.0, floor=0.0)


# ---------------------------------------------------------------------------
# COV-02 integration: taper folded into support + mean flat-field invariance
# ---------------------------------------------------------------------------
import types
from astropy.io import fits

from seestar.queuep.queue_manager import SeestarQueuedStacker


def _mean_stack(tmp_path, apply_taper):
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.output_folder = str(tmp_path)
    o.update_progress = lambda *a, **k: None
    o.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None, debug=lambda *a, **k: None,
        info=lambda *a, **k: None, error=lambda *a, **k: None)
    o.stacking_mode = "mean"
    o.normalize_method = "none"
    o.weighting_method = "none"
    o.use_quality_weighting = False
    o.weight_by_snr = False
    o.weight_by_stars = False
    o.snr_exponent = 1.0
    o.stars_exponent = 0.5
    o.min_weight = 0.0
    o.apply_batch_feathering = apply_taper
    o.support_taper_px = 4.0
    o.support_taper_floor = 0.0
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o.drizzle_active_session = False
    o.is_mosaic_run = False
    o.stack_kappa_low = 3.0
    o.stack_kappa_high = 3.0
    o.winsor_limits = (0.05, 0.05)
    o.stack_reject_algo = "none"
    o.max_hq_mem = 1_000_000_000
    o.batch_size = 10
    o.settings = None
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.interbatch_norm_active = False
    o.max_stack_workers = 1
    o._current_batch_paths = []
    o._norm_reference = None
    o._is_plain_classic = lambda: False
    o._support_state_available = False
    return o


def test_support_domain_folds_footprint_taper(tmp_path):
    qm = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    qm.output_folder = str(tmp_path)
    qm._support_state_available = True
    qm._support_unavailable_reason = None
    qm.apply_batch_feathering = True
    qm.support_taper_px = 4.0
    qm.support_taper_floor = 0.0
    qm.coverage_sup_w1_memmap = None
    qm.coverage_sup_w2_memmap = None
    qm._create_support_memmaps((20, 20))
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    qm._apply_support_payload([(mask, 2.0)])
    w1 = qm.coverage_sup_w1_memmap
    assert w1[10, 10] == 2.0        # interior: 2.0 * 1.0
    assert 0.0 < w1[5, 5] < 2.0     # boundary: 2.0 * taper < 2.0
    assert w1[0, 0] == 0.0          # outside: 0


def test_mean_flat_field_invariance_with_taper(tmp_path):
    o = _mean_stack(tmp_path, apply_taper=True)
    H = W = 32
    m0 = np.zeros((H, W), bool)
    m0[2:30, 2:30] = True
    m1 = np.zeros((H, W), bool)
    m1[4:32, 4:32] = True
    const = 1000.0

    def item(img, mask):
        return (img, fits.Header(), {"snr": 1.0, "stars": 0.0}, None, mask)

    stacked, hdr, cov = o._stack_batch(
        [item(np.full((H, W, 3), const, np.float32), m0),
         item(np.full((H, W, 3), const, np.float32), m1)],
        1, 1,
    )
    assert stacked is not None
    valid = m0 | m1
    # flat field: where support > 0 the mean stays ~const (no centre/edge gain)
    assert np.allclose(stacked[valid], const, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# COV-06 BLOCKER A: symmetric boundary feathering
# ---------------------------------------------------------------------------


def test_all_valid_rect_symmetric_boundaries():
    m = np.ones((32, 32), bool)
    t = make_footprint_taper(m, feather_px=5.0, floor=0.0)
    # four corners equivalent
    assert abs(t[0, 0] - t[0, 31]) < 1e-6
    assert abs(t[0, 0] - t[31, 0]) < 1e-6
    assert abs(t[0, 0] - t[31, 31]) < 1e-6
    # top == bottom, left == right edge midpoints
    assert abs(t[0, 16] - t[31, 16]) < 1e-6
    assert abs(t[16, 0] - t[16, 31]) < 1e-6
    # centre is interior (unity)
    assert t[16, 16] == 1.0
    # boundary is low, not unity
    assert 0.0 < t[0, 0] < 1.0


def test_all_valid_rect_each_edge_low():
    m = np.ones((40, 40), bool)
    t = make_footprint_taper(m, feather_px=6.0, floor=0.0)
    for idx in (0, 20, 39):
        assert t[0, idx] < 1.0     # top
        assert t[39, idx] < 1.0    # bottom
        assert t[idx, 0] < 1.0     # left
        assert t[idx, 39] < 1.0    # right


def test_mask_touching_one_boundary():
    # footprint touches only the left edge of the image
    m = np.zeros((40, 40), bool)
    m[10:30, 0:20] = True
    t = make_footprint_taper(m, feather_px=4.0, floor=0.0)
    # left boundary (image edge) feathers, interior right side reaches 1.0
    assert t[20, 0] < 1.0          # on left image edge -> low
    assert t[20, 10] == 1.0        # interior -> unity
    # top and bottom of the footprint also feather (they are true boundaries)
    assert t[10, 10] < 1.0
    assert t[29, 10] < 1.0


def test_mask_touching_several_boundaries():
    # footprint touches top and left edges
    m = np.zeros((40, 40), bool)
    m[0:20, 0:20] = True
    t = make_footprint_taper(m, feather_px=4.0, floor=0.0)
    assert t[0, 0] < 1.0      # corner
    assert t[0, 10] < 1.0     # top edge
    assert t[10, 0] < 1.0     # left edge
    assert t[10, 10] == 1.0   # interior -> unity


def test_internal_invalid_island_feathers_inward():
    m = np.ones((32, 32), bool)
    m[14:18, 14:18] = False   # internal invalid island
    t = make_footprint_taper(m, feather_px=4.0, floor=0.0)
    # pixels adjacent to the island are low (distance 1 -> 1/4)
    assert t[13, 15] <= 0.26
    assert t[18, 15] <= 0.26
    assert t[15, 13] <= 0.26
    assert t[15, 18] <= 0.26
    # far interior stays unity
    assert t[4, 4] == 1.0
    # the island itself is 0
    assert t[15, 15] == 0.0


def test_fallback_matches_primary_convention(monkeypatch):
    # ensure the fallback uses the same symmetric boundary convention
    import seestar.enhancement.weight_utils as wu
    m = np.ones((24, 24), bool)
    primary = make_footprint_taper(m, feather_px=4.0, floor=0.0)
    fallback = wu._footprint_distance_fallback(m)
    frac = np.clip(fallback / 4.0, 0.0, 1.0)
    fb_taper = np.where(m, frac, np.float32(0.0)).astype(np.float32)
    # fallback symmetric at boundaries
    assert abs(fb_taper[0, 0] - fb_taper[0, 23]) < 1e-4
    assert abs(fb_taper[0, 0] - fb_taper[23, 0]) < 1e-4
    assert abs(fb_taper[0, 0] - fb_taper[23, 23]) < 1e-4
