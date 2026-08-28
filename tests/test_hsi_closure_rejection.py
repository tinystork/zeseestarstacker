"""HSI Closure P3 — rejection / clipping family inventory and executable
non-associativity witnesses.

This is the *investigation-first* closure for HSI Closure section 3: it
inventories every currently supported stacking family, classifies its
post-current-code semantics, and adds small deterministic witnesses that
distinguish **global reduction** (feed every observation to the kernel at once)
from **hierarchical local reduction** (reduce subgroups, then compose the
``(V, W)`` pairs through the production ``SUM += V*W ; WHT += W`` accumulation
implemented by ``_combine_batch_result``).

The kernels exercised here are the *real* production numerical kernels in
``seestar/core/stack_methods.py`` (imported directly, never mocked); the
composition helper below re-implements nothing — it performs the exact
``SUM = Σ V·W`` / ``WHT = Σ W`` accumulation that ``_combine_batch_result``
performs on the ``(V, W)`` outputs of ``_stack_batch``.  The effective
denominator (``W``) returned by every kernel is the *same* per-pixel /
per-channel denominator that production feeds into that accumulation, so the
witnesses are faithful to the shipped representation ``SUM / WHT``.

Classification vocabulary (mirrors the HSI verdict table):

* ``EXACT``                 — composing local ``(V, W)`` pairs reproduces the
                              global result (up to float32 reduction order).
* ``EXACT UNDER CONDITIONS`` — composes only under stated conditions.
* ``APPROXIMATE BY DESIGN``  — global vs hierarchical differ by construction
                              (the documented bounded-memory algorithm); the
                              difference is the *expected* non-associativity,
                              not a lost denominator.
* ``NOT COMPOSABLE``         — the statistic has no linear ``SUM/WHT``
                              composition (median).
* ``DEFECT``                 — a demonstrated, production-reachable bug that
                              must be reported and reserved for a separate
                              bounded correction (never silently repaired).

Family inventory (user-visible -> kernel -> dispatch key), verified executable:

=================  ===============================  =========================  ===========
family             canonical kernel                 dispatch vocabulary         semantics
=================  ===============================  =========================  ===========
mean               ``_stack_mean``                  ``mode`` fallthrough        linear
median             ``_stack_median``                ``mode == "median"``        order stat.
kappa-sigma        ``_stack_kappa_sigma``           ``mode == "kappa-sigma"``   rejection
linear-fit clip    ``_stack_linear_fit_clip``       ``_is_linear_fit_clip_mode`` rejection
winsorized sigma   ``_stack_winsorized_sigma``      ``_is_winsorized_mode``     substitution
=================  ===============================  =========================  ===========

Source locations (queue_manager routing identical in ``_stack_worker``,
``_stack_batch`` and ``_combine_hq_by_tiles``): ``seestar/queuep/queue_manager.py``
lines ~867-911 (worker) / ~11260-11420 (``_stack_batch``) / ~10760-10800
(``_combine_hq_by_tiles``); ``_is_winsorized_mode`` + ``_WINSORIZED_MODE_ALIASES``
lines ~625-644 and ``_is_linear_fit_clip_mode`` +
``_LINEAR_FIT_CLIP_MODE_ALIASES`` lines ~647-666.  The canonical kernels live in
``seestar/core/stack_methods.py`` (``_stack_mean`` ~176, ``_stack_median`` ~207,
``_stack_kappa_sigma`` ~216, ``_stack_linear_fit_clip`` ~253,
``_stack_winsorized_sigma_iter`` ~288).

Accepted winsorized aliases (normalized at the boundary by
``_is_winsorized_mode``): ``"winsorized-sigma"``, ``"winsorized-sigma-clip"``,
``"winsorized_sigma"``, ``"winsorized_sigma_clip"``.

Accepted linear-fit clip aliases (normalized at the boundary by
``_is_linear_fit_clip_mode``): ``"linear_fit_clip"``, ``"linear-fit-clip"``
(the GUI/settings layer derives the hyphen spelling via
``stack_method.replace("_", "-")`` and the Qt shell advertises
``"linear-fit-clip"`` as its backend key).

``apply_rewinsor`` contract (winsorized sigma only): a *rejected* (valid but
clipped) sample is **substituted** with the nearest winsor bound of the
survivor distribution and keeps its full weight in the mean — its contribution
is retained, never dropped.  ``NaN`` marks a *missing* sample and never
contributes.
"""

import glob
import os
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
import importlib.util as _ilu  # noqa: E402

# ---------------------------------------------------------------------------
# Optional-dependency stubs (mirrors the sibling HSI closure suites) so the
# queue_manager imports without the GUI/ccdproc/drizzle stack.
# ---------------------------------------------------------------------------
_missing_optional = {
    _name for _name in ("cv2", "astroalign", "ccdproc", "drizzle")
    if _ilu.find_spec(_name) is None
}
for _name in ("cv2", "astroalign"):
    if _name in _missing_optional:
        import sys

        sys.modules.setdefault(_name, types.ModuleType(_name))
if "ccdproc" in _missing_optional:
    import sys

    _ccdproc = types.ModuleType("ccdproc")
    _ccdproc.combine = None
    sys.modules.setdefault("ccdproc", _ccdproc)
if "drizzle" in _missing_optional:
    import sys

    _drizzle = types.ModuleType("drizzle")
    _dr = types.ModuleType("drizzle.resample")

    class _DummyDrizzle:
        pass

    _dr.Drizzle = _DummyDrizzle
    _drizzle.resample = _dr
    sys.modules.setdefault("drizzle", _drizzle)
    sys.modules.setdefault("drizzle.resample", _dr)
from seestar.core.stack_methods import (  # noqa: E402
    _stack_kappa_sigma,
    _stack_linear_fit_clip,
    _stack_mean,
    _stack_median,
    _stack_winsorized_sigma,
)
from seestar.queuep.queue_manager import (  # noqa: E402
    _LINEAR_FIT_CLIP_MODE_ALIASES,
    _is_winsorized_mode,
    _is_linear_fit_clip_mode,
    _stack_worker,
    _WINSORIZED_MODE_ALIASES,
)

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
# ``EXACT`` composition (mean) differs only by float32 reduction order, observed
# <= 1e-6.  Non-associative families differ by O(1..1e2) on the chosen inputs,
# so a ``NONASSOC_MIN_DELTA`` of 1.0 cleanly separates "expected
# non-associativity" from "lost denominator".
EXACT_TOL = 1e-4
NONASSOC_MIN_DELTA = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def px(values):
    """Single-pixel grayscale observations ``(1, 1)`` from scalar values.

    Reducing over the image axis is then a pure reduction over the *observation*
    axis, which makes the global-vs-hierarchical mathematics exact and
    checkable by hand.
    """
    return [np.array([[float(v)]], dtype=np.float32) for v in values]


def scalar(arr):
    """Return the single float stored in a ``(1, 1[, C])`` kernel output."""
    a = np.asarray(arr, dtype=np.float64)
    return float(a.ravel()[0])


def compose(pairs):
    """Production ``SUM/WHT`` accumulation of local ``(V, W)`` outputs.

    Mirrors ``_combine_batch_result``: ``SUM += V * W ; WHT += W``, final
    ``V = SUM / WHT``.  Returns ``(final_V, final_WHT, SUM)``.
    """
    s = 0.0
    w = 0.0
    for V, W in pairs:
        s += scalar(V) * scalar(W)
        w += scalar(W)
    return s / w, w, s


def worker(mode, values, weights=None, kappa_low=3.0, kappa_high=3.0,
           winsor_limits=(0.05, 0.05), apply_rewinsor=True):
    """Run the *production* ``_stack_worker`` dispatcher and return ``(V, W, rej)``."""
    imgs = px(values)
    w = (np.asarray(weights, dtype=np.float32) if weights is not None
         else np.ones(len(values), dtype=np.float32))
    res = _stack_worker((mode, imgs, w, kappa_low, kappa_high, winsor_limits,
                         apply_rewinsor, True))
    V, W, rej = res
    return scalar(V), scalar(W), float(rej)


def _min_stack(mode, max_hq_mem=1_000_000_000, batch_size=10, tile_h=8):
    """Build a minimal ``SeestarQueuedStacker`` (no ``__init__``) in plain-classic
    mode (mosaic/drizzle/reproject all False) with quality weighting disabled, so
    the effective denominator is the raw valid count."""
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
    )
    o.stacking_mode = mode
    o.normalize_method = "none"
    o.weighting_method = "none"
    o.use_quality_weighting = False
    o.weight_by_snr = True
    o.weight_by_stars = False
    o.snr_exponent = 1.0
    o.stars_exponent = 0.5
    o.min_weight = 0.0
    o.apply_batch_feathering = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o.drizzle_active_session = False
    o.is_mosaic_run = False
    o.stack_kappa_low = 3.0
    o.stack_kappa_high = 3.0
    o.winsor_limits = (0.05, 0.05)
    o.stack_reject_algo = "none"
    o.max_hq_mem = max_hq_mem
    o.batch_size = batch_size
    o.settings = types.SimpleNamespace(TILE_HEIGHT=tile_h, batch_size=batch_size)
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.interbatch_norm_active = False
    o.max_stack_workers = 1
    o._current_batch_paths = []
    o._norm_reference = None
    return o


def _rejection_items(shape=(64, 64, 3)):
    """Nine identical inliers plus one global-offset outlier as batch items."""
    from astropy.io import fits as _fits

    H, W, C = shape
    ii = np.arange(H, dtype=np.float32)[:, None]
    jj = np.arange(W, dtype=np.float32)[None, :]
    ramp = 100.0 + 200.0 * (ii / max(H - 1, 1)) + 60.0 * (jj / max(W - 1, 1))
    A = np.stack([ramp] * C, axis=-1).astype(np.float32)
    obs = [A.copy() for _ in range(9)] + [(A + 1000.0).astype(np.float32)]
    mask = np.ones((H, W), dtype=bool)
    items = [
        (o, _fits.Header(), {"snr": 1.0, "stars": 0.0}, None, mask.copy())
        for o in obs
    ]
    return A, items


def _release_memmap(arr):
    """Release a possibly memmap-backed array and delete its backing file."""
    mm = getattr(arr, "_mmap", None)
    if mm is not None:
        try:
            arr.flush()
            mm.close()
        except Exception:
            pass
    fn = getattr(arr, "filename", None)
    if fn and os.path.exists(fn):
        try:
            os.remove(fn)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 1. Inventory / dispatch vocabulary (executable, not source-string)
# ---------------------------------------------------------------------------


def test_winsorized_alias_vocabulary():
    """``_is_winsorized_mode`` accepts exactly the four documented spellings and
    rejects the other family keys (so a winsorized alias can never fall through
    to the arithmetic mean)."""
    accepted = {
        "winsorized-sigma",
        "winsorized-sigma-clip",
        "winsorized_sigma",
        "winsorized_sigma_clip",
    }
    assert _WINSORIZED_MODE_ALIASES == accepted
    for alias in accepted:
        assert _is_winsorized_mode(alias), alias
        assert _is_winsorized_mode(alias.upper()), alias  # case-insensitive
    for not_winsor in ("kappa-sigma", "linear_fit_clip", "linear-fit-clip",
                       "median", "mean", "none", ""):
        assert not _is_winsorized_mode(not_winsor), not_winsor


def test_linear_fit_clip_alias_vocabulary():
    """``_is_linear_fit_clip_mode`` accepts exactly the canonical underscore key
    and the GUI hyphen spelling, and rejects every other family key (so no other
    mode can be mis-routed into linear-fit clipping)."""
    accepted = {"linear_fit_clip", "linear-fit-clip"}
    assert _LINEAR_FIT_CLIP_MODE_ALIASES == accepted
    for alias in accepted:
        assert _is_linear_fit_clip_mode(alias), alias
        assert _is_linear_fit_clip_mode(alias.upper()), alias  # case-insensitive
    for not_lfc in ("kappa-sigma", "median", "mean", "none", "",
                    "winsorized-sigma", "winsorized-sigma-clip",
                    "winsorized_sigma", "winsorized_sigma_clip"):
        assert not _is_linear_fit_clip_mode(not_lfc), not_lfc


def test_dispatch_routes_each_canonical_mode_to_its_kernel():
    """The production ``_stack_worker`` dispatcher routes every canonical key to
    the matching numerical kernel.  Each family produces a distinguishable
    scalar on the 9-inlier + 1-outlier dataset, so a mis-route is observable."""
    vals = [100.0] * 9 + [1100.0]

    # kappa-sigma rejects the outlier -> V == 100, W == 9.
    v, w, rej = worker("kappa-sigma", vals)
    assert v == pytest.approx(100.0, abs=EXACT_TOL)
    assert w == pytest.approx(9.0, abs=EXACT_TOL)
    assert rej == pytest.approx(10.0)

    # linear_fit_clip (underscore, canonical) rejects the outlier too.
    v, w, rej = worker("linear_fit_clip", vals)
    assert v == pytest.approx(100.0, abs=EXACT_TOL)
    assert w == pytest.approx(9.0, abs=EXACT_TOL)

    # median is an order statistic -> 100 (unaffected by the outlier).
    v, w, rej = worker("median", vals)
    assert v == pytest.approx(100.0, abs=EXACT_TOL)
    assert w == pytest.approx(10.0, abs=EXACT_TOL)

    # mean averages the outlier in -> 200.
    v, w, rej = worker("mean", vals)
    assert v == pytest.approx(200.0, abs=EXACT_TOL)
    assert w == pytest.approx(10.0, abs=EXACT_TOL)

    # winsorized sigma (substitution) keeps W == 10 but clamps the outlier.
    for alias in ("winsorized-sigma", "winsorized-sigma-clip",
                  "winsorized_sigma", "winsorized_sigma_clip"):
        v, w, rej = worker(alias, vals, winsor_limits=(0.2, 0.2))
        assert v == pytest.approx(100.0, abs=EXACT_TOL), alias
        assert w == pytest.approx(10.0, abs=EXACT_TOL), alias

    # The canonical dispatch equals the direct kernel call bit-for-bit.
    V, W, rej = _stack_kappa_sigma(px(vals), None, sigma_low=3.0, sigma_high=3.0,
                                   return_weights=True)
    assert worker("kappa-sigma", vals)[0] == scalar(V)
    assert worker("kappa-sigma", vals)[1] == scalar(W)


# ---------------------------------------------------------------------------
# 2. mean — EXACT composable control (nonuniform weights + partial coverage)
# ---------------------------------------------------------------------------


def test_mean_composes_exactly_with_nonuniform_weights_and_partial_coverage():
    """Classification: **EXACT**.

    The plain weighted mean is linear, so ``SUM = Σ V·W`` over subgroup means
    telescopes to the global weighted numerator and ``WHT = Σ W`` telescopes to
    the global denominator.  Here the witness carries nonuniform deterministic
    quality weights ``[1, 2, 3, 4]`` over ``[2, 4, 6, 8]``:

        global  V = (2·1 + 4·2 + 6·3 + 8·4) / 10 = 6.0,  W = 10
        local   [2,4] -> V=10/3, W=3 ; [6,8] -> V=50/7, W=7
        compose (10/3·3 + 50/7·7) / 10 = (10 + 50) / 10 = 6.0

    A partial-coverage variant (a missing sample) composes exactly as well:
    missing samples contribute 0 to both numerator and denominator.
    """
    vals = [2.0, 4.0, 6.0, 8.0]
    weights = [1.0, 2.0, 3.0, 4.0]

    Vg, Wg, _ = _stack_mean(px(vals), np.asarray(weights, dtype=np.float32),
                            return_weights=True)
    A = _stack_mean(px([2.0, 4.0]), np.asarray([1.0, 2.0], dtype=np.float32),
                    return_weights=True)
    B = _stack_mean(px([6.0, 8.0]), np.asarray([3.0, 4.0], dtype=np.float32),
                    return_weights=True)
    Vh, Wh, _ = compose([(A[0], A[1]), (B[0], B[1])])

    assert scalar(Vg) == pytest.approx(6.0, abs=EXACT_TOL)
    assert scalar(Wg) == pytest.approx(10.0, abs=EXACT_TOL)
    assert Vh == pytest.approx(6.0, abs=EXACT_TOL)
    assert Wh == pytest.approx(10.0, abs=EXACT_TOL)

    # Partial coverage: image 2 is missing, so its weight 2.0 is excluded.
    m = np.array([[np.nan]], dtype=np.float32)
    Vg2, Wg2, _ = _stack_mean([np.array([[10.0]], dtype=np.float32), m,
                               np.array([[30.0]], dtype=np.float32)],
                              np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                              return_weights=True)
    assert scalar(Vg2) == pytest.approx(25.0, abs=EXACT_TOL)   # (10 + 90) / 4
    assert scalar(Wg2) == pytest.approx(4.0, abs=EXACT_TOL)    # 1 + 3


# ---------------------------------------------------------------------------
# 3. median — NOT COMPOSABLE control
# ---------------------------------------------------------------------------


def test_median_is_not_composable():
    """Classification: **NOT COMPOSABLE**.

    The median of grouped medians is not the median of the union.  For
    ``[1, 2, 3, 100]`` the global median is ``(2 + 3) / 2 = 2.5``, but the
    subgroup medians ``[1,2] -> 1.5`` and ``[3,100] -> 51.5`` compose to
    ``(1.5·2 + 51.5·2) / 4 = 26.5``.  The denominator ``W`` is the *valid
    count* in every case (4), so this is pure statistic non-composability, not
    a lost denominator.  Median therefore has no meaningful ``SUM/WHT``
    composition and is included only as the non-associative stacking-family
    control for the final table.
    """
    Vg, Wg, _ = _stack_median(px([1.0, 2.0, 3.0, 100.0]), None, return_weights=True)
    A = _stack_median(px([1.0, 2.0]), None, return_weights=True)
    B = _stack_median(px([3.0, 100.0]), None, return_weights=True)
    Vh, Wh, _ = compose([(A[0], A[1]), (B[0], B[1])])

    assert scalar(Vg) == pytest.approx(2.5, abs=EXACT_TOL)
    assert scalar(Wg) == pytest.approx(4.0, abs=EXACT_TOL)
    assert Vh == pytest.approx(26.5, abs=EXACT_TOL)
    assert abs(Vh - scalar(Vg)) > NONASSOC_MIN_DELTA
    # Denominator is *not* lost: both are the valid count.
    assert Wh == pytest.approx(scalar(Wg), abs=EXACT_TOL)


# ---------------------------------------------------------------------------
# 4. kappa-sigma — APPROXIMATE BY DESIGN + effective survivor denominator
# ---------------------------------------------------------------------------


def test_kappa_sigma_global_vs_hierarchical_and_effective_denominator():
    """Classification: **APPROXIMATE BY DESIGN** (non-associative).

    Global kappa-sigma rejects the ``+1000`` outlier: median 100, std 300,
    ``3σ = 900``, so ``1100`` is clipped and ``V = 100``, ``W = 9``.  The
    hierarchical split keeps the outlier: subgroup ``[100×4, 1100]`` has
    ``3σ = 1200 > 1000`` so nothing is rejected locally, ``V = 300``, ``W = 5``.

        compose = (100·5 + 300·5) / 10 = 200   (global 100)

    This is the *expected* bounded-memory non-associativity.  Crucially the
    effective denominator is propagated correctly under the implemented
    survivor contract, not lost: global ``W = 9`` (the rejected sample's weight
    is excluded) vs hierarchical ``W = 5 + 5 = 10`` (the outlier survived
    locally and its weight is kept).  Rejected samples cease contributing in
    every local reduction; ``V·W`` equals the sum of survivor contributions.
    """
    vals = [100.0] * 9 + [1100.0]

    Vg, Wg, rej_g = _stack_kappa_sigma(px(vals), None, sigma_low=3.0,
                                       sigma_high=3.0, return_weights=True)
    A = _stack_kappa_sigma(px([100.0] * 5), None, sigma_low=3.0, sigma_high=3.0,
                           return_weights=True)
    B = _stack_kappa_sigma(px([100.0] * 4 + [1100.0]), None, sigma_low=3.0,
                           sigma_high=3.0, return_weights=True)
    Vh, Wh, Sh = compose([(A[0], A[1]), (B[0], B[1])])

    assert scalar(Vg) == pytest.approx(100.0, abs=EXACT_TOL)
    assert scalar(Wg) == pytest.approx(9.0, abs=EXACT_TOL)
    assert rej_g == pytest.approx(10.0)
    assert Vh == pytest.approx(200.0, abs=EXACT_TOL)
    assert abs(Vh - scalar(Vg)) > NONASSOC_MIN_DELTA

    # Independent effective-denominator proof (survivor contract):
    # global numerator V·W == 100·9 == 900 == Σ survivor contributions (9×100),
    # and the rejected +1000 contributes exactly 0.
    assert scalar(Vg) * scalar(Wg) == pytest.approx(9.0 * 100.0, abs=EXACT_TOL)
    # hierarchical: both subgroups keep their local survivors, W = 5 + 5.
    assert scalar(A[1]) == pytest.approx(5.0, abs=EXACT_TOL)
    assert scalar(B[1]) == pytest.approx(5.0, abs=EXACT_TOL)
    assert Wh == pytest.approx(10.0, abs=EXACT_TOL)


# ---------------------------------------------------------------------------
# 5. linear-fit clip — APPROXIMATE BY DESIGN + shift-invariance
# ---------------------------------------------------------------------------


def test_linear_fit_clip_global_vs_hierarchical_and_residual_centering():
    """Classification: **APPROXIMATE BY DESIGN** (non-associative).

    Like kappa-sigma, linear-fit clipping is median-centred and rejects the
    ``+1000`` outlier only in the *global* reduction (median 100, residual std
    ~316, ``3σ ≈ 948 < 1000``) but keeps it in the small subgroup (residual std
    ~447, ``3σ ≈ 1342 > 1000``), so ``global V = 100 / W = 9`` versus
    ``compose V = 200 / W = 10``.

    Linear-fit clip operates on median-centred *residuals* (``x - median``)
    rather than raw values, so its rejection decision — and hence the effective
    denominator ``W`` — is invariant to a global additive offset: adding the
    same constant to every observation rejects exactly the same samples
    (``W`` unchanged), while the resulting *location* ``V`` shifts by that same
    constant.  This residual centering is the documented distinction from
    kappa-sigma (which clips raw values against ``median ± κ·σ``).
    """
    vals = [100.0] * 9 + [1100.0]

    Vg, Wg, _ = _stack_linear_fit_clip(px(vals), None, sigma=3.0, return_weights=True)
    A = _stack_linear_fit_clip(px([100.0] * 5), None, sigma=3.0, return_weights=True)
    B = _stack_linear_fit_clip(px([100.0] * 4 + [1100.0]), None, sigma=3.0,
                               return_weights=True)
    Vh, Wh, _ = compose([(A[0], A[1]), (B[0], B[1])])

    assert scalar(Vg) == pytest.approx(100.0, abs=EXACT_TOL)
    assert scalar(Wg) == pytest.approx(9.0, abs=EXACT_TOL)
    assert Vh == pytest.approx(200.0, abs=EXACT_TOL)
    assert abs(Vh - scalar(Vg)) > NONASSOC_MIN_DELTA

    # Offset invariance of the *rejection decision* (denominator), while the
    # location statistic shifts with the offset.
    shifted = [v + 500.0 for v in vals]
    Vg_s, Wg_s, _ = _stack_linear_fit_clip(px(shifted), None, sigma=3.0,
                                           return_weights=True)
    assert scalar(Wg_s) == pytest.approx(scalar(Wg), abs=EXACT_TOL)   # same mask
    assert scalar(Vg_s) == pytest.approx(scalar(Vg) + 500.0, abs=EXACT_TOL)


# ---------------------------------------------------------------------------
# 6. winsorized sigma — substitution contract + non-associativity
# ---------------------------------------------------------------------------


def test_winsorized_sigma_substitution_contract_and_non_associative():
    """Classification: **APPROXIMATE BY DESIGN** for the statistic; the
    denominator follows the *substitution* contract exactly (no weight loss).

    With ``apply_rewinsor=True`` (production default) a rejected sample is
    **substituted** by the nearest survivor-distribution winsor bound and keeps
    its full weight.  On ``[100×9, 1100]`` with ``winsor_limits=(0.2, 0.2)`` the
    outlier is clipped to 100 and stays in the mean: ``V = 100`` and, unlike
    kappa-sigma, ``W = 10`` (not 9).  ``V·W = 1000 = 9·100 + 1·100`` — the
    substituted contribution is 100, neither the raw 1100 nor a dropped 0.

    The statistic is still non-associative: on ``[0..6, 100]`` the global
    reduction (winsor bound over 8 samples) gives ``V = 3.25`` while the split
    ``[0,1,2,3] + [4,5,6,100]`` gives ``1.5`` and ``28.75`` which compose to
    ``15.125`` — because the local winsor/sigma bounds are computed per
    subgroup.  ``W`` is identical (8) in both cases, so this is *value*
    non-associativity with a correctly retained denominator, not lost WHT.
    """
    # Substitution contract (denominator retained).
    vals = [100.0] * 9 + [1100.0]
    Vg, Wg, rej = _stack_winsorized_sigma(px(vals), None, kappa=3.0,
                                          winsor_limits=(0.2, 0.2),
                                          apply_rewinsor=True, return_weights=True)
    assert scalar(Vg) == pytest.approx(100.0, abs=EXACT_TOL)
    assert scalar(Wg) == pytest.approx(10.0, abs=EXACT_TOL)   # substitution keeps W
    assert scalar(Vg) * scalar(Wg) == pytest.approx(9 * 100.0 + 100.0, abs=EXACT_TOL)

    # Non-associativity of the winsorized statistic (denominator unchanged).
    vals2 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]
    Vg2, Wg2, _ = _stack_winsorized_sigma(px(vals2), None, kappa=3.0,
                                          winsor_limits=(0.2, 0.2),
                                          apply_rewinsor=True, return_weights=True)
    A = _stack_winsorized_sigma(px([0.0, 1.0, 2.0, 3.0]), None, kappa=3.0,
                                winsor_limits=(0.2, 0.2), apply_rewinsor=True,
                                return_weights=True)
    B = _stack_winsorized_sigma(px([4.0, 5.0, 6.0, 100.0]), None, kappa=3.0,
                                winsor_limits=(0.2, 0.2), apply_rewinsor=True,
                                return_weights=True)
    Vh2, Wh2, _ = compose([(A[0], A[1]), (B[0], B[1])])

    assert scalar(Vg2) == pytest.approx(3.25, abs=EXACT_TOL)
    assert Vh2 == pytest.approx(15.125, abs=EXACT_TOL)
    assert abs(Vh2 - scalar(Vg2)) > NONASSOC_MIN_DELTA
    assert scalar(Wg2) == pytest.approx(8.0, abs=EXACT_TOL)
    assert Wh2 == pytest.approx(8.0, abs=EXACT_TOL)   # denominator not lost


# ---------------------------------------------------------------------------
# 7. Winsorized singleton identity — post-fix regressions (was DEFECT)
# ---------------------------------------------------------------------------


def test_winsorized_singleton_identity_unweighted():
    """Post-fix regression (was DEFECT): a lone valid contribution is a
    no-rejection identity — ``V`` equals the sample exactly, ``W == 1``,
    ``rejected == 0`` — for both ``apply_rewinsor`` settings.

    Root cause of the former defect: the location/scale estimator used
    ``nanstd(..., ddof=1)``, which is NaN for a single valid sample, so
    ``low/high`` became NaN and the only valid sample was fully rejected;
    rewinsorization over the empty survivor set then produced non-finite
    (+inf) bounds.  The kernel now treats an exactly-one-valid column as a
    no-rejection identity, preserving the lone observation unchanged.
    """
    for apply_rewinsor in (True, False):
        V, W, rej = _stack_winsorized_sigma(
            px([100.0]), None, kappa=3.0, winsor_limits=(0.05, 0.05),
            apply_rewinsor=apply_rewinsor, return_weights=True,
        )
        assert np.isfinite(scalar(V)), apply_rewinsor
        assert scalar(V) == pytest.approx(100.0, abs=EXACT_TOL)
        assert scalar(W) == pytest.approx(1.0, abs=EXACT_TOL)
        assert rej == pytest.approx(0.0)


def test_winsorized_singleton_identity_weighted():
    """Post-fix regression: a lone valid contribution keeps its exact scientific
    weight in ``WHT`` (never collapsed to 1)."""
    V, W, rej = _stack_winsorized_sigma(
        px([100.0]), np.asarray([3.5], dtype=np.float32), kappa=3.0,
        winsor_limits=(0.05, 0.05), apply_rewinsor=True, return_weights=True,
    )
    assert np.isfinite(scalar(V))
    assert scalar(V) == pytest.approx(100.0, abs=EXACT_TOL)
    assert scalar(W) == pytest.approx(3.5, abs=EXACT_TOL)
    assert rej == pytest.approx(0.0)


def test_winsorized_singleton_stack_batch_identity():
    """Post-fix regression: the real ``_stack_batch`` (winsorized mode, one valid
    image) returns the finite identity instead of the former non-finite output.

    The singleton fast path protects the arithmetic ``mean`` only; winsorized
    mode enters the real kernel, so this witness exercises the production path
    end-to-end.
    """
    import types as _types
    from astropy.io import fits as _fits
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.logger = _types.SimpleNamespace(warning=lambda *a, **k: None,
                                      debug=lambda *a, **k: None,
                                      info=lambda *a, **k: None)
    o.stacking_mode = "winsorized-sigma-clip"
    o.normalize_method = "none"
    o.weighting_method = "none"
    o.use_quality_weighting = False
    o.weight_by_snr = True
    o.weight_by_stars = False
    o.snr_exponent = 1.0
    o.stars_exponent = 0.5
    o.min_weight = 0.0
    o.apply_batch_feathering = False
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
    o._is_plain_classic = lambda: False  # bypass normalization for this witness

    img = np.array([[100.0]], dtype=np.float32)
    item = (img, _fits.Header(), {"snr": 1.0, "stars": 0.0}, None,
            np.ones((1, 1), dtype=bool))
    Vb, _, Wb = o._stack_batch([item], 1, 1)
    assert np.isfinite(float(np.asarray(Vb).ravel()[0]))
    assert float(np.asarray(Vb).ravel()[0]) == pytest.approx(100.0, abs=EXACT_TOL)
    assert float(np.asarray(Wb).ravel()[0]) == pytest.approx(1.0, abs=EXACT_TOL)


def test_winsorized_partial_coverage_per_column_and_channel_identity():
    """Post-fix regression: partial spatial/channel coverage leaves some columns
    with exactly one valid sample and others with >=2.  The kernel must return
    finite ``V`` with correct per-column/per-channel ``WHT`` — lone
    contributors preserved at full weight, multi-sample columns reduced with
    the ordinary winsorized-sigma semantics, and zero false rejection.
    """
    # Grayscale (1, 3): columns 0 and 1 have exactly one valid sample each;
    # column 2 has two valid samples (50, 60) -> mean 55.
    imgs = [
        np.array([[100.0, np.nan, np.nan]], dtype=np.float32),
        np.array([[np.nan, 200.0, 50.0]], dtype=np.float32),
        np.array([[np.nan, np.nan, 60.0]], dtype=np.float32),
    ]
    V, W, rej = _stack_winsorized_sigma(imgs, None, kappa=3.0,
                                        winsor_limits=(0.05, 0.05),
                                        apply_rewinsor=True, return_weights=True)
    V = np.asarray(V).ravel()
    W = np.asarray(W).ravel()
    assert np.all(np.isfinite(V))
    assert V == pytest.approx([100.0, 200.0, 55.0], abs=EXACT_TOL)
    assert W == pytest.approx([1.0, 1.0, 2.0], abs=EXACT_TOL)
    assert rej == pytest.approx(0.0)

    # Weighted variant: lone contributors keep their exact non-unit weights;
    # column 2 -> (50·2 + 60·4) / 6 = 340/6, WHT = 6.
    weights = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    Vw, Ww, _ = _stack_winsorized_sigma(imgs, weights, kappa=3.0,
                                        winsor_limits=(0.05, 0.05),
                                        apply_rewinsor=True, return_weights=True)
    Vw = np.asarray(Vw).ravel()
    Ww = np.asarray(Ww).ravel()
    assert np.all(np.isfinite(Vw))
    assert Vw == pytest.approx([100.0, 200.0, 340.0 / 6.0], abs=EXACT_TOL)
    assert Ww == pytest.approx([1.0, 2.0, 6.0], abs=EXACT_TOL)

    # Colour (1, 1, 3): channel 0 has one valid sample, channel 1 two, channel 2
    # three — per-channel WHT stays independent and finite.
    c0 = np.array([[[100.0, np.nan, np.nan]]], dtype=np.float32)
    c1 = np.array([[[np.nan, 40.0, 10.0]]], dtype=np.float32)
    c2 = np.array([[[np.nan, 60.0, 20.0]]], dtype=np.float32)
    c3 = np.array([[[np.nan, np.nan, 30.0]]], dtype=np.float32)
    Vc, Wc, _ = _stack_winsorized_sigma([c0, c1, c2, c3], None, kappa=3.0,
                                        winsor_limits=(0.05, 0.05),
                                        apply_rewinsor=True, return_weights=True)
    Vc = np.asarray(Vc).ravel()
    Wc = np.asarray(Wc).ravel()
    assert np.all(np.isfinite(Vc))
    assert Vc == pytest.approx([100.0, 50.0, 20.0], abs=EXACT_TOL)
    assert Wc == pytest.approx([1.0, 2.0, 3.0], abs=EXACT_TOL)

    # Zero-valid control: the singleton guard must not manufacture either a
    # value or a denominator for an entirely missing column.
    missing = [np.array([[np.nan]], dtype=np.float32) for _ in range(3)]
    V0, W0, rej0 = _stack_winsorized_sigma(
        missing, None, kappa=3.0, winsor_limits=(0.05, 0.05),
        apply_rewinsor=True, return_weights=True,
    )
    assert np.isfinite(scalar(V0))
    assert scalar(V0) == pytest.approx(0.0, abs=EXACT_TOL)
    assert scalar(W0) == pytest.approx(0.0, abs=EXACT_TOL)
    assert rej0 == pytest.approx(0.0)


def test_linear_fit_clip_hyphen_alias_routes_to_real_clipping():
    """Regression (was DEFECT, now fixed): the GUI hyphen spelling
    ``"linear-fit-clip"`` routes to the real linear-fit clipping kernel and
    produces exactly the same V / effective-WHT / rejection semantics as the
    canonical underscore key — never the arithmetic mean.

    The GUI derives ``stacking_mode = stack_method.replace("_", "-")``
    (``seestar/gui/settings.py`` ~259) and the Qt shell advertises
    ``"linear-fit-clip"`` as its backend key, so ``linear_fit_clip`` reaches the
    backend as ``"linear-fit-clip"``.  ``_is_linear_fit_clip_mode`` normalizes
    both spellings at the boundary (mirroring ``_is_winsorized_mode``), so the
    hyphen spelling can no longer fall through to the mean.

    On ``[100×9, 1100]`` the outlier is rejected (``V=100, W=9, rej=10%``) by
    both spellings, while the arithmetic mean averages it in
    (``V=200, W=10, rej=0%``).
    """
    vals = [100.0] * 9 + [1100.0]

    v_hyphen, w_hyphen, rej_hyphen = worker("linear-fit-clip", vals)
    v_underscore, w_underscore, rej_underscore = worker("linear_fit_clip", vals)
    v_mean, w_mean, rej_mean = worker("mean", vals)

    # Hyphen == underscore == real clipping (outlier rejected).
    assert v_hyphen == pytest.approx(100.0, abs=EXACT_TOL)
    assert w_hyphen == pytest.approx(9.0, abs=EXACT_TOL)
    assert rej_hyphen == pytest.approx(10.0)
    assert v_hyphen == pytest.approx(v_underscore, abs=EXACT_TOL)
    assert w_hyphen == pytest.approx(w_underscore, abs=EXACT_TOL)
    assert rej_hyphen == pytest.approx(rej_underscore, abs=EXACT_TOL)

    # ... and observably distinct from the arithmetic mean.
    assert v_mean == pytest.approx(200.0, abs=EXACT_TOL)
    assert w_mean == pytest.approx(10.0, abs=EXACT_TOL)
    assert rej_mean == pytest.approx(0.0)
    assert abs(v_hyphen - v_mean) > NONASSOC_MIN_DELTA


def test_linear_fit_clip_hyphen_alias_backend_dispatch(tmp_path, monkeypatch):
    """The GUI hyphen spelling reaches the real linear-fit clipping kernel in the
    RAM, tiled/HQ and memmap production backends — never the mean fallthrough.

    Uses the deterministic ``[A×9, A+1000]`` HWC dataset so linear-fit clipping
    rejects the outlier (``W`` drops from 10 to 9) while the mean keeps it
    (``V = A + 100``).  For each backend, ``"linear-fit-clip"`` and
    ``"linear_fit_clip"`` must produce matching ``V`` and ``W`` equal to the
    inlier ``A``, and both must remain observably distinct from ``mean``.  The
    tiled/HQ and memmap runs use ``group_size >= N`` (mirroring the parity
    harness) so the witness isolates *storage/backend dispatch* from local
    nonlinear subgrouping.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    A, items = _rejection_items()

    def run(mode, backend):
        if backend == "ram":
            s = _min_stack(mode, max_hq_mem=1_000_000_000, batch_size=10)
        elif backend == "tiled":
            s = _min_stack(mode, max_hq_mem=100_000, batch_size=10, tile_h=8)
        elif backend == "memmap":
            s = _min_stack(mode, max_hq_mem=100_000, batch_size=1, tile_h=8)
        else:
            raise ValueError(backend)
        V_raw, _, W = s._stack_batch(items, 1, 1)
        V = np.array(V_raw, dtype=np.float64, copy=True)
        _release_memmap(V_raw)
        return V, np.array(W, dtype=np.float64)

    results = {}
    for mode in ("linear-fit-clip", "linear_fit_clip", "mean"):
        for backend in ("ram", "tiled", "memmap"):
            results[(mode, backend)] = run(mode, backend)

    for backend in ("ram", "tiled", "memmap"):
        Vh, Wh = results[("linear-fit-clip", backend)]
        Vu, Wu = results[("linear_fit_clip", backend)]
        Vm, Wm = results[("mean", backend)]

        # Hyphen and underscore both reach real clipping: V == A (inlier).
        assert np.allclose(Vh, A, rtol=1e-5, atol=1e-2), (
            backend, float(np.abs(Vh - A).max())
        )
        assert np.allclose(Vh, Vu, rtol=1e-5, atol=1e-3), (
            backend, float(np.abs(Vh - Vu).max())
        )
        # Effective denominator drops to 9 (outlier rejected) for both
        # spellings; the mean keeps the full count of 10.
        assert np.allclose(Wh, 9.0, atol=1e-3), (backend, Wh.min(), Wh.max())
        assert np.allclose(Wh, Wu, rtol=1e-5, atol=1e-3), backend
        assert np.allclose(Wm, 10.0, atol=1e-3), backend
        # Mean is observably distinct from real clipping.
        assert float(np.abs(Vm - Vh).max()) > NONASSOC_MIN_DELTA, backend

    # No memmap artifacts may remain after release.
    assert not glob.glob(os.path.join(str(tmp_path), "hq_batch*.dat*"))


# ---------------------------------------------------------------------------
# 8. Per-channel / partial-coverage effective denominator
# ---------------------------------------------------------------------------


def test_per_channel_rejection_denominator():
    """Rejection is per-channel and per-pixel: a per-channel outlier is rejected
    only in its own channel, leaving the other channels' denominator intact.
    Missing (``NaN``) samples never contribute.

    Colour observation ``(1, 1, 3)``, outlier in the red channel only:
    ``W = [9, 10, 10]`` and ``V = [100, 100, 100]``.
    """
    inl = np.array([[[100.0, 100.0, 100.0]]], dtype=np.float32)
    out = np.array([[[1100.0, 100.0, 100.0]]], dtype=np.float32)
    imgs = [inl.copy() for _ in range(9)] + [out.copy()]

    V, W, rej = _stack_kappa_sigma(imgs, None, sigma_low=3.0, sigma_high=3.0,
                                   return_weights=True)
    V = np.asarray(V).ravel()
    W = np.asarray(W).ravel()
    assert V == pytest.approx([100.0, 100.0, 100.0], abs=EXACT_TOL)
    assert W == pytest.approx([9.0, 10.0, 10.0], abs=EXACT_TOL)

    # Partial coverage: a missing sample is excluded from the denominator and
    # never treated as a numeric observation.
    a = np.array([[10.0]], dtype=np.float32)
    b = np.array([[np.nan]], dtype=np.float32)
    c = np.array([[100.0]], dtype=np.float32)
    V2, W2, _ = _stack_kappa_sigma([a, b, c], None, sigma_low=3.0,
                                   sigma_high=3.0, return_weights=True)
    assert scalar(W2) == pytest.approx(2.0, abs=EXACT_TOL)   # NaN excluded
    assert scalar(V2) == pytest.approx(55.0, abs=EXACT_TOL)  # median(10, 100)
