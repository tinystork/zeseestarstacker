"""Pure float preview-analysis core (ZSSS-OTPUX-PREVIEW-CORE-01) — display-only.

Toolkit-free float-domain analysis for the Qt preview pipeline.  This module
implements the ratified contracts in ``docs/output_truthfulness_preview_audit.md``:

* §5.2 — Option A: backend carries ``(legacy_normalized, raw_linear)``; Qt owns
  the stable/adaptive anchor mapping (p0.5 / p95, finite min/max fallback only
  when degenerate), so small changes preserve a fixed pixel's mapping while
  genuine photometric drift can widen the display range;
* §5.3 — 512-bin float histogram over ``[0, 1]`` (RGB overlay or L mono),
  ``log1p`` visualization counts, per-channel min/max/median/mean/std on the
  *exact same* deterministic sample, robust plotted X range + explicit full
  ``[0, 1]`` range metadata;
* §5.5 — Auto Stretch background-population algorithm (no min/max
  renormalization);
* §5.6 — Auto WB true-background-band algorithm.

It is deliberately *pure*:

* no Qt widgets / ``QImage`` (the legacy ``uint8`` ``QImage`` estimators remain
  in ``preview_adjust.py``; this module consumes/produces float ``ndarray``
  only);
* no scientific-engine imports (it never touches the science / alignment /
  enhancement / queue-manager modules);
* numpy is imported *lazily*, inside each function, so a fresh
  ``import seestar.gui_qt`` (or even ``import seestar.gui_qt.preview_analysis``)
  never pulls numpy into ``sys.modules``;
* every helper returns a fresh array and never mutates its input — the backend
  ``raw_linear`` array, the scientific accumulators and the caller's arrays are
  all left bit-identical.

Inputs are float arrays in one of two layouts: 2D mono ``(H, W)`` or 3D
channels-last ``(H, W, C)`` (RGB/RGBA uses the first three channels).  All
algorithms operate on the ``[0, 1]`` domain produced by the anchor mapping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Ratified constants
# --------------------------------------------------------------------------

# §5.3: exactly 512 bins over [0, 1] (float domain).
HISTOGRAM_BINS = 512
HISTOGRAM_RANGE = (0.0, 1.0)

# §5.2: anchor separation epsilon.
ANCHOR_SEP = 1e-4

# §5.2 anchor percentiles.  The low anchor is the robust dark floor
# (p0.5); the high anchor is the *scene* top (p95), not the bright star
# tail (p99.5).  Anchoring the high end to the star tail makes the display
# mapping track the brightest pixels, which drift independently of the
# stable scene on deep stacks (S/N improvement, transient bright frames,
# exposure changes).  The monotonic drift ratchet then widens the span
# permanently, compressing the scene toward 0 in mapped space and driving
# Auto Stretch's black point to 0 on long runs (progressive darkening).
# Anchoring to the scene top keeps the mapping stable while the top
# (100 - ANCHOR_HI_PCT)% of pixels (stars) are allowed to saturate - normal
# and desirable for astro display.  Display-only.
ANCHOR_LO_PCT = 0.5
ANCHOR_HI_PCT = 95.0

# §5.2 (drift accommodation): hysteresis dead-band for the frozen-anchor
# display mapping.  A new raw frame re-anchors (widens) only when its robust
# percentile range (p0.5 / p95) has drifted beyond the frozen range by more
# than this fraction of the frozen span.  This keeps small frame-to-frame
# photometric evolution *stable* (anti-pumping) while a legitimate 2x-3x
# global drift widens the mapping before the preview white-outs.  Display-only;
# it never touches the scientific accumulators.
ANCHOR_DRIFT_HYSTERESIS = 0.25

# §5.2 / §5.3: deterministic sampling cap (documented).  At most this many
# pixels are fed to the percentile/median/histogram computations; larger arrays
# are subsampled with a *fixed stride* so results are deterministic.
MAX_SAMPLE_PIXELS = 1_000_000

# §5.5 Auto Stretch constants.
AUTO_STRETCH_MIN_SAMPLE = 20
AUTO_STRETCH_DEFAULTS = (0.01, 0.99)
AUTO_STRETCH_PCT_LO = 0.5
AUTO_STRETCH_PCT_BG = 60.0
AUTO_STRETCH_PCT_HI = 99.5
AUTO_STRETCH_BG_SPREAD_BP = 2.8
AUTO_STRETCH_BG_SPREAD_WP = 8.0
AUTO_STRETCH_MAD_SCALE = 1.4826

# §5.6 Auto WB constants.
AUTO_WB_SATURATION = 0.98
AUTO_WB_MIN_SAMPLE = 64
AUTO_WB_CENTRE_FLOOR = 1e-6
AUTO_WB_GAIN_MIN = 0.2
AUTO_WB_GAIN_MAX = 5.0
AUTO_WB_LUMA = (0.299, 0.587, 0.114)
NEUTRAL_WB = (1.0, 1.0, 1.0)

# §5.3 point 6: robust plotted X range percentiles.
X_RANGE_PCT_LO = 0.5
X_RANGE_PCT_HI = 99.5


def _load_numpy():
    """Lazily import numpy (the module object, or ``None`` when unavailable)."""
    try:
        import importlib

        return importlib.import_module("numpy")
    except Exception:
        return None


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _cap_sample(np: Any, vals, max_pixels: int = MAX_SAMPLE_PIXELS):
    """Deterministic fixed-stride subsample bounding ``vals`` to ``max_pixels``.

    ``vals`` is a 1D array in C (row-major) order.  When it is longer than the
    cap we keep every ``ceil(n / max_pixels)``-th element, so the result has at
    most ``max_pixels`` elements and is fully deterministic (no RNG).
    """
    n = int(vals.size)
    if n <= max_pixels:
        return vals
    stride = max(1, int(-(-n // max_pixels)))  # ceil(n / max_pixels)
    return vals[::stride]


def _finite_positive_sample(np: Any, arr):
    """Deterministic finite-positive sample (finite and > 0), capped/stride.

    Returns ``None`` when no finite-positive element exists.
    """
    vals = arr[np.isfinite(arr) & (arr > 0.0)]
    if vals.size == 0:
        return None
    return _cap_sample(np, vals)


def _luminance(np: Any, arr):
    """Rec.601 luminance of a mapped float buffer (2D or ``(H, W, 3+)``)."""
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return (
            AUTO_WB_LUMA[0] * arr[..., 0]
            + AUTO_WB_LUMA[1] * arr[..., 1]
            + AUTO_WB_LUMA[2] * arr[..., 2]
        )
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]
    return arr


def _as_float_array(np: Any, obj: Any) -> Optional[Any]:
    """Return ``obj`` as a validated 2D/3D float array, or ``None``."""
    try:
        arr = np.asarray(obj)
    except Exception:
        return None
    if arr.dtype.kind not in "fiu":
        return None
    if arr.ndim not in (2, 3):
        return None
    if arr.size == 0:
        return None
    return arr.astype(np.float64, copy=True)


def _analysis_channels(np: Any, arr):
    """Return ``[(name, 1D capped sample)]`` for the analysis buffer.

    RGB (>=3 channels) yields ``R``/``G``/``B`` (first three channels); mono 2D
    (or ``(H, W, 1)``) yields a single ``"L"`` channel.  The same deterministic
    capped sample is used for both the histogram and the per-channel stats
    (§5.3 point 4/5), so "same-sample" semantics hold by construction.
    """
    if arr.ndim == 2:
        return [("L", _cap_sample(np, arr.ravel()))]
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return [("L", _cap_sample(np, arr[..., 0].ravel()))]
        if arr.shape[2] >= 3:
            return [
                ("R", _cap_sample(np, arr[..., 0].ravel())),
                ("G", _cap_sample(np, arr[..., 1].ravel())),
                ("B", _cap_sample(np, arr[..., 2].ravel())),
            ]
    return []


def _sample_stats(np: Any, sample):
    """Per-channel ``{min, max, median, mean, std}`` over the given sample."""
    return {
        "min": float(np.min(sample)),
        "max": float(np.max(sample)),
        "median": float(np.median(sample)),
        "mean": float(np.mean(sample)),
        "std": float(np.std(sample)),
    }


def _in_domain_sample(np: Any, sample):
    """Finite ``[0, 1]`` values of a capped sample.

    The histogram counts and the per-channel stats must describe the *exact
    same* finite in-domain ``[0, 1]`` sample (§5.3 point 5): non-finite values
    and values outside ``[0, 1]`` are excluded here, so ``np.histogram`` never
    silently drops a value that the stats would otherwise describe.
    """
    return sample[np.isfinite(sample) & (sample >= 0.0) & (sample <= 1.0)]


def _robust_x_range_from_samples(np: Any, channels) -> Tuple[float, float]:
    """Robust plotted X range from the *finite* part of the channel samples.

    Guarded against empty / degenerate samples: returns the full ``(0, 1)``
    range when nothing usable remains.
    """
    parts = []
    for _, sample in channels:
        in_domain = _in_domain_sample(np, sample)
        if in_domain.size:
            parts.append(in_domain)
    if not parts:
        return (0.0, 1.0)
    all_vals = np.concatenate(parts)
    lo = float(np.percentile(all_vals, X_RANGE_PCT_LO))
    hi = float(np.percentile(all_vals, X_RANGE_PCT_HI))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return (0.0, 1.0)
    return (lo, hi)


# --------------------------------------------------------------------------
# §5.2 — Option A payload extraction + stable/adaptive anchor mapping
# --------------------------------------------------------------------------

def extract_raw_linear(data: Any) -> Optional[Any]:
    """Extract an immutable raw-linear float array from a preview payload.

    Option A payloads are tuples ``(legacy_normalized, raw_linear)``: the
    *second* element is returned as an independent float copy.  Legacy
    single-array payloads (a lone array, or a tuple whose second element is not
    a 2D/3D array) fall back to a copy of the first/only array, so old
    producers keep working.  Returns ``None`` for missing / non-array data.

    Empty tuples/lists and one-element malformed sequences return ``None``
    without raising.
    """
    np = _load_numpy()
    if np is None:
        return None
    arr = None
    if isinstance(data, (tuple, list)):
        if len(data) >= 2:
            arr = _as_float_array(np, data[1])
        if arr is None and len(data) >= 1:
            arr = _as_float_array(np, data[0])
    else:
        arr = _as_float_array(np, data)
    return arr  # already a fresh float64 copy


def compute_anchors(raw_linear, sep: float = ANCHOR_SEP) -> Tuple[float, float]:
    """Compute fixed normalization anchors ``(lo, hi)`` from a raw-linear array.

    §5.2 Option A: anchors come from a deterministic finite-positive sample
    (``percentile(sample, ANCHOR_LO_PCT)`` / ``percentile(sample, ANCHOR_HI_PCT)``), falling back
    to the finite min/max *only* when that sample is degenerate (empty,
    non-finite, or ``hi <= lo + sep``).  Always returns ``(lo, hi)`` with
    ``hi > lo`` so the mapping is non-degenerate.
    """
    np = _load_numpy()
    if np is None:
        return (0.0, 1.0)
    arr = np.asarray(raw_linear, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 1.0)

    sample = _finite_positive_sample(np, arr)
    if sample is not None and sample.size > 0:
        lo = float(np.percentile(sample, ANCHOR_LO_PCT))
        hi = float(np.percentile(sample, ANCHOR_HI_PCT))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo + sep:
            return (lo, hi)

    # Degenerate percentile sample: fall back to the finite min/max.
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi - lo > sep:
        return (lo, hi)
    # Constant finite image: widen symmetrically so the value maps to 0.5.
    mid = 0.5 * (lo + hi)
    return (mid - sep, mid + sep)


def map_raw_linear(raw_linear, anchor_lo: float, anchor_hi: float) -> Any:
    """Map raw-linear values through frozen anchors into ``[0, 1]``.

    ``mapped = clip((raw - lo) / (hi - lo), 0, 1)`` using the same anchors
    across successive previews (§5.2 regression: a fixed raw pixel maps
    identically when later-frame extrema change).  Non-mutating.  For drift
    accommodation, callers re-anchor via :func:`adapt_anchors_for_drift` before
    calling this mapping.
    """
    np = _load_numpy()
    if np is None:
        return None
    arr = np.asarray(raw_linear, dtype=np.float64)
    lo = float(anchor_lo)
    hi = float(anchor_hi)
    denom = hi - lo
    if denom <= 0.0:
        denom = ANCHOR_SEP
    with np.errstate(invalid="ignore", divide="ignore"):
        mapped = (arr - lo) / denom
    return np.clip(mapped, 0.0, 1.0)


def adapt_anchors_for_drift(
    anchor_lo: float,
    anchor_hi: float,
    raw_linear,
    hysteresis: float = ANCHOR_DRIFT_HYSTERESIS,
    sep: float = ANCHOR_SEP,
) -> Tuple[float, float]:
    """Return the effective frozen anchors for a new raw-linear frame.

    §5.2 drift accommodation (display-only).  The frozen anchors are kept
    bit-identical while the new frame's robust percentile range (p0.5 / p95)
    stays within a hysteresis band around them — so a fixed raw pixel keeps
    mapping identically across small (frame-to-frame) evolution, preserving the
    anti-pumping intent.  When the new frame's robust range has drifted beyond
    that band, the anchors are widened **monotonically outward** (a ratchet:
    ``lo`` only decreases, ``hi`` only increases) to cover the new robust
    range, so a legitimate 2x-3x photometric drift no longer maps the bulk of
    the image to exactly ``1.0`` (artificial saturation / white-out).

    Monotonicity means the mapping can only "zoom out" across a context, never
    oscillate: once widened for a brighter stack, a transient dimmer frame does
    not shrink the range back (strong temporal anti-pumping).  The expansion is
    *bounded* — it is driven by the new frame's robust p0.5/p99.5 (never by a
    single outlier pixel) and never overshoots the data.  Returns a fresh
    ``(lo, hi)`` pair with ``hi > lo``; the input array is never mutated.

    A degenerate new frame (no finite-positive sample) carries no drift
    information and leaves the anchors unchanged.
    """
    np = _load_numpy()
    if np is None:
        return (float(anchor_lo), float(anchor_hi))
    arr = np.asarray(raw_linear, dtype=np.float64)
    if arr.size == 0:
        return (float(anchor_lo), float(anchor_hi))

    lo = float(anchor_lo)
    hi = float(anchor_hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        # No usable frozen anchors: fall back to a fresh anchor computation.
        return compute_anchors(arr, sep=sep)

    sample = _finite_positive_sample(np, arr)
    if sample is None or sample.size == 0:
        return (lo, hi)
    cur_lo = float(np.percentile(sample, ANCHOR_LO_PCT))
    cur_hi = float(np.percentile(sample, ANCHOR_HI_PCT))
    if not (np.isfinite(cur_lo) and np.isfinite(cur_hi)):
        return (lo, hi)

    span = hi - lo
    if span <= 0.0:
        span = sep
    band = float(hysteresis) * span

    new_lo = lo
    new_hi = hi
    # Bright drift: the robust high tail escapes the frozen range -> widen up.
    if cur_hi > hi + band:
        new_hi = cur_hi
    # Dark drift: the robust low tail escapes the frozen range -> widen down.
    if cur_lo < lo - band:
        new_lo = cur_lo

    # Never return a degenerate pair; symmetric widening is a last-resort guard
    # and only triggers when the frozen anchors were already degenerate.
    if new_hi - new_lo <= sep:
        mid = 0.5 * (new_lo + new_hi)
        new_lo = mid - sep
        new_hi = mid + sep
    return (new_lo, new_hi)


# --------------------------------------------------------------------------
# §5.3 — WB-only derivation + float histogram / stats / X range
# --------------------------------------------------------------------------

def apply_wb_float(mapped, wb=NEUTRAL_WB) -> Any:
    """Apply white-balance gains to a mapped float ``[0, 1]`` buffer.

    Produces the WB-only analysis buffer (§5.3): per-channel multiply by the
    R/G/B gains, clipped to ``[0, 1]``.  Mono (2D) data is unaffected.  Returns
    a fresh array; the input is never mutated.
    """
    np = _load_numpy()
    if np is None:
        return None
    arr = np.asarray(mapped, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr.copy()
    r, g, b = (float(wb[0]), float(wb[1]), float(wb[2]))
    out = arr.copy()
    out[..., 0] = np.clip(arr[..., 0] * r, 0.0, 1.0)
    out[..., 1] = np.clip(arr[..., 1] * g, 0.0, 1.0)
    out[..., 2] = np.clip(arr[..., 2] * b, 0.0, 1.0)
    return out


def compute_histogram_float(mapped) -> Optional[Dict[str, Any]]:
    """Compute the §5.3 float histogram + stats from a mapped ``[0, 1]`` buffer.

    Returns a dict with:

    * ``bins`` / ``range`` — ``HISTOGRAM_BINS`` (512) bins over ``[0, 1]``
      (fixed by contract; there is no per-call bin override);
    * ``channels`` — ``["L"]`` (mono) or ``["R", "G", "B"]``;
    * ``counts`` — per-channel ``int64`` bin counts;
    * ``log_counts`` — ``log1p(counts)`` visualization counts (empty bin == 0);
    * ``stats`` — per-channel ``{min, max, median, mean, std}`` computed on the
      *exact same* deterministic sample as ``counts``;
    * ``x_range`` — robust plotted X range (percentile-based);
    * ``full_range`` — explicit ``(0.0, 1.0)`` full-domain metadata.

    The histogram counts and all five stats are computed over the *exact same*
    finite in-domain ``[0, 1]`` sample (non-finite values and values outside
    ``[0, 1]`` are excluded from both).  When a required channel has no usable
    in-domain sample the analysis fails closed and returns ``None`` — it never
    fabricates synthetic pixels.

    Returns ``None`` for missing / unusable input.
    """
    np = _load_numpy()
    if np is None:
        return None
    arr = np.asarray(mapped, dtype=np.float64)
    if arr.size == 0 or arr.ndim not in (2, 3):
        return None
    channels = _analysis_channels(np, arr)
    if not channels:
        return None
    counts: Dict[str, Any] = {}
    log_counts: Dict[str, Any] = {}
    stats: Dict[str, Dict[str, float]] = {}
    for name, sample in channels:
        in_domain = _in_domain_sample(np, sample)
        if in_domain.size == 0:
            # Fail closed: never fabricate a synthetic pixel for an unusable
            # required channel (all-NaN / out-of-domain channel).
            return None
        hist, _ = np.histogram(in_domain, bins=HISTOGRAM_BINS, range=HISTOGRAM_RANGE)
        hist = hist.astype(np.int64)
        counts[name] = hist
        log_counts[name] = np.log1p(hist.astype(np.float64))
        stats[name] = _sample_stats(np, in_domain)
    return {
        "bins": HISTOGRAM_BINS,
        "range": HISTOGRAM_RANGE,
        "channels": [name for name, _ in channels],
        "counts": counts,
        "log_counts": log_counts,
        "stats": stats,
        "x_range": _robust_x_range_from_samples(np, channels),
        "full_range": (0.0, 1.0),
    }


def compute_histogram_stats_float(mapped) -> Optional[Dict[str, Dict[str, float]]]:
    """Per-channel ``{min, max, median, mean, std}`` from the §5.3 sample.

    Thin wrapper over :func:`compute_histogram_float` so the stats are always
    derived from the *exact same* deterministic sample as the histogram counts.
    """
    result = compute_histogram_float(mapped)
    if result is None:
        return None
    return result["stats"]


def compute_robust_x_range(mapped) -> Tuple[float, float]:
    """Robust plotted X range for the analysis buffer (§5.3 point 6).

    Percentile-based over the finite sample, guarded against empty/degenerate
    samples; the explicit full ``[0, 1]`` range is the caller's toggle.
    """
    np = _load_numpy()
    if np is None:
        return (0.0, 1.0)
    arr = np.asarray(mapped, dtype=np.float64)
    if arr.size == 0 or arr.ndim not in (2, 3):
        return (0.0, 1.0)
    channels = _analysis_channels(np, arr)
    if not channels:
        return (0.0, 1.0)
    return _robust_x_range_from_samples(np, channels)


# --------------------------------------------------------------------------
# §5.5 — Auto Stretch (background-population algorithm)
# --------------------------------------------------------------------------

def _stretch_sample(np: Any, arr, mask=None):
    """§5.5 input sample ``S``: finite WB-only mapped values, excluding exact 0/1.

    RGB data is reduced to its Rec.601 luminance (a single global BP/WP pair
    applies to every channel); mono data is used directly.  A 2D validity mask
    (``mask[pixel] > 0``) is applied when provided.  The result is capped via
    the deterministic stride.
    """
    lum = _luminance(np, arr)
    flat = lum.ravel()
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 2 and m.shape == lum.shape and m.size == flat.size:
            flat = flat[m.ravel() > 0]
    finite = flat[np.isfinite(flat)]
    finite = finite[(finite > 0.0) & (finite < 1.0)]
    if finite.size == 0:
        return None
    return _cap_sample(np, finite)


def compute_auto_stretch_float(mapped, mask=None, sep: float = ANCHOR_SEP) -> Tuple[float, float]:
    """Auto Stretch black/white points (§5.5 exact algorithm).

    Operates on the WB-only mapped float ``[0, 1]`` buffer.  Steps:

    1. keep finite pixels (and ``mask[pixel] > 0`` when a mask is given),
       excluding exact clipped ``0.0`` / ``1.0``;
    2. ``|S| < 20`` -> deterministic defaults ``(0.01, 0.99)``;
    3. ``p005 = percentile(S, 0.5)``, ``p60 = percentile(S, 60)``,
       ``p995 = percentile(S, 99.5)``;
    4. background ``B = { s <= p60 }``; ``bg = median(B)``;
       ``sigma = 1.4826 * MAD(B)``;
    5. ``bp = clip(max(p005, bg - 2.8 sigma), 0, 1 - sep)``;
    6. ``wp = clip(max(p995, bg + 8 sigma, bp + sep), bp + sep, 1)``;
    7. degenerate fallback (empty/non-finite ``B`` or ``sigma == 0``):
       ``bp = p005``, ``wp = p995``, then clamp/separate as above.

    There is **no** min/max renormalization step.
    """
    np = _load_numpy()
    if np is None:
        return AUTO_STRETCH_DEFAULTS
    arr = np.asarray(mapped, dtype=np.float64)
    if arr.size == 0 or arr.ndim not in (2, 3):
        return AUTO_STRETCH_DEFAULTS

    S = _stretch_sample(np, arr, mask)
    if S is None or S.size < AUTO_STRETCH_MIN_SAMPLE:
        return AUTO_STRETCH_DEFAULTS

    p005 = float(np.percentile(S, AUTO_STRETCH_PCT_LO))
    p60 = float(np.percentile(S, AUTO_STRETCH_PCT_BG))
    p995 = float(np.percentile(S, AUTO_STRETCH_PCT_HI))

    B = S[S <= p60]
    if B.size == 0 or not np.all(np.isfinite(B)):
        bp, wp = p005, p995
    else:
        bg = float(np.median(B))
        mad = float(np.median(np.abs(B - bg)))
        sigma = AUTO_STRETCH_MAD_SCALE * mad
        if sigma == 0.0 or not np.isfinite(sigma):
            bp, wp = p005, p995
        else:
            bp = max(p005, bg - AUTO_STRETCH_BG_SPREAD_BP * sigma)
            wp = max(p995, bg + AUTO_STRETCH_BG_SPREAD_WP * sigma)

    # Step 5: clip bp; step 6: clip/separate wp.  The ``bp + sep`` lower bound
    # of the clip enforces the spec's ``max(..., bp + sep)`` term exactly.
    bp = float(np.clip(bp, 0.0, 1.0 - sep))
    wp = float(np.clip(wp, bp + sep, 1.0))
    return (bp, wp)


# --------------------------------------------------------------------------
# §5.6 — Auto WB (true-background-band algorithm)
# --------------------------------------------------------------------------

def compute_auto_wb_float(mapped) -> Tuple[float, float, float]:
    """Auto WB gains (§5.6 true-background-band algorithm).

    Estimates from the *pristine pre-WB* mapped float ``[0, 1]`` source (never
    from already-WB pixels), so repeated AutoWB is deterministic/idempotent.

    1. common finite RGB set (>=3 channels, all three channels finite);
    2. every channel in ``(0, 0.98)`` (excludes zero/dark borders and
       near-saturated stars);
    3. true-background luminance band: keep ``p5 <= lum <= p60`` (degenerate
       ``p60 <= p5`` -> neutral);
    4. per-channel centre = median on the *same* selected pixels;
    5. gains relative green, clipped to ``[0.2, 5.0]``; any ``centre <= 1e-6``
       -> neutral;
    6. fewer than 64 valid pixels after exclusion -> neutral.

    Deterministic sampling cap applies before the percentile/median work.
    """
    np = _load_numpy()
    if np is None:
        return NEUTRAL_WB
    arr = np.asarray(mapped, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return NEUTRAL_WB

    R = arr[..., 0]
    G = arr[..., 1]
    B = arr[..., 2]
    valid = (
        np.isfinite(R)
        & np.isfinite(G)
        & np.isfinite(B)
        & (R > 0.0)
        & (G > 0.0)
        & (B > 0.0)
        & (R < AUTO_WB_SATURATION)
        & (G < AUTO_WB_SATURATION)
        & (B < AUTO_WB_SATURATION)
    )

    idx = np.flatnonzero(valid.ravel())
    if idx.size > MAX_SAMPLE_PIXELS:
        stride = max(1, int(-(-idx.size // MAX_SAMPLE_PIXELS)))
        idx = idx[::stride]
    if idx.size < AUTO_WB_MIN_SAMPLE:
        return NEUTRAL_WB

    Rv = R.ravel()[idx]
    Gv = G.ravel()[idx]
    Bv = B.ravel()[idx]
    lum = AUTO_WB_LUMA[0] * Rv + AUTO_WB_LUMA[1] * Gv + AUTO_WB_LUMA[2] * Bv

    p5 = float(np.percentile(lum, 5.0))
    p60 = float(np.percentile(lum, 60.0))
    if not (p60 > p5):
        return NEUTRAL_WB

    band = (lum >= p5) & (lum <= p60)
    sel = idx[band]
    if sel.size < AUTO_WB_MIN_SAMPLE:
        return NEUTRAL_WB

    centre_r = float(np.median(R.ravel()[sel]))
    centre_g = float(np.median(G.ravel()[sel]))
    centre_b = float(np.median(B.ravel()[sel]))
    if min(centre_r, centre_g, centre_b) <= AUTO_WB_CENTRE_FLOOR:
        return NEUTRAL_WB

    gain_r = float(np.clip(centre_g / centre_r, AUTO_WB_GAIN_MIN, AUTO_WB_GAIN_MAX))
    gain_g = 1.0
    gain_b = float(np.clip(centre_g / centre_b, AUTO_WB_GAIN_MIN, AUTO_WB_GAIN_MAX))
    return (gain_r, gain_g, gain_b)
