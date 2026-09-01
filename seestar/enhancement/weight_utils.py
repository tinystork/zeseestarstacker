import numpy as np


def make_radial_weight_map(h, w, feather_fraction=0.92, floor=0.10):
    """Return a float32 (h, w) array with radial falloff."""
    Y, X = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.hypot(Y - cy, X - cx) / np.hypot(cy, cx)
    w_map = np.ones((h, w), dtype=np.float32)
    m = r >= feather_fraction
    w_map[m] = np.clip(
        1.0 - (r[m] - feather_fraction) / (1.0 - feather_fraction),
        floor,
        1.0,
    )
    return w_map


def _footprint_distance_fallback(mask):
    """Approximate Euclidean distance to nearest boundary without scipy.

    Fallback only: scipy.ndimage.distance_transform_edt is the primary path.
    Chamfer 3-4 (two-pass) so the taper still follows the real footprint
    (never a radial distance from the image centre).
    """
    h, w = mask.shape
    INF = 1 << 20
    d = np.where(mask, INF, 0).astype(np.int32)
    for i in range(1, h):
        up = d[i - 1]
        d[i] = np.minimum(d[i], up + 2)
        d[i, 1:] = np.minimum(d[i, 1:], up[:-1] + 3)
        d[i, :-1] = np.minimum(d[i, :-1], up[1:] + 3)
    for i in range(h - 2, -1, -1):
        dn = d[i + 1]
        d[i] = np.minimum(d[i], dn + 2)
        d[i, 1:] = np.minimum(d[i, 1:], dn[:-1] + 3)
        d[i, :-1] = np.minimum(d[i, :-1], dn[1:] + 3)
    for j in range(1, w):
        d[:, j] = np.minimum(d[:, j], d[:, j - 1] + 2)
    for j in range(w - 2, -1, -1):
        d[:, j] = np.minimum(d[:, j], d[:, j + 1] + 2)
    d = np.where(mask, d, 0).astype(np.float32) / 2.0
    return d


def make_footprint_taper(mask, feather_px=8.0, floor=0.0):
    """Return a float32 (h, w) taper that follows the real footprint boundary.

    Coverage-aware replacement for the historical radial falloff: 1.0 in the
    interior of the valid footprint, ramping smoothly toward floor over
    feather_px pixels near the actual transformed support boundary, and 0.0
    outside.  Translation/rotation invariant; never a radial distance from the
    image centre.
    """
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        raise ValueError("make_footprint_taper: mask must be boolean")
    if m.ndim != 2:
        raise ValueError("make_footprint_taper: mask must be 2-D")
    h, w = m.shape
    floor = float(floor)
    if not np.isfinite(floor) or not (0.0 <= floor < 1.0):
        raise ValueError("make_footprint_taper: floor must be finite in [0, 1)")
    feather_px = float(feather_px)
    if not np.isfinite(feather_px) or feather_px <= 0.0:
        raise ValueError("make_footprint_taper: feather_px must be finite > 0")

    if not np.any(m):
        return np.zeros((h, w), dtype=np.float32)

    try:
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(m)
    except Exception:
        dist = _footprint_distance_fallback(m)

    frac = np.clip(dist.astype(np.float32) / float(feather_px), 0.0, 1.0)
    taper = np.where(m, floor + (1.0 - floor) * frac, np.float32(0.0))
    return taper.astype(np.float32)
