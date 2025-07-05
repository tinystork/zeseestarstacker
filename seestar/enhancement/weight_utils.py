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
