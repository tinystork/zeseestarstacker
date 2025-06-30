import numpy as np
from seestar.enhancement.drizzle_integration import DrizzleIntegrator


def test_cumulative_preview():
    integ = DrizzleIntegrator()
    sci = np.full((4, 4), 2.0, np.float32)
    wht = np.ones_like(sci) * 4.0
    integ.add(sci, wht)
    preview = integ.cumulative_preview()
    assert np.allclose(preview, 0.5)
