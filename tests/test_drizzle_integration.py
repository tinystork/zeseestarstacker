import numpy as np
from seestar.enhancement.drizzle_integration import DrizzleIntegrator


def test_current_preview():
    integ = DrizzleIntegrator()
    sci = np.ones((10, 10), np.float32)
    wht = np.ones_like(sci) * 2
    integ.add(sci, wht)
    preview = integ.current_preview()
    assert np.allclose(preview, 0.5)
