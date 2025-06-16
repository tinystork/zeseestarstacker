import numpy as np
from seestar.enhancement.drizzle_integration import DrizzleIntegrator

def test_drizzle_integrator_renorm_max():
    rng = np.random.default_rng(0)
    first = rng.random((5, 5), dtype=np.float32)
    frames = [first / 77 for _ in range(77)]
    integrator = DrizzleIntegrator(renormalize="max")
    for f in frames:
        integrator.add(f, np.ones_like(f, dtype=np.float32))
    stack = integrator.finalize()
    assert np.isclose(stack.max(), first.max(), rtol=0.02)

