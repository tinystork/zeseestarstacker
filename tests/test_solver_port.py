"""Tests for the internal SolverPort boundary (``seestar.alignment.solver_port``).

The module is loaded directly by file path (mirroring the pattern used by
``tests/test_solver_config.py``) so these tests do not pull in the heavy
``seestar`` package tree, which requires optional dependencies (OpenCV, etc.)
that are absent from this test environment.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "seestar_solver_port", ROOT / "seestar" / "alignment" / "solver_port.py"
)
solver_port = importlib.util.module_from_spec(spec)
sys.modules["seestar_solver_port"] = solver_port
spec.loader.exec_module(solver_port)


def test_importing_solver_port_does_not_import_zesolver():
    # The boundary must never pull in the optional ZeSolver dependency.
    assert not any(
        k == "zesolver" or k.startswith("zesolver.") for k in sys.modules
    )


def test_solve_outcome_defaults():
    outcome = solver_port.SolverOutcome()
    assert outcome.status is solver_port.SolveStatus.UNAVAILABLE
    assert outcome.wcs is None
    assert outcome.header is None
    assert outcome.should_write_header_back is False
    assert outcome.failure_code is None
    assert outcome.message is None
    assert outcome.backend_used is None


def test_solve_status_values():
    assert solver_port.SolveStatus.SOLVED.value == "solved"
    assert solver_port.SolveStatus.FAILED.value == "failed"
    assert solver_port.SolveStatus.SKIPPED.value == "skipped"
    assert solver_port.SolveStatus.CANCELLED.value == "cancelled"
    assert solver_port.SolveStatus.UNAVAILABLE.value == "unavailable"


def test_discovery_state_values():
    assert solver_port.DiscoveryState.AVAILABLE.value == "available"
    assert solver_port.DiscoveryState.NOT_INSTALLED.value == "not_installed"
    assert solver_port.DiscoveryState.INCOMPATIBLE.value == "incompatible"
    assert solver_port.DiscoveryState.UNHEALTHY.value == "unhealthy"
    assert solver_port.DiscoveryState.NOT_OPERATIONAL.value == "not_operational"


def test_solver_discovery_defaults():
    disc = solver_port.SolverDiscovery(state=solver_port.DiscoveryState.AVAILABLE)
    assert disc.state is solver_port.DiscoveryState.AVAILABLE
    assert disc.api_version is None
    assert disc.product_version is None
    assert disc.message is None
    assert disc.capabilities == ()
    assert disc.configuration_needed is False
    assert disc.operational is None


def test_contract_constants():
    assert solver_port.REQUIRED_CAPABILITIES == ("wcs_write",)
    assert solver_port.SOLVE_BACKEND_CAPABILITIES == ("near_solve", "blind_solve")
    assert solver_port.OPTIONAL_CAPABILITIES == ("cancel", "gpu")
    assert solver_port.SUPPORTED_API_MAJOR == 1


def test_solver_adapter_protocol_accepts_conforming_adapter():
    class DummyAdapter:
        name = "dummy"

        def solve(self, **kwargs):
            return solver_port.SolverOutcome(
                status=solver_port.SolveStatus.SOLVED,
                backend_used=self.name,
            )

    adapter = DummyAdapter()
    assert adapter.name == "dummy"
    outcome = adapter.solve(
        image_fits_path="/tmp/x.fits", fits_header={}, settings={}
    )
    assert outcome.status is solver_port.SolveStatus.SOLVED
    assert outcome.backend_used == "dummy"
