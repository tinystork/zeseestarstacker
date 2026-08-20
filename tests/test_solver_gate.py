"""Targeted tests for the M2b solver gate helper and source cleanliness.

``resolve_solver_gate`` lives in ``seestar.core.solver_config`` and is a pure,
GUI-free helper consumed by the GUI reproject gate.  These tests load the module
directly by file path (mirroring ``tests/test_solver_config.py``) so they do not
pull in the heavy ``seestar`` package tree.

The second half asserts the M2b source-level cleanup: the banned ANSVR /
Astrometry.net / API-key identifiers no longer appear anywhere in ``seestar/``,
while the deliberate migration strings ``"ansvr"`` / ``"astrometry"`` are still
present in the two places that own legacy-preference handling.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "seestar_solver_gate_config", ROOT / "seestar" / "core" / "solver_config.py"
)
solver_config = importlib.util.module_from_spec(spec)
sys.modules["seestar_solver_gate_config"] = solver_config
spec.loader.exec_module(solver_config)

resolve_solver_gate = solver_config.resolve_solver_gate


# ---------------------------------------------------------------------------
# resolve_solver_gate
# ---------------------------------------------------------------------------

def test_gate_zesolver_available_selected_allowed():
    allowed, reason = resolve_solver_gate("zesolver", True, False)
    assert allowed is True
    assert reason is None


def test_gate_zesolver_absent_astap_configured_allowed():
    # ZeSolver unavailable -> ASTAP autonomous fallback is acceptable.
    allowed, reason = resolve_solver_gate("zesolver", False, True)
    assert allowed is True
    assert reason is None


def test_gate_zesolver_absent_astap_absent_blocked():
    allowed, reason = resolve_solver_gate("zesolver", False, False)
    assert allowed is False
    assert reason == "zesolver_unavailable_no_astap"


def test_gate_astap_configured_allowed():
    allowed, reason = resolve_solver_gate("astap", False, True)
    assert allowed is True
    assert reason is None


def test_gate_astap_not_configured_blocked():
    allowed, reason = resolve_solver_gate("astap", False, False)
    assert allowed is False
    assert reason == "astap_not_configured"


def test_gate_none_blocks_even_with_astap():
    # The gate is only invoked from solve-requiring flows; "none" means no
    # solver is wanted, so it must block with a clear "no solver" reason even
    # when ASTAP happens to be configured.
    allowed, reason = resolve_solver_gate("none", True, True)
    assert allowed is False
    assert reason == "no_solver_configured"


def test_gate_legacy_prefs_treated_as_zesolver():
    # Legacy "ansvr"/"astrometry" must behave exactly like "zesolver".
    assert resolve_solver_gate("ansvr", True, False) == (True, None)
    assert resolve_solver_gate("astrometry", False, True) == (True, None)
    allowed, reason = resolve_solver_gate("ansvr", False, False)
    assert allowed is False
    assert reason == "zesolver_unavailable_no_astap"


def test_gate_unrecognised_pref_blocks():
    allowed, reason = resolve_solver_gate("bogus", True, True)
    assert allowed is False
    assert reason == "no_solver_configured"


# ---------------------------------------------------------------------------
# Source-level cleanup assertions (M2b)
# ---------------------------------------------------------------------------

BANNED_TOKENS = ("local_ansvr_path", "ansvr_", "api_key", "astrometry_net_")


def _seestar_py_files():
    root = ROOT / "seestar"
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def test_banned_solver_tokens_absent_from_source():
    offenders = []
    for path in _seestar_py_files():
        text = path.read_text(encoding="utf-8")
        for token in BANNED_TOKENS:
            if token in text:
                offenders.append((str(path), token))
    assert not offenders, (
        "banned ANSVR/ANet/API-key identifiers still present: " + repr(offenders)
    )


def test_legacy_migration_strings_preserved():
    # The bare "ansvr"/"astrometry" migration strings must remain in the two
    # places that own legacy-preference handling (this is the documented
    # exception to the banned-token assertion above).
    astro_src = (ROOT / "seestar" / "alignment" / "astrometry_solver.py").read_text(
        encoding="utf-8"
    )
    assert '_LEGACY_PREFERENCES = ("ansvr", "astrometry")' in astro_src

    config_src = (ROOT / "seestar" / "core" / "solver_config.py").read_text(
        encoding="utf-8"
    )
    assert '_LEGACY_SOLVER_PREFERENCES = ("ansvr", "astrometry")' in config_src
