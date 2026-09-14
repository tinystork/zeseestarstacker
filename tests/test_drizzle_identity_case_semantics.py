"""ZSSS-DRIZZLE-REJECTED-SOURCE REWORK 2 — host-aware identity case semantics.

The real Windows witness: the persisted identity ``name`` preserves original
display casing (``Light_SH2-101_...``) while the live queue identity derives
its basename from an already ``normcase()``-normalized Windows path
(``light_sh2-101_...``).  A raw case-sensitive basename comparison refused a
perfectly identical observation (same normalized path, same size, same
mtime_ns).

Contract: basename identity follows HOST filesystem path semantics.

* Windows (case-insensitive filesystem): a case-only basename difference is
  the same name and MUST match.
* POSIX (case-sensitive): ``Foo.fit`` vs ``foo.fit`` MUST remain distinct.

The normalization is the standard-library ``os.path.normcase`` semantics
(``ntpath.normcase`` / ``posixpath.normcase`` injected explicitly for
cross-platform tests) — never a custom lowercase rule, and never applied to
the size / mtime_ns / legal-location checks, which stay strict.
"""

import ntpath
import posixpath

import pytest

import seestar.queuep.queue_manager as qmm
from seestar.queuep.queue_manager import SeestarQueuedStacker

try:
    from seestar.queuep.queue_manager import _source_names_equivalent
except ImportError:  # pre-fix tree: the helper does not exist yet
    _source_names_equivalent = None

requires_helper = pytest.mark.skipif(
    _source_names_equivalent is None,
    reason="host-aware name helper absent on the pre-fix tree",
)

MIXED = "Light_SH2-101_20.0s_IRCUT_20260815-214521.fit"
LOWER = "light_sh2-101_20.0s_ircut_20260815-214521.fit"
OTHER = "Light_SH2-101_20.0s_IRCUT_20260815-214522.fit"
BASE_DIR = "d:/tulip/s50/altaz&eq/ircut/analyzed"


# ---------------------------------------------------------------------------
# helper semantics
# ---------------------------------------------------------------------------

@requires_helper
def test_helper_windows_semantics_case_only_equivalent():
    assert _source_names_equivalent(MIXED, LOWER, normcase=ntpath.normcase)


@requires_helper
def test_helper_windows_semantics_different_name_still_distinct():
    assert not _source_names_equivalent(MIXED, OTHER, normcase=ntpath.normcase)


@requires_helper
def test_helper_posix_semantics_case_only_distinct():
    assert not _source_names_equivalent(MIXED, LOWER, normcase=posixpath.normcase)
    assert _source_names_equivalent(MIXED, MIXED, normcase=posixpath.normcase)


@requires_helper
def test_helper_default_is_host_normcase():
    # On this (POSIX) host the default preserves case distinction.
    assert _source_names_equivalent(MIXED, MIXED)
    assert not _source_names_equivalent(MIXED, LOWER)


@requires_helper
def test_helper_malformed_never_matches():
    assert not _source_names_equivalent(None, MIXED)
    assert not _source_names_equivalent(MIXED, 42)


# ---------------------------------------------------------------------------
# production comparator gates
# ---------------------------------------------------------------------------

def _ident(name, path=None, size=4152960, mtime_ns=1786823120000000000):
    if path is None:
        path = f"{BASE_DIR}/{LOWER}"
    return {"path": path, "name": name, "size": size, "mtime_ns": mtime_ns}


def _run_comparator(actual, expected, normcase):
    """Run the REAL production comparator ``_queue_item_matches_plan_identity``
    with the module-level name helper resolved through the injected host
    semantics (the exact Windows-simulation seam; the size/mtime/location
    checks inside the comparator stay untouched)."""
    orig = getattr(qmm, "_source_names_equivalent", None)
    qmm._source_names_equivalent = lambda a, b: _source_names_equivalent(
        a, b, normcase=normcase
    )
    try:
        qm = object.__new__(SeestarQueuedStacker)
        qm._stat_identity = lambda path: dict(actual)
        qm.stacked_subdir_name = "stacked"
        return qm._queue_item_matches_plan_identity(actual["path"], expected)
    finally:
        if orig is None:
            try:
                del qmm._source_names_equivalent
            except AttributeError:
                pass
        else:
            qmm._source_names_equivalent = orig


def test_windows_case_only_witness_matches():
    """The exact real Windows witness: case-only name difference, same
    normalized path/size/mtime -> MUST match under Windows semantics."""
    expected = _ident(name=MIXED)
    actual = _ident(name=LOWER)
    assert _run_comparator(actual, expected, ntpath.normcase) is True


def test_posix_case_only_witness_refused():
    """On POSIX the same case-only difference MUST stay distinct."""
    expected = _ident(name=MIXED)
    actual = _ident(name=LOWER)
    assert _run_comparator(actual, expected, posixpath.normcase) is False


@pytest.mark.parametrize("normcase", [ntpath.normcase, posixpath.normcase])
def test_strict_gates_refuse(normcase):
    """Security / fail-closed gates: every evidence or location violation
    must refuse under BOTH host semantics."""
    base_expected = _ident(name=MIXED)

    # 1. same basename but wrong size
    assert _run_comparator(
        _ident(name=LOWER, size=123), base_expected, normcase
    ) is False
    # 2. same basename but wrong mtime_ns
    assert _run_comparator(
        _ident(name=LOWER, mtime_ns=1), base_expected, normcase
    ) is False
    # 3. different basename beyond case semantics
    assert _run_comparator(
        _ident(name=OTHER), base_expected, normcase
    ) is False
    # 4. path outside the original directory
    assert _run_comparator(
        _ident(name=LOWER, path="d:/elsewhere/" + LOWER),
        base_expected,
        normcase,
    ) is False
    # 5. path outside the exact stacked counterpart
    assert _run_comparator(
        _ident(
            name=LOWER,
            path=f"{BASE_DIR}/other_subdir/{LOWER}",
        ),
        base_expected,
        normcase,
    ) is False
    # 6. collision-renamed destination
    assert _run_comparator(
        _ident(name="light_sh2_dup_1726000000.fit"),
        base_expected,
        normcase,
    ) is False
    # 8. tampered source: same name, same dir, both evidence fields wrong
    assert _run_comparator(
        _ident(name=LOWER, size=999, mtime_ns=999),
        base_expected,
        normcase,
    ) is False


def test_stacked_counterpart_is_legal_location():
    """The exact stacked counterpart (the only ``_move_to_stacked``
    destination) remains a legal location under Windows semantics."""
    expected = _ident(name=MIXED)
    stacked_actual = _ident(name=LOWER, path=f"{BASE_DIR}/stacked/{LOWER}")
    assert _run_comparator(stacked_actual, expected, ntpath.normcase) is True
