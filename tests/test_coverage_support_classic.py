"""COV-01B-CLASSIC focused tests: plain-classic support-domain wiring.

Unit-level only (no full pipeline run): the ordered per-exposure support
payload, the atomic accumulate_support_pair seam, boolean-mask validation,
fail-before-mutation, and ARRAY-EXACT decomposition invariance (61 vs 3+17+41
vs singletons).
"""

import numpy as np
import pytest

from seestar.core.coverage_support import accumulate_support_pair
from seestar.queuep.queue_manager import SeestarQueuedStacker, _ResumeCheckpointError


def _fresh(tmp_path, name, shape=(4, 5)):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    qm = object.__new__(SeestarQueuedStacker)
    qm.output_folder = str(d)
    qm.coverage_sup_w1_memmap = None
    qm.coverage_sup_w2_memmap = None
    qm._support_state_available = True
    qm._support_unavailable_reason = None
    qm._create_support_memmaps(shape)
    return qm


def _payload(masks, scalars):
    return [(m, float(s)) for m, s in zip(masks, scalars)]


# ---------------------------------------------------------------------------
# accumulate_support_pair adversarial (dtype / existing accumulator state)
# ---------------------------------------------------------------------------
def test_accumulate_pair_rejects_wrong_accumulator_dtype():
    w1 = np.zeros((2, 2), dtype=np.int64)
    w2 = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(TypeError):
        accumulate_support_pair(w1, w2, np.ones((2, 2)), dtype=np.float64)


def test_accumulate_pair_rejects_existing_nonfinite_accumulator():
    w1 = np.zeros((2, 2), dtype=np.float64)
    w2 = np.full((2, 2), np.inf, dtype=np.float64)
    with pytest.raises(ValueError):
        accumulate_support_pair(w1, w2, np.ones((2, 2)), dtype=np.float64)


def test_accumulate_pair_rejects_existing_negative_accumulator():
    w1 = np.zeros((2, 2), dtype=np.float64)
    w2 = np.full((2, 2), -1.0, dtype=np.float64)
    with pytest.raises(ValueError):
        accumulate_support_pair(w1, w2, np.ones((2, 2)), dtype=np.float64)


def test_accumulate_pair_failure_is_byte_identical():
    w1 = np.full((2, 2), 1.0, dtype=np.float64)
    w2 = np.full((2, 2), 2.0, dtype=np.float64)
    w1_before = w1.copy()
    w2_before = w2.copy()
    with pytest.raises(ValueError):
        accumulate_support_pair(w1, w2, np.full((2, 2), np.nan), dtype=np.float64)
    assert np.array_equal(w1, w1_before)
    assert np.array_equal(w2, w2_before)


# ---------------------------------------------------------------------------
# _build_support_payload validation
# ---------------------------------------------------------------------------
def test_build_payload_cardinality_mismatch_fails_closed(tmp_path):
    qm = _fresh(tmp_path, "card")
    with pytest.raises(_ResumeCheckpointError):
        qm._build_support_payload(
            [np.ones((4, 5), bool), np.ones((4, 5), bool)], [1.0]
        )


def test_build_payload_invalid_scalar_fails_closed(tmp_path):
    qm = _fresh(tmp_path, "scalar")
    with pytest.raises(_ResumeCheckpointError):
        qm._build_support_payload([np.ones((4, 5), bool)], [np.nan])
    with pytest.raises(_ResumeCheckpointError):
        qm._build_support_payload([np.ones((4, 5), bool)], [-1.0])


def test_build_payload_missing_masks_fails_closed(tmp_path):
    qm = _fresh(tmp_path, "missing")
    with pytest.raises(_ResumeCheckpointError):
        qm._build_support_payload(None, [1.0])


# ---------------------------------------------------------------------------
# _apply_support_payload validation + order
# ---------------------------------------------------------------------------
def test_apply_payload_rejects_nonboolean_mask(tmp_path):
    qm = _fresh(tmp_path, "nonbool")
    with pytest.raises(_ResumeCheckpointError):
        qm._apply_support_payload([(np.ones((4, 5), np.float32), 1.0)])


def test_apply_payload_rejects_shape_mismatch_no_transpose(tmp_path):
    qm = _fresh(tmp_path, "shape")
    with pytest.raises(_ResumeCheckpointError):
        qm._apply_support_payload([(np.ones((3, 5), bool), 1.0)])
    # A transposed mask must also fail (no silent transpose re-binding).
    with pytest.raises(_ResumeCheckpointError):
        qm._apply_support_payload([(np.ones((5, 4), bool), 1.0)])


def test_known_masks_scalars_exact_sup(tmp_path):
    qm = _fresh(tmp_path, "known", shape=(2, 2))
    m0 = np.array([[True, True], [False, True]], dtype=bool)
    m1 = np.array([[True, False], [True, True]], dtype=bool)
    qm._apply_support_payload(_payload([m0, m1], [2.0, 3.0]))
    w1 = qm.coverage_sup_w1_memmap
    w2 = qm.coverage_sup_w2_memmap
    expected_w1 = m0.astype(np.float64) * 2.0 + m1.astype(np.float64) * 3.0
    expected_w2 = m0.astype(np.float64) * 4.0 + m1.astype(np.float64) * 9.0
    assert np.array_equal(w1, expected_w1)
    assert np.array_equal(w2, expected_w2)


def test_decomposition_exact_61_vs_partitions(tmp_path):
    rng = np.random.default_rng(0)
    shape = (4, 5)
    masks = [rng.random(shape) > 0.5 for _ in range(61)]
    scalars = [1.0 + 0.01 * i for i in range(61)]

    qm_all = _fresh(tmp_path, "all", shape)
    qm_all._apply_support_payload(_payload(masks, scalars))

    qm_part = _fresh(tmp_path, "part", shape)
    qm_part._apply_support_payload(_payload(masks[:3], scalars[:3]))
    qm_part._apply_support_payload(_payload(masks[3:20], scalars[3:20]))
    qm_part._apply_support_payload(_payload(masks[20:], scalars[20:]))

    qm_single = _fresh(tmp_path, "single", shape)
    for m, s in zip(masks, scalars):
        qm_single._apply_support_payload([(m, float(s))])

    w1_all = qm_all.coverage_sup_w1_memmap
    w2_all = qm_all.coverage_sup_w2_memmap
    assert np.array_equal(w1_all, qm_part.coverage_sup_w1_memmap)
    assert np.array_equal(w2_all, qm_part.coverage_sup_w2_memmap)
    assert np.array_equal(w1_all, qm_single.coverage_sup_w1_memmap)
    assert np.array_equal(w2_all, qm_single.coverage_sup_w2_memmap)

