"""Unit tests for the LOO evaluator helpers."""

from __future__ import annotations

import numpy as np

from app.services import evaluator


def test_precision_at_k_full_relevance() -> None:
    rel = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    assert evaluator.precision_at_k(rel, 4) == 0.75
    assert evaluator.precision_at_k(rel, 3) == 1.0


def test_precision_at_k_clamps_to_array_size() -> None:
    rel = np.array([1.0, 0.0], dtype=np.float32)
    assert evaluator.precision_at_k(rel, 10) == 0.5


def test_precision_at_k_handles_empty_or_zero_k() -> None:
    rel = np.array([], dtype=np.float32)
    assert evaluator.precision_at_k(rel, 5) == 0.0
    assert evaluator.precision_at_k(np.array([1.0]), 0) == 0.0


def test_average_precision_at_k_textbook_example() -> None:
    """Top-5 = [1,0,1,0,1], total relevant = 3 → AP@5 = (1/1 + 2/3 + 3/5)/3."""
    rel = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    expected = (1.0 + 2.0 / 3.0 + 3.0 / 5.0) / 3.0
    assert evaluator.average_precision_at_k(rel, total_relevant=3, k=5) == \
        np_isclose(expected)


def test_average_precision_at_k_zero_when_no_hits() -> None:
    rel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    assert evaluator.average_precision_at_k(rel, total_relevant=2, k=3) == 0.0


def test_average_precision_at_k_zero_when_total_relevant_zero() -> None:
    rel = np.array([1.0, 1.0], dtype=np.float32)
    assert evaluator.average_precision_at_k(rel, total_relevant=0, k=2) == 0.0


def np_isclose(expected: float):
    """Tiny helper so direct `==` against floats reads cleanly above."""
    import pytest

    return pytest.approx(expected, abs=1e-6)


def _make_snapshot(
    labels: list[str], matrices: dict[str, np.ndarray]
) -> evaluator.LabeledSnapshot:
    return evaluator.LabeledSnapshot(
        image_ids=tuple(range(1, len(labels) + 1)),
        labels=tuple(labels),
        matrices=matrices,
    )


def test_evaluate_with_sims_perfect_corpus() -> None:
    """Two classes that cluster perfectly — P@K and MAP@K must hit 1.0."""
    cats = np.tile(np.array([1.0, 0.0], dtype=np.float32), (3, 1))
    dogs = np.tile(np.array([0.0, 1.0], dtype=np.float32), (3, 1))
    matrix = np.vstack([cats, dogs]).astype(np.float32)
    matrices = {"hsv": matrix, "hog": matrix}
    snapshot = _make_snapshot(["cat"] * 3 + ["dog"] * 3, matrices)
    full_sims = evaluator._full_similarity(snapshot)
    weights = {"hsv": 0.5, "hog": 0.5}

    overall, per_class = evaluator._evaluate_with_sims(
        snapshot, full_sims, weights, top_k=2
    )
    assert overall.n_queries == 6
    assert overall.precision_at_k == np_isclose(1.0)
    assert overall.map_at_k == np_isclose(1.0)
    assert set(per_class) == {"cat", "dog"}
    assert per_class["cat"].precision_at_k == np_isclose(1.0)
    assert per_class["dog"].map_at_k == np_isclose(1.0)


def test_evaluate_with_sims_skips_singleton_classes() -> None:
    """A label with only one image must be excluded from the metric averages."""
    cats = np.tile(np.array([1.0, 0.0], dtype=np.float32), (2, 1))
    dogs = np.tile(np.array([0.0, 1.0], dtype=np.float32), (2, 1))
    fox = np.array([[0.5, 0.5]], dtype=np.float32)
    matrix = np.vstack([cats, dogs, fox]).astype(np.float32)
    snapshot = _make_snapshot(
        ["cat", "cat", "dog", "dog", "fox"], {"hsv": matrix, "hog": matrix}
    )
    full_sims = evaluator._full_similarity(snapshot)
    overall, per_class = evaluator._evaluate_with_sims(
        snapshot, full_sims, {"hsv": 1.0, "hog": 0.0}, top_k=2
    )
    # The singleton "fox" must not appear in the per-class breakdown.
    assert set(per_class) == {"cat", "dog"}
    assert overall.n_queries == 4


def test_evaluate_with_sims_under_two_rows_returns_empty() -> None:
    matrix = np.array([[1.0, 0.0]], dtype=np.float32)
    snapshot = _make_snapshot(["cat"], {"hsv": matrix})
    overall, per_class = evaluator._evaluate_with_sims(
        snapshot, evaluator._full_similarity(snapshot), {"hsv": 1.0}, top_k=5
    )
    assert overall.n_queries == 0
    assert per_class == {}


def test_full_similarity_is_symmetric_and_self_one() -> None:
    matrix = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32
    )
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    snapshot = _make_snapshot(["a", "b", "c"], {"hsv": matrix})
    sims = evaluator._full_similarity(snapshot)
    s = sims["hsv"]
    np.testing.assert_allclose(np.diag(s), 1.0, atol=1e-6)
    np.testing.assert_allclose(s, s.T, atol=1e-6)


def test_fuse_full_combines_weighted_features() -> None:
    a = np.full((2, 2), 0.5, dtype=np.float32)
    b = np.full((2, 2), 0.1, dtype=np.float32)
    fused = evaluator._fuse_full({"hsv": a, "hog": b}, {"hsv": 0.4, "hog": 0.6})
    assert fused.shape == (2, 2)
    np.testing.assert_allclose(fused, 0.5 * 0.4 + 0.1 * 0.6, atol=1e-6)
