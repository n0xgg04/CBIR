"""Focused tests for the scientific fusion-weight learning script."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.learn_search_weights import (
    Calibration,
    FeatureDataset,
    fit_ranknet_weights,
    standardized_to_raw,
    stratified_folds,
)


def test_ranknet_prefers_consistently_discriminative_feature() -> None:
    # Feature 0 always ranks the positive above the negative. Feature 1 is
    # uninformative and changes sign, so its optimum contribution is near zero.
    examples = np.array(
        [
            [2.0, 1.0],
            [2.0, -1.0],
            [1.5, 0.5],
            [1.5, -0.5],
        ],
        dtype=np.float64,
    )
    weights = fit_ranknet_weights(examples, regularization=0.0)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0)
    assert weights[0] > 0.99


def test_ranknet_respects_positive_weight_floor() -> None:
    examples = np.array(
        [
            [2.0, -1.0, -1.0],
            [1.5, -0.5, -0.5],
            [1.0, -0.2, -0.2],
        ],
        dtype=np.float64,
    )
    weights = fit_ranknet_weights(
        examples,
        regularization=0.0,
        min_weight=0.05,
    )
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0.05 - 1e-8)
    assert weights[0] == pytest.approx(0.90, abs=1e-6)


def test_standardized_to_raw_preserves_candidate_order() -> None:
    calibration = Calibration(
        mean=np.array([0.3, 0.8], dtype=np.float64),
        std=np.array([0.2, 0.05], dtype=np.float64),
    )
    standardized = np.array([0.6, 0.4], dtype=np.float64)
    raw = standardized_to_raw(standardized, calibration)
    similarities = np.array(
        [[0.2, 0.80], [0.4, 0.79], [0.3, 0.85]],
        dtype=np.float64,
    )
    z_scores = (similarities - calibration.mean) / calibration.std
    standardized_order = np.argsort(-(z_scores @ standardized))
    raw_order = np.argsort(-(similarities @ raw))
    np.testing.assert_array_equal(standardized_order, raw_order)


def test_stratified_folds_cover_each_row_once() -> None:
    labels = np.asarray(["cat"] * 6 + ["dog"] * 6 + ["wild"] * 6)
    indices = np.arange(labels.size, dtype=np.int64)
    folds = stratified_folds(labels, indices, n_splits=3, seed=7)
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), indices)
    for fold in folds:
        classes, counts = np.unique(labels[fold], return_counts=True)
        assert set(classes) == {"cat", "dog", "wild"}
        np.testing.assert_array_equal(counts, [2, 2, 2])


def test_feature_dataset_size() -> None:
    dataset = FeatureDataset(
        labels=np.asarray(["cat", "dog"]),
        matrices={"dummy": np.zeros((2, 1), dtype=np.float32)},
    )
    assert dataset.size == 2
