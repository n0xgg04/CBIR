"""Learn scientifically grounded CBIR fusion weights with pairwise ranking.

The model follows the pairwise probabilistic ranking objective used by
RankNet, restricted to a transparent linear scoring function:

    score(q, d) = sum_f w_f * z_f(q, d)

where ``z_f`` is a training-set z-score of the cosine similarity for feature
``f``. The optimizer enforces ``w_f >= 0`` and ``sum(w) = 1``. Hyperparameter
selection and performance estimation use nested stratified cross-validation.

The script prints two weight vectors:

* standardized weights: coefficients of the calibrated z-scores;
* deployable raw weights: equivalent coefficients for the current raw-cosine
  fusion in ``search_engine.py``.

Examples:

    cd backend
    uv run python scripts/learn_search_weights.py --images-dir ../data/afhq
    uv run python scripts/learn_search_weights.py --db-url "$DATABASE_URL"

Method references:

* Burges et al. (2005), "Learning to Rank using Gradient Descent" (RankNet).
* Binder et al. (2011), "Insights from Classifying Visual Concepts with
  Multiple Kernel Learning" (learned combinations of image similarities).
* Varma and Simon (2006), "Bias in error estimation when using
  cross-validation for model selection" (nested cross-validation).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.config import get_settings  # noqa: E402
from app.models import FeatureSet, Image  # noqa: E402
from app.services import features as feat  # noqa: E402
from app.services.preprocess import preprocess_path  # noqa: E402
from app.services.search_engine import DEFAULT_WEIGHTS  # noqa: E402

FEATURE_NAMES: Final[tuple[str, ...]] = tuple(feat.EXPECTED_DIMS)
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)
DEFAULT_LAMBDAS: Final[tuple[float, ...]] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
DEFAULT_MIN_WEIGHTS: Final[tuple[float, ...]] = (0.01, 0.02, 0.03, 0.05)


@dataclass(frozen=True)
class FeatureDataset:
    labels: np.ndarray
    matrices: dict[str, np.ndarray]

    @property
    def size(self) -> int:
        return int(self.labels.size)


@dataclass(frozen=True)
class Calibration:
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class RankingMetrics:
    macro_p5: float
    macro_map10: float
    micro_p5: float
    micro_map10: float


@dataclass(frozen=True)
class SamplingConfig:
    positives_per_query: int
    random_negatives_per_query: int
    hard_negatives_per_query: int


def _validate_dataset(dataset: FeatureDataset, required_folds: int) -> None:
    if dataset.size < 4:
        raise ValueError("need at least 4 images")
    classes, counts = np.unique(dataset.labels, return_counts=True)
    if classes.size < 2:
        raise ValueError("need at least 2 animal_type classes")
    if int(counts.min()) < required_folds:
        raise ValueError(
            f"smallest class has {int(counts.min())} images; "
            f"need at least {required_folds} for stratified CV"
        )
    for name in FEATURE_NAMES:
        matrix = dataset.matrices.get(name)
        expected = (dataset.size, feat.EXPECTED_DIMS[name])
        if matrix is None or matrix.shape != expected:
            actual = None if matrix is None else matrix.shape
            raise ValueError(f"feature {name!r} has shape {actual}, expected {expected}")


def load_directory(
    root: Path,
    *,
    max_per_class: int | None = None,
) -> FeatureDataset:
    """Extract features from ``root/<animal_type>/*``."""
    labels: list[str] = []
    rows: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_NAMES}
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"no class directories found under {root}")

    total = 0
    for class_dir in class_dirs:
        paths = [
            path
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if max_per_class is not None:
            paths = paths[:max_per_class]
        print(f"[extract] {class_dir.name}: {len(paths)} images")
        for path in paths:
            vectors = feat.extract_all(preprocess_path(path))
            labels.append(class_dir.name)
            for name in FEATURE_NAMES:
                rows[name].append(vectors[name])
            total += 1
            if total % 100 == 0:
                print(f"[extract] completed {total} images")

    return FeatureDataset(
        labels=np.asarray(labels, dtype=str),
        matrices={
            name: np.ascontiguousarray(np.stack(values), dtype=np.float32)
            for name, values in rows.items()
        },
    )


async def load_database(db_url: str) -> FeatureDataset:
    """Load the current extractor version from corpus-only PostgreSQL rows."""
    engine = create_async_engine(db_url, pool_pre_ping=True)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_maker() as session:
            query = (
                select(FeatureSet, Image.animal_type)
                .join(Image, Image.id == FeatureSet.image_id)
                .where(
                    Image.role == "corpus",
                    FeatureSet.extractor_ver == feat.EXTRACTOR_VERSION,
                )
                .order_by(FeatureSet.image_id)
            )
            rows = (await session.execute(query)).all()
    finally:
        await engine.dispose()

    if not rows:
        raise ValueError(
            "no corpus feature rows found for extractor version "
            f"{feat.EXTRACTOR_VERSION!r}"
        )

    labels = np.asarray([str(row[1]) for row in rows], dtype=str)
    matrices: dict[str, np.ndarray] = {}
    for name in FEATURE_NAMES:
        matrices[name] = np.ascontiguousarray(
            np.stack(
                [
                    np.asarray(getattr(row[0], f"vec_{name}"), dtype=np.float32)
                    for row in rows
                ]
            ),
            dtype=np.float32,
        )
    return FeatureDataset(labels=labels, matrices=matrices)


def save_cache(path: Path, dataset: FeatureDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=dataset.labels,
        **{f"feature_{name}": matrix for name, matrix in dataset.matrices.items()},
    )


def load_cache(path: Path) -> FeatureDataset:
    with np.load(path, allow_pickle=False) as payload:
        return FeatureDataset(
            labels=np.asarray(payload["labels"], dtype=str),
            matrices={
                name: np.asarray(payload[f"feature_{name}"], dtype=np.float32)
                for name in FEATURE_NAMES
            },
        )


def stratified_folds(
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> list[np.ndarray]:
    """Return deterministic validation folds with each class represented."""
    rng = np.random.default_rng(seed)
    fold_parts: list[list[np.ndarray]] = [[] for _ in range(n_splits)]
    subset_labels = labels[indices]
    for label in np.unique(subset_labels):
        class_indices = indices[subset_labels == label].copy()
        rng.shuffle(class_indices)
        chunks = np.array_split(class_indices, n_splits)
        for fold, chunk in zip(fold_parts, chunks, strict=True):
            fold.append(chunk)
    return [
        np.sort(np.concatenate(parts)).astype(np.int64)
        for parts in fold_parts
    ]


def fit_calibration(
    dataset: FeatureDataset,
    train_indices: np.ndarray,
    *,
    max_pairs: int,
    seed: int,
) -> Calibration:
    """Estimate per-feature cosine mean/std from training pairs only."""
    n = int(train_indices.size)
    if n < 2:
        raise ValueError("calibration needs at least 2 training images")
    pair_count = min(max_pairs, n * (n - 1))
    rng = np.random.default_rng(seed)
    left = rng.choice(train_indices, size=pair_count, replace=True)
    right = rng.choice(train_indices, size=pair_count, replace=True)
    equal = left == right
    while np.any(equal):
        right[equal] = rng.choice(train_indices, size=int(equal.sum()), replace=True)
        equal = left == right

    means = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    stds = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for column, name in enumerate(FEATURE_NAMES):
        matrix = dataset.matrices[name]
        similarities = np.einsum(
            "ij,ij->i", matrix[left], matrix[right], optimize=True
        )
        means[column] = float(similarities.mean())
        stds[column] = max(float(similarities.std()), 1e-6)
    return Calibration(mean=means, std=stds)


def _similarity_vector(
    dataset: FeatureDataset,
    query_index: int,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    values = np.empty((candidate_indices.size, len(FEATURE_NAMES)), dtype=np.float64)
    for column, name in enumerate(FEATURE_NAMES):
        matrix = dataset.matrices[name]
        values[:, column] = matrix[candidate_indices] @ matrix[query_index]
    return values


def build_pairwise_examples(
    dataset: FeatureDataset,
    train_indices: np.ndarray,
    calibration: Calibration,
    *,
    sampling: SamplingConfig,
    seed: int,
) -> np.ndarray:
    """Build ``z(q, positive) - z(q, negative)`` RankNet examples."""
    rng = np.random.default_rng(seed)
    labels = dataset.labels
    examples: list[np.ndarray] = []

    for query_index in train_indices:
        positives = train_indices[
            (labels[train_indices] == labels[query_index])
            & (train_indices != query_index)
        ]
        negatives = train_indices[labels[train_indices] != labels[query_index]]
        if positives.size == 0 or negatives.size == 0:
            continue

        positive_count = min(sampling.positives_per_query, positives.size)
        selected_positives = rng.choice(positives, size=positive_count, replace=False)

        random_count = min(sampling.random_negatives_per_query, negatives.size)
        random_negatives = rng.choice(negatives, size=random_count, replace=False)

        negative_raw = _similarity_vector(dataset, int(query_index), negatives)
        negative_z = (negative_raw - calibration.mean) / calibration.std
        hard_count = min(sampling.hard_negatives_per_query, negatives.size)
        hard_order = np.argsort(-negative_z.mean(axis=1), kind="stable")[:hard_count]
        selected_negatives = np.unique(
            np.concatenate((random_negatives, negatives[hard_order]))
        )

        positive_z = (
            _similarity_vector(dataset, int(query_index), selected_positives)
            - calibration.mean
        ) / calibration.std
        selected_negative_z = (
            _similarity_vector(dataset, int(query_index), selected_negatives)
            - calibration.mean
        ) / calibration.std

        differences = (
            positive_z[:, np.newaxis, :]
            - selected_negative_z[np.newaxis, :, :]
        ).reshape(-1, len(FEATURE_NAMES))
        examples.append(differences)

    if not examples:
        raise ValueError("could not construct relevant/irrelevant training pairs")
    return np.ascontiguousarray(np.vstack(examples), dtype=np.float64)


def fit_ranknet_weights(
    examples: np.ndarray,
    regularization: float,
    *,
    min_weight: float = 0.0,
) -> np.ndarray:
    """Fit sum-to-one linear RankNet coefficients on a bounded simplex."""
    feature_count = examples.shape[1]
    if min_weight < 0 or min_weight * feature_count >= 1.0:
        raise ValueError(
            f"min_weight must satisfy 0 <= min_weight < {1.0 / feature_count:.6f}"
        )
    initial = np.full(feature_count, 1.0 / feature_count, dtype=np.float64)

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        margins = examples @ weights
        loss = float(np.logaddexp(0.0, -margins).mean())
        penalty = float(regularization * np.dot(weights, weights))
        gradient = -(examples.T @ expit(-margins)) / examples.shape[0]
        gradient += 2.0 * regularization * weights
        return loss + penalty, gradient

    result: OptimizeResult = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=True,
        bounds=[(min_weight, 1.0)] * feature_count,
        constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"weight optimization failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, None)
    return weights / weights.sum()


def standardized_to_raw(
    standardized_weights: np.ndarray,
    calibration: Calibration,
) -> np.ndarray:
    """Convert z-score coefficients to ranking-equivalent raw-cosine weights."""
    raw = standardized_weights / calibration.std
    return raw / raw.sum()


def raw_to_standardized(
    raw_weights: np.ndarray,
    calibration: Calibration,
) -> np.ndarray:
    standardized = raw_weights * calibration.std
    return standardized / standardized.sum()


def evaluate_ranking(
    dataset: FeatureDataset,
    *,
    query_indices: np.ndarray,
    gallery_indices: np.ndarray,
    standardized_weights: np.ndarray,
    calibration: Calibration,
) -> RankingMetrics:
    """Evaluate held-out queries against a disjoint training gallery."""
    scores = np.zeros((query_indices.size, gallery_indices.size), dtype=np.float64)
    for column, name in enumerate(FEATURE_NAMES):
        raw = dataset.matrices[name][query_indices] @ dataset.matrices[name][gallery_indices].T
        scores += standardized_weights[column] * (
            (raw - calibration.mean[column]) / calibration.std[column]
        )

    p5_values: list[float] = []
    ap10_values: list[float] = []
    by_class: dict[str, list[tuple[float, float]]] = {}
    gallery_labels = dataset.labels[gallery_indices]
    for row, query_index in enumerate(query_indices):
        query_label = str(dataset.labels[query_index])
        total_relevant = int(np.sum(gallery_labels == query_label))
        if total_relevant == 0:
            continue
        order = np.argsort(-scores[row], kind="stable")
        relevance = gallery_labels[order] == query_label
        p5 = float(np.sum(relevance[:5])) / float(min(5, relevance.size))
        hits = 0
        ap10_sum = 0.0
        for rank, relevant in enumerate(relevance[:10], start=1):
            if relevant:
                hits += 1
                ap10_sum += hits / rank
        ap10 = ap10_sum / min(10, total_relevant)
        p5_values.append(p5)
        ap10_values.append(ap10)
        by_class.setdefault(query_label, []).append((p5, ap10))

    if not p5_values:
        raise ValueError("no evaluable held-out queries")
    class_p5 = [np.mean([value[0] for value in rows]) for rows in by_class.values()]
    class_map = [np.mean([value[1] for value in rows]) for rows in by_class.values()]
    return RankingMetrics(
        macro_p5=float(np.mean(class_p5)),
        macro_map10=float(np.mean(class_map)),
        micro_p5=float(np.mean(p5_values)),
        micro_map10=float(np.mean(ap10_values)),
    )


def select_regularization(
    dataset: FeatureDataset,
    train_indices: np.ndarray,
    *,
    lambdas: tuple[float, ...],
    min_weights: tuple[float, ...],
    inner_folds: int,
    calibration_pairs: int,
    sampling: SamplingConfig,
    seed: int,
) -> tuple[tuple[float, float], dict[tuple[float, float], tuple[float, float]]]:
    """Select ``(lambda, min_weight)`` by inner CV and the one-SE rule."""
    candidates = tuple(
        (regularization, min_weight)
        for regularization in lambdas
        for min_weight in min_weights
    )
    fold_scores = {candidate: [] for candidate in candidates}
    folds = stratified_folds(
        dataset.labels,
        train_indices,
        n_splits=inner_folds,
        seed=seed,
    )
    for fold_number, validation_indices in enumerate(folds):
        inner_train = np.setdiff1d(train_indices, validation_indices, assume_unique=True)
        calibration = fit_calibration(
            dataset,
            inner_train,
            max_pairs=calibration_pairs,
            seed=seed + fold_number,
        )
        examples = build_pairwise_examples(
            dataset,
            inner_train,
            calibration,
            sampling=sampling,
            seed=seed + 100 + fold_number,
        )
        for regularization, min_weight in candidates:
            weights = fit_ranknet_weights(
                examples,
                regularization,
                min_weight=min_weight,
            )
            metrics = evaluate_ranking(
                dataset,
                query_indices=validation_indices,
                gallery_indices=inner_train,
                standardized_weights=weights,
                calibration=calibration,
            )
            fold_scores[(regularization, min_weight)].append(metrics.macro_map10)

    summary = {
        candidate: (
            float(np.mean(scores)),
            float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        )
        for candidate, scores in fold_scores.items()
    }
    best_candidate = max(candidates, key=lambda candidate: summary[candidate][0])
    best_mean, best_std = summary[best_candidate]
    threshold = best_mean - best_std / np.sqrt(inner_folds)
    eligible = [
        candidate
        for candidate in candidates
        if summary[candidate][0] >= threshold
    ]
    # Within one standard error, prefer the model that guarantees greater use
    # of every feature, then stronger L2 regularization for stability.
    selected = max(eligible, key=lambda candidate: (candidate[1], candidate[0]))
    return selected, summary


def _weights_dict(weights: np.ndarray) -> dict[str, float]:
    return {
        name: float(weights[column])
        for column, name in enumerate(FEATURE_NAMES)
    }


def _metrics_dict(metrics: RankingMetrics) -> dict[str, float]:
    return {
        "macro_p5": metrics.macro_p5,
        "macro_map10": metrics.macro_map10,
        "micro_p5": metrics.micro_p5,
        "micro_map10": metrics.micro_map10,
    }


def run_nested_cv(
    dataset: FeatureDataset,
    *,
    outer_folds: int,
    inner_folds: int,
    lambdas: tuple[float, ...],
    min_weights: tuple[float, ...],
    calibration_pairs: int,
    sampling: SamplingConfig,
    seed: int,
) -> dict[str, object]:
    _validate_dataset(dataset, max(outer_folds, inner_folds + 1))
    all_indices = np.arange(dataset.size, dtype=np.int64)
    outer_splits = stratified_folds(
        dataset.labels,
        all_indices,
        n_splits=outer_folds,
        seed=seed,
    )
    current_raw = np.asarray(
        [DEFAULT_WEIGHTS.get(name, 0.0) for name in FEATURE_NAMES],
        dtype=np.float64,
    )
    current_raw /= current_raw.sum()
    equal_raw = np.full(len(FEATURE_NAMES), 1.0 / len(FEATURE_NAMES))
    hog_raw = np.asarray([1.0 if name == "hog" else 0.0 for name in FEATURE_NAMES])

    fold_reports: list[dict[str, object]] = []
    standardized_weights: list[np.ndarray] = []
    raw_weights: list[np.ndarray] = []
    for fold_number, test_indices in enumerate(outer_splits, start=1):
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        (selected_lambda, selected_min_weight), inner_summary = select_regularization(
            dataset,
            train_indices,
            lambdas=lambdas,
            min_weights=min_weights,
            inner_folds=inner_folds,
            calibration_pairs=calibration_pairs,
            sampling=sampling,
            seed=seed + fold_number * 1000,
        )
        calibration = fit_calibration(
            dataset,
            train_indices,
            max_pairs=calibration_pairs,
            seed=seed + fold_number,
        )
        examples = build_pairwise_examples(
            dataset,
            train_indices,
            calibration,
            sampling=sampling,
            seed=seed + 100 + fold_number,
        )
        learned = fit_ranknet_weights(
            examples,
            selected_lambda,
            min_weight=selected_min_weight,
        )
        learned_raw = standardized_to_raw(learned, calibration)
        standardized_weights.append(learned)
        raw_weights.append(learned_raw)

        methods = {
            "learned": learned,
            "equal": raw_to_standardized(equal_raw, calibration),
            "current": raw_to_standardized(current_raw, calibration),
            "hog_only": raw_to_standardized(hog_raw, calibration),
        }
        metrics = {
            name: _metrics_dict(
                evaluate_ranking(
                    dataset,
                    query_indices=test_indices,
                    gallery_indices=train_indices,
                    standardized_weights=weights,
                    calibration=calibration,
                )
            )
            for name, weights in methods.items()
        }
        print(
            f"[outer {fold_number}/{outer_folds}] lambda={selected_lambda:g} "
            f"min_weight={selected_min_weight:g} "
            f"learned MAP@10={metrics['learned']['macro_map10']:.4f} "
            f"P@5={metrics['learned']['macro_p5']:.4f}"
        )
        fold_reports.append(
            {
                "fold": fold_number,
                "selected_lambda": selected_lambda,
                "selected_min_weight": selected_min_weight,
                "inner_cv": {
                    f"lambda={regularization:g},min_weight={min_weight:g}": {
                        "mean_map10": mean,
                        "std_map10": std,
                    }
                    for (regularization, min_weight), (mean, std) in inner_summary.items()
                },
                "standardized_weights": _weights_dict(learned),
                "raw_weights": _weights_dict(learned_raw),
                "metrics": metrics,
            }
        )

    # Select lambda once more on all development data, then fit the deployable model.
    (final_lambda, final_min_weight), final_inner_summary = select_regularization(
        dataset,
        all_indices,
        lambdas=lambdas,
        min_weights=min_weights,
        inner_folds=inner_folds,
        calibration_pairs=calibration_pairs,
        sampling=sampling,
        seed=seed + 9999,
    )
    final_calibration = fit_calibration(
        dataset,
        all_indices,
        max_pairs=calibration_pairs,
        seed=seed + 10000,
    )
    final_examples = build_pairwise_examples(
        dataset,
        all_indices,
        final_calibration,
        sampling=sampling,
        seed=seed + 10001,
    )
    final_standardized = fit_ranknet_weights(
        final_examples,
        final_lambda,
        min_weight=final_min_weight,
    )
    final_raw = standardized_to_raw(final_standardized, final_calibration)

    method_names = ("learned", "equal", "current", "hog_only")
    aggregate_metrics = {
        method: {
            metric: float(
                np.mean(
                    [
                        report["metrics"][method][metric]  # type: ignore[index]
                        for report in fold_reports
                    ]
                )
            )
            for metric in ("macro_p5", "macro_map10", "micro_p5", "micro_map10")
        }
        for method in method_names
    }
    standardized_stack = np.stack(standardized_weights)
    raw_stack = np.stack(raw_weights)
    return {
        "method": "linear_pairwise_ranknet_nested_cv",
        "dataset_size": dataset.size,
        "class_counts": {
            str(label): int(np.sum(dataset.labels == label))
            for label in np.unique(dataset.labels)
        },
        "feature_order": list(FEATURE_NAMES),
        "outer_folds": fold_reports,
        "aggregate_metrics": aggregate_metrics,
        "weight_stability": {
            "standardized_mean": _weights_dict(standardized_stack.mean(axis=0)),
            "standardized_std": _weights_dict(standardized_stack.std(axis=0, ddof=1)),
            "raw_mean": _weights_dict(raw_stack.mean(axis=0)),
            "raw_std": _weights_dict(raw_stack.std(axis=0, ddof=1)),
        },
        "final_model": {
            "selected_lambda": final_lambda,
            "selected_min_weight": final_min_weight,
            "inner_cv": {
                f"lambda={regularization:g},min_weight={min_weight:g}": {
                    "mean_map10": mean,
                    "std_map10": std,
                }
                for (regularization, min_weight), (mean, std)
                in final_inner_summary.items()
            },
            "calibration_mean": _weights_dict(final_calibration.mean),
            "calibration_std": _weights_dict(final_calibration.std),
            "standardized_weights": _weights_dict(final_standardized),
            "deployable_raw_cosine_weights": _weights_dict(final_raw),
        },
    }


def print_report(report: dict[str, object]) -> None:
    print("\n=== Nested-CV metrics (mean over outer folds) ===")
    aggregate = report["aggregate_metrics"]
    for method in ("learned", "current", "equal", "hog_only"):
        metrics = aggregate[method]  # type: ignore[index]
        print(
            f"{method:<10} "
            f"Macro-P@5={metrics['macro_p5']:.4f} "
            f"Macro-MAP@10={metrics['macro_map10']:.4f}"
        )

    stability = report["weight_stability"]
    raw_mean = stability["raw_mean"]  # type: ignore[index]
    raw_std = stability["raw_std"]  # type: ignore[index]
    print("\n=== Raw-cosine weight stability across outer folds ===")
    for name in FEATURE_NAMES:
        print(f"{name:<5} mean={raw_mean[name]:.6f}  std={raw_std[name]:.6f}")

    final = report["final_model"]
    print("\n=== Final deployable raw-cosine weights ===")
    print(f"selected lambda: {final['selected_lambda']}")
    print(f"selected standardized minimum weight: {final['selected_min_weight']}")
    weights = final["deployable_raw_cosine_weights"]
    print("DEFAULT_WEIGHTS = {")
    for name in FEATURE_NAMES:
        print(f'    "{name}": {weights[name]:.6f},')
    print("}")


def _parse_lambdas(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("lambdas must be comma-separated non-negative numbers")
    return tuple(sorted(set(values)))


def _parse_min_weights(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    maximum = 1.0 / len(FEATURE_NAMES)
    if not values or any(value <= 0 or value >= maximum for value in values):
        raise argparse.ArgumentTypeError(
            f"min weights must be comma-separated values in (0, {maximum:.6f})"
        )
    return tuple(sorted(set(values)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--images-dir",
        type=Path,
        help="Dataset root laid out as <root>/<animal_type>/*.",
    )
    source.add_argument(
        "--db-url",
        help="Async SQLAlchemy URL. Defaults to DATABASE_URL when no image directory is given.",
    )
    parser.add_argument("--cache-npz", type=Path, help="Load/save extracted directory features.")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument(
        "--lambdas",
        type=_parse_lambdas,
        default=DEFAULT_LAMBDAS,
        help="Comma-separated L2 regularization candidates.",
    )
    parser.add_argument(
        "--min-weights",
        type=_parse_min_weights,
        default=DEFAULT_MIN_WEIGHTS,
        help=(
            "Comma-separated lower bounds for every standardized feature weight. "
            "The value is selected by inner cross-validation."
        ),
    )
    parser.add_argument("--calibration-pairs", type=int, default=100_000)
    parser.add_argument("--positives-per-query", type=int, default=8)
    parser.add_argument("--random-negatives-per-query", type=int, default=12)
    parser.add_argument("--hard-negatives-per-query", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cache_npz is not None and args.cache_npz.exists():
        print(f"[data] loading feature cache {args.cache_npz}")
        dataset = load_cache(args.cache_npz)
    elif args.images_dir is not None:
        dataset = load_directory(
            args.images_dir.resolve(),
            max_per_class=args.max_per_class,
        )
        if args.cache_npz is not None:
            print(f"[data] saving feature cache {args.cache_npz}")
            save_cache(args.cache_npz, dataset)
    else:
        db_url = args.db_url or get_settings().database_url
        print("[data] loading corpus features from PostgreSQL")
        dataset = asyncio.run(load_database(db_url))

    sampling = SamplingConfig(
        positives_per_query=args.positives_per_query,
        random_negatives_per_query=args.random_negatives_per_query,
        hard_negatives_per_query=args.hard_negatives_per_query,
    )
    report = run_nested_cv(
        dataset,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        lambdas=args.lambdas,
        min_weights=args.min_weights,
        calibration_pairs=args.calibration_pairs,
        sampling=sampling,
        seed=args.seed,
    )
    print_report(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"\n[report] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
