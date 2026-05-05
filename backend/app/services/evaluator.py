"""Leave-one-out retrieval evaluation: Precision@K, MAP@K, ablation runner.

Phase 5 demo gate. For every corpus image we treat its own feature vectors as
the *query*, search the rest of the corpus, then score the top-K against the
query's `animal_type` ground-truth label:

    relevance[i] = 1 if retrieved.animal_type == query.animal_type else 0
    P@K  = sum(relevance[:K]) / K
    AP@K = sum(precision@i * relevance[i]) / min(K, total_relevant)

The ablation runner re-runs the same loop with each individual feature's
weight zeroed so the impact of removing one feature is measurable per-class.
Single-class corpora and 2-image corpora are not evaluable — the caller gets
a 409 response shaped error from the router layer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FeatureSet, Image
from app.services import features as feat
from app.services.search_engine import DEFAULT_WEIGHTS, normalise_weights

DEFAULT_TOP_K_PRECISION: Final[int] = 5
DEFAULT_TOP_K_MAP: Final[int] = 10


@dataclass(frozen=True)
class LabeledSnapshot:
    """All corpus rows joined with their `animal_type` label."""

    image_ids: tuple[int, ...]
    labels: tuple[str, ...]
    matrices: dict[str, np.ndarray]

    @property
    def size(self) -> int:
        return len(self.image_ids)


@dataclass(frozen=True)
class Metrics:
    precision_at_k: float
    map_at_k: float
    n_queries: int


@dataclass(frozen=True)
class EvaluationReport:
    method: str
    weights: dict[str, float]
    top_k: int
    overall: Metrics
    per_class: dict[str, Metrics] = field(default_factory=dict)


@dataclass(frozen=True)
class AblationReport:
    """Base run plus one variant per feature where its weight is set to 0."""

    base: EvaluationReport
    variants: dict[str, EvaluationReport]
    top_k: int


# ---------------------------------------------------------------------------
# Pure metric helpers
# ---------------------------------------------------------------------------


def precision_at_k(relevance: np.ndarray, k: int) -> float:
    """`P@K` for a single ranked list — fraction of the top-K that are relevant."""
    if k <= 0 or relevance.size == 0:
        return 0.0
    k = min(k, relevance.size)
    return float(np.sum(relevance[:k])) / float(k)


def average_precision_at_k(
    relevance: np.ndarray, total_relevant: int, k: int
) -> float:
    """`AP@K` — mean of precisions at the ranks of relevant hits inside the top-K.

    `total_relevant` is the number of relevant items in the *whole* corpus
    (excluding the query itself for leave-one-out). Dividing by it is the
    standard definition; `min(K, total_relevant)` clamps for safety.
    """
    if k <= 0 or relevance.size == 0 or total_relevant <= 0:
        return 0.0
    k = min(k, relevance.size)
    cumulative_hits = 0
    score = 0.0
    for i in range(k):
        if relevance[i] > 0.5:
            cumulative_hits += 1
            score += cumulative_hits / float(i + 1)
    denominator = min(k, total_relevant)
    if denominator <= 0:
        return 0.0
    return score / float(denominator)


# ---------------------------------------------------------------------------
# Snapshot loader
# ---------------------------------------------------------------------------


async def load_labeled_snapshot(session: AsyncSession) -> LabeledSnapshot:
    """Join `feature_sets` with `images.animal_type`, ordered by image_id."""
    rows_q = (
        select(FeatureSet, Image.animal_type)
        .join(Image, Image.id == FeatureSet.image_id)
        .order_by(FeatureSet.image_id)
    )
    result = await session.execute(rows_q)
    rows = result.all()
    if not rows:
        return LabeledSnapshot(
            image_ids=(),
            labels=(),
            matrices={
                name: np.zeros((0, feat.EXPECTED_DIMS[name]), dtype=np.float32)
                for name in feat.EXPECTED_DIMS
            },
        )

    image_ids = tuple(row[0].image_id for row in rows)
    labels = tuple(row[1] for row in rows)
    matrices: dict[str, np.ndarray] = {}
    for name in feat.EXPECTED_DIMS:
        expected_dim = feat.EXPECTED_DIMS[name]
        stacked = np.zeros((len(rows), expected_dim), dtype=np.float32)
        for i, row in enumerate(rows):
            vec = getattr(row[0], f"vec_{name}")
            if vec is None:
                raise ValueError(
                    f"feature_set image_id={row[0].image_id} missing vector '{name}'"
                )
            arr = np.asarray(vec, dtype=np.float32)
            if arr.shape != (expected_dim,):
                raise ValueError(
                    f"feature_set image_id={row[0].image_id} vector '{name}' has "
                    f"shape {arr.shape}, expected ({expected_dim},)"
                )
            stacked[i] = arr
        matrices[name] = stacked
    return LabeledSnapshot(image_ids=image_ids, labels=labels, matrices=matrices)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _full_similarity(
    snapshot: LabeledSnapshot,
) -> dict[str, np.ndarray]:
    """Per-feature `(N, N)` cosine similarity (already-normalised vectors)."""
    sims: dict[str, np.ndarray] = {}
    for name, matrix in snapshot.matrices.items():
        if matrix.shape[0] == 0:
            sims[name] = np.zeros((0, 0), dtype=np.float32)
        else:
            sims[name] = (matrix @ matrix.T).astype(np.float32)
    return sims


def _fuse_full(
    sims: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    fused: np.ndarray | None = None
    for name, m in sims.items():
        weighted = m * float(weights.get(name, 0.0))
        fused = weighted if fused is None else fused + weighted
    if fused is None or fused.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return fused.astype(np.float32)


def _evaluate_with_sims(
    snapshot: LabeledSnapshot,
    full_sims: dict[str, np.ndarray],
    weights: dict[str, float],
    top_k: int,
) -> tuple[Metrics, dict[str, Metrics]]:
    """Run leave-one-out scoring given pre-computed per-feature similarity tables."""
    n = snapshot.size
    if n < 2:
        return Metrics(0.0, 0.0, 0), {}

    fused = _fuse_full(full_sims, weights)
    np.fill_diagonal(fused, -np.inf)  # exclude self-match.
    # Count relevant items per class once.
    label_counts: dict[str, int] = {}
    for label in snapshot.labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    overall_p = 0.0
    overall_map = 0.0
    n_queries = 0
    per_class_acc: dict[str, list[tuple[float, float]]] = {}
    for i in range(n):
        query_label = snapshot.labels[i]
        total_relevant = label_counts[query_label] - 1  # exclude self
        if total_relevant <= 0:
            # Singleton class — skip from metrics, otherwise AP@K is undefined.
            continue
        scores = fused[i]
        order = np.argsort(-scores, kind="stable")[:top_k]
        retrieved_labels = np.array(
            [snapshot.labels[int(j)] for j in order], dtype=object
        )
        relevance = (retrieved_labels == query_label).astype(np.float32)
        p = precision_at_k(relevance, top_k)
        ap = average_precision_at_k(relevance, total_relevant, top_k)
        overall_p += p
        overall_map += ap
        n_queries += 1
        per_class_acc.setdefault(query_label, []).append((p, ap))

    if n_queries == 0:
        return Metrics(0.0, 0.0, 0), {}

    overall = Metrics(
        precision_at_k=overall_p / n_queries,
        map_at_k=overall_map / n_queries,
        n_queries=n_queries,
    )
    per_class: dict[str, Metrics] = {}
    for label, samples in per_class_acc.items():
        ps = [s[0] for s in samples]
        aps = [s[1] for s in samples]
        per_class[label] = Metrics(
            precision_at_k=sum(ps) / len(ps),
            map_at_k=sum(aps) / len(aps),
            n_queries=len(ps),
        )
    return overall, per_class


async def evaluate(
    session: AsyncSession,
    *,
    weights: dict[str, float] | None = None,
    top_k: int = DEFAULT_TOP_K_MAP,
    method: str = "default",
) -> EvaluationReport:
    """Run a single leave-one-out evaluation under the given weights."""
    snapshot = await load_labeled_snapshot(session)
    return _build_report(
        snapshot=snapshot,
        full_sims=await asyncio.to_thread(_full_similarity, snapshot),
        weights=normalise_weights(weights or DEFAULT_WEIGHTS),
        top_k=top_k,
        method=method,
    )


def _build_report(
    *,
    snapshot: LabeledSnapshot,
    full_sims: dict[str, np.ndarray],
    weights: dict[str, float],
    top_k: int,
    method: str,
) -> EvaluationReport:
    overall, per_class = _evaluate_with_sims(snapshot, full_sims, weights, top_k)
    return EvaluationReport(
        method=method,
        weights=dict(weights),
        top_k=top_k,
        overall=overall,
        per_class=per_class,
    )


async def run_ablation(
    session: AsyncSession,
    *,
    top_k: int = DEFAULT_TOP_K_MAP,
) -> AblationReport:
    """Base + one variant per feature with that feature's weight forced to 0."""
    snapshot = await load_labeled_snapshot(session)
    full_sims = await asyncio.to_thread(_full_similarity, snapshot)
    base_weights = normalise_weights(DEFAULT_WEIGHTS)
    base = _build_report(
        snapshot=snapshot,
        full_sims=full_sims,
        weights=base_weights,
        top_k=top_k,
        method="default",
    )
    variants: dict[str, EvaluationReport] = {}
    for feature_name in feat.EXPECTED_DIMS:
        ablated = {k: (0.0 if k == feature_name else v) for k, v in DEFAULT_WEIGHTS.items()}
        variants[f"minus_{feature_name}"] = _build_report(
            snapshot=snapshot,
            full_sims=full_sims,
            weights=normalise_weights(ablated),
            top_k=top_k,
            method=f"minus_{feature_name}",
        )
    return AblationReport(base=base, variants=variants, top_k=top_k)


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def metrics_to_jsonable(m: Metrics) -> dict[str, object]:
    return {
        "precision_at_k": m.precision_at_k,
        "map_at_k": m.map_at_k,
        "n_queries": m.n_queries,
    }


def report_to_jsonable(r: EvaluationReport) -> dict[str, object]:
    return {
        "method": r.method,
        "weights": r.weights,
        "top_k": r.top_k,
        "overall": metrics_to_jsonable(r.overall),
        "per_class": {k: metrics_to_jsonable(v) for k, v in r.per_class.items()},
    }


def ablation_to_jsonable(a: AblationReport) -> dict[str, object]:
    return {
        "top_k": a.top_k,
        "base": report_to_jsonable(a.base),
        "variants": {k: report_to_jsonable(v) for k, v in a.variants.items()},
    }
