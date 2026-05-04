"""Cosine-similarity retrieval engine over the cached feature matrices.

Pipeline (matches PLAN.md §6):

    decode → preprocess → extract 6 features → cosine vs corpus → weighted fuse → top-K

Vectors are pre-L2-normalised by the extractors, so cosine reduces to a dot
product. Per-feature sub-scores are kept around so later re-ranks (PATCH
`/search/{id}/weights`) can avoid re-running extraction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from app.services import feature_cache
from app.services import features as feat
from app.services.feature_cache import FeatureMatrices
from app.services.pipeline_emitter import PipelineEmitter

# Aligned with PLAN.md §11 — `cm` is the internal key for "colour moments".
DEFAULT_WEIGHTS: Final[dict[str, float]] = {
    "hog": 0.25,
    "hsv": 0.20,
    "lbp": 0.15,
    "glcm": 0.15,
    "hu": 0.15,
    "cm": 0.10,
}

DEFAULT_TOP_K: Final[int] = 5


def normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    """Drop unknown keys, clamp negatives, renormalise to sum-to-1.

    Empty / all-zero inputs fall back to `DEFAULT_WEIGHTS` so the caller
    never gets a silent zero ranking.
    """
    cleaned: dict[str, float] = {}
    for name in feat.EXPECTED_DIMS:
        value = float(weights.get(name, 0.0))
        cleaned[name] = max(value, 0.0)
    total = sum(cleaned.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {name: value / total for name, value in cleaned.items()}


@dataclass(frozen=True)
class SearchResult:
    image_id: int
    score: float
    rank: int
    per_feature: dict[str, float]


@dataclass(frozen=True)
class TraceStage:
    """One row in the pipeline trace persisted to `search_runs.pipeline_trace`."""

    name: str
    elapsed_ms: int
    detail: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchOutcome:
    """End-to-end result of one search call."""

    results: list[SearchResult]
    weights: dict[str, float]
    trace: list[TraceStage]
    elapsed_ms: int
    query_dims: dict[str, int]
    corpus_size: int


def _cosine_per_feature(
    snapshot: FeatureMatrices, query_vectors: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return `{feature: (N,) cosine similarity vector}` against the corpus."""
    sims: dict[str, np.ndarray] = {}
    for name, matrix in snapshot.matrices.items():
        q = np.asarray(query_vectors[name], dtype=np.float32)
        if matrix.shape[0] == 0:
            sims[name] = np.zeros((0,), dtype=np.float32)
            continue
        sims[name] = (matrix @ q).astype(np.float32)
    return sims


def _fuse(
    per_feature_sims: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    """Weighted sum across features → fused score per corpus row."""
    fused: np.ndarray | None = None
    for name, sims in per_feature_sims.items():
        weighted = sims * float(weights.get(name, 0.0))
        fused = weighted if fused is None else fused + weighted
    if fused is None:
        return np.zeros((0,), dtype=np.float32)
    return fused.astype(np.float32)


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Stable top-K by descending score; ties broken by ascending index."""
    if scores.size == 0:
        return np.empty((0,), dtype=np.int64)
    k = min(k, scores.size)
    # `argsort` ascending on `-scores` gives descending; stable for tie-breaks.
    order = np.argsort(-scores, kind="stable")
    return order[:k]


def assemble_results(
    snapshot: FeatureMatrices,
    per_feature_sims: dict[str, np.ndarray],
    fused: np.ndarray,
    top_k: int,
) -> list[SearchResult]:
    """Pack the top-K rows into structured results with per-feature sub-scores."""
    indices = _top_k_indices(fused, top_k)
    results: list[SearchResult] = []
    for rank, idx in enumerate(indices.tolist(), start=1):
        per_feature = {
            name: float(sims[idx]) for name, sims in per_feature_sims.items()
        }
        results.append(
            SearchResult(
                image_id=int(snapshot.image_ids[idx]),
                score=float(fused[idx]),
                rank=rank,
                per_feature=per_feature,
            )
        )
    return results


def rerank(
    snapshot: FeatureMatrices,
    per_feature_sims: dict[str, np.ndarray],
    weights: dict[str, float],
    top_k: int = DEFAULT_TOP_K,
) -> list[SearchResult]:
    """Re-fuse cached sub-scores under a new weighting (Phase 4 hot path)."""
    cleaned = normalise_weights(weights)
    fused = _fuse(per_feature_sims, cleaned)
    return assemble_results(snapshot, per_feature_sims, fused, top_k)


def rerank_from_persisted(
    *,
    image_ids: list[int],
    per_feature_sims: dict[str, np.ndarray],
    weights: dict[str, float],
    top_k: int = DEFAULT_TOP_K,
) -> list[SearchResult]:
    """Re-fuse stored sub-scores without needing a live FeatureMatrices snapshot.

    PATCH `/search/{id}/weights` replays a previous run's per-feature scores —
    we only need the original `image_ids` ordering to map fused-row indices
    back to image IDs; the corpus matrices themselves are not required.
    """
    cleaned = normalise_weights(weights)
    fused = _fuse(per_feature_sims, cleaned)
    indices = _top_k_indices(fused, top_k)
    results: list[SearchResult] = []
    for rank, idx in enumerate(indices.tolist(), start=1):
        per_feature = {
            name: float(sims[idx]) for name, sims in per_feature_sims.items()
        }
        results.append(
            SearchResult(
                image_id=int(image_ids[idx]),
                score=float(fused[idx]),
                rank=rank,
                per_feature=per_feature,
            )
        )
    return results


def parse_persisted_sub_scores(
    pipeline_trace: list[dict[str, object]] | None,
) -> tuple[list[int], dict[str, np.ndarray]] | None:
    """Pull `(image_ids, per_feature_sims)` out of the trailing `_sub_scores` stage.

    Returns `None` when the trace is missing the expected stash — the caller
    should respond 409 rather than re-running the full pipeline silently.
    """
    if not pipeline_trace:
        return None
    last = pipeline_trace[-1]
    if not isinstance(last, dict) or last.get("name") != "_sub_scores":
        return None
    detail = last.get("detail") or {}
    if not isinstance(detail, dict):
        return None
    raw_ids = detail.get("image_ids")
    raw_scores = detail.get("scores")
    if not isinstance(raw_ids, list) or not isinstance(raw_scores, dict):
        return None
    image_ids = [int(i) for i in raw_ids]
    sims: dict[str, np.ndarray] = {}
    for name, values in raw_scores.items():
        if not isinstance(values, list):
            return None
        sims[str(name)] = np.asarray(values, dtype=np.float32)
    return image_ids, sims


async def run_search(
    session,
    *,
    preprocessed: np.ndarray,
    weights: dict[str, float] | None = None,
    top_k: int = DEFAULT_TOP_K,
    emitter: PipelineEmitter | None = None,
) -> tuple[SearchOutcome, dict[str, np.ndarray]]:
    """Run the full search pipeline against the cached corpus matrices.

    Returns `(outcome, per_feature_sims)` so the caller can persist the raw
    sub-scores alongside the search_runs row for cheap re-ranks later.

    When `emitter` is provided every stage (and every per-feature step inside
    the extract / cosine stages) is published as a `PipelineEvent` so a live
    WebSocket subscriber can render the timeline as the search runs.
    """
    if preprocessed.dtype != np.uint8 or preprocessed.ndim != 3:
        raise ValueError("run_search expects a uint8 BGR preprocessed image")

    cleaned_weights = normalise_weights(weights or DEFAULT_WEIGHTS)
    trace: list[TraceStage] = []
    started = time.perf_counter()

    # Stage: feature extraction (per-feature events so the timeline shows
    # which extractor is currently running — HOG dominates the budget).
    if emitter is not None:
        await emitter.emit("stage.start", {"name": "extract"})
    extract_started = time.perf_counter()
    query_vectors: dict[str, np.ndarray] = {}
    feature_stages: list[TraceStage] = []
    for name, module in feat.EXTRACTORS.items():
        if emitter is not None:
            await emitter.emit("feature.start", {"name": name, "stage": "extract"})
        feat_started = time.perf_counter()
        query_vectors[name] = module.extract(preprocessed)  # type: ignore[attr-defined]
        feat_elapsed_ms = int((time.perf_counter() - feat_started) * 1000)
        feature_stages.append(
            TraceStage(
                name=f"feature.{name}",
                elapsed_ms=feat_elapsed_ms,
                detail={"dim": int(query_vectors[name].size)},
            )
        )
        if emitter is not None:
            await emitter.emit(
                "feature.done",
                {
                    "name": name,
                    "stage": "extract",
                    "elapsed_ms": feat_elapsed_ms,
                    "dim": int(query_vectors[name].size),
                },
            )
    extract_elapsed_ms = int((time.perf_counter() - extract_started) * 1000)
    trace.append(
        TraceStage(
            name="extract",
            elapsed_ms=extract_elapsed_ms,
            detail={"dims": dict(feat.EXPECTED_DIMS)},
        )
    )
    trace.extend(feature_stages)
    if emitter is not None:
        await emitter.emit(
            "stage.done",
            {"name": "extract", "elapsed_ms": extract_elapsed_ms},
        )

    # Stage: corpus snapshot
    if emitter is not None:
        await emitter.emit("stage.start", {"name": "load_corpus"})
    cache_started = time.perf_counter()
    snapshot = await feature_cache.get_matrices(session)
    cache_elapsed_ms = int((time.perf_counter() - cache_started) * 1000)
    trace.append(
        TraceStage(
            name="load_corpus",
            elapsed_ms=cache_elapsed_ms,
            detail={"corpus_size": snapshot.size},
        )
    )
    if emitter is not None:
        await emitter.emit(
            "stage.done",
            {
                "name": "load_corpus",
                "elapsed_ms": cache_elapsed_ms,
                "corpus_size": snapshot.size,
            },
        )

    # Stage: per-feature cosine
    if emitter is not None:
        await emitter.emit("stage.start", {"name": "cosine"})
    sims_started = time.perf_counter()
    per_feature_sims: dict[str, np.ndarray] = {}
    for name, matrix in snapshot.matrices.items():
        if emitter is not None:
            await emitter.emit("feature.start", {"name": name, "stage": "cosine"})
        feat_started = time.perf_counter()
        q = np.asarray(query_vectors[name], dtype=np.float32)
        if matrix.shape[0] == 0:
            per_feature_sims[name] = np.zeros((0,), dtype=np.float32)
        else:
            per_feature_sims[name] = (matrix @ q).astype(np.float32)
        feat_elapsed_ms = int((time.perf_counter() - feat_started) * 1000)
        if emitter is not None:
            await emitter.emit(
                "feature.done",
                {
                    "name": name,
                    "stage": "cosine",
                    "elapsed_ms": feat_elapsed_ms,
                    "rows": int(per_feature_sims[name].size),
                },
            )
    sims_elapsed_ms = int((time.perf_counter() - sims_started) * 1000)
    trace.append(
        TraceStage(
            name="cosine",
            elapsed_ms=sims_elapsed_ms,
            detail={name: sims.size for name, sims in per_feature_sims.items()},
        )
    )
    if emitter is not None:
        await emitter.emit(
            "stage.done",
            {"name": "cosine", "elapsed_ms": sims_elapsed_ms},
        )

    # Stage: fuse + rank
    if emitter is not None:
        await emitter.emit("stage.start", {"name": "rank"})
    rank_started = time.perf_counter()
    fused = _fuse(per_feature_sims, cleaned_weights)
    results = assemble_results(snapshot, per_feature_sims, fused, top_k)
    rank_elapsed_ms = int((time.perf_counter() - rank_started) * 1000)
    if emitter is not None:
        for r in results:
            await emitter.emit(
                "rank.tick",
                {"rank": r.rank, "image_id": r.image_id, "score": r.score},
            )
    trace.append(
        TraceStage(
            name="rank",
            elapsed_ms=rank_elapsed_ms,
            detail={"top_k": len(results), "weights": cleaned_weights},
        )
    )
    if emitter is not None:
        await emitter.emit(
            "stage.done",
            {"name": "rank", "elapsed_ms": rank_elapsed_ms},
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    outcome = SearchOutcome(
        results=results,
        weights=cleaned_weights,
        trace=trace,
        elapsed_ms=elapsed_ms,
        query_dims={name: int(vec.size) for name, vec in query_vectors.items()},
        corpus_size=snapshot.size,
    )
    if emitter is not None:
        await emitter.emit(
            "rank.done",
            {
                "elapsed_ms": elapsed_ms,
                "corpus_size": snapshot.size,
                "top_k": len(results),
                "results": [
                    {"rank": r.rank, "image_id": r.image_id, "score": r.score}
                    for r in results
                ],
            },
        )
    return outcome, per_feature_sims


def trace_to_jsonable(trace: list[TraceStage]) -> list[dict[str, object]]:
    """Serialise the trace for the JSONB column on `search_runs`."""
    return [
        {"name": s.name, "elapsed_ms": s.elapsed_ms, "detail": s.detail}
        for s in trace
    ]


def results_to_jsonable(results: list[SearchResult]) -> list[dict[str, object]]:
    """Serialise top-K results for the JSONB column on `search_runs`."""
    return [
        {
            "image_id": r.image_id,
            "score": r.score,
            "rank": r.rank,
            "per_feature": r.per_feature,
        }
        for r in results
    ]


def per_feature_sims_to_jsonable(
    snapshot: FeatureMatrices, per_feature_sims: dict[str, np.ndarray]
) -> dict[str, object]:
    """Serialise corpus indexing + per-feature scores for cheap re-ranks.

    Stored under `search_runs.pipeline_trace[-1].sub_scores` (or however the
    caller chooses) — kept here so PATCH /weights can rebuild rankings without
    redoing extraction or cosine math.
    """
    return {
        "image_ids": list(snapshot.image_ids),
        "scores": {
            name: sims.astype(float).tolist()
            for name, sims in per_feature_sims.items()
        },
    }
