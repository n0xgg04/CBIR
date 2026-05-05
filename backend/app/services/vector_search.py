"""Database-backed ANN search via pgvector HNSW.

Returns per-feature candidate sets that search_engine.py fuses using the same
weighted cosine logic as before — API contract is unchanged.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Columns to search and their internal feature key
FEATURE_COLS: dict[str, str] = {
    "hog": "vec_hog",
    "hsv": "vec_hsv",
    "lbp": "vec_lbp",
    "glcm": "vec_glcm",
    "hu": "vec_hu",
    "cm": "vec_cm",
}

DEFAULT_CANDIDATE_K: int = 50  # ANN over-fetch factor; tune upward if recall drops
DEFAULT_EF_SEARCH: int = 40  # HNSW beam width at query time


async def ann_candidates(
    session: AsyncSession,
    query_vectors: dict[str, np.ndarray],
    candidate_k: int = DEFAULT_CANDIDATE_K,
    ef_search: int = DEFAULT_EF_SEARCH,
) -> dict[str, dict[int, float]]:
    """Run one HNSW query per feature. Returns {feature_name: {image_id: cosine_similarity}}."""
    await session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))

    results: dict[str, dict[int, float]] = {}
    for name, col in FEATURE_COLS.items():
        q = query_vectors[name].tolist()
        rows = await session.execute(
            text(f"""
                SELECT image_id,
                       1.0 - ({col} <=> CAST(:q AS vector)) AS cosine_sim
                FROM feature_sets
                ORDER BY {col} <=> CAST(:q AS vector)
                LIMIT :k
            """),
            {"q": str(q), "k": candidate_k},
        )
        results[name] = {row.image_id: float(row.cosine_sim) for row in rows}
    return results


def fuse_candidates(
    per_feature: dict[str, dict[int, float]],
    weights: dict[str, float],
) -> list[tuple[int, float, dict[str, float]]]:
    """Weighted fusion over the union of ANN candidate sets.

    Returns list of (image_id, fused_score, per_feature_sims) sorted by
    fused_score descending. Identical contract to search_engine._fuse().
    Missing scores (image not in a feature's candidate set) default to 0.0.
    """
    all_ids: set[int] = set()
    for candidates in per_feature.values():
        all_ids.update(candidates.keys())

    fused: list[tuple[int, float, dict[str, float]]] = []
    for image_id in all_ids:
        per_f: dict[str, float] = {}
        score = 0.0
        for name, candidates in per_feature.items():
            sim = candidates.get(image_id, 0.0)
            per_f[name] = sim
            score += weights.get(name, 0.0) * sim
        fused.append((image_id, score, per_f))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused
