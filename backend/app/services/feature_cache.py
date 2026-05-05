"""In-memory matrix cache for the corpus feature set.

Cosine similarity is the hot path of `POST /search`. Stacking each extractor's
already-L2-normalised vectors into an `(N, D)` float32 matrix lets us run
`sims = M @ q` — N dot products as a single BLAS call — instead of fetching
JSON rows per query. Phase 2 budgets <1 s end-to-end on ~few-thousand corpus
images, which this comfortably hits.

The cache is a process-local singleton: rebuilt lazily on first access and
when ingestion flips a dirty flag. Concurrent reads are guarded by an asyncio
lock so a stampede after `mark_dirty()` rebuilds exactly once.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Final

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FeatureSet
from app.services import features as feat


@dataclass(frozen=True)
class FeatureMatrices:
    """Snapshot returned to callers — immutable for the lifetime of one query."""

    image_ids: tuple[int, ...]
    matrices: dict[str, np.ndarray]
    extractor_ver: str

    @property
    def size(self) -> int:
        return len(self.image_ids)


_FEATURE_NAMES: Final[tuple[str, ...]] = tuple(feat.EXPECTED_DIMS.keys())

BRUTE_FORCE_THRESHOLD: int = 0  # always use ANN (pgvector HNSW); in-memory cache disabled

_lock: asyncio.Lock = asyncio.Lock()
_cached: FeatureMatrices | None = None
_dirty: bool = True


def mark_dirty() -> None:
    """Flag the cache for rebuild on next access. Cheap; no IO."""
    global _dirty
    _dirty = True


async def _count_corpus(session: AsyncSession) -> int:
    result = await session.execute(select(func.count(FeatureSet.image_id)))
    return int(result.scalar_one())


def _empty_snapshot() -> FeatureMatrices:
    matrices = {
        name: np.zeros((0, feat.EXPECTED_DIMS[name]), dtype=np.float32)
        for name in _FEATURE_NAMES
    }
    return FeatureMatrices(
        image_ids=(),
        matrices=matrices,
        extractor_ver=feat.EXTRACTOR_VERSION,
    )


def _build_snapshot(rows: list[FeatureSet]) -> FeatureMatrices:
    """Stack vector lists from `feature_sets` rows into per-feature matrices."""
    if not rows:
        return _empty_snapshot()

    image_ids = tuple(row.image_id for row in rows)
    matrices: dict[str, np.ndarray] = {}
    for name in _FEATURE_NAMES:
        expected_dim = feat.EXPECTED_DIMS[name]
        col_name = f"vec_{name}"
        stacked = np.zeros((len(rows), expected_dim), dtype=np.float32)
        for i, row in enumerate(rows):
            vec = getattr(row, col_name)
            if vec is None:
                raise ValueError(
                    f"feature_set image_id={row.image_id} missing vector '{name}'"
                )
            arr = np.asarray(vec, dtype=np.float32)
            if arr.shape != (expected_dim,):
                raise ValueError(
                    f"feature_set image_id={row.image_id} vector '{name}' has "
                    f"shape {arr.shape}, expected ({expected_dim},)"
                )
            stacked[i] = arr
        matrices[name] = stacked
    return FeatureMatrices(
        image_ids=image_ids,
        matrices=matrices,
        extractor_ver=feat.EXTRACTOR_VERSION,
    )


async def _load(session: AsyncSession) -> FeatureMatrices:
    """Read every feature_set row, ordered by image_id for deterministic indexing."""
    result = await session.execute(select(FeatureSet).order_by(FeatureSet.image_id))
    rows = list(result.scalars().all())
    return _build_snapshot(rows)


async def get_matrices(session: AsyncSession) -> FeatureMatrices | None:
    """Return a process-cached snapshot, rebuilding it if dirty.

    Returns ``None`` when the corpus exceeds ``BRUTE_FORCE_THRESHOLD`` so that
    callers fall back to the pgvector ANN path instead of loading everything
    into RAM.

    Automatically invalidates the cache when the corpus size changes so that
    seeding or deletion in another process is picked up on the next request.

    Callers receive an immutable view; mutation of the returned arrays would
    poison subsequent searches and is not supported.
    """
    global _cached, _dirty
    corpus_size = await _count_corpus(session)
    if corpus_size > BRUTE_FORCE_THRESHOLD:
        return None
    if _cached is not None and not _dirty and _cached.size != corpus_size:
        mark_dirty()
    if _cached is not None and not _dirty:
        return _cached
    async with _lock:
        if _cached is not None and not _dirty and _cached.size != corpus_size:
            mark_dirty()
        if _cached is not None and not _dirty:
            return _cached
        _cached = await _load(session)
        _dirty = False
        return _cached


def reset() -> None:
    """Drop the in-memory cache. For tests and pipeline restarts only."""
    global _cached, _dirty
    _cached = None
    _dirty = True
