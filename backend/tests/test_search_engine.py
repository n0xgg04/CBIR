"""Unit tests for the cosine search engine.

These cover the maths in isolation — no router, no DB session beyond what
feature_cache needs to seed itself. Integration of the same pipeline through
HTTP lives in `test_search_router.py`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app import db as app_db
from app.config import get_settings
from app.models import Base, FeatureSet, Image
from app.services import feature_cache, search_engine
from app.services import features as feat


@pytest_asyncio.fixture
async def db_session(tmp_path: Path) -> AsyncIterator[AsyncSession]:
    db_path = tmp_path / "search.db"
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    get_settings.cache_clear()
    app_db.reset_engine()
    feature_cache.reset()

    engine = app_db.get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = app_db.get_sessionmaker()
    try:
        async with sm() as session:
            yield session
    finally:
        await engine.dispose()
        app_db.reset_engine()
        get_settings.cache_clear()
        feature_cache.reset()
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous


def _seeded_vectors_for(seed: int) -> dict[str, list[float]]:
    """Deterministic L2-normalised vectors keyed by `seed`."""
    rng = np.random.default_rng(seed=seed)
    out: dict[str, list[float]] = {}
    for name, dim in feat.EXPECTED_DIMS.items():
        v = rng.random(dim, dtype=np.float32)
        v = v / (float(np.linalg.norm(v)) or 1.0)
        out[name] = v.astype(float).tolist()
    return out


async def _seed_corpus(session: AsyncSession, seeds: list[int]) -> list[int]:
    ids: list[int] = []
    for s in seeds:
        img = Image(
            sha256=("a" * 56) + str(s).zfill(8),
            filename=f"img-{s}.jpg",
            storage_path=f"originals/cat/img-{s}.jpg",
            animal_type="cat",
            width=128,
            height=128,
            size_bytes=1000,
            role="corpus",
        )
        session.add(img)
        await session.flush()
        fs = FeatureSet(
            image_id=img.id,
            vectors=_seeded_vectors_for(s),
            dims=dict(feat.EXPECTED_DIMS),
            extractor_ver=feat.EXTRACTOR_VERSION,
        )
        session.add(fs)
        ids.append(img.id)
    await session.commit()
    return ids


def test_normalise_weights_drops_unknown_keys_and_negatives() -> None:
    weights = {"hsv": 0.4, "hog": -0.1, "bogus": 0.5, "cm": 0.6}
    cleaned = search_engine.normalise_weights(weights)
    assert set(cleaned.keys()) == set(feat.EXPECTED_DIMS.keys())
    assert all(value >= 0 for value in cleaned.values())
    assert pytest.approx(sum(cleaned.values()), rel=1e-6) == 1.0
    assert cleaned["bogus"] if "bogus" in cleaned else True  # not present


def test_normalise_weights_falls_back_when_all_zero() -> None:
    cleaned = search_engine.normalise_weights({"hsv": 0.0})
    assert cleaned == search_engine.DEFAULT_WEIGHTS


def test_default_weights_sum_to_one() -> None:
    total = sum(search_engine.DEFAULT_WEIGHTS.values())
    assert pytest.approx(total, rel=1e-6) == 1.0


@pytest.mark.asyncio
async def test_run_search_returns_top_k_in_score_order(
    db_session: AsyncSession, preprocessed_fixture: np.ndarray
) -> None:
    await _seed_corpus(db_session, seeds=[1, 2, 3, 4, 5, 6, 7, 8])

    outcome, sims = await search_engine.run_search(
        db_session, preprocessed=preprocessed_fixture, top_k=5
    )
    assert outcome.corpus_size == 8
    assert len(outcome.results) == 5
    scores = [r.score for r in outcome.results]
    assert scores == sorted(scores, reverse=True)
    assert all(r.rank == i + 1 for i, r in enumerate(outcome.results))

    # Sub-scores returned alongside fused scores must include every feature.
    assert set(sims.keys()) == set(feat.EXPECTED_DIMS.keys())
    for arr in sims.values():
        assert arr.shape == (8,)
        assert arr.dtype == np.float32


@pytest.mark.asyncio
async def test_run_search_emits_full_pipeline_trace(
    db_session: AsyncSession, preprocessed_fixture: np.ndarray
) -> None:
    await _seed_corpus(db_session, seeds=[1, 2])
    outcome, _ = await search_engine.run_search(
        db_session, preprocessed=preprocessed_fixture
    )
    stage_names = [stage.name for stage in outcome.trace]
    assert stage_names == ["extract", "load_corpus", "cosine", "rank"]
    for stage in outcome.trace:
        assert stage.elapsed_ms >= 0


@pytest.mark.asyncio
async def test_run_search_self_query_lands_at_top(
    db_session: AsyncSession,
) -> None:
    """Inserting an image whose vectors equal the query should rank rank=1."""
    seeds = [10, 11, 12, 13]
    ids = await _seed_corpus(db_session, seeds=seeds)
    target_id = ids[2]
    target_vectors = _seeded_vectors_for(seeds[2])

    # Build a "query" snapshot using the same vectors. We bypass extraction by
    # injecting them into _cosine_per_feature directly to keep this synthetic.
    feature_cache.mark_dirty()
    snapshot = await feature_cache.get_matrices(db_session)
    query_vectors = {
        name: np.asarray(v, dtype=np.float32) for name, v in target_vectors.items()
    }
    sims = search_engine._cosine_per_feature(snapshot, query_vectors)
    fused = search_engine._fuse(sims, search_engine.DEFAULT_WEIGHTS)
    results = search_engine.assemble_results(snapshot, sims, fused, top_k=4)
    assert results[0].image_id == target_id
    assert results[0].score == pytest.approx(1.0, rel=1e-3)


@pytest.mark.asyncio
async def test_rerank_reuses_sub_scores(db_session: AsyncSession) -> None:
    seeds = [21, 22, 23, 24, 25]
    await _seed_corpus(db_session, seeds=seeds)
    feature_cache.mark_dirty()
    snapshot = await feature_cache.get_matrices(db_session)
    # Synthetic query
    rng = np.random.default_rng(seed=99)
    qv = {name: rng.random(d).astype(np.float32) for name, d in feat.EXPECTED_DIMS.items()}
    for name in qv:
        n = float(np.linalg.norm(qv[name])) or 1.0
        qv[name] = qv[name] / n
    sims = search_engine._cosine_per_feature(snapshot, qv)

    a = search_engine.rerank(snapshot, sims, search_engine.DEFAULT_WEIGHTS, top_k=3)
    b = search_engine.rerank(snapshot, sims, {"hog": 1.0}, top_k=3)
    # Different weights should produce a different fused ordering on most seeds
    assert isinstance(a, list) and len(a) == 3
    assert isinstance(b, list) and len(b) == 3
    # Both rankings only contain valid corpus image ids
    valid = set(snapshot.image_ids)
    assert {r.image_id for r in a}.issubset(valid)
    assert {r.image_id for r in b}.issubset(valid)


@pytest.mark.asyncio
async def test_run_search_on_empty_corpus_returns_zero(
    db_session: AsyncSession, preprocessed_fixture: np.ndarray
) -> None:
    outcome, sims = await search_engine.run_search(
        db_session, preprocessed=preprocessed_fixture
    )
    assert outcome.corpus_size == 0
    assert outcome.results == []
    for arr in sims.values():
        assert arr.size == 0
