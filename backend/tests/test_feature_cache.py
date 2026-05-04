"""Unit tests for the in-memory feature matrix cache."""

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
from app.services import feature_cache
from app.services import features as feat


@pytest_asyncio.fixture
async def db_session(tmp_path: Path) -> AsyncIterator[AsyncSession]:
    """Per-test SQLite DB with a clean cache in front of it."""
    db_path = tmp_path / "cache.db"
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


def _fake_vectors() -> dict[str, list[float]]:
    """Synthetic L2-normalised vectors of the right shape for every feature."""
    out: dict[str, list[float]] = {}
    rng = np.random.default_rng(seed=42)
    for name, dim in feat.EXPECTED_DIMS.items():
        v = rng.random(dim, dtype=np.float32)
        v = v / (float(np.linalg.norm(v)) or 1.0)
        out[name] = v.astype(float).tolist()
    return out


async def _seed_image(session: AsyncSession, sha_suffix: str) -> int:
    """Insert one (Image, FeatureSet) pair, return the image id."""
    img = Image(
        sha256=("0" * 56) + sha_suffix.zfill(8),
        filename=f"f-{sha_suffix}.jpg",
        storage_path=f"originals/cat/f-{sha_suffix}.jpg",
        animal_type="cat",
        width=128,
        height=128,
        size_bytes=1234,
        role="corpus",
    )
    session.add(img)
    await session.flush()
    fs = FeatureSet(
        image_id=img.id,
        vectors=_fake_vectors(),
        dims=dict(feat.EXPECTED_DIMS),
        extractor_ver=feat.EXTRACTOR_VERSION,
    )
    session.add(fs)
    await session.commit()
    return img.id


@pytest.mark.asyncio
async def test_empty_corpus_returns_zero_row_matrices(db_session: AsyncSession) -> None:
    snap = await feature_cache.get_matrices(db_session)
    assert snap.size == 0
    assert snap.image_ids == ()
    for name, dim in feat.EXPECTED_DIMS.items():
        assert snap.matrices[name].shape == (0, dim)
        assert snap.matrices[name].dtype == np.float32


@pytest.mark.asyncio
async def test_loads_matrix_with_correct_shape_and_dtype(db_session: AsyncSession) -> None:
    ids = []
    for i in range(3):
        ids.append(await _seed_image(db_session, str(i + 1)))
    feature_cache.mark_dirty()  # invalidate after writes

    snap = await feature_cache.get_matrices(db_session)
    assert snap.size == 3
    assert list(snap.image_ids) == ids
    for name, dim in feat.EXPECTED_DIMS.items():
        m = snap.matrices[name]
        assert m.shape == (3, dim)
        assert m.dtype == np.float32


@pytest.mark.asyncio
async def test_cache_returns_same_snapshot_until_marked_dirty(
    db_session: AsyncSession,
) -> None:
    await _seed_image(db_session, "1")
    feature_cache.mark_dirty()
    first = await feature_cache.get_matrices(db_session)
    second = await feature_cache.get_matrices(db_session)
    assert first is second  # identity preserved while clean

    await _seed_image(db_session, "2")
    # without mark_dirty, cache still serves the stale snapshot
    third = await feature_cache.get_matrices(db_session)
    assert third is first

    feature_cache.mark_dirty()
    fourth = await feature_cache.get_matrices(db_session)
    assert fourth is not first
    assert fourth.size == 2


@pytest.mark.asyncio
async def test_image_ids_are_sorted_for_deterministic_indexing(
    db_session: AsyncSession,
) -> None:
    ids = []
    for i in range(5):
        ids.append(await _seed_image(db_session, str(i + 1)))
    feature_cache.mark_dirty()

    snap = await feature_cache.get_matrices(db_session)
    assert list(snap.image_ids) == sorted(ids)


@pytest.mark.asyncio
async def test_extractor_ver_propagated_to_snapshot(db_session: AsyncSession) -> None:
    await _seed_image(db_session, "1")
    feature_cache.mark_dirty()
    snap = await feature_cache.get_matrices(db_session)
    assert snap.extractor_ver == feat.EXTRACTOR_VERSION


@pytest.mark.asyncio
async def test_reset_drops_cached_snapshot(db_session: AsyncSession) -> None:
    await _seed_image(db_session, "1")
    feature_cache.mark_dirty()
    first = await feature_cache.get_matrices(db_session)
    feature_cache.reset()
    second = await feature_cache.get_matrices(db_session)
    assert first is not second
    assert second.size == first.size  # same data, fresh object
