"""Integration test for the Phase 1 demo gate.

Hits `POST /api/v1/images` against a real (SQLite-backed) DB and a
temp on-disk storage root, then asserts that:

  * a row landed in `images`,
  * a row landed in `feature_sets` with all six vectors,
  * the JSON response carries the same data the DB now contains,
  * a re-upload of the same bytes is idempotent (no duplicate rows).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import db as app_db
from app.config import get_settings
from app.main import create_app
from app.models import Base, FeatureSet, Image
from app.services import features as feat


@pytest_asyncio.fixture
async def integration_env(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, AsyncSession]]:
    """Spin up a fresh SQLite DB + storage root + ASGI client for one test."""
    db_path = tmp_path / "cbir.db"
    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    # Steer pydantic-settings via the environment, then bust caches so the
    # next get_settings()/get_engine() see the overrides.
    previous_env = {
        "DATABASE_URL": os.environ.get("DATABASE_URL"),
        "STORAGE_ROOT": os.environ.get("STORAGE_ROOT"),
    }
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    os.environ["STORAGE_ROOT"] = str(storage_root)
    get_settings.cache_clear()
    app_db.reset_engine()

    engine = app_db.get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    app = create_app()
    transport = ASGITransport(app=app)
    sm = app_db.get_sessionmaker()
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            async with sm() as session:
                yield client, session
    finally:
        await engine.dispose()
        app_db.reset_engine()
        get_settings.cache_clear()
        for k, v in previous_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.asyncio
async def test_post_images_creates_image_and_feature_set(
    integration_env: tuple[AsyncClient, AsyncSession],
    fixture_image_bytes: bytes,
) -> None:
    client, session = integration_env

    files = {"file": ("kitty.jpg", fixture_image_bytes, "image/jpeg")}
    data = {"animal_type": "cat", "role": "corpus"}
    response = await client.post("/api/v1/images", files=files, data=data)

    assert response.status_code == 201, response.text
    body = response.json()

    image_payload = body["image"]
    feature_payload = body["features"]

    assert image_payload["animal_type"] == "cat"
    assert image_payload["role"] == "corpus"
    assert image_payload["filename"] == "kitty.jpg"
    assert image_payload["size_bytes"] == len(fixture_image_bytes)
    assert image_payload["storage_path"].startswith("originals/cat/")
    assert image_payload["storage_path"].endswith(".jpg")

    assert set(feature_payload["vectors"].keys()) == set(feat.EXPECTED_DIMS.keys())
    for name, dim in feat.EXPECTED_DIMS.items():
        assert len(feature_payload["vectors"][name]) == dim
        assert feature_payload["dims"][name] == dim
    assert feature_payload["extractor_ver"] == feat.EXTRACTOR_VERSION

    # DB-side assertions: exactly one image and one feature_set row.
    image_count = await session.scalar(select(func.count()).select_from(Image))
    feature_count = await session.scalar(select(func.count()).select_from(FeatureSet))
    assert image_count == 1
    assert feature_count == 1

    image_row = (await session.execute(select(Image))).scalar_one()
    feature_row = (await session.execute(select(FeatureSet))).scalar_one()
    assert image_row.id == image_payload["id"]
    assert feature_row.image_id == image_row.id
    assert feature_row.extractor_ver == feat.EXTRACTOR_VERSION
    assert set(feature_row.dims) == set(feat.EXPECTED_DIMS.keys())


@pytest.mark.asyncio
async def test_post_images_is_idempotent_for_identical_payload(
    integration_env: tuple[AsyncClient, AsyncSession],
    fixture_image_bytes: bytes,
) -> None:
    client, session = integration_env

    files = {"file": ("kitty.jpg", fixture_image_bytes, "image/jpeg")}
    data = {"animal_type": "cat", "role": "corpus"}

    first = await client.post("/api/v1/images", files=files, data=data)
    second = await client.post(
        "/api/v1/images",
        files={"file": ("dup.jpg", fixture_image_bytes, "image/jpeg")},
        data=data,
    )
    assert first.status_code == 201
    assert second.status_code in (200, 201)
    assert first.json()["image"]["id"] == second.json()["image"]["id"]

    image_count = await session.scalar(select(func.count()).select_from(Image))
    feature_count = await session.scalar(select(func.count()).select_from(FeatureSet))
    assert image_count == 1
    assert feature_count == 1


@pytest.mark.asyncio
async def test_post_images_rejects_non_image_payload(
    integration_env: tuple[AsyncClient, AsyncSession],
) -> None:
    client, _ = integration_env
    files = {"file": ("evil.bin", b"not-an-image", "application/octet-stream")}
    data = {"animal_type": "cat", "role": "corpus"}
    response = await client.post("/api/v1/images", files=files, data=data)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_post_images_rejects_empty_upload(
    integration_env: tuple[AsyncClient, AsyncSession],
) -> None:
    client, _ = integration_env
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    data = {"animal_type": "cat", "role": "corpus"}
    response = await client.post("/api/v1/images", files=files, data=data)
    assert response.status_code == 400
