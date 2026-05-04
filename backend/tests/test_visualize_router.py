"""Integration tests for `GET /api/v1/visualize/{image_id}/{feature}`."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app import db as app_db
from app.config import get_settings
from app.main import create_app
from app.models import Base
from app.services import feature_cache, plot


@pytest_asyncio.fixture
async def env(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, Path]]:
    db_path = tmp_path / "viz.db"
    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    previous = {
        "DATABASE_URL": os.environ.get("DATABASE_URL"),
        "STORAGE_ROOT": os.environ.get("STORAGE_ROOT"),
    }
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    os.environ["STORAGE_ROOT"] = str(storage_root)
    get_settings.cache_clear()
    app_db.reset_engine()
    feature_cache.reset()

    engine = app_db.get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    app = create_app()
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, storage_root
    finally:
        await engine.dispose()
        app_db.reset_engine()
        get_settings.cache_clear()
        feature_cache.reset()
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


async def _ingest(client: AsyncClient, payload: bytes) -> int:
    files = {"file": ("kitty.jpg", payload, "image/jpeg")}
    data = {"animal_type": "cat", "role": "corpus"}
    response = await client.post("/api/v1/images", files=files, data=data)
    assert response.status_code == 201
    return response.json()["image"]["id"]


@pytest.mark.asyncio
async def test_visualize_returns_png_for_each_feature(
    env: tuple[AsyncClient, Path], fixture_image_bytes: bytes
) -> None:
    client, storage_root = env
    image_id = await _ingest(client, fixture_image_bytes)

    for feature in plot.SUPPORTED_PLOTS:
        response = await client.get(f"/api/v1/visualize/{image_id}/{feature}")
        assert response.status_code == 200, (feature, response.text)
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")
        assert plot.plot_path(storage_root, image_id, feature).exists()


@pytest.mark.asyncio
async def test_visualize_uses_disk_cache_on_second_call(
    env: tuple[AsyncClient, Path], fixture_image_bytes: bytes
) -> None:
    client, storage_root = env
    image_id = await _ingest(client, fixture_image_bytes)

    first = await client.get(f"/api/v1/visualize/{image_id}/preprocess")
    assert first.status_code == 200

    cached_path = plot.plot_path(storage_root, image_id, "preprocess")
    cached_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-cache")

    second = await client.get(f"/api/v1/visualize/{image_id}/preprocess")
    assert second.status_code == 200
    assert second.content == b"\x89PNG\r\n\x1a\nfake-cache"


@pytest.mark.asyncio
async def test_visualize_404_for_missing_image(env: tuple[AsyncClient, Path]) -> None:
    client, _ = env
    response = await client.get("/api/v1/visualize/9999/hsv")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_visualize_400_for_unknown_feature(
    env: tuple[AsyncClient, Path], fixture_image_bytes: bytes
) -> None:
    client, _ = env
    image_id = await _ingest(client, fixture_image_bytes)
    response = await client.get(f"/api/v1/visualize/{image_id}/bogus")
    assert response.status_code == 400
