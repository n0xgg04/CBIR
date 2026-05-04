"""Integration tests for `POST /api/v1/search` — the Phase 2 demo gate."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import db as app_db
from app.config import get_settings
from app.main import create_app
from app.models import Base, SearchRun
from app.services import feature_cache


@pytest_asyncio.fixture
async def search_env(
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncClient, AsyncSession]]:
    db_path = tmp_path / "search_router.db"
    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    previous_env = {
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
    sm = app_db.get_sessionmaker()
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            async with sm() as session:
                yield client, session
    finally:
        await engine.dispose()
        app_db.reset_engine()
        get_settings.cache_clear()
        feature_cache.reset()
        for k, v in previous_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


async def _ingest_corpus(client: AsyncClient, payloads: list[bytes]) -> list[int]:
    """Push N images through the existing /images endpoint to seed the corpus."""
    ids: list[int] = []
    for i, payload in enumerate(payloads):
        files = {"file": (f"img-{i}.jpg", payload, "image/jpeg")}
        data = {"animal_type": "cat", "role": "corpus"}
        response = await client.post("/api/v1/images", files=files, data=data)
        assert response.status_code == 201, response.text
        ids.append(response.json()["image"]["id"])
    return ids


@pytest.mark.asyncio
async def test_search_returns_top_k_and_persists_run(
    search_env: tuple[AsyncClient, AsyncSession],
    fixture_image_bytes: bytes,
) -> None:
    client, session = search_env

    # Seed with a couple distinct images. Re-encoding the same fixture with
    # different JPEG quality steps gives us deterministic but-not-identical bytes.
    import cv2
    import numpy as np

    arr = np.frombuffer(fixture_image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded is not None

    payloads: list[bytes] = []
    for q in (60, 70, 80, 92):
        ok, buf = cv2.imencode(".jpg", decoded, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        assert ok
        payloads.append(buf.tobytes())
    await _ingest_corpus(client, payloads)

    files = {"file": ("query.jpg", fixture_image_bytes, "image/jpeg")}
    response = await client.post(
        "/api/v1/search", files=files, data={"top_k": "3"}
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["corpus_size"] == len(payloads)
    assert len(body["results"]) == 3
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    for r in body["results"]:
        assert r["per_feature"]
        assert r["image"] is not None
        assert r["image"]["animal_type"] == "cat"

    # Pipeline trace surfaced for the UI inspector.
    stage_names = [stage["name"] for stage in body["pipeline_trace"]]
    assert stage_names == ["extract", "load_corpus", "cosine", "rank"]

    run = (await session.execute(select(SearchRun))).scalar_one()
    assert run.id == body["run_id"]
    assert run.elapsed_ms >= 0
    assert run.weights["hog"] == pytest.approx(0.25)
    # Sub-scores stashed for cheap re-ranks later.
    assert run.pipeline_trace[-1]["name"] == "_sub_scores"


@pytest.mark.asyncio
async def test_search_with_custom_weights_overrides_default(
    search_env: tuple[AsyncClient, AsyncSession],
    fixture_image_bytes: bytes,
) -> None:
    client, _ = search_env
    await _ingest_corpus(client, [fixture_image_bytes])
    weights = json.dumps({"hog": 1.0})
    response = await client.post(
        "/api/v1/search",
        files={"file": ("q.jpg", fixture_image_bytes, "image/jpeg")},
        data={"top_k": "1", "weights": weights},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["weights"]["hog"] == pytest.approx(1.0)
    assert body["weights"]["hsv"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_search_rejects_empty_corpus(
    search_env: tuple[AsyncClient, AsyncSession],
    fixture_image_bytes: bytes,
) -> None:
    client, _ = search_env
    response = await client.post(
        "/api/v1/search",
        files={"file": ("q.jpg", fixture_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_search_rejects_invalid_image(
    search_env: tuple[AsyncClient, AsyncSession],
    fixture_image_bytes: bytes,
) -> None:
    client, _ = search_env
    await _ingest_corpus(client, [fixture_image_bytes])
    response = await client.post(
        "/api/v1/search",
        files={"file": ("evil.bin", b"not-an-image", "application/octet-stream")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_search_rejects_invalid_weights_json(
    search_env: tuple[AsyncClient, AsyncSession],
    fixture_image_bytes: bytes,
) -> None:
    client, _ = search_env
    await _ingest_corpus(client, [fixture_image_bytes])
    response = await client.post(
        "/api/v1/search",
        files={"file": ("q.jpg", fixture_image_bytes, "image/jpeg")},
        data={"weights": "{not-json"},
    )
    assert response.status_code == 400
