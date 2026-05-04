"""Integration tests for `POST /api/v1/evaluate` (Phase 5 demo gate)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import cv2
import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app import db as app_db
from app.config import get_settings
from app.main import create_app
from app.models import Base, EvaluationRun
from app.services import feature_cache, pipeline_emitter


@pytest_asyncio.fixture
async def env(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, Path]]:
    db_path = tmp_path / "evaluate.db"
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
    pipeline_emitter.reset()

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
        pipeline_emitter.reset()
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _palette_image(seed: int) -> bytes:
    """Each seed produces a distinct synthetic image so corpora vary by class."""
    rng = np.random.default_rng(seed=seed)
    h, w = 128, 128
    base = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    base[..., 0] = ((xx + seed * 5) % 256).astype(np.uint8)
    base[..., 1] = ((yy + seed * 9) % 256).astype(np.uint8)
    base[..., 2] = ((xx + yy + seed * 17) % 256).astype(np.uint8)
    noise = rng.integers(0, 16, size=base.shape, dtype=np.int16)
    image = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return buf.tobytes()


async def _ingest(
    client: AsyncClient, payload: bytes, *, animal_type: str, name: str
) -> int:
    response = await client.post(
        "/api/v1/images",
        files={"file": (name, payload, "image/jpeg")},
        data={"animal_type": animal_type, "role": "corpus"},
    )
    assert response.status_code == 201, response.text
    return response.json()["image"]["id"]


async def _seed_two_class_corpus(client: AsyncClient) -> None:
    for i in range(3):
        await _ingest(client, _palette_image(seed=i), animal_type="cat", name=f"c{i}.jpg")
    for i in range(3, 6):
        await _ingest(client, _palette_image(seed=i), animal_type="dog", name=f"d{i}.jpg")


@pytest.mark.asyncio
async def test_evaluate_default_returns_metrics_and_persists_run(
    env: tuple[AsyncClient, Path],
) -> None:
    client, _ = env
    await _seed_two_class_corpus(client)

    response = await client.post(
        "/api/v1/evaluate", json={"method": "default", "top_k": 5}
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["method"] == "default"
    assert body["corpus_size"] == 6
    assert body["top_k"] == 5
    assert body["report"] is not None
    assert body["ablation"] is None

    overall = body["report"]["overall"]
    assert 0.0 <= overall["precision_at_k"] <= 1.0
    assert 0.0 <= overall["map_at_k"] <= 1.0
    assert overall["n_queries"] == 6
    # Both classes contribute to the breakdown.
    assert set(body["report"]["per_class"].keys()) == {"cat", "dog"}

    engine = app_db.get_engine()
    async with engine.begin() as conn:
        cursor = await conn.execute(
            EvaluationRun.__table__.select().where(EvaluationRun.id == body["run_id"])
        )
        row = cursor.mappings().one()
    assert row["method"] == "default"
    assert row["precision_at_5"] is not None
    assert row["map_at_10"] is not None
    assert row["finished_at"] is not None
    assert row["per_class"] is not None
    assert row["ablation"] is None


@pytest.mark.asyncio
async def test_evaluate_ablation_returns_one_variant_per_feature(
    env: tuple[AsyncClient, Path],
) -> None:
    client, _ = env
    await _seed_two_class_corpus(client)

    response = await client.post(
        "/api/v1/evaluate", json={"method": "ablation", "top_k": 3}
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["method"] == "ablation"
    assert body["report"] is None
    ablation = body["ablation"]
    assert ablation is not None
    assert ablation["top_k"] == 3
    assert ablation["base"]["method"] == "default"
    expected_variants = {f"minus_{n}" for n in ("hsv", "cm", "lbp", "glcm", "hog", "hu")}
    assert set(ablation["variants"]) == expected_variants
    for variant_name, variant in ablation["variants"].items():
        assert variant["method"] == variant_name
        # Feature corresponding to the variant has weight 0 after re-norm.
        zeroed = variant_name.removeprefix("minus_")
        assert variant["weights"][zeroed] == pytest.approx(0.0)

    engine = app_db.get_engine()
    async with engine.begin() as conn:
        cursor = await conn.execute(
            EvaluationRun.__table__.select().where(EvaluationRun.id == body["run_id"])
        )
        row = cursor.mappings().one()
    assert row["method"] == "ablation"
    assert row["ablation"] is not None
    assert row["per_class"] is None


@pytest.mark.asyncio
async def test_evaluate_409_when_corpus_too_small(
    env: tuple[AsyncClient, Path],
) -> None:
    client, _ = env
    response = await client.post("/api/v1/evaluate", json={"method": "default"})
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_evaluate_409_when_only_one_class(
    env: tuple[AsyncClient, Path],
) -> None:
    client, _ = env
    for i in range(3):
        await _ingest(client, _palette_image(seed=i), animal_type="cat", name=f"c{i}.jpg")

    response = await client.post("/api/v1/evaluate", json={"method": "default"})
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_evaluate_default_honours_custom_weights(
    env: tuple[AsyncClient, Path],
) -> None:
    client, _ = env
    await _seed_two_class_corpus(client)
    response = await client.post(
        "/api/v1/evaluate",
        json={"method": "default", "top_k": 3, "weights": {"hsv": 1.0}},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["report"]["weights"]["hsv"] == pytest.approx(1.0)
    for name in ("cm", "lbp", "glcm", "hog", "hu"):
        assert body["report"]["weights"][name] == pytest.approx(0.0)
