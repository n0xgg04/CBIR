"""Unit + integration tests for PATCH /search/{run_id}/weights."""

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
from app.models import Base
from app.services import feature_cache, pipeline_emitter, search_engine


@pytest_asyncio.fixture
async def env(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, Path]]:
    db_path = tmp_path / "rerank.db"
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


def _jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buf.tobytes()


async def _ingest(client: AsyncClient, payload: bytes, name: str) -> int:
    response = await client.post(
        "/api/v1/images",
        files={"file": (name, payload, "image/jpeg")},
        data={"animal_type": "cat", "role": "corpus"},
    )
    assert response.status_code == 201, response.text
    return response.json()["image"]["id"]


# ---------- Pure helpers (no DB needed) ----------------------------------------


def test_parse_persisted_sub_scores_round_trips() -> None:
    trace = [
        {"name": "extract", "elapsed_ms": 10, "detail": {"dims": {"hsv": 768}}},
        {
            "name": "_sub_scores",
            "elapsed_ms": 0,
            "detail": {
                "image_ids": [1, 2, 3],
                "scores": {
                    "hsv": [0.1, 0.5, 0.9],
                    "hog": [0.7, 0.2, 0.1],
                },
            },
        },
    ]
    parsed = search_engine.parse_persisted_sub_scores(trace)
    assert parsed is not None
    image_ids, sims = parsed
    assert image_ids == [1, 2, 3]
    np.testing.assert_allclose(sims["hsv"], [0.1, 0.5, 0.9], atol=1e-6)
    np.testing.assert_allclose(sims["hog"], [0.7, 0.2, 0.1], atol=1e-6)
    assert sims["hsv"].dtype == np.float32


def test_parse_persisted_sub_scores_returns_none_when_missing() -> None:
    assert search_engine.parse_persisted_sub_scores([]) is None
    assert (
        search_engine.parse_persisted_sub_scores(
            [{"name": "extract", "elapsed_ms": 1, "detail": {}}]
        )
        is None
    )
    assert (
        search_engine.parse_persisted_sub_scores(
            [{"name": "_sub_scores", "elapsed_ms": 0, "detail": {"image_ids": "bad"}}]
        )
        is None
    )


def test_rerank_from_persisted_picks_top_image_under_new_weights() -> None:
    image_ids = [10, 20, 30]
    sims = {
        "hsv": np.array([0.9, 0.1, 0.1], dtype=np.float32),
        "hog": np.array([0.1, 0.1, 0.9], dtype=np.float32),
    }
    favouring_hsv = search_engine.rerank_from_persisted(
        image_ids=image_ids,
        per_feature_sims=sims,
        weights={"hsv": 1.0, "hog": 0.0},
        top_k=2,
    )
    assert [r.image_id for r in favouring_hsv] == [10, 20]
    assert favouring_hsv[0].rank == 1

    favouring_hog = search_engine.rerank_from_persisted(
        image_ids=image_ids,
        per_feature_sims=sims,
        weights={"hsv": 0.0, "hog": 1.0},
        top_k=2,
    )
    assert favouring_hog[0].image_id == 30


# ---------- HTTP integration ---------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_changes_results_under_new_weights(
    env: tuple[AsyncClient, Path], fixture_image_bgr: np.ndarray, fixture_image_bytes: bytes
) -> None:
    client, _ = env
    for q in (60, 75, 92):
        await _ingest(client, _jpeg(fixture_image_bgr, q), name=f"c{q}.jpg")

    initial = await client.post(
        "/api/v1/search",
        files={"file": ("q.jpg", fixture_image_bytes, "image/jpeg")},
        data={"top_k": "3"},
    )
    assert initial.status_code == 200, initial.text
    initial_body = initial.json()
    run_id = initial_body["run_id"]
    initial_top = [r["image_id"] for r in initial_body["results"]]

    rerank = await client.patch(
        f"/api/v1/search/{run_id}/weights",
        json={"weights": {"hu": 1.0}, "top_k": 3},
    )
    assert rerank.status_code == 200, rerank.text
    body = rerank.json()
    assert body["run_id"] == run_id
    rerank_top = [r["image_id"] for r in body["results"]]
    assert set(rerank_top) == set(initial_top)  # same corpus
    # Weights are renormalised; only `hu` should remain non-zero.
    assert body["weights"]["hu"] == pytest.approx(1.0)
    for name, w in body["weights"].items():
        if name != "hu":
            assert w == pytest.approx(0.0)
    # Sub-second budget per PLAN.md §11.
    assert body["elapsed_ms"] < 1000
    # The rerank trace stage should be present in the projected pipeline_trace.
    stage_names = [s["name"] for s in body["pipeline_trace"]]
    assert "rerank" in stage_names


@pytest.mark.asyncio
async def test_rerank_preserves_sub_scores_for_chained_reranks(
    env: tuple[AsyncClient, Path], fixture_image_bgr: np.ndarray, fixture_image_bytes: bytes
) -> None:
    client, _ = env
    for q in (60, 80):
        await _ingest(client, _jpeg(fixture_image_bgr, q), name=f"c{q}.jpg")

    first = await client.post(
        "/api/v1/search",
        files={"file": ("q.jpg", fixture_image_bytes, "image/jpeg")},
        data={"top_k": "2"},
    )
    run_id = first.json()["run_id"]

    # Two re-ranks back-to-back must both succeed (sub-scores stash preserved).
    r1 = await client.patch(
        f"/api/v1/search/{run_id}/weights", json={"weights": {"hsv": 1.0}}
    )
    assert r1.status_code == 200
    r2 = await client.patch(
        f"/api/v1/search/{run_id}/weights", json={"weights": {"hog": 1.0}}
    )
    assert r2.status_code == 200
    # Each re-rank logs an independent stage (so two `rerank` entries by now).
    rerank_stages = [s for s in r2.json()["pipeline_trace"] if s["name"] == "rerank"]
    assert len(rerank_stages) >= 2


@pytest.mark.asyncio
async def test_rerank_404_for_unknown_run(env: tuple[AsyncClient, Path]) -> None:
    client, _ = env
    response = await client.patch(
        "/api/v1/search/9999/weights", json={"weights": {"hsv": 1.0}}
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_rerank_409_when_sub_scores_missing(
    env: tuple[AsyncClient, Path], fixture_image_bgr: np.ndarray, fixture_image_bytes: bytes
) -> None:
    client, _ = env
    await _ingest(client, _jpeg(fixture_image_bgr, 80), name="c.jpg")
    initial = await client.post(
        "/api/v1/search",
        files={"file": ("q.jpg", fixture_image_bytes, "image/jpeg")},
        data={"top_k": "1"},
    )
    run_id = initial.json()["run_id"]

    # Strip the persisted sub-scores stash directly via SQL.
    from sqlalchemy import update

    from app.models import SearchRun

    engine = app_db.get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            update(SearchRun)
            .where(SearchRun.id == run_id)
            .values(pipeline_trace=[{"name": "extract", "elapsed_ms": 1, "detail": {}}])
        )

    response = await client.patch(
        f"/api/v1/search/{run_id}/weights", json={"weights": {"hsv": 1.0}}
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_rerank_persists_new_weights_to_db(
    env: tuple[AsyncClient, Path], fixture_image_bgr: np.ndarray, fixture_image_bytes: bytes
) -> None:
    client, _ = env
    await _ingest(client, _jpeg(fixture_image_bgr, 80), name="c.jpg")
    initial = await client.post(
        "/api/v1/search",
        files={"file": ("q.jpg", fixture_image_bytes, "image/jpeg")},
        data={"top_k": "1"},
    )
    run_id = initial.json()["run_id"]

    custom_weights = {"hu": 0.7, "hog": 0.3}
    response = await client.patch(
        f"/api/v1/search/{run_id}/weights", json={"weights": custom_weights}
    )
    assert response.status_code == 200

    # Verify by re-loading the row directly.
    from app.models import SearchRun

    engine = app_db.get_engine()
    async with engine.begin() as conn:
        cursor = await conn.execute(
            SearchRun.__table__.select().where(SearchRun.id == run_id)
        )
        row = cursor.mappings().one()
    persisted_weights = row["weights"]
    assert persisted_weights["hu"] == pytest.approx(0.7)
    assert persisted_weights["hog"] == pytest.approx(0.3)
    for name in ("hsv", "cm", "lbp", "glcm"):
        assert persisted_weights[name] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_rerank_with_empty_weights_falls_back_to_defaults(
    env: tuple[AsyncClient, Path], fixture_image_bgr: np.ndarray, fixture_image_bytes: bytes
) -> None:
    client, _ = env
    await _ingest(client, _jpeg(fixture_image_bgr, 80), name="c.jpg")
    initial = await client.post(
        "/api/v1/search",
        files={"file": ("q.jpg", fixture_image_bytes, "image/jpeg")},
        data={"top_k": "1"},
    )
    run_id = initial.json()["run_id"]

    response = await client.patch(
        f"/api/v1/search/{run_id}/weights", json={"weights": {}}
    )
    assert response.status_code == 200
    body = response.json()
    # Defaults sum to ~1.0 and contain all six features.
    assert sum(body["weights"].values()) == pytest.approx(1.0, abs=1e-3)
    assert set(body["weights"]) == {"hsv", "cm", "lbp", "glcm", "hog", "hu"}
