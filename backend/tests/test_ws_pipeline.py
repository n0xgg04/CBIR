"""Integration tests for the WebSocket pipeline streaming endpoint."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import db as app_db
from app.config import get_settings
from app.main import create_app
from app.models import Base
from app.services import feature_cache, pipeline_emitter


@pytest.fixture
def env(tmp_path: Path) -> Iterator[tuple[TestClient, Path]]:
    """Synchronous TestClient bound to a fresh sqlite DB and storage root."""
    db_path = tmp_path / "ws.db"
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

    # Bootstrap schema synchronously via the new engine.
    import asyncio

    async def _create_schema() -> None:
        engine = app_db.get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create_schema())
    app_db.reset_engine()  # re-create the engine inside TestClient's loop

    app = create_app()
    with TestClient(app) as client:
        yield client, storage_root

    app_db.reset_engine()
    get_settings.cache_clear()
    feature_cache.reset()
    pipeline_emitter.reset()
    for k, v in previous.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _jpeg_bytes(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buf.tobytes()


def _ingest(client: TestClient, payload: bytes, name: str = "kitty.jpg") -> int:
    files = {"file": (name, payload, "image/jpeg")}
    data = {"animal_type": "cat", "role": "corpus"}
    response = client.post("/api/v1/images", files=files, data=data)
    assert response.status_code == 201, response.text
    return response.json()["image"]["id"]


def test_create_stream_endpoint_returns_unique_ids(env: tuple[TestClient, Path]) -> None:
    client, _ = env
    a = client.post("/api/v1/search/streams")
    b = client.post("/api/v1/search/streams")
    assert a.status_code == 201
    assert b.status_code == 201
    assert a.json()["stream_id"] != b.json()["stream_id"]


def test_ws_replays_pipeline_events_for_completed_search(
    env: tuple[TestClient, Path], fixture_image_bgr: np.ndarray, fixture_image_bytes: bytes
) -> None:
    client, _ = env

    # Seed corpus with a few JPEG-quality variants so cosine results are distinct.
    for q in (60, 75, 92):
        _ingest(client, _jpeg_bytes(fixture_image_bgr, q), name=f"c{q}.jpg")

    stream_resp = client.post("/api/v1/search/streams")
    assert stream_resp.status_code == 201
    stream_id = stream_resp.json()["stream_id"]

    # Synchronous POST /search buffers every event in the emitter's history.
    files = {"file": ("query.jpg", fixture_image_bytes, "image/jpeg")}
    data = {"top_k": "3", "stream_id": stream_id}
    search_resp = client.post("/api/v1/search", files=files, data=data)
    assert search_resp.status_code == 200, search_resp.text

    events: list[dict] = []
    with client.websocket_connect(f"/api/v1/ws/search/{stream_id}") as ws:
        try:
            while True:
                events.append(ws.receive_json())
        except WebSocketDisconnect:
            pass

    kinds = [e["kind"] for e in events]
    assert kinds[0] == "stage.start"
    assert events[0]["payload"]["name"] == "extract"
    assert kinds[-1] == "rank.done"
    # All four pipeline stages should appear at least once each.
    stage_names = [
        e["payload"]["name"] for e in events if e["kind"] == "stage.start"
    ]
    assert stage_names == ["extract", "load_corpus", "cosine", "rank"]
    # Per-feature events must cover all six extractors at least once.
    feature_names = {
        e["payload"]["name"] for e in events if e["kind"] == "feature.done"
    }
    assert feature_names >= {"hsv", "cm", "lbp", "glcm", "hog", "hu"}
    # Sequence numbers are strictly increasing.
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_ws_unknown_stream_id_closes_with_policy_error(
    env: tuple[TestClient, Path],
) -> None:
    client, _ = env
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/api/v1/ws/search/nope") as ws:
            ws.receive_json()
    assert excinfo.value.code == 1008


def test_search_with_unknown_stream_id_returns_404(
    env: tuple[TestClient, Path], fixture_image_bgr: np.ndarray, fixture_image_bytes: bytes
) -> None:
    client, _ = env
    _ingest(client, _jpeg_bytes(fixture_image_bgr, 80))
    files = {"file": ("query.jpg", fixture_image_bytes, "image/jpeg")}
    data = {"top_k": "1", "stream_id": "missing-id"}
    response = client.post("/api/v1/search", files=files, data=data)
    assert response.status_code == 404


def test_search_without_stream_id_still_works(
    env: tuple[TestClient, Path], fixture_image_bgr: np.ndarray, fixture_image_bytes: bytes
) -> None:
    """Streaming is opt-in — searches without a stream_id keep their old shape."""
    client, _ = env
    _ingest(client, _jpeg_bytes(fixture_image_bgr, 80))
    files = {"file": ("query.jpg", fixture_image_bytes, "image/jpeg")}
    data = {"top_k": "1"}
    response = client.post("/api/v1/search", files=files, data=data)
    assert response.status_code == 200
    body = response.json()
    assert body["corpus_size"] == 1
    assert body["results"][0]["rank"] == 1
