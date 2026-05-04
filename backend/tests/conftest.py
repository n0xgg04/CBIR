"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import cv2
import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """ASGI in-process httpx client — no network, no live server required."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        yield http_client


@pytest.fixture
def fixture_image_bgr() -> np.ndarray:
    """Deterministic 256×256 BGR image with a structured colour gradient.

    Used by feature/preprocess tests so output values are stable across runs.
    """
    rng = np.random.default_rng(seed=7)
    h, w = 256, 256
    base = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    base[..., 0] = (xx % 256).astype(np.uint8)              # B gradient (left→right)
    base[..., 1] = (yy % 256).astype(np.uint8)              # G gradient (top→bottom)
    base[..., 2] = ((xx + yy) % 256).astype(np.uint8)       # R diagonal
    noise = rng.integers(0, 12, size=base.shape, dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def fixture_image_path(tmp_path: Path, fixture_image_bgr: np.ndarray) -> Path:
    """Persisted PNG of `fixture_image_bgr` for path-based code paths."""
    out = tmp_path / "fixture.png"
    cv2.imwrite(str(out), fixture_image_bgr)
    return out


@pytest.fixture
def fixture_image_bytes(fixture_image_bgr: np.ndarray) -> bytes:
    """In-memory JPEG-encoded fixture image."""
    ok, buf = cv2.imencode(".jpg", fixture_image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok, "failed to encode fixture as JPEG"
    return buf.tobytes()


@pytest.fixture
def preprocessed_fixture(fixture_image_bgr: np.ndarray) -> np.ndarray:
    """Convenience: the 128×128 preprocessed version of the fixture."""
    from app.services.preprocess import preprocess

    return preprocess(fixture_image_bgr)
