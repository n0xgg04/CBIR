"""Tests for `scripts/seed_afhq.py` — run against a tiny synthetic dataset."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import db as app_db
from app.config import get_settings
from app.models import Image
from app.services import feature_cache
from scripts.seed_afhq import _SeedConfig, main, seed


def _palette_image_bytes(seed_value: int) -> bytes:
    rng = np.random.default_rng(seed=seed_value)
    h, w = 96, 96
    base = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    base[..., 0] = ((xx + seed_value * 5) % 256).astype(np.uint8)
    base[..., 1] = ((yy + seed_value * 9) % 256).astype(np.uint8)
    base[..., 2] = ((xx + yy + seed_value * 17) % 256).astype(np.uint8)
    noise = rng.integers(0, 16, size=base.shape, dtype=np.int16)
    image = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return buf.tobytes()


def _build_dataset(root: Path, *, classes: dict[str, int]) -> None:
    """Materialise `<root>/<class>/<i>.jpg` files for each class count."""
    seed_counter = 0
    for animal_type, count in classes.items():
        class_dir = root / animal_type
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (class_dir / f"img_{i:03d}.jpg").write_bytes(
                _palette_image_bytes(seed_counter)
            )
            seed_counter += 1


@pytest.fixture
def isolated_env(tmp_path: Path):
    """Point the app config at an aiosqlite + tmp storage root."""
    db_path = tmp_path / "seed.db"
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

    try:
        yield {
            "db_url": os.environ["DATABASE_URL"],
            "storage_root": storage_root,
            "db_path": db_path,
        }
    finally:
        get_settings.cache_clear()
        app_db.reset_engine()
        feature_cache.reset()
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.asyncio
async def test_seed_inserts_rows_for_each_class(
    isolated_env: dict[str, object], tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    _build_dataset(dataset, classes={"cat": 3, "dog": 2})

    config = _SeedConfig(
        source=dataset,
        storage_root=isolated_env["storage_root"],  # type: ignore[arg-type]
        db_url=isolated_env["db_url"],  # type: ignore[arg-type]
        max_per_class=None,
        dry_run=False,
        verbose=False,
    )
    stats = await seed(config)
    assert stats.classes_seen == 2
    assert stats.files_seen == 5
    assert stats.inserted == 5
    assert stats.deduped == 0
    assert stats.failed == 0
    assert stats.per_class_inserted == {"cat": 3, "dog": 2}

    # Verify rows landed and feature_sets came along for every image.
    engine = create_async_engine(isolated_env["db_url"])  # type: ignore[arg-type]
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as session:
        images = (await session.execute(select(Image))).scalars().all()
        assert len(images) == 5
        labels = sorted(img.animal_type for img in images)
        assert labels == ["cat", "cat", "cat", "dog", "dog"]
    await engine.dispose()


@pytest.mark.asyncio
async def test_seed_is_idempotent_under_rerun(
    isolated_env: dict[str, object], tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    _build_dataset(dataset, classes={"cat": 2})

    config = _SeedConfig(
        source=dataset,
        storage_root=isolated_env["storage_root"],  # type: ignore[arg-type]
        db_url=isolated_env["db_url"],  # type: ignore[arg-type]
        max_per_class=None,
        dry_run=False,
        verbose=False,
    )
    first = await seed(config)
    second = await seed(config)
    assert first.inserted == 2
    assert second.inserted == 0
    assert second.deduped == 2


@pytest.mark.asyncio
async def test_seed_max_per_class_caps_count(
    isolated_env: dict[str, object], tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    _build_dataset(dataset, classes={"cat": 5, "dog": 5})
    config = _SeedConfig(
        source=dataset,
        storage_root=isolated_env["storage_root"],  # type: ignore[arg-type]
        db_url=isolated_env["db_url"],  # type: ignore[arg-type]
        max_per_class=2,
        dry_run=False,
        verbose=False,
    )
    stats = await seed(config)
    assert stats.inserted == 4
    assert stats.per_class_inserted == {"cat": 2, "dog": 2}


@pytest.mark.asyncio
async def test_seed_dry_run_writes_nothing(
    isolated_env: dict[str, object], tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    _build_dataset(dataset, classes={"cat": 2})
    config = _SeedConfig(
        source=dataset,
        storage_root=isolated_env["storage_root"],  # type: ignore[arg-type]
        db_url=isolated_env["db_url"],  # type: ignore[arg-type]
        max_per_class=None,
        dry_run=True,
        verbose=False,
    )
    stats = await seed(config)
    assert stats.inserted == 0
    assert stats.files_seen == 2
    # No originals were copied.
    assert list((isolated_env["storage_root"]).rglob("*.jpg")) == []  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_seed_handles_corrupt_files_without_aborting(
    isolated_env: dict[str, object], tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    cat_dir = dataset / "cat"
    cat_dir.mkdir(parents=True)
    cat_dir.joinpath("good.jpg").write_bytes(_palette_image_bytes(1))
    cat_dir.joinpath("bad.jpg").write_bytes(b"this-is-not-an-image")

    config = _SeedConfig(
        source=dataset,
        storage_root=isolated_env["storage_root"],  # type: ignore[arg-type]
        db_url=isolated_env["db_url"],  # type: ignore[arg-type]
        max_per_class=None,
        dry_run=False,
        verbose=False,
    )
    stats = await seed(config)
    assert stats.inserted == 1
    assert stats.failed == 1


def test_main_cli_runs_end_to_end(
    isolated_env: dict[str, object], tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    _build_dataset(dataset, classes={"fox": 2})
    rc = main(
        [
            "--source", str(dataset),
            "--storage-root", str(isolated_env["storage_root"]),
            "--db-url", str(isolated_env["db_url"]),
        ]
    )
    assert rc == 0


def test_main_cli_missing_source_returns_2(
    isolated_env: dict[str, object], tmp_path: Path
) -> None:
    rc = main(
        [
            "--source", str(tmp_path / "does-not-exist"),
            "--storage-root", str(isolated_env["storage_root"]),
            "--db-url", str(isolated_env["db_url"]),
        ]
    )
    assert rc == 2
