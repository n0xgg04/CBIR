"""Seed script: ingest a local AFHQ-style dataset tree into the corpus.

Phase 6 demo gate. Walks a directory laid out as

    <source>/<animal_type>/*.jpg

and ingests each image directly through the service layer (preprocess →
extract all six features → persist as `images` + `feature_sets` rows). The
script is intentionally network-free: download AFHQ once into `data/afhq/`
on the host (e.g. via `huggingface-cli download huggingface/cats_vs_dogs`
or by extracting the published Kaggle archive), then run

    uv run python -m scripts.seed_afhq --source data/afhq --max-per-class 200

CLI flags
---------
--source       Path to the dataset root (required). Each immediate
               subdirectory is treated as one `animal_type`.
--max-per-class
               Cap how many images to ingest per class (default: no cap).
--storage-root Where to copy originals to (overrides `STORAGE_ROOT`).
--db-url       SQLAlchemy async URL (overrides `DATABASE_URL`).
--dry-run      Walk and validate but do not write anything to DB / storage.
--verbose      Per-image progress output.

The script is idempotent — same `sha256` content is deduped via the unique
constraint on `images.sha256`, so re-running over the same dataset is safe.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Allow `python scripts/seed_afhq.py` to find the `app` package on sys.path.
_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import selectinload  # noqa: E402

from app.models import Base, FeatureSet, Image  # noqa: E402
from app.services import feature_cache  # noqa: E402
from app.services import features as feat  # noqa: E402
from app.services.preprocess import decode_bgr, preprocess  # noqa: E402
from app.services.storage import LocalStorage  # noqa: E402

SUPPORTED_EXTS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)
PROGRESS_INTERVAL: Final[int] = 25


@dataclass
class SeedStats:
    """Aggregate counters for one seed run."""

    classes_seen: int = 0
    files_seen: int = 0
    inserted: int = 0
    deduped: int = 0
    failed: int = 0
    elapsed_s: float = 0.0
    per_class_inserted: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _SeedConfig:
    source: Path
    storage_root: Path
    db_url: str
    max_per_class: int | None
    dry_run: bool
    verbose: bool


def _iter_class_dirs(source: Path) -> Iterable[tuple[str, Path]]:
    """Yield `(animal_type, dir_path)` for every immediate subdir of `source`."""
    for entry in sorted(source.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            yield entry.name, entry


def _iter_images(class_dir: Path, limit: int | None) -> Iterable[Path]:
    """Yield image paths under `class_dir`, capped by `limit` if set."""
    seen = 0
    for path in sorted(class_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        yield path
        seen += 1
        if limit is not None and seen >= limit:
            break


async def _ingest_one(
    session: AsyncSession,
    storage: LocalStorage,
    image_path: Path,
    animal_type: str,
) -> str:
    """Run the full ingest pipeline for one file. Returns 'inserted'|'deduped'."""
    payload = image_path.read_bytes()
    if not payload:
        raise ValueError(f"{image_path}: empty file")

    try:
        decoded = decode_bgr(payload)
    except ValueError as exc:
        raise ValueError(f"{image_path}: invalid image — {exc}") from exc
    height, width = decoded.shape[:2]

    stored = storage.save(
        payload, animal_type=animal_type, original_filename=image_path.name
    )

    existing = (
        await session.execute(
            select(Image)
            .options(selectinload(Image.feature_set))
            .where(Image.sha256 == stored.sha256)
        )
    ).scalar_one_or_none()

    if existing is not None and existing.feature_set is not None:
        return "deduped"

    if existing is None:
        image_row = Image(
            sha256=stored.sha256,
            filename=image_path.name,
            storage_path=stored.relative_path,
            animal_type=animal_type,
            width=width,
            height=height,
            size_bytes=stored.size_bytes,
            role="corpus",
        )
        session.add(image_row)
        await session.flush()
    else:
        image_row = existing

    preprocessed = preprocess(decoded)
    vectors = feat.extract_all(preprocessed)
    feature_row = FeatureSet(
        image_id=image_row.id,
        extractor_ver=feat.EXTRACTOR_VERSION,
        vec_hog=vectors["hog"].astype(float).tolist(),
        vec_hsv=vectors["hsv"].astype(float).tolist(),
        vec_lbp=vectors["lbp"].astype(float).tolist(),
        vec_glcm=vectors["glcm"].astype(float).tolist(),
        vec_hu=vectors["hu"].astype(float).tolist(),
        vec_cm=vectors["cm"].astype(float).tolist(),
    )
    session.add(feature_row)
    await session.commit()
    return "inserted"


async def seed(config: _SeedConfig) -> SeedStats:
    """Top-level seed routine — counts inserts, deduplicates by sha256."""
    if not config.source.is_dir():
        raise FileNotFoundError(f"source directory not found: {config.source}")

    stats = SeedStats()
    storage = LocalStorage(config.storage_root)

    engine = create_async_engine(config.db_url, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    started = time.perf_counter()
    try:
        async with session_maker() as session:
            for animal_type, class_dir in _iter_class_dirs(config.source):
                stats.classes_seen += 1
                stats.per_class_inserted.setdefault(animal_type, 0)
                if config.verbose:
                    print(f"[seed] class={animal_type!r} dir={class_dir}")

                for image_path in _iter_images(class_dir, config.max_per_class):
                    stats.files_seen += 1
                    if config.dry_run:
                        if config.verbose:
                            print(f"[seed] (dry) would ingest {image_path}")
                        continue
                    try:
                        outcome = await _ingest_one(
                            session, storage, image_path, animal_type
                        )
                    except (OSError, ValueError) as exc:
                        stats.failed += 1
                        print(f"[seed] FAILED {image_path}: {exc}", file=sys.stderr)
                        continue
                    if outcome == "inserted":
                        stats.inserted += 1
                        stats.per_class_inserted[animal_type] = (
                            stats.per_class_inserted.get(animal_type, 0) + 1
                        )
                    else:
                        stats.deduped += 1
                    if (
                        config.verbose
                        or stats.files_seen % PROGRESS_INTERVAL == 0
                    ):
                        print(
                            f"[seed] {stats.files_seen} seen | "
                            f"{stats.inserted} inserted | "
                            f"{stats.deduped} deduped | "
                            f"{stats.failed} failed"
                        )
    finally:
        await engine.dispose()

    if not config.dry_run:
        feature_cache.mark_dirty()
    stats.elapsed_s = time.perf_counter() - started
    return stats


def _parse_args(argv: list[str] | None = None) -> _SeedConfig:
    parser = argparse.ArgumentParser(
        description="Seed the CBIR corpus from a local AFHQ-style directory tree."
    )
    parser.add_argument(
        "--source", required=True, type=Path,
        help="Dataset root. Each immediate subdir is one animal_type.",
    )
    parser.add_argument(
        "--storage-root", type=Path, default=None,
        help="Override STORAGE_ROOT. Defaults to ./storage.",
    )
    parser.add_argument(
        "--db-url", type=str, default=None,
        help="Async SQLAlchemy URL. Defaults to env DATABASE_URL.",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Cap how many images to ingest per animal_type.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Walk and validate, but do not write anything.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Per-image log lines instead of every PROGRESS_INTERVAL.",
    )
    args = parser.parse_args(argv)

    storage_root = args.storage_root or Path(
        os.environ.get("STORAGE_ROOT", "./storage")
    )
    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        parser.error(
            "--db-url not provided and DATABASE_URL not set in the environment"
        )
    return _SeedConfig(
        source=args.source.resolve(),
        storage_root=storage_root.resolve(),
        db_url=db_url,
        max_per_class=args.max_per_class,
        dry_run=bool(args.dry_run),
        verbose=bool(args.verbose),
    )


def _print_summary(stats: SeedStats, *, dry_run: bool) -> None:
    title = "DRY-RUN" if dry_run else "SEED"
    print(f"\n[{title}] complete in {stats.elapsed_s:.2f}s")
    print(f"  classes seen   : {stats.classes_seen}")
    print(f"  files seen     : {stats.files_seen}")
    if not dry_run:
        print(f"  inserted       : {stats.inserted}")
        print(f"  deduped        : {stats.deduped}")
        print(f"  failed         : {stats.failed}")
        if stats.per_class_inserted:
            print("  per class:")
            for cls, count in sorted(stats.per_class_inserted.items()):
                print(f"    {cls:<16} {count}")


def main(argv: list[str] | None = None) -> int:
    config = _parse_args(argv)
    try:
        stats = asyncio.run(seed(config))
    except FileNotFoundError as exc:
        print(f"[seed] ERROR: {exc}", file=sys.stderr)
        return 2
    _print_summary(stats, dry_run=config.dry_run)
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
