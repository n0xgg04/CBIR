"""One-shot migration: read feature_sets.vectors (JSONB) → write vec_* columns.

Safe to run multiple times (idempotent: skips rows where vec_hog IS NOT NULL).
Usage:
    cd backend && uv run python scripts/backfill_pgvector.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select

from app.db import get_engine, get_sessionmaker
from app.models import Base, FeatureSet

BATCH = 500  # rows per commit

VEC_COLS = ["vec_hog", "vec_hsv", "vec_lbp", "vec_glcm", "vec_hu", "vec_cm"]
VEC_KEYS = ["hog", "hsv", "lbp", "glcm", "hu", "cm"]


async def backfill() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    sm = get_sessionmaker()
    async with sm() as session:
        result = await session.execute(
            select(FeatureSet).where(FeatureSet.vec_hog.is_(None))
        )
        rows = result.scalars().all()
        total = len(rows)
        if total == 0:
            print("No rows need backfill — all vec_* columns are already populated.")
            return

        print(f"Backfilling {total} rows...")
        for i, row in enumerate(rows):
            for col, key in zip(VEC_COLS, VEC_KEYS, strict=True):
                setattr(row, col, row.vectors[key])
            if (i + 1) % BATCH == 0:
                await session.commit()
                print(f"  committed {i + 1}/{total}")
        await session.commit()
        print(f"Backfill complete: {total} rows updated.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(backfill())
