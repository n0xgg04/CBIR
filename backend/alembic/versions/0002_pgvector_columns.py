"""Add pgvector typed columns to feature_sets.

Revision ID: 0002_pgvector_columns
Revises: 0001_initial
Create Date: 2026-05-05

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_pgvector_columns"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    """Return True when the current dialect is PostgreSQL."""
    return op.get_context().dialect.name == "postgresql"


def upgrade() -> None:
    if _is_postgres():
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Add typed vector columns — nullable initially so backfill can run safely.
    # On SQLite these degrade to JSON/BLOB and are ignored in search queries.
    if _is_postgres():
        op.execute("ALTER TABLE feature_sets ADD COLUMN vec_hog vector(8100)")
        op.execute("ALTER TABLE feature_sets ADD COLUMN vec_hsv vector(768)")
        op.execute("ALTER TABLE feature_sets ADD COLUMN vec_lbp vector(18)")
        op.execute("ALTER TABLE feature_sets ADD COLUMN vec_glcm vector(40)")
        op.execute("ALTER TABLE feature_sets ADD COLUMN vec_hu vector(7)")
        op.execute("ALTER TABLE feature_sets ADD COLUMN vec_cm vector(9)")
    else:
        op.add_column("feature_sets", sa.Column("vec_hog", sa.JSON(), nullable=True))
        op.add_column("feature_sets", sa.Column("vec_hsv", sa.JSON(), nullable=True))
        op.add_column("feature_sets", sa.Column("vec_lbp", sa.JSON(), nullable=True))
        op.add_column("feature_sets", sa.Column("vec_glcm", sa.JSON(), nullable=True))
        op.add_column("feature_sets", sa.Column("vec_hu", sa.JSON(), nullable=True))
        op.add_column("feature_sets", sa.Column("vec_cm", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("feature_sets", "vec_cm")
    op.drop_column("feature_sets", "vec_hu")
    op.drop_column("feature_sets", "vec_glcm")
    op.drop_column("feature_sets", "vec_lbp")
    op.drop_column("feature_sets", "vec_hsv")
    op.drop_column("feature_sets", "vec_hog")
