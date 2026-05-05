"""Set NOT NULL on vec_* columns and create HNSW indexes.

Revision ID: 0003_pgvector_notnull_indexes
Revises: 0002_pgvector_columns
Create Date: 2026-05-05

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_pgvector_notnull_indexes"
down_revision: str | None = "0002_pgvector_columns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    """Return True when the current dialect is PostgreSQL."""
    return op.get_context().dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        # SQLite does not support pgvector — skip index creation.
        return

    op.alter_column("feature_sets", "vec_hog", nullable=False)
    op.alter_column("feature_sets", "vec_hsv", nullable=False)
    op.alter_column("feature_sets", "vec_lbp", nullable=False)
    op.alter_column("feature_sets", "vec_glcm", nullable=False)
    op.alter_column("feature_sets", "vec_hu", nullable=False)
    op.alter_column("feature_sets", "vec_cm", nullable=False)

    # HNSW indexes for cosine similarity on each feature vector.
    # m = 16 (edges per node), ef_construction = 64 (build-time beam width).
    # NOTE: vec_hog (8100-D) exceeds pgvector's 2000-D index limit, so it is
    # intentionally left un-indexed. Exact HOG cosine is computed at fusion time
    # on the candidate subset produced by the other five features.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hnsw_hsv
            ON feature_sets USING hnsw (vec_hsv vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hnsw_lbp
            ON feature_sets USING hnsw (vec_lbp vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hnsw_glcm
            ON feature_sets USING hnsw (vec_glcm vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hnsw_hu
            ON feature_sets USING hnsw (vec_hu vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hnsw_cm
            ON feature_sets USING hnsw (vec_cm vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    if not _is_postgres():
        return

    op.execute("DROP INDEX IF EXISTS idx_hnsw_cm")
    op.execute("DROP INDEX IF EXISTS idx_hnsw_hu")
    op.execute("DROP INDEX IF EXISTS idx_hnsw_glcm")
    op.execute("DROP INDEX IF EXISTS idx_hnsw_lbp")
    op.execute("DROP INDEX IF EXISTS idx_hnsw_hsv")

    op.alter_column("feature_sets", "vec_hog", nullable=True)
    op.alter_column("feature_sets", "vec_hsv", nullable=True)
    op.alter_column("feature_sets", "vec_lbp", nullable=True)
    op.alter_column("feature_sets", "vec_glcm", nullable=True)
    op.alter_column("feature_sets", "vec_hu", nullable=True)
    op.alter_column("feature_sets", "vec_cm", nullable=True)
