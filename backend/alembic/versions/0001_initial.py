"""initial schema: images, feature_sets, search_runs, evaluation_runs

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-03 23:30:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> sa.types.TypeEngine:
    """JSONB on Postgres, JSON elsewhere (so SQLite test runs don't break)."""
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def _big_pk() -> sa.types.TypeEngine:
    """BIGINT on Postgres, plain INTEGER on SQLite (only INTEGER autoincrements there)."""
    return sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "images",
        sa.Column("id", _big_pk(), primary_key=True, autoincrement=True),
        sa.Column("sha256", sa.String(64), nullable=False, unique=True),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("animal_type", sa.Text(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column(
            "role",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'corpus'"),
        ),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint("role IN ('corpus','query')", name="ck_images_role"),
    )
    op.create_index("idx_images_animal_type", "images", ["animal_type"])
    op.create_index("idx_images_role", "images", ["role"])

    op.create_table(
        "feature_sets",
        sa.Column(
            "image_id",
            _big_pk(),
            sa.ForeignKey("images.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("vectors", _jsonb(), nullable=False),
        sa.Column("dims", _jsonb(), nullable=False),
        sa.Column(
            "extracted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("extractor_ver", sa.Text(), nullable=False),
    )

    op.create_table(
        "search_runs",
        sa.Column("id", _big_pk(), primary_key=True, autoincrement=True),
        sa.Column(
            "query_image_id",
            _big_pk(),
            sa.ForeignKey("images.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("weights", _jsonb(), nullable=False),
        sa.Column("results", _jsonb(), nullable=False),
        sa.Column("pipeline_trace", _jsonb(), nullable=False),
        sa.Column("elapsed_ms", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )

    op.create_table(
        "evaluation_runs",
        sa.Column("id", _big_pk(), primary_key=True, autoincrement=True),
        sa.Column("method", sa.Text(), nullable=False),
        sa.Column("precision_at_5", sa.Numeric(5, 4), nullable=True),
        sa.Column("map_at_10", sa.Numeric(5, 4), nullable=True),
        sa.Column("per_class", _jsonb(), nullable=True),
        sa.Column("ablation", _jsonb(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("evaluation_runs")
    op.drop_table("search_runs")
    op.drop_table("feature_sets")
    op.drop_index("idx_images_role", table_name="images")
    op.drop_index("idx_images_animal_type", table_name="images")
    op.drop_table("images")
