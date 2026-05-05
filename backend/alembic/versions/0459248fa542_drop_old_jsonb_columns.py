"""drop old jsonb columns

Revision ID: 0459248fa542
Revises: 0003_pgvector_notnull_indexes
Create Date: 2026-05-05 14:57:45.524135

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0459248fa542'
down_revision: str | None = '0003_pgvector_notnull_indexes'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("feature_sets", "vectors")
    op.drop_column("feature_sets", "dims")
    op.drop_column("feature_sets", "extracted_at")


def downgrade() -> None:
    op.add_column(
        "feature_sets",
        sa.Column("vectors", sa.JSON(), nullable=False, server_default="{}"),
    )
    op.add_column(
        "feature_sets",
        sa.Column("dims", sa.JSON(), nullable=False, server_default="{}"),
    )
    op.add_column(
        "feature_sets",
        sa.Column(
            "extracted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
