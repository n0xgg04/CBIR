"""SQLAlchemy 2.0 ORM models matching the Phase 1 schema in PLAN.md §5."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Project-wide declarative base."""


# JSON column type that materialises as JSONB on Postgres and JSON elsewhere
# (so SQLite-backed test runs don't need an extension).
JsonField = JSON().with_variant(JSONB(astext_type=Text()), "postgresql")

# BIGINT primary keys do not autoincrement on SQLite — only the magic
# `INTEGER PRIMARY KEY` does. Variant keeps Postgres on bigint, SQLite on int.
BigPK = BigInteger().with_variant(Integer(), "sqlite")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    animal_type: Mapped[str] = mapped_column(Text, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False, default="corpus")
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    feature_set: Mapped[FeatureSet | None] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
        uselist=False,
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint("role IN ('corpus','query')", name="ck_images_role"),
        Index("idx_images_animal_type", "animal_type"),
        Index("idx_images_role", "role"),
    )


class FeatureSet(Base):
    __tablename__ = "feature_sets"

    image_id: Mapped[int] = mapped_column(
        BigPK,
        ForeignKey("images.id", ondelete="CASCADE"),
        primary_key=True,
    )
    vectors: Mapped[dict[str, list[float]]] = mapped_column(JsonField, nullable=False)
    dims: Mapped[dict[str, int]] = mapped_column(JsonField, nullable=False)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    extractor_ver: Mapped[str] = mapped_column(Text, nullable=False)

    image: Mapped[Image] = relationship(back_populates="feature_set")


class SearchRun(Base):
    __tablename__ = "search_runs"

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    query_image_id: Mapped[int | None] = mapped_column(
        BigPK,
        ForeignKey("images.id", ondelete="SET NULL"),
        nullable=True,
    )
    weights: Mapped[dict[str, float]] = mapped_column(JsonField, nullable=False)
    results: Mapped[list[dict[str, Any]]] = mapped_column(JsonField, nullable=False)
    pipeline_trace: Mapped[list[dict[str, Any]]] = mapped_column(JsonField, nullable=False)
    elapsed_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    method: Mapped[str] = mapped_column(Text, nullable=False)
    precision_at_5: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    map_at_10: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    per_class: Mapped[dict[str, Any] | None] = mapped_column(JsonField, nullable=True)
    ablation: Mapped[dict[str, Any] | None] = mapped_column(JsonField, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
