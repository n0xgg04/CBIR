"""Pydantic DTOs for the public API surface."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

ImageRole = Literal["corpus", "query"]
FeatureName = Literal["hsv", "cm", "lbp", "glcm", "hog", "hu"]


class ImageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sha256: str
    filename: str
    storage_path: str
    animal_type: str
    width: int
    height: int
    size_bytes: int
    role: ImageRole
    uploaded_at: datetime


class FeatureSetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    image_id: int
    vectors: dict[str, list[float]]
    dims: dict[str, int]
    extracted_at: datetime
    extractor_ver: str


class ImageWithFeatures(BaseModel):
    """Response payload for `POST /images` once features have been extracted."""

    image: ImageRead
    features: FeatureSetRead


class ImageUploadForm(BaseModel):
    """Multipart form fields accompanying the file upload."""

    animal_type: Annotated[str, Field(min_length=1, max_length=64)]
    role: ImageRole = "corpus"


class SearchResultItem(BaseModel):
    """One row of the ranked top-K search response."""

    image_id: int
    rank: int
    score: float
    per_feature: dict[str, float]
    image: ImageRead | None = None


class SearchTraceStage(BaseModel):
    """One step in the pipeline trace shown by the UI inspector."""

    name: str
    elapsed_ms: int
    detail: dict[str, object] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Full POST /search payload."""

    run_id: int
    weights: dict[str, float]
    results: list[SearchResultItem]
    pipeline_trace: list[SearchTraceStage]
    elapsed_ms: int
    corpus_size: int
    query_dims: dict[str, int]


class RerankRequest(BaseModel):
    """PATCH /search/{run_id}/weights body."""

    weights: dict[str, float] = Field(
        default_factory=dict,
        description='Per-feature weights, e.g. {"hsv":0.2,"hog":0.4}',
    )
    top_k: Annotated[int, Field(ge=1, le=50)] = 5


# ---------------------------------------------------------------------------
# Evaluation (Phase 5)
# ---------------------------------------------------------------------------


EvaluationMethod = Literal["default", "ablation"]


class EvaluationMetricsRead(BaseModel):
    """One `(P@K, MAP@K)` summary."""

    precision_at_k: float
    map_at_k: float
    n_queries: int


class EvaluationReportRead(BaseModel):
    """One leave-one-out evaluation pass."""

    method: str
    weights: dict[str, float]
    top_k: int
    overall: EvaluationMetricsRead
    per_class: dict[str, EvaluationMetricsRead] = Field(default_factory=dict)


class AblationReportRead(BaseModel):
    """Base run plus one variant per ablated feature."""

    top_k: int
    base: EvaluationReportRead
    variants: dict[str, EvaluationReportRead] = Field(default_factory=dict)


class EvaluationRequest(BaseModel):
    """POST /api/v1/evaluate body."""

    method: EvaluationMethod = "default"
    top_k: Annotated[int, Field(ge=1, le=50)] = 10
    weights: dict[str, float] | None = Field(
        default=None,
        description="Override default weights — only honoured for `method=default`.",
    )


class EvaluationResponse(BaseModel):
    """POST /api/v1/evaluate response."""

    run_id: int
    method: str
    top_k: int
    corpus_size: int
    report: EvaluationReportRead | None = None
    ablation: AblationReportRead | None = None
    elapsed_ms: int


class CompareResponse(BaseModel):
    """POST /api/v1/compare response."""

    fused_score: float
    per_feature: dict[str, float]
    left_dims: dict[str, int]
    right_dims: dict[str, int]
    elapsed_ms: int
