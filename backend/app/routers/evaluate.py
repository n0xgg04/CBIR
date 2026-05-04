"""POST /api/v1/evaluate — run leave-one-out P@K / MAP@K (and optional ablation).

Phase 5 demo gate. Persists each run as a row in `evaluation_runs` so the
frontend can display the most recent metric snapshot without recomputing.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import EvaluationRun
from app.schemas import (
    AblationReportRead,
    EvaluationMetricsRead,
    EvaluationReportRead,
    EvaluationRequest,
    EvaluationResponse,
)
from app.services import evaluator
from app.services import features as feat
from app.services.search_engine import DEFAULT_WEIGHTS, normalise_weights

router = APIRouter(prefix="/api/v1", tags=["evaluate"])

# Standard demo grid from PLAN.md §11 — P@5 and MAP@10 are the persisted columns.
P_AT_K_DEMO: int = 5
MAP_AT_K_DEMO: int = 10


def _metrics_to_read(m: evaluator.Metrics) -> EvaluationMetricsRead:
    return EvaluationMetricsRead(
        precision_at_k=m.precision_at_k,
        map_at_k=m.map_at_k,
        n_queries=m.n_queries,
    )


def _report_to_read(r: evaluator.EvaluationReport) -> EvaluationReportRead:
    return EvaluationReportRead(
        method=r.method,
        weights=r.weights,
        top_k=r.top_k,
        overall=_metrics_to_read(r.overall),
        per_class={k: _metrics_to_read(v) for k, v in r.per_class.items()},
    )


def _ablation_to_read(a: evaluator.AblationReport) -> AblationReportRead:
    return AblationReportRead(
        top_k=a.top_k,
        base=_report_to_read(a.base),
        variants={k: _report_to_read(v) for k, v in a.variants.items()},
    )


def _compute_demo_metrics(
    snapshot: evaluator.LabeledSnapshot,
    full_sims: dict,
    weights: dict[str, float],
) -> tuple[evaluator.Metrics, evaluator.Metrics]:
    """Compute the two metrics persisted on `evaluation_runs` (P@5, MAP@10)."""
    overall_p5, _ = evaluator._evaluate_with_sims(
        snapshot, full_sims, weights, P_AT_K_DEMO
    )
    overall_m10, _ = evaluator._evaluate_with_sims(
        snapshot, full_sims, weights, MAP_AT_K_DEMO
    )
    return overall_p5, overall_m10


def _build_default_report(
    snapshot: evaluator.LabeledSnapshot,
    full_sims: dict,
    weights: dict[str, float],
    top_k: int,
) -> evaluator.EvaluationReport:
    overall, per_class = evaluator._evaluate_with_sims(
        snapshot, full_sims, weights, top_k
    )
    return evaluator.EvaluationReport(
        method="default",
        weights=dict(weights),
        top_k=top_k,
        overall=overall,
        per_class=per_class,
    )


def _build_ablation_report(
    snapshot: evaluator.LabeledSnapshot,
    full_sims: dict,
    top_k: int,
) -> evaluator.AblationReport:
    base_weights = normalise_weights(DEFAULT_WEIGHTS)
    base = evaluator._build_report(
        snapshot=snapshot,
        full_sims=full_sims,
        weights=base_weights,
        top_k=top_k,
        method="default",
    )
    variants: dict[str, evaluator.EvaluationReport] = {}
    for feature_name in feat.EXPECTED_DIMS:
        ablated = {
            k: (0.0 if k == feature_name else v) for k, v in DEFAULT_WEIGHTS.items()
        }
        variants[f"minus_{feature_name}"] = evaluator._build_report(
            snapshot=snapshot,
            full_sims=full_sims,
            weights=normalise_weights(ablated),
            top_k=top_k,
            method=f"minus_{feature_name}",
        )
    return evaluator.AblationReport(base=base, variants=variants, top_k=top_k)


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Leave-one-out P@K / MAP@K (optionally ablating each feature).",
)
async def evaluate_corpus(
    db: Annotated[AsyncSession, Depends(get_db)],
    body: EvaluationRequest,
) -> EvaluationResponse:
    snapshot = await evaluator.load_labeled_snapshot(db)
    if snapshot.size < 2:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"corpus size {snapshot.size} is too small to evaluate (need >= 2 images)",
        )
    distinct_labels = set(snapshot.labels)
    if len(distinct_labels) < 2:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "evaluation requires at least 2 distinct animal_type classes in the corpus",
        )

    full_sims = await asyncio.to_thread(evaluator._full_similarity, snapshot)
    started_wall = datetime.now(UTC)
    started = time.perf_counter()

    report: evaluator.EvaluationReport | None = None
    ablation: evaluator.AblationReport | None = None
    persisted_weights: dict[str, float]

    if body.method == "default":
        weights = normalise_weights(body.weights or DEFAULT_WEIGHTS)
        report = await asyncio.to_thread(
            _build_default_report, snapshot, full_sims, weights, body.top_k
        )
        persisted_weights = weights
    else:  # method == "ablation"
        ablation = await asyncio.to_thread(
            _build_ablation_report, snapshot, full_sims, body.top_k
        )
        persisted_weights = normalise_weights(DEFAULT_WEIGHTS)

    p5_metrics, m10_metrics = await asyncio.to_thread(
        _compute_demo_metrics, snapshot, full_sims, persisted_weights
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    run_row = EvaluationRun(
        method=body.method,
        precision_at_5=(
            float(p5_metrics.precision_at_k) if p5_metrics.n_queries else None
        ),
        map_at_10=float(m10_metrics.map_at_k) if m10_metrics.n_queries else None,
        per_class=(
            {k: evaluator.metrics_to_jsonable(v) for k, v in report.per_class.items()}
            if report is not None
            else None
        ),
        ablation=(
            evaluator.ablation_to_jsonable(ablation) if ablation is not None else None
        ),
        started_at=started_wall,
        finished_at=datetime.now(UTC),
    )
    db.add(run_row)
    await db.commit()
    await db.refresh(run_row)

    return EvaluationResponse(
        run_id=run_row.id,
        method=body.method,
        top_k=body.top_k,
        corpus_size=snapshot.size,
        report=_report_to_read(report) if report is not None else None,
        ablation=_ablation_to_read(ablation) if ablation is not None else None,
        elapsed_ms=elapsed_ms,
    )
