"""POST /api/v1/search — query upload → preprocess → cosine ranking → top-K.

Phase 2 demo gate. The router persists every run to `search_runs` so the
Phase 3 visualiser and the Phase 4 PATCH /weights re-rank can replay it
without re-uploading the image.
"""

from __future__ import annotations

import json
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi import Path as PathParam
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Image, SearchRun
from app.schemas import (
    ImageRead,
    RerankRequest,
    SearchResponse,
    SearchResultItem,
    SearchTraceStage,
)
from app.services import pipeline_emitter, search_engine
from app.services.preprocess import decode_bgr, preprocess

router = APIRouter(prefix="/api/v1", tags=["search"])

MAX_QUERY_BYTES: int = 8 * 1024 * 1024
MIN_TOP_K: int = 1
MAX_TOP_K: int = 50


def _parse_weights(raw: str | None) -> dict[str, float] | None:
    """JSON-encoded weights field — `None` falls back to defaults."""
    if raw is None or raw.strip() == "":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"weights is not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "weights must be a JSON object"
        )
    out: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"weight for '{key}' is not a number",
            ) from exc
    return out


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search the corpus by query image — preprocess, fuse, rank top-K.",
)
async def search_images(
    file: Annotated[UploadFile, File(description="Query image (jpg/png/webp).")],
    db: Annotated[AsyncSession, Depends(get_db)],
    top_k: Annotated[int, Form(ge=MIN_TOP_K, le=MAX_TOP_K)] = search_engine.DEFAULT_TOP_K,
    weights: Annotated[
        str | None,
        Form(description='Optional JSON object: {"hsv":0.2,"hog":0.3,...}'),
    ] = None,
    stream_id: Annotated[
        str | None,
        Form(description="Stream ID from POST /api/v1/search/streams for live timeline."),
    ] = None,
) -> SearchResponse:
    payload = await file.read()
    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty query upload")
    if len(payload) > MAX_QUERY_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"query exceeds {MAX_QUERY_BYTES} bytes",
        )

    decode_started = time.perf_counter()
    try:
        decoded = decode_bgr(payload)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"invalid image: {exc}"
        ) from exc
    decode_elapsed_ms = int((time.perf_counter() - decode_started) * 1000)

    user_weights = _parse_weights(weights)

    preprocess_started = time.perf_counter()
    preprocessed = preprocess(decoded)
    preprocess_elapsed_ms = int((time.perf_counter() - preprocess_started) * 1000)

    emitter = None
    if stream_id is not None and stream_id.strip() != "":
        emitter = await pipeline_emitter.get_stream(stream_id)
        if emitter is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"unknown stream_id {stream_id!r}; allocate one via POST /api/v1/search/streams",
            )

    try:
        outcome, per_feature_sims = await search_engine.run_search(
            db,
            preprocessed=preprocessed,
            weights=user_weights,
            top_k=top_k,
            emitter=emitter,
        )
    except Exception as exc:
        if emitter is not None:
            await emitter.close_with_error(str(exc))
        raise

    if outcome.corpus_size == 0:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "corpus is empty — ingest images via POST /api/v1/images first",
        )

    pipeline_trace_jsonable = [
        {"name": "decode", "elapsed_ms": decode_elapsed_ms, "detail": {}},
        {"name": "preprocess", "elapsed_ms": preprocess_elapsed_ms, "detail": {}},
        *search_engine.trace_to_jsonable(outcome.trace),
    ]
    sub_scores = search_engine.per_feature_sims_to_jsonable(
        outcome.image_ids,
        per_feature_sims,
    )
    pipeline_trace_jsonable.append(
        {"name": "_sub_scores", "elapsed_ms": 0, "detail": sub_scores}
    )

    total_elapsed_ms = decode_elapsed_ms + preprocess_elapsed_ms + outcome.elapsed_ms
    run_row = SearchRun(
        query_image_id=None,  # Phase 2: queries are not persisted as images
        weights=outcome.weights,
        results=search_engine.results_to_jsonable(outcome.results),
        pipeline_trace=pipeline_trace_jsonable,
        elapsed_ms=total_elapsed_ms,
    )
    db.add(run_row)
    await db.commit()
    await db.refresh(run_row)

    # Hydrate the result items with thin Image DTOs so the UI can render
    # thumbnails without an extra round-trip.
    result_image_ids = [r.image_id for r in outcome.results]
    image_lookup: dict[int, Image] = {}
    if result_image_ids:
        rows = await db.execute(
            select(Image).where(Image.id.in_(result_image_ids))
        )
        image_lookup = {img.id: img for img in rows.scalars().all()}

    items: list[SearchResultItem] = []
    for r in outcome.results:
        img = image_lookup.get(r.image_id)
        items.append(
            SearchResultItem(
                image_id=r.image_id,
                rank=r.rank,
                score=r.score,
                per_feature=r.per_feature,
                image=ImageRead.model_validate(img) if img is not None else None,
            )
        )

    # Build the full trace DTO (decode + preprocess + engine trace, skip _sub_scores tail)
    full_trace: list[SearchTraceStage] = []
    for raw in pipeline_trace_jsonable:
        if isinstance(raw, dict) and raw.get("name") != "_sub_scores":
            full_trace.append(
                SearchTraceStage(
                    name=str(raw["name"]),
                    elapsed_ms=int(raw.get("elapsed_ms", 0)),
                    detail=raw.get("detail") or {},
                )
            )

    return SearchResponse(
        run_id=run_row.id,
        weights=outcome.weights,
        results=items,
        pipeline_trace=full_trace,
        elapsed_ms=total_elapsed_ms,
        corpus_size=outcome.corpus_size,
        query_dims=outcome.query_dims,
    )


async def _hydrate_results(
    db: AsyncSession, results: list[search_engine.SearchResult]
) -> list[SearchResultItem]:
    """Attach thin `ImageRead` DTOs so the frontend can render thumbnails."""
    image_ids = [r.image_id for r in results]
    image_lookup: dict[int, Image] = {}
    if image_ids:
        rows = await db.execute(select(Image).where(Image.id.in_(image_ids)))
        image_lookup = {img.id: img for img in rows.scalars().all()}
    return [
        SearchResultItem(
            image_id=r.image_id,
            rank=r.rank,
            score=r.score,
            per_feature=r.per_feature,
            image=ImageRead.model_validate(image_lookup[r.image_id])
            if r.image_id in image_lookup
            else None,
        )
        for r in results
    ]


def _stages_from_trace(
    trace: list[dict[str, object]],
) -> list[SearchTraceStage]:
    """Project the JSONB-stored trace back into typed DTOs (skipping `_sub_scores`)."""
    out: list[SearchTraceStage] = []
    for raw in trace:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not isinstance(name, str) or name == "_sub_scores":
            continue
        elapsed_ms = raw.get("elapsed_ms", 0)
        detail = raw.get("detail") or {}
        out.append(
            SearchTraceStage(
                name=name,
                elapsed_ms=int(elapsed_ms) if isinstance(elapsed_ms, int | float) else 0,
                detail=detail if isinstance(detail, dict) else {},
            )
        )
    return out


def _query_dims_from_trace(trace: list[dict[str, object]]) -> dict[str, int]:
    """Extract per-feature dims from the persisted `extract` stage detail."""
    for raw in trace:
        if isinstance(raw, dict) and raw.get("name") == "extract":
            detail = raw.get("detail") or {}
            dims = detail.get("dims") if isinstance(detail, dict) else None
            if isinstance(dims, dict):
                return {str(k): int(v) for k, v in dims.items()}
    return {}


@router.patch(
    "/search/{run_id}/weights",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Re-rank a stored search under fresh weights — no extraction.",
)
async def rerank_search(
    db: Annotated[AsyncSession, Depends(get_db)],
    run_id: Annotated[int, PathParam(ge=1)],
    body: RerankRequest,
) -> SearchResponse:
    run = await db.get(SearchRun, run_id)
    if run is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"search run {run_id} not found"
        )

    parsed = search_engine.parse_persisted_sub_scores(run.pipeline_trace)
    if parsed is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"search run {run_id} has no persisted sub-scores; cannot re-rank",
        )
    image_ids, per_feature_sims = parsed

    started = time.perf_counter()
    cleaned_weights = search_engine.normalise_weights(body.weights or {})
    new_results = search_engine.rerank_from_persisted(
        image_ids=image_ids,
        per_feature_sims=per_feature_sims,
        weights=body.weights,
        top_k=body.top_k,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # Splice a `rerank` stage in front of the trailing `_sub_scores` stash so
    # subsequent re-ranks can keep replaying the same sub-scores.
    existing_trace = list(run.pipeline_trace or [])
    sub_scores_tail = (
        existing_trace.pop()
        if existing_trace and existing_trace[-1].get("name") == "_sub_scores"
        else None
    )
    existing_trace.append(
        {
            "name": "rerank",
            "elapsed_ms": elapsed_ms,
            "detail": {
                "top_k": len(new_results),
                "weights": cleaned_weights,
                "previous_weights": dict(run.weights or {}),
            },
        }
    )
    if sub_scores_tail is not None:
        existing_trace.append(sub_scores_tail)

    run.weights = cleaned_weights
    run.results = search_engine.results_to_jsonable(new_results)
    run.pipeline_trace = existing_trace
    run.elapsed_ms = elapsed_ms
    await db.commit()
    await db.refresh(run)

    items = await _hydrate_results(db, new_results)
    return SearchResponse(
        run_id=run.id,
        weights=cleaned_weights,
        results=items,
        pipeline_trace=_stages_from_trace(run.pipeline_trace),
        elapsed_ms=run.elapsed_ms,
        corpus_size=len(image_ids),
        query_dims=_query_dims_from_trace(run.pipeline_trace),
    )
