"""POST /api/v1/compare — upload two images, extract features, return similarity.

The endpoint runs the full preprocess + extract pipeline on both sides,
computes per-feature cosine similarity (dot product because vectors are
L2-normalised), and returns the weighted fused score.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas import CompareResponse
from app.services import features as feat
from app.services.preprocess import decode_bgr, preprocess
from app.services.search_engine import normalise_weights

router = APIRouter(prefix="/api/v1", tags=["compare"])

MAX_IMAGE_BYTES: int = 8 * 1024 * 1024


@router.post(
    "/compare",
    response_model=CompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare two uploaded images by handcrafted features.",
)
async def compare_images(
    left: Annotated[UploadFile, File(description="First image (jpg/png/webp).")],
    right: Annotated[UploadFile, File(description="Second image (jpg/png/webp).")],
) -> CompareResponse:
    started = time.perf_counter()

    left_bytes = await left.read()
    right_bytes = await right.read()
    if not left_bytes or not right_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "both images are required")
    if len(left_bytes) > MAX_IMAGE_BYTES or len(right_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"each image must be <= {MAX_IMAGE_BYTES} bytes",
        )

    # Decode + preprocess both sides
    try:
        left_decoded = decode_bgr(left_bytes)
        right_decoded = decode_bgr(right_bytes)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"invalid image: {exc}"
        ) from exc

    left_pre = preprocess(left_decoded)
    right_pre = preprocess(right_decoded)

    # Extract features
    left_vectors = feat.extract_all(left_pre)
    right_vectors = feat.extract_all(right_pre)

    # Cosine similarity per feature (vectors are L2-normalised → dot = cosine)
    weights = normalise_weights({})
    per_feature: dict[str, float] = {}
    fused = 0.0
    for name in feat.EXPECTED_DIMS:
        lv = left_vectors[name]
        rv = right_vectors[name]
        sim = float(lv @ rv)
        per_feature[name] = sim
        fused += sim * float(weights.get(name, 0.0))

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return CompareResponse(
        fused_score=fused,
        per_feature=per_feature,
        left_dims={name: int(vec.size) for name, vec in left_vectors.items()},
        right_dims={name: int(vec.size) for name, vec in right_vectors.items()},
        elapsed_ms=elapsed_ms,
    )
