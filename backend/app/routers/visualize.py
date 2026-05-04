"""GET /api/v1/visualize — render the inspector panel images on demand."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi import Path as PathParam
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_db
from app.models import Image
from app.services import plot
from app.services.preprocess import read_bgr
from app.services.storage import LocalStorage

router = APIRouter(prefix="/api/v1/visualize", tags=["visualize"])


async def _load_image_or_404(db: AsyncSession, image_id: int) -> Image:
    img = await db.get(Image, image_id)
    if img is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"image {image_id} not found")
    return img


def _read_original(settings: Settings, image: Image) -> np.ndarray:
    storage = LocalStorage(settings.storage_root)
    absolute = storage.absolute(image.storage_path)
    return read_bgr(absolute)


def _png_response(data: bytes) -> Response:
    headers = {"Cache-Control": "public, max-age=300"}
    return Response(content=data, media_type="image/png", headers=headers)


@router.get(
    "/{image_id}/{feature}",
    summary="Render the inspector panel for a feature (or 'preprocess').",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}},
        404: {"description": "Image not found"},
        400: {"description": "Unknown feature"},
    },
)
async def get_plot(
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    image_id: Annotated[int, PathParam(ge=1)],
    feature: Annotated[str, PathParam(description="One of: preprocess, hsv, cm, lbp, glcm, hog, hu.")],
) -> Response:
    if feature not in plot.SUPPORTED_PLOTS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unsupported feature {feature!r}; choose one of {plot.SUPPORTED_PLOTS}",
        )
    image = await _load_image_or_404(db, image_id)

    cached_path = plot.plot_path(settings.storage_root, image_id, feature)
    if cached_path.exists():
        return _png_response(cached_path.read_bytes())

    img_bgr = _read_original(settings, image)
    data = plot.render(image_id, feature, img_bgr, settings.storage_root)
    return _png_response(data)
