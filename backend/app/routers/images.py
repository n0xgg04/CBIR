"""POST /api/v1/images — upload → preprocess → extract → persist.

This is the Phase 1 demo gate: a `curl -F file=@cat.jpg` round-trip that
inserts an `images` row, a `feature_sets` row with all six handcrafted
vectors, and returns the full payload to the client.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from fastapi import Path as PathParam
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import Settings, get_settings
from app.db import get_db
from app.models import FeatureSet, Image
from app.schemas import FeatureSetRead, ImageRead, ImageWithFeatures
from app.services import feature_cache
from app.services import features as feat
from app.services.preprocess import decode_bgr, preprocess
from app.services.storage import LocalStorage

router = APIRouter(prefix="/api/v1", tags=["images"])

# Hard cap on upload size — defence against memory blow-ups. Adjust upstream
# (PLAN.md §11 expects single-image uploads <2 MB).
MAX_UPLOAD_BYTES: int = 8 * 1024 * 1024

ImageRole = Literal["corpus", "query"]


def _build_storage(settings: Settings) -> LocalStorage:
    return LocalStorage(settings.storage_root)


@router.post(
    "/images",
    response_model=ImageWithFeatures,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest one image — preprocess, extract all six features, persist.",
)
async def upload_image(
    file: Annotated[UploadFile, File(description="Image file (jpg/png/webp).")],
    animal_type: Annotated[
        str,
        Form(min_length=1, max_length=64, description="Species label, e.g. 'cat'."),
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    role: Annotated[ImageRole, Form(description="'corpus' or 'query'.")] = "corpus",
) -> ImageWithFeatures:
    payload = await file.read()
    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"upload exceeds {MAX_UPLOAD_BYTES} bytes",
        )

    try:
        decoded = decode_bgr(payload)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"invalid image: {exc}") from exc
    height, width = decoded.shape[:2]

    storage = _build_storage(settings)
    stored = storage.save(payload, animal_type=animal_type, original_filename=file.filename)

    existing = (
        await db.execute(
            select(Image)
            .options(selectinload(Image.feature_set))
            .where(Image.sha256 == stored.sha256)
        )
    ).scalar_one_or_none()

    if existing is not None and existing.feature_set is not None:
        # Idempotent: same content, same animal — return what we already have.
        return ImageWithFeatures(
            image=ImageRead.model_validate(existing),
            features=FeatureSetRead.model_validate(existing.feature_set),
        )

    if existing is None:
        image_row = Image(
            sha256=stored.sha256,
            filename=file.filename or stored.relative_path.rsplit("/", 1)[-1],
            storage_path=stored.relative_path,
            animal_type=animal_type,
            width=width,
            height=height,
            size_bytes=stored.size_bytes,
            role=role,
        )
        db.add(image_row)
        await db.flush()  # populate image_row.id
    else:
        image_row = existing

    preprocessed = preprocess(decoded)
    vectors = feat.extract_all(preprocessed)
    feature_row = FeatureSet(
        image_id=image_row.id,
        vectors=feat.vectors_to_lists(vectors),
        dims=dict(feat.EXPECTED_DIMS),
        extractor_ver=feat.EXTRACTOR_VERSION,
    )
    db.add(feature_row)
    await db.commit()
    await db.refresh(image_row)
    await db.refresh(feature_row)

    # Invalidate the in-memory feature matrix so the next /search reload sees us.
    feature_cache.mark_dirty()

    return ImageWithFeatures(
        image=ImageRead.model_validate(image_row),
        features=FeatureSetRead.model_validate(feature_row),
    )


@router.get(
    "/images/{image_id}/raw",
    response_class=Response,
    responses={
        200: {"content": {"image/*": {}}},
        404: {"description": "Image not found"},
    },
)
async def get_raw_image(
    image_id: Annotated[int, PathParam(ge=1)],
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    img = await db.get(Image, image_id)
    if img is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"image {image_id} not found")
    storage = _build_storage(settings)
    absolute = storage.absolute(img.storage_path)
    if not absolute.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"file not found on disk: {img.storage_path}"
        )
    data = absolute.read_bytes()
    ext = absolute.suffix.lstrip(".").lower()
    media_type = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tif": "image/tiff",
        "tiff": "image/tiff",
    }.get(ext, "application/octet-stream")
    return Response(content=data, media_type=media_type)
