from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import compare as compare_router
from app.routers import evaluate as evaluate_router
from app.routers import images as images_router
from app.routers import search as search_router
from app.routers import visualize as visualize_router
from app.routers import ws_pipeline as ws_pipeline_router


def create_app() -> FastAPI:
    """Application factory. Keeps configuration injection testable."""
    settings = get_settings()

    app = FastAPI(
        title="Animal Face CBIR API",
        version=settings.app_version,
        description=(
            "Content-based retrieval over animal-face images "
            "using handcrafted CV features (HSV / CM / LBP / GLCM / HOG / Hu)."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/ping", tags=["meta"])
    async def ping() -> dict[str, object]:
        return {
            "pong": True,
            "service": settings.app_name,
            "version": settings.app_version,
        }

    app.include_router(images_router.router)
    app.include_router(search_router.router)
    app.include_router(visualize_router.router)
    app.include_router(ws_pipeline_router.router)
    app.include_router(evaluate_router.router)
    app.include_router(compare_router.router)

    return app


app = create_app()
