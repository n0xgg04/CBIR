"""Matplotlib renderers that turn an image (or its feature vector) into a PNG.

Phase 3 ships an "inspector" UI that shows what each preprocessing step and
each handcrafted extractor produces. To avoid recomputing the same picture
on every page load, each plot is cached on disk under
`<storage_root>/plots/<image_id>/<feature>_v<extractor_version>.png`.

All matplotlib calls go through the headless `Agg` backend so importing this
module from FastAPI never tries to attach to a display.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.services import features as feat  # noqa: E402
from app.services.features import (  # noqa: E402
    color_moments as cm_mod,
)
from app.services.features import (
    glcm as glcm_mod,
)
from app.services.features import (
    hog as hog_mod,
)
from app.services.features import (
    hsv as hsv_mod,
)
from app.services.features import (
    hu as hu_mod,
)
from app.services.features import (
    lbp as lbp_mod,
)
from app.services.preprocess import (  # noqa: E402
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    GAUSSIAN_KERNEL,
    TARGET_SIZE,
    _apply_clahe,
    _resize,
    preprocess,
)

PLOTS_DIRNAME: Final[str] = "plots"
DPI: Final[int] = 100
FIGSIZE_SQUARE: Final[tuple[float, float]] = (5.0, 5.0)
FIGSIZE_BAR: Final[tuple[float, float]] = (8.0, 3.0)
FIGSIZE_PANEL: Final[tuple[float, float]] = (8.0, 8.0)


def plot_path(storage_root: str | Path, image_id: int, feature: str) -> Path:
    """Return the on-disk path where this plot is (or will be) cached."""
    safe_feature = feature.replace("/", "_")
    return (
        Path(storage_root)
        / PLOTS_DIRNAME
        / str(image_id)
        / f"{safe_feature}_v{feat.EXTRACTOR_VERSION}.png"
    )


def _save_figure(fig: plt.Figure, target: Path) -> bytes:
    """Render `fig` to PNG bytes, atomically write to `target`, and return them."""
    target.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    data = buf.getvalue()
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    return data


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def render_preprocess_panel(img_bgr: np.ndarray, target: Path) -> bytes:
    """2×2 panel: original → resized → blurred → CLAHE."""
    resized = _resize(img_bgr, TARGET_SIZE)
    blurred = cv2.GaussianBlur(resized, GAUSSIAN_KERNEL, 0)
    clahed = _apply_clahe(blurred)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_PANEL)
    panels = [
        (axes[0][0], img_bgr, f"original ({img_bgr.shape[1]}×{img_bgr.shape[0]})"),
        (axes[0][1], resized, f"resized ({TARGET_SIZE[0]}×{TARGET_SIZE[1]})"),
        (axes[1][0], blurred, f"blurred (Gaussian {GAUSSIAN_KERNEL[0]}×{GAUSSIAN_KERNEL[1]})"),
        (axes[1][1], clahed, f"CLAHE (clip={CLAHE_CLIP_LIMIT}, grid={CLAHE_TILE_GRID})"),
    ]
    for ax, panel, title in panels:
        ax.imshow(_bgr_to_rgb(panel))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Preprocess pipeline", fontsize=12)
    fig.tight_layout()
    return _save_figure(fig, target)


def _bar(values: np.ndarray, title: str, target: Path, *, xlabel: str = "bin") -> bytes:
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    ax.bar(np.arange(values.size), values, width=1.0, color="#3b82f6")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("magnitude")
    ax.margins(x=0)
    fig.tight_layout()
    return _save_figure(fig, target)


def render_hsv(img_bgr: np.ndarray, target: Path) -> bytes:
    vec = hsv_mod.extract(preprocess(img_bgr))
    return _bar(
        vec,
        f"HSV 3D histogram ({hsv_mod.H_BINS}×{hsv_mod.S_BINS}×{hsv_mod.V_BINS} = {hsv_mod.DIM} bins, L2-normalised)",
        target,
        xlabel="flattened H·S·V bin",
    )


def render_color_moments(img_bgr: np.ndarray, target: Path) -> bytes:
    vec = cm_mod.extract(preprocess(img_bgr))
    labels = [
        "H mean", "H std", "H skew",
        "S mean", "S std", "S skew",
        "V mean", "V std", "V skew",
    ]
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    ax.bar(labels, vec, color="#8b5cf6")
    ax.set_title("Color moments — mean / std / skew per HSV channel (L2-normalised)", fontsize=11)
    ax.set_ylabel("magnitude")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save_figure(fig, target)


def render_lbp(img_bgr: np.ndarray, target: Path) -> bytes:
    vec = lbp_mod.extract(preprocess(img_bgr))
    return _bar(
        vec,
        f"Uniform LBP histogram ({lbp_mod.DIM} bins, radius={lbp_mod.RADIUS}, points={lbp_mod.N_POINTS})",
        target,
        xlabel="LBP pattern bin",
    )


def render_glcm(img_bgr: np.ndarray, target: Path) -> bytes:
    vec = glcm_mod.extract(preprocess(img_bgr))
    n_props = len(glcm_mod.PROPERTIES)
    n_dist = len(glcm_mod.DISTANCES)
    n_ang = len(glcm_mod.ANGLES)
    grid = vec.reshape(n_props, n_dist * n_ang)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_yticks(range(n_props))
    ax.set_yticklabels(glcm_mod.PROPERTIES)
    angle_labels = [f"d={d}, θ={int(np.degrees(a))}°"
                    for d in glcm_mod.DISTANCES for a in glcm_mod.ANGLES]
    ax.set_xticks(range(n_dist * n_ang))
    ax.set_xticklabels(angle_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"GLCM Haralick — {glcm_mod.DIM}-D (L2-normalised)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save_figure(fig, target)


def render_hog(img_bgr: np.ndarray, target: Path) -> bytes:
    vec = hog_mod.extract(preprocess(img_bgr))
    # 8100-D vector is too dense for a bar chart; sample 256 bins for a sense of structure.
    stride = max(1, vec.size // 256)
    sampled = vec[::stride][:256]
    return _bar(
        sampled,
        f"HOG descriptor ({hog_mod.DIM}-D, showing every {stride}th bin)",
        target,
        xlabel="HOG bin (subsampled)",
    )


def render_hu(img_bgr: np.ndarray, target: Path) -> bytes:
    vec = hu_mod.extract(preprocess(img_bgr))
    return _bar(
        vec,
        "Hu moments — log-transformed, L2-normalised (7 invariants)",
        target,
        xlabel="Hu moment index",
    )


# Plot key → (renderer, doc string for the OpenAPI summary)
_RENDERERS: Final[dict[str, callable]] = {  # type: ignore[type-arg]
    "preprocess": render_preprocess_panel,
    "hsv": render_hsv,
    "cm": render_color_moments,
    "lbp": render_lbp,
    "glcm": render_glcm,
    "hog": render_hog,
    "hu": render_hu,
}

SUPPORTED_PLOTS: Final[tuple[str, ...]] = tuple(_RENDERERS.keys())


def render(image_id: int, feature: str, img_bgr: np.ndarray, storage_root: str | Path) -> bytes:
    """Render `feature` for `image_id`, caching the PNG on disk."""
    if feature not in _RENDERERS:
        raise KeyError(f"unknown plot: {feature!r}")
    target = plot_path(storage_root, image_id, feature)
    if target.exists():
        return target.read_bytes()
    return _RENDERERS[feature](img_bgr, target)
