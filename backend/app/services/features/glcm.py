"""GLCM Haralick features — 40-D L2-normalised vector.

5 properties × 2 distances × 4 angles, computed on a 32-level quantised
grayscale image.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from ._common import l2_normalize, require_bgr_uint8

LEVELS: int = 32
DISTANCES: tuple[int, ...] = (1, 3)
ANGLES: tuple[float, ...] = (
    0.0,
    np.pi / 4,
    np.pi / 2,
    3 * np.pi / 4,
)
PROPERTIES: tuple[str, ...] = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
)
DIM: int = len(PROPERTIES) * len(DISTANCES) * len(ANGLES)  # 5 × 2 × 4 = 40


def extract(img: np.ndarray) -> np.ndarray:
    require_bgr_uint8(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Reduce gray levels to LEVELS bins (256 / 8 = 32) for tractable GLCM.
    quantised = (gray // (256 // LEVELS)).astype(np.uint8)

    glcm = graycomatrix(
        quantised,
        distances=list(DISTANCES),
        angles=list(ANGLES),
        levels=LEVELS,
        symmetric=True,
        normed=True,
    )

    feats: list[float] = []
    for prop in PROPERTIES:
        values = graycoprops(glcm, prop)  # (len(distances), len(angles))
        # NaNs can appear when an image has no variance (constant patches);
        # fall back to zero so the L2-norm is well-defined.
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        feats.extend(values.flatten().tolist())
    return l2_normalize(np.asarray(feats, dtype=np.float32))
