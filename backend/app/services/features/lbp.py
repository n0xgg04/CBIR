"""Uniform LBP histogram — 18-D L2-normalised vector (radius=2, n_points=16)."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from ._common import l2_normalize, require_bgr_uint8

RADIUS: int = 2
N_POINTS: int = 16
DIM: int = N_POINTS + 2  # 18 — uniform LBP has n_points+2 patterns


def extract(img: np.ndarray) -> np.ndarray:
    require_bgr_uint8(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=DIM, range=(0, DIM))
    return l2_normalize(hist.astype(np.float32))
