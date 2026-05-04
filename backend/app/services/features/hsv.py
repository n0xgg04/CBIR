"""HSV 3D histogram (12×8×8 = 768 bins, L2-normalised)."""

from __future__ import annotations

import cv2
import numpy as np

from ._common import l2_normalize, require_bgr_uint8

H_BINS: int = 12
S_BINS: int = 8
V_BINS: int = 8
DIM: int = H_BINS * S_BINS * V_BINS  # 768


def extract(img: np.ndarray) -> np.ndarray:
    """Return a 768-D L2-normalised HSV histogram."""
    require_bgr_uint8(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [H_BINS, S_BINS, V_BINS],
        [0, 180, 0, 256, 0, 256],
    )
    return l2_normalize(hist.astype(np.float32))
