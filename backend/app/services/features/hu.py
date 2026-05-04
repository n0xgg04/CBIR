"""Hu moments — 7-D log-transformed L2-normalised vector."""

from __future__ import annotations

import cv2
import numpy as np

from ._common import l2_normalize, require_bgr_uint8

DIM: int = 7


def extract(img: np.ndarray) -> np.ndarray:
    require_bgr_uint8(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()  # shape (7,)
    # Log-transform compresses the dynamic range (Hu values span many decades).
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return l2_normalize(hu_log.astype(np.float32))
