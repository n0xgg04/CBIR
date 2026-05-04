"""Color moments on HSV channels — 9-D L2-normalised vector."""

from __future__ import annotations

import cv2
import numpy as np

from ._common import l2_normalize, require_bgr_uint8

DIM: int = 9


def extract(img: np.ndarray) -> np.ndarray:
    """Mean / std / skew on each of H, S, V → 9-D L2-normalised vector."""
    require_bgr_uint8(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    moments: list[float] = []
    for ch in range(3):
        channel = hsv[..., ch].astype(np.float64)
        mean = float(channel.mean())
        std = float(channel.std())
        # cube-root preserves sign of the third central moment
        skew = float(np.cbrt(np.mean((channel - mean) ** 3)))
        moments.extend([mean, std, skew])
    return l2_normalize(np.asarray(moments, dtype=np.float32))
