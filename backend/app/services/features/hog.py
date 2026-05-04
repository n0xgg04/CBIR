"""HOG descriptor on 128×128 grayscale — 8100-D L2-normalised vector.

cv2.HOGDescriptor with winSize=(128,128), blockSize=(16,16),
blockStride=(8,8), cellSize=(8,8), nbins=9.

Per cv2: 15×15 block positions × 4 cells/block × 9 bins = 8100.
"""

from __future__ import annotations

import cv2
import numpy as np

from ._common import l2_normalize, require_bgr_uint8

WIN_SIZE: tuple[int, int] = (128, 128)
BLOCK_SIZE: tuple[int, int] = (16, 16)
BLOCK_STRIDE: tuple[int, int] = (8, 8)
CELL_SIZE: tuple[int, int] = (8, 8)
NBINS: int = 9
DIM: int = 8100


_descriptor: cv2.HOGDescriptor | None = None


def _hog() -> cv2.HOGDescriptor:
    global _descriptor
    if _descriptor is None:
        _descriptor = cv2.HOGDescriptor(
            _winSize=WIN_SIZE,
            _blockSize=BLOCK_SIZE,
            _blockStride=BLOCK_STRIDE,
            _cellSize=CELL_SIZE,
            _nbins=NBINS,
        )
    return _descriptor


def extract(img: np.ndarray) -> np.ndarray:
    require_bgr_uint8(img)
    if img.shape[:2] != WIN_SIZE:
        raise ValueError(
            f"HOG expects {WIN_SIZE} grayscale input; got {img.shape[:2]} "
            "(run preprocess() first)"
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = _hog().compute(gray)
    return l2_normalize(feature.astype(np.float32))
