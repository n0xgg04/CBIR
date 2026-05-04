"""Image preprocessing pipeline: resize → blur → CLAHE.

Identical to the BASE.md spec: target 128×128, mild Gaussian, CLAHE on the L
channel of LAB, returned as BGR uint8.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TARGET_SIZE: tuple[int, int] = (128, 128)
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)
GAUSSIAN_KERNEL: tuple[int, int] = (3, 3)


def decode_bgr(buf: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR uint8 array."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image bytes")
    return img


def read_bgr(path: str | Path) -> np.ndarray:
    """Read an image from disk into a BGR uint8 array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image at {path}")
    return img


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    src_h, src_w = img.shape[:2]
    dst_w, dst_h = size
    interpolation = cv2.INTER_AREA if (src_h * src_w) > (dst_h * dst_w) else cv2.INTER_LINEAR
    return cv2.resize(img, (dst_w, dst_h), interpolation=interpolation)


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_eq = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_eq, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def preprocess(img: np.ndarray, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Resize → light Gaussian blur → CLAHE-on-L. BGR uint8 in, BGR uint8 out."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("preprocess expects a 3-channel BGR image")
    if img.dtype != np.uint8:
        raise ValueError("preprocess expects uint8 input")

    resized = _resize(img, target_size)
    blurred = cv2.GaussianBlur(resized, GAUSSIAN_KERNEL, 0)
    return _apply_clahe(blurred)


def preprocess_bytes(buf: bytes, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Decode raw image bytes and run the preprocess pipeline in one shot."""
    return preprocess(decode_bgr(buf), target_size=target_size)


def preprocess_path(path: str | Path, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Read from disk and preprocess in one shot."""
    return preprocess(read_bgr(path), target_size=target_size)
