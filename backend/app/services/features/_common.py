"""Shared helpers for feature extractors."""

from __future__ import annotations

import numpy as np

EPS: float = 1e-12


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return `vec` divided by its L2 norm; passthrough for zero vectors."""
    arr = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(arr))
    if norm <= EPS:
        return arr
    return (arr / norm).astype(np.float32)


def require_bgr_uint8(img: np.ndarray) -> None:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected a 3-channel BGR image")
    if img.dtype != np.uint8:
        raise ValueError("expected uint8 image (preprocess output)")
