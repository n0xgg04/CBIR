"""Aggregator: run all six feature extractors against a preprocessed image."""

from __future__ import annotations

from typing import Final

import numpy as np

from . import color_moments, glcm, hog, hsv, hu, lbp

# String key → extractor module. Order matches PLAN.md §5/§6.
EXTRACTORS: Final[dict[str, object]] = {
    "hsv": hsv,
    "cm": color_moments,
    "lbp": lbp,
    "glcm": glcm,
    "hog": hog,
    "hu": hu,
}

EXPECTED_DIMS: Final[dict[str, int]] = {
    "hsv": hsv.DIM,
    "cm": color_moments.DIM,
    "lbp": lbp.DIM,
    "glcm": glcm.DIM,
    "hog": hog.DIM,
    "hu": hu.DIM,
}

# Bumped whenever any extractor's parameters/normalisation change.
EXTRACTOR_VERSION: Final[str] = "v1.1"


def extract_all(img: np.ndarray) -> dict[str, np.ndarray]:
    """Return `{name: l2-normalised 1-D vector}` for every supported feature."""
    return {name: mod.extract(img) for name, mod in EXTRACTORS.items()}  # type: ignore[attr-defined]


def vectors_to_lists(features: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """JSON-serialisable view: float32 arrays → plain Python lists."""
    return {name: vec.astype(float).tolist() for name, vec in features.items()}
