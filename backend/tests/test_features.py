"""Unit tests for the six feature extractors and the `extract_all` aggregator.

Each test pins the four invariants required by PLAN.md §11:
    1. shape (length matches the documented DIM)
    2. dtype is float32
    3. L2 norm == 1 (or 0 for trivial inputs)
    4. determinism — same input twice produces bit-identical output
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services import features as F
from app.services.features import color_moments, glcm, hog, hsv, hu, lbp


@pytest.mark.parametrize(
    "module, expected_dim",
    [
        (hsv, 768),
        (color_moments, 9),
        (lbp, 18),
        (glcm, 40),
        (hog, 8100),
        (hu, 7),
    ],
)
def test_extractor_shape_dtype_and_norm(
    preprocessed_fixture: np.ndarray, module, expected_dim: int
) -> None:
    vec = module.extract(preprocessed_fixture)
    assert vec.ndim == 1
    assert vec.shape == (expected_dim,)
    assert vec.dtype == np.float32
    norm = float(np.linalg.norm(vec))
    assert norm == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    "module",
    [hsv, color_moments, lbp, glcm, hog, hu],
)
def test_extractor_is_deterministic(preprocessed_fixture: np.ndarray, module) -> None:
    a = module.extract(preprocessed_fixture)
    b = module.extract(preprocessed_fixture.copy())
    np.testing.assert_array_equal(a, b)


def test_extract_all_returns_six_features(preprocessed_fixture: np.ndarray) -> None:
    feats = F.extract_all(preprocessed_fixture)
    assert set(feats.keys()) == {"hsv", "cm", "lbp", "glcm", "hog", "hu"}
    for name, vec in feats.items():
        assert vec.dtype == np.float32, name
        assert vec.shape == (F.EXPECTED_DIMS[name],), name


def test_vectors_to_lists_is_json_safe(preprocessed_fixture: np.ndarray) -> None:
    feats = F.extract_all(preprocessed_fixture)
    serialisable = F.vectors_to_lists(feats)
    import json

    payload = json.dumps(serialisable)
    assert isinstance(payload, str) and len(payload) > 0
    for name, vec in serialisable.items():
        assert isinstance(vec, list)
        assert len(vec) == F.EXPECTED_DIMS[name]


def test_extractors_reject_wrong_input() -> None:
    junk = np.zeros((128, 128), dtype=np.uint8)  # 2-D, not 3-channel
    for module in (hsv, color_moments, lbp, glcm, hu):
        with pytest.raises(ValueError):
            module.extract(junk)


def test_hog_requires_target_size(fixture_image_bgr: np.ndarray) -> None:
    with pytest.raises(ValueError):
        hog.extract(fixture_image_bgr)  # 256×256, not 128×128
