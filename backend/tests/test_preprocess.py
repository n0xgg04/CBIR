"""Unit tests for the preprocess pipeline."""

from __future__ import annotations

import numpy as np

from app.services import preprocess as P


def test_preprocess_returns_target_shape(fixture_image_bgr: np.ndarray) -> None:
    out = P.preprocess(fixture_image_bgr)
    assert out.shape == (128, 128, 3)
    assert out.dtype == np.uint8


def test_preprocess_is_deterministic(fixture_image_bgr: np.ndarray) -> None:
    a = P.preprocess(fixture_image_bgr)
    b = P.preprocess(fixture_image_bgr.copy())
    assert np.array_equal(a, b)


def test_preprocess_path_roundtrip(fixture_image_path) -> None:
    out = P.preprocess_path(fixture_image_path)
    assert out.shape == (128, 128, 3)
    assert out.dtype == np.uint8


def test_preprocess_bytes_roundtrip(fixture_image_bytes: bytes) -> None:
    out = P.preprocess_bytes(fixture_image_bytes)
    assert out.shape == (128, 128, 3)
    assert out.dtype == np.uint8


def test_preprocess_rejects_wrong_shape() -> None:
    flat = np.zeros((128, 128), dtype=np.uint8)
    try:
        P.preprocess(flat)
    except ValueError as exc:
        assert "BGR" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-BGR input")


def test_preprocess_rejects_wrong_dtype() -> None:
    img = np.zeros((10, 10, 3), dtype=np.float32)
    try:
        P.preprocess(img)
    except ValueError as exc:
        assert "uint8" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-uint8 input")
