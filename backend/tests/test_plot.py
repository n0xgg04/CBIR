"""Unit tests for the matplotlib plot renderers and on-disk caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services import plot


def _is_png(data: bytes) -> bool:
    """PNG magic header check."""
    return data.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("feature", plot.SUPPORTED_PLOTS)
def test_renderer_emits_png_and_caches_to_disk(
    feature: str, fixture_image_bgr: np.ndarray, tmp_path: Path
) -> None:
    image_id = 42
    target = plot.plot_path(tmp_path, image_id, feature)
    assert not target.exists()

    data = plot.render(image_id, feature, fixture_image_bgr, tmp_path)
    assert _is_png(data)
    assert target.exists()
    assert target.read_bytes() == data


def test_render_uses_cache_on_second_call(
    fixture_image_bgr: np.ndarray, tmp_path: Path
) -> None:
    feature = "lbp"
    image_id = 7
    first = plot.render(image_id, feature, fixture_image_bgr, tmp_path)
    target = plot.plot_path(tmp_path, image_id, feature)

    # Mutate cache file to detect whether the second call regenerates.
    target.write_bytes(b"\x89PNG\r\n\x1a\nfaked-cache")
    cached = plot.render(image_id, feature, fixture_image_bgr, tmp_path)
    assert cached == b"\x89PNG\r\n\x1a\nfaked-cache"
    assert cached != first


def test_render_unknown_feature_raises(
    fixture_image_bgr: np.ndarray, tmp_path: Path
) -> None:
    with pytest.raises(KeyError):
        plot.render(1, "bogus", fixture_image_bgr, tmp_path)


def test_plot_path_includes_extractor_version(tmp_path: Path) -> None:
    target = plot.plot_path(tmp_path, 99, "hsv")
    from app.services import features as feat

    assert feat.EXTRACTOR_VERSION in target.name
    assert target.suffix == ".png"
