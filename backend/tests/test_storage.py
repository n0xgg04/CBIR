"""Unit tests for the local filesystem storage adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.services.storage.local import (
    DEFAULT_EXTENSION,
    ORIGINALS_DIRNAME,
    LocalStorage,
)


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path)


def test_save_writes_under_originals(storage: LocalStorage, fixture_image_bytes: bytes) -> None:
    obj = storage.save(fixture_image_bytes, "cat", "kitty.jpg")
    assert obj.absolute_path.exists()
    assert obj.absolute_path.read_bytes() == fixture_image_bytes
    assert obj.relative_path.startswith(f"{ORIGINALS_DIRNAME}/cat/")
    assert obj.relative_path.endswith(".jpg")
    assert obj.size_bytes == len(fixture_image_bytes)
    assert obj.deduped is False


def test_save_returns_sha256_of_content(storage: LocalStorage) -> None:
    payload = b"\x89PNG fake-bytes for hashing"
    obj = storage.save(payload, "dog", "x.png")
    assert obj.sha256 == hashlib.sha256(payload).hexdigest()
    assert obj.sha256 in obj.relative_path


def test_save_dedupes_identical_payload(
    storage: LocalStorage, fixture_image_bytes: bytes
) -> None:
    first = storage.save(fixture_image_bytes, "cat", "a.jpg")
    second = storage.save(fixture_image_bytes, "cat", "b.jpg")  # different name
    assert first.relative_path == second.relative_path
    assert first.sha256 == second.sha256
    assert second.deduped is True
    # On disk, only one file should exist for this hash
    siblings = list(first.absolute_path.parent.glob(f"{first.sha256}.*"))
    assert len(siblings) == 1


def test_save_normalises_species_directory(storage: LocalStorage) -> None:
    obj = storage.save(b"abc", "Husky Dog", "x.png")
    assert "/husky_dog/" in obj.relative_path


def test_save_falls_back_to_default_extension_for_unknown_suffix(
    storage: LocalStorage,
) -> None:
    obj = storage.save(b"abc", "cat", "evil.exe")
    assert obj.relative_path.endswith(f".{DEFAULT_EXTENSION}")


def test_save_falls_back_to_default_extension_when_filename_missing(
    storage: LocalStorage,
) -> None:
    obj = storage.save(b"abc", "cat", None)
    assert obj.relative_path.endswith(f".{DEFAULT_EXTENSION}")


def test_save_rejects_empty_payload(storage: LocalStorage) -> None:
    with pytest.raises(ValueError):
        storage.save(b"", "cat", "x.jpg")


def test_save_rejects_blank_animal_type(storage: LocalStorage) -> None:
    with pytest.raises(ValueError):
        storage.save(b"abc", "  ", "x.jpg")


def test_save_rejects_non_bytes_payload(storage: LocalStorage) -> None:
    with pytest.raises(TypeError):
        storage.save("not bytes", "cat", "x.jpg")  # type: ignore[arg-type]


def test_absolute_resolves_relative_path(storage: LocalStorage) -> None:
    obj = storage.save(b"abc", "cat", "x.png")
    resolved = storage.absolute(obj.relative_path)
    assert resolved == obj.absolute_path


def test_absolute_rejects_path_traversal(storage: LocalStorage) -> None:
    with pytest.raises(ValueError):
        storage.absolute("../../etc/passwd")
