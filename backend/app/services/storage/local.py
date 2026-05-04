"""Filesystem-backed adapter for storing the original uploaded image bytes.

Layout:
    {storage_root}/originals/{animal_type}/{sha256}.{ext}

Content addressing on the filename means re-uploading the same image is a
no-op — `save()` returns the existing path. The DB layer relies on this
for the `images.sha256` UNIQUE constraint to dedupe rows.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ORIGINALS_DIRNAME: Final[str] = "originals"
DEFAULT_EXTENSION: Final[str] = "bin"

# A loose allow-list — the FastAPI router will reject other content types
# upstream, but we still want defence-in-depth on filenames.
_ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"}
)
_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass(frozen=True)
class StoredObject:
    """Result of `LocalStorage.save()` — describes one persisted blob."""

    sha256: str
    relative_path: str  # e.g. "originals/cat/<sha>.jpg" — DB-friendly
    absolute_path: Path  # full filesystem path for downstream readers
    size_bytes: int
    deduped: bool  # True when the file already existed on disk


class LocalStorage:
    """Tiny filesystem adapter — no async I/O needed; writes are <100 KB."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        data: bytes,
        animal_type: str,
        original_filename: str | None = None,
    ) -> StoredObject:
        """Persist `data` and return its `StoredObject`.

        Same content + same `animal_type` ⇒ idempotent (existing file is
        reused, `deduped=True`).
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")
        if not data:
            raise ValueError("cannot store empty payload")
        species = _slug_species(animal_type)
        if not species:
            raise ValueError("animal_type must contain at least one alphanumeric char")

        sha = hashlib.sha256(data).hexdigest()
        ext = _safe_extension(original_filename)
        rel = Path(ORIGINALS_DIRNAME) / species / f"{sha}.{ext}"
        absolute = self._root / rel

        if absolute.exists() and absolute.stat().st_size == len(data):
            return StoredObject(
                sha256=sha,
                relative_path=rel.as_posix(),
                absolute_path=absolute,
                size_bytes=len(data),
                deduped=True,
            )

        absolute.parent.mkdir(parents=True, exist_ok=True)
        # `xb` would race two concurrent uploads; instead write to a temp
        # sibling then rename — atomic on POSIX, "good enough" on Windows.
        tmp = absolute.with_suffix(absolute.suffix + ".part")
        tmp.write_bytes(bytes(data))
        tmp.replace(absolute)

        return StoredObject(
            sha256=sha,
            relative_path=rel.as_posix(),
            absolute_path=absolute,
            size_bytes=len(data),
            deduped=False,
        )

    def absolute(self, relative_path: str) -> Path:
        """Resolve a stored relative path back to an absolute filesystem path.

        Raises if the path escapes the storage root (defence against
        path-traversal payloads coming from the DB).
        """
        candidate = (self._root / relative_path).resolve()
        if self._root not in candidate.parents and candidate != self._root:
            raise ValueError(f"path escapes storage root: {relative_path!r}")
        return candidate


def _slug_species(animal_type: str) -> str:
    """Lowercase, ASCII-only directory name. Keeps the filesystem tidy."""
    return _SLUG_RE.sub("_", animal_type.strip().lower()).strip("_")


def _safe_extension(filename: str | None) -> str:
    """Pull a known image extension off `filename`; fall back to `bin`."""
    if not filename:
        return DEFAULT_EXTENSION
    suffix = Path(filename).suffix.lstrip(".").lower()
    return suffix if suffix in _ALLOWED_EXTENSIONS else DEFAULT_EXTENSION
